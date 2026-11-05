// ===================================================================================
// trex_tikhonov.cpp
// ===================================================================================
/**
 * @file trex_tikhonov.cpp
 *
 * @brief Implementation of TRexTikhonovSelector (general-Tikhonov T-Rex).
 */
// ===================================================================================

// Class header
#include <trex_selector_methods/ad/trex_tikhonov/trex_tikhonov.hpp>

// std includes
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// lambda_2 CV tuners (sparse-K overloads)
#include <ml_methods/model_selection/ienet_cv_ccd.hpp>
#include <ml_methods/model_selection/tikhonov_cv_svd.hpp>

// Shared cluster-aware dummy machinery (GVS parity)
#include <trex_selector_methods/ad/trex_utils/trex_cluster_dummies.hpp>
#include <trex_selector_methods/ad/trex_utils/trex_cluster_hac.hpp>

// Seed mixing (CV fold seed derivation, mirrors TRexGVSSelector)
#include <utils/datageneration/utils_dummygen.hpp>
#include <utils/logging/logger.hpp>

// ===================================================================================

namespace trex::trex_selector_methods::ad::trex_tikhonov {

// Local namespace aliases
namespace cdum = trex::trex_selector_methods::ad::utils::cluster_dummies;
namespace dn   = trex::trex_selector_methods::ad::utils::data_normalizer;
namespace ms   = trex::ml_methods::model_selection;


// ===================================================================================
// Constructor
// ===================================================================================

TRexTikhonovSelector::TRexTikhonovSelector(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::VectorXd>& y,
    double                        tFDR,
    TRexTikhonovControlParameter  trex_tik_ctrl,
    int                           seed,
    bool                          verbose
) :
    tc::TRexSelector(X, y, tFDR, trex_tik_ctrl.trex_ctrl, seed, verbose),
    trex_tik_ctrl_(std::move(trex_tik_ctrl))
{
    // ---- Penalty matrix / structure ---------------------------------------
    const auto p_idx = static_cast<Eigen::Index>(p_);
    if (trex_tik_ctrl_.penalty_structure ==
        TikPenaltyStructure::GROUP_CENTERING) {
        // Structured fast path: groups are REQUIRED (they define K_GC and
        // feed the O(1) solver hooks); K is built internally when absent
        // (the CV tuners consume it).
        if (trex_tik_ctrl_.prior_groups.empty()) {
            throw std::invalid_argument(
                "TRexTikhonovSelector: penalty_structure = GROUP_CENTERING "
                "requires prior_groups (they define K_GC and the O(1) "
                "solver hooks).");
        }
        cdum::validatePriorGroups(trex_tik_ctrl_.prior_groups, p_,
                                  "TRexTikhonovSelector");
        if (trex_tik_ctrl_.tikhonov_K.nonZeros() == 0) {
            trex_tik_ctrl_.tikhonov_K =
                buildGroupCenteringK(trex_tik_ctrl_.prior_groups);
        }
        groups_vec_i_.resize(p_idx);
        for (std::size_t j = 0; j < p_; ++j) {
            groups_vec_i_(static_cast<Eigen::Index>(j)) =
                static_cast<int>(trex_tik_ctrl_.prior_groups[j]);
        }
    }
    // An ALL-SINGLETON partition under GROUP_CENTERING is legitimate: every
    // block of K_GC = I - J/p_m vanishes, the penalty is identically zero and
    // the path degenerates to the lasso with per-variable cluster dummies —
    // the "near-white" regime a too-high HAC cut produces. K then stays a
    // p x p sparse matrix without nonzeros, which every consumer handles;
    // only the user-supplied SPARSE_GENERAL route requires a non-empty K.
    const bool zero_K_from_singleton_groups =
        trex_tik_ctrl_.penalty_structure == TikPenaltyStructure::GROUP_CENTERING
        && trex_tik_ctrl_.tikhonov_K.nonZeros() == 0;
    if (trex_tik_ctrl_.tikhonov_K.nonZeros() == 0 &&
        !zero_K_from_singleton_groups) {
        throw std::invalid_argument(
            "TRexTikhonovSelector: tikhonov_K is empty; supply a p x p PSD "
            "penalty matrix (e.g. via gammaToK()).");
    }
    if (trex_tik_ctrl_.tikhonov_K.rows() != p_idx ||
        trex_tik_ctrl_.tikhonov_K.cols() != p_idx) {
        throw std::invalid_argument(
            "TRexTikhonovSelector: tikhonov_K must be p x p "
            "(p = X.cols(); finiteness, symmetry, and PSD are enforced by "
            "the solver constructors).");
    }
    trex_tik_ctrl_.tikhonov_K.makeCompressed();

    // ---- Solver gate ------------------------------------------------------
    // Two solvers implement the general sparse-K mode: TCDTIKNET (CCD,
    // exact fixed-lambda1 minimizers; FOLDED or INDEPENDENT_RIDGE dummy
    // coupling) and TIKNET (native LARS homotopy, FOLDED-only). The shared
    // dispatch would reject anything K-aware for other enumerators anyway,
    // so fail fast with a precise message.
    if (trex_ctrl_.solver_type != disp::SolverTypeForTRex::TCDTIKNET &&
        trex_ctrl_.solver_type != disp::SolverTypeForTRex::TIKNET) {
        throw std::invalid_argument(
            "TRexTikhonovSelector: solver_type must be TCDTIKNET or TIKNET "
            "(the only solvers with a general Tikhonov-K mode).");
    }
    if (!trex_tik_ctrl_.fold_dummy_coupling &&
        trex_ctrl_.solver_type != disp::SolverTypeForTRex::TCDTIKNET) {
        throw std::invalid_argument(
            "TRexTikhonovSelector: fold_dummy_coupling = false "
            "(INDEPENDENT_RIDGE) exists only on the TCDTIKNET path; TIKNET "
            "implements FOLDED coupling exclusively.");
    }

    // ---- Cluster-aware dummy gates ----------------------------------------
    if (trex_tik_ctrl_.cluster_dummies) {
        // All stored, on-demand, and memory-mapped strategies are served by
        // the DummyGenerator cluster mode (deterministic layer streams; the
        // mmap writer regenerates layers bit-exactly). Only the PERMUTATION
        // variants are rejected: K row-permuted copies of ONE base cannot
        // carry the per-cluster MVN convention (same reasoning as
        // TRexGVSSelector).
        switch (trex_ctrl_.lloop_strategy) {
            case tc::LLoopStrategy::PERMUTATION:
            case tc::LLoopStrategy::PERMUTATION_SEEDED:
                throw std::invalid_argument(
                    "TRexTikhonovSelector: the PERMUTATION-family L-loop "
                    "strategies cannot carry cluster-MVN dummies; choose "
                    "STANDARD, HCONCAT, SKIPL, or SEEDED, or set "
                    "cluster_dummies = false.");
            default:
                break;
        }
        cdum::validatePriorGroups(trex_tik_ctrl_.prior_groups, p_,
                                  "TRexTikhonovSelector");
        if (trex_tik_ctrl_.prior_groups.empty() &&
            (trex_tik_ctrl_.corr_max < 0.0 || trex_tik_ctrl_.corr_max > 1.0)) {
            throw std::invalid_argument(
                "TRexTikhonovSelector: corr_max must be in [0, 1]. Got: " +
                std::to_string(trex_tik_ctrl_.corr_max));
        }
    }

    // ---- Pilot rule gates -------------------------------------------------
    if (pilotRequested()) {
        (void)validate_lambda2_grid(trex_tik_ctrl_.pilot_grid);
        if (trex_tik_ctrl_.pilot_K < 2) {
            throw std::invalid_argument(
                "TRexTikhonovSelector: pilot_K must be >= 2 (it is capped at "
                "trex_ctrl.K).");
        }
        if (trex_tik_ctrl_.pilot_L_multiplier < 1 ||
            trex_tik_ctrl_.pilot_L_multiplier > trex_ctrl_.max_dummy_multiplier) {
            throw std::invalid_argument(
                "TRexTikhonovSelector: pilot_L_multiplier must be in "
                "[1, trex_ctrl.max_dummy_multiplier].");
        }
        if (trex_tik_ctrl_.pilot_T < 1) {
            throw std::invalid_argument(
                "TRexTikhonovSelector: pilot_T must be >= 1.");
        }
        if (!(trex_tik_ctrl_.pilot_knee_tolerance >= 0.0)) {
            throw std::invalid_argument(
                "TRexTikhonovSelector: pilot_knee_tolerance must be >= 0.");
        }
        if (trex_tik_ctrl_.pilot_statistic == PilotStatistic::GROUP_MAJORITY) {
            if (trex_tik_ctrl_.prior_groups.size() != static_cast<std::size_t>(p_)) {
                throw std::invalid_argument(
                    "TRexTikhonovSelector: PilotStatistic::GROUP_MAJORITY reads "
                    "the pilot count per group and needs prior_groups (one id "
                    "per predictor).");
            }
            if (!(trex_tik_ctrl_.pilot_group_fraction > 0.0 &&
                  trex_tik_ctrl_.pilot_group_fraction <= 1.0)) {
                throw std::invalid_argument(
                    "TRexTikhonovSelector: pilot_group_fraction must be in (0, 1].");
            }
        }
        if (trex_tik_ctrl_.pilot_shared_dummies &&
            (trex_ctrl_.lloop_strategy == tc::LLoopStrategy::PERMUTATION ||
             trex_ctrl_.lloop_strategy == tc::LLoopStrategy::PERMUTATION_SEEDED)) {
            throw std::invalid_argument(
                "TRexTikhonovSelector: pilot_shared_dummies has no first "
                "block to share under the PERMUTATION family; use the "
                "independent pilot dummies (default).");
        }
    }

    // ---- CV tuner gates ---------------------------------------------------
    // Support cap of the IEN-geometry tuner: any finite value (<= 0 disables).
    if (!std::isfinite(trex_tik_ctrl_.cv_dfmax_ratio)) {
        throw std::invalid_argument(
            "TRexTikhonovSelector: cv_dfmax_ratio must be finite (<= 0 "
            "disables the support cap). Got: " +
            std::to_string(trex_tik_ctrl_.cv_dfmax_ratio));
    }

    // ---- CV fold seed (mirrors TRexGVSSelector) ---------------------------
    if (trex_tik_ctrl_.cv_seed >= 0) {
        resolved_cv_seed_ = static_cast<unsigned int>(trex_tik_ctrl_.cv_seed);
    } else if (seed_ >= 0) {
        resolved_cv_seed_ = trex::utils::datageneration::dummygen::mix_seed(
            static_cast<std::uint32_t>(seed_), 1u);
    } else {
        resolved_cv_seed_ = std::random_device{}();
    }
}


// ===================================================================================
// gammaToK — K = Gamma^T Gamma
// ===================================================================================

Eigen::SparseMatrix<double> TRexTikhonovSelector::gammaToK(
    const Eigen::SparseMatrix<double>& gamma)
{
    Eigen::SparseMatrix<double> K =
        (Eigen::SparseMatrix<double>(gamma.transpose()) * gamma).pruned();
    K.makeCompressed();
    return K;
}


// ===================================================================================
// buildGroupCenteringK — K_GC = blockdiag(I - J/p_m)
// ===================================================================================

Eigen::SparseMatrix<double> TRexTikhonovSelector::buildGroupCenteringK(
    const std::vector<Eigen::Index>& prior_groups)
{
    const auto p = static_cast<Eigen::Index>(prior_groups.size());

    // Per-group column lists (ids validated by the caller / constructor).
    Eigen::Index M = 0;
    for (const Eigen::Index g : prior_groups) { M = std::max(M, g + 1); }
    std::vector<std::vector<Eigen::Index>> cols(static_cast<std::size_t>(M));
    for (std::size_t j = 0; j < prior_groups.size(); ++j) {
        cols[static_cast<std::size_t>(prior_groups[j])]
            .push_back(static_cast<Eigen::Index>(j));
    }

    std::vector<Eigen::Triplet<double>> trip;
    for (const auto& members : cols) {
        if (members.size() < 2) { continue; }   // singleton: no penalty
        const double w = 1.0 / static_cast<double>(members.size());
        for (const Eigen::Index a : members) {
            for (const Eigen::Index b : members) {
                const double delta = (a == b) ? 1.0 : 0.0;
                trip.emplace_back(a, b, delta - w);
            }
        }
    }
    Eigen::SparseMatrix<double> K(p, p);
    K.setFromTriplets(trip.begin(), trip.end());
    K.makeCompressed();
    return K;
}


// ===================================================================================
// onSelectBegin — resolve lambda_2 before the L-loop
// ===================================================================================

void TRexTikhonovSelector::onSelectBegin() {

    // 1. Dummy cluster model (X_ is already centered + scaled by the base
    //    constructor, so both the HAC distance and Sigma_m are valid).
    if (trex_tik_ctrl_.cluster_dummies) {
        if (verbose_) {
            printProgress("Tikhonov: Setting up dummy cluster structure...");
        }
        setupClusterModel();
        if (verbose_) {
            printProgress("Tikhonov: dummy clusters M = " +
                          std::to_string(clusters_list_.size()));
        }
    }

    // 2. lambda_2 (solver scale). Timed here rather than inside
    //    computeLambda2(), so a CV tuner's (or the pilot's) cost is
    //    attributable on its own instead of hiding inside the first solver
    //    run. The pilot needs the cluster model of step 1 (its dummies are
    //    drawn by the same generator) and the runner, hence its place here.
    {
        const auto t0 = std::chrono::steady_clock::now();
        lambda2_pilot_ = Lambda2PilotResult{};
        lambda2_ = pilotRequested() ? runLambda2Pilot() : computeLambda2();
        lambda2_cv_diag_.seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
    }
    if (verbose_) {
        std::ostringstream oss;
        oss << "Tikhonov: lambda_2 (solver scale) = " << lambda2_;
        printProgress(oss.str());
    }
}


// ===================================================================================
// setupClusterModel — dummy clustering + per-cluster Cholesky factors
// ===================================================================================

void TRexTikhonovSelector::setupClusterModel() {

    if (!trex_tik_ctrl_.prior_groups.empty()) {
        clusters_list_ = cdum::clustersFromPriorGroups(
            trex_tik_ctrl_.prior_groups, p_);
    } else {
        const auto merges = cdum::runCorrelationHAC(
            *X_, trex_tik_ctrl_.hc_linkage, verbose_);
        clusters_list_ = cdum::clustersFromHAC(
            merges, p_, trex_tik_ctrl_.corr_max);
    }

    auto cholesky_lower_list = cdum::buildClusterCholeskys(
        *X_, clusters_list_, n_, "TRexTikhonovSelector");

    // Hand the model to the base DummyGenerator: from here on, EVERY
    // generation entry point — stored (STANDARD / HCONCAT / SKIPL),
    // on-demand (one D_k at a time, serial), and the memory-mapped writers
    // (mmap pre-allocated at maximum dimensions, filled per L-iteration) —
    // draws per-cluster colored MVN layers on the shared streams instead of
    // i.i.d. columns. No hook override is needed; the base
    // prepareDummiesForLStep / ExperimentRunner machinery runs unchanged.
    dummy_gen_.setClusterModel(clusters_list_,
                               std::move(cholesky_lower_list), p_);
}


// ===================================================================================
// buildRunnerConfig — base config + Tikhonov injection
// ===================================================================================

er::ExperimentRunnerConfig TRexTikhonovSelector::buildRunnerConfig(
    std::size_t num_dummies,
    std::size_t T_stop,
    bool use_warm_start,
    er::ExperimentStrategy strategy,
    std::size_t seed_factor,
    std::size_t existing_cols_on_disk) const
{
    er::ExperimentRunnerConfig cfg = tc::TRexSelector::buildRunnerConfig(
        num_dummies, T_stop, use_warm_start, strategy, seed_factor,
        existing_cols_on_disk);

    // The K matrix lives in trex_tik_ctrl_ and outlives every run() call;
    // lambda2_ was resolved in onSelectBegin().
    cfg.tikhonov_K            = &trex_tik_ctrl_.tikhonov_K;
    cfg.tikhonov_fold_dummies = trex_tik_ctrl_.fold_dummy_coupling;
    cfg.solver_params.lambda2 = lambda2_;

    // GROUP_CENTERING fast path: route the solver through the O(1) hooks
    // (groups_vec_i_ is owned by this selector and outlives the run).
    if (trex_tik_ctrl_.penalty_structure ==
        TikPenaltyStructure::GROUP_CENTERING) {
        cfg.tikhonov_group_centering = true;
        cfg.tikhonov_groups          = &groups_vec_i_;
    }

    return cfg;
}


// ===================================================================================
// lambda_2 — user supplied or auto-computed via the sparse-K tuners
// ===================================================================================

namespace {

/** @brief The boundary-pick guard (warning-only): one warning
 *  when the returned lambda_2 sits on its grid boundary or, for the IEN
 *  tuner, when its profiled lambda_1 optimum sits on the truncation edge.
 *  The pick is returned unchanged; getLambda2CVDiagnostics() carries the
 *  flags. Not gated on verbose: a boundary pick is a result-quality
 *  condition the caller must see. */
void warnIfBoundaryPick(const ms::lambda2_cv_diagnostics& d,
                        const char* method, double lambda2) {
    const std::string msg =
        ms::boundary_pick_message(d, "TRexTikhonovSelector", method, lambda2);
    if (!msg.empty()) { TREX_WARN(msg); }
}

} // namespace


double TRexTikhonovSelector::computeLambda2() const {

    lambda2_cv_diag_ = ms::lambda2_cv_diagnostics{};

    if (trex_tik_ctrl_.lambda_2 >= 0.0) {
        // Fixed user value (>= 0); lambda_2 == 0 -> pure lasso path.
        // Interpreted in the WORKING column scale (trex_ctrl_.scaling_mode).
        return trex_tik_ctrl_.lambda_2;
    }

    // CV is entered with X centered + scaled per trex_ctrl_.scaling_mode and
    // y centered (base-class constructor). The tuners re-normalize per fold
    // to unit L2, so the selected lambda_2 is on the unit-L2 solver scale;
    // the ZSCORE working scale (squared column norm n - 1 instead of 1)
    // needs the (n - 1) factor — same conversion as the IEN-geometry
    // methods in TRexGVSSelector::computeLambda2(). L2 and NONE (user-managed
    // design, unit-L2-commensurate by contract) need no factor.
    const double scale_adjust =
        (trex_ctrl_.scaling_mode == dn::ScalingMode::ZSCORE && n_ > 1)
            ? static_cast<double>(n_ - 1)
            : 1.0;

    switch (trex_tik_ctrl_.lambda2_method) {
        case TikLambda2Method::CV_1SE_IEN_CCD:
        case TikLambda2Method::CV_MIN_IEN_CCD: {
            // 2D (lambda_2 x lambda_1) CV with lambda_1 profiled out;
            // well-posed in any p/n regime. The lambda_2 grid size is
            // ctrl.cv_n_lambda2; the nuisance lambda_1 path length is
            // ctrl.cv_n_lambda1. Warm-started across lambda_2 with the
            // glmnet fdev/devmax/dfmax truncation of a full-data reference
            // scan; the dfmax support cap (ctrl.cv_dfmax_ratio) is what makes
            // it affordable. max_iter / tol are the tuner's defaults.
            ms::ienet_cv_ccd cv;
            cv.set_num_threads(trex_ctrl_.max_inner_threads);
            cv.fit(*X_, y_, trex_tik_ctrl_.tikhonov_K,
                   trex_tik_ctrl_.cv_n_folds,
                   trex_tik_ctrl_.cv_n_lambda2,
                   /*lambda2_ratio=*/1e4,
                   trex_tik_ctrl_.cv_n_lambda1,
                   /*lambda1_min_ratio=*/-1.0,
                   resolved_cv_seed_,
                   /*max_iter=*/100000,
                   /*tol=*/1e-4,
                   trex_tik_ctrl_.cv_dfmax_ratio);
            const bool one_se =
                (trex_tik_ctrl_.lambda2_method ==
                 TikLambda2Method::CV_1SE_IEN_CCD);
            lambda2_cv_diag_ = cv.diagnostics(one_se);
            const double l2 = one_se ? cv.cv_1se() : cv.cv_min();
            warnIfBoundaryPick(lambda2_cv_diag_,
                               one_se ? "CV_1SE_IEN_CCD" : "CV_MIN_IEN_CCD",
                               l2 * scale_adjust);
            return l2 * scale_adjust;
        }
        case TikLambda2Method::CV_1SE_TIK_SVD:
        case TikLambda2Method::CV_MIN_TIK_SVD: {
            // Analytic lambda_1 = 0 backbone; refused when the unpenalized
            // null-space block interpolates the training folds (any pick
            // there is floor-driven).
            ms::tikhonov_cv_svd cv;
            cv.set_num_threads(trex_ctrl_.max_inner_threads);
            cv.fit(*X_, y_, trex_tik_ctrl_.tikhonov_K,
                   trex_tik_ctrl_.cv_n_folds,
                   trex_tik_ctrl_.cv_n_lambda,
                   /*lambda_ratio=*/1e4,
                   resolved_cv_seed_);
            if (cv.contrast_interpolates()) {
                throw std::invalid_argument(
                    "TRexTikhonovSelector::computeLambda2: CV_*_TIK_SVD is "
                    "structurally uninformative on this design "
                    "(null_dim >= n_train: the unpenalized null-space block "
                    "interpolates the training folds). Use CV_*_IEN_CCD "
                    "instead.");
            }
            lambda2_cv_diag_.ran = true;
            const bool one_se =
                (trex_tik_ctrl_.lambda2_method ==
                 TikLambda2Method::CV_1SE_TIK_SVD);
            ms::record_grid_pick(lambda2_cv_diag_,
                                 one_se ? cv.index_1se() : cv.index_min(),
                                 cv.lambdas().size());
            const double l2 = one_se ? cv.cv_1se() : cv.cv_min();
            warnIfBoundaryPick(lambda2_cv_diag_,
                               one_se ? "CV_1SE_TIK_SVD" : "CV_MIN_TIK_SVD",
                               l2 * scale_adjust);
            return l2 * scale_adjust;
        }
        case TikLambda2Method::PILOT_KNEE:
            throw std::logic_error(
                "TRexTikhonovSelector::computeLambda2: the pilot rule is "
                "resolved by runLambda2Pilot(), not here.");
    }
    throw std::invalid_argument(
        "TRexTikhonovSelector::computeLambda2: unknown TikLambda2Method.");
}


// ===================================================================================
// lambda_2 pilot — the power rule read off a few random experiments
// ===================================================================================

bool TRexTikhonovSelector::pilotRequested() const noexcept {
    return trex_tik_ctrl_.lambda_2 < 0.0 &&
           trex_tik_ctrl_.lambda2_method == TikLambda2Method::PILOT_KNEE;
}


std::size_t TRexTikhonovSelector::pilotDummyTag() const {
    // The tag under which the main strategy draws its FIRST block of p
    // dummies per experiment (trex_dummy_generator.hpp, tag conventions):
    // prefix-stable strategies live under tag 0 (in cluster mode layer li
    // draws from stream li + 1, so one layer is stream 1 either way);
    // STANDARD redraws under tag L_iter; SKIPL under the maximum multiplier;
    // SEEDED in cluster mode derives the tag from the requested width.
    switch (trex_ctrl_.lloop_strategy) {
        case tc::LLoopStrategy::HCONCAT:
            return 0;
        case tc::LLoopStrategy::STANDARD:
            return trex_tik_ctrl_.pilot_L_multiplier;
        case tc::LLoopStrategy::SKIPL:
            return trex_ctrl_.max_dummy_multiplier;
        case tc::LLoopStrategy::SEEDED:
            return trex_tik_ctrl_.cluster_dummies
                       ? trex_tik_ctrl_.pilot_L_multiplier : 0;
        case tc::LLoopStrategy::PERMUTATION:
        case tc::LLoopStrategy::PERMUTATION_SEEDED:
            break;
    }
    throw std::logic_error(
        "TRexTikhonovSelector::pilotDummyTag: the PERMUTATION family has no "
        "shared pilot block (refused at construction).");
}


double TRexTikhonovSelector::runLambda2Pilot() {

    const TRexTikhonovControlParameter& c = trex_tik_ctrl_;
    const std::size_t K_pilot = std::min(c.pilot_K, trex_ctrl_.K);
    const std::size_t L_0     = c.pilot_L_multiplier * p_;
    const std::size_t half_n  = (n_ + 1) / 2;   // ceil(n / 2), the T-loop's cap
    const std::size_t T_pilot =
        std::max<std::size_t>(1, std::min({c.pilot_T, L_0, half_n}));
    const std::vector<double> grid   = validate_lambda2_grid(c.pilot_grid);
    const Eigen::VectorXd     v_grid = createVotingGrid(K_pilot);
    const std::size_t tag = c.pilot_shared_dummies ? pilotDummyTag()
                                                   : kPilotDummyTag;
    const std::size_t G = grid.size();
    const auto t_all = std::chrono::steady_clock::now();

    Lambda2PilotResult res;
    res.ran       = true;
    res.grid      = grid;
    res.K_pilot   = K_pilot;
    res.T_pilot   = T_pilot;
    res.L_0       = L_0;
    res.dummy_tag = tag;
    res.counts.assign(G, 0);
    res.calibrated.assign(G, 0);
    res.group_majority.assign(G, 0);
    res.groups_qualifying.assign(G, 0);
    res.curve_majority.assign(G, std::vector<std::size_t>(T_pilot, 0));
    res.curve_soft.assign(G, std::vector<double>(T_pilot, 0.0));
    res.probe_seconds.assign(G, 0.0);

    // One dummy set for every candidate (common random numbers): the probes
    // differ by lambda_2 alone. Retained warm-start solvers view the store,
    // so they are dropped first (the pilot runs cold anyway).
    warm_start_mgr_.invalidate();
    if (!trex_ctrl_.use_memory_mapping) {
        dummy_gen_.generateAndStore(K_pilot, L_0, tag);
    }

    for (std::size_t g = 0; g < G; ++g) {
        const auto t0 = std::chrono::steady_clock::now();
        lambda2_ = grid[g];   // buildRunnerConfig() injects it into the solver
        er::ExperimentRunnerConfig cfg = buildRunnerConfig(
            L_0, T_pilot, /*use_warm_start=*/false,
            er::ExperimentStrategy::Standard, /*seed_factor=*/tag,
            /*existing_cols_on_disk=*/0);
        cfg.K = K_pilot;   // the runner aggregates by 1 / cfg.K
        const er::ExperimentResults exp = experiment_runner_->run(cfg);

        for (std::size_t t = 0; t < T_pilot; ++t) {
            const auto col = exp.phi_T_mat.col(static_cast<Eigen::Index>(t));
            res.curve_majority[g][t] =
                static_cast<std::size_t>((col.array() > 0.5).count());
            res.curve_soft[g][t] = col.sum();
        }
        res.calibrated[g] = static_cast<std::size_t>(
            selectFromPhiT(exp.phi_T_mat, L_0, v_grid,
                           /*drop_last_row=*/false).selected_var.sum());
        if (c.prior_groups.size() == static_cast<std::size_t>(p_)) {
            const Eigen::VectorXd phi_T =
                exp.phi_T_mat.col(static_cast<Eigen::Index>(T_pilot - 1));
            const GroupMajorityCount gm = group_majority_count(
                phi_T, c.prior_groups, c.pilot_group_fraction);
            res.group_majority[g]    = gm.members;
            res.groups_qualifying[g] = gm.groups;
        }
        switch (c.pilot_statistic) {
            case PilotStatistic::MAJORITY:
                res.counts[g] = res.curve_majority[g][T_pilot - 1];
                break;
            case PilotStatistic::CALIBRATED:
                res.counts[g] = res.calibrated[g];
                break;
            case PilotStatistic::GROUP_MAJORITY:
                res.counts[g] = res.group_majority[g];
                break;
        }
        res.probe_seconds[g] = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
    }

    res.pick         = pick_lambda2_index(res.counts, c.pilot_knee_tolerance);
    res.index        = res.pick.index;
    res.lambda2      = grid[res.index];
    res.at_lower_end = (res.index == 0);
    res.at_upper_end = (res.index + 1 == G);

    // Leave nothing of the pilot behind: the main L-loop regenerates or
    // rewrites its dummies from scratch, and a SEEDED main run never touches
    // the store, so the store is emptied explicitly.
    warm_start_mgr_.invalidate();
    if (!trex_ctrl_.use_memory_mapping) {
        dummy_gen_.clearStored();
    }
    lambda2_cv_diag_     = ms::lambda2_cv_diagnostics{};
    lambda2_cv_diag_.ran = true;
    res.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_all).count();

    if (verbose_) {
        std::ostringstream oss;
        oss << "Tikhonov: lambda_2 pilot (K " << K_pilot << ", T " << T_pilot
            << ", L_0 " << L_0 << ") counts:";
        for (std::size_t g = 0; g < G; ++g) {
            oss << " " << grid[g] << ":" << res.counts[g];
        }
        oss << " -> pick " << res.lambda2;
        printProgress(oss.str());
    }

    lambda2_pilot_ = std::move(res);
    return lambda2_pilot_.lambda2;
}


const Lambda2PilotResult& TRexTikhonovSelector::runPilotOnly() {
    if (!pilotRequested()) {
        throw std::invalid_argument(
            "TRexTikhonovSelector::runPilotOnly: requires lambda_2 < 0 and a "
            "PILOT_* lambda2_method.");
    }
    if (trex_tik_ctrl_.cluster_dummies) {
        setupClusterModel();
    }
    const auto t0 = std::chrono::steady_clock::now();
    lambda2_pilot_ = Lambda2PilotResult{};
    lambda2_ = runLambda2Pilot();
    lambda2_cv_diag_.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    denormalizeAndReleaseX();
    return lambda2_pilot_;
}

// ===================================================================================
} /* End of namespace trex::trex_selector_methods::ad::trex_tikhonov */
// ===================================================================================
