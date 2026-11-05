// ============================================================================
// tcdlasso_solver.cpp
// ============================================================================
/**
 * @file tcdlasso_solver.cpp
 *
 * @brief Implementation (.cpp) file for the Terminating Cyclic Coordinate
 * Descent (T-CDLASSO) solver.
 */
// ============================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

// tsolvers includes
#include <tsolvers/ad/linear_model/cd_based/tcdlasso_solver.hpp>
#include <tsolvers/ad/solver_utils/memory_bytes.hpp>
#include <tsolvers/ad/solver_utils/solver_preprocessing.hpp>

// utils includes
#include <utils/serialization/utils_cereal_eigen.hpp>
#include <utils/openmp/utils_openmp.hpp>
#include <utils/fp_classify/fp_classify.hpp>

// ============================================================================

// Embedded into trex::tsolvers::ad::linear_model::cd_based namespace
namespace trex::tsolvers::ad::linear_model::cd_based {

// ============================================================================

namespace fpc = trex::utils::fp_classify;

namespace {
inline double softThreshold(double z, double lambda1) {
    if (z > lambda1)  { return z - lambda1; }
    if (z < -lambda1) { return z + lambda1; }
    return 0.0;
}
} // anonymous namespace


// ============================================================================
// Constructors
// ============================================================================

TCDLASSO_Solver::TCDLASSO_Solver(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D,
    Eigen::Map<Eigen::VectorXd>& y,
    const SolverOptions& options
    )
    : TCDLASSO_Solver(X, D, y, options, SolverTypeCdBased::TCDLASSO) {}


TCDLASSO_Solver::TCDLASSO_Solver(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D,
    Eigen::Map<Eigen::VectorXd>& y,
    const SolverOptions& options,
    SolverTypeCdBased algorithm_type
    )
    : TSolver_Base(X, D, y, options),
      algo_type_(algorithm_type) {

    // 0. CD knobs (construction-time options; the setters keep
    //    their validation and warn on an out-of-range value).
    setLambdaRelTol(options.cd_lambda_rel_tol);
    setCdTolerances(options.cd_tol_certify, options.cd_tol_probe);
    setGramCap(options.cd_gram_cap);
    setMaxSweepsPerSolve(options.cd_max_sweeps);
    setCertifiedLocalization(options.cd_certified_localization);

    // 1. Inherited preprocessing handles X, D, y centering and normalization
    dropped_indices_.clear();
    if (normalize_ || intercept_) { preprocess(); }
    initializeInactives();

    // 2. State Limit (recorded steps: one per dummy crossing / grid point)
    std::size_t p_total = p_original_ + num_dummies_;
    max_actives_ = std::min(p_total - dropped_indices_.size(), effective_n_);
    maxSteps_ = 8 * std::min(max_actives_, effective_n_);

    logInfo(concatMsg(
            solverTypeToString(), " sequence\n",
            "  Samples (n): ", X.rows(), "\n",
            "  Features (p): ", p_total, " (",
            p_original_, " real + ",
            num_dummies_, " dummies)\n",
            "  Max steps: ", maxSteps_, "\n"
    ));

    // 3. Initialize shared path state variables
    r_ = y_;
    initBetaPathStorage();

    double rss = y_.dot(y_);
    RSS_.emplace_back(rss);
    R2_.emplace_back(0.0);
    DoF_.emplace_back(interceptDoF());
    dummies_at_step_.reserve(maxSteps_ + 1);
    dummies_at_step_.push_back(0);

    // 4. Initialize CCD specific states
    lambda_.reserve(maxSteps_ + 1);
    initializeCorrelations();
    lambda_max_ = correlations_.cwiseAbs().maxCoeff();
    lambda_current_ = lambda_max_;
}


TCDLASSO_Solver::TCDLASSO_Solver(SolverTypeCdBased type) : TSolver_Base(), algo_type_(type) {}
TCDLASSO_Solver::TCDLASSO_Solver() : TSolver_Base(), algo_type_(SolverTypeCdBased::TCDLASSO) {}


// ============================================================================
// Core Algorithm Implementation
// ============================================================================

void TCDLASSO_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();

    // Set openMP dominance for the entire algorithm
    EigenThreadBudgetGuard eigen_thread_budget_guard(n_threads_);

    ensureCdState();

    const bool full_path = (!early_stop || T_stop == 0);
    const long point_budget = lambda_points_ +
                              static_cast<long>(max_lambda_points_);

    // Active-set cap: the lasso support cannot exceed effective_n_, but for
    // lambda2 > 0 the penalized minimizer's support can (the ridge part makes
    // the problem strictly convex in every direction) — same convention as
    // the LARS-EN side (TENET lifts the cap to all columns for lambda2 > 0).
    const std::size_t p_total_cap = p_original_ + num_dummies_;
    const std::size_t active_cap =
        (lambda2_ > 0.0 && penalty_kind_ != PenaltyKind::NONE)
            ? (p_total_cap - dropped_indices_.size())
            : effective_n_;

    // ========================================================
    // Full-path mode: geometric lambda1 grid (glmnet-style)
    // ========================================================
    if (full_path) {
        std::size_t grid_remaining = grid_points_;
        const double decay =
            std::pow(lambda_min_ratio_, 1.0 / static_cast<double>(grid_points_));
        while (currentStep_ < maxSteps_ &&
               actives_.size() < active_cap &&
               lambda_points_ < point_budget &&
               grid_remaining > 0 &&
               lambda_current_ > lambda_min_ratio_ * lambda_max_ * (1.0 + 1e-12)) {
            double lam_next = std::max(lambda_current_ * decay,
                                       lambda_min_ratio_ * lambda_max_);
            cd_sweeps_ += solveAtLambda(lam_next, cd_tol_, nullptr);
            lambda_points_++;
            lambda_current_ = lam_next;
            grid_remaining--;
            recordCrossingStep(lam_next);
        }
        return;
    }

    if (localization_ == LocalizationMode::CERTIFIED && !certified_degraded_) {
        executeStepCertified(T_stop, point_budget, active_cap);
    } else {
        executeStepLegacy(T_stop, point_budget, active_cap);
        if (certified_degraded_) { padUncertifiedRecords(); }
    }
}


void TCDLASSO_Solver::padUncertifiedRecords() {
    // Records issued by the LEGACY fallback carry no certificates: keep the
    // per-record diagnostics aligned with the step count using -1 sentinels.
    while (record_gap_.size() < currentStep_) { record_gap_.push_back(-1.0); }
    while (record_kkt_.size() < currentStep_) { record_kkt_.push_back(-1.0); }
}


void TCDLASSO_Solver::fallBackToLegacy(std::size_t T_stop, long point_budget,
                                       std::size_t active_cap) {
    // A0-w boundary (e.g. lasso saturation, |A| -> n: singular H_SS, CD
    // limit cycles): certificates are genuinely unavailable (Lemma 3/4).
    // Continue the path with the LEGACY localization instead of stopping —
    // the T-Rex layer needs the deep-tail records like the T-solvers produce
    // them — and flag every further record as uncertified (Def. 16).
    logWarning("certified localization: certification unavailable (A0-w "
               "boundary, e.g. active set near saturation); falling back to "
               "LEGACY localization for the remaining path — further records "
               "are uncertified (gap/kkt = -1).");
    certified_degraded_ = true;
    if (!frontier_top_.conventional && !frontier_top_.support.empty()) {
        loadIterate(frontier_top_);
    }
    // Refresh correlations_ for the legacy jump predictor.
    cd_sweeps_ += solveAtLambda(lambda_current_, cd_tol_probe_, nullptr);
    lambda_points_++;
    executeStepLegacy(T_stop, point_budget, active_cap);
    padUncertifiedRecords();
}


void TCDLASSO_Solver::executeStepLegacy(std::size_t T_stop, long point_budget,
                                        std::size_t active_cap) {
    // Main T-CDLASSO loop with early stopping based on dummy variable count
    while (
        currentStep_ < maxSteps_ &&
        actives_.size() < active_cap &&
        lambda_points_ < point_budget &&
        count_active_dummies_ < T_stop
    ) {
        // ========================================================
        // STEP 1: Predict the next dummy crossing and jump
        // ========================================================
        const std::size_t next_T = count_active_dummies_ + 1;

        double target = predictNextCrossing();
        double lam_next = std::min(target * jump_margin_,
                                   lambda_current_ * 0.999);
        if (!(lam_next > 0.0)) { lam_next = lambda_current_ * 0.5; }
        if (lam_next < lambda_current_ * 1e-4) {
            lam_next = lambda_current_ * 1e-4;  // jump floor (safety)
        }

        std::vector<std::size_t> support_hi(actives_);   // bracket top support
        cd_sweeps_ += solveAtLambda(lam_next, cd_tol_probe_, nullptr);
        lambda_points_++;
        std::size_t cd = countActiveDummiesCd();

        if (cd < next_T) {                       // undershoot: re-predict
            lambda_current_ = lam_next;
            count_active_dummies_ = cd;
            continue;
        }

        // ========================================================
        // STEP 2: Localize the crossing on [lam_next, lambda_current_]
        // ========================================================
        std::vector<std::size_t> candidates;
        collectCandidates(lam_next, candidates);

        double lo = lam_next;
        double hi = lambda_current_;
        std::vector<std::size_t> support_lo(actives_);
        std::size_t count_lo = cd;

        while (true) {
            // Support-nesting early stop: only when the bracket's support
            // changes are entering dummies AND the bracket bottom holds
            // EXACTLY the target count — then the crossing's support is
            // fully determined. With more dummies than the target at the
            // bottom, keep narrowing so every dummy crossing is recorded at
            // its own count (skipped counts would silently zero their
            // phi_T contributions in the T-Rex calibration); the later
            // crossings are re-localized by subsequent loop iterations.
            if (count_lo == next_T &&
                supportOnlyDummyGrowth(support_hi, support_lo)) { break; }
            if ((hi - lo) / hi <= lambda_rel_tol_) { break; }

            double mid = 0.5 * (lo + hi);
            cd_sweeps_ += solveAtLambda(mid, cd_tol_probe_, &candidates);
            lambda_points_++;
            const std::size_t cmid = countActiveDummiesCd();
            if (cmid >= next_T) {
                lo = mid;
                support_lo = actives_;
                count_lo = cmid;
            } else {
                hi = mid;
                support_hi = actives_;
            }
        }

        // ========================================================
        // STEP 3: Certify the crossing with a full KKT pass and record
        // ========================================================
        cd_sweeps_ += solveAtLambda(lo, cd_tol_, nullptr);
        lambda_points_++;
        lambda_current_ = lo;
        std::size_t cd_lo = countActiveDummiesCd();
        count_active_dummies_ = cd_lo;
        if (cd_lo < next_T) { continue; }        // certification moved the count

        recordCrossingStep(lo);
    }
}


// ============================================================================
// Helper Implementations
// ============================================================================

void TCDLASSO_Solver::ensureCdState() {
    if (cd_ready_) { return; }

    const std::size_t p_total = p_original_ + num_dummies_;
    const Eigen::Index pt = static_cast<Eigen::Index>(p_total);

    // Fresh start (constructor path) vs reconnect/load (state deserialized).
    if (beta_cd_.size() != pt) {
        beta_cd_ = Eigen::VectorXd::Zero(pt);
        l2kb_ = Eigen::VectorXd::Zero(pt);
        in_active_.assign(p_total, 0);
        r_ = y_;
    }

    // Dropped-column lookup.
    is_dropped_.assign(p_total, 0);
    for (std::size_t j : dropped_indices_) { is_dropped_[j] = 1; }

    // Squared column norms (1.0 for unit-L2 standardized columns).
    colnorm2_.setOnes(pt);
    for (std::size_t j = 0; j < p_total; ++j) {
        if (!is_dropped_[j]) { colnorm2_(static_cast<Eigen::Index>(j)) =
            getColumn(j).squaredNorm(); }
    }

    // Pristine X^T y (correlations_ tracks the path, so recompute).
    correlations_y_.setZero(pt);
    [[maybe_unused]] const int nt = threadsFor(static_cast<double>(X_->rows()) * static_cast<double>(p_total));
    #pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
    for (std::size_t j = 0; j < p_total; ++j) {
        if (!is_dropped_[j]) { correlations_y_(static_cast<Eigen::Index>(j)) =
            getColumn(j).dot(y_); }
    }

    // Penalty diagonal over the unified space.
    configurePenaltyDiag();

    // Empty gram cache (ever-active slots are rebuilt lazily).
    slot_of_.assign(p_total, -1);
    slot_var_.clear();
    gram_rows_.clear();
    gram_q_.resize(0);
    gram_active_ = true;
    r_dirty_ = false;

    // Re-enter the current support into the fresh gram cache: after
    // reconnect()/load the slots are empty while actives_ is not, and the
    // sweep path reads gram_q_(slot_of_[j]) — slot -1 would be indexed.
    if (!actives_.empty()) {
        if (actives_.size() >= gram_cap_) {
            gram_active_ = false;    // naive sweeps; r_ is (de)serialized
        } else {
            for (std::size_t j : actives_) { gramEnter(j); }
            // gramEnter maintains q incrementally under the zero-coefficient
            // entry invariant of the solve loop; the deserialized support
            // carries nonzero coefficients, so recompute
            // q(s) = x_s^T (y - X beta) over the completed gram directly.
            const Eigen::Index S = static_cast<Eigen::Index>(slot_var_.size());
            for (Eigen::Index s = 0; s < S; ++s) {
                double q = correlations_y_(
                    static_cast<Eigen::Index>(slot_var_[static_cast<std::size_t>(s)]));
                for (Eigen::Index t = 0; t < S; ++t) {
                    q -= gram_rows_[static_cast<std::size_t>(s)](t) *
                         beta_cd_(static_cast<Eigen::Index>(
                             slot_var_[static_cast<std::size_t>(t)]));
                }
                gram_q_(s) = q;
            }
        }
    }

    cd_ready_ = true;
}


void TCDLASSO_Solver::configurePenaltyDiag() {
    const std::size_t p_total = p_original_ + num_dummies_;
    penalty_diag_.setZero(static_cast<Eigen::Index>(p_total));
    if (penalty_kind_ == PenaltyKind::NONE || lambda2_ <= 0.0) { return; }
    if (penalty_kind_ == PenaltyKind::GROUP ||
        penalty_kind_ == PenaltyKind::GROUP_CENTERING) {
        throw std::logic_error(concatMsg(
            solverTypeToString(),
            "::configurePenaltyDiag: GROUP/GROUP_CENTERING kinds require a "
            "specialized descendant override."));
    }

    if (penalty_kind_ == PenaltyKind::DIAGONAL) {
        if (static_cast<std::size_t>(penalty_dvec_.size()) != p_original_) {
            throw std::invalid_argument(concatMsg(
                solverTypeToString(),
                "::configurePenaltyDiag: diagonal weights size (",
                penalty_dvec_.size(), ") does not match p (", p_original_, ")"));
        }
        penalty_diag_.head(static_cast<Eigen::Index>(p_original_)) =
            lambda2_ * penalty_dvec_;
    } else {  // SPARSE
        if (static_cast<std::size_t>(tikhonov_K_.rows()) != p_original_ ||
            static_cast<std::size_t>(tikhonov_K_.cols()) != p_original_) {
            throw std::invalid_argument(concatMsg(
                solverTypeToString(),
                "::configurePenaltyDiag: Tikhonov matrix is ",
                tikhonov_K_.rows(), "x", tikhonov_K_.cols(),
                ", expected ", p_original_, "x", p_original_));
        }
        for (std::size_t j = 0; j < p_original_; ++j) {
            penalty_diag_(static_cast<Eigen::Index>(j)) =
                lambda2_ * tikhonov_K_.coeff(static_cast<Eigen::Index>(j),
                                             static_cast<Eigen::Index>(j));
        }
    }
    // Dummy block: always ridge with kappa_dummy_ (exchangeability default).
    penalty_diag_.tail(static_cast<Eigen::Index>(num_dummies_))
        .setConstant(lambda2_ * kappa_dummy_);
}


void TCDLASSO_Solver::penaltyRankUpdate(std::size_t j, double delta) {
    switch (penalty_kind_) {
        case PenaltyKind::NONE:
            return;
        case PenaltyKind::DIAGONAL:
            l2kb_(static_cast<Eigen::Index>(j)) +=
                delta * penalty_diag_(static_cast<Eigen::Index>(j));
            return;
        case PenaltyKind::SPARSE:
            if (j < p_original_) {
                const double f = lambda2_ * delta;
                for (Eigen::SparseMatrix<double>::InnerIterator
                         it(tikhonov_K_, static_cast<Eigen::Index>(j));
                     it; ++it) {
                    l2kb_(it.row()) += f * it.value();
                }
            } else {  // dummy ridge block
                l2kb_(static_cast<Eigen::Index>(j)) +=
                    delta * penalty_diag_(static_cast<Eigen::Index>(j));
            }
            return;
        case PenaltyKind::GROUP:
        case PenaltyKind::GROUP_CENTERING:
            throw std::logic_error(concatMsg(
                solverTypeToString(),
                "::penaltyRankUpdate: GROUP/GROUP_CENTERING kinds require a "
                "specialized descendant override."));
    }
}


double TCDLASSO_Solver::coordinateUpdate(std::size_t j, double xr, double lambda1) {
    const Eigen::Index ji = static_cast<Eigen::Index>(j);
    const double n2 = colnorm2_(ji);
    const double denom = n2 + penalty_diag_(ji);
    const double b = xr + n2 * beta_cd_(ji) -
                     (penaltyGradient(j) - penalty_diag_(ji) * beta_cd_(ji));
    const double bn = softThreshold(b, lambda1) / denom;
    const double delta = bn - beta_cd_(ji);
    if (delta != 0.0) {
        beta_cd_(ji) = bn;
        penaltyRankUpdate(j, delta);
    }
    return delta;
}


void TCDLASSO_Solver::refreshResidualCd() {
    r_ = y_;
    for (std::size_t j : actives_) {
        const Eigen::Index ji = static_cast<Eigen::Index>(j);
        if (beta_cd_(ji) != 0.0) { r_ -= beta_cd_(ji) * getColumn(j); }
    }
    r_dirty_ = false;
}


Eigen::Index TCDLASSO_Solver::gramEnter(std::size_t j) {
    if (slot_of_[j] >= 0) { return slot_of_[j]; }    // ever-active: free re-entry

    const Eigen::Index s = static_cast<Eigen::Index>(slot_var_.size());
    slot_var_.push_back(j);
    for (auto& row : gram_rows_) {
        row.conservativeResize(s + 1);
        row(s) = 0.0;
    }
    gram_rows_.emplace_back(Eigen::VectorXd::Zero(s + 1));
    gram_q_.conservativeResize(s + 1);
    slot_of_[j] = s;

    double qs = correlations_y_(static_cast<Eigen::Index>(j));
    for (Eigen::Index t = 0; t < s; ++t) {
        const std::size_t k = slot_var_[static_cast<std::size_t>(t)];
        const double g = getColumn(j).dot(getColumn(k));
        gram_rows_[static_cast<std::size_t>(s)](t) = g;
        gram_rows_[static_cast<std::size_t>(t)](s) = g;
        qs -= g * beta_cd_(static_cast<Eigen::Index>(k));
    }
    gram_rows_[static_cast<std::size_t>(s)](s) =
        colnorm2_(static_cast<Eigen::Index>(j));
    gram_q_(s) = qs;
    return s;
}


long TCDLASSO_Solver::solveAtLambda(double lambda1, double cd_tol,
                                const std::vector<std::size_t>* candidates) {
    const std::size_t p_total = p_original_ + num_dummies_;
    long sweeps = 0;
    const long sweep_budget = static_cast<long>(max_sweeps_per_solve_);
    bool budget_hit = false;

    while (true) {
        // ========================================================
        // Inner loop: cycle the support to convergence. The stopping
        // rule is RELATIVE to the coefficient scale (support decisions
        // only need KKT-slack precision, which is relative); the sweep
        // budget bounds limit cycles on nearly collinear active sets.
        // ========================================================
        if (gram_active_) {
            while (true) {
                sweeps++;
                double max_delta = 0.0;
                double beta_scale = 1.0;
                for (std::size_t j : actives_) {
                    const Eigen::Index s = slot_of_[j];
                    const double delta =
                        coordinateUpdate(j, gram_q_(s), lambda1);
                    if (delta != 0.0) {
                        r_dirty_ = true;
                        const Eigen::Index ns =
                            static_cast<Eigen::Index>(slot_var_.size());
                        gram_q_.head(ns) -=
                            delta * gram_rows_[static_cast<std::size_t>(s)].head(ns);
                        max_delta = std::max(max_delta, std::abs(delta));
                    }
                    beta_scale = std::max(
                        beta_scale,
                        std::abs(beta_cd_(static_cast<Eigen::Index>(j))));
                }
                if (max_delta < cd_tol * beta_scale) { break; }
                if (sweeps >= sweep_budget) { budget_hit = true; break; }
            }
        } else {
            if (r_dirty_) { refreshResidualCd(); }
            while (true) {
                sweeps++;
                double max_delta = 0.0;
                double beta_scale = 1.0;
                for (std::size_t j : actives_) {
                    const double delta =
                        coordinateUpdate(j, getColumn(j).dot(r_), lambda1);
                    if (delta != 0.0) {
                        r_ -= delta * getColumn(j);
                        max_delta = std::max(max_delta, std::abs(delta));
                    }
                    beta_scale = std::max(
                        beta_scale,
                        std::abs(beta_cd_(static_cast<Eigen::Index>(j))));
                }
                if (max_delta < cd_tol * beta_scale) { break; }
                if (sweeps >= sweep_budget) { budget_hit = true; break; }
            }
        }

        // ========================================================
        // Outer loop: KKT pass over the non-support
        // ========================================================
        const std::size_t limit =
            candidates ? candidates->size() : p_total;

        bool need_residual = !gram_active_;
        if (!need_residual) {
            for (std::size_t idx = 0; idx < limit; ++idx) {
                const std::size_t j = candidates ? (*candidates)[idx] : idx;
                if (!in_active_[j] && !is_dropped_[j] && slot_of_[j] < 0) {
                    need_residual = true;
                    break;
                }
            }
        }
        if (need_residual && r_dirty_) { refreshResidualCd(); }
        if (candidates) { restricted_kkt_passes_++; } else { full_kkt_passes_++; }

        bool violated = false;
        const double thresh = lambda1 * (1.0 + kkt_slack_);
        for (std::size_t idx = 0; idx < limit; ++idx) {
            const std::size_t j = candidates ? (*candidates)[idx] : idx;
            if (in_active_[j] || is_dropped_[j]) { continue; }

            const Eigen::Index ji = static_cast<Eigen::Index>(j);
            const double cj = (gram_active_ && slot_of_[j] >= 0)
                                  ? gram_q_(slot_of_[j])
                                  : getColumn(j).dot(r_);
            correlations_(ji) = cj;

            if (std::abs(cj - penaltyGradient(j)) > thresh) {  // penalized KKT entry
                actives_.push_back(j);
                in_active_[j] = 1;
                if (gram_active_) {
                    if (slot_var_.size() >= gram_cap_ && slot_of_[j] < 0) {
                        // Ever-active set too large: gram build would cost
                        // O(n * S^2) — fall back to naive sweeps for good.
                        gram_active_ = false;
                        if (r_dirty_) { refreshResidualCd(); }
                    } else {
                        gramEnter(j);
                    }
                }
                violated = true;
            }
        }
        if (!violated) { break; }
        if (budget_hit) { break; }        // stall guard: accept current state
    }
    if (budget_hit) {
        sweep_cap_hits_++;
        if (sweep_cap_hits_ == 1) {
            logWarning(concatMsg(
                "solveAtLambda hit the sweep budget (",
                max_sweeps_per_solve_, "); accepting the current iterate "
                "(further hits counted silently, see getSweepCapHits())."));
        }
    }

    // Prune exact zeros (pruned variables keep their gram slot).
    std::vector<std::size_t> keep;
    keep.reserve(actives_.size());
    for (std::size_t j : actives_) {
        if (beta_cd_(static_cast<Eigen::Index>(j)) != 0.0) {
            keep.push_back(j);
        } else {
            in_active_[j] = 0;
        }
    }
    actives_.swap(keep);

    // Refresh correlations_ on the support side (exact values feed the
    // crossing predictor's dummy scan).
    if (gram_active_) {
        for (std::size_t j : actives_) {
            correlations_(static_cast<Eigen::Index>(j)) = gram_q_(slot_of_[j]);
        }
    } else {
        if (r_dirty_) { refreshResidualCd(); }
        for (std::size_t j : actives_) {
            correlations_(static_cast<Eigen::Index>(j)) = getColumn(j).dot(r_);
        }
    }
    return sweeps;
}


double TCDLASSO_Solver::predictNextCrossing() {
    const std::size_t p_total = p_original_ + num_dummies_;

    // Zeroth order: an inactive dummy enters when lambda1 falls to its
    // penalized correlation |x_j^T r - lambda2 (K beta)_j|.
    double pred = 0.0;
    for (std::size_t j = dummy_start_idx_; j < p_total; ++j) {
        if (in_active_[j] || is_dropped_[j]) { continue; }
        const Eigen::Index ji = static_cast<Eigen::Index>(j);
        pred = std::max(pred,
                        std::abs(correlations_(ji) - penaltyGradient(j)));
    }

    // Secant refinement: the dummy correlations decay as lambda1 drops
    // (entering reals absorb residual), so extrapolate the decay line and
    // intersect it with the identity.
    double target = pred;
    if (pred_prev_lambda_ > 0.0 &&
        pred_prev_lambda_ > lambda_current_ * (1.0 + 1e-12)) {
        const double slope = (pred - pred_prev_value_) /
                             (lambda_current_ - pred_prev_lambda_);
        if (slope < 0.999) {
            const double lstar = (pred - slope * lambda_current_) / (1.0 - slope);
            if (lstar > 0.0 && lstar < lambda_current_) { target = lstar; }
        }
    }
    pred_prev_lambda_ = lambda_current_;
    pred_prev_value_ = pred;
    return target;
}


std::size_t TCDLASSO_Solver::countActiveDummiesCd() const {
    std::size_t cnt = 0;
    for (std::size_t j : actives_) {
        if (j >= dummy_start_idx_ &&
            beta_cd_(static_cast<Eigen::Index>(j)) != 0.0) { cnt++; }
    }
    return cnt;
}


bool TCDLASSO_Solver::supportOnlyDummyGrowth(
    const std::vector<std::size_t>& hi_support,
    const std::vector<std::size_t>& lo_support) const {

    std::unordered_set<std::size_t> lo(lo_support.begin(), lo_support.end());
    for (std::size_t j : hi_support) {
        if (lo.find(j) == lo.end()) { return false; }   // not nested
    }
    std::unordered_set<std::size_t> hi(hi_support.begin(), hi_support.end());
    for (std::size_t j : lo_support) {
        if (hi.find(j) == hi.end() && j < dummy_start_idx_) { return false; }
    }
    return true;
}


void TCDLASSO_Solver::collectCandidates(double lambda_lo,
                                    std::vector<std::size_t>& candidates) const {
    const std::size_t p_total = p_original_ + num_dummies_;
    candidates.clear();
    for (std::size_t j = 0; j < p_total; ++j) {
        if (is_dropped_[j]) { continue; }
        const Eigen::Index ji = static_cast<Eigen::Index>(j);
        if (in_active_[j] ||
            std::abs(correlations_(ji) - penaltyGradient(j)) >=
                0.9 * lambda_lo) {
            candidates.push_back(j);
        }
    }
}


void TCDLASSO_Solver::recordCrossingStep(double lambda1) {
    currentStep_++;
    lambda_.push_back(lambda1);

    // Actions: support diff vs the previously recorded step (R lars
    // convention via actionAdd/actionDrop).
    std::vector<int> actions_this_step;
    std::unordered_set<std::size_t> prev(beta_idx_.begin(), beta_idx_.end());
    std::unordered_set<std::size_t> curr(actives_.begin(), actives_.end());
    for (std::size_t j : actives_) {
        if (prev.find(j) == prev.end()) {
            actions_this_step.push_back(actionAdd(j));
            num_additions_++;
        }
    }
    for (std::size_t j : beta_idx_) {
        if (curr.find(j) == curr.end()) {
            actions_this_step.push_back(actionDrop(j));
            num_removals_++;
        }
    }
    actions_.emplace_back(std::move(actions_this_step));

    // Sparse beta snapshot of the certified solution.
    Eigen::VectorXd values(static_cast<Eigen::Index>(actives_.size()));
    for (std::size_t k = 0; k < actives_.size(); ++k) {
        values(static_cast<Eigen::Index>(k)) =
            beta_cd_(static_cast<Eigen::Index>(actives_[k]));
    }
    setRunningBeta(actives_, values);
    recordBetaStep();

    // Diagnostics.
    if (r_dirty_) { refreshResidualCd(); }
    const double rss = r_.dot(r_);
    RSS_.push_back(rss);
    R2_.push_back(1.0 - rss / RSS_[0]);
    count_active_dummies_ = countActiveDummiesCd();
    updateDummyTracking();
    DoF_.push_back(actives_.size() + interceptDoF());

    // Rebuild the inactive set (base rebuild is gated on any_dropped_).
    any_dropped_ = true;
    updateInactiveSet();
}


double TCDLASSO_Solver::getKktViolation(double lambda1) {
    validateConnected();
    ensureCdState();
    const double lam = (lambda1 > 0.0) ? lambda1 : lambda_current_;
    if (r_dirty_) { refreshResidualCd(); }

    const std::size_t p_total = p_original_ + num_dummies_;
    double viol = 0.0;
    for (std::size_t j = 0; j < p_total; ++j) {
        if (is_dropped_[j]) { continue; }
        const Eigen::Index ji = static_cast<Eigen::Index>(j);
        const double grad = getColumn(j).dot(r_) - penaltyGradient(j);
        if (in_active_[j] && beta_cd_(ji) != 0.0) {
            viol = std::max(viol, std::abs(
                grad - lam * ((beta_cd_(ji) > 0.0) ? 1.0 : -1.0)));
        } else {
            viol = std::max(viol, std::abs(grad) - lam);
        }
    }
    return std::max(viol, 0.0);
}


// ============================================================================
// Certified localization (T-CD theory note, Part II; §6 pseudocode)
// ============================================================================

// ----------------------------------------------------------------------------
// Penalty hooks over the unified index space
// ----------------------------------------------------------------------------

double TCDLASSO_Solver::penaltyPair(std::size_t i, std::size_t j) const {
    if (lambda2_ <= 0.0 || penalty_kind_ == PenaltyKind::NONE) { return 0.0; }
    switch (penalty_kind_) {
        case PenaltyKind::DIAGONAL:
            return (i == j) ? penalty_diag_(static_cast<Eigen::Index>(i)) : 0.0;
        case PenaltyKind::SPARSE:
            if (i < p_original_ && j < p_original_) {
                return lambda2_ * tikhonov_K_.coeff(static_cast<Eigen::Index>(i),
                                                    static_cast<Eigen::Index>(j));
            }
            // Dummy ridge block (INDEPENDENT_RIDGE): no real-dummy coupling.
            return (i == j) ? penalty_diag_(static_cast<Eigen::Index>(i)) : 0.0;
        default:
            throw std::logic_error(concatMsg(
                solverTypeToString(),
                "::penaltyPair: GROUP/GROUP_CENTERING kinds require a "
                "specialized descendant override."));
    }
}


void TCDLASSO_Solver::penaltyMatVec(const std::vector<std::size_t>& A,
                                    const Eigen::VectorXd& vA,
                                    Eigen::VectorXd& out) const {
    const std::size_t p_total = p_original_ + num_dummies_;
    out.setZero(static_cast<Eigen::Index>(p_total));
    if (lambda2_ <= 0.0 || penalty_kind_ == PenaltyKind::NONE) { return; }
    switch (penalty_kind_) {
        case PenaltyKind::DIAGONAL:
            for (std::size_t k = 0; k < A.size(); ++k) {
                const Eigen::Index j = static_cast<Eigen::Index>(A[k]);
                out(j) += penalty_diag_(j) * vA(static_cast<Eigen::Index>(k));
            }
            return;
        case PenaltyKind::SPARSE:
            for (std::size_t k = 0; k < A.size(); ++k) {
                const std::size_t j = A[k];
                const double vk = vA(static_cast<Eigen::Index>(k));
                if (j < p_original_) {
                    const double f = lambda2_ * vk;
                    for (Eigen::SparseMatrix<double>::InnerIterator
                             it(tikhonov_K_, static_cast<Eigen::Index>(j));
                         it; ++it) {
                        out(it.row()) += f * it.value();
                    }
                } else {
                    out(static_cast<Eigen::Index>(j)) +=
                        penalty_diag_(static_cast<Eigen::Index>(j)) * vk;
                }
            }
            return;
        default:
            throw std::logic_error(concatMsg(
                solverTypeToString(),
                "::penaltyMatVec: GROUP/GROUP_CENTERING kinds require a "
                "specialized descendant override."));
    }
}


void TCDLASSO_Solver::zeroPenaltyState() { l2kb_.setZero(); }


void TCDLASSO_Solver::rebuildPenaltyState() {
    zeroPenaltyState();
    for (std::size_t j : actives_) {
        const double b = beta_cd_(static_cast<Eigen::Index>(j));
        if (b != 0.0) { penaltyRankUpdate(j, b); }
    }
}


double TCDLASSO_Solver::strongConvexityMu() const {
    // H = X^T X + diag(penalty) >= diag(penalty) for diagonal kinds.
    if (lambda2_ <= 0.0 || penalty_kind_ != PenaltyKind::DIAGONAL) { return 0.0; }
    const std::size_t p_total = p_original_ + num_dummies_;
    double mu = 0.0;
    bool found = false;
    for (std::size_t j = 0; j < p_total; ++j) {
        if (is_dropped_[j]) { continue; }
        const double d = penalty_diag_(static_cast<Eigen::Index>(j));
        if (!found || d < mu) { mu = d; found = true; }
    }
    return (found && mu > 0.0) ? mu : 0.0;
}


double TCDLASSO_Solver::dataGram(std::size_t i, std::size_t j) const {
    if (gram_active_ && slot_of_[i] >= 0 && slot_of_[j] >= 0) {
        return gram_rows_[static_cast<std::size_t>(slot_of_[i])](slot_of_[j]);
    }
    if (i == j) { return colnorm2_(static_cast<Eigen::Index>(i)); }
    return getColumn(i).dot(getColumn(j));
}


// ----------------------------------------------------------------------------
// Point certificates (Defs. 8-11; Algorithm 1 CERTIFY)
// ----------------------------------------------------------------------------

double TCDLASSO_Solver::currentGap(double lambda1, double& c_inf) {
    // Augmented-form dual (Def. 9): dual point alpha * r_bar with
    // alpha = min(1, lambda / ||c||_inf); everything from maintained state.
    const std::size_t p_total = p_original_ + num_dummies_;
    if (r_dirty_) { refreshResidualCd(); }

    c_inf = 0.0;
    for (std::size_t j = 0; j < p_total; ++j) {
        if (is_dropped_[j]) { continue; }
        const double c = correlations_(static_cast<Eigen::Index>(j)) - penaltyGradient(j);
        c_inf = std::max(c_inf, std::abs(c));
    }
    double l1 = 0.0;      // ||beta||_1
    double l2bKb = 0.0;   // lambda2 * beta^T K beta = sum_j beta_j [lambda2 K beta]_j
    for (std::size_t j : actives_) {
        const double b = beta_cd_(static_cast<Eigen::Index>(j));
        l1 += std::abs(b);
        l2bKb += b * penaltyGradient(j);
    }
    // Cancellation-free form. With alpha = 1 - eps and
    // ||r||^2 + lambda2 b'Kb - y'r = -b'c(b):
    //   G = sum_j b_j (lambda s_j - c_j)  +  eps b'c  +  eps^2/2 (||r||^2 + lambda2 b'Kb),
    // every term a product of small numbers (the naive difference of O(||y||^2)
    // quantities floors at ~1e-10 and cannot certify tiny coefficients, e.g. at
    // lambda2 ~ 1e6 where beta ~ 1e-8).
    const double eps = (c_inf > lambda1) ? (c_inf - lambda1) / c_inf : 0.0;
    double kkt_term = 0.0;   // sum_j b_j (lambda s_j - c_j) >= 0 up to rounding
    double bc = 0.0;         // b'c
    for (std::size_t j : actives_) {
        const Eigen::Index ji = static_cast<Eigen::Index>(j);
        const double b = beta_cd_(ji);
        const double c = correlations_(ji) - penaltyGradient(j);
        kkt_term += b * (lambda1 * ((b > 0.0) ? 1.0 : -1.0) - c);
        bc += b * c;
    }
    (void)l1;
    const double rn2 = r_.squaredNorm();
    const double G = kkt_term + eps * bc + 0.5 * eps * eps * (rn2 + l2bKb);
    return std::max(G, 0.0);
}


double TCDLASSO_Solver::outTestCoefficient(std::size_t j) const {
    // General: sqrt(H_jj) (Lemma 2(b)). Diagonal kinds (NONE / DIAGONAL): an
    // active coordinate has |x_j' r*| = lambda + lambda2 K_jj |beta_j*| > lambda,
    // and |x_j' (r* - r)| <= ||x_j|| ||X (beta - beta*)|| <= ||x_j|| sqrt(2G);
    // for an inactive coordinate the raw and the penalized correlation
    // coincide, so the coefficient ||x_j|| replaces sqrt(H_jj): no sqrt(lambda2)
    // amplification.
    const Eigen::Index ji = static_cast<Eigen::Index>(j);
    if (penalty_kind_ == PenaltyKind::NONE || penalty_kind_ == PenaltyKind::DIAGONAL) {
        return std::sqrt(colnorm2_(ji));
    }
    return std::sqrt(colnorm2_(ji) + penalty_diag_(ji));
}


bool TCDLASSO_Solver::certifyPoint(double lambda1, CertifiedPoint& P) {
    const std::size_t p_total = p_original_ + num_dummies_;
    certified_solves_++;

    // Certification solves use STRICT KKT entry (no relative slack): the
    // slack would leave inactive |c_j| above lambda by up to slack*lambda,
    // pushing the dual point's alpha below 1 and putting a floor under the
    // gap (G ~ slack * lambda * ||beta||_1) that no CD tolerance removes.
    const double saved_slack = kkt_slack_;
    kkt_slack_ = 0.0;

    double tol = cd_tol_;
    bool certified = false;
    std::vector<std::size_t> S;
    S.reserve(p_total);

    for (std::size_t round = 0; round < max_certify_rounds_; ++round) {
        cd_sweeps_ += solveAtLambda(lambda1, tol, nullptr);
        lambda_points_++;
        if (r_dirty_) { refreshResidualCd(); }

        double c_inf = 0.0;
        const double G = currentGap(lambda1, c_inf);
        const double sq2G = std::sqrt(2.0 * G);

        // OUT statuses (Def. 10, assumption-free) for the INACTIVE
        // coordinates: |c_j| + sqrt(H_jj) sqrt(2G) < lambda. Active
        // coordinates are decided by the IN test only (for them the OUT
        // test is rounding noise at |c_j| = lambda). S = actives + unscreened
        // inactives contains the true support and leaves beta^S = beta.
        S.clear();
        double kkt = 0.0;
        for (std::size_t j = 0; j < p_total; ++j) {
            if (is_dropped_[j]) { continue; }
            const Eigen::Index ji = static_cast<Eigen::Index>(j);
            const double cj = correlations_(ji) - penaltyGradient(j);
            if (in_active_[j] && beta_cd_(ji) != 0.0) {
                kkt = std::max(kkt, std::abs(cj - lambda1 * ((beta_cd_(ji) > 0.0) ? 1.0 : -1.0)));
                S.push_back(j);
                continue;
            }
            kkt = std::max(kkt, std::abs(cj) - lambda1);
            const bool out = (std::abs(cj) + outTestCoefficient(j) * sq2G < lambda1);
            if (!out) { S.push_back(j); }
        }

        // Coefficient-error radius: global (A0-s) or restricted to S
        // (Lemma 2(c)/(d)); unavailable while S is too large for the
        // eigen-solve or H_SS is numerically singular.
        double radius = -1.0;
        {
            const double mu = strongConvexityMu();
            if (mu > 0.0) {
                radius = std::sqrt(2.0 * G / mu);
            } else if (S.size() <= mu_s_cap_) {
                const Eigen::Index m = static_cast<Eigen::Index>(S.size());
                Eigen::MatrixXd H(m, m);
                for (Eigen::Index a = 0; a < m; ++a) {
                    for (Eigen::Index b = 0; b <= a; ++b) {
                        const double v = dataGram(S[static_cast<std::size_t>(a)],
                                                  S[static_cast<std::size_t>(b)]) +
                                         penaltyPair(S[static_cast<std::size_t>(a)],
                                                     S[static_cast<std::size_t>(b)]);
                        H(a, b) = v;
                        H(b, a) = v;
                    }
                }
                if (m > 0) {
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H, Eigen::EigenvaluesOnly);
                    const double lmin = es.eigenvalues().minCoeff();
                    const double lmax = es.eigenvalues().maxCoeff();
                    if (lmin > 1e-10 * std::max(lmax, 1.0)) {
                        radius = std::sqrt(2.0 * G / lmin);
                    }
                } else {
                    radius = 0.0;   // S empty: nothing to decide IN
                }
            }
        }

        // IN statuses (with sign) / undecided.
        std::size_t undecided = 0;
        if (radius >= 0.0) {
            for (std::size_t j : S) {
                const Eigen::Index ji = static_cast<Eigen::Index>(j);
                if (in_active_[j] && std::abs(beta_cd_(ji)) > radius) { continue; }
                undecided++;
            }
        } else {
            undecided = S.size();
        }

        if (undecided == 0) {
            P = CertifiedPoint{};
            P.lambda = lambda1;
            P.support = actives_;
            std::sort(P.support.begin(), P.support.end());
            P.sign.resize(P.support.size());
            P.beta.resize(P.support.size());
            for (std::size_t k = 0; k < P.support.size(); ++k) {
                const double b = beta_cd_(static_cast<Eigen::Index>(P.support[k]));
                P.sign[k] = (b > 0.0) ? 1 : -1;
                P.beta[k] = b;
            }
            P.num_dummies = countActiveDummiesCd();
            P.gap = G;
            P.kkt = std::max(kkt, 0.0);
            P.rss = r_.squaredNorm();
            P.conventional = false;
            lambda_current_ = lambda1;
            certified = true;
            break;
        }
        tol = std::max(tol * 0.1, 1e-15);
    }

    kkt_slack_ = saved_slack;
    if (!certified) { certify_stalls_++; }
    return certified;
}


bool TCDLASSO_Solver::certifyPointNudged(double& lambda1, double lambda_floor,
                                         CertifiedPoint& P) {
    for (std::size_t k = 0; k <= max_nudges_; ++k) {
        if (certifyPoint(lambda1, P)) { return true; }
        const double next = lambda1 * (1.0 - 2.0 * lambda_rel_tol_);
        if (!(next > lambda_floor) || !(next > 0.0)) { return false; }
        lambda1 = next;
    }
    return false;
}


// ----------------------------------------------------------------------------
// Closed-form continuation (Def. 5 / Prop. 2; Algorithm 2)
// ----------------------------------------------------------------------------

bool TCDLASSO_Solver::continuation(const CertifiedPoint& P, bool downward,
                                   Continuation& C) {
    C = Continuation{};
    C.downward = downward;
    C.lambda0 = P.lambda;
    C.has_next = false;
    C.t_next = std::numeric_limits<double>::max();
    if (P.conventional) { return false; }

    const std::size_t p_total = p_original_ + num_dummies_;
    const std::size_t m = P.support.size();
    if (m == 0) { return false; }
    const Eigen::Index mi = static_cast<Eigen::Index>(m);

    // H_AA = X_A^T X_A + lambda2 K_AA from the Gram cache + penalty pairs.
    Eigen::MatrixXd H(mi, mi);
    for (Eigen::Index a = 0; a < mi; ++a) {
        for (Eigen::Index b = 0; b <= a; ++b) {
            const std::size_t ja = P.support[static_cast<std::size_t>(a)];
            const std::size_t jb = P.support[static_cast<std::size_t>(b)];
            const double v = dataGram(ja, jb) + penaltyPair(ja, jb);
            H(a, b) = v;
            H(b, a) = v;
        }
    }
    Eigen::LLT<Eigen::MatrixXd> llt(H);
    if (llt.info() != Eigen::Success) { return false; }

    Eigen::VectorXd sgn(mi), rhs(mi);
    for (Eigen::Index a = 0; a < mi; ++a) {
        const std::size_t ja = P.support[static_cast<std::size_t>(a)];
        sgn(a) = static_cast<double>(P.sign[static_cast<std::size_t>(a)]);
        rhs(a) = correlations_y_(static_cast<Eigen::Index>(ja)) - P.lambda * sgn(a);
    }
    C.beta_star = llt.solve(rhs);     // closed-form anchor (Prop. 1(b))
    C.dir = llt.solve(sgn);           // direction d_A = H_AA^{-1} s_A

    // One data pass: c(lambda0) = X^T (y - X_A beta*) - lambda2 K_{:,A} beta*,
    // cdot = -X^T (X_A d) - lambda2 K_{:,A} d.
    const Eigen::Index n = y_.size();
    Eigen::VectorXd u = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(n);
    for (Eigen::Index a = 0; a < mi; ++a) {
        const std::size_t ja = P.support[static_cast<std::size_t>(a)];
        u += C.beta_star(a) * getColumn(ja);
        v += C.dir(a) * getColumn(ja);
    }
    const Eigen::VectorXd res = y_ - u;
    Eigen::VectorXd pb, pd;
    penaltyMatVec(P.support, C.beta_star, pb);
    penaltyMatVec(P.support, C.dir, pd);

    C.c_exact.setZero(static_cast<Eigen::Index>(p_total));
    C.cdot.setZero(static_cast<Eigen::Index>(p_total));
    [[maybe_unused]] const int nt = threadsFor(2.0 * static_cast<double>(X_->rows()) * static_cast<double>(p_total));
    #pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
    for (std::size_t j = 0; j < p_total; ++j) {
        if (is_dropped_[j]) { continue; }
        const Eigen::Index ji = static_cast<Eigen::Index>(j);
        C.c_exact(ji) = getColumn(j).dot(res) - pb(ji);
        C.cdot(ji) = -(getColumn(j).dot(v)) - pd(ji);
    }
    std::vector<char> in_sup(p_total, 0);
    for (Eigen::Index a = 0; a < mi; ++a) {
        const std::size_t ja = P.support[static_cast<std::size_t>(a)];
        in_sup[ja] = 1;
        C.c_exact(static_cast<Eigen::Index>(ja)) = P.lambda * sgn(a);   // exact on A
        C.cdot(static_cast<Eigen::Index>(ja)) = -sgn(a);
    }

    // Candidate event times (Def. 5), downward: lambda = lambda0 - t;
    // upward: lambda = lambda0 + t (direction -d).
    struct Cand { double t; std::size_t j; int kind; int sign; };
    std::vector<Cand> cands;
    for (Eigen::Index a = 0; a < mi; ++a) {
        const double d = C.dir(a);
        if (d == 0.0) { continue; }
        const double t = downward ? -C.beta_star(a) / d : C.beta_star(a) / d;
        if (t > 0.0) {
            cands.push_back({t, P.support[static_cast<std::size_t>(a)], -1,
                             P.sign[static_cast<std::size_t>(a)]});
        }
    }
    for (std::size_t j = 0; j < p_total; ++j) {
        if (is_dropped_[j] || in_sup[j]) { continue; }
        const Eigen::Index ji = static_cast<Eigen::Index>(j);
        const double cj = C.c_exact(ji);
        const double cd = C.cdot(ji);
        for (int sigma = -1; sigma <= 1; sigma += 2) {
            const double sg = static_cast<double>(sigma);
            const double den = 1.0 + sg * cd;
            const double num = std::max(P.lambda - sg * cj, 0.0);   // > 0 by strict inactivity
            if (downward) {
                if (den > 0.0) { cands.push_back({num / den, j, +1, sigma}); }
            } else {
                if (den < 0.0) { cands.push_back({num / (-den), j, +1, sigma}); }
            }
        }
    }
    continuations_++;
    C.ok = true;
    if (cands.empty()) { return true; }   // no knot in this direction (has_next = false)

    double t_next = cands.front().t;
    for (const auto& c : cands) { if (c.t < t_next) { t_next = c.t; } }
    C.t_next = t_next;
    C.has_next = true;
    const double eta = lambda_rel_tol_ * P.lambda;
    for (const auto& c : cands) {
        if (c.t <= t_next + eta) {
            C.event_vars.push_back(c.j);
            C.event_kind.push_back(c.kind);
            C.event_sign.push_back(c.sign);
        }
    }
    return true;
}


bool TCDLASSO_Solver::continuationCached(const CertifiedPoint& P, bool downward,
                                         Continuation& C) {
    for (const auto& e : cont_cache_) {
        if (e.ok && e.downward == downward && e.lambda0 == P.lambda &&
            static_cast<std::size_t>(e.beta_star.size()) == P.support.size()) {
            C = e;
            return true;
        }
    }
    if (!continuation(P, downward, C)) { return false; }
    if (cont_cache_.size() >= cont_cache_cap_) { cont_cache_.erase(cont_cache_.begin()); }
    cont_cache_.push_back(C);
    return true;
}


// ----------------------------------------------------------------------------
// Bracket certificates (Defs. 12-13, Prop. 3; Algorithm 3)
// ----------------------------------------------------------------------------

TCDLASSO_Solver::BracketResult TCDLASSO_Solver::certifyBracket(
    const CertifiedPoint& P_hi, const CertifiedPoint& P_lo,
    const Continuation* C_down) {

    BracketResult R;
    const std::size_t p_total = p_original_ + num_dummies_;
    const double width = P_hi.lambda - P_lo.lambda;
    const double eta = lambda_rel_tol_ * P_hi.lambda;
    R.supports_differ = (P_hi.support != P_lo.support) || (P_hi.sign != P_lo.sign);

    Continuation Cd_local;
    const Continuation* Cd = C_down;
    const Eigen::VectorXd* c_hi = nullptr;
    if (P_hi.conventional) {
        // lambda_max is a knot (Def. 1): kappa_down is known exactly, and
        // c(lambda_max) = X^T y.
        R.kappa_down = P_hi.lambda;
        c_hi = &correlations_y_;
    } else {
        if (Cd == nullptr || !Cd->ok || Cd->lambda0 != P_hi.lambda || !Cd->downward) {
            if (continuationCached(P_hi, true, Cd_local)) { Cd = &Cd_local; } else { Cd = nullptr; }
        }
        if (Cd == nullptr || !Cd->ok) { return R; }            // UNRESOLVED, knots unknown
        if (!R.supports_differ && Cd->t_next > width + eta) {   // Def. 12 (from above)
            R.tag = BracketTag::KF;
            brackets_kf_++;
            return R;
        }
        R.kappa_down = Cd->has_next ? P_hi.lambda - Cd->t_next : P_lo.lambda;
        c_hi = &Cd->c_exact;
    }

    // Dummy-bounded bracket (Prop. 3): every dummy inactive at the top stays
    // inactive on the whole bracket, so N_D <= N_D(lambda_hi) throughout.
    if (P_lo.num_dummies <= P_hi.num_dummies) {
        std::vector<char> in_hi(p_total, 0);
        for (std::size_t j : P_hi.support) { in_hi[j] = 1; }
        const double ynorm = y_.norm();
        const double bound = ynorm * (1.0 / P_lo.lambda - 1.0 / P_hi.lambda);
        bool all_pass = true;
        for (std::size_t j = dummy_start_idx_; j < p_total; ++j) {
            if (is_dropped_[j] || in_hi[j]) { continue; }
            const Eigen::Index ji = static_cast<Eigen::Index>(j);
            if (std::abs((*c_hi)(ji)) / P_hi.lambda + outTestCoefficient(j) * bound >= 1.0) {
                all_pass = false;
                break;
            }
        }
        if (all_pass) {
            R.tag = BracketTag::DB;
            brackets_db_++;
            return R;
        }
    }

    // Upward continuation from the lower point.
    Continuation Cu;
    if (!continuationCached(P_lo, false, Cu) || !Cu.ok) { return R; }
    if (!R.supports_differ && Cu.t_next > width + eta) {        // Def. 12 (from below)
        R.tag = BracketTag::KF;
        brackets_kf_++;
        return R;
    }
    R.kappa_up = Cu.has_next ? P_lo.lambda + Cu.t_next : P_hi.lambda;

    // Single knot / tie batch (Def. 13 item 3).
    if (R.supports_differ && R.kappa_down - R.kappa_up <= eta) {
        R.tag = BracketTag::SK;
        R.knot = R.kappa_down;
        brackets_sk_++;
        return R;
    }
    return R;   // UNRESOLVED: >= 2 knots, cut inside (kappa_up, kappa_down)
}


// ----------------------------------------------------------------------------
// Live-state installation and records
// ----------------------------------------------------------------------------

void TCDLASSO_Solver::loadIterate(const CertifiedPoint& P) {
    for (std::size_t j : actives_) { in_active_[j] = 0; }
    beta_cd_.setZero();
    actives_ = P.support;
    for (std::size_t k = 0; k < P.support.size(); ++k) {
        beta_cd_(static_cast<Eigen::Index>(P.support[k])) = P.beta[k];
        in_active_[P.support[k]] = 1;
    }
    rebuildPenaltyState();

    if (gram_active_) {
        for (std::size_t j : actives_) {
            if (slot_of_[j] >= 0) { continue; }
            if (slot_var_.size() >= gram_cap_) { gram_active_ = false; break; }
            gramEnter(j);
        }
    }
    if (gram_active_) {
        // q(s) = x_s^T (y - X beta) over the completed slot set.
        const Eigen::Index S = static_cast<Eigen::Index>(slot_var_.size());
        for (Eigen::Index s = 0; s < S; ++s) {
            double q = correlations_y_(
                static_cast<Eigen::Index>(slot_var_[static_cast<std::size_t>(s)]));
            for (Eigen::Index t = 0; t < S; ++t) {
                q -= gram_rows_[static_cast<std::size_t>(s)](t) *
                     beta_cd_(static_cast<Eigen::Index>(
                         slot_var_[static_cast<std::size_t>(t)]));
            }
            gram_q_(s) = q;
        }
        r_dirty_ = true;
    } else {
        refreshResidualCd();
    }
    lambda_current_ = P.lambda;
}


void TCDLASSO_Solver::loadContinuation(const CertifiedPoint& P,
                                       const Continuation& C, double t) {
    if (!C.ok || !C.has_next || t <= 0.0) { loadIterate(P); return; }
    CertifiedPoint Q = P;
    Q.conventional = false;
    const double sg = C.downward ? 1.0 : -1.0;
    Q.lambda = P.lambda - sg * t;
    for (std::size_t k = 0; k < Q.support.size(); ++k) {
        const Eigen::Index a = static_cast<Eigen::Index>(k);
        Q.beta[k] = C.beta_star(a) + sg * t * C.dir(a);
    }
    loadIterate(Q);
}


void TCDLASSO_Solver::frontierInsert(CertifiedPoint P) {
    auto it = frontier_below_.begin();
    while (it != frontier_below_.end() && it->lambda > P.lambda) { ++it; }
    if (it != frontier_below_.end() && it->lambda == P.lambda) { *it = std::move(P); return; }
    frontier_below_.insert(it, std::move(P));
}


void TCDLASSO_Solver::closeBatch(CertifiedPoint& P_lo, double knot) {
    // Tie rule (note §4): every further knot within the tie tolerance below
    // the record's knot belongs to the same batch; extend the record over it.
    for (int it = 0; it < 8; ++it) {
        const double eta = lambda_rel_tol_ * knot;
        if (P_lo.lambda <= knot - eta) { return; }
        Continuation Cd;
        if (!continuationCached(P_lo, true, Cd) || !Cd.ok || !Cd.has_next) { return; }
        const double kappa = P_lo.lambda - Cd.t_next;
        if (kappa <= knot - eta) { return; }                    // batch closed

        double lam = kappa * (1.0 - 2.0 * lambda_rel_tol_);
        if (!(lam > 0.0)) { return; }
        loadContinuation(P_lo, Cd, Cd.t_next);                    // warm start at the tie knot
        CertifiedPoint P2;
        if (!certifyPointNudged(lam, 0.0, P2)) { return; }
        const BracketResult R = certifyBracket(P_lo, P2, &Cd);
        if (R.tag == BracketTag::SK) {
            P_lo = P2;                                            // merged into the batch
            // Points of the frontier at or above the new lower point are covered.
            frontier_below_.erase(
                std::remove_if(frontier_below_.begin(), frontier_below_.end(),
                               [&](const CertifiedPoint& q) { return q.lambda >= P2.lambda; }),
                frontier_below_.end());
            continue;
        }
        // Unresolved/other: keep the certified point for the main loop and
        // leave the batch as is (split batches are recorded separately).
        frontierInsert(std::move(P2));
        return;
    }
}


double TCDLASSO_Solver::predictNextCrossingFrom(const Eigen::VectorXd& c_pen,
                                                double lambda_cur) {
    // Same predictor as predictNextCrossing(), fed with exact penalized
    // correlations c_pen (inactive dummies: |c_pen| < lambda_cur).
    const std::size_t p_total = p_original_ + num_dummies_;
    double pred = 0.0;
    for (std::size_t j = dummy_start_idx_; j < p_total; ++j) {
        if (is_dropped_[j]) { continue; }
        const double a = std::abs(c_pen(static_cast<Eigen::Index>(j)));
        if (a < lambda_cur) { pred = std::max(pred, a); }   // inactive dummies only
    }
    double target = pred;
    if (pred_prev_lambda_ > 0.0 &&
        pred_prev_lambda_ > lambda_cur * (1.0 + 1e-12)) {
        const double slope = (pred - pred_prev_value_) /
                             (lambda_cur - pred_prev_lambda_);
        if (slope < 0.999) {
            const double lstar = (pred - slope * lambda_cur) / (1.0 - slope);
            if (lstar > 0.0 && lstar < lambda_cur) { target = lstar; }
        }
    }
    pred_prev_lambda_ = lambda_cur;
    pred_prev_value_ = pred;
    return target;
}


// ----------------------------------------------------------------------------
// The certified crossing loop (Algorithm 4: frontier / ledger)
// ----------------------------------------------------------------------------

void TCDLASSO_Solver::executeStepCertified(std::size_t T_stop, long point_budget,
                                           std::size_t active_cap) {
    cont_cache_.clear();
    // Frontier initialization: the conventional top lambda_max on a fresh
    // path; otherwise (mode switch / legacy checkpoint) certify the live state.
    if (!frontier_init_) {
        frontier_below_.clear();
        if (currentStep_ == 0 && actives_.empty()) {
            frontier_top_ = CertifiedPoint{};
            frontier_top_.lambda = lambda_max_;
            frontier_top_.conventional = true;
        } else {
            double lam = lambda_current_;
            CertifiedPoint P;
            if (!certifyPointNudged(lam, 0.0, P)) {
                fallBackToLegacy(T_stop, point_budget, active_cap);
                return;
            }
            frontier_top_ = P;
        }
        frontier_init_ = true;
    }

    while (
        currentStep_ < maxSteps_ &&
        actives_.size() < active_cap &&
        lambda_points_ < point_budget &&
        count_active_dummies_ < T_stop
    ) {
        const std::size_t next_T = count_active_dummies_ + 1;
        const CertifiedPoint top = frontier_top_;     // copy: frontier_top_ is reassigned below
        const double lam_cur = top.lambda;

        // Downward continuation from the ledger top: path-exhausted test,
        // jump predictor input, warm start, and the KF/SK tests of the
        // bracket below it.
        Continuation Cd;
        bool have_cd = false;
        if (!top.conventional) {
            have_cd = continuationCached(top, true, Cd) && Cd.ok;
            if (have_cd && (!Cd.has_next || !(Cd.t_next < lam_cur))) {
                logInfo("certified localization: no knot below the ledger top — "
                        "path exhausted before T_stop dummies entered.");
                break;
            }
        }
        const double kappa_down = top.conventional ? lam_cur
                                : ((have_cd && Cd.has_next) ? lam_cur - Cd.t_next : 0.0);

        // (a) Nothing certified below: jump (R5 heuristics) and certify.
        if (frontier_below_.empty()) {
            double target;
            if (top.conventional) {
                target = predictNextCrossingFrom(correlations_y_, lam_cur);
            } else if (have_cd) {
                target = predictNextCrossingFrom(Cd.c_exact, lam_cur);
            } else {
                target = 0.9 * lam_cur;
            }
            double lam_next = std::min(target * jump_margin_, lam_cur * 0.999);
            if (!(lam_next > 0.0)) { lam_next = lam_cur * 0.5; }
            if (lam_next < lam_cur * 1e-4) { lam_next = lam_cur * 1e-4; }
            if (kappa_down > 0.0) {
                // Never stop above the first knot: the continuation already
                // certifies that stretch (Prop. 2).
                lam_next = std::min(lam_next, kappa_down * (1.0 - 2.0 * lambda_rel_tol_));
            }
            if (have_cd) { loadContinuation(top, Cd, Cd.t_next); }   // warm start at the knot
            CertifiedPoint P;
            if (!certifyPointNudged(lam_next, 0.0, P)) {
                fallBackToLegacy(T_stop, point_budget, active_cap);
                return;
            }
            frontierInsert(std::move(P));
        }

        // (b) Cover the next bracket [P_a, top].
        CertifiedPoint P_a = frontier_below_.front();
        const BracketResult R = certifyBracket(top, P_a, have_cd ? &Cd : nullptr);

        if (R.tag == BracketTag::KF || R.tag == BracketTag::DB) {
            frontier_below_.erase(frontier_below_.begin());
            frontier_top_ = std::move(P_a);
        } else if (R.tag == BracketTag::SK) {
            frontier_below_.erase(frontier_below_.begin());
            if (P_a.num_dummies >= next_T) {
                closeBatch(P_a, R.knot);
                if (P_a.num_dummies >= next_T) {
                    // A crossing of every order next_T .. N_D(P_a) at the knot
                    // (Def. 6 batch semantics; first-reach extraction).
                    loadIterate(P_a);
                    recordCrossingStep(R.knot);
                    record_gap_.push_back(P_a.gap);
                    record_kkt_.push_back(P_a.kkt);
                }
            }
            frontier_top_ = std::move(P_a);
        } else {
            // UNRESOLVED: >= 2 knots inside. Cut between the two located
            // knots (or at the midpoint) and certify there.
            const double lo_b = P_a.lambda;
            const double hi_b = lam_cur;
            const double eta = lambda_rel_tol_ * hi_b;
            double lam_mid;
            if (R.kappa_up > 0.0 && R.kappa_down > 0.0 && R.kappa_down > R.kappa_up) {
                lam_mid = 0.5 * (R.kappa_up + R.kappa_down);
            } else {
                lam_mid = 0.5 * (lo_b + hi_b);
            }
            lam_mid = std::min(std::max(lam_mid, lo_b + 2.0 * eta), hi_b - 2.0 * eta);
            if (!(lam_mid > lo_b && lam_mid < hi_b)) {
                // Bracket narrower than the tie tolerance: by definition one
                // batch. Treat as SK at the upper located knot.
                frontier_below_.erase(frontier_below_.begin());
                const double knot = (R.kappa_down > 0.0) ? R.kappa_down : hi_b;
                if (P_a.num_dummies >= next_T) {
                    loadIterate(P_a);
                    recordCrossingStep(knot);
                    record_gap_.push_back(P_a.gap);
                    record_kkt_.push_back(P_a.kkt);
                }
                brackets_sk_++;
                frontier_top_ = std::move(P_a);
            } else {
                if (have_cd) {
                    loadContinuation(top, Cd, std::min(Cd.t_next, hi_b - lam_mid));
                }
                CertifiedPoint Pm;
                if (!certifyPointNudged(lam_mid, lo_b + eta, Pm)) {
                    fallBackToLegacy(T_stop, point_budget, active_cap);
                    return;
                }
                brackets_cut_++;
                frontierInsert(std::move(Pm));
            }
        }
        lambda_current_ = frontier_top_.lambda;
    }

    // Sync the live iterate to the ledger top (the state the T-Rex layer
    // and warm restarts read).
    if (!frontier_top_.conventional) { loadIterate(frontier_top_); }
}


// ============================================================================
// Setters
// ============================================================================

void TCDLASSO_Solver::setLambdaRelTol(double tol) {
    if (!(tol > 0.0) || tol >= 1.0) {
        logWarning(concatMsg(
            "Lambda tolerance must be in (0, 1); keeping current value (",
            lambda_rel_tol_, ")."
        ));
        return;
    }
    lambda_rel_tol_ = tol;
}


void TCDLASSO_Solver::setCdTolerances(double cd_tol, double cd_tol_probe) {
    if (!(cd_tol > 0.0) || !(cd_tol_probe > 0.0)) {
        logWarning("CD tolerances must be positive; keeping current values.");
        return;
    }
    cd_tol_ = cd_tol;
    cd_tol_probe_ = cd_tol_probe;
}


void TCDLASSO_Solver::setFullPathGrid(std::size_t grid_points,
                                  double lambda_min_ratio) {
    if (grid_points < 2 || !(lambda_min_ratio > 0.0) || lambda_min_ratio >= 1.0) {
        logWarning("Invalid full-path grid configuration; keeping current values.");
        return;
    }
    grid_points_ = grid_points;
    lambda_min_ratio_ = lambda_min_ratio;
}


// ============================================================================
// Memory report
// ============================================================================

std::size_t TCDLASSO_Solver::memoryBytes() const noexcept {
    using solver_utils::memory_bytes::bytesOf;

    // Path, penalty and CD state
    std::size_t bytes = TSolver_Base::memoryBytes() + bytesOf(lambda_) +
                        bytesOf(penalty_dvec_) + bytesOf(tikhonov_K_) +
                        bytesOf(penalty_diag_) + bytesOf(l2kb_) + bytesOf(beta_cd_) +
                        bytesOf(colnorm2_) + bytesOf(in_active_) + bytesOf(is_dropped_);

    // Gram cache over the ever-active slots
    bytes += bytesOf(correlations_y_) + bytesOf(slot_of_) + bytesOf(slot_var_) +
             bytesOf(gram_rows_) + bytesOf(gram_q_);

    // Certified frontier, continuation cache and step records
    const auto point = [](const CertifiedPoint& P) noexcept {
        return bytesOf(P.support) + bytesOf(P.sign) + bytesOf(P.beta);
    };
    bytes += point(frontier_top_) + bytesOf(frontier_below_, point);
    bytes += bytesOf(cont_cache_, [](const Continuation& C) noexcept {
        return bytesOf(C.event_vars) + bytesOf(C.event_kind) + bytesOf(C.event_sign) +
               bytesOf(C.beta_star) + bytesOf(C.dir) + bytesOf(C.c_exact) +
               bytesOf(C.cdot);
    });
    bytes += bytesOf(record_gap_) + bytesOf(record_kkt_);
    return bytes;
}


// ============================================================================
// Serialization Implementation
// ============================================================================

void TCDLASSO_Solver::save(const std::string& filename) const { saveImpl(*this, filename); }


TCDLASSO_Solver TCDLASSO_Solver::load(const std::string& filename,
                              Eigen::Map<Eigen::MatrixXd>& X,
                              Eigen::Map<Eigen::MatrixXd>& D) {
    return loadImpl<TCDLASSO_Solver>(filename, X, D);
}

// ==========================================================================
} /* End of namespace trex::tsolvers::ad::linear_model::cd_based */
