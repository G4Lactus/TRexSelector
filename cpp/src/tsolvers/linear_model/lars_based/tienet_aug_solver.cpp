// ============================================================================
// tienet_aug_solver.cpp
// ============================================================================
/**
 * @file tienet_aug_solver.cpp
 *
 * @brief Implementation of TIENETAug_Solver: thin wrapper around
 * TLASSO_Solver / TLARS_Solver on a group-informed row-augmented system
 * encoding the Informed Elastic Net (IEN) penalty.
 */
// ============================================================================

// std includes
#include <cmath>
#include <fstream>
#include <stdexcept>

// tsolvers includes
#include <tsolvers/linear_model/lars_based/tienet_aug_solver.hpp>

// ============================================================================

namespace trex::tsolvers::linear_model::lars_based {

// ============================================================================
// Alias
// ============================================================================
namespace {
using MapMatXd = Eigen::Map<Eigen::MatrixXd>;
using MapVecXd = Eigen::Map<Eigen::VectorXd>;
} // anonymous namespace

// ============================================================================
// Static helpers
// ============================================================================

Eigen::MatrixXd TIENETAug_Solver::buildGroupBlock(
    const Eigen::VectorXi& groups,
    std::size_t M,
    double lambda2)
{
    const auto p     = groups.size();
    const auto M_idx = static_cast<Eigen::Index>(M);
    const double sqrt_l2 = std::sqrt(lambda2);

    // Group sizes p_m.
    Eigen::VectorXd sizes = Eigen::VectorXd::Zero(M_idx);
    for (Eigen::Index j = 0; j < p; ++j) {
        sizes(groups(j)) += 1.0;
    }

    // B(m, j) = sqrt(lambda2) / sqrt(p_m) on group-m columns
    // (Theorem 1, Machkour et al., CAMSAP 2023; R lm_dummy.R:
    //  sqrt(l2) * (1/sqrt(cluster_sizes)) * IEN_cl_id_vectors).
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(M_idx, p);
    for (Eigen::Index j = 0; j < p; ++j) {
        const Eigen::Index m = groups(j);
        B(m, j) = sqrt_l2 / std::sqrt(sizes(m));
    }
    return B;
}

// ----------------------------------------------------------------------------

void TIENETAug_Solver::centerAndScaleColumns(Eigen::MatrixXd& A, double eps,
                                             ScalingMode scaling_mode)
{
    const Eigen::Index n_rows = A.rows();
    if (n_rows == 0) { return; }

    // z-score divides by the sample SD = ||x_c|| / sqrt(rows - 1); fall back
    // to plain L2 if rows < 2. Mirrors the shared data-normalizer convention.
    const double sd_factor =
        (scaling_mode == ScalingMode::ZSCORE && n_rows > 1)
            ? std::sqrt(static_cast<double>(n_rows - 1))
            : 1.0;

    for (Eigen::Index j = 0; j < A.cols(); ++j) {
        A.col(j).array() -= A.col(j).mean();
        double scale_j = A.col(j).norm() / sd_factor;
        // Degenerate (near-constant) column: keep scale 1.0 instead of
        // throwing, consistent with centerAndL2NormalizeX / -Matrix.
        if (scale_j < eps) { scale_j = 1.0; }
        A.col(j) /= scale_j;
    }
}

// ----------------------------------------------------------------------------

void TIENETAug_Solver::buildMaps() {
    X_aug_map_ = std::make_unique<MapMatXd>(
        X_aug_owned_.data(), X_aug_owned_.rows(), X_aug_owned_.cols());
    D_aug_map_ = std::make_unique<MapMatXd>(
        D_aug_owned_.data(), D_aug_owned_.rows(), D_aug_owned_.cols());
    y_aug_map_ = std::make_unique<MapVecXd>(
        y_aug_owned_.data(), y_aug_owned_.size());
}

void TIENETAug_Solver::buildInnerSolver(bool normalize, bool intercept, bool verbose,
                                        ScalingMode scaling_mode) {
    if (use_lars_inner_) {
        // Pure-LARS inner solver: never drops variables (matches R type="lar").
        inner_ = std::make_unique<TLARS_Solver>(
            *X_aug_map_, *D_aug_map_, *y_aug_map_,
            normalize, intercept, verbose,
            SolverTypeLarsBased::TLARS, scaling_mode);
    } else {
        // Default LARS-LASSO inner solver: can drop variables on sign change.
        inner_ = std::make_unique<TLASSO_Solver>(
            *X_aug_map_, *D_aug_map_, *y_aug_map_,
            normalize, intercept, verbose, scaling_mode);
    }
}

// ============================================================================
// Constructor
// ============================================================================

TIENETAug_Solver::TIENETAug_Solver(
    MapMatXd& X,
    MapMatXd& D,
    MapVecXd& y,
    double lambda2,
    const Eigen::VectorXi& groups,
    bool normalize,
    bool intercept,
    bool verbose,
    ScalingMode scaling_mode,
    bool use_lars_inner
) {
    if (lambda2 < 0.0) {
        throw std::invalid_argument("TIENETAug_Solver: lambda2 must be non-negative.");
    }

    const std::size_t p = static_cast<std::size_t>(X.cols());
    const std::size_t L = static_cast<std::size_t>(D.cols());

    // ---- Validate the group structure --------------------------------------
    if (static_cast<std::size_t>(groups.size()) != p) {
        throw std::invalid_argument(
            "TIENETAug_Solver: groups length must equal the number of "
            "predictors p.");
    }
    Eigen::Index max_id = -1;
    for (Eigen::Index j = 0; j < groups.size(); ++j) {
        if (groups(j) < 0) {
            throw std::invalid_argument(
                "TIENETAug_Solver: groups must be 0-based non-negative ids.");
        }
        if (groups(j) > max_id) { max_id = groups(j); }
    }
    const std::size_t M = static_cast<std::size_t>(max_id + 1);
    {
        std::vector<bool> seen(M, false);
        for (Eigen::Index j = 0; j < groups.size(); ++j) {
            seen[static_cast<std::size_t>(groups(j))] = true;
        }
        for (std::size_t m = 0; m < M; ++m) {
            if (!seen[m]) {
                throw std::invalid_argument(
                    "TIENETAug_Solver: group ids are non-contiguous (id " +
                    std::to_string(m) + " is missing).");
            }
        }
    }

    lambda2_        = lambda2;
    groups_         = groups;
    M_              = M;
    use_lars_inner_ = use_lars_inner;

    // Populate the inherited static metadata so the base getters and the
    // synced diagnostics operate in the original (p + L) index space.
    p_original_      = p;
    num_dummies_     = L;
    dummy_start_idx_ = p;
    normalize_       = normalize;
    intercept_       = intercept;
    verbose_         = verbose;
    effective_n_     = static_cast<std::size_t>(X.rows()) - (intercept ? 1 : 0);

    // Inner-solver preprocessing configuration. Defaults to the caller's
    // request (used only for the non-augmented lambda2 == 0 path, where the
    // external scaling of X/D must be respected).
    bool inner_normalize = normalize;
    bool inner_intercept = intercept;

    if (lambda2_ == 0.0) {
        // lambda2 = 0: the group penalty vanishes. TIENETAug collapses to a
        // plain TLASSO / TLARS on the original data — no augmentation rows
        // (mirror of TENETAug_Solver's degenerate path; using the (n+M)-row
        // system here would inflate effective_n_).
        X_aug_owned_ = Eigen::MatrixXd(X);
        D_aug_owned_ = Eigen::MatrixXd(D);
        y_aug_owned_ = Eigen::VectorXd(y);
    } else {
        // The T-Rex+GVS dummy convention: D consists of L/p layers of p
        // columns, dummy column j of each layer sharing variable j's group,
        // so the group-ridge block B tiles once per layer.
        if (L % p != 0) {
            throw std::invalid_argument(
                "TIENETAug_Solver: D.cols() must be a multiple of X.cols() "
                "(one group-aligned dummy layer per p columns).");
        }
        const std::size_t w = L / p;  // number of dummy layers

        const auto n     = static_cast<Eigen::Index>(X.rows());
        const auto p_idx = static_cast<Eigen::Index>(p);
        const auto L_idx = static_cast<Eigen::Index>(L);
        const auto M_idx = static_cast<Eigen::Index>(M);

        const Eigen::MatrixXd B = buildGroupBlock(groups_, M_, lambda2_);

        // X_aug = [ X ; B ]  ((n + M) x p). No global 1/sqrt(1+lambda2)
        // factor for the IEN (Theorem 1; R lm_dummy.R: the sqrt(l2) and
        // 1/sqrt(l2) factors cancel on the data block).
        X_aug_owned_.resize(n + M_idx, p_idx);
        X_aug_owned_.topRows(n)        = X;
        X_aug_owned_.bottomRows(M_idx) = B;

        // D_aug = [ D ; B tiled per layer ]  ((n + M) x L).
        D_aug_owned_.resize(n + M_idx, L_idx);
        D_aug_owned_.topRows(n) = D;
        for (std::size_t li = 0; li < w; ++li) {
            D_aug_owned_.block(n, static_cast<Eigen::Index>(li) * p_idx,
                               M_idx, p_idx) = B;
        }

        // y_aug = [ y ; 0_M ].
        y_aug_owned_.resize(n + M_idx);
        y_aug_owned_.head(n) = y;
        y_aug_owned_.tail(M_idx).setZero();

        // R-faithful post-augmentation preprocessing (lm_dummy.R:
        // `X_Dummy <- scale(X_Dummy); y <- y - mean(y)`): center + rescale the
        // augmented columns, center the augmented response, then run the
        // inner solver on the system AS PREPROCESSED.
        centerAndScaleColumns(X_aug_owned_, eps_, scaling_mode);
        centerAndScaleColumns(D_aug_owned_, eps_, scaling_mode);
        y_aug_owned_.array() -= y_aug_owned_.mean();

        inner_normalize = false;
        inner_intercept = false;
    }

    buildMaps();
    buildInnerSolver(inner_normalize, inner_intercept, verbose, scaling_mode);
    syncDiagnosticsFromInner();
}

// ============================================================================
// Core interface delegation
// ============================================================================

void TIENETAug_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    inner_->executeStep(T_stop, early_stop);
    syncDiagnosticsFromInner();
}


void TIENETAug_Solver::syncDiagnosticsFromInner() {
    if (!inner_) { return; }

    RSS_             = inner_->getRSS();
    R2_              = inner_->getR2();
    DoF_             = inner_->getDoF();
    actions_         = inner_->getActions();
    actives_         = inner_->getActives();
    inactives_       = inner_->getInactives();
    dropped_indices_ = inner_->getDroppedIndices();
    r_               = inner_->getResiduals();
    currentStep_     = inner_->getNumSteps();
    count_active_dummies_ = inner_->getActiveDummyIndices().size();
    is_connected_    = inner_->isConnected();
    warnings_        = inner_->getWarnings();
}


double TIENETAug_Solver::getIntercept(int step) const {
    return inner_->getIntercept(step);
}

Eigen::MatrixXd TIENETAug_Solver::getBetaPath() const {
    // Coefficients on the normalized augmented-column scale (no global
    // rescaling factor exists for the IEN; see the class documentation).
    return inner_->getBetaPath();
}


SparseBetaPath TIENETAug_Solver::getBetaPathSparse() const {
    return inner_->getBetaPathSparse();
}

Eigen::VectorXd TIENETAug_Solver::getBeta(int step) const {
    return inner_->getBeta(step);
}

// ============================================================================
// De-/Serialization
// ============================================================================

void TIENETAug_Solver::save(const std::string& filename) const { saveImpl(*this, filename); }


TIENETAug_Solver TIENETAug_Solver::load(
    const std::string& filename,
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D
) {
    TIENETAug_Solver solver;
    std::ifstream is(filename, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error(
            solver.concatMsg(solver.solverTypeToString(),
                             "::load: Cannot open '", filename, "'")
        );
    }
    cereal::PortableBinaryInputArchive iarchive(is);
    iarchive(solver);

    // The wrapper owns its augmented buffers; the passed maps are only
    // validated against the serialized dimensions (API parity with siblings).
    if (static_cast<std::size_t>(X.cols()) != solver.p_original_ ||
        static_cast<std::size_t>(D.cols()) != solver.num_dummies_) {
        throw std::runtime_error(
            solver.concatMsg(solver.solverTypeToString(),
                             "::load: X/D dimensions do not match serialized state.")
        );
    }
    return solver;
}

// ============================================================================
} // namespace trex::tsolvers::linear_model::lars_based
// ============================================================================
