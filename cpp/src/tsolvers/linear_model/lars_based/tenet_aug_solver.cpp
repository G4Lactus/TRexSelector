// ============================================================================
// tenet_aug_solver.cpp
// ============================================================================
/**
 * @file tenet_aug_solver.cpp
 *
 * @brief Implementation of TENETAug_Solver: thin wrapper around TLASSO_Solver
 * on a data-augmented system encoding the EN ridge penalty.
 */
// ============================================================================

// std includes
#include <cmath>

// tsolvers includes
#include <tsolvers/linear_model/lars_based/tenet_aug_solver.hpp>

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

Eigen::MatrixXd TENETAug_Solver::buildXAug(
    const MapMatXd& X,
    std::size_t p,
    std::size_t L,
    double d1,
    double d2)
{
    const auto n     = static_cast<Eigen::Index>(X.rows());
    const auto p_idx = static_cast<Eigen::Index>(p);
    const auto L_idx = static_cast<Eigen::Index>(L);

    // d2 * [ X      ]   (n rows: data)
    //      [ d1·Ip  ]   (p rows: X ridge block)
    //      [ 0      ]   (L rows: D-ridge placeholder)
    //
    // The ridge diagonal entry is the FIXED constant d2*d1 (= d2*sqrt(lambda2)),
    // matching the R original (lm_dummy.R: diag(sqrt(lambda_2_lars))). The
    // subsequent scale() of the augmented matrix (applied by the inner solver,
    // see the constructor) re-normalises every column, so the data-to-ridge
    // balance is governed solely by the column mass of X (z-scored => n-1) vs
    // lambda2, exactly as in the reference implementation.
    Eigen::MatrixXd X_aug = Eigen::MatrixXd::Zero(n + p_idx + L_idx, p_idx);
    X_aug.topRows(n) = d2 * X;
    for (Eigen::Index j = 0; j < p_idx; ++j) {
        X_aug(n + j, j) = d2 * d1;
    }
    return X_aug;
}

// ----------------------------------------------------------------------------

Eigen::MatrixXd TENETAug_Solver::buildDAug(
    const MapMatXd& D,
    std::size_t p,
    std::size_t L,
    double d1,
    double d2)
{
    const auto n     = static_cast<Eigen::Index>(D.rows());
    const auto p_idx = static_cast<Eigen::Index>(p);
    const auto L_idx = static_cast<Eigen::Index>(L);

    // d2 * [ D      ]   (n rows: data)
    //      [ 0      ]   (p rows: X-ridge placeholder)
    //      [ d1·IL  ]   (L rows: D ridge block)
    //
    // Fixed ridge diagonal entry d2*d1 (see buildXAug), matching the R original.
    Eigen::MatrixXd D_aug = Eigen::MatrixXd::Zero(n + p_idx + L_idx, L_idx);
    D_aug.topRows(n) = d2 * D;
    for (Eigen::Index j = 0; j < L_idx; ++j) {
        D_aug(n + p_idx + j, j) = d2 * d1;
    }
    return D_aug;
}

// ----------------------------------------------------------------------------

Eigen::VectorXd TENETAug_Solver::buildYAug(
    const MapVecXd& y,
    std::size_t pL)
{
    const auto n      = static_cast<Eigen::Index>(y.size());
    const auto pL_idx = static_cast<Eigen::Index>(pL);
    Eigen::VectorXd y_aug(n + pL_idx);
    y_aug.head(n).array()     = y.array();
    y_aug.tail(pL_idx).setZero();
    return y_aug;
}

// ----------------------------------------------------------------------------

void TENETAug_Solver::buildMaps() {
    X_aug_map_ = std::make_unique<MapMatXd>(
        X_aug_owned_.data(), X_aug_owned_.rows(), X_aug_owned_.cols());
    D_aug_map_ = std::make_unique<MapMatXd>(
        D_aug_owned_.data(), D_aug_owned_.rows(), D_aug_owned_.cols());
    y_aug_map_ = std::make_unique<MapVecXd>(
        y_aug_owned_.data(), y_aug_owned_.size());
}

void TENETAug_Solver::buildInnerSolver(bool normalize, bool intercept, bool verbose,
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

TENETAug_Solver::TENETAug_Solver(
    MapMatXd& X,
    MapMatXd& D,
    MapVecXd& y,
    double lambda2,
    bool normalize,
    bool intercept,
    bool verbose,
    ScalingMode scaling_mode,
    bool use_lars_inner
) {
    lambda2_ = lambda2;
    use_lars_inner_ = use_lars_inner;
    const double d1 = std::sqrt(lambda2_);
    d2_ = 1.0 / std::sqrt(1.0 + lambda2_);

    const std::size_t p = static_cast<std::size_t>(X.cols());
    const std::size_t L = static_cast<std::size_t>(D.cols());

    // Inner-solver scaling configuration. Defaults to whatever the caller
    // requested (used only for the non-augmented lambda2 == 0 path, where the
    // external scaling of X/D must be respected).
    bool inner_normalize       = normalize;
    bool inner_intercept       = intercept;
    ScalingMode inner_scaling  = scaling_mode;

    if (lambda2_ == 0.0) {
        // lambda2 = 0: no ridge penalty.  TENETAug collapses to TLASSO on
        // the original data — no augmentation rows, d2_ = 1.0 (already set).
        // Using the augmented (n+p+L)-row system here would inflate
        // effective_n_ from n to n+p+L, giving wrong max_actives/loop limits.
        X_aug_owned_ = Eigen::MatrixXd(X);
        D_aug_owned_ = Eigen::MatrixXd(D);
        y_aug_owned_ = Eigen::VectorXd(y);
    } else {
        // lambda2 > 0: build the augmented system d2 * [X; d1·I], d2 * [D; d1·I].
        X_aug_owned_ = buildXAug(X, p, L, d1, d2_);
        D_aug_owned_ = buildDAug(D, p, L, d1, d2_);
        y_aug_owned_ = buildYAug(y, p + L);

        // Run the inner TLASSO on the augmented system EXACTLY as the reference
        // augmented-LASSO recipe (R Demo_06b_enet_comparison.R `lasso_star`):
        //     Xstar = d2 * rbind(X, d1·I);  ystar = c(y, 0)
        //     lars(Xstar, ystar, normalize = FALSE, intercept = FALSE)
        //     beta  = lasso_star$beta / d2
        // => normalize=false, intercept=false.  This was VERIFIED in
        // demo_ts_13_tenet_aug_comparison: a plain TLASSO on these augmented
        // matrices (normalize=false) matches the Gram-based TENET_Solver to
        // machine precision (max L2 < 1e-13) under BOTH L2 and z-score scaling,
        // whereas inner re-normalisation (normalize=true) makes the path diverge
        // from the reference. The augmented columns must be fed to the LASSO
        // as constructed; the caller is responsible for any external scaling of
        // X/D (and centering of y), exactly as in the R reference.
        inner_normalize = false;
        inner_intercept = false;
        inner_scaling   = scaling_mode;
    }

    buildMaps();
    buildInnerSolver(inner_normalize, inner_intercept, verbose, inner_scaling);
}

// ============================================================================
// Core interface delegation
// ============================================================================

void TENETAug_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    inner_->executeStep(T_stop, early_stop);
}

Eigen::MatrixXd TENETAug_Solver::getBetaPath() const {
    // Inner LASSO betas are in the augmented space (scaled by d2).
    // Divide by d2 (= multiply by 1/d2 = sqrt(1+lambda2)) to recover
    // original-scale coefficients, matching Demo_06b_enet_comparison.R:
    //   lasso_star_beta <- lasso_star$beta / d2
    return inner_->getBetaPath() / d2_;
}

Eigen::VectorXd TENETAug_Solver::getBeta(int step) const {
    return inner_->getBeta(step) / d2_;
}

// ============================================================================
} // namespace trex::tsolvers::linear_model::lars_based
// ============================================================================
