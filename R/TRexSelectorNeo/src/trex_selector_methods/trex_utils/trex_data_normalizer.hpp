// =================================================================================
// trex_data_normalizer.hpp
// =================================================================================
#ifndef TREX_TREX_SELECTOR_METHODS_UTILS_DATA_NORMALIZER_HPP
#define TREX_TREX_SELECTOR_METHODS_UTILS_DATA_NORMALIZER_HPP
// =================================================================================
/**
 * @file trex_data_normalizer.hpp
 *
 * @brief External function interface for data normalization in T-Rex Selector.
 * Centers columns of X and L2-normalizes them.
 */
// =================================================================================

// std includes
#include <limits>

// Eigen includes
#include <Eigen/Dense>

// =================================================================================

namespace trex::trex_selector_methods::utils::data_normalizer {

// =================================================================================

/**
 * @brief Column scaling convention applied after centering.
 */
enum class ScalingMode {
    /** @brief Divide each centered column by its L2 norm (||x_j||_2 = 1).
     *  Legacy / default behaviour. */
    L2,
    /** @brief Divide each centered column by its sample standard deviation
     *  (denominator n-1), i.e. z-scoring to unit variance. Matches the
     *  glmnet `standardize=TRUE` convention; a unit-SD column equals
     *  sqrt(n-1) times the corresponding unit-L2 column. */
    ZSCORE
};

// =================================================================================

/**
 * @brief Struct to hold normalization parameters for X and y.
 */
struct NormalizationParams {

    /** @brief Means of each column in X. */
    Eigen::VectorXd X_means;

    /** @brief Per-column scale divisor actually applied (the L2 norm in
     *  ScalingMode::L2, the sample SD in ScalingMode::ZSCORE). Used as-is by
     *  denormalizeX to invert the scaling regardless of mode. */
    Eigen::VectorXd X_l2norms;

    /** @brief Mean of y. */
    double y_mean = 0.0;
};

// ===========================================================

/**
 * @brief Centers and L2-normalizes columns of X in-place.
 *
 * @details A fused single-pass per column: mean → center → L2-norm → scale.
 *          Stores means and norms in params.X_means and params.X_l2norms.
 *          Parallel across columns via OpenMP when p > 100.
 *          Columns with near-zero L2 norm get norm set to 1.0 (no scaling).
 *
 * @param X Matrix to normalize (modified in-place).
 * @param params Struct to store normalization parameters (means and norms).
 * @param eps Tolerance for zero-norm detection (default: machine epsilon).
 * @param verbose If true, warn about zero-norm columns to stdout.
 * @param mode Column scaling convention (default: L2). ZSCORE divides by the
 *             sample SD (n-1) instead of the L2 norm.
 */
void centerAndL2NormalizeX(Eigen::Ref<Eigen::MatrixXd> X,
                           NormalizationParams& params,
                           double eps = std::numeric_limits<double>::epsilon(),
                           bool verbose = false,
                           ScalingMode mode = ScalingMode::L2);

/**
 * @brief Center and L2-normalize an arbitrary matrix in-place (e.g., dummy block D).
 *
 * @details Same fused single-pass as centerAndL2NormalizeX but does not store parameters.
 *          Inside T-Rex it is used for dummy matrices whose parameters are not needed later.
 *
 * @param M Matrix to normalize (modified in-place).
 * @param eps Tolerance for zero-norm detection (default: machine epsilon).
 * @param verbose If true, warn about zero-norm columns to stdout.
 * @param mode Column scaling convention (default: L2). ZSCORE divides by the
 *             sample SD (n-1) instead of the L2 norm.
 */
void centerAndL2NormalizeMatrix(Eigen::Ref<Eigen::MatrixXd> M,
                                double eps = std::numeric_limits<double>::epsilon(),
                                bool verbose = false,
                                ScalingMode mode = ScalingMode::L2);

/**
 * @brief Reverses column normalization: X = X_normalized * X_l2norms + X_means.
 *
 * @param X Matrix to denormalize (modified in-place).
 * @param params Normalization parameters (means and norms from a prior call to
 *               centerAndL2NormalizeX).
 */
void denormalizeX(Eigen::Ref<Eigen::MatrixXd> X,
                  const NormalizationParams& params);

/**
 * @brief Reverses column normalization of an arbitrary matrix:
 *        M = M_normalized * l2norms + means.
 *
 * @param M      Matrix to denormalize (modified in-place).
 * @param params Normalization parameters from a prior call to
 *               centerAndL2NormalizeMatrix (with params).
 */
void denormalizeMatrix(Eigen::Ref<Eigen::MatrixXd> M,
                        const NormalizationParams& params);

/**
 * @brief Centers y; stores y_mean in params.
 *
 * @param y Vector to center (modified in-place).
 * @param params In output struct y_mean is set.
 */
void centerY(Eigen::VectorXd& y, NormalizationParams& params);

/**
 * @brief Reverse centering: y = y_centered + y_mean.
 *
 * @param y Vector to decenter (modified in-place).
 * @param params Normalization parameters (y_mean from a prior call to centerY).
 */
void decenterY(Eigen::VectorXd& y, const NormalizationParams& params);


// =================================================================================
} /* End of namespace trex::trex_selector_methods::utils::data_normalizer */

#endif /* End of TREX_TREX_SELECTOR_METHODS_UTILS_DATA_NORMALIZER_HPP */
