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

// utils includes
#include <utils/openmp/utils_openmp.hpp>

// =================================================================================

namespace trex::trex_selector_methods::ad::utils::data_normalizer {

// =================================================================================

/**
 * @brief Column scaling convention of the T-Rex working design.
 *
 * @details L2 and ZSCORE center every column of X and y, then scale the
 *  columns of X; the dummies follow the same convention. NONE is the
 *  user-managed design: X and y enter the selector exactly as supplied
 *  (no centering, no scaling; the stored parameters are the identity), so a
 *  caller can pre-center and pre-scale in whatever way the model demands,
 *  or use an intercept-free dictionary (steering vectors, Vandermonde,
 *  Fourier atoms) whose columns must not be centered. The contract under
 *  NONE is that the design is unit-L2-commensurate: the dummy blocks keep
 *  the unit-L2 standard (centered, ||d_j||_2 = 1), the lambda_2 tuners
 *  report on the unit-L2 scale, and the FDR calibration reference is the
 *  same one the L2 default uses. A design pre-normalized with
 *  centerAndL2NormalizeX / centerY and run under NONE therefore selects
 *  exactly what the raw design selects under L2 (test-enforced).
 */
enum class ScalingMode {
    /** @brief Divide each centered column by its L2 norm (||x_j||_2 = 1).
     *  Legacy / default behaviour. */
    L2,
    /** @brief Divide each centered column by its sample standard deviation
     *  (denominator n-1), i.e. z-scoring to unit variance. Matches the
     *  glmnet `standardize=TRUE` convention; a unit-SD column equals
     *  sqrt(n-1) times the corresponding unit-L2 column. */
    ZSCORE,
    /** @brief Leave X and y untouched (user-managed centering and scaling).
     *  The stored normalization parameters are the identity (means 0,
     *  divisors 1, y_mean 0), so denormalization is a no-op. Dummies keep
     *  the unit-L2 standard; the design is expected to be unit-L2-
     *  commensurate (a warning names columns whose L2 norm leaves [0.25, 4]).
     *  Non-finite entries are rejected in every mode; a column of L2 norm
     *  below eps (raw norm here, centered norm under L2 / ZSCORE) is
     *  rejected as dead. */
    NONE
};

// =================================================================================

/**
 * @brief Struct to hold normalization parameters for X and y.
 */
struct NormalizationParams {

    /** @brief Means of each column in X (all zero under ScalingMode::NONE). */
    Eigen::VectorXd X_means;

    /** @brief Per-column scale divisor actually applied (the L2 norm in
     *  ScalingMode::L2, the sample SD in ScalingMode::ZSCORE, 1.0 in
     *  ScalingMode::NONE). Used as-is by denormalizeX to invert the scaling
     *  regardless of mode. */
    Eigen::VectorXd X_l2norms;

    /** @brief Mean of y (0.0 under ScalingMode::NONE). */
    double y_mean = 0.0;

    /** @brief The mode the parameters were recorded under; denormalizeX and
     *  decenterY return immediately for ScalingMode::NONE. */
    ScalingMode mode = ScalingMode::L2;
};

// ===========================================================

/**
 * @brief Centers and L2-normalizes columns of X in-place (the user's design).
 *
 * @details A fused single-pass per column: mean → center → L2-norm → scale.
 *          Stores means and norms in params.X_means and params.X_l2norms
 *          and the mode in params.mode. Parallel across columns via OpenMP
 *          when p > 100. Under ScalingMode::NONE the matrix is left
 *          untouched and the identity parameters are stored.
 *
 *          Input validation (every mode): a non-finite entry throws
 *          std::invalid_argument naming the column; a dead column — L2 norm
 *          below eps, measured after centering under L2 / ZSCORE and on the
 *          raw column under NONE — throws std::invalid_argument naming the
 *          column (a constant predictor dies here, not silently inside a
 *          solver). Under NONE a column whose raw L2 norm
 *          leaves [0.25, 4] is counted and one warning is emitted (the
 *          dummies follow the unit-L2 standard, so the user-managed design
 *          should be unit-L2-commensurate). The matrix is unmodified when
 *          an exception is thrown.
 *
 * @param X Matrix to normalize (modified in-place).
 * @param params Struct to store normalization parameters (means and norms).
 * @param eps Dead-column threshold on the working L2 norm (default: machine
 *            epsilon; the selectors pass the solver tolerance).
 * @param verbose Unused for X (the rejection is an
 *                exception and the NONE warning is not gated); kept for
 *                source compatibility.
 * @param mode Column scaling convention (default: L2). ZSCORE divides by the
 *             sample SD (n-1) instead of the L2 norm; NONE leaves X as is.
 *
 * @throws std::invalid_argument on a non-finite entry or a dead column.
 */
void centerAndL2NormalizeX(Eigen::Ref<Eigen::MatrixXd> X,
                           NormalizationParams& params,
                           double eps = std::numeric_limits<double>::epsilon(),
                           bool verbose = false,
                           ScalingMode mode = ScalingMode::L2,
                           int n_threads = ::trex::utils::openmp::get_num_threads());

/**
 * @brief Center and L2-normalize an arbitrary matrix in-place (e.g., dummy block D).
 *
 * @details Same fused single-pass as centerAndL2NormalizeX but does not store
 *          parameters and does not reject: inside T-Rex it is used for dummy
 *          matrices (random draws, never user data), where a near-zero
 *          column keeps scale 1.0 with an optional warning. Under
 *          ScalingMode::NONE the matrix is left untouched (the
 *          DummyGenerator never passes NONE; it maps NONE to the unit-L2
 *          standard before calling).
 *
 * @param M Matrix to normalize (modified in-place).
 * @param eps Tolerance for zero-norm detection (default: machine epsilon).
 * @param verbose If true, warn about zero-norm columns.
 * @param mode Column scaling convention (default: L2). ZSCORE divides by the
 *             sample SD (n-1) instead of the L2 norm; NONE is a no-op.
 */
void centerAndL2NormalizeMatrix(Eigen::Ref<Eigen::MatrixXd> M,
                                double eps = std::numeric_limits<double>::epsilon(),
                                bool verbose = false,
                                ScalingMode mode = ScalingMode::L2,
                                int n_threads = ::trex::utils::openmp::get_num_threads());

/**
 * @brief Reverses column normalization: X = X_normalized * X_l2norms + X_means.
 *
 * @details Returns immediately when params.mode is ScalingMode::NONE (the
 *          parameters are the identity and X was never touched).
 *
 * @param X Matrix to denormalize (modified in-place).
 * @param params Normalization parameters (means and norms from a prior call to
 *               centerAndL2NormalizeX).
 */
void denormalizeX(Eigen::Ref<Eigen::MatrixXd> X,
                  const NormalizationParams& params,
                  int n_threads = ::trex::utils::openmp::get_num_threads());

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
 * @details Under ScalingMode::NONE y is left untouched and y_mean is 0.
 *          A non-finite entry throws std::invalid_argument in every mode
 *          (y unmodified).
 *
 * @param y Vector to center (modified in-place).
 * @param params In output struct y_mean is set.
 * @param mode Scaling convention of the design (default: L2); only NONE
 *             changes the behaviour.
 *
 * @throws std::invalid_argument on a non-finite entry.
 */
void centerY(Eigen::VectorXd& y, NormalizationParams& params,
             ScalingMode mode = ScalingMode::L2);

/**
 * @brief Reverse centering: y = y_centered + y_mean.
 *
 * @details Returns immediately when params.mode is ScalingMode::NONE.
 *
 * @param y Vector to decenter (modified in-place).
 * @param params Normalization parameters (y_mean from a prior call to centerY).
 */
void decenterY(Eigen::VectorXd& y, const NormalizationParams& params);


// =================================================================================
} /* End of namespace trex::trex_selector_methods::ad::utils::data_normalizer */

#endif /* End of TREX_TREX_SELECTOR_METHODS_UTILS_DATA_NORMALIZER_HPP */
