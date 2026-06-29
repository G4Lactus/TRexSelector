// =============================================================================
// solver_preprocessing.hpp
// =============================================================================
#ifndef TSOLVERS_SOLVER_UTILS_SOLVER_PREPROCESSING_HPP
#define TSOLVERS_SOLVER_UTILS_SOLVER_PREPROCESSING_HPP
// =============================================================================
/**
 * @file solver_preprocessing.hpp
 *
 * @brief Stateless, reusable preprocessing utilities for solvers, including
 * centering, l2-normalization, for unified data standardization across all
 * solvers.
 */
 // ============================================================================

// std includes
#include <cstddef>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// =============================================================================

// Embedded into trex::tsolvers::solver_utils::preprocessing namespace
namespace trex::tsolvers::solver_utils::preprocessing {

// =============================================================================

/**
 * @brief Column-scaling convention applied during normalization.
 *
 * @details
 *  - `L2`     : each (centered) column is divided by its L2-norm  ->  unit-L2.
 *  - `ZSCORE` : each (centered) column is divided by its sample standard
 *               deviation (= L2-norm / sqrt(n - 1))  ->  unit variance.
 *
 *  Because the divisor stored in `normsx` is the value actually used, the
 *  restore and beta de-normalization paths are identical for both modes.
 */
enum class ScalingMode {
    L2,      /* Divide centered columns by their L2-norm (unit-L2). */
    ZSCORE   /* Divide centered columns by their sample SD (unit variance). */
};

// =============================================================================

/**
 * @brief Centers and/or normalizes a data matrix in-place.
 *
 * @param X Matrix to transform (in-place).
 * @param center If true, subtracts column means.
 * @param normalize If true, scales columns (see @p mode).
 * @param eps Tolerance threshold for dropping zero-variance columns.
 *
 * @param[out] meansx Populated with column means.
 * @param[out] normsx Populated with the divisor actually applied per column
 *             (L2-norm for ScalingMode::L2, sample SD for ScalingMode::ZSCORE).
 * @param[in,out] dropped_indices Appended with indices of zero-variance columns.
 * @param mode Column-scaling convention (default: ScalingMode::L2).
 */
void preprocessMatStatistics(
    Eigen::Ref<Eigen::MatrixXd> X,
    bool center,
    bool normalize,
    double eps,
    Eigen::VectorXd& meansx,
    Eigen::VectorXd& normsx,
    std::vector<std::size_t>& dropped_indices,
    ScalingMode mode = ScalingMode::L2
);


/**
 * @brief Reverses the centering and normalization of a data matrix in-place.
 *
 * @param X Matrix to restore (in-place).
 * @param centered If true, adds back column means.
 * @param normalized If true, multiplies back column norms.
 * @param meansx Vector of column means to add back.
 * @param normsx Vector of column norms to multiply back.
 */
void restoreMatStatistics(
    Eigen::Ref<Eigen::MatrixXd> X,
    bool centered,
    const Eigen::VectorXd& meansx,
    bool normalized,
    const Eigen::VectorXd& normsx
);


/**
 * @brief Centers a continuous response vector y.
 *
 * @param y Response vector to center (in-place).
 * @param mu_y Mean of the response vector to subtract.
 */
void centerResponse(Eigen::Ref<Eigen::VectorXd> y, double& mu_y);


/**
 * @brief Restores a centered continuous response vector y.
 *
 * @param y Response vector to restore (in-place).
 * @param mu_y Mean of the response vector to add back.
 */
void restoreResponse(Eigen::Ref<Eigen::VectorXd> y, double mu_y);

// ============================================================================

} // namespace trex::tsolvers::solver_utils::preprocessing

#endif /* TSOLVERS_SOLVER_UTILS_SOLVER_PREPROCESSING_HPP */
