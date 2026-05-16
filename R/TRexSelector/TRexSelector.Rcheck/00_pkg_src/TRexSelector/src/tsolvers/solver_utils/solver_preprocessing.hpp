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
 * @brief Centers and/or L2-normalizes a data matrix in-place.
 *
 * @param X Matrix to transform (in-place).
 * @param center If true, subtracts column means.
 * @param normalize If true, scales columns to unit L2-norm.
 * @param eps Tolerance threshold for dropping zero-variance columns.
 *
 * @param[out] meansx Populated with column means.
 * @param[out] normsx Populated with column norms.
 * @param[in,out] dropped_indices Appended with indices of zero-variance columns.
 */
void preprocessMatStatistics(
    Eigen::Ref<Eigen::MatrixXd> X,
    bool center,
    bool normalize,
    double eps,
    Eigen::VectorXd& meansx,
    Eigen::VectorXd& normsx,
    std::vector<std::size_t>& dropped_indices
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
