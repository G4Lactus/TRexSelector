// =============================================================================
// tikhonov_validation.hpp
// =============================================================================
#ifndef TSOLVERS_SOLVER_UTILS_TIKHONOV_VALIDATION_HPP
#define TSOLVERS_SOLVER_UTILS_TIKHONOV_VALIDATION_HPP
// =============================================================================
/**
 * @file tikhonov_validation.hpp
 *
 * @brief Content validation of a user-supplied sparse Tikhonov penalty
 * matrix K = Gamma^T Gamma, shared by the sparse-K solvers
 * (TIKNET / TCDTIKNET).
 */
 // ============================================================================

// std includes
#include <cmath>
#include <stdexcept>
#include <string>

// Eigen includes
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

// utils includes
#include <utils/fp_classify/fp_classify.hpp>

// =============================================================================

// Embedded into trex::tsolvers::ad::solver_utils::tikhonov namespace
namespace trex::tsolvers::ad::solver_utils::tikhonov {

// =============================================================================

/**
 * @brief Validates the content of a sparse Tikhonov penalty matrix K:
 * every stored entry finite, K symmetric, and K positive semi-definite,
 * each within a relative tolerance. The caller has already verified the
 * p x p dimensions (solver-specific message).
 *
 * @details The solvers' Cholesky/coordinate updates evaluate sqrt(d_jj)
 * and sqrt(delta_j) on K-derived pivots and assume K = K^T when folding
 * columns, so an asymmetric or indefinite K produces silent NaNs on the
 * hot path — this guard is the only one. With s = max(1, max_ij |K_ij|):
 *
 *  - symmetry: |K_ij - K_ji| <= tol * s entrywise;
 *  - PSD: SimplicialLDLT of K + (tol * s) I must factorize with
 *    min diag(D) >= -(tol * s). The diagonal shift keeps exactly singular
 *    PSD penalties (group-centering projectors, group-mean K; eigenvalue
 *    0) clear of the zero-pivot boundary.
 *
 * @param K Sparse penalty matrix to validate (square, compressed or not).
 * @param who Solver name used as the exception-message prefix.
 * @param tol Relative tolerance for the symmetry and PSD checks.
 *
 * @throws std::invalid_argument on the first violated property.
 */
inline void validateTikhonovK(const Eigen::SparseMatrix<double>& K,
                              const std::string& who,
                              double tol = 1e-10) {
    namespace fpc = trex::utils::fp_classify;

    // ---- Finiteness + magnitude scale (one sweep over the nonzeros) --------
    double maxabs = 0.0;
    for (int j = 0; j < K.outerSize(); ++j) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(K, j); it; ++it) {
            if (!fpc::isfinite(it.value())) {
                throw std::invalid_argument(who +
                    ": Tikhonov matrix has a non-finite entry at (" +
                    std::to_string(it.row()) + ", " +
                    std::to_string(it.col()) + ").");
            }
            maxabs = std::max(maxabs, std::abs(it.value()));
        }
    }
    const double scale = std::max(1.0, maxabs);

    // ---- Symmetry: K == K^T entrywise within tol * scale -------------------
    Eigen::SparseMatrix<double> Kt = K.transpose();  // realize to ColMajor
    Eigen::SparseMatrix<double> diff = K - Kt;
    for (int j = 0; j < diff.outerSize(); ++j) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(diff, j); it; ++it) {
            if (std::abs(it.value()) > tol * scale) {
                throw std::invalid_argument(who +
                    ": Tikhonov matrix is not symmetric within tolerance "
                    "at (" + std::to_string(it.row()) + ", " +
                    std::to_string(it.col()) + ").");
            }
        }
    }

    // ---- PSD: LDLT pivots of the tol-shifted K must clear -tol * scale -----
    const double shift = tol * scale;
    Eigen::SparseMatrix<double> eye(K.rows(), K.cols());
    eye.setIdentity();
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(K + shift * eye);
    if (ldlt.info() != Eigen::Success ||
        (ldlt.vectorD().size() > 0 && ldlt.vectorD().minCoeff() < -shift)) {
        throw std::invalid_argument(who +
            ": Tikhonov matrix is not positive semi-definite within "
            "tolerance (LDLT pivot below the -tol bound).");
    }
}

// =============================================================================

}  // namespace trex::tsolvers::ad::solver_utils::tikhonov

// =============================================================================
#endif  // TSOLVERS_SOLVER_UTILS_TIKHONOV_VALIDATION_HPP
// =============================================================================
