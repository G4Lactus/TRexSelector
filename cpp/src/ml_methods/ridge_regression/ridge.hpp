// ===================================================================================
// ridge.hpp
// ===================================================================================
#ifndef TREX_ML_METHODS_RIDGE_HPP
#define TREX_ML_METHODS_RIDGE_HPP
// ===================================================================================
/**
 * @file ridge.hpp
 *
 * @brief Standalone Ridge Regression solver optimized for single-lambda solves.
 *        Single-λ Cholesky solver. For multi-λ cross-validated model selection, see
 *        model_selection/ridge_cv_svd.hpp and model_selection/enet_cv_ccd.hpp.
 */
// ===================================================================================

// std includes
#include <stdexcept>

// Eigen includes
#include <Eigen/Dense>

// utils includes
#include <utils/openmp/utils_eigen_threads.hpp>

// ===================================================================================

// Namespace integration
namespace trex {
namespace ml_methods {
namespace ridge {

// ===================================================================================

/**
 * @brief Single-lambda ridge regression solver via a Cholesky factorization.
 *
 *  References:
 *  -----------
 *   A. E. Hoerl and R. W. Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems,"
 *     Technometrics, vol. 42, no. 1, pp. 80-86, 2000 (Reprint of the 1970 article in the Special 40th
 *     Anniversary Issue).
 */
class RidgeSolver {
public:
    /** @brief Default constructor */
    RidgeSolver() = default;

    /** @brief Default destructor */
    ~RidgeSolver() = default;

    /**
     * @brief Solves the Ridge Regression problem:
     *        min ||y - X*beta||_2^2 + lambda*||beta||_2^2
     *        for a single lambda value.
     *
     * @param X The input data matrix (n_samples x n_features).
     * @param y The target vector (n_samples).
     * @param lambda The regularization parameter.
     *
     * @return The coefficient vector (n_features).
     */
    static Eigen::VectorXd solve(const Eigen::Ref<const Eigen::MatrixXd>& X,
                                 const Eigen::Ref<const Eigen::VectorXd>& y,
                                 double lambda,
                                 int n_threads = ::trex::utils::openmp::get_num_threads()) {

        if (n_threads < 1) {
            throw std::invalid_argument("RidgeSolver::solve: n_threads must be >= 1.");
        }
        // Eigen's products run on the budget for the scope of the solve
        ::trex::utils::openmp::EigenThreadBudgetGuard eigen_thread_budget_guard(n_threads);

        Eigen::Index n = X.rows();
        Eigen::Index p = X.cols();

        if (n != y.size()) {
            throw std::invalid_argument("RidgeSolver::solve: X.rows() must match y.size().");
        }
        if (lambda < 0.0) {
            throw std::invalid_argument("RidgeSolver::solve: lambda must be >= 0.");
        }

        // Dispatch based on dimensionality
        if (n >= p) {
            return solve_primal(X, y, lambda);
        } else {
            return solve_dual(X, y, lambda);
        }
    }

private:

    /**
     * @brief Solves the Ridge Regression problem in the primal form.
     *
     * @details Uses LLT (Cholesky) as the fast primary path since X^T X + λI is
     *          positive-definite for any λ > 0. Falls back to LDLT if LLT reports
     *          a numerical issue (e.g. near-perfectly correlated columns with a very
     *          small λ).
     *
     * @param X      The input data matrix (n_samples x n_features).
     * @param y      The target vector (n_samples).
     * @param lambda The regularization parameter.
     *
     * @return The coefficient vector (n_features).
     */
    static Eigen::VectorXd solve_primal(const Eigen::Ref<const Eigen::MatrixXd>& X,
                                        const Eigen::Ref<const Eigen::VectorXd>& y,
                                        double lambda) {
        Eigen::Index p = X.cols();

        // Normal equation: (X^T X + lambda I) beta = X^T y. The Gram matrix
        // is accumulated from column blocks of the (possibly memory-mapped)
        // view; X itself is never copied.
        Eigen::MatrixXd XtX = Eigen::MatrixXd::Zero(p, p);
        XtX.selfadjointView<Eigen::Lower>().rankUpdate(X.transpose());

        // Add Ridge penalty to the diagonal
        XtX.diagonal().array() += lambda;

        Eigen::VectorXd Xty = X.transpose() * y;

        // Fast Cholesky decomposition (X^T X + λI is SPD for λ > 0), IN PLACE
        // in XtX's storage: one p x p matrix instead of two (same algorithm,
        // bit-identical).
        Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(XtX);
        if (llt.info() == Eigen::Success) {
            return llt.solve(Xty);
        }

        // Fallback: LDLT handles the near-singular case. The failed in-place
        // LLT has overwritten the lower triangle, so the Gram matrix is
        // rebuilt (rare path; a second pass over X, no copy of it).
        XtX.setZero();
        XtX.selfadjointView<Eigen::Lower>().rankUpdate(X.transpose());
        XtX.diagonal().array() += lambda;
        Eigen::LDLT<Eigen::Ref<Eigen::MatrixXd>> ldlt(XtX);
        if (ldlt.info() == Eigen::Success) {
            return ldlt.solve(Xty);
        }

        throw std::runtime_error(
            "RidgeSolver::solve_primal: Both LLT and LDLT factorizations failed."
        );
    }


    /**
     * @brief Solves the Ridge Regression problem in the dual form, which can be more efficient
     *        when n < p.
     *
     * @details The fallback from LLT -> LDLT addess the lambda_2 trap. In the case that active
     *          subset contains perfectly correlated variables and lambda2 = 10^{-6} does not
     *          guarantee positive-definitness, the standard LLT will encounter an
     *          Eigen::NumericalIssue
     *
     * @param X The input data matrix (n_samples x n_features).
     * @param y The target vector (n_samples).
     * @param lambda The regularization parameter.
     *
     * @return The coefficient vector (n_features).
     */
    static Eigen::VectorXd solve_dual(const Eigen::Ref<const Eigen::MatrixXd>& X,
                                      const Eigen::Ref<const Eigen::VectorXd>& y,
                                      double lambda) {
        Eigen::Index n = X.rows();

        // Form the Gram matrix
        Eigen::MatrixXd XXt = Eigen::MatrixXd::Zero(n, n);
        XXt.selfadjointView<Eigen::Lower>().rankUpdate(X);

        // Add Ridge penalty to the diagonal
        XXt.diagonal().array() += lambda;

        Eigen::VectorXd alpha;

        // Fast Cholesky decomposition, IN PLACE in XXt's storage (one n x n
        // matrix instead of two; bit-identical)
        Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(XXt);
        if (llt.info() == Eigen::Success) {
            alpha = llt.solve(y);
        } else {
            // Fall back to LDLT if LLT fails: the failed in-place LLT has
            // overwritten the lower triangle, so the kernel is rebuilt (a
            // second pass over X, no copy of it).
            XXt.setZero();
            XXt.selfadjointView<Eigen::Lower>().rankUpdate(X);
            XXt.diagonal().array() += lambda;
            Eigen::LDLT<Eigen::Ref<Eigen::MatrixXd>> ldlt(XXt);
            if (ldlt.info() == Eigen::Success) {
                alpha = ldlt.solve(y);
            } else {
                throw std::runtime_error(
                    "RidgeSolver::solve_dual: Both LLT and LDLT factorizations failed."
                );
            }
        }

        // Recover primal coefficients: beta = X^T alpha
        return X.transpose() * alpha;
    }

};


// ===================================================================================
} /* namespace ridge */
} /* namespace ml_methods */
} /* namespace trex */
// ===================================================================================
#endif /* TREX_ML_METHODS_RIDGE_HPP */
