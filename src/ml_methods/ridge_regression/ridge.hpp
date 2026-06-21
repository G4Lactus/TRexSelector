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
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <stdexcept>

// Eigen includes
#include <Eigen/Dense>

// ===================================================================================

// Namespace integration
namespace trex {
namespace ml_methods {
namespace ridge {

// ===================================================================================

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
                          double lambda) {
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

        // Normal equation: (X^T X + lambda I) beta = X^T y
        Eigen::MatrixXd XtX = Eigen::MatrixXd(p, p);
        XtX.selfadjointView<Eigen::Lower>().rankUpdate(X.transpose());

        // Add Ridge penalty to the diagonal
        XtX.diagonal().array() += lambda;

        Eigen::VectorXd Xty = X.transpose() * y;

        // Fast Cholesky decomposition (X^T X + λI is SPD for λ > 0)
        Eigen::LLT<Eigen::MatrixXd> llt(XtX);
        if (llt.info() == Eigen::Success) {
            return llt.solve(Xty);
        }

        // Fallback: LDLT handles the near-singular case
        Eigen::LDLT<Eigen::MatrixXd> ldlt(XtX);
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

        // Fast Cholesky decomposition
        Eigen::LLT<Eigen::MatrixXd> llt(XXt);
        if (llt.info() == Eigen::Success) {
            alpha = llt.solve(y);
        } else {
            // Fall back to LDLT if LLT fails
            Eigen::LDLT<Eigen::MatrixXd> ldlt(XXt);
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
