/// @file norm_corrective_gmp.hpp
/// @brief Norm-Corrective Generalized Matching Pursuit (Algorithm 4,
///        Locatello et al., AISTATS 2017).
///
/// Header-only C++20 implementation using Eigen.
///
/// The algorithm solves:
///     min_{x in lin(A)} f(x)
/// where A is a bounded, symmetric dictionary in R^d, and f is L-smooth.
///
/// Two update variants are provided:
///   - Variant 0 (line-search):  x_{t+1} = argmin_{z = x_t + gamma * z_t}  ||z - b||^2
///     Closed-form:  gamma = -<x_t - b, z_t> / ||z_t||^2
///   - Variant 1 (fully-corrective): x_{t+1} = argmin_{z in lin(S)}  ||z - b||^2
///     Solved via least-squares over the active atom matrix.

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gmp {

/// @brief Update variant for the GMP algorithm.
enum class Variant : int {
    LineSearch       = 0,  ///< Variant 0: step along newest atom only.
    FullyCorrective  = 1   ///< Variant 1: re-optimise over lin(S).
};

/// @brief Result struct returned by the solver.
struct Result {
    Eigen::VectorXd x;                 ///< Final iterate.
    std::vector<double> obj_values;    ///< f(x_t) at each iteration.
    Eigen::MatrixXd active_atoms;      ///< Columns = selected atoms (including duplicates).
    Eigen::VectorXd weights;           ///< Coefficients so that x ≈ active_atoms * weights.
    int iterations;                    ///< Number of iterations executed.
};

// ────────────────────────────────────────────────────────────────────
// Objective-function concept (duck-typed, no virtual dispatch).
//
// A conforming Objective must provide:
//   double      value(const VectorXd& x)          — f(x)
//   VectorXd    gradient(const VectorXd& x)       — ∇f(x)
//   double      smoothness()                      — Lipschitz constant L
// ────────────────────────────────────────────────────────────────────

/// @brief Norm-Corrective Generalized Matching Pursuit solver.
///
/// @tparam Objective  A type satisfying the objective concept above.
template <typename Objective>
class NormCorrectiveGMP {
public:
    /// @brief Construct the solver.
    ///
    /// @param obj       Objective function (moved in).
    /// @param atoms     Dictionary matrix A ∈ R^{d × k}.  Each column is an atom.
    ///                  The algorithm automatically symmetrises: it searches
    ///                  over A ∪ (−A) when querying the LMO.
    /// @param variant   Update variant (LineSearch or FullyCorrective).
    /// @param max_iter  Maximum number of outer iterations T.
    /// @param tol       Convergence tolerance on |f(x_t) − f(x_{t−1})|.
    NormCorrectiveGMP(Objective obj,
                      Eigen::MatrixXd atoms,
                      Variant variant   = Variant::LineSearch,
                      int     max_iter  = 500,
                      double  tol       = 1e-10)
        : obj_(std::move(obj))
        , A_(std::move(atoms))
        , variant_(variant)
        , max_iter_(max_iter)
        , tol_(tol)
    {
        if (A_.cols() == 0 || A_.rows() == 0)
            throw std::invalid_argument("NormCorrectiveGMP: atom matrix must be non-empty.");
    }

    // ── setters ────────────────────────────────────────────────────
    void set_variant(Variant v)    { variant_  = v; }
    void set_max_iter(int m)       { max_iter_ = m; }
    void set_tolerance(double t)   { tol_      = t; }

    // ── main entry point ───────────────────────────────────────────

    /// @brief Run the algorithm.
    ///
    /// @param x0  Initial iterate.  Must satisfy x0 ∈ lin(A) (not checked).
    /// @return     Result struct with the final iterate, objective trace,
    ///             active atoms, weights, and iteration count.
    Result solve(const Eigen::VectorXd& x0) const
    {
        const int d = static_cast<int>(A_.rows());
        if (x0.size() != d)
            throw std::invalid_argument("NormCorrectiveGMP::solve: x0 dimension mismatch.");

        const double L = obj_.smoothness();

        Result res;
        res.obj_values.reserve(static_cast<std::size_t>(max_iter_ + 1));

        Eigen::VectorXd x = x0;
        double fx = obj_.value(x);
        res.obj_values.push_back(fx);

        // Active-set bookkeeping: columns of S store the selected atoms.
        // We keep a dynamic list; for Variant 1 we solve the LS problem
        // over these columns at every step.
        std::vector<Eigen::VectorXd> active_list;

        for (int t = 0; t < max_iter_; ++t) {
            // ── Step 3: LMO over A ∪ (−A) ──────────────────────────
            Eigen::VectorXd grad = obj_.gradient(x);
            Eigen::VectorXd zt   = lmo_symmetric(grad);

            // ── Step 4: S := S ∪ {z_t} ─────────────────────────────
            active_list.push_back(zt);

            // ── Step 5: b := x_t − (1/L) ∇f(x_t) ──────────────────
            Eigen::VectorXd b = x - (1.0 / L) * grad;

            // ── Step 6: update ──────────────────────────────────────
            if (variant_ == Variant::LineSearch) {
                // Variant 0:  gamma = −<x − b, z_t> / ||z_t||^2
                const double zt_sq = zt.squaredNorm();
                if (zt_sq < std::numeric_limits<double>::epsilon()) {
                    // Degenerate atom; skip update.
                } else {
                    const double gamma = -(x - b).dot(zt) / zt_sq;
                    x += gamma * zt;
                }
            } else {
                // Variant 1:  x_{t+1} = argmin_{z ∈ lin(S)} ||z − b||^2
                //   ⟺  S w = b  in the least-squares sense.
                x = solve_ls_active(active_list, b);
            }

            // ── convergence check ───────────────────────────────────
            const double fx_new = obj_.value(x);
            res.obj_values.push_back(fx_new);

            if (std::abs(fx_new - fx) < tol_) {
                fx = fx_new;
                ++t;
                res.iterations = t;
                break;
            }
            fx = fx_new;
            res.iterations = t + 1;
        }

        // ── pack result ─────────────────────────────────────────────
        res.x = x;

        const int s = static_cast<int>(active_list.size());
        res.active_atoms.resize(d, s);
        for (int j = 0; j < s; ++j)
            res.active_atoms.col(j) = active_list[static_cast<std::size_t>(j)];

        // Compute final weights:  active_atoms * w ≈ x
        res.weights = res.active_atoms.colPivHouseholderQr().solve(x);

        return res;
    }

private:
    Objective        obj_;
    Eigen::MatrixXd  A_;         ///< Dictionary  (d × k).
    Variant          variant_;
    int              max_iter_;
    double           tol_;

    // ── Linear Minimisation Oracle over A ∪ (−A) ──────────────────
    //
    // Returns  z = argmin_{z ∈ A ∪ (−A)}  <d, z>
    //        = the atom (or its negation) whose inner product with d is
    //          most negative.
    //
    Eigen::VectorXd lmo_symmetric(const Eigen::VectorXd& d) const
    {
        // inner products:  A^T d   (size k)
        Eigen::VectorXd ip = A_.transpose() * d;    // (k × 1)

        // For the symmetric set A ∪ (−A), the minimum is either at
        // some column j  with ip(j), or at −column j  with −ip(j).
        // Equivalently: find j that maximises |ip(j)|, then use sign.
        Eigen::Index best_j = 0;
        double best_val = std::abs(ip(0));
        for (Eigen::Index j = 1; j < ip.size(); ++j) {
            const double v = std::abs(ip(j));
            if (v > best_val) {
                best_val = v;
                best_j   = j;
            }
        }

        // If ip(best_j) > 0, the minimum is at −A.col(best_j).
        if (ip(best_j) > 0.0)
            return -A_.col(best_j);
        else
            return  A_.col(best_j);
    }

    // ── Least-squares solve over lin(S) ───────────────────────────
    //
    // Given atoms z_0, …, z_{t} (columns of the active set), solve
    //   min_w  ||S w − b||^2
    // and return  x = S w.
    //
    static Eigen::VectorXd solve_ls_active(
        const std::vector<Eigen::VectorXd>& active,
        const Eigen::VectorXd& b)
    {
        const int d = static_cast<int>(b.size());
        const int s = static_cast<int>(active.size());

        Eigen::MatrixXd S(d, s);
        for (int j = 0; j < s; ++j)
            S.col(j) = active[static_cast<std::size_t>(j)];

        // Solve via column-pivoted QR (robust to rank deficiency).
        Eigen::VectorXd w = S.colPivHouseholderQr().solve(b);
        return S * w;
    }
};

// ====================================================================
// Convenience: Least-Squares objective  f(x) = 0.5 ||y − x||^2
// ====================================================================

/// @brief Least-squares objective  f(x) = 0.5 ||y − x||^2.
///
/// For this choice the smoothness constant is L = 1, and
/// the gradient is  ∇f(x) = x − y.
struct LeastSquaresObjective {
    Eigen::VectorXd y;   ///< Target signal.

    explicit LeastSquaresObjective(Eigen::VectorXd target)
        : y(std::move(target)) {}

    double value(const Eigen::VectorXd& x) const {
        return 0.5 * (y - x).squaredNorm();
    }

    Eigen::VectorXd gradient(const Eigen::VectorXd& x) const {
        return x - y;
    }

    double smoothness() const { return 1.0; }
};

// ====================================================================
// Convenience: Quadratic objective  f(x) = 0.5 x^T H x − c^T x
// ====================================================================

/// @brief Quadratic objective  f(x) = 0.5 x^T H x − c^T x.
///
/// Smoothness constant equals the largest eigenvalue of H.
struct QuadraticObjective {
    Eigen::MatrixXd H;   ///< Positive semi-definite Hessian.
    Eigen::VectorXd c;   ///< Linear term.
    double L;             ///< Smoothness constant (max eigenvalue of H).

    /// @brief Construct from H and c.  L is computed automatically.
    QuadraticObjective(Eigen::MatrixXd H_in, Eigen::VectorXd c_in)
        : H(std::move(H_in))
        , c(std::move(c_in))
    {
        // Compute largest eigenvalue via self-adjoint solver.
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H, Eigen::EigenvaluesOnly);
        L = eig.eigenvalues().maxCoeff();
    }

    double value(const Eigen::VectorXd& x) const {
        return 0.5 * x.dot(H * x) - c.dot(x);
    }

    Eigen::VectorXd gradient(const Eigen::VectorXd& x) const {
        return H * x - c;
    }

    double smoothness() const { return L; }
};

}  // namespace gmp
