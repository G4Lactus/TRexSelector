#pragma once
// ═════════════════════════════════════════════════════════════════════════════
// afs.hpp  —  Adaptive Forward Stepwise Regression (linear regression)
// Zhang & Tibshirani, JMLR 27 (2026)
//
// Requires: Eigen 3.4+
// Build:    g++ -O2 -std=c++17 -I<eigen_path> main.cpp -o afs_demo
// ═════════════════════════════════════════════════════════════════════════════
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>

using MatXd = Eigen::MatrixXd;
using VecXd = Eigen::VectorXd;

// ─────────────────────────────────────────────────────────────────────────────
// AFSPath — returned by AFS::fit()
// ─────────────────────────────────────────────────────────────────────────────
struct AFSPath {
    MatXd            coef;    // p × T  coefficient path (column m = β_m)
    std::vector<int> order;   // variables in their entry order
    VecXd            l1norm;  // ||β_m||_1 at each step
    int              T;       // actual steps completed
};

// ─────────────────────────────────────────────────────────────────────────────
// AFS — Adaptive Forward Stepwise solver
//
// Hyperparameters (both chosen by cross-validation in practice):
//   rho  ∈ (0,1]  shrinkage step size
//                   rho = 1  →  Forward Stepwise  (no shrinkage)
//                   rho → 0  →  LAR / LASSO       (maximum shrinkage)
//   M              maximum number of steps / max active set size
//   h              L1 norm cap; set to +inf when n > p (default)
//   eps            collinearity threshold for Cholesky pivot check
// ─────────────────────────────────────────────────────────────────────────────
class AFS {
public:
    double rho;
    int    M;
    double h;
    double eps;

    explicit AFS(double rho,
                 int    M,
                 double h   = std::numeric_limits<double>::infinity(),
                 double eps = 1e-10)
        : rho(rho), M(M), h(h), eps(eps)
    {
        if (rho <= 0.0 || rho > 1.0)
            throw std::domain_error("AFS: rho must be in (0, 1]");
        if (M <= 0)
            throw std::domain_error("AFS: M must be positive");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // fit()
    //   X : n × p  design matrix  (column-centred, no intercept)
    //   y : n      response vector (centred)
    // ─────────────────────────────────────────────────────────────────────────
    AFSPath fit(const MatXd& X, const VecXd& y) const
    {
        const int n   = static_cast<int>(X.rows());
        const int p   = static_cast<int>(X.cols());
        const int cap = std::min({M, n - 1, p});   // max active set size

        // ── Persistent state ─────────────────────────────────────────────────
        VecXd beta = VecXd::Zero(p);   // AFS coefficients  β_m
        VecXd r    = y;                // residual           r_m = y - Xβ_m
        VecXd Xty  = X.transpose() * y;
        VecXd corr = Xty;             // correlation vector X^T r_m

        // ── Growing Cholesky factor ───────────────────────────────────────────
        // L L^T = X_{A_m}^T X_{A_m}
        // Only the top-left (s × s) block is live at step m.
        // Pre-allocated once; extended by one row/col per accepted step.
        MatXd L    = MatXd::Zero(cap, cap);
        VecXd XAty = VecXd::Zero(cap);   // X_{A_m}^T y, extended per step

        // ── Path storage ─────────────────────────────────────────────────────
        MatXd path   = MatXd::Zero(p, M);
        VecXd l1norm = VecXd::Zero(M);
        std::vector<int> order;
        order.reserve(cap);
        std::vector<bool> selected(p, false);
        int T = 0;

        // ── Main loop ─────────────────────────────────────────────────────────
        for (int m = 0; m < M; ++m) {

            // ── Step 1 : SCREEN ───────────────────────────────────────────────
            // j* = argmax_{j ∉ A} |x_j^T r_{m-1}|
            const int j_star = screen(corr, selected, p);
            if (j_star < 0) break;          // all p variables already active

            const int s  = static_cast<int>(order.size());
            const int s1 = s + 1;
            if (s1 > cap) break;

            order.push_back(j_star);
            selected[j_star] = true;

            const VecXd x_new = X.col(j_star);

            // ── Step 2 : CHOLESKY EXTENSION ───────────────────────────────────
            //
            //   b    = X_{A_{m-1}}^T x_{j*}          ∈ R^s
            //   c_jj = ||x_{j*}||^2                   scalar
            //   ℓ    = L_s^{-1} b                    ∈ R^s
            //   d    = sqrt(c_jj - ||ℓ||^2)
            //
            //   L_{s1} = [ L_s   0 ]
            //             [ ℓ^T   d ]
            VecXd b(s);
            for (int i = 0; i < s; ++i)
                b(i) = X.col(order[i]).dot(x_new);

            const double c_jj = x_new.squaredNorm();

            VecXd ell = VecXd::Zero(s);
            if (s > 0)
                ell = L.topLeftCorner(s, s)
                       .triangularView<Eigen::Lower>()
                       .solve(b);                   // O(s^2)

            const double d2 = c_jj - ell.squaredNorm();
            if (d2 < eps * eps) {
                // Near-collinear: skip this variable
                order.pop_back();
                selected[j_star] = false;
                continue;
            }
            const double d = std::sqrt(d2);

            if (s > 0)
                L.block(s, 0, 1, s) = ell.transpose();
            L(s, s) = d;
            XAty(s) = x_new.dot(y);

            // ── Step 3 : SOLVE  ν_m = (X_{A_m}^T X_{A_m})^{-1} X_{A_m}^T y ─
            const auto  Ls  = L.topLeftCorner(s1, s1).triangularView<Eigen::Lower>();
            const VecXd rhs = XAty.head(s1);
            const VecXd w   = Ls.solve(rhs);             // forward   O(s^2)
            const VecXd nu  = Ls.transpose().solve(w);   // backward  O(s^2)

            // ── Step 4 : BLEND ────────────────────────────────────────────────
            // β_m = (1-ρ) β_{m-1}  +  ρ ν_m
            beta *= (1.0 - rho);
            for (int i = 0; i < s1; ++i)
                beta(order[i]) += rho * nu(i);

            // ── Step 5 : UPDATE RESIDUAL & CORRELATION ────────────────────────
            // r_m   = (1-ρ) r_{m-1} + ρ (y - X_{A_m} ν_m)
            // corr_m = (1-ρ) corr_{m-1} + ρ (X^T y - X^T X_{A_m} ν_m)
            VecXd XA_nu = VecXd::Zero(n);
            for (int i = 0; i < s1; ++i)
                XA_nu += nu(i) * X.col(order[i]);         // O(n·s1)

            r    = (1.0 - rho) * r    + rho * (y - XA_nu);
            corr = (1.0 - rho) * corr + rho * (Xty - X.transpose() * XA_nu);

            // ── Step 6 : RECORD ───────────────────────────────────────────────
            path.col(m) = beta;
            l1norm(m)   = beta.lpNorm<1>();
            T = m + 1;

            if (l1norm(m) >= h) break;
        }

        AFSPath result;
        result.coef   = path.leftCols(T);
        result.order  = order;
        result.l1norm = l1norm.head(T);
        result.T      = T;
        return result;
    }

private:
    static int screen(const VecXd&             corr,
                      const std::vector<bool>&  selected,
                      int                       p)
    {
        int    j_star = -1;
        double best   = -1.0;
        for (int j = 0; j < p; ++j) {
            if (selected[j]) continue;
            const double v = std::abs(corr(j));
            if (v > best) { best = v; j_star = j; }
        }
        return j_star;
    }
};
