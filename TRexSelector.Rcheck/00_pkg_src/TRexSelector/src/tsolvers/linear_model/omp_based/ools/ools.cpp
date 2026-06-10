/**
 * @file ools.cpp
 * @brief Implementation of Orthogonal Least Squares (OLS) with QR and Cholesky back-ends.
 *
 * ------------------------------------------------------------------
 * Algorithm recap  (Blumensath & Davies, 2007, §III)
 * ------------------------------------------------------------------
 * OLS selects the atom whose *projection onto the orthogonal complement*
 * of the current subspace yields the largest inner product with the
 * residual.  Concretely, one maintains a shadow copy  Φ̂_Λ  of the
 * not-yet-selected columns that have been:
 *     (a) orthogonalised to all columns in Q_Γ,   and
 *     (b) re-normalised to unit ℓ₂-norm.
 * The selection criterion then reduces to
 *     i_max = argmax_i  |  φ̂_i^T  r  |
 * which is the same *form* as OMP, but uses Φ̂ instead of Φ.
 *
 * BACK-END  1 – QR  (Modified Gram-Schmidt)
 * ─────────────────────────────────────────
 *   • Maintain  Q ∈ R^{n×k},  R ∈ R^{k×k}  for the active set.
 *   • Maintain a shadow matrix  Φ̂_Λ  whose columns are orthogonal
 *     to span(Q) and unit-normalised.
 *   • After each selection, update Q, R via one MGS pass,
 *     then deflate every remaining shadow column by the new q-vector.
 *   • Solve  R s = Q^T x  via back-substitution at the end.
 *
 * BACK-END  2 – Cholesky rank-1 update
 * ─────────────────────────────────────
 *   • Maintain  L ∈ R^{k×k}  lower-triangular such that
 *         L L^T = Φ_Γ^T Φ_Γ
 *   • For the OLS selection criterion we still need the
 *     projected-and-normalised inner products.  We compute:
 *       h_j = Φ_Γ^T φ_j                (cross-correlations)
 *       w   = L^{-1} h_j               (forward-solve)
 *       projection norm²  = ‖w‖²
 *       orthogonal component² = ‖φ_j‖² − ‖w‖²   (= ‖φ̂_j‖²)
 *       effective correlation = φ_j^T r / ‖φ̂_j‖
 *     and select j* maximising |effective correlation|.
 *   • After selection, rank-1 Cholesky update:
 *         new row / col of the Gram  →  append to L.
 *   • Solve  L L^T s = Φ_Γ^T x  at the end.
 */

#include "ools/ools.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace ools {

// =====================================================================
//  Constructor
// =====================================================================
OOLS::OOLS(MatRef Phi, VecRef x, Backend backend, bool normalize_cols)
    : Phi_(Phi),
      x_(x),
      col_norms_(Eigen::VectorXd::Ones(Phi.cols())),
      n_(Phi.rows()),
      p_(Phi.cols()),
      backend_(backend),
      normalised_(normalize_cols)
{
    if (Phi.rows() != x.size())
        throw std::invalid_argument("OOLS: Phi.rows() != x.size()");

    if (normalize_cols) {
        col_norms_.resize(p_);
        for (Index j = 0; j < p_; ++j) {
            double nrm = Phi_.col(j).norm();
            if (nrm < 1e-15)
                throw std::invalid_argument("OOLS: zero-norm column detected");
            col_norms_(j) = nrm;
            Phi_.col(j) /= nrm;
        }
    }
}

// =====================================================================
//  Public driver
// =====================================================================
OOLSResult OOLS::solve(std::size_t max_iter, double tol) const
{
    if (max_iter == 0)
        throw std::invalid_argument("OOLS::solve: max_iter must be > 0");

    // Clamp to min(n, p)
    const auto K = static_cast<std::size_t>(std::min(n_, p_));
    max_iter = std::min(max_iter, K);

    OOLSResult res = (backend_ == Backend::QR)
                         ? solve_qr(max_iter, tol)
                         : solve_cholesky(max_iter, tol);

    // If we normalised columns, rescale coefficients:  s_j  /=  ‖φ_j‖
    if (normalised_) {
        for (std::size_t i = 0; i < res.support.size(); ++i)
            res.coefficients(static_cast<Index>(i)) /= col_norms_(res.support[i]);
    }
    return res;
}

// =====================================================================
//  BACK-END 1 :  QR  (Modified Gram-Schmidt with shadow matrix)
// =====================================================================
OOLSResult OOLS::solve_qr(std::size_t max_iter, double tol) const
{
    const Index n = n_;
    const Index p = p_;

    // Active-set bookkeeping
    std::vector<Index> support;
    support.reserve(static_cast<std::size_t>(std::min(n, p)));

    std::vector<bool> selected(static_cast<std::size_t>(p), false);

    // Q  (n × k)  and  R  (k × k)  – grow column by column
    Eigen::MatrixXd Q(n, max_iter);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(
        static_cast<Index>(max_iter), static_cast<Index>(max_iter));

    // Shadow matrix: Φ̂ = copy of Phi, progressively orthogonalised
    Eigen::MatrixXd Phi_hat = Phi_;                        // n × p
    Eigen::VectorXd shadow_norms(p);                       // ‖φ̂_j‖
    for (Index j = 0; j < p; ++j)
        shadow_norms(j) = Phi_hat.col(j).norm();

    // z = Q^T x  (accumulated incrementally)
    Eigen::VectorXd z(static_cast<Index>(max_iter));

    // Residual
    Eigen::VectorXd r = x_;
    double r_norm = r.norm();

    std::size_t iter = 0;

    while (iter < max_iter && r_norm > tol) {
        const auto k = static_cast<Index>(iter);  // current step index

        // ── OLS selection ──────────────────────────────────────────
        //  α_j = φ̂_j^T r / ‖φ̂_j‖    for  j ∉ Γ
        //  (the shadow columns are already orthogonal to span(Q),
        //   but may not be unit-norm yet — we normalise on the fly)
        Index best_j  = -1;
        double best_score = -1.0;

        for (Index j = 0; j < p; ++j) {
            if (selected[static_cast<std::size_t>(j)]) continue;

            const double nrm = shadow_norms(j);
            if (nrm < 1e-14) continue;  // numerically zero

            const double alpha = Phi_hat.col(j).dot(r) / nrm;
            const double score = std::abs(alpha);

            if (score > best_score) {
                best_score = score;
                best_j     = j;
            }
        }

        if (best_j < 0) break;  // no admissible column left

        // ── Update Q, R via MGS ────────────────────────────────────
        //  q_new  =  φ̂_{best_j}  /  ‖φ̂_{best_j}‖
        //  (the shadow column is already orthogonal to previous q's,
        //   so we just normalise it)
        const double nrm_sel = shadow_norms(best_j);
        Eigen::VectorXd q_new = Phi_hat.col(best_j) / nrm_sel;

        // Fill R column:  R(0..k-1, k) = Q(:,0..k-1)^T  φ_{best_j}  (original column)
        // But since q_new = φ̂/‖φ̂‖ and φ̂ = φ − Q Q^T φ,
        // we record R from the *original* column of Phi:
        Eigen::VectorXd phi_orig = Phi_.col(best_j);
        for (Index i = 0; i < k; ++i)
            R(i, k) = Q.col(i).dot(phi_orig);
        R(k, k) = nrm_sel;  // = ‖φ̂_{best_j}‖  before normalisation

        Q.col(k) = q_new;

        // ── Update z and residual ──────────────────────────────────
        const double z_new = q_new.dot(x_);
        z(k) = z_new;
        r -= z_new * q_new;
        r_norm = r.norm();

        // ── Mark selected ──────────────────────────────────────────
        selected[static_cast<std::size_t>(best_j)] = true;
        support.push_back(best_j);

        // ── Deflate shadow matrix ──────────────────────────────────
        //  For every j ∉ Γ:  φ̂_j  ←  φ̂_j − (q_new^T φ̂_j) q_new
        //  then update ‖φ̂_j‖
        for (Index j = 0; j < p; ++j) {
            if (selected[static_cast<std::size_t>(j)]) continue;
            const double proj = q_new.dot(Phi_hat.col(j));
            Phi_hat.col(j) -= proj * q_new;
            shadow_norms(j) = Phi_hat.col(j).norm();
        }

        ++iter;
    }

    // ── Recover coefficients:  R_k s = z_k  via back-substitution ──
    const auto K = static_cast<Index>(iter);
    Eigen::VectorXd s = R.topLeftCorner(K, K)
                            .triangularView<Eigen::Upper>()
                            .solve(z.head(K));

    return OOLSResult{
        .support       = std::move(support),
        .coefficients  = std::move(s),
        .residual      = std::move(r),
        .residual_norm = r_norm,
        .iterations    = iter
    };
}

// =====================================================================
//  BACK-END 2 :  Cholesky rank-1 update
// =====================================================================
/**
 *  Normal-equation view
 *  --------------------
 *  After selecting Γ = {γ₁, …, γ_k},
 *
 *      G_k = Φ_Γ^T Φ_Γ  ∈  R^{k×k}     (Gram on active set)
 *      L_k L_k^T = G_k                   (Cholesky)
 *
 *  Coefficients:   s_Γ  = G_k^{-1} Φ_Γ^T x   (solved via L_k)
 *  Residual:       r    = x − Φ_Γ s_Γ
 *
 *  OLS selection criterion  (for candidate j ∉ Γ):
 *  ───────────────────────────────────────────────
 *  We need the inner product of the residual with the *normalised
 *  projection* of φ_j onto the orthogonal complement of span(Φ_Γ).
 *
 *      h_j   = Φ_Γ^T φ_j                     (k × 1)
 *      w     = L_k^{-1} h_j                   (forward-solve)
 *      ‖φ̂_j‖² = ‖φ_j‖² − ‖w‖²              (orth. complement norm²)
 *      score_j = | φ_j^T r | / ‖φ̂_j‖
 *
 *  Select j* = argmax_{j ∉ Γ}  score_j.
 *
 *  Rank-1 Cholesky update after selecting j*:
 *  ───────────────────────────────────────────
 *      h   = Φ_Γ^T φ_{j*}
 *      w   = L_k^{-1} h
 *      ℓ   = sqrt( ‖φ_{j*}‖² − ‖w‖² )
 *
 *      L_{k+1} = [ L_k    0 ]
 *                [ w^T    ℓ ]
 */
OOLSResult OOLS::solve_cholesky(std::size_t max_iter, double tol) const
{
    const Index n = n_;
    const Index p = p_;

    // Active-set bookkeeping
    std::vector<Index> support;
    support.reserve(static_cast<std::size_t>(std::min(n, p)));

    std::vector<bool> selected(static_cast<std::size_t>(p), false);

    // Pre-compute column squared norms (should be ~1 if normalised)
    Eigen::VectorXd col_sq_norms(p);
    for (Index j = 0; j < p; ++j)
        col_sq_norms(j) = Phi_.col(j).squaredNorm();

    // Cholesky factor  L  (k × k), grows incrementally
    const auto K_max = static_cast<Index>(max_iter);
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(K_max, K_max);

    // Right-hand side for normal equations:  b = Φ_Γ^T x
    // (accumulated incrementally)
    Eigen::VectorXd b(K_max);

    // We also store the active-set sub-matrix columns for cross-correlations
    // Phi_Gamma(:, i)  =  Phi_(:, support[i])
    // Stored as  n × K_max  to avoid repeated indexing
    Eigen::MatrixXd Phi_Gamma(n, K_max);

    // Residual
    Eigen::VectorXd r = x_;
    double r_norm = r.norm();

    std::size_t iter = 0;

    while (iter < max_iter && r_norm > tol) {
        const auto k = static_cast<Index>(iter);  // number of already-selected atoms

        // ── OLS selection ──────────────────────────────────────────
        Index  best_j     = -1;
        double best_score = -1.0;

        for (Index j = 0; j < p; ++j) {
            if (selected[static_cast<std::size_t>(j)]) continue;

            // Cross-correlation with active set
            // h_j = Phi_Gamma(:, 0..k-1)^T  *  phi_j
            double orth_sq;
            if (k == 0) {
                orth_sq = col_sq_norms(j);
            } else {
                Eigen::VectorXd h_j = Phi_Gamma.leftCols(k).transpose() * Phi_.col(j);
                // Forward solve  L w = h_j
                Eigen::VectorXd w = L.topLeftCorner(k, k)
                                        .triangularView<Eigen::Lower>()
                                        .solve(h_j);
                orth_sq = col_sq_norms(j) - w.squaredNorm();
            }

            if (orth_sq < 1e-28) continue;  // numerically dependent

            const double orth_norm = std::sqrt(orth_sq);
            const double corr      = Phi_.col(j).dot(r);
            const double score     = std::abs(corr) / orth_norm;

            if (score > best_score) {
                best_score = score;
                best_j     = j;
            }
        }

        if (best_j < 0) break;

        // ── Cholesky rank-1 update ─────────────────────────────────
        // Append column best_j to the active set
        Phi_Gamma.col(k) = Phi_.col(best_j);

        if (k == 0) {
            L(0, 0) = std::sqrt(col_sq_norms(best_j));
        } else {
            Eigen::VectorXd h = Phi_Gamma.leftCols(k).transpose() * Phi_.col(best_j);
            Eigen::VectorXd w = L.topLeftCorner(k, k)
                                    .triangularView<Eigen::Lower>()
                                    .solve(h);
            double ell_sq = col_sq_norms(best_j) - w.squaredNorm();
            if (ell_sq < 1e-28) break;  // numerical rank deficiency
            double ell = std::sqrt(ell_sq);

            // New row of L
            L.block(k, 0, 1, k) = w.transpose();
            L(k, k) = ell;
        }

        // Accumulate b
        b(k) = Phi_.col(best_j).dot(x_);

        // Mark selected
        selected[static_cast<std::size_t>(best_j)] = true;
        support.push_back(best_j);
        ++iter;

        // ── Solve  L L^T s = b  and update residual ───────────────
        const auto kk = static_cast<Index>(iter);
        Eigen::VectorXd y_tmp = L.topLeftCorner(kk, kk)
                                    .triangularView<Eigen::Lower>()
                                    .solve(b.head(kk));
        Eigen::VectorXd s = L.topLeftCorner(kk, kk)
                                .transpose()
                                .triangularView<Eigen::Upper>()
                                .solve(y_tmp);

        // r = x - Phi_Gamma * s
        r = x_ - Phi_Gamma.leftCols(kk) * s;
        r_norm = r.norm();
    }

    // ── Final coefficients ─────────────────────────────────────────
    const auto K = static_cast<Index>(iter);
    Eigen::VectorXd y_tmp = L.topLeftCorner(K, K)
                                .triangularView<Eigen::Lower>()
                                .solve(b.head(K));
    Eigen::VectorXd s = L.topLeftCorner(K, K)
                            .transpose()
                            .triangularView<Eigen::Upper>()
                            .solve(y_tmp);

    return OOLSResult{
        .support       = std::move(support),
        .coefficients  = std::move(s),
        .residual      = std::move(r),
        .residual_norm = r_norm,
        .iterations    = iter
    };
}

}  // namespace ools
