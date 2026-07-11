// ====================================================================
// vd_afs_logistic.cpp — GLM forward stepwise with logistic link.
// ====================================================================

#include <tsolvers/vd_tsolvers/vd_afs_logistic.hpp>

#include <algorithm>

// ====================================================================

namespace trex::tsolvers::vd {

// ---------- Sigmoid ----------
Vec VD_AFS_Logistic::sigmoid_(const Vec& eta) {
    return (1.0 + (-eta.array()).exp()).inverse().matrix();
}

// ---------- Update score from current eta ----------
void VD_AFS_Logistic::update_score_() {
    prob_ = sigmoid_(eta_);
    score_ = y_binary_ - prob_;
}

// ---------- Init ----------
void VD_AFS_Logistic::init_logistic_() {
    if (logistic_inited_) return;
    rho_ = std::max(1e-6, std::min(opt_.rho, 1.0));

    // Reconstruct binary labels from (possibly centered) y
    y_binary_.resize(n_);
    for (int i = 0; i < n_; ++i)
        y_binary_(i) = (y_(i) > 0.0) ? 1.0 : 0.0;

    // Initial state: no active features → eta = 0, prob = 0.5
    eta_  = Vec::Zero(n_);
    prob_ = Vec::Constant(n_, 0.5);
    score_ = y_binary_ - prob_;

    // Overwrite the correlations that init_common_state_ computed with y_
    // to use the logistic score instead
    vd_detail::gemv_Xt(X_, score_, corr_);

    // Recompute VD correlations with score
    const int nb = basis_size_();
    const int m_rows = std::min(nb, vd_rows_filled_);
    if (m_rows > 0 && L_ > 0) {
        Vec bp = basis_.leftCols(nb).transpose() * score_;
        vd_corr_.noalias() =
            vd_proj_.topRows(m_rows).transpose() * bp.head(m_rows);
    }

    nu_active_.resize(0);
    irls_stale_ = true;
    logistic_inited_ = true;
}

// ---------- Candidate search (same as AFS: scan ALL features) ----------
std::optional<VD_AFS_Logistic::Candidate>
VD_AFS_Logistic::find_best_candidate_() const {
    Candidate best{Candidate::Pool::Real, -1, 0.0, true};

    // 1. Check Dummies first to be conservative
    for (int d = 0; d < L_; ++d) {
        if (vd_is_realized_[d]) continue;
        double ac = std::abs(vd_corr_(d));
        if (ac > best.abs_corr) best = {Candidate::Pool::VD, d, ac, true};
    }

    // 2. Check Realized Dummies (also nulls)
    for (int j = 0; j < T_realized_; ++j) {
        double ac = std::abs(corr_realized_(j));
        if (ac > best.abs_corr)
            best = {Candidate::Pool::RealizedDummy, j, ac, false};
    }

    // 3. Check Real features LAST
    // If a real feature has exactly the same correlation as a dummy,
    // the '>' will fail, and the dummy will remain the 'best'.
    for (int j = 0; j < p_; ++j) {
        double ac = std::abs(corr_(j));
        if (ac > best.abs_corr) best = {Candidate::Pool::Real, j, ac, !is_active_[j]};
    }

    if (best.index < 0 || best.abs_corr < 100.0 * opt_.eps)
        return std::nullopt;
    return best;
}

// ---------- IRLS solve on active set ----------
void VD_AFS_Logistic::irls_solve_(int max_iter, double tol) {
    const int k = (int)active_features_.size();
    if (k == 0) { nu_active_.resize(0); return; }

    // Start from current beta restricted to active set
    nu_active_.resize(k);
    for (int i = 0; i < k; ++i) {
        const auto& af = active_features_[i];
        if (af.kind == ActiveFeature::Kind::Real)
            nu_active_(i) = beta_(af.index);
        else if (af.index < beta_dummy_.size())
            nu_active_(i) = beta_dummy_(af.index);
        else
            nu_active_(i) = 0.0;
    }

    const double clamp_lo = 1e-10;
    const double clamp_hi = 1.0 - 1e-10;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Linear predictor from active columns
        Vec eta_a = X_active_.leftCols(k) * nu_active_;

        // Probabilities (clamped)
        Vec p_a = sigmoid_(eta_a);
        for (int i = 0; i < n_; ++i)
            p_a(i) = std::clamp(p_a(i), clamp_lo, clamp_hi);

        // Weights and working response
        Vec w = p_a.array() * (1.0 - p_a.array());
        Vec z = eta_a.array() + (y_binary_.array() - p_a.array()) / w.array();

        // Weighted normal equations: (X_A^T W X_A) beta = X_A^T W z
        // Form sqrt(W) * X_A and sqrt(W) * z
        Vec sqrt_w = w.cwiseSqrt();
        MatC Xw(n_, k);
        for (int j = 0; j < k; ++j)
            Xw.col(j) = X_active_.col(j).cwiseProduct(sqrt_w);
        Vec zw = z.cwiseProduct(sqrt_w);

        // Gram matrix and RHS
        MatC G = Xw.transpose() * Xw;
        Vec rhs = Xw.transpose() * zw;

        // Regularize slightly for numerical stability
        G.diagonal().array() += opt_.eps;

        // Solve via Cholesky
        Eigen::LLT<MatC> llt(G);
        if (llt.info() != Eigen::Success) break;  // fallback: keep current
        Vec nu_new = llt.solve(rhs);

        // Check convergence
        double delta = (nu_new - nu_active_).squaredNorm();
        nu_active_ = nu_new;
        if (delta < tol * tol * (1.0 + nu_active_.squaredNorm())) break;
    }

    irls_stale_ = false;
}

// ---------- AFS blend ----------
void VD_AFS_Logistic::afs_blend_() {
    const int k = (int)active_features_.size();
    if (k == 0) return;
    if (irls_stale_) irls_solve_();

    // Blend coefficients: beta = (1-rho)*beta_old + rho*nu
    beta_ *= (1.0 - rho_);
    for (int i = 0; i < k; ++i) {
        const auto& af = active_features_[i];
        if (af.kind == ActiveFeature::Kind::Real)
            beta_(af.index) += rho_ * nu_active_(i);
    }
    beta_dummy_.head(std::max(T_realized_, 1)) *= (1.0 - rho_);
    for (int i = 0; i < k; ++i) {
        const auto& af = active_features_[i];
        if (af.kind == ActiveFeature::Kind::Dummy && af.index < beta_dummy_.size())
            beta_dummy_(af.index) += rho_ * nu_active_(i);
    }

    // Recompute linear predictor from blended beta
    Vec beta_active(k);
    for (int i = 0; i < k; ++i) {
        const auto& af = active_features_[i];
        if (af.kind == ActiveFeature::Kind::Real)
            beta_active(i) = beta_(af.index);
        else if (af.index < beta_dummy_.size())
            beta_active(i) = beta_dummy_(af.index);
        else
            beta_active(i) = 0.0;
    }
    eta_ = X_active_.leftCols(k) * beta_active;

    // Update score: s = y_binary - sigmoid(eta)
    update_score_();

    // Also update mu_ and residuals_ for compatibility with base class
    mu_ = sigmoid_(eta_);
    residuals_ = score_;  // so that any base-class code using residuals_ sees the score

    // Full correlation refresh using score_direction_() → score_
    full_corr_refresh_();
}

// ---------- Run ----------
MatC VD_AFS_Logistic::run(int T) {
    init_logistic_();

    std::vector<Vec> path;
    auto record = [&]() {
        Vec s = beta_;
        if (normx_.size() == beta_.size())
            s.array() /= normx_.array();
        path.emplace_back(std::move(s));
    };

    if (step_ == 0) record();
    const int max_steps = p_ + opt_.T_stop;

    for (int it = step_; it < max_steps; ++it) {
        const int prev = T_realized_;
        auto cand = find_best_candidate_();
        if (!cand) break;

        if (cand->is_new) {
            Vec x_col;
            if (cand->pool == Candidate::Pool::VD) {
                realize_dummy_(cand->index);
                int jslot = T_realized_ - 1;
                active_features_.push_back({ActiveFeature::Kind::Dummy, jslot});
                x_col = X_realized_.col(jslot);
            } else if (cand->pool == Candidate::Pool::Real) {
                int j = cand->index;
                actives_.push_back(j);
                is_active_[j] = 1;
                active_features_.push_back({ActiveFeature::Kind::Real, j});
                x_col = X_.col(j);
            } else {
                int jslot = cand->index;
                active_features_.push_back({ActiveFeature::Kind::Dummy, jslot});
                x_col = X_realized_.col(jslot);
            }

            // Store column in X_active_ for IRLS
            const int k_new = (int)active_features_.size() - 1;
            X_active_.col(k_new) = x_col;

            irls_stale_ = true;
        }

        // Blend (calls IRLS if stale)
        afs_blend_();

        // Grow basis from score direction (only when new feature entered)
        if (cand->is_new) {
            grow_basis_from_score_();
        }

        record();

        if (T_realized_ > prev && T_realized_ >= T) { step_ = it + 1; break; }
        step_ = it + 1;
    }

    int cols = (int)path.size();
    MatC out(p_, std::max(cols, 1));
    if (cols == 0) { out.col(0).setZero(); return out; }
    for (int m = 0; m < cols; ++m) out.col(m) = path[m];
    return out;
}

}  // namespace trex::tsolvers::vd
