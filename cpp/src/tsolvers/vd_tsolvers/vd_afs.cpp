// ====================================================================
// vd_afs.cpp
// ====================================================================

#include <tsolvers/vd_tsolvers/vd_afs.hpp>

// ====================================================================

namespace trex::tsolvers::vd {

void VD_AFS::init_afs_() {
    if (afs_inited_) return;
    rho_ = std::max(1e-6, std::min(opt_.rho, 1.0));
    Xty_active_.resize(0);
    nu_active_.resize(0);
    nu_stale_ = true;
    afs_inited_ = true;
}

std::optional<VD_AFS::Candidate> VD_AFS::find_best_candidate_() const {
    Candidate best{Candidate::Pool::Real, -1, 0.0, true};

    // Unrealized VD
    for (int d = 0; d < L_; ++d) {
        if (vd_is_realized_[d]) continue;
        double ac = std::abs(vd_corr_(d));
        if (ac > best.abs_corr) best = {Candidate::Pool::VD, d, ac, true};
    }
    // Realized dummies (always active, can be re-selected)
    for (int j = 0; j < T_realized_; ++j) {
        double ac = std::abs(corr_realized_(j));
        if (ac > best.abs_corr)
            best = {Candidate::Pool::RealizedDummy, j, ac, false};
    }
    // All reals (active + inactive)
    for (int j = 0; j < p_; ++j) {
        double ac = std::abs(corr_(j));
        if (ac > best.abs_corr) best = {Candidate::Pool::Real, j, ac, !is_active_[j]};
    }
    if (best.index < 0 || best.abs_corr < 100.0 * opt_.eps)
        return std::nullopt;
    return best;
}

void VD_AFS::append_to_factor_(const Vec& x_col) {
    const int k = (int)active_features_.size() - 1;
    X_active_.col(k) = x_col;
    Xty_active_.conservativeResize(k + 1);
    Xty_active_(k) = x_col.dot(y_);

    if (k == 0) {
        double s = x_col.squaredNorm();
        if (s <= opt_.eps) s = opt_.eps;
        chol_factor_.resize(1, 1);
        chol_factor_(0, 0) = std::sqrt(s);
        return;
    }
    Eigen::VectorXd v(k);
    v.noalias() = X_active_.leftCols(k).transpose() * x_col;
    double s = x_col.squaredNorm();
    if (!chol_append(chol_factor_, v, s, opt_.eps)) {
        MatC G = X_active_.leftCols(k + 1).transpose() * X_active_.leftCols(k + 1);
        Eigen::LLT<MatC> llt(G);
        if (llt.info() == Eigen::Success) chol_factor_ = llt.matrixU();
    }
}

void VD_AFS::ols_solve_() {
    const int k = (int)active_features_.size();
    if (k == 0) { nu_active_.resize(0); return; }
    nu_active_ = Xty_active_;
    chol_factor_.topLeftCorner(k, k).transpose()
        .template triangularView<Eigen::Lower>().solveInPlace(nu_active_);
    chol_factor_.topLeftCorner(k, k)
        .template triangularView<Eigen::Upper>().solveInPlace(nu_active_);
    nu_stale_ = false;
}

void VD_AFS::afs_blend_() {
    const int k = (int)active_features_.size();
    if (k == 0) return;
    if (nu_stale_) ols_solve_();

    Vec Xa_nu(n_);
    Xa_nu.noalias() = X_active_.leftCols(k) * nu_active_;

    mu_ = (1.0 - rho_) * mu_ + rho_ * Xa_nu;

    beta_ *= (1.0 - rho_);
    for (int i = 0; i < k; ++i) {
        const auto& af = active_features_[i];
        if (af.kind == ActiveFeature::Kind::Real) beta_(af.index) += rho_ * nu_active_(i);
    }
    beta_dummy_.head(std::max(T_realized_, 1)) *= (1.0 - rho_);
    for (int i = 0; i < k; ++i) {
        const auto& af = active_features_[i];
        if (af.kind == ActiveFeature::Kind::Dummy && af.index < beta_dummy_.size())
            beta_dummy_(af.index) += rho_ * nu_active_(i);
    }

    residuals_ = y_ - mu_;
    full_corr_refresh_();
}

MatC VD_AFS::run(int T) {
    init_afs_();

    std::vector<Vec> path;
    auto record = [&]() {
        Vec s = beta_;
        if (normx_.size() == beta_.size()) s.array() /= normx_.array();
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
                if (auto vo = orthonormalize_(x_col)) {
                    basis_.col(basis_size_()) = *vo;
                    basis_indices_.push_back(j);
                    update_virtual_dummies_();
                }
            } else {
                int jslot = cand->index;
                active_features_.push_back({ActiveFeature::Kind::Dummy, jslot});
                x_col = X_realized_.col(jslot);
            }
            append_to_factor_(x_col);
            nu_stale_ = true;
        }

        afs_blend_();
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
