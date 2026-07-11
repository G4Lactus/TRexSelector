// ====================================================================
// vd_omp.cpp
// ====================================================================

#include <tsolvers/vd_tsolvers/vd_omp.hpp>

// ====================================================================

namespace trex::tsolvers::vd {

void VD_OMP::init_omp_() {
    if (omp_inited_) return;
    Xty_active_.resize(0);
    omp_inited_ = true;
}

std::optional<VD_OMP::Candidate> VD_OMP::find_best_candidate_() const {
    Candidate best{Candidate::Pool::Real, -1, 0.0};

    // Unrealized VD
    for (int d = 0; d < L_; ++d) {
        if (vd_is_realized_[d]) continue;
        double ac = std::abs(vd_corr_(d));
        if (ac > best.abs_corr) best = {Candidate::Pool::VD, d, ac};
    }
    // Realized dummies (always active, can be re-selected)
    for (int j = 0; j < T_realized_; ++j) {
        bool already = false;
        for (const auto& af : active_features_)
            if (af.kind == ActiveFeature::Kind::Dummy && af.index == j)
                { already = true; break; }
        if (already) continue;
        double ac = std::abs(corr_realized_(j));
        if (ac > best.abs_corr) best = {Candidate::Pool::Realized, j, ac};
    }
    // All reals (active + inactive)
    for (int j = 0; j < p_; ++j) {
        if (is_active_[j]) continue;
        double ac = std::abs(corr_(j));
        if (ac > best.abs_corr) best = {Candidate::Pool::Real, j, ac};
    }
    if (best.index < 0 || best.abs_corr < 100.0 * opt_.eps)
        return std::nullopt;
    return best;
}

void VD_OMP::append_to_factor_(const Vec& x_col) {
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

void VD_OMP::ols_refit_() {
    const int k = (int)active_features_.size();
    if (k == 0) return;

    Vec z = Xty_active_;
    chol_factor_.topLeftCorner(k, k).transpose()
        .template triangularView<Eigen::Lower>().solveInPlace(z);
    chol_factor_.topLeftCorner(k, k)
        .template triangularView<Eigen::Upper>().solveInPlace(z);

    beta_.setZero();
    beta_dummy_.head(std::max(T_realized_, 1)).setZero();
    for (int i = 0; i < k; ++i) {
        const auto& af = active_features_[i];
        if (af.kind == ActiveFeature::Kind::Real) beta_(af.index) = z(i);
        else if (af.index < beta_dummy_.size()) beta_dummy_(af.index) = z(i);
    }

    mu_.noalias() = X_active_.leftCols(k) * z;
    residuals_ = y_ - mu_;
    full_corr_refresh_();
}

MatC VD_OMP::run(int T) {
    init_omp_();

    std::vector<Vec> path;
    auto record = [&]() {
        Vec s = beta_;
        if (normx_.size() == beta_.size()) s.array() /= normx_.array();
        path.emplace_back(std::move(s));
    };

    if (step_ == 0) record();
    const int max_steps = std::min(n_, p_) + opt_.T_stop;

    for (int it = step_; it < max_steps; ++it) {
        const int prev = T_realized_;
        auto cand = find_best_candidate_();
        if (!cand) break;

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
        ols_refit_();
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
