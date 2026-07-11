// ====================================================================
// vd_lars.cpp — LARS-specific logic
// ====================================================================

#include <tsolvers/vd_tsolvers/vd_lars.hpp>

#include <stdexcept>

// ====================================================================

namespace trex::tsolvers::vd {

void VD_LARS::init_lars_() {
    if (lars_inited_) return;
    pos_of_.assign(p_, -1);
    actives_set_.clear();
    signs_.resize(0);
    signs_dummy_ = Vec::Zero(opt_.T_stop);
    lars_inited_ = true;
}

// ---------- Find and add active features ----------
double VD_LARS::find_and_add_active_() {
    const double C_real = (p_ > 0) ? corr_.cwiseAbs().maxCoeff() : 0.0;

    double C_vd = 0.0;
    bool any_unrealized = false;
    if (L_ > 0) {
        Eigen::ArrayXd abs_vd = vd_corr_.cwiseAbs().array();
        for (int d = 0; d < L_; ++d) {
            if (vd_is_realized_[d])
                abs_vd[d] = -std::numeric_limits<double>::infinity();
            else
                any_unrealized = true;
        }
        if (any_unrealized) C_vd = abs_vd.maxCoeff();
    }

    const double C_rd = (T_realized_ > 0)
        ? corr_realized_.head(T_realized_).cwiseAbs().maxCoeff() : 0.0;

    const double C = std::max(C_real, std::max(C_vd, C_rd));
    if (C < 100.0 * opt_.eps) return 0.0;

    // VD wins before any reals are active
    if (active_features_.empty() && any_unrealized &&
        C_vd >= C_real - opt_.eps && C_vd >= C_rd - opt_.eps) {
        int best = -1;
        double bestv = -std::numeric_limits<double>::infinity();
        for (int d = 0; d < L_; ++d) if (!vd_is_realized_[d]) {
            double v = std::abs(vd_corr_(d));
            if (v > bestv) { bestv = v; best = d; }
        }
        if (best >= 0) {
            realize_dummy_(best);
            int jslot = T_realized_ - 1;
            active_features_.push_back({ActiveFeature::Kind::Dummy, jslot});

            if (signs_dummy_.size() < T_realized_)
                signs_dummy_.conservativeResize(T_realized_);
            signs_dummy_(jslot) =
                (corr_realized_(jslot) >= 0.0) ? 1.0 : -1.0;
        }
        return C;
    }

    // Add all reals with |corr| >= C - eps
    std::vector<int> new_actives;
    new_actives.reserve(16);
    const double thresh = C - opt_.eps;
    for (int j = 0; j < p_; ++j) {
        if (!is_active_[j] && std::abs(corr_(j)) >= thresh)
            new_actives.push_back(j);
    }

    for (int j : new_actives) {
        actives_.push_back(j);
        actives_set_.insert(j);
        is_active_[j] = 1;
        pos_of_[j] = (int)actives_.size() - 1;
        active_features_.push_back({ActiveFeature::Kind::Real, j});
    }

    if (!new_actives.empty()) {
        signs_.resize((int)actives_.size());
        for (int idx = 0; idx < (int)actives_.size(); ++idx) {
            int j = actives_[idx];
            signs_(idx) = (corr_(j) >= 0.0) ? 1.0 : -1.0;
        }
        for (int j : new_actives) {
            int pos = pos_of_[j];
            double s = signs_(pos);
            Vec v = X_.col(j) * s;
            if (auto v_ortho = orthonormalize_(v)) {
                basis_.col(basis_size_()) = *v_ortho;
                basis_indices_.push_back(j);
                update_virtual_dummies_();
            }
        }
    }
    return C;
}

// ---------- Update Cholesky with signed columns ----------
void VD_LARS::update_factor_() {
    if (active_features_.empty()) return;
    const int have = chol_factor_.rows();
    const int want = (int)active_features_.size();
    if (have >= want) return;

    const auto& af = active_features_[want - 1];
    const int idx = af.index;

    Vec x_new;
    if (af.kind == ActiveFeature::Kind::Real) {
        int pos = pos_of_[idx];
        double s = signs_(pos);
        x_new = X_.col(idx) * s;
    } else {
        x_new = X_realized_.col(idx) * signs_dummy_(idx);
    }
    X_active_.col(have) = x_new;

    if (have == 0) {
        double s = x_new.squaredNorm();
        if (s <= opt_.eps) {
            double nrm = x_new.norm();
            if (nrm > opt_.eps) { X_active_.col(0) = x_new / nrm; s = 1.0; }
            else if (basis_size_() > 0) { X_active_.col(0) = basis_.col(0); s = 1.0; }
            else { X_active_.col(0).setZero(); X_active_.col(0)(0) = 1.0; s = 1.0; }
        }
        chol_factor_.resize(1, 1);
        chol_factor_(0, 0) = std::sqrt(s);
        return;
    }

    Eigen::VectorXd v(have);
    v.noalias() = X_active_.leftCols(have).transpose() * x_new;
    double s = x_new.squaredNorm();
    if (!chol_append(chol_factor_, v, s, opt_.eps)) {
        MatC R = chol_factor_;
        if (!chol_append(R, v, s + 10.0 * opt_.eps, opt_.eps)) {
            MatC G = X_active_.leftCols(have + 1).transpose() * X_active_.leftCols(have + 1);
            Eigen::LLT<MatC> llt(G);
            if (llt.info() == Eigen::Success) chol_factor_ = llt.matrixU();
            else return;
        } else {
            chol_factor_.swap(R);
        }
    }
}

// ---------- Equiangular direction ----------
VD_LARS::Direction VD_LARS::compute_direction_() {
    if (chol_factor_.rows() == 0)
        throw std::runtime_error("chol_factor not initialized");
    const int nb = basis_size_();
    const int m  = chol_factor_.rows();

    Eigen::VectorXd ones = Eigen::VectorXd::Ones(m);
    Eigen::VectorXd z = ones;
    chol_factor_.transpose().template triangularView<Eigen::Lower>().solveInPlace(z);
    Eigen::VectorXd w0 = z;
    chol_factor_.template triangularView<Eigen::Upper>().solveInPlace(w0);

    double denom = ones.dot(w0);
    double A_act = 1.0 / std::sqrt(std::max(denom, opt_.eps));
    Eigen::VectorXd w = A_act * w0;

    Vec u(n_);
    u.noalias() = X_active_.leftCols(m) * w;

    Vec a(p_);
    vd_detail::gemv_Xt(X_, u, a);

    Eigen::VectorXd u_prime(nb);
    if (nb > 0) u_prime.noalias() = basis_.leftCols(nb).transpose() * u;

    int m_rows = std::min(nb, vd_rows_filled_);
    Eigen::VectorXd a_vd;
    if (m_rows > 0) {
        a_vd = vd_proj_.topRows(m_rows).transpose() * u_prime.head(m_rows);
    } else { a_vd.resize(L_); a_vd.setZero(); }

    Eigen::VectorXd a_rd;
    if (T_realized_ > 0)
        a_rd = X_realized_.leftCols(T_realized_).transpose() * u;
    else a_rd.resize(0);

    Direction out;
    out.A_active = A_act;
    out.w = std::move(w); out.u = std::move(u); out.a = std::move(a);
    out.a_vd = std::move(a_vd); out.a_rd = std::move(a_rd);
    return out;
}

// ---------- Take step ----------
double VD_LARS::take_step_(double C, double A_active,
    const Vec& w, const Vec& u,
    const Vec& a, const Vec& a_vd, const Vec& a_rd)
{
    const double tol = opt_.eps;

    Vec d_real  = Vec::Zero(p_);
    Vec d_dummy = Vec::Zero(T_realized_);
    int w_idx = 0;
    for (const auto& af : active_features_) {
        double w_m = w(w_idx++);
        if (af.kind == ActiveFeature::Kind::Real) {
            int pos = pos_of_[af.index];
            d_real(af.index) = signs_(pos) * w_m;
        } else {
            if (af.index < T_realized_)
                d_dummy(af.index) = signs_dummy_(af.index) * w_m;
        }
    }

    // Entering real features need no index bookkeeping here: they are picked
    // up by find_and_add_active_() at the start of the next iteration.
    double gamma_real = std::numeric_limits<double>::infinity();
    for (int j = 0; j < p_; ++j) {
        if (is_active_[j]) continue;
        double aj = a(j), cj = corr_(j);
        double den1 = A_active - aj;
        if (den1 > tol) { double t = (C - cj) / den1; if (t > tol && t < gamma_real) gamma_real = t; }
        double den2 = A_active + aj;
        if (den2 > tol) { double t = (C + cj) / den2; if (t > tol && t < gamma_real) gamma_real = t; }
    }

    double gamma_vd = std::numeric_limits<double>::infinity();
    int idx_vd = -1;
    for (int d = 0; d < L_; ++d) {
        if (vd_is_realized_[d]) continue;
        double aj = a_vd(d), cj = vd_corr_(d);
        double den1 = A_active - aj;
        if (den1 > tol) { double t = (C - cj) / den1; if (t > tol && t < gamma_vd) { gamma_vd = t; idx_vd = d; } }
        double den2 = A_active + aj;
        if (den2 > tol) { double t = (C + cj) / den2; if (t > tol && t < gamma_vd) { gamma_vd = t; idx_vd = d; } }
    }

    bool winner_is_vd = (gamma_vd < gamma_real + tol);
    double gamma = winner_is_vd ? gamma_vd : gamma_real;
    if (!(gamma > opt_.eps)) return 0.0;

    mu_.noalias()   += gamma * u;
    beta_.noalias() += gamma * d_real;
    if (T_realized_ > 0 && d_dummy.size() == T_realized_)
        beta_dummy_.head(T_realized_).noalias() += gamma * d_dummy;

    residuals_ = y_ - mu_;
    corr_.noalias() -= gamma * a;

    for (int d = 0; d < L_; ++d)
        if (!vd_is_realized_[d]) vd_corr_(d) -= gamma * a_vd(d);

    if (T_realized_ > 0 && a_rd.size() == T_realized_)
        corr_realized_.head(T_realized_).noalias() -= gamma * a_rd;

    if (winner_is_vd && idx_vd >= 0) {
        realize_dummy_(idx_vd);
        int jslot = T_realized_ - 1;
        active_features_.push_back({ActiveFeature::Kind::Dummy, jslot});

        if (signs_dummy_.size() < T_realized_)
            signs_dummy_.conservativeResize(T_realized_);
        signs_dummy_(jslot) =
            (corr_realized_(jslot) >= 0.0) ? 1.0 : -1.0;
    }
    return gamma;
}

// ---------- Run ----------
MatC VD_LARS::run(int T) {
    init_lars_();

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
        double C = find_and_add_active_();
        if (C <= 0.0) break;
        update_factor_();
        Direction dir = compute_direction_();
        double gamma = take_step_(C, dir.A_active, dir.w, dir.u,
                                  dir.a, dir.a_vd, dir.a_rd);
        if (gamma <= opt_.eps) break;
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
