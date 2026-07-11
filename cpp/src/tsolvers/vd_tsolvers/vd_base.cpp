// ====================================================================
// vd_base.cpp
// ====================================================================

#include <tsolvers/vd_tsolvers/vd_base.hpp>

// ====================================================================

namespace trex::tsolvers::vd {

using DummyLaw = VDDummyLaw;

// ---------- Constructors ----------
VD_Base::VD_Base(const double* Xptr, int n, int p,
                 const double* yptr, int ny,
                 int num_dummies, const VDOptions& o)
  : n_(n), p_(p), L_(num_dummies),
    opt_(o), rng_(opt_.seed),
    X_(Xptr, n, p),
    y_(yptr, ny)
{
    init_common_state_();
}

VD_Base::VD_Base(const Eigen::Ref<const MatC>& X_in,
                 const Eigen::Ref<const Vec>&  y_in,
                 int num_dummies, const VDOptions& o)
  : VD_Base(X_in.data(), (int)X_in.rows(), (int)X_in.cols(),
            y_in.data(), (int)y_in.size(), num_dummies, o)
{}

// ---------- Common initialization ----------
void VD_Base::init_common_state_() {
    const int min_np = std::min(n_, p_);

    standardized_ = false;
    y_norm_       = y_.norm();
    normx_        = Vec::Ones(p_);

    beta_       = Vec::Zero(p_);
    beta_dummy_ = Vec::Zero(opt_.T_stop);
    mu_         = Vec::Zero(n_);
    residuals_  = y_;

    corr_.resize(p_);

    // Initial correlations
    vd_detail::gemv_Xt(X_, residuals_, corr_);

    // Active set
    is_active_.assign(p_, 0);
    actives_.clear();
    actives_.reserve(min_np);
    active_features_.clear();
    active_features_.reserve(min_np);

    // Basis
    basis_.resize(n_, min_np);
    basis_indices_.clear();
    basis_indices_.reserve(min_np);

    // Reusable buffers
    proj_coeffs_.resize(std::max(1, min_np));
    ortho_buffer_.resize(n_);

    // Active columns & Cholesky
    X_active_.resize(n_, min_np);
    chol_factor_.resize(0, 0);

    // Realized dummies
    X_realized_.resize(n_, opt_.T_stop);
    corr_realized_.resize(opt_.T_stop);
    T_realized_ = 0;

    // Virtual-dummy pool
    vd_stick_ = Vec::Ones(L_);
    vd_corr_  = Vec::Zero(L_);
    vd_is_realized_.assign(L_, 0);
    vd_unrealized_idx_.clear();
    vd_unrealized_idx_.reserve(L_);
    vd_rows_cap_ = std::min(opt_.max_vd_proj, std::max(8, 128));
    vd_proj_.resize(vd_rows_cap_, L_);
    vd_rows_filled_ = 0;

    step_ = 0;

    // Seed basis with y, init VD
    initialize_basis_();
    initialize_virtual_dummies_();
}

// ---------- Basis ----------
void VD_Base::initialize_basis_() {
    Eigen::Map<const Vec> yv(y_.data(), n_);
    const double yn = yv.norm();
    if (yn > opt_.eps * std::sqrt(double(n_))) {
        basis_.col(0) = yv / yn;
        basis_indices_.push_back(VD_Y_SENTINEL);
    }
}

Eigen::VectorXi VD_Base::active_indices() const {
    Eigen::VectorXi out(int(actives_.size()));
    for (int i = 0; i < (int)actives_.size(); ++i) out(i) = actives_[i];
    return out;
}

std::optional<Vec> VD_Base::orthonormalize_(const Vec& v) const {
    const int nb = basis_size_();
    if (nb == 0) {
        const double nv = v.norm();
        if (nv <= opt_.eps * std::sqrt(double(n_))) return std::nullopt;
        return Vec(v / nv);
    }
    proj_coeffs_.head(nb).noalias() = basis_.leftCols(nb).transpose() * v;
    ortho_buffer_.noalias() = v - basis_.leftCols(nb) * proj_coeffs_.head(nb);
    const double nu = ortho_buffer_.norm();
    if (nu <= opt_.eps * std::sqrt(double(n_))) return std::nullopt;
    return Vec(ortho_buffer_ / nu);
}

Vec VD_Base::project_to_Vperp_(const Vec& z) const {
    const int nb = basis_size_();
    if (nb == 0) return z;
    Vec coeff = basis_.leftCols(nb).transpose() * z;
    return z - basis_.leftCols(nb) * coeff;
}

// ---------- VD pool ----------
void VD_Base::ensure_vd_rows_capacity_(int need) {
    if (need <= vd_rows_cap_) return;
    int new_cap = std::min(opt_.max_vd_proj,
        std::max(vd_rows_cap_ > 0 ? vd_rows_cap_ * 2 : 128, need));
    MatR tmp(new_cap, L_);
    if (vd_rows_filled_ > 0)
        tmp.topRows(vd_rows_filled_) = vd_proj_.topRows(vd_rows_filled_);
    vd_proj_.swap(tmp);
    vd_rows_cap_ = new_cap;
}

void VD_Base::initialize_virtual_dummies_() {
    ensure_vd_rows_capacity_(1);
    std::fill(vd_is_realized_.begin(), vd_is_realized_.end(), 0);
    vd_unrealized_idx_.clear();
    for (int d = 0; d < L_; ++d) vd_unrealized_idx_.push_back(d);
    vd_rows_filled_ = 1;

    if (opt_.dummy_law == DummyLaw::Spherical) {
        const double b = std::max(0.5 * double(n_ - 2), 1e-8);
        std::normal_distribution<double> N01(0.0, 1.0);
        double* row0 = vd_proj_.row(0).data();
        for (int d = 0; d < L_; ++d) {
            double z = N01(rng_);
            double X = 0.5 * z * z;
            double Y = vd_detail::gamma_mt(b, rng_);
            double U = X / (X + Y);
            double sgn = (rng_() & 1ULL) ? 1.0 : -1.0;
            row0[d] = sgn * std::sqrt(U);
            vd_stick_(d) = 1.0 - row0[d] * row0[d];
        }
    } else {
        std::normal_distribution<double> N01(0.0, 1.0);
        double scale = 1.0 / std::sqrt(double(n_));
        double* row0 = vd_proj_.row(0).data();
        for (int d = 0; d < L_; ++d) {
            row0[d] = scale * N01(rng_);
            vd_stick_(d) = 1.0;
        }
    }
    vd_corr_ = y_norm_ * vd_proj_.row(0).transpose();
}

void VD_Base::update_virtual_dummies_() {
    const int m = basis_size_() - 1;
    if (m < 0 || m >= opt_.max_vd_proj) return;
    ensure_vd_rows_capacity_(m + 1);

    if (opt_.dummy_law == DummyLaw::Gaussian) {
        std::normal_distribution<double> N01(0.0, 1.0);
        double scale = 1.0 / std::sqrt(double(n_));
        double* row_m = vd_proj_.row(m).data();
        for (int d = 0; d < L_; ++d) {
            if (!vd_is_realized_[d]) row_m[d] = scale * N01(rng_);
        }
        if (vd_rows_filled_ < m + 1) vd_rows_filled_ = m + 1;
        return;
    }

    const double b = std::max(0.5 * double(n_ - m - 2), 1e-8);
    if ((int)vd_unrealized_idx_.size() == 0 || vd_rows_filled_ == 0) {
        vd_unrealized_idx_.clear();
        for (int d = 0; d < L_; ++d)
            if (!vd_is_realized_[d] && vd_stick_(d) > opt_.eps)
                vd_unrealized_idx_.push_back(d);
    }
    if (vd_unrealized_idx_.empty()) {
        if (vd_rows_filled_ < m + 1) vd_rows_filled_ = m + 1;
        return;
    }

    std::normal_distribution<double> N01(0.0, 1.0);
    double* row_m = vd_proj_.row(m).data();
    int keep = 0;
    for (int i = 0; i < (int)vd_unrealized_idx_.size(); ++i) {
        const int d = vd_unrealized_idx_[i];
        double z   = N01(rng_);
        double X   = 0.5 * z * z;
        double Y   = vd_detail::gamma_mt(b, rng_);
        double U   = X / (X + Y);
        double sgn = (rng_() & 1ULL) ? 1.0 : -1.0;
        double st  = vd_stick_(d);
        double a   = sgn * std::sqrt(st * U);
        row_m[d] = a;
        double st_new = st - a * a;
        vd_stick_(d) = (st_new > 0.0 ? st_new : 0.0);
        if (!vd_is_realized_[d] && vd_stick_(d) > opt_.eps)
            vd_unrealized_idx_[keep++] = d;
    }
    if (keep != (int)vd_unrealized_idx_.size())
        vd_unrealized_idx_.resize(keep);
    if (vd_rows_filled_ < m + 1) vd_rows_filled_ = m + 1;
}

void VD_Base::realize_dummy_(int vd_idx) {
    const int j = T_realized_;
    const int nb = basis_size_();
    if (nb == 0) return;
    if (vd_proj_.rows() < nb) return;

    Eigen::VectorXd alphas = vd_proj_.topRows(nb).col(vd_idx);
    Vec explained(n_);
    explained.noalias() = basis_.leftCols(nb) * alphas;

    double stick = vd_stick_(vd_idx);
    if (stick < 0.0) stick = 0.0;
    const double r = std::sqrt(stick);

    std::normal_distribution<double> N01(0.0, 1.0);
    Eigen::VectorXd v(n_);
    for (int i = 0; i < n_; ++i) v(i) = N01(rng_);
    double mean = v.mean();
    v.array() -= mean;

    Vec x;
    if (opt_.dummy_law == DummyLaw::Spherical) {
        auto u_opt = orthonormalize_(v);
        if (!u_opt) return;
        x = explained + r * (*u_opt);
    } else {
        const double scale = 1.0 / std::sqrt(double(n_));
        Vec g_perp = project_to_Vperp_(scale * v);
        if (g_perp.norm() < opt_.eps * std::sqrt(double(n_))) return;
        x = explained + g_perp;
    }

    if (X_realized_.cols() <= j)
        X_realized_.conservativeResize(Eigen::NoChange, j + 1);
    X_realized_.col(j) = x;

    if (auto vo = orthonormalize_(x)) {
        const int col = basis_size_();
        basis_.col(col) = *vo;
        basis_indices_.push_back(VD_DUMMY_SENTINEL);
        update_virtual_dummies_();
    }

    if (corr_realized_.size() < j + 1) corr_realized_.conservativeResize(j + 1);
    corr_realized_(j) = x.dot(residuals_);

    if (vd_idx >= 0 && vd_idx < vd_corr_.size()) vd_corr_(vd_idx) = 0.0;
    if (vd_idx >= 0 && vd_idx < (int)vd_is_realized_.size())
        vd_is_realized_[vd_idx] = 1;
    vd_stick_(vd_idx) = 0.0;

    ++T_realized_;
}

// ---------- Cholesky ----------
bool VD_Base::chol_append(MatC& R,
                          const Eigen::Ref<const Vec>& v,
                          double s, double eps)
{
    const int t = R.rows();
    if (t == 0) {
        R.resize(1, 1);
        R(0, 0) = std::sqrt(std::max(s, eps));
        return true;
    }
    Eigen::VectorXd z = v;
    R.transpose().template triangularView<Eigen::Lower>().solveInPlace(z);
    double r2 = s - z.squaredNorm();
    const double floor = std::max(eps, 1e-14 * std::max(1.0, s));
    if (r2 <= floor) r2 = floor;
    R.conservativeResize(t + 1, t + 1);
    R.topRightCorner(t, 1) = z;
    R.row(t).setZero();
    R(t, t) = std::sqrt(r2);
    return true;
}

// ---------- Full correlation refresh ----------
void VD_Base::full_corr_refresh_() {
    const Vec& s = score_direction_();

    // Real features: corr = X^T s
    vd_detail::gemv_Xt(X_, s, corr_);

    // Virtual dummies: vd_corr = A_k^T (E_k^T s)
    const int nb = basis_size_();
    const int m_rows = std::min(nb, vd_rows_filled_);
    if (m_rows > 0 && L_ > 0) {
        Vec basis_proj = basis_.leftCols(nb).transpose() * s;
        vd_corr_.noalias() =
            vd_proj_.topRows(m_rows).transpose() * basis_proj.head(m_rows);
        for (int d = 0; d < L_; ++d)
            if (vd_is_realized_[d]) vd_corr_(d) = 0.0;
    }

    // Realized dummies
    if (T_realized_ > 0) {
        corr_realized_.head(T_realized_).noalias() =
            X_realized_.leftCols(T_realized_).transpose() * s;
    }
}

// ---------- Grow basis from score direction ----------
void VD_Base::grow_basis_from_score_() {
    const Vec& s = score_direction_();
    if (auto vo = orthonormalize_(s)) {
        basis_.col(basis_size_()) = *vo;
        basis_indices_.push_back(VD_DUMMY_SENTINEL);  // generic: not a specific feature
        update_virtual_dummies_();
    }
}

}  // namespace trex::tsolvers::vd
