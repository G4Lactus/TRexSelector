// ====================================================================
// vd_base.hpp
// ====================================================================
#ifndef TSOLVERS_VD_TSOLVERS_VD_BASE_HPP
#define TSOLVERS_VD_TSOLVERS_VD_BASE_HPP
// ====================================================================
/**
 * @file vd_base.hpp
 *
 * @brief Base class for the VD-* virtual-dummy forward selectors.
 *
 * @details Owns all shared state: data views, VD pool, basis, Cholesky,
 *          model state, scratch buffers, and public accessors.
 *          Solvers (LARS, OMP, AFS) inherit and implement run().
 */
// ====================================================================

// VD includes
#include <tsolvers/vd_tsolvers/vd_common.hpp>

// ====================================================================

// Embedded into namespace trex::tsolvers::vd
namespace trex::tsolvers::vd {

class VD_Base {
public:
    // ---------- Constructors ----------
    VD_Base(const double* Xptr, int n, int p,
            const double* yptr, int ny,
            int num_dummies, const VDOptions& o);

    VD_Base(const Eigen::Ref<const MatC>& X_in,
            const Eigen::Ref<const Vec>&  y_in,
            int num_dummies, const VDOptions& o);

    virtual ~VD_Base() = default;

    // Pure virtual: each solver implements its own run loop.
    virtual MatC run(int T = 1) = 0;

    // ---------- Public accessors ----------
    int num_dummies()          const noexcept { return L_; }
    int num_realized_dummies() const noexcept { return T_realized_; }
    int n_features()           const noexcept { return p_; }
    int n_samples()            const noexcept { return n_; }
    int basis_size()           const noexcept { return basis_size_(); }

    Eigen::VectorXd beta_view_copy()   const { return beta_; }
    Eigen::VectorXd corr_view_copy()   const { return corr_; }
    Eigen::VectorXi active_indices()   const;

    Eigen::VectorXd corr_realized_view_copy() const {
        return corr_realized_.head(T_realized_);
    }
    Eigen::VectorXi is_dummy_realized_view() const {
        Eigen::VectorXi out(L_);
        for (int d = 0; d < L_; ++d) out(d) = vd_is_realized_[d] ? 1 : 0;
        return out;
    }

    std::vector<ActiveFeature> active_features_copy() const { return active_features_; }

    Eigen::VectorXd beta_real() const {
        Eigen::VectorXd out(p_);
        if (beta_.size() >= p_) out = beta_.head(p_);
        else                    out.setZero();
        return out;
    }

    const Vec&  normx_view()    const noexcept { return normx_; }
    const MatR& vd_proj_view()  const noexcept { return vd_proj_; }
    const Vec&  vd_corr_view()  const noexcept { return vd_corr_; }
    const Vec&  vd_stick_view() const noexcept { return vd_stick_; }

protected:
    // ---------- Problem sizes & options ----------
    int n_ = 0, p_ = 0, L_ = 0;
    VDOptions opt_;

    // ---------- Data (mapped views) ----------
    MapMatC X_;
    MapVec  y_;

    // ---------- RNG ----------
    std::mt19937_64 rng_{0};

    // ---------- Global scalars ----------
    bool   standardized_ = false;
    double y_norm_ = 0.0;

    // ---------- Model state ----------
    Vec normx_;
    Vec beta_, beta_dummy_, mu_, residuals_, corr_;

    // ---------- Active set ----------
    std::vector<int> actives_;                    // real-feature indices
    std::vector<ActiveFeature> active_features_;  // all active (real + dummy)
    std::vector<char> is_active_;                 // O(1) lookup for reals

    // ---------- Basis ----------
    MatC basis_;
    std::vector<int> basis_indices_;
    inline int basis_size_() const noexcept {
        return static_cast<int>(basis_indices_.size());
    }

    // ---------- Reusable buffers ----------
    mutable Vec proj_coeffs_;
    mutable Vec ortho_buffer_;

    // ---------- Active columns & Cholesky ----------
    MatC X_active_;
    MatC chol_factor_;

    // ---------- Realized dummies ----------
    MatC X_realized_;
    Vec  corr_realized_;
    int  T_realized_ = 0;

    // ---------- Virtual-dummy pool ----------
    MatR vd_proj_;
    Vec  vd_stick_, vd_corr_;
    std::vector<char> vd_is_realized_;
    std::vector<int>  vd_unrealized_idx_;
    int vd_rows_filled_ = 0, vd_rows_cap_ = 0;

    // ---------- Bookkeeping ----------
    int step_ = 0;

    // ---------- Shared setup ----------
    void init_common_state_();    // alloc + init everything above
    void initialize_basis_();
    void ensure_vd_rows_capacity_(int need);

    // ---------- Shared VD machinery ----------
    std::optional<Vec> orthonormalize_(const Vec& v) const;
    Vec  project_to_Vperp_(const Vec& z) const;
    void initialize_virtual_dummies_();
    void update_virtual_dummies_();
    void realize_dummy_(int vd_idx);

    // ---------- Score direction (the hook for GLMs) ----------
    //
    // Returns the direction used for:
    //   (a) growing the basis (orthonormalize this, not x_j)
    //   (b) computing correlations (corr_ = X^T score, not X^T r)
    //
    // Default (OLS): returns residuals_ = y - X_A beta.
    // GLM override:  returns score s_k = y - g(X_A beta_k).
    //
    // Called by grow_basis_and_refresh_() after each selection step.
    //
    virtual const Vec& score_direction_() const { return residuals_; }

    // ---------- Combined basis growth + correlation refresh ----------
    //
    // After selecting a new feature (real or dummy), call this instead
    // of manually calling orthonormalize_ + update_virtual_dummies_ +
    // full_corr_refresh_ separately.  Uses score_direction_() to
    // determine what enters the basis and what gets correlated.
    //
    void grow_basis_from_score_();
    void full_corr_refresh_();    // corr_ = X^T score, vd_corr, corr_realized

    // ---------- Shared Cholesky ----------
    static bool chol_append(MatC& R,
                            const Eigen::Ref<const Vec>& v,
                            double s, double eps);

private:
    VD_Base(const VD_Base&) = delete;
    VD_Base& operator=(const VD_Base&) = delete;
};

}  // namespace trex::tsolvers::vd

// ====================================================================
#endif /* TSOLVERS_VD_TSOLVERS_VD_BASE_HPP */
// ====================================================================
