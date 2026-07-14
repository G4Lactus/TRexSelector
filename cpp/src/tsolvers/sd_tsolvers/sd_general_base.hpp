// ===================================================================================
// sd_general_base.hpp
// ===================================================================================
#ifndef TSOLVERS_SD_GENERAL_BASE_HPP
#define TSOLVERS_SD_GENERAL_BASE_HPP
// ===================================================================================
/**
 * @file sd_general_base.hpp
 *
 * @brief Family base of the general-k SD solvers (SD_TLARS / SD_TOMP /
 * SD_TAFS): the explicit virtual-dummy pool raced on demand against the
 * originals, the materialized cache for winning dummies, and the shared
 * auto-calibration/budget resolution (rho_d = 0, L_max = 0).
 *
 * @details The pool race is identical across the three algorithms: fresh
 * balanced sparse Rademacher dummies are generated (identical RNG draw
 * pattern in every solver, so all race the same dummy stream for the same
 * seed) until one beats the reference bar or the budget L_max is exhausted.
 * Fixed-L race semantics: the only cap is the total budget L_max — every
 * step therefore races against (effectively) all L dummies, exactly like
 * the classic pre-filled dummy matrix. (The earlier milestone ladder
 * min(m*p, L_max) under-supplied dummies in early steps relative to the L
 * the FDP estimator assumes, which was anti-conservative whenever L > p.)
 */
// ===================================================================================

#include "sd_tsolver_base.hpp"
#include "sd_calibration.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>
#include <unordered_map>

namespace trex::tsolvers {

class SDGeneralSolver_Base : public SDTSolver_Base {
protected:
    // --- Virtual Dummy Mechanics ---
    struct VirtualDummy {
        uint64_t seed;
        std::vector<int> P_indices;
        std::vector<int> M_indices;
        double current_correlation{0.0};
        double current_a{0.0};  // <d, u> of the last step (for c -= gamma*a)
    };

    std::vector<VirtualDummy> pool_Q_;
    Eigen::MatrixXd D_materialized_;
    std::unordered_map<std::size_t, Eigen::Index> active_dummy_map_;
    Eigen::Index current_materialized_cols_{0};

    std::size_t L_max_;
    uint64_t virtual_seed_counter_{0}; // Tracks global dummy index j

    // Set when the ctor ran auto-calibration (rho_d == 0).
    std::optional<sd_calibration::Result> auto_calibration_{};

    // ==========================================================================
    // Unified Inline Accessor (Zero Virtual Overhead)
    // ==========================================================================
    inline Eigen::Ref<const Eigen::VectorXd> getColumn(std::size_t j) const {
        if (j < p_original_) {
            return X_->col(static_cast<Eigen::Index>(j));
        } else {
            return D_materialized_.col(active_dummy_map_.at(j));
        }
    }

    // ==========================================================================
    // Materialization Logic
    // ==========================================================================
    void materializeDummy(const VirtualDummy& dummy, std::size_t global_j) {
        auto col_view = D_materialized_.col(current_materialized_cols_);
        for (int idx : dummy.P_indices) col_view(idx) =  dummy_scale_;
        for (int idx : dummy.M_indices) col_view(idx) = -dummy_scale_;
        active_dummy_map_[global_j] = current_materialized_cols_++;
    }

    /** @brief True when the winner cache cannot admit another dummy
     *  (executeStep called beyond the ctor's T_stop capacity). */
    bool dummyCacheFull() const {
        return current_materialized_cols_ == D_materialized_.cols();
    }

    /**
     * @brief Admit the pool dummy with global index @p winning_j into the
     * active-dummy cache: materialize its column, remove it from the pool,
     * and count it toward the early-stopping threshold T.
     *
     * @return The admitted dummy's record (index sets), for solvers that
     *         keep racing active dummies (AFS re-selection).
     */
    VirtualDummy admitPoolWinner(std::size_t winning_j) {
        auto it = std::find_if(pool_Q_.begin(), pool_Q_.end(),
                               [winning_j](const VirtualDummy& d) {
                                   return d.seed == winning_j;
                               });
        VirtualDummy d = std::move(*it);
        materializeDummy(d, winning_j);
        pool_Q_.erase(it);
        count_active_dummies_++;
        return d;
    }

    /** @brief Undo admitPoolWinner() after a collinear append failure:
     *  discard the candidate permanently and free its cache column. */
    void rollbackDummyAdmission(std::size_t winning_j) {
        active_dummy_map_.erase(winning_j);
        --current_materialized_cols_;
        D_materialized_.col(current_materialized_cols_).setZero();
        --count_active_dummies_;
    }

    // ==========================================================================
    // Pool Race (shared across LARS / OMP / AFS)
    // ==========================================================================
    /** @brief Draw a fresh balanced sparse Rademacher dummy. Partial
     *  Fisher-Yates shuffle — identical draw pattern in every general
     *  solver, so all race the same dummy stream for the same seed. */
    VirtualDummy generateVirtualDummy(uint64_t seed) {
        VirtualDummy dummy;
        dummy.seed = seed;
        dummy.P_indices.reserve(k_sparse_);
        dummy.M_indices.reserve(k_sparse_);

        std::vector<int> candidates(y_.size());
        std::iota(candidates.begin(), candidates.end(), 0);

        for (std::size_t i = 0; i < 2 * k_sparse_; ++i) {
            std::uniform_int_distribution<std::size_t> dist(i, candidates.size() - 1);
            std::swap(candidates[i], candidates[dist(dummy_rng_)]);
        }

        dummy.P_indices.assign(candidates.begin(),
                               candidates.begin() + static_cast<std::ptrdiff_t>(k_sparse_));
        dummy.M_indices.assign(candidates.begin() + static_cast<std::ptrdiff_t>(k_sparse_),
                               candidates.begin() + static_cast<std::ptrdiff_t>(2 * k_sparse_));
        return dummy;
    }

    /**
     * @brief Expand the pool with fresh dummies until one beats @p c_ref
     * (the reference bar: max inactive-real correlation for LARS/OMP, best
     * non-pool candidate for AFS) or the total budget L_max is exhausted.
     * Generation stops at the first beater (on-demand economy).
     */
    void expandPool(double c_ref) {
        double c_Q_max = 0.0;
        for (const auto& dummy : pool_Q_) {
            c_Q_max = std::max(c_Q_max, std::abs(dummy.current_correlation));
        }

        while (c_Q_max <= c_ref && virtual_seed_counter_ < L_max_) {
            uint64_t next_j = dummy_start_idx_ + virtual_seed_counter_++;
            VirtualDummy new_dummy = generateVirtualDummy(next_j);

            double c_new = 0.0;
            for (int idx : new_dummy.P_indices) c_new += r_(idx);
            for (int idx : new_dummy.M_indices) c_new -= r_(idx);
            c_new *= dummy_scale_;
            new_dummy.current_correlation = c_new;

            c_Q_max = std::max(c_Q_max, std::abs(c_new));
            pool_Q_.push_back(std::move(new_dummy));
        }
    }

    /** @brief Recompute all pool correlations from the residual (exact, no
     *  drift): c_j = d^T r is an O(k) index-set sum per dummy. Used by the
     *  refit solvers (OMP/AFS); LARS maintains them incrementally. */
    void refreshPoolCorrelations() {
        #pragma omp parallel for
        for (std::size_t i = 0; i < pool_Q_.size(); ++i) {
            double c = 0.0;
            for (int idx : pool_Q_[i].P_indices) c += r_(idx);
            for (int idx : pool_Q_[i].M_indices) c -= r_(idx);
            pool_Q_[i].current_correlation = c * dummy_scale_;
        }
    }

    /** @brief Argmax of |correlation| over inactive reals and the pool.
     *  @return (Cmax, winning global index, winner is a dummy). */
    std::tuple<double, std::size_t, bool> findGlobalWinner() const {
        double Cmax = 0.0;
        std::size_t winning_j = 0;
        bool is_dummy = false;

        for (std::size_t j : inactives_) {
            if (std::abs(correlations_(j)) > Cmax) {
                Cmax = std::abs(correlations_(j));
                winning_j = j;
            }
        }
        for (const auto& dummy : pool_Q_) {
            if (std::abs(dummy.current_correlation) > Cmax) {
                Cmax = std::abs(dummy.current_correlation);
                winning_j = dummy.seed; // seed acts as the global j tracking index
                is_dummy = true;
            }
        }
        return {Cmax, winning_j, is_dummy};
    }

    /**
     * @brief Grow the active set by column @p winning_j (Cholesky append via
     * the materialized cache).
     *
     * @param xty_out When non-null, receives x_new^T y (maintained by the
     *                refit solvers for their OLS solves; LARS passes null).
     *
     * @return false on collinear failure (state untouched).
     */
    bool appendActiveColumn(std::size_t winning_j, double* xty_out = nullptr) {
        const std::size_t m = actives_.size();
        const auto xnew = getColumn(winning_j);
        const double xtx = xnew.dot(xnew);

        Eigen::VectorXd cross_prod(static_cast<Eigen::Index>(m));
        for (std::size_t k = 0; k < m; ++k) {
            cross_prod(static_cast<Eigen::Index>(k)) = xnew.dot(getColumn(actives_[k]));
        }

        if (!tryAppendCholesky(xtx, cross_prod)) return false;

        actives_.push_back(winning_j);
        if (xty_out != nullptr) *xty_out = xnew.dot(y_);
        return true;
    }

public:
    /**
     * @brief Shared construction of the general-k family.
     *
     * @param rho_d Dummy non-zero fraction (2k ~= rho_d * n). Pass 0 to
     *              auto-calibrate k from the data via sd_calibration
     *              (pilot screen + MC; seconds at large p). The result is
     *              readable via getAutoCalibration().
     * @param L_max Total dummy budget. Pass 0 for the auto budget:
     *              the calibration's L when rho_d == 0, else 2p.
     */
    SDGeneralSolver_Base(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
                         double rho_d, std::size_t L_max, std::size_t T_stop,
                         bool intercept, uint64_t seed)
        : SDTSolver_Base(X, y, rho_d, T_stop, intercept, seed), L_max_(L_max) {

        // rho_d == 0: auto-calibrate the dummy sparsity (and, when L_max == 0,
        // the budget) from the data. Explicit rho_d with L_max == 0 gets the
        // default budget 2p without running the calibration.
        if (rho_d == 0.0) {
            sd_calibration::Options copt;
            if (L_max_ != 0) copt.L = L_max_;
            auto_calibration_ = sd_calibration::calibrate(*X_, y_, copt);
            configureSparsity(auto_calibration_->rho_d);
            if (L_max_ == 0) L_max_ = auto_calibration_->L;
        } else if (L_max_ == 0) {
            L_max_ = 2 * p_original_;
        }

        D_materialized_ = Eigen::MatrixXd::Zero(X.rows(),
                                                static_cast<Eigen::Index>(T_stop_limit_));
        pool_Q_.reserve(p_original_ * 2);
    }

    // ==========================================================================
    // Read-only accessors
    // ==========================================================================
    std::size_t getNumGeneratedDummies() const { return virtual_seed_counter_; }
    std::size_t getPoolSize() const { return pool_Q_.size(); }
    std::size_t getLMax() const { return L_max_; }
    const std::optional<sd_calibration::Result>& getAutoCalibration() const {
        return auto_calibration_;
    }
};

} // namespace trex::tsolvers
#endif
