// ===================================================================================
// sd_tafs_solver.cpp
// ===================================================================================
#include "sd_tafs_solver.hpp"
#include <numeric>
#include <algorithm>
#include <cmath>

namespace trex::tsolvers::linear_model::afs_based {

SD_TAFS_Solver::SD_TAFS_Solver(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
                               double rho_d, std::size_t L_max, std::size_t T_stop, bool intercept,
                               uint64_t seed, double rho)
    : SDTSolver_Base(X, y, rho_d, T_stop, intercept, seed), L_max_(L_max) {

    if (rho <= 0.0 || rho > 1.0) {
        throw std::invalid_argument("SD_TAFS_Solver: rho must be in (0, 1].");
    }
    rho_ = rho;

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

    r_ = y_;
    mu_ = Eigen::VectorXd::Zero(y_.size());
    // Re-selection steps do not grow the active set, so the step budget is
    // scaled by 1/rho relative to the greedy solvers.
    maxSteps_ = static_cast<std::size_t>(std::ceil(8.0 / rho_)) *
                std::min(p_original_, effective_n_);
    correlations_ = X_->transpose() * r_;
    real_state_.assign(p_original_, kInactive);
    pool_Q_.reserve(p_original_ * 2);
}

SDTSolver_Base::VirtualDummy SD_TAFS_Solver::generateVirtualDummy(uint64_t seed) {
    VirtualDummy dummy;
    dummy.seed = seed;
    dummy.P_indices.reserve(k_sparse_);
    dummy.M_indices.reserve(k_sparse_);

    std::vector<int> candidates(y_.size());
    std::iota(candidates.begin(), candidates.end(), 0);

    // Partial Fisher-Yates shuffle (identical draw pattern to SD_TLARS, so
    // both solvers race the same dummy stream for the same seed).
    for(std::size_t i = 0; i < 2 * k_sparse_; ++i) {
        std::uniform_int_distribution<std::size_t> dist(i, candidates.size() - 1);
        std::swap(candidates[i], candidates[dist(dummy_rng_)]);
    }

    dummy.P_indices.assign(candidates.begin(), candidates.begin() + k_sparse_);
    dummy.M_indices.assign(candidates.begin() + k_sparse_, candidates.begin() + 2 * k_sparse_);
    return dummy;
}

// Pool correlations are recomputed from the residual after every blend
// (exact, no drift): c_j = d^T r is an O(k) index-set sum per dummy.
void SD_TAFS_Solver::refreshPoolCorrelations() {
    #pragma omp parallel for
    for (std::size_t i = 0; i < pool_Q_.size(); ++i) {
        double c = 0.0;
        for (int idx : pool_Q_[i].P_indices) c += r_(idx);
        for (int idx : pool_Q_[i].M_indices) c -= r_(idx);
        pool_Q_[i].current_correlation = c * dummy_scale_;
    }
}

// c_ref is the best non-pool candidate (reals and active dummies): fresh
// dummies must beat it to win, so generation stops at the first beater.
void SD_TAFS_Solver::expandPool(double c_ref) {
    double c_Q_max = 0.0;
    for (const auto& dummy : pool_Q_) {
        c_Q_max = std::max(c_Q_max, std::abs(dummy.current_correlation));
    }

    // Fixed-L race semantics: generation stops at the first beater, and the
    // only cap is the total budget L_max — every step therefore races against
    // (effectively) all L dummies, exactly like the classic pre-filled dummy
    // matrix. (The earlier milestone ladder min(m*p, L_max) under-supplied
    // dummies in early steps relative to the L the FDP estimator assumes,
    // which was anti-conservative whenever L > p.)
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

SD_TAFS_Solver::Candidate SD_TAFS_Solver::findBestNonPool() const {
    Candidate best{};

    // All reals — active ones included (re-selection blends them further).
    for (std::size_t j = 0; j < p_original_; ++j) {
        if (real_state_[j] == kBanned) continue;
        const double ac = std::abs(correlations_(j));
        if (ac > best.abs_corr) best = {ac, j, real_state_[j] == kInactive};
    }
    // Active dummies (index-set sums; re-selection blends, never re-counts).
    for (const auto& rec : active_dummy_recs_) {
        double c = 0.0;
        for (int idx : rec.P_indices) c += r_(idx);
        for (int idx : rec.M_indices) c -= r_(idx);
        c *= dummy_scale_;
        const double ac = std::abs(c);
        if (ac > best.abs_corr) best = {ac, static_cast<std::size_t>(rec.seed), false};
    }
    return best;
}

void SD_TAFS_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    while (currentStep_ < maxSteps_ && actives_.size() < effective_n_ &&
          (!early_stop || count_active_dummies_ < T_stop)) {

        Candidate best = findBestNonPool();
        expandPool(best.abs_corr);

        double pool_max = 0.0;
        std::size_t pool_idx = 0;
        for (std::size_t q = 0; q < pool_Q_.size(); ++q) {
            const double ac = std::abs(pool_Q_[q].current_correlation);
            if (ac > pool_max) { pool_max = ac; pool_idx = q; }
        }
        const bool pool_wins = pool_max > best.abs_corr;
        const double Cmax = pool_wins ? pool_max : best.abs_corr;

        if (Cmax < 100 * eps_) break;

        if (pool_wins) {
            // Dummy cache exhausted (executeStep called beyond the ctor's
            // T_stop capacity): cannot admit further dummies.
            if (current_materialized_cols_ == D_materialized_.cols()) break;

            VirtualDummy d = pool_Q_[pool_idx];
            materializeDummy(d, d.seed);
            pool_Q_.erase(pool_Q_.begin() + static_cast<std::ptrdiff_t>(pool_idx));
            count_active_dummies_++;

            if (!appendToActiveSet(d.seed)) {
                // Collinear with the active set (e.g. a duplicate sparse
                // dummy): discard the candidate permanently and continue.
                active_dummy_map_.erase(d.seed);
                --current_materialized_cols_;
                D_materialized_.col(current_materialized_cols_).setZero();
                --count_active_dummies_;
                continue;
            }
            active_dummy_recs_.push_back(std::move(d));
        } else if (best.is_new) {
            if (!appendToActiveSet(best.j)) {
                real_state_[best.j] = kBanned;
                continue;
            }
            real_state_[best.j] = kActive;
        }
        // Re-selected active feature: no append, blend only.

        currentStep_++;
        lambda_.push_back(Cmax);

        afsBlend();
    }
}

// ==========================================================================
// Mathematical Core (Cholesky append & blended refit)
// ==========================================================================

bool SD_TAFS_Solver::appendToActiveSet(std::size_t winning_j) {
    const std::size_t m = actives_.size();
    const auto xnew = getColumn(winning_j);
    const double xtx = xnew.dot(xnew);

    if (m == 0) {
        R_ = Eigen::MatrixXd::Constant(1, 1, std::sqrt(xtx));
    } else {
        Eigen::VectorXd cross_prod(m);
        for (std::size_t k = 0; k < m; ++k) {
            cross_prod(k) = xnew.dot(getColumn(actives_[k]));
        }

        Eigen::VectorXd r_vec = R_.transpose().triangularView<Eigen::Lower>().solve(cross_prod);
        double rpp_sq = xtx - r_vec.dot(r_vec);

        if (rpp_sq < eps_ * xtx) return false; // Collinear failure

        Eigen::MatrixXd newR = Eigen::MatrixXd::Zero(m + 1, m + 1);
        newR.topLeftCorner(m, m) = R_;
        newR.block(0, m, m, 1) = r_vec;
        newR(m, m) = std::sqrt(rpp_sq);
        R_ = newR;
    }

    actives_.push_back(winning_j);
    Xty_active_.conservativeResize(static_cast<Eigen::Index>(m) + 1);
    Xty_active_(static_cast<Eigen::Index>(m)) = xnew.dot(y_);
    return true;
}

void SD_TAFS_Solver::afsBlend() {
    const auto m = static_cast<Eigen::Index>(actives_.size());

    // nu = OLS(A) via the Cholesky factor R (upper).
    Eigen::VectorXd nu = Xty_active_;
    R_.transpose().triangularView<Eigen::Lower>().solveInPlace(nu);
    R_.triangularView<Eigen::Upper>().solveInPlace(nu);

    beta_active_.resize(actives_.size(), 0.0);
    for (Eigen::Index i = 0; i < m; ++i) {
        auto& b = beta_active_[static_cast<std::size_t>(i)];
        b = (1.0 - rho_) * b + rho_ * nu(i);
    }
    betaPathCompact_.push_back(beta_active_);

    Eigen::VectorXd Xa_nu = Eigen::VectorXd::Zero(y_.size());
    for (Eigen::Index i = 0; i < m; ++i) {
        Xa_nu += nu(i) * getColumn(actives_[static_cast<std::size_t>(i)]);
    }
    mu_ = (1.0 - rho_) * mu_ + rho_ * Xa_nu;
    r_ = y_ - mu_;

    correlations_.noalias() = X_->transpose() * r_;
    refreshPoolCorrelations();
}

} // namespace trex::tsolvers::linear_model::afs_based
