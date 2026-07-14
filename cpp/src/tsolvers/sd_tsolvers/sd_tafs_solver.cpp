// ===================================================================================
// sd_tafs_solver.cpp
// ===================================================================================
#include "sd_tafs_solver.hpp"
#include <algorithm>
#include <cmath>

namespace trex::tsolvers::linear_model::afs_based {

SD_TAFS_Solver::SD_TAFS_Solver(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
                               double rho_d, std::size_t L_max, std::size_t T_stop, bool intercept,
                               uint64_t seed, double rho)
    : SDGeneralSolver_Base(X, y, rho_d, L_max, T_stop, intercept, seed) {

    if (rho <= 0.0 || rho > 1.0) {
        throw std::invalid_argument("SD_TAFS_Solver: rho must be in (0, 1].");
    }
    rho_ = rho;

    r_ = y_;
    mu_ = Eigen::VectorXd::Zero(y_.size());
    // Re-selection steps do not grow the active set, so the step budget is
    // scaled by 1/rho relative to the greedy solvers.
    maxSteps_ = static_cast<std::size_t>(std::ceil(8.0 / rho_)) *
                std::min(p_original_, effective_n_);
    correlations_ = X_->transpose() * r_;
    real_state_.assign(p_original_, kInactive);
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

        // c_ref is the best non-pool candidate (reals and active dummies):
        // fresh dummies must beat it to win.
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
            if (dummyCacheFull()) break;

            const std::size_t winning_j = pool_Q_[pool_idx].seed;
            VirtualDummy d = admitPoolWinner(winning_j);

            if (!appendToActiveSet(winning_j)) {
                // Collinear with the active set (e.g. a duplicate sparse
                // dummy): discard the candidate permanently and continue.
                rollbackDummyAdmission(winning_j);
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
    const auto m = static_cast<Eigen::Index>(actives_.size());

    double xty = 0.0;
    if (!appendActiveColumn(winning_j, &xty)) return false; // Collinear failure

    Xty_active_.conservativeResize(m + 1);
    Xty_active_(m) = xty;
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
