// ===================================================================================
// sd2_tafs_solver.cpp
// ===================================================================================
#include "sd2_tafs_solver.hpp"
#include <algorithm>
#include <cmath>

namespace trex::tsolvers::linear_model::afs_based {

SD2_TAFS_Solver::SD2_TAFS_Solver(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
                                 std::size_t L_max, std::size_t T_stop, bool intercept,
                                 uint64_t seed, SD2GenPolicy policy, double rho)
    : SD2PairSolver_Base(X, y, L_max, T_stop, intercept, seed, policy) {

    if (rho <= 0.0 || rho > 1.0) {
        throw std::invalid_argument("SD2_TAFS_Solver: rho must be in (0, 1].");
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

SD2_TAFS_Solver::Candidate SD2_TAFS_Solver::findBestNonPool() const {
    Candidate best{};

    // All reals — active ones included (re-selection blends them further).
    for (std::size_t j = 0; j < p_original_; ++j) {
        if (real_state_[j] == kBanned) continue;
        const double ac = std::abs(correlations_(j));
        if (ac > best.abs_corr) best = {ac, j, real_state_[j] == kInactive};
    }
    // Active pairs (one subtraction each; re-selection blends, never
    // re-counts).
    for (const auto& [gj, q] : active_pairs_) {
        const double ac = std::abs(dummy_scale_ * (r_(q.first) - r_(q.second)));
        if (ac > best.abs_corr) best = {ac, gj, false};
    }
    return best;
}

void SD2_TAFS_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    while (currentStep_ < maxSteps_ && actives_.size() < effective_n_ &&
          (!early_stop || count_active_dummies_ < T_stop)) {

        Candidate best = findBestNonPool();

        bool dummy_wins = false;
        double Cmax = best.abs_corr;
        PairDummy winner{};
        std::size_t pool_idx = 0;

        if (policy_ == SD2GenPolicy::OnDemand) {
            expandPool(best.abs_corr);
            double pool_max = 0.0;
            for (std::size_t q = 0; q < pool_.size(); ++q) {
                const double ac = std::abs(pool_[q].corr);
                if (ac > pool_max) { pool_max = ac; pool_idx = q; }
            }
            if (pool_max > best.abs_corr) {
                dummy_wins = true;
                Cmax = pool_max;
            }
        } else {
            if (geometricExpansion(best.abs_corr, winner)) {
                dummy_wins = true;   // a beating pair always exceeds c_ref
                Cmax = std::abs(winner.corr);
            }
        }

        if (Cmax < 100 * eps_) break;

        if (dummy_wins) {
            if (policy_ == SD2GenPolicy::OnDemand) {
                winner = pool_[pool_idx];
                pool_.erase(pool_.begin() + static_cast<std::ptrdiff_t>(pool_idx));
            }
            count_active_dummies_++;

            if (!appendToActiveSet(winner.seed, /*is_dummy=*/true, winner.i, winner.j)) {
                // Collinear with the active set (e.g. a duplicate pair):
                // discard the candidate permanently and continue.
                --count_active_dummies_;
                continue;
            }
        } else if (best.is_new) {
            if (!appendToActiveSet(best.j, /*is_dummy=*/false, -1, -1)) {
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
// Mathematical Core (Cholesky & blended refit) — dot-free on the dummy path
// ==========================================================================

bool SD2_TAFS_Solver::appendToActiveSet(std::size_t winning_j, bool is_dummy,
                                        int pi, int pj) {
    const auto m = static_cast<Eigen::Index>(actives_.size());

    double xty = 0.0;
    if (!pairAppendActive(winning_j, is_dummy, pi, pj, &xty)) {
        return false; // Collinear failure
    }

    Xty_active_.conservativeResize(m + 1);
    Xty_active_(m) = xty;
    return true;
}

void SD2_TAFS_Solver::afsBlend() {
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
        const std::size_t a = actives_[static_cast<std::size_t>(i)];
        if (a < p_original_) {
            Xa_nu += nu(i) * X_->col(static_cast<Eigen::Index>(a));
        } else {
            const auto& q = active_pairs_.at(a);
            Xa_nu(q.first)  += nu(i) * dummy_scale_;
            Xa_nu(q.second) -= nu(i) * dummy_scale_;
        }
    }
    mu_ = (1.0 - rho_) * mu_ + rho_ * Xa_nu;
    r_ = y_ - mu_;

    correlations_.noalias() = X_->transpose() * r_;
    refreshPoolCorrelations();  // no-op under Geometric (pool stays virtual)
}

} // namespace trex::tsolvers::linear_model::afs_based
