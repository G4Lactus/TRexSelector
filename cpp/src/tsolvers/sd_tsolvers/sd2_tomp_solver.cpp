// ===================================================================================
// sd2_tomp_solver.cpp
// ===================================================================================
#include "sd2_tomp_solver.hpp"
#include <algorithm>

namespace trex::tsolvers::linear_model::omp_based {

SD2_TOMP_Solver::SD2_TOMP_Solver(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
                                 std::size_t L_max, std::size_t T_stop, bool intercept,
                                 uint64_t seed, SD2GenPolicy policy)
    : SD2PairSolver_Base(X, y, L_max, T_stop, intercept, seed, policy) {

    r_ = y_;
    mu_ = Eigen::VectorXd::Zero(y_.size());
    maxSteps_ = 8 * std::min(p_original_, effective_n_);
    correlations_ = X_->transpose() * r_;

    for (std::size_t j = 0; j < p_original_; ++j) inactives_.push_back(j);
}

void SD2_TOMP_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    while (currentStep_ < maxSteps_ && actives_.size() < effective_n_ &&
          (!early_stop || count_active_dummies_ < T_stop)) {

        double Cmax = 0.0;
        std::size_t winning_j = 0;
        bool is_dummy = false;
        std::size_t pool_idx = 0;
        PairDummy geom_winner{};

        if (policy_ == SD2GenPolicy::OnDemand) {
            expandPool(inactivesMaxAbsCorr());
            std::tie(Cmax, winning_j, is_dummy, pool_idx) = findGlobalWinner();
        } else {
            for (std::size_t j : inactives_) {
                if (std::abs(correlations_(j)) > Cmax) {
                    Cmax = std::abs(correlations_(j));
                    winning_j = j;
                }
            }
            if (geometricExpansion(Cmax, geom_winner)) {
                is_dummy = true;
                winning_j = geom_winner.seed;
                Cmax = std::abs(geom_winner.corr);
            }
        }

        if (Cmax < 100 * eps_) break;

        int pi = -1, pj = -1;
        if (is_dummy) {
            if (policy_ == SD2GenPolicy::OnDemand) {
                pi = pool_[pool_idx].i;
                pj = pool_[pool_idx].j;
                pool_.erase(pool_.begin() + static_cast<std::ptrdiff_t>(pool_idx));
            } else {
                pi = geom_winner.i;
                pj = geom_winner.j;
            }
            count_active_dummies_++;
        } else {
            std::erase(inactives_, winning_j);
        }

        if (!appendToActiveSet(winning_j, is_dummy, pi, pj)) {
            // Collinear with the active set (e.g. a duplicate pair — the
            // universe has only C(n,2) distinct dummies): discard the
            // candidate permanently and continue the path.
            if (is_dummy) --count_active_dummies_;
            continue;
        }

        currentStep_++;
        lambda_.push_back(Cmax);

        olsRefit();
    }
}

// ==========================================================================
// Mathematical Core (Cholesky & OLS refit) — dot-free on the dummy path
// ==========================================================================

bool SD2_TOMP_Solver::appendToActiveSet(std::size_t winning_j, bool is_dummy,
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

void SD2_TOMP_Solver::olsRefit() {
    const auto m = static_cast<Eigen::Index>(actives_.size());

    // z = (X_A^T X_A)^{-1} X_A^T y via the Cholesky factor R (upper).
    Eigen::VectorXd z = Xty_active_;
    R_.transpose().triangularView<Eigen::Lower>().solveInPlace(z);
    R_.triangularView<Eigen::Upper>().solveInPlace(z);

    // OMP overwrites the coefficients each step (full refit).
    beta_active_.resize(actives_.size());
    for (Eigen::Index i = 0; i < m; ++i) beta_active_[static_cast<std::size_t>(i)] = z(i);
    betaPathCompact_.push_back(beta_active_);

    mu_.setZero();
    for (Eigen::Index i = 0; i < m; ++i) {
        const std::size_t a = actives_[static_cast<std::size_t>(i)];
        if (a < p_original_) {
            mu_ += z(i) * X_->col(static_cast<Eigen::Index>(a));
        } else {
            const auto& q = active_pairs_.at(a);
            mu_(q.first)  += z(i) * dummy_scale_;
            mu_(q.second) -= z(i) * dummy_scale_;
        }
    }
    r_ = y_ - mu_;

    correlations_.noalias() = X_->transpose() * r_;
    refreshPoolCorrelations();  // no-op under Geometric (pool stays virtual)
}

} // namespace trex::tsolvers::linear_model::omp_based
