// ===================================================================================
// sd2_tlars_solver.cpp
// ===================================================================================
#include "sd2_tlars_solver.hpp"
#include <algorithm>
#include <cmath>

namespace trex::tsolvers::linear_model::lars_based {

SD2_TLARS_Solver::SD2_TLARS_Solver(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
                                   std::size_t L_max, std::size_t T_stop, bool intercept,
                                   uint64_t seed, SD2GenPolicy policy)
    : SD2PairSolver_Base(X, y, L_max, T_stop, intercept, seed, policy) {

    r_ = y_;
    maxSteps_ = 8 * std::min(p_original_, effective_n_);
    correlations_ = X_->transpose() * r_;

    for (std::size_t j = 0; j < p_original_; ++j) inactives_.push_back(j);
}

void SD2_TLARS_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    while (currentStep_ < maxSteps_ && actives_.size() < effective_n_ &&
          (!early_stop || count_active_dummies_ < T_stop)) {

        double Cmax = 0.0;
        std::size_t winning_j = 0;
        bool is_dummy = false;
        std::size_t pool_idx = 0;
        PairDummy geom_winner{};

        if (policy_ == SD2GenPolicy::OnDemand) {
            // Pool correlations are maintained incrementally at the end of
            // each step (c_j -= gamma * a_j, exact since c_j = d^T r).
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
        double winner_corr;
        if (is_dummy) {
            if (policy_ == SD2GenPolicy::OnDemand) {
                pi = pool_[pool_idx].i;
                pj = pool_[pool_idx].j;
                winner_corr = pool_[pool_idx].corr;
                pool_.erase(pool_.begin() + static_cast<std::ptrdiff_t>(pool_idx));
            } else {
                pi = geom_winner.i;
                pj = geom_winner.j;
                winner_corr = geom_winner.corr;
            }
            count_active_dummies_++;
        } else {
            winner_corr = correlations_(winning_j);
            std::erase(inactives_, winning_j);
        }

        if (!updateActiveSet(winning_j, is_dummy, pi, pj, winner_corr)) {
            // Collinear with the active set (e.g. a duplicate pair — the
            // universe has only C(n,2) distinct dummies): discard the
            // candidate permanently and continue the path.
            if (is_dummy) --count_active_dummies_;
            continue;
        }

        currentStep_++;
        lambda_.push_back(Cmax);

        Eigen::VectorXd sign_vec = signVector();
        Eigen::VectorXd w_A = equiangularDirection(sign_vec);
        Eigen::VectorXd u = equiangularVector(w_A);

        // Not a structured binding: clang's OpenMP cannot capture those in the
        // parallel correlation update below.
        std::pair<double, Eigen::VectorXd> step = computeStepSize(Cmax, u);
        const double gamma = step.first;
        const Eigen::VectorXd& a = step.second;

        // Update sparse beta path and residual
        beta_active_.resize(actives_.size(), 0.0);
        for (std::size_t k = 0; k < actives_.size(); ++k) {
            beta_active_[k] += gamma * w_A(k);
        }
        betaPathCompact_.push_back(beta_active_);

        r_ -= gamma * u;

        // Fast Correlation Update (Standard Variables)
        #pragma omp parallel for
        for (std::size_t i = 0; i < inactives_.size(); ++i) {
            std::size_t j = inactives_[i];
            correlations_(j) -= gamma * a(i);
        }

        // Fast Correlation Update (Pool Dummies): exact, since c_j = d^T r
        // and r <- r - gamma*u  =>  c_j <- c_j - gamma * <d, u>.
        // (No-op under the Geometric policy: the pool stays virtual.)
        #pragma omp parallel for
        for (std::size_t i = 0; i < pool_.size(); ++i) {
            pool_[i].corr -= gamma * pool_[i].current_a;
        }
    }
}

// ==========================================================================
// Mathematical Core (LARS geometry) — dot-free on the dummy path
// ==========================================================================

bool SD2_TLARS_Solver::updateActiveSet(std::size_t winning_j, bool is_dummy,
                                       int pi, int pj, double corr) {
    if (!pairAppendActive(winning_j, is_dummy, pi, pj)) return false;

    Sign_.push_back((corr >= 0) ? 1 : -1);
    return true;
}

Eigen::VectorXd SD2_TLARS_Solver::signVector() const {
    Eigen::VectorXd s(Sign_.size());
    for (std::size_t i = 0; i < Sign_.size(); ++i) s(i) = Sign_[i];
    return s;
}

Eigen::VectorXd SD2_TLARS_Solver::equiangularDirection(const Eigen::Ref<const Eigen::VectorXd>& sign) {
    Eigen::VectorXd GAi_s = R_.triangularView<Eigen::Upper>().solve(
                            R_.transpose().triangularView<Eigen::Lower>().solve(sign));
    A_A_ = 1.0 / std::sqrt(sign.dot(GAi_s));
    return A_A_ * GAi_s;
}

Eigen::VectorXd SD2_TLARS_Solver::equiangularVector(const Eigen::Ref<const Eigen::VectorXd>& w_A) const {
    Eigen::VectorXd u = Eigen::VectorXd::Zero(X_->rows());
    for (std::size_t k = 0; k < actives_.size(); ++k) {
        const std::size_t a = actives_[k];
        if (a < p_original_) {
            u += w_A(k) * X_->col(static_cast<Eigen::Index>(a));
        } else {
            const auto& q = active_pairs_.at(a);
            u(q.first)  += w_A(k) * dummy_scale_;
            u(q.second) -= w_A(k) * dummy_scale_;
        }
    }
    return u;
}

std::pair<double, Eigen::VectorXd> SD2_TLARS_Solver::computeStepSize(double Cmax, const Eigen::Ref<const Eigen::VectorXd>& u) {
    Eigen::VectorXd a(inactives_.size());
    double gamma = std::numeric_limits<double>::max();

    #pragma omp parallel
    {
        double local_gamma = std::numeric_limits<double>::max();

        // 1. BLAS check against Original Inactive Features
        #pragma omp for schedule(static) nowait
        for (std::size_t i = 0; i < inactives_.size(); ++i) {
            std::size_t j = inactives_[i];
            double a_j = X_->col(static_cast<Eigen::Index>(j)).dot(u);
            a(i) = a_j;

            double c_j = correlations_(j);
            double g1 = (Cmax - c_j) / (A_A_ - a_j);
            double g2 = (Cmax + c_j) / (A_A_ + a_j);

            if (g1 > eps_ && g1 < local_gamma) local_gamma = g1;
            if (g2 > eps_ && g2 < local_gamma) local_gamma = g2;
        }

        // 2. Pair check against the candidate dummies: one subtraction each
        #pragma omp for schedule(static) nowait
        for (std::size_t i = 0; i < pool_.size(); ++i) {
            auto& dummy = pool_[i];

            double a_j = dummy_scale_ * (u(dummy.i) - u(dummy.j));
            dummy.current_a = a_j;

            double c_j = dummy.corr;
            double g1 = (Cmax - c_j) / (A_A_ - a_j);
            double g2 = (Cmax + c_j) / (A_A_ + a_j);

            if (g1 > eps_ && g1 < local_gamma) local_gamma = g1;
            if (g2 > eps_ && g2 < local_gamma) local_gamma = g2;
        }

        #pragma omp critical
        {
            if (local_gamma < gamma) gamma = local_gamma;
        }
    }

    return {gamma, a};
}

} // namespace trex::tsolvers::linear_model::lars_based
