// ===================================================================================
// sd_tlars_solver.cpp
// ===================================================================================
#include "sd_tlars_solver.hpp"
#include <algorithm>
#include <cmath>

namespace trex::tsolvers::linear_model::lars_based {

SD_TLARS_Solver::SD_TLARS_Solver(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
                                 double rho_d, std::size_t L_max, std::size_t T_stop, bool intercept,
                                 uint64_t seed)
    : SDGeneralSolver_Base(X, y, rho_d, L_max, T_stop, intercept, seed) {

    r_ = y_;
    maxSteps_ = 8 * std::min(p_original_, effective_n_);
    correlations_ = X_->transpose() * r_;

    for (std::size_t j = 0; j < p_original_; ++j) inactives_.push_back(j);
}

void SD_TLARS_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    while (currentStep_ < maxSteps_ && actives_.size() < effective_n_ &&
          (!early_stop || count_active_dummies_ < T_stop)) {

        // Pool correlations are maintained incrementally at the end of each
        // step (c_j -= gamma * a_j, exact since c_j = d^T r), so only the
        // reference bar of the inactive reals is recomputed here.
        expandPool(inactivesMaxAbsCorr());
        auto [Cmax, winning_j, is_dummy] = findGlobalWinner();

        if (Cmax < 100 * eps_) break;

        // Materialize virtual dummy if it won
        if (is_dummy) {
            if (dummyCacheFull()) break;
            admitPoolWinner(winning_j);
        } else {
            std::erase(inactives_, winning_j);
        }

        if (!updateActiveSet(winning_j)) {
            // Collinear with the active set (e.g. a duplicate sparse dummy —
            // likely for small k, where the pair universe is only C(n,2)):
            // discard the candidate permanently and continue the path.
            if (is_dummy) rollbackDummyAdmission(winning_j);
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
        #pragma omp parallel for
        for (std::size_t i = 0; i < pool_Q_.size(); ++i) {
            pool_Q_[i].current_correlation -= gamma * pool_Q_[i].current_a;
        }
    }
}

// ==========================================================================
// Mathematical Core (LARS geometry)
// ==========================================================================

bool SD_TLARS_Solver::updateActiveSet(std::size_t winning_j) {
    // Sign from the correlation (dummies: computed on the fly) — captured
    // BEFORE the append so the collinear-failure path stays side-effect-free.
    double corr = (winning_j < p_original_) ? correlations_(winning_j)
                                            : getColumn(winning_j).dot(r_);

    if (!appendActiveColumn(winning_j)) return false; // Collinear drop

    Sign_.push_back((corr >= 0) ? 1 : -1);
    return true;
}

Eigen::VectorXd SD_TLARS_Solver::signVector() const {
    Eigen::VectorXd s(Sign_.size());
    for (std::size_t i = 0; i < Sign_.size(); ++i) s(i) = Sign_[i];
    return s;
}

Eigen::VectorXd SD_TLARS_Solver::equiangularDirection(const Eigen::Ref<const Eigen::VectorXd>& sign) {
    Eigen::VectorXd GAi_s = R_.triangularView<Eigen::Upper>().solve(
                            R_.transpose().triangularView<Eigen::Lower>().solve(sign));
    A_A_ = 1.0 / std::sqrt(sign.dot(GAi_s));
    return A_A_ * GAi_s;
}

Eigen::VectorXd SD_TLARS_Solver::equiangularVector(const Eigen::Ref<const Eigen::VectorXd>& w_A) const {
    Eigen::VectorXd u = Eigen::VectorXd::Zero(X_->rows());
    for (std::size_t k = 0; k < actives_.size(); ++k) {
        u += w_A(k) * getColumn(actives_[k]);
    }
    return u;
}

std::pair<double, Eigen::VectorXd> SD_TLARS_Solver::computeStepSize(double Cmax, const Eigen::Ref<const Eigen::VectorXd>& u) {
    Eigen::VectorXd a(inactives_.size());
    double gamma = std::numeric_limits<double>::max();

    #pragma omp parallel
    {
        double local_gamma = std::numeric_limits<double>::max();

        // 1. BLAS check against Original Inactive Features
        #pragma omp for schedule(static) nowait
        for (std::size_t i = 0; i < inactives_.size(); ++i) {
            std::size_t j = inactives_[i];
            double a_j = getColumn(j).dot(u);
            a(i) = a_j;

            double c_j = correlations_(j);
            double g1 = (Cmax - c_j) / (A_A_ - a_j);
            double g2 = (Cmax + c_j) / (A_A_ + a_j);

            if (g1 > eps_ && g1 < local_gamma) local_gamma = g1;
            if (g2 > eps_ && g2 < local_gamma) local_gamma = g2;
        }

        // 2. Combinatorial check against Virtual Candidate Dummies
        #pragma omp for schedule(static) nowait
        for (std::size_t i = 0; i < pool_Q_.size(); ++i) {
            auto& dummy = pool_Q_[i];

            double a_j = 0.0;
            for (int idx : dummy.P_indices) a_j += u(idx);
            for (int idx : dummy.M_indices) a_j -= u(idx);
            a_j *= dummy_scale_;
            dummy.current_a = a_j;

            double c_j = dummy.current_correlation;
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
