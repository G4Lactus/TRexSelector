// ===================================================================================
// sd_tomp_solver.cpp
// ===================================================================================
#include "sd_tomp_solver.hpp"
#include <algorithm>
#include <cmath>

namespace trex::tsolvers::linear_model::omp_based {

SD_TOMP_Solver::SD_TOMP_Solver(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
                               double rho_d, std::size_t L_max, std::size_t T_stop, bool intercept,
                               uint64_t seed)
    : SDGeneralSolver_Base(X, y, rho_d, L_max, T_stop, intercept, seed) {

    r_ = y_;
    mu_ = Eigen::VectorXd::Zero(y_.size());
    maxSteps_ = 8 * std::min(p_original_, effective_n_);
    correlations_ = X_->transpose() * r_;

    for (std::size_t j = 0; j < p_original_; ++j) inactives_.push_back(j);
}

void SD_TOMP_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    while (currentStep_ < maxSteps_ && actives_.size() < effective_n_ &&
          (!early_stop || count_active_dummies_ < T_stop)) {

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

        if (!appendToActiveSet(winning_j)) {
            // Collinear with the active set. Nearly unreachable in OMP
            // (r is orthogonal to the active span, so a dependent column's
            // correlation is ~0), kept as a safety guard: discard the
            // candidate permanently and continue the path.
            if (is_dummy) rollbackDummyAdmission(winning_j);
            continue;
        }

        currentStep_++;
        lambda_.push_back(Cmax);

        olsRefit();
    }
}

// ==========================================================================
// Mathematical Core (Cholesky append & OLS refit)
// ==========================================================================

bool SD_TOMP_Solver::appendToActiveSet(std::size_t winning_j) {
    const auto m = static_cast<Eigen::Index>(actives_.size());

    double xty = 0.0;
    if (!appendActiveColumn(winning_j, &xty)) return false; // Collinear failure

    Xty_active_.conservativeResize(m + 1);
    Xty_active_(m) = xty;
    return true;
}

void SD_TOMP_Solver::olsRefit() {
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
        mu_ += z(i) * getColumn(actives_[static_cast<std::size_t>(i)]);
    }
    r_ = y_ - mu_;

    correlations_.noalias() = X_->transpose() * r_;
    refreshPoolCorrelations();
}

} // namespace trex::tsolvers::linear_model::omp_based
