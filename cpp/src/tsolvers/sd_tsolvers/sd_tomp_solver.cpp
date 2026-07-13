// ===================================================================================
// sd_tomp_solver.cpp
// ===================================================================================
#include "sd_tomp_solver.hpp"
#include <numeric>
#include <algorithm>
#include <cmath>

namespace trex::tsolvers::linear_model::omp_based {

SD_TOMP_Solver::SD_TOMP_Solver(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
                               double rho_d, std::size_t L_max, std::size_t T_stop, bool intercept,
                               uint64_t seed)
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

    r_ = y_;
    mu_ = Eigen::VectorXd::Zero(y_.size());
    maxSteps_ = 8 * std::min(p_original_, effective_n_);
    correlations_ = X_->transpose() * r_;
    pool_Q_.reserve(p_original_ * 2);

    for (std::size_t j = 0; j < p_original_; ++j) inactives_.push_back(j);
}

SDTSolver_Base::VirtualDummy SD_TOMP_Solver::generateVirtualDummy(uint64_t seed) {
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

// Pool correlations are recomputed from the residual after every OLS refit
// (exact, no drift): c_j = d^T r is an O(k) index-set sum per dummy.
void SD_TOMP_Solver::refreshPoolCorrelations() {
    #pragma omp parallel for
    for (std::size_t i = 0; i < pool_Q_.size(); ++i) {
        double c = 0.0;
        for (int idx : pool_Q_[i].P_indices) c += r_(idx);
        for (int idx : pool_Q_[i].M_indices) c -= r_(idx);
        pool_Q_[i].current_correlation = c * dummy_scale_;
    }
}

void SD_TOMP_Solver::evaluateAndExpandPool() {
    double c_X_max = 0.0;
    for (std::size_t j : inactives_) {
        c_X_max = std::max(c_X_max, std::abs(correlations_(j)));
    }

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
    while (c_Q_max <= c_X_max && virtual_seed_counter_ < L_max_) {
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

std::tuple<double, std::size_t, bool> SD_TOMP_Solver::findGlobalWinner() const {
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

void SD_TOMP_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    while (currentStep_ < maxSteps_ && actives_.size() < effective_n_ &&
          (!early_stop || count_active_dummies_ < T_stop)) {

        evaluateAndExpandPool();
        auto [Cmax, winning_j, is_dummy] = findGlobalWinner();

        if (Cmax < 100 * eps_) break;

        // Materialize virtual dummy if it won
        if (is_dummy) {
            // Dummy cache exhausted (executeStep called beyond the ctor's
            // T_stop capacity): cannot admit further dummies.
            if (current_materialized_cols_ == D_materialized_.cols()) break;

            auto it = std::find_if(pool_Q_.begin(), pool_Q_.end(),
                                   [winning_j](const VirtualDummy& d) { return d.seed == winning_j; });
            materializeDummy(*it, winning_j);
            pool_Q_.erase(it);
            count_active_dummies_++;
        } else {
            std::erase(inactives_, winning_j);
        }

        if (!appendToActiveSet(winning_j)) {
            // Collinear with the active set. Nearly unreachable in OMP
            // (r is orthogonal to the active span, so a dependent column's
            // correlation is ~0), kept as a safety guard: discard the
            // candidate permanently and continue the path.
            if (is_dummy) {
                active_dummy_map_.erase(winning_j);
                --current_materialized_cols_;
                D_materialized_.col(current_materialized_cols_).setZero();
                --count_active_dummies_;
            }
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
