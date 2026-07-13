// ===================================================================================
// sd_tlars_solver.cpp
// ===================================================================================
#include "sd_tlars_solver.hpp"
#include <numeric>
#include <algorithm>
#include <cmath>

namespace trex::tsolvers::linear_model::lars_based {

SD_TLARS_Solver::SD_TLARS_Solver(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
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
    maxSteps_ = 8 * std::min(p_original_, effective_n_);
    correlations_ = X_->transpose() * r_;
    pool_Q_.reserve(p_original_ * 2);

    for (std::size_t j = 0; j < p_original_; ++j) inactives_.push_back(j);
}

SDTSolver_Base::VirtualDummy SD_TLARS_Solver::generateVirtualDummy(uint64_t seed) {
    VirtualDummy dummy;
    dummy.seed = seed;
    dummy.P_indices.reserve(k_sparse_);
    dummy.M_indices.reserve(k_sparse_);

    std::vector<int> candidates(y_.size());
    std::iota(candidates.begin(), candidates.end(), 0);

    // Partial Fisher-Yates shuffle
    for(std::size_t i = 0; i < 2 * k_sparse_; ++i) {
        std::uniform_int_distribution<std::size_t> dist(i, candidates.size() - 1);
        std::swap(candidates[i], candidates[dist(dummy_rng_)]);
    }

    dummy.P_indices.assign(candidates.begin(), candidates.begin() + k_sparse_);
    dummy.M_indices.assign(candidates.begin() + k_sparse_, candidates.begin() + 2 * k_sparse_);
    return dummy;
}

void SD_TLARS_Solver::evaluateAndExpandPool() {
    double c_X_max = 0.0;
    for (std::size_t j : inactives_) {
        c_X_max = std::max(c_X_max, std::abs(correlations_(j)));
    }

    // Pool correlations are maintained incrementally at the end of each
    // step (c_j -= gamma * a_j, exact since c_j = d^T r); only the running
    // maximum is scanned here.
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

std::tuple<double, std::size_t, bool> SD_TLARS_Solver::findGlobalWinner() {
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

void SD_TLARS_Solver::executeStep(std::size_t T_stop, bool early_stop) {
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

        if (!updateActiveSet(winning_j)) {
            // Collinear with the active set (e.g. a duplicate sparse dummy —
            // likely for small k, where the pair universe is only C(n,2)):
            // discard the candidate permanently and continue the path.
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
// Mathematical Core (Cholesky & Step Size)
// ==========================================================================

bool SD_TLARS_Solver::updateActiveSet(std::size_t winning_j) {
    Eigen::MatrixXd newR = updateR(getColumn(winning_j));
    if (newR.rows() == R_.rows()) return false; // Collinear drop

    R_ = newR;
    actives_.push_back(winning_j);

    // Determine sign based on correlation logic
    // (If dummy, we compute on the fly or pull from correlation)
    double corr = (winning_j < p_original_) ? correlations_(winning_j) : getColumn(winning_j).dot(r_);
    Sign_.push_back((corr >= 0) ? 1 : -1);
    return true;
}

Eigen::MatrixXd SD_TLARS_Solver::updateR(const Eigen::Ref<const Eigen::VectorXd>& xnew) {
    std::size_t m = actives_.size();
    double xtx = xnew.dot(xnew);

    if (m == 0) {
        Eigen::MatrixXd newR(1, 1);
        newR(0, 0) = std::sqrt(xtx);
        return newR;
    }

    Eigen::VectorXd cross_prod(m);
    for (std::size_t k = 0; k < m; ++k) {
        cross_prod(k) = xnew.dot(getColumn(actives_[k]));
    }

    Eigen::VectorXd r_vec = R_.transpose().triangularView<Eigen::Lower>().solve(cross_prod);
    double rpp_sq = xtx - r_vec.dot(r_vec);

    if (rpp_sq < eps_ * xtx) return R_; // Collinear failure

    Eigen::MatrixXd newR = Eigen::MatrixXd::Zero(m + 1, m + 1);
    newR.topLeftCorner(m, m) = R_;
    newR.block(0, m, m, 1) = r_vec;
    newR(m, m) = std::sqrt(rpp_sq);
    return newR;
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
