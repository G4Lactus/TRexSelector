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
    : SDTSolver_Base(X, y, /*rho_d=*/2.0 / static_cast<double>(X.rows()),
                     T_stop, intercept, seed),
      L_max_(L_max), policy_(policy) {

    if (L_max_ == 0) L_max_ = 2 * p_original_;  // auto budget

    r_ = y_;
    maxSteps_ = 8 * std::min(p_original_, effective_n_);
    correlations_ = X_->transpose() * r_;
    pool_.reserve(std::min(L_max_, p_original_ * 2));

    for (std::size_t j = 0; j < p_original_; ++j) inactives_.push_back(j);
}

SD2_TLARS_Solver::PairDummy SD2_TLARS_Solver::generatePair(uint64_t seed) {
    const std::size_t n = static_cast<std::size_t>(y_.size());

    // Two draws replicating the general solver's partial Fisher-Yates at
    // k = 1 (same distributions, same order), so both solvers consume the
    // identical RNG stream: after swap(c[0], c[a]), c[0] = a; after
    // swap(c[1], c[b]), c[1] = (b == a ? 0 : b).
    std::uniform_int_distribution<std::size_t> first_draw(0, n - 1);
    std::uniform_int_distribution<std::size_t> second_draw(1, n - 1);
    const std::size_t a = first_draw(dummy_rng_);
    const std::size_t b = second_draw(dummy_rng_);
    const std::size_t plus  = a;
    const std::size_t minus = (b == a) ? 0 : b;

    return PairDummy{seed, static_cast<int>(plus), static_cast<int>(minus), 0.0};
}

void SD2_TLARS_Solver::evaluateAndExpandPool() {
    double c_X_max = 0.0;
    for (std::size_t j : inactives_) {
        c_X_max = std::max(c_X_max, std::abs(correlations_(j)));
    }

    // Pool correlations are maintained incrementally at the end of each
    // step (c_j -= gamma * a_j, exact since c_j = d^T r); only the running
    // maximum is scanned here.
    double c_Q_max = 0.0;
    for (const auto& dummy : pool_) {
        c_Q_max = std::max(c_Q_max, std::abs(dummy.corr));
    }

    // Fixed-L race semantics: generation stops at the first beater, and the
    // only cap is the total budget L_max — every step therefore races against
    // (effectively) all L dummies, exactly like the classic pre-filled dummy
    // matrix. (The earlier milestone ladder min(m*p, L_max) under-supplied
    // dummies in early steps relative to the L the FDP estimator assumes,
    // which was anti-conservative whenever L > p.)
    while (c_Q_max <= c_X_max && virtual_seed_counter_ < L_max_) {
        uint64_t next_j = dummy_start_idx_ + virtual_seed_counter_++;
        PairDummy new_dummy = generatePair(next_j);
        new_dummy.corr = dummy_scale_ * (r_(new_dummy.i) - r_(new_dummy.j));

        c_Q_max = std::max(c_Q_max, std::abs(new_dummy.corr));
        pool_.push_back(new_dummy);
    }
}

std::tuple<double, std::size_t, bool, std::size_t> SD2_TLARS_Solver::findGlobalWinner() const {
    double Cmax = 0.0;
    std::size_t winning_j = 0;
    bool is_dummy = false;
    std::size_t pool_idx = 0;

    for (std::size_t j : inactives_) {
        if (std::abs(correlations_(j)) > Cmax) {
            Cmax = std::abs(correlations_(j));
            winning_j = j;
        }
    }
    for (std::size_t q = 0; q < pool_.size(); ++q) {
        if (std::abs(pool_[q].corr) > Cmax) {
            Cmax = std::abs(pool_[q].corr);
            winning_j = pool_[q].seed;
            is_dummy = true;
            pool_idx = q;
        }
    }
    return {Cmax, winning_j, is_dummy, pool_idx};
}

// ==========================================================================
// Geometric policy: exact pair null via the sorted residual
// ==========================================================================

void SD2_TLARS_Solver::buildBeatingSet(double c_threshold, BeatingSet& bs) const {
    const auto n = static_cast<std::size_t>(y_.size());
    bs.sorted.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        bs.sorted[i] = {r_(static_cast<Eigen::Index>(i)), static_cast<int>(i)};
    }
    std::sort(bs.sorted.begin(), bs.sorted.end());

    // Ordered pair (plus, minus) beats iff r_plus - r_minus > t. For the
    // ascending sort, count for each plus-position i the minus-positions j
    // with v_j < v_i - t (two-pointer, O(n) after the sort).
    const double t = c_threshold / dummy_scale_;
    bs.prefix.assign(n, 0);
    std::size_t lo = 0, run = 0;
    for (std::size_t i = 0; i < n; ++i) {
        while (lo < n && bs.sorted[lo].first < bs.sorted[i].first - t) ++lo;
        run += lo;
        bs.prefix[i] = run;
    }
    bs.one_side = run;
    bs.pi = (n > 1)
        ? (2.0 * static_cast<double>(run)) /
          (static_cast<double>(n) * static_cast<double>(n - 1))
        : 0.0;
}

SD2_TLARS_Solver::PairDummy SD2_TLARS_Solver::sampleBeatingPair(const BeatingSet& bs) {
    // Uniform over the 2 * one_side ordered beating pairs (both orientations).
    std::uniform_int_distribution<std::size_t> rank_draw(0, 2 * bs.one_side - 1);
    std::size_t rank = rank_draw(dummy_rng_);
    const bool flipped = (rank >= bs.one_side);
    if (flipped) rank -= bs.one_side;

    const auto it = std::upper_bound(bs.prefix.begin(), bs.prefix.end(), rank);
    const auto plus_pos = static_cast<std::size_t>(it - bs.prefix.begin());
    const std::size_t base = (plus_pos == 0) ? 0 : bs.prefix[plus_pos - 1];
    const std::size_t minus_pos = rank - base;

    int plus  = bs.sorted[plus_pos].second;
    int minus = bs.sorted[minus_pos].second;
    if (flipped) std::swap(plus, minus);

    const double corr = dummy_scale_ * (r_(plus) - r_(minus));
    return PairDummy{0, plus, minus, corr};
}

bool SD2_TLARS_Solver::geometricExpansion(double c_X_max, PairDummy& winner) {
    BeatingSet bs;
    buildBeatingSet(c_X_max, bs);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // 1. Standing virtual pool: P(one of F frozen draws beats) = 1-(1-pi)^F.
    //    Exchangeability approximation: frozen failures are treated as fresh
    //    draws against the current residual.
    if (virtual_pool_ > 0 && bs.pi > 0.0) {
        const double p_hit =
            1.0 - std::pow(1.0 - bs.pi, static_cast<double>(virtual_pool_));
        if (unif(dummy_rng_) < p_hit) {
            winner = sampleBeatingPair(bs);
            winner.seed = dummy_start_idx_ + L_max_ + geom_win_counter_++;
            --virtual_pool_;
            return true;
        }
    }

    // 2. Expansion: Geometric(pi) count of fresh draws, capped by the total
    //    budget L_max (fixed-L race semantics, mirrors the on-demand loop).
    std::size_t budget = 0;
    if (L_max_ > virtual_seed_counter_) {
        budget = static_cast<std::size_t>(L_max_ - virtual_seed_counter_);
    }

    if (budget > 0 && bs.pi > 0.0) {
        const double u = 1.0 - unif(dummy_rng_);   // (0, 1]
        const double draws = 1.0 + std::floor(std::log(u) / std::log1p(-bs.pi));
        if (draws <= static_cast<double>(budget)) {
            const auto g = static_cast<std::size_t>(draws);
            virtual_seed_counter_ += g;
            virtual_pool_ += g - 1;
            winner = sampleBeatingPair(bs);
            winner.seed = dummy_start_idx_ + L_max_ + geom_win_counter_++;
            return true;
        }
    }

    // No success within the budget: the whole budget is spent on failures.
    virtual_seed_counter_ += budget;
    virtual_pool_ += budget;
    return false;
}

// ==========================================================================

void SD2_TLARS_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    while (currentStep_ < maxSteps_ && actives_.size() < effective_n_ &&
          (!early_stop || count_active_dummies_ < T_stop)) {

        double Cmax = 0.0;
        std::size_t winning_j = 0;
        bool is_dummy = false;
        std::size_t pool_idx = 0;
        PairDummy geom_winner{};

        if (policy_ == SD2GenPolicy::OnDemand) {
            evaluateAndExpandPool();
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
// Mathematical Core (Cholesky & Step Size) — dot-free on the dummy path
// ==========================================================================

bool SD2_TLARS_Solver::updateActiveSet(std::size_t winning_j, bool is_dummy,
                                       int pi, int pj, double corr) {
    const std::size_t m = actives_.size();
    const double s = dummy_scale_;

    // xtx and cross products against the active set, without any dummy dots.
    double xtx;
    Eigen::VectorXd cross_prod(m);

    if (is_dummy) {
        xtx = 2.0 * (s * s);
        for (std::size_t k = 0; k < m; ++k) {
            const std::size_t a = actives_[k];
            if (a < p_original_) {
                const auto xa = X_->col(static_cast<Eigen::Index>(a));
                cross_prod(k) = s * xa(pi) - s * xa(pj);
            } else {
                const auto& q = active_pairs_.at(a);
                const int overlap = (pi == q.first) + (pj == q.second)
                                  - (pi == q.second) - (pj == q.first);
                cross_prod(k) = overlap * (s * s);
            }
        }
    } else {
        const auto xnew = X_->col(static_cast<Eigen::Index>(winning_j));
        xtx = xnew.dot(xnew);
        for (std::size_t k = 0; k < m; ++k) {
            const std::size_t a = actives_[k];
            if (a < p_original_) {
                cross_prod(k) = xnew.dot(X_->col(static_cast<Eigen::Index>(a)));
            } else {
                const auto& q = active_pairs_.at(a);
                cross_prod(k) = s * xnew(q.first) - s * xnew(q.second);
            }
        }
    }

    // Cholesky append (same math as the general solver's updateR)
    if (m == 0) {
        R_ = Eigen::MatrixXd::Constant(1, 1, std::sqrt(xtx));
    } else {
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
    if (is_dummy) active_pairs_[winning_j] = {pi, pj};
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
