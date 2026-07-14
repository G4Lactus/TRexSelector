// ===================================================================================
// sd2_pair_base.hpp
// ===================================================================================
#ifndef TSOLVERS_SD2_PAIR_BASE_HPP
#define TSOLVERS_SD2_PAIR_BASE_HPP
// ===================================================================================
/**
 * @file sd2_pair_base.hpp
 *
 * @brief Family base of the pair (k = 1) SD solvers (SD2_TLARS / SD2_TOMP /
 * SD2_TAFS): fully dot-free dummy arithmetic (a pair dummy is two indices,
 * every inner product is O(1)), no materialized cache, and the two
 * generation policies (OnDemand pool / Geometric exact pair null).
 *
 * @details The twins replicate the general solvers' RNG draw pattern at
 * k = 1 (same distributions, same order), so for the same seed the general
 * and pair solvers produce the identical selection path — the property the
 * unit tests pin. Fixed-L race semantics as in the general family.
 */
// ===================================================================================

#include "sd_tsolver_base.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace trex::tsolvers {

/**
 * @brief Dummy-generation policy for the pair (k = 1) solver family
 *        (SD2_TLARS / SD2_TOMP / SD2_TAFS).
 *
 * @details OnDemand mirrors the general-k solvers: an explicit pool of
 *          sampled pairs, expanded one draw at a time until a dummy beats
 *          the best real feature. Geometric exploits the exact 2-sparse
 *          null: the fraction pi of beating pairs is computed from the
 *          sorted residual (two-pointer, O(n log n)), the number of draws
 *          until the first success is sampled as Geometric(pi), and the
 *          winner is drawn uniformly from the beating set. Failures are
 *          never instantiated — only the virtual pool size is tracked.
 *          Standing failures are treated as exchangeable fresh draws
 *          against the current residual (their historical conditioning is
 *          ignored), which is the approximation the policy comparison
 *          quantifies.
 */
enum class SD2GenPolicy : uint8_t {
    OnDemand  = 0,
    Geometric = 1
};

class SD2PairSolver_Base : public SDTSolver_Base {
protected:
    struct PairDummy {
        std::size_t seed;      // global index j >= p
        int i;                 // +s position
        int j;                 // -s position
        double corr;           // current correlation s * (r_i - r_j)
        double current_a{0.0}; // <d, u> of the last step (LARS c -= gamma*a)
    };

    std::size_t L_max_;
    std::vector<PairDummy> pool_;
    std::unordered_map<std::size_t, std::pair<int, int>> active_pairs_;
    uint64_t virtual_seed_counter_{0};

    // --- Geometric policy state ---
    SD2GenPolicy policy_{SD2GenPolicy::OnDemand};
    std::size_t virtual_pool_{0};   // unmaterialized (failure) draws
    uint64_t geom_win_counter_{0};  // unique global ids for geometric winners

    // Sorted-residual view of the exact pair null at a given threshold.
    struct BeatingSet {
        std::vector<std::pair<double, int>> sorted;  // (r_i, index), ascending
        std::vector<std::size_t> prefix;             // cumulative beating counts
        std::size_t one_side{0};                     // beating pairs, one orientation
        double pi{0.0};                              // beating fraction, ordered pairs
    };

    // ==========================================================================
    // Pair generation & pool race (shared across LARS / OMP / AFS)
    // ==========================================================================
    /** @brief Draw a fresh pair replicating the general solver's partial
     *  Fisher-Yates at k = 1 (same distributions, same order), so both
     *  solvers consume the identical RNG stream: after swap(c[0], c[a]),
     *  c[0] = a; after swap(c[1], c[b]), c[1] = (b == a ? 0 : b). */
    PairDummy generatePair(uint64_t seed) {
        const std::size_t n = static_cast<std::size_t>(y_.size());

        std::uniform_int_distribution<std::size_t> first_draw(0, n - 1);
        std::uniform_int_distribution<std::size_t> second_draw(1, n - 1);
        const std::size_t a = first_draw(dummy_rng_);
        const std::size_t b = second_draw(dummy_rng_);
        const std::size_t plus  = a;
        const std::size_t minus = (b == a) ? 0 : b;

        return PairDummy{seed, static_cast<int>(plus), static_cast<int>(minus), 0.0};
    }

    /** @brief Recompute all pool correlations from the residual (exact, no
     *  drift): one subtraction per pair. Used by the refit solvers
     *  (OMP/AFS); LARS maintains them incrementally. No-op under the
     *  Geometric policy (the pool stays virtual). */
    void refreshPoolCorrelations() {
        #pragma omp parallel for
        for (std::size_t i = 0; i < pool_.size(); ++i) {
            pool_[i].corr = dummy_scale_ * (r_(pool_[i].i) - r_(pool_[i].j));
        }
    }

    /**
     * @brief Expand the pool with fresh pairs until one beats @p c_ref or
     * the total budget L_max is exhausted (fixed-L race semantics,
     * generation stops at the first beater).
     */
    void expandPool(double c_ref) {
        double c_Q_max = 0.0;
        for (const auto& dummy : pool_) {
            c_Q_max = std::max(c_Q_max, std::abs(dummy.corr));
        }

        while (c_Q_max <= c_ref && virtual_seed_counter_ < L_max_) {
            uint64_t next_j = dummy_start_idx_ + virtual_seed_counter_++;
            PairDummy new_dummy = generatePair(next_j);
            new_dummy.corr = dummy_scale_ * (r_(new_dummy.i) - r_(new_dummy.j));

            c_Q_max = std::max(c_Q_max, std::abs(new_dummy.corr));
            pool_.push_back(new_dummy);
        }
    }

    /** @brief Argmax of |correlation| over inactive reals and the pool.
     *  @return (Cmax, winning global index, winner is a dummy, pool index). */
    std::tuple<double, std::size_t, bool, std::size_t> findGlobalWinner() const {
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
    void buildBeatingSet(double c_threshold, BeatingSet& bs) const {
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

    PairDummy sampleBeatingPair(const BeatingSet& bs) {
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

    bool geometricExpansion(double c_ref, PairDummy& winner) {
        BeatingSet bs;
        buildBeatingSet(c_ref, bs);
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
    // Dot-free active-set growth
    // ==========================================================================
    /**
     * @brief Grow the active set by column @p winning_j — a real feature or
     * a pair dummy (pi, pj) — without any dummy dot products: xtx, x^T y and
     * the cross products against active pairs are index arithmetic.
     *
     * @param xty_out When non-null, receives x_new^T y (maintained by the
     *                refit solvers for their OLS solves; LARS passes null).
     *
     * @return false on collinear failure (state untouched; the pair
     *         universe has only C(n,2) distinct dummies, so duplicates are
     *         a real possibility).
     */
    bool pairAppendActive(std::size_t winning_j, bool is_dummy, int pi, int pj,
                          double* xty_out = nullptr) {
        const std::size_t m = actives_.size();
        const double s = dummy_scale_;

        double xtx;
        Eigen::VectorXd cross_prod(static_cast<Eigen::Index>(m));

        if (is_dummy) {
            xtx = 2.0 * (s * s);
            if (xty_out != nullptr) *xty_out = s * (y_(pi) - y_(pj));
            for (std::size_t k = 0; k < m; ++k) {
                const std::size_t a = actives_[k];
                if (a < p_original_) {
                    const auto xa = X_->col(static_cast<Eigen::Index>(a));
                    cross_prod(static_cast<Eigen::Index>(k)) = s * xa(pi) - s * xa(pj);
                } else {
                    const auto& q = active_pairs_.at(a);
                    const int overlap = (pi == q.first) + (pj == q.second)
                                      - (pi == q.second) - (pj == q.first);
                    cross_prod(static_cast<Eigen::Index>(k)) = overlap * (s * s);
                }
            }
        } else {
            const auto xnew = X_->col(static_cast<Eigen::Index>(winning_j));
            xtx = xnew.dot(xnew);
            if (xty_out != nullptr) *xty_out = xnew.dot(y_);
            for (std::size_t k = 0; k < m; ++k) {
                const std::size_t a = actives_[k];
                if (a < p_original_) {
                    cross_prod(static_cast<Eigen::Index>(k)) =
                        xnew.dot(X_->col(static_cast<Eigen::Index>(a)));
                } else {
                    const auto& q = active_pairs_.at(a);
                    cross_prod(static_cast<Eigen::Index>(k)) =
                        s * xnew(q.first) - s * xnew(q.second);
                }
            }
        }

        if (!tryAppendCholesky(xtx, cross_prod)) return false;

        actives_.push_back(winning_j);
        if (is_dummy) active_pairs_[winning_j] = {pi, pj};
        return true;
    }

public:
    /// @param L_max Total dummy budget; 0 selects the auto budget 2p.
    SD2PairSolver_Base(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
                       std::size_t L_max, std::size_t T_stop, bool intercept,
                       uint64_t seed, SD2GenPolicy policy)
        : SDTSolver_Base(X, y, /*rho_d=*/2.0 / static_cast<double>(X.rows()),
                         T_stop, intercept, seed),
          L_max_(L_max), policy_(policy) {

        if (L_max_ == 0) L_max_ = 2 * p_original_;  // auto budget
        pool_.reserve(std::min(L_max_, p_original_ * 2));
    }

    // ==========================================================================
    // Read-only accessors
    // ==========================================================================
    std::size_t getNumGeneratedDummies() const { return virtual_seed_counter_; }
    std::size_t getPoolSize() const { return pool_.size(); }
    std::size_t getLMax() const { return L_max_; }
    SD2GenPolicy getPolicy() const { return policy_; }
};

} // namespace trex::tsolvers
#endif
