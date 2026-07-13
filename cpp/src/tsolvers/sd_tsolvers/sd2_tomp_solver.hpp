// ===================================================================================
// sd2_tomp_solver.hpp
// ===================================================================================
#ifndef TSOLVERS_SD2_TOMP_SOLVER_HPP
#define TSOLVERS_SD2_TOMP_SOLVER_HPP
// ===================================================================================
/**
 * @file sd2_tomp_solver.hpp
 *
 * @brief SD2-TOMP: Orthogonal Matching Pursuit specialized to 2-sparse
 *        Rademacher dummies d = s*(e_i - e_j), s = ||x|| / sqrt(2).
 *        Pair-arithmetic twin of SD_TOMP_Solver.
 *
 * @details Every dummy operation is dot-free:
 *            - generation:              O(1) (two uniform index draws)
 *            - correlation  <d, r>:     s * (r_i - r_j)
 *            - Cholesky cross products: O(1) per active column
 *            - fitted values mu:        two entries touched per active dummy
 *          No dummy column is ever materialized (the base-class cache stays
 *          unused); an active dummy is just its (i, j) pair. Consequently
 *          executeStep(T) is not limited by the constructor's T_stop.
 *
 *          The RNG draw pattern replicates SD_TOMP_Solver's partial
 *          Fisher-Yates at k = 1, so for the same seed both solvers race the
 *          same dummy stream and should produce the same selection path (up
 *          to floating-point rounding differences at exact ties).
 *
 *          Supports both SD2GenPolicy values: OnDemand (explicit standing
 *          pool, correlations recomputed from r after each refit) and
 *          Geometric (exact pair null per step against the fully
 *          re-orthogonalized residual; failures stay virtual).
 */
// ===================================================================================

#include "sd_tsolver_base.hpp"

#include <tuple>
#include <utility>

namespace trex::tsolvers::linear_model::omp_based {

class SD2_TOMP_Solver : public SDTSolver_Base {
private:
    struct PairDummy {
        std::size_t seed;  // global index j >= p
        int i;             // +s position
        int j;             // -s position
        double corr;       // current correlation s * (r_i - r_j)
    };

    std::size_t L_max_;
    double eps_{1e-12};
    Eigen::VectorXd Xty_active_;
    Eigen::VectorXd mu_;
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

    PairDummy generatePair(uint64_t seed);
    void refreshPoolCorrelations();
    void evaluateAndExpandPool();
    std::tuple<double, std::size_t, bool, std::size_t> findGlobalWinner() const;

    void buildBeatingSet(double c_threshold, BeatingSet& bs) const;
    PairDummy sampleBeatingPair(const BeatingSet& bs);
    bool geometricExpansion(double c_X_max, PairDummy& winner);

    bool appendToActiveSet(std::size_t winning_j, bool is_dummy, int pi, int pj);
    void olsRefit();

public:
    /// @param L_max Total dummy budget; 0 selects the auto budget 2p.
    SD2_TOMP_Solver(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
                    std::size_t L_max, std::size_t T_stop, bool intercept = true,
                    uint64_t seed = 0,
                    SD2GenPolicy policy = SD2GenPolicy::OnDemand);

    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    std::size_t getNumGeneratedDummies() const { return virtual_seed_counter_; }
    std::size_t getPoolSize() const {
        return policy_ == SD2GenPolicy::Geometric ? virtual_pool_ : pool_.size();
    }
    SD2GenPolicy getPolicy() const { return policy_; }
};

} // namespace trex::tsolvers::linear_model::omp_based
#endif
