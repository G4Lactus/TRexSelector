// ===================================================================================
// sd2_tafs_solver.hpp
// ===================================================================================
#ifndef TSOLVERS_SD2_TAFS_SOLVER_HPP
#define TSOLVERS_SD2_TAFS_SOLVER_HPP
// ===================================================================================
/**
 * @file sd2_tafs_solver.hpp
 *
 * @brief SD2-TAFS: Adaptive Forward Stepwise specialized to 2-sparse
 *        Rademacher dummies d = s*(e_i - e_j), s = ||x|| / sqrt(2).
 *        Pair-arithmetic twin of SD_TAFS_Solver.
 *
 * @details Blended variant of SD2_TOMP_Solver: the argmax over all
 *          candidates — including already-active features and active pairs,
 *          which may be re-selected — enters (or re-enters) and the fit
 *          moves a fraction rho toward the OLS solution on the active set.
 *          rho = 1 reduces to SD2-TOMP. Only the first entry of a dummy
 *          counts toward the early-stopping threshold T.
 *
 *          All dummy arithmetic is dot-free (see sd2_tomp_solver.hpp); no
 *          dummy column is ever materialized. Supports both SD2GenPolicy
 *          values: OnDemand (explicit standing pool) and Geometric (exact
 *          pair null per step; failures stay virtual). The RNG draw pattern
 *          replicates SD_TAFS_Solver's partial Fisher-Yates at k = 1, so
 *          for the same seed both solvers race the same dummy stream.
 */
// ===================================================================================

#include "sd_tsolver_base.hpp"

#include <tuple>
#include <utility>

namespace trex::tsolvers::linear_model::afs_based {

class SD2_TAFS_Solver : public SDTSolver_Base {
private:
    // Real-feature candidate state: 0 = inactive, 1 = active (re-selectable),
    // 2 = banned (collinear append failure; never scanned again).
    enum : uint8_t { kInactive = 0, kActive = 1, kBanned = 2 };

    struct PairDummy {
        std::size_t seed;  // global index j >= p
        int i;             // +s position
        int j;             // -s position
        double corr;       // current correlation s * (r_i - r_j)
    };

    std::size_t L_max_;
    double eps_{1e-12};
    double rho_{0.3};
    Eigen::VectorXd Xty_active_;
    Eigen::VectorXd mu_;
    std::vector<PairDummy> pool_;
    std::unordered_map<std::size_t, std::pair<int, int>> active_pairs_;
    std::vector<uint8_t> real_state_;
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

    struct Candidate {
        double abs_corr{0.0};
        std::size_t j{0};    // global index
        bool is_new{true};   // false: re-selected active feature (blend only)
    };

    PairDummy generatePair(uint64_t seed);
    void refreshPoolCorrelations();
    void expandPool(double c_ref);
    Candidate findBestNonPool() const;

    void buildBeatingSet(double c_threshold, BeatingSet& bs) const;
    PairDummy sampleBeatingPair(const BeatingSet& bs);
    bool geometricExpansion(double c_X_max, PairDummy& winner);

    bool appendToActiveSet(std::size_t winning_j, bool is_dummy, int pi, int pj);
    void afsBlend();

public:
    /**
     * @param L_max Total dummy budget; 0 selects the auto budget 2p.
     * @param rho   AFS blending fraction in (0, 1]; 1 reduces to OMP.
     */
    SD2_TAFS_Solver(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
                    std::size_t L_max, std::size_t T_stop, bool intercept = true,
                    uint64_t seed = 0,
                    SD2GenPolicy policy = SD2GenPolicy::OnDemand,
                    double rho = 0.3);

    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    double getRho() const { return rho_; }
    std::size_t getNumGeneratedDummies() const { return virtual_seed_counter_; }
    std::size_t getPoolSize() const {
        return policy_ == SD2GenPolicy::Geometric ? virtual_pool_ : pool_.size();
    }
    SD2GenPolicy getPolicy() const { return policy_; }
};

} // namespace trex::tsolvers::linear_model::afs_based
#endif
