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
 * @details Every dummy operation is dot-free (see sd2_pair_base.hpp):
 *            - generation:              O(1) (two uniform index draws)
 *            - correlation  <d, r>:     s * (r_i - r_j)
 *            - Cholesky cross products: O(1) per active column
 *            - fitted values mu:        two entries touched per active dummy
 *          No dummy column is ever materialized; an active dummy is just its
 *          (i, j) pair. Consequently executeStep(T) is not limited by the
 *          constructor's T_stop.
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

#include "sd2_pair_base.hpp"

#include <tuple>
#include <utility>

namespace trex::tsolvers::linear_model::omp_based {

using trex::tsolvers::SD2GenPolicy;

class SD2_TOMP_Solver : public SD2PairSolver_Base {
private:
    Eigen::VectorXd Xty_active_;
    Eigen::VectorXd mu_;

    bool appendToActiveSet(std::size_t winning_j, bool is_dummy, int pi, int pj);
    void olsRefit();

public:
    /// @param L_max Total dummy budget; 0 selects the auto budget 2p.
    SD2_TOMP_Solver(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
                    std::size_t L_max, std::size_t T_stop, bool intercept = true,
                    uint64_t seed = 0,
                    SD2GenPolicy policy = SD2GenPolicy::OnDemand);

    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;
};

} // namespace trex::tsolvers::linear_model::omp_based
#endif
