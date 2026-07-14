// ===================================================================================
// sd2_tlars_solver.hpp
// ===================================================================================
#ifndef TSOLVERS_SD2_TLARS_SOLVER_HPP
#define TSOLVERS_SD2_TLARS_SOLVER_HPP
// ===================================================================================
/**
 * @file sd2_tlars_solver.hpp
 *
 * @brief SD-TLARS specialized to 2-sparse Rademacher dummies d = s*(e_i - e_j),
 *        s = ||x|| / sqrt(2). Pair-arithmetic twin of SD_TLARS_Solver.
 *
 * @details Every dummy operation is dot-free (see sd2_pair_base.hpp for the
 *          shared pair machinery):
 *            - generation:              O(1) (two uniform index draws)
 *            - correlation  <d, r>:     s * (r_i - r_j)
 *            - step size    <d, u>:     s * (u_i - u_j)
 *            - Cholesky cross products: O(1) per active column
 *            - equiangular vector u:    two entries touched per active dummy
 *          No dummy column is ever materialized; an active dummy is just its
 *          (i, j) pair. Consequently executeStep(T) is not limited by the
 *          constructor's T_stop.
 *
 *          The RNG draw pattern replicates SD_TLARS_Solver's partial
 *          Fisher-Yates at k = 1, so for the same seed both solvers race the
 *          same dummy stream and should produce the same selection path (up
 *          to floating-point rounding differences at exact ties).
 *
 *          Under the Geometric policy virtual dummies contribute no
 *          constraint to the LARS step size.
 */
// ===================================================================================

#include "sd2_pair_base.hpp"

#include <tuple>
#include <utility>

namespace trex::tsolvers::linear_model::lars_based {

// SD2GenPolicy lives in sd2_pair_base.hpp (shared by the whole pair
// family); re-exported here so lars_based::SD2GenPolicy keeps resolving.
using trex::tsolvers::SD2GenPolicy;

class SD2_TLARS_Solver : public SD2PairSolver_Base {
private:
    double A_A_{0.0};
    std::vector<int> Sign_;

    bool updateActiveSet(std::size_t winning_j, bool is_dummy,
                         int pi, int pj, double corr);
    Eigen::VectorXd signVector() const;
    Eigen::VectorXd equiangularDirection(const Eigen::Ref<const Eigen::VectorXd>& sign);
    Eigen::VectorXd equiangularVector(const Eigen::Ref<const Eigen::VectorXd>& w_A) const;
    std::pair<double, Eigen::VectorXd> computeStepSize(double Cmax, const Eigen::Ref<const Eigen::VectorXd>& u);

public:
    /// @param L_max Total dummy budget; 0 selects the auto budget 2p.
    SD2_TLARS_Solver(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
                     std::size_t L_max, std::size_t T_stop, bool intercept = true,
                     uint64_t seed = 0,
                     SD2GenPolicy policy = SD2GenPolicy::OnDemand);

    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;
};

} // namespace trex::tsolvers::linear_model::lars_based
#endif
