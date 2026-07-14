// ===================================================================================
// sd_tomp_solver.hpp
// ===================================================================================
#ifndef TSOLVERS_SD_TOMP_SOLVER_HPP
#define TSOLVERS_SD_TOMP_SOLVER_HPP
// ===================================================================================
/**
 * @file sd_tomp_solver.hpp
 *
 * @brief SD-TOMP: Orthogonal Matching Pursuit racing the original features
 *        against sparse balanced Rademacher dummies.
 *
 * @details Greedy twin of SD_TLARS_Solver (same dummy law, same on-demand
 *          pool race): each step takes the argmax of |<., r>| over all
 *          candidates and performs a full OLS refit on the active set
 *          (Cholesky append + triangular solves). The full refit keeps r
 *          orthogonal to the active span, which structurally removes the
 *          duplicate-collinearity problem (a dependent column's correlation
 *          is ~0). Only the first materialization of a dummy counts toward
 *          the early-stopping threshold T.
 */
// ===================================================================================

#include "sd_general_base.hpp"

namespace trex::tsolvers::linear_model::omp_based {

class SD_TOMP_Solver : public SDGeneralSolver_Base {
private:
    Eigen::VectorXd Xty_active_;
    Eigen::VectorXd mu_;

    bool appendToActiveSet(std::size_t winning_j);
    void olsRefit();

public:
    /**
     * @brief SD-TOMP over sparse balanced Rademacher dummies.
     *
     * @param rho_d Dummy non-zero fraction (2k ~= rho_d * n). Pass 0 to
     *              auto-calibrate k from the data via sd_calibration.
     *              The result is readable via getAutoCalibration().
     * @param L_max Total dummy budget. Pass 0 for the auto budget:
     *              the calibration's L when rho_d == 0, else 2p.
     */
    SD_TOMP_Solver(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
                   double rho_d, std::size_t L_max, std::size_t T_stop, bool intercept = true,
                   uint64_t seed = 0);

    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;
};

} // namespace trex::tsolvers::linear_model::omp_based
#endif
