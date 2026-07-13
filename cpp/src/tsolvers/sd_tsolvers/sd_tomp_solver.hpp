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
 * @details Greedy twin of SD_TLARS_Solver on the same dummy law and race
 *          semantics (standing on-demand pool, fixed total budget L_max).
 *          Each step: the pool is expanded until a dummy beats the
 *          best inactive real (or the budget is exhausted), the global argmax of
 *          |<., r>| enters the active set, and the coefficients are refit by
 *          full OLS on the active set (Cholesky append + triangular solves).
 *
 *          The refit re-orthogonalizes the residual, so pool correlations
 *          are recomputed from r directly — O(k) index sums per dummy, no
 *          dots and no incremental bookkeeping. A pool dummy that duplicates
 *          an active one is harmless here: after the refit r is orthogonal
 *          to the active span, so its correlation is exactly zero and it can
 *          never win (no rollback machinery needed in practice; the guard is
 *          kept for safety).
 */
// ===================================================================================

#include "sd_tsolver_base.hpp"
#include "sd_calibration.hpp"

namespace trex::tsolvers::linear_model::omp_based {

class SD_TOMP_Solver : public SDTSolver_Base {
private:
    std::size_t L_max_;
    double eps_{1e-12};
    Eigen::VectorXd Xty_active_;
    Eigen::VectorXd mu_;
    uint64_t virtual_seed_counter_{0}; // Tracks global dummy index j

    // Set when the ctor ran auto-calibration (rho_d == 0).
    std::optional<sd_calibration::Result> auto_calibration_{};

    VirtualDummy generateVirtualDummy(uint64_t seed);
    void refreshPoolCorrelations();
    void evaluateAndExpandPool();
    std::tuple<double, std::size_t, bool> findGlobalWinner() const;

    bool appendToActiveSet(std::size_t winning_j);
    void olsRefit();

public:
    /**
     * @brief SD-TOMP over sparse balanced Rademacher dummies.
     *
     * @param rho_d Dummy non-zero fraction (2k ~= rho_d * n). Pass 0 to
     *              auto-calibrate k from the data via sd_calibration
     *              (pilot screen + MC; seconds at large p). The result is
     *              readable via getAutoCalibration().
     * @param L_max Total dummy budget. Pass 0 for the auto budget:
     *              the calibration's L when rho_d == 0, else 2p.
     */
    SD_TOMP_Solver(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
                   double rho_d, std::size_t L_max, std::size_t T_stop, bool intercept = true,
                   uint64_t seed = 0);

    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    std::size_t getNumGeneratedDummies() const { return virtual_seed_counter_; }
    std::size_t getPoolSize() const { return pool_Q_.size(); }
    std::size_t getLMax() const { return L_max_; }
    const std::optional<sd_calibration::Result>& getAutoCalibration() const {
        return auto_calibration_;
    }
};

} // namespace trex::tsolvers::linear_model::omp_based
#endif
