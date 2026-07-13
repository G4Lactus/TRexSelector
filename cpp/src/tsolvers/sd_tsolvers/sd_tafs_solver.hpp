// ===================================================================================
// sd_tafs_solver.hpp
// ===================================================================================
#ifndef TSOLVERS_SD_TAFS_SOLVER_HPP
#define TSOLVERS_SD_TAFS_SOLVER_HPP
// ===================================================================================
/**
 * @file sd_tafs_solver.hpp
 *
 * @brief SD-TAFS: Adaptive Forward Stepwise racing the original features
 *        against sparse balanced Rademacher dummies.
 *
 * @details Blended twin of SD_TOMP_Solver (same dummy law, same on-demand
 *          pool race). Each step the argmax of |<., r>| over all candidates
 *          is taken — including already-active features, which may be
 *          re-selected — and the fit moves a fraction rho toward the OLS
 *          solution on the active set:
 *
 *              mu   <- (1 - rho) * mu   + rho * X_A nu,   nu = OLS(A)
 *              beta <- (1 - rho) * beta + rho * nu
 *
 *          A new winner is appended to the active set before the blend; a
 *          re-selected winner only blends. rho = 1 reduces to SD-TOMP
 *          (active correlations are exactly zero after a full refit, so
 *          re-selection never fires); rho -> 0 approaches LARS-like
 *          behavior. Only the first materialization of a dummy counts
 *          toward the early-stopping threshold T.
 */
// ===================================================================================

#include "sd_tsolver_base.hpp"
#include "sd_calibration.hpp"

namespace trex::tsolvers::linear_model::afs_based {

class SD_TAFS_Solver : public SDTSolver_Base {
private:
    // Real-feature candidate state: 0 = inactive, 1 = active (re-selectable),
    // 2 = banned (collinear append failure; never scanned again).
    enum : uint8_t { kInactive = 0, kActive = 1, kBanned = 2 };

    std::size_t L_max_;
    double eps_{1e-12};
    double rho_{0.3};
    Eigen::VectorXd Xty_active_;
    Eigen::VectorXd mu_;
    std::vector<uint8_t> real_state_;
    std::vector<VirtualDummy> active_dummy_recs_; // index sets of active dummies
    uint64_t virtual_seed_counter_{0};            // Tracks global dummy index j

    // Set when the ctor ran auto-calibration (rho_d == 0).
    std::optional<sd_calibration::Result> auto_calibration_{};

    struct Candidate {
        double abs_corr{0.0};
        std::size_t j{0};    // global index
        bool is_new{true};   // false: re-selected active feature (blend only)
    };

    VirtualDummy generateVirtualDummy(uint64_t seed);
    void refreshPoolCorrelations();
    void expandPool(double c_ref);
    Candidate findBestNonPool() const;

    bool appendToActiveSet(std::size_t winning_j);
    void afsBlend();

public:
    /**
     * @brief SD-TAFS over sparse balanced Rademacher dummies.
     *
     * @param rho_d Dummy non-zero fraction (2k ~= rho_d * n). Pass 0 to
     *              auto-calibrate k from the data via sd_calibration.
     * @param L_max Total dummy budget. Pass 0 for the auto budget:
     *              the calibration's L when rho_d == 0, else 2p.
     * @param rho   AFS blending fraction in (0, 1]; 1 reduces to OMP.
     */
    SD_TAFS_Solver(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
                   double rho_d, std::size_t L_max, std::size_t T_stop, bool intercept = true,
                   uint64_t seed = 0, double rho = 0.3);

    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;

    double getRho() const { return rho_; }
    std::size_t getNumGeneratedDummies() const { return virtual_seed_counter_; }
    std::size_t getPoolSize() const { return pool_Q_.size(); }
    std::size_t getLMax() const { return L_max_; }
    const std::optional<sd_calibration::Result>& getAutoCalibration() const {
        return auto_calibration_;
    }
};

} // namespace trex::tsolvers::linear_model::afs_based
#endif
