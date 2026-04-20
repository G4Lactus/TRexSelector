// ==============================================================================
// demo_trx_da_04_bt.cpp
// ==============================================================================
/**
 * @file demo_trx_da_04_bt.cpp
 *
 * @brief DA-TRex demo and MC simulation for the binary-tree (BT) DGP.
 *
 * @details
 *  Part 1: Single-run demo (n=150, p=500, s=5, 10 blocks, rho_w=0.8, rho_b=0.1).
 *  Part 2: MC simulation sweeping SNR (n=300, p=1000, s=10, 200 MC, 3 solvers).
 */
// ==============================================================================

#include "trex_da_sim_common.hpp"

using namespace da_sim;


int main() {

    std::cout.setf(std::ios::unitbuf);

    constexpr int    n_blocks    = 10;
    constexpr double rho_within  = 0.8;
    constexpr double rho_between = 0.1;

    // ══════════════════════════════════════════════════════════════════════════
    //  Part 1: Single-run demo
    // ══════════════════════════════════════════════════════════════════════════
    {
        SimConfig cfg;
        cfg.n = 150; cfg.p = 500; cfg.s = 5;
                constexpr int max_gap = 20;  // max gap for capped spread support
        const auto policy = SupportPolicy::CappedSpread;

        cdianostics::print_section_header(
            "BT Demo  |  n=" + std::to_string(cfg.n)
            + "  p=" + std::to_string(cfg.p) + "  s=" + std::to_string(cfg.s)
            + "  blocks=" + std::to_string(n_blocks)
            + "  rho_w=" + std::to_string(rho_within)
            + "  rho_b=" + std::to_string(rho_between)
            + "  support=" + support_policy_label(policy));

        auto dat = dgp_bt(cfg.n, cfg.p,
                          make_support(policy, cfg.s, cfg.p, max_gap,
                                       static_cast<unsigned>(cfg.base_seed)),
                          cfg.amplitude,
                          n_blocks, rho_within, rho_between,
                          cfg.snr, static_cast<unsigned>(cfg.base_seed));

        auto trex_ctrl = make_trex_control(cfg.K);
        trex_ctrl.solver_type = SolverTypeForTRex::TOMP;

        auto da_ctrl = make_da_bt_control();

        Eigen::Map<Eigen::MatrixXd> X_map(
            dat.X.data(), dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        TRexDASelector selector(
            X_map, y_map, cfg.tFDR, da_ctrl, trex_ctrl,
            cfg.base_seed, /*verbose=*/true);
        selector.select();

        auto m = evaluate_selection("BT DA+BT (avg)",
                                   selector.getSelectedIndices(),
                                   dat.true_support);
        print_single_result(m, cfg.tFDR);
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  Part 2: Monte Carlo simulation — SNR sweep
    // ══════════════════════════════════════════════════════════════════════════
    {
        SimConfig cfg;  // n=300, p=1000, s=10
                constexpr int max_gap = 20;  // max gap for capped spread support
        const auto policy = SupportPolicy::CappedSpread;

        auto da_ctrl = make_da_bt_control();

        run_snr_sweep(
            "da_bt", {0.1, 0.2, 0.5, 1.0, 2.0, 5.0},
            cfg.num_MC, cfg.tFDR, cfg.base_seed,
            default_solvers(),
            [&](double snr, unsigned seed) {
                return dgp_bt(cfg.n, cfg.p,
                              make_support(policy, cfg.s, cfg.p, max_gap, seed),
                              cfg.amplitude,
                              n_blocks, rho_within, rho_between,
                              snr, seed);
            },
            da_ctrl,
            make_trex_control(cfg.K),
            "BT 10 blocks, rho_w=0.8, rho_b=0.1, n=" + std::to_string(cfg.n)
            + ", p=" + std::to_string(cfg.p)
            + ", support=" + support_policy_label(policy));
    }

    std::cout << "\nBT demo complete.\n";
    return 0;
}
