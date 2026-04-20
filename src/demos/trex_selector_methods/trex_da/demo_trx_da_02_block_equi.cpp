// ==============================================================================
// demo_trx_da_06_block_equi.cpp
// ==============================================================================
/**
 * @file demo_trx_da_06_block_equi.cpp
 *
 * @brief DA-TRex demo and MC simulation for the block-equicorrelated DGP.
 *
 * @details
 *  Part 1: Single-run demo (n=150, p=500, G=5, g=50, rho=0.7, SNR=2.0).
 *  Part 2: MC simulation sweeping SNR (n=300, p=1000, G=5, g=100, 200 MC,
 *           3 solvers).
 *
 *  Uses DA method BT with average linkage (the block-equicorrelated DGP
 *  has one active variable per active block, matching the BT grouping).
 */
// ==============================================================================

#include "trex_da_sim_common.hpp"

using namespace da_sim;


int main() {

    std::cout.setf(std::ios::unitbuf);

    constexpr double rho = 0.7;

    // ══════════════════════════════════════════════════════════════════════════
    //  Part 1: Single-run demo
    // ══════════════════════════════════════════════════════════════════════════
    {
        BlockSimConfig cfg;
        constexpr int max_gap = 20;  // max gap for capped spread support
        cfg.G = 5; cfg.g = 50;  // 10 blocks of 50, 5 active

        cdianostics::print_section_header(
            "Block-Equi Demo  |  n=" + std::to_string(cfg.n)
            + "  p=" + std::to_string(cfg.p) + "  G=" + std::to_string(cfg.G)
            + "  g=" + std::to_string(cfg.g)
            + "  rho=" + std::to_string(rho));

        auto dat = dgp_block_equi(cfg.n, cfg.p, cfg.G, cfg.g,
                                  rho, cfg.snr,
                                  static_cast<unsigned>(cfg.base_seed));

        auto trex_ctrl = make_trex_control(cfg.K);
        trex_ctrl.solver_type = SolverTypeForTRex::TOMP;

        auto da_ctrl = make_da_bt_control();

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(), dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        TRexDASelector selector(X_map, y_map, cfg.tFDR, da_ctrl, trex_ctrl,
                                cfg.base_seed, /*verbose=*/true);
        selector.select();

        auto m = evaluate_selection("Block-Equi DA+BT (avg)",
                                   selector.getSelectedIndices(),
                                   dat.true_support);
        print_single_result(m, cfg.tFDR);
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  Part 2: Monte Carlo simulation — SNR sweep
    // ══════════════════════════════════════════════════════════════════════════
    {
        BlockSimConfig cfg;
        cfg.n = 300; cfg.p = 1000; cfg.G = 5; cfg.g = 100;  // 10 blocks of 100

        auto da_ctrl = make_da_bt_control();

        run_snr_sweep(
            "da_block_equi",
            {0.1, 0.2, 0.5, 1.0, 2.0, 5.0},
            cfg.num_MC, cfg.tFDR, cfg.base_seed,
            default_solvers(),
            [&](double snr, unsigned seed) {
                return dgp_block_equi(cfg.n, cfg.p, cfg.G, cfg.g,
                                      rho, snr, seed);
            },
            da_ctrl,
            make_trex_control(cfg.K),
            "Block-Equi G=5, g=100, rho=0.7, n=" + std::to_string(cfg.n)
            + ", p=" + std::to_string(cfg.p));
    }

    std::cout << "\nBlock-Equi demo complete.\n";
    return 0;
}
