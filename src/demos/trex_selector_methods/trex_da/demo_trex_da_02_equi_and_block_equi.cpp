// ==============================================================================
// demo_trex_da_02_equi_and_block_equi.cpp
// ==============================================================================
/**
 * @file demo_trex_da_02_equi_and_block_equi.cpp
 *
 * @brief DA-TRex demo for equicorrelated and BT (hierarchical-block) DGPs.
 *
 * @details
 *  Covers the R references demo_trex_da_02_equi.R and
 *  demo_trex_da_03_bt_equi_blocks.R.
 *
 *  Section 1: Single-run equicorrelated demo (n=150, p=500, s=5, rho=0.25,
 *             Random support).
 *
 *  Section 2: Single-run BT demo (n=150, p=500, s=5, n_blocks=5,
 *             rho_within=0.2, rho_between=0.05, OnePerBlock support).
 *
 *  Section 3: MC SNR sweep — Equi DGP (n=300, p=1000, rho=0.25, Random
 *             support, 3 solvers + base T-Rex comparison).
 *
 *  Section 4: MC rho sweep — Equi DGP (n=300, p=1000, SNR=2.0, Random
 *             support, 3 solvers + base T-Rex comparison).
 *
 *  Section 5: MC SNR sweep — BT DGP (n=300, p=1000, n_blocks=10,
 *             rho_within=0.5, rho_between=0.1, linkage=Single, OnePerBlock
 *             support, 3 solvers + base T-Rex comparison).
 *
 *  Section 6: MC linkage sweep — BT DGP (n=300, p=1000, n_blocks=10,
 *             rho_within=0.8, rho_between=0.1, OnePerBlock support; outer
 *             loop over Single / Complete / Average linkage).
 */
// ==============================================================================

#include "trex_da_sim_common.hpp"

using namespace da_sim;


int main() {

    std::cout.setf(std::ios::unitbuf);

    const std::vector<double> snr_grid = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0};
    const std::vector<double> rho_grid = {
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    // ══════════════════════════════════════════════════════════════════════════
    //  Section 1: Single-run equicorrelated demo
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        SimConfig cfg;
        cfg.n = 150; cfg.p = 500; cfg.s = 5;
        constexpr double rho = 0.25;
        const auto policy = SupportPolicy::Random;

        cdianostics::print_section_header(
            "Equi Demo  |  n=" + std::to_string(cfg.n)
            + "  p=" + std::to_string(cfg.p) + "  s=" + std::to_string(cfg.s)
            + "  rho=" + std::to_string(rho)
            + "  support=" + support_policy_label(policy));

        auto dat = dgp_equi(cfg.n, cfg.p,
                            make_support(policy, cfg.s, cfg.p, 0,
                                         static_cast<unsigned>(cfg.base_seed)),
                            cfg.amplitude,
                            rho, cfg.snr, static_cast<unsigned>(cfg.base_seed));

        auto trex_ctrl = make_trex_control(cfg.K);
        trex_ctrl.solver_type = SolverTypeForTRex::TOMP;

        TRexDAControlParameter da_ctrl;
        da_ctrl.method = DAMethod::EQUI;

        Eigen::Map<Eigen::MatrixXd> X_map(
            dat.X.data(), dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        TRexDASelector selector(
            X_map, y_map, cfg.tFDR, da_ctrl, trex_ctrl,
            cfg.base_seed, /*verbose=*/true);
        selector.select();

        auto m = evaluate_selection("Equi DA+EQUI",
                                   selector.getSelectedIndices(),
                                   dat.true_support);
        print_single_result(m, cfg.tFDR);
    }


    // ══════════════════════════════════════════════════════════════════════════
    //  Section 2: Single-run BT (hierarchical-block) demo
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        SimConfig cfg;
        cfg.n = 150; cfg.p = 500; cfg.s = 5;
        constexpr int    n_blocks   = 5;
        constexpr double rho_within  = 0.2;
        constexpr double rho_between = 0.05;
        const auto policy = SupportPolicy::OnePerBlock;

        cdianostics::print_section_header(
            "BT Demo  |  n=" + std::to_string(cfg.n)
            + "  p=" + std::to_string(cfg.p) + "  s=" + std::to_string(cfg.s)
            + "  n_blocks=" + std::to_string(n_blocks)
            + "  rho_w=" + std::to_string(rho_within)
            + "  rho_b=" + std::to_string(rho_between)
            + "  support=" + support_policy_label(policy));

        auto support = make_support(policy, cfg.s, cfg.p, 0,
                                    static_cast<unsigned>(cfg.base_seed));
        auto dat = dgp_bt(cfg.n, cfg.p, support, cfg.amplitude,
                          n_blocks, rho_within, rho_between,
                          cfg.snr, static_cast<unsigned>(cfg.base_seed));

        auto trex_ctrl = make_trex_control(cfg.K);
        trex_ctrl.solver_type = SolverTypeForTRex::TOMP;

        auto da_ctrl = make_da_bt_control();
        da_ctrl.hc_linkage = hac::LinkageMethod::Single;

        Eigen::Map<Eigen::MatrixXd> X_map(
            dat.X.data(), dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        TRexDASelector selector(
            X_map, y_map, cfg.tFDR, da_ctrl, trex_ctrl,
            cfg.base_seed, /*verbose=*/true);
        selector.select();

        auto m = evaluate_selection("BT DA+BT (single)",
                                   selector.getSelectedIndices(),
                                   dat.true_support);
        print_single_result(m, cfg.tFDR);
    }


    // ══════════════════════════════════════════════════════════════════════════
    //  Section 3: MC SNR sweep — equicorrelated
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        constexpr int    n = 300, p = 1000, s = 10;
        constexpr double rho = 0.25;
        const auto policy = SupportPolicy::Random;
        constexpr int    K        = 20;
        constexpr int    num_MC   = 200;
        constexpr double tFDR     = 0.2;
        constexpr int    base_seed = 2026;

        TRexDAControlParameter da_ctrl;
        da_ctrl.method = DAMethod::EQUI;

        run_snr_sweep(
            "da_equi_snr",
            snr_grid,
            num_MC,
            tFDR,
            base_seed,
            default_solvers(),
            [&](double snr, unsigned seed) {
                return dgp_equi(n, p,
                                make_support(policy, s, p, 0, seed),
                                /*amplitude=*/3.0,
                                rho, snr, seed);
            },
            da_ctrl,
            make_trex_control(K),
            "Equi rho=" + std::to_string(rho)
            + ", n=" + std::to_string(n)
            + ", p=" + std::to_string(p)
            + ", support=" + support_policy_label(policy),
            /*include_base_trex=*/true);
    }


    // ══════════════════════════════════════════════════════════════════════════
    //  Section 4: MC rho sweep — equicorrelated
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        constexpr int    n = 300, p = 1000, s = 10;
        constexpr double snr = 2.0;
        const auto policy = SupportPolicy::Random;
        constexpr int    K         = 20;
        constexpr int    num_MC    = 200;
        constexpr double tFDR      = 0.2;
        constexpr int    base_seed = 2026;

        TRexDAControlParameter da_ctrl;
        da_ctrl.method = DAMethod::EQUI;

        run_param_sweep(
            "da_equi_rho",
            "Rho",
            rho_grid,
            num_MC,
            tFDR,
            base_seed,
            default_solvers(),
            [&](double rho, unsigned seed) {
                return dgp_equi(n, p,
                                make_support(policy, s, p, 0, seed),
                                /*amplitude=*/3.0,
                                rho, snr, seed);
            },
            da_ctrl,
            make_trex_control(K),
            "Equi SNR=" + std::to_string(snr)
            + ", n=" + std::to_string(n)
            + ", p=" + std::to_string(p)
            + ", support=" + support_policy_label(policy),
            /*include_base_trex=*/true);
    }


    // ══════════════════════════════════════════════════════════════════════════
    //  Section 5: MC SNR sweep — BT (hierarchical-block), linkage=Single
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        constexpr int    n = 300, p = 1000, s = 10;
        constexpr int    n_blocks    = 10;
        constexpr double rho_within  = 0.5;
        constexpr double rho_between = 0.1;
        const auto policy = SupportPolicy::OnePerBlock;
        constexpr int    K         = 20;
        constexpr int    num_MC    = 200;
        constexpr double tFDR      = 0.2;
        constexpr int    base_seed = 2026;

        // Fixed support (OnePerBlock; seed-independent for fixed s/p/n_blocks)
        const auto support_fixed =
            make_support(policy, s, p, 0, 0u);

        auto da_ctrl = make_da_bt_control();
        da_ctrl.hc_linkage = hac::LinkageMethod::Single;

        run_snr_sweep(
            "da_bt_snr_single",
            snr_grid,
            num_MC,
            tFDR,
            base_seed,
            default_solvers(),
            [&](double snr, unsigned seed) {
                return dgp_bt(n, p, support_fixed,
                              /*amplitude=*/3.0,
                              n_blocks, rho_within, rho_between,
                              snr, seed);
            },
            da_ctrl,
            make_trex_control(K),
            "BT n_blocks=" + std::to_string(n_blocks)
            + " rho_w=" + std::to_string(rho_within)
            + " rho_b=" + std::to_string(rho_between)
            + ", n=" + std::to_string(n) + ", p=" + std::to_string(p)
            + ", linkage=Single"
            + ", support=" + support_policy_label(policy),
            /*include_base_trex=*/true);
    }


    // ══════════════════════════════════════════════════════════════════════════
    //  Section 6: MC linkage sweep — BT, high within-correlation
    //  Outer loop: Single / Complete / Average linkage methods
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        constexpr int    n = 300, p = 1000, s = 10;
        constexpr int    n_blocks    = 10;
        constexpr double rho_within  = 0.8;
        constexpr double rho_between = 0.1;
        const auto policy = SupportPolicy::OnePerBlock;
        constexpr int    K         = 20;
        constexpr int    num_MC    = 200;
        constexpr double tFDR      = 0.2;
        constexpr int    base_seed = 2026;

        const auto support_fixed =
            make_support(policy, s, p, 0, 0u);

        const std::vector<std::pair<std::string, hac::LinkageMethod>> linkages = {
            {"Single",   hac::LinkageMethod::Single},
            {"Complete", hac::LinkageMethod::Complete},
            {"Average",  hac::LinkageMethod::Average}
        };

        for (const auto& [lnk_str, lnk_val] : linkages) {
            auto da_ctrl = make_da_bt_control();
            da_ctrl.hc_linkage = lnk_val;

            run_snr_sweep(
                "da_bt_snr_" + lnk_str,
                snr_grid,
                num_MC,
                tFDR,
                base_seed,
                default_solvers(),
                [&](double snr, unsigned seed) {
                    return dgp_bt(n, p, support_fixed,
                                  /*amplitude=*/3.0,
                                  n_blocks, rho_within, rho_between,
                                  snr, seed);
                },
                da_ctrl,
                make_trex_control(K),
                "BT n_blocks=" + std::to_string(n_blocks)
                + " rho_w=" + std::to_string(rho_within)
                + " rho_b=" + std::to_string(rho_between)
                + ", n=" + std::to_string(n) + ", p=" + std::to_string(p)
                + ", linkage=" + lnk_str
                + ", support=" + support_policy_label(policy),
                /*include_base_trex=*/false);
        }
    }

    std::cout << "\nEqui + BT demo complete.\n";
    return 0;
}
