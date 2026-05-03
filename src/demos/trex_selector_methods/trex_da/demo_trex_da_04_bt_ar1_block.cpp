// ==============================================================================
// demo_trex_da_04_bt_ar1_block.cpp
// ==============================================================================
/**
 * @file demo_trex_da_04_bt_ar1_block.cpp
 *
 * @brief DA-TRex MC simulations for the AR(1)-block DGP using BT aggregation.
 *
 * @details
 *  Corresponds to R reference: demo_trex_da_05_bt_ar1_block_sweeps.R.
 *
 *  DGP: dgp_ar1_block() — block-diagonal AR(1) covariance. One active variable
 *  per block (OnePerBlock support). All M blocks are active (s = M).
 *
 *  Base params: n=150, M=5, Q=5 (p=25, s=5), rho=0.7, amplitude=1.0,
 *               tFDR=0.2, K=20, num_MC=200, seed=2026.
 *
 *  Every section loops over linkage in {Single, Complete, Average}. All 3
 *  solvers (TLARS, TAFS, TOMP) run inside each linkage iteration.
 *
 *  Section 0: Single-run demo (if false).
 *
 *  Section 1: SNR sweep    SNR in {0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0}.
 *
 *  Section 2: rho sweep    rho in {0.0, 0.1, ..., 0.9}.
 *
 *  Section 3: Q sweep      Q in {5, 10, ..., 50}; p = M*Q varies; s = M = 5 fixed.
 *
 *  Section 4: M sweep      M in {1, 3, 5, 10, 15, 20, 30}; p = M*Q and s = M vary.
 */
// ==============================================================================

#include "trex_da_sim_common.hpp"

using namespace da_sim;


int main() {

    std::cout.setf(std::ios::unitbuf);

    // ── shared base parameters ─────────────────────────────────────────────────
    constexpr int    n         = 150;
    constexpr int    M         = 5;      // number of AR(1) blocks
    constexpr int    Q         = 5;      // block size; base p = M * Q = 25
    constexpr double amplitude = 1.0;
    constexpr double rho       = 0.7;
    constexpr double snr       = 2.0;
    constexpr double tFDR      = 0.2;
    constexpr int    K         = 20;
    constexpr int    num_MC    = 200;
    constexpr int    base_seed = 2026;

    const std::vector<std::pair<std::string, hac::LinkageMethod>> linkages = {
        {"Single",   hac::LinkageMethod::Single},
        {"Complete", hac::LinkageMethod::Complete},
        {"Average",  hac::LinkageMethod::Average}
    };

    const int  base_p       = M * Q;   // 25 at base setting
    const auto support_base =
        make_support(SupportPolicy::OnePerBlock, M, base_p, 0, 0u);


    // ══════════════════════════════════════════════════════════════════════════
    //  Section 0: Single-run demo
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        cdianostics::print_section_header(
            "AR(1)-block+BT Demo  |  n=" + std::to_string(n)
            + "  M=" + std::to_string(M) + "  Q=" + std::to_string(Q)
            + "  p=" + std::to_string(base_p) + "  s=" + std::to_string(M)
            + "  rho=" + std::to_string(rho) + "  linkage=Average");

        auto dat = dgp_ar1_block(n, base_p, support_base, amplitude,
                                 M, rho, snr, static_cast<unsigned>(base_seed));

        auto trex_ctrl = make_trex_control(K);
        trex_ctrl.solver_type = SolverTypeForTRex::TOMP;

        auto da_ctrl = make_da_bt_control();
        da_ctrl.hc_linkage = hac::LinkageMethod::Average;

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(), dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        TRexDASelector selector(X_map, y_map, tFDR, da_ctrl, trex_ctrl,
                                base_seed, /*verbose=*/true);
        selector.select();

        auto m = evaluate_selection("AR(1)-block+BT",
                                    selector.getSelectedIndices(),
                                    dat.true_support);
        print_single_result(m, tFDR);
    }


    // ══════════════════════════════════════════════════════════════════════════
    //  Section 1: MC SNR sweep
    //  Fixed: n=150, M=5, Q=5 (p=25, s=5), rho=0.7.
    //  Swept: SNR in {0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0}.
    //  Outer loop: linkage in {Single, Complete, Average}.
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        const std::vector<double> snr_grid = {
            0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0};

        for (const auto& [lnk_str, lnk_val] : linkages) {
            auto da_ctrl = make_da_bt_control();
            da_ctrl.hc_linkage = lnk_val;

            run_snr_sweep(
                "da_ar1_blk_snr_" + lnk_str,
                snr_grid,
                num_MC,
                tFDR,
                base_seed,
                default_solvers(),
                [&](double snr_val, unsigned seed) {
                    return dgp_ar1_block(n, base_p, support_base, amplitude,
                                        M, rho, snr_val, seed);
                },
                da_ctrl,
                make_trex_control(K),
                "AR(1)-block SNR sweep  n=" + std::to_string(n)
                + "  M=" + std::to_string(M) + "  Q=" + std::to_string(Q)
                + "  p=" + std::to_string(base_p) + "  s=" + std::to_string(M)
                + "  rho=" + std::to_string(rho) + "  linkage=" + lnk_str,
                /*include_base_trex=*/false);
        }
    }


    // ══════════════════════════════════════════════════════════════════════════
    //  Section 2: MC rho sweep
    //  Fixed: n=150, M=5, Q=5 (p=25, s=5), SNR=2.
    //  Swept: rho in {0.0, 0.1, ..., 0.9}.
    //  Outer loop: linkage in {Single, Complete, Average}.
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        const std::vector<double> rho_grid = {
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
            0.7, 0.8, 0.9
        };

        for (const auto& [lnk_str, lnk_val] : linkages) {
            auto da_ctrl = make_da_bt_control();
            da_ctrl.hc_linkage = lnk_val;

            run_param_sweep(
                "da_ar1_blk_rho_" + lnk_str,
                "Rho",
                rho_grid,
                num_MC,
                tFDR,
                base_seed,
                default_solvers(),
                [&](double rho_val, unsigned seed) {
                    return dgp_ar1_block(n, base_p, support_base, amplitude,
                                        M, rho_val, snr, seed);
                },
                da_ctrl,
                make_trex_control(K),
                "AR(1)-block rho sweep  n=" + std::to_string(n)
                + "  M=" + std::to_string(M) + "  Q=" + std::to_string(Q)
                + "  p=" + std::to_string(base_p) + "  s=" + std::to_string(M)
                + "  SNR=" + std::to_string(snr) + "  linkage=" + lnk_str,
                /*include_base_trex=*/false);
        }
    }


    // ══════════════════════════════════════════════════════════════════════════
    //  Section 3: MC Q sweep (block size)
    //  Fixed: n=150, M=5, rho=0.7, SNR=2.  s = M = 5 throughout.
    //  Swept: Q in {5, 10, ..., 50}; p = M*Q varies.
    //  Support recomputed per grid point (OnePerBlock on p_val = M*Q).
    //  Outer loop: linkage in {Single, Complete, Average}.
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        const std::vector<double> q_grid = {5, 10, 15, 20, 25,
             30, 35, 40, 45, 50};

        for (const auto& [lnk_str, lnk_val] : linkages) {
            auto da_ctrl = make_da_bt_control();
            da_ctrl.hc_linkage = lnk_val;

            run_param_sweep(
                "da_ar1_blk_Q_" + lnk_str,
                "Q",
                q_grid,
                num_MC,
                tFDR,
                base_seed,
                default_solvers(),
                [&](double q_val, unsigned seed) {
                    int p_val = M * static_cast<int>(q_val);
                    auto sup  =
                        make_support(
                            SupportPolicy::OnePerBlock,
                            M,
                            p_val,
                            0,
                            0u);
                    return dgp_ar1_block(n, p_val, sup, amplitude,
                                        M, rho, snr, seed);
                },
                da_ctrl,
                make_trex_control(K),
                "AR(1)-block Q sweep  n=" + std::to_string(n)
                + "  M=" + std::to_string(M) + "  s=" + std::to_string(M)
                + "  rho=" + std::to_string(rho) + "  SNR=" + std::to_string(snr)
                + "  p=M*Q varies  linkage=" + lnk_str,
                /*include_base_trex=*/false);
        }
    }


    // ══════════════════════════════════════════════════════════════════════════
    //  Section 4: MC M sweep (number of blocks)
    //  Fixed: n=150, Q=5, rho=0.7, SNR=2.
    //  Swept: M in {1, 3, 5, 10, 15, 20, 30}; p = M*Q and s = M both vary.
    //  Support recomputed per grid point.
    //  Outer loop: linkage in {Single, Complete, Average}.
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        const std::vector<double> m_grid = {1, 3, 5, 10, 15, 20, 30};

        for (const auto& [lnk_str, lnk_val] : linkages) {
            auto da_ctrl = make_da_bt_control();
            da_ctrl.hc_linkage = lnk_val;

            run_param_sweep(
                "da_ar1_blk_M_" + lnk_str,
                "M",
                m_grid,
                num_MC,
                tFDR,
                base_seed,
                default_solvers(),
                [&](double m_val, unsigned seed) {
                    int m_int = static_cast<int>(m_val);
                    int p_val = m_int * Q;
                    auto sup  = make_support(SupportPolicy::OnePerBlock, m_int, p_val, 0, 0u);
                    return dgp_ar1_block(n, p_val, sup, amplitude,
                                        m_int, rho, snr, seed);
                },
                da_ctrl,
                make_trex_control(K),
                "AR(1)-block M sweep  n=" + std::to_string(n)
                + "  Q=" + std::to_string(Q)
                + "  rho=" + std::to_string(rho) + "  SNR=" + std::to_string(snr)
                + "  p=M*Q and s=M vary  linkage=" + lnk_str,
                /*include_base_trex=*/false);
        }
    }


    std::cout << "\nAR(1)-block BT sweeps demo complete.\n";
    return 0;
}
