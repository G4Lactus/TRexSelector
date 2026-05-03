// ==============================================================================
// demo_trex_da_06_heavy_tailed_block_sweeps.cpp
// ==============================================================================
/**
 * @file demo_trex_da_06_heavy_tailed_block_sweeps.cpp
 *
 * @brief DA-TRex MC simulations for the heavy-tailed block DGP (BT aggregation).
 *
 * @details
 *  Corresponds to R reference: demo_trex_da_06_bt_heavy_tailed_block_sweeps.R.
 *
 *  DGP: dgp_block_toeplitz_hvt() -- block-diagonal Toeplitz covariance, predictors
 *  and optionally noise are t(nu=3)-distributed. G = M blocks of size g = Q each.
 *  Support is determined internally (OnePerBlock); no support/amplitude args.
 *
 *  Base params: n=150, M=5, Q=5 (p=25, s=5), rho=0.8, nu=3.0,
 *               tFDR=0.2, K=20, num_MC=200, seed=2026.
 *
 *  Outer loops: h_noise in {false ("s1_Gauss"), true ("s2_Heavy")}
 *               linkage in {Single, Complete, Average}
 *  All 3 solvers (TLARS, TAFS, TOMP) run inside each linkage iteration.
 *
 *  Section 0: Single-run demo (if false).
 *  Section 1: SNR sweep    SNR in {0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0}.
 *  Section 2: rho sweep    rho in {0.0, 0.1, ..., 0.9}.
 *  Section 3: Q sweep      Q in {5, 10, ..., 50}; p = M*Q varies.
 *  Section 4: M sweep      M in {1, 3, 5, 10, 15, 20, 30}; p = M*Q and s = M vary.
 *  Section 5: tFDR sweep   tFDR in {0.05, 0.10, ..., 0.50}.
 *  Section 6: Linkage sweep  encoded as {1.0=Single, 2.0=Complete, 3.0=Average}.
 */
// ==============================================================================

#include "trex_da_sim_common.hpp"

using namespace da_sim;


int main() {

    std::cout.setf(std::ios::unitbuf);

    // -- shared base parameters ------------------------------------------------
    constexpr int    n         = 150;
    constexpr int    M         = 5;      // number of HT blocks (G)
    constexpr int    Q         = 5;      // block size (g); base p = M * Q = 25
    constexpr double rho       = 0.8;
    constexpr double nu        = 3.0;
    constexpr double snr       = 2.0;
    constexpr double tFDR      = 0.2;
    constexpr int    K         = 20;
    constexpr int    num_MC    = 200;
    constexpr int    base_seed = 2026;

    const int base_p = M * Q;   // 25

    // Noise scenarios: {tag_suffix, heavy_noise_flag}
    const std::vector<std::pair<std::string, bool>> scenarios = {
        {"s1_Gauss", false},
        {"s2_Heavy", true}
    };

    const std::vector<std::pair<std::string, hac::LinkageMethod>> linkages = {
        {"Single",   hac::LinkageMethod::Single},
        {"Complete", hac::LinkageMethod::Complete},
        {"Average",  hac::LinkageMethod::Average}
    };


    // ==========================================================================
    //  Section 0: Single-run demo
    // ==========================================================================
    if (false)
    {
        cdianostics::print_section_header(
            "HT-block+BT Demo  |  n=" + std::to_string(n)
            + "  M=" + std::to_string(M) + "  Q=" + std::to_string(Q)
            + "  p=" + std::to_string(base_p) + "  s=" + std::to_string(M)
            + "  rho=" + std::to_string(rho) + "  nu=" + std::to_string(nu)
            + "  heavy_noise=false  linkage=Average");

        auto dat = dgp_block_toeplitz_hvt(
            n, base_p, M, Q, rho, snr,
            static_cast<unsigned>(base_seed),
            /*heavy_X=*/true, /*heavy_noise=*/false, nu, /*random_support=*/false);

        auto trex_ctrl = make_trex_control(K);
        trex_ctrl.solver_type = SolverTypeForTRex::TOMP;

        auto da_ctrl = make_da_bt_control();
        da_ctrl.hc_linkage = hac::LinkageMethod::Average;

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(), dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        TRexDASelector selector(X_map, y_map, tFDR, da_ctrl, trex_ctrl,
                                base_seed, /*verbose=*/true);
        selector.select();

        auto m = evaluate_selection("HT-block+BT",
                                    selector.getSelectedIndices(),
                                    dat.true_support);
        print_single_result(m, tFDR);
    }


    // ==========================================================================
    //  Section 1: MC SNR sweep
    //  Fixed: n=150, M=5, Q=5 (p=25, s=5), rho=0.8, nu=3.0.
    //  Swept: SNR in {0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0}.
    //  Outer loops: h_noise scenario x linkage.
    // ==========================================================================
    if (false)
    {
        const std::vector<double> snr_grid = {0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0};

        for (const auto& [scen_str, h_noise] : scenarios) {
            for (const auto& [lnk_str, lnk_val] : linkages) {
                auto da_ctrl = make_da_bt_control();
                da_ctrl.hc_linkage = lnk_val;

                run_snr_sweep(
                    "da_ht_snr_" + scen_str + "_" + lnk_str,
                    snr_grid,
                    num_MC,
                    tFDR,
                    base_seed,
                    default_solvers(),
                    [&](double snr_val, unsigned seed) {
                        return dgp_block_toeplitz_hvt(
                            n, base_p, M, Q, rho, snr_val, seed,
                            /*heavy_X=*/true, h_noise, nu, /*random_support=*/false);
                    },
                    da_ctrl,
                    make_trex_control(K),
                    "HT-block SNR sweep  n=" + std::to_string(n)
                    + "  M=" + std::to_string(M) + "  Q=" + std::to_string(Q)
                    + "  p=" + std::to_string(base_p) + "  s=" + std::to_string(M)
                    + "  rho=" + std::to_string(rho) + "  nu=" + std::to_string(nu)
                    + "  noise=" + scen_str + "  linkage=" + lnk_str,
                    /*include_base_trex=*/false);
            }
        }
    }


    // ==========================================================================
    //  Section 2: MC rho sweep
    //  Fixed: n=150, M=5, Q=5 (p=25, s=5), SNR=2, nu=3.0.
    //  Swept: rho in {0.0, 0.1, ..., 0.9}.
    //  Outer loops: h_noise scenario x linkage.
    // ==========================================================================
    if (false)
    {
        const std::vector<double> rho_grid = {
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

        for (const auto& [scen_str, h_noise] : scenarios) {
            for (const auto& [lnk_str, lnk_val] : linkages) {
                auto da_ctrl = make_da_bt_control();
                da_ctrl.hc_linkage = lnk_val;

                run_param_sweep(
                    "da_ht_rho_" + scen_str + "_" + lnk_str,
                    "Rho",
                    rho_grid,
                    num_MC,
                    tFDR,
                    base_seed,
                    default_solvers(),
                    [&](double rho_val, unsigned seed) {
                        return dgp_block_toeplitz_hvt(
                            n, base_p, M, Q, rho_val, snr, seed,
                            /*heavy_X=*/true, h_noise, nu, /*random_support=*/false);
                    },
                    da_ctrl,
                    make_trex_control(K),
                    "HT-block rho sweep  n=" + std::to_string(n)
                    + "  M=" + std::to_string(M) + "  Q=" + std::to_string(Q)
                    + "  s=" + std::to_string(M) + "  SNR=" + std::to_string(snr)
                    + "  nu=" + std::to_string(nu)
                    + "  noise=" + scen_str + "  linkage=" + lnk_str,
                    /*include_base_trex=*/false);
            }
        }
    }


    // ==========================================================================
    //  Section 3: MC Q sweep (block size)
    //  Fixed: n=150, M=5, rho=0.8, SNR=2, nu=3.0.  s = M = 5 throughout.
    //  Swept: Q in {5, 10, ..., 50}; p = M*Q varies.
    //  Outer loops: h_noise scenario x linkage.
    // ==========================================================================
    if (false)
    {
        const std::vector<double> q_grid = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};

        for (const auto& [scen_str, h_noise] : scenarios) {
            for (const auto& [lnk_str, lnk_val] : linkages) {
                auto da_ctrl = make_da_bt_control();
                da_ctrl.hc_linkage = lnk_val;

                run_param_sweep(
                    "da_ht_Q_" + scen_str + "_" + lnk_str,
                    "Q",
                    q_grid,
                    num_MC,
                    tFDR,
                    base_seed,
                    default_solvers(),
                    [&](double q_val, unsigned seed) {
                        int q_int = static_cast<int>(q_val);
                        int p_val = M * q_int;
                        return dgp_block_toeplitz_hvt(
                            n, p_val, M, q_int, rho, snr, seed,
                            /*heavy_X=*/true, h_noise, nu, /*random_support=*/false);
                    },
                    da_ctrl,
                    make_trex_control(K),
                    "HT-block Q sweep  n=" + std::to_string(n)
                    + "  M=" + std::to_string(M) + "  s=" + std::to_string(M)
                    + "  rho=" + std::to_string(rho) + "  SNR=" + std::to_string(snr)
                    + "  nu=" + std::to_string(nu)
                    + "  p=M*Q varies  noise=" + scen_str + "  linkage=" + lnk_str,
                    /*include_base_trex=*/false);
            }
        }
    }


    // ==========================================================================
    //  Section 4: MC M sweep (number of blocks)
    //  Fixed: n=150, Q=5, rho=0.8, SNR=2, nu=3.0.
    //  Swept: M in {1, 3, 5, 10, 15, 20, 30}; p = M*Q and s = M both vary.
    //  Outer loops: h_noise scenario x linkage.
    // ==========================================================================
    if (false)
    {
        const std::vector<double> m_grid = {1, 3, 5, 10, 15, 20, 30};

        for (const auto& [scen_str, h_noise] : scenarios) {
            for (const auto& [lnk_str, lnk_val] : linkages) {
                auto da_ctrl = make_da_bt_control();
                da_ctrl.hc_linkage = lnk_val;

                run_param_sweep(
                    "da_ht_M_" + scen_str + "_" + lnk_str,
                    "M",
                    m_grid,
                    num_MC,
                    tFDR,
                    base_seed,
                    default_solvers(),
                    [&](double m_val, unsigned seed) {
                        int m_int = static_cast<int>(m_val);
                        int p_val = m_int * Q;
                        return dgp_block_toeplitz_hvt(
                            n, p_val, m_int, Q, rho, snr, seed,
                            /*heavy_X=*/true, h_noise, nu, /*random_support=*/false);
                    },
                    da_ctrl,
                    make_trex_control(K),
                    "HT-block M sweep  n=" + std::to_string(n)
                    + "  Q=" + std::to_string(Q)
                    + "  rho=" + std::to_string(rho) + "  SNR=" + std::to_string(snr)
                    + "  nu=" + std::to_string(nu)
                    + "  p=M*Q and s=M vary  noise=" + scen_str + "  linkage=" + lnk_str,
                    /*include_base_trex=*/false);
            }
        }
    }


    // ==========================================================================
    //  Section 5: MC tFDR sweep
    //  Fixed: n=150, M=5, Q=5 (p=25, s=5), rho=0.8, SNR=2, nu=3.0.
    //  Swept: tFDR in {0.05, 0.10, ..., 0.50}.
    //  Outer loops: h_noise scenario x linkage.
    //  (tFDR is a selector parameter -- run_mc_trials called per solver x grid point.)
    // ==========================================================================
    if (false)
    {
        std::vector<double> tfdr_grid;
        for (int i = 1; i <= 10; ++i) tfdr_grid.push_back(0.05 * i);

        const auto ngrid = static_cast<Eigen::Index>(tfdr_grid.size());

        for (const auto& [scen_str, h_noise] : scenarios) {
            for (const auto& [lnk_str, lnk_val] : linkages) {
                auto da_ctrl = make_da_bt_control();
                da_ctrl.hc_linkage = lnk_val;

                auto solvers = default_solvers();
                std::map<std::string, Eigen::VectorXd> fdr_map, tpr_map,
                                                       sd_fdr_map, sd_tpr_map,
                                                       avg_L_map, avg_T_map;

                for (const auto& sol : solvers) {
                    fdr_map[sol.name]    = Eigen::VectorXd::Zero(ngrid);
                    tpr_map[sol.name]    = Eigen::VectorXd::Zero(ngrid);
                    sd_fdr_map[sol.name] = Eigen::VectorXd::Zero(ngrid);
                    sd_tpr_map[sol.name] = Eigen::VectorXd::Zero(ngrid);
                    avg_L_map[sol.name]  = Eigen::VectorXd::Zero(ngrid);
                    avg_T_map[sol.name]  = Eigen::VectorXd::Zero(ngrid);
                }

                auto trex_ctrl = make_trex_control(K);

                for (const auto& sol : solvers) {
                    trex_ctrl.solver_type           = sol.type;
                    trex_ctrl.solver_params.rho_afs = sol.rho_afs;
                    trex_ctrl.tloop_stagnation_stop = sol.use_stagnation;

                    for (Eigen::Index gi = 0; gi < ngrid; ++gi) {
                        double tfdr_val = tfdr_grid[static_cast<std::size_t>(gi)];
                        std::string lbl = "tFDR = " + std::to_string(tfdr_val);

                        auto dgp_fn = [&](unsigned seed) {
                            return dgp_block_toeplitz_hvt(
                                n, base_p, M, Q, rho, snr, seed,
                                /*heavy_X=*/true, h_noise, nu, /*random_support=*/false);
                        };

                        unsigned seed_offset = static_cast<unsigned>(base_seed)
                                             + static_cast<unsigned>(gi) * 10000u;

                        auto res = run_mc_trials(
                            num_MC, lbl, dgp_fn, tfdr_val, trex_ctrl, da_ctrl, seed_offset);

                        fdr_map[sol.name](gi)    = res.avg_fdr;
                        tpr_map[sol.name](gi)    = res.avg_tpr;
                        sd_fdr_map[sol.name](gi) = res.sd_fdr;
                        sd_tpr_map[sol.name](gi) = res.sd_tpr;
                        avg_L_map[sol.name](gi)  = res.avg_L;
                        avg_T_map[sol.name](gi)  = res.avg_T;
                    }
                }

                save_and_print_grid_results(
                    "da_ht_tFDR_" + scen_str + "_" + lnk_str,
                    "tFDR",
                    tfdr_grid,
                    num_MC,
                    solvers,
                    fdr_map, tpr_map, sd_fdr_map, sd_tpr_map, avg_L_map, avg_T_map,
                    "HT-block tFDR sweep  n=" + std::to_string(n)
                    + "  M=" + std::to_string(M) + "  Q=" + std::to_string(Q)
                    + "  p=" + std::to_string(base_p) + "  s=" + std::to_string(M)
                    + "  rho=" + std::to_string(rho) + "  SNR=" + std::to_string(snr)
                    + "  nu=" + std::to_string(nu)
                    + "  noise=" + scen_str + "  linkage=" + lnk_str);
            }
        }
    }


    // ==========================================================================
    //  Section 6: Linkage sweep
    //  Fixed: n=150, M=5, Q=5 (p=25, s=5), rho=0.8, SNR=2, nu=3.0, tFDR=0.2.
    //  Swept: linkage encoded as {1.0=Single, 2.0=Complete, 3.0=Average}.
    //  Outer loop: h_noise scenario.
    // ==========================================================================
    if (false)
    {
        const std::vector<std::pair<std::string, hac::LinkageMethod>> all_linkages = {
            {"Single",   hac::LinkageMethod::Single},
            {"Complete", hac::LinkageMethod::Complete},
            {"Average",  hac::LinkageMethod::Average}
        };
        const std::vector<double> lnk_vals_d = {1.0, 2.0, 3.0};
        const auto ngrid = static_cast<Eigen::Index>(lnk_vals_d.size());

        for (const auto& [scen_str, h_noise] : scenarios) {
            auto solvers = default_solvers();
            std::map<std::string, Eigen::VectorXd> fdr_map, tpr_map,
                                                   sd_fdr_map, sd_tpr_map,
                                                   avg_L_map, avg_T_map;

            for (const auto& sol : solvers) {
                fdr_map[sol.name]    = Eigen::VectorXd::Zero(ngrid);
                tpr_map[sol.name]    = Eigen::VectorXd::Zero(ngrid);
                sd_fdr_map[sol.name] = Eigen::VectorXd::Zero(ngrid);
                sd_tpr_map[sol.name] = Eigen::VectorXd::Zero(ngrid);
                avg_L_map[sol.name]  = Eigen::VectorXd::Zero(ngrid);
                avg_T_map[sol.name]  = Eigen::VectorXd::Zero(ngrid);
            }

            auto trex_ctrl = make_trex_control(K);

            for (const auto& sol : solvers) {
                trex_ctrl.solver_type           = sol.type;
                trex_ctrl.solver_params.rho_afs = sol.rho_afs;
                trex_ctrl.tloop_stagnation_stop = sol.use_stagnation;

                for (Eigen::Index gi = 0; gi < ngrid; ++gi) {
                    const auto& [lnk_str, lnk_val] = all_linkages[static_cast<std::size_t>(gi)];
                    auto da_ctrl = make_da_bt_control();
                    da_ctrl.hc_linkage = lnk_val;

                    std::string lbl = "Linkage = " + lnk_str;
                    unsigned seed_offset = static_cast<unsigned>(base_seed)
                                         + static_cast<unsigned>(gi) * 10000u;

                    auto dgp_fn = [&](unsigned seed) {
                        return dgp_block_toeplitz_hvt(
                            n, base_p, M, Q, rho, snr, seed,
                            /*heavy_X=*/true, h_noise, nu, /*random_support=*/false);
                    };

                    auto res = run_mc_trials(
                        num_MC, lbl, dgp_fn, tFDR, trex_ctrl, da_ctrl, seed_offset);

                    fdr_map[sol.name](gi)    = res.avg_fdr;
                    tpr_map[sol.name](gi)    = res.avg_tpr;
                    sd_fdr_map[sol.name](gi) = res.sd_fdr;
                    sd_tpr_map[sol.name](gi) = res.sd_tpr;
                    avg_L_map[sol.name](gi)  = res.avg_L;
                    avg_T_map[sol.name](gi)  = res.avg_T;
                }
            }

            save_and_print_grid_results(
                "da_ht_linkage_" + scen_str,
                "Linkage (1=Single,2=Complete,3=Average)",
                lnk_vals_d,
                num_MC,
                solvers,
                fdr_map, tpr_map, sd_fdr_map, sd_tpr_map, avg_L_map, avg_T_map,
                "HT-block Linkage sweep  n=" + std::to_string(n)
                + "  M=" + std::to_string(M) + "  Q=" + std::to_string(Q)
                + "  p=" + std::to_string(base_p) + "  s=" + std::to_string(M)
                + "  rho=" + std::to_string(rho) + "  SNR=" + std::to_string(snr)
                + "  nu=" + std::to_string(nu)
                + "  tFDR=" + std::to_string(tFDR) + "  noise=" + scen_str);
        }
    }


    std::cout << "\nHeavy-tailed block BT sweeps demo complete.\n";
    return 0;
}
