// ==============================================================================
// demo_trex_da_03b_nn_ar.cpp
// ==============================================================================
/**
 * @file demo_trex_da_03b_nn_ar.cpp
 *
 * @brief DA+NN selector applied to AR(1) data.
 *
 * @details
 *  Uses an AR(1) DGP but runs DA-TRex with DAMethod::NN, exploring whether the
 *  nearest-neighbour aggregation can track the AR(1) correlation structure.
 *  Companion to demo_trex_da_03_nn.cpp (NN DGP).
 *
 *  Section 1: Single-run demo (n=150, p=500, s=5, rho=0.7, SNR=2.0, Random).
 *
 *  Section 2: MC SNR sweep (n=300, p=1000, rho=0.7; CappedSpread vs Random;
 *             3 solvers + base T-Rex comparison).
 *
 *  Section 3: MC rho sweep (n=300, p=1000, SNR=2.0; dual support).
 *
 *  Section 4: 2D SNR × rho sweep (n=300, p=1000; all 3 solvers; CappedSpread
 *             and Random sub-sections).
 */
// ==============================================================================

#include "trex_da_sim_common.hpp"

using namespace da_sim;

int main() {

    std::cout.setf(std::ios::unitbuf);

    // ── shared grids ──────────────────────────────────────────────────────────
    const std::vector<double> snr_grid = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0};
    const std::vector<double> rho_grid = {
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    // ── shared MC parameters ─────────────────────────────────────────────────
    constexpr int    mc_n      = 300;
    constexpr int    mc_p      = 1000;
    constexpr int    mc_s      = 10;
    constexpr double mc_rho    = 0.7;
    constexpr double mc_snr    = 2.0;
    constexpr double tFDR      = 0.2;
    constexpr int    K         = 20;
    constexpr int    num_MC    = 200;
    constexpr int    base_seed = 2026;
    constexpr int    max_gap   = 20;


    // ══════════════════════════════════════════════════════════════════════════
    //  Section 1: Single-run demo — AR(1) data with DA+NN
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        constexpr int    n   = 150;
        constexpr int    p   = 500;
        constexpr int    s   = 5;
        constexpr double rho = 0.7;
        constexpr double snr = 2.0;
        constexpr double amp = 3.0;
        const auto policy = SupportPolicy::Random;

        cdianostics::print_section_header(
            "AR(1)+NN Demo  |  n=" + std::to_string(n)
            + "  p=" + std::to_string(p) + "  s=" + std::to_string(s)
            + "  rho=" + std::to_string(rho)
            + "  support=" + support_policy_label(policy));

        auto dat = dgp_ar1(n, p,
                           make_support(policy, s, p, 0,
                                        static_cast<unsigned>(base_seed)),
                           amp, rho, snr,
                           static_cast<unsigned>(base_seed));

        auto trex_ctrl = make_trex_control(K);
        trex_ctrl.solver_type = SolverTypeForTRex::TOMP;

        TRexDAControlParameter da_ctrl;
        da_ctrl.method = DAMethod::NN;

        Eigen::Map<Eigen::MatrixXd> X_map(
            dat.X.data(), dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        TRexDASelector selector(X_map,
                                y_map,
                                tFDR,
                                da_ctrl,
                                trex_ctrl,
                                base_seed,
                                true);
        selector.select();

        auto m = evaluate_selection("AR(1)+NN DA+NN",
                                    selector.getSelectedIndices(),
                                    dat.true_support);
        print_single_result(m, tFDR);
    }


    // ══════════════════════════════════════════════════════════════════════════
    //  Section 2: MC SNR sweep — AR(1) + DA+NN
    //  Fixed: rho=0.7.  CappedSpread(max_gap=20) vs Random.
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        TRexDAControlParameter da_ctrl;
        da_ctrl.method = DAMethod::NN;

        const auto support_fixed =
            capped_spread_support(mc_s, mc_p, max_gap);

        const std::string hdr_base =
            "AR(1)+NN rho=" + std::to_string(mc_rho)
            + ", n=" + std::to_string(mc_n)
            + ", p=" + std::to_string(mc_p);

        // A — CappedSpread
        run_snr_sweep(
            "da_nn_ar_snr_capped",
            snr_grid,
            num_MC, tFDR, base_seed,
            default_solvers(),
            [&](double snr, unsigned seed) {
                return dgp_ar1(mc_n, mc_p, support_fixed,
                               /*amplitude=*/3.0, mc_rho, snr, seed);
            },
            da_ctrl, make_trex_control(K),
            hdr_base + ", support=CappedSpread(max_gap="
                + std::to_string(max_gap) + ")",
            /*include_base_trex=*/true);

        // B — Random
        run_snr_sweep(
            "da_nn_ar_snr_random",
            snr_grid,
            num_MC, tFDR, base_seed,
            default_solvers(),
            [&](double snr, unsigned seed) {
                return dgp_ar1(mc_n, mc_p,
                               make_support(SupportPolicy::Random,
                                            mc_s, mc_p, 0, seed),
                               /*amplitude=*/3.0, mc_rho, snr, seed);
            },
            da_ctrl, make_trex_control(K),
            hdr_base + ", support=Random (per-trial)",
            /*include_base_trex=*/true);
    }


    // ══════════════════════════════════════════════════════════════════════════
    //  Section 3: MC rho sweep — AR(1) + DA+NN
    //  Fixed: SNR=2.0.  CappedSpread(max_gap=20) vs Random.
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        TRexDAControlParameter da_ctrl;
        da_ctrl.method = DAMethod::NN;

        const auto support_fixed =
            capped_spread_support(mc_s, mc_p, max_gap);

        const std::string hdr_base =
            "AR(1)+NN SNR=" + std::to_string(mc_snr)
            + ", n=" + std::to_string(mc_n)
            + ", p=" + std::to_string(mc_p);

        // A — CappedSpread
        run_param_sweep(
            "da_nn_ar_rho_capped", "Rho", rho_grid,
            num_MC, tFDR, base_seed, default_solvers(),
            [&](double rho, unsigned seed) {
                return dgp_ar1(mc_n, mc_p, support_fixed,
                               /*amplitude=*/3.0, rho, mc_snr, seed);
            },
            da_ctrl, make_trex_control(K),
            hdr_base + ", support=CappedSpread(max_gap="
                + std::to_string(max_gap) + ")",
            /*include_base_trex=*/true);


        // B — Random
        run_param_sweep(
            "da_nn_ar_rho_random", "Rho", rho_grid,
            num_MC, tFDR, base_seed, default_solvers(),
            [&](double rho, unsigned seed) {
                return dgp_ar1(mc_n, mc_p,
                               make_support(SupportPolicy::Random,
                                            mc_s, mc_p, 0, seed),
                               /*amplitude=*/3.0, rho, mc_snr, seed);
            },
            da_ctrl, make_trex_control(K),
            hdr_base + ", support=Random (per-trial)",
            /*include_base_trex=*/true);
    }


    // ══════════════════════════════════════════════════════════════════════════
    //  Section 4: 2D SNR × rho sweep — AR(1) + DA+NN
    //  Fixed: amplitude=3.0.
    //  Outer loop: solver.  Inner: SNR (rows) × rho (cols).
    //
    //  Sub-section A: CappedSpread(max_gap=20) fixed support.
    //  Sub-section B: Random support (redrawn each trial via seed).
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        const std::vector<double> snr_2d = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0};
        const std::vector<double> rho_2d = {
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

        const auto n_snr = snr_2d.size();
        const auto n_rho = rho_2d.size();

        TRexDAControlParameter da_ctrl;
        da_ctrl.method = DAMethod::NN;

        const auto support_cs =
            capped_spread_support(mc_s, mc_p, max_gap);

        const std::vector<SolverInfo> solvers = default_solvers();

        // Inline matrix printer (rows = SNR, cols = rho)
        auto print_matrix = [&](const std::string& label,
                                const Eigen::MatrixXd& mat) {
            std::cout << "\n  " << label << ":\n";
            std::cout << "  " << std::left << std::setw(14) << "";
            for (double r : rho_2d)
                std::cout << std::right << std::setw(9)
                          << ("rho=" + [r](){
                                  std::ostringstream o;
                                  o << std::fixed << std::setprecision(1) << r;
                                  return o.str(); }());
            std::cout << "\n  " << std::string(14 + 9 * n_rho, '-') << "\n";
            for (std::size_t is = 0; is < n_snr; ++is) {
                std::ostringstream rl;
                rl << "SNR=" << std::fixed << std::setprecision(2) << snr_2d[is];
                std::cout << "  " << std::left << std::setw(14) << rl.str();
                for (std::size_t ir = 0; ir < n_rho; ++ir)
                    std::cout << std::right << std::fixed << std::setprecision(3)
                              << std::setw(9)
                              << mat(static_cast<Eigen::Index>(is),
                                     static_cast<Eigen::Index>(ir));
                std::cout << "\n";
            }
            std::cout << "\n";
        };

        cdianostics::print_section_header(
            "AR(1)+NN 2D SNR×rho sweep"
            "  n=" + std::to_string(mc_n)
            + "  p=" + std::to_string(mc_p)
            + "  s=" + std::to_string(mc_s)
            + "  " + std::to_string(num_MC) + " MC");

        for (const auto& sv : solvers) {
            std::cout << "\n══ Solver: " << sv.name << " ══\n";

            auto trex_ctrl = make_trex_control(K);
            trex_ctrl.solver_type           = sv.type;
            trex_ctrl.solver_params.rho_afs = sv.rho_afs;
            trex_ctrl.tloop_stagnation_stop  = sv.use_stagnation;

            Eigen::MatrixXd tpp_cs = Eigen::MatrixXd::Zero(
                static_cast<Eigen::Index>(n_snr),
                static_cast<Eigen::Index>(n_rho));
            Eigen::MatrixXd fdp_cs   = tpp_cs;
            Eigen::MatrixXd tpp_rand = tpp_cs;
            Eigen::MatrixXd fdp_rand = tpp_cs;

            const int total_cells = static_cast<int>(n_snr * n_rho);

            // ---- A: CappedSpread(max_gap=20) ----
            std::cout << "\n  [A] CappedSpread(max_gap=" << max_gap << ")\n\n";
            int cell_idx = 0;
            for (std::size_t is = 0; is < n_snr; ++is) {
                const double snr_val = snr_2d[is];
                for (std::size_t ir = 0; ir < n_rho; ++ir) {
                    ++cell_idx;
                    const double rho_val = rho_2d[ir];

                    std::ostringstream lbl;
                    lbl << "[" << cell_idx << "/" << total_cells
                        << "] SNR=" << std::fixed << std::setprecision(2) << snr_val
                        << " rho=" << std::fixed << std::setprecision(1) << rho_val;

                    auto res = run_mc_trials(
                        num_MC, lbl.str(),
                        [&](unsigned seed) {
                            return dgp_ar1(mc_n, mc_p, support_cs,
                                           /*amplitude=*/3.0,
                                           rho_val, snr_val, seed);
                        },
                        tFDR, trex_ctrl, da_ctrl,
                        static_cast<unsigned>(base_seed));

                    tpp_cs(static_cast<Eigen::Index>(is),
                           static_cast<Eigen::Index>(ir)) = res.avg_tpr;
                    fdp_cs(static_cast<Eigen::Index>(is),
                           static_cast<Eigen::Index>(ir)) = res.avg_fdr;

                    std::cout << "    done. TPP=" << std::fixed
                              << std::setprecision(3) << res.avg_tpr
                              << " FDP=" << res.avg_fdr << "\n\n";
                }
            }

            // ---- B: Random support ----
            std::cout << "\n  [B] Random support (redrawn each trial)\n\n";
            cell_idx = 0;
            for (std::size_t is = 0; is < n_snr; ++is) {
                const double snr_val = snr_2d[is];
                for (std::size_t ir = 0; ir < n_rho; ++ir) {
                    ++cell_idx;
                    const double rho_val = rho_2d[ir];

                    std::ostringstream lbl;
                    lbl << "[" << cell_idx << "/" << total_cells
                        << "] SNR=" << std::fixed << std::setprecision(2) << snr_val
                        << " rho=" << std::fixed << std::setprecision(1) << rho_val
                        << " Random";

                    auto res = run_mc_trials(
                        num_MC, lbl.str(),
                        [&](unsigned seed) {
                            return dgp_ar1(mc_n, mc_p,
                                           make_support(SupportPolicy::Random,
                                                        mc_s, mc_p, 0, seed),
                                           /*amplitude=*/3.0,
                                           rho_val, snr_val, seed);
                        },
                        tFDR, trex_ctrl, da_ctrl,
                        static_cast<unsigned>(base_seed));

                    tpp_rand(static_cast<Eigen::Index>(is),
                             static_cast<Eigen::Index>(ir)) = res.avg_tpr;
                    fdp_rand(static_cast<Eigen::Index>(is),
                             static_cast<Eigen::Index>(ir)) = res.avg_fdr;

                    std::cout << "    done. TPP=" << std::fixed
                              << std::setprecision(3) << res.avg_tpr
                              << " FDP=" << res.avg_fdr << "\n\n";
                }
            }

            // ---- Print results for this solver ----
            std::cout << "\n  Solver: " << sv.name
                      << "  (rows=SNR, cols=rho)\n";
            print_matrix("mean_TPP  [CappedSpread]", tpp_cs);
            print_matrix("mean_FDP  [CappedSpread]", fdp_cs);
            print_matrix("mean_TPP  [Random]",       tpp_rand);
            print_matrix("mean_FDP  [Random]",       fdp_rand);
        }
    }

    std::cout << "\nAR(1)+NN demo complete.\n";
    return 0;
}
