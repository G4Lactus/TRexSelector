// ==============================================================================
// demo_trex_da_03_nn.cpp
// ==============================================================================
/**
 * @file demo_trex_da_03_nn.cpp
 *
 * @brief DA-TRex demo and MC simulation for the nearest-neighbours (NN) DGP.
 *
 * @details
 *  All p predictors share a banded MA(kappa) covariance structure.
 *
 *  Section 1: Single-run demo (n=150, p=500, s=5, kappa=3, rho=0.7, SNR=2.0,
 *             Random support).
 *  Section 2: MC SNR sweep (n=300, p=1000, kappa=3, rho=0.7; CappedSpread vs
 *             Random support; 3 solvers + base T-Rex comparison).
 *  Section 3: MC rho sweep (n=300, p=1000, kappa=3, SNR=2.0; dual support).
 *  Section 4: MC kappa sweep (n=300, p=1000, rho=0.7, SNR=2.0; dual support).
 *  Section 5: 2D kappa x rho sweep (n=300, p=1000, SNR=2.0; all 3 solvers;
 *             CappedSpread and Random sub-sections).
 */
// ==============================================================================

#include "trex_da_sim_common.hpp"

using namespace da_sim;

int main() {

    std::cout.setf(std::ios::unitbuf);

    // ── shared grids ──────────────────────────────────────────────────────────
    const std::vector<double> snr_grid   = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0};
    const std::vector<double> rho_grid   = {
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    const std::vector<double> kappa_grid = {
        1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0};

    // ── shared MC parameters ─────────────────────────────────────────────────
    constexpr int    mc_n        = 300;
    constexpr int    mc_p        = 1000;
    constexpr int    mc_s        = 10;
    constexpr int    mc_kappa    = 3;
    constexpr double mc_rho      = 0.7;
    constexpr double mc_snr      = 2.0;
    constexpr double tFDR        = 0.2;
    constexpr int    K           = 20;
    constexpr int    num_MC      = 200;
    constexpr int    base_seed   = 2026;
    constexpr int    max_gap     = 20;


    // ══════════════════════════════════════════════════════════════════════════
    //  Section 1: Single-run demo
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        constexpr int    n      = 150;
        constexpr int    p      = 500;
        constexpr int    s      = 5;
        constexpr int    kappa  = 3;
        constexpr double rho    = 0.7;
        constexpr double snr    = 2.0;
        constexpr double amp    = 3.0;
        const auto policy = SupportPolicy::Random;

        cdianostics::print_section_header(
            "NN Demo  |  n=" + std::to_string(n)
            + "  p=" + std::to_string(p) + "  s=" + std::to_string(s)
            + "  kappa=" + std::to_string(kappa)
            + "  rho=" + std::to_string(rho)
            + "  support=" + support_policy_label(policy));

        auto dat = dgp_nn(n, p,
                          make_support(policy, s, p, 0,
                                       static_cast<unsigned>(base_seed)),
                          amp, kappa, rho, snr,
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

        auto m = evaluate_selection("NN DA+NN",
                                    selector.getSelectedIndices(),
                                    dat.true_support);
        print_single_result(m, tFDR);
    }


    // ══════════════════════════════════════════════════════════════════════════
    //  Section 2: MC SNR sweep — NN DGP
    //  CappedSpread (max_gap=20) vs Random support.
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        TRexDAControlParameter da_ctrl;
        da_ctrl.method = DAMethod::NN;

        const auto support_fixed =
            capped_spread_support(mc_s, mc_p, max_gap);

        const std::string hdr_base =
            "NN kappa=" + std::to_string(mc_kappa)
            + ", rho=" + std::to_string(mc_rho)
            + ", n=" + std::to_string(mc_n)
            + ", p=" + std::to_string(mc_p);

        // A — CappedSpread(max_gap=20)
        run_snr_sweep(
            "da_nn_snr_capped", snr_grid,
            num_MC, tFDR, base_seed, default_solvers(),
            [&](double snr, unsigned seed) {
                return dgp_nn(mc_n, mc_p, support_fixed,
                              /*amplitude=*/3.0, mc_kappa, mc_rho, snr, seed);
            },
            da_ctrl, make_trex_control(K),
            hdr_base + ", support=CappedSpread(max_gap="
                + std::to_string(max_gap) + ")",
            /*include_base_trex=*/true);

        // B — Random (redrawn each trial)
        run_snr_sweep(
            "da_nn_snr_random", snr_grid,
            num_MC, tFDR, base_seed, default_solvers(),
            [&](double snr, unsigned seed) {
                return dgp_nn(mc_n, mc_p,
                              make_support(SupportPolicy::Random,
                                           mc_s, mc_p, 0, seed),
                              /*amplitude=*/3.0, mc_kappa, mc_rho, snr, seed);
            },
            da_ctrl, make_trex_control(K),
            hdr_base + ", support=Random (per-trial)",
            /*include_base_trex=*/true);
    }


    // ══════════════════════════════════════════════════════════════════════════
    //  Section 3: MC rho sweep — NN DGP
    //  Fixed: kappa=3, SNR=2.0.  CappedSpread vs Random.
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        TRexDAControlParameter da_ctrl;
        da_ctrl.method = DAMethod::NN;

        const auto support_fixed =
            capped_spread_support(mc_s, mc_p, max_gap);

        const std::string hdr_base =
            "NN kappa=" + std::to_string(mc_kappa)
            + ", SNR=" + std::to_string(mc_snr)
            + ", n=" + std::to_string(mc_n)
            + ", p=" + std::to_string(mc_p);

        // A — CappedSpread
        run_param_sweep(
            "da_nn_rho_capped", "Rho", rho_grid,
            num_MC, tFDR, base_seed, default_solvers(),
            [&](double rho, unsigned seed) {
                return dgp_nn(mc_n, mc_p, support_fixed,
                              /*amplitude=*/3.0, mc_kappa, rho, mc_snr, seed);
            },
            da_ctrl, make_trex_control(K),
            hdr_base + ", support=CappedSpread(max_gap="
                + std::to_string(max_gap) + ")",
            /*include_base_trex=*/true);

        // B — Random
        run_param_sweep(
            "da_nn_rho_random", "Rho", rho_grid,
            num_MC, tFDR, base_seed, default_solvers(),
            [&](double rho, unsigned seed) {
                return dgp_nn(mc_n, mc_p,
                              make_support(SupportPolicy::Random,
                                           mc_s, mc_p, 0, seed),
                              /*amplitude=*/3.0, mc_kappa, rho, mc_snr, seed);
            },
            da_ctrl, make_trex_control(K),
            hdr_base + ", support=Random (per-trial)",
            /*include_base_trex=*/true);
    }


    // ══════════════════════════════════════════════════════════════════════════
    //  Section 4: MC kappa sweep — NN DGP
    //  Fixed: rho=0.7, SNR=2.0.  CappedSpread vs Random.
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        TRexDAControlParameter da_ctrl;
        da_ctrl.method = DAMethod::NN;

        const auto support_fixed =
            capped_spread_support(mc_s, mc_p, max_gap);

        const std::string hdr_base =
            "NN rho=" + std::to_string(mc_rho)
            + ", SNR=" + std::to_string(mc_snr)
            + ", n=" + std::to_string(mc_n)
            + ", p=" + std::to_string(mc_p);

        // A — CappedSpread
        run_param_sweep(
            "da_nn_kappa_capped", "Kappa", kappa_grid,
            num_MC, tFDR, base_seed, default_solvers(),
            [&](double kappa_d, unsigned seed) {
                return dgp_nn(mc_n, mc_p, support_fixed,
                              /*amplitude=*/3.0,
                              static_cast<int>(kappa_d),
                              mc_rho, mc_snr, seed);
            },
            da_ctrl, make_trex_control(K),
            hdr_base + ", support=CappedSpread(max_gap="
                + std::to_string(max_gap) + ")",
            /*include_base_trex=*/true);

        // B — Random
        run_param_sweep(
            "da_nn_kappa_random", "Kappa", kappa_grid,
            num_MC, tFDR, base_seed, default_solvers(),
            [&](double kappa_d, unsigned seed) {
                return dgp_nn(mc_n, mc_p,
                              make_support(SupportPolicy::Random,
                                           mc_s, mc_p, 0, seed),
                              /*amplitude=*/3.0,
                              static_cast<int>(kappa_d),
                              mc_rho, mc_snr, seed);
            },
            da_ctrl, make_trex_control(K),
            hdr_base + ", support=Random (per-trial)",
            /*include_base_trex=*/true);
    }


    // ══════════════════════════════════════════════════════════════════════════
    //  Section 5: 2D kappa × rho sweep — NN DGP
    //  Fixed: SNR=2.0, amplitude=3.0.
    //  Outer loop: solver.  Inner: kappa × rho.
    //
    //  Sub-section A: CappedSpread(max_gap=20) fixed support.
    //  Sub-section B: Random support (redrawn each trial via seed).
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        const std::vector<int>    kappa_2d = {1, 2, 3, 5, 7, 10, 15};
        const std::vector<double> rho_2d   = {0.1, 0.3, 0.5, 0.7, 0.8, 0.9};

        const auto n_kappa = kappa_2d.size();
        const auto n_rho   = rho_2d.size();

        TRexDAControlParameter da_ctrl;
        da_ctrl.method = DAMethod::NN;

        const auto support_cs =
            capped_spread_support(mc_s, mc_p, max_gap);

        const std::vector<SolverInfo> solvers = default_solvers();

        // Inline matrix printer (rows = kappa, cols = rho)
        auto print_matrix = [&](const std::string& label,
                                const Eigen::MatrixXd& mat) {
            std::cout << "\n  " << label << ":\n";
            std::cout << "  " << std::left << std::setw(12) << "";
            for (double r : rho_2d)
                std::cout << std::right << std::setw(9)
                          << ("rho=" + [r](){
                                  std::ostringstream o;
                                  o << std::fixed << std::setprecision(1) << r;
                                  return o.str(); }());
            std::cout << "\n  " << std::string(12 + 9 * n_rho, '-') << "\n";
            for (std::size_t ik = 0; ik < n_kappa; ++ik) {
                std::cout << "  " << std::left << std::setw(12)
                          << ("kappa=" + std::to_string(kappa_2d[ik]));
                for (std::size_t ir = 0; ir < n_rho; ++ir)
                    std::cout << std::right << std::fixed << std::setprecision(3)
                              << std::setw(9)
                              << mat(static_cast<Eigen::Index>(ik),
                                     static_cast<Eigen::Index>(ir));
                std::cout << "\n";
            }
            std::cout << "\n";
        };

        cdianostics::print_section_header(
            "NN 2D kappa×rho sweep"
            "  n=" + std::to_string(mc_n)
            + "  p=" + std::to_string(mc_p)
            + "  s=" + std::to_string(mc_s)
            + "  SNR=" + std::to_string(mc_snr)
            + "  " + std::to_string(num_MC) + " MC");

        for (const auto& sv : solvers) {
            std::cout << "\n══ Solver: " << sv.name << " ══\n";

            auto trex_ctrl = make_trex_control(K);
            trex_ctrl.solver_type           = sv.type;
            trex_ctrl.solver_params.rho_afs = sv.rho_afs;
            trex_ctrl.tloop_stagnation_stop  = sv.use_stagnation;

            Eigen::MatrixXd tpp_cs = Eigen::MatrixXd::Zero(
                static_cast<Eigen::Index>(n_kappa),
                static_cast<Eigen::Index>(n_rho));
            Eigen::MatrixXd fdp_cs = tpp_cs;
            Eigen::MatrixXd tpp_rand = tpp_cs;
            Eigen::MatrixXd fdp_rand = tpp_cs;

            const int total_cells =
                static_cast<int>(n_kappa * n_rho);

            // ---- A: CappedSpread(max_gap=20) ----
            std::cout << "\n  [A] CappedSpread(max_gap=" << max_gap << ")\n\n";
            int cell_idx = 0;
            for (std::size_t ik = 0; ik < n_kappa; ++ik) {
                const int kappa_val = kappa_2d[ik];
                for (std::size_t ir = 0; ir < n_rho; ++ir) {
                    ++cell_idx;
                    const double rho_val = rho_2d[ir];

                    std::ostringstream lbl;
                    lbl << "[" << cell_idx << "/" << total_cells
                        << "] kappa=" << kappa_val
                        << " rho=" << std::fixed << std::setprecision(1) << rho_val;

                    auto res = run_mc_trials(
                        num_MC, lbl.str(),
                        [&](unsigned seed) {
                            return dgp_nn(mc_n, mc_p, support_cs,
                                          /*amplitude=*/3.0,
                                          kappa_val, rho_val, mc_snr, seed);
                        },
                        tFDR, trex_ctrl, da_ctrl,
                        static_cast<unsigned>(base_seed));

                    tpp_cs(static_cast<Eigen::Index>(ik),
                           static_cast<Eigen::Index>(ir)) = res.avg_tpr;
                    fdp_cs(static_cast<Eigen::Index>(ik),
                           static_cast<Eigen::Index>(ir)) = res.avg_fdr;

                    std::cout << "    done. TPP=" << std::fixed
                              << std::setprecision(3) << res.avg_tpr
                              << " FDP=" << res.avg_fdr << "\n\n";
                }
            }

            // ---- B: Random support (redrawn each trial) ----
            std::cout << "\n  [B] Random support (redrawn each trial)\n\n";
            cell_idx = 0;
            for (std::size_t ik = 0; ik < n_kappa; ++ik) {
                const int kappa_val = kappa_2d[ik];
                for (std::size_t ir = 0; ir < n_rho; ++ir) {
                    ++cell_idx;
                    const double rho_val = rho_2d[ir];

                    std::ostringstream lbl;
                    lbl << "[" << cell_idx << "/" << total_cells
                        << "] kappa=" << kappa_val
                        << " rho=" << std::fixed << std::setprecision(1) << rho_val
                        << " Random";

                    auto res = run_mc_trials(
                        num_MC, lbl.str(),
                        [&](unsigned seed) {
                            return dgp_nn(mc_n, mc_p,
                                          make_support(SupportPolicy::Random,
                                                       mc_s, mc_p, 0, seed),
                                          /*amplitude=*/3.0,
                                          kappa_val, rho_val, mc_snr, seed);
                        },
                        tFDR, trex_ctrl, da_ctrl,
                        static_cast<unsigned>(base_seed));

                    tpp_rand(static_cast<Eigen::Index>(ik),
                             static_cast<Eigen::Index>(ir)) = res.avg_tpr;
                    fdp_rand(static_cast<Eigen::Index>(ik),
                             static_cast<Eigen::Index>(ir)) = res.avg_fdr;

                    std::cout << "    done. TPP=" << std::fixed
                              << std::setprecision(3) << res.avg_tpr
                              << " FDP=" << res.avg_fdr << "\n\n";
                }
            }

            // ---- Print results for this solver ----
            std::cout << "\n  Solver: " << sv.name
                      << "  (rows=kappa, cols=rho)\n";
            print_matrix("mean_TPP  [CappedSpread]", tpp_cs);
            print_matrix("mean_FDP  [CappedSpread]", fdp_cs);
            print_matrix("mean_TPP  [Random]",       tpp_rand);
            print_matrix("mean_FDP  [Random]",       fdp_rand);
        }
    }

    std::cout << "\nNN demo complete.\n";
    return 0;
}
