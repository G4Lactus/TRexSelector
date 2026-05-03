// ==============================================================================
// demo_trx_da_01_ar1.cpp
// ==============================================================================
/**
 * @file demo_trx_da_01_ar1.cpp
 *
 * @brief DA-TRex demo and MC simulation for the AR(1) Toeplitz DGP.
 *
 * @details
 *  Part 1: Single-run demo (n=150, p=500, s=5, rho=0.7, SNR=2.0).
 *
 *  Part 2: MC simulation sweeping SNR (n=300, p=1000, s=10, rho=0.7, 200 MC,
 *          3 solvers).
 *
 *  Part 3: MC simulation sweeping rho (n=300, p=1000, s=10, SNR=2.0, 200 MC,
 *          3 solvers).
 *
 *  Part 4: 2D Gap x Rho sweep. Explores the kappa-boundary where
 *          gap < kappa collapses TPP. The support is generated via
 *          CappedSpread(gap) vs Random policies.
 */
// ==============================================================================

#include "trex_da_sim_common.hpp"

// ==============================================================================

// Namespace alias
using namespace da_sim;


int main() {

    std::cout.setf(std::ios::unitbuf);

    // ==========================================================================
    //  Part 1: Single-run demo
    // ==========================================================================
    if (true)
    {
        // Simulation config
        SimConfig cfg;
        cfg.n = 150;
        cfg.p = 500;
        cfg.s = 5;
        constexpr double rho = 0.7;

        // Print header
        cdianostics::print_section_header(
            "AR(1) Demo  |  n=" + std::to_string(cfg.n)
            + "  p=" + std::to_string(cfg.p) + "  s=" + std::to_string(cfg.s)
            + "  rho=" + std::to_string(rho)
            + "  support=Random");

        // Generate data
        auto dat = dgp_ar1(cfg.n,
                           cfg.p,
                           make_support(SupportPolicy::Random,
                                        cfg.s,
                                        cfg.p,
                                        0,
                                        static_cast<unsigned>(cfg.base_seed)),
                           cfg.amplitude,
                           rho,
                           cfg.snr,
                           static_cast<unsigned>(cfg.base_seed)
                           );

        // TRex control parameters
        auto trex_ctrl = make_trex_control(cfg.K);
        trex_ctrl.solver_type = SolverTypeForTRex::TLARS;
        trex_ctrl.tloop_stagnation_stop = false;

        // DA control parameters
        TRexDAControlParameter da_ctrl;
        da_ctrl.method = DAMethod::AR1;

        // Set Map views for X and y
        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(),
                                          dat.X.rows(),
                                          dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        // Setup selector and perform selection
        TRexDASelector selector(X_map,
                                y_map,
                                cfg.tFDR,
                                da_ctrl,
                                trex_ctrl,
                                cfg.base_seed,
                                /*verbose=*/true);
        selector.select();

        // Evaluate and print results
        auto m = evaluate_selection(
            "AR(1) DA+AR1",
            selector.getSelectedIndices(),
            dat.true_support
        );
        print_single_result(m, cfg.tFDR);
    }


    // ==========================================================================
    //  Part 2: Monte Carlo simulation — SNR sweep
    // ==========================================================================
    if (false)
    {
        // Simulation config
        SimConfig cfg;
        cfg.n = 300;
        cfg.p = 1000;
        cfg.s = 10;
        // amplitude = 3.0 and tFDR = 0.2 from SimConfig defaults
        constexpr int max_gap = 20;
        constexpr double rho = 0.7;

        // TRex control parameters
        TRexDAControlParameter da_ctrl;
        da_ctrl.method = DAMethod::AR1;

        const std::vector<double> snr_grid = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0};
        const std::string hdr_base =
            "AR(1) rho=" + std::to_string(rho) +
            ", n=" + std::to_string(cfg.n) +
            ", p=" + std::to_string(cfg.p);

        // --- Run A: CappedSpread(max_gap=20) ---
        run_snr_sweep(
            "da_ar1_snr_capped",
            snr_grid,
            cfg.num_MC,
            cfg.tFDR,
            cfg.base_seed,
            default_solvers(),
            [&](double snr, unsigned seed) {
                return dgp_ar1(cfg.n,
                               cfg.p,
                               make_support(
                                    SupportPolicy::CappedSpread,
                                    cfg.s,
                                    cfg.p,
                                    max_gap,
                                    seed),
                               cfg.amplitude, rho, snr, seed);
            },
            da_ctrl,
            make_trex_control(cfg.K),
            hdr_base + ", support=CappedSpread(max_gap=" +
                          std::to_string(max_gap) + ")",
            true);

        // --- Run B: Random support (redrawn per trial) ---
        run_snr_sweep(
            "da_ar1_snr_random",
            snr_grid,
            cfg.num_MC,
            cfg.tFDR,
            cfg.base_seed,
            default_solvers(),
            [&](double snr, unsigned seed) {
                return dgp_ar1(cfg.n,
                               cfg.p,
                               make_support(
                                    SupportPolicy::Random,
                                    cfg.s,
                                    cfg.p,
                                    0,
                                    seed),
                               cfg.amplitude, rho, snr, seed);
            },
            da_ctrl,
            make_trex_control(cfg.K),
            hdr_base + ", support=Random (redrawn per trial)",
            true);
    }


    // ==========================================================================
    //  Part 3: Monte Carlo simulation — Rho sweep
    // ==========================================================================
    if (false)
    {
        // Simulation config
        SimConfig cfg;
        cfg.n = 300;
        cfg.p = 1000;
        cfg.s = 10;
        // amplitude = 3.0 and tFDR = 0.2 from SimConfig defaults
        constexpr int max_gap = 20;
        constexpr double snr_fixed = 2.0;

        // TRex control parameters
        TRexDAControlParameter da_ctrl;
        da_ctrl.method = DAMethod::AR1;

        const std::vector<double> rho_grid = {
            0.0, 0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8, 0.9
        };
        const std::string hdr_base =
            "AR(1) SNR=" + std::to_string(snr_fixed) +
            ", n=" + std::to_string(cfg.n) +
            ", p=" + std::to_string(cfg.p);

        // --- Run A: CappedSpread(max_gap=20) ---
        run_param_sweep(
            "da_ar1_rho_capped",
            "Rho",
            rho_grid,
            cfg.num_MC,
            cfg.tFDR,
            cfg.base_seed,
            default_solvers(),
            [&](double rho_val, unsigned seed) {
                return dgp_ar1(cfg.n,
                               cfg.p,
                               make_support(
                                SupportPolicy::CappedSpread,
                                cfg.s, cfg.p, max_gap, seed),
                               cfg.amplitude, rho_val, snr_fixed, seed);
            },
            da_ctrl,
            make_trex_control(cfg.K),
            hdr_base + ", support=CappedSpread(max_gap=" +
                         std::to_string(max_gap) + ")",
            true);

        // --- Run B: Random support (redrawn per trial) ---
        run_param_sweep(
            "da_ar1_rho_random",
            "Rho", rho_grid,
            cfg.num_MC,
            cfg.tFDR,
            cfg.base_seed,
            default_solvers(),
            [&](double rho_val, unsigned seed) {
                return dgp_ar1(cfg.n, cfg.p,
                               make_support(SupportPolicy::Random,
                                            cfg.s, cfg.p, 0, seed),
                               cfg.amplitude, rho_val, snr_fixed, seed);
            },
            da_ctrl,
            make_trex_control(cfg.K),
            hdr_base + ", support=Random (redrawn per trial)",
            true);
    }


    // ==========================================================================
    //  Part 4: 2D Gap x Rho sweep
    //
    //  Runs 200 MC trials per (gap, rho) cell for two support types:
    //    A) CappedSpread(gap) — deterministic, evenly-spaced support.
    //    B) Random           — support redrawn each trial (not gap-dependent).
    //
    //  The kappa boundary (kappa = ceiling(log(0.02) / log(rho))) is the DA+AR1
    //  correction window half-width.  When gap < kappa, active predictors fall
    //  inside each other's correction windows and TPP collapses.
    //
    //  gap_grid : {100, 50, 20, 15, 10, 5, 1}
    //  rho_grid : {0.0, 0.1, ..., 0.9}
    // ==========================================================================
    if (false)
    {
        SimConfig cfg;
        cfg.n = 300; cfg.p = 1000; cfg.s = 10;
        // amplitude = 3.0 and tFDR = 0.2 from SimConfig defaults (match R)
        cfg.num_MC = 200;
        constexpr double snr_fixed = 2.0;

        const std::vector<int> gap_grid = {100, 50, 20, 15, 10, 5, 1};
        const std::vector<double> rho_grid = {
            0.0, 0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8, 0.9
        };

        const auto n_gap = gap_grid.size();
        const auto n_rho = rho_grid.size();

        // Kappa boundary: ceil(log(0.02) / log(rho)).
        // For rho=0 (independent predictors) kappa=0 → gap always sufficient.
        std::vector<int> kappa_boundary(n_rho);
        for (std::size_t ir = 0; ir < n_rho; ++ir) {
            if (rho_grid[ir] <= 0.0)
                kappa_boundary[ir] = 0;
            else
                kappa_boundary[ir] = static_cast<int>(
                    std::ceil(std::log(0.02) / std::log(rho_grid[ir])));
        }

        // Result matrices [rho][gap] + Random baseline vector [rho]
        Eigen::MatrixXd tpp_cs = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(n_rho),
            static_cast<Eigen::Index>(n_gap));
        Eigen::MatrixXd fdp_cs = tpp_cs;
        Eigen::VectorXd tpp_rand = Eigen::VectorXd::Zero(
            static_cast<Eigen::Index>(n_rho));
        Eigen::VectorXd fdp_rand = tpp_rand;

        // Solver setup: TLARS only (matching R reference)
        auto trex_ctrl = make_trex_control(cfg.K);
        trex_ctrl.solver_type = SolverTypeForTRex::TLARS;
        trex_ctrl.tloop_stagnation_stop = false;

        TRexDAControlParameter da_ctrl;
        da_ctrl.method = DAMethod::AR1;

        const unsigned base_offset = static_cast<unsigned>(cfg.base_seed);

        // --- Header ---
        cdianostics::print_section_header("AR(1) Part 4: Gap x Rho sweep");
        std::cout << "  n=" << cfg.n << ", p=" << cfg.p << ", s=" << cfg.s
                  << ", SNR=" << snr_fixed << ", " << cfg.num_MC << " MC"
                  << ", tFDR=" << cfg.tFDR
                  << ", amplitude=" << cfg.amplitude << "\n";
        std::cout << "  gap_grid: {";
        for (std::size_t i = 0; i < n_gap; ++i)
            std::cout << (i ? ", " : "") << gap_grid[i];
        std::cout << "}\n  rho_grid: {";
        for (std::size_t i = 0; i < n_rho; ++i)
            std::cout << (i ? ", " : "")
                      << std::fixed << std::setprecision(1) << rho_grid[i];
        std::cout << "}\n\n";


        // ---- Section A: CappedSpread(gap) x rho ----
        std::cout << "  [A] CappedSpread(gap) sweep ...\n\n";
        int cell_idx = 0;
        const int total_cells = static_cast<int>(n_rho * n_gap);

        for (std::size_t ir = 0; ir < n_rho; ++ir) {
            const double rho_val = rho_grid[ir];
            for (std::size_t ig = 0; ig < n_gap; ++ig) {
                ++cell_idx;
                const int gap_val = gap_grid[ig];
                auto support = capped_spread_support(
                    cfg.s, cfg.p, gap_val);

                std::ostringstream lbl;
                lbl << "[" << cell_idx << "/" << total_cells
                    << "] rho=" << std::fixed << std::setprecision(1) << rho_val
                    << " gap=" << std::setw(3) << gap_val
                    << " kappa=" << kappa_boundary[ir];

                auto res = run_mc_trials(
                    cfg.num_MC, lbl.str(),
                    [&](unsigned seed) {
                        return dgp_ar1(cfg.n, cfg.p, support,
                                       cfg.amplitude, rho_val, snr_fixed, seed);
                    },
                    cfg.tFDR, trex_ctrl, da_ctrl, base_offset);

                tpp_cs(static_cast<Eigen::Index>(ir),
                       static_cast<Eigen::Index>(ig)) = res.avg_tpr;
                fdp_cs(static_cast<Eigen::Index>(ir),
                       static_cast<Eigen::Index>(ig)) = res.avg_fdr;

                std::cout << "    done. TPP=" << std::fixed << std::setprecision(3)
                          << res.avg_tpr << " FDP=" << res.avg_fdr << "\n\n";
            }
        }


        // ---- Section B: Random support sweep (rho only) ----
        std::cout << "\n  [B] Random support sweep (support redrawn each trial) ...\n\n";

        for (std::size_t ir = 0; ir < n_rho; ++ir) {
            const double rho_val = rho_grid[ir];

            std::ostringstream lbl;
            lbl << "[" << (ir + 1) << "/" << n_rho
                << "] rho=" << std::fixed << std::setprecision(1) << rho_val
                << " kappa=" << kappa_boundary[ir] << " Random";

            auto res = run_mc_trials(
                cfg.num_MC, lbl.str(),
                [&](unsigned seed) {
                    return dgp_ar1(cfg.n, cfg.p,
                                   make_support(SupportPolicy::Random,
                                                cfg.s, cfg.p, 0, seed),
                                   cfg.amplitude, rho_val, snr_fixed, seed);
                },
                cfg.tFDR, trex_ctrl, da_ctrl, base_offset);

            tpp_rand(static_cast<Eigen::Index>(ir)) = res.avg_tpr;
            fdp_rand(static_cast<Eigen::Index>(ir)) = res.avg_fdr;

            std::cout << "    done. TPP=" << std::fixed << std::setprecision(3)
                      << res.avg_tpr << " FDP=" << res.avg_fdr << "\n\n";
        }


        // ---- Print results ----

        // Kappa reference
        std::cout << "\n  DA+AR1 correction window: "
                  << "kappa = ceil(log(0.02) / log(rho))\n";
        std::cout << "  " << std::left << std::setw(8) << "rho";
        for (std::size_t ir = 0; ir < n_rho; ++ir)
            std::cout << std::fixed << std::setprecision(1)
                      << std::setw(6) << rho_grid[ir];
        std::cout << "\n  " << std::left << std::setw(8) << "kappa";
        for (std::size_t ir = 0; ir < n_rho; ++ir)
            std::cout << std::setw(6) << kappa_boundary[ir];
        std::cout << "\n";

        // Combined matrix printer
        auto print_matrix = [&](const std::string& label,
                                const Eigen::MatrixXd& mat_cs,
                                const Eigen::VectorXd& vec_rand) {
            std::cout << "\n  " << label << ":\n";

            // Header row
            std::cout << "  " << std::left << std::setw(12) << "";
            for (std::size_t ig = 0; ig < n_gap; ++ig)
                std::cout << std::right << std::setw(9)
                          << ("gap=" + std::to_string(gap_grid[ig]));
            std::cout << std::setw(10) << "Random" << "\n";
            std::cout << "  " << std::string(12 + 9 * n_gap + 10, '-') << "\n";

            // Data rows
            for (std::size_t ir = 0; ir < n_rho; ++ir) {
                std::ostringstream rl;
                rl << "rho=" << std::fixed << std::setprecision(1) << rho_grid[ir];
                std::cout << "  " << std::left << std::setw(12) << rl.str();
                for (std::size_t ig = 0; ig < n_gap; ++ig)
                    std::cout << std::right << std::fixed << std::setprecision(3)
                              << std::setw(9)
                              << mat_cs(static_cast<Eigen::Index>(ir),
                                        static_cast<Eigen::Index>(ig));
                std::cout << std::fixed << std::setprecision(3)
                          << std::setw(10)
                          << vec_rand(static_cast<Eigen::Index>(ir)) << "\n";
            }
            std::cout << "\n";
        };

        print_matrix("mean_TPP  (cols: CappedSpread gap values + Random)",
                     tpp_cs, tpp_rand);
        print_matrix("mean_FDP  (cols: CappedSpread gap values + Random)",
                     fdp_cs, fdp_rand);
    }

    std::cout << "\nAR(1) demo complete.\n";
    return 0;
}
