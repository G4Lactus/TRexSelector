// ==============================================================================
// demo_trx_02_screen_trex.cpp
// ==============================================================================
/**
 * @file demo_trx_02_screen_trex.cpp
 *
 * @brief Demonstrations of the Screen-TRex Selector.
 *
 * @details
 *  Demo 01 — Basic single-run functionality (ordinary and bootstrap).
 *  Demo 02 — Monte Carlo comparison: Ordinary vs. Bootstrap Screen-TRex (SNR sweep).
 *
 *  In-memory variant. See demo_trx_03_screen_trex_mmap.cpp for the memory-mapped workflow.
 *
 *  All demos use only the modern API:
 *    - trex_selector_methods/trex_screening/trex_screening.hpp
 *    - utils/datageneration/utils_datagen.hpp
 *    - utils/eval_metrics/utils_eval_cdiagnostics.hpp
 *    - utils/eval_metrics/utils_eval_rates.hpp
 */
// ==============================================================================

// std includes
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// TRex includes
#include <trex_selector_methods/trex_screening/trex_screening.hpp>
#include <utils/datageneration/utils_datagen.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>

// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace cdiag     = trex::utils::eval::cdiagnostics;
namespace datagen   = trex::utils::datageneration::datagen;
namespace dummygen  = trex::utils::datageneration::dummygen;
namespace rates     = trex::utils::eval::rates;

// Screen-TRex types
using trex::trex_selector_methods::trex_core::TRexControlParameter;
using trex::trex_selector_methods::trex_core::LLoopStrategy;
using trex::trex_selector_methods::utils::solver_dispatch::SolverTypeForTRex;
using trex::trex_selector_methods::trex_screening::ScreenTRexControlParameter;
using trex::trex_selector_methods::trex_screening::ScreenTRexMethod;
using trex::trex_selector_methods::trex_screening::ScreenTRexSelector;

// ==============================================================================
// Demo 01: Basic Screen-TRex Selector functionality
// ==============================================================================

/**
 * @brief Run a single Screen-TRex selection and print detailed diagnostics.
 *
 * @param high_dim  If true, generate p > n data (n=1000, p=5000).
 * @param rnd_coef  If true, use heterogeneous random coefficients.
 * @param method    Screen-TRex variant (TREX, TREX_DA_AR1, TREX_DA_EQUI).
 * @param bootstrap If true, use bootstrap-CI selection instead of Phi > 0.5.
 */
void demo_ScreenTRexSelector(bool high_dim,
                              bool rnd_coef,
                              ScreenTRexMethod method   = ScreenTRexMethod::TREX,
                              bool             bootstrap = false)
{
    cdiag::print_section_header("Demo: Screen-TRex Selector — Basic Functionality");

    const std::size_t n = high_dim ? 300  : 1000;
    const std::size_t p = high_dim ? 1000 : 300;

    const std::string dim_label =
        high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)";
    std::cout << dim_label << "\n";

    const std::vector<std::size_t> true_support = {27, 149, 398, 4, 42};
    const std::vector<double> true_coefs = rnd_coef
        ? std::vector<double>{-0.4, -0.25, -0.8, 1.1, 2.5}
        : std::vector<double>{ 1.0,   1.0,  1.0, 1.0, 1.0};
    const double snr  = 1.0;
    // Generate synthetic Gaussian data
    std::cout << "Generating synthetic data...\n";
    datagen::SyntheticData data(
        static_cast<Eigen::Index>(n),
        static_cast<Eigen::Index>(p),
        true_support, true_coefs, snr, /*seed=*/58);

    Eigen::Map<Eigen::MatrixXd> X_map(
        data.getX().data(), data.rows(), data.cols());
    Eigen::Map<Eigen::VectorXd> y_map(
        data.getY().data(), data.rows());

    // -----------------------------------------------------------------------
    // TRexControlParameter — Screen-TRex enforces T=1, L=p.
    // Only STANDARD and PERMUTATION lloop_strategy are valid.
    // -----------------------------------------------------------------------
    TRexControlParameter trex_ctrl;
    trex_ctrl.K                       = 20;
    trex_ctrl.max_dummy_multiplier    = 10;   // unused for Screen-TRex (L=p fixed)
    trex_ctrl.use_max_T_stop          = true; // unused for Screen-TRex (T=1 fixed)
    trex_ctrl.dummy_distribution      = dummygen::Distribution::Normal();
    trex_ctrl.lloop_strategy          = LLoopStrategy::STANDARD;
    trex_ctrl.solver_type             = SolverTypeForTRex::TLARS;
    trex_ctrl.parallel_rnd_experiments = false;
    trex_ctrl.use_memory_mapping      = false;

    // Screen-TRex specific parameters
    ScreenTRexControlParameter screen_ctrl;
    screen_ctrl.trex_method      = method;
    screen_ctrl.use_bootstrap_CI = bootstrap;
    screen_ctrl.R_boot           = 1000;
    screen_ctrl.ci_grid_step     = 0.001;
    screen_ctrl.rho_thr_DA       = 0.02;

    // -----------------------------------------------------------------------
    // Construct and run
    // -----------------------------------------------------------------------
    std::cout << "Creating Screen-TRex Selector...\n";
    ScreenTRexSelector sctrex(
        X_map, y_map,
        screen_ctrl,
        trex_ctrl,
        /*seed=*/42,
        /*verbose=*/true
    );

    std::cout << "Running selection...\n";
    sctrex.select();

    // -----------------------------------------------------------------------
    // Display results
    // -----------------------------------------------------------------------
    const auto& screen_result = sctrex.getScreenResult();
    auto selected_indices     = sctrex.getSelectedIndices();

    std::cout << "\n";
    std::cout << "Estimated FDR:      " << screen_result.estimated_FDR << "\n";
    std::cout << "Bootstrap applied:  " << (screen_result.used_bootstrap ? "yes" : "no") << "\n";
    if (screen_result.used_bootstrap) {
        std::cout << "Confidence level:   " << screen_result.confidence_level << "\n";
    }
    if (screen_result.estimated_correlation != 0.0) {
        std::cout << "Estimated ρ (DA):   "
                  << screen_result.estimated_correlation << "\n";
    }

    std::cout << "Selected indices:   ";
    for (const auto idx : selected_indices) { std::cout << idx << " "; }
    std::cout << "\n";

    const double fdp = rates::compute_fdp(selected_indices, true_support);
    const double tpp = rates::compute_tpp(selected_indices, true_support);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "FDP:  " << fdp << "\n";
    std::cout << "TPP:  " << tpp << "\n\n";
}


// ==============================================================================
// Demo 02: Monte Carlo simulation — Ordinary vs. Bootstrap Screen-TRex
// ==============================================================================

/**
 * @brief SNR-sweep Monte Carlo comparison of Screen-TRex variants.
 *
 * @details Compares Ordinary Screen-TRex and Bootstrap-CI Screen-TRex over a
 *          range of SNR values.  Reports FDR, TPR, and estimated FDR averaged
 *          over num_MC Monte Carlo runs.
 *
 * @param num_MC   Number of Monte Carlo runs per (method, SNR) pair.
 * @param high_dim If true, use a high-dimensional setting (n=300, p=1000).
 * @param rnd_coef If true, draw support coefficients from N(0,1) each run.
 */
void demo_ScreenTRex_MonteCarlo(std::size_t num_MC,
                                 bool        high_dim,
                                 bool        rnd_coef)
{
    std::cout.setf(std::ios::unitbuf);
    cdiag::print_section_header("Demo: Screen-TRex Monte Carlo Comparison");

    const std::size_t n = high_dim ? 300  : 1000;
    const std::size_t p = high_dim ? 1000 : 300;
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const std::size_t support_size = 10;
    const std::vector<double> snr_values =
        {0.01, 0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0};

    // -----------------------------------------------------------------------
    // Methods to compare
    // -----------------------------------------------------------------------
    struct MethodInfo {
        std::string name;
        bool        bootstrap;
    };

    const std::vector<MethodInfo> methods = {
        {"Screen-TRex",           false},
        {"Screen-TRex-Bootstrap", true }
    };

    // Storage: method → SNR-indexed vectors
    std::map<std::string, Eigen::VectorXd> est_fdr_map, fdr_map, tpr_map;
    for (const auto& m : methods) {
        est_fdr_map[m.name] = Eigen::VectorXd::Zero(
            static_cast<Eigen::Index>(snr_values.size()));
        fdr_map[m.name]     = Eigen::VectorXd::Zero(
            static_cast<Eigen::Index>(snr_values.size()));
        tpr_map[m.name]     = Eigen::VectorXd::Zero(
            static_cast<Eigen::Index>(snr_values.size()));
    }

    // -----------------------------------------------------------------------
    // TRexControlParameter — fixed for all runs
    // -----------------------------------------------------------------------
    TRexControlParameter trex_ctrl;
    trex_ctrl.K                    = 20;
    trex_ctrl.lloop_strategy       = LLoopStrategy::STANDARD;
    trex_ctrl.solver_type          = SolverTypeForTRex::TLARS;
    trex_ctrl.use_memory_mapping   = false;
    trex_ctrl.dummy_distribution   = dummygen::Distribution::Normal();

    // -----------------------------------------------------------------------
    // RNG for random support + coefficients
    // -----------------------------------------------------------------------
    std::mt19937 rng(24);
    std::uniform_int_distribution<std::size_t> idx_dist(0, p - 1);
    std::normal_distribution<double> coef_dist(0.0, 1.0);

    // -----------------------------------------------------------------------
    // Method loop
    // -----------------------------------------------------------------------
    for (const auto& method : methods) {
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Method: " << method.name << "\n";
        std::cout << std::string(70, '=') << "\n";

        // SNR loop -----------------------------------------------------------
        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
            const double snr = snr_values[snr_idx];
            double total_est_fdr = 0.0;
            double total_fdp     = 0.0;
            double total_tpp     = 0.0;

            std::cout << "SNR = " << std::fixed << std::setprecision(3) << snr << "\n";

            // Monte Carlo loop -----------------------------------------------
            for (std::size_t mc = 0; mc < num_MC; ++mc) {

                // Draw a random support of size support_size
                std::unordered_set<std::size_t> support_set;
                while (support_set.size() < support_size) {
                    support_set.insert(idx_dist(rng));
                }
                const std::vector<std::size_t> true_support(
                    support_set.begin(), support_set.end());

                // Draw coefficients
                std::vector<double> true_coefs;
                true_coefs.reserve(support_size);
                for (std::size_t i = 0; i < support_size; ++i) {
                    true_coefs.push_back(rnd_coef ? coef_dist(rng) : 1.0);
                }

                // Progress indicator
                std::cout << "  Run [" << std::setw(3) << (mc + 1) << "/"
                          << num_MC << "]  Support: ";
                for (const auto idx : true_support) { std::cout << idx << " "; }
                std::cout << "\n";

                // Generate data
                datagen::SyntheticData data(
                    static_cast<Eigen::Index>(n),
                    static_cast<Eigen::Index>(p),
                    true_support, true_coefs, snr,
                    /*seed=*/static_cast<int>(24 + snr_idx * 1000 + mc));

                Eigen::Map<Eigen::MatrixXd> X_map(
                    data.getX().data(), data.rows(), data.cols());
                Eigen::Map<Eigen::VectorXd> y_map(
                    data.getY().data(), data.rows());

                // Screen-TRex parameters
                ScreenTRexControlParameter screen_ctrl;
                screen_ctrl.trex_method      = ScreenTRexMethod::TREX;
                screen_ctrl.use_bootstrap_CI = method.bootstrap;
                screen_ctrl.R_boot           = 1000;
                screen_ctrl.ci_grid_step     = 0.001;

                // Construct and run
                ScreenTRexSelector sctrex(
                    X_map, y_map,
                    screen_ctrl,
                    trex_ctrl,
                    /*seed=*/-1,
                    /*verbose=*/false
                );
                sctrex.select();

                // Accumulate metrics
                const auto selected = sctrex.getSelectedIndices();
                total_est_fdr += sctrex.getScreenResult().estimated_FDR;
                total_fdp     += rates::compute_fdp(selected, true_support);
                total_tpp     += rates::compute_tpp(selected, true_support);

                std::cout << "  Progress: "
                          << std::fixed << std::setprecision(1)
                          << (100.0 * static_cast<double>(mc + 1) / static_cast<double>(num_MC))
                          << "%\r";
            }
            std::cout << std::string(80, ' ') << "\r";
            std::cout << "  Completed: " << num_MC << " runs\n\n";

            const double inv_MC = 1.0 / static_cast<double>(num_MC);
            est_fdr_map[method.name](static_cast<Eigen::Index>(snr_idx)) =
                total_est_fdr * inv_MC;
            fdr_map[method.name](static_cast<Eigen::Index>(snr_idx)) =
                total_fdp * inv_MC;
            tpr_map[method.name](static_cast<Eigen::Index>(snr_idx)) =
                total_tpp * inv_MC;
        }
    }

    // -----------------------------------------------------------------------
    // Print results table
    // -----------------------------------------------------------------------
    std::cout << "\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "=== Screen-TRex Results (averaged over " << num_MC << " runs) ===\n";
    std::cout << std::string(80, '=') << "\n\n";

    const int name_w   = 25;
    const int metric_w = 15;
    const int col_w    = 10;

    // Header row
    std::cout << std::setw(name_w) << "Method"
              << std::setw(metric_w) << "Metric";
    for (const double snr : snr_values) {
        std::cout << std::fixed << std::setprecision(2) << std::setw(col_w) << snr;
    }
    std::cout << "\n";
    std::cout << std::string(name_w + metric_w + col_w * snr_values.size(), '-') << "\n";

    // Data rows
    for (const auto& method : methods) {
        const std::string& nm = method.name;

        auto print_row = [&](const std::string& label, const Eigen::VectorXd& vec) {
            std::cout << std::setw(name_w) << ""
                      << std::setw(metric_w) << label;
            for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(snr_values.size()); ++i) {
                std::cout << std::fixed << std::setprecision(4)
                          << std::setw(col_w) << vec(i);
            }
            std::cout << "\n";
        };

        std::cout << std::setw(name_w) << nm;
        // First data row of each method shares the name column
        std::cout << std::setw(metric_w) << "FDR";
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(snr_values.size()); ++i) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(col_w) << fdr_map[nm](i);
        }
        std::cout << "\n";

        print_row("TPR",            tpr_map[nm]);
        print_row("Est. FDR",       est_fdr_map[nm]);
        std::cout << "\n";
    }
    std::cout << "\n";
}


// ==============================================================================
// main
// ==============================================================================

int main() {
    std::cout.setf(std::ios::unitbuf);

    // -------------------------------------------------------------------------
    // Demo 01: Basic Screen-TRex (ordinary, high-dim, fixed coefs)
    // -------------------------------------------------------------------------
    demo_ScreenTRexSelector(
        /*high_dim=*/true,
        /*rnd_coef=*/false,
        ScreenTRexMethod::TREX,
        /*bootstrap=*/false
    );

    // -------------------------------------------------------------------------
    // Demo 01b: Bootstrap-CI Screen-TRex
    // -------------------------------------------------------------------------
    demo_ScreenTRexSelector(
        /*high_dim=*/true,
        /*rnd_coef=*/false,
        ScreenTRexMethod::TREX,
        /*bootstrap=*/true
    );

    // -------------------------------------------------------------------------
    // Demo 01c: DA-AR1 Screen-TRex
    // -------------------------------------------------------------------------
    // demo_ScreenTRexSelector(
    //     /*high_dim=*/true,
    //     /*rnd_coef=*/false,
    //     ScreenTRexMethod::TREX_DA_AR1,
    //     /*bootstrap=*/false
    // );

    // -------------------------------------------------------------------------
    // Demo 02: Monte Carlo SNR sweep (Ordinary vs. Bootstrap)
    // -------------------------------------------------------------------------
    demo_ScreenTRex_MonteCarlo(
        /*num_MC=*/200,
        /*high_dim=*/true,
        /*rnd_coef=*/true
    );

    return 0;
}
