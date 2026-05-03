// ==============================================================================
// demo_trex_scr_01_screen_trex.cpp
// ==============================================================================
/**
 * @file demo_trex_scr_01_screen_trex.cpp
 *
 * @brief Demonstrations of the Screen-TRex Selector.
 *
 * @details
 *  Demo 01 — Basic single-run functionality (ordinary and bootstrap).
 *  Demo 02 — Monte Carlo comparison: Ordinary vs. Bootstrap Screen-TRex (SNR sweep).
 *            Simulations are parallelised via OpenMP (run_mc_trials_screen_trex).
 *
 *  In-memory variant. See demo_trex_scr_02_screen_trex_mmap.cpp for the
 *  memory-mapped workflow.
 */
// ==============================================================================

// std includes
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// TRex includes
#include <utils/datageneration/utils_datagen.hpp>

// Demo utilities
#include "demo_trex_scr_common.hpp"

// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace datagen = trex::utils::datageneration::datagen;
using namespace scr_demo;

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
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const std::vector<std::size_t> true_support = {27, 149, 398, 4, 42};
    const std::vector<double> true_coefs = rnd_coef
        ? std::vector<double>{-0.4, -0.25, -0.8, 1.1, 2.5}
        : std::vector<double>{ 1.0,   1.0,  1.0, 1.0, 1.0};
    const double snr = 5.0;

    std::cout << "Generating synthetic data...\n";
    datagen::SyntheticData data(
        static_cast<Eigen::Index>(n),
        static_cast<Eigen::Index>(p),
        true_support, true_coefs, snr, /*seed=*/58);

    Eigen::Map<Eigen::MatrixXd> X_map(
        data.getX().data(), data.rows(), data.cols());
    Eigen::Map<Eigen::VectorXd> y_map(
        data.getY().data(), data.rows());

    auto trex_ctrl   = make_trex_control();
    auto screen_ctrl = make_screen_control(method, bootstrap);

    std::cout << "Running selection...\n";
    ScreenTRexSelector sctrex(X_map,
                              y_map,
                              screen_ctrl,
                              trex_ctrl,
                              42,
                              true);
    sctrex.select();

    print_screen_result(sctrex.getScreenResult(),
                        sctrex.getSelectedIndices(), true_support);
}


// ==============================================================================
// Demo 02: Monte Carlo simulation — Ordinary vs. Bootstrap Screen-TRex
// ==============================================================================

/**
 * @brief SNR-sweep Monte Carlo comparison of Screen-TRex variants.
 *
 * @details Compares Ordinary Screen-TRex and Bootstrap-CI Screen-TRex over a
 *          range of SNR values. Reports FDR, TPR, and estimated FDR averaged
 *          over num_MC Monte Carlo runs. Parallelised via OpenMP.
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

    const auto methods = default_methods();

    // Storage: method → SNR-indexed vectors
    std::map<std::string, Eigen::VectorXd> est_fdr_map, fdr_map, tpr_map;
    for (const auto& m : methods) {
        const auto len = static_cast<Eigen::Index>(snr_values.size());
        est_fdr_map[m.name] = Eigen::VectorXd::Zero(len);
        fdr_map[m.name]     = Eigen::VectorXd::Zero(len);
        tpr_map[m.name]     = Eigen::VectorXd::Zero(len);
    }

    auto trex_ctrl = make_trex_control();

    // -----------------------------------------------------------------------
    // Method loop
    // -----------------------------------------------------------------------
    for (const auto& method : methods) {
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Method: " << method.name << "\n";
        std::cout << std::string(70, '=') << "\n";

        auto screen_ctrl = make_screen_control(method.method, method.bootstrap);

        // SNR loop -----------------------------------------------------------
        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
            const double snr = snr_values[snr_idx];

            const auto make_data = [=](unsigned seed) -> ScrDGPData {
                // Isolate support-selection RNG to offset 500000 from trial seed
                std::mt19937 rng_sup(seed + 500000u);
                std::uniform_int_distribution<std::size_t> idx_dist(0, p - 1);
                std::unordered_set<std::size_t> support_set;
                while (support_set.size() < support_size)
                    support_set.insert(idx_dist(rng_sup));
                std::vector<std::size_t> true_support(
                    support_set.begin(), support_set.end());

                // Isolate coefficient RNG to offset 600000 from trial seed
                std::vector<double> true_coefs;
                true_coefs.reserve(support_size);
                if (rnd_coef) {
                    std::mt19937 rng_coef(seed + 600000u);
                    std::normal_distribution<double> coef_dist(0.0, 1.0);
                    for (std::size_t i = 0; i < support_size; ++i)
                        true_coefs.push_back(coef_dist(rng_coef));
                } else {
                    true_coefs.assign(support_size, 1.0);
                }

                datagen::SyntheticData data(
                    static_cast<Eigen::Index>(n),
                    static_cast<Eigen::Index>(p),
                    true_support, true_coefs, snr, static_cast<int>(seed));
                return {data.getX(), data.getY(), true_support};
            };

            const unsigned base_seed = 24u + static_cast<unsigned>(snr_idx) * 1000u;

            std::ostringstream lbl;
            lbl << method.name << "  SNR=" << std::fixed << std::setprecision(2) << snr;
            const auto result = run_mc_trials_screen_trex(
                num_MC, lbl.str(), make_data, trex_ctrl, screen_ctrl, base_seed);

            const auto ei = static_cast<Eigen::Index>(snr_idx);
            est_fdr_map[method.name](ei) = result.avg_est_fdr;
            fdr_map    [method.name](ei) = result.avg_fdr;
            tpr_map    [method.name](ei) = result.avg_tpr;
        }
    }

    const std::string file_stem =
        "d01_screen_trex_mc_n" + std::to_string(n)
        + "_p" + std::to_string(p);
    save_and_print_mc_results(num_MC, file_stem, snr_values, methods,
                              fdr_map, tpr_map, est_fdr_map);
}


// ==============================================================================
// main
// ==============================================================================

int main() {
    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(6);

    // -------------------------------------------------------------------------
    // Demo 01: Basic Screen-TRex (ordinary, high-dim, fixed coefs)
    // -------------------------------------------------------------------------
    demo_ScreenTRexSelector(
        true,
        false,
        ScreenTRexMethod::TREX,
        false
    );

    // -------------------------------------------------------------------------
    // Demo 01b: Bootstrap-CI Screen-TRex
    // -------------------------------------------------------------------------
    demo_ScreenTRexSelector(
        true,
        false,
        ScreenTRexMethod::TREX,
        true
    );

    // -------------------------------------------------------------------------
    // Demo 02: Monte Carlo SNR sweep (Ordinary vs. Bootstrap)
    // -------------------------------------------------------------------------
    demo_ScreenTRex_MonteCarlo(
        500,
        true,
        false
    );

    return 0;
}
