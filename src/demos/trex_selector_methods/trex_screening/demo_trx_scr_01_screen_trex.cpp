// ==============================================================================
// demo_trx_scr_01_screen_trex.cpp
// ==============================================================================
/**
 * @file demo_trx_scr_01_screen_trex.cpp
 *
 * @brief Demonstrations of the Screen-TRex Selector.
 *
 * @details
 *  Demo 01 — Basic single-run functionality (ordinary and bootstrap).
 *  Demo 02 — Monte Carlo comparison: Ordinary vs. Bootstrap Screen-TRex (SNR sweep).
 *
 *  In-memory variant. See demo_trx_scr_02_screen_trex_mmap.cpp for the
 *  memory-mapped workflow.
 */
// ==============================================================================

// std includes
#include <iostream>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// TRex includes
#include <utils/datageneration/utils_datagen.hpp>

// Demo utilities
#include "demo_trx_scr_common.hpp"

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

    const auto methods = default_methods();

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

    auto trex_ctrl = make_trex_control();

    // RNG for random support + coefficients
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

                // Generate data
                datagen::SyntheticData data(
                    static_cast<Eigen::Index>(n),
                    static_cast<Eigen::Index>(p),
                    true_support,
                    true_coefs, snr,
                    static_cast<int>(24 + snr_idx * 1000 + mc)
                );

                Eigen::Map<Eigen::MatrixXd> X_map(
                    data.getX().data(), data.rows(), data.cols());
                Eigen::Map<Eigen::VectorXd> y_map(
                    data.getY().data(), data.rows());

                auto screen_ctrl =
                    make_screen_control(method.method,
                                        method.bootstrap);

                ScreenTRexSelector sctrex(
                    X_map,
                    y_map,
                    screen_ctrl,
                    trex_ctrl,
                    -1,
                    false);
                sctrex.select();

                const auto selected = sctrex.getSelectedIndices();
                total_est_fdr += sctrex.getScreenResult().estimated_FDR;
                total_fdp     += rates::compute_fdp(selected, true_support);
                total_tpp     += rates::compute_tpp(selected, true_support);

                print_mc_progress(snr, mc, num_MC);
            }
            print_mc_done(snr, num_MC);

            const double inv_MC = 1.0 / static_cast<double>(num_MC);
            est_fdr_map[method.name](static_cast<Eigen::Index>(snr_idx)) =
                total_est_fdr * inv_MC;
            fdr_map[method.name](static_cast<Eigen::Index>(snr_idx)) =
                total_fdp * inv_MC;
            tpr_map[method.name](static_cast<Eigen::Index>(snr_idx)) =
                total_tpp * inv_MC;
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
