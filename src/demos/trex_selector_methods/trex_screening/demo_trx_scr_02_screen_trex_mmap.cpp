// ==============================================================================
// demo_trx_scr_02_screen_trex_mmap.cpp
// ==============================================================================
/**
 * @file demo_trx_scr_02_screen_trex_mmap.cpp
 *
 * @brief Screen-TRex Selector — memory-mapped D-file variant.
 *
 * @details
 *  Repeats the single-run demo from demo_trx_scr_01_screen_trex.cpp with
 *  TRexControlParameter::use_memory_mapping = true.
 *
 *  Demo 01a — Screen-TRex Ordinary, high-dim, memory-mapped D files.
 *  Demo 01b — Screen-TRex Bootstrap, high-dim, memory-mapped D files.
 *  Demo 02  — Monte Carlo comparison: Ordinary vs. Bootstrap (memory-mapped).
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
// Demo 01a: Screen-TRex Ordinary — memory-mapped
// ==============================================================================

void demo_ScreenTRex_MMap_Ordinary(bool high_dim)
{
    cdiag::print_section_header(
        "Demo: Screen-TRex Ordinary — Memory-Mapped D Storage");

    const std::size_t n = high_dim ? 300  : 1000;
    const std::size_t p = high_dim ? 1000 : 300;
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const std::vector<std::size_t> true_support = {27, 149, 398, 4, 42};
    const std::vector<double>      true_coefs   = {1.0, 1.0, 1.0, 1.0, 1.0};

    datagen::SyntheticData data(
        static_cast<Eigen::Index>(n),
        static_cast<Eigen::Index>(p),
        true_support, true_coefs, /*snr=*/5.0, /*seed=*/58);

    Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(),
                                      data.rows(),
                                      data.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

    auto trex_ctrl   = make_trex_control_mmap();
    auto screen_ctrl = make_screen_control(
        ScreenTRexMethod::TREX, /*bootstrap=*/false);

    std::cout << "Running Screen-TRex Ordinary (use_memory_mapping=true)...\n";
    ScreenTRexSelector sctrex(X_map,
                              y_map,
                              screen_ctrl,
                              trex_ctrl,
                              /*seed=*/42,
                              /*verbose=*/true);
    sctrex.select();

    print_screen_result(sctrex.getScreenResult(),
                        sctrex.getSelectedIndices(), true_support);
}

// ==============================================================================
// Demo 01b: Screen-TRex Bootstrap — memory-mapped
// ==============================================================================

void demo_ScreenTRex_MMap_Bootstrap(bool high_dim)
{
    cdiag::print_section_header(
        "Demo: Screen-TRex Bootstrap — Memory-Mapped D Storage");

    const std::size_t n = high_dim ? 300  : 1000;
    const std::size_t p = high_dim ? 1000 : 300;
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const std::vector<std::size_t> true_support = {27, 149, 398, 4, 42};
    const std::vector<double>      true_coefs   = {1.0, 1.0, 1.0, 1.0, 1.0};

    datagen::SyntheticData data(
        static_cast<Eigen::Index>(n),
        static_cast<Eigen::Index>(p),
        true_support, true_coefs, /*snr=*/5.0, /*seed=*/58);

    Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(),
                                      data.rows(),
                                      data.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

    auto trex_ctrl   = make_trex_control_mmap();
    auto screen_ctrl = make_screen_control(
        ScreenTRexMethod::TREX, /*bootstrap=*/true);

    std::cout << "Running Screen-TRex Bootstrap (use_memory_mapping=true)...\n";
    ScreenTRexSelector sctrex(X_map,
                              y_map,
                              screen_ctrl,
                              trex_ctrl,
                              /*seed=*/42,
                              /*verbose=*/true);
    sctrex.select();

    print_screen_result(sctrex.getScreenResult(),
                        sctrex.getSelectedIndices(), true_support);
}


// ==============================================================================
// Demo 02: Monte Carlo simulation — memory-mapped
// ==============================================================================

/**
 * @brief SNR-sweep Monte Carlo comparison of Screen-TRex variants (mmap).
 *
 * @details Mirrors demo_trx_scr_01's MC simulation but with
 *          use_memory_mapping = true.  Compares Ordinary Screen-TRex and
 *          Bootstrap-CI Screen-TRex over a range of SNR values.
 *
 * @param num_MC   Number of Monte Carlo runs per (method, SNR) pair.
 * @param high_dim If true, use a high-dimensional setting (n=300, p=1000).
 * @param rnd_coef If true, draw support coefficients from N(0,1) each run.
 */
void demo_ScreenTRex_MMap_MonteCarlo(std::size_t num_MC,
                                      bool        high_dim,
                                      bool        rnd_coef)
{
    std::cout.setf(std::ios::unitbuf);
    cdiag::print_section_header(
        "Demo: Screen-TRex Monte Carlo — Memory-Mapped");

    const std::size_t n = high_dim ? 300  : 1000;
    const std::size_t p = high_dim ? 1000 : 300;
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)")
              << "  |  use_memory_mapping = true\n";

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

    auto trex_ctrl = make_trex_control_mmap();

    // RNG for random support + coefficients
    std::mt19937 rng(24);
    std::uniform_int_distribution<std::size_t> idx_dist(0, p - 1);
    std::normal_distribution<double> coef_dist(0.0, 1.0);

    // -------------------------------------------------------------------
    // Method loop
    // -------------------------------------------------------------------
    for (const auto& method : methods) {
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Method: " << method.name << "\n";
        std::cout << std::string(70, '=') << "\n";

        // SNR loop -------------------------------------------------------
        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
            const double snr = snr_values[snr_idx];
            double total_est_fdr = 0.0;
            double total_fdp     = 0.0;
            double total_tpp     = 0.0;

            // Monte Carlo loop -------------------------------------------
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
                    true_support, true_coefs, snr,
                    static_cast<int>(24 + snr_idx * 1000 + mc));

                Eigen::Map<Eigen::MatrixXd> X_map(
                    data.getX().data(), data.rows(), data.cols());
                Eigen::Map<Eigen::VectorXd> y_map(
                    data.getY().data(), data.rows());

                auto screen_ctrl =
                    make_screen_control(method.method, method.bootstrap);

                ScreenTRexSelector sctrex(
                    X_map, y_map, screen_ctrl, trex_ctrl,
                    -1, false);
                sctrex.select();

                const auto selected = sctrex.getSelectedIndices();
                total_est_fdr += sctrex.getScreenResult().estimated_FDR;
                total_fdp     += rates::compute_fdp(selected, true_support);
                total_tpp     += rates::compute_tpp(selected, true_support);

                print_mc_progress(snr, mc, num_MC);
            }
            print_mc_done(snr, num_MC);

            const double inv_MC = 1.0 / static_cast<double>(num_MC);
            const auto idx = static_cast<Eigen::Index>(snr_idx);
            est_fdr_map[method.name](idx) = total_est_fdr * inv_MC;
            fdr_map[method.name](idx)     = total_fdp * inv_MC;
            tpr_map[method.name](idx)     = total_tpp * inv_MC;
        }
    }

    const std::string file_stem =
        "d02_screen_trex_mmap_mc_n" + std::to_string(n)
        + "_p" + std::to_string(p);
    save_and_print_mc_results(num_MC, file_stem, snr_values, methods,
                              fdr_map, tpr_map, est_fdr_map);
}


// ==============================================================================
// Main
// ==============================================================================

int main()
{
    std::cout.setf(std::ios::unitbuf);

    // Demo 01a: Screen-TRex Ordinary — memory-mapped
    demo_ScreenTRex_MMap_Ordinary(/*high_dim=*/true);

    // Demo 01b: Screen-TRex Bootstrap — memory-mapped
    demo_ScreenTRex_MMap_Bootstrap(/*high_dim=*/true);

    // Demo 02: Monte Carlo SNR sweep (Ordinary vs. Bootstrap, mmap)
    demo_ScreenTRex_MMap_MonteCarlo(
        500,
        true,
        false
    );

    return 0;
}
