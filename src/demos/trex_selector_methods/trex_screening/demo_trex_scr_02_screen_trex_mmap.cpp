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

            const auto idx = static_cast<Eigen::Index>(snr_idx);
            est_fdr_map[method.name](idx) = result.avg_est_fdr;
            fdr_map    [method.name](idx) = result.avg_fdr;
            tpr_map    [method.name](idx) = result.avg_tpr;
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
    omp_set_num_threads(6);

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
