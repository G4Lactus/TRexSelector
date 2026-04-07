// ==============================================================================
// demo_trx_03_screen_trex_mmap.cpp
// ==============================================================================
/**
 * @file demo_trx_03_screen_trex_mmap.cpp
 *
 * @brief Screen-TRex Selector — memory-mapped D-file variant.
 *
 * @details
 *  Repeats the single-run demo from demo_trx_02_screen_trex.cpp with
 *  TRexControlParameter::use_memory_mapping = true.
 *
 *  The algorithm and results are identical to the in-memory path; only the
 *  dummy-matrix storage backend changes (Boost memory-mapped files instead of
 *  heap-allocated Eigen matrices).
 *
 *  Demo 01a — Ordinary Screen-TRex, high-dim, memory-mapped D files.
 *  Demo 01b — Bootstrap-CI Screen-TRex, high-dim, memory-mapped D files.
 */
// ==============================================================================

// std includes
#include <iomanip>
#include <iostream>
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

namespace cdiag    = trex::utils::eval::cdiagnostics;
namespace datagen  = trex::utils::datageneration::datagen;
namespace dummygen = trex::utils::datageneration::dummygen;
namespace rates    = trex::utils::eval::rates;

using trex::trex_selector_methods::trex_core::TRexControlParameter;
using trex::trex_selector_methods::trex_core::LLoopStrategy;
using trex::trex_selector_methods::utils::solver_dispatch::SolverTypeForTRex;
using trex::trex_selector_methods::trex_screening::ScreenTRexControlParameter;
using trex::trex_selector_methods::trex_screening::ScreenTRexMethod;
using trex::trex_selector_methods::trex_screening::ScreenTRexSelector;

// ==============================================================================
// Shared setup
// ==============================================================================

/**
 * @brief Build a TRexControlParameter with use_memory_mapping=true.
 */
static TRexControlParameter makeMMapControl()
{
    TRexControlParameter ctrl;
    ctrl.K                     = 20;
    ctrl.lloop_strategy        = LLoopStrategy::STANDARD;
    ctrl.solver_type           = SolverTypeForTRex::TLARS;
    ctrl.dummy_distribution    = dummygen::Distribution::Normal();
    ctrl.use_memory_mapping    = true;    // <-- memory-mapped D files
    return ctrl;
}

// ==============================================================================
// Demo 01: Screen-TRex (ordinary) — memory-mapped
// ==============================================================================

void demo_ScreenTRexSelector_MMap_Ordinary(bool high_dim)
{
    cdiag::print_section_header(
        "Demo: Screen-TRex (Ordinary) — Memory-Mapped D Storage");

    const std::size_t n = high_dim ? 300  : 1000;
    const std::size_t p = high_dim ? 1000 : 300;
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const std::vector<std::size_t> true_support = {27, 149, 398, 4, 42};
    const std::vector<double>      true_coefs   = {1.0, 1.0, 1.0, 1.0, 1.0};
    const double snr = 1.0;

    datagen::SyntheticData data(
        static_cast<Eigen::Index>(n),
        static_cast<Eigen::Index>(p),
        true_support, true_coefs, snr, /*seed=*/58);

    Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), data.rows(), data.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

    TRexControlParameter trex_ctrl = makeMMapControl();

    ScreenTRexControlParameter screen_ctrl;
    screen_ctrl.trex_method      = ScreenTRexMethod::TREX;
    screen_ctrl.use_bootstrap_CI = false;

    std::cout << "Running ordinary Screen-TRex (use_memory_mapping=true)...\n";
    ScreenTRexSelector sctrex(X_map, y_map, screen_ctrl, trex_ctrl, /*seed=*/42, /*verbose=*/true);
    sctrex.select();

    const auto& result = sctrex.getScreenResult();
    const auto  sel    = sctrex.getSelectedIndices();

    std::cout << "\nEstimated FDR:  " << result.estimated_FDR << "\n";
    std::cout << "Selected:       ";
    for (const auto idx : sel) { std::cout << idx << " "; }
    std::cout << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "FDP: " << rates::compute_fdp(sel, true_support) << "\n";
    std::cout << "TPP: " << rates::compute_tpp(sel, true_support) << "\n\n";
}

// ==============================================================================
// Demo 01b: Screen-TRex (bootstrap-CI) — memory-mapped
// ==============================================================================

void demo_ScreenTRexSelector_MMap_Bootstrap(bool high_dim)
{
    cdiag::print_section_header(
        "Demo: Screen-TRex (Bootstrap-CI) — Memory-Mapped D Storage");

    const std::size_t n = high_dim ? 300  : 1000;
    const std::size_t p = high_dim ? 1000 : 300;
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const std::vector<std::size_t> true_support = {27, 149, 398, 4, 42};
    const std::vector<double>      true_coefs   = {1.0, 1.0, 1.0, 1.0, 1.0};
    const double snr = 1.0;

    datagen::SyntheticData data(
        static_cast<Eigen::Index>(n),
        static_cast<Eigen::Index>(p),
        true_support, true_coefs, snr, /*seed=*/58);

    Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), data.rows(), data.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

    TRexControlParameter trex_ctrl = makeMMapControl();

    ScreenTRexControlParameter screen_ctrl;
    screen_ctrl.trex_method      = ScreenTRexMethod::TREX;
    screen_ctrl.use_bootstrap_CI = true;
    screen_ctrl.R_boot           = 1000;
    screen_ctrl.ci_grid_step     = 0.001;

    std::cout << "Running bootstrap-CI Screen-TRex (use_memory_mapping=true)...\n";
    ScreenTRexSelector sctrex(X_map, y_map, screen_ctrl, trex_ctrl, /*seed=*/42, /*verbose=*/true);
    sctrex.select();

    const auto& result = sctrex.getScreenResult();
    const auto  sel    = sctrex.getSelectedIndices();

    std::cout << "\nEstimated FDR:    " << result.estimated_FDR << "\n";
    std::cout << "Bootstrap applied: yes\n";
    std::cout << "Confidence level:  " << result.confidence_level << "\n";
    std::cout << "Selected:         ";
    for (const auto idx : sel) { std::cout << idx << " "; }
    std::cout << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "FDP: " << rates::compute_fdp(sel, true_support) << "\n";
    std::cout << "TPP: " << rates::compute_tpp(sel, true_support) << "\n\n";
}

// ==============================================================================
// main
// ==============================================================================

int main()
{
    std::cout.setf(std::ios::unitbuf);

    // Demo 01a: Ordinary Screen-TRex — memory-mapped
    demo_ScreenTRexSelector_MMap_Ordinary(/*high_dim=*/true);

    // Demo 01b: Bootstrap-CI Screen-TRex — memory-mapped
    demo_ScreenTRexSelector_MMap_Bootstrap(/*high_dim=*/true);

    return 0;
}
