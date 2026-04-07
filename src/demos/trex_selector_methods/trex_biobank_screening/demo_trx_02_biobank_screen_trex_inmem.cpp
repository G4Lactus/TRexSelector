// ==============================================================================
// demo_trx_04_biobank_screen_trex_inmem.cpp
// ==============================================================================
/**
 * @file demo_trx_04_biobank_screen_trex_inmem.cpp
 *
 * @brief Demonstrations of the Biobank Screen-TRex Selector (Algorithm 1).
 *
 * @details
 *  In-memory variant (use_memory_mapping = false).
 *  See demo_trx_05_biobank_screen_trex_mmap.cpp for the memory-mapped workflow.
 *
 *  Demo 01 — Single phenotype basic functionality.
 *  Demo 02 — Multiple phenotype screening (shared design matrix).
 *  Demo 03 — Monte Carlo SNR sweep with per-method statistics.
 */
// ==============================================================================

// std includes
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// TRex includes
#include <trex_selector_methods/trex_screening/trex_biobank_screening.hpp>
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
namespace tbs      = trex::trex_selector_methods::trex_biobank_screening;
namespace ts       = trex::trex_selector_methods::trex_screening;
namespace tc       = trex::trex_selector_methods::trex_core;

using trex::trex_selector_methods::utils::solver_dispatch::SolverTypeForTRex;

// ==============================================================================
// Shared helper: build a default BiobankScreenTRexControl (in-memory)
// ==============================================================================

static tbs::BiobankScreenTRexControl makeDefaultControl(
    int K = 20, double R_boot = 1000, double ci_grid_step = 0.001)
{
    tbs::BiobankScreenTRexControl ctrl;
    ctrl.lower_bound_FDR = 0.05;
    ctrl.upper_bound_FDR = 0.15;
    ctrl.target_FDR_trex = 0.10;

    ctrl.screen_ctrl.trex_method  = ts::ScreenTRexMethod::TREX;
    ctrl.screen_ctrl.R_boot       = static_cast<std::size_t>(R_boot);
    ctrl.screen_ctrl.ci_grid_step = ci_grid_step;

    ctrl.trex_ctrl.K                    = K;
    ctrl.trex_ctrl.max_dummy_multiplier = 10;
    ctrl.trex_ctrl.use_max_T_stop       = true;
    ctrl.trex_ctrl.dummy_distribution   = dummygen::Distribution::Normal();
    ctrl.trex_ctrl.lloop_strategy       = tc::LLoopStrategy::STANDARD;
    ctrl.trex_ctrl.solver_type          = SolverTypeForTRex::TLARS;
    ctrl.trex_ctrl.use_memory_mapping   = false;    // <-- in-memory D storage

    return ctrl;
}

// ==============================================================================
// Demo 01: Single phenotype
// ==============================================================================

void demo_BiobankScreenTrex_SinglePhenotype(bool rnd_coef = false)
{
    cdiag::print_section_header("Demo: Biobank Screen-TRex — Single Phenotype (In-Memory)");

    const std::size_t n = 300;
    const std::size_t p = 1000;
    const std::vector<std::size_t> true_support = {27, 149, 398, 4, 42};
    const std::vector<double> true_coefs = rnd_coef
        ? std::vector<double>{-0.4, -0.25, -0.8, 1.1, 2.5}
        : std::vector<double>{ 1.0,   1.0,  1.0, 1.0, 1.0};
    const double snr = 1.0;

    std::cout << "Generating synthetic data...\n";
    datagen::SyntheticData data(
        static_cast<Eigen::Index>(n),
        static_cast<Eigen::Index>(p),
        true_support, true_coefs, snr, /*seed=*/123);

    Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), data.rows(), data.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

    tbs::BiobankScreenTRexControl ctrl = makeDefaultControl();

    std::cout << "Running Algorithm 1...\n";
    tbs::BiobankScreenTRex biosctrex(X_map, y_map, ctrl, /*seed=*/42, /*verbose=*/true);
    auto result = biosctrex.screenPhenotype();

    std::cout << "\nMethod used:     " << result.method_used << "\n";
    std::cout << "Estimated FDR:   " << std::fixed << std::setprecision(4)
              << result.estimated_FDR << "\n";
    std::cout << "  α̂  (ordinary):  " << result.estimated_FDR_screen_ordinary << "\n";
    std::cout << "  α̂_C (bootstrap): " << result.estimated_FDR_screen_bootstrap << "\n";
    std::cout << "Selected (" << result.selected_indices.size() << "):  ";
    for (const auto idx : result.selected_indices) { std::cout << idx << " "; }
    std::cout << "\n";
    std::cout << "FDP: " << rates::compute_fdp(result.selected_indices, true_support) << "\n";
    std::cout << "TPP: " << rates::compute_tpp(result.selected_indices, true_support) << "\n\n";
}

// ==============================================================================
// Demo 02: Multiple phenotype screening
// ==============================================================================

void demo_BiobankScreenTrex_MultiplePhenotypes(bool rnd_coef = false)
{
    cdiag::print_section_header(
        "Demo: Biobank Screen-TRex — Multiple Phenotypes (In-Memory)");

    const std::size_t n = 300;
    const std::size_t p = 1000;
    const std::size_t q = 5;

    std::mt19937 rng(4212);
    std::normal_distribution<double> normal(0.0, 1.0);

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n); ++i)
        for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(p); ++j)
            X(i, j) = normal(rng);

    Eigen::MatrixXd Y(n, q);
    std::vector<std::vector<std::size_t>> true_supports(q);
    const std::vector<double> snr_values = {0.1, 0.5, 1.0, 2.0, 5.0};
    std::uniform_int_distribution<std::size_t> col_dist(0, p - 1);

    for (std::size_t k = 0; k < q; ++k) {
        const std::size_t sup_size = 5 + (k % 3);
        std::unordered_set<std::size_t> sup_set;
        while (sup_set.size() < sup_size) { sup_set.insert(col_dist(rng)); }
        true_supports[k] = std::vector<std::size_t>(sup_set.begin(), sup_set.end());

        std::vector<double> coefs(sup_size);
        for (auto& c : coefs) { c = rnd_coef ? normal(rng) : 1.0; }

        datagen::SyntheticData col_data(
            static_cast<Eigen::Index>(n),
            static_cast<Eigen::Index>(p),
            true_supports[k], coefs, snr_values[k],
            /*seed=*/100 + static_cast<int>(k));

        Y.col(static_cast<Eigen::Index>(k)) = col_data.getY();

        std::cout << "Phenotype " << k
                  << ": support=" << sup_size
                  << " SNR=" << std::fixed << std::setprecision(2) << snr_values[k] << "\n";
    }
    std::cout << "\n";

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> Y_map(Y.data(), Y.rows(), Y.cols());

    tbs::BiobankScreenTRexControl ctrl = makeDefaultControl(/*K=*/20, /*R_boot=*/500,
                                                             /*ci_grid_step=*/0.005);

    std::cout << "Running Algorithm 1 for all phenotypes...\n\n";
    tbs::BiobankScreenTRex biosctrex(X_map, Y_map, ctrl, /*seed=*/42, /*verbose=*/true);
    auto results = biosctrex.screenPhenotypes();
    auto stats   = biosctrex.getBiobankScreenTRexResult(results);

    const int cw = 12, nw = 22;
    std::cout << "\n" << std::string(90, '=') << "\n";
    std::cout << "=== Results by Phenotype ===\n";
    std::cout << std::string(90, '=') << "\n\n";
    std::cout << std::setw(8) << "Pheno" << std::setw(nw) << "Method"
              << std::setw(cw) << "Est.FDR" << std::setw(cw) << "Selected"
              << std::setw(cw) << "FDP" << std::setw(cw) << "TPP" << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::cout << std::setw(8) << i
                  << std::setw(nw) << r.method_used
                  << std::fixed << std::setprecision(4)
                  << std::setw(cw) << r.estimated_FDR
                  << std::setw(cw) << r.selected_indices.size()
                  << std::setw(cw) << rates::compute_fdp(r.selected_indices, true_supports[i])
                  << std::setw(cw) << rates::compute_tpp(r.selected_indices, true_supports[i])
                  << "\n";
    }

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "=== Summary ===\n" << std::string(70, '=') << "\n";
    std::cout << "Total phenotypes:        " << stats.num_phenotypes << "\n";
    std::cout << "Screen-TRex (ordinary):  " << stats.num_used_screen_ordinary << "\n";
    std::cout << "Screen-TRex (bootstrap): " << stats.num_used_screen_bootstrap << "\n";
    std::cout << "T-Rex (fallback):        " << stats.num_used_trex << "\n";
    std::cout << "Avg estimated FDR:       " << std::fixed << std::setprecision(4)
              << stats.avg_estimated_FDR << "\n";
    std::cout << "Avg selected count:      " << std::fixed << std::setprecision(2)
              << stats.avg_selected_count << "\n\n";
}

// ==============================================================================
// Demo 03: Monte Carlo SNR sweep
// ==============================================================================

void demo_BiobankScreenTrex_MonteCarlo(std::size_t num_MC, bool rnd_coef)
{
    cdiag::print_section_header("Demo: Biobank Screen-TRex — Monte Carlo (In-Memory)");

    const std::size_t n = 300, p = 1000, support_size = 10;
    const std::vector<double> snr_values = {0.1, 0.5, 1.0, 2.0, 5.0};

    const std::vector<std::string> method_names = {
        "Screen-TRex (ordinary)",
        "Screen-TRex (bootstrap-CI)",
        "T-Rex (fallback)"
    };

    const Eigen::Index ns = static_cast<Eigen::Index>(snr_values.size());
    std::map<std::string, Eigen::VectorXd> fdp_res, tpp_res, est_fdr, usage;
    for (const auto& nm : method_names) {
        fdp_res[nm] = Eigen::VectorXd::Zero(ns);
        tpp_res[nm] = Eigen::VectorXd::Zero(ns);
        est_fdr[nm] = Eigen::VectorXd::Zero(ns);
        usage[nm]   = Eigen::VectorXd::Zero(ns);
    }

    std::mt19937 rng(24);
    std::uniform_int_distribution<std::size_t> idx_dist(0, p - 1);
    std::normal_distribution<double> coef_dist(0.0, 1.0);

    for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
        const double snr = snr_values[snr_idx];
        std::cout << "SNR = " << std::fixed << std::setprecision(2) << snr << "\n";

        for (std::size_t mc = 0; mc < num_MC; ++mc) {
            std::unordered_set<std::size_t> sup_set;
            while (sup_set.size() < support_size) { sup_set.insert(idx_dist(rng)); }
            const std::vector<std::size_t> true_support(sup_set.begin(), sup_set.end());

            std::vector<double> true_coefs(support_size);
            for (auto& c : true_coefs) { c = rnd_coef ? coef_dist(rng) : 1.0; }

            datagen::SyntheticData data(
                static_cast<Eigen::Index>(n),
                static_cast<Eigen::Index>(p),
                true_support, true_coefs, snr,
                /*seed=*/static_cast<int>(1000 * snr_idx + mc));

            Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), data.rows(), data.cols());
            Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

            tbs::BiobankScreenTRexControl ctrl = makeDefaultControl();
            tbs::BiobankScreenTRex biosctrex(X_map, y_map, ctrl, /*seed=*/-1, /*verbose=*/false);
            auto result = biosctrex.screenPhenotype();

            const Eigen::Index si = static_cast<Eigen::Index>(snr_idx);
            fdp_res[result.method_used](si) += rates::compute_fdp(result.selected_indices, true_support);
            tpp_res[result.method_used](si) += rates::compute_tpp(result.selected_indices, true_support);
            usage[result.method_used](si)   += 1.0;
            est_fdr["Screen-TRex (ordinary)"](si)     += result.estimated_FDR_screen_ordinary;
            est_fdr["Screen-TRex (bootstrap-CI)"](si) += result.estimated_FDR_screen_bootstrap;
            est_fdr["T-Rex (fallback)"](si)           += ctrl.target_FDR_trex;

            std::cout << "  Progress: " << std::fixed << std::setprecision(1)
                      << (100.0 * static_cast<double>(mc + 1) / static_cast<double>(num_MC))
                      << "%\r" << std::flush;
        }
        std::cout << std::string(60, ' ') << "\r" << "  Completed " << num_MC << " runs\n\n";
    }

    for (const auto& nm : method_names) {
        for (Eigen::Index i = 0; i < ns; ++i) {
            if (usage[nm](i) > 0) {
                fdp_res[nm](i) /= usage[nm](i);
                tpp_res[nm](i) /= usage[nm](i);
            }
            est_fdr[nm](i) /= static_cast<double>(num_MC);
        }
    }

    const int nw = 30, mw = 14, cw = 12;
    std::cout << "\n" << std::string(90, '=') << "\n";
    std::cout << "=== Biobank Screen-TRex Results (avg over " << num_MC << " runs) ===\n";
    std::cout << std::string(90, '=') << "\n\n";
    std::cout << std::setw(nw) << "Method" << std::setw(mw) << "Metric";
    for (const double snr : snr_values)
        std::cout << std::fixed << std::setprecision(2) << std::setw(cw) << snr;
    std::cout << "\n" << std::string(nw + mw + cw * static_cast<int>(snr_values.size()), '-') << "\n";

    for (const auto& nm : method_names) {
        auto print_row = [&](const std::string& label, const Eigen::VectorXd& v) {
            std::cout << std::setw(nw) << "" << std::setw(mw) << label;
            for (Eigen::Index i = 0; i < ns; ++i)
                std::cout << std::fixed << std::setprecision(4) << std::setw(cw) << v(i);
            std::cout << "\n";
        };

        std::cout << std::setw(nw) << nm << std::setw(mw) << "Usage (%)";
        for (Eigen::Index i = 0; i < ns; ++i)
            std::cout << std::fixed << std::setprecision(1) << std::setw(cw)
                      << (100.0 * usage[nm](i) / static_cast<double>(num_MC));
        std::cout << "\n";
        print_row("FDR",      fdp_res[nm]);
        print_row("TPR",      tpp_res[nm]);
        print_row("Est. FDR", est_fdr[nm]);
        std::cout << "\n";
    }
}

// ==============================================================================
// main
// ==============================================================================

int main()
{
    std::cout.setf(std::ios::unitbuf);

    // Demo 01: single phenotype
    demo_BiobankScreenTrex_SinglePhenotype(/*rnd_coef=*/false);

    // Demo 02: multiple phenotypes
    // demo_BiobankScreenTrex_MultiplePhenotypes(/*rnd_coef=*/false);

    // Demo 03: Monte Carlo SNR sweep
    // demo_BiobankScreenTrex_MonteCarlo(/*num_MC=*/200, /*rnd_coef=*/true);

    return 0;
}
