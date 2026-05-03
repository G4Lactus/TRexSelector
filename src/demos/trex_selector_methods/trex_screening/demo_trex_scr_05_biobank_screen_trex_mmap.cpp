// ==============================================================================
// demo_trx_scr_05_biobank_screen_trex_mmap.cpp
// ==============================================================================
/**
 * @file demo_trx_scr_05_biobank_screen_trex_mmap.cpp
 *
 * @brief Biobank Screen-TRex Selector (Algorithm 1) — memory-mapped D-file variant.
 *
 * @details Dummy matrices are stored on disk via Boost memory-mapped files
 *          (use_memory_mapping = true). This workflow is designed for biobank-scale
 *          problems where in-memory storage of D matrices is infeasible.
 *          See demo_trx_scr_04_biobank_screen_trex_inmem.cpp for the in-memory
 *          reference workflow.
 *
 *  Demo 01 — Single phenotype: basic Algorithm 1 functionality (memory-mapped).
 *  Demo 02 — Multiple phenotypes: shared design matrix, per-phenotype response.
 *  Demo 03 — Monte Carlo SNR sweep (single phenotype): per-method Usage %, FDR, TPR, Est. FDR.
 *  Demo 04 — Monte Carlo SNR sweep (multiple phenotypes): same per-method metrics.
 */
// ==============================================================================

// std includes
#include <random>
#include <unordered_set>

// TRex includes
#include <utils/datageneration/utils_datagen.hpp>

// Biobank Screen-TRex demo infrastructure
#include "demo_trex_scr_bb_common.hpp"

// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace datagen = trex::utils::datageneration::datagen;
using namespace scr_demo;


// ==============================================================================
// Demo 01 — Single phenotype (Memory-Mapped)
// ==============================================================================

void demo_MMap_Biobank_SinglePhenotype()
{
    cdiag::print_section_header(
        "Demo 01: Biobank Screen-TRex — Single Phenotype (Memory-Mapped)");

    const std::size_t n = 300, p = 1000;
    const std::vector<std::size_t> true_support = {4, 27, 42, 149, 398};
    const std::vector<double> true_coefs         = {1.0, 1.0, 1.0, 1.0, 1.0};
    const double snr = 1.0;

    std::cout << "DGP: n=" << n << ", p=" << p
              << ", p1=" << true_support.size() << ", SNR=" << snr << "\n\n";

    datagen::SyntheticData data(
        static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(p),
        true_support, true_coefs, snr, /*seed=*/123);

    Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(),
                                      data.rows(),
                                      data.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

    auto ctrl = makeBiobankControl(/*use_mmap=*/true);

    tbs::BiobankScreenTRex biosctrex(X_map,
                                     y_map,
                                     ctrl,
                                     /*seed=*/42,
                                     /*verbose=*/true);
    const auto result = biosctrex.screenPhenotype();

    print_biobank_single_result(result, true_support);
}


// ==============================================================================
// Demo 02 — Multiple phenotypes (Memory-Mapped)
// ==============================================================================

void demo_MMap_Biobank_MultiplePhenotypes()
{
    cdiag::print_section_header(
        "Demo 02: Biobank Screen-TRex — Multiple Phenotypes (Memory-Mapped)");

    const std::size_t n = 300, p = 1000, q = 5;
    const int         x_seed    = 4212;   // fixed → all phenotypes share the same X
    const std::vector<double>      snr_values = {1.0, 2.0, 5.0, 10.0, 20.0};
    const std::vector<std::size_t> sup_sizes  = {5, 6, 5, 7, 5};

    std::mt19937 rng(777);
    std::uniform_int_distribution<std::size_t> col_dist(0, p - 1);

    // ── Generate all phenotype data with the same X_seed ──────────────────
    // SyntheticData(n, p, num_dummies, support, coefs, snr, base_seed, X_seed, dummy_seed)
    // Fixing X_seed ensures every SyntheticData produces the exact same X.
    // getX() from phenotype 0 is used for the shared Eigen::Map.
    std::vector<std::vector<std::size_t>> true_supports(q);
    std::vector<datagen::SyntheticData>   pheno_data;
    pheno_data.reserve(q);

    for (std::size_t k = 0; k < q; ++k) {
        const std::size_t sup_size = sup_sizes[k];
        std::unordered_set<std::size_t> sup_set;
        while (sup_set.size() < sup_size) { sup_set.insert(col_dist(rng)); }
        true_supports[k] = std::vector<std::size_t>(sup_set.begin(), sup_set.end());

        pheno_data.emplace_back(
            static_cast<Eigen::Index>(n),
            static_cast<Eigen::Index>(p),
            /*num_dummies=*/0,
            true_supports[k],
            std::vector<double>(sup_size, 1.0),
            snr_values[k],
            /*seed=*/100 + static_cast<int>(k),
            /*X_seed=*/x_seed,
            /*dummy_seed=*/-1);

        std::cout << "Phenotype " << k << ": p1=" << sup_size
                  << ", SNR=" << std::fixed << std::setprecision(2) << snr_values[k] << "\n";
    }
    std::cout << "\n";

    // X is identical for all phenotypes (same X_seed) — use pointer from phenotype 0
    Eigen::Map<Eigen::MatrixXd> X_map(
        pheno_data[0].getX().data(), pheno_data[0].rows(), pheno_data[0].cols());

    // Stack Y columns
    Eigen::MatrixXd Y(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(q));
    for (std::size_t k = 0; k < q; ++k)
        Y.col(static_cast<Eigen::Index>(k)) = pheno_data[k].getY();
    Eigen::Map<Eigen::MatrixXd> Y_map(Y.data(), Y.rows(), Y.cols());

    auto ctrl = makeBiobankControl(/*use_mmap=*/true, /*K=*/20);
    ctrl.screen_ctrl.R_boot       = 500;
    ctrl.screen_ctrl.ci_grid_step = 0.005;

    std::cout << "Running Algorithm 1 for " << q << " phenotypes...\n\n";
    tbs::BiobankScreenTRex biosctrex(X_map,
                                     Y_map,
                                     ctrl,
                                     /*seed=*/42,
                                     /*verbose=*/true);
    const auto results = biosctrex.screenPhenotypes();
    const auto stats   = biosctrex.getBiobankScreenTRexResult(results);

    print_biobank_phenotype_table(results, true_supports);
    print_biobank_summary(stats);
}


// ==============================================================================
// Demo 03 — Monte Carlo SNR sweep (Memory-Mapped)
// ==============================================================================

void demo_MMap_Biobank_MonteCarlo(std::size_t num_MC)
{
    std::cout.setf(std::ios::unitbuf);
    cdiag::print_section_header(
        "Demo 03: Biobank Screen-TRex — MC SNR Sweep (Memory-Mapped)");

    const std::size_t n            = 300;
    const std::size_t p            = 1000;
    const std::size_t support_size = 10;
    const std::vector<double> snr_values = {0.1, 0.5, 1.0, 2.0, 5.0};
    const std::vector<std::string> method_names = {
        "Screen-TRex (ordinary)",
        "Screen-TRex (bootstrap-CI)",
        "T-Rex (fallback)"
    };
    const Eigen::Index ns = static_cast<Eigen::Index>(snr_values.size());

    // Accumulators: fdp/tpp are conditional on method being used
    std::map<std::string, Eigen::VectorXd> fdp_map, tpp_map, est_fdr_map, usage_map;
    for (const auto& nm : method_names) {
        fdp_map[nm]     = Eigen::VectorXd::Zero(ns);
        tpp_map[nm]     = Eigen::VectorXd::Zero(ns);
        est_fdr_map[nm] = Eigen::VectorXd::Zero(ns);
        usage_map[nm]   = Eigen::VectorXd::Zero(ns);
    }

    std::mt19937 rng(24);
    std::uniform_int_distribution<std::size_t> idx_dist(0, p - 1);

    for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
        const double       snr = snr_values[snr_idx];
        const Eigen::Index si  = static_cast<Eigen::Index>(snr_idx);

        for (std::size_t mc = 0; mc < num_MC; ++mc) {
            // Random sparse support
            std::unordered_set<std::size_t> sup_set;
            while (sup_set.size() < support_size) { sup_set.insert(idx_dist(rng)); }
            const std::vector<std::size_t> true_support(sup_set.begin(), sup_set.end());
            const std::vector<double>      true_coefs(support_size, 1.0);

            datagen::SyntheticData data(
                static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(p),
                true_support, true_coefs, snr,
                /*seed=*/static_cast<int>(1000 * snr_idx + mc));

            Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(),
                                              data.rows(),
                                              data.cols());
            Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

            auto ctrl = makeBiobankControl(/*use_mmap=*/true);
            tbs::BiobankScreenTRex biosctrex(X_map,
                                             y_map,
                                             ctrl,
                                             /*seed=*/-1,
                                             /*verbose=*/false);
            const auto result = biosctrex.screenPhenotype();

            // Accumulate FDP/TPP only for the method that was actually used
            fdp_map[result.method_used](si)  +=
                rates::compute_fdp(result.selected_indices, true_support);
            tpp_map[result.method_used](si)  +=
                rates::compute_tpp(result.selected_indices, true_support);
            usage_map[result.method_used](si) += 1.0;

            // Estimated FDR values accumulated unconditionally for all methods
            est_fdr_map["Screen-TRex (ordinary)"](si)     += result.estimated_FDR_screen_ordinary;
            est_fdr_map["Screen-TRex (bootstrap-CI)"](si) += result.estimated_FDR_screen_bootstrap;
            est_fdr_map["T-Rex (fallback)"](si)           += ctrl.target_FDR_trex;

            print_mc_progress(snr, mc, num_MC);
        }
        print_mc_done(snr, num_MC);
    }

    // ── Normalize ──────────────────────────────────────────────────────────
    for (const auto& nm : method_names) {
        for (Eigen::Index i = 0; i < ns; ++i) {
            if (usage_map[nm](i) > 0.0) {
                fdp_map[nm](i) /= usage_map[nm](i);   // conditional mean FDP
                tpp_map[nm](i) /= usage_map[nm](i);   // conditional mean TPR
            }
            est_fdr_map[nm](i) /= static_cast<double>(num_MC);   // unconditional mean
            usage_map[nm](i)   /= static_cast<double>(num_MC);   // fraction in [0, 1]
        }
    }

    // ── Build file stem and save results ───────────────────────────────────
    std::ostringstream stem;
    stem << "d05_biobank_screen_trex_mc_snr_n" << n << "_p" << p
         << "_s" << support_size << "_mmap";

    save_and_print_biobank_mc_results(
        num_MC, stem.str(), snr_values, method_names,
        fdp_map, tpp_map, est_fdr_map, usage_map);
}


// ==============================================================================
// Demo 04 — MC SNR sweep — multiple phenotypes (Memory-Mapped)
// ==============================================================================

void demo_MMap_Biobank_MultiplePhenotypesMC(std::size_t num_MC)
{
    std::cout.setf(std::ios::unitbuf);
    cdiag::print_section_header(
        "Demo 04: Biobank Screen-TRex — MC SNR Sweep, Multiple Phenotypes (Memory-Mapped)");

    const std::size_t n            = 300;
    const std::size_t p            = 1000;
    const std::size_t q            = 5;    // phenotypes per MC run
    const std::size_t support_size = 5;
    const std::vector<double> snr_values = {0.1, 0.5, 1.0, 2.0, 5.0};
    const std::vector<std::string> method_names = {
        "Screen-TRex (ordinary)",
        "Screen-TRex (bootstrap-CI)",
        "T-Rex (fallback)"
    };
    const Eigen::Index ns = static_cast<Eigen::Index>(snr_values.size());

    // Accumulators over (q × num_MC) total phenotype screenings per SNR level
    std::map<std::string, Eigen::VectorXd> fdp_map, tpp_map, est_fdr_map, usage_map;
    for (const auto& nm : method_names) {
        fdp_map[nm]     = Eigen::VectorXd::Zero(ns);
        tpp_map[nm]     = Eigen::VectorXd::Zero(ns);
        est_fdr_map[nm] = Eigen::VectorXd::Zero(ns);
        usage_map[nm]   = Eigen::VectorXd::Zero(ns);
    }

    std::mt19937 rng(77);
    std::uniform_int_distribution<std::size_t> idx_dist(0, p - 1);
    const std::vector<double> coefs(support_size, 1.0);

    for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
        const double       snr = snr_values[snr_idx];
        const Eigen::Index si  = static_cast<Eigen::Index>(snr_idx);

        for (std::size_t mc = 0; mc < num_MC; ++mc) {
            // All phenotypes in this run share the same X via a fixed X_seed
            const int x_seed = static_cast<int>(5000 * snr_idx + mc + 1);

            // Draw random sparse supports for all q phenotypes
            std::vector<std::vector<std::size_t>> true_supports(q);
            for (std::size_t k = 0; k < q; ++k) {
                std::unordered_set<std::size_t> sup_set;
                while (sup_set.size() < support_size) { sup_set.insert(idx_dist(rng)); }
                true_supports[k] = std::vector<std::size_t>(sup_set.begin(), sup_set.end());
            }

            // Generate all SyntheticData with the same X_seed → shared design matrix
            std::vector<datagen::SyntheticData> pheno_data;
            pheno_data.reserve(q);
            for (std::size_t k = 0; k < q; ++k) {
                pheno_data.emplace_back(
                    static_cast<Eigen::Index>(n),
                    static_cast<Eigen::Index>(p),
                    /*num_dummies=*/0,
                    true_supports[k], coefs, snr,
                    /*seed=*/static_cast<int>(q * (num_MC * snr_idx + mc) + k + 1),
                    /*X_seed=*/x_seed,
                    /*dummy_seed=*/-1);
            }

            // X is identical for all phenotypes — taken from phenotype 0
            Eigen::Map<Eigen::MatrixXd> X_map(
                pheno_data[0].getX().data(),
                pheno_data[0].rows(),
                pheno_data[0].cols());

            // Stack Y columns
            Eigen::MatrixXd Y(static_cast<Eigen::Index>(n),
                              static_cast<Eigen::Index>(q));
            for (std::size_t k = 0; k < q; ++k)
                Y.col(static_cast<Eigen::Index>(k)) = pheno_data[k].getY();
            Eigen::Map<Eigen::MatrixXd> Y_map(Y.data(), Y.rows(), Y.cols());

            auto ctrl = makeBiobankControl(/*use_mmap=*/true);
            tbs::BiobankScreenTRex biosctrex(X_map,
                                             Y_map,
                                             ctrl,
                                             /*seed=*/-1,
                                             /*verbose=*/false);
            const auto results = biosctrex.screenPhenotypes();

            // Accumulate per phenotype
            for (std::size_t k = 0; k < q; ++k) {
                const auto& r = results[k];
                fdp_map[r.method_used](si)  += rates::compute_fdp(r.selected_indices,
                                                                   true_supports[k]);
                tpp_map[r.method_used](si)  += rates::compute_tpp(r.selected_indices,
                                                                   true_supports[k]);
                usage_map[r.method_used](si) += 1.0;
                est_fdr_map["Screen-TRex (ordinary)"](si)     += r.estimated_FDR_screen_ordinary;
                est_fdr_map["Screen-TRex (bootstrap-CI)"](si) += r.estimated_FDR_screen_bootstrap;
                est_fdr_map["T-Rex (fallback)"](si)           += ctrl.target_FDR_trex;
            }

            print_mc_progress(snr, mc, num_MC);
        }
        print_mc_done(snr, num_MC);
    }

    // ── Normalize over (q × num_MC) total phenotype screenings ────────────
    const double total = static_cast<double>(q) * static_cast<double>(num_MC);
    for (const auto& nm : method_names) {
        for (Eigen::Index i = 0; i < ns; ++i) {
            if (usage_map[nm](i) > 0.0) {
                fdp_map[nm](i) /= usage_map[nm](i);   // conditional mean FDP
                tpp_map[nm](i) /= usage_map[nm](i);   // conditional mean TPR
            }
            est_fdr_map[nm](i) /= total;   // unconditional mean
            usage_map[nm](i)   /= total;   // fraction in [0, 1]
        }
    }

    // ── Build file stem and save results ───────────────────────────────────
    std::ostringstream stem;
    stem << "d05_biobank_screen_trex_mc_multi_n" << n << "_p" << p
         << "_q" << q << "_s" << support_size << "_mmap";

    save_and_print_biobank_mc_results(
        num_MC, stem.str(), snr_values, method_names,
        fdp_map, tpp_map, est_fdr_map, usage_map);
}


// ==============================================================================
// main
// ==============================================================================

int main()
{
    std::cout.setf(std::ios::unitbuf);

    // Demo 01: Single phenotype — basic Algorithm 1 functionality (memory-mapped)
    demo_MMap_Biobank_SinglePhenotype();

    // Demo 02: Multiple phenotype screening with shared design matrix
    demo_MMap_Biobank_MultiplePhenotypes();

    // Demo 03: Monte Carlo SNR sweep — single phenotype (runs for a while)
    demo_MMap_Biobank_MonteCarlo(/*num_MC=*/500);

    // Demo 04: Monte Carlo SNR sweep — multiple phenotypes (runs for a while)
    // demo_MMap_Biobank_MultiplePhenotypesMC(/*num_MC=*/100);

    return 0;
}
// ==============================================================================
