// ===================================================================================
// trex_biobank_screening.cpp
// ===================================================================================
/**
 * @file trex_biobank_screening.cpp
 *
 * @brief Biobank Screen-TRex implementation (Algorithm 1).
 */
// ===================================================================================

// std includes
#include <utils/logging/logger.hpp>
#include <iomanip>
#include <iostream>
#include <stdexcept>

// Header
#include <trex_selector_methods/trex_screening/trex_biobank_screening.hpp>

// ===================================================================================

namespace trex::trex_selector_methods::trex_biobank_screening {

// Namespace aliases
namespace tc = trex::trex_selector_methods::trex_core;
namespace ts = trex::trex_selector_methods::trex_screening;

// ===================================================================================
// ======================================================================
// Constructors
// ======================================================================

BiobankScreenTRex::BiobankScreenTRex(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::VectorXd>& y,
    BiobankScreenTRexControl biosctrex_ctrl,
    int seed,
    bool verbose
) : X_(&X),
    y_(y),
    Y_(nullptr),
    biosctrex_ctrl_(biosctrex_ctrl),
    seed_(seed),
    verbose_(verbose)
{
    printProgressMessage(
        "Biobank Screen-TRex: Single phenotype mode"
    );
}


BiobankScreenTRex::BiobankScreenTRex(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& Y,
    BiobankScreenTRexControl biosctrex_ctrl,
    int seed,
    bool verbose
) : X_(&X),
    y_(),
    Y_(&Y),
    biosctrex_ctrl_(biosctrex_ctrl),
    seed_(seed),
    verbose_(verbose)
{
    printProgressMessage(
        "Biobank Screen-TRex: Multiple phenotype mode"
    );
}


// ======================================================================
// Main Screening Methods
// ======================================================================

BiobankScreenTRexResult BiobankScreenTRex::screenPhenotype() {
    if (Y_ != nullptr) {
        throw std::runtime_error(
            "BiobankScreenTRex::screenPhenotype() called without Y matrix. "
            "Use screenPhenotypes() instead."
        );
    }

    return screenSinglePhenotype(y_, 0);
}


std::vector<BiobankScreenTRexResult> BiobankScreenTRex::screenPhenotypes() {
    if (Y_ == nullptr) {
        throw std::runtime_error(
            "BiobankScreenTRex::screenPhenotypes() called without Y matrix. "
            "Use screenPhenotype() for single phenotype."
        );
    }

    const std::size_t num_phenotypes = Y_->cols();
    std::vector<BiobankScreenTRexResult> results;
    results.reserve(num_phenotypes);

    if (verbose_) {
        TREX_INFO( "=== Biobank Screen-TRex (Algorithm 1) ===" );
        TREX_INFO( "Phenotypes: " << num_phenotypes );
        TREX_INFO( "FDR range: [" << biosctrex_ctrl_.lower_bound_FDR
                  << ", " << biosctrex_ctrl_.upper_bound_FDR << "]" );
        TREX_INFO( "Fallback target FDR: " << biosctrex_ctrl_.target_FDR_trex << "\n" );
    }

    for (std::size_t pheno_idx = 0; pheno_idx < num_phenotypes; ++pheno_idx) {
        if (verbose_) {
            printProgressMessage(
                "Screening phenotype " + std::to_string(pheno_idx + 1) +
                "/" + std::to_string(num_phenotypes)
            );
        }

        Eigen::VectorXd y_col = Y_->col(static_cast<Eigen::Index>(pheno_idx));
        results.push_back(screenSinglePhenotype(y_col, pheno_idx));

        if (verbose_) {
            const auto& res = results.back();
            TREX_INFO( " Method: " << res.method_used );
            TREX_INFO( " Selected: " << res.selected_indices.size() );
            TREX_INFO( " Est. FDR: " << std::fixed << std::setprecision(4)
                      << res.estimated_FDR );
            TREX_INFO( "  α̂ (ordinary): " << res.estimated_FDR_screen_ordinary );
            TREX_INFO( "  α̂_C (bootstrap): " << res.estimated_FDR_screen_bootstrap << "\n" );
        }
    }

    return results;
}


// ======================================================================
// Algorithm 1
// ======================================================================

BiobankScreenTRexResult BiobankScreenTRex::screenSinglePhenotype(
    const Eigen::VectorXd& y,
    std::size_t phenotype_index
) {
    // Create non-const map for y
    Eigen::Map<Eigen::VectorXd> y_map(
        const_cast<double*>(y.data()),
        static_cast<Eigen::Index>(y.size())
    );

    // =========================================================
    // 2.1: Run Screen-TRex ONCE (ordinary mode) and derive
    //      both α̂ (ordinary) and α̂_C (bootstrap) from the
    //      same K experiments (Algorithm 1, Step 2.1).
    // =========================================================

    if (verbose_) {
        printProgressMessage("  Running Screen-TRex (K experiments)...");
    }

    // Run ordinary Screen-TRex (collects Phi, beta_mat, dummy_betas)
    biosctrex_ctrl_.screen_ctrl.use_bootstrap_CI = false;

    ts::ScreenTRexSelector sctrex(
        *X_,
        y_map,
        biosctrex_ctrl_.screen_ctrl,
        biosctrex_ctrl_.trex_ctrl,
        seed_,
        false
    );

    sctrex.select();
    auto result_sct_ordinary = sctrex.getScreenResult();

    // Derive bootstrap result from the SAME experiment data
    if (verbose_) {
        printProgressMessage("  Applying bootstrap screening to shared data...");
    }

    auto result_sct_bootstrap =
        sctrex.applyBootstrapToOrdinaryResult(result_sct_ordinary);


    // =========================================================
    // 2.2: Apply Algorithm 1 decision logic
    // =========================================================

    return applyDecisionLogic(
        result_sct_bootstrap,
        result_sct_ordinary,
        y,
        phenotype_index
    );
}


BiobankScreenTRexResult BiobankScreenTRex::runFallbackTRexSelector(
    const Eigen::VectorXd& y,
    std::size_t phenotype_index
) {
    // Create non-const map for y (T-Rex)
    Eigen::Map<Eigen::VectorXd> y_map(
        const_cast<double*>(y.data()),
        static_cast<Eigen::Index>(y.size())
    );

    // Run T-Rex selector
    tc::TRexSelector trex_selector(
        *X_,
        y_map,
        biosctrex_ctrl_.target_FDR_trex,
        biosctrex_ctrl_.trex_ctrl,
        seed_,
        false
    );

    // Execute selection
    trex_selector.select();

    // Prepare result structure
    BiobankScreenTRexResult result;
    result.phenotype_index = phenotype_index;
    result.selected_indices = trex_selector.getSelectedIndices();
    result.estimated_FDR = biosctrex_ctrl_.target_FDR_trex;
    result.method_used = "T-Rex (fallback)";
    result.used_fallback_trex = true;
    return result;
}


std::vector<std::size_t> BiobankScreenTRex::selectedVarsToIndices(
    const Eigen::VectorXi& mask
) const {
    std::vector<std::size_t> indices;
    indices.reserve(mask.sum());

    for (int i = 0; i < mask.size(); ++i) {
        if (mask(i) == 1) {
            indices.push_back(static_cast<std::size_t>(i));
        }
    }
    return indices;
}


BiobankScreenTRexResult BiobankScreenTRex::applyDecisionLogic(
    const ts::ScreenTRexSelectionResult& sctrex_result_bootstrap,
    const ts::ScreenTRexSelectionResult& sctrex_result_ordinary,
    const Eigen::VectorXd& y,
    std::size_t phenotype_index
) {
    // 1. Initialize result structure and store diagnostics
    BiobankScreenTRexResult final_result;
    final_result.phenotype_index = phenotype_index;
    final_result.used_fallback_trex = false;

    // FDR bounds from control
    const double alpha_l = biosctrex_ctrl_.lower_bound_FDR;
    const double alpha_u = biosctrex_ctrl_.upper_bound_FDR;

    // Alpha estimates (Diagnostic storage)
    const double alpha_hat = sctrex_result_ordinary.estimated_FDR;
    const double alpha_hat_C = sctrex_result_bootstrap.estimated_FDR;

    final_result.estimated_FDR_screen_ordinary = alpha_hat;
    final_result.estimated_FDR_screen_bootstrap = alpha_hat_C;

    // Store selected indices from both methods
    final_result.selected_indices_screen_ordinary =
        selectedVarsToIndices(sctrex_result_ordinary.selected_var);
    final_result.selected_indices_screen_bootstrap =
        selectedVarsToIndices(sctrex_result_bootstrap.selected_var);


    // ==================================================
    // Algorithm 1: Step 2.2 Decision Logic
    // ==================================================
    auto alpha_in_bounds = [&](double alpha_val) {
        return (alpha_l <= alpha_val) && (alpha_val <= alpha_u);
    };


    // ---------------------------------------------------
    // Case 1: Bootstrap-CI Screen-TRex
    // Condition: α_l ≤ α̂_C ≤ α_u AND
    //            max{α̂_C, α̂·I(α̂ ≤ α_u)} = α̂_C
    // ---------------------------------------------------
    if (alpha_in_bounds(alpha_hat_C)) {

        // Compute max{α̂_C, α̂·I(α̂ ≤ α_u)}
        const double alpha_hat_weighted = (alpha_hat <= alpha_u) ? alpha_hat : 0.0;
        const double max_val_bootstrap = std::max(alpha_hat_C, alpha_hat_weighted);

        if (std::abs(max_val_bootstrap - alpha_hat_C) <= 1e-10) {
            final_result.selected_indices = final_result.selected_indices_screen_bootstrap;
            final_result.estimated_FDR = alpha_hat_C;
            final_result.method_used = "Screen-TRex (bootstrap-CI)";
            return final_result;
        }
    }

    // ---------------------------------------------------
    // Case 2: Ordinary Screen-TRex
    // Condition: α_l ≤ α̂ ≤ α_u AND
    //            max{α̂_C·I(α̂_C ≤ α_u), α̂} = α̂
    // ---------------------------------------------------
    if (alpha_in_bounds(alpha_hat)) {

        // Compute max{α̂_C·I(α̂_C ≤ α_u), α̂}
        const double alpha_hat_C_weighted = (alpha_hat_C <= alpha_u) ? alpha_hat_C : 0.0;
        const double max_val_ordinary = std::max(alpha_hat, alpha_hat_C_weighted);

        if (std::abs(max_val_ordinary - alpha_hat) <= 1e-10) {
            final_result.selected_indices = final_result.selected_indices_screen_ordinary;
            final_result.estimated_FDR = alpha_hat;
            final_result.method_used = "Screen-TRex (ordinary)";
            return final_result;
        }
    }


    // ---------------------------------------------------
    // Case 3: Neither condition met -> Fallback to T-Rex
    // ---------------------------------------------------
    if (verbose_) {
        TREX_INFO( "  [Info] Screen-TRex estimates (Ord: " << alpha_hat
                  << ", Boot: " << alpha_hat_C << ") outside bounds ["
                  << alpha_l << ", " << alpha_u << "]. Running Fallback T-Rex..." );
    }

    // Run fallback
    BiobankScreenTRexResult trex_res = runFallbackTRexSelector(y, phenotype_index);

    // Merge results into existing object to preserve diagnostics
    final_result.selected_indices = trex_res.selected_indices;
    final_result.estimated_FDR = trex_res.estimated_FDR;
    final_result.method_used = trex_res.method_used;
    final_result.used_fallback_trex = true;

    return final_result;
}


// ======================================================================
// Statistics
// ======================================================================

BiobankScreenTRex::BiobankScreenTRexStatistics BiobankScreenTRex::getBiobankScreenTRexResult(
    const std::vector<BiobankScreenTRexResult>& results
) const {
    BiobankScreenTRexStatistics stats;
    stats.num_phenotypes = results.size();

    if (results.empty()) {
        stats.avg_estimated_FDR = 0.0;
        stats.avg_selected_count = 0.0;
        return stats;
    }

    double sum_fdr = 0.0;
    double sum_selected = 0.0;

    for (const auto& res : results) {
        sum_fdr += res.estimated_FDR;
        sum_selected += static_cast<double>(res.selected_indices.size());

        // Count which method was used
        if (res.used_fallback_trex) {
            ++stats.num_used_trex;
        } else if (res.method_used.find("bootstrap") != std::string::npos) {
            ++stats.num_used_screen_bootstrap;
        } else {
            ++stats.num_used_screen_ordinary;
        }

    }

    stats.avg_estimated_FDR = sum_fdr / static_cast<double>(results.size());
    stats.avg_selected_count = sum_selected / static_cast<double>(results.size());

    return stats;
}


// ======================================================================
// Utilities
// ======================================================================

void BiobankScreenTRex::printProgressMessage(const std::string& msg) const {
    if (verbose_) {
        TREX_INFO( msg << "\n");
    }
}

// ===================================================================================
} /* End of namespace trex::trex_selector_methods::trex_biobank_screening */
