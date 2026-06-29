// ===================================================================================
// utils_cdiagnostics.hpp
// ===================================================================================
#ifndef UTILS_EVAL_CDIAGNOSTICS_HPP
#define UTILS_EVAL_CDIAGNOSTICS_HPP
// ===================================================================================
/**
 * @file utils_cdiagnostics.hpp
 *
 * @brief Console diagnostics utilities for console print outs and special print outs
 *        for evaluating path-based solver performance.
 *
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <vector>

// utils includes
#include "utils/eval_metrics/utils_eval_counts.hpp"
#include "utils/eval_metrics/utils_eval_rates.hpp"
#include "utils/eval_metrics/utils_eval_composites.hpp"


// ===================================================================================

// Embedded into namespace trex::utils::eval::cdiagnostics
namespace trex::utils::eval::cdiagnostics {

// ===================================================================================
// Set up namespace aliases
// ===================================================================================
namespace eval_counts = trex::utils::eval::counts;
namespace eval_rates = trex::utils::eval::rates;
namespace eval_composites = trex::utils::eval::composites;
namespace eval_cdiagnostics = trex::utils::eval::cdiagnostics;

// ===========================================================
// Utility Functions
// ===========================================================

// Print demo title on console
// ===========================================================
/**
 * @brief Get the mutable pointer to the default output stream.
 *
 * @return Reference to the static pointer to the current default output stream.
 */
inline std::ostream*& get_default_stream_ptr() {
    static std::ostream *default_stream = &std::cout;
    return default_stream;
}


/**
 * @brief Get the default stream object
 *
 * @return std::ostream& Reference to the default output stream
 */
inline std::ostream &get_default_stream() {
    return *get_default_stream_ptr();
}


/**
 * @brief Set the default stream object
 *
 * @param stream Reference to the output stream to set as default
 */
inline void set_default_stream(std::ostream &stream) {
    get_default_stream_ptr() = &stream;
}


/**
 * @brief Print section header
 *
 * @param title String with algorithm or section name
 * @param os Output stream (default: std::cout)
 */
inline void print_section_header(const std::string &title,
                                 std::ostream &os = get_default_stream()) {
    os << "\n" << std::string(60, '=') << "\n";
    os << title << "\n";
    os << std::string(60, '=') << "\n";
}
// ===========================================================


// Print vector
// ==========================================================
/**
 * @brief Print vector in comma-separated format: {1, 2, 3}
 *
 * @tparam T Element type of the vector.
 *
 * @param vec Vector to print.
 * @param max_show Maximum number of elements to show (default: all).
 */
template <typename T>
inline void print_vector(
    const std::vector<T> &vec,
    std::size_t max_show = SIZE_MAX
) {

    std::size_t n_show = std::min(max_show, vec.size());

    std::cout << "{";
    for (std::size_t i = 0; i < n_show; ++i) {
        std::cout << vec[i];
        if (i < n_show - 1) {
            std::cout << ", ";
        }
    }
    if (n_show < vec.size()) {
        std::cout << ", ...";
    }
    std::cout << "}";
}


/**
 * @brief Print a compact actions log.
 *
 * @details Each step prints a line with all additions (+k) or drops (-k) performed.
 *
 * @param actions Vector of per-step action lists (positive = add, negative = drop).
 */
void print_actions_log(const std::vector<std::vector<int>> &actions)
{
    for (std::size_t step = 0; step < actions.size(); ++step)
    {
        std::cout << "Step " << std::setw(3) << step << ": ";
        for (int idx : actions[step]) {
            if (idx >= 0)
                std::cout << " +" << idx;
            else
                std::cout << " " << idx; // Negative is drop
        }
        std::cout << "\n";
    }
}
// ==========================================================



// Print demo data configuration
// ===========================================================

/**
 * @brief Print demo data configuration to test terminating path algorithm.
 *
 * @param n Number of observations.
 * @param p Number of predictors.
 * @param num_dummies Number of dummy variables.
 * @param T_stop Stopping parameter.
 * @param support Indices of true support.
 * @param coefs Values of true coefficients.
 * @param snr Signal-to-noise ratio (linear).
 */
inline void print_talgo_demo_config(
    std::size_t n,
    std::size_t p,
    std::size_t num_dummies,
    std::size_t T_stop,
    const std::vector<std::size_t>& support,
    const std::vector<double>& coefs,
    double snr
) {
    std::cout << "Configuration:\n";
    std::cout << "  n = " << n << ", p = " << p << "\n";
    std::cout << "  num_dummies = " << num_dummies << "\n";
    std::cout << "  T_stop = " << T_stop << "\n";
    std::cout << "  True support size = " << support.size() << "\n";
    std::cout << "  True support: ";
    for (std::size_t idx : support) {
        std::cout << idx << " ";
    }
    std::cout << "\n";
    std::cout << "  True coefficients: ";
    for (double coef : coefs) {
        std::cout << coef << " ";
    }
    std::cout << "\n";
    std::cout << "  SNR = " << snr << "\n\n";
}
// ==========================================================



// Print selection diagnostics
// ===========================================================

/**
 * @brief Print selected variables and dummies from fitted solver's solution path
 *        with color coding.
 *
 * @tparam SolverType Concrete solver type exposing getActivePredictorIndices(),
 *                    getActiveDummyIndices(), getActions(), and getDummyStartIndex().
 *
 * @param solver Fitted solver instance.
 * @param true_support Indices of true non-zero coefficients.
 */
template<typename SolverType>
inline void print_selection(
    const SolverType& solver,
    const std::vector<std::size_t>& true_support
) {
    std::cout << "\nSelection:\n";

    // 1. Predictors in active set
    std::cout << "  Predictors: ";
    print_vector(solver.getActivePredictorIndices());

    // 2. Dummies in active set
    std::cout << "\n  Dummies: ";
    print_vector(solver.getActiveDummyIndices());

    // 3. Full selection path (with dummies colored)
    auto actions = solver.getActions();
    std::size_t dummy_start = solver.getDummyStartIndex();
    std::cout << "\n  Path: [";
    bool first = true;
    for (std::size_t step = 0; step < actions.size(); ++step) {
        for (int action : actions[step]) {
            if (!first) std::cout << ", ";
            first = false;

            if (action < 0) {
                // REMOVAL EVENT
                std::size_t removed_idx = static_cast<std::size_t>(-action);
                if (removed_idx >= dummy_start) {
                    // Dummy removal: magenta
                    std::cout << "\033[35m-D" << (removed_idx - dummy_start)
                              << "\033[0m";
                } else {
                    // Variable removal: yellow
                    std::cout << "\033[33m-v" << removed_idx << "\033[0m";
                }
            } else {
                // ADDITION EVENT
                std::size_t idx = static_cast<std::size_t>(action);
                bool is_tp = std::find(true_support.begin(),
                                       true_support.end(), idx)
                                       != true_support.end();
                if (idx >= dummy_start) {
                    // Dummy addition: blue
                    std::cout << "\033[34m+D" << (idx - dummy_start)
                              << "\033[0m";
                } else if (is_tp) {
                    // True positive variable addition: green
                    std::cout << "\033[32m+v" << idx << "\033[0m";
                } else {
                    // False positive variable addition: red
                    std::cout << "\033[31m+v" << idx << "\033[0m";
                }
            }
        }
    }
    std::cout << "]\n";
}

// ==========================================================



// Print selection quality
// ==========================================================

/**
 * @brief Prints the quality metrics of the variable selection.
 *
 * @tparam TSolver Concrete solver type exposing getActivePredictorIndices().
 *
 * @param solver The solver instance after fitting.
 * @param true_support The indices of the true support variables.
 */
template <typename TSolver>
void print_selection_quality(
    const TSolver& solver,
    const std::vector<std::size_t>& true_support
) {
    // Get selected indices from TSolver
    auto selected = solver.getActivePredictorIndices();

    // Compute counts and rates
    std::size_t true_positives = eval_counts::true_positives_count(
        selected,
        true_support
    );
    std::size_t false_positives = selected.size() - true_positives;
    std::size_t false_negatives = true_support.size() - true_positives;

    double fdp = eval_rates::compute_fdp(
        selected, true_support
    );
    double tpp = eval_rates::compute_tpp(
        selected, true_support
    );

    double precision = eval_rates::compute_precision(
        selected, true_support
    );
    double recall = eval_rates::compute_recall(
        selected, true_support
    );
    double f1_score = eval_composites::f1_score(
        precision, recall
    );

    // Print report
    std::cout << "\nSelection Quality:\n";
    std::cout << "  True Positives:  " << true_positives << "\n";
    std::cout << "  False Positives: " << false_positives << "\n";
    std::cout << "  False Negatives: " << false_negatives << "\n";
    std::cout << "  FDP:             " << fdp << "\n";
    std::cout << "  TPP:             " << tpp << "\n";
    std::cout << "  Precision:       " << precision << "\n";
    std::cout << "  Recall:          " << recall << "\n";
    std::cout << "  F1 Score:        " << f1_score << "\n";
}


// ===================================================================================

} /* End of namespace trex::utils::eval::cdiagnostics */

#endif /* UTILS_EVAL_CDIAGNOSTICS_HPP */
