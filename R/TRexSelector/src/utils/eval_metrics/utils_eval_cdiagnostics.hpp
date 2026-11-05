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
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
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
 * @details Each step prints a line with all additions (+k) or drops (-k)
 * performed, shown as 0-based variable indices. Entries are stored as
 * 1-based signed indices (R lars convention, see TSolver_Base::getActions()).
 *
 * @param actions Vector of per-step action lists (positive = add, negative = drop).
 */
inline void print_actions_log(const std::vector<std::vector<int>> &actions)
{
    for (std::size_t step = 0; step < actions.size(); ++step)
    {
        std::cout << "Step " << std::setw(3) << step << ": ";
        for (int action : actions[step]) {
            if (action > 0)
                std::cout << " +" << (action - 1);
            else
                std::cout << " -" << (-action - 1); // Negative is drop
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



// Selection of a fitted solver
// ===========================================================

/**
 * @brief Variables selected by a fitted solver, split into predictors and
 *        dummies (unified indices; dummies start at getDummyStartIndex()).
 */
struct Selection {
    /** @brief Selected predictors: active ones in active-set order, then the
     *  frozen ones in ascending order. */
    std::vector<std::size_t> predictors;
    /** @brief Selected dummies, in the same order convention. */
    std::vector<std::size_t> dummies;
    /** @brief Selected variables outside the active set (predictors and
     *  dummies, ascending): T-Stagewise keeps the coefficient of a variable
     *  its NNLS reduction removes from the active set. */
    std::vector<std::size_t> frozen;
};

/**
 * @brief The selection of a fitted solver as the T-Rex selector reads it:
 *        the variables with |beta_j| > machine epsilon at the last recorded
 *        step of the coefficient path.
 *
 * @details For most solvers this is the active set. T-Stagewise keeps the
 *          coefficients of variables it removes from the active set, so its
 *          selection can exceed the active set; a solver whose working set runs
 *          ahead of its last recorded step (a CD solve at its sweep budget) is
 *          reported by the recorded step.
 *
 * @tparam SolverType Concrete solver type exposing getBetaPathSparse(),
 *                    getActives() and getDummyStartIndex().
 *
 * @param solver Fitted solver instance.
 *
 * @return Selected predictors, dummies and the frozen part.
 */
template<typename SolverType>
inline Selection selected_variables(const SolverType& solver) {
    Selection sel;
    const auto path = solver.getBetaPathSparse();
    if (path.steps.empty()) { return sel; }

    const auto& last = path.steps.back();
    const double eps = std::numeric_limits<double>::epsilon();
    std::vector<std::size_t> support;
    for (std::size_t k = 0; k < last.idx.size(); ++k) {
        if (std::abs(last.val[k]) > eps) { support.push_back(last.idx[k]); }
    }
    std::sort(support.begin(), support.end());
    auto in_support = [&support](std::size_t j) {
        return std::binary_search(support.begin(), support.end(), j);
    };

    const std::size_t dummy_start = solver.getDummyStartIndex();
    std::vector<std::size_t> listed;
    for (std::size_t j : solver.getActives()) {
        if (!in_support(j)) { continue; }
        (j >= dummy_start ? sel.dummies : sel.predictors).push_back(j);
        listed.push_back(j);
    }
    std::sort(listed.begin(), listed.end());
    for (std::size_t j : support) {
        if (std::binary_search(listed.begin(), listed.end(), j)) { continue; }
        (j >= dummy_start ? sel.dummies : sel.predictors).push_back(j);
        sel.frozen.push_back(j);
    }
    return sel;
}
// ==========================================================



// Print selection diagnostics
// ===========================================================

/**
 * @brief Print selected variables and dummies from fitted solver's solution path
 *        with color coding.
 *
 * @details The selection is read from the coefficient path
 *          (selected_variables()), as the T-Rex selector reads it.
 *
 * @tparam SolverType Concrete solver type exposing getBetaPathSparse(),
 *                    getActives(), getActions(), and getDummyStartIndex().
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
    const Selection sel = selected_variables(solver);

    // 1. Selected predictors
    std::cout << "  Predictors: ";
    print_vector(sel.predictors);

    // 2. Selected dummies
    std::cout << "\n  Dummies: ";
    print_vector(sel.dummies);

    // Variables that carry a coefficient outside the active set
    if (!sel.frozen.empty()) {
        std::cout << "\n  Frozen (coefficient kept outside the active set): ";
        print_vector(sel.frozen);
    }

    // 3. Full selection path (with dummies colored)
    // Actions are stored as 1-based signed indices (R lars convention,
    // see TSolver_Base::getActions()); decode with |a| - 1.
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
                std::size_t removed_idx = static_cast<std::size_t>(-action) - 1;
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
                std::size_t idx = static_cast<std::size_t>(action) - 1;
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
 * @details The selected predictors are read from the coefficient path
 *          (selected_variables()), as the T-Rex selector reads them.
 *
 * @tparam TSolver Concrete solver type exposing getBetaPathSparse(),
 *                 getActives() and getDummyStartIndex().
 *
 * @param solver The solver instance after fitting.
 * @param true_support The indices of the true support variables.
 */
template <typename TSolver>
void print_selection_quality(
    const TSolver& solver,
    const std::vector<std::size_t>& true_support
) {
    // Selected predictors from the coefficient path
    auto selected = selected_variables(solver).predictors;

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
