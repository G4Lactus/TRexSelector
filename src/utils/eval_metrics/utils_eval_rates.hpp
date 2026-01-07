// ===================================================================================
// utils_fdr_control.hpp
// ===================================================================================
/**
 * @brief Functions to compute false discovery proportion (FDP) and
 *        true positive proportion (TPP).
 */
// ===================================================================================

#ifndef TREX_UTILS_EVAL_RATES_HPP
#define TREX_UTILS_EVAL_RATES_HPP

// ===================================================================================

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>
#include <unordered_set>

#include <Eigen/Core>
#include <Eigen/Dense>


// ===================================================================================

namespace trex {
namespace utils {
namespace eval {
namespace rates {

// ===================================================================================
// Set up namespace aliases
// ===================================================================================
namespace eval = trex::utils::eval;

// ===================================================================================

// FDP and TPP computation
// ===========================================================


// Index-Based API
// ================================================

/**
 * @brief Compute FDP from index sets.
 *
 * @param selected_indices Indices of selected variables.
 * @param true_support Indices of true active variables.
 *
 * @return FDP = |false positives| / max(|selected variables|, 1)
 */
inline double compute_fdp(
    const std::vector<std::size_t>& selected_indices,
    const std::vector<std::size_t>& true_support
) {
    if (true_support.empty() || selected_indices.empty()) {
        return 0.0;
    }

    // Convert true support to set for O(log n) lookup
    std::unordered_set<std::size_t> true_support_set(
        true_support.begin(), true_support.end()
    );

    // Count false positives: selected but not in true support
    std::size_t num_false_positives = eval::counts::false_positives_count(
        selected_indices, true_support
    );

    // Compute FDP
    return static_cast<double>(num_false_positives) /
           static_cast<double>(selected_indices.size());
}


/**
 * @brief Compute TPP from index sets.
 *
 * @param selected_indices Indices of selected variables.
 * @param true_support Indices of true active variables.
 *
 * @return TPP = |true positives| / max(|active variables|, 1)
 */
inline double compute_tpp(
    const std::vector<std::size_t>& selected_indices,
    const std::vector<std::size_t>& true_support
) {
    if (true_support.empty() || selected_indices.empty()) {
        return 0.0;
    }

    // Convert selected to set for O(log n) lookup
    std::unordered_set<std::size_t> selected_set(
        selected_indices.begin(),
        selected_indices.end()
    );

    // Count true positives: in true_support AND selected
    std::size_t num_true_positives = eval::counts::true_positives_count(
        selected_indices, true_support
    );

    return static_cast<double>(num_true_positives) /
           static_cast<double>(true_support.size());
}



// Dense Coefficient Vector Based API
// ================================================

/**
 * @brief False Discovery Proportion (FDP)
 *
 * @details
 * Computes the FDP based on estimated and true regression coefficients.
 * FDP = (# false positives) / max(# selected variables, 1)
 *
 * @param beta_hat Estimated regression coefficient vector (Map-compatible)
 * @param beta True regeression coefficient vector (Map-compatible)
 *
 * @return False discovery proportion (FDP)
 */
template <typename Derived1, typename Derived2>
inline double compute_fdp(const Eigen::MatrixBase<Derived1>& beta_hat,
                   const Eigen::MatrixBase<Derived2>& beta,
                   double eps = std::numeric_limits<double>::epsilon()) {

    // Input validation
    if (beta_hat.size() != beta.size()) {
        throw std::invalid_argument("Size of beta_hat and beta must be the same.");
    }

    // Count selected variables: |beta_hat| > eps
    int num_selected_var = (beta_hat.array().abs() > eps).count();

    // Count false positives: |beta| <= eps && AND |beta_hat| > eps
    int num_false_positives = ((beta.array().abs() <= eps) &&
                              (beta_hat.array().abs() > eps)).count();

    // Compute FDP
    if (num_selected_var == 0) {
        return 0.0; // Define FDP as 0 when no variables are selected
    } else {
        return static_cast<double>(num_false_positives) / static_cast<double>(num_selected_var);
    }
}

/**
 * @brief True Positive Proporiton (TPP)
 *
 * @details
 * Computes the TPP (sensitivity/recall) based on estimated and true regression coefficients
 * vectors.
 * TPP = (# true positives) / max(# active variables, 1)
 *
 * @param beta_hat Estimated regression coefficient vector (Map-compatible).
 * @param beta True regeression coefficient vector (Map-compatible).
 * @param eps Numerical zero treshold.
 *
 * @return True positive proportion (TPP)
 */
template <typename Derived1, typename Derived2>
inline double compute_tpp(const Eigen::MatrixBase<Derived1>& beta_hat,
                   const Eigen::MatrixBase<Derived2>& beta,
                   double eps = std::numeric_limits<double>::epsilon()) {

    // Input validation
    if (beta_hat.size() != beta.size()) {
        throw std::invalid_argument("Size of beta_hat and beta must be the same.");
    }

    // Count active variables: |beta| > eps
    int num_active_vars = (beta.array().abs() > eps).count();

    // Count true positives: |beta| > eps && AND |beta_hat| > eps
    int num_true_positives = ((beta.array().abs() > eps) &&
                              (beta_hat.array().abs() > eps)).count();

    // Compute TPP
    if (num_active_vars == 0) {
        return 0.0; // Define TPP as 0 when no active variables
    } else {
        return static_cast<double>(num_true_positives) / static_cast<double>(num_active_vars);
    }
}


// Precision and Recall
// ===========================================================

/**
 * @brief Computes the precision based on selected indices and true support.
 *
 * @param selected_indices Proposed selected variable indices.
 * @param true_support Indices of true active variables.
 *
 * @return Precision
 */
inline double compute_precision(
    const std::vector<std::size_t>& selected_indices,
    const std::vector<std::size_t>& true_support
) {
    if (selected_indices.empty()) {
        return 0.0;
    }

    // Count true positives
    std::size_t num_true_positives = eval::counts::true_positives_count(
        selected_indices, true_support
    );

    return static_cast<double>(num_true_positives) /
           static_cast<double>(selected_indices.size());
}


/**
 * @brief Computes the recall based on selected indices and true support.
 *
 * @param selected_indices Proposed selected variable indices.
 * @param true_support Indices of true active variables.
 *
 * @return Recall
 */
inline double compute_recall(
    const std::vector<std::size_t>& selected_indices,
    const std::vector<std::size_t>& true_support
) {
    if (true_support.empty()) {
        return 0.0;
    }

    // Count true positives
    std::size_t num_true_positives = eval::counts::true_positives_count(
        selected_indices, true_support
    );

    return static_cast<double>(num_true_positives) /
           static_cast<double>(true_support.size());
}




// ===================================================================================

} /* End of namespace rates */
} /* End of namespace eval */
} /* End of namespace utils */
} /* End of namespace trex */

#endif /* End of TREX_UTILS_EVAL_RATES_HPP */
