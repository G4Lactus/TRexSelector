#ifndef UTILS_FDR_CONTROL_HPP
#define UTILS_FDR_CONTROL_HPP

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>
#include <unordered_set>

#include <Eigen/Core>
#include <Eigen/Dense>



namespace utils_fdr_control {





// ===========================================================
// 1. Index-Based API
// ===========================================================

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
    if (selected_indices.empty() || true_support.empty()) {
        return 0.0; // No selection -> no false discoveries
    }

    // Convert true support to set for O(log n) lookup
    std::unordered_set<std::size_t> true_support_set(
        true_support.begin(), true_support.end()
    );

    // Count false positives: selected but not in true support
    std::size_t num_false_positives = 0;
    for (auto idx : selected_indices) {
        if (true_support_set.find(idx) == true_support_set.end()) {
            num_false_positives++;
        }
    }

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
        return 0.0;  // No active variables → undefined, define as 0
    }

    // Convert selected to set for O(log n) lookup
    std::unordered_set<std::size_t> selected_set(
        selected_indices.begin(),
        selected_indices.end()
    );

    // Count true positives: in true_support AND selected
    std::size_t num_true_positives = 0;
    for (auto idx : true_support) {
        if (selected_set.find(idx) != selected_set.end()) {
            ++num_true_positives;
        }
    }

    return static_cast<double>(num_true_positives) /
           static_cast<double>(true_support.size());
}


// ===========================================================
// 3. Dense Coefficient Vector API
// ===========================================================

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


} /* End of namespace utils_fdr_control */
#endif /* End of UTILS_FDR_CONTROL_HPP */
