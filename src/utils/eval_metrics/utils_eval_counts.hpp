// ===================================================================================
// utils_eval_counts.hpp
// ===================================================================================
/**
 * @file utils_eval_counts.hpp
 *
 * @brief Functions to compute counts of true positives, false positives, and
 *        false negatives.
 *
 */
// ===================================================================================

#ifndef TREX_UTILS_EVAL_METRICS_COUNTS_HPP
#define TREX_UTILS_EVAL_METRICS_COUNTS_HPP

// ===================================================================================

#include <algorithm>
#include <iostream>
#include <vector>
#include <unordered_set>


// ===================================================================================

namespace trex {
namespace utils {
namespace eval {
namespace counts {

// ===================================================================================


/**
 * @brief Count for the number of true positives in a vector of selected indices.
 *
 * @param selected_indices Proposed selected variable indices.
 * @param true_support Indices of true active variables.
 *
 * @return Number of true positives.
 */
std::size_t true_positives_count(
    const std::vector<std::size_t>& selected_indices,
    const std::vector<std::size_t>& true_support
) {
    if (selected_indices.empty() || true_support.empty()) {
        return 0;
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

    return num_true_positives;
}


/**
 * @brief Count for the number of false positives in a vector of selected indices.
 *
 * @param selected_indices Proposed selected variable indices.
 * @param true_support Indices of true active variables.
 *
 * @return Number of false positives.
 */
std::size_t false_positives_count(
    const std::vector<std::size_t>& selected_indices,
    const std::vector<std::size_t>& true_support
) {
    if (selected_indices.empty()) {
        return 0;
    }

    // Convert true support to set for O(log n) lookup
    std::unordered_set<std::size_t> true_support_set(
        true_support.begin(), true_support.end()
    );

    // Count false positives: selected but not in true support
    std::size_t num_false_positives = 0;
    for (auto idx : selected_indices) {
        if (true_support_set.find(idx) == true_support_set.end()) {
            ++num_false_positives;
        }
    }

    return num_false_positives;
}


/**
 * @brief Count for the number of false negatives in a vector of selected indices.
 *
 * @param selected_indices Proposed selected variable indices.
 * @param true_support Indices of true active variables.
 *
 * @return Number of false negatives.
 */
std::size_t false_negatives_count(
    const std::vector<std::size_t>& selected_indices,
    const std::vector<std::size_t>& true_support
) {
    if (true_support.empty()) {
        return 0;
    }

    // Convert selected to set for O(log n) lookup
    std::unordered_set<std::size_t> selected_set(
        selected_indices.begin(),
        selected_indices.end()
    );

    // Count false negatives: in true_support but not selected
    std::size_t num_false_negatives = 0;
    for (auto idx : true_support) {
        if (selected_set.find(idx) == selected_set.end()) {
            ++num_false_negatives;
        }
    }

    return num_false_negatives;
}


// ===================================================================================

} /* End of namespace counts */
} /* End of namespace eval */
} /* End of namespace utils */
} /* End of namespace trex */

#endif /* End of TREX_UTILS_EVAL_METRICS_COUNTS_HPP */
