// ===================================================================================
// tik_lambda2_pick.hpp
// ===================================================================================
#ifndef TREX_SELECTOR_METHODS_TREX_TIKHONOV_TIK_LAMBDA2_PICK_HPP
#define TREX_SELECTOR_METHODS_TREX_TIKHONOV_TIK_LAMBDA2_PICK_HPP
// ===================================================================================
/**
 * @file tik_lambda2_pick.hpp
 *
 * @brief The pick of a lambda_2 from a selection-count profile over a grid —
 *        the pure part of the power rule, shared by the external wrapper
 *        (tik_lambda2_power_rule.hpp, one full selection per grid point) and
 *        the in-selector pilot rule (TRexTikhonovSelector, a few random
 *        experiments per grid point).
 *
 * @details With `n_k` the selection count at the k-th grid point (ascending
 *   lambda_2), the pick is the knee of the profile: the smallest k with
 *   n_k >= max_k n_k - tolerance (the "minimum sufficient shrinkage"
 *   reading). A tolerance of 0 reads the largest selection, ties resolved
 *   toward the smallest lambda_2 (the least reliance on the prior).
 *
 *   Selector-free on purpose: this header includes nothing of the T-Rex, so
 *   the selector can include it.
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

// Eigen includes
#include <Eigen/Core>

// Bit-based finiteness: std::isfinite folds to true when a consumer compiles
// this header with -ffast-math (the library itself is built IEEE-conformant).
#include <utils/fp_classify/fp_classify.hpp>

// ===================================================================================

namespace trex {
namespace trex_selector_methods {
namespace ad {
namespace trex_tikhonov {

namespace fpc = trex::utils::fp_classify;

/** @brief What the pick over the counts looked like. */
struct Lambda2PickInfo {

    /** @brief Index of the pick into the ascending grid. */
    std::size_t index = 0;

    /** @brief The largest selection count on the grid. */
    std::size_t max_selected = 0;

    /** @brief Grid points within the tolerance of the maximum (at
     *  tolerance 0: the points attaining it). */
    std::size_t n_ties = 0;
};

/**
 * @brief Read the pick off a selection-count profile (ascending lambda_2).
 *
 * @param counts         Selection counts per grid point.
 * @param knee_tolerance The plateau tolerance in selections (>= 0); 0 reads
 *                       the largest selection.
 *
 * @throws std::invalid_argument on an empty profile or a negative tolerance.
 */
inline Lambda2PickInfo pick_lambda2_index(const std::vector<std::size_t>& counts,
                                          double knee_tolerance = 1.0)
{
    if (counts.empty()) {
        throw std::invalid_argument("pick_lambda2_index: empty selection-count profile");
    }
    if (!(knee_tolerance >= 0.0)) {
        throw std::invalid_argument("pick_lambda2_index: knee_tolerance must be >= 0");
    }
    Lambda2PickInfo info;
    info.max_selected = *std::max_element(counts.begin(), counts.end());
    const double threshold = static_cast<double>(info.max_selected) - knee_tolerance;
    bool found = false;
    for (std::size_t k = 0; k < counts.size(); ++k) {
        if (static_cast<double>(counts[k]) >= threshold) {
            if (!found) {
                info.index = k;
                found      = true;
            }
            ++info.n_ties;
        }
    }
    return info;
}

/** @brief The group-majority count of one pilot probe. */
struct GroupMajorityCount {

    /** @brief Members ahead (phi > 1/2), summed over the qualifying groups. */
    std::size_t members = 0;

    /** @brief Groups in which at least `fraction` of the members are ahead. */
    std::size_t groups = 0;
};

/**
 * @brief The majority count read per group: a group counts only if at least
 *        `fraction` of its members are ahead (phi > 1/2), and then with all
 *        of its members ahead. Scattered entries of a group — the false
 *        positives a group prior admits from a null group correlated with an
 *        active one — contribute nothing; a group entering as a whole
 *        contributes as a whole.
 *
 * @param phi       One value per predictor (the T-Rex's relative occurrence
 *                  at the pilot stopping time); "ahead" is phi > 1/2.
 * @param groups    Group id per predictor (any non-negative ids).
 * @param fraction  The share of a group that must be ahead, in (0, 1].
 *
 * @throws std::invalid_argument when the sizes differ, the fraction is
 *         outside (0, 1], or an id is negative.
 */
inline GroupMajorityCount group_majority_count(const Eigen::VectorXd& phi,
                                               const std::vector<Eigen::Index>& groups,
                                               double fraction)
{
    if (static_cast<std::size_t>(phi.size()) != groups.size()) {
        throw std::invalid_argument("group_majority_count: phi and groups differ in size");
    }
    if (!(fraction > 0.0 && fraction <= 1.0)) {
        throw std::invalid_argument("group_majority_count: fraction must be in (0, 1]");
    }
    Eigen::Index M = 0;
    for (const Eigen::Index g : groups) {
        if (g < 0) {
            throw std::invalid_argument("group_majority_count: negative group id");
        }
        M = std::max(M, g + 1);
    }
    std::vector<std::size_t> size(static_cast<std::size_t>(M), 0);
    std::vector<std::size_t> ahead(static_cast<std::size_t>(M), 0);
    for (std::size_t j = 0; j < groups.size(); ++j) {
        const auto g = static_cast<std::size_t>(groups[j]);
        ++size[g];
        if (phi(static_cast<Eigen::Index>(j)) > 0.5) { ++ahead[g]; }
    }
    GroupMajorityCount out;
    for (std::size_t g = 0; g < size.size(); ++g) {
        if (size[g] == 0) { continue; }
        if (static_cast<double>(ahead[g]) >= fraction * static_cast<double>(size[g])) {
            out.members += ahead[g];
            ++out.groups;
        }
    }
    return out;
}

/**
 * @brief The validated, ascending, de-duplicated lambda_2 grid.
 *
 * @throws std::invalid_argument on an empty grid or a negative / non-finite
 *         value.
 */
inline std::vector<double> validate_lambda2_grid(const std::vector<double>& grid_in)
{
    if (grid_in.empty()) {
        throw std::invalid_argument("validate_lambda2_grid: the lambda_2 grid is empty");
    }
    for (const double v : grid_in) {
        if (!(v >= 0.0) || !fpc::isfinite(v)) {
            throw std::invalid_argument("validate_lambda2_grid: lambda_2 values must be finite and >= 0");
        }
    }
    std::vector<double> grid = grid_in;
    std::sort(grid.begin(), grid.end());
    grid.erase(std::unique(grid.begin(), grid.end()), grid.end());
    return grid;
}

}  // namespace trex_tikhonov
}  // namespace ad
}  // namespace trex_selector_methods
}  // namespace trex

#endif  // TREX_SELECTOR_METHODS_TREX_TIKHONOV_TIK_LAMBDA2_PICK_HPP
