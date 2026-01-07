// ===================================================================================
// utils_eval_composite.hpp
// ===================================================================================
/**
 * @file utils_eval_composite.hpp
 *
 * @brief Composite evaluation utilities for console diagnostics.
 *
 */
// ===================================================================================

#ifndef TREX_UTILS_EVAL_COMPOSITE_HPP
#define TREX_UTILS_EVAL_COMPOSITE_HPP

// ===================================================================================


// ===================================================================================
namespace trex {
namespace utils {
namespace eval {
namespace composites {

// ===================================================================================
// Set up namespace aliases
// ===================================================================================
// none

// ===================================================================================

/**
 * @brief F1 Score computation
 *
 * @param precision The precision value.
 * @param recall The recall value.
 *
 * @return The computed F1 score.
 */
double f1_score(
    double precision,
    double recall
) {
    if (precision + recall == 0.0) {
        return 0.0;
    } else {
        return 2.0 * (precision * recall) / (precision + recall);
    }
}


// ===================================================================================

} /* End of namespace composites */
} /* End of namespace eval */
} /* End of namespace utils */
} /* End of namespace trex */

#endif /* End of TREX_UTILS_EVAL_COMPOSITE_HPP */
