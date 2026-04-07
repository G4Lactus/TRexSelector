// ===================================================================================
// utils_eval_composite.hpp
// ===================================================================================
#ifndef UTILS_EVAL_COMPOSITE_HPP
#define UTILS_EVAL_COMPOSITE_HPP
// ===================================================================================
/**
 * @file utils_eval_composite.hpp
 *
 * @brief Composite evaluation utilities for console diagnostics.
 *
 */
// ===================================================================================

// Embedded into namespace trex::utils::eval::composites
namespace trex::utils::eval::composites {

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

} /* End of namespace trex::utils::eval::composites */

#endif /* UTILS_EVAL_COMPOSITE_HPP */
