// ===================================================================================
// lp_norm_scaler.cpp
// ===================================================================================
/**
 * @file lp_norm_scaler.cpp
 *
 * @brief Implementation of the LpNormScaler class for L1/L2 normalization and centering.
 *
 * @details The class is separated into a header (lp_norm_scaler.hpp) and source file
 *  (lp_norm_scaler.cpp) for better organization.
 * |--LpNormScaler
 *    |--lp_norm_scaler.hpp  (class declaration)
 *    |--lp_norm_scaler.cpp  (class implementation)
 */
// ===================================================================================

#include <ml_methods/scaler_methods/lp_norm_scaler.hpp>

// ===================================================================================

namespace trex {
namespace ml_methods {
namespace scaler_methods {

// ===================================================================================

double LpNormScaler::computeScaleStat(const Eigen::Ref<const Eigen::VectorXd>& col,
                                      double center) const {
    // Lp norm around the applied center (center == 0 when centering is
    // disabled); expression templates avoid a temporary column copy.
    const auto col_centered = col.array() - center;
    return (norm_type_ == NormType::L2)
        ? col_centered.matrix().norm()
        : col_centered.matrix().lpNorm<1>();
}

// ===================================================================================

} /* End of namespace scaler_methods */
} /* End of namespace ml_methods */
} /* End of namespace trex */
