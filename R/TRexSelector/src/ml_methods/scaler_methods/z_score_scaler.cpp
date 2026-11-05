// ===================================================================================
// z_score_scaler.cpp
// ===================================================================================
/**
 * @file z_score_scaler.cpp
 *
 * @brief Implementation of the ZScoreScaler class for feature scaling.
 *
 * @details The class is separated into a header (z_score_scaler.hpp) and source file
 * (z_score_scaler.cpp) for better organization.
 * |--ZScoreScaler
 *    |--z_score_scaler.hpp  (class declaration)
 *    |--z_score_scaler.cpp  (class implementation)
 */
// ===================================================================================

#include <cmath>

#include <ml_methods/scaler_methods/z_score_scaler.hpp>

// ===================================================================================

namespace trex {
namespace ml_methods {
namespace scaler_methods {

// ===================================================================================

double ZScoreScaler::computeScaleStat(const Eigen::Ref<const Eigen::VectorXd>& col,
                                      double center) const {
    // Bessel-corrected SD around the applied center; with centering disabled
    // (center == 0) this is the root-mean-square, matching R's scale().
    const auto col_centered = col.array() - center;
    const double variance =
        col_centered.square().sum() / static_cast<double>(col.size() - 1);
    return std::sqrt(variance);
}


void ZScoreScaler::checkFitPreconditions(Eigen::Index nrows) const {
    if (scale_ && nrows < 2) {
        throw std::invalid_argument(
            get_name() + "::fit: scale requires at least 2 samples "
            "(the Bessel-corrected standard deviation divides by n - 1)");
    }
}

// ===================================================================================

} /* End of namespace scaler_methods */
} /* End of namespace ml_methods */
} /* End of namespace trex */
