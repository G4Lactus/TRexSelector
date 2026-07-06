// ===================================================================================
// z_score_scaler.cpp
// ===================================================================================
/**
 * @file z_score_scaler.cpp
 *
 * @brief Implementation of the ZScoreScaler class for feature scaling.
 *
 * @details The class is separated into a header (ZScoreScaler.hpp) and source file
 * (ZScoreScaler.cpp) for better organization.
 * |--ZScoreScaler
 *    |--ZScoreScaler.hpp  (class declaration)
 *    |--ZScoreScaler.cpp  (class implementation)
 */
// ===================================================================================

#include <algorithm>

#include <ml_methods/scaler_methods/z_score_scaler.hpp>
#include <utils/openmp/utils_openmp.hpp>

// ===================================================================================

namespace trex {
namespace ml_methods {
namespace scaler_methods {

// ===================================================================================

DataTransformer& ZScoreScaler::fit(Eigen::Map<Eigen::MatrixXd>& X,
                                   double min_std_threshold) {

    Eigen::Index ncols = X.cols();
    Eigen::Index nrows = X.rows();

    if (nrows == 0 || ncols == 0) {
        throw std::invalid_argument(
            get_name() + "::fit: Empty matrix provided");
    }
    if (with_std_ && nrows < 2) {
        throw std::invalid_argument(
            get_name() + "::fit: with_std requires at least 2 samples "
            "(the Bessel-corrected standard deviation divides by n - 1)");
    }

    // Allocate storage
    means_.resize(ncols);
    scales_.resize(ncols);
    dropped_indices_.clear();

    #pragma omp parallel
    {
        // Thread-local storage for dropped indices
        std::vector<std::size_t> thread_dropped;

        #pragma omp for schedule(static) nowait
        for (Eigen::Index j = 0; j < ncols; ++j) {
            // Compute mean
            means_[j] = with_mean_ ? X.col(j).mean() : 0.0;

            // Compute standard deviation
            if (with_std_) {
                // Use expression templates to avoid allocation
                auto col_centered = X.col(j).array() - means_[j];
                double variance = col_centered.square().sum() / static_cast<double>((nrows - 1));
                double std_val = std::sqrt(variance);

                // Check threshold
                if (std_val < min_std_threshold) {
                    thread_dropped.push_back(j);
                    scales_[j] = 1.0;  // Prevent division by zero
                } else {
                    scales_[j] = std_val;
                }
            } else {
                scales_[j] = 1.0;  // No scaling
            }
        }

        // Merge thread-local dropped indices
        if (!thread_dropped.empty()) {
            #pragma omp critical
            {
                dropped_indices_.insert(dropped_indices_.end(),
                                    thread_dropped.begin(),
                                    thread_dropped.end());
            }
        }
    }

    // Sort dropped indices for consistent ordering
    std::sort(dropped_indices_.begin(), dropped_indices_.end());

    fitted_ = true;
    return *this;  // Enable method chaining
}


void ZScoreScaler::transform_inplace(Eigen::Map<Eigen::MatrixXd>& X) const {

    const std::string transformer_name = get_name();

    if (!fitted_) {
        throw std::logic_error(transformer_name + "::transform_inplace: Not fitted");
    }

    if (X.cols() != static_cast<Eigen::Index>(scales_.size())) {
        throw std::invalid_argument(
            transformer_name + "::transform_inplace: Expected " +
            std::to_string(scales_.size()) + " features, got " +
            std::to_string(X.cols()));
    }

    const bool has_dropped = !dropped_indices_.empty();

    #pragma omp parallel for schedule(static)
    for (Eigen::Index j = 0; j < X.cols(); ++j) {

        // Skip dropped columns
        if (has_dropped &&
            std::binary_search(dropped_indices_.begin(),
                               dropped_indices_.end(), j)) {
            continue;
        }

        // Center
        if (with_mean_) {
            X.col(j).array() -= means_[j];
        }

        // Scale
        if (with_std_ && scales_[j] > 0) {
            X.col(j).array() /= scales_[j];
        }
    }
}


void ZScoreScaler::inverse_transform_inplace(Eigen::Map<Eigen::MatrixXd>& X_scaled) const {

    const std::string transformer_name = get_name();

    if (!fitted_) {
        throw std::logic_error(transformer_name + "::inverse_transform_inplace: Not fitted");
    }

    if (X_scaled.cols() != static_cast<Eigen::Index>(scales_.size())) {
        throw std::invalid_argument(
            transformer_name + "::inverse_transform_inplace: Expected " +
            std::to_string(scales_.size()) + " features, got " +
            std::to_string(X_scaled.cols()));
    }

    const bool has_dropped = !dropped_indices_.empty();

    #pragma omp parallel for schedule(static)
    for (Eigen::Index j = 0; j < X_scaled.cols(); ++j) {

        // Skip dropped columns
        if (has_dropped &&
            std::binary_search(dropped_indices_.begin(),
                               dropped_indices_.end(), j)) {
            continue;
        }

        // Unscale (inverse order of transform)
        if (with_std_ && scales_[j] > 0) {
            X_scaled.col(j).array() *= scales_[j];
        }

        // Uncenter
        if (with_mean_) {
            X_scaled.col(j).array() += means_[j];
        }
    }
}

// ===================================================================================

} /* End of namespace scaler_methods */
} /* End of namespace ml_methods */
} /* End of namespace trex */
