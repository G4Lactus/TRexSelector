// ===================================================================================
// data_transformer.cpp
// ===================================================================================
/**
 * @file data_transformer.cpp
 *
 * @brief Shared fit/transform machinery for column-wise data transformations.
 *
 * @details Implements the template-method core used by every scaler: fit()
 * computes per-column centers and delegates the scale statistic to the derived
 * class; transform_inplace()/inverse_transform_inplace() apply the affine
 * column map and its inverse.
 */
// ===================================================================================

#include <algorithm>

#include <ml_methods/scaler_methods/data_transformer.hpp>
#include <utils/openmp/utils_openmp.hpp>

// ===================================================================================

namespace trex {
namespace ml_methods {
namespace scaler_methods {

// ===================================================================================

DataTransformer& DataTransformer::fit(Eigen::Map<Eigen::MatrixXd>& X,
                                      double threshold) {

    const Eigen::Index ncols = X.cols();
    const Eigen::Index nrows = X.rows();

    if (nrows == 0 || ncols == 0) {
        throw std::invalid_argument(get_name() + "::fit: Empty matrix provided");
    }
    checkFitPreconditions(nrows);

    // Allocate storage
    centers_.resize(ncols);
    scales_.resize(ncols);
    dropped_indices_.clear();

    #pragma omp parallel
    {
        // Thread-local storage for degenerate column indices
        std::vector<std::size_t> thread_dropped;

        #pragma omp for schedule(static) nowait
        for (Eigen::Index j = 0; j < ncols; ++j) {

            // Compute center
            centers_[j] = center_ ? X.col(j).mean() : 0.0;

            // Compute the scale statistic around the applied center
            if (scale_) {
                double stat = computeScaleStat(X.col(j), centers_[j]);

                // Degenerate column: force scale 1.0 (still centered, never
                // divided) and report it — sklearn behavior.
                if (stat < threshold) {
                    thread_dropped.push_back(static_cast<std::size_t>(j));
                    scales_[j] = 1.0;
                } else {
                    scales_[j] = stat;
                }
            } else {
                scales_[j] = 1.0;  // No scaling
            }
        }

        // Merge thread-local degenerate indices
        if (!thread_dropped.empty()) {
            #pragma omp critical
            {
                dropped_indices_.insert(dropped_indices_.end(),
                                        thread_dropped.begin(),
                                        thread_dropped.end());
            }
        }
    }

    // Sort degenerate indices for consistent ordering
    std::sort(dropped_indices_.begin(), dropped_indices_.end());

    fitted_ = true;
    return *this;  // Enable method chaining
}


void DataTransformer::transform_inplace(Eigen::Map<Eigen::MatrixXd>& X) const {

    if (!fitted_) {
        throw std::logic_error(get_name() + "::transform_inplace: Not fitted");
    }

    if (X.cols() != static_cast<Eigen::Index>(scales_.size())) {
        throw std::invalid_argument(
            get_name() + "::transform_inplace: Expected " +
            std::to_string(scales_.size()) + " features, got " +
            std::to_string(X.cols()));
    }

    // Degenerate columns need no special-casing: their scale factor is 1.0,
    // so they are centered like every other column and never divided.
    #pragma omp parallel for schedule(static)
    for (Eigen::Index j = 0; j < X.cols(); ++j) {

        // Center
        if (center_) {
            X.col(j).array() -= centers_[j];
        }

        // Scale (fit guarantees scales_[j] >= threshold or == 1.0)
        if (scale_) {
            X.col(j).array() /= scales_[j];
        }
    }
}


void DataTransformer::inverse_transform_inplace(
    Eigen::Map<Eigen::MatrixXd>& X_transformed) const {

    if (!fitted_) {
        throw std::logic_error(get_name() + "::inverse_transform_inplace: Not fitted");
    }

    if (X_transformed.cols() != static_cast<Eigen::Index>(scales_.size())) {
        throw std::invalid_argument(
            get_name() + "::inverse_transform_inplace: Expected " +
            std::to_string(scales_.size()) + " features, got " +
            std::to_string(X_transformed.cols()));
    }

    #pragma omp parallel for schedule(static)
    for (Eigen::Index j = 0; j < X_transformed.cols(); ++j) {

        // Unscale (inverse order of transform)
        if (scale_) {
            X_transformed.col(j).array() *= scales_[j];
        }

        // Uncenter
        if (center_) {
            X_transformed.col(j).array() += centers_[j];
        }
    }
}

// ===================================================================================

} /* End of namespace scaler_methods */
} /* End of namespace ml_methods */
} /* End of namespace trex */
