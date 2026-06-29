// ===================================================================================
// lp_norm_scaler.cpp
// ===================================================================================
/**
 * @file lp_norm_scaler.cpp
 *
 * @brief Implementation of the LpNormScaler class for L1/L2 normalization and centering.
 *
 * @details The class is separated into a header (LpNormScaler.hpp) and source file
 *  (LpNormScaler.cpp) for better organization.
 * |--LpNormScaler
 *    |--LpNormScaler.hpp  (class declaration)
 *    |--LpNormScaler.cpp  (class implementation)
 * Serialization is implemented in the header to allow for template binding with Cereal.
 */
// ===================================================================================

#include <algorithm>

#include <utils/openmp/utils_openmp.hpp>
#include <ml_methods/scaler_methods/lp_norm_scaler.hpp>

// ===================================================================================

namespace trex {
namespace ml_methods {
namespace scaler_methods {

// ===================================================================================

DataTransformer& LpNormScaler::fit(
    Eigen::Map<Eigen::MatrixXd>& X,
    double min_norm_threshold
    )
{

    Eigen::Index ncols = X.cols();
    Eigen::Index nrows = X.rows();

    if (nrows == 0 || ncols == 0) {
        throw std::invalid_argument(get_name() + "::fit: Empty matrix provided");
    }

    // Allocate storage
    means_.resize(ncols);
    norms_.resize(ncols);
    dropped_indices_.clear();

    #pragma omp parallel
    {
        // Thread-local storage for dropped indices
        std::vector<std::size_t> thread_dropped;

        #pragma omp for schedule(static) nowait
        for (Eigen::Index j = 0; j < ncols; ++j) {
            // Compute mean
            means_[j] = with_mean_ ? X.col(j).mean() : 0.0;

            // Compute norm
            if (with_norm_) {
                double norm_val;

                if (with_mean_) {
                    // Use expression templates to avoid allocation
                    auto col_centered = X.col(j).array() - means_[j];
                    norm_val = (norm_type_ == NormType::L2)
                        ? col_centered.matrix().norm()
                        : col_centered.matrix().template lpNorm<1>();
                } else {
                    norm_val = (norm_type_ == NormType::L2)
                        ? X.col(j).norm()
                        : X.col(j).template lpNorm<1>();
                }

                // Check threshold
                if (norm_val < min_norm_threshold) {
                    thread_dropped.push_back(j);
                    norms_[j] = 1.0;  // Prevent division by zero
                } else {
                    norms_[j] = norm_val;
                }
            } else {
                norms_[j] = 1.0;  // No normalization
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


void LpNormScaler::transform_inplace(Eigen::Map<Eigen::MatrixXd>& X) const {

    const std::string transformer_name = get_name();

    if (!fitted_) {
        throw std::logic_error(transformer_name + "::transform_inplace: Not fitted");
    }

    if (X.cols() != static_cast<Eigen::Index>(norms_.size())) {
        throw std::invalid_argument(
            transformer_name +
            "::transform_inplace: Expected " +
            std::to_string(norms_.size()) + " features, got " +
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
        if (with_norm_ && norms_[j] > 0) {
            X.col(j).array() /= norms_[j];
        }
    }
}


void LpNormScaler::inverse_transform_inplace(Eigen::Map<Eigen::MatrixXd>& X_normed) const {

    const std::string transformer_name = get_name();

    if (!fitted_) {
        throw std::logic_error(
            transformer_name + "::inverse_transform_inplace: Not fitted");
    }

    if (X_normed.cols() != static_cast<Eigen::Index>(norms_.size())) {
        throw std::invalid_argument(
            transformer_name +
            "::inverse_transform_inplace: Expected " +
            std::to_string(norms_.size()) + " features, got " +
            std::to_string(X_normed.cols()));
    }

    const bool has_dropped = !dropped_indices_.empty();

    #pragma omp parallel for schedule(static)
    for (Eigen::Index j = 0; j < X_normed.cols(); ++j) {

        // Skip dropped columns
        if (has_dropped &&
            std::binary_search(dropped_indices_.begin(),
                               dropped_indices_.end(), j)) {
            continue;
        }

        // Unscale (inverse order of transform)
        if (with_norm_ && norms_[j] > 0) {
            X_normed.col(j).array() *= norms_[j];
        }

        // Uncenter
        if (with_mean_) {
            X_normed.col(j).array() += means_[j];
        }
    }
}


// ===================================================================================
} /* End of namespace scaler_methods */
} /* End of namespace ml_methods */
} /* End of namespace trex */
