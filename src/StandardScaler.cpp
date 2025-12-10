#include "StandardScaler.hpp"
#include <algorithm>


DataTransformer& StandardScaler::fit(const Eigen::Map<const Eigen::MatrixXd>& X,
                                     double min_std_threshold) {

    std::size_t ncols = X.cols();
    std::size_t nrows = X.rows();

    if (nrows == 0 || ncols == 0) {
        throw std::invalid_argument(
            get_name() + "::fit: Empty matrix provided");
    }

    // Allocate storage
    means_.resize(ncols);
    scales_.resize(ncols);
    dropped_indices_.clear();

    // Collect dropped indices per-thread to avoid critical section bottleneck
    std::vector<std::size_t> local_dropped;

    #pragma omp parallel
    {
        // Thread-local storage for dropped indices
        std::vector<std::size_t> thread_dropped;

        #pragma omp for schedule(static) nowait
        for (std::size_t j = 0; j < ncols; ++j) {
            // Compute mean
            means_[j] = with_mean_ ? X.col(j).mean() : 0.0;

            // Compute standard deviation
            if (with_std_) {
                // Use expression templates to avoid allocation
                auto col_centered = X.col(j).array() - means_[j];
                double variance = col_centered.square().sum() / (nrows - 1);
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
        #pragma omp critical
        {
            local_dropped.insert(local_dropped.end(),
                                thread_dropped.begin(),
                                thread_dropped.end());
        }
    }

    // Sort dropped indices for consistent ordering
    dropped_indices_ = std::move(local_dropped);
    std::sort(dropped_indices_.begin(), dropped_indices_.end());

    fitted_ = true;
    return *this;  // Enable method chaining
}


void StandardScaler::transform_inplace(Eigen::Map<Eigen::MatrixXd>& X) const {
    check_fitted("transform_inplace");
    check_dimension(scales_.size(), X.cols(), "transform_inplace");

    #pragma omp parallel for schedule(static)
    for (std::size_t j = 0; j < static_cast<std::size_t>(X.cols()); ++j) {
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


void StandardScaler::inverse_transform_inplace(
    Eigen::Map<Eigen::MatrixXd>& X_scaled) const {
    check_fitted("inverse_transform_inplace");
    check_dimension(scales_.size(), X_scaled.cols(), "inverse_transform_inplace");

    #pragma omp parallel for schedule(static)
    for (std::size_t j = 0; j < static_cast<std::size_t>(X_scaled.cols()); ++j) {
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


Eigen::VectorXd StandardScaler::get_l2_conversion_factor(std::size_t n_samples) const {
    check_fitted("get_l2_conversion_factor");

    if (n_samples == 0) {
        throw std::invalid_argument(
            "get_l2_conversion_factor: n_samples must be > 0");
    }

    Eigen::VectorXd conversion(scales_.size());

    #pragma omp parallel for schedule(static)
    for (std::size_t j = 0; j < static_cast<std::size_t>(scales_.size()); ++j) {
        // Convert from standardization to L2 scale
        // Formula: β_L2 = β_scaled * scale_j / sqrt(n)
        if (with_std_) {
            conversion[j] = scales_[j] / std::sqrt(static_cast<double>(n_samples));
        } else {
            // No scaling applied
            conversion[j] = 1.0 / std::sqrt(static_cast<double>(n_samples));
        }
    }

    return conversion;
}
