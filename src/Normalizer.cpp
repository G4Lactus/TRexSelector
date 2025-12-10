#include "Normalizer.hpp"
#include <algorithm>

DataTransformer& Normalizer::fit(const Eigen::Map<const Eigen::MatrixXd>& X,
                                  double min_norm_threshold) {

    std::size_t ncols = X.cols();
    std::size_t nrows = X.rows();

    if (nrows == 0 || ncols == 0) {
        throw std::invalid_argument(
            get_name() + "::fit: Empty matrix provided");
    }

    // Allocate storage
    means_.resize(ncols);
    norms_.resize(ncols);
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

void Normalizer::transform_inplace(Eigen::Map<Eigen::MatrixXd>& X) const {
    check_fitted("transform_inplace");
    check_dimension(norms_.size(), X.cols(), "transform_inplace");

    #pragma omp parallel for schedule(static)
    for (std::size_t j = 0; j < static_cast<std::size_t>(X.cols()); ++j) {
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

void Normalizer::inverse_transform_inplace(
    Eigen::Map<Eigen::MatrixXd>& X_normed) const {
    check_fitted("inverse_transform_inplace");
    check_dimension(norms_.size(), X_normed.cols(), "inverse_transform_inplace");

    #pragma omp parallel for schedule(static)
    for (std::size_t j = 0; j < static_cast<std::size_t>(X_normed.cols()); ++j) {
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

Eigen::VectorXd Normalizer::get_l2_conversion_factor(std::size_t n_samples) const {
    check_fitted("get_l2_conversion_factor");

    if (n_samples == 0) {
        throw std::invalid_argument(
            "get_l2_conversion_factor: n_samples must be > 0");
    }

    Eigen::VectorXd conversion(norms_.size());

    #pragma omp parallel for schedule(static)
    for (std::size_t j = 0; j < static_cast<std::size_t>(norms_.size()); ++j) {
        // Convert from normalization to L2 scale
        // Formula: β_L2 = β_norm * norm_j / sqrt(n)
        if (with_norm_ && norm_type_ == NormType::L2) {
            // Already L2 normalized, just adjust for sample size
            conversion[j] = norms_[j] / std::sqrt(static_cast<double>(n_samples));

        } else if (with_norm_ && norm_type_ == NormType::L1) {
            // L1 normalized - convert to L2 scale
            // This is an approximation; exact conversion depends on data distribution
            conversion[j] = norms_[j] / std::sqrt(static_cast<double>(n_samples));

        } else {
            // No normalization applied
            conversion[j] = 1.0 / std::sqrt(static_cast<double>(n_samples));
        }
    }

    return conversion;
}
