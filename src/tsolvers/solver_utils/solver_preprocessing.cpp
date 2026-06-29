// =============================================================================
// solver_preprocessing.cpp
// =============================================================================
/**
 * @brief Implementation (.cpp) file for solver preprocessing utilities.
 */
// =============================================================================

// std includes
#include <cmath>
#include <algorithm>

// tsolvers includes
#include <tsolvers/solver_utils/solver_preprocessing.hpp>

// utils includes
#include <utils/openmp/utils_openmp.hpp>

// =============================================================================

// Embedded into trex::tsolvers::solver_utils::preprocessing namespace
namespace trex::tsolvers::solver_utils::preprocessing {

// ============================================================================

// Implementation of preprocessing functions

void preprocessMatStatistics(
    Eigen::Ref<Eigen::MatrixXd> X,
    bool center,
    bool normalize,
    double eps,
    Eigen::VectorXd& meansx,
    Eigen::VectorXd& normsx,
    std::vector<std::size_t>& dropped_indices,
    ScalingMode mode
) {
    Eigen::Index nrows = X.rows();
    Eigen::Index pcols = X.cols();

    if (center) meansx.resize(pcols);
    if (normalize) normsx.resize(pcols);

    // For z-score scaling the column divisor is the sample standard deviation
    // (= L2-norm / sqrt(n - 1)); this factor is identical for every column, so
    // it is computed once. For unit-L2 scaling the factor is 1.
    const double sd_factor =
        (mode == ScalingMode::ZSCORE && nrows > 1)
            ? std::sqrt(static_cast<double>(nrows - 1))
            : 1.0;

    // Single parallel region to minimize thread-spawning overhead
    #pragma omp parallel
    {
        // Enforce Eigen uses only 1 thread inside this OpenMP region
        Eigen::setNbThreads(1);

        // --------------------------------------------------------------------
        // Phase 1: Centering
        // --------------------------------------------------------------------
        if (center) {
            #pragma omp for schedule(static)
            for (Eigen::Index j = 0; j < pcols; ++j) {
                meansx(j) = X.col(j).mean();
            }

            #pragma omp for schedule(static)
            for (Eigen::Index j = 0; j < pcols; ++j) {
                X.col(j).array() -= meansx(j);
            }
        }

        // --------------------------------------------------------------------
        // Phase 2: Normalization
        // --------------------------------------------------------------------
        if (normalize) {
            // Thread-local buffer to avoid critical section bottleneck
            std::vector<std::size_t> thread_dropped;

            #pragma omp for schedule(static) nowait
            for (Eigen::Index j = 0; j < pcols; ++j) {
                double norm_j = X.col(j).norm();
                double min_norm = eps * std::sqrt(static_cast<double>(nrows));

                // The drop test is on the raw L2-norm (variance proxy) and is
                // independent of the scaling mode. The stored divisor, however,
                // is expressed in the selected mode (norm for L2, SD for ZSCORE).
                double used_norm;
                if (norm_j < min_norm) {
                    used_norm = min_norm; // Prevent division by zero

                    // Only add to dropped if it isn't already there
                    // (useful if called multiple times or pre-populated)
                    if (std::ranges::find(dropped_indices, j) == dropped_indices.end()) {
                        thread_dropped.push_back(static_cast<std::size_t>(j));
                    }
                } else {
                    used_norm = norm_j;
                }

                normsx(j) = used_norm / sd_factor;

                // Scale column
                X.col(j) /= normsx(j);
            }

            // Merge thread-local drops safely
            if (!thread_dropped.empty()) {
                #pragma omp critical
                {
                    dropped_indices.insert(
                        dropped_indices.end(),
                        thread_dropped.begin(),
                        thread_dropped.end()
                    );
                }
            }
        }
    } // End of #pragma omp parallel

    // Ensure dropped indices are sorted for consistent downstream logic (C++20)
    if (normalize && !dropped_indices.empty()) {
        std::ranges::sort(dropped_indices);
    }
}


void restoreMatStatistics(
    Eigen::Ref<Eigen::MatrixXd> X,
    bool centered,
    const Eigen::VectorXd& meansx,
    bool normalized,
    const Eigen::VectorXd& normsx
) {
    Eigen::Index pcols = X.cols();

    // Un-scale
    if (normalized) {
        #pragma omp parallel for schedule(static)
        for (Eigen::Index j = 0; j < pcols; ++j) {
            X.col(j) *= normsx(j);
        }
    }

    // Un-center
    if (centered) {
        #pragma omp parallel for schedule(static)
        for (Eigen::Index j = 0; j < pcols; ++j) {
            X.col(j).array() += meansx(j);
        }
    }
}


void centerResponse(Eigen::Ref<Eigen::VectorXd> y, double& mu_y) {
    mu_y = y.mean();
    y.array() -= mu_y;
}


void restoreResponse(Eigen::Ref<Eigen::VectorXd> y, double mu_y) {
    y.array() += mu_y;
}


// ============================================================================
} /* End of namespace trex::tsolvers::solver_utils::preprocessing */
