// =================================================================================
// trex_data_normalizer.cpp
// =================================================================================
/**
 * @file trex_data_normalizer.cpp
 *
 * @brief Implementation of external data normalization functions for T-Rex
 * Selector.
 */
// =================================================================================

// TRex includes
#include <utils/logging/logger.hpp>
#include <trex_selector_methods/trex_utils/trex_data_normalizer.hpp>

// std includes
#include <cmath>

// utils includes
#include <utils/openmp/utils_openmp.hpp>

// =================================================================================

// Embedded into namespace trex::trex_selector_methods::utils::data_normalizer
namespace trex::trex_selector_methods::utils::data_normalizer {

// =================================================================================

void centerAndL2NormalizeX(Eigen::Ref<Eigen::MatrixXd> X,
                           NormalizationParams& params,
                           double eps,
                           bool verbose,
                           ScalingMode mode) {

    const Eigen::Index p = X.cols();
    const Eigen::Index n = X.rows();

    // z-score divides by sample SD = ||x_c|| / sqrt(n-1); fall back to L2 if n<2.
    const double sd_factor =
        (mode == ScalingMode::ZSCORE && n > 1)
            ? std::sqrt(static_cast<double>(n - 1))
            : 1.0;

    // Allocate storage for normalization parameters
    params.X_means.resize(p);
    params.X_l2norms.resize(p);

    // Parallelize across columns if p is large
    #pragma omp parallel for if (p > 100)
    for (Eigen::Index j = 0; j < p; ++j) {

        // Compute column mean j
        double mean_j = X.col(j).mean();
        params.X_means(j) = mean_j;

        // Center column j
        X.col(j).array() -= mean_j;

        // Compute the scale divisor of centered column j:
        //   L2     -> ||x_c||
        //   ZSCORE -> ||x_c|| / sqrt(n-1)  (sample SD)
        double scale_j = X.col(j).norm() / sd_factor;

        // Handle near-zero norm case
        if (scale_j < eps) {
            if (verbose) {
                #pragma omp critical
                {
                    TREX_WARN( "Warning: Column " << j << " has near-zero scale ("
                              << scale_j
                              << "). Setting scale to 1.0 to avoid division by zero." );
                }
            }
            scale_j = 1.0;
        }
        params.X_l2norms(j) = scale_j;

        // Normalize column j
        X.col(j).array() /= scale_j;
    }
}


void denormalizeX(Eigen::Ref<Eigen::MatrixXd> X,
                  const NormalizationParams& params
                  )
{

    const Eigen::Index p = X.cols();
    #pragma omp parallel for schedule(static) if(p > 100)
    for (Eigen::Index j = 0; j < p; ++j) {

        auto col = X.col(j);

        // Scale back from unit L2 norm
        col.array() *= params.X_l2norms(j);

        // Add back the mean
        col.array() += params.X_means(j);
    }
}


void centerAndL2NormalizeMatrix(Eigen::Ref<Eigen::MatrixXd> M,
                                 double eps,
                                 bool verbose,
                                 ScalingMode mode) {

    const Eigen::Index p = M.cols();
    const Eigen::Index n = M.rows();

    const double sd_factor =
        (mode == ScalingMode::ZSCORE && n > 1)
            ? std::sqrt(static_cast<double>(n - 1))
            : 1.0;

    #pragma omp parallel for if (p > 100)
    for (Eigen::Index j = 0; j < p; ++j) {

        // Center column j
        M.col(j).array() -= M.col(j).mean();

        // Compute the scale divisor (L2 norm, or sample SD for ZSCORE)
        double scale_j = M.col(j).norm() / sd_factor;

        if (scale_j < eps) {
            if (verbose) {
                #pragma omp critical
                TREX_WARN( "Warning: Dummy column " << j
                          << " has near-zero scale. Skipping normalization." );
            }
            scale_j = 1.0;
        }

        // Normalize column j
        M.col(j).array() /= scale_j;
    }
}


void centerY(Eigen::VectorXd& y, NormalizationParams& params) {
    double y_mean = y.mean();
    params.y_mean = y_mean;
    y.array() -= y_mean;
}

void decenterY(Eigen::VectorXd& y, const NormalizationParams& params) {
    y.array() += params.y_mean;
}


// =================================================================================
} /* End of namespace trex::trex_selector_methods::utils::data_normalizer */
