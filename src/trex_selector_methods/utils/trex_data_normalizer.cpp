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
#include <trex_selector_methods/utils/trex_data_normalizer.hpp>

// utils includes
#include <utils/openmp/utils_openmp.hpp>

// =================================================================================

// Embedded into namespace trex::trex_selector_methods::utils::data_normalizer
namespace trex::trex_selector_methods::utils::data_normalizer {

// =================================================================================

void centerAndL2NormalizeX(Eigen::Ref<Eigen::MatrixXd> X,
                           NormalizationParams& params,
                           double eps,
                           bool verbose) {

    const Eigen::Index p = X.cols();

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

        // Compute L2 norm of centered column j
        double l2norm_j = X.col(j).norm();

        // Handle near-zero norm case
        if (l2norm_j < eps) {
            if (verbose) {
                #pragma omp critical
                {
                    std::cerr << "Warning: Column " << j << " has near-zero L2 norm ("
                              << l2norm_j
                              << "). Setting norm to 1.0 to avoid division by zero.\n";
                }
            }
            l2norm_j = 1.0;
        }
        params.X_l2norms(j) = l2norm_j;

        // Normalize column j
        X.col(j).array() /= l2norm_j;
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
                                 bool verbose) {

    const Eigen::Index p = M.cols();

    #pragma omp parallel for if (p > 100)
    for (Eigen::Index j = 0; j < p; ++j) {

        // Center column j
        M.col(j).array() -= M.col(j).mean();

        // Compute L2 norm
        double l2norm_j = M.col(j).norm();

        if (l2norm_j < eps) {
            if (verbose) {
                #pragma omp critical
                std::cerr << "Warning: Dummy column " << j
                          << " has near-zero L2 norm. Skipping normalization.\n";
            }
            l2norm_j = 1.0;
        }

        // Normalize column j
        M.col(j).array() /= l2norm_j;
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
