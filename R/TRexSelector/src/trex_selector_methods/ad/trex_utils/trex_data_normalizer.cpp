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
#include <trex_selector_methods/ad/trex_utils/trex_data_normalizer.hpp>

// std includes
#include <cmath>
#include <sstream>
#include <stdexcept>

// utils includes
#include <utils/fp_classify/fp_classify.hpp>
#include <utils/openmp/utils_openmp.hpp>

// =================================================================================

// Embedded into namespace trex::trex_selector_methods::ad::utils::data_normalizer
namespace trex::trex_selector_methods::ad::utils::data_normalizer {

// =================================================================================

namespace {

namespace fpc = trex::utils::fp_classify;

/** @brief Row index of the first non-finite entry of @p col, or -1. */
template <typename ColumnType>
Eigen::Index firstNonFinite(const ColumnType& col) {
    for (Eigen::Index i = 0; i < col.size(); ++i) {
        if (!fpc::isfinite(col(i))) { return i; }
    }
    return -1;
}

/** @brief Sample-SD divisor factor: sqrt(n-1) for ZSCORE (n > 1), else 1. */
double sdFactor(ScalingMode mode, Eigen::Index n) {
    return (mode == ScalingMode::ZSCORE && n > 1)
        ? std::sqrt(static_cast<double>(n - 1))
        : 1.0;
}

// Band of raw column L2 norms accepted silently under ScalingMode::NONE. The
// dummy blocks follow the unit-L2 standard, so a user-managed design should
// be unit-L2-commensurate; the band never trips on legitimately unit-L2
// data and always trips on z-scored (norm sqrt(n-1)) or raw-amplitude data.
constexpr double kNoneNormLower = 0.25;
constexpr double kNoneNormUpper = 4.0;

} // namespace

// =================================================================================

void centerAndL2NormalizeX(Eigen::Ref<Eigen::MatrixXd> X,
                           NormalizationParams& params,
                           double eps,
                           bool verbose,
                           ScalingMode mode,
                           int n_threads) {

    (void)verbose;  // rejection is an exception; the NONE warning is not gated

    const Eigen::Index p = X.cols();
    const Eigen::Index n = X.rows();
    const double sd_factor = sdFactor(mode, n);

    // Team of the column passes: the budget, or 1 below the work threshold
    [[maybe_unused]] const int nt = ::trex::utils::openmp::threads_for(
        n_threads, static_cast<double>(n) * static_cast<double>(p));

    // ---------------------------------------------------------------------
    // Pass 1 — validation on the untouched matrix (X is unmodified when an
    // exception leaves this function). The lowest failing column wins so the
    // message is deterministic regardless of thread scheduling.
    // ---------------------------------------------------------------------
    Eigen::Index bad_col = -1;      // first non-finite column
    Eigen::Index bad_row = -1;      // its first non-finite row
    Eigen::Index dead_col = -1;     // first dead column
    double       dead_norm = 0.0;   // its working L2 norm
    Eigen::Index stray = 0;         // NONE: columns off the unit-L2 band

    #pragma omp parallel num_threads(nt) if (nt > 1)
    {
        ::trex::utils::openmp::team_probe::record();
        Eigen::VectorXd centered(n);

        #pragma omp for schedule(static) reduction(+:stray)
        for (Eigen::Index j = 0; j < p; ++j) {
            const auto col = X.col(j);

            const Eigen::Index row = firstNonFinite(col);
            if (row >= 0) {
                #pragma omp critical
                {
                    if (bad_col < 0 || j < bad_col) { bad_col = j; bad_row = row; }
                }
                continue;
            }

            double working;  // the norm the dead test is keyed on
            if (mode == ScalingMode::NONE) {
                working = col.norm();
                if (working >= eps &&
                    (working < kNoneNormLower || working > kNoneNormUpper)) {
                    ++stray;
                }
            } else {
                centered = col.array() - col.mean();
                working = centered.norm() / sd_factor;
            }

            if (working < eps) {
                #pragma omp critical
                {
                    if (dead_col < 0 || j < dead_col) { dead_col = j; dead_norm = working; }
                }
            }
        }
    }

    if (bad_col >= 0) {
        std::ostringstream msg;
        msg << "TRex data normalizer: X(" << bad_row << ", " << bad_col
            << ") is not finite (NaN or Inf; 0-based indices). T-Rex needs a "
               "finite design matrix: clean or impute column " << bad_col
            << " before constructing the selector.";
        throw std::invalid_argument(msg.str());
    }
    if (dead_col >= 0) {
        std::ostringstream msg;
        if (mode == ScalingMode::NONE) {
            msg << "TRex data normalizer: column " << dead_col
                << " (0-based) of X has L2 norm " << dead_norm << " < eps " << eps
                << " under ScalingMode::NONE (an all-zero column cannot be "
                   "selected): remove it from X.";
        } else {
            msg << "TRex data normalizer: column " << dead_col
                << " (0-based) of X is constant (centered "
                << (mode == ScalingMode::ZSCORE ? "sample SD " : "L2 norm ")
                << dead_norm << " < eps " << eps
                << "). A constant predictor cannot be selected: remove it from "
                   "X (an intercept is handled by centering; for an intercept-"
                   "free dictionary with a constant atom use ScalingMode::NONE).";
        }
        throw std::invalid_argument(msg.str());
    }
    if (stray > 0) {
        TREX_WARN("ScalingMode::NONE: " << stray << " of " << p
                  << " design columns have L2 norm outside ["
                  << kNoneNormLower << ", " << kNoneNormUpper << "]. The dummy "
                  "blocks follow the unit-L2 standard, so a user-managed design "
                  "should be unit-L2-commensurate (for z-scored data use "
                  "ScalingMode::ZSCORE, otherwise rescale the columns).");
    }

    // ---------------------------------------------------------------------
    // Record the parameters; NONE stores the identity and leaves X as is.
    // ---------------------------------------------------------------------
    params.mode = mode;
    params.X_means.resize(p);
    params.X_l2norms.resize(p);

    if (mode == ScalingMode::NONE) {
        params.X_means.setZero();
        params.X_l2norms.setOnes();
        return;
    }

    // ---------------------------------------------------------------------
    // Pass 2 — the fused legacy transform: mean → center → norm → scale.
    // Pass 1 rejected every dead column, so the divisor is at least eps up
    // to rounding of the centered norm; no pinning to 1.0 any more.
    // ---------------------------------------------------------------------
    #pragma omp parallel for num_threads(nt) if (nt > 1)
    for (Eigen::Index j = 0; j < p; ++j) {
        ::trex::utils::openmp::team_probe::record();

        // Compute column mean j
        double mean_j = X.col(j).mean();
        params.X_means(j) = mean_j;

        // Center column j
        X.col(j).array() -= mean_j;

        // Compute the scale divisor of centered column j:
        //   L2     -> ||x_c||
        //   ZSCORE -> ||x_c|| / sqrt(n-1)  (sample SD)
        double scale_j = X.col(j).norm() / sd_factor;
        params.X_l2norms(j) = scale_j;

        // Normalize column j
        X.col(j).array() /= scale_j;
    }
}


void denormalizeX(Eigen::Ref<Eigen::MatrixXd> X,
                  const NormalizationParams& params,
                  int n_threads)
{
    if (params.mode == ScalingMode::NONE) { return; }  // identity parameters

    const Eigen::Index p = X.cols();
    [[maybe_unused]] const int nt = ::trex::utils::openmp::threads_for(
        n_threads, static_cast<double>(X.rows()) * static_cast<double>(p));
    #pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
    for (Eigen::Index j = 0; j < p; ++j) {
        ::trex::utils::openmp::team_probe::record();

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
                                 ScalingMode mode,
                                 int n_threads) {

    if (mode == ScalingMode::NONE) { return; }  // user-managed: untouched

    const Eigen::Index p = M.cols();
    const Eigen::Index n = M.rows();
    const double sd_factor = sdFactor(mode, n);

    // Team of the column loop: the budget, or 1 below the work threshold
    [[maybe_unused]] const int nt = ::trex::utils::openmp::threads_for(
        n_threads, static_cast<double>(n) * static_cast<double>(p));

    #pragma omp parallel for num_threads(nt) if (nt > 1)
    for (Eigen::Index j = 0; j < p; ++j) {
        ::trex::utils::openmp::team_probe::record();

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


void centerY(Eigen::VectorXd& y, NormalizationParams& params, ScalingMode mode) {
    const Eigen::Index row = firstNonFinite(y);
    if (row >= 0) {
        std::ostringstream msg;
        msg << "TRex data normalizer: y(" << row
            << ") is not finite (NaN or Inf; 0-based index). T-Rex needs a "
               "finite response vector.";
        throw std::invalid_argument(msg.str());
    }
    if (mode == ScalingMode::NONE) {
        params.y_mean = 0.0;
        return;
    }
    double y_mean = y.mean();
    params.y_mean = y_mean;
    y.array() -= y_mean;
}

void decenterY(Eigen::VectorXd& y, const NormalizationParams& params) {
    if (params.mode == ScalingMode::NONE) { return; }
    y.array() += params.y_mean;
}


// =================================================================================
} /* End of namespace trex::trex_selector_methods::ad::utils::data_normalizer */
