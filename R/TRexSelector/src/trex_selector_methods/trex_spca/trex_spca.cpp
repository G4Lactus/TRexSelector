// =============================================================================
// trex_spca.cpp
// ==============================================================================
/**
 * @file trex_spca.cpp
 *
 * @brief Implementation of the T-Rex Sparse PCA and Thresholded PCA selector.
 */
// ==============================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

// trex_selector_methods includes
#include <trex_selector_methods/trex_spca/trex_spca.hpp>
#include <trex_selector_methods/trex_core/trex.hpp>

// ml_methods includes
#include <ml_methods/pca/pca.hpp>
#include <ml_methods/ridge_regression/ridge.hpp>
#include <ml_methods/standardization/lp_norm_scaler.hpp>

// ==============================================================================

namespace trex {
namespace trex_selector_methods {
namespace trex_spca {

// Namespace aliases
namespace pca        = trex::ml_methods::pca;
namespace ridge      = trex::ml_methods::ridge;
namespace tc         = trex::trex_selector_methods::trex_core;
namespace std_scaler = trex::ml_methods::standardization;


// ==============================================================================
// TRexSPCA::select
// ==============================================================================

TRexSPCAResult TRexSPCA::select(Eigen::Map<Eigen::MatrixXd>& X,
                                Eigen::Index M,
                                double tFDR,
                                TRexSPCAControlParameter spca_ctrl) {

    Eigen::Index n = X.rows();
    Eigen::Index p = X.cols();

    if (M > std::min(n, p)) {
        throw std::invalid_argument("TRexSPCA::select: M must not exceed min(n, p).");
    }

    TRexSPCAResult result;
    result.V = Eigen::MatrixXd::Zero(p, M);
    result.Z = Eigen::MatrixXd::Zero(n, M);
    result.active_sets.resize(M);

    // 0. Center X column-wise (centering only, no L2 scaling).
    //    LpNormScaler stores column means and restores X exactly at the end of select().
    std_scaler::LpNormScaler scaler(
        std_scaler::LpNormScaler::NormType::L2, /*with_mean=*/true, /*with_norm=*/false);
    scaler.fit(X);
    scaler.transform_inplace(X);  // X -= means

    // 1. Compute ordinary PCA to obtain the PC response vectors (centered X, read-only).
    pca::PCAResult ord_pca = pca::PCA(false).fit(X, M);

    // 2. Iterate over each PC.
    for (Eigen::Index m = 0; m < M; ++m) {

        // 3. Run T-Rex selector on z_m ~ X.
        //    TRexSelector requires Eigen::Map (non-const, mutable).
        //    TRexSelector copies y internally, so z_m is not mutated.
        Eigen::VectorXd z_m = ord_pca.Z.col(m);

        std::vector<Eigen::Index> active_set;
        {
            Eigen::Map<Eigen::VectorXd> z_map(z_m.data(), n);
            tc::TRexSelector trex(X, z_map, tFDR,
                                  spca_ctrl.trex_ctrl,
                                  spca_ctrl.seed, false);
            trex.select();

            // X is normalized here
            // extract indices before restoration.
            const auto& raw_indices = trex.getSelectedIndices();
            active_set.reserve(raw_indices.size());
            for (std::size_t idx : raw_indices) {
                active_set.push_back(static_cast<Eigen::Index>(idx));
            }
        } // TRexSelector destructor restores X to its centered state here.

        if (active_set.empty()) {
            // active set empty -> component is effectively null.
            continue;
        }

        result.active_sets[m] = active_set;

        // 4. Assemble the sparse loading vector.
        if (spca_ctrl.mode == SPCAMode::ActiveSet) {
            assemble_active_set_loadings(
                X, z_m, active_set, spca_ctrl.lambda2, result.V.col(m)
            );
        } else {
            assemble_thresholded_loadings(
                ord_pca.V.col(m),
                static_cast<Eigen::Index>(active_set.size()),
                result.V.col(m)
            );
        }

        // 5. Compute sparse PC scores over the active set:
        //      z_hat_m = X_{A_m} * v_hat_m.
        for (Eigen::Index idx : active_set) {
            result.Z.col(m) += X.col(idx) * result.V(idx, m);
        }
    }

    // 6. Compute adjusted explained variance over the fully-assembled score matrix.
    //    Sparse PCs are not strictly orthogonal thus we adjust by factoring Z = QR
    //    and using the squared diagonal entries of R as marginal variances.
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(result.Z);
    Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

    result.adjusted_ev   = Eigen::VectorXd::Zero(M);
    result.cumulative_ev = Eigen::VectorXd::Zero(M);

    // Total variance of the centered data: trace(X^T X) / (n - 1).
    // squaredNorm() efficiently sums all squared elements since X is centered.
    double total_variance = X.squaredNorm() / static_cast<double>(n - 1);
    double cum_ev = 0.0;

    for (Eigen::Index k = 0; k < M; ++k) {
        double r_kk = R(k, k);
        double marginal_ev = (r_kk * r_kk) / static_cast<double>(n - 1);
        result.adjusted_ev(k) = marginal_ev;

        cum_ev += marginal_ev;
        result.cumulative_ev(k) = cum_ev / total_variance;
    }

    // Restore X to its original (uncentered) state.
    scaler.inverse_transform_inplace(X);  // X += means

    return result;
}


// ==============================================================================
// TRexSPCA::assemble_active_set_loadings
// ==============================================================================

void TRexSPCA::assemble_active_set_loadings(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& z_m,
    const std::vector<Eigen::Index>& active_set,
    double lambda2,
    Eigen::Ref<Eigen::VectorXd> v_out) {

    Eigen::Index k = static_cast<Eigen::Index>(active_set.size());
    Eigen::Index n = X.rows();

    // Extract a contiguous dense subset of the active columns.
    Eigen::MatrixXd X_active(n, k);
    for (Eigen::Index i = 0; i < k; ++i) {
        X_active.col(i) = X.col(active_set[i]);
    }

    // Solve Ridge regression on the active subset.
    Eigen::VectorXd beta_ridge = ridge::RidgeSolver::solve(X_active, z_m, lambda2);

    // Normalize and scatter back into the full p-dimensional loading vector.
    beta_ridge.normalize();
    for (Eigen::Index i = 0; i < k; ++i) {
        v_out(active_set[i]) = beta_ridge(i);
    }
}


// ==============================================================================
// TRexSPCA::assemble_thresholded_loadings
// ==============================================================================

void TRexSPCA::assemble_thresholded_loadings(
    const Eigen::Ref<const Eigen::VectorXd>& ord_v,
    Eigen::Index active_size,
    Eigen::Ref<Eigen::VectorXd> v_out) {

    Eigen::Index p = ord_v.size();

    // Copy absolute values; avoids modifying the original loadings.
    std::vector<double> abs_vals(static_cast<std::size_t>(p));
    for (Eigen::Index i = 0; i < p; ++i) {
        abs_vals[static_cast<std::size_t>(i)] = std::abs(ord_v(i));
    }

    // O(p) selection: find the threshold that retains the top |A| elements.
    auto nth_it = abs_vals.begin() + (p - active_size);
    std::nth_element(abs_vals.begin(), nth_it, abs_vals.end());
    double threshold = *nth_it;

    // Apply threshold and copy values at or above it.
    for (Eigen::Index i = 0; i < p; ++i) {
        if (std::abs(ord_v(i)) >= threshold) {
            v_out(i) = ord_v(i);
        }
    }

    // Normalize the truncated loading vector to unit length.
    v_out.normalize();
}


// ==============================================================================
} /* namespace trex_spca */
} /* namespace trex_selector_methods */
} /* namespace trex */
// ==============================================================================
