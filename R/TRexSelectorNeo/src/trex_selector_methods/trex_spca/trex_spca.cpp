// =============================================================================
// trex_spca.cpp
// ==============================================================================
/**
 * @file trex_spca.cpp
 *
 * @brief Implementation of the T-Rex Sparse PCA selector.
 */
// ==============================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

// trex_selector_methods includes
#include <trex_selector_methods/trex_spca/trex_spca.hpp>
#include <trex_selector_methods/trex_gvs/trex_gvs.hpp>

// ml_methods includes
#include <ml_methods/pca/pca.hpp>
#include <ml_methods/ridge_regression/ridge.hpp>
#include <ml_methods/scaler_methods/lp_norm_scaler.hpp>

// ==============================================================================

namespace trex::trex_selector_methods::trex_spca {

// Namespace aliases
namespace pca        = trex::ml_methods::pca;
namespace ridge      = trex::ml_methods::ridge;
namespace tg         = trex::trex_selector_methods::trex_gvs;
namespace tc         = trex::trex_selector_methods::trex_core;
namespace sd         = trex::trex_selector_methods::utils::solver_dispatch;
// std_scaler alias inherited from header via using-declaration of the namespace block


// ==============================================================================
// Constructor
// ==============================================================================

TRexSPCA::TRexSPCA(Eigen::Map<Eigen::MatrixXd>& X,
                   Eigen::Index M,
                   double tFDR,
                   TRexSPCAControlParameter spca_ctrl,
                   int seed,
                   bool verbose)
    : X_(&X)
    , M_(M)
    , tFDR_(tFDR)
    , spca_ctrl_(std::move(spca_ctrl))
    , seed_(seed)
    , verbose_(verbose)
    , scaler_(std_scaler::LpNormScaler::NormType::L2,
              /*center=*/true,
              /*scale=*/false)   // center-only: ordinary PCA must be a covariance PCA
{
    if (M_ > std::min(X_->rows(), X_->cols())) {
        throw std::invalid_argument(
            "TRexSPCA: M must not exceed min(n, p).");
    }

    // Fail fast on strategies the per-PC GVS runs will reject anyway:
    // without this pre-check the error only surfaces at PC 0 INSIDE
    // select(), after the (potentially expensive) ordinary PCA has already
    // been computed, and with a message naming TRexGVSSelector.
    const auto strat = spca_ctrl_.gvs_ctrl.trex_ctrl.lloop_strategy;
    if (strat == tc::LLoopStrategy::PERMUTATION ||
        strat == tc::LLoopStrategy::PERMUTATION_SEEDED) {
        throw std::invalid_argument(
            "TRexSPCA: the PERMUTATION-family L-loop strategies are not "
            "supported (the per-PC TRexGVSSelector rejects them: row "
            "permutations of one base cannot carry the per-cluster MVN "
            "dummy convention).");
    }
}


// ==============================================================================
// Destructor
// ==============================================================================

TRexSPCA::~TRexSPCA() {
    // Safety net: if select() was interrupted by an exception after centering X
    // but before restoring it, restore X here so the caller's data is not corrupted.
    if (X_is_centered_) {
        scaler_.inverse_transform_inplace(*X_);  // X += means (center-only scaler)
        X_is_centered_ = false;
    }
}


// ==============================================================================
// TRexSPCA::select
// ==============================================================================

TRexSPCAResult TRexSPCA::select() {

    const Eigen::Index n = X_->rows();
    const Eigen::Index p = X_->cols();

    result_ = TRexSPCAResult{};
    result_.V = Eigen::MatrixXd::Zero(p, M_);
    result_.Z = Eigen::MatrixXd::Zero(n, M_);
    result_.active_sets.resize(static_cast<std::size_t>(M_));

    // 0. Center X column-wise (mean subtraction only, no L2 scaling).
    //    Covariance PCA: column scales carry the factor signal, so they must
    //    NOT be normalized. scaler_ is a member so the destructor can restore
    //    X on exception.
    scaler_.fit(*X_);
    scaler_.transform_inplace(*X_);  // X -= means
    X_is_centered_ = true;

    // 1. Compute ordinary PCA on the centered X (no further preprocessing).
    pca::PCAResult ord_pca = pca::PCA(*X_, M_, /*center=*/false, /*normalize=*/false).fit();

    // 2. Per-PC loop.
    for (Eigen::Index m = 0; m < M_; ++m) {

        if (verbose_) {
            TREX_INFO("[TRexSPCA] Processing PC " << (m + 1) << " / " << M_);
        }

        // 3. Run TRexGVSSelector on z_m ~ X.
        //    TRexGVSSelector copies y internally, so z_m is not mutated.
        //    Seed: per-PC variation when seed_ >= 0; hardware entropy when seed_ < 0.
        Eigen::VectorXd z_m = ord_pca.Z.col(m);
        const int seed_pc = (seed_ >= 0)
            ? static_cast<int>(trex::utils::datageneration::dummygen::mix_seed(
                static_cast<std::uint32_t>(seed_), m) & 0x7FFFFFFFu)
            : -1;

        tg::TRexGVSControlParameter gvs_ctrl = spca_ctrl_.gvs_ctrl;

        // EN is always forced (IEN is not used by TRexSPCA). The solver
        // variant is user-selectable via spca_ctrl_.en_solver
        // (TENET / TENET_AUG / TCENET — all equivalent for lambda2 > 0);
        // solver_type is set to the matching enumerator so a conflicting
        // value in the forwarded gvs_ctrl can never reach GVS's derivation
        // check.
        gvs_ctrl.gvs_type              = tg::GVSType::EN;
        gvs_ctrl.en_solver             = spca_ctrl_.en_solver;
        gvs_ctrl.trex_ctrl.solver_type = tg::toSolverType(spca_ctrl_.en_solver);

        std::vector<Eigen::Index> active_set;
        {
            Eigen::Map<Eigen::VectorXd> z_map(z_m.data(), n);
            tg::TRexGVSSelector gvs(*X_, z_map, tFDR_,
                                    gvs_ctrl,
                                    seed_pc, /*verbose=*/false);
            gvs.select();

            // X is column-normalized here; extract indices before restoration.
            const auto& raw_indices = gvs.getSelectedIndices();
            active_set.reserve(raw_indices.size());
            for (std::size_t idx : raw_indices) {
                active_set.push_back(static_cast<Eigen::Index>(idx));
            }
        } // TRexGVSSelector destructor restores X to its centered state here.

        if (active_set.empty()) {
            continue;
        }

        result_.active_sets[static_cast<std::size_t>(m)] = active_set;

        // 4. Assemble sparse loading vector.
        if (spca_ctrl_.mode == SPCAMode::ActiveSet) {
            assembleActiveSetLoadings(*X_, z_m, active_set, result_.V.col(m));
        } else {
            assembleThresholdedLoadings(
                ord_pca.V.col(m),
                static_cast<Eigen::Index>(active_set.size()),
                result_.V.col(m));
        }

        // 5. Compute sparse PC scores: z_hat_m = X * v_hat_m, iterating over the
        //    nonzero support of v_hat_m. In ActiveSet mode that support equals
        //    A_m; in Thresholded mode it is the top-|A_m| ordinary loadings,
        //    which generally differs from A_m.
        for (Eigen::Index j = 0; j < p; ++j) {
            const double v_jm = result_.V(j, m);
            if (v_jm != 0.0) {
                result_.Z.col(m) += X_->col(j) * v_jm;
            }
        }
    }

    // 6. Adjust explained variance via QR decomposition of the score matrix.
    adjustExplainedVariance(result_, *X_);

    // 7. Restore X to its original (uncentered) state.
    scaler_.inverse_transform_inplace(*X_);  // X += means (center-only scaler)
    X_is_centered_ = false;

    return result_;
}


// ==============================================================================
// TRexSPCA::assembleActiveSetLoadings
// ==============================================================================

void TRexSPCA::assembleActiveSetLoadings(
    const Eigen::Ref<const Eigen::MatrixXd>& X_ref,
    const Eigen::Ref<const Eigen::VectorXd>& z_m,
    const std::vector<Eigen::Index>& active_set,
    Eigen::Ref<Eigen::VectorXd> v_out) const {

    const Eigen::Index k = static_cast<Eigen::Index>(active_set.size());
    const Eigen::Index n = X_ref.rows();

    // Extract a contiguous dense subset of the active columns.
    Eigen::MatrixXd X_active(n, k);
    for (Eigen::Index i = 0; i < k; ++i) {
        X_active.col(i) = X_ref.col(active_set[static_cast<std::size_t>(i)]);
    }

    // Ridge regression on the active subset; normalize and scatter back.
    Eigen::VectorXd beta_ridge = ridge::RidgeSolver::solve(
        X_active, z_m, spca_ctrl_.lambda2_ridge_loadings
    );
    beta_ridge.normalize();

    for (Eigen::Index i = 0; i < k; ++i) {
        v_out(active_set[static_cast<std::size_t>(i)]) = beta_ridge(i);
    }
}


// ==============================================================================
// TRexSPCA::assembleThresholdedLoadings
// ==============================================================================

void TRexSPCA::assembleThresholdedLoadings(
    const Eigen::Ref<const Eigen::VectorXd>& ord_v,
    Eigen::Index active_size,
    Eigen::Ref<Eigen::VectorXd> v_out) const {

    const Eigen::Index p = ord_v.size();

    // Copy absolute values to find the threshold without modifying ord_v.
    std::vector<double> abs_vals(static_cast<std::size_t>(p));
    for (Eigen::Index i = 0; i < p; ++i) {
        abs_vals[static_cast<std::size_t>(i)] = std::abs(ord_v(i));
    }

    // O(p) nth_element to find the cutoff for the top |A| entries.
    auto nth_it = abs_vals.begin() + (p - active_size);
    std::nth_element(abs_vals.begin(), nth_it, abs_vals.end());
    const double threshold = *nth_it;

    for (Eigen::Index i = 0; i < p; ++i) {
        if (std::abs(ord_v(i)) >= threshold) {
            v_out(i) = ord_v(i);
        }
    }
    v_out.normalize();
}


// ==============================================================================
// TRexSPCA::adjustExplainedVariance
// ==============================================================================

void TRexSPCA::adjustExplainedVariance(
    TRexSPCAResult& result,
    const Eigen::Ref<const Eigen::MatrixXd>& X_ref) const {

    // Sparse PCs are not strictly orthogonal; adjust by factoring Z = QR and
    // using squared diagonal entries of R as marginal variances.
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(result.Z);
    const Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

    result.adjusted_ev   = Eigen::VectorXd::Zero(M_);
    result.cumulative_ev = Eigen::VectorXd::Zero(M_);

    // Total variance of centered X: trace(X^T X) / (n - 1).
    const double total_variance =
        X_ref.squaredNorm() / static_cast<double>(X_ref.rows() - 1);
    double cum_ev = 0.0;

    for (Eigen::Index k = 0; k < M_; ++k) {
        const double r_kk = R(k, k);
        const double marginal_ev = (r_kk * r_kk) / static_cast<double>(X_ref.rows() - 1);
        result.adjusted_ev(k) = marginal_ev;

        cum_ev += marginal_ev;
        result.cumulative_ev(k) = cum_ev / total_variance;
    }
}


// ==============================================================================
} /* End of namespace trex::trex_selector_methods::trex_spca */
// ==============================================================================
