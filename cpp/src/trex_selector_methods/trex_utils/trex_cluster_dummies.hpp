// ===================================================================================
// trex_cluster_dummies.hpp
// ===================================================================================
#ifndef TREX_SELECTOR_METHODS_UTILS_TREX_CLUSTER_DUMMIES_HPP
#define TREX_SELECTOR_METHODS_UTILS_TREX_CLUSTER_DUMMIES_HPP
// ===================================================================================
/**
 * @file trex_cluster_dummies.hpp
 *
 * @brief Shared cluster-aware dummy machinery for the T-Rex selectors.
 *
 * @details
 *  Single home of the cluster-MVN dummy convention used by `TRexGVSSelector`
 *  and (via the DummyGenerator cluster mode) `TRexTikhonovSelector`:
 *  variables are partitioned into clusters, the per-cluster covariance of
 *  the working-scale X is Cholesky-factorized once, and dummy layers are
 *  drawn as per-cluster colored MVN blocks. Keeping the numerics-critical
 *  pieces here guarantees all consumers stay on the IDENTICAL convention:
 *
 *   - `makeLayerStream()`        : the shared mt19937 seeding scheme
 *                                  (full-state seed_seq over base seed, k,
 *                                  tag) — one stream recipe for GVS and the
 *                                  DummyGenerator, so equal (seed, k, tag)
 *                                  means bit-identical draws everywhere.
 *   - `buildClusterCholeskys()`  : Sigma_m = X_m^T X_m / (n - 1) + 1e-10 I,
 *                                  lower Cholesky factor per cluster. The
 *                                  estimator auto-adapts to the column
 *                                  scaling of X (L2: diag = 1/(n-1); ZSCORE:
 *                                  diag = 1), so sampled dummies share the
 *                                  scale of X without any renormalization.
 *   - `drawClusterDummyLayer()`  : one (n x p) layer of MVN draws, columns
 *                                  scattered back to the cluster positions.
 *                                  Draw order (clusters ascending; Z filled
 *                                  column-by-column) is part of the
 *                                  reproducibility contract — do not reorder.
 *
 *  This header is deliberately Eigen-only (no clustering-backend includes)
 *  so the DummyGenerator can include it without dragging the agglomerative
 *  machinery into the base T-Rex translation units; the HAC-dependent
 *  cluster discovery lives in trex_cluster_hac.hpp.
 *
 *  Column post-processing is the CALLER's choice and is deliberately NOT
 *  done here: per the FIX (2026-07-08) convention, dummies enter the solvers
 *  center-only with their REALIZED column norms (the chi fluctuation of the
 *  MVN draw is part of the dummy null distribution; equalizing the norms
 *  made dummies systematically less competitive and inflated the realized
 *  FDR — see TRexGVSSelector::prepareDummiesForLStep).
 */
// ===================================================================================

// std includes
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// ===================================================================================

// Embedded into namespace trex::trex_selector_methods::utils::cluster_dummies
namespace trex::trex_selector_methods::utils::cluster_dummies {


// ===================================================================================
// Shared per-(k, tag) layer stream
// ===================================================================================

/**
 * @brief Construct the shared mt19937 stream for cluster-dummy draws.
 *
 * @details Seeds the FULL mt19937 state via std::seed_seq. Seeding mt19937
 *  from a single 32-bit value (e.g. a scalar mix_seed result) leaves nearby
 *  seeds able to emit correlated initial output — fatal for T-Rex, whose
 *  FDR calibration assumes the K dummy realisations are mutually
 *  independent and independent of X. seed_seq scrambles the supplied
 *  entropy words across the entire 624-word generator state, so every
 *  (experiment k, tag) pair draws a decorrelated dummy stream.
 *
 *  Tag convention (mirrors TRexGVSSelector): the L-loop iteration LL for
 *  fresh-per-iteration strategies (STANDARD; SKIPL passes L_max), the
 *  1-based layer index for prefix-stable per-layer strategies (HCONCAT
 *  appends layer li from the stream tagged li + 1, which at iteration LL
 *  is exactly the stream tagged LL).
 *
 * @param base_seed 64-bit base seed (user seed, or entropy when seed < 0).
 * @param k         Experiment index.
 * @param tag       Stream tag (see convention above).
 *
 * @return Fully seeded mt19937.
 */
inline std::mt19937 makeLayerStream(std::uint64_t base_seed,
                                    std::size_t k,
                                    std::size_t tag)
{
    std::seed_seq seq{
        static_cast<std::uint32_t>(base_seed & 0xFFFFFFFFu),
        static_cast<std::uint32_t>(base_seed >> 32),
        static_cast<std::uint32_t>(k),
        static_cast<std::uint32_t>(tag),
        0x9E3779B9u  // golden-ratio constant for extra avalanche mixing
    };
    return std::mt19937(seq);
}


// ===================================================================================
// Prior-group validation / cluster construction
// ===================================================================================

/**
 * @brief Validate a prior-groups vector (length p, 0-based, contiguous IDs).
 *
 * @param prior_groups 0-based cluster ID per variable; may be empty (no-op).
 * @param p            Number of variables.
 * @param class_tag    Prefix for exception messages (e.g. "TRexTikhonovSelector").
 *
 * @throws std::invalid_argument on length mismatch, negative IDs, or a
 *         missing ID in [0, max_id].
 */
inline void validatePriorGroups(const std::vector<Eigen::Index>& prior_groups,
                                std::size_t p,
                                const std::string& class_tag)
{
    if (prior_groups.empty()) return;

    if (prior_groups.size() != p) {
        throw std::invalid_argument(
            class_tag + ": prior_groups length must equal p = " +
            std::to_string(p) +
            ". Got: " + std::to_string(prior_groups.size()));
    }

    Eigen::Index max_id = -1;
    for (auto id : prior_groups) {
        if (id < 0) {
            throw std::invalid_argument(
                class_tag + ": prior_groups must be 0-based "
                "non-negative integers.");
        }
        if (id > max_id) max_id = id;
    }

    std::vector<bool> seen(static_cast<std::size_t>(max_id + 1), false);
    for (auto id : prior_groups) {
        seen[static_cast<std::size_t>(id)] = true;
    }
    for (std::size_t m = 0; m < seen.size(); ++m) {
        if (!seen[m]) {
            throw std::invalid_argument(
                class_tag + ": prior_groups IDs are non-contiguous. "
                "Cluster ID " + std::to_string(m) + " is missing.");
        }
    }
}


/**
 * @brief Build the per-cluster column-index lists from validated prior groups.
 *
 * @param prior_groups 0-based cluster ID per variable (validated, non-empty).
 * @param p            Number of variables.
 *
 * @return clusters_list of length M = max_id + 1.
 */
inline std::vector<std::vector<Eigen::Index>> clustersFromPriorGroups(
    const std::vector<Eigen::Index>& prior_groups,
    std::size_t p)
{
    Eigen::Index max_id = -1;
    for (auto id : prior_groups) {
        if (id > max_id) max_id = id;
    }
    std::vector<std::vector<Eigen::Index>> clusters(
        static_cast<std::size_t>(max_id + 1));
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(p); ++j) {
        clusters[static_cast<std::size_t>(
            prior_groups[static_cast<std::size_t>(j)])].push_back(j);
    }
    return clusters;
}


// ===================================================================================
// Per-cluster covariance Cholesky factors
// ===================================================================================

/**
 * @brief Compute the per-cluster lower Cholesky factors L_m of
 *        Sigma_m = X_m^T X_m / (n - 1) + 1e-10 I.
 *
 * @details The estimator auto-adapts to the column scaling of X:
 *   - ScalingMode::L2     -> ||x_j|| = 1,         diag(Sigma_m) = 1/(n-1)
 *   - ScalingMode::ZSCORE -> ||x_j|| = sqrt(n-1), diag(Sigma_m) = 1
 *  so MVN draws colored with L_m share the scale of X without any
 *  per-column renormalization.
 *
 * @param X             Working design matrix (already centered + scaled).
 * @param clusters_list Per-cluster column indices.
 * @param n             Number of observations.
 * @param class_tag     Prefix for the failure exception message.
 *
 * @return One lower-triangular factor per cluster (length M).
 *
 * @throws std::runtime_error if a Cholesky factorization fails.
 */
inline std::vector<Eigen::MatrixXd> buildClusterCholeskys(
    const Eigen::Map<Eigen::MatrixXd>& X,
    const std::vector<std::vector<Eigen::Index>>& clusters_list,
    std::size_t n,
    const std::string& class_tag)
{
    const std::size_t M = clusters_list.size();
    const double inv_nm1 =
        1.0 / static_cast<double>(static_cast<Eigen::Index>(n) - 1);

    std::vector<Eigen::MatrixXd> chol_list(M);

    for (std::size_t m = 0; m < M; ++m) {

        const auto& cols = clusters_list[m];
        const auto p_m = static_cast<Eigen::Index>(cols.size());

        Eigen::MatrixXd Sigma_m(p_m, p_m);
        for (Eigen::Index i = 0; i < p_m; ++i) {
            for (Eigen::Index j = i; j < p_m; ++j) {
                const double v = inv_nm1 *
                                 X.col(cols[static_cast<std::size_t>(i)])
                                 .dot(X.col(cols[static_cast<std::size_t>(j)]));
                Sigma_m(i, j) = v;
                Sigma_m(j, i) = v;
            }
            Sigma_m(i, i) += 1e-10;
        }

        Eigen::LLT<Eigen::MatrixXd> llt(Sigma_m);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error(
                class_tag + ": Cholesky factorization failed for cluster " +
                std::to_string(m) + ".");
        }
        chol_list[m] = llt.matrixL();
    }

    return chol_list;
}


// ===================================================================================
// Cluster-aware MVN dummy layer draw
// ===================================================================================

/**
 * @brief Draw one cluster-aware dummy layer (n x p).
 *
 * @details Per cluster m: draw Z ~ N(0, I) of shape (n x p_m), color with
 *  block = Z * L_m^T (rows ~ N(0, Sigma_m)), scatter into the cluster's
 *  column positions. The draw order — clusters ascending, Z filled column
 *  by column — is part of the reproducibility contract shared by
 *  TRexGVSSelector and TRexTikhonovSelector; do not reorder.
 *
 * @param n                   Number of rows.
 * @param p                   Number of columns of the layer.
 * @param clusters_list       Per-cluster column indices.
 * @param cholesky_lower_list Per-cluster lower Cholesky factors L_m.
 * @param rng                 Mersenne-Twister RNG, advanced in-place.
 *
 * @return One (n x p) layer of MVN draws (NOT centered; caller's choice).
 */
inline Eigen::MatrixXd drawClusterDummyLayer(
    std::size_t n,
    std::size_t p,
    const std::vector<std::vector<Eigen::Index>>& clusters_list,
    const std::vector<Eigen::MatrixXd>& cholesky_lower_list,
    std::mt19937& rng)
{
    const auto n_rows = static_cast<Eigen::Index>(n);
    const auto p_idx = static_cast<Eigen::Index>(p);

    Eigen::MatrixXd layer = Eigen::MatrixXd::Zero(n_rows, p_idx);
    std::normal_distribution<double> N01(0.0, 1.0);

    for (std::size_t m = 0; m < clusters_list.size(); ++m) {

        const auto& cols = clusters_list[m];
        const auto p_m = static_cast<Eigen::Index>(cols.size());

        // skip empty clusters (should not happen)
        if (p_m == 0) {
            continue;
        }

        // drawn univariate Z ~ N(0, I) of shape (n_rows x p_m).
        Eigen::MatrixXd Z(n_rows, p_m);
        for (Eigen::Index j = 0; j < p_m; ++j) {
            for (Eigen::Index i = 0; i < n_rows; ++i) {
                Z(i, j) = N01(rng);
            }
        }

        // coloring transform: block = Z * L_m^T  (rows of block ~ N(0, Sigma_m)).
        Eigen::MatrixXd block = Z * cholesky_lower_list[m].transpose();

        // Scatter into output columns.
        for (Eigen::Index j = 0; j < p_m; ++j) {
            layer.col(cols[static_cast<std::size_t>(j)]) = block.col(j);
        }
    }

    return layer;
}

// ===================================================================================
} /* End of namespace trex::trex_selector_methods::utils::cluster_dummies */
// ===================================================================================
#endif /* End of TREX_SELECTOR_METHODS_UTILS_TREX_CLUSTER_DUMMIES_HPP */
