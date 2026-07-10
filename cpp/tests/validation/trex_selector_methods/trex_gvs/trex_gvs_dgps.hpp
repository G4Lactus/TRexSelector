// ==============================================================================
// trex_gvs_dgps.hpp
// ==============================================================================
#ifndef DEMOS_TREX_SELECTOR_METHODS_GVS_DGPS_HPP
#define DEMOS_TREX_SELECTOR_METHODS_GVS_DGPS_HPP
// ==============================================================================
/**
 * @file trex_gvs_dgps.hpp
 *
 * @brief Data-generating processes (DGPs) for T-Rex+GVS demos and simulations.
 *
 * All generators return a GVSDGPResult containing the design matrix X, response
 * y, true coefficient vector beta, true support, and per-column group
 * assignments suitable for the GVS prior-groups parameter.
 *
 * Generators provided:
 *   - make_hastie_dgp()             — 3 equicorrelated groups (Zou & Hastie 2005)
 *   - make_ien_grouping_dgp()       — IEN grouping-effect scenario (Machkour et al.)
 *   - make_scattered_grouped_dgp()  — n_groups randomly scattered across p columns
 *   - make_unequal_blocks_dgp()     — 3 contiguous blocks of unequal size
 *   - make_mixed_blocks_dgp()       — 3 active + 1 inactive block, random order
 *   - make_random_blocks_dgp()      — 3 active blocks, randomly ordered
 *   - make_neg_corr_dgp()           — active group + negative-correlation trap
 *   - make_neg_traps_dgp()          — active group + two spatially separated traps
 *   - make_block_equicorr_dgp()     — G equal-size blocks, 3 truth scenarios
 *   - make_t3_equicorr_blocks_dgp() — 4 heavy-tailed t(3) equicorrelated blocks
 *   - make_ar1_blocks_dgp()         — 4 blocks with AR(1) Toeplitz within-block covariance
 *   - make_arma_blocks_dgp()        — 4 blocks with heterogeneous AR/MA/ARMA structure
 *   - make_hapgen_cholesky()        — Cholesky factor of the quasi-hapgen LD-block correlation
 *   - make_hapgen_dgp()             — quasi-hapgen 7-block LD design (p=500, s=130)
 *
 */
// ==============================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

// Eigen includes
#include <Eigen/Dense>


// ==============================================================================
namespace gvs_demo {
// ==============================================================================


// ==============================================================================
// DGP result
// ==============================================================================

/**
 * @brief Container returned by every DGP generator in this header.
 *
 * Provides the design matrix, response, true signal, and metadata required
 * by both single-run demos and MC harnesses.
 */
struct GVSDGPResult {

    /** @brief Design matrix (n × p). */
    Eigen::MatrixXd X;

    /** @brief Response vector (n). */
    Eigen::VectorXd y;

    /** @brief True coefficient vector (p); zero for inactive variables. */
    Eigen::VectorXd beta;

    /** @brief Sorted 0-based indices of active variables. */
    std::vector<std::size_t> true_support;

    /**
     * @brief 0-based cluster (group) assignment per column (length p).
     *
     * Suitable for passing directly to TRexGVSControlParameter::prior_groups.
     * Background (i.i.d.) columns each receive a unique singleton group ID.
     */
    std::vector<Eigen::Index> prior_groups;

    // -- scalar metadata --

    /** @brief Number of observations/rows of X. */
    int    n       = 0;

    /** @brief Number of predictors/columns of X. */
    int    p       = 0;

    /** @brief Number of active predictors. */
    int    s       = 0;

    /** @brief Target signal-to-noise ratio. */
    double snr     = 0.0;

    /** @brief Realised noise standard deviation. */
    double sigma_y = 0.0;

    /** @brief Within-group predictor noise standard deviation. */
    double sd_x    = 0.0;

    // -- optional block/group metadata (populated by block DGPs) --

    /** @brief Sizes of each correlated block/group (mixed/random DGPs). */
    std::vector<int>  group_sizes;

    /** @brief Shuffled placement order of blocks (mixed/random DGPs). */
    std::vector<int>  group_order;

    /** @brief Whether each block is active (mixed_blocks DGP only). */
    std::vector<bool> group_active;

    /** @brief 0-based indices of the active blocks (block_equicorr DGP). */
    std::vector<int> active_blocks;
};


// ==============================================================================
// Truth scenarios for the block-equicorrelated DGP
// ==============================================================================

/**
 * @brief Truth-scenario selector for make_block_equicorr_dgp().
 *
 * - INDIVIDUAL      — exactly 1 variable per active block is active
 * - REPRESENTATIVE  — 2 or 3 variables per active block are active
 * - WHOLE           — all block_size variables per active block are active
 */
enum class BlockScenario {INDIVIDUAL, REPRESENTATIVE, WHOLE};


// ==============================================================================
// DGP: Hastie (Zou & Hastie 2005, Example 4)
// ==============================================================================

/**
 * @brief Hastie-style DGP: 3 equicorrelated groups of 50 variables each.
 *
 * @details
 *  All 150 grouped variables are active (beta_j = 3 for j < 150).
 *  The remaining p - 150 columns are i.i.d. N(0, 1) background.
 *
 *  Within-group correlation: rho = 1 / (1 + sd_x^2).
 *  To target a specific rho, convert via sd_x = sqrt((1 - rho) / rho).
 *  The default sd_x = sqrt(0.01) gives rho ≈ 0.99.
 *
 *  The noise standard deviation sigma_y is calibrated to the target SNR:
 *    signal  = X * beta
 *    sigma_y = sqrt( Var(signal) / snr )      (biased variance)
 *
 * @param n     Number of observations.
 * @param p     Number of predictors (must be >= 150).
 * @param snr   Target signal-to-noise ratio Var(signal) / sigma_y^2.
 * @param sd_x  Within-group idiosyncratic noise sd. Default sqrt(0.01).
 * @param seed  RNG seed.
 *
 * @return Populated GVSDGPResult.
 */
inline GVSDGPResult make_hastie_dgp(int n,
                                    int p,
                                    double snr,
                                    double sd_x,
                                    unsigned seed)
{
    if (p < 150)
        throw std::invalid_argument("make_hastie_dgp: p must be >= 150");

    std::mt19937 rng(seed);
    std::normal_distribution<double> N01(0.0, 1.0);

    GVSDGPResult out;
    out.n    = n;
    out.p    = p;
    out.snr  = snr;
    out.sd_x = sd_x;

    out.X.resize(n, p);
    out.prior_groups.resize(static_cast<std::size_t>(p));

    // Three latent group factors
    Eigen::MatrixXd Z(n, 3);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k)
            Z(i, k) = N01(rng);

    // Grouped predictors (groups 0, 1, 2 — 50 columns each)
    for (int grp = 0; grp < 3; ++grp) {
        for (int k = 0; k < 50; ++k) {
            const int j = grp * 50 + k;
            out.prior_groups[static_cast<std::size_t>(j)] = grp;
            for (int i = 0; i < n; ++i)
                out.X(i, j) = Z(i, grp) + sd_x * N01(rng);
        }
    }

    // Background predictors — each is its own singleton group
    for (int k = 150; k < p; ++k) {
        out.prior_groups[static_cast<std::size_t>(k)] = k;  // unique ID
        for (int i = 0; i < n; ++i)
            out.X(i, k) = N01(rng);
    }

    // True signal: beta_j = 3 for all 150 grouped variables
    out.s = 150;
    out.beta.setZero(p);
    for (int j = 0; j < 150; ++j) {
        out.beta(j) = 3.0;
        out.true_support.push_back(static_cast<std::size_t>(j));
    }

    // SNR-calibrated noise
    const Eigen::VectorXd signal = out.X * out.beta;
    const double sig_var = (signal.array() - signal.mean()).square().mean();
    out.sigma_y = std::sqrt(sig_var / snr);

    out.y.resize(n);
    for (int i = 0; i < n; ++i)
        out.y(i) = signal(i) + out.sigma_y * N01(rng);

    return out;
}


// ==============================================================================
// DGP: IEN grouping effect (Machkour et al. – Section IV-A)
// ==============================================================================

/**
 * @brief IEN grouping-effect DGP from the reference paper (Section IV-A).
 *
 * @details
 *  Two small correlated groups and a background of independent variables:
 *
 *    G₁ = {0, 1, 2} : β_j = +1,  driven by latent Z₁
 *    G₂ = {3, 4, 5} : β_j = −1,  driven by latent Z₂
 *    Background = {6, …, p−1} : i.i.d. N(0, 1), β_j = 0
 *
 *  Within-group construction:
 *    X_j = Z_k + sd_x * ξ_j,  ξ_j ~ N(0, 1).
 *  Correlation within a group: ρ = 1 / (1 + sd_x²).
 *  For ρ = 0.75: sd_x = sqrt(1/3).
 *
 *  Default parameters from the paper:  n=150, p=100, SNR=3, ρ=0.75.
 *
 * @param n      Number of observations (default: 150).
 * @param p      Number of predictors (must be >= 6; default: 100).
 * @param snr    Target signal-to-noise ratio Var(Xβ)/σ² (default: 3.0).
 * @param rho    Desired within-group correlation (default: 0.75).
 * @param seed   RNG seed.
 *
 * @return Populated GVSDGPResult.
 */
inline GVSDGPResult make_ien_grouping_dgp(int n, int p, double snr,
                                          double rho, unsigned seed)
{
    if (p < 6)
        throw std::invalid_argument("make_ien_grouping_dgp: p must be >= 6");
    if (rho <= 0.0 || rho >= 1.0)
        throw std::invalid_argument("make_ien_grouping_dgp: rho must be in (0,1)");

    // Within-group noise sd derived from desired correlation:
    // rho = 1 / (1 + sd_x^2) => sd_x = sqrt((1 - rho) / rho)
    const double sd_x = std::sqrt((1.0 - rho) / rho);

    std::mt19937 rng(seed);
    std::normal_distribution<double> N01(0.0, 1.0);

    GVSDGPResult out;
    out.n    = n;
    out.p    = p;
    out.snr  = snr;
    out.sd_x = sd_x;

    out.X.resize(n, p);
    out.prior_groups.resize(static_cast<std::size_t>(p));

    // Two latent group factors
    Eigen::VectorXd Z1(n), Z2(n);
    for (int i = 0; i < n; ++i) {
        Z1(i) = N01(rng);
        Z2(i) = N01(rng);
    }

    // G1 = {0, 1, 2}: beta = +1
    for (int j = 0; j < 3; ++j) {
        out.prior_groups[static_cast<std::size_t>(j)] = 0;
        for (int i = 0; i < n; ++i)
            out.X(i, j) = Z1(i) + sd_x * N01(rng);
    }

    // G2 = {3, 4, 5}: beta = -1
    for (int j = 3; j < 6; ++j) {
        out.prior_groups[static_cast<std::size_t>(j)] = 1;
        for (int i = 0; i < n; ++i)
            out.X(i, j) = Z2(i) + sd_x * N01(rng);
    }

    // Background: i.i.d. N(0,1), singleton group IDs
    for (int j = 6; j < p; ++j) {
        out.prior_groups[static_cast<std::size_t>(j)] = j;  // unique ID
        for (int i = 0; i < n; ++i)
            out.X(i, j) = N01(rng);
    }

    // True signal
    out.s = 6;
    out.beta.setZero(p);
    for (int j = 0; j < 3; ++j) {
        out.beta(j) = 1.0;
        out.true_support.push_back(static_cast<std::size_t>(j));
    }
    for (int j = 3; j < 6; ++j) {
        out.beta(j) = -1.0;
        out.true_support.push_back(static_cast<std::size_t>(j));
    }

    // SNR-calibrated noise
    const Eigen::VectorXd signal = out.X * out.beta;
    const double sig_var = (signal.array() - signal.mean()).square().mean();
    out.sigma_y = std::sqrt(sig_var / snr);

    out.y.resize(n);
    for (int i = 0; i < n; ++i)
        out.y(i) = signal(i) + out.sigma_y * N01(rng);

    return out;
}


// ==============================================================================
// DGP: Scattered grouped
// ==============================================================================

/**
 * @brief Scattered-grouped DGP: n_groups groups of group_size variables
 *        randomly scattered across p columns.
 *
 * @details
 *  Each group is driven by one latent factor Z_m ~ N(0, 1).
 *  Within-group variable: x_j = Z_{m(j)} + sd_x * eps_j.
 *  Within-group correlation: rho = 1 / (1 + sd_x^2).
 *
 *  All grouped variables are active (beta = 3, s = n_groups * group_size).
 *  Background columns are i.i.d. N(0, 1); each gets a unique singleton
 *  group ID.
 *
 * @param n           Number of observations.
 * @param p           Number of predictors (must be >= n_groups * group_size).
 * @param n_groups    Number of correlated groups.
 * @param group_size  Number of variables per group.
 * @param snr         Target signal-to-noise ratio.
 * @param sd_x        Within-group noise sd. Default sqrt(0.01) => rho ≈ 0.99.
 * @param seed        RNG seed.
 *
 * @return Populated GVSDGPResult.
 */
inline GVSDGPResult make_scattered_grouped_dgp(int n, int p,
                                               int n_groups, int group_size,
                                               double snr, double sd_x,
                                               unsigned seed)
{
    const int n_grouped = n_groups * group_size;
    if (p < n_grouped)
        throw std::invalid_argument(
            "make_scattered_grouped_dgp: p must be >= n_groups * group_size");

    std::mt19937 rng(seed);
    std::normal_distribution<double> N01(0.0, 1.0);

    GVSDGPResult out;
    out.n    = n;
    out.p    = p;
    out.snr  = snr;
    out.sd_x = sd_x;

    out.X.resize(n, p);
    out.prior_groups.resize(static_cast<std::size_t>(p));

    // Randomly shuffle column indices, then assign blocks of group_size
    // contiguously to each group.
    std::vector<int> col_perm(p);
    std::iota(col_perm.begin(), col_perm.end(), 0);
    std::shuffle(col_perm.begin(), col_perm.end(), rng);

    // Per-group latent factors
    Eigen::MatrixXd Z(n, n_groups);
    for (int i = 0; i < n; ++i)
        for (int m = 0; m < n_groups; ++m)
            Z(i, m) = N01(rng);

    // Fill grouped columns: all group members are active (beta = 3)
    out.s = n_groups * group_size;
    out.beta.setZero(p);

    for (int m = 0; m < n_groups; ++m) {
        for (int k = 0; k < group_size; ++k) {
            const int  j  = col_perm[m * group_size + k];
            const auto ju = static_cast<std::size_t>(j);
            out.prior_groups[ju] = m;
            for (int i = 0; i < n; ++i)
                out.X(i, j) = Z(i, m) + sd_x * N01(rng);
            out.true_support.push_back(static_cast<std::size_t>(j));
            out.beta(static_cast<Eigen::Index>(j)) = 3.0;
        }
    }
    std::sort(out.true_support.begin(), out.true_support.end());

    // Fill background columns (singletons — independent white noise)
    for (int k = n_grouped; k < p; ++k) {
        const int j = col_perm[k];
        out.prior_groups[static_cast<std::size_t>(j)] =
            n_groups + (k - n_grouped);  // unique singleton ID
        for (int i = 0; i < n; ++i)
            out.X(i, j) = N01(rng);
    }

    // SNR-calibrated noise
    const Eigen::VectorXd signal = out.X * out.beta;
    const double sig_var = (signal.array() - signal.mean()).square().mean();
    out.sigma_y = std::sqrt(sig_var / snr);

    out.y.resize(n);
    for (int i = 0; i < n; ++i)
        out.y(i) = signal(i) + out.sigma_y * N01(rng);

    return out;
}


// ==============================================================================
// DGP: Unequal blocks
// ==============================================================================

/**
 * @brief Unequal-blocks DGP: three contiguous correlated blocks of sizes
 *        {20, 50, 80}, all active (β = 3, s = 150).
 *
 * @details
 *  Block 0: columns   0–19   (group_id = 0), driven by latent Z₀.
 *  Block 1: columns  20–69   (group_id = 1), driven by latent Z₁.
 *  Block 2: columns  70–149  (group_id = 2), driven by latent Z₂.
 *  Background: columns 150..p−1 — i.i.d. N(0, 1), singleton group IDs.
 *
 *  Within-group column: X_j = Z_m + sd_x * ξ_j,  ξ_j ~ N(0, 1).
 *  Within-group correlation: ρ = 1 / (1 + sd_x²).
 *  sd_x = sqrt(0.01) gives ρ ≈ 0.99.
 *
 * @param n     Number of observations.
 * @param p     Number of predictors (must be >= 150).
 * @param snr   Target signal-to-noise ratio.
 * @param sd_x  Within-group idiosyncratic noise sd.
 * @param seed  RNG seed.
 *
 * @return Populated GVSDGPResult with group_sizes = {20, 50, 80}.
 */
inline GVSDGPResult make_unequal_blocks_dgp(int n, int p, double snr,
                                            double sd_x, unsigned seed)
{
    if (p < 150)
        throw std::invalid_argument("make_unequal_blocks_dgp: p must be >= 150");

    std::mt19937 rng(seed);
    std::normal_distribution<double> N01(0.0, 1.0);

    GVSDGPResult out;
    out.n    = n;
    out.p    = p;
    out.snr  = snr;
    out.sd_x = sd_x;

    out.X.resize(n, p);
    out.beta.setZero(p);
    out.prior_groups.resize(static_cast<std::size_t>(p));

    // Latent factors for the three blocks
    const std::vector<int> block_sizes = {20, 50, 80};
    Eigen::MatrixXd Z(n, 3);
    for (int i = 0; i < n; ++i)
        for (int m = 0; m < 3; ++m)
            Z(i, m) = N01(rng);

    // Fill block columns
    int col = 0;
    for (int m = 0; m < 3; ++m) {
        for (int k = 0; k < block_sizes[static_cast<std::size_t>(m)]; ++k, ++col) {
            out.prior_groups[static_cast<std::size_t>(col)] = m;
            for (int i = 0; i < n; ++i)
                out.X(i, col) = Z(i, m) + sd_x * N01(rng);
            out.beta(col) = 3.0;
            out.true_support.push_back(static_cast<std::size_t>(col));
        }
    }
    out.s = 150;

    // Background columns — i.i.d. N(0, 1), singleton group IDs
    for (int j = 150; j < p; ++j) {
        out.prior_groups[static_cast<std::size_t>(j)] = j;
        for (int i = 0; i < n; ++i)
            out.X(i, j) = N01(rng);
    }

    // SNR-calibrated noise
    const Eigen::VectorXd signal = out.X * out.beta;
    const double sig_var = (signal.array() - signal.mean()).square().mean();
    out.sigma_y = std::sqrt(sig_var / snr);

    out.y.resize(n);
    for (int i = 0; i < n; ++i)
        out.y(i) = signal(i) + out.sigma_y * N01(rng);

    // Metadata
    out.group_sizes = {20, 50, 80};

    return out;
}


// ==============================================================================
// DGP: Mixed blocks
// ==============================================================================

/**
 * @brief Mixed-blocks DGP: four blocks of sizes {20, 50, 80, 65} placed in
 *        random order, separated by random noise gaps.  The last block is
 *        inactive; the first three are active (β = 3, s = 150).
 *
 * @details
 *  Block IDs 0–2 are active; block ID 3 is inactive (β = 0).
 *  The block placement order is a uniform random permutation of {0,1,2,3}.
 *  Noise gaps between/around blocks are drawn with a multinomial-like scheme
 *  that guarantees at least one noise column between any two consecutive blocks.
 *
 *  Gap algorithm (mirrors R's rmultinom + gap increment):
 *    total_grouped = 215,  num_blocks = 4
 *    rem_noise = p − 215 − 3   (reserve 1 per internal gap)
 *    gaps[0..4] initialised to 0; rem_noise distributed uniformly across them
 *    gaps[1], gaps[2], gaps[3] each incremented by 1  (internal separation)
 *
 *  Column order: gap[0] | block[order[0]] | gap[1] | block[order[1]] | ...
 *
 * @param n     Number of observations.
 * @param p     Number of predictors (must be >= 215 + 3 = 218).
 * @param snr   Target signal-to-noise ratio.
 * @param sd_x  Within-group idiosyncratic noise sd.
 * @param seed  RNG seed.
 *
 * @return Populated GVSDGPResult with group_sizes, group_order, group_active.
 */
inline GVSDGPResult make_mixed_blocks_dgp(int n, int p, double snr,
                                          double sd_x, unsigned seed)
{
    const std::vector<int>  block_sizes  = {20, 50, 80, 65};
    const std::vector<bool> is_active    = {true, true, true, false};
    const int num_blocks   = 4;
    const int total_grouped = 215;

    if (p < total_grouped + (num_blocks - 1))
        throw std::invalid_argument(
            "make_mixed_blocks_dgp: p must be >= 218");

    std::mt19937 rng(seed);
    std::normal_distribution<double> N01(0.0, 1.0);

    // --- Block order ---
    std::vector<int> block_order = {0, 1, 2, 3};
    std::shuffle(block_order.begin(), block_order.end(), rng);

    // --- Gap distribution ---
    // 5 gap slots: before block[0], between blocks[0-1], ..., after block[3]
    const int rem_noise = p - total_grouped - (num_blocks - 1);
    std::vector<int> gaps(static_cast<std::size_t>(num_blocks + 1), 0);
    {
        std::uniform_int_distribution<int> slot_dist(0, num_blocks);
        for (int k = 0; k < rem_noise; ++k)
            gaps[static_cast<std::size_t>(slot_dist(rng))]++;
    }
    // Add 1 to internal gaps (indices 1 .. num_blocks-1)
    for (int i = 1; i < num_blocks; ++i)
        gaps[static_cast<std::size_t>(i)]++;

    // --- Latent factors (one per block, drawn in block_order) ---
    Eigen::MatrixXd Z(n, num_blocks);
    for (int i = 0; i < n; ++i)
        for (int m = 0; m < num_blocks; ++m)
            Z(i, m) = N01(rng);

    GVSDGPResult out;
    out.n    = n;
    out.p    = p;
    out.snr  = snr;
    out.sd_x = sd_x;

    out.X.resize(n, p);
    out.beta.setZero(p);
    out.prior_groups.resize(static_cast<std::size_t>(p));

    // --- Fill columns ---
    int col = 0;
    for (int slot = 0; slot <= num_blocks; ++slot) {
        // Noise gap
        const int g = gaps[static_cast<std::size_t>(slot)];
        for (int k = 0; k < g; ++k, ++col) {
            out.prior_groups[static_cast<std::size_t>(col)] = col;  // singleton
            for (int i = 0; i < n; ++i)
                out.X(i, col) = N01(rng);
        }

        if (slot == num_blocks) break;

        // Block
        const int blk  = block_order[static_cast<std::size_t>(slot)];
        const int bsize = block_sizes[static_cast<std::size_t>(blk)];
        const bool active = is_active[static_cast<std::size_t>(blk)];

        for (int k = 0; k < bsize; ++k, ++col) {
            out.prior_groups[static_cast<std::size_t>(col)] = blk;
            for (int i = 0; i < n; ++i)
                out.X(i, col) = Z(i, blk) + sd_x * N01(rng);
            if (active) {
                out.beta(col) = 3.0;
                out.true_support.push_back(static_cast<std::size_t>(col));
            }
        }
    }
    std::sort(out.true_support.begin(), out.true_support.end());
    out.s = static_cast<int>(out.true_support.size());  // = 150

    // SNR-calibrated noise
    const Eigen::VectorXd signal = out.X * out.beta;
    const double sig_var = (signal.array() - signal.mean()).square().mean();
    out.sigma_y = std::sqrt(sig_var / snr);

    out.y.resize(n);
    for (int i = 0; i < n; ++i)
        out.y(i) = signal(i) + out.sigma_y * N01(rng);

    // Metadata
    out.group_sizes  = {20, 50, 80, 65};
    out.group_order  = block_order;
    out.group_active = {true, true, true, false};

    return out;
}


// ==============================================================================
// DGP: Random blocks
// ==============================================================================

/**
 * @brief Random-blocks DGP: three active blocks of sizes {20, 50, 80} placed
 *        in random order, separated by random noise gaps (s = 150).
 *
 * @details
 *  All three blocks are active (β = 3).  The block placement order is a
 *  uniform random permutation of {0, 1, 2}.
 *
 *  Gap algorithm (4 slots, mirrors mixed_blocks with num_blocks = 3):
 *    rem_noise = p − 150 − 2   (reserve 1 per internal gap)
 *    gaps[0..3] initialised to 0; rem_noise distributed uniformly
 *    gaps[1] and gaps[2] each incremented by 1
 *
 * @param n     Number of observations.
 * @param p     Number of predictors (must be >= 150 + 2 = 152).
 * @param snr   Target signal-to-noise ratio.
 * @param sd_x  Within-group idiosyncratic noise sd.
 * @param seed  RNG seed.
 *
 * @return Populated GVSDGPResult with group_sizes and group_order.
 */
inline GVSDGPResult make_random_blocks_dgp(int n, int p, double snr,
                                           double sd_x, unsigned seed)
{
    const std::vector<int> block_sizes = {20, 50, 80};
    const int num_blocks    = 3;
    const int total_grouped = 150;

    if (p < total_grouped + (num_blocks - 1))
        throw std::invalid_argument(
            "make_random_blocks_dgp: p must be >= 152");

    std::mt19937 rng(seed);
    std::normal_distribution<double> N01(0.0, 1.0);

    // --- Block order ---
    std::vector<int> block_order = {0, 1, 2};
    std::shuffle(block_order.begin(), block_order.end(), rng);

    // --- Gap distribution ---
    // 4 gap slots: before block[0], between blocks[0-1], [1-2], after block[2]
    const int rem_noise = p - total_grouped - (num_blocks - 1);
    std::vector<int> gaps(static_cast<std::size_t>(num_blocks + 1), 0);
    {
        std::uniform_int_distribution<int> slot_dist(0, num_blocks);
        for (int k = 0; k < rem_noise; ++k)
            gaps[static_cast<std::size_t>(slot_dist(rng))]++;
    }
    // Add 1 to internal gaps (indices 1 .. num_blocks-1)
    for (int i = 1; i < num_blocks; ++i)
        gaps[static_cast<std::size_t>(i)]++;

    // --- Latent factors (one per block) ---
    Eigen::MatrixXd Z(n, num_blocks);
    for (int i = 0; i < n; ++i)
        for (int m = 0; m < num_blocks; ++m)
            Z(i, m) = N01(rng);

    GVSDGPResult out;
    out.n    = n;
    out.p    = p;
    out.snr  = snr;
    out.sd_x = sd_x;

    out.X.resize(n, p);
    out.beta.setZero(p);
    out.prior_groups.resize(static_cast<std::size_t>(p));

    // --- Fill columns ---
    int col = 0;
    for (int slot = 0; slot <= num_blocks; ++slot) {
        // Noise gap
        const int g = gaps[static_cast<std::size_t>(slot)];
        for (int k = 0; k < g; ++k, ++col) {
            out.prior_groups[static_cast<std::size_t>(col)] = col;  // singleton
            for (int i = 0; i < n; ++i)
                out.X(i, col) = N01(rng);
        }

        if (slot == num_blocks) break;

        // Block (all active)
        const int blk   = block_order[static_cast<std::size_t>(slot)];
        const int bsize = block_sizes[static_cast<std::size_t>(blk)];

        for (int k = 0; k < bsize; ++k, ++col) {
            out.prior_groups[static_cast<std::size_t>(col)] = blk;
            for (int i = 0; i < n; ++i)
                out.X(i, col) = Z(i, blk) + sd_x * N01(rng);
            out.beta(col) = 3.0;
            out.true_support.push_back(static_cast<std::size_t>(col));
        }
    }
    std::sort(out.true_support.begin(), out.true_support.end());
    out.s = 150;

    // SNR-calibrated noise
    const Eigen::VectorXd signal = out.X * out.beta;
    const double sig_var = (signal.array() - signal.mean()).square().mean();
    out.sigma_y = std::sqrt(sig_var / snr);

    out.y.resize(n);
    for (int i = 0; i < n; ++i)
        out.y(i) = signal(i) + out.sigma_y * N01(rng);

    // Metadata
    out.group_sizes = {20, 50, 80};
    out.group_order = block_order;

    return out;
}


// ==============================================================================
// DGP: Negative correlations
// ==============================================================================

/**
 * @brief Negative-correlation DGP: one active group with sign-flipped halves
 *        and an inactive trap group with the same structure.
 *
 * @details
 *  G₁ (columns 0–99): active.
 *    Positive half  (0–49):  X_j = +Z₁ + sd_x·ξ_j,  β_j = +3
 *    Negative half (50–99):  X_j = −Z₁ + sd_x·ξ_j,  β_j = −3
 *    → correlations within group are both +ρ (with Z₁) and −ρ across halves.
 *
 *  G₂ (columns 100–199): inactive trap.
 *    Positive half (100–149): X_j = +Z₂ + sd_x·ξ_j,  β_j = 0
 *    Negative half (150–199): X_j = −Z₂ + sd_x·ξ_j,  β_j = 0
 *
 *  Background (200..p−1): i.i.d. N(0, 1),  β_j = 0,  singleton group IDs.
 *
 *  True support = {0, …, 99},  s = 100.
 *
 * @param n     Number of observations.
 * @param p     Number of predictors (must be >= 200).
 * @param snr   Target signal-to-noise ratio.
 * @param sd_x  Within-group idiosyncratic noise sd.
 * @param seed  RNG seed.
 *
 * @return Populated GVSDGPResult.
 */
inline GVSDGPResult make_neg_corr_dgp(int n, int p, double snr,
                                      double sd_x, unsigned seed)
{
    if (p < 200)
        throw std::invalid_argument("make_neg_corr_dgp: p must be >= 200");

    std::mt19937 rng(seed);
    std::normal_distribution<double> N01(0.0, 1.0);

    // Latent factors
    Eigen::VectorXd Z1(n), Z2(n);
    for (int i = 0; i < n; ++i) {
        Z1(i) = N01(rng);
        Z2(i) = N01(rng);
    }

    GVSDGPResult out;
    out.n    = n;
    out.p    = p;
    out.snr  = snr;
    out.sd_x = sd_x;

    out.X.resize(n, p);
    out.beta.setZero(p);
    out.prior_groups.resize(static_cast<std::size_t>(p));

    // G1 positive half (0–49): group_id = 0, active
    for (int j = 0; j < 50; ++j) {
        out.prior_groups[static_cast<std::size_t>(j)] = 0;
        for (int i = 0; i < n; ++i)
            out.X(i, j) = Z1(i) + sd_x * N01(rng);
        out.beta(j) = 3.0;
        out.true_support.push_back(static_cast<std::size_t>(j));
    }

    // G1 negative half (50–99): group_id = 0, active
    for (int j = 50; j < 100; ++j) {
        out.prior_groups[static_cast<std::size_t>(j)] = 0;
        for (int i = 0; i < n; ++i)
            out.X(i, j) = -Z1(i) + sd_x * N01(rng);
        out.beta(j) = -3.0;
        out.true_support.push_back(static_cast<std::size_t>(j));
    }

    // G2 positive half (100–149): group_id = 1, inactive trap
    for (int j = 100; j < 150; ++j) {
        out.prior_groups[static_cast<std::size_t>(j)] = 1;
        for (int i = 0; i < n; ++i)
            out.X(i, j) = Z2(i) + sd_x * N01(rng);
    }

    // G2 negative half (150–199): group_id = 1, inactive trap
    for (int j = 150; j < 200; ++j) {
        out.prior_groups[static_cast<std::size_t>(j)] = 1;
        for (int i = 0; i < n; ++i)
            out.X(i, j) = -Z2(i) + sd_x * N01(rng);
    }

    // Background (200..p-1): i.i.d. N(0,1), singleton group IDs
    for (int j = 200; j < p; ++j) {
        out.prior_groups[static_cast<std::size_t>(j)] = j;
        for (int i = 0; i < n; ++i)
            out.X(i, j) = N01(rng);
    }

    out.s = 100;

    // SNR-calibrated noise
    const Eigen::VectorXd signal = out.X * out.beta;
    const double sig_var = (signal.array() - signal.mean()).square().mean();
    out.sigma_y = std::sqrt(sig_var / snr);

    out.y.resize(n);
    for (int i = 0; i < n; ++i)
        out.y(i) = signal(i) + out.sigma_y * N01(rng);

    return out;
}


// ==============================================================================
// DGP: Negative traps
// ==============================================================================

/**
 * @brief Negative-traps DGP: one active group surrounded by two spatially
 *        separated inactive trap groups with sign-flipped structure.
 *
 * @details
 *  G₁  (0–99):   active.    0–49: +Z₁,β=+3;  50–99: −Z₁,β=−3.  group_id=0
 *  Trap₁(100–199): inactive. 100–149: +Z₂,β=0; 150–199: −Z₂,β=0.  group_id=1
 *  Noise(200–299): i.i.d. N(0,1). singleton group IDs.
 *  Trap₂(300–359): inactive. 300–329: +Z₃,β=0; 330–359: −Z₃,β=0.  group_id=2
 *  Noise(360..p−1): i.i.d. N(0,1). singleton group IDs.
 *
 *  True support = {0, …, 99},  s = 100.
 *
 * @param n     Number of observations.
 * @param p     Number of predictors (must be >= 360).
 * @param snr   Target signal-to-noise ratio.
 * @param sd_x  Within-group idiosyncratic noise sd.
 * @param seed  RNG seed.
 *
 * @return Populated GVSDGPResult.
 */
inline GVSDGPResult make_neg_traps_dgp(int n, int p, double snr,
                                       double sd_x, unsigned seed)
{
    if (p < 360)
        throw std::invalid_argument("make_neg_traps_dgp: p must be >= 360");

    std::mt19937 rng(seed);
    std::normal_distribution<double> N01(0.0, 1.0);

    // Latent factors
    Eigen::VectorXd Z1(n), Z2(n), Z3(n);
    for (int i = 0; i < n; ++i) {
        Z1(i) = N01(rng);
        Z2(i) = N01(rng);
        Z3(i) = N01(rng);
    }

    GVSDGPResult out;
    out.n    = n;
    out.p    = p;
    out.snr  = snr;
    out.sd_x = sd_x;

    out.X.resize(n, p);
    out.beta.setZero(p);
    out.prior_groups.resize(static_cast<std::size_t>(p));

    // G1 positive half (0–49): active
    for (int j = 0; j < 50; ++j) {
        out.prior_groups[static_cast<std::size_t>(j)] = 0;
        for (int i = 0; i < n; ++i)
            out.X(i, j) = Z1(i) + sd_x * N01(rng);
        out.beta(j) = 3.0;
        out.true_support.push_back(static_cast<std::size_t>(j));
    }

    // G1 negative half (50–99): active
    for (int j = 50; j < 100; ++j) {
        out.prior_groups[static_cast<std::size_t>(j)] = 0;
        for (int i = 0; i < n; ++i)
            out.X(i, j) = -Z1(i) + sd_x * N01(rng);
        out.beta(j) = -3.0;
        out.true_support.push_back(static_cast<std::size_t>(j));
    }

    // Trap1 positive half (100–149): inactive
    for (int j = 100; j < 150; ++j) {
        out.prior_groups[static_cast<std::size_t>(j)] = 1;
        for (int i = 0; i < n; ++i)
            out.X(i, j) = Z2(i) + sd_x * N01(rng);
    }

    // Trap1 negative half (150–199): inactive
    for (int j = 150; j < 200; ++j) {
        out.prior_groups[static_cast<std::size_t>(j)] = 1;
        for (int i = 0; i < n; ++i)
            out.X(i, j) = -Z2(i) + sd_x * N01(rng);
    }

    // Noise1 (200–299): i.i.d. N(0,1), singleton group IDs
    for (int j = 200; j < 300; ++j) {
        out.prior_groups[static_cast<std::size_t>(j)] = j;
        for (int i = 0; i < n; ++i)
            out.X(i, j) = N01(rng);
    }

    // Trap2 positive half (300–329): inactive
    for (int j = 300; j < 330; ++j) {
        out.prior_groups[static_cast<std::size_t>(j)] = 2;
        for (int i = 0; i < n; ++i)
            out.X(i, j) = Z3(i) + sd_x * N01(rng);
    }

    // Trap2 negative half (330–359): inactive
    for (int j = 330; j < 360; ++j) {
        out.prior_groups[static_cast<std::size_t>(j)] = 2;
        for (int i = 0; i < n; ++i)
            out.X(i, j) = -Z3(i) + sd_x * N01(rng);
    }

    // Noise2 (360..p-1): i.i.d. N(0,1), singleton group IDs
    for (int j = 360; j < p; ++j) {
        out.prior_groups[static_cast<std::size_t>(j)] = j;
        for (int i = 0; i < n; ++i)
            out.X(i, j) = N01(rng);
    }

    out.s = 100;

    // SNR-calibrated noise
    const Eigen::VectorXd signal = out.X * out.beta;
    const double sig_var = (signal.array() - signal.mean()).square().mean();
    out.sigma_y = std::sqrt(sig_var / snr);

    out.y.resize(n);
    for (int i = 0; i < n; ++i)
        out.y(i) = signal(i) + out.sigma_y * N01(rng);

    return out;
}


// ==============================================================================
// DGP: Block-equicorrelated Gaussian model
// ==============================================================================

/**
 * @brief Block-equicorrelated DGP with G equal-size contiguous blocks.
 *
 * @details
 *  The design matrix has p = G * block_size columns arranged in G contiguous
 *  blocks of equal size.  Within each block the exact equicorrelation is rho:
 *
 *    X_{ij} = sqrt(rho) * Z_i^(k)  +  sqrt(1-rho) * eps_{ij}
 *
 *  where Z^(k) ~ N(0,1) is the block-k latent factor and eps_{ij} ~ N(0,1)
 *  are i.i.d. idiosyncratic terms.  Each column therefore has unit variance
 *  and Cov(X_{ij}, X_{ij'}) = rho for j != j' in the same block and 0 otherwise.
 *
 *  n_active_blocks blocks are chosen uniformly without replacement.  Each
 *  active block receives a random sign (+1 or -1) and nonzero coefficient b.
 *  Which variables in the block are active depends on the scenario:
 *
 *    INDIVIDUAL    — the first variable of each active block receives beta = sign * b
 *    REPRESENTATIVE — a randomly chosen subset of 2 or 3 variables per active block
 *    WHOLE         — all block_size variables of each active block
 *
 *  sigma_y is calibrated to the target SNR (biased variance convention, matching
 *  all other DGPs in this header).
 *
 *  prior_groups[j] = j / block_size (0-based, contiguous [0, G-1]) so that the
 *  result can be passed directly to TRexGVSControlParameter::prior_groups.
 *
 * @param n               Number of observations.
 * @param p               Number of predictors; must equal G * block_size.
 * @param G               Number of blocks.
 * @param block_size      Number of variables per block.
 * @param n_active_blocks Number of blocks that carry signal.
 * @param rho             Within-block equicorrelation in (0, 1).
 * @param snr             Target SNR = Var(X*beta) / sigma_y^2.
 * @param scenario        INDIVIDUAL, REPRESENTATIVE, or WHOLE.
 * @param seed            RNG seed.
 * @param b               Nonzero coefficient magnitude (default 3.0).
 *
 * @return Populated GVSDGPResult; active_blocks contains 0-based block IDs.
 */
inline GVSDGPResult make_block_equicorr_dgp(
    int           n,
    int           p,
    int           G,
    int           block_size,
    int           n_active_blocks,
    double        rho,
    double        snr,
    BlockScenario scenario,
    unsigned      seed,
    double        b = 3.0)
{
    if (p != G * block_size)
        throw std::invalid_argument(
            "make_block_equicorr_dgp: p must equal G * block_size");
    if (n_active_blocks > G)
        throw std::invalid_argument(
            "make_block_equicorr_dgp: n_active_blocks must be <= G");
    if (rho <= 0.0 || rho >= 1.0)
        throw std::invalid_argument(
            "make_block_equicorr_dgp: rho must be in (0, 1)");

    std::mt19937 rng(seed);
    std::normal_distribution<double>  N01(0.0, 1.0);
    std::uniform_int_distribution<int> fair_coin(0, 1);

    GVSDGPResult out;
    out.n   = n;
    out.p   = p;
    out.snr = snr;

    // -------------------------------------------------------------------------
    // Build X via latent factor model
    // -------------------------------------------------------------------------
    out.X.resize(n, p);
    out.prior_groups.resize(static_cast<std::size_t>(p));

    // One latent factor per block
    Eigen::MatrixXd Z(n, G);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < G; ++k)
            Z(i, k) = N01(rng);

    const double sqrt_rho     = std::sqrt(rho);
    const double sqrt_one_rho = std::sqrt(1.0 - rho);

    for (int k = 0; k < G; ++k) {
        for (int jj = 0; jj < block_size; ++jj) {
            const int j = k * block_size + jj;
            out.prior_groups[static_cast<std::size_t>(j)] =
                static_cast<Eigen::Index>(k);
            for (int i = 0; i < n; ++i)
                out.X(i, j) = sqrt_rho * Z(i, k) +
                                       sqrt_one_rho * N01(rng);
        }
    }

    // -------------------------------------------------------------------------
    // Choose active blocks without replacement
    // -------------------------------------------------------------------------
    std::vector<int> block_ids(static_cast<std::size_t>(G));
    std::iota(block_ids.begin(), block_ids.end(), 0);
    std::shuffle(block_ids.begin(), block_ids.end(), rng);
    out.active_blocks.assign(
        block_ids.begin(),
        block_ids.begin() + static_cast<std::ptrdiff_t>(n_active_blocks));
    std::sort(out.active_blocks.begin(), out.active_blocks.end());

    // -------------------------------------------------------------------------
    // Build beta according to scenario
    // -------------------------------------------------------------------------
    out.beta.setZero(p);

    // Uniform distribution for REPRESENTATIVE subset size (2 or 3)
    std::uniform_int_distribution<int> subset23(2, 3);
    // Uniform distribution for picking variable positions within a block
    std::uniform_int_distribution<int> pick_pos(0, block_size - 1);

    for (int blk : out.active_blocks) {
        const double sign = (fair_coin(rng) == 0) ? 1.0 : -1.0;
        const int    base = blk * block_size;

        switch (scenario) {
            case BlockScenario::INDIVIDUAL: {
                out.beta(base) = sign * b;
                out.true_support.push_back(
                    static_cast<std::size_t>(base));
                break;
            }
            case BlockScenario::REPRESENTATIVE: {
                const int n_active = subset23(rng);
                // Pick n_active distinct positions within the block
                std::vector<int> positions(static_cast<std::size_t>(block_size));
                std::iota(positions.begin(), positions.end(), 0);
                std::shuffle(positions.begin(), positions.end(), rng);
                for (int r = 0; r < n_active; ++r) {
                    const int j = base + positions[static_cast<std::size_t>(r)];
                    out.beta(j) = sign * b;
                    out.true_support.push_back(static_cast<std::size_t>(j));
                }
                break;
            }
            case BlockScenario::WHOLE: {
                for (int jj = 0; jj < block_size; ++jj) {
                    const int j = base + jj;
                    out.beta(j) = sign * b;
                    out.true_support.push_back(static_cast<std::size_t>(j));
                }
                break;
            }
        }
    }

    std::sort(out.true_support.begin(), out.true_support.end());
    out.s = static_cast<int>(out.true_support.size());

    // -------------------------------------------------------------------------
    // SNR-calibrated noise (biased variance convention)
    // -------------------------------------------------------------------------
    const Eigen::VectorXd signal = out.X * out.beta;
    const double sig_var = (signal.array() - signal.mean()).square().mean();
    out.sigma_y = std::sqrt(sig_var / snr);

    out.y.resize(n);
    for (int i = 0; i < n; ++i)
        out.y(i) = signal(i) + out.sigma_y * N01(rng);

    return out;
}


// ==============================================================================
// DGP: Heavy-tailed equicorrelated blocks  (used by Demo 05)
// ==============================================================================

/**
 * @brief Heavy-tailed equicorrelated blocks DGP: four blocks {20,50,80,65}
 *        with scaled Student-t(3) latent factors and idiosyncratic noise.
 *
 * @details
 *  Identical block structure and active/trap assignment to make_mixed_blocks_dgp,
 *  but using unit-variance scaled t(3) distributions throughout:
 *
 *    X_j = sqrt(rho) * Z_blk + sqrt(1-rho) * E_j
 *
 *  where Z_blk, E_j ~ t(3)/sqrt(3)  (variance = 1).
 *  Response noise: epsilon ~ t(3)/sqrt(3) * sigma_y.
 *  Block 3 (size 65) is the inactive equicorrelated trap (beta = 0).
 *
 * @param n    Number of observations.
 * @param p    Number of predictors (>= 218).
 * @param snr  Target signal-to-noise ratio.
 * @param rho  Within-block equicorrelation in (0, 1).
 * @param seed RNG seed.
 *
 * @return Populated GVSDGPResult (s = 150, group_sizes = {20,50,80,65}).
 */
inline GVSDGPResult make_t3_equicorr_blocks_dgp(int n, int p, double snr,
                                                 double rho, unsigned seed)
{
    if (rho <= 0.0 || rho >= 1.0)
        throw std::invalid_argument(
            "make_t3_equicorr_blocks_dgp: rho must be in (0, 1)");

    const std::vector<int>  block_sizes = {20, 50, 80, 65};
    const std::vector<bool> is_active   = {true, true, true, false};
    const int num_blocks    = 4;
    const int total_grouped = 215;

    if (p < total_grouped + (num_blocks - 1))
        throw std::invalid_argument(
            "make_t3_equicorr_blocks_dgp: p must be >= 218");

    std::mt19937 rng(seed);
    std::student_t_distribution<double> T3(3.0);
    // Var(t(3)) = 3/(3-2) = 3; divide by sqrt(3) to obtain unit variance.
    const double t3_scale = std::sqrt(3.0);
    auto t3_unit = [&]() { return T3(rng) / t3_scale; };

    // --- Block order (shuffled) ---
    std::vector<int> block_order = {0, 1, 2, 3};
    std::shuffle(block_order.begin(), block_order.end(), rng);

    // --- Gap distribution (identical to make_mixed_blocks_dgp) ---
    const int rem_noise = p - total_grouped - (num_blocks - 1);
    std::vector<int> gaps(static_cast<std::size_t>(num_blocks + 1), 0);
    {
        std::uniform_int_distribution<int> slot_dist(0, num_blocks);
        for (int k = 0; k < rem_noise; ++k)
            gaps[static_cast<std::size_t>(slot_dist(rng))]++;
    }
    for (int i = 1; i < num_blocks; ++i)
        gaps[static_cast<std::size_t>(i)]++;

    const double sqrt_rho     = std::sqrt(rho);
    const double sqrt_one_rho = std::sqrt(1.0 - rho);

    // --- One unit-variance t(3) latent factor per block ---
    Eigen::MatrixXd Z(n, num_blocks);
    for (int i = 0; i < n; ++i)
        for (int m = 0; m < num_blocks; ++m)
            Z(i, m) = t3_unit();

    GVSDGPResult out;
    out.n    = n;
    out.p    = p;
    out.snr  = snr;
    out.sd_x = std::sqrt((1.0 - rho) / rho);  // equivalent latent-factor sd

    out.X.resize(n, p);
    out.beta.setZero(p);
    out.prior_groups.resize(static_cast<std::size_t>(p));

    // --- Fill columns ---
    int col = 0;
    for (int slot = 0; slot <= num_blocks; ++slot) {
        const int g = gaps[static_cast<std::size_t>(slot)];
        for (int k = 0; k < g; ++k, ++col) {
            out.prior_groups[static_cast<std::size_t>(col)] = col;
            for (int i = 0; i < n; ++i)
                out.X(i, col) = t3_unit();
        }
        if (slot == num_blocks) break;

        const int  blk    = block_order[static_cast<std::size_t>(slot)];
        const int  bsize  = block_sizes[static_cast<std::size_t>(blk)];
        const bool active = is_active[static_cast<std::size_t>(blk)];

        for (int k = 0; k < bsize; ++k, ++col) {
            out.prior_groups[static_cast<std::size_t>(col)] = blk;
            for (int i = 0; i < n; ++i)
                out.X(i, col) = sqrt_rho * Z(i, blk) + sqrt_one_rho * t3_unit();
            if (active) {
                out.beta(col) = 3.0;
                out.true_support.push_back(static_cast<std::size_t>(col));
            }
        }
    }
    std::sort(out.true_support.begin(), out.true_support.end());
    out.s = 150;

    // SNR-calibrated noise (biased variance convention)
    const Eigen::VectorXd signal = out.X * out.beta;
    const double sig_var = (signal.array() - signal.mean()).square().mean();
    out.sigma_y = std::sqrt(sig_var / snr);

    // Response with t(3) noise
    out.y.resize(n);
    for (int i = 0; i < n; ++i)
        out.y(i) = signal(i) + out.sigma_y * t3_unit();

    out.group_sizes  = {20, 50, 80, 65};
    out.group_order  = block_order;
    out.group_active = {true, true, true, false};

    return out;
}


// ==============================================================================
// DGP: AR(1)-within-block  (used by Demo 06)
// ==============================================================================

/**
 * @brief AR(1)-within-block DGP: four blocks {20,50,80,65} with AR(1)
 *        covariance structure, placed in random order with noise gaps.
 *
 * @details
 *  Within each block the n rows (observations) are independent draws from
 *  N(0, Sigma_b) where Sigma_b[i,j] = rho^|i-j| (AR(1) Toeplitz matrix).
 *  The Cholesky factorisation of each block's Sigma_b is computed once via
 *  Eigen::LLT; columns are generated as X_block = Z * L^T, Z ~ N(0, I).
 *
 *  Block 3 (size 65) is the inactive AR(1) trap (beta = 0).
 *  Background (gap) columns are i.i.d. N(0, 1).
 *
 * @param n    Number of observations.
 * @param p    Number of predictors (>= 218).
 * @param snr  Target signal-to-noise ratio.
 * @param rho  AR(1) lag-1 correlation in [0, 1).
 * @param seed RNG seed.
 *
 * @return Populated GVSDGPResult (s = 150, group_sizes = {20,50,80,65}).
 */
inline GVSDGPResult make_ar1_blocks_dgp(int n, int p, double snr,
                                         double rho, unsigned seed)
{
    if (rho < 0.0 || rho >= 1.0)
        throw std::invalid_argument(
            "make_ar1_blocks_dgp: rho must be in [0, 1)");

    const std::vector<int>  block_sizes = {20, 50, 80, 65};
    const std::vector<bool> is_active   = {true, true, true, false};
    const int num_blocks    = 4;
    const int total_grouped = 215;

    if (p < total_grouped + (num_blocks - 1))
        throw std::invalid_argument(
            "make_ar1_blocks_dgp: p must be >= 218");

    std::mt19937 rng(seed);
    std::normal_distribution<double> N01(0.0, 1.0);

    // --- Block order ---
    std::vector<int> block_order = {0, 1, 2, 3};
    std::shuffle(block_order.begin(), block_order.end(), rng);

    // --- Gap distribution ---
    const int rem_noise = p - total_grouped - (num_blocks - 1);
    std::vector<int> gaps(static_cast<std::size_t>(num_blocks + 1), 0);
    {
        std::uniform_int_distribution<int> slot_dist(0, num_blocks);
        for (int k = 0; k < rem_noise; ++k)
            gaps[static_cast<std::size_t>(slot_dist(rng))]++;
    }
    for (int i = 1; i < num_blocks; ++i)
        gaps[static_cast<std::size_t>(i)]++;

    // --- Precompute Cholesky of AR(1) Sigma for each logical block ---
    // Sigma_b[i,j] = rho^|i-j|
    std::vector<Eigen::MatrixXd> chol_L(static_cast<std::size_t>(num_blocks));
    for (int blk = 0; blk < num_blocks; ++blk) {
        const int bs = block_sizes[static_cast<std::size_t>(blk)];
        Eigen::MatrixXd Sigma(bs, bs);
        for (int ii = 0; ii < bs; ++ii)
            for (int jj = 0; jj < bs; ++jj)
                Sigma(ii, jj) = std::pow(rho,
                    static_cast<double>(std::abs(ii - jj)));
        chol_L[static_cast<std::size_t>(blk)] =
            Eigen::LLT<Eigen::MatrixXd>(Sigma).matrixL();
    }

    GVSDGPResult out;
    out.n    = n;
    out.p    = p;
    out.snr  = snr;
    out.sd_x = 0.0;   // AR(1): no single latent-factor sd_x

    out.X.resize(n, p);
    out.beta.setZero(p);
    out.prior_groups.resize(static_cast<std::size_t>(p));

    // --- Fill columns ---
    int col = 0;
    for (int slot = 0; slot <= num_blocks; ++slot) {
        const int g = gaps[static_cast<std::size_t>(slot)];
        for (int k = 0; k < g; ++k, ++col) {
            out.prior_groups[static_cast<std::size_t>(col)] = col;
            for (int i = 0; i < n; ++i)
                out.X(i, col) = N01(rng);
        }
        if (slot == num_blocks) break;

        const int  blk    = block_order[static_cast<std::size_t>(slot)];
        const int  bsize  = block_sizes[static_cast<std::size_t>(blk)];
        const bool active = is_active[static_cast<std::size_t>(blk)];
        const Eigen::MatrixXd& L = chol_L[static_cast<std::size_t>(blk)];

        // Draw Z (n x bsize) ~ N(0,I); X_block = Z * L^T ~ N(0, Sigma_b)
        Eigen::MatrixXd Z_blk(n, bsize);
        for (int i = 0; i < n; ++i)
            for (int jj = 0; jj < bsize; ++jj)
                Z_blk(i, jj) = N01(rng);
        const Eigen::MatrixXd X_blk = Z_blk * L.transpose();

        for (int k = 0; k < bsize; ++k, ++col) {
            out.prior_groups[static_cast<std::size_t>(col)] = blk;
            out.X.col(col) = X_blk.col(k);
            if (active) {
                out.beta(col) = 3.0;
                out.true_support.push_back(static_cast<std::size_t>(col));
            }
        }
    }
    std::sort(out.true_support.begin(), out.true_support.end());
    out.s = static_cast<int>(out.true_support.size());  // 150

    // SNR-calibrated noise
    const Eigen::VectorXd signal = out.X * out.beta;
    const double sig_var = (signal.array() - signal.mean()).square().mean();
    out.sigma_y = std::sqrt(sig_var / snr);

    out.y.resize(n);
    for (int i = 0; i < n; ++i)
        out.y(i) = signal(i) + out.sigma_y * N01(rng);

    out.group_sizes  = {20, 50, 80, 65};
    out.group_order  = block_order;
    out.group_active = {true, true, true, false};

    return out;
}


// ==============================================================================
// DGP: Heterogeneous ARMA blocks  (used by Demo 07)
// ==============================================================================

/**
 * @brief Heterogeneous ARMA blocks DGP: four blocks {20,50,80,65} each with
 *        a different within-block ARMA time-series structure.
 *
 * @details
 *  Block 0 (size 20, active): AR(1),    phi = ar_coef
 *  Block 1 (size 50, active): MA(3),    theta = (0.5, 0.3, 0.1)
 *  Block 2 (size 80, active): ARMA(2,1), AR = (ar_coef, ar_coef/5), MA = 0.5
 *  Block 3 (size 65, trap):   AR(1),    phi = ar_coef
 *
 *  For each observation row i, an independent ARMA series of length = block_size
 *  is simulated (100-step burn-in), then all columns in the block are
 *  standardised to unit empirical variance (matching R's arima.sim approach).
 *
 *  The ARMA(2,1) block is stationary only for ar_coef < 0.833; column
 *  standardisation suppresses near-non-stationarity at higher ar_coef values.
 *
 * @param n        Number of observations.
 * @param p        Number of predictors (>= 218).
 * @param snr      Target signal-to-noise ratio.
 * @param ar_coef  Shared AR coefficient in (0, 1) for AR/ARMA blocks.
 * @param seed     RNG seed.
 *
 * @return Populated GVSDGPResult (s = 150, group_sizes = {20,50,80,65}).
 */
inline GVSDGPResult make_arma_blocks_dgp(int n, int p, double snr,
                                          double ar_coef, unsigned seed)
{
    const std::vector<int>  block_sizes = {20, 50, 80, 65};
    const std::vector<bool> is_active   = {true, true, true, false};
    const int num_blocks    = 4;
    const int total_grouped = 215;

    if (p < total_grouped + (num_blocks - 1))
        throw std::invalid_argument(
            "make_arma_blocks_dgp: p must be >= 218");

    // ARMA model specs per logical block index
    const std::vector<std::vector<double>> ar_per_block = {
        {ar_coef},                       // Block 0: AR(1)
        {},                              // Block 1: MA(3)
        {ar_coef, ar_coef / 5.0},        // Block 2: ARMA(2,1)
        {ar_coef}                        // Block 3: AR(1) trap
    };
    const std::vector<std::vector<double>> ma_per_block = {
        {},                              // Block 0: AR(1)
        {0.5, 0.3, 0.1},                 // Block 1: MA(3)
        {0.5},                           // Block 2: ARMA(2,1)
        {}                               // Block 3: AR(1) trap
    };

    std::mt19937 rng(seed);
    std::normal_distribution<double> N01(0.0, 1.0);

    // Inline ARMA simulator: returns one series of given length (with burn-in).
    // Captures rng and N01 by reference — not thread-safe but only called in
    // the DGP generator (single-threaded data generation phase).
    auto sim_arma = [&](const std::vector<double>& ar,
                         const std::vector<double>& ma,
                         int length) -> std::vector<double>
    {
        const int p_ar  = static_cast<int>(ar.size());
        const int q_ma  = static_cast<int>(ma.size());
        const int burn  = std::max(100, p_ar + q_ma + 10);
        const int total = burn + length;

        std::vector<double> x(static_cast<std::size_t>(total), 0.0);
        std::vector<double> eps(static_cast<std::size_t>(total), 0.0);

        for (int t = 0; t < total; ++t) {
            const auto tu = static_cast<std::size_t>(t);
            eps[tu] = N01(rng);
            double val = eps[tu];
            for (int k = 0; k < p_ar; ++k)
                if (t - k - 1 >= 0)
                    val += ar[static_cast<std::size_t>(k)] *
                           x[static_cast<std::size_t>(t - k - 1)];
            for (int k = 0; k < q_ma; ++k)
                if (t - k - 1 >= 0)
                    val += ma[static_cast<std::size_t>(k)] *
                           eps[static_cast<std::size_t>(t - k - 1)];
            x[tu] = val;
        }
        return std::vector<double>(x.begin() + burn, x.end());
    };

    // --- Block order ---
    std::vector<int> block_order = {0, 1, 2, 3};
    std::shuffle(block_order.begin(), block_order.end(), rng);

    // --- Gap distribution ---
    const int rem_noise = p - total_grouped - (num_blocks - 1);
    std::vector<int> gaps(static_cast<std::size_t>(num_blocks + 1), 0);
    {
        std::uniform_int_distribution<int> slot_dist(0, num_blocks);
        for (int k = 0; k < rem_noise; ++k)
            gaps[static_cast<std::size_t>(slot_dist(rng))]++;
    }
    for (int i = 1; i < num_blocks; ++i)
        gaps[static_cast<std::size_t>(i)]++;

    GVSDGPResult out;
    out.n    = n;
    out.p    = p;
    out.snr  = snr;
    out.sd_x = 0.0;   // ARMA: no latent-factor sd_x

    out.X.resize(n, p);
    out.beta.setZero(p);
    out.prior_groups.resize(static_cast<std::size_t>(p));

    // --- Fill columns ---
    int col = 0;
    for (int slot = 0; slot <= num_blocks; ++slot) {
        const int g = gaps[static_cast<std::size_t>(slot)];
        for (int k = 0; k < g; ++k, ++col) {
            out.prior_groups[static_cast<std::size_t>(col)] = col;
            for (int i = 0; i < n; ++i)
                out.X(i, col) = N01(rng);
        }
        if (slot == num_blocks) break;

        const int  blk    = block_order[static_cast<std::size_t>(slot)];
        const int  bsize  = block_sizes[static_cast<std::size_t>(blk)];
        const bool active = is_active[static_cast<std::size_t>(blk)];

        // Generate n independent ARMA series of length bsize, column-standardise
        Eigen::MatrixXd X_blk(n, bsize);
        for (int i = 0; i < n; ++i) {
            auto series = sim_arma(ar_per_block[static_cast<std::size_t>(blk)],
                                    ma_per_block[static_cast<std::size_t>(blk)],
                                    bsize);
            for (int k = 0; k < bsize; ++k)
                X_blk(i, k) = series[static_cast<std::size_t>(k)];
        }
        for (int k = 0; k < bsize; ++k) {
            const double mean = X_blk.col(k).mean();
            const double sd   = std::sqrt(
                (X_blk.col(k).array() - mean).square().mean());
            if (sd > 1e-12)
                X_blk.col(k) = (X_blk.col(k).array() - mean) / sd;
        }

        for (int k = 0; k < bsize; ++k, ++col) {
            out.prior_groups[static_cast<std::size_t>(col)] = blk;
            out.X.col(col) = X_blk.col(k);
            if (active) {
                out.beta(col) = 3.0;
                out.true_support.push_back(static_cast<std::size_t>(col));
            }
        }
    }
    std::sort(out.true_support.begin(), out.true_support.end());
    out.s = static_cast<int>(out.true_support.size());  // 150

    // SNR-calibrated noise
    const Eigen::VectorXd signal = out.X * out.beta;
    const double sig_var = (signal.array() - signal.mean()).square().mean();
    out.sigma_y = std::sqrt(sig_var / snr);

    out.y.resize(n);
    for (int i = 0; i < n; ++i)
        out.y(i) = signal(i) + out.sigma_y * N01(rng);

    out.group_sizes  = {20, 50, 80, 65};
    out.group_order  = block_order;
    out.group_active = {true, true, true, false};

    return out;
}


// ==============================================================================
// DGP: Quasi-hapgen LD blocks  (auxiliary; not wired into a numbered demo)
// ==============================================================================

/**
 * @brief Compute the Cholesky factor of the hapgen LD-block correlation matrix.
 *
 * @details
 *  Fixed 7-block LD structure for p = 500 columns.  AR(1) within each block;
 *  two injected cross-block long-range LD pairs:
 *
 *  Block | Cols      | Size | rho_base | Active
 *  ------|-----------|------|----------|-------
 *  B1    |   0– 29   |   30 |     0.85 | yes
 *  B2    |  35–114   |   80 |     0.45 | trap
 *  B3    | 120–179   |   60 |     0.80 | yes
 *  B4    | 185–224   |   40 |     0.90 | yes
 *  B5    | 230–329   |  100 |     0.55 | trap
 *  B6    | 335–384   |   50 |     0.80 | trap
 *  B7    | 390–499   |  110 |     0.30 | trap
 *
 *  Gaps: 5 white-noise columns between consecutive blocks.
 *  Long-range LD (cross-block off-diagonal correlations):
 *    Cols 10–15  ↔  Cols 340–345 : strength min(0.30 * rho_scale, 0.999)
 *    Cols 130–135 ↔ Cols 240–245 : strength min(0.25 * rho_scale, 0.999)
 *
 *  PSD repair: negative eigenvalues clamped to 1e-6, then diagonal rescaled
 *  back to 1 to restore a proper correlation matrix.
 *
 *  Call this function once per rho_scale value and reuse the returned
 *  Cholesky factor across all MC trials via make_hapgen_dgp() to avoid
 *  repeated O(p^3) eigenvalue decompositions.
 *
 * @param rho_scale  Scales all base correlations uniformly in [0, 1].
 * @return  Lower Cholesky factor L (500 × 500) s.t. L * L^T = Sigma_PSD.
 */
inline Eigen::MatrixXd make_hapgen_cholesky(double rho_scale)
{
    constexpr int p = 500;

    struct HapBlock { int start; int sz; double rho_base; };
    const std::vector<HapBlock> hap_blocks = {
        {  0,  30, 0.85},
        { 35,  80, 0.45},
        {120,  60, 0.80},
        {185,  40, 0.90},
        {230, 100, 0.55},
        {335,  50, 0.80},
        {390, 110, 0.30}
    };

    Eigen::MatrixXd Sigma = Eigen::MatrixXd::Identity(p, p);

    // Within-block AR(1) correlations
    for (const auto& b : hap_blocks) {
        const double rho_eff = std::min(b.rho_base * rho_scale, 0.999);
        for (int ii = 0; ii < b.sz; ++ii)
            for (int jj = 0; jj < b.sz; ++jj)
                Sigma(b.start + ii, b.start + jj) =
                    std::pow(rho_eff, static_cast<double>(std::abs(ii - jj)));
    }

    // Long-range cross-block LD
    const double ld1 = std::min(0.30 * rho_scale, 0.999);
    const double ld2 = std::min(0.25 * rho_scale, 0.999);
    for (int i = 10; i <= 15; ++i)
        for (int j = 340; j <= 345; ++j) { Sigma(i, j) = ld1; Sigma(j, i) = ld1; }
    for (int i = 130; i <= 135; ++i)
        for (int j = 240; j <= 245; ++j) { Sigma(i, j) = ld2; Sigma(j, i) = ld2; }

    // PSD repair: clamp negative eigenvalues to 1e-6
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sigma);
    const Eigen::VectorXd evals = es.eigenvalues().cwiseMax(1e-6);
    Sigma = es.eigenvectors() * evals.asDiagonal() *
            es.eigenvectors().transpose();

    // Rescale diagonal back to 1 (restore correlation matrix)
    const Eigen::VectorXd D_inv = Sigma.diagonal().cwiseSqrt().cwiseInverse();
    Sigma = D_inv.asDiagonal() * Sigma * D_inv.asDiagonal();

    return Eigen::LLT<Eigen::MatrixXd>(Sigma).matrixL();
}


/**
 * @brief Quasi-hapgen LD-block DGP using a precomputed Cholesky factor.
 *
 * @details
 *  Fixed p = 500, s = 130 active columns (B1: 0–29, B3: 120–179, B4: 185–224).
 *  X = Z * L^T where Z ~ N(0, I_{n x p}) and L = make_hapgen_cholesky().
 *
 *  Call make_hapgen_cholesky(rho_scale) once per rho_scale value and reuse L
 *  across all MC trials to avoid repeated O(p^3) PSD-repair costs.
 *
 * @param n    Number of observations.
 * @param snr  Target signal-to-noise ratio.
 * @param L    Precomputed Cholesky factor from make_hapgen_cholesky().
 * @param seed RNG seed.
 *
 * @return Populated GVSDGPResult (p = 500, s = 130).
 */
inline GVSDGPResult make_hapgen_dgp(int n, double snr,
                                     const Eigen::MatrixXd& L,
                                     unsigned seed)
{
    constexpr int p = 500;

    std::mt19937 rng(seed);
    std::normal_distribution<double> N01(0.0, 1.0);

    // X = Z * L^T,  Z ~ N(0, I_{n x p})
    Eigen::MatrixXd Z(n, p);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j)
            Z(i, j) = N01(rng);

    GVSDGPResult out;
    out.n    = n;
    out.p    = p;
    out.snr  = snr;
    out.sd_x = 0.0;
    out.X    = Z * L.transpose();
    out.beta.setZero(p);
    out.prior_groups.resize(static_cast<std::size_t>(p));

    // Initialize all columns as singletons
    for (int j = 0; j < p; ++j)
        out.prior_groups[static_cast<std::size_t>(j)] = j;

    // Block group IDs and beta
    struct HapBlockInfo { int start; int sz; bool active; int group_id; };
    const std::vector<HapBlockInfo> block_info = {
        {  0,  30, true,  0},   // B1 — active
        { 35,  80, false, 1},   // B2 — trap
        {120,  60, true,  2},   // B3 — active
        {185,  40, true,  3},   // B4 — active
        {230, 100, false, 4},   // B5 — trap
        {335,  50, false, 5},   // B6 — trap
        {390, 110, false, 6}    // B7 — trap
    };

    for (const auto& b : block_info) {
        for (int k = 0; k < b.sz; ++k) {
            const int j = b.start + k;
            out.prior_groups[static_cast<std::size_t>(j)] = b.group_id;
            if (b.active) {
                out.beta(j) = 3.0;
                out.true_support.push_back(static_cast<std::size_t>(j));
            }
        }
    }
    std::sort(out.true_support.begin(), out.true_support.end());
    out.s = static_cast<int>(out.true_support.size());  // 130

    // SNR-calibrated noise
    const Eigen::VectorXd signal = out.X * out.beta;
    const double sig_var = (signal.array() - signal.mean()).square().mean();
    out.sigma_y = std::sqrt(sig_var / snr);

    out.y.resize(n);
    for (int i = 0; i < n; ++i)
        out.y(i) = signal(i) + out.sigma_y * N01(rng);

    return out;
}


// ==============================================================================
} /* End of namespace gvs_demo */
// ==============================================================================

#endif /* End of DEMOS_TREX_SELECTOR_METHODS_GVS_DGPS_HPP */
