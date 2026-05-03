// ==============================================================================
// dgp_generators.hpp
// ==============================================================================
#ifndef DEMOS_TREX_SELECTOR_METHODS_DA_DGP_GENERATORS_HPP
#define DEMOS_TREX_SELECTOR_METHODS_DA_DGP_GENERATORS_HPP
// ==============================================================================
/**
 * @file dgp_generators.hpp
 *
 * @brief Data generating process (DGP) generators for MC simulations.
 *
 * Provides:
 *   - DGPData               — (X, y, true_support) triplet
 *   - make_beta_at()         — sparse coefficient vector at arbitrary positions
 *   - make_snr_response()    — SNR-calibrated noise
 *   - DGP generators:
 *     - dgp_ar1()            — AR(1) Toeplitz covariance
 *     - dgp_equi()           — Compound symmetry via single latent factor
 *     - dgp_nn()             — Banded covariance via MA(kappa) convolution
 *     - dgp_bt()             — Two-level hierarchical block covariance
 *     - dgp_groups()         — Multi-level latent factor model
 *     - dgp_ar1_block()      — Block-diagonal AR(1)
 *     - dgp_ar1_block_white()— Block-diagonal AR(1) + white-noise block
 *     - dgp_block_toeplitz_hvt() — Block-diagonal Toeplitz (Cholesky, heavy tails)
 *     - dgp_block_equi()     — Block-equicorrelated (Gaussian)
 *   - block_random_support() — helper for block-based support
 */
// ==============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

#include <Eigen/Dense>


// ==============================================================================
namespace da_sim {
// ==============================================================================


// ==============================================================================
// DGP result
// ==============================================================================

/// Container for DGP output: (X, y, true_support)
struct DGPData {
    /// Design matrix (n x p)
    Eigen::MatrixXd X;

    /// Response vector (n)
    Eigen::VectorXd y;

    /// True support indices
    std::vector<std::size_t> true_support;
};


// ==============================================================================
// Support helpers
// ==============================================================================

/**
 * @brief Build block-based support with one active variable per active block,
 *        at a uniformly random position within each block.
 *
 * @param G   Number of active blocks.
 * @param g   Block size.
 * @param rng Random number generator (modified in-place).
 *
 * @return Vector of G support indices, one per active block.
 */
inline std::vector<std::size_t> block_random_support(int G, int g,
                                                     std::mt19937& rng) {
    std::uniform_int_distribution<int> offset(0, g - 1);
    std::vector<std::size_t> sup;
    sup.reserve(static_cast<std::size_t>(G));
    for (int b = 0; b < G; ++b) {
        sup.push_back(static_cast<std::size_t>(b * g + offset(rng)));
    }
    return sup;
}


// ==============================================================================
// DGP helpers
// ==============================================================================

/**
 * @brief Build sparse coefficient vector at arbitrary support positions.
 *
 * @param p         Total number of predictors.
 * @param support   Indices of active predictors.
 * @param amplitude Signal coefficient for active predictors.
 */
inline Eigen::VectorXd make_beta_at(int p,
                                    const std::vector<std::size_t>& support,
                                    double amplitude) {
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    for (auto idx : support) {
        beta(static_cast<Eigen::Index>(idx)) = amplitude;
    }
    return beta;
}


/**
 * @brief Generate y = X*beta + noise with SNR-calibrated noise variance.
 *
 * @param X     Design matrix.
 * @param beta  Coefficient vector.
 * @param snr   Signal-to-noise ratio.
 * @param rng   Random number generator.
 *
 * @return Response vector y.
 */
inline Eigen::VectorXd make_snr_response(const Eigen::MatrixXd& X,
                                         const Eigen::VectorXd& beta,
                                         double snr, std::mt19937& rng) {
    Eigen::VectorXd signal = X * beta;
    const int n = static_cast<int>(X.rows());
    double mean_sig = signal.mean();
    double var_sig  = (signal.array() - mean_sig).square().sum() / (n - 1.0);
    double noise_sigma = std::sqrt(var_sig / snr);

    std::normal_distribution<double> dist(0.0, noise_sigma);
    for (int i = 0; i < n; ++i) {
        signal(i) += dist(rng);
    }

    return signal;
}


// ==============================================================================
// Block-diagonal Toeplitz DGP
// ==============================================================================

/**
 * @brief Block-diagonal Toeplitz DGP.
 *
 * Designs p predictors in p/g blocks of size g.
 * Within each block the covariance is Toeplitz:
 *      Sigma_{block}[i,j] = rho^|i-j|.
 * Between blocks: zero.
 *
 * Active variables: one per active block.  By default at position b*g
 * (first element), or at a uniformly random offset when random_support=true.
 * Response: y = X * beta + noise
 * The noise is calibrated according to target SNR.
 *
 * @param n              Number of observations.
 * @param p              Number of predictors (must be divisible by g).
 * @param G              Number of active groups (blocks with an active variable).
 * @param g              Group (block) size.
 * @param rho            Within-block correlation.
 * @param snr            Signal-to-noise ratio.
 * @param seed           Random seed.
 * @param heavy_X        If true, rows of X follow multivariate t(df).
 * @param heavy_noise    If true, noise is t(df)-distributed.
 * @param df             Degrees of freedom for t-distribution (default 3).
 * @param random_support If true, active position within each block is random.
 *
 * @return DGPData with (X, y, true_support)
 */
inline DGPData dgp_block_toeplitz_hvt(int n,
                                      int p,
                                      int G,
                                      int g,
                                      double rho,
                                      double snr,
                                      unsigned seed,
                                      bool heavy_X,
                                      bool heavy_noise,
                                      double df = 3.0,
                                      bool random_support = false) {
    std::mt19937 rng(seed);
    std::normal_distribution<double>      norm(0.0, 1.0);
    std::chi_squared_distribution<double> chi2(df);
    std::student_t_distribution<double>   t_dist(df);

    const int num_blocks = p / g;

    // Build g x g Toeplitz covariance block
    Eigen::MatrixXd Sigma(g, g);
    for (Eigen::Index i = 0; i < g; ++i) {
        for (Eigen::Index j = 0; j < g; ++j) {
            Sigma(i, j) = std::pow(rho, std::abs(i - j));
        }
    }

    // Build Cholesky factor (shared by all blocks)
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
    Eigen::MatrixXd L = llt.matrixL();

    // Generate X block-wise: X_block = Z_block * L^T
    // If heavy_X: scale each row by sqrt(df / chi2(df)) for multivariate t(df)
    Eigen::MatrixXd X(n, p);
    for (Eigen::Index b = 0; b < num_blocks; ++b) {
        Eigen::MatrixXd Z(n, g);
        for (Eigen::Index i = 0; i < n; ++i) {
            for (Eigen::Index j = 0; j < g; ++j) {
                Z(i, j) = norm(rng);
            }
        }

        X.block(0, b * g, n, g).noalias() =
            Z * L.transpose();

        if (heavy_X) {
            for (Eigen::Index i = 0; i < n; ++i) {
                double u = chi2(rng);
                X.block(i, b * g, 1, g) *=
                    std::sqrt(df / u);
            }
        }
    }

    // Sparse beta: one active per active block
    auto support = random_support ? block_random_support(G, g, rng)
                                  : [&]() {
                                        std::vector<std::size_t> s;
                                        s.reserve(static_cast<std::size_t>(G));
                                        for (int b = 0; b < G; ++b)
                                            s.push_back(static_cast<std::size_t>(b * g));
                                        return s;
                                    }();

    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    for (auto idx : support) {
        beta(static_cast<Eigen::Index>(idx)) = 1.0;
    }

    // Response with SNR-calibrated noise
    Eigen::VectorXd signal = X * beta;
    double mean_sig = signal.mean();
    double var_sig  = (signal.array() - mean_sig).square().sum() / (n - 1.0);
    double target_noise_var = var_sig / snr;

    Eigen::VectorXd y = signal;
    if (heavy_noise) {
        // t(df) noise: Var[t(df)] = df/(df - 2), rescale to match target variance
        double t_var = df / (df - 2.0);
        double scale = std::sqrt(target_noise_var / t_var);
        for (Eigen::Index i = 0; i < n; ++i)
            y(i) += t_dist(rng) * scale;
    } else {
        double noise_sigma = std::sqrt(target_noise_var);
        for (Eigen::Index i = 0; i < n; ++i)
            y(i) += norm(rng) * noise_sigma;
    }

    return {std::move(X), std::move(y), std::move(support)};
}


// ==============================================================================
// Block-equicorrelated DGP (Gaussian)
// ==============================================================================

/**
 * @brief Block-equicorrelated DGP.
 *
 * Designs p predictors in n_blocks blocks.
 * Within each block the covariance is compound symmetry:
 *   Sigma[j,k] = rho (off-diagonal), 1 (diagonal).
 * Between blocks: zero correlation.
 *
 * Constructed via one shared latent factor per block:
 *   x_{i,j} = sqrt(rho) * z_k(i) + sqrt(1 - rho) * w_{i,j}
 *
 * Active variables: one per active block.  By default at position b*g
 * (first element), or at a uniformly random offset when random_support=true.
 * Response: y = X*beta + noise
 * The noise is calibrated according to the target SNR.
 *
 * @param random_support If true, active position within each block is random.
 */
inline DGPData dgp_block_equi(int n,
                              int p,
                              int G,
                              int g,
                              double rho,
                              double snr,
                              unsigned seed,
                              bool random_support = false) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    const int num_blocks = p / g;
    const double load_f   = std::sqrt(rho);
    const double load_eta = std::sqrt(1.0 - rho);

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index b = 0; b < num_blocks; ++b) {
        // One shared factor per block
        Eigen::VectorXd z_k(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            z_k(i) = norm(rng);
        }

        for (Eigen::Index j = 0; j < g; ++j) {
            const Eigen::Index col = b * g + j;
            for (Eigen::Index i = 0; i < n; ++i) {
                X(i, col) = load_f * z_k(i) + load_eta * norm(rng);
            }
        }
    }

    // Sparse beta: one active per active block
    auto support = random_support ? block_random_support(G, g, rng)
                                  : [&]() {
                                        std::vector<std::size_t> s;
                                        s.reserve(static_cast<std::size_t>(G));
                                        for (int b = 0; b < G; ++b)
                                            s.push_back(static_cast<std::size_t>(b * g));
                                        return s;
                                    }();

    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    for (auto idx : support) {
        beta(static_cast<Eigen::Index>(idx)) = 1.0;
    }

    // Response with SNR-calibrated noise
    Eigen::VectorXd signal = X * beta;
    double mean_sig = signal.mean();
    double var_sig  = (signal.array() - mean_sig).square().sum() / (n - 1.0);
    double noise_sigma = std::sqrt(var_sig / snr);

    Eigen::VectorXd y = signal;
    for (Eigen::Index i = 0; i < n; ++i) {
        y(i) += norm(rng) * noise_sigma;
    }

    return {std::move(X), std::move(y), std::move(support)};
}


// ==============================================================================
// AR(1) Toeplitz DGP
// ==============================================================================

/**
 * @brief AR(1) Toeplitz covariance: Sigma[j,k] = rho^|j-k|.
 *
 * X rows generated via: X[i,j] = rho * X[i,j-1] + sqrt(1-rho^2) * eta[i,j].
 * Caller provides the support vector (active variable indices).
 *
 * @param n              Number of observations.
 * @param p              Number of predictors.
 * @param support        Indices of active predictors.
 * @param amplitude      Signal coefficient for active predictors.
 * @param rho            AR(1) correlation parameter.
 * @param snr            Signal-to-noise ratio.
 * @param seed           Random seed.
 *
 * @return DGPData with (X, y, true_support).
 */
inline DGPData dgp_ar1(int n,
                       int p,
                       const std::vector<std::size_t>& support,
                       double amplitude,
                       double rho,
                       double snr,
                       unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    Eigen::MatrixXd X(n, p);
    const double scale = std::sqrt(1.0 - rho * rho);
    for (Eigen::Index i = 0; i < n; ++i) {
        X(i, 0) = norm(rng);
        for (Eigen::Index j = 1; j < p; ++j) {
            X(i, j) = rho * X(i, j - 1) + scale * norm(rng);
        }
    }

    auto beta    = make_beta_at(p, support, amplitude);
    auto y       = make_snr_response(X, beta, snr, rng);
    return {std::move(X), std::move(y), support};
}


// ==============================================================================
// Equicorrelation DGP
// ==============================================================================

/**
 * @brief Compound symmetry via single latent factor:
 *        X[i,j] = sqrt(rho) * f[i] + sqrt(1 - rho) * eta[i,j].
 *
 * Caller provides the support vector (active variable indices).
 *
 * @param n            Number of observations.
 * @param p            Number of predictors.
 * @param support      Indices of active predictors.
 * @param amplitude    Signal coefficient for active predictors.
 * @param rho          Equicorrelation parameter (0 <= rho < 1).
 * @param snr          Signal-to-noise ratio.
 * @param seed         Random seed.
 *
 * @return DGPData with (X, y, true_support).
 */
inline DGPData dgp_equi(int n,
                        int p,
                        const std::vector<std::size_t>& support,
                        double amplitude,
                        double rho,
                        double snr,
                        unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    const double load_f   = std::sqrt(rho);
    const double load_eta = std::sqrt(1.0 - rho);

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        const double f_i = norm(rng);
        for (Eigen::Index j = 0; j < p; ++j) {
            X(i, j) = load_f * f_i + load_eta * norm(rng);
        }
    }

    auto beta    = make_beta_at(p, support, amplitude);
    auto y       = make_snr_response(X, beta, snr, rng);
    return {std::move(X), std::move(y), support};
}


// ==============================================================================
// Nearest Neighbours (MA(kappa)) DGP
// ==============================================================================

/**
 * @brief Banded covariance via MA(kappa) convolution:
 *        X[i,j] = sum_{l=0}^{kappa} theta[l] * eta[i, j+l],
 *        with theta[l] = c * rho^l (normalised to unit variance).
 *
 * Caller provides the support vector (active variable indices).
 *
 * @param n            Number of observations.
 * @param p            Number of predictors.
 * @param support      Indices of active predictors.
 * @param amplitude    Signal coefficient for active predictors.
 * @param kappa        MA order (bandwidth).
 * @param rho          Decay parameter for MA coefficients (theta[l] ~ rho^l).
 * @param snr          Signal-to-noise ratio.
 * @param seed         Random seed.
 *
 * @return DGPData with (X, y, true_support).
 */
inline DGPData dgp_nn(int n,
                      int p,
                      const std::vector<std::size_t>& support,
                      double amplitude,
                      int kappa,
                      double rho,
                      double snr,
                      unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    std::vector<double> theta(static_cast<std::size_t>(kappa + 1));
    double sum_sq = 0.0;
    for (int l = 0; l <= kappa; ++l) {
        theta[static_cast<std::size_t>(l)] = std::pow(rho, l);
        sum_sq += theta[static_cast<std::size_t>(l)]
                * theta[static_cast<std::size_t>(l)];
    }
    const double c = 1.0 / std::sqrt(sum_sq);
    for (auto& t : theta) t *= c;

    const Eigen::Index ext_cols = static_cast<Eigen::Index>(p) + kappa;
    Eigen::MatrixXd eta(n, ext_cols);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < ext_cols; ++j) {
            eta(i, j) = norm(rng);
        }
    }

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index j = 0; j < p; ++j) {
        X.col(j).setZero();
        for (int l = 0; l <= kappa; ++l) {
            X.col(j) += theta[static_cast<std::size_t>(l)] * eta.col(j + l);
        }
    }

    auto beta    = make_beta_at(p, support, amplitude);
    auto y       = make_snr_response(X, beta, snr, rng);
    return {std::move(X), std::move(y), support};
}


// ==============================================================================
// Binary Tree (hierarchical blocks) DGP
// ==============================================================================

/**
 * @brief Two-level block covariance:
 *   X[i,j] = sqrt(rho_b) * f0
 *            + sqrt(rho_w - rho_b) * f_block
 *            + sqrt(1 - rho_w) * eta.
 *
 * Caller provides the support vector (active variable indices).
 *
 * @param n              Number of observations.
 * @param p              Number of predictors (must be divisible by n_blocks).
 * @param support        Indices of active predictors.
 * @param amplitude      Signal coefficient for active predictors.
 * @param n_blocks       Number of blocks (p must be divisible by n_blocks).
 * @param rho_within     Within-block correlation (must satisfy rho_within > rho_between).
 * @param rho_between    Between-block correlation.
 * @param snr            Signal-to-noise ratio.
 * @param seed           Random seed.
 *
 * @return DGPData with (X, y, true_support).
 */
inline DGPData dgp_bt(int n,
                      int p,
                      const std::vector<std::size_t>& support,
                      double amplitude,
                      int n_blocks,
                      double rho_within,
                      double rho_between,
                      double snr,
                      unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    const Eigen::Index block_size = static_cast<Eigen::Index>(p) / n_blocks;
    const double a  = std::sqrt(rho_between);
    const double b  = std::sqrt(rho_within - rho_between);
    const double cc = std::sqrt(1.0 - rho_within);

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        const double f0 = norm(rng);
        std::vector<double> f_block(static_cast<std::size_t>(n_blocks));
        for (int m = 0; m < n_blocks; ++m)
            f_block[static_cast<std::size_t>(m)] = norm(rng);

        for (Eigen::Index j = 0; j < p; ++j) {
            const int bid = static_cast<int>(j / block_size);
            X(i, j) = a * f0
                     + b * f_block[static_cast<std::size_t>(bid)]
                     + cc * norm(rng);
        }
    }

    auto beta = make_beta_at(p, support, amplitude);
    auto y    = make_snr_response(X, beta, snr, rng);
    return {std::move(X), std::move(y), support};
}


// ==============================================================================
// Prior Groups (multi-level latent factor) DGP
// ==============================================================================

/**
 * @brief Multi-level latent model:
 *   X[i,j] = sum_x sqrt(rho[x] - rho[x+1]) * f_{g_x(j)} + sqrt(1 - rho[0]) * eta.
 *
 * Caller provides the support vector (active variable indices).
 *
 * @param n            Number of observations.
 * @param p            Number of predictors.
 * @param support      Indices of active predictors.
 * @param amplitude    Signal coefficient for active predictors.
 * @param groups       L vectors of length p (group labels, fine to coarse).
 * @param rho_levels   L strictly decreasing correlation levels in (0, 1).
 * @param snr          Signal-to-noise ratio.
 * @param seed         Random seed.
 *
 * @return DGPData with (X, y, true_support).
 */
inline DGPData dgp_groups(int n,
                          int p,
                          const std::vector<std::size_t>& support,
                          double amplitude,
                          const std::vector<std::vector<Eigen::Index>>& groups,
                          const std::vector<double>& rho_levels,
                          double snr,
                          unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    const auto L = groups.size();
    std::vector<double> rho_ext(rho_levels.begin(), rho_levels.end());
    rho_ext.push_back(0.0);

    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(n, p);

    for (std::size_t x = 0; x < L; ++x) {
        const double loading = std::sqrt(rho_ext[x] - rho_ext[x + 1]);
        std::vector<Eigen::Index> unique_ids(groups[x].begin(), groups[x].end());
        std::sort(unique_ids.begin(), unique_ids.end());
        unique_ids.erase(std::unique(unique_ids.begin(), unique_ids.end()),
                         unique_ids.end());

        for (auto gid : unique_ids) {
            Eigen::VectorXd f(n);
            for (Eigen::Index i = 0; i < n; ++i) {
                f(i) = norm(rng);
            }
            for (Eigen::Index j = 0; j < p; ++j) {
                if (groups[x][static_cast<std::size_t>(j)] == gid) {
                    X.col(j) += loading * f;
                }
            }
        }
    }

    const double eta_scale = std::sqrt(1.0 - rho_levels[0]);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) {
            X(i, j) += eta_scale * norm(rng);
        }
    }


    auto beta    = make_beta_at(p, support, amplitude);
    auto y       = make_snr_response(X, beta, snr, rng);

    return {std::move(X), std::move(y), support};
}


// ==============================================================================
// Block-diagonal AR(1) DGP
// ==============================================================================

/**
 * @brief Block-diagonal AR(1) covariance.
 *
 * Partitions p predictors into n_blocks independent blocks of equal size.
 * Within each block, columns follow an AR(1) recursion:
 *   X[i, col_start+j] = rho * X[i, col_start+j-1] + sqrt(1-rho^2) * eta.
 * Between blocks: zero correlation.
 *
 * Caller provides the support vector (active variable indices).
 *
 * @param n          Number of observations.
 * @param p          Number of predictors (must be divisible by n_blocks).
 * @param support    Indices of active predictors.
 * @param amplitude  Signal coefficient for active predictors.
 * @param n_blocks   Number of blocks.
 * @param rho        AR(1) correlation parameter (|rho| < 1).
 * @param snr        Signal-to-noise ratio.
 * @param seed       Random seed.
 *
 * @return DGPData with (X, y, true_support).
 */
inline DGPData dgp_ar1_block(int n,
                              int p,
                              const std::vector<std::size_t>& support,
                              double amplitude,
                              int n_blocks,
                              double rho,
                              double snr,
                              unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    const Eigen::Index block_size = static_cast<Eigen::Index>(p) / n_blocks;
    const double innov_sd = std::sqrt(1.0 - rho * rho);

    Eigen::MatrixXd X(n, p);
    for (int m = 0; m < n_blocks; ++m) {
        const Eigen::Index col_start =
            static_cast<Eigen::Index>(m) * block_size;

        // First column of block
        for (Eigen::Index i = 0; i < n; ++i) {
            X(i, col_start) = norm(rng);
        }

        // AR(1) recursion within block (column-major, matching R reference)
        for (Eigen::Index j = col_start + 1; j < col_start + block_size; ++j) {
            for (Eigen::Index i = 0; i < n; ++i) {
                X(i, j) = rho * X(i, j - 1) + innov_sd * norm(rng);
            }
        }
    }

    auto beta = make_beta_at(p, support, amplitude);
    auto y    = make_snr_response(X, beta, snr, rng);
    return {std::move(X), std::move(y), support};
}


// ==============================================================================
// Block-diagonal AR(1) + white-noise block DGP
// ==============================================================================

/**
 * @brief Block-diagonal AR(1) with an appended white-noise block.
 *
 * Constructs p_ar predictors in n_blocks AR(1) blocks (as in dgp_ar1_block),
 * then appends p_white i.i.d. N(0,1) columns.  Active predictors are
 * restricted to the AR part (indices 0..p_ar-1).  Total dimension: p_ar + p_white.
 *
 * @param n          Number of observations.
 * @param p_ar       Number of AR(1) predictors (must be divisible by n_blocks).
 * @param p_white    Number of i.i.d. white-noise predictors.
 * @param support    Indices of active predictors (must be < p_ar).
 * @param amplitude  Signal coefficient for active predictors.
 * @param n_blocks   Number of AR(1) blocks.
 * @param rho        AR(1) correlation parameter (|rho| < 1).
 * @param snr        Signal-to-noise ratio.
 * @param seed       Random seed.
 *
 * @return DGPData with (X, y, true_support); X has p_ar + p_white columns.
 */
inline DGPData dgp_ar1_block_white(int n,
                                   int p_ar,
                                   int p_white,
                                   const std::vector<std::size_t>& support,
                                   double amplitude,
                                   int n_blocks,
                                   double rho,
                                   double snr,
                                   unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    const int p = p_ar + p_white;
    const Eigen::Index block_size = static_cast<Eigen::Index>(p_ar) / n_blocks;
    const double innov_sd = std::sqrt(1.0 - rho * rho);

    Eigen::MatrixXd X(n, p);

    // AR(1) blocks
    for (int m = 0; m < n_blocks; ++m) {
        const Eigen::Index col_start =
            static_cast<Eigen::Index>(m) * block_size;

        for (Eigen::Index i = 0; i < n; ++i) {
            X(i, col_start) = norm(rng);
        }

        for (Eigen::Index j = col_start + 1; j < col_start + block_size; ++j) {
            for (Eigen::Index i = 0; i < n; ++i) {
                X(i, j) = rho * X(i, j - 1) + innov_sd * norm(rng);
            }
        }
    }

    // White-noise block: i.i.d. N(0,1)
    for (Eigen::Index j = static_cast<Eigen::Index>(p_ar); j < p; ++j) {
        for (Eigen::Index i = 0; i < n; ++i) {
            X(i, j) = norm(rng);
        }
    }

    auto beta = make_beta_at(p, support, amplitude);
    auto y    = make_snr_response(X, beta, snr, rng);
    return {std::move(X), std::move(y), support};
}


// ==============================================================================
// Block-diagonal heavy-tailed AR(1) + white-noise block DGP
// ==============================================================================

/**
 * @brief Heavy-tailed AR(1)-block + white-noise DGP.
 *
 * Constructs p_ar predictors in n_blocks AR(1)-Toeplitz multivariate Student-t
 * blocks (X is always heavy-tailed), then appends p_white i.i.d. Student-t(nu)
 * white-noise columns.  Active predictors are restricted to the AR part
 * (indices 0..p_ar-1).  Total dimension: p_ar + p_white.
 *
 * Noise is Gaussian when heavy_noise=false, or t(nu)-scaled when heavy_noise=true.
 *
 * @param n           Number of observations.
 * @param p_ar        Number of heavy-tailed AR-block predictors (divisible by n_blocks).
 * @param p_white     Number of i.i.d. Student-t white-noise predictors.
 * @param support     Indices of active predictors (must be < p_ar).
 * @param amplitude   Signal coefficient for active predictors.
 * @param n_blocks    Number of AR(1) blocks.
 * @param rho         AR(1) within-block correlation parameter (|rho| < 1).
 * @param snr         Signal-to-noise ratio.
 * @param seed        Random seed.
 * @param heavy_noise If true, noise is t(nu)-distributed; otherwise Gaussian.
 * @param nu          Degrees of freedom for Student-t distributions (default 3.0).
 *
 * @return DGPData with (X, y, true_support); X has p_ar + p_white columns.
 */
inline DGPData dgp_ht_block_white(int n,
                                   int p_ar,
                                   int p_white,
                                   const std::vector<std::size_t>& support,
                                   double amplitude,
                                   int n_blocks,
                                   double rho,
                                   double snr,
                                   unsigned seed,
                                   bool heavy_noise = false,
                                   double nu = 3.0)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double>      norm(0.0, 1.0);
    std::chi_squared_distribution<double> chi2(nu);
    std::student_t_distribution<double>   t_dist(nu);

    const int p = p_ar + p_white;
    const Eigen::Index block_size = static_cast<Eigen::Index>(p_ar) / n_blocks;

    // Build AR(1) Toeplitz covariance for one block
    Eigen::MatrixXd Sigma(block_size, block_size);
    for (Eigen::Index i = 0; i < block_size; ++i) {
        for (Eigen::Index j = 0; j < block_size; ++j) {
            Sigma(i, j) = std::pow(rho, std::abs(i - j));
        }
    }
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
    Eigen::MatrixXd L = llt.matrixL();

    Eigen::MatrixXd X(n, p);

    // Heavy-tailed AR(1) blocks (multivariate t(nu))
    for (int m = 0; m < n_blocks; ++m) {
        const Eigen::Index col_start = static_cast<Eigen::Index>(m) * block_size;

        Eigen::MatrixXd Z(n, block_size);
        for (Eigen::Index i = 0; i < n; ++i) {
            for (Eigen::Index j = 0; j < block_size; ++j) {
                Z(i, j) = norm(rng);
            }
        }

        X.block(0, col_start, n, block_size).noalias() = Z * L.transpose();

        // Scale each row by sqrt(nu / chi2(nu)) for multivariate t(nu)
        for (Eigen::Index i = 0; i < n; ++i) {
            double u = chi2(rng);
            X.block(i, col_start, 1, block_size) *= std::sqrt(nu / u);
        }
    }

    // White-noise block: i.i.d. t(nu) columns
    for (Eigen::Index j = static_cast<Eigen::Index>(p_ar); j < p; ++j) {
        for (Eigen::Index i = 0; i < n; ++i) {
            X(i, j) = t_dist(rng);
        }
    }

    auto beta = make_beta_at(p, support, amplitude);

    // SNR-calibrated response
    Eigen::VectorXd signal = X * beta;
    const double n_d = static_cast<double>(n);
    double mean_sig = signal.mean();
    double var_sig  = (signal.array() - mean_sig).square().sum() / (n_d - 1.0);
    double target_noise_var = var_sig / snr;

    Eigen::VectorXd y = signal;
    if (heavy_noise) {
        // t(nu) noise: Var[t(nu)] = nu/(nu-2); rescale to match target variance
        double t_var = nu / (nu - 2.0);
        double scale = std::sqrt(target_noise_var / t_var);
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n); ++i) {
            y(i) += t_dist(rng) * scale;
        }
    } else {
        double noise_sigma = std::sqrt(target_noise_var);
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n); ++i) {
            y(i) += norm(rng) * noise_sigma;
        }
    }

    return {std::move(X), std::move(y), support};
}


// ==============================================================================
} /* End of namespace da_sim */
// ==============================================================================
#endif /* End of DEMOS_TREX_SELECTOR_METHODS_DA_DGP_GENERATORS_HPP */
