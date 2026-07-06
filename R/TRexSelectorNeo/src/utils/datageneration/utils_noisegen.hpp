// ===================================================================================
// utils_noisegen.hpp
// ===================================================================================
#ifndef UTILS_DATAGEN_NOISEGEN_HPP
#define UTILS_DATAGEN_NOISEGEN_HPP
// ===================================================================================
/**
 * @file utils_noisegen.hpp
 *
 * @brief Utilities for generating and scaling additive noise for synthetic regression.
 *
 * @details Encapsulates the mathematics for strictly controlling the Signal-to-Noise
 * Ratio (SNR) across multiple noise distributions, including heavy-tailed
 * and bi-modal mixture distributions.
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

// Eigen includes
#include <Eigen/Dense>

// dummygen includes
#include <utils/datageneration/utils_dummygen.hpp>

// ===================================================================================

// Embedded into namespace trex::utils::datageneration
namespace trex::utils::datageneration::noisegen {

// ===================================================================================

namespace noise_policy {

    /** @brief Policy for iid Normal noise. */
    struct Normal {};

    /** @brief Policy for heavy-tailed Student-T noise. */
    struct StudentT {
        double df;
        explicit StudentT(double df = 5.0) : df(df) {
            if (df <= 0.0) { throw std::invalid_argument("df must be > 0"); }
        }
    };

    /** @brief Policy for AR(1) temporally correlated noise. */
    struct AR1 {
        double rho;
        explicit AR1(double rho = 0.5) : rho(rho) {
            if (std::abs(rho) >= 1.0) {
                throw std::invalid_argument("AR(1) rho must be in (-1, 1).");
            }
        }
    };

    /** @brief Policy for symmetric bi-modal Normal mixture noise. */
    struct NormalMixture {
        double separation;
        explicit NormalMixture(double sep = 2.0) : separation(sep) {}
    };

    /** @brief Policy for symmetric bi-modal heavy-tailed Student-T mixture noise. */
    struct StudentTMixture {
        double df;
        double separation;
        explicit StudentTMixture(double df = 5.0, double sep = 2.0) : df(df), separation(sep) {
            if (df <= 0.0) { throw std::invalid_argument("df must be > 0"); }
        }
    };

} /* End of namespace noise_policy */
// ==============================================================================


namespace detail {

/**
 * @brief Calculate noise parameters (signal power and noise standard deviation) based on SNR.
 *
 * @param y Response vector (signal).
 * @param n Number of observations.
 * @param snr Signal-to-noise ratio (linear scale).
 */
template<typename VectorType>
inline std::pair<double, double> calculate_noise_params(
    const VectorType& y,
    std::size_t n,
    double snr
) {
    double mean_y = y.mean();
    double var_sum = (y.array() - mean_y).square().sum();
    double denom = (n <= 100) ? static_cast<double>(std::max<std::size_t>(n, 2) - 1)
                              : static_cast<double>(n);
    double signal_power = var_sum / denom;

    double noise_std = 0.0;
    if (snr > 0 && signal_power > 0) {
        noise_std = std::sqrt(signal_power / snr);
    }
    return {signal_power, noise_std};

}

} /* End of namespace detail */
// ==============================================================================


// ==============================================================================
// Noise Generators as overloaded functions based on noise policy
// ==============================================================================

/**
 * @brief Add noise to a vector based on the specified noise policy.
 *
 * @tparam VectorType Type of the vector.
 *
 * @param y Vector to which noise will be added.
 * @param n Number of elements in the vector.
 * @param noise_std Standard deviation of the noise.
 * @param noise_seed Seed for the random number generator.
 */
template <typename VectorType>
inline void add_noise(
    VectorType& y,
    std::size_t n,
    double noise_std,
    const noise_policy::Normal&,
    unsigned int noise_seed
) {
    if (noise_std < 0) {
        throw std::invalid_argument("Noise standard deviation must be non-negative.");
    }
    if (noise_std == 0) { return; } // Noiseless / null-model data: nothing to add
    std::mt19937 noise_gen(noise_seed);
    std::normal_distribution<double> noise_dist(0.0, noise_std);
    for (std::size_t i = 0; i < n; ++i) { y(i) += noise_dist(noise_gen); }
}


/**
 * @brief Add noise to a vector based on the specified noise policy.
 *
 * @tparam VectorType Type of the vector.
 *
 * @param y Vector to which noise will be added.
 * @param n Number of elements in the vector.
 * @param noise_std Standard deviation of the noise.
 * @param policy Noise policy specifying the parameters of the Student-T distribution.
 * @param noise_seed Seed for the random number generator.
 */
template <typename VectorType>
inline void add_noise(
    VectorType& y,
    std::size_t n,
    double noise_std,
    const noise_policy::StudentT& policy,
    unsigned int noise_seed
) {
    if (noise_std < 0) {
        throw std::invalid_argument("Noise standard deviation must be non-negative.");
    }
    if (noise_std == 0) { return; } // Noiseless / null-model data: nothing to add
    std::mt19937 noise_gen(noise_seed);

    std::student_t_distribution<double> noise_dist(policy.df);
    double base_scale = (policy.df > 2.0) ? std::sqrt((policy.df - 2.0) / policy.df) : 1.0;
    double scale = (policy.df > 2.0) ? noise_std * base_scale : noise_std;
    for (std::size_t i = 0; i < n; ++i) { y(i) += scale * noise_dist(noise_gen); }
}


/**
 * @brief Add noise to a vector based on the specified noise policy.
 *
 * @tparam VectorType Type of the vector.
 *
 * @param y Vector to which noise will be added.
 * @param n Number of elements in the vector.
 * @param noise_std Standard deviation of the noise.
 * @param policy Noise policy specifying the parameters of the AR1 process.
 * @param noise_seed Seed for the random number generator.
 */
template <typename VectorType>
inline void add_noise(
    VectorType& y,
    std::size_t n,
    double noise_std,
    const noise_policy::AR1& policy,
    unsigned int noise_seed
) {
    if (noise_std < 0) {
        throw std::invalid_argument("Noise standard deviation must be non-negative.");
    }
    if (noise_std == 0) { return; } // Noiseless / null-model data: nothing to add
    // Stationary AR(1): |rho| < 1 is enforced by the policy constructor.
    // Innovations are scaled so the marginal variance equals noise_std^2, and
    // the initial value is drawn from the stationary marginal distribution.
    double innovation_std =
        noise_std * std::sqrt(std::max(0.0, 1.0 - policy.rho * policy.rho));

    std::mt19937 noise_gen(noise_seed);
    std::normal_distribution<double> innovation_dist(0.0, innovation_std);

    double current_val = std::normal_distribution<double>(0.0, noise_std)(noise_gen);

    y(0) += current_val;
    for (std::size_t i = 1; i < n; ++i) {
        current_val = policy.rho * current_val + innovation_dist(noise_gen);
        y(i) += current_val;
    }
}

/**
 * @brief Add noise to a vector based on the specified noise policy.
 *
 * @tparam VectorType Type of the vector.
 *
 * @param y Vector to which noise will be added.
 * @param n Number of elements in the vector.
 * @param noise_std Standard deviation of the noise.
 * @param policy Noise policy specifying the parameters of the normal mixture.
 * @param noise_seed Seed for the random number generator.
 */
template <typename VectorType>
inline void add_noise(
    VectorType& y,
    std::size_t n,
    double noise_std,
    const noise_policy::NormalMixture& policy,
    unsigned int noise_seed
) {
    if (noise_std < 0) {
        throw std::invalid_argument("Noise standard deviation must be non-negative.");
    }
    if (noise_std == 0) { return; } // Noiseless / null-model data: nothing to add

    std::mt19937 noise_gen(noise_seed);
    std::normal_distribution<double> norm_dist(0.0, 1.0);
    std::bernoulli_distribution coin(0.5);
    double scale = noise_std / std::sqrt(policy.separation * policy.separation + 1.0);
    for (std::size_t i = 0; i < n; ++i) {
        double mode = coin(noise_gen) ? policy.separation : -policy.separation;
        y(i) += scale * (mode + norm_dist(noise_gen));
    }
}

/**
 * @brief Add noise to a vector based on the specified noise policy.
 *
 * @tparam VectorType Type of the vector.
 *
 * @param y Vector to which noise will be added.
 * @param n Number of elements in the vector.
 * @param noise_std Standard deviation of the noise.
 * @param policy Noise policy specifying the parameters of the Student-T mixture.
 * @param noise_seed Seed for the random number generator.
 */
template <typename VectorType>
inline void add_noise(
    VectorType& y,
    std::size_t n,
    double noise_std,
    const noise_policy::StudentTMixture& policy,
    unsigned int noise_seed
) {
    if (noise_std < 0) {
        throw std::invalid_argument("Noise standard deviation must be non-negative.");
    }
    if (noise_std == 0) { return; } // Noiseless / null-model data: nothing to add

    std::mt19937 noise_gen(noise_seed);
    std::student_t_distribution<double> t_dist(policy.df);

    std::bernoulli_distribution coin(0.5);
    double t_scale = (policy.df > 2.0) ? std::sqrt((policy.df - 2.0) / policy.df) : 1.0;
    double scale = noise_std / std::sqrt(policy.separation * policy.separation + 1.0);

    for (std::size_t i = 0; i < n; ++i) {
        double mode = coin(noise_gen) ? policy.separation : -policy.separation;
        y(i) += scale * (mode + (t_dist(noise_gen) * t_scale));
    }
}
// ===================================================================================
} /* End of namespace trex::utils::datageneration::noisegen */

#endif /* UTILS_DATAGEN_NOISEGEN_HPP */
