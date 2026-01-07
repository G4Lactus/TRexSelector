// ===================================================================================
// utils_dummygen.hpp
// ===================================================================================
/**
 * @file utils_dummygen.hpp
 *
 * @brief Dummy variable generation with multiple distributions for T-Solvers.
 *
 * @details Provides distribution types and parallel generation functions for
 *          dummy varibale generation and augmentation of feature matrices.
 */
// ===================================================================================

#ifndef TREX_UTILS_DATAGEN_DUMMYGEN_HPP
#define TREX_UTILS_DATAGEN_DUMMYGEN_HPP

// ===================================================================================

#include <cmath>
#include <random>
#include <stdexcept>
#include <omp.h>

#include <Eigen/Dense>

#include <boost/random/laplace_distribution.hpp>
#include <boost/math/distributions/holtsmark.hpp>

// ===================================================================================

namespace trex {
namespace utils {
namespace dummygen {

// ============================================
// Unified Distribution Configuration
// ============================================

/**
 * @brief Configuration for dummmy variable distribution.
 *
 * @details Administration structure to specify distribution type and parameters.
 *
 */
struct Distribution {

    // ============================================
    // Enumeration for distribution types
    // ============================================
    /**
     * @brief Distribution type enumeration.
     */
    enum class Type {
        /** @brief Standard Normal distribution N(0, 1). */
        Normal,

        /** @brief Uniform distribution U(-a, a) with variance a^2 / 3. */
        Uniform,

        /** @brief Rademacher distribution P(X = -1) = P(X = 1) = 0.5. */
        Rademacher,

        /** @brief Student's t-distribution with df degrees of freedom. */
        StudentT,

        /** @brief Laplace distribution with scale parameter b. */
        Laplace,

        /** @brief Gumbel distribution (Extreme value distribution). */
        Gumbel,

        /** @brief Holtsmark distribution (heavy-tailed, stable with alpha=1.5). */
        Holtsmark
    };


    // ============================================
    // Core member: distribution type
    // ============================================
    /** @brief Distribution type (Default: Normal). */
    Type type = Type::Normal;


    // ============================================
    // Distribution-specific parameters
    // ============================================

    /** @brief Uniform: half-range parameter (default: sqrt(3)). */
    double uniform_a = std::sqrt(3.0);

    /** @brief Student's t-distribution: degrees of freedom (default: 5.0). */
    double student_t_df = 5.0;

    /** @brief Laplace distribution: location parameter (default: 0.0). */
    double laplace_location = 0.0;

    /** @brief Laplace distribution: scale parameter (default: 1/sqrt(2)). */
    double laplace_scale = 1.0 / std::sqrt(2.0);

    /** @brief Gumbel distribution: location parameter (default: 0.0). */
    double gumbel_location = 0.0;

    /** @brief Gumbel distribution: scale parameter (default: 1.0). */
    double gumbel_scale = 1.0;

    /** @brief Holtsmark distribution: location parameter (default: 0.0). */
    double holtsmark_location = 0.0;

    /** @brief Holtsmark distribution: scale parameter (default: 1.0). */
    double holtsmark_scale = 1.0;


    // ============================================
    // Constructors
    // ============================================

    /** @brief Default constructor (Normal distribution) */
    Distribution() : type(Type::Normal) {};

    /** Construct with specific type, using default paramters */
    explicit Distribution(Type t): type(t) {};

    // ============================================
    // Named constructors (factory pattern)
    // ============================================

    /**
     * @brief Create standard normal N(0, 1) distribution.
     *
     * @return Distribution instance for Normal distribution.
     */
    static Distribution Normal() {
        return Distribution(Type::Normal);
    }

    /**
     * @brief Create uniform U(-a, a) distribution.
     *
     * @param a Half-range parameter (default: sqrt(3)).
     *
     * @return Distribution instance for Uniform distribution.
     */
    static Distribution Uniform(double a = std::sqrt(3.0)) {
        Distribution dist(Type::Uniform);
        dist.uniform_a = a;
        return dist;
    }

    /**
     * @brief Create Rademacher distribution {-1, +1} distribution.
     *
     * @return Distribution instance for Rademacher distribution.
     */
    static Distribution Rademacher() {
        return Distribution(Type::Rademacher);
    }

    /**
     * @brief Create Student's t-distribution with df degrees of freedom.
     *
     * @param df Degrees of freedom (default: 5.0).
     *
     * @return Distribution instance for Student's t-distribution.
     */
    static Distribution StudentT(double df = 5.0) {
        Distribution dist(Type::StudentT);
        dist.student_t_df = df;
        return dist;
    }

    /**
     * @brief Create Laplace distribution with scale b.
     *
     * @param location Location parameter (default: 0.0).
     * @param scale Scale parameter (default: 1/sqrt(2)).
     *
     * @return Distribution instance for Laplace distribution.
     */
    static Distribution Laplace(double location = 0.0, double scale = 1.0 / std::sqrt(2.0)) {
        Distribution dist(Type::Laplace);
        dist.laplace_location = location;
        dist.laplace_scale = scale;
        return dist;
    }

    /**
     * @brief Create Gumbel distribution.
     *
     * @param location  Location parameter (default: 0.0).
     *
     * @return Distribution instance for Gumbel distribution.
     */
    static Distribution Gumbel(double location = 0.0, double scale = 1.0) {
        Distribution dist(Type::Gumbel);
        dist.gumbel_location = location;
        dist.gumbel_scale = scale;
        return dist;
    }

    /**
     * @brief Create Holtsmark distribution.
     *
     * @param location Location parameter (default: 0.0)
     * @param scale Scale parameter (default: 1.0)
     *
     * @return Distribution instance for Holtsmark distribution.
     */
    static Distribution Holtsmark(double location = 0.0, double scale = 1.0) {
        Distribution dist(Type::Holtsmark);
        dist.holtsmark_location = location;
        dist.holtsmark_scale = scale;
        return dist;
    }

    // ============================================
    // Accessors
    // ============================================

    /** @brief Get distribution type. */
    Type get_type() const { return type; }
};


// ============================================
// Core Functionalities
// ============================================

/**
 * @brief Generate dummy variables with specified distribution.
 *
 * @tparam MatrixType Type of the matrix (e.g., Eigen::MatrixXd).
 *
 * @param X Reference to the matrix to be filled with dummies.
 * @param n Number of rows.
 * @param p Number of columns (number of dummies).
 * @param base_seed Base seed for random number generation.
 * @param dist_type Distribution configuration.
 */
template<typename MatrixType>
inline void generate_dummies(
    MatrixType& X,
    std::size_t n,
    std::size_t p,
    unsigned int base_seed,
    const Distribution& dist = Distribution::Normal()
) {
    switch (dist.get_type()) {

        /** @brief Generate Normal distributed dummies. */
        case Distribution::Type::Normal: {
            std::normal_distribution<double> distribution(0.0, 1.0);
            #pragma omp parallel for schedule(static)
            for (std::size_t j = 0; j < p; ++j) {
                std::mt19937 gen(base_seed + static_cast<unsigned int>(j) + 5000);
                for (std::size_t i = 0; i < n; ++i) {
                    X(i, j) = distribution(gen);
                }
            }
            break;
        }

        /** @brief Generate Uniform distributed dummies. */
        case Distribution::Type::Uniform: {
            double a = dist.uniform_a;
            std::uniform_real_distribution<double> distribution(-a, a);
            #pragma omp parallel for schedule(static)
            for (std::size_t j = 0; j < p; ++j) {
                std::mt19937 gen(base_seed + static_cast<unsigned int>(j) + 6000);
                for (std::size_t i = 0; i < n; ++i) {
                    X(i, j) = distribution(gen);
                }
            }
            break;
        }

        /** @brief Generate Rademacher distributed dummies. */
        case Distribution::Type::Rademacher: {
            std::bernoulli_distribution distribution(0.5);
            #pragma omp parallel for schedule(static)
            for (std::size_t j = 0; j < p; ++j) {
                std::mt19937 gen(base_seed + static_cast<unsigned int>(j) + 7000);
                for (std::size_t i = 0; i < n; ++i) {
                    X(i, j) = distribution(gen) ? 1.0 : -1.0;
                }
            }
            break;
        }

        /** @brief Generate Student's t-distributed dummies. */
        case Distribution::Type::StudentT: {
            double df = dist.student_t_df;
            std::student_t_distribution<double> distribution(df);
            // Scale to unit variance if df > 2
            double scale = (df > 2.0) ? std::sqrt((df - 2.0) / df) :  1.0;
            #pragma omp parallel for schedule(static)
            for (std::size_t j = 0; j < p; ++j) {
                std::mt19937 gen(base_seed + static_cast<unsigned int>(j) + 8000);
                for (std::size_t i = 0; i < n; ++i) {
                    X(i, j) = distribution(gen) * scale;
                }
            }
            break;
        }

        /** @brief Generate Laplace distributed dummies using Boost.Math. */
        case Distribution::Type::Laplace: {
            double location = dist.laplace_location;
            double scale = dist.laplace_scale;
            boost::random::laplace_distribution<double> distribution(location, scale);
            #pragma omp parallel for schedule(static)
            for (std::size_t j = 0; j < p; ++j) {
                std::mt19937 gen(base_seed + static_cast<unsigned int>(j) + 9000);
                for (std::size_t i = 0; i < n; ++i) {
                    X(i, j) = distribution(gen);
                }
            }
            break;
        }

        /** @brief Generate Gumbel distributed dummies. */
        case Distribution::Type::Gumbel: {
            double location = dist.gumbel_location;
            double scale = dist.gumbel_scale;
            std::extreme_value_distribution<double> distribution(location, scale);
            #pragma omp parallel for schedule(static)
            for (std::size_t j = 0; j < p; ++j) {
                std::mt19937 gen(base_seed + static_cast<unsigned int>(j) + 10000);
                for (std::size_t i = 0; i < n; ++i) {
                    X(i, j) = distribution(gen);
                }
            }
            break;
        }

        /** @brief Generate Holtsmark distributed dummies. */
        case Distribution::Type::Holtsmark: {
            double location = dist.holtsmark_location;
            double scale = dist.holtsmark_scale;

            // Create Boost distribution for inverse transform
            boost::math::holtsmark_distribution<double> distribution(location, scale);
            std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

            #pragma omp parallel for schedule(static)
            for (std::size_t j = 0; j < p; ++j) {
                std::mt19937 gen(base_seed + static_cast<unsigned int>(j) + 11000);
                for (std::size_t i = 0; i < n; ++i) {
                    double u = uniform_dist(gen);
                    X(i, j) = boost::math::quantile(distribution, u);
                }
            }
            break;
        }

        default: {
            throw std::invalid_argument("Unsupported dummy distribution.");
        }
    }
}

// ===================================================================================

} /* End of namespace dummygen */
} /* End of namespace utils */
} /* End of namespace trex */

#endif /* End of TREX_UTILS_DATAGEN_DUMMYGEN_HPP */
