// ===================================================================================
// utils_dummygen.hpp
// ===================================================================================
#ifndef UTILS_DATAGENERATION_DUMMYGEN_HPP
#define UTILS_DATAGENERATION_DUMMYGEN_HPP
// ===================================================================================
/**
 * @file utils_dummygen.hpp
 *
 * @brief Dummy variable generation with multiple distributions.
 *
 * @details Provides distribution types and parallel generation functions for dummy
 * variable generation and augmentation of feature matrices.
 *
 * All distributions are centered (mean ≈ 0) and scaled appropriately
 * for use as null features in false discovery rate control methods.
 *
 * Distributions must satisfy:
 * - Exchangeability: No inherent ordering or correlation structure
 * - Centered: E[D] ≈ 0 to avoid spurious correlation with response
 * - Appropriate variance: Similar scale to real predictors
 *
 * Note: In T-Rex dummy variables are L2-normalized after generation.
 */
// ===================================================================================

// std includes
#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <stdexcept>

// Eigen includes
#include <Eigen/Dense>

// Boost includes
#include <boost/random/laplace_distribution.hpp>
#include <boost/math/distributions/holtsmark.hpp>
#include <boost/random/uniform_on_sphere.hpp>
#include <boost/random/generalized_inverse_gaussian_distribution.hpp>
#include <boost/random/triangle_distribution.hpp>

// utils includes
#include <utils/openmp/utils_openmp.hpp>

// ===================================================================================

// Embedded into namespace trex::utils::datageneration::dummygen
namespace trex::utils::datageneration::dummygen {

// ===================================================================================

// ============================================
// Seeding Helper
// ============================================

/**
 * @brief High-quality integer mixer (MurmurHash3 finalizer).
 *
 * @details Converts a (base, index) pair into a statistically independent random seed.
 * This is O(1) and extremely fast (bitwise ops only).
 * It ensures that sequential inputs (e.g., k=1, k=2) produce widely
 * separated, uncorrelated seeds, preventing overlap or thread collisions.
 *
 * @see https://en.wikipedia.org/wiki/MurmurHash
 *
 * @param base The base identifier (e.g., experiment seed).
 * @param index The second identifier (e.g., column index or loop index).
 *
 * @return std::uint32_t Unique mixed seed.
 */
inline std::uint32_t mix_seed(std::uint32_t base, std::size_t index) {
    // 1. Combine inputs non-linearly (Standard hash combine)
    //  0x9e3779b9 is the golden ratio constant (prevents zero-stuck states)
    std::uint32_t h = base ^ (static_cast<std::uint32_t>(index) + 0x9e3779b9 +
                      (base << 6) + (base >> 2));

    // 2. Apply MurmurHash3 finalizer
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}


// ============================================
// Unified Distribution Configuration
// ============================================

/**
 * @brief Configuration for dummy variable distribution.
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
        Holtsmark,

        /** @brief Triangular distribution with mean 0 and range [-b, b]. */
        Triangle,

        /** @brief Uniform distribution on the unit sphere. */
        UniformSphere,

        /** @brief Mammen distribution (two-point distribution). */
        Mammen,

        /** @brief Sparsity-constrained Rademacher distribution with sparsity density parameter s. */
        ConstrainedSparseRademacher,

        /** @brief Logistic distribution. */
        Logistic
    };


    // ============================================
    // Core member: distribution type
    // ============================================
    /** @brief Distribution type (Default: Normal). */
    Type type = Type::Normal;


    // ===================================================
    // Default members: Distribution-specific parameters
    // ===================================================

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

    /** @brief Triangle: lower bound (default: -sqrt(6) for unit variance) */
    double triangle_a = -std::sqrt(6.0);

    /** @brief Triangle: mode (default: 0.0 for symmetric) */
    double triangle_b = 0.0;

    /** @brief Triangle: upper bound (default: +sqrt(6) for unit variance) */
    double triangle_c = std::sqrt(6.0);

    /** @brief Uniform Sphere: dimension of the sphere (default: 3 for 3D sphere). */
    int sphere_dim = 3;

    /** @brief Mammen: p1 for value -(sqrt(5) - 1) / 2 */
    double mammen_param_p1 = (std::sqrt(5.0) + 1.0) / (2.0 * std::sqrt(5.0));

    /** @brief Mammen: p2 for value (sqrt(5) + 1) / 2 */
    double mammen_param_p2 = (std::sqrt(5.0) - 1.0) / (2.0 * std::sqrt(5.0));

    /** @brief Constrained Sparse Rademacher distribution: sparsity parameter (default: 0.1). */
    double constrained_sparse_rademacher_s = 0.1;

    /** @brief Logistic distribution: location parameter (default: 0.0). */
    double logistic_location = 0.0;

    /** @brief Logistic distribution: scale parameter (default: sqrt(3)/pi for unit variance). */
    double logistic_scale = std::sqrt(3.0) / std::numbers::pi;


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
     * @brief Create standard normal distribution N(0, 1).
     *
     * @details Properties: Mean = 0, Variance = 1.
     *
     * @see https://en.wikipedia.org/wiki/Normal_distribution
     *
     * @return Distribution instance for Normal distribution.
     */
    static Distribution Normal() {
        return Distribution(Type::Normal);
    }


    /**
     * @brief Create uniform U(-a, a) distribution.
     *
     * @details The uniform distribution has mean = 0 and variance = a²/3.
     *          For unit variance, use a = sqrt(3) (default).
     *
     * @see https://en.wikipedia.org/wiki/Continuous_uniform_distribution
     *
     * @param a Half-range parameter (default: sqrt(3) for unit variance).
     *
     * @return Distribution instance for Uniform distribution.
     */
    static Distribution Uniform(double a = std::sqrt(3.0)) {
        if (a <= 0.0) {
            throw std::invalid_argument("Uniform distribution parameter a must be > 0.");
        }
        Distribution dist(Type::Uniform);
        dist.uniform_a = a;
        return dist;
    }


    /**
     * @brief Create Rademacher distribution {-1, +1} with equal probability Pr = 0.5.
     *
     * @details The Rademacher distribution takes values {-1, +1} with P(X = -1) = P(X = +1) = 0.5.
     *          Properties: Mean = 0, Variance = 1.
     *          Named after Hans Rademacher.
     *
     * @see https://en.wikipedia.org/wiki/Rademacher_distribution
     *
     * @return Distribution instance for Rademacher distribution.
     */
    static Distribution Rademacher() {
        return Distribution(Type::Rademacher);
    }


    /**
     * @brief Create Student's t-distribution with df degrees of freedom.
     *
     * @details The Student's t-distribution with df degrees of freedom has mean = 0 for df > 1
     *          and variance = df / (df - 2) for df > 2.
     *          Implementation automatically scales to unit variance when df > 2.
     *
     * @see https://en.wikipedia.org/wiki/Student%27s_t-distribution
     *
     * @param df Degrees of freedom (default: 5.0).
     *
     * @return Distribution instance for Student's t-distribution.
     */
    static Distribution StudentT(double df = 5.0) {
        if (df <= 0.0) {
            throw std::invalid_argument("Student-t degrees of freedom must be > 0.");
        }
        Distribution dist(Type::StudentT);
        dist.student_t_df = df;
        return dist;
    }


    /**
     * @brief Create Laplace (double exponential) distribution.
     *
     * @details  The Laplace distribution has mean = location and variance = 2 * scale^2.
     *           Default scale = 1/sqrt(2) which provides unit variance.
     *
     * @see https://en.wikipedia.org/wiki/Laplace_distribution
     *
     * @param location Location parameter (default: 0.0).
     * @param scale Scale parameter (default: 1/sqrt(2)).
     *
     * @return Distribution instance for Laplace distribution.
     */
    static Distribution Laplace(double location = 0.0, double scale = 1.0 / std::sqrt(2.0)) {
        if (scale <= 0.0) {
            throw std::invalid_argument("Laplace distribution scale parameter must be > 0.");
        }
        Distribution dist(Type::Laplace);
        dist.laplace_location = location;
        dist.laplace_scale = scale;
        return dist;
    }


    /**
     * @brief Create Gumbel (Type-I extreme value) distribution.
     *
     * @details The Gumbel distribution has mean = location + scale * gamma and
     *          variance = (pi^2 / 6) * scale^2, where gamma is the Euler-Mascheroni constant.
     *          The implementation centers the distribution at location = 0.
     *
     * @see https://en.wikipedia.org/wiki/Gumbel_distribution
     *
     * @param location Location parameter (default: 0.0, automatically adjusted for centering).
     * @param scale Scale parameter (default: 1.0).
     *
     * @return Distribution instance for Gumbel distribution.
     */
    static Distribution Gumbel(double location = 0.0, double scale = 1.0) {
        if (scale <= 0.0) {
            throw std::invalid_argument("Gumbel distribution scale parameter must be > 0.");
        }
        Distribution dist(Type::Gumbel);
        dist.gumbel_location = location;
        dist.gumbel_scale = scale;
        return dist;
    }


    /**
     * @brief Create Holtsmark distribution.
     *
     * @details The Holtsmark distribution is a symmetric stable distribution with shape
     *          parameter alpha = 3/2 and skewness parameter beta = 0.
     *
     *          Properties:
     *          - Mean: location (well-defined, alpha = 3/2 > 1)
     *          - Variance: INFINITE (since alpha = 3/2 < 2)
     *          - All higher moments: infinite
     *          - Symmetric around location parameter
     *
     *          Applications: Plasma physics, astrophysics (Coulomb forces), testing
     *          robustness to extreme heavy-tailed noise.
     *
     * @see https://en.wikipedia.org/wiki/Holtsmark_distribution
     *
     * @param location Location parameter (mean, median, mode; default: 0.0)
     * @param scale Scale parameter (controls dispersion; default: 1.0)
     *
     * @return Distribution instance for Holtsmark distribution.
     */
    static Distribution Holtsmark(double location = 0.0, double scale = 1.0) {
        if (scale <= 0.0) {
            throw std::invalid_argument("Holtsmark distribution scale parameter must be > 0.");
        }
        Distribution dist(Type::Holtsmark);
        dist.holtsmark_location = location;
        dist.holtsmark_scale = scale;
        return dist;
    }


    /**
     * @brief Create Triangle distribution.
     *
     * @details The Triangle distribution with parameters (a, b, c) has:
     *          - Mean = (a + b + c) / 3
     *          - Variance = (a^2 + b^2 + c^2 - ab - ac - bc) / 18.
     *
     *          Default parameters (a = -sqrt(6), b = 0, c = +sqrt(6)) yield unit variance and
     *          mean = 0.
     *
     * @see https://en.wikipedia.org/wiki/Triangular_distribution
     *
     * @param a Lower bound (default: -sqrt(6) for unit variance)
     * @param b Mode (default: 0.0 for symmetric distribution)
     * @param c Upper bound (default: +sqrt(6) for unit variance)
     *
     * @return Distribution instance for Triangle distribution.
     */
    static Distribution Triangle(
        double a = -std::sqrt(6.0),
        double b = 0.0,
        double c = std::sqrt(6.0)
    ) {
        if (!(a < b && b < c)) {
            throw std::invalid_argument("Triangle distribution parameters must satisfy a < b < c.");
        }
        Distribution dist(Type::Triangle);
        dist.triangle_a = a;
        dist.triangle_b = b;
        dist.triangle_c = c;

        return dist;
    }


    /**
     * @brief Create Uniform distribution on the unit d-sphere.
     *
     * @details Generates points uniformly distributed on the surface of a unit sphere in
     *          d-dimensional space. Each coordinate has mean = 0 and variance = 1 / d.
     *
     *          Note: Requires that the number of dummy columns p is actually a multiple of dim.
     *
     * @see https://en.wikipedia.org/wiki/N-sphere#Uniformly_at_random_on_the_sphere
     *
     * @param dim Dimension of the sphere (default: 3 for 3D sphere, must be >= 2).
     *
     * @return Distribution instance for uniform distribution on the unit sphere.
     */
    static Distribution UniformSphere(int dim = 3) {
        if (dim < 2) {
            throw std::invalid_argument("Sphere dimension must be >= 2.");
        }
        Distribution dist(Type::UniformSphere);
        dist.sphere_dim = dim;
        return dist;
    }


    /**
     * @brief Create Mammen two-point distribution.
     *
     * @details The Mammen distribution is a two-point distribution proposed by Mammen (1993)
     *          for use in the wild bootstrap. It takes values:
     *
     *          - X = -(sqrt(5) - 1) / 2  with probability (sqrt(5) + 1) / (2 * sqrt(5))
     *          - X = +(sqrt(5) + 1) / 2  with probability (sqrt(5) - 1) / (2 * sqrt(5))
     *
     *          Properties: Mean = 0, Variance = 1.
     *          These values are based on the golden ratio for which the distribution is
     *          also called the "golden ratio distribution".
     *
     * @see: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
     *
     * @param p1 Probability for value -(sqrt(5) - 1) / 2
     * @param p2 Probability for value +(sqrt(5) + 1) / 2
     *
     * @return Distribution instance for Mammen distribution.
     */
    static Distribution Mammen(
        double p1 = (std::sqrt(5.0) + 1.0) / (2.0 * std::sqrt(5.0)),
        double p2 = (std::sqrt(5.0) - 1.0) / (2.0 * std::sqrt(5.0))
    ) {
        Distribution dist(Type::Mammen);
        dist.mammen_param_p1 = p1;
        dist.mammen_param_p2 = p2;
        return dist;
    }

    /**
     * @brief Create Constrained Sparse Rademacher distribution.
     *
     * @details Generates a sparse sign matrix with sparsity parameter s:
     *          each entry is +1 or -1 with probability s/2 each, and 0 with
     *          probability 1 - s.
     *
     * @param s Sparsity parameter in (0, 1]. s = 1 recovers the Rademacher distribution.
     *
     * @return Distribution instance for Constrained Sparse Rademacher distribution.
     *
     * @throws std::invalid_argument if s is not in (0, 1].
     */
    static Distribution ConstrainedSparseRademacher(const double s) {
        if (s <= 0.0 || s > 1.0) {
            throw std::invalid_argument("Sparsity parameter s must be in (0, 1].");
        }
        Distribution dist(Type::ConstrainedSparseRademacher);
        dist.constrained_sparse_rademacher_s = s;
        return dist;
    }


    /**
     * @brief Create Logistic distribution.
     *
     * @details The Logistic distribution has:
     *          - Mean = location
     *          - Variance = (pi^2 / 3) * scale^2.
     *          The default scale = sqrt(3) / pi provides unit variance.
     *
     *          The logistic distribution has heavier tails than the normal distribution but
     *          lighter than Cauchy and Laplace and can be understood as a middle ground.
     *
     * @see https://en.wikipedia.org/wiki/Logistic_distribution
     *
     * @param location Location parameter (default: 0.0).
     * @param scale Scale parameter (default: sqrt(3)/pi for unit variance).
     *
     * @return Distribution instance for Logistic distribution.
     */
    static Distribution Logistic(
        double location = 0.0,
        double scale = std::sqrt(3.0) / std::numbers::pi
    ) {
        if (scale <= 0.0) {
            throw std::invalid_argument("Logistic distribution scale parameter must be > 0.");
        }
        Distribution dist(Type::Logistic);
        dist.logistic_location = location;
        dist.logistic_scale = scale;
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
 * @brief Generate dummy variables using hashed seeding for uniqueness.
 *
 * @details Avoid the overflow/overlap risk of linear seeding (base + i).
 *
 * @tparam MatrixType Eigen-compatible matrix type (e.g. Eigen::MatrixXd or Eigen::Map).
 *
 * @param D Reference to the matrix to be filled with dummies.
 * @param n Number of rows.
 * @param p Number of columns (number of dummies).
 * @param base_seed Base seed for random number generation.
 * @param dist Distribution configuration.
 */
template<typename MatrixType>
inline void generate_dummies(
    MatrixType& D,
    std::size_t n,
    std::size_t p,
    unsigned int base_seed,
    const Distribution& dist = Distribution::Normal()
) {
    // OpenMP Safety:
    // Each column iteration uses a local PRNG.
    // The seed is mix_seed(base, j), which is deterministic and unique per column.
    // Threads cannot generate duplicates or race on the PRNG state.

    // Choose generation method based on distribution type
    switch (dist.get_type()) {

        /** @brief Generate Normal distributed dummies. */
        case Distribution::Type::Normal: {
            std::normal_distribution<double> distribution(0.0, 1.0);

            #pragma omp parallel for schedule(static)
            for (std::size_t j = 0; j < p; ++j) {
                std::mt19937 gen(mix_seed(base_seed, j));
                for (std::size_t i = 0; i < n; ++i) {
                    D(i, j) = distribution(gen);
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
                std::mt19937 gen(mix_seed(base_seed, j));
                for (std::size_t i = 0; i < n; ++i) {
                    D(i, j) = distribution(gen);
                }
            }
            break;
        }


        /** @brief Generate Rademacher distributed dummies. */
        case Distribution::Type::Rademacher: {
            std::bernoulli_distribution distribution(0.5);

            #pragma omp parallel for schedule(static)
            for (std::size_t j = 0; j < p; ++j) {
                std::mt19937 gen(mix_seed(base_seed, j));
                for (std::size_t i = 0; i < n; ++i) {
                    D(i, j) = distribution(gen) ? 1.0 : -1.0;
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
                std::mt19937 gen(mix_seed(base_seed, j));
                for (std::size_t i = 0; i < n; ++i) {
                    D(i, j) = distribution(gen) * scale;
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
                std::mt19937 gen(mix_seed(base_seed, j));
                for (std::size_t i = 0; i < n; ++i) {
                    D(i, j) = distribution(gen);
                }
            }
            break;
        }


        /** @brief Generate Gumbel distributed dummies. */
        case Distribution::Type::Gumbel: {
            double location = dist.gumbel_location;
            double scale = dist.gumbel_scale;

            // Gumbel mean: location + scale * gamma
            double adjusted_location = location + scale * std::numbers::egamma;
            std::extreme_value_distribution<double> distribution(adjusted_location, scale);

            #pragma omp parallel for schedule(static)
            for (std::size_t j = 0; j < p; ++j) {
                std::mt19937 gen(mix_seed(base_seed, j));
                for (std::size_t i = 0; i < n; ++i) {
                    D(i, j) = distribution(gen);
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
                std::mt19937 gen(mix_seed(base_seed, j));
                for (std::size_t i = 0; i < n; ++i) {
                    double u = uniform_dist(gen);
                    D(i, j) = boost::math::quantile(distribution, u);
                }
            }
            break;
        }


        /** @brief Generate Triangle distributed dummies using Boost.Random. */
        case Distribution::Type::Triangle: {
            double a = dist.triangle_a;
            double b = dist.triangle_b;
            double c = dist.triangle_c;
            boost::random::triangle_distribution<double> distribution(a, b, c);

            #pragma omp parallel for schedule(static)
            for (std::size_t j = 0; j < p; ++j) {
                std::mt19937 gen(mix_seed(base_seed, j));
                for (std::size_t i = 0; i < n; ++i) {
                    D(i, j) = distribution(gen);
                }
            }
            break;
        }


        /** @brief Generate Unit Sphere distributed dummies using Boost.Random. */
        case Distribution::Type::UniformSphere: {
            int dim = dist.sphere_dim;

            // Validate dimensions
            if (p % static_cast<std::size_t>(dim) != 0) {
                throw std::invalid_argument(
                    "Number of dummies p must be a multiple of sphere dimension."
                );
            }

            // Boost uniform_on_sphere distribution
            boost::random::uniform_on_sphere<double> distribution(dim);

            #pragma omp parallel for schedule(static)
            for (std::size_t j = 0; j < p; j += static_cast<std::size_t>(dim)) {
                std::mt19937 gen(mix_seed(base_seed, j));
                for (std::size_t i = 0; i < n; ++i) {
                    // Generate one point on sphere
                    std::vector<double> point = distribution(gen);

                    // Store across dim consecutive columns
                    for (int k = 0; k < dim; ++k) {
                        D(i, j + k) = point[k];
                    }
                }
            }
            break;
        }


        /** @brief Generate Mammen distributed dummies. */
        case Distribution::Type::Mammen: {
            double p1 = dist.mammen_param_p1; // Pr for value -(sqrt(5) - 1) / 2
            double p2 = dist.mammen_param_p2; // Pr for value +(sqrt(5) + 1) / 2
            std::discrete_distribution<int> distribution({p1, p2});

            #pragma omp parallel for schedule(static)
            for (std::size_t j = 0; j < p; ++j) {
                std::mt19937 gen(mix_seed(base_seed, j));
                for (std::size_t i = 0; i < n; ++i) {
                    int sample = distribution(gen);
                    D(i, j) = (sample == 0) ? (-(std::sqrt(5.0) - 1.0) / 2.0)
                                            : (+(std::sqrt(5.0) + 1.0) / 2.0);
                }
            }
            break;
        }

        /** @brief Generate Constrained Sparse Rademacher distributed dummies. */
        case Distribution::Type::ConstrainedSparseRademacher: {
            double s = dist.constrained_sparse_rademacher_s;

            #pragma omp parallel for schedule(static)
            for (std::size_t j = 0; j < p; ++j) {
                std::mt19937 gen(mix_seed(base_seed, j));

                // Calculate even number of non-zeros: k = floor(s * n / 2) (floor -> std::size_t)
                std::size_t k = 2 * static_cast<std::size_t>(static_cast<double>(n) * s / 2.0);

                // Handle ultra-sparse case (s < 2 / n)
                // Ensure a minimum of 2 non-zeros for balance
                if (k < 2) { k = 2; }
                std::size_t k_half = k / 2;

                // Initialize column to zeros
                D.col(j).setZero();

                // Create shuffled index vector for random placement
                std::vector<std::size_t> indices(n);
                std::iota(indices.begin(), indices.end(), 0);
                std::shuffle(indices.begin(), indices.end(), gen);

                // Assign +1 to first k/2 and -1 to next k/2 shuffled positions
                // splitting saves wasted 50% of lookups in rejection sampling
                for (std::size_t i = 0; i < k_half; ++i) {
                    D(indices[i], j) = 1.0;
                }
                for (std::size_t i = k_half; i < k; ++i) {
                    D(indices[i], j) = -1.0;
                }
            }
            break;
        }


        /** @brief Generate Logistic distributed dummies */
        case Distribution::Type::Logistic: {
            double location = dist.logistic_location;
            double scale = dist.logistic_scale;
            std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

            #pragma omp parallel for schedule (static)
            for (std::size_t j = 0; j < p; ++j) {
                std::mt19937 gen(mix_seed(base_seed, j));
                for (std::size_t i = 0; i < n; ++i) {
                    double u = uniform_dist(gen);
                    D(i, j) = location + scale * std::log(u / (1.0 - u));
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

} /* End of namespace trex::utils::datageneration::dummygen */

#endif /* UTILS_DATAGENERATION_DUMMYGEN_HPP */
