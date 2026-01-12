// ===================================================================================
// utils_datagen.hpp
// ===================================================================================
/**
 * @file utils_datagen.hpp
 *
 * @brief Utilities for synthetic data generation for regression problems with known
 *        support.
 *
 * @details Provides both in-memory and memory-mapped data generation functionalities.
 *
 */
// ===================================================================================

#ifndef TREX_UTILS_DATAGEN_HPP
#define TREX_UTILS_DATAGEN_HPP

// ===================================================================================

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>

#include <Eigen/Dense>

#include "utils/openMP/utils_openmp.hpp"
#include "utils/memmap/MemoryMappedMatrix.hpp"
#include "utils/datagen/utils_dummygen.hpp"

// ===================================================================================

namespace trex {
namespace utils {
namespace datagen {

// ==============================================================================
// Set aliases for namespaces
namespace memmap = trex::utils::memmap;
namespace dummygen = trex::utils::dummygen;

// =============================================================================
// Helper Functions (private)
// =============================================================================

namespace detail {

/**
 * @brief Validate support indices and coefficients.
 *
 * @param support Vector of support indices.
 * @param coefs Vector of coefficient values.
 * @param p Total number of predictors.
 */
inline void validate_support(
    const std::vector<std::size_t>& support,
    const std::vector<double>& coefs,
    std::size_t p
) {
    if (support.size() != coefs.size()) {
        throw std::invalid_argument(
            "Mismatch between support and coefficients size."
        );
    }

    for (auto idx : support) {
        if (idx >= p) {
            throw std::out_of_range(
                "Support index " + std::to_string(idx) +
                " out of range [0, " + std::to_string(p) + ")."
            );
        }
    }
}


/**
 * @brief Generate random seed from input or system entropy.
 *
 * @param seed Input seed (< 0 for random).
 */
inline unsigned int get_base_seed(int seed) {
    return (seed >= 0) ? static_cast<unsigned int>(seed) :
                        std::random_device{}();
}


/**
 * @brief Generate standard normal matrix.
 *
 * @tparam MatrixType Type of the matrix (e.g., Eigen::MatrixXd).
 *
 * @param X Reference to the matrix to be filled.
 * @param n Number of rows.
 * @param p Number of columns.
 * @param base_seed Base seed for random number generation.
 */
template<typename MatrixType>
inline void generate_standard_normal_matrix(
    MatrixType& X,
    std::size_t n,
    std::size_t p,
    unsigned int base_seed
) {
    // Use dummygen to generate standard normal dummies
    dummygen::generate_dummies(X, n, p, base_seed, dummygen::Distribution::Normal());
}


/**
 * @brief Generate Student's t-distribution matrix.
 *
 * @tparam MatrixType Type of the matrix (e.g., Eigen::MatrixXd).
 *
 * @param X Reference to the matrix to be filled.
 * @param n Number of rows.
 * @param p Number of columns.
 * @param df Degrees of freedom.
 * @param base_seed Base seed for random number generation.
 */
template<typename MatrixType>
inline void generate_student_t_matrix(
    MatrixType& X,
    std::size_t n,
    std::size_t p,
    std::size_t df,
    unsigned int base_seed
) {
    // Use dummygen to generate Student's t-distribution dummies
    dummygen::generate_dummies(X, n, p, base_seed, dummygen::Distribution::StudentT(df));
}


/**
 * @brief Generate Gumbel distribution matrix.
 *
 * @tparam MatrixType Type of the matrix (e.g., Eigen::MatrixXd).
 *
 * @param X Reference to the matrix to be filled.
 * @param n Number of rows.
 * @param p Number of columns.
 * @param location Location parameter.
 * @param scale Scale parameter.
 * @param base_seed Base seed for random number generation.
 */
template<typename MatrixType>
inline void generate_gumbel_matrix(
    MatrixType& X,
    std::size_t n,
    std::size_t p,
    unsigned int base_seed,
    double location = 0.0,
    double scale = 1.0
) {
    // Use dummygen to generate Gumbel distribution dummies
    dummygen::generate_dummies(X, n, p, base_seed, dummygen::Distribution::Gumbel(location, scale));
}


/**
 * @brief Generate AR(1) row-wise correlated matrix.
 *
 * @tparam MatrixType Type of the matrix (e.g., Eigen::MatrixXd).
 *
 * @param X Reference to the matrix to be filled.
 * @param n Number of rows.
 * @param p Number of columns.
 * @param rho Autoregressive parameter.
 * @param base_seed Base seed for random number generation.
 */
template<typename MatrixType>
inline void generate_ar1_rowwise_matrix(
    MatrixType& X,
    std::size_t n,
    std::size_t p,
    double rho,
    unsigned int base_seed
) {
    // double variance for unit marginal variance: sigma^2 = 1 - rho^2
    double innovation_std = std::sqrt(1.0 - rho * rho);

    // Data generation with row-wise AR(1) process
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        // Thread-safe seeding using mix_seed
        std::mt19937 row_gen(dummygen::mix_seed(base_seed, i));
        std::normal_distribution<double> dist(0.0, 1.0);

        // Initialize first column elements with stationary N(0, 1)
        X(i, 0) = dist(row_gen);

        // Generate AR(1) process across columns
        for (std::size_t j = 1; j < p; ++j) {
            double innovation = innovation_std * dist(row_gen);
            X(i, j) = rho * X(i, j - 1) + innovation;
        }
    }
}


/**
 * @brief Generate response vector y in a linear model given X, support, and coefficients.
 *
 * @tparam MatrixType Type of the matrix (e.g., Eigen::MatrixXd).
 * @tparam VectorType Type of the vector (e.g., Eigen::VectorXd).
 *
 * @param y Reference to the response vector to be filled.
 * @param X Predictor matrix.
 * @param support Indices of the true support.
 * @param coefs Coefficient values corresponding to the support.
 */
template<typename MatrixType, typename VectorType>
inline void generate_signal(
    VectorType& y,
    const MatrixType& X,
    const std::vector<std::size_t>& support,
    const std::vector<double>& coefs
) {
    y.setZero();
    for (std::size_t i = 0; i < support.size(); ++i) {
        y += coefs[i] * X.col(support[i]);
    }
}


/**
 * @brief Calculate noise parameters (signal power and noise standard deviation) based on SNR.
 *
 * @tparam VectorType
 *
 * @param y Response vector.
 * @param n Number of observations.
 * @param snr Signal-to-noise ratio (linear scale).
 *
 * @return Pair of signal power and noise standard deviation.
 */
template<typename VectorType>
inline std::pair<double, double> calculate_noise_params(
    const VectorType& y,
    std::size_t n,
    double snr
) {
    // Calculate signal power
    double mean_y = y.mean();
    double var_sum = (y.array() - mean_y).square().sum();
    double signal_power = (n <= 100) ? var_sum / (n - 1) : var_sum / n;

    // Calculate noise standard deviation
    double noise_std = 0.0;
    if (snr > 0 && signal_power > 0) {
        noise_std = std::sqrt(signal_power / snr);
    }

    return {signal_power, noise_std};
}


/**
 * @brief Add Gaussian noise to response vector y.
 *
 * @tparam VectorType Type of the vector (e.g., Eigen::VectorXd).
 *
 * @param y Reference to the response vector to which noise will be added.
 * @param n Number of observations.
 * @param noise_std Target standard deviation of the noise.
 * @param noise_seed Seed for random number generation.
 */
template<typename VectorType>
inline void add_gaussian_noise(
    VectorType& y,
    std::size_t n,
    double noise_std,
    unsigned int noise_seed
) {
    if (noise_std <= 0) {
        throw std::invalid_argument("Noise standard deviation must be > 0.");
    }

    std::mt19937 noise_gen(noise_seed + 999999);
    std::normal_distribution<double> noise_dist(0.0, noise_std);

    for (std::size_t i = 0; i < n; ++i) {
        y(i) += noise_dist(noise_gen);
    }
}


/**
 * @brief Add Student's t-distributed noise to response vector y.
 *
 * @details Generates i.i.d. Student-t noise with heavy tails.
 *          Automatically scales to achieve target noise_std when df > 2.
 *
 * @tparam VectorType Type of the vector (e.g., Eigen::VectorXd).
 *
 * @param y Reference to the response vector to which noise will be added.
 * @param n Number of observations.
 * @param noise_std Target standard deviation of the noise.
 * @param df Degrees of freedom for Student's t-distribution (default: 5.0).
 * @param noise_seed Seed for random number generation.
 */
template <typename VectorType>
inline void add_student_t_noise(
    VectorType& y,
    std::size_t n,
    double noise_std,
    double df,
    unsigned int noise_seed
) {
    if (df <= 0.0) {
        throw std::invalid_argument("Student-t degrees of freedom must be > 0.");
    }
    if (noise_std <= 0) return;

    std::mt19937 noise_gen(noise_seed + 999999);
    std::student_t_distribution<double> noise_dist(df);

    // variance of Student's t: df / (df - 2) for df > 2
    double base_scale = (df > 2.0) ? std::sqrt((df - 2.0) / df) : 1.0;
    double scale = (df > 2.0) ? noise_std * base_scale : noise_std;

    for (std::size_t i = 0; i < n; ++i) {
        y(i) += scale * noise_dist(noise_gen);
    }
}


/**
 * @brief Generate AR(1) noise and add it to the response vector y.
 *
 * @details Generates an AR(1) process:
 *          epsilon[i] = rho * epsilon[i-1] + eta[i]
 *          where eta[i] ~ N(0, sigma_eta^2) with sigma_eta^2 = (1 - rho^2) * noise_std^2.
 *
 *          Properties (|rho| < 1, stationary):
 *          - Mean = 0
 *          - Marginal variance: noise_std^2
 *          - Autocorrelation: Corr(epsilon[i], epsilon[j]) = rho^|i-j|
 *
 *          Unit root / Explosive process (|rho| >= 1):
 *          - Variance grows over time (non-stationary)
 *          - SNR control is NOT guaranteed
 *          - innovation_std = noise_std (no scaling adjustment)
 *
 *          Note: AR(1) noise violates i.i.d. assumption.
 *
 * @tparam VectorType Type of the vector (e.g., Eigen::VectorXd).
 *
 * @param y Reference to the response vector to which noise will be added.
 * @param n Number of observations.
 * @param noise_std Target standard deviation of the noise (only for |rho| < 1).
 * @param rho Autoregressive parameter. |rho| < 1 for stationary process, |rho| = 1 for unit root,
 *            |rho| > 1 for explosive process.
 * @param noise_seed Seed for random number generation.
 */
template<typename VectorType>
inline void add_ar1_noise(
    VectorType& y,
    std::size_t n,
    double noise_std,
    double rho,
    unsigned int noise_seed
) {
    if (noise_std <= 0) {
        throw std::invalid_argument("Noise standard deviation must be > 0.");
    }

    // Innovation variance: sigma_eta^2 = (1 - rho^2) * noise_std^2
    double innovation_std;
    if (std::abs(rho) < 1.0) {
        // Stationary case: sigma_eta^2 = (1 - rho^2) * noise_std^2
        innovation_std = noise_std * std::sqrt(1.0 - rho * rho);
    } else {
        // Unit root or explosive case: no scaling possible
        innovation_std = noise_std;
    }

    std::mt19937 noise_gen(noise_seed + 999999);
    std::normal_distribution<double> innovation_dist(0.0, innovation_std);

    // Generate AR(1) noise
    VectorType ar1_noise(n);

    // Initialize first value
    if (std::abs(rho) < 1.0) {
        // Stationary case: draw from N(0, noise_std^2)
        std::normal_distribution<double> stationary_dist(0.0, noise_std);
        ar1_noise(0) = stationary_dist(noise_gen);
    } else {
        // Unit root or explosive case: start from innovation
        ar1_noise(0) = innovation_dist(noise_gen);
    }
    // Remainder recursion
    for (std::size_t i = 1; i < n; ++i) {
        double eta = innovation_dist(noise_gen);
        ar1_noise(i) = rho * ar1_noise(i - 1) + eta;
    }

    // Add AR(1) noise to response
    y += ar1_noise;
}

} /* End of namespace detail */


// ==============================================================================
// Noise Type Enumeration
// ==============================================================================

/**
 * @brief Enumeration for noise types.
 *
 * @details Specifies the distribution of additive noise in synthetic data generation.
 *          The noise is additive and designed for a linear model: y = X * beta + noise.
 */
enum class NoiseType {
    /** @brief Gaussian (Normal) noise: epsilon ~N(0, sigma^2). */
    Gaussian,

    /**
     * @brief Student's t-distributed noise: epsilon ~ t(df) scaled to sigma^2.
     *        Heavy-tailed noise for robustness testing.
     */
    StudentT,

    /**
     * @brief AR(1) autoregressive noise: epsilon[i] = rho * epsilon[i-1] + eta[i].
     *        Temporally correlated. Violates i.i.d. assumption.
     *        Properties:
     *        - |rho| < 1: stationary with controlled variance.
     *        - |rho| >= 1: unit root/explosive, SNR control lost.
     */
    AR1
};


/**
 * @brief Configuration for noise generation.
 */
struct NoiseConfig {
    // =======================================
    // Data members
    // =======================================

    /** @brief Type of noise distribution (default: Gaussian). */
    NoiseType type = NoiseType::Gaussian;

    /** @brief Degrees of freedom for Student's t-distribution (default: 5.0) */
    double student_t_df = 5.0;

    /**
     * @brief Autoregressive parameter for AR(1) noise (default: 0.5).
     *        - |rho| < 1: stationary
     *        - |rho| = 1: unit root
     *        - |rho| > 1: explosive
     */
    double ar1_rho = 0.5;


    // =======================================
    // Constructors
    // =======================================

    /** @brief Default constructor (Gaussian noise). */
    NoiseConfig() : type(NoiseType::Gaussian) {}

    /** @brief Construct with specific noise type. */
    explicit NoiseConfig(NoiseType t) : type(t) {}

    // =======================================
    // Named constructors (factory pattern)
    // =======================================

    /** @brief Create Gaussian noise configuration. */
    static NoiseConfig Gaussian() {
        return NoiseConfig(NoiseType::Gaussian);
    }

    /**
     * @brief Create Student's t-distributed noise configuration.
     *
     * @param df Degrees of freedom (default: 5.0).
     */
    static NoiseConfig StudentT(double df = 5.0) {
        NoiseConfig config(NoiseType::StudentT);
        config.student_t_df = df;
        return config;
    }

    /**
     * @brief Create AR(1) noise configuration.
     *
     * @param rho Autoregressive parameter.
    */
    static NoiseConfig AR1(double rho = 0.5) {
        NoiseConfig config(NoiseType::AR1);
        config.ar1_rho = rho;
        return config;
    }
};


/**
 * @brief Add noise to response vector y based on noise configuration.
 *
 * @tparam VectorType Type of the vector (e.g., Eigen::VectorXd).
 *
 * @param y Reference to the response vector to which noise will be added.
 * @param n Number of observations.
 * @param noise_std Target standard deviation of the noise (only for |rho| < 1).
 * @param config Noise configuration specifying the distribution and parameters of the noise.
 * @param noise_seed Seed for noise random number generation.
 */
template <typename VectorType>
inline void add_noise(
    VectorType& y,
    std::size_t n,
    double noise_std,
    const NoiseConfig& config,
    unsigned int noise_seed
) {
    switch (config.type) {
        case NoiseType::Gaussian:
            detail::add_gaussian_noise(y, n, noise_std, noise_seed);
            break;
        case NoiseType::StudentT:
            detail::add_student_t_noise(y, n, noise_std, config.student_t_df, noise_seed);
            break;
        case NoiseType::AR1:
            detail::add_ar1_noise(y, n, noise_std, config.ar1_rho, noise_seed);
            break;
        default:
            throw std::invalid_argument("Unsupported noise type.");
    }
}


// ==============================================================================
// Public Classes for Data Generation
// ==============================================================================

/**
 * @brief Generate synthetic regression data with known support and configurable noise.
 *
 * @details Generates in-memory data according to the linear model.
 *          y = X[, support] * coefs + noise.
 *          Uses openMP for parallel random number generation.
 */
class SyntheticData {
private:
    // =======================================
    // Data members
    // =======================================

    /** @brief Synthetic predictor matrix */
    Eigen::MatrixXd X_;

    /** @brief Synthetic response vector */
    Eigen::VectorXd y_;

    /** @brief Signal power */
    double signal_power_;

    /** @brief Noise standard deviation */
    double noise_std_;

public:
    // =======================================
    // Constructor
    // =======================================

    /**
     * @brief Generate synthetic regression data with known support
     *
     * @param n Number of observations
     * @param p Number of predictors
     * @param support Indices of true support (0-indexed)
     * @param coefs Values of true coefficients
     * @param snr Signal-to-noise ratio (linear scale)
     * @param seed Seed for random number generation (Default: -1 for random seed, >= for
     *             reproducible data).
     * @param noise_config Noise configuration (default: Gaussian).
     */
    SyntheticData(
        std::size_t n,
        std::size_t p,
        const std::vector<std::size_t>& support,
        const std::vector<double>& coefs,
        double snr = 1.0,
        int seed = -1,
        const NoiseConfig& noise_config = NoiseConfig::Gaussian()
    ) {
        // Validate inputs
        detail::validate_support(support, coefs, p);

        // Get reproducible seed
        unsigned int base_seed = detail::get_base_seed(seed);

        // Generate X ~ N(0, 1)
        X_.resize(n, p);
        detail::generate_standard_normal_matrix(X_, n, p, base_seed);

        // Generate signal: y = X[support] * coefs
        y_.resize(n);
        detail::generate_signal(y_, X_, support, coefs);

        // Calculate noise level and add noise
        std::tie(signal_power_, noise_std_) = detail::calculate_noise_params(y_, n, snr);
        add_noise(y_, n, noise_std_, noise_config, base_seed);
    }

    // =======================================
    // Accessors
    // =======================================

    /** @brief Get the predictor matrix */
    const Eigen::MatrixXd& getX() const { return X_; }

    /** @brief Get the response vector */
    const Eigen::VectorXd& getY() const { return y_; }

    /** @brief Get the predictor matrix (modifiable) */
    Eigen::MatrixXd& getX() { return X_; }

    /** @brief Get the response vector (modifiable) */
    Eigen::VectorXd& getY() { return y_; }

    /** @brief Get the signal power */
    double getSignalPower() const { return signal_power_; }

    /** @brief Get the noise standard deviation */
    double getNoiseStd() const { return noise_std_; }

    /** @brief Get the signal-to-noise ratio (linear) */
    double snr() const {
        return (noise_std_ > 0) ? signal_power_ / (noise_std_ * noise_std_) : 0.0;
    }

    /** @brief Get the number of rows (observations) */
    std::size_t rows() const { return X_.rows(); }

    /** @brief Get the number of columns (predictors) */
    std::size_t cols() const { return X_.cols(); }
};

// ==============================================================================


/**
 * @brief Memory-mapped synthetic data for large-scale datasets.
 *
 * @details Uses disk-backed storage via Boost.Iostreams or similar for for
 *          memory mapping. Suitable for datasets that do not fit into RAM,
 *          such as biobank-scale data (n approx 1e5, p approx 1e6).
 */
class SyntheticDataMapped {
private:
    // =======================================
    // Data members
    // =======================================

    /** @brief Memory-mapped predictor matrix */
    std::unique_ptr<memmap::MemoryMappedMatrix<double>> X_mmap_;

    /** @brief Memory-mapped response vector */
    std::unique_ptr<memmap::MemoryMappedMatrix<double>> y_mmap_;

    /** @brief File path for memory-mapped predictor matrix */
    std::string X_filepath_;

    /** @brief File path for memory-mapped response vector */
    std::string y_filepath_;

    /** @brief Number of observations */
    std::size_t n_;

    /** @brief Number of predictors */
    std::size_t p_;

    /** @brief Signal power */
    double signal_power_;

    /** @brief Noise standard deviation */
    double noise_std_;


public:
    // =======================================
    // Constructor
    // =======================================

    /**
     * @brief Generate memory-mapped synthetic data.
     *
     * @param X_filepath File path for memory-mapped predictor matrix.
     * @param y_filepath File path for memory-mapped response vector.
     * @param n Number of observations.
     * @param p Number of predictors.
     * @param support Indices of true support (0-indexed).
     * @param coefs Values of true coefficients.
     * @param snr Signal-to-noise ratio (linear scale).
     * @param seed Seed for random number generation (Default: -1 for random seed, >= 0
     *             for reproducible).
     * @param noise_config Noise configuration (default: Gaussian).
     */
    SyntheticDataMapped(
        const std::string& X_filepath,
        const std::string& y_filepath,
        std::size_t n,
        std::size_t p,
        const std::vector<std::size_t>& support,
        const std::vector<double>& coefs,
        double snr = 1.0,
        int seed = -1,
        const NoiseConfig& noise_config = NoiseConfig::Gaussian()
    ) : X_filepath_(X_filepath),
        y_filepath_(y_filepath),
        n_(n),
        p_(p)
    {
        // Validate inputs
        detail::validate_support(support, coefs, p);

        // Get reproducible seed
        unsigned int base_seed = detail::get_base_seed(seed);

        // Create memory-mapped matrices
        X_mmap_ = std::make_unique<memmap::MemoryMappedMatrix<double>>(
            X_filepath_, n, p, memmap::AccessMode::ReadWrite
        );
        y_mmap_ = std::make_unique<memmap::MemoryMappedMatrix<double>>(
            y_filepath_, n, 1, memmap::AccessMode::ReadWrite
        );

        // Get Eigen::Map views
        auto X_map = X_mmap_->getMap();
        auto y_map = y_mmap_->getMap();

        // Generate X ~ N(0, 1)
        detail::generate_standard_normal_matrix(X_map, n, p, base_seed);

        // Generate signal
        Eigen::VectorXd y_vec = y_map.col(0);
        detail::generate_signal(y_vec, X_map, support, coefs);
        y_map.col(0) = y_vec;

        // Calculate noise level and add noise
        std::tie(signal_power_, noise_std_) = detail::calculate_noise_params(y_vec, n, snr);
        add_noise(y_vec, n, noise_std_, noise_config, base_seed);

        // Update response in memory-mapped file
        y_map.col(0) = y_vec;
    }

    // =======================================
    // Accessors
    // =======================================

    /** @brief Get the memory-mapped predictor matrix */
    auto getX() { return X_mmap_->getMap(); }

    /** @brief Get the memory-mapped predictor matrix (const) */
    auto getX() const { return X_mmap_->getMap(); }

    /** @brief Get the memory-mapped response vector */
    auto getY() { return y_mmap_->getMap(); }

    /** @brief Get the memory-mapped response vector (const) */
    auto getY() const { return y_mmap_->getMap(); }

    /** @brief Get the signal power */
    double getSignalPower() const { return signal_power_; }

    /** @brief Get the noise standard deviation */
    double getNoiseStd() const { return noise_std_; }

    /** @brief Get the signal-to-noise ratio (linear scale) */
    double snr() const {
        return (noise_std_ > 0) ? signal_power_ / (noise_std_ * noise_std_) : 0.0;
    }

    /** @brief Get the number of rows (observations) */
    std::size_t rows() const { return n_; }

    /** @brief Get the number of columns (predictors) */
    std::size_t cols() const { return p_; }

    /** @brief Get the file path for the memory-mapped predictor matrix */
    const std::string& X_filepath() const { return X_filepath_; }

    /** @brief Get the file path for the memory-mapped response vector */
    const std::string& y_filepath() const { return y_filepath_; }
};


// ==============================================================================

/**
 * @brief Memory-mapped synthetic data matrix with dummy augmentation.
 *
 * @details Creates augmented predictor matrix X_aug = [X | D] where D contains
 *          dummy variables, directly on disk. Uses sparse file allocation.
 *
 */
class SyntheticDataMappedWithDummies {
private:
    // =======================================
    // Data members
    // =======================================

    /** @brief Memory-mapped augmented predictor matrix */
    std::unique_ptr<memmap::MemoryMappedMatrix<double>> X_aug_mmap_;

    /** @brief Memory-mapped response vector */
    std::unique_ptr<memmap::MemoryMappedMatrix<double>> y_mmap_;

    /** @brief File path for memory-mapped augmented predictor matrix */
    std::string X_aug_filepath_;

    /** @brief File path for memory-mapped response vector */
    std::string y_filepath_;

    /** @brief Number of observations */
    std::size_t n_;

    /** @brief Number of original predictors */
    std::size_t p_;

    /** @brief Number of dummy variables */
    std::size_t num_dummies_;

    /** @brief Signal power */
    double signal_power_;

    /** @brief Noise standard deviation */
    double noise_std_;


public:
    // =======================================
    // Constructor
    // =======================================

    /** @brief Constructor for SyntheticDataMappedWithDummies with configurable noise.
     *
     * @param X_aug_filepath File path for memory-mapped augmented predictor matrix.
     * @param y_filepath File path for memory-mapped response vector.
     * @param n Number of observations.
     * @param p Number of original predictors.
     * @param num_dummies Number of dummy variables to append.
     * @param support Indices of true support (0-indexed).
     * @param coefs Values of true coefficients.
     * @param snr Signal-to-noise ratio (linear scale).
     * @param seed Seed for random number generation (Default: -1 for random seed, >= 0
     *             for reproducible).
     * @param X_seed Seed for generating original predictors (Default: -1 for random seed,
     *               >= 0 for reproducible).
     * @param dummy_seed Seed for generating dummy variables (Default: -1 for random seed,
     *                   >= 0 for reproducible).
     * @param dummy_dist Distribution for dummy variable generation (Default: Normal).
     * @param noise_config Noise configuration (default: Gaussian).
     */
    SyntheticDataMappedWithDummies(
        const std::string& X_aug_filepath,
        const std::string& y_filepath,
        std::size_t n,
        std::size_t p,
        std::size_t num_dummies,
        const std::vector<std::size_t>& support,
        const std::vector<double>& coefs,
        double snr = 1.0,
        int seed = -1,
        int X_seed = -1,
        int dummy_seed = -1,
        const dummygen::Distribution& dummy_dist = dummygen::Distribution::Normal(),
        const NoiseConfig& noise_config = NoiseConfig::Gaussian()
    ) : X_aug_filepath_(X_aug_filepath),
        y_filepath_(y_filepath),
        n_(n),
        p_(p),
        num_dummies_(num_dummies)
    {
        // Validate inputs
        detail::validate_support(support, coefs, p);

        // Get seeds
        unsigned int base_seed = detail::get_base_seed(seed);
        unsigned int x_gen_seed = (X_seed >= 0) ? static_cast<unsigned int>(X_seed) :
                                              base_seed + 51242;
        unsigned int dummy_gen_seed = (dummy_seed >= 0) ? static_cast<unsigned int>(dummy_seed) :
                                                  base_seed + 14273785;

        // Create memory-mapped augmented matrix (sparse file)
        X_aug_mmap_ = std::make_unique<memmap::MemoryMappedMatrix<double>>(
            X_aug_filepath_, n, p + num_dummies, memmap::AccessMode::ReadWrite
        );

        y_mmap_ = std::make_unique<memmap::MemoryMappedMatrix<double>>(
            y_filepath_, n, 1, memmap::AccessMode::ReadWrite
        );

        // Get map view
        auto X_aug_map = X_aug_mmap_->getMap();
        auto y_map = y_mmap_->getMap();

        // Generate X in first p columns
        auto X_view = X_aug_map.leftCols(p);
        detail::generate_standard_normal_matrix(X_view, n, p, x_gen_seed);

        // Generate dummies in last num_dummies columns
        auto dummies_view = X_aug_map.rightCols(num_dummies);
        dummygen::generate_dummies(dummies_view, n, num_dummies, dummy_gen_seed, dummy_dist);

        // Generate signal y
        Eigen::VectorXd y_vec = y_map.col(0);
        detail::generate_signal(y_vec, X_view, support, coefs);
        y_map.col(0) = y_vec;

        // Add noise
        std::tie(signal_power_, noise_std_) = detail::calculate_noise_params(y_vec, n, snr);
        add_noise(y_vec, n, noise_std_, noise_config, base_seed);

        // Update response in memory-mapped file
        y_map.col(0) = y_vec;
    }

    // =======================================
    // Accessors
    // =======================================

    /** @brief Get augmented X view with dummies */
    auto getXAug() { return X_aug_mmap_->getMap(); }

    /** @brief Get augmented X view with dummies (const) */
    auto getXAug() const { return X_aug_mmap_->getMap(); }

    /** @brief Get response vector */
    auto getY() { return y_mmap_->getMap(); }

    /** @brief Get response vector (const) */
    auto getY() const { return y_mmap_->getMap(); }

    // Original X view (without dummies)
    /** @brief Get original X view (without dummies) */
    auto getX() {
        return getXAug().leftCols(p_);
    }

    /** @brief Get original X view (without dummies) (const) */
    auto getX() const {
        return getXAug().leftCols(p_);
    }

    /** @brief Get signal power */
    double getSignalPower() const { return signal_power_; }

    /** @brief Get noise standard deviation */
    double getNoiseStd() const { return noise_std_; }

    /** @brief Get signal-to-noise ratio (linear scale) */
    double snr() const {
        return (noise_std_ > 0) ? signal_power_ / (noise_std_ * noise_std_) : 0.0;
    }

    /** @brief Get number of rows */
    std::size_t rows() const { return n_; }

    /** @brief Get number of original columns (without dummies) */
    std::size_t cols() const { return p_; }

    /** @brief Get number of dummy variables */
    std::size_t num_dummies() const { return num_dummies_; }

    /** @brief Get total number of columns (with dummies) */
    std::size_t total_cols() const { return p_ + num_dummies_; }

    /** @brief Get file path for augmented X matrix */
    const std::string& X_aug_filepath() const { return X_aug_filepath_; }

    /** @brief Get file path for response vector */
    const std::string& y_filepath() const { return y_filepath_; }

};


// ==============================================================================
// Public Utility Functions
// ==============================================================================

/**
 * @brief Appends dummy variables to the augmented feature matrix.
 *
 * @param X Original feature matrix.
 * @param num_dummies Number of dummy variables to append.
 * @param seed Random seed for dummy generation (Default: -1 for random seed, >= 0 for
 *             reproducible).
 * @param dummy_dist Distribution for dummy variable generation (Default: Normal).
 */
inline Eigen::MatrixXd append_dummies_to_matrix(
    const Eigen::MatrixXd& X,
    std::size_t num_dummies,
    int seed = -1,
    const dummygen::Distribution& dummy_dist = dummygen::Distribution::Normal()
) {
    // Get dimensions
    const std::size_t n = X.rows();
    const std::size_t p = X.cols();

    // Setup augmented matrix
    Eigen::MatrixXd X_aug(n, p + num_dummies);
    X_aug.leftCols(p) = X;

    // Generate dummies using existing detail helper
    auto dummies_view = X_aug.rightCols(num_dummies);
    unsigned int base_seed = detail::get_base_seed(seed);
    dummygen::generate_dummies(dummies_view, n, num_dummies, base_seed, dummy_dist);

    return X_aug;
}


/**
 * @brief Create augmented memory-mapped matrix from existing data file.
 *
 * @details Reads existing predictor matrix, creates augmented memory-mapped matrix
 *          [X | D] with dummy variables. Uses openMP for parallel column-wise
 *          copying operations and dummy generation.
 *
 * @param X_filepath Path to existing predictor matrix (raw binary, column-major).
 * @param X_aug_filepath Output path for augmented matrix.
 * @param n Number of observations.
 * @param p Number of original predictors.
 * @param num_dummies Number of dummy variables to append.
 * @param seed Random seed for dummy generation (Default: -1 for random seed, >= 0 for
 *             reproducible).
 * @param dummy_dist Distribution for dummy variable generation (Default: Normal).
 */
inline void augment_existing_data_with_dummies(
    const std::string& X_filepath,
    const std::string& X_aug_filepath,
    std::size_t n,
    std::size_t p,
    std::size_t num_dummies,
    int seed = -1,
    const dummygen::Distribution& dummy_dist = dummygen::Distribution::Normal()
) {
    namespace fs = std::filesystem;

    // Validate input
    if (!fs::exists(X_filepath)) {
        throw std::runtime_error("Input file not found: " + X_filepath);
    }

    // Expected size
    std::size_t expected_bytes = n * p * sizeof(double);
    if (fs::file_size(X_filepath) != expected_bytes) {
        throw std::runtime_error(
            "File size mismatch. Expected " + std::to_string(expected_bytes) +
            " bytes for (" + std::to_string(n) + " x " + std::to_string(p) + ") matrix."
        );
    }

    // Create input memory map
    memmap::MemoryMappedMatrix<double> X_mmap(
        X_filepath, n, p, memmap::AccessMode::ReadOnly
    );

    // Create output memory map for augmented matrix
    memmap::MemoryMappedMatrix<double> X_aug_mmap(
        X_aug_filepath, n, p + num_dummies, memmap::AccessMode::ReadWrite
    );

    // Get maps
    auto X_map = X_mmap.getMap();
    auto X_aug_map = X_aug_mmap.getMap();

    // Copy original data
    #pragma omp parallel for schedule(static)
    for (std::size_t j = 0; j < p; ++j) {
        X_aug_map.col(j) = X_map.col(j);
    }

    // Generate dummies
    auto dummies_view = X_aug_map.rightCols(num_dummies);
    unsigned int base_seed = detail::get_base_seed(seed);
    dummygen::generate_dummies(dummies_view, n, num_dummies, base_seed, dummy_dist);
}



// ==============================================================================

} /* End of namespace datagen */
} /* End of namespace utils */
} /* End of namespace trex */

#endif /* End of TREX_UTILS_DATAGEN_HPP */
