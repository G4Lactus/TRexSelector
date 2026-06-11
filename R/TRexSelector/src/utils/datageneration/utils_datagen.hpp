// ===================================================================================
// utils_datagen.hpp
// ===================================================================================
#ifndef UTILS_DATAGENERATION_DATAGEN_HPP
#define UTILS_DATAGENERATION_DATAGEN_HPP
// ===================================================================================
/**
 * @file utils_datagen.hpp
 *
 * @brief Utility functions for data generation.
 *
 * @details Description for various predictor matrix distributions
 * (normal, Student's t, Gumbel, AR(1)) and noise distributions, according to
 * utils_noisegen.hpp.
 * Provides both in-memory and memory-mapped data generation classes
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>

// Eigen includes
#include <Eigen/Dense>

// utils includes
#include <utils/openmp/utils_openmp.hpp>
#include <utils/memmap/memory_mapped_matrix.hpp>
#include <utils/datageneration/utils_dummygen.hpp>
#include <utils/datageneration/utils_noisegen.hpp>

// ===================================================================================

namespace trex::utils::datageneration::datagen {

namespace memmap = trex::utils::memmap;
namespace dummygen = trex::utils::datageneration::dummygen;
namespace noisegen = trex::utils::datageneration::noisegen;

// ==============================================================================
namespace predictor_policy {
    /** @brief Policy struct for standard normal distribution (mean=0, std=1). */
    struct Normal {};

    /** @brief Policy struct for Student's t-distribution (default df=5). */
    struct StudentT {
        double df;
        explicit StudentT(double df = 5.0) : df(df) {
            if (df <= 0.0) { throw std::invalid_argument("df must be > 0"); }
        }
    };

    /** @brief Policy struct for Gumbel distribution (default location=0, scale=1). */
    struct Gumbel {
        double location;
        double scale;
        explicit Gumbel(double loc = 0.0, double sc = 1.0) : location(loc), scale(sc) {
            if (sc <= 0.0) { throw std::invalid_argument("scale must be > 0"); }
        }
    };

    /** @brief Policy struct for AR(1) process (default rho=0.5). */
    struct AR1 {
        double rho;
        explicit AR1(double rho = 0.5) : rho(rho) {}
    };

} /* End of namespace predictor_policy */
// ==============================================================================


// ==============================================================================
namespace detail {

    /**
     * @brief Validate the support and coefficient vectors for a given number of predictors.
     *
     * @param support Indices of the predictors to be used.
     * @param coefs Coefficients corresponding to the support indices.
     * @param p Total number of predictors.
     */
    inline void validate_support(
        const std::vector<std::size_t>& support,
        const std::vector<double>& coefs,
        std::size_t p
    ) {
        if (support.size() != coefs.size()) {
            throw std::invalid_argument("Mismatch between support and coefficients size.");
        }
        for (auto idx : support) {
            if (idx >= p) { throw std::out_of_range("Support index out of range [0, p)."); }
        }
    }

    /**
     * @brief Get the base seed object for random number generation, using the provided seed or a
     * random seed from std::random_device.
     *
     * @param seed Base seed for reproducibility. If negative, a random seed from std::random_device
     * is used.
     *
     * @return Base seed for random number generation.
     */
    inline unsigned int get_base_seed(int seed) {
        return (seed >= 0) ? static_cast<unsigned int>(seed) : std::random_device{}();
    }

    /**
     * @brief Generate a predictor matrix based on the specified policy.
     *
     * @tparam MatrixType Type of the matrix to be generated.
     *
     * @param X Matrix to store the generated predictors.
     * @param n Number of rows (samples).
     * @param p Number of columns (predictors).
     * @param base_seed Base seed for random number generation.
     * @param policy Policy specifying the distribution or process for generating predictors.
     */
    template<typename MatrixType>
    inline void generate_predictor_matrix(
        MatrixType& X,
        std::size_t n,
        std::size_t p,
        unsigned int base_seed,
        const predictor_policy::Normal&
    ) {
        dummygen::generate_dummies(X, n, p, base_seed, dummygen::Distribution::Normal());
    }


    /**
     * @brief Generate a predictor matrix based on the Student's t-distribution policy.
     *
     * @tparam MatrixType Type of the matrix to be generated.
     *
     * @param X Matrix to store the generated predictors.
     * @param n Number of rows (samples).
     * @param p Number of columns (predictors).
     * @param base_seed Base seed for random number generation.
     * @param policy Policy specifying the Student's t-distribution parameters.
     */
    template<typename MatrixType>
    inline void generate_predictor_matrix(
        MatrixType& X,
        std::size_t n,
        std::size_t p,
        unsigned int base_seed,
        const predictor_policy::StudentT& policy
    ) {
        dummygen::generate_dummies(X, n, p, base_seed,
            dummygen::Distribution::StudentT(policy.df));
    }


    /**
     * @brief Generate a predictor matrix based on the Gumbel distribution policy.
     *
     * @tparam MatrixType Type of the matrix to be generated.
     *
     * @param X Matrix to store the generated predictors.
     * @param n Number of rows (samples).
     * @param p Number of columns (predictors).
     * @param base_seed Base seed for random number generation.
     * @param policy Policy specifying the Gumbel distribution parameters.
     */
    template<typename MatrixType>
    inline void generate_predictor_matrix(
        MatrixType& X,
        std::size_t n,
        std::size_t p,
        unsigned int base_seed,
        const predictor_policy::Gumbel& policy
    ) {
        dummygen::generate_dummies(X, n, p, base_seed,
            dummygen::Distribution::Gumbel(policy.location, policy.scale));
    }


    /**
     * @brief Generate a predictor matrix based on the AR(1) process policy.
     *
     * @tparam MatrixType Type of the matrix to be generated.
     * @param X Matrix to store the generated predictors.
     * @param n Number of rows (samples).
     * @param p Number of columns (predictors).
     * @param base_seed Base seed for random number generation.
     * @param policy Policy specifying the AR(1) process parameter (rho).
     */
    template<typename MatrixType>
    inline void generate_predictor_matrix(
        MatrixType& X,
        std::size_t n,
        std::size_t p,
        unsigned int base_seed,
        const predictor_policy::AR1& policy
    ) {
        double innovation_std = (std::abs(policy.rho) >= 1.0) ?
                                1.0 :
                                std::sqrt(std::max(0.0, 1.0 - policy.rho * policy.rho));

        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            std::mt19937 row_gen(dummygen::mix_seed(base_seed, i));
            std::normal_distribution<double> dist(0.0, 1.0);
            X(i, 0) = dist(row_gen);
            for (std::size_t j = 1; j < p; ++j) {
                X(i, j) = policy.rho * X(i, j - 1) + innovation_std * dist(row_gen);
            }
        }
    }


    /**
     * @brief Generate a signal vector based on the given predictor matrix and coefficients.
     *
     * @tparam MatrixType Type of the predictor matrix.
     * @tparam VectorType Type of the signal vector.
     *
     * @param y Signal vector to be generated.
     * @param X Predictor matrix.
     * @param support Indices of the predictors to be used.
     * @param coefs Coefficients for the selected predictors.
     */
    template<typename MatrixType, typename VectorType>
    inline void generate_signal(
        VectorType& y,
        const MatrixType& X,
        const std::vector<std::size_t>& support,
        const std::vector<double>& coefs
    ) {
        y.setZero();
        for (std::size_t i = 0; i < support.size(); ++i) { y += coefs[i] * X.col(support[i]); }
    }

} /* End of namespace detail */
// ==============================================================================


// ==============================================================================
// Public Classes for Data Generation
// ==============================================================================

/**
 * @brief Unified generator for in-memory synthetic regression data (with optional dummies).
 */
class SyntheticData {
private:
    Eigen::MatrixXd X_;
    Eigen::MatrixXd D_; // Remains empty (n x 0) if no dummies requested
    Eigen::VectorXd y_;
    double signal_power_;
    double noise_std_;

public:
    /**
     * @brief Constructor WITH explicitly requested dummies
     *
     * @tparam PredictorPolicy Type of the predictor policy.
     * @tparam NoisePolicy Type of the noise policy.
     *
     * @param n Number of samples.
     * @param p Number of predictors.
     * @param num_dummies Number of dummy predictors.
     * @param support Indices of the predictors to be used.
     * @param coefs Coefficients for the selected predictors.
     * @param snr Signal-to-noise ratio.
     * @param seed Base seed for random number generation.
     * @param X_seed Seed for predictor matrix generation.
     * @param dummy_seed Seed for dummy matrix generation.
     * @param predictor_policy Policy for generating predictors.
     * @param dummy_dist Distribution for generating dummies.
     * @param noise_policy Policy for generating noise.
     */
    template <
        typename PredictorPolicy = predictor_policy::Normal,
        typename NoisePolicy = noisegen::noise_policy::Normal
    >
    SyntheticData(
        Eigen::Index n,
        Eigen::Index p,
        Eigen::Index num_dummies,
        const std::vector<std::size_t>& support,
        const std::vector<double>& coefs,
        double snr = 1.0,
        int seed = -1,
        int X_seed = -1,
        int dummy_seed = -1,
        const PredictorPolicy& predictor_policy = PredictorPolicy(),
        const dummygen::Distribution& dummy_dist = dummygen::Distribution::Normal(),
        const NoisePolicy& noise_policy = NoisePolicy()
    ) {
        if (n <= 0) throw std::invalid_argument("n must be positive (got 0)");
        if (p <= 0) throw std::invalid_argument("p must be positive (got 0)");
        detail::validate_support(support, coefs, p);

        unsigned int base_seed = detail::get_base_seed(seed);

        unsigned int x_gen_seed = (X_seed >= 0) ?
                                  static_cast<unsigned int>(X_seed) :
                                  (seed >= 0 ? dummygen::mix_seed(base_seed, 1) :
                                  std::random_device{}());

        unsigned int dummy_gen_seed = (dummy_seed >= 0) ?
                                      static_cast<unsigned int>(dummy_seed) :
                                      (seed >= 0 ? dummygen::mix_seed(base_seed, 2) :
                                      std::random_device{}());

        unsigned int noise_gen_seed = (seed >= 0) ?
                                      dummygen::mix_seed(base_seed, 3) :
                                      std::random_device{}();

        X_.resize(n, p);
        detail::generate_predictor_matrix(X_, n, p, x_gen_seed, predictor_policy);

        if (num_dummies > 0) {
            D_.resize(n, num_dummies);
            dummygen::generate_dummies(D_, n, num_dummies,
                                        dummy_gen_seed, dummy_dist);
        }

        y_.resize(n);
        detail::generate_signal(y_, X_, support, coefs);

        std::tie(signal_power_, noise_std_) =
            noisegen::detail::calculate_noise_params(y_, n, snr);
        noisegen::add_noise(y_, n, noise_std_, noise_policy, noise_gen_seed);
    }


    /**
     * @brief Construct a new Synthetic Data object WITHOUT dummies (Delegates to main constructor)
     *
     * @tparam PredictorPolicy Type of the predictor policy.
     * @tparam NoisePolicy Type of the noise policy.
     *
     * @param n Number of samples.
     * @param p Number of predictors.
     * @param support Indices of the predictors to be used.
     * @param coefs Coefficients for the selected predictors.
     * @param snr Signal-to-noise ratio.
     * @param seed Base seed for random number generation.
     * @param predictor_policy Policy for generating predictors.
     * @param noise_policy Policy for generating noise.
     */
    template <
        typename PredictorPolicy = predictor_policy::Normal,
        typename NoisePolicy = noisegen::noise_policy::Normal
    >
    SyntheticData(
        Eigen::Index n,
        Eigen::Index p,
        const std::vector<std::size_t>& support,
        const std::vector<double>& coefs,
        double snr = 1.0, int seed = -1,
        const PredictorPolicy& predictor_policy = PredictorPolicy(),
        const NoisePolicy& noise_policy = NoisePolicy()
    ) : SyntheticData(n, p, 0, support, coefs, snr, seed, -1, -1, predictor_policy,
        dummygen::Distribution::Normal(), noise_policy) {}

    /** @brief Get the predictor matrix reference (read-only) */
    const Eigen::MatrixXd& getX() const { return X_; }

    /** @brief Get the dummy matrix reference (read-only) */
    const Eigen::MatrixXd& getD() const { return D_; }

    /** @brief Get the response vector reference (read-only) */
    const Eigen::VectorXd& getY() const { return y_; }

    /** @brief Get the predictor matrix reference (modifiable) */
    Eigen::MatrixXd& getX() { return X_; }

    /** @brief Get the dummy matrix reference (modifiable) */
    Eigen::MatrixXd& getD() { return D_; }

    /** @brief Get the response vector reference (modifiable) */
    Eigen::VectorXd& getY() { return y_; }

    /** @brief Check if the dataset has dummy variables */
    bool hasDummies() const { return D_.cols() > 0; }

    /** @brief Get the signal power */
    double getSignalPower() const { return signal_power_; }

    /** @brief Get the noise standard deviation */
    double getNoiseStd() const { return noise_std_; }

    /** @brief Get the signal-to-noise ratio */
    double snr() const { return (noise_std_ > 0) ?
                         signal_power_ / (noise_std_ * noise_std_) : 0.0; }

    /** @brief Get the number of rows */
    Eigen::Index rows() const { return X_.rows(); }

    /** @brief Get the number of predictor columns */
    Eigen::Index cols() const { return X_.cols(); }

    /** @brief Get the number of dummy columns */
    Eigen::Index num_dummies() const { return D_.cols(); }
};


/**
 * @brief Unified generator for memory-mapped synthetic data (with optional dummies).
 */
class SyntheticDataMapped {
private:
    /** @brief Mapped predictor matrix */
    std::unique_ptr<memmap::MemoryMappedMatrix<double>> X_mmap_;

    /** @brief Mapped dummy matrix (nullptr if no dummies requested) */
    std::unique_ptr<memmap::MemoryMappedMatrix<double>> D_mmap_;

    /** @brief Mapped response vector */
    std::unique_ptr<memmap::MemoryMappedMatrix<double>> y_mmap_;

    /** @brief File paths for the memory-mapped matrices */
    std::string X_filepath_, D_filepath_, y_filepath_;

    /** @brief Number of rows */
    Eigen::Index n_;

    /** @brief Number of columns */
    Eigen::Index p_;

    /** @brief Number of dummy variables */
    Eigen::Index num_dummies_;

    /** @brief Signal power*/
    double signal_power_;

    /** @brief Noise standard deviation */
    double noise_std_;

public:
    /**
     * @brief Constructor with explicitly requested dummies
     *
     * @tparam PredictorPolicy Type of distribution for predictor generation.
     * @tparam NoisePolicy Type of distribution for noise generation.
     *
     * @param X_filepath File path for memory-mapped predictor matrix.
     * @param D_filepath File path for memory-mapped dummy matrix (ignored if num_dummies is 0).
     * @param y_filepath File path for memory-mapped response vector.
     * @param n Number of rows (samples).
     * @param p Number of predictor columns.
     * @param num_dummies Number of dummy predictor columns (set to 0 for no dummies).
     * @param support Indices of the true predictors.
     * @param coefs Coefficients for the true predictors (must match size of support).
     * @param snr Desired signal-to-noise ratio (linear). Default is 1.0.
     * @param seed Base random seed for reproducibility (optional). Default is -1, which uses a
     *  random seed from std::random_device.
     * @param X_seed Random seed for predictor generation (optional, overrides base seed).
     *  Default is -1.
     * @param dummy_seed Random seed for dummy generation (optional, overrides base seed).
     *  Default is -1.
     * @param predictor_policy Policy object specifying the distribution for predictor generation.
     *  Default is predictor_policy::Normal().
     * @param dummy_dist Distribution type for dummy generation.
     *  Default is dummygen::Distribution::Normal().
     * @param noise_policy Policy object specifying the distribution for noise generation.
     *  Default is noisegen::noise_policy::Normal().
     */
    template <
        typename PredictorPolicy = predictor_policy::Normal,
        typename NoisePolicy = noisegen::noise_policy::Normal
    >
    SyntheticDataMapped(
        const std::string& X_filepath,
        const std::string& D_filepath,
        const std::string& y_filepath,
        Eigen::Index n,
        Eigen::Index p,
        Eigen::Index num_dummies,
        const std::vector<std::size_t>& support,
        const std::vector<double>& coefs,
        double snr = 1.0,
        int seed = -1,
        int X_seed = -1,
        int dummy_seed = -1,
        const PredictorPolicy& predictor_policy = PredictorPolicy(),
        const dummygen::Distribution& dummy_dist = dummygen::Distribution::Normal(),
        const NoisePolicy& noise_policy = NoisePolicy()
    ) : X_filepath_(X_filepath), D_filepath_(D_filepath), y_filepath_(y_filepath),
        n_(n), p_(p), num_dummies_(num_dummies) {

        detail::validate_support(support, coefs, p);

        unsigned int base_seed = detail::get_base_seed(seed);

        unsigned int x_gen_seed = (X_seed >= 0) ?
            static_cast<unsigned int>(X_seed) :
            (seed >= 0 ? dummygen::mix_seed(base_seed, 1) : std::random_device{}());

        unsigned int dummy_gen_seed = (dummy_seed >= 0) ?
            static_cast<unsigned int>(dummy_seed) :
            (seed >= 0 ? dummygen::mix_seed(base_seed, 2) : std::random_device{}());

        unsigned int noise_gen_seed = (seed >= 0) ?
                                      dummygen::mix_seed(base_seed, 3) :
                                      std::random_device{}();

        // Map X and Y
        X_mmap_ = std::make_unique<memmap::MemoryMappedMatrix<double>>(X_filepath_, n,
            p, memmap::AccessMode::ReadWrite);

        y_mmap_ = std::make_unique<memmap::MemoryMappedMatrix<double>>(y_filepath_, n,
            1, memmap::AccessMode::ReadWrite);

        auto X_map = X_mmap_->getMap();
        auto y_map = y_mmap_->getMap();

        detail::generate_predictor_matrix(X_map, n, p, x_gen_seed, predictor_policy);

        // Map D (Only if requested)
        if (num_dummies > 0 && !D_filepath_.empty()) {
            D_mmap_ = std::make_unique<memmap::MemoryMappedMatrix<double>>(D_filepath_, n,
                num_dummies, memmap::AccessMode::ReadWrite);
            auto D_map = D_mmap_->getMap();
            dummygen::generate_dummies(D_map, n, num_dummies, dummy_gen_seed,
                dummy_dist);
        }

        Eigen::VectorXd y_vec = y_map.col(0);
        detail::generate_signal(y_vec, X_map, support, coefs);

        std::tie(signal_power_, noise_std_) = noisegen::detail::calculate_noise_params(y_vec,
            n, snr);

        noisegen::add_noise(y_vec, n, noise_std_, noise_policy, noise_gen_seed);

        y_map.col(0) = y_vec;
    }

    /**
     * @brief Constructor WITHOUT dummies (Delegates to main constructor with empty D_filepath
     * and 0 dummies).
     *
     * @tparam PredictorPolicy Type of distribution for predictor generation.
     * @tparam NoisePolicy Type of distribution for noise generation.
     *
     * @param X_filepath File path for memory-mapped predictor matrix.
     * @param y_filepath File path for memory-mapped response vector.
     * @param n Number of rows (samples).
     * @param p Number of predictor columns.
     * @param support Indices of the true predictors.
     * @param coefs Coefficients for the true predictors (must match size of support).
     * @param snr Desired signal-to-noise ratio (linear). Default is 1.0.
     * @param seed Base random seed for reproducibility (optional). Default is -1, which uses a
     * random seed from std::random_device.
     */
    template <
        typename PredictorPolicy = predictor_policy::Normal,
        typename NoisePolicy = noisegen::noise_policy::Normal
    >
    SyntheticDataMapped(
        const std::string& X_filepath,
        const std::string& y_filepath,
        Eigen::Index n,
        Eigen::Index p,
        const std::vector<std::size_t>& support,
        const std::vector<double>& coefs,
        double snr = 1.0,
        int seed = -1,
        const PredictorPolicy& predictor_policy = PredictorPolicy(),
        const NoisePolicy& noise_policy = NoisePolicy()
    ) :
    SyntheticDataMapped(X_filepath, "", y_filepath, n, p, 0, support, coefs, snr, seed, -1, -1,
         predictor_policy, dummygen::Distribution::Normal(), noise_policy) {}

    /** @brief Get the memory-mapped predictor matrix. */
    auto getX() { return X_mmap_->getMap(); }

    /** @brief Get the memory-mapped predictor matrix (read-only). */
    auto getX() const { return X_mmap_->getMap(); }

    /** @brief Get the memory-mapped dummy matrix. */
    auto getD() {
        if (!D_mmap_) throw std::runtime_error("Dummy matrix was not allocated/mapped.");
        return D_mmap_->getMap();
    }

    /** @brief Get the memory-mapped dummy matrix (read-only). */
    auto getD() const {
        if (!D_mmap_) throw std::runtime_error("Dummy matrix was not allocated/mapped.");
        return D_mmap_->getMap();
    }

    /** @brief Get the memory-mapped response vector. */
    auto getY() { return y_mmap_->getMap(); }

    /** @brief Get the memory-mapped response vector (read-only). */
    auto getY() const { return y_mmap_->getMap(); }

    /** @brief Check if the dummy matrix is allocated/mapped. */
    bool hasDummies() const { return D_mmap_ != nullptr; }

    /** @brief Get the signal power of the generated data. */
    double getSignalPower() const { return signal_power_; }

    /** @brief Get the standard deviation of the noise in the generated data. */
    double getNoiseStd() const { return noise_std_; }

    /** @brief Get the signal-to-noise ratio of the generated data. */
    double snr() const { return (noise_std_ > 0) ?
                         signal_power_ / (noise_std_ * noise_std_) : 0.0; }

    /** @brief Get the number of rows (samples) in the generated data. */
    Eigen::Index rows() const { return n_; }

    /** @brief Get the number of predictor columns in the generated data. */
    Eigen::Index cols() const { return p_; }

    /** @brief Get the number of dummy predictors in the generated data. */
    Eigen::Index num_dummies() const { return num_dummies_; }

    /** @brief Get the file path of the memory-mapped predictor matrix. */
    const std::string& X_filepath() const { return X_filepath_; }

    /** @brief Get the file path of the memory-mapped dummy matrix. */
    const std::string& D_filepath() const { return D_filepath_; }

    /** @brief Get the file path of the memory-mapped response vector. */
    const std::string& y_filepath() const { return y_filepath_; }
};


// ==============================================================================
// Public Utility Functions
// ==============================================================================

/**
 * @brief Create a dummy matrix object
 *
 * @param n Number of rows (samples).
 * @param num_dummies Number of dummy predictors (columns).
 * @param seed Random seed for dummy generation.
 * @param dummy_dist Distribution type for dummy generation.
 * @return Eigen::MatrixXd The generated dummy matrix.
 */
inline Eigen::MatrixXd create_dummy_matrix(
    std::size_t n,
    std::size_t num_dummies,
    int seed = -1,
    const dummygen::Distribution& dummy_dist = dummygen::Distribution::Normal()
) {
    Eigen::MatrixXd D(n, num_dummies);
    unsigned int base_seed = detail::get_base_seed(seed);
    dummygen::generate_dummies(D, n, num_dummies, base_seed, dummy_dist);
    return D;
}

/**
 * @brief Create a mapped dummy matrix object
 *
 * @param D_filepath File path for memory-mapped dummy matrix.
 * @param n  Number of rows (samples).
 * @param num_dummies Number of dummy predictors (columns).
 * @param seed Random seed for dummy generation.
 * @param dummy_dist Distribution type for dummy generation.
 */
inline void create_mapped_dummy_matrix(
    const std::string& D_filepath,
    std::size_t n,
    std::size_t num_dummies,
    int seed = -1,
    const dummygen::Distribution& dummy_dist = dummygen::Distribution::Normal()
) {
    memmap::MemoryMappedMatrix<double> D_mmap(
        D_filepath,
        n,
        num_dummies,
        memmap::AccessMode::ReadWrite
    );
    auto D_map = D_mmap.getMap();
    unsigned int base_seed = detail::get_base_seed(seed);
    dummygen::generate_dummies(D_map, n, num_dummies, base_seed, dummy_dist);
}

} /* End of namespace trex::utils::datageneration::datagen */
// ==============================================================================

#endif /* UTILS_DATAGENERATION_DATAGEN_HPP */
