#ifndef DATA_TRANSFORMER_HPP
#define DATA_TRANSFORMER_HPP

#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/polymorphic.hpp>

#include "utils_cereal_eigen.hpp"


/**
 * @brief Abstract base class for column-wise data transformations.
 *
 * Supports both in-memory and memory-mapped data via Eigen::Map.
 * All transformations operate column-wise and support high-dimensional data (p>>n) with
 * in-place operations.
 *
 *
 * @example In-memory usage:
 * @code
 * double* data = new double[n * p];
 * Eigen::Map<Eigen::MatrixXd> X(data, n, p);
 *
 * Normalizer normalizer;
 * normalizer.fit(X);                        // learn statistics
 * normalizer.transform_inplace(X);          // transform X in place
 * normalizer.inverse_transform_inplace(X);  // revert transformation for X in place
 * @endcode
 *
 * @example Memory-mapped usage:
 * @code
 * int fd = open("data.bin", O_RDWR);
 * double* mmap_ptr = static_cast<double*>(
 *     mmap(nullptr, n * p * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
 *
 * Eigen::Map<Eigen::MatrixXd> X(mmap_ptr, n, p);
 *
 * Normalizer normalizer;
 * normalizer.fit(X);                        // learn statistics (read-only operation)
 * normalizer.transform_inplace(X);          // transform X in place (modifies file)
 * normalizer.inverse_transform_inplace(X);  // revert transformation for X in place
 *
 * munmap(mmap_ptr, n * p * sizeof(double));
 * close(fd);
 * @endcode
 *
 */
class DataTransformer {

protected:

    // ========================================================================
    // State Variabeles
    // ========================================================================

    /** @brief Indices of dropped features due to zero variance/norm */
    std::vector<std::size_t> dropped_indices_;

    /** @brief Whether the transformer has been fitted */
    bool fitted_{false};

    // ========================================================================
    // Protected constructors to support inheritance and prevent slicing
    // ========================================================================

    /** @brief Default constructor */
    DataTransformer() = default;

    /** @brief Default copy constructor */
    DataTransformer(const DataTransformer&) = default;

    /** @brief Default copy assignment */
    DataTransformer& operator=(const DataTransformer&) = default;

    /** @brief Default move constructor */
    DataTransformer(DataTransformer&&) noexcept = default;

    /** @brief Default move assignment */
    DataTransformer& operator=(DataTransformer&&) noexcept = default;

public:

    /** @brief Virtual destructor */
    virtual ~DataTransformer() = default;

    // ========================================================================
    // Core Interface
    // ========================================================================

    /** @brief Fit the transformer to the input data (read-only)
     *
     * Computes statistics (means, scales, norms) without modifying X.
     *
     * @param X Input data matrix (n_samples x n_features)
     * @param threshold Minimum threshold for numerical stability
     *
     * @return Reference to the method for chaining.
     */
    virtual DataTransformer& fit(Eigen::Map<Eigen::MatrixXd>& X,
                                 double threshold = 1e-12) = 0;

    /** @brief Applies the transformation in place (modifies X)
     *
     * @param X Input/output matrix to transform
     */
    virtual void transform_inplace(
        Eigen::Map<Eigen::MatrixXd>& X) const = 0;

    /**
     * @brief Applies the inverse transformation in place (modifies X)
     *
     * @param X_transformed Input/output matrix to inverse transform
     *
     */
    virtual void inverse_transform_inplace(
        Eigen::Map<Eigen::MatrixXd>& X_transformed) const = 0;

    // ========================================================================
    // Accessors
    // ========================================================================

    /** @brief Get column means or centers (zeros if not centered) */
    virtual const Eigen::VectorXd& get_means() const = 0;

    /** @brief Get column scales or norms */
    virtual const Eigen::VectorXd& get_scales() const = 0;

    /** @brief Get transformer name for serialization */
    virtual std::string get_name() const = 0;

    /** @brief Get indices of dropped features (empty if none) */
    const std::vector<std::size_t>& get_dropped_indices() const noexcept {
        return dropped_indices_;
    }

    /** @brief Check if transformer has been fitted */
    bool is_fitted() const noexcept {
        return fitted_;
    }


    // ========================================================================
    // De-/Serialization (Public Interface)
    // ========================================================================

    /**
     * @brief Save transformer state to binary file.
     *
     * @param filename Path to output file
     * @throws std::runtime_error if file cannot be opened
     */
    void save(const std::string& filename) const {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs.is_open()) {
            throw std::runtime_error(
                get_name() + "::save: Cannot open '" + filename + "'");
        }

        cereal::PortableBinaryOutputArchive archive(ofs);

        // Save type identifier for polymorphic loading
        std::string type_name = get_name();
        archive(CEREAL_NVP(type_name));

        // Serialize the object (const_cast needed for cereal)
        const_cast<DataTransformer*>(this)->serialize(archive);
    }

    /**
     * @brief Load transformer state from binary file.
     *
     * @param filename Path to input file
     * @throws std::runtime_error if file cannot be opened or type mismatch
     */
    void load(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) {
            throw std::runtime_error(
                get_name() +  "::load: Cannot open '" + filename + "'");
        }

        cereal::PortableBinaryInputArchive archive(ifs);

        // Load and validate type
        std::string type_name;
        archive(CEREAL_NVP(type_name));

        if (type_name != get_name()) {
            throw std::runtime_error(
                "Type mismatch: expected " + get_name() +
                ", got " + type_name);
        }

        serialize(archive);

        if (!is_fitted()) {
            throw std::runtime_error(
                get_name() + "::load: Loaded transformer not fitted");
        }
    }

protected:

    // ========================================================================
    // De-/Serialization
    // ========================================================================

    /**
     * @brief Serialize transformer state.
     *
     * Implemented by derived classes using CEREAL_NVP.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(dropped_indices_),
                CEREAL_NVP(fitted_)
            );
    }

    // Grant Cereal access
    friend class cereal::access;
};

#endif
