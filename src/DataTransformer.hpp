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
 */
class DataTransformer {
protected:
    std::vector<std::size_t> dropped_indices_;
    bool fitted_{false};

    // Protected constructors to prevent slicing
    DataTransformer() = default;
    DataTransformer(const DataTransformer&) = default;
    DataTransformer& operator=(const DataTransformer&) = default;
    DataTransformer(DataTransformer&&) noexcept = default;
    DataTransformer& operator=(DataTransformer&&) noexcept = default;

public:
    /** @brief Public metadata (not polymorphic behavior) */
    std::vector<std::string> feature_names;

    virtual ~DataTransformer() = default;

    // ===== Core Interface =====

    virtual DataTransformer& fit(
        const Eigen::Map<const Eigen::MatrixXd>& X,
        double threshold = 1e-12) = 0;

    virtual void transform_inplace(Eigen::Map<Eigen::MatrixXd>& X) const = 0;
    virtual void inverse_transform_inplace(
        Eigen::Map<Eigen::MatrixXd>& X_transformed) const = 0;

    // ===== Accessors =====

    virtual const Eigen::VectorXd& get_means() const = 0;
    virtual const Eigen::VectorXd& get_scales() const = 0;
    virtual std::string get_name() const = 0;

    const std::vector<std::size_t>& get_dropped_indices() const noexcept {
        return dropped_indices_;
    }

    bool is_fitted() const noexcept { return fitted_; }

    std::size_t get_num_features() const noexcept {
        return get_feature_dimension();
    }

    // ===== Polymorphic Copying =====

    virtual std::unique_ptr<DataTransformer> clone() const = 0;

    // ===== Persistence =====

    /**
     * @brief Save transformer to binary file.
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

        // Serialize the object (const_cast needed for archive)
        const_cast<DataTransformer*>(this)->serialize(archive);
    }

    /**
     * @brief Load transformer from binary file.
     */
    void load(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) {
            throw std::runtime_error(
                "DataTransformer::load: Cannot open '" + filename + "'");
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

    // ===== Utility =====

    virtual Eigen::VectorXd get_l2_conversion_factor(
        std::size_t n_samples) const = 0;

protected:
    virtual std::size_t get_feature_dimension() const = 0;

    /**
     * @brief Serialize transformer state.
     *
     * Implemented by derived classes using CEREAL_NVP.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(dropped_indices_),
                CEREAL_NVP(fitted_),
                CEREAL_NVP(feature_names)
            );
    }

    // Grant Cereal access
    friend class cereal::access;

    /**
     * @brief Check if fitted before operations.
     */
    void check_fitted(const std::string& method_name) const {
        if (!fitted_) {
            throw std::logic_error(method_name + ": Transformer not fitted");
        }
    }

    /**
     * @brief Validate dimension consistency.
     */
    void check_dimension(std::size_t expected,
                         std::size_t actual,
                         const std::string& method_name) const {
        if (expected != actual) {
            throw std::invalid_argument(
                method_name + ": Expected " + std::to_string(expected) +
                " features, got " + std::to_string(actual));
        }
    }
};

#endif
