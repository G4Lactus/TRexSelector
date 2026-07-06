// ===================================================================================
// lp_norm_scaler.hpp
// ===================================================================================
/**
 * @file lp_norm_scaler.hpp
 *
 * @brief Declaration of the LpNormScaler class for L1/L2 normalization and centering.
 *
 * @details The class is separated into a header (LpNormScaler.hpp) and source file
 *  (LpNormScaler.cpp) for better organization.
 * |--LpNormScaler
 *    |--LpNormScaler.hpp  (class declaration)
 *    |--LpNormScaler.cpp  (class implementation)
 * Serialization is implemented in the header to allow for template binding with Cereal.
 */
// ===================================================================================
#ifndef ML_METHODS_SCALER_METHODS_LP_NORM_SCALER_HPP
#define ML_METHODS_SCALER_METHODS_LP_NORM_SCALER_HPP
// ===================================================================================

// std includes
#include <fstream>
#include <string>

// ml_methods includes
#include <ml_methods/scaler_methods/data_transformer.hpp>

// ===================================================================================

// namespace embedding: trex::ml_methods::scaler_methods
namespace trex {
namespace ml_methods {
namespace scaler_methods {

// ===================================================================================

/**
 * @brief L1/L2 normalization: x' = (x - μ) / ||x - μ||_p
 *
 * Centers data (optional) and normalizes columns to unit norm.
 * Designed for high-dimensional data (p >> n) with in-place transformations.
 */
class LpNormScaler : public DataTransformer {
public:
    enum class NormType { L1, L2 };

private:
    // ========================================================================
    // State Variables
    // ========================================================================

    /** @brief Column means (zero if with_mean_ is false) */
    Eigen::VectorXd means_;

    /** @brief Column norms (L1 or L2, depending on norm_type_) */
    Eigen::VectorXd norms_;

    /** @brief Normalization type (L1 or L2) */
    NormType norm_type_{NormType::L2};

    /** @brief Whether to center data to mean = 0 */
    bool with_mean_{true};

    /** @brief Whether to scale data to unit norm */
    bool with_norm_{true};

public:
    // ========================================================================
    // Constructor
    // ========================================================================

    /**
     * @brief Construct a normalizer object.
     *
     * @param norm L1 or L2 normalization
     * @param with_mean If true, center data to mean = 0
     * @param with_norm If true, scale columns to unit norm
     */
    explicit LpNormScaler(NormType norm = NormType::L2,
                        bool with_mean = true,
                        bool with_norm = true)
        : norm_type_(norm), with_mean_(with_mean), with_norm_(with_norm) {}

    // ========================================================================
    // Core Interface
    // ========================================================================

    /**
     * @brief Fit the normalizer to the input data (read-only operation).
     *
     * @param X Input data matrix (n_samples x n_features)
     * @param min_norm_threshold Minimum threshold for numerical stability
     *
     * @return Reference to this for method chaining
     */
    DataTransformer& fit(
        Eigen::Map<Eigen::MatrixXd>& X,
        double min_norm_threshold = 1e-12) override;

    /**
     * @brief Apply normalization in place (modifies X).
     *
     * @param X Input/output matrix to be normalized
     */
    void transform_inplace(Eigen::Map<Eigen::MatrixXd>& X) const override;

    /**
     * @brief Apply inverse normalization in place (modifies X).
     *
     * @param X_normed Input/output matrix to be inverse normalized
     */
    void inverse_transform_inplace(Eigen::Map<Eigen::MatrixXd>& X_normed) const override;

    // ========================================================================
    // Accessors
    // ========================================================================

    /** @brief Get column means or centers (zeros if not centered) */
    const Eigen::VectorXd& get_means() const noexcept override { return means_; }

    /** @brief Get column scales or norms */
    const Eigen::VectorXd& get_scales() const noexcept override { return norms_; }

    /** @brief Get transformer name for serialization */
    std::string get_name() const override { return "LpNormScaler"; }

    /** @brief Get normalization type (L1 or L2) */
    NormType get_norm_type() const noexcept { return norm_type_; }

    /** @brief Check if normalizer centers data to column mean = 0 */
    bool get_with_mean() const noexcept { return with_mean_; }

    /** @brief Check if normalizer scales columns to unit norm */
    bool get_with_norm() const noexcept { return with_norm_; }

    // ========================================================================
    // De-/Serialization (Public Interface)
    // ========================================================================

    void save(const std::string& filename) const override {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error(get_name() + "::save: Cannot open'" +
                                     filename + "' for writing");
        }
        cereal::PortableBinaryOutputArchive archive(ofs);
        std::string type_name = get_name();
        archive(
            CEREAL_NVP(type_name)
        );
        const_cast<LpNormScaler*>(this)->serialize(archive);
    }

    void load(const std::string& filename) override {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) {
            throw std::runtime_error(get_name() + "::load: Cannot open '" +
                                     filename + "' for reading");
        }
        cereal::PortableBinaryInputArchive archive(ifs);
        std::string type_name;
        archive(
            CEREAL_NVP(type_name)
        );

        std::string expected_name = get_name();
        if (type_name != expected_name) {
            throw std::runtime_error("Type mismatch: expected '" + get_name() +
                                     "', got '" + type_name + "'");
        }

        serialize(archive);
        if (!is_fitted()) {
            throw std::runtime_error(get_name() + "::load: Loaded state is not fitted");
        }
    }


protected:
    // ========================================================================
    // Serialization
    // ========================================================================

    /**
     * @brief Serialize normalizer state.
     *
     * Serializes both base class and derived class members.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        DataTransformer::serialize(archive);

        archive(
            CEREAL_NVP(means_),
            CEREAL_NVP(norms_),
            CEREAL_NVP(norm_type_),
            CEREAL_NVP(with_mean_),
            CEREAL_NVP(with_norm_)
        );
    }

    friend class cereal::access;
};

// ===================================================================================
} /* End of namespace scaler_methods */
} /* End of namespace ml_methods */
} /* End of namespace trex */
// ===================================================================================
#endif /* End of ML_METHODS_SCALER_METHODS_LP_NORM_SCALER_HPP */
