// ===================================================================================
// lp_norm_scaler.hpp
// ===================================================================================
/**
 * @file lp_norm_scaler.hpp
 *
 * @brief Declaration of the LpNormScaler class for L1/L2 normalization and centering.
 *
 * @details The class is separated into a header (lp_norm_scaler.hpp) and source file
 *  (lp_norm_scaler.cpp) for better organization.
 * |--LpNormScaler
 *    |--lp_norm_scaler.hpp  (class declaration)
 *    |--lp_norm_scaler.cpp  (class implementation)
 */
// ===================================================================================
#ifndef ML_METHODS_SCALER_METHODS_LP_NORM_SCALER_HPP
#define ML_METHODS_SCALER_METHODS_LP_NORM_SCALER_HPP
// ===================================================================================

// std includes
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
 * @details Centers columns (optional) and scales them to unit Lp norm, where
 * the norm is computed around the applied center (around 0 when `center` is
 * false). Designed for high-dimensional data (p >> n) with in-place
 * transformations.
 */
class LpNormScaler : public DataTransformer {
public:
    enum class NormType { L1, L2 };

private:
    // ========================================================================
    // State Variables
    // ========================================================================

    /** @brief Normalization type (L1 or L2) */
    NormType norm_type_{NormType::L2};

public:
    // ========================================================================
    // Constructor
    // ========================================================================

    /**
     * @brief Construct a normalizer object.
     *
     * @param norm L1 or L2 normalization.
     * @param center If true, center columns to mean = 0.
     * @param scale If true, scale columns to unit Lp norm (computed around
     *              the applied center; around 0 when `center` is false).
     */
    explicit LpNormScaler(NormType norm = NormType::L2,
                          bool center = true,
                          bool scale = true)
        : DataTransformer(center, scale), norm_type_(norm) {}

    // ========================================================================
    // Identification & Accessors
    // ========================================================================

    /** @brief Get transformer name for serialization */
    std::string get_name() const override { return "LpNormScaler"; }

    /** @brief Get normalization type (L1 or L2) */
    NormType get_norm_type() const noexcept { return norm_type_; }

protected:

    // ========================================================================
    // Template-Method Hooks
    // ========================================================================

    /** @brief Lp norm (L1 or L2) around the applied center. */
    double computeScaleStat(const Eigen::Ref<const Eigen::VectorXd>& col,
                            double center) const override;

    // ========================================================================
    // De-/Serialization
    // ========================================================================

    void serialize_to(cereal::PortableBinaryOutputArchive& archive) const override {
        const_cast<LpNormScaler*>(this)->serialize(archive);
    }

    void serialize_from(cereal::PortableBinaryInputArchive& archive) override {
        serialize(archive);
    }

    /**
     * @brief Serialize normalizer state.
     *
     * Serializes both base class and derived class members.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        DataTransformer::serialize(archive);

        archive(
            CEREAL_NVP(norm_type_)
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
