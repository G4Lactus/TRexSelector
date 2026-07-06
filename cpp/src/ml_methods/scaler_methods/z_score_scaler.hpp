// ===================================================================================
// z_score_scaler.hpp
// ===================================================================================
/**
 * @file z_score_scaler.hpp
 *
 * @brief ZScoreScaler class for z-score normalization.
 *
 * @details The class is separated into a header (z_score_scaler.hpp) and source file
 * (z_score_scaler.cpp) for better organization.
 * |--ZScoreScaler
 *    |--z_score_scaler.hpp  (class declaration)
 *    |--z_score_scaler.cpp  (class implementation)
 */
// ===================================================================================
#ifndef ML_METHODS_SCALER_METHODS_Z_SCORE_SCALER_HPP
#define ML_METHODS_SCALER_METHODS_Z_SCORE_SCALER_HPP
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
 * @class ZScoreScaler
 *
 * @brief Standard scaling (Z-score normalization): x' = (x - μ) / σ
 *
 * @details Centers columns to mean 0 and scales them to unit standard
 * deviation, with the Bessel correction (divides by n - 1). The `center` and
 * `scale` switches follow R's scale(): with `center = false, scale = true`
 * columns are divided by the root-mean-square around 0 (R's documented
 * behavior), not by the standard deviation.
 */
class ZScoreScaler : public DataTransformer {

public:

    // =========================================================================
    // Constructor
    // =========================================================================

    /**
     * @brief Construct a new standard scaler object.
     *
     * @param center If true, center columns to mean = 0.
     * @param scale If true, scale columns to unit SD (Bessel correction);
     *              around 0 instead of the mean when `center` is false.
     */
    explicit ZScoreScaler(bool center = true, bool scale = true)
        : DataTransformer(center, scale) {}

    // ========================================================================
    // Identification
    // ========================================================================

    /**
     * @brief Get the transformer name for serialization.
     *
     * @return "ZScoreScaler"
     */
    std::string get_name() const override { return "ZScoreScaler"; }

protected:

    // ========================================================================
    // Template-Method Hooks
    // ========================================================================

    /** @brief Bessel-corrected standard deviation around the applied center. */
    double computeScaleStat(const Eigen::Ref<const Eigen::VectorXd>& col,
                            double center) const override;

    /** @brief The Bessel correction (n - 1) requires at least 2 samples. */
    void checkFitPreconditions(Eigen::Index nrows) const override;

    // ========================================================================
    // De-/Serialization
    // ========================================================================

    void serialize_to(cereal::PortableBinaryOutputArchive& archive) const override {
        const_cast<ZScoreScaler*>(this)->serialize(archive);
    }

    void serialize_from(cereal::PortableBinaryInputArchive& archive) override {
        serialize(archive);
    }

    /**
     * @brief Serialize scaler state using Cereal NVPs.
     *
     * @param archive Cereal archive (input or output).
     */
    template<class Archive>
    void serialize(Archive& archive) {
        // The z-score scaler carries no state beyond the base class
        DataTransformer::serialize(archive);
    }

    // Grant Cereal access to protected serialize
    friend class cereal::access;
};

// ===================================================================================
} /* End of namespace scaler_methods */
} /* End of namespace ml_methods */
} /* End of namespace trex */
// ===================================================================================
#endif /* End of ML_METHODS_SCALER_METHODS_Z_SCORE_SCALER_HPP */
