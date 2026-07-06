// ===================================================================================
// z_score_scaler.hpp
// ===================================================================================
/**
 * @file z_score_scaler.hpp
 *
 * @brief ZScoreScaler class for z-score normalization.
 *
 * @details The class is separated into a header (ZScoreScaler.hpp) and source file
 * (ZScoreScaler.cpp) for better organization.
 * |--ZScoreScaler
 *    |--ZScoreScaler.hpp  (class declaration)
 *    |--ZScoreScaler.cpp  (class implementation)
 */
// ===================================================================================
#ifndef ML_METHODS_SCALER_METHODS_Z_SCORE_SCALER_HPP
#define ML_METHODS_SCALER_METHODS_Z_SCORE_SCALER_HPP
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
 * @class ZScoreScaler
 *
 * @brief Standard scaling (Z-score normalization): x' = (x - μ) / σ
 *
 * Centers data to column mean = 0 and scales columns to unit variance.
 * Standard deviation uses Bessel correction (divides by n-1).
 */
class ZScoreScaler : public DataTransformer {
private:

    // ========================================================================
    // State Variables
    // ========================================================================

    /** @brief Column means (zero if with_mean_ is false). */
    Eigen::VectorXd means_;

    /** @brief Column standard deviations (Bessel correction applied if with_std_ is true).
     *  @note Bessel correction divides by (n-1) instead of n for an unbiased estimate.
     */
    Eigen::VectorXd scales_;

    /** @brief  Whether to center data to mean = 0 */
    bool with_mean_{true};

    /** @brief  Whether to scale data to std = 1 */
    bool with_std_{true};

public:

    // =========================================================================
    // Constructor
    // =========================================================================

    /**
     * @brief Construct a new standard scaler object.
     *
     * @param with_mean If true, center data to column mean = 0.
     * @param with_std If true, scale data to std = 1 (using Bessel correction).
     *
     */
    explicit ZScoreScaler(bool with_mean = true, bool with_std = true)
        : with_mean_(with_mean), with_std_(with_std) {}


    // ========================================================================
    // Core Interface
    // ========================================================================

    /**
     * @brief Fit the scaler to the input data (read-only operation).
     *
     * @param X Input matrix (rows = samples, columns = features).
     * @param min_std_threshold Minimum std deviation for numerical stability.
     */
    DataTransformer& fit(Eigen::Map<Eigen::MatrixXd>& X,
                         double min_std_threshold = 1e-12) override;

    /**
     * @brief Apply the scaling in place.
     *
     * Transforms: z = (x - μ) / σ
     *
     * @param X Input/output matrix to be scaled.
     */
    void transform_inplace(Eigen::Map<Eigen::MatrixXd>& X) const override;

    /**
     * @brief Apply the inverse scaling in place.
     *
     * Transforms: x = z * σ + μ
     *
     * @param X_scaled Input/output matrix to be inverse scaled.
     */
    void inverse_transform_inplace(Eigen::Map<Eigen::MatrixXd>& X_scaled) const override;


    // ========================================================================
    // Accessors
    // ========================================================================

    /**
     * @brief Get column means (zeros if not centered).
     *
     * @return Vector of means for each column.
     */
    const Eigen::VectorXd& get_means() const noexcept override { return means_; }

    /**
     * @brief Get the column standard deviations (scales).
     *
     * @return Vector of standard deviations for each column.
     */
    const Eigen::VectorXd& get_scales() const noexcept override { return scales_; }

    /**
     * @brief Get the transformer name for serialization.
     *
     * @return "ZScoreScaler"
     */
    std::string get_name() const override { return "ZScoreScaler"; }

    /**
     * @brief Check if centering is enabled.
     *
     * @return True if data is centered to mean = 0.
     */
    bool get_with_mean() const noexcept{ return with_mean_; }

    /**
     * @brief Check if scaling to unit variance is enabled.
     *
     * @return True if data is scaled to std = 1.
     */
    bool get_with_std() const noexcept { return with_std_; }

    // ========================================================================
    // De-/Serialization
    // ========================================================================

    void save(const std::string& filename) const override {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error(
                get_name() + "::save: Cannot open '" + filename + "' for writing"
            );
        }
        cereal::PortableBinaryOutputArchive archive(ofs);
        std::string type_name = get_name();
        archive(
            CEREAL_NVP(type_name)
        );
        const_cast<ZScoreScaler*>(this)->serialize(archive);
    }

    void load(const std::string& filename) override {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) {
            throw std::runtime_error(
                get_name() + "::load: Cannot open '" + filename + "' for reading"
            );
        }
        cereal::PortableBinaryInputArchive archive(ifs);
        std::string type_name;
        archive(
            CEREAL_NVP(type_name)
        );

        if (type_name != get_name()) {
            throw std::runtime_error(
                get_name() + "::load: Type mismatch. Expected '" + get_name() +
                "', got '" + type_name + "'"
            );
        }

        serialize(archive);
        if (!is_fitted()) {
            throw std::runtime_error(get_name() + "::load: Loaded state is not fitted");
        }
    }


protected:

    // ========================================================================
    // De-/Serialization
    // ========================================================================

    /**
     * @brief Serialize scaler state using Cereal NVPs.
     *
     * @param archive Cereal archive (input or output).
     */
    template<class Archive>
    void serialize(Archive& archive) {
        // Serialize base class members first
        DataTransformer::serialize(archive);

        // Serialize derived class members
        archive(
            CEREAL_NVP(means_),
            CEREAL_NVP(scales_),
            CEREAL_NVP(with_mean_),
            CEREAL_NVP(with_std_)
        );
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
