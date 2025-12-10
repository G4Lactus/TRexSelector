#ifndef NORMALIZER_HPP
#define NORMALIZER_HPP

#include "DataTransformer.hpp"

/**
 * @brief L1/L2 normalization: x' = (x - μ) / ||x - μ||_p
 *
 * Centers data (optional) and normalizes columns to unit norm.
 * Designed for high-dimensional data (p >> n) with in-place transformations.
 */
class Normalizer : public DataTransformer {
public:
    enum class NormType { L1, L2 };

private:
    /** @brief Column means (zero if with_mean_ is false). */
    Eigen::VectorXd means_;

    /** @brief Column norms (L1 or L2, depending on norm_type_). */
    Eigen::VectorXd norms_;

    /** @brief Normalization type (L1 or L2). */
    NormType norm_type_{NormType::L2};

    /** @brief  Whether to center data to mean = 0. */
    bool with_mean_{true};

    /** @brief  Whether to scale data to unit norm. */
    bool with_norm_{true};

public:
    /**
     * @brief Construct a normalizer object.
     *
     * @param norm L1 or L2 normalization.
     * @param with_mean If true, center data to mean = 0.
     * @param with_norm If true, scale columns to unit norm.
     */
    explicit Normalizer(NormType norm = NormType::L2,
                        bool with_mean = true,
                        bool with_norm = true)
        : norm_type_(norm), with_mean_(with_mean), with_norm_(with_norm) {}

    // ===== Core Interface =====

    /**
     * @brief Fit the normalizer to the input data.
     *
     * @param X Input matrix.
     * @param min_norm_threshold Minimum threshold for numerical stability.
     *
     * @return Reference to the fitted normalizer for chaining.
     */
    DataTransformer& fit(const Eigen::Map<const Eigen::MatrixXd>& X,
                         double min_norm_threshold = 1e-12) override;

    /**
     * @brief Apply the normalization in place.
     *
     * @param X Input/output matrix to be normalized.
     */
    void transform_inplace(Eigen::Map<Eigen::MatrixXd>& X) const override;

    /**
     * @brief Apply the inverse normalization in place.
     *
     * @param X_normed Input/output matrix to be inverse normalized.
     */
    void inverse_transform_inplace(Eigen::Map<Eigen::MatrixXd>& X_normed) const override;

    // ===== Accessors =====

    /**
     * @brief Get the means of the normalization.
     *
     * @return Vector of means for each column.
     */
    const Eigen::VectorXd& get_means() const noexcept override { return means_; }

    /**
     * @brief Get the norms of the normalization.
     *
     * @return Vector of norms for each column.
     */
    const Eigen::VectorXd& get_scales() const noexcept override { return norms_; }

    /**
     * @brief Get the name of the normalizer.
     *
     * @return Name of the normalizer as a string.
     */
    std::string get_name() const override { return "Normalizer"; }

    /**
     * @brief Get the normalization type (L1 or L2).
     *
     * @return The normalization type.
     */
    NormType get_norm_type() const noexcept {return norm_type_; }

    /**
     * @brief Check if the normalizer centers data to column mean = 0.
     *
     * @return True if centering is enabled, false otherwise.
     */
    bool get_with_mean() const noexcept { return with_mean_; }

    /**
     * @brief Check if the normalizer scales columns to unit norm.
     *
     * @return True if scaling is enabled, false otherwise.
     */
    bool get_with_norm() const noexcept { return with_norm_; }

    // ===== Utility =====

    /**
     * @brief Get the l2 conversion factor for coefficients.
     *
     * For algorithms requiring standardized data (z-scale), this recovers
     * the L2 scale of regression coefficients.
     *
     * @param n_samples Number of training samples.
     *
     * @return Per-feature conversion factor
     */
    Eigen::VectorXd get_l2_conversion_factor(std::size_t n_samples) const override;

    /**
     * @brief Create a deep copy of the normalizer.
     *
     * @return A unique pointer to the cloned normalizer.
     */
    std::unique_ptr<DataTransformer> clone() const override {
        return std::make_unique<Normalizer>(*this);
    }

protected:
    /**
     * @brief Get the number of features.
     *
     * @return Number of features.
     */
    std::size_t get_feature_dimension() const override {
        return norms_.size();
    }

    /**
     * @brief Serialize normalizer state using Cereal NVPs.
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
            CEREAL_NVP(norms_),
            CEREAL_NVP(norm_type_),
            CEREAL_NVP(with_mean_),
            CEREAL_NVP(with_norm_)
        );
    }

    // Grant Cereal access
    friend class cereal::access;

};

#endif
