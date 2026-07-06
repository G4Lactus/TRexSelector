// ===================================================================================
// data_transformer.hpp
// ===================================================================================
/**
 * @file data_transformer.hpp
 *
 * @brief Abstract base class for column-wise data normalization and transformation.
 */
// ===================================================================================
#ifndef ML_METHODS_SCALER_METHODS_DATA_TRANSFORMER_HPP
#define ML_METHODS_SCALER_METHODS_DATA_TRANSFORMER_HPP
// ===================================================================================

// std includes
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// Cereal serialization includes
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/polymorphic.hpp>

// utils includes
#include <utils/serialization/utils_cereal_eigen.hpp>

// ===================================================================================

// Namespace embedding: trex::ml_methods::scaler_methods
namespace trex {
namespace ml_methods {
namespace scaler_methods {

// ===================================================================================

/**
 * @brief Abstract base class for column-wise data transformations.
 *
 * @details Implements the shared center/scale machinery once (template-method
 * pattern): fit() computes per-column centers and scale statistics, and
 * transform_inplace() / inverse_transform_inplace() apply x' = (x - center) / scale
 * and its inverse. Derived classes only define the scale statistic of a single
 * column (e.g. Bessel-corrected SD, Lp norm) via computeScaleStat().
 *
 * The `center` / `scale` switches follow R's scale() and sklearn's scalers:
 *  - center: subtract the column mean (centers are 0 when disabled).
 *  - scale:  divide by the column's scale statistic, computed around the
 *            APPLIED center. As in R, disabling `center` while keeping `scale`
 *            therefore scales by the statistic around 0 (root-mean-square for
 *            the z-score scaler), not around the mean.
 *
 * Degenerate columns (statistic below the fit threshold) follow sklearn:
 * their scale factor is forced to 1.0 so they are still centered but never
 * divided; their indices are reported via get_dropped_indices().
 *
 * Supports both in-memory and memory-mapped data via Eigen::Map, is designed
 * for high-dimensional data (p >> n), and transforms strictly in place.
 */
class DataTransformer {

protected:

    // ========================================================================
    // State Variables
    // ========================================================================

    /** @brief Per-column centers (zeros if `center` is disabled). */
    Eigen::VectorXd centers_;

    /** @brief Per-column scale factors (ones if `scale` is disabled). */
    Eigen::VectorXd scales_;

    /** @brief Whether fit()/transform() center each column. */
    bool center_{true};

    /** @brief Whether fit()/transform() scale each column. */
    bool scale_{true};

    /** @brief Indices of degenerate columns (scale statistic below threshold;
     *  scale factor forced to 1.0, still centered). */
    std::vector<std::size_t> dropped_indices_;

    /** @brief Whether the transformer has been fitted. */
    bool fitted_{false};

    // ========================================================================
    // Protected constructors to support inheritance and prevent slicing
    // ========================================================================

    /** @brief Construct with the standard center/scale switches. */
    explicit DataTransformer(bool center = true, bool scale = true)
        : center_(center), scale_(scale) {}

    /** @brief Default copy constructor */
    DataTransformer(const DataTransformer&) = default;

    /** @brief Default copy assignment */
    DataTransformer& operator=(const DataTransformer&) = default;

    /** @brief Default move constructor */
    DataTransformer(DataTransformer&&) noexcept = default;

    /** @brief Default move assignment */
    DataTransformer& operator=(DataTransformer&&) noexcept = default;

    // ========================================================================
    // Template-Method Hooks
    // ========================================================================

    /**
     * @brief Compute the scale statistic of one column around the applied center.
     *
     * @param col Column view into the data matrix.
     * @param center The center that transform_inplace() will subtract
     *               (0.0 when centering is disabled).
     *
     * @return The (non-negative) scale statistic.
     */
    virtual double computeScaleStat(const Eigen::Ref<const Eigen::VectorXd>& col,
                                    double center) const = 0;

    /**
     * @brief Validate fit() preconditions beyond non-emptiness.
     *
     * @param nrows Number of samples in the matrix passed to fit().
     * @throws std::invalid_argument if the derived statistic is undefined.
     */
    virtual void checkFitPreconditions(Eigen::Index nrows) const { (void)nrows; }

public:

    /** @brief Virtual destructor */
    virtual ~DataTransformer() = default;

    // ========================================================================
    // Core Interface
    // ========================================================================

    /**
     * @brief Fit the transformer to the input data (read-only).
     *
     * Computes per-column centers and scale statistics without modifying X.
     *
     * @param X Input data matrix (n_samples x n_features).
     * @param threshold Minimum scale statistic; columns below it are treated
     *                  as degenerate (scale factor 1.0, reported as dropped).
     *
     * @return Reference to the transformer for chaining.
     */
    DataTransformer& fit(Eigen::Map<Eigen::MatrixXd>& X, double threshold = 1e-12);

    /**
     * @brief Apply the transformation in place: x' = (x - center) / scale.
     *
     * @param X Input/output matrix to transform.
     */
    void transform_inplace(Eigen::Map<Eigen::MatrixXd>& X) const;

    /**
     * @brief Apply the inverse transformation in place: x = x' * scale + center.
     *
     * @param X_transformed Input/output matrix to inverse transform.
     */
    void inverse_transform_inplace(Eigen::Map<Eigen::MatrixXd>& X_transformed) const;

    /**
     * @brief Fit and transform in one call (R's scale()).
     *
     * @param X Input/output matrix; fitted read-only, then transformed in place.
     * @param threshold Minimum scale statistic (see fit()).
     *
     * @return Reference to the transformer for chaining.
     */
    DataTransformer& fit_transform_inplace(Eigen::Map<Eigen::MatrixXd>& X,
                                           double threshold = 1e-12) {
        fit(X, threshold);
        transform_inplace(X);
        return *this;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /** @brief Get per-column centers (zeros if `center` is disabled). */
    const Eigen::VectorXd& get_centers() const noexcept { return centers_; }

    /** @brief Get per-column scale factors (ones if `scale` is disabled). */
    const Eigen::VectorXd& get_scales() const noexcept { return scales_; }

    /** @brief Whether columns are centered. */
    bool get_center() const noexcept { return center_; }

    /** @brief Whether columns are scaled. */
    bool get_scale() const noexcept { return scale_; }

    /** @brief Get transformer name for serialization */
    virtual std::string get_name() const = 0;

    /** @brief Get indices of degenerate columns (empty if none). */
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
     * @brief Save the transformer state to a binary file.
     *
     * @param filename Path to output file.
     * @throws std::runtime_error if the file cannot be opened.
     */
    void save(const std::string& filename) const {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error(
                get_name() + "::save: Cannot open '" + filename + "' for writing");
        }
        cereal::PortableBinaryOutputArchive archive(ofs);
        std::string type_name = get_name();
        archive(CEREAL_NVP(type_name));
        serialize_to(archive);
    }

    /**
     * @brief Load the transformer state from a binary file.
     *
     * @param filename Path to input file.
     * @throws std::runtime_error if the file cannot be opened, the stored type
     *         does not match, or the loaded state is not fitted.
     */
    void load(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) {
            throw std::runtime_error(
                get_name() + "::load: Cannot open '" + filename + "' for reading");
        }
        cereal::PortableBinaryInputArchive archive(ifs);
        std::string type_name;
        archive(CEREAL_NVP(type_name));

        if (type_name != get_name()) {
            throw std::runtime_error(
                get_name() + "::load: Type mismatch. Expected '" + get_name() +
                "', got '" + type_name + "'");
        }

        serialize_from(archive);
        if (!is_fitted()) {
            throw std::runtime_error(get_name() + "::load: Loaded state is not fitted");
        }
    }

protected:

    // ========================================================================
    // De-/Serialization
    // ========================================================================

    /** @brief Write the full (base + derived) state to an output archive. */
    virtual void serialize_to(cereal::PortableBinaryOutputArchive& archive) const = 0;

    /** @brief Read the full (base + derived) state from an input archive. */
    virtual void serialize_from(cereal::PortableBinaryInputArchive& archive) = 0;

    /**
     * @brief Serialize the base-class state.
     *
     * @note Derived classes call this first inside their template serialize().
     */
    template<class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(centers_),
                CEREAL_NVP(scales_),
                CEREAL_NVP(center_),
                CEREAL_NVP(scale_),
                CEREAL_NVP(dropped_indices_),
                CEREAL_NVP(fitted_)
            );
    }

    // Grant Cereal access
    friend class cereal::access;
};

// ===================================================================================
} /* End of namespace scaler_methods */
} /* End of namespace ml_methods */
} /* End of namespace trex */
// ===================================================================================
#endif /* End of ML_METHODS_SCALER_METHODS_DATA_TRANSFORMER_HPP */
