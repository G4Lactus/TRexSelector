// =====================================================================================
// ml_bindings.hpp - C++ bindings for machine learning utilities used in TRexSelector.
// =====================================================================================
#ifndef TREX_PYTHON_ML_BINDINGS_HPP
#define TREX_PYTHON_ML_BINDINGS_HPP
// =====================================================================================
/**
 * @file ml_bindings.hpp
 *
 * @brief Pybind11 bindings for machine learning utilities used in the T-Rex Selector
 *        package.
 *
 * @details This file defines pybind11 bindings for various machine learning utilities
 *          that are used across different components of the T-Rex Selector, such as:
 *          - Data standardization
 *          - Ridge regression model selection
 *          - Singular value decomposition (SVD)
 *          - Principal component analysis (PCA)
 *          - Ridge regression
 *          The bindings are organized into a function `bind_ml_methods` that can be
 *          called from the main binding module to expose these utilities to Python.
 */
// =====================================================================================

// pybind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

// ml_methods includes
#include <ml_methods/scaler_methods/z_score_scaler.hpp>
#include <ml_methods/scaler_methods/lp_norm_scaler.hpp>
#include <ml_methods/scaler_methods/data_transformer.hpp>
// ml_methods includes (model_selection: ridge_cv/ridge_gcv moved to TRex_Research — bindings TBD)
// #include <ml_methods/model_selection/ridge_cv.hpp>
// #include <ml_methods/model_selection/ridge_gcv.hpp>
#include <ml_methods/svd/svd.hpp>
#include <ml_methods/pca/pca.hpp>
#include <ml_methods/ridge_regression/ridge.hpp>

// =====================================================================================

namespace py = pybind11;

// =====================================================================================

/**
 * @brief Zero-copy Python wrapper preserving the historical PCA API shape.
 *
 * @details The refactored C++ PCA constructs with (X, M, center, normalize) and
 *          preprocesses X in place, with a no-argument fit() and an X-taking
 *          restore(X). This wrapper keeps the Python-facing contract
 *          `PCA(center, normalize)` + `fit(X, M)` + no-arg `restore()` by holding
 *          an Eigen::Map into the (Fortran-contiguous) numpy buffer between calls.
 */
class PyPCA {
public:
    explicit PyPCA(bool center = true, bool normalize = true)
        : center_(center), normalize_(normalize) {}

    trex::ml_methods::pca::PCAResult fit(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Index M) {
        x_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X.data(), X.rows(), X.cols());
        pca_ = std::make_unique<trex::ml_methods::pca::PCA>(*x_map_, M, center_, normalize_);
        return pca_->fit();
    }

    Eigen::MatrixXd transform(const Eigen::Ref<const Eigen::MatrixXd>& X_new) const {
        ensureFitted();
        return pca_->transform(X_new);
    }

    void restore() {
        ensureFitted();
        pca_->restore(*x_map_);
    }

    const Eigen::RowVectorXd& getMeans() const { ensureFitted(); return pca_->getMeans(); }
    const Eigen::RowVectorXd& getNorms() const { ensureFitted(); return pca_->getNorms(); }
    const Eigen::MatrixXd&     getLoadings() const { ensureFitted(); return pca_->getLoadings(); }
    const Eigen::VectorXd&     getExplainedVariance() const { ensureFitted(); return pca_->getExplainedVariance(); }

private:
    void ensureFitted() const {
        if (!pca_) throw std::runtime_error("PCA: call fit() before accessing results.");
    }

    bool center_;
    bool normalize_;
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>>   x_map_;
    std::unique_ptr<trex::ml_methods::pca::PCA>    pca_;
};

// =====================================================================================

/**
 * @brief Bind machine learning methods to a Python module.
 *
 * @param m The Python module to which the methods will be bound.
 */
inline void bind_ml_methods(py::module& m) {

    // =================================================================================
    // Data Standardization bindings
    // =================================================================================
    using namespace trex::ml_methods::scaler_methods;

    // Abstract DataTransformer can't bind easily without trampoline class.

    /**
     * @brief Standardizes features by removing the mean and scaling to unit variance.
     *
     * @details This scaler provides similar functionality to
     *          `sklearn.preprocessing.StandardScaler` with R's scale() argument
     *          names: `center` subtracts the column mean, `scale` divides by the
     *          Bessel-corrected SD (around 0 when `center` is disabled, as in R).
     *          It computes the statistics on a training set so they can later be
     *          applied to a validation or test set via `transform_inplace`.
     *
     * @note The *_inplace methods mutate the passed array directly (zero-copy);
     *       it must be Fortran-ordered float64 (e.g. np.asfortranarray(X)).
     */
    py::class_<ZScoreScaler>(m, "ZScoreScaler")
        .def(py::init<bool, bool>(),
             py::arg("center") = true, py::arg("scale") = true)
        .def("fit",
            [](ZScoreScaler& self, Eigen::Ref<Eigen::MatrixXd> X, double threshold) -> ZScoreScaler& {
            Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
            self.fit(X_map, threshold);
            return self;
        }, py::arg("X"), py::arg("threshold") = 1e-12,
           py::return_value_policy::reference_internal)
        .def("transform_inplace",
            [](ZScoreScaler& self, Eigen::Ref<Eigen::MatrixXd> X) {
            Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
            self.transform_inplace(X_map);
        }, py::arg("X"))
        .def("inverse_transform_inplace",
            [](const ZScoreScaler& self, Eigen::Ref<Eigen::MatrixXd> X) {
            Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
            self.inverse_transform_inplace(X_map);
        }, py::arg("X_scaled"))
        .def("fit_transform_inplace",
            [](ZScoreScaler& self, Eigen::Ref<Eigen::MatrixXd> X, double threshold) -> ZScoreScaler& {
            Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
            self.fit_transform_inplace(X_map, threshold);
            return self;
        }, py::arg("X"), py::arg("threshold") = 1e-12,
           py::return_value_policy::reference_internal)
        .def("is_fitted", &ZScoreScaler::is_fitted)
        .def("get_dropped_indices", &ZScoreScaler::get_dropped_indices)
        .def("get_centers", &ZScoreScaler::get_centers)
        .def("get_scales", &ZScoreScaler::get_scales)
        .def("get_center", &ZScoreScaler::get_center)
        .def("get_scale", &ZScoreScaler::get_scale)
        .def("save", &ZScoreScaler::save, py::arg("filename"))
        .def("load", &ZScoreScaler::load, py::arg("filename"));


    /**
     * @brief Defines the type of mathematical norm used for Lp-normalization.
     *
     * @details Contains L1 (sum of absolute values) and L2 (Euclidean norm) options.
     */
    py::enum_<LpNormScaler::NormType>(m, "NormType")
        .value("L1", LpNormScaler::NormType::L1)
        .value("L2", LpNormScaler::NormType::L2)
        .export_values();


    /**
     * @brief Scales input features to unit norm based on the chosen L1 or L2 formulation.
     *
     * @details Depending on the chosen `NormType`, this scaler normalizes the data such
     *          that the Lp norm of each column is exactly one. The `center`/`scale`
     *          switches follow R's scale(): the norm is computed around the applied
     *          center (around 0 when `center` is disabled).
     *
     * @note The *_inplace methods mutate the passed array directly (zero-copy);
     *       it must be Fortran-ordered float64 (e.g. np.asfortranarray(X)).
     */
    py::class_<LpNormScaler>(m, "LpNormScaler")
        .def(py::init<LpNormScaler::NormType, bool, bool>(),
             py::arg("norm_type") = LpNormScaler::NormType::L2,
             py::arg("center") = true,
             py::arg("scale") = true)
        .def("fit", [](LpNormScaler& self,
            Eigen::Ref<Eigen::MatrixXd> X, double threshold) -> LpNormScaler& {
                Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
                self.fit(X_map, threshold);
                return self;
        }, py::arg("X"), py::arg("threshold") = 1e-12,
           py::return_value_policy::reference_internal)
        .def("transform_inplace",
             [](LpNormScaler& self, Eigen::Ref<Eigen::MatrixXd> X) {
            Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
            self.transform_inplace(X_map);
        }, py::arg("X"))
        .def("inverse_transform_inplace",
             [](const LpNormScaler& self, Eigen::Ref<Eigen::MatrixXd> X) {
            Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
            self.inverse_transform_inplace(X_map);
        }, py::arg("X_normed"))
        .def("fit_transform_inplace",
             [](LpNormScaler& self, Eigen::Ref<Eigen::MatrixXd> X, double threshold) -> LpNormScaler& {
            Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
            self.fit_transform_inplace(X_map, threshold);
            return self;
        }, py::arg("X"), py::arg("threshold") = 1e-12,
           py::return_value_policy::reference_internal)
        .def("is_fitted", &LpNormScaler::is_fitted)
        .def("get_dropped_indices", &LpNormScaler::get_dropped_indices)
        .def("get_centers", &LpNormScaler::get_centers)
        .def("get_scales", &LpNormScaler::get_scales)
        .def("get_center", &LpNormScaler::get_center)
        .def("get_scale", &LpNormScaler::get_scale)
        .def("get_norm_type", &LpNormScaler::get_norm_type)
        .def("save", &LpNormScaler::save, py::arg("filename"))
        .def("load", &LpNormScaler::load, py::arg("filename"));

    // =================================================================================


    // =========================================================================
    // Ridge Regression CV bindings intentionally omitted: ridge_cv / ridge_gcv
    // were moved to TRex_Research. Python bindings for ridge_cv_svd and
    // enet_cv_ccd are out of scope here (see project notes).
    // =========================================================================


    // =================================================================================
    // SVD Solver
    // =================================================================================
    using namespace trex::ml_methods::svd;

    /**
     * @brief Container for a truncated SVD decomposition.
     *
     * @details Returned by SVDSolver::compute(). Holds U (n × M), S (M), V (p × M).
     */
    py::class_<SVDResult>(m, "SVDResult")
        .def(py::init<>())
        .def_readwrite("U", &SVDResult::U)
        .def_readwrite("S", &SVDResult::S)
        .def_readwrite("V", &SVDResult::V);

    /**
     * @brief Computes the top-M truncated SVD of a matrix.
     *
     * @details Dispatches among direct (tall), Gram (wide), and randomized paths
     *          based on matrix shape.
     */
    py::class_<SVDSolver>(m, "SVDSolver")
        .def(py::init<>())
        .def("compute",
            [](SVDSolver& self,
               Eigen::Ref<const Eigen::MatrixXd> X, // NOLINT(performance-unnecessary-value-param)
               Eigen::Index M) {
                return self.compute(X, M);
            },
            py::arg("X"), py::arg("M"));


    // =================================================================================
    // PCA
    // =================================================================================
    using namespace trex::ml_methods::pca;

    /**
     * @brief Container for a PCA fit result.
     *
     * @details Holds Z (n × M scores), V (p × M loadings), and explained_variance (M).
     */
    py::class_<PCAResult>(m, "PCAResult")
        .def(py::init<>())
        .def_readwrite("Z",                  &PCAResult::Z)
        .def_readwrite("V",                  &PCAResult::V)
        .def_readwrite("explained_variance", &PCAResult::explained_variance);

    /**
     * @brief Fits PCA to a data matrix with optional in-place centering (RAII).
     *
     * @details When center=true, X is modified in-place during fit() and restored
     *          when restore() is called or the object is destroyed.
     */
    py::class_<PyPCA>(m, "PCA")
        .def(py::init<bool, bool>(), py::arg("center") = true, py::arg("normalize") = true)
        .def("fit",
            [](PyPCA& self, Eigen::Ref<Eigen::MatrixXd> X, Eigen::Index M) {
                return self.fit(X, M);
            },
            py::arg("X"), py::arg("M"), py::keep_alive<1, 2>())
        .def("transform",
            [](PyPCA& self,
               Eigen::Ref<const Eigen::MatrixXd> X_new // NOLINT(performance-unnecessary-value-param)
            ) {
                return self.transform(X_new);
            },
            py::arg("X_new"))
        .def("restore", &PyPCA::restore)
        .def("get_mean",               &PyPCA::getMeans)
        .def("get_means",              &PyPCA::getMeans)
        .def("get_norms",              &PyPCA::getNorms)
        .def("get_loadings",           &PyPCA::getLoadings)
        .def("get_explained_variance", &PyPCA::getExplainedVariance);


    // =================================================================================
    // Ridge Regression (standalone solver)
    // =================================================================================
    using namespace trex::ml_methods::ridge;

    /**
     * @brief Solves ridge regression for a single lambda value.
     *
     * @details Dispatches between primal (n >= p) and dual (n < p) Cholesky solvers.
     */
    py::class_<RidgeSolver>(m, "RidgeSolver")
        .def(py::init<>())
        .def_static("solve",
            [](Eigen::Ref<const Eigen::MatrixXd> X, // NOLINT(performance-unnecessary-value-param)
               Eigen::Ref<const Eigen::VectorXd> y, // NOLINT(performance-unnecessary-value-param)
               double lambda) {
                return RidgeSolver::solve(X, y, lambda);
            },
            py::arg("X"), py::arg("y"), py::arg("lambda_val"));


// =====================================================================================
}

// =====================================================================================
#endif /* End of TREX_PYTHON_ML_BINDINGS_HPP */
