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
     *          `sklearn.preprocessing.StandardScaler`.
     *          It computes the mean and standard deviation on a training set so they can be
     *          later applied to a validation or test set via `transform_inplace`.
     */
    py::class_<ZScoreScaler>(m, "ZScoreScaler")
        .def(py::init<bool, bool>(),
             py::arg("with_mean") = true, py::arg("with_std") = true)
        .def("fit",
            [](ZScoreScaler& self, Eigen::MatrixXd& X, double threshold) {
            Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
            self.fit(X_map, threshold);
            return &self;
        }, py::arg("X"), py::arg("threshold") = 1e-12)
        .def("transform_inplace",
            [](ZScoreScaler& self, Eigen::MatrixXd& X) {
            Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
            self.transform_inplace(X_map);
            return X; // Return modified matrix for convenience
        }, py::arg("X"))
        .def("inverse_transform_inplace",
            [](const ZScoreScaler& self, Eigen::MatrixXd& X) {
            Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
            self.inverse_transform_inplace(X_map);
            return X; // Return modified matrix for convenience
        }, py::arg("X_scaled"))
        .def("is_fitted", &ZScoreScaler::is_fitted)
        .def("get_dropped_indices", &ZScoreScaler::get_dropped_indices)
        .def("get_means", &ZScoreScaler::get_means)
        .def("get_scales", &ZScoreScaler::get_scales)
        .def("get_with_mean", &ZScoreScaler::get_with_mean)
        .def("get_with_std", &ZScoreScaler::get_with_std)
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
     * @details Depending on the chosen `NormType`, this scaler normalizes the data such that
     *          the norm of each feature vector is exactly one. Mean centering is optionally
     *          applied prior to scaling.
     */
    py::class_<LpNormScaler>(m, "LpNormScaler")
        .def(py::init<LpNormScaler::NormType, bool>(),
             py::arg("norm_type") = LpNormScaler::NormType::L2,
             py::arg("with_mean") = true)
        .def("fit", [](LpNormScaler& self,
            Eigen::MatrixXd& X, double threshold) {
                Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
                self.fit(X_map, threshold);
                return &self;
        }, py::arg("X"), py::arg("threshold") = 1e-12)
        .def("transform_inplace",
             [](LpNormScaler& self, Eigen::MatrixXd& X) {
            Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
            self.transform_inplace(X_map);
            return X; // Return modified matrix for convenience
        }, py::arg("X"))
        .def("inverse_transform_inplace",
             [](const LpNormScaler& self, Eigen::MatrixXd& X) {
            Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
            self.inverse_transform_inplace(X_map);
            return X; // Return modified matrix for convenience
        }, py::arg("X_normed"))
        .def("is_fitted", &LpNormScaler::is_fitted)
        .def("get_dropped_indices", &LpNormScaler::get_dropped_indices)
        .def("get_means", &LpNormScaler::get_means)
        .def("get_scales", &LpNormScaler::get_scales)
        .def("get_with_mean", &LpNormScaler::get_with_mean)
        .def("get_with_norm", &LpNormScaler::get_with_norm)
        .def("get_norm_type", &LpNormScaler::get_norm_type)
        .def("save", &LpNormScaler::save, py::arg("filename"))
        .def("load", &LpNormScaler::load, py::arg("filename"));

    // =================================================================================


    // =========================================================================
    // Ridge Regression CV bindings (ridge_cv / ridge_gcv moved to TRex_Research
    // — Python bindings for ridge_cv_svd and enet_cv_ccd are TBD)
    // =========================================================================

    /*
    using namespace trex::ml_methods::model_selection;

    /**
     * @brief Stores the comprehensive results of a ridge regression parameter sweep.
     *
     * @details A strict data structure holding evaluated lambdas, their corresponding
     *          regression coefficients, GCV scores, and the effective degrees of freedom.
     *          `best_index` tracks the index of the theoretically optimal lambda.
     */
    // py::class_<ridge_path>(m, "RidgePath")
    //     .def(py::init<>())
    //     .def_readwrite("lambdas", &ridge_path::lambdas)
    //     .def_readwrite("coefficients", &ridge_path::coefficients)
    //     .def_readwrite("gcv_scores", &ridge_path::gcv_scores)
    //     .def_readwrite("df_effective", &ridge_path::df_effective)
    //     .def_readwrite("best_index", &ridge_path::best_index);

    // py::class_<ridge_gcv>(m, "RidgeGCV")
    //     ... (full binding TBD after TRex_Research merge)

    // py::class_<ridge_cv>(m, "RidgeCV")
    //     ... (full binding TBD after TRex_Research merge)
    */


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
    py::class_<PCA>(m, "PCA")
        .def(py::init<bool>(), py::arg("center") = true)
        .def("fit",
            [](PCA& self, Eigen::Ref<Eigen::MatrixXd> X, Eigen::Index M) {
                Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
                return self.fit(X_map, M);
            },
            py::arg("X"), py::arg("M"))
        .def("transform",
            [](PCA& self,
               Eigen::Ref<const Eigen::MatrixXd> X_new // NOLINT(performance-unnecessary-value-param)
            ) {
                return self.transform(X_new);
            },
            py::arg("X_new"))
        .def("restore", &PCA::restore)
        .def("get_mean",               &PCA::getMean)
        .def("get_loadings",           &PCA::getLoadings)
        .def("get_explained_variance", &PCA::getExplainedVariance);


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
