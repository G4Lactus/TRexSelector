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
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

// ml_methods includes
#include <ml_methods/scaler_methods/z_score_scaler.hpp>
#include <ml_methods/scaler_methods/lp_norm_scaler.hpp>
#include <ml_methods/scaler_methods/data_transformer.hpp>
// ml_methods includes (model_selection: the former ridge_cv/ridge_gcv were
// moved to TRex_Research; the SVD-path ridge CV and the CCD elastic-net remain)
#include <ml_methods/model_selection/ridge_cv_svd.hpp>
#include <ml_methods/model_selection/enet_cv_ccd.hpp>
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
    explicit PyPCA(bool center = true, bool normalize = true,
                   std::optional<int> n_threads = std::nullopt)
        : center_(center), normalize_(normalize),
          n_threads_(n_threads.value_or(trex::utils::openmp::get_num_threads())) {
        if (n_threads_ < 1) throw std::invalid_argument("PCA: n_threads must be >= 1.");
    }

    trex::ml_methods::pca::PCAResult fit(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Index M) {
        x_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X.data(), X.rows(), X.cols());
        pca_ = std::make_unique<trex::ml_methods::pca::PCA>(*x_map_, M, center_, normalize_,
                                                            n_threads_);
        return pca_->fit();
    }

    // The budget of the next fit(); a fitted PCA keeps the budget it was fitted with
    void setNumThreads(int n_threads) {
        if (n_threads < 1) throw std::invalid_argument("PCA: n_threads must be >= 1.");
        n_threads_ = n_threads;
    }
    int getNumThreads() const { return n_threads_; }

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
    int n_threads_;
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>>   x_map_;
    std::unique_ptr<trex::ml_methods::pca::PCA>    pca_;
};

// =====================================================================================

/**
 * @brief Returning, layout-agnostic counterparts to the scalers' zero-copy
 *        `*_inplace` methods.
 *
 * @details A `const Eigen::Ref<const Eigen::MatrixXd>&` parameter accepts any
 *          2-D float64 numpy array regardless of memory order — pybind copies
 *          when the layout/dtype does not match — so the caller's array is
 *          never mutated. Returning an `Eigen::MatrixXd` by value yields a
 *          fresh (column-major) numpy array. This restores the natural
 *          "transform and assign" idiom (`Xt = scaler.transform(X)`) that the
 *          in-place API cannot offer, and sidesteps the requirement that the
 *          input be a writeable Fortran-ordered buffer.
 */
template <typename Scaler>
Eigen::MatrixXd scaler_transform_copy(Scaler& self,
                                      const Eigen::Ref<const Eigen::MatrixXd>& X) {
    Eigen::MatrixXd out = X;
    Eigen::Map<Eigen::MatrixXd> m(out.data(), out.rows(), out.cols());
    self.transform_inplace(m);
    return out;
}

template <typename Scaler>
Eigen::MatrixXd scaler_fit_transform_copy(Scaler& self,
                                          const Eigen::Ref<const Eigen::MatrixXd>& X,
                                          double threshold) {
    Eigen::MatrixXd out = X;
    Eigen::Map<Eigen::MatrixXd> m(out.data(), out.rows(), out.cols());
    self.fit_transform_inplace(m, threshold);
    return out;
}

template <typename Scaler>
Eigen::MatrixXd scaler_inverse_transform_copy(const Scaler& self,
                                              const Eigen::Ref<const Eigen::MatrixXd>& X) {
    Eigen::MatrixXd out = X;
    Eigen::Map<Eigen::MatrixXd> m(out.data(), out.rows(), out.cols());
    self.inverse_transform_inplace(m);
    return out;
}

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
     * @note Two families of methods are provided. The `transform` /
     *       `fit_transform` / `inverse_transform` methods accept an array of
     *       any memory order and RETURN a new transformed array, leaving the
     *       input untouched (the ergonomic default). The `*_inplace` methods
     *       mutate the passed array directly (zero-copy) and therefore require
     *       a writeable Fortran-ordered float64 array (e.g. np.asfortranarray(X)).
     */
    py::class_<ZScoreScaler>(m, "ZScoreScaler")
        .def(py::init([](bool center, bool scale, std::optional<int> n_threads) {
                 return ZScoreScaler(center, scale,
                                     n_threads.value_or(trex::utils::openmp::get_num_threads()));
             }),
             py::arg("center") = true, py::arg("scale") = true,
             py::arg("n_threads") = py::none(),
             "n_threads: the thread budget of the column passes (None: the library's "
             "default budget, utils.get_num_threads(), initially 1).")
        .def("set_num_threads", &ZScoreScaler::set_num_threads, py::arg("n_threads"),
             "Set the thread budget of the column passes (>= 1).")
        .def("get_num_threads", &ZScoreScaler::get_num_threads,
             "The thread budget of the column passes.")
        .def("fit",
            [](ZScoreScaler& self, Eigen::Ref<const Eigen::MatrixXd> X, double threshold) -> ZScoreScaler& {
                // read-only: any float64 array, read-only or C-ordered (converted once)
                self.fit(X, threshold);
            return self;
        }, py::arg("X"), py::arg("threshold") = 1e-12,
           py::return_value_policy::reference_internal)
        .def("transform", &scaler_transform_copy<ZScoreScaler>, py::arg("X"),
             "Return a scaled copy of X (any memory order accepted; X unchanged).")
        .def("fit_transform", &scaler_fit_transform_copy<ZScoreScaler>,
             py::arg("X"), py::arg("threshold") = 1e-12,
             "Fit on X, then return a scaled copy (any memory order; X unchanged).")
        .def("inverse_transform", &scaler_inverse_transform_copy<ZScoreScaler>,
             py::arg("X_scaled"),
             "Return an unscaled copy of X_scaled (any memory order; input unchanged).")
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
     * @note Two families of methods are provided. The `transform` /
     *       `fit_transform` / `inverse_transform` methods accept an array of
     *       any memory order and RETURN a new transformed array, leaving the
     *       input untouched (the ergonomic default). The `*_inplace` methods
     *       mutate the passed array directly (zero-copy) and therefore require
     *       a writeable Fortran-ordered float64 array (e.g. np.asfortranarray(X)).
     */
    py::class_<LpNormScaler>(m, "LpNormScaler")
        .def(py::init([](LpNormScaler::NormType norm_type, bool center, bool scale,
                         std::optional<int> n_threads) {
                 return LpNormScaler(norm_type, center, scale,
                                     n_threads.value_or(trex::utils::openmp::get_num_threads()));
             }),
             py::arg("norm_type") = LpNormScaler::NormType::L2,
             py::arg("center") = true,
             py::arg("scale") = true,
             py::arg("n_threads") = py::none(),
             "n_threads: the thread budget of the column passes (None: the library's "
             "default budget, utils.get_num_threads(), initially 1).")
        .def("set_num_threads", &LpNormScaler::set_num_threads, py::arg("n_threads"),
             "Set the thread budget of the column passes (>= 1).")
        .def("get_num_threads", &LpNormScaler::get_num_threads,
             "The thread budget of the column passes.")
        .def("fit", [](LpNormScaler& self,
            Eigen::Ref<const Eigen::MatrixXd> X, double threshold) -> LpNormScaler& {
                // read-only: any float64 array, read-only or C-ordered (converted once)
                self.fit(X, threshold);
                return self;
        }, py::arg("X"), py::arg("threshold") = 1e-12,
           py::return_value_policy::reference_internal)
        .def("transform", &scaler_transform_copy<LpNormScaler>, py::arg("X"),
             "Return a normalized copy of X (any memory order accepted; X unchanged).")
        .def("fit_transform", &scaler_fit_transform_copy<LpNormScaler>,
             py::arg("X"), py::arg("threshold") = 1e-12,
             "Fit on X, then return a normalized copy (any memory order; X unchanged).")
        .def("inverse_transform", &scaler_inverse_transform_copy<LpNormScaler>,
             py::arg("X_normed"),
             "Return a denormalized copy of X_normed (any memory order; input unchanged).")
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
    // Model selection: Ridge CV (SVD path) and Elastic-Net (CCD path + CV)
    // =========================================================================
    using trex::ml_methods::model_selection::ridge_cv_svd;
    using trex::ml_methods::model_selection::enet_gaussian;
    using trex::ml_methods::model_selection::enet_cv_ccd;

    /**
     * @brief K-fold cross-validated ridge regression (SVD path per fold).
     *
     * @details Selects the ridge penalty by K-fold CV over a log-spaced grid,
     *          returning glmnet-style lambda.min / lambda.1se. Mirrors the R
     *          RidgeCV R6 class. Indices are 0-based (add 1 for R parity).
     */
    py::class_<ridge_cv_svd>(m, "RidgeCV",
        "K-fold cross-validated ridge regression (SVD path per fold).\n"
        "\n"
        "References\n"
        "----------\n"
        "A. E. Hoerl and R. W. Kennard, \"Ridge Regression: Biased Estimation for Nonorthogonal\n"
        "  Problems,\" Technometrics, vol. 42, no. 1, pp. 80-86, 2000 (Reprint of the 1970 article\n"
        "  in the Special 40th Anniversary Issue).\n"
        "G. Guennebaud, B. Jacob et al., Eigen v3, https://libeigen.gitlab.io/, 2010.\n")
        .def(py::init<>())
        .def(py::init([](bool low_memory, std::optional<int> n_threads) {
                 ridge_cv_svd cv(low_memory);
                 cv.set_num_threads(n_threads.value_or(trex::utils::openmp::get_num_threads()));
                 return cv;
             }),
             py::arg("low_memory"), py::arg("n_threads") = py::none(),
             "low_memory=True: the dual (kernel) fold engine — n x n kernels accumulated "
             "from column blocks of X read in place, no p-dimensional coefficients; same "
             "estimator, grid and rank rule, agreement to numerical precision; memory "
             "O(n^2) instead of O(n p), the route for p >> n. n_threads: the thread budget "
             "of the fit's matrix products (None: the library's default budget, "
             "utils.get_num_threads(), initially 1).")
        .def("set_num_threads", &ridge_cv_svd::set_num_threads, py::arg("n_threads"),
             "Set the thread budget of the fit's matrix products (>= 1).")
        .def("num_threads", &ridge_cv_svd::num_threads,
             "The thread budget of the fit's matrix products.")
        .def("set_low_memory", &ridge_cv_svd::set_low_memory, py::arg("low_memory"),
             "Switch the fold engine; takes effect at the next fit().")
        .def("low_memory", &ridge_cv_svd::low_memory,
             "Whether the kernel fold engine is selected.")
        .def("interpolates", &ridge_cv_svd::interpolates,
             "Whether the smallest training fold has p >= n_train, so the lambda -> 0 end of the grid interpolates.")
        .def("selection_at_grid_edge", &ridge_cv_svd::selection_at_grid_edge,
             "Whether lambda.min or lambda.1se sits on the first or last grid point.")
        .def("fit",
            [](ridge_cv_svd& self, Eigen::Ref<const Eigen::MatrixXd> X,
               Eigen::Ref<const Eigen::VectorXd> y, int n_folds,
               Eigen::Index n_lambda, double lambda_ratio, unsigned int seed)
               -> ridge_cv_svd& {
                self.fit(X, y, n_folds, n_lambda, lambda_ratio, seed);
                return self;
            },
            py::arg("X"), py::arg("y"), py::arg("n_folds") = 10,
            py::arg("n_lambda") = 100, py::arg("lambda_ratio") = 1000.0,
            py::arg("seed") = 0, py::return_value_policy::reference_internal,
            "Fit K-fold CV over a log-spaced ridge grid.")
        .def("cv_min", &ridge_cv_svd::cv_min, "lambda minimizing mean CV MSE.")
        .def("cv_1se", &ridge_cv_svd::cv_1se, "Largest lambda within 1 SE of the min.")
        .def("index_min", &ridge_cv_svd::index_min, "0-based index of lambda.min.")
        .def("index_1se", &ridge_cv_svd::index_1se, "0-based index of lambda.1se.")
        .def("lambdas", &ridge_cv_svd::lambdas, "Lambda grid used (descending).")
        .def("cv_mse", &ridge_cv_svd::cv_mse, "Mean CV MSE per lambda.")
        .def("cv_sem", &ridge_cv_svd::cv_sem, "Standard error of the CV MSE per lambda.");

    /**
     * @brief Elastic-net coordinate-descent path (glmnet-equivalent).
     *
     * @details Fits the EN coefficient path over an auto-generated glmnet-style
     *          lambda grid, or over an explicit grid via fit_grid(). Reports
     *          coefficients in the ORIGINAL predictor scale, intercepts, the
     *          deviance-ratio path, and predictions for new data.
     */
    py::class_<enet_gaussian>(m, "ElasticNet",
        "Elastic-net coordinate-descent path (glmnet-equivalent).\n"
        "\n"
        "References\n"
        "----------\n"
        "H. Zou and T. Hastie, \"Regularization and Variable Selection via the Elastic Net,\" J. R.\n"
        "  Stat. Soc. Ser. B Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.\n"
        "J. Friedman, T. Hastie, and R. Tibshirani, \"Regularization Paths for Generalized Linear\n"
        "  Models via Coordinate Descent,\" J. Stat. Softw., vol. 33, no. 1, pp. 1-22, 2010,\n"
        "  doi: 10.18637/jss.v033.i01.\n")
        .def(py::init<>())
        .def(py::init([](bool low_memory, std::optional<int> n_threads) {
                 enet_gaussian path(low_memory);
                 path.set_num_threads(n_threads.value_or(trex::utils::openmp::get_num_threads()));
                 return path;
             }),
             py::arg("low_memory"), py::arg("n_threads") = py::none(),
             "low_memory=False (default): the coordinate descent runs on a centered and "
             "standardized working copy of X (one n x p matrix while fitting). "
             "low_memory=True: it runs on X itself through the caller's view (a "
             "MemoryMappedMatrix.to_numpy() view included) with no allocation of the size "
             "of X; same estimator and grid, the two engines agree to rounding (about "
             "1e-12 in the coefficients). n_threads: the thread budget of the fit's matrix "
             "products (None: the library's default budget, utils.get_num_threads(), "
             "initially 1).")
        .def("set_num_threads", &enet_gaussian::set_num_threads, py::arg("n_threads"),
             "Set the thread budget of the fit's matrix products (>= 1).")
        .def("num_threads", &enet_gaussian::num_threads,
             "The thread budget of the fit's matrix products.")
        .def("set_low_memory", &enet_gaussian::set_low_memory, py::arg("low_memory"),
             "Switch the engine; takes effect at the next fit().")
        .def("low_memory", &enet_gaussian::low_memory,
             "Whether the raw-X (no working copy) engine is selected.")
        .def("fit",
            [](enet_gaussian& self, Eigen::Ref<const Eigen::MatrixXd> X,
               Eigen::Ref<const Eigen::VectorXd> y, double alpha,
               Eigen::Index n_lambda, double lambda_min_ratio, bool standardize,
               bool intercept, bool use_strong_rule, int max_iter, double tol)
               -> enet_gaussian& {
                self.fit(X, y, alpha, n_lambda, lambda_min_ratio, standardize,
                         intercept, use_strong_rule, max_iter, tol);
                return self;
            },
            py::arg("X"), py::arg("y"), py::arg("alpha") = 1.0,
            py::arg("n_lambda") = 100, py::arg("lambda_min_ratio") = -1.0,
            py::arg("standardize") = true, py::arg("intercept") = true,
            py::arg("use_strong_rule") = false, py::arg("max_iter") = 100000,
            py::arg("tol") = 1e-7, py::return_value_policy::reference_internal,
            "Fit the EN path over an auto-generated glmnet-style lambda grid "
            "(alpha: 1 = lasso, 0 = ridge).")
        .def("fit_grid",
            [](enet_gaussian& self, Eigen::Ref<const Eigen::MatrixXd> X,
               Eigen::Ref<const Eigen::VectorXd> y,
               Eigen::Ref<const Eigen::VectorXd> lambda_grid, double alpha,
               bool standardize, bool intercept, bool use_strong_rule,
               int max_iter, double tol) -> enet_gaussian& {
                self.fit(X, y, lambda_grid, alpha, standardize, intercept,
                         use_strong_rule, max_iter, tol);
                return self;
            },
            py::arg("X"), py::arg("y"), py::arg("lambda_grid"),
            py::arg("alpha") = 1.0, py::arg("standardize") = true,
            py::arg("intercept") = true, py::arg("use_strong_rule") = false,
            py::arg("max_iter") = 100000, py::arg("tol") = 1e-7,
            py::return_value_policy::reference_internal,
            "Fit the EN path at an explicit lambda grid (sorted descending "
            "internally); use to evaluate at exactly glmnet's lambda sequence.")
        .def("coef", &enet_gaussian::coef,
             "Coefficient path in original predictor scale (p x n_lambda).")
        .def("intercepts", &enet_gaussian::intercepts,
             "Intercept per lambda (zero if intercept=False).")
        .def("lambdas", &enet_gaussian::lambdas, "Lambda grid used (descending).")
        .def("dev_ratio", &enet_gaussian::dev_ratio,
             "Fraction of null deviance explained per fitted lambda (glmnet %Dev).")
        .def("converged", &enet_gaussian::converged,
             "True if every lambda reached the convergence tolerance.")
        .def("n_nonconverged", &enet_gaussian::n_nonconverged,
             "Number of lambda points that hit max_iter without converging.")
        .def("predict",
            [](const enet_gaussian& self, Eigen::Ref<const Eigen::MatrixXd> X_new) {
                return self.predict(X_new);
            },
            py::arg("X_new"), "Predictions per lambda: (X_new @ coef) + intercept.");

    /**
     * @brief K-fold cross-validated elastic net (coordinate-descent per fold).
     *
     * @details Selects the elastic-net penalty by K-fold CV over a shared
     *          glmnet-style grid, returning lambda.min / lambda.1se. The mixing
     *          alpha defaults to 0 (ridge), matching the C++ default. Indices
     *          are 0-based (add 1 for R parity).
     */
    py::class_<enet_cv_ccd>(m, "ElasticNetCV",
        "K-fold cross-validated elastic net (coordinate-descent per fold).\n"
        "\n"
        "References\n"
        "----------\n"
        "H. Zou and T. Hastie, \"Regularization and Variable Selection via the Elastic Net,\" J. R.\n"
        "  Stat. Soc. Ser. B Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.\n"
        "J. Friedman, T. Hastie, and R. Tibshirani, \"Regularization Paths for Generalized Linear\n"
        "  Models via Coordinate Descent,\" J. Stat. Softw., vol. 33, no. 1, pp. 1-22, 2010,\n"
        "  doi: 10.18637/jss.v033.i01.\n"
        "R. Tibshirani, J. Bien, J. Friedman, T. Hastie, N. Simon, J. Taylor, and R. J.\n"
        "  Tibshirani, \"Strong Rules for Discarding Predictors in Lasso-Type Problems,\" J. R.\n"
        "  Stat. Soc. Ser. B Stat. Methodol., vol. 74, no. 2, pp. 245-266, 2012,\n"
        "  doi: 10.1111/j.1467-9868.2011.01004.x.\n"
        "L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone, Classification and Regression\n"
        "  Trees. Wadsworth, Belmont, CA, 1984.\n")
        .def(py::init<>())
        .def(py::init([](bool low_memory, std::optional<int> n_threads) {
                 enet_cv_ccd cv(low_memory);
                 cv.set_num_threads(n_threads.value_or(trex::utils::openmp::get_num_threads()));
                 return cv;
             }),
             py::arg("low_memory"), py::arg("n_threads") = py::none(),
             "low_memory=True runs the full-data fit and every fold on X itself with a 0/1 "
             "training mask (no allocation of the size of X); see ElasticNet. n_threads: "
             "the thread budget of the matrix products of the fit and its folds (None: the "
             "library's default budget, utils.get_num_threads(), initially 1).")
        .def("set_num_threads", &enet_cv_ccd::set_num_threads, py::arg("n_threads"),
             "Set the thread budget of the matrix products of the fit and its folds (>= 1).")
        .def("num_threads", &enet_cv_ccd::num_threads,
             "The thread budget of the matrix products of the fit and its folds.")
        .def("set_low_memory", &enet_cv_ccd::set_low_memory, py::arg("low_memory"),
             "Switch the engine; takes effect at the next fit().")
        .def("low_memory", &enet_cv_ccd::low_memory,
             "Whether the raw-X engine is selected.")
        .def("selection_at_grid_edge", &enet_cv_ccd::selection_at_grid_edge,
             "Whether lambda.min or lambda.1se sits on the first or last grid point.")
        .def("grid_truncated", &enet_cv_ccd::grid_truncated,
             "Whether the deviance rules stopped the path before the requested n_lambda.")
        .def("fit",
            [](enet_cv_ccd& self, Eigen::Ref<const Eigen::MatrixXd> X,
               Eigen::Ref<const Eigen::VectorXd> y, double alpha, int n_folds,
               Eigen::Index n_lambda, double lambda_min_ratio, unsigned int seed,
               bool standardize, bool intercept, int max_iter, double tol)
               -> enet_cv_ccd& {
                self.fit(X, y, alpha, n_folds, n_lambda, lambda_min_ratio, seed,
                         standardize, intercept, max_iter, tol);
                return self;
            },
            py::arg("X"), py::arg("y"), py::arg("alpha") = 0.0,
            py::arg("n_folds") = 10, py::arg("n_lambda") = 100,
            py::arg("lambda_min_ratio") = -1.0, py::arg("seed") = 0,
            py::arg("standardize") = true, py::arg("intercept") = true,
            py::arg("max_iter") = 100000, py::arg("tol") = 1e-7,
            py::return_value_policy::reference_internal,
            "Fit K-fold CV over a glmnet-style EN grid (alpha: 1 = lasso, "
            "0 = ridge).")
        .def("cv_min", &enet_cv_ccd::cv_min, "lambda minimizing mean CV MSE.")
        .def("cv_1se", &enet_cv_ccd::cv_1se, "Largest lambda within 1 SE of the min.")
        .def("index_min", &enet_cv_ccd::index_min, "0-based index of lambda.min.")
        .def("index_1se", &enet_cv_ccd::index_1se, "0-based index of lambda.1se.")
        .def("lambdas", &enet_cv_ccd::lambdas, "Lambda grid used (descending).")
        .def("cv_mse", &enet_cv_ccd::cv_mse, "Mean CV MSE per lambda.")
        .def("cv_sem", &enet_cv_ccd::cv_sem, "Standard error of the CV MSE per lambda.");


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
    py::class_<SVDSolver>(m, "SVDSolver",
        "Computes the top-M truncated SVD of a matrix.\n"
        "\n"
        "References\n"
        "----------\n"
        "G. Guennebaud, B. Jacob et al., Eigen v3, https://libeigen.gitlab.io/, 2010.\n")
        .def(py::init<>())
        .def(py::init([](Eigen::Index min_dim_thresh, Eigen::Index block_size,
                         std::optional<int> n_threads) {
                 return SVDSolver(min_dim_thresh, block_size,
                                  n_threads.value_or(trex::utils::openmp::get_num_threads()));
             }),
             py::arg("min_dim_thresh") = 10000, py::arg("block_size") = 1024,
             py::arg("n_threads") = py::none(),
             "n_threads: the thread budget of the matrix products (None: the library's "
             "default budget, utils.get_num_threads(), initially 1). "
             "Dispatch thresholds: the direct Jacobi path copies X "
             "and computes every singular triplet; when min(n, p) exceeds min_dim_thresh "
             "and M < min(n, p) / 10 the randomized path runs two block-wise passes over X "
             "(no copy; the route for a memory-mapped X), and p > 2n always takes the Gram "
             "path. Lowering min_dim_thresh selects the randomized path for smaller "
             "designs; block_size is the column block of the block-wise passes.")
        .def("set_num_threads", &SVDSolver::set_num_threads, py::arg("n_threads"),
             "Set the thread budget of the matrix products (>= 1).")
        .def("num_threads", &SVDSolver::num_threads,
             "The thread budget of the matrix products.")
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
    py::class_<PyPCA>(m, "PCA",
        "Fits PCA to a data matrix with optional in-place centering (RAII).\n"
        "\n"
        "References\n"
        "----------\n"
        "G. Guennebaud, B. Jacob et al., Eigen v3, https://libeigen.gitlab.io/, 2010.\n")
        .def(py::init<bool, bool, std::optional<int>>(),
             py::arg("center") = true, py::arg("normalize") = true,
             py::arg("n_threads") = py::none(),
             "n_threads: the thread budget of the matrix products of fit() and transform() "
             "(None: the library's default budget, utils.get_num_threads(), initially 1).")
        .def("set_num_threads", &PyPCA::setNumThreads, py::arg("n_threads"),
             "Set the thread budget of the next fit() (>= 1).")
        .def("get_num_threads", &PyPCA::getNumThreads,
             "The thread budget of the matrix products.")
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
    py::class_<RidgeSolver>(m, "RidgeSolver",
        "Solves ridge regression for a single lambda value.\n"
        "\n"
        "References\n"
        "----------\n"
        "A. E. Hoerl and R. W. Kennard, \"Ridge Regression: Biased Estimation for Nonorthogonal\n"
        "  Problems,\" Technometrics, vol. 42, no. 1, pp. 80-86, 2000 (Reprint of the 1970 article\n"
        "  in the Special 40th Anniversary Issue).\n")
        .def(py::init<>())
        .def_static("solve",
            [](Eigen::Ref<const Eigen::MatrixXd> X, // NOLINT(performance-unnecessary-value-param)
               Eigen::Ref<const Eigen::VectorXd> y, // NOLINT(performance-unnecessary-value-param)
               double lambda, std::optional<int> n_threads) {
                return RidgeSolver::solve(X, y, lambda,
                                          n_threads.value_or(trex::utils::openmp::get_num_threads()));
            },
            py::arg("X"), py::arg("y"), py::arg("lambda_val"), py::arg("n_threads") = py::none(),
            "n_threads: the thread budget of the matrix products (None: the library's "
            "default budget, utils.get_num_threads(), initially 1).");


// =====================================================================================
}

// =====================================================================================
#endif /* End of TREX_PYTHON_ML_BINDINGS_HPP */
