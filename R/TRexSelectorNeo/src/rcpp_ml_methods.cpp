#include <RcppEigen.h>
#include <ml_methods/scaler_methods/z_score_scaler.hpp>
#include <ml_methods/scaler_methods/lp_norm_scaler.hpp>
#include <ml_methods/model_selection/ridge_cv_svd.hpp>
#include <ml_methods/svd/svd.hpp>
#include <ml_methods/pca/pca.hpp>
#include <ml_methods/ridge_regression/ridge.hpp>

// [[Rcpp::plugins(cpp20)]]
// [[Rcpp::depends(RcppEigen)]]

using namespace trex::ml_methods::scaler_methods;
using namespace trex::ml_methods::model_selection;
using namespace trex::ml_methods::svd;
using namespace trex::ml_methods::pca;
using namespace trex::ml_methods::ridge;
using namespace Rcpp;

// The exported ridge_cv_* endpoints keep their names; the backing class is the
// SVD-path implementation (the former ridge_cv and the GCV variant were removed).
using ridge_cv = trex::ml_methods::model_selection::ridge_cv_svd;

// =================================================================================

/*
 * Note on Roxygen Documentation:
 * All functions exported via Rcpp in this file are documented using standard Roxygen tags
 * but strictly include the `@noRd` tag. These are internal C++ bindings that are wrapped
 * by user-friendly R6 classes and R functions in the package namespace. Generating `.Rd`
 * manuals for these endpoints would clutter the public API.
 */

// Data Standardization: ZScoreScaler
// =================================================================================

//' @title Create ZScoreScaler
//'
//' @param with_mean Center data
//' @param with_std Scale data
//' @return XPtr to ZScoreScaler
//' @noRd
// [[Rcpp::export]]
XPtr<ZScoreScaler> zscore_scaler_create(bool with_mean, bool with_std) {
    return XPtr<ZScoreScaler>(new ZScoreScaler(with_mean, with_std));
}

//' @title Fit ZScoreScaler
//'
//' @param ptr XPtr to ZScoreScaler
//' @param X Matrix to fit
//' @param threshold Numerical stability threshold
//' @noRd
// [[Rcpp::export]]
void zscore_scaler_fit(
    XPtr<ZScoreScaler> ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> X,
    double threshold = 1e-12
) {
    ptr->fit(X, threshold);
}

//' @title Transform Inplace ZScoreScaler
//'
//' @param ptr XPtr to ZScoreScaler
//' @param X Matrix to transform
//' @noRd
// [[Rcpp::export]]
void zscore_scaler_transform_inplace(
    XPtr<ZScoreScaler> ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> X
) {
    ptr->transform_inplace(X);
}

//' @title Inverse Transform Inplace ZScoreScaler
//'
//' @param ptr XPtr to ZScoreScaler
//' @param X Matrix to transform
//' @noRd
// [[Rcpp::export]]
void zscore_scaler_inverse_transform_inplace(
    XPtr<ZScoreScaler> ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> X
) {
    ptr->inverse_transform_inplace(X);
}

//' @title Check if ZScoreScaler is fitted
//'
//' @param ptr XPtr to ZScoreScaler
//' @return Boolean indicating fit status
//' @noRd
// [[Rcpp::export]]
bool zscore_scaler_is_fitted(
    XPtr<ZScoreScaler> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->is_fitted();
}

//' @title Get dropped indices from ZScoreScaler
//'
//' @param ptr XPtr to ZScoreScaler
//' @return Vector of dropped indices
//' @noRd
// [[Rcpp::export]]
std::vector<std::size_t> zscore_scaler_get_dropped_indices(
    XPtr<ZScoreScaler> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->get_dropped_indices();
}

//' @title Get with_mean flag from ZScoreScaler
//'
//' @param ptr XPtr to ZScoreScaler
//' @return Boolean with_mean
//' @noRd
// [[Rcpp::export]]
bool zscore_scaler_get_with_mean(
    XPtr<ZScoreScaler> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->get_with_mean();
}

//' @title Get with_std flag from ZScoreScaler
//'
//' @param ptr XPtr to ZScoreScaler
//' @return Boolean with_std
//' @noRd
// [[Rcpp::export]]
bool zscore_scaler_get_with_std(
    XPtr<ZScoreScaler> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->get_with_std();
}

//' @title Get means from ZScoreScaler
//'
//' @param ptr XPtr to ZScoreScaler
//' @return Vector of means
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd zscore_scaler_get_means(
    XPtr<ZScoreScaler> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->get_means();
}

//' @title Get scales from ZScoreScaler
//'
//' @param ptr XPtr to ZScoreScaler
//' @return Vector of scales
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd zscore_scaler_get_scales(
    XPtr<ZScoreScaler> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->get_scales();
}

//' @title Save ZScoreScaler
//'
//' @param ptr XPtr to ZScoreScaler
//' @param filename Output filename
//' @noRd
// [[Rcpp::export]]
void zscore_scaler_save(
    XPtr<ZScoreScaler> ptr, // NOLINT(performance-unnecessary-value-param)
    const std::string& filename
) {
    ptr->save(filename);
}

//' @title Load ZScoreScaler
//'
//' @param ptr XPtr to ZScoreScaler
//' @param filename Input filename
//' @noRd
// [[Rcpp::export]]
void zscore_scaler_load(
    XPtr<ZScoreScaler> ptr, // NOLINT(performance-unnecessary-value-param)
    const std::string& filename
) {
    ptr->load(filename);
}

// =================================================================================
// Data Standardization: LpNormScaler
// =================================================================================

//' @title Create LpNormScaler
//'
//' @param norm_type Type of norm (1 for L1, 2 for L2)
//' @param with_mean Center data
//' @return XPtr to LpNormScaler
//' @noRd
// [[Rcpp::export]]
XPtr<LpNormScaler> lpnorm_scaler_create(int norm_type, bool with_mean) {
    LpNormScaler::NormType type = (norm_type == 1) ? LpNormScaler::NormType::L1 : LpNormScaler::NormType::L2;
    return XPtr<LpNormScaler>(new LpNormScaler(type, with_mean));
}

//' @title Fit LpNormScaler
//'
//' @param ptr XPtr to LpNormScaler
//' @param X Matrix to fit
//' @param threshold Numerical stability threshold
//' @noRd
// [[Rcpp::export]]
void lpnorm_scaler_fit(
    XPtr<LpNormScaler> ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> X,
    double threshold = 1e-12
) {
    ptr->fit(X, threshold);
}

//' @title Transform Inplace LpNormScaler
//'
//' @param ptr XPtr to LpNormScaler
//' @param X Matrix to transform
//' @noRd
// [[Rcpp::export]]
void lpnorm_scaler_transform_inplace(
    XPtr<LpNormScaler> ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> X
) {
    ptr->transform_inplace(X);
}

//' @title Inverse Transform Inplace LpNormScaler
//'
//' @param ptr XPtr to LpNormScaler
//' @param X Matrix to transform
//' @noRd
// [[Rcpp::export]]
void lpnorm_scaler_inverse_transform_inplace(
    XPtr<LpNormScaler> ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> X
) {
    ptr->inverse_transform_inplace(X);
}

//' @title Check if LpNormScaler is fitted
//'
//' @param ptr XPtr to LpNormScaler
//' @return Boolean indicating fit status
//' @noRd
// [[Rcpp::export]]
bool lpnorm_scaler_is_fitted(
    XPtr<LpNormScaler> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->is_fitted();
}

//' @title Get dropped indices from LpNormScaler
//'
//' @param ptr XPtr to LpNormScaler
//' @return Vector of dropped indices
//' @noRd
// [[Rcpp::export]]
std::vector<std::size_t> lpnorm_scaler_get_dropped_indices(
    XPtr<LpNormScaler> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->get_dropped_indices();
}

//' @title Get means from LpNormScaler
//'
//' @param ptr XPtr to LpNormScaler
//' @return Vector of means
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd lpnorm_scaler_get_means(
    XPtr<LpNormScaler> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->get_means();
}

//' @title Get scales from LpNormScaler
//'
//' @param ptr XPtr to LpNormScaler
//' @return Vector of scales
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd lpnorm_scaler_get_scales(
    XPtr<LpNormScaler> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->get_scales();
}

//' @title Get with_mean flag from LpNormScaler
//'
//' @param ptr XPtr to LpNormScaler
//' @return Boolean with_mean
//' @noRd
// [[Rcpp::export]]
bool lpnorm_scaler_get_with_mean(
    XPtr<LpNormScaler> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->get_with_mean();
}

//' @title Get with_norm flag from LpNormScaler
//'
//' @param ptr XPtr to LpNormScaler
//' @return Boolean with_norm
//' @noRd
// [[Rcpp::export]]
bool lpnorm_scaler_get_with_norm(
    XPtr<LpNormScaler> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->get_with_norm();
}

//' @title Get norm type from LpNormScaler
//'
//' @param ptr XPtr to LpNormScaler
//' @return Integer norm type
//' @noRd
// [[Rcpp::export]]
int lpnorm_scaler_get_norm_type(
    XPtr<LpNormScaler> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return static_cast<int>(ptr->get_norm_type());
}

//' @title Save LpNormScaler
//'
//' @param ptr XPtr to LpNormScaler
//' @param filename Output filename
//' @noRd
// [[Rcpp::export]]
void lpnorm_scaler_save(
    XPtr<LpNormScaler> ptr, // NOLINT(performance-unnecessary-value-param)
    const std::string& filename
) {
    ptr->save(filename);
}

//' @title Load LpNormScaler
//'
//' @param ptr XPtr to LpNormScaler
//' @param filename Input filename
//' @noRd
// [[Rcpp::export]]
void lpnorm_scaler_load(
    XPtr<LpNormScaler> ptr, // NOLINT(performance-unnecessary-value-param)
    const std::string& filename
) {
    ptr->load(filename);
}

// =================================================================================
// Model Selection: RidgeCV
// =================================================================================

//' @title Create Ridge CV
//'
//' @return XPtr to ridge_cv
//' @noRd
// [[Rcpp::export]]
XPtr<ridge_cv> ridge_cv_create() {
    return XPtr<ridge_cv>(new ridge_cv());
}

//' @title Fit Ridge CV
//'
//' @param ptr XPtr to ridge_cv
//' @param X Design matrix
//' @param y Response vector
//' @param num_folds Number of folds
//' @param n_lambda Number of lambdas
//' @param lambda_ratio Lambda ratio
//' @param seed Random seed
//' @noRd
// [[Rcpp::export]]
void ridge_cv_fit(
    XPtr<ridge_cv> ptr, // NOLINT(performance-unnecessary-value-param)
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    int num_folds,
    int n_lambda,
    double lambda_ratio,
    unsigned int seed
) {
    ptr->fit(X, y, num_folds, n_lambda, lambda_ratio, seed);
}

//' @title Get min CV error
//'
//' @param ptr XPtr to ridge_cv
//' @return Min CV error
//' @noRd
// [[Rcpp::export]]
double ridge_cv_min(
    XPtr<ridge_cv> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->cv_min();
}

//' @title Get 1SE CV error
//'
//' @param ptr XPtr to ridge_cv
//' @return 1SE CV error
//' @noRd
// [[Rcpp::export]]
double ridge_cv_1se(
    XPtr<ridge_cv> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->cv_1se();
}

//' @title Get min index
//'
//' @param ptr XPtr to ridge_cv
//' @return Index of min CV error
//' @noRd
// [[Rcpp::export]]
int ridge_cv_index_min(
    XPtr<ridge_cv> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return static_cast<int>(ptr->index_min());
}

//' @title Get 1SE index
//'
//' @param ptr XPtr to ridge_cv
//' @return Index of 1SE CV error
//' @noRd
// [[Rcpp::export]]
int ridge_cv_index_1se(
    XPtr<ridge_cv> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return static_cast<int>(ptr->index_1se());
}

//' @title Get Lambdas
//'
//' @param ptr XPtr to ridge_cv
//' @return Vector of lambdas
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ridge_cv_get_lambdas(
    XPtr<ridge_cv> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->lambdas();
}

//' @title Get CV Errors
//'
//' @param ptr XPtr to ridge_cv
//' @return Vector of CV errors
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ridge_cv_get_cv_errors(
    XPtr<ridge_cv> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->cv_mse();
}

//' @title Get CV Standard Errors
//'
//' @param ptr XPtr to ridge_cv
//' @return Vector of CV standard errors
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ridge_cv_get_cv_std(
    XPtr<ridge_cv> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->cv_sem();
}

// =================================================================================
// SVD Solver
// =================================================================================

//' @title Compute SVD
//'
//' @description Computes the top-M SVD of a matrix, dispatching between direct,
//'   Gram, and randomized paths based on matrix shape.
//'
//' @param X Numeric matrix (n x p).
//' @param M Integer, number of singular components to compute.
//'
//' @return Named list with elements \code{U} (n x M), \code{S} (M),
//'   and \code{V} (p x M).
//' @noRd
// [[Rcpp::export]]
Rcpp::List svd_compute(const Eigen::MatrixXd& X, int M) {
    SVDResult res = SVDSolver{}.compute(X, static_cast<Eigen::Index>(M));
    return Rcpp::List::create(
        Rcpp::Named("U") = res.U,
        Rcpp::Named("S") = res.S,
        Rcpp::Named("V") = res.V
    );
}


// =================================================================================
// PCA
// =================================================================================

//' @title Create PCA object
//'
//' @description Constructs a PCA instance bound to \code{X}. Centering and/or
//'   normalization are applied to \code{X} in-place immediately; call
//'   \code{pca_restore()} with the same matrix to undo them.
//'
//' @param X Numeric matrix (n x p). Modified in-place when center/normalize.
//' @param M Integer number of principal components.
//' @param center Logical, whether to center X in-place.
//' @param normalize Logical, whether to scale X columns in-place.
//'
//' @return XPtr to a PCA instance.
//' @noRd
// [[Rcpp::export]]
Rcpp::XPtr<PCA> pca_create(
    Rcpp::NumericMatrix X,
    int M,
    bool center = true,
    bool normalize = true
) {
    // Zero-copy view over the R buffer (X is REALSXP by contract). The PCA
    // constructor preprocesses X in place but retains no map into it, so the
    // returned XPtr carries no write-back-on-destruction hazard and needs no
    // prot slot; restore() is explicit via pca_restore(ptr, X).
    Eigen::Map<Eigen::MatrixXd> X_map(X.begin(), X.nrow(), X.ncol());
    return Rcpp::XPtr<PCA>(
        new PCA(X_map, static_cast<Eigen::Index>(M), center, normalize));
}

//' @title Fit PCA
//'
//' @description Fits the PCA bound at creation time; single-use per instance.
//'
//' @param ptr XPtr to PCA.
//'
//' @return Named list with \code{Z} (n x M scores), \code{V} (p x M loadings),
//'   and \code{explained_variance} (M-vector).
//' @noRd
// [[Rcpp::export]]
Rcpp::List pca_fit(
    Rcpp::XPtr<PCA> ptr // NOLINT(performance-unnecessary-value-param)
) {
    PCAResult res = ptr->fit();
    return Rcpp::List::create(
        Rcpp::Named("Z")                  = res.Z,
        Rcpp::Named("V")                  = res.V,
        Rcpp::Named("explained_variance") = res.explained_variance
    );
}

//' @title Restore X after PCA centering/normalization
//'
//' @param ptr XPtr to PCA.
//' @param X The same matrix that was passed to \code{pca_create()}.
//' @noRd
// [[Rcpp::export]]
void pca_restore(
    Rcpp::XPtr<PCA> ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix X
) {
    Eigen::Map<Eigen::MatrixXd> X_map(X.begin(), X.nrow(), X.ncol());
    ptr->restore(X_map);
}

//' @title Transform new data with fitted PCA
//'
//' @param ptr XPtr to PCA.
//' @param X_new Numeric matrix (n_new x p).
//'
//' @return Score matrix (n_new x M).
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd pca_transform(
    Rcpp::XPtr<PCA> ptr, // NOLINT(performance-unnecessary-value-param)
    const Eigen::MatrixXd& X_new
) {
    return ptr->transform(X_new);
}

//' @title Get PCA column means
//'
//' @param ptr XPtr to PCA.
//' @return Vector of column means (zero-vector when center = FALSE).
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd pca_get_means(
    Rcpp::XPtr<PCA> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->getMeans().transpose();
}

//' @title Get PCA column norms
//'
//' @param ptr XPtr to PCA.
//' @return Vector of column scaling factors (ones when normalize = FALSE).
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd pca_get_norms(
    Rcpp::XPtr<PCA> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->getNorms().transpose();
}

//' @title Get PCA loadings
//'
//' @param ptr XPtr to PCA.
//' @return Loading matrix (p x M).
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd pca_get_loadings(
    Rcpp::XPtr<PCA> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->getLoadings();
}

//' @title Get PCA explained variance
//'
//' @param ptr XPtr to PCA.
//' @return Numeric vector of explained variance per component.
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd pca_get_explained_variance(
    Rcpp::XPtr<PCA> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return ptr->getExplainedVariance();
}


// =================================================================================
// Ridge Regression (standalone solver)
// =================================================================================

//' @title Solve Ridge Regression
//'
//' @description Solves \eqn{\min \|y - X\beta\|^2 + \lambda\|\beta\|^2} for a
//'   single regularization value, dispatching between primal (n >= p) and dual
//'   (n < p) Cholesky solvers automatically.
//'
//' @param X Numeric matrix (n x p).
//' @param y Numeric response vector (n).
//' @param lambda Non-negative regularization parameter.
//'
//' @return Coefficient vector of length p.
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ridge_solve(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    double lambda
) {
    return RidgeSolver::solve(X, y, lambda);
}
