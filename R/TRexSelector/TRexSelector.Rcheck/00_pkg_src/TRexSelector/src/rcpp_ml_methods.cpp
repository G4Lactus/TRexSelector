#include <RcppEigen.h>
#include <ml_methods/standardization/z_score_scaler.hpp>
#include <ml_methods/standardization/lp_norm_scaler.hpp>
#include <ml_methods/model_selection/ridge_cv.hpp>
#include <ml_methods/model_selection/ridge_gcv.hpp>

// [[Rcpp::plugins(cpp20)]]
// [[Rcpp::depends(RcppEigen)]]

using namespace trex::ml_methods::standardization;
using namespace trex::ml_methods::model_selection;
using namespace Rcpp;

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
// Model Selection: RidgeGCV
// =================================================================================

//' @title Create Ridge GCV
//'
//' @return XPtr to ridge_gcv
//' @noRd
// [[Rcpp::export]]
XPtr<ridge_gcv> ridge_gcv_create() {
    return XPtr<ridge_gcv>(new ridge_gcv());
}

//' @title Fit Ridge GCV
//'
//' @param ptr XPtr to ridge_gcv
//' @param X Design matrix
//' @param y Response vector
//' @noRd
// [[Rcpp::export]]
void ridge_gcv_fit(
    XPtr<ridge_gcv> ptr, // NOLINT(performance-unnecessary-value-param)
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y
) {
    ptr->fit(X, y);
}

//' @title Solve Ridge GCV
//'
//' @param ptr XPtr to ridge_gcv
//' @param lambda_val Lambda value
//' @return Coefficients
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ridge_gcv_solve(
    XPtr<ridge_gcv> ptr, // NOLINT(performance-unnecessary-value-param)
    double lambda_val
) {
    return ptr->solve(lambda_val);
}

//' @title Get GCV Score
//'
//' @param ptr XPtr to ridge_gcv
//' @param lambda_val Lambda value
//' @return GCV score
//' @noRd
// [[Rcpp::export]]
double ridge_gcv_gcv_score(
    XPtr<ridge_gcv> ptr, // NOLINT(performance-unnecessary-value-param)
    double lambda_val
) {
    return ptr->gcv(lambda_val);
}

//' @title Predict with Ridge GCV
//'
//' @param ptr XPtr to ridge_gcv
//' @param X_new New data
//' @param lambda_val Lambda value
//' @return Predictions
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ridge_gcv_predict(
    XPtr<ridge_gcv> ptr, // NOLINT(performance-unnecessary-value-param)
    const Eigen::MatrixXd& X_new,
    double lambda_val
) {
    return ptr->predict(X_new, lambda_val);
}

//' @title Predict Path with Ridge GCV
//'
//' @param ptr XPtr to ridge_gcv
//' @param X_new New data
//' @param lambdas Lambda values
//' @return Predictions
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd ridge_gcv_predict_path(
    XPtr<ridge_gcv> ptr, // NOLINT(performance-unnecessary-value-param)
    const Eigen::MatrixXd& X_new,
    const Eigen::VectorXd& lambdas
) {
    return ptr->predict_path(X_new, lambdas);
}

//' @title Get Default Lambda Grid from Ridge GCV
//'
//' @param ptr XPtr to ridge_gcv
//' @param num_lambda Number of lambdas
//' @param ratio Ratio
//' @return Lambda grid
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ridge_gcv_default_lambda_grid(
    XPtr<ridge_gcv> ptr, // NOLINT(performance-unnecessary-value-param)
    int num_lambda,
    double ratio
) {
    return ptr->default_lambda_grid(num_lambda, ratio);
}

//' @title Get number of samples from Ridge GCV
//'
//' @param ptr XPtr to ridge_gcv
//' @return Number of samples
//' @noRd
// [[Rcpp::export]]
int ridge_gcv_n_samples(
    XPtr<ridge_gcv> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return static_cast<int>(ptr->n_samples());
}

//' @title Get number of features from Ridge GCV
//'
//' @param ptr XPtr to ridge_gcv
//' @return Number of features
//' @noRd
// [[Rcpp::export]]
int ridge_gcv_n_features(
    XPtr<ridge_gcv> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return static_cast<int>(ptr->n_features());
}

//' @title Get rank from Ridge GCV
//'
//' @param ptr XPtr to ridge_gcv
//' @return Rank
//' @noRd
// [[Rcpp::export]]
int ridge_gcv_rank(
    XPtr<ridge_gcv> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return static_cast<int>(ptr->rank());
}

//' @title Solve Path Ridge GCV
//'
//' @param ptr XPtr to ridge_gcv
//' @param lambdas Lambda values
//' @return List of path results
//' @noRd
// [[Rcpp::export]]
List ridge_gcv_solve_path(
    XPtr<ridge_gcv> ptr, // NOLINT(performance-unnecessary-value-param)
    const Eigen::VectorXd& lambdas
) {
    auto path = ptr->solve_path(lambdas);
    return List::create(
        Named("lambdas") = path.lambdas,
        Named("coefficients") = path.coefficients,
        Named("gcv_scores") = path.gcv_scores,
        Named("df_effective") = path.df_effective,
        Named("best_index") = path.best_index
    );
}

//' @title Get optimal lambda from Ridge GCV
//'
//' @param ptr XPtr to ridge_gcv
//' @param num_grid Number of grid points
//' @param tol Tolerance
//' @return Optimal GCV lambda
//' @noRd
// [[Rcpp::export]]
double ridge_gcv_optimal(
    XPtr<ridge_gcv> ptr, // NOLINT(performance-unnecessary-value-param)
    int num_grid,
    double tol
) {
    return ptr->gcv_optimal(num_grid, tol);
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
