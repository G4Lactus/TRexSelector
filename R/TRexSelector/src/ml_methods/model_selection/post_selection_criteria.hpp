// ===================================================================================
// post_selection_criteria.hpp
// ===================================================================================
#ifndef TREX_ML_METHODS_MODEL_SELECTION_POST_SELECTION_CRITERIA_HPP
#define TREX_ML_METHODS_MODEL_SELECTION_POST_SELECTION_CRITERIA_HPP
// ===================================================================================
/**
 * @file post_selection_criteria.hpp
 *
 * @brief Fit criteria evaluated on a SELECTED SET: the least-squares refit on
 *        the selected columns and the information criteria built on its
 *        residual sum of squares (BIC, the extended BIC of Chen & Chen 2008,
 *        adjusted R^2).
 *
 * @details These are the classical "size-penalized fit" rules, provided so
 *   that a selection procedure's hyperparameter (the Tikhonov T-Rex's
 *   lambda_2, say) can be compared against them on identical runs. They
 *   judge the estimate, not the selection: with an FDR-controlled selection
 *   every extra member is a true positive with probability 1 - FDR, yet
 *   these criteria charge log n (BIC) or more (EBIC) for it, so they fall
 *   back to smaller supports wherever the added members carry coefficients
 *   an individual fit cannot justify. See tik_lambda2_power_rule.hpp for the
 *   selection-side rule.
 *
 *   Conventions. `ols_refit()` fits y on the selected columns of X; with
 *   `center = true` both are centered first (an intercept that is not
 *   counted in k). The least-norm solution is returned when the support is
 *   larger than the number of rows. With
 *
 *       RSS  = || y_c - X_S b ||^2,   n rows,   k = |S|,   p columns,
 *
 *       BIC        = n log(RSS / n) + k log n,
 *       EBIC_gamma = BIC + 2 gamma log C(p, k),          gamma in [0, 1],
 *       R^2_adj    = 1 - (RSS / (n - k - 1)) / (TSS / (n - 1)).
 *
 *   BIC / EBIC return +infinity for a saturated or degenerate fit
 *   (k >= n or RSS <= 0), which keeps such supports from winning an argmin.
 *
 *  References:
 *  -----------
 *   J. Chen and Z. Chen, "Extended Bayesian Information Criteria for Model Selection with Large Model
 *     Spaces," Biometrika, vol. 95, no. 3, pp. 759-771, 2008, doi: 10.1093/biomet/asn034.
 */
// ===================================================================================

// std includes
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

// Eigen includes
#include <Eigen/Core>
#include <Eigen/QR>

// Bit-based finiteness: std::isfinite folds to true when a consumer compiles
// this header with -ffast-math (the library itself is built IEEE-conformant).
#include <utils/fp_classify/fp_classify.hpp>

// ===================================================================================

namespace trex {
namespace ml_methods {
namespace model_selection {

namespace fpc = trex::utils::fp_classify;

/** @brief The least-squares refit on a selected set. */
struct refit_result {

    /** @brief Coefficients on the FULL column index (0 off the support). */
    Eigen::VectorXd coef;

    /** @brief Residual sum of squares of the refit. */
    double rss = 0.0;

    /** @brief Total sum of squares of the (centered) response. */
    double tss = 0.0;

    /** @brief Rows. */
    Eigen::Index n = 0;

    /** @brief Support size. */
    Eigen::Index k = 0;
};

/**
 * @brief Least-squares refit of y on the columns `support` of X.
 *
 * @param X       Design (n x p).
 * @param y       Response (n).
 * @param support 0-based column indices (the selected set; may be empty).
 * @param center  Center X's selected columns and y first (intercept model).
 *
 * @throws std::invalid_argument on a row mismatch or an index >= p.
 */
inline refit_result ols_refit(const Eigen::MatrixXd& X,
                              const Eigen::VectorXd& y,
                              const std::vector<std::size_t>& support,
                              bool center = true)
{
    if (X.rows() != y.size()) {
        throw std::invalid_argument("ols_refit: X and y disagree in the number of rows");
    }
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    refit_result r;
    r.n    = n;
    r.k    = static_cast<Eigen::Index>(support.size());
    r.coef = Eigen::VectorXd::Zero(p);

    Eigen::VectorXd yc = y;
    if (center) { yc.array() -= yc.mean(); }
    r.tss = yc.squaredNorm();
    if (support.empty()) {
        r.rss = r.tss;
        return r;
    }

    Eigen::MatrixXd Xs(n, r.k);
    for (Eigen::Index j = 0; j < r.k; ++j) {
        const auto col = static_cast<Eigen::Index>(support[static_cast<std::size_t>(j)]);
        if (col < 0 || col >= p) {
            throw std::invalid_argument("ols_refit: support index out of range");
        }
        Xs.col(j) = X.col(col);
        if (center) { Xs.col(j).array() -= Xs.col(j).mean(); }
    }
    const Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(Xs);
    const Eigen::VectorXd b = cod.solve(yc);
    for (Eigen::Index j = 0; j < r.k; ++j) {
        r.coef(static_cast<Eigen::Index>(support[static_cast<std::size_t>(j)])) = b(j);
    }
    r.rss = (yc - Xs * b).squaredNorm();
    return r;
}

/** @brief log C(p, k) through lgamma. */
inline double log_binomial(Eigen::Index p, Eigen::Index k)
{
    if (k < 0 || k > p) { return -std::numeric_limits<double>::infinity(); }
    const auto dp = static_cast<double>(p);
    const auto dk = static_cast<double>(k);
    return std::lgamma(dp + 1.0) - std::lgamma(dk + 1.0) - std::lgamma(dp - dk + 1.0);
}

/** @brief BIC = n log(RSS / n) + k log n; +inf when saturated (k >= n) or
 *  RSS <= 0. */
inline double bic(double rss, Eigen::Index n, Eigen::Index k)
{
    if (n <= 0 || k >= n || !(rss > 0.0)) { return std::numeric_limits<double>::infinity(); }
    const auto dn = static_cast<double>(n);
    return dn * std::log(rss / dn) + static_cast<double>(k) * std::log(dn);
}

/** @brief Extended BIC (Chen & Chen 2008): BIC + 2 gamma log C(p, k). */
inline double ebic(double rss, Eigen::Index n, Eigen::Index k, Eigen::Index p, double gamma)
{
    if (gamma < 0.0) { throw std::invalid_argument("ebic: gamma must be >= 0"); }
    const double b = bic(rss, n, k);
    if (!fpc::isfinite(b)) { return b; }
    return b + 2.0 * gamma * log_binomial(p, k);
}

/** @brief Adjusted R^2 of a fit with k slopes and an intercept; -inf when
 *  the degrees of freedom are exhausted. */
inline double adjusted_r2(double rss, double tss, Eigen::Index n, Eigen::Index k)
{
    if (n - k - 1 <= 0 || !(tss > 0.0)) { return -std::numeric_limits<double>::infinity(); }
    const double dof_res = static_cast<double>(n - k - 1);
    const double dof_tot = static_cast<double>(n - 1);
    return 1.0 - (rss / dof_res) / (tss / dof_tot);
}

/** @brief Index of the smallest value; the FIRST one on ties. */
inline std::size_t argmin_first(const std::vector<double>& values)
{
    if (values.empty()) { throw std::invalid_argument("argmin_first: empty input"); }
    std::size_t best = 0;
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] < values[best]) { best = i; }
    }
    return best;
}

/** @brief Index of the largest value; the FIRST one on ties. */
inline std::size_t argmax_first(const std::vector<double>& values)
{
    if (values.empty()) { throw std::invalid_argument("argmax_first: empty input"); }
    std::size_t best = 0;
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[best]) { best = i; }
    }
    return best;
}

}  // namespace model_selection
}  // namespace ml_methods
}  // namespace trex

#endif  // TREX_ML_METHODS_MODEL_SELECTION_POST_SELECTION_CRITERIA_HPP
