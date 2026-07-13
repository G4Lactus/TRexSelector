// ===================================================================================
// sd_calibration.hpp
// ===================================================================================
#ifndef TSOLVERS_SD_CALIBRATION_HPP
#define TSOLVERS_SD_CALIBRATION_HPP
// ===================================================================================
/**
 * @file sd_calibration.hpp
 *
 * @brief Pre-run calibration of the sparse-dummy race: choose the dummy
 *        sparsity k (rho_d) and budget L so the dummies can compete with
 *        the extreme null correlations at the given (n, p).
 *
 * @details The T-Rex race is sequential max-vs-max: as variables enter, the
 *          bar a dummy must clear drops through the order statistics of the
 *          null correlations |x_j^T r|. A sparsity-k dummy is bounded by its
 *          ceiling (top-k minus bottom-k residual entries, scaled), and with
 *          budget L its reach is the (1 - 1/L) quantile of its null. For
 *          every candidate k this utility computes the exact ceiling, an MC
 *          estimate of that quantile, and the implied entry position
 *          m_k(L) = #null bars above the dummy reach = how many null reals
 *          must enter before the first dummy can win. The recommendation is
 *          the smallest k with m_k(L) <= m_tol.
 *
 *          The bars are measured against a pilot residual: y is projected
 *          off its `pilot_screen` most correlated columns (a screening proxy
 *          for the post-signal state; the screened columns are excluded from
 *          the bars). X must be preprocessed to the solver contract
 *          (centered, equal-norm columns; y centered).
 *
 *          Cost: two X^T v passes, one sort of p bars, and mc_draws * |grid|
 *          MC samples — O(seconds) even at p = 1e6.
 */
// ===================================================================================

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace trex::tsolvers::sd_calibration {

struct Options {
    std::size_t L{0};             ///< dummy budget; 0 -> L_factor * p
    double L_factor{2.0};         ///< default budget multiple of p
    std::size_t m_tol{1};         ///< max #nulls allowed before the first dummy win
    std::size_t pilot_screen{20}; ///< #columns projected off y for the pilot residual
    std::vector<std::size_t> k_grid{1, 3, 7, 15, 25, 50, 75};
    std::size_t mc_draws{2'000'000};
    uint64_t seed{123};
};

struct Row {
    std::size_t k{0};
    double ceiling{0.0};      ///< max possible dummy |corr| given the residual
    double q_L{0.0};          ///< MC (1 - 1/L) quantile, capped at the ceiling
    std::size_t m_entry{0};   ///< #null bars above q_L
};

struct Result {
    bool feasible{false};     ///< some k in the grid met m_entry <= m_tol
    std::size_t k{0};         ///< recommended sparsity (smallest feasible,
                              ///< else argmin m_entry)
    double rho_d{0.0};        ///< rho_d that yields k in the SD solvers
    std::size_t L{0};         ///< recommended dummy budget
    double sigma_hat{0.0};    ///< ||r||/sqrt(n) of the pilot residual
    double bar{0.0};          ///< top null bar (m = 1)
    std::vector<Row> table;   ///< full per-k diagnostics
};

inline Result calibrate(const Eigen::Ref<const Eigen::MatrixXd>& X,
                        const Eigen::Ref<const Eigen::VectorXd>& y,
                        const Options& opt = {}) {
    const auto n = static_cast<std::size_t>(X.rows());
    const auto p = static_cast<std::size_t>(X.cols());

    Result res;
    res.L = (opt.L != 0) ? opt.L
                         : static_cast<std::size_t>(opt.L_factor * static_cast<double>(p));

    // --- 1. Pilot residual: project y off the strongest-screened columns ---
    Eigen::VectorXd corr = X.transpose() * y;
    Eigen::VectorXd r = y;
    std::vector<char> screened(p, 0);

    const std::size_t a = std::min<std::size_t>(opt.pilot_screen, n / 2);
    if (a > 0) {
        std::vector<std::size_t> ord(p);
        std::iota(ord.begin(), ord.end(), 0);
        std::nth_element(ord.begin(), ord.begin() + static_cast<std::ptrdiff_t>(a),
                         ord.end(), [&](std::size_t u, std::size_t v) {
                             return std::abs(corr(static_cast<Eigen::Index>(u))) >
                                    std::abs(corr(static_cast<Eigen::Index>(v)));
                         });
        Eigen::MatrixXd Xa(n, a);
        for (std::size_t i = 0; i < a; ++i) {
            Xa.col(static_cast<Eigen::Index>(i)) =
                X.col(static_cast<Eigen::Index>(ord[i]));
            screened[ord[i]] = 1;
        }
        r = y - Xa * Xa.colPivHouseholderQr().solve(y);
        corr = X.transpose() * r;
    }
    res.sigma_hat = r.norm() / std::sqrt(static_cast<double>(n));

    // --- 2. Null bars: |corr| of unscreened columns, sorted descending ---
    std::vector<double> bars;
    bars.reserve(p);
    for (std::size_t j = 0; j < p; ++j) {
        if (!screened[j]) bars.push_back(std::abs(corr(static_cast<Eigen::Index>(j))));
    }
    std::sort(bars.begin(), bars.end(), std::greater<double>());
    res.bar = bars.empty() ? 0.0 : bars[0];

    // --- 3. Per-k diagnostics: ceiling, MC quantile at 1 - 1/L, entry m ---
    std::vector<double> rv(n), rs(n);
    for (std::size_t i = 0; i < n; ++i) rv[i] = r(static_cast<Eigen::Index>(i));
    rs = rv;
    std::sort(rs.begin(), rs.end());

    std::mt19937_64 rng(opt.seed);
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<double> samples(opt.mc_draws);

    std::size_t best_m = static_cast<std::size_t>(-1);
    for (std::size_t k : opt.k_grid) {
        if (2 * k > n) continue;
        const double s = 1.0 / std::sqrt(2.0 * static_cast<double>(k));

        double top = 0.0, bot = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            top += rs[n - 1 - i];
            bot += rs[i];
        }
        const double ceiling = s * (top - bot);

        // MC of the dummy null (partial Fisher-Yates on a persistent array;
        // uniformity holds from any starting permutation).
        for (std::size_t t = 0; t < opt.mc_draws; ++t) {
            double acc = 0.0;
            for (std::size_t i = 0; i < 2 * k; ++i) {
                std::uniform_int_distribution<std::size_t> draw(i, n - 1);
                std::swap(idx[i], idx[draw(rng)]);
                acc += (i < k) ? rv[static_cast<std::size_t>(idx[i])]
                               : -rv[static_cast<std::size_t>(idx[i])];
            }
            samples[t] = std::abs(s * acc);
        }

        const double depth =
            static_cast<double>(opt.mc_draws) *
            (1.0 - 1.0 / static_cast<double>(res.L));
        const auto rank = static_cast<std::size_t>(
            std::min(depth, static_cast<double>(opt.mc_draws) - 1.0));
        std::nth_element(samples.begin(),
                         samples.begin() + static_cast<std::ptrdiff_t>(rank),
                         samples.end());
        const double q_L = std::min(samples[rank], ceiling);

        const auto m_entry = static_cast<std::size_t>(
            std::lower_bound(bars.begin(), bars.end(), q_L,
                             std::greater<double>()) - bars.begin());

        res.table.push_back(Row{k, ceiling, q_L, m_entry});

        if (!res.feasible && m_entry <= opt.m_tol) {
            res.feasible = true;
            res.k = k;
        } else if (!res.feasible && m_entry < best_m) {
            res.k = k;  // fallback: best-effort k if nothing is feasible
        }
        best_m = std::min(best_m, m_entry);
    }

    // rho_d that reproduces k under the solvers' 2*floor(n*rho/2) formula
    // (+0.5 guards against floating-point floor(k - ulp) -> k-1).
    if (res.k > 0) {
        res.rho_d = (2 * res.k == n)
            ? 1.0
            : (2.0 * static_cast<double>(res.k) + 0.5) / static_cast<double>(n);
    }
    return res;
}

}  // namespace trex::tsolvers::sd_calibration

// ===================================================================================
#endif /* TSOLVERS_SD_CALIBRATION_HPP */
// ===================================================================================
