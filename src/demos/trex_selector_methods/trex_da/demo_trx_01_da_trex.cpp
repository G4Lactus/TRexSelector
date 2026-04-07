// ==============================================================================
// demo_trx_01_da_trex.cpp
// ==============================================================================
/**
 * @file demo_trx_01_da_trex.cpp
 *
 * @brief Demonstration of the Dependency-Aware T-Rex (DA-TRex) Selector.
 *
 * @details
 *  Runs all five DA methods on purpose-built synthetic data:
 *
 *    1. AR(1)         — ordered predictors, Toeplitz covariance
 *    2. Equi          — single latent factor, compound symmetry
 *    3. NN            — banded MA(κ) covariance
 *    4. BT            — two-level hierarchical block structure
 *    5. Prior groups  — multi-level latent factors, known labels
 *
 *  Each scenario uses n = 150, p = 500, s = 5, amplitude = 3, K = 20,
 *  tFDR = 0.2.
 *
 *  Reference: R/references/trex_da/demo_trex_da.R,
 *             R/references/trex_da/dgp_functions.R
 */
// ==============================================================================

// std includes
#include "trex_selector_methods/utils/trex_solver_dispatch.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// T-Rex DA includes
#include <trex_selector_methods/trex_da/trex_da.hpp>

// Evaluation metrics
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>


// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace cdianostics = trex::utils::eval::cdiagnostics;
namespace rates       = trex::utils::eval::rates;

using trex::trex_selector_methods::trex_da::TRexDASelector;
using trex::trex_selector_methods::trex_da::TRexDAControlParameter;
using trex::trex_selector_methods::trex_da::DAMethod;
using trex::trex_selector_methods::trex_core::TRexControlParameter;
using trex::trex_selector_methods::trex_core::LLoopStrategy;
using trex::trex_selector_methods::utils::solver_dispatch::SolverTypeForTRex;

namespace hac = trex::ml_methods::clustering::hierarchical::agglomerative;


// ==============================================================================
// Global parameters (matching the R demo)
// ==============================================================================

struct DemoParams {
    Eigen::Index n         = 150;    // observations
    Eigen::Index p         = 500;    // predictors
    Eigen::Index s         = 5;      // active predictors
    double       amplitude = 3.0;    // signal coefficient value
    double       sigma     = 1.0;    // noise std
    std::size_t  K         = 20;     // random experiments
    double       tFDR      = 0.2;    // target FDR
    int          seed      = 2026;   // base RNG seed
};


// ==============================================================================
// DGP result structure
// ==============================================================================

struct DGPResult {
    Eigen::MatrixXd X;                    // n x p
    Eigen::VectorXd y;                    // n
    std::vector<std::size_t> true_support; // indices of active predictors
};


// ==============================================================================
// Inline DGP helpers
// ==============================================================================

namespace {

/**
 * @brief Build sparse coefficient vector (first-s active).
 */
Eigen::VectorXd make_beta(Eigen::Index p, Eigen::Index s, double amplitude) {
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    for (Eigen::Index j = 0; j < s; ++j) {
        beta(j) = amplitude;
    }
    return beta;
}

/**
 * @brief Generate response y = X * beta + eps.
 */
Eigen::VectorXd make_response(const Eigen::MatrixXd& X,
                               const Eigen::VectorXd& beta,
                               double sigma, std::mt19937& rng) {
    const Eigen::Index n = X.rows();
    std::normal_distribution<double> dist(0.0, sigma);

    Eigen::VectorXd y = X * beta;
    for (Eigen::Index i = 0; i < n; ++i) {
        y(i) += dist(rng);
    }
    return y;
}


// ── DGP 1: AR(1) ─────────────────────────────────────────────────────────────

/**
 * @brief AR(1) Toeplitz covariance: Sigma[j,k] = rho^|j-k|.
 *
 * X rows generated via: X[i,j] = rho * X[i,j-1] + sqrt(1-rho^2) * eta[i,j].
 */
DGPResult dgp_ar1(const DemoParams& par, double rho) {
    std::mt19937 rng(static_cast<unsigned>(par.seed));
    std::normal_distribution<double> norm(0.0, 1.0);

    Eigen::MatrixXd X(par.n, par.p);
    const double scale = std::sqrt(1.0 - rho * rho);

    for (Eigen::Index i = 0; i < par.n; ++i) {
        X(i, 0) = norm(rng);
        for (Eigen::Index j = 1; j < par.p; ++j) {
            X(i, j) = rho * X(i, j - 1) + scale * norm(rng);
        }
    }

    auto beta = make_beta(par.p, par.s, par.amplitude);
    auto y    = make_response(X, beta, par.sigma, rng);

    std::vector<std::size_t> support(static_cast<std::size_t>(par.s));
    std::iota(support.begin(), support.end(), 0);

    return {std::move(X), std::move(y), std::move(support)};
}


// ── DGP 2: Equicorrelation ───────────────────────────────────────────────────

/**
 * @brief Compound symmetry via single latent factor:
 *        X[i,j] = sqrt(rho) * f[i] + sqrt(1-rho) * eta[i,j].
 */
DGPResult dgp_equi(const DemoParams& par, double rho) {
    std::mt19937 rng(static_cast<unsigned>(par.seed));
    std::normal_distribution<double> norm(0.0, 1.0);

    const double load_f   = std::sqrt(rho);
    const double load_eta = std::sqrt(1.0 - rho);

    Eigen::MatrixXd X(par.n, par.p);
    for (Eigen::Index i = 0; i < par.n; ++i) {
        const double f_i = norm(rng);
        for (Eigen::Index j = 0; j < par.p; ++j) {
            X(i, j) = load_f * f_i + load_eta * norm(rng);
        }
    }

    auto beta = make_beta(par.p, par.s, par.amplitude);
    auto y    = make_response(X, beta, par.sigma, rng);

    std::vector<std::size_t> support(static_cast<std::size_t>(par.s));
    std::iota(support.begin(), support.end(), 0);

    return {std::move(X), std::move(y), std::move(support)};
}


// ── DGP 3: Nearest Neighbours (NN) ──────────────────────────────────────────

/**
 * @brief Banded covariance via MA(κ) convolution:
 *        X[i,j] = Σ_{l=0}^{κ} θ[l] * eta[i, j+l],
 *        with θ[l] = c * rho^l (normalised to unit variance).
 */
DGPResult dgp_nn(const DemoParams& par, int kappa, double rho) {
    std::mt19937 rng(static_cast<unsigned>(par.seed));
    std::normal_distribution<double> norm(0.0, 1.0);

    // MA coefficients normalised to unit variance
    std::vector<double> theta(static_cast<std::size_t>(kappa + 1));
    double sum_sq = 0.0;
    for (int l = 0; l <= kappa; ++l) {
        theta[static_cast<std::size_t>(l)] = std::pow(rho, l);
        sum_sq += theta[static_cast<std::size_t>(l)] * theta[static_cast<std::size_t>(l)];
    }
    const double c = 1.0 / std::sqrt(sum_sq);
    for (auto& t : theta) t *= c;

    // Extended noise: n x (p + kappa)
    const Eigen::Index ext_cols = par.p + kappa;
    Eigen::MatrixXd eta(par.n, ext_cols);
    for (Eigen::Index i = 0; i < par.n; ++i) {
        for (Eigen::Index j = 0; j < ext_cols; ++j) {
            eta(i, j) = norm(rng);
        }
    }

    // MA convolution
    Eigen::MatrixXd X(par.n, par.p);
    for (Eigen::Index j = 0; j < par.p; ++j) {
        X.col(j).setZero();
        for (int l = 0; l <= kappa; ++l) {
            X.col(j) += theta[static_cast<std::size_t>(l)] * eta.col(j + l);
        }
    }

    auto beta = make_beta(par.p, par.s, par.amplitude);
    auto y    = make_response(X, beta, par.sigma, rng);

    std::vector<std::size_t> support(static_cast<std::size_t>(par.s));
    std::iota(support.begin(), support.end(), 0);

    return {std::move(X), std::move(y), std::move(support)};
}


// ── DGP 4: Binary Tree (hierarchical blocks) ─────────────────────────────────

/**
 * @brief Two-level block covariance:
 *   X[i,j] = sqrt(rho_b) * f0 + sqrt(rho_w - rho_b) * f_block + sqrt(1-rho_w) * eta.
 */
DGPResult dgp_bt(const DemoParams& par,
                  int n_blocks, double rho_within, double rho_between) {
    std::mt19937 rng(static_cast<unsigned>(par.seed));
    std::normal_distribution<double> norm(0.0, 1.0);

    const Eigen::Index block_size = par.p / n_blocks;

    const double a = std::sqrt(rho_between);
    const double b = std::sqrt(rho_within - rho_between);
    const double c = std::sqrt(1.0 - rho_within);

    Eigen::MatrixXd X(par.n, par.p);

    for (Eigen::Index i = 0; i < par.n; ++i) {
        const double f0 = norm(rng);

        // Per-block factors
        std::vector<double> f_block(static_cast<std::size_t>(n_blocks));
        for (int m = 0; m < n_blocks; ++m) {
            f_block[static_cast<std::size_t>(m)] = norm(rng);
        }

        for (Eigen::Index j = 0; j < par.p; ++j) {
            const int block_id = static_cast<int>(j / block_size);
            X(i, j) = a * f0
                     + b * f_block[static_cast<std::size_t>(block_id)]
                     + c * norm(rng);
        }
    }

    auto beta = make_beta(par.p, par.s, par.amplitude);
    auto y    = make_response(X, beta, par.sigma, rng);

    std::vector<std::size_t> support(static_cast<std::size_t>(par.s));
    std::iota(support.begin(), support.end(), 0);

    return {std::move(X), std::move(y), std::move(support)};
}


// ── DGP 5: Prior Groups (multi-level latent factor) ──────────────────────────

/**
 * @brief Multi-level latent model:
 *   X[i,j] = Σ_x sqrt(rho[x] - rho[x+1]) * f_{g_x(j)} + sqrt(1 - rho[0]) * eta.
 *
 * @param groups     L vectors of length p (group labels, fine to coarse)
 * @param rho_levels L strictly decreasing correlation levels in (0, 1)
 */
DGPResult dgp_groups(const DemoParams& par,
                      const std::vector<std::vector<Eigen::Index>>& groups,
                      const std::vector<double>& rho_levels) {
    std::mt19937 rng(static_cast<unsigned>(par.seed));
    std::normal_distribution<double> norm(0.0, 1.0);

    const auto L = groups.size();

    // Extended rho: rho[L] = 0
    std::vector<double> rho_ext(rho_levels.begin(), rho_levels.end());
    rho_ext.push_back(0.0);

    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(par.n, par.p);

    for (std::size_t x = 0; x < L; ++x) {
        const double loading = std::sqrt(rho_ext[x] - rho_ext[x + 1]);

        // Find unique group ids at this level
        std::vector<Eigen::Index> unique_ids(groups[x].begin(), groups[x].end());
        std::sort(unique_ids.begin(), unique_ids.end());
        unique_ids.erase(std::unique(unique_ids.begin(), unique_ids.end()),
                         unique_ids.end());

        for (auto gid : unique_ids) {
            // Sample one factor per group
            Eigen::VectorXd f(par.n);
            for (Eigen::Index i = 0; i < par.n; ++i) {
                f(i) = norm(rng);
            }

            // Add factor contribution to all group members
            for (Eigen::Index j = 0; j < par.p; ++j) {
                if (groups[x][static_cast<std::size_t>(j)] == gid) {
                    X.col(j) += loading * f;
                }
            }
        }
    }

    // Idiosyncratic noise
    const double eta_scale = std::sqrt(1.0 - rho_levels[0]);
    for (Eigen::Index i = 0; i < par.n; ++i) {
        for (Eigen::Index j = 0; j < par.p; ++j) {
            X(i, j) += eta_scale * norm(rng);
        }
    }

    auto beta = make_beta(par.p, par.s, par.amplitude);
    auto y    = make_response(X, beta, par.sigma, rng);

    std::vector<std::size_t> support(static_cast<std::size_t>(par.s));
    std::iota(support.begin(), support.end(), 0);

    return {std::move(X), std::move(y), std::move(support)};
}

} // anonymous namespace


// ==============================================================================
// Evaluation helper
// ==============================================================================

struct ScenarioMetrics {
    std::string name;
    int  n_selected = 0;
    int  TP         = 0;
    int  FP         = 0;
    double TPP      = 0.0;
    double FDP      = 0.0;
    int  T_stop     = 0;
    int  num_dummies= 0;
    double v_thresh = 0.0;
    double rho_thresh = std::numeric_limits<double>::quiet_NaN();
};


ScenarioMetrics evaluate(const std::string& scenario_name,
                          const TRexDASelector::DASelectionResult& result,
                          const std::vector<std::size_t>& true_support) {

    // Collect selected indices
    std::vector<std::size_t> selected;
    for (Eigen::Index j = 0; j < result.selected_var.size(); ++j) {
        if (result.selected_var(j) != 0) {
            selected.push_back(static_cast<std::size_t>(j));
        }
    }

    double fdp = rates::compute_fdp(selected, true_support);
    double tpp = rates::compute_tpp(selected, true_support);

    int n_sel = static_cast<int>(selected.size());
    int tp    = static_cast<int>(std::round(tpp * static_cast<double>(true_support.size())));
    int fp    = n_sel - tp;

    return {
        scenario_name,
        n_sel, tp, fp,
        tpp, fdp,
        static_cast<int>(result.T_stop),
        static_cast<int>(result.num_dummies),
        result.v_thresh,
        result.rho_thresh
    };
}


void print_scenario(const ScenarioMetrics& m, double tFDR) {
    std::cout << "\n"
              << std::string(60, '-') << "\n"
              << "  " << m.name << "\n"
              << std::string(60, '-') << "\n";
    std::cout << std::fixed;
    std::cout << "  Calibration: T_stop = " << m.T_stop
              << ",  dummies = " << m.num_dummies
              << ",  v* = " << std::setprecision(3) << m.v_thresh << "\n";
    if (!std::isnan(m.rho_thresh)) {
        std::cout << "               rho* = " << std::setprecision(4) << m.rho_thresh << "\n";
    }
    std::cout << "  Selection:   " << m.n_selected << " selected  |  TP = " << m.TP
              << "  FP = " << m.FP << "\n";
    std::cout << "               TPP = " << std::setprecision(2) << m.TPP
              << "  |  FDP = " << m.FDP
              << "  (target <= " << tFDR << ")\n";
    std::cout << std::string(60, '-') << "\n";
}


// ==============================================================================
// Summary table
// ==============================================================================

void print_summary(const std::vector<ScenarioMetrics>& results, double tFDR, int s) {
    std::cout << "\n" << std::string(78, '=') << "\n";
    std::cout << std::left  << "  " << std::setw(28) << "Scenario"
              << std::right << std::setw(5)  << "Sel."
              << std::setw(4)  << "TP"
              << std::setw(4)  << "FP"
              << std::setw(7)  << "TPP"
              << std::setw(7)  << "FDP"
              << std::setw(7)  << "T_stop"
              << "\n";
    std::cout << std::string(78, '-') << "\n";

    for (const auto& m : results) {
        std::cout << std::fixed;
        std::cout << std::left  << "  " << std::setw(28) << m.name
                  << std::right << std::setw(5)  << m.n_selected
                  << std::setw(4)  << m.TP
                  << std::setw(4)  << m.FP
                  << std::setw(7)  << std::setprecision(3) << m.TPP
                  << std::setw(7)  << std::setprecision(3) << m.FDP
                  << std::setw(7)  << m.T_stop
                  << "\n";
    }

    std::cout << std::string(78, '=') << "\n";
    std::cout << "  Target FDR: " << std::setprecision(2) << tFDR
              << "   |   True active set: first " << s << " variables\n";
    std::cout << std::string(78, '=') << "\n\n";
}


// ==============================================================================
// Main
// ==============================================================================

int main() {
    DemoParams par;

    cdianostics::print_section_header(
        "DA-TRex Demo  |  n = " + std::to_string(par.n)
        + "  p = " + std::to_string(par.p)
        + "  s = " + std::to_string(par.s));

    // Base T-Rex control (shared across all scenarios)
    TRexControlParameter trex_ctrl;
    trex_ctrl.K = par.K;
    trex_ctrl.solver_type = SolverTypeForTRex::TOMP;

    std::vector<ScenarioMetrics> all_metrics;


    // ── Scenario 1: AR(1) ────────────────────────────────────────────────────

    std::cout << "\n[1/5] Generating AR(1) data  (rho = 0.7) ...\n";
    auto dat_ar1 = dgp_ar1(par, 0.7);

    {
        Eigen::Map<Eigen::MatrixXd> X_map(dat_ar1.X.data(), par.n, par.p);
        Eigen::Map<Eigen::VectorXd> y_map(dat_ar1.y.data(), par.n);

        TRexDAControlParameter da_ctrl;
        da_ctrl.method = DAMethod::AR1;

        std::cout << "[1/5] Running trex+DA+AR1 ...\n";
        TRexDASelector selector(X_map,
                                y_map,
                                par.tFDR,
                                da_ctrl,
                                trex_ctrl,
                                par.seed,
                                true);
        selector.select();

        auto m = evaluate(
            "AR(1)  trex+DA+AR1",
            selector.getDAResult(),
            dat_ar1.true_support
        );
        print_scenario(m, par.tFDR);
        all_metrics.push_back(std::move(m));
    }


    // ── Scenario 2: Equicorrelation ──────────────────────────────────────────

    std::cout << "\n[2/5] Generating equicorrelation data  (rho = 0.5) ...\n";
    auto dat_equi = dgp_equi(par, 0.5);

    {
        Eigen::Map<Eigen::MatrixXd> X_map(dat_equi.X.data(), par.n, par.p);
        Eigen::Map<Eigen::VectorXd> y_map(dat_equi.y.data(), par.n);

        TRexDAControlParameter da_ctrl;
        da_ctrl.method = DAMethod::EQUI;

        std::cout << "[2/5] Running trex+DA+equi ...\n";
        TRexDASelector selector(X_map,
                                y_map,
                                par.tFDR,
                                da_ctrl,
                                trex_ctrl,
                                par.seed,
                                true);
        selector.select();

        auto m = evaluate(
            "Equi  trex+DA+equi",
            selector.getDAResult(),
            dat_equi.true_support
        );
        print_scenario(m, par.tFDR);
        all_metrics.push_back(std::move(m));
    }


    // ── Scenario 3: Nearest Neighbours ───────────────────────────────────────

    std::cout << "\n[3/5] Generating NN data  (kappa = 3, rho = 0.7) ...\n";
    auto dat_nn = dgp_nn(par, 3, 0.7);

    {
        Eigen::Map<Eigen::MatrixXd> X_map(dat_nn.X.data(), par.n, par.p);
        Eigen::Map<Eigen::VectorXd> y_map(dat_nn.y.data(), par.n);

        TRexDAControlParameter da_ctrl;
        da_ctrl.method = DAMethod::NN;

        std::cout << "[3/5] Running trex+DA+NN ...\n";
        TRexDASelector selector(X_map,
                                y_map,
                                par.tFDR,
                                da_ctrl,
                                trex_ctrl,
                                par.seed,
                                true);
        selector.select();

        auto m = evaluate(
            "NN  trex+DA+NN",
            selector.getDAResult(),
            dat_nn.true_support
        );
        print_scenario(m, par.tFDR);
        all_metrics.push_back(std::move(m));
    }


    // ── Scenario 4: Binary Tree ──────────────────────────────────────────────

    std::cout << "\n[4/5] Generating BT data  (10 blocks of 50, rho_w=0.8, rho_b=0.1) ...\n";
    auto dat_bt = dgp_bt(par, 10, 0.8, 0.1);

    {
        Eigen::Map<Eigen::MatrixXd> X_map(dat_bt.X.data(), par.n, par.p);
        Eigen::Map<Eigen::VectorXd> y_map(dat_bt.y.data(), par.n);

        TRexDAControlParameter da_ctrl;
        da_ctrl.method     = DAMethod::BT;
        da_ctrl.hc_linkage = hac::LinkageMethod::Average;

        std::cout << "[4/5] Running trex+DA+BT  (average linkage) ...\n";
        TRexDASelector selector(X_map,
                                y_map,
                                par.tFDR,
                                da_ctrl,
                                trex_ctrl,
                                par.seed,
                                true);
        selector.select();

        auto m = evaluate(
            "BT  trex+DA+BT (avg)",
            selector.getDAResult(),
            dat_bt.true_support
        );
        print_scenario(m, par.tFDR);
        all_metrics.push_back(std::move(m));
    }


    // ── Scenario 5: Prior Groups ─────────────────────────────────────────────

    std::cout << "\n[5/5] Generating prior-groups data  (3 levels: 50/10/2 groups) ...\n";

    // Build group labels (matching R demo)
    std::vector<std::vector<Eigen::Index>> groups_demo(3);
    for (Eigen::Index j = 0; j < par.p; ++j) {
        groups_demo[0].push_back(j / 10);          // level 1: 50 groups of 10
        groups_demo[1].push_back(j / 50);          // level 2: 10 groups of 50
        groups_demo[2].push_back(j / 250);          // level 3:  2 groups of 250
    }
    std::vector<double> rho_levels = {0.80, 0.50, 0.20};

    auto dat_grp = dgp_groups(par, groups_demo, rho_levels);

    {
        Eigen::Map<Eigen::MatrixXd> X_map(dat_grp.X.data(), par.n, par.p);
        Eigen::Map<Eigen::VectorXd> y_map(dat_grp.y.data(), par.n);

        TRexDAControlParameter da_ctrl;
        da_ctrl.prior_groups    = groups_demo;
        da_ctrl.rho_grid_labels = rho_levels;

        std::cout << "[5/5] Running trex+DA with prior groups ...\n";
        TRexDASelector selector(X_map,
                                y_map,
                                par.tFDR,
                                da_ctrl,
                                trex_ctrl,
                                par.seed,
                                true);
        selector.select();

        auto m = evaluate(
            "Prior groups",
            selector.getDAResult(),
            dat_grp.true_support
        );
        print_scenario(m, par.tFDR);
        all_metrics.push_back(std::move(m));
    }


    // ── Summary ──────────────────────────────────────────────────────────────

    print_summary(all_metrics, par.tFDR, static_cast<int>(par.s));

    return 0;
}
