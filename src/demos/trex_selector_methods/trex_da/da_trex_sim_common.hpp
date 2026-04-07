// ==============================================================================
// da_trex_sim_common.hpp
// ==============================================================================
#ifndef DEMOS_TREX_SELECTOR_METHODS_DA_TREX_SIM_COMMON_HPP
#define DEMOS_TREX_SELECTOR_METHODS_DA_TREX_SIM_COMMON_HPP
// ==============================================================================
/**
 * @file da_trex_sim_common.hpp
 *
 * @brief Shared infrastructure for block-diagonal Toeplitz DA-TRex MC sims.
 *
 * Provides:
 *   - BlockSimConfig — base simulation parameters
 *   - SolverInfo     — lightweight solver descriptor
 *   - DGPData        — (X, y, true_support) triplet
 *   - dgp_block_toeplitz()       — Gaussian block-diag Toeplitz DGP
 *   - dgp_block_toeplitz_heavy() — heavy-tailed extension (t-distributed)
 *   - make_trex_control()        — default TRexControlParameter
 *   - make_da_bt_control()       — default DA BT control
 *   - save_and_print_grid_results() — tabular output to console + file
 */
// ==============================================================================

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <trex_selector_methods/trex_da/trex_da.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>


// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace da_sim {

namespace cdianostics = trex::utils::eval::cdiagnostics;
namespace rates       = trex::utils::eval::rates;
namespace dummygen    = trex::utils::datageneration::dummygen;
namespace hac         = trex::ml_methods::clustering::hierarchical::agglomerative;

using trex::trex_selector_methods::trex_da::TRexDASelector;
using trex::trex_selector_methods::trex_da::TRexDAControlParameter;
using trex::trex_selector_methods::trex_da::DAMethod;
using trex::trex_selector_methods::trex_core::TRexControlParameter;
using trex::trex_selector_methods::trex_core::LLoopStrategy;
using trex::trex_selector_methods::utils::solver_dispatch::SolverTypeForTRex;
using trex::trex_selector_methods::utils::solver_dispatch::SolverHyperparameters;


// ==============================================================================
// Configuration
// ==============================================================================

struct BlockSimConfig {
    int  n         = 150;     // observations
    int  p         = 500;     // predictors
    int  G         = 5;       // active groups (blocks with an active variable)
    int  g         = 5;       // group (block) size
    double rho     = 0.7;     // within-block correlation
    double snr     = 2.0;     // signal-to-noise ratio
    double tFDR    = 0.2;     // target FDR
    std::size_t K  = 20;      // random experiments per selector run
    std::size_t num_MC = 200; // Monte Carlo trials
    int base_seed  = 2026;    // base RNG seed
};

struct SolverInfo {
    SolverTypeForTRex type;
    std::string       name;
    double            rho_afs = 0.0;
};


// ==============================================================================
// DGP result
// ==============================================================================

struct DGPData {
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    std::vector<std::size_t> true_support;
};


// ==============================================================================
// Block-diagonal Toeplitz DGP (Gaussian)
// ==============================================================================

/**
 * @brief Block-diagonal Toeplitz DGP.
 *
 * p predictors in p/g blocks of size g.  Within each block the covariance
 * is Toeplitz: Sigma_{block}[i,j] = rho^|i-j|.  Between blocks: zero.
 *
 * Active variables: one per active block at positions {0, g, 2g, ..., (G-1)*g}.
 * Response: y = X*beta + noise with noise calibrated to target SNR.
 */
inline DGPData dgp_block_toeplitz(int n, int p, int G, int g,
                                   double rho, double snr, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    const int num_blocks = p / g;

    // Build g x g Toeplitz block and its Cholesky factor
    Eigen::MatrixXd T(g, g);
    for (int i = 0; i < g; ++i)
        for (int j = 0; j < g; ++j)
            T(i, j) = std::pow(rho, std::abs(i - j));

    Eigen::LLT<Eigen::MatrixXd> llt(T);
    Eigen::MatrixXd L = llt.matrixL();  // g x g

    // Generate X block-wise: X_block = Z_block * L^T
    Eigen::MatrixXd X(n, p);
    for (int b = 0; b < num_blocks; ++b) {
        Eigen::MatrixXd Z(n, g);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < g; ++j)
                Z(i, j) = norm(rng);
        X.block(0, b * g, n, g).noalias() = Z * L.transpose();
    }

    // Sparse beta: one active per active block
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    std::vector<std::size_t> support;
    support.reserve(static_cast<std::size_t>(G));
    for (int b = 0; b < G; ++b) {
        beta(b * g) = 1.0;
        support.push_back(static_cast<std::size_t>(b * g));
    }

    // Response with SNR-calibrated noise
    Eigen::VectorXd signal = X * beta;
    double mean_sig = signal.mean();
    double var_sig  = (signal.array() - mean_sig).square().sum() / (n - 1.0);
    double noise_sigma = std::sqrt(var_sig / snr);

    Eigen::VectorXd y = signal;
    for (int i = 0; i < n; ++i)
        y(i) += norm(rng) * noise_sigma;

    return {std::move(X), std::move(y), std::move(support)};
}


// ==============================================================================
// Block-diagonal Toeplitz DGP (heavy-tailed)
// ==============================================================================

/**
 * @brief Heavy-tailed extension of the block-diagonal Toeplitz DGP.
 *
 * @param heavy_X     If true, rows of X follow multivariate t(df).
 * @param heavy_noise If true, noise is t(df)-distributed.
 * @param df          Degrees of freedom for t-distribution (default 3).
 */
inline DGPData dgp_block_toeplitz_heavy(int n, int p, int G, int g,
                                         double rho, double snr, unsigned seed,
                                         bool heavy_X, bool heavy_noise,
                                         double df = 3.0) {
    std::mt19937 rng(seed);
    std::normal_distribution<double>       norm(0.0, 1.0);
    std::chi_squared_distribution<double>  chi2(df);
    std::student_t_distribution<double>    t_dist(df);

    const int num_blocks = p / g;

    // Build g x g Toeplitz block Cholesky
    Eigen::MatrixXd T(g, g);
    for (int i = 0; i < g; ++i)
        for (int j = 0; j < g; ++j)
            T(i, j) = std::pow(rho, std::abs(i - j));
    Eigen::MatrixXd L = Eigen::LLT<Eigen::MatrixXd>(T).matrixL();

    Eigen::MatrixXd X(n, p);
    for (int b = 0; b < num_blocks; ++b) {
        Eigen::MatrixXd Z(n, g);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < g; ++j)
                Z(i, j) = norm(rng);

        Eigen::MatrixXd block = Z * L.transpose();

        if (heavy_X) {
            // Scale each row to produce multivariate t(df)
            for (int i = 0; i < n; ++i) {
                double u = chi2(rng);
                double scale = std::sqrt(df / u);
                block.row(i) *= scale;
            }
        }

        X.block(0, b * g, n, g) = block;
    }

    // Sparse beta
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    std::vector<std::size_t> support;
    support.reserve(static_cast<std::size_t>(G));
    for (int b = 0; b < G; ++b) {
        beta(b * g) = 1.0;
        support.push_back(static_cast<std::size_t>(b * g));
    }

    // Signal and SNR-calibrated noise
    Eigen::VectorXd signal = X * beta;
    double var_sig = (signal.array() - signal.mean()).square().sum() / (n - 1.0);
    double target_noise_var = var_sig / snr;

    Eigen::VectorXd y = signal;
    if (heavy_noise) {
        double t_var = df / (df - 2.0);
        double scale = std::sqrt(target_noise_var / t_var);
        for (int i = 0; i < n; ++i)
            y(i) += t_dist(rng) * scale;
    } else {
        double noise_sigma = std::sqrt(target_noise_var);
        for (int i = 0; i < n; ++i)
            y(i) += norm(rng) * noise_sigma;
    }

    return {std::move(X), std::move(y), std::move(support)};
}


// ==============================================================================
// Default control parameter factories
// ==============================================================================

inline TRexControlParameter make_trex_control(std::size_t K = 20) {
    TRexControlParameter ctrl;
    ctrl.K                        = K;
    ctrl.max_dummy_multiplier     = 10;
    ctrl.use_max_T_stop           = true;
    ctrl.dummy_distribution       = dummygen::Distribution::Normal();
    ctrl.lloop_strategy           = LLoopStrategy::HCONCAT;
    ctrl.tloop_stagnation_stop    = true;
    ctrl.tloop_max_stagnant_steps = 5;
    ctrl.use_memory_mapping       = false;
    return ctrl;
}

inline TRexDAControlParameter make_da_bt_control() {
    TRexDAControlParameter da;
    da.method     = DAMethod::BT;
    da.hc_linkage = hac::LinkageMethod::Average;
    return da;
}


// ==============================================================================
// Standard MC inner loop
// ==============================================================================

struct MCTrialResult {
    double fdp = 0.0;
    double tpp = 0.0;
    double L   = 0.0;
    double T   = 0.0;
};

/**
 * @brief Aggregate (FDR, TPR, avg L, avg T) for each grid point.
 */
struct GridPointResult {
    double avg_fdr = 0.0;
    double avg_tpr = 0.0;
    double avg_L   = 0.0;
    double avg_T   = 0.0;
};

using DGPFactory = std::function<DGPData(unsigned seed)>;

/**
 * @brief Run num_MC sequential trials, printing progress on one line.
 */
inline GridPointResult run_mc_trials(
    std::size_t num_MC,
    const std::string& progress_label,
    const DGPFactory& make_data,
    double tFDR,
    const TRexControlParameter& trex_ctrl,
    const TRexDAControlParameter& da_ctrl,
    unsigned base_seed_offset)
{
    double total_fdp = 0.0, total_tpp = 0.0;
    double total_L = 0.0, total_T = 0.0;

    for (std::size_t mc = 0; mc < num_MC; ++mc) {

        std::cout << "\r    " << progress_label << "  ["
                  << std::setw(3) << (mc + 1) << "/" << num_MC << "] "
                  << std::fixed << std::setprecision(1)
                  << (100.0 * static_cast<double>(mc + 1)
                            / static_cast<double>(num_MC))
                  << "%" << std::flush;

        unsigned trial_seed = base_seed_offset + static_cast<unsigned>(mc);
        auto dat = make_data(trial_seed);

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(), dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        TRexDASelector selector(X_map, y_map, tFDR, da_ctrl, trex_ctrl,
                                /*seed=*/-1, /*verbose=*/false);
        selector.select();

        auto sel_idx = selector.getSelectedIndices();
        total_fdp += rates::compute_fdp(sel_idx, dat.true_support);
        total_tpp += rates::compute_tpp(sel_idx, dat.true_support);
        total_L   += static_cast<double>(selector.getDummyMultiplierL());
        total_T   += static_cast<double>(selector.getStoppingTimeT());
    }

    std::cout << "\r    " << progress_label
              << "  done" << std::string(30, ' ') << "\n";

    const double dMC = static_cast<double>(num_MC);
    return {total_fdp / dMC, total_tpp / dMC, total_L / dMC, total_T / dMC};
}


// ==============================================================================
// Save / print grid results
// ==============================================================================

/**
 * @brief Save a grid-sweep result table to console and file.
 *
 * @param scenario_tag   Short scenario identifier (used in filename).
 * @param grid_label     Column header for the grid dimension (e.g. "SNR", "Rho").
 * @param grid_values    The swept values.
 * @param solvers        Solver list.
 * @param fdr_map        fdr_map[solver_name](grid_idx).
 * @param tpr_map        tpr_map[solver_name](grid_idx).
 * @param avg_L_map      Average L results (optional, may be empty).
 * @param avg_T_map      Average T results (optional, may be empty).
 * @param header_extra   Extra info line printed after the main header.
 * @param output_dir     Output directory (default: "simulations/EUSIPCO26/").
 */
inline void save_and_print_grid_results(
    const std::string& scenario_tag,
    const std::string& grid_label,
    const std::vector<double>& grid_values,
    std::size_t num_MC,
    const std::vector<SolverInfo>& solvers,
    const std::map<std::string, Eigen::VectorXd>& fdr_map,
    const std::map<std::string, Eigen::VectorXd>& tpr_map,
    const std::map<std::string, Eigen::VectorXd>& avg_L_map,
    const std::map<std::string, Eigen::VectorXd>& avg_T_map,
    const std::string& header_extra = "",
    const std::string& output_dir   = "simulations/EUSIPCO26/")
{
    const std::string filename = "da_trex_mc_" + scenario_tag + ".txt";
    std::ofstream out_file(output_dir + filename);

    auto print_dual = [&](const std::string& text) {
        std::cout << text;
        if (out_file.is_open()) out_file << text;
    };

    // Header
    std::stringstream hdr;
    hdr << "\n" << std::string(78, '=') << "\n"
        << "  DA-TRex MC: " << scenario_tag
        << "  (MC=" << num_MC << ")\n";
    if (!header_extra.empty())
        hdr << "  " << header_extra << "\n";
    hdr << std::string(78, '=') << "\n\n";
    print_dual(hdr.str());

    // Table dimensions
    constexpr std::size_t sw = 16, mw = 8, cw = 10;
    const auto ncols = grid_values.size();

    // Column header
    std::stringstream th;
    th << std::left  << std::setw(sw) << "Solver"
       << std::left  << std::setw(mw) << "Metric"
       << std::right << std::setw(5)  << grid_label;
    for (double v : grid_values) {
        if (v == std::floor(v) && std::abs(v) < 1e6)
            th << std::setw(cw) << static_cast<int>(v);
        else
            th << std::fixed << std::setprecision(2) << std::setw(cw) << v;
    }
    th << "\n" << std::string(sw + mw + 5 + cw * ncols, '-') << "\n";
    print_dual(th.str());

    // Row printer
    auto print_row = [&](const std::string& solver, const std::string& metric,
                         const Eigen::VectorXd& data, bool first) {
        std::stringstream row;
        row << std::left  << std::setw(sw) << (first ? solver : "")
            << std::left  << std::setw(mw) << metric
            << std::setw(5) << ""
            << std::right;
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(ncols); ++i)
            row << std::fixed << std::setprecision(4) << std::setw(cw) << data(i);
        row << "\n";
        print_dual(row.str());
    };

    for (const auto& sv : solvers) {
        const auto& nm = sv.name;
        print_row(nm, "FDR", fdr_map.at(nm), true);
        print_row(nm, "TPR", tpr_map.at(nm), false);
        if (avg_L_map.count(nm)) print_row(nm, "Avg L", avg_L_map.at(nm), false);
        if (avg_T_map.count(nm)) print_row(nm, "Avg T", avg_T_map.at(nm), false);
        print_dual("\n");
    }

    print_dual(std::string(78, '=') + "\n\n");

    if (out_file.is_open()) {
        std::cout << "[Info] Saved to: " << output_dir + filename << "\n\n";
        out_file.close();
    }
}


// ==============================================================================
// Default solver set
// ==============================================================================

inline std::vector<SolverInfo> default_solvers() {
    return {
        {SolverTypeForTRex::TLARS, "TLARS"},
        {SolverTypeForTRex::TAFS,  "TAFS",  0.3},
        {SolverTypeForTRex::TOMP,  "TOMP"},
    };
}

} // namespace da_sim

#endif /* DEMOS_TREX_SELECTOR_METHODS_DA_TREX_SIM_COMMON_HPP */
