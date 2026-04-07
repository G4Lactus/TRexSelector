// ==============================================================================
// sim_trx_01_da_trex_mc.cpp
// ==============================================================================
/**
 * @file sim_trx_01_da_trex_mc.cpp
 *
 * @brief Monte Carlo simulation for the DA-TRex Selector.
 *
 * @details
 *  Evaluates all five DA methods across an SNR grid with 200 MC trials each.
 *  Three solvers are compared: TLARS, TAFS, and TOMP.
 *
 *  Scenarios:
 *    1. AR(1)         — Toeplitz covariance (ρ = 0.7)
 *    2. Equi          — compound symmetry (ρ = 0.5)
 *    3. NN            — banded MA(3) covariance (ρ = 0.7)
 *    4. BT            — 10 blocks of 50 (ρ_w = 0.8, ρ_b = 0.1)
 *    5. Prior groups  — 3-level hierarchy (50/10/2 groups)
 *
 *  Settings: n = 300, p = 1000, s = 10, K = 20, tFDR = 0.2.
 *  Results are saved to simulations/EUSIPCO26/.
 */
// ==============================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// T-Rex DA includes
#include <trex_selector_methods/trex_da/trex_da.hpp>

// utils includes
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>
#include <utils/openmp/utils_openmp.hpp>


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
using trex::trex_selector_methods::utils::solver_dispatch::SolverHyperparameters;

namespace hac = trex::ml_methods::clustering::hierarchical::agglomerative;
namespace dummygen = trex::utils::datageneration::dummygen;


// ==============================================================================
// Simulation configuration
// ==============================================================================

struct SimConfig {
    Eigen::Index n   = 300;     // observations
    Eigen::Index p   = 1000;    // predictors
    Eigen::Index s   = 10;      // active predictors
    double  amplitude = 3.0;    // signal coefficient (before SNR scaling)
    double  tFDR     = 0.2;     // target FDR
    std::size_t K    = 20;      // random experiments
    std::size_t num_MC = 200;   // Monte Carlo trials
    int base_seed    = 2026;    // base RNG seed
};

struct SolverInfo {
    SolverTypeForTRex type;
    std::string       name;
    double            rho_afs = 0.0;
};


// ==============================================================================
// DGP helpers (inline data generators)
// ==============================================================================

namespace {

struct DGPData {
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    std::vector<std::size_t> true_support;
};


/**
 * @brief Build sparse beta (first s active) with SNR-controlled amplitude.
 *
 * We set beta_j = amplitude for the first s predictors.  The caller
 * controls the effective SNR via SyntheticData's noise scaling or
 * by computing noise_sigma = sqrt(var(X*beta) / snr).
 */
Eigen::VectorXd make_beta(Eigen::Index p, Eigen::Index s, double amplitude) {
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    for (Eigen::Index j = 0; j < s; ++j) {
        beta(j) = amplitude;
    }
    return beta;
}


/**
 * @brief Generate y = X * beta + noise, where noise std is calibrated to target SNR.
 */
Eigen::VectorXd make_response(const Eigen::MatrixXd& X,
                               const Eigen::VectorXd& beta,
                               double snr, std::mt19937& rng) {
    Eigen::VectorXd signal = X * beta;
    const Eigen::Index n = X.rows();

    // Empirical signal variance
    double mean_sig = signal.mean();
    double var_sig  = (signal.array() - mean_sig).square().sum()
                      / static_cast<double>(n - 1);
    double noise_sigma = std::sqrt(var_sig / snr);

    std::normal_distribution<double> dist(0.0, noise_sigma);
    for (Eigen::Index i = 0; i < n; ++i) {
        signal(i) += dist(rng);
    }
    return signal;
}

std::vector<std::size_t> first_s_support(Eigen::Index s) {
    std::vector<std::size_t> sup(static_cast<std::size_t>(s));
    std::iota(sup.begin(), sup.end(), 0);
    return sup;
}


// ── DGP 1: AR(1) ─────────────────────────────────────────────────────────────

DGPData dgp_ar1(Eigen::Index n, Eigen::Index p, Eigen::Index s,
                double amplitude, double rho, double snr, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    Eigen::MatrixXd X(n, p);
    const double scale = std::sqrt(1.0 - rho * rho);
    for (Eigen::Index i = 0; i < n; ++i) {
        X(i, 0) = norm(rng);
        for (Eigen::Index j = 1; j < p; ++j) {
            X(i, j) = rho * X(i, j - 1) + scale * norm(rng);
        }
    }

    auto beta = make_beta(p, s, amplitude);
    auto y    = make_response(X, beta, snr, rng);
    return {std::move(X), std::move(y), first_s_support(s)};
}


// ── DGP 2: Equicorrelation ───────────────────────────────────────────────────

DGPData dgp_equi(Eigen::Index n, Eigen::Index p, Eigen::Index s,
                  double amplitude, double rho, double snr, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    const double load_f   = std::sqrt(rho);
    const double load_eta = std::sqrt(1.0 - rho);

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        const double f_i = norm(rng);
        for (Eigen::Index j = 0; j < p; ++j) {
            X(i, j) = load_f * f_i + load_eta * norm(rng);
        }
    }

    auto beta = make_beta(p, s, amplitude);
    auto y    = make_response(X, beta, snr, rng);
    return {std::move(X), std::move(y), first_s_support(s)};
}


// ── DGP 3: Nearest Neighbours (MA(κ)) ───────────────────────────────────────

DGPData dgp_nn(Eigen::Index n, Eigen::Index p, Eigen::Index s,
               double amplitude, int kappa, double rho, double snr, unsigned seed) {
    std::mt19937 rng(seed);
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

    const Eigen::Index ext_cols = p + kappa;
    Eigen::MatrixXd eta(n, ext_cols);
    for (Eigen::Index i = 0; i < n; ++i)
        for (Eigen::Index j = 0; j < ext_cols; ++j)
            eta(i, j) = norm(rng);

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index j = 0; j < p; ++j) {
        X.col(j).setZero();
        for (int l = 0; l <= kappa; ++l)
            X.col(j) += theta[static_cast<std::size_t>(l)] * eta.col(j + l);
    }

    auto beta = make_beta(p, s, amplitude);
    auto y    = make_response(X, beta, snr, rng);
    return {std::move(X), std::move(y), first_s_support(s)};
}


// ── DGP 4: Binary Tree (hierarchical blocks) ────────────────────────────────

DGPData dgp_bt(Eigen::Index n, Eigen::Index p, Eigen::Index s,
               double amplitude, int n_blocks, double rho_within,
               double rho_between, double snr, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    const Eigen::Index block_size = p / n_blocks;
    const double a = std::sqrt(rho_between);
    const double b = std::sqrt(rho_within - rho_between);
    const double cc = std::sqrt(1.0 - rho_within);

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        const double f0 = norm(rng);
        std::vector<double> f_block(static_cast<std::size_t>(n_blocks));
        for (int m = 0; m < n_blocks; ++m)
            f_block[static_cast<std::size_t>(m)] = norm(rng);

        for (Eigen::Index j = 0; j < p; ++j) {
            const int bid = static_cast<int>(j / block_size);
            X(i, j) = a * f0 + b * f_block[static_cast<std::size_t>(bid)] + cc * norm(rng);
        }
    }

    auto beta = make_beta(p, s, amplitude);
    auto y    = make_response(X, beta, snr, rng);
    return {std::move(X), std::move(y), first_s_support(s)};
}


// ── DGP 5: Prior Groups (multi-level latent factor) ──────────────────────────

DGPData dgp_groups(Eigen::Index n, Eigen::Index p, Eigen::Index s,
                    double amplitude,
                    const std::vector<std::vector<Eigen::Index>>& groups,
                    const std::vector<double>& rho_levels,
                    double snr, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    const auto L = groups.size();
    std::vector<double> rho_ext(rho_levels.begin(), rho_levels.end());
    rho_ext.push_back(0.0);

    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(n, p);

    for (std::size_t x = 0; x < L; ++x) {
        const double loading = std::sqrt(rho_ext[x] - rho_ext[x + 1]);
        std::vector<Eigen::Index> unique_ids(groups[x].begin(), groups[x].end());
        std::sort(unique_ids.begin(), unique_ids.end());
        unique_ids.erase(std::unique(unique_ids.begin(), unique_ids.end()), unique_ids.end());

        for (auto gid : unique_ids) {
            Eigen::VectorXd f(n);
            for (Eigen::Index i = 0; i < n; ++i) f(i) = norm(rng);
            for (Eigen::Index j = 0; j < p; ++j) {
                if (groups[x][static_cast<std::size_t>(j)] == gid)
                    X.col(j) += loading * f;
            }
        }
    }

    const double eta_scale = std::sqrt(1.0 - rho_levels[0]);
    for (Eigen::Index i = 0; i < n; ++i)
        for (Eigen::Index j = 0; j < p; ++j)
            X(i, j) += eta_scale * norm(rng);

    auto beta = make_beta(p, s, amplitude);
    auto y    = make_response(X, beta, snr, rng);
    return {std::move(X), std::move(y), first_s_support(s)};
}

} // anonymous namespace


// ==============================================================================
// Result I/O
// ==============================================================================

void save_and_print_results(
    const std::string& scenario_tag,
    const SimConfig& cfg,
    const std::vector<double>& snr_values,
    const std::vector<SolverInfo>& solvers,
    const std::map<std::string, Eigen::VectorXd>& fdr_map,
    const std::map<std::string, Eigen::VectorXd>& tpr_map,
    const std::map<std::string, Eigen::VectorXd>& avg_L_map,
    const std::map<std::string, Eigen::VectorXd>& avg_T_map)
{
    const std::string folder   = "simulations/EUSIPCO26/";
    const std::string filename = "da_trex_mc_" + scenario_tag
                                 + "_n" + std::to_string(cfg.n)
                                 + "_p" + std::to_string(cfg.p)
                                 + "_s" + std::to_string(cfg.s)
                                 + ".txt";

    std::ofstream out_file(folder + filename);

    auto print_dual = [&](const std::string& text) {
        std::cout << text;
        if (out_file.is_open()) out_file << text;
    };

    // Header
    std::stringstream hdr;
    hdr << "\n"
        << std::string(78, '=') << "\n"
        << "  DA-TRex MC Results: " << scenario_tag
        << "  (MC=" << cfg.num_MC << ", n=" << cfg.n
        << ", p=" << cfg.p << ", s=" << cfg.s
        << ", tFDR=" << cfg.tFDR << ")\n"
        << std::string(78, '=') << "\n\n";
    print_dual(hdr.str());

    // Table header
    constexpr std::size_t sw = 16;   // solver column width
    constexpr std::size_t mw = 8;    // metric column width
    constexpr std::size_t cw = 10;   // data column width

    std::stringstream th;
    th << std::left  << std::setw(sw) << "Solver"
       << std::left  << std::setw(mw) << "Metric"
       << std::right << std::setw(5)  << "SNR";
    for (double snr : snr_values)
        th << std::fixed << std::setprecision(1) << std::setw(cw) << snr;
    th << "\n" << std::string(sw + mw + 5 + cw * snr_values.size(), '-') << "\n";
    print_dual(th.str());

    // Data rows
    auto print_row = [&](const std::string& solver, const std::string& metric,
                         const Eigen::VectorXd& data, bool first) {
        std::stringstream row;
        row << std::left  << std::setw(sw) << (first ? solver : "")
            << std::left  << std::setw(mw) << metric
            << std::setw(5) << ""
            << std::right;
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(snr_values.size()); ++i)
            row << std::fixed << std::setprecision(4) << std::setw(cw) << data(i);
        row << "\n";
        print_dual(row.str());
    };

    for (const auto& sv : solvers) {
        const std::string& nm = sv.name;
        print_row(nm, "FDR", fdr_map.at(nm), true);
        print_row(nm, "TPR", tpr_map.at(nm), false);
        if (avg_L_map.count(nm)) print_row(nm, "Avg L", avg_L_map.at(nm), false);
        if (avg_T_map.count(nm)) print_row(nm, "Avg T", avg_T_map.at(nm), false);
        print_dual("\n");
    }

    print_dual(std::string(78, '=') + "\n\n");

    if (out_file.is_open()) {
        std::cout << "[Info] Results saved to: " << folder + filename << "\n\n";
        out_file.close();
    }
}


// ==============================================================================
// Generic MC runner (one scenario)
// ==============================================================================

/**
 * @brief DGP factory: generates data for a single MC trial.
 */
using DGPFactory = std::function<DGPData(const SimConfig& cfg, double snr, unsigned seed)>;

/**
 * @brief DA control factory: builds the DA control parameters for a scenario.
 */
using DAControlFactory = std::function<TRexDAControlParameter()>;


void run_scenario(
    const std::string& scenario_tag,
    const SimConfig& cfg,
    const std::vector<double>& snr_values,
    const std::vector<SolverInfo>& solvers,
    const DGPFactory& make_data,
    const DAControlFactory& make_da_ctrl)
{
    cdianostics::print_section_header("Scenario: " + scenario_tag);

    std::map<std::string, Eigen::VectorXd> fdr_map, tpr_map, avg_L_map, avg_T_map;
    for (const auto& sv : solvers) {
        const auto nsnr = static_cast<Eigen::Index>(snr_values.size());
        fdr_map[sv.name]   = Eigen::VectorXd::Zero(nsnr);
        tpr_map[sv.name]   = Eigen::VectorXd::Zero(nsnr);
        avg_L_map[sv.name] = Eigen::VectorXd::Zero(nsnr);
        avg_T_map[sv.name] = Eigen::VectorXd::Zero(nsnr);
    }

    // Base TRex control (shared, solver_type overridden per solver)
    TRexControlParameter trex_ctrl;
    trex_ctrl.K                        = cfg.K;
    trex_ctrl.max_dummy_multiplier     = 10;
    trex_ctrl.use_max_T_stop           = true;
    trex_ctrl.dummy_distribution       = dummygen::Distribution::Normal();
    trex_ctrl.lloop_strategy           = LLoopStrategy::HCONCAT;
    trex_ctrl.tloop_stagnation_stop    = true;
    trex_ctrl.tloop_max_stagnant_steps = 5;
    trex_ctrl.use_memory_mapping       = false;

    // ── Solver loop ──────────────────────────────────────────────────────────

    for (const auto& current_solver : solvers) {

        std::cout << "\n  Solver: " << current_solver.name << "\n";

        trex_ctrl.solver_type = current_solver.type;
        trex_ctrl.solver_params.rho_afs = current_solver.rho_afs;

        // ── SNR loop ─────────────────────────────────────────────────────────

        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
            const double snr = snr_values[snr_idx];

            double total_fdp = 0.0;
            double total_tpp = 0.0;
            double total_L   = 0.0;
            double total_T   = 0.0;

            std::cout << "    SNR = " << std::fixed << std::setprecision(1) << snr << "  ";

            // ── Monte Carlo loop ─────────────────────────────────────────────

            for (std::size_t mc = 0; mc < cfg.num_MC; ++mc) {

                std::cout << "\r    SNR = " << std::fixed << std::setprecision(1)
                          << snr << "  ["
                          << std::setw(3) << (mc + 1) << "/" << cfg.num_MC << "] "
                          << std::fixed << std::setprecision(1)
                          << (100.0 * static_cast<double>(mc + 1)
                                    / static_cast<double>(cfg.num_MC))
                          << "%" << std::flush;

                // Unique seed per (snr, mc)
                const unsigned trial_seed =
                    static_cast<unsigned>(cfg.base_seed)
                    + static_cast<unsigned>(snr_idx) * 10000u
                    + static_cast<unsigned>(mc);

                // Generate data
                auto dat = make_data(cfg, snr, trial_seed);

                // Maps
                Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(), cfg.n, cfg.p);
                Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), cfg.n);

                // DA control
                auto da_ctrl = make_da_ctrl();

                // Create DA-TRex instance
                TRexDASelector selector(
                    X_map, y_map, cfg.tFDR,
                    da_ctrl, trex_ctrl,
                    /*seed=*/-1, /*verbose=*/false);

                selector.select();

                // Evaluate
                auto sel_idx = selector.getSelectedIndices();
                total_fdp += rates::compute_fdp(sel_idx, dat.true_support);
                total_tpp += rates::compute_tpp(sel_idx, dat.true_support);
                total_L   += static_cast<double>(selector.getDummyMultiplierL());
                total_T   += static_cast<double>(selector.getStoppingTimeT());
            }

            const double dMC = static_cast<double>(cfg.num_MC);
            const auto idx   = static_cast<Eigen::Index>(snr_idx);
            fdr_map[current_solver.name](idx)   = total_fdp / dMC;
            tpr_map[current_solver.name](idx)   = total_tpp / dMC;
            avg_L_map[current_solver.name](idx) = total_L / dMC;
            avg_T_map[current_solver.name](idx) = total_T / dMC;

            std::cout << "\r    SNR = " << std::fixed << std::setprecision(1)
                      << snr << "  done" << std::string(30, ' ') << "\n";
        }
    }

    // Save
    save_and_print_results(scenario_tag, cfg, snr_values, solvers,
                           fdr_map, tpr_map, avg_L_map, avg_T_map);
}


// ==============================================================================
// Main
// ==============================================================================

int main() {

    std::cout.setf(std::ios::unitbuf);

    // ── Configuration ────────────────────────────────────────────────────────

    SimConfig cfg;
    cfg.n      = 300;
    cfg.p      = 1000;
    cfg.s      = 10;
    cfg.amplitude = 3.0;
    cfg.tFDR   = 0.2;
    cfg.K      = 20;
    cfg.num_MC = 200;
    cfg.base_seed = 2026;

    const std::vector<double> snr_values = {
        0.1, 0.2, 0.5, 1.0, 2.0, 5.0
    };

    const std::vector<SolverInfo> solvers = {
        {SolverTypeForTRex::TLARS, "TLARS"},
        {SolverTypeForTRex::TAFS,  "TAFS",  0.3},
        {SolverTypeForTRex::TOMP,  "TOMP"},
    };

    cdianostics::print_section_header(
        "DA-TRex Monte Carlo Simulation  |  MC = "
        + std::to_string(cfg.num_MC)
        + "  n = " + std::to_string(cfg.n)
        + "  p = " + std::to_string(cfg.p)
        + "  s = " + std::to_string(cfg.s));


    // ── Prior groups structure (for scenario 5) ──────────────────────────────

    std::vector<std::vector<Eigen::Index>> groups_demo(3);
    for (Eigen::Index j = 0; j < cfg.p; ++j) {
        groups_demo[0].push_back(j / 10);    // level 1: 100 groups of 10
        groups_demo[1].push_back(j / 50);    // level 2:  20 groups of 50
        groups_demo[2].push_back(j / 250);   // level 3:   4 groups of 250
    }
    std::vector<double> rho_levels = {0.80, 0.50, 0.20};


    // ══════════════════════════════════════════════════════════════════════════
    //  Scenario 1: AR(1)  — ρ = 0.7
    // ══════════════════════════════════════════════════════════════════════════

    run_scenario(
        "AR1",
        cfg, snr_values, solvers,
        // DGP factory
        [](const SimConfig& c, double snr, unsigned seed) -> DGPData {
            return dgp_ar1(c.n, c.p, c.s, c.amplitude, 0.7, snr, seed);
        },
        // DA control factory
        []() -> TRexDAControlParameter {
            TRexDAControlParameter da;
            da.method = DAMethod::AR1;
            return da;
        }
    );


    // ══════════════════════════════════════════════════════════════════════════
    //  Scenario 2: Equicorrelation  — ρ = 0.5
    // ══════════════════════════════════════════════════════════════════════════

    run_scenario(
        "EQUI",
        cfg, snr_values, solvers,
        [](const SimConfig& c, double snr, unsigned seed) -> DGPData {
            return dgp_equi(c.n, c.p, c.s, c.amplitude, 0.5, snr, seed);
        },
        []() -> TRexDAControlParameter {
            TRexDAControlParameter da;
            da.method = DAMethod::EQUI;
            return da;
        }
    );


    // ══════════════════════════════════════════════════════════════════════════
    //  Scenario 3: NN  — κ = 3, ρ = 0.7
    // ══════════════════════════════════════════════════════════════════════════

    run_scenario(
        "NN",
        cfg, snr_values, solvers,
        [](const SimConfig& c, double snr, unsigned seed) -> DGPData {
            return dgp_nn(c.n, c.p, c.s, c.amplitude, 3, 0.7, snr, seed);
        },
        []() -> TRexDAControlParameter {
            TRexDAControlParameter da;
            da.method = DAMethod::NN;
            return da;
        }
    );


    // ══════════════════════════════════════════════════════════════════════════
    //  Scenario 4: BT  — 10 blocks of 100, ρ_w = 0.8, ρ_b = 0.1, avg linkage
    // ══════════════════════════════════════════════════════════════════════════

    run_scenario(
        "BT",
        cfg, snr_values, solvers,
        [](const SimConfig& c, double snr, unsigned seed) -> DGPData {
            return dgp_bt(c.n, c.p, c.s, c.amplitude, 10, 0.8, 0.1, snr, seed);
        },
        []() -> TRexDAControlParameter {
            TRexDAControlParameter da;
            da.method     = DAMethod::BT;
            da.hc_linkage = hac::LinkageMethod::Average;
            return da;
        }
    );


    // ══════════════════════════════════════════════════════════════════════════
    //  Scenario 5: Prior Groups  — 3-level hierarchy
    // ══════════════════════════════════════════════════════════════════════════

    run_scenario(
        "GROUPS",
        cfg, snr_values, solvers,
        [&groups_demo, &rho_levels](const SimConfig& c, double snr, unsigned seed) -> DGPData {
            return dgp_groups(c.n, c.p, c.s, c.amplitude,
                              groups_demo, rho_levels, snr, seed);
        },
        [&groups_demo, &rho_levels]() -> TRexDAControlParameter {
            TRexDAControlParameter da;
            da.prior_groups    = groups_demo;
            da.rho_grid_labels = rho_levels;
            return da;
        }
    );


    std::cout << "\nAll scenarios complete.\n";
    return 0;
}
