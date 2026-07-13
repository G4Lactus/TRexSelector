// ===================================================================================
// sim_trex_sd_mc.cpp
// ===================================================================================
/**
 * @file sim_trex_sd_mc.cpp
 *
 * @brief Monte Carlo study of TRexSD: FDR control and power over independent
 *        trials.
 *
 * @details Each trial draws fresh data (fresh design, support placement, and
 *          noise) and a fresh selector seed from a master RNG, runs a
 *          single-path baseline (T = 1) and TRexSD at tFDR in {0.1, 0.2},
 *          and records FDP / TPR / |S^| / T* / v*. Reported are the trial
 *          means with standard errors: mean FDP estimates the FDR, which is
 *          the quantity the selector guarantees — single realizations do
 *          not.
 *
 *          Usage:
 *            sim_trex_sd_mc [trials] [rho_d]
 *                    DEFAULT: algorithm comparison LARS vs OMP vs
 *                    AFS(rho=0.3) under the General dummy law, over the SNR
 *                    grid {0.1, 0.2, 0.5, 0.6, 1, 2, 5}; per trial all
 *                    algorithms see the same data and selector seed (paired
 *                    comparison).
 *            sim_trex_sd_mc signal|null|snr [trials] [rho_d]
 *                    signal: n=300, |S|=10, snr=2 (detection-boundary-ish)
 *                    null:   n=300, |S|=1, snr=0.01 (y ~ noise; any
 *                            selection is counted as false unless it is the
 *                            one negligible signal)
 *                    snr:    full cross product dummy law x algorithm —
 *                            (General / Pair / PairGeometric) x (LARS /
 *                            OMP / AFS(rho=0.3)) — over the same SNR grid,
 *                            paired; one policy table per algorithm (the
 *                            LARS block matches the earlier TLARS-only
 *                            policy studies).
 *            trials: default 100 (200 for signal/null)
 *            rho_d:  dummy sparsity of the General solver, default 0.047
 *                    (k=7); 0.5 gives k=75. Pair policies are k=1 by
 *                    construction.
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

// trex includes
#include <trex_selector_methods/trex_sd/trex_sd.hpp>
#include <tsolvers/sd_tsolvers/sd_tlars_solver.hpp>

// demo-local includes
#include "linear_simulator.hpp"

// ===================================================================================

namespace {

namespace sd  = trex::tsolvers::linear_model::lars_based;
namespace tsd = trex::trex_selector_methods::trex_sd;
using trex::simulation::LinearSimulator;

// One trial's metrics: computed inside the parallel MC loop into a pre-sized
// per-trial slot (lock-free, see trex_sim_utils.hpp for the pattern), then
// merged into the Tally in deterministic trial order after the loop.
struct TrialStat {
    double fdp = 0.0, tpr = 0.0, R = 0.0, T = 0.0, v = 0.0, L = 0.0;
};

struct Tally {
    std::vector<double> fdp, tpr, R, T, v, L;

    static TrialStat eval(const std::vector<std::size_t>& sel,
                          const std::vector<std::size_t>& support,
                          double T_star = 0.0, double v_star = 0.0,
                          double L_used = 0.0) {
        int tp = 0;
        for (std::size_t j : sel)
            if (std::find(support.begin(), support.end(), j) != support.end()) tp++;
        const int fp = (int)sel.size() - tp;
        TrialStat s;
        s.fdp = sel.empty() ? 0.0 : double(fp) / double(sel.size());
        s.tpr = support.empty() ? 1.0 : double(tp) / double(support.size());
        s.R   = (double)sel.size();
        s.T   = T_star;
        s.v   = v_star;
        s.L   = L_used;
        return s;
    }

    void push(const TrialStat& s) {
        fdp.push_back(s.fdp);
        tpr.push_back(s.tpr);
        R.push_back(s.R);
        T.push_back(s.T);
        v.push_back(s.v);
        L.push_back(s.L);
    }

    static std::pair<double, double> meanSE(const std::vector<double>& x) {
        if (x.empty()) return {0.0, 0.0};
        double m = 0.0;
        for (double val : x) m += val;
        m /= double(x.size());
        double s2 = 0.0;
        for (double val : x) s2 += (val - m) * (val - m);
        s2 = x.size() > 1 ? s2 / double(x.size() - 1) : 0.0;
        return {m, std::sqrt(s2 / double(x.size()))};
    }
};

void printRow(const std::string& label, const Tally& t) {
    auto [f, fse] = Tally::meanSE(t.fdp);
    auto [q, qse] = Tally::meanSE(t.tpr);
    auto [r, rse] = Tally::meanSE(t.R);
    auto [T, Tse] = Tally::meanSE(t.T);
    (void)Tse; (void)rse;
    std::cout << std::left << std::setw(26) << label << std::right << std::fixed
              << "  FDR=" << std::setprecision(3) << f << " +- " << fse
              << "  TPR=" << q << " +- " << qse
              << "  |S^|=" << std::setprecision(1) << r
              << "  T*=" << T
              << std::defaultfloat << "\n";
}

// -----------------------------------------------------------------------------------
// SNR study: dummy law x algorithm over an SNR grid, paired trials. All nine
// combinations (General / Pair / PairGeometric) x (LARS / OMP / AFS) see the
// same data and selector seed per trial; tables are printed per algorithm
// with policy columns, so the LARS block stays comparable with the earlier
// (TLARS-only) policy studies.
// -----------------------------------------------------------------------------------
int runSnrStudy(int TRIALS, double rho_d) {
    const std::size_t n = 300, p = 10'000, sparsity = 10;
    const double afs_rho = 0.3;
    const std::vector<double> snr_grid = {0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0};
    const std::vector<double> targets = {0.1, 0.2};
    const std::vector<std::pair<tsd::SDSolverType, const char*>> policies = {
        {tsd::SDSolverType::General,       "General"},
        {tsd::SDSolverType::Pair,          "Pair"},
        {tsd::SDSolverType::PairGeometric, "PairGeometric"},
    };
    const std::vector<std::pair<tsd::SDAlgo, const char*>> algos = {
        {tsd::SDAlgo::LARS, "LARS"},
        {tsd::SDAlgo::OMP,  "OMP"},
        {tsd::SDAlgo::AFS,  "AFS(0.3)"},
    };

    std::cout << "===== TRexSD SNR x policy x algorithm Monte Carlo =====\n"
              << "trials=" << TRIALS << "  n=" << n << "  p=" << p
              << "  |S|=" << sparsity << "  K=20  L=L-loop"
              << "  rho_d(General)=" << rho_d
              << "  AFS rho=" << afs_rho
              << "  tFDR={0.1, 0.2}\n"
              << "snr grid: {0.1, 0.2, 0.5, 0.6, 1, 2, 5}\n\n";

    // tally[snr][algo][policy][target]
    std::vector<std::vector<std::vector<std::vector<Tally>>>> tally(
        snr_grid.size(),
        std::vector<std::vector<std::vector<Tally>>>(
            algos.size(),
            std::vector<std::vector<Tally>>(policies.size(),
                                            std::vector<Tally>(targets.size()))));

    std::mt19937_64 master(20260712ULL);
    std::uniform_int_distribution<uint32_t> seed_draw(1u, 2147483647u);

    const auto t_begin = std::chrono::steady_clock::now();

    for (std::size_t si = 0; si < snr_grid.size(); ++si) {
        // Pre-draw the per-trial seeds (identical master stream to the
        // serial version), then run the trials in parallel: each trial owns
        // its data and selectors, so the trial is the parallel axis and the
        // inner K-experiment loop runs single-threaded.
        std::vector<uint32_t> data_seeds(TRIALS), sel_seeds(TRIALS);
        for (int trial = 0; trial < TRIALS; ++trial) {
            data_seeds[trial] = seed_draw(master);
            sel_seeds[trial]  = seed_draw(master);
        }

        // Per-trial result slots (lock-free): stats[trial][algo][policy][target]
        std::vector<std::vector<std::vector<std::vector<TrialStat>>>> stats(
            TRIALS,
            std::vector<std::vector<std::vector<TrialStat>>>(
                algos.size(),
                std::vector<std::vector<TrialStat>>(
                    policies.size(), std::vector<TrialStat>(targets.size()))));

        #pragma omp parallel for schedule(dynamic)
        for (int trial = 0; trial < TRIALS; ++trial) {
            auto data = LinearSimulator::generate(n, p, sparsity, 1.0,
                                                  snr_grid[si], 0.0,
                                                  data_seeds[trial]);
            for (Eigen::Index j = 0; j < data.X.cols(); ++j) {
                data.X.col(j).array() -= data.X.col(j).mean();
                const double nrm = data.X.col(j).norm();
                if (nrm > 0.0) data.X.col(j) /= nrm;
            }
            data.y.array() -= data.y.mean();

            for (std::size_t ai = 0; ai < algos.size(); ++ai) {
                for (std::size_t pi = 0; pi < policies.size(); ++pi) {
                    for (std::size_t ti = 0; ti < targets.size(); ++ti) {
                        tsd::TRexSDOptions opt;
                        opt.tFDR      = targets[ti];
                        opt.K         = 20;
                        opt.L_factor  = 2;
                        opt.rho_d     = rho_d;
                        opt.solver    = policies[pi].first;
                        opt.algo      = algos[ai].first;
                        opt.rho       = afs_rho;
                        opt.calib     = tsd::CalibMode::CalibrateBoth;
                        opt.seed      = sel_seeds[trial];
                        opt.verbose   = false;
                        opt.n_threads = 1;   // trials are the parallel axis

                        tsd::TRexSD sel(opt);
                        tsd::TRexSDResult res = sel.run(data.X, data.y);

                        std::vector<std::size_t> chosen(
                            res.selected_var.data(),
                            res.selected_var.data() + res.selected_var.size());
                        stats[trial][ai][pi][ti] =
                            Tally::eval(chosen, data.support_true,
                                        (double)res.T_stop, res.v_thresh,
                                    (double)res.num_dummies);
                    }
                }
            }
        }

        // Merge in deterministic trial order
        for (int trial = 0; trial < TRIALS; ++trial)
            for (std::size_t ai = 0; ai < algos.size(); ++ai)
                for (std::size_t pi = 0; pi < policies.size(); ++pi)
                    for (std::size_t ti = 0; ti < targets.size(); ++ti)
                        tally[si][ai][pi][ti].push(stats[trial][ai][pi][ti]);

        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_begin).count();
        std::cout << "  snr=" << snr_grid[si] << " done (" << std::fixed
                  << std::setprecision(0) << secs << " s elapsed)"
                  << std::defaultfloat << "\n";
    }

    // ---- Tables: FDR and TPR per (target, algorithm), rows = snr,
    //      columns = policy ----
    auto printTable = [&](std::size_t ti, std::size_t ai, bool use_fdr) {
        std::cout << "\n===== tFDR=" << targets[ti] << " : "
                  << (use_fdr ? "FDR" : "TPR")
                  << " [" << algos[ai].second << "] (mean +- SE) =====\n";
        std::cout << std::setw(6) << "snr";
        for (const auto& pol : policies)
            std::cout << std::setw(12) << pol.second << std::setw(9) << " ";
        std::cout << "\n";
        for (std::size_t si = 0; si < snr_grid.size(); ++si) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(1)
                      << snr_grid[si];
            for (std::size_t pi = 0; pi < policies.size(); ++pi) {
                const auto& src = use_fdr ? tally[si][ai][pi][ti].fdp
                                          : tally[si][ai][pi][ti].tpr;
                auto [m, se] = Tally::meanSE(src);
                std::cout << std::setw(10) << std::setprecision(3) << m
                          << " +-" << std::setw(6) << se << "  ";
            }
            std::cout << std::defaultfloat << "\n";
        }
    };

    for (std::size_t ti = 0; ti < targets.size(); ++ti) {
        for (std::size_t ai = 0; ai < algos.size(); ++ai)
            printTable(ti, ai, /*use_fdr=*/true);
        for (std::size_t ai = 0; ai < algos.size(); ++ai)
            printTable(ti, ai, /*use_fdr=*/false);
    }

    const double total_secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_begin).count();
    std::cout << "\ntotal runtime: " << std::fixed << std::setprecision(0)
              << total_secs << " s" << std::defaultfloat << "\n";
    return 0;
}

// -----------------------------------------------------------------------------------
// Default study: LARS vs OMP vs AFS(rho=0.3) under the General dummy law,
// over an SNR grid, paired trials (same data and selector seed per trial).
// -----------------------------------------------------------------------------------
int runAlgoStudy(int TRIALS, double rho_d) {
    const std::size_t n = 300, p = 10'000, sparsity = 10;
    const double afs_rho = 0.3;
    const std::vector<double> snr_grid = {0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0};
    const std::vector<double> targets = {0.1, 0.2};

    struct Config {
        tsd::SDSolverType law;
        tsd::SDAlgo       algo;
        const char*       name;
    };
    const std::vector<Config> configs = {
        {tsd::SDSolverType::General, tsd::SDAlgo::LARS, "LARS"},
        {tsd::SDSolverType::General, tsd::SDAlgo::OMP,  "OMP"},
        {tsd::SDSolverType::General, tsd::SDAlgo::AFS,  "AFS(0.3)"},
    };

    std::cout << "===== TRexSD algorithm Monte Carlo: LARS vs OMP vs AFS =====\n"
              << "trials=" << TRIALS << "  n=" << n << "  p=" << p
              << "  |S|=" << sparsity << "  K=20  L=L-loop"
              << "  rho_d=" << rho_d
              << "  AFS rho=" << afs_rho
              << "  tFDR={0.1, 0.2}\n"
              << "snr grid: {0.1, 0.2, 0.5, 0.6, 1, 2, 5}\n\n";

    // tally[snr][config][target]
    std::vector<std::vector<std::vector<Tally>>> tally(
        snr_grid.size(),
        std::vector<std::vector<Tally>>(configs.size(),
                                        std::vector<Tally>(targets.size())));

    std::mt19937_64 master(20260712ULL);
    std::uniform_int_distribution<uint32_t> seed_draw(1u, 2147483647u);

    const auto t_begin = std::chrono::steady_clock::now();

    for (std::size_t si = 0; si < snr_grid.size(); ++si) {
        // Pre-draw the per-trial seeds (identical master stream to the
        // serial version), then run the trials in parallel: each trial owns
        // its data and selectors, so the trial is the parallel axis and the
        // inner K-experiment loop runs single-threaded.
        std::vector<uint32_t> data_seeds(TRIALS), sel_seeds(TRIALS);
        for (int trial = 0; trial < TRIALS; ++trial) {
            data_seeds[trial] = seed_draw(master);
            sel_seeds[trial]  = seed_draw(master);
        }

        // Per-trial result slots (lock-free): stats[trial][config][target]
        std::vector<std::vector<std::vector<TrialStat>>> stats(
            TRIALS,
            std::vector<std::vector<TrialStat>>(
                configs.size(), std::vector<TrialStat>(targets.size())));

        #pragma omp parallel for schedule(dynamic)
        for (int trial = 0; trial < TRIALS; ++trial) {
            auto data = LinearSimulator::generate(n, p, sparsity, 1.0,
                                                  snr_grid[si], 0.0,
                                                  data_seeds[trial]);
            for (Eigen::Index j = 0; j < data.X.cols(); ++j) {
                data.X.col(j).array() -= data.X.col(j).mean();
                const double nrm = data.X.col(j).norm();
                if (nrm > 0.0) data.X.col(j) /= nrm;
            }
            data.y.array() -= data.y.mean();

            for (std::size_t ci = 0; ci < configs.size(); ++ci) {
                for (std::size_t ti = 0; ti < targets.size(); ++ti) {
                    tsd::TRexSDOptions opt;
                    opt.tFDR      = targets[ti];
                    opt.K         = 20;
                    opt.L_factor  = 2;
                    opt.rho_d     = rho_d;
                    opt.solver    = configs[ci].law;
                    opt.algo      = configs[ci].algo;
                    opt.rho       = afs_rho;
                    opt.calib     = tsd::CalibMode::CalibrateBoth;
                    opt.seed      = sel_seeds[trial];
                    opt.verbose   = false;
                    opt.n_threads = 1;   // trials are the parallel axis

                    tsd::TRexSD sel(opt);
                    tsd::TRexSDResult res = sel.run(data.X, data.y);

                    std::vector<std::size_t> chosen(
                        res.selected_var.data(),
                        res.selected_var.data() + res.selected_var.size());
                    stats[trial][ci][ti] =
                        Tally::eval(chosen, data.support_true,
                                    (double)res.T_stop, res.v_thresh,
                                    (double)res.num_dummies);
                }
            }
        }

        // Merge in deterministic trial order
        for (int trial = 0; trial < TRIALS; ++trial)
            for (std::size_t ci = 0; ci < configs.size(); ++ci)
                for (std::size_t ti = 0; ti < targets.size(); ++ti)
                    tally[si][ci][ti].push(stats[trial][ci][ti]);

        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_begin).count();
        std::cout << "  snr=" << snr_grid[si] << " done (" << std::fixed
                  << std::setprecision(0) << secs << " s elapsed)"
                  << std::defaultfloat << "\n";
    }

    // ---- Tables: FDR and TPR per target, rows = snr, columns = config ----
    auto printTable = [&](std::size_t ti, bool use_fdr) {
        std::cout << "\n===== tFDR=" << targets[ti] << " : "
                  << (use_fdr ? "FDR" : "TPR") << " (mean +- SE) =====\n";
        std::cout << std::setw(6) << "snr";
        for (const auto& cfg : configs)
            std::cout << std::setw(12) << cfg.name << std::setw(9) << " ";
        std::cout << "\n";
        for (std::size_t si = 0; si < snr_grid.size(); ++si) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(1)
                      << snr_grid[si];
            for (std::size_t ci = 0; ci < configs.size(); ++ci) {
                const auto& src = use_fdr ? tally[si][ci][ti].fdp
                                          : tally[si][ci][ti].tpr;
                auto [m, se] = Tally::meanSE(src);
                std::cout << std::setw(10) << std::setprecision(3) << m
                          << " +-" << std::setw(6) << se << "  ";
            }
            std::cout << std::defaultfloat << "\n";
        }
    };

    for (std::size_t ti = 0; ti < targets.size(); ++ti) {
        printTable(ti, /*use_fdr=*/true);
        printTable(ti, /*use_fdr=*/false);
    }

    // Mean stopping point T* per config at the higher target (diagnostic:
    // the greedy solvers should calibrate to much smaller T than LARS).
    std::cout << "\n===== tFDR=" << targets.back() << " : mean T* =====\n";
    std::cout << std::setw(6) << "snr";
    for (const auto& cfg : configs) std::cout << std::setw(11) << cfg.name;
    std::cout << "\n";
    for (std::size_t si = 0; si < snr_grid.size(); ++si) {
        std::cout << std::setw(6) << std::fixed << std::setprecision(1)
                  << snr_grid[si];
        for (std::size_t ci = 0; ci < configs.size(); ++ci) {
            auto [m, se] = Tally::meanSE(tally[si][ci].back().T);
            (void)se;
            std::cout << std::setw(11) << std::setprecision(1) << m;
        }
        std::cout << std::defaultfloat << "\n";
    }

    // Mean L (in units of p) the L-loop accepted (diagnostic: the classic
    // T-Rex procedure inflates L only when T=1 stopping overdraws).
    std::cout << "\n===== tFDR=" << targets.back() << " : mean L/p =====\n";
    std::cout << std::setw(6) << "snr";
    for (const auto& cfg : configs) std::cout << std::setw(11) << cfg.name;
    std::cout << "\n";
    for (std::size_t si = 0; si < snr_grid.size(); ++si) {
        std::cout << std::setw(6) << std::fixed << std::setprecision(1)
                  << snr_grid[si];
        for (std::size_t ci = 0; ci < configs.size(); ++ci) {
            auto [m, se] = Tally::meanSE(tally[si][ci].back().L);
            (void)se;
            std::cout << std::setw(11) << std::setprecision(1) << m / double(p);
        }
        std::cout << std::defaultfloat << "\n";
    }

    const double total_secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_begin).count();
    std::cout << "\ntotal runtime: " << std::fixed << std::setprecision(0)
              << total_secs << " s" << std::defaultfloat << "\n";
    return 0;
}

}  // namespace

// ===================================================================================
// Main
// ===================================================================================

int main(int argc, char** argv) {
    // First argument: a named legacy mode (signal / null / snr), or directly
    // the trial count for the default LARS-vs-OMP-vs-AFS comparison.
    std::string mode;
    int argi = 1;
    if (argc > 1) {
        const std::string first = argv[1];
        if (first == "signal" || first == "null" || first == "snr") {
            mode = first;
            argi = 2;
        }
    }
    const int TRIALS = (argc > argi) ? std::stoi(argv[argi])
                                     : (mode.empty() ? 100 : 200);
    const double rho_d = (argc > argi + 1) ? std::stod(argv[argi + 1]) : 0.047;

    omp_set_num_threads(6);
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";
    std::cout << "Running with " << omp_get_max_threads() << " threads ("
              << trex::utils::openmp::status_string() << ")\n\n";

    if (mode.empty())  return runAlgoStudy(TRIALS, rho_d);
    if (mode == "snr") return runSnrStudy(TRIALS, rho_d);

    const std::size_t n = 300, p = 10'000;
    const bool is_null = (mode == "null");
    const std::size_t sparsity = is_null ? 1 : 10;
    const double snr = is_null ? 0.01 : 2.0;
    const std::vector<double> targets = {0.1, 0.2};

    std::cout << "===== TRexSD Monte Carlo =====\n"
              << "mode=" << mode << "  trials=" << TRIALS
              << "  n=" << n << "  p=" << p << "  |S|=" << sparsity
              << "  snr=" << snr << "  rho_d=" << rho_d
              << "  K=20  L=L-loop  tFDR={0.1, 0.2}\n\n";

    std::mt19937_64 master(20260712ULL);
    std::uniform_int_distribution<uint32_t> seed_draw(1u, 2147483647u);

    Tally single_path;
    std::vector<Tally> selector(targets.size());

    const auto t_begin = std::chrono::steady_clock::now();

    // Pre-draw the per-trial seeds (identical master stream to the serial
    // version), then run the trials in parallel: each trial owns its data
    // and selectors, so the trial is the parallel axis and the inner
    // K-experiment loop runs single-threaded (pattern: trex_sim_utils.hpp).
    std::vector<uint32_t> data_seeds(TRIALS), sel_seeds(TRIALS);
    for (int trial = 0; trial < TRIALS; ++trial) {
        data_seeds[trial] = seed_draw(master);
        sel_seeds[trial]  = seed_draw(master);
    }

    // Per-trial result slots (lock-free)
    std::vector<TrialStat> sp_stats(TRIALS);
    std::vector<std::vector<TrialStat>> sel_stats(
        TRIALS, std::vector<TrialStat>(targets.size()));

    std::cout << "  Running " << TRIALS << " MC trials ...\n" << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int trial = 0; trial < TRIALS; ++trial) {
        auto data = LinearSimulator::generate(n, p, sparsity, 1.0, snr, 0.0,
                                              data_seeds[trial]);
        for (Eigen::Index j = 0; j < data.X.cols(); ++j) {
            data.X.col(j).array() -= data.X.col(j).mean();
            const double nrm = data.X.col(j).norm();
            if (nrm > 0.0) data.X.col(j) /= nrm;
        }
        data.y.array() -= data.y.mean();

        // --- single-path baseline, T = 1 ---
        {
            Eigen::Map<Eigen::MatrixXd> Xm(data.X.data(), data.X.rows(), data.X.cols());
            Eigen::Map<Eigen::VectorXd> ym(data.y.data(), data.y.size());
            sd::SD_TLARS_Solver solver(Xm, ym, rho_d, 2 * p, 1, true, sel_seeds[trial]);
            solver.executeStep(1, true);
            sp_stats[trial] = Tally::eval(solver.getSelectedOriginals(),
                                          data.support_true);
        }

        // --- TRexSD at each target ---
        for (std::size_t ti = 0; ti < targets.size(); ++ti) {
            tsd::TRexSDOptions opt;
            opt.tFDR      = targets[ti];
            opt.K         = 20;
            opt.L_factor  = 2;
            opt.rho_d     = rho_d;      // fixed across trials (design parameter)
            opt.solver    = tsd::SDSolverType::PairGeometric;
            opt.calib     = tsd::CalibMode::CalibrateBoth;
            opt.seed      = sel_seeds[trial];
            opt.verbose   = false;
            opt.n_threads = 1;          // trials are the parallel axis

            tsd::TRexSD sel(opt);
            tsd::TRexSDResult res = sel.run(data.X, data.y);

            std::vector<std::size_t> chosen(res.selected_var.data(),
                                            res.selected_var.data() + res.selected_var.size());
            sel_stats[trial][ti] = Tally::eval(chosen, data.support_true,
                                               (double)res.T_stop, res.v_thresh,
                                    (double)res.num_dummies);
        }
    }

    // Merge in deterministic trial order
    for (int trial = 0; trial < TRIALS; ++trial) {
        single_path.push(sp_stats[trial]);
        for (std::size_t ti = 0; ti < targets.size(); ++ti)
            selector[ti].push(sel_stats[trial][ti]);
    }

    const double total_secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_begin).count();

    std::cout << "\n===== results over " << TRIALS << " trials (mean +- SE) =====\n";
    printRow("single path, T=1", single_path);
    for (std::size_t ti = 0; ti < targets.size(); ++ti) {
        printRow("TRexSD, tFDR=" + std::to_string(targets[ti]).substr(0, 4),
                 selector[ti]);
    }
    std::cout << "\ntotal runtime: " << std::fixed << std::setprecision(0)
              << total_secs << " s" << std::defaultfloat << "\n";
    return 0;
}
