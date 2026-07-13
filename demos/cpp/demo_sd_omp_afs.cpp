// ===================================================================================
// demo_sd_omp_afs.cpp
// ===================================================================================
/**
 * @file demo_sd_omp_afs.cpp
 *
 * @brief Demo 3: greedy SD solvers — OMP and Adaptive Forward Stepwise.
 *
 * @details Compares, on the same data and dummy law, the three SD selection
 *          algorithms:
 *
 *            SD_TLARS:  LARS equiangular path (reference).
 *            SD_TOMP:   greedy argmax + full OLS refit per step. The refit
 *                       re-orthogonalizes the residual, so pool correlations
 *                       are recomputed from r via O(k) index sums.
 *            SD_TAFS:   adaptive forward stepwise, mu <- (1-rho)mu + rho*OLS;
 *                       active features may be re-selected. rho = 1 is OMP,
 *                       rho -> 0 approaches LARS.
 *
 *          Scenario A: algorithm comparison at fixed k (T-sweep tables and
 *                      colored entry paths; green = TP, red = FP, blue =
 *                      dummy).
 *          Scenario B: pair twins at k = 1 — SD2_TOMP / SD2_TAFS in both
 *                      generation policies (OnDemand races the identical
 *                      dummy stream as the general solver at k = 1 and must
 *                      reproduce its path; Geometric samples the exact pair
 *                      null).
 *          Scenario C: large p with auto-calibrated k and L (rho_d = 0,
 *                      L_max = 0), timing all three algorithms.
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// tsolvers includes
#include <tsolvers/sd_tsolvers/sd_tlars_solver.hpp>
#include <tsolvers/sd_tsolvers/sd_tomp_solver.hpp>
#include <tsolvers/sd_tsolvers/sd_tafs_solver.hpp>
#include <tsolvers/sd_tsolvers/sd2_tomp_solver.hpp>
#include <tsolvers/sd_tsolvers/sd2_tafs_solver.hpp>

// demo-local includes
#include "linear_simulator.hpp"

// ===================================================================================

namespace {

namespace lars = trex::tsolvers::linear_model::lars_based;
namespace omp  = trex::tsolvers::linear_model::omp_based;
namespace afs  = trex::tsolvers::linear_model::afs_based;
using trex::tsolvers::SD2GenPolicy;
using trex::simulation::LinearSimulator;

constexpr double kAfsRho = 0.3;

// ANSI colors for the entry sequence
constexpr const char* kGreen = "\033[32m";
constexpr const char* kRed   = "\033[31m";
constexpr const char* kBlue  = "\033[34m";
constexpr const char* kReset = "\033[0m";

bool inSupport(std::size_t j, const std::vector<std::size_t>& support) {
    return std::find(support.begin(), support.end(), j) != support.end();
}

void preprocess(Eigen::MatrixXd& X, Eigen::VectorXd& y) {
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
        X.col(j).array() -= X.col(j).mean();
        const double nrm = X.col(j).norm();
        if (nrm > 0.0) X.col(j) /= nrm;
    }
    y.array() -= y.mean();
}

// -----------------------------------------------------------------------------------
// T-sweep of one solver: quality table rows + colored path + solve time
// -----------------------------------------------------------------------------------
struct SweepRow {
    std::size_t T, steps, actives, reals, dummies;
    int tp, fp;
    double fdp, tpr, lambda;
};

struct SweepResult {
    std::vector<SweepRow> rows;
    std::vector<std::size_t> actives;
    std::vector<std::vector<std::size_t>> actives_per_T;  // path snapshot per T
    std::size_t generated = 0;
    double millis = 0.0;
};

template <typename Solver>
SweepResult sweep(Solver& solver, const std::vector<std::size_t>& T_grid,
                  const std::vector<std::size_t>& support) {
    SweepResult res;
    const auto t0 = std::chrono::steady_clock::now();

    for (std::size_t T : T_grid) {
        solver.executeStep(T, /*early_stop=*/true);

        const auto selected = solver.getSelectedOriginals();
        SweepRow row{};
        row.T = T;
        row.steps = solver.getNumSteps();
        row.actives = solver.getActives().size();
        row.reals = selected.size();
        row.dummies = solver.getNumActiveDummies();
        for (std::size_t j : selected) {
            if (inSupport(j, support)) row.tp++;
            else row.fp++;
        }
        row.fdp = selected.empty() ? 0.0 : double(row.fp) / double(selected.size());
        row.tpr = support.empty() ? 1.0 : double(row.tp) / double(support.size());
        row.lambda = solver.getLambda().empty() ? 0.0 : solver.getLambda().back();
        res.rows.push_back(row);
        res.actives_per_T.push_back(solver.getActives());
    }

    const auto t1 = std::chrono::steady_clock::now();
    res.millis = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.actives = solver.getActives();
    res.generated = solver.getNumGeneratedDummies();
    return res;
}

void printSweep(const std::string& name, const SweepResult& res,
                std::size_t p, const std::vector<std::size_t>& support,
                bool with_paths = true) {
    std::cout << "--- " << name << " ---\n";
    std::cout << std::setw(6)  << "T"
              << std::setw(8)  << "steps"
              << std::setw(10) << "actives"
              << std::setw(8)  << "reals"
              << std::setw(9)  << "dummies"
              << std::setw(6)  << "TP"
              << std::setw(6)  << "FP"
              << std::setw(9)  << "FDP"
              << std::setw(9)  << "TPR"
              << std::setw(13) << "lambda_stop" << "\n";
    std::cout << std::string(84, '-') << "\n";
    for (const auto& row : res.rows) {
        std::cout << std::setw(6)  << row.T
                  << std::setw(8)  << row.steps
                  << std::setw(10) << row.actives
                  << std::setw(8)  << row.reals
                  << std::setw(9)  << row.dummies
                  << std::setw(6)  << row.tp
                  << std::setw(6)  << row.fp
                  << std::setw(9)  << std::fixed << std::setprecision(3) << row.fdp
                  << std::setw(9)  << row.tpr
                  << std::setw(13) << std::setprecision(4) << row.lambda
                  << std::defaultfloat << "\n";
    }

    // Colored entry sequences, one snapshot per T stop (blank line between):
    // green TP real, red FP real, blue dummy.
    std::cout << "\n";
    if (with_paths) {
        for (std::size_t ti = 0; ti < res.rows.size(); ++ti) {
            std::cout << "path (T=" << res.rows[ti].T << "): ";
            for (std::size_t j : res.actives_per_T[ti]) {
                if (j >= p)                     std::cout << kBlue  << "D" << kReset << ' ';
                else if (inSupport(j, support)) std::cout << kGreen << "R" << kReset << ' ';
                else                            std::cout << kRed   << "R" << kReset << ' ';
            }
            std::cout << "\n\n";
        }
    }
    std::cout << "sweep time: " << std::fixed << std::setprecision(1)
              << res.millis << " ms   (dummies generated: " << res.generated
              << ")" << std::defaultfloat << "\n\n";
}

void printAgreement(const std::string& what, const SweepResult& a, const SweepResult& b) {
    const bool same = (a.actives == b.actives);
    std::cout << what << ": "
              << (same ? "identical selection paths"
                       : "PATHS DIFFER (expected only at floating-point ties)")
              << "  [" << a.actives.size() << " vs " << b.actives.size()
              << " actives]\n";
}

// -----------------------------------------------------------------------------------
// Scenario A: LARS vs OMP vs AFS at fixed k
// -----------------------------------------------------------------------------------
void scenarioAlgorithms() {
    const std::size_t n = 300, p = 1000, sparsity = 10;
    const double snr = 1.0, rho_d = 0.05;   // k = 7 at n = 300
    const std::size_t L_max = 2 * p;
    const std::vector<std::size_t> T_grid{1, 2, 5, 10, 20};
    const uint32_t data_seed = 42;
    const uint64_t solver_seed = 1234;

    std::cout << "\n=== Scenario A: LARS vs OMP vs AFS (rho=" << kAfsRho << "), k=7 ===\n"
              << "n=" << n << "  p=" << p << "  |support|=" << sparsity
              << "  snr=" << snr << "  L_max=" << L_max
              << "   [" << kGreen << "R" << kReset << "=TP, "
              << kRed << "R" << kReset << "=FP, "
              << kBlue << "D" << kReset << "=dummy]\n\n";

    auto data = LinearSimulator::generate(n, p, sparsity, 1.0, snr, 0.0, data_seed);
    const std::size_t T_max = T_grid.back();

    {
        Eigen::MatrixXd X = data.X; Eigen::VectorXd y = data.y;
        preprocess(X, y);
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
        lars::SD_TLARS_Solver solver(Xm, ym, rho_d, L_max, T_max, true, solver_seed);
        auto res = sweep(solver, T_grid, data.support_true);
        printSweep("SD_TLARS (reference)", res, p, data.support_true);
    }
    {
        Eigen::MatrixXd X = data.X; Eigen::VectorXd y = data.y;
        preprocess(X, y);
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
        omp::SD_TOMP_Solver solver(Xm, ym, rho_d, L_max, T_max, true, solver_seed);
        auto res = sweep(solver, T_grid, data.support_true);
        printSweep("SD_TOMP (greedy, full OLS refit)", res, p, data.support_true);
    }
    {
        Eigen::MatrixXd X = data.X; Eigen::VectorXd y = data.y;
        preprocess(X, y);
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
        afs::SD_TAFS_Solver solver(Xm, ym, rho_d, L_max, T_max, true, solver_seed, kAfsRho);
        auto res = sweep(solver, T_grid, data.support_true);
        printSweep("SD_TAFS (rho=0.3, re-selection blend)", res, p, data.support_true);
    }
}

// -----------------------------------------------------------------------------------
// Scenario B: pair twins at k = 1, both generation policies
// -----------------------------------------------------------------------------------
void scenarioPairTwins() {
    const std::size_t n = 300, p = 1000, sparsity = 10;
    const double snr = 1.0;
    const std::size_t L_max = 2 * p;
    const std::vector<std::size_t> T_grid{1, 2, 5, 10, 20};
    const uint32_t data_seed = 42;
    const uint64_t solver_seed = 1234;

    std::cout << "\n=== Scenario B: pair twins (k=1) — OMP and AFS ===\n"
              << "n=" << n << "  p=" << p << "  |support|=" << sparsity
              << "  snr=" << snr << "  L_max=" << L_max << "\n\n";

    auto data = LinearSimulator::generate(n, p, sparsity, 1.0, snr, 0.0, data_seed);
    const std::size_t T_max = T_grid.back();

    auto freshMaps = [&](Eigen::MatrixXd& X, Eigen::VectorXd& y) {
        X = data.X; y = data.y;
        preprocess(X, y);
    };

    SweepResult res_g_omp, res_q_omp, res_geo_omp;
    SweepResult res_g_afs, res_q_afs, res_geo_afs;
    {
        Eigen::MatrixXd X; Eigen::VectorXd y; freshMaps(X, y);
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
        omp::SD_TOMP_Solver solver(Xm, ym, 2.0 / double(n), L_max, T_max, true, solver_seed);
        res_g_omp = sweep(solver, T_grid, data.support_true);
        printSweep("SD_TOMP (k=1, general arithmetic)", res_g_omp, p, data.support_true,
                   /*with_paths=*/false);
    }
    {
        Eigen::MatrixXd X; Eigen::VectorXd y; freshMaps(X, y);
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
        omp::SD2_TOMP_Solver solver(Xm, ym, L_max, T_max, true, solver_seed,
                                    SD2GenPolicy::OnDemand);
        res_q_omp = sweep(solver, T_grid, data.support_true);
        printSweep("SD2_TOMP (pair, on-demand pool)", res_q_omp, p, data.support_true,
                   /*with_paths=*/false);
    }
    {
        Eigen::MatrixXd X; Eigen::VectorXd y; freshMaps(X, y);
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
        omp::SD2_TOMP_Solver solver(Xm, ym, L_max, T_max, true, solver_seed,
                                    SD2GenPolicy::Geometric);
        res_geo_omp = sweep(solver, T_grid, data.support_true);
        printSweep("SD2_TOMP (pair, geometric)", res_geo_omp, p, data.support_true,
                   /*with_paths=*/false);
    }
    {
        Eigen::MatrixXd X; Eigen::VectorXd y; freshMaps(X, y);
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
        afs::SD_TAFS_Solver solver(Xm, ym, 2.0 / double(n), L_max, T_max, true,
                                   solver_seed, kAfsRho);
        res_g_afs = sweep(solver, T_grid, data.support_true);
        printSweep("SD_TAFS (k=1, general arithmetic)", res_g_afs, p, data.support_true,
                   /*with_paths=*/false);
    }
    {
        Eigen::MatrixXd X; Eigen::VectorXd y; freshMaps(X, y);
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
        afs::SD2_TAFS_Solver solver(Xm, ym, L_max, T_max, true, solver_seed,
                                    SD2GenPolicy::OnDemand, kAfsRho);
        res_q_afs = sweep(solver, T_grid, data.support_true);
        printSweep("SD2_TAFS (pair, on-demand pool)", res_q_afs, p, data.support_true,
                   /*with_paths=*/false);
    }
    {
        Eigen::MatrixXd X; Eigen::VectorXd y; freshMaps(X, y);
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
        afs::SD2_TAFS_Solver solver(Xm, ym, L_max, T_max, true, solver_seed,
                                    SD2GenPolicy::Geometric, kAfsRho);
        res_geo_afs = sweep(solver, T_grid, data.support_true);
        printSweep("SD2_TAFS (pair, geometric)", res_geo_afs, p, data.support_true,
                   /*with_paths=*/false);
    }

    printAgreement("OMP twin (SD_TOMP k=1 vs SD2_TOMP on-demand)", res_g_omp, res_q_omp);
    printAgreement("AFS twin (SD_TAFS k=1 vs SD2_TAFS on-demand)", res_g_afs, res_q_afs);
    std::cout << "\n";
}

// -----------------------------------------------------------------------------------
// Scenario C: large p, auto-calibrated k and L
// -----------------------------------------------------------------------------------
void scenarioLargeP() {
    const std::size_t n = 300, p = 100000, sparsity = 10;
    const double snr = 2.0;
    const std::vector<std::size_t> T_grid{1, 5, 10, 20};
    const uint32_t data_seed = 42;
    const uint64_t solver_seed = 1234;

    std::cout << "\n=== Scenario C: large p, auto-calibrated (rho_d=0, L_max=0) ===\n"
              << "n=" << n << "  p=" << p << "  |support|=" << sparsity
              << "  snr=" << snr << "\n\n";

    auto data = LinearSimulator::generate(n, p, sparsity, 1.0, snr, 0.0, data_seed);
    const std::size_t T_max = T_grid.back();

    bool calib_printed = false;
    auto printCalib = [&](const auto& solver) {
        if (calib_printed || !solver.getAutoCalibration()) return;
        const auto& cal = *solver.getAutoCalibration();
        std::cout << "auto-calibration: k=" << cal.k << "  rho_d="
                  << std::fixed << std::setprecision(4) << cal.rho_d
                  << std::defaultfloat << "  L=" << cal.L
                  << "  (feasible=" << (cal.feasible ? "yes" : "no") << ")\n\n";
        calib_printed = true;
    };

    {
        Eigen::MatrixXd X = data.X; Eigen::VectorXd y = data.y;
        preprocess(X, y);
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
        lars::SD_TLARS_Solver solver(Xm, ym, /*rho_d=*/0.0, /*L_max=*/0, T_max, true, solver_seed);
        printCalib(solver);
        auto res = sweep(solver, T_grid, data.support_true);
        printSweep("SD_TLARS (auto)", res, p, data.support_true, /*with_paths=*/false);
    }
    {
        Eigen::MatrixXd X = data.X; Eigen::VectorXd y = data.y;
        preprocess(X, y);
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
        omp::SD_TOMP_Solver solver(Xm, ym, /*rho_d=*/0.0, /*L_max=*/0, T_max, true, solver_seed);
        auto res = sweep(solver, T_grid, data.support_true);
        printSweep("SD_TOMP (auto)", res, p, data.support_true, /*with_paths=*/false);
    }
    {
        Eigen::MatrixXd X = data.X; Eigen::VectorXd y = data.y;
        preprocess(X, y);
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
        afs::SD_TAFS_Solver solver(Xm, ym, /*rho_d=*/0.0, /*L_max=*/0, T_max, true,
                                   solver_seed, kAfsRho);
        auto res = sweep(solver, T_grid, data.support_true);
        printSweep("SD_TAFS (auto, rho=0.3)", res, p, data.support_true, /*with_paths=*/false);
    }
}

} // namespace

// ===================================================================================

int main() {
    std::cout << "Demo 3: SD-OMP and SD-AFS — greedy solvers on sparse Rademacher dummies\n";
    scenarioAlgorithms();
    scenarioPairTwins();
    scenarioLargeP();
    return 0;
}
