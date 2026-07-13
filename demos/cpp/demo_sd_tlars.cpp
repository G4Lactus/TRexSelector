// ===================================================================================
// demo_sd_tlars.cpp
// ===================================================================================
/**
 * @file demo_sd_tlars.cpp
 *
 * @brief Demo 1: SD-TLARS (sparse-dummy T-LARS) with early stopping.
 *
 * @details Generates a sparse linear model y = X*beta + eps with
 *          LinearSimulator, preprocesses X and y (column-centered X,
 *          centered y), and runs SD_TLARS_Solver with an increasing dummy
 *          budget T. Per T it reports: path length, selected originals,
 *          included dummies, and TP/FP/FDP/TPR against the known support —
 *          i.e., the stopping behavior of the sparse-Rademacher dummy race.
 *
 *          The solver scales every dummy statistic by ||x||/sqrt(2k), where
 *          ||x|| is the common column norm of X, so the dummies carry exactly
 *          the column norm of X (fair race). The dummy law matches the
 *          package's ConstrainedSparseRademacher(s = rho_d): balanced +-1
 *          entries, exact column mean zero, deterministic norm sqrt(2k).
 *
 *          Two equivalent preprocessing contracts are demonstrated:
 *
 *          (a) unit-L2 columns (canonical package contract) — the dummy
 *              scale is 1/sqrt(2k);
 *
 *          (b) z-scored columns (||x_j|| = sqrt(n-1)) — the dummy scale is
 *              sqrt((n-1)/2k), i.e., the z-score scaling of a sparse
 *              Rademacher column.
 *
 *          Both differ only by the global constant sqrt(n-1), so the
 *          selection path and stopping behavior must be identical; the demo
 *          verifies this empirically.
 *
 *          Reference pattern: TRexSelector_Examples demo_ts_01_tlars, Demo 1
 *          (T-LARS with Early Stopping).
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// tsolvers includes
#include <tsolvers/sd_tsolvers/sd_tlars_solver.hpp>

// demo-local includes
#include "linear_simulator.hpp"

// ===================================================================================

namespace {

namespace sd = trex::tsolvers::linear_model::lars_based;
using trex::simulation::LinearSimulator;

// ANSI colors for the entry sequence
constexpr const char* kGreen = "\033[32m";
constexpr const char* kRed   = "\033[31m";
constexpr const char* kBlue  = "\033[34m";
constexpr const char* kReset = "\033[0m";

// -----------------------------------------------------------------------------------
// Selection quality against the known support
// -----------------------------------------------------------------------------------
struct Quality {
    int tp = 0;
    int fp = 0;
    double fdp = 0.0;
    double tpr = 0.0;
};

Quality assess(const std::vector<std::size_t>& selected,
               const std::vector<std::size_t>& support) {
    Quality q;
    for (std::size_t j : selected) {
        if (std::find(support.begin(), support.end(), j) != support.end()) q.tp++;
        else q.fp++;
    }
    q.fdp = selected.empty() ? 0.0 : double(q.fp) / double(selected.size());
    q.tpr = support.empty() ? 1.0 : double(q.tp) / double(support.size());
    return q;
}

// Colored entry sequence of the path: green R = true positive,
// red R = false positive, blue D = dummy.
std::string entrySequence(const std::vector<std::size_t>& actives, std::size_t p,
                          const std::vector<std::size_t>& support) {
    std::string s;
    s.reserve(12 * actives.size());
    for (std::size_t j : actives) {
        if (j >= p) {
            s += kBlue;
            s += 'D';
        } else if (std::find(support.begin(), support.end(), j) != support.end()) {
            s += kGreen;
            s += 'R';
        } else {
            s += kRed;
            s += 'R';
        }
        s += kReset;
        s += ' ';
    }
    return s;
}

// Result of one configuration, for cross-contract comparison.
struct ConfigResult {
    std::vector<std::size_t> actives;
    std::vector<std::size_t> selected;
    double lambda_stop = 0.0;
};

// -----------------------------------------------------------------------------------
// One configuration: preprocess to a target column norm, sweep the dummy budget T
// -----------------------------------------------------------------------------------
ConfigResult runConfig(const std::string& label,
                       const LinearSimulator::Dataset& data,
                       double target_col_norm,
                       double rho_d,
                       std::size_t L_max,
                       const std::vector<std::size_t>& T_grid,
                       uint64_t solver_seed) {

    const std::size_t n = data.X.rows();
    const std::size_t p = data.X.cols();

    // --- Preprocessing: center columns, rescale to target norm; center y ---
    Eigen::MatrixXd X = data.X;
    Eigen::VectorXd y = data.y;

    for (Eigen::Index j = 0; j < X.cols(); ++j) {
        X.col(j).array() -= X.col(j).mean();
        const double nrm = X.col(j).norm();
        if (nrm > 0.0) X.col(j) *= target_col_norm / nrm;
    }
    y.array() -= y.mean();

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    // --- Solver over the full budget; executeStep(T) continues the same path ---
    const std::size_t T_max = T_grid.back();
    sd::SD_TLARS_Solver solver(X_map, y_map, rho_d, L_max, T_max,
                               /*intercept=*/true, solver_seed);

    std::cout << "\n=== " << label << " ===\n"
              << "n=" << n << "  p=" << p
              << "  ||x_j||=" << target_col_norm
              << "  k=" << solver.sparsityK()
              << "  dummy_scale=||x||/sqrt(2k)=" << solver.getDummyScale()
              << "  L_max=" << L_max << "\n\n";

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

    for (std::size_t T : T_grid) {
        solver.executeStep(T, /*early_stop=*/true);

        const auto selected = solver.getSelectedOriginals();
        const Quality q = assess(selected, data.support_true);
        const double lam = solver.getLambda().empty() ? 0.0 : solver.getLambda().back();

        std::cout << std::setw(6)  << T
                  << std::setw(8)  << solver.getNumSteps()
                  << std::setw(10) << solver.getActives().size()
                  << std::setw(8)  << selected.size()
                  << std::setw(9)  << solver.getNumActiveDummies()
                  << std::setw(6)  << q.tp
                  << std::setw(6)  << q.fp
                  << std::setw(9)  << std::fixed << std::setprecision(3) << q.fdp
                  << std::setw(9)  << q.tpr
                  << std::setw(13) << std::setprecision(4) << lam
                  << std::defaultfloat << "\n";
    }

    // --- Path anatomy at the final budget ---
    std::cout << "\nentry sequence (" << kGreen << "R" << kReset
              << "=true positive, " << kRed << "R" << kReset
              << "=false positive, " << kBlue << "D" << kReset << "=dummy):\n  "
              << entrySequence(solver.getActives(), p, data.support_true) << "\n";

    ConfigResult res;
    res.actives = solver.getActives();
    res.selected = solver.getSelectedOriginals();
    res.lambda_stop = solver.getLambda().empty() ? 0.0 : solver.getLambda().back();

    std::cout << "selected originals (entry order):";
    for (std::size_t j : res.selected) std::cout << " " << j;
    std::cout << "\ntrue support:                    ";
    for (std::size_t j : data.support_true) std::cout << " " << j;
    std::cout << "\n";

    return res;
}

}  // namespace

// ===================================================================================
// Main
// ===================================================================================

int main() {

    std::cout << "==========================================================\n"
              << " Demo 1: SD-TLARS with Early Stopping (sparse dummies)\n"
              << "==========================================================\n";

    // ---- Scenario: sparse high-dimensional linear model ----
    const std::size_t n = 300;
    const std::size_t p = 100'000;
    const std::size_t sparsity = 10;
    const double beta_val = 1.0;
    const double snr = 3.0;
    const double toeplitz_rho = 0.0;
    const uint32_t data_seed = 42;

    const double rho_d = 0.5;            // dummy non-zero fraction: 2k = rho_d * n
    const std::size_t L_max = 2 * p;     // dummy-pool ceiling
    const std::vector<std::size_t> T_grid = {1, 2, 3, 5, 10, 20};
    const uint64_t solver_seed = 42;

    std::cout << "data: n=" << n << " p=" << p << " |support|=" << sparsity
              << " beta=" << beta_val << " snr=" << snr
              << " seed=" << data_seed << "\n";

    auto data = LinearSimulator::generate(n, p, sparsity, beta_val, snr,
                                          toeplitz_rho, data_seed);

    try {
        // (a) Canonical package contract: unit-L2 columns.
        const ConfigResult a =
            runConfig("Config A: unit-L2 contract (||x_j|| = 1)",
                      data, /*target_col_norm=*/1.0,
                      rho_d, L_max, T_grid, solver_seed);

        // (b) Z-score contract: ||x_j|| = sqrt(n-1).
        const ConfigResult b =
            runConfig("Config B: z-score contract (||x_j|| = sqrt(n-1))",
                      data, /*target_col_norm=*/std::sqrt(double(n - 1)),
                      rho_d, L_max, T_grid, solver_seed);

        // --- Scale-equivariance check: identical race up to a global constant ---
        std::cout << "\n=== Contract equivalence ===\n"
                  << "identical selection paths: "
                  << (a.actives == b.actives ? "YES" : "NO") << "\n"
                  << "lambda ratio B/A = " << b.lambda_stop / a.lambda_stop
                  << "  (expected sqrt(n-1) = " << std::sqrt(double(n - 1))
                  << ")\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }

    std::cout << "\nDone.\n";
    return 0;
}
