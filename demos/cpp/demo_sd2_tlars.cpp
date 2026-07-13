// ===================================================================================
// demo_sd2_tlars.cpp
// ===================================================================================
/**
 * @file demo_sd2_tlars.cpp
 *
 * @brief Demo 2: 2-sparse dummy arithmetic and generation policies.
 *
 * @details Compares, on the same data and dummy law:
 *
 *            SD_TLARS (k=1):    general index-set arithmetic (materialized
 *                               winners, O(n) generation and dots).
 *            SD2 (on-demand):   pair arithmetic, dot-free dummy path,
 *                               explicit pool, generate-until-beat loop.
 *                               Races the identical dummy stream as
 *                               SD_TLARS for the same seed.
 *            SD2 (geometric):   exact pair null — beating fraction pi from
 *                               the sorted residual, Geometric(pi) draw for
 *                               the generation count, winner sampled from
 *                               the beating set; failures stay virtual
 *                               (only their count is tracked).
 *            TLARS (explicit D): classic solver, same 2-sparse law as
 *                               materialized columns (timing reference,
 *                               fixed-L race semantics).
 *
 *          Per solver: the full early-stopping table over a T grid (steps,
 *          selected reals, dummies, TP/FP/FDP/TPR, lambda) and the colored
 *          entry sequence (green = true positive, red = false positive,
 *          blue = dummy). All paths use the sparse (compact) beta path.
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// tsolvers includes
#include <tsolvers/sd_tsolvers/sd_tlars_solver.hpp>
#include <tsolvers/sd_tsolvers/sd2_tlars_solver.hpp>
#include <tsolvers/sd_tsolvers/sd_calibration.hpp>
#include <tsolvers/linear_model/lars_based/tlars_solver.hpp>

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

bool inSupport(std::size_t j, const std::vector<std::size_t>& support) {
    return std::find(support.begin(), support.end(), j) != support.end();
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
                std::size_t p, const std::vector<std::size_t>& support) {
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
    for (std::size_t ti = 0; ti < res.rows.size(); ++ti) {
        std::cout << "path (T=" << res.rows[ti].T << "): ";
        for (std::size_t j : res.actives_per_T[ti]) {
            if (j >= p)                     std::cout << kBlue  << "D" << kReset << ' ';
            else if (inSupport(j, support)) std::cout << kGreen << "R" << kReset << ' ';
            else                            std::cout << kRed   << "R" << kReset << ' ';
        }
        std::cout << "\n\n";
    }
    std::cout << "sweep time: " << std::fixed << std::setprecision(1)
              << res.millis << " ms   (dummies generated: " << res.generated
              << ")" << std::defaultfloat << "\n\n";
}

// -----------------------------------------------------------------------------------
// One scenario: all solvers on the same data
// -----------------------------------------------------------------------------------
void preprocess(Eigen::MatrixXd& X, Eigen::VectorXd& y) {
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
        X.col(j).array() -= X.col(j).mean();
        const double nrm = X.col(j).norm();
        if (nrm > 0.0) X.col(j) /= nrm;
    }
    y.array() -= y.mean();
}

void scenario(const char* title,
              std::size_t n, std::size_t p, std::size_t sparsity, double snr,
              std::size_t L_max, const std::vector<std::size_t>& T_grid,
              uint32_t data_seed, uint64_t solver_seed,
              bool include_classic = false) {

    std::cout << "\n=== " << title << " ===\n"
              << "n=" << n << "  p=" << p << "  |support|=" << sparsity
              << "  snr=" << snr << "  L_max=" << L_max
              << "  seed=" << solver_seed
              << "   [" << kGreen << "R" << kReset << "=true positive, "
              << kRed << "R" << kReset << "=false positive, "
              << kBlue << "D" << kReset << "=dummy]\n\n";

    auto data = LinearSimulator::generate(n, p, sparsity, 1.0, snr, 0.0, data_seed);
    const std::size_t T_max = T_grid.back();

    SweepResult res_general, res_pair, res_geom;
    {
        Eigen::MatrixXd X = data.X;
        Eigen::VectorXd y = data.y;
        preprocess(X, y);
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());

        sd::SD_TLARS_Solver general(Xm, ym, /*rho_d=*/2.0 / double(n),
                                    L_max, T_max, /*intercept=*/true, solver_seed);
        res_general = sweep(general, T_grid, data.support_true);
        printSweep("SD_TLARS (k=1, on-demand pool)", res_general, p, data.support_true);
    }
    {
        Eigen::MatrixXd X = data.X;
        Eigen::VectorXd y = data.y;
        preprocess(X, y);
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());

        sd::SD2_TLARS_Solver pair(Xm, ym, L_max, T_max, /*intercept=*/true,
                                  solver_seed, sd::SD2GenPolicy::OnDemand);
        res_pair = sweep(pair, T_grid, data.support_true);
        printSweep("SD2_TLARS (pair, on-demand pool)", res_pair, p, data.support_true);
    }
    {
        Eigen::MatrixXd X = data.X;
        Eigen::VectorXd y = data.y;
        preprocess(X, y);
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());

        sd::SD2_TLARS_Solver geom(Xm, ym, L_max, T_max, /*intercept=*/true,
                                  solver_seed, sd::SD2GenPolicy::Geometric);
        res_geom = sweep(geom, T_grid, data.support_true);
        printSweep("SD2_TLARS (pair, geometric draw)", res_geom, p, data.support_true);
    }

    // Classic T-LARS with the same 2-sparse law as explicit columns.
    if (include_classic) {
        Eigen::MatrixXd X = data.X;
        Eigen::VectorXd y = data.y;
        preprocess(X, y);

        const double s = 1.0 / std::sqrt(2.0);
        Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n, L_max);
        std::mt19937_64 rng(solver_seed);
        std::uniform_int_distribution<std::size_t> first_draw(0, n - 1);
        std::uniform_int_distribution<std::size_t> second_draw(1, n - 1);
        for (std::size_t l = 0; l < L_max; ++l) {
            const std::size_t a = first_draw(rng);
            const std::size_t b = second_draw(rng);
            const std::size_t m = (b == a) ? 0 : b;
            D(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(l)) =  s;
            D(static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(l)) = -s;
        }

        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::MatrixXd> Dm(D.data(), D.rows(), D.cols());
        Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());

        trex::tsolvers::linear_model::lars_based::TLARS_Solver classic(
            Xm, Dm, ym, /*normalize=*/false, /*intercept=*/false, /*verbose=*/false);

        const auto t0 = std::chrono::steady_clock::now();
        classic.executeStep(T_max, /*early_stop=*/true);
        const auto t1 = std::chrono::steady_clock::now();
        const double solve_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "--- TLARS (explicit D, timing reference) ---\n"
                  << "steps: " << classic.getNumSteps()
                  << "   solve time (T=" << T_max << "): "
                  << std::fixed << std::setprecision(1) << solve_ms << " ms"
                  << "   explicit D: " << std::setprecision(0)
                  << (double(n * L_max) * 8.0 / 1048576.0) << " MiB"
                  << std::defaultfloat << "   (fixed-L race semantics)\n\n";
    }

    // Cross-checks
    std::cout << "SD vs SD2 (on-demand) paths: "
              << (res_general.actives == res_pair.actives ? "IDENTICAL" : "DIFFERENT")
              << " | sweep-time ratio general/pair: "
              << std::fixed << std::setprecision(2)
              << res_general.millis / res_pair.millis
              << " | geometric/pair: " << res_geom.millis / res_pair.millis
              << std::defaultfloat << "\n";
}

// -----------------------------------------------------------------------------------
// Scenario C: p = 1e6 — the dummy race calibrated by sd_calibration
// -----------------------------------------------------------------------------------
// At p >> n the race has two failure modes (both quantified by the
// calibration): a budget barrier (L must scale with p; the classic explicit
// dummy matrix would need n x L = terabytes here) and a tail barrier (the
// k=1 pair null is bounded by range(r)/sqrt(2), below the extreme null
// correlations). The calibrated k restores the race; the k=1 run at the
// same scaled budget is kept as the boundary exhibit.
void scenarioLargeP() {
    const std::size_t n = 300, p = 1'000'000, sparsity = 10;
    const double snr = 3.0;
    const uint32_t data_seed = 42;
    const uint64_t solver_seed = 42;

    std::cout << "\n=== Scenario C: p = 1e6 — calibrated race (auto k, L) ===\n"
              << "n=" << n << "  p=" << p << "  |support|=" << sparsity
              << "  snr=" << snr
              << "   generating data (~2.4 GB, several seconds)...\n";

    auto data = LinearSimulator::generate(n, p, sparsity, 1.0, snr, 0.0, data_seed);
    preprocess(data.X, data.y);  // in place; shared by every run below

    // --- Auto-calibrated general SD run (rho_d = 0, L_max = 0 -> auto) ---
    std::size_t L_used = 0;
    {
        Eigen::Map<Eigen::MatrixXd> Xm(data.X.data(), data.X.rows(), data.X.cols());
        Eigen::Map<Eigen::VectorXd> ym(data.y.data(), data.y.size());

        std::cout << "constructing SD_TLARS with rho_d = auto "
                     "(in-ctor calibration: pilot screen + MC)...\n\n";
        sd::SD_TLARS_Solver solver(Xm, ym, /*rho_d=*/0.0, /*L_max=*/0, 10,
                                   /*intercept=*/true, solver_seed);

        const auto& calib = *solver.getAutoCalibration();
        L_used = calib.L;

        std::cout << "sigma_hat=" << std::fixed << std::setprecision(4) << calib.sigma_hat
                  << "  top null bar=" << calib.bar
                  << " (" << std::setprecision(2) << calib.bar / calib.sigma_hat
                  << " sigma)\n" << std::defaultfloat;
        std::cout << std::setw(5) << "k" << std::setw(12) << "ceiling"
                  << std::setw(12) << "q(1-1/L)" << std::setw(10) << "m_entry" << "\n"
                  << std::string(39, '-') << "\n";
        for (const auto& row : calib.table) {
            std::cout << std::setw(5) << row.k
                      << std::setw(12) << std::fixed << std::setprecision(4) << row.ceiling
                      << std::setw(12) << row.q_L
                      << std::setw(10) << row.m_entry
                      << std::defaultfloat << "\n";
        }
        std::cout << "recommendation: k=" << calib.k << " (rho_d=" << calib.rho_d
                  << "), L=" << calib.L
                  << (calib.feasible ? "" : "  [NO feasible k in grid — best effort]")
                  << "   (solver: k=" << solver.sparsityK()
                  << ", L_max=" << solver.getLMax() << ")\n\n";

        auto res = sweep(solver, {1, 5, 10}, data.support_true);
        printSweep("SD_TLARS (rho_d = auto)", res, p, data.support_true);
    }

    // --- Boundary exhibit: pairs at the same scaled budget ---
    // The pair ceiling range(r)/sqrt(2) sits below the extreme null bar, so
    // the race saturates: after the first dummy (~50 false reals late), the
    // path runs to n-1 actives with at most a handful of dummies ever
    // entering — the T = 5 and T = 10 rows are identical by then.
    {
        Eigen::Map<Eigen::MatrixXd> Xm(data.X.data(), data.X.rows(), data.X.cols());
        Eigen::Map<Eigen::VectorXd> ym(data.y.data(), data.y.size());

        sd::SD2_TLARS_Solver pair(Xm, ym, L_used, 10, /*intercept=*/true,
                                  solver_seed, sd::SD2GenPolicy::OnDemand);
        auto res = sweep(pair, {1, 5, 10}, data.support_true);
        printSweep("SD2_TLARS (pair, k=1 — beyond its regime)", res, p,
                   data.support_true);
    }
}

}  // namespace

// ===================================================================================
// Main
// ===================================================================================

int main() {

    std::cout << "==========================================================\n"
              << " Demo 2: SD2-TLARS — pair arithmetic & generation policies\n"
              << "==========================================================\n";

    try {
        scenario("Scenario A: path agreement & policy comparison",
                 /*n=*/300, /*p=*/1'000, /*sparsity=*/10, /*snr=*/3.0,
                 /*L_max=*/2000, /*T_grid=*/{1, 5, 10, 20},
                 /*data_seed=*/42, /*solver_seed=*/42);

        scenario("Scenario B: timing, larger n",
                 /*n=*/2000, /*p=*/500, /*sparsity=*/10, /*snr=*/1.0,
                 /*L_max=*/10000, /*T_grid=*/{1, 5, 10, 30},
                 /*data_seed=*/7, /*solver_seed=*/7,
                 /*include_classic=*/true);

        // p = 1e6: calibrated k and L (auto), plus the k=1 boundary exhibit.
        scenarioLargeP();

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }

    std::cout << "\nDone.\n";
    return 0;
}
