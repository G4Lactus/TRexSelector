// ===================================================================================
// demo_trex_sd.cpp
// ===================================================================================
/**
 * @file demo_trex_sd.cpp
 *
 * @brief Demo 3: TRexSD — the K-experiment Sparse-Dummy T-Rex selector.
 *
 * @details The single-path SD race loses power at the detection boundary:
 *          weak signals rank below the strongest null correlations, so a
 *          single FDR-controlled path must refuse them (demo 1 at this
 *          scale: TPR 0.8 at T=1). The T-Rex selector recovers that power by
 *          voting over K random experiments — a boundary signal outranks the
 *          dummies in most experiments, while an exchangeable null cannot
 *          sustain a high relative occurrence Phi_j.
 *
 *          This demo runs, on the same data (n=300, p=1e5, |S|=10, snr=3):
 *            1. a single-path SD baseline (T = 1 and 5), and
 *            2. TRexSD (K=20, tFDR=0.1, auto-k, L=2p, (T*, v*) calibration),
 *          and compares TP/FP/FDP/TPR.
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

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

bool inSupport(std::size_t j, const std::vector<std::size_t>& support) {
    return std::find(support.begin(), support.end(), j) != support.end();
}

void report(const char* label, const std::vector<std::size_t>& selected,
            const std::vector<std::size_t>& support) {
    int tp = 0;
    for (std::size_t j : selected)
        if (inSupport(j, support)) tp++;
    const int fp = (int)selected.size() - tp;
    const double fdp = selected.empty() ? 0.0 : double(fp) / double(selected.size());
    const double tpr = support.empty() ? 1.0 : double(tp) / double(support.size());

    std::cout << std::left << std::setw(34) << label << std::right
              << "  |S^|=" << std::setw(3) << selected.size()
              << "  TP=" << std::setw(2) << tp
              << "  FP=" << std::setw(3) << fp
              << "  FDP=" << std::fixed << std::setprecision(3) << fdp
              << "  TPR=" << tpr << std::defaultfloat << "\n";
}

}  // namespace

// ===================================================================================
// Main
// ===================================================================================

int main() {

    std::cout << "==========================================================\n"
              << " Demo 3: TRexSD — K-experiment Sparse-Dummy T-Rex\n"
              << "==========================================================\n";

    // ---- Scenario: the detection-boundary regime of demo 1 ----
    const std::size_t n = 300, p = 100'000, sparsity = 10;
    const double snr = 3.0;
    const uint32_t data_seed = 42;

    std::cout << "data: n=" << n << " p=" << p << " |support|=" << sparsity
              << " snr=" << snr << " seed=" << data_seed
              << "   (generating...)\n";

    auto data = LinearSimulator::generate(n, p, sparsity, 1.0, snr, 0.0, data_seed);

    // Preprocess in place: centered, unit-L2 columns; centered y.
    for (Eigen::Index j = 0; j < data.X.cols(); ++j) {
        data.X.col(j).array() -= data.X.col(j).mean();
        const double nrm = data.X.col(j).norm();
        if (nrm > 0.0) data.X.col(j) /= nrm;
    }
    data.y.array() -= data.y.mean();

    try {
        // ---- 1. Single-path baseline (one random experiment) ----
        std::cout << "\n--- single-path SD baseline (rho_d = auto) ---\n";
        std::vector<std::size_t> single_T1, single_T5;
        {
            Eigen::MatrixXd X = data.X;   // the solver holds a map; keep X alive
            Eigen::VectorXd y = data.y;
            Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
            Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());

            sd::SD_TLARS_Solver solver(Xm, ym, /*rho_d=*/0.0, /*L_max=*/0,
                                       /*T_stop=*/5, /*intercept=*/true, /*seed=*/42);
            solver.executeStep(1, true);
            single_T1 = solver.getSelectedOriginals();
            solver.executeStep(5, true);
            single_T5 = solver.getSelectedOriginals();
        }
        report("single path, T=1", single_T1, data.support_true);
        report("single path, T=5", single_T5, data.support_true);

        // ---- 2. TRexSD: K experiments, (T*, v*) calibration, two targets ----
        for (double tFDR : {0.1, 0.2}) {
            std::cout << "\n--- TRexSD (K=20, tFDR=" << tFDR
                      << ", L=2p, auto-k) ---\n";
            tsd::TRexSDOptions opt;
            opt.tFDR     = tFDR;
            opt.K        = 20;
            opt.L_factor = 2;
            opt.rho_d    = 0.0;   // auto-calibrate once
            opt.solver   = tsd::SDSolverType::General;
            opt.calib    = tsd::CalibMode::CalibrateT;
            opt.seed     = 42;
            opt.verbose  = true;

            tsd::TRexSD selector(opt);

            const auto t0 = std::chrono::steady_clock::now();
            tsd::TRexSDResult res = selector.run(data.X, data.y);
            const auto t1 = std::chrono::steady_clock::now();

            std::vector<std::size_t> selected(res.selected_var.data(),
                                              res.selected_var.data() + res.selected_var.size());
            std::sort(selected.begin(), selected.end());

            std::cout << "\n";
            report("TRexSD (v*, T*)", selected, data.support_true);
            std::cout << "v*=" << res.v_thresh << "  T*=" << res.T_stop
                      << "  L=" << res.num_dummies
                      << "  k=" << res.k_used
                      << "  runtime=" << std::fixed << std::setprecision(1)
                      << std::chrono::duration<double>(t1 - t0).count() << " s"
                      << std::defaultfloat << "\n";

            std::cout << "\nselected:    ";
            for (std::size_t j : selected) std::cout << " " << j;
            std::cout << "\ntrue support:";
            for (std::size_t j : data.support_true) std::cout << " " << j;
            std::cout << "\n";
        }

        // ---- 3. TRexSD with the T-Rex L-loop (CalibrateBoth) ----
        {
            std::cout << "\n--- TRexSD (K=20, tFDR=0.2, L-loop + (T,v) search) ---\n";
            tsd::TRexSDOptions opt;
            opt.tFDR     = 0.2;
            opt.K        = 20;
            opt.rho_d    = 0.0;
            opt.solver   = tsd::SDSolverType::General;
            opt.calib    = tsd::CalibMode::CalibrateBoth;
            opt.seed     = 42;
            opt.verbose  = true;

            tsd::TRexSD selector(opt);
            tsd::TRexSDResult res = selector.run(data.X, data.y);

            std::vector<std::size_t> selected(res.selected_var.data(),
                                              res.selected_var.data() + res.selected_var.size());
            std::sort(selected.begin(), selected.end());
            std::cout << "\n";
            report("TRexSD (L-loop, v*, T*)", selected, data.support_true);
            std::cout << "v*=" << res.v_thresh << "  T*=" << res.T_stop
                      << "  L=" << res.num_dummies << "  k=" << res.k_used << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }

    std::cout << "\nDone.\n";
    return 0;
}
