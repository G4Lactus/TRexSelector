// ==============================================================================
// demo_trx_02_classic_trex_mmap.cpp
// ==============================================================================
/**
 * @file demo_trx_02_classic_trex_mmap.cpp
 *
 * @brief Demo of T-Rex Selector with memory-mapped workflows.
 *
 * @details Shows two single-run memory-mapped usage patterns:
 *
 *  (A) In-memory X with internal mmap support enabled:
 *      Setting use_memory_mapping = true activates both solver serialization
 *      (e.g., the LARS-path state of size (p+L) x T_stop per solver is written to
 *      disk between T-loop iterations instead of being kept in RAM) and
 *      memory-mapped storage for the internal dummy matrices D. Solvers never
 *      store X or D; their footprint scales as O(T * (p + L)). Solver serialization
 *      and D mmap are currently coupled via the single use_memory_mapping flag.
 *
 *  (B) Fully memory-mapped pipeline:
 *      X itself is backed by a Boost memory-mapped file (SyntheticDataMapped),
 *      so the design matrix never resides fully in RAM. Combined with
 *      use_memory_mapping = true, this demonstrates the full mmap pipeline:
 *      X mmap + D mmap + solver serialization.
 */
// ==============================================================================

// std includes
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

// T-Rex Selector includes
#include <trex_selector_methods/trex_core/trex.hpp>
#include <utils/datageneration/utils_datagen.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>

// Demo utilities
#include "demo_trx_utils.hpp"

// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace cdianostics = trex::utils::eval::cdiagnostics;
namespace datagen     = trex::utils::datageneration::datagen;
namespace dummygen    = trex::utils::datageneration::dummygen;
namespace rates       = trex::utils::eval::rates;
namespace fs          = std::filesystem;

// T-Rex types
using trex::trex_selector_methods::trex_core::LLoopStrategy;
using trex::trex_selector_methods::trex_core::TRexControlParameter;
using trex::trex_selector_methods::trex_core::TRexSelector;
using trex::trex_selector_methods::utils::solver_dispatch::SolverTypeForTRex;

// ==============================================================================


// ==============================================================================
// Shared demo configuration
// ==============================================================================

struct MmapDemoConfig {
    Eigen::Index             n, p;
    std::vector<std::size_t> true_support;
    std::vector<double>      true_coefs;
    double snr  = 1.0;
    double tFDR = 0.1;
};

static MmapDemoConfig make_mmap_demo_config(bool high_dim, bool rnd_coef) {
    MmapDemoConfig cfg;
    cfg.n            = high_dim ? 1000 : 5000;
    cfg.p            = high_dim ? 5000 : 1000;
    cfg.true_support = {27, 149, 398, 420, 4};
    cfg.true_coefs   = rnd_coef
        ? std::vector<double>{-0.4, -0.25, -0.8, 1.1, 2.5}
        : std::vector<double>{1, 1, 1, 1, 1};
    cfg.snr  = 1.0;
    cfg.tFDR = 0.1;
    return cfg;
}

/**
 * @brief Shared TRexControlParameter baseline for all mmap demos.
 * @note  use_memory_mapping = true couples two behaviours:
 *   (1) Internal dummy matrices D are memory-mapped (TRexMemMapManager).
 *   (2) Solver warm-start state is serialized to disk between T-loop iterations
 *       (SERIALIZED WarmStartMode). Solver footprint scales as O(T * (p + L)).
 */
static TRexControlParameter make_mmap_trex_ctrl() {
    TRexControlParameter ctrl;
    ctrl.K                     = 20;
    ctrl.max_dummy_multiplier  = 10;
    ctrl.use_max_T_stop        = true;
    ctrl.dummy_distribution    = dummygen::Distribution::Normal();
    ctrl.lloop_strategy        = LLoopStrategy::HCONCAT;
    ctrl.tloop_stagnation_stop = true;
    ctrl.use_memory_mapping    = true;  // enables D mmap + solver serialization
    ctrl.solver_type           = SolverTypeForTRex::TLARS;
    return ctrl;
}

// ==============================================================================
// Demo A: In-memory X + use_memory_mapping=true
//         Activates solver serialization and memory-mapped dummy matrices D.
// ==============================================================================

void demo_TRexSelector_d_mmap_solver_serial(bool high_dim, bool rnd_coef) {

    cdianostics::print_section_header(
        "Demo A: In-Memory X — Solver Serialization + D Memory-Mapping");

    const auto cfg = make_mmap_demo_config(high_dim, rnd_coef);

    // Setup dual output (console + file)
    const std::string folder = "simulations/";
    const std::string stem   = "trex_mmap_demo_a_n" + std::to_string(cfg.n) +
                               "_p" + std::to_string(cfg.p);
    std::ofstream out_file(folder + stem + ".txt");
    PrintFn print_dual = [&](const std::string& s) {
        std::cout << s;
        if (out_file.is_open()) out_file << s;
    };

    print_dual(std::string(high_dim ? "High-dimensional (p > n)"
                                    : "Low-dimensional (n > p)") + "\n");

    // Generate design matrix and response in RAM
    print_dual("Generating synthetic data (in-memory)...\n");
    datagen::SyntheticData data(cfg.n,
                                cfg.p,
                                cfg.true_support,
                                cfg.true_coefs,
                                cfg.snr,
                                58);

    // Eigen::Map views required by TRexSelector
    Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(),
                                      data.rows(),
                                      data.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

    TRexControlParameter trex_ctrl = make_mmap_trex_ctrl();

    print_dual("Creating T-Rex Selector instance...\n");
    TRexSelector trex(X_map,
                      y_map,
                      cfg.tFDR,
                      trex_ctrl,
                      -1,
                      true);

    print_dual("Executing T-Rex Selector...\n");
    trex.select();

    print_results(trex.getSelectedIndices(),
                  cfg.true_support,
                  trex,
                  print_dual);
    save_selection_csv(folder + stem + ".csv",
                       cfg.p,
                       trex.getPhiPrime(),
                       trex.getSelectedIndices(),
                       cfg.true_support);

    print_dual("\n\n");

    if (out_file.is_open()) {
        std::cout << "[Info] Results saved to:              " << folder + stem + ".txt\n";
        out_file.close();
    }
}


// ==============================================================================
// Demo B: Memory-mapped X + use_memory_mapping=true
//         Full mmap pipeline: X mmap + D mmap + solver serialization.
// ==============================================================================

void demo_TRexSelector_full_mmap(bool high_dim, bool rnd_coef) {

    cdianostics::print_section_header(
        "Demo B: Memory-Mapped X — Full mmap Pipeline (X + D + Solver Serialization)");

    const auto cfg = make_mmap_demo_config(high_dim, rnd_coef);

    // Setup dual output (console + file)
    const std::string folder = "simulations/";
    const std::string stem   = "trex_mmap_demo_b_n" + std::to_string(cfg.n) +
                               "_p" + std::to_string(cfg.p);
    std::ofstream out_file(folder + stem + ".txt");
    PrintFn print_dual = [&](const std::string& s) {
        std::cout << s;
        if (out_file.is_open()) out_file << s;
    };

    print_dual(std::string(high_dim ? "High-dimensional (p > n)"
                                    : "Low-dimensional (n > p)") + "\n");

    const std::string X_filepath = "X_mmap.dat";
    const std::string y_filepath = "y_mmap.dat";

    // RAII guard: removes X/y backing files on scope exit — even if an exception
    // is thrown. Declared BEFORE `data` so that C++ reverse-LIFO destruction
    // runs:
    // (1) data dtor closes mmap handles,
    // (2) guard dtor removes the files.
    MmapFileGuard mmap_guard{{X_filepath, y_filepath}};

    // X and y are written to disk and never fully resides in RAM
    print_dual("Generating synthetic data (memory-mapped files: " +
               X_filepath + ", " + y_filepath + ")...\n");
    datagen::SyntheticDataMapped data(
        X_filepath,
        y_filepath,
        cfg.n,
        cfg.p,
        cfg.true_support,
        cfg.true_coefs,
        cfg.snr,
        58
    );

    // Eigen::Map views as lvalues (required by TRexSelector constructor)
    Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(),
                                      data.rows(),
                                      data.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

    // NOTE: On top of the already memory-mapped X, use_memory_mapping = true
    // additionally stores the internal dummy matrices D in mmap files and
    // serializes solver LARS-path checkpoints ((p+L) x T_stop per solver) to
    // disk between T-loop iterations.
    TRexControlParameter trex_ctrl = make_mmap_trex_ctrl();

    print_dual("Creating T-Rex Selector instance...\n");
    TRexSelector trex(X_map,
                      y_map,
                      cfg.tFDR,
                      trex_ctrl,
                      -1,
                      true);

    print_dual("Executing T-Rex Selector...\n");
    trex.select();

    print_results(trex.getSelectedIndices(),
                  cfg.true_support,
                  trex,
                  print_dual);
    save_selection_csv(folder + stem + ".csv",
                       cfg.p,
                       trex.getPhiPrime(),
                       trex.getSelectedIndices(),
                       cfg.true_support);

    print_dual("\n\n");

    if (out_file.is_open()) {
        std::cout << "[Info] Results saved to:              " << folder + stem + ".txt\n";
        out_file.close();
    }
    // mmap_guard dtor removes X_filepath + y_filepath
}


// ==============================================================================
// Main
// ==============================================================================

int main() {

    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(6);
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    // ============================================================
    // Demo A: single run — in-memory X + solver serialization + D mmap
    // ============================================================
    if (false)
        demo_TRexSelector_d_mmap_solver_serial(/*high_dim=*/false, /*rnd_coef=*/false);
    if (false)
        demo_TRexSelector_d_mmap_solver_serial(/*high_dim=*/true,  /*rnd_coef=*/false);

    // ============================================================
    // Demo B: single run — fully memory-mapped pipeline (X + D + solver serialization)
    // ============================================================
    if (false)
        demo_TRexSelector_full_mmap(/*high_dim=*/false, /*rnd_coef=*/false);
    if (false)
        demo_TRexSelector_full_mmap(/*high_dim=*/true,  /*rnd_coef=*/false);

    return 0;
}
// ==============================================================================
