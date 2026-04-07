// ==============================================================================
// demo_trx_02_classic_trex_mmap.cpp
// ==============================================================================
/**
 * @file demo_trx_02_classic_trex_mmap.cpp
 *
 * @brief Demonstration of T-Rex Selector with memory-mapped workflows.
 *
 * @details Shows two memory-mapped usage patterns:
 *
 *  (A) In-memory X with internal mmap support enabled:
 *      Setting use_memory_mapping = true activates both solver serialization
 *      (e.g., the LARS-path state of size (p+L) x T_stop per solver is written to
 *      disk between T-loop iterations instead of being kept in RAM) and
 *      memory-mapped storage for the internal dummy matrices D. Solvers never
 *      store X or D; their footprint scales as O(T*(p+L)). Solver serialization
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
#include <filesystem>
#include <iomanip>
#include <iostream>
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
// Helper: print selection results
// ==============================================================================

static void print_results(
    const std::vector<std::size_t>& selected_indices,
    const std::vector<std::size_t>& true_support)
{
    std::cout << "Selected indices: ";
    for (const auto& idx : selected_indices) { std::cout << idx << " "; }
    std::cout << "\n";

    const double fdp = rates::compute_fdp(selected_indices, true_support);
    const double tpp = rates::compute_tpp(selected_indices, true_support);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "False Discovery Proportion (FDP): " << fdp << "\n";
    std::cout << "True Positive Proportion (TPP):   " << tpp << "\n";
}


// ==============================================================================
// Demo A: In-memory X + use_memory_mapping=true
//         Activates solver serialization and memory-mapped dummy matrices D.
// ==============================================================================

void demo_TRexSelector_d_mmap_solver_serial(bool high_dim, bool rnd_coef) {

    cdianostics::print_section_header(
        "Demo A: In-Memory X — Solver Serialization + D Memory-Mapping");

    const Eigen::Index n = high_dim ? 1000 : 5000;
    const Eigen::Index p = high_dim ? 5000 : 1000;
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const std::vector<std::size_t> true_support = {27, 149, 398, 420, 4};
    const std::vector<double> true_coefs = rnd_coef ?
        std::vector<double>{-0.4, -0.25, -0.8, 1.1, 2.5} :
        std::vector<double>{1, 1, 1, 1, 1};
    const double snr = 1.0;

    // Generate design matrix and response in RAM
    std::cout << "Generating synthetic data (in-memory)...\n";
    datagen::SyntheticData data(n, p, true_support, true_coefs, snr, /*seed=*/58);

    // Eigen::Map views required by TRexSelector
    Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), data.rows(), data.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

    // NOTE: use_memory_mapping = true couples two behaviours:
    //   (1) Internal dummy matrices D are stored in Boost memory-mapped files
    //       rather than in RAM (MemmapManager).
    //   (2) Solver warm-start state (the LARS path of size (p+L) x T_stop,
    //       where L is the number of dummies and T_stop the selected stopping
    //       step) is serialized to disk between T-loop iterations instead of
    //       being kept as in-memory objects (SERIALIZED WarmStartMode).
    //       Solvers never store X or D; their footprint scales as O(T*(p+L)).
    // Both are kept coupled for now; a dedicated serialize_solvers flag
    // can be introduced independently once a concrete use case arises.
    TRexControlParameter trex_ctrl;
    trex_ctrl.K                     = 20;
    trex_ctrl.max_dummy_multiplier  = 10;
    trex_ctrl.use_max_T_stop        = true;
    trex_ctrl.dummy_distribution    = dummygen::Distribution::Normal();
    trex_ctrl.lloop_strategy        = LLoopStrategy::HCONCAT;
    trex_ctrl.tloop_stagnation_stop = true;
    trex_ctrl.use_memory_mapping    = true;   // <<< enables D mmap + solver serialization
    trex_ctrl.solver_type           = SolverTypeForTRex::TLARS;

    std::cout << "Creating T-Rex Selector instance...\n";
    TRexSelector trex(X_map, y_map, /*tFDR=*/0.1, trex_ctrl, /*seed=*/-1,
                        /*verbose=*/true);

    std::cout << "Executing T-Rex Selector...\n";
    trex.select();

    print_results(trex.getSelectedIndices(), true_support);
    std::cout << "\n\n";
}


// ==============================================================================
// Demo B: Memory-mapped X + use_memory_mapping=true
//         Full mmap pipeline: X mmap + D mmap + solver serialization.
// ==============================================================================

void demo_TRexSelector_full_mmap(bool high_dim, bool rnd_coef) {

    cdianostics::print_section_header(
        "Demo B: Memory-Mapped X — Full mmap Pipeline (X + D + Solver Serialization)");

    const Eigen::Index n = high_dim ? 1000 : 5000;
    const Eigen::Index p = high_dim ? 5000 : 1000;
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const std::vector<std::size_t> true_support = {27, 149, 398, 420, 4};
    const std::vector<double> true_coefs = rnd_coef ?
        std::vector<double>{-0.4, -0.25, -0.8, 1.1, 2.5} :
        std::vector<double>{1, 1, 1, 1, 1};
    const double snr = 1.0;

    // X and y are written to disk and accessed via Boost mmap —
    // the design matrix never fully resides in RAM.
    const std::string X_filepath = "X_mmap.dat";
    const std::string y_filepath = "y_mmap.dat";

    std::cout << "Generating synthetic data (memory-mapped files: "
              << X_filepath << ", " << y_filepath << ")...\n";
    datagen::SyntheticDataMapped data(
        X_filepath, y_filepath,
        n, p,
        true_support, true_coefs,
        snr, /*seed=*/58
    );

    // Eigen::Map views as lvalues (required by TRexSelector constructor)
    Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), data.rows(),
                                      data.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

    // NOTE: On top of the already memory-mapped X, use_memory_mapping = true
    // additionally stores the internal dummy matrices D in mmap files and
    // serializes solver LARS-path checkpoints ((p+L) x T_stop per solver) to
    // disk between T-loop iterations.
    TRexControlParameter trex_ctrl;
    trex_ctrl.K                     = 20;
    trex_ctrl.max_dummy_multiplier  = 10;
    trex_ctrl.use_max_T_stop        = true;
    trex_ctrl.dummy_distribution    = dummygen::Distribution::Normal();
    trex_ctrl.lloop_strategy        = LLoopStrategy::HCONCAT;
    trex_ctrl.tloop_stagnation_stop = true;
    trex_ctrl.use_memory_mapping    = true;   // <<< enables D mmap + solver serialization
    trex_ctrl.solver_type           = SolverTypeForTRex::TLARS;

    std::cout << "Creating T-Rex Selector instance...\n";
    TRexSelector trex(X_map, y_map, /*tFDR=*/0.1, trex_ctrl, /*seed=*/-1,
                        /*verbose=*/true);

    std::cout << "Executing T-Rex Selector...\n";
    trex.select();

    print_results(trex.getSelectedIndices(), true_support);

    // Remove mmap backing files
    fs::remove(X_filepath);
    fs::remove(y_filepath);

    std::cout << "\n\n";
}


// ==============================================================================
// Main
// ==============================================================================

int main() {

    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(6);
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    // ============================================================
    // Demo A: in-memory X + solver serialization + D mmap
    // ============================================================
    demo_TRexSelector_d_mmap_solver_serial(/*high_dim=*/false, /*rnd_coef=*/false);
    demo_TRexSelector_d_mmap_solver_serial(/*high_dim=*/true,  /*rnd_coef=*/false);

    // ============================================================
    // Demo B: fully memory-mapped pipeline (X + D + solver serialization)
    // ============================================================
    demo_TRexSelector_full_mmap(/*high_dim=*/false, /*rnd_coef=*/false);
    demo_TRexSelector_full_mmap(/*high_dim=*/true,  /*rnd_coef=*/false);

    return 0;
}
