// ==============================================================================
// validation_trex_spca_03_scaling_comparison.cpp
// ==============================================================================
/**
 * @file validation_trex_spca_03_scaling_comparison.cpp
 *
 * @brief Compare column-scaling conventions (unit-L2 vs z-score) for T-Rex SPCA.
 *
 * @details
 *  Same DGP as demo_01 (sparse M-factor model, n=50, p=100, M=3, p1=5,
 *  overlap_pool_size=30). For the EN/TENET_AUG selector we sweep SNR and
 *  report FDR/TPR/PEV under both scaling modes:
 *
 *      - *-L2 : ScalingMode::L2     — center then divide each column by ||x||_2.
 *      - *-Z  : ScalingMode::ZSCORE — center then divide by sample SD (n-1),
 *               i.e. glmnet `standardize=TRUE`.
 *
 *  Motivation: plain LASSO selection is invariant under uniform column scaling,
 *  but the Elastic-Net ridge term is NOT. lambda_2 is calibrated by
 *  ridge_cv_glmnet in the unit-SD (glmnet) world; applying it to a unit-L2 X
 *  mismatches the data:ridge ratio in the augmented system. This demo checks
 *  whether switching to z-score scaling pulls the -10 dB FDR toward the target.
 *
 *  Methods compared (PC1 support metrics; num_MC-averaged):
 *      1. ENaug-Act-L2 — TENET_AUG + ActiveSet,   unit-L2 scaling
 *      2. ENaug-Act-Z  — TENET_AUG + ActiveSet,   z-score scaling
 *      3. ENaug-Thr-L2 — TENET_AUG + Thresholded, unit-L2 scaling
 *      4. ENaug-Thr-Z  — TENET_AUG + Thresholded, z-score scaling
 */
// ==============================================================================

// std includes
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

// Demo utilities (transitively includes trex_spca.hpp, pca.hpp, utils_openmp.hpp)
#include "trex_spca_sim_utils.hpp"

// ==============================================================================

// Namespace usage
using namespace spca_sim;

// ==============================================================================
// Scaling comparison — SNR sweep
// ==============================================================================

/**
 * @brief Run the L2-vs-ZSCORE SNR sweep for EN/TENET_AUG and save results.
 *
 * @param cfg        Simulation parameters (n, p, p1, M, tFDR, num_MC, seed).
 * @param snr_values SNR grid in dB.
 * @param stem       Output file stem (no folder, no extension).
 */
void demo_trex_spca_scaling_sweep(const SimConfig&           cfg,
                                  const std::vector<double>& snr_values,
                                  const std::string&         stem)
{
    std::cout << "================================================================\n";
    std::cout << "  T-Rex SPCA — Scaling Comparison (L2 vs z-score), SNR Sweep\n";
    std::cout << "================================================================\n";
    std::cout << "  n=" << cfg.n << "  p=" << cfg.p << "  M=" << cfg.M
              << "  p1=" << cfg.p1 << "  tFDR=" << cfg.tFDR
              << "  num_MC=" << cfg.num_MC << "\n\n";

    const std::vector<std::string> method_names = {
        "ENaug-Act-L2",
        "ENaug-Act-Z",
        "ENaug-Thr-L2",
        "ENaug-Thr-Z"
    };

    std::map<std::string, Eigen::VectorXd> fdr_map, tpr_map, pev_map;
    for (const auto& name : method_names) {
        fdr_map[name] = Eigen::VectorXd(snr_values.size());
        tpr_map[name] = Eigen::VectorXd(snr_values.size());
        pev_map[name] = Eigen::VectorXd(snr_values.size());
    }

    for (std::size_t si = 0; si < snr_values.size(); ++si) {
        const double   snr     = snr_values[si];
        const int      snr_int = static_cast<int>(std::round(snr));
        const unsigned offset  = static_cast<unsigned>(cfg.base_seed + snr_int);

        std::ostringstream snr_ss;
        snr_ss << std::fixed << std::setprecision(1) << snr;
        const std::string snr_lbl = "SNR=" + snr_ss.str() + " dB";

        std::cout << "--------------------------------------------------\n";
        std::cout << snr_lbl << "\n\n";

        // Capture all DGP parameters by value for OMP thread safety
        const auto make_data = [cfg, snr](unsigned seed) -> FactorModelData {
            return DataGenerator::generate_sparse_factor_model(
                cfg.n, cfg.p, cfg.p1, cfg.M, snr,
                cfg.overlap_pool_size, seed);
        };

        const auto ei = static_cast<Eigen::Index>(si);

        // 1. ENaug + ActiveSet, L2 scaling
        auto r1 = run_mc_trials_trex_spca(
            cfg.num_MC, snr_lbl + " [ENaug-Act-L2]",
            make_data, cfg.tFDR, SPCAMode::ActiveSet, offset,
            cfg.lambda_2, ENSolverType::TENET_AUG, ScalingMode::L2);
        fdr_map["ENaug-Act-L2"](ei) = r1.avg_fdr;
        tpr_map["ENaug-Act-L2"](ei) = r1.avg_tpr;
        pev_map["ENaug-Act-L2"](ei) = r1.avg_pev;

        // 2. ENaug + ActiveSet, z-score scaling
        auto r2 = run_mc_trials_trex_spca(
            cfg.num_MC, snr_lbl + " [ENaug-Act-Z]",
            make_data, cfg.tFDR, SPCAMode::ActiveSet, offset,
            cfg.lambda_2, ENSolverType::TENET_AUG, ScalingMode::ZSCORE);
        fdr_map["ENaug-Act-Z"](ei) = r2.avg_fdr;
        tpr_map["ENaug-Act-Z"](ei) = r2.avg_tpr;
        pev_map["ENaug-Act-Z"](ei) = r2.avg_pev;

        // 3. ENaug + Thresholded, L2 scaling
        auto r3 = run_mc_trials_trex_spca(
            cfg.num_MC, snr_lbl + " [ENaug-Thr-L2]",
            make_data, cfg.tFDR, SPCAMode::Thresholded, offset,
            cfg.lambda_2, ENSolverType::TENET_AUG, ScalingMode::L2);
        fdr_map["ENaug-Thr-L2"](ei) = r3.avg_fdr;
        tpr_map["ENaug-Thr-L2"](ei) = r3.avg_tpr;
        pev_map["ENaug-Thr-L2"](ei) = r3.avg_pev;

        // 4. ENaug + Thresholded, z-score scaling
        auto r4 = run_mc_trials_trex_spca(
            cfg.num_MC, snr_lbl + " [ENaug-Thr-Z]",
            make_data, cfg.tFDR, SPCAMode::Thresholded, offset,
            cfg.lambda_2, ENSolverType::TENET_AUG, ScalingMode::ZSCORE);
        fdr_map["ENaug-Thr-Z"](ei) = r4.avg_fdr;
        tpr_map["ENaug-Thr-Z"](ei) = r4.avg_tpr;
        pev_map["ENaug-Thr-Z"](ei) = r4.avg_pev;
    }

    save_and_print_spca_mc_results(cfg.num_MC, stem,
                                    snr_values, method_names,
                                    fdr_map, tpr_map, pev_map);
}


// ==============================================================================
// Main
// ==============================================================================

int main() {

    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(6);
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    SimConfig cfg;
    cfg.n                 = 50;
    cfg.p                 = 100;
    cfg.p1                = 5;
    cfg.M                 = 3;
    cfg.overlap_pool_size = 30;
    cfg.tFDR              = 0.10;
    cfg.num_MC            = 200;
    cfg.base_seed         = 42;
    cfg.lambda_2          = -1.0; // < 0 triggers k-fold CV for lambda_2 selection (0 = no ridge)

    const std::vector<double> snr_values = {-10.0, -7, -5.0, -3,
                                            0.0,
                                            3.0, 5.0, 7.0, 10.0};

    demo_trex_spca_scaling_sweep(cfg, snr_values,
                                 "validation_trex_spca_03_scaling_comparison");

    return 0;
}
