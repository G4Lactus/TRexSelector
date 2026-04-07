// ==============================================================================
// sim_trx_03_da_fdr_mc.cpp
// ==============================================================================
/**
 * @file sim_trx_03_da_fdr_mc.cpp
 *
 * @brief DA-TRex MC simulation: target-FDR sweep for multiple within-block
 *        correlation levels on block-diagonal Toeplitz data.
 *
 * @details
 *  Base setting: n=150, p=500, G=5, g=5, SNR=2.0.
 *
 *  For each rho in {0.7, 0.8, 0.9}:
 *      sweep tFDR in {0.00, 0.05, 0.10, ..., 0.50}
 *
 *  200 MC trials per grid point, 3 solvers (TLARS, TAFS, TOMP), DA method BT.
 *
 *  Reference: Sim_03.cpp (stub file).
 */
// ==============================================================================

#include "da_trex_sim_common.hpp"

using namespace da_sim;


// ==============================================================================
// Main
// ==============================================================================

int main() {

    std::cout.setf(std::ios::unitbuf);

    BlockSimConfig cfg;  // n=150, p=500, G=5, g=5
    auto solvers = default_solvers();

    const std::vector<double> rho_values  = {0.7, 0.8, 0.9};
    const std::vector<double> tfdr_values = {0.00, 0.05, 0.10, 0.15, 0.20,
                                             0.25, 0.30, 0.35, 0.40, 0.45, 0.50};

    cdianostics::print_section_header(
        "DA-TRex tFDR Sweep  |  MC=" + std::to_string(cfg.num_MC)
        + "  n=" + std::to_string(cfg.n) + "  p=" + std::to_string(cfg.p));


    for (double rho : rho_values) {

        cdianostics::print_section_header(
            "tFDR sweep  |  rho = " + std::to_string(rho));

        const auto ngrid = static_cast<Eigen::Index>(tfdr_values.size());
        std::map<std::string, Eigen::VectorXd> fdr_map, tpr_map, avg_L_map, avg_T_map;
        for (const auto& sv : solvers) {
            fdr_map[sv.name]   = Eigen::VectorXd::Zero(ngrid);
            tpr_map[sv.name]   = Eigen::VectorXd::Zero(ngrid);
            avg_L_map[sv.name] = Eigen::VectorXd::Zero(ngrid);
            avg_T_map[sv.name] = Eigen::VectorXd::Zero(ngrid);
        }

        auto trex_ctrl = make_trex_control(cfg.K);
        auto da_ctrl   = make_da_bt_control();

        // Pre-generate DGP data with this rho (DGP is independent of tFDR)
        // so we can reuse data across tFDR values for a fair comparison.
        //
        // Actually, for consistency with Monte Carlo convention and because
        // each trial needs a fresh draw, we generate anew each time but use
        // the same seed sequence across tFDR values.

        for (const auto& sv : solvers) {
            std::cout << "\n  Solver: " << sv.name << "\n";

            trex_ctrl.solver_type           = sv.type;
            trex_ctrl.solver_params.rho_afs = sv.rho_afs;

            for (std::size_t gi = 0; gi < tfdr_values.size(); ++gi) {
                const double tFDR_val = tfdr_values[gi];

                std::stringstream lbl;
                lbl << "tFDR = " << std::fixed << std::setprecision(2) << tFDR_val;

                auto dgp_factory = [&](unsigned seed) -> DGPData {
                    return dgp_block_toeplitz(cfg.n, cfg.p, cfg.G, cfg.g,
                                              rho, cfg.snr, seed);
                };

                unsigned base_offset = static_cast<unsigned>(cfg.base_seed)
                                     + static_cast<unsigned>(gi) * 10000u;

                auto res = run_mc_trials(
                    cfg.num_MC, lbl.str(), dgp_factory,
                    tFDR_val, trex_ctrl, da_ctrl, base_offset);

                auto idx = static_cast<Eigen::Index>(gi);
                fdr_map[sv.name](idx)   = res.avg_fdr;
                tpr_map[sv.name](idx)   = res.avg_tpr;
                avg_L_map[sv.name](idx) = res.avg_L;
                avg_T_map[sv.name](idx) = res.avg_T;
            }
        }

        std::stringstream tag;
        tag << "tFDR_rho" << std::fixed << std::setprecision(1) << rho;

        save_and_print_grid_results(
            tag.str(), "tFDR", tfdr_values, cfg.num_MC,
            solvers, fdr_map, tpr_map, avg_L_map, avg_T_map,
            "Base: SNR=2.0, G=5, g=5, rho=" + std::to_string(rho));
    }


    std::cout << "\nAll tFDR sweeps complete.\n";
    return 0;
}
