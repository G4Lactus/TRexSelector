// ==============================================================================
// sim_trx_02_da_block_mc.cpp
// ==============================================================================
/**
 * @file sim_trx_02_da_block_mc.cpp
 *
 * @brief DA-TRex MC simulation: 4 ceteris paribus parameter sweeps on
 *        block-diagonal Toeplitz data.
 *
 * @details
 *  Base setting: n=150, p=500, G=5, g=5, rho=0.7, SNR=2.0, tFDR=0.2.
 *  Each sweep varies exactly one parameter while fixing the rest:
 *
 *    1. SNR  in {0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0}
 *    2. rho  in {0.0, 0.1, 0.2, ..., 0.9}
 *    3. g    in {5, 10, 20, 25, 50, 100}       (divisors of p=500)
 *    4. G    in {1, 3, 5, 10, 15, 20, 30}
 *
 *  200 MC trials per grid point, 3 solvers (TLARS, TAFS, TOMP), DA method BT.
 *
 *  Reference: Sim_01.cpp + Sim_02.cpp (stub files).
 */
// ==============================================================================

#include "da_trex_sim_common.hpp"

using namespace da_sim;


// ==============================================================================
// Generic parameter sweep
// ==============================================================================

/**
 * @brief Run a ceteris paribus sweep over one parameter.
 *
 * @param scenario_tag  Short label for filenames (e.g. "block_SNR").
 * @param grid_label    Column header (e.g. "SNR").
 * @param grid_values   The values to sweep.
 * @param cfg           Base simulation config (used as template).
 * @param solvers       Solver list.
 * @param make_dgp      Lambda (grid_value, seed) -> DGPData.
 * @param header_extra  Extra info for the output header.
 */
template <typename MakeDGP>
void run_sweep(
    const std::string& scenario_tag,
    const std::string& grid_label,
    const std::vector<double>& grid_values,
    const BlockSimConfig& cfg,
    const std::vector<SolverInfo>& solvers,
    MakeDGP make_dgp,
    const std::string& header_extra = "")
{
    cdianostics::print_section_header("Sweep: " + scenario_tag);

    const auto ngrid = static_cast<Eigen::Index>(grid_values.size());
    std::map<std::string, Eigen::VectorXd> fdr_map, tpr_map, avg_L_map, avg_T_map;
    for (const auto& sv : solvers) {
        fdr_map[sv.name]   = Eigen::VectorXd::Zero(ngrid);
        tpr_map[sv.name]   = Eigen::VectorXd::Zero(ngrid);
        avg_L_map[sv.name] = Eigen::VectorXd::Zero(ngrid);
        avg_T_map[sv.name] = Eigen::VectorXd::Zero(ngrid);
    }

    auto trex_ctrl = make_trex_control(cfg.K);
    auto da_ctrl   = make_da_bt_control();

    for (const auto& sv : solvers) {
        std::cout << "\n  Solver: " << sv.name << "\n";

        trex_ctrl.solver_type           = sv.type;
        trex_ctrl.solver_params.rho_afs = sv.rho_afs;

        for (std::size_t gi = 0; gi < grid_values.size(); ++gi) {
            const double gval = grid_values[gi];

            // Build a progress label
            std::stringstream lbl;
            lbl << grid_label << " = " << std::fixed << std::setprecision(2) << gval;

            // DGP factory for this grid point
            auto dgp_factory = [&](unsigned seed) -> DGPData {
                return make_dgp(gval, seed);
            };

            unsigned base_offset = static_cast<unsigned>(cfg.base_seed)
                                 + static_cast<unsigned>(gi) * 10000u;

            auto res = run_mc_trials(
                cfg.num_MC, lbl.str(), dgp_factory,
                cfg.tFDR, trex_ctrl, da_ctrl, base_offset);

            auto idx = static_cast<Eigen::Index>(gi);
            fdr_map[sv.name](idx)   = res.avg_fdr;
            tpr_map[sv.name](idx)   = res.avg_tpr;
            avg_L_map[sv.name](idx) = res.avg_L;
            avg_T_map[sv.name](idx) = res.avg_T;
        }
    }

    save_and_print_grid_results(
        scenario_tag, grid_label, grid_values, cfg.num_MC,
        solvers, fdr_map, tpr_map, avg_L_map, avg_T_map, header_extra);
}


// ==============================================================================
// Main
// ==============================================================================

int main() {

    std::cout.setf(std::ios::unitbuf);

    BlockSimConfig cfg;  // n=150, p=500, G=5, g=5, rho=0.7, snr=2.0
    auto solvers = default_solvers();

    cdianostics::print_section_header(
        "DA-TRex Block-Toeplitz MC  |  MC=" + std::to_string(cfg.num_MC)
        + "  n=" + std::to_string(cfg.n) + "  p=" + std::to_string(cfg.p)
        + "  G=" + std::to_string(cfg.G) + "  g=" + std::to_string(cfg.g));


    // ══════════════════════════════════════════════════════════════════════════
    //  Sweep 1: SNR
    // ══════════════════════════════════════════════════════════════════════════

    run_sweep(
        "block_SNR", "SNR",
        {0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0},
        cfg, solvers,
        [&](double snr_val, unsigned seed) {
            return dgp_block_toeplitz(cfg.n, cfg.p, cfg.G, cfg.g,
                                      cfg.rho, snr_val, seed);
        },
        "Base: rho=0.7, G=5, g=5");


    // ══════════════════════════════════════════════════════════════════════════
    //  Sweep 2: rho (within-block correlation)
    // ══════════════════════════════════════════════════════════════════════════

    run_sweep(
        "block_Rho", "Rho",
        {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9},
        cfg, solvers,
        [&](double rho_val, unsigned seed) {
            return dgp_block_toeplitz(cfg.n, cfg.p, cfg.G, cfg.g,
                                      rho_val, cfg.snr, seed);
        },
        "Base: SNR=2.0, G=5, g=5");


    // ══════════════════════════════════════════════════════════════════════════
    //  Sweep 3: g (block / group size)
    //  Only values that divide p=500: {5, 10, 20, 25, 50, 100}
    // ══════════════════════════════════════════════════════════════════════════

    run_sweep(
        "block_g", "g",
        {5, 10, 20, 25, 50, 100},
        cfg, solvers,
        [&](double g_val, unsigned seed) {
            return dgp_block_toeplitz(cfg.n, cfg.p, cfg.G,
                                      static_cast<int>(g_val),
                                      cfg.rho, cfg.snr, seed);
        },
        "Base: SNR=2.0, rho=0.7, G=5");


    // ══════════════════════════════════════════════════════════════════════════
    //  Sweep 4: G (number of active groups)
    // ══════════════════════════════════════════════════════════════════════════

    run_sweep(
        "block_G", "G",
        {1, 3, 5, 10, 15, 20, 30},
        cfg, solvers,
        [&](double G_val, unsigned seed) {
            return dgp_block_toeplitz(cfg.n, cfg.p,
                                      static_cast<int>(G_val), cfg.g,
                                      cfg.rho, cfg.snr, seed);
        },
        "Base: SNR=2.0, rho=0.7, g=5");


    std::cout << "\nAll sweeps complete.\n";
    return 0;
}
