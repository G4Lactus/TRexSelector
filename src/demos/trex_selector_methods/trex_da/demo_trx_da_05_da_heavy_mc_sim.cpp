// ==============================================================================
// sim_trx_04_da_heavy_mc.cpp
// ==============================================================================
/**
 * @file sim_trx_04_da_heavy_mc.cpp
 *
 * @brief DA-TRex MC simulation: heavy-tailed X and/or noise variants on
 *        block-diagonal Toeplitz data.
 *
 * @details
 *  Base setting: n=150, p=500, G=5, g=5, rho=0.8, tFDR=0.2.
 *
 *  Four scenarios (crossed heavy_X × heavy_noise):
 *    1. Gaussian X,    Gaussian noise   (baseline)
 *    2. Heavy X (t_3), Gaussian noise
 *    3. Gaussian X,    heavy noise (t_3)
 *    4. Heavy X (t_3), heavy noise (t_3)
 *
 *  Each scenario is swept over SNR in {0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0}.
 *  200 MC trials per grid point, 3 solvers (TLARS, TAFS, TOMP), DA method BT.
 *
 *  Reference: Sim_04.cpp (stub file).
 */
// ==============================================================================

#include "da_trex_sim_common.hpp"

using namespace da_sim;


// ==============================================================================
// Scenario descriptor
// ==============================================================================

struct HeavyScenario {
    bool        heavy_X;
    bool        heavy_noise;
    std::string name;
};


// ==============================================================================
// Main
// ==============================================================================

int main() {

    std::cout.setf(std::ios::unitbuf);

    BlockSimConfig cfg;
    cfg.rho = 0.8;  // slightly higher base rho for the heavy-tailed study
    auto solvers = default_solvers();

    const std::vector<double> snr_values = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0};

    const std::vector<HeavyScenario> scenarios = {
        {false, false, "Gauss_X__Gauss_noise"},
        {true,  false, "Heavy_X__Gauss_noise"},
        {false, true,  "Gauss_X__Heavy_noise"},
        {true,  true,  "Heavy_X__Heavy_noise"},
    };

    cdianostics::print_section_header(
        "DA-TRex Heavy-Tail MC  |  MC=" + std::to_string(cfg.num_MC)
        + "  n=" + std::to_string(cfg.n) + "  p=" + std::to_string(cfg.p)
        + "  rho=" + std::to_string(cfg.rho));


    for (const auto& scenario : scenarios) {

        cdianostics::print_section_header(
            "Scenario: " + scenario.name + "  |  SNR sweep");

        const auto ngrid = static_cast<Eigen::Index>(snr_values.size());
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

            for (std::size_t gi = 0; gi < snr_values.size(); ++gi) {
                const double snr_val = snr_values[gi];

                std::stringstream lbl;
                lbl << "SNR = " << std::fixed << std::setprecision(2) << snr_val;

                auto dgp_factory = [&](unsigned seed) -> DGPData {
                    return dgp_block_toeplitz_heavy(
                        cfg.n, cfg.p, cfg.G, cfg.g,
                        cfg.rho, snr_val, seed,
                        scenario.heavy_X, scenario.heavy_noise);
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
            "heavy_" + scenario.name, "SNR", snr_values, cfg.num_MC,
            solvers, fdr_map, tpr_map, avg_L_map, avg_T_map,
            "heavy_X=" + std::string(scenario.heavy_X ? "true" : "false")
            + "  heavy_noise=" + std::string(scenario.heavy_noise ? "true" : "false")
            + "  rho=0.8, G=5, g=5");
    }


    std::cout << "\nAll heavy-tail scenarios complete.\n";
    return 0;
}
