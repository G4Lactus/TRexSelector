// ==============================================================================
// sim_trex_gvs_08_block_bench.cpp
// ==============================================================================
/**
 * @file sim_trex_gvs_08_block_bench.cpp
 *
 * @brief Monte Carlo benchmark: T-Rex+GVS on the block-equicorrelated DGP.
 *
 * @details
 *  Tests four method variants over a (snr x rho) grid for each of three
 *  truth scenarios:
 *
 *    M1  T-Rex + EN,  HAC-clustered groups
 *    M2  T-Rex + EN,  oracle groups
 *    M3  T-Rex + IEN, HAC-clustered groups
 *    M4  T-Rex + IEN, oracle groups
 *
 *  Scenarios:
 *    INDIVIDUAL     — 1 active variable per active block  (s = n_active_blocks)
 *    REPRESENTATIVE — 2–3 active variables per block      (s ≈ 2.5 * n_active_blocks)
 *    WHOLE          — all block_size variables per block  (s = n_active_blocks * block_size)
 *
 *  Fixed constants:
 *    n = 200, p = 500, G = 100, block_size = 5, n_active = 10
 *    tFDR = 0.1, K = 20, corr_max = 0.5, b = 3.0
 *    snr_grid = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0}
 *    rho_grid = {0.5, 0.9}
 *    num_MC   = 500
 *
 *  Note: IEN is known to collapse at rho ≈ 0.99 (T_stop ≈ 6, TPR ≈ 0).
 *  The rho_grid deliberately avoids this regime; rho=0.9 is expected to
 *  be safe but may show some degradation relative to EN at high rho.
 *
 *  Scientific questions addressed:
 *    Q1: When does oracle group information (M2, M4) outperform clustered (M1, M3)?
 *    Q2: Is IEN primarily a representative-from-group method?
 *    Q3: Does EN dominate IEN for whole-group recovery?
 *    Q4: At what SNR does each method become reliable?
 *    Q5: How much runtime is saved by IEN relative to EN?
 *
 *  Single-trial sanity check in demo_trex_gvs_08_block_bench.cpp.
 */
// ==============================================================================

#include "trex_gvs_block_bench_utils.hpp"

#include <sstream>
#include <string>
#include <iomanip>


// ==============================================================================

using namespace gvs_demo;


// ==============================================================================
// Simulation parameters
// ==============================================================================

static constexpr int         N              = 200;
static constexpr int         P              = 500;
static constexpr int         G              = 100;
static constexpr int         BLOCK_SIZE     = 5;
static constexpr int         N_ACTIVE       = 10;
static constexpr double      TFDR           = 0.1;
static constexpr std::size_t K              = 20;
static constexpr double      CORR_MAX       = 0.5;
static constexpr int         BASE_SEED      = 2026;
static constexpr std::size_t NUM_MC         = 500;
static constexpr double      B              = 3.0;

static const std::vector<double> SNR_GRID = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0};
static const std::vector<double> RHO_GRID = {0.5, 0.9};


// ==============================================================================
// Scenario runner
// ==============================================================================

/**
 * @brief Run the full (rho × snr) MC grid for one scenario and print results.
 *
 * Seeds are staggered by cell index so that each (rho, snr) cell uses a
 * distinct seed band and trials are reproducible regardless of grid order.
 */
static void run_scenario(BlockScenario scenario, const std::string& tag)
{
    cdiag::print_section_header(
        "Block Bench MC: " + tag +
        "\nn=" + std::to_string(N) +
        "  p=" + std::to_string(P) +
        "  G=" + std::to_string(G) +
        "  block=" + std::to_string(BLOCK_SIZE) +
        "  active=" + std::to_string(N_ACTIVE) +
        "  tFDR=" + std::to_string(TFDR).substr(0, 3) +
        "  K=" + std::to_string(K) +
        "  MC=" + std::to_string(NUM_MC) +
        "  corr_max=" + std::to_string(CORR_MAX).substr(0, 3));

    const std::size_t n_rho = RHO_GRID.size();
    const std::size_t n_snr = SNR_GRID.size();

    // results[i_rho][i_snr]
    std::vector<std::vector<BlockGridResult>> results(
        n_rho, std::vector<BlockGridResult>(n_snr));

    for (std::size_t i_rho = 0; i_rho < n_rho; ++i_rho) {
        const double rho = RHO_GRID[i_rho];

        for (std::size_t i_snr = 0; i_snr < n_snr; ++i_snr) {
            const double snr = SNR_GRID[i_snr];

            // Stagger seeds so different (rho, snr) cells never share the same
            // trial seeds.  Cell offset = (i_rho * n_snr + i_snr) * NUM_MC.
            BlockBenchConfig cfg;
            cfg.n               = N;
            cfg.p               = P;
            cfg.G               = G;
            cfg.block_size      = BLOCK_SIZE;
            cfg.n_active_blocks = N_ACTIVE;
            cfg.tFDR            = TFDR;
            cfg.K               = K;
            cfg.num_MC          = NUM_MC;
            cfg.base_seed       = BASE_SEED + static_cast<int>(
                (i_rho * n_snr + i_snr) * NUM_MC);
            cfg.corr_max        = CORR_MAX;
            cfg.b               = B;

            const auto trex_ctrl = make_gvs_trex_control(cfg.K);

            // DGP closure: captures scenario, rho, snr, and all config constants
            auto dgp_fn = [&](unsigned seed) -> GVSDGPResult {
                return make_block_equicorr_dgp(
                    cfg.n, cfg.p, cfg.G,
                    cfg.block_size, cfg.n_active_blocks,
                    rho, snr, scenario, seed, cfg.b);
            };

            // Progress label
            std::ostringstream lbl;
            lbl << tag
                << "  rho=" << std::fixed << std::setprecision(2) << rho
                << "  snr=" << snr
                << "  [" << (i_rho * n_snr + i_snr + 1) << "/"
                << (n_rho * n_snr) << "]";

            results[i_rho][i_snr] =
                run_block_mc(dgp_fn, cfg, trex_ctrl, lbl.str());
        }
    }

    // Print aggregate table
    BlockBenchConfig print_cfg;
    print_cfg.n               = N;
    print_cfg.p               = P;
    print_cfg.G               = G;
    print_cfg.block_size      = BLOCK_SIZE;
    print_cfg.n_active_blocks = N_ACTIVE;
    print_cfg.tFDR            = TFDR;
    print_cfg.K               = K;
    print_cfg.num_MC          = NUM_MC;
    print_cfg.base_seed       = BASE_SEED;
    print_cfg.corr_max        = CORR_MAX;
    print_cfg.b               = B;

    print_block_grid_table(tag, scenario, RHO_GRID, SNR_GRID, results,
                            print_cfg);
}


// ==============================================================================
// main
// ==============================================================================

int main()
{
    std::cout.setf(std::ios::unitbuf);

    run_scenario(BlockScenario::INDIVIDUAL,    "INDIVIDUAL");
    run_scenario(BlockScenario::REPRESENTATIVE, "REPRESENTATIVE");
    run_scenario(BlockScenario::WHOLE,          "WHOLE");

    return 0;
}
