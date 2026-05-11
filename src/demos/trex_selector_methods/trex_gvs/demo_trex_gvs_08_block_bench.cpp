// ==============================================================================
// demo_trex_gvs_08_block_bench.cpp
// ==============================================================================
/**
 * @file demo_trex_gvs_08_block_bench.cpp
 *
 * @brief Single-trial benchmark: T-Rex+GVS on the block-equicorrelated DGP.
 *
 * @details
 *  Runs one dataset per scenario (INDIVIDUAL, REPRESENTATIVE, WHOLE) at a
 *  fixed (snr=2.0, rho=0.5) operating point and displays all four method
 *  variants side by side:
 *
 *    M1  T-Rex + EN,  HAC-clustered groups
 *    M2  T-Rex + EN,  oracle groups
 *    M3  T-Rex + IEN, HAC-clustered groups
 *    M4  T-Rex + IEN, oracle groups
 *
 *  DGP:  n=200, p=500, G=100 blocks of size 5, 10 active blocks, b=3.
 *  Selector: K=20, tFDR=0.1, corr_max=0.5, single-linkage HAC, GCV lambda2.
 *
 *  Full Monte Carlo simulations are in sim_trex_gvs_08_block_bench.cpp.
 */
// ==============================================================================

#include "trex_gvs_block_bench_utils.hpp"


// ==============================================================================

using namespace gvs_demo;


// ==============================================================================
// Fixed constants
// ==============================================================================

static constexpr int         N              = 200;
static constexpr int         P              = 500;
static constexpr int         G              = 100;
static constexpr int         BLOCK_SIZE     = 5;
static constexpr int         N_ACTIVE       = 10;
static constexpr double      TFDR           = 0.1;
static constexpr std::size_t K              = 20;
static constexpr double      CORR_MAX       = 0.5;
static constexpr unsigned    SEED           = 2026;
static constexpr double      SNR            = 2.0;
static constexpr double      RHO            = 0.5;
static constexpr double      B              = 3.0;


// ==============================================================================
// One demo section per scenario
// ==============================================================================

static void run_scenario(BlockScenario scenario, const std::string& tag)
{
    cdiag::print_section_header(
        "Block Bench — " + tag +
        "\nn=" + std::to_string(N) +
        "  p=" + std::to_string(P) +
        "  G=" + std::to_string(G) +
        "  block=" + std::to_string(BLOCK_SIZE) +
        "  active=" + std::to_string(N_ACTIVE) +
        "  SNR=" + std::to_string(SNR).substr(0, 3) +
        "  rho=" + std::to_string(RHO).substr(0, 3) +
        "  tFDR=" + std::to_string(TFDR).substr(0, 3));

    auto dat = make_block_equicorr_dgp(N, P, G, BLOCK_SIZE, N_ACTIVE,
                                        RHO, SNR, scenario,
                                        SEED, B);

    const auto trex_ctrl = make_gvs_trex_control(K);

    const auto res = run_block_single(dat, G, BLOCK_SIZE, TFDR,
                                       trex_ctrl, CORR_MAX,
                                       static_cast<int>(SEED));

    print_block_trial_result(tag, dat, res, TFDR, G, BLOCK_SIZE, RHO);
}


// ==============================================================================
// main
// ==============================================================================

int main()
{
    run_scenario(BlockScenario::INDIVIDUAL,    "INDIVIDUAL");
    run_scenario(BlockScenario::REPRESENTATIVE, "REPRESENTATIVE");
    run_scenario(BlockScenario::WHOLE,          "WHOLE");

    return 0;
}
