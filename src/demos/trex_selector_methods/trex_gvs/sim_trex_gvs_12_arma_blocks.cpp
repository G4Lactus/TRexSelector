// ==============================================================================
// sim_trex_gvs_12_arma_blocks.cpp
// ==============================================================================
/**
 * @file sim_trex_gvs_12_arma_blocks.cpp
 *
 * @brief MC simulation: T-Rex+GVS on heterogeneous ARMA blocks.
 *
 * @details
 *  Three parts, each in its own function:
 *    sim_arma_part1() — SNR sweep (fixed ar_coef = 0.8)
 *    sim_arma_part2() — ar_coef sweep (fixed SNR = 2.0)
 *    sim_arma_part3() — 2-D SNR × ar_coef grid
 *
 *  DGP: make_arma_blocks_dgp — 4 blocks {20,50,80,65} with heterogeneous
 *       ARMA structures per block (AR1 / MA3 / ARMA21 / AR1-trap).
 *       s=150, blocks 0-2 active (beta=3), block 3 is the AR(1) trap.
 *
 *  Selector: K=20, tFDR=0.1, corr_max=0.98, single-linkage HAC, GCV lambda2.
 *
 *  MC: 200 trials per grid point.
 */
// ==============================================================================

// project includes
#include "trex_gvs_dgps.hpp"
#include "trex_gvs_simulation_utils.hpp"

// ==============================================================================

using namespace gvs_demo;

// ==============================================================================
//  Part 1 — SNR sweep  (fixed ar_coef = 0.8)
// ==============================================================================

static void sim_arma_part1(const GVSSimConfig& cfg, const std::string& tag)
{
    const std::vector<double> snr_grid = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0};
    constexpr double fixed_ar = 0.8;

    GVSDGPFactory snr_fn = [&cfg](double snr, unsigned seed) {
        return make_arma_blocks_dgp(cfg.n, cfg.p, snr, fixed_ar, seed);
    };

    auto trex_ctrl = make_gvs_trex_control(cfg.K);

    TRexGVSControlParameter gvs_ctrl_en;
    gvs_ctrl_en.gvs_type       = GVSType::EN;
    gvs_ctrl_en.lambda2_method = LambdaSelectionMethod::COND_NUM;
    gvs_ctrl_en.corr_max       = cfg.corr_max;
    gvs_ctrl_en.hc_linkage     = hac::LinkageMethod::Single;

    TRexGVSControlParameter gvs_ctrl_ien;
    gvs_ctrl_ien.gvs_type       = GVSType::IEN;
    gvs_ctrl_ien.lambda2_method = LambdaSelectionMethod::COND_NUM;
    gvs_ctrl_ien.corr_max       = cfg.corr_max;
    gvs_ctrl_ien.hc_linkage     = hac::LinkageMethod::Single;

    cdiag::print_section_header(
        "Part 1: SNR Sweep  |  " + tag +
        "\nn=" + std::to_string(cfg.n) + "  p=" + std::to_string(cfg.p) +
        "  s=150  MC=" + std::to_string(cfg.num_MC) +
        "  tFDR=" + std::to_string(cfg.tFDR).substr(0, 3) +
        "  ar_coef=0.8 (fixed)");

    auto en_snr  = run_gvs_snr_sweep(
        snr_fn, snr_grid, cfg, gvs_ctrl_en,  trex_ctrl, "EN");
    auto ien_snr = run_gvs_snr_sweep(
        snr_fn, snr_grid, cfg, gvs_ctrl_ien, trex_ctrl, "IEN");

    print_mc_snr_table(tag, snr_grid, en_snr, ien_snr, cfg);
}


// ==============================================================================
//  Part 2 — ar_coef sweep  (fixed SNR = 2.0)
// ==============================================================================

static void sim_arma_part2(const GVSSimConfig& cfg, const std::string& tag)
{
    const std::vector<double> ar_coef_grid = {
        0.10, 0.20, 0.30, 0.40,
        0.50, 0.60, 0.70,
        0.80, 0.90, 0.95};

    // GVSRhoDGPFactory: first argument is ar_coef (not rho) here.
    GVSRhoDGPFactory ac_fn = [&cfg](double ar_coef, unsigned seed) {
        return make_arma_blocks_dgp(cfg.n, cfg.p, cfg.snr, ar_coef, seed);
    };

    auto trex_ctrl = make_gvs_trex_control(cfg.K);

    TRexGVSControlParameter gvs_ctrl_en;
    gvs_ctrl_en.gvs_type       = GVSType::EN;
    gvs_ctrl_en.lambda2_method = LambdaSelectionMethod::GCV;
    gvs_ctrl_en.corr_max       = cfg.corr_max;
    gvs_ctrl_en.hc_linkage     = hac::LinkageMethod::Single;

    TRexGVSControlParameter gvs_ctrl_ien;
    gvs_ctrl_ien.gvs_type       = GVSType::IEN;
    gvs_ctrl_ien.lambda2_method = LambdaSelectionMethod::GCV;
    gvs_ctrl_ien.corr_max       = cfg.corr_max;
    gvs_ctrl_ien.hc_linkage     = hac::LinkageMethod::Single;

    cdiag::print_section_header(
        "Part 2: ar_coef Sweep  |  " + tag +
        "  (SNR=" + std::to_string(cfg.snr).substr(0, 3) + ")");

    auto en_param  = run_gvs_rho_sweep(
        ac_fn, ar_coef_grid, cfg, gvs_ctrl_en,  trex_ctrl, "EN");

    auto ien_param = run_gvs_rho_sweep(
        ac_fn, ar_coef_grid, cfg, gvs_ctrl_ien, trex_ctrl, "IEN");

    print_mc_param_sweep_table(tag, "ar_coef", ar_coef_grid,
                                en_param, ien_param, cfg);
}


// ==============================================================================
//  Part 3 — 2-D SNR × ar_coef sweep
// ==============================================================================

static void sim_arma_part3(const GVSSimConfig& cfg, const std::string& tag)
{
    const std::vector<double> snr_grid_2d     = {0.2, 0.5, 1.0, 2.0, 5.0};
    const std::vector<double> ar_coef_grid_2d = {0.30, 0.50, 0.70, 0.80,
                                                 0.90, 0.95};

    GVS2DDGPFactory dgp_2d = [&cfg](double snr, double ar_coef, unsigned seed) {
        return make_arma_blocks_dgp(cfg.n, cfg.p, snr, ar_coef, seed);
    };

    auto trex_ctrl = make_gvs_trex_control(cfg.K);

    TRexGVSControlParameter gvs_ctrl_en;
    gvs_ctrl_en.gvs_type       = GVSType::EN;
    gvs_ctrl_en.lambda2_method = LambdaSelectionMethod::GCV;
    gvs_ctrl_en.corr_max       = cfg.corr_max;
    gvs_ctrl_en.hc_linkage     = hac::LinkageMethod::Single;

    TRexGVSControlParameter gvs_ctrl_ien;
    gvs_ctrl_ien.gvs_type       = GVSType::IEN;
    gvs_ctrl_ien.lambda2_method = LambdaSelectionMethod::GCV;
    gvs_ctrl_ien.corr_max       = cfg.corr_max;
    gvs_ctrl_ien.hc_linkage     = hac::LinkageMethod::Single;

    cdiag::print_section_header("Part 3: 2-D SNR x ar_coef  |  " + tag);

    auto en_2d  = run_gvs_2d_sweep(dgp_2d, snr_grid_2d,
                                   ar_coef_grid_2d, cfg,
                                   gvs_ctrl_en,  trex_ctrl, "EN");

    auto ien_2d = run_gvs_2d_sweep(dgp_2d, snr_grid_2d,
                                   ar_coef_grid_2d, cfg,
                                   gvs_ctrl_ien, trex_ctrl, "IEN");

    std::vector<std::string> snr_labels, ac_labels;
    for (double s : snr_grid_2d)
        snr_labels.push_back("snr=" + std::to_string(s).substr(0, 4));
    for (double a : ar_coef_grid_2d)
        ac_labels.push_back("ac=" + std::to_string(a).substr(0, 4));

    print_mc_matrix("mean_FDP [EN]",  snr_labels, ac_labels,
                    en_2d,  false);

    print_mc_matrix("mean_TPP [EN]",  snr_labels, ac_labels,
                    en_2d,  true);

    print_mc_matrix("mean_FDP [IEN]", snr_labels, ac_labels,
                    ien_2d, false);

    print_mc_matrix("mean_TPP [IEN]", snr_labels, ac_labels,
                    ien_2d, true);
}


// ==============================================================================
//  main
// ==============================================================================

int main()
{
    std::cout.setf(std::ios::unitbuf);

    GVSSimConfig cfg;
    cfg.n         = 200;
    cfg.p         = 500;
    cfg.sd_x      = 0.0;    // not used; ar_coef passed directly to DGP
    cfg.tFDR      = 0.1;
    cfg.K         = 20;
    cfg.num_MC    = 200;
    cfg.base_seed = 2026;
    cfg.corr_max  = 0.98;
    cfg.snr       = 2.0;    // fixed SNR for Part 2 (ar_coef sweep)

    const std::string scenario_tag = "ARMA-Blocks";

    sim_arma_part1(cfg, scenario_tag);
    sim_arma_part2(cfg, scenario_tag);
    sim_arma_part3(cfg, scenario_tag);

    std::cout << scenario_tag << " GVS MC simulations complete.\n";
    return 0;
}
