// ==============================================================================
// sim_trex_gvs_09_equicorr_blocks.cpp
// ==============================================================================
/**
 * @file sim_trex_gvs_09_equicorr_blocks.cpp
 *
 * @brief MC simulation: T-Rex+GVS on Gaussian equicorrelated blocks.
 *
 * @details
 *  Three parts, each in its own function:
 *    sim_equicorr_part1() — SNR sweep (fixed rho = 0.75)
 *    sim_equicorr_part2() — rho sweep (fixed SNR = 2.0)
 *    sim_equicorr_part3() — 2-D SNR × rho grid
 *
 *  DGP: make_mixed_blocks_dgp — 4 blocks {20,50,80,65}; blocks 0-2 active
 *       (beta=3, s=150); block 3 is an equicorrelated trap (beta=0).
 *  Selector: K=20, tFDR=0.1, corr_max=0.98, single-linkage HAC, GCV lambda2.
 *  MC: 200 trials per grid point.
 */
// ==============================================================================

#include "trex_gvs_dgps.hpp"
#include "trex_gvs_simulation_utils.hpp"

// ==============================================================================

using namespace gvs_demo;

// ==============================================================================
//  Part 1 — SNR sweep  (fixed rho = 0.75)
// ==============================================================================

static void sim_equicorr_part1(const GVSSimConfig& cfg, const std::string& tag)
{
    const std::vector<double> snr_grid = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0};

    // Fixed rho=0.75 => sd_x = sqrt((1-rho)/rho) = sqrt(1/3)
    const double fixed_rho = 0.75;
    const double fixed_sd_x = std::sqrt((1.0 - fixed_rho) / fixed_rho);

    GVSDGPFactory snr_fn = [&cfg, fixed_sd_x](double snr, unsigned seed) {
        return make_mixed_blocks_dgp(cfg.n, cfg.p, snr, fixed_sd_x, seed);
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
        "Part 1: SNR Sweep  |  " + tag +
        "\nn=" + std::to_string(cfg.n) + "  p=" + std::to_string(cfg.p) +
        "  s=150  MC=" + std::to_string(cfg.num_MC) +
        "  tFDR=" + std::to_string(cfg.tFDR).substr(0, 3) +
        "  rho=0.75 (fixed)");

    auto en_snr  = run_gvs_snr_sweep(
        snr_fn, snr_grid, cfg, gvs_ctrl_en,  trex_ctrl, "EN");
    auto ien_snr = run_gvs_snr_sweep(
        snr_fn, snr_grid, cfg, gvs_ctrl_ien, trex_ctrl, "IEN");

    print_mc_snr_table(tag, snr_grid, en_snr, ien_snr, cfg);
}


// ==============================================================================
//  Part 2 — rho sweep  (fixed SNR = 2.0)
// ==============================================================================

static void sim_equicorr_part2(const GVSSimConfig& cfg, const std::string& tag)
{
    const std::vector<double> rho_grid = {
        0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99};

    GVSRhoDGPFactory rho_fn = [&cfg](double rho, unsigned seed) {
        const double sd_x = std::sqrt((1.0 - rho) / rho);
        return make_mixed_blocks_dgp(cfg.n, cfg.p, cfg.snr, sd_x, seed);
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
        "Part 2: rho Sweep  |  " + tag +
        "  (SNR=" + std::to_string(cfg.snr).substr(0, 3) + ")");

    auto en_rho  = run_gvs_rho_sweep(
        rho_fn, rho_grid, cfg, gvs_ctrl_en,  trex_ctrl, "EN");
    auto ien_rho = run_gvs_rho_sweep(
        rho_fn, rho_grid, cfg, gvs_ctrl_ien, trex_ctrl, "IEN");

    print_mc_rho_table(tag, rho_grid, en_rho, ien_rho, cfg);
}


// ==============================================================================
//  Part 3 — 2-D SNR × rho sweep
// ==============================================================================

static void sim_equicorr_part3(const GVSSimConfig& cfg, const std::string& tag)
{
    const std::vector<double> snr_grid_2d = {0.2, 0.5, 1.0, 2.0, 5.0};
    const std::vector<double> rho_grid_2d = {0.30, 0.50, 0.70, 0.85, 0.90, 0.99};

    GVS2DDGPFactory dgp_2d = [&cfg](double snr, double rho, unsigned seed) {
        const double sd_x = std::sqrt((1.0 - rho) / rho);
        return make_mixed_blocks_dgp(cfg.n, cfg.p, snr, sd_x, seed);
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

    cdiag::print_section_header("Part 3: 2-D SNR x rho  |  " + tag);

    auto en_2d  = run_gvs_2d_sweep(dgp_2d, snr_grid_2d, rho_grid_2d, cfg,
                                    gvs_ctrl_en,  trex_ctrl, "EN");
    auto ien_2d = run_gvs_2d_sweep(dgp_2d, snr_grid_2d, rho_grid_2d, cfg,
                                    gvs_ctrl_ien, trex_ctrl, "IEN");

    std::vector<std::string> snr_labels, rho_labels;
    for (double s : snr_grid_2d)
        snr_labels.push_back("snr=" + std::to_string(s).substr(0, 4));
    for (double r : rho_grid_2d)
        rho_labels.push_back("rho=" + std::to_string(r).substr(0, 4));

    print_mc_matrix("mean_FDP [EN]",  snr_labels, rho_labels, en_2d,  false);
    print_mc_matrix("mean_TPP [EN]",  snr_labels, rho_labels, en_2d,  true);
    print_mc_matrix("mean_FDP [IEN]", snr_labels, rho_labels, ien_2d, false);
    print_mc_matrix("mean_TPP [IEN]", snr_labels, rho_labels, ien_2d, true);
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
    cfg.sd_x      = std::sqrt(1.0 / 3.0);   // rho=0.75 equivalent
    cfg.tFDR      = 0.1;
    cfg.K         = 20;
    cfg.num_MC    = 200;
    cfg.base_seed = 2026;
    cfg.corr_max  = 0.98;
    cfg.snr       = 2.0;    // fixed SNR for Part 2 (rho sweep)

    const std::string scenario_tag = "Equicorr-Blocks";

    sim_equicorr_part1(cfg, scenario_tag);
    sim_equicorr_part2(cfg, scenario_tag);
    sim_equicorr_part3(cfg, scenario_tag);

    std::cout << scenario_tag << " GVS MC simulations complete.\n";
    return 0;
}
