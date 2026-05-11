// ==============================================================================
// sim_trex_gvs_13_hapgen.cpp
// ==============================================================================
/**
 * @file sim_trex_gvs_13_hapgen.cpp
 *
 * @brief MC simulation: T-Rex+GVS on the quasi-hapgen LD-block DGP.
 *
 * @details
 *  Three parts, each in its own function:
 *    sim_hapgen_part1() — SNR sweep (fixed rho_scale = 1.0)
 *    sim_hapgen_part2() — rho_scale sweep (fixed SNR = 2.0)
 *    sim_hapgen_part3() — 2-D SNR × rho_scale grid
 *
 *  DGP: make_hapgen_dgp — fixed p=500, 7 irregular LD blocks with heterogeneous
 *       AR(1) within-block correlations and cross-block long-range LD.
 *       s=130, B1/B3/B4 active (beta=3), B2/B5/B6/B7 are traps.
 *
 *  Performance: make_hapgen_cholesky() performs an O(p^3) eigenvalue
 *  decomposition and is called ONCE per rho_scale value; the resulting
 *  Cholesky factor is reused across all MC trials and SNR values.
 *
 *  Selector: K=20, tFDR=0.1, corr_max=0.98, single-linkage HAC, GCV lambda2.
 *  MC: 200 trials per grid point.
 */
// ==============================================================================

#include "trex_gvs_dgps.hpp"
#include "trex_gvs_simulation_utils.hpp"

// ==============================================================================

using namespace gvs_demo;

// ==============================================================================
//  Helper: build GVS control parameters
// ==============================================================================

static TRexGVSControlParameter make_hapgen_gvs_ctrl(GVSType type,
                                                      double corr_max)
{
    TRexGVSControlParameter ctrl;
    ctrl.gvs_type       = type;
    ctrl.lambda2_method = LambdaSelectionMethod::GCV;
    ctrl.corr_max       = corr_max;
    ctrl.hc_linkage     = hac::LinkageMethod::Single;
    return ctrl;
}

// ==============================================================================
//  Part 1 — SNR sweep  (fixed rho_scale = 1.0)
// ==============================================================================

static void sim_hapgen_part1(const GVSSimConfig& cfg, const std::string& tag)
{
    const std::vector<double> snr_grid = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0};
    constexpr double fixed_rho_scale = 1.0;

    // Compute Cholesky factor once for the fixed rho_scale
    const Eigen::MatrixXd L = make_hapgen_cholesky(fixed_rho_scale);

    auto trex_ctrl    = make_gvs_trex_control(cfg.K);
    auto gvs_ctrl_en  = make_hapgen_gvs_ctrl(GVSType::EN,  cfg.corr_max);
    auto gvs_ctrl_ien = make_hapgen_gvs_ctrl(GVSType::IEN, cfg.corr_max);

    cdiag::print_section_header(
        "Part 1: SNR Sweep  |  " + tag +
        "\nn=" + std::to_string(cfg.n) + "  p=500" +
        "  s=130  MC=" + std::to_string(cfg.num_MC) +
        "  tFDR=" + std::to_string(cfg.tFDR).substr(0, 3) +
        "  rho_scale=1.0 (fixed)");

    std::vector<GVSGridPointResult> en_snr, ien_snr;
    en_snr.reserve(snr_grid.size());
    ien_snr.reserve(snr_grid.size());

    for (double snr : snr_grid) {
        // Factory: L captured by ref (const, thread-safe); snr by value.
        auto dgp_en = [&cfg, &L, snr](double /*param*/, unsigned seed) {
            return make_hapgen_dgp(cfg.n, snr, L, seed);
        };
        en_snr.push_back(run_gvs_mc_trials(
            dgp_en, snr, cfg.num_MC, "EN",
            cfg.tFDR, gvs_ctrl_en, trex_ctrl,
            static_cast<unsigned>(cfg.base_seed)));

        auto dgp_ien = [&cfg, &L, snr](double /*param*/, unsigned seed) {
            return make_hapgen_dgp(cfg.n, snr, L, seed);
        };
        ien_snr.push_back(run_gvs_mc_trials(
            dgp_ien, snr, cfg.num_MC, "IEN",
            cfg.tFDR, gvs_ctrl_ien, trex_ctrl,
            static_cast<unsigned>(cfg.base_seed)));
    }

    print_mc_snr_table(tag, snr_grid, en_snr, ien_snr, cfg);
}


// ==============================================================================
//  Part 2 — rho_scale sweep  (fixed SNR = 2.0)
// ==============================================================================

static void sim_hapgen_part2(const GVSSimConfig& cfg, const std::string& tag)
{
    const std::vector<double> rs_grid = {0.15, 0.30, 0.50, 0.70, 0.85, 1.00};

    auto trex_ctrl    = make_gvs_trex_control(cfg.K);
    auto gvs_ctrl_en  = make_hapgen_gvs_ctrl(GVSType::EN,  cfg.corr_max);
    auto gvs_ctrl_ien = make_hapgen_gvs_ctrl(GVSType::IEN, cfg.corr_max);

    cdiag::print_section_header(
        "Part 2: rho_scale Sweep  |  " + tag +
        "  (SNR=" + std::to_string(cfg.snr).substr(0, 3) + ")");

    std::vector<GVSGridPointResult> en_param, ien_param;
    en_param.reserve(rs_grid.size());
    ien_param.reserve(rs_grid.size());

    for (double rs : rs_grid) {
        // Compute Cholesky once per rho_scale value
        const Eigen::MatrixXd L = make_hapgen_cholesky(rs);

        auto dgp_en = [&cfg, &L](double /*param*/, unsigned seed) {
            return make_hapgen_dgp(cfg.n, cfg.snr, L, seed);
        };
        en_param.push_back(run_gvs_mc_trials(
            dgp_en, rs, cfg.num_MC, "EN",
            cfg.tFDR, gvs_ctrl_en, trex_ctrl,
            static_cast<unsigned>(cfg.base_seed)));

        auto dgp_ien = [&cfg, &L](double /*param*/, unsigned seed) {
            return make_hapgen_dgp(cfg.n, cfg.snr, L, seed);
        };
        ien_param.push_back(run_gvs_mc_trials(
            dgp_ien, rs, cfg.num_MC, "IEN",
            cfg.tFDR, gvs_ctrl_ien, trex_ctrl,
            static_cast<unsigned>(cfg.base_seed)));
    }

    print_mc_param_sweep_table(tag, "rho_sc", rs_grid, en_param, ien_param, cfg);
}


// ==============================================================================
//  Part 3 — 2-D SNR × rho_scale sweep
// ==============================================================================

static void sim_hapgen_part3(const GVSSimConfig& cfg, const std::string& tag)
{
    const std::vector<double> snr_grid_2d = {0.2, 0.5, 1.0, 2.0, 5.0};
    const std::vector<double> rs_grid_2d  = {0.15, 0.30, 0.50, 0.70, 1.00};

    const std::size_t n_snr = snr_grid_2d.size();
    const std::size_t n_rs  = rs_grid_2d.size();

    auto trex_ctrl    = make_gvs_trex_control(cfg.K);
    auto gvs_ctrl_en  = make_hapgen_gvs_ctrl(GVSType::EN,  cfg.corr_max);
    auto gvs_ctrl_ien = make_hapgen_gvs_ctrl(GVSType::IEN, cfg.corr_max);

    cdiag::print_section_header("Part 3: 2-D SNR x rho_scale  |  " + tag);

    // en_2d[i_snr][i_rs], ien_2d[i_snr][i_rs]
    std::vector<std::vector<GVSGridPointResult>>
        en_2d(n_snr,  std::vector<GVSGridPointResult>(n_rs)),
        ien_2d(n_snr, std::vector<GVSGridPointResult>(n_rs));

    // Outer loop over rho_scale: compute L once per rs, inner loop over SNR.
    for (std::size_t i_rs = 0; i_rs < n_rs; ++i_rs) {
        const double rs = rs_grid_2d[i_rs];
        const Eigen::MatrixXd L = make_hapgen_cholesky(rs);

        for (std::size_t i_snr = 0; i_snr < n_snr; ++i_snr) {
            const double snr = snr_grid_2d[i_snr];
            const std::size_t cell_idx = i_rs * n_snr + i_snr + 1;
            const std::size_t total_cells = n_rs * n_snr;

            std::cout << "  [EN/IEN] [" << cell_idx << "/" << total_cells << "]"
                      << "  SNR=" << std::fixed << std::setprecision(2) << snr
                      << "  rho_scale=" << rs
                      << "  running " << cfg.num_MC << " MC trials...\n"
                      << std::flush;

            auto dgp_en = [&cfg, &L, snr](double /*param*/, unsigned seed) {
                return make_hapgen_dgp(cfg.n, snr, L, seed);
            };
            en_2d[i_snr][i_rs] = run_gvs_mc_trials(
                dgp_en, snr, cfg.num_MC, "EN",
                cfg.tFDR, gvs_ctrl_en, trex_ctrl,
                static_cast<unsigned>(cfg.base_seed));

            auto dgp_ien = [&cfg, &L, snr](double /*param*/, unsigned seed) {
                return make_hapgen_dgp(cfg.n, snr, L, seed);
            };
            ien_2d[i_snr][i_rs] = run_gvs_mc_trials(
                dgp_ien, snr, cfg.num_MC, "IEN",
                cfg.tFDR, gvs_ctrl_ien, trex_ctrl,
                static_cast<unsigned>(cfg.base_seed));
        }
    }

    std::vector<std::string> snr_labels, rs_labels;
    for (double s : snr_grid_2d)
        snr_labels.push_back("snr=" + std::to_string(s).substr(0, 4));
    for (double r : rs_grid_2d)
        rs_labels.push_back("rs=" + std::to_string(r).substr(0, 4));

    print_mc_matrix("mean_FDP [EN]",  snr_labels, rs_labels, en_2d,  false);
    print_mc_matrix("mean_TPP [EN]",  snr_labels, rs_labels, en_2d,  true);
    print_mc_matrix("mean_FDP [IEN]", snr_labels, rs_labels, ien_2d, false);
    print_mc_matrix("mean_TPP [IEN]", snr_labels, rs_labels, ien_2d, true);
}


// ==============================================================================
//  main
// ==============================================================================

int main()
{
    std::cout.setf(std::ios::unitbuf);

    GVSSimConfig cfg;
    cfg.n         = 200;
    cfg.p         = 500;    // fixed for hapgen DGP
    cfg.sd_x      = 0.0;    // not used; structure encoded in Cholesky L
    cfg.tFDR      = 0.1;
    cfg.K         = 20;
    cfg.num_MC    = 200;
    cfg.base_seed = 2026;
    cfg.corr_max  = 0.98;
    cfg.snr       = 2.0;    // fixed SNR for Part 2 (rho_scale sweep)

    const std::string scenario_tag = "Hapgen-LD";

    sim_hapgen_part1(cfg, scenario_tag);
    sim_hapgen_part2(cfg, scenario_tag);
    sim_hapgen_part3(cfg, scenario_tag);

    std::cout << scenario_tag << " GVS MC simulations complete.\n";
    return 0;
}
