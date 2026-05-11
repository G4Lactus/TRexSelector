// ==============================================================================
// demo_trex_gvs_05_random_blocks.cpp
// ==============================================================================
/**
 * @file demo_trex_gvs_05_random_blocks.cpp
 *
 * @brief Single-trial sweep demo of T-Rex+GVS on the random-blocks DGP.
 *
 * @details
 *  Two sweep parts, each in its own function:
 *
 *    demo_random_blocks_snr_sweep() — SNR in {0.1, 0.2, 0.5, 1.0, 2.0, 5.0},
 *                                     fixed sd_x = sqrt(0.01) (rho ≈ 0.99)
 *
 *    demo_random_blocks_rho_sweep() — rho in {0.10, 0.20, ..., 0.90, 0.99},
 *                                     sd_x = sqrt((1-rho)/rho), fixed SNR = 2.0
 *
 *  For each grid point one trial is run (seed = 2026) for both EN and IEN.
 *
 *  DGP: 3 blocks {20, 50, 80} in random order with noise gaps (s = 150).
 *  Selector: K=20, tFDR=0.1, corr_max=0.98, single-linkage HAC, GCV lambda2.
 *
 *  Monte Carlo simulations are in sim_trex_gvs_05_random_blocks.cpp.
 */
// ==============================================================================

// project includes
#include "trex_gvs_dgps.hpp"
#include "trex_gvs_simulation_utils.hpp"


// ==============================================================================

using namespace gvs_demo;

// ==============================================================================
//  Part 1 — SNR sweep
// ==============================================================================

static void demo_random_blocks_snr_sweep()
{
    constexpr int         n        = 200;
    constexpr int         p        = 500;
    constexpr double      sd_x     = 0.1;    // sqrt(0.01) => rho ≈ 0.99
    constexpr double      tFDR     = 0.1;
    constexpr std::size_t K        = 20;
    constexpr unsigned    seed     = 2026;
    constexpr double      corr_max = 0.98;

    const std::vector<double> snr_grid = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0};

    auto trex_ctrl = make_gvs_trex_control(K);

    TRexGVSControlParameter gvs_ctrl_en;
    gvs_ctrl_en.gvs_type       = GVSType::EN;
    gvs_ctrl_en.lambda2_method = LambdaSelectionMethod::GCV;
    gvs_ctrl_en.corr_max       = corr_max;
    gvs_ctrl_en.hc_linkage     = hac::LinkageMethod::Single;

    TRexGVSControlParameter gvs_ctrl_ien;
    gvs_ctrl_ien.gvs_type       = GVSType::IEN;
    gvs_ctrl_ien.lambda2_method = LambdaSelectionMethod::GCV;
    gvs_ctrl_ien.corr_max       = corr_max;
    gvs_ctrl_ien.hc_linkage     = hac::LinkageMethod::Single;

    cdiag::print_section_header(
        "Part 1: SNR Sweep  |  Random-Blocks\n"
        "n=" + std::to_string(n) + "  p=" + std::to_string(p) +
        "  blocks={20,50,80}  sd_x=0.1 (rho≈0.99)"
        "  tFDR=" + std::to_string(tFDR).substr(0, 3));

    for (double snr : snr_grid) {
        auto dat = make_random_blocks_dgp(n, p, snr, sd_x, seed);

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(),
                                           dat.X.rows(), dat.X.cols());

        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        const std::string lbl = "Random-Blocks  snr=" + std::to_string(snr).substr(0, 3);

        auto en_res = run_gvs_single(X_map, y_map,
            dat.true_support, tFDR,
            gvs_ctrl_en, trex_ctrl,
            static_cast<int>(seed));

        print_single_run_result(lbl + "  [EN]",  dat, en_res,  tFDR);

        auto ien_res = run_gvs_single(X_map, y_map,
            dat.true_support, tFDR,
            gvs_ctrl_ien, trex_ctrl,
            static_cast<int>(seed));

        print_single_run_result(lbl + "  [IEN]", dat, ien_res, tFDR);
    }
}


// ==============================================================================
//  Part 2 — rho sweep
// ==============================================================================

static void demo_random_blocks_rho_sweep()
{
    constexpr int         n         = 200;
    constexpr int         p         = 500;
    constexpr double      fixed_snr = 2.0;
    constexpr double      tFDR      = 0.1;
    constexpr std::size_t K         = 20;
    constexpr unsigned    seed      = 2026;
    constexpr double      corr_max  = 0.98;

    const std::vector<double> rho_grid = {
        0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
        0.90, 0.99};

    auto trex_ctrl = make_gvs_trex_control(K);

    TRexGVSControlParameter gvs_ctrl_en;
    gvs_ctrl_en.gvs_type       = GVSType::EN;
    gvs_ctrl_en.lambda2_method = LambdaSelectionMethod::GCV;
    gvs_ctrl_en.corr_max       = corr_max;
    gvs_ctrl_en.hc_linkage     = hac::LinkageMethod::Single;

    TRexGVSControlParameter gvs_ctrl_ien;
    gvs_ctrl_ien.gvs_type       = GVSType::IEN;
    gvs_ctrl_ien.lambda2_method = LambdaSelectionMethod::GCV;
    gvs_ctrl_ien.corr_max       = corr_max;
    gvs_ctrl_ien.hc_linkage     = hac::LinkageMethod::Single;

    cdiag::print_section_header(
        "Part 2: rho Sweep  |  Random-Blocks\n"
        "n=" + std::to_string(n) + "  p=" + std::to_string(p) +
        "  SNR=" + std::to_string(fixed_snr).substr(0, 3) +
        "  tFDR=" + std::to_string(tFDR).substr(0, 3));

    for (double rho : rho_grid) {
        const double sd_x = std::sqrt((1.0 - rho) / rho);
        auto dat = make_random_blocks_dgp(n, p, fixed_snr, sd_x, seed);

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(),
                                           dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        const std::string lbl = "Random-Blocks  rho=" + std::to_string(rho).substr(0, 4);

        auto en_res = run_gvs_single(X_map, y_map,
            dat.true_support, tFDR,
            gvs_ctrl_en, trex_ctrl,
            static_cast<int>(seed));

        print_single_run_result(lbl + "  [EN]",  dat, en_res,  tFDR);

        auto ien_res = run_gvs_single(X_map, y_map,
            dat.true_support, tFDR,
            gvs_ctrl_ien, trex_ctrl,
            static_cast<int>(seed));

        print_single_run_result(lbl + "  [IEN]", dat, ien_res, tFDR);
    }
}


// ==============================================================================
//  main
// ==============================================================================

int main()
{
    std::cout.setf(std::ios::unitbuf);

    demo_random_blocks_snr_sweep();
    demo_random_blocks_rho_sweep();

    std::cout << "Random-Blocks GVS demo complete.\n";
    return 0;
}
