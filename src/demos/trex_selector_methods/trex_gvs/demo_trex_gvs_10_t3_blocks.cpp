// ==============================================================================
// demo_trex_gvs_10_t3_blocks.cpp
// ==============================================================================
/**
 * @file demo_trex_gvs_10_t3_blocks.cpp
 *
 * @brief Single-trial sweep demo: T-Rex+GVS on heavy-tailed equicorr blocks.
 *
 * @details
 *  Two sweep parts:
 *    demo_t3_snr_sweep() — SNR in {0.1,0.2,0.5,1.0,2.0,5.0}, fixed rho=0.75
 *    demo_t3_rho_sweep() — rho in {0.10,...,0.99}, fixed SNR=2.0
 *
 *  DGP: make_t3_equicorr_blocks_dgp (same 4-block structure; t(3) noise).
 *  MC simulations are in sim_trex_gvs_10_t3_blocks.cpp.
 */
// ==============================================================================

#include "trex_gvs_dgps.hpp"
#include "trex_gvs_simulation_utils.hpp"

// ==============================================================================

using namespace gvs_demo;

// ==============================================================================
//  Part 1 — SNR sweep
// ==============================================================================

static void demo_t3_snr_sweep()
{
    constexpr int         n        = 200;
    constexpr int         p        = 500;
    constexpr double      tFDR     = 0.1;
    constexpr std::size_t K        = 20;
    constexpr unsigned    seed     = 2026;
    constexpr double      corr_max = 0.98;
    constexpr double      rho      = 0.75;

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
        "Part 1: SNR Sweep  |  t3-Blocks\n"
        "n=" + std::to_string(n) + "  p=" + std::to_string(p) +
        "  s=150  rho=0.75  t(3)-noise  tFDR=" +
        std::to_string(tFDR).substr(0, 3));

    for (double snr : snr_grid) {
        auto dat = make_t3_equicorr_blocks_dgp(n, p, snr, rho, seed);

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(),
                                           dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        const std::string lbl =
            "t3-Blocks  snr=" + std::to_string(snr).substr(0, 3);

        auto en_res = run_gvs_single(X_map, y_map, dat.true_support, tFDR,
                                      gvs_ctrl_en, trex_ctrl,
                                      static_cast<int>(seed));
        print_single_run_result(lbl + "  [EN]",  dat, en_res,  tFDR);

        auto ien_res = run_gvs_single(X_map, y_map, dat.true_support, tFDR,
                                       gvs_ctrl_ien, trex_ctrl,
                                       static_cast<int>(seed));
        print_single_run_result(lbl + "  [IEN]", dat, ien_res, tFDR);
    }
}


// ==============================================================================
//  Part 2 — rho sweep
// ==============================================================================

static void demo_t3_rho_sweep()
{
    constexpr int         n         = 200;
    constexpr int         p         = 500;
    constexpr double      fixed_snr = 2.0;
    constexpr double      tFDR      = 0.1;
    constexpr std::size_t K         = 20;
    constexpr unsigned    seed      = 2026;
    constexpr double      corr_max  = 0.98;

    const std::vector<double> rho_grid = {
        0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99};

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
        "Part 2: rho Sweep  |  t3-Blocks\n"
        "n=" + std::to_string(n) + "  p=" + std::to_string(p) +
        "  SNR=" + std::to_string(fixed_snr).substr(0, 3) +
        "  tFDR=" + std::to_string(tFDR).substr(0, 3));

    for (double rho : rho_grid) {
        auto dat = make_t3_equicorr_blocks_dgp(n, p, fixed_snr, rho, seed);

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(),
                                           dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        const std::string lbl =
            "t3-Blocks  rho=" + std::to_string(rho).substr(0, 4);

        auto en_res = run_gvs_single(X_map, y_map, dat.true_support, tFDR,
                                      gvs_ctrl_en, trex_ctrl,
                                      static_cast<int>(seed));
        print_single_run_result(lbl + "  [EN]",  dat, en_res,  tFDR);

        auto ien_res = run_gvs_single(X_map, y_map, dat.true_support, tFDR,
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

    demo_t3_snr_sweep();
    demo_t3_rho_sweep();

    std::cout << "t3-Blocks GVS demo complete.\n";
    return 0;
}
