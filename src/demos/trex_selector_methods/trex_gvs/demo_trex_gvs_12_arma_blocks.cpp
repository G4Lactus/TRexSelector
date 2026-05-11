// ==============================================================================
// demo_trex_gvs_12_arma_blocks.cpp
// ==============================================================================
/**
 * @file demo_trex_gvs_12_arma_blocks.cpp
 *
 * @brief Single-trial sweep demo: T-Rex+GVS on heterogeneous ARMA blocks.
 *
 * @details
 *  Two sweep parts:
 *    demo_arma_snr_sweep()    — SNR in {0.1,...,5.0}, fixed ar_coef=0.8
 *    demo_arma_ar_coef_sweep() — ar_coef in {0.10,...,0.95}, fixed SNR=2.0
 *
 *  DGP: make_arma_blocks_dgp (4 blocks {20,50,80,65}; het ARMA structures).
 *  MC simulations are in sim_trex_gvs_12_arma_blocks.cpp.
 */
// ==============================================================================

#include "trex_gvs_dgps.hpp"
#include "trex_gvs_simulation_utils.hpp"

// ==============================================================================

using namespace gvs_demo;

// ==============================================================================
//  Part 1 — SNR sweep
// ==============================================================================

static void demo_arma_snr_sweep()
{
    constexpr int         n        = 200;
    constexpr int         p        = 500;
    constexpr double      tFDR     = 0.1;
    constexpr std::size_t K        = 20;
    constexpr unsigned    seed     = 2026;
    constexpr double      corr_max = 0.98;
    constexpr double      ar_coef  = 0.8;

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
        "Part 1: SNR Sweep  |  ARMA-Blocks\n"
        "n=" + std::to_string(n) + "  p=" + std::to_string(p) +
        "  s=150  ar_coef=0.8  tFDR=" + std::to_string(tFDR).substr(0, 3));

    for (double snr : snr_grid) {
        auto dat = make_arma_blocks_dgp(n, p, snr, ar_coef, seed);

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(),
                                           dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        const std::string lbl =
            "ARMA-Blocks  snr=" + std::to_string(snr).substr(0, 3);

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
//  Part 2 — ar_coef sweep
// ==============================================================================

static void demo_arma_ar_coef_sweep()
{
    constexpr int         n         = 200;
    constexpr int         p         = 500;
    constexpr double      fixed_snr = 2.0;
    constexpr double      tFDR      = 0.1;
    constexpr std::size_t K         = 20;
    constexpr unsigned    seed      = 2026;
    constexpr double      corr_max  = 0.98;

    const std::vector<double> ar_coef_grid = {
        0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95};

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
        "Part 2: ar_coef Sweep  |  ARMA-Blocks\n"
        "n=" + std::to_string(n) + "  p=" + std::to_string(p) +
        "  SNR=" + std::to_string(fixed_snr).substr(0, 3) +
        "  tFDR=" + std::to_string(tFDR).substr(0, 3));

    for (double ar_coef : ar_coef_grid) {
        auto dat = make_arma_blocks_dgp(n, p, fixed_snr, ar_coef, seed);

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(),
                                           dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        const std::string lbl =
            "ARMA-Blocks  ar_coef=" + std::to_string(ar_coef).substr(0, 4);

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

    demo_arma_snr_sweep();
    demo_arma_ar_coef_sweep();

    std::cout << "ARMA-Blocks GVS demo complete.\n";
    return 0;
}
