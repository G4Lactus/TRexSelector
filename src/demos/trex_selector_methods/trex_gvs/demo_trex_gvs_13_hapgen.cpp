// ==============================================================================
// demo_trex_gvs_13_hapgen.cpp
// ==============================================================================
/**
 * @file demo_trex_gvs_13_hapgen.cpp
 *
 * @brief Single-trial sweep demo: T-Rex+GVS on quasi-hapgen LD-block DGP.
 *
 * @details
 *  Two sweep parts:
 *    demo_hapgen_snr_sweep()      — SNR in {0.1,...,5.0}, fixed rho_scale=1.0
 *    demo_hapgen_rho_scale_sweep() — rho_scale in {0.15,...,1.0}, fixed SNR=2.0
 *
 *  DGP: make_hapgen_dgp (p=500 fixed; 7 irregular LD blocks; s=130 active).
 *
 *  Performance: make_hapgen_cholesky() (O(p^3)) is called once per rho_scale
 *  value before the per-trial loop; the Cholesky factor L is reused.
 *
 *  MC simulations are in sim_trex_gvs_13_hapgen.cpp.
 */
// ==============================================================================

#include "trex_gvs_dgps.hpp"
#include "trex_gvs_simulation_utils.hpp"

// ==============================================================================

using namespace gvs_demo;

// ==============================================================================
//  Part 1 — SNR sweep  (fixed rho_scale = 1.0)
// ==============================================================================

static void demo_hapgen_snr_sweep()
{
    constexpr int         n         = 200;
    constexpr double      tFDR      = 0.1;
    constexpr std::size_t K         = 20;
    constexpr unsigned    seed      = 2026;
    constexpr double      corr_max  = 0.98;
    constexpr double      rho_scale = 1.0;

    const std::vector<double> snr_grid = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0};

    // Precompute Cholesky factor once for the fixed rho_scale
    const Eigen::MatrixXd L = make_hapgen_cholesky(rho_scale);

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
        "Part 1: SNR Sweep  |  Hapgen-LD\n"
        "n=" + std::to_string(n) + "  p=500  s=130"
        "  rho_scale=1.0  tFDR=" + std::to_string(tFDR).substr(0, 3));

    for (double snr : snr_grid) {
        auto dat = make_hapgen_dgp(n, snr, L, seed);

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(),
                                           dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        const std::string lbl =
            "Hapgen-LD  snr=" + std::to_string(snr).substr(0, 3);

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
//  Part 2 — rho_scale sweep  (fixed SNR = 2.0)
// ==============================================================================

static void demo_hapgen_rho_scale_sweep()
{
    constexpr int         n         = 200;
    constexpr double      fixed_snr = 2.0;
    constexpr double      tFDR      = 0.1;
    constexpr std::size_t K         = 20;
    constexpr unsigned    seed      = 2026;
    constexpr double      corr_max  = 0.98;

    const std::vector<double> rs_grid = {0.15, 0.30, 0.50, 0.70, 0.85, 1.00};

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
        "Part 2: rho_scale Sweep  |  Hapgen-LD\n"
        "n=" + std::to_string(n) + "  p=500  s=130"
        "  SNR=" + std::to_string(fixed_snr).substr(0, 3) +
        "  tFDR=" + std::to_string(tFDR).substr(0, 3));

    for (double rs : rs_grid) {
        // Precompute Cholesky factor once per rho_scale value
        const Eigen::MatrixXd L = make_hapgen_cholesky(rs);

        auto dat = make_hapgen_dgp(n, fixed_snr, L, seed);

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(),
                                           dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        const std::string lbl =
            "Hapgen-LD  rho_scale=" + std::to_string(rs).substr(0, 4);

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

    demo_hapgen_snr_sweep();
    demo_hapgen_rho_scale_sweep();

    std::cout << "Hapgen-LD GVS demo complete.\n";
    return 0;
}
