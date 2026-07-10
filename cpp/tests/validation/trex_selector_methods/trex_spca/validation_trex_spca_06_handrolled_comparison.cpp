// ==============================================================================
// validation_trex_spca_06_handrolled_comparison.cpp
// ==============================================================================
/**
 * @file validation_trex_spca_06_handrolled_comparison.cpp
 *
 * @brief Hand-rolled per-PC T-Rex SPCA pipeline (legacy R recipe) vs. the
 *        TRexSPCA class, on identical centered data.
 *
 * @details
 *  Background: the legacy CRAN R reference
 *  (TRex Legacy CRAN Simulations/trex_spca/demo_trex_spca_01.R) achieves
 *  FDR ~ 0.10 / TPR ~ 1.0 at -10 dB on the sparse 3-factor model, evaluating
 *  on CENTER-ONLY X (covariance PCA). The demo pipeline later gained a shared
 *  z-score standardization step, which changes the problem itself (correlation
 *  PCA; the factor amplitude signal in the column scales is destroyed) and
 *  pushes the measured FDR to ~ 0.5 at the same nominal SNR.
 *
 *  This program removes the demo layer from the equation. Per trial, on the
 *  SAME center-only X:
 *
 *    Method A — hand-rolled pipeline, replicating the legacy R recipe
 *      step by step:
 *        1. thin SVD of centered X  ->  V_ord (p x M), Z_ord = X V_ord;
 *        2. per PC m: TRexGVSSelector(EN/TENET, corr_max 0.5, Single linkage,
 *           lambda_2 = -1 -> internal 10-fold CV, CV_1SE_CCD) on (X, Z_ord[,m]);
 *        3. ActiveSet assembly: ridge (lambda 1e-6) restricted to the active
 *           columns, normalized (legacy: solve(X_A'X_A + 1e-6 I) X_A'y);
 *           Thresholded assembly: top-|A| entries of V_ord[,m], normalized;
 *        4. scores Z_est = X V_est; PC1 FDR/TPR + QR-adjusted cumulative PEV.
 *
 *    Method B — the TRexSPCA class with identical control parameters
 *      (EN/TENET, lambda_2 = -1, CV_1SE_CCD, default L2 selector scaling),
 *      both SPCAMode::ActiveSet and SPCAMode::Thresholded.
 *
 *  If class and hand-rolled agree, the TRexSPCA class is a faithful
 *  implementation of the legacy recipe and any FDR discrepancy seen in the
 *  demos comes from the demo preprocessing, not from the class.
 *
 *  Reference values (legacy CRAN R, 200 MC, -10 dB, center-only X):
 *    T-Rex EN Act: FDR 0.100  TPR 0.999  PEV 0.117
 *    T-Rex EN Thr: FDR 0.100  TPR 1.000  PEV 0.128
 *
 *  Flags: --n <num_MC> (default 200), --snr <dB> (default -10),
 *         --seed <base> (default 42), --threads <k> (default 6).
 */
// ==============================================================================

// std includes
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

// Demo utilities (DGP + evaluate_spca; transitively includes trex_spca.hpp,
// trex_gvs.hpp)
#include "trex_spca_sim_utils.hpp"

// ==============================================================================

using namespace spca_sim;
namespace tg = trex::trex_selector_methods::trex_gvs;
namespace sd = trex::trex_selector_methods::utils::solver_dispatch;


// ==============================================================================
// Hand-rolled pipeline (legacy R recipe)
// ==============================================================================

namespace {

// Column centering comes from spca_sim::center_columns (in place).

/** @brief GVS control mirroring the legacy trex(..., method="trex+GVS",
 *         GVS_type="EN") call: EN/TENET, corr_max 0.5, Single linkage,
 *         lambda_2 auto via internal 10-fold CV (CV_1SE_CCD). */
tg::TRexGVSControlParameter make_legacy_gvs_ctrl()
{
    tg::TRexGVSControlParameter ctrl;
    ctrl.gvs_type              = tg::GVSType::EN;
    ctrl.en_solver             = tg::ENSolverType::TENET;
    ctrl.lambda_2              = -1.0;
    ctrl.lambda2_method        = tg::LambdaSelectionMethod::CV_1SE_CCD;
    ctrl.trex_ctrl.solver_type = sd::SolverTypeForTRex::TENET;
    return ctrl;
}

struct HandRolledResult {
    SPCASingleResult act;
    SPCASingleResult thr;
};

/**
 * @brief Run the hand-rolled legacy pipeline on centered X.
 *
 * @param Xc      Centered observation matrix (n x p); restored by each GVS
 *                selector's destructor, so it is unchanged on return.
 * @param V_true  True loading matrix (p x M).
 * @param M       Number of components.
 * @param tFDR    Target FDR for each per-PC selection.
 */
HandRolledResult run_handrolled(Eigen::MatrixXd& Xc,
                                const Eigen::MatrixXd& V_true,
                                Eigen::Index M,
                                double tFDR)
{
    const Eigen::Index n = Xc.rows();
    const Eigen::Index p = Xc.cols();

    // 1. Ordinary (covariance) PCA via thin SVD.
    Eigen::BDCSVD<Eigen::MatrixXd> svd(Xc, Eigen::ComputeThinV);
    const Eigen::MatrixXd V_ord = svd.matrixV().leftCols(M);
    const Eigen::MatrixXd Z_ord = Xc * V_ord;

    Eigen::MatrixXd V_act = Eigen::MatrixXd::Zero(p, M);
    Eigen::MatrixXd V_thr = Eigen::MatrixXd::Zero(p, M);

    for (Eigen::Index m = 0; m < M; ++m) {
        Eigen::VectorXd y_m = Z_ord.col(m);

        // 2. Per-PC T-Rex+GVS(EN) selection (selector seed -1: HW entropy).
        std::vector<Eigen::Index> active_set;
        {
            Eigen::Map<Eigen::MatrixXd> X_map(Xc.data(), n, p);
            Eigen::Map<Eigen::VectorXd> y_map(y_m.data(), n);
            tg::TRexGVSSelector gvs(X_map, y_map, tFDR,
                                    make_legacy_gvs_ctrl(), -1, false);
            gvs.select();
            for (std::size_t idx : gvs.getSelectedIndices())
                active_set.push_back(static_cast<Eigen::Index>(idx));
        } // destructor restores Xc to its centered state

        const auto k = static_cast<Eigen::Index>(active_set.size());
        if (k == 0) continue;

        // 3a. ActiveSet assembly: ridge (lambda 1e-6) on the active columns.
        Eigen::MatrixXd X_A(n, k);
        for (Eigen::Index i = 0; i < k; ++i)
            X_A.col(i) = Xc.col(active_set[static_cast<std::size_t>(i)]);

        Eigen::MatrixXd G = X_A.transpose() * X_A;
        G.diagonal().array() += 1e-6;
        Eigen::VectorXd beta = G.ldlt().solve(X_A.transpose() * y_m);
        beta.normalize();
        for (Eigen::Index i = 0; i < k; ++i)
            V_act(active_set[static_cast<std::size_t>(i)], m) = beta(i);

        // 3b. Thresholded assembly: top-k entries of the ordinary loading.
        Eigen::VectorXd v_m = V_ord.col(m);
        std::vector<double> abs_vals(static_cast<std::size_t>(p));
        for (Eigen::Index i = 0; i < p; ++i)
            abs_vals[static_cast<std::size_t>(i)] = std::abs(v_m(i));
        std::nth_element(abs_vals.begin(), abs_vals.begin() + (p - k),
                         abs_vals.end());
        const double thresh = abs_vals[static_cast<std::size_t>(p - k)];
        for (Eigen::Index i = 0; i < p; ++i)
            if (std::abs(v_m(i)) >= thresh) V_thr(i, m) = v_m(i);
        V_thr.col(m).normalize();
    }

    // 4. Scores + metrics on the same centered X.
    const Eigen::MatrixXd Z_act = Xc * V_act;
    const Eigen::MatrixXd Z_thr = Xc * V_thr;

    return {evaluate_spca(Xc, V_act, Z_act, V_true),
            evaluate_spca(Xc, V_thr, Z_thr, V_true)};
}

/** @brief Run the TRexSPCA class on (a copy of) centered X. */
SPCASingleResult run_class(const Eigen::MatrixXd& Xc,
                           const Eigen::MatrixXd& V_true,
                           Eigen::Index M,
                           double tFDR,
                           SPCAMode mode)
{
    Eigen::MatrixXd X_work = Xc;  // class centers (no-op) and restores

    TRexSPCAControlParameter ctrl;
    ctrl.mode                    = mode;
    ctrl.en_solver               = ENSolverType::TENET;
    ctrl.gvs_ctrl.lambda_2       = -1.0;
    ctrl.gvs_ctrl.lambda2_method = LambdaSelectionMethod::CV_1SE_CCD;
    // Selector scaling: library default (L2) — matches the legacy lm_dummy.R
    // column centering + L2 normalization. NO demo-level z-scoring anywhere.

    Eigen::Map<Eigen::MatrixXd> X_map(X_work.data(), X_work.rows(), X_work.cols());
    TRexSPCA       spca(X_map, M, tFDR, ctrl, -1);
    TRexSPCAResult res = spca.select();

    return evaluate_spca(Xc, res.V, res.Z, V_true);
}

/** @brief Accumulate mean FDR/TPR/PEV over trials. */
struct Acc {
    double fdr = 0.0, tpr = 0.0, pev = 0.0;
    void add(const SPCASingleResult& r) { fdr += r.fdr; tpr += r.tpr; pev += r.pev; }
    void avg(double dMC) { fdr /= dMC; tpr /= dMC; pev /= dMC; }
};

} // namespace


// ==============================================================================
// Main
// ==============================================================================

int main(int argc, char** argv)
{
    std::cout.setf(std::ios::unitbuf);

    // ----- configuration -----
    std::size_t num_MC    = 200;
    double      snr_db    = -10.0;
    int         base_seed = 42;
    int         threads   = 6;

    for (int i = 1; i < argc - 1; ++i) {
        if (!std::strcmp(argv[i], "--n"))       num_MC    = static_cast<std::size_t>(std::stoul(argv[i + 1]));
        if (!std::strcmp(argv[i], "--snr"))     snr_db    = std::stod(argv[i + 1]);
        if (!std::strcmp(argv[i], "--seed"))    base_seed = std::stoi(argv[i + 1]);
        if (!std::strcmp(argv[i], "--threads")) threads   = std::stoi(argv[i + 1]);
    }
    omp_set_num_threads(threads);

    const Eigen::Index n = 50, p = 100, p1 = 5, M = 3, pool = 30;
    const double tFDR = 0.10;

    std::cout << "================================================================\n"
              << "  T-Rex SPCA: hand-rolled legacy pipeline vs. TRexSPCA class\n"
              << "================================================================\n"
              << "  n=" << n << "  p=" << p << "  p1=" << p1 << "  M=" << M
              << "  tFDR=" << tFDR << "  SNR=" << snr_db << " dB"
              << "  num_MC=" << num_MC << "  threads=" << threads << "\n"
              << "  X preprocessing: CENTER ONLY (covariance PCA, legacy footing)\n\n";

    const int iMC = static_cast<int>(num_MC);
    std::vector<SPCASingleResult> hr_act(num_MC), hr_thr(num_MC),
                                  cl_act(num_MC), cl_thr(num_MC);

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        const unsigned seed = static_cast<unsigned>(base_seed)
                            + static_cast<unsigned>(mc) * 1000u;
        auto dat = DataGenerator::generate_sparse_factor_model(
            n, p, p1, M, snr_db, pool, seed);

        Eigen::MatrixXd Xc = dat.X;
        center_columns(Xc);  // spca_sim: mean subtraction in place

        const auto hr = run_handrolled(Xc, dat.V, M, tFDR);
        hr_act[static_cast<std::size_t>(mc)] = hr.act;
        hr_thr[static_cast<std::size_t>(mc)] = hr.thr;

        cl_act[static_cast<std::size_t>(mc)] =
            run_class(Xc, dat.V, M, tFDR, SPCAMode::ActiveSet);
        cl_thr[static_cast<std::size_t>(mc)] =
            run_class(Xc, dat.V, M, tFDR, SPCAMode::Thresholded);

        if ((mc + 1) % 25 == 0) {
            #pragma omp critical
            std::cout << "  ... " << (mc + 1) << " / " << num_MC << " trials\n";
        }
    }

    Acc a_hr_act, a_hr_thr, a_cl_act, a_cl_thr;
    for (std::size_t i = 0; i < num_MC; ++i) {
        a_hr_act.add(hr_act[i]);  a_hr_thr.add(hr_thr[i]);
        a_cl_act.add(cl_act[i]);  a_cl_thr.add(cl_thr[i]);
    }
    const auto dMC = static_cast<double>(num_MC);
    a_hr_act.avg(dMC);  a_hr_thr.avg(dMC);  a_cl_act.avg(dMC);  a_cl_thr.avg(dMC);

    auto row = [](const std::string& name, const Acc& a) {
        std::cout << "  " << std::left << std::setw(22) << name
                  << std::right << std::fixed << std::setprecision(4)
                  << std::setw(10) << a.fdr << std::setw(10) << a.tpr
                  << std::setw(10) << a.pev << "\n";
    };

    std::cout << "\n----------------------------------------------------------------\n"
              << "  " << std::left << std::setw(22) << "Method"
              << std::right << std::setw(10) << "FDR"
              << std::setw(10) << "TPR" << std::setw(10) << "PEV" << "\n"
              << "----------------------------------------------------------------\n";
    row("HandRolled-EN-Act", a_hr_act);
    row("Class-EN-Act",      a_cl_act);
    std::cout << "\n";
    row("HandRolled-EN-Thr", a_hr_thr);
    row("Class-EN-Thr",      a_cl_thr);
    std::cout << "----------------------------------------------------------------\n"
              << "  Legacy CRAN R reference (200 MC, -10 dB, center-only):\n"
              << "    T-Rex EN Act: FDR 0.100  TPR 0.999  PEV 0.117\n"
              << "    T-Rex EN Thr: FDR 0.100  TPR 1.000  PEV 0.128\n"
              << "  (Dummies use hardware entropy: agreement is within MC noise,\n"
              << "   not trial-for-trial.)\n";

    return 0;
}
