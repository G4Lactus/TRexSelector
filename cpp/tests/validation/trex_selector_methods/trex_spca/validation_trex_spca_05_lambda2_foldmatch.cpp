// ==============================================================================
// validation_trex_spca_05_lambda2_foldmatch.cpp
// ==============================================================================
/**
 * @file validation_trex_spca_05_lambda2_foldmatch.cpp
 *
 * @brief Identical-fold, full-CV-curve head-to-head between the C++ coordinate-
 *        descent ridge CV (elastic_net_cv_gaussian, alpha=0) and R glmnet's
 *        cv.glmnet(alpha=0), to settle whether the lambda.1se "deviation" seen
 *        in demo_02 is a genuine bug or merely CV fold-randomisation noise.
 *
 * @details
 *  demo_02 compared the two implementations with DIFFERENT random folds, so
 *  lambda.1se could only agree up to fold noise (R's own lambda.1se has
 *  sd ~ 17 on this dataset). This demo removes that confound entirely:
 *
 *    1. R (lambda2_foldmatch.R) fixes a deterministic foldid, runs cv.glmnet
 *       on the dumped (X, y), and writes its lambda grid + cvm + cvsd + foldid.
 *    2. This demo reads those, then re-runs the EXACT library CV recipe
 *       (per-fold explicit-grid CD fit at R's grid, glmnet SEM aggregation)
 *       on the SAME (X, y), SAME grid and SAME folds.
 *    3. It diffs cvm and cvsd point-by-point and checks that lambda.min /
 *       lambda.1se land on the same grid index.
 *
 *  Interpretation:
 *    - max|Δcvm| ~ 1e-6 and identical lambda.min/1se indices  => CD == glmnet;
 *      the demo_02 lambda.1se gap was pure fold noise (NOT a bug).
 *    - a localised divergence  => the first lambda where cvm splits localises
 *      the discrepancy (per-fold standardisation, y-scaling, prediction, ...).
 *
 *  Prerequisite: run R/.../trex_spca/lambda2_foldmatch.R first (it consumes the
 *  lambda2_probe_X.csv / lambda2_probe_y.csv produced by
 *  validation_trex_spca_01_lambda2_probe and produces the fm_*.csv files this
 *  demo reads). Note: validation_trex_spca_01 is currently disabled in CMake
 *  (pending a rewrite to the new CV API), so this chain runs off the committed
 *  lambda2_probe_*.csv files rather than regenerating them.
 */
// ==============================================================================

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <ml_methods/model_selection/enet_cv_ccd.hpp>

#include "trex_spca_sim_utils.hpp"   // DEMO_OUTPUT_DIR

namespace ms = trex::ml_methods::model_selection;

// ------------------------------------------------------------------------------
// Minimal CSV readers
// ------------------------------------------------------------------------------

/** @brief Read a comma-separated dense matrix (no header). */
static Eigen::MatrixXd read_matrix_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
        rows.push_back(std::move(row));
    }
    if (rows.empty()) throw std::runtime_error("empty matrix " + path);
    Eigen::MatrixXd M(static_cast<Eigen::Index>(rows.size()),
                      static_cast<Eigen::Index>(rows[0].size()));
    for (Eigen::Index i = 0; i < M.rows(); ++i)
        for (Eigen::Index j = 0; j < M.cols(); ++j)
            M(i, j) = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    return M;
}

/** @brief Read a one-value-per-line numeric column (tolerates whitespace). */
static Eigen::VectorXd read_col(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<double> v;
    std::string line;
    while (std::getline(f, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        v.push_back(std::stod(line));
    }
    Eigen::VectorXd out(static_cast<Eigen::Index>(v.size()));
    for (Eigen::Index i = 0; i < out.size(); ++i) out(i) = v[static_cast<std::size_t>(i)];
    return out;
}

// ------------------------------------------------------------------------------
int main() {
    const std::string dir(DEMO_OUTPUT_DIR);

    // ---- load shared inputs + R reference curve ------------------------------
    const Eigen::MatrixXd X     = read_matrix_csv(dir + "lambda2_probe_X.csv");
    const Eigen::VectorXd y     = read_col(dir + "lambda2_probe_y.csv");
    const Eigen::VectorXd fid_d = read_col(dir + "fm_foldid.csv");      // 0-based
    const Eigen::VectorXd lam   = read_col(dir + "fm_r_lambda.csv");    // grid
    const Eigen::VectorXd r_cvm = read_col(dir + "fm_r_cvm.csv");
    const Eigen::VectorXd r_sd  = read_col(dir + "fm_r_cvsd.csv");

    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    const Eigen::Index K = lam.size();

    std::vector<int> foldid(static_cast<std::size_t>(n));
    int nfolds = 0;
    for (Eigen::Index i = 0; i < n; ++i) {
        foldid[static_cast<std::size_t>(i)] = static_cast<int>(std::lround(fid_d(i)));
        nfolds = std::max(nfolds, foldid[static_cast<std::size_t>(i)] + 1);
    }

    std::cout << "\n==============================================================\n"
              << "  lambda.1se fold-matched head-to-head (CD vs R cv.glmnet)\n"
              << "  X: " << n << " x " << p << ",  grid K=" << K
              << ",  folds=" << nfolds << "\n"
              << "==============================================================\n";

    // ---- replicate the library CV recipe at R's grid + folds -----------------
    // Same call the production path / elastic_net_cv_gaussian makes per fold:
    //   alpha=0, standardize=true, intercept=false, strong_rule off.
    // tol tightened so CD convergence slack does not pollute the comparison.
    constexpr double kTol     = 1e-12;
    constexpr int    kMaxIter = 1000000;

    Eigen::MatrixXd fold_mse(K, nfolds);
    fold_mse.setZero();

    for (int fcur = 0; fcur < nfolds; ++fcur) {
        std::vector<Eigen::Index> te, tr;
        for (Eigen::Index i = 0; i < n; ++i)
            (foldid[static_cast<std::size_t>(i)] == fcur ? te : tr).push_back(i);

        const Eigen::Index n_te = static_cast<Eigen::Index>(te.size());
        const Eigen::Index n_tr = static_cast<Eigen::Index>(tr.size());
        Eigen::MatrixXd X_tr(n_tr, p), X_te(n_te, p);
        Eigen::VectorXd y_tr(n_tr),    y_te(n_te);
        for (Eigen::Index k = 0; k < n_tr; ++k) { X_tr.row(k) = X.row(tr[static_cast<std::size_t>(k)]); y_tr(k) = y(tr[static_cast<std::size_t>(k)]); }
        for (Eigen::Index k = 0; k < n_te; ++k) { X_te.row(k) = X.row(te[static_cast<std::size_t>(k)]); y_te(k) = y(te[static_cast<std::size_t>(k)]); }

        ms::enet_gaussian en;
        en.fit(X_tr, y_tr, lam, /*alpha=*/0.0,
               /*standardize=*/true, /*intercept=*/false,
               /*use_strong_rule=*/false, kMaxIter, kTol);

        const Eigen::MatrixXd preds = en.predict(X_te);   // n_te x K
        fold_mse.col(fcur) =
            (preds.colwise() - y_te).colwise().squaredNorm().transpose()
            / static_cast<double>(n_te);
    }

    // glmnet SEM aggregation (grouped=TRUE, equal fold sizes):
    //   cvm  = mean over folds
    //   cvsd = sqrt( sum_f (mse_f - cvm)^2 / (F (F-1)) )
    Eigen::VectorXd cvm(K), sem(K);
    const double invFm1 = 1.0 / static_cast<double>(nfolds - 1);
    const double invF   = 1.0 / static_cast<double>(nfolds);
    for (Eigen::Index k = 0; k < K; ++k) {
        const double m = fold_mse.row(k).mean();
        cvm(k) = m;
        double var = 0.0;
        for (int f = 0; f < nfolds; ++f) { const double d = fold_mse(k, f) - m; var += d * d; }
        sem(k) = std::sqrt(var * invFm1 * invF);
    }

    Eigen::Index k_min = 0; double best = std::numeric_limits<double>::max();
    for (Eigen::Index k = 0; k < K; ++k) if (cvm(k) < best) { best = cvm(k); k_min = k; }
    const double thr = cvm(k_min) + sem(k_min);
    Eigen::Index k_1se = k_min;
    for (Eigen::Index k = 0; k < K; ++k) if (cvm(k) <= thr) { k_1se = k; break; }

    // R reference indices
    Eigen::Index r_kmin = 0; double rbest = std::numeric_limits<double>::max();
    for (Eigen::Index k = 0; k < K; ++k) if (r_cvm(k) < rbest) { rbest = r_cvm(k); r_kmin = k; }
    const double r_thr = r_cvm(r_kmin) + r_sd(r_kmin);
    Eigen::Index r_k1se = r_kmin;
    for (Eigen::Index k = 0; k < K; ++k) if (r_cvm(k) <= r_thr) { r_k1se = k; break; }

    // ---- curve diff ----------------------------------------------------------
    double max_dcvm = 0.0, max_dsd = 0.0, max_rel_cvm = 0.0;
    Eigen::Index arg_dcvm = 0;
    for (Eigen::Index k = 0; k < K; ++k) {
        const double dcvm = std::abs(cvm(k) - r_cvm(k));
        const double dsd  = std::abs(sem(k) - r_sd(k));
        if (dcvm > max_dcvm) { max_dcvm = dcvm; arg_dcvm = k; }
        max_dsd = std::max(max_dsd, dsd);
        if (r_cvm(k) > 0) max_rel_cvm = std::max(max_rel_cvm, dcvm / r_cvm(k));
    }

    std::cout << std::scientific << std::setprecision(4)
              << "\n  max|Δcvm|  = " << max_dcvm << "  (at k=" << arg_dcvm
              << ", lambda=" << lam(arg_dcvm) << ")\n"
              << "  max rel cvm = " << max_rel_cvm << "\n"
              << "  max|Δcvsd| = " << max_dsd << "\n";

    std::cout << std::fixed << std::setprecision(6)
              << "\n  lambda.min : CD idx=" << k_min  << " (" << lam(k_min)
              << ")   R idx=" << r_kmin << " (" << lam(r_kmin) << ")   "
              << (k_min == r_kmin ? "MATCH" : "DIFFER") << "\n"
              << "  lambda.1se : CD idx=" << k_1se  << " (" << lam(k_1se)
              << ")   R idx=" << r_k1se << " (" << lam(r_k1se) << ")   "
              << (k_1se == r_k1se ? "MATCH" : "DIFFER") << "\n";

    // ---- per-lambda table (subset around min/1se) ----------------------------
    std::cout << "\n  " << std::setw(4) << "idx" << std::setw(14) << "lambda"
              << std::setw(16) << "cvm_CD" << std::setw(16) << "cvm_R"
              << std::setw(14) << "Δcvm"
              << std::setw(14) << "sem_CD" << std::setw(14) << "sem_R" << "\n"
              << "  " << std::string(88, '-') << "\n";
    for (Eigen::Index k = 0; k < K; ++k) {
        const bool show = (k < 2) || (k >= K - 2)
                        || std::abs(k - k_min) <= 3 || std::abs(k - k_1se) <= 3;
        if (!show) continue;
        std::cout << "  " << std::setw(4) << k
                  << std::setw(14) << std::setprecision(4) << lam(k)
                  << std::setw(16) << std::setprecision(6) << cvm(k)
                  << std::setw(16) << r_cvm(k)
                  << std::setw(14) << std::scientific << std::setprecision(2)
                  << std::abs(cvm(k) - r_cvm(k)) << std::fixed
                  << std::setw(14) << std::setprecision(6) << sem(k)
                  << std::setw(14) << r_sd(k);
        if (k == k_min) std::cout << "  <- CD min";
        if (k == k_1se) std::cout << "  <- CD 1se";
        std::cout << "\n";
    }

    std::cout << "\n==============================================================\n"
              << "  VERDICT: ";
    if (k_min == r_kmin && k_1se == r_k1se) {
        std::cout << "CD == glmnet on identical folds -- lambda.min AND\n"
                     "           lambda.1se land on the SAME grid index. The\n"
                     "           demo_02 lambda.1se gap was pure CV fold noise.\n";
        if (max_rel_cvm < 1e-5)
            std::cout << "           cvm curves agree to " << std::scientific
                      << std::setprecision(1) << max_rel_cvm
                      << " rel (machine-precision algorithm match).\n";
        else
            std::cout << "           Residual cvm diff (" << std::scientific
                      << std::setprecision(1) << max_rel_cvm
                      << " rel) is glmnet solver-tolerance slack; it does NOT\n"
                         "           change the selected lambda indices.\n";
    } else {
        std::cout << "DIVERGENCE -- selected indices differ; inspect the first\n"
                     "           cvm split above to localise the discrepancy.\n";
    }
    std::cout << "==============================================================\n";
    return 0;
}
