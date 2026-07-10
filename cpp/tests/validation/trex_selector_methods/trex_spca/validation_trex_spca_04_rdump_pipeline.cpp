// ==============================================================================
// validation_trex_spca_04_rdump_pipeline.cpp
// ==============================================================================
/**
 * @file validation_trex_spca_04_rdump_pipeline.cpp
 *
 * @brief Step-by-step C++/R controlled comparison on identical R-generated data.
 *
 * @details
 *  This demo deliberately bypasses the TRexSPCA class and rebuilds the per-PC1
 *  pipeline of the R reference (demo_trex_spca_01.R, "T-Rex EN" method) as an
 *  explicit sequence of steps, so each intermediate quantity can be diffed
 *  against R on the SAME input matrices:
 *
 *      step 1  load centered X_i.csv  (n x p, R-generated, already centered)
 *      step 2  ordinary PCA on X      ->  PC1 scores  y = Z[:,0]   (== R: X %*% svd(X)$v[,1])
 *      step 3  lambda_2  (LARS units): either auto CV (glmnet-style, == R cv.glmnet*p/2)
 *                                      or a fixed value supplied per-trial from R
 *      step 4  TRexGVSSelector (GVSType::EN, TENET_AUG) on (X, y)  ->  selected indices
 *      step 5  FDR / TPR of the PC1 support vs the R-dumped truth (optional)
 *
 *  Inputs (all under <rdump_dir>; <rdump_dir> defaults to a machine-specific
 *  absolute path ending in R/trex_selector_methods/validation/trex_spca/rdump, and is
 *  overridable at runtime with `--dir <path>`):
 *    - X_<mc>.csv         : REQUIRED. n x p centered design, comma-separated, no header.
 *    - truth.csv          : OPTIONAL. Header "mc,indices"; indices = dash-joined
 *                           0-based PC1 true support (e.g. "0,3-8-12-19-27").
 *    - r_lambda2.csv      : OPTIONAL. Header "mc,lambda2"; R's lambda_2_lars per trial.
 *                           When present (and --use-r-lambda2), this exact value is
 *                           passed to the selector, removing CV-fold RNG from the
 *                           comparison so any selection difference is purely the
 *                           T-Rex selector.
 *
 *  Outputs (written into <rdump_dir>):
 *    - cpp_pipeline.csv   : mc,lambda2,k,T_stop,num_dummies,fdr,tpr,indices  (0-based indices)
 *    - cpp_pc1.csv        : mc, then n PC1 score values  (compare |corr| with R's PC1)
 *
 *  Conventions: ALL variable indices are 0-based (matches getSelectedIndices()).
 *  R side must dump truth/selected indices as `which(...) - 1L`.
 */
// ==============================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

// T-Rex GVS selector + PCA baseline
#include <trex_selector_methods/trex_gvs/trex_gvs.hpp>
#include <ml_methods/pca/pca.hpp>

// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace tg     = trex::trex_selector_methods::trex_gvs;
namespace tc     = trex::trex_selector_methods::trex_core;
namespace pca_ns = trex::ml_methods::pca;
namespace dn     = trex::trex_selector_methods::utils::data_normalizer;

// ==============================================================================
// CSV helpers
// ==============================================================================

/** @brief Read a headerless numeric CSV into a dense matrix (row-major file order). */
static Eigen::MatrixXd read_csv_matrix(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open matrix CSV: " + path.string());
    }

    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        rows.push_back(std::move(row));
    }

    if (rows.empty()) {
        throw std::runtime_error("Empty matrix CSV: " + path.string());
    }

    const Eigen::Index n = static_cast<Eigen::Index>(rows.size());
    const Eigen::Index p = static_cast<Eigen::Index>(rows.front().size());
    Eigen::MatrixXd M(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        if (static_cast<Eigen::Index>(rows[static_cast<std::size_t>(i)].size()) != p) {
            throw std::runtime_error("Ragged matrix CSV: " + path.string());
        }
        for (Eigen::Index j = 0; j < p; ++j) {
            M(i, j) = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        }
    }
    return M;
}

/** @brief Parse a dash-joined index list "3-8-12" into a sorted 0-based vector. */
static std::vector<Eigen::Index> parse_dash_indices(const std::string& field)
{
    std::vector<Eigen::Index> out;
    if (field.empty() || field == "NA") {
        return out;
    }
    std::stringstream ss(field);
    std::string tok;
    while (std::getline(ss, tok, '-')) {
        if (!tok.empty()) {
            out.push_back(static_cast<Eigen::Index>(std::stoll(tok)));
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

/** @brief Read "mc,indices" truth file -> map mc -> 0-based true support. */
static std::unordered_map<int, std::vector<Eigen::Index>>
read_truth(const std::filesystem::path& path)
{
    std::unordered_map<int, std::vector<Eigen::Index>> out;
    std::ifstream in(path);
    if (!in) {
        return out;  // optional
    }
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (first) { first = false; continue; }  // skip header
        std::stringstream ss(line);
        std::string mc_str, idx_str;
        std::getline(ss, mc_str, ',');
        std::getline(ss, idx_str, ',');
        out[std::stoi(mc_str)] = parse_dash_indices(idx_str);
    }
    return out;
}

/** @brief Read "mc,lambda2" file -> map mc -> R lambda_2_lars. */
static std::unordered_map<int, double>
read_lambda2(const std::filesystem::path& path)
{
    std::unordered_map<int, double> out;
    std::ifstream in(path);
    if (!in) {
        return out;  // optional
    }
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (first) { first = false; continue; }  // skip header
        std::stringstream ss(line);
        std::string mc_str, lam_str;
        std::getline(ss, mc_str, ',');
        std::getline(ss, lam_str, ',');
        out[std::stoi(mc_str)] = std::stod(lam_str);
    }
    return out;
}

/** @brief Join a 0-based index vector as "a-b-c". */
static std::string join_dash(const std::vector<Eigen::Index>& v)
{
    std::ostringstream os;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) os << '-';
        os << v[i];
    }
    return os.str();
}

// ==============================================================================
// Per-trial pipeline
// ==============================================================================

struct TrialOut {
    int                       mc           = 0;
    double                    lambda2      = 0.0;
    std::size_t               k            = 0;
    std::size_t               T_stop       = 0;
    std::size_t               num_dummies  = 0;
    double                    fdr          = -1.0;  // -1 => NA (no truth)
    double                    tpr          = -1.0;
    std::vector<Eigen::Index> indices;             // 0-based, sorted
    Eigen::VectorXd           pc1;                 // n PC1 scores
};

int main(int argc, char** argv)
{
    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(1);  // deterministic, single-threaded for step-by-step diffing

    // ----- configuration -----
    // External R dump (not committed in the library; ~12 MB). Regenerate it with
    // the R generator in the examples package and pass it via --dir <path>.
    std::filesystem::path rdump_dir =
        "R/trex_selector_methods/validation/trex_spca/rdump";
    int    num_trials   = 100;
    double tFDR         = 0.10;
    Eigen::Index M      = 3;        // ordinary PCA rank (PC1 used for the metric)
    double corr_max     = 0.5;      // R trex+GVS default
    int    seed_base    = 1000;     // deterministic dummy seed base (reproducible C++ runs)
    bool   use_r_lambda2 = false;   // when true + r_lambda2.csv present, feed R's lambda2
    bool   zscore        = false;   // column scaling inside the selector
    bool   no_stagnation = false;   // disable T-loop stagnation early-stop (match R)
    bool   tenet_aug_lars = false;  // TENET_AUG inner solver = pure LARS (match R type="lar")

    for (int a = 1; a < argc; ++a) {
        std::string arg = argv[a];
        if (arg == "--dir" && a + 1 < argc)            rdump_dir   = argv[++a];
        else if (arg == "--n" && a + 1 < argc)         num_trials  = std::stoi(argv[++a]);
        else if (arg == "--seed" && a + 1 < argc)      seed_base   = std::stoi(argv[++a]);
        else if (arg == "--use-r-lambda2")             use_r_lambda2 = true;
        else if (arg == "--zscore")                    zscore        = true;
        else if (arg == "--no-stagnation")             no_stagnation = true;
        else if (arg == "--tenet-aug-lars")            tenet_aug_lars = true;
    }

    const std::filesystem::path truth_path   = rdump_dir / "truth.csv";
    const std::filesystem::path rlam_path    = rdump_dir / "r_lambda2.csv";
    const auto truth_map   = read_truth(truth_path);
    const auto rlam_map    = read_lambda2(rlam_path);

    std::cout << "================================================================\n";
    std::cout << "  T-Rex SPCA -- step-by-step pipeline on R-generated X\n";
    std::cout << "================================================================\n";
    std::cout << "  rdump_dir     = " << rdump_dir << "\n";
    std::cout << "  num_trials    = " << num_trials << "\n";
    std::cout << "  tFDR          = " << tFDR << "   M=" << M
              << "   corr_max=" << corr_max << "\n";
    std::cout << "  scaling       = " << (zscore ? "ZSCORE" : "L2") << "\n";
    std::cout << "  tenet_aug     = " << (tenet_aug_lars ? "TLARS (pure LARS)" : "TLASSO (default)") << "\n";
    std::cout << "  truth.csv     = " << (truth_map.empty() ? "absent" : "present")
              << "   r_lambda2.csv = " << (rlam_map.empty() ? "absent" : "present")
              << (use_r_lambda2 ? " (USED)" : "") << "\n\n";

    std::vector<TrialOut> results;
    results.reserve(static_cast<std::size_t>(num_trials));

    double sum_fdr = 0.0, sum_tpr = 0.0;
    int    cnt_eval = 0;
    double sum_k = 0.0;

    for (int mc = 0; mc < num_trials; ++mc) {
        const std::filesystem::path xpath =
            rdump_dir / ("X_" + std::to_string(mc) + ".csv");
        if (!std::filesystem::exists(xpath)) {
            std::cout << "  [skip] missing " << xpath.filename() << "\n";
            continue;
        }

        // step 1: load centered X (n x p)
        Eigen::MatrixXd X = read_csv_matrix(xpath);
        const Eigen::Index n = X.rows();

        // step 2: ordinary PCA on the already-centered X -> PC1 = Z.col(0)
        //         (center=false, normalize=false : covariance PCA, == R svd path)
        pca_ns::PCAResult ord = pca_ns::PCA(X, M, /*center=*/false, /*normalize=*/false).fit();
        Eigen::VectorXd y = ord.Z.col(0);

        // step 3: lambda_2 (LARS units)
        tg::TRexGVSControlParameter gvs_ctrl;
        gvs_ctrl.gvs_type   = tg::GVSType::EN;
        gvs_ctrl.en_solver  = tg::ENSolverType::TENET_AUG;
        gvs_ctrl.corr_max   = corr_max;
        gvs_ctrl.tenet_aug_use_lars = tenet_aug_lars;
        if (use_r_lambda2) {
            auto it = rlam_map.find(mc);
            if (it == rlam_map.end()) {
                throw std::runtime_error(
                    "--use-r-lambda2 set but r_lambda2.csv has no row for mc=" +
                    std::to_string(mc));
            }
            gvs_ctrl.lambda_2 = it->second;            // fixed: removes CV-fold RNG
        } else {
            gvs_ctrl.lambda_2       = -1.0;            // <0 sentinel: auto-compute via CV
            gvs_ctrl.lambda2_method = tg::LambdaSelectionMethod::CV_1SE_CCD;
            gvs_ctrl.cv_n_folds     = 10;
            gvs_ctrl.cv_seed        = seed_base + mc;  // deterministic folds
        }

        tc::TRexControlParameter trex_ctrl;
        trex_ctrl.solver_type   = trex::trex_selector_methods::utils::solver_dispatch::
                                      SolverTypeForTRex::TENET_AUG;
        trex_ctrl.scaling_mode  = zscore ? dn::ScalingMode::ZSCORE : dn::ScalingMode::L2;
        if (no_stagnation) {
            trex_ctrl.tloop_stagnation_stop = false;  // run full T-loop like R
        }
        gvs_ctrl.trex_ctrl = trex_ctrl;

        // step 4: run the GVS EN selector on (X, y)
        Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(y.data(), n);
        tg::TRexGVSSelector gvs(X_map, y_map, tFDR, gvs_ctrl,
                                /*seed=*/seed_base + mc, /*verbose=*/false);
        gvs.select();

        TrialOut to;
        to.mc          = mc;
        to.lambda2     = gvs.getGVSResult().lambda2_used;
        to.T_stop      = gvs.getTStop();
        to.num_dummies = gvs.getGVSResult().num_dummies;
        to.pc1         = y;

        const auto& raw = gvs.getSelectedIndices();  // 0-based
        to.indices.reserve(raw.size());
        for (std::size_t idx : raw) {
            to.indices.push_back(static_cast<Eigen::Index>(idx));
        }
        std::sort(to.indices.begin(), to.indices.end());
        to.k = to.indices.size();

        // step 5: FDR / TPR vs truth (PC1 support), if truth available
        auto tit = truth_map.find(mc);
        if (tit != truth_map.end()) {
            const auto& truth = tit->second;
            std::size_t tp = 0;
            for (Eigen::Index s : to.indices) {
                if (std::binary_search(truth.begin(), truth.end(), s)) {
                    ++tp;
                }
            }
            const std::size_t sel = to.indices.size();
            to.fdr = (sel == 0) ? 0.0
                   : static_cast<double>(sel - tp) / static_cast<double>(sel);
            to.tpr = truth.empty() ? 1.0
                   : static_cast<double>(tp) / static_cast<double>(truth.size());
            sum_fdr += to.fdr; sum_tpr += to.tpr; ++cnt_eval;
        }
        sum_k += static_cast<double>(to.k);

        results.push_back(std::move(to));

        if ((mc + 1) % 10 == 0) {
            std::cout << "  ... processed " << (mc + 1) << " trials\n";
        }
    }

    // ----- write cpp_pipeline.csv -----
    {
        std::ofstream out(rdump_dir / "cpp_pipeline.csv");
        out << "mc,lambda2,k,T_stop,num_dummies,fdr,tpr,indices\n";
        out << std::setprecision(10);
        for (const auto& r : results) {
            out << r.mc << ',' << r.lambda2 << ',' << r.k << ','
                << r.T_stop << ',' << r.num_dummies << ',';
            if (r.fdr < 0.0) out << "NA,"; else out << r.fdr << ',';
            if (r.tpr < 0.0) out << "NA,"; else out << r.tpr << ',';
            out << join_dash(r.indices) << '\n';
        }
    }

    // ----- write cpp_pc1.csv -----
    {
        std::ofstream out(rdump_dir / "cpp_pc1.csv");
        out << std::setprecision(12);
        for (const auto& r : results) {
            out << r.mc;
            for (Eigen::Index i = 0; i < r.pc1.size(); ++i) {
                out << ',' << r.pc1(i);
            }
            out << '\n';
        }
    }

    // ----- summary -----
    std::cout << "\n----------------------------------------------------------------\n";
    std::cout << "  trials evaluated : " << results.size() << "\n";
    std::cout << "  mean k (selected): "
              << std::fixed << std::setprecision(3)
              << (results.empty() ? 0.0 : sum_k / static_cast<double>(results.size()))
              << "\n";
    if (cnt_eval > 0) {
        std::cout << "  mean FDR (PC1)   : " << (sum_fdr / cnt_eval) << "\n";
        std::cout << "  mean TPR (PC1)   : " << (sum_tpr / cnt_eval) << "\n";
    } else {
        std::cout << "  (no truth.csv -> FDR/TPR not computed)\n";
    }
    std::cout << "\n  wrote: " << (rdump_dir / "cpp_pipeline.csv") << "\n";
    std::cout << "  wrote: " << (rdump_dir / "cpp_pc1.csv") << "\n";

    return 0;
}
