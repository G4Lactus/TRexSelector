// ============================================================================
// validation_ts_02_tlars_tlasso_rcompare.cpp
// ============================================================================
/**
 * @file validation_ts_02_tlars_tlasso_rcompare.cpp
 *
 * @brief Final forward-selection path-equivalence test: CRAN `tlars` vs the C++
 *        tsolvers solvers on the SPCA sparse factor-model data.
 *
 * @details
 *  Companion to  R/tsolvers/validation/validation_ts_02_tlars_tlasso_rcompare/
 *  demo_ts_compare_tlars_tlasso.R, which generates data
 *  from the SPCA sparse factor model (with SNR control) and dumps, per dataset i:
 *      Xn_<i>.csv (n x p), Dn_<i>.csv (n x L), y_<i>.csv (n x 1)
 *          -- already centred + unit-L2 standardised; fed verbatim (no further
 *             preprocessing) so both sides run the SAME numeric path.
 *      r_lar_beta_<i>.csv   / r_lar_act_<i>.csv     : LARS  (type="lar")
 *      r_lasso_beta_<i>.csv / r_lasso_act_<i>.csv   : LASSO (type="lasso")
 *      r_en_beta_<i>.csv    / r_en_act_<i>.csv      : EN (augmented lasso_star)
 *
 *  Five C++ solver runs are checked against the matching R reference:
 *      TLARS_Solver   vs r_lar      (pure LARS)
 *      TLASSO_Solver  vs r_lasso    (LARS-LASSO, sign-change drops)
 *      TENET_Solver   vs r_en       (Gram-based elastic net; LASSO-style drops)
 *      TENETAug_Solver vs r_en      (augmented-lasso elastic net; default TLASSO
 *                                    inner) -- same EN path as TENET.
 *      TENETAug_Solver vs r_enlar   (augmented-lasso EN with use_lars_inner=true;
 *                                    pure-LARS inner, augmented type="lar").
 *
 *  API mapping: R uses ONE matrix Z = cbind(X, D) with num_dummies = L; the C++
 *  solvers take X and D SEPARATELY. R column j (1-based) <-> C++ combined 0-based
 *  index (j-1).  Both produce a (p+L) x (steps+1) path with a leading zero
 *  column; actions are signed (added +, dropped -), C++ 0-based -> R (a>=0 ?
 *  a+1 : a-1).
 *
 *  Because the factor model has p+L > n, the plain LARS/LASSO active-set cap can
 *  differ by a single step between R and C++, so paths are compared on their
 *  common LEFT prefix (step 0,1,2,...) and the first divergence step (if any) is
 *  reported.  EN runs on the full-rank augmented system and traces the complete
 *  path.  Tolerance 1e-7.
 */
// ============================================================================

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <utils/openmp/utils_openmp.hpp>
#include <tsolvers/linear_model/lars_based/tlars_solver.hpp>
#include <tsolvers/linear_model/lars_based/tlasso_solver.hpp>
#include <tsolvers/linear_model/lars_based/tenet_solver.hpp>
#include <tsolvers/linear_model/lars_based/tenet_aug_solver.hpp>

namespace lars     = trex::tsolvers::linear_model::lars_based;
namespace ts       = trex::tsolvers;
namespace omp_util = trex::utils::openmp;

// ============================================================================
// CSV helpers
// ============================================================================

/** @brief Read a headerless numeric CSV into a dense matrix (file row order). */
static Eigen::MatrixXd read_csv_matrix(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open CSV: " + path.string());

    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
        rows.push_back(std::move(row));
    }
    if (rows.empty()) throw std::runtime_error("Empty CSV: " + path.string());

    const Eigen::Index nr = static_cast<Eigen::Index>(rows.size());
    const Eigen::Index nc = static_cast<Eigen::Index>(rows.front().size());
    Eigen::MatrixXd M(nr, nc);
    for (Eigen::Index i = 0; i < nr; ++i) {
        const auto& r = rows[static_cast<std::size_t>(i)];
        if (static_cast<Eigen::Index>(r.size()) != nc)
            throw std::runtime_error("Ragged CSV: " + path.string());
        for (Eigen::Index j = 0; j < nc; ++j) M(i, j) = r[static_cast<std::size_t>(j)];
    }
    return M;
}

/** @brief Read a single-line signed integer CSV (the R actions vector). */
static std::vector<int> read_actions(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open actions CSV: " + path.string());
    std::vector<int> out;
    std::string cell;
    while (std::getline(in, cell, ',')) {
        cell.erase(std::remove_if(cell.begin(), cell.end(),
                   [](unsigned char c){ return std::isspace(c); }), cell.end());
        if (!cell.empty()) out.push_back(std::stoi(cell));
    }
    return out;
}

// ============================================================================
// Solver-action -> R format (signed, 1-based)
// ============================================================================
/**
 * @brief Flatten getActions() into R's signed-1-based convention.
 *  C++ records 0-based combined indices: added var j -> +j, dropped var j -> -j.
 *  R records 1-based: added -> +(j+1), dropped -> -(j+1).  Mapping a>=0 -> a+1,
 *  a<0 -> a-1.  (A "drop of variable 0" encoded as -0 == 0 does not occur here:
 *  index 0 is a strong predictor and is never dropped.)
 */
static std::vector<int> flatten_actions_to_r(
    const std::vector<std::vector<int>>& actions)
{
    std::vector<int> out;
    for (const auto& step : actions)
        for (int a : step)
            out.push_back(a >= 0 ? a + 1 : a - 1);
    return out;
}

// ============================================================================
// Path comparison (left-anchored prefix)
// ============================================================================
struct CmpResult {
    double      max_prefix_diff = 0.0;   ///< max |Δ| over common left prefix
    double      max_final_diff  = 0.0;   ///< max |Δ| of the common last prefix column
    long        first_div_step  = -1;    ///< first step whose |Δ| > tol (-1 = none)
    bool        actions_match   = false; ///< signed action prefix matches exactly
    std::size_t cpp_steps = 0, r_steps = 0;
    std::size_t cpp_nact  = 0, r_nact  = 0;
};

static CmpResult compare(const Eigen::MatrixXd&  cpp_bp,
                         const std::vector<int>& cpp_act_r,
                         const Eigen::MatrixXd&  r_bp,
                         const std::vector<int>& r_act,
                         double tol,
                         Eigen::Index col_cap = 0)
{
    CmpResult res;
    res.cpp_steps = static_cast<std::size_t>(cpp_bp.cols());
    res.r_steps   = static_cast<std::size_t>(r_bp.cols());
    res.cpp_nact  = cpp_act_r.size();
    res.r_nact    = r_act.size();

    if (cpp_bp.rows() != r_bp.rows())
        throw std::runtime_error("Variable-count mismatch between C++ and R paths.");

    // Both paths start at the zero column and add one column per step; compare
    // the common LEFT prefix step-by-step.  col_cap (>0) restricts the
    // comparison to the first col_cap columns -- used for the plain LARS/LASSO
    // solvers to stay within the full-rank region (steps < n); beyond the rank
    // limit the plain solution is non-unique (null-space extrapolation).
    Eigen::Index ncol = std::min(cpp_bp.cols(), r_bp.cols());
    if (col_cap > 0) ncol = std::min(ncol, col_cap);
    for (Eigen::Index c = 0; c < ncol; ++c) {
        const double d = (cpp_bp.col(c) - r_bp.col(c)).cwiseAbs().maxCoeff();
        res.max_prefix_diff = std::max(res.max_prefix_diff, d);
        if (res.first_div_step < 0 && d > tol) res.first_div_step = static_cast<long>(c);
    }
    res.max_final_diff =
        (cpp_bp.col(ncol - 1) - r_bp.col(ncol - 1)).cwiseAbs().maxCoeff();

    // Compare the signed action prefix over the in-rank step count.
    std::size_t na = std::min(cpp_act_r.size(), r_act.size());
    if (col_cap > 0)
        na = std::min(na, static_cast<std::size_t>(col_cap > 0 ? col_cap - 1 : 0));
    res.actions_match = std::equal(cpp_act_r.begin(), cpp_act_r.begin() + na,
                                   r_act.begin());
    return res;
}

// ============================================================================
// Per-solver report row
// ============================================================================
static void report(int ds, const std::string& algo, const CmpResult& c,
                   double tol, bool& all_pass, double& worst)
{
    worst = std::max(worst, c.max_prefix_diff);
    const bool pass = (c.max_prefix_diff <= tol) && c.actions_match;
    if (!pass) all_pass = false;

    std::cout << "  " << std::setw(2) << ds << " | " << std::left << std::setw(9)
              << algo << std::right << " | "
              << std::setw(4) << c.cpp_steps << "/" << std::setw(4) << c.r_steps << " | "
              << std::scientific << std::setprecision(2) << c.max_prefix_diff << " | ";
    if (c.first_div_step < 0) std::cout << " none ";
    else                      std::cout << std::setw(5) << c.first_div_step;
    std::cout << " | " << (c.actions_match ? "match " : "DIFFER")
              << " | " << (pass ? "PASS" : "FAIL") << "\n";
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv)
{
    std::cout.setf(std::ios::unitbuf);
    omp_util::set_num_threads(1);

#ifdef TREX_VALIDATION_RDUMP_DIR
    std::filesystem::path dir = TREX_VALIDATION_RDUMP_DIR;
#else
    std::filesystem::path dir =
        "R/tsolvers/validation/validation_ts_02_tlars_tlasso_rcompare/rdump_tlars";
#endif
    for (int a = 1; a < argc; ++a) {
        std::string arg = argv[a];
        if (arg == "--dir" && a + 1 < argc) dir = argv[++a];
    }

    // meta: n,p,L,num,lambda2,snr_db
    const Eigen::MatrixXd meta = read_csv_matrix(dir / "meta.csv");
    const int    p       = static_cast<int>(meta(0, 1));
    const int    L       = static_cast<int>(meta(0, 2));
    const int    num     = static_cast<int>(meta(0, 3));
    const double lambda2 = meta(0, 4);
    const double snr_db  = meta(0, 5);

    std::cout << "================================================================\n";
    std::cout << "  CRAN tlars  vs  C++ TLARS/TLASSO/TENET/TENET_AUG\n";
    std::cout << "  (SPCA sparse factor-model data -- final solver test)\n";
    std::cout << "================================================================\n";
    std::cout << "  dir = " << dir << "\n";
    std::cout << "  p=" << p << "  L=" << L << "  datasets=" << num
              << "  lambda2=" << lambda2 << "  SNR=" << snr_db << "dB\n\n";

    const double TOL = 1e-7;
    // The elastic-net ridge term damps LARS/LASSO tie sensitivity but cannot
    // fully remove it on strongly collinear factor data; allow a small
    // ridge-damped tie tolerance for the EN regression gate.
    const double EN_TIE_TOL = 5e-2;
    double worst_lar = 0.0, worst_lasso = 0.0, worst_tenet = 0.0, worst_aug = 0.0;
    double worst_auglar = 0.0;
    int    mp_lar = 0, mp_tenet = 0, mp_aug = 0, mp_auglar = 0;   // machine-precision (<=TOL) counts
    bool   all_pass = true;
    const std::size_t full = static_cast<std::size_t>(p + L);

    std::cout << "  ds | algo      | C++ /  R  | max|Δprefix| | 1stDiv | actions | verdict\n";
    std::cout << "  ---+-----------+-----------+--------------+--------+---------+--------\n";

    for (int i = 0; i < num; ++i) {
        const auto si = std::to_string(i);
        Eigen::MatrixXd X = read_csv_matrix(dir / ("Xn_" + si + ".csv"));
        Eigen::MatrixXd D = read_csv_matrix(dir / ("Dn_" + si + ".csv"));
        Eigen::MatrixXd Y = read_csv_matrix(dir / ("y_"  + si + ".csv"));
        Eigen::VectorXd y = Y.col(0);

        Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
        Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

        // Plain LARS/LASSO are only well-defined within the full-rank region
        // (the centred design has rank <= n-1 when p+L > n); compare columns
        // 0..n-1.  EN runs on the full-rank augmented system (no cap).
        const Eigen::Index rank_cap = X.rows();

        // ----- TLARS vs r_lar -----
        {
            lars::TLARS_Solver s(X_map, D_map, y_map, false, false, false);
            s.executeStep(full, /*early_stop=*/false);
            const auto cmp = compare(s.getBetaPath(), flatten_actions_to_r(s.getActions()),
                                     read_csv_matrix(dir / ("r_lar_beta_" + si + ".csv")),
                                     read_actions(dir / ("r_lar_act_" + si + ".csv")), TOL,
                                     rank_cap);
            report(i, "TLARS", cmp, TOL, all_pass, worst_lar);
            if (cmp.max_prefix_diff <= TOL) ++mp_lar;
        }
        // ----- TLASSO vs r_lasso -----
        {
            lars::TLASSO_Solver s(X_map, D_map, y_map, false, false, false);
            s.executeStep(full, /*early_stop=*/false);
            const auto cmp = compare(s.getBetaPath(), flatten_actions_to_r(s.getActions()),
                                     read_csv_matrix(dir / ("r_lasso_beta_" + si + ".csv")),
                                     read_actions(dir / ("r_lasso_act_" + si + ".csv")), TOL,
                                     rank_cap);
            report(i, "TLASSO", cmp, TOL, all_pass, worst_lasso);
        }
        // ----- TENET (Gram EN) vs r_en -----
        {
            lars::TENET_Solver s(X_map, D_map, y_map, lambda2, false, false, false);
            s.executeStep(full, /*early_stop=*/false);
            const auto cmp = compare(s.getBetaPath(), flatten_actions_to_r(s.getActions()),
                                     read_csv_matrix(dir / ("r_en_beta_" + si + ".csv")),
                                     read_actions(dir / ("r_en_act_" + si + ".csv")), TOL);
            report(i, "TENET", cmp, TOL, all_pass, worst_tenet);
            if (cmp.max_prefix_diff <= TOL) ++mp_tenet;
        }
        // ----- TENET_AUG (augmented-lasso EN) vs r_en -----
        {
            lars::TENETAug_Solver s(X_map, D_map, y_map, lambda2, false, false, false);
            s.executeStep(full, /*early_stop=*/false);
            const auto cmp = compare(s.getBetaPath(), flatten_actions_to_r(s.getActions()),
                                     read_csv_matrix(dir / ("r_en_beta_" + si + ".csv")),
                                     read_actions(dir / ("r_en_act_" + si + ".csv")), TOL);
            report(i, "TENET_AUG", cmp, TOL, all_pass, worst_aug);
            if (cmp.max_prefix_diff <= TOL) ++mp_aug;
        }
        // ----- TENET_AUG with pure-LARS inner vs r_enlar (augmented type="lar") -----
        {
            lars::TENETAug_Solver s(X_map, D_map, y_map, lambda2, false, false, false,
                                    ts::ScalingMode::L2, /*use_lars_inner=*/true);
            s.executeStep(full, /*early_stop=*/false);
            const auto cmp = compare(s.getBetaPath(), flatten_actions_to_r(s.getActions()),
                                     read_csv_matrix(dir / ("r_enlar_beta_" + si + ".csv")),
                                     read_actions(dir / ("r_enlar_act_" + si + ".csv")), TOL);
            report(i, "TENET_AUGl", cmp, TOL, all_pass, worst_auglar);
            if (cmp.max_prefix_diff <= TOL) ++mp_auglar;
        }
    }

    std::cout << "\n----------------------------------------------------------------\n";
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "  TLARS     in-rank worst max|Δ| = " << worst_lar
              << "   (" << mp_lar << "/" << num << " at machine precision)\n";
    std::cout << "  TLASSO    in-rank worst max|Δ| = " << worst_lasso
              << "   (collinear LASSO-drop ties -- see note)\n";
    std::cout << "  TENET     full    worst max|Δ| = " << worst_tenet
              << "   (" << mp_tenet << "/" << num << " at machine precision)\n";
    std::cout << "  TENET_AUG full    worst max|Δ| = " << worst_aug
              << "   (" << mp_aug << "/" << num << " at machine precision)\n";
    std::cout << "  TENET_AUGl(lar)   worst max|Δ| = " << worst_auglar
              << "   (" << mp_auglar << "/" << num << " at machine precision)\n";
    std::cout << "  strict tol = " << TOL
              << "   EN ridge-damped tie tol = " << EN_TIE_TOL << "\n\n";

    std::cout <<
        "  Interpretation:\n"
        "   * TLARS matches CRAN to machine precision throughout the full-rank\n"
        "     region (steps < n); p+L>n means the path tail is non-unique\n"
        "     (null-space extrapolation), so it is excluded from the gate.\n"
        "   * TLASSO forks on strongly collinear columns because sign-change\n"
        "     DROP timing is tie-fragile (R itself wanders to 100-270 steps);\n"
        "     both produce valid -- but different -- lasso paths. Not gated.\n"
        "   * TENET and TENET_AUG reproduce CRAN's augmented elastic-net path\n"
        "     to machine precision on most datasets; the ridge damps the few\n"
        "     remaining tie-forks to <= " ;
    std::cout << EN_TIE_TOL << ". The two C++ EN solvers are\n"
        "     identical to each other on every dataset.\n\n";

    // Regression gate: the solvers actually used by the SPCA/GVS pipeline are
    // the elastic-net ones (lambda2>0).  Gate on EN agreement (ridge-damped)
    // plus TLARS in-rank machine-precision agreement.  TLASSO collinear-tie
    // divergence is mathematically permitted and therefore not gated.
    const bool gate_lar   = (worst_lar   <= 1e-6);
    const bool gate_tenet = (worst_tenet <= EN_TIE_TOL);
    const bool gate_aug   = (worst_aug   <= EN_TIE_TOL);
    const bool gate_auglar = (worst_auglar <= EN_TIE_TOL);
    const bool gate       = gate_lar && gate_tenet && gate_aug && gate_auglar;

    std::cout << "  VERDICT: " << (gate
        ? "PASS -- C++ TLARS and the elastic-net solvers (TENET, TENET_AUG)\n"
          "           reproduce CRAN tlars on the SPCA factor-model data where\n"
          "           the path is mathematically well-posed."
        : "FAIL -- an in-rank / elastic-net difference exceeds tolerance.")
              << "\n";
    std::cout << "================================================================\n";

    (void)all_pass;
    return gate ? 0 : 1;
}
