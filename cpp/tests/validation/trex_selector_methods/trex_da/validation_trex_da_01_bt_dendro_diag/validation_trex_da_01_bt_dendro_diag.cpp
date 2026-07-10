// ==============================================================================
// validation_trex_da_01_bt_dendro_diag.cpp
// ==============================================================================
/**
 * @file validation_trex_da_01_bt_dendro_diag.cpp
 *
 * @brief Single-dataset DA-BT diagnostic harness for the C++/R FDR-gap study.
 *
 * @details
 *  Freezes ONE AR(1)-block dataset (the base config of demo 04) and dumps every
 *  intermediate quantity needed to localise the C++/R FDR discrepancy:
 *
 *    1. The exact design matrix X, response y, and true support -> CSV, so the
 *       R comparison script (diag_bt_dendro_compare.R) clusters the IDENTICAL X.
 *    2. The single-linkage dendrogram merge heights (correlation distance),
 *       reproduced with the SAME clustering call the selector uses internally.
 *    3. The rho_grid (built exactly as TRexDASelector::setupDA_BT) and, for every
 *       rho level, the per-variable group sizes / memberships after cutting the
 *       tree at height = 1 - rho.
 *    4. The full TRexDASelector(BT, Single) result: rho_grid, T_stop, dummy
 *       multiplier L, selected variables, FDP/TPP, and the chosen rho threshold.
 *
 *  Compare diag_cpp_heights.csv / diag_cpp_groups.csv against the R dumps to see
 *  whether the divergence is in the CLUSTERING (heights/groups differ for the
 *  same X) or DOWNSTREAM (clustering matches but Phi / selection differ).
 *
 *  Output goes to this validation's validation_results/ folder (DEMO_OUTPUT_DIR).
 */
// ==============================================================================

#include "trex_da_sim_common.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>

// ==============================================================================

using namespace da_sim;

namespace {
    // Base config — identical to demo 04 (AR(1)-block SNR-sweep base point).
    constexpr int      n         = 150;
    constexpr int      M         = 5;     // AR(1) blocks
    constexpr int      Q         = 5;     // block size; p = M * Q
    constexpr int      p         = M * Q; // 25
    constexpr double   amplitude = 1.0;
    constexpr double   rho       = 0.7;
    constexpr double   snr       = 2.0;
    constexpr double   tFDR      = 0.2;
    constexpr int      K         = 20;
    constexpr unsigned seed      = 2026u;

    const std::string OUT = DEMO_OUTPUT_DIR;
} // namespace


// ------------------------------------------------------------------------------
// Center + L2-normalize each column (mirrors the base T-Rex normalizer). The
// correlation distance divides out the norms, so this reproduces |Pearson corr|
// exactly — i.e. the distance the selector's internal clustering sees.
// ------------------------------------------------------------------------------
static Eigen::MatrixXd center_l2(const Eigen::MatrixXd& X) {
    Eigen::MatrixXd Xc = X;
    for (Eigen::Index j = 0; j < Xc.cols(); ++j) {
        Xc.col(j).array() -= Xc.col(j).mean();
        const double nrm = Xc.col(j).norm();
        if (nrm > 1e-12) { Xc.col(j) /= nrm; }
    }
    return Xc;
}


int main() {
    std::filesystem::create_directories(OUT);

    // --------------------------------------------------------------------------
    // 1. Fixed dataset (deterministic OnePerBlock support).
    // --------------------------------------------------------------------------
    const auto support =
        make_support(SupportPolicy::OnePerBlock, M, p, 0, 0u);
    auto dat = dgp_ar1_block(n, p, support, amplitude, M, rho, snr, seed);

    std::cout << "DA-BT dendrogram diagnostic\n"
              << "  n=" << n << "  p=" << p << "  M=" << M << "  Q=" << Q
              << "  rho=" << rho << "  snr=" << snr << "  seed=" << seed << "\n"
              << "  true support (0-based): ";
    for (auto s : dat.true_support) std::cout << s << " ";
    std::cout << "\n\n";

    // --------------------------------------------------------------------------
    // 2. Export X, y, support so the R script clusters the IDENTICAL matrix.
    // --------------------------------------------------------------------------
    {
        std::ofstream fx(OUT + "diag_X.csv");
        fx << std::setprecision(17);
        for (Eigen::Index i = 0; i < dat.X.rows(); ++i) {
            for (Eigen::Index j = 0; j < dat.X.cols(); ++j) {
                fx << dat.X(i, j);
                if (j + 1 < dat.X.cols()) fx << ",";
            }
            fx << "\n";
        }
    }
    {
        std::ofstream fy(OUT + "diag_y.csv");
        fy << std::setprecision(17);
        for (Eigen::Index i = 0; i < dat.y.rows(); ++i) fy << dat.y(i) << "\n";
    }
    {
        std::ofstream fs(OUT + "diag_support0.csv");  // 0-based indices
        for (auto s : dat.true_support) fs << s << "\n";
    }

    // --------------------------------------------------------------------------
    // 3. Reproduce the selector's clustering: single linkage on |Pearson|.
    // --------------------------------------------------------------------------
    Eigen::MatrixXd Xc = center_l2(dat.X);
    using MapT    = Eigen::Map<Eigen::MatrixXd>;
    using CorrDist = hac::DistancePolicy<MapT, hac::DistanceMetric::Correlation>;
    MapT Xmap(Xc.data(), Xc.rows(), Xc.cols());

    auto merges = hac::AgglomerativeClustering::cluster<
        MapT, CorrDist, hac::LinkageMethod::Single>(Xmap, /*use_mmap=*/false,
                                                    /*verbose=*/false);

    {
        std::ofstream fh(OUT + "diag_cpp_heights.csv");
        fh << "merge,cluster1,cluster2,distance,new_cluster_size\n"
           << std::setprecision(17);
        for (std::size_t k = 0; k < merges.size(); ++k) {
            const auto& m = merges[k];
            fh << k << "," << m.cluster1 << "," << m.cluster2 << ","
               << m.distance << "," << m.new_cluster_size << "\n";
        }
    }

    // --------------------------------------------------------------------------
    // 4. Build rho_grid exactly as TRexDASelector::setupDA_BT and cut per level.
    // --------------------------------------------------------------------------
    const std::size_t grid_len =
        std::min<std::size_t>(20, static_cast<std::size_t>(p));

    std::vector<double> rho_grid_full;
    rho_grid_full.reserve(static_cast<std::size_t>(p));
    for (auto it = merges.rbegin(); it != merges.rend(); ++it)
        rho_grid_full.push_back(1.0 - it->distance);
    rho_grid_full.push_back(1.0);

    std::vector<double> rho_grid(grid_len);
    if (grid_len == 1) {
        rho_grid[0] = rho_grid_full.back();
    } else {
        for (std::size_t i = 0; i < grid_len; ++i) {
            double frac = static_cast<double>(i) * static_cast<double>(p - 1)
                        / static_cast<double>(grid_len - 1);
            std::size_t idx = static_cast<std::size_t>(std::round(frac));
            if (idx >= rho_grid_full.size()) idx = rho_grid_full.size() - 1;
            rho_grid[i] = rho_grid_full[idx];
        }
    }

    {
        std::ofstream fg(OUT + "diag_cpp_groups.csv");
        fg << "rho_level,rho,cut_height,var,group_size,group_members\n"
           << std::setprecision(10);
        for (std::size_t r = 0; r < grid_len; ++r) {
            const double cut_height = 1.0 - rho_grid[r];
            auto labels = hac::DendrogramUtils::cut_tree_by_height(
                merges, static_cast<Eigen::Index>(p), cut_height);
            auto groups = hac::DendrogramUtils::group_indices_by_label(labels);

            for (Eigen::Index j = 0; j < p; ++j) {
                const auto& cluster =
                    groups[static_cast<std::size_t>(labels[static_cast<std::size_t>(j)])];
                fg << r << "," << rho_grid[r] << "," << cut_height << "," << j
                   << "," << cluster.size() << ",";
                for (std::size_t t = 0; t < cluster.size(); ++t) {
                    fg << cluster[t];
                    if (t + 1 < cluster.size()) fg << "|";
                }
                fg << "\n";
            }
        }
    }

    // Console summary: number of clusters + support-var group sizes per level.
    std::cout << "rho-grid cut summary (level: rho -> #clusters | support group sizes)\n";
    for (std::size_t r = 0; r < grid_len; ++r) {
        const double cut_height = 1.0 - rho_grid[r];
        auto labels = hac::DendrogramUtils::cut_tree_by_height(
            merges, static_cast<Eigen::Index>(p), cut_height);
        auto groups = hac::DendrogramUtils::group_indices_by_label(labels);
        std::cout << "  [" << std::setw(2) << r << "] rho="
                  << std::fixed << std::setprecision(4) << rho_grid[r]
                  << "  #clusters=" << groups.size() << "  supp_sizes=";
        for (auto s : dat.true_support) {
            auto lab = labels[static_cast<std::size_t>(s)];
            std::cout << groups[static_cast<std::size_t>(lab)].size() << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // --------------------------------------------------------------------------
    // 5. Full selector run — downstream diagnostics.
    // --------------------------------------------------------------------------
    MapT X_sel(dat.X.data(), dat.X.rows(), dat.X.cols());
    Eigen::Map<Eigen::VectorXd> y_sel(dat.y.data(), dat.y.rows());

    auto da_ctrl = make_da_bt_control();
    da_ctrl.hc_linkage = hac::LinkageMethod::Single;
    auto trex_ctrl = make_trex_control(K);
    // Mirror the sweep's TLARS configuration (default_solvers): a LARS-based
    // solver runs with stagnation control OFF (monotone path), so the T-loop
    // terminates on the FDP criterion / max_T rather than on stagnation.
    trex_ctrl.solver_type           = SolverTypeForTRex::TLARS;
    trex_ctrl.tloop_stagnation_stop = false;
    trex_ctrl.lloop_strategy        = LLoopStrategy::STANDARD;
    da_ctrl.trex_ctrl = trex_ctrl;

    TRexDASelector selector(X_sel, y_sel, tFDR, da_ctrl,
                            static_cast<int>(seed), false);
    selector.select();

    const auto& res     = selector.getDAResult();
    const auto  sel_idx = selector.getSelectedIndices();
    const double fdp    = rates::compute_fdp(sel_idx, dat.true_support);
    const double tpp    = rates::compute_tpp(sel_idx, dat.true_support);

    std::cout << "selector result:\n"
              << "  T_stop=" << selector.getTStop()
              << "  L(dummy mult)=" << selector.getDummyMultiplierL()
              << "  rho_thresh=" << res.rho_thresh << "\n"
              << "  #selected=" << sel_idx.size() << "  FDP=" << fdp
              << "  TPP=" << tpp << "\n  selected (0-based): ";
    for (auto s : sel_idx) std::cout << s << " ";
    std::cout << "\n";

    {
        std::ofstream fr(OUT + "diag_cpp_selector.csv");
        fr << std::setprecision(17);
        fr << "key,value\n";
        fr << "T_stop," << selector.getTStop() << "\n";
        fr << "L," << selector.getDummyMultiplierL() << "\n";
        fr << "rho_thresh," << res.rho_thresh << "\n";
        fr << "n_selected," << sel_idx.size() << "\n";
        fr << "FDP," << fdp << "\n";
        fr << "TPP," << tpp << "\n";
        fr << "selected_0based,";
        for (std::size_t t = 0; t < sel_idx.size(); ++t) {
            fr << sel_idx[t];
            if (t + 1 < sel_idx.size()) fr << "|";
        }
        fr << "\n";
        fr << "rho_grid,";
        for (Eigen::Index i = 0; i < res.rho_grid.size(); ++i) {
            fr << res.rho_grid(i);
            if (i + 1 < res.rho_grid.size()) fr << "|";
        }
        fr << "\n";
    }

    std::cout << "\n[Info] Diagnostics written to: " << OUT << "\n";
    return 0;
}
