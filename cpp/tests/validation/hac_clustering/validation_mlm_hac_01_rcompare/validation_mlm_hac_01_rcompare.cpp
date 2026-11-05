// ============================================================================
// validation_mlm_hac_01_rcompare.cpp
// ============================================================================
/**
 * @file validation_mlm_hac_01_rcompare.cpp
 *
 * @brief Head-to-head check: C++ single-linkage hierarchical clustering vs R
 *        `hclust(method="single")` on the SAME design matrices X.
 *
 * @details
 *  Companion to
 *  R/tsolvers/validation/validation_ts_02_tlars_tlasso_rcompare/demo_ts_compare_tlars_tlasso.R,
 *  which (besides the solver references) clusters the p columns of every Xn_<i>.csv with
 *      hclust(as.dist(1 - abs(cor(Xn))), method = "single")
 *  and dumps:
 *      r_clust_height_<i>.csv   : sorted merge heights (p-1 values, ascending)
 *      r_clust_labels_<i>.csv   : p x p integer matrix; row k = cutree(hc, k=k)
 *
 *  This demo reads each Xn_<i>.csv and runs the C++ engine
 *      AgglomerativeClustering::cluster<Map, DistancePolicy<Correlation>, Single>
 *  whose metric is exactly 1 - |corr| on centred columns (== Pearson, since the
 *  R generator centres the columns; unit-L2 scaling is correlation-invariant).
 *
 *  Two comparisons per dataset:
 *   1. Merge heights: max |Δ| between the C++ sorted merge distances and R's
 *      sorted hclust heights (the dendrogram heights are well-defined even when
 *      internal node IDs / merge-row order differ between implementations).
 *   2. Flat partitions: for every K = 2..p-1, cut_tree(K) in C++ vs cutree(k=K)
 *      in R, compared by the Adjusted Rand Index (label-permutation invariant).
 *      ARI = 1 means the partitions are identical.  We report the minimum ARI
 *      over K and the number of K with ARI < 1.
 *
 *  PASS if heights agree to <= 1e-9 and every K-partition has ARI = 1 on all
 *  datasets (single linkage gives a unique dendrogram when distances are
 *  distinct, which holds for the factor-model data).
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
#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_dispatcher.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/distance_policy.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/dendrogram_utils.hpp>

namespace hac      = trex::ml_methods::clustering::hierarchical::agglomerative;
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


// ============================================================================
// Adjusted Rand Index between two label vectors (permutation invariant)
// ============================================================================
static double adjusted_rand_index(const std::vector<Eigen::Index>& a,
                                  const std::vector<Eigen::Index>& b)
{
    if (a.size() != b.size() || a.empty()) return 0.0;
    const Eigen::Index n  = static_cast<Eigen::Index>(a.size());
    const Eigen::Index ka = *std::max_element(a.begin(), a.end()) + 1;
    const Eigen::Index kb = *std::max_element(b.begin(), b.end()) + 1;

    Eigen::MatrixXd c = Eigen::MatrixXd::Zero(ka, kb);
    for (Eigen::Index i = 0; i < n; ++i) c(a[i], b[i]) += 1.0;

    auto choose2 = [](double x) { return x * (x - 1.0) / 2.0; };

    double sum_ij = 0.0;
    for (Eigen::Index i = 0; i < ka; ++i)
        for (Eigen::Index j = 0; j < kb; ++j) sum_ij += choose2(c(i, j));

    double sum_a = 0.0;
    for (Eigen::Index i = 0; i < ka; ++i) sum_a += choose2(c.row(i).sum());
    double sum_b = 0.0;
    for (Eigen::Index j = 0; j < kb; ++j) sum_b += choose2(c.col(j).sum());

    const double total    = choose2(static_cast<double>(n));
    const double expected = (sum_a * sum_b) / total;
    const double max_idx  = 0.5 * (sum_a + sum_b);
    const double denom    = max_idx - expected;
    if (std::abs(denom) < 1e-15) return 1.0;   // both partitions identical
    return (sum_ij - expected) / denom;
}


// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv)
{
    try {
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
    const int p   = static_cast<int>(meta(0, 1));
    const int num = static_cast<int>(meta(0, 3));

    std::cout << "================================================================\n";
    std::cout << "  C++ single-linkage clustering  vs  R hclust(method=\"single\")\n";
    std::cout << "  metric = 1 - |corr|  on the columns of X (factor-model data)\n";
    std::cout << "================================================================\n";
    std::cout << "  dir = " << dir << "\n";
    std::cout << "  p (objects) = " << p << "   datasets = " << num << "\n\n";

    using MapType  = Eigen::Map<Eigen::MatrixXd>;
    using CorrDist = hac::DistancePolicy<MapType, hac::DistanceMetric::Correlation>;

    const double HEIGHT_TOL = 1e-9;
    double worst_height = 0.0;
    double min_ari      = 1.0;
    long   total_k_mismatch = 0;
    bool   all_pass = true;

    std::cout << "  ds | max|Δheight| | minARI(K) | #K ARI<1 | verdict\n";
    std::cout << "  ---+--------------+-----------+----------+--------\n";

    for (int i = 0; i < num; ++i) {
        const auto si = std::to_string(i);
        Eigen::MatrixXd X = read_csv_matrix(dir / ("Xn_" + si + ".csv"));
        if (X.cols() != p)
            throw std::runtime_error("Xn column count != p for dataset " + si);
        Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());

        // ----- C++ single-linkage dendrogram -----
        const std::vector<hac::MergeStep> merges =
            hac::AgglomerativeClustering::cluster<MapType, CorrDist,
                                                  hac::LinkageMethod::Single>(X_map);

        // (1) merge heights (already ascending by construction)
        std::vector<double> cpp_h;
        cpp_h.reserve(merges.size());
        for (const auto& m : merges) cpp_h.push_back(m.distance);

        const Eigen::MatrixXd r_h = read_csv_matrix(dir / ("r_clust_height_" + si + ".csv"));
        double ds_height = 0.0;
        const std::size_t nh = std::min<std::size_t>(cpp_h.size(),
                                   static_cast<std::size_t>(r_h.rows()));
        for (std::size_t k = 0; k < nh; ++k)
            ds_height = std::max(ds_height, std::abs(cpp_h[k] - r_h(static_cast<Eigen::Index>(k), 0)));
        worst_height = std::max(worst_height, ds_height);

        // (2) per-K partition agreement via ARI
        const Eigen::MatrixXd r_lab = read_csv_matrix(dir / ("r_clust_labels_" + si + ".csv"));
        double ds_min_ari = 1.0;
        long   ds_mismatch = 0;
        for (int K = 2; K < p; ++K) {
            const std::vector<Eigen::Index> cpp_lab =
                hac::DendrogramUtils::cut_tree(merges, p, K);

            std::vector<Eigen::Index> r_row(static_cast<std::size_t>(p));
            for (int j = 0; j < p; ++j)
                r_row[static_cast<std::size_t>(j)] =
                    static_cast<Eigen::Index>(std::lround(r_lab(K - 1, j)));

            const double ari = adjusted_rand_index(cpp_lab, r_row);
            ds_min_ari = std::min(ds_min_ari, ari);
            if (ari < 1.0 - 1e-9) ++ds_mismatch;
        }
        min_ari        = std::min(min_ari, ds_min_ari);
        total_k_mismatch += ds_mismatch;

        const bool pass = (ds_height <= HEIGHT_TOL) && (ds_mismatch == 0);
        if (!pass) all_pass = false;

        std::cout << "  " << std::setw(2) << i << " | "
                  << std::scientific << std::setprecision(2) << std::setw(12) << ds_height
                  << " | " << std::fixed << std::setprecision(4) << std::setw(9) << ds_min_ari
                  << " | " << std::setw(8) << ds_mismatch
                  << " | " << (pass ? "PASS" : "FAIL") << "\n";
    }

    std::cout << "\n----------------------------------------------------------------\n";
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "  worst max|Δheight|        = " << worst_height
              << "   (tol " << HEIGHT_TOL << ")\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  min ARI over all (ds,K)   = " << min_ari << "\n";
    std::cout << "  total K-partition mismatch = " << total_k_mismatch << "\n";
    std::cout << "  VERDICT: " << (all_pass
        ? "PASS -- C++ single linkage reproduces R hclust(\"single\") exactly\n"
          "           (identical dendrogram heights AND identical partitions)."
        : "FAIL -- a height or partition difference was found (see rows).") << "\n";
    std::cout << "================================================================\n";

        return all_pass ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "ERROR: unknown exception\n";
        return 1;
    }
}
