// ============================================================================
// validation_mlm_hac_02_gvs_dummy_rcompare.cpp
// ============================================================================
/**
 * @file validation_mlm_hac_02_gvs_dummy_rcompare.cpp
 *
 * @brief Verifies that the C++ GVS dummy matrix is built on the SAME clusters
 *        (and the SAME per-cluster generating covariance) that R's
 *        `TRexSelector:::add_dummies_GVS()` derives from the dendrogram cut.
 *
 * @details
 *  The user observation: "the dummy matrix is generated based on the clustering
 *  result of X -- the R dummy matrix is built upon the clusters the dendrogram
 *  finds at some cut point; we must also check that for C++."
 *
 *  Both implementations follow the identical recipe (verified by source read):
 *
 *    R  TRexSelector:::add_dummies_GVS(X, num_dummies, corr_max):
 *       d = as.dist(1 - abs(cov2cor(cov(X))))     # == 1 - |cor(X)|
 *       fit      = hclust(d, method = "single")
 *       clusters = cutree(fit, h = 1 - corr_max)  # CUT POINT
 *       dummy_m ~ MASS::mvrnorm(n, 0, cov(sub_X)) # per-cluster MVN
 *
 *    C++ TRexGVSSelector::setupGVS_Cluster() + finalizeSetup():
 *       CorrDist  = 1 - |corr| on centred columns (== Pearson)
 *       merges    = cluster<Map, CorrDist, Single>(X)
 *       labels    = cut_tree_by_height(merges, p, 1 - corr_max)   # CUT POINT
 *       Sigma_m   = (1/(n-1)) X_m^T X_m  (== cov(sub_X) on centred X)
 *       L_m       = chol(Sigma_m); block = Z * L_m^T  (rows ~ N(0, Sigma_m))
 *
 *  The mvrnorm / coloring DRAWS are random and cannot be compared by value, but
 *  the two DETERMINISTIC ingredients that the dummy block depends on can:
 *
 *   (1) the cluster PARTITION at the cut point h = 1 - corr_max
 *       (R cutree(h)  vs  C++ cut_tree_by_height(h), both inclusive '<='), and
 *   (2) the per-cluster generating COVARIANCE, encoded as the p x p
 *       block-diagonal matrix  Sigma_gen[i,j] = cov(X)[i,j] if same cluster
 *       else 0 -- exactly the covariance the dummies are sampled from.
 *
 *  This demo re-runs the C++ GVS clustering setup on each Xn_<i>.csv for every
 *  corr_max in gvs_corrmax.csv and compares against the R dumps
 *      r_gvs_labels_<i>_<cm>.csv   (1 x p ints, 1-based cluster id)
 *      r_gvs_sigma_<i>_<cm>.csv    (p x p generating covariance)
 *  produced by
 *  R/tsolvers/validation/validation_ts_02_tlars_tlasso_rcompare/demo_ts_compare_tlars_tlasso.R.
 *
 *  PASS if, for every (dataset, corr_max): the partition ARI = 1 (identical
 *  clusters) AND max|Sigma_gen^cpp - Sigma_gen^R| <= 1e-9.  (The C++ setup adds
 *  a 1e-10 ridge to the Cholesky diagonal for numerical stability only; that is
 *  below this tolerance and is intentionally excluded from Sigma_gen here, which
 *  compares the mathematical generating covariance.)
 */
// ============================================================================

#include <algorithm>
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
    if (std::abs(denom) < 1e-15) return 1.0;   // both partitions trivial/identical
    return (sum_ij - expected) / denom;
}

/** @brief Zero-padded 3-digit tag from corr_max (0.50 -> "050"), matching R. */
static std::string corrmax_tag(double cm)
{
    std::ostringstream os;
    os << std::setw(3) << std::setfill('0')
       << static_cast<int>(std::lround(cm * 100.0));
    return os.str();
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
    const int p   = static_cast<int>(meta(0, 1));
    const int num = static_cast<int>(meta(0, 3));

    // corr_max grid (1 x C)
    const Eigen::MatrixXd cmrow = read_csv_matrix(dir / "gvs_corrmax.csv");
    std::vector<double> corr_max;
    for (Eigen::Index j = 0; j < cmrow.cols(); ++j) corr_max.push_back(cmrow(0, j));

    std::cout << "================================================================\n";
    std::cout << "  C++ GVS dummy-from-clustering  vs  R add_dummies_GVS()\n";
    std::cout << "  cut point h = 1 - corr_max ;  per-cluster generating cov\n";
    std::cout << "================================================================\n";
    std::cout << "  dir = " << dir << "\n";
    std::cout << "  p = " << p << "   datasets = " << num << "   corr_max = {";
    for (std::size_t c = 0; c < corr_max.size(); ++c)
        std::cout << (c ? ", " : "") << corr_max[c];
    std::cout << "}\n\n";

    using MapType  = Eigen::Map<Eigen::MatrixXd>;
    using CorrDist = hac::DistancePolicy<MapType, hac::DistanceMetric::Correlation>;

    const double SIGMA_TOL = 1e-9;
    double worst_sigma = 0.0;
    double min_ari     = 1.0;
    long   total_part_mismatch = 0;
    bool   all_pass = true;

    std::cout << "  ds | corr_max | h=1-cm |  M | ARI(part) | max|ΔΣ_gen| | verdict\n";
    std::cout << "  ---+----------+--------+----+-----------+-------------+--------\n";

    for (int i = 0; i < num; ++i) {
        const auto si = std::to_string(i);
        Eigen::MatrixXd X = read_csv_matrix(dir / ("Xn_" + si + ".csv"));
        if (X.cols() != p)
            throw std::runtime_error("Xn column count != p for dataset " + si);
        Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
        const double inv_nm1 = 1.0 / static_cast<double>(X.rows() - 1);

        // C++ single-linkage dendrogram (identical to TRexGVSSelector::runClustering)
        const std::vector<hac::MergeStep> merges =
            hac::AgglomerativeClustering::cluster<MapType, CorrDist,
                                                  hac::LinkageMethod::Single>(X_map);

        for (double cm : corr_max) {
            const std::string tag = corrmax_tag(cm);
            const double height   = 1.0 - cm;

            // ----- C++ GVS setup: cut at h = 1 - corr_max, then group -----
            const std::vector<Eigen::Index> labels =
                hac::DendrogramUtils::cut_tree_by_height(
                    merges, static_cast<Eigen::Index>(p), height);
            const std::vector<std::vector<Eigen::Index>> clusters =
                hac::DendrogramUtils::group_indices_by_label(labels);
            const std::size_t M = clusters.size();

            // C++ generating covariance Sigma_gen[i,j] = (1/(n-1)) X_i.X_j on same cluster
            Eigen::MatrixXd Sig_cpp = Eigen::MatrixXd::Zero(p, p);
            for (const auto& cols : clusters) {
                for (std::size_t a = 0; a < cols.size(); ++a) {
                    for (std::size_t b = a; b < cols.size(); ++b) {
                        const Eigen::Index ci = cols[a];
                        const Eigen::Index cj = cols[b];
                        const double v = inv_nm1 * X.col(ci).dot(X.col(cj));
                        Sig_cpp(ci, cj) = v;
                        Sig_cpp(cj, ci) = v;
                    }
                }
            }

            // ----- R reference -----
            const Eigen::MatrixXd r_lab_row =
                read_csv_matrix(dir / ("r_gvs_labels_" + si + "_" + tag + ".csv"));
            std::vector<Eigen::Index> r_lab(static_cast<std::size_t>(p));
            for (int j = 0; j < p; ++j)
                r_lab[static_cast<std::size_t>(j)] =
                    static_cast<Eigen::Index>(std::lround(r_lab_row(0, j)));

            const Eigen::MatrixXd Sig_r =
                read_csv_matrix(dir / ("r_gvs_sigma_" + si + "_" + tag + ".csv"));

            // ----- compare -----
            const double ari = adjusted_rand_index(labels, r_lab);
            const double dsig = (Sig_cpp - Sig_r).cwiseAbs().maxCoeff();

            min_ari      = std::min(min_ari, ari);
            worst_sigma  = std::max(worst_sigma, dsig);
            const bool part_ok = (ari >= 1.0 - 1e-9);
            if (!part_ok) ++total_part_mismatch;
            const bool pass = part_ok && (dsig <= SIGMA_TOL);
            if (!pass) all_pass = false;

            std::cout << "  " << std::setw(2) << i
                      << " | " << std::fixed << std::setprecision(2) << std::setw(8) << cm
                      << " | " << std::setprecision(3) << std::setw(6) << height
                      << " | " << std::setw(2) << M
                      << " | " << std::setprecision(4) << std::setw(9) << ari
                      << " | " << std::scientific << std::setprecision(2) << std::setw(11) << dsig
                      << " | " << (pass ? "PASS" : "FAIL") << "\n";
        }
    }

    std::cout << "\n----------------------------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  min ARI over all (ds,corr_max)   = " << min_ari << "\n";
    std::cout << "  partition mismatches             = " << total_part_mismatch << "\n";
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "  worst max|ΔΣ_gen|                = " << worst_sigma
              << "   (tol " << SIGMA_TOL << ")\n";
    std::cout << "  VERDICT: " << (all_pass
        ? "PASS -- C++ GVS dummies are built on the SAME dendrogram-cut\n"
          "           clusters AND the SAME per-cluster generating covariance\n"
          "           as R add_dummies_GVS() at every corr_max cut point."
        : "FAIL -- a cluster partition or generating-covariance difference\n"
          "           was found (see rows).") << "\n";
    std::cout << "================================================================\n";

    return all_pass ? 0 : 1;
}
