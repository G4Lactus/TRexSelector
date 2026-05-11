// ==================================================================================
// demo_mlm_03_hac_mmap.cpp
// ==================================================================================
/**
 * @file demo_mlm_03_hac_mmap.cpp
 *
 * @brief LSH Linkage Comparison at biobank-representative scale via memory-mapped I/O.
 *
 * @details Replicates the geometry of demo_02_hac's `demo_lsh_linkage_comparison` at
 * extreme scale (N=100,000 samples, P=400,000 variables, K=10 clusters) without ever
 * allocating the full 320 GB matrix in RAM. The dataset is generated and standardised
 * in-place via a memory-mapped file; the OS page cache streams data on demand.
 *
 * The demo is organised into three sequential phases:
 *
 *   Phase 1 | Generation   : Latent-factor correlated blocks are written column-by-column
 *                            into the mmap file in parallel.  RAM cost is bounded by the
 *                            latent signal matrix (N × K ≈ 8 MB) and the OS page cache.
 *
 *   Phase 2 | Standardise  : Each column is mean-centred and L2-normalised in-place
 *                            (required for Pearson-correlation distances).
 *
 *   Phase 3 | Clustering   : Four linkage strategies are benchmarked under the pure
 *                            O(1) LSH Correlation Approximation:
 *                              A. Single Linkage  – SLINK
 *                              B. Complete Linkage – NN-Chain
 *                              C. Average Linkage  – UPGMA (NN-Chain)
 *                              D. Weighted Average – WPGMA (NN-Chain)
 *
 * Expected result: all four methods achieve ARI ≈ 1.0000 because the block geometry
 * creates well-separated clusters in correlation space; the O(1) SimHash approximation
 * recovers the exact solution without paying the O(P) exact-distance cost.
 *
 * @note  The memory-mapped file path is injected at compile time via the CMake
 *        TREX_DEMO_MMAP_PATH definition (defaults to a relative path if not set).
 *        The file is removed automatically on clean exit.
 */
// ==================================================================================

// std includes
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// ml_methods includes
#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_dispatcher.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/dendrogram_utils.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/distance_policy.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/generic_linkage_core.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/nnchain_core.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/slink_core.hpp>

// utils includes
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/memmap/memory_mapped_matrix.hpp>
#include <utils/openmp/utils_openmp.hpp>

// ==================================================================================
// Namespace aliases
// ==================================================================================

namespace hac      = trex::ml_methods::clustering::hierarchical::agglomerative;
namespace memmap   = trex::utils::memmap;
namespace omp_utils = trex::utils::openmp;
namespace cdiagnostics = trex::utils::eval::cdiagnostics;

// Type aliases for the memory-mapped matrix and its Eigen views
using MmapMatrix    = memmap::MemoryMappedMatrix<double>;
using EigenMap      = MmapMatrix::MapType;        ///< Eigen::Map<MatrixXd, ColMajor> (mutable)
using ConstEigenMap = MmapMatrix::ConstMapType;   ///< Eigen::Map<const MatrixXd, ColMajor>


// ==================================================================================
// Part 1: Data Generation, Preprocessing, and Evaluation
// ==================================================================================

// ----------------------------------------------------------------------------------

/**
 * @brief Computes the Adjusted Rand Index (ARI) to evaluate clustering accuracy.
 *
 * @param true_labels Ground-truth cluster assignments (size P).
 * @param pred_labels Predicted cluster assignments (size P).
 * @return ARI in [-1, 1]; perfect agreement yields 1.0.
 */
double compute_ari(
    const std::vector<Eigen::Index>& true_labels,
    const std::vector<Eigen::Index>& pred_labels)
{
    if (true_labels.size() != pred_labels.size() || true_labels.empty()) return 0.0;

    const Eigen::Index n        = static_cast<Eigen::Index>(true_labels.size());
    const Eigen::Index max_true = *std::max_element(true_labels.begin(),
                                                    true_labels.end());
    const Eigen::Index max_pred = *std::max_element(pred_labels.begin(),
                                                    pred_labels.end());

    // Contingency table
    Eigen::MatrixXd contingency = Eigen::MatrixXd::Zero(max_true + 1, max_pred + 1);
    for (Eigen::Index i = 0; i < n; ++i)
        contingency(true_labels[i], pred_labels[i])++;

    auto choose2 = [](double x) { return x * (x - 1.0) / 2.0; };

    double sum_comb_c = 0.0;
    for (Eigen::Index i = 0; i <= max_true; ++i)
        for (Eigen::Index j = 0; j <= max_pred; ++j)
            sum_comb_c += choose2(contingency(i, j));

    double sum_comb_a = 0.0;
    for (Eigen::Index i = 0; i <= max_true; ++i)
        sum_comb_a += choose2(contingency.row(i).sum());

    double sum_comb_b = 0.0;
    for (Eigen::Index i = 0; i <= max_pred; ++i)
        sum_comb_b += choose2(contingency.col(i).sum());

    const double n_choose_2    = choose2(static_cast<double>(n));
    const double expected_index = (sum_comb_a * sum_comb_b) / n_choose_2;
    const double max_index      = (sum_comb_a + sum_comb_b) / 2.0;

    if (max_index == expected_index) return 1.0;
    return (sum_comb_c - expected_index) / (max_index - expected_index);
}

// ----------------------------------------------------------------------------------

/** @brief Configuration parameters for the synthetic data generator. */
struct DataGenConfig {
    Eigen::Index n_samples;
    Eigen::Index p_features;
    Eigen::Index k_clusters;
};

// ----------------------------------------------------------------------------------

/**
 * @brief Computes per-feature ground-truth cluster labels from block sizes.
 *
 * @param block_sizes Exact number of features per cluster (size = k_clusters).
 * @param k_clusters  Number of clusters.
 * @return Vector of length P containing the block label for each feature.
 */
std::vector<Eigen::Index> build_true_labels(
    const std::vector<Eigen::Index>& block_sizes,
    Eigen::Index                      k_clusters)
{
    Eigen::Index total_features = 0;
    for (auto s : block_sizes) total_features += s;

    std::vector<Eigen::Index> true_labels(total_features);
    Eigen::Index current = 0;
    for (Eigen::Index k = 0; k < k_clusters; ++k)
        for (Eigen::Index j = 0; j < block_sizes[k]; ++j)
            true_labels[current++] = k;

    return true_labels;
}

// ----------------------------------------------------------------------------------

/**
 * @brief Generates correlated variable blocks directly into a memory-mapped matrix.
 *
 * @details Uses a Latent Factor Model:
 * @code
 *   X(i, j) = mu_k + sigma_k * [ signal * latent(i, k) + noise * epsilon(i, j) ]
 * @endcode
 * where `k = block(j)`. The latent signal matrix (N × K ≈ 8 MB) is held in RAM;
 * all N × P features are written column-by-column into the mmap, never requiring
 * the full matrix in RAM simultaneously.
 *
 * @param map         Writable Eigen::Map onto the memory-mapped file (N × P, ColMajor).
 * @param config      Structural dimensions {n_samples, p_features, k_clusters}.
 * @param block_sizes Exact number of features per cluster (must sum to p_features).
 * @param true_labels Pre-computed per-feature block labels (size = p_features).
 * @param centers     Per-block mean offsets (size = k_clusters).
 * @param std_devs    Per-block scale multipliers (size = k_clusters).
 */
void generate_correlated_variables_to_mmap(
    EigenMap&                          map,
    const DataGenConfig&               config,
    const std::vector<Eigen::Index>&   true_labels,
    const std::vector<double>&         centers,
    const std::vector<double>&         std_devs)
{
    // -- Latent signals: (N × K) in RAM  (~8 MB for N=100k, K=10) -----------------
    Eigen::MatrixXd latent(config.n_samples, config.k_clusters);
    {
        std::mt19937 rng(42);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (Eigen::Index b = 0; b < config.k_clusters; ++b)
            for (Eigen::Index i = 0; i < config.n_samples; ++i)
                latent(i, b) = dist(rng);
    }

    constexpr double signal_strength = 0.85;
    constexpr double noise_strength  = 0.15;

    // -- Stream features column-by-column into the mmap (ColMajor: cache-optimal) -
    #pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        std::mt19937 local_rng(495 + thread_id);
        std::normal_distribution<double> local_dist(0.0, 1.0);

        #pragma omp for schedule(static)
        for (Eigen::Index j = 0; j < config.p_features; ++j) {
            const Eigen::Index blk  = true_labels[j];
            const double       mu   = centers[blk];
            const double       sigma = std_devs[blk];

            for (Eigen::Index i = 0; i < config.n_samples; ++i) {
                double base = signal_strength * latent(i, blk)
                            + noise_strength  * local_dist(local_rng);
                map(i, j) = mu + sigma * base;
            }
        }
    }
}

// ----------------------------------------------------------------------------------

/**
 * @brief In-place mean-centring and L2-normalisation (Pearson-correlation scaling).
 *
 * @details Operates column-by-column which is cache-friendly for ColMajor matrices
 * and sequential-streaming-optimal for memory-mapped files.
 *
 * @param mat Any Eigen dense expression with writable columns.
 */
template <typename Derived>
void standardize_for_correlation(Eigen::MatrixBase<Derived>& mat) {
    #pragma omp parallel for schedule(static)
    for (Eigen::Index j = 0; j < mat.cols(); ++j) {
        const double mean = mat.col(j).mean();
        mat.col(j).array() -= mean;

        const double norm = mat.col(j).norm();
        if (norm > 1e-12)
            mat.col(j) /= norm;
        else
            mat.col(j).setZero();
    }
}


// ==================================================================================
// Part 2: Demo
// ==================================================================================

/**
 * @brief Mmap-backed LSH Linkage Comparison at biobank-representative scale.
 *
 * Generates N=100,000 × P=400,000 correlated variables in K=10 imbalanced blocks,
 * standardises them, then benchmarks four HAC linkage strategies under the O(1)
 * LSH Correlation Approximation. Disk footprint ≈ 320 GB; RAM footprint is
 * determined by the OS page cache and the O(P) HAC bookkeeping structures.
 */
void demo_lsh_linkage_comparison_mmap() {
    cdiagnostics::print_section_header(
        "Demo 03: LSH Linkage Comparison via Memory-Mapped I/O (N=100k, P=400k, K=10)");

    // -------------------------------------------------------------------------
    // 0. Configuration
    // -------------------------------------------------------------------------

    // File path injected by CMake; falls back to relative path if not set.
#ifdef TREX_DEMO_MMAP_PATH
    const std::string MMAP_PATH = TREX_DEMO_MMAP_PATH;
#else
    const std::string MMAP_PATH = "trex_demo03_hac_mmap.bin";
#endif

    const DataGenConfig config {
        .n_samples  = 100'000,
        .p_features = 400'000,
        .k_clusters = 10
    };

    // Imbalanced block sizes — 10 blocks summing exactly to 400,000.
    // The deliberate imbalance (ranging from 33k to 50k features) tests
    // robustness of the LSH approximation to non-uniform cluster sizes.
    const std::vector<Eigen::Index> block_sizes = {
        50'000, 48'000, 45'000, 42'000, 40'000,
        38'000, 35'000, 33'000, 35'000, 34'000
    };
    static_assert(
        50'000 + 48'000 + 45'000 + 42'000 + 40'000 +
        38'000 + 35'000 + 33'000 + 35'000 + 34'000 == 400'000,
        "block_sizes must sum to p_features"
    );

    // Per-block cluster centres — spread across [-5.0, +5.0]
    const std::vector<double> centers = {
         0.0,  2.0, -2.0,  4.0, -4.0,
         1.5, -1.5,  3.0, -3.0,  5.0
    };

    // Per-block noise scales — varied to create realistic heteroscedasticity
    const std::vector<double> std_devs = {
        1.5, 1.2, 0.8, 2.0, 3.5,
        1.0, 1.8, 0.6, 2.5, 1.3
    };

    // Precompute true labels — deterministic from block_sizes, O(P)
    const std::vector<Eigen::Index> true_labels =
        build_true_labels(block_sizes, config.k_clusters);

    // Disk footprint
    const std::size_t file_bytes =
        static_cast<std::size_t>(config.n_samples) *
        static_cast<std::size_t>(config.p_features) * sizeof(double);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Memory-mapped file : " << MMAP_PATH << "\n";
    std::cout << "File size          : "
              << static_cast<double>(file_bytes) / static_cast<double>(1ULL << 30)
              << " GB\n";
    std::cout << "Latent RAM (N x K)   : "
              << static_cast<double>(config.n_samples * config.k_clusters * sizeof(double))
                  / (1ULL << 20)
              << " MB\n\n";


    // -------------------------------------------------------------------------
    // Phase 1 — Data generation (ReadWrite)
    // Phase 2 — Standardisation (in-place, same ReadWrite mapping)
    // Skipped when the mmap file already exists and has the expected size.
    // -------------------------------------------------------------------------
    if (!std::filesystem::exists(MMAP_PATH) ||
         std::filesystem::file_size(MMAP_PATH) != file_bytes) {
        std::cout << "=== Phase 1: Generating correlated variables into mmap ===\n";
        std::cout << "    N=" << config.n_samples
                  << ", P=" << config.p_features
                  << ", K=" << config.k_clusters << "\n";

        MmapMatrix mmap(MMAP_PATH,
                        static_cast<std::size_t>(config.n_samples),
                        static_cast<std::size_t>(config.p_features),
                        memmap::AccessMode::ReadWrite);
        EigenMap X_rw = mmap.getMap();

        const double t0 = omp_get_wtime();
        generate_correlated_variables_to_mmap(
            X_rw, config, true_labels, centers, std_devs);
        const double t1 = omp_get_wtime();

        std::cout << "    Done. (" << std::setprecision(1) << (t1 - t0) << " s)\n\n";

        std::cout << "=== Phase 2: Standardising columns (mean=0, L2=1) in-place ===\n";

        const double t2 = omp_get_wtime();
        standardize_for_correlation(X_rw);
        const double t3 = omp_get_wtime();

        std::cout << "    Done. (" << std::setprecision(1) << (t3 - t2) << " s)\n\n";

        // MmapMatrix destructor closes and flushes the mapping here (RAII).
    } else {
        std::cout << "=== Phases 1 & 2: Skipped (mmap file exists with correct size) ===\n\n";
    }


    // -------------------------------------------------------------------------
    // Phase 3 — Clustering (reopen ReadWrite; HAC does not mutate the data)
    // -------------------------------------------------------------------------
    std::cout << "=== Phase 3: Reopening mmap for clustering ===\n";

    MmapMatrix mmap_cluster(MMAP_PATH,
                            static_cast<std::size_t>(config.n_samples),
                            static_cast<std::size_t>(config.p_features),
                            memmap::AccessMode::ReadWrite);
    EigenMap X_map = mmap_cluster.getMap();

    // Pure O(1) LSH Correlation Approximation (64-bit SimHash)
    std::cout << "    Distance policy: Correlation LSH Approximation (O(1) SimHash)\n\n";

    using LshDistPol = hac::DistancePolicy<
        EigenMap, hac::DistanceMetric::Correlation_LSH_Approx
    >;

    // Test A — Single Linkage via Dispatcher (Routes to SLINK)
    // -------------------------------------------------------------------------
    {
        std::cout << "[Test A] Single Linkage (SLINK + LSH):\n";

        const double t0 = omp_get_wtime();
        auto merges = hac::AgglomerativeClustering::cluster<
            EigenMap, LshDistPol, hac::LinkageMethod::Single>(X_map);
        const double t1 = omp_get_wtime();

        auto labels = hac::DendrogramUtils::cut_tree(
            merges, X_map.cols(), config.k_clusters);
        const double ari = compute_ari(true_labels, labels);

        std::cout << "    Time : " << std::setprecision(4) << (t1 - t0) << " s\n";
        std::cout << "    ARI  : " << std::setprecision(4) << ari << " / 1.0000\n\n";
    }

    // Test B — Complete Linkage & WPGMA (Mathematically Banned at this scale)
    // -------------------------------------------------------------------------
    {
        std::cout << "[Test B] Complete Linkage & WPGMA:\n";
        std::cout << "    [SKIPPED] These methods lack a geometric center of mass.\n";
        std::cout << "    They strictly require the O(P^2) Lance-Williams matrix.\n";
        std::cout << "    At P=400,000, this requires 1.2 TB of NVMe I/O, which is intractable.\n\n";
    }

    // Test C — Average Linkage (UPGMA) via Dispatcher
    // Routes to Projected 64D RAM Engine!
    // -------------------------------------------------------------------------
    {
        std::cout << "[Test C] UPGMA Average Linkage (Projected 64D RAM Engine + LSH):\n";

        const double t0 = omp_get_wtime();
        // Notice we don't need use_mmap=true anymore. The dispatcher intercepts
        // this and builds the compressed 204 MB RAM matrix.
        auto merges = hac::AgglomerativeClustering::cluster<
            EigenMap, LshDistPol, hac::LinkageMethod::Average>(X_map);
        const double t1 = omp_get_wtime();

        auto labels = hac::DendrogramUtils::cut_tree(
            merges, X_map.cols(), config.k_clusters);
        const double ari = compute_ari(true_labels, labels);

        std::cout << "    Time : " << std::setprecision(4) << (t1 - t0) << " s\n";
        std::cout << "    ARI  : " << std::setprecision(4) << ari << " / 1.0000\n\n";
    }

    // Test D — Ward's Minimum Variance via Dispatcher
    // Routes to Projected 64D RAM Engine!
    // -------------------------------------------------------------------------
    {
        std::cout << "[Test D] Ward's Minimum Variance (Projected 64D RAM Engine + LSH):\n";

        const double t0 = omp_get_wtime();
        // The dispatcher intercepts this as well and runs it purely in RAM.
        auto merges = hac::AgglomerativeClustering::cluster<
            EigenMap, LshDistPol, hac::LinkageMethod::Ward>(X_map);
        const double t1 = omp_get_wtime();

        auto labels = hac::DendrogramUtils::cut_tree(
            merges, X_map.cols(), config.k_clusters);
        const double ari = compute_ari(true_labels, labels);

        std::cout << "    Time : " << std::setprecision(4) << (t1 - t0) << " s\n";
        std::cout << "    ARI  : " << std::setprecision(4) << ari << " / 1.0000\n\n";
    }

    std::cout << "[Demo 03 Complete]\n\n";


    // -------------------------------------------------------------------------
    // Cleanup: remove the mmap file
    // -------------------------------------------------------------------------
    {
        namespace fs = std::filesystem;
        std::error_code ec;
        if (fs::remove(MMAP_PATH, ec))
            std::cout << "Cleaned up mmap file: " << MMAP_PATH << "\n";
        else
            std::cerr << "[WARN] Could not remove mmap file: " << ec.message() << "\n";
    }
}



// ==================================================================================
// 3. Main
// ==================================================================================

void init_omp_environment(int threads = 6) {
    omp_set_num_threads(threads);
    omp_set_schedule(omp_sched_static, 0);
}

int main() {
    std::cout << std::unitbuf; // Flush after every write — keeps log file current.
    std::cout << "\n";
    cdiagnostics::print_section_header(
        "HAC Memory-Mapped Demo Suite (N=100k x P=400k)");

    omp_utils::print_info();
    init_omp_environment(6);
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    try {
        demo_lsh_linkage_comparison_mmap();
        cdiagnostics::print_section_header("Demo 03 completed successfully");
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n\n";
        return 1;
    }

    return 0;
}
