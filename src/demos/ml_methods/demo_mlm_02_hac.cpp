// ==================================================================================
// demo_mlm_02_hac.cpp
// ==================================================================================
/**
 * @file demo_mlm_02_hac.cpp
 *
 * @brief Demonstration of dendrogram visualization for hierarchical clustering.
 *
 * @details We analyze the following scenarios:
 * |- Demo 1: Clustering samples using SLINK with Euclidean distance.
 * |- Demo 2: Clustering variables using SLINK with Pearson Correlation.
 * |- Demo 3: Clustering variables using SLINK with Correlation + Local Sensitivity Hashing (LSH).
 * |- Demo 4: Comparison of Linkage Methods (SLINK vs. NN-CHAIN variants vs. General Linkage).
 */
// ==================================================================================

// std includes
#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>
#include <utility>

// Eigen includes
#include <Eigen/Dense>

// ml_methods includes
#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_dispatcher.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/distance_policy.hpp>
#include <ml_methods/clustering/hierarchical/agglomerative/dendrogram_utils.hpp>

// utils includes
#include <utils/openmp/utils_openmp.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>

// ==================================================================================
// Namespace aliases
// ==================================================================================

namespace hac = trex::ml_methods::clustering::hierarchical::agglomerative;
namespace omp_utils = trex::utils::openmp;
namespace cdiagnostics = trex::utils::eval::cdiagnostics;

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


// ==================================================================================
// Part 1: Data Generation & Preprocessing & Evaluation Metrics
// ==================================================================================

// ---------------------------------------------------------------------------------

/**
 * @brief Computes the Adjusted Rand Index (ARI) to evaluate clustering accuracy.
 *
 * @return ARI score between -1.0 (worst) and 1.0 (perfect).
 */
double compute_ari(
    const std::vector<Eigen::Index>& true_labels,
    const std::vector<Eigen::Index>& pred_labels
    )
{
    if (true_labels.size() != pred_labels.size() || true_labels.empty()) {
        return 0.0;
    }
    Eigen::Index n = static_cast<Eigen::Index>(true_labels.size());

    Eigen::Index max_true = *std::max_element(true_labels.begin(), true_labels.end());
    Eigen::Index max_pred = *std::max_element(pred_labels.begin(), pred_labels.end());

    // Build contingency table
    Eigen::MatrixXd contingency = Eigen::MatrixXd::Zero(max_true + 1, max_pred + 1);
    for (Eigen::Index i = 0; i < n; ++i) {
        contingency(true_labels[i], pred_labels[i])++;
    }

    auto choose2 = [](double x) { return x * (x - 1) / 2.0; };

    double sum_comb_c = 0.0;
    for (Eigen::Index i = 0; i <= max_true; ++i) {
        for (Eigen::Index j = 0; j <= max_pred; ++j) {
            sum_comb_c += choose2(contingency(i, j));
        }
    }

    double sum_comb_a = 0.0;
    for (Eigen::Index i = 0; i <= max_true; ++i) {
        sum_comb_a += choose2(contingency.row(i).sum());
    }

    double sum_comb_b = 0.0;
    for (Eigen::Index i = 0; i <= max_pred; ++i) {
        sum_comb_b += choose2(contingency.col(i).sum());
    }

    double n_choose_2 = choose2(static_cast<double>(n));
    double expected_index = (sum_comb_a * sum_comb_b) / n_choose_2;
    double max_index = (sum_comb_a + sum_comb_b) / 2.0;

    if (max_index == expected_index) { return 1.0; }

    return (sum_comb_c - expected_index) / (max_index - expected_index);
}

// ---------------------------------------------------------------------------------

/** @brief Configuration parameters for generating hierarchical data. */
struct DataGenConfig {
    Eigen::Index n_samples;
    Eigen::Index p_features;
    Eigen::Index k_clusters;
};

/**
 * @brief Generates row-wise clusters with custom centers and distinct variances.
 *
 * @param config Configuration parameters for data generation.
 * @param centers A vector containing the center value for each cluster. Must match
 *  config.k_clusters.
 * @param std_devs A vector containing the standard deviation for each cluster. Must match
 *  config.k_clusters.
 *
 * @return A pair containing the generated matrix and the true labels for the SAMPLES (rows).
 */
template <typename MatrixType>
std::pair<MatrixType, std::vector<Eigen::Index>> generate_hierarchical_row_data(
    const DataGenConfig& config,
    const std::vector<double>& centers,
    const std::vector<double>& std_devs
    )
{
    if (config.n_samples % config.k_clusters != 0) {
        throw std::invalid_argument(
            "n_samples must be perfectly divisible by k_clusters.");
    }
    if (static_cast<Eigen::Index>(centers.size()) != config.k_clusters) {
        throw std::invalid_argument(
            "The size of 'centers' array must exactly match 'k_clusters'.");
    }
    if (static_cast<Eigen::Index>(std_devs.size()) != config.k_clusters) {
        throw std::invalid_argument(
            "The size of 'std_devs' array must exactly match 'k_clusters'.");
    }

    MatrixType X(config.n_samples, config.p_features);
    std::vector<Eigen::Index> true_labels(config.n_samples);
    Eigen::Index cluster_size = config.n_samples / config.k_clusters;

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        std::mt19937 local_rng(42 + thread_id);
        std::normal_distribution<double> standard_norm(0.0, 1.0);

        #pragma omp for schedule(static)
        for (Eigen::Index i = 0; i < config.n_samples; ++i) {

            // Identify the cluster
            Eigen::Index label = i / cluster_size;

            // Safety clamp
            if (label >= config.k_clusters) label = config.k_clusters - 1;

            // Fetch specific parameters for this cluster
            double mu = centers[label];
            double sigma = std_devs[label];

             // Assign true label
            true_labels[i] = label;

            for (Eigen::Index j = 0; j < config.p_features; ++j) {
                // Apply Location-Scale Transformation: X = mu + (sigma * Z)
                X(i, j) = mu + (sigma * standard_norm(local_rng));
            }
        }
    }
    return {X, true_labels};
}

// ---------------------------------------------------------------------------------

/**
 * @brief Generates a matrix with correlated variables in distinct blocks.
 * Uses a latent variable approach to generate massive dimensions in O(N * P) time.
 *
 * @details The data are modeled as a Latent Factor model with Collinearity within blocks.
 * The function generates a latent signal for each block of column
 * X_{i,j} = (Signal * Latent_{k}) + (Noise * Epsilon_{i,j}).
 * Example: all tech stocks move together based on a hidden market signal.
 *
 * @param config The structural dimensions (samples, features, clusters).
 * @param block_sizes Array containing the exact number of features per cluster.
 * @param centers Array containing the mean offset for each block.
 * @param std_devs Array containing the standard deviation multiplier for each block.
 *
 * @return A pair containing the generated data matrix and the true labels.
 */
std::pair<Eigen::MatrixXd, std::vector<Eigen::Index>> generate_correlated_variables(
    const DataGenConfig& config,
    const std::vector<Eigen::Index>& block_sizes,
    const std::vector<double>& centers,
    const std::vector<double>& std_devs
    )
{
    // Validation
    if (static_cast<Eigen::Index>(block_sizes.size()) != config.k_clusters ||
        static_cast<Eigen::Index>(centers.size()) != config.k_clusters ||
        static_cast<Eigen::Index>(std_devs.size())!= config.k_clusters)
    {
        throw std::invalid_argument(
            "Parameter arrays must match k_clusters in size."
        );
    }

    Eigen::Index total_features = 0;
    for (auto size: block_sizes) {
        total_features += size;
    }
    if (total_features != config.p_features) {
        throw std::invalid_argument(
            "Sum of block_sizes must exactly equal config.p_features."
        );
    }

    Eigen::MatrixXd X(config.n_samples, config.p_features);
    std::vector<Eigen::Index> true_labels(config.p_features);

    // Pre-Compute true labels based on custom block sizes O(P)
    Eigen::Index current_feature = 0;
    for (Eigen::Index k = 0; k < config.k_clusters; ++k) {
        for (Eigen::Index j = 0; j < block_sizes[k]; ++j) {
            true_labels[current_feature++] = k;
        }
    }

    // Generate the latent signals for each block O(N * K)
    Eigen::MatrixXd latent_signals(config.n_samples, config.k_clusters);
    std::mt19937 global_rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (Eigen::Index b = 0; b < config.k_clusters; ++b) {
        for (Eigen::Index i = 0; i < config.n_samples; ++i) {
            latent_signals(i, b) = dist(global_rng);
        }
    }

    double signal_strength = 0.85;  // within-group shared signal
    double noise_strength = 0.15;   // independent noise

    // Generate features with Location-Scale transformation
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        std::mt19937 local_rng(495 + thread_id);
        std::normal_distribution<double> local_dist(0.0, 1.0);

        #pragma omp for schedule(static)
        for (Eigen::Index j = 0; j < config.p_features; ++j) {

            // O(1) lookup for block assignments
            Eigen::Index block_idx = true_labels[j];
            double mu = centers[block_idx];
            double sigma = std_devs[block_idx];

            for (Eigen::Index i = 0; i < config.n_samples; ++i) {
                // Base correlated signal
                double base_val = (signal_strength * latent_signals(i, block_idx)) +
                                  (noise_strength * local_dist(local_rng));

                // Location-Scale transformation: X = mu + (sigma * base_val)
                X(i, j) = mu + (sigma * base_val);
            }
        }
    }

    return {X, true_labels};
}

// ---------------------------------------------------------------------------------

/**
 * @brief In-place mean-centering and L2-normalization for correlation clustering.
 *
 * @param mat The matrix to standardize.
 */
template <typename Derived>
void standardize_for_correlation(Eigen::MatrixBase<Derived>& mat) {
    #pragma omp parallel for schedule(static)
    for (Eigen::Index j = 0; j < mat.cols(); ++j) {
        double mean = mat.col(j).mean();
        mat.col(j).array() -= mean;

        double norm = mat.col(j).norm();
        if (norm > 1e-12) {
            mat.col(j) /= norm;
        } else {
            mat.col(j).setZero();
        }
    }
}

// ---------------------------------------------------------------------------------

/**
 * @brief Generates a matrix with block-diagonal AR(1) Toeplitz correlation structures.
 *
 * @param config The structural dimensions (samples, features, clusters/blocks).
 * @param block_sizes Array containing the exact number of features per block.
 * @param rhos Array containing the AR(1) correlation coefficient (rho) for each block.
 *
 * @return A pair containing the generated data matrix and the true block labels.
 */
std::pair<Eigen::MatrixXd, std::vector<Eigen::Index>> generate_ar1_block_variables(
    const DataGenConfig& config,
    const std::vector<Eigen::Index>& block_sizes,
    const std::vector<double>& rhos)
{
    // 1. Validation
    if (static_cast<Eigen::Index>(block_sizes.size()) != config.k_clusters ||
        static_cast<Eigen::Index>(rhos.size()) != config.k_clusters)
    {
        throw std::invalid_argument("Parameter arrays must match k_clusters in size.");
    }

    Eigen::Index total_features = 0;
    for (auto size : block_sizes) { total_features += size; }
    if (total_features != config.p_features) {
        throw std::invalid_argument("Sum of block_sizes must exactly equal config.p_features.");
    }

    Eigen::MatrixXd X(config.n_samples, config.p_features);
    std::vector<Eigen::Index> true_labels(config.p_features);

    // 2. Pre-compute true labels and block start indices
    Eigen::Index current_feature = 0;
    std::vector<Eigen::Index> block_starts(config.k_clusters);
    for (Eigen::Index k = 0; k < config.k_clusters; ++k) {
        block_starts[k] = current_feature;
        for (Eigen::Index j = 0; j < block_sizes[k]; ++j) {
            true_labels[current_feature++] = k;
        }
    }

    // 3. Generate independent standard normal noise Z in parallel
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        std::mt19937 local_rng(42 + thread_id);
        std::normal_distribution<double> local_dist(0.0, 1.0);

        // Fill Column-Major matrix optimally
        #pragma omp for schedule(static)
        for (Eigen::Index j = 0; j < config.p_features; ++j) {
            for (Eigen::Index i = 0; i < config.n_samples; ++i) {
                X(i, j) = local_dist(local_rng);
            }
        }
    }

    // 4. Apply the AR(1) dependency column-by-column using highly optimized Eigen vector math
    for (Eigen::Index k = 0; k < config.k_clusters; ++k) {
        double rho = rhos[k];
        if (rho <= -1.0 || rho >= 1.0) {
            throw std::invalid_argument("Rho must be strictly between -1.0 and 1.0.");
        }

        double scale = std::sqrt(1.0 - rho * rho);
        Eigen::Index start = block_starts[k];
        Eigen::Index size = block_sizes[k];

        // The first column of the block remains pure noise.
        // Subsequent columns are built iteratively based on the previous column.
        for (Eigen::Index j = 1; j < size; ++j) {
            X.col(start + j) = (rho * X.col(start + j - 1)) + (scale * X.col(start + j));
        }
    }

    return {X, true_labels};
}



// ==================================================================================
// Part 2: Demos
// ==================================================================================

/**
 * @brief Demo 1: Clusters samples according to Euclidean distances.
 * Uses RowMajor layout and on-the-fly transposition to maximize cache hits.
 */
void demo_slink_sample_clustering() {
    cdiagnostics::print_section_header("Demo 1: SLINK Sample Clustering (Euclidean)");

    // Data Setup
    const DataGenConfig config {
        .n_samples = 1'500,
        .p_features = 50'000,
        .k_clusters = 3
    };
    const std::vector<double> cluster_centers = {0.0, 3.0, 20.0};
    const std::vector<double> cluster_std_devs = {1, 1, 1};

    // Generate Data
    std::cout << "Generating Row-Major data (N=" << config.n_samples <<
                  ", P=" << config.p_features << ")...\n";

    auto [X, true_labels] =
        generate_hierarchical_row_data<RowMatrixXd>(config,
                                        cluster_centers,
                                        cluster_std_devs);

    // Setup Map
    Eigen::Map<RowMatrixXd> X_map(X.data(), config.n_samples, config.p_features);
    std::cout << "Creating transposed view for column-wise clustering...\n";
    auto X_T_view = X_map.transpose();
    using TransposedViewType = decltype(X_T_view);

    // Choose distance policy
    std::cout << "Set distance policy for sample clustering...\n";
    using DistancePol = hac::DistancePolicy<TransposedViewType, hac::DistanceMetric::Euclidean>;

    // Apply SLINK algorithm
    std::cout << "Building Single Linkage tree (SLINK)...\n";
    hac::SLINK<TransposedViewType, DistancePol> slink_model(X_T_view);

    // Measure timing
    double start_time = omp_get_wtime();
    slink_model.cluster();
    double end_time = omp_get_wtime();
    std::cout << "✓ Tree built successfully in "
              << std::fixed << (end_time - start_time) << " seconds.\n\n";

    auto merge_matrix = hac::DendrogramUtils::pointer_to_merge_matrix(
        slink_model.get_pi(), slink_model.get_lambda()
    );

    std::cout << "Cutting tree to form " << config.k_clusters << " clusters...\n";
    auto pred_labels = hac::DendrogramUtils::cut_tree(
        merge_matrix, X_T_view.cols(), config.k_clusters
    );

    // Compute Adjusted Rand Index using our native ground truth
    double ari_score = compute_ari(true_labels, pred_labels);

    std::cout << "Resulting Variable Clustering:\n";
    std::vector<Eigen::Index> cluster_counts(config.k_clusters, 0);
    for (auto label : pred_labels) { cluster_counts[label]++; }
    for (Eigen::Index i = 0; i < config.k_clusters; ++i) {
        std::cout << "  Predicted Cluster " << i << ": " << cluster_counts[i] << " variables\n";
    }

    std::cout << "\n--- CLUSTERING PERFORMANCE ---\n";
    std::cout << "Adjusted Rand Index (ARI): " << std::fixed << std::setprecision(4)
              << ari_score << " / 1.0000\n";
    std::cout << "\n[Demo 1 Complete]\n\n";
}


// -------------------------------------------------


/**
 * @brief Demo 2: Cluster variables using Pearson Correlation. Uses ColMajor layout and no
 *        transposition.
 */
void demo_slink_variable_clustering() {
    cdiagnostics::print_section_header("Demo 2: SLINK Variable Clustering (Correlation)");

    const DataGenConfig config {
        .n_samples = 10'000,
        .p_features = 50'000,
        .k_clusters = 5
    };
    std::vector<Eigen::Index> block_size = {20'000, 10'000, 8'000, 7'000,
        5'000};
    std::vector<double> centers = {0.0, 5.0, -5.0, 15.0, -15.0};
    std::vector<double> std_devs = {1.0, 1.2, 0.8, 2.0, 3.5};

    std::cout << "Generating Correlated Variables (N="
              << config.n_samples << ", P=" << config.p_features << ")...\n";
    auto [X, true_labels] =
        generate_correlated_variables(config, block_size, centers, std_devs);
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(),
                                    config.n_samples,
                                    config.p_features);

    std::cout << "Standardizing variables (Mean=0, L2=1) for correlation...\n";
    standardize_for_correlation(X_map);

    std::cout << "Set distance policy for variable clustering (Correlation)...\n";
    using DistancePol = hac::DistancePolicy<
        Eigen::Map<Eigen::MatrixXd>, hac::DistanceMetric::Correlation
    >;

    std::cout << "Building Single Linkage tree for Variables...\n";
    hac::SLINK<Eigen::Map<Eigen::MatrixXd>, DistancePol> slink_model(X_map);
    double start_time = omp_get_wtime();
    slink_model.cluster();
    double end_time = omp_get_wtime();

    std::cout << "✓ Tree built successfully in "
              << std::fixed << (end_time - start_time) << " seconds.\n\n";

    auto merge_matrix =
        hac::DendrogramUtils::pointer_to_merge_matrix(
            slink_model.get_pi(), slink_model.get_lambda()
        );

    std::cout << "Cutting tree to form " << config.k_clusters << " clusters...\n";
    auto pred_labels =
        hac::DendrogramUtils::cut_tree(
            merge_matrix, X_map.cols(), config.k_clusters
        );

    // Compute Adjusted Rand Index using our native ground truth
    double ari_score = compute_ari(true_labels, pred_labels);

    std::cout << "Resulting Variable Cluster Sizes:\n";
    std::vector<Eigen::Index> cluster_counts(config.k_clusters, 0);
    for (auto label : pred_labels) { cluster_counts[label]++; }
    for (Eigen::Index i = 0; i < config.k_clusters; ++i) {
        std::cout << "  Predicted Cluster " << i << ": " << cluster_counts[i] << " variables\n";
    }

    std::cout << "\n--- CLUSTERING PERFORMANCE ---\n";
    std::cout << "Adjusted Rand Index (ARI): " << std::fixed << std::setprecision(4)
              << ari_score << " / 1.0000\n";
    std::cout << "\n[Demo 2 Complete]\n\n";
}


// -------------------------------------------------

/** @brief Demo 3: SLINK Variable Clustering with Local Sensitivity Hashing (LSH) Comparison */
void demo_slink_variable_clustering_lsh() {
    cdiagnostics::print_section_header(
        "Demo 3: SLINK Variable Clustering (LSH Benchmark)");

    // Generate data
    const DataGenConfig config {
        .n_samples = 40'000,
        .p_features = 50'000,
        .k_clusters = 5
    };
    std::vector<Eigen::Index> block_sizes = {20'000, 10'000, 8'000, 7'000, 5'000};
    std::vector<double> centers = {0.0, 2.5, -2.5, 15.0, -15.0};
    std::vector<double> std_devs = {1.0, 1.2, 0.8, 2.0, 3.5};

    std::cout << "Generating Imbalanced Correlated Variables (N=" << config.n_samples
              << ", P=" << config.p_features << ")...\n";
    auto [X, true_labels] =
        generate_correlated_variables(config, block_sizes, centers, std_devs);

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), config.n_samples, config.p_features);

    std::cout << "Standardizing variables (Mean=0, L2=1) for correlation...\n";
    standardize_for_correlation(X_map);

    // =====================================================================
    // TEST A: Exact Pearson Correlation
    // =====================================================================
    {
        std::cout << "\n------------------------------------------------------------\n";
        std::cout << "[Test A] Exact Pearson Correlation (The Baseline)\n";
        using ExactDistPol = hac::DistancePolicy<
            Eigen::Map<Eigen::MatrixXd>, hac::DistanceMetric::Correlation>;

        hac::SLINK<Eigen::Map<Eigen::MatrixXd>, ExactDistPol> slink_model(X_map);

        double start_time = omp_get_wtime();
        slink_model.cluster();
        double end_time = omp_get_wtime();

        auto merge_matrix = hac::DendrogramUtils::pointer_to_merge_matrix(
            slink_model.get_pi(), slink_model.get_lambda());
        auto pred_labels = hac::DendrogramUtils::cut_tree(
            merge_matrix, X_map.cols(), config.k_clusters);
        double ari_score = compute_ari(true_labels, pred_labels);

        std::cout << "✓ Tree built successfully in " << std::fixed
                  << (end_time - start_time) << " seconds.\n";
        std::cout << "ARI: " << std::fixed << std::setprecision(4) << ari_score << " / 1.0000\n";
    }

    // =====================================================================
    // TEST B: LSH Filter (Gatekeeper -> Exact Dot Product)
    // =====================================================================
    {
        std::cout << "\n------------------------------------------------------------\n";
        std::cout << "[Test B] LSH Filter (Gatekeeper to Exact Dot Product)\n";
        using FilterDistPol = hac::DistancePolicy<
            Eigen::Map<Eigen::MatrixXd>, hac::DistanceMetric::Correlation_LSH_Filter>;

        hac::SLINK<Eigen::Map<Eigen::MatrixXd>, FilterDistPol> slink_model(X_map);

        double start_time = omp_get_wtime();
        slink_model.cluster();
        double end_time = omp_get_wtime();

        auto merge_matrix = hac::DendrogramUtils::pointer_to_merge_matrix(
            slink_model.get_pi(), slink_model.get_lambda());
        auto pred_labels = hac::DendrogramUtils::cut_tree(
            merge_matrix, X_map.cols(), config.k_clusters);
        double ari_score = compute_ari(true_labels, pred_labels);

        std::cout << "✓ Tree built successfully in " << std::fixed
                  << (end_time - start_time) << " seconds.\n";
        std::cout << "ARI: " << std::fixed << std::setprecision(4) << ari_score << " / 1.0000\n";
    }

    // =====================================================================
    // TEST C: LSH Approx (Pure O(1) SimHash Angular Estimation)
    // =====================================================================
    {
        std::cout << "\n------------------------------------------------------------\n";
        std::cout << "[Test C] LSH Approx (Pure O(1) SimHash Angular Estimation)\n";
        using ApproxDistPol = hac::DistancePolicy<
            Eigen::Map<Eigen::MatrixXd>, hac::DistanceMetric::Correlation_LSH_Approx>;

        hac::SLINK<Eigen::Map<Eigen::MatrixXd>, ApproxDistPol> slink_model(X_map);

        double start_time = omp_get_wtime();
        slink_model.cluster();
        double end_time = omp_get_wtime();

        auto merge_matrix = hac::DendrogramUtils::pointer_to_merge_matrix(
            slink_model.get_pi(), slink_model.get_lambda());
        auto pred_labels = hac::DendrogramUtils::cut_tree(
            merge_matrix, X_map.cols(), config.k_clusters);
        double ari_score = compute_ari(true_labels, pred_labels);

        std::cout << "✓ Tree built successfully in " << std::fixed
                  << (end_time - start_time) << " seconds.\n";

        std::cout << "\nResulting Variable Cluster Sizes (Approx):\n";
        std::vector<Eigen::Index> cluster_counts(config.k_clusters, 0);
        for (auto label : pred_labels) { cluster_counts[label]++; }
        for (Eigen::Index i = 0; i < config.k_clusters; ++i) {
            std::cout << "  Predicted Cluster " << i << ": " << cluster_counts[i] << " variables\n";
        }

        std::cout << "ARI: " << std::fixed << std::setprecision(4) << ari_score << " / 1.0000\n";
    }

    std::cout << "\n[Demo 3 Complete]\n\n";
}


// -------------------------------------------------

/** @brief Demo 04: Comparison of Linkage Methods: SLINK vs. CLINK vs NN-CHAIN based variants */
void demo_compare_linkage() {
    cdiagnostics::print_section_header("Demo 4: Linkage Comparison");

    const DataGenConfig config {
        .n_samples = 1'500,
        .p_features = 50'000,
        .k_clusters = 5
    };

    std::cout << "1. Generating Clean Correlated Variables (N=" << config.n_samples
              << ", P=" << config.p_features << ")...\n";
    std::vector<Eigen::Index> block_size = {20'000, 10'000, 8'000, 7'000,
         5'000};
    std::vector<double> centers = {0.0, 5.0, -5.0, 15.0, -15.0};
    std::vector<double> std_devs = {1.0, 1.2, 0.8, 2.0, 3.5};

    auto [X, true_labels] =
        generate_correlated_variables(config, block_size, centers, std_devs);
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(),
                                      config.n_samples, config.p_features);

    std::cout << "2. Standardizing variables (Mean=0, L2=1) for correlation...\n";
    standardize_for_correlation(X_map);

    std::cout << "3. Setting distance policy (Correlation)...\n";
    using DistancePol = hac::DistancePolicy<
        Eigen::Map<Eigen::MatrixXd>, hac::DistanceMetric::Correlation
    >;

    // ------------------------------------------------
    // Test A: Single Linkage (via Sibsons' SLINK)
    // ------------------------------------------------
    {
        std::cout << "\n[TEST A] SLINK:\n";
        hac::SLINK<Eigen::Map<Eigen::MatrixXd>, DistancePol> slink_model(X_map);

        double start_time = omp_get_wtime();
        slink_model.cluster();
        double end_time = omp_get_wtime();

        auto merge_slink = hac::DendrogramUtils::pointer_to_merge_matrix(
            slink_model.get_pi(), slink_model.get_lambda()
        );

        auto labels_slink = hac::DendrogramUtils::cut_tree(
            merge_slink, X_map.cols(), config.k_clusters
        );

        double ari_slink = compute_ari(true_labels, labels_slink);

        std::cout << "Time: " << std::fixed << (end_time - start_time) << " seconds\n";
        std::cout << "ARI:  " << std::fixed << std::setprecision(4) << ari_slink
                    << " / 1.0000\n";
    }


    // ---------------------------------------------------------------------------
    // Test B: Single Linkage (via Dispatcher -> SLINK)
    // ---------------------------------------------------------------------------
    {
    std::cout << "\n[Test B] Single Linkage (via Dispatcher):\n";
        double start = omp_get_wtime();

        auto merges = hac::AgglomerativeClustering::cluster<
            Eigen::Map<Eigen::MatrixXd>, DistancePol, hac::LinkageMethod::Single>(X_map);

        double end = omp_get_wtime();
        auto labels = hac::DendrogramUtils::cut_tree(merges,
            X_map.cols(), config.k_clusters);

        std::cout << "Time: " << std::fixed << (end - start) << " seconds\n";
        std::cout << "ARI:  " << std::fixed << std::setprecision(4) << compute_ari(true_labels,
            labels) << " / 1.0000\n";
    }


    // ---------------------------------------------------------------------------
    // Test C: Complete Linkage (via Dispatcher -> Block-Tiled Matrix -> NN-Chain)
    // ---------------------------------------------------------------------------
    {
        std::cout << "\n[Test C] Complete Linkage (via Dispatcher -> Disk/Matrix Bound):\n";
        double start = omp_get_wtime();

        auto merges = hac::AgglomerativeClustering::cluster<
            Eigen::Map<Eigen::MatrixXd>, DistancePol, hac::LinkageMethod::Complete>(X_map);

        double end = omp_get_wtime();
        auto labels = hac::DendrogramUtils::cut_tree(merges,
            X_map.cols(), config.k_clusters);

        std::cout << "Time: " << std::fixed << (end - start) << " seconds\n";
        std::cout << "ARI:  " << std::fixed << std::setprecision(4) << compute_ari(true_labels,
            labels) << " / 1.0000\n";
    }


    // ---------------------------------------------------------------------------
    // Test D: Average Linkage (via Dispatcher -> Geometric Centroids -> NN-Chain)
    // ---------------------------------------------------------------------------
    {
        std::cout << "\n[Test D] UPGMA (via Dispatcher -> RAM/Geometric Bound):\n";
        double start = omp_get_wtime();

        auto merges = hac::AgglomerativeClustering::cluster<
            Eigen::Map<Eigen::MatrixXd>, DistancePol, hac::LinkageMethod::Average>(X_map);

        double end = omp_get_wtime();
        auto labels = hac::DendrogramUtils::cut_tree(merges,
            X_map.cols(), config.k_clusters);

        std::cout << "Time: " << std::fixed << (end - start) << " seconds\n";
        std::cout << "ARI:  " << std::fixed << std::setprecision(4) << compute_ari(true_labels,
            labels) << " / 1.0000\n";
    }


    // --------------------------------------------------------------------------
    // Test E: WPGMA (via Dispatcher -> Block-Tiled Matrix -> NN-Chain)
    // --------------------------------------------------------------------------
    {
        std::cout << "\n[Test E] WPGMA (via Dispatcher -> Disk/Matrix Bound):\n";
        double start = omp_get_wtime();

        auto merges = hac::AgglomerativeClustering::cluster<
            Eigen::Map<Eigen::MatrixXd>, DistancePol, hac::LinkageMethod::WPGMA>(X_map);

        double end = omp_get_wtime();
        auto labels = hac::DendrogramUtils::cut_tree(merges,
            X_map.cols(), config.k_clusters);

        std::cout << "Time: " << std::fixed << (end - start) << " seconds\n";
        std::cout << "ARI:  " << std::fixed << std::setprecision(4) << compute_ari(true_labels,
            labels) << " / 1.0000\n";
    }


    // --------------------------------------------------------------
    // Test F: WARD (via Dispatcher -> Geometric Centroids -> NN-Chain)
    // --------------------------------------------------------------
    {
        std::cout << "\n[Test F] WARD (via Dispatcher -> RAM/Geometric Bound):\n";
        std::cout << " -> Note: Switching to Squared Euclidean Distance!\n";
        using EuclidDistancePol = hac::DistancePolicy<Eigen::Map<Eigen::MatrixXd>,
                                    hac::DistanceMetric::Euclidean>;

        double start = omp_get_wtime();

        auto merges = hac::AgglomerativeClustering::cluster<
            Eigen::Map<Eigen::MatrixXd>, EuclidDistancePol, hac::LinkageMethod::Ward>(X_map);

        double end = omp_get_wtime();
        auto labels = hac::DendrogramUtils::cut_tree(merges,
            X_map.cols(), config.k_clusters);

        std::cout << "Time: " << std::fixed << (end - start) << " seconds\n";
        std::cout << "ARI:  " << std::fixed << std::setprecision(4) << compute_ari(true_labels,
            labels) << " / 1.0000\n";
    }


    // -------------------------------------------------------
    // Test G: Centroid Linkage (via Dispatcher -> Generic Linkage)
    // -------------------------------------------------------
    {
        std::cout << "\n[Test G] Centroid Linkage (via Dispatcher -> Generic Linkage):\n";
        using EuclidDistancePol = hac::DistancePolicy<Eigen::Map<Eigen::MatrixXd>,
                                    hac::DistanceMetric::Euclidean>;

        double start = omp_get_wtime();

        auto merges = hac::AgglomerativeClustering::cluster<
            Eigen::Map<Eigen::MatrixXd>, EuclidDistancePol, hac::LinkageMethod::Centroid>(
                X_map);

        double end = omp_get_wtime();
        auto labels = hac::DendrogramUtils::cut_tree(merges,
            X_map.cols(), config.k_clusters);

        std::cout << "Time: " << std::fixed << (end - start) << " seconds\n";
        std::cout << "ARI:  " << std::fixed << std::setprecision(4) << compute_ari(true_labels,
            labels) << " / 1.0000\n";
    }

    // -------------------------------------------------------
    // Test H: Median Linkage (via Dispatcher -> Generic Linkage)
    // -------------------------------------------------------
    {
        std::cout << "\n[Test H] Median Linkage (via Dispatcher -> Generic Linkage):\n";
        using EuclidDistancePol = hac::DistancePolicy<Eigen::Map<Eigen::MatrixXd>,
                                    hac::DistanceMetric::Euclidean>;

        double start = omp_get_wtime();

        auto merges = hac::AgglomerativeClustering::cluster<
            Eigen::Map<Eigen::MatrixXd>, EuclidDistancePol, hac::LinkageMethod::Median>(
                X_map);

        double end = omp_get_wtime();
        auto labels = hac::DendrogramUtils::cut_tree(merges,
            X_map.cols(), config.k_clusters);

        std::cout << "Time: " << std::fixed << (end - start) << " seconds\n";
        std::cout << "ARI:  " << std::fixed << std::setprecision(4) << compute_ari(true_labels,
            labels) << " / 1.0000\n";
    }

    // -------------------------------------------------------

    std::cout << "\n[Demo 4 Complete]\n\n";
}


// -------------------------------------------------

/**
 * @brief Demo 5: Impact of LSH Approximation across different Linkage Methods
 *
 * @note This demo illustrates the geometry of latent factors. This demo generates "dense blocks"
 * of variables driven by hidden signals. Because the geometry forms distinct clusters,
 * the O(1) LSH SimHash approximation excells here, yielding massive acceleration with an ARI of
 * 1.0000 across SLINK, U/W-PGMA, and CLINK.
 */
void demo_lsh_linkage_comparison() {
    cdiagnostics::print_section_header("Demo 5: LSH Approx Linkage Comparison");

    // Generate imbalanced data
    const DataGenConfig config {
        .n_samples = 50'000,
        .p_features = 100'000,
        .k_clusters = 5
    };
    std::vector<Eigen::Index> block_sizes = {20'000, 15'000, 18'000, 17'000,
        30'000};
    std::vector<double> centers = {0.0, 1.0, -1.0, 5.0, -5.0};
    std::vector<double> std_devs = {1.5, 1.2, 0.8, 2.0, 3.5};

    std::cout << "1. Generating Imbalanced Correlated Variables (N=" << config.n_samples
              << ", P=" << config.p_features << ")...\n";

    auto [X, true_labels] = generate_correlated_variables(
        config, block_sizes, centers, std_devs);

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), config.n_samples,
                                        config.p_features);

    std::cout << "2. Standardizing variables (Mean=0, L2=1) for correlation...\n";
    standardize_for_correlation(X_map);

    // Set the global policy to the pure O(1) LSH Approximation
    std::cout << "3. Setting distance policy (Pure O(1) Correlation LSH Approx)...\n";
    using LshDistPol = hac::DistancePolicy<
        Eigen::Map<Eigen::MatrixXd>, hac::DistanceMetric::Correlation_LSH_Approx
    >;

    // ------------------------------------------------
    // Test A: Single Linkage (via SLINK)
    // ------------------------------------------------
    {
        std::cout << "\n[TEST A] Single Linkage (via SLINK with LSH):\n";
        hac::SLINK<Eigen::Map<Eigen::MatrixXd>, LshDistPol> slink_model(X_map);

        double start_time = omp_get_wtime();
        slink_model.cluster();
        double end_time = omp_get_wtime();

        auto merge_slink = hac::DendrogramUtils::pointer_to_merge_matrix(
            slink_model.get_pi(), slink_model.get_lambda());
        auto labels_slink = hac::DendrogramUtils::cut_tree(
            merge_slink, X_map.cols(), config.k_clusters);
        double ari_slink = compute_ari(true_labels, labels_slink);

        std::cout << "Time: " << std::fixed << (end_time - start_time) << " seconds\n";
        std::cout << "ARI:  " << std::fixed << std::setprecision(4) << ari_slink << " / 1.0000\n";
    }

    // ---------------------------------------------------------------------------
    // Test B: Complete Linkage (via NN-Chain with LSH)
    // ---------------------------------------------------------------------------
    {
        std::cout << "\n[Test B] Complete Linkage (via Dispatcher -> Disk/Matrix Bound):\n";
        double start = omp_get_wtime();

        auto merges = hac::AgglomerativeClustering::cluster<
            Eigen::Map<Eigen::MatrixXd>,
            LshDistPol /* or ExactDistPol */,
            hac::LinkageMethod::Complete /* or Average, etc */>(X_map);

        double end = omp_get_wtime();
        auto labels = hac::DendrogramUtils::cut_tree(merges,
            X_map.cols(), config.k_clusters);

        std::cout << "Time: " << std::fixed << (end - start) << " seconds\n";
        std::cout << "ARI:  " << std::fixed << std::setprecision(4) << compute_ari(true_labels,
            labels) << " / 1.0000\n";
    }

    // ---------------------------------------------------------------------------
    // Test C: Average Linkage (UPGMA via NN-Chain with LSH)
    // ---------------------------------------------------------------------------
    {
        std::cout << "\n[Test C] UPGMA (via Dispatcher -> RAM/Geometric Bound):\n";
        double start = omp_get_wtime();

        auto merges = hac::AgglomerativeClustering::cluster<
            Eigen::Map<Eigen::MatrixXd>,
            LshDistPol /* or ExactDistPol */,
            hac::LinkageMethod::Average /* or Average, etc */>(X_map);

        double end = omp_get_wtime();
        auto labels = hac::DendrogramUtils::cut_tree(merges,
            X_map.cols(), config.k_clusters);

        std::cout << "Time: " << std::fixed << (end - start) << " seconds\n";
        std::cout << "ARI:  " << std::fixed << std::setprecision(4) << compute_ari(true_labels,
            labels) << " / 1.0000\n";
    }

    // --------------------------------------------------------------------------
    // Test D: Weighted Average (WPGMA via NN-Chain with LSH)
    // --------------------------------------------------------------------------
    {
        std::cout << "\n[Test D] WPGMA (via Dispatcher -> Disk/Matrix Bound):\n";
        double start = omp_get_wtime();

        auto merges = hac::AgglomerativeClustering::cluster<
            Eigen::Map<Eigen::MatrixXd>,
            LshDistPol /* or ExactDistPol */,
            hac::LinkageMethod::WPGMA /* or Average, etc */>(X_map);

        double end = omp_get_wtime();
        auto labels = hac::DendrogramUtils::cut_tree(merges,
            X_map.cols(), config.k_clusters);

        std::cout << "Time: " << std::fixed << (end - start) << " seconds\n";
        std::cout << "ARI:  " << std::fixed << std::setprecision(4) << compute_ari(true_labels,
            labels) << " / 1.0000\n";
    }

    std::cout << "\n[Demo 5 Complete]\n\n";
}


// -------------------------------------------------

/** @brief Demo 6: AR(1) Toeplitz Correlation Clustering
 *
 * @note **The Geometry of AR(1) Chains:** This demo generates sequences where correlation
 * decays over time/space (e.g., variable 1 is close to 2, but far from 100).
 * - **Algorithmic Rule:** Single Linkage (SLINK) perfectly chains this together.
 *   Complete and Average linkage will catastrophically shatter the clusters.
 * - **LSH Rule:** Do not use LSH here. The 64-bit SimHash quantization destroys the
 * continuous correlation gradient, causing even SLINK to fail. Use Exact Correlation.
 */
void demo_ar1_linkage_comparison() {
    cdiagnostics::print_section_header("Demo 6: AR(1) Block Linkage Comparison");

    // Generate imbalanced AR(1) chaining data
    const DataGenConfig config {
        .n_samples = 3'000,
        .p_features = 50'000,
        .k_clusters = 5
    };

    std::vector<Eigen::Index> block_sizes = {20'000, 10'000, 8'000,
                                             7'000, 5'000};

    // rhos: The correlation between strictly adjacent variables in each block.
    // Notice how even at 0.95, the correlation decays to ~0 after just 100 steps.
    std::vector<double> rhos = {0.95, 0.90, 0.85, 0.80, 0.75};

    std::cout << "1. Generating AR(1) Toeplitz Variables (N=" << config.n_samples
              << ", P=" << config.p_features << ")...\n";

    auto [X, true_labels] = generate_ar1_block_variables(
        config, block_sizes, rhos);

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(),
                                      config.n_samples,
                                      config.p_features);

    std::cout << "2. Standardizing variables (Mean=0, L2=1) for correlation...\n";
    standardize_for_correlation(X_map);

    // Using Exact Correlation to observe pure algorithmic behavior
    std::cout << "3. Setting distance policy (Exact Pearson Correlation)...\n";
    using ExactDistPol = hac::DistancePolicy<
        Eigen::Map<Eigen::MatrixXd>, hac::DistanceMetric::Correlation
    >;

    // ------------------------------------------------
    // Test A: Single Linkage (via SLINK)
    // ------------------------------------------------
    {
        std::cout << "\n[TEST A] Single Linkage (via SLINK):\n";
        hac::SLINK<Eigen::Map<Eigen::MatrixXd>, ExactDistPol> slink_model(X_map);

        double start_time = omp_get_wtime();
        slink_model.cluster();
        double end_time = omp_get_wtime();

        auto merge_slink = hac::DendrogramUtils::pointer_to_merge_matrix(
            slink_model.get_pi(), slink_model.get_lambda());
        auto labels_slink = hac::DendrogramUtils::cut_tree(
            merge_slink, X_map.cols(), config.k_clusters);
        double ari_slink = compute_ari(true_labels, labels_slink);

        std::cout << "Time: " << std::fixed << (end_time - start_time) << " seconds\n";
        std::cout << "ARI:  " << std::fixed << std::setprecision(4) << ari_slink << " / 1.0000\n";
    }

    // ---------------------------------------------------------------------------
    // Test B: Complete Linkage (via NN-Chain)
    // ---------------------------------------------------------------------------
    {
        std::cout << "\n[Test B] Complete Linkage (via Dispatcher -> Disk/Matrix Bound):\n";
        double start = omp_get_wtime();

        auto merges = hac::AgglomerativeClustering::cluster<
            Eigen::Map<Eigen::MatrixXd>,
            ExactDistPol /* or ExactDistPol */,
            hac::LinkageMethod::Complete /* or Average, etc */>(X_map);

        double end = omp_get_wtime();
        auto labels = hac::DendrogramUtils::cut_tree(merges,
            X_map.cols(), config.k_clusters);

        std::cout << "Time: " << std::fixed << (end - start) << " seconds\n";
        std::cout << "ARI:  " << std::fixed << std::setprecision(4) << compute_ari(true_labels,
            labels) << " / 1.0000\n";
    }

    // ---------------------------------------------------------------------------
    // Test C: Average Linkage (UPGMA via NN-Chain)
    // ---------------------------------------------------------------------------
    {
        std::cout << "\n[Test C] UPGMA (via Dispatcher -> RAM/Geometric Bound):\n";
        double start = omp_get_wtime();

        auto merges = hac::AgglomerativeClustering::cluster<
            Eigen::Map<Eigen::MatrixXd>,
            ExactDistPol /* or ExactDistPol */,
            hac::LinkageMethod::Average /* or Average, etc */>(X_map);

        double end = omp_get_wtime();
        auto labels = hac::DendrogramUtils::cut_tree(merges,
            X_map.cols(), config.k_clusters);

        std::cout << "Time: " << std::fixed << (end - start) << " seconds\n";
        std::cout << "ARI:  " << std::fixed << std::setprecision(4) << compute_ari(true_labels,
            labels) << " / 1.0000\n";
    }

    std::cout << "\n[Demo 6 Complete]\n\n";
}



// ==================================================================================
// 3. Main
// ==================================================================================

void init_omp_environment(int threads = 6) {
    omp_set_num_threads(threads);
    omp_set_schedule(omp_sched_static, 0);
}

int main() {
    std::cout << "\n";
    cdiagnostics::print_section_header("Dendrograms Demo Suite");

    omp_utils::print_info();
    init_omp_environment(6);
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    try {
        if (true)
            // Dispatch Demo 1: Cluster the Samples
            demo_slink_sample_clustering();

        if (true)
            // Dispatch Demo 2: Cluster the Variables
            demo_slink_variable_clustering();

        if (true)
            // Dispatch Demo 3: Cluster the Variables with LSH
            demo_slink_variable_clustering_lsh();

        if (true)
            // Dispatch Demo 4: Compare Linkage Methods
            demo_compare_linkage();

        if (true)
            // Dispatch Demo 5: Compare Linkage Methods with LSH Approximation
            demo_lsh_linkage_comparison();

        if (true)
            // Dispatch Demo 6: AR(1) Toeplitz Correlation Clustering
            demo_ar1_linkage_comparison();

        cdiagnostics::print_section_header("All clustering demos completed successfully");

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n\n";
        return 1;
    }

    return 0;
}
