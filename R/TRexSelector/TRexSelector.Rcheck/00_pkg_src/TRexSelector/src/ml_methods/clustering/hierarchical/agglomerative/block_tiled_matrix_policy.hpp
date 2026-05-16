// ===================================================================================
// block_tiled_matrix_policy.hpp
// ===================================================================================
#ifndef ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_BLOCK_TILED_POLICY_HPP
#define ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_BLOCK_TILED_POLICY_HPP
// ===================================================================================
/**
 * @file block_tiled_matrix_policy.hpp
 *
 * @brief Block-tiled distance-matrix policy for cache-efficient hierarchical clustering.
 *
 * @details Stores the full condensed distance matrix in a block-tiled layout to improve
 * CPU cache utilization during the Lance-Williams update step. Supports both
 * in-memory (RAM) and memory-mapped (mmap) backends for large datasets.
 *
 */
 // ===================================================================================

#include <utils/logging/logger.hpp>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <random>
#include <memory>

#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include <Eigen/Dense>

#include <ml_methods/clustering/hierarchical/agglomerative/agglomerative_types.hpp>

namespace trex::ml_methods::clustering::hierarchical::agglomerative {

namespace detail {
    struct LanceWilliamsCoefficients {
        double alpha_c = 0.0;
        double alpha_d = 0.0;
        double beta = 0.0;
        double gamma = 0.0;
    };

    template <LinkageMethod SelectedMethod>
    inline LanceWilliamsCoefficients compute_LW_coeffs(double n_c, double n_d, double n_L) {
        LanceWilliamsCoefficients coeffs;
        if constexpr (SelectedMethod == LinkageMethod::Average) {
            double total_n_cd = n_c + n_d;
            coeffs.alpha_c = n_c / total_n_cd;
            coeffs.alpha_d = n_d / total_n_cd;

        } else if constexpr (SelectedMethod == LinkageMethod::Complete) {
            coeffs.alpha_c = 0.5;
            coeffs.alpha_d = 0.5;
            coeffs.gamma = 0.5;

        } else if constexpr (SelectedMethod == LinkageMethod::WPGMA) {
            coeffs.alpha_c = 0.5;
            coeffs.alpha_d = 0.5;

        } else if constexpr (SelectedMethod == LinkageMethod::Centroid) {
            double total_n_cd = n_c + n_d;
            coeffs.alpha_c = n_c / total_n_cd;
            coeffs.alpha_d = n_d / total_n_cd;
            coeffs.beta = -(n_c * n_d) / (total_n_cd * total_n_cd);

        } else if constexpr (SelectedMethod == LinkageMethod::Median) {
            coeffs.alpha_c = 0.5;
            coeffs.alpha_d = 0.5;
            coeffs.beta = -0.25;

        } else if constexpr (SelectedMethod == LinkageMethod::Ward) {
            double total_n = n_c + n_d + n_L;
            coeffs.alpha_c = (n_c + n_L) / total_n;
            coeffs.alpha_d = (n_d + n_L) / total_n;
            coeffs.beta = -n_L / total_n;

        } else {
            static_assert(sizeof(SelectedMethod) == 0, "Unsupported Linkage Method for Matrix Policy.");
        }
        return coeffs;
    }
}

/**
 * @brief Block-tiled distance-matrix policy for cache-efficient hierarchical agglomerative clustering.
 *
 * @details Stores the full pairwise distance matrix in a cache-friendly block-tiled layout
 * (tile size 256 × 256) and applies Lance-Williams updates in-place. Supports both
 * heap-allocated RAM and memory-mapped (Boost.Interprocess) backends.
 *
 * @tparam MatrixType        Type of the input data matrix (e.g., Eigen::MatrixXf or Eigen::Map).
 * @tparam DistancePolicyType Distance metric policy type; must provide `get_distance(i, j)`.
 * @tparam SelectedMethod    Compile-time linkage method (determines Lance-Williams coefficients).
 */
template <typename MatrixType, typename DistancePolicyType, LinkageMethod SelectedMethod>
class BlockTiledMatrixPolicy {
public:
    using Scalar = typename MatrixType::Scalar;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

private:
    static constexpr Eigen::Index BLOCK_SIZE = 256;

    // --- Hybrid Backend Storage ---
    bool use_mmap_;
    std::vector<RealScalar> ram_backend_; // Used if use_mmap_ == false

    std::filesystem::path temp_filepath_;
    std::unique_ptr<boost::interprocess::file_mapping> file_mapping_;
    std::unique_ptr<boost::interprocess::mapped_region> mapped_region_;

    // The Universal Pointer (points to either RAM or MMAP)
    RealScalar* data_ptr_ = nullptr;
    // ------------------------------

    Eigen::Index num_original_;
    Eigen::Index blocks_per_row_;

    std::vector<Eigen::Index> id_to_idx_;
    std::vector<Eigen::Index> idx_to_id_;

    DistancePolicyType dist_policy_;

    inline Eigen::Index get_tiled_index(Eigen::Index r, Eigen::Index c) const {
        const Eigen::Index block_r = r / BLOCK_SIZE;
        const Eigen::Index block_c = c / BLOCK_SIZE;
        const Eigen::Index in_block_r = r % BLOCK_SIZE;
        const Eigen::Index in_block_c = c % BLOCK_SIZE;

        const Eigen::Index block_id = block_r * blocks_per_row_ + block_c;
        const Eigen::Index offset_in_block = in_block_r * BLOCK_SIZE + in_block_c;

        return (block_id * BLOCK_SIZE * BLOCK_SIZE) + offset_in_block;
    }

    inline double get_physical_distance(Eigen::Index r, Eigen::Index c) const {
        return data_ptr_[get_tiled_index(r, c)];
    }

    inline void set_physical_distance(Eigen::Index r, Eigen::Index c, double value) {
        data_ptr_[get_tiled_index(r, c)] = value;
        data_ptr_[get_tiled_index(c, r)] = value;
    }

public:
    /**
     * @brief Constructor evaluating the optional memory map flag.
     */
    BlockTiledMatrixPolicy(const MatrixType& data, bool use_mmap = false)
        : use_mmap_(use_mmap), dist_policy_(data)
    {
        namespace fs = std::filesystem;
        namespace bip = boost::interprocess;

        num_original_ = data.cols();
        blocks_per_row_ = static_cast<Eigen::Index>(std::ceil(static_cast<double>(num_original_) / BLOCK_SIZE));
        Eigen::Index total_elements = blocks_per_row_ * blocks_per_row_ * BLOCK_SIZE * BLOCK_SIZE;

        if (use_mmap_) {
            std::uintmax_t file_size_bytes = static_cast<std::uintmax_t>(total_elements * sizeof(RealScalar));

            std::random_device rd;
            std::mt19937_64 gen(rd());
            std::uniform_int_distribution<uint64_t> dis;
            temp_filepath_ = fs::temp_directory_path() / ("trex_hac_matrix_" + std::to_string(dis(gen)) + ".bin");

            {
                std::ofstream ofs(temp_filepath_, std::ios::out | std::ios::binary);
                if (!ofs.is_open()) throw std::runtime_error("Failed to create mmap file.");
            }

            std::error_code ec;
            fs::resize_file(temp_filepath_, file_size_bytes, ec);
            if (ec) {
                fs::remove(temp_filepath_, ec);
                throw std::runtime_error("Failed to allocate disk space.");
            }

            try {
                file_mapping_ = std::make_unique<bip::file_mapping>(temp_filepath_.string().c_str(), bip::read_write);
                mapped_region_ = std::make_unique<bip::mapped_region>(*file_mapping_, bip::read_write);
                data_ptr_ = static_cast<RealScalar*>(mapped_region_->get_address());
            } catch (const bip::interprocess_exception& e) {
                fs::remove(temp_filepath_, ec);
                throw std::runtime_error(std::string("Boost mmap failed: ") + e.what());
            }
            TREX_INFO( "  -> [Backend] Mmap allocated " << (file_size_bytes / (1ULL << 30)) << " GB on NVMe.\n");
        }
        else {
            // Use physical RAM
            ram_backend_.assign(total_elements, 0.0);
            data_ptr_ = ram_backend_.data();
            TREX_INFO( "  -> [Backend] RAM allocated " << (total_elements * sizeof(RealScalar) / (1ULL << 20)) << " MB.\n");
        }

        id_to_idx_.assign(num_original_ * 2 - 1, -1);
        idx_to_id_.assign(num_original_, -1);

        #pragma omp parallel for schedule(dynamic)
        for (Eigen::Index i = 0; i < num_original_; ++i) {
            id_to_idx_[i] = i;
            idx_to_id_[i] = i;
            set_physical_distance(i, i, 0.0);
            for (Eigen::Index j = 0; j < i; ++j) {
                set_physical_distance(i, j, dist_policy_.get_distance(i, j));
            }
        }
    }

    /** @brief Destructor. Unmaps and removes the temporary mmap file if applicable. */
    ~BlockTiledMatrixPolicy() {
        if (use_mmap_) {
            mapped_region_.reset();
            file_mapping_.reset();
            std::error_code ec;
            if (std::filesystem::exists(temp_filepath_, ec)) {
                std::filesystem::remove(temp_filepath_, ec);
            }
        }
    }

    // --- Public policy interface ---

    /**
     * @brief Compute the logical distance between two clusters.
     *
     * @param c_id Logical cluster ID of the first cluster.
     * @param d_id Logical cluster ID of the second cluster.
     * @return Stored pairwise distance; returns `numeric_limits::max()` for dead clusters.
     */
    double get_logical_distance(Eigen::Index c_id, Eigen::Index d_id) const {
        const Eigen::Index idx_c = id_to_idx_[c_id];
        const Eigen::Index idx_d = id_to_idx_[d_id];
        if (idx_c == -1 || idx_d == -1) return std::numeric_limits<double>::max();
        return get_physical_distance(idx_c, idx_d);
    }

    /**
     * @brief Find the nearest active neighbor of a given cluster.
     *
     * @param c_id              Logical cluster ID to search from.
     * @param active_clusters   Boolean flags indicating which cluster IDs are still active.
     * @param current_cluster_id Upper bound (exclusive) of cluster IDs to consider.
     * @param min_dist          Output: distance to the nearest neighbor found.
     * @return Logical cluster ID of the nearest neighbor, or -1 if none found.
     */
    Eigen::Index find_nearest_neighbor(Eigen::Index c_id, const std::vector<bool>& active_clusters, Eigen::Index current_cluster_id, double& min_dist) const {
        min_dist = std::numeric_limits<double>::max();
        Eigen::Index best_d = -1;
        const Eigen::Index idx_c = id_to_idx_[c_id];
        for (Eigen::Index i = 0; i < current_cluster_id; ++i) {
            if (!active_clusters[i] || i == c_id) continue;
            const double d = get_physical_distance(idx_c, id_to_idx_[i]);
            if (d < min_dist) { min_dist = d; best_d = i; }
        }
        return best_d;
    }

    /**
     * @brief Merge two clusters and update all pairwise distances via the Lance-Williams formula.
     *
     * @param args          Named-argument struct specifying c_id, d_id, and new_id.
     * @param cluster_sizes Current cluster sizes indexed by logical cluster ID.
     */
    void merge_clusters(const MergeClustersArgs& args, const std::vector<Eigen::Index>& cluster_sizes) {
        const Eigen::Index idx_c = id_to_idx_[args.c_id];
        const Eigen::Index idx_d = id_to_idx_[args.d_id];
        const double n_c = static_cast<double>(cluster_sizes[args.c_id]);
        const double n_d = static_cast<double>(cluster_sizes[args.d_id]);
        const double dist_cd = get_physical_distance(idx_c, idx_d);

        std::vector<Eigen::Index> active_k;
        std::vector<double> dist_cL, dist_dL;
        active_k.reserve(num_original_);

        for (Eigen::Index k = 0; k < num_original_; ++k) {
            if (k == idx_c || k == idx_d || idx_to_id_[k] == -1) continue;
            active_k.push_back(k);
            dist_cL.push_back(get_physical_distance(idx_c, k));
            dist_dL.push_back(get_physical_distance(idx_d, k));
        }

        id_to_idx_[args.new_id] = idx_c;
        idx_to_id_[idx_c] = args.new_id;
        idx_to_id_[idx_d] = -1;

        for (std::size_t i = 0; i < active_k.size(); ++i) {
            const Eigen::Index k = active_k[i];
            const Eigen::Index L = idx_to_id_[k];
            const double n_L = static_cast<double>(cluster_sizes[L]);
            const auto coeffs = detail::compute_LW_coeffs<SelectedMethod>(n_c, n_d, n_L);

            double new_dist = coeffs.alpha_c * dist_cL[i] + coeffs.alpha_d * dist_dL[i] +
                              coeffs.beta * dist_cd + coeffs.gamma * std::abs(dist_cL[i] - dist_dL[i]);

            if (new_dist < 0.0) new_dist = 0.0;
            set_physical_distance(idx_c, k, new_dist);
        }
    }
};

}
#endif /* ML_METHODS_CLUSTERING_HC_AGGLOMERATIVE_BLOCK_TILED_POLICY_HPP */
