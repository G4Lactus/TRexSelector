// ==========================================================================
// trex_memmap_manager.hpp
// ==========================================================================
#ifndef TREX_SELECTOR_METHODS_UTILS_MEMMAP_MANAGER_HPP
#define TREX_SELECTOR_METHODS_UTILS_MEMMAP_MANAGER_HPP
// ==========================================================================
/**
 * @file trex_memmap_manager.hpp
 *
 * @brief RAII manager for memory-mapped matrices in the T-Rex Selector.
 * Engineered to integrate with TempDirManager and MemoryMappedMatrix.
 */
// ==========================================================================

// std includes
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <filesystem>

// trex_selector_methods includes
#include <trex_selector_methods/utils/trex_temp_dir_manager.hpp>

// utils includes
#include <utils/memmap/memory_mapped_matrix.hpp>

// =========================================================================

namespace trex::trex_selector_methods::utils::memmap_manager {

namespace mm = trex::utils::memmap;
namespace txfs = trex::trex_selector_methods::utils::filesystem;

// =========================================================================

class MemmapManager {
private:
    // --- Configuration State ---
    /** @brief Number of rows in the matrix. */
    std::size_t n_{0};
    /** @brief Maximum number of dummy columns. */
    std::size_t max_dummies_{0};
    /** @brief Number of experiments (K). */
    std::size_t K_{0};
    /** @brief Flag indicating whether the memory-mapped files are shared across experiments. */
    bool shared_{false};
    /** @brief Verbosity flag for debug printing. */
    [[maybe_unused]] bool verbose_{false};

    // --- Execution State ---
    /** @brief Path to the temporary directory for memory-mapped files. */
    std::string temp_dir_{};
    /** @brief Flag indicating whether the manager has been initialized. */
    bool initialized_{false};

    // Use unique_ptr so we can dynamically initialize the files after directory creation
    std::vector<std::unique_ptr<mm::MemoryMappedMatrix<double>>> D_memmaps_;

public:
    /** @brief Default constructor */
    MemmapManager() = default;

    /**
     * @brief Parameterized constructor expected by TRexSelector.
     *
     * @param n Number of rows in the matrix.
     * @param max_dummies Maximum number of dummy columns.
     * @param K Number of experiments (K).
     * @param shared Whether to use a single shared file (true) or one per experiment (false).
     * @param verbose Verbosity flag for debug printing.
     */
    MemmapManager(std::size_t n, std::size_t max_dummies, std::size_t K, bool shared, bool verbose)
        : n_(n), max_dummies_(max_dummies), K_(K), shared_(shared), verbose_(verbose)
    {
        // Pre-allocate the vector capacity
        std::size_t num_files = shared_ ? 1 : K_;
        D_memmaps_.reserve(num_files);
    }

    /** @brief RAII destructor — delegates to non-throwing cleanup. */
    ~MemmapManager() noexcept {
        internalCleanup(false);
    }

    // =========================================================================
    // Core API
    // =========================================================================

    /** @brief Check if the manager has been initialized */
    bool isInitialized() const { return initialized_; }

    /** @brief Get the number of currently mapped files */
    std::size_t numFiles() const { return D_memmaps_.size(); }

    /** @brief Returns the path to the current temporary directory. */
    const std::string& getTempDir() const { return temp_dir_; }

    /**
     * @brief Creates the temp directory and instantiates the MemoryMappedMatrix objects.
     */
    void initialize() {
        if (initialized_) return;

        // 1. Create a safe temporary directory
        temp_dir_ = txfs::TempDirManager::create("trex_memmap_");

        // 2. Initialize memory-mapped files on disk
        std::size_t num_files = shared_ ? 1 : K_;
        namespace fs = std::filesystem;

        for (std::size_t i = 0; i < num_files; ++i) {
            fs::path file_path = fs::path(temp_dir_) / ("D_" + std::to_string(i) + ".bin");

            // Instantiate the matrix in ReadWrite mode so it can be populated
            D_memmaps_.push_back(std::make_unique<mm::MemoryMappedMatrix<double>>(
                file_path.string(), n_, max_dummies_,
                mm::AccessMode::ReadWrite
            ));
        }

        initialized_ = true;
    }

    /**
     * @brief Fetches an Eigen::Map view for experiment k, subsetted to num_dummies columns.
     *
     * @details Returns a true Eigen::Map (not a Block expression) over the first
     *          num_dummies columns.  This is valid because the underlying storage is
     *          column-major, so those columns are contiguous in memory.  Callers can
     *          pass the result by non-const lvalue reference to dispatchForExperiment
     *          and writeDummiesToMap.
     *
     * @param k           Index of the experiment (0-based; maps to file 0 if shared).
     * @param num_dummies Number of dummy columns to include in the map (must be <= max_dummies_).
     *
     * @return Eigen::Map over the first num_dummies columns of the k-th memory-mapped file.
     *
     * @throws std::runtime_error if idx is out of range or the matrix is uninitialized.
     */
    Eigen::Map<Eigen::MatrixXd> getDMap(std::size_t k, std::size_t num_dummies) {
        std::size_t idx = shared_ ? 0 : k;

        if (idx >= D_memmaps_.size() || !D_memmaps_[idx]) {
            throw std::runtime_error("MemmapManager: Invalid map index or uninitialized matrix.");
        }

        return Eigen::Map<Eigen::MatrixXd>(
            D_memmaps_[idx]->getMap().data(),
            static_cast<Eigen::Index>(n_),
            static_cast<Eigen::Index>(num_dummies)
        );
    }

    /**
     * @brief Release all memory-mapped files and remove the temporary directory.
     *
     * @throws std::runtime_error on filesystem errors.
     */
    void cleanup() {
        internalCleanup(true);
    }

private:
    /**
     * @brief Unified cleanup logic. Unmaps memory before deleting the directory.
     *
     * @param throw_on_error If true, exceptions during cleanup will be propagated.
     */
    void internalCleanup(bool throw_on_error) {
        // 1. Release file handles (unmaps memory via MemoryMappedMatrix destructors)
        D_memmaps_.clear();

        // 2. Delegate folder deletion to the shared TempDirManager
        if (initialized_ && !temp_dir_.empty()) {
            txfs::TempDirManager::remove(temp_dir_, throw_on_error);
        }

        temp_dir_.clear();
        initialized_ = false;
    }
// =========================================================================
};

// =========================================================================
} /* End of namespace trex::trex_selector_methods::utils::memmap_manager */
// =========================================================================
#endif /* End of TREX_SELECTOR_METHODS_UTILS_MEMMAP_MANAGER_HPP */
