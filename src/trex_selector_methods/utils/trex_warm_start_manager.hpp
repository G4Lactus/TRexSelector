// ===================================================================================
// trex_warm_start_manager.hpp
// ===================================================================================
#ifndef TREX_SELECTOR_METHODS_UTILS_WARM_START_MANAGER_HPP
#define TREX_SELECTOR_METHODS_UTILS_WARM_START_MANAGER_HPP
// ===================================================================================
/**
 * @file trex_warm_start_manager.hpp
 *
 * @brief RAII manager for solver warm-start state across T-loop iterations.
 *
 * @details
 * Two operating modi:
 *
 * Modus 1: IN-MEMORY  (for in-memory X and D)
 * - Solvers are retained as unique_ptr<TSolver_Base> in a vector.
 * - The L-loop invalidates retained solvers (dummies changed).
 * - On the first T-loop iteration each solver is constructed fresh.
 * - On subsequent iterations the retained solver is re-executed with a higher T_stop
 *   an in-memory warm start.
 * - Optionally, the user can request solver serialization (a snapshot) of the
 *   in-memory solvers to disk, and later restore them.
 *
 *   This is useful for RAM savings or checkpointing long runs.
 *
 * Modus 2: SERIALIZED  (for memory-mapped data, or when RAM is scarce)
 * - Solvers are never held in memory between iterations.
 * - After each execution the solver is serialized to a binary file.
 * - On the next T-loop iteration the solver is deserialized, reconnected
 *   to data, executed, and saved again.
 */
// ===================================================================================

// std includes
#include <string>
#include <vector>
#include <memory>

// TSolver base inclusion
#include <tsolvers/tsolver_base.hpp>

// Shared filesystem utility
#include <trex_selector_methods/utils/trex_temp_dir_manager.hpp>

// ===================================================================================

namespace trex::trex_selector_methods::utils::warm_start_manager {

namespace txfs = trex::trex_selector_methods::utils::filesystem;

// ===================================================================================

enum class WarmStartMode { IN_MEMORY, SERIALIZED };

class WarmStartManager {
private:
    /** @brief Current warm start mode. */
    WarmStartMode mode_{WarmStartMode::IN_MEMORY};

    /** @brief Verbosity flag for debug printing. */
    [[maybe_unused]] bool verbose_{false};

    // In-Memory state
    // ---------------------
    /** @brief Vector of in-memory solvers. */
    std::vector<std::unique_ptr<tsolvers::TSolver_Base>> solvers_{};

    // Serialized state
    // ---------------------
    /** @brief Vector of file paths for serialized solvers. */
    std::vector<std::string> solver_files_{};

    /** @brief Path to the temporary directory for serialized solvers. */
    std::string temp_dir_{};

    /** @brief Flag indicating whether the manager has been initialized. */
    bool initialized_{false};

public:
    /** @brief Default constructor. */
    WarmStartManager() = default;

    /**
     * @brief Main constructor used by TRexSelector.
     *
     * @details Pre-allocates vectors to size K to ensure thread-safe
     *          concurrent access across OpenMP threads.
     *
     * @param K       Number of random experiments (determines vector pre-allocation size).
     * @param mode    Warm-start mode (IN_MEMORY or SERIALIZED).
     * @param verbose Enable verbose debug output.
     */
    WarmStartManager(std::size_t K, WarmStartMode mode, bool verbose)
        : mode_(mode), verbose_(verbose) {

        // CRITICAL: Pre-allocate vectors so OpenMP threads can safely
        // access indices 0 through K-1 without resizing the vector concurrently.
        solvers_.resize(K);
        solver_files_.resize(K);
    }

    /**
     * @brief RAII destructor; delegates to non-throwing cleanup.
     */
    ~WarmStartManager() noexcept {
        internalCleanup(false);
    }

    // =========================================================================
    // Core API
    // =========================================================================

    /**
     * @brief Creates the temporary directory required for serialized warm starts.
     */
    void initializeTempDirectory() {
        if (initialized_) return;

        // Example: Require 5GB of free space for solver checkpoints. Adjust as needed.
        constexpr std::uintmax_t MIN_SPACE_REQUIRED = 5ULL * 1024 * 1024 * 1024;

        temp_dir_ = txfs::TempDirManager::create(
            "trex_warmstart_", MIN_SPACE_REQUIRED
        );

        initialized_ = true;
    }

    /**
     * @brief Release all in-memory solvers and serialized files, and remove the
     * temporary directory.
     * @throws std::runtime_error on filesystem errors.
     */
    void cleanup() {
        internalCleanup(true);
    }

    /**
     * @brief Invalidates all current solvers.
     * @details Called when dummy matrices change (e.g., new L-loop iteration).
     * Discards in-memory states and resets the serialized file tracking.
     */
    void invalidate() {
        // 1. Clear in-memory solvers
        solvers_.clear();

        // 2. Clear tracked files
        // (Note: The files will be overwritten or cleaned up by the temp directory manager,
        //  but we must clear the vector so hasSolver() returns false for the new iteration)
        solver_files_.clear();
    }

    // =========================================================================
    // Getters and Setters
    // =========================================================================

    /** @brief Returns the current warm start mode. */
    WarmStartMode mode() const { return mode_; }

    /**
     * @brief Sets the warm start mode. Note: Changing modes does not automatically
     * clear existing state; the user should call cleanup() if needed.
     */
    void setMode(WarmStartMode m) { mode_ = m; }

    /** @brief Checks if a solver exists at the given index. */
    bool hasSolver(std::size_t k) const {
        if (mode_ == WarmStartMode::IN_MEMORY) {
            return k < solvers_.size() && solvers_[k] != nullptr;
        } else {
            return k < solver_files_.size() && !solver_files_[k].empty();
        }
    }

    /**
     * @brief Returns the file path of the solver at the given index.
     *
     * @param k Index of the solver.
     *
     * @return File path if it exists, empty string otherwise.
     */
    std::string solverFile(std::size_t k) const {
        if (k < solver_files_.size()) return solver_files_[k];
        return "";
    }

private:
    /**
     * @brief Unified cleanup logic handling both throwing and non-throwing contexts.
     *
     * @note This function purposely does NOT have a noexcept specifier so it can
     * conditionally throw, but is safe to call from the noexcept destructor
     * when throw_on_error is false.
     */
    void internalCleanup(bool throw_on_error) {
        // 1. Release in-memory solvers
        solvers_.clear();

        // 2. Delegate folder deletion to the shared TempDirManager
        if (initialized_ && !temp_dir_.empty()) {
            txfs::TempDirManager::remove(temp_dir_, throw_on_error);
        }

        temp_dir_.clear();
        solver_files_.clear();
        initialized_ = false;
    }

// =========================================================================
}; /* End of class WarmStartManager */

// ===================================================================================
} /* End of namespace trex::trex_selector_methods::utils::warm_start_manager */
// ===================================================================================
#endif /* End of TREX_SELECTOR_METHODS_UTILS_WARM_START_MANAGER_HPP */
