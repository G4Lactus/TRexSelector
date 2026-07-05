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
#include <utility>
#include <vector>
#include <memory>

// Eigen includes
#include <Eigen/Dense>

// TSolver base inclusion
#include <tsolvers/tsolver_base.hpp>

// Shared filesystem utility
#include <trex_selector_methods/trex_utils/trex_temp_dir_manager.hpp>

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

    /** @brief Number of experiment slots (fixed at construction). */
    std::size_t K_{0};

    // In-Memory state
    // ---------------------
    /** @brief Vector of in-memory solvers. */
    std::vector<std::unique_ptr<tsolvers::TSolver_Base>> solvers_{};

    /** @brief Per-slot dummy matrices co-retained with the solvers.
     *
     *  Solvers hold non-owning Eigen::Map views into their dummy matrix D_k.
     *  For strategies whose D_k is a per-call temporary (PERMUTATION, DIRECT),
     *  the buffer must be kept alive alongside the retained solver; moving the
     *  MatrixXd here transfers the heap buffer without changing its address,
     *  so the solver's view stays valid. Slots stay empty (0 x 0) for
     *  strategies whose dummies persist elsewhere (STANDARD / memory-mapped).
     */
    std::vector<Eigen::MatrixXd> retained_D_{};

    // Serialized state
    // ---------------------
    /** @brief Vector of file paths for serialized solvers (registered after
     *  the first successful save; empty = no saved state yet). */
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
        : mode_(mode), verbose_(verbose), K_(K) {

        // CRITICAL: Pre-allocate vectors so OpenMP threads can safely
        // access indices 0 through K-1 without resizing the vector concurrently.
        solvers_.resize(K);
        retained_D_.resize(K);
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
     * Discards in-memory states (and their co-retained dummy buffers) and
     * resets the serialized file tracking. The K slots are preserved so that
     * subsequent storeSolver() / registerSolverFile() calls can repopulate
     * them (a plain clear() would leave hasSolver() permanently false).
     */
    void invalidate() {
        // 1. Drop in-memory solvers (and the dummy buffers they view),
        //    keeping the slot count.
        solvers_.clear();
        solvers_.resize(K_);
        retained_D_.clear();
        retained_D_.resize(K_);

        // 2. Deregister tracked files.
        // (The files themselves are overwritten on the next save or removed by
        //  the temp directory manager; deregistering makes hasSolver() false
        //  for the new iteration.)
        solver_files_.assign(K_, std::string{});
    }

    // =========================================================================
    // In-memory solver retention
    // =========================================================================

    /**
     * @brief Retain a solver for experiment k (in-memory warm start).
     *
     * @details Thread-safe for distinct k (pre-allocated slots, no resize).
     *          The solver holds non-owning views into X and its dummy matrix
     *          D_k; the caller must guarantee those buffers outlive the
     *          retention (co-retain per-call temporaries via retainDummies()).
     */
    void storeSolver(std::size_t k, std::unique_ptr<tsolvers::TSolver_Base> solver) {
        if (k < solvers_.size()) {
            solvers_[k] = std::move(solver);
        }
    }

    /**
     * @brief Access the retained solver for experiment k (nullptr if none).
     */
    tsolvers::TSolver_Base* getSolver(std::size_t k) {
        return (k < solvers_.size()) ? solvers_[k].get() : nullptr;
    }

    /**
     * @brief Keep the dummy matrix of experiment k alive alongside its solver.
     *
     * @details For strategies whose D_k is a per-call temporary (PERMUTATION,
     *          DIRECT). Moving the MatrixXd transfers its heap buffer without
     *          changing the address, so the solver's non-owning view into D_k
     *          remains valid.
     */
    void retainDummies(std::size_t k, Eigen::MatrixXd&& D_k) {
        if (k < retained_D_.size()) {
            retained_D_[k] = std::move(D_k);
        }
    }

    // =========================================================================
    // Serialized solver registration
    // =========================================================================

    /**
     * @brief Deterministic checkpoint path for experiment k.
     *
     * @return "<temp_dir>/tsolver_state_<k>.bin", or "" if the temporary
     *         directory has not been initialized yet.
     */
    std::string plannedSolverFile(std::size_t k) const {
        if (!initialized_ || temp_dir_.empty()) return "";
        return temp_dir_ + "/tsolver_state_" + std::to_string(k) + ".bin";
    }

    /**
     * @brief Mark experiment k's checkpoint as saved, so hasSolver(k) reports
     *        a resumable state on the next iteration.
     */
    void registerSolverFile(std::size_t k) {
        if (k < solver_files_.size()) {
            solver_files_[k] = plannedSolverFile(k);
        }
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
        // 1. Release in-memory solvers and their co-retained dummy buffers
        //    (solvers first — they hold views into the buffers).
        solvers_.clear();
        retained_D_.clear();

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
