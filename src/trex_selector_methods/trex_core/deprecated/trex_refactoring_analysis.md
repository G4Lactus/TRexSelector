# TRexSelector Refactoring Analysis

**Target class:** `trex::trex_methods::trex_core::TRexSelector`  
**Codebase size:** Header ~1107 lines · Implementation ~2651 lines · Total ~3800 lines  
**Document audience:** C++ developer familiar with the T-Rex project but new to TRexSelector internals  
**Refactoring philosophy:** Preserve all existing behavior, the public API, and the polymorphic inheritance design. No algorithmic changes.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Diagnosis](#2-diagnosis)
3. [Proposed Module Decomposition](#3-proposed-module-decomposition)
4. [Refactoring Roadmap](#4-refactoring-roadmap)
5. [Impact on Derived Classes](#5-impact-on-derived-classes)
6. [Proposed Directory Structure](#6-proposed-directory-structure)

---

## 1. Executive Summary

`TRexSelector` is the correct, battle-tested implementation of the T-Rex selector for FDR-controlled variable selection. The algorithm works. The inheritance chain — DA-TRex, Screen-TRex, DP-TRex, GVS-TRex, Sparse-PCA-TRex and others — is a sound design that allows variant-specific behaviour to override `select()` and its sub-procedures via virtual dispatch.

The problem is structural, not algorithmic. At ~3800 lines the class has accumulated at least **nine distinct responsibilities** that are tangled together through 30+ protected data members. The practical consequences are:

- A developer fixing a warm-start bug must first understand dummy generation, memory-map lifecycle, and the seven-way solver switch to locate the relevant code.
- The `runSingleExperimentWithSolver` method (lines 2085–2225) contains 140 lines of code where only the type name changes across seven near-identical blocks. Any change to the warm-start protocol requires seven parallel edits.
- The six `runRandomExperiments*` methods share ~80 % of their body; a logic fix in aggregation must be applied to all six independently.
- Unit-testing any single concern (e.g., normalization, solver dispatch) is impossible without constructing a full `TRexSelector` object.

The refactoring proposed in this document extracts six focused modules (A–F) without altering any virtual method signatures or any algorithmic logic. After the refactoring, `TRexSelector` becomes a thin orchestrator that owns instances of the extracted helpers and delegates to them.

---

## 2. Diagnosis

Each problem is assigned a severity:

| ID | Severity | Problem |
|----|----------|---------|
| P1 | **Critical** | Solver dispatch duplication — 7× copy-paste in `runSingleExperimentWithSolver` |
| P2 | **High** | Six experiment runner methods with ~80 % shared boilerplate |
| P3 | **High** | In-memory vs. memory-mapped branching duplicated across all strategies |
| P4 | **Medium** | Dummy generation / permutation logic embedded in the selector |
| P5 | **Medium** | Normalization logic embedded in the selector |
| P6 | **Medium** | Warm-start / temp-directory management embedded in the selector |
| P7 | **Low** | 30+ protected members creating cognitive overload |
| P8 | **Low** | L-loop strategies share boilerplate while-loop body |

---

### P1 — Critical: Solver Dispatch Duplication

**Location:** `runSingleExperimentWithSolver`, lines 2085–2225 (approximately 140 lines)

The method contains a `switch (solver_type)` with seven `case` arms — one each for `TLARS`, `TLASSO`, `TSTEPWISE`, `TENET`, `TOMP`, `TGP`, and `TACGP`. Every arm executes the same four-step protocol:

```
1. if (use_warm_start && has_serialized_solvers_) { load solver from file } 
   else { construct fresh solver from XD, y }
2. solver.executeStep(T_stop)
3. beta_path = solver.getBetaPath()
4. if (use_warm_start) { serialize solver to solver_files_[k] }
```

The only structural difference is the concrete type in step 1 and 4. `TENET` additionally passes `lambda2` to its constructor. Because the arms differ only in type, not in control flow, this is a textbook use-case for a template function. The duplication means:

- Any change to the warm-start protocol (e.g., adding a version field to serialized files) must be made identically in seven places.
- Any new solver type (`TNEWSOLVER`) requires copy-pasting an entire arm rather than adding one line.
- Dead code is easy to introduce: a fix applied to six arms but missed in the seventh silently diverges.

**Impact on maintainability:** Every warm-start, serialization, or solver-lifecycle change carries a ×7 multiplier on the required effort and a near-certain risk of inconsistency.

---

### P2 — High: Experiment Runner Boilerplate (6 methods × 80 % shared code)

**Location:** `runRandomExperiments`, `runRandomExperiments_Permutation`, `runRandomExperiments_Direct` (lines ~1680–2084) and their MemMap counterparts `runRandomExperimentsMemMap`, `runRandomExperimentsMemMap_Permutation`, `runRandomExperimentsMemMap_Direct` (lines ~2226–2651).

All six methods share the following structure:

```
for k = 0..K-1:
    1. Determine solver_file_path (identical block in all 6)
    2. Warm-start setup boilerplate (lines ~1720-1724, ~1896-1902, ~1996-2001,
                                          ~2405-2410, ~2557-2561)
    3. [DIFFERS] Generate dummies (standard / permuted / direct variants)
    4. [DIFFERS] Assemble augmented matrix (in-memory XD  vs. memory-mapped XD)
    5. Call runSingleExperimentWithSolver(k, XD, ...)
    6. Aggregate result into voting_grid_ (identical in all 6)
    7. Update solver_files_ tracking (identical in all 6)
```

The solver-file-path determination block (step 1) and the result aggregation (step 6) are copy-pasted verbatim. The warm-start setup (step 2) is copied at five separate callsites (see exact line ranges above).

**Impact on maintainability:** Any change to aggregation logic or warm-start setup requires locating and editing six method bodies. The methods are long enough (~100–200 lines each) that the shared sections are not visually obvious.

---

### P3 — High: In-Memory vs. Memory-Mapped Branching

Every L-loop strategy and every experiment runner contains an explicit branch:

```cpp
if (!trex_ctrl_.use_memory_mapping) {
    // assemble XD as Eigen::MatrixXd
} else {
    // write to memory-mapped files, pass file handles
}
```

This branching is structural, not algorithmic. It splits every strategy method into two parallel code paths that evolve independently. A bug fix in one branch is often not applied to the other, and the asymmetry is invisible from a diff of the algorithm logic.

**Precise locations:** Present in all five L-loop strategy methods (`runLLoopCalibration_SKIP`, `_Standard`, `_HCONCAT`, `_Permutation`, `_Direct`) and in all six experiment runner methods.

**Impact on maintainability:** New matrix access patterns (e.g., a GPU-resident variant) would require eleven more branch insertions. The branching obscures the algorithm, which is identical in both paths.

---

### P4 — Medium: Dummy Generation / Permutation Logic Embedded in TRexSelector

**Members involved:** `base_dummies_perm_`, `base_seed_perm_`, `base_dummies_initialized_`  
**Methods involved:** `generateDummies`, `generateDummiesForExperiment_Direct`, `applyRowPermutation`, `getPermutedDummiesForExperiment`

The class owns the full lifecycle of dummy generation including permutation-state tracking (`base_dummies_initialized_` flag, seed management). This logic is self-contained but is spread across the header and implementation alongside unrelated solver and normalization code.

**Impact on maintainability:** Independently unit-testing a new dummy generation strategy requires constructing a full selector object. The `base_dummies_initialized_` guard is set in one method and checked in another with no obvious connection in the source layout.

---

### P5 — Medium: Normalization Logic Embedded in TRexSelector

**Members involved:** `X_means_`, `X_l2norms_`, `y_mean_`  
**Methods involved:** `centerAndL2NormalizeX`, `centerAndL2NormalizeMatrix`, `denormalizeX`, `centerY`, `decenterY`

These five methods form a self-contained normalization pipeline. They have no dependency on solver state, dummy state, or memory-map state. They use only the three normalization parameter members. There is no architectural reason for them to be members of the selector rather than free functions in a utilities namespace.

**Impact on maintainability:** Low on its own. The cost is that `TRexSelector` carries three data members (`X_means_`, `X_l2norms_`, `y_mean_`) and five method declarations whose purpose is orthogonal to selection.

---

### P6 — Medium: Warm-Start / Temp-Directory Management Embedded in TRexSelector

**Members involved:** `temp_dir_`, `solver_files_`, `has_serialized_solvers_`  
**Methods involved:** `createTempDirectory`, `cleanupTempDirectory`

The class owns a temporary directory used to persist solver state between experiments. The directory is created in `select()`, paths are populated in the experiment runners, and cleanup is in `select()`. This lifecycle management is scattered across the file and is interleaved with algorithm code.

**Impact on maintainability:** RAII guarantees are absent — if an exception escapes from an experiment runner, `cleanupTempDirectory` in `select()` may not execute. Extracting this into a dedicated RAII class eliminates the risk and concentrates the filesystem logic.

---

### P7 — Low: 30+ Protected Data Members

The protected section of the class header spans at minimum the following conceptual groups:

| Group | Members (examples) |
|-------|--------------------|
| Configuration / control | `trex_ctrl_`, `n_`, `p_`, `K_`, `T_stop_` |
| Normalization state | `X_means_`, `X_l2norms_`, `y_mean_` |
| Result containers | `voting_grid_`, `selected_`, `FDP_hat_`, `Phi_prime_` |
| L-loop state | `dummy_multiplier_`, `num_dummies_`, current L-loop strategy |
| Warm-start / serialization | `temp_dir_`, `solver_files_`, `has_serialized_solvers_` |
| Memory-mapping | `XD_memmaps_`, `max_cols_memmap_` |
| Dummy generation state | `base_dummies_perm_`, `base_seed_perm_`, `base_dummies_initialized_` |

A reader arriving at any single method must mentally track all 30+ members to understand preconditions and postconditions. The groups are not visually delimited in the header.

**Impact on maintainability:** Primarily cognitive — any serious change requires a full mental model of all members before any code can be written safely.

---

### P8 — Low: L-Loop Strategy Boilerplate While-Loop Body

**Location:** Inside `runLLoopCalibration_Standard`, `_HCONCAT`, `_Permutation`, `_Direct`

Each strategy's while loop shares the same tail:

```cpp
FDP_hat_ = computeFDPHat(...);
Phi_prime_ = computePhiPrime(...);
if (trex_ctrl_.verbose) { /* identical log output */ }
dummy_multiplier_++;
```

This tail is duplicated four times. A change to the convergence condition or verbose output format must be made in four places.

---

## 3. Proposed Module Decomposition

The six modules below are ordered from lowest to highest coupling with TRexSelector internals. Each section specifies exact methods and members to migrate, the interaction pattern with the remaining selector, and a representative interface sketch.

---

### Module A: `DataNormalizer` — Free Functions in `trex::utils`

**New files:** `trex_data_normalizer.h` (header only — all free functions)

**Purpose:** Own the normalization parameter types and the five normalization/denormalization operations. No class state; all parameters are passed explicitly.

**What moves out of TRexSelector:**

| Currently in TRexSelector | Destination |
|---------------------------|-------------|
| `X_means_` (member) | `NormalizationParams::X_means` |
| `X_l2norms_` (member) | `NormalizationParams::X_l2norms` |
| `y_mean_` (member) | `NormalizationParams::y_mean` |
| `centerAndL2NormalizeX(X)` | `trex::utils::centerAndL2NormalizeX(X, params)` |
| `centerAndL2NormalizeMatrix(M)` | `trex::utils::centerAndL2NormalizeMatrix(M, params)` |
| `denormalizeX(X)` | `trex::utils::denormalizeX(X, params)` |
| `centerY(y)` | `trex::utils::centerY(y, params)` |
| `decenterY(y)` | `trex::utils::decenterY(y, params)` |

**How TRexSelector interacts with it:** `TRexSelector` holds a single `trex::utils::NormalizationParams norm_params_` member and calls the free functions, passing `norm_params_` by reference.

**Interface sketch:**

```cpp
// trex_data_normalizer.h
namespace trex::utils {

struct NormalizationParams {
    Eigen::VectorXd X_means;
    Eigen::VectorXd X_l2norms;
    double          y_mean = 0.0;
};

/// Centers columns of X and L2-normalizes them.
/// Fills params.X_means and params.X_l2norms as a side-effect.
void centerAndL2NormalizeX(Eigen::MatrixXd& X, NormalizationParams& params);

/// Applies the already-computed column params to a new matrix (e.g., a dummy block).
void centerAndL2NormalizeMatrix(Eigen::MatrixXd& M,
                                const NormalizationParams& params);

/// Reverses column normalization.
void denormalizeX(Eigen::MatrixXd& X, const NormalizationParams& params);

/// Centers y; stores y_mean in params.
void centerY(Eigen::VectorXd& y, NormalizationParams& params);

/// Re-adds y_mean.
void decenterY(Eigen::VectorXd& y, const NormalizationParams& params);

} // namespace trex::utils
```

**Why free functions rather than a class:** Normalization has no mutable state beyond the parameter struct. Free functions are more easily unit-tested, more composable, and carry no virtual-dispatch overhead.

---

### Module B: `DummyGenerator` — Class in `trex::trex_methods::trex_core`

**New files:** `trex_dummy_generator.h`, `trex_dummy_generator.cpp`

**Purpose:** Own the lifecycle and permutation state of base dummies. Encapsulate the `base_dummies_initialized_` guard and the seed management.

**What moves out of TRexSelector:**

| Currently in TRexSelector | Destination |
|---------------------------|-------------|
| `base_dummies_perm_` | `DummyGenerator::base_dummies_perm_` |
| `base_seed_perm_` | `DummyGenerator::base_seed_perm_` |
| `base_dummies_initialized_` | `DummyGenerator::initialized_` |
| `generateDummies(n, p, num_dummies, seed)` | `DummyGenerator::generate(n, p, num_dummies, seed)` |
| `generateDummiesForExperiment_Direct(k, ...)` | `DummyGenerator::generateForExperiment(k, ...)` |
| `applyRowPermutation(D, perm)` | `DummyGenerator::applyRowPermutation(D, perm)` |
| `getPermutedDummiesForExperiment(k, ...)` | `DummyGenerator::getPermuted(k, ...)` |

**How TRexSelector interacts with it:** Composition — `TRexSelector` owns a `DummyGenerator dummy_gen_` member. Derived classes that need to override dummy generation can override a `virtual DummyGenerator& getDummyGenerator()` accessor, or the generator can be injected by a derived class constructor.

**Interface sketch:**

```cpp
// trex_dummy_generator.h
namespace trex::trex_methods::trex_core {

class DummyGenerator {
public:
    explicit DummyGenerator(std::size_t n, std::size_t p);

    /// Standard iid dummy generation.
    Eigen::MatrixXd generate(std::size_t num_dummies, unsigned seed) const;

    /// Direct strategy: generates dummies for a single experiment k.
    Eigen::MatrixXd generateForExperiment(std::size_t k,
                                           std::size_t num_dummies,
                                           unsigned seed) const;

    /// Returns base_dummies with rows permuted according to the k-th permutation.
    /// Caches the base permutation on first call (lazy initialization).
    Eigen::MatrixXd getPermuted(std::size_t k,
                                 const Eigen::MatrixXd& base_dummies,
                                 unsigned seed);

    void reset(); ///< Clears cached permutation state.

private:
    std::size_t n_, p_;
    Eigen::PermutationMatrix<Eigen::Dynamic> base_dummies_perm_;
    unsigned                                 base_seed_perm_ = 0;
    bool                                     initialized_    = false;

    static void applyRowPermutation(Eigen::MatrixXd& D,
                                    const Eigen::PermutationMatrix<Eigen::Dynamic>& perm);
};

} // namespace trex::trex_methods::trex_core
```

---

### Module C: `SolverDispatch` — Template Helper

**New files:** `trex_solver_dispatch.h` (header only — template)

**Purpose:** Replace the 140-line switch/case in `runSingleExperimentWithSolver` (lines 2085–2225) with a single generic function that handles the warm-start / execute / serialize protocol for any solver type.

**What moves out of TRexSelector:** The body of every `case` arm in `runSingleExperimentWithSolver`. The dispatcher shim (a free function that maps `SolverTypeForTRex` to the template instantiation) also lives here.

**Why this is the highest-priority fix:** Adding a new solver (`TNEWSOLVER`) currently requires copy-pasting ~18 lines. After this change, it requires one new line in the dispatcher and one template instantiation.

**Interface sketch:**

```cpp
// trex_solver_dispatch.h
namespace trex::trex_methods::trex_core {

/// Generic warm-start / execute / serialize protocol.
/// SolverT must provide:
///   - SolverT(const Eigen::MatrixXd& XD, const Eigen::VectorXd& y, ...)
///   - SolverT::deserialize(path)  -> SolverT
///   - void executeStep(std::size_t T_stop)
///   - Eigen::MatrixXd getBetaPath() const
///   - void serialize(const std::string& path) const
template <typename SolverT, typename... CtorArgs>
Eigen::MatrixXd runSolverExperiment(
    const Eigen::MatrixXd&  XD,
    const Eigen::VectorXd&  y,
    std::size_t             T_stop,
    const std::string&      solver_file,
    bool                    use_warm_start,
    bool                    has_serialized,
    CtorArgs&&...           ctor_args)
{
    SolverT solver = (use_warm_start && has_serialized)
        ? SolverT::deserialize(solver_file)
        : SolverT(XD, y, std::forward<CtorArgs>(ctor_args)...);

    solver.executeStep(T_stop);
    Eigen::MatrixXd beta_path = solver.getBetaPath();

    if (use_warm_start) {
        solver.serialize(solver_file);
    }
    return beta_path;
}

/// Thin dispatcher: maps a SolverTypeForTRex enum value to the appropriate
/// template instantiation. This is the only location that names concrete types.
///
/// Returns the beta_path matrix for experiment k.
Eigen::MatrixXd dispatchSolverExperiment(
    SolverTypeForTRex       solver_type,
    const Eigen::MatrixXd&  XD,
    const Eigen::VectorXd&  y,
    std::size_t             T_stop,
    const std::string&      solver_file,
    bool                    use_warm_start,
    bool                    has_serialized,
    double                  lambda2 = 0.0);   // TENET only; ignored for others

} // namespace trex::trex_methods::trex_core
```

**Usage in TRexSelector (after refactoring):**

```cpp
// Before refactoring: 140-line switch/case
// After refactoring:
beta_path = dispatchSolverExperiment(
    trex_ctrl_.solver_type, XD, y_, T_stop_,
    warm_start_mgr_.getSolverFile(k),
    trex_ctrl_.use_warm_start,
    warm_start_mgr_.hasSerialized(),
    trex_ctrl_.lambda2);
```

---

### Module D: `WarmStartManager` — RAII Class

**New files:** `trex_warm_start_manager.h`, `trex_warm_start_manager.cpp`

**Purpose:** Encapsulate the temporary directory lifecycle and solver-file tracking with RAII semantics. Eliminates the risk of directory leaks on exception.

**What moves out of TRexSelector:**

| Currently in TRexSelector | Destination |
|---------------------------|-------------|
| `temp_dir_` | `WarmStartManager::temp_dir_` |
| `solver_files_` | `WarmStartManager::solver_files_` |
| `has_serialized_solvers_` | `WarmStartManager::has_serialized_` |
| `createTempDirectory()` | `WarmStartManager::WarmStartManager(K)` (constructor) |
| `cleanupTempDirectory()` | `WarmStartManager::~WarmStartManager()` (destructor) |

**Interface sketch:**

```cpp
// trex_warm_start_manager.h
namespace trex::trex_methods::trex_core {

class WarmStartManager {
public:
    /// Creates a unique temp directory and pre-populates solver_files_ with K paths.
    /// Throws std::runtime_error if directory creation fails.
    explicit WarmStartManager(std::size_t K);

    /// Removes the temp directory and all serialized solver files.
    ~WarmStartManager() noexcept;

    // Non-copyable, movable
    WarmStartManager(const WarmStartManager&)            = delete;
    WarmStartManager& operator=(const WarmStartManager&) = delete;
    WarmStartManager(WarmStartManager&&)                 noexcept = default;

    /// Returns the filesystem path for experiment k's solver file.
    const std::string& getSolverFile(std::size_t k) const;

    /// Records that experiment k has a serialized solver available.
    void markSerialized(std::size_t k);

    /// True if experiment k has been serialized at least once.
    bool hasSerialized(std::size_t k) const;

private:
    std::string              temp_dir_;
    std::vector<std::string> solver_files_;
    std::vector<bool>        has_serialized_;
};

} // namespace trex::trex_methods::trex_core
```

**TRexSelector interaction:** Owns `std::optional<WarmStartManager> warm_start_mgr_`. Constructed in `select()` only when `trex_ctrl_.use_warm_start == true`. The optional is reset at the end of `select()` (or destroyed on exception), triggering RAII cleanup automatically — eliminating the explicit `cleanupTempDirectory()` call.

---

### Module E: `MemoryMapManager` — RAII Class

**New files:** `trex_memmap_manager.h`, `trex_memmap_manager.cpp`

**Purpose:** Centralize memory-mapped matrix lifecycle. Provide a uniform `getAugmentedMatrix(k, num_dummies)` interface so that call sites in the experiment runners and L-loop strategies do not need to branch on `use_memory_mapping` themselves.

**What moves out of TRexSelector:**

| Currently in TRexSelector | Destination |
|---------------------------|-------------|
| `XD_memmaps_` | `MemoryMapManager::memmaps_` |
| `max_cols_memmap_` | `MemoryMapManager::max_cols_` |
| `initializeMemoryMappedMatrices(...)` | `MemoryMapManager::initialize(n, p, K, max_multiplier, strategy)` |
| `writeXToMemoryMappedMatrices(X)` | `MemoryMapManager::writeX(X)` |
| `cleanupMemoryMappedMatrices()` | `MemoryMapManager::~MemoryMapManager()` |

**Interface sketch:**

```cpp
// trex_memmap_manager.h
namespace trex::trex_methods::trex_core {

class MemoryMapManager {
public:
    MemoryMapManager(std::size_t n, std::size_t p, std::size_t K,
                     std::size_t max_dummy_multiplier,
                     LLoopStrategy strategy);

    ~MemoryMapManager() noexcept; // Unmaps and removes backing files

    MemoryMapManager(const MemoryMapManager&)            = delete;
    MemoryMapManager& operator=(const MemoryMapManager&) = delete;

    /// Write the predictor matrix into the memory-mapped backing store.
    void writeX(const Eigen::MatrixXd& X);

    /// Returns a reference to the memory-mapped region for experiment k,
    /// sized to accommodate num_dummies dummy columns.
    /// The caller writes dummy columns into the returned view.
    Eigen::Map<Eigen::MatrixXd> getActiveMap(std::size_t k,
                                              std::size_t num_dummies);

private:
    std::size_t                                n_, p_, K_, max_cols_;
    std::vector</* platform mmap handle */>    memmaps_;
};

} // namespace trex::trex_methods::trex_core
```

**Critical design note:** The `getAugmentedMatrix` / `getActiveMap` interface is the key to resolving P3. After this extraction, the experiment runners and L-loop strategies obtain their `[X|D]` view through the same call site regardless of whether memory mapping is active. The `if (!trex_ctrl_.use_memory_mapping)` branches in TRexSelector are replaced with a unified accessor:

```cpp
// Before refactoring (in every experiment runner):
if (!trex_ctrl_.use_memory_mapping) {
    Eigen::MatrixXd XD(n_, p_ + num_dummies);
    XD << X_, D;
    beta_path = dispatchSolverExperiment(..., XD, ...);
} else {
    auto XD_map = XD_memmaps_[k].getActiveMap(num_dummies);
    XD_map.rightCols(num_dummies) = D;
    beta_path = dispatchSolverExperiment(..., XD_map, ...);
}

// After refactoring (unified):
auto XD = matrix_store_.getAugmentedMatrix(k, D); // returns Eigen::Ref<>
beta_path = dispatchSolverExperiment(..., XD, ...);
```

`getAugmentedMatrix` accepts the dummy block and returns an `Eigen::Ref<const Eigen::MatrixXd>` — backed either by a freshly allocated heap matrix (in-memory mode) or by a memory-mapped view (mmap mode). The caller is agnostic.

---

### Module F: `ExperimentRunner` — Class

**New files:** `trex_experiment_runner.h`, `trex_experiment_runner.cpp`

**Purpose:** This is the highest-impact extraction. All six `runRandomExperiments*` methods move here, reducing TRexSelector to a single `runner_.run(...)` call. The runner owns the coordination logic: iterate experiments, generate dummies, assemble XD, dispatch solver, aggregate results.

**What moves out of TRexSelector:**

| Currently in TRexSelector | Destination |
|---------------------------|-------------|
| `runRandomExperiments` | `ExperimentRunner::run(Standard)` |
| `runRandomExperiments_Permutation` | `ExperimentRunner::run(Permutation)` |
| `runRandomExperiments_Direct` | `ExperimentRunner::run(Direct)` |
| `runRandomExperimentsMemMap` | unified with above via `MemoryMapManager` |
| `runRandomExperimentsMemMap_Permutation` | unified with above |
| `runRandomExperimentsMemMap_Direct` | unified with above |
| `runSingleExperimentWithSolver` | replaced by `dispatchSolverExperiment` |

**Dependencies:** ExperimentRunner holds references to (or is constructed with):
- `DummyGenerator&`
- `WarmStartManager*` (nullable when warm-start is disabled)
- `MemoryMapManager*` (nullable when memory mapping is disabled)

**Interface sketch:**

```cpp
// trex_experiment_runner.h
namespace trex::trex_methods::trex_core {

enum class ExperimentStrategy { Standard, Permutation, Direct };

struct ExperimentRunnerConfig {
    std::size_t        K;
    std::size_t        T_stop;
    std::size_t        num_dummies;
    ExperimentStrategy strategy;
    SolverTypeForTRex  solver_type;
    bool               use_warm_start;
    double             lambda2;       // TENET only
};

class ExperimentRunner {
public:
    ExperimentRunner(DummyGenerator&     dummy_gen,
                     WarmStartManager*   warm_start_mgr,   // nullable
                     MemoryMapManager*   memmap_mgr,       // nullable
                     const Eigen::MatrixXd& X,
                     const Eigen::VectorXd& y);

    /// Runs all K experiments and returns the accumulated voting matrix.
    /// Each column of the returned matrix is the beta_path for one experiment.
    Eigen::MatrixXd run(const ExperimentRunnerConfig& cfg);

private:
    Eigen::MatrixXd runSingleExperiment(std::size_t              k,
                                         std::size_t              num_dummies,
                                         ExperimentStrategy       strategy,
                                         const ExperimentRunnerConfig& cfg);

    DummyGenerator&        dummy_gen_;
    WarmStartManager*      warm_start_mgr_;
    MemoryMapManager*      memmap_mgr_;
    const Eigen::MatrixXd& X_;
    const Eigen::VectorXd& y_;
};

} // namespace trex::trex_methods::trex_core
```

**TRexSelector after extraction:**

```cpp
// Inside TRexSelector::runLLoopCalibration_Standard (pseudocode):
ExperimentRunnerConfig cfg;
cfg.K            = K_;
cfg.T_stop       = T_stop_;
cfg.num_dummies  = num_dummies_;
cfg.strategy     = ExperimentStrategy::Standard;
cfg.solver_type  = trex_ctrl_.solver_type;
cfg.use_warm_start = trex_ctrl_.use_warm_start;

voting_grid_ = experiment_runner_.run(cfg);
```

All six methods collapse into one call with a strategy tag.

---

## 4. Refactoring Roadmap

Each phase is sequenced to deliver value quickly, keep risk contained, and ensure that tests can be written before the next phase begins.

---

### Phase 1 — Quick Wins (Low Risk)

**Target problems:** P1, P5

**Actions:**

1. **Extract `SolverDispatch`** (Module C)  
   - Introduce `trex_solver_dispatch.h` with `runSolverExperiment<SolverT>` and `dispatchSolverExperiment`.
   - Refactor the 140-line switch/case in `runSingleExperimentWithSolver` to delegate to `dispatchSolverExperiment`.
   - The method signature of `runSingleExperimentWithSolver` is unchanged — it becomes a 10-line wrapper.

2. **Extract `DataNormalizer`** (Module A)  
   - Introduce `trex_data_normalizer.h`.
   - Replace the five member-function bodies in the `.cpp` with one-line delegations to the free functions.
   - Add `NormalizationParams norm_params_` to TRexSelector; replace the three individual member references with `norm_params_.X_means`, etc.

**Estimated effort:** 1–2 days

**Risk level:** Low. No virtual methods are touched. No interaction between the two changes.

**Testing strategy:**
- Unit tests for `DataNormalizer`: construct known `X` and `y`, call each function, verify means/norms/values.
- Unit tests for `dispatchSolverExperiment`: mock the solver types (inject a test double), verify that warm-start path calls `deserialize` and cold-start path calls the constructor.
- Run the existing integration test suite against the unchanged `TRexSelector` public API.

---

### Phase 2 — Medium Effort

**Target problems:** P4, P6

**Actions:**

1. **Extract `WarmStartManager`** (Module D)  
   - Introduce `trex_warm_start_manager.h/.cpp`.
   - Replace `createTempDirectory`, `cleanupTempDirectory`, and the three members with `std::optional<WarmStartManager> warm_start_mgr_` in TRexSelector.
   - In `select()`, replace:
     ```cpp
     createTempDirectory();
     // ... algorithm ...
     cleanupTempDirectory();
     ```
     with:
     ```cpp
     std::optional<WarmStartManager> warm_start_mgr;
     if (trex_ctrl_.use_warm_start) warm_start_mgr.emplace(K_);
     // ... algorithm uses warm_start_mgr if set ...
     // destructor handles cleanup automatically
     ```

2. **Extract `DummyGenerator`** (Module B)  
   - Introduce `trex_dummy_generator.h/.cpp`.
   - Add `DummyGenerator dummy_gen_` as a member of TRexSelector.
   - Replace all four method bodies with delegations.
   - Protected members `base_dummies_perm_`, `base_seed_perm_`, `base_dummies_initialized_` are removed from TRexSelector's protected section.

**Estimated effort:** 2–3 days

**Risk level:** Medium. The warm-start extraction changes exception safety semantics (for the better — RAII now handles cleanup). Confirm that no derived class calls `createTempDirectory` or `cleanupTempDirectory` directly before removing them.

**Testing strategy:**
- Unit test `WarmStartManager` lifecycle: construct with K=5, verify five files exist, let the object destruct, verify files are removed.
- Unit test exception safety: throw inside a lambda that holds a `WarmStartManager`, verify cleanup runs.
- Unit test `DummyGenerator`: verify that repeated calls to `getPermuted` with the same seed return identical results, and that `reset()` clears the cache.
- Full regression test with integration suite.

---

### Phase 3 — High Impact

**Target problems:** P2, P3

**Actions:**

1. **Extract `MemoryMapManager`** (Module E)  
   - Introduce `trex_memmap_manager.h/.cpp`.
   - Implement `getAugmentedMatrix(k, D)` returning `Eigen::Ref<const Eigen::MatrixXd>`.
   - Remove `initializeMemoryMappedMatrices`, `writeXToMemoryMappedMatrices`, `cleanupMemoryMappedMatrices` from TRexSelector.
   - Eliminate all `if (!trex_ctrl_.use_memory_mapping)` branches from L-loop strategies — they now call `getAugmentedMatrix` uniformly.

2. **Extract `ExperimentRunner`** (Module F)  
   - Introduce `trex_experiment_runner.h/.cpp`.
   - Move the bodies of all six `runRandomExperiments*` methods into `ExperimentRunner::run` and `runSingleExperiment`.
   - TRexSelector holds `ExperimentRunner experiment_runner_`.
   - All six public `runRandomExperiments*` methods in TRexSelector become either removed (if they are only called internally) or thin one-line delegations.

**Estimated effort:** 4–6 days (the largest phase)

**Risk level:** High. This is the most code-movement and requires careful attention to:
- Derived-class overrides of any of the six experiment runner methods (audit inheritance chain first).
- Thread safety: if `ExperimentRunner::run` is called from multiple threads, shared state in `DummyGenerator` or `WarmStartManager` must be guarded.
- The `Eigen::Ref` return from `MemoryMapManager::getAugmentedMatrix` must not outlive the manager.

**Testing strategy:**
- Integration tests comparing outputs of each `runRandomExperiments*` variant before and after, using fixed random seeds.
- Dedicated tests for `MemoryMapManager`: write a known X, assemble XD with known dummies, verify values at specific indices.
- Valgrind / AddressSanitizer run to confirm no use-after-free on the `Eigen::Ref` lifetime.
- Benchmark to confirm no regression in runtime performance from the abstraction layers.

---

### Phase 4 — Polish

**Target problems:** P7, P8

**Actions:**

1. **Reduce protected member count:**  
   After Phases 1–3, the following groups have already migrated:
   - Normalization state → `norm_params_`
   - Warm-start state → `warm_start_mgr_`
   - Memory-map state → `memmap_mgr_`
   - Dummy state → `dummy_gen_`
   
   Remaining protected members should be reviewed. Configuration members (`n_`, `p_`, `K_`, `T_stop_`) can be grouped into a `SelectionContext` struct. Result containers (`voting_grid_`, `selected_`, `FDP_hat_`) can remain but should be documented with clear ownership comments.

2. **Consolidate L-loop while-loop boilerplate:**  
   Extract the shared tail of each L-loop strategy's while loop:
   ```cpp
   void TRexSelector::advanceLLoop(double& FDP_hat, double& Phi_prime,
                                    int& dummy_multiplier, ...) {
       FDP_hat       = computeFDPHat(...);
       Phi_prime     = computePhiPrime(...);
       if (trex_ctrl_.verbose) { logLLoopIteration(...); }
       dummy_multiplier++;
   }
   ```
   Each of the four strategy while loops calls `advanceLLoop()` at the bottom.

3. **Group header members with comments:**  
   Divide the protected section into clearly labelled subsections with inline Doxygen comments explaining each group's lifecycle.

**Estimated effort:** 1–2 days

**Risk level:** Low. These are purely organizational changes — no behavioral difference.

**Testing strategy:** Compile-only verification plus existing integration suite.

---

## 5. Impact on Derived Classes

### Current Inheritance Structure

The known derived classes are:

| Class | Primary override |
|-------|-----------------|
| DA-TRex | `select()`, possibly L-loop strategy methods |
| Screen-TRex | `select()`, experiment runner selection |
| DP-TRex | `select()` |
| GVS-TRex | `select()`, dummy generation |
| Sparse-PCA-TRex | `select()`, normalization |

All derived classes access TRexSelector's protected members and call its protected methods.

### What Changes for Derived Classes

**Nothing changes in Phase 1 (SolverDispatch + DataNormalizer).**  
The free functions in `trex::utils` are accessible from derived classes. The normalization member functions remain in TRexSelector as thin wrappers that delegate to the free functions — derived class call sites do not change.

**Phase 2 changes two things for derived classes:**

1. `createTempDirectory()` and `cleanupTempDirectory()` will no longer exist as callable protected methods. Before removing them, audit every derived class. If any derived class calls them directly, replace those calls with explicit `WarmStartManager` construction at the derived call site, or expose `WarmStartManager& getWarmStartManager()` as a protected accessor.

2. `generateDummies(...)` and its siblings become delegations to `dummy_gen_.generate(...)`. Any derived class that has overridden `generateDummies` must be reviewed. The preferred pattern for derived classes that need custom dummy generation is to override `virtual DummyGenerator& getDummyGenerator()` to return a subclass of `DummyGenerator`. If that virtual point of extension is added to TRexSelector, derived classes retain full control of dummy logic without bypassing the encapsulation.

**Phase 3 changes the experiment runner surface:**

The six `runRandomExperiments*` methods are virtual (or called from virtual methods) in some derived classes. The refactoring must preserve the same virtual dispatch points. Two options:

- **Option A (recommended):** Keep thin virtual stubs in TRexSelector that delegate to `ExperimentRunner`. Derived classes that currently override `runRandomExperiments_Permutation`, for example, can either override the stub or configure the `ExperimentRunner` with a custom strategy.

- **Option B:** Make `ExperimentRunner` itself subclassable and expose `virtual ExperimentRunner& getExperimentRunner()` on TRexSelector. Derived classes that need a completely custom experiment loop return their own runner.

Option A requires fewer changes to derived classes and is the lower-risk choice for an initial refactoring. Option B is architecturally cleaner but requires all derived classes to be updated simultaneously.

### Preserving the Polymorphic Contract

The following virtual methods in TRexSelector's public/protected interface must not change signature:

```
virtual void select()
virtual Eigen::MatrixXd computeFDPHat(...)
virtual Eigen::MatrixXd computePhiPrime(...)
virtual void runTLoopCalibration(...)
virtual void applyLLoopStrategy(...)
```

All proposed extractions operate strictly below these virtual dispatch points. Derived classes that override only `select()` are unaffected by all phases. Derived classes that override deeper methods (e.g., `runRandomExperiments_Permutation`) require the stub preservation described in Option A above.

---

## 6. Proposed Directory Structure

The current layout (assumed to be a single `trex_selector.h` / `trex_selector.cpp` pair plus derived class files) becomes:

```
trex/
├── core/
│   ├── trex_selector.h                  ← slimmed TRexSelector (owns the modules)
│   ├── trex_selector.cpp
│   │
│   ├── trex_data_normalizer.h           ← Module A: free functions, NormalizationParams
│   │                                      (header-only, no .cpp needed)
│   │
│   ├── trex_dummy_generator.h           ← Module B: DummyGenerator class
│   ├── trex_dummy_generator.cpp
│   │
│   ├── trex_solver_dispatch.h           ← Module C: runSolverExperiment<T>, dispatcher
│   │                                      (header-only template + inline dispatcher)
│   │
│   ├── trex_warm_start_manager.h        ← Module D: WarmStartManager RAII class
│   ├── trex_warm_start_manager.cpp
│   │
│   ├── trex_memmap_manager.h            ← Module E: MemoryMapManager RAII class
│   ├── trex_memmap_manager.cpp
│   │
│   └── trex_experiment_runner.h         ← Module F: ExperimentRunner class
│       trex_experiment_runner.cpp
│
├── variants/
│   ├── da_trex_selector.h
│   ├── da_trex_selector.cpp
│   ├── screen_trex_selector.h
│   ├── screen_trex_selector.cpp
│   ├── dp_trex_selector.h
│   ├── dp_trex_selector.cpp
│   ├── gvs_trex_selector.h
│   ├── gvs_trex_selector.cpp
│   └── sparse_pca_trex_selector.h
│       sparse_pca_trex_selector.cpp
│
├── utils/
│   └── trex_data_normalizer.h           ← (alternative placement if preferred
│                                            over core/ for the free-function module)
│
└── tests/
    ├── test_data_normalizer.cpp         ← Unit tests for Module A
    ├── test_dummy_generator.cpp         ← Unit tests for Module B
    ├── test_solver_dispatch.cpp         ← Unit tests for Module C
    ├── test_warm_start_manager.cpp      ← Unit tests for Module D
    ├── test_memmap_manager.cpp          ← Unit tests for Module E
    ├── test_experiment_runner.cpp       ← Unit tests for Module F
    └── test_trex_selector_integration.cpp  ← Existing integration tests (unchanged)
```

### Summary: Before and After

| Metric | Before | After (all phases) |
|--------|--------|--------------------|
| TRexSelector header lines | ~1107 | ~350 (estimated) |
| TRexSelector implementation lines | ~2651 | ~600 (estimated) |
| Distinct responsibilities in TRexSelector | 9 | 1 (orchestration) |
| Warm-start protocol change effort | Edit 7 switch arms | Edit 1 template function |
| Adding a new solver type | Copy-paste 18 lines | Add 1 dispatcher line |
| Unit-testable modules | 0 | 6 |
| `if (!use_memory_mapping)` branches | 11 | 0 (hidden in `MemoryMapManager`) |

---
