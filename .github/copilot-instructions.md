# TRexSelector — Agent Instructions

C++20 statistical computing library implementing the **T-Rex Selector** algorithm for FDR-controlled variable selection in regression/classification. Uses Eigen3, Boost, Cereal, OpenMP, and BLAS/LAPACK.

## Build and Test

```bash
# Configure + build (Debug)
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build/debug

# Configure + build (Release)
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release

# Or use VS Code tasks: "CMake Full Build (Debug)" / "CMake Full Build (Release)"
```

Executables land in `build/{debug,release}/bin/`. There is no automated test suite yet — validation is done via demo executables in `src/demos/` and simulation outputs in `simulations/`.

**macOS pitfall:** CMake must use Homebrew LLVM, not Apple Clang (OpenMP requires it). If IntelliSense or builds break after a branch switch or library upgrade, follow the cache-clearing and reconfiguration steps in [README.md](../README.md).

## Architecture

Layered dependency graph (bottom → top):

| Layer                  | CMake target                  | Directory                    |
| ---------------------- | ----------------------------- | ---------------------------- |
| Foundation             | `trex::utils`                 | `src/utils/`                 |
| ML algorithms          | `trex::ml_methods`            | `src/ml_methods/`            |
| Terminating solvers    | `trex::tsolvers`              | `src/tsolvers/`              |
| TRex selector variants | `trex::trex_selector_methods` | `src/trex_selector_methods/` |
| Demos / executables    | —                             | `src/demos/`                 |

Upper layers link lower layers via `target_link_libraries(... PUBLIC ...)` — transitively. Do not add reverse dependencies.

**Key classes:**

- `TRexSelector` (`trex_core/trex.hpp`) — thin orchestrator; entry point is `SelectionResult select()`; owns six extracted utility classes (refactoring complete — see `trex_core/deprecated/trex_refactoring_analysis.md`)
- `TSolver_Base` (`tsolvers/tsolver_base.hpp`) — abstract base for all solvers; concrete solvers live in `tsolvers/linear_model/` (LARS-based and OMP-based families; logistic model is planned but unimplemented)
- `MemoryMappedMatrix<Scalar>` (`utils/memmap/`) — Boost-backed mmap'd Eigen matrices for large datasets

**Extracted utility classes in `trex_selector_methods/utils/`:**

- `TRexDataNormalizer` — column normalisation
- `TRexDummyGenerator` — pluggable dummy-matrix construction (virtual hooks for subclasses)
- `trex_solver_dispatch` — `SolverConfig` struct + `dispatchSolver<SolverT>(config)` template; eliminates per-solver switch arms
- `WarmStartManager` — RAII temp-directory + K solver-file lifetimes (exception-safe)
- `TRexMemMapManager` — uniform `getActiveMap(k, num_dummies)` interface; callers have zero `if (use_memory_mapping)` branches
- `ExperimentRunner` — single experiment-loop implementation shared by all `runRandomExperiments*` variants

## Conventions

**Files:** `ClassName.hpp` / `ClassName.cpp` split. Header-only for fully templated code. Umbrella headers (e.g., `lm_tsolvers.hpp`) aggregate related types.

**Include guards:** `#ifndef TREX_NAMESPACE_CLASSNAME_HPP`

**Namespaces:** Hierarchical, underscore-separated (`trex::tsolvers::linear_model::lars_based`). Use namespace aliases in `.cpp` files for brevity (`namespace sd = trex::...::solver_dispatch;`).

**Documentation:** Doxygen `/** @brief / @param / @return */` on every public class and method. Section headers with `// ===...===` visual separators.

**Eigen idioms:** Data matrices passed as `Eigen::Map<Eigen::MatrixXd>&`; stored as raw pointers in classes (null-initialized). Smart pointers (`std::unique_ptr`) for long-lived helpers.

**Serialization:** All solver classes register with Cereal polymorphic serialization — add `CEREAL_REGISTER_TYPE(MyNewSolver)` when adding a new concrete solver. Use `cereal::PortableBinaryArchive`; expose instance `save(file)` and static `load(file, X)` methods in `.cpp`.

**Adding a new solver** requires three constructors (full, default for Cereal, protected type-enum for inheritance), two pure virtual overrides (`executeStep(T_stop, early_stop)` and `solverTypeToString()`), and a `serialize(Archive&)` template calling `cereal::base_class<Base>(this)`. See `TENET_Solver` as a reference implementation.

**Adding a new demo:** Use the `add_demo_target()` macro in `src/demos/CMakeLists.txt` and link against the appropriate preset group (`DATA_PREPROCESSING_LIBS`, `T_ALGORITHM_LIBS`, or `T_REX_SELECTOR_LIBS`).

## Known Issues / Active Debt

- Logistic model solvers (`tsolvers/logistic_model/`) are planned but not yet implemented.
- See [notes/Issues_with_GVS_TRex.md](../notes/Issues_with_GVS_TRex.md) for known GVS variant issues.

**Previously noted debt that has been resolved:**

- Solver dispatch now uses `SolverConfig` + `dispatchSolver<SolverT>()` — no 7-arm switch.
- Experiment loops unified via `ExperimentRunner`.
- Warm-start temp files are RAII-guarded via `WarmStartManager`.
