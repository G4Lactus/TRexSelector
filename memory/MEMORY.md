# TRexSelector Project Memory

## Project Structure
- C++20, CMake, Eigen, Boost.Iostreams, OpenMP, cereal
- Build: cmake + ninja/make (Release: -O3 -march=native -ffast-math)
- Libraries: `trex::ml_methods` (HAC, standardization) → `trex::utils` (memmap, openmp, eval)

## Key Paths
- `src/demos/ml_methods/` — ML method demos
- `src/demos/CMakeLists.txt` — single CMakeLists for ALL demos (uses `add_demo_target` macro)
- `src/ml_methods/` — HAC (SLINK, NNChain, GenericLinkage), standardization
- `src/utils/memmap/` — MemoryMappedMatrix<Scalar> (Boost-backed, ColMajor Eigen::Map)
- `src/utils/openMP/` — omp_utils::print_info()
- `src/utils/eval_metrics/` — cdiagnostics namespace

## HAC Namespace
`trex::ml_methods::clustering::hierarchical::agglomerative` (alias: `hac`)
- `hac::SLINK<MatType, DistPol>`
- `hac::NNChain<MatType, UpdaterPol>`
- `hac::DistancePolicy<MatType, hac::DistanceMetric::Correlation_LSH_Approx>`
- `hac::ClusterUpdatePolicy<MatType, DistPol, hac::LinkageMethod::Complete|Average|WPGMA, true>`
- `hac::DendrogramUtils::pointer_to_merge_matrix / format_nnchain_merges / cut_tree`

## MemoryMappedMatrix
- Template: `MemoryMappedMatrix<Scalar>` in `trex::utils::memmap`
- ColMajor layout; `getMap()` returns `Eigen::Map<Matrix<Scalar,Dyn,Dyn,ColMajor>>`
- Constructor auto-creates/resizes file for ReadWrite; validates size for ReadOnly
- `applyAccessHints()` applies `MADV_SEQUENTIAL` on POSIX
- Explicitly instantiated for: char, uchar, short, ushort, int, float, double, complex<float/double>

## Demos Registered
- `demo_01_scaler` — DATA_PREPROCESSING_LIBS
- `demo_02_hac` — DATA_PREPROCESSING_LIBS (all HAC linkages + data gen helpers)
- `demo_03_hac_mmap` — DATA_PREPROCESSING_LIBS + `TREX_DEMO_MMAP_PATH` define (N=100k, P=400k mmap demo)

## User Preferences
- High-quality, expert-level demos with clear phase structure
- `DATA_PREPROCESSING_LIBS` covers Boost + BLAS/LAPACK + trex::ml_methods (which pulls trex::utils)
- OMP threads: 6 (init_omp_environment), static schedule
