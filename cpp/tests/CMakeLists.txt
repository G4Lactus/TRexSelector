# =============================================================================
# C++ Unit Tests (GoogleTest)
# =============================================================================

# Find GoogleTest package
find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

# Common helper function to add a target
function(add_trex_test TEST_NAME)
    # Use ${ARGN} to accept the custom file paths passed to the function
    add_executable(${TEST_NAME} ${ARGN})
    target_link_libraries(${TEST_NAME}
        PRIVATE
            GTest::gtest_main
            GTest::gtest
            trex::utils
            trex::tsolvers
            trex::ml_methods
            trex::trex_selector_methods
    )
    target_compile_features(${TEST_NAME} PRIVATE cxx_std_20)
    # PRE_TEST: run test discovery serially at ctest time instead of as a
    # POST_BUILD step. During a saturated parallel build the post-build
    # discovery process can get starved/killed on slow CI runners, which
    # current CMake surfaces as a JSON parse error in ParseTestList.cmake
    # (empty --gtest_output=json result; seen on the macos-26 runner even
    # with DISCOVERY_TIMEOUT 30).
    gtest_discover_tests(${TEST_NAME}
        DISCOVERY_TIMEOUT 30
        DISCOVERY_MODE PRE_TEST)
endfunction()

# 1. Add the Sanity Test Executable
add_trex_test(test_sanity test_sanity.cpp)

# 2. Update paths for utils tests
add_trex_test(test_utils_eval_metrics test_utils/test_utils_eval_metrics.cpp)
add_trex_test(test_utils_serialization test_utils/test_utils_serialization.cpp)
add_trex_test(test_utils_fp_classify test_utils/test_utils_fp_classify.cpp)
add_trex_test(test_utils_openmp test_utils/test_utils_openmp.cpp)
add_trex_test(test_utils_thread_budget test_utils/test_utils_thread_budget.cpp)
add_trex_test(test_utils_logging test_utils/test_utils_logging.cpp)
add_trex_test(test_utils_memmap test_utils/test_utils_memmap.cpp)
add_trex_test(test_utils_dummygen test_utils/test_utils_dummygen.cpp)
add_trex_test(test_utils_dummy_seeding test_utils/test_utils_dummy_seeding.cpp)
add_trex_test(test_utils_dummy_scaling test_utils/test_utils_dummy_scaling.cpp)
add_trex_test(test_utils_noisegen test_utils/test_utils_noisegen.cpp)
add_trex_test(test_utils_datagen test_utils/test_utils_datagen.cpp)

# 3. Add tests for ML Methods
add_trex_test(test_ml_methods_standardization test_ml_methods/test_ml_methods_standardization.cpp)
add_trex_test(test_ml_methods_model_selection test_ml_methods/test_ml_methods_model_selection.cpp)
add_trex_test(test_ml_methods_agglomerative_cores test_ml_methods/test_ml_methods_agglomerative_cores.cpp)
add_trex_test(test_ml_methods_agglomerative_dispatcher test_ml_methods/test_ml_methods_agglomerative_dispatcher.cpp)
add_trex_test(test_ml_methods_svd test_ml_methods/test_ml_methods_svd.cpp)
add_trex_test(test_ml_methods_pca test_ml_methods/test_ml_methods_pca.cpp)
add_trex_test(test_ml_methods_ridge test_ml_methods/test_ml_methods_ridge.cpp)
# Zero-copy harness: bit-identity against
# a recorded reference (skipped unless TREX_ML_REF_DIR is set) and the peak-
# memory budgets of the cores in units of X.
add_trex_test(test_ml_methods_reference test_ml_methods/test_ml_methods_reference.cpp)
add_trex_test(test_ml_methods_memory test_ml_methods/test_ml_methods_memory.cpp)
add_trex_test(test_ml_methods_low_memory test_ml_methods/test_ml_methods_low_memory.cpp)

# 4. Add tests for TSolvers
add_trex_test(test_tsolvers_initialization test_tsolvers/ad/test_tsolvers_initialization.cpp)
add_trex_test(test_tsolvers_execution test_tsolvers/ad/test_tsolvers_execution.cpp)
add_trex_test(test_tsolvers_getters test_tsolvers/ad/test_tsolvers_getters.cpp)
add_trex_test(test_tsolvers_serialization test_tsolvers/ad/test_tsolvers_serialization.cpp)
add_trex_test(test_tsolvers_mmap test_tsolvers/ad/test_tsolvers_mmap.cpp)
add_trex_test(test_tsolvers_polymorphic test_tsolvers/ad/test_tsolvers_polymorphic.cpp)
add_trex_test(test_tsolvers_cd test_tsolvers/ad/test_tsolvers_cd.cpp)
add_trex_test(test_tsolvers_tik test_tsolvers/ad/test_tsolvers_tik.cpp)
add_trex_test(test_tsolvers_buffer_guard test_tsolvers/ad/test_tsolvers_buffer_guard.cpp)
# memoryBytes(): hand counts, growth, exclusions; the peak test replaces
# malloc / operator new in its own executable (macOS and glibc, else skipped).
add_trex_test(test_tsolvers_memory test_tsolvers/ad/test_tsolvers_memory.cpp)
add_trex_test(test_tsolvers_memory_peak test_tsolvers/ad/test_tsolvers_memory_peak.cpp)
add_trex_test(test_trex_restore test_trex_selector_methods/ad/test_trex_restore.cpp)
add_trex_test(test_sd_tsolvers test_tsolvers/sd/test_sd_tsolvers.cpp)

# 5. Add tests for TRex Selector Methods
add_trex_test(test_trex_core test_trex_selector_methods/ad/test_trex_core.cpp)
add_trex_test(test_trex_lloop_seeded test_trex_selector_methods/ad/test_trex_lloop_seeded.cpp)
add_trex_test(test_trex_da test_trex_selector_methods/ad/test_trex_da.cpp)
add_trex_test(test_trex_gvs test_trex_selector_methods/ad/test_trex_gvs.cpp)
add_trex_test(test_trex_screening test_trex_selector_methods/ad/test_trex_screening.cpp)
add_trex_test(test_trex_biobank_screening test_trex_selector_methods/ad/test_trex_biobank_screening.cpp)
add_trex_test(test_trex_spca test_trex_selector_methods/ad/test_trex_spca.cpp)
add_trex_test(test_trex_tikhonov test_trex_selector_methods/ad/test_trex_tikhonov.cpp)
add_trex_test(test_tik_lambda2_power_rule test_trex_selector_methods/ad/test_tik_lambda2_power_rule.cpp)
add_trex_test(test_ml_methods_agglomerative_guards test_ml_methods/test_ml_methods_agglomerative_guards.cpp)
add_trex_test(test_utils_checked_arith test_utils/test_utils_checked_arith.cpp)
add_trex_test(test_utils_disk_space test_utils/test_utils_disk_space.cpp)
# Thread hygiene: the OpenMP half runs in every OpenMP build (skipped in the
# single-threaded one, TREX_WITH_OPENMP=OFF); the Eigen half is
# compiled in only without EIGEN_USE_BLAS (Eigen stubs its thread count
# under a BLAS — see the file header), i.e. in TREX_USE_BLAS_LAPACK=OFF
# builds: CI's cpp-tests job, the R package, the Python wheel, and locally
# cpp/build/release-noblas.
add_trex_test(test_trex_thread_hygiene test_trex_selector_methods/ad/test_trex_thread_hygiene.cpp)
add_trex_test(test_trex_thread_determinism test_trex_selector_methods/ad/test_trex_thread_determinism.cpp)
add_trex_test(test_trex_vd test_trex_selector_methods/vd/test_trex_vd.cpp)
add_trex_test(test_trex_scaling_none test_trex_selector_methods/ad/test_trex_scaling_none.cpp)
add_trex_test(test_trex_lambda2_guard test_trex_selector_methods/ad/test_trex_lambda2_guard.cpp)

# 6. Opt-in cross-language validations. NOT registered with CTest and excluded
#    from the default build; run on demand via the `run_validations` target
#    (cmake --build <dir> --target run_validations). See tests/validation/.
add_subdirectory(validation)
