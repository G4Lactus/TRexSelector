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
    gtest_discover_tests(${TEST_NAME})
endfunction()

# 1. Add the Sanity Test Executable
add_trex_test(test_sanity test_sanity.cpp)

# 2. Update paths for utils tests
add_trex_test(test_utils_eval_metrics test_utils/test_utils_eval_metrics.cpp)
add_trex_test(test_utils_serialization test_utils/test_utils_serialization.cpp)
add_trex_test(test_utils_fp_classify test_utils/test_utils_fp_classify.cpp)
add_trex_test(test_utils_openmp test_utils/test_utils_openmp.cpp)
add_trex_test(test_utils_logging test_utils/test_utils_logging.cpp)
add_trex_test(test_utils_memmap test_utils/test_utils_memmap.cpp)
add_trex_test(test_utils_dummygen test_utils/test_utils_dummygen.cpp)
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

# 4. Add tests for TSolvers
add_trex_test(test_tsolvers_initialization test_tsolvers/test_tsolvers_initialization.cpp)
add_trex_test(test_tsolvers_execution test_tsolvers/test_tsolvers_execution.cpp)
add_trex_test(test_tsolvers_getters test_tsolvers/test_tsolvers_getters.cpp)
add_trex_test(test_tsolvers_serialization test_tsolvers/test_tsolvers_serialization.cpp)
add_trex_test(test_tsolvers_mmap test_tsolvers/test_tsolvers_mmap.cpp)
add_trex_test(test_tsolvers_polymorphic test_tsolvers/test_tsolvers_polymorphic.cpp)

# 5. Add tests for TRex Selector Methods
add_trex_test(test_trex_core test_trex_selector_methods/test_trex_core.cpp)
add_trex_test(test_trex_da test_trex_selector_methods/test_trex_da.cpp)
add_trex_test(test_trex_gvs test_trex_selector_methods/test_trex_gvs.cpp)
add_trex_test(test_trex_screening test_trex_selector_methods/test_trex_screening.cpp)
add_trex_test(test_trex_biobank_screening test_trex_selector_methods/test_trex_biobank_screening.cpp)
add_trex_test(test_trex_spca test_trex_selector_methods/test_trex_spca.cpp)
