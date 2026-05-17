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
