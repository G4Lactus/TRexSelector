// ========================================================================================
/**
 * @file test_utils_openmp.cpp
 *
 * @brief Unit tests for OpenMP utilities in utils_openmp.hpp
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// project utils includes
#include <utils/openmp/utils_openmp.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::utils::openmp {

/**
 * @brief Test that the wrapper functions complete successfully without crashing, both in
 *        OpenMP-enabled and disabled environments.
 */
TEST(OpenMPTest, CoreFunctionStubs) {
    // Basic threads info
    int num_threads = omp_get_num_threads();
    int max_threads = omp_get_max_threads();
    int thread_num = omp_get_thread_num();

    EXPECT_GE(num_threads, 1);
    EXPECT_GE(max_threads, 1);
    EXPECT_GE(thread_num, 0);

    // Boolean checks
    int in_parallel = omp_in_parallel();
    EXPECT_GE(in_parallel, 0);

    // Timers: the epoch is arbitrary (the stub counts from its first call, so 0 is valid)
    double time = omp_get_wtime();
    double later = omp_get_wtime();
    double tick = omp_get_wtick();
    EXPECT_GE(time, 0.0);
    EXPECT_GE(later, time);
    EXPECT_GT(tick, 0.0);
}


/** @brief Test OpenMP utilities in the utils namespace. */
TEST(OpenMPTest, UtilsNamespace) {
    // Just verifying strings and calls are correctly hooked up
    bool available = trex::utils::openmp::is_available();
    std::string status = trex::utils::openmp::status_string();

    if (available) {
        EXPECT_EQ(status, "OpenMP enabled");
    } else {
        EXPECT_EQ(status, "Single-threaded (OpenMP disabled)");
    }

    // Should not crash
    trex::utils::openmp::print_info();
}


/** @brief Without OpenMP the stubs report one thread and ignore a requested count. */
TEST(OpenMPTest, SingleThreadedStubs) {
    if (trex::utils::openmp::is_available()) {
        GTEST_SKIP() << "OpenMP is enabled in this build; the stubs are not compiled.";
    }
    EXPECT_EQ(omp_get_max_threads(), 1);
    EXPECT_EQ(trex::utils::openmp::get_max_threads(), 1);
    trex::utils::openmp::set_num_threads(4);
    EXPECT_EQ(trex::utils::openmp::get_max_threads(), 1);
    EXPECT_EQ(trex::utils::openmp::version(), 0);
    std::cout << "single-threaded build: omp_get_max_threads() = " << omp_get_max_threads() << "\n";
}

// ========================================================================================
} /* End ofnamespace trex::test::utils::openmp */
