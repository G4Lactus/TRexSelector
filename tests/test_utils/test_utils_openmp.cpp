#include <gtest/gtest.h>
#include <utils/openmp/utils_openmp.hpp>

// Verify that the wrapper functions complete successfully 
// without crashing, both in OpenMP-enabled and disabled environments.
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

    // Timers
    double time = omp_get_wtime();
    double tick = omp_get_wtick();
    EXPECT_GT(time, 0.0);
    EXPECT_GT(tick, 0.0);
}

TEST(OpenMPTest, UtilsNamespace) {
    // Just verifying strings and calls are correctly hooked up
    bool available = trex::utils::openmp::is_available();
    std::string status = trex::utils::openmp::status_string();
    
    if (available) {
        EXPECT_EQ(status, "OpenMP enabled");
    } else {
        EXPECT_EQ(status, "Disabled");
    }

    // Should not crash
    trex::utils::openmp::print_info();
}
