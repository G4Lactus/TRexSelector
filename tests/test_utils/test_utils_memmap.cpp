#include <gtest/gtest.h>
#include <utils/memmap/memory_mapped_matrix.hpp>
#include <fstream>
#include <filesystem>

TEST(MemMapTest, WriteAndRead) {
    std::string temp_file = "test_memory_mapped_matrix.bin";
    std::size_t rows = 4;
    std::size_t cols = 5;

    // Remove if it exists
    if (std::filesystem::exists(temp_file)) {
        std::filesystem::remove(temp_file);
    }

    // 1. Create and Write
    {
        trex::utils::memmap::MemoryMappedMatrix<double> mm_write(
            temp_file, rows, cols, trex::utils::memmap::AccessMode::ReadWrite
        );
        
        EXPECT_EQ(mm_write.rows(), rows);
        EXPECT_EQ(mm_write.cols(), cols);
        EXPECT_EQ(mm_write.size(), rows * cols);
        
        Eigen::Map<Eigen::MatrixXd> mat = mm_write.getMap();
        mat.setConstant(3.14159);
        mat(0, 0) = 42.0;

        // Scope end forces destruction and memory sync
    }

    // 2. Open and Read
    {
        const trex::utils::memmap::MemoryMappedMatrix<double> mm_read(
            temp_file, rows, cols, trex::utils::memmap::AccessMode::ReadOnly
        );

        const auto mat = mm_read.getMap();
        EXPECT_DOUBLE_EQ(mat(0, 0), 42.0);
        EXPECT_DOUBLE_EQ(mat(1, 1), 3.14159);
        EXPECT_DOUBLE_EQ(mat(rows - 1, cols - 1), 3.14159);
    }

    // Cleanup
    std::filesystem::remove(temp_file);
}
