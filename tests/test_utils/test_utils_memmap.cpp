// ========================================================================================
/**
 * @file test_utils_memmap.cpp
 * @brief Unit tests for memory-mapped matrix utilities in memory_mapped_matrix.hpp
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// project utils includes
#include <utils/memmap/memory_mapped_matrix.hpp>
#include <utils/serialization/utils_cereal_eigen.hpp>

// std includes
#include <filesystem>
#include <sstream>
#include <cereal/archives/portable_binary.hpp>

// ========================================================================================

/** @brief Test for writing and reading a memory-mapped matrix */
TEST(MemMapTest, WriteAndRead) {
    std::string temp_file = "test_memory_mapped_matrix.bin";
    Eigen::Index rows = 4;
    Eigen::Index cols = 5;

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


/** @brief Test for Memory mapping serialization compatibility */
TEST(MemMapTest, SerializationCompatibility) {
    std::string temp_file = "test_memory_mapped_serialization.bin";
    Eigen::Index rows = 3;
    Eigen::Index cols = 3;

    if (std::filesystem::exists(temp_file)) {
        std::filesystem::remove(temp_file);
    }

    Eigen::MatrixXd original_matrix = Eigen::MatrixXd::Random(rows, cols);

    // 1. Write random data to mem map
    {
        trex::utils::memmap::MemoryMappedMatrix<double> mm_write(
            temp_file, rows, cols, trex::utils::memmap::AccessMode::ReadWrite
        );
        auto mat = mm_write.getMap();
        mat = original_matrix;
    }

    // 2. Read mem map and serialize it via Cereal
    std::stringstream string_stream;
    {
        const trex::utils::memmap::MemoryMappedMatrix<double> mm_read(
            temp_file, rows, cols, trex::utils::memmap::AccessMode::ReadOnly
        );
        const auto mapped_data = mm_read.getMap();

        // Convert to concrete matrix for serialization
        Eigen::MatrixXd mat_to_serialize = mapped_data;

        cereal::PortableBinaryOutputArchive oarchive(string_stream);
        oarchive(mat_to_serialize);
    }

    // 3. Deserialize and verify
    Eigen::MatrixXd deserialized_matrix;
    {
        cereal::PortableBinaryInputArchive iarchive(string_stream);
        iarchive(deserialized_matrix);
    }

    EXPECT_TRUE(original_matrix.isApprox(deserialized_matrix));

    // Cleanup
    if (std::filesystem::exists(temp_file)) {
        std::filesystem::remove(temp_file);
    }
}


/** @brief Test for safe file deletion after closing the memory mapping */
TEST(MemMapTest, SafeDeletion) {
    std::string temp_file = "test_memory_mapped_safe_deletion.bin";
    Eigen::Index rows = 2;
    Eigen::Index cols = 2;

    if (std::filesystem::exists(temp_file)) {
        std::filesystem::remove(temp_file);
    }

    // Create a memory mapped matrix and force it out of scope
    {
        trex::utils::memmap::MemoryMappedMatrix<double> mm(
            temp_file, rows, cols, trex::utils::memmap::AccessMode::ReadWrite
        );

        // Touch the map
        auto mat = mm.getMap();
        mat.setZero();

        EXPECT_TRUE(std::filesystem::exists(temp_file));
    } // mm goes out of scope here: destructor MUST successfully release the file handles/mapping

    // Test that we can safely remove the file now (ensures no hanging locks remain)
    std::error_code ec;
    bool removed = std::filesystem::remove(temp_file, ec);

    EXPECT_FALSE(ec) << "File removal failed with error: " << ec.message();
    EXPECT_TRUE(removed) << "File was not correctly removed, it may still be locked.";
    EXPECT_FALSE(std::filesystem::exists(temp_file));
}
