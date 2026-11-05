// ========================================================================================
/**
 * @file test_utils_checked_arith.cpp
 * @brief Overflow-checked size arithmetic (checked_arith.hpp) and its use in the
 *  memory-mapped matrix: a wrapping rows * cols * sizeof
 *  must throw before any file is touched instead of mapping a too-small file.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// project utils includes
#include <utils/checked_arith.hpp>
#include <utils/memmap/memory_mapped_matrix.hpp>

// std includes
#include <cstddef>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

// ========================================================================================

namespace trex::test::utils::checked_arith {

using trex::utils::checked_arith::checked_bytes;
using trex::utils::checked_arith::checked_mul;

// ========================================================================================

TEST(CheckedArithTest, ProductsThatFitPassThrough) {
    EXPECT_EQ(checked_mul(3, 4), 12u);
    EXPECT_EQ(checked_mul(0, std::numeric_limits<std::size_t>::max()), 0u);
    EXPECT_EQ(checked_mul(std::numeric_limits<std::size_t>::max(), 1),
              std::numeric_limits<std::size_t>::max());
    EXPECT_EQ(checked_bytes(200, 500, sizeof(double)), 200u * 500u * sizeof(double));
}


TEST(CheckedArithTest, WrappingProductsThrow) {
    const std::size_t half = std::numeric_limits<std::size_t>::max() / 2 + 1;
    EXPECT_THROW(checked_mul(half, 2), std::overflow_error);
    // (2^61 + 1) * 8 wraps to 8 in 64-bit arithmetic.
    EXPECT_THROW(checked_bytes((std::size_t{1} << 61) + 1, 1, 8), std::overflow_error);
    EXPECT_THROW(checked_bytes(std::size_t{1} << 40, std::size_t{1} << 40, 1),
                 std::overflow_error);
    try {
        (void)checked_bytes((std::size_t{1} << 61) + 1, 1, 8, "probe");
        FAIL() << "expected std::overflow_error";
    } catch (const std::overflow_error& e) {
        EXPECT_NE(std::string(e.what()).find("probe"), std::string::npos) << e.what();
    }
}


/** @brief The memory-mapped matrix refuses a wrapping extent before creating
 *  or resizing its file. */
TEST(CheckedArithTest, MemoryMappedMatrixRejectsOverflowBeforeTouchingTheFile) {
    const std::string path = ::testing::TempDir() + "checked_arith_overflow.bin";
    std::filesystem::remove(path);
    EXPECT_THROW(
        trex::utils::memmap::MemoryMappedMatrix<double>(
            path, (std::size_t{1} << 61) + 1, 8,
            trex::utils::memmap::AccessMode::ReadWrite),
        std::overflow_error);
    EXPECT_FALSE(std::filesystem::exists(path));
}

// ========================================================================================
} /* End of namespace trex::test::utils::checked_arith */
