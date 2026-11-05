// ========================================================================================
/**
 * @file test_utils_disk_space.cpp
 * @brief Free-space checks (disk_space.hpp) and their use in the memory-mapped
 *  matrix, the temp-dir manager and the dummy-matrix manager: a mapping that
 *  cannot fit on its volume must throw before any file or directory is created,
 *  instead of ending the process with SIGBUS at a later mapped write.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// project includes
#include <utils/disk_space.hpp>
#include <utils/memmap/memory_mapped_matrix.hpp>
#include <trex_selector_methods/ad/trex_utils/trex_memmap_manager.hpp>
#include <trex_selector_methods/ad/trex_utils/trex_temp_dir_manager.hpp>

// std includes
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

// ========================================================================================

namespace trex::test::utils::disk_space {

namespace ds = trex::utils::disk_space;
namespace fs = std::filesystem;

// Far beyond any real volume, still representable
constexpr std::uintmax_t HUGE_BYTES = std::numeric_limits<std::uintmax_t>::max() / 2;

/** @brief Number of entries in dir whose name starts with prefix. */
static std::size_t countWithPrefix(const fs::path& dir, const std::string& prefix) {
    std::size_t count = 0;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; it != end && !ec; it.increment(ec)) {
        if (it->path().filename().string().rfind(prefix, 0) == 0) ++count;
    }
    return count;
}

// ========================================================================================
// Helper functions
// ========================================================================================

/** @brief A missing path resolves to its nearest existing ancestor. */
TEST(DiskSpace, ExistingDirectoryWalksUpToAnAncestor) {
    const fs::path tmp = fs::temp_directory_path();
    const fs::path missing = tmp / "trex_no_such_dir_8f3a" / "deeper" / "file.bin";
    EXPECT_EQ(fs::canonical(ds::existingDirectory(missing)), fs::canonical(tmp));
    EXPECT_FALSE(ds::existingDirectory("").empty());   // working directory
}

/** @brief Small and zero requirements pass; the free space is queryable. */
TEST(DiskSpace, SmallRequirementPasses) {
    std::uintmax_t available = 0;
    ASSERT_TRUE(ds::availableBytes(fs::temp_directory_path(), available));
    EXPECT_GT(available, 0u);
    EXPECT_NO_THROW(ds::requireFreeSpace(fs::temp_directory_path(), 0, "zero"));
    EXPECT_NO_THROW(ds::requireFreeSpace(fs::temp_directory_path(), 1, "one byte"));
}

/** @brief An unfulfillable requirement throws with sizes, directory and hint. */
TEST(DiskSpace, HugeRequirementThrowsWithActionableMessage) {
    const fs::path tmp = fs::temp_directory_path();
    try {
        ds::requireFreeSpace(tmp, HUGE_BYTES, "The test data", "Use a bigger disk.");
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("The test data needs up to"), std::string::npos) << msg;
        EXPECT_NE(msg.find("GiB"), std::string::npos) << msg;
        EXPECT_NE(msg.find("SIGBUS"), std::string::npos) << msg;
        EXPECT_NE(msg.find("Use a bigger disk."), std::string::npos) << msg;
    }
}

// ========================================================================================
// Users of the check
// ========================================================================================

/** @brief A matrix too large for its volume throws and leaves no file behind. */
TEST(DiskSpace, MemoryMappedMatrixTooLargeThrowsBeforeCreatingTheFile) {
    const fs::path file = fs::temp_directory_path() / "trex_disk_space_test_matrix.bin";
    fs::remove(file);
    // 2^20 x 2^30 doubles = 8 PiB: no size_t overflow, but no disk holds it
    const std::size_t rows = std::size_t{1} << 20;
    const std::size_t cols = std::size_t{1} << 30;
    EXPECT_THROW((trex::utils::memmap::MemoryMappedMatrix<double>(
                     file.string(), rows, cols, trex::utils::memmap::AccessMode::ReadWrite)),
                 std::runtime_error);
    EXPECT_FALSE(fs::exists(file));
}

/** @brief A matrix that fits is still created as before. */
TEST(DiskSpace, MemoryMappedMatrixThatFitsIsCreated) {
    const fs::path file = fs::temp_directory_path() / "trex_disk_space_test_small.bin";
    fs::remove(file);
    {
        trex::utils::memmap::MemoryMappedMatrix<double> mm(
            file.string(), 8, 4, trex::utils::memmap::AccessMode::ReadWrite);
        mm.getMap().setConstant(1.0);
    }
    EXPECT_EQ(fs::file_size(file), 8u * 4u * sizeof(double));
    fs::remove(file);
}

/** @brief TempDirManager refuses to create a directory it cannot fill. */
TEST(DiskSpace, TempDirManagerChecksBeforeCreating) {
    namespace txfs = trex::trex_selector_methods::ad::utils::filesystem;
    const fs::path base = txfs::TempDirManager::baseDir();
    const std::size_t before = countWithPrefix(base, "trex_disk_space_test_");
    EXPECT_THROW(txfs::TempDirManager::create("trex_disk_space_test_", HUGE_BYTES, "The test dir"),
                 std::runtime_error);
    EXPECT_EQ(countWithPrefix(base, "trex_disk_space_test_"), before);
}

/** @brief The dummy-matrix manager checks the worst case at the dummy cap. */
TEST(DiskSpace, MemmapManagerChecksWorstCaseBeforeCreating) {
    namespace mmm = trex::trex_selector_methods::ad::utils::memmap_manager;
    namespace txfs = trex::trex_selector_methods::ad::utils::filesystem;
    const fs::path base = txfs::TempDirManager::baseDir();
    const std::size_t before = countWithPrefix(base, "trex_memmap_");

    // K = 20 files of 2^20 x 2^24 doubles (128 TiB each): too large for any disk
    mmm::MemmapManager manager(std::size_t{1} << 20, std::size_t{1} << 24, 20, false, false);
    EXPECT_EQ(manager.requiredBytes(), (std::size_t{1} << 20) * (std::size_t{1} << 24) * 8u * 20u);
    try {
        manager.initialize();
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("MemmapManager"), std::string::npos) << msg;
        EXPECT_NE(msg.find("max_dummy_multiplier"), std::string::npos) << msg;
    }
    EXPECT_FALSE(manager.isInitialized());
    EXPECT_EQ(countWithPrefix(base, "trex_memmap_"), before);

    // A small manager still initializes and cleans up
    mmm::MemmapManager small(10, 20, 2, false, false);
    EXPECT_NO_THROW(small.initialize());
    EXPECT_EQ(small.numFiles(), 2u);
    small.cleanup();
}

// ========================================================================================
} /* End of namespace trex::test::utils::disk_space */
