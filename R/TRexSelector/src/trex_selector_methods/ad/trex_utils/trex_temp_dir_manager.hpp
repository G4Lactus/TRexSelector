// =========================================================================
// utils/filesystem/temp_dir_manager.hpp
// =========================================================================
#ifndef UTILS_FILESYSTEM_TEMP_DIR_MANAGER_HPP
#define UTILS_FILESYSTEM_TEMP_DIR_MANAGER_HPP
// =========================================================================
/**
 * @file trex_temp_dir_manager.hpp
 *
 * @brief Utility for creating and cleaning up temporary directories with
 * safety checks.
 */
// =========================================================================

// std includes
#include <filesystem>
#include <string>
#include <random>
#include <system_error>
#include <stdexcept>
#include <sstream>
#include <iostream>

// utils includes
#include <utils/disk_space.hpp>
#include <utils/global_temp_dir.hpp>

// =========================================================================

// Embedding into namespace trex::trex_selector_methods::ad::utils::filesystem
namespace trex::trex_selector_methods::ad::utils::filesystem {

// =========================================================================

class TempDirManager {
private:
    inline static std::string custom_temp_dir = "";

public:
    /**
     * @brief Set a custom base directory for temporary files (e.g. for R compatibility).
     * @param temp_dir The absolute path to use instead of the OS default temp directory.
     */
    static void setCustomTempDir(const std::string& temp_dir) {
        custom_temp_dir = temp_dir;
        trex::utils::set_global_temp_dir(temp_dir);
    }

    /**
     * @brief Base directory new temporary directories are created in (the
     *        custom one if set, else the OS default, which honours TMPDIR).
     */
    static std::filesystem::path baseDir() {
        namespace fs = std::filesystem;
        return custom_temp_dir.empty() ? fs::temp_directory_path() : fs::path(custom_temp_dir);
    }

    /**
     * @brief Creates a unique temporary directory with safety checks.
     * @param prefix The prefix for the folder name (e.g., "trex_memmap_", "trex_warmstart_").
     * @param min_space_required Minimum free bytes required
     *        (e.g., 10GB = 10ULL * 1024 * 1024 * 1024).
     * @param what Description of the data for the disk-space error message.
     * @param hint Remedy appended to the disk-space error message.
     * @return The absolute path to the created temporary directory.
     * @throws std::runtime_error if creation fails or disk space is insufficient.
     */
    static std::string create(const std::string& prefix, std::uintmax_t min_space_required = 0,
                              const std::string& what = "TempDirManager: the temporary files",
                              const std::string& hint = "") {
        namespace fs = std::filesystem;

        const fs::path temp_base = baseDir();

        // 1. Disk space check (before anything is created)
        trex::utils::disk_space::requireFreeSpace(temp_base, min_space_required, what, hint);

        // 2. RNG for unique hex string
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dis;

        // 3. Retry loop for directory creation
        const int max_retries = 10;
        for (int i = 0; i < max_retries; ++i) {
            std::stringstream hex_str;
            hex_str << std::hex << dis(gen);
            fs::path candidate = temp_base / (prefix + hex_str.str());

            std::error_code ec;
            if (fs::create_directory(candidate, ec)) {
                return candidate.string();
            }
        }

        throw std::runtime_error("TempDirManager: Failed to create temp directory after " +
                                    std::to_string(max_retries) + " attempts.");
    }


    /**
     * @brief Safely removes a directory and all its contents.
     * @param path The path to the directory.
     * @param throw_on_error If false, filesystem errors are swallowed (useful for destructors).
     */
    static void remove(const std::string& path, bool throw_on_error = true) {
        namespace fs = std::filesystem;
        if (path.empty() || !fs::exists(path)) return;

        std::error_code ec;
        fs::remove_all(path, ec);

        if (ec && throw_on_error) {
            throw std::runtime_error("TempDirManager: Failed to remove directory '" +
                path + "': " + ec.message());
        }
    }
};

// =========================================================================
} /* End of namespace trex::trex_selector_methods::ad::utils::filesystem */
// =========================================================================
#endif /* End of UTILS_FILESYSTEM_TEMP_DIR_MANAGER_HPP */
