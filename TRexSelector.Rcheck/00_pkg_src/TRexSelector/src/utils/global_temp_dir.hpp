// =========================================================================
// utils/global_temp_dir.hpp
// =========================================================================
#ifndef TREX_UTILS_GLOBAL_TEMP_DIR_HPP
#define TREX_UTILS_GLOBAL_TEMP_DIR_HPP
// =========================================================================
/**
 * @file global_temp_dir.hpp
 *
 * @brief Foundation-level utility for a process-wide custom temporary
 * directory path.
 *
 * @details Provides a thin, header-only singleton string accessible from
 * any layer. Higher-level components (e.g., TempDirManager) set the value;
 * lower-level components (e.g., BlockTiledMatrixPolicy) read it without
 * creating an upward layer dependency.
 */
// =========================================================================

// std includes
#include <filesystem>
#include <string>

// =========================================================================

namespace trex::utils {

// =========================================================================

/**
 * @brief Returns a reference to the process-wide custom temporary directory
 * string. Empty string means "use OS default".
 */
inline std::string& _global_temp_dir_store() {
    static std::string s;
    return s;
}

/**
 * @brief Set the process-wide custom temporary directory.
 * @param dir Absolute path to use instead of the OS default temp directory.
 */
inline void set_global_temp_dir(const std::string& dir) {
    _global_temp_dir_store() = dir;
}

/**
 * @brief Get the process-wide temporary directory as a filesystem path.
 * @return The custom temp directory if set; otherwise the OS default.
 */
inline std::filesystem::path get_global_temp_dir() {
    const std::string& s = _global_temp_dir_store();
    return s.empty() ? std::filesystem::temp_directory_path()
                     : std::filesystem::path(s);
}

// =========================================================================
} /* End of namespace trex::utils */

#endif /* End of TREX_UTILS_GLOBAL_TEMP_DIR_HPP */
