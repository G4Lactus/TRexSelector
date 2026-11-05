// =================================================================================
// disk_space.hpp
// =================================================================================
#ifndef TREX_UTILS_DISK_SPACE_HPP
#define TREX_UTILS_DISK_SPACE_HPP
// =================================================================================
/**
 * @file disk_space.hpp
 *
 * @brief Free-space checks for the files that back memory mappings.
 *
 * @details Memory-mapped files are created sparse: resizing a file to its full
 *  logical size reserves no blocks, and the disk is only consumed when pages
 *  are written. A full disk cannot be reported at that point, so the process
 *  ends with SIGBUS. These helpers compare the bytes a mapping may write with
 *  the free space of the target volume up front and throw std::runtime_error
 *  with the sizes, the directory and a remedy instead.
 */
// =================================================================================

#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

// =================================================================================

namespace trex::utils::disk_space {

// =================================================================================

/**
 * @brief Nearest existing directory at or above path (empty if none exists).
 *
 * @param path File or directory location; an empty path means the working directory.
 */
inline std::filesystem::path existingDirectory(const std::filesystem::path& path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path dir = path.empty() ? fs::current_path(ec) : fs::absolute(path, ec);
    if (ec) return {};
    while (!dir.empty()) {
        if (fs::is_directory(dir, ec)) return dir;
        const fs::path parent = dir.parent_path();
        if (parent == dir) break;
        dir = parent;
    }
    return {};
}

/**
 * @brief Free bytes available to this process on the volume holding path.
 *
 * @param path      File or directory location.
 * @param available Receives the free bytes on success.
 * @return false if the volume cannot be queried (no existing ancestor, or the
 *         file system reports no space information).
 */
inline bool availableBytes(const std::filesystem::path& path, std::uintmax_t& available) {
    const std::filesystem::path dir = existingDirectory(path);
    if (dir.empty()) return false;
    std::error_code ec;
    const std::filesystem::space_info info = std::filesystem::space(dir, ec);
    if (ec) return false;
    available = info.available;
    return true;
}

/** @brief Byte count in GiB with two decimals, e.g. "1898.68". */
inline std::string formatGiB(std::uintmax_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2)
        << static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    return out.str();
}

/**
 * @brief Throw std::runtime_error unless required_bytes are free on the volume
 *        holding path.
 *
 * @details A volume that cannot be queried passes (the check cannot block a
 *  run it cannot judge). The check is a snapshot: other processes may still
 *  fill the disk later.
 *
 * @param path           Directory or file location the bytes will be written to.
 * @param required_bytes Bytes that may be written there.
 * @param what           Short description of the data, starts the message.
 * @param hint           Remedy appended to the message (may be empty).
 */
inline void requireFreeSpace(const std::filesystem::path& path, std::uintmax_t required_bytes,
                             const std::string& what, const std::string& hint = "") {
    if (required_bytes == 0) return;
    std::uintmax_t available = 0;
    if (!availableBytes(path, available) || available >= required_bytes) return;

    std::ostringstream msg;
    msg << what << " needs up to " << formatGiB(required_bytes)
        << " GiB of disk space in '" << existingDirectory(path).string()
        << "', but only " << formatGiB(available) << " GiB are free. A write into a "
        << "memory-mapped file on a full disk ends the process with SIGBUS, so the "
        << "run stops here.";
    if (!hint.empty()) msg << " " << hint;
    throw std::runtime_error(msg.str());
}

// =================================================================================
} /* End of namespace trex::utils::disk_space */

#endif /* TREX_UTILS_DISK_SPACE_HPP */
