// =================================================================================
// physical_memory.cpp
// =================================================================================
/**
 * @file physical_memory.cpp
 *
 * @brief Platform query of the installed physical memory.
 */
// =================================================================================

// utils includes
#include "utils/physical_memory.hpp"

// std includes
#include <cstdint>

// OS-specific headers; windows.h stays lean and defines no min/max macros
#if defined(_WIN32)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#elif defined(__APPLE__)
    #include <sys/sysctl.h>
#else // POSIX: Linux and the BSDs
    #include <unistd.h>
#endif

// =================================================================================

namespace trex::utils::physical_memory {

// =================================================================================

std::size_t physicalMemoryBytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return 0;
    return static_cast<std::size_t>(status.ullTotalPhys);
#elif defined(__APPLE__)
    std::uint64_t bytes = 0;
    std::size_t len = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &len, nullptr, 0) != 0) return 0;
    return static_cast<std::size_t>(bytes);
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages <= 0 || page_size <= 0) return 0;
    return static_cast<std::size_t>(pages) * static_cast<std::size_t>(page_size);
#endif
}

// =================================================================================
} /* End of namespace trex::utils::physical_memory */
// =================================================================================
