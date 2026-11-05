// =================================================================================
// physical_memory.hpp
// =================================================================================
#ifndef TREX_UTILS_PHYSICAL_MEMORY_HPP
#define TREX_UTILS_PHYSICAL_MEMORY_HPP
// =================================================================================
/**
 * @file physical_memory.hpp
 *
 * @brief Physical memory of the machine, for buffers that must fit in RAM.
 *
 * @details The allocator does not refuse a heap buffer beyond the physical
 *  memory. Its zero fill touches page after page until the kernel kills the
 *  process. Callers compare the bytes they are about to allocate with this
 *  size up front and throw with a remedy instead.
 */
// =================================================================================

#include <cstddef>

// =================================================================================

namespace trex::utils::physical_memory {

// =================================================================================

/**
 * @brief Physical memory installed in the machine, in bytes.
 *
 * @return The size reported by the operating system, or 0 when it cannot be
 *         queried (a check cannot block a run it cannot judge).
 */
std::size_t physicalMemoryBytes();

// =================================================================================
} /* End of namespace trex::utils::physical_memory */

#endif /* TREX_UTILS_PHYSICAL_MEMORY_HPP */
