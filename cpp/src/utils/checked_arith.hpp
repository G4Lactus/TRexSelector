// =================================================================================
// checked_arith.hpp
// =================================================================================
#ifndef TREX_UTILS_CHECKED_ARITH_HPP
#define TREX_UTILS_CHECKED_ARITH_HPP
// =================================================================================
/**
 * @file checked_arith.hpp
 *
 * @brief Overflow-checked size arithmetic for the byte counts of files and
 *        memory mappings.
 *
 * @details A wrapped product (rows * cols * sizeof) silently sizes a file or a
 *  mapping too small while the matrix view still claims the full extent:
 *  every later in-bounds-looking access then faults (SIGBUS / SIGSEGV) or
 *  corrupts memory. (2^61 + 1) * 8 wraps to 8, for instance. These helpers
 *  throw std::overflow_error instead, before anything touches the disk.
 *  Portable (no compiler builtins): the division test is exact for
 *  unsigned integers.
 */
// =================================================================================

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

// =================================================================================

namespace trex::utils::checked_arith {

// =================================================================================

/**
 * @brief a * b, or std::overflow_error when the product does not fit std::size_t.
 *
 * @param context Name of the caller for the error message.
 */
inline std::size_t checked_mul(std::size_t a, std::size_t b,
                               const char* context = "checked_mul") {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::overflow_error(
            std::string(context) + ": size_t overflow in " + std::to_string(a) +
            " * " + std::to_string(b));
    }
    return a * b;
}

/**
 * @brief rows * cols * elem_size with both products checked.
 *
 * @param context Name of the caller for the error message.
 */
inline std::size_t checked_bytes(std::size_t rows, std::size_t cols,
                                 std::size_t elem_size,
                                 const char* context = "checked_bytes") {
    return checked_mul(checked_mul(rows, cols, context), elem_size, context);
}

// =================================================================================
} /* End of namespace trex::utils::checked_arith */

#endif /* TREX_UTILS_CHECKED_ARITH_HPP */
