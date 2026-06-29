// =================================================================================
// fp_classify.hpp
// =================================================================================
#ifndef TREX_UTILS_FP_CLASSIFY_HPP
#define TREX_UTILS_FP_CLASSIFY_HPP
// =================================================================================
/**
 * @file fp_classify.hpp
 *
 * @brief Floating-point classification helpers that work under -ffast-math.
 *
 * @details
 *  -ffast-math implies -ffinite-math-only, which lets the compiler assume all
 *  FP values are finite. Under recent Clang versions this causes even
 *  __builtin_isnan / __builtin_isinf / __builtin_isfinite to emit
 *  -Wnan-infinity-disabled warnings (and technically constitutes UB).
 *
 *  These helpers instead inspect IEEE 754 bit patterns directly via
 *  std::memcpy — well-defined type-punning in C++20 — and use only integer
 *  arithmetic, making them completely immune to any floating-point compiler
 *  options. No <cmath> classification functions are needed.
 *
 *  IEEE 754 double layout (64-bit):
 *    bit 63      : sign
 *    bits 62..52 : exponent (11 bits)
 *    bits 51..0  : mantissa (52 bits)
 *
 *  Bit-pattern rules (magnitude = bits & 0x7FFF'FFFF'FFFF'FFFF):
 *    NaN    : magnitude  > 0x7FF0'0000'0000'0000  (exp=all-ones, mantissa≠0)
 *    ±Inf   : magnitude == 0x7FF0'0000'0000'0000  (exp=all-ones, mantissa=0)
 *    Finite : exponent field != all-ones
 */
// =================================================================================

#include <cstdint>
#include <cstring>
#include <Eigen/Dense>

// =================================================================================

namespace trex::utils::fp_classify {

// =================================================================================

namespace detail {

/** @brief Reinterpret a double as its raw uint64 bit pattern (well-defined via memcpy). */
[[nodiscard]] inline std::uint64_t to_bits(double x) noexcept {
    std::uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
}

} // namespace detail

// =================================================================================

/** @brief True if @p x is NaN. Safe under -ffast-math / -ffinite-math-only. */
[[nodiscard]] inline bool isnan(double x) noexcept {
    return (detail::to_bits(x) & 0x7FFF'FFFF'FFFF'FFFFu) > 0x7FF0'0000'0000'0000u;
}

/** @brief True if @p x is ±Inf. Safe under -ffast-math / -ffinite-math-only. */
[[nodiscard]] inline bool isinf(double x) noexcept {
    return (detail::to_bits(x) & 0x7FFF'FFFF'FFFF'FFFFu) == 0x7FF0'0000'0000'0000u;
}

/** @brief True if @p x is finite (not NaN and not ±Inf). Safe under -ffast-math. */
[[nodiscard]] inline bool isfinite(double x) noexcept {
    return (detail::to_bits(x) & 0x7FF0'0000'0000'0000u) != 0x7FF0'0000'0000'0000u;
}

/** @brief True if every element of @p v is finite. Safe under -ffast-math. */
inline bool allFinite(const Eigen::VectorXd& v) {
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        if (!isfinite(v(i))) return false;
    }
    return true;
}

// =================================================================================
} /* End of namespace trex::utils::fp_classify */

#endif /* TREX_UTILS_FP_CLASSIFY_HPP */
