// ============================================================================
// memory_bytes.hpp
// ============================================================================
#ifndef TSOLVERS_AD_SOLVER_UTILS_MEMORY_BYTES_HPP
#define TSOLVERS_AD_SOLVER_UTILS_MEMORY_BYTES_HPP
// ============================================================================
/**
 * @file memory_bytes.hpp
 * @brief Heap bytes of the containers a T-solver holds, the parts of
 *        TSolver_Base::memoryBytes(): vectors by capacity, dense Eigen objects
 *        by size, sparse matrices by their arrays, a hash map as an estimate.
 */
// ============================================================================

// std includes
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/SparseCore>

// utils includes
#include <utils/checked_arith.hpp>

// ============================================================================

namespace trex::tsolvers::ad::solver_utils::memory_bytes {

/** @brief rows * cols * elem_size through checked_bytes. An allocated object
 *  cannot overflow the product (its allocation bounded it), so the saturated
 *  fallback is unreachable; it keeps the callers noexcept. */
inline std::size_t heapProduct(std::size_t rows, std::size_t cols,
                               std::size_t elem_size) noexcept {
    try {
        return ::trex::utils::checked_arith::checked_bytes(rows, cols, elem_size,
                                                           "memoryBytes");
    } catch (...) {
        return std::numeric_limits<std::size_t>::max();
    }
}

/** @brief Heap bytes of a std::string: capacity() + 1 when the buffer lies
 *  outside the object (small strings live inside it and count 0). */
inline std::size_t bytesOf(const std::string& s) noexcept {
    const auto object = reinterpret_cast<std::uintptr_t>(&s);
    const auto buffer = reinterpret_cast<std::uintptr_t>(s.data());
    const bool inside = buffer >= object && buffer < object + sizeof(std::string);
    return inside ? 0 : s.capacity() + 1;
}

/** @brief Heap bytes of a dense Eigen matrix, vector or array: size()
 *  coefficients, 0 for fixed-size storage. */
template <class Derived>
std::size_t bytesOf(const Eigen::PlainObjectBase<Derived>& m) noexcept {
    if constexpr (Derived::MaxSizeAtCompileTime != Eigen::Dynamic) {
        return 0;
    } else {
        return heapProduct(static_cast<std::size_t>(m.rows()),
                           static_cast<std::size_t>(m.cols()),
                           sizeof(typename Derived::Scalar));
    }
}

/** @brief Heap bytes of an Eigen sparse matrix: the outer index (outerSize()
 *  + 1 entries, also for an empty matrix), the nonzero counts of an
 *  uncompressed one, and the allocated value and inner index arrays. */
template <class Scalar, int Options, class StorageIndex>
std::size_t bytesOf(const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& K) noexcept {
    const auto outer = static_cast<std::size_t>(K.outerSize());
    std::size_t bytes = heapProduct(outer + 1, 1, sizeof(StorageIndex));
    if (!K.isCompressed()) { bytes += heapProduct(outer, 1, sizeof(StorageIndex)); }
    bytes += heapProduct(static_cast<std::size_t>(K.data().allocatedSize()), 1,
                         sizeof(Scalar) + sizeof(StorageIndex));
    return bytes;
}

/** @brief Estimated heap bytes of a node-based hash map: one pointer per
 *  bucket and a node of two words plus the entry per element (libc++ and
 *  MSVC nodes; libstdc++ saves one word). A single bucket lives in the
 *  object (libstdc++); libc++ never has exactly one. */
template <class K, class V, class H, class E, class A>
std::size_t bytesOf(const std::unordered_map<K, V, H, E, A>& m) noexcept {
    const std::size_t buckets =
        m.bucket_count() > 1 ? heapProduct(m.bucket_count(), 1, sizeof(void*)) : 0;
    const std::size_t node = 2 * sizeof(void*) + sizeof(std::pair<const K, V>);
    return buckets + heapProduct(m.size(), 1, node);
}

/** @brief Heap bytes of a std::vector: capacity() elements plus, per stored
 *  element, what @p element_bytes returns for it. */
template <class T, class A, class ElementBytes>
std::size_t bytesOf(const std::vector<T, A>& v, ElementBytes element_bytes) noexcept {
    std::size_t bytes = heapProduct(v.capacity(), 1, sizeof(T));
    for (const T& e : v) { bytes += element_bytes(e); }
    return bytes;
}

/** @brief Heap bytes of a std::vector: capacity() elements, plus the heap of
 *  each element for element types with an overload here (vectors, strings,
 *  dense Eigen objects). */
template <class T, class A>
std::size_t bytesOf(const std::vector<T, A>& v) noexcept {
    static_assert(!std::is_same_v<T, bool>, "std::vector<bool> stores bits");
    if constexpr (std::is_trivially_copyable_v<T>) {
        return heapProduct(v.capacity(), 1, sizeof(T));
    } else {
        return bytesOf(v, [](const T& e) noexcept { return bytesOf(e); });
    }
}

// ============================================================================
} /* End of namespace trex::tsolvers::ad::solver_utils::memory_bytes */

#endif /* TSOLVERS_AD_SOLVER_UTILS_MEMORY_BYTES_HPP */
