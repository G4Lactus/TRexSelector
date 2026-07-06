// ===================================================================================
// utils_cereal_eigen.hpp
// ===================================================================================
#ifndef UTILS_CEREAL_EIGEN_SERIALIZATION_HPP
#define UTILS_CEREAL_EIGEN_SERIALIZATION_HPP
// ===================================================================================
/**
 * @file utils_cereal_eigen.hpp
 *
 * @brief Augment Eigen namespace for CEREAL serialization support for dense and
 * sparse matrices.
 */
// ===================================================================================

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Sparse>

// CEREAL includes
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/list.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

// ===================================================================================
// Augment Eigen namespace for CEREAL serialization support
// ===================================================================================
namespace Eigen {

// ===================================================================================

// Dense Matrix/Vector support
// -----------------------------------------------------------------------------

/**
 * @brief Save an Eigen dense matrix to a CEREAL archive.
 *
 * @tparam Archive Serial CEREAL archive type.
 * @tparam Derived Eigen matrix derived type (captures RowMajor/ColMajor/Sizes).
 *
 * @param ar CEREAL archive.
 * @param m Eigen matrix to be saved.
 */
template <class Archive, class Derived>
void save(Archive &ar, const Eigen::PlainObjectBase<Derived>& m) {
    using Scalar = typename Derived::Scalar;
    static_assert(std::is_arithmetic<Scalar>::value,
                 "Scalar must be an arithmetic type");
    std::size_t rows = m.rows();
    std::size_t cols = m.cols();
    ar(rows, cols);
    ar(cereal::binary_data(m.data(), rows * cols * sizeof(Scalar)));
}


/**
 * @brief Load an Eigen dense matrix/vector from a CEREAL archive.
 *
 * @tparam Archive Serial CEREAL archive type.
 * @tparam Derived Eigen matrix derived type (captures RowMajor/ColMajor/Sizes).
 *
 * @param ar CEREAL archive.
 * @param m Eigen matrix to be loaded.
 */
template <class Archive, class Derived>
void load(Archive &ar, Eigen::PlainObjectBase<Derived>& m) {
    using Scalar = typename Derived::Scalar;
    static_assert(std::is_arithmetic<Scalar>::value,
                 "Scalar must be an arithmetic type");
    std::size_t rows, cols;
    ar(rows, cols);
    m.resize(rows, cols);
    ar(cereal::binary_data(m.data(), rows * cols * sizeof(Scalar)));
}


// Sparse Matrix/Vector support
// -----------------------------------------------------------------------------

/**
 * @brief Save an Eigen sparse matrix to a CEREAL archive
 *
 * @tparam Archive Serial CEREAL archive type
 * @tparam Scalar Element type of the Eigen sparse matrix
 *
 * @param ar CEREAL archive
 * @param sm Eigen sparse matrix to be saved
 */
template <class Archive, class Scalar>
void save(Archive &ar, const Eigen::SparseMatrix<Scalar> &sm) {
    static_assert(std::is_arithmetic<Scalar>::value,
                  "Scalar must be an arithmetic type");
    std::size_t rows = sm.rows();
    std::size_t cols = sm.cols();
    std::size_t nonZeros = sm.nonZeros();
    ar(rows, cols, nonZeros);
    for (Eigen::Index k = 0; k < sm.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(sm, k);
             it; ++it) {
             ar(it.row(), it.col(), it.value());
        }
    }
}

/**
 * @brief Load an Eigen sparse matrix from a CEREAL archive
 *
 * @tparam Archive Serial CEREAL archive type
 * @tparam Scalar Element type of the Eigen sparse matrix
 *
 * @param ar CEREAL archive
 * @param sm Eigen sparse matrix to be loaded
 */
template <class Archive, class Scalar>
void load(Archive &ar, Eigen::SparseMatrix<Scalar> &sm) {
    static_assert(std::is_arithmetic<Scalar>::value,
                  "Scalar must be an arithmetic type");
    std::size_t rows, cols, nonZeros;
    ar(rows, cols, nonZeros);
    sm.resize(rows, cols);
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(nonZeros);
    for (std::size_t i = 0; i < nonZeros; ++i) {
        // Entry indices are written as Eigen::Index by save(); read the same
        // type so the archive layout matches exactly.
        Eigen::Index row, col;
        Scalar value;
        ar(row, col, value);
        triplets.emplace_back(row, col, value);
    }
    sm.setFromTriplets(triplets.begin(), triplets.end());
}

// ===================================================================================

} /* End of namespace Eigen */

#endif /* UTILS_CEREAL_EIGEN_SERIALIZATION_HPP */
