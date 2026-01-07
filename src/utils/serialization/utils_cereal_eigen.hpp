// ===================================================================================
// utils_cereal_eigen.hpp
// ===================================================================================
/**
 * @file utils_cereal_eigen.hpp
 *
 * @brief CEREAL serialization support for Eigen matrices and sparse matrices by
 *        augmenting the prime Eigen namespace.
 *
 */

// ===================================================================================

#ifndef TREX_UTILS_CEREAL_EIGEN_SERIALIZATION_HPP
#define TREX_UTILS_CEREAL_EIGEN_SERIALIZATION_HPP

// ===================================================================================

#include <string>

#include <Eigen/Dense>
#include <Eigen/Sparse>

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

/** @brief Augment Eigen namespace for CEREAL serialization support */

/** @brief Dense Matrix/Vector support */

/**
 * @brief Save an Eigen dense matrix to a CEREAL archive
 *
 * @tparam Archive Serial CEREAL archive type
 * @tparam Scalar Element type of the Eigen matrix
 * @tparam Rows Number of rows in the Eigen matrix
 * @tparam Cols Number of columns in the Eigen matrix
 *
 * @param ar CEREAL archive
 * @param m Eigen matrix to be saved
 */
template <class Archive, class Scalar, int Rows, int Cols>
void save(Archive &ar, const Eigen::Matrix<Scalar, Rows, Cols> &m) {
    static_assert(std::is_arithmetic<Scalar>::value,
                 "Scalar must be an arithmetic type");
    std::size_t rows = m.rows();
    std::size_t cols = m.cols();
    ar(rows, cols);
    ar(cereal::binary_data(m.data(), rows * cols * sizeof(Scalar)));
}


/**
 * @brief Load an Eigen dense matrix from a CEREAL archive
 *
 * @tparam Archive Serial CEREAL archive type
 * @tparam Scalar Element type of the Eigen matrix
 * @tparam Rows Number of rows in the Eigen matrix
 * @tparam Cols Number of columns in the Eigen matrix
 *
 * @param ar CEREAL archive
 * @param m Eigen matrix to be loaded
 */
template <class Archive, class Scalar, int Rows, int Cols>
void load(Archive &ar, Eigen::Matrix<Scalar, Rows, Cols> &m) {
    static_assert(std::is_arithmetic<Scalar>::value,
                 "Scalar must be an arithmetic type");
    std::size_t rows, cols;
    ar(rows, cols);
    m.resize(rows, cols);
    ar(cereal::binary_data(m.data(), rows * cols * sizeof(Scalar)));
}


/** @brief Sparse Matrix support */

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
    for (std::size_t k = 0; k < sm.outerSize(); ++k) {
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
        std::size_t row, col;
        Scalar value;
        ar(row, col, value);
        triplets.emplace_back(row, col, value);
    }
    sm.setFromTriplets(triplets.begin(), triplets.end());
}

// ===================================================================================

} /* End of namespace Eigen */

#endif /* End of TREX_UTILS_CEREAL_EIGEN_SERIALIZATION_HPP */
