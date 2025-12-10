#ifndef UTILS_CEREAL_EIGEN_SERIALIZATION_HPP
#define UTILS_CEREAL_EIGEN_SERIALIZATION_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/list.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>


namespace Eigen {

// Dense Matrix/Vector support
template <class Archive, class Scalar, int Rows, int Cols>
void save(Archive &ar, const Eigen::Matrix<Scalar, Rows, Cols> &m) {
    static_assert(std::is_arithmetic<Scalar>::value,
                 "Scalar must be an arithmetic type");
    std::size_t rows = m.rows();
    std::size_t cols = m.cols();
    ar(rows, cols);
    ar(cereal::binary_data(m.data(), rows * cols * sizeof(Scalar)));
}

template <class Archive, class Scalar, int Rows, int Cols>
void load(Archive &ar, Eigen::Matrix<Scalar, Rows, Cols> &m) {
    static_assert(std::is_arithmetic<Scalar>::value,
                 "Scalar must be an arithmetic type");
    std::size_t rows, cols;
    ar(rows, cols);
    m.resize(rows, cols);
    ar(cereal::binary_data(m.data(), rows * cols * sizeof(Scalar)));
}

// Sparse Matrix support
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

} /* End of augmented namespace Eigen */

#endif
