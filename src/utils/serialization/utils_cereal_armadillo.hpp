// ===================================================================================
// utils_cereal_armadillo.hpp
// ===================================================================================
#ifndef UTILS_CEREAL_ARMA_SERIALIZATION_HPP
#define UTILS_CEREAL_ARMA_SERIALIZATION_HPP
// ===================================================================================
/**
 * @file utils_cereal_armadillo.hpp
 *
 * @brief CEREAL serialization support for Armadillo matrices and vectors by augmenting
 * the prime arma namespace.
 *
 */
// ===================================================================================

// std includes
#include <armadillo>

// CEREAL includes
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/list.hpp>

// ===================================================================================
// Augment arma namespace for CEREAL serialization support
// ===================================================================================
namespace arma {

// ===================================================================================

// Dense Matrix/Vector support
// -----------------------------------------------------------------------------

/**
 * @brief Save an Armadillo matrix to a CEREAL archive.
 *
 * @tparam Archive Serial CEREAL archive type.
 * @tparam eT Element type of the Armadillo matrix.
 */
template <class Archive, class eT>
void save(Archive &ar, const arma::Mat<eT>& m) {
    static_assert(std::is_arithmetic<eT>::value,
                  "eT must be an arithmetic type");
    arma::uword n_rows = m.n_rows;
    arma::uword n_cols = m.n_cols;
    ar(n_rows, n_cols);
    ar(cereal::binary_data(const_cast<eT*>(m.memptr()),
                           n_rows * n_cols * sizeof(eT)));
}


/**
 * @brief Load an Armadillo matrix from a CEREAL archive.
 *
 * @tparam Archive Serial CEREAL archive type.
 * @tparam eT Element type of the Armadillo matrix.
 */
template<class Archive, class eT>
void load(Archive &ar, arma::Mat<eT>& m) {
    static_assert(std::is_arithmetic<eT>::value,
                  "eT must be an arithmetic type");
    arma::uword n_rows{}, n_cols{};
    ar(n_rows, n_cols);
    m.set_size(n_rows, n_cols);
    ar(cereal::binary_data(m.memptr(), n_rows * n_cols * sizeof(eT)));
}


/**
 * @brief Save an Armadillo column vector to a CEREAL archive.
 *
 * @tparam Archive Serial CEREAL archive type.
 * @tparam eT Element type of the Armadillo column vector.
 */
template <class Archive, class eT>
void save(Archive &ar, const arma::Col<eT>& v) {
    save(ar, static_cast<const arma::Mat<eT>&>(v));
}


/**
 * @brief Load an Armadillo column vector from a CEREAL archive.
 *
 * @tparam Archive Serial CEREAL archive type.
 * @tparam eT Element type of the Armadillo column vector.
 *
 * @param ar CEREAL archive to load from.
 * @param v Armadillo column vector to load into.
 */
template<class Archive, class eT>
void load(Archive &ar, arma::Col<eT> &v) {
    load(ar, static_cast<arma::Mat<eT>&>(v));
}


/**
 * @brief Save an Armadillo row vector to a CEREAL archive.
 *
 * @tparam Archive Serial CEREAL archive type.
 * @tparam eT Element type of the Armadillo row vector.
 *
 * @param ar CEREAL archive to save to.
 * @param v Armadillo row vector to save.
 */
template<class Archive, class eT>
void save(Archive &ar, const arma::Row<eT> &v) {
    save(ar, static_cast<const arma::Mat<eT>&>(v));
}


/**
 * @brief Load an Armadillo row vector from a CEREAL archive.
 *
 * @tparam Archive Serial CEREAL archive type.
 * @tparam eT Element type of the Armadillo row vector.
 *
 * @param ar CEREAL archive to load from.
 * @param v Armadillo row vector to load into.
 */
template<class Archive, class eT>
void load(Archive &ar, arma::Row<eT> &v) {
    load(ar, static_cast<arma::Mat<eT>&>(v));
}


// Sparse Matrix Support for (de-)serialization
// -----------------------------------------------------------------------------

/**
 * @brief Save an Armadillo sparse matrix to a CEREAL archive.
 *
 * @tparam Archive Serial CEREAL archive type.
 * @tparam eT Element type of the Armadillo sparse matrix.
 *
 * @param ar CEREAL archive to save to.
 * @param sm Armadillo sparse matrix to save.
 */
template<class Archive, class eT>
void save(Archive &ar, const arma::SpMat<eT> &sm) {
    ar(sm.n_rows, sm.n_cols, sm.n_nonzero);
    for(auto it = sm.begin(); it != sm.end(); ++it)
        ar(it.row(), it.col(), *it);
}


/**
 * @brief Load an Armadillo sparse matrix from a CEREAL archive.
 *
 * @tparam Archive Serial CEREAL archive type.
 * @tparam eT Element type of the Armadillo sparse matrix.
 *
 * @param ar CEREAL archive to load from.
 * @param sm Armadillo sparse matrix to load into.
 */
template<class Archive, class eT>
void load(Archive &ar, arma::SpMat<eT> &sm) {
    arma::uword n_rows, n_cols, n_nonzero;
    ar(n_rows, n_cols, n_nonzero);
    sm.zeros(n_rows, n_cols);
    for(arma::uword i = 0; i < n_nonzero; ++i) {
        arma::uword row, col;
        eT val; // Automatically matches precision
        ar(row, col, val);
        sm(row, col) = val;
    }
}
// ===================================================================================

} /* End of namespace arma */

#endif /* UTILS_CEREAL_ARMA_SERIALIZATION_HPP */
