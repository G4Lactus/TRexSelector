#ifndef UTILS_CEREAL_ARMA_SERIALIZATION_HPP
#define UTILS_CEREAL_ARMA_SERIALIZATION_HPP

#include <armadillo>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/list.hpp>

namespace arma {

    // Dense Matrix/Vector support
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

    template<class Archive, class eT>
    void load(Archive &ar, arma::Mat<eT>& m) {
        static_assert(std::is_arithmetic<eT>::value,
                      "eT must be an arithmetic type");
        arma::uword n_rows{}, n_cols{};
        ar(n_rows, n_cols);
        m.set_size(n_rows, n_cols);
        ar(cereal::binary_data(m.memptr(), n_rows * n_cols * sizeof(eT)));
    }

    // Dense Vector (rowvec/colvec)
    // Save/load for column vectors
    template <class Archive, class eT>
    void save(Archive &ar, const arma::Col<eT>& v) {
        save(ar, static_cast<const arma::Mat<eT>&>(v));
    }
    template<class Archive, class eT>
    void load(Archive &ar, arma::Col<eT> &v) {
        load(ar, static_cast<arma::Mat<eT>&>(v));
    }
    // Save/load for row vectors
    template<class Archive, class eT>
    void save(Archive &ar, const arma::Row<eT> &v) {
        save(ar, static_cast<const arma::Mat<eT>&>(v));
    }
    template<class Archive, class eT>
    void load(Archive &ar, arma::Row<eT> &v) {
         load(ar, static_cast<arma::Mat<eT>&>(v));
    }

    // Sparse Matrix Support for (de-)serialization
    template<class Archive>
    void save(Archive &ar, const arma::sp_mat &sm) {
        ar(sm.n_rows, sm.n_cols, sm.n_nonzero);
        for(auto it = sm.begin(); it != sm.end(); ++it)
            ar(it.row(), it.col(), *it);
    }
    template<class Archive>
    void load(Archive &ar, arma::sp_mat &sm) {
        arma::uword n_rows, n_cols, n_nonzero;
        ar(n_rows, n_cols, n_nonzero);
        sm.zeros(n_rows, n_cols);
        for(arma::uword i = 0; i < n_nonzero; ++i) {
            arma::uword row, col;
            double val;
            ar(row, col, val);
            sm(row, col) = val;
        }
    }
}

#endif
