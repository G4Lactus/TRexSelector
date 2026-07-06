// =====================================================================================
// trex_screening_biobanks_bindings.hpp - Pybind11 bindings for BiobankScreenTRex
// =====================================================================================
#ifndef TREX_PYTHON_TREX_SCREENING_BIOBANKS_BINDINGS_HPP
#define TREX_PYTHON_TREX_SCREENING_BIOBANKS_BINDINGS_HPP
// =====================================================================================
/**
 * @file trex_screening_biobanks_bindings.hpp
 * @brief Pybind11 bindings for BiobankScreenTRex.
 *
 */
// =====================================================================================

// pybind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

// trex includes
#include <trex_selector_methods/trex_screening/trex_biobank_screening.hpp>

// =====================================================================================

namespace py = pybind11;
using namespace trex::trex_selector_methods::trex_screening;
using namespace trex::trex_selector_methods::trex_biobank_screening;

// =====================================================================================

/**
 * @brief Python wrapper class for BiobankScreenTRex leveraging zero-copy references.
 */
class PyBiobankScreenTRex {
private:
    /** @brief Map for the design matrix (X) */
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>> X_map_;

    /** @brief Map for the response vector (y) when single phenotype mapping is used. */
    std::unique_ptr<Eigen::Map<Eigen::VectorXd>> y_map_1d_;

    /** @brief Map for the response matrix (Y) when multiple phenotype mapping is used. */
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>> Y_map_2d_;

    /** @brief Pointer to the underlying BiobankScreenTRex selector instance. */
    std::unique_ptr<BiobankScreenTRex> selector_;

public:

    /**
     * @brief Zero-copy constructor for single phenotype mapping.
     *
     * @param X Reference to the design matrix (n × p).
     * @param y Reference to the response vector (n × 1).
     * @param biosctrex_ctrl Configuration parameters specific to the BiobankScreenTRex method.
     * @param seed Random seed for reproducible computations.
     * @param verbose If true, prompts standard out messaging during select().
     */
    PyBiobankScreenTRex(
        Eigen::Ref<Eigen::MatrixXd> X,
        Eigen::Ref<Eigen::VectorXd> y,
        const BiobankScreenTRexControl& biosctrex_ctrl,
        int seed,
        bool verbose
    ) {
        X_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X.data(),
                                                                  X.rows(),
                                                                  X.cols());

        y_map_1d_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(y.data(),
                                                                     y.size());

        selector_ = std::make_unique<BiobankScreenTRex>(
            *X_map_, *y_map_1d_, biosctrex_ctrl, seed, verbose
        );
    }


    /**
     * @brief Zero-copy constructor for multiple phenotype mapping.
     */
    PyBiobankScreenTRex(
        Eigen::Ref<Eigen::MatrixXd> X,
        Eigen::Ref<Eigen::MatrixXd> Y,
        const BiobankScreenTRexControl& biosctrex_ctrl,
        int seed,
        bool verbose
    ) {
        X_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X.data(),
                                                                  X.rows(),
                                                                  X.cols());

        Y_map_2d_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(Y.data(),
                                                                     Y.rows(),
                                                                     Y.cols());

        selector_ = std::make_unique<BiobankScreenTRex>(
            *X_map_, *Y_map_2d_, biosctrex_ctrl, seed, verbose
        );
    }


    /**
     * @brief Execute the Biobank Screening mapping for the loaded phenotype.
     */
    BiobankScreenTRexResult screenPhenotype() {
        return selector_->screenPhenotype();
    }


    /**
     * @brief Execute multi-phenotype bounds sequence mapping.
     */
    std::vector<BiobankScreenTRexResult> screenPhenotypes() {
        return selector_->screenPhenotypes();
    }
};


/**
 * @brief Binds the BiobankScreenTRex and related configurations.
 *
 * @param m The Python module to bind against.
 */
inline void bind_trex_screening_biobanks(py::module& m) {
    py::class_<BiobankScreenTRexControl>(m, "BiobankScreenTRexControl", "Control parameters for TRexBiobankScreeningSelector.")
        .def(py::init<>())
        .def_readwrite("target_FDR_trex", &BiobankScreenTRexControl::target_FDR_trex, "Target FDR for the core T-Rex fallback (default 0.1).")
        .def_readwrite("lower_bound_FDR", &BiobankScreenTRexControl::lower_bound_FDR, "Lower bound on the FDR search range.")
        .def_readwrite("upper_bound_FDR", &BiobankScreenTRexControl::upper_bound_FDR, "Upper bound on the FDR search range.");

    py::class_<BiobankScreenTRexResult>(m, "BiobankScreenTRexResult", "Result for a single phenotype returned by screenPhenotype() or screenPhenotypes().")
        .def_readonly("phenotype_index", &BiobankScreenTRexResult::phenotype_index, "0-based column index of the screened phenotype.")
        .def_readonly("selected_indices", &BiobankScreenTRexResult::selected_indices, "0-based indices of selected predictors.")
        .def_readonly("estimated_FDR", &BiobankScreenTRexResult::estimated_FDR, "Estimated FDR of the selection.")
        .def_readonly("method_used", &BiobankScreenTRexResult::method_used, "Method used: 'screening_ordinary', 'screening_bootstrap', or 'trex'.")
        .def_readonly("estimated_FDR_screen_ordinary", &BiobankScreenTRexResult::estimated_FDR_screen_ordinary, "Estimated FDR from the ordinary (non-bootstrap) screening method.")
        .def_readonly("estimated_FDR_screen_bootstrap", &BiobankScreenTRexResult::estimated_FDR_screen_bootstrap, "Estimated FDR from the bootstrap screening method.")
        .def_readonly("selected_indices_screen_ordinary", &BiobankScreenTRexResult::selected_indices_screen_ordinary, "Selected indices from the ordinary screening method (0-based).")
        .def_readonly("selected_indices_screen_bootstrap", &BiobankScreenTRexResult::selected_indices_screen_bootstrap, "Selected indices from the bootstrap screening method (0-based).")
        .def_readonly("used_fallback_trex", &BiobankScreenTRexResult::used_fallback_trex, "True if the T-Rex fallback was used instead of screening.");

    py::class_<PyBiobankScreenTRex>(m, "TRexBiobankScreeningSelector", "Biobank-scale T-Rex selector for FDR-controlled variable selection across one or many phenotypes.")
        .def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::VectorXd>, const BiobankScreenTRexControl&, int, bool>(),
             py::arg("X"), py::arg("y"),
             py::arg("biosctrex_ctrl") = BiobankScreenTRexControl(),
             py::arg("seed") = -1, py::arg("verbose") = false,
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
             "Construct for a single phenotype (1-D y). X and y are accessed zero-copy.")
        .def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>, const BiobankScreenTRexControl&, int, bool>(),
             py::arg("X"), py::arg("Y"),
             py::arg("biosctrex_ctrl") = BiobankScreenTRexControl(),
             py::arg("seed") = -1, py::arg("verbose") = false,
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
             "Construct for multiple phenotypes (2-D Y). X and Y are accessed zero-copy.")
        .def("screenPhenotype", &PyBiobankScreenTRex::screenPhenotype, py::call_guard<py::gil_scoped_release>(), "Run screening for the single loaded phenotype.")
        .def("screenPhenotypes", &PyBiobankScreenTRex::screenPhenotypes, py::call_guard<py::gil_scoped_release>(), "Run screening for all loaded phenotypes.");

// =====================================================================================
}
// =====================================================================================
#endif /* End of TREX_PYTHON_TREX_SCREENING_BIOBANKS_BINDINGS_HPP */
