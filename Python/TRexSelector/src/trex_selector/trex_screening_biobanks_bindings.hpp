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
    py::class_<BiobankScreenTRexControl>(m, "BiobankScreenTRexControl", "Configurations orchestrating the Biobank bounds execution.")
        .def(py::init<>())
        .def_readwrite("target_FDR_trex", &BiobankScreenTRexControl::target_FDR_trex, "Initial FDR constraint strictly loaded.")
        .def_readwrite("lower_bound_FDR", &BiobankScreenTRexControl::lower_bound_FDR, "Lowest scaling FDR factor mapping boundaries.")
        .def_readwrite("upper_bound_FDR", &BiobankScreenTRexControl::upper_bound_FDR, "Maximum bounded target bounds allowed natively.");

    py::class_<BiobankScreenTRexResult>(m, "BiobankScreenTRexResult", "Diagnostic statistics bounds mapping phenotype solutions.")
        .def_readonly("phenotype_index", &BiobankScreenTRexResult::phenotype_index, "Numerical mapping matching specific target column bound ranges.")
        .def_readonly("selected_indices", &BiobankScreenTRexResult::selected_indices, "Vector output mapping original unfiltered arrays correctly.")
        .def_readonly("estimated_FDR", &BiobankScreenTRexResult::estimated_FDR, "Explicit mapped estimation values evaluated against boundary limits.")
        .def_readonly("method_used", &BiobankScreenTRexResult::method_used, "Fallback bound configuration indication algorithm string.")
        .def_readonly("estimated_FDR_screen_ordinary", &BiobankScreenTRexResult::estimated_FDR_screen_ordinary, "Base computation FDR bound unweighted.")
        .def_readonly("estimated_FDR_screen_bootstrap", &BiobankScreenTRexResult::estimated_FDR_screen_bootstrap, "Resampled approximation mapping distributions natively.")
        .def_readonly("selected_indices_screen_ordinary", &BiobankScreenTRexResult::selected_indices_screen_ordinary, "Unweighted selected indices mapping elements internally.")
        .def_readonly("selected_indices_screen_bootstrap", &BiobankScreenTRexResult::selected_indices_screen_bootstrap, "Bounded inclusion distributions properly restricted.")
        .def_readonly("used_fallback_trex", &BiobankScreenTRexResult::used_fallback_trex, "Indicator highlighting basic fallback methods executing tightly.");

    py::class_<PyBiobankScreenTRex>(m, "BiobankScreenTRex", "Highly scalable algorithm handling iterative massive trait distributions tightly.")
        .def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::VectorXd>, const BiobankScreenTRexControl&, int, bool>(),
             py::arg("X"), py::arg("y"),
             py::arg("biosctrex_ctrl") = BiobankScreenTRexControl(),
             py::arg("seed") = -1, py::arg("verbose") = false,
             "Single phenotype initialization engine directly utilizing Python NumPy states strictly via zero-copy paths.")
        .def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>, const BiobankScreenTRexControl&, int, bool>(),
             py::arg("X"), py::arg("Y"),
             py::arg("biosctrex_ctrl") = BiobankScreenTRexControl(),
             py::arg("seed") = -1, py::arg("verbose") = false,
             "Multiple phenotype 2D initialization utilizing pure matrix representations handling mappings natively.")
        .def("screenPhenotype", &PyBiobankScreenTRex::screenPhenotype, py::call_guard<py::gil_scoped_release>(), "Computes selections explicitly over the distinct variable subset.")
        .def("screenPhenotypes", &PyBiobankScreenTRex::screenPhenotypes, py::call_guard<py::gil_scoped_release>(), "Iteratively traverses and determines bounds properly across 2D distributions.");

// =====================================================================================
}
// =====================================================================================
#endif /* End of TREX_PYTHON_TREX_SCREENING_BIOBANKS_BINDINGS_HPP */
