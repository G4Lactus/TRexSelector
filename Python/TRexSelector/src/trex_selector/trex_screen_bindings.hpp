// =====================================================================================
// trex_screen_bindings.hpp - Pybind11 bindings for ScreenTRexSelector
// =====================================================================================
#ifndef TREX_PYTHON_TREX_SCREEN_BINDINGS_HPP
#define TREX_PYTHON_TREX_SCREEN_BINDINGS_HPP
// =====================================================================================
/**
 * @file trex_screen_bindings.hpp
 *
 * @brief Pybind11 bindings for ScreenTRexSelector.
 *
 */
// =====================================================================================

// pybind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

// trex includes
#include <trex_selector_methods/trex_screening/trex_screening.hpp>
#include "trex_core_bindings.hpp"

// =====================================================================================
namespace py = pybind11;
using namespace trex::trex_selector_methods::trex_screening;
using namespace trex::trex_selector_methods::trex_core;

// =====================================================================================

/**
 * @brief Python wrapper class for ScreenTRexSelector leveraging zero-copy references.
 */
class PyScreenTRexSelector : public PyTRexSelector {
public:
    /**
     * @brief Zero-copy constructor for the PyScreenTRexSelector wrapper.
     *
     * @param X Reference to the design matrix (n × p).
     * @param y Reference to the response vector (n × 1).
     * @param screen_control Configuration parameters specific to the Screening TRex method.
     * @param trex_control General configuration parameters for the TRex algorithm.
     * @param seed Random seed for reproducible computations.
     * @param verbose If true, prompts standard out messaging during select().
     */
    PyScreenTRexSelector(
        Eigen::Ref<Eigen::MatrixXd> X,
        Eigen::Ref<Eigen::VectorXd> y,
        const ScreenTRexControlParameter& screen_control,
        const TRexControlParameter& trex_control,
        int seed,
        bool verbose
    ) {
        this->X_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X.data(),
                                                                  X.rows(),
                                                                  X.cols());
        this->y_map_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(y.data(),
                                                                  y.size());

        this->selector_ = std::make_unique<ScreenTRexSelector>(
            *(this->X_map_), *(this->y_map_), screen_control, trex_control, seed, verbose
        );
    }

    /**
     * @brief Get the complete Screening selection result.
     */
    const ScreenTRexSelectionResult& getScreenResult() const {
        return static_cast<ScreenTRexSelector*>(this->selector_.get())->getScreenResult();
    }
};


/**
 * @brief Binds the ScreenTRexSelector and related configurations.
 *
 * @param m The Python module to bind against.
 */
inline void bind_trex_screen(py::module& m) {
    py::enum_<ScreenTRexMethod>(m, "ScreenTRexMethod", "Screening TRex baseline variants.")
        .value("TREX", ScreenTRexMethod::TREX, "Vanilla TRex Selector baseline.")
        .value("TREX_DA_AR1", ScreenTRexMethod::TREX_DA_AR1, "Data Augmentation with AR1 bounds.")
        .value("TREX_DA_EQUI", ScreenTRexMethod::TREX_DA_EQUI, "Data Augmentation with Equicorrelated matrices.")
        .value("TREX_DA_BLOCK_EQUI", ScreenTRexMethod::TREX_DA_BLOCK_EQUI, "Data Augmentation via Block-Equicorrelations.")
        .export_values();

    py::class_<ScreenTRexControlParameter>(m, "ScreenTRexControlParameter", "Configuration parameters for Screen metrics.")
        .def(py::init<>())
        .def_readwrite("use_bootstrap_CI", &ScreenTRexControlParameter::use_bootstrap_CI, "Whether to strictly employ bootstrapping.")
        .def_readwrite("R_boot", &ScreenTRexControlParameter::R_boot, "Bootstrap iteration counts.")
        .def_readwrite("ci_grid_step", &ScreenTRexControlParameter::ci_grid_step, "Step size utilized when gridding Confidence Intervals.")
        .def_readwrite("trex_method", &ScreenTRexControlParameter::trex_method, "Core method wrapper invoked for evaluation.")
        .def_readwrite("rho_thr_DA", &ScreenTRexControlParameter::rho_thr_DA, "Correlation bounds strictly passed into inner DA steps.")
        .def_readwrite("cor_coef", &ScreenTRexControlParameter::cor_coef, "Coefficient utilized defining underlying structure limits.");

    py::class_<ScreenTRexSelectionResult, TRexSelector::SelectionResult>(m, "ScreenTRexSelectionResult", "Extended SelectionResult holding specific Screen outputs.")
        .def_readonly("estimated_FDR", &ScreenTRexSelectionResult::estimated_FDR, "Derived total FDR bound estimation mapped.")
        .def_readonly("used_bootstrap", &ScreenTRexSelectionResult::used_bootstrap, "Indicates if bootstrap sampling was engaged.")
        .def_readonly("confidence_level", &ScreenTRexSelectionResult::confidence_level, "Bound evaluation level achieved correctly.");

    py::class_<PyScreenTRexSelector, PyTRexSelector>(m, "ScreenTRexSelector", "TRex Selector with fast variable Screening heuristics.")
        .def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::VectorXd>, const ScreenTRexControlParameter&, const TRexControlParameter&, int, bool>(),
             py::arg("X"), py::arg("y"),
             py::arg("screen_control") = ScreenTRexControlParameter(),
             py::arg("trex_control") = TRexControlParameter(),
             py::arg("seed") = -1, py::arg("verbose") = true,
             "Mapping engine directly utilizing Python NumPy states strictly via zero-copy paths.")
        .def("getScreenResult", &PyScreenTRexSelector::getScreenResult, "Fetch the detailed Screen selection metric results.");
}

#endif // TREX_PYTHON_TREX_SCREEN_BINDINGS_HPP
