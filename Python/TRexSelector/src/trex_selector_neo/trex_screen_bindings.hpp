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

        // The refactored C++ constructor takes a single screening control
        // struct that nests the base algorithmic parameters as `trex_ctrl`.
        // Merge the separately-supplied `trex_control` into that nested field.
        ScreenTRexControlParameter screen_ctrl = screen_control;
        screen_ctrl.trex_ctrl = trex_control;

        this->selector_ = std::make_unique<ScreenTRexSelector>(
            *(this->X_map_), *(this->y_map_), screen_ctrl, seed, verbose
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
        .value("TREX", ScreenTRexMethod::TREX, "Standard T-Rex screening without data augmentation.")
        .value("TREX_DA_AR1", ScreenTRexMethod::TREX_DA_AR1, "T-Rex screening with AR(1) data augmentation.")
        .value("TREX_DA_EQUI", ScreenTRexMethod::TREX_DA_EQUI, "T-Rex screening with equicorrelated data augmentation.")
        .value("TREX_DA_BLOCK_EQUI", ScreenTRexMethod::TREX_DA_BLOCK_EQUI, "T-Rex screening with block-equicorrelated data augmentation.")
        .export_values();

    py::class_<ScreenTRexControlParameter>(m, "ScreenTRexControlParameter", "Control parameters for TRexScreeningSelector.")
        .def(py::init<>())
        .def_readwrite("use_bootstrap_CI", &ScreenTRexControlParameter::use_bootstrap_CI, "Use bootstrap confidence intervals for FDR estimation (default false).")
        .def_readwrite("R_boot", &ScreenTRexControlParameter::R_boot, "Number of bootstrap replicates (default 1000).")
        .def_readwrite("ci_grid_step", &ScreenTRexControlParameter::ci_grid_step, "Step size for the confidence-interval grid.")
        .def_readwrite("trex_method", &ScreenTRexControlParameter::trex_method, "Screening method variant to use (see ScreenTRexMethod).")
        .def_readwrite("rho_thr_DA", &ScreenTRexControlParameter::rho_thr_DA, "Correlation threshold passed to the inner DA step. Only applies for TREX_DA_AR1 and TREX_DA_EQUI.")
        .def_readwrite("cor_coef",   &ScreenTRexControlParameter::cor_coef,   "Fixed correlation coefficient for AR1 or equicorrelated structure. Estimated automatically if <= -1. Only applies for TREX_DA_AR1 and TREX_DA_EQUI.")
        .def_readwrite("n_blocks",   &ScreenTRexControlParameter::n_blocks,   "Number of equicorrelated blocks. Only applies for TREX_DA_BLOCK_EQUI.")
        .def_readwrite("trex_ctrl",  &ScreenTRexControlParameter::trex_ctrl,  "Nested base T-Rex algorithmic control parameters (overridden by the separate trex_control argument to TRexScreeningSelector).");

    py::class_<ScreenTRexSelectionResult, TRexSelector::SelectionResult>(m, "ScreenTRexSelectionResult", "SelectionResult extended with screening-specific fields.")
        .def_readonly("estimated_FDR", &ScreenTRexSelectionResult::estimated_FDR, "Estimated false discovery rate.")
        .def_readonly("used_bootstrap", &ScreenTRexSelectionResult::used_bootstrap, "Whether bootstrap CI was used for FDR estimation.")
        .def_readonly("confidence_level",      &ScreenTRexSelectionResult::confidence_level,      "Bootstrap confidence level achieved.")
        .def_readonly("beta_mat",               &ScreenTRexSelectionResult::beta_mat,               "Coefficient matrix averaged over K experiments (p x T_stop).")
        .def_readonly("dummy_betas",            &ScreenTRexSelectionResult::dummy_betas,            "Non-zero dummy coefficients collected during screening.")
        .def_readonly("estimated_correlation",  &ScreenTRexSelectionResult::estimated_correlation,  "Estimated correlation used by DA-based screening variants.");

    py::class_<PyScreenTRexSelector, PyTRexSelector>(m, "TRexScreeningSelector", "T-Rex Selector with fast variable screening via dummy comparison.")
        .def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::VectorXd>, const ScreenTRexControlParameter&, const TRexControlParameter&, int, bool>(),
             py::arg("X"), py::arg("y"),
             py::arg("screen_control") = ScreenTRexControlParameter(),
             py::arg("trex_control") = TRexControlParameter(),
             py::arg("seed") = -1, py::arg("verbose") = true,
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
             "Construct the screening selector. X and y are accessed zero-copy via Eigen::Map.")
        .def("getScreenResult", &PyScreenTRexSelector::getScreenResult, "Return the ScreenTRexSelectionResult after calling select().");
}

#endif // TREX_PYTHON_TREX_SCREEN_BINDINGS_HPP
