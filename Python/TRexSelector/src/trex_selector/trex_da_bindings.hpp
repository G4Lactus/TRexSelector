// =====================================================================================
// trex_da_bindings.hpp - Pybind11 bindings for TRexDASelector
// =====================================================================================
#ifndef TREX_PYTHON_TREX_DA_BINDINGS_HPP
#define TREX_PYTHON_TREX_DA_BINDINGS_HPP
// =====================================================================================
/**
 * @file trex_da_bindings.hpp
 * @brief Pybind11 bindings for TRexDASelector.
 *
 */
// =====================================================================================

// pybind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

// trex includes
#include <trex_selector_methods/trex_da/trex_da.hpp>
#include "trex_core_bindings.hpp"

// =====================================================================================

namespace py = pybind11;
using namespace trex::trex_selector_methods::trex_da;
using namespace trex::trex_selector_methods::trex_core;

// =====================================================================================

/**
 * @brief Python wrapper class for TRexDASelector leveraging zero-copy references.
 */
class PyTRexDASelector : public PyTRexSelector {
public:
    /**
     * @brief Zero-copy constructor for the PyTRexDASelector wrapper.
     *
     * @param X Reference to the design matrix (n × p).
     * @param y Reference to the response vector (n × 1).
     * @param tFDR Target False Discovery Rate threshold.
     * @param da_control Configuration parameters specific to the Data Augmentation method.
     * @param trex_control General configuration parameters for the TRex algorithm.
     * @param seed Random seed for reproducible computations.
     * @param verbose If true, prompts standard out messaging during select().
     */
    PyTRexDASelector(
        Eigen::Ref<Eigen::MatrixXd> X,
        Eigen::Ref<Eigen::VectorXd> y,
        double tFDR,
        const TRexDAControlParameter& da_control,
        const TRexControlParameter& trex_control,
        int seed,
        bool verbose
    ) {
        this->X_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X.data(),
                                                                  X.rows(),
                                                                  X.cols());
        this->y_map_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(y.data(),
                                                                  y.size());

        // The refactored C++ constructor takes a single DA control struct that
        // nests the base algorithmic parameters as `trex_ctrl`. Merge the
        // separately-supplied `trex_control` into that nested field.
        TRexDAControlParameter da_ctrl = da_control;
        da_ctrl.trex_ctrl = trex_control;

        this->selector_ = std::make_unique<TRexDASelector>(
            *(this->X_map_), *(this->y_map_), tFDR, da_ctrl, seed, verbose
        );
    }

    /**
     * @brief Get the complete DA selection result.
     */
    const TRexDASelector::DASelectionResult& getDAResult() const {
        return static_cast<TRexDASelector*>(this->selector_.get())->getDAResult();
    }
};


/**
 * @brief Binds the TRexDASelector and related configurations.
 *
 * @param m The Python module to bind against.
 */
inline void bind_trex_da(py::module& m) {
    py::enum_<DAMethod>(m, "DAMethod", "Data Augmentation methods.")
        .value("AR1", DAMethod::AR1, "Autoregressive order 1.")
        .value("EQUI", DAMethod::EQUI, "Equicorrelated structure.")
        .value("BT", DAMethod::BT, "Block-Toeplitz structure.")
        .value("NN", DAMethod::NN, "Nearest-Neighbor approach.")
        .value("PRIOR_GROUPS", DAMethod::PRIOR_GROUPS, "Prior group correlation structures.")
        .export_values();

    py::enum_<BTSelectionMode>(m, "BTSelectionMode", "Cell-selection policy for the BT (binary-tree) DA method.")
        .value("FeasibleOnly", BTSelectionMode::FeasibleOnly, "Only FDP-feasible cells may be selected (default; controls realized FDR).")
        .value("RFaithful",    BTSelectionMode::RFaithful,    "Reproduces the R reference; may inflate FDR at high within-group correlation.")
        .export_values();

    py::class_<TRexDAControlParameter>(m, "TRexDAControlParameter", "Control parameters for TRexDASelector.")
        .def(py::init<>())
        .def_readwrite("method", &TRexDAControlParameter::method, "The data augmentation method to use.")
        .def_readwrite("cor_coef", &TRexDAControlParameter::cor_coef, "Correlation coefficient parameter.")
        .def_readwrite("rho_thr_DA", &TRexDAControlParameter::rho_thr_DA, "Correlation threshold for DA.")
        .def_readwrite("hc_linkage", &TRexDAControlParameter::hc_linkage, "Hierarchical clustering linkage method.")
        .def_readwrite("hc_grid_length", &TRexDAControlParameter::hc_grid_length, "Grid length for hierarchical clustering.")
        .def_readwrite("bt_selection_mode", &TRexDAControlParameter::bt_selection_mode, "Cell-selection policy for the BT method (FeasibleOnly or RFaithful).")
        .def_readwrite("prior_groups", &TRexDAControlParameter::prior_groups, "Prior groupings provided to the selector.")
        .def_readwrite("rho_grid_labels", &TRexDAControlParameter::rho_grid_labels, "Labels for the correlation grid.")
        .def_readwrite("trex_ctrl", &TRexDAControlParameter::trex_ctrl, "Nested base T-Rex algorithmic control parameters (overridden by the separate trex_control argument to TRexDASelector).");

    py::class_<TRexDASelector::DASelectionResult, TRexSelector::SelectionResult>(m, "DASelectionResult", "Extended SelectionResult holding specific DA outputs.")
        .def_readonly("rho_thresh", &TRexDASelector::DASelectionResult::rho_thresh, "The threshold applied for rho.")
        .def_readonly("rho_grid", &TRexDASelector::DASelectionResult::rho_grid, "Grid values used for rho.")
        .def_readonly("method",           &TRexDASelector::DASelectionResult::method,           "The resulting DA method applied.")
        .def_readonly("cor_coef",          &TRexDASelector::DASelectionResult::cor_coef,          "Auto-estimated or supplied correlation coefficient.")
        .def_readonly("fdp_hat_array_bt",  &TRexDASelector::DASelectionResult::FDP_hat_array_BT,  "Per-rho-level FDP estimate matrices (BT method only).")
        .def_readonly("phi_array_bt",      &TRexDASelector::DASelectionResult::Phi_array_BT,      "Per-rho-level deflated Phi matrices (BT method only).")
        .def_readonly("r_array_bt",        &TRexDASelector::DASelectionResult::R_array_BT,        "Per-rho-level R matrices (BT method only).");

    py::class_<PyTRexDASelector, PyTRexSelector>(m, "TRexDASelector", "TRex Selector with Data Augmentation support.")
        .def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::VectorXd>, double, const TRexDAControlParameter&, const TRexControlParameter&, int, bool>(),
             py::arg("X"), py::arg("y"), py::arg("tFDR") = 0.1,
             py::arg("da_control") = TRexDAControlParameter(),
             py::arg("trex_control") = TRexControlParameter(),
             py::arg("seed") = -1, py::arg("verbose") = true,
             "Construct the DA selector. X and y are accessed zero-copy via Eigen::Map.")
        .def("getDAResult", &PyTRexDASelector::getDAResult, "Fetch the detailed DA selection result.");
}
// =====================================================================================
#endif /* End of TREX_PYTHON_TREX_DA_BINDINGS_HPP */
