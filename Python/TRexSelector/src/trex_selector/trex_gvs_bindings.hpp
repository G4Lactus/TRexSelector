// =====================================================================================
// trex_gvs_bindings.hpp - Pybind11 bindings for TRexGVSSelector
// =====================================================================================
#ifndef TREX_PYTHON_TREX_GVS_BINDINGS_HPP
#define TREX_PYTHON_TREX_GVS_BINDINGS_HPP
// =====================================================================================
/**
 * @file trex_gvs_bindings.hpp
 * @brief Pybind11 bindings for TRexGVSSelector.
 *
 */
// =====================================================================================

// pybind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

// trex includes
#include <trex_selector_methods/trex_gvs/trex_gvs.hpp>

// =====================================================================================

namespace py = pybind11;
using namespace trex::trex_selector_methods::trex_gvs;
using namespace trex::trex_selector_methods::trex_core;

// =====================================================================================

/**
 * @brief Python wrapper class for TRexGVSSelector leveraging zero-copy references.
 */
class PyTRexGVSSelector {
private:
    /** @brief Map for the design matrix (X) */
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>> X_map_;

    /** @brief Map for the response vector (y) */
    std::unique_ptr<Eigen::Map<Eigen::VectorXd>> y_map_;

    /** @brief Pointer to the underlying TRexGVSSelector */
    std::unique_ptr<TRexGVSSelector> selector_;

public:
    /**
     * @brief Zero-copy constructor for the PyTRexGVSSelector wrapper.
     *
     * @param X Reference to the design matrix (n × p).
     * @param y Reference to the response vector (n × 1).
     * @param tFDR Target False Discovery Rate threshold.
     * @param gvs_control Configuration parameters specific to the Group Variable Selection method.
     * @param trex_control General configuration parameters for the TRex algorithm.
     * @param seed Random seed for reproducible computations.
     * @param verbose If true, prompts standard out messaging during select().
     */
    PyTRexGVSSelector(
        Eigen::Ref<Eigen::MatrixXd> X,
        Eigen::Ref<Eigen::VectorXd> y,
        double tFDR,
        const TRexGVSControlParameter& gvs_control,
        const TRexControlParameter& trex_control,
        int seed,
        bool verbose
    ) {
        X_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X.data(),
                                                                  X.rows(),
                                                                  X.cols());
        y_map_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(y.data(),
                                                                  y.size());

        selector_ = std::make_unique<TRexGVSSelector>(
            *X_map_, *y_map_, tFDR, gvs_control, trex_control, seed, verbose
        );
    }

    /**
     * @brief Execute the Group Variable Selection algorithms.
     */
    TRexSelector::SelectionResult select() {
        return selector_->select();
    }

    /**
     * @brief Get the complete GVS selection result.
     */
    const TRexGVSSelector::GVSSelectionResult& getGVSResult() const {
        return selector_->getGVSResult();
    }
};


/**
 * @brief Binds the TRexGVSSelector and related configurations.
 *
 * @param m The Python module to bind against.
 */
inline void bind_trex_gvs(py::module& m) {
    py::enum_<GVSType>(m, "GVSType", "Group Variable Selection method variants.")
        .value("EN", GVSType::EN, "Elastic Net type standard methodology.")
        .value("IEN", GVSType::IEN, "Interleaved Elastic Net variant.")
        .export_values();

    py::class_<TRexGVSControlParameter>(m, "TRexGVSControlParameter", "Configuration parameters for the GVS logic.")
        .def(py::init<>())
        .def_readwrite("gvs_type", &TRexGVSControlParameter::gvs_type, "Type of GVS method to execute.")
        .def_readwrite("corr_max", &TRexGVSControlParameter::corr_max, "Maximum correlation bounds for grouping.");

    py::class_<TRexGVSSelector::GVSSelectionResult, TRexSelector::SelectionResult>(m, "GVSSelectionResult", "Extended SelectionResult holding specific GVS outputs.")
        .def_readonly("lambda2_used", &TRexGVSSelector::GVSSelectionResult::lambda2_used, "Lambda2 regularization term value chosen.")
        .def_readonly("gvs_type", &TRexGVSSelector::GVSSelectionResult::gvs_type, "GVS method that was executed.")
        .def_readonly("max_clusters", &TRexGVSSelector::GVSSelectionResult::max_clusters, "Maximum number of independent group bounds clustered.")
        .def_readonly("hc_method_used", &TRexGVSSelector::GVSSelectionResult::hc_method_used, "Specific Hierarchical link method evaluated.")
        .def_readonly("groups_vec", &TRexGVSSelector::GVSSelectionResult::groups_vec, "Discrete arrays defining explicit groups assignments.")
        .def_readonly("group_labels", &TRexGVSSelector::GVSSelectionResult::group_labels, "Labeling metrics applied over groups.");

    py::class_<PyTRexGVSSelector>(m, "TRexGVSSelector", "TRex Selector with Group Variable Selection support.")
        .def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::VectorXd>, double, const TRexGVSControlParameter&, const TRexControlParameter&, int, bool>(),
             py::arg("X"), py::arg("y"), py::arg("tFDR") = 0.1,
             py::arg("gvs_control") = TRexGVSControlParameter(),
             py::arg("trex_control") = TRexControlParameter(),
             py::arg("seed") = -1, py::arg("verbose") = true,
             "Mapping engine directly utilizing Python NumPy states strictly via zero-copy paths.")
        .def("select", &PyTRexGVSSelector::select, py::call_guard<py::gil_scoped_release>(), "Run the selection algorithm.")
        .def("getGVSResult", &PyTRexGVSSelector::getGVSResult, "Fetch the detailed GVS selection result.");
}
// =====================================================================================
#endif /* End of TREX_PYTHON_TREX_GVS_BINDINGS_HPP */
