// =====================================================================================
// trex_spca_bindings.hpp - Pybind11 bindings for TRexSPCA
// =====================================================================================
#ifndef TREX_PYTHON_TREX_SPCA_BINDINGS_HPP
#define TREX_PYTHON_TREX_SPCA_BINDINGS_HPP
// =====================================================================================
/**
 * @file trex_spca_bindings.hpp
 *
 * @brief Pybind11 bindings for the T-Rex Sparse PCA selector.
 *
 * @details Exposes TRexSPCAResult, TRexSPCAControlParameter, SPCAMode, and the
 *          PyTRexSPCASelector instance wrapper to Python.
 */
// =====================================================================================

// pybind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

// trex_spca include
#include <trex_selector_methods/trex_spca/trex_spca.hpp>

// =====================================================================================

namespace py = pybind11;

using namespace trex::trex_selector_methods::trex_spca;
namespace tc = trex::trex_selector_methods::trex_core;
namespace tg = trex::trex_selector_methods::trex_gvs;

// =====================================================================================

/**
 * @brief Zero-copy Python wrapper for TRexSPCA.
 *
 * @details Holds an Eigen::Map pointing into the numpy array's buffer (no copy),
 *          and a TRexSPCA instance. Mirrors the pattern of PyTRexGVSSelector.
 */
class PyTRexSPCASelector {
public:
    /**
     * @brief Construct a PyTRexSPCASelector.
     *
     * @param X        Design matrix (n × p) — accessed zero-copy via Eigen::Map.
     * @param M        Number of sparse PCs to extract.
     * @param tFDR     Target false discovery rate in (0, 1).
     * @param ctrl     Algorithmic control parameters.
     * @param seed     Random seed (-1 for non-deterministic).
     * @param verbose  Enable verbose output.
     */
    PyTRexSPCASelector(
        Eigen::Ref<Eigen::MatrixXd> X,
        Eigen::Index M,
        double tFDR,
        const TRexSPCAControlParameter& ctrl,
        int seed,
        bool verbose
    ) {
        X_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X.data(), X.rows(), X.cols());
        selector_ = std::make_unique<TRexSPCA>(*X_map_, M, tFDR, ctrl, seed, verbose);
    }

    /** @brief Run T-Rex Sparse PCA selection. */
    void select() {
        result_ = selector_->select();
        has_result_ = true;
    }

    /** @brief Return the result from the last select() call. */
    const TRexSPCAResult& getResult() const {
        if (!has_result_)
            throw std::runtime_error("Call select() before accessing results.");
        return result_;
    }

private:
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>> X_map_;
    std::unique_ptr<TRexSPCA>                    selector_;
    TRexSPCAResult                               result_;
    bool                                         has_result_{false};
};

// =====================================================================================

/**
 * @brief Bind TRexSPCA types and PyTRexSPCASelector to a Python module.
 *
 * @param m The Python module (trex_selector_methods submodule).
 */
inline void bind_trex_spca(py::module& m) {

    // =========================================================================
    // SPCAMode enum
    // =========================================================================

    py::enum_<SPCAMode>(m, "SPCAMode")
        .value("ActiveSet",   SPCAMode::ActiveSet,
               "Ridge regression on the T-Rex active set to compute loadings.")
        .value("Thresholded", SPCAMode::Thresholded,
               "Keep and normalize the top |A| ordinary PCA loadings.")
        .export_values();


    // =========================================================================
    // TRexSPCAControlParameter
    // =========================================================================

    py::class_<TRexSPCAControlParameter>(m, "TRexSPCAControlParameter",
        "Algorithmic control parameters for the T-Rex Sparse PCA selector.")
        .def(py::init<>())
        .def_readwrite("mode",                  &TRexSPCAControlParameter::mode,
                       "Sparse loading strategy (default: ActiveSet).")
        .def_readwrite("lambda2_ridge_loadings", &TRexSPCAControlParameter::lambda2_ridge_loadings,
                       "Ridge penalty for ActiveSet loading assembly (default: 1e-6).")
        .def_readwrite("gvs_ctrl",              &TRexSPCAControlParameter::gvs_ctrl,
                       "TRexGVSControlParameter forwarded to each per-PC GVS run. "
                       "Set gvs_ctrl.gvs_type = GVSType.EN or IEN. "
                       "Set gvs_ctrl.lambda_2 > 0 to bypass auto-determination. "
                       "The base T-Rex parameters are configured via gvs_ctrl.trex_ctrl.")
        .def_readwrite("en_solver",             &TRexSPCAControlParameter::en_solver,
                       "EN solver variant for the per-PC GVS(EN) sub-selector "
                       "(TENET or TENET_AUG; default TENET_AUG).");


    // =========================================================================
    // TRexSPCAResult
    // =========================================================================

    py::class_<TRexSPCAResult>(m, "TRexSPCAResult",
        "Container for the T-Rex Sparse PCA output.")
        .def(py::init<>())
        .def_readwrite("Z",            &TRexSPCAResult::Z,
                       "Sparse PC score matrix (n × M).")
        .def_readwrite("V",            &TRexSPCAResult::V,
                       "Sparse loading matrix (p × M).")
        .def_readwrite("active_sets",  &TRexSPCAResult::active_sets,
                       "Support indices per PC (list of M index vectors, 0-based).")
        .def_readwrite("adjusted_ev",  &TRexSPCAResult::adjusted_ev,
                       "Marginal adjusted explained variance per component (M-vector).")
        .def_readwrite("cumulative_ev",&TRexSPCAResult::cumulative_ev,
                       "Cumulative percentage of explained variance (M-vector).");


    // =========================================================================
    // PyTRexSPCASelector (instance wrapper)
    // =========================================================================

    py::class_<PyTRexSPCASelector>(m, "TRexSPCASelector",
        "T-Rex Sparse PCA selector — instance wrapper (zero-copy Eigen::Map).")
        .def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Index, double,
                      const TRexSPCAControlParameter&, int, bool>(),
             py::arg("X"),
             py::arg("M"),
             py::arg("tFDR"),
             py::arg("ctrl")    = TRexSPCAControlParameter(),
             py::arg("seed")    = -1,
             py::arg("verbose") = false,
             py::keep_alive<1, 2>(),
             "Construct the SPCA selector. X is accessed zero-copy via Eigen::Map.")
        .def("select",    &PyTRexSPCASelector::select,
             "Run T-Rex Sparse PCA selection.")
        .def("getResult", &PyTRexSPCASelector::getResult,
             py::return_value_policy::reference_internal,
             "Return the TRexSPCAResult after calling select().");
}

// =====================================================================================
#endif /* End of TREX_PYTHON_TREX_SPCA_BINDINGS_HPP */
