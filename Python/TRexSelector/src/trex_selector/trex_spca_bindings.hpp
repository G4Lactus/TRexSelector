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
 *          static TRexSPCA::select() entry point to Python.
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

// =====================================================================================

/**
 * @brief Bind TRexSPCA types and the static select() entry point to a Python module.
 *
 * @param m The Python module (trex_selector_methods submodule).
 */
inline void bind_trex_spca(py::module& m) {

    // =========================================================================
    // SPCAMode enum
    // =========================================================================

    /**
     * @brief Selects the sparse loading assembly strategy.
     */
    py::enum_<SPCAMode>(m, "SPCAMode")
        .value("ActiveSet",   SPCAMode::ActiveSet,
               "Ridge regression on the T-Rex active set to compute loadings.")
        .value("Thresholded", SPCAMode::Thresholded,
               "Keep and normalize the top |A| ordinary PCA loadings.")
        .export_values();


    // =========================================================================
    // TRexSPCAControlParameter
    // =========================================================================

    /**
     * @brief Algorithmic control parameters for the T-Rex Sparse PCA selector.
     */
    py::class_<TRexSPCAControlParameter>(m, "TRexSPCAControlParameter")
        .def(py::init<>())
        .def_readwrite("mode",     &TRexSPCAControlParameter::mode,
                       "Sparse loading strategy (default: ActiveSet).")
        .def_readwrite("lambda2",  &TRexSPCAControlParameter::lambda2,
                       "Ridge penalty for ActiveSet loading assembly (default: 1e-6).")
        .def_readwrite("seed",     &TRexSPCAControlParameter::seed,
                       "Random seed forwarded to each per-PC T-Rex run (-1 = hardware entropy).")
        .def_readwrite("trex_ctrl",&TRexSPCAControlParameter::trex_ctrl,
                       "TRexControlParameter forwarded to each per-PC T-Rex run.");


    // =========================================================================
    // TRexSPCAResult
    // =========================================================================

    /**
     * @brief Container for the T-Rex Sparse PCA output.
     *
     * @details Holds score matrix Z (n × M), loading matrix V (p × M),
     *          per-component active sets, and explained variance vectors.
     */
    py::class_<TRexSPCAResult>(m, "TRexSPCAResult")
        .def(py::init<>())
        .def_readwrite("Z",           &TRexSPCAResult::Z,
                       "Sparse PC score matrix (n × M).")
        .def_readwrite("V",           &TRexSPCAResult::V,
                       "Sparse loading matrix (p × M).")
        .def_readwrite("active_sets", &TRexSPCAResult::active_sets,
                       "Support indices per PC (list of M index vectors, 0-based).")
        .def_readwrite("adjusted_ev", &TRexSPCAResult::adjusted_ev,
                       "Marginal adjusted explained variance per component (M-vector).")
        .def_readwrite("cumulative_ev",&TRexSPCAResult::cumulative_ev,
                       "Cumulative percentage of explained variance (M-vector).");


    // =========================================================================
    // TRexSPCA (static entry point only)
    // =========================================================================

    /**
     * @brief T-Rex Sparse PCA orchestrator — all methods are static.
     */
    py::class_<TRexSPCA>(m, "TRexSPCA")
        .def(py::init<>())
        .def_static("select",
            [](Eigen::Ref<Eigen::MatrixXd> X,
               Eigen::Index M,
               double tFDR,
               TRexSPCAControlParameter spca_ctrl) {
                Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
                return TRexSPCA::select(X_map, M, tFDR, spca_ctrl);
            },
            py::arg("X"),
            py::arg("M"),
            py::arg("tFDR"),
            py::arg("spca_ctrl") = TRexSPCAControlParameter(),
            "Run T-Rex Sparse PCA selection.\n\n"
            "Parameters\n----------\n"
            "X         : ndarray (n x p), column-centered internally, restored on return.\n"
            "M         : int, number of sparse PCs to extract.\n"
            "tFDR      : float, target false discovery rate in (0, 1).\n"
            "spca_ctrl : TRexSPCAControlParameter, algorithmic control.\n\n"
            "Returns\n-------\n"
            "TRexSPCAResult");
}

// =====================================================================================
#endif /* End of TREX_PYTHON_TREX_SPCA_BINDINGS_HPP */
