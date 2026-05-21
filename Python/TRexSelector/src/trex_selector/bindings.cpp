// =====================================================================================
// bindings.cpp - Core Python bindings for TRexSelector
// =====================================================================================
/**
 * @file bindings.cpp
 *
 * @brief Main Pybind11 entry point for the TRexSelector Python package.
 *        Constructs the modules, binds the core TRexSelector orchestrator,
 *        and orchestrates the inclusion of other sub-module bindings.
 */
// =====================================================================================

// pybind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

// trexselector project utils include
#include <utils/logging/logger.hpp>

// project
#include "tsolver_bindings.hpp"
#include "ml_bindings.hpp"
#include "clustering_bindings.hpp"
#include "trex_core_bindings.hpp"
#include "utils_bindings.hpp"

// Extracted Subclass bindings
#include "trex_da_bindings.hpp"
#include "trex_gvs_bindings.hpp"
#include "trex_screen_bindings.hpp"
#include "trex_screening_biobanks_bindings.hpp"

// =====================================================================================

namespace py = pybind11;

// =====================================================================================
// Sub-Module Export Definitions
// =====================================================================================

PYBIND11_MODULE(_core, m) {
    m.doc() = "Core C++ bindings for TRexSelector";

    // Define submodules mapping to C++ directories
    py::module m_utils = m.def_submodule("utils", "Utility classes and functions");
    py::module m_ml = m.def_submodule("ml_methods", "Machine learning methods");
    py::module m_tsolvers = m.def_submodule("tsolvers", "Terminating Solvers");
    py::module m_tsm = m.def_submodule("trex_selector_methods",
                                       "T-Rex Selector variants and core algorithms");

    // Call external binder functions
    bind_trex_core(m_tsm);
    bind_tsolvers_module(m_tsolvers);
    bind_ml_methods(m_ml);
    bind_utils_module(m_utils);

    // Bind the extracted subclass logic tightly mapping to m_tsm
    bind_trex_da(m_tsm);
    bind_trex_gvs(m_tsm);
    bind_trex_screen(m_tsm);
    bind_trex_screening_biobanks(m_tsm);

    // Bind Hierarchical clustering as a submodule of m_ml
    py::module m_clustering = m_ml.def_submodule("clustering",
                                                 "Hierarchical clustering algorithms");
    trex::python_bindings::bind_clustering(m_clustering);
}

// =====================================================================================
