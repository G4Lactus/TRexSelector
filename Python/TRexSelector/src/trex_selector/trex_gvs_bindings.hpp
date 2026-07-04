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
#include "trex_core_bindings.hpp"

// =====================================================================================

namespace py = pybind11;
using namespace trex::trex_selector_methods::trex_gvs;
using namespace trex::trex_selector_methods::trex_core;
namespace gvs_sd = trex::trex_selector_methods::utils::solver_dispatch;

// =====================================================================================

/**
 * @brief Python wrapper class for TRexGVSSelector leveraging zero-copy references.
 */
class PyTRexGVSSelector : public PyTRexSelector {
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
        this->X_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X.data(),
                                                                  X.rows(),
                                                                  X.cols());
        this->y_map_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(y.data(),
                                                                  y.size());

        // The refactored C++ constructor takes a single control struct that
        // nests the base algorithmic parameters as `trex_ctrl`. Merge the
        // separately-supplied `trex_control` and derive the solver_type GVS
        // requires (EN -> TENET/TENET_AUG, IEN -> TLASSO), overriding whatever
        // the caller left in `trex_control.solver_type`.
        TRexGVSControlParameter gvs_ctrl = gvs_control;
        gvs_ctrl.trex_ctrl = trex_control;
        if (gvs_ctrl.gvs_type == GVSType::IEN) {
            gvs_ctrl.trex_ctrl.solver_type = gvs_sd::SolverTypeForTRex::TLASSO;
        } else { // EN
            gvs_ctrl.trex_ctrl.solver_type =
                (gvs_ctrl.en_solver == ENSolverType::TENET_AUG)
                    ? gvs_sd::SolverTypeForTRex::TENET_AUG
                    : gvs_sd::SolverTypeForTRex::TENET;
        }

        this->selector_ = std::make_unique<TRexGVSSelector>(
            *(this->X_map_), *(this->y_map_), tFDR, gvs_ctrl, seed, verbose
        );
    }

    /**
     * @brief Get the complete GVS selection result.
     */
    const TRexGVSSelector::GVSSelectionResult& getGVSResult() const {
        return static_cast<TRexGVSSelector*>(this->selector_.get())->getGVSResult();
    }
};


/**
 * @brief Binds the TRexGVSSelector and related configurations.
 *
 * @param m The Python module to bind against.
 */
inline void bind_trex_gvs(py::module& m) {
    py::enum_<GVSType>(m, "GVSType", "Group Variable Selection method variants.")
        .value("EN", GVSType::EN, "Elastic-Net group variable selection (uses TENET solver).")
        .value("IEN", GVSType::IEN, "Informed Elastic-Net using group structure from clustering (uses TLASSO solver).")
        .export_values();

    py::enum_<ENSolverType>(m, "ENSolverType", "Elastic-Net solver variant used when gvs_type == EN.")
        .value("TENET",     ENSolverType::TENET,     "Gram-based Elastic Net (ridge absorbed via Cholesky update).")
        .value("TENET_AUG", ENSolverType::TENET_AUG, "Augmented-LASSO Elastic Net (row-augmented system).")
        .export_values();

    py::enum_<LambdaSelectionMethod>(m, "LambdaSelectionMethod", "Rule for auto-selecting the ridge parameter lambda_2 when lambda_2 < 0.")
        .value("CV_1SE_SVD", LambdaSelectionMethod::CV_1SE_SVD, "JacobiSVD ridge CV, 1-SE rule.")
        .value("CV_MIN_SVD", LambdaSelectionMethod::CV_MIN_SVD, "JacobiSVD ridge CV, minimum-CV-error rule.")
        .value("CV_1SE_CCD", LambdaSelectionMethod::CV_1SE_CCD, "Coordinate-descent ridge CV, glmnet-faithful 1-SE rule (default).")
        .value("CV_MIN_CCD", LambdaSelectionMethod::CV_MIN_CCD, "Coordinate-descent ridge CV, glmnet-faithful minimum rule.")
        .export_values();

    py::class_<TRexGVSControlParameter>(m, "TRexGVSControlParameter", "Control parameters for TRexGVSSelector.")
        .def(py::init<>())
        .def_readwrite("gvs_type",    &TRexGVSControlParameter::gvs_type,    "GVS method variant (EN = Elastic Net, IEN = Informed Elastic Net).")
        .def_readwrite("en_solver",   &TRexGVSControlParameter::en_solver,   "EN solver variant when gvs_type == EN (TENET or TENET_AUG).")
        .def_readwrite("corr_max",    &TRexGVSControlParameter::corr_max,    "Maximum pairwise correlation for automatic cluster formation.")
        .def_readwrite("hc_linkage",  &TRexGVSControlParameter::hc_linkage,  "Hierarchical clustering linkage method.")
        .def_readwrite("lambda_2", &TRexGVSControlParameter::lambda_2, "Ridge L2 penalty in LARS units (< 0 = auto-select, 0 = pure TLASSO, > 0 = fixed).")
        .def_readwrite("lambda2_method", &TRexGVSControlParameter::lambda2_method, "Rule for auto-selecting lambda_2 when lambda_2 < 0.")
        .def_readwrite("cv_n_folds", &TRexGVSControlParameter::cv_n_folds, "Number of folds for cross-validated lambda_2 selection.")
        .def_readwrite("cv_n_lambda", &TRexGVSControlParameter::cv_n_lambda, "Number of points on the lambda grid searched during CV.")
        .def_readwrite("cv_seed", &TRexGVSControlParameter::cv_seed, "Seed for the CV fold-permutation RNG (-1 = derive from the T-Rex seed).")
        .def_readwrite("prior_groups", &TRexGVSControlParameter::prior_groups, "Manually specified group assignments (0-based; empty = auto-cluster).")
        .def_readwrite("group_labels", &TRexGVSControlParameter::group_labels, "Optional human-readable cluster names (size must equal number of clusters).")
        .def_readwrite("tenet_aug_use_lars", &TRexGVSControlParameter::tenet_aug_use_lars, "For TENET_AUG: use a pure-LARS inner solver (R-parity diagnostic).")
        .def_readwrite("trex_ctrl", &TRexGVSControlParameter::trex_ctrl, "Nested base T-Rex algorithmic control parameters (used directly by TRexSPCASelector; overridden by the separate trex_control argument in TRexGVSSelector).");

    py::class_<TRexGVSSelector::GVSSelectionResult, TRexSelector::SelectionResult>(m, "GVSSelectionResult", "SelectionResult extended with GVS-specific fields.")
        .def_readonly("lambda2_used", &TRexGVSSelector::GVSSelectionResult::lambda2_used, "L2 penalty lambda2 used (auto-selected via GCV if not specified).")
        .def_readonly("gvs_type", &TRexGVSSelector::GVSSelectionResult::gvs_type, "GVS method used (EN or IEN).")
        .def_readonly("max_clusters", &TRexGVSSelector::GVSSelectionResult::max_clusters, "Number of variable groups identified by hierarchical clustering.")
        .def_readonly("hc_method_used", &TRexGVSSelector::GVSSelectionResult::hc_method_used, "Hierarchical clustering linkage method used.")
        .def_readonly("groups_vec", &TRexGVSSelector::GVSSelectionResult::groups_vec, "Group assignment vector (0-based group index per predictor).")
        .def_readonly("group_labels", &TRexGVSSelector::GVSSelectionResult::group_labels, "Unique group label integers.");

    py::class_<PyTRexGVSSelector, PyTRexSelector>(m, "TRexGVSSelector", "TRex Selector with Group Variable Selection support.")
        .def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::VectorXd>, double, const TRexGVSControlParameter&, const TRexControlParameter&, int, bool>(),
             py::arg("X"), py::arg("y"), py::arg("tFDR") = 0.1,
             py::arg("gvs_control") = TRexGVSControlParameter(),
             py::arg("trex_control") = TRexControlParameter(),
             py::arg("seed") = -1, py::arg("verbose") = true,
             "Construct the GVS selector. X and y are accessed zero-copy via Eigen::Map.")
        .def("getGVSResult", &PyTRexGVSSelector::getGVSResult, "Return the GVSSelectionResult after calling select().");
}
// =====================================================================================
#endif /* End of TREX_PYTHON_TREX_GVS_BINDINGS_HPP */
