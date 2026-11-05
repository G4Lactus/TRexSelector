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
#include <ml_methods/model_selection/lambda2_cv_diagnostics.hpp>
#include <trex_selector_methods/ad/trex_gvs/trex_gvs.hpp>
#include "trex_core_bindings.hpp"

// =====================================================================================

namespace py = pybind11;
using namespace trex::trex_selector_methods::ad::trex_gvs;
using namespace trex::trex_selector_methods::ad::trex_core;
namespace gvs_sd = trex::trex_selector_methods::ad::utils::solver_dispatch;

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

        // The C++ constructor takes a single control struct that nests the
        // base algorithmic parameters as `trex_ctrl`. Merge the separately-
        // supplied `trex_control` and set the solver_type GVS requires,
        // mirroring the R surface:
        //   EN  -> derived from en_solver (TENET / TENET_AUG / TCDENET);
        //   IEN -> honors an IEN-family trex_control.solver_type (TIENET /
        //          TIENET_AUG / TCDIENET), defaulting anything else to
        //          TIENET (the native pathwise IEN solver).
        TRexGVSControlParameter gvs_ctrl = gvs_control;
        gvs_ctrl.trex_ctrl = trex_control;
        if (gvs_ctrl.gvs_type == GVSType::IEN) {
            const auto st = gvs_ctrl.trex_ctrl.solver_type;
            const bool ien_family =
                st == gvs_sd::SolverTypeForTRex::TIENET ||
                st == gvs_sd::SolverTypeForTRex::TIENET_AUG ||
                st == gvs_sd::SolverTypeForTRex::TCDIENET;
            if (!ien_family) {
                gvs_ctrl.trex_ctrl.solver_type =
                    gvs_sd::SolverTypeForTRex::TIENET;
            }
        } else { // EN
            switch (gvs_ctrl.en_solver) {
                case ENSolverType::TENET_AUG:
                    gvs_ctrl.trex_ctrl.solver_type =
                        gvs_sd::SolverTypeForTRex::TENET_AUG;
                    break;
                case ENSolverType::TCDENET:
                    gvs_ctrl.trex_ctrl.solver_type =
                        gvs_sd::SolverTypeForTRex::TCDENET;
                    break;
                default:
                    gvs_ctrl.trex_ctrl.solver_type =
                        gvs_sd::SolverTypeForTRex::TENET;
            }
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

    /**
     * @brief The lambda_2 tuner health record of the last select().
     */
    const trex::ml_methods::model_selection::lambda2_cv_diagnostics&
    getLambda2CVDiagnostics() const {
        return static_cast<TRexGVSSelector*>(this->selector_.get())
            ->getLambda2CVDiagnostics();
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
        .value("IEN", GVSType::IEN, "Informed Elastic-Net using group structure from clustering (solver via "
            "trex_control.solver_type: TIENET default, TIENET_AUG, TCDIENET).")
        .export_values();

    py::enum_<ENSolverType>(m, "ENSolverType", "Elastic-Net solver variant used when gvs_type == EN.")
        .value("TENET",     ENSolverType::TENET,     "Gram-based Elastic Net (ridge absorbed via Cholesky update).")
        .value("TENET_AUG", ENSolverType::TENET_AUG, "Augmented-LASSO Elastic Net (row-augmented system).")
        .value("TCDENET",   ENSolverType::TCDENET,   "Coordinate-descent Elastic Net (exact fixed-lambda1 minimizers "
            "per dummy crossing).")
        .export_values();

    py::class_<trex::ml_methods::model_selection::lambda2_cv_diagnostics>(m, "Lambda2CVDiagnostics",
        "Health record of a lambda_2 tuner run: whether it ran, its cost, the truncation-edge count, and the "
            "boundary-pick guard fields for the returned pick (grid_index / grid_size, at_grid_edge, profile_at_edge, "
            "boundary_pick).")
        .def(py::init<>())
        .def_readonly("ran", &trex::ml_methods::model_selection::lambda2_cv_diagnostics::ran, "A tuner ran (False: "
            "lambda_2 came from the caller).")
        .def_readonly("seconds", &trex::ml_methods::model_selection::lambda2_cv_diagnostics::seconds, "Wall-clock "
            "seconds spent resolving lambda_2.")
        .def_readonly("n_profile_at_edge",
            &trex::ml_methods::model_selection::lambda2_cv_diagnostics::n_profile_at_edge, "lambda_2 "
            ""
            "values whose profiled lambda_1 optimum landed on a truncation edge (0 = healthy).")
        .def_readonly("support_cap", &trex::ml_methods::model_selection::lambda2_cv_diagnostics::support_cap, "Support "
            "cap (glmnet dfmax) applied, or -1 when disabled.")
        .def_readonly("dfmax_ratio", &trex::ml_methods::model_selection::lambda2_cv_diagnostics::dfmax_ratio, "IEN "
            "tuner: the support-cap ratio it was called with (cv_dfmax_ratio, as passed; <= 0 disabled the cap). 0 "
                "when "
            "no IEN-geometry tuner ran.")
        .def_readonly("low_dimensional", &trex::ml_methods::model_selection::lambda2_cv_diagnostics::low_dimensional,
            "p "
            ""
            "<= cap: the cap was inert.")
        // Boundary-pick guard: the chosen pick on its grid.
        .def_readonly("grid_index", &trex::ml_methods::model_selection::lambda2_cv_diagnostics::grid_index, "0-based "
            "index of the returned lambda_2 on the tuner's grid (-1: no tuner ran).")
        .def_readonly("grid_size", &trex::ml_methods::model_selection::lambda2_cv_diagnostics::grid_size, "Number of "
            "lambda_2 grid points searched (0: no tuner ran).")
        .def_readonly("at_grid_edge", &trex::ml_methods::model_selection::lambda2_cv_diagnostics::at_grid_edge, "The "
            "returned lambda_2 is the first or last grid point: the grid boundary, not a CV optimum.")
        .def_readonly("profile_at_edge", &trex::ml_methods::model_selection::lambda2_cv_diagnostics::profile_at_edge,
            "IEN "
            ""
            "tuner: the profiled lambda_1 optimum of the chosen lambda_2 sat on the truncation edge of the lambda_1 "
            "path.")
        .def_readonly("profile_at_support_cap",
            &trex::ml_methods::model_selection::lambda2_cv_diagnostics::profile_at_support_cap, "IEN "
            ""
            "tuner: profile_at_edge, and that edge is the support cap (raise cv_dfmax_ratio to move it).")
        .def_readonly("boundary_pick", &trex::ml_methods::model_selection::lambda2_cv_diagnostics::boundary_pick,
            "at_grid_edge "
            ""
            "or profile_at_edge: the selector warned once and returned the pick unchanged.");

    py::enum_<LambdaSelectionMethod>(m, "LambdaSelectionMethod", "Rule for auto-selecting the ridge parameter lambda_2 "
        "when lambda_2 < 0.")
        .value("CV_1SE_SVD", LambdaSelectionMethod::CV_1SE_SVD, "JacobiSVD ridge CV, 1-SE rule.")
        .value("CV_MIN_SVD", LambdaSelectionMethod::CV_MIN_SVD, "JacobiSVD ridge CV, minimum-CV-error rule.")
        .value("CV_1SE_CCD", LambdaSelectionMethod::CV_1SE_CCD, "Coordinate-descent ridge CV, glmnet-faithful 1-SE "
            "rule "
            "(default).")
        .value("CV_MIN_CCD", LambdaSelectionMethod::CV_MIN_CCD, "Coordinate-descent ridge CV, glmnet-faithful minimum "
            "rule.")
        .value("CV_1SE_IEN_CCD", LambdaSelectionMethod::CV_1SE_IEN_CCD, "IEN-geometry coordinate-descent CV (lambda1 "
            "profiled, group-consuming), 1-SE rule. IEN default.")
        .value("CV_MIN_IEN_CCD", LambdaSelectionMethod::CV_MIN_IEN_CCD, "IEN-geometry coordinate-descent CV (lambda1 "
            "profiled, group-consuming), minimum rule.")
        .value("CV_1SE_TIK_SVD", LambdaSelectionMethod::CV_1SE_TIK_SVD, "Generalized-Tikhonov SVD ridge CV (IEN "
            "geometry), 1-SE rule.")
        .value("CV_MIN_TIK_SVD", LambdaSelectionMethod::CV_MIN_TIK_SVD, "Generalized-Tikhonov SVD ridge CV (IEN "
            "geometry), minimum rule.")
        .export_values();

    py::class_<TRexGVSControlParameter>(m, "TRexGVSControlParameter", "Control parameters for TRexGVSSelector.")
        .def(py::init<>())
        .def_readwrite("gvs_type",    &TRexGVSControlParameter::gvs_type,    "GVS method variant (EN = Elastic Net, "
            "IEN "
            "= Informed Elastic Net).")
        .def_readwrite("en_solver",   &TRexGVSControlParameter::en_solver,   "EN solver variant when gvs_type == EN: "
            "TENET (default), TENET_AUG, or TCDENET.")
        .def_readwrite("corr_max",    &TRexGVSControlParameter::corr_max,    "Maximum pairwise correlation for "
            "automatic cluster formation.")
        .def_readwrite("hc_linkage",  &TRexGVSControlParameter::hc_linkage,  "Hierarchical clustering linkage method.")
        .def_readwrite("lambda_2", &TRexGVSControlParameter::lambda_2, "Ridge L2 penalty in LARS units (< 0 = "
            "auto-select, 0 = pure TLASSO, > 0 = fixed).")
        .def_readwrite("lambda2_method", &TRexGVSControlParameter::lambda2_method, "Rule for auto-selecting lambda_2 "
            "when lambda_2 < 0.")
        .def_readwrite("cv_n_folds", &TRexGVSControlParameter::cv_n_folds, "Number of folds for cross-validated "
            "lambda_2 selection.")
        .def_readwrite("cv_n_lambda", &TRexGVSControlParameter::cv_n_lambda, "Number of points on the lambda grid "
            "searched during CV (EN-shaped rules).")
        .def_readwrite("cv_n_lambda1", &TRexGVSControlParameter::cv_n_lambda1, "IEN-geometry CD tuners (CV_*_IEN_CCD): "
            "points on the profiled lambda_1 path per lambda_2 value (default 40).")
        .def_readwrite("cv_dfmax_ratio", &TRexGVSControlParameter::cv_dfmax_ratio, "IEN-geometry CD tuners "
            "(CV_*_IEN_CCD): support cap as a fraction of the smallest fold's training size (default 0.8; <= 0 "
                "disables "
            "the cap; larger values allow larger supports at a higher tuner cost; must be finite).")
        .def_readwrite("cv_n_lambda2", &TRexGVSControlParameter::cv_n_lambda2, "IEN-geometry CD tuners: points on the "
            "lambda_2 grid (default 100).")
        .def_readwrite("cv_seed", &TRexGVSControlParameter::cv_seed, "Seed for the CV fold-permutation RNG (-1 = "
            "derive "
            "from the T-Rex seed).")
        .def_readwrite("prior_groups", &TRexGVSControlParameter::prior_groups, "Manually specified group assignments "
            "(0-based; empty = auto-cluster).")
        .def_readwrite("group_labels", &TRexGVSControlParameter::group_labels, "Optional human-readable cluster names "
            "(size must equal number of clusters).")
        .def_readwrite("tenet_aug_use_lars", &TRexGVSControlParameter::tenet_aug_use_lars, "For TENET_AUG: use a "
            "pure-LARS inner solver (R-parity diagnostic).")
        .def_readwrite("trex_ctrl", &TRexGVSControlParameter::trex_ctrl, "Nested base T-Rex algorithmic control "
            "parameters (used directly by TRexSPCASelector; overridden by the separate trex_control argument in "
            "TRexGVSSelector).");

    py::class_<TRexGVSSelector::GVSSelectionResult, TRexSelector::SelectionResult>(m, "GVSSelectionResult",
        "SelectionResult "
        ""
        "extended with GVS-specific fields.")
        .def_readonly("lambda2_used", &TRexGVSSelector::GVSSelectionResult::lambda2_used, "L2 penalty lambda2 used "
            "(auto-selected via GCV if not specified).")
        .def_readonly("gvs_type", &TRexGVSSelector::GVSSelectionResult::gvs_type, "GVS method used (EN or IEN).")
        .def_readonly("max_clusters", &TRexGVSSelector::GVSSelectionResult::max_clusters, "Number of variable groups "
            "identified by hierarchical clustering.")
        .def_readonly("hc_method_used", &TRexGVSSelector::GVSSelectionResult::hc_method_used, "Hierarchical clustering "
            "linkage method used.")
        .def_readonly("groups_vec", &TRexGVSSelector::GVSSelectionResult::groups_vec, "Group assignment vector "
            "(0-based "
            "group index per predictor).")
        .def_readonly("group_labels", &TRexGVSSelector::GVSSelectionResult::group_labels, "Unique group label "
            "integers.")
        .def_readonly("lambda2_cv", &TRexGVSSelector::GVSSelectionResult::lambda2_cv, "Health record of the lambda_2 "
            "tuner (Lambda2CVDiagnostics).");

    py::class_<PyTRexGVSSelector, PyTRexSelector>(m, "TRexGVSSelector",
        "TRex Selector with Group Variable Selection support.\n"
        "\n"
        "References\n"
        "----------\n"
        "J. Machkour, M. Muma, and D. P. Palomar, \"False Discovery Rate Control for Grouped\n"
        "  Variable Selection in High-Dimensional Linear Models Using the T-Knock Filter,\" in\n"
        "  Proc. 30th Eur. Signal Process. Conf. (EUSIPCO), 2022, pp. 892-896.\n"
        "J. Machkour, M. Muma, and D. P. Palomar, \"The Informed Elastic Net for Fast Grouped\n"
        "  Variable Selection and FDR Control in Genomics Research,\" in Proc. IEEE 9th Int.\n"
        "  Workshop Comput. Adv. Multi-Sensor Adapt. Process. (CAMSAP), 2023, pp. 466-470.\n"
        "H. Zou and T. Hastie, \"Regularization and Variable Selection via the Elastic Net,\" J. R.\n"
        "  Stat. Soc. Ser. B Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.\n"
        "J. Machkour, M. Muma, and D. P. Palomar, \"The Terminating-Random Experiments Selector:\n"
        "  Fast High-Dimensional Variable Selection with False Discovery Rate Control,\" Signal\n"
        "  Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.\n")
        .def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::VectorXd>, double, const TRexGVSControlParameter&,
            const TRexControlParameter&, int, bool>(),
             py::arg("X"), py::arg("y"), py::arg("tFDR") = 0.1,
             py::arg("gvs_control") = TRexGVSControlParameter(),
             py::arg("trex_control") = TRexControlParameter(),
             py::arg("seed") = -1, py::arg("verbose") = true,
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
             "Construct the GVS selector. X and y are accessed zero-copy via Eigen::Map.")
        .def("getGVSResult", &PyTRexGVSSelector::getGVSResult, "Return the GVSSelectionResult after calling select().")
        .def("getLambda2CVDiagnostics", &PyTRexGVSSelector::getLambda2CVDiagnostics, "Return the lambda_2 tuner health "
            "record of the last select().");
}
// =====================================================================================
#endif /* End of TREX_PYTHON_TREX_GVS_BINDINGS_HPP */
