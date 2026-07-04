// =====================================================================================
// trex_core_bindings.hpp - Core Python bindings for TRexSelector classes
// =====================================================================================
#ifndef TREX_PYTHON_TREX_CORE_BINDINGS_HPP
#define TREX_PYTHON_TREX_CORE_BINDINGS_HPP
// =====================================================================================
/**
 * @file trex_core_bindings.hpp
 *
 * @brief Pybind11 bindings for the core TRexSelector algorithms and configurations.
 */
// =====================================================================================

// pybind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

// Core TRexSelector includes
#include <trex_selector_methods/trex_core/trex.hpp>
#include <trex_selector_methods/trex_utils/trex_solver_dispatch.hpp>
#include <utils/datageneration/utils_dummygen.hpp>

// std includes
#include <numbers>

// =====================================================================================

namespace py = pybind11;
using namespace trex::trex_selector_methods::trex_core;

// =====================================================================================
// Core Class Mapping Wrapper
// =====================================================================================

/**
 * @brief Python wrapper class for TRexSelector leveraging zero-copy references.
 *
 * @details Retains references to the underlying numpy arrays via Eigen::Map without performing
 *          unnecessary deep copies of the data, analogous to the solver wrappers.
 */
class PyTRexSelector {
protected:
    PyTRexSelector() = default;
public:
    virtual ~PyTRexSelector() = default;
protected:
    /** @brief Map for the original design matrix (X) */
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>> X_map_;

    /** @brief Map for the response vector (y) */
    std::unique_ptr<Eigen::Map<Eigen::VectorXd>> y_map_;

    /** @brief Pointer to the underlying core orchestrator instance */
    std::unique_ptr<TRexSelector> selector_;

public:
    /**
     * @brief Zero-copy constructor for the TRexSelector wrapper.
     *
     * @param X Reference to the design matrix (n × p).
     * @param y Reference to the response vector (n × 1).
     * @param tFDR Target False Discovery Rate threshold.
     * @param trex_control Configuration parameters for the algorithm.
     * @param seed Random seed for reproducible computations.
     * @param verbose If true, prompts standard out messaging during select().
     */
    PyTRexSelector(
        Eigen::Ref<Eigen::MatrixXd> X,
        Eigen::Ref<Eigen::VectorXd> y,
        double tFDR,
        const TRexControlParameter& trex_control,
        int seed,
        bool verbose
    ) {
        X_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X.data(),
                                                                  X.rows(),
                                                                      X.cols());
        y_map_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(y.data(),
                                                                  y.size());

        selector_ = std::make_unique<TRexSelector>(
            *X_map_, *y_map_, tFDR, trex_control, seed, verbose
        );
    }

    /**
     * @brief Execute the full selection algorithm.
     *
     * @return SelectionResult Containing all solution structures, selected variables,
     *         and thresholds.
     */
    virtual TRexSelector::SelectionResult select() {
        return selector_->select();
    }

    /** @brief Get Target FDR. */
    double getTFDR() const { return selector_->getTFDR(); }

    /** @brief Get total configurations count for K random simulations. */
    std::size_t getK() const { return selector_->getK(); }

    /** @brief Get the explicitly stated dummy boundary variable count. */
    std::size_t getMaxNumDummies() const { return selector_->getMaxNumDummies(); }

    /** @brief Get exact strict upper length paths executed across arrays. */
    bool getMaxTStop() const { return selector_->getMaxTStop(); }

    /** @brief Get flag highlighting stagnation stoppage active evaluations. */
    bool getStagnationCheck() const { return selector_->getStagnationCheck(); }

    /** @brief Get Predictor X Array normalizations mapping averages. */
    const Eigen::VectorXd& getXMeans() const { return selector_->getXMeans(); }

    /** @brief Get Precalculated absolute variance measurements from matrix entries. */
    const Eigen::VectorXd& getXL2Norms() const { return selector_->getXL2Norms(); }

    /** @brief Get Target Y distribution uniform calculations. */
    double getYMean() const { return selector_->getYMean(); }

    /** @brief Get specific 0|1 integer indications mapping FDR inclusion assumptions. */
    const Eigen::VectorXi& getSelectedVar() const { return selector_->getSelectedVar(); }

    /** @brief Get actual un-ordered variable keys included internally. */
    const std::vector<std::size_t>& getSelectedIndices() const { return selector_->getSelectedIndices(); }

    /** @brief Get total completely bound variables filtered inside prediction criteria. */
    std::size_t getNumSelected() const { return selector_->getNumSelected(); }

    /** @brief Get execution limit bound configurations assigned initially. */
    std::size_t getTStop() const { return selector_->getTStop(); }

    /** @brief Returns final array statistics defining voting weights per execution bounds. */
    const Eigen::VectorXd& getVotingGrid() const { return selector_->getVotingGrid(); }

    /** @brief Numeric scalar utilized determining exact acceptance barriers. */
    double getVotingThreshold() const { return selector_->getVotingThreshold(); }

    /** @brief Secondary FDP Hat indicator metric arrays maps. */
    const Eigen::MatrixXd& getFDPHatMat() const { return selector_->getFDPHatMat(); }

    /** @brief Predictor distributions mapping to Dummy component densities. */
    const Eigen::MatrixXd& getPhiMat() const { return selector_->getPhiMat(); }

    /** @brief R Matrix tracking metrics of exact internal executions. */
    const Eigen::MatrixXd& getRMat() const { return selector_->getRMat(); }

    /** @brief Prime derivative probability arrays indicating exact inclusion metrics. */
    const Eigen::VectorXd& getPhiPrime() const { return selector_->getPhiPrime(); }

    /** @brief Fetch scaling ratio configured strictly to size variables internally. */
    std::size_t getDummyMultiplierL() const { return selector_->getDummyMultiplierL(); }
};


/**
 * @brief Binds the TRexSelector core configurations and main wrapper object.
 *
 * @param m_tsm The submodule to bind against.
 */
inline void bind_trex_core(py::module& m_tsm) {
    namespace sd = trex::trex_selector_methods::utils::solver_dispatch;
    namespace dg = trex::utils::datageneration::dummygen;
    namespace dn = trex::trex_selector_methods::utils::data_normalizer;

    // Bind ScalingMode enum
    py::enum_<dn::ScalingMode>(m_tsm, "ScalingMode", "Column scaling convention applied to X (and dummies) after centering.")
        .value("L2",     dn::ScalingMode::L2,     "Divide each centered column by its L2 norm (unit L2 norm).")
        .value("ZSCORE", dn::ScalingMode::ZSCORE, "Divide each centered column by its sample SD (n-1); glmnet standardize=TRUE convention.")
        .export_values();

    // Bind LLoopStrategy enum
    py::enum_<LLoopStrategy>(m_tsm, "LLoopStrategy", "L-loop strategy controlling how dummy variables are constructed in each experiment.")
        .value("SKIPL", LLoopStrategy::SKIPL, "Skips L-loop iteration (L=1 only).")
        .value("STANDARD", LLoopStrategy::STANDARD, "One dummy block per L value, averaged over L.")
        .value("HCONCAT", LLoopStrategy::HCONCAT, "Horizontal concatenation of all dummy blocks across L values.")
        .value("PERMUTATION", LLoopStrategy::PERMUTATION, "Dummy variables are column-wise permutations of X.")
        .value("PERMUTATION_DIRECT", LLoopStrategy::PERMUTATION_DIRECT, "Direct permutation of X columns without cycling.")
        .value("DIRECT", LLoopStrategy::DIRECT, "Direct dummy generation bypassing L-loop aggregation.")
        .export_values();

    // Bind SolverTypeForTRex enum
    py::enum_<sd::SolverTypeForTRex>(m_tsm, "SolverTypeForTRex", "Solver algorithm variants available for T-Rex.")
        .value("TLARS",      sd::SolverTypeForTRex::TLARS,      "Terminating LARS.")
        .value("TLASSO",     sd::SolverTypeForTRex::TLASSO,     "Terminating LASSO.")
        .value("TENET",      sd::SolverTypeForTRex::TENET,      "Terminating Elastic Net.")
        .value("TSTEPWISE",  sd::SolverTypeForTRex::TSTEPWISE,  "Terminating Stepwise.")
        .value("TSTAGEWISE", sd::SolverTypeForTRex::TSTAGEWISE, "Terminating Stagewise.")
        .value("TOMP",       sd::SolverTypeForTRex::TOMP,       "Terminating OMP.")
        .value("TGP",        sd::SolverTypeForTRex::TGP,        "Terminating Gradient Pursuit.")
        .value("TACGP",      sd::SolverTypeForTRex::TACGP,      "Terminating Approximate Conjugate Gradient Pursuit.")
        .value("TMP",        sd::SolverTypeForTRex::TMP,        "Terminating Matching Pursuit.")
        .value("TNCGMP",     sd::SolverTypeForTRex::TNCGMP,     "Terminating Norm-Corrected Generalized Matching Pursuit.")
        .value("TOOLS",      sd::SolverTypeForTRex::TOOLS,      "Terminating Orthogonal Least Squares.")
        .value("TAFS",       sd::SolverTypeForTRex::TAFS,       "Terminating AFS.")
        .value("TENET_AUG",  sd::SolverTypeForTRex::TENET_AUG,  "Terminating Elastic Net via augmented LASSO (GVS).")
        .export_values();

    // Bind SolverHyperparameters struct
    py::class_<sd::SolverHyperparameters>(m_tsm, "SolverHyperparameters", "Hyperparameters for the selected T-Rex solver.")
        .def(py::init<>())
        .def_readwrite("lambda2",       &sd::SolverHyperparameters::lambda2,       "L2 penalty for ENET-type solvers.")
        .def_readwrite("rho_afs",       &sd::SolverHyperparameters::rho_afs,       "Shrinkage step size for AFS solver in (0, 1].")
        .def_readwrite("ncgmp_variant", &sd::SolverHyperparameters::ncgmp_variant, "NCGMP variant: 0 = LineSearch, 1 = FullyCorrective.")
        .def_readwrite("tol",           &sd::SolverHyperparameters::tol,           "Numerical tolerance for solver steps.");

    // Bind DummyDistribution (wraps C++ dummygen::Distribution)
    py::class_<dg::Distribution> dummy_dist(m_tsm, "DummyDistribution",
        "Specification of the dummy variable distribution injected by T-Rex.");

    py::enum_<dg::Distribution::Type>(dummy_dist, "Type",
        "Dummy distribution type enumeration.")
        .value("Normal",                      dg::Distribution::Type::Normal)
        .value("Uniform",                     dg::Distribution::Type::Uniform)
        .value("Rademacher",                  dg::Distribution::Type::Rademacher)
        .value("StudentT",                    dg::Distribution::Type::StudentT)
        .value("Laplace",                     dg::Distribution::Type::Laplace)
        .value("Gumbel",                      dg::Distribution::Type::Gumbel)
        .value("Holtsmark",                   dg::Distribution::Type::Holtsmark)
        .value("Triangle",                    dg::Distribution::Type::Triangle)
        .value("UniformSphere",               dg::Distribution::Type::UniformSphere)
        .value("Mammen",                      dg::Distribution::Type::Mammen)
        .value("ConstrainedSparseRademacher", dg::Distribution::Type::ConstrainedSparseRademacher)
        .value("Logistic",                    dg::Distribution::Type::Logistic)
        .export_values();

    dummy_dist
        .def(py::init<>(), "Default constructor: Normal distribution.")
        .def_static("normal",     &dg::Distribution::Normal,     "Create Normal distribution.")
        .def_static("rademacher", &dg::Distribution::Rademacher, "Create Rademacher distribution.")
        .def_static("mammen",     &dg::Distribution::Mammen,
            py::arg("p1") = (std::sqrt(5.0) + 1.0) / (2.0 * std::sqrt(5.0)),
            py::arg("p2") = (std::sqrt(5.0) - 1.0) / (2.0 * std::sqrt(5.0)),
            "Create Mammen two-point distribution.")
        .def_static("uniform", &dg::Distribution::Uniform,
            py::arg("a") = std::sqrt(3.0),
            "Create Uniform U(-a, a) distribution.")
        .def_static("student_t", &dg::Distribution::StudentT,
            py::arg("df") = 5.0,
            "Create Student's t-distribution.")
        .def_static("laplace", &dg::Distribution::Laplace,
            py::arg("location") = 0.0, py::arg("scale") = 1.0 / std::sqrt(2.0),
            "Create Laplace distribution.")
        .def_static("gumbel", &dg::Distribution::Gumbel,
            py::arg("location") = 0.0, py::arg("scale") = 1.0,
            "Create Gumbel distribution.")
        .def_static("holtsmark", &dg::Distribution::Holtsmark,
            py::arg("location") = 0.0, py::arg("scale") = 1.0,
            "Create Holtsmark distribution.")
        .def_static("triangle", &dg::Distribution::Triangle,
            py::arg("a") = -std::sqrt(6.0), py::arg("b") = 0.0, py::arg("c") = std::sqrt(6.0),
            "Create Triangle distribution.")
        .def_static("uniform_sphere", &dg::Distribution::UniformSphere,
            py::arg("dim") = 3,
            "Create Uniform distribution on the unit d-sphere.")
        .def_static("constrained_sparse_rademacher", &dg::Distribution::ConstrainedSparseRademacher,
            py::arg("s"),
            "Create Constrained Sparse Rademacher distribution.")
        .def_static("logistic", &dg::Distribution::Logistic,
            py::arg("location") = 0.0, py::arg("scale") = std::sqrt(3.0) / std::numbers::pi,
            "Create Logistic distribution.")
        .def("get_type", &dg::Distribution::get_type, "Get the distribution type.")
        .def_readwrite("uniform_a",                       &dg::Distribution::uniform_a)
        .def_readwrite("student_t_df",                    &dg::Distribution::student_t_df)
        .def_readwrite("laplace_location",                &dg::Distribution::laplace_location)
        .def_readwrite("laplace_scale",                   &dg::Distribution::laplace_scale)
        .def_readwrite("gumbel_location",                 &dg::Distribution::gumbel_location)
        .def_readwrite("gumbel_scale",                    &dg::Distribution::gumbel_scale)
        .def_readwrite("holtsmark_location",              &dg::Distribution::holtsmark_location)
        .def_readwrite("holtsmark_scale",                 &dg::Distribution::holtsmark_scale)
        .def_readwrite("triangle_a",                      &dg::Distribution::triangle_a)
        .def_readwrite("triangle_b",                      &dg::Distribution::triangle_b)
        .def_readwrite("triangle_c",                      &dg::Distribution::triangle_c)
        .def_readwrite("sphere_dim",                      &dg::Distribution::sphere_dim)
        .def_readwrite("mammen_param_p1",                 &dg::Distribution::mammen_param_p1)
        .def_readwrite("mammen_param_p2",                 &dg::Distribution::mammen_param_p2)
        .def_readwrite("constrained_sparse_rademacher_s", &dg::Distribution::constrained_sparse_rademacher_s)
        .def_readwrite("logistic_location",               &dg::Distribution::logistic_location)
        .def_readwrite("logistic_scale",                  &dg::Distribution::logistic_scale);

    // Bind TRexControlParameter
    py::class_<TRexControlParameter>(m_tsm, "TRexControlParameter", "Control parameters for the core T-Rex selection algorithm.")
        .def(py::init<>())
        .def_readwrite("K", &TRexControlParameter::K, "Number of random simulation experiments to run over variable inclusions.")
        .def_readwrite("max_dummy_multiplier", &TRexControlParameter::max_dummy_multiplier, "Maximum dummy multiplier L (default 10).")
        .def_readwrite("use_max_T_stop", &TRexControlParameter::use_max_T_stop, "If true, always run solvers to T_stop regardless of early stopping.")
        .def_readwrite("opt_threshold", &TRexControlParameter::opt_threshold, "Voting threshold parameter (default 0.75).")
        .def_readwrite("lloop_strategy", &TRexControlParameter::lloop_strategy, "L-loop strategy for constructing dummy variables.")
        .def_readwrite("tloop_stagnation_stop", &TRexControlParameter::tloop_stagnation_stop, "If true, stop the T-loop early when stagnation is detected.")
        .def_readwrite("tloop_max_stagnant_steps", &TRexControlParameter::tloop_max_stagnant_steps, "Maximum consecutive stagnant T-loop steps before early stopping.")
        .def_readwrite("parallel_rnd_experiments", &TRexControlParameter::parallel_rnd_experiments, "Run K random experiments in parallel using OpenMP.")
        .def_readwrite("use_memory_mapping",  &TRexControlParameter::use_memory_mapping,  "Use out-of-core memory-mapped dummy matrix.")
        .def_readwrite("max_outer_threads",   &TRexControlParameter::max_outer_threads,   "Thread count for K repetitions (outer parallelism).")
        .def_readwrite("max_inner_threads",   &TRexControlParameter::max_inner_threads,   "Thread count for inner solver operations.")
        .def_readwrite("solver_type",         &TRexControlParameter::solver_type,         "Solver algorithm to use for T-Rex.")
        .def_readwrite("solver_params",       &TRexControlParameter::solver_params,       "Hyperparameters for the selected solver.")
        .def_readwrite("scaling_mode",        &TRexControlParameter::scaling_mode,        "Column scaling convention applied to X after centering (L2 or ZSCORE).")
        .def_readwrite("dummy_distribution",  &TRexControlParameter::dummy_distribution,  "Distribution used to generate dummy variables.");

    // Bind TRexSelector::SelectionResult
    py::class_<TRexSelector::SelectionResult>(m_tsm, "SelectionResult", "Return value of TRexSelector.select(): selected variables and diagnostic matrices.")
        .def_readonly("selected_var", &TRexSelector::SelectionResult::selected_var, "Selection indicator vector (length p; 1 = selected, 0 = not selected).")
        .def_readonly("v_thresh", &TRexSelector::SelectionResult::v_thresh, "Voting threshold v applied to control the FDR.")
        .def_readonly("r_mat", &TRexSelector::SelectionResult::R_mat, "R matrix of empirical tail probabilities (p x T_stop).")
        .def_readonly("T_stop", &TRexSelector::SelectionResult::T_stop, "T_stop: maximum number of LARS steps run.")
        .def_readonly("num_dummies", &TRexSelector::SelectionResult::num_dummies, "Total number of dummy variables used.")
        .def_readonly("dummy_factor_L", &TRexSelector::SelectionResult::dummy_factor_L, "Dummy multiplier L used.")
        .def_readonly("fdp_hat_mat", &TRexSelector::SelectionResult::FDP_hat_mat, "FDP hat matrix (p x |voting_grid|).")
        .def_readonly("phi_mat", &TRexSelector::SelectionResult::Phi_mat, "Phi matrix: relative occurrence counts per predictor.")
        .def_readonly("phi_prime", &TRexSelector::SelectionResult::Phi_prime, "Smoothed selection probability vector phi'.")
        .def_readonly("voting_grid", &TRexSelector::SelectionResult::voting_grid, "Voting grid: per-candidate vote totals across K experiments.");

    // Bind PyTRexSelector
    py::class_<PyTRexSelector>(m_tsm, "TRexSelector", "T-Rex Selector: FDR-controlled variable selection via dummy comparison.")
        .def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::VectorXd>, double, const TRexControlParameter&, int, bool>(),
             py::arg("X"), py::arg("y"), py::arg("tFDR") = 0.1,
             py::arg("trex_control") = TRexControlParameter(),
             py::arg("seed") = -1, py::arg("verbose") = true,
             "Construct the selector. X and y are accessed zero-copy via Eigen::Map.")
        .def("select", &PyTRexSelector::select, py::call_guard<py::gil_scoped_release>(),
             "Run the T-Rex algorithm and return a SelectionResult.")
        .def("getTFDR", &PyTRexSelector::getTFDR, "Return the target FDR.")
        .def("getK", &PyTRexSelector::getK, "Return K: the number of random experiments.")
        .def("getMaxNumDummies", &PyTRexSelector::getMaxNumDummies, "Return the maximum number of dummy variables per L level.")
        .def("getMaxTStop", &PyTRexSelector::getMaxTStop, "Return whether use_max_T_stop is active.")
        .def("getStagnationCheck", &PyTRexSelector::getStagnationCheck, "Return whether T-loop stagnation stopping is active.")
        .def("getXMeans", &PyTRexSelector::getXMeans, "Return the column means of X.")
        .def("getXL2Norms", &PyTRexSelector::getXL2Norms, "Return the column-wise L2 norms of X after centering.")
        .def("getYMean", &PyTRexSelector::getYMean, "Return the mean of y.")
        .def("getSelectedVar", &PyTRexSelector::getSelectedVar, "Return the binary selection indicator vector (1 = selected).")
        .def("getSelectedIndices", &PyTRexSelector::getSelectedIndices, "Return 0-based indices of selected predictors.")
        .def("getNumSelected", &PyTRexSelector::getNumSelected, "Return the number of selected predictors.")
        .def("getTStop", &PyTRexSelector::getTStop, "Return T_stop: the LARS step count used.")
        .def("getVotingGrid", &PyTRexSelector::getVotingGrid, "Return the voting grid.")
        .def("getVotingThreshold", &PyTRexSelector::getVotingThreshold, "Return the voting threshold v.")
        .def("getFDPHatMat", &PyTRexSelector::getFDPHatMat, "Return the FDP hat matrix.")
        .def("getPhiMat", &PyTRexSelector::getPhiMat, "Return the Phi occurrence-count matrix.")
        .def("getRMat", &PyTRexSelector::getRMat, "Return the R matrix of empirical tail probabilities.")
        .def("getPhiPrime", &PyTRexSelector::getPhiPrime, "Return the smoothed selection probability vector phi'.")
        .def("getDummyMultiplierL", &PyTRexSelector::getDummyMultiplierL, "Return L: the dummy multiplier used.");
}
// =====================================================================================
#endif /* End of TREX_PYTHON_TREX_CORE_BINDINGS_HPP */
