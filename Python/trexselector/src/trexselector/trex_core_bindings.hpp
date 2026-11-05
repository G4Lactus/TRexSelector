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
#include <trex_selector_methods/ad/trex_core/trex.hpp>
#include <trex_selector_methods/ad/trex_utils/trex_solver_dispatch.hpp>
#include <utils/datageneration/utils_dummygen.hpp>

// std includes
#include <numbers>

// =====================================================================================

namespace py = pybind11;
using namespace trex::trex_selector_methods::ad::trex_core;

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

    /** @brief Give X back before select() (idempotent; no-op after select()). */
    void restore() { selector_->restore(); }

    /** @brief Number of dummies used by the last select() (the L-loop's calibration). */
    std::size_t getNumDummies() const noexcept { return selector_->getNumDummies(); }

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
    namespace disp = trex::trex_selector_methods::ad::utils::solver_dispatch;
    namespace dg = trex::utils::datageneration::dummygen;
    namespace dn = trex::trex_selector_methods::ad::utils::data_normalizer;

    // Bind ScalingMode enum
    py::enum_<dn::ScalingMode>(m_tsm, "ScalingMode",
        "Working-design convention of the T-Rex selectors. L2 and ZSCORE center X and y and "
        "scale the columns of X (dummies follow); NONE leaves X and y exactly as supplied "
        "(user-managed centering and scaling, e.g. a pre-normalized design or an intercept-free "
        "dictionary), with the dummies kept on the unit-L2 standard.")
        .value("L2",     dn::ScalingMode::L2,     "Divide each centered column by its L2 norm (unit L2 norm).")
        .value("ZSCORE", dn::ScalingMode::ZSCORE, "Divide each centered column by its sample SD (n-1); glmnet "
            "standardize=TRUE convention.")
        .value("NONE",   dn::ScalingMode::NONE,   "Leave X and y untouched (no centering, no scaling); the design "
            "should be unit-L2-commensurate since the dummies keep the unit-L2 standard. A design pre-normalized with "
            "normalize_design() selects under NONE exactly what the raw design selects under L2.")
        .export_values();

    // The library's own normalization as a user-callable helper, so a caller
    // can reproduce the selector's L2 / ZSCORE preprocessing (then run under
    // NONE), or inspect the parameters. Copies; the inputs are untouched.
    m_tsm.def("normalize_design",
        [](Eigen::MatrixXd X, Eigen::VectorXd y, dn::ScalingMode scaling_mode, double eps,
           std::optional<int> n_threads) {
            dn::NormalizationParams params;
            dn::centerY(y, params, scaling_mode);
            dn::centerAndL2NormalizeX(X, params, eps, /*verbose=*/false, scaling_mode,
                                      n_threads.value_or(trex::utils::openmp::get_num_threads()));
            py::dict out;
            out["X"]        = X;
            out["y"]        = y;
            out["X_means"]  = params.X_means;
            out["X_scales"] = params.X_l2norms;
            out["y_mean"]   = params.y_mean;
            return out;
        },
        py::arg("X"), py::arg("y"),
        py::arg("scaling_mode") = dn::ScalingMode::L2,
        py::arg("eps") = 1e-6,
        py::arg("n_threads") = py::none(),
        "Apply the selectors' data normalization to copies of X and y and return a dict with "
        "keys 'X', 'y' (the normalized copies), 'X_means', 'X_scales' (the per-column divisors: "
        "L2 norms for L2, sample SDs for ZSCORE, ones for NONE) and 'y_mean'. Under "
        "ScalingMode.NONE the copies equal the inputs and the parameters are the identity. "
        "eps is the dead-column threshold on the working L2 norm (the selectors use the solver "
        "tolerance, default 1e-6). n_threads is the thread budget of the column passes (None: "
        "the library's default budget, utils.get_num_threads(), initially 1). Raises ValueError "
        "on a non-finite entry or a dead column (constant under L2 / ZSCORE, all-zero under "
        "NONE), naming the column (0-based).");

    // Bind LLoopStrategy enum
    py::enum_<LLoopStrategy>(m_tsm, "LLoopStrategy", "L-loop strategy controlling how dummy variables are constructed "
        "in each experiment.")
        .value("SKIPL", LLoopStrategy::SKIPL, "Stored, one-shot: fixed dummies = max_dummy_multiplier * p.")
        .value("STANDARD", LLoopStrategy::STANDARD, "Stored; fresh dummies at each L-loop iteration (conservative).")
        .value("HCONCAT", LLoopStrategy::HCONCAT, "Stored; horizontally expand dummy matrices across L (faster, same "
            "result).")
        .value("PERMUTATION", LLoopStrategy::PERMUTATION, "Stored base dummy matrix + deterministic row permutations "
            "per experiment.")
        .value("PERMUTATION_SEEDED", LLoopStrategy::PERMUTATION_SEEDED, "Seed-derived base + row permutations per "
            "experiment; nothing stored.")
        .value("SEEDED", LLoopStrategy::SEEDED, "The storage-free twin of STANDARD: fresh dummies at every L step, "
            "re-derived from the seed per experiment and bit-identical to STANDARD for the same seed; nothing stored.")
        .export_values();

    // Bind SolverTypeForTRex enum
    py::enum_<disp::SolverTypeForTRex>(m_tsm, "SolverTypeForTRex", "Solver algorithm variants available for T-Rex.")
        .value("TLARS",      disp::SolverTypeForTRex::TLARS,      "Terminating LARS.")
        .value("TLASSO",     disp::SolverTypeForTRex::TLASSO,     "Terminating LASSO.")
        .value("TENET",      disp::SolverTypeForTRex::TENET,      "Terminating Elastic Net.")
        .value("TSTEPWISE",  disp::SolverTypeForTRex::TSTEPWISE,  "Terminating Stepwise.")
        .value("TSTAGEWISE", disp::SolverTypeForTRex::TSTAGEWISE, "Terminating Stagewise.")
        .value("TOMP",       disp::SolverTypeForTRex::TOMP,       "Terminating OMP.")
        .value("TGP",        disp::SolverTypeForTRex::TGP,        "Terminating Gradient Pursuit.")
        .value("TACGP",      disp::SolverTypeForTRex::TACGP,      "Terminating Approximate Conjugate Gradient Pursuit.")
        .value("TMP",        disp::SolverTypeForTRex::TMP,        "Terminating Matching Pursuit.")
        .value("TNCGMP",     disp::SolverTypeForTRex::TNCGMP,     "Terminating Norm-Corrected Generalized Matching "
            "Pursuit.")
        .value("TOOLS",      disp::SolverTypeForTRex::TOOLS,      "Terminating Orthogonal Least Squares.")
        .value("TAFS",       disp::SolverTypeForTRex::TAFS,       "Terminating AFS.")
        .value("TENET_AUG",  disp::SolverTypeForTRex::TENET_AUG,  "Terminating Elastic Net via augmented LASSO; "
            "accepted wherever TENET runs (LASSO inner path standard, LARS via tenet_aug_use_lars).")
        .value("TCDLASSO",   disp::SolverTypeForTRex::TCDLASSO,   "Terminating coordinate-descent LASSO (exact "
            "penalized-lasso minimizers per crossing; certified crossing localization by default).")
        .value("TCDENET",    disp::SolverTypeForTRex::TCDENET,    "Terminating coordinate-descent Elastic Net.")
        .value("TIENET",     disp::SolverTypeForTRex::TIENET,     "Terminating Informed Elastic Net, native pathwise "
            "(GVS-only, gvs_type=IEN; IEN default).")
        .value("TIENET_AUG", disp::SolverTypeForTRex::TIENET_AUG, "Terminating Informed Elastic Net via augmented "
            "LASSO "
            "(GVS-only, gvs_type=IEN).")
        .value("TCDIENET",   disp::SolverTypeForTRex::TCDIENET,   "Terminating coordinate-descent Informed Elastic Net "
            "(GVS-only, gvs_type=IEN).")
        .value("TCDTIKNET",  disp::SolverTypeForTRex::TCDTIKNET,  "Terminating coordinate-descent Tikhonov Net "
            "(sparse-K; needs a penalty matrix, TRexTikhonovSelector only — not yet bound in Python).")
        .value("TIKNET",     disp::SolverTypeForTRex::TIKNET,     "Terminating Tikhonov Net, native pathwise "
            "(sparse-K, "
            "FOLDED; TRexTikhonovSelector only).")
        .export_values();

    // Bind SolverHyperparameters struct
    py::class_<disp::SolverHyperparameters>(m_tsm, "SolverHyperparameters", "Hyperparameters for the selected T-Rex "
        "solver.")
        .def(py::init<>())
        .def_readwrite("lambda2",       &disp::SolverHyperparameters::lambda2,       "L2 penalty for ENET-type "
            "solvers.")
        .def_readwrite("rho_afs",       &disp::SolverHyperparameters::rho_afs,       "Shrinkage step size for AFS "
            "solver in (0, 1].")
        .def_readwrite("ncgmp_variant", &disp::SolverHyperparameters::ncgmp_variant, "NCGMP variant: 0 = LineSearch, 1 "
            "= FullyCorrective.")
        .def_readwrite("tol",           &disp::SolverHyperparameters::tol,           "Numerical tolerance for solver "
            "steps.")
        .def_readwrite("exch_tie_alpha", &disp::SolverHyperparameters::exch_tie_alpha, "Exchangeable-tie band width "
            "for "
            "greedy solvers (TOMP/TAFS) in pairwise ranking-noise sd units; 0 = off. Recommended under trex+DA: 0.25.")
        .def_readwrite("exch_tie_floor", &disp::SolverHyperparameters::exch_tie_floor, "Minimum |correlation| for "
            "exchangeable-tie candidates in (0, 1); ignored unless exch_tie_alpha > 0.")
        .def_readwrite("tenet_aug_use_lars", &disp::SolverHyperparameters::tenet_aug_use_lars, "TENET_AUG: False "
            "(default) = standard LASSO inner path; True = optional pure-LARS inner path that never drops variables.")
        .def_readwrite("cd_lambda_rel_tol", &disp::SolverHyperparameters::cd_lambda_rel_tol, "CCD family: relative "
            "lambda step tolerance of the crossing search (<= 0 keeps the solver default).")
        .def_readwrite("cd_tol_certify", &disp::SolverHyperparameters::cd_tol_certify, "CCD family: KKT certification "
            "tolerance; applied only when set together with cd_tol_probe (both > 0).")
        .def_readwrite("cd_tol_probe", &disp::SolverHyperparameters::cd_tol_probe, "CCD family: probing tolerance; "
            "applied only when set together with cd_tol_certify (both > 0).")
        .def_readwrite("cd_gram_cap", &disp::SolverHyperparameters::cd_gram_cap, "CCD family: column cap for the "
            "cached "
            "Gram matrix (0 keeps the solver default).")
        .def_readwrite("cd_max_sweeps", &disp::SolverHyperparameters::cd_max_sweeps, "CCD family: maximum coordinate "
            "sweeps per subproblem (0 keeps the solver default).")
        .def_readwrite("cd_certified_localization", &disp::SolverHyperparameters::cd_certified_localization, "CCD "
            "family: True (default) = certified crossing localization (duality-gap and coordinate-status certificates, "
            "closed-form continuation, records at the knot); False = LEGACY probe bisection with the support-nesting "
            "stop.");

    // Bind DummyDistribution (wraps C++ dummygen::Distribution)
    py::class_<dg::Distribution> dummy_dist(m_tsm, "DummyDistribution",
        "Specification of the dummy variable distribution injected by T-Rex.\n"
        "\n"
        "References\n"
        "----------\n"
        "J. Machkour, M. Muma, and D. P. Palomar, \"The Terminating-Random Experiments Selector:\n"
        "  Fast High-Dimensional Variable Selection with False Discovery Rate Control,\" Signal\n"
        "  Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.\n"
        "E. Mammen, \"Bootstrap and Wild Bootstrap for High Dimensional Linear Models,\" Ann.\n"
        "  Statist., vol. 21, no. 1, pp. 255-285, 1993, doi: 10.1214/aos/1176349025.\n");

    py::enum_<dg::Distribution::Type>(dummy_dist, "Type",
        "Dummy distribution type enumeration.")
        .value("Normal",                      dg::Distribution::Type::Normal)
        .value("Uniform",                     dg::Distribution::Type::Uniform)
        .value("Rademacher",                  dg::Distribution::Type::Rademacher)
        .value("StudentT",                    dg::Distribution::Type::StudentT)
        .value("Laplace",                     dg::Distribution::Type::Laplace)
        .value("Gumbel",                      dg::Distribution::Type::Gumbel)
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
    py::class_<TRexControlParameter>(m_tsm, "TRexControlParameter", "Control parameters for the core T-Rex selection "
        "algorithm.")
        .def(py::init<>())
        .def_readwrite("K", &TRexControlParameter::K, "Number of random simulation experiments to run over variable "
            "inclusions.")
        .def_readwrite("max_dummy_multiplier", &TRexControlParameter::max_dummy_multiplier, "Maximum dummy multiplier "
            "L "
            "(default 10).")
        .def_readwrite("use_max_T_stop", &TRexControlParameter::use_max_T_stop, "If true, always run solvers to T_stop "
            "regardless of early stopping.")
        .def_readwrite("opt_threshold", &TRexControlParameter::opt_threshold, "Voting threshold parameter (default "
            "0.75).")
        .def_readwrite("lloop_strategy", &TRexControlParameter::lloop_strategy, "L-loop strategy for constructing "
            "dummy "
            "variables.")
        .def_readwrite("tloop_stagnation_stop", &TRexControlParameter::tloop_stagnation_stop,
                       "T-loop early stopping on support-set stagnation. None (default) resolves "
                       "by solver family: disabled for equiangular LARS-path solvers "
                       "(TLARS/TLASSO/TENET/TENET_AUG, matching the R reference), enabled for "
                       "greedy solvers (guards their noise trap). Set True/False to override.")
        .def_readwrite("tloop_max_stagnant_steps", &TRexControlParameter::tloop_max_stagnant_steps, "Maximum "
            "consecutive stagnant T-loop steps before early stopping.")
        .def_readwrite("parallel_rnd_experiments", &TRexControlParameter::parallel_rnd_experiments,
                       "Run the K random experiments in parallel (default False: in sequence). Inside a "
                       "parallel run every solver runs serially; the lean L-loop strategies (SEEDED, "
                       "PERMUTATION, PERMUTATION_SEEDED) always run their experiments in sequence.")
        .def_readwrite("use_memory_mapping",  &TRexControlParameter::use_memory_mapping,  "Use out-of-core "
            "memory-mapped dummy matrix.")
        .def_readwrite("max_outer_threads",   &TRexControlParameter::max_outer_threads,
                       "Threads of the parallel K-loop (default 1; a whole number of at least 1), capped at "
                       "K. Ignored while parallel_rnd_experiments is False.")
        .def_readwrite("max_inner_threads",   &TRexControlParameter::max_inner_threads,
                       "Thread budget of the solvers' own passes over the design (default 1, serial; a whole "
                       "number of at least 1). Live when the experiments run in sequence; inside a parallel "
                       "run the solvers stay serial. A pass splits only above about a million entries.")
        .def_readwrite("solver_type",         &TRexControlParameter::solver_type,         "Solver algorithm to use for "
            "T-Rex.")
        .def_readwrite("solver_params",       &TRexControlParameter::solver_params,       "Hyperparameters for the "
            "selected solver.")
        .def_readwrite("scaling_mode",        &TRexControlParameter::scaling_mode,        "Working-design convention: "
            "L2 (default) or ZSCORE center X and y and scale X; NONE leaves X and y as supplied (user-managed; dummies "
            "keep the unit-L2 standard). Non-finite entries and dead columns are rejected in every mode (ValueError).")
        .def_readwrite("dummy_distribution",  &TRexControlParameter::dummy_distribution,  "Distribution used to "
            "generate dummy variables.");

    // Bind TRexSelector::SelectionResult
    py::class_<TRexSelector::SelectionResult>(m_tsm, "SelectionResult", "Return value of TRexSelector.select(): "
        "selected variables and diagnostic matrices.")
        .def_readonly("selected_var", &TRexSelector::SelectionResult::selected_var, "Selection indicator vector "
            "(length "
            "p; 1 = selected, 0 = not selected).")
        .def_readonly("v_thresh", &TRexSelector::SelectionResult::v_thresh, "Voting threshold v applied to control the "
            "FDR.")
        .def_readonly("r_mat", &TRexSelector::SelectionResult::R_mat, "R matrix of empirical tail probabilities (p x "
            "T_stop).")
        .def_readonly("T_stop", &TRexSelector::SelectionResult::T_stop, "T_stop: maximum number of LARS steps run.")
        .def_readonly("num_dummies", &TRexSelector::SelectionResult::num_dummies, "Total number of dummy variables "
            "used.")
        .def_readonly("dummy_factor_L", &TRexSelector::SelectionResult::dummy_factor_L, "Dummy multiplier L used.")
        .def_readonly("fdp_hat_mat", &TRexSelector::SelectionResult::FDP_hat_mat, "FDP hat matrix (p x |voting_grid|).")
        .def_readonly("phi_mat", &TRexSelector::SelectionResult::Phi_mat, "Phi matrix: relative occurrence counts per "
            "predictor.")
        .def_readonly("phi_prime", &TRexSelector::SelectionResult::Phi_prime, "Smoothed selection probability vector "
            "phi'.")
        .def_readonly("voting_grid", &TRexSelector::SelectionResult::voting_grid, "Voting grid: per-candidate vote "
            "totals across K experiments.");

    // Bind PyTRexSelector
    py::class_<PyTRexSelector>(m_tsm, "TRexSelector",
        "T-Rex Selector: FDR-controlled variable selection via dummy comparison.\n"
        "\n"
        "References\n"
        "----------\n"
        "J. Machkour, M. Muma, and D. P. Palomar, \"The Terminating-Random Experiments Selector:\n"
        "  Fast High-Dimensional Variable Selection with False Discovery Rate Control,\" Signal\n"
        "  Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.\n")
        .def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::VectorXd>, double, const TRexControlParameter&,
            int, bool>(),
             py::arg("X"), py::arg("y"), py::arg("tFDR") = 0.1,
             py::arg("trex_control") = TRexControlParameter(),
             py::arg("seed") = -1, py::arg("verbose") = true,
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
             "Construct the selector. X and y are accessed zero-copy via Eigen::Map.")
        .def("getNumDummies", &PyTRexSelector::getNumDummies,
             "Number of dummies used by the last select() (the L-loop's calibration).")
        .def("restore", &PyTRexSelector::restore,
             "Give X back before select(): undo the in-place normalization applied at "
             "construction and release the buffer claim (idempotent; a no-op after "
             "select(), which restores X itself). A restored selector cannot run: "
             "select() then raises, construct a new one. Without it, an unselected "
             "selector gives X back only when it is garbage-collected.")
        .def("select", &PyTRexSelector::select, py::call_guard<py::gil_scoped_release>(),
             "Run the T-Rex algorithm and return a SelectionResult.")
        .def("getTFDR", &PyTRexSelector::getTFDR, "Return the target FDR.")
        .def("getK", &PyTRexSelector::getK, "Return K: the number of random experiments.")
        .def("getMaxNumDummies", &PyTRexSelector::getMaxNumDummies, "Return the maximum number of dummy variables per "
            "L "
            "level.")
        .def("getMaxTStop", &PyTRexSelector::getMaxTStop, "Return whether use_max_T_stop is active.")
        .def("getStagnationCheck", &PyTRexSelector::getStagnationCheck, "Return whether T-loop stagnation stopping is "
            "active.")
        .def("getXMeans", &PyTRexSelector::getXMeans, "Return the column means of X.")
        .def("getXL2Norms", &PyTRexSelector::getXL2Norms, "Return the column-wise L2 norms of X after centering.")
        .def("getYMean", &PyTRexSelector::getYMean, "Return the mean of y.")
        .def("getSelectedVar", &PyTRexSelector::getSelectedVar, "Return the binary selection indicator vector (1 = "
            "selected).")
        .def("getSelectedIndices", &PyTRexSelector::getSelectedIndices, "Return 0-based indices of selected "
            "predictors.")
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
