// =====================================================================================
// tsolver_bindings.hpp - Python bindings for T-Solvers
// =====================================================================================
#ifndef TREX_PYTHON_TSOLVER_BINDINGS_HPP
#define TREX_PYTHON_TSOLVER_BINDINGS_HPP
// =====================================================================================
/**
 * @file tsolver_bindings.hpp
 *
 * @brief Pybind11 bindings for T-Solvers used in the T-Rex Selector package.
 */
// =====================================================================================

 // pybind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

// T-Solvers includes
// LARS-based solvers
#include <tsolvers/linear_model/lars_based/tlars_solver.hpp>
#include <tsolvers/linear_model/lars_based/tlasso_solver.hpp>
#include <tsolvers/linear_model/lars_based/tstepwise_solver.hpp>
#include <tsolvers/linear_model/lars_based/tstagewise_solver.hpp>
#include <tsolvers/linear_model/lars_based/tenet_solver.hpp>
#include <tsolvers/linear_model/lars_based/tenet_aug_solver.hpp>
#include <tsolvers/linear_model/lars_based/tienet_aug_solver.hpp>

// OMP-based solvers
#include <tsolvers/linear_model/omp_based/tomp_solver.hpp>
#include <tsolvers/linear_model/omp_based/tgp_solver.hpp>
#include <tsolvers/linear_model/omp_based/tacgp_solver.hpp>
#include <tsolvers/linear_model/omp_based/tmp_solver.hpp>
#include <tsolvers/linear_model/omp_based/tncgmp_solver.hpp>
#include <tsolvers/linear_model/omp_based/tools_solver.hpp>

// AFS-based solvers
#include <tsolvers/linear_model/afs_based/tafs_solver.hpp>

// =====================================================================================

namespace py = pybind11;

/**
 * @brief Python wrapper class for T-Solvers.
 *
 * @details Retains references to the underlying numpy arrays via Eigen::Map without performing
 *          unnecessary deep copies of the data. The wrapper operates zero-copy directly on the
 *          Python buffers.
 *
 * @tparam Solver The specific underlying T-Solver class from the C++ library.
 */
template <typename Solver>
class PySolverWrapper {
protected:

    /** @brief Map for the design matrix (X) */
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>> X_map_;

    /** @brief Map for the dummy variable matrix (D) */
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>> D_map_;

    /** @brief Map for the response vector (y) */
    std::unique_ptr<Eigen::Map<Eigen::VectorXd>> y_map_;

    /** @brief Pointer to the underlying solver instance */
    std::unique_ptr<Solver> solver_;

public:
    /**
     * @brief Construct a new Py Solver Wrapper object utilizing zero-copy mapping.
     *
     * @tparam Args Variadic template parameters for the specific solver's constructor.
     *
     * @param X Reference to the design matrix (n × p).
     * @param D Reference to the dummy variable matrix (n × num_dummies).
     * @param y Reference to the response vector (n × 1).
     */
    template <typename... Args>
    PySolverWrapper(Eigen::Ref<Eigen::MatrixXd> X,
                    Eigen::Ref<Eigen::MatrixXd> D,
                    Eigen::Ref<Eigen::VectorXd> y,
                    Args&&... args)
    {
        X_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X.data(),
                                                               X.rows(),
                                                               X.cols());
        D_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(D.data(),
                                                               D.rows(),
                                                               D.cols());
        y_map_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(y.data(), y.size());

        solver_ = std::make_unique<Solver>(*X_map_, *D_map_, *y_map_, std::forward<Args>(args)...);
    }

    /**
     * @brief Execute the solver algorithm.
     *
     * @details Computes the solution path by iteratively including variables into the active set.
     *          If `T_stop` is set to 0 (or -1 in Python, mapping to max unsigned integer), the
     *          solver calculates the full solution path until all possible variables are included
     *          or the constraints prevent further additions.
     *
     * @param T_stop The maximum number of dummy variables in the active set.
     *               Pass 0 or -1 to calculate the full solution path.
     * @param early_stop Whether to break execution early immediately upon selecting `T_stop`
     *                   dummy variables. Default is true.
     */
    void executeStep(std::size_t T_stop = 0, bool early_stop = true) {
        solver_->executeStep(T_stop, early_stop);
    }

    /**
     * @brief Get the regression coefficients (Beta) for a specific step.
     *
     * @param step Specific path step to retrieve Beta for. Default is -1, returning the
     *             coefficients from the final computed step.
     *
     * @return Eigen::VectorXd containing the coefficients.
     */
    Eigen::VectorXd getBeta(int step = -1) const { return solver_->getBeta(step); }

    /**
     * @brief Get the intercept for a specific step.
     *
     * @param step Specific path step to retrieve the intercept for. Default is -1, returning
     *             the intercept from the final computed step.
     *
     * @return double representing the intercept value.
     */
    double getIntercept(int step = -1) const { return solver_->getIntercept(step); }

    /**
     * @brief Retrieve the full path history of the Beta coefficients.
     *
     * @return Eigen::MatrixXd where each column represents the Beta coefficient state at a
     *         specific step in the sequential path computation.
     */
    Eigen::MatrixXd getBetaPath() const { return solver_->getBetaPath(); }

    /**
     * @brief Retrieve the string identifier for the underlying T-Solver class used.
     *
     * @return std::string representation of the solver engine.
     */
    std::string solverTypeToString() const { return solver_->solverTypeToString(); }

    // =================================================================================
    // API Functionality Expansions (Metrics, Path Info, Diagnostics, and State Mgmt)
    // =================================================================================

    /** @brief Get the estimated Mallow's Cp values at each step. */
    std::vector<double> getCp() const { return solver_->getCp(); }

    /** @brief Get the estimated Mallow's Cp values at each step with provided sigma squared. */
    std::vector<double> getCpWithSigma(double sigma_hat_sq) const {
        return static_cast<const trex::tsolvers::TSolver_Base*>(solver_.get())->getCp(sigma_hat_sq);
    }

    /** @brief Get the residual sum of squares at each step. */
    const std::vector<double>& getRSS() const { return solver_->getRSS(); }

    /** @brief Get the coefficient of determination (R^2) at each step. */
    const std::vector<double>& getR2() const { return solver_->getR2(); }

    /** @brief Get the degrees of freedom at each step. */
    const std::vector<std::size_t>& getDoF() const { return solver_->getDoF(); }

    /** @brief Get the residual vector at the current step. */
    Eigen::VectorXd getResiduals() const { return solver_->getResiduals(); }

    /** @brief Get the integer codes for actions taken at each step (e.g., param addition/drop). */
    const std::vector<std::vector<int>>& getActions() const { return solver_->getActions(); }

    /** @brief Get indices of variables that are active at the current step. */
    const std::vector<std::size_t>& getActives() const { return solver_->getActives(); }

    /** @brief Get indices of variables that are inactive at the current step. */
    const std::vector<std::size_t>& getInactives() const { return solver_->getInactives(); }

    /** @brief Get indices of variables that were dropped during the execution. */
    const std::vector<std::size_t>& getDroppedIndices() const { return solver_->getDroppedIndices(); }

    /** @brief Get indices of active predictor (non-dummy) variables at the current step. */
    std::vector<std::size_t> getActivePredictorIndices() const {
        return solver_->getActivePredictorIndices();
    }

    /** @brief Get indices of active dummy variables at the current step. */
    std::vector<std::size_t> getActiveDummyIndices() const {
        return solver_->getActiveDummyIndices();
    }

    /** @brief Calculate how many variables are currently active. */
    std::size_t getNumActives() const { return solver_->getNumActives(); }

    /** @brief Find where dummy variables start within the merged predictor index space. */
    std::size_t getDummyStartIndex() const { return solver_->getDummyStartIndex(); }

    /** @brief Quantify the number of path steps successfully taken thus far. */
    std::size_t getNumSteps() const { return solver_->getNumSteps(); }

    /** @brief Check whether the solver has mapped access to data predictors. */
    bool isConnected() const { return solver_->isConnected(); }

    /** @brief Set the numerical tolerance used for internal numeric comparisons. */
    void setTolerance(double tol) { solver_->setTolerance(tol); }

    /** @brief Set the random seed for reproducible tie-breaking among dummies. */
    void setTieSeed(uint32_t seed) { solver_->setTieSeed(seed); }

    /** @brief Get all recorded warnings (dropped columns, collinear rejections, ...). */
    const std::vector<std::string>& getWarnings() const { return solver_->getWarnings(); }

    /** @brief Clear the recorded warnings. */
    void clearWarnings() { solver_->clearWarnings(); }

    // =================================================================================
    // Solver-Specific Getters (only valid for the corresponding solver type)
    // =================================================================================

    /** @brief Get regularization parameter sequence (TLARS-specific). */
    const std::vector<double>& getLambda() const { return solver_->getLambda(); }

    /** @brief Get the current Kahan summation refresh interval (TLARS-specific). */
    std::size_t getKahanRefreshInterval() const { return solver_->getKahanRefreshInterval(); }

    /** @brief Set the Kahan summation refresh interval (TLARS-specific). */
    void setKahanRefreshInterval(std::size_t interval) { solver_->setKahanRefreshInterval(interval); }

    /** @brief Get total number of variable removals (TLASSO/TSTAGEWISE/TENET-specific). */
    std::size_t getNumRemovals() const { return solver_->getNumRemovals(); }

    /** @brief Get cycling ratio of removals to additions (TLASSO/TSTAGEWISE/TENET-specific). */
    double getCyclingRatio() const { return solver_->getCyclingRatio(); }

    /** @brief Get the Elastic Net mixing parameter alpha (TENET-specific). */
    double getAlpha() const { return solver_->getAlpha(); }

    /** @brief Get the secondary L2/group-ridge penalty lambda2 (TIENETAug-specific). */
    double getLambda2() const { return solver_->getLambda2(); }

    /** @brief Get the 0-based group id per original variable (TIENETAug-specific). */
    Eigen::VectorXi getGroups() const { return solver_->getGroups(); }

    /** @brief Get the number of disjoint groups (TIENETAug-specific). */
    std::size_t getNumGroups() const { return solver_->getNumGroups(); }

    // =================================================================================
    // Structural Bindings (State reconnects and Cereal Serialization)
    // =================================================================================

    /**
     * @brief Reconnect the underlying native solver engine to a new batch of underlying Maps.
     */
    void reconnect(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<Eigen::MatrixXd> D) {
        X_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X.data(),
                                                              X.rows(),
                                                               X.cols());

        D_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(D.data(),
                                                               D.rows(),
                                                               D.cols());

        solver_->reconnect(*X_map_, *D_map_);
    }

    /**
     * @brief Restore solver state parameters based on fresh target views.
     */
    void restore(Eigen::Ref<Eigen::MatrixXd> X,
                 Eigen::Ref<Eigen::MatrixXd> D,
                 Eigen::Ref<Eigen::VectorXd> y) {
        X_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X.data(),
                                                               X.rows(),
                                                               X.cols());

        D_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(D.data(),
                                                               D.rows(),
                                                               D.cols());

        y_map_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(y.data(), y.size());

        solver_->restore(*X_map_, *D_map_, *y_map_);
    }

    /**
     * @brief Serialize calculation state directly to disk utilizing Cereal.
     */
    void save(const std::string& filepath) const {
        solver_->save(filepath);
    }

    /**
     * @brief Hydrate a solver explicitly from a pre-calculated Cereal output file on disk.
     */
    void load(const std::string& filepath) {
        solver_ = std::make_unique<Solver>(Solver::load(filepath, *X_map_, *D_map_));
    }
};


/**
 * @brief Attach the method set shared by every T-Solver to an existing class binding.
 *
 * @details Binds all TSolver_Base-backed methods (path execution, coefficient and
 *          diagnostics getters, state management, and Cereal serialization). The
 *          constructor is intentionally NOT bound here — each solver attaches the
 *          init matching its own C++ constructor signature.
 *
 * @tparam Solver The specific underlying T-Solver class being wrapped.
 *
 * @param cls The Pybind11 class the methods are attached to.
 */
template<typename Solver>
py::class_<PySolverWrapper<Solver>>& add_standard_tsolver_methods(
    py::class_<PySolverWrapper<Solver>>& cls) {
    cls.def("executeStep", &PySolverWrapper<Solver>::executeStep,
             py::arg("T_stop") = 0,
             py::arg("early_stop") = true,
             "Executes the underlying algorithmic solve mechanics up to constraint `T_stop`."
             " Pass 0 or -1 to calculate the full sequential path.")
        .def("getBeta", &PySolverWrapper<Solver>::getBeta, py::arg("step") = -1,
             "Retrieves the active regression coefficients (Beta) for a specific step.")
        .def("getIntercept", &PySolverWrapper<Solver>::getIntercept, py::arg("step") = -1,
             "Retrieves the calculated model intercept for a specific step.")
        .def("getBetaPath", &PySolverWrapper<Solver>::getBetaPath,
             "Retrieves a historical path history block containing all calculated Beta coefficients"
             " across execution steps.")
        .def("solverTypeToString", &PySolverWrapper<Solver>::solverTypeToString,
             "Returns a string descriptor mapping for the underlying wrapped T-Solver.")
        .def("getCp", &PySolverWrapper<Solver>::getCp,
             "Get the estimated Mallow's Cp values at each step.")
        .def("getCpWithSigma", &PySolverWrapper<Solver>::getCpWithSigma, py::arg("sigma_hat_sq"),
             "Get the estimated Mallow's Cp values at each step with provided sigma squared.")
        .def("getRSS", &PySolverWrapper<Solver>::getRSS,
             "Get the residual sum of squares at each step.")
        .def("getR2", &PySolverWrapper<Solver>::getR2,
             "Get the coefficient of determination (R^2) at each step.")
        .def("getDoF", &PySolverWrapper<Solver>::getDoF,
             "Get the degrees of freedom at each step.")
        .def("getResiduals", &PySolverWrapper<Solver>::getResiduals,
             "Get the residual vector at the current step.")
        .def("getActions", &PySolverWrapper<Solver>::getActions,
             "Get the integer codes for actions taken at each step. Entries "
             "follow the R lars convention: 1-based signed indices, +(j+1) "
             "for the addition and -(j+1) for the removal of 0-based "
             "variable j (decode with abs(a) - 1).")
        .def("getActives", &PySolverWrapper<Solver>::getActives,
             "Get indices of variables that are active at the current step.")
        .def("getInactives", &PySolverWrapper<Solver>::getInactives,
             "Get indices of variables that are inactive at the current step.")
        .def("getDroppedIndices", &PySolverWrapper<Solver>::getDroppedIndices,
             "Get indices of variables that were dropped during the execution.")
        .def("getActivePredictorIndices", &PySolverWrapper<Solver>::getActivePredictorIndices,
             "Get indices of active predictor (non-dummy) variables at the current step.")
        .def("getActiveDummyIndices", &PySolverWrapper<Solver>::getActiveDummyIndices,
             "Get indices of active dummy variables at the current step.")
        .def("getNumActives", &PySolverWrapper<Solver>::getNumActives,
             "Calculate how many variables are currently active.")
        .def("getDummyStartIndex", &PySolverWrapper<Solver>::getDummyStartIndex,
             "Find where dummy variables start within the merged predictor index space.")
        .def("getNumSteps", &PySolverWrapper<Solver>::getNumSteps,
             "Quantify the number of path steps successfully taken thus far.")
        .def("isConnected", &PySolverWrapper<Solver>::isConnected,
             "Check whether the solver has mapped access to data predictors.")
        .def("setTolerance", &PySolverWrapper<Solver>::setTolerance, py::arg("tol"),
             "Set the numerical tolerance used for internal numeric comparisons.")
        .def("setTieSeed", &PySolverWrapper<Solver>::setTieSeed, py::arg("seed"),
             "Set the random seed for reproducible tie-breaking among dummy candidates.")
        .def("getWarnings", &PySolverWrapper<Solver>::getWarnings,
             "Get all recorded warnings (dropped columns, collinear rejections, ...).")
        .def("clearWarnings", &PySolverWrapper<Solver>::clearWarnings,
             "Clear the recorded warnings.")
        .def("reconnect", &PySolverWrapper<Solver>::reconnect, py::arg("X"), py::arg("D"),
             "Reconnect the underlying native solver engine to a new batch of underlying Matrices.")
        .def("restore", &PySolverWrapper<Solver>::restore, py::arg("X"), py::arg("D"),
             py::arg("y"),
             "Restore solver state parameters based on fresh target views.")
        .def("save", &PySolverWrapper<Solver>::save, py::arg("filepath"),
             "Serialize calculation state directly to disk utilizing Cereal.")
        .def("load", &PySolverWrapper<Solver>::load, py::arg("filepath"),
             "Hydrate a solver explicitly from a pre-calculated Cereal output file on disk.");
    return cls;
}


/**
 * @brief Standardized Pybind11 registration for typical T-Solvers.
 *
 * @details Binds the canonical `(X, D, y, normalize, intercept, verbose, scaling_mode)`
 *          constructor plus the shared method set. Solvers with a different constructor
 *          signature bind their own init and call add_standard_tsolver_methods().
 *
 * @tparam Solver The specific underlying T-Solver class being wrapped.
 *
 * @param m The Pybind11 sub-module where this solver will be exposed.
 * @param name The class name assigned in the Python wrapper space.
 */
template<typename Solver>
py::class_<PySolverWrapper<Solver>> bind_standard_tsolver(py::module& m, const std::string& name) {
    using ScalingMode = trex::tsolvers::solver_utils::preprocessing::ScalingMode;
    py::class_<PySolverWrapper<Solver>> cls(m, name.c_str());
    cls.def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>,
                      Eigen::Ref<Eigen::VectorXd>, bool, bool, bool, ScalingMode>(),
             py::arg("X"),
             py::arg("D"),
             py::arg("y"),
             py::arg("normalize") = true,
             py::arg("intercept") = true,
             py::arg("verbose") = false,
             py::arg("scaling_mode") = ScalingMode::L2,
             "Initializes the solver via zero-copy views to facilitate safe in-place mutations.");
    add_standard_tsolver_methods<Solver>(cls);
    return cls;
}


/**
 * @brief Binds all T-Solvers to the given Pybind11 module.
 *
 * @param m The Pybind11 module where the solvers will be exposed.
 */
inline void bind_tsolvers_module(py::module& m) {

    using ScalingMode = trex::tsolvers::solver_utils::preprocessing::ScalingMode;

    // Column-scaling convention used by all solver constructors. This is the
    // solver-side enum (solver_utils::preprocessing::ScalingMode); the
    // selector-level trex_selector_methods.ScalingMode mirrors it.
    py::enum_<ScalingMode>(m, "ScalingMode",
                           "Column scaling convention applied by the solver after centering.")
        .value("L2", ScalingMode::L2,
               "Divide each centered column by its L2 norm (unit L2 norm).")
        .value("ZSCORE", ScalingMode::ZSCORE,
               "Divide each centered column by its sample SD (n-1).")
        .export_values();

    // -------------------------------------------------------------------------
    // LARS-based sub-module
    // -------------------------------------------------------------------------

    py::module m_lars = m.def_submodule("lars_based", "LARS-based T-Solvers");
    using namespace trex::tsolvers::linear_model::lars_based;

    // TLARS: the C++ constructor carries the internal algorithm_type discriminator
    // before scaling_mode; pin it to TLARS and expose the canonical signature.
    py::class_<PySolverWrapper<TLARS_Solver>> cls_tlars(m_lars, "TLARS_Solver");
    cls_tlars.def(py::init([](Eigen::Ref<Eigen::MatrixXd> X,
                              Eigen::Ref<Eigen::MatrixXd> D,
                              Eigen::Ref<Eigen::VectorXd> y,
                              bool normalize, bool intercept, bool verbose,
                              ScalingMode scaling_mode) {
                     return std::make_unique<PySolverWrapper<TLARS_Solver>>(
                         X, D, y, normalize, intercept, verbose,
                         SolverTypeLarsBased::TLARS, scaling_mode);
                 }),
             py::arg("X"),
             py::arg("D"),
             py::arg("y"),
             py::arg("normalize") = true,
             py::arg("intercept") = true,
             py::arg("verbose") = false,
             py::arg("scaling_mode") = ScalingMode::L2,
             "Initializes the solver via zero-copy views to facilitate safe in-place mutations.");
    add_standard_tsolver_methods<TLARS_Solver>(cls_tlars);
    cls_tlars
        .def("getLambda", &PySolverWrapper<TLARS_Solver>::getLambda,
             "Get the regularization parameter sequence (max correlation per step).")
        .def("getKahanRefreshInterval", &PySolverWrapper<TLARS_Solver>::getKahanRefreshInterval,
             "Get the current Kahan summation refresh interval.")
        .def("setKahanRefreshInterval", &PySolverWrapper<TLARS_Solver>::setKahanRefreshInterval,
             py::arg("interval"),
             "Set the Kahan summation refresh interval (call before executeStep).");

    bind_standard_tsolver<TLASSO_Solver>(m_lars, "TLASSO_Solver")
        .def("getNumRemovals", &PySolverWrapper<TLASSO_Solver>::getNumRemovals,
             "Get the total number of variable removals during solution path.")
        .def("getCyclingRatio", &PySolverWrapper<TLASSO_Solver>::getCyclingRatio,
             "Get the cycling ratio (removals / additions).");

    bind_standard_tsolver<TSTEPWISE_Solver>(m_lars, "TSTEPWISE_Solver");

    bind_standard_tsolver<TSTAGEWISE_Solver>(m_lars, "TSTAGEWISE_Solver")
        .def("getNumRemovals", &PySolverWrapper<TSTAGEWISE_Solver>::getNumRemovals,
             "Get the total number of coefficient removals.")
        .def("getCyclingRatio", &PySolverWrapper<TSTAGEWISE_Solver>::getCyclingRatio,
             "Get the cycling ratio (removals / additions).");

    // TENET: constructor takes the secondary L2 penalty lambda2
    py::class_<PySolverWrapper<TENET_Solver>> cls_tenet(m_lars, "TENET_Solver");
    cls_tenet.def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>,
                           Eigen::Ref<Eigen::VectorXd>, double, bool, bool, bool, ScalingMode>(),
             py::arg("X"),
             py::arg("D"),
             py::arg("y"),
             py::arg("lambda2"),
             py::arg("normalize") = true,
             py::arg("intercept") = true,
             py::arg("verbose") = false,
             py::arg("scaling_mode") = ScalingMode::L2,
             "Initializes the TENET solver array, using an explicit secondary L2"
             " penalty `lambda2`.");
    add_standard_tsolver_methods<TENET_Solver>(cls_tenet);
    cls_tenet
        .def("getAlpha", &PySolverWrapper<TENET_Solver>::getAlpha,
             "Get the Elastic Net mixing parameter alpha implied by lambda2.")
        .def("getNumRemovals", &PySolverWrapper<TENET_Solver>::getNumRemovals,
             "Get the total number of variable removals during solution path.")
        .def("getCyclingRatio", &PySolverWrapper<TENET_Solver>::getCyclingRatio,
             "Get the cycling ratio (removals / additions).");

    // TENETAug: Elastic Net via augmented LASSO (constructor takes lambda2)
    py::class_<PySolverWrapper<TENETAug_Solver>> cls_tenet_aug(m_lars, "TENETAug_Solver");
    cls_tenet_aug.def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>,
                               Eigen::Ref<Eigen::VectorXd>, double, bool, bool, bool,
                               ScalingMode, bool>(),
             py::arg("X"),
             py::arg("D"),
             py::arg("y"),
             py::arg("lambda2"),
             py::arg("normalize") = true,
             py::arg("intercept") = true,
             py::arg("verbose") = false,
             py::arg("scaling_mode") = ScalingMode::L2,
             py::arg("use_lars_inner") = false,
             "Initializes the augmented-LASSO Elastic Net solver with an explicit"
             " secondary L2 penalty `lambda2`. Set `use_lars_inner` to run the"
             " inner path with a pure-LARS solver (never drops variables).");
    add_standard_tsolver_methods<TENETAug_Solver>(cls_tenet_aug);

    // TIENETAug: Informed Elastic Net via augmented LASSO (group-ridge penalty)
    py::class_<PySolverWrapper<TIENETAug_Solver>> cls_tienet_aug(m_lars, "TIENETAug_Solver");
    cls_tienet_aug.def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>,
                                Eigen::Ref<Eigen::VectorXd>, double, const Eigen::VectorXi&,
                                bool, bool, bool, ScalingMode, bool>(),
             py::arg("X"),
             py::arg("D"),
             py::arg("y"),
             py::arg("lambda2"),
             py::arg("groups"),
             py::arg("normalize") = true,
             py::arg("intercept") = true,
             py::arg("verbose") = false,
             py::arg("scaling_mode") = ScalingMode::L2,
             py::arg("use_lars_inner") = false,
             "Initializes the augmented-LASSO Informed Elastic Net solver with a"
             " group-ridge penalty `lambda2` and a 0-based group id per original"
             " variable (`groups`, length p).");
    add_standard_tsolver_methods<TIENETAug_Solver>(cls_tienet_aug);
    cls_tienet_aug
        .def("getLambda2", &PySolverWrapper<TIENETAug_Solver>::getLambda2,
             "Get the group-ridge penalty lambda2 used.")
        .def("getGroups", &PySolverWrapper<TIENETAug_Solver>::getGroups,
             "Get the 0-based group id per original variable (length p).")
        .def("getNumGroups", &PySolverWrapper<TIENETAug_Solver>::getNumGroups,
             "Get the number of disjoint groups.");

    // -------------------------------------------------------------------------
    // OMP-based sub-module
    // -------------------------------------------------------------------------

    py::module m_omp = m.def_submodule("omp_based", "OMP-based T-Solvers");
    using namespace trex::tsolvers::linear_model::omp_based;

    // TOMP: like TLARS, pin the internal algorithm_type discriminator
    py::class_<PySolverWrapper<TOMP_Solver>> cls_tomp(m_omp, "TOMP_Solver");
    cls_tomp.def(py::init([](Eigen::Ref<Eigen::MatrixXd> X,
                             Eigen::Ref<Eigen::MatrixXd> D,
                             Eigen::Ref<Eigen::VectorXd> y,
                             bool normalize, bool intercept, bool verbose,
                             ScalingMode scaling_mode) {
                    return std::make_unique<PySolverWrapper<TOMP_Solver>>(
                        X, D, y, normalize, intercept, verbose,
                        SolverTypeOMPBased::TOMP, scaling_mode);
                }),
             py::arg("X"),
             py::arg("D"),
             py::arg("y"),
             py::arg("normalize") = true,
             py::arg("intercept") = true,
             py::arg("verbose") = false,
             py::arg("scaling_mode") = ScalingMode::L2,
             "Initializes the solver via zero-copy views to facilitate safe in-place mutations.");
    add_standard_tsolver_methods<TOMP_Solver>(cls_tomp);

    bind_standard_tsolver<TGP_Solver>(m_omp, "TGP_Solver");
    bind_standard_tsolver<TACGP_Solver>(m_omp, "TACGP_Solver");
    bind_standard_tsolver<TMP_Solver>(m_omp, "TMP_Solver");
    bind_standard_tsolver<TOOLS_Solver>(m_omp, "TOOLS_Solver");

    /**
     * @brief Variants defining the internal optimization mode utilized by the TNCGMP solver.
     */
    py::enum_<NCGMPVariant>(m_omp, "NCGMPVariant")
        .value("LineSearch",
               NCGMPVariant::LineSearch,
               "Computes updates using standard backtracking / exact line search.")
        .value("FullyCorrective",
              NCGMPVariant::FullyCorrective,
              "Computes fully corrective projections for all active set members at each step.")
        .export_values();

    // TNCGMP requires variant
    py::class_<PySolverWrapper<TNCGMP_Solver>> cls_tncgmp(m_omp, "TNCGMP_Solver");
    cls_tncgmp.def(py::init<Eigen::Ref<Eigen::MatrixXd>,
                            Eigen::Ref<Eigen::MatrixXd>,
                            Eigen::Ref<Eigen::VectorXd>,
                            NCGMPVariant,
                            bool, bool, bool, ScalingMode>(),
             py::arg("X"),
             py::arg("D"),
             py::arg("y"),
             py::arg("variant") = NCGMPVariant::LineSearch,
             py::arg("normalize") = true,
             py::arg("intercept") = true,
             py::arg("verbose") = false,
             py::arg("scaling_mode") = ScalingMode::L2,
             "Initializes the TNCGMP solver array, enforcing a strictly specified LineSearch"
             " or FullyCorrective variant.");
    add_standard_tsolver_methods<TNCGMP_Solver>(cls_tncgmp);

    // -------------------------------------------------------------------------
    // AFS-based sub-module
    // -------------------------------------------------------------------------

    py::module m_afs = m.def_submodule("afs_based", "AFS-based T-Solvers");
    using namespace trex::tsolvers::linear_model::afs_based;

    py::class_<PySolverWrapper<TAFS_Solver>> cls_tafs(m_afs, "TAFS_Solver");
    cls_tafs.def(py::init<Eigen::Ref<Eigen::MatrixXd>,
                          Eigen::Ref<Eigen::MatrixXd>,
                          Eigen::Ref<Eigen::VectorXd>,
                          double, bool, bool, bool, ScalingMode>(),
             py::arg("X"),
             py::arg("D"),
             py::arg("y"),
             py::arg("rho") = 1.0,
             py::arg("normalize") = true,
             py::arg("intercept") = true,
             py::arg("verbose") = false,
             py::arg("scaling_mode") = ScalingMode::L2,
             "Initializes the AFS solver array allowing for custom `rho` tuning.");
    add_standard_tsolver_methods<TAFS_Solver>(cls_tafs);
}
// =====================================================================================
#endif /* End of TREX_PYTHON_TSOLVER_BINDINGS_HPP */
// =====================================================================================
