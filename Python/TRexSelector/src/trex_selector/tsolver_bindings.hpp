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

    /** @brief Calculate how many variables are currently active. */
    std::size_t getNumActives() const { return solver_->getNumActives(); }

    /** @brief Find where dummy variables start within the merged predictor index space. */
    std::size_t getDummyStartIndex() const { return solver_->getDummyStartIndex(); }

    /** @brief Quantify the number of path steps successfully taken thus far. */
    std::size_t getNumSteps() const { return solver_->getNumSteps(); }

    /** @brief Check whether the solver has mapped access to data predictors. */
    bool isConnected() const { return solver_->isConnected(); }

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
 * @brief Standardized Pybind11 registration for typical T-Solvers.
 *
 * @tparam Solver The specific underlying T-Solver class being wrapped.
 *
 * @param m The Pybind11 sub-module where this solver will be exposed.
 * @param name The class name assigned in the Python wrapper space.
 */
template<typename Solver>
void bind_standard_tsolver(py::module& m, const std::string& name) {
    py::class_<PySolverWrapper<Solver>>(m, name.c_str())
        .def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>,
                      Eigen::Ref<Eigen::VectorXd>, bool, bool, bool>(),
             py::arg("X"),
             py::arg("D"),
             py::arg("y"),
             py::arg("normalize") = true,
             py::arg("intercept") = true,
             py::arg("verbose") = false,
             "Initializes the solver via zero-copy views to facilitate safe in-place mutations.")
        .def("executeStep", &PySolverWrapper<Solver>::executeStep,
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
             "Get the integer codes for actions taken at each step.")
        .def("getActives", &PySolverWrapper<Solver>::getActives,
             "Get indices of variables that are active at the current step.")
        .def("getInactives", &PySolverWrapper<Solver>::getInactives,
             "Get indices of variables that are inactive at the current step.")
        .def("getDroppedIndices", &PySolverWrapper<Solver>::getDroppedIndices,
             "Get indices of variables that were dropped during the execution.")
        .def("getNumActives", &PySolverWrapper<Solver>::getNumActives,
             "Calculate how many variables are currently active.")
        .def("getDummyStartIndex", &PySolverWrapper<Solver>::getDummyStartIndex,
             "Find where dummy variables start within the merged predictor index space.")
        .def("getNumSteps", &PySolverWrapper<Solver>::getNumSteps,
             "Quantify the number of path steps successfully taken thus far.")
        .def("isConnected", &PySolverWrapper<Solver>::isConnected,
             "Check whether the solver has mapped access to data predictors.")
        .def("reconnect", &PySolverWrapper<Solver>::reconnect, py::arg("X"), py::arg("D"),
             "Reconnect the underlying native solver engine to a new batch of underlying Matrices.")
        .def("restore", &PySolverWrapper<Solver>::restore, py::arg("X"), py::arg("D"),
             py::arg("y"),
             "Restore solver state parameters based on fresh target views.")
        .def("save", &PySolverWrapper<Solver>::save, py::arg("filepath"),
             "Serialize calculation state directly to disk utilizing Cereal.")
        .def("load", &PySolverWrapper<Solver>::load, py::arg("filepath"),
             "Hydrate a solver explicitly from a pre-calculated Cereal output file on disk.");
}


/**
 * @brief Binds all T-Solvers to the given Pybind11 module.
 *
 * @param m The Pybind11 module where the solvers will be exposed.
 */
inline void bind_tsolvers_module(py::module& m) {

    // -------------------------------------------------------------------------
    // LARS-based sub-module
    // -------------------------------------------------------------------------

    py::module m_lars = m.def_submodule("lars_based", "LARS-based T-Solvers");
    using namespace trex::tsolvers::linear_model::lars_based;

    bind_standard_tsolver<TLARS_Solver>(m_lars, "TLarsSolver");
    bind_standard_tsolver<TLASSO_Solver>(m_lars, "TLassoSolver");
    bind_standard_tsolver<TSTEPWISE_Solver>(m_lars, "TStepwiseSolver");
    bind_standard_tsolver<TSTAGEWISE_Solver>(m_lars, "TStagewiseSolver");

    // TENET specifically has a constructor with lambda2
    py::class_<PySolverWrapper<TENET_Solver>>(m_lars, "TENET_Solver")
        .def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>,
            Eigen::Ref<Eigen::VectorXd>, double, bool, bool, bool>(),
             py::arg("X"),
             py::arg("D"),
             py::arg("y"),
             py::arg("lambda2"),
             py::arg("normalize") = true,
             py::arg("intercept") = true,
             py::arg("verbose") = false,
             "Initializes the TENET solver array, using an explicit secondary L2"
             " penalty `lambda2`.")
        .def("executeStep", &PySolverWrapper<TENET_Solver>::executeStep,
             py::arg("T_stop") = 0,
             py::arg("early_stop") = true,
             "Executes the underlying TENET solve mechanics up to constraint `T_stop`."
             " Pass 0 or -1 to calculate the full sequential path.")
        .def("getBeta", &PySolverWrapper<TENET_Solver>::getBeta,
            py::arg("step") = -1)
        .def("getIntercept", &PySolverWrapper<TENET_Solver>::getIntercept,
            py::arg("step") = -1)
        .def("getBetaPath", &PySolverWrapper<TENET_Solver>::getBetaPath)
        .def("solverTypeToString", &PySolverWrapper<TENET_Solver>::solverTypeToString)
        .def("getCp", &PySolverWrapper<TENET_Solver>::getCp)
        .def("getCpWithSigma", &PySolverWrapper<TENET_Solver>::getCpWithSigma,
                    py::arg("sigma_hat_sq"))
        .def("getRSS", &PySolverWrapper<TENET_Solver>::getRSS)
        .def("getR2", &PySolverWrapper<TENET_Solver>::getR2)
        .def("getDoF", &PySolverWrapper<TENET_Solver>::getDoF)
        .def("getResiduals", &PySolverWrapper<TENET_Solver>::getResiduals)
        .def("getActions", &PySolverWrapper<TENET_Solver>::getActions)
        .def("getActives", &PySolverWrapper<TENET_Solver>::getActives)
        .def("getInactives", &PySolverWrapper<TENET_Solver>::getInactives)
        .def("getDroppedIndices", &PySolverWrapper<TENET_Solver>::getDroppedIndices)
        .def("getNumActives", &PySolverWrapper<TENET_Solver>::getNumActives)
        .def("getDummyStartIndex", &PySolverWrapper<TENET_Solver>::getDummyStartIndex)
        .def("getNumSteps", &PySolverWrapper<TENET_Solver>::getNumSteps)
        .def("isConnected", &PySolverWrapper<TENET_Solver>::isConnected)
        .def("reconnect", &PySolverWrapper<TENET_Solver>::reconnect, py::arg("X"),
                    py::arg("D"))
        .def("restore", &PySolverWrapper<TENET_Solver>::restore, py::arg("X"),
                    py::arg("D"), py::arg("y"))
        .def("save", &PySolverWrapper<TENET_Solver>::save, py::arg("filepath"))
        .def("load", &PySolverWrapper<TENET_Solver>::load, py::arg("filepath"));


    // -------------------------------------------------------------------------
    // OMP-based sub-module
    // -------------------------------------------------------------------------

    py::module m_omp = m.def_submodule("omp_based", "OMP-based T-Solvers");
    using namespace trex::tsolvers::linear_model::omp_based;

    bind_standard_tsolver<TOMP_Solver>(m_omp, "TOMP_Solver");
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
    py::class_<PySolverWrapper<TNCGMP_Solver>>(m_omp, "TNCGMP_Solver")
        .def(py::init<Eigen::Ref<Eigen::MatrixXd>,
                            Eigen::Ref<Eigen::MatrixXd>,
                            Eigen::Ref<Eigen::VectorXd>,
                            NCGMPVariant,
                            bool, bool, bool>(),
             py::arg("X"),
             py::arg("D"),
             py::arg("y"),
             py::arg("variant") = NCGMPVariant::LineSearch,
             py::arg("normalize") = true,
             py::arg("intercept") = true,
             py::arg("verbose") = false,
             "Initializes the TNCGMP solver array, enforcing a strictly specified LineSearch"
             " or FullyCorrective variant.")
        .def("executeStep", &PySolverWrapper<TNCGMP_Solver>::executeStep,
             py::arg("T_stop") = 0, py::arg("early_stop") = true)
        .def("getBeta", &PySolverWrapper<TNCGMP_Solver>::getBeta,
                py::arg("step") = -1)
        .def("getIntercept", &PySolverWrapper<TNCGMP_Solver>::getIntercept,
                py::arg("step") = -1)
        .def("getBetaPath", &PySolverWrapper<TNCGMP_Solver>::getBetaPath)
        .def("solverTypeToString", &PySolverWrapper<TNCGMP_Solver>::solverTypeToString)
        .def("getCp", &PySolverWrapper<TNCGMP_Solver>::getCp)
        .def("getCpWithSigma", &PySolverWrapper<TNCGMP_Solver>::getCpWithSigma,
                    py::arg("sigma_hat_sq"))
        .def("getRSS", &PySolverWrapper<TNCGMP_Solver>::getRSS)
        .def("getR2", &PySolverWrapper<TNCGMP_Solver>::getR2)
        .def("getDoF", &PySolverWrapper<TNCGMP_Solver>::getDoF)
        .def("getResiduals", &PySolverWrapper<TNCGMP_Solver>::getResiduals)
        .def("getActions", &PySolverWrapper<TNCGMP_Solver>::getActions)
        .def("getActives", &PySolverWrapper<TNCGMP_Solver>::getActives)
        .def("getInactives", &PySolverWrapper<TNCGMP_Solver>::getInactives)
        .def("getDroppedIndices", &PySolverWrapper<TNCGMP_Solver>::getDroppedIndices)
        .def("getNumActives", &PySolverWrapper<TNCGMP_Solver>::getNumActives)
        .def("getDummyStartIndex", &PySolverWrapper<TNCGMP_Solver>::getDummyStartIndex)
        .def("getNumSteps", &PySolverWrapper<TNCGMP_Solver>::getNumSteps)
        .def("isConnected", &PySolverWrapper<TNCGMP_Solver>::isConnected)
        .def("reconnect", &PySolverWrapper<TNCGMP_Solver>::reconnect,
                    py::arg("X"), py::arg("D"))
        .def("restore", &PySolverWrapper<TNCGMP_Solver>::restore,
                    py::arg("X"), py::arg("D"),
                    py::arg("y"))
        .def("save", &PySolverWrapper<TNCGMP_Solver>::save, py::arg("filepath"))
        .def("load", &PySolverWrapper<TNCGMP_Solver>::load, py::arg("filepath"));


    // -------------------------------------------------------------------------
    // AFS-based sub-module
    // -------------------------------------------------------------------------

    py::module m_afs = m.def_submodule("afs_based", "AFS-based T-Solvers");
    using namespace trex::tsolvers::linear_model::afs_based;

    py::class_<PySolverWrapper<TAFS_Solver>>(m_afs, "TAFS_Solver")
        .def(py::init<Eigen::Ref<Eigen::MatrixXd>,
                            Eigen::Ref<Eigen::MatrixXd>,
                            Eigen::Ref<Eigen::VectorXd>,
                            double, bool, bool, bool>(),
             py::arg("X"),
             py::arg("D"),
             py::arg("y"),
             py::arg("rho") = 1.0,
             py::arg("normalize") = true,
             py::arg("intercept") = true,
             py::arg("verbose") = false,
             "Initializes the AFS solver array allowing for custom `rho` tuning.")
        .def("executeStep", &PySolverWrapper<TAFS_Solver>::executeStep,
             py::arg("T_stop") = 0, py::arg("early_stop") = true,
             "Executes the underlying AFS solve mechanics up to constraint `T_stop`."
              " Pass 0 or -1 to calculate the full sequential path.")
        .def("getBeta", &PySolverWrapper<TAFS_Solver>::getBeta,
            py::arg("step") = -1)
        .def("getIntercept", &PySolverWrapper<TAFS_Solver>::getIntercept,
            py::arg("step") = -1)
        .def("getBetaPath", &PySolverWrapper<TAFS_Solver>::getBetaPath)
        .def("solverTypeToString", &PySolverWrapper<TAFS_Solver>::solverTypeToString)
        .def("getCp", &PySolverWrapper<TAFS_Solver>::getCp)
        .def("getCpWithSigma", &PySolverWrapper<TAFS_Solver>::getCpWithSigma,
                    py::arg("sigma_hat_sq"))
        .def("getRSS", &PySolverWrapper<TAFS_Solver>::getRSS)
        .def("getR2", &PySolverWrapper<TAFS_Solver>::getR2)
        .def("getDoF", &PySolverWrapper<TAFS_Solver>::getDoF)
        .def("getResiduals", &PySolverWrapper<TAFS_Solver>::getResiduals)
        .def("getActions", &PySolverWrapper<TAFS_Solver>::getActions)
        .def("getActives", &PySolverWrapper<TAFS_Solver>::getActives)
        .def("getInactives", &PySolverWrapper<TAFS_Solver>::getInactives)
        .def("getDroppedIndices", &PySolverWrapper<TAFS_Solver>::getDroppedIndices)
        .def("getNumActives", &PySolverWrapper<TAFS_Solver>::getNumActives)
        .def("getDummyStartIndex", &PySolverWrapper<TAFS_Solver>::getDummyStartIndex)
        .def("getNumSteps", &PySolverWrapper<TAFS_Solver>::getNumSteps)
        .def("isConnected", &PySolverWrapper<TAFS_Solver>::isConnected)
        .def("reconnect", &PySolverWrapper<TAFS_Solver>::reconnect, py::arg("X"),
                    py::arg("D"))
        .def("restore", &PySolverWrapper<TAFS_Solver>::restore, py::arg("X"),
                    py::arg("D"),
                    py::arg("y"))
        .def("save", &PySolverWrapper<TAFS_Solver>::save, py::arg("filepath"))
        .def("load", &PySolverWrapper<TAFS_Solver>::load, py::arg("filepath"));
}
// =====================================================================================
#endif /* End of TREX_PYTHON_TSOLVER_BINDINGS_HPP */
// =====================================================================================
