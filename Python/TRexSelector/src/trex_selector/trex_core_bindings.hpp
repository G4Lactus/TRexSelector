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

    // Bind LLoopStrategy enum
    py::enum_<LLoopStrategy>(m_tsm, "LLoopStrategy", "Operational modes determining exactly how Dummy Matrices correlate to Predictors.")
        .value("SKIPL", LLoopStrategy::SKIPL, "L=1 Dummy calculation mode.")
        .value("STANDARD", LLoopStrategy::STANDARD, "Averaged correlation processing logic for internal comparisons.")
        .value("HCONCAT", LLoopStrategy::HCONCAT, "Aggregates and flattens all internal dummy variables horizontally.")
        .value("PERMUTATION", LLoopStrategy::PERMUTATION, "Creates dummy statistics explicitly through standard sequential Permutations.")
        .value("PERMUTATION_DIRECT", LLoopStrategy::PERMUTATION_DIRECT, "Directly constructs baseline dummies from uniform permutations.")
        .value("DIRECT", LLoopStrategy::DIRECT, "Generates dummy components directly without cyclical aggregation assumptions.")
        .export_values();

    // Bind TRexControlParameter
    py::class_<TRexControlParameter>(m_tsm, "TRexControlParameter", "Configures advanced execution mechanisms controlling the core Selection processes.")
        .def(py::init<>())
        .def_readwrite("K", &TRexControlParameter::K, "Number of random simulation experiments to run over variable inclusions.")
        .def_readwrite("max_dummy_multiplier", &TRexControlParameter::max_dummy_multiplier, "Coefficient determining multiplier iterations for large variables sets.")
        .def_readwrite("use_max_T_stop", &TRexControlParameter::use_max_T_stop, "Force solving towards exactly T_stop constraints blindly.")
        .def_readwrite("opt_threshold", &TRexControlParameter::opt_threshold, "Optional fractional boundary adjusting selection strictness.")
        .def_readwrite("lloop_strategy", &TRexControlParameter::lloop_strategy, "Identifies the core structural logic to run for L-loop variables.")
        .def_readwrite("tloop_stagnation_stop", &TRexControlParameter::tloop_stagnation_stop, "Break the execution paths strictly if sequential bounds fail to expand.")
        .def_readwrite("tloop_max_stagnant_steps", &TRexControlParameter::tloop_max_stagnant_steps, "Quantity bounds assigned preventing non-updating steps.")
        .def_readwrite("parallel_rnd_experiments", &TRexControlParameter::parallel_rnd_experiments, "Perform the K simulated sets asynchronously utilizing local cores.");

    // Bind TRexSelector::SelectionResult
    py::class_<TRexSelector::SelectionResult>(m_tsm, "SelectionResult", "Encapsulates diagnostic boundaries and final matrix values produced from `.select()`.")
        .def_readonly("selected_var", &TRexSelector::SelectionResult::selected_var, "Indicators (0 or 1) mapping predictors selected strictly via FDR boundary checks.")
        .def_readonly("v_thresh", &TRexSelector::SelectionResult::v_thresh, "Computed FDR Threshold value generated by selection formula bounds.")
        .def_readonly("R_mat", &TRexSelector::SelectionResult::R_mat, "Indicator metric history of chosen predictors mapped against experiments generated.")
        .def_readonly("T_stop", &TRexSelector::SelectionResult::T_stop, "Stop evaluation max boundaries assigned strictly.")
        .def_readonly("num_dummies", &TRexSelector::SelectionResult::num_dummies, "Calculated total dummy components produced per sequence step.")
        .def_readonly("dummy_factor_L", &TRexSelector::SelectionResult::dummy_factor_L, "Actual scaled Dummy multiplier utilized in evaluating boundaries.")
        .def_readonly("FDP_hat_mat", &TRexSelector::SelectionResult::FDP_hat_mat, "Statistical FDP hat approximation tracking metrics.")
        .def_readonly("Phi_mat", &TRexSelector::SelectionResult::Phi_mat, "Variable density array matrix for Phi estimations.")
        .def_readonly("Phi_prime", &TRexSelector::SelectionResult::Phi_prime, "Derived secondary probability approximation.")
        .def_readonly("voting_grid", &TRexSelector::SelectionResult::voting_grid, "Summations corresponding identically across iteration maps.");

    // Bind PyTRexSelector
    py::class_<PyTRexSelector>(m_tsm, "TRexSelector", "Core Execution class for configuring and calculating T-Rex framework predictions.")
        .def(py::init<Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::VectorXd>, double, const TRexControlParameter&, int, bool>(),
             py::arg("X"), py::arg("y"), py::arg("tFDR") = 0.1,
             py::arg("trex_control") = TRexControlParameter(),
             py::arg("seed") = -1, py::arg("verbose") = true,
             "Mapping engine directly utilizing Python NumPy states strictly via zero-copy paths.")
        .def("select", &PyTRexSelector::select, py::call_guard<py::gil_scoped_release>(),
             "Main solver method initiating calculation sequences iteratively through algorithm classes.")
        .def("getTFDR", &PyTRexSelector::getTFDR, "Fetch Target FDR.")
        .def("getK", &PyTRexSelector::getK, "Fetch total configurations count for K random simulations.")
        .def("getMaxNumDummies", &PyTRexSelector::getMaxNumDummies, "Extract the explicitly stated dummy boundary variable count.")
        .def("getMaxTStop", &PyTRexSelector::getMaxTStop, "Determine exact strict upper length paths executed across arrays.")
        .def("getStagnationCheck", &PyTRexSelector::getStagnationCheck, "Flag highlighting stagnation stoppage active evaluations.")
        .def("getXMeans", &PyTRexSelector::getXMeans, "Fetch Predictor X Array normalizations mapping averages.")
        .def("getXL2Norms", &PyTRexSelector::getXL2Norms, "Fetch Precalculated absolute variance measurements from matrix entries.")
        .def("getYMean", &PyTRexSelector::getYMean, "Get Target Y distribution uniform calculations.")
        .def("getSelectedVar", &PyTRexSelector::getSelectedVar, "Acquires specific 0|1 integer indications mapping FDR inclusion assumptions.")
        .def("getSelectedIndices", &PyTRexSelector::getSelectedIndices, "Locates actual un-ordered variable keys included internally.")
        .def("getNumSelected", &PyTRexSelector::getNumSelected, "Totals completely bound variables filtered inside prediction criteria.")
        .def("getTStop", &PyTRexSelector::getTStop, "Retrieves execution limit bound configurations assigned initially.")
        .def("getVotingGrid", &PyTRexSelector::getVotingGrid, "Returns final array statistics defining voting weights per execution bounds.")
        .def("getVotingThreshold", &PyTRexSelector::getVotingThreshold, "Numeric scalar utilized determining exact acceptance barriers.")
        .def("getFDPHatMat", &PyTRexSelector::getFDPHatMat, "Secondary FDP Hat indicator metric arrays maps.")
        .def("getPhiMat", &PyTRexSelector::getPhiMat, "Predictor distributions mapping to Dummy component densities.")
        .def("getRMat", &PyTRexSelector::getRMat, "R Matrix tracking metrics of exact internal executions.")
        .def("getPhiPrime", &PyTRexSelector::getPhiPrime, "Prime derivative probability arrays indicating exact inclusion metrics.")
        .def("getDummyMultiplierL", &PyTRexSelector::getDummyMultiplierL, "Fetch scaling ratio configured strictly to size variables internally.");
}
// =====================================================================================
#endif /* End of TREX_PYTHON_TREX_CORE_BINDINGS_HPP */
