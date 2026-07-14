// ===================================================================================
// tomp_solver.cpp
// ===================================================================================

// std includes
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

// tsolvers includes
#include <tsolvers/linear_model/omp_based/tomp_solver.hpp>

// utils includes
#include <utils/openmp/utils_openmp.hpp>

// ==================================================================================

// Embedded into trex::tsolvers::linear_model::omp_based namespace
namespace trex::tsolvers::linear_model::omp_based {

// ==================================================================================


// ==================================================================================
// Constructor Implementation
// ==================================================================================

TOMP_Solver::TOMP_Solver(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D,
    Eigen::Map<Eigen::VectorXd>& y,
    bool normalize,
    bool intercept,
    bool verbose,
    SolverTypeOMPBased algorithm_type,
    ScalingMode scaling_mode)
    : TSolver_Base(X, D, y, normalize, intercept, verbose, scaling_mode),
      algo_type_(algorithm_type) {

    // 1. Inherited Preprocessing
    dropped_indices_.clear();
    if (normalize_ || intercept_) { preprocess(); }
    initializeInactives();

    // 2. State Limits
    std::size_t p_tot = p_original_ + num_dummies_;
    max_actives_ = std::min(p_tot - dropped_indices_.size(), effective_n_);
    maxSteps_ = 8 * std::min(max_actives_, effective_n_);

    logInfo(concatMsg(
            solverTypeToString(), " sequence\n",
            "  Samples (n): ", X.rows(), "\n",
            "  Features (p): ", p_tot, " (", p_original_, " real + ", num_dummies_, " dummies)\n",
            "  Max steps: ", maxSteps_, "\n"
    ));

    // 3. Initialize State
    r_ = y_;
    initBetaPathStorage();

    double rss = y_.dot(y_);
    RSS_.emplace_back(rss);
    R2_.emplace_back(0.0);
    DoF_.emplace_back(intercept_ ? 1 : 0);
    dummies_at_step_.reserve(maxSteps_ + 1);
    dummies_at_step_.push_back(0);

    // 4. Initial correlations
    initializeCorrelations();
}

// Delegator constructor for protected use
TOMP_Solver::TOMP_Solver(SolverTypeOMPBased type) : TSolver_Base(), algo_type_(type) {}
TOMP_Solver::TOMP_Solver() : TSolver_Base(), algo_type_(SolverTypeOMPBased::TOMP) {}


// ============================================================================
// Core Algorithm Implementation
// ============================================================================

void TOMP_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();
    // Set openMP dominance for the entire algorithm
    EigenSingleThreadGuard eigen_single_thread_guard;

    while (currentStep_ < maxSteps_ && !inactives_.empty() &&
           actives_.size() < effective_n_ &&
           (!early_stop || T_stop == 0 || count_active_dummies_ < T_stop)) {

        // ========================================================
        // STEP 1: Max correlation
        // ========================================================
        auto [Cmax, new_vars] = findTiedMaxCorrelations();
        if (new_vars.empty() || Cmax < 100 * eps_) {
            logInfo("Max |corr| approx 0 or no variables to add; exiting.");
            break;
        }
        pruneTiedDummies(new_vars, T_stop, early_stop);

        // ========================================================
        // STEP 2: Active set update
        // ========================================================
        actions_.emplace_back(updateActiveSet(new_vars));
        if (actives_.empty() || R_.size() == 0) {
            logWarning("Active set or Cholesky is empty; exiting.");
            break;
        }

        // ========================================================
        // STEP 3: Beta path and residuals
        // ========================================================
        currentStep_++;
        updateBetaPath();
        updateResiduals();

        double rss = r_.dot(r_);
        RSS_.push_back(rss);
        R2_.push_back(1.0 - rss / RSS_[0]);
        updateDummyTracking();
        DoF_.push_back(actives_.size() + (intercept_ ? 1 : 0));

        // ========================================================
        // STEP 4: Inactives and Correlations update
        // ========================================================
        updateInactiveSet();
        updateCorrelations();
    }
}

// ==================================================================================
// Math Helpers (Routed via getColumn)
// ==================================================================================

void TOMP_Solver::updateCorrelations() {
    refreshInactiveCorrelations();
}


std::vector<int> TOMP_Solver::updateActiveSet(const std::vector<std::size_t>& new_vars) {
    std::vector<int> actions_this_step;
    any_dropped_ = false;

    for (auto j_new : new_vars) {
        Eigen::MatrixXd R_backup = R_;
        int rankR_backup = last_updateR_rank_;
        std::size_t active_size_backup = actives_.size();

        Eigen::MatrixXd newR = updateR(getColumn(j_new), eps_);

        if (last_updateR_rank_ == static_cast<int>(active_size_backup)) {
            R_ = R_backup;
            last_updateR_rank_ = rankR_backup;
            dropped_indices_.push_back(j_new);
            actions_this_step.push_back(actionDrop(j_new));
            any_dropped_ = true;
            logWarning(concatMsg("Variable ", j_new, " collinear; dropped."));
        } else {
            R_ = newR;
            actives_.push_back(j_new);
            actions_this_step.push_back(actionAdd(j_new));
            num_additions_++;
            std::erase(inactives_, j_new);
            if (j_new >= dummy_start_idx_) count_active_dummies_++;
        }
    }
    return actions_this_step;
}


void TOMP_Solver::updateBetaPath() {
    std::size_t k = actives_.size();
    Eigen::VectorXd b(k);
    for (std::size_t i = 0; i < k; ++i) {
        b(static_cast<Eigen::Index>(i)) = getColumn(actives_[i]).dot(y_);
    }

    Eigen::VectorXd u = backsolveT(R_, b);
    Eigen::VectorXd beta_A = backsolve(R_, u);

    // Full OLS refit: the step's support IS the active set.
    setRunningBeta(actives_, beta_A);
    recordBetaStep();
}


void TOMP_Solver::updateResiduals() {
    r_ = y_;
    std::size_t k = actives_.size();

    for (std::size_t i = 0; i < k; ++i) {
        std::size_t j = actives_[i];
        r_ -= getColumn(j) * runningBeta(j);
    }
}


// ==================================================================================
// Serialization
// ==================================================================================

void TOMP_Solver::save(const std::string& filename) const { saveImpl(*this, filename); }


TOMP_Solver TOMP_Solver::load(const std::string& filename,
                              Eigen::Map<Eigen::MatrixXd>& X,
                              Eigen::Map<Eigen::MatrixXd>& D) {
    return loadImpl<TOMP_Solver>(filename, X, D);
}

// ==================================================================================
} /* End of namespace namespace trex::tsolvers::linear_model::omp_based */
