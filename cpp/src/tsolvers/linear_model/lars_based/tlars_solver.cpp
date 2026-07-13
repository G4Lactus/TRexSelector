// ============================================================================
// tlars_solver.cpp
// ============================================================================
/**
 * @file tlars_solver.cpp
 *
 * @brief Implementation (.cpp) file for the Terminating Least Angle Regression
 * (T-LARS) solver.
 */
// ============================================================================

// std includes
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdexcept>

// tsolvers includes
#include <tsolvers/linear_model/lars_based/tlars_solver.hpp>
#include <tsolvers/solver_utils/solver_preprocessing.hpp>

// utils includes
#include <utils/serialization//utils_cereal_eigen.hpp>
#include <utils/openmp/utils_openmp.hpp>
#include <utils/fp_classify/fp_classify.hpp>

// ============================================================================

// Embedded into trex::tsolvers::linear_model::lars_based namespace
namespace trex::tsolvers::linear_model::lars_based {

// ============================================================================

// Use namespace alias for solver_utils::preprocessing
namespace solv_preproc = trex::tsolvers::solver_utils::preprocessing;
namespace fpc = trex::utils::fp_classify;


// ============================================================================
// Constructors
// ============================================================================

TLARS_Solver::TLARS_Solver(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D,
    Eigen::Map<Eigen::VectorXd>& y,
    bool normalize,
    bool intercept,
    bool verbose,
    SolverTypeLarsBased algorithm_type,
    ScalingMode scaling_mode
    )
    : TSolver_Base(X, D, y, normalize, intercept, verbose, scaling_mode),
      algo_type_(algorithm_type) {

    // 1. Inherited proprocessing handles X, D, y centering and normalization
    dropped_indices_.clear();
    if (normalize_ || intercept_) { preprocess(); }
    initializeInactives();

    // 2. State Limit
    std::size_t p_total = p_original_ + num_dummies_;
    max_actives_ = std::min(p_total - dropped_indices_.size(), effective_n_);
    maxSteps_ = 8 * std::min(max_actives_, effective_n_);

    logInfo(concatMsg(
            solverTypeToString(), " sequence\n",
            "  Samples (n): ", X.rows(), "\n",
            "  Features (p): ", p_total, " (",
            p_original_, " real + ",
            num_dummies_, " dummies)\n",
            "  Max steps: ", maxSteps_, "\n"
    ));

    // 3. Initialize shared path state variables
    r_ = y_;
    initBetaPathStorage();

    double rss = y_.dot(y_);
    RSS_.emplace_back(rss);
    R2_.emplace_back(0.0);
    DoF_.emplace_back(intercept ? 1 : 0);
    dummies_at_step_.reserve(maxSteps_ + 1);
    dummies_at_step_.push_back(0);

    // 4. Initialize LARS specific states
    lambda_.reserve(maxSteps_ + 1);
    correlation_compensation_ = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(p_total));
    initializeCorrelations();
}


TLARS_Solver::TLARS_Solver(SolverTypeLarsBased type) : TSolver_Base(), algo_type_(type) {}
TLARS_Solver::TLARS_Solver() : TSolver_Base(), algo_type_(SolverTypeLarsBased::TLARS) {}


// ============================================================================
// Core Algorithm Implementation
// ============================================================================

void TLARS_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();

    // Set openMP dominance for the entire algorithm
    EigenSingleThreadGuard eigen_single_thread_guard;

    // Main T-LARS loop with early stopping based on dummy variable count
    while (
        currentStep_ < maxSteps_ && !inactives_.empty() &&
        actives_.size() < effective_n_ &&
        (!early_stop || T_stop == 0 || count_active_dummies_ < T_stop)
    ) {

        // ========================================================
        // STEP 1: Max correlation among inactives
        // ========================================================
        auto [Cmax, new_vars] = findTiedMaxCorrelations();
        if (new_vars.empty() || Cmax < 100 * eps_) {
            logInfo("Max |corr| approx. 0 or no variables to add; terminating.");
            break;
        }
        pruneTiedDummies(new_vars, T_stop, early_stop);

        // Increment step counter and record lambdas
        currentStep_++;
        lambda_.push_back(Cmax);

        // ========================================================
        // STEP 2: Active set update
        // ========================================================
        actions_.emplace_back(updateActiveSet(new_vars));
        if (actives_.empty() || R_.size() == 0) {
            logWarning("Active set or Cholesky is empty; aborting.");
            // Roll back the aborted step so actions_/lambda_ stay aligned
            // with the diagnostics arrays (rejections remain recorded in
            // dropped_indices_) and rebuild inactives_ so dropped columns
            // cannot be re-selected.
            actions_.pop_back();
            lambda_.pop_back();
            currentStep_--;
            updateInactiveSet();
            break;
        }

        // ========================================================
        // STEP 3: Equi-angular computations
        // ========================================================
        Eigen::VectorXd sign_vec = signVector();
        Eigen::VectorXd w_A = equiangularDirection(sign_vec);
        Eigen::VectorXd u = equiangularVector(w_A);
        auto [gamma, a] = computeStepSize(Cmax, u);

        // ========================================================
        // STEP 4: Updates paths and residuals
        // ========================================================
        updateBetaPath(w_A, gamma);
        recordBetaStep();
        r_ -= gamma * u;

        // ========================================================
        // STEP 5: Update diagnostics
        // ========================================================
        double rss = r_.dot(r_);
        RSS_.push_back(rss);
        R2_.push_back(1.0 - rss / RSS_[0]);
        updateDummyTracking();
        DoF_.push_back(actives_.size() + (intercept_ ? 1 : 0));

        // ========================================================
        // STEP 6: Update inactives and correlations
        // ========================================================
        updateCorrelations(gamma, a);
        updateInactiveSet();
    }
}

// ============================================================================
// Helper Implementations
// ============================================================================

void TLARS_Solver::updateCorrelations(
    double gamma,
    const Eigen::Ref<const Eigen::VectorXd>& a
) {
    std::size_t num_inactives = inactives_.size();

    if ((currentStep_ % kahan_refresh_interval_) == 0) {
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < num_inactives; ++i) {
            Eigen::Index j = static_cast<Eigen::Index>(inactives_[i]);
            correlations_(j) = getColumn(j).dot(r_);
            correlation_compensation_(j) = 0.0;
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < num_inactives; ++i) {
            Eigen::Index j = static_cast<Eigen::Index>(inactives_[i]);
            double update = -gamma * a(static_cast<Eigen::Index>(i)) -
                             correlation_compensation_(j);
            double temp = correlations_(j) + update;
            correlation_compensation_(j) = (temp - correlations_(j)) - update;
            correlations_(j) = temp;
        }
    }
}


std::vector<int> TLARS_Solver::updateActiveSet (const std::vector<std::size_t>& new_vars) {
    std::vector<int> actions_this_step;
    any_dropped_ = false;

    for (auto j_new : new_vars) {
        // updateR never mutates R_, so only the rank needs restoring on rejection
        int rankR_backup = last_updateR_rank_;
        std::size_t active_size_backup = actives_.size();

        Eigen::MatrixXd newR = updateR(getColumn(j_new), eps_);

        if (last_updateR_rank_ == static_cast<int>(active_size_backup)) {
            last_updateR_rank_ = rankR_backup;
            dropped_indices_.push_back(j_new);
            actions_this_step.push_back(actionDrop(j_new));
            any_dropped_ = true;
            logWarning(concatMsg("Variable ", j_new, " collinear; dropped."));
        } else {
            R_ = newR;
            actives_.push_back(j_new);
            actions_this_step.push_back(actionAdd(j_new));
            Sign_.push_back((correlations_(static_cast<Eigen::Index>(j_new)) >= 0) ?
                                1 : -1);
            num_additions_++;
            std::erase(inactives_, j_new);

            if (j_new >= dummy_start_idx_) count_active_dummies_++;
        }
    }
    return actions_this_step;
}


void TLARS_Solver::updateBetaPath(const Eigen::Ref<const Eigen::VectorXd>& w_A, double gamma) {
    // Advance the running sparse coefficients in place; the caller records
    // the step snapshot (after any same-step LASSO drops) via recordBetaStep().
    for (std::size_t k = 0; k < actives_.size(); ++k) {
        runningBetaRef(actives_[k]) += gamma * w_A(static_cast<Eigen::Index>(k));
    }
}


Eigen::VectorXd TLARS_Solver::signVector() const {
    Eigen::VectorXd sign_vec(static_cast<Eigen::Index>(Sign_.size()));
    for (std::size_t i = 0; i < Sign_.size(); ++i) {
        sign_vec(static_cast<Eigen::Index>(i)) = static_cast<double>(Sign_[i]);
    }
    return sign_vec;
}


Eigen::VectorXd TLARS_Solver::equiangularDirection(const Eigen::Ref<const Eigen::VectorXd>& sign) {

    std::size_t m = R_.rows();
    if (m == 1) {
        Eigen::VectorXd GAi_s(1);
        double diag = R_(0, 0);
        GAi_s(0) = sign(0) / (diag * diag);
        A_A_ = 1.0 / std::sqrt(sign(0) * GAi_s(0));
        return A_A_ * GAi_s;
    } else {
        Eigen::VectorXd GAi_s = backsolve(R_, backsolveT(R_, sign));
        A_A_ = 1.0 / std::sqrt(sign.dot(GAi_s));
        return A_A_ * GAi_s;
    }
}


Eigen::VectorXd TLARS_Solver::equiangularVector(const Eigen::Ref<const Eigen::VectorXd>& w_A) const {
    Eigen::VectorXd u = Eigen::VectorXd::Zero(X_->rows());

    for (std::size_t k = 0; k < static_cast<std::size_t>(actives_.size()); ++k) {
        u += w_A(static_cast<Eigen::Index>(k)) *
             getColumn(actives_[k]);
    }
    return u;
}


std::pair<double, Eigen::VectorXd> TLARS_Solver::computeStepSize(
    double Cmax,
    const Eigen::Ref<const Eigen::VectorXd>& u) const
{

    std::size_t num_inactives = inactives_.size();
    Eigen::VectorXd a(num_inactives);
    double gamma = std::numeric_limits<double>::max();

    #pragma omp parallel
    {
        double local_gamma = std::numeric_limits<double>::max();

        #pragma omp for schedule(static) nowait
        for (std::size_t i = 0; i < num_inactives; ++i) {
            std::size_t j = inactives_[i];
            double a_j = getColumn(j).dot(u);
            a(static_cast<Eigen::Index>(i)) = a_j;
            double c_j = correlations_(static_cast<Eigen::Index>(j));

            double g1 = (Cmax - c_j) / (A_A_ - a_j);
            double g2 = (Cmax + c_j) / (A_A_ + a_j);

            // local gamma update
            if (g1 > eps_ && fpc::isfinite(g1) && g1 < local_gamma) {
                local_gamma = g1;
            }
            if (g2 > eps_ && fpc::isfinite(g2) && g2 < local_gamma) {
                local_gamma = g2;
            }
        }

        #pragma omp critical
        {
            if (local_gamma < gamma) { gamma = local_gamma; }
        }
    }

    if (gamma == std::numeric_limits<double>::max()) {
        logWarning("No valid gamma found, using fallback.");
        return {Cmax / A_A_, a};
    }

    return {gamma, a};
}


// ============================================================================
// Shared LASSO-style drop machinery
// ============================================================================

double TLARS_Solver::computeGammaSignChange(
    double gamhat,
    std::vector<bool>& drops,
    const Eigen::Ref<const Eigen::VectorXd>& w_A) const
{
    // Extract current beta values for active variables (the running sparse
    // coefficients still hold the previous step here — updateBetaPath() has
    // not run yet for this step).
    Eigen::VectorXd beta_curr(static_cast<Eigen::Index>(actives_.size()));
    for (std::size_t i = 0; i < actives_.size(); ++i) {
        beta_curr(static_cast<Eigen::Index>(i)) = runningBeta(actives_[i]);
    }

    // Zero crossing: z = -beta / w (step size where beta + gamma * w = 0)
    Eigen::VectorXd z = -beta_curr.array() / w_A.array();

    // Find the smallest positive crossing before the next variable entry
    double zmin = gamhat;
    bool any_crossing = false;
    for (Eigen::Index i = 0; i < z.size(); ++i) {
        if (z(i) > eps_ && z(i) < zmin) {
            zmin = z(i);
            any_crossing = true;
        }
    }

    if (any_crossing) {
        gamhat = zmin;
        for (Eigen::Index i = 0; i < z.size(); ++i) {
            drops[static_cast<std::size_t>(i)] = (std::abs(z(i) - zmin) <= eps_);
        }
    }
    return gamhat;
}


void TLARS_Solver::processLassoDrops(std::vector<bool>& drops) {
    // Process drops in reverse order to maintain index stability
    for (std::size_t i = actives_.size(); i-- > 0;) {
        if (!drops[i]) { continue; }
        std::size_t dropped_var = actives_[i];

        // Track if dropping a dummy
        if (dropped_var >= dummy_start_idx_ && count_active_dummies_ > 0) {
            count_active_dummies_--;
            logInfo(concatMsg("Dummy variable ", dropped_var,
                              " removed from active set (count: ",
                              count_active_dummies_, ")"));
        }

        // Downdate Cholesky factor and zero the coefficient at this step
        // (the drop happens before the caller's recordBetaStep(), so the
        // recorded snapshot omits the dropped variable).
        R_ = downdateR(R_, i, eps_);
        removeRunningBeta(dropped_var);

        // Record negative action (removal) and re-add to inactives_ (cycling)
        actions_.back().push_back(actionDrop(dropped_var));
        inactives_.push_back(dropped_var);

        // Refresh correlation for potential re-entry
        refreshDroppedCorrelation(dropped_var);

        actives_.erase(actives_.begin() + static_cast<Eigen::Index>(i));
        Sign_.erase(Sign_.begin() + static_cast<Eigen::Index>(i));
        num_removals_++;
    }
}


void TLARS_Solver::refreshDroppedCorrelation(std::size_t dropped_var) {
    Eigen::Index j = static_cast<Eigen::Index>(dropped_var);
    correlations_(j) = getColumn(dropped_var).dot(r_);
    correlation_compensation_(j) = 0.0;
}


// ============================================================================
// Setter
// ============================================================================

void TLARS_Solver::setKahanRefreshInterval(std::size_t interval) {
    if (interval < 1) {
        logWarning(concatMsg(
            "Kahan refresh interval must be at least 1; keeping current value (",
            kahan_refresh_interval_, ")."
        ));
        return;
    }

    if (interval > 100) {
        logWarning(concatMsg(
            "Kahan refresh interval (", interval,
            ") is large; numerical drift may accumulate. Recommended range: 10-50."
        ));
    }

    kahan_refresh_interval_ = interval;
}


// ============================================================================
// Serialization Implementation
// ============================================================================

void TLARS_Solver::save(const std::string& filename) const { saveImpl(*this, filename); }


TLARS_Solver TLARS_Solver::load(const std::string& filename,
                              Eigen::Map<Eigen::MatrixXd>& X,
                              Eigen::Map<Eigen::MatrixXd>& D) {
    return loadImpl<TLARS_Solver>(filename, X, D);
}

// ==========================================================================
} /* End of namespace tsolvers::linear_model::lars_based */
