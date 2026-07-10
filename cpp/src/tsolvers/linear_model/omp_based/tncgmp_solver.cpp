// ===================================================================================
// tncgmp_solver.cpp
// ===================================================================================

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <tsolvers/linear_model/omp_based/tncgmp_solver.hpp>
#include <utils/openmp/utils_openmp.hpp>

// ==================================================================================

// Embedding into namespace trex::tsolvers::linear_model::omp_based
namespace trex::tsolvers::linear_model::omp_based {

// ==================================================================================
// Constructor Implementation
// ==================================================================================

TNCGMP_Solver::TNCGMP_Solver(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D,
    Eigen::Map<Eigen::VectorXd>& y,
    NCGMPVariant variant,
    bool normalize,
    bool intercept,
    bool verbose,
    ScalingMode scaling_mode)

    : TSolver_Base(X, D, y, normalize, intercept, verbose, scaling_mode),
      variant_(variant) {

    dropped_indices_.clear();
    if (normalize_ || intercept_) { preprocess(); }
    initializeInactives();

    std::size_t p_tot = p_original_ + num_dummies_;
    max_actives_ = std::min(p_tot - dropped_indices_.size(), effective_n_);

    // Line Search (MP) may require more steps than dimensions since it doesn't orthogonalize
    maxSteps_ = (variant_ == NCGMPVariant::LineSearch) ?
                            20 * p_tot :
                            8 * std::min(max_actives_, effective_n_);

    logInfo(concatMsg(
            solverTypeToString(), " sequence\n",
            "  Samples (n): ", X.rows(), "\n",
            "  Features (p): ", p_tot, "\n",
            "  Max steps: ", maxSteps_, "\n"
    ));

    // The LineSearch (MP) variant stores one dense beta-path column per step
    // and has a uniquely large step ceiling (20 * p_tot, vs 8 * min(...) for the
    // OMP-family solvers). The path is allocated incrementally and bounded by
    // T_stop / early stopping, so the full worst case is virtually never reached.
    // This is an informational size note for debugging, not a fault: emit it via
    // logInfo (verbose-only), consistent with the rest of the solver suite where
    // logWarning is reserved for genuine numerical/structural problems.
    if (variant_ == NCGMPVariant::LineSearch) {
        double worst_case_gib = static_cast<double>(p_tot) *
                                static_cast<double>(maxSteps_) * 8.0 /
                                (1024.0 * 1024.0 * 1024.0);
        if (worst_case_gib > 1.0) {
            logInfo(concatMsg(
                "LineSearch worst-case beta path is ~", worst_case_gib,
                " GiB (", p_tot, " x ", maxSteps_,
                " doubles); allocated incrementally, bounded by T_stop early stopping."));
        }
    }

    r_ = y_;
    betaPath_.resize(static_cast<Eigen::Index>(p_tot), 1);
    betaPath_.col(0).setZero();

    double rss = y_.dot(y_);
    RSS_.emplace_back(rss);
    R2_.emplace_back(0.0);
    DoF_.emplace_back(intercept_ ? 1 : 0);
    dummies_at_step_.push_back(0);

    initializeCorrelations();
}

TNCGMP_Solver::TNCGMP_Solver() : TSolver_Base() {}

// ============================================================================
// Core Algorithm Implementation
// ============================================================================

void TNCGMP_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();
    EigenSingleThreadGuard eigen_single_thread_guard;

    // Fully Corrective: collinear rejections must not create standalone
    // actions_ entries (they would desync actions_ from the per-step
    // diagnostics arrays); they are prepended to the next successful step.
    std::vector<int> pending_drop_actions;

    // Line Search (MP) re-selects active atoms, so it is not bounded by the
    // inactive set or the active-set size; it stops via maxSteps_, T_stop,
    // or vanishing correlations.
    while (currentStep_ < maxSteps_ &&
           (variant_ == NCGMPVariant::LineSearch ||
            (!inactives_.empty() && actives_.size() < effective_n_)) &&
           (!early_stop || T_stop == 0 || count_active_dummies_ < T_stop)) {

        // 1. Linear Minimization Oracle (LMO) - Max Correlation
        auto [Cmax, new_vars] = findTiedMaxCorrelations();
        if (new_vars.empty() || Cmax < 100 * eps_) {
            logInfo("Max |corr| approx 0; exiting.");
            break;
        }
        pruneTiedDummies(new_vars, T_stop, early_stop);

        std::size_t j_new = new_vars[0]; // Take the top tied variable

        if (variant_ == NCGMPVariant::LineSearch) {
            // Variant 0: Line Search (Standard MP)
            // -------------------------------------------------------------------
            // x_{t+1} = x_t + gamma * z_t
            double gamma = correlations_(
                static_cast<Eigen::Index>(j_new)) / getColumn(j_new).squaredNorm();

            currentStep_++;

            // Expand beta path
            ensureBetaPathCapacity(currentStep_ + 1);
            betaPath_.col(static_cast<Eigen::Index>(currentStep_)) =
                betaPath_.col(static_cast<Eigen::Index>(currentStep_ - 1));
            betaPath_.col(static_cast<Eigen::Index>(currentStep_))(
                static_cast<Eigen::Index>(j_new)) += gamma;

            r_ -= gamma * getColumn(j_new);

            if (std::find(actives_.begin(), actives_.end(), j_new) ==
                actives_.end()) {
                actives_.push_back(j_new);
                // Keep the actives/inactives partition invariant intact for
                // the public getters (re-selection scans all non-dropped
                // columns via the variant-aware overrides below)
                std::erase(inactives_, j_new);
                if (j_new >= dummy_start_idx_) { count_active_dummies_++; }
            }
            actions_.push_back({actionAdd(j_new)});

            updateInactiveSet();

        } else {
            // Variant 1: Fully Corrective (Orthogonal Matching Pursuit)
            // -------------------------------------------------------------------
            Eigen::MatrixXd newR = updateR(getColumn(j_new), eps_);

            if (last_updateR_rank_ == static_cast<int>(actives_.size())) {
                dropped_indices_.push_back(j_new);
                pending_drop_actions.push_back(actionDrop(j_new));
                any_dropped_ = true;
                logWarning(concatMsg("Variable ", j_new, " collinear; dropped."));
                updateInactiveSet();
                continue; // Skip residual update, try next iteration
            } else {
                R_ = newR;
                actives_.push_back(j_new);
                pending_drop_actions.push_back(actionAdd(j_new));
                actions_.push_back(std::move(pending_drop_actions));
                pending_drop_actions.clear();
                std::erase(inactives_, j_new);
                if (j_new >= dummy_start_idx_) { count_active_dummies_++; }
            }
            currentStep_++;
            updateBetaPathAndResiduals();
            updateInactiveSet();
        }

        // Diagnostics
        double rss = r_.dot(r_);
        RSS_.push_back(rss);
        R2_.push_back(1.0 - rss / RSS_[0]);
        updateDummyTracking();
        DoF_.push_back(actives_.size() + (intercept_ ? 1 : 0));

        updateCorrelations();
    }

    // Drop any spare beta-path capacity now that execution stopped
    trimBetaPathToRecordedSteps();
}

// ==================================================================================
// Math Helpers
// ==================================================================================

void TNCGMP_Solver::updateCorrelations() {
    if (variant_ == NCGMPVariant::LineSearch) {
        // MP re-selection: refresh ALL non-dropped variables (actives included)
        refreshAllCorrelations();
    } else {
        // Fully Corrective (OMP): inactive variables only
        refreshInactiveCorrelations();
    }
}


std::pair<double, std::vector<std::size_t>>
TNCGMP_Solver::findTiedMaxCorrelations() const {
    // Line Search (MP) allows re-selection: scan all non-dropped variables.
    // Fully Corrective (OMP) uses the base inactive-only scan.
    return (variant_ == NCGMPVariant::LineSearch)
               ? findTiedMaxCorrelationsAllNonDropped()
               : TSolver_Base::findTiedMaxCorrelations();
}

void TNCGMP_Solver::updateBetaPathAndResiduals() {
    std::size_t k = actives_.size();
    Eigen::VectorXd b(k);
    for (std::size_t i = 0; i < k; ++i) {
        b(static_cast<Eigen::Index>(i)) = getColumn(actives_[i]).dot(y_);
    }

    Eigen::VectorXd u = backsolveT(R_, b);
    Eigen::VectorXd beta_A = backsolve(R_, u);

    Eigen::Index p_tot = static_cast<Eigen::Index>(p_original_ + num_dummies_);
    Eigen::VectorXd beta_hat = Eigen::VectorXd::Zero(p_tot);

    for (std::size_t i = 0; i < k; ++i) {
        beta_hat(static_cast<Eigen::Index>(actives_[i])) =
            beta_A(static_cast<Eigen::Index>(i));
    }

    ensureBetaPathCapacity(currentStep_ + 1);
    betaPath_.col(static_cast<Eigen::Index>(currentStep_)) = beta_hat;

    r_ = y_;
    for (std::size_t i = 0; i < k; ++i) {
        std::size_t j = actives_[i];
        r_ -= getColumn(j) * beta_hat(static_cast<Eigen::Index>(j));
    }
}

// ==================================================================================
// Serialization
// ==================================================================================

void TNCGMP_Solver::save(const std::string& filename) const { saveImpl(*this, filename); }

TNCGMP_Solver TNCGMP_Solver::load(const std::string& filename,
                                  Eigen::Map<Eigen::MatrixXd>& X,
                                  Eigen::Map<Eigen::MatrixXd>& D) {
    return loadImpl<TNCGMP_Solver>(filename, X, D);
}

// ==================================================================================
} /* End of namespace trex::tsolvers::linear_model::omp_based */
// ==================================================================================
