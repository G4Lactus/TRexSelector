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
    Eigen::setNbThreads(1);

    while (currentStep_ < maxSteps_ && !inactives_.empty() &&
           (variant_ == NCGMPVariant::LineSearch || actives_.size() < effective_n_) &&
           (count_active_dummies_ < T_stop || !early_stop)) {

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
            Eigen::Index p_tot = static_cast<Eigen::Index>(p_original_ + num_dummies_);
            if (currentStep_ >= static_cast<std::size_t>(betaPath_.cols())) {
                betaPath_.conservativeResize(
                    p_tot, static_cast<Eigen::Index>(currentStep_ + 1));
            }
            betaPath_.col(static_cast<Eigen::Index>(currentStep_)) =
                betaPath_.col(static_cast<Eigen::Index>(currentStep_ - 1));
            betaPath_.col(static_cast<Eigen::Index>(currentStep_))(
                static_cast<Eigen::Index>(j_new)) += gamma;

            r_ -= gamma * getColumn(j_new);

            if (std::find(actives_.begin(), actives_.end(), j_new) ==
                actives_.end()) {
                actives_.push_back(j_new);
                if (j_new >= dummy_start_idx_) { count_active_dummies_++; }
            }
            actions_.push_back({static_cast<int>(j_new)});

            rebuildInactiveSetForLineSearch();

        } else {
            // Variant 1: Fully Corrective (Orthogonal Matching Pursuit)
            // -------------------------------------------------------------------
            Eigen::MatrixXd newR = updateR(getColumn(j_new), eps_);

            if (last_updateR_rank_ == static_cast<int>(actives_.size())) {
                dropped_indices_.push_back(j_new);
                actions_.push_back({-static_cast<int>(j_new)});
                any_dropped_ = true;
                logWarning(concatMsg("Variable ", j_new, " collinear; dropped."));
                updateInactiveSet();
                continue; // Skip residual update, try next iteration
            } else {
                R_ = newR;
                actives_.push_back(j_new);
                actions_.push_back({static_cast<int>(j_new)});
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
}

// ==================================================================================
// Math Helpers
// ==================================================================================

void TNCGMP_Solver::updateCorrelations() {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < inactives_.size(); ++i) {
        std::size_t j = inactives_[i];
        correlations_(static_cast<Eigen::Index>(j)) = getColumn(j).dot(r_);
    }
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

    if (currentStep_ >= static_cast<std::size_t>(betaPath_.cols())) {
        betaPath_.conservativeResize(p_tot,
                                     static_cast<Eigen::Index>(currentStep_ + 1));
    }
    betaPath_.col(static_cast<Eigen::Index>(currentStep_)) = beta_hat;

    r_ = y_;
    for (std::size_t i = 0; i < k; ++i) {
        std::size_t j = actives_[i];
        r_ -= getColumn(j) * beta_hat(static_cast<Eigen::Index>(j));
    }
}

void TNCGMP_Solver::rebuildInactiveSetForLineSearch() {
    // In Line Search (MP), we DO NOT remove active variables from the inactive set
    // because standard MP allows re-selecting the same atom multiple times.
    if (!any_dropped_) return;

    inactives_.clear();
    std::size_t p_tot = p_original_ + num_dummies_;
    for (std::size_t j = 0; j < p_tot; ++j) {
        if (std::ranges::find(dropped_indices_, j) == dropped_indices_.end()) {
            inactives_.push_back(j);
        }
    }
    any_dropped_ = false;
}

// ==================================================================================
// Serialization
// ==================================================================================

void TNCGMP_Solver::save(const std::string& filename) const {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error(concatMsg(solverTypeToString(),
                                "::save: Cannot open '", filename, "'"));
    }
    cereal::PortableBinaryOutputArchive oarchive(ofs);
    oarchive(*this);
    logInfo(concatMsg("[", solverTypeToString(), "] saved to '", filename, "'\n"));
}

TNCGMP_Solver TNCGMP_Solver::load(
    const std::string& filename,
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D) {

    TNCGMP_Solver solver;
    std::ifstream is(filename, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error(solver.concatMsg(solver.solverTypeToString(),
                                 "::load: Cannot open '", filename, "'"));
    }
    cereal::PortableBinaryInputArchive iarchive(is);
    iarchive(solver);

    solver.reconnect(X, D);
    return solver;
}

// ==================================================================================
} /* End of namespace trex::tsolvers::linear_model::omp_based */
// ==================================================================================
