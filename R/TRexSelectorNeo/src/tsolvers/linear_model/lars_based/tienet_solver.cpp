// ============================================================================
// tienet_solver.cpp
// ============================================================================
/**
 * @file tienet_solver.cpp
 *
 * @brief Implementation of the Terminating Informed Elastic Net (T-IENET)
 * solver via the native pathwise LARS-IEN algorithm.
 */
// ============================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <stdexcept>

// tsolvers includes
#include <tsolvers/linear_model/lars_based/tienet_solver.hpp>

// utils includes
#include <utils/fp_classify/fp_classify.hpp>

// ============================================================================

// Embedded into namespace trex::tsolvers::linear_model::lars_based
namespace trex::tsolvers::linear_model::lars_based {

namespace fpc = trex::utils::fp_classify;

// ============================================================================
// Constructors
// ============================================================================

TIENET_Solver::TIENET_Solver(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D,
    Eigen::Map<Eigen::VectorXd>& y,
    double lambda2,
    const Eigen::VectorXi& groups,
    bool normalize,
    bool intercept,
    bool verbose,
    ScalingMode scaling_mode
)
    : TLARS_Solver(X, D, y, normalize, intercept, verbose,
                   SolverTypeLarsBased::TIENET, scaling_mode),
      lambda2_(lambda2),
      groups_(groups) {

    if (lambda2_ < 0.0) {
        throw std::invalid_argument(
            "TIENET_Solver: group-ridge penalty lambda2 must be non-negative.");
    }

    const std::size_t p = p_original_;
    const std::size_t L = num_dummies_;

    // ---- Validate the group structure (mirrors TIENETAug_Solver) ----------
    if (static_cast<std::size_t>(groups_.size()) != p) {
        throw std::invalid_argument(
            "TIENET_Solver: groups length must equal the number of "
            "predictors p.");
    }
    Eigen::Index max_id = -1;
    for (Eigen::Index j = 0; j < groups_.size(); ++j) {
        if (groups_(j) < 0) {
            throw std::invalid_argument(
                "TIENET_Solver: groups must be 0-based non-negative ids.");
        }
        if (groups_(j) > max_id) { max_id = groups_(j); }
    }
    M_ = static_cast<std::size_t>(max_id + 1);
    {
        std::vector<bool> seen(M_, false);
        for (Eigen::Index j = 0; j < groups_.size(); ++j) {
            seen[static_cast<std::size_t>(groups_(j))] = true;
        }
        for (std::size_t m = 0; m < M_; ++m) {
            if (!seen[m]) {
                throw std::invalid_argument(
                    "TIENET_Solver: group ids are non-contiguous (id " +
                    std::to_string(m) + " is missing).");
            }
        }
    }
    if (lambda2_ > 0.0 && L > 0 && L % p != 0) {
        throw std::invalid_argument(
            "TIENET_Solver: D.cols() must be a multiple of X.cols() "
            "(one group-aligned dummy layer per p columns).");
    }

    // ---- Unified group mapping (dummy layers share their variable's group) -
    group_of_unified_.assign(p + L, -1);
    for (std::size_t j = 0; j < p; ++j) {
        group_of_unified_[j] = groups_(static_cast<Eigen::Index>(j));
    }
    for (std::size_t t = 0; t < L; ++t) {
        group_of_unified_[p + t] = groups_(static_cast<Eigen::Index>(t % p));
    }

    // ---- Group sizes p_m (REAL sizes) and derived constants ----------------
    group_size_ = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(M_));
    for (std::size_t j = 0; j < p; ++j) {
        group_size_(groups_(static_cast<Eigen::Index>(j))) += 1.0;
    }
    l2_over_pm_ = lambda2_ * group_size_.cwiseInverse();
    group_sums_ = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(M_));
    group_dir_sums_ = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(M_));

    // ---- State limits: rank(X^T X + lambda2 W) <= effective_n + M ----------
    std::size_t pcols_available = p + L - dropped_indices_.size();
    if (lambda2_ > 0.0) {
        max_actives_ = std::min(pcols_available, effective_n_ + M_);
    } else {
        max_actives_ = std::min(pcols_available, effective_n_);
    }
    maxSteps_ = 8 * max_actives_;

    // The TLARS base constructor already computed correlations_ = X^T y over
    // all inactive columns; at initialization sigma = 0, so the modified
    // correlations coincide with the plain ones — no rescaling needed.
}


// ============================================================================
// Core executeStep()
// ============================================================================

void TIENET_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();

    // Set openMP dominance for the entire algorithm
    EigenSingleThreadGuard eigen_single_thread_guard;

    while (currentStep_ < maxSteps_ &&
           !inactives_.empty() &&
           actives_.size() < max_actives_ &&
           (!early_stop || T_stop == 0 || count_active_dummies_ < T_stop)) {

        // ========================================================
        // STEP 1: Find maximum modified correlation
        // ========================================================
        auto [Cmax, new_vars] = findTiedMaxCorrelations();
        if (new_vars.empty() || Cmax < 100 * eps_) {
            logInfo("Max |corr| approx 0 or no variables to add; exiting.");
            break;
        }
        pruneTiedDummies(new_vars, T_stop, early_stop);

        // ========================================================
        // STEP 2: Increment step counter and record lambda
        // ========================================================
        currentStep_++;
        lambda_.push_back(Cmax);

        // ========================================================
        // STEP 3: Active set update (IEN Cholesky with group coupling)
        // ========================================================
        actions_.emplace_back(updateActiveSetIEN(new_vars));

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
        // STEP 4: Equi-angular computation (observation space) and
        //         group direction sums tau
        // ========================================================
        Eigen::VectorXd sign_vec = signVector();
        Eigen::VectorXd w_A = equiangularDirection(sign_vec);
        Eigen::VectorXd u = equiangularVector(w_A);
        computeGroupDirSums(w_A);

        // ========================================================
        // STEP 5: Compute IEN step size and projection
        // ========================================================
        auto [gamma_entry, a] = computeStepSizeIEN(Cmax, u);

        // ========================================================
        // STEP 6: T-LASSO modification - check for sign changes
        // ========================================================
        std::vector<bool> drops(actives_.size(), false);
        double gamma = computeGammaSignChange(gamma_entry, drops, w_A);

        // ========================================================
        // STEP 7: Update beta coefficients, residuals, group sums
        // ========================================================
        updateBetaPath(w_A, gamma);
        r_ -= gamma * u;
        group_sums_ += gamma * group_dir_sums_;   // sigma_m drift along step

        // ========================================================
        // STEP 8: Update modified correlations
        // ========================================================
        updateCorrelationsIEN(gamma, a);

        // ========================================================
        // STEP 9: T-LASSO drops and update inactives internally
        // ========================================================
        processLassoDrops(drops);
        // Record after the drops so this step's snapshot has the dropped
        // coefficients at exactly zero (absent from the support).
        recordBetaStep();

        // ========================================================
        // STEP 10: Update diagnostics and tracking
        // ========================================================
        double rss = r_.dot(r_);
        RSS_.push_back(rss);
        R2_.push_back(1.0 - rss / RSS_[0]);

        updateDummyTracking();
        DoF_.push_back(actives_.size() + (intercept_ ? 1 : 0));

        // ========================================================
        // STEP 11: Update inactive set for next iteration (collinearity)
        // ========================================================
        updateInactiveSet();
    }
}


// ============================================================================
// IEN-specific path machinery
// ============================================================================

std::vector<int> TIENET_Solver::updateActiveSetIEN(
    const std::vector<std::size_t>& new_vars) {

    std::vector<int> actions_this_step;
    any_dropped_ = false;

    for (auto j_new : new_vars) {
        // updateRIEN never mutates R_, so only the rank needs restoring on
        // rejection (mirror of TLARS_Solver::updateActiveSet).
        int rankR_backup = last_updateR_rank_;
        std::size_t active_size_backup = actives_.size();

        Eigen::MatrixXd newR = updateRIEN(j_new, eps_);

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


Eigen::MatrixXd TIENET_Solver::updateRIEN(std::size_t j_new, double eps) {
    std::size_t m = actives_.size();
    const int g_new = group_of_unified_[j_new];
    const double coupling = l2_over_pm_(g_new);

    Eigen::Ref<const Eigen::VectorXd> xnew = getColumn(j_new);
    // IEN diagonal entry: x_j^T x_j + lambda2 / p_m.
    double xtx = xnew.dot(xnew) + coupling;

    if (R_.size() == 0 || m == 0) {
        Eigen::MatrixXd newR(1, 1);
        newR(0, 0) = std::sqrt(xtx);
        last_updateR_rank_ = 1;
        return newR;
    }

    // Cross-products with the within-group coupling on same-group actives.
    Eigen::VectorXd cross_prod(m);
    for (std::size_t k = 0; k < m; ++k) {
        double v = xnew.dot(getColumn(actives_[k]));
        if (group_of_unified_[actives_[k]] == g_new) { v += coupling; }
        cross_prod(static_cast<Eigen::Index>(k)) = v;
    }

    Eigen::VectorXd r = backsolveT(R_, cross_prod);
    double rpp_sq = xtx - r.dot(r);
    double rpp{};
    int new_rank = static_cast<int>(R_.rows());

    if (rpp_sq < eps * xtx) {
        rpp = eps;
    } else {
        rpp = std::sqrt(rpp_sq);
        ++new_rank;
    }

    Eigen::MatrixXd newR = Eigen::MatrixXd::Zero(R_.rows() + 1, R_.cols() + 1);
    if (R_.rows() > 0 && R_.cols() > 0) {
        newR.topLeftCorner(R_.rows(), R_.cols()) = R_;
    }
    if (r.size() > 0) {
        newR.block(0, R_.cols(), R_.rows(), 1) = r;
    }
    newR(R_.rows(), R_.cols()) = rpp;

    last_updateR_rank_ = new_rank;
    return newR;
}


void TIENET_Solver::computeGroupDirSums(
    const Eigen::Ref<const Eigen::VectorXd>& w_A) {

    group_dir_sums_.setZero();
    for (std::size_t k = 0; k < actives_.size(); ++k) {
        group_dir_sums_(group_of_unified_[actives_[k]]) +=
            w_A(static_cast<Eigen::Index>(k));
    }
}


std::pair<double, Eigen::VectorXd> TIENET_Solver::computeStepSizeIEN(
    double Cmax,
    const Eigen::Ref<const Eigen::VectorXd>& u) const {

    std::size_t num_inactives = inactives_.size();
    Eigen::VectorXd a(num_inactives);
    double gamma = std::numeric_limits<double>::max();

    #pragma omp parallel
    {
        double local_gamma = std::numeric_limits<double>::max();

        #pragma omp for schedule(static) nowait
        for (std::size_t i = 0; i < num_inactives; ++i) {
            std::size_t j = inactives_[i];
            const int g_j = group_of_unified_[j];

            // IEN joining rate: a_j = x_j^T u + (lambda2/p_m) tau_m.
            // The group-coupling term is the drift of sigma_m along the step
            // (the penalty-row component of the augmented-system inner
            // product); it vanishes when G_m has no active members
            // (tau_m = 0).
            double a_j = getColumn(j).dot(u) +
                         l2_over_pm_(g_j) * group_dir_sums_(g_j);
            a(static_cast<Eigen::Index>(i)) = a_j;
            double c_j = correlations_(static_cast<Eigen::Index>(j));

            double g1 = (Cmax - c_j) / (A_A_ - a_j);
            double g2 = (Cmax + c_j) / (A_A_ + a_j);

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
        if (num_inactives == 0) {
            logInfo("Inactive set empty; taking terminal LS step.");
        } else {
            logWarning("No valid gamma among " + std::to_string(num_inactives)
                       + " inactive candidates; using fallback step.");
        }
        return {Cmax / A_A_, a};
    }

    return {gamma, a};
}


void TIENET_Solver::updateCorrelationsIEN(
    double gamma,
    const Eigen::Ref<const Eigen::VectorXd>& a) {

    std::size_t num_inactives = inactives_.size();

    if ((currentStep_ % kahan_refresh_interval_) == 0) {
        // Full refresh: recompute the group sums from the running sparse
        // coefficients first (removes incremental fp drift in sigma), then
        // the modified correlations c_j = x_j^T r - (lambda2/p_m) sigma_m.
        group_sums_.setZero();
        for (std::size_t s = 0; s < beta_idx_.size(); ++s) {
            group_sums_(group_of_unified_[beta_idx_[s]]) += beta_val_[s];
        }

        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < num_inactives; ++i) {
            std::size_t j = inactives_[i];
            const int g_j = group_of_unified_[j];
            Eigen::Index ji = static_cast<Eigen::Index>(j);
            correlations_(ji) = getColumn(j).dot(r_) -
                                l2_over_pm_(g_j) * group_sums_(g_j);
            correlation_compensation_(ji) = 0.0;
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


void TIENET_Solver::refreshDroppedCorrelation(std::size_t dropped_var) {
    const int g_j = group_of_unified_[dropped_var];
    Eigen::Index j = static_cast<Eigen::Index>(dropped_var);
    correlations_(j) = getColumn(dropped_var).dot(r_) -
                       l2_over_pm_(g_j) * group_sums_(g_j);
    correlation_compensation_(j) = 0.0;
}


// ============================================================================
// Serialization
// ============================================================================

void TIENET_Solver::save(const std::string& filename) const { saveImpl(*this, filename); }


TIENET_Solver TIENET_Solver::load(const std::string& filename,
                                  Eigen::Map<Eigen::MatrixXd>& X,
                                  Eigen::Map<Eigen::MatrixXd>& D) {
    return loadImpl<TIENET_Solver>(filename, X, D);
}

// ============================================================================
} // End of namespace trex::tsolvers::linear_model::lars_based
