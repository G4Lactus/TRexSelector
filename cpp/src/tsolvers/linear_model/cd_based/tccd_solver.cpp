// ============================================================================
// tccd_solver.cpp
// ============================================================================
/**
 * @file tccd_solver.cpp
 *
 * @brief Implementation (.cpp) file for the Terminating Cyclic Coordinate
 * Descent (T-CCD) solver.
 */
// ============================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

// tsolvers includes
#include <tsolvers/linear_model/cd_based/tccd_solver.hpp>
#include <tsolvers/solver_utils/solver_preprocessing.hpp>

// utils includes
#include <utils/serialization/utils_cereal_eigen.hpp>
#include <utils/openmp/utils_openmp.hpp>
#include <utils/fp_classify/fp_classify.hpp>

// ============================================================================

// Embedded into trex::tsolvers::linear_model::cd_based namespace
namespace trex::tsolvers::linear_model::cd_based {

// ============================================================================

namespace fpc = trex::utils::fp_classify;

namespace {
inline double softThreshold(double z, double lambda1) {
    if (z > lambda1)  { return z - lambda1; }
    if (z < -lambda1) { return z + lambda1; }
    return 0.0;
}
} // anonymous namespace


// ============================================================================
// Constructors
// ============================================================================

TCCD_Solver::TCCD_Solver(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D,
    Eigen::Map<Eigen::VectorXd>& y,
    bool normalize,
    bool intercept,
    bool verbose,
    SolverTypeCdBased algorithm_type,
    ScalingMode scaling_mode
    )
    : TSolver_Base(X, D, y, normalize, intercept, verbose, scaling_mode),
      algo_type_(algorithm_type) {

    // 1. Inherited preprocessing handles X, D, y centering and normalization
    dropped_indices_.clear();
    if (normalize_ || intercept_) { preprocess(); }
    initializeInactives();

    // 2. State Limit (recorded steps: one per dummy crossing / grid point)
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

    // 4. Initialize CCD specific states
    lambda_.reserve(maxSteps_ + 1);
    initializeCorrelations();
    lambda_max_ = correlations_.cwiseAbs().maxCoeff();
    lambda_current_ = lambda_max_;
}


TCCD_Solver::TCCD_Solver(SolverTypeCdBased type) : TSolver_Base(), algo_type_(type) {}
TCCD_Solver::TCCD_Solver() : TSolver_Base(), algo_type_(SolverTypeCdBased::TCCD) {}


// ============================================================================
// Core Algorithm Implementation
// ============================================================================

void TCCD_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();

    // Set openMP dominance for the entire algorithm
    EigenSingleThreadGuard eigen_single_thread_guard;

    ensureCdState();

    const bool full_path = (!early_stop || T_stop == 0);
    std::size_t grid_remaining = grid_points_;
    const long point_budget = lambda_points_ +
                              static_cast<long>(max_lambda_points_);

    // Main T-CCD loop with early stopping based on dummy variable count
    while (
        currentStep_ < maxSteps_ &&
        actives_.size() < effective_n_ &&
        lambda_points_ < point_budget &&
        (full_path
             ? (grid_remaining > 0 &&
                lambda_current_ > lambda_min_ratio_ * lambda_max_ * (1.0 + 1e-12))
             : count_active_dummies_ < T_stop)
    ) {

        // ========================================================
        // Full-path mode: geometric lambda1 grid (glmnet-style)
        // ========================================================
        if (full_path) {
            const double decay =
                std::pow(lambda_min_ratio_,
                         1.0 / static_cast<double>(grid_points_));
            double lam_next = std::max(lambda_current_ * decay,
                                       lambda_min_ratio_ * lambda_max_);
            cd_sweeps_ += solveAtLambda(lam_next, cd_tol_, nullptr);
            lambda_points_++;
            lambda_current_ = lam_next;
            grid_remaining--;
            recordCrossingStep(lam_next);
            continue;
        }

        // ========================================================
        // STEP 1: Predict the next dummy crossing and jump
        // ========================================================
        const std::size_t next_T = count_active_dummies_ + 1;

        double target = predictNextCrossing();
        double lam_next = std::min(target * jump_margin_,
                                   lambda_current_ * 0.999);
        if (!(lam_next > 0.0)) { lam_next = lambda_current_ * 0.5; }
        if (lam_next < lambda_current_ * 1e-4) {
            lam_next = lambda_current_ * 1e-4;  // jump floor (safety)
        }

        std::vector<std::size_t> support_hi(actives_);   // bracket top support
        cd_sweeps_ += solveAtLambda(lam_next, cd_tol_probe_, nullptr);
        lambda_points_++;
        std::size_t cd = countActiveDummiesCd();

        if (cd < next_T) {                       // undershoot: re-predict
            lambda_current_ = lam_next;
            count_active_dummies_ = cd;
            continue;
        }

        // ========================================================
        // STEP 2: Localize the crossing on [lam_next, lambda_current_]
        // ========================================================
        std::vector<std::size_t> candidates;
        collectCandidates(lam_next, candidates);

        double lo = lam_next;
        double hi = lambda_current_;
        std::vector<std::size_t> support_lo(actives_);
        std::size_t count_lo = cd;

        while (true) {
            // Support-nesting early stop: only when the bracket's support
            // changes are entering dummies AND the bracket bottom holds
            // EXACTLY the target count — then the crossing's support is
            // fully determined. With more dummies than the target at the
            // bottom, keep narrowing so every dummy crossing is recorded at
            // its own count (skipped counts would silently zero their
            // phi_T contributions in the T-Rex calibration); the later
            // crossings are re-localized by subsequent loop iterations.
            if (count_lo == next_T &&
                supportOnlyDummyGrowth(support_hi, support_lo)) { break; }
            if ((hi - lo) / hi <= lambda_rel_tol_) { break; }

            double mid = 0.5 * (lo + hi);
            cd_sweeps_ += solveAtLambda(mid, cd_tol_probe_, &candidates);
            lambda_points_++;
            const std::size_t cmid = countActiveDummiesCd();
            if (cmid >= next_T) {
                lo = mid;
                support_lo = actives_;
                count_lo = cmid;
            } else {
                hi = mid;
                support_hi = actives_;
            }
        }

        // ========================================================
        // STEP 3: Certify the crossing with a full KKT pass and record
        // ========================================================
        cd_sweeps_ += solveAtLambda(lo, cd_tol_, nullptr);
        lambda_points_++;
        lambda_current_ = lo;
        std::size_t cd_lo = countActiveDummiesCd();
        count_active_dummies_ = cd_lo;
        if (cd_lo < next_T) { continue; }        // certification moved the count

        recordCrossingStep(lo);
    }
}


// ============================================================================
// Helper Implementations
// ============================================================================

void TCCD_Solver::ensureCdState() {
    if (cd_ready_) { return; }

    const std::size_t p_total = p_original_ + num_dummies_;
    const Eigen::Index pt = static_cast<Eigen::Index>(p_total);

    // Fresh start (constructor path) vs reconnect/load (state deserialized).
    if (beta_cd_.size() != pt) {
        beta_cd_ = Eigen::VectorXd::Zero(pt);
        l2kb_ = Eigen::VectorXd::Zero(pt);
        in_active_.assign(p_total, 0);
        r_ = y_;
    }

    // Dropped-column lookup.
    is_dropped_.assign(p_total, 0);
    for (std::size_t j : dropped_indices_) { is_dropped_[j] = 1; }

    // Squared column norms (1.0 for unit-L2 standardized columns).
    colnorm2_.setOnes(pt);
    for (std::size_t j = 0; j < p_total; ++j) {
        if (!is_dropped_[j]) { colnorm2_(static_cast<Eigen::Index>(j)) =
            getColumn(j).squaredNorm(); }
    }

    // Pristine X^T y (correlations_ tracks the path, so recompute).
    correlations_y_.setZero(pt);
    #pragma omp parallel for schedule(static)
    for (std::size_t j = 0; j < p_total; ++j) {
        if (!is_dropped_[j]) { correlations_y_(static_cast<Eigen::Index>(j)) =
            getColumn(j).dot(y_); }
    }

    // Penalty diagonal over the unified space.
    configurePenaltyDiag();

    // Empty gram cache (ever-active slots are rebuilt lazily).
    slot_of_.assign(p_total, -1);
    slot_var_.clear();
    gram_rows_.clear();
    gram_q_.resize(0);
    gram_active_ = true;
    r_dirty_ = false;

    // Re-enter the current support into the fresh gram cache: after
    // reconnect()/load the slots are empty while actives_ is not, and the
    // sweep path reads gram_q_(slot_of_[j]) — slot -1 would be indexed.
    if (!actives_.empty()) {
        if (actives_.size() >= gram_cap_) {
            gram_active_ = false;    // naive sweeps; r_ is (de)serialized
        } else {
            for (std::size_t j : actives_) { gramEnter(j); }
            // gramEnter maintains q incrementally under the zero-coefficient
            // entry invariant of the solve loop; the deserialized support
            // carries nonzero coefficients, so recompute
            // q(s) = x_s^T (y - X beta) over the completed gram directly.
            const Eigen::Index S = static_cast<Eigen::Index>(slot_var_.size());
            for (Eigen::Index s = 0; s < S; ++s) {
                double q = correlations_y_(
                    static_cast<Eigen::Index>(slot_var_[static_cast<std::size_t>(s)]));
                for (Eigen::Index t = 0; t < S; ++t) {
                    q -= gram_rows_[static_cast<std::size_t>(s)](t) *
                         beta_cd_(static_cast<Eigen::Index>(
                             slot_var_[static_cast<std::size_t>(t)]));
                }
                gram_q_(s) = q;
            }
        }
    }

    cd_ready_ = true;
}


void TCCD_Solver::configurePenaltyDiag() {
    const std::size_t p_total = p_original_ + num_dummies_;
    penalty_diag_.setZero(static_cast<Eigen::Index>(p_total));
    if (penalty_kind_ == PenaltyKind::NONE || lambda2_ <= 0.0) { return; }
    if (penalty_kind_ == PenaltyKind::GROUP) {
        throw std::logic_error(concatMsg(
            solverTypeToString(),
            "::configurePenaltyDiag: GROUP kind requires a specialized "
            "descendant override."));
    }

    if (penalty_kind_ == PenaltyKind::DIAGONAL) {
        if (static_cast<std::size_t>(penalty_dvec_.size()) != p_original_) {
            throw std::invalid_argument(concatMsg(
                solverTypeToString(),
                "::configurePenaltyDiag: diagonal weights size (",
                penalty_dvec_.size(), ") does not match p (", p_original_, ")"));
        }
        penalty_diag_.head(static_cast<Eigen::Index>(p_original_)) =
            lambda2_ * penalty_dvec_;
    } else {  // SPARSE
        if (static_cast<std::size_t>(tikhonov_K_.rows()) != p_original_ ||
            static_cast<std::size_t>(tikhonov_K_.cols()) != p_original_) {
            throw std::invalid_argument(concatMsg(
                solverTypeToString(),
                "::configurePenaltyDiag: Tikhonov matrix is ",
                tikhonov_K_.rows(), "x", tikhonov_K_.cols(),
                ", expected ", p_original_, "x", p_original_));
        }
        for (std::size_t j = 0; j < p_original_; ++j) {
            penalty_diag_(static_cast<Eigen::Index>(j)) =
                lambda2_ * tikhonov_K_.coeff(static_cast<Eigen::Index>(j),
                                             static_cast<Eigen::Index>(j));
        }
    }
    // Dummy block: always ridge with kappa_dummy_ (exchangeability default).
    penalty_diag_.tail(static_cast<Eigen::Index>(num_dummies_))
        .setConstant(lambda2_ * kappa_dummy_);
}


void TCCD_Solver::penaltyRankUpdate(std::size_t j, double delta) {
    switch (penalty_kind_) {
        case PenaltyKind::NONE:
            return;
        case PenaltyKind::DIAGONAL:
            l2kb_(static_cast<Eigen::Index>(j)) +=
                delta * penalty_diag_(static_cast<Eigen::Index>(j));
            return;
        case PenaltyKind::SPARSE:
            if (j < p_original_) {
                const double f = lambda2_ * delta;
                for (Eigen::SparseMatrix<double>::InnerIterator
                         it(tikhonov_K_, static_cast<Eigen::Index>(j));
                     it; ++it) {
                    l2kb_(it.row()) += f * it.value();
                }
            } else {  // dummy ridge block
                l2kb_(static_cast<Eigen::Index>(j)) +=
                    delta * penalty_diag_(static_cast<Eigen::Index>(j));
            }
            return;
        case PenaltyKind::GROUP:
            throw std::logic_error(concatMsg(
                solverTypeToString(),
                "::penaltyRankUpdate: GROUP kind requires a specialized "
                "descendant override."));
    }
}


double TCCD_Solver::coordinateUpdate(std::size_t j, double xr, double lambda1) {
    const Eigen::Index ji = static_cast<Eigen::Index>(j);
    const double n2 = colnorm2_(ji);
    const double denom = n2 + penalty_diag_(ji);
    const double b = xr + n2 * beta_cd_(ji) -
                     (penaltyGradient(j) - penalty_diag_(ji) * beta_cd_(ji));
    const double bn = softThreshold(b, lambda1) / denom;
    const double delta = bn - beta_cd_(ji);
    if (delta != 0.0) {
        beta_cd_(ji) = bn;
        penaltyRankUpdate(j, delta);
    }
    return delta;
}


void TCCD_Solver::refreshResidualCd() {
    r_ = y_;
    for (std::size_t j : actives_) {
        const Eigen::Index ji = static_cast<Eigen::Index>(j);
        if (beta_cd_(ji) != 0.0) { r_ -= beta_cd_(ji) * getColumn(j); }
    }
    r_dirty_ = false;
}


Eigen::Index TCCD_Solver::gramEnter(std::size_t j) {
    if (slot_of_[j] >= 0) { return slot_of_[j]; }    // ever-active: free re-entry

    const Eigen::Index s = static_cast<Eigen::Index>(slot_var_.size());
    slot_var_.push_back(j);
    for (auto& row : gram_rows_) {
        row.conservativeResize(s + 1);
        row(s) = 0.0;
    }
    gram_rows_.emplace_back(Eigen::VectorXd::Zero(s + 1));
    gram_q_.conservativeResize(s + 1);
    slot_of_[j] = s;

    double qs = correlations_y_(static_cast<Eigen::Index>(j));
    for (Eigen::Index t = 0; t < s; ++t) {
        const std::size_t k = slot_var_[static_cast<std::size_t>(t)];
        const double g = getColumn(j).dot(getColumn(k));
        gram_rows_[static_cast<std::size_t>(s)](t) = g;
        gram_rows_[static_cast<std::size_t>(t)](s) = g;
        qs -= g * beta_cd_(static_cast<Eigen::Index>(k));
    }
    gram_rows_[static_cast<std::size_t>(s)](s) =
        colnorm2_(static_cast<Eigen::Index>(j));
    gram_q_(s) = qs;
    return s;
}


long TCCD_Solver::solveAtLambda(double lambda1, double cd_tol,
                                const std::vector<std::size_t>* candidates) {
    const std::size_t p_total = p_original_ + num_dummies_;
    long sweeps = 0;
    const long sweep_budget = static_cast<long>(max_sweeps_per_solve_);
    bool budget_hit = false;

    while (true) {
        // ========================================================
        // Inner loop: cycle the support to convergence. The stopping
        // rule is RELATIVE to the coefficient scale (support decisions
        // only need KKT-slack precision, which is relative); the sweep
        // budget bounds limit cycles on nearly collinear active sets.
        // ========================================================
        if (gram_active_) {
            while (true) {
                sweeps++;
                double max_delta = 0.0;
                double beta_scale = 1.0;
                for (std::size_t j : actives_) {
                    const Eigen::Index s = slot_of_[j];
                    const double delta =
                        coordinateUpdate(j, gram_q_(s), lambda1);
                    if (delta != 0.0) {
                        r_dirty_ = true;
                        const Eigen::Index ns =
                            static_cast<Eigen::Index>(slot_var_.size());
                        gram_q_.head(ns) -=
                            delta * gram_rows_[static_cast<std::size_t>(s)].head(ns);
                        max_delta = std::max(max_delta, std::abs(delta));
                    }
                    beta_scale = std::max(
                        beta_scale,
                        std::abs(beta_cd_(static_cast<Eigen::Index>(j))));
                }
                if (max_delta < cd_tol * beta_scale) { break; }
                if (sweeps >= sweep_budget) { budget_hit = true; break; }
            }
        } else {
            if (r_dirty_) { refreshResidualCd(); }
            while (true) {
                sweeps++;
                double max_delta = 0.0;
                double beta_scale = 1.0;
                for (std::size_t j : actives_) {
                    const double delta =
                        coordinateUpdate(j, getColumn(j).dot(r_), lambda1);
                    if (delta != 0.0) {
                        r_ -= delta * getColumn(j);
                        max_delta = std::max(max_delta, std::abs(delta));
                    }
                    beta_scale = std::max(
                        beta_scale,
                        std::abs(beta_cd_(static_cast<Eigen::Index>(j))));
                }
                if (max_delta < cd_tol * beta_scale) { break; }
                if (sweeps >= sweep_budget) { budget_hit = true; break; }
            }
        }

        // ========================================================
        // Outer loop: KKT pass over the non-support
        // ========================================================
        const std::size_t limit =
            candidates ? candidates->size() : p_total;

        bool need_residual = !gram_active_;
        if (!need_residual) {
            for (std::size_t idx = 0; idx < limit; ++idx) {
                const std::size_t j = candidates ? (*candidates)[idx] : idx;
                if (!in_active_[j] && !is_dropped_[j] && slot_of_[j] < 0) {
                    need_residual = true;
                    break;
                }
            }
        }
        if (need_residual && r_dirty_) { refreshResidualCd(); }
        if (candidates) { restricted_kkt_passes_++; } else { full_kkt_passes_++; }

        bool violated = false;
        const double thresh = lambda1 * (1.0 + kkt_slack_);
        for (std::size_t idx = 0; idx < limit; ++idx) {
            const std::size_t j = candidates ? (*candidates)[idx] : idx;
            if (in_active_[j] || is_dropped_[j]) { continue; }

            const Eigen::Index ji = static_cast<Eigen::Index>(j);
            const double cj = (gram_active_ && slot_of_[j] >= 0)
                                  ? gram_q_(slot_of_[j])
                                  : getColumn(j).dot(r_);
            correlations_(ji) = cj;

            if (std::abs(cj - penaltyGradient(j)) > thresh) {  // penalized KKT entry
                actives_.push_back(j);
                in_active_[j] = 1;
                if (gram_active_) {
                    if (slot_var_.size() >= gram_cap_ && slot_of_[j] < 0) {
                        // Ever-active set too large: gram build would cost
                        // O(n * S^2) — fall back to naive sweeps for good.
                        gram_active_ = false;
                        if (r_dirty_) { refreshResidualCd(); }
                    } else {
                        gramEnter(j);
                    }
                }
                violated = true;
            }
        }
        if (!violated) { break; }
        if (budget_hit) { break; }        // stall guard: accept current state
    }
    if (budget_hit) {
        sweep_cap_hits_++;
        if (sweep_cap_hits_ == 1) {
            logWarning(concatMsg(
                "solveAtLambda hit the sweep budget (",
                max_sweeps_per_solve_, "); accepting the current iterate "
                "(further hits counted silently, see getSweepCapHits())."));
        }
    }

    // Prune exact zeros (pruned variables keep their gram slot).
    std::vector<std::size_t> keep;
    keep.reserve(actives_.size());
    for (std::size_t j : actives_) {
        if (beta_cd_(static_cast<Eigen::Index>(j)) != 0.0) {
            keep.push_back(j);
        } else {
            in_active_[j] = 0;
        }
    }
    actives_.swap(keep);

    // Refresh correlations_ on the support side (exact values feed the
    // crossing predictor's dummy scan).
    if (gram_active_) {
        for (std::size_t j : actives_) {
            correlations_(static_cast<Eigen::Index>(j)) = gram_q_(slot_of_[j]);
        }
    } else {
        if (r_dirty_) { refreshResidualCd(); }
        for (std::size_t j : actives_) {
            correlations_(static_cast<Eigen::Index>(j)) = getColumn(j).dot(r_);
        }
    }
    return sweeps;
}


double TCCD_Solver::predictNextCrossing() {
    const std::size_t p_total = p_original_ + num_dummies_;

    // Zeroth order: an inactive dummy enters when lambda1 falls to its
    // penalized correlation |x_j^T r - lambda2 (K beta)_j|.
    double pred = 0.0;
    for (std::size_t j = dummy_start_idx_; j < p_total; ++j) {
        if (in_active_[j] || is_dropped_[j]) { continue; }
        const Eigen::Index ji = static_cast<Eigen::Index>(j);
        pred = std::max(pred,
                        std::abs(correlations_(ji) - penaltyGradient(j)));
    }

    // Secant refinement: the dummy correlations decay as lambda1 drops
    // (entering reals absorb residual), so extrapolate the decay line and
    // intersect it with the identity.
    double target = pred;
    if (pred_prev_lambda_ > 0.0 &&
        pred_prev_lambda_ > lambda_current_ * (1.0 + 1e-12)) {
        const double slope = (pred - pred_prev_value_) /
                             (lambda_current_ - pred_prev_lambda_);
        if (slope < 0.999) {
            const double lstar = (pred - slope * lambda_current_) / (1.0 - slope);
            if (lstar > 0.0 && lstar < lambda_current_) { target = lstar; }
        }
    }
    pred_prev_lambda_ = lambda_current_;
    pred_prev_value_ = pred;
    return target;
}


std::size_t TCCD_Solver::countActiveDummiesCd() const {
    std::size_t cnt = 0;
    for (std::size_t j : actives_) {
        if (j >= dummy_start_idx_ &&
            beta_cd_(static_cast<Eigen::Index>(j)) != 0.0) { cnt++; }
    }
    return cnt;
}


bool TCCD_Solver::supportOnlyDummyGrowth(
    const std::vector<std::size_t>& hi_support,
    const std::vector<std::size_t>& lo_support) const {

    std::unordered_set<std::size_t> lo(lo_support.begin(), lo_support.end());
    for (std::size_t j : hi_support) {
        if (lo.find(j) == lo.end()) { return false; }   // not nested
    }
    std::unordered_set<std::size_t> hi(hi_support.begin(), hi_support.end());
    for (std::size_t j : lo_support) {
        if (hi.find(j) == hi.end() && j < dummy_start_idx_) { return false; }
    }
    return true;
}


void TCCD_Solver::collectCandidates(double lambda_lo,
                                    std::vector<std::size_t>& candidates) const {
    const std::size_t p_total = p_original_ + num_dummies_;
    candidates.clear();
    for (std::size_t j = 0; j < p_total; ++j) {
        if (is_dropped_[j]) { continue; }
        const Eigen::Index ji = static_cast<Eigen::Index>(j);
        if (in_active_[j] ||
            std::abs(correlations_(ji) - penaltyGradient(j)) >=
                0.9 * lambda_lo) {
            candidates.push_back(j);
        }
    }
}


void TCCD_Solver::recordCrossingStep(double lambda1) {
    currentStep_++;
    lambda_.push_back(lambda1);

    // Actions: support diff vs the previously recorded step (R lars
    // convention via actionAdd/actionDrop).
    std::vector<int> actions_this_step;
    std::unordered_set<std::size_t> prev(beta_idx_.begin(), beta_idx_.end());
    std::unordered_set<std::size_t> curr(actives_.begin(), actives_.end());
    for (std::size_t j : actives_) {
        if (prev.find(j) == prev.end()) {
            actions_this_step.push_back(actionAdd(j));
            num_additions_++;
        }
    }
    for (std::size_t j : beta_idx_) {
        if (curr.find(j) == curr.end()) {
            actions_this_step.push_back(actionDrop(j));
            num_removals_++;
        }
    }
    actions_.emplace_back(std::move(actions_this_step));

    // Sparse beta snapshot of the certified solution.
    Eigen::VectorXd values(static_cast<Eigen::Index>(actives_.size()));
    for (std::size_t k = 0; k < actives_.size(); ++k) {
        values(static_cast<Eigen::Index>(k)) =
            beta_cd_(static_cast<Eigen::Index>(actives_[k]));
    }
    setRunningBeta(actives_, values);
    recordBetaStep();

    // Diagnostics.
    if (r_dirty_) { refreshResidualCd(); }
    const double rss = r_.dot(r_);
    RSS_.push_back(rss);
    R2_.push_back(1.0 - rss / RSS_[0]);
    count_active_dummies_ = countActiveDummiesCd();
    updateDummyTracking();
    DoF_.push_back(actives_.size() + (intercept_ ? 1 : 0));

    // Rebuild the inactive set (base rebuild is gated on any_dropped_).
    any_dropped_ = true;
    updateInactiveSet();
}


double TCCD_Solver::getKktViolation(double lambda1) {
    validateConnected();
    ensureCdState();
    const double lam = (lambda1 > 0.0) ? lambda1 : lambda_current_;
    if (r_dirty_) { refreshResidualCd(); }

    const std::size_t p_total = p_original_ + num_dummies_;
    double viol = 0.0;
    for (std::size_t j = 0; j < p_total; ++j) {
        if (is_dropped_[j]) { continue; }
        const Eigen::Index ji = static_cast<Eigen::Index>(j);
        const double grad = getColumn(j).dot(r_) - penaltyGradient(j);
        if (in_active_[j] && beta_cd_(ji) != 0.0) {
            viol = std::max(viol, std::abs(
                grad - lam * ((beta_cd_(ji) > 0.0) ? 1.0 : -1.0)));
        } else {
            viol = std::max(viol, std::abs(grad) - lam);
        }
    }
    return std::max(viol, 0.0);
}


// ============================================================================
// Setters
// ============================================================================

void TCCD_Solver::setLambdaRelTol(double tol) {
    if (!(tol > 0.0) || tol >= 1.0) {
        logWarning(concatMsg(
            "Lambda tolerance must be in (0, 1); keeping current value (",
            lambda_rel_tol_, ")."
        ));
        return;
    }
    lambda_rel_tol_ = tol;
}


void TCCD_Solver::setCdTolerances(double cd_tol, double cd_tol_probe) {
    if (!(cd_tol > 0.0) || !(cd_tol_probe > 0.0)) {
        logWarning("CD tolerances must be positive; keeping current values.");
        return;
    }
    cd_tol_ = cd_tol;
    cd_tol_probe_ = cd_tol_probe;
}


void TCCD_Solver::setFullPathGrid(std::size_t grid_points,
                                  double lambda_min_ratio) {
    if (grid_points < 2 || !(lambda_min_ratio > 0.0) || lambda_min_ratio >= 1.0) {
        logWarning("Invalid full-path grid configuration; keeping current values.");
        return;
    }
    grid_points_ = grid_points;
    lambda_min_ratio_ = lambda_min_ratio;
}


// ============================================================================
// Serialization Implementation
// ============================================================================

void TCCD_Solver::save(const std::string& filename) const { saveImpl(*this, filename); }


TCCD_Solver TCCD_Solver::load(const std::string& filename,
                              Eigen::Map<Eigen::MatrixXd>& X,
                              Eigen::Map<Eigen::MatrixXd>& D) {
    return loadImpl<TCCD_Solver>(filename, X, D);
}

// ==========================================================================
} /* End of namespace trex::tsolvers::linear_model::cd_based */
