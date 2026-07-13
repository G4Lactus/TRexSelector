// ============================================================================
// tenet_solver.cpp
// ============================================================================
/**
 * @file tenet_solver.cpp
 * @brief Implementation of T-ElasticNet solver via T-LARS-EN algorithm with
 * L2-modified Cholesky updates and dummy variables.
 */
 // ============================================================================

// std includes
#include <algorithm>
#include <fstream>
#include <stdexcept>

// tsolvers includes
#include <tsolvers/linear_model/lars_based/tenet_solver.hpp>

// utils includes
#include <utils/fp_classify/fp_classify.hpp>

// ============================================================================

// Embedded into namespace trex::tsolvers::linear_model::lars_based
namespace trex::tsolvers::linear_model::lars_based {

namespace fpc = trex::utils::fp_classify;

// ============================================================================
// Constructors
// ============================================================================

TENET_Solver::TENET_Solver(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D,
    Eigen::Map<Eigen::VectorXd>& y,
    double lambda2,
    bool normalize,
    bool intercept,
    bool verbose,
    ScalingMode scaling_mode
)
    : TLARS_Solver(X, D, y, normalize, intercept, verbose,
                   SolverTypeLarsBased::TENET, scaling_mode),
      lambda2_(validateLambda2(lambda2)),
      d1_(std::sqrt(lambda2)),
      d2_(1.0 / std::sqrt(1.0 + lambda2)),
      l1norm_(0.0) {

    std::size_t p_total = p_original_ + num_dummies_;

    // EN-specific: Ridge allows full-rank active set (Zou & Hastie 2005)
    // if (lambda2 > 0) maxvars = p
    // if (lambda2 == 0) maxvars = min(p, n-1)
    std::size_t pcols_available = p_total - dropped_indices_.size();
    if (lambda2_ > 0.0) {
        max_actives_ = pcols_available;
    } else {
        max_actives_ = std::min(pcols_available, effective_n_);
    }

    // Max steps setup (8 times is the Hastie rule to support variable cycling in Lasso variants)
    maxSteps_ = 8 * max_actives_;

    // EN-specific: Initialize augmented residuals r = [y; 0_p]
    r_.resize(static_cast<Eigen::Index>(y_.size() + p_total));
    r_.head(y_.size()) = y_;
    r_.tail(p_total).setZero();

    // The TLARS base constructor already computed correlations_ = X^T y over
    // all inactive columns; at initialization the EN formula only rescales
    // them by d2 (r_obs = y, penalty residual = 0), so no second O(np) pass
    // is needed.
    correlations_ *= d2_;
}

double TENET_Solver::validateLambda2(double x) {
    if (x < 0.0) {
        throw std::invalid_argument("Ridge penalty lambda2 must be non-negative");
    }
    return x;
}


// ============================================================================
// Core executeStep()
// ============================================================================

void TENET_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();
    EigenSingleThreadGuard eigen_single_thread_guard;

    while (
        currentStep_ < maxSteps_ &&
        !inactives_.empty() &&
        actives_.size() < max_actives_ &&
        (!early_stop || T_stop == 0 || count_active_dummies_ < T_stop)
    ) {

        // ========================================================
        // STEP 1: Extract correlations for inactives
        // ========================================================

        auto [Cmax, new_vars] = findTiedMaxCorrelations();
        if (new_vars.empty() || Cmax < 100 * eps_) {
            logInfo("No variables to add or max correlation approx 0; exiting.");
            break;
        }
        pruneTiedDummies(new_vars, T_stop, early_stop);

        // ========================================================
        // STEP 2: Active set update
        // ========================================================

        // EXACT L1 Tracking: Initialize at Step 0 (Matches R: penalty <- max(abs(Cvec)))
        if (currentStep_ == 0 && lambda_.empty()) {
            lambda_.push_back(Cmax);
        }

        currentStep_++;
        actions_.emplace_back(updateActiveSet(new_vars));

        if (actives_.empty() || R_.size() == 0) {
            logWarning("Active set or Cholesky is empty; aborting.");
            // Roll back the aborted step (the initial lambda_ entry is kept;
            // rejections remain recorded in dropped_indices_) and rebuild
            // inactives_ so dropped columns cannot be re-selected.
            actions_.pop_back();
            currentStep_--;
            updateInactiveSet();
            break;
        }

        // ========================================================
        // STEP 3: Equi-angular computation
        // ========================================================

        Eigen::VectorXd sign_vec = signVector();
        Eigen::VectorXd w_A = equiangularDirection(sign_vec);

        // ========================================================
        // STEP 4: Compute equiangular vector u
        // ========================================================

        Eigen::VectorXd u1 = equiangularVectorU1(w_A);
        Eigen::VectorXd u2 = equiangularVectorU2(w_A);
        Eigen::VectorXd u(u1.size() + u2.size());
        u << u1, u2;

        // ========================================================
        // STEP 5: update maxvars if collinear variables are
        // discovered. For EN (lambda2 > 0) all non-collinear
        // variables can be active.
        // ========================================================

        if (any_dropped_) {
            std::size_t p_total = p_original_ + num_dummies_;
            max_actives_= (lambda2_ > 0) ?
                            p_total - dropped_indices_.size() :
                            std::min(p_total - dropped_indices_.size(), effective_n_);
        }


        // ========================================================
        // STEP 6: Compute step size gamma for joining times
        // ========================================================

        std::vector<bool> drops(actives_.size(), false);
        double gamma_entry = computeStepSizeEN(Cmax, u1, u2);
        double gamma = computeGammaSignChange(gamma_entry, drops, w_A);

        // ========================================================
        // STEP 7: State Updates
        // ========================================================

        // Update beta path
        updateBetaPath(w_A, gamma);

        // Update augmented residuals
        r_ -= gamma * u;

        // Update penalty/lambda: penalty[k] = penalty[k-1] - |gamma * A|
        // -> tracks the decreasing L1 constraint as we move along the path
        double next_lambda = lambda_.back() - std::abs(gamma * A_A_);
        lambda_.push_back(std::max(0.0, next_lambda));

        // Update diagnostics
        double rss = r_.head(y_.size()).dot(r_.head(y_.size()));
        RSS_.push_back(rss);
        R2_.push_back(1.0 - rss / RSS_[0]);

        // Compute L1 norm for active coefficients (running sparse state holds
        // this step's values; drops below zero their entries afterwards)
        l1norm_ = 0.0;
        for (std::size_t j : actives_) {
            l1norm_ += std::abs(runningBeta(j));
        }
        l1norm_ /= d2_;

        // ========================================================
        // STEP 9: Handle LASSO drops
        // ========================================================
        updateCorrelationsENET();
        processLassoDrops(drops);
        // Record after the drops so this step's snapshot has the dropped
        // coefficients at exactly zero (absent from the support).
        recordBetaStep();

        // T-ENET: Record dummy count and DoF after all adds/removes
        updateDummyTracking();
        DoF_.push_back(actives_.size() + (intercept_ ? 1 : 0));

        // ========================================================
        // STEP 10: Update inactive set
        // ========================================================
        updateInactiveSet();
    }
}


// ============================================================================
// Internal Helper
// ============================================================================



void TENET_Solver::updateCorrelationsENET() {
    // EN-specific correlation formula (Zou & Hastie 2005)
    // Cvec <- (drop(t(residuals[1:n]) %*% x) + d1 * residuals[-(1:n)]) * d2
    // where residuals is augmented (n + p) vector: [y_residuals; coefficient_penalties]

    std::size_t n = X_->rows();
    // First n elements (observation residuals)
    Eigen::VectorXd r_obs = r_.head(n);
    // Last p elements (coefficient penalties)
    Eigen::VectorXd r_coef = r_.tail(r_.size() - n);

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < inactives_.size(); ++i) {
        std::size_t j = inactives_[i];
        // C_j = (X_j^T * r_obs + d1 * r_coef[j]) * d2
        double corr_obs = getColumn(j).dot(r_obs);
        double corr_pen = d1_ * r_coef[static_cast<Eigen::Index>(j)];
        correlations_[static_cast<Eigen::Index>(j)] = (corr_obs + corr_pen) * d2_;
    }
}


Eigen::MatrixXd TENET_Solver::updateR(const Eigen::Ref<const Eigen::VectorXd>& xnew, double eps) {

    std::size_t m = actives_.size();

    // Elastic Net Gram matrix with (1 + lambda2) scaling
    double d3 = 1 + lambda2_;
    double xtx = (xnew.dot(xnew) + lambda2_) / d3;
    double norm_xnew = std::sqrt(xtx);

    // First variable case: create a (1,1) matrix (scalar)
    if (R_.size() == 0 || m == 0) {
        Eigen::MatrixXd newR(1, 1);
        newR(0, 0) = norm_xnew;
        last_updateR_rank_ = 1;  // First pivot: rank is 1
        return newR;
    }

    // Compute cross-products: Xtx = Xold^T * xnew with (1 + lambda2) scaling
    Eigen::VectorXd cross_prod(m);
    for (std::size_t k = 0; k < m; ++k) {
        cross_prod[static_cast<Eigen::Index>(k)] = xnew.dot(getColumn(actives_[k])) / d3;
    }

    // Solve R^T * r = Xtx for r (r is length m)
    Eigen::VectorXd r = backsolveT(R_, cross_prod);

    // Compute new diagonal entry
    double rpp_sq = xtx - r.dot(r);
    double rpp{};
    int new_rank = static_cast<int>(R_.rows());  // Current rank backup

    if (rpp_sq < eps * xtx) {
        // xnew is numerically in span(X_A)
        rpp = eps;  // Defensive: eps to avoid nan, no rank increment!
        // new_rank remains unchanged
    } else {
        rpp = std::sqrt(rpp_sq);
        ++new_rank;               // True rank increases
    }

    // Build new augmented Cholesky matrix (m+1 by m+1)
    Eigen::MatrixXd newR(m + 1, m + 1);
    newR.setZero();
    if (m > 0) {
        newR.topLeftCorner(m, m) = R_;
        newR.block(0, static_cast<Eigen::Index>(m),
                   m, 1) = r;
    }
    newR(static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(m)) = rpp;

    // Update global state variable for rank
    last_updateR_rank_ = new_rank;

    return newR;
}


Eigen::VectorXd TENET_Solver::equiangularVectorU1(
    const Eigen::Ref<const Eigen::VectorXd>& w_A
) const {

    Eigen::VectorXd u1 = Eigen::VectorXd::Zero(X_->rows());
    for (std::size_t k = 0; k < actives_.size(); ++k) {
        u1 += w_A[static_cast<Eigen::Index>(k)] * getColumn(actives_[k]) * d2_;
    }
    return u1;
}


Eigen::VectorXd TENET_Solver::equiangularVectorU2(
    const Eigen::Ref<const Eigen::VectorXd>& w_A
) const {

    double factor = d1_ * d2_;
    std::size_t p_total = p_original_ + num_dummies_;
    Eigen::VectorXd u2 = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(p_total));
    for (std::size_t k = 0; k < actives_.size(); ++k) {
        u2[static_cast<Eigen::Index>(actives_[k])] += w_A[static_cast<Eigen::Index>(k)] * factor;
    }
    return u2;
}


double TENET_Solver::computeStepSizeEN(double Cmax,
                                       const Eigen::Ref<const Eigen::VectorXd>& u1,
                                       const Eigen::Ref<const Eigen::VectorXd>& u2) const {

    // Case 1: Active set saturated: take full step
    if (actives_.size() >= max_actives_) { return Cmax / A_A_; }

    // Case 2: Compute joining times for inactives
    double gamma_min = Cmax / A_A_;

    for (std::size_t j :inactives_) {

        // Compute rate of correlation change for variable j
        // a_j = d2 * (u1^T X_j + d1 * u2_j)
        double a_j = d2_ * (getColumn(j).dot(u1) + d1_ * u2[static_cast<Eigen::Index>(j)]);
        double C_j = correlations_[static_cast<Eigen::Index>(j)];

        // Two cases when |C_j| meets Cmax
        // Case (+): (Cmax - C_j) / (A_A_ - a_j) -> positive step
        // Case (-): (Cmax + C_j) / (A_A_ + a_j) -> negative step
        double denom1 = A_A_ - a_j;
        double denom2 = A_A_ + a_j;

        if (std::abs(denom1) > eps_) {
            double gamma1 = (Cmax - C_j) / denom1;
            if (gamma1 > eps_ && gamma1 < gamma_min) {
                gamma_min = gamma1;
            }
        }

        if (std::abs(denom2) > eps_) {
            double gamma2 = (Cmax + C_j) / denom2;
            if (gamma2 > eps_ && gamma2 < gamma_min) {
                gamma_min = gamma2;
            }
        }
    }

    return gamma_min;
}




void TENET_Solver::refreshDroppedCorrelation(std::size_t dropped_var) {
    // EN-specific correlation for potential re-entry (Zou & Hastie 2005):
    // C_j = (x_j^T r_obs + d1 * r_coef[j]) * d2 over the augmented residual
    Eigen::Index n = X_->rows();
    Eigen::Index j = static_cast<Eigen::Index>(dropped_var);
    double corr_obs = getColumn(dropped_var).dot(r_.head(n));
    double corr_pen = d1_ * r_(n + j);
    correlations_(j) = (corr_obs + corr_pen) * d2_;
    correlation_compensation_(j) = 0.0;
}



std::vector<double> TENET_Solver::getCp() const {
    std::vector<double> cp_out;

    // If no RSS or DoF information, return empty
    if (RSS_.empty() || DoF_.empty()) { return cp_out; }

    // Derive n from serialized state so Cp is available on disconnected solvers
    std::size_t n = effective_n_ + (intercept_ ? 1 : 0);
    double rss_final = RSS_.back();
    double p_total = static_cast<double>(p_original_ + num_dummies_);

    if (!fpc::isfinite(rss_final) || rss_final <= 0) {
        cp_out.assign(RSS_.size(), fpc::quiet_nan());
        return cp_out;
    }

    // The residual-variance estimate rss_final / (n - p_total - 1) requires
    // n > p_total + 1; in the high-dimensional regime (typical for T-Rex,
    // where p_total = p + num_dummies) Cp is undefined — return NaN instead
    // of garbage from a negative variance estimate.
    if (static_cast<double>(n) - p_total - 1.0 <= 0.0) {
        cp_out.assign(RSS_.size(), fpc::quiet_nan());
        return cp_out;
    }

    cp_out.reserve(RSS_.size());
    for (std::size_t k = 0; k < RSS_.size(); ++k) {
        double dof = static_cast<double>(DoF_[k]);

        if (DoF_[k] < n && RSS_[k] > 0 && fpc::isfinite(RSS_[k])) {
            // MATCHES R CODE: ((n - m - 1) * RSS) / RSS_final - n + 2 * df
            // Note: first term uses p_total (m), second term uses dof (df)
            double cp_val = ((static_cast<double>(n) - p_total - 1.0) * RSS_[k]) / rss_final
                          - static_cast<double>(n) + 2.0 * dof;
            cp_out.push_back(cp_val);
        } else {
            cp_out.push_back(fpc::quiet_nan());
        }
    }

    return cp_out;
}

// ============================================================================
// Coefficient Accessors with EN Scaling
// ============================================================================

Eigen::VectorXd TENET_Solver::getBeta(int step) const {
    if (betaPathSparse_.empty()) throw std::runtime_error("No path available.");
    std::size_t use_step = (step < 0) ? betaPathSparse_.size() - 1
                                      : static_cast<std::size_t>(step);
    if (use_step >= betaPathSparse_.size()) {
        throw std::invalid_argument(
            concatMsg(solverTypeToString(), "::getBeta: step ", step,
                      " out of range [0, ", betaPathSparse_.size() - 1, "].")
        );
    }

    // Recover naive elastic net coefficients (beta_pure = beta_aug / d2)
    Eigen::VectorXd coef = Eigen::VectorXd::Zero(
        static_cast<Eigen::Index>(p_original_ + num_dummies_));
    const SparseBetaStep& s = betaPathSparse_[use_step];
    for (std::size_t k = 0; k < s.idx.size(); ++k) {
        coef(static_cast<Eigen::Index>(s.idx[k])) = s.val[k];
    }
    coef /= d2_;

    // De-normalize real predictors and dummies with their respective norms
    if (normalize_) {
        if (normsx_.size() > 0) coef.head(p_original_).array() /= normsx_.array();
        if (normsd_.size() > 0) coef.tail(num_dummies_).array() /= normsd_.array();
    }
    return coef;
}


Eigen::MatrixXd TENET_Solver::getBetaPath() const {
    // EN-specific: Divide by d₂ = 1/√(1+λ₂), i.e., multiply by √(1+λ₂)
    // This matches R enet post-processing: beta.pure <- beta.pure / d2
    Eigen::MatrixXd beta_orig = densifyBetaPath() / d2_;

    // De-normalize real predictors and dummies with their respective norms
    if (normalize_) {
        for (Eigen::Index j = 0; j < beta_orig.cols(); ++j) {
            if (normsx_.size() > 0) {
                beta_orig.col(j).head(p_original_).array() /= normsx_.array();
            }
            if (normsd_.size() > 0) {
                beta_orig.col(j).tail(num_dummies_).array() /= normsd_.array();
            }
        }
    }
    return beta_orig;
}


SparseBetaPath TENET_Solver::getBetaPathSparse() const {
    // Same scaling chain as getBetaPath(): val / d2 first, then column norms.
    return makeScaledSparsePath(d2_);
}


double TENET_Solver::getAlpha() const {
    // Compute mixing parameter α = λ₁/(λ₁ + λ₂)
    if (lambda_.empty()) { return 1.0; }

    // lambda_ holds the initial penalty plus one entry per step; the current
    // value lives at index currentStep_.
    std::size_t idx = std::min(currentStep_, lambda_.size() - 1);

    double lambda1 = lambda_[idx];
    double total = lambda1 + lambda2_;
    // Guard against division by zero (both penalties zero = no regularization)
    if (total < eps_) return 1.0;
    return lambda1 / total;
}


// ============================================================================
// Serialization
// ============================================================================

void TENET_Solver::save(const std::string& filename) const { saveImpl(*this, filename); }


TENET_Solver TENET_Solver::load(const std::string& filename,
                                Eigen::Map<Eigen::MatrixXd>& X,
                                Eigen::Map<Eigen::MatrixXd>& D) {
    return loadImpl<TENET_Solver>(filename, X, D);
}


void TENET_Solver::reconnect(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::MatrixXd>& D) {
    TSolver_Base::reconnect(X, D);

    std::size_t p_total = p_original_ + num_dummies_;
    std::size_t expected_r_size = X.rows() + p_total;
    if (r_.size() > 0 && static_cast<std::size_t>(r_.size()) != expected_r_size) {
        throw std::runtime_error("Augmented residual dimension mismatch.");
    }
}

// ============================================================================
} // namespace trex::tsolvers::linear_model::lars_based
