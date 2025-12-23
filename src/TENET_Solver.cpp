#include "TENET_Solver.hpp"

// ============================================================================
// Constructors
// ============================================================================

TENET_Solver::TENET_Solver(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::VectorXd>& y,
    std::size_t num_dummies,
    double lambda2,
    bool normalize,
    bool intercept,
    bool verbose)
    : lambda2_(lambda2),
      d1_(std::sqrt(lambda2)),
      d2_(1.0 / std::sqrt(1.0 + lambda2)),
      l1norm_(0.0),
      num_removals_(0) {

    // Assign base class members directly in body
    algo_type_ = SolverTypeLarsBased::TENET;
    X_ = &X;
    y_ = y;
    normalize_ = normalize;
    intercept_ = intercept;
    verbose_ = verbose;
    currentStep_ = 0;
    is_connected_ = true;
    num_dummies_ = num_dummies;
    count_active_dummies_ = 0;
    p_original_ = X.cols() - num_dummies;
    dummy_start_idx_ = p_original_;

    const std::string solver{solverTypeToString()};

    // 1. Input Validation
    if (X.rows() != y.size()) {
        std::stringstream oss;
        oss << solver << ": X rows (" << X.rows()
            << ") do not match y entries (" << y.size() << ")";
        throw std::invalid_argument(oss.str());
    }

    if (lambda2_ < 0.0) {
        throw std::invalid_argument(
            concatMsg(
                solver, ": Ridge penalty lambda2 must be non-negative (got ", lambda2_, ")"
            )
        );
    }

    // 2. Problem shape/state setup
    std::size_t nrows = X.rows();
    std::size_t pcols = X.cols();

    // Validate dummy configuration
    if (pcols < num_dummies_) {
        std::stringstream oss;
        oss << solver << ": num_dummies (" << num_dummies_
            << ") cannot exceed total columns (" << pcols << ")";
        throw std::invalid_argument(oss.str());
    }
    if (num_dummies_ == 0) {
        std::stringstream oss;
        oss << solver << ": num_dummies must be > 0.";
        throw std::invalid_argument(oss.str());
    }

    // Calculate original predictor count and dummy index start
    p_original_ = pcols - num_dummies_;
    dummy_start_idx_ = p_original_;

    // Effective sample size: n-1 if intercept, otherwise n
    if (nrows == 1 && intercept_) {
        throw std::invalid_argument(
            solver + ": cannot fit intercept with only one sample."
        );
    }
    effective_n_ = nrows - (intercept_ ? 1 : 0);

    // 3. Preprocessing: modifies X in place
    dropped_indices_.clear();
    if (normalize_ || intercept_) { preprocess(); }
    initializeInactives();

    // 4. EN-specific: Ridge allows full-rank active set (Zou & Hastie 2005)
    // if (lambda2 > 0) maxvars = p
    // if (lambda2 == 0) maxvars = min(p, n-1)
    std::size_t pcols_available = pcols - dropped_indices_.size();
    if (lambda2_ > 0.0) {
        max_actives_ = pcols_available;
    } else {
        max_actives_ = std::min(pcols_available, effective_n_);
    }

    // 5. Max steps setup (8 times is the Hastie rule to support variable cycling in Lasso variants)
    maxSteps_ = 8 * max_actives_;

    logInfo(concatMsg(
        solverTypeToString(), " sequence\n",
        "  Samples (n): ", nrows, "\n",
        "  Features (p): ", pcols, " (", p_original_, " real + ", num_dummies_, " dummies)\n",
        "  Ridge penalty λ₂: ", lambda2_, "\n",
        "  Max active set: ", max_actives_, "\n",
        "  Max steps: ", maxSteps_, "\n"
    ));

    // 6. EN-specific: Initialize augmented residuals r = [y; 0_p]
    r_.resize(y_.size() + pcols);
    r_.head(y_.size()) = y_;
    r_.tail(pcols).setZero();
    last_updateR_rank_ = 0;

    // 7. Path/diagnostics setup
    lambda_.reserve(maxSteps_ + 1);
    actions_.reserve(maxSteps_);
    RSS_.reserve(maxSteps_ + 1);
    R2_.reserve(maxSteps_ + 1);
    DoF_.reserve(maxSteps_ + 1);
    actives_.reserve(max_actives_);

    // 8. Path initializations
    betaPath_.resize(pcols, 1);
    betaPath_.col(0).setZero();
    double rss = y_.dot(y_);
    RSS_.emplace_back(rss);
    R2_.emplace_back(0.0);
    DoF_.emplace_back(intercept ? 1 : 0);
    R_.resize(0, 0);
    A_A_ = 0.0;

    // 9. Initialize dummy tracking
    dummies_at_step_.reserve(maxSteps_ + 1);
    dummies_at_step_.push_back(0);

    // 10. Initialize correlations for all inactives
    initializeCorrelations();
}


// ============================================================================
// Core executeStep() Implementation
// ============================================================================

void TENET_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();

    while (
        currentStep_ < maxSteps_ &&
        !inactives_.empty() &&
        actives_.size() < max_actives_ &&
        (count_active_dummies_ < T_stop || !early_stop)
    ) {

        // ========================================================
        // STEP 1: Extract correlations for inactives
        // ========================================================
        auto [Cmax, new_vars] = findTiedMaxCorrelations();
        if (new_vars.empty()) {
            logInfo("No variables to add; exiting.");
            break;
        }

        if (Cmax < 100 * eps_) { // Model is saturated
            logInfo("Max |corr| approx 0; exiting.");
            break;
        }

        // ========================================================
        // STEP 2: Active set update
        // ========================================================
        currentStep_++;
        actions_.emplace_back(updateActiveSet(new_vars));

        if (actives_.empty() || R_.size() == 0) {
            logWarning("Active set or Cholesky is empty; aborting.");
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
            std::size_t pcols = X_->cols();
            max_actives_= (lambda2_ > 0) ?
                            pcols - dropped_indices_.size() :
                            std::min(pcols - dropped_indices_.size(), effective_n_);
            logInfo(concatMsg(
                "Updated max active set to ", max_actives_, " due to collinear drops."
            ));
        }


        // ========================================================
        // STEP 6: Compute step size gamma for joining times
        // ========================================================

        std::vector<bool> drops(actives_.size(), false);
        double gamma_entry = computeStepSizeEN(Cmax, u1, u2);
        double gamma = computeGammaSignChange(gamma_entry, drops, w_A);

        // ========================================================
        // STEP 8: Updates
        // ========================================================

        // Update beta path
        updateBetaPath(w_A, gamma);

        // Update augmented residuals
        r_ -= gamma * u;

        // Update penalty/lambda: penalty[k] = penalty[k-1] - |gamma * A|
        // -> tracks the decreasing L1 constraint as we move along the path
        double new_lambda = (lambda_.empty() ? Cmax : lambda_.back()) - std::abs(gamma * A_A_);
        lambda_.push_back(std::max(0.0, new_lambda));

        // Update diagnostics
        double rss = r_.dot(r_);
        RSS_.push_back(rss);
        R2_.push_back(1.0 - rss / RSS_[0]);

        // Compute L1 norm for active coefficients (vectorized)
        const Eigen::VectorXd& beta_current = betaPath_.col(betaPath_.cols() - 1);
        l1norm_ = 0.0;
        for (std::size_t j : actives_) {
            l1norm_ += std::abs(beta_current[j]);
        }
        l1norm_ /= d2_;

        // ========================================================
        // STEP 9: Handle LASSO drops
        // ========================================================

        processLassoDrops(drops);

        // T-ENET: Record dummy count and DoF after all adds/removes
        updateDummyTracking();
        DoF_.push_back(actives_.size() + (intercept_ ? 1 : 0));

        // ========================================================
        // STEP 10: Update inactive set
        // ========================================================
        updateInactiveSet();

        // ========================================================
        // STEP 10: Recompute correlations for next iteration
        // ========================================================

        updateCorrelationsENET();
    }
}


// ============================================================================
// Internal Helper
// ============================================================================

void TENET_Solver::initializeCorrelations() {
    correlations_.resize(X_->cols());
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < inactives_.size(); ++i) {
        std::size_t j = inactives_[i];
        correlations_[j] = d2_ * X_->col(j).dot(y_);
    }
}


void TENET_Solver::updateCorrelationsENET() {
    // EN-specific correlation formula (Zou & Hastie 2005)
    // Cvec <- (drop(t(residuals[1:n]) %*% x) + d1 * residuals[-(1:n)]) * d2
    // where residuals is augmented (n + m) vector: [y_residuals; coefficient_penalties]

    std::size_t n = X_->rows();
    Eigen::VectorXd r_obs = r_.head(n);      // First n elements (observation residuals)
    Eigen::VectorXd r_coef = r_.tail(r_.size() - n);  // Last m elements (coefficient penalties)

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < inactives_.size(); ++i) {
        std::size_t j = inactives_[i];
        // C_j = (X_j^T * r_obs + d1 * r_coef[j]) * d2
        double corr_obs = X_->col(j).dot(r_obs);
        double corr_pen = d1_ * r_coef[j];
        correlations_[j] = (corr_obs + corr_pen) * d2_;
    }
}


Eigen::MatrixXd TENET_Solver::updateR(const Eigen::Ref<const Eigen::VectorXd>& xnew, double eps) {

    std::size_t m = actives_.size();

    // Elastic Net Gram matrix with (1 + lambda2) scaling
    double d3 = 1 + lambda2_;
    double xtx = (xnew.dot(xnew) + lambda2_) / d3;
    double norm_xnew = std::sqrt(xtx);

    // First variable case: create a (1,1) matrix
    if (R_.size() == 0 || m == 0) {
        Eigen::MatrixXd newR(1, 1);
        newR(0, 0) = norm_xnew;
        last_updateR_rank_ = 1;  // First pivot: rank is 1
        return newR;
    }

    // Compute cross-products: Xtx = Xold^T * xnew with (1 + lambda2) scaling
    Eigen::VectorXd cross_prod(m);
    for (std::size_t k = 0; k < m; ++k) {
        cross_prod[k] = xnew.dot(X_->col(actives_[k])) / d3;
    }

    // Solve R^T * r = Xtx for r (r is length m)
    Eigen::VectorXd r = backsolveT(R_, cross_prod);

    // Compute new diagonal entry
    double rpp_sq = xtx - r.dot(r);
    double rpp{};
    int new_rank = static_cast<int>(R_.rows());  // Current rank backup

    if (rpp_sq < eps * eps) {
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
        newR.block(0, m, m, 1) = r;
    }
    newR(m, m) = rpp;

    // Update global state variable for rank
    last_updateR_rank_ = new_rank;

    return newR;
}


Eigen::VectorXd TENET_Solver::equiangularVectorU1(const Eigen::Ref<const Eigen::VectorXd>& w_A) const {

    Eigen::VectorXd u1 = Eigen::VectorXd::Zero(X_->rows());
    for (std::size_t k = 0; k < actives_.size(); ++k) {
        u1 += w_A[k] * X_->col(actives_[k]) * d2_;
    }

    return u1;
}


Eigen::VectorXd TENET_Solver::equiangularVectorU2(const Eigen::Ref<const Eigen::VectorXd>& w_A) const {

    double factor = d1_ * d2_;
    Eigen::VectorXd u2 = Eigen::VectorXd::Zero(X_->cols());
    for (std::size_t k = 0; k < actives_.size(); ++k) {
        u2[actives_[k]] += w_A[k] * factor;
    }

    return u2;
}


double TENET_Solver::computeStepSizeEN(double Cmax,
                                         const Eigen::Ref<const Eigen::VectorXd>& u1,
                                         const Eigen::Ref<const Eigen::VectorXd>& u2) const {

    // Case 1: Active set saturated: take full step
    if (actives_.size() >= max_actives_) {
        return Cmax / A_A_;
    }

    // Case 2: Compute joining times for inactives
    double gamma_min = Cmax / A_A_;

    for (std::size_t j :inactives_) {

        // Compute rate of correlation change for variable j
        // a_j = d2 * (u1^T X_j + d1 * u2_j)
        double a_j = d2_ * (X_->col(j).dot(u1) + d1_ * u2[j]);
        double C_j = correlations_[j];

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


double TENET_Solver::computeGammaSignChange(double gamhat,
                                              std::vector<bool>& drops,
                                              const Eigen::Ref<const Eigen::VectorXd>& w_A) {

    // Step 1: Get current active coefficients
    Eigen::VectorXd beta_curr(actives_.size());
    for (std::size_t i = 0; i < actives_.size(); ++i) {
        beta_curr[i] = betaPath_(actives_[i], currentStep_ - 1);
    }

    // Step 2: Check for zero-crossings
    // z = -β / w  (step size where β + γw = 0)
    Eigen::VectorXd z = -beta_curr.array() / w_A.array();

    // Step 3: Find minimum postive crossing time
    double zmin = gamhat;
    bool any_crossing = false;

    for (std::size_t i = 0; i < actives_.size(); ++i) {
        if (z[i] > eps_ && z[i] < zmin) {
            zmin = z[i];
            any_crossing = true;
        }
    }

    if (any_crossing) {
        gamhat = zmin;
        for (std::size_t i = 0; i < actives_.size(); ++i) {
            if (std::abs(z[i] - zmin) <= eps_) {
                drops[i] = true;
            }
        }
    }

    return gamhat;
}


void TENET_Solver::processLassoDrops(std::vector<bool>& drops) {

    // Process drops in reverse order to maintain indices
    for (std::size_t i = actives_.size(); i-- > 0;) {
        if (drops[i]) {
            std::size_t dropped_var = actives_[i];

            // Track if dropping a dummy
            if (dropped_var >= dummy_start_idx_ && count_active_dummies_ > 0) {
                count_active_dummies_--;
                logInfo(concatMsg("Dummy variable ", dropped_var,
                    " removed from active set (count: ",
                    count_active_dummies_, ")"));
            }

            // Downdate Cholesky factor
            R_ = downdateR(R_, i);
            // Zero out coefficient
            betaPath_(dropped_var, currentStep_) = 0.0;
            // Record negative action (removal)
            actions_.back().push_back(-static_cast<int>(dropped_var));
            // Remove from active set
            actives_.erase(actives_.begin() + i);
            Sign_.erase(Sign_.begin() + i);
            num_removals_++;
        }
    }

}


double TENET_Solver::getCyclingRatio() const {
    if (num_additions_ == 0) { return 0.0; }

    return static_cast<double>(num_removals_) / static_cast<double>(num_additions_);
}


std::vector<double> TENET_Solver::getCp() const {
    std::vector<double> cp_out;

    // If no RSS or DoF information, return empty
    if (RSS_.empty() || DoF_.empty()) { return cp_out; }

    std::size_t n = X_->rows();

    // Get final RSS for Elastic Net Cp formula
    // Formula: Cp = ((n - m - 1) * RSS) / RSS_final - n + 2 * m
    // where m is DoF (degrees of freedom used)
    double rss_final = RSS_.back();

    // Check if RSS_final is valid
    if (!std::isfinite(rss_final) || rss_final <= 0) {
        cp_out.assign(RSS_.size(), std::numeric_limits<double>::quiet_NaN());
        return cp_out;
    }

    cp_out.reserve(RSS_.size());
    for (std::size_t k = 0; k < RSS_.size(); ++k) {
        double dof = static_cast<double>(DoF_[k]);

        // Check if this step is well-posed: DoF < n and RSS > 0
        if (DoF_[k] < n && RSS_[k] > 0 && std::isfinite(RSS_[k])) {
            // Hastie's Elastic Net formula: ((n - m - 1) * RSS) / RSS_final - n + 2*m
            double cp_val = ((static_cast<double>(n) - dof - 1.0) * RSS_[k]) / rss_final
                          - static_cast<double>(n) + 2.0 * dof;
            cp_out.push_back(cp_val);
        } else {
            cp_out.push_back(std::numeric_limits<double>::quiet_NaN());
        }
    }

    return cp_out;
}



// ============================================================================
// Coefficient Accessors with EN Scaling
// ============================================================================

Eigen::VectorXd TENET_Solver::getBeta(int step) const {
    if (betaPath_.cols() == 0) {
        throw std::runtime_error(
            concatMsg(solverTypeToString(), "::getBeta: No solution path available.")
        );
    }

    std::size_t use_step = (step < 0) ? betaPath_.cols() - 1 : static_cast<std::size_t>(step);
    if (use_step >= static_cast<std::size_t>(betaPath_.cols())) {
        throw std::out_of_range("Step index out of range");
    }

    // EN-specific: Divide by d₂ = 1/√(1+λ₂), i.e., multiply by √(1+λ₂)
    // This recovers naive elastic net coefficients (R enet: beta.pure <- beta.pure / d2)
    Eigen::VectorXd coef = betaPath_.col(use_step) / d2_;

    // Undo normalization if applied
    if (normalize_ && normsx_.size() == coef.size()) {
        coef.array() /= normsx_.array();
    }

    return coef;
}


Eigen::MatrixXd TENET_Solver::getBetaPath() const {
    // EN-specific: Divide by d₂ = 1/√(1+λ₂), i.e., multiply by √(1+λ₂)
    // This matches R enet post-processing: beta.pure <- beta.pure / d2
    Eigen::MatrixXd beta_orig = betaPath_ / d2_;

    // Undo normalization row-wise for all steps (columns)
    if (normalize_ && normsx_.size() == beta_orig.rows()) {
        for (std::size_t j = 0; j < static_cast<std::size_t>(beta_orig.rows()); ++j) {
            beta_orig.row(j) /= normsx_[j];
        }
    }

    return beta_orig;
}


double TENET_Solver::getAlpha() const {

    // Compute mixing parameter α = λ₁/(λ₁ + λ₂)
    if (currentStep_ >= lambda_.size() || currentStep_ == 0) {
        return 1.0;  // Not yet computed or initial step, default to LASSO
    }

    double lambda1 = lambda_[currentStep_];
    double total = lambda1 + lambda2_;

    // Guard against division by zero (both penalties zero = no regularization)
    if (total < eps_) {
        return 1.0;  // Default to LASSO side
    }

    return lambda1 / total;
}


// ============================================================================
// Serialization
// ============================================================================

void TENET_Solver::save(const std::string& filename) const {
    std::ofstream os(filename, std::ios::binary);

    if (!os.is_open()) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                "::save: Cannot open file '", filename,
                "' for writing. Check permissions and path."
            )
        );
    }

    try {
        cereal::PortableBinaryOutputArchive oarchive(os);
        oarchive(*this);

        logInfo(concatMsg(
            "[", solverTypeToString(), "] saved to '", filename, "'\n"
        ));

    } catch (const std::exception& e) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                "::save: Serialization to file '", filename,
                "' failed with error: ", e.what()
            )
        );
    }
}


TENET_Solver TENET_Solver::load(const std::string& filename, Eigen::Map<Eigen::MatrixXd>& X) {

    TENET_Solver tenet;  // No-args constructor

    const std::string solver{TENET_Solver::solverTypeToString(SolverTypeLarsBased::TENET)};

    // 1. Deserialize from file with error handling
    {
        std::ifstream is(filename, std::ios::binary);
        if (!is.is_open()) {
            std::ostringstream oss;
            oss << solver << "::load: Can't open file " << filename;
            throw std::runtime_error(oss.str());
        }

        try {
            cereal::PortableBinaryInputArchive iarchive(is);
            iarchive(tenet);
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << solver << "::load: Deserialization failed - " << e.what();
            throw std::runtime_error(oss.str());
        }
    }

    // 2. Validate loaded state integrity
    if (tenet.betaPath_.rows() == 0) {
        throw std::runtime_error(
            tenet.concatMsg(
                solver, "::load: Loaded state is corrupted (betaPath_ is empty)"
            )
        );
    }

    // 3. Reconnect with error handling
    try {
        tenet.reconnect(X);

        tenet.logInfo(tenet.concatMsg(
            "[", tenet.solverTypeToString(), "] loaded from '", filename, "'\n",
            "  - Algorithm: ", tenet.solverTypeToString(), "\n",
            "  - Loaded at step: ", tenet.currentStep_, "/", tenet.maxSteps_, "\n",
            "  - Active variables: ", tenet.actives_.size(), "\n",
            "  - Matrix dimensions: ", X.rows(), " x ", X.cols(), "\n",
            "  - Status: Ready for continued execution"
        ));
    } catch (const std::exception& e) {
        throw std::runtime_error(
            tenet.concatMsg(
                tenet.solverTypeToString(), "::load: Reconnection failed - ", e.what()
            )
        );
    }

    return tenet;
}


// ============================================================================
// Reconnection (override for augmented dimensions)
// ============================================================================

void TENET_Solver::reconnect(Eigen::Map<Eigen::MatrixXd>& X) {
    // Column count check
    if (X.cols() != betaPath_.rows()) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                "::reconnect: Column dimension mismatch.\n",
                "  Expected: ", betaPath_.rows(), " columns (from saved beta_)\n",
                "  Got: ", X.cols(), " columns\n",
                "  Hint: Ensure X is the same design matrix used during save()"
            )
        );
    }

    // Row count non-zero
    if (X.rows() == 0) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                "::reconnect: Matrix has zero rows"
            )
        );
    }

    // EN-specific: Match residual dimension (augmented size = n + p)
    // r_ has (n+p) elements: [residuals for data; residuals for L2 penalty]
    // X has n rows (original data), so we check: r_.n_elem == n + p
    std::size_t expected_r_size = X.rows() + X.cols();
    if (r_.size() > 0 && static_cast<std::size_t>(r_.size()) != expected_r_size) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                "::reconnect: Augmented residual dimension mismatch.\n",
                "  Expected: ", expected_r_size, " elements (n+p from augmented formulation)\n",
                "  Got: ", r_.size(), " elements\n",
                "  Hint: Design matrix must have same dimensions as during save() (n=",
                X.rows(), ", p=", X.cols(), ")"
            )
        );
    }

    // Consistency with effective_n_ (matches original data rows)
    std::size_t expected_n = effective_n_ + (intercept_ ? 1 : 0);
    if (static_cast<std::size_t>(X.rows()) != expected_n) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                "::reconnect: Row count inconsistent with effective_n_.\n",
                "  Expected: ", expected_n, " rows\n",
                "  Got: ", X.rows(), " rows\n",
                "  (effective_n_=", effective_n_,
                ", intercept=", (intercept_ ? "true" : "false"), ")"
            )
        );
    }

    // Reconnect pointer
    X_ = &X;

    // Validate internal state consistency after reconnection
    if (correlations_.size() != X.cols()) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                "::reconnect: Correlations vector size mismatch.\n",
                "  Expected: ", X.cols(), " elements\n",
                "  Got: ", correlations_.size(), " elements"
            )
        );
    }

    if (y_.size() != X.rows()) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                "::reconnect: Response vector size mismatch.\n",
                "  Expected: ", X.rows(), " elements\n",
                "  Got: ", y_.size(), " elements"
            )
        );
    }

    // Mark as connected
    is_connected_ = true;

    logInfo(concatMsg(
        "Design matrix reconnected successfully\n",
        "  - Dimensions: ", X.rows(), " x ", X.cols(), "\n",
        "  - Augmented residuals: ", r_.size(), " elements (n+p)\n",
        "  - Ready for warm-start from step ", currentStep_
    ));
}
