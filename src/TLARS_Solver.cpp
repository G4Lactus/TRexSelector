#include "TLARS_Solver.hpp"


// ============================================================================
// Constructor Implementation
// ============================================================================

TLARS_Solver::TLARS_Solver(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::VectorXd>& y,
    std::size_t num_dummies,
    bool normalize,
    bool intercept,
    bool verbose,
    SolverType algorithm_type)
    : X_(&X),
      y_(y),
      normalize_(normalize),
      intercept_(intercept),
      verbose_(verbose),
      currentStep_(0),
      is_connected_(true),
      num_dummies_(num_dummies),
      count_active_dummies_(0),
      algo_type_(algorithm_type) {

    const std::string solver{solverTypeToString()};

    // 1. Input Validation
    if (X.rows() != y.size()) {
        std::stringstream oss;
        oss << solver << ": X rows (" << X.rows()
            << ") do not match y entries (" << y.size() << ")";
        throw std::invalid_argument(oss.str());
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

    // 4. Maximum active set size = min(available predictors, effective n))
    max_actives_ = std::min(pcols - dropped_indices_.size(), effective_n_);

    // 5. Max steps setup (8 times is the Hastie rule to support variable
    //   cycling in Lasso variants)
    maxSteps_ = 8 * std::min(max_actives_, effective_n_);

    logInfo(concatMsg(
            solverTypeToString(), " sequence\n",
            "  Samples (n): ", nrows, "\n",
            "  Features (p): ", pcols, " (", p_original_, " real + ",
            num_dummies_, " dummies)\n",
            "  Max steps: ", maxSteps_, "\n"
    ));

    // 6. Initialize state variables
    r_ = y_;
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

    // 10. Initialize correlations and Kahan compensation for all inactives
    correlation_compensation_ = Eigen::VectorXd::Zero(pcols);
    initializeCorrelations();

}


TLARS_Solver::TLARS_Solver(SolverType type)
    : X_(nullptr), is_connected_(false), algo_type_(type) {};


TLARS_Solver::TLARS_Solver()
    : X_(nullptr), is_connected_(false), algo_type_(SolverType::TLARS) {};


void TLARS_Solver::initializeInactives() {
    inactives_.clear();
    inactives_.reserve(X_->cols() - dropped_indices_.size());
    constexpr std::size_t HASH_THRESHOLD{10};

    if (dropped_indices_.size() < HASH_THRESHOLD) {
        // Linear search for small sets
        for (std::size_t j{0}; j < static_cast<std::size_t>(X_->cols()); ++j) {
            if (std::find(dropped_indices_.begin(), dropped_indices_.end(), j)
                == dropped_indices_.end()) {
                    inactives_.push_back(j);
            }
        }
    } else {
        // Hash set with O(1) average lookup
        std::unordered_set<std::size_t> dropped_set(
            dropped_indices_.begin(),
            dropped_indices_.end()
        );

        for (std::size_t j{0}; j < static_cast<std::size_t>(X_->cols()); ++j) {
            if (dropped_set.find(j) == dropped_set.end()) {
                inactives_.push_back(j);
            }
        }
    }
}


void TLARS_Solver::preprocess() {
    std::size_t nrows = X_->rows();
    std::size_t pcols = X_->cols();

    // 1. Intercept (true/false)
    if (intercept_) {
        meansx_.resize(pcols);

        // Setup parallel region
        #pragma omp parallel
        {
            // Phase 1: Compute means
            #pragma omp for schedule(static)
            for (std::size_t j = 0; j < pcols; ++j) {
                meansx_(j) = X_->col(j).mean();
            }

            // Phase 2: Center columns
            #pragma omp for schedule(static)
            for (std::size_t j = 0; j < pcols; ++j) {
                X_->col(j).array() -= meansx_(j);
            }
        }

        // Center y
        mu_y_ = y_.mean();
        y_.array() -= mu_y_;
    }

    // 2. Normalization (L2 scaling)
    if (normalize_) {
        normsx_.resize(pcols);
        std::vector<std::vector<std::size_t>> thread_dropped;

        #pragma omp parallel
        {
            // Ensure Eigen does not spawn nested threads itself
            Eigen::setNbThreads(1);
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();

            #pragma omp single
            { thread_dropped.resize(nthreads); }

            #pragma omp for schedule(static) nowait
            for (std::size_t j = 0; j < pcols; ++j) {
                // Compute L2 norm
                double norm_j = X_->col(j).norm();

                if (norm_j / std::sqrt(static_cast<double>(nrows)) < eps_) {
                    normsx_(j) = eps_ * std::sqrt(static_cast<double>(nrows));

                    // check if already dropped
                    if (std::find(dropped_indices_.begin(),
                                  dropped_indices_.end(), j)
                                  == dropped_indices_.end()) {
                        thread_dropped[tid].push_back(j);
                    }

                } else {
                    normsx_(j) = norm_j;
                }

                // scale column j
                X_->col(j) /= normsx_(j);
            }
        }

        for (const auto& local : thread_dropped) {
            for (std::size_t j : local) {
                dropped_indices_.push_back(j);
                logWarning(concatMsg("Column ", j,
                     " has variance < eps_; dropped"));
            }
        }
    }
}


void TLARS_Solver::restore(Eigen::Map<Eigen::MatrixXd>& X,
                           Eigen::Map<Eigen::VectorXd>& y) const {

    std::size_t pcols = X.cols();

    if (normalize_ && static_cast<std::size_t>(normsx_.size()) != pcols) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                "::restore: normsx_ dimension mismatch"
            )
        );
    }

    if (intercept_ && static_cast<std::size_t>(meansx_.size()) != pcols) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                "::restore: meansx_ dimension mismatch"
            )
        );
    }

    if (y.size() != X.rows()) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                "::restore: y length mismatch with X rows"
            )
        );
    }

    if (normalize_) {
        #pragma omp parallel for schedule(static)
        for (std::size_t j = 0; j < pcols; ++j) {
            X.col(j) *= normsx_(j);
        }
    }

    if (intercept_) {
        #pragma omp parallel for schedule(static)
        for (std::size_t j = 0; j < pcols; ++j) {
            X.col(j).array() += meansx_(j);
        }
        y.array() += mu_y_;
    }
}


// ============================================================================
// Validation Helper
// ============================================================================

void TLARS_Solver::validateConnected() const {
    if (!is_connected_ || X_ == nullptr) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                "Solver is not connected to design matrix. ",
                "Call reconnect(X) after loading from file."
            )
        );
    }

    if (X_->rows() == 0) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                ": Design matrix has zero rows"
            )
        );
    }

    if (betaPath_.rows() > 0 && X_->cols() != betaPath_.rows()) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                ": X columns does not match betaPath rows.\n",
                "X_: ", std::to_string(X_->cols()), "\n",
                "betaPath_.rows(): ", std::to_string(betaPath_.rows())
            )
        );
    }
}


// ============================================================================
// Logger/Status information
// ============================================================================

void TLARS_Solver::logMsg(const std::string& msg) const {
    if (verbose_) {
        std::cout << msg << std::endl;
    }
}


void TLARS_Solver::logWarning(const std::string& msg) const {
    logMsg("[WARNING] " + msg);
}


void TLARS_Solver::logInfo(const std::string& msg) const {
    logMsg("[Info] " + msg);
}


// ============================================================================
// Core Algorithm Implementation: executeStep()
// ============================================================================

void TLARS_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();

    // Add profiling structure
    struct StepTimings {
        double find_max_corr = 0.0;
        double update_active = 0.0;
        double cholesky = 0.0;
        double equiangular = 0.0;
        double step_size = 0.0;
        double beta_update = 0.0;
        double residual_update = 0.0;
        double corr_update = 0.0;
        double inactive_update = 0.0;
        std::size_t num_steps = 0;
    } timings;

    while (
        currentStep_ < maxSteps_ &&
        !inactives_.empty() &&
        actives_.size() < effective_n_ &&
        (count_active_dummies_ < T_stop || !early_stop)
    ) {

        // ========================================================
        // STEP 1: Max correlation among inactives
        // ========================================================
        auto t1 = std::chrono::high_resolution_clock::now();
        auto [Cmax, new_vars] = findTiedMaxCorrelations();
        auto t2 = std::chrono::high_resolution_clock::now();
        timings.find_max_corr += std::chrono::duration<double, std::milli>(t2-t1).count();
        if (new_vars.empty()) {
            logInfo("No variables to add; exiting.");
            break;
        }

        if (Cmax < 100 * eps_) {
            logInfo("Max |corr| approx 0; exiting.");
            break;
        }

        // ========================================================
        // STEP 2: Increment step counter and record lambda
        // ========================================================
        currentStep_++;
        lambda_.push_back(Cmax);

        // ========================================================
        // STEP 3: Active set update
        // ========================================================
        t1 = std::chrono::high_resolution_clock::now();
        actions_.emplace_back(updateActiveSet(new_vars));
        t2 = std::chrono::high_resolution_clock::now();
        timings.update_active += std::chrono::duration<double, std::milli>(t2-t1).count();
        if (actives_.empty() || R_.size() == 0) {
            logWarning("Active set or Cholesky is empty; aborting.");
            break;
        }

        // ========================================================
        // STEP 4: Equi-angular computations
        // ========================================================
        t1 = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd sign_vec = signVector();
        Eigen::VectorXd w_A = equiangularDirection(sign_vec);
        Eigen::VectorXd u = equiangularVector(w_A);
        t2 = std::chrono::high_resolution_clock::now();
        timings.equiangular += std::chrono::duration<double, std::milli>(t2-t1).count();

        t1 = std::chrono::high_resolution_clock::now();
        auto [gamma, a] = computeStepSize(Cmax, u);
        t2 = std::chrono::high_resolution_clock::now();
        timings.step_size += std::chrono::duration<double, std::milli>(t2-t1).count();

        // ========================================================
        // STEP 5: Update beta coefficients
        // ========================================================
        t1 = std::chrono::high_resolution_clock::now();
        updateBetaPath(w_A, gamma);
        t2 = std::chrono::high_resolution_clock::now();
        timings.beta_update += std::chrono::duration<double, std::milli>(t2-t1).count();

        // ========================================================
        // STEP 6: Update residuals
        // ========================================================
        t1 = std::chrono::high_resolution_clock::now();
        r_ -= gamma * u;
        t2 = std::chrono::high_resolution_clock::now();
        timings.residual_update += std::chrono::duration<double, std::milli>(t2-t1).count();

        // ========================================================
        // STEP 7: Update diagnostics
        // ========================================================
        double rss = r_.dot(r_);
        RSS_.push_back(rss);
        R2_.push_back(1.0 - rss / RSS_[0]);

        updateDummyTracking();
        DoF_.push_back(actives_.size() + (intercept_ ? 1 : 0));

        // ========================================================
        // STEP 8: Update inactive set
        // ========================================================
        t1 = std::chrono::high_resolution_clock::now();
        updateInactiveSet();
        t2 = std::chrono::high_resolution_clock::now();
        timings.inactive_update += std::chrono::duration<double, std::milli>(t2-t1).count();

        // ========================================================
        // STEP 9: Recompute correlations for next iteration
        // ========================================================
        t1 = std::chrono::high_resolution_clock::now();
        updateCorrelations(gamma, a);
        t2 = std::chrono::high_resolution_clock::now();
        timings.corr_update += std::chrono::duration<double, std::milli>(t2-t1).count();

        timings.num_steps++;
    }

    // Print profiling summary
    if (verbose_ && timings.num_steps > 0) {
        std::cout << "\n=== TLARS Profiling Summary ===\n";
        std::cout << "Total steps: " << timings.num_steps << "\n";

        // =========================================
        // Time per step computations
        // =========================================
        std::cout << "Time per step (avg):\n" << std::fixed << std::setprecision(1);
        printProfileLine("Find max corr:", timings.find_max_corr / timings.num_steps, " ms");
        printProfileLine("Update active:", timings.update_active / timings.num_steps, " ms");
        printProfileLine("Equiangular dir:", timings.equiangular / timings.num_steps, " ms");
        printProfileLine("Compute stepsize:", timings.step_size / timings.num_steps, " ms");
        printProfileLine("Update beta:", timings.beta_update / timings.num_steps, " ms");
        printProfileLine("Update residual:", timings.residual_update / timings.num_steps, " ms");
        printProfileLine("Update correlations:", timings.corr_update / timings.num_steps, " ms");
        printProfileLine("Update inactives:", timings.inactive_update / timings.num_steps, " ms");

        // =========================================
        // Percentage breakdown section
        // =========================================
        double total = timings.find_max_corr + timings.update_active +
                       timings.equiangular + timings.step_size +
                       timings.beta_update + timings.residual_update +
                       timings.corr_update + timings.inactive_update;

        // =========================================
        // Percentage breakdown section
        // =========================================
        std::cout << "\nPercentage breakdown:\n";
        printProfileLine("Find max corr:", 100.0 * timings.find_max_corr / total, "%");
        printProfileLine("Update active:", 100.0 * timings.update_active / total, "%");
        printProfileLine("Equiangular dir:", 100.0 * timings.equiangular / total, "%");
        printProfileLine("Compute stepsize:", 100.0 * timings.step_size / total, "%");
        printProfileLine("Update beta:", 100.0 * timings.beta_update / total, "%");
        printProfileLine("Update residual:", 100.0 * timings.residual_update / total, "%");
        printProfileLine("Update correlations:", 100.0 * timings.corr_update / total, "%");
        printProfileLine("Update inactives:", 100.0 * timings.inactive_update / total, "%");

        std::cout << "===================================\n\n";
    }
}


// ============================================================================
// Helper Implementations
// ============================================================================

void TLARS_Solver::initializeCorrelations() {
    correlations_.resize(X_->cols());
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < inactives_.size(); ++i) {
        std::size_t j = inactives_[i];
        correlations_(j) = X_->col(j).dot(y_);
    }
}


void TLARS_Solver::updateCorrelations(double gamma, const Eigen::VectorXd& a) {

    std::size_t num_inactives = inactives_.size();

    if ((currentStep_ % kahan_refresh_interval_) == 0) {
        // Full recomputation from residuals (reset numerical errors - classical)
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < num_inactives; ++i) {
            std::size_t j = inactives_[i];
            correlations_(j) = X_->col(j).dot(r_);
            correlation_compensation_(j) = 0.0; // reset compensation
        }
    } else {
        // Note: Old version
        // Incremental update: c_j = c_j - gamma * a_j (recursive definition); O(n p_inactive)
        // #pragma omp parallel for schedule(static)
        // for (std::size_t i = 0; i < num_inactives; ++i) {
        //     std::size_t j = inactives_[i];
        //     correlations_(j) -= gamma * a(i);
        // }

        // better: Kahan Algorithm compensated incremental update; O(p_inactive)
        // https://en.wikipedia.org/wiki/Kahan_summation_algorithm
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < num_inactives; ++i) {
            std::size_t j = inactives_[i];

            // Kahan summation: compensate for accumulated error
            double update = -gamma * a(i) - correlation_compensation_(j);
            double temp = correlations_(j) + update;
            correlation_compensation_(j) = (temp - correlations_(j)) - update;
            correlations_(j) = temp;
        }
    }
}


std::pair<double, std::vector<std::size_t>> TLARS_Solver::findTiedMaxCorrelations() const {
    double Cmax = 0.0;
    std::vector<std::size_t> tied;
    tied.reserve(10);

    // Find maximum
    for (std::size_t j : inactives_) {
        double abs_corr = std::abs(correlations_(j));

        if (abs_corr > Cmax + eps_) {
            // new max, reset ties
            Cmax = abs_corr;
            tied.clear();
            tied.push_back(j);
        } else if (abs_corr >= Cmax - eps_) {
            // add to existing maximum set
            tied.push_back(j);
        }
    }

    return {Cmax, tied};
}


std::vector<int> TLARS_Solver::updateActiveSet (const std::vector<std::size_t>& new_vars) {

        std::vector<int> actions_this_step;
        any_dropped_ = false;

        for (auto j_new : new_vars) {
            Eigen::MatrixXd R_backup = R_;
            int rankR_backup = last_updateR_rank_;
            std::size_t active_size_backup = actives_.size();

            // Attempt to update Cholesky factor with new variable
            Eigen::MatrixXd newR = updateR(X_->col(j_new), eps_);

            // Accept or reject based on numerical rank
            if (last_updateR_rank_ == static_cast<int>(active_size_backup)) {
                // Collinear: did NOT increase rank
                R_ = R_backup;
                last_updateR_rank_ = rankR_backup;

                dropped_indices_.push_back(j_new);
                actions_this_step.push_back(-static_cast<int>(j_new));
                any_dropped_ = true;

                logWarning(concatMsg("Variable ", j_new, " collinear; dropped."));

            } else {
                // Valid addition: rank increased
                R_ = newR;
                actives_.push_back(j_new);
                actions_this_step.push_back(static_cast<int>(j_new));
                Sign_.push_back((correlations_(j_new) >= 0) ? 1 : -1);
                num_additions_++;

                // Remove from inactives incrementally
                auto it = std::find(inactives_.begin(), inactives_.end(), j_new);

                // O(p_inactive) worst case but preserve order
                // Swap-and-pop would be O(1) but changes order and corrupted indices
                if (it != inactives_.end()) {
                     inactives_.erase(it);
                }

                // Track dummy variable entry for early stopping
                if (j_new >= dummy_start_idx_) {
                    count_active_dummies_++;
                }
            }
        }

        return actions_this_step;
}


void TLARS_Solver::updateInactiveSet() {

    if (!any_dropped_) return;

    // Full rebuild only if collinear drops occured
    inactives_.clear();

    // Create hash sets for O(1) membership testing
    std::unordered_set<std::size_t> active_set(actives_.begin(), actives_.end());
    std::unordered_set<std::size_t> dropped_set(dropped_indices_.begin(),
                                                dropped_indices_.end());

    // Setup thread-local buffers
    std::vector<std::vector<std::size_t>> thread_buffers(omp_get_max_threads());

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        thread_buffers[tid].reserve(X_->cols() / omp_get_max_threads());

        #pragma omp for schedule(static)
        for (std::size_t j = 0; j < static_cast<std::size_t>(X_->cols()); ++j) {
            // Only add if not active and not dropped (O(1) lookups)
            if (active_set.find(j) == active_set.end() &&
                dropped_set.find(j) == dropped_set.end()) {
                    thread_buffers[tid].push_back(j);
            }
        }
    }

    // Merge thread buffers
    for (const auto& buf : thread_buffers) {
        inactives_.insert(inactives_.end(), buf.begin(), buf.end());
    }

    // Reset drop flag
    any_dropped_ = false;
}


void TLARS_Solver::updateBetaPath(const Eigen::VectorXd& w_A, double gamma) {

    Eigen::VectorXd beta_new = betaPath_.col(betaPath_.cols() - 1);
    for (std::size_t k = 0; k < actives_.size(); ++k) {
        beta_new(actives_[k]) += gamma * w_A(k);
    }

    betaPath_.conservativeResize(Eigen::NoChange, betaPath_.cols() + 1);
    betaPath_.col(betaPath_.cols() - 1) = beta_new;
}


Eigen::VectorXd TLARS_Solver::signVector() const {
    Eigen::VectorXd sign_vec(Sign_.size());
    for (std::size_t i = 0; i < Sign_.size(); ++i) {
        sign_vec(i) = static_cast<double>(Sign_[i]);
    }
    return sign_vec;
}


Eigen::VectorXd TLARS_Solver::equiangularDirection(const Eigen::VectorXd& sign) {

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


Eigen::VectorXd TLARS_Solver::equiangularVector(const Eigen::VectorXd& w_A) const {

    Eigen::VectorXd u = Eigen::VectorXd::Zero(X_->rows());
    for (std::size_t k = 0; k < static_cast<std::size_t>(actives_.size()); ++k) {
        u += w_A(k) * X_->col(actives_[k]);
    }

    return u;
}


std::pair<double, Eigen::VectorXd> TLARS_Solver::computeStepSize(double Cmax,
                                                                 const Eigen::VectorXd& u) const {

    std::size_t num_inactives = inactives_.size();
    Eigen::VectorXd a(num_inactives);
    double gamma = std::numeric_limits<double>::max();

    #pragma omp parallel
    {
        double local_gamma = std::numeric_limits<double>::max();

        #pragma omp for schedule(static) nowait
        for (std::size_t i = 0; i < num_inactives; ++i) {
            std::size_t j = inactives_[i];
            double a_j = X_->col(j).dot(u);
            a(i) = a_j;
            double c_j = correlations_(j);

            double g1 = (Cmax - c_j) / (A_A_ - a_j);
            double g2 = (Cmax + c_j) / (A_A_ + a_j);

            // local gamma update
            if (g1 > eps_ && std::isfinite(g1) && g1 < local_gamma) {
                local_gamma = g1;
            }
            if (g2 > eps_ && std::isfinite(g2) && g2 < local_gamma) {
                local_gamma = g2;
            }
        }

        #pragma omp critical
        {
            if (local_gamma < gamma) {
                gamma = local_gamma;
            }
        }
    }

    if (gamma == std::numeric_limits<double>::max()) {
        logWarning("No valid gamma found, using fallback.");
        return {Cmax / A_A_, a};
    }

    return {gamma, a};
}


void TLARS_Solver::updateDummyTracking() {
    dummies_at_step_.push_back(count_active_dummies_);
}


std::vector<std::size_t> TLARS_Solver::getActivePredictorIndices() const {
    std::vector<std::size_t> predictor_indices;
    predictor_indices.reserve(actives_.size() - count_active_dummies_);

    for (std::size_t j : actives_) {
        if (j < dummy_start_idx_) {
            predictor_indices.push_back(j);
        }
    }

    return predictor_indices;
}

std::vector<std::size_t> TLARS_Solver::getActiveDummyIndices() const {
    std::vector<std::size_t> dummy_indices;
    dummy_indices.reserve(count_active_dummies_);

    for (std::size_t j : actives_) {
        if (j >= dummy_start_idx_) {
            dummy_indices.push_back(j);
        }
    }

    return dummy_indices;
}


// ============================================================================
// Triangular Solvers
// ============================================================================

Eigen::VectorXd TLARS_Solver::backsolve(const Eigen::Ref<const Eigen::MatrixXd>& R,
                                        const Eigen::Ref<const Eigen::VectorXd>& b) const {
    return R.triangularView<Eigen::Upper>().solve(b);
}


Eigen::VectorXd TLARS_Solver::backsolveT(const Eigen::Ref<const Eigen::MatrixXd>& R,
                                         const Eigen::Ref<const Eigen::VectorXd>& b) const {
    return R.transpose().triangularView<Eigen::Lower>().solve(b);
}


// ============================================================================
// Cholesky Update
// ============================================================================

Eigen::MatrixXd TLARS_Solver::updateR(const Eigen::Ref<const Eigen::VectorXd>& xnew, double eps) {

    std::size_t m = actives_.size();
    double xtx = xnew.dot(xnew);

    // First variable case: create a (1,1) matrix
    if (R_.size() == 0 || m == 0) {
        Eigen::MatrixXd newR(1, 1);
        newR(0, 0) = std::sqrt(xtx);
        last_updateR_rank_ = 1; // first pivot: rank is 1
        return newR;
    }

    // Compute cross-products: Xold^T * xnew
    Eigen::VectorXd cross_prod(m);
    for (std::size_t k = 0; k < m; ++k) {
        cross_prod(k) = xnew.dot(X_->col(actives_[k]));
    }

    // Solve R^T * r = Xtx for r (r is length m)
    Eigen::VectorXd r = backsolveT(R_, cross_prod);

    // Compute new diagonal entry
    double rpp_sq = xtx - r.dot(r);
    double rpp{};
    int new_rank = static_cast<int>(R_.rows()); // current rank backup

    if (rpp_sq < eps * eps) {
        // xnew is numerically in span(X_A)
        rpp = eps;  // Defensive: eps to avoid nan, no rank increment!
        // new_rank remains unchanged
    } else {
        rpp = std::sqrt(rpp_sq);
        ++new_rank; // True rank increases
    }

    // Build new Cholesky matrix (m+1 by m+1)
    Eigen::MatrixXd newR = Eigen::MatrixXd::Zero(R_.rows() + 1, R_.cols() + 1);

    // Copy R_ into top-left block
    if (R_.rows() > 0 && R_.cols() > 0) {
        newR.topLeftCorner(R_.rows(), R_.cols()) = R_;
    }

    // Place r into last column (upper triangle)
    if (r.size() > 0) {
        newR.block(0, R_.cols(), R_.rows(), 1) = r;
    }

    // Put rpp in bottom-right
    newR(R_.rows(), R_.cols()) = rpp;

    // Update global state variable for rank
    last_updateR_rank_ = new_rank;

    return newR;
}


// ============================================================================
// CHOLESKY DOWNDATE (Givens rotations)
// ============================================================================

Eigen::MatrixXd TLARS_Solver::downdateR(const Eigen::MatrixXd& R,
                                        std::size_t k) {

    // Handle singleton active set decrement
    std::size_t p = R.rows();
    if (p == 1) {
        return Eigen::MatrixXd();
    }

    // Main algorithm
    // Step 1: Remove column k (creates p × (p-1) matrix)
    Eigen::MatrixXd Rnew(R.rows(), R.cols() - 1);
    if (k > 0) {
        Rnew.leftCols(k) = R.leftCols(k);
    }
    if (k < static_cast<std::size_t>(R.cols()) - 1) {
        Rnew.rightCols(R.cols() - k - 1) = R.rightCols(R.cols() - k - 1);
    }

    // Step 2: Apply Givens rotations to restore upper triangular form
    //         and zero out subdiagonal entries in column k:
    //         (k+1, k), (k+2, k), ..., (p-1, k)
    for (std::size_t i = k; i < p - 1; ++i) {
        // Check if need to zero element (i+1, i)
        if (i < static_cast<std::size_t>(Rnew.cols())) {
            double a = Rnew(i, i);      // diagonal element
            double b = Rnew(i + 1, i);  // subdiagonal element to zero
            // Compute Givens rotation parameters
            double r = std::hypot(a, b);  // sqrt(a^2 + b^2) (stable)

            if (r > eps_) {
                double c = a / r;   // cosine
                double s = -b / r;  // sine

                // Apply rotation to rows i and i+1 from column i to end
                for (std::size_t j = i; j < static_cast<std::size_t>(Rnew.cols()); ++j) {
                    double temp_i = Rnew(i, j);
                    double temp_ip1 = Rnew(i + 1, j);
                    Rnew(i, j) = c * temp_i - s * temp_ip1;
                    Rnew(i + 1, j) = s * temp_i + c * temp_ip1;
                }
            } else {
                Rnew(i, i) = eps_;
            }
        }
    }

    // Step 3: Remove last row (which should now be ~zero after rotations)
    // This gives us (p-1) × (p-1) upper triangular matrix
    Rnew.conservativeResize(p - 1, Rnew.cols());

    // Safety: Ensure upper-triangular is still square
    if (Rnew.rows() != Rnew.cols()) {
        logWarning(concatMsg(
            "[downdateR ERROR] Non-square matrix: ",
            Rnew.rows(), " x ", Rnew.cols()
        ));

        // Fallback: Factorize Gram from scratch for remaining actives
        std::vector<std::size_t> remaining_active;
        for (std::size_t i = 0; i < actives_.size(); ++i)
            if (i != k) { remaining_active.push_back(actives_[i]); }

        if (remaining_active.empty()) {
            // empty set case
            return Eigen::MatrixXd();
        }

        logWarning(concatMsg(
            "[", solverTypeToString(),
            "] Recalculating Gram and Cholesky factor"
        ));

        Eigen::MatrixXd Xactive(X_->rows(), remaining_active.size());
        for (std::size_t i = 0; i < remaining_active.size(); ++i) {
            Xactive.col(i) = X_->col(remaining_active[i]);
        }
        Eigen::MatrixXd Gram = Xactive.transpose() * Xactive;
        Eigen::LLT<Eigen::MatrixXd> llt(Gram);
        Rnew = llt.matrixU();
    }

    return Rnew;
}


// ============================================================================
// Getters
// ============================================================================

Eigen::VectorXd TLARS_Solver::getBeta(int step) const {
    if (betaPath_.cols() == 0) {
        throw std::runtime_error(
            concatMsg(solverTypeToString(),
                     "::getBeta: No solution path available.")
        );
    }

    std::size_t use_step = (step < 0) ? betaPath_.cols() - 1 :
                                        static_cast<std::size_t>(step);
    if (use_step >= static_cast<std::size_t>(betaPath_.cols())) {
        throw std::out_of_range("Step index out of range");
    }

    Eigen::VectorXd coef = betaPath_.col(use_step);
    if (normalize_ && normsx_.size() == coef.size()) {
        coef.array() /= normsx_.array();
    }

    return coef;
}


double TLARS_Solver::getIntercept(int step) const {
    if (!intercept_) { return 0.0; }

    std::size_t use_step = (step < 0) ? betaPath_.cols() - 1 :
                                        static_cast<std::size_t>(step);
    if (use_step >= static_cast<std::size_t>(betaPath_.cols())) {
        throw std::out_of_range("Step index out of range");
    }
    Eigen::VectorXd beta_orig = getBeta(step);

    return mu_y_ - meansx_.dot(beta_orig);
}


Eigen::MatrixXd TLARS_Solver::getBetaPath() const {
    Eigen::MatrixXd beta_orig = betaPath_;
    // Undo normalization row-wise for all steps (columns)
    if (normalize_ && normsx_.size() == beta_orig.rows()) {
        for (std::size_t j = 0; j < static_cast<std::size_t>(beta_orig.rows()); ++j) {
            beta_orig.row(j) /= normsx_(j);
        }
    }

    return beta_orig;
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
// Cp statistics
// ============================================================================

std::vector<double> TLARS_Solver::getCp() const {

    std::vector<double> cp_out;
    if (RSS_.empty() || DoF_.empty()) { return cp_out; }

    std::size_t n = X_->rows();

    // Find first valid index for full model (DOF < n, RSS > 0)
    double resvar = std::numeric_limits<double>::quiet_NaN();
    for (int i = RSS_.size() - 1; i >= 0; --i) {
        if (DoF_[i] < n && RSS_[i] > 0) {
            resvar = RSS_[i] / static_cast<double>(n - DoF_[i]);
            break;
        }
    }

    if (!std::isfinite(resvar)) {
        cp_out.assign(RSS_.size(), std::numeric_limits<double>::quiet_NaN());
        return cp_out;
    }

    cp_out.reserve(RSS_.size());
    for (std::size_t k = 0; k < RSS_.size(); ++k) {
        double cp_val = (DoF_[k] < n && resvar > 0)
            ? (RSS_[k] / resvar - static_cast<double>(n) + 2.0 * DoF_[k])
            : std::numeric_limits<double>::quiet_NaN();
        cp_out.push_back(cp_val);
    }

    return cp_out;
}


std::vector<double> TLARS_Solver::getCp(double sigma_hat_sq) const {
    std::vector<double> cp_out;
    std::size_t n = X_->rows();

    bool valid_sigma = std::isfinite(sigma_hat_sq) && sigma_hat_sq > 0;

    cp_out.reserve(RSS_.size());
    for (std::size_t k = 0; k < RSS_.size(); ++k) {
        double cp_val = (DoF_[k] < n && valid_sigma)
            ? (RSS_[k] / sigma_hat_sq - static_cast<double>(n) + 2.0 * DoF_[k])
            : std::numeric_limits<double>::quiet_NaN();
        cp_out.push_back(cp_val);
    }

    return cp_out;
}


// ============================================================================
// Serialization Implementation
// ============================================================================

void TLARS_Solver::save(const std::string& filename) const {

    std::ofstream ofs(filename, std::ios::binary);

    if (!ofs.is_open()) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                "::save: Cannot open file '", filename,
                "' for writing. Check permissions and path."
            )
        );
    }

    try {
        cereal::PortableBinaryOutputArchive oarchive(ofs);
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


TLARS_Solver TLARS_Solver::load(const std::string& filename,
                                Eigen::Map<Eigen::MatrixXd>& X) {
    TLARS_Solver tlars;
    const std::string solver{TLARS_Solver::solverTypeToString(
        SolverType::TLARS)};

    // 1. Deserialize from file
    {
        std::ifstream is(filename, std::ios::binary);
        if (!is.is_open()) {
            std::ostringstream oss;
            oss << solver << "::load: Cannot open file " << filename;
            throw std::runtime_error(oss.str());
        }

        try {
            cereal::PortableBinaryInputArchive iarchive(is);
            iarchive(tlars);
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << solver << "::load: Deserialization failed - " << e.what();
            throw std::runtime_error(oss.str());
        }
    }

    // 2. Validate loaded state
    if (tlars.betaPath_.rows() == 0) {
        throw std::runtime_error(
            tlars.concatMsg(
                solver, "::load: Loaded state is corrupted (betaPath_ is empty)"
            )
        );
    }

    // 3. Reconnect
    try {
        tlars.reconnect(X);

        tlars.logInfo(tlars.concatMsg(
            "[", tlars.solverTypeToString(), "] loaded from '",
            filename, "'\n",
            "  - Algorithm: ", tlars.solverTypeToString(), "\n",
            "  - Loaded at step: ", tlars.currentStep_, "/",
            tlars.maxSteps_, "\n",
            "  - Active variables: ", tlars.actives_.size(), "\n",
            "  - Matrix dimensions: ", X.rows(), " x ", X.cols(), "\n",
            "  - Status: Ready for continued execution"
        ));
    } catch (const std::exception& e) {
        throw std::runtime_error(
            tlars.concatMsg(
                tlars.solverTypeToString(), "::load: Reconnection failed - ",
                e.what()
            )
        );
    }

    return tlars;
}


void TLARS_Solver::reconnect(Eigen::Map<Eigen::MatrixXd>& X) {
    // Column count
    if (X.cols() != betaPath_.rows()) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                "::reconnect: Column dimension mismatch.\n",
                "  Expected: ", betaPath_.rows(),
                " columns (from saved beta_)\n",
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

    // Match residual dimension
    if (r_.size() > 0 && X.rows() != r_.size()) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                "::reconnect: Row dimension mismatch.\n",
                "  Expected: ", r_.size(), " rows (from saved residuals)\n",
                "  Got: ", X.rows(), " rows\n",
                "  Hint: Design matrix must have same sample size as during save()"
            )
        );
    }

    // Consistency with effective_n_
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

    // Set both pointer and flag
    X_ = &X;
    is_connected_ = true;

    logInfo(concatMsg(
        "[", solverTypeToString(), "] reconnect successful:\n",
        "  - Design matrix: ", X_->rows(), " x ", X_->cols(), "\n",
        "  - Current step: ", currentStep_, "\n",
        "  - Active variables: ", actives_.size()
    ));
}


// ==========================================================================
// Profiling
// ==========================================================================

void TLARS_Solver::printProfileLine(const std::string& label,
                                    double time_value,
                                    const std::string& unit) const {

    std::cout << "  " << std::left << std::setw(22) << label
              << std::right << std::setw(8) << time_value
              << " " << unit << "\n";
}
