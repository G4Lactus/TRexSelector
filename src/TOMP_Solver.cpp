
#include "TOMP_Solver.hpp"

// ==================================================================================
// Constructor Implementation
// ==================================================================================

TOMP_Solver::TOMP_Solver(
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

    // 1. Input validation
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

    // 5. Max steps setup (8 times is the Hastie rule)
    maxSteps_ = 8 * std::min(max_actives_, effective_n_);

    logInfo(concatMsg(
            solverTypeToString(), " sequence\n",
            "  Samples (n): ", nrows, "\n",
            "  Features (p): ", pcols, " (", p_original_, " real + ", num_dummies_, " dummies)\n",
            "  Max steps: ", maxSteps_, "\n"
    ));

    // 6. Initialize state variables
    r_ = y_;
    last_updateR_rank_ = 0;

    // 7. Path/diagnostics setup
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

    // 9. Initialize dummy tracking
    dummies_at_step_.reserve(maxSteps_ + 1);
    dummies_at_step_.push_back(0);

    // 10. Compute initial correlations
    initializeCorrelations();
}


TOMP_Solver::TOMP_Solver(SolverType type)
    : X_(nullptr), is_connected_(false), algo_type_(type) {};


TOMP_Solver::TOMP_Solver()
    : X_(nullptr), is_connected_(false), algo_type_(SolverType::TOMP) {};


void TOMP_Solver::initializeInactives() {
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


void TOMP_Solver::preprocess() {
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

                // Scale column j
                X_->col(j) /= normsx_(j);
            }
        }

        for (const auto& local : thread_dropped)
            for (std::size_t j : local) {
                dropped_indices_.push_back(j);
                logWarning(concatMsg("Column ", j, " has variance < eps_; dropped"));
            }
    }
}


void TOMP_Solver::restore(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y) const {
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

void TOMP_Solver::validateConnected() const {
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

void TOMP_Solver::logMsg(const std::string& msg) const {
    if (verbose_) {
        std::cout << msg << std::endl;
    }
}


void TOMP_Solver::logWarning(const std::string& msg) const {
    logMsg("[WARNING] " + msg);
}


void TOMP_Solver::logInfo(const std::string& msg) const {
    logMsg("[Info] " + msg);
}


// ============================================================================
// Core Algorithm Implementation: executeStep()
// ============================================================================

void TOMP_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();

    while (
        currentStep_ < maxSteps_ &&
        !inactives_.empty() &&
        actives_.size() < effective_n_ &&
        (count_active_dummies_ < T_stop || !early_stop)
    ) {

        // ========================================================
        // STEP 1: Max correlation among inactives
        // ========================================================

        auto [Cmax, new_vars] = findTiedMaxCorrelations();
        if (new_vars.empty()) {
            logInfo("T-OMP: No variables to add; exiting.");
            break;
        }

        if (Cmax < 100 * eps_) {
            logInfo("Max |corr| approx 0; exiting.");
            break;
        }

        // ========================================================
        // STEP 2: Add variable to support and update active set
        // ========================================================

        currentStep_++;
        actions_.emplace_back(updateActiveSet(new_vars));

        if (actives_.empty() || R_.size() == 0) {
            logWarning("Active set or Cholesky is empty; exiting.");
            break;
        }

        // ========================================================
        // STEP 3: Solution, residual, path, diagnostics updates
        // ========================================================

        updateBetaPath();
        updateResiduals();

        double rss = r_.dot(r_);
        RSS_.emplace_back(rss);
        R2_.emplace_back(1.0 - rss / RSS_[0]);

        // T-OMP: Track dummy count and compute DoF (includes dummies)
        updateDummyTracking();
        DoF_.emplace_back(actives_.size() + (intercept_ ? 1 : 0));

        // ========================================================
        // STEP 4: Update inactive set and compute new correlations
        // ========================================================

        updateInactiveSet();
        computeCorrelations();
    }
}


// ==================================================================================
// Helper Implementations
// ==================================================================================

void TOMP_Solver::initializeCorrelations() {
    correlations_.resize(X_->cols());
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < inactives_.size(); ++i) {
        std::size_t j = inactives_[i];
        correlations_(j) = X_->col(j).dot(y_);
    }
}


void TOMP_Solver::computeCorrelations() {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < inactives_.size(); ++i) {
        std::size_t j = inactives_[i];
        correlations_(j) = X_->col(j).dot(r_);
    }
}


std::pair<double, std::vector<std::size_t>> TOMP_Solver::findTiedMaxCorrelations() const {
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
        } // else case ignored, variables skipped, as abs_corr < Cmax - eps_ is too low
    }

    return {Cmax, tied};
}


std::vector<int> TOMP_Solver::updateActiveSet(const std::vector<std::size_t>& new_vars) {

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
                num_additions_++;

                // Track dummy variable entry for early stopping
                if (j_new >= dummy_start_idx_) {
                    count_active_dummies_++;
                }
            }
        }

        return actions_this_step;
}


void TOMP_Solver::updateDummyTracking() {
    dummies_at_step_.push_back(count_active_dummies_);
}


void TOMP_Solver::updateInactiveSet() {

    inactives_.clear();

    // Create hash sets for O(1) lookup instead of O(k) linear search
    std::unordered_set<std::size_t> active_set(actives_.begin(), actives_.end());
    std::unordered_set<std::size_t> dropped_set(dropped_indices_.begin(), dropped_indices_.end());

    // Setup thread-local buffers
    std::vector<std::vector<std::size_t>> thread_buffers(omp_get_max_threads());

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        thread_buffers[tid].reserve(X_->cols() / omp_get_max_threads());

        #pragma omp for schedule(static)
        for (std::size_t j = 0; j < static_cast<std::size_t>(X_->cols()); ++j) {
            // Only add if not active and not dropped (now O(1) lookups)
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
}


void TOMP_Solver::updateBetaPath() {
    std::size_t k = actives_.size();
    Eigen::VectorXd b(k);
    for (std::size_t i = 0; i < k; ++i) {
        b(i) = X_->col(actives_[i]).dot(y_);
    }

    // Cholesky solve R^T u = b
    Eigen::VectorXd u = backsolveT(R_, b);

    // Cholesky solve R x_A = u
    Eigen::VectorXd beta_A = backsolve(R_, u);

    // Update beta_hat in-place
    Eigen::VectorXd beta_hat = Eigen::VectorXd::Zero(X_->cols());
    #pragma omp parallel for schedule(static) if(k > 100)
    for (std::size_t i = 0; i < k; ++i) {
        beta_hat(actives_[i]) = beta_A(i);
    }

    // Expand path matrix
    if (currentStep_ >= static_cast<std::size_t>(betaPath_.cols())) {
        betaPath_.conservativeResize(X_->cols(), currentStep_ + 1);
    }
    betaPath_.col(currentStep_) = beta_hat;
}


void TOMP_Solver::updateResiduals() {
    // Compute residual: r = y - X_active * beta_active
    // Use manual loop to avoid copying large columns (important for p >> n)
    r_ = y_;
    Eigen::VectorXd beta = betaPath_.col(currentStep_);
    std::size_t k = actives_.size();

    for (std::size_t i = 0; i < k; ++i) {
        std::size_t j = actives_[i];
        r_ -= X_->col(j) * beta(j);
    }
}


// ==================================================================================
// Triangular Solvers
// ==================================================================================

Eigen::VectorXd TOMP_Solver::backsolve(const Eigen::Ref<const Eigen::MatrixXd>& R,
                                       const Eigen::Ref<const Eigen::VectorXd>& b) const {
    return R.triangularView<Eigen::Upper>().solve(b);
}


Eigen::VectorXd TOMP_Solver::backsolveT(const Eigen::Ref<const Eigen::MatrixXd> &R,
                                        const Eigen::Ref<const Eigen::VectorXd> &b) const {
    return R.transpose().triangularView<Eigen::Lower>().solve(b);
}


// ===================================================================================
// CHOLESKY Update
// ===================================================================================

Eigen::MatrixXd TOMP_Solver::updateR(const Eigen::Ref<const Eigen::VectorXd>& xnew, double eps) {

    std::size_t m = actives_.size();
    double xtx = xnew.dot(xnew);

    // First variable case: create a (1,1) matrix
    if (R_.size() == 0 || m == 0) {
        Eigen::MatrixXd newR(1, 1);
        newR(0, 0) = std::sqrt(xtx);
        last_updateR_rank_ = 1;  // First pivot: rank is 1
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
    int new_rank = static_cast<int>(R_.rows());  // Current rank backup

    if (rpp_sq < eps * eps) {
        // xnew is numerically in span(X_A)
        rpp = eps;  // Defensive: eps to avoid nan, no rank increment!
        // new_rank remains unchanged
    } else {
        rpp = std::sqrt(rpp_sq);
        ++new_rank;               // True rank increases
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


// ==================================================================================
// Getters
// ==================================================================================

Eigen::VectorXd TOMP_Solver::getBeta(int step) const {
    if (betaPath_.cols() == 0) {
        throw std::runtime_error(
            concatMsg(solverTypeToString(), "::getBeta: No solution path available.")
        );
    }

    std::size_t use_step = (step < 0) ? betaPath_.cols() - 1 : static_cast<std::size_t>(step);
    if (use_step >= static_cast<std::size_t>(betaPath_.cols())) {
        throw std::out_of_range("Step index out of range");
    }

    Eigen::VectorXd coef = betaPath_.col(use_step);
    if (normalize_ && normsx_.size() == coef.size()) { coef.array() /= normsx_.array(); }

    return coef;
}


double TOMP_Solver::getIntercept(int step) const {
    if (!intercept_) { return 0.0; }

    std::size_t use_step = (step < 0) ? betaPath_.cols() - 1 : static_cast<std::size_t>(step);
    if (use_step >= static_cast<std::size_t>(betaPath_.cols())) {
        throw std::out_of_range("Step index out of range");
    }
    Eigen::VectorXd beta_orig = getBeta(step);

    return mu_y_ - meansx_.dot(beta_orig);
}


Eigen::MatrixXd TOMP_Solver::getBetaPath() const {
    Eigen::MatrixXd beta_orig = betaPath_;
    // Undo normalization row-wise for all steps (columns)
    if (normalize_ && normsx_.size() == beta_orig.rows()) {
        for (std::size_t j = 0; j < static_cast<std::size_t>(beta_orig.rows()); ++j) {
            beta_orig.row(j) /= normsx_(j);
        }
    }

    return beta_orig;
}


std::vector<std::size_t> TOMP_Solver::getActivePredictorIndices() const {
    std::vector<std::size_t> predictor_indices;
    predictor_indices.reserve(actives_.size() - count_active_dummies_);

    for (std::size_t j : actives_) {
        if (j < dummy_start_idx_) {
            predictor_indices.push_back(j);
        }
    }

    return predictor_indices;
}


std::vector<std::size_t> TOMP_Solver::getActiveDummyIndices() const {
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
// Model Selection: Mallows' Cp
// ============================================================================

std::vector<double> TOMP_Solver::getCp() const {

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


std::vector<double> TOMP_Solver::getCp(double sigma_hat_sq) const {
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

// ==================================================================================
// Serialization Implementation
// ==================================================================================

void TOMP_Solver::save(const std::string& filename) const {

    std::ofstream ofs(filename, std::ios::binary);

    if (!ofs.is_open()) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                "::save: Cannot open file '" + filename +
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


TOMP_Solver TOMP_Solver::load(const std::string& filename, Eigen::Map<Eigen::MatrixXd>& X) {

    TOMP_Solver tomp; // Uses no-arg constructor
    const std::string solver{TOMP_Solver::solverTypeToString(SolverType::TOMP)}; // static

    // 1. Deserialize from file with error handling
    {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) {
            std::ostringstream oss;
            oss << solver << "::load: Cannot open file '" << filename;
            throw std::runtime_error(oss.str());
        }

        try {
            cereal::PortableBinaryInputArchive iarchive(ifs);
            iarchive(tomp);
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << solver << "::load: Deserialization failed - " << e.what();
            throw std::runtime_error(oss.str());
        }
    }

    // 2. Validate loaded state integrity
    if (tomp.getBetaPath().rows() != X.cols()) {
        throw std::runtime_error(
            tomp.concatMsg(
                solver, "::load: Loaded solver design matrix columns (",
                tomp.getBetaPath().rows(), ") do not match X columns (",
                X.cols(), "). Ensure X matches the original training matrix."
            )
        );
    }

    // 3. Reconnect with error handling
    try {
        tomp.reconnect(X);

        tomp.logInfo(tomp.concatMsg(
            "[", tomp.solverTypeToString(), "] loaded from '", filename, "'\n",
            "  - Algorithm: ", tomp.solverTypeToString(), "\n",
            "  - Loaded at step: ", tomp.currentStep_, "/",
            "  - Active variables: ", tomp.actives_.size(), "\n",
            "  - Matrix dimensions: ", X.rows(), " x ", X.cols(), "\n",
            "  - Status: Ready for continued execution"
        ));
    } catch (const std::exception& e) {
        throw std::runtime_error(
            tomp.concatMsg(
                tomp.solverTypeToString(), "::load: Reconnection failed - ", e.what()
            )
        );
    }

    return tomp;
}


void TOMP_Solver::reconnect(Eigen::Map<Eigen::MatrixXd>& X) {
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

    // Informative output
    logInfo(concatMsg("Design matrix reconnected successfully\n",
                  "  - Dimensions: ", X.rows(), " x ", X.cols(), "\n",
                  "  - Solver is now ready for executeStep()"));
}
