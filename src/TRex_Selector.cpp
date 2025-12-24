#include "TRex_Selector.hpp"

// ===========================================================
// Constructor
// ===========================================================

TRexSelector::TRexSelector(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::VectorXd>& y,
    double tFDR,
    std::size_t K,
    SolverTypeForTRex solver,
    std::size_t max_num_dummies,
    bool max_T_stop,
    int seed,
    bool verbose
    ) :
    X_(&X),
    y_(y),
    solver_(solver),
    tFDR_(tFDR),
    K_(K),
    max_num_dummies_(max_num_dummies),
    max_T_stop_(max_T_stop),
    seed_(seed),
    verbose_(verbose),
    T_stop_(0),
    num_dummies_(0),
    voting_threshold_(0.0),
    opt_point_(0),
    dummy_multiplier_LL_(0)
{
    // ===========================================================
    // 1. Extract dimensions
    // ===========================================================

    n_ = static_cast<std::size_t>(X_->rows());
    p_ = static_cast<std::size_t>(X_->cols());

    // ===========================================================
    // 2. Validate input parameters
    // ===========================================================

    validateTRexParameters();

    // ===========================================================
    // 3. Normalize X in-place (z-score normalization), center y
    // ===========================================================

    X_means_.resize(p_);
    X_stds_.resize(p_);

    normalizeZscore(*X_, X_means_, X_stds_);
    y_mean_ = centerY();

    // ===========================================================
    // 5. Initialize result containers
    // ===========================================================

    selected_var_.resize(0);
    selected_indices_.clear();
    voting_grid_.resize(0);
    FDP_hat_mat_.resize(0,0);
    Phi_mat_.resize(0,0);
    R_mat_.resize(0,0);
    Phi_prime_.resize(0);
    phi_T_mat_.resize(0,0);
}

// ===========================================================
// Constructor Helpers
// ===========================================================

void TRexSelector::validateTRexParameters() const {

    // Check X dimensions
    if (n_ == 0 || p_ == 0) {
        throw std::invalid_argument("Feature matrix X must have non-zero dimensions.");
    }

    // Check y_dimensions
    if (static_cast<std::size_t>(y_.size()) != n_) {
        throw std::invalid_argument(
            "Response vector y must have length equal to the number of rows in X. "
            "Got y.size() = " + std::to_string(y_.size()) +
            ", X.rows() = " + std::to_string(n_)
        );
    }

    // Check for NaN/Inf in y
    if (!y_.allFinite()) {
        throw std::invalid_argument(
            "Response vector y contains NaN or Inf values."
        );
    }

    // Validate tFDR
    if (tFDR_ < 0.0 || tFDR_ > 1.0) { // All inclusive [0, 1]
        throw std::invalid_argument(
            "Target FDR tFDR must be in the interval [0, 1]."
            " Got tFDR = " + std::to_string(tFDR_)
        );
    }

    // Validate K
    if (K_ < 2) {
        throw std::invalid_argument(
            "Number of random experiments 'K' must be >= 2. Got: "
            + std::to_string(K_)
        );
    }

    // Validate max_num_dummies
    if (max_num_dummies_ < 1) {
        throw std::invalid_argument(
            "Maximum number of dummy variables 'max_num_dummies' must be >= 1. Got: "
            + std::to_string(max_num_dummies_)
        );
    }

}


// ==========================================================
// Normalzation Helpers
// ==========================================================

void TRexSelector::normalizeZscore(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::VectorXd& means,
    Eigen::VectorXd& stds
    ) const
{
    const std::size_t n = X.rows();
    const std::size_t p = X.cols();

    // Compute column means
    means = X.colwise().mean();

    // Center X in-place
    for (std::size_t j = 0; j < p; ++j) {
        X.col(j).array() -= means(j);
    }

    // Compute column standard deviations (unbiased: n-1)
    stds.resize(p);
    for (std::size_t j = 0; j < p; ++j) {
        double var = X.col(j).squaredNorm() / (n - 1);
        stds(j) = std::sqrt(var);

        // Avoid division by zero for constant columns
        if (stds(j) < eps_) {
            stds(j) = 1.0;
            if (verbose_) {
                printProgress(
                    "Warning: Column " + std::to_string(j) +
                    " has zero variance. Setting std = 1.0"
                );
            }
        }
    }

    // Scale X in-place
    for (std::size_t j = 0; j < p; ++j) {
        X.col(j) /= stds(j);
    }
}


double TRexSelector::centerY() {
    const double y_mean = y_.mean();
    y_.array() -= y_mean;
    return y_mean;
}


void TRexSelector::normalizeL2Norm(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::VectorXd& means,
    Eigen::VectorXd& l2norms
) const {

    const std::size_t p = X.cols();

    // Compute column means (for centering)
    means = X.colwise().mean();

    // Center X in-place
    for (std::size_t j = 0; j < p; ++j) {
        X.col(j).array() -= means(j);
    }

    // Compute L2 norms
    l2norms.resize(p);
    for (std::size_t j = 0; j < p; ++j) {
        l2norms(j) = X.col(j).norm();

        // Avoid division by zero
        if (l2norms(j) < eps_) {
            l2norms(j) = 1.0;
            if (verbose_) {
                printProgress(
                    "Warning: Column " + std::to_string(j) +
                    " has zero L2 norm. Setting norm = 1.0"
                );
            }
        }
    }

    // Normalize X in-place
    for (std::size_t j = 0; j < p; ++j) {
        X.col(j) /= l2norms(j);
    }
}


void TRexSelector::convertZscoreToL2(
    Eigen::Map<Eigen::MatrixXd>& X
) const {
    const std::size_t p = X.cols();
    const double scale_factor = std::sqrt(static_cast<double>(n_ - 1));

    // Column-wise in-place scaling
    for (std::size_t j = 0; j < p; ++j) {
        X.col(j).array() /= scale_factor;
    }
}


void TRexSelector::convertL2ToZscore(
    Eigen::Map<Eigen::MatrixXd>& X
) const {
    const std::size_t p = X.cols();
    const double scale_factor = std::sqrt(static_cast<double>(n_ - 1));

    // Column-wise in-place scaling with explicit array semantics
    for (std::size_t j = 0; j < p; ++j) {
        X.col(j).array() *= scale_factor;
    }
}


void TRexSelector::denormalizeX() {

    for (std::size_t j = 0; j < p_; ++j) {
        X_->col(j).array() = X_->col(j).array() * X_stds_(j) + X_means_(j);
    }
}


void TRexSelector::decenterY() {
    y_.array() += y_mean_;
}


// ===========================================================
// Main Selection Method
// ===========================================================

TRexSelector::SelectionResult TRexSelector::select() {

    // 1. Initialize voting grid and set 75% optimization point
    voting_grid_ = createVotingGrid();
    const std::size_t v_len = voting_grid_.size();
    set75PercentOptPoint(v_len);


    // 2. L-Loop: Calibrate num_dummies_ with  T_stop = 1
    Eigen::VectorXd FDP_hat;
    ExperimentResults exp_results;
    runLLoopCalibration(FDP_hat, exp_results);


    // 3. T-Loop: Calibrate T_stop
    runTLoopCalibration(FDP_hat, exp_results);


    // 4. Store final phi_T_mat and Phi_prime
    phi_T_mat_ = exp_results.phi_T_mat;
    Phi_prime_ = computePhiPrime(phi_T_mat_, exp_results.Phi, num_dummies_);


    // 5. Variable Selection
    SelectionResult result = selectedVariables(
        FDP_hat_mat_,
        Phi_mat_,
        voting_grid_,
        T_stop_
    );

    // Store additional information in result
    result.T_stop = T_stop_;
    result.num_dummies = num_dummies_;
    result.FDP_hat_mat = FDP_hat_mat_;
    result.Phi_mat = Phi_mat_;
    result.Phi_prime = Phi_prime_;
    result.voting_grid = voting_grid_;

    // Store results internally
    storeResults(result);

    if (verbose_) {
        printProgress(
            "Selection complete. Selected " +
            std::to_string(getNumSelected()) + " variables."
        );
    }

    // Restoration for X
    if (verbose_) printProgress("Denormalizing X to original scale...");
    denormalizeX();

    return result;
}


// ===========================================================
// FDP Computation
// ===========================================================

Eigen::VectorXd TRexSelector::computeFDPHat(
    const Eigen::VectorXd& voting_grid,
    const Eigen::VectorXd& Phi,
    const Eigen::VectorXd& Phi_prime
    ) const
{
    const std::size_t v_len = voting_grid.size();
    Eigen::VectorXd fdp_hat = Eigen::VectorXd::Zero(v_len);

    for (std::size_t i = 0; i < v_len; ++i) {
        // Count variables with Phi > V[i]
        std::size_t num_sel_var = (Phi.array() > voting_grid(i)).count();

        if (num_sel_var < eps_) {
            fdp_hat(i) = 0.0;
        } else {
            // Sum (1 - Phi_prime) for variables where Phi > V[i]
            double sum_deflated = 0.0;
            for (std::size_t j = 0; j < static_cast<std::size_t>(Phi.size()); ++j) {
                if (Phi(j) > voting_grid(i)) {
                    sum_deflated += (1.0 - Phi_prime(j));
                }
            }
            fdp_hat(i) = std::min(1.0, sum_deflated / num_sel_var);
        }
    }

    return fdp_hat;
}

// ===========================================================
// Phi Prime Computation
// ===========================================================

Eigen::VectorXd TRexSelector::computePhiPrime(
    const Eigen::MatrixXd& phi_T_mat,
    const Eigen::VectorXd& Phi,
    std::size_t num_dummies) const
{
    // Average number of variables selected at each T
    Eigen::VectorXd av_num_var_sel = phi_T_mat.colwise().sum();

    // Filter: keep only rows where Phi > 0.5
    std::vector<std::size_t> phi_geg_fifty_indices;
    for (std::size_t j = 0; j < p_; ++j) {
        if (Phi(j) > 0.5) {
            phi_geg_fifty_indices.push_back(j);
        }
    }

    // Sum phi_T_mat for filtered rows
    Eigen::VectorXd delta_av_num_var_sel = Eigen::VectorXd::Zero(T_stop_);
    for (const auto idx : phi_geg_fifty_indices) {
        delta_av_num_var_sel += phi_T_mat.row(idx).transpose();
    }

    // Compute differences (if T_stop > 1)
    Eigen::MatrixXd phi_T_mat_mod = phi_T_mat;
    if (T_stop_ > 1) {
        for (std::size_t t = T_stop_ - 1; t > 0; --t) {
            delta_av_num_var_sel(t) -= delta_av_num_var_sel(t - 1);
            phi_T_mat_mod.col(t) -= phi_T_mat_mod.col(t - 1);
        }
    }

    // Compute phi_scale vector
    Eigen::VectorXd phi_scale = Eigen::VectorXd::Zero(T_stop_);
    for (std::size_t t = 0; t < T_stop_; ++t) {
        if (delta_av_num_var_sel(t) > eps_) {
            // R index is (t+1)
            double numerator = (p_ - av_num_var_sel(t)) /
                               (static_cast<double>(num_dummies) - static_cast<double>(t) );
            phi_scale(t) = 1 - (numerator / delta_av_num_var_sel(t));
        } else {
            phi_scale(t) = 0.0;
        }
    }

    // Matrix-vector product: Phi_prime = phi_T_mat_mod * phi_scale
    Eigen::VectorXd Phi_prime = phi_T_mat_mod * phi_scale;

    return Phi_prime;
}


// ===========================================================
// Variable Selection
// ===========================================================

TRexSelector::SelectionResult TRexSelector::selectedVariables(
        const Eigen::MatrixXd& FDP_hat_mat,
        const Eigen::MatrixXd& Phi_mat,
        const Eigen::VectorXd& voting_grid,
        std::size_t T_stop
    ) const
{
    // Remove last row in FDP_hat_mat and Phi_matif T_stop_ > 1
    Eigen::MatrixXd FDP_use = FDP_hat_mat;
    Eigen::MatrixXd Phi_use = Phi_mat;
    if (T_stop > 1) {
        FDP_use = FDP_hat_mat.topRows(T_stop - 1);
        Phi_use = Phi_mat.topRows(T_stop - 1);
    }

    const std::size_t n_rows = FDP_use.rows();
    const std::size_t n_cols = FDP_use.cols();

    // Generate R_mat
    Eigen::MatrixXd R_mat = Eigen::MatrixXd::Zero(n_rows, n_cols);
    // TODO: column-major optimization
    for (std::size_t TT = 0; TT < n_rows; ++TT) {
        for (std::size_t VV = 0; VV < n_cols; ++VV) {
            int count = 0;
            for (std::size_t j = 0; j < p_; ++j) {
                if (Phi_use(TT, j) > voting_grid(VV)) {
                    count ++;
                }
            }
            R_mat(TT, VV) = count;
        }
    }

    // Find maximum in R_mat where FDP_use <= tFDR_ (no infinity!)
    double val_max = -1.0;  // Start with invalid value
    std::size_t ind_max_row = 0;
    std::size_t ind_max_col = 0;
    bool found_valid = false;

    // Iterate in COLUMN-MAJOR order to match R
    for (std::size_t j = 0; j < n_cols; ++j) {      // Columns
        for (std::size_t i = 0; i < n_rows; ++i) {  // Rows
            // Check FDP constraint directly (no infinity needed)
            if (FDP_use(i, j) <= tFDR_ && R_mat(i, j) > val_max) {
                val_max = R_mat(i, j);
                ind_max_row = i;
                ind_max_col = j;
                found_valid = true;
            }
        }
    }

    // Valid outcome: no variables selected under FDR control
    if (std::isinf(val_max) || val_max < 0) {
        SelectionResult result;
        result.selected_var = Eigen::VectorXi::Zero(p_);
        result.v_thresh = 1.0;  // maximum threshold - no selection
        result.R_mat = R_mat;
        return result;
    }

    double threshold = voting_grid(ind_max_col);

    // Select variables where Phi > threshold
    Eigen::VectorXi selected_var = Eigen::VectorXi::Zero(p_);
    for (std::size_t j = 0; j < p_; ++j) {
        if (Phi_use(ind_max_row, j) > threshold) {
            selected_var(j) = 1;
        }
    }

    // compile results
    SelectionResult result;
    result.selected_var = selected_var;
    result.v_thresh = threshold;
    result.R_mat = R_mat;

    return result;
}


// ===========================================================
// Helper Methods
// ===========================================================

Eigen::VectorXd TRexSelector::createVotingGrid() const {
    double step = 1.0 / K_;
    std::size_t v_len = static_cast<std::size_t>(std::floor((1.0 - eps_ - 0.5) / step)) + 1;

    Eigen::VectorXd voting_grid(v_len);
    for (std::size_t i = 0; i < v_len; ++i) {
        voting_grid(i) = 0.5 + i * step;
    }

    return voting_grid;
}


void TRexSelector::set75PercentOptPoint(const std::size_t v_len) {

    opt_point_ = 0;
    double target_v = 0.75;
    double min_diff = std::abs(voting_grid_(0) - target_v);

    for (std::size_t i = 1; i < v_len; ++i) {
        double diff = std::abs(voting_grid_(i) - target_v);
        if (diff < min_diff) {
            min_diff = diff;
            opt_point_ = i;
        }
    }

    // Ensure opt_point_ corresponds to voting_grid_ value <= 0.75
    if (opt_point_ == 0 || voting_grid_(opt_point_) >= target_v) {
        for (std::size_t i = 0; i < v_len; ++i) {
            if (voting_grid_(i) < target_v) {
                opt_point_ = i;
            } else {
                break;
            }
        }
    }

}


void TRexSelector::runLLoopCalibration(Eigen::VectorXd& FDP_hat, ExperimentResults& exp_results) {

    // L-loop starts with T_stop_ = 1
    T_stop_ = 1;
    std::size_t dummy_multiplier_LL = 1;    // LL in R
    FDP_hat.resize(0);                      // not initialized

    while (dummy_multiplier_LL <= max_num_dummies_ &&
          (FDP_hat.size() == 0 || FDP_hat(opt_point_) > tFDR_)) {

        // num_dummies = LL * p
        num_dummies_ = dummy_multiplier_LL * p_;

        // Run K experiments with NEW dummies
        exp_results = runRandomExperiments(
            num_dummies_,
            T_stop_,
            /*generate_new_dummies=*/true
        );

        // Compute Phi_prime for FDP estimation
        Eigen::VectorXd Phi_prime = computePhiPrime(
            exp_results.phi_T_mat,
            exp_results.Phi,
            num_dummies_
        );

        // Compute FDP estimate
        FDP_hat = computeFDPHat(voting_grid_, exp_results.Phi, Phi_prime);

        if (verbose_) {
            printProgress("Appended dummies: " + std::to_string(num_dummies_));
        }

        ++dummy_multiplier_LL;
    }

    dummy_multiplier_LL_ = (dummy_multiplier_LL > 1) ? (dummy_multiplier_LL - 1) : 1;
}


void TRexSelector::runTLoopCalibration(Eigen::VectorXd& FDP_hat, ExperimentResults& exp_results) {

    const std::size_t v_len = voting_grid_.size();

    // Initialize matrices for accumulation
    FDP_hat_mat_ = FDP_hat.transpose();      // 1 x v_len
    Phi_mat_ = exp_results.Phi.transpose();  // 1 x p

    // Compute maximum T
    std::size_t max_T = max_T_stop_ ?
        std::min(num_dummies_, static_cast<std::size_t>(std::ceil(n_ / 2.0))) :
        num_dummies_;

    while (FDP_hat(v_len - 1) <= tFDR_ && T_stop_ < max_T) {
        T_stop_++;

        // Run K experiments with SAME dummies
        exp_results = runRandomExperiments(
            num_dummies_,
            T_stop_,
            /*generate_new_dummies=*/false  // REUSE
        );

        // Compute Phi_prime for FDP estimation
        Eigen::VectorXd Phi_prime = computePhiPrime(
            exp_results.phi_T_mat,
            exp_results.Phi,
            num_dummies_
        );

        // Compute FDP estimate
        FDP_hat = computeFDPHat(voting_grid_, exp_results.Phi, Phi_prime);

        // Accumulate matrices
        Phi_mat_.conservativeResize(Phi_mat_.rows() + 1, Eigen::NoChange);
        Phi_mat_.row(Phi_mat_.rows() - 1) = exp_results.Phi;

        FDP_hat_mat_.conservativeResize(FDP_hat_mat_.rows() + 1, Eigen::NoChange);
        FDP_hat_mat_.row(FDP_hat_mat_.rows() - 1) = FDP_hat;

        if (verbose_) {
            printProgress(
                "Included dummies before stopping: " + std::to_string(T_stop_)
            );
        }
    }
}


void TRexSelector::storeResults(const SelectionResult& result) {
    selected_var_ = result.selected_var;
    voting_threshold_ = result.v_thresh;
    R_mat_ = result.R_mat;

    updateSelectedIndices();
}


void TRexSelector::updateSelectedIndices() {
    selected_indices_.clear();
    for (std::size_t j = 0; j < static_cast<std::size_t>(selected_var_.size()); ++j) {
        if (selected_var_(j) == 1) {
            selected_indices_.push_back(j);
        }
    }
}


void TRexSelector::printProgress(const std::string& message) const {
    if (verbose_) {
        std::cout << "[T-Rex Selector] " << message << std::endl;
    }
}


// ===========================================================
// Random Experiments
// ===========================================================

TRexSelector::ExperimentResults TRexSelector::runRandomExperiments(
    std::size_t num_dummies,
    std::size_t T_stop,
    bool generate_new_dummies
) {

    // ===========================================================
    // 1. Generate or reuse dummy matrices
    // ===========================================================

    if (generate_new_dummies) {
        // Clear old dummies and generate K new ones
        stored_dummies_.clear();
        stored_dummies_.reserve(K_);
        for (std::size_t k = 0; k < K_; ++k) {
            Eigen::MatrixXd D = generateDummies(num_dummies, k);
            stored_dummies_.push_back(std::move(D));
        }

        if (verbose_) {
            printProgress(
                "Generated " + std::to_string(K_) +
                " dummy matrices (" + std::to_string(num_dummies) +
                " dummies each)."
            );
        }
    } else {
        // Reuse existing dummies from stored_dummies_
        if (stored_dummies_.size() != K_) {
            throw std::runtime_error(
                "Cannot reuse dummies: stored_dummies_ has size " +
                std::to_string(stored_dummies_.size()) +
                ", but K = " + std::to_string(K_)
            );
        }

        if (verbose_) {
            printProgress(
                "Reusing " + std::to_string(K_) + " dummy matrices."
            );
        }
    }

    // ===========================================================
    // 2. Run K experiments and collect paths
    // ===========================================================

    std::vector<Eigen::MatrixXd> beta_paths;
    beta_paths.reserve(K_);

    for (std::size_t k = 0; k < K_; ++k) {

        // Create augmented matrix [X | D]
        Eigen::MatrixXd XD = createAugmentedMatrix(stored_dummies_[k]);

        // Create Eigen::Map for solver
        Eigen::Map<Eigen::MatrixXd> XD_map(XD.data(), XD.rows(), XD.cols());
        Eigen::Map<Eigen::VectorXd> y_map(y_.data(), y_.size());

        // Convert from z-score (var 1) to L2 norm (unit energy)
        convertZscoreToL2(XD_map);

        // Run T-Selector on augmented data
        Eigen::MatrixXd pathMatrix = runSingleExperiment(XD_map, y_map, T_stop);
        beta_paths.push_back(std::move(pathMatrix));
    }

    // ===========================================================
    // 3. Compute phi_T_mat from betaPaths
    // ===========================================================

    Eigen::MatrixXd phi_T_mat = computePhiTMat(beta_paths, num_dummies, T_stop);

    // ===========================================================
    // 4. Compute Phi vector
    // ===========================================================

    Eigen::VectorXd Phi = phi_T_mat.col(T_stop - 1);

    // Return results
    ExperimentResults results;
    results.phi_T_mat = phi_T_mat;
    results.Phi = Phi;

    return results;
}


Eigen::MatrixXd TRexSelector::generateDummies(
    std::size_t num_dummies,
    std::size_t experiment_id
) const {
    // Create random number generator with experiment-specific seed
    std::size_t rng_seed;
    if (seed_ >= 0) {
        rng_seed = static_cast<std::size_t>(seed_) + experiment_id;
    } else {
        // Use random device for non-deterministic seed
        std::random_device rd;
        rng_seed = rd() + experiment_id;
    }

    std::mt19937_64 rng(rng_seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    // Generate dummy matrix (n × num_dummies) with N(0,1) entries
    Eigen::MatrixXd D(n_, num_dummies);

    for (std::size_t j = 0; j < num_dummies; ++j) {
        for (std::size_t i = 0; i < n_; ++i) {
            D(i, j) = dist(rng);
        }
    }

    // Normalize dummies column-wise (z-score)
    for (std::size_t j = 0; j < num_dummies; ++j) {
        // Center
        double col_mean = D.col(j).mean();
        D.col(j).array() -= col_mean;

        // Scale to unit variance
        double col_var = D.col(j).squaredNorm() / (n_ - 1);
        double col_std = std::sqrt(col_var);

        if (col_std > eps_) {
            D.col(j) /= col_std;
        } else {
            // Degenerate case - keep as zeros
            D.col(j).setZero();
        }
    }

    return D;
}


// Augmented Matrix Creation
// ===========================================================

Eigen::MatrixXd TRexSelector::createAugmentedMatrix(
    const Eigen::MatrixXd& D
) const {
    // Create XD = [X | D]
    // X is (n × p), D is (n × num_dummies)
    // XD is (n × (p + num_dummies))

    const std::size_t num_dummies = D.cols();
    Eigen::MatrixXd XD(n_, p_ + num_dummies);

    // Copy X (leftmost p columns)
    for (std::size_t j = 0; j < p_; ++j) {
        XD.col(j) = X_->col(j);
    }

    // Copy D (rightmost num_dummies columns)
    for (std::size_t j = 0; j < num_dummies; ++j) {
        XD.col(p_ + j) = D.col(j);
    }

    return XD;
}


Eigen::MatrixXd TRexSelector::computePhiTMat(
    const std::vector<Eigen::MatrixXd>& beta_paths,
    std::size_t num_dummies,
    std::size_t T_stop
) const {
    // Initialize phi_T_mat: (p × T_stop)
    Eigen::MatrixXd phi_T_mat = Eigen::MatrixXd::Zero(p_, T_stop);

    // For each experiment
    for (const auto& beta_path_k : beta_paths) {

        // beta_path is (p + num_dummies) x num_steps
        const std::size_t num_steps = static_cast<std::size_t>(beta_path_k.cols());

        if (num_steps == 0) continue;

        // Count dummies included at each step
        Eigen::VectorXi dummy_num_path = Eigen::VectorXi::Zero(num_steps);
        for (std::size_t step = 0; step < num_steps; ++step) {
            int count = 0;
            // Count nonzero dummy coefficients (rows p to p + num_dummies - 1)
            for (std::size_t d = p_; d < p_ + num_dummies; ++d) {
                if (std::abs(beta_path_k(d, step)) > eps_) {
                    count++;
                }
            }
            dummy_num_path(step) = count;
        }

        // For each T from 1 to T_stop
        for (std::size_t t = 1; t <= T_stop; ++t) {
            // Find first step where exactly t dummies are included
            std::size_t step_idx = num_steps - 1; // default to last step
            bool found = false;

            for (std::size_t step = 0; step < num_steps; ++step) {
                if (dummy_num_path(step) == static_cast<int>(t)) {
                    step_idx = step;
                    found = true;
                    break;
                }
            }

            if (!found && verbose_) {
                printProgress(
                    "Warning: Experiment did not reach T= " + std::to_string(t) +
                    " dummies included. Using last step."
                );
            }

            // Mark original features with nonzeros coefficients at this step
            for (std::size_t j = 0; j < p_; ++j) {
                if (std::abs(beta_path_k(j, step_idx)) > eps_) {
                    phi_T_mat(j, t - 1) += (1.0 / static_cast<double>(K_));
                }
            }
        }
    }

    return phi_T_mat;
}


Eigen::MatrixXd TRexSelector::runSingleExperiment(
    Eigen::Map<Eigen::MatrixXd>& XD_map,
    Eigen::Map<Eigen::VectorXd>& y_map,
    std::size_t T_stop
) {
    Eigen::MatrixXd pathMatrix;

    // Dispatch to appropriate solver
    switch (solver_) {
        case SolverTypeForTRex::TLARS: {
            TLARS_Solver tlars_solver(XD_map, y_map, num_dummies_, false, false, false);
            tlars_solver.executeStep(T_stop, /*early_stop=*/true);
            pathMatrix = tlars_solver.getBetaPath();
            break;
        }

        case SolverTypeForTRex::TLASSO: {
            TLASSO_Solver tlasso_solver(XD_map, y_map, num_dummies_, false, false, false);
            tlasso_solver.executeStep(T_stop, /*early_stop=*/true);
            pathMatrix = tlasso_solver.getBetaPath();
            break;
        }

        case SolverTypeForTRex::TSTEPWISE: {
            TSTEPWISE_Solver tstepwise_solver(XD_map, y_map, num_dummies_, false, false, false);
            tstepwise_solver.executeStep(T_stop, /*early_stop=*/true);
            pathMatrix = tstepwise_solver.getBetaPath();
            break;
        }

        case SolverTypeForTRex::TENET: {
            TENET_Solver tenet_solver(XD_map, y_map, num_dummies_, false, false, false);
            tenet_solver.executeStep(T_stop, /*early_stop=*/true);
            pathMatrix = tenet_solver.getBetaPath();
            break;
        }

        case SolverTypeForTRex::TOMP: {
            TOMP_Solver tomp_solver(XD_map, y_map, num_dummies_, false, false, false);
            tomp_solver.executeStep(T_stop, /*early_stop=*/true);
            pathMatrix = tomp_solver.getBetaPath();
            break;
        }

        case SolverTypeForTRex::TGP: {
            TGP_Solver tgp_solver(XD_map, y_map, num_dummies_, false, false, false);
            tgp_solver.executeStep(T_stop, /*early_stop=*/true);
            pathMatrix = tgp_solver.getBetaPath();
            break;
        }

        case SolverTypeForTRex::TACGP: {
            TACGP_Solver tacgp_solver(XD_map, y_map, num_dummies_, false, false, false);
            tacgp_solver.executeStep(T_stop, /*early_stop=*/true);
            pathMatrix = tacgp_solver.getBetaPath();
            break;
        }

        default: {
            throw std::invalid_argument("Unsupported solver type.");
        }
    }

    return pathMatrix;
}


void TRexSelector::printDiagnostics() const {
std::cout << "\n=== T-Rex Diagnostics ===\n";
    std::cout << "n = " << n_ << ", p = " << p_ << "\n";
    std::cout << "K = " << K_ << ", tFDR = " << tFDR_ << "\n";
    std::cout << "T_stop = " << T_stop_ << "\n";
    std::cout << "num_dummies = " << num_dummies_ << "\n";
    std::cout << "dummy_multiplier_LL = " << dummy_multiplier_LL_ << "\n";
    std::cout << "Selected variables: " << getNumSelected() << "\n";
    std::cout << "Voting threshold: " << voting_threshold_ << "\n";
    std::cout << "=======================\n\n";
}
