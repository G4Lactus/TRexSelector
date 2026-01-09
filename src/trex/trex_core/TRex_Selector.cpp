// std includes
#include <cstdlib>
#include <cassert>
#include <filesystem>
#include <sstream>

// T-Rex Selector includes
#include "TRex_Selector.hpp"

// ===================================================================================
// Debug mode only Macro
// ===================================================================================

#ifndef NDEBUG
    #define TREX_DEBUG_PRINT(msg) \
      do { \
        if (verbose_) { \
            std::cout << "T-Rex Selector: [DEBUG]" << (msg) << std::endl; \
        } \
      } while (0)
#else
    #define TREX_DEBUG_PRINT(msg) ((void)0)
#endif

// ===================================================================================


// ===========================================================
// Constructor
// ===========================================================

TRexSelector::TRexSelector(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::VectorXd>& y,
    double tFDR,
    TRexControlParameter trex_control,
    int seed,
    bool verbose
    ) :
    X_(&X),
    y_(y),
    tFDR_(tFDR),
    trex_ctrl_(trex_control),
    seed_(seed),
    verbose_(verbose),
    T_stop_(0),
    num_dummies_(0),
    voting_threshold_(0.0),
    opt_point_(0),
    dummy_multiplier_LL_(0),
    has_serialized_solvers_(false),
    max_cols_memmap_(0)
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
    // 3. Normalize X (center & l2-score normalization), center y
    // ===========================================================

    // Center y
    centerY();

    // L2 normalize X
    centerAndL2NormalizeX();

    // ===========================================================
    // 4. Initialize result containers
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
    if (trex_ctrl_.K < 2) {
        throw std::invalid_argument(
            "Number of random experiments 'K' must be >= 2. Got: "
            + std::to_string(trex_ctrl_.K)
        );
    }

    // Validate max_num_dummies
    if (trex_ctrl_.max_num_dummies == 0) {
        throw std::invalid_argument(
            "Maximum number of dummy variables (max_num_dummies) must be >= 1. Got: "
            + std::to_string(trex_ctrl_.max_num_dummies)
        );
    }
}


// ==========================================================
// Normalzation Helpers
// ==========================================================

void TRexSelector::centerY() {
    y_mean_ = y_.mean();
    y_.array() -= y_mean_;
}


void TRexSelector::decenterY() {
    y_.array() += y_mean_;
}


void TRexSelector::denormalizeX() {
    #pragma omp parallel for schedule(static) if(p_ > 100)
    for (std::size_t j = 0; j < p_; ++j) {
        // Get column reference
        auto col = X_->col(j);

        // Scale back from unit L2 norm
        col.array() *= X_l2norms_(j);

        // Add back the mean
        col.array() += X_means_(j);
    }
}


void TRexSelector::centerAndL2NormalizeX() {
    // Allocate storage for normalization parameters
    X_means_.resize(p_);
    X_l2norms_.resize(p_);

    // Fused ONE pass operation: mean + center + norm + normalize
    // Parallel across independent columns
    #pragma omp parallel for schedule(static) if(p_ > 100)
    for (std::size_t j = 0; j < p_; ++j) {
        // Get column reference
        auto col = X_->col(j);

        // Compute mean
        const double mean_j = col.mean();
        X_means_[j] = mean_j;

        // Center column
        col.array() -= mean_j;

        // Compute L2 norm
        double l2norm_j = col.norm();

        // Handle zero norm (avoid division by zero)
        if (l2norm_j < eps_) {
            l2norm_j = 1.0;
            if (verbose_) {
                // Note: Only thread 0 should print to avoid race
                #pragma omp critical
                {
                    printProgress("Warning: Column " + std::to_string(j) +
                                " has zero L2 norm. Setting norm = 1.0");
                }
            }
        }
        X_l2norms_[j] = l2norm_j;

        // Normalize column
        col.array() /= l2norm_j;
    }
}


void TRexSelector::centerAndL2NormalizeMatrix(
    Eigen::Map<Eigen::MatrixXd>& M
) const {
    // Extract dimensions
    const std::size_t p_local = M.cols();

    // Fused ONE pass operation: mean + center + norm + normalize
    // Parallel across independent columns
    #pragma omp parallel for schedule(static) if(p_local > 100)
    for (std::size_t j = 0; j < p_local; ++j) {
        // Get column reference
        auto col = M.col(j);

        // Compute mean
        const double mean_j = col.mean();

        // Center column
        col.array() -= mean_j;

        // Compute L2 norm
        double l2norm_j = col.norm();

        // Handle zero norm (avoid division by zero)
        if (l2norm_j < eps_) {
            l2norm_j = 1.0;
            if (verbose_) {
                // Note: Only thread 0 should print to avoid race
                #pragma omp critical
                {
                    printProgress("Warning: Column " + std::to_string(j) +
                                " has zero L2 norm. Setting norm = 1.0");
                }
            }
        }

        // Normalize column
        col.array() /= l2norm_j;
    }
}



// ===========================================================
// Member Methods for Solver Serialization
// ===========================================================

std::string TRexSelector::createTempDirectory() {

    namespace fs = std::filesystem;

    // robust randomization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> uidist(10000, 99999);

    // create temp directory in system temp location
    fs::path temp_base = fs::temp_directory_path();
    fs::path temp_dir = temp_base / ("trex_solvers_" + std::to_string(uidist(gen)));

    // Create directory
    fs::create_directories(temp_dir);

    if (verbose_) {
        printProgress("Created temp directory: " + temp_dir.string());
    }

    return temp_dir.string();
}


void TRexSelector::cleanupTempDirectory() {

    namespace fs = std::filesystem;

    if (!temp_dir_.empty() && fs::exists(temp_dir_)) {
        try {
            // Remove directory and all contents
            fs::remove_all(temp_dir_);

            if (verbose_) {
                printProgress("Cleaned up temp directory: " + temp_dir_);
            }
        } catch (const std::exception& e) {
            if (verbose_) {
                printProgress("Warning: Failed to cleanup temp directory: " +
                               std::string(e.what()));
            }
        }
    }

    temp_dir_.clear();
    solver_files_.clear();
    has_serialized_solvers_ = false;
}



// ===========================================================
// Main Selection Method
// ===========================================================

TRexSelector::SelectionResult TRexSelector::select() {

    // If Memory-mapped data initialize
    if (trex_ctrl_.use_memory_mapping) {
        initializeMemoryMappedMatrices();
        writeXToMemoryMappedMatrices();
    }

    // TRex Selection Procedure
    // ===============================
    // 1. Initialize voting grid and set 75% optimization point
    voting_grid_ = createVotingGrid();
    const std::size_t v_len = voting_grid_.size();
    set75PercentOptPoint(v_len);

    // 2. L-Loop: Calibrate num_dummies_ with  T_stop = 1
    if (verbose_) {
        printProgress("L-loop: Calibrating number of dummies...");
    }
    Eigen::VectorXd FDP_hat;
    ExperimentResults exp_results;
    applyLLoopStrategy(FDP_hat, exp_results);
    if (verbose_) {
        printProgress("L-loop: converged with " +
                      std::to_string(num_dummies_) + " dummies.\n");
    }


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
    // ===============================

    // Store results & infors in result
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

    // Clean-up memory-mapped files
    if (trex_ctrl_.use_memory_mapping) {
        cleanupMemoryMappedMatrices();
    }

    // Restoration for X
    if (verbose_) printProgress("Denormalizing X to original scale.\n");
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
    const std::size_t p = static_cast<std::size_t>(Phi.size());

    Eigen::VectorXd fdp_hat = Eigen::VectorXd::Zero(v_len);

    // Precompute once (1 - Phi_prime)
    const Eigen::VectorXd one_minus_phi_prime =
        Eigen::VectorXd::Ones(p) - Phi_prime;

    for (std::size_t i = 0; i < v_len; ++i) {
        const double threshold = voting_grid(i);

        // Vectorized comparison (Eigen SIMD-accelerated)
        const Eigen::Array<bool, Eigen::Dynamic, 1> selected_mask =
            (Phi.array() > threshold);

        const std::size_t num_sel_var = selected_mask.count();

        if (num_sel_var == 0) {
            fdp_hat(i) = 0.0;
            continue;
        }

        // R equivalent: sum((1 - Phi_prime)[Phi > V[i]])
        // Eigen's select() is branchless and vectorized
        const double sum_deflated =
            selected_mask.select(one_minus_phi_prime.array(), 0.0).sum();

        fdp_hat(i) = std::min(1.0, sum_deflated / static_cast<double>(num_sel_var));
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
    // Use actual matrix dimensions
    std::size_t current_T = phi_T_mat.cols();

    // Average number of variables selected at each T
    Eigen::VectorXd av_num_var_sel = phi_T_mat.colwise().sum(); // Size: current_T

    // Filter: keep only rows where Phi > 0.5
    std::vector<std::size_t> phi_geg_fifty_indices;
    phi_geg_fifty_indices.reserve(p_); // Optimization
    for (std::size_t j = 0; j < p_; ++j) {
        if (Phi(j) > 0.5) {
            phi_geg_fifty_indices.push_back(j);
        }
    }

    // Sum phi_T_mat for filtered rows
    // Use current_T instead of T_stop_
    Eigen::VectorXd delta_av_num_var_sel = Eigen::VectorXd::Zero(current_T);

    for (const auto idx : phi_geg_fifty_indices) {
#ifndef NDEBUG
        // Verify dimension match (debug only)
        assert(static_cast<std::size_t>(phi_T_mat.row(idx).size()) == current_T);
#endif

        delta_av_num_var_sel += phi_T_mat.row(idx).transpose();
    }

    // Compute differences (if current_T > 1)
    Eigen::MatrixXd phi_T_mat_mod = phi_T_mat;
    if (current_T > 1) {
        for (std::size_t t = current_T - 1; t > 0; --t) {
            delta_av_num_var_sel(t) -= delta_av_num_var_sel(t - 1);
            phi_T_mat_mod.col(t) -= phi_T_mat_mod.col(t - 1);
        }
    }

    // Compute phi_scale vector
    Eigen::VectorXd phi_scale = Eigen::VectorXd::Zero(current_T);
    for (std::size_t t = 0; t < current_T; ++t) {
        if (delta_av_num_var_sel(t) > eps_) {
            // Note: reference R index is (t+1)
            double num_remaining_orig = static_cast<double>(p_) - av_num_var_sel(t);
            double num_remaining_dummies = static_cast<double>(num_dummies) -
                                            static_cast<double>(t);
            double ratio_remaining = num_remaining_orig / num_remaining_dummies;

            phi_scale(t) = 1.0 - (ratio_remaining / delta_av_num_var_sel(t));
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
    for (std::size_t VV = 0; VV < n_cols; ++VV) {
        const double threshold = voting_grid(VV);

        for (std::size_t TT = 0; TT < n_rows; ++TT) {
            // Vectorized count
            R_mat(TT, VV) = (Phi_use.row(TT).array() > threshold).count();
        }
    }

    // Find maximum in R_mat where FDP_use <= tFDR_ (no infinity!)
    double val_max = -1.0;  // Start with invalid value
    std::size_t ind_max_row = 0;
    std::size_t ind_max_col = 0;

    // Iterate in COLUMN-MAJOR order
    for (std::size_t j = 0; j < n_cols; ++j) {      // Columns
        for (std::size_t i = 0; i < n_rows; ++i) {  // Rows
            // Check FDP constraint directly (no infinity needed)
            if (FDP_use(i, j) <= tFDR_ && R_mat(i, j) >= val_max) { // NOTE: changed == to >=
                val_max = R_mat(i, j);
                ind_max_row = i;
                ind_max_col = j;
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
    const double v_start = 0.5;
    const double v_end = 1.0 - eps_;
    const double v_step = 1.0 / trex_ctrl_.K;

    // Number of elements in sequence [0.5, 1-eps] as [start, start+step, ..., end]
    const std::size_t v_len = static_cast<std::size_t>(
        std::floor((v_end - v_start) / v_step)) + 1;

    Eigen::VectorXd voting_grid(v_len);
    for (std::size_t i = 0; i < v_len; ++i) {
        voting_grid(i) = v_start + i * v_step;
    }

    return voting_grid;
}


void TRexSelector::set75PercentOptPoint(const std::size_t v_len) {
    opt_point_ = 0;
    double target_v = 0.75;

    // Ensure opt_point_ corresponds to voting_grid_ value <= 0.75
    for (std::size_t i = 0; i < v_len; ++i) {
        if (voting_grid_(i) < target_v) { opt_point_ = i; }
        else { break; } // grid is sorted -> break allowed
    }

}

// ===========================================================
// Loop Calibration Methods
// ===========================================================

// Variant without dummy matrix size calibration

void TRexSelector::runLLoopCalibration_SKIP(
    Eigen::VectorXd& FDP_hat,
    ExperimentResults& exp_results
) {
    // Set fixed number of dummies
    num_dummies_ = trex_ctrl_.max_num_dummies * p_;

    // Initialize Tstop
    T_stop_ = 1;
    FDP_hat.resize(0);     // not initialized

    if (verbose_) {
        printProgress(
            "L-Loop: SKIP strategy. Using fixed number of dummies = " +
            std::to_string(num_dummies_)
        );
    }

    // Run K random experiments in L-loop
    if (!trex_ctrl_.use_memory_mapping) { // In-memory branch
        // Run single experiment with fixed dummies for initial FDP estimates
        exp_results = runRandomExperiments(
            num_dummies_,
            T_stop_,
            /*generate_new_dummies=*/true,
            /*use_warm_start=*/false,
            /*seed_offset=*/0
        );

    } else { // Memory-mapped branch
        // Write dummies to memory-mapped matrices
        for (std::size_t k = 0; k < trex_ctrl_.K; ++k) {
            auto XD_map = XD_memmaps_[k]->getMap();

            // Generate and write dummies to columns [p_, p_ + num_dummies_)
            XD_map.block(0, p_, n_, num_dummies_) = generateDummies(num_dummies_, k);
        }

        // Run experiments with memory-mapped matrices
        exp_results = runRandomExperiments_MemMap(
            num_dummies_,
            T_stop_,
            /*use_warm_start=*/false
        );
    }

    // Compute Phi_prime for FDP estimation
    Eigen::VectorXd Phi_prime = computePhiPrime(
        exp_results.phi_T_mat,
        exp_results.Phi,
        num_dummies_
    );

    // Compute FDP estimate
    FDP_hat = computeFDPHat(voting_grid_, exp_results.Phi, Phi_prime);

    if (verbose_) {
        printProgress("Using fixed number of dummies: " + std::to_string(num_dummies_));
    }

}


// Variant with new dummies each iteration

void TRexSelector::runLLoopCalibration(Eigen::VectorXd& FDP_hat, ExperimentResults& exp_results) {

    // L-loop starts with T_stop_ = 1
    T_stop_ = 1;
    dummy_multiplier_LL_ = 1;    // LL in R
    FDP_hat.resize(0);           // not initialized

    while (dummy_multiplier_LL_ <= trex_ctrl_.max_num_dummies &&
          (FDP_hat.size() == 0 || FDP_hat(opt_point_) > tFDR_)) {

        // num_dummies = LL * p
        num_dummies_ = dummy_multiplier_LL_ * p_;

        if (!trex_ctrl_.use_memory_mapping) { // In-memory branch
            // Run K experiments with NEW dummies
            exp_results = runRandomExperiments(
                num_dummies_,
                T_stop_,
                /*generate_new_dummies=*/true,   // New dummies each time
                /*use_warm_start=*/false,        // Cold start required
                /*seed_offset=*/dummy_multiplier_LL_ * 10000
            );

        } else { // Memory-mapped branch
            // Overwrite dummies in memory-mapped matrices
            for (std::size_t k = 0; k < trex_ctrl_.K; ++k) {
                auto XD_map = XD_memmaps_[k]->getMap();

                // Generate NEW dummies for this iteration
                XD_map.block(0, p_, n_, num_dummies_) = generateDummies(num_dummies_,
                                                         k + dummy_multiplier_LL_ * 10000);

            }

            // Run K experiments with memory-mapped matrices
            exp_results = runRandomExperiments_MemMap(
                num_dummies_,
                T_stop_,
                /*use_warm_start=*/false
            );
        }

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

        ++dummy_multiplier_LL_;
    }

    dummy_multiplier_LL_ = (dummy_multiplier_LL_ > 1) ? (dummy_multiplier_LL_ - 1) : 1;
}


// Variant with dummy matrix augmentation

void TRexSelector::runLLoopCalibration_Adaptive(
    Eigen::VectorXd& FDP_hat,
    ExperimentResults& exp_results
) {

    // L-loop starts with T_stop_ = 1
    T_stop_ = 1;
    dummy_multiplier_LL_ = 1;    // LL in R
    FDP_hat.resize(0);           // not initialized

    // Intial setup
    if (!trex_ctrl_.use_memory_mapping) { // In-memory branch
        // Initialize dummy storage
        stored_dummies_.clear();
        stored_dummies_.reserve(trex_ctrl_.K);
        num_dummies_ = p_;  // Start with p dummies

        // Generate initial p dummies for each experiment
        for (std::size_t k = 0; k < trex_ctrl_.K; ++k) {
            stored_dummies_.emplace_back(generateDummies(p_, k));
        }

    } else { // Memory-mapped branch
        // Write initial p dummies into memory-mapped matrices
        num_dummies_ = p_;
        for (std::size_t k = 0; k < trex_ctrl_.K; ++k) {
            auto XD_map = XD_memmaps_[k]->getMap();
            XD_map.block(0, p_, n_, p_) = generateDummies(p_, k);
        }

    }

    // Adaptive L-loop
    while (dummy_multiplier_LL_ <= trex_ctrl_.max_num_dummies &&
          (FDP_hat.size() == 0 || FDP_hat(opt_point_) > tFDR_)) {

        // Only augment AFTER first iteration
        if (dummy_multiplier_LL_ > 1) {

            if (!trex_ctrl_.use_memory_mapping) { // In-memory branch
                for (std::size_t k = 0; k < trex_ctrl_.K; ++k) {
                    // Horizontally concatenate: [D_old | D_new]
                    std::size_t old_cols = stored_dummies_[k].cols();
                    stored_dummies_[k].conservativeResize(
                        Eigen::NoChange,
                        old_cols + p_
                    );

                    // Generate p new dummies with different seed offset
                    stored_dummies_[k].rightCols(p_) = generateDummies(
                        p_,
                        k + dummy_multiplier_LL_ * 10000
                    );
                }

            } else { // Memory-mapped branch
                // Append to memory-mapped matrices
                std::size_t start_col = p_ + (dummy_multiplier_LL_ - 1) * p_;

                for (std::size_t k = 0; k < trex_ctrl_.K; ++k) {
                    auto XD_map = XD_memmaps_[k]->getMap();

                    // Generate p new dummies with different seed offset
                    XD_map.block(0, start_col, n_, p_) = generateDummies(
                        p_,
                        k + dummy_multiplier_LL_ * 10000
                    );
                }
            }
        }

        // Update num_dummies to match actual matrix size
        num_dummies_ = dummy_multiplier_LL_ * p_;

        // Run K experiments with EXPANDED dummies
        if (!trex_ctrl_.use_memory_mapping) { // In-memory branch
            exp_results = runRandomExperiments(
                num_dummies_,
                T_stop_,
                /*generate_new_dummies=*/false,  // Reuse stored dummies
                /*use_warm_start=*/false,        // Cold start required
                /*seed_offset=*/0
            );

        } else { // Memory-mapped branch
            exp_results = runRandomExperiments_MemMap(
                num_dummies_,
                T_stop_,
                /*use_warm_start=*/false
            );
        }


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

        ++dummy_multiplier_LL_;
    }

    dummy_multiplier_LL_ = (dummy_multiplier_LL_ > 1) ? (dummy_multiplier_LL_ - 1) : 1;
}


// L-loop strategy dispatcher

void TRexSelector::applyLLoopStrategy(
    Eigen::VectorXd& FDP_hat,
    ExperimentResults& exp_results
) {
    switch (trex_ctrl_.lloop_strategy) {
        case LLoopStrategy::SKIP:
            runLLoopCalibration_SKIP(FDP_hat, exp_results);
            break;

        case LLoopStrategy::STANDARD:
            runLLoopCalibration(FDP_hat, exp_results);
            break;

        case LLoopStrategy::ADAPTIVE:
            runLLoopCalibration_Adaptive(FDP_hat, exp_results);
            break;

        default:
            throw std::invalid_argument("Invalid L-loop strategy.");
    }
}




// ===========================================================
// T-Loop Calibration
// ===========================================================

void TRexSelector::runTLoopCalibration(
    Eigen::VectorXd& FDP_hat,
    ExperimentResults& exp_results
) {
    if (verbose_) {
        printProgress("T-loop: Calibrating stopping threshold (T_stop)...");
    }

    const std::size_t v_len = voting_grid_.size();

    // Compute maximum T
    std::size_t max_T = trex_ctrl_.max_T_stop ?
        std::min(num_dummies_, static_cast<std::size_t>(std::ceil(n_ / 2.0))) :
        num_dummies_;

    // Initialize matrices for accumulation (with capacity doubling)
    const std::size_t initial_capacity = 20;
    std::size_t capacity = std::min(initial_capacity, max_T);

    Phi_mat_ = Eigen::MatrixXd::Zero(capacity, p_);
    FDP_hat_mat_ = Eigen::MatrixXd::Zero(capacity, v_len);

    // Track selection size for stagnation detection
    std::size_t prev_num_selected = 0;
    std::size_t stagnant_iterations = 0;

    // Check FDP at highest v
    std::size_t row_idx = 0;
    while (FDP_hat(v_len - 1) <= tFDR_ && T_stop_ < max_T) {

        // Run K experiments with SAME dummies (warm start)
        if (!trex_ctrl_.use_memory_mapping) { // In-memory branch
            exp_results = runRandomExperiments(
                num_dummies_,
                T_stop_,
                /*generate_new_dummies=*/false,
                /*use_warm_start=*/true,
                /*seed_offset=*/0
            );

        } else { // Memory-mapped branch
            exp_results = runRandomExperiments_MemMap(
                num_dummies_,
                T_stop_,
                /*use_warm_start=*/true
            );
        }

        // Compute Phi_prime for FDP estimation
        Eigen::VectorXd Phi_prime = computePhiPrime(
            exp_results.phi_T_mat,
            exp_results.Phi,
            num_dummies_
        );

        // Compute FDP estimate
        FDP_hat = computeFDPHat(voting_grid_, exp_results.Phi, Phi_prime);

        // Find optimal (v*, T) at current T
        std::size_t v_star_idx = 0;
        for (std::size_t i = 0; i < v_len; ++i) {
            if (FDP_hat(i) <= tFDR_) {
                v_star_idx = i;
            }
        }
        double v_star = (v_star_idx < v_len) ? voting_grid_(v_star_idx) : 0.0;

        // Count variables selected at v*
        std::size_t num_selected = 0;
        if (v_star > eps_) {
            num_selected = (exp_results.Phi.array() > v_star).count();
        }

        // Stagnation check
        if (trex_ctrl_.tloop_stagnation_stop && T_stop_ > 1) {
            if (num_selected == prev_num_selected) {
                stagnant_iterations++;

                if (stagnant_iterations >= trex_ctrl_.max_stagnant_steps) {
                    if (verbose_) {
                        printProgress("T-loop EARLY STOP: Stagnation detected");
                        printProgress("  S(v*) = " + std::to_string(v_star) +
                                      ", T = " + std::to_string(T_stop_) +
                                      " → " + std::to_string(num_selected));
                    }
                    break;  // early stop due to stagnation
                }
            } else {
                stagnant_iterations = 0;
            }
        }
        prev_num_selected = num_selected;


        // Enhanced diagnostic output
        if (verbose_) {
            const double min_FDP = FDP_hat.minCoeff();
            const double max_FDP = FDP_hat.maxCoeff();

            std::ostringstream oss;
            oss << "T = " << std::setw(3) << T_stop_
                << " | v* = " << v_star
                << ", S = " << num_selected
                << ", FDP ∈ [" << min_FDP << ", " << max_FDP << "]";
            if (stagnant_iterations > 0) {
                oss << " (stagnant: " << stagnant_iterations << ")";
            }
            printProgress(oss.str());
        }

        // Dynamic capacity growth
        if (row_idx >= capacity) {
            std::size_t new_capacity = std::min(capacity * 2, max_T);
            Phi_mat_.conservativeResize(new_capacity, Eigen::NoChange);
            FDP_hat_mat_.conservativeResize(new_capacity, Eigen::NoChange);
            Phi_mat_.bottomRows(new_capacity - capacity).setZero();
            FDP_hat_mat_.bottomRows(new_capacity - capacity).setZero();
            capacity = new_capacity;
        }

        // Store results
        Phi_mat_.row(row_idx) = exp_results.Phi;
        FDP_hat_mat_.row(row_idx) = FDP_hat;
        row_idx++;
        T_stop_++;
    }

    // Revert look-ahead increment for T_stop_
    if (T_stop_ > 1) { T_stop_--; }

    // Trim to to actual size
    Phi_mat_.conservativeResize(row_idx, Eigen::NoChange);
    FDP_hat_mat_.conservativeResize(row_idx, Eigen::NoChange);

    // Enhanced final summary
    if (verbose_) {
        std::string stop_reason;
        if (stagnant_iterations >= trex_ctrl_.max_stagnant_steps) {
            stop_reason = " (early stop: stagnation)";
        } else if (T_stop_ >= max_T) {
            stop_reason = " (reached max_T)";
        } else {
            stop_reason = " (FDP > target FDR)";
        }

        printProgress("T-loop converged at T_stop = " +
                      std::to_string(T_stop_) + stop_reason + "\n");
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

// In-memory version for run K random experiments

TRexSelector::ExperimentResults TRexSelector::runRandomExperiments(
    std::size_t num_dummies,
    std::size_t T_stop,
    bool generate_new_dummies,
    bool use_warm_start,
    std::size_t seed_offset
) {

    // ===========================================================
    // 1. Generate or reuse dummy matrices
    // ===========================================================

    if (generate_new_dummies) {

        // Clear old dummies and generate K new ones
        stored_dummies_.clear();
        stored_dummies_.reserve(trex_ctrl_.K);

        // Clear serialized solvers (dummies changed)
        cleanupTempDirectory();

        // Generate K new dummy matrices
        for (std::size_t k = 0; k < trex_ctrl_.K; ++k) {
            stored_dummies_.emplace_back(generateDummies(num_dummies, k + seed_offset));
        }

        if (verbose_) {
            printProgress(
                "Generated " + std::to_string(trex_ctrl_.K) +
                " dummy matrices (" + std::to_string(num_dummies) +
                " dummies each)."
            );
        }
    } else {
        // Reuse existing dummies from stored_dummies_
        if (stored_dummies_.size() != trex_ctrl_.K) {
            if (stored_dummies_.size() != trex_ctrl_.K) {
                throw std::runtime_error("Cannot reuse dummies: stored_dummies has size " +
                                         std::to_string(stored_dummies_.size()) + ", but K = " +
                                         std::to_string(trex_ctrl_.K)
                                        );
            }
        }
#ifndef NDEBUG
        if (verbose_) {
            printProgress(
                "Reusing " + std::to_string(trex_ctrl_.K) + " dummy matrices" +
                (use_warm_start && has_serialized_solvers_ ? " (warm start)" : "")
            );
        }
#endif
    }

    // ===========================================================
    // 2. Setup temp directory for solver serialization
    // ===========================================================

    if (use_warm_start && temp_dir_.empty()) {
        temp_dir_ = createTempDirectory();
        solver_files_.clear();
        solver_files_.reserve(trex_ctrl_.K);
    }

    // ===========================================================
    // 3. Run K random experiments
    // ===========================================================

    // Determine if experiment are sequential/parallel
    const bool do_parallel_rnd_experiments = trex_ctrl_.parallel_rnd_experiments &&
                                            (trex_ctrl_.K > 1);

    // Prepare storage for beta paths
    std::vector<Eigen::MatrixXd> beta_paths(trex_ctrl_.K);

    // Prepare new solver file paths if needed
    std::vector<std::string> new_solver_files;
    if (use_warm_start && !has_serialized_solvers_) {
        new_solver_files.resize(trex_ctrl_.K);
    }

    // Parallel loop over K experiments if do_parallel_rnd_experiments is true else sequential
    #pragma omp parallel if(do_parallel_rnd_experiments)
    {
        // Create augmented data matrix XD
        Eigen::MatrixXd XD(n_, p_ + num_dummies);
        XD.leftCols(p_) = *X_; // copy X only once

        // Due to time varying solvers (different dummies), use dynamic scheduling
        // is sequential if do_parallel_rnd_experiments is false
        #pragma omp for schedule(dynamic)
        for (std::size_t k = 0; k < trex_ctrl_.K; ++k) {

            // Addpend dummies [p_, p_ + num_dummies_)
            XD.rightCols(num_dummies) = stored_dummies_[k];

            // Setup Eigen maps
            Eigen::Map<Eigen::MatrixXd> XD_map(XD.data(), XD.rows(), XD.cols());
            Eigen::Map<Eigen::VectorXd> y_map(y_.data(), y_.size());

            // Determine solver file path
            std::string solver_file;
            if (use_warm_start) {
                if (has_serialized_solvers_ && k < solver_files_.size()) {
                    // Load from existing
                    solver_file = solver_files_[k];
                } else {
                    // Create new path
                    solver_file = temp_dir_ + "/solver_" + std::to_string(k) + ".bin";
                    // SAFE write: only if vector was pre-sized
                    if (!new_solver_files.empty()) {
                        new_solver_files[k] = solver_file;
                    } else if (k == 0 && verbose_) {
                        // Defensive & should not happen
                        printProgress(
                            "WARNING: new_solver_files is empty but trying to create new solver! " +
                            std::string("has_serialized=") +
                            (has_serialized_solvers_ ? "true" : "false") +
                            ", solver_files_.size()=" + std::to_string(solver_files_.size())
                        );
                    }
                }
            }

            // Catch empty solver_file early
            if (use_warm_start && solver_file.empty()) {
                throw std::runtime_error(
                    "solver_file is empty at k=" + std::to_string(k) +
                    "! has_serialized=" + (has_serialized_solvers_ ? "true" : "false") +
                    ", solver_files_.size()=" + std::to_string(solver_files_.size()) +
                    ", temp_dir_='" + temp_dir_ + "'"
                );
            }

            beta_paths[k] = runSingleExperimentWithSolver(
                XD_map, y_map, T_stop, use_warm_start, solver_file
            );
        }
    }

    // Update solver file paths
    if (use_warm_start && !new_solver_files.empty()) {
        solver_files_ = std::move(new_solver_files);
        has_serialized_solvers_ = true;
#ifndef NDEBUG
        if (verbose_) {
            printProgress("Moved " + std::to_string(solver_files_.size()) + " solver file paths");
        }
#endif
    } else if (use_warm_start) {
        has_serialized_solvers_ = true;
#ifndef NDEBUG
        if (verbose_) {
            printProgress("Kept existing " + std::to_string(solver_files_.size()) +
                          " solver file paths");
        }
#endif
    }


    // ===========================================================
    // 4. Compute phi_T_mat from betaPaths
    // ===========================================================

    Eigen::MatrixXd phi_T_mat = computePhiTMat(beta_paths, num_dummies, T_stop);

    // ===========================================================
    // 5. Compute Phi vector
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
    std::size_t rng_seed = seed_ >= 0 ? static_cast<std::size_t>(seed_) + experiment_id :
                                        std::random_device{}() + experiment_id;

    // Generate dummy matrix (n x num_dummies)
    Eigen::MatrixXd D(n_, num_dummies);

    // Generate dummies using configured distribution
    trex::utils::dummygen::generate_dummies(
        D,                                   // Matrix to fill
        n_,                                  // Number of rows
        num_dummies,                         // Number of columns
        rng_seed,                            // RNG seed
        trex_ctrl_.dummy_distribution        // Distribution configuration
    );

    // Normalize dummies column-wise (center + L2 norm)
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    centerAndL2NormalizeMatrix(D_map);

    return D;
}


Eigen::MatrixXd TRexSelector::computePhiTMat(
    const std::vector<Eigen::MatrixXd>& beta_paths,
    std::size_t num_dummies,
    std::size_t T_stop
) const {
    // Initialize phi_T_mat: (p x T_stop)
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
                // NOTE: >= rather than == : more robust (non-monotonic paths)
                if (dummy_num_path(step) >= static_cast<int>(t)) {
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
                    phi_T_mat(j, t - 1) += (1.0 / static_cast<double>(trex_ctrl_.K));
                }
            }
        }
    }

    return phi_T_mat;
}



Eigen::MatrixXd TRexSelector::runSingleExperimentWithSolver(
    Eigen::Map<Eigen::MatrixXd>& XD_map,
    Eigen::Map<Eigen::VectorXd>& y_map,
    std::size_t T_stop,
    bool use_warm_start,
    const std::string& solver_file
) {
    Eigen::MatrixXd pathMatrix;

    // Dispatch to appropriate solver
    switch (trex_ctrl_.solver_type) {

        case SolverTypeForTRex::TLARS: {
            if (use_warm_start && has_serialized_solvers_) {
                TLARS_Solver solver = TLARS_Solver::load(solver_file, XD_map);
                solver.executeStep(T_stop, /*early_stop=*/true);
                pathMatrix = solver.getBetaPath();
                solver.save(solver_file);
            } else {
                TLARS_Solver solver(XD_map, y_map, num_dummies_, false, false, false);
                solver.executeStep(T_stop, /*early_stop=*/true);
                pathMatrix = solver.getBetaPath();
                if (!solver_file.empty()) {
                    solver.save(solver_file);
                }
            }
            break;
        }

        case SolverTypeForTRex::TLASSO: {
            if (use_warm_start && has_serialized_solvers_) {
                TLASSO_Solver solver = TLASSO_Solver::load(solver_file, XD_map);
                solver.executeStep(T_stop, /*early_stop=*/true);
                pathMatrix = solver.getBetaPath();
                solver.save(solver_file);
            } else {
                TLASSO_Solver solver(XD_map, y_map, num_dummies_, false, false, false);
                solver.executeStep(T_stop, /*early_stop=*/true);
                pathMatrix = solver.getBetaPath();
                if (!solver_file.empty()) {
                    solver.save(solver_file);
                }
            }
            break;
        }

        case SolverTypeForTRex::TSTEPWISE: {
            if (use_warm_start && has_serialized_solvers_) {
                TSTEPWISE_Solver solver = TSTEPWISE_Solver::load(solver_file, XD_map);
                solver.executeStep(T_stop, /*early_stop=*/true);
                pathMatrix = solver.getBetaPath();
                solver.save(solver_file);
            } else {
                TSTEPWISE_Solver solver(XD_map, y_map, num_dummies_, false, false, false);
                solver.executeStep(T_stop, /*early_stop=*/true);
                pathMatrix = solver.getBetaPath();
                if (!solver_file.empty()) {
                    solver.save(solver_file);
                }
            }
            break;
        }

        case SolverTypeForTRex::TENET: {
            double lambda2 = trex_ctrl_.tenet_lambda2;

            if (use_warm_start && has_serialized_solvers_) {
                TENET_Solver solver = TENET_Solver::load(solver_file, XD_map);
                solver.executeStep(T_stop, /*early_stop=*/true);
                pathMatrix = solver.getBetaPath();
                solver.save(solver_file);
            } else {
                TENET_Solver solver(XD_map, y_map, num_dummies_, lambda2,
                                    false, false, false);
                solver.executeStep(T_stop, /*early_stop=*/true);
                pathMatrix = solver.getBetaPath();
                if (!solver_file.empty()) {
                    solver.save(solver_file);
                }
            }
            break;
        }

        case SolverTypeForTRex::TOMP: {
            if (use_warm_start && has_serialized_solvers_) {
                TOMP_Solver solver = TOMP_Solver::load(solver_file, XD_map);
                solver.executeStep(T_stop, /*early_stop=*/true);
                pathMatrix = solver.getBetaPath();
                solver.save(solver_file);
            } else {
                TOMP_Solver solver(XD_map, y_map, num_dummies_, false, false, false);
                solver.executeStep(T_stop, /*early_stop=*/true);
                pathMatrix = solver.getBetaPath();
                if (!solver_file.empty()) {
                    solver.save(solver_file);
                }
            }
            break;
        }

        case SolverTypeForTRex::TGP: {
            if (use_warm_start && has_serialized_solvers_) {
                TGP_Solver solver = TGP_Solver::load(solver_file, XD_map);
                solver.executeStep(T_stop, /*early_stop=*/true);
                pathMatrix = solver.getBetaPath();
                solver.save(solver_file);
            } else {
                TGP_Solver solver(XD_map, y_map, num_dummies_, false, false, false);
                solver.executeStep(T_stop, /*early_stop=*/true);
                pathMatrix = solver.getBetaPath();
                if (!solver_file.empty()) {
                    solver.save(solver_file);
                }
            }
            break;
        }

        case SolverTypeForTRex::TACGP: {
            if (use_warm_start && has_serialized_solvers_) {
                TACGP_Solver solver = TACGP_Solver::load(solver_file, XD_map);
                solver.executeStep(T_stop, /*early_stop=*/true);
                pathMatrix = solver.getBetaPath();
                solver.save(solver_file);
            } else {
                TACGP_Solver solver(XD_map, y_map, num_dummies_, false, false, false);
                solver.executeStep(T_stop, /*early_stop=*/true);
                pathMatrix = solver.getBetaPath();
                if (!solver_file.empty()) {
                    solver.save(solver_file);
                }
            }
            break;
        }

        default: {
            throw std::invalid_argument("Unsupported solver type.");
        }
    }

    return pathMatrix;
}



// ===========================================================
// Memory-Mapped Matrix Management
// ===========================================================


// Setup memory-mapped matrices

void TRexSelector::initializeMemoryMappedMatrices() {
    if (!trex_ctrl_.use_memory_mapping) { return; }

    // Ensure temp directory exists
    if (temp_dir_.empty()) { temp_dir_ = createTempDirectory(); }

    // Compute max columns
    max_cols_memmap_ = p_ + (trex_ctrl_.max_num_dummies * p_);

    if (verbose_) {
        printProgress("Initializing " + std::to_string(trex_ctrl_.K) +
                      " memory-mapped matrices...");
    }

    // Create K memory-mapped files
    XD_memmaps_.clear();
    XD_memmaps_.reserve(trex_ctrl_.K);

    // Setup memory-mapped matrices
    for (std::size_t k = 0; k < trex_ctrl_.K; ++k) {
        std::string filename = temp_dir_ + "/XD_aug_memmap_" + std::to_string(k) + ".bin";

        try {

            auto memmap = std::make_unique<memmap::MemoryMappedMatrix<double>>(
                filename,
                n_,
                max_cols_memmap_,
                memmap::AccessMode::ReadWrite
            );

            XD_memmaps_.push_back(std::move(memmap));

        } catch (const std::exception& e) {
            // Clean up all previously created memory-mapped matrices
            cleanupMemoryMappedMatrices();
            throw std::runtime_error(
                "Failed to create memory-mapped matrix " + std::to_string(k) +
                ": " + std::string(e.what())
            );
        }
    }
}


// Write X to memory-mapped matrices

void TRexSelector::writeXToMemoryMappedMatrices() {
    if (!trex_ctrl_.use_memory_mapping) { return; }

    if (verbose_) { printProgress("Writing X to memory-mapped matrices..."); }

    // Write X to each memory-mapped matrix
    for (std::size_t k = 0; k < trex_ctrl_.K; ++k) {
        auto XD_map = XD_memmaps_[k]->getMap();

        // Copy X to first p columns
        XD_map.leftCols(p_) = *X_;
    }
}


// Clean up memory-mapped matrices

void TRexSelector::cleanupMemoryMappedMatrices() {
    if (!trex_ctrl_.use_memory_mapping) { return; }

    if (verbose_) { printProgress("Cleaning up memory-mapped matrices..."); }

    // Clear vector of memory-mapped matrix pointers
    XD_memmaps_.clear();

    // Clean temporary directory
    cleanupTempDirectory();

    if (verbose_) { printProgress("Memory-mapped matrices cleaned up."); }
}


// Run random experiments with memory-mapped matrices

TRexSelector::ExperimentResults TRexSelector::runRandomExperiments_MemMap(
    std::size_t num_dummies,
    std::size_t T_stop,
    bool use_warm_start
) {
    if (verbose_) {
        printProgress("Running " + std::to_string(trex_ctrl_.K) +
                      " experiments (memory-mapped)" +
                      (use_warm_start && has_serialized_solvers_ ? " (warm start)" : "")
        );
    }

    // Verify memory-mapped matrices are initialized
    if (XD_memmaps_.empty()) {
        throw std::runtime_error(
            "Memory-mapped matrices are not initialized."
            " Call initializeMemoryMappedMatrices() first."
        );
    }

    // Setup temp directory for solver serialization
    if (use_warm_start && temp_dir_.empty()) {
        temp_dir_ = createTempDirectory();
        solver_files_.clear();
        solver_files_.reserve(trex_ctrl_.K);
    }

    // Setup K random experiments
    std::vector<Eigen::MatrixXd> beta_paths(trex_ctrl_.K);

    // Storage for new solver file paths
    std::vector<std::string> new_solver_files;
    if (use_warm_start && !has_serialized_solvers_) {
        new_solver_files.reserve(trex_ctrl_.K);
    }

    // Determine if experiments are sequential/parallel
    const bool do_parallel_rnd_experiments = trex_ctrl_.parallel_rnd_experiments &&
                                            (trex_ctrl_.K > 1);

    // Parallel execution over K experiments (if enabled)
    #pragma omp parallel if(do_parallel_rnd_experiments)
    {
        // No thread-local variables here, each k has its own file
        // Same structure as in-memory version for consistency

        #pragma omp for schedule(dynamic)
        for (std::size_t k = 0; k < trex_ctrl_.K; ++k) {
            // Get memory-mapped augmented matrix pointers
            auto full_map = XD_memmaps_[k]->getMap();

            // Create Eigen::Map over active columns [0, p + num_dummies)
            Eigen::Map<Eigen::MatrixXd> XD_active(
                full_map.data(),
                n_,
                p_ + num_dummies
            );

            Eigen::Map<Eigen::VectorXd> y_map(y_.data(), y_.size());

            // Determine solver file path
            std::string solver_file;
            if (use_warm_start) {
                if (has_serialized_solvers_ && k < solver_files_.size()) {
                    solver_file = solver_files_[k];
                } else {
                    solver_file = temp_dir_ + "/solver_" + std::to_string(k) + ".bin";
                    if (!new_solver_files.empty()) {
                        new_solver_files[k] = solver_file;
                    }
                }
            }

            // Run solver and obtain beta path matrix
            beta_paths[k] = runSingleExperimentWithSolver(
                XD_active, y_map, T_stop, use_warm_start, solver_file
            );
        }
    }

    // Update solver file paths
    if (use_warm_start && !new_solver_files.empty()) {
        solver_files_ = std::move(new_solver_files);
        has_serialized_solvers_ = true;
    } else if (use_warm_start) {
        has_serialized_solvers_ = true;
    }

    // Compute phi_T_mat from betaPaths
    Eigen::MatrixXd phi_T_mat = computePhiTMat(beta_paths, num_dummies, T_stop);

    // Compute Phi vector
    Eigen::VectorXd Phi = phi_T_mat.col(T_stop - 1);

    // Gather results and return
    ExperimentResults results;
    results.phi_T_mat = phi_T_mat;
    results.Phi = Phi;

    return results;
}
