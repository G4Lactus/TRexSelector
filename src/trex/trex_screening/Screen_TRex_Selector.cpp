#include <cmath>
#include <numeric>
#include <map>
#include <boost/math/special_functions/erf.hpp>

#include "Screen_TRex_Selector.hpp"


// =============================================================
// // Constructor
// =============================================================

ScreenTRexSelector::ScreenTRexSelector(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::VectorXd>& y,
    ScreenTRexControlParameter screen_control,
    TRexControlParameter trex_control,
    SolverControl solver_control,
    int seed,
    bool verbose
) : TRexSelector(X,
                 y,
                 0.0/* tFDR ignored in constructor*/,
                 trex_control,
                 std::move(solver_control),
                 seed,
                 verbose),
    screen_ctrl_(screen_control)
{
    /*
       Screen-TRex enforces: L = p.
       Base class constructor normalizes and performs checks.
     */
}


// =============================================================
// Main method
// =============================================================

TRexSelector::SelectionResult ScreenTRexSelector::select() {

    // 1. Force Screen-TRex parameters
    T_stop_ = 1;                     // T = 1
    num_dummies_ = p_;               // L = p

    if (verbose_) {
        std::string mode = "Ordinary";
        if (screen_ctrl_.use_bootstrap_CI) { mode = "Confidence-Based"; };
        std::string trex_method = "TRex";
        if (screen_ctrl_.trex_method == ScreenTRexMethod::TREX_DA_AR1) {
            trex_method = "DA-AR1";
        } else if (screen_ctrl_.trex_method == ScreenTRexMethod::TREX_DA_EQUI) {
            trex_method = "DA-EQUI";
        }
        printProgress("Starting Screen-TRex selection (T=1, L=p)...");
    }

    // 2. Run Experiments & capture betas
    runScreenExperiments();

    // 3. Compute Statistics
    // Average betas: (p x 1)
    Eigen::VectorXd beta_means = accumulated_beta_mat_.col(0) /
                            static_cast<double>(trex_ctrl_.K);

    // Compute Phi (relative occurence) from random experiments
    Eigen::VectorXd Phi = phi_T_mat_.col(0);

    // Apply Dependency Aware Adjustment if required
    if (screen_ctrl_.trex_method != ScreenTRexMethod::TREX) {
        applyDependencyAwareAdjustment(Phi);
    }

    // 4. Selection Logic
    if (screen_ctrl_.use_bootstrap_CI) {
        performBootstrapSelection(Phi, beta_means, screen_result_);
    } else {
        performOrdinarySelection(Phi, screen_result_);
    }

    // 5. Compute Estimated FDR = 1 / max(1, R)
    computeEstimatedFDR(screen_result_);

    // 6. Populate standard fields for base class compatibility
    screen_result_.T_stop = 1;
    screen_result_.num_dummies = p_;
    screen_result_.beta_mat = beta_means;
    screen_result_.Phi_mat = Phi;
    screen_result_.Phi_prime = Phi; // Approximation for T = 1

    // R_mat: (T x V), T = 1, V = 1
    screen_result_.R_mat = Eigen::MatrixXd::Zero(1,1);
    screen_result_.R_mat(0,0) = screen_result_.selected_var.sum();

    // Store in base class state
    storeResults(screen_result_);

    // Restore X to original scale
    denormalizeX();

    // Return sliced obj
    return static_cast<TRexSelector::SelectionResult>(screen_result_);
}

// ============================================================
// Screen-TRex specific public getter
// ============================================================

const ScreenTRexSelectionResult& ScreenTRexSelector::getScreenResult() const {
    return screen_result_;
}


// =============================================================
// Dependency Aware Helpers
// =============================================================

double ScreenTRexSelector::estimateAR1Correlation() const {
    /*
        Estimates AR(1) process along the variables (columns) for each row.
        Estimates AR(1) autocorrelation: x_j = ρ·x_{j-1} + ε_j

        Uses Yule-Walker estimator (MLE for Gaussian AR(1)):
        ρ̂ = Σ(x_j - x̄)(x_{j-1} - x̄) / Σ(x_j - x̄)²

        R reference:
            mean(apply(X, 1, function(smpl) {
                stats::coef(stats::arima(smpl, order = c(1, 0, 0),
                                    include.mean = FALSE, method = "ML"))
            }))

        Note: Columns are L2 normalized, not rows.

        OPTIMIZATION: Two-pass algorithm minimizing temporaries
        - First pass: compute row means
        - Second pass: accumulate lag-1 autocorrelation
        - Avoid row copies

        Note: Welford's algorithm for online mean computation is possible but costs memory.
     */

     double total_rho = 0.0;

    // Iterate over rows (time series samples)
    for (std::size_t i = 0; i < n_; ++i) {

        // 1. Pass: Compute row means
        double row_sum = 0.0;
        for (std::size_t j = 0; j < p_; ++j) {
            row_sum += (*X_)(i, j);
        }
        double row_mean = row_sum / static_cast<double>(p_);


        // 2. Pass: Compute lag-1 autocorrelation (Yule-Walker)
        // hat_rho = sum(x_t * x_{t-1}) / sum(x_{t}^2)
        // Note: denominator in Yule-Walker is the variance (sum of squares)
        double numerator = 0.0;
        double denominator = 0.0;

        // Cache first centered value for lag computation
        double x_prev = (*X_)(i, 0) - row_mean;
        denominator += x_prev * x_prev;

        for (std::size_t j = 1; j < p_; ++j) {
            double x_curr = (*X_)(i, j) - row_mean;

            // Accumulate variance
            denominator += x_curr * x_curr;

            // Accumulate lag-1 covariance
            numerator += x_curr * x_prev;

            // Update lag
            x_prev = x_curr;
        }

        // Accumulate autocorrelation for this row
        if (denominator > eps_) {
            total_rho += (numerator / denominator);
        }
    }

    // Return averaged absolute autocorrelation
    return std::abs(total_rho / static_cast<double>(n_));
}


double ScreenTRexSelector::estimateEquiCorrelation() const {
    /*
        For L2-normalized columns: Var(row_sums) = p + p(p-1)ρ̄
        Therefore: ρ̄ = (||row_sums||² - p) / (p(p-1))

        OPTIMIZATION: Column-wise accumulation for Eigen::Map
        - Sequential memory access (optimal for memory-mapped data)
        - Full SIMD vectorization
    */

    // Accumulate row sums column-wise (cache-friendly, vectorized)
    double sum_sq_row_sums = 0.0;
    {
        Eigen::VectorXd row_sums = Eigen::VectorXd::Zero(n_);

        for (std::size_t j = 0; j < p_; ++j) {
            // Sequential access, fully vectorized, no alias hint: now overlap with X_->col(j)
            row_sums.noalias() += X_->col(j);
        }

        // Compute ||row_sums||²
        sum_sq_row_sums = row_sums.squaredNorm();
    }

     // 2. Solve for rho: sum_sq = p + p (p - 1) * rho
     double numerator = sum_sq_row_sums - static_cast<double>(p_);
     double denominator = static_cast<double>(p_) * static_cast<double>(p_ - 1);

     if (denominator < eps_) { return 0.0; } // Edge case p = 1

    return numerator / denominator;
}


void ScreenTRexSelector::applyDependencyAwareAdjustment(Eigen::VectorXd& Phi) {

    // 1. Estimate Correlation if needed
    double rho = screen_ctrl_.cor_coef;

    if (std::isnan(rho)) {
        switch (screen_ctrl_.trex_method) {
            case ScreenTRexMethod::TREX_DA_AR1:
                 rho = estimateAR1Correlation();
                 break;
            case ScreenTRexMethod::TREX_DA_EQUI:
                rho = estimateEquiCorrelation();
                break;
            default: // Should not happen
                throw std::runtime_error("Invalid Screen-TRex method for DA adjustment.");
        }
    }
    screen_result_.estimated_correlation = rho;

    if (verbose_) {
        printProgress("DA rho = " + std::to_string(rho));
    }


    // 2. Compute DA adjustment factors based on method
    Eigen::VectorXd DA_delta = [&]() -> Eigen::VectorXd {
        switch (screen_ctrl_.trex_method) {
            case ScreenTRexMethod::TREX_DA_AR1:
                return computeDA_AR1_Delta(Phi, rho);
            case ScreenTRexMethod::TREX_DA_EQUI:
                return computeDA_EQUI_Delta(Phi, rho);
            default:
                throw std::runtime_error("Unreachable: Invalid method in DA adjustment.");
        }
    }();


    // 3. Apply Adjustment to Phi
    /* R reference:
        phi_T_mat = phi_T_mat / DA_delta
        Phi = Phi / DA_delta
    */
    for (std::size_t j = 0; j < p_; ++j) {
        if (DA_delta(j) > eps_) {
            phi_T_mat_(j, 0) /= DA_delta(j);
            Phi(j) /= DA_delta(j);
        }
    }
}

// =============================================================
// DA Strategies
// =============================================================

// DA-AR1 Strategy

Eigen::VectorXd ScreenTRexSelector::computeDA_AR1_Delta(
    const Eigen::VectorXd& Phi,
    double rho
) const {
    /*
        AR(1) Dependency-Aware Adjustment:

        For each variable j, find the minimum absolute difference |Phi[j] - Phi[k]|
        within a local window of neighbors [j-kap, j+kap].

        Window size kap = ceiling(log(rho_thr_DA) / log(rho))
        captures the range where AR(1) correlation decays below threshold.

        Adjustment: DA_delta[j] = 2 - min_diff
        Effect: Variables with similar neighbors get stronger deflation.
    */

    // Compute window size from correlation decay
    double valid_rho = std::min(0.999, std::max(0.001, std::abs(rho)));
    int kap = static_cast<int>(
        std::ceil(std::log(screen_ctrl_.rho_thr_DA) / std::log(valid_rho))
    );
    kap = std::max(1, kap);

    if (verbose_) {
        printProgress("DA-AR1: Window size kap = " + std::to_string(kap));
    }

    Eigen::VectorXd DA_delta = Eigen::VectorXd::Ones(p_);

    for (std::size_t j = 0; j < p_; ++j) {
        double min_diff = 2.0;

        // Sliding window [j - kap, j + kap] (excluding j itself)
        long start = std::max(0L, static_cast<long>(j) - kap);
        long end = std::min(static_cast<long>(p_) - 1, static_cast<long>(j) + kap);

        // Find minimum difference in neighborhood
        for (long k = start; k <= end; ++k) {
            if (k != static_cast<long>(j)) {
                min_diff = std::min(min_diff, std::abs(Phi(j) - Phi(k)));
            }
        }

        // Edge case: isolated variable (no neighbors) or no variation in neighborhood
        DA_delta(j) = 2.0 - (min_diff > 1.5 ? 0.0 : min_diff);
    }

    return DA_delta;
}


// DA-EQUI Strategy

Eigen::VectorXd ScreenTRexSelector::computeDA_EQUI_Delta(
    const Eigen::VectorXd& Phi,
    double rho
) const {
    /*
        EQUI Dependency-Aware Adjustment:

        For each variable j, find the minimum absolute difference |Phi[j] - Phi[k]|
        across all other variables k != j.

        Only apply adjustment if |rho| > rho_thr_DA (significant global correlation).

        Implementation:
        Exact O(p^2) with Eigen vectorization.
        Each iteration computes |Phi - Phi[j]| vectorized, then finds min excluding self.

        Adjustment: DA_delta[j] = 2 - min_diff
        Effect: Variables in dense clusters get stronger deflation.

        Note:
        Exact is computationally intensive for large p. A O(p log p) approximation exists,
        but comes at a 5% error on average.
    */

    Eigen::VectorXd DA_delta = Eigen::VectorXd::Ones(p_);

    // Apply adjustment only if correlation exceeds threshold
    if (std::abs(rho) <= screen_ctrl_.rho_thr_DA) {
        if (verbose_) {
            printProgress("DA-EQUI: Skipping (|rho| = " + std::to_string(std::abs(rho)) +
                          " <= rho_thr_DA = " + std::to_string(screen_ctrl_.rho_thr_DA) + ").");
        }
        return DA_delta;
    }

    // Global comparison (exact, vectorized)
    if (verbose_) {
        printProgress("DA-EQUI: Computing exact O(p^2) adjustment");
    }

    for (std::size_t j = 0; j < p_; ++j) {
        // Vectorized: compute |Phi - Phi[j]|
        Eigen::VectorXd diffs = (Phi.array() - Phi(j)).abs();

        // Exclude self-comparison
        diffs(j) = std::numeric_limits<double>::max();

        // Find minimum difference
        double min_diff = diffs.minCoeff();

        // Edge case: all other values identical to Phi[j]
        // Conservative choice: no adjustment
        DA_delta(j) = 2.0 - (min_diff > 1.5 ? 0.0 : min_diff);
    }

    return DA_delta;
}


// =============================================================
// Selection Logic Helpers
// =============================================================

void ScreenTRexSelector::performOrdinarySelection(
    const Eigen::VectorXd& Phi,
    ScreenTRexSelectionResult& result_out
) const {
    result_out.used_bootstrap = false;
    result_out.confidence_level = 0.0;

    // Select variables with Phi > 0.5
    result_out.selected_var = (Phi.array() > 0.5).cast<int>();
}


void ScreenTRexSelector::computeEstimatedFDR(
    ScreenTRexSelectionResult& result_out
) const {
    double R = result_out.selected_var.sum();

    // FDP estimate: 1 / max(1, R)
    if (R < 1.0) {
        result_out.estimated_FDR = 0.0;
    } else {
        result_out.estimated_FDR = 1.0/ R;
    }
}


// =============================================================
// Experiment Execution
// =============================================================

void ScreenTRexSelector::runScreenExperiments() {

    // Reset storage
    accumulated_beta_mat_ = Eigen::MatrixXd::Zero(p_, 1);
    collected_dummy_betas_.clear();

    // Reset base class Phi tracking (p x 1)
    phi_T_mat_ = Eigen::MatrixXd::Zero(p_, 1);

    // Clean temp dir for solvers
    cleanupTempDirectory();
    temp_dir_ = createTempDirectory();

    // Run base class experiments with T = 1, L = p
    for (std::size_t k = 0; k < trex_ctrl_.K; ++k) {
        // 1. Generate dummies (L = p)
        // Base generateDummies (n x L)
        Eigen::MatrixXd D = generateDummies(p_, k);

        // 2. Augment Matrix [X | D] (n x (p + L))
        Eigen::MatrixXd XD = createAugmentedMatrix(D);
        Eigen::Map<Eigen::MatrixXd> XD_map(XD.data(), XD.rows(), XD.cols());
        Eigen::Map<Eigen::VectorXd> y_map(y_.data(), y_.size());

        // 3. Run Solver (T_stop = 1)
        std::string solver_file = temp_dir_ + "/solver_" + std::to_string(k) + ".bin";

        Eigen::MatrixXd beta_path = runSingleExperimentWithSolver(
            XD_map,
            y_map,
            1,                  // T_stop = 1
            false,              // No warm start
            solver_file
        );

        // 4. Extract Coefficients from final step
        if (beta_path.cols() == 0) { continue; }
        std::size_t last_step = beta_path.cols() - 1;
        Eigen::VectorXd betas_k = beta_path.col(last_step);

        // 5. Process Coefficients
        // Original features:
        for (std::size_t j = 0; j < p_; ++j) {
            double beta_val = betas_k(j);
            accumulated_beta_mat_(j, 0) += beta_val;

            // Update relative occurences
            if (std::abs(beta_val) > eps_) {
                phi_T_mat_(j, 0) += (1.0 / static_cast<double>(trex_ctrl_.K));
            }
        }

        // Dummy Variables
        for (std::size_t d = 0; d < p_; ++d) {
            double dummy_beta_val = betas_k(p_ + d);
            if (std::abs(dummy_beta_val) > eps_) {
                collected_dummy_betas_.push_back(dummy_beta_val);
            }
        }
    }
}


// =============================================================
// Bootstrap Helpers
// =============================================================

double ScreenTRexSelector::probit(double p) const {
    if (p <= 0.0 || p >= 1.0) { return 0.0; }
    return std::sqrt(2) * boost::math::erf_inv(2.0 * p - 1.0);
}


void ScreenTRexSelector::performBootstrapSelection(
    const Eigen::VectorXd& Phi,
    const Eigen::VectorXd& beta_means,
    ScreenTRexSelectionResult& result
) {
    result.used_bootstrap = true;

    // 1. Count "Ordinary" selections for constraint
    long R_ordinary = (Phi.array() > 0.5).count();

    std::size_t n_dummies = collected_dummy_betas_.size();
    if (n_dummies == 0) {
        if (verbose_) {
            printProgress("No dummies selected during experiments."
                          " Falling back to ordinary selection.");
        }
        // No dummies selected: fallback to ordinary selection
        result.used_bootstrap = false;
        result.selected_var = (Phi.array() > 0.5).cast<int>();
        return;
    }

    // 2. Perform Boostrap on Dummy Coefficients
    std::vector<double> boot_means;
    boot_means.reserve(screen_ctrl_.R_boot);

    std::mt19937_64 gen(seed_ >= 0 ? seed_ : std::random_device{}());
    std::uniform_int_distribution<std::size_t> dist(0, n_dummies - 1);

    for (std::size_t r = 0; r < screen_ctrl_.R_boot; ++r) {
        double sum = 0.0;

        // R default bootstrap size is length(data)
        for (std::size_t i = 0; i < n_dummies; ++i) {
            sum += collected_dummy_betas_[dist(gen)];
        }
        boot_means.push_back(sum / static_cast<double>(n_dummies));
    }

    // 3. Grid search for Confidence Level gamma
    bool found = false;
    double optimal_gamma = 0.0;
    Eigen::VectorXi optimal_mask = Eigen::VectorXi::Zero(p_);

    for (double gamma = 0.0; gamma < 1.0; gamma += screen_ctrl_.ci_grid_step) {

        std::pair<double, double> ci = calculateBoostrapCI(boot_means, gamma);
        double lower = ci.first;
        double upper = ci.second;

        // Count selections: betas > upper CI bound
        long count = 0;
        Eigen::VectorXi current_mask = Eigen::VectorXi::Zero(p_);

        for (std::size_t j = 0; j < p_; ++j) {
            if (beta_means(j) < lower || beta_means(j) > upper) {
                count++;
                current_mask(j) = 1;
            }
        }

        // Find first gamma where count <= R_ordinary
        if (count <= R_ordinary) {
            optimal_gamma = gamma;
            optimal_mask = current_mask;
            found = true;
            break;
        }
    }

    if (!found) {
        // Conservative fallback: even at 99.9% confidence too many selections
        optimal_gamma = 1.0;
        optimal_mask.setZero();
    }

    result.selected_var = optimal_mask;
    result.confidence_level = optimal_gamma;
    result.dummy_betas = Eigen::Map<Eigen::VectorXd>(collected_dummy_betas_.data(),
                                                     collected_dummy_betas_.size());
}


std::pair<double, double> ScreenTRexSelector::calculateBoostrapCI(
    const std::vector<double>& means_distribution,
    double confidence_level
) const {

    // Caclulate mean and stddev of bootstrap means
    double sum = std::accumulate(means_distribution.begin(), means_distribution.end(), 0.0);
    double mean_bootdist = sum / static_cast<double>(means_distribution.size());

    double sq_sum = 0.0;
    for (double v : means_distribution) {
        double diff = v - mean_bootdist;
        sq_sum += diff * diff;
    }
    double sd_boot = std::sqrt(sq_sum / (means_distribution.size() - 1));

    // Calculate Normal CI: mean +/- z * sd
    double alpha = (1.0 - confidence_level) / 2.0;

    // z-score for 1-alpha
    double z = probit(1.0 - alpha);

    return {mean_bootdist - z * sd_boot,
            mean_bootdist + z * sd_boot};
}
