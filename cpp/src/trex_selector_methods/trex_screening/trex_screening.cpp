// ===================================================================================
// trex_screening.cpp
// ===================================================================================
/**
 * @file trex_screening.cpp
 *
 * @brief Screen-TRex Selector implementation.
 */
// ===================================================================================

// std includes
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

// Boost Math (probit via erf_inv)
#include <boost/math/special_functions/erf.hpp>

// Header
#include <trex_selector_methods/trex_screening/trex_screening.hpp>

// ===================================================================================

namespace trex::trex_selector_methods::trex_screening {

// Namespace aliases
namespace tc       = trex::trex_selector_methods::trex_core;
namespace dn       = trex::trex_selector_methods::utils::data_normalizer;
namespace er       = trex::trex_selector_methods::utils::experiment_runner;
namespace dummygen = trex::utils::datageneration::dummygen;

// ===================================================================================
// Constructor
// ===================================================================================

ScreenTRexSelector::ScreenTRexSelector(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::VectorXd>& y,
    ScreenTRexControlParameter   screen_control,
    tc::TRexControlParameter     trex_control,
    int  seed,
    bool verbose
) : tc::TRexSelector(X, y,
                     /*tFDR=*/0.0,  // Screen-TRex does not use FDR calibration
                     trex_control,
                     seed,
                     verbose),
    screen_ctrl_(screen_control)
{
    validateScreenTRexStrategy();
}


// ===================================================================================
// Validation
// ===================================================================================

void ScreenTRexSelector::validateScreenTRexStrategy() const {
    switch (trex_ctrl_.lloop_strategy) {
        case tc::LLoopStrategy::STANDARD:
        case tc::LLoopStrategy::PERMUTATION:
            break;  // Supported
        default:
            throw std::invalid_argument(
                "ScreenTRexSelector: only STANDARD and PERMUTATION dummy "
                "strategies are supported. Got lloop_strategy = "
                + std::to_string(static_cast<int>(trex_ctrl_.lloop_strategy)));
    }
}


// ===================================================================================
// Main method: select()
// ===================================================================================

tc::TRexSelector::SelectionResult ScreenTRexSelector::select() {

    // 1. Enforce Screen-TRex fixed parameters: T = 1, L = p
    T_stop_             = 1;
    num_dummies_        = p_;
    dummy_multiplier_LL_ = 1;

    if (verbose_) {
        const std::string mode =
            screen_ctrl_.use_bootstrap_CI ? "Confidence-Based" : "Ordinary";
        std::string variant = "TRex";
        if (screen_ctrl_.trex_method == ScreenTRexMethod::TREX_DA_AR1) {
            variant = "DA-AR1";
        } else if (screen_ctrl_.trex_method == ScreenTRexMethod::TREX_DA_EQUI) {
            variant = "DA-EQUI";
        } else if (screen_ctrl_.trex_method == ScreenTRexMethod::TREX_DA_BLOCK_EQUI) {
            variant = "DA-BLOCK-EQUI";
        }
        printProgress("Starting Screen-TRex [" + mode + ", " + variant
                      + "] (T=1, L=p=" + std::to_string(p_) + ")...");
    }

    // 2. Run K experiments at T=1, L=p; collect phi, beta sums, dummy betas
    er::ExperimentResults exp_results = runScreenExperiments();

    // 3. Compute statistics
    //    beta_means: average over K experiments
    const Eigen::VectorXd beta_means =
        accumulated_beta_mat_.col(0) / static_cast<double>(trex_ctrl_.K);

    //    Phi: relative occurrences at T=1 (p × 1)
    Eigen::VectorXd Phi = exp_results.phi_T_mat.col(0);

    // 4. Dependency-Aware adjustment (modifies Phi in-place)
    if (screen_ctrl_.trex_method != ScreenTRexMethod::TREX) {
        applyDependencyAwareAdjustment(Phi);
    }

    // 5. Selection
    if (screen_ctrl_.use_bootstrap_CI) {
        performBootstrapSelection(Phi, beta_means, screen_result_);
    } else {
        performOrdinarySelection(Phi, screen_result_);
    }

    // 6. Estimated FDR = 1 / max(1, R)
    computeEstimatedFDR(screen_result_);

    // 6b. Always store collected dummy betas in result (needed by biobank bootstrap reuse)
    if (!collected_dummy_betas_.empty()) {
        screen_result_.dummy_betas = Eigen::Map<Eigen::VectorXd>(
            collected_dummy_betas_.data(),
            static_cast<Eigen::Index>(collected_dummy_betas_.size()));
    }

    // 7. Populate SelectionResult fields
    screen_result_.T_stop         = 1;
    screen_result_.num_dummies    = p_;
    screen_result_.dummy_factor_L = 1;
    screen_result_.v_thresh       = 0.5;
    screen_result_.beta_mat       = beta_means;

    // Phi_mat: T_stop × p = 1 × p
    screen_result_.Phi_mat = Phi.transpose();

    // Phi_prime: approximation for T=1 (equals Phi; Phi_prime is unused in selection)
    screen_result_.Phi_prime = Phi;

    // R_mat: 1 × 1 with the number of selected variables
    screen_result_.R_mat =
        Eigen::MatrixXd::Ones(1, 1)
        * static_cast<double>(screen_result_.selected_var.sum());

    // FDP_hat_mat: not calibrated for Screen-TRex (no T-loop / FDR sweep)
    screen_result_.FDP_hat_mat = Eigen::MatrixXd::Zero(1, 1);

    // voting_grid: standard K-grid (for downstream compatibility)
    screen_result_.voting_grid = createVotingGrid();

    // 8. Sync base class result members accessible via getters
    Phi_mat_     = screen_result_.Phi_mat;
    FDP_hat_mat_ = screen_result_.FDP_hat_mat;
    Phi_prime_   = screen_result_.Phi_prime;
    voting_grid_ = screen_result_.voting_grid;

    // 9. Store via base class (updates selected_var_, selected_indices_,
    //    voting_threshold_, R_mat_)
    storeResults(screen_result_);

    if (verbose_) {
        printProgress("Screen-TRex: selected "
                      + std::to_string(getNumSelected()) + " variables.");
    }

    // 10. Cleanup helpers
    warm_start_mgr_.cleanup();
    if (memmap_mgr_) { memmap_mgr_->cleanup(); }

    // 11. Restore X to original scale
    if (X_is_normalized_) {
        dn::denormalizeX(*X_, norm_params_);
        X_is_normalized_ = false;
    }

    return static_cast<tc::TRexSelector::SelectionResult>(screen_result_);
}


// ===================================================================================
// Public getter
// ===================================================================================

const ScreenTRexSelectionResult& ScreenTRexSelector::getScreenResult() const {
    return screen_result_;
}


ScreenTRexSelectionResult ScreenTRexSelector::applyBootstrapToOrdinaryResult(
    const ScreenTRexSelectionResult& ordinary_result
) {
    // Restore collected_dummy_betas_ from the ordinary result
    const Eigen::VectorXd& db = ordinary_result.dummy_betas;
    collected_dummy_betas_.assign(db.data(), db.data() + db.size());

    // Extract Phi (1 × p row → p × 1 column) and beta_means from ordinary result
    const Eigen::VectorXd Phi        = ordinary_result.Phi_mat.row(0).transpose();
    const Eigen::VectorXd& beta_means = ordinary_result.beta_mat;

    // Perform bootstrap selection using the shared experiment data
    ScreenTRexSelectionResult bootstrap_result;
    performBootstrapSelection(Phi, beta_means, bootstrap_result);
    computeEstimatedFDR(bootstrap_result);

    // Copy structural fields from the ordinary result
    bootstrap_result.T_stop         = ordinary_result.T_stop;
    bootstrap_result.num_dummies    = ordinary_result.num_dummies;
    bootstrap_result.dummy_factor_L = ordinary_result.dummy_factor_L;
    bootstrap_result.v_thresh       = ordinary_result.v_thresh;
    bootstrap_result.beta_mat       = ordinary_result.beta_mat;
    bootstrap_result.Phi_mat        = ordinary_result.Phi_mat;
    bootstrap_result.Phi_prime      = ordinary_result.Phi_prime;
    bootstrap_result.R_mat          = Eigen::MatrixXd::Ones(1, 1)
        * static_cast<double>(bootstrap_result.selected_var.sum());
    bootstrap_result.FDP_hat_mat    = Eigen::MatrixXd::Zero(1, 1);
    bootstrap_result.voting_grid    = ordinary_result.voting_grid;
    bootstrap_result.estimated_correlation = ordinary_result.estimated_correlation;

    return bootstrap_result;
}


// ===================================================================================
// Core: runScreenExperiments
// ===================================================================================

er::ExperimentResults ScreenTRexSelector::runScreenExperiments() {

    // Reset storage
    accumulated_beta_mat_.resize(static_cast<Eigen::Index>(p_), 1);
    accumulated_beta_mat_.setZero();
    collected_dummy_betas_.clear();

    // Map lloop_strategy to ExperimentStrategy and pre-generate dummies.
    // Memory-mapped path generates directly into mmap files via ExperimentRunner;
    // in-memory path needs dummies pre-generated into DummyGenerator storage.
    er::ExperimentStrategy strategy;
    const bool use_mmap = trex_ctrl_.use_memory_mapping;

    switch (trex_ctrl_.lloop_strategy) {
        case tc::LLoopStrategy::STANDARD: {
            strategy = er::ExperimentStrategy::Standard;
            if (!use_mmap) {
                dummy_gen_.generateAndStore(trex_ctrl_.K, p_);
            }
            break;
        }
        case tc::LLoopStrategy::PERMUTATION: {
            strategy = er::ExperimentStrategy::Permutation;
            // Generate and store the base dummy matrix for row-permutation
            const unsigned int base_seed =
                (seed_ >= 0) ? static_cast<unsigned int>(seed_)
                             : std::random_device{}();
            Eigen::MatrixXd base = dummy_gen_.generate(p_, base_seed);
            dummy_gen_.storeBaseDummies(std::move(base), base_seed);
            break;
        }
        default:
            // validateScreenTRexStrategy() in constructor prevents reaching here
            throw std::logic_error(
                "ScreenTRexSelector::runScreenExperiments: unreachable strategy branch.");
    }

    // Build ExperimentRunnerConfig: T=1, L=p, no warm-start, beta collection on
    auto cfg = buildRunnerConfig(
        p_,       // num_dummies = p  (L = p)
        1,        // T_stop = 1
        false,    // no warm start
        strategy
    );
    cfg.collect_screen_betas = true;

    // Run K experiments via ExperimentRunner
    er::ExperimentResults results = experiment_runner_->run(cfg);

    // Extract beta statistics from the extended result fields
    accumulated_beta_mat_.col(0) = results.beta_sums;
    collected_dummy_betas_       = std::move(results.dummy_betas);

    return results;
}


// ===================================================================================
// Selection Logic
// ===================================================================================

void ScreenTRexSelector::performOrdinarySelection(
    const Eigen::VectorXd&    Phi,
    ScreenTRexSelectionResult& result_out
) const {
    result_out.used_bootstrap   = false;
    result_out.confidence_level = 0.0;
    result_out.selected_var     = (Phi.array() > 0.5).cast<int>();
}


void ScreenTRexSelector::computeEstimatedFDR(
    ScreenTRexSelectionResult& result_out
) const {
    const double R = static_cast<double>(result_out.selected_var.sum());
    result_out.estimated_FDR = (R < 1.0) ? 0.0 : (1.0 / R);
}


// ===================================================================================
// Bootstrap Helpers
// ===================================================================================

double ScreenTRexSelector::probit(double p) const {
    if (p <= 0.0 || p >= 1.0) { return 0.0; }
    return std::sqrt(2.0) * boost::math::erf_inv(2.0 * p - 1.0);
}


void ScreenTRexSelector::performBootstrapSelection(
    const Eigen::VectorXd&    Phi,
    const Eigen::VectorXd&    beta_means,
    ScreenTRexSelectionResult& result
) {
    result.used_bootstrap = true;

    // Ordinary selection count as upper-bound constraint
    const long R_ordinary =
        static_cast<long>((Phi.array() > 0.5).count());

    const std::size_t n_dummies = collected_dummy_betas_.size();
    if (n_dummies == 0) {
        if (verbose_) {
            printProgress("No dummies selected during experiments. "
                          "Falling back to ordinary selection.");
        }
        result.used_bootstrap = false;
        result.selected_var   = (Phi.array() > 0.5).cast<int>();
        return;
    }

    // Original statistic t0 = mean of active dummy betas (for bias correction)
    const double t0 = std::accumulate(
        collected_dummy_betas_.begin(), collected_dummy_betas_.end(), 0.0)
        / static_cast<double>(n_dummies);

    // Bootstrap resampling of non-zero dummy betas
    std::vector<double> boot_means;
    boot_means.reserve(screen_ctrl_.R_boot);

    std::mt19937_64 gen(
        seed_ >= 0 ? static_cast<uint64_t>(seed_)
                   : static_cast<uint64_t>(std::random_device{}()));
    std::uniform_int_distribution<std::size_t> dist(0, n_dummies - 1);

    for (std::size_t r = 0; r < screen_ctrl_.R_boot; ++r) {
        double sum = 0.0;
        for (std::size_t i = 0; i < n_dummies; ++i) {
            sum += collected_dummy_betas_[dist(gen)];
        }
        boot_means.push_back(sum / static_cast<double>(n_dummies));
    }

    // Grid search: smallest gamma where selection count <= R_ordinary
    bool found             = false;
    double optimal_gamma   = 0.0;
    Eigen::VectorXi optimal_mask =
        Eigen::VectorXi::Zero(static_cast<Eigen::Index>(p_));

    for (double gamma = 0.0; gamma < 1.0; gamma += screen_ctrl_.ci_grid_step) {

        const auto ci = calculateBootstrapCI(
            boot_means, t0, gamma);
        const double lower   = ci.first;
        const double upper   = ci.second;

        long count = 0;
        Eigen::VectorXi current_mask =
            Eigen::VectorXi::Zero(static_cast<Eigen::Index>(p_));

        for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(p_); ++j) {
            if (beta_means(j) < lower || beta_means(j) > upper) {
                ++count;
                current_mask(j) = 1;
            }
        }

        if (count <= R_ordinary) {
            optimal_gamma = gamma;
            optimal_mask  = current_mask;
            found         = true;
            break;
        }
    }

    if (!found) {
        optimal_gamma = 1.0;
        optimal_mask.setZero();
    }

    result.selected_var     = optimal_mask;
    result.confidence_level = optimal_gamma;
    result.dummy_betas      = Eigen::Map<Eigen::VectorXd>(
        collected_dummy_betas_.data(),
        static_cast<Eigen::Index>(collected_dummy_betas_.size()));
}


std::pair<double, double> ScreenTRexSelector::calculateBootstrapCI(
    const std::vector<double>& boot_means,
    double t0,
    double confidence_level
) const {
    /*
        Bias-corrected normal CI (matches R's boot::boot.ci type="norm"):
            center = 2·t₀ − mean(t*)
            CI     = center ± z · se(t*)
    */
    const double n   = static_cast<double>(boot_means.size());
    const double sum = std::accumulate(
        boot_means.begin(), boot_means.end(), 0.0);
    const double boot_mean = sum / n;

    double sq_sum = 0.0;
    for (const double v : boot_means) {
        const double d = v - boot_mean;
        sq_sum += d * d;
    }
    const double sd = std::sqrt(sq_sum / (n - 1.0));

    // Bias-corrected center: 2·t₀ − mean(t*)
    const double center = 2.0 * t0 - boot_mean;

    const double alpha = (1.0 - confidence_level) / 2.0;
    const double z     = probit(1.0 - alpha);

    return {center - z * sd, center + z * sd};
}


// ===================================================================================
// Dependency-Aware Adjustment
// ===================================================================================

void ScreenTRexSelector::applyDependencyAwareAdjustment(Eigen::VectorXd& Phi) {

    // Use supplied correlation or estimate it
    double rho = screen_ctrl_.cor_coef;
    if (rho == tc::AUTO_ESTIMATE_CORRELATION) {
        switch (screen_ctrl_.trex_method) {
            case ScreenTRexMethod::TREX_DA_AR1:
                rho = estimateAR1Correlation();
                break;
            case ScreenTRexMethod::TREX_DA_EQUI:
                rho = estimateEquiCorrelation();
                break;
            case ScreenTRexMethod::TREX_DA_BLOCK_EQUI:
                rho = estimateBlockEquiCorrelation();
                break;
            default:
                throw std::runtime_error(
                    "ScreenTRexSelector: invalid method for DA adjustment.");
        }
    }
    screen_result_.estimated_correlation = rho;

    if (verbose_) {
        printProgress("DA: rho = " + std::to_string(rho));
    }

    // Compute adjustment deltas
    const Eigen::VectorXd DA_delta = [&]() -> Eigen::VectorXd {
        switch (screen_ctrl_.trex_method) {
            case ScreenTRexMethod::TREX_DA_AR1:
                return computeDA_AR1_Delta(Phi, rho);
            case ScreenTRexMethod::TREX_DA_EQUI:
                return computeDA_EQUI_Delta(Phi, rho);
            case ScreenTRexMethod::TREX_DA_BLOCK_EQUI:
                return computeDA_BLOCK_EQUI_Delta(Phi, rho);
            default:
                throw std::runtime_error(
                    "ScreenTRexSelector: unreachable DA variant.");
        }
    }();

    // Apply: Phi[j] /= DA_delta[j]
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(p_); ++j) {
        if (DA_delta(j) > eps_) {
            Phi(j) /= DA_delta(j);
        }
    }
}


// ===================================================================================
// Correlation Estimation
// ===================================================================================

double ScreenTRexSelector::estimateAR1Correlation() const {
    /*
        AR(1) Yule-Walker estimator per row, averaged over n rows:
            ρ̂ = Σ(x_t · x_{t-1}) / Σ(x_t²)
        Returns |mean ρ̂|.

        No row-mean centering: matches the R reference which fits
        arima(row, order=c(1,0,0), include.mean=FALSE) on column-
        centered (but not row-centered) data.
    */
    double total_rho = 0.0;

    for (std::size_t i = 0; i < n_; ++i) {
        double numerator   = 0.0;
        double denominator = 0.0;

        double x_prev = (*X_)(static_cast<Eigen::Index>(i), 0);
        denominator  += x_prev * x_prev;

        for (std::size_t j = 1; j < p_; ++j) {
            const double x_curr =
                (*X_)(static_cast<Eigen::Index>(i),
                      static_cast<Eigen::Index>(j));
            denominator += x_curr * x_curr;
            numerator   += x_curr * x_prev;
            x_prev       = x_curr;
        }

        if (denominator > eps_) {
            total_rho += numerator / denominator;
        }
    }

    return std::abs(total_rho / static_cast<double>(n_));
}


double ScreenTRexSelector::estimateEquiCorrelation() const {
    /*
        Let u_j = x_j / ‖x_j‖ be the unit-norm version of each (already
        centered) column. Then corr_jk = u_j · u_k, the full sum of the
        correlation matrix equals ‖Σ_j u_j‖², and its diagonal is exactly p
        (u_j · u_j = 1). This holds regardless of the scaling mode: under
        ScalingMode::L2 the columns are already unit-norm, while under
        ScalingMode::ZSCORE the common sqrt(n-1) scale is divided out here
        (matches TRexDASelector::estimateEquiCorrelation).
            → ρ̄ = (‖Σ_j u_j‖² − p) / (p(p−1))
        Sequential column access — cache-friendly for memory-mapped X.
    */
    Eigen::VectorXd row_sums =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_));

    for (std::size_t j = 0; j < p_; ++j) {
        const double norm_j = X_->col(static_cast<Eigen::Index>(j)).norm();
        if (norm_j > eps_) {
            row_sums.noalias() += X_->col(static_cast<Eigen::Index>(j)) / norm_j;
        }
    }

    const double sum_sq    = row_sums.squaredNorm();
    const double numerator = sum_sq - static_cast<double>(p_);
    const double denom =
        static_cast<double>(p_) * static_cast<double>(p_ - 1);

    if (denom < eps_) { return 0.0; }
    return numerator / denom;
}


// ===================================================================================
// DA Delta Computation
// ===================================================================================

Eigen::VectorXd ScreenTRexSelector::computeDA_AR1_Delta(
    const Eigen::VectorXd& Phi,
    double rho
) const {
    /*
        Window size κ = ⌈log(ρ_thr_DA) / log(ρ)⌉.
        For each j:
            DA_delta[j] = 2 − min_{k ∈ [j−κ, j+κ], k≠j} |Phi[j] − Phi[k]|
    */
    const double valid_rho = std::min(0.999, std::max(0.001, std::abs(rho)));
    const int kap = std::max(1, static_cast<int>(
        std::ceil(std::log(screen_ctrl_.rho_thr_DA) / std::log(valid_rho))));

    if (verbose_) {
        printProgress("DA-AR1: window size κ = " + std::to_string(kap));
    }

    Eigen::VectorXd DA_delta =
        Eigen::VectorXd::Ones(static_cast<Eigen::Index>(p_));

    for (std::size_t j = 0; j < p_; ++j) {
        double min_diff = 2.0;

        const long start = std::max(0L, static_cast<long>(j) - kap);
        const long end   = std::min(static_cast<long>(p_) - 1,
                                    static_cast<long>(j) + kap);

        for (long k = start; k <= end; ++k) {
            if (k != static_cast<long>(j)) {
                min_diff = std::min(min_diff,
                    std::abs(Phi(static_cast<Eigen::Index>(j))
                           - Phi(static_cast<Eigen::Index>(k))));
            }
        }

        DA_delta(static_cast<Eigen::Index>(j)) =
            2.0 - (min_diff > 1.5 ? 0.0 : min_diff);
    }

    return DA_delta;
}


Eigen::VectorXd ScreenTRexSelector::computeDA_EQUI_Delta(
    const Eigen::VectorXd& Phi,
    double rho
) const {
    /*
        Exact O(p²) global comparison.  For each j:
            DA_delta[j] = 2 − min_{k≠j} |Phi[j] − Phi[k]|
        Skipped when |ρ| ≤ ρ_thr_DA (no significant global correlation).
    */
    Eigen::VectorXd DA_delta =
        Eigen::VectorXd::Ones(static_cast<Eigen::Index>(p_));

    if (std::abs(rho) <= screen_ctrl_.rho_thr_DA) {
        if (verbose_) {
            printProgress("DA-EQUI: skipped (|ρ| = "
                + std::to_string(std::abs(rho)) + " ≤ ρ_thr_DA = "
                + std::to_string(screen_ctrl_.rho_thr_DA) + ").");
        }
        return DA_delta;
    }

    if (verbose_) {
        printProgress("DA-EQUI: computing exact O(p²) adjustment.");
    }

    for (std::size_t j = 0; j < p_; ++j) {
        Eigen::VectorXd diffs =
            (Phi.array() - Phi(static_cast<Eigen::Index>(j))).abs();
        diffs(static_cast<Eigen::Index>(j)) =
            std::numeric_limits<double>::max();

        const double min_diff = diffs.minCoeff();
        DA_delta(static_cast<Eigen::Index>(j)) =
            2.0 - (min_diff > 1.5 ? 0.0 : min_diff);
    }

    return DA_delta;
}

// ===================================================================================
// Block-Equicorrelation Estimation and Delta
// ===================================================================================

double ScreenTRexSelector::estimateBlockEquiCorrelation() const {
    /*
        Average within-block equi-correlation.
        Partition p columns into n_blocks contiguous blocks.
        For each block k of size b_k, accumulate UNIT-NORM columns
        u_j = x_j / ‖x_j‖ (scale-mode agnostic — see estimateEquiCorrelation):
            row_sums_k = sum of u_j for j in block k
            rho_k = (||row_sums_k||^2 - b_k) / (b_k * (b_k - 1))
        Return weighted average: sum(b_k * rho_k) / p.
    */
    const std::size_t n_blocks = screen_ctrl_.n_blocks;
    const std::size_t base_size = p_ / n_blocks;
    const std::size_t remainder = p_ % n_blocks;

    double weighted_rho = 0.0;
    std::size_t col_start = 0;

    for (std::size_t k = 0; k < n_blocks; ++k) {
        const std::size_t bk = base_size + (k < remainder ? 1 : 0);
        if (bk < 2) {
            col_start += bk;
            continue;
        }

        Eigen::VectorXd row_sums =
            Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_));
        for (std::size_t j = col_start; j < col_start + bk; ++j) {
            const double norm_j = X_->col(static_cast<Eigen::Index>(j)).norm();
            if (norm_j > eps_) {
                row_sums.noalias() += X_->col(static_cast<Eigen::Index>(j)) / norm_j;
            }
        }

        const double sum_sq = row_sums.squaredNorm();
        const double bk_d   = static_cast<double>(bk);
        const double denom  = bk_d * (bk_d - 1.0);
        const double rho_k  = (sum_sq - bk_d) / denom;

        weighted_rho += bk_d * rho_k;
        col_start += bk;
    }

    return weighted_rho / static_cast<double>(p_);
}


Eigen::VectorXd ScreenTRexSelector::computeDA_BLOCK_EQUI_Delta(
    const Eigen::VectorXd& Phi,
    double rho
) const {
    /*
        Within-block neighbors only.  For each variable j in block k:
            DA_delta[j] = 2 - min_{m in block(k), m != j} |Phi[j] - Phi[m]|
        Skipped when |rho| <= rho_thr_DA.
    */
    Eigen::VectorXd DA_delta =
        Eigen::VectorXd::Ones(static_cast<Eigen::Index>(p_));

    if (std::abs(rho) <= screen_ctrl_.rho_thr_DA) {
        if (verbose_) {
            printProgress("DA-BLOCK-EQUI: skipped (|rho| = "
                + std::to_string(std::abs(rho)) + " <= rho_thr_DA = "
                + std::to_string(screen_ctrl_.rho_thr_DA) + ").");
        }
        return DA_delta;
    }

    const std::size_t n_blocks = screen_ctrl_.n_blocks;
    const std::size_t base_size = p_ / n_blocks;
    const std::size_t remainder = p_ % n_blocks;

    if (verbose_) {
        printProgress("DA-BLOCK-EQUI: computing within-block adjustment ("
            + std::to_string(n_blocks) + " blocks).");
    }

    std::size_t col_start = 0;
    for (std::size_t k = 0; k < n_blocks; ++k) {
        const std::size_t bk = base_size + (k < remainder ? 1 : 0);

        for (std::size_t j = col_start; j < col_start + bk; ++j) {
            double min_diff = 2.0;
            const auto j_idx = static_cast<Eigen::Index>(j);

            for (std::size_t m = col_start; m < col_start + bk; ++m) {
                if (m != j) {
                    const double diff = std::abs(
                        Phi(j_idx) - Phi(static_cast<Eigen::Index>(m)));
                    min_diff = std::min(min_diff, diff);
                }
            }

            DA_delta(j_idx) = 2.0 - (min_diff > 1.5 ? 0.0 : min_diff);
        }

        col_start += bk;
    }

    return DA_delta;
}


// ===================================================================================
} /* End of namespace trex::trex_selector_methods::trex_screening */
// ===================================================================================
