// ===================================================================================
// trex_da.cpp
// ===================================================================================
/**
 * @file trex_da.cpp
 *
 * @brief Implementation of the Dependency-Aware T-Rex (DA-TRex) Selector.
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// DA-TRex header
#include <trex_selector_methods/trex_da/trex_da.hpp>

// ===================================================================================

namespace trex::trex_selector_methods::trex_da {

// Namespace aliases
namespace tc  = trex::trex_selector_methods::trex_core;
namespace dn  = trex::trex_selector_methods::utils::data_normalizer;
namespace er  = trex::trex_selector_methods::utils::experiment_runner;
namespace hac = trex::ml_methods::clustering::hierarchical::agglomerative;


// ===================================================================================
// Constructor
// ===================================================================================

TRexDASelector::TRexDASelector(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::VectorXd>& y,
    double tFDR,
    TRexDAControlParameter da_control,
    tc::TRexControlParameter trex_control,
    int seed,
    bool verbose
) :
    TRexSelector(X, y, tFDR, trex_control, seed, verbose),
    da_ctrl_(std::move(da_control))
{
    // Resolve hc_grid_length default
    if (da_ctrl_.hc_grid_length == 0) {
        da_ctrl_.hc_grid_length = std::min<std::size_t>(20, p_);
    }

    // Prior groups override method
    if (!da_ctrl_.prior_groups.empty()) {
        da_ctrl_.method = DAMethod::PRIOR_GROUPS;
    }

    validateDAParameters();
}


// ===================================================================================
// Validation
// ===================================================================================

void TRexDASelector::validateDAParameters() const {

    if (!da_ctrl_.prior_groups.empty()) {
        for (const auto& level : da_ctrl_.prior_groups) {
            if (level.size() != p_) {
                throw std::invalid_argument(
                    "Each prior_groups level must have length p = " +
                    std::to_string(p_) +
                    ". Got: " + std::to_string(level.size()));
            }
        }
        if (!da_ctrl_.rho_grid_labels.empty() &&
            da_ctrl_.rho_grid_labels.size() != da_ctrl_.prior_groups.size()) {
            throw std::invalid_argument(
                "rho_grid_labels must have the same length as prior_groups.");
        }
    }

    if (da_ctrl_.rho_thr_DA < 0.0 || da_ctrl_.rho_thr_DA > 1.0) {
        throw std::invalid_argument(
            "rho_thr_DA must be in [0, 1]. Got: " +
            std::to_string(da_ctrl_.rho_thr_DA));
    }

    // Check linkage method is one of the supported ones
    if (da_ctrl_.method == DAMethod::BT) {
        auto lm = da_ctrl_.hc_linkage;
        if (lm != hac::LinkageMethod::Single &&
            lm != hac::LinkageMethod::Complete &&
            lm != hac::LinkageMethod::Average &&
            lm != hac::LinkageMethod::WPGMA) {
            throw std::invalid_argument(
                "Unsupported linkage method for DA-BT. "
                "Supported: Single, Complete, Average, WPGMA.");
        }
    }
}


// ===================================================================================
// Main DA-TRex algorithm: select()
// ===================================================================================

TRexDASelector::SelectionResult TRexDASelector::select() {

    // 1. Voting grid and optimization point
    voting_grid_ = createVotingGrid();
    const std::size_t v_len = voting_grid_.size();
    setPercentOptPoint(v_len, trex_ctrl_.opt_threshold);

    // 2. Build neighbourhood / correlation structure
    if (verbose_) { printProgress("DA: Building dependency structure..."); }
    setupDA();

    if (verbose_) {
        std::string method_str;
        switch (da_ctrl_.method) {
            case DAMethod::AR1:          method_str = "AR1";          break;
            case DAMethod::EQUI:         method_str = "EQUI";         break;
            case DAMethod::BT:           method_str = "BT";           break;
            case DAMethod::NN:           method_str = "NN";           break;
            case DAMethod::PRIOR_GROUPS: method_str = "PRIOR_GROUPS"; break;
        }
        printProgress("DA: Method = " + method_str +
                      (da_setup_.use_BT_style
                          ? " (BT-style, rho_grid_len = " +
                            std::to_string(da_setup_.rho_grid_len) + ")"
                          : ""));
    }

    // 3. L-loop calibration (DA's evaluateStep override)
    if (verbose_) { printProgress("L-loop: Calibrating number of dummies..."); }
    Eigen::VectorXd FDP_hat;
    er::ExperimentResults exp_results;
    runLLoop(FDP_hat, exp_results);

    if (verbose_) {
        printProgress("L-loop: converged with " +
                      std::to_string(num_dummies_) + " dummies.\n");
    }

    // 4. T-loop calibration (base driver, DA's evaluateStep + accumulator
    //    overrides; capacity-doubling, BT side state, parallel experiments,
    //    memory mapping all inherited from base).
    runTLoop(FDP_hat, exp_results);

    // 5. Assemble results, cleanup, denormalise, return base slice.
    return assembleDAResult();
}


// ===================================================================================
// Hook override: evaluateStep — experiment + DA deflation + FDP
// ===================================================================================

TRexDASelector::StepView TRexDASelector::evaluateStep(
    std::size_t num_dummies,
    std::size_t T_stop,
    bool use_warm_start,
    er::ExperimentStrategy strategy,
    std::size_t seed_factor,
    std::size_t existing_on_disk)
{
    auto cfg = buildRunnerConfig(num_dummies, T_stop, use_warm_start,
                                 strategy, seed_factor, existing_on_disk);

    StepView view;
    view.exp_results = experiment_runner_->run(cfg);

    // DA deflation
    DACorrectionResult da_corr = daCorrect(view.exp_results.phi_T_mat,
                                           view.exp_results.Phi,
                                           T_stop);

    if (da_setup_.use_BT_style) {
        auto [phi_prime_bt,
              fdp_hat_bt] = computeFDP_BT(da_corr, num_dummies);

        Phi_BT_last_       = std::move(da_corr.Phi_BT);
        Phi_prime_BT_last_ = std::move(phi_prime_bt);
        FDP_hat_BT_last_   = std::move(fdp_hat_bt);

        const auto opt_BT = static_cast<Eigen::Index>(da_setup_.opt_point_BT);
        view.Phi       = Phi_BT_last_.col(opt_BT);
        view.Phi_prime = Phi_prime_BT_last_.col(opt_BT);
        view.FDP_hat   = FDP_hat_BT_last_.col(opt_BT);
    } else {
        view.Phi       = da_corr.Phi;
        view.Phi_prime = computePhiPrime(da_corr.phi_T_mat,
                                         da_corr.Phi,
                                         num_dummies);
        view.FDP_hat   = computeFDPHat(voting_grid_,
                                       view.Phi,
                                       view.Phi_prime);
    }

    Phi_prime_last_ = view.Phi_prime;
    return view;
}


// ===================================================================================
// Overrides: T-loop accumulators (extend base with BT side accumulators)
// ===================================================================================

void TRexDASelector::initTAccumulators(std::size_t capacity, std::size_t v_len) {
    TRexSelector::initTAccumulators(capacity, v_len);

    if (da_setup_.use_BT_style) {
        const auto cap_idx = static_cast<Eigen::Index>(capacity);
        const auto p_idx   = static_cast<Eigen::Index>(p_);
        const auto v_idx   = static_cast<Eigen::Index>(v_len);

        phi_acc_bt_.assign(da_setup_.rho_grid_len,
                           Eigen::MatrixXd::Zero(cap_idx, p_idx));
        fdp_acc_bt_.assign(da_setup_.rho_grid_len,
                           Eigen::MatrixXd::Zero(cap_idx, v_idx));
    } else {
        phi_acc_bt_.clear();
        fdp_acc_bt_.clear();
    }
}


void TRexDASelector::growTAccumulators(std::size_t old_capacity,
                                       std::size_t new_capacity) {
    TRexSelector::growTAccumulators(old_capacity, new_capacity);

    if (!da_setup_.use_BT_style) return;

    const auto old_cap = static_cast<Eigen::Index>(old_capacity);
    const auto new_cap = static_cast<Eigen::Index>(new_capacity);
    for (std::size_t r = 0; r < da_setup_.rho_grid_len; ++r) {
        phi_acc_bt_[r].conservativeResize(new_cap, Eigen::NoChange);
        fdp_acc_bt_[r].conservativeResize(new_cap, Eigen::NoChange);
        phi_acc_bt_[r].bottomRows(new_cap - old_cap).setZero();
        fdp_acc_bt_[r].bottomRows(new_cap - old_cap).setZero();
    }
}


void TRexDASelector::appendTRow(Eigen::Index row_idx, const StepView& view) {
    TRexSelector::appendTRow(row_idx, view);

    if (!da_setup_.use_BT_style) return;

    for (std::size_t r = 0; r < da_setup_.rho_grid_len; ++r) {
        const auto rc = static_cast<Eigen::Index>(r);
        phi_acc_bt_[r].row(row_idx) = Phi_BT_last_.col(rc).transpose();
        fdp_acc_bt_[r].row(row_idx) = FDP_hat_BT_last_.col(rc).transpose();
    }
}


void TRexDASelector::trimTAccumulators(Eigen::Index n_rows) {
    TRexSelector::trimTAccumulators(n_rows);

    if (!da_setup_.use_BT_style) return;

    for (std::size_t r = 0; r < da_setup_.rho_grid_len; ++r) {
        phi_acc_bt_[r].conservativeResize(n_rows, Eigen::NoChange);
        fdp_acc_bt_[r].conservativeResize(n_rows, Eigen::NoChange);
    }
}


// ===================================================================================
// Result assembly: populate da_result_, cleanup, denormalise
// ===================================================================================

TRexDASelector::SelectionResult TRexDASelector::assembleDAResult() {

    const bool use_BT = da_setup_.use_BT_style;

    // 1. Final Phi_prime
    if (use_BT) {
        Phi_prime_ = Phi_prime_BT_last_.col(
            static_cast<Eigen::Index>(da_setup_.opt_point_BT));
    } else {
        Phi_prime_ = Phi_prime_last_;
    }

    // 2. Variable selection
    if (use_BT) {
        da_result_ = selectVariables_BT(
            fdp_acc_bt_,
            phi_acc_bt_,
            voting_grid_,
            da_setup_.rho_grid,
            T_stop_);
    } else {
        auto base_result =
            selectedVariables(FDP_hat_mat_,
                              Phi_mat_,
                              voting_grid_,
                              T_stop_);

        da_result_.selected_var = base_result.selected_var;
        da_result_.v_thresh     = base_result.v_thresh;
        da_result_.R_mat        = base_result.R_mat;
        da_result_.rho_thresh   = tc::AUTO_ESTIMATE_CORRELATION;
    }

    // 3. Common fields
    da_result_.T_stop         = T_stop_;
    da_result_.num_dummies    = num_dummies_;
    da_result_.dummy_factor_L = dummy_multiplier_LL_;
    da_result_.Phi_prime      = Phi_prime_;
    da_result_.voting_grid    = voting_grid_;
    da_result_.method         = da_ctrl_.method;
    da_result_.cor_coef       = da_setup_.cor_coef;
    da_result_.rho_grid       = da_setup_.rho_grid;

    if (use_BT) {
        da_result_.FDP_hat_mat      = Eigen::MatrixXd();  // Not used for BT
        da_result_.Phi_mat          = Eigen::MatrixXd();
        da_result_.FDP_hat_array_BT = fdp_acc_bt_;
        da_result_.Phi_array_BT     = phi_acc_bt_;
    } else {
        da_result_.FDP_hat_mat = FDP_hat_mat_;
        da_result_.Phi_mat     = Phi_mat_;
    }

    storeResults(da_result_);

    if (verbose_) {
        printProgress("DA: Selection complete. Selected " +
                      std::to_string(getNumSelected()) + " variables.");
    }

    // 4. Cleanup
    warm_start_mgr_.cleanup();
    if (memmap_mgr_) { memmap_mgr_->cleanup(); }

    if (verbose_) { printProgress("Denormalizing X to original scale.\n"); }
    if (X_is_normalized_) {
        dn::denormalizeX(*X_, norm_params_);
        X_is_normalized_ = false;
    }

    // 5. Return base-sliced result
    SelectionResult base_result;
    base_result.selected_var   = da_result_.selected_var;
    base_result.v_thresh       = da_result_.v_thresh;
    base_result.R_mat          = da_result_.R_mat;
    base_result.T_stop         = da_result_.T_stop;
    base_result.num_dummies    = da_result_.num_dummies;
    base_result.dummy_factor_L = da_result_.dummy_factor_L;
    base_result.FDP_hat_mat    = da_result_.FDP_hat_mat;
    base_result.Phi_mat        = da_result_.Phi_mat;
    base_result.Phi_prime      = da_result_.Phi_prime;
    base_result.voting_grid    = da_result_.voting_grid;
    return base_result;
}



// ===================================================================================
// DA Setup — Dispatcher
// ===================================================================================

void TRexDASelector::setupDA() {
    if (!da_ctrl_.prior_groups.empty()) {
        setupDA_PriorGroups();
    } else {
        switch (da_ctrl_.method) {
            case DAMethod::BT:           setupDA_BT();   break;
            case DAMethod::NN:           setupDA_NN();   break;
            case DAMethod::AR1:          setupDA_AR1();  break;
            case DAMethod::EQUI:         setupDA_EQUI(); break;
            case DAMethod::PRIOR_GROUPS:
                throw std::logic_error("PRIOR_GROUPS method requires prior_groups to be set.");
        }
    }
}


// ===================================================================================
// DA Setup 1: Prior Groups
// ===================================================================================

void TRexDASelector::setupDA_PriorGroups() {
    const auto& groups = da_ctrl_.prior_groups;
    const std::size_t rho_grid_len = groups.size();
    const auto p = static_cast<Eigen::Index>(p_);

    da_setup_.use_BT_style = true;
    da_setup_.rho_grid_len = rho_grid_len;
    // R (1-indexed): round(0.75 * L) -> element L*0.75.
    // C++ (0-indexed): subtract 1 to match.
    {
        std::size_t raw = static_cast<std::size_t>(
            std::round(0.75 * static_cast<double>(rho_grid_len)));
        da_setup_.opt_point_BT = (raw > 0) ? raw - 1 : 0;
    }
    // Clamp to valid range
    if (da_setup_.opt_point_BT >= rho_grid_len) {
        da_setup_.opt_point_BT = rho_grid_len - 1;
    }
    da_setup_.cor_coef = da_ctrl_.cor_coef;
    da_setup_.kap = 0;

    // Build rho_grid
    if (!da_ctrl_.rho_grid_labels.empty()) {
        da_setup_.rho_grid.resize(static_cast<Eigen::Index>(rho_grid_len));
        for (std::size_t i = 0; i < rho_grid_len; ++i) {
            da_setup_.rho_grid(static_cast<Eigen::Index>(i)) = da_ctrl_.rho_grid_labels[i];
        }
    } else {
        da_setup_.rho_grid = Eigen::VectorXd::LinSpaced(
            static_cast<Eigen::Index>(rho_grid_len), 1.0,
            static_cast<double>(rho_grid_len));
    }

    // Build gr_j_list
    da_setup_.gr_j_list.resize(p_);
    for (Eigen::Index j = 0; j < p; ++j) {
        da_setup_.gr_j_list[j].resize(rho_grid_len);
        for (std::size_t r = 0; r < rho_grid_len; ++r) {
            Eigen::Index label_j = groups[r][static_cast<std::size_t>(j)];
            for (Eigen::Index k = 0; k < p; ++k) {
                if (k != j && groups[r][static_cast<std::size_t>(k)] == label_j) {
                    da_setup_.gr_j_list[j][r].push_back(k);
                }
            }
        }
    }
}


// ===================================================================================
// DA Setup 2: BT (dendrogram-based)
// ===================================================================================

void TRexDASelector::setupDA_BT() {
    const auto p = static_cast<Eigen::Index>(p_);
    const std::size_t grid_len = da_ctrl_.hc_grid_length;

    // Run hierarchical clustering using correlation distance
    auto merges = runClustering();

    // Build rho_grid from dendrogram heights
    // heights are sorted ascending (merge distances in [0, 1] for correlation)
    // rho_grid_full = { 1 - heights[p-2], ..., 1 - heights[0], 1.0 }
    //               = correlation levels from low to high
    std::vector<double> rho_grid_full;
    rho_grid_full.reserve(static_cast<std::size_t>(p));
    for (auto it = merges.rbegin(); it != merges.rend(); ++it) {
        rho_grid_full.push_back(1.0 - it->distance);
    }
    rho_grid_full.push_back(1.0);
    // rho_grid_full now has p elements, sorted ascending

    // Subsample to grid_len evenly spaced indices.
    // Guard against grid_len == 1 (would divide by zero below): use the
    // highest-correlation point as a single-element grid.
    da_setup_.rho_grid.resize(static_cast<Eigen::Index>(grid_len));
    if (grid_len == 1) {
        da_setup_.rho_grid(0) = rho_grid_full.back();
    } else {
        for (std::size_t i = 0; i < grid_len; ++i) {
            double frac = static_cast<double>(i) * static_cast<double>(p_ - 1)
                        / static_cast<double>(grid_len - 1);
            std::size_t idx = static_cast<std::size_t>(std::round(frac));
            if (idx >= rho_grid_full.size()) idx = rho_grid_full.size() - 1;
            da_setup_.rho_grid(static_cast<Eigen::Index>(i)) = rho_grid_full[idx];
        }
    }

    da_setup_.use_BT_style = true;
    da_setup_.rho_grid_len = grid_len;
    // R (1-indexed): round(0.75 * L) -> element L*0.75.
    // C++ (0-indexed): subtract 1 to match.
    {
        std::size_t raw = static_cast<std::size_t>(
            std::round(0.75 * static_cast<double>(grid_len)));
        da_setup_.opt_point_BT = (raw > 0) ? raw - 1 : 0;
    }
    if (da_setup_.opt_point_BT >= grid_len) {
        da_setup_.opt_point_BT = grid_len - 1;
    }
    da_setup_.cor_coef = da_ctrl_.cor_coef;
    da_setup_.kap = 0;

    // For each rho level, cut dendrogram and build group-mate lists
    da_setup_.gr_j_list.resize(p_);
    for (Eigen::Index j = 0; j < p; ++j) {
        da_setup_.gr_j_list[j].resize(grid_len);
    }

    for (std::size_t r = 0; r < grid_len; ++r) {
        double cut_height = 1.0 - da_setup_.rho_grid(static_cast<Eigen::Index>(r));
        auto labels = hac::DendrogramUtils::cut_tree_by_height(
            merges, p, cut_height);
        auto groups = hac::DendrogramUtils::group_indices_by_label(labels);

        for (Eigen::Index j = 0; j < p; ++j) {
            Eigen::Index label_j = labels[static_cast<std::size_t>(j)];
            const auto& cluster = groups[static_cast<std::size_t>(label_j)];
            for (auto idx : cluster) {
                if (idx != j) {
                    da_setup_.gr_j_list[j][r].push_back(idx);
                }
            }
        }
    }

    if (verbose_) {
        printProgress("DA-BT: Clustering complete. Grid length = " +
                      std::to_string(grid_len) + ", rho_grid in [" +
                      std::to_string(da_setup_.rho_grid(0)) + ", " +
                      std::to_string(da_setup_.rho_grid(
                          static_cast<Eigen::Index>(grid_len - 1))) + "]");
    }
}


// ===================================================================================
// DA Setup 3: NN (nearest-neighbour threshold sweep)
// ===================================================================================

void TRexDASelector::setupDA_NN() {
    const auto p = static_cast<Eigen::Index>(p_);
    const std::size_t grid_len = da_ctrl_.hc_grid_length;

    // Build rho_grid = linspace(0, 1, grid_len)
    da_setup_.rho_grid = Eigen::VectorXd::LinSpaced(
        static_cast<Eigen::Index>(grid_len), 0.0, 1.0);

    da_setup_.use_BT_style = true;
    da_setup_.rho_grid_len = grid_len;
    // R (1-indexed): round(0.75 * L) -> element L*0.75.
    // C++ (0-indexed): subtract 1 to match.
    {
        std::size_t raw = static_cast<std::size_t>(
            std::round(0.75 * static_cast<double>(grid_len)));
        da_setup_.opt_point_BT = (raw > 0) ? raw - 1 : 0;
    }
    if (da_setup_.opt_point_BT >= grid_len) {
        da_setup_.opt_point_BT = grid_len - 1;
    }
    da_setup_.cor_coef = da_ctrl_.cor_coef;
    da_setup_.kap = 0;

    // Build gr_j_list pair-wise: for each ordered pair (j, k) with j < k,
    // compute the true Pearson correlation |<x_j, x_k>| / (||x_j|| ||x_k||),
    // then for every rho level r where |cor| >= rho_grid(r) record the
    // symmetric neighbour relation.
    //
    // This avoids materialising the full p x p correlation matrix
    // (X^T * X), which would dominate memory at large p.
    da_setup_.gr_j_list.assign(
        p_, std::vector<std::vector<Eigen::Index>>(grid_len));

    // Precompute per-column inverse norms so the pairwise correlation is a true
    // Pearson correlation independent of the scaling mode. X is already centered
    // by the base normalizer; under ScalingMode::L2 the norms are ~1 (this
    // reduces to the raw dot product), while under ScalingMode::ZSCORE the
    // columns carry a common sqrt(n-1) scale that must be divided out. A
    // near-zero norm (constant column) yields inv_norm = 0 so it has no
    // neighbours at any positive rho threshold.
    const Eigen::VectorXd col_norms = X_->colwise().norm();
    Eigen::VectorXd inv_norms(p);
    for (Eigen::Index j = 0; j < p; ++j) {
        inv_norms(j) = (col_norms(j) > eps_) ? (1.0 / col_norms(j)) : 0.0;
    }

    for (Eigen::Index j = 0; j < p; ++j) {
        for (Eigen::Index k = j + 1; k < p; ++k) {
            const double abs_cor =
                std::abs(X_->col(j).dot(X_->col(k))) *
                           inv_norms(j) *
                           inv_norms(k);
            for (std::size_t r = 0; r < grid_len; ++r) {
                const double rho_threshold =
                    da_setup_.rho_grid(static_cast<Eigen::Index>(r));
                if (abs_cor >= rho_threshold) {
                    da_setup_.gr_j_list[j][r].push_back(k);
                    da_setup_.gr_j_list[k][r].push_back(j);
                }
            }
        }
    }

    if (verbose_) {
        printProgress(
            "DA-NN: Pairwise correlations computed. Grid length = " +
            std::to_string(grid_len)
        );
    }
}


// ===================================================================================
// DA Setup 4: AR(1)
// ===================================================================================

void TRexDASelector::setupDA_AR1() {
    da_setup_.use_BT_style = false;
    da_setup_.rho_grid_len = 0;
    da_setup_.kap = 0;

    // Estimate AR(1) coefficient if not supplied
    if (da_ctrl_.cor_coef == tc::AUTO_ESTIMATE_CORRELATION) {
        da_setup_.cor_coef = estimateAR1Correlation();
    } else {
        da_setup_.cor_coef = da_ctrl_.cor_coef;
    }

    // Compute window half-width: kap = ceil(log(rho_thr_DA) / log(cor_coef))
    if (da_setup_.cor_coef > 0.0 && da_setup_.cor_coef < 1.0 && da_ctrl_.rho_thr_DA > 0.0) {
        da_setup_.kap = static_cast<std::size_t>(
            std::ceil(std::log(da_ctrl_.rho_thr_DA) / std::log(da_setup_.cor_coef)));
    } else {
        da_setup_.kap = p_;  // fallback: full window
    }

    if (verbose_) {
        printProgress(
            "DA-AR1: cor_coef = " + std::to_string(da_setup_.cor_coef) +
            ", kap = " + std::to_string(da_setup_.kap)
        );
    }
}


// ===================================================================================
// DA Setup 5: Equicorrelation
// ===================================================================================

void TRexDASelector::setupDA_EQUI() {
    da_setup_.use_BT_style = false;
    da_setup_.rho_grid_len = 0;
    da_setup_.kap = 0;

    // Estimate equicorrelation if not supplied
    if (da_ctrl_.cor_coef == tc::AUTO_ESTIMATE_CORRELATION) {
        da_setup_.cor_coef = estimateEquiCorrelation();
    } else {
        da_setup_.cor_coef = da_ctrl_.cor_coef;
    }

    if (verbose_) {
        printProgress(
            "DA-EQUI: cor_coef = " + std::to_string(da_setup_.cor_coef) +
            " (threshold = " + std::to_string(da_ctrl_.rho_thr_DA) + ")"
        );
    }
}


// ===================================================================================
// Correlation Estimation — AR(1)
// ===================================================================================

double TRexDASelector::estimateAR1Correlation() const {
    // Row-wise Yule-Walker:
    // --------------------
    // ρ̂_i = Σ_{t=0}^{p-2} x_{i, t} * x_{i, t + 1} / Σ_{t=0}^{p-1} x_{i, t}^2
    // (X is already centered)
    //
    // A note on the Scaling-mode: this estimator is only *approximately* invariant
    // to the per-column scaling. Each column t carries its own scale factor s_t,
    // so the ratio above cancels it exactly only when the factors are (near) constant
    // across the lag — which holds for the stationary AR(1) design this method
    // targets (all columns share ~equal variance/norm). It is therefore left
    // unnormalized on purpose; do NOT "fix" it by dividing per-column without
    // re-validating the AR1 FDR/TPR numbers.

    const auto n = static_cast<Eigen::Index>(n_);
    const auto p = static_cast<Eigen::Index>(p_);

    double sum_rho = 0.0;
    for (Eigen::Index i = 0; i < n; ++i) {
        double numer = 0.0;
        double denom = 0.0;
        for (Eigen::Index t = 0; t < p; ++t) {
            double x_t = (*X_)(i, t);
            denom += x_t * x_t;
            if (t < p - 1) {
                numer += x_t * (*X_)(i, t + 1);
            }
        }
        if (denom > eps_) {
            sum_rho += numer / denom;
        }
    }

    return std::abs(sum_rho / static_cast<double>(n));
}


// ===================================================================================
// Correlation Estimation — Equicorrelation
// ===================================================================================

double TRexDASelector::estimateEquiCorrelation() const {
    // Mean of the off-diagonal entries of the Pearson correlation matrix.
    //
    // Let u_j = x_j / ||x_j|| be the unit-norm version of each (already
    // centered) column. Then corr_jk = u_j · u_k, the full sum of the
    // correlation matrix equals || Σ_j u_j ||², and its diagonal is exactly p
    // (u_j · u_j = 1). This holds regardless of the scaling mode: under
    // ScalingMode::L2 the columns are already unit-norm, while under
    // ScalingMode::ZSCORE the common sqrt(n-1) scale is divided out here.

    const auto p = static_cast<Eigen::Index>(p_);

    // Accumulate the row sums of the unit-norm columns: Σ_j x_j / ||x_j||.
    Eigen::VectorXd row_sums = Eigen::VectorXd::Zero(X_->rows());
    for (Eigen::Index j = 0; j < p; ++j) {
        const double norm_j = X_->col(j).norm();
        if (norm_j > eps_) {
            row_sums += X_->col(j) / norm_j;
        }
    }
    const double sum_all = row_sums.squaredNorm();

    // Mean off-diagonal = (Σ_all - diag_sum) / (p * (p - 1)), diag_sum = p.
    return (sum_all - static_cast<double>(p)) /
           (static_cast<double>(p) * (static_cast<double>(p) - 1.0));
}


// ===================================================================================
// Clustering Dispatch
// ===================================================================================

std::vector<hac::MergeStep> TRexDASelector::runClustering() const {
    using MapType = Eigen::Map<Eigen::MatrixXd>;
    using CorrDist = hac::DistancePolicy<MapType, hac::DistanceMetric::Correlation>;

    switch (da_ctrl_.hc_linkage) {
        case hac::LinkageMethod::Single:
            return hac::AgglomerativeClustering::cluster<
                MapType, CorrDist, hac::LinkageMethod::Single>(*X_,
                    false, verbose_);

        case hac::LinkageMethod::Complete:
            return hac::AgglomerativeClustering::cluster<
                MapType, CorrDist, hac::LinkageMethod::Complete>(*X_,
                    false, verbose_);

        case hac::LinkageMethod::Average:
            return hac::AgglomerativeClustering::cluster<
                MapType, CorrDist, hac::LinkageMethod::Average>(*X_,
                    false, verbose_);

        case hac::LinkageMethod::WPGMA:
            return hac::AgglomerativeClustering::cluster<
                MapType, CorrDist, hac::LinkageMethod::WPGMA>(*X_,
                    false, verbose_);

        default:
            throw std::invalid_argument("Unsupported linkage method for DA-BT.");
    }
}


// ===================================================================================
// DA Correction
// ===================================================================================

DACorrectionResult TRexDASelector::daCorrect(
    const Eigen::MatrixXd& phi_T_mat,
    const Eigen::VectorXd& Phi,
    std::size_t T_stop) const
{
    DACorrectionResult result;
    result.phi_T_mat = phi_T_mat;  // copy (will be modified for AR1/EQUI)
    result.Phi = Phi;              // copy

    const auto p = static_cast<Eigen::Index>(p_);
    const auto T = static_cast<Eigen::Index>(T_stop);
    const std::size_t kap = da_setup_.kap;


    // ── AR1 ─────────────────────────────────────────────────────────────────
    if (da_ctrl_.method == DAMethod::AR1) {
        Eigen::MatrixXd delta = Eigen::MatrixXd::Constant(p, T, 2.0);

        for (Eigen::Index t = 0; t < T; ++t) {
            for (Eigen::Index j = 0; j < p; ++j) {
                double min_diff = 2.0;
                // Window: [j-kap, j+kap] excluding j
                Eigen::Index lo = std::max(static_cast<Eigen::Index>(0),
                                           j - static_cast<Eigen::Index>(kap));
                Eigen::Index hi = std::min(p - 1, j + static_cast<Eigen::Index>(kap));
                for (Eigen::Index k = lo; k <= hi; ++k) {
                    if (k != j) {
                        double diff = std::abs(result.phi_T_mat(j, t) -
                                                  result.phi_T_mat(k, t));
                        if (diff < min_diff) { min_diff = diff; }
                    }
                }
                delta(j, t) = 2.0 - min_diff;
            }
        }

        // Deflate
        for (Eigen::Index t = 0; t < T; ++t) {
            for (Eigen::Index j = 0; j < p; ++j) {
                if (delta(j, t) > eps_) {
                    result.phi_T_mat(j, t) /= delta(j, t);
                }
            }
        }
        for (Eigen::Index j = 0; j < p; ++j) {
            if (delta(j, T - 1) > eps_) {
                result.Phi(j) /= delta(j, T - 1);
            }
        }
    }


    // ── EQUI ────────────────────────────────────────────────────────────────
    else if (da_ctrl_.method == DAMethod::EQUI &&
             std::abs(da_setup_.cor_coef) > da_ctrl_.rho_thr_DA) {
        Eigen::MatrixXd delta = Eigen::MatrixXd::Constant(p, T, 2.0);

        for (Eigen::Index t = 0; t < T; ++t) {
            for (Eigen::Index j = 0; j < p; ++j) {
                double min_diff = 2.0;
                for (Eigen::Index k = 0; k < p; ++k) {
                    if (k != j) {
                        double diff = std::abs(result.phi_T_mat(j, t) -
                                        result.phi_T_mat(k, t));
                        if (diff < min_diff) { min_diff = diff; }
                    }
                }
                delta(j, t) = 2.0 - min_diff;
            }
        }

        for (Eigen::Index t = 0; t < T; ++t) {
            for (Eigen::Index j = 0; j < p; ++j) {
                if (delta(j, t) > eps_) {
                    result.phi_T_mat(j, t) /= delta(j, t);
                }
            }
        }
        for (Eigen::Index j = 0; j < p; ++j) {
            if (delta(j, T - 1) > eps_) {
                result.Phi(j) /= delta(j, T - 1);
            }
        }
    }


    // ── BT-style (BT, NN, PRIOR_GROUPS) ─────────────────────────────────────
    if (da_setup_.use_BT_style) {
        const std::size_t rho_grid_len = da_setup_.rho_grid_len;

        // DA_delta_mat_BT: p x rho_grid_len
        Eigen::MatrixXd delta_BT = Eigen::MatrixXd::Constant(
            p, static_cast<Eigen::Index>(rho_grid_len), 2.0);

        for (Eigen::Index j = 0; j < p; ++j) {
            for (std::size_t r = 0; r < rho_grid_len; ++r) {
                const auto& group_mates = da_setup_.gr_j_list[j][r];
                if (!group_mates.empty()) {
                    double min_diff = 2.0;
                    for (auto k : group_mates) {
                        double diff = std::abs(Phi(j) - Phi(k));
                        if (diff < min_diff) { min_diff = diff; }
                    }
                    delta_BT(j, static_cast<Eigen::Index>(r)) = 2.0 - min_diff;
                }
                // else: delta stays at 2.0 (no deflation)
            }
        }

        // phi_T_array_BT: for each rho level, phi_T_mat / delta_col (broadcast across T)
        result.phi_T_array_BT.resize(rho_grid_len);
        for (std::size_t r = 0; r < rho_grid_len; ++r) {
            result.phi_T_array_BT[r].resize(p, T);
            for (Eigen::Index j = 0; j < p; ++j) {
                double d = delta_BT(j, static_cast<Eigen::Index>(r));
                if (d > eps_) {
                    result.phi_T_array_BT[r].row(j) = phi_T_mat.row(j) / d;
                } else {
                    result.phi_T_array_BT[r].row(j) = phi_T_mat.row(j);
                }
            }
        }

        // Phi_BT: p x rho_grid_len
        result.Phi_BT.resize(p, static_cast<Eigen::Index>(rho_grid_len));
        for (std::size_t r = 0; r < rho_grid_len; ++r) {
            for (Eigen::Index j = 0; j < p; ++j) {
                double d = delta_BT(j, static_cast<Eigen::Index>(r));
                if (d > eps_) {
                    result.Phi_BT(j, static_cast<Eigen::Index>(r)) = Phi(j) / d;
                } else {
                    result.Phi_BT(j, static_cast<Eigen::Index>(r)) = Phi(j);
                }
            }
        }
    }

    return result;
}


// ===================================================================================
// FDP Computation for BT-style
// ===================================================================================

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> TRexDASelector::computeFDP_BT(
    const DACorrectionResult& da_corr,
    std::size_t num_dummies) const
{
    const std::size_t rho_grid_len = da_setup_.rho_grid_len;
    const auto p = static_cast<Eigen::Index>(p_);
    const auto v_len = static_cast<Eigen::Index>(voting_grid_.size());

    // Phi_prime_BT: p x rho_grid_len
    Eigen::MatrixXd Phi_prime_BT(p, static_cast<Eigen::Index>(rho_grid_len));

    // FDP_hat_BT: v_len x rho_grid_len
    Eigen::MatrixXd FDP_hat_BT(v_len, static_cast<Eigen::Index>(rho_grid_len));

    for (std::size_t r = 0; r < rho_grid_len; ++r) {
        Eigen::VectorXd phi_prime_r = computePhiPrime(
            da_corr.phi_T_array_BT[r],
            da_corr.Phi_BT.col(static_cast<Eigen::Index>(r)),
            num_dummies);

        Eigen::VectorXd fdp_hat_r = computeFDPHat(
            voting_grid_,
            da_corr.Phi_BT.col(static_cast<Eigen::Index>(r)),
            phi_prime_r);

        Phi_prime_BT.col(static_cast<Eigen::Index>(r)) = phi_prime_r;
        FDP_hat_BT.col(static_cast<Eigen::Index>(r)) = fdp_hat_r;
    }

    return {Phi_prime_BT, FDP_hat_BT};
}


// ===================================================================================
// Variable Selection for BT-style
// ===================================================================================

TRexDASelector::DASelectionResult TRexDASelector::selectVariables_BT(
    const std::vector<Eigen::MatrixXd>& FDP_hat_array_BT,
    const std::vector<Eigen::MatrixXd>& Phi_array_BT,
    const Eigen::VectorXd& voting_grid,
    const Eigen::VectorXd& rho_grid,
    std::size_t T_stop) const
{
    // FDP_hat_array_BT[rho] = (T_stop × V_len)
    // Phi_array_BT[rho]     = (T_stop × p)
    const auto p = static_cast<Eigen::Index>(p_);
    const std::size_t rho_grid_len = static_cast<std::size_t>(rho_grid.size());
    const auto v_len = static_cast<Eigen::Index>(voting_grid.size());

    // Number of T rows to use (remove last if T_stop > 1, matching R convention)
    std::size_t n_T = (T_stop > 1) ? (T_stop - 1) : T_stop;
    // don't exceed actual rows
    if (!FDP_hat_array_BT.empty()) {
        n_T = std::min(n_T, static_cast<std::size_t>(FDP_hat_array_BT[0].rows()));
    }

    // Build R_array (counts of variables with Phi > V).
    std::vector<Eigen::MatrixXd> R_array(rho_grid_len);
    for (std::size_t r = 0; r < rho_grid_len; ++r) {
        R_array[r] = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(n_T), v_len);
        for (Eigen::Index TT = 0; TT < static_cast<Eigen::Index>(n_T); ++TT) {
            for (Eigen::Index VV = 0; VV < v_len; ++VV) {
                R_array[r](TT, VV) = static_cast<double>(
                    (Phi_array_BT[r].row(TT).array() > voting_grid(VV)).count());
            }
        }
    }

    // Argmax selection (cf. R's select_var_fun_DA_BT):
    //   1. val_max = max of R over FDP-FEASIBLE cells (FDP_hat <= tFDR).
    //   2. Candidate set among cells whose R count equals val_max — either
    //      FDP-feasible only (default, FDR-controlling) or ALL cells including
    //      infeasible ones (RFaithful, reproduces the reference which evaluates
    //      which(R_array == val_max) over the whole array). See bt_selection_mode.
    //   3. Tie-break among candidates: max VV, then max RR (rho), then max TT.

    // Step 1: val_max over feasible cells only.
    double val_max = -1.0;
    bool any_feasible = false;
    for (std::size_t r = 0; r < rho_grid_len; ++r) {
        for (Eigen::Index TT = 0; TT < static_cast<Eigen::Index>(n_T); ++TT) {
            for (Eigen::Index VV = 0; VV < v_len; ++VV) {
                if (FDP_hat_array_BT[r](TT, VV) <= tFDR_) {
                    any_feasible = true;
                    if (R_array[r](TT, VV) > val_max) {
                        val_max = R_array[r](TT, VV);
                    }
                }
            }
        }
    }

    DASelectionResult result;
    result.R_array_BT = R_array;

    if (!any_feasible) {
        // No FDP-feasible cell exists -> no selection.
        result.selected_var = Eigen::VectorXi::Zero(p);
        result.v_thresh = 1.0;
        result.rho_thresh = tc::AUTO_ESTIMATE_CORRELATION;
        return result;
    }

    // Steps 2-3: scan the R_array for cells equal to val_max and pick the winner
    // by tie-break priority max VV -> max RR (rho) -> max TT. Candidate eligibility
    // depends on da_ctrl_.bt_selection_mode: FeasibleOnly (default) restricts to
    // FDP-feasible cells (FDR-controlling); RFaithful also admits infeasible cells,
    // reproducing the R reference (which scans the whole R_array).
    const bool feasible_only =
        (da_ctrl_.bt_selection_mode == BTSelectionMode::FeasibleOnly);

    bool found = false;
    Eigen::Index best_TT = 0, best_VV = -1;
    std::size_t best_RR = 0;
    for (std::size_t r = 0; r < rho_grid_len; ++r) {
        for (Eigen::Index TT = 0; TT < static_cast<Eigen::Index>(n_T); ++TT) {
            for (Eigen::Index VV = 0; VV < v_len; ++VV) {
                if (R_array[r](TT, VV) != val_max) {
                    continue;
                }
                if (feasible_only &&
                    FDP_hat_array_BT[r](TT, VV) > tFDR_) {
                    continue;
                }
                if (!found ||
                    VV > best_VV ||
                    (VV == best_VV && r > best_RR) ||
                    (VV == best_VV && r == best_RR && TT > best_TT)) {
                    best_VV = VV;
                    best_RR = r;
                    best_TT = TT;
                    found = true;
                }
            }
        }
    }

    // Select variables & compile results
    double best_threshold = voting_grid(best_VV);
    result.selected_var = Eigen::VectorXi::Zero(p);
    for (Eigen::Index j = 0; j < p; ++j) {
        if (Phi_array_BT[best_RR](best_TT, j) > best_threshold) {
            result.selected_var(j) = 1;
        }
    }

    result.v_thresh = best_threshold;
    result.rho_thresh = rho_grid(static_cast<Eigen::Index>(best_RR));
    // populate base R_mat with selected rho level
    result.R_mat = R_array[best_RR];

    return result;
}


// ===================================================================================
} /* End of namespace trex::trex_selector_methods::trex_da */
// ===================================================================================
