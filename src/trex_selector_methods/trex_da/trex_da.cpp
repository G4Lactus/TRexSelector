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
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
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
    TRexSelector(X, y, tFDR, std::move(trex_control), seed, verbose),
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
                    "Each prior_groups level must have length p = " + std::to_string(p_) +
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
            "rho_thr_DA must be in [0, 1]. Got: " + std::to_string(da_ctrl_.rho_thr_DA));
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
    // -----------------------------------------------------------------
    voting_grid_ = createVotingGrid();
    const std::size_t v_len = voting_grid_.size();
    setPercentOptPoint(v_len, trex_ctrl_.opt_threshold);

    // 2. Build neighbourhood / correlation structure
    // -----------------------------------------------------------------
    if (verbose_) { printProgress("DA: Building dependency structure..."); }
    setupDA();

    const bool use_BT = da_setup_.use_BT_style;
    const std::size_t rho_grid_len = da_setup_.rho_grid_len;

    if (verbose_) {
        std::string method_str;
        switch (da_ctrl_.method) {
            case DAMethod::AR1:          method_str = "AR1";          break;
            case DAMethod::EQUI:         method_str = "EQUI";        break;
            case DAMethod::BT:           method_str = "BT";          break;
            case DAMethod::NN:           method_str = "NN";          break;
            case DAMethod::PRIOR_GROUPS: method_str = "PRIOR_GROUPS"; break;
        }
        printProgress("DA: Method = " + method_str +
                      (use_BT ? " (BT-style, rho_grid_len = " +
                       std::to_string(rho_grid_len) + ")" : ""));
    }

    // 3. Helper lambda: run one experiment step with DA correction + FDP
    // -----------------------------------------------------------------
    // Returns: (FDP_hat_vec, FDP_hat_BT_mat, da_corr, exp_results)
    // For non-BT: FDP_hat_vec has size v_len
    // For BT: FDP_hat_BT_mat has size (v_len x rho_grid_len)

    struct StepResult {
        Eigen::VectorXd FDP_hat_vec;       // non-BT (v_len)
        Eigen::MatrixXd FDP_hat_BT_mat;    // BT (v_len x rho_grid_len)
        DACorrectionResult da_corr;
        er::ExperimentResults exp_results;
        Eigen::VectorXd Phi_prime;         // non-BT
        Eigen::MatrixXd Phi_prime_BT;      // BT (p x rho_grid_len)
    };

    auto runStep = [&](std::size_t T_stop_cur, std::size_t num_dummies_cur,
                       bool use_warm_start) -> StepResult
    {
        // Determine experiment strategy based on L-loop config
        er::ExperimentStrategy exp_strategy;
        switch (trex_ctrl_.lloop_strategy) {
            case tc::LLoopStrategy::PERMUTATION:
            case tc::LLoopStrategy::PERMUTATION_DIRECT:
                exp_strategy = er::ExperimentStrategy::Permutation;
                break;
            case tc::LLoopStrategy::DIRECT:
                exp_strategy = er::ExperimentStrategy::Direct;
                break;
            default:
                exp_strategy = er::ExperimentStrategy::Standard;
                break;
        }

        auto cfg = buildRunnerConfig(num_dummies_cur, T_stop_cur, use_warm_start,
                                     exp_strategy, dummy_multiplier_LL_);
        StepResult step;
        step.exp_results = experiment_runner_->run(cfg);

        // DA correction
        step.da_corr = daCorrect(step.exp_results.phi_T_mat,
                                  step.exp_results.Phi,
                                  T_stop_cur);

        // FDP computation
        if (use_BT) {
            auto [phi_prime_bt, fdp_hat_bt] = computeFDP_BT(step.da_corr, num_dummies_cur);
            step.Phi_prime_BT = std::move(phi_prime_bt);
            step.FDP_hat_BT_mat = std::move(fdp_hat_bt);
        } else {
            step.Phi_prime = computePhiPrime(step.da_corr.phi_T_mat,
                                              step.da_corr.Phi,
                                              num_dummies_cur);
            step.FDP_hat_vec = computeFDPHat(voting_grid_,
                                              step.da_corr.Phi,
                                              step.Phi_prime);
        }

        return step;
    };

    // FDP reference-point accessors
    auto fdpRef = [&](const StepResult& step) -> double {
        if (use_BT) {
            return step.FDP_hat_BT_mat(
                static_cast<Eigen::Index>(opt_point_),
                static_cast<Eigen::Index>(da_setup_.opt_point_BT));
        }
        return step.FDP_hat_vec(static_cast<Eigen::Index>(opt_point_));
    };

    auto fdpEdge = [&](const StepResult& step) -> double {
        if (use_BT) {
            return step.FDP_hat_BT_mat(
                static_cast<Eigen::Index>(v_len - 1),
                static_cast<Eigen::Index>(da_setup_.opt_point_BT));
        }
        return step.FDP_hat_vec(static_cast<Eigen::Index>(v_len - 1));
    };

    // 4. L-Loop: calibrate num_dummies with T_stop = 1
    // -----------------------------------------------------------------
    if (verbose_) { printProgress("DA L-loop: Calibrating number of dummies..."); }

    T_stop_ = 1;
    dummy_multiplier_LL_ = 1;
    StepResult step;
    bool first_pass = true;

    while (dummy_multiplier_LL_ <= trex_ctrl_.max_dummy_multiplier &&
           (first_pass || fdpRef(step) > tFDR_)) {

        num_dummies_ = dummy_multiplier_LL_ * p_;

        // Dummy generation (matching base HCONCAT / STANDARD pattern)
        if (trex_ctrl_.lloop_strategy == tc::LLoopStrategy::HCONCAT) {
            if (dummy_multiplier_LL_ == 1) {
                dummy_gen_.generateAndStore(trex_ctrl_.K, p_, /*seed_factor=*/0);
            } else {
                dummy_gen_.expandStored(p_, dummy_multiplier_LL_);
                warm_start_mgr_.invalidate();
            }
        } else {
            dummy_gen_.generateAndStore(trex_ctrl_.K, num_dummies_, dummy_multiplier_LL_);
            warm_start_mgr_.invalidate();
        }

        step = runStep(T_stop_, num_dummies_, /*use_warm_start=*/false);
        first_pass = false;

        if (verbose_) {
            printProgress("  Appended dummies: " + std::to_string(num_dummies_));
        }

        ++dummy_multiplier_LL_;
    }

    dummy_multiplier_LL_ = (dummy_multiplier_LL_ > 1) ? (dummy_multiplier_LL_ - 1) : 1;

    if (verbose_) {
        printProgress("DA L-loop: converged with " +
                      std::to_string(num_dummies_) + " dummies.\n");
    }

    // 5. T-Loop: extend T_stop while FDP at edge is below target
    // -----------------------------------------------------------------
    if (verbose_) {
        printProgress("DA T-loop: Calibrating stopping threshold (T_stop)...");
    }

    std::size_t max_T = trex_ctrl_.use_max_T_stop
        ? std::min(num_dummies_,
                   static_cast<std::size_t>(std::ceil(static_cast<double>(n_) / 2.0)))
        : num_dummies_;

    // Accumulators
    // Non-BT: Phi_mat_ (T x p), FDP_hat_mat_ (T x V_len)
    // BT: Phi_acc_BT[rho] (T x p), FDP_acc_BT[rho] (T x V_len)
    std::vector<Eigen::MatrixXd> Phi_acc_BT;
    std::vector<Eigen::MatrixXd> FDP_acc_BT;
    if (use_BT) {
        Phi_acc_BT.resize(rho_grid_len);
        FDP_acc_BT.resize(rho_grid_len);
    }

    // Store first T=1 result
    auto appendResult = [&](const StepResult& s, std::size_t row_idx) {
        if (use_BT) {
            for (std::size_t r = 0; r < rho_grid_len; ++r) {
                auto ri = static_cast<Eigen::Index>(row_idx);
                auto p_idx = static_cast<Eigen::Index>(p_);
                auto v_idx = static_cast<Eigen::Index>(v_len);

                if (row_idx == 0) {
                    Phi_acc_BT[r].resize(1, p_idx);
                    FDP_acc_BT[r].resize(1, v_idx);
                } else {
                    Phi_acc_BT[r].conservativeResize(ri + 1, Eigen::NoChange);
                    FDP_acc_BT[r].conservativeResize(ri + 1, Eigen::NoChange);
                }

                Phi_acc_BT[r].row(ri) = s.da_corr.Phi_BT.col(static_cast<Eigen::Index>(r)).transpose();
                FDP_acc_BT[r].row(ri) = s.FDP_hat_BT_mat.col(static_cast<Eigen::Index>(r)).transpose();
            }
        } else {
            auto ri = static_cast<Eigen::Index>(row_idx);
            auto p_idx = static_cast<Eigen::Index>(p_);
            auto v_idx = static_cast<Eigen::Index>(v_len);

            if (row_idx == 0) {
                Phi_mat_.resize(1, p_idx);
                FDP_hat_mat_.resize(1, v_idx);
            } else {
                Phi_mat_.conservativeResize(ri + 1, Eigen::NoChange);
                FDP_hat_mat_.conservativeResize(ri + 1, Eigen::NoChange);
            }

            Phi_mat_.row(ri) = s.da_corr.Phi.transpose();
            FDP_hat_mat_.row(ri) = s.FDP_hat_vec.transpose();
        }
    };

    std::size_t row_idx = 0;
    appendResult(step, row_idx);
    ++row_idx;

    // Stagnation tracking
    std::size_t prev_num_selected = 0;
    std::size_t stagnant_iterations = 0;

    while (fdpEdge(step) <= tFDR_ && T_stop_ < max_T) {
        ++T_stop_;
        step = runStep(T_stop_, num_dummies_, /*use_warm_start=*/true);

        // Stagnation check (using the base Phi for simplicity)
        if (trex_ctrl_.tloop_stagnation_stop && T_stop_ > 1) {
            // Quick check: how many selected at default operating point?
            double v_star = voting_grid_(static_cast<Eigen::Index>(opt_point_));
            const Eigen::VectorXd& check_phi = use_BT
                ? step.da_corr.Phi_BT.col(static_cast<Eigen::Index>(da_setup_.opt_point_BT))
                : step.da_corr.Phi;
            std::size_t num_selected = (check_phi.array() > v_star).count();

            if (num_selected <= prev_num_selected) {
                ++stagnant_iterations;
                if (stagnant_iterations >= trex_ctrl_.tloop_max_stagnant_steps) {
                    if (verbose_) {
                        printProgress("DA T-loop EARLY STOP: Stagnation detected");
                    }
                    break;
                }
            } else {
                stagnant_iterations = 0;
            }
            prev_num_selected = num_selected;
        }

        appendResult(step, row_idx);
        ++row_idx;

        if (verbose_) {
            std::ostringstream oss;
            oss << "T = " << std::setw(3) << T_stop_;
            if (use_BT) {
                oss << " | FDP_BT[edge] = " << fdpEdge(step);
            } else {
                oss << " | FDP in [" << step.FDP_hat_vec.minCoeff()
                    << ", " << step.FDP_hat_vec.maxCoeff() << "]";
            }
            if (stagnant_iterations > 0) {
                oss << " (stagnant: " << stagnant_iterations << ")";
            }
            printProgress(oss.str());
        }
    }

    if (verbose_) {
        printProgress("DA T-loop converged at T_stop = " +
                      std::to_string(T_stop_) + "\n");
    }

    // 6. Final Phi_prime
    // -----------------------------------------------------------------
    if (use_BT) {
        Phi_prime_ = step.Phi_prime_BT.col(
            static_cast<Eigen::Index>(da_setup_.opt_point_BT));
    } else {
        Phi_prime_ = step.Phi_prime;
    }

    // 7. Variable selection
    // -----------------------------------------------------------------
    if (use_BT) {
        da_result_ = selectVariables_BT(
            FDP_acc_BT, Phi_acc_BT, voting_grid_, da_setup_.rho_grid, T_stop_);
    } else {
        // Use base class selection
        auto base_result = selectedVariables(FDP_hat_mat_, Phi_mat_, voting_grid_, T_stop_);
        da_result_.selected_var = base_result.selected_var;
        da_result_.v_thresh = base_result.v_thresh;
        da_result_.R_mat = base_result.R_mat;
        da_result_.rho_thresh = std::numeric_limits<double>::quiet_NaN();
    }

    // Fill common fields
    da_result_.T_stop = T_stop_;
    da_result_.num_dummies = num_dummies_;
    da_result_.dummy_factor_L = dummy_multiplier_LL_;
    da_result_.Phi_prime = Phi_prime_;
    da_result_.voting_grid = voting_grid_;
    da_result_.method = da_ctrl_.method;
    da_result_.cor_coef = da_setup_.cor_coef;
    da_result_.rho_grid = da_setup_.rho_grid;

    if (use_BT) {
        da_result_.FDP_hat_mat = Eigen::MatrixXd();  // Not used for BT
        da_result_.Phi_mat = Eigen::MatrixXd();
        da_result_.FDP_hat_array_BT = FDP_acc_BT;
        da_result_.Phi_array_BT = Phi_acc_BT;
    } else {
        da_result_.FDP_hat_mat = FDP_hat_mat_;
        da_result_.Phi_mat = Phi_mat_;
    }

    storeResults(da_result_);

    if (verbose_) {
        printProgress("DA: Selection complete. Selected " +
                      std::to_string(getNumSelected()) + " variables.");
    }

    // 8. Cleanup
    // -----------------------------------------------------------------
    warm_start_mgr_.cleanup();
    if (memmap_mgr_) { memmap_mgr_->cleanup(); }

    if (verbose_) { printProgress("Denormalizing X to original scale.\n"); }
    if (X_is_normalized_) {
        dn::denormalizeX(*X_, norm_params_);
        X_is_normalized_ = false;
    }

    // 9. Return base-sliced result
    // -----------------------------------------------------------------
    SelectionResult base_result;
    base_result.selected_var = da_result_.selected_var;
    base_result.v_thresh = da_result_.v_thresh;
    base_result.R_mat = da_result_.R_mat;
    base_result.T_stop = da_result_.T_stop;
    base_result.num_dummies = da_result_.num_dummies;
    base_result.dummy_factor_L = da_result_.dummy_factor_L;
    base_result.FDP_hat_mat = da_result_.FDP_hat_mat;
    base_result.Phi_mat = da_result_.Phi_mat;
    base_result.Phi_prime = da_result_.Phi_prime;
    base_result.voting_grid = da_result_.voting_grid;

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
// DA Setup — Route 1: Prior Groups
// ===================================================================================

void TRexDASelector::setupDA_PriorGroups() {
    const auto& groups = da_ctrl_.prior_groups;
    const std::size_t rho_grid_len = groups.size();
    const auto p = static_cast<Eigen::Index>(p_);

    da_setup_.use_BT_style = true;
    da_setup_.rho_grid_len = rho_grid_len;
    da_setup_.opt_point_BT = static_cast<std::size_t>(
        std::round(0.75 * static_cast<double>(rho_grid_len)));
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
// DA Setup — Route 2: BT (dendrogram-based)
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

    // Subsample to grid_len evenly spaced indices
    da_setup_.rho_grid.resize(static_cast<Eigen::Index>(grid_len));
    for (std::size_t i = 0; i < grid_len; ++i) {
        double frac = static_cast<double>(i) * static_cast<double>(p_ - 1)
                    / static_cast<double>(grid_len - 1);
        std::size_t idx = static_cast<std::size_t>(std::round(frac));
        if (idx >= rho_grid_full.size()) idx = rho_grid_full.size() - 1;
        da_setup_.rho_grid(static_cast<Eigen::Index>(i)) = rho_grid_full[idx];
    }

    da_setup_.use_BT_style = true;
    da_setup_.rho_grid_len = grid_len;
    da_setup_.opt_point_BT = static_cast<std::size_t>(
        std::round(0.75 * static_cast<double>(grid_len)));
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
        auto labels = hac::DendrogramUtils::cut_tree_by_height(merges, p, cut_height);
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
// DA Setup — Route 3: NN (nearest-neighbour threshold sweep)
// ===================================================================================

void TRexDASelector::setupDA_NN() {
    const auto p = static_cast<Eigen::Index>(p_);
    const std::size_t grid_len = da_ctrl_.hc_grid_length;

    // Compute correlation matrix (X is centered + L2-normalized)
    Eigen::MatrixXd cor_mat = X_->transpose() * (*X_);

    // Build rho_grid = linspace(0, 1, grid_len)
    da_setup_.rho_grid = Eigen::VectorXd::LinSpaced(
        static_cast<Eigen::Index>(grid_len), 0.0, 1.0);

    da_setup_.use_BT_style = true;
    da_setup_.rho_grid_len = grid_len;
    da_setup_.opt_point_BT = static_cast<std::size_t>(
        std::round(0.75 * static_cast<double>(grid_len)));
    if (da_setup_.opt_point_BT >= grid_len) {
        da_setup_.opt_point_BT = grid_len - 1;
    }
    da_setup_.cor_coef = da_ctrl_.cor_coef;
    da_setup_.kap = 0;

    // Build gr_j_list: for each j and rho level, collect neighbours with |cor| >= rho
    da_setup_.gr_j_list.resize(p_);
    for (Eigen::Index j = 0; j < p; ++j) {
        da_setup_.gr_j_list[j].resize(grid_len);
        for (std::size_t r = 0; r < grid_len; ++r) {
            double rho_threshold = da_setup_.rho_grid(static_cast<Eigen::Index>(r));
            for (Eigen::Index k = 0; k < p; ++k) {
                if (k != j && std::abs(cor_mat(j, k)) >= rho_threshold) {
                    da_setup_.gr_j_list[j][r].push_back(k);
                }
            }
        }
    }

    if (verbose_) {
        printProgress("DA-NN: Correlation matrix computed. Grid length = " +
                      std::to_string(grid_len));
    }
}


// ===================================================================================
// DA Setup — Route 4: AR(1)
// ===================================================================================

void TRexDASelector::setupDA_AR1() {
    da_setup_.use_BT_style = false;
    da_setup_.rho_grid_len = 0;
    da_setup_.kap = 0;

    // Estimate AR(1) coefficient if not supplied
    if (std::isnan(da_ctrl_.cor_coef)) {
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
        printProgress("DA-AR1: cor_coef = " + std::to_string(da_setup_.cor_coef) +
                      ", kap = " + std::to_string(da_setup_.kap));
    }
}


// ===================================================================================
// DA Setup — Route 5: Equicorrelation
// ===================================================================================

void TRexDASelector::setupDA_EQUI() {
    da_setup_.use_BT_style = false;
    da_setup_.rho_grid_len = 0;
    da_setup_.kap = 0;

    // Estimate equicorrelation if not supplied
    if (std::isnan(da_ctrl_.cor_coef)) {
        da_setup_.cor_coef = estimateEquiCorrelation();
    } else {
        da_setup_.cor_coef = da_ctrl_.cor_coef;
    }

    if (verbose_) {
        printProgress("DA-EQUI: cor_coef = " + std::to_string(da_setup_.cor_coef) +
                      " (threshold = " + std::to_string(da_ctrl_.rho_thr_DA) + ")");
    }
}


// ===================================================================================
// Correlation Estimation — AR(1)
// ===================================================================================

double TRexDASelector::estimateAR1Correlation() const {
    // Row-wise Yule-Walker: ρ̂_i = Σ_{t=0}^{p-2} x_{i,t} * x_{i,t+1} / Σ_{t=0}^{p-1} x_{i,t}²
    // (X is already centered)
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
    // Mean of off-diagonal correlation matrix elements.
    // Since X is centered and L2-normalized: cor_mat = X^T * X
    // Efficient computation via row sums: Σ_all = ||row_sums||²
    const auto p = static_cast<Eigen::Index>(p_);

    // Compute row sums of X
    Eigen::VectorXd row_sums = X_->rowwise().sum();
    double sum_all = row_sums.squaredNorm();

    // Mean off-diagonal = (sum_all - p) / (p * (p - 1))
    // Diagonal of X^T*X = 1.0 per column (L2-normalized), so diagonal sum = p
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
                MapType, CorrDist, hac::LinkageMethod::Single>(*X_);

        case hac::LinkageMethod::Complete:
            return hac::AgglomerativeClustering::cluster<
                MapType, CorrDist, hac::LinkageMethod::Complete>(*X_);

        case hac::LinkageMethod::Average:
            return hac::AgglomerativeClustering::cluster<
                MapType, CorrDist, hac::LinkageMethod::Average>(*X_);

        case hac::LinkageMethod::WPGMA:
            return hac::AgglomerativeClustering::cluster<
                MapType, CorrDist, hac::LinkageMethod::WPGMA>(*X_);

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
                Eigen::Index lo = std::max(static_cast<Eigen::Index>(0), j - static_cast<Eigen::Index>(kap));
                Eigen::Index hi = std::min(p - 1, j + static_cast<Eigen::Index>(kap));
                for (Eigen::Index k = lo; k <= hi; ++k) {
                    if (k != j) {
                        double diff = std::abs(result.phi_T_mat(j, t) - result.phi_T_mat(k, t));
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
                        double diff = std::abs(result.phi_T_mat(j, t) - result.phi_T_mat(k, t));
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
    // Ensure we don't exceed actual rows
    if (!FDP_hat_array_BT.empty()) {
        n_T = std::min(n_T, static_cast<std::size_t>(FDP_hat_array_BT[0].rows()));
    }

    // Build R_array and apply FDP constraint
    // R_array[rho] = (n_T × V_len): count of variables with Phi > V
    std::vector<Eigen::MatrixXd> R_array(rho_grid_len);
    double val_max = -1.0;
    Eigen::Index best_TT = 0, best_VV = 0;
    std::size_t best_RR = 0;

    for (std::size_t r = 0; r < rho_grid_len; ++r) {
        R_array[r] = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(n_T), v_len);

        for (Eigen::Index TT = 0; TT < static_cast<Eigen::Index>(n_T); ++TT) {
            for (Eigen::Index VV = 0; VV < v_len; ++VV) {
                double threshold = voting_grid(VV);
                R_array[r](TT, VV) = static_cast<double>(
                    (Phi_array_BT[r].row(TT).array() > threshold).count());

                // Check if this is a valid candidate (FDP <= tFDR)
                double fdp = FDP_hat_array_BT[r](TT, VV);
                if (fdp <= tFDR_) {
                    double r_count = R_array[r](TT, VV);
                    if (r_count > val_max ||
                        (r_count == val_max && VV > best_VV)) {
                        val_max = r_count;
                        best_TT = TT;
                        best_VV = VV;
                        best_RR = r;
                    }
                }
            }
        }
    }

    DASelectionResult result;
    result.R_array_BT = R_array;

    if (val_max < 0.0) {
        // No valid selection found
        result.selected_var = Eigen::VectorXi::Zero(p);
        result.v_thresh = 1.0;
        result.rho_thresh = std::numeric_limits<double>::quiet_NaN();
        return result;
    }

    // Select variables
    double best_threshold = voting_grid(best_VV);
    result.selected_var = Eigen::VectorXi::Zero(p);
    for (Eigen::Index j = 0; j < p; ++j) {
        if (Phi_array_BT[best_RR](best_TT, j) > best_threshold) {
            result.selected_var(j) = 1;
        }
    }

    result.v_thresh = best_threshold;
    result.rho_thresh = rho_grid(static_cast<Eigen::Index>(best_RR));
    result.R_mat = R_array[best_RR];  // populate base R_mat with selected rho level

    return result;
}


// ===================================================================================
} /* End of namespace trex::trex_selector_methods::trex_da */
// ===================================================================================
