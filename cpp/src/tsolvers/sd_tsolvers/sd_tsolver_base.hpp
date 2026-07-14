// ===================================================================================
// sd_tsolver_base.hpp
// ===================================================================================
#ifndef TSOLVERS_SD_TSOLVER_BASE_HPP
#define TSOLVERS_SD_TSOLVER_BASE_HPP
// ===================================================================================
/**
 * @file sd_tsolver_base.hpp
 *
 * @brief Base class header for SD-TLARS. The class strips away the dummy matrix,
 * managing only the original features X and the materialized cache for winning dummies.
 *
 */
// ===================================================================================

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <optional>
#include <string>

namespace trex::tsolvers {

/**
 * @brief Dummy-generation policy for the pair (k = 1) solver family
 *        (SD2_TLARS / SD2_TOMP / SD2_TAFS).
 *
 * @details OnDemand mirrors the general-k solvers: an explicit pool of
 *          sampled pairs, expanded one draw at a time until a dummy beats
 *          the best real feature. Geometric exploits the exact 2-sparse
 *          null: the fraction pi of beating pairs is computed from the
 *          sorted residual (two-pointer, O(n log n)), the number of draws
 *          until the first success is sampled as Geometric(pi), and the
 *          winner is drawn uniformly from the beating set. Failures are
 *          never instantiated — only the virtual pool size is tracked.
 *          Standing failures are treated as exchangeable fresh draws
 *          against the current residual (their historical conditioning is
 *          ignored), which is the approximation the policy comparison
 *          quantifies.
 */
enum class SD2GenPolicy : uint8_t {
    OnDemand  = 0,
    Geometric = 1
};

class SDTSolver_Base {
protected:
    // --- Data Maps ---
    std::optional<Eigen::Map<Eigen::MatrixXd>> X_{};
    Eigen::VectorXd y_{};
    Eigen::VectorXd r_{};

    // --- Dimensionality & Sparsity ---
    double rho_d_{0.5};
    double dummy_scale_{1.0};
    double x_col_norm_{1.0};
    std::size_t k_sparse_{0};
    std::size_t p_original_{0};
    std::size_t dummy_start_idx_{0};
    std::size_t T_stop_limit_{0};
    std::size_t effective_n_{0};
    bool intercept_{true};

    // --- Virtual Dummy Mechanics ---
    struct VirtualDummy {
        uint64_t seed;
        std::vector<int> P_indices;
        std::vector<int> M_indices;
        double current_correlation{0.0};
        double current_a{0.0};  // <d, u> of the last step (for c -= gamma*a)
    };

    std::vector<VirtualDummy> pool_Q_;
    Eigen::MatrixXd D_materialized_;
    std::unordered_map<std::size_t, Eigen::Index> active_dummy_map_;
    Eigen::Index current_materialized_cols_{0};

    std::mt19937_64 dummy_rng_;

    // --- Core State ---
    std::vector<std::size_t> actives_;
    std::vector<std::size_t> inactives_;
    Eigen::VectorXd correlations_;
    Eigen::MatrixXd R_;

    // Sparse beta path: actives_ is append-only, so step t only needs the
    // coefficients of its |actives(t)| active variables. Memory is
    // O(steps^2) instead of O((p + L_max) * maxSteps) for a dense path.
    std::vector<double> beta_active_;                   // aligned with actives_
    std::vector<std::vector<double>> betaPathCompact_;  // one entry per step

    std::size_t currentStep_{0};
    std::size_t maxSteps_{0};
    std::size_t count_active_dummies_{0};

    // --- Diagnostics ---
    std::vector<double> RSS_;
    std::vector<double> R2_;
    std::vector<double> lambda_;

    // ==========================================================================
    // Unified Inline Accessor (Zero Virtual Overhead)
    // ==========================================================================
    inline Eigen::Ref<const Eigen::VectorXd> getColumn(std::size_t j) const {
        if (j < p_original_) {
            return X_->col(static_cast<Eigen::Index>(j));
        } else {
            return D_materialized_.col(active_dummy_map_.at(j));
        }
    }

    // ==========================================================================
    // Sparsity Configuration (callable after construction for rho_d = auto)
    // ==========================================================================
    // Balanced sparsity k (pairs of +1 / -1), matching the package's
    // ConstrainedSparseRademacher(s = rho_d): 2k = 2 * floor(s*n/2), clamped
    // to [2, n] and even. Fair-race dummy scale: a balanced +-1 dummy has
    // exact L2 norm sqrt(2k); every dummy statistic is scaled by
    // col_norm/sqrt(2k) so dummies carry the same norm as the (equal-norm)
    // columns of X. Unit-L2 X gives scale 1/sqrt(2k); z-scored X gives the
    // z-score scaling sqrt((n-1)/2k) automatically. X columns must be
    // centered and of equal norm for the selection race to be fair.
    void configureSparsity(double rho_d) {
        if (rho_d <= 0.0 || rho_d > 1.0) {
            throw std::invalid_argument("SDTSolver_Base: rho_d must be in (0, 1].");
        }
        rho_d_ = rho_d;

        const std::size_t n = static_cast<std::size_t>(y_.size());
        std::size_t total_non_zeros =
            2 * static_cast<std::size_t>(static_cast<double>(n) * rho_d_ / 2.0);
        if (total_non_zeros < 2) total_non_zeros = 2;
        if (total_non_zeros > n) total_non_zeros = n - (n % 2);
        k_sparse_ = total_non_zeros / 2;

        dummy_scale_ = x_col_norm_ / std::sqrt(static_cast<double>(total_non_zeros));
    }

    // ==========================================================================
    // Materialization Logic
    // ==========================================================================
    void materializeDummy(const VirtualDummy& dummy, std::size_t global_j) {
        auto col_view = D_materialized_.col(current_materialized_cols_);
        for (int idx : dummy.P_indices) col_view(idx) =  dummy_scale_;
        for (int idx : dummy.M_indices) col_view(idx) = -dummy_scale_;
        active_dummy_map_[global_j] = current_materialized_cols_++;
    }

public:
    SDTSolver_Base(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
                   double rho_d, std::size_t T_stop, bool intercept = true,
                   uint64_t seed = 0)
        : X_(X), y_(y), rho_d_(rho_d), T_stop_limit_(T_stop), intercept_(intercept) {

        p_original_ = X.cols();
        effective_n_ = X.rows() - (intercept_ ? 1 : 0);
        dummy_start_idx_ = p_original_;

        // Fair-race reference norm: dummies must carry the same L2 norm as
        // the (equal-norm) columns of X (see configureSparsity()).
        double norm_sum = 0.0;
        for (Eigen::Index j = 0; j < X.cols(); ++j) norm_sum += X.col(j).norm();
        x_col_norm_ = norm_sum / static_cast<double>(X.cols());

        // rho_d == 0 defers the sparsity setup: the derived class must call
        // configureSparsity() itself (e.g. after auto-calibration).
        if (rho_d_ != 0.0) configureSparsity(rho_d_);

        D_materialized_ = Eigen::MatrixXd::Zero(X.rows(), T_stop_limit_);
        if (seed == 0) dummy_rng_.seed(std::random_device{}());
        else           dummy_rng_.seed(seed);
    }

    virtual ~SDTSolver_Base() = default;
    virtual void executeStep(std::size_t T_stop = 0, bool early_stop = true) = 0;

    // ==========================================================================
    // Read-only accessors (diagnostics for demos / callers)
    // ==========================================================================
    std::size_t getNumSteps() const { return currentStep_; }
    std::size_t getNumActiveDummies() const { return count_active_dummies_; }
    std::size_t numOriginalFeatures() const { return p_original_; }
    std::size_t sparsityK() const { return k_sparse_; }
    double getDummyScale() const { return dummy_scale_; }
    double getXColumnNorm() const { return x_col_norm_; }
    const std::vector<std::size_t>& getActives() const { return actives_; }
    const std::vector<double>& getLambda() const { return lambda_; }

    // Selected original variables (dummies filtered out), in entry order.
    std::vector<std::size_t> getSelectedOriginals() const {
        std::vector<std::size_t> out;
        for (std::size_t j : actives_)
            if (j < p_original_) out.push_back(j);
        return out;
    }

    // Coefficients of the original variables at the current path position.
    Eigen::VectorXd getBeta() const {
        Eigen::VectorXd out =
            Eigen::VectorXd::Zero(static_cast<Eigen::Index>(p_original_));
        for (std::size_t k = 0; k < beta_active_.size(); ++k) {
            if (actives_[k] < p_original_)
                out(static_cast<Eigen::Index>(actives_[k])) = beta_active_[k];
        }
        return out;
    }

    // Full sparse path: entry t holds the coefficients of the first
    // |actives(t)| active variables (global indices in getActives()).
    // NOTE: not consumed by TRexSD (which reads the live active set per
    // T-step) — kept for standalone use: coefficient-path plots and
    // diagnostics of a single solver run.
    const std::vector<std::vector<double>>& getBetaPathCompact() const {
        return betaPathCompact_;
    }
};

} // namespace trex::tsolvers
#endif
