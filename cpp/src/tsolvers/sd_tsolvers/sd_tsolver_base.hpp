// ===================================================================================
// sd_tsolver_base.hpp
// ===================================================================================
#ifndef TSOLVERS_SD_TSOLVER_BASE_HPP
#define TSOLVERS_SD_TSOLVER_BASE_HPP
// ===================================================================================
/**
 * @file sd_tsolver_base.hpp
 *
 * @brief Shared core of the SD (sparse-dummy) solver family: data views,
 * dummy sparsity/scaling contract, active-set state, the Cholesky append
 * shared by every solver, and the diagnostics accessors.
 *
 * @details Family layering (mirrors the classic tsolvers organization):
 *
 *   SDTSolver_Base            — this class: everything both families share.
 *   SDGeneralSolver_Base      — general-k solvers (SD_TLARS/TOMP/TAFS):
 *                               explicit virtual-dummy pool, winner
 *                               materialization, auto-calibration.
 *   SD2PairSolver_Base        — pair (k = 1) twins (SD2_*): dot-free pair
 *                               arithmetic, OnDemand/Geometric policies.
 */
// ===================================================================================

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <optional>

namespace trex::tsolvers {

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
    double eps_{1e-12};

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
    // Shared numerics
    // ==========================================================================
    /**
     * @brief Append one column to the active set's Cholesky factor R_
     * (upper triangular, R^T R = X_A^T X_A) from its precomputed Gram data.
     *
     * @param xtx        Squared L2 norm of the entering column.
     * @param cross_prod Inner products of the entering column with the
     *                   current active columns (length = actives_.size()).
     *
     * @return false on collinear failure (R_ untouched), true on success
     *         (R_ grown by one row/column).
     */
    bool tryAppendCholesky(double xtx,
                           const Eigen::Ref<const Eigen::VectorXd>& cross_prod) {
        const std::size_t m = actives_.size();

        if (m == 0) {
            R_ = Eigen::MatrixXd::Constant(1, 1, std::sqrt(xtx));
            return true;
        }

        Eigen::VectorXd r_vec =
            R_.transpose().triangularView<Eigen::Lower>().solve(cross_prod);
        double rpp_sq = xtx - r_vec.dot(r_vec);

        if (rpp_sq < eps_ * xtx) return false; // Collinear failure

        const auto mi = static_cast<Eigen::Index>(m);
        Eigen::MatrixXd newR = Eigen::MatrixXd::Zero(mi + 1, mi + 1);
        newR.topLeftCorner(mi, mi) = R_;
        newR.block(0, mi, mi, 1) = r_vec;
        newR(mi, mi) = std::sqrt(rpp_sq);
        R_ = newR;
        return true;
    }

    /** @brief Max |correlation| over the inactive original features (the bar
     *  fresh dummies must beat in the LARS/OMP races). */
    double inactivesMaxAbsCorr() const {
        double c_X_max = 0.0;
        for (std::size_t j : inactives_) {
            c_X_max = std::max(c_X_max, std::abs(correlations_(j)));
        }
        return c_X_max;
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
