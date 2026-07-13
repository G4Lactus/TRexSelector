// ===================================================================================
// tsolver_base.cpp
// ===================================================================================
/**
 * @file tsolver_base.cpp
 *
 * @brief Implementation file for the TSolver_Base class, providing common
 * functionality for all terminating variable selection solvers.
 */
// ===================================================================================

// std includes
#include <utils/logging/logger.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>


// tsolvers includes
#include <tsolvers/tsolver_base.hpp>
#include <tsolvers/solver_utils/solver_preprocessing.hpp>

// utils includes
#include <utils/openmp/utils_openmp.hpp>
#include <utils/fp_classify/fp_classify.hpp>

// ===================================================================================

// Embedded into trex::tsolvers namespace
namespace trex::tsolvers {

// ===================================================================================

// Alias namespaces
namespace su_preproc = trex::tsolvers::solver_utils::preprocessing;
namespace fpc = trex::utils::fp_classify;

// ===================================================================================

// ============================================================================
// Constructor
// ============================================================================

TSolver_Base::TSolver_Base(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D,
    Eigen::Map<Eigen::VectorXd>& y,
    bool normalize,
    bool intercept,
    bool verbose,
    ScalingMode scaling_mode
    )
    : X_(X), D_(D), y_(y), normalize_(normalize), scaling_mode_(scaling_mode),
      intercept_(intercept), verbose_(verbose), is_connected_(true) {

    p_original_ = static_cast<std::size_t>(X.cols());
    num_dummies_ = static_cast<std::size_t>(D.cols());
    dummy_start_idx_ = p_original_;

    // Validation
    if (X.rows() != y.size()) {
        throw std::invalid_argument(
            concatMsg("TSolver_Base: X rows (", X.rows(), ") do not match y entries (",
                         y.size(), ")")
        );
    }
    if (D.rows() != X.rows()) {
        throw std::invalid_argument(
            concatMsg("TSolver_Base: D rows (", D.rows(), ") do not match X rows (", X.rows(),
             ")")
        );
    }

    if (X.rows() == 1 && intercept_) {
        throw std::invalid_argument(
            "TSolver_Base: cannot fit intercept with only one sample."
        );
    }

    // Set effective_n_ for diagnostics and Cp calculation, depending on intercept handling
    effective_n_ = X.rows() - (intercept_ ? 1 : 0);
}


// ============================================================================
// Internal Tools & Utilities
// ============================================================================

void TSolver_Base::preprocess() {
    // 1. Preprocess original predictors X
    su_preproc::preprocessMatStatistics(*X_, intercept_, normalize_,
        eps_, meansx_, normsx_, dropped_indices_, scaling_mode_);

    // 2. Preprocess dummy predictors D (Offset dropped indices)
    if (D_.has_value() && num_dummies_ > 0) {
        std::vector<std::size_t> d_dropped;
        su_preproc::preprocessMatStatistics(*D_, intercept_, normalize_,
            eps_, meansd_, normsd_, d_dropped, scaling_mode_);

        // Map D drops into the unified index space
        for (std::size_t j : d_dropped) {
            dropped_indices_.push_back(j + dummy_start_idx_);
        }
    }

    // 3. Center Response
    if (intercept_) { su_preproc::centerResponse(y_, mu_y_); }

    // 4. Log any dropped columns due to low variance
    if (normalize_ && !dropped_indices_.empty()) {
        std::sort(dropped_indices_.begin(), dropped_indices_.end());
        for (std::size_t j : dropped_indices_) {
            logWarning(concatMsg("Column ", j, " has variance < eps_; dropped"));
        }
    }
}


void TSolver_Base::restore(Eigen::Map<Eigen::MatrixXd>& X,
                           Eigen::Map<Eigen::MatrixXd>& D,
                           Eigen::Map<Eigen::VectorXd>& y) const {

    // Safety assertions
    if (static_cast<std::size_t>(X.cols()) != p_original_) {
        throw std::invalid_argument(
            concatMsg(solverTypeToString(), "::restore: X columns mismatch.")
        );
    }
    if (static_cast<std::size_t>(D.cols()) != num_dummies_) {
        throw std::invalid_argument(
            concatMsg(solverTypeToString(), "::restore: D columns mismatch.")
        );
    }
    if (y.size() != X.rows()) {
        throw std::invalid_argument(
            concatMsg(solverTypeToString(), "::restore: y length mismatch with X rows.")
        );
    }

    su_preproc::restoreMatStatistics(
        X, intercept_, meansx_, normalize_, normsx_);

    if (num_dummies_ > 0) {
        su_preproc::restoreMatStatistics(
            D, intercept_, meansd_, normalize_, normsd_);
    }

    // NOTE: y is NOT modified. The constructor copies the response into y_
    // and centers only the internal copy, so the caller's y buffer was never
    // preprocessed in place — adding mu_y_ back would corrupt it.
}


void TSolver_Base::updateDummyTracking() {
    dummies_at_step_.push_back(count_active_dummies_);
}


void TSolver_Base::initializeInactives() {
    inactives_.clear();
    std::size_t p_tot = p_original_ + num_dummies_;
    inactives_.reserve(p_tot - dropped_indices_.size());
    constexpr std::size_t HASH_THRESHOLD{10};

    if (dropped_indices_.size() < HASH_THRESHOLD) {
        for (std::size_t j = 0; j < p_tot; ++j) {
            if (std::ranges::find(dropped_indices_, j) == dropped_indices_.end())
                inactives_.push_back(j);
        }
    } else {
        std::unordered_set<std::size_t> dropped_set(
            dropped_indices_.begin(), dropped_indices_.end());
        for (std::size_t j = 0; j < p_tot; ++j) {
            if (dropped_set.find(j) == dropped_set.end())
                inactives_.push_back(j);
        }
    }
}


void TSolver_Base::updateInactiveSet() {
    if (!any_dropped_) return;

    inactives_.clear();
    std::unordered_set<std::size_t> active_set(actives_.begin(), actives_.end());
    std::unordered_set<std::size_t> dropped_set(
        dropped_indices_.begin(), dropped_indices_.end());
    std::size_t p_tot = p_original_ + num_dummies_;

    std::vector<std::vector<std::size_t>> thread_buffers(omp_get_max_threads());

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        thread_buffers[tid].reserve(p_tot / omp_get_max_threads());

        #pragma omp for schedule(static)
        for (std::size_t j = 0; j < p_tot; ++j) {
            if (active_set.find(j) == active_set.end() &&
                dropped_set.find(j) == dropped_set.end())
                thread_buffers[tid].push_back(j);
        }
    }

    for (const auto& buf : thread_buffers) {
        inactives_.insert(inactives_.end(), buf.begin(), buf.end());
    }

    any_dropped_ = false;
}

// ============================================================================
// Beta Path Storage (sparse)
// ============================================================================

void TSolver_Base::initBetaPathStorage() {
    clearRunningBeta();
    betaPathSparse_.clear();
    betaPathSparse_.emplace_back();  // all-zero step 0
}


double TSolver_Base::runningBeta(std::size_t j) const {
    auto it = beta_pos_.find(j);
    return (it == beta_pos_.end()) ? 0.0 : beta_val_[it->second];
}


double& TSolver_Base::runningBetaRef(std::size_t j) {
    auto it = beta_pos_.find(j);
    if (it != beta_pos_.end()) { return beta_val_[it->second]; }

    beta_pos_[j] = beta_idx_.size();
    beta_idx_.push_back(j);
    beta_val_.push_back(0.0);
    return beta_val_.back();
}


void TSolver_Base::removeRunningBeta(std::size_t j) {
    auto it = beta_pos_.find(j);
    if (it == beta_pos_.end()) { return; }

    // Swap-remove; fix the lookup of the entry moved into the freed slot.
    std::size_t slot = it->second;
    std::size_t last = beta_idx_.size() - 1;
    if (slot != last) {
        beta_idx_[slot] = beta_idx_[last];
        beta_val_[slot] = beta_val_[last];
        beta_pos_[beta_idx_[slot]] = slot;
    }
    beta_idx_.pop_back();
    beta_val_.pop_back();
    beta_pos_.erase(it);
}


void TSolver_Base::clearRunningBeta() {
    beta_idx_.clear();
    beta_val_.clear();
    beta_pos_.clear();
}


void TSolver_Base::setRunningBeta(const std::vector<std::size_t>& idx,
                                  const Eigen::Ref<const Eigen::VectorXd>& values) {
    clearRunningBeta();
    beta_idx_ = idx;
    beta_val_.assign(values.data(), values.data() + values.size());
    beta_pos_.reserve(idx.size());
    for (std::size_t s = 0; s < idx.size(); ++s) {
        beta_pos_[idx[s]] = s;
    }
}


void TSolver_Base::recordBetaStep() {
    SparseBetaStep step;
    step.idx = beta_idx_;
    step.val = beta_val_;
    betaPathSparse_.push_back(std::move(step));
}


SparseBetaPath TSolver_Base::makeScaledSparsePath(double pre_divisor) const {
    SparseBetaPath out;
    out.p_total = p_original_ + num_dummies_;
    out.steps = betaPathSparse_;

    for (SparseBetaStep& step : out.steps) {
        for (std::size_t k = 0; k < step.idx.size(); ++k) {
            // Same operation order as the dense accessors: penalty rescale
            // first (x / 1.0 is bit-exact for the unpenalized case), then
            // de-normalization by the column norm.
            double v = step.val[k] / pre_divisor;
            if (normalize_) {
                std::size_t j = step.idx[k];
                if (j < p_original_) {
                    if (normsx_.size() > 0) {
                        v /= normsx_(static_cast<Eigen::Index>(j));
                    }
                } else if (normsd_.size() > 0) {
                    v /= normsd_(static_cast<Eigen::Index>(j - p_original_));
                }
            }
            step.val[k] = v;
        }
    }
    return out;
}


// ============================================================================
// Correlation Utilities
// ============================================================================

void TSolver_Base::initializeCorrelations() {
    // setZero (not resize): entries of dropped columns are never written but
    // are serialized — leaving them uninitialized makes checkpoints
    // nondeterministic.
    correlations_.setZero(static_cast<Eigen::Index>(p_original_ + num_dummies_));
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < inactives_.size(); ++i) {
        std::size_t j = inactives_[i];
        correlations_(static_cast<Eigen::Index>(j)) = getColumn(j).dot(y_);
    }
}


std::pair<double, std::vector<std::size_t>>
TSolver_Base::findTiedMaxCorrelations() const {
    double Cmax = 0.0;
    std::vector<std::size_t> tied;
    tied.reserve(10);

    for (std::size_t j : inactives_) {
        double abs_corr = std::abs(correlations_(static_cast<Eigen::Index>(j)));
        if (abs_corr > Cmax + eps_) {
            Cmax = abs_corr;
            tied.clear();
            tied.push_back(j);
        } else if (abs_corr >= Cmax - eps_) {
            tied.push_back(j);
        }
    }
    return {Cmax, tied};
}


std::pair<double, std::vector<std::size_t>>
TSolver_Base::findTiedMaxCorrelationsAllNonDropped() const {
    double Cmax = 0.0;
    std::vector<std::size_t> tied;
    tied.reserve(10);
    std::size_t p_tot = p_original_ + num_dummies_;

    std::unordered_set<std::size_t> dropped_set(dropped_indices_.begin(),
                                                dropped_indices_.end());
    for (std::size_t j = 0; j < p_tot; ++j) {
        if (dropped_set.find(j) != dropped_set.end()) { continue; }

        double abs_corr = std::abs(correlations_(static_cast<Eigen::Index>(j)));
        if (abs_corr > Cmax + eps_) {
            Cmax = abs_corr;
            tied.clear();
            tied.push_back(j);
        } else if (abs_corr >= Cmax - eps_) {
            tied.push_back(j);
        }
    }
    return {Cmax, tied};
}


void TSolver_Base::refreshInactiveCorrelations() {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < inactives_.size(); ++i) {
        std::size_t j = inactives_[i];
        correlations_(static_cast<Eigen::Index>(j)) = getColumn(j).dot(r_);
    }
}


void TSolver_Base::refreshAllCorrelations() {
    std::size_t p_tot = p_original_ + num_dummies_;
    std::unordered_set<std::size_t> dropped_set(dropped_indices_.begin(),
                                                dropped_indices_.end());
    #pragma omp parallel for schedule(static)
    for (std::size_t j = 0; j < p_tot; ++j) {
        correlations_(static_cast<Eigen::Index>(j)) =
            dropped_set.contains(j) ? 0.0 : getColumn(j).dot(r_);
    }
}

// ============================================================================
// Cholesky update/downdate
// ============================================================================

Eigen::MatrixXd TSolver_Base::updateR(const Eigen::Ref<const Eigen::VectorXd>& xnew, double eps) {
    std::size_t m = actives_.size();
    double xtx = xnew.dot(xnew);

    if (R_.size() == 0 || m == 0) {
        Eigen::MatrixXd newR(1, 1);
        newR(0, 0) = std::sqrt(xtx);
        last_updateR_rank_ = 1;
        return newR;
    }

    Eigen::VectorXd cross_prod(m);
    for (std::size_t k = 0; k < m; ++k) {
        cross_prod(static_cast<Eigen::Index>(k)) = xnew.dot(getColumn(actives_[k]));
    }

    Eigen::VectorXd r = backsolveT(R_, cross_prod);
    double rpp_sq = xtx - r.dot(r);
    double rpp{};
    int new_rank = static_cast<int>(R_.rows());

    if (rpp_sq < eps * xtx) {
        rpp = eps;
    } else {
        rpp = std::sqrt(rpp_sq);
        ++new_rank;
    }

    Eigen::MatrixXd newR = Eigen::MatrixXd::Zero(R_.rows() + 1, R_.cols() + 1);
    if (R_.rows() > 0 && R_.cols() > 0) {
        newR.topLeftCorner(R_.rows(), R_.cols()) = R_;
    }
    if (r.size() > 0) {
        newR.block(0, R_.cols(), R_.rows(), 1) = r;
    }
    newR(R_.rows(), R_.cols()) = rpp;

    last_updateR_rank_ = new_rank;
    return newR;
}


Eigen::MatrixXd TSolver_Base::downdateR(const Eigen::Ref<const Eigen::MatrixXd>& R,
                                        std::size_t k,
                                        double eps
) {
    std::size_t p = R.rows();
    if (p == 1) { return Eigen::MatrixXd(); } // Dropping last variable

    // 1. Delete k-th row and k-th column
    Eigen::MatrixXd Rnew(R.rows(), R.cols() - 1);
    if (k > 0) {
        Rnew.leftCols(k) = R.leftCols(k);
    }
    if (k < static_cast<std::size_t>(R.cols()) - 1) {
        Rnew.rightCols(R.cols() - k - 1) = R.rightCols(R.cols() - k - 1);
    }

    // 2. Restore upper-triangular form via Givens rotations
    for (std::size_t i = k; i < p - 1; ++i) {
        if (i < static_cast<std::size_t>(Rnew.cols())) {
            double a = Rnew(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i));
            double b = Rnew(static_cast<Eigen::Index>(i + 1), static_cast<Eigen::Index>(i));
            double r = std::hypot(a, b);

            if (r > eps) {
                double c = a / r;
                double s = -b / r;
                for (std::size_t j = i; j < static_cast<std::size_t>(Rnew.cols()); ++j) {

                    double temp_i = Rnew(static_cast<Eigen::Index>(i),
                                         static_cast<Eigen::Index>(j));

                    double temp_ip1 = Rnew(static_cast<Eigen::Index>(i + 1),
                                           static_cast<Eigen::Index>(j));

                    Rnew(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
                        c * temp_i - s * temp_ip1;

                    Rnew(static_cast<Eigen::Index>(i + 1), static_cast<Eigen::Index>(j)) =
                        s * temp_i + c * temp_ip1;
                }
            } else {
                // Prevent zeros on the diagonal
                Rnew(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) = eps;
            }
        }
    }

    // 3. Drop bottom row
    Rnew.conservativeResize(static_cast<Eigen::Index>(p) - 1, Rnew.cols());
    return Rnew;
}


// ============================================================================
// Triangular Solvers
// ============================================================================

Eigen::VectorXd TSolver_Base::backsolve(
    const Eigen::Ref<const Eigen::MatrixXd>& R,
    const Eigen::Ref<const Eigen::VectorXd>& b
) {
    return R.triangularView<Eigen::Upper>().solve(b);
}


Eigen::VectorXd TSolver_Base::backsolveT(
    const Eigen::Ref<const Eigen::MatrixXd>& R,
    const Eigen::Ref<const Eigen::VectorXd>& b
) {
    return R.transpose().triangularView<Eigen::Lower>().solve(b);
}


// ============================================================================
// Tie Handling
// ============================================================================

void TSolver_Base::pruneTiedDummies(std::vector<std::size_t>& new_vars,
                                    std::size_t T_stop,
                                    bool early_stop
) {
    if (!early_stop || T_stop == 0 || new_vars.size() <= 1)  return;

    // Check before subtracting: the unsigned difference would wrap if the
    // budget were already exhausted, silently disabling the pruning below.
    if (count_active_dummies_ >= T_stop) {
        throw std::logic_error(
            "TSolver_Base::pruneTiedDummies: dummy budget already exhausted "
            "(count_active_dummies_ >= T_stop)");
    }
    std::size_t budget = T_stop - count_active_dummies_;

    std::vector<std::size_t> real_vars, dummy_vars;
    for (std::size_t j : new_vars) {
        (j >= dummy_start_idx_ ? dummy_vars : real_vars).push_back(j);
    }

    if (dummy_vars.size() > budget) {
        std::shuffle(dummy_vars.begin(), dummy_vars.end(), rng_);
        dummy_vars.resize(budget);
        new_vars = std::move(real_vars);
        new_vars.insert(new_vars.end(), dummy_vars.begin(), dummy_vars.end());
    }
}


// ============================================================================
// Getters and Diagnostics
// ============================================================================

Eigen::VectorXd TSolver_Base::getBeta(int step) const {
    if (betaPathSparse_.empty()) {
        throw std::runtime_error(
            concatMsg(solverTypeToString(), "::getBeta: No path available.")
        );
    }
    std::size_t use_step = (step < 0) ? betaPathSparse_.size() - 1
                                      : static_cast<std::size_t>(step);
    if (use_step >= betaPathSparse_.size()) {
        throw std::invalid_argument(
            concatMsg(solverTypeToString(), "::getBeta: step ", step,
                      " out of range [0, ", betaPathSparse_.size() - 1, "].")
        );
    }

    Eigen::VectorXd coef = Eigen::VectorXd::Zero(
        static_cast<Eigen::Index>(p_original_ + num_dummies_));
    const SparseBetaStep& s = betaPathSparse_[use_step];
    for (std::size_t k = 0; k < s.idx.size(); ++k) {
        coef(static_cast<Eigen::Index>(s.idx[k])) = s.val[k];
    }
    if (normalize_) {
        if (normsx_.size() > 0) coef.head(p_original_).array() /= normsx_.array();
        if (normsd_.size() > 0) coef.tail(num_dummies_).array() /= normsd_.array();
    }
    return coef;
}


double TSolver_Base::getIntercept(int step) const {
    if (!intercept_) return 0.0;
    Eigen::VectorXd beta_orig = getBeta(step);

    double intercept = mu_y_;
    if (meansx_.size() > 0) { intercept -= meansx_.dot(beta_orig.head(p_original_)); }
    if (meansd_.size() > 0) { intercept -= meansd_.dot(beta_orig.tail(num_dummies_)); }
    return intercept;
}


SparseBetaPath TSolver_Base::getBetaPathSparse() const {
    return makeScaledSparsePath(1.0);
}


std::vector<double> TSolver_Base::getCp() const {
    std::vector<double> cp_out;
    if (RSS_.empty() || DoF_.empty()) return cp_out;

    // Derive n from serialized state so Cp is available on disconnected solvers
    std::size_t n = effective_n_ + (intercept_ ? 1 : 0);
    double resvar = fpc::quiet_nan();

    for (int i = static_cast<int>(RSS_.size()) - 1; i >= 0; --i) {
        if (DoF_[i] < n && RSS_[i] > 0) {
            resvar = RSS_[i] / static_cast<double>(n - DoF_[i]);
            break;
        }
    }

    if (!fpc::isfinite(resvar)) return std::vector<double>(
        RSS_.size(), fpc::quiet_nan());

    for (std::size_t k = 0; k < RSS_.size(); ++k) {
        cp_out.push_back((DoF_[k] < n && resvar > 0) ?
            (RSS_[k] / resvar - static_cast<double>(n) + 2.0 * static_cast<double>(DoF_[k])) :
            fpc::quiet_nan());
    }
    return cp_out;
}


std::vector<double> TSolver_Base::getCp(double sigma_hat_sq) const {
    std::vector<double> cp_out;
    cp_out.reserve(RSS_.size());

    // Derive n from serialized state so Cp is available on disconnected solvers
    std::size_t n = effective_n_ + (intercept_ ? 1 : 0);
    bool valid = fpc::isfinite(sigma_hat_sq) && sigma_hat_sq > 0;
    for (std::size_t k = 0; k < RSS_.size(); ++k) {
        cp_out.push_back((DoF_[k] < n && valid) ? (RSS_[k] / sigma_hat_sq -
            static_cast<double>(n) + 2.0 * static_cast<double>(DoF_[k])) :
            fpc::quiet_nan());
    }
    return cp_out;
}


std::vector<std::size_t> TSolver_Base::getActivePredictorIndices() const {
    std::vector<std::size_t> res;
    res.reserve(actives_.size() - count_active_dummies_);
    for (std::size_t j : actives_) if (j < dummy_start_idx_) res.push_back(j);
    return res;
}


std::vector<std::size_t> TSolver_Base::getActiveDummyIndices() const {
    std::vector<std::size_t> res;
    res.reserve(count_active_dummies_);
    for (std::size_t j : actives_) if (j >= dummy_start_idx_) res.push_back(j);
    return res;
}

// ============================================================================
// Logging & Profiling
// ============================================================================


void TSolver_Base::logMsg(const std::string& msg) const { if (verbose_) TREX_INFO(msg); }

void TSolver_Base::logWarning(const std::string& msg) const {
    // Warnings are user-relevant (dropped columns, collinear rejections):
    // record for programmatic retrieval and print unconditionally.
    warnings_.push_back(msg);
    TREX_WARN("[" + solverTypeToString() + "] [WARNING] " + msg);
}

void TSolver_Base::logInfo(const std::string& msg) const { logMsg("[Info] " + msg); }


// ============================================================================
// Serialization
// ============================================================================

void TSolver_Base::validateConnected() const {
    if (!is_connected_ || !X_.has_value()) {
        throw std::runtime_error(
            concatMsg(solverTypeToString(), " is not connected to design matrix. Call reconnect().")
        );
    }

    if (num_dummies_ > 0 && !D_.has_value()) {
        throw std::runtime_error(
            concatMsg(solverTypeToString(), " is not connected to dummy matrix. Call reconnect().")
        );
    }
    // The sparse path stores global indices bounded by construction; the
    // former dense-path row check has no equivalent invariant to validate.
}


void TSolver_Base::reconnect(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::MatrixXd>& D) {
    if (static_cast<std::size_t>(X.cols()) != p_original_ || X.rows() == 0) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(), "::reconnect: X dimensions do not match serialized state."
            )
        );
    }
    if (static_cast<std::size_t>(D.cols()) != num_dummies_ || D.rows() != X.rows()) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(), "::reconnect: D dimensions do not match serialized state."
            )
        );
    }

    // Re-bind the views in place (Eigen::Map has no sane copy assignment)
    X_.emplace(X);
    D_.emplace(D);
    is_connected_ = true;
    logInfo(
        concatMsg("[", solverTypeToString(), "] reconnect successful: Unified space ready.")
    );
}

// ============================================================================
} /* End of namespace trex::tsolvers */
// ============================================================================
