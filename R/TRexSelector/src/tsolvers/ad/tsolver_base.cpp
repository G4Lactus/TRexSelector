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
#include <tsolvers/ad/tsolver_base.hpp>
#include <tsolvers/ad/solver_utils/memory_bytes.hpp>
#include <tsolvers/ad/solver_utils/solver_preprocessing.hpp>

// utils includes
#include <utils/openmp/utils_openmp.hpp>
#include <utils/fp_classify/fp_classify.hpp>

// ===================================================================================

// Embedded into trex::tsolvers::ad namespace
namespace trex::tsolvers::ad {

// ===================================================================================

// Alias namespaces
namespace su_preproc = trex::tsolvers::ad::solver_utils::preprocessing;
namespace bufreg     = trex::tsolvers::ad::solver_utils::buffer_registry;
namespace fpc = trex::utils::fp_classify;

// ===================================================================================

// ============================================================================
// Constructor
// ============================================================================

TSolver_Base::TSolver_Base(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D,
    Eigen::Map<Eigen::VectorXd>& y,
    const SolverOptions& options
    )
    : X_(X), D_(D), y_(y), normalize_(options.normalize),
      scaling_mode_(options.scaling_mode), intercept_(options.intercept),
      precentered_(options.precentered), verbose_(options.verbose),
      is_connected_(true) {

    // Numerics and tie-breaking (construction-time options).
    eps_ = options.tol;
    setExchangeableTie(options.exch_tie_alpha, options.exch_tie_floor);
    if (options.tie_seed >= 0) {
        rng_.seed(static_cast<std::uint32_t>(options.tie_seed));
    }
    n_threads_ = resolveThreadBudget(options.n_threads);

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

    // Row bound of the active set and n of the Cp calculation: centered data
    // (fitted intercept or caller-centered) have rank at most n - 1
    effective_n_ = X.rows() - interceptDoF();
}


// ============================================================================
// Thread budget
// ============================================================================

int TSolver_Base::resolveThreadBudget(int n_threads) const {
    if (n_threads < 1) {
        throw std::invalid_argument(
            concatMsg("TSolver_Base: n_threads must be >= 1. Got: ", n_threads));
    }
    return n_threads;
}


int TSolver_Base::threadsFor(double work) const noexcept {
    return trex::utils::openmp::threads_for(n_threads_, work);
}


// ============================================================================
// Internal Tools & Utilities
// ============================================================================

void TSolver_Base::preprocess() {
    // 0. Single-owner invariant for in-place preprocessing: this
    //    solver is about to transform the caller's X and D through the maps,
    //    so it claims both buffers for its lifetime (or until restore()). A
    //    second writer on a claimed buffer throws: a second in-place
    //    transformation is not idempotent, it moves the data by rounding and
    //    downstream near-ties resolve differently. Readers (normalize = false,
    //    intercept = false) never reach this function and never claim.
    claim_x_ = bufreg::BufferClaim(X_->data(), writerClaimMessage("X"));
    if (D_.has_value() && num_dummies_ > 0) {
        claim_d_ = bufreg::BufferClaim(D_->data(), writerClaimMessage("D"));
    }

    // 1. Preprocess original predictors X
    su_preproc::preprocessMatStatistics(*X_, intercept_, normalize_,
        eps_, meansx_, normsx_, dropped_indices_, scaling_mode_,
        threadsFor(static_cast<double>(X_->rows()) * static_cast<double>(X_->cols())));

    // 2. Preprocess dummy predictors D (Offset dropped indices)
    if (D_.has_value() && num_dummies_ > 0) {
        std::vector<std::size_t> d_dropped;
        su_preproc::preprocessMatStatistics(*D_, intercept_, normalize_,
            eps_, meansd_, normsd_, d_dropped, scaling_mode_,
            threadsFor(static_cast<double>(D_->rows()) * static_cast<double>(D_->cols())));

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

    // 5. The statistics just computed say whether the buffers were already
    //    centered and unit-norm before this solver touched them (a
    //    pre-scaled pipeline that should run as constructed, or a previous
    //    writer that was destroyed without restore()). A warning, not an
    //    error: standardized raw data are legitimate. No extra pass.
    warnIfAlreadyPreprocessed("X", meansx_, normsx_);
    if (D_.has_value() && num_dummies_ > 0) {
        warnIfAlreadyPreprocessed("D", meansd_, normsd_);
    }

    // From here on this object is responsible for writing back.
    holds_preprocessed_.value = true;
}


TSolver_Base::~TSolver_Base() {
    // A writer that was never restore()d — including a derived constructor
    // that threw after this base had preprocessed — writes X and D back
    // here, so the caller's buffers never stay preprocessed without an
    // owner (the T-Rex selector does the same). No-op for readers.
    writeBackOwnedBuffers();
}


void TSolver_Base::writeBackOwnedBuffers() const noexcept {
    if (!holds_preprocessed_.value) { return; }
    try {
        // The stored maps view the caller's writable buffers (the constructor
        // received them as non-const maps); the const_cast only undoes the
        // constness of this method, which restore() has always been.
        if (X_.has_value()) {
            Eigen::Map<Eigen::MatrixXd> xm(const_cast<double*>(X_->data()),
                                           X_->rows(), X_->cols());
            su_preproc::restoreMatStatistics(
                xm, intercept_, meansx_, normalize_, normsx_,
                threadsFor(static_cast<double>(xm.rows()) * static_cast<double>(xm.cols())));
        }
        if (D_.has_value() && num_dummies_ > 0) {
            Eigen::Map<Eigen::MatrixXd> dm(const_cast<double*>(D_->data()),
                                           D_->rows(), D_->cols());
            su_preproc::restoreMatStatistics(
                dm, intercept_, meansd_, normalize_, normsd_,
                threadsFor(static_cast<double>(dm.rows()) * static_cast<double>(dm.cols())));
        }
    } catch (...) {
        // never throw from the destructor path; the claims are released below
    }
    // The buffers are back in the caller's state: hand them over (a later
    // writer may claim them).
    claim_x_.release();
    claim_d_.release();
    holds_preprocessed_.value = false;
}


void TSolver_Base::restore(Eigen::Map<Eigen::MatrixXd>& X,
                           Eigen::Map<Eigen::MatrixXd>& D,
                           Eigen::Map<Eigen::VectorXd>& y) const {

    // Idempotent, and a no-op for readers and resumed solvers: only the
    // object whose preprocessing is in effect writes back.
    if (!holds_preprocessed_.value) { return; }

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

    // The passed maps must be the buffers this solver was constructed on;
    // the write-back goes through the stored maps (same memory).
    if (X.data() != X_->data()) {
        throw std::invalid_argument(
            concatMsg(solverTypeToString(),
                      "::restore: X is not the matrix this solver was constructed on.")
        );
    }
    if (num_dummies_ > 0 && D_.has_value() && D.data() != D_->data()) {
        throw std::invalid_argument(
            concatMsg(solverTypeToString(),
                      "::restore: D is not the matrix this solver was constructed on.")
        );
    }

    // NOTE: y is NOT modified. The constructor copies the response into y_
    // and centers only the internal copy, so the caller's y buffer was never
    // preprocessed in place — adding mu_y_ back would corrupt it.
    writeBackOwnedBuffers();
}


std::string TSolver_Base::writerClaimMessage(const char* which) const {
    return concatMsg(
        solverTypeToString(), ": the input matrix ", which,
        " is already held by another live T-solver or T-Rex selector that "
        "preprocessed it in place. A second in-place preprocessing is not "
        "idempotent (it moves the data by rounding) and would corrupt both. "
        "Pre-scale once in front and construct the solvers with normalize = "
        "false and intercept = false (readers never claim a buffer), or "
        "restore() the previous solver, or let it go out of scope first.");
}


void TSolver_Base::warnIfAlreadyPreprocessed(const char* which,
                                             const Eigen::VectorXd& means,
                                             const Eigen::VectorXd& norms) const {
    constexpr double tol = 1e-10;
    if (intercept_) {
        if (means.size() == 0 || (means.array().abs() > tol).any()) { return; }
    }
    if (normalize_) {
        if (norms.size() == 0 || ((norms.array() - 1.0).abs() > tol).any()) { return; }
    }
    logWarning(concatMsg(
        "the input matrix ", which, " was already ",
        (intercept_ && normalize_) ? "centered and unit-norm"
                                   : (intercept_ ? "centered" : "unit-norm"),
        " (every column within 1e-10) before this solver preprocessed it in "
        "place; the second transformation moves the data by rounding. If ",
        which, " was pre-scaled, construct the solver with normalize = false "
        "and intercept = false (run as constructed); if a previous solver "
        "preprocessed it, call restore() on that solver first."));
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

    // A serial O(p_tot) scan: the hash lookups never pay a fork-join.
    inactives_.reserve(p_tot);
    for (std::size_t j = 0; j < p_tot; ++j) {
        if (!active_set.contains(j) && !dropped_set.contains(j)) {
            inactives_.push_back(j);
        }
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
    [[maybe_unused]] const int nt = threadsFor(static_cast<double>(X_->rows()) *
                              static_cast<double>(inactives_.size()));
    #pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
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
    [[maybe_unused]] const int nt = threadsFor(static_cast<double>(X_->rows()) *
                              static_cast<double>(inactives_.size()));
    #pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
    for (std::size_t i = 0; i < inactives_.size(); ++i) {
        std::size_t j = inactives_[i];
        correlations_(static_cast<Eigen::Index>(j)) = getColumn(j).dot(r_);
    }
}


void TSolver_Base::refreshAllCorrelations() {
    std::size_t p_tot = p_original_ + num_dummies_;
    std::unordered_set<std::size_t> dropped_set(dropped_indices_.begin(),
                                                dropped_indices_.end());
    [[maybe_unused]] const int nt = threadsFor(static_cast<double>(X_->rows()) * static_cast<double>(p_tot));
    #pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
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


void TSolver_Base::applyExchangeableTieBreak(std::vector<std::size_t>& new_vars) {
    if (exch_tie_alpha_ <= 0.0 || new_vars.empty()) { return; }

    // Top candidate among the step's proposed additions.
    std::size_t jstar = new_vars[0];
    double cmax = std::abs(correlations_(static_cast<Eigen::Index>(jstar)));
    for (std::size_t j : new_vars) {
        const double a = std::abs(correlations_(static_cast<Eigen::Index>(j)));
        if (a > cmax) { cmax = a; jstar = j; }
    }
    if (jstar >= dummy_start_idx_) { return; }  // dummy pick stays deterministic

    // Pairwise ranking-noise scale: the gap c_j* - c_k = (x_j* - x_k)^T r has
    // sd ~ ||x_j* -/+ x_k|| * ||r|| / sqrt(n) under exchangeability. The
    // sign-matched distance nj2 + nk2 - 2|dot| also covers anti-correlated
    // pairs (the |c| ranking is sign-blind).
    const Eigen::VectorXd xj = getColumn(jstar);
    const double nj2 = xj.squaredNorm();
    const double sig = r_.norm() /
        std::sqrt(static_cast<double>(std::max<Eigen::Index>(r_.size(), 1)));

    std::vector<std::size_t> tie_set{jstar};
    for (std::size_t k : inactives_) {
        if (k == jstar || k >= dummy_start_idx_) { continue; }
        const auto xk = getColumn(k);
        const double nk2   = xk.squaredNorm();
        const double dot   = xj.dot(xk);
        const double denom = std::sqrt(nj2 * nk2);
        if (denom <= eps_) { continue; }
        if (std::abs(dot) / denom < exch_tie_floor_) { continue; }
        const double band = exch_tie_alpha_ * sig *
            std::sqrt(std::max(0.0, nj2 + nk2 - 2.0 * std::abs(dot)));
        const double gap =
            cmax - std::abs(correlations_(static_cast<Eigen::Index>(k)));
        if (gap <= band) { tie_set.push_back(k); }
    }
    if (tie_set.size() <= 1) { return; }

    std::uniform_int_distribution<std::size_t> pick(0, tie_set.size() - 1);
    new_vars.assign(1, tie_set[pick(rng_)]);
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
    std::size_t n = effective_n_ + interceptDoF();
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
    std::size_t n = effective_n_ + interceptDoF();
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


std::size_t TSolver_Base::memoryBytes() const noexcept {
    using solver_utils::memory_bytes::bytesOf;

    // X_ and D_ view the caller's buffers (borrowed); y_ is an own copy
    std::size_t bytes = bytesOf(y_) + bytesOf(r_) + bytesOf(correlations_) + bytesOf(R_);

    // Sparse path and the running coefficients
    bytes += bytesOf(betaPathSparse_, [](const SparseBetaStep& step) noexcept {
        return bytesOf(step.idx) + bytesOf(step.val);
    });
    bytes += bytesOf(beta_idx_) + bytesOf(beta_val_) + bytesOf(beta_pos_);

    // Diagnostics, index sets, dummy tracking
    bytes += bytesOf(RSS_) + bytesOf(R2_) + bytesOf(DoF_) + bytesOf(actions_) +
             bytesOf(warnings_);
    bytes += bytesOf(actives_) + bytesOf(inactives_) + bytesOf(dropped_indices_) +
             bytesOf(dummies_at_step_);

    // Writer statistics (empty for readers)
    bytes += bytesOf(meansx_) + bytesOf(normsx_) + bytesOf(meansd_) + bytesOf(normsd_);
    return bytes;
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
} /* End of namespace trex::tsolvers::ad */
// ============================================================================
