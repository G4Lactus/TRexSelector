// ============================================================================
// tcdienet_solver.cpp
// ============================================================================
/**
 * @file tcdienet_solver.cpp
 *
 * @brief Implementation of the Terminating CCD Informed Elastic Net
 * (T-CDIENET) solver class, extending the T-CDENET solver.
 */
// ============================================================================

// std includes
#include <algorithm>
#include <stdexcept>

// tsolvers includes
#include <tsolvers/ad/linear_model/cd_based/tcdienet_solver.hpp>
#include <tsolvers/ad/solver_utils/memory_bytes.hpp>

// ============================================================================

// Embedded into namespace trex::tsolvers::ad::linear_model::cd_based
namespace trex::tsolvers::ad::linear_model::cd_based {

// ============================================================================
// Group administration (GROUP kind)
// ============================================================================

void TCDIENET_Solver::assignGroupsLayered(const Eigen::VectorXi& groups) {
    const std::size_t p = p_original_;
    const std::size_t L = num_dummies_;

    if (static_cast<std::size_t>(groups.size()) != p) {
        throw std::invalid_argument(concatMsg(
            solverTypeToString(),
            "::assignGroupsLayered: groups length must equal p."));
    }
    Eigen::Index max_id = -1;
    for (Eigen::Index j = 0; j < groups.size(); ++j) {
        if (groups(j) < 0) {
            throw std::invalid_argument(concatMsg(
                solverTypeToString(),
                "::assignGroupsLayered: groups must be 0-based ids."));
        }
        if (groups(j) > max_id) { max_id = groups(j); }
    }
    const std::size_t M = static_cast<std::size_t>(max_id + 1);
    {
        std::vector<bool> seen(M, false);
        for (Eigen::Index j = 0; j < groups.size(); ++j) {
            seen[static_cast<std::size_t>(groups(j))] = true;
        }
        for (std::size_t m = 0; m < M; ++m) {
            if (!seen[m]) {
                throw std::invalid_argument(concatMsg(
                    solverTypeToString(),
                    "::assignGroupsLayered: group id ", m, " is missing."));
            }
        }
    }
    if (L > 0 && L % p != 0) {
        throw std::invalid_argument(concatMsg(
            solverTypeToString(),
            "::assignGroupsLayered: D.cols() must be a multiple of X.cols() "
            "(one group-aligned dummy layer per p columns)."));
    }

    group_of_.assign(p + L, -1);
    for (std::size_t j = 0; j < p; ++j) {
        group_of_[j] = groups(static_cast<Eigen::Index>(j));
    }
    for (std::size_t t = 0; t < L; ++t) {
        group_of_[p + t] = groups(static_cast<Eigen::Index>(t % p));
    }

    group_size_.assign(M, 0); // REAL sizes only
    for (std::size_t j = 0; j < p; ++j) {
        group_size_[static_cast<std::size_t>(group_of_[j])]++;
    }

    logInfo(concatMsg("[", solverTypeToString(), "] layered groups: M = ", M,
                      " over p = ", p, " predictors, ",
                      (p > 0 ? L / p : 0), " dummy layer(s)."));
}

// ============================================================================
// Specialized penalty hooks
// ============================================================================

void TCDIENET_Solver::ensureCdState() {
    TCDLASSO_Solver::ensureCdState();     // base caches + configurePenaltyDiag()

    if (penalty_kind_ != PenaltyKind::GROUP &&
        penalty_kind_ != PenaltyKind::GROUP_CENTERING) {
        return;
    }

    // Per-group constants lambda2 / p_m.
    const std::size_t M = group_size_.size();
    lambda2_over_pm_.assign(M, 0.0);
    for (std::size_t m = 0; m < M; ++m) {
        lambda2_over_pm_[m] = lambda2_ / static_cast<double>(group_size_[m]);
    }

    // Rebuild sigma_m = sum_{i in G_m} beta_i from the (possibly
    // deserialized) coefficients over all unified columns.
    group_sums_.assign(M, 0.0);
    for (std::size_t j = 0; j < group_of_.size(); ++j) {
        const double b = beta_cd_(static_cast<Eigen::Index>(j));
        if (b != 0.0) {
            group_sums_[static_cast<std::size_t>(group_of_[j])] += b;
        }
    }

    // GROUP_CENTERING additionally maintains the per-base folded sums
    // s_i = beta_i + sum of variable i's dummy-copy betas.
    if (penalty_kind_ == PenaltyKind::GROUP_CENTERING) {
        const auto p_idx = static_cast<Eigen::Index>(p_original_);
        folded_s_.setZero(p_idx);
        const std::size_t p_total = p_original_ + num_dummies_;
        for (std::size_t t = 0; t < p_total; ++t) {
            const double b = beta_cd_(static_cast<Eigen::Index>(t));
            if (b != 0.0) {
                folded_s_(static_cast<Eigen::Index>(foldedBase(t))) += b;
            }
        }
    }
}


void TCDIENET_Solver::configurePenaltyDiag() {
    if (penalty_kind_ != PenaltyKind::GROUP &&
        penalty_kind_ != PenaltyKind::GROUP_CENTERING) {
        TCDLASSO_Solver::configurePenaltyDiag();
        return;
    }
    const std::size_t p_total = p_original_ + num_dummies_;
    penalty_diag_.setZero(static_cast<Eigen::Index>(p_total));
    if (lambda2_ <= 0.0) { return; }

    if (group_of_.size() != p_total) {
        throw std::logic_error(concatMsg(
            solverTypeToString(),
            "::configurePenaltyDiag: groups not assigned (size ",
            group_of_.size(), ", expected ", p_total, ")."));
    }
    // GROUP: diag = lambda2 * W_jj = lambda2 / p_m.
    // GROUP_CENTERING: diag = lambda2 * (I - W)_jj = lambda2 * (1 - 1/p_m).
    const bool centering = (penalty_kind_ == PenaltyKind::GROUP_CENTERING);
    for (std::size_t j = 0; j < p_total; ++j) {
        const double inv_pm = 1.0 / static_cast<double>(
            group_size_[static_cast<std::size_t>(group_of_[j])]);
        penalty_diag_(static_cast<Eigen::Index>(j)) =
            centering ? lambda2_ * (1.0 - inv_pm) : lambda2_ * inv_pm;
    }
}


void TCDIENET_Solver::penaltyRankUpdate(std::size_t j, double delta) {
    if (penalty_kind_ == PenaltyKind::GROUP) {
        group_sums_[static_cast<std::size_t>(group_of_[j])] += delta;
        return;
    }
    if (penalty_kind_ == PenaltyKind::GROUP_CENTERING) {
        group_sums_[static_cast<std::size_t>(group_of_[j])] += delta;
        folded_s_(static_cast<Eigen::Index>(foldedBase(j))) += delta;
        return;
    }
    TCDLASSO_Solver::penaltyRankUpdate(j, delta);
}

// ============================================================================
// Certified-localization penalty hooks (unified index space)
// ============================================================================

double TCDIENET_Solver::penaltyPair(std::size_t i, std::size_t j) const {
    if (penalty_kind_ != PenaltyKind::GROUP &&
        penalty_kind_ != PenaltyKind::GROUP_CENTERING) {
        return TCDLASSO_Solver::penaltyPair(i, j);
    }
    if (lambda2_ <= 0.0) { return 0.0; }
    const int gi = group_of_[i];
    const int gj = group_of_[j];
    const double same_group = (gi == gj)
        ? lambda2_over_pm_[static_cast<std::size_t>(gi)] : 0.0;
    if (penalty_kind_ == PenaltyKind::GROUP) { return same_group; }
    // GROUP_CENTERING: lambda2 * ([base(i) == base(j)] - [same group] / p_m)
    const double same_base = (foldedBase(i) == foldedBase(j)) ? lambda2_ : 0.0;
    return same_base - same_group;
}


void TCDIENET_Solver::penaltyMatVec(const std::vector<std::size_t>& A,
                                    const Eigen::VectorXd& vA,
                                    Eigen::VectorXd& out) const {
    if (penalty_kind_ != PenaltyKind::GROUP &&
        penalty_kind_ != PenaltyKind::GROUP_CENTERING) {
        TCDLASSO_Solver::penaltyMatVec(A, vA, out);
        return;
    }
    const std::size_t p_total = p_original_ + num_dummies_;
    out.setZero(static_cast<Eigen::Index>(p_total));
    if (lambda2_ <= 0.0) { return; }

    // Group sums of v over A (dummies included), and folded base sums.
    const std::size_t M = group_size_.size();
    std::vector<double> sig(M, 0.0);
    Eigen::VectorXd sbase;
    const bool centering = (penalty_kind_ == PenaltyKind::GROUP_CENTERING);
    if (centering) { sbase.setZero(static_cast<Eigen::Index>(p_original_)); }
    for (std::size_t k = 0; k < A.size(); ++k) {
        const std::size_t j = A[k];
        const double vk = vA(static_cast<Eigen::Index>(k));
        sig[static_cast<std::size_t>(group_of_[j])] += vk;
        if (centering) { sbase(static_cast<Eigen::Index>(foldedBase(j))) += vk; }
    }
    for (std::size_t j = 0; j < p_total; ++j) {
        const std::size_t m = static_cast<std::size_t>(group_of_[j]);
        double v = lambda2_over_pm_[m] * sig[m];           // GROUP: lambda2 sigma_m / p_m
        if (centering) {
            v = lambda2_ * sbase(static_cast<Eigen::Index>(foldedBase(j))) - v;
        }
        out(static_cast<Eigen::Index>(j)) = v;
    }
}


void TCDIENET_Solver::zeroPenaltyState() {
    TCDLASSO_Solver::zeroPenaltyState();
    std::fill(group_sums_.begin(), group_sums_.end(), 0.0);
    if (folded_s_.size() > 0) { folded_s_.setZero(); }
}

// ============================================================================
// Memory report
// ============================================================================

std::size_t TCDIENET_Solver::memoryBytes() const noexcept {
    using solver_utils::memory_bytes::bytesOf;
    return TCDENET_Solver::memoryBytes() + bytesOf(group_of_) + bytesOf(group_size_) +
           bytesOf(group_sums_) + bytesOf(lambda2_over_pm_) + bytesOf(folded_s_);
}

// ============================================================================
// Serialization
// ============================================================================

void TCDIENET_Solver::save(const std::string& filename) const {
    saveImpl(*this, filename);
}


TCDIENET_Solver TCDIENET_Solver::load(const std::string& filename,
                                    Eigen::Map<Eigen::MatrixXd>& X,
                                    Eigen::Map<Eigen::MatrixXd>& D) {
    return loadImpl<TCDIENET_Solver>(filename, X, D);
}

// ============================================================================
}  /* End of namespace trex::tsolvers::ad::linear_model::cd_based */
