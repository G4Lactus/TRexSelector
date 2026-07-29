// ============================================================================
// tcienet_solver.cpp
// ============================================================================
/**
 * @file tcienet_solver.cpp
 *
 * @brief Implementation of the Terminating CCD Informed Elastic Net
 * (T-CIENET) solver class, extending the T-CENET solver.
 */
// ============================================================================

// std includes
#include <stdexcept>

// tsolvers includes
#include <tsolvers/linear_model/cd_based/tcienet_solver.hpp>

// ============================================================================

// Embedded into namespace trex::tsolvers::linear_model::cd_based
namespace trex::tsolvers::linear_model::cd_based {

// ============================================================================
// Group administration (GROUP kind)
// ============================================================================

void TCIENET_Solver::assignGroupsLayered(const Eigen::VectorXi& groups) {
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

void TCIENET_Solver::ensureCdState() {
    TCCD_Solver::ensureCdState();     // base caches + configurePenaltyDiag()

    if (penalty_kind_ == PenaltyKind::SPARSE && fold_dummies_) {
        // Rebuild the folded coupling state w = lambda2 * K * s from the
        // (possibly deserialized) coefficients: s_i sums variable i's
        // coefficient and those of all its dummy copies.
        const auto p_idx = static_cast<Eigen::Index>(p_original_);
        Eigen::VectorXd s = Eigen::VectorXd::Zero(p_idx);
        const std::size_t p_total = p_original_ + num_dummies_;
        for (std::size_t t = 0; t < p_total; ++t) {
            const double b = beta_cd_(static_cast<Eigen::Index>(t));
            if (b != 0.0) {
                s(static_cast<Eigen::Index>(foldedBase(t))) += b;
            }
        }
        folded_w_ = lambda2_ * (tikhonov_K_ * s);
        return;
    }

    if (penalty_kind_ != PenaltyKind::GROUP) { return; }

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
}


void TCIENET_Solver::configurePenaltyDiag() {
    if (penalty_kind_ == PenaltyKind::SPARSE && fold_dummies_) {
        // FOLDED coupling: every unified column carries the diagonal of its
        // base variable, lambda2 * K_bb — dummies included (kappa_dummy_
        // is ignored; the dummy block is coupled through K, not ridged).
        const std::size_t p_total = p_original_ + num_dummies_;
        penalty_diag_.setZero(static_cast<Eigen::Index>(p_total));
        if (lambda2_ <= 0.0) { return; }

        Eigen::VectorXd kdiag(static_cast<Eigen::Index>(p_original_));
        for (std::size_t j = 0; j < p_original_; ++j) {
            kdiag(static_cast<Eigen::Index>(j)) =
                tikhonov_K_.coeff(static_cast<Eigen::Index>(j),
                                  static_cast<Eigen::Index>(j));
        }
        for (std::size_t t = 0; t < p_total; ++t) {
            penalty_diag_(static_cast<Eigen::Index>(t)) =
                lambda2_ * kdiag(static_cast<Eigen::Index>(foldedBase(t)));
        }
        return;
    }
    if (penalty_kind_ != PenaltyKind::GROUP) {
        TCCD_Solver::configurePenaltyDiag();   // Tikhonov diag + dummy ridge
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
    for (std::size_t j = 0; j < p_total; ++j) {
        penalty_diag_(static_cast<Eigen::Index>(j)) =
            lambda2_ / static_cast<double>(
                group_size_[static_cast<std::size_t>(group_of_[j])]);
    }
}


void TCIENET_Solver::penaltyRankUpdate(std::size_t j, double delta) {
    if (penalty_kind_ == PenaltyKind::GROUP) {
        group_sums_[static_cast<std::size_t>(group_of_[j])] += delta;
        return;
    }
    if (penalty_kind_ == PenaltyKind::SPARSE && fold_dummies_) {
        // FOLDED coupling: s_base(j) += delta, maintained through
        // w += lambda2 * delta * K.col(base(j)) — O(nnz(K col)).
        const auto b = static_cast<Eigen::Index>(foldedBase(j));
        const double f = lambda2_ * delta;
        for (Eigen::SparseMatrix<double>::InnerIterator it(tikhonov_K_, b);
             it; ++it) {
            folded_w_(it.row()) += f * it.value();
        }
        return;
    }
    TCCD_Solver::penaltyRankUpdate(j, delta);
}

// ============================================================================
// Serialization
// ============================================================================

void TCIENET_Solver::save(const std::string& filename) const {
    saveImpl(*this, filename);
}


TCIENET_Solver TCIENET_Solver::load(const std::string& filename,
                                    Eigen::Map<Eigen::MatrixXd>& X,
                                    Eigen::Map<Eigen::MatrixXd>& D) {
    return loadImpl<TCIENET_Solver>(filename, X, D);
}

// ============================================================================
}  /* End of namespace trex::tsolvers::linear_model::cd_based */
