// ============================================================================
// tcdtiknet_solver.cpp
// ============================================================================
/**
 * @file tcdtiknet_solver.cpp
 *
 * @brief Implementation of the Terminating CCD Tikhonov Net (T-CDTIKNET)
 * solver class, extending the T-CDENET solver.
 */
// ============================================================================

// tsolvers includes
#include <tsolvers/ad/linear_model/cd_based/tcdtiknet_solver.hpp>
#include <tsolvers/ad/solver_utils/memory_bytes.hpp>

// ============================================================================

// Embedded into namespace trex::tsolvers::ad::linear_model::cd_based
namespace trex::tsolvers::ad::linear_model::cd_based {

// ============================================================================
// Specialized penalty hooks (FOLDED dummy coupling)
// ============================================================================

void TCDTIKNET_Solver::ensureCdState() {
    TCDLASSO_Solver::ensureCdState();     // base caches + configurePenaltyDiag()

    if (!fold_dummies_) { return; }   // INDEPENDENT_RIDGE: base machinery

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
}


void TCDTIKNET_Solver::configurePenaltyDiag() {
    if (!fold_dummies_) {
        TCDLASSO_Solver::configurePenaltyDiag();   // Tikhonov diag + dummy ridge
        return;
    }

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
}


void TCDTIKNET_Solver::penaltyRankUpdate(std::size_t j, double delta) {
    if (!fold_dummies_) {
        TCDLASSO_Solver::penaltyRankUpdate(j, delta);
        return;
    }

    // FOLDED coupling: s_base(j) += delta, maintained through
    // w += lambda2 * delta * K.col(base(j)) — O(nnz(K col)).
    const auto b = static_cast<Eigen::Index>(foldedBase(j));
    const double f = lambda2_ * delta;
    for (Eigen::SparseMatrix<double>::InnerIterator it(tikhonov_K_, b);
         it; ++it) {
        folded_w_(it.row()) += f * it.value();
    }
}

// ============================================================================
// Certified-localization penalty hooks (FOLDED dummy coupling)
// ============================================================================

double TCDTIKNET_Solver::penaltyPair(std::size_t i, std::size_t j) const {
    if (!fold_dummies_) { return TCDLASSO_Solver::penaltyPair(i, j); }
    if (lambda2_ <= 0.0) { return 0.0; }
    return lambda2_ * tikhonov_K_.coeff(static_cast<Eigen::Index>(foldedBase(i)),
                                        static_cast<Eigen::Index>(foldedBase(j)));
}


void TCDTIKNET_Solver::penaltyMatVec(const std::vector<std::size_t>& A,
                                     const Eigen::VectorXd& vA,
                                     Eigen::VectorXd& out) const {
    if (!fold_dummies_) { TCDLASSO_Solver::penaltyMatVec(A, vA, out); return; }
    const std::size_t p_total = p_original_ + num_dummies_;
    out.setZero(static_cast<Eigen::Index>(p_total));
    if (lambda2_ <= 0.0) { return; }

    // Folded base sums of v over A, then w = lambda2 K s and out_j = w_base(j).
    Eigen::VectorXd s = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(p_original_));
    for (std::size_t k = 0; k < A.size(); ++k) {
        s(static_cast<Eigen::Index>(foldedBase(A[k]))) += vA(static_cast<Eigen::Index>(k));
    }
    const Eigen::VectorXd w = lambda2_ * (tikhonov_K_ * s);
    for (std::size_t t = 0; t < p_total; ++t) {
        out(static_cast<Eigen::Index>(t)) = w(static_cast<Eigen::Index>(foldedBase(t)));
    }
}


void TCDTIKNET_Solver::zeroPenaltyState() {
    TCDLASSO_Solver::zeroPenaltyState();
    if (folded_w_.size() > 0) { folded_w_.setZero(); }
}

// ============================================================================
// Memory report
// ============================================================================

std::size_t TCDTIKNET_Solver::memoryBytes() const noexcept {
    return TCDENET_Solver::memoryBytes() + solver_utils::memory_bytes::bytesOf(folded_w_);
}

// ============================================================================
// Serialization
// ============================================================================

void TCDTIKNET_Solver::save(const std::string& filename) const {
    saveImpl(*this, filename);
}


TCDTIKNET_Solver TCDTIKNET_Solver::load(const std::string& filename,
                                        Eigen::Map<Eigen::MatrixXd>& X,
                                        Eigen::Map<Eigen::MatrixXd>& D) {
    return loadImpl<TCDTIKNET_Solver>(filename, X, D);
}

// ============================================================================
}  /* End of namespace trex::tsolvers::ad::linear_model::cd_based */
