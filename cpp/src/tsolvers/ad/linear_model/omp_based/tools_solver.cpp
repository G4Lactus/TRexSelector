// ===================================================================================
// tools_solver.cpp
// ===================================================================================

// std includes
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

// tsolvers includes
#include <tsolvers/ad/linear_model/omp_based/tools_solver.hpp>
#include <tsolvers/ad/solver_utils/memory_bytes.hpp>

// utils includes
#include <utils/openmp/utils_openmp.hpp>

// ==================================================================================

namespace trex::tsolvers::ad::linear_model::omp_based {

// ==================================================================================
// Constructor Implementation
// ==================================================================================

TOOLS_Solver::TOOLS_Solver(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D,
    Eigen::Map<Eigen::VectorXd>& y,
    const SolverOptions& options)
    : TOMP_Solver(X, D, y, options, SolverTypeOMPBased::TOOLS)
{
    // Empty active set: the projected norms are the column norms (after the
    // base constructor's preprocessing).
    const auto p_tot = static_cast<Eigen::Index>(p_original_ + num_dummies_);
    proj_sq_.resize(p_tot);
    [[maybe_unused]] const int nt = threadsFor(static_cast<double>(X_->rows()) * static_cast<double>(p_tot));
    #pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
    for (Eigen::Index j = 0; j < p_tot; ++j) {
        proj_sq_(j) = getColumn(static_cast<std::size_t>(j)).squaredNorm();
    }
    proj_rank_ = 0;
}

TOOLS_Solver::TOOLS_Solver() : TOMP_Solver() {}

// ============================================================================
// Core Algorithm Implementation
// ============================================================================

void TOOLS_Solver::updateCorrelations() {
    const std::size_t m = actives_.size();
    const auto n = static_cast<Eigen::Index>(r_.size());
    const auto n_inactive = static_cast<Eigen::Index>(inactives_.size());

    // One pass per new active position k: q = X_A R^{-1} e_k is the new
    // orthonormal direction of the active span (R_ is the Cholesky factor of
    // X_A^T X_A, prefix-consistent), and ||P_perp x_j||^2 loses (q^T x_j)^2.
    // The correlation refresh X^T r rides on the pass of the last position.
    Eigen::VectorXd z;
    Eigen::VectorXd q(n);
    [[maybe_unused]] const int nt = threadsFor(static_cast<double>(n) * static_cast<double>(n_inactive));
    for (std::size_t k = proj_rank_; k < m; ++k) {
        const auto kk = static_cast<Eigen::Index>(k);
        z = Eigen::VectorXd::Zero(kk + 1);
        z(kk) = 1.0;
        R_.topLeftCorner(kk + 1, kk + 1).triangularView<Eigen::Upper>().solveInPlace(z);
        q.setZero();
        for (Eigen::Index i = 0; i <= kk; ++i) {
            q.noalias() += z(i) * getColumn(actives_[static_cast<std::size_t>(i)]);
        }

        const bool refresh = (k + 1 == m);
        #pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
        for (Eigen::Index i = 0; i < n_inactive; ++i) {
            const std::size_t j = inactives_[static_cast<std::size_t>(i)];
            const auto jj = static_cast<Eigen::Index>(j);
            const auto xj = getColumn(j);
            const double d = xj.dot(q);
            proj_sq_(jj) -= d * d;
            if (refresh) { correlations_(jj) = xj.dot(r_); }
        }
    }

    // No new position (every candidate of the step was rejected as
    // collinear): the residual is unchanged, refresh as T-OMP does.
    if (proj_rank_ == m) { refreshInactiveCorrelations(); }
    proj_rank_ = m;
}


std::pair<double, std::vector<std::size_t>> TOOLS_Solver::findTiedMaxCorrelations() const {
    double Cmax = 0.0;
    std::vector<std::size_t> tied;

    if (inactives_.empty()) return {Cmax, tied};

    Eigen::VectorXd scores(inactives_.size());
    const bool empty_active_set = actives_.empty();

    // OLS selection score of every inactive feature from the downdated
    // projected norms (an O(m) loop: serial)
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(inactives_.size()); ++i) {
        Eigen::Index j = static_cast<Eigen::Index>(inactives_[i]);
        double corr = correlations_(j);

        if (empty_active_set) {
            double nrm = getColumn(j).norm();
            scores(i) = (nrm > eps_) ? std::abs(corr) / nrm : -1.0;
            continue;
        }

        // Score only if the candidate is not numerically collinear
        const double orth_sq = proj_sq_(j);
        if (orth_sq < eps_) {
            scores(i) = -1.0;
        } else {
            scores(i) = std::abs(corr) / std::sqrt(orth_sq);
        }
    }

    // Single-threaded reduction to find max and ties
    tied.reserve(10);
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(inactives_.size()); ++i) {
        if (scores(i) > Cmax + eps_) {
            Cmax = scores(i);
            tied.clear();
            tied.push_back(inactives_[i]);
        } else if (scores(i) >= Cmax - eps_ && scores(i) > 0.0) {
            tied.push_back(inactives_[i]);
        }
    }

    return {Cmax, tied};
}

// ==================================================================================
// Memory report
// ==================================================================================

std::size_t TOOLS_Solver::memoryBytes() const noexcept {
    return TOMP_Solver::memoryBytes() + solver_utils::memory_bytes::bytesOf(proj_sq_);
}

// ==================================================================================
// Serialization
// ==================================================================================

void TOOLS_Solver::save(const std::string& filename) const { saveImpl(*this, filename); }

TOOLS_Solver TOOLS_Solver::load(const std::string& filename,
                              Eigen::Map<Eigen::MatrixXd>& X,
                              Eigen::Map<Eigen::MatrixXd>& D) {
    return loadImpl<TOOLS_Solver>(filename, X, D);
}

// ==================================================================================
} /* End of namespace trex::tsolvers::ad::linear_model::omp_based */
// ==================================================================================
