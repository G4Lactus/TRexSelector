// ===================================================================================
// tools_solver.cpp
// ===================================================================================

// std includes
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

// tsolvers includes
#include <tsolvers/linear_model/omp_based/tools_solver.hpp>

// utils includes
#include <utils/openmp/utils_openmp.hpp>

// ==================================================================================

namespace trex::tsolvers::linear_model::omp_based {

// ==================================================================================
// Constructor Implementation
// ==================================================================================

TOOLS_Solver::TOOLS_Solver(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D,
    Eigen::Map<Eigen::VectorXd>& y,
    bool normalize,
    bool intercept,
    bool verbose)
    : TOMP_Solver(X, D, y, normalize, intercept, verbose)
{
    // State is fully initialized by the TOMP_Solver base constructor
}

TOOLS_Solver::TOOLS_Solver() : TOMP_Solver() {}

// ============================================================================
// Core Algorithm Implementation
// ============================================================================

std::pair<double, std::vector<std::size_t>> TOOLS_Solver::findTiedMaxCorrelations() const {
    double Cmax = 0.0;
    std::vector<std::size_t> tied;

    if (inactives_.empty()) return {Cmax, tied};

    Eigen::VectorXd scores(inactives_.size());
    Eigen::Index m = static_cast<Eigen::Index>(actives_.size());

    // Compute the OLS selection score for all inactive features in parallel
    #pragma omp parallel for schedule(static)
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(inactives_.size()); ++i) {
        Eigen::Index j = static_cast<Eigen::Index>(inactives_[i]);
        double corr = correlations_(j);

        if (m == 0) {
            double nrm = getColumn(j).norm();
            scores(i) = (nrm > eps_) ? std::abs(corr) / nrm : -1.0;
            continue;
        }

        // 1. Cross-correlations with the active set: h_j = Phi_Gamma^T * phi_j
        Eigen::VectorXd hj(m);
        for (Eigen::Index k = 0; k < m; ++k) {
            hj(k) = getColumn(actives_[k]).dot(getColumn(j));
        }

        // 2. Forward solve equivalent using upper-triangular R_: R_^T * w = h_j
        Eigen::VectorXd w = TSolver_Base::backsolveT(R_, hj);

        // 3. Compute the squared norm of the orthogonal projection
        double orth_sq = getColumn(j).squaredNorm() - w.squaredNorm();

        // 4. Calculate final score if not numerically collinear
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
// Serialization
// ==================================================================================

void TOOLS_Solver::save(const std::string& filename) const {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error(
            concatMsg(solverTypeToString(), "::save: Cannot open '", filename, "'")
        );
    }
    cereal::PortableBinaryOutputArchive oarchive(ofs);
    oarchive(*this);
    logInfo(concatMsg("[", solverTypeToString(), "] saved to '", filename, "'\n"));
}

TOOLS_Solver TOOLS_Solver::load(
    const std::string& filename,
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D
) {
    TOOLS_Solver tools;
    std::ifstream is(filename, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error(
            tools.concatMsg(tools.solverTypeToString(), "::load: Cannot open '", filename, "'")
        );
    }
    cereal::PortableBinaryInputArchive iarchive(is);
    iarchive(tools);

    tools.reconnect(X, D);
    return tools;
}

// ==================================================================================
} /* End of namespace trex::tsolvers::linear_model::omp_based */
// ==================================================================================
