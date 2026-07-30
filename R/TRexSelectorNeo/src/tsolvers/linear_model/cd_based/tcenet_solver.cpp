// ============================================================================
// tcenet_solver.cpp
// ============================================================================
/**
 * @file tcenet_solver.cpp
 *
 * @brief Implementation of the Terminating CCD Elastic Net (T-CENET) solver
 * class, extending the T-CCD solver.
 */
// ============================================================================

// tsolvers includes
#include <tsolvers/linear_model/cd_based/tcenet_solver.hpp>

// ============================================================================

// Embedded into namespace trex::tsolvers::linear_model::cd_based
namespace trex::tsolvers::linear_model::cd_based {

// ============================================================================
// Serialization
// ============================================================================

void TCENET_Solver::save(const std::string& filename) const { saveImpl(*this, filename); }


TCENET_Solver TCENET_Solver::load(const std::string& filename,
                                  Eigen::Map<Eigen::MatrixXd>& X,
                                  Eigen::Map<Eigen::MatrixXd>& D) {
    return loadImpl<TCENET_Solver>(filename, X, D);
}

// ============================================================================
}  /* End of namespace trex::tsolvers::linear_model::cd_based */
