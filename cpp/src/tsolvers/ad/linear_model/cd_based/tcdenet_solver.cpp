// ============================================================================
// tcdenet_solver.cpp
// ============================================================================
/**
 * @file tcdenet_solver.cpp
 *
 * @brief Implementation of the Terminating CCD Elastic Net (T-CDENET) solver
 * class, extending the T-CDLASSO solver.
 */
// ============================================================================

// tsolvers includes
#include <tsolvers/ad/linear_model/cd_based/tcdenet_solver.hpp>

// ============================================================================

// Embedded into namespace trex::tsolvers::ad::linear_model::cd_based
namespace trex::tsolvers::ad::linear_model::cd_based {

// ============================================================================
// Serialization
// ============================================================================

void TCDENET_Solver::save(const std::string& filename) const { saveImpl(*this, filename); }


TCDENET_Solver TCDENET_Solver::load(const std::string& filename,
                                  Eigen::Map<Eigen::MatrixXd>& X,
                                  Eigen::Map<Eigen::MatrixXd>& D) {
    return loadImpl<TCDENET_Solver>(filename, X, D);
}

// ============================================================================
}  /* End of namespace trex::tsolvers::ad::linear_model::cd_based */
