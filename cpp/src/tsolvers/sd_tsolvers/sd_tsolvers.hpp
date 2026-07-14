// ====================================================================
// sd_tsolvers.hpp
// ====================================================================
#ifndef TSOLVERS_SD_TSOLVERS_HPP
#define TSOLVERS_SD_TSOLVERS_HPP
// ====================================================================
/**
 * @file sd_tsolvers.hpp
 *
 * @brief Umbrella header for the SD (sparse-dummy) terminating solvers:
 *        both family bases, the six solvers, and the calibration helper.
 *
 * @details Family layering (see sd_tsolver_base.hpp):
 *          SDTSolver_Base -> SDGeneralSolver_Base -> SD_TLARS/TOMP/TAFS
 *                         -> SD2PairSolver_Base   -> SD2_TLARS/TOMP/TAFS
 */
// ====================================================================

#include <tsolvers/sd_tsolvers/sd_tsolver_base.hpp>
#include <tsolvers/sd_tsolvers/sd_general_base.hpp>
#include <tsolvers/sd_tsolvers/sd2_pair_base.hpp>

#include <tsolvers/sd_tsolvers/sd_tlars_solver.hpp>
#include <tsolvers/sd_tsolvers/sd_tomp_solver.hpp>
#include <tsolvers/sd_tsolvers/sd_tafs_solver.hpp>

#include <tsolvers/sd_tsolvers/sd2_tlars_solver.hpp>
#include <tsolvers/sd_tsolvers/sd2_tomp_solver.hpp>
#include <tsolvers/sd_tsolvers/sd2_tafs_solver.hpp>

#include <tsolvers/sd_tsolvers/sd_calibration.hpp>

// ====================================================================
#endif /* TSOLVERS_SD_TSOLVERS_HPP */
