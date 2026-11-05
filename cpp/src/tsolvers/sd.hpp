// ====================================================================
// sd.hpp
// ====================================================================
#ifndef TSOLVERS_SD_HPP
#define TSOLVERS_SD_HPP
// ====================================================================
/**
 * @file sd.hpp
 *
 * @brief Umbrella header for the SD (sparse-dummy) terminating solvers:
 *        both family bases, the six solvers, and the calibration helper.
 *
 * @details Family layering (see sd_tsolver_base.hpp):
 *          SDTSolver_Base -> SDGeneralSolver_Base -> SD_TLARS/TOMP/TAFS
 *                         -> SD2PairSolver_Base   -> SD2_TLARS/TOMP/TAFS
 */
// ====================================================================

#include <tsolvers/sd/sd_tsolver_base.hpp>
#include <tsolvers/sd/sd_general_base.hpp>
#include <tsolvers/sd/sd2_pair_base.hpp>

#include <tsolvers/sd/sd_tlars_solver.hpp>
#include <tsolvers/sd/sd_tomp_solver.hpp>
#include <tsolvers/sd/sd_tafs_solver.hpp>

#include <tsolvers/sd/sd2_tlars_solver.hpp>
#include <tsolvers/sd/sd2_tomp_solver.hpp>
#include <tsolvers/sd/sd2_tafs_solver.hpp>

#include <tsolvers/sd/sd_calibration.hpp>

// ====================================================================
#endif /* TSOLVERS_SD_HPP */
