// ========================================================================================
// test_sd_tsolvers.cpp
// ========================================================================================
/**
 * @file test_sd_tsolvers.cpp
 *
 * @brief Invariant tests for the SD (sparse-dummy) solver family. All
 *        invariants are two runs on identical fixed-seed data inside one
 *        binary, so they hold exactly (no tolerances on selections):
 *
 *        1. Twin equivalence: at k = 1 (rho_d = 2/n) the general solvers
 *           replicate the SD2 pair solvers' RNG draw pattern — same seed
 *           gives the identical selection path (LARS / OMP / AFS).
 *        2. AFS(rho = 1) == OMP: a full refit zeroes active correlations,
 *           so re-selection never fires (general and pair families).
 *        3. Resumability: executeStep(T1) then executeStep(T2) equals a
 *           fresh executeStep(T2) under the same seed.
 *        4. TRexSD determinism: same options + seed => same selection.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <random>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// SD solver includes
#include <tsolvers/sd_tsolvers/sd_tlars_solver.hpp>
#include <tsolvers/sd_tsolvers/sd2_tlars_solver.hpp>
#include <tsolvers/sd_tsolvers/sd_tomp_solver.hpp>
#include <tsolvers/sd_tsolvers/sd2_tomp_solver.hpp>
#include <tsolvers/sd_tsolvers/sd_tafs_solver.hpp>
#include <tsolvers/sd_tsolvers/sd2_tafs_solver.hpp>

// TRexSD include
#include <trex_selector_methods/trex_sd/trex_sd.hpp>

// ========================================================================================

namespace trex::test::tsolvers::sd {

using namespace trex::tsolvers;
namespace lars = trex::tsolvers::linear_model::lars_based;
namespace omp  = trex::tsolvers::linear_model::omp_based;
namespace afs  = trex::tsolvers::linear_model::afs_based;
namespace tsd  = trex::trex_selector_methods::trex_sd;

// ----------------------------------------------------------------------------------------
// Fixed-seed data on the SD solver contract: X centered with unit-L2 columns,
// y centered. Deterministic across runs (mt19937_64 is a portable generator).
// ----------------------------------------------------------------------------------------
struct SDData {
    Eigen::MatrixXd X;
    Eigen::VectorXd y;

    SDData(int n, int p, unsigned long long seed) {
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> N(0.0, 1.0);

        X.resize(n, p);
        for (int j = 0; j < p; ++j)
            for (int i = 0; i < n; ++i) X(i, j) = N(rng);
        for (int j = 0; j < p; ++j) {
            X.col(j).array() -= X.col(j).mean();
            X.col(j) /= X.col(j).norm();
        }

        // y = sum of the first three predictors + mild noise, centered
        y = X.col(0) + X.col(1) + X.col(2);
        for (int i = 0; i < n; ++i) y(i) += 0.3 * N(rng);
        y.array() -= y.mean();
    }

    Eigen::Map<Eigen::MatrixXd> Xmap() {
        return Eigen::Map<Eigen::MatrixXd>(X.data(), X.rows(), X.cols());
    }
    Eigen::Map<Eigen::VectorXd> ymap() {
        return Eigen::Map<Eigen::VectorXd>(y.data(), y.size());
    }
};

constexpr int          kN    = 120;
constexpr int          kP    = 60;
constexpr std::size_t  kLMax = 120;   // 2p
constexpr std::size_t  kT    = 3;
constexpr uint64_t     kSeed = 20260714ULL;

// rho_d giving exactly k = 1 (one +1/-1 pair): 2k = 2*floor(n*rho_d/2) = 2.
inline double pairRho() { return 2.0 / static_cast<double>(kN); }

// ========================================================================================
// 1. Twin equivalence: general solver at k = 1 == SD2 pair solver (OnDemand)
// ========================================================================================

TEST(SDTwinEquivalence, LarsPairTwinIdenticalSelectionPath) {
    SDData data(kN, kP, 1);
    auto Xm1 = data.Xmap(); auto ym1 = data.ymap();
    auto Xm2 = data.Xmap(); auto ym2 = data.ymap();

    lars::SD_TLARS_Solver general(Xm1, ym1, pairRho(), kLMax, kT, true, kSeed);
    lars::SD2_TLARS_Solver pair(Xm2, ym2, kLMax, kT, true, kSeed,
                                SD2GenPolicy::OnDemand);

    general.executeStep(kT, true);
    pair.executeStep(kT, true);

    EXPECT_EQ(general.sparsityK(), 1u);
    EXPECT_EQ(general.getActives(), pair.getActives());
    EXPECT_EQ(general.getNumSteps(), pair.getNumSteps());
    EXPECT_EQ(general.getNumActiveDummies(), pair.getNumActiveDummies());
}

TEST(SDTwinEquivalence, OmpPairTwinIdenticalSelectionPath) {
    SDData data(kN, kP, 2);
    auto Xm1 = data.Xmap(); auto ym1 = data.ymap();
    auto Xm2 = data.Xmap(); auto ym2 = data.ymap();

    omp::SD_TOMP_Solver general(Xm1, ym1, pairRho(), kLMax, kT, true, kSeed);
    omp::SD2_TOMP_Solver pair(Xm2, ym2, kLMax, kT, true, kSeed,
                              SD2GenPolicy::OnDemand);

    general.executeStep(kT, true);
    pair.executeStep(kT, true);

    EXPECT_EQ(general.getActives(), pair.getActives());
    EXPECT_EQ(general.getNumActiveDummies(), pair.getNumActiveDummies());
}

TEST(SDTwinEquivalence, AfsPairTwinIdenticalSelectionPath) {
    SDData data(kN, kP, 3);
    auto Xm1 = data.Xmap(); auto ym1 = data.ymap();
    auto Xm2 = data.Xmap(); auto ym2 = data.ymap();

    afs::SD_TAFS_Solver general(Xm1, ym1, pairRho(), kLMax, kT, true, kSeed, 0.3);
    afs::SD2_TAFS_Solver pair(Xm2, ym2, kLMax, kT, true, kSeed,
                              SD2GenPolicy::OnDemand, 0.3);

    general.executeStep(kT, true);
    pair.executeStep(kT, true);

    EXPECT_EQ(general.getActives(), pair.getActives());
    EXPECT_EQ(general.getNumActiveDummies(), pair.getNumActiveDummies());
}

// ========================================================================================
// 2. AFS(rho = 1) reduces to OMP exactly
// ========================================================================================

TEST(SDAfsOmpReduction, GeneralFamilyRhoOneEqualsOmp) {
    SDData data(kN, kP, 4);
    auto Xm1 = data.Xmap(); auto ym1 = data.ymap();
    auto Xm2 = data.Xmap(); auto ym2 = data.ymap();

    const double rho_d = 0.1;
    afs::SD_TAFS_Solver afs_solver(Xm1, ym1, rho_d, kLMax, kT, true, kSeed,
                                   /*rho=*/1.0);
    omp::SD_TOMP_Solver omp_solver(Xm2, ym2, rho_d, kLMax, kT, true, kSeed);

    afs_solver.executeStep(kT, true);
    omp_solver.executeStep(kT, true);

    EXPECT_EQ(afs_solver.getActives(), omp_solver.getActives());
    EXPECT_EQ(afs_solver.getNumActiveDummies(),
              omp_solver.getNumActiveDummies());
}

TEST(SDAfsOmpReduction, PairFamilyRhoOneEqualsOmp) {
    SDData data(kN, kP, 5);

    for (SD2GenPolicy policy : {SD2GenPolicy::OnDemand, SD2GenPolicy::Geometric}) {
        auto Xm1 = data.Xmap(); auto ym1 = data.ymap();
        auto Xm2 = data.Xmap(); auto ym2 = data.ymap();

        afs::SD2_TAFS_Solver afs_solver(Xm1, ym1, kLMax, kT, true, kSeed,
                                        policy, /*rho=*/1.0);
        omp::SD2_TOMP_Solver omp_solver(Xm2, ym2, kLMax, kT, true, kSeed,
                                        policy);

        afs_solver.executeStep(kT, true);
        omp_solver.executeStep(kT, true);

        EXPECT_EQ(afs_solver.getActives(), omp_solver.getActives())
            << "policy=" << static_cast<int>(policy);
    }
}

// ========================================================================================
// 3. Resumability: incremental T equals fresh full run
// ========================================================================================

TEST(SDResumability, LarsIncrementalEqualsFresh) {
    SDData data(kN, kP, 6);
    auto Xm1 = data.Xmap(); auto ym1 = data.ymap();
    auto Xm2 = data.Xmap(); auto ym2 = data.ymap();

    const double rho_d = 0.1;
    lars::SD_TLARS_Solver fresh(Xm1, ym1, rho_d, kLMax, kT, true, kSeed);
    fresh.executeStep(kT, true);

    lars::SD_TLARS_Solver resumed(Xm2, ym2, rho_d, kLMax, kT, true, kSeed);
    resumed.executeStep(1, true);
    resumed.executeStep(kT, true);

    EXPECT_EQ(fresh.getActives(), resumed.getActives());
    EXPECT_EQ(fresh.getNumSteps(), resumed.getNumSteps());
    EXPECT_EQ(fresh.getNumActiveDummies(), resumed.getNumActiveDummies());
}

TEST(SDResumability, OmpIncrementalEqualsFresh) {
    SDData data(kN, kP, 7);
    auto Xm1 = data.Xmap(); auto ym1 = data.ymap();
    auto Xm2 = data.Xmap(); auto ym2 = data.ymap();

    const double rho_d = 0.1;
    omp::SD_TOMP_Solver fresh(Xm1, ym1, rho_d, kLMax, kT, true, kSeed);
    fresh.executeStep(kT, true);

    omp::SD_TOMP_Solver resumed(Xm2, ym2, rho_d, kLMax, kT, true, kSeed);
    resumed.executeStep(1, true);
    resumed.executeStep(kT, true);

    EXPECT_EQ(fresh.getActives(), resumed.getActives());
    EXPECT_EQ(fresh.getNumActiveDummies(), resumed.getNumActiveDummies());
}

// ========================================================================================
// 4. Signal recovery sanity + TRexSD determinism
// ========================================================================================

TEST(SDSelection, LarsRecoversPlantedSupportBeforeDummies) {
    SDData data(kN, kP, 8);
    auto Xm = data.Xmap(); auto ym = data.ymap();

    lars::SD_TLARS_Solver solver(Xm, ym, 0.1, kLMax, kT, true, kSeed);
    solver.executeStep(kT, true);

    // The three planted predictors carry equal strong signal: they must all
    // be selected (dummies race the same argmax, so this also guards the
    // fair-race scaling).
    std::vector<std::size_t> selected = solver.getSelectedOriginals();
    for (std::size_t j : {0u, 1u, 2u}) {
        EXPECT_TRUE(std::find(selected.begin(), selected.end(), j)
                    != selected.end())
            << "planted predictor " << j << " not selected";
    }
}

TEST(SDSelection, TRexSDDeterministicUnderSeed) {
    SDData data(kN, kP, 9);

    tsd::TRexSDOptions opts;
    opts.tFDR      = 0.2;
    opts.K         = 5;
    opts.rho_d     = 0.1;          // explicit: skip the calibration MC
    opts.calib     = tsd::CalibMode::CalibrateT;
    opts.L_factor  = 1;
    opts.verbose   = false;
    opts.seed      = kSeed;
    opts.n_threads = 1;

    tsd::TRexSD selector1(opts);
    tsd::TRexSD selector2(opts);
    tsd::TRexSDResult r1 = selector1.run(data.X, data.y);
    tsd::TRexSDResult r2 = selector2.run(data.X, data.y);

    ASSERT_EQ(r1.selected_var.size(), r2.selected_var.size());
    for (Eigen::Index i = 0; i < r1.selected_var.size(); ++i) {
        EXPECT_EQ(r1.selected_var(i), r2.selected_var(i));
    }
    EXPECT_EQ(r1.T_stop, r2.T_stop);
    EXPECT_DOUBLE_EQ(r1.v_thresh, r2.v_thresh);
}

}  // namespace trex::test::tsolvers::sd
