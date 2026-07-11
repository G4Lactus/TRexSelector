// =================================================================================
// test_trex_vd.cpp
// =================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen include
#include <Eigen/Dense>

// std includes
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

// project trex includes
#include <trex_selector_methods/trex_vd/trex_vd.hpp>

// =================================================================================

// Embed into test namespace
namespace trex::test::trex_selector_methods::trex_vd {

// Namespace aliases for convenience
using namespace trex::trex_selector_methods::trex_vd;
namespace vds = trex::tsolvers::vd;

// =================================================================================
// Test fixture: sparse linear model with preprocessed design
// =================================================================================

class TRexVDTest : public ::testing::Test {
protected:
    // Problem sizes kept small so the whole suite stays fast.
    static constexpr int n_ = 120;
    static constexpr int p_ = 80;
    static constexpr int s_ = 5;      // true support: features 0..4

    MatC X_;
    Vec  y_;
    Vec  beta_true_;

    void SetUp() override {
        std::mt19937_64 rng(42);
        std::normal_distribution<double> N01(0.0, 1.0);

        X_.resize(n_, p_);
        for (int j = 0; j < p_; ++j)
            for (int i = 0; i < n_; ++i)
                X_(i, j) = N01(rng);

        // Preprocessing contract of the VD solvers:
        // X column-centered and unit-L2 normalized, y centered.
        for (int j = 0; j < p_; ++j) {
            X_.col(j).array() -= X_.col(j).mean();
            const double nrm = X_.col(j).norm();
            if (nrm > 1e-12) X_.col(j) /= nrm;
        }

        beta_true_ = Vec::Zero(p_);
        for (int j = 0; j < s_; ++j) beta_true_(j) = 1.0;

        y_ = X_ * beta_true_;
        for (int i = 0; i < n_; ++i) y_(i) += 0.1 * N01(rng);
        y_.array() -= y_.mean();
    }
};

// =================================================================================
// VD solver smoke tests
// =================================================================================

TEST_F(TRexVDTest, VDLarsRunsAndRealizesDummies) {
    vds::VDOptions o;
    o.T_stop = 10;
    o.seed = 123;
    o.dummy_law = vds::VDDummyLaw::Spherical;

    vds::VD_LARS solver(X_, y_, /*num_dummies=*/5 * p_, o);
    solver.run(/*T=*/3);

    EXPECT_EQ(solver.n_samples(), n_);
    EXPECT_EQ(solver.n_features(), p_);
    EXPECT_EQ(solver.num_dummies(), 5 * p_);
    EXPECT_GE(solver.num_realized_dummies(), 3);
    EXPECT_LE(solver.num_realized_dummies(), o.T_stop);

    // The strong true support should enter before 3 dummies are realized.
    Vec beta = solver.beta_real();
    int true_hits = 0;
    for (int j = 0; j < s_; ++j)
        if (std::abs(beta(j)) > 1e-12) ++true_hits;
    EXPECT_GE(true_hits, s_ - 1);
}

TEST_F(TRexVDTest, VDOmpAndAfsRun) {
    vds::VDOptions o;
    o.T_stop = 5;
    o.seed = 7;

    vds::VD_OMP omp_solver(X_, y_, 3 * p_, o);
    omp_solver.run(2);
    EXPECT_GE(omp_solver.num_realized_dummies(), 2);

    o.rho = 0.7;
    vds::VD_AFS afs_solver(X_, y_, 3 * p_, o);
    afs_solver.run(2);
    EXPECT_GE(afs_solver.num_realized_dummies(), 2);
}

TEST_F(TRexVDTest, VDLarsIsDeterministicForFixedSeed) {
    vds::VDOptions o;
    o.T_stop = 5;
    o.seed = 99;

    vds::VD_LARS s1(X_, y_, 3 * p_, o);
    vds::VD_LARS s2(X_, y_, 3 * p_, o);
    s1.run(2);
    s2.run(2);

    EXPECT_TRUE(s1.beta_real().isApprox(s2.beta_real()));
    EXPECT_EQ(s1.num_realized_dummies(), s2.num_realized_dummies());
}

// =================================================================================
// TRexVD end-to-end smoke tests
// =================================================================================

TEST_F(TRexVDTest, EndToEndCalibrateTRecoversSupport) {
    TRexVDOptions opt;
    opt.tFDR = 0.2;
    opt.K = 10;
    opt.L_factor = 5;
    opt.solver = SolverType::LARS;
    opt.calib = CalibMode::CalibrateT;
    opt.dummy_law = VDDummyLaw::Spherical;
    opt.seed = 42;
    opt.verbose = false;

    TRexVD selector(opt);
    TRexVDResult result = selector.run(X_, y_);

    // The 5 strong true features should be selected; with tFDR = 0.2 the
    // selection may carry at most a couple of extras on this easy problem.
    ASSERT_GE(result.selected_var.size(), s_);
    std::vector<int> sel(result.selected_var.data(),
                         result.selected_var.data() + result.selected_var.size());
    for (int j = 0; j < s_; ++j)
        EXPECT_TRUE(std::find(sel.begin(), sel.end(), j) != sel.end())
            << "true support feature " << j << " missing from selection";

    const int false_hits = (int)sel.size() - s_;
    EXPECT_LE(false_hits, 3);

    // Result bookkeeping
    EXPECT_GT(result.T_stop, 0);
    EXPECT_EQ(result.num_dummies, opt.L_factor * p_);
    EXPECT_EQ(result.K, opt.K);
    EXPECT_GT(result.v_thresh, 0.5);
    EXPECT_EQ(result.Phi_mat.cols(), p_);
    EXPECT_EQ(result.FDP_hat_mat.rows(), result.T_stop);
}

TEST_F(TRexVDTest, PhiPrimeIsClampedToUnitInterval) {
    TRexVDOptions opt;
    opt.tFDR = 0.2;
    opt.K = 10;
    opt.L_factor = 5;
    opt.calib = CalibMode::CalibrateT;
    opt.seed = 42;
    opt.verbose = false;

    TRexVD selector(opt);
    TRexVDResult result = selector.run(X_, y_);

    // Deliberate deviation from the vd_selectors reference (kept in line with
    // trex_core::computePhiPrime): Phi_prime must stay inside [0, 1].
    ASSERT_GT(result.Phi_prime.size(), 0);
    EXPECT_GE(result.Phi_prime.minCoeff(), 0.0);
    EXPECT_LE(result.Phi_prime.maxCoeff(), 1.0);
}

TEST_F(TRexVDTest, EndToEndIsDeterministicForFixedSeed) {
    TRexVDOptions opt;
    opt.tFDR = 0.2;
    opt.K = 8;
    opt.L_factor = 4;
    opt.calib = CalibMode::CalibrateT;
    opt.seed = 1234;
    opt.verbose = false;
    opt.n_threads = 1;  // single-threaded for a bitwise-reproducible run

    TRexVD s1(opt);
    TRexVD s2(opt);
    TRexVDResult r1 = s1.run(X_, y_);
    TRexVDResult r2 = s2.run(X_, y_);

    ASSERT_EQ(r1.selected_var.size(), r2.selected_var.size());
    for (int i = 0; i < (int)r1.selected_var.size(); ++i)
        EXPECT_EQ(r1.selected_var(i), r2.selected_var(i));
    EXPECT_DOUBLE_EQ(r1.v_thresh, r2.v_thresh);
    EXPECT_EQ(r1.T_stop, r2.T_stop);
}

TEST_F(TRexVDTest, FixedTLModeRuns) {
    TRexVDOptions opt;
    opt.tFDR = 0.2;
    opt.K = 8;
    opt.L_factor = 4;
    opt.T_stop = 3;
    opt.calib = CalibMode::FixedTL;
    opt.seed = 5;
    opt.verbose = false;

    TRexVD selector(opt);
    TRexVDResult result = selector.run(X_, y_);

    EXPECT_EQ(result.T_stop, 3);
    EXPECT_EQ(result.Phi_mat.rows(), 1);   // fixed-T mode records final T only
    EXPECT_GE(result.selected_var.size(), 0);
}

}  // namespace trex::test::trex_selector_methods::trex_vd
