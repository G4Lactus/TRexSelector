// ========================================================================================
// test_tsolvers_getters.cpp
// ========================================================================================
/**
 * @file test_tsolvers_getters.cpp
 *
 * @brief Unit tests verifying API read methods for model state, path iteration,
 *        and intercepts for TSolver_Base.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <cmath>
#include <filesystem>
#include <fstream>

// Cereal includes
#include <cereal/archives/portable_binary.hpp>

// Eigen includes
#include <Eigen/Dense>

// tsolvers includes
#include <tsolvers/linear_model/lars_based/tlars_solver.hpp>
#include <tsolvers/linear_model/lars_based/tlasso_solver.hpp>
#include <tsolvers/linear_model/lars_based/tenet_solver.hpp>
#include <tsolvers/linear_model/lars_based/tenet_aug_solver.hpp>

// utils includes
#include <utils/datageneration/utils_datagen.hpp>
#include <utils/fp_classify/fp_classify.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::tsolvers::linear_model {

using namespace trex::tsolvers::linear_model::lars_based;
using namespace trex::utils::datageneration::datagen;

// ========================================================================================

/** @brief Test to verify extraction of specific iteration path coefficients */
TEST(TSolverGettersTest, ValidateCoefficientPathExtraction) {
    SyntheticData data(
        100, 20, 20,
        {1, 2},
        {5.0, 2.0},
        1.0, 42, -1, -1,
        predictor_policy::Normal(),
        dummygen::Distribution::Normal(),
        noisegen::noise_policy::Normal()
    );

    auto X = data.getX();
    auto D = data.getD();
    auto y = data.getY();

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    // Intercept activated
    TLARS_Solver solver(X_map, D_map, y_map,
                        true, true, false);
    solver.executeStep(2, true);

    // Check sizes match step execution
    std::size_t steps = solver.getNumSteps();
    Eigen::MatrixXd beta_path = solver.getBetaPath();

    // Path columns should correspond to tracked actions/steps + 1
    // (for the zero-initialization step 0)
    EXPECT_EQ(beta_path.cols(), solver.getActions().size() + 1);

    for (int step = 0; step < steps; ++step) {
        // Individual beta vector uncentering scale retrieval vs full uncentered path retrieval
        auto beta_at_step = solver.getBeta(step);
        EXPECT_TRUE(beta_path.col(step).isApprox(beta_at_step, 1e-10));
    }
}

/** @brief getBeta with an out-of-range step must throw instead of reading
 *         out of bounds; any negative step returns the last path column. */
TEST(TSolverGettersTest, GetBetaStepOutOfRangeThrows) {
    SyntheticData data(
        100, 20, 20,
        {1, 2},
        {5.0, 2.0},
        1.0, 42, -1, -1,
        predictor_policy::Normal(),
        dummygen::Distribution::Normal(),
        noisegen::noise_policy::Normal()
    );

    auto X = data.getX();
    auto D = data.getD();
    auto y = data.getY();

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TLARS_Solver solver(X_map, D_map, y_map, true, true, false);
    solver.executeStep(2, true);

    Eigen::Index n_cols = solver.getBetaPath().cols();
    EXPECT_NO_THROW(solver.getBeta(static_cast<int>(n_cols) - 1));
    EXPECT_NO_THROW(solver.getBeta(-1));
    EXPECT_THROW(solver.getBeta(static_cast<int>(n_cols)), std::invalid_argument);
    EXPECT_THROW(solver.getBeta(1000000), std::invalid_argument);
}


/** @brief Regression test: getCp must not dereference the (null) design matrix
 *         pointer on a deserialized solver that has not been reconnected. */
TEST(TSolverGettersTest, GetCpWorksOnDisconnectedSolver) {
    namespace fs = std::filesystem;

    SyntheticData data(
        100, 20, 20,
        {1, 2},
        {5.0, 2.0},
        1.0, 42, -1, -1,
        predictor_policy::Normal(),
        dummygen::Distribution::Normal(),
        noisegen::noise_policy::Normal()
    );

    auto X = data.getX();
    auto D = data.getD();
    auto y = data.getY();

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TLARS_Solver solver(X_map, D_map, y_map, true, true, false);
    solver.executeStep(3, true);
    std::vector<double> cp_connected = solver.getCp();

    const std::string checkpoint = ::testing::TempDir() + "temp_getcp_disconnected.bin";
    solver.save(checkpoint);

    // Deserialize WITHOUT reconnecting (bypass TLARS_Solver::load on purpose)
    TLARS_Solver detached;
    {
        std::ifstream is(checkpoint, std::ios::binary);
        cereal::PortableBinaryInputArchive iarchive(is);
        iarchive(detached);
    }
    ASSERT_FALSE(detached.isConnected());

    std::vector<double> cp_detached = detached.getCp();
    ASSERT_EQ(cp_detached.size(), cp_connected.size());
    for (std::size_t k = 0; k < cp_connected.size(); ++k) {
        EXPECT_DOUBLE_EQ(cp_detached[k], cp_connected[k]) << "Cp differs at step " << k;
    }

    fs::remove(checkpoint);
}


/** @brief Regression test: TENET getBeta/getBetaPath must de-normalize dummy
 *         coefficients with normsd_, exactly like real predictors with normsx_.
 *
 *  Differential setup: solver A runs on raw data with internal normalization;
 *  solver B runs on manually centered/normalized copies with preprocessing
 *  disabled. Both see bit-identical internal data, so A's original-scale beta
 *  must equal B's normalized-scale beta divided elementwise by the norms. */
TEST(TSolverGettersTest, TENETBetaDeNormalizesDummies) {
    std::srand(11);
    const Eigen::Index n = 150, p = 15, L = 15;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p) * 3.0;
    Eigen::MatrixXd D = Eigen::MatrixXd::Random(n, L) * 5.0;
    Eigen::VectorXd y = 4.0 * X.col(0) - 2.0 * X.col(3)
                        + 0.5 * Eigen::VectorXd::Random(n);

    // Solver A: raw copies, internal centering + normalization
    Eigen::MatrixXd XA = X, DA = D;
    Eigen::VectorXd yA = y;
    Eigen::Map<Eigen::MatrixXd> XA_map(XA.data(), n, p);
    Eigen::Map<Eigen::MatrixXd> DA_map(DA.data(), n, L);
    Eigen::Map<Eigen::VectorXd> yA_map(yA.data(), n);
    TENET_Solver A(XA_map, DA_map, yA_map, 0.5, true, true, false);
    A.executeStep(2, true);

    // Solver B: manually preprocessed copies, internal preprocessing disabled
    Eigen::MatrixXd XB = X, DB = D;
    Eigen::VectorXd yB = y;
    Eigen::VectorXd norms(p + L);
    for (Eigen::Index j = 0; j < p; ++j) {
        XB.col(j).array() -= XB.col(j).mean();
        norms(j) = XB.col(j).norm();
        XB.col(j) /= norms(j);
    }
    for (Eigen::Index j = 0; j < L; ++j) {
        DB.col(j).array() -= DB.col(j).mean();
        norms(p + j) = DB.col(j).norm();
        DB.col(j) /= norms(p + j);
    }
    yB.array() -= yB.mean();

    Eigen::Map<Eigen::MatrixXd> XB_map(XB.data(), n, p);
    Eigen::Map<Eigen::MatrixXd> DB_map(DB.data(), n, L);
    Eigen::Map<Eigen::VectorXd> yB_map(yB.data(), n);
    TENET_Solver B(XB_map, DB_map, yB_map, 0.5, false, false, false);
    B.executeStep(2, true);

    ASSERT_EQ(A.getNumSteps(), B.getNumSteps());
    // At least one dummy must be active, otherwise the dummy branch is untested
    ASSERT_GT(A.getActiveDummyIndices().size(), 0u);

    Eigen::VectorXd betaA = A.getBeta(-1);
    Eigen::VectorXd expected = B.getBeta(-1).array() / norms.array();
    EXPECT_TRUE(betaA.isApprox(expected, 1e-9))
        << "max abs diff: " << (betaA - expected).cwiseAbs().maxCoeff();

    Eigen::MatrixXd pathA = A.getBetaPath();
    EXPECT_TRUE(pathA.col(pathA.cols() - 1).isApprox(expected, 1e-9))
        << "getBetaPath last column disagrees with getBeta(-1)";
}

/** @brief ZSCORE scaling mode must be equivalent to manual z-scoring:
 *         solver A standardizes internally, solver B runs preprocessing-free
 *         on manually standardized data; original-scale betas must agree. */
TEST(TSolverGettersTest, ZScoreScalingMatchesManualStandardization) {
    std::srand(19);
    const Eigen::Index n = 120, p = 10, L = 10;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p) * 2.0;
    X.rowwise() += Eigen::VectorXd::Constant(p, 7.0).transpose();
    Eigen::MatrixXd D = Eigen::MatrixXd::Random(n, L) * 3.0;
    Eigen::VectorXd y = 4.0 * X.col(0) - 2.0 * X.col(3)
                        + 0.5 * Eigen::VectorXd::Random(n);

    // Solver A: internal centering + z-score scaling
    Eigen::MatrixXd XA = X, DA = D;
    Eigen::VectorXd yA = y;
    Eigen::Map<Eigen::MatrixXd> XA_map(XA.data(), n, p);
    Eigen::Map<Eigen::MatrixXd> DA_map(DA.data(), n, L);
    Eigen::Map<Eigen::VectorXd> yA_map(yA.data(), n);
    TLARS_Solver A(XA_map, DA_map, yA_map, true, true, false,
                   SolverTypeLarsBased::TLARS, trex::tsolvers::ScalingMode::ZSCORE);
    A.executeStep(2, true);

    // Solver B: manual z-scoring (divisor = sample SD), preprocessing disabled
    Eigen::MatrixXd XB = X, DB = D;
    Eigen::VectorXd yB = y;
    const double sd_factor = std::sqrt(static_cast<double>(n - 1));
    Eigen::VectorXd divisors(p + L);
    for (Eigen::Index j = 0; j < p; ++j) {
        XB.col(j).array() -= XB.col(j).mean();
        divisors(j) = XB.col(j).norm() / sd_factor;
        XB.col(j) /= divisors(j);
    }
    for (Eigen::Index j = 0; j < L; ++j) {
        DB.col(j).array() -= DB.col(j).mean();
        divisors(p + j) = DB.col(j).norm() / sd_factor;
        DB.col(j) /= divisors(p + j);
    }
    yB.array() -= yB.mean();

    Eigen::Map<Eigen::MatrixXd> XB_map(XB.data(), n, p);
    Eigen::Map<Eigen::MatrixXd> DB_map(DB.data(), n, L);
    Eigen::Map<Eigen::VectorXd> yB_map(yB.data(), n);
    TLARS_Solver B(XB_map, DB_map, yB_map, false, false, false);
    B.executeStep(2, true);

    ASSERT_EQ(A.getNumSteps(), B.getNumSteps());
    Eigen::VectorXd expected = B.getBeta(-1).array() / divisors.array();
    EXPECT_TRUE(A.getBeta(-1).isApprox(expected, 1e-9))
        << "max abs diff: " << (A.getBeta(-1) - expected).cwiseAbs().maxCoeff();
}


/** @brief Parity: the augmented-LASSO elastic net (TENETAug) must match the
 *         Gram-based TENET on identically pre-standardized data (Zou & Hastie
 *         2005, Lemma 1; previously verified only in a demo). */
TEST(TSolverGettersTest, TENETAugMatchesGramBasedTENET) {
    std::srand(23);
    const Eigen::Index n = 150, p = 15, L = 15;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    Eigen::MatrixXd D = Eigen::MatrixXd::Random(n, L);
    Eigen::VectorXd y = 4.0 * X.col(0) - 2.0 * X.col(3)
                        + 0.5 * Eigen::VectorXd::Random(n);

    // Pre-standardize exactly as the augmented-LASSO recipe requires
    // (center + unit-L2 columns, centered response)
    for (Eigen::Index j = 0; j < p; ++j) {
        X.col(j).array() -= X.col(j).mean();
        X.col(j) /= X.col(j).norm();
    }
    for (Eigen::Index j = 0; j < L; ++j) {
        D.col(j).array() -= D.col(j).mean();
        D.col(j) /= D.col(j).norm();
    }
    y.array() -= y.mean();

    const double lambda2 = 0.5;

    Eigen::MatrixXd X1 = X, D1 = D;
    Eigen::VectorXd y1 = y;
    Eigen::Map<Eigen::MatrixXd> X1_map(X1.data(), n, p);
    Eigen::Map<Eigen::MatrixXd> D1_map(D1.data(), n, L);
    Eigen::Map<Eigen::VectorXd> y1_map(y1.data(), n);
    TENET_Solver gram(X1_map, D1_map, y1_map, lambda2, false, false, false);
    gram.executeStep(3, true);

    Eigen::MatrixXd X2 = X, D2 = D;
    Eigen::VectorXd y2 = y;
    Eigen::Map<Eigen::MatrixXd> X2_map(X2.data(), n, p);
    Eigen::Map<Eigen::MatrixXd> D2_map(D2.data(), n, L);
    Eigen::Map<Eigen::VectorXd> y2_map(y2.data(), n);
    TENETAug_Solver aug(X2_map, D2_map, y2_map, lambda2);
    aug.executeStep(3, true);

    ASSERT_EQ(gram.getNumSteps(), aug.getNumSteps());
    EXPECT_TRUE(gram.getBeta(-1).isApprox(aug.getBeta(-1), 1e-8))
        << "max abs diff: "
        << (gram.getBeta(-1) - aug.getBeta(-1)).cwiseAbs().maxCoeff();
}

/** @brief TENET Cp is undefined when n <= p_total + 1 (the typical T-Rex
 *         high-dimensional regime): it must return NaN, not garbage from a
 *         negative residual-variance estimate. */
TEST(TSolverGettersTest, TENETCpIsNaNInHighDimensionalRegime) {
    std::srand(29);
    const Eigen::Index n = 20, p = 15, L = 15; // p_total = 30 > n
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    Eigen::MatrixXd D = Eigen::MatrixXd::Random(n, L);
    Eigen::VectorXd y = 3.0 * X.col(0) + 0.5 * Eigen::VectorXd::Random(n);

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), n, p);
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), n, L);
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), n);

    TENET_Solver solver(X_map, D_map, y_map, 0.5, true, true, false);
    solver.executeStep(2, true);

    std::vector<double> cp = solver.getCp();
    ASSERT_FALSE(cp.empty());
    for (std::size_t k = 0; k < cp.size(); ++k) {
        // Bit-level NaN check: std::isnan is constant-folded to false under
        // the Release build's -ffast-math (-ffinite-math-only).
        EXPECT_TRUE(trex::utils::fp_classify::isnan(cp[k]))
            << "Cp[" << k << "] = " << cp[k] << " but n <= p_total + 1";
    }
}


/** @brief Parity: TENET with lambda2 = 0 must reduce exactly to T-LASSO
 *         (documented in the TENET class docs, previously untested). */
TEST(TSolverGettersTest, TENETWithZeroLambda2MatchesTLASSO) {
    std::srand(31);
    const Eigen::Index n = 150, p = 15, L = 15;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);
    Eigen::MatrixXd D = Eigen::MatrixXd::Random(n, L);
    Eigen::VectorXd y = 4.0 * X.col(0) - 2.0 * X.col(3)
                        + 0.5 * Eigen::VectorXd::Random(n);

    Eigen::MatrixXd X1 = X, D1 = D;
    Eigen::VectorXd y1 = y;
    Eigen::Map<Eigen::MatrixXd> X1_map(X1.data(), n, p);
    Eigen::Map<Eigen::MatrixXd> D1_map(D1.data(), n, L);
    Eigen::Map<Eigen::VectorXd> y1_map(y1.data(), n);
    TENET_Solver tenet(X1_map, D1_map, y1_map, 0.0, true, true, false);
    tenet.executeStep(3, true);

    Eigen::MatrixXd X2 = X, D2 = D;
    Eigen::VectorXd y2 = y;
    Eigen::Map<Eigen::MatrixXd> X2_map(X2.data(), n, p);
    Eigen::Map<Eigen::MatrixXd> D2_map(D2.data(), n, L);
    Eigen::Map<Eigen::VectorXd> y2_map(y2.data(), n);
    TLASSO_Solver tlasso(X2_map, D2_map, y2_map, true, true, false);
    tlasso.executeStep(3, true);

    ASSERT_EQ(tenet.getNumSteps(), tlasso.getNumSteps());
    EXPECT_TRUE(tenet.getBeta(-1).isApprox(tlasso.getBeta(-1), 1e-9))
        << "max abs diff: "
        << (tenet.getBeta(-1) - tlasso.getBeta(-1)).cwiseAbs().maxCoeff();
}

// ========================================================================================
} /* End of namespace trex::test::tsolvers::linear_model */
