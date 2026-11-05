// ========================================================================================
// test_tsolvers_initialization.cpp
// ========================================================================================
/**
 * @file test_tsolvers_initialization.cpp
 * @brief Unit tests verifying initialization behavior (preprocessing, mappings,
 * collinearity checks) for TSolver_Base via the concrete proxy TLARS_Solver.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <algorithm>

// Eigen includes
#include <Eigen/Dense>

// tsolvers includes
#include <tsolvers/ad/linear_model/lars_based/tlars_solver.hpp>

// utils includes
#include <utils/datageneration/utils_datagen.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::tsolvers::ad::linear_model {

using namespace trex::tsolvers::ad::linear_model::lars_based;
using namespace trex::utils::datageneration::datagen;

// ========================================================================================

class TSolverInitTest : public ::testing::Test {
protected:
    Eigen::MatrixXd X, D;
    Eigen::VectorXd y;

    void SetUp() override {
        // Use a fixed random seed for reproducible testing
        std::srand(42);

        X = Eigen::MatrixXd::Random(200, 10);
        D = Eigen::MatrixXd::Random(200, 10);
        y = Eigen::VectorXd::Random(200);

        // Add extreme offsets to effectively test centering
        X.rowwise() += Eigen::VectorXd::Constant(10, 15.0).transpose();
        D.rowwise() += Eigen::VectorXd::Constant(10, -50.0).transpose();
        y.array() += 100.0;
    }
};

// ========================================================================================

/** @brief Test to verify unit L2-normalization and centroid subtraction during initialization */
TEST_F(TSolverInitTest, NormalizationAndCentering) {

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    // Creating the solver triggers the preprocessing inside TSolver_Base
    TLARS_Solver solver(X_map, D_map, y_map);

    // After centering and scaling, each column should have a zero mean and unit L2-norm
    for (int j = 0; j < X.cols(); ++j) {
        EXPECT_NEAR(X_map.col(j).mean(), 0.0, 1e-10);
        EXPECT_NEAR(X_map.col(j).norm(), 1.0, 1e-10);
    }

    for (int j = 0; j < D.cols(); ++j) {
        EXPECT_NEAR(D_map.col(j).mean(), 0.0, 1e-10);
        EXPECT_NEAR(D_map.col(j).norm(), 1.0, 1e-10);
    }
}


/** @brief Test to verify collinear columns are caught and tracked correctly to avoid singularity */
TEST_F(TSolverInitTest, CollinearDroppingLogic) {
    // Force a perfect collinearity (Column 1 is exactly 3x Column 0)
    X.col(1) = X.col(0) * 3.0;

    // Force a constant (zero variance) column
    X.col(2) = Eigen::VectorXd::Constant(200, 42.0);

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TLARS_Solver solver(X_map, D_map, y_map);

    // Since we forced dependent columns, there should be dropped indices
    const auto& dropped = solver.getDroppedIndices();
    EXPECT_FALSE(dropped.empty());

    // The perfectly collinear and constant columns should be safely dropped
    // ensuring executeStep doesn't encounter floating point limits
    EXPECT_NO_THROW(solver.executeStep(2, true));
}

/** @brief Regression test: an exactly constant column whose mean is not exactly
 *         representable leaves O(|c| * eps) rounding residue after centering.
 *         It must be dropped — not scaled by its own tiny residue norm into a
 *         unit-norm noise column that competes in the correlation screening. */
TEST_F(TSolverInitTest, NearConstantColumnIsDropped) {
    const double c = 1000.0 / 3.0;
    X.col(3) = Eigen::VectorXd::Constant(200, c);

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TLARS_Solver solver(X_map, D_map, y_map);

    const auto& dropped = solver.getDroppedIndices();
    EXPECT_TRUE(std::find(dropped.begin(), dropped.end(), 3u) != dropped.end())
        << "near-constant column normalized into a noise column instead of dropped "
        << "(residual norm after centering: " << X_map.col(3).norm() << ")";
}

/** @brief restore() must exactly reverse the in-place preprocessing
 *         (centering + scaling of X, D and centering of y). */
TEST_F(TSolverInitTest, RestoreReversesPreprocessing) {
    Eigen::MatrixXd X0 = X, D0 = D;
    Eigen::VectorXd y0 = y;

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TLARS_Solver solver(X_map, D_map, y_map);

    // Preprocessing mutated the buffers in place
    ASSERT_FALSE(X.isApprox(X0, 1e-12));

    solver.restore(X_map, D_map, y_map);

    EXPECT_TRUE(X.isApprox(X0, 1e-9)) << "X not restored";
    EXPECT_TRUE(D.isApprox(D0, 1e-9)) << "D not restored";
    EXPECT_TRUE(y.isApprox(y0, 1e-9)) << "y not restored";
}

/** @brief The default tie seed is fixed: two solvers built with default
 *         options start from the same tie-break engine state (reproducible
 *         runs also where exact ties occur); a negative seed draws from
 *         std::random_device. Readers, so both share the fixture buffers. */
TEST_F(TSolverInitTest, DefaultTieSeedIsReproducible) {
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    trex::tsolvers::ad::SolverOptions reader;
    reader.normalize = false;
    reader.intercept = false;
    EXPECT_EQ(reader.tie_seed, 0);

    TLARS_Solver a(X_map, D_map, y_map, reader);
    TLARS_Solver b(X_map, D_map, y_map, reader);
    EXPECT_TRUE(a.getRNG() == b.getRNG());

    trex::tsolvers::ad::SolverOptions entropy = reader;
    entropy.tie_seed = -1;
    TLARS_Solver c(X_map, D_map, y_map, entropy);
    TLARS_Solver d(X_map, D_map, y_map, entropy);
    EXPECT_FALSE(c.getRNG() == d.getRNG());  // equal with probability 2^-32
}


/** @brief The thread budget of a solver's own regions defaults to 1 (serial),
 *         a value below 1 throws (there is no "every core" value), and the
 *         setter follows the same rules. */
TEST_F(TSolverInitTest, ThreadBudgetDefaultsToSerial) {
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    trex::tsolvers::ad::SolverOptions reader;
    reader.normalize = false;
    reader.intercept = false;
    EXPECT_EQ(reader.n_threads, 1);

    TLARS_Solver serial(X_map, D_map, y_map, reader);
    EXPECT_EQ(serial.getNumThreads(), 1);

    trex::tsolvers::ad::SolverOptions zero = reader;
    zero.n_threads = 0;   // no "every core" value: below 1 throws
    EXPECT_THROW(TLARS_Solver(X_map, D_map, y_map, zero), std::invalid_argument);

    trex::tsolvers::ad::SolverOptions negative = reader;
    negative.n_threads = -1;
    EXPECT_THROW(TLARS_Solver(X_map, D_map, y_map, negative), std::invalid_argument);

    serial.setNumThreads(4);
    EXPECT_EQ(serial.getNumThreads(), 4);
    EXPECT_THROW(serial.setNumThreads(-2), std::invalid_argument);
}

// ========================================================================================
} /* End of namespace trex::test::tsolvers::ad::linear_model */
