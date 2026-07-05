// ========================================================================================
// test_tsolvers_polymorphic.cpp
// ========================================================================================
/**
 * @file test_tsolvers_polymorphic.cpp
 *
 * @brief Polymorphic smoke tests verifying execution and serialization for all 12 solvers.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <filesystem>
#include <memory>
#include <set>

// Eigen includes
#include <Eigen/Dense>

// LARS-based solvers
#include <tsolvers/linear_model/lars_based/tlars_solver.hpp>
#include <tsolvers/linear_model/lars_based/tlasso_solver.hpp>
#include <tsolvers/linear_model/lars_based/tenet_solver.hpp>
#include <tsolvers/linear_model/lars_based/tenet_aug_solver.hpp>
#include <tsolvers/linear_model/lars_based/tstepwise_solver.hpp>
#include <tsolvers/linear_model/lars_based/tstagewise_solver.hpp>

// OMP-based solvers
#include <tsolvers/linear_model/omp_based/tomp_solver.hpp>
#include <tsolvers/linear_model/omp_based/tmp_solver.hpp>
#include <tsolvers/linear_model/omp_based/tgp_solver.hpp>
#include <tsolvers/linear_model/omp_based/tacgp_solver.hpp>
#include <tsolvers/linear_model/omp_based/tncgmp_solver.hpp>
#include <tsolvers/linear_model/omp_based/tools_solver.hpp>

// AFS-based solver
#include <tsolvers/linear_model/afs_based/tafs_solver.hpp>

// datagen
#include <utils/datageneration/utils_datagen.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::tsolvers::linear_model {

// namespaces
using namespace trex::tsolvers::linear_model::lars_based;
using namespace trex::tsolvers::linear_model::omp_based;
using namespace trex::tsolvers::linear_model::afs_based;
using namespace trex::utils::datageneration::datagen;
namespace fs = std::filesystem;

// ========================================================================================

// Type List for the 13 solvers
typedef ::testing::Types<
    TLARS_Solver, TLASSO_Solver, TENET_Solver, TENETAug_Solver, TSTEPWISE_Solver,
    TSTAGEWISE_Solver, TOMP_Solver, TMP_Solver, TGP_Solver, TACGP_Solver,
    TNCGMP_Solver, TOOLS_Solver, TAFS_Solver
> AllSolvers;

// Factory for construction
template <typename T>
std::unique_ptr<T> create_solver(Eigen::Map<Eigen::MatrixXd>& X,
                                 Eigen::Map<Eigen::MatrixXd>& D,
                                 Eigen::Map<Eigen::VectorXd>& y) {
    if constexpr (std::is_same_v<T, TENET_Solver> ||
                  std::is_same_v<T, TENETAug_Solver>) {
        // TENET / TENETAug require lambda2
        return std::make_unique<T>(X, D, y, 0.5, true, true, false);
    } else if constexpr (std::is_same_v<T, TAFS_Solver>) {
        // TAFS has rho with a default, but to be sure we just match signature
        return std::make_unique<T>(X, D, y, 1.0, true, true, false);
    } else if constexpr (std::is_same_v<T, TNCGMP_Solver>) {
        // TNCGMP expects NCGMPVariant variant
        return std::make_unique<T>(X, D, y, NCGMPVariant::LineSearch, true, true, false);
    } else {
        return std::make_unique<T>(X, D, y, true, true, false);
    }
}

template <typename T>
class TSolverPolymorphicTest : public ::testing::Test {
protected:
    void SetUp() override {
        data = std::make_unique<SyntheticData>(
            50, 10, 10,
            std::vector<std::size_t>{1, 2, 3},
            std::vector<double>{5.0, -3.0, 2.0},
            1.0, 42, -1, -1,
            predictor_policy::Normal(),
            dummygen::Distribution::Normal(),
            noisegen::noise_policy::Normal()
        );
    }

    std::unique_ptr<SyntheticData> data;
};


/** @brief Test suite for all solvers. */
TYPED_TEST_SUITE(TSolverPolymorphicTest, AllSolvers);


/** @brief Test to verify that the solver executes `executeStep` without errors. */
TYPED_TEST(TSolverPolymorphicTest, ExecutionSmokeTest) {
    auto X = this->data->getX();
    auto D = this->data->getD();
    auto y = this->data->getY();

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    auto solver = create_solver<TypeParam>(X_map, D_map, y_map);

    solver->executeStep(5, true);

    EXPECT_GT(solver->getNumSteps(), 0);
    EXPECT_FALSE(solver->getActions().empty());

    if (solver->getNumSteps() > 0) {
        EXPECT_GT(solver->getRSS().front(), solver->getRSS().back());
    }
}


/** @brief Test to verify that serialization and deserialization of the solver
           maintains equivalence.
*/
TYPED_TEST(TSolverPolymorphicTest, SerializationEquivalence) {
    auto X = this->data->getX();
    auto D = this->data->getD();
    auto y = this->data->getY();

    auto X2 = X; auto D2 = D; auto y2 = y;

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    // 1. Continuous Reference execution
    auto ref_solver = create_solver<TypeParam>(X_map, D_map, y_map);
    ref_solver->executeStep(8, true);

    Eigen::Map<Eigen::MatrixXd> X_map2(X2.data(), X2.rows(), X2.cols());
    Eigen::Map<Eigen::MatrixXd> D_map2(D2.data(), D2.rows(), D2.cols());
    Eigen::Map<Eigen::VectorXd> y_map2(y2.data(), y2.size());

    // 2. Partial execution followed by save and resume
    auto partial_solver = create_solver<TypeParam>(X_map2, D_map2, y_map2);
    partial_solver->executeStep(4, true);

    std::string checkpoint = ::testing::TempDir() + "temp_poly_checkpoint.bin";

    partial_solver->save(checkpoint);
    EXPECT_TRUE(fs::exists(checkpoint));

    // Reload from file
    TypeParam resumed_solver = TypeParam::load(checkpoint, X_map2, D_map2);
    resumed_solver.executeStep(8, true);

    // Validate path equality
    EXPECT_EQ(ref_solver->getActions(), resumed_solver.getActions());

    auto ref_beta = ref_solver->getBeta(-1);
    auto res_beta = resumed_solver.getBeta(-1);
    EXPECT_TRUE(ref_beta.isApprox(res_beta, 1e-9));

    // Cleanup
    fs::remove(checkpoint);
}

/** @brief Invariant: actives, inactives, and dropped indices must partition the
 *         unified index space after execution — no index in two sets, none missing.
 *         Guards the getInactives() contract for re-selection solvers (TMP, TGP,
 *         TACGP, TNCGMP), which scan all columns internally.
 */
TYPED_TEST(TSolverPolymorphicTest, ActiveInactiveDroppedPartition) {
    auto X = this->data->getX();
    auto D = this->data->getD();
    auto y = this->data->getY();

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    auto solver = create_solver<TypeParam>(X_map, D_map, y_map);
    solver->executeStep(5, true);

    const std::size_t p_tot = static_cast<std::size_t>(X.cols() + D.cols());

    std::set<std::size_t> seen;
    auto insert_all = [&seen](const std::vector<std::size_t>& v, const char* name) {
        for (std::size_t j : v) {
            EXPECT_TRUE(seen.insert(j).second)
                << "index " << j << " appears in multiple sets (last seen in " << name << ")";
        }
    };
    insert_all(solver->getActives(), "actives");
    insert_all(solver->getInactives(), "inactives");
    insert_all(solver->getDroppedIndices(), "dropped");

    EXPECT_EQ(seen.size(), p_tot) << "actives/inactives/dropped do not cover all indices";
}

// ========================================================================================
} /* End of namespace trex::test::tsolvers::linear_model */
