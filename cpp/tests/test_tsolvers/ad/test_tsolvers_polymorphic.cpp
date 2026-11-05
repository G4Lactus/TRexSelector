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
#include <cmath>
#include <filesystem>
#include <memory>
#include <random>
#include <set>
#include <type_traits>

// Eigen includes
#include <Eigen/Dense>

// LARS-based solvers
#include <tsolvers/ad/linear_model/lars_based/tlars_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tlasso_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tenet_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tenet_aug_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tienet_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tienet_aug_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tstepwise_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tstagewise_solver.hpp>

// OMP-based solvers
#include <tsolvers/ad/linear_model/omp_based/tomp_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tmp_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tgp_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tacgp_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tncgmp_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tools_solver.hpp>

// CCD-based solvers
#include <tsolvers/ad/linear_model/cd_based/tcdlasso_solver.hpp>
#include <tsolvers/ad/linear_model/cd_based/tcdenet_solver.hpp>
#include <tsolvers/ad/linear_model/cd_based/tcdienet_solver.hpp>

// AFS-based solver
#include <tsolvers/ad/linear_model/afs_based/tafs_solver.hpp>

// datagen
#include <utils/datageneration/utils_datagen.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::tsolvers::ad::linear_model {

// namespaces
using namespace trex::tsolvers::ad::linear_model::lars_based;
using namespace trex::tsolvers::ad::linear_model::cd_based;
using namespace trex::tsolvers::ad::linear_model::omp_based;
using namespace trex::tsolvers::ad::linear_model::afs_based;
using namespace trex::utils::datageneration::datagen;
namespace fs = std::filesystem;

// ========================================================================================

// Type List for the 18 solvers
typedef ::testing::Types<
    TLARS_Solver, TLASSO_Solver, TENET_Solver, TENETAug_Solver, TIENET_Solver,
    TIENETAug_Solver, TSTEPWISE_Solver, TSTAGEWISE_Solver,
    TCDLASSO_Solver, TCDENET_Solver, TCDIENET_Solver,
    TOMP_Solver, TMP_Solver, TGP_Solver, TACGP_Solver, TNCGMP_Solver,
    TOOLS_Solver, TAFS_Solver
> AllSolvers;

// Factory for construction
template <typename T>
std::unique_ptr<T> create_solver(Eigen::Map<Eigen::MatrixXd>& X,
                                 Eigen::Map<Eigen::MatrixXd>& D,
                                 Eigen::Map<Eigen::VectorXd>& y) {
    // Every solver takes (X, D, y, options); the EN / IEN families read
    // lambda2 from the options, the IEN family also takes the group vector.
    trex::tsolvers::ad::SolverOptions opts;
    opts.lambda2 = 0.5;
    if constexpr (std::is_same_v<T, TENET_Solver> ||
                  std::is_same_v<T, TENETAug_Solver> ||
                  std::is_same_v<T, TCDENET_Solver>) {
        return std::make_unique<T>(X, D, y, opts);
    } else if constexpr (std::is_same_v<T, TIENET_Solver> ||
                         std::is_same_v<T, TIENETAug_Solver> ||
                         std::is_same_v<T, TCDIENET_Solver>) {
        // Singleton groups (each variable its own group) exercise the full
        // group machinery; per Corollary 1 of the IEN paper this is the
        // EN-like special case.
        Eigen::VectorXi groups = Eigen::VectorXi::LinSpaced(
            X.cols(), 0, static_cast<int>(X.cols()) - 1);
        return std::make_unique<T>(X, D, y, groups, opts);
    } else {
        // TAFS (rho_afs = 1), TNCGMP (LineSearch) and the plain solvers run
        // on the option defaults.
        return std::make_unique<T>(X, D, y);
    }
}

/** @brief Solvers whose active set may exceed the row bound by design: the
 *  quadratic penalty (lambda2 > 0) lifts the cap of the elastic nets. */
template <typename T>
constexpr bool lifts_row_bound =
    std::is_same_v<T, TENET_Solver> || std::is_same_v<T, TENETAug_Solver> ||
    std::is_same_v<T, TIENET_Solver> || std::is_same_v<T, TIENETAug_Solver> ||
    std::is_same_v<T, TCDENET_Solver> || std::is_same_v<T, TCDIENET_Solver>;

// Factory with explicit preprocessing options (lambda2 = 0.5 as above)
template <typename T>
std::unique_ptr<T> create_solver_with(Eigen::Map<Eigen::MatrixXd>& X,
                                      Eigen::Map<Eigen::MatrixXd>& D,
                                      Eigen::Map<Eigen::VectorXd>& y,
                                      trex::tsolvers::ad::SolverOptions opts) {
    opts.lambda2 = 0.5;
    if constexpr (std::is_same_v<T, TIENET_Solver> ||
                  std::is_same_v<T, TIENETAug_Solver> ||
                  std::is_same_v<T, TCDIENET_Solver>) {
        Eigen::VectorXi groups = Eigen::VectorXi::LinSpaced(
            X.cols(), 0, static_cast<int>(X.cols()) - 1);
        return std::make_unique<T>(X, D, y, groups, opts);
    } else {
        return std::make_unique<T>(X, D, y, opts);
    }
}

/** @brief Integer-valued n x p matrix whose columns sum to exactly zero (the
 *  last row is minus the sum of the others): the floating-point column means
 *  are exactly 0, so a solver's own centering is a bitwise no-op. */
Eigen::MatrixXd exactly_centered_integers(Eigen::Index n, Eigen::Index p, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> draw(-6, 6);
    Eigen::MatrixXd M(n, p);
    for (Eigen::Index j = 0; j < p; ++j) {
        double sum = 0.0;
        for (Eigen::Index i = 0; i + 1 < n; ++i) {
            M(i, j) = static_cast<double>(draw(rng));
            sum += M(i, j);
        }
        M(n - 1, j) = -sum;
    }
    return M;
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

    // Per-type filename: the typed-test instances run concurrently under
    // `ctest -j`, and a shared path makes the save/remove calls race.
    std::string checkpoint = ::testing::TempDir() + "temp_poly_checkpoint_"
                             + partial_solver->solverTypeToString() + ".bin";

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

/** @brief Contract of SolverOptions::precentered: a reader on caller-centered
 *         data counts the intercept exactly as intercept = true does on the
 *         same data (same path, degrees of freedom and Cp), and the active set
 *         stays within the n - 1 row bound of the centered design unless the
 *         quadratic penalty lifts the cap. The data are exactly centered, so
 *         the intercept run's own centering is a bitwise no-op.
 */
TYPED_TEST(TSolverPolymorphicTest, PrecenteredCountsInterceptLikeIntercept) {
    constexpr Eigen::Index n = 20, p = 30, L = 30;
    const Eigen::MatrixXd X0 = exactly_centered_integers(n, p, 11);
    const Eigen::MatrixXd D0 = exactly_centered_integers(n, L, 12);
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    beta(1) = 3.0; beta(4) = -2.0; beta(9) = 2.0;
    const Eigen::VectorXd y0 = X0 * beta + exactly_centered_integers(n, 1, 13).col(0);

    trex::tsolvers::ad::SolverOptions base;
    base.normalize = false;
    base.tie_seed = 7;

    // intercept = true: the solver centers its (separate) buffers itself
    Eigen::MatrixXd Xa = X0, Da = D0;
    Eigen::VectorXd ya = y0;
    Eigen::Map<Eigen::MatrixXd> Xa_map(Xa.data(), n, p);
    Eigen::Map<Eigen::MatrixXd> Da_map(Da.data(), n, L);
    Eigen::Map<Eigen::VectorXd> ya_map(ya.data(), n);
    trex::tsolvers::ad::SolverOptions with_intercept = base;
    with_intercept.intercept = true;
    auto fitted = create_solver_with<TypeParam>(Xa_map, Da_map, ya_map, with_intercept);
    fitted->executeStep(L, true);

    // precentered reader on the caller-centered data
    Eigen::MatrixXd Xb = X0, Db = D0;
    Eigen::VectorXd yb = y0;
    Eigen::Map<Eigen::MatrixXd> Xb_map(Xb.data(), n, p);
    Eigen::Map<Eigen::MatrixXd> Db_map(Db.data(), n, L);
    Eigen::Map<Eigen::VectorXd> yb_map(yb.data(), n);
    trex::tsolvers::ad::SolverOptions reader = base;
    reader.intercept = false;
    reader.precentered = true;
    auto precentered = create_solver_with<TypeParam>(Xb_map, Db_map, yb_map, reader);
    precentered->executeStep(L, true);

    EXPECT_EQ(fitted->getActions(), precentered->getActions());
    EXPECT_EQ(fitted->getDoF(), precentered->getDoF());
    EXPECT_EQ(fitted->getNumActives(), precentered->getNumActives());
    const auto cp_fitted = fitted->getCp();
    const auto cp_precentered = precentered->getCp();
    ASSERT_EQ(cp_fitted.size(), cp_precentered.size());
    for (std::size_t k = 0; k < cp_fitted.size(); ++k) {
        if (std::isnan(cp_fitted[k])) {
            EXPECT_TRUE(std::isnan(cp_precentered[k])) << "step " << k;
        } else {
            EXPECT_EQ(cp_fitted[k], cp_precentered[k]) << "step " << k;
        }
    }
    EXPECT_EQ(precentered->getDoF().front(), 1u);
    if constexpr (!lifts_row_bound<TypeParam>) {
        EXPECT_LE(precentered->getNumActives(), static_cast<std::size_t>(n - 1));
    }
}


/** @brief The path of every solver is bitwise independent of its thread
 *         budget: a kept region splits a pass whose results are written per
 *         element, and every reduction runs in a fixed order. The design lies
 *         above the work threshold of the regions, so the budget is live.
 */
TYPED_TEST(TSolverPolymorphicTest, PathIsIndependentOfTheThreadBudget) {
    if (!trex::utils::openmp::is_available() || omp_get_num_procs() < 2) {
        GTEST_SKIP() << "needs OpenMP and at least two cores";
    }
    constexpr Eigen::Index n = 400, p = 1500, L = 1500;   // 1.2 M entries per pass
    SyntheticData data(n, p, L, std::vector<std::size_t>{3, 17, 41, 77, 101, 150, 222},
                       std::vector<double>{3.0, -2.0, 2.0, -3.0, 2.5, 1.5, -1.5}, 10.0, 7, -1, -1,
                       predictor_policy::Normal(), dummygen::Distribution::Normal(),
                       noisegen::noise_policy::Normal());
    const Eigen::MatrixXd X0 = data.getX(), D0 = data.getD();
    const Eigen::VectorXd y0 = data.getY();

    // Writers on private copies: the preprocessing passes take the budget too
    const auto run = [&](int n_threads) {
        Eigen::MatrixXd X = X0, D = D0;
        Eigen::VectorXd y = y0;
        Eigen::Map<Eigen::MatrixXd> Xm(X.data(), n, p), Dm(D.data(), n, L);
        Eigen::Map<Eigen::VectorXd> ym(y.data(), n);
        trex::tsolvers::ad::SolverOptions opts;
        opts.n_threads = n_threads;
        auto solver = create_solver_with<TypeParam>(Xm, Dm, ym, opts);
        EXPECT_EQ(solver->getNumThreads(), n_threads);
        solver->executeStep(4, true);
        return std::make_pair(solver->getActions(), solver->getBetaPathSparse());
    };
    const auto [actions_1, path_1] = run(1);
    const auto [actions_4, path_4] = run(4);

    EXPECT_EQ(actions_1, actions_4);
    ASSERT_EQ(path_1.steps.size(), path_4.steps.size());
    for (std::size_t s = 0; s < path_1.steps.size(); ++s) {
        EXPECT_EQ(path_1.steps[s].idx, path_4.steps[s].idx) << "step " << s;
        EXPECT_EQ(path_1.steps[s].val, path_4.steps[s].val) << "step " << s;
    }
}

// ========================================================================================
} /* End of namespace trex::test::tsolvers::ad::linear_model */
