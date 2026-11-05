// ========================================================================================
// test_tsolvers_memory.cpp
// ========================================================================================
/**
 * @file test_tsolvers_memory.cpp
 * @brief TSolver_Base::memoryBytes(): hand counts at construction, growth bounds,
 *        the borrowed X and D, the writer statistics, every T-solver, the
 *        checkpoint. The peak of a step: test_tsolvers_memory_peak.cpp.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/SparseCore>

// std includes
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

// LARS-based solvers
#include <tsolvers/ad/linear_model/lars_based/tlars_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tlasso_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tstepwise_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tstagewise_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tenet_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tenet_aug_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tienet_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tienet_aug_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tiknet_solver.hpp>

// OMP-based solvers
#include <tsolvers/ad/linear_model/omp_based/tomp_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tgp_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tacgp_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tmp_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tncgmp_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tools_solver.hpp>

// AFS- and CD-based solvers
#include <tsolvers/ad/linear_model/afs_based/tafs_solver.hpp>
#include <tsolvers/ad/linear_model/cd_based/tcdlasso_solver.hpp>
#include <tsolvers/ad/linear_model/cd_based/tcdenet_solver.hpp>
#include <tsolvers/ad/linear_model/cd_based/tcdienet_solver.hpp>
#include <tsolvers/ad/linear_model/cd_based/tcdtiknet_solver.hpp>

// memory helpers
#include <tsolvers/ad/solver_utils/memory_bytes.hpp>

// utils includes
#include <utils/datageneration/utils_datagen.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::tsolvers::ad::memory {

using namespace trex::tsolvers::ad::linear_model::lars_based;
using namespace trex::tsolvers::ad::linear_model::omp_based;
using namespace trex::tsolvers::ad::linear_model::afs_based;
using namespace trex::tsolvers::ad::linear_model::cd_based;
using trex::tsolvers::ad::SolverOptions;
using trex::tsolvers::ad::SparseBetaStep;
using trex::tsolvers::ad::TSolver_Base;
using trex::tsolvers::ad::solver_utils::memory_bytes::bytesOf;
namespace datagen = trex::utils::datageneration::datagen;
namespace fs = std::filesystem;

constexpr std::size_t kD = sizeof(double);
constexpr std::size_t kZ = sizeof(std::size_t);

// ========================================================================================
// Helpers
// ========================================================================================

/** @brief Owned design, dummies and response; the solvers borrow them. */
struct Problem {
    Eigen::MatrixXd X, D;
    Eigen::VectorXd y;
};

/** @brief Gaussian design with three true variables (not standardized). */
Problem makeProblem(Eigen::Index n, Eigen::Index p, Eigen::Index L, int seed = 42) {
    datagen::SyntheticData data(
        n, p, L, {0, 1, 2}, {3.0, -2.0, 2.5}, 1.0, seed, -1, -1,
        datagen::predictor_policy::Normal(),
        datagen::dummygen::Distribution::Normal(),
        datagen::noisegen::noise_policy::Normal());
    return {data.getX(), data.getD(), data.getY()};
}

/** @brief Maps onto a Problem, as the solvers take them. */
struct Maps {
    Eigen::Map<Eigen::MatrixXd> X, D;
    Eigen::Map<Eigen::VectorXd> y;
    explicit Maps(Problem& P)
        : X(P.X.data(), P.X.rows(), P.X.cols()),
          D(P.D.data(), P.D.rows(), P.D.cols()),
          y(P.y.data(), P.y.size()) {}
};

/** @brief Reader options: the solver never touches X and D. */
SolverOptions readerOptions() {
    SolverOptions o;
    o.normalize = false;
    o.intercept = false;
    return o;
}

/** @brief The estimate for a never-filled coefficient lookup map, as the solver
 *  holds it after construction (0 with libc++ and libstdc++). */
std::size_t emptyLookupBytes() {
    std::unordered_map<std::size_t, std::size_t> lookup;
    lookup.clear();
    return bytesOf(lookup);
}

/** @brief Base state of a reader at construction: y copy and residual (n each),
 *  correlations and inactive list (P each), the all-zero path step, one RSS, R2
 *  and DoF entry, the dummy record reserved for M + 1 steps, the lookup map. */
std::size_t baseAtConstruction(std::size_t n, std::size_t P, std::size_t M) {
    return 2 * kD * n + kD * P + kZ * P + sizeof(SparseBetaStep) + 2 * kD + kZ +
           kZ * (M + 1) + emptyLookupBytes();
}

/** @brief Exposes the lookup map estimate of a solver (protected member). */
template <class S>
struct LookupProbe : S {
    using S::S;
    std::size_t lookupBytes() const { return bytesOf(this->beta_pos_); }
};

/** @brief Whole file as a string (checkpoint comparison). */
std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// ========================================================================================
// Helpers of memory_bytes.hpp
// ========================================================================================

/** @brief Capacity for vectors, size for dense Eigen objects, nested heap added. */
TEST(TSolverMemoryHelpers, CountCapacityAndNestedHeap) {
    std::vector<double> v;
    v.reserve(40);
    v.push_back(1.0);
    EXPECT_EQ(bytesOf(v), 40 * kD);

    std::vector<std::vector<int>> nested(3);
    nested[1].reserve(5);
    EXPECT_EQ(bytesOf(nested), nested.capacity() * sizeof(std::vector<int>) + 5 * sizeof(int));

    EXPECT_EQ(bytesOf(Eigen::MatrixXd(7, 3)), 21 * kD);
    EXPECT_EQ(bytesOf(Eigen::VectorXi(9)), 9 * sizeof(int));
    EXPECT_EQ(bytesOf(Eigen::Matrix3d()), 0u);

    const std::string small = "abc";
    const std::string large(200, 'x');
    EXPECT_EQ(bytesOf(small), 0u);
    EXPECT_EQ(bytesOf(large), large.capacity() + 1);

    // Sparse: outer index (p + 1), values and inner indices of the nonzeros
    Eigen::SparseMatrix<double> K(6, 6);
    K.setIdentity();
    K.makeCompressed();
    EXPECT_EQ(bytesOf(K), 7 * sizeof(int) + 6 * (kD + sizeof(int)));
}

// ========================================================================================
// Exact counts at construction (readers, n = 20, p = L = 10)
// ========================================================================================

class TSolverMemoryHandCount : public ::testing::Test {
protected:
    static constexpr std::size_t n = 20, p = 10, L = 10, P = p + L;

    /** @brief maxSteps = 8 min(max_actives, effective_n) with max_actives =
     *  min(P, n) = 20 and effective_n = n (no intercept, not precentered). */
    static constexpr std::size_t M = 8 * (P < n ? P : n);

    Problem prob = makeProblem(n, p, L);
    Maps maps{prob};
};

TEST_F(TSolverMemoryHandCount, TLARS) {
    TLARS_Solver s(maps.X, maps.D, maps.y, readerOptions());
    // lambda sequence reserved for M + 1 entries, Kahan compensation over P
    EXPECT_EQ(s.memoryBytes(), baseAtConstruction(n, P, M) + kD * (M + 1) + kD * P);
}

TEST_F(TSolverMemoryHandCount, TOMP) {
    TOMP_Solver s(maps.X, maps.D, maps.y, readerOptions());
    EXPECT_EQ(s.memoryBytes(), baseAtConstruction(n, P, M));
}

TEST_F(TSolverMemoryHandCount, TOOLS) {
    TOOLS_Solver s(maps.X, maps.D, maps.y, readerOptions());
    // projected norms of all P columns
    EXPECT_EQ(s.memoryBytes(), baseAtConstruction(n, P, M) + kD * P);
}

TEST_F(TSolverMemoryHandCount, TAFSRhoHalf) {
    SolverOptions o = readerOptions();
    o.rho_afs = 0.5;
    TAFS_Solver s(maps.X, maps.D, maps.y, o);
    // maxSteps = ceil(8 min(max_actives, effective_n) / rho) = 320
    EXPECT_EQ(s.memoryBytes(), baseAtConstruction(n, P, 320));
}

TEST_F(TSolverMemoryHandCount, TCDLASSO) {
    TCDLASSO_Solver s(maps.X, maps.D, maps.y, readerOptions());
    // lambda sequence reserved for M + 1 entries; the empty sparse K keeps a
    // one-entry outer index; the CD caches are built by the first step
    EXPECT_EQ(s.memoryBytes(), baseAtConstruction(n, P, M) + kD * (M + 1) + sizeof(int));
}

TEST_F(TSolverMemoryHandCount, TENETAug) {
    SolverOptions o = readerOptions();
    o.lambda2 = 0.5;
    TENETAug_Solver s(maps.X, maps.D, maps.y, o);

    // Owned augmented system: (n + P) rows for X, D and y
    const std::size_t na = n + P;
    const std::size_t owned = kD * na * p + kD * na * L + kD * na;

    // Inner TLASSO reader on the augmented rows: max_actives = P, maxSteps = 8 P
    const std::size_t Mi = 8 * P;
    const std::size_t inner = baseAtConstruction(na, P, Mi) + kD * (Mi + 1) + kD * P;

    // Wrapper mirror: RSS, R2, DoF, inactive list, residual, own lookup map
    const std::size_t mirror = 2 * kD + kZ + kZ * P + kD * na + emptyLookupBytes();

    // Heap objects of the wrapper: three views and the inner solver
    const std::size_t objects = 2 * sizeof(Eigen::Map<Eigen::MatrixXd>) +
                                sizeof(Eigen::Map<Eigen::VectorXd>) + sizeof(TLASSO_Solver);

    EXPECT_EQ(s.memoryBytes(), owned + inner + mirror + objects);
}

// ========================================================================================
// Growth after executeStep()
// ========================================================================================

/** @brief Growth over one executeStep(): at least the sizes of what grew (k x k
 *  Cholesky factor, path, RSS / R2 / DoF, actions, active set, running
 *  coefficients), at most twice that (geometric capacity growth) plus the
 *  lookup map and one own per-active vector (T-LARS signs, T-OMP cache). */
template <class S>
void expectGrowthBounds(SolverOptions options) {
    Problem prob = makeProblem(60, 30, 30, 7);
    Maps m(prob);
    LookupProbe<S> s(m.X, m.D, m.y, options);
    const std::size_t before = s.memoryBytes();
    s.executeStep(5, true);
    ASSERT_GT(s.getNumSteps(), 0u);
    const std::size_t growth = s.memoryBytes() - before;

    const std::size_t k = s.getNumActives();
    const std::size_t steps = s.getRSS().size() - 1;
    const auto path = s.getBetaPathSparse();
    std::size_t path_entries = 0;
    for (std::size_t t = 1; t < path.steps.size(); ++t) {
        path_entries += (kZ + kD) * path.steps[t].idx.size();
    }
    std::size_t actions = s.getActions().size() * sizeof(std::vector<int>);
    for (const auto& a : s.getActions()) { actions += sizeof(int) * a.size(); }
    const std::size_t running = (kZ + kD) * path.steps.back().idx.size();

    const std::size_t lower = kD * k * k + path_entries + steps * sizeof(SparseBetaStep) +
                              steps * (2 * kD + kZ) + actions + kZ * k + running;
    EXPECT_GE(growth, lower) << s.solverTypeToString();
    EXPECT_LE(growth, 2 * lower + s.lookupBytes() + 2 * kD * k) << s.solverTypeToString();
}

TEST(TSolverMemoryGrowth, BoundedByActiveSetAndPath) {
    expectGrowthBounds<TLARS_Solver>(readerOptions());
    expectGrowthBounds<TOMP_Solver>(readerOptions());
    expectGrowthBounds<TOOLS_Solver>(readerOptions());
    SolverOptions afs = readerOptions();
    afs.rho_afs = 0.5;
    expectGrowthBounds<TAFS_Solver>(afs);
}

// ========================================================================================
// Borrowed X and D, writer statistics
// ========================================================================================

/** @brief Report of a reader at construction with n rows, p = L = 10. */
template <class S>
std::size_t constructionBytes(Eigen::Index n, SolverOptions options) {
    Problem prob = makeProblem(n, 10, 10);
    Maps m(prob);
    S s(m.X, m.D, m.y, options);
    return s.memoryBytes();
}

/** @brief Doubling n at P = 20 <= n keeps maxSteps and moves only the y copy
 *  and the residual (2 x 8 x 40 bytes); counted X and D would add 12,800. */
TEST(TSolverMemoryExclusions, DoublingNChangesOnlyTheNVectors) {
    const std::size_t expected = 2 * kD * 40;
    const SolverOptions r = readerOptions();
    SolverOptions afs = r;
    afs.rho_afs = 0.5;
    EXPECT_EQ(constructionBytes<TLARS_Solver>(80, r) - constructionBytes<TLARS_Solver>(40, r), expected);
    EXPECT_EQ(constructionBytes<TOMP_Solver>(80, r) - constructionBytes<TOMP_Solver>(40, r), expected);
    EXPECT_EQ(constructionBytes<TOOLS_Solver>(80, r) - constructionBytes<TOOLS_Solver>(40, r), expected);
    EXPECT_EQ(constructionBytes<TAFS_Solver>(80, afs) - constructionBytes<TAFS_Solver>(40, afs), expected);
    EXPECT_EQ(constructionBytes<TCDLASSO_Solver>(80, r) - constructionBytes<TCDLASSO_Solver>(40, r),
              expected);
}

/** @brief A writer (normalize and intercept) holds the means and norms of X and
 *  D, 16 (p + L) bytes, over a precentered reader with the same step limit. */
template <class S>
void expectWriterMinusReader() {
    Problem wp = makeProblem(20, 10, 10, 3);
    Problem rp = wp;
    Maps wm(wp), rm(rp);

    SolverOptions writer;
    writer.normalize = true;
    writer.intercept = true;
    SolverOptions reader = readerOptions();
    reader.precentered = true;

    S w(wm.X, wm.D, wm.y, writer);
    S r(rm.X, rm.D, rm.y, reader);
    ASSERT_TRUE(w.getWarnings().empty()) << w.solverTypeToString();
    EXPECT_EQ(w.memoryBytes() - r.memoryBytes(), 2 * kD * (10 + 10)) << w.solverTypeToString();
}

TEST(TSolverMemoryExclusions, WriterMinusReaderIsTheStatistics) {
    expectWriterMinusReader<TLARS_Solver>();
    expectWriterMinusReader<TOMP_Solver>();
    expectWriterMinusReader<TAFS_Solver>();
    expectWriterMinusReader<TCDLASSO_Solver>();
}

// ========================================================================================
// Every T-solver
// ========================================================================================

typedef ::testing::Types<
    TLARS_Solver, TLASSO_Solver, TSTEPWISE_Solver, TSTAGEWISE_Solver, TENET_Solver,
    TENETAug_Solver, TIENET_Solver, TIENETAug_Solver, TIKNET_Solver,
    TOMP_Solver, TGP_Solver, TACGP_Solver, TMP_Solver, TNCGMP_Solver, TOOLS_Solver,
    TAFS_Solver,
    TCDLASSO_Solver, TCDENET_Solver, TCDIENET_Solver, TCDTIKNET_Solver
> AllSolvers;

/** @brief Every solver on (X, D, y, options) with lambda2 = 0.5; the IEN family
 *  takes singleton groups, the Tikhonov family K = I. */
template <typename T>
std::unique_ptr<T> makeSolver(Maps& m, SolverOptions options) {
    options.lambda2 = 0.5;
    const Eigen::Index p = m.X.cols();
    if constexpr (std::is_same_v<T, TIENET_Solver> || std::is_same_v<T, TIENETAug_Solver> ||
                  std::is_same_v<T, TCDIENET_Solver>) {
        const Eigen::VectorXi groups =
            Eigen::VectorXi::LinSpaced(p, 0, static_cast<int>(p) - 1);
        return std::make_unique<T>(m.X, m.D, m.y, groups, options);
    } else if constexpr (std::is_same_v<T, TIKNET_Solver> || std::is_same_v<T, TCDTIKNET_Solver>) {
        Eigen::SparseMatrix<double> K(p, p);
        K.setIdentity();
        return std::make_unique<T>(m.X, m.D, m.y, K, options);
    } else {
        return std::make_unique<T>(m.X, m.D, m.y, options);
    }
}

template <typename T>
class TSolverMemoryAll : public ::testing::Test {};

TYPED_TEST_SUITE(TSolverMemoryAll, AllSolvers);

/** @brief Positive at construction, larger after a step, the same through the base. */
TYPED_TEST(TSolverMemoryAll, PositiveAndGrowsWithThePath) {
    Problem prob = makeProblem(50, 10, 10);
    Maps m(prob);
    auto solver = makeSolver<TypeParam>(m, SolverOptions{});

    const std::size_t at_construction = solver->memoryBytes();
    EXPECT_GT(at_construction, 0u);

    solver->executeStep(5, true);
    ASSERT_GT(solver->getNumSteps(), 0u);
    const TSolver_Base& base = *solver;
    EXPECT_EQ(base.memoryBytes(), solver->memoryBytes());
    EXPECT_GT(solver->memoryBytes(), at_construction);
}

/** @brief The report adds no state: the checkpoint written around it is byte
 *  for byte the same. A resumed solver holds no more than the saved one
 *  (reservations and transient caches are not archived). */
TYPED_TEST(TSolverMemoryAll, CheckpointUnchanged) {
    Problem prob = makeProblem(50, 10, 10);
    Maps m(prob);
    auto solver = makeSolver<TypeParam>(m, SolverOptions{});
    solver->executeStep(5, true);

    const std::string stem = ::testing::TempDir() + "memory_checkpoint_" +
                             solver->solverTypeToString();
    const std::string first = stem + "_a.bin", second = stem + "_b.bin";
    solver->save(first);
    const std::size_t held = solver->memoryBytes();
    solver->save(second);
    EXPECT_EQ(readFile(first), readFile(second));
    EXPECT_EQ(solver->memoryBytes(), held);

    TypeParam resumed = TypeParam::load(first, m.X, m.D);
    EXPECT_GT(resumed.memoryBytes(), 0u);
    EXPECT_LE(resumed.memoryBytes(), held);

    fs::remove(first);
    fs::remove(second);
}

// ========================================================================================
} /* End of namespace trex::test::tsolvers::ad::memory */
