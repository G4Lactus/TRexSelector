// ========================================================================================
// test_tsolvers_memory_peak.cpp
// ========================================================================================
/**
 * @file test_tsolvers_memory_peak.cpp
 * @brief Peak heap bytes inside executeStep() against memoryBytes() after it,
 *        measured with a counting allocator. memoryBytes() is the held state;
 *        the temporaries of a step come on top and are gone when it returns.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <string>
#include <type_traits>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/SparseCore>

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

// utils includes
#include <utils/datageneration/utils_datagen.hpp>

// ========================================================================================
// Counting allocator (this executable only)
// ========================================================================================

// Eigen allocates through std::malloc inside the library archives, which a
// replaced operator new alone would miss, so malloc itself is replaced: over
// the malloc zones on macOS, over the __libc_* entry points on glibc. Other
// platforms and sanitizer builds skip the test.

#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || \
    __has_feature(memory_sanitizer)
#define TREX_TEST_SANITIZED 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define TREX_TEST_SANITIZED 1
#endif

#if !defined(TREX_TEST_SANITIZED) && (defined(__APPLE__) || defined(__GLIBC__))
#define TREX_COUNTING_ALLOCATOR 1
#else
#define TREX_COUNTING_ALLOCATOR 0
#endif

#if TREX_COUNTING_ALLOCATOR

#if defined(__APPLE__)
#include <malloc/malloc.h>
#define TREX_MALLOC_NOEXCEPT
#else
#include <malloc.h>
#define TREX_MALLOC_NOEXCEPT __THROW
extern "C" {
void* __libc_malloc(std::size_t size);
void* __libc_calloc(std::size_t count, std::size_t size);
void* __libc_realloc(void* ptr, std::size_t size);
void* __libc_memalign(std::size_t alignment, std::size_t size);
void __libc_free(void* ptr);
}
#endif

namespace {

std::atomic<long long> g_in_use{0};
std::atomic<long long> g_peak{0};

// System allocator of the platform, bypassing the hooks below
#if defined(__APPLE__)
void* sysMalloc(std::size_t n) { return malloc_zone_malloc(malloc_default_zone(), n); }
void* sysCalloc(std::size_t c, std::size_t n) { return malloc_zone_calloc(malloc_default_zone(), c, n); }
void* sysRealloc(void* q, std::size_t n) { return malloc_zone_realloc(malloc_zone_from_ptr(q), q, n); }
void* sysMemalign(std::size_t a, std::size_t n) { return malloc_zone_memalign(malloc_default_zone(), a, n); }
void sysFree(void* q) { malloc_zone_free(malloc_zone_from_ptr(q), q); }
long long blockBytes(const void* q) { return static_cast<long long>(malloc_size(q)); }
#else
void* sysMalloc(std::size_t n) { return __libc_malloc(n); }
void* sysCalloc(std::size_t c, std::size_t n) { return __libc_calloc(c, n); }
void* sysRealloc(void* q, std::size_t n) { return __libc_realloc(q, n); }
void* sysMemalign(std::size_t a, std::size_t n) { return __libc_memalign(a, n); }
void sysFree(void* q) { __libc_free(q); }
long long blockBytes(void* q) { return static_cast<long long>(malloc_usable_size(q)); }
#endif

void countIn(void* q) {
    if (q == nullptr) { return; }
    const long long now = g_in_use.fetch_add(blockBytes(q)) + blockBytes(q);
    long long peak = g_peak.load();
    while (now > peak && !g_peak.compare_exchange_weak(peak, now)) {}
}

void countOut(void* q) {
    if (q != nullptr) { g_in_use.fetch_sub(blockBytes(q)); }
}

} // namespace

extern "C" {

void* malloc(std::size_t n) TREX_MALLOC_NOEXCEPT {
    void* q = sysMalloc(n);
    countIn(q);
    return q;
}

void* calloc(std::size_t c, std::size_t n) TREX_MALLOC_NOEXCEPT {
    void* q = sysCalloc(c, n);
    countIn(q);
    return q;
}

void* realloc(void* old, std::size_t n) TREX_MALLOC_NOEXCEPT {
    if (old == nullptr) { return malloc(n); }
    const long long before = blockBytes(old);
    void* q = sysRealloc(old, n);
    if (q != nullptr) {
        g_in_use.fetch_sub(before);
        countIn(q);
    }
    return q;
}

void free(void* q) TREX_MALLOC_NOEXCEPT {
    if (q == nullptr) { return; }
    countOut(q);
    sysFree(q);
}

int posix_memalign(void** out, std::size_t alignment, std::size_t n) TREX_MALLOC_NOEXCEPT {
    void* q = sysMemalign(alignment, n);
    if (q == nullptr) { return ENOMEM; }
    countIn(q);
    *out = q;
    return 0;
}

void* aligned_alloc(std::size_t alignment, std::size_t n) TREX_MALLOC_NOEXCEPT {
    void* q = sysMemalign(alignment, n);
    countIn(q);
    return q;
}

#if !defined(__APPLE__)
void* memalign(std::size_t alignment, std::size_t n) TREX_MALLOC_NOEXCEPT {
    void* q = sysMemalign(alignment, n);
    countIn(q);
    return q;
}
#endif

} // extern "C"

// Global operator new / delete through the hooked malloc (libc++ would call the
// system allocator directly)
void* operator new(std::size_t n) {
    void* q = std::malloc(n == 0 ? 1 : n);
    if (q == nullptr) { throw std::bad_alloc(); }
    return q;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept { return std::malloc(n == 0 ? 1 : n); }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { return std::malloc(n == 0 ? 1 : n); }
void operator delete(void* q) noexcept { std::free(q); }
void operator delete[](void* q) noexcept { std::free(q); }
void operator delete(void* q, std::size_t) noexcept { std::free(q); }
void operator delete[](void* q, std::size_t) noexcept { std::free(q); }
void operator delete(void* q, const std::nothrow_t&) noexcept { std::free(q); }
void operator delete[](void* q, const std::nothrow_t&) noexcept { std::free(q); }

void* operator new(std::size_t n, std::align_val_t a) {
    void* q = nullptr;
    if (posix_memalign(&q, static_cast<std::size_t>(a), n == 0 ? 1 : n) != 0) {
        throw std::bad_alloc();
    }
    return q;
}
void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
void operator delete(void* q, std::align_val_t) noexcept { std::free(q); }
void operator delete[](void* q, std::align_val_t) noexcept { std::free(q); }
void operator delete(void* q, std::size_t, std::align_val_t) noexcept { std::free(q); }
void operator delete[](void* q, std::size_t, std::align_val_t) noexcept { std::free(q); }

#endif // TREX_COUNTING_ALLOCATOR

// ========================================================================================

// Embed into test namespace
namespace trex::test::tsolvers::ad::memory_peak {

using namespace trex::tsolvers::ad::linear_model::lars_based;
using namespace trex::tsolvers::ad::linear_model::omp_based;
using namespace trex::tsolvers::ad::linear_model::afs_based;
using namespace trex::tsolvers::ad::linear_model::cd_based;
using trex::tsolvers::ad::SolverOptions;
namespace datagen = trex::utils::datageneration::datagen;

/** @brief Peak and held bytes of one solver run, relative to the heap in use
 *  before the solver was constructed. */
struct Measurement {
    long long held_allocator{0};   // in use after the step (object included)
    long long peak_step{0};        // high-water mark inside executeStep()
    std::size_t reported{0};       // memoryBytes() after the step
    std::size_t object{0};         // sizeof the solver object
};

/** @brief Pre-scaled problem as in the Examples memory demo (centered unit-norm
 *  columns, centered y; the solvers run as precentered readers). */
struct Problem {
    Eigen::MatrixXd X, D;
    Eigen::VectorXd y;
};

Problem makeProblem(Eigen::Index n, Eigen::Index p, Eigen::Index L) {
    datagen::SyntheticData data(
        n, p, L, {0, 1, 2, 3, 4}, {3.0, -2.0, 2.5, 1.5, -1.5}, 1.0, 11, -1, -1,
        datagen::predictor_policy::Normal(),
        datagen::dummygen::Distribution::Normal(),
        datagen::noisegen::noise_policy::Normal());
    Problem prob{data.getX(), data.getD(), data.getY()};
    for (Eigen::MatrixXd* M : {&prob.X, &prob.D}) {
        M->rowwise() -= M->colwise().mean();
        M->array().rowwise() /= M->colwise().norm().array();
    }
    prob.y.array() -= prob.y.mean();
    return prob;
}

template <typename T>
std::unique_ptr<T> makeSolver(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::MatrixXd>& D,
                              Eigen::Map<Eigen::VectorXd>& y) {
    SolverOptions o;
    o.normalize = false;
    o.intercept = false;
    o.precentered = true;
    o.lambda2 = 0.5;
    const Eigen::Index p = X.cols();
    if constexpr (std::is_same_v<T, TIENET_Solver> || std::is_same_v<T, TIENETAug_Solver> ||
                  std::is_same_v<T, TCDIENET_Solver>) {
        // groups of five neighbouring variables
        Eigen::VectorXi groups(p);
        for (Eigen::Index j = 0; j < p; ++j) { groups(j) = static_cast<int>(j / 5); }
        return std::make_unique<T>(X, D, y, groups, o);
    } else if constexpr (std::is_same_v<T, TIKNET_Solver> || std::is_same_v<T, TCDTIKNET_Solver>) {
        Eigen::SparseMatrix<double> K(p, p);
        K.setIdentity();
        return std::make_unique<T>(X, D, y, K, o);
    } else {
        return std::make_unique<T>(X, D, y, o);
    }
}

// Measured on macOS arm64 and Linux arm64 (2026-09-28), n = 200, p = L = 500,
// T = 10, peak / (memoryBytes() + sizeof): OMP, MP, OLS, AFS 1.02-1.20; LARS
// family and augmented wrappers 1.00-1.45; CD family 1.21-4.30 (the certified
// localization builds the restricted Gram H_SS, |S| up to 400, for an
// eigen-solve when the penalty is not strongly convex). Held: 1.00-1.09.

/** @brief Upper bound of peak / held per solver family, with headroom over the
 *  measured values above for other platforms, allocators and data. */
template <typename T>
constexpr double peakBound() {
    if constexpr (std::is_base_of_v<TLARS_Solver, T> || std::is_same_v<T, TENETAug_Solver> ||
                  std::is_same_v<T, TIENETAug_Solver>) {
        return 2.0;
    } else if constexpr (std::is_base_of_v<TCDLASSO_Solver, T>) {
        return 6.0;
    } else {
        return 1.5;
    }
}

/** @brief One problem for all solvers (readers never write it). */
Problem& sharedProblem() {
    static Problem prob = makeProblem(200, 500, 500);
    return prob;
}

template <typename T>
class TSolverMemoryPeak : public ::testing::Test {};

typedef ::testing::Types<
    TLARS_Solver, TLASSO_Solver, TSTEPWISE_Solver, TSTAGEWISE_Solver, TENET_Solver,
    TENETAug_Solver, TIENET_Solver, TIENETAug_Solver, TIKNET_Solver,
    TOMP_Solver, TGP_Solver, TACGP_Solver, TMP_Solver, TNCGMP_Solver, TOOLS_Solver,
    TAFS_Solver,
    TCDLASSO_Solver, TCDENET_Solver, TCDIENET_Solver, TCDTIKNET_Solver
> AllSolvers;

TYPED_TEST_SUITE(TSolverMemoryPeak, AllSolvers);

/** @brief The report matches the allocator's held bytes, and the peak of a
 *  step stays within the documented factor of the held state. */
TYPED_TEST(TSolverMemoryPeak, PeakOfAStepAgainstTheReport) {
#if !TREX_COUNTING_ALLOCATOR
    GTEST_SKIP() << "counting allocator only on macOS and glibc, not under sanitizers";
#else
    Problem& prob = sharedProblem();
    Eigen::Map<Eigen::MatrixXd> X(prob.X.data(), prob.X.rows(), prob.X.cols());
    Eigen::Map<Eigen::MatrixXd> D(prob.D.data(), prob.D.rows(), prob.D.cols());
    Eigen::Map<Eigen::VectorXd> y(prob.y.data(), prob.y.size());

    Measurement m;
    const long long before = g_in_use.load();
    auto solver = makeSolver<TypeParam>(X, D, y);
    g_peak.store(g_in_use.load());
    solver->executeStep(10, true);
    m.peak_step = g_peak.load() - before;
    m.held_allocator = g_in_use.load() - before;
    m.reported = solver->memoryBytes();
    m.object = sizeof(TypeParam);

    const double reported_with_object = static_cast<double>(m.reported + m.object);
    const double held_ratio = static_cast<double>(m.held_allocator) / reported_with_object;
    const double peak_ratio = static_cast<double>(m.peak_step) / reported_with_object;
    ::testing::Test::RecordProperty("memory_bytes", std::to_string(m.reported));
    ::testing::Test::RecordProperty("held_ratio", std::to_string(held_ratio));
    ::testing::Test::RecordProperty("peak_ratio", std::to_string(peak_ratio));
    std::printf("%-10s memoryBytes %9zu  held/report %.3f  peak/report %.3f\n",
                solver->solverTypeToString().c_str(), m.reported, held_ratio, peak_ratio);

    // Held: the allocator rounds blocks up
    EXPECT_GE(held_ratio, 0.98);
    EXPECT_LE(held_ratio, 1.15);
    // Peak: the temporaries of a step on top of the held state
    EXPECT_GE(peak_ratio, 0.98);
    EXPECT_LE(peak_ratio, peakBound<TypeParam>());
#endif
}

// ========================================================================================
} /* End of namespace trex::test::tsolvers::ad::memory_peak */
