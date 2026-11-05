// ========================================================================================
// test_tsolvers_buffer_guard.cpp
// ========================================================================================
/**
 * @file test_tsolvers_buffer_guard.cpp
 *
 * @brief Tests for the shared-buffer claim of the T-solvers:
 *        the single-owner invariant for buffers preprocessed in place.
 *
 * @details A T-solver constructed with normalize = true or intercept = true
 *          is a WRITER: it transforms the caller's X and D in place and holds
 *          a claim on both buffers until restore() or destruction. A second
 *          writer on a claimed buffer throws, because a second in-place
 *          transformation is not idempotent (rounding moves the data, and
 *          downstream near-ties resolve differently — the T-Stagewise
 *          checkpoint finding). READERS (normalize = false, intercept =
 *          false) and solvers resumed from a checkpoint never claim. The
 *          registry is shared with the T-Rex selectors. The last test pins
 *          the recommended pipeline pattern: pre-scale once in front, then
 *          reference / checkpoint / resumed solvers on the same buffers as
 *          readers, bit-identical paths.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>

// std includes
#include <filesystem>
#include <stdexcept>
#include <string>

// tsolvers includes
#include <tsolvers/ad/linear_model/lars_based/tlars_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tstagewise_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tenet_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tomp_solver.hpp>
#include <tsolvers/ad/solver_utils/buffer_registry.hpp>

// selector include (shared registry)
#include <trex_selector_methods/ad/trex_core/trex.hpp>

// ml_methods includes
#include <ml_methods/scaler_methods/lp_norm_scaler.hpp>

// utils includes
#include <utils/datageneration/utils_datagen.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::tsolvers::ad::buffer_guard {

using namespace trex::tsolvers::ad::linear_model::lars_based;
using namespace trex::tsolvers::ad::linear_model::omp_based;
using trex::tsolvers::ad::SolverOptions;
using trex::ml_methods::scaler_methods::LpNormScaler;
using trex::trex_selector_methods::ad::trex_core::TRexSelector;
using trex::trex_selector_methods::ad::trex_core::TRexControlParameter;
using trex::trex_selector_methods::ad::utils::solver_dispatch::SolverTypeForTRex;
namespace bufreg = trex::tsolvers::ad::solver_utils::buffer_registry;
namespace datagen = trex::utils::datageneration::datagen;
namespace fs = std::filesystem;

// ========================================================================================

class TSolverBufferGuardTest : public ::testing::Test {
protected:
    Eigen::MatrixXd X, D;
    Eigen::VectorXd y;

    void SetUp() override {
        datagen::SyntheticData data(
            100, 50, 50,
            {3, 10, 20},
            {3.0, -2.0, 2.5},
            1.0, 42, -1, -1,
            datagen::predictor_policy::Normal(),
            datagen::dummygen::Distribution::Normal(),
            datagen::noisegen::noise_policy::Normal()
        );
        X = data.getX();
        D = data.getD();
        y = data.getY();
    }

    Eigen::Map<Eigen::MatrixXd> mapX() { return {X.data(), X.rows(), X.cols()}; }
    Eigen::Map<Eigen::MatrixXd> mapD() { return {D.data(), D.rows(), D.cols()}; }
    Eigen::Map<Eigen::VectorXd> mapY() { return {y.data(), y.size()}; }

    static SolverOptions writer() {
        return SolverOptions{.normalize = true, .intercept = true, .verbose = false};
    }
    static SolverOptions reader() {
        return SolverOptions{.normalize = false, .intercept = false, .verbose = false};
    }
};

// ========================================================================================
// Registry primitives
// ========================================================================================

/** @brief The registry itself: claim, double claim throws, release, RAII. */
TEST(BufferRegistry, ClaimReleaseAndRaii) {
    int dummy = 0;
    const void* ptr = &dummy;

    EXPECT_FALSE(bufreg::isClaimed(ptr));
    bufreg::claimOrThrow(ptr, "first");
    EXPECT_TRUE(bufreg::isClaimed(ptr));
    EXPECT_THROW(bufreg::claimOrThrow(ptr, "second"), std::runtime_error);
    bufreg::release(ptr);
    EXPECT_FALSE(bufreg::isClaimed(ptr));

    // Null pointers are never claimed and never throw
    EXPECT_NO_THROW(bufreg::claimOrThrow(nullptr, "null"));
    EXPECT_FALSE(bufreg::isClaimed(nullptr));

    {
        bufreg::BufferClaim claim(ptr, "raii");
        EXPECT_TRUE(claim.active());
        EXPECT_TRUE(bufreg::isClaimed(ptr));

        // The claim travels with a move
        bufreg::BufferClaim moved(std::move(claim));
        EXPECT_FALSE(claim.active());
        EXPECT_TRUE(moved.active());
        EXPECT_TRUE(bufreg::isClaimed(ptr));

        moved.release();
        EXPECT_FALSE(moved.active());
        EXPECT_FALSE(bufreg::isClaimed(ptr));
        moved.release();   // idempotent
    }
    EXPECT_FALSE(bufreg::isClaimed(ptr));

    {
        bufreg::BufferClaim scoped(ptr, "scoped");
        EXPECT_TRUE(bufreg::isClaimed(ptr));
    }
    EXPECT_FALSE(bufreg::isClaimed(ptr));   // released by the destructor
}

// ========================================================================================
// Writer claims
// ========================================================================================

/** @brief A writer holds X and D while alive; a second writer throws. */
TEST_F(TSolverBufferGuardTest, SecondWriterOnLiveBufferThrows) {
    auto Xm = mapX(); auto Dm = mapD(); auto ym = mapY();

    TSTAGEWISE_Solver first(Xm, Dm, ym, writer());
    EXPECT_TRUE(bufreg::isClaimed(X.data()));
    EXPECT_TRUE(bufreg::isClaimed(D.data()));

    // Same family or another family: the buffer is the buffer.
    EXPECT_THROW(TSTAGEWISE_Solver second(Xm, Dm, ym, writer()), std::runtime_error);
    EXPECT_THROW(TLARS_Solver third(Xm, Dm, ym, writer()), std::runtime_error);
    EXPECT_THROW(TOMP_Solver fourth(Xm, Dm, ym, writer()), std::runtime_error);

    // The message tells the user what to do
    try {
        TLARS_Solver again(Xm, Dm, ym, writer());
        FAIL() << "expected a throw";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("already held"), std::string::npos);
        EXPECT_NE(msg.find("normalize = false"), std::string::npos);
        EXPECT_NE(msg.find("restore()"), std::string::npos);
    }

    // The failed constructions did not disturb the first writer's claim
    EXPECT_TRUE(bufreg::isClaimed(X.data()));
    EXPECT_TRUE(bufreg::isClaimed(D.data()));
}


/** @brief Intercept-only and normalize-only solvers are writers too. */
TEST_F(TSolverBufferGuardTest, InterceptOnlyAndNormalizeOnlyClaim) {
    auto Xm = mapX(); auto Dm = mapD(); auto ym = mapY();
    {
        TLARS_Solver s(Xm, Dm, ym,
                       SolverOptions{.normalize = false, .intercept = true, .verbose = false});
        EXPECT_TRUE(bufreg::isClaimed(X.data()));
        EXPECT_TRUE(bufreg::isClaimed(D.data()));
        s.restore(Xm, Dm, ym);
    }
    EXPECT_FALSE(bufreg::isClaimed(X.data()));
    {
        TLARS_Solver s(Xm, Dm, ym,
                       SolverOptions{.normalize = true, .intercept = false, .verbose = false});
        EXPECT_TRUE(bufreg::isClaimed(X.data()));
        EXPECT_TRUE(bufreg::isClaimed(D.data()));
    }
    EXPECT_FALSE(bufreg::isClaimed(X.data()));
    EXPECT_FALSE(bufreg::isClaimed(D.data()));
}


/** @brief restore() hands the buffers back: a new writer may claim them. */
TEST_F(TSolverBufferGuardTest, RestoreReleasesTheClaim) {
    auto Xm = mapX(); auto Dm = mapD(); auto ym = mapY();
    const Eigen::MatrixXd X0 = X, D0 = D;

    TSTAGEWISE_Solver first(Xm, Dm, ym, writer());
    first.executeStep(5, true);
    first.restore(Xm, Dm, ym);
    EXPECT_FALSE(bufreg::isClaimed(X.data()));
    EXPECT_FALSE(bufreg::isClaimed(D.data()));
    EXPECT_TRUE(X.isApprox(X0, 1e-12));
    EXPECT_TRUE(D.isApprox(D0, 1e-12));

    // `first` is still alive but has released: a second writer is fine now
    EXPECT_NO_THROW({
        TSTAGEWISE_Solver second(Xm, Dm, ym, writer());
        EXPECT_TRUE(bufreg::isClaimed(X.data()));
        second.restore(Xm, Dm, ym);
    });
    EXPECT_FALSE(bufreg::isClaimed(X.data()));
}


/** @brief restore() is idempotent and a no-op for readers and resumed
 *         solvers: only the object whose preprocessing is in effect writes
 *         back (holdsPreprocessed()). A second restore() must not move X. */
TEST_F(TSolverBufferGuardTest, RestoreIsIdempotentAndOnlyTheWriterWritesBack) {
    auto Xm = mapX(); auto Dm = mapD(); auto ym = mapY();
    const Eigen::MatrixXd X0 = X, D0 = D;
    const std::string checkpoint = ::testing::TempDir() + "temp_guard_idempotent.bin";

    // A reader never holds and its restore() leaves the data untouched
    {
        TSTAGEWISE_Solver r(Xm, Dm, ym, reader());
        EXPECT_FALSE(r.holdsPreprocessed());
        r.restore(Xm, Dm, ym);
        EXPECT_EQ((X - X0).cwiseAbs().maxCoeff(), 0.0);
    }

    TSTAGEWISE_Solver w(Xm, Dm, ym, writer());
    EXPECT_TRUE(w.holdsPreprocessed());
    w.executeStep(3, true);
    w.save(checkpoint);

    // A solver resumed from the checkpoint is a reader: no claim, no write-back
    {
        TSTAGEWISE_Solver resumed = TSTAGEWISE_Solver::load(checkpoint, Xm, Dm);
        EXPECT_FALSE(resumed.holdsPreprocessed());
        const Eigen::MatrixXd X_pre = X;
        resumed.restore(Xm, Dm, ym);
        EXPECT_EQ((X - X_pre).cwiseAbs().maxCoeff(), 0.0);   // untouched
        EXPECT_TRUE(bufreg::isClaimed(X.data()));            // still the writer's
    }

    // The writer restores once ...
    w.restore(Xm, Dm, ym);
    EXPECT_FALSE(w.holdsPreprocessed());
    EXPECT_TRUE(X.isApprox(X0, 1e-12));
    const Eigen::MatrixXd X_restored = X, D_restored = D;

    // ... and a second restore() is a no-op (the former unconditional inverse
    // would have scaled and shifted the data a second time)
    w.restore(Xm, Dm, ym);
    EXPECT_EQ((X - X_restored).cwiseAbs().maxCoeff(), 0.0);
    EXPECT_EQ((D - D_restored).cwiseAbs().maxCoeff(), 0.0);
    EXPECT_FALSE(bufreg::isClaimed(X.data()));
    fs::remove(checkpoint);
}


/** @brief Destruction writes back and releases the claim (restore() not
 *         called): the caller's buffers never stay preprocessed without an
 *         owner. A derived constructor that throws after the base
 *         preprocessed is covered by the same destructor. */
TEST_F(TSolverBufferGuardTest, DestructionWritesBackAndReleasesTheClaim) {
    auto Xm = mapX(); auto Dm = mapD(); auto ym = mapY();
    const Eigen::MatrixXd X0 = X, D0 = D;
    {
        TLARS_Solver s(Xm, Dm, ym, writer());
        EXPECT_TRUE(bufreg::isClaimed(X.data()));
        EXPECT_FALSE(X.isApprox(X0, 1e-12));   // preprocessed in place
    }
    EXPECT_FALSE(bufreg::isClaimed(X.data()));
    EXPECT_FALSE(bufreg::isClaimed(D.data()));
    EXPECT_TRUE(X.isApprox(X0, 1e-12));        // written back
    EXPECT_TRUE(D.isApprox(D0, 1e-12));

    // Claimable again, and the data is raw: no already-preprocessed warning
    {
        TLARS_Solver again(Xm, Dm, ym, writer());
        EXPECT_TRUE(again.getWarnings().empty());
    }

    // A constructor that throws after the base preprocessed (TENET validates
    // lambda2 after the base constructor ran) leaves X and D restored.
    EXPECT_THROW(TENET_Solver bad(Xm, Dm, ym,
                     SolverOptions{.normalize = true, .intercept = true,
                                   .verbose = false, .lambda2 = -1.0}),
                 std::invalid_argument);
    EXPECT_FALSE(bufreg::isClaimed(X.data()));
    EXPECT_TRUE(X.isApprox(X0, 1e-12));
    EXPECT_TRUE(D.isApprox(D0, 1e-12));
}


/** @brief A moved-from solver hands its claim and its write-back duty to
 *         the moved-to object; the moved-from destructor must not restore. */
TEST_F(TSolverBufferGuardTest, MoveTransfersTheClaimAndTheWriteBack) {
    auto Xm = mapX(); auto Dm = mapD(); auto ym = mapY();
    const Eigen::MatrixXd X0 = X;
    {
        TLARS_Solver src(Xm, Dm, ym, writer());
        TLARS_Solver dst(std::move(src));
        EXPECT_FALSE(src.holdsPreprocessed());   // NOLINT(bugprone-use-after-move)
        EXPECT_TRUE(dst.holdsPreprocessed());
        EXPECT_TRUE(bufreg::isClaimed(X.data()));
        // dst owns the claim: a second writer still throws
        EXPECT_THROW(TLARS_Solver w(Xm, Dm, ym, writer()), std::runtime_error);
        // src is destroyed first (reverse order) and must leave X alone: the
        // data is still preprocessed when dst's destructor writes it back.
    }
    EXPECT_FALSE(bufreg::isClaimed(X.data()));
    EXPECT_TRUE(X.isApprox(X0, 1e-12));   // exactly one write-back
}

// ========================================================================================
// Readers
// ========================================================================================

/** @brief Readers never claim, and may coexist with a live writer. */
TEST_F(TSolverBufferGuardTest, ReadersNeverClaim) {
    auto Xm = mapX(); auto Dm = mapD(); auto ym = mapY();

    {
        TSTAGEWISE_Solver r(Xm, Dm, ym, reader());
        EXPECT_FALSE(bufreg::isClaimed(X.data()));
        EXPECT_FALSE(bufreg::isClaimed(D.data()));
    }

    TSTAGEWISE_Solver w(Xm, Dm, ym, writer());
    EXPECT_TRUE(bufreg::isClaimed(X.data()));

    // Readers on the writer's (now preprocessed) buffers are allowed
    EXPECT_NO_THROW({
        TSTAGEWISE_Solver r1(Xm, Dm, ym, reader());
        TLARS_Solver r2(Xm, Dm, ym, reader());
        r1.executeStep(3, true);
        r2.executeStep(3, true);
    });
    EXPECT_TRUE(bufreg::isClaimed(X.data()));   // still the writer's
}


/** @brief A solver resumed from a checkpoint is a reader: no claim. */
TEST_F(TSolverBufferGuardTest, LoadedSolverNeverClaims) {
    auto Xm = mapX(); auto Dm = mapD(); auto ym = mapY();
    const std::string checkpoint = ::testing::TempDir() + "temp_guard_checkpoint.bin";

    {
        TSTAGEWISE_Solver w(Xm, Dm, ym, writer());
        w.executeStep(5, true);
        w.save(checkpoint);
        // Loading onto the writer's live buffers is the checkpoint pattern
        // and must not collide with the writer's claim.
        EXPECT_NO_THROW({
            TSTAGEWISE_Solver resumed = TSTAGEWISE_Solver::load(checkpoint, Xm, Dm);
            EXPECT_EQ(resumed.getNumSteps(), w.getNumSteps());
        });
        EXPECT_TRUE(bufreg::isClaimed(X.data()));
    }
    EXPECT_FALSE(bufreg::isClaimed(X.data()));

    {
        TSTAGEWISE_Solver resumed = TSTAGEWISE_Solver::load(checkpoint, Xm, Dm);
        EXPECT_FALSE(bufreg::isClaimed(X.data()));
        EXPECT_FALSE(bufreg::isClaimed(D.data()));
    }
    fs::remove(checkpoint);
}

// ========================================================================================
// Already-preprocessed warning
// ========================================================================================

/** @brief A writer on data that is already centered and unit-norm warns
 *         (does not throw): the caller probably preprocessed in front. */
TEST_F(TSolverBufferGuardTest, WriterOnPreprocessedDataWarns) {
    auto Xm = mapX(); auto Dm = mapD(); auto ym = mapY();

    // Raw data: no warning
    {
        TLARS_Solver s(Xm, Dm, ym, writer());
        EXPECT_TRUE(s.getWarnings().empty());
        s.restore(Xm, Dm, ym);
    }

    // Pre-scaled once in front (centered, unit L2)
    LpNormScaler sx(LpNormScaler::NormType::L2, true, true);
    LpNormScaler sd(LpNormScaler::NormType::L2, true, true);
    sx.fit_transform_inplace(Xm);
    sd.fit_transform_inplace(Dm);

    {
        TLARS_Solver s(Xm, Dm, ym, writer());   // a writer on preprocessed data
        ASSERT_GE(s.getWarnings().size(), 2u);   // one for X, one for D
        bool saw_x = false, saw_d = false;
        for (const auto& w : s.getWarnings()) {
            if (w.find("X") != std::string::npos && w.find("already") != std::string::npos) saw_x = true;
            if (w.find("D") != std::string::npos && w.find("already") != std::string::npos) saw_d = true;
        }
        EXPECT_TRUE(saw_x);
        EXPECT_TRUE(saw_d);
        s.restore(Xm, Dm, ym);
    }

    // A reader on preprocessed data is the intended pattern: silent
    {
        TLARS_Solver r(Xm, Dm, ym, reader());
        EXPECT_TRUE(r.getWarnings().empty());
    }

    sx.inverse_transform_inplace(Xm);
    sd.inverse_transform_inplace(Dm);
}

// ========================================================================================
// Shared registry with the T-Rex selector
// ========================================================================================

/** @brief A live selector and a live writer solver share the registry. */
TEST_F(TSolverBufferGuardTest, SelectorAndWriterSolverShareTheRegistry) {
    auto Xm = mapX(); auto Dm = mapD(); auto ym = mapY();

    TRexControlParameter ctrl;
    ctrl.K = 5;
    ctrl.max_dummy_multiplier = 2;
    ctrl.solver_type = SolverTypeForTRex::TLARS;

    {
        TRexSelector sel(Xm, ym, /*tFDR=*/0.2, ctrl, /*seed=*/7, /*verbose=*/false);
        EXPECT_TRUE(bufreg::isClaimed(X.data()));
        // A writer solver on the selector's X must throw
        EXPECT_THROW(TLARS_Solver w(Xm, Dm, ym, writer()), std::runtime_error);
        // A reader is fine
        EXPECT_NO_THROW(TLARS_Solver r(Xm, Dm, ym, reader()));
    }
    EXPECT_FALSE(bufreg::isClaimed(X.data()));

    {
        TLARS_Solver w(Xm, Dm, ym, writer());
        // A selector on the writer's X must throw
        EXPECT_THROW(TRexSelector sel(Xm, ym, 0.2, ctrl, 7, false), std::runtime_error);
        w.restore(Xm, Dm, ym);
        EXPECT_NO_THROW({
            TRexSelector sel(Xm, ym, 0.2, ctrl, 7, false);
        });
    }
    EXPECT_FALSE(bufreg::isClaimed(X.data()));
}

// ========================================================================================
// The recommended pipeline pattern (A' of the T-Stagewise finding)
// ========================================================================================

/** @brief Pre-scale once, then reference / checkpoint / resumed solvers as
 *         readers on the SAME buffers: the paths are bit-identical. This is
 *         the pattern the demos use and the one that failed before with
 *         writers on shared maps (double preprocessing). */
TEST_F(TSolverBufferGuardTest, PreScaledSharedBufferCheckpointRoundTripIsBitIdentical) {
    auto Xm = mapX(); auto Dm = mapD(); auto ym = mapY();
    const Eigen::MatrixXd X0 = X, D0 = D;
    const std::string checkpoint = ::testing::TempDir() + "temp_guard_stagewise.bin";

    LpNormScaler sx(LpNormScaler::NormType::L2, true, true);
    LpNormScaler sd(LpNormScaler::NormType::L2, true, true);
    sx.fit_transform_inplace(Xm);
    sd.fit_transform_inplace(Dm);
    // The solver centers its own internal copy of y; nothing to do for y.

    constexpr int T_partial = 7, T_final = 15;

    // Reference: continuous run
    TSTAGEWISE_Solver ref(Xm, Dm, ym, reader());
    ref.executeStep(T_final, true);

    // Checkpoint after T_partial, resume on the same buffers
    {
        TSTAGEWISE_Solver part(Xm, Dm, ym, reader());
        part.executeStep(T_partial, true);
        part.save(checkpoint);
    }
    TSTAGEWISE_Solver resumed = TSTAGEWISE_Solver::load(checkpoint, Xm, Dm);
    resumed.executeStep(T_final, true);

    EXPECT_FALSE(bufreg::isClaimed(X.data()));
    EXPECT_FALSE(bufreg::isClaimed(D.data()));

    // Bit-identical: same actions, same coefficients
    EXPECT_EQ(ref.getNumSteps(), resumed.getNumSteps());
    EXPECT_EQ(ref.getActions(), resumed.getActions());
    const Eigen::VectorXd b_ref = ref.getBeta(-1);
    const Eigen::VectorXd b_res = resumed.getBeta(-1);
    ASSERT_EQ(b_ref.size(), b_res.size());
    EXPECT_EQ((b_ref - b_res).cwiseAbs().maxCoeff(), 0.0);

    // Restore in the scaler (the statistics live there) and clean up
    sx.inverse_transform_inplace(Xm);
    sd.inverse_transform_inplace(Dm);
    EXPECT_TRUE(X.isApprox(X0, 1e-12));
    EXPECT_TRUE(D.isApprox(D0, 1e-12));
    fs::remove(checkpoint);
}

// ========================================================================================
} /* End of namespace trex::test::tsolvers::ad::buffer_guard */
