// ===================================================================================
// test_trex_thread_determinism.cpp
// ===================================================================================
/**
 * @file test_trex_thread_determinism.cpp
 *
 * @brief Selections are bitwise independent of the thread options.
 *
 * @details Every selector runs a seeded problem under six thread settings:
 *  experiments in sequence with a solver budget of 1, 2 and 4, experiments in
 *  parallel on 2 and 4 threads, and both options at 4 at once. The
 *  selected support, the stopping time, the voting threshold, the Phi and
 *  FDP-hat matrices and the R matrix of every setting equal those of the
 *  sequential run with one thread. Experiment k derives its own seed, every
 *  reduction is summed in a fixed order, and a pass split over threads writes
 *  its results per element, so no setting may move a bit.
 *
 *  Without OpenMP the settings are inert and the identities hold trivially;
 *  the test still runs, so the option plumbing is exercised in every build.
 */
// ===================================================================================

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cstddef>
#include <vector>

#include <trex_selector_methods/ad/trex_core/trex.hpp>
#include <trex_selector_methods/ad/trex_da/trex_da.hpp>
#include <trex_selector_methods/ad/trex_gvs/trex_gvs.hpp>
#include <trex_selector_methods/ad/trex_screening/trex_screening.hpp>
#include <trex_selector_methods/ad/trex_screening/trex_biobank_screening.hpp>
#include <trex_selector_methods/ad/trex_spca/trex_spca.hpp>
#include <trex_selector_methods/ad/trex_tikhonov/trex_tikhonov.hpp>

#include <utils/datageneration/utils_datagen.hpp>
#include <utils/openmp/utils_openmp.hpp>

// ===================================================================================

namespace trex::test::trex_selector_methods::ad::trex_core {

using namespace trex::trex_selector_methods::ad::trex_core;
using namespace trex::utils::datageneration::datagen;
namespace da   = trex::trex_selector_methods::ad::trex_da;
namespace gvs  = trex::trex_selector_methods::ad::trex_gvs;
namespace scr  = trex::trex_selector_methods::ad::trex_screening;
namespace bio  = trex::trex_selector_methods::ad::trex_biobank_screening;
namespace spca = trex::trex_selector_methods::ad::trex_spca;
namespace tik  = trex::trex_selector_methods::ad::trex_tikhonov;

namespace {

/** @brief One thread setting of the control. */
struct ThreadSetting {
    bool parallel;
    int outer;
    int inner;
    const char* label;
};

/** @brief The reference comes first: experiments in sequence, one thread. */
constexpr ThreadSetting kSettings[] = {
    {false, 1, 1, "sequential, one thread"},
    {false, 1, 2, "sequential, solver budget 2"},
    {false, 1, 4, "sequential, solver budget 4"},
    {true,  2, 1, "parallel experiments on 2 threads"},
    {true,  4, 1, "parallel experiments on 4 threads"},
    {true,  4, 4, "parallel experiments on 4 threads with a solver budget of 4"},
};

void applySetting(TRexControlParameter& ctrl, const ThreadSetting& s) {
    ctrl.parallel_rnd_experiments = s.parallel;
    ctrl.max_outer_threads = s.outer;
    ctrl.max_inner_threads = s.inner;
    ctrl.K = 4;
    ctrl.max_dummy_multiplier = 2;
}

/** @brief The shared seeded problem; every run gets private copies. */
struct Problem {
    Eigen::MatrixXd X;
    Eigen::VectorXd y;

    Problem() {
        std::vector<std::size_t> support = {1, 2, 3, 7};
        std::vector<double> coefs = {5.0, -3.0, 2.0, 4.0};
        SyntheticData data(120, 40, support, coefs, 1.0, 42);
        X = data.getX();
        y = data.getY();
    }
};

template <class Derived>
void expectBitwiseEqual(const Eigen::MatrixBase<Derived>& a,
                        const Eigen::MatrixBase<Derived>& b, const char* what) {
    ASSERT_EQ(a.rows(), b.rows()) << what;
    ASSERT_EQ(a.cols(), b.cols()) << what;
    EXPECT_TRUE((a.array() == b.array()).all()) << what << " differs from the sequential run";
}

void expectSameSelection(const TRexSelector::SelectionResult& ref,
                         const TRexSelector::SelectionResult& run) {
    expectBitwiseEqual(ref.selected_var, run.selected_var, "selected_var");
    EXPECT_EQ(ref.T_stop, run.T_stop);
    EXPECT_EQ(ref.v_thresh, run.v_thresh);
    EXPECT_EQ(ref.dummy_factor_L, run.dummy_factor_L);
    EXPECT_EQ(ref.num_dummies, run.num_dummies);
    expectBitwiseEqual(ref.Phi_mat, run.Phi_mat, "Phi_mat");
    expectBitwiseEqual(ref.FDP_hat_mat, run.FDP_hat_mat, "FDP_hat_mat");
    expectBitwiseEqual(ref.R_mat, run.R_mat, "R_mat");
    expectBitwiseEqual(ref.Phi_prime, run.Phi_prime, "Phi_prime");
}

/** @brief Run every setting and compare it with the first. */
template <class Run, class Compare>
void checkEverySetting(Run run, Compare compare) {
    const auto ref = run(kSettings[0]);
    for (std::size_t i = 1; i < std::size(kSettings); ++i) {
        SCOPED_TRACE(kSettings[i].label);
        compare(ref, run(kSettings[i]));
    }
}

}  // namespace

// ===================================================================================

TEST(TRexThreadDeterminismTest, CoreSelector) {
    Problem pb;
    for (auto strategy : {LLoopStrategy::HCONCAT, LLoopStrategy::STANDARD,
                          LLoopStrategy::SEEDED, LLoopStrategy::PERMUTATION_SEEDED}) {
        SCOPED_TRACE(static_cast<int>(strategy));
        checkEverySetting(
            [&](const ThreadSetting& s) {
                Eigen::MatrixXd X = pb.X;
                Eigen::VectorXd y = pb.y;
                Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
                Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
                TRexControlParameter ctrl;
                applySetting(ctrl, s);
                ctrl.lloop_strategy = strategy;
                TRexSelector sel(Xm, ym, 0.1, ctrl, 42, false);
                return sel.select();
            },
            expectSameSelection);
    }
}


TEST(TRexThreadDeterminismTest, DependencyAwareSelector) {
    Problem pb;
    checkEverySetting(
        [&](const ThreadSetting& s) {
            Eigen::MatrixXd X = pb.X;
            Eigen::VectorXd y = pb.y;
            Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
            Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
            da::TRexDAControlParameter ctrl;
            ctrl.method = da::DAMethod::AR1;
            applySetting(ctrl.trex_ctrl, s);
            da::TRexDASelector sel(Xm, ym, 0.1, ctrl, 42, false);
            return sel.select();
        },
        expectSameSelection);
}


TEST(TRexThreadDeterminismTest, GroupedVariableSelector) {
    Problem pb;
    checkEverySetting(
        [&](const ThreadSetting& s) {
            Eigen::MatrixXd X = pb.X;
            Eigen::VectorXd y = pb.y;
            Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
            Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
            gvs::TRexGVSControlParameter ctrl;
            ctrl.gvs_type = gvs::GVSType::EN;
            ctrl.lambda_2 = 1.0;
            applySetting(ctrl.trex_ctrl, s);
            gvs::TRexGVSSelector sel(Xm, ym, 0.1, ctrl, 42, false);
            return sel.select();
        },
        expectSameSelection);
}


TEST(TRexThreadDeterminismTest, ScreeningSelector) {
    Problem pb;
    checkEverySetting(
        [&](const ThreadSetting& s) {
            Eigen::MatrixXd X = pb.X;
            Eigen::VectorXd y = pb.y;
            Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
            Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
            scr::ScreenTRexControlParameter ctrl;
            ctrl.trex_method = scr::ScreenTRexMethod::TREX;
            applySetting(ctrl.trex_ctrl, s);
            ctrl.trex_ctrl.lloop_strategy = LLoopStrategy::STANDARD;
            scr::ScreenTRexSelector sel(Xm, ym, ctrl, 42, false);
            sel.select();
            return sel.getScreenResult();   // select() slices to the base type
        },
        [](const scr::ScreenTRexSelectionResult& ref, const scr::ScreenTRexSelectionResult& run) {
            expectBitwiseEqual(ref.selected_var, run.selected_var, "selected_var");
            EXPECT_EQ(ref.estimated_FDR, run.estimated_FDR);
            expectBitwiseEqual(ref.beta_mat, run.beta_mat, "beta_mat");
        });
}


TEST(TRexThreadDeterminismTest, BiobankScreeningSelector) {
    Problem pb;
    checkEverySetting(
        [&](const ThreadSetting& s) {
            Eigen::MatrixXd X = pb.X;
            Eigen::VectorXd y = pb.y;
            Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
            Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
            bio::BiobankScreenTRexControl ctrl;
            ctrl.trex_screen_ctrl.trex_method = scr::ScreenTRexMethod::TREX;
            applySetting(ctrl.trex_screen_ctrl.trex_ctrl, s);
            ctrl.trex_screen_ctrl.trex_ctrl.lloop_strategy = LLoopStrategy::STANDARD;
            bio::BiobankScreenTRex sel(Xm, ym, ctrl, 42, false);
            return sel.screenPhenotype();
        },
        [](const bio::BiobankScreenTRexResult& ref, const bio::BiobankScreenTRexResult& run) {
            EXPECT_EQ(ref.selected_indices, run.selected_indices);
            EXPECT_EQ(ref.estimated_FDR, run.estimated_FDR);
            EXPECT_EQ(ref.method_used, run.method_used);
        });
}


TEST(TRexThreadDeterminismTest, SparsePcaSelector) {
    Problem pb;
    checkEverySetting(
        [&](const ThreadSetting& s) {
            Eigen::MatrixXd X = pb.X;
            Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
            spca::TRexSPCAControlParameter ctrl;
            ctrl.gvs_ctrl.lambda_2 = 1.0;
            applySetting(ctrl.gvs_ctrl.trex_ctrl, s);
            spca::TRexSPCA sel(Xm, 2, 0.1, ctrl, 42, false);
            return sel.select();
        },
        [](const spca::TRexSPCAResult& ref, const spca::TRexSPCAResult& run) {
            EXPECT_EQ(ref.active_sets, run.active_sets);
            expectBitwiseEqual(ref.V, run.V, "V");
            expectBitwiseEqual(ref.Z, run.Z, "Z");
        });
}


TEST(TRexThreadDeterminismTest, TikhonovSelector) {
    Problem pb;
    const Eigen::Index p = pb.X.cols();
    checkEverySetting(
        [&](const ThreadSetting& s) {
            Eigen::MatrixXd X = pb.X;
            Eigen::VectorXd y = pb.y;
            Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
            Eigen::Map<Eigen::VectorXd> ym(y.data(), y.size());
            tik::TRexTikhonovControlParameter ctrl;
            Eigen::SparseMatrix<double> K(p, p);
            K.setIdentity();
            ctrl.tikhonov_K = K;
            ctrl.lambda_2 = 1.0;
            applySetting(ctrl.trex_ctrl, s);
            tik::TRexTikhonovSelector sel(Xm, ym, 0.1, ctrl, 42, false);
            return sel.select();
        },
        expectSameSelection);
}

// ========================================================================================
} /* End of namespace trex::test::trex_selector_methods::ad::trex_core */
