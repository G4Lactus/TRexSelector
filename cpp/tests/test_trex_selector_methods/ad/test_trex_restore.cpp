// ========================================================================================
/**
 * @file test_trex_restore.cpp
 *
 * @brief TRexSelector::restore(): give X back before select().
 *
 * @details The selector normalizes X in place at construction and restores
 *          it when select() returns (and in the destructor). restore() hands
 *          X back right away for a selector that will not run; it is
 *          idempotent, a no-op after select(), and a restored selector
 *          refuses to run. The screening selector, which overrides select(),
 *          honours the same rule.
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <stdexcept>
#include <vector>

// project includes
#include <trex_selector_methods/ad/trex_core/trex.hpp>
#include <trex_selector_methods/ad/trex_screening/trex_screening.hpp>
#include <tsolvers/ad/solver_utils/buffer_registry.hpp>
#include <utils/datageneration/utils_datagen.hpp>

// Eigen includes
#include <Eigen/Dense>

// ========================================================================================

namespace trex::test::trex_selector_methods::restore {

namespace datagen = trex::utils::datageneration::datagen;
namespace bufreg = trex::tsolvers::ad::solver_utils::buffer_registry;
using trex::trex_selector_methods::ad::trex_core::TRexSelector;
using trex::trex_selector_methods::ad::trex_core::TRexControlParameter;
using trex::trex_selector_methods::ad::trex_screening::ScreenTRexSelector;
using trex::trex_selector_methods::ad::trex_screening::ScreenTRexControlParameter;
using trex::trex_selector_methods::ad::utils::solver_dispatch::SolverTypeForTRex;

namespace {

struct Data {
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    Data() {
        const Eigen::Index n = 50, p = 40;
        std::vector<std::size_t> sup = {2, 9, 17};
        std::vector<double> coefs(sup.size(), 2.0);
        datagen::SyntheticData data(n, p, sup, coefs, /*snr=*/3.0, /*seed=*/4242,
                                    datagen::predictor_policy::Normal(),
                                    datagen::noisegen::noise_policy::Normal());
        X = data.getX();
        y = data.getY();
    }
};

TRexControlParameter smallControl() {
    TRexControlParameter ctrl;
    ctrl.K = 5;
    ctrl.max_dummy_multiplier = 2;
    ctrl.solver_type = SolverTypeForTRex::TLARS;
    return ctrl;
}

} // namespace

// ========================================================================================

/** @brief restore() before select() gives X back and releases the claim;
 *         it is idempotent; select() afterwards throws. */
TEST(TRexRestore, RestoreBeforeSelectGivesXBack) {
    Data d;
    const Eigen::MatrixXd X0 = d.X;
    Eigen::Map<Eigen::MatrixXd> Xm(d.X.data(), d.X.rows(), d.X.cols());
    Eigen::Map<Eigen::VectorXd> ym(d.y.data(), d.y.size());

    TRexSelector sel(Xm, ym, /*tFDR=*/0.2, smallControl(), /*seed=*/3, /*verbose=*/false);
    EXPECT_FALSE(d.X.isApprox(X0, 1e-12));           // normalized in place
    EXPECT_TRUE(bufreg::isClaimed(d.X.data()));

    sel.restore();
    EXPECT_TRUE(d.X.isApprox(X0, 1e-12));
    EXPECT_FALSE(bufreg::isClaimed(d.X.data()));

    sel.restore();                                    // idempotent
    EXPECT_TRUE(d.X.isApprox(X0, 1e-12));

    EXPECT_THROW(sel.select(), std::runtime_error);   // a restored selector refuses to run
    EXPECT_TRUE(d.X.isApprox(X0, 1e-12));             // and left X alone

    // The buffer is free for a new selector, which runs normally
    TRexSelector fresh(Xm, ym, 0.2, smallControl(), 3, false);
    EXPECT_NO_THROW(fresh.select());
    EXPECT_TRUE(d.X.isApprox(X0, 1e-12));
}


/** @brief After select() the selector has restored X itself; restore() is
 *         a no-op and does not move the data. */
TEST(TRexRestore, RestoreAfterSelectIsNoOp) {
    Data d;
    const Eigen::MatrixXd X0 = d.X;
    Eigen::Map<Eigen::MatrixXd> Xm(d.X.data(), d.X.rows(), d.X.cols());
    Eigen::Map<Eigen::VectorXd> ym(d.y.data(), d.y.size());

    TRexSelector sel(Xm, ym, 0.2, smallControl(), 3, false);
    sel.select();
    EXPECT_TRUE(d.X.isApprox(X0, 1e-12));
    const Eigen::MatrixXd X_after = d.X;
    sel.restore();
    EXPECT_EQ((d.X - X_after).cwiseAbs().maxCoeff(), 0.0);
}


/** @brief The screening selector overrides select() and honours the rule. */
TEST(TRexRestore, ScreeningSelectorHonoursRestore) {
    Data d;
    const Eigen::MatrixXd X0 = d.X;
    Eigen::Map<Eigen::MatrixXd> Xm(d.X.data(), d.X.rows(), d.X.cols());
    Eigen::Map<Eigen::VectorXd> ym(d.y.data(), d.y.size());

    ScreenTRexControlParameter sctrl;
    ScreenTRexSelector sel(Xm, ym, sctrl, /*seed=*/7);
    sel.restore();
    EXPECT_TRUE(d.X.isApprox(X0, 1e-12));
    EXPECT_THROW(sel.select(), std::runtime_error);
}

} // namespace trex::test::trex_selector_methods::restore
