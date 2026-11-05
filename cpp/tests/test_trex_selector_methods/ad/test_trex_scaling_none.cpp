// ========================================================================================
// test_trex_scaling_none.cpp
// ========================================================================================
/**
 * @file test_trex_scaling_none.cpp
 *
 * @brief ScalingMode::NONE (user-managed design) and the normalizer's input
 *        rejection.
 *
 * @details Contracts pinned here:
 *   - NONE leaves X and y bit-identical, stores the identity parameters, and
 *     denormalization is a no-op.
 *   - A design pre-normalized with centerAndL2NormalizeX / centerY and run
 *     under NONE selects exactly what the raw design selects under L2
 *     (same seed): Phi_prime, selected_var and T_stop agree bitwise.
 *   - Intercept-free dictionaries (the motivating case): a Vandermonde +
 *     Fourier dictionary with a constant atom is rejected under L2 (the
 *     atom is constant after centering) and works under NONE, where the
 *     constant atom is selectable.
 *   - Rejection in every mode: non-finite entries in X or y, dead columns
 *     (constant under L2 / ZSCORE, all-zero under NONE), with the column
 *     named; X is unmodified afterwards. A constant, non-zero atom passes
 *     under NONE.
 *   - A z-scored design under NONE runs but warns (unit-L2 tripwire).
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// Eigen includes
#include <Eigen/Dense>

// std includes
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// project trex includes
#include <trex_selector_methods/ad/trex_core/trex.hpp>
#include <trex_selector_methods/ad/trex_da/trex_da.hpp>
#include <trex_selector_methods/ad/trex_gvs/trex_gvs.hpp>
#include <trex_selector_methods/ad/trex_utils/trex_data_normalizer.hpp>

// project utils includes
#include <utils/datageneration/utils_datagen.hpp>
#include <utils/fp_classify/fp_classify.hpp>

// ========================================================================================

namespace trex::test::trex_selector_methods::scaling_none {

using namespace trex::trex_selector_methods::ad::trex_core;
using trex::utils::datageneration::datagen::SyntheticData;
namespace dn  = trex::trex_selector_methods::ad::utils::data_normalizer;
namespace fpc = trex::utils::fp_classify;

namespace {

// ----------------------------------------------------------------------------
// Fixtures
// ----------------------------------------------------------------------------

struct RawDesign {
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
};

/** @brief The core test's synthetic design (support {1, 2, 3}). */
RawDesign syntheticDesign(Eigen::Index n = 100, Eigen::Index p = 30, int seed = 42) {
    std::vector<std::size_t> support = {1, 2, 3};
    std::vector<double> coefs = {5.0, -3.0, 2.0};
    SyntheticData data(n, p, support, coefs, 1.0, seed);
    return {data.getX(), data.getY()};
}

/** @brief Scale every column of M to unit L2 norm WITHOUT centering. */
void unitL2Columns(Eigen::MatrixXd& M) {
    for (Eigen::Index j = 0; j < M.cols(); ++j) {
        M.col(j) /= M.col(j).norm();
    }
}

/**
 * @brief An intercept-free dictionary: a constant (DC) atom, two monomials
 *        (Vandermonde), and cos / sin atoms of harmonics 1..8 on a partial
 *        window t in [0, 1.3) so that no atom is mean-free. All columns are
 *        unit-L2 and NOT centered — centering would zero the DC atom and
 *        destroy the atom structure.
 *
 * Column layout: 0 = DC, 1 = t, 2 = t^2, 3 + 2(k-1) = cos(2 pi k t),
 * 4 + 2(k-1) = sin(2 pi k t) for k = 1..8 (p = 19).
 */
struct Dictionary {
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    std::vector<std::size_t> support;  // DC, cos(3), sin(5)
};

Dictionary interceptFreeDictionary(Eigen::Index n = 120, int seed = 7) {
    constexpr int harmonics = 8;
    const Eigen::Index p = 3 + 2 * harmonics;
    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        const double t = 1.3 * static_cast<double>(i) / static_cast<double>(n);
        X(i, 0) = 1.0;
        X(i, 1) = t;
        X(i, 2) = t * t;
        for (int k = 1; k <= harmonics; ++k) {
            X(i, 3 + 2 * (k - 1)) = std::cos(2.0 * std::numbers::pi * k * t);
            X(i, 4 + 2 * (k - 1)) = std::sin(2.0 * std::numbers::pi * k * t);
        }
    }
    unitL2Columns(X);

    const std::size_t dc = 0;
    const std::size_t cos3 = 3 + 2 * (3 - 1);
    const std::size_t sin5 = 4 + 2 * (5 - 1);

    std::mt19937 rng(static_cast<unsigned>(seed));
    std::normal_distribution<double> noise(0.0, 0.05);
    Eigen::VectorXd y = 4.0 * X.col(dc) + 2.0 * X.col(cos3) - 3.0 * X.col(sin5);
    for (Eigen::Index i = 0; i < n; ++i) { y(i) += noise(rng); }

    return {X, y, {dc, cos3, sin5}};
}

/** @brief Run the base selector on (X, y) with the given scaling mode. */
struct SelectorRun {
    Eigen::VectorXi selected_var;
    Eigen::VectorXd phi_prime;
    std::size_t T_stop;
    Eigen::VectorXd X_means;
    Eigen::VectorXd X_l2norms;
    double y_mean;
};

SelectorRun runSelector(Eigen::MatrixXd& X, Eigen::VectorXd& y, dn::ScalingMode mode,
                double tFDR = 0.2, int seed = 11, std::size_t K = 10) {
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter ctrl;
    ctrl.K = K;
    ctrl.max_dummy_multiplier = 2;
    ctrl.scaling_mode = mode;

    TRexSelector sel(X_map, y_map, tFDR, ctrl, seed, /*verbose=*/false);
    const auto res = sel.select();
    return {res.selected_var, sel.getPhiPrime(), sel.getTStop(),
            sel.getXMeans(), sel.getXL2Norms(), sel.getYMean()};
}

std::vector<std::size_t> selectedIndices(const Eigen::VectorXi& sv) {
    std::vector<std::size_t> out;
    for (Eigen::Index j = 0; j < sv.size(); ++j) {
        if (sv(j) != 0) { out.push_back(static_cast<std::size_t>(j)); }
    }
    return out;
}

} // namespace

// ========================================================================================
// Normalizer level
// ========================================================================================

TEST(ScalingNone, NormalizerLeavesXAndYUntouchedWithIdentityParams) {
    auto d = syntheticDesign(50, 8);
    unitL2Columns(d.X);
    const Eigen::MatrixXd X0 = d.X;
    const Eigen::VectorXd y0 = d.y;

    dn::NormalizationParams params;
    dn::centerY(d.y, params, dn::ScalingMode::NONE);
    dn::centerAndL2NormalizeX(d.X, params, 1e-6, false, dn::ScalingMode::NONE);

    EXPECT_TRUE(d.X == X0);
    EXPECT_TRUE(d.y == y0);
    EXPECT_EQ(params.mode, dn::ScalingMode::NONE);
    ASSERT_EQ(params.X_means.size(), X0.cols());
    ASSERT_EQ(params.X_l2norms.size(), X0.cols());
    EXPECT_TRUE(params.X_means.isZero(0.0));
    EXPECT_TRUE((params.X_l2norms.array() == 1.0).all());
    EXPECT_EQ(params.y_mean, 0.0);

    // Denormalization is a no-op (bitwise).
    dn::denormalizeX(d.X, params);
    dn::decenterY(d.y, params);
    EXPECT_TRUE(d.X == X0);
    EXPECT_TRUE(d.y == y0);

    // The dummy helper is a no-op under NONE too.
    Eigen::MatrixXd M = X0;
    dn::centerAndL2NormalizeMatrix(M, 1e-6, false, dn::ScalingMode::NONE);
    EXPECT_TRUE(M == X0);
}


TEST(ScalingNone, NormalizerL2AndZscoreStillRoundTrip) {
    for (auto mode : {dn::ScalingMode::L2, dn::ScalingMode::ZSCORE}) {
        auto d = syntheticDesign(60, 12);
        const Eigen::MatrixXd X0 = d.X;
        dn::NormalizationParams params;
        dn::centerAndL2NormalizeX(d.X, params, 1e-6, false, mode);
        EXPECT_EQ(params.mode, mode);
        const double target = (mode == dn::ScalingMode::ZSCORE) ? std::sqrt(59.0) : 1.0;
        for (Eigen::Index j = 0; j < d.X.cols(); ++j) {
            EXPECT_NEAR(d.X.col(j).mean(), 0.0, 1e-12);
            EXPECT_NEAR(d.X.col(j).norm(), target, 1e-10 * target);
        }
        dn::denormalizeX(d.X, params);
        EXPECT_TRUE(d.X.isApprox(X0, 1e-12));
    }
}


// ----------------------------------------------------------------------------
// Rejection (every mode)
// ----------------------------------------------------------------------------

TEST(ScalingNone, NonFiniteEntryInXIsRejectedInEveryModeAndXIsUnmodified) {
    for (auto mode : {dn::ScalingMode::L2, dn::ScalingMode::ZSCORE, dn::ScalingMode::NONE}) {
        auto d = syntheticDesign(40, 8);
        d.X(3, 5) = std::numeric_limits<double>::quiet_NaN();
        const Eigen::MatrixXd X0 = d.X;
        dn::NormalizationParams params;
        try {
            dn::centerAndL2NormalizeX(d.X, params, 1e-6, false, mode);
            FAIL() << "NaN accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("X(3, 5)"), std::string::npos) << e.what();
            EXPECT_NE(std::string(e.what()).find("not finite"), std::string::npos);
        }
        // Bitwise: NaN != NaN, so compare the finite part and the NaN cell.
        for (Eigen::Index j = 0; j < X0.cols(); ++j) {
            for (Eigen::Index i = 0; i < X0.rows(); ++i) {
                if (i == 3 && j == 5) {
                    EXPECT_TRUE(fpc::isnan(d.X(i, j)));
                } else {
                    EXPECT_EQ(d.X(i, j), X0(i, j));
                }
            }
        }

        auto d2 = syntheticDesign(40, 8);
        d2.X(0, 0) = std::numeric_limits<double>::infinity();
        EXPECT_THROW(dn::centerAndL2NormalizeX(d2.X, params, 1e-6, false, mode),
                     std::invalid_argument);
    }
}


TEST(ScalingNone, NonFiniteEntryInYIsRejectedInEveryMode) {
    for (auto mode : {dn::ScalingMode::L2, dn::ScalingMode::ZSCORE, dn::ScalingMode::NONE}) {
        auto d = syntheticDesign(40, 8);
        d.y(2) = std::numeric_limits<double>::infinity();
        dn::NormalizationParams params;
        try {
            dn::centerY(d.y, params, mode);
            FAIL() << "Inf in y accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("y(2)"), std::string::npos) << e.what();
        }
    }
}


TEST(ScalingNone, ConstantColumnIsRejectedUnderL2AndZscoreNamingTheColumn) {
    for (auto mode : {dn::ScalingMode::L2, dn::ScalingMode::ZSCORE}) {
        auto d = syntheticDesign(40, 8);
        d.X.col(4).setConstant(3.5);
        const Eigen::MatrixXd X0 = d.X;
        dn::NormalizationParams params;
        try {
            dn::centerAndL2NormalizeX(d.X, params, 1e-6, false, mode);
            FAIL() << "constant column accepted";
        } catch (const std::invalid_argument& e) {
            const std::string what = e.what();
            EXPECT_NE(what.find("column 4"), std::string::npos) << what;
            EXPECT_NE(what.find("constant"), std::string::npos) << what;
        }
        EXPECT_TRUE(d.X == X0) << "X modified by a rejected call";
    }
}


TEST(ScalingNone, AllZeroColumnIsRejectedUnderNoneButAConstantAtomPasses) {
    auto d = syntheticDesign(40, 8);
    unitL2Columns(d.X);
    d.X.col(4).setZero();
    dn::NormalizationParams params;
    try {
        dn::centerAndL2NormalizeX(d.X, params, 1e-6, false, dn::ScalingMode::NONE);
        FAIL() << "all-zero column accepted under NONE";
    } catch (const std::invalid_argument& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("column 4"), std::string::npos) << what;
        EXPECT_NE(what.find("ScalingMode::NONE"), std::string::npos) << what;
    }

    // A constant, non-zero atom (DC) is a legitimate column under NONE.
    d.X.col(4).setConstant(1.0 / std::sqrt(40.0));
    const Eigen::MatrixXd X0 = d.X;
    EXPECT_NO_THROW(dn::centerAndL2NormalizeX(d.X, params, 1e-6, false, dn::ScalingMode::NONE));
    EXPECT_TRUE(d.X == X0);
}


TEST(ScalingNone, ZscoredDesignUnderNoneWarnsAboutTheUnitL2Band) {
#ifndef CRAN_BUILD
    auto d = syntheticDesign(60, 8);
    dn::NormalizationParams params;
    dn::centerAndL2NormalizeX(d.X, params, 1e-6, false, dn::ScalingMode::ZSCORE);
    // Columns now have norm sqrt(59) ~ 7.7: outside [0.25, 4].
    testing::internal::CaptureStderr();
    dn::NormalizationParams none_params;
    dn::centerAndL2NormalizeX(d.X, none_params, 1e-6, false, dn::ScalingMode::NONE);
    const std::string err = testing::internal::GetCapturedStderr();
    EXPECT_NE(err.find("ScalingMode::NONE"), std::string::npos) << err;
    EXPECT_NE(err.find("8 of 8"), std::string::npos) << err;

    // A unit-L2 design is silent.
    auto u = syntheticDesign(60, 8);
    unitL2Columns(u.X);
    testing::internal::CaptureStderr();
    dn::centerAndL2NormalizeX(u.X, none_params, 1e-6, false, dn::ScalingMode::NONE);
    EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
#else
    SUCCEED();
#endif
}


// ========================================================================================
// Selector level
// ========================================================================================

TEST(ScalingNone, SelectorLeavesXAndYUntouchedAndReportsIdentityParams) {
    auto d = syntheticDesign(80, 20);
    unitL2Columns(d.X);
    const Eigen::MatrixXd X0 = d.X;
    const Eigen::VectorXd y0 = d.y;

    {
        Eigen::Map<Eigen::MatrixXd> X_map(d.X.data(), d.X.rows(), d.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(d.y.data(), d.y.size());
        TRexControlParameter ctrl;
        ctrl.K = 5;
        ctrl.max_dummy_multiplier = 2;
        ctrl.scaling_mode = dn::ScalingMode::NONE;
        TRexSelector sel(X_map, y_map, 0.2, ctrl, 3, false);
        EXPECT_TRUE(d.X == X0) << "X modified at construction under NONE";
        (void)sel.select();
        EXPECT_TRUE(d.X == X0) << "X modified by select() under NONE";
        EXPECT_TRUE(sel.getXMeans().isZero(0.0));
        EXPECT_TRUE((sel.getXL2Norms().array() == 1.0).all());
        EXPECT_EQ(sel.getYMean(), 0.0);
    }
    EXPECT_TRUE(d.X == X0) << "X modified at destruction under NONE";
    EXPECT_TRUE(d.y == y0);
}


TEST(ScalingNone, NoneOnPrenormalizedDesignEqualsL2OnRawDesign) {
    // Raw design under the L2 default (n large enough that the planted
    // support {1, 2, 3} is found, so the equivalence is not vacuous).
    auto raw = syntheticDesign(300, 60);
    const SelectorRun l2 = runSelector(raw.X, raw.y, dn::ScalingMode::L2,
                                       /*tFDR=*/0.2, /*seed=*/11, /*K=*/20);

    // The same design pre-normalized by the user with the library helpers,
    // then run under NONE with the same seed.
    auto pre = syntheticDesign(300, 60);
    dn::NormalizationParams params;
    dn::centerY(pre.y, params);
    dn::centerAndL2NormalizeX(pre.X, params, 1e-6, false, dn::ScalingMode::L2);
    const SelectorRun none = runSelector(pre.X, pre.y, dn::ScalingMode::NONE,
                                         /*tFDR=*/0.2, /*seed=*/11, /*K=*/20);

    EXPECT_EQ(l2.T_stop, none.T_stop);
    EXPECT_TRUE(l2.selected_var == none.selected_var);
    EXPECT_TRUE(l2.phi_prime == none.phi_prime) << "Phi_prime differs bitwise";
    // Non-vacuity: something from the planted support {1, 2, 3} is selected.
    const auto selected = selectedIndices(l2.selected_var);
    EXPECT_FALSE(selected.empty()) << "T_stop " << l2.T_stop;
    for (std::size_t j : selected) { EXPECT_TRUE(j >= 1 && j <= 3) << "false positive " << j; }
}


TEST(ScalingNone, InterceptFreeDictionaryIsRejectedUnderL2AndSelectableUnderNone) {
    // Under the L2 default the DC atom is constant after centering: rejected
    // with the column named, X unmodified.
    {
        auto dict = interceptFreeDictionary();
        const Eigen::MatrixXd X0 = dict.X;
        Eigen::Map<Eigen::MatrixXd> X_map(dict.X.data(), dict.X.rows(), dict.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dict.y.data(), dict.y.size());
        TRexControlParameter ctrl;
        ctrl.K = 10;
        try {
            TRexSelector sel(X_map, y_map, 0.2, ctrl, 11, false);
            FAIL() << "constant DC atom accepted under L2";
        } catch (const std::invalid_argument& e) {
            const std::string what = e.what();
            EXPECT_NE(what.find("column 0"), std::string::npos) << what;
            EXPECT_NE(what.find("ScalingMode::NONE"), std::string::npos) << what;
        }
        EXPECT_TRUE(dict.X == X0);
    }

    // Under NONE the dictionary is used as supplied and the DC atom is a
    // selectable variable: the planted support (DC, cos 3, sin 5) is
    // recovered, with at most one extra atom (tFDR 0.2 permits a false
    // positive; the partial window correlates neighbouring harmonics).
    {
        auto dict = interceptFreeDictionary();
        const Eigen::MatrixXd X0 = dict.X;
        const SelectorRun none = runSelector(dict.X, dict.y, dn::ScalingMode::NONE,
                                     /*tFDR=*/0.2, /*seed=*/11, /*K=*/20);
        EXPECT_TRUE(dict.X == X0);
        const auto selected = selectedIndices(none.selected_var);
        for (std::size_t j : dict.support) {
            EXPECT_NE(none.selected_var(static_cast<Eigen::Index>(j)), 0)
                << "planted atom " << j << " not selected; T_stop " << none.T_stop
                << ", Phi_prime " << none.phi_prime.transpose();
        }
        EXPECT_LE(selected.size(), dict.support.size() + 1)
            << "Phi_prime " << none.phi_prime.transpose();
        // The DC atom is the strongest variable of the run.
        EXPECT_EQ(none.phi_prime(0), none.phi_prime.maxCoeff());
    }
}


TEST(ScalingNone, SelectorRejectsNonFiniteInputInEveryMode) {
    for (auto mode : {dn::ScalingMode::L2, dn::ScalingMode::ZSCORE, dn::ScalingMode::NONE}) {
        auto d = syntheticDesign(40, 8);
        unitL2Columns(d.X);
        d.X(1, 2) = std::numeric_limits<double>::quiet_NaN();
        Eigen::Map<Eigen::MatrixXd> X_map(d.X.data(), d.X.rows(), d.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(d.y.data(), d.y.size());
        TRexControlParameter ctrl;
        ctrl.K = 3;
        ctrl.scaling_mode = mode;
        EXPECT_THROW(TRexSelector(X_map, y_map, 0.2, ctrl, 1, false), std::invalid_argument);

        // A failed constructor must release the buffer claim: a second
        // selector on the same (repaired) buffer constructs.
        d.X(1, 2) = 0.1;
        EXPECT_NO_THROW(TRexSelector(X_map, y_map, 0.2, ctrl, 1, false));
    }
}



// ----------------------------------------------------------------------------
// Derived selectors: their dummies are drawn from the WORKING X (cluster
// Choleskys, dependency estimates), so NONE on the pre-normalized design must
// reproduce L2 on the raw design there as well.
// ----------------------------------------------------------------------------

namespace {

struct DerivedRun {
    Eigen::VectorXi selected_var;
    Eigen::VectorXd phi_prime;
    std::size_t T_stop;
};

DerivedRun runGvs(Eigen::MatrixXd& X, Eigen::VectorXd& y, dn::ScalingMode mode) {
    using trex::trex_selector_methods::ad::trex_gvs::TRexGVSSelector;
    using trex::trex_selector_methods::ad::trex_gvs::TRexGVSControlParameter;
    using trex::trex_selector_methods::ad::trex_gvs::GVSType;
    namespace disp = trex::trex_selector_methods::ad::utils::solver_dispatch;

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexGVSControlParameter ctrl;
    ctrl.gvs_type = GVSType::EN;
    ctrl.lambda_2 = 0.5;                       // fixed: no CV tuner
    ctrl.trex_ctrl.solver_type = disp::SolverTypeForTRex::TENET;
    ctrl.trex_ctrl.K = 5;
    ctrl.trex_ctrl.max_dummy_multiplier = 2;
    ctrl.trex_ctrl.scaling_mode = mode;

    TRexGVSSelector sel(X_map, y_map, 0.2, ctrl, 11, false);
    (void)sel.select();
    return {sel.getSelectedVar(), sel.getPhiPrime(), sel.getTStop()};
}

DerivedRun runDa(Eigen::MatrixXd& X, Eigen::VectorXd& y, dn::ScalingMode mode) {
    using trex::trex_selector_methods::ad::trex_da::TRexDASelector;
    using trex::trex_selector_methods::ad::trex_da::TRexDAControlParameter;
    using trex::trex_selector_methods::ad::trex_da::DAMethod;

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexDAControlParameter ctrl;
    ctrl.method = DAMethod::BT;
    ctrl.trex_ctrl.K = 5;
    ctrl.trex_ctrl.max_dummy_multiplier = 2;
    ctrl.trex_ctrl.scaling_mode = mode;

    TRexDASelector sel(X_map, y_map, 0.2, ctrl, 11, false);
    (void)sel.select();
    return {sel.getSelectedVar(), sel.getPhiPrime(), sel.getTStop()};
}

RawDesign prenormalized(const RawDesign& raw) {
    RawDesign pre = raw;
    dn::NormalizationParams params;
    dn::centerY(pre.y, params);
    dn::centerAndL2NormalizeX(pre.X, params, 1e-6, false, dn::ScalingMode::L2);
    return pre;
}

} // namespace


TEST(ScalingNone, GvsNoneOnPrenormalizedDesignEqualsL2OnRawDesign) {
    auto raw = syntheticDesign(120, 30);
    auto pre = prenormalized(raw);
    const DerivedRun l2   = runGvs(raw.X, raw.y, dn::ScalingMode::L2);
    const DerivedRun none = runGvs(pre.X, pre.y, dn::ScalingMode::NONE);
    EXPECT_EQ(l2.T_stop, none.T_stop);
    EXPECT_TRUE(l2.selected_var == none.selected_var);
    EXPECT_TRUE(l2.phi_prime == none.phi_prime) << "Phi_prime differs bitwise";
}


TEST(ScalingNone, DaNoneOnPrenormalizedDesignEqualsL2OnRawDesign) {
    auto raw = syntheticDesign(120, 30);
    auto pre = prenormalized(raw);
    const DerivedRun l2   = runDa(raw.X, raw.y, dn::ScalingMode::L2);
    const DerivedRun none = runDa(pre.X, pre.y, dn::ScalingMode::NONE);
    EXPECT_EQ(l2.T_stop, none.T_stop);
    EXPECT_TRUE(l2.selected_var == none.selected_var);
    EXPECT_TRUE(l2.phi_prime == none.phi_prime) << "Phi_prime differs bitwise";
}

} // namespace trex::test::trex_selector_methods::scaling_none
