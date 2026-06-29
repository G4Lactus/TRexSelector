// ========================================================================================
/**
 * @file test_utils_eval_metrics.cpp
 *
 * @brief Unit tests for evaluation metrics in utils_eval_metrics.hpp
 */
// ========================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <vector>

// project utils includes
#include <utils/eval_metrics/utils_eval_counts.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>
#include <utils/eval_metrics/utils_eval_composites.hpp>

// ========================================================================================

// Embed into test namespace
namespace trex::test::utils::eval {

// Namespace aliases for convenience
namespace counts = trex::utils::eval::counts;
namespace rates = trex::utils::eval::rates;
namespace composites = trex::utils::eval::composites;

// ========================================================================================

/** @brief Test for true positives count. */
TEST(EvalMetricsTest, TruePositivesCount) {
    std::vector<std::size_t> truth = {1, 3, 5};
    std::vector<std::size_t> pred = {1, 2, 5};

    EXPECT_EQ(counts::true_positives_count(pred, truth), 2);
    EXPECT_EQ(counts::true_positives_count({}, truth), 0);
    EXPECT_EQ(counts::true_positives_count(pred, {}), 0);
}


/** @brief Test for false positives count. */
TEST(EvalMetricsTest, FalsePositivesCount) {
    std::vector<std::size_t> truth = {1, 3, 5};
    std::vector<std::size_t> pred = {1, 2, 5};

    EXPECT_EQ(counts::false_positives_count(pred, truth), 1);
    EXPECT_EQ(counts::false_positives_count({}, truth), 0);
    EXPECT_EQ(counts::false_positives_count(pred, {}), 3);
}


/** @brief Test for false negatives count. */
TEST(EvalMetricsTest, ComputeFDP) {
    std::vector<std::size_t> truth = {1, 3, 5};
    std::vector<std::size_t> pred = {1, 2, 5};

    // FDP = FP / max(|selected|, 1) => 1 / 3
    EXPECT_DOUBLE_EQ(rates::compute_fdp(pred,
                                                   truth),
                                                   1.0 / 3.0);

    // Empty selections should have 0 FDP
    EXPECT_DOUBLE_EQ(rates::compute_fdp({},
                                                   truth), 0.0);
}


/** @brief Test for true positive proportion (TPP). */
TEST(EvalMetricsTest, ComputeTPP) {
    std::vector<std::size_t> truth = {1, 3, 5};
    std::vector<std::size_t> pred = {1, 2, 5};

    // TPP = TP / max(|truth|, 1) => 2 / 3
    EXPECT_DOUBLE_EQ(rates::compute_tpp(pred, truth),
                     2.0 / 3.0);

    // Empty prediction gives 0 TPP
    EXPECT_DOUBLE_EQ(rates::compute_tpp({}, truth),
                     0.0);
}


/** @brief Test for F1 score. */
TEST(EvalMetricsTest, F1Score) {
    // Formula: 2 * (precision * recall) / (precision + recall)
    EXPECT_DOUBLE_EQ(composites::f1_score(1.0, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(composites::f1_score(0.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(composites::f1_score(1.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(composites::f1_score(0.5, 0.5), 0.5);
}

// ========================================================================================
} /* End ofnamespace trex::test::utils::eval */
