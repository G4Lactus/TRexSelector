// ===================================================================================
/**
 * @file test_utils_serialization.cpp
 *
 * @brief Unit tests for serialization utilities in utils_cereal_eigen.hpp
 */
// ===================================================================================

// google test includes
#include <gtest/gtest.h>

// std includes
#include <sstream>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Sparse>

// project utils includes
#include <utils/serialization/utils_cereal_eigen.hpp>

// ===================================================================================

/** @brief Test serialization and deserialization of Eigen dense matrices. */
TEST(SerializationTest, DenseMatrix) {
    // Create an Eigen matrix and populate it
    Eigen::MatrixXd original(2, 2);
    original << 1.5, 2.5,
                3.5, 4.5;

    // Serialize to string stream
    std::stringstream ss;
    {
        cereal::PortableBinaryOutputArchive oarchive(ss);
        oarchive(original);
    }

    // Deserialize to a new instance
    Eigen::MatrixXd loaded;
    {
        cereal::PortableBinaryInputArchive iarchive(ss);
        iarchive(loaded);
    }

    // Check dimensions and values
    ASSERT_EQ(loaded.rows(), 2);
    ASSERT_EQ(loaded.cols(), 2);
    EXPECT_TRUE(original.isApprox(loaded));
}


/** @brief Test serialization and deserialization of Eigen sparse matrices. */
TEST(SerializationTest, SparseMatrix) {
    // Create an Eigen sparse matrix
    Eigen::SparseMatrix<double> original(3, 3);
    original.insert(0, 0) = 1.0;
    original.insert(1, 2) = -2.0;
    original.insert(2, 1) = 3.14;
    original.makeCompressed();

    // Serialize to string stream
    std::stringstream ss;
    {
        cereal::PortableBinaryOutputArchive oarchive(ss);
        oarchive(original);
    }

    // Deserialize to a new instance
    Eigen::SparseMatrix<double> loaded;
    {
        cereal::PortableBinaryInputArchive iarchive(ss);
        iarchive(loaded);
    }

    // Check dimensions and values
    ASSERT_EQ(loaded.rows(), 3);
    ASSERT_EQ(loaded.cols(), 3);
    ASSERT_EQ(loaded.nonZeros(), 3);
    EXPECT_TRUE(original.isApprox(loaded));
}
