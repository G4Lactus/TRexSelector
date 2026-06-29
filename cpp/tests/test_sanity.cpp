#include <gtest/gtest.h>

// Sanity check - project independent
TEST(SanityCheck, BasicMath) {
    EXPECT_EQ(2 + 2, 4);
    EXPECT_TRUE(1 < 2);
}

// Add more complex tests below, including project headers when ready
