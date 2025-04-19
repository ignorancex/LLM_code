#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>
#include <random>

#define G_n 128

#include "container.h"
#include "helper.h"


using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using BinaryContainerTest = BinaryContainer<G_n>;

TEST(Cmp, Simple_Everything_False) {
	BinaryContainerTest b1;
	BinaryContainerTest b2;
	b1.zero(); b2.one();
	EXPECT_EQ(false, BinaryContainerTest::cmp(b1, b2));

}

TEST(Cmp, Simple_Everything_True) {
	BinaryContainerTest b1;
	BinaryContainerTest b2;

	b1.zero(); b2.zero();

	EXPECT_EQ(true, BinaryContainerTest::cmp(b1, b2));
}

TEST(Cmp, Complex_One) {
	BinaryContainerTest b1;
	BinaryContainerTest b2;

	b1.zero();

	for (int k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (int k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			b2.zero();
			for (int i = k_lower; i < k_higher; ++i) {
				b2[i] = true;
			}

			EXPECT_EQ(false, BinaryContainerTest::cmp(b1, b2));
		}
	}
}

TEST(Cmp, Complex_Zero) {
	BinaryContainerTest b1;
	BinaryContainerTest b2;

	b1.zero();
	b2.zero();

	for (int i = 0; i < b1.size(); ++i) { b1[i] = true; }

	for (int k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (int k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			b2.zero();
			for (int i = k_lower; i < k_higher; ++i) {
				b2[i] = true;
			}

			EXPECT_EQ(false, BinaryContainerTest::cmp(b1, b2));



			b2.zero();
			EXPECT_EQ(false, BinaryContainerTest::cmp(b1, b2));
		}
	}
}


TEST(Special_is_equal, Z_is_zero) {
	BinaryContainerTest b1;
	BinaryContainerTest b2;
	BinaryContainerTest z;
	z.zero();
	b1.zero(); b2.one();
	EXPECT_EQ(true, b1.is_equal(b2, z));
	z.one();
	b1.one(); b2.one();
	EXPECT_EQ(true, b1.is_equal(b2, z));

	z.one();
	b1.zero(); b2.zero();
	EXPECT_EQ(true, b1.is_equal(b2, z));
}

TEST(Special_is_equal, Z_is_small) {
	BinaryContainerTest b1;
	BinaryContainerTest b2;
	BinaryContainerTest z;
	z.zero();
	b1.zero(); b2.one();
	z[0] = true;

	EXPECT_EQ(false, b1.is_equal(b2, z));

	b1.zero(); b2.zero();
	b2[0] = true;
	EXPECT_EQ(false, b1.is_equal(b2, z));

	z.zero();
	b1.zero(); b2.zero();
	EXPECT_EQ(true, b1.is_equal(b2, z));

}



TEST(Special_is_lower, Z_is_zero) {
	BinaryContainerTest b1;
	BinaryContainerTest b2;
	BinaryContainerTest z;
	z.zero();
	b1.zero(); b2.one();
	EXPECT_EQ(false, b1.is_lower(b2, z));
	z.one();
	b1.one(); b2.one();
	EXPECT_EQ(false, b1.is_lower(b2, z));

	z.one();
	b1.zero(); b2.zero();
	EXPECT_EQ(false, b1.is_lower(b2, z));
}

TEST(Special_is_lower, Z_is_small) {
	BinaryContainerTest b1;
	BinaryContainerTest b2;
	BinaryContainerTest z;
	z.zero();
	b1.zero(); b2.one();
	z[0] = true;

	EXPECT_EQ(true, b1.is_lower(b2, z));

	b1.zero(); b2.zero();
	b2[0] = true;
	EXPECT_EQ(true, b1.is_lower(b2, z));

	b1.zero(); b2.zero();
	b2[1] = true;
	EXPECT_EQ(false, b1.is_lower(b2, z));

	z.zero();
	b1.zero(); b2.zero();
	EXPECT_EQ(false, b1.is_lower(b2, z));

	for (int i = 0; i < G_n; i+=64) {
		z.zero();
		b1.zero(); b2.one();
		z[i] = true;
		EXPECT_EQ(true, b1.is_lower(b2, z));

		b1.zero(); b2.zero();
		b2[i] = true;
		EXPECT_EQ(true, b1.is_lower(b2, z));

		b1.zero(); b2.zero();
		b2[i+1] = true;
		EXPECT_EQ(false, b1.is_lower(b2, z));


		z.zero();
		b1.zero(); b2.zero();
		EXPECT_EQ(false, b1.is_lower(b2, z));

	}
}


int main(int argc, char **argv) {
	srand(0);
	InitGoogleTest(&argc, argv);

	return RUN_ALL_TESTS();
}