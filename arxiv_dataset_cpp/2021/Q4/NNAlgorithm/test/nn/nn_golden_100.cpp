#include <gtest/gtest.h>
#include <cstdint>

#define NN_CONFIG_SET
#define G_n                     100

#include "bench.h"
#include "helper.h"
#include "container.h"
#include "list.h"
#include "windowed_nn_v2.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;



TEST(GoldenNearestNeighbor_100_10, Quadratic) {
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	constexpr uint64_t w = 24;
	constexpr uint64_t d = 1;
	constexpr uint64_t r = 1;
	constexpr uint64_t N = 1;
	constexpr uint64_t size =  (1u << 10u);

	NearestNeighbor::create_test_lists(L1, L2, size, w, pos1, pos2);
	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];

	NearestNeighbor nn{L1, L2, w, r, N, d};
	uint64_t found = nn.NN(gold1, gold2);
	EXPECT_EQ(gold1.is_equal(nn.sols_1[found]), true);
	EXPECT_EQ(gold2.is_equal(nn.sols_2[found]), true);
	EXPECT_EQ(nn.print_result(gold1, gold2), true);
	EXPECT_EQ(L1.size(), size);
	EXPECT_EQ(L2.size(), size);
}

TEST(GoldenNearestNeighbor_100_10, Windowed) {
	// Laut Script für n = 100 \lambda = 0.1 da size = 10
	// n: 100 lam: 0.1 w: 0.24300385380895362 r: 1 N: 132 d: 0.5 q: 0.7569961461910464
	// n: 100 size: 10.0 w: 24.300385380895364 r: 1 N: 132 d: 50.0 q: 0.7569961461910464
	constexpr uint64_t w = 24;
	constexpr uint64_t d = 40;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 132;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 10u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:100:0:10");
}

TEST(GoldenNearestNeighbor_100_10, WindowedWithEpsilon) {
	constexpr uint64_t w = 24;
	constexpr uint64_t d = 40;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 100;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 10u);
	constexpr uint64_t epsilon =  1;

	run(w, d, r, N, size, find_all, THRESHHOLD, epsilon, "w:100:1:10");

}

TEST(GoldenNearestNeighbor_100_15, Windowed) {
	// Laut Script für n = 100 \lambda = 0.15 da size = 15
	// n: 100 lam: 0.15 w: 0.18929770537062493 r: 2.257724967479859 N: 358 d: 0.1468235237870819 q: 0.2792906220053647
	//n: 100 size: 15.0 w: 18.929770537062492 r: 2 N: 358 d: 14.682352378708192 q: 0.2792906220053647
	constexpr uint64_t w = 18;
	constexpr uint64_t d = 15;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 358;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size = (1u << 15u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:100:0:15");
}

TEST(GoldenNearestNeighbor_100_15, WindowedWithEpsilon) {
	constexpr uint64_t w = 24;
	constexpr uint64_t d = 40;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 358;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 15u);
	constexpr uint64_t epsilon =  1;

	run(w, d, r, N, size, find_all, THRESHHOLD, epsilon, "w:100:1:15");
}


TEST(GoldenNearestNeighbor_100_20, Windowed) {
	// Laut Script für n = 100 \lambda = 0.2 da size = 20
	// n: 100 lam: 0.2 w: 0.14610240341188663 r: 3.010299956639812 N: 661 d: 0.0957516856574359 q: 0.15110836419740273
	// n: 100 size: 20.0 w: 14.610240341188662 r: 3 N: 661 d: 9.57516856574359 q: 0.15110836419740273
	constexpr uint64_t w = 15;
	constexpr uint64_t d = 28;
	constexpr uint64_t r = 3;
	constexpr uint64_t N = 661;
	constexpr uint64_t THRESHHOLD = 1000;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 20u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:100:0:20");
}

TEST(GoldenNearestNeighbor_100_20, WindowedWithEpsilon) {
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	constexpr uint64_t w = 15;
	constexpr uint64_t d = 12;
	constexpr uint64_t r = 3;
	constexpr uint64_t N = 661;
	constexpr uint64_t THRESHHOLD = 500;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 20u);
	constexpr uint64_t epsilon =  1;

	run(w, d, r, N, size, find_all, THRESHHOLD, epsilon, "w:100:1:20");
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
