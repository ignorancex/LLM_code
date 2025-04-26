#include <gtest/gtest.h>
#include <cstdint>

#define NN_CONFIG_SET
#define G_n                     128
constexpr uint64_t gam = 10;

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



TEST(Dist_GoldenNearestNeighbor_128_10, Quadratic) {
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	constexpr uint64_t w = 34;
	constexpr uint64_t d = 1;
	constexpr uint64_t r = 1;
	constexpr uint64_t N = 1;
	constexpr uint64_t size =  (1u << 10u);

	NearestNeighbor::create_test_lists_with_distribution(L1, L2, size, gam, w, pos1, pos2);
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

TEST(Dist_GoldenNearestNeighbor_128_10, Windowed) {
	// Laut Script für n = 128 \lambda = 0.078125 da size = 10
	// n: 128 lam: 0.078125 w: 0.271599172304043 r: 1.4285714285714286 N: 238 d: 0.28708274953355223 q: 0.5368665378595762
	// n: 128 size: 10.0 w: 34.764694054917506 r: 1 N: 238 d: 36.746591940294685 q: 0.5368665378595762
	constexpr uint64_t w = 28;
	constexpr uint64_t d = 45;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 238;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 10u);

	run_dist(gam, w , d, r, N, size, find_all, THRESHHOLD, 0, "w:128:0:10");
}

TEST(Dist_GoldenNearestNeighbor_128_10, WindowedWithEpsilon) {
	constexpr uint64_t w = 34;
	constexpr uint64_t d = 55;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 175;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 10u);
	constexpr uint64_t epsilon =  1;

	run_dist(gam, w , d, r, N, size, find_all, THRESHHOLD, epsilon, "w:128:1:10");
}

TEST(Dist_GoldenNearestNeighbor_128_15, Windowed) {
	// Laut Script für n = 128 \lambda = 0.1171875 da size = 15
	// n: 128 lam: 0.1171875 w: 0.2230096042403977 r: 2.142857142857143 N: 454 d: 0.1651517530476932 q: 0.28154086496420877
	// n: 128 size: 15.0 w: 28.545229342770906 r: 2 N: 454 d: 21.13942439010473 q: 0.28154086496420877
	constexpr uint64_t w = 28;
	constexpr uint64_t d = 50;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 1454;
	constexpr uint64_t THRESHHOLD = 20;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 15u);

	run_dist(gam, w , d, r, N, size, find_all, THRESHHOLD, 0, "w:128:0:15");
}

TEST(Dist_GoldenNearestNeighbor_128_15, WindowedWithEpsilon) {
	constexpr uint64_t w = 28;
	constexpr uint64_t d = 45;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 1454;
	constexpr uint64_t THRESHHOLD = 20;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 15u);
	constexpr uint64_t epsilon =  1;

	run_dist(gam, w , d, r, N, size, find_all, THRESHHOLD, epsilon, "w:128:1:15");
}

// 20 s
TEST(Dist_GoldenNearestNeighbor_128_20, Windowed) {
	// Laut Script für n = 128 \lambda = 0.15625 da size = 20
	// n: 128 lam: 0.15625 w: 0.18341914250537633 r: 2.857142857142857 N: 1008 d: 0.11011788229269559 q: 0.1269708942029263
	// n: 128 size: 20.0 w: 23.47765024068817 r: 2 N: 1008 d: 14.095088933465036 q: 0.1269708942029263
	constexpr uint64_t w = 24;
	constexpr uint64_t d = 52;
	constexpr uint64_t r = 4;
	constexpr uint64_t N = 2008;
	constexpr uint64_t THRESHHOLD = 500;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 20u);

	run_dist(gam, w , d, r, N, size, find_all, THRESHHOLD, 0, "w:128:0:20");
}

// 7s
TEST(Dist_GoldenNearestNeighbor_128_20, WindowedWithEpsilon) {
	constexpr uint64_t w = 24;
	constexpr uint64_t d = 40;
	constexpr uint64_t r = 4;
	constexpr uint64_t N = 2008;
	constexpr uint64_t THRESHHOLD = 500;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 20u);
	constexpr uint64_t epsilon =  1;

	run_dist(gam, w , d, r, N, size, find_all, THRESHHOLD, epsilon, "w:128:1:20");
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
