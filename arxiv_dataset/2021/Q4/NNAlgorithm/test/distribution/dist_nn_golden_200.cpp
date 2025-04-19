#include <gtest/gtest.h>
#include <cstdint>

#define NN_CONFIG_SET
#define G_n                     200
constexpr uint64_t gam = 10;

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



TEST(Dist_GoldenNearestNeighbor_200, Quadratic) {
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	constexpr uint64_t w = 63;
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

TEST(Dist_GoldenNearestNeighbor_200, Windowed) {
	// Laut Script für n = 40 \lambda = 0.25 da size = 10
	// n: 200 lam: 0.05 w: 0.31601934632360656 r: 1.3082402064781278 N: 356 d: 0.33341709641198103 q: 0.5617829249472905
	// n: 200 size: 10.0 w: 63.20386926472131 r: 1 N: 356 d: 66.6834192823962 q: 0.5617829249472905
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	constexpr uint64_t w = 64;
	constexpr uint64_t d = 90;
	constexpr uint64_t r = 4;
	constexpr uint64_t N = 356;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false;
	constexpr uint64_t size =  (1u << 10u);

	NearestNeighbor::create_test_lists_with_distribution(L1, L2, size, gam, w, pos1, pos2, true, r);
	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];

	WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d, find_all, THRESHHOLD, 0};
	uint64_t found = nn.NN(gold1, gold2);
	EXPECT_EQ(gold1.is_equal(nn.sols_1[found]), true);
	EXPECT_EQ(gold2.is_equal(nn.sols_2[found]), true);
	EXPECT_EQ(nn.print_result(gold1, gold2), true);
	EXPECT_EQ(L1.size(), size);
	EXPECT_EQ(L2.size(), size);
}

TEST(Dist_GoldenNearestNeighbor_200, WindowedWithEpsilon) {
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	constexpr uint64_t w = 64;
	constexpr uint64_t d = 90;
	constexpr uint64_t r = 4;
	constexpr uint64_t N = 356;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false;
	constexpr uint64_t size =  (1u << 10u);
	constexpr uint64_t epsilon =  1;


	NearestNeighbor::create_test_lists_with_distribution(L1, L2, size, gam, w, pos1, pos2, true, r);
	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];

	WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d, find_all, THRESHHOLD, epsilon};
	uint64_t found = nn.NN(gold1, gold2);
	EXPECT_EQ(gold1.is_equal(nn.sols_1[found]), true);
	EXPECT_EQ(gold2.is_equal(nn.sols_2[found]), true);
	EXPECT_EQ(nn.print_result(gold1, gold2), true);
	EXPECT_EQ(L1.size(), size);
	EXPECT_EQ(L2.size(), size);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
