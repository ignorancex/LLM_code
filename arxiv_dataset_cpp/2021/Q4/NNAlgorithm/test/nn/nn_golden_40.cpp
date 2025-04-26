#include <gtest/gtest.h>
#include <cstdint>

#define NN_CONFIG_SET
#define G_n                     40
#define TEST_BASE_LIST_SIZE (1u << 10u)

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



TEST(GoldenNearestNeighbor_40, Quadratic) {
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	constexpr uint64_t w = 4;
	constexpr uint64_t d = 10;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 150;
	constexpr uint64_t size =  (1u << 10u);

	NearestNeighbor::create_test_lists(L1, L2, size, w, pos1, pos2);
	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];

	NearestNeighbor nn{L1, L2, w, r, N, d};
	uint64_t found = nn.NN(gold1, gold2);
	EXPECT_EQ(gold1.is_equal(nn.sols_1[found]), true);
	EXPECT_EQ(gold2.is_equal(nn.sols_2[found]), true);
	EXPECT_EQ(nn.print_result(gold1, gold2), true);
}

TEST(GoldenNearestNeighbor_40_10, Windowed) {
	// Laut Script für n = 40 \lambda = 0.25 da size = 10
	// n: 40 lam: 0.25 w: 4.4 r: 1 N: 44 d: 20.0 q: 0.89
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	constexpr uint64_t w = 4;
	constexpr uint64_t d = 10;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 150;
	constexpr uint64_t THRESHHOLD = 10;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 10u);

	NearestNeighbor::create_test_lists(L1, L2, size, w, pos1, pos2, true, r);
	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];

	WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d, find_all, THRESHHOLD, 1};
	uint64_t found = nn.NN(gold1, gold2);
	EXPECT_EQ(gold1.is_equal(nn.sols_1[found]), true);
	EXPECT_EQ(gold2.is_equal(nn.sols_2[found]), true);
	EXPECT_EQ(nn.print_result(gold1, gold2), true);
	EXPECT_EQ(L1.size(), size);
	EXPECT_EQ(L2.size(), size);
}

TEST(GoldenNearestNeighbor_40_10, WindowedWithEpsilon) {
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	constexpr uint64_t w = 2;
	constexpr uint64_t d = 10;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 150;
	constexpr uint64_t THRESHHOLD = 10;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 10u);
	constexpr uint64_t epsilon = 1;

	NearestNeighbor::create_test_lists(L1, L2, size, w, pos1, pos2, true, r);
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

/*  NOT WORKING
TEST(GoldenNearestNeighbor_40_15, Windowed) {
	// Laut Script für n = 40 \lambda = 0.375 da size = 15
	// n: 40 lam: 0.375 w: 0.0416926902736565 r: 2.8185273706366134 N: 128 d: 0.07764425053151044 q: 0.3110054298475601
	// n: 40 size: 15.0 w: 1.6677076109462599 r: 2 N: 128 d: 3.105770021260418 q: 0.3110054298475601
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	constexpr uint64_t w = 2;
	constexpr uint64_t d = 4;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 256;
	constexpr uint64_t THRESHHOLD = 10;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 15u);

	NearestNeighbor::create_test_lists(L1, L2, size, w, pos1, pos2, true, r);
	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];

	WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d, find_all, THRESHHOLD, 1};
	uint64_t found = nn.NN(gold1, gold2);
	EXPECT_EQ(gold1.is_equal(nn.sols_1[found]), true);
	EXPECT_EQ(gold2.is_equal(nn.sols_2[found]), true);
	EXPECT_EQ(nn.print_result(gold1, gold2), true);
	EXPECT_EQ(L1.get_size(), size);
	EXPECT_EQ(L2.get_size(), size);
}

TEST(GoldenNearestNeighbor_40_15, WindowedWithEpsilon) {
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	constexpr uint64_t w = 1;
	constexpr uint64_t d = 10;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 128;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 15u);
	constexpr uint64_t epsilon = 1;

	NearestNeighbor::create_test_lists(L1, L2, size, w, pos1, pos2, true, r);
	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];

	WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d, find_all, THRESHHOLD, epsilon};
	uint64_t found = nn.NN(gold1, gold2);
	EXPECT_EQ(gold1.is_equal(nn.sols_1[found]), true);
	EXPECT_EQ(gold2.is_equal(nn.sols_2[found]), true);
	EXPECT_EQ(nn.print_result(gold1, gold2), true);
	EXPECT_EQ(L1.get_size(), size);
	EXPECT_EQ(L2.get_size(), size);
}
*/

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
