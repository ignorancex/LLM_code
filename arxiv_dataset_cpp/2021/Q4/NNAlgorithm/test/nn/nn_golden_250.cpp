#include <gtest/gtest.h>
#include <cstdint>

#define NN_CONFIG_SET
#define G_n                     250

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



TEST(GoldenNearestNeighbor_250, Quadratic) {
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	constexpr uint64_t w = 83;
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

TEST(GoldenNearestNeighbor_250, Windowed) {
	// Laut Script für n = 40 \lambda = 0.25 da size = 10
	// n: 250 lam: 0.04 w: 0.33504702397337227 r: 1.255369169267456 N: 437 d: 0.35602762353826173 q: 0.5719631942622386
	// n: 250 size: 10.0 w: 83.76175599334307 r: 1 N: 437 d: 89.00690588456543 q: 0.5719631942622386
	constexpr uint64_t w = 84;
	constexpr uint64_t d = 89;
	constexpr uint64_t r = 6;
	constexpr uint64_t N = 437;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 10u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:250:0:10");

}

TEST(GoldenNearestNeighbor_250, WindowedWithEpsilon) {
	constexpr uint64_t w = 84;
	constexpr uint64_t d = 89;
	constexpr uint64_t r = 6;
	constexpr uint64_t N = 437;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 10u);
	constexpr uint64_t epsilon =  1;

	run(w, d, r, N, size, find_all, THRESHHOLD, epsilon, "w:250:0:10");
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
