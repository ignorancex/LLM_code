#include <gtest/gtest.h>
#include <cstdint>

#define NN_CONFIG_SET
#define G_n                     256
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



TEST(Dist_GoldenNearestNeighbor_256_10, Quadratic) {
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	constexpr uint64_t w = 24;
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

TEST(Dist_GoldenNearestNeighbor_256_10, Windowed) {
	// Laut Script für n = 256 \lambda = 0.0390625 da size = 10
	// n: 256 lam: 0.0390625 w: 0.3369549996940837 r: 1.25 N: 446 d: 0.3584098841512485 q: 0.5729355014372951
	// n: 256 size: 10.0 w: 86.26047992168543 r: 1 N: 446 d: 91.75293034271962 q: 0.5729355014372951
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	constexpr uint64_t w = 88;
	constexpr uint64_t d = 140;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 1446;
	constexpr uint64_t THRESHHOLD = 10;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 10u);

	run_dist(gam, w , d, r, N, size, find_all, THRESHHOLD, 0, "w:256:0:10");
}

TEST(Dist_GoldenNearestNeighbor_256_10, WindowedWithEpsilon) {
	constexpr uint64_t w = 88;
	constexpr uint64_t d = 140;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 1446;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 10u);
	constexpr uint64_t epsilon =  1;

	run_dist(gam, w , d, r, N, size, find_all, THRESHHOLD, epsilon, "w:256:1:10");
}

TEST(Dist_GoldenNearestNeighbor_256_15, Windowed) {
	// Laut Script für n = 256 \lambda = 0.05859375 da size = 15
	// n: 256 lam: 0.05859375 w: 0.3012491540443123 r: 1.875 N: 826 d: 0.21491245047530094 q: 0.309633111241245
	// n: 256 size: 15.0 w: 77.11978343534395 r: 1 N: 826 d: 55.01758732167704 q: 0.309633111241245
	constexpr uint64_t w = 76;
	constexpr uint64_t d = 116;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 1826;
	constexpr uint64_t THRESHHOLD = 30;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 15u);

	run_dist(gam, w , d, r, N, size, find_all, THRESHHOLD, 0, "w:256:0:15");

}

TEST(Dist_GoldenNearestNeighbor_256_15, WindowedWithEpsilon) {
	constexpr uint64_t w = 76;
	constexpr uint64_t d = 110;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 1826;
	constexpr uint64_t THRESHHOLD = 10;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 15u);
	constexpr uint64_t epsilon =  1;

	run_dist(gam, w , d, r, N, size, find_all, THRESHHOLD, epsilon, "w:256:1:15");
}

TEST(Dist_GoldenNearestNeighbor_256_20, Windowed) {
	// Laut Script für n = 256 \lambda = 0.078125 da size = 20
	// n: 256 lam: 0.078125 w: 0.271599172304043 r: 2.5 N: 2640 d: 0.14929471420827492 q: 0.09694183771265091
	// n: 256 size: 20.0 w: 69.52938810983501 r: 2 N: 2640 d: 38.21944683731838 q: 0.09694183771265091
	constexpr uint64_t w = 70;
	constexpr uint64_t d = 105;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 3640;
	constexpr uint64_t THRESHHOLD = 500;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 20u);

	run_dist(gam, w , d, r, N, size, find_all, THRESHHOLD, 0, "w:256:0:20");
}

TEST(Dist_GoldenNearestNeighbor_256_20, WindowedWithEpsilon) {
	constexpr uint64_t w = 70;
	constexpr uint64_t d = 98;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 3640;
	constexpr uint64_t THRESHHOLD = 200;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 20u);
	constexpr uint64_t epsilon =  1;

	run_dist(gam, w , d, r, N, size, find_all, THRESHHOLD, epsilon, "w:256:1:20");
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
