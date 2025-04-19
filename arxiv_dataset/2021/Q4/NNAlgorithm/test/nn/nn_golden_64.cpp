#include <gtest/gtest.h>
#include <cstdint>

#define NN_CONFIG_SET
#define G_n                     64

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



TEST(GoldenNearestNeighbor_64_10, Quadratic) {
	constexpr uint64_t w = 12;
	constexpr uint64_t size = (1u<<10);

	run_quadratic(w, size, "q:64:10");
}

TEST(GoldenNearestNeighbor_64_10, Windowed) {
	// Laut Script für n = 64 \lambda = 0.15625 da size = 10
	// n: 64 lam: 0.15625 w: 0.18341914250537633 r: 1 N: 78 d: 0.5 q: 0.8165808574946236
	// n: 64 size: 10.0 w: 11.738825120344085 r: 1 N: 78 d: 32.0 q: 0.8165808574946236

	constexpr uint64_t w = 12;
	constexpr uint64_t d = 32;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 100;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size = (1u << 10u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:64:0:10");
}

TEST(GoldenNearestNeighbor_64_10, WindowedWithEpsilon) {
	constexpr uint64_t w = 12;
	constexpr uint64_t d = 32;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 100;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 10u);
	constexpr uint64_t epsilon =  1;

	run(w, d, r, N, size, find_all, THRESHHOLD, epsilon, "w:64:1:10");
}

TEST(GoldenNearestNeighbor_64_10, IndykMotwani) {
	constexpr uint64_t w = 12;
	constexpr uint64_t l = 10u;

	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << l);

	run_indyk(w, l, find_all, size,  "i:64:10");
}

TEST(GoldenNearestNeighbor_64_15, Quadratic) {
	constexpr uint64_t w = 8;
	constexpr uint64_t size = (1u<<15);

	run_quadratic(w, size, "q:64:15");
}

TEST(GoldenNearestNeighbor_64_15, Windowed) {
	// Laut Script für n = 64 \lambda = 0.234375 da size = 10
	// n: 64 lam: 0.234375 w: 0.12065823972424823 r: 2.5 N: 222 d: 0.11316082423714649 q: 0.2882185503700659
	// n: 64 size: 15.0 w: 7.722127342351887 r: 2 N: 222 d: 7.242292751177375 q: 0.2882185503700659
	constexpr uint64_t w = 8;
	constexpr uint64_t d = 24;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 222;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 15u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:64:0:15");

}

TEST(GoldenNearestNeighbor_64_15, WindowedWithEpsilon) {
	constexpr uint64_t w = 8;
	constexpr uint64_t d = 20;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 222;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 15u);
	constexpr uint64_t epsilon =  1;

	run(w, d, r, N, size, find_all, THRESHHOLD, 1, "w:64:1:15");
}

TEST(GoldenNearestNeighbor_64_15, IndykMotwani) {
	constexpr uint64_t w = 8;
	constexpr uint64_t l = 15u;

	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << l);

	run_indyk(w, l, find_all, size,  "i:64:15");
}

TEST(GoldenNearestNeighbor_64_20, Quadratic) {
	constexpr uint64_t w = 4;
	constexpr uint64_t size = (1u<<20);

	run_quadratic(w, size, "q:64:20");
}

// 12s
TEST(GoldenNearestNeighbor_64_20, Windowed) {
	// Laut Script für n = 64 \lambda = 0.3125 da size = 20
	// n: 64 lam: 0.3125 w: 0.0724497922261487 r: 3.3333333333333335 N: 308 d: 0.06955979912988478 q: 0.2071689039171905
	// n: 64 size: 20.0 w: 4.6367867024735165 r: 3 N: 308 d: 4.451827144312626 q: 0.2071689039171905
	constexpr uint64_t w = 4;
	constexpr uint64_t d = 20;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 20; // 100
	constexpr uint64_t THRESHHOLD = 10;
	constexpr bool find_all = true; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 20u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:64:0:20");
}

//2,3s
TEST(GoldenNearestNeighbor_64_20, WindowedWithEpsilon) {
	constexpr uint64_t w = 4;
	constexpr uint64_t d = 18;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 20;
	constexpr uint64_t THRESHHOLD = 10;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 20u);
	constexpr uint64_t epsilon =  1;

	run(w, d, r, N, size, find_all, THRESHHOLD, epsilon, "w:64:1:20");
}

TEST(GoldenNearestNeighbor_64_20, IndykMotwani) {
	constexpr uint64_t w = 4;
	constexpr uint64_t l = 20u;

	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << l);

	run_indyk(w, l, find_all, size,  "i:64:20");
}

// Multisolution test
// 4min oder so
TEST(GoldenNearestNeighbor_64_20, Windowed_8) {
	constexpr uint64_t w = 8;
	constexpr uint64_t d = 18;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 20; // 100
	constexpr uint64_t THRESHHOLD = 50;
	constexpr bool find_all = true; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 20u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:64:0:20");
}

// 48 min

TEST(GoldenNearestNeighbor_64_20, WindowedWithEpsilon_8) {
	constexpr uint64_t w = 8;
	constexpr uint64_t d = 14;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 20;
	constexpr uint64_t THRESHHOLD = 10;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 20u);
	constexpr uint64_t epsilon =  1;

	run(w, d, r, N, size, find_all, THRESHHOLD, epsilon, "w:64:1:20");
}

TEST(GoldenNearestNeighbor_64_20, IndykMotwani_8) {
	constexpr uint64_t w = 8;
	constexpr uint64_t l = 20u;

	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << l);

	run_indyk(w, l, find_all, size,  "i:64:20");
}

// Just to get a reference time for our implementation.
TEST(GoldenNearestNeighbor_64_17, Quadratic) {
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	constexpr uint64_t w = 4;
	constexpr uint64_t d = 1;
	constexpr uint64_t r = 1;
	constexpr uint64_t N = 1;
	constexpr uint64_t size = (1u<<17);

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

// NOT WORKING
TEST(GoldenNearestNeighbor_64_25, Windowed) {
	constexpr uint64_t w = 4;
	constexpr uint64_t d = 16;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 20; // 100
	constexpr uint64_t THRESHHOLD = 200;
	constexpr bool find_all = true; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 25u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:64:0:25");
}


TEST(GoldenNearestNeighbor_64_25, IndykMotwani) {
	constexpr uint64_t w = 4;
	constexpr uint64_t l = 25u;

	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << l);

	run_indyk(w, l, find_all, size,  "i:64:25");
}

/* Not enough mem
TEST(GoldenNearestNeighbor_64_30, IndykMotwani) {
	constexpr uint64_t w = 4;
	constexpr uint64_t l = 30u;

	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << l);

	run_indyk(w, l, find_all, size,  "i:64:30");
}
*/

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
