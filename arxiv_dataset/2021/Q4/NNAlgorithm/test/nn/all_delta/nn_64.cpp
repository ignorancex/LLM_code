#include <gtest/gtest.h>
#include <cstdint>

#define NN_CONFIG_SET
#define G_n                     64
#define ALL_DELTA

#include "bench.h"
#include "list.h"
#include "windowed_nn_v2.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(GoldenNearestNeighbor_64_10, Windowed_all_delta) {
	// Laut Script für n = 64 \lambda = 0.15625 da size = 10
	// n: 64 lam: 0.15625 w: 0.18341914250537633 r: 1 N: 78 d: 0.5 q: 0.8165808574946236
	// n: 64 size: 10.0 w: 11.738825120344085 r: 1 N: 78 d: 32.0 q: 0.8165808574946236
	constexpr uint64_t w = 12;
	constexpr uint64_t d = 22;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 100;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size = (1u << 10u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:64:0:10");
}

TEST(GoldenNearestNeighbor_64_15, Windowed_all_delt) {
	// Laut Script für n = 64 \lambda = 0.234375 da size = 10
	// n: 64 lam: 0.234375 w: 0.12065823972424823 r: 2.5 N: 222 d: 0.11316082423714649 q: 0.2882185503700659
	// n: 64 size: 15.0 w: 7.722127342351887 r: 2 N: 222 d: 7.242292751177375 q: 0.2882185503700659
	constexpr uint64_t w = 8;
	constexpr uint64_t d = 22;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 100;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 15u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:64:0:15");

}


// 12s
TEST(GoldenNearestNeighbor_64_20, Windowed_all_delt) {
	// Laut Script für n = 64 \lambda = 0.3125 da size = 20
	// n: 64 lam: 0.3125 w: 0.0724497922261487 r: 3.3333333333333335 N: 308 d: 0.06955979912988478 q: 0.2071689039171905
	// n: 64 size: 20.0 w: 4.6367867024735165 r: 3 N: 308 d: 4.451827144312626 q: 0.2071689039171905
	constexpr uint64_t w = 4;
	constexpr uint64_t d = 18;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 20; // 100
	constexpr uint64_t THRESHHOLD = 10;
	constexpr bool find_all = true; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 20u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:64:0:20");
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
