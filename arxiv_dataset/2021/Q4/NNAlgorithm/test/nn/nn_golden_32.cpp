#include <gtest/gtest.h>
#include <cstdint>

#define NN_CONFIG_SET
#define G_n                     32

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



TEST(GoldenNearestNeighbor_32_10, Quadratic) {
	constexpr uint64_t w = 8;
	constexpr uint64_t size = (1u<<10);

	run_quadratic(w, size, "q:32:10");
}

TEST(GoldenNearestNeighbor_32_10, Windowed) {
	constexpr uint64_t w = 8;
	constexpr uint64_t d = 16;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 100;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size = (1u << 10u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:32:0:10");
}

TEST(GoldenNearestNeighbor_32_10, WindowedWithEpsilon) {
	constexpr uint64_t w = 8;
	constexpr uint64_t d = 16;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 100;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 10u);
	constexpr uint64_t epsilon =  1;

	run(w, d, r, N, size, find_all, THRESHHOLD, epsilon, "w:32:1:10");
}

TEST(GoldenNearestNeighbor_32_15, Quadratic) {
	constexpr uint64_t w = 0;
	constexpr uint64_t size = (1u<<15);

	run_quadratic(w, size, "q:32:15");
}

TEST(GoldenNearestNeighbor_32_15, Windowed) {
	constexpr uint64_t w = 6;
	constexpr uint64_t d = 12;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 222;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 15u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:32:0:15");

}

TEST(GoldenNearestNeighbor_32_15, WindowedWithEpsilon) {
	constexpr uint64_t w = 6;
	constexpr uint64_t d = 12;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 222;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 15u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 1, "w:32:1:15");
}

TEST(GoldenNearestNeighbor_32_20, Quadratic) {
	constexpr uint64_t w = 4;
	constexpr uint64_t size = (1u<<20);

	run_quadratic(w, size, "q:32:20");
}

TEST(GoldenNearestNeighbor_32_20, Windowed) {
	constexpr uint64_t w = 4;
	constexpr uint64_t d = 8;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 20; // 100
	constexpr uint64_t THRESHHOLD = 10;
	constexpr bool find_all = true; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 20u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:64:0:20");
}

TEST(GoldenNearestNeighbor_32_20, WindowedWithEpsilon) {
	constexpr uint64_t w = 4;
	constexpr uint64_t d = 6;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 20;
	constexpr uint64_t THRESHHOLD = 10;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 20u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 1, "w:32:1:20");
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
