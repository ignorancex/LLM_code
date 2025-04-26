#include <gtest/gtest.h>
#include <cstdint>

#define NN_CONFIG_SET
#define G_n                     128

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



TEST(GoldenNearestNeighbor_128_10, Quadratic) {
	constexpr uint64_t w = 34;
	constexpr uint64_t size =  (1u << 10u);

	run_quadratic(w, size, "q:128:10");
}

TEST(GoldenNearestNeighbor_128_10, Windowed) {
	// Laut Script für n = 128 \lambda = 0.078125 da size = 10
	// n: 128 lam: 0.078125 w: 0.271599172304043 r: 1.4285714285714286 N: 238 d: 0.28708274953355223 q: 0.5368665378595762
	// n: 128 size: 10.0 w: 34.764694054917506 r: 1 N: 238 d: 36.746591940294685 q: 0.5368665378595762

	constexpr uint64_t w = 32;
	constexpr uint64_t d = 45;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 238;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 10u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:128:0:10");
}

TEST(GoldenNearestNeighbor_128_10, WindowedWithEpsilon) {
	constexpr uint64_t w = 32;
	constexpr uint64_t d = 55;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 175;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 10u);
	constexpr uint64_t epsilon =  1;

	run(w, d, r, N, size, find_all, THRESHHOLD, epsilon, "w:128:1:10");
}
/* not working
TEST(GoldenNearestNeighbor_128_10, IndykMotwani) {
	constexpr uint64_t w = 34;
	constexpr uint64_t l = 10;

	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << l);

	run_indyk(w, l, find_all, size,  "i:128:10");
}
*/

TEST(GoldenNearestNeighbor_128_15, Quadratic) {
	constexpr uint64_t w = 28;
	constexpr uint64_t size =  (1u << 15u);

	run_quadratic(w, size, "q:128:15");
}

TEST(GoldenNearestNeighbor_128_15, Windowed) {
	// Laut Script für n = 128 \lambda = 0.1171875 da size = 15
	// n: 128 lam: 0.1171875 w: 0.2230096042403977 r: 2.142857142857143 N: 454 d: 0.1651517530476932 q: 0.28154086496420877
	// n: 128 size: 15.0 w: 28.545229342770906 r: 2 N: 454 d: 21.13942439010473 q: 0.28154086496420877
	constexpr uint64_t w = 28;
	constexpr uint64_t d = 52;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 1454;
	constexpr uint64_t THRESHHOLD = 20;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 15u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:128:0:15");
}

TEST(GoldenNearestNeighbor_128_15, WindowedWithEpsilon) {
	constexpr uint64_t w = 28;
	constexpr uint64_t d = 48;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 1454;
	constexpr uint64_t THRESHHOLD = 20;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 15u);
	constexpr uint64_t epsilon =  1;

	run(w, d, r, N, size, find_all, THRESHHOLD, epsilon, "w:128:1:15");
}

/* not working
TEST(GoldenNearestNeighbor_128_15, IndykMotwani) {
	constexpr uint64_t w = 28;
	constexpr uint64_t l = 15;

	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << l);

	run_indyk(w, l, find_all, size,  "i:128:15");
}
*/

TEST(GoldenNearestNeighbor_128_20, Quadratic) {
	constexpr uint64_t w = 24;
	constexpr uint64_t size = (1u<<20);

	run_quadratic(w, size, "q:128:20");
}

// 20s - 120s
TEST(GoldenNearestNeighbor_128_20, Windowed) {
	// Laut Script für n = 128 \lambda = 0.15625 da size = 20
	// n: 128 lam: 0.15625 w: 0.18341914250537633 r: 2.857142857142857 N: 1008 d: 0.11011788229269559 q: 0.1269708942029263
	// n: 128 size: 20.0 w: 23.47765024068817 r: 2 N: 1008 d: 14.095088933465036 q: 0.1269708942029263
	constexpr uint64_t w = 24; // 24
	constexpr uint64_t d = 52;
	constexpr uint64_t r = 4;
	constexpr uint64_t N = 20;  // 2000
	constexpr uint64_t THRESHHOLD = 500;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 20u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:128:0:20");
}

// 33.3511, 391.489s Fastest parameter set as benchmarks show
TEST(GoldenNearestNeighbor_128_20, Windowed_Fastest) {
	// Laut Script für n = 128 \lambda = 0.15625 da size = 20
	// n: 128 lam: 0.15625 w: 0.18341914250537633 r: 2.857142857142857 N: 1008 d: 0.11011788229269559 q: 0.1269708942029263
	// n: 128 size: 20.0 w: 23.47765024068817 r: 2 N: 1008 d: 14.095088933465036 q: 0.1269708942029263
	constexpr uint64_t w = 24;
	constexpr uint64_t d = 48;
	constexpr uint64_t r = 4;
	constexpr uint64_t N = 10;
	constexpr uint64_t THRESHHOLD = 10;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 20u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:128:0:20");
}

// 7s, 53, 73s
TEST(GoldenNearestNeighbor_128_20, WindowedWithEpsilon) {
	constexpr uint64_t w = 24;
	constexpr uint64_t d = 44;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 1000;
	constexpr uint64_t THRESHHOLD = 20;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 20u);
	constexpr uint64_t epsilon =  1;

	run(w, d, r, N, size, find_all, THRESHHOLD, epsilon, "w:128:1:20");
}


/* not working
TEST(GoldenNearestNeighbor_128_20, IndykMotwani) {
	constexpr uint64_t w = 24;
	constexpr uint64_t l = 20;

	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << l);

	run_indyk(w, l, find_all, size,  "i:128:20");
}
*/
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
