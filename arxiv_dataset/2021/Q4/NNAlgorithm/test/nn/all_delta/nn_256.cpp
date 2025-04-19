#include <gtest/gtest.h>
#include <cstdint>

#define NN_CONFIG_SET
#define G_n                     256
#define ALL_DELTA

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



TEST(GoldenNearestNeighbor_256_10_all_delta, Windowed) {
	// n = 256 \lambda = 0.0390625, because size = 10
	// n: 256 lam: 0.0390625 w: 0.3369549996940837 r: 1.25 N: 446 d: 0.3584098841512485 q: 0.5729355014372951
	// n: 256 size: 10.0 w: 86.26047992168543 r: 1 N: 446 d: 91.75293034271962 q: 0.5729355014372951
	constexpr uint64_t w = 88;
	constexpr uint64_t d = 120;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 10;
	constexpr uint64_t THRESHHOLD = 10;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 10u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:256:0:10");
}

TEST(GoldenNearestNeighbor_256_15_all_delta, Windowed) {
	// n = 256 \lambda = 0.05859375, because size = 15
	// n: 256 lam: 0.05859375 w: 0.3012491540443123 r: 1.875 N: 826 d: 0.21491245047530094 q: 0.309633111241245
	// n: 256 size: 15.0 w: 77.11978343534395 r: 1 N: 826 d: 55.01758732167704 q: 0.309633111241245
	constexpr uint64_t w = 76;
	constexpr uint64_t d = 110;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 20;
	constexpr uint64_t THRESHHOLD = 30;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 15u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:256:0:15");
}

TEST(GoldenNearestNeighbor_256_20_all_delta, Windowed) {
	// n = 256 \lambda = 0.078125, because size = 20
	// n: 256 lam: 0.078125 w: 0.271599172304043 r: 2.5 N: 2640 d: 0.14929471420827492 q: 0.09694183771265091
	// n: 256 size: 20.0 w: 69.52938810983501 r: 2 N: 2640 d: 38.21944683731838 q: 0.09694183771265091
	constexpr uint64_t w = 64;
	constexpr uint64_t d = 100;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 100;
	constexpr uint64_t THRESHHOLD = 50;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 20u);

	run(w, d, r, N, size, find_all, THRESHHOLD, 0, "w:256:0:20");
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
