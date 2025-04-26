#include "benchmark/benchmark.h"
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#include "options.h"
#include "windowed_nn_v2.h"


NNList L1{1}, L2{1};
uint64_t pos1, pos2;


static void do_setup(const benchmark::State& state) {
	NearestNeighbor::create_test_lists(L1, L2, TEST_BASE_LIST_SIZE, w, pos1, pos2);
}


// Define another benchmark
static void BM_quad(benchmark::State& state) {
	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];
	NearestNeighbor nn{L1, L2, w, r, N, d};

	for (auto _ : state) {
		nn.NN(gold1, gold2);
	}
}

static void BM_window(benchmark::State& state) {
	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];
	WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d};

	for (auto _ : state) {
		nn.NN(gold1, gold2);
	}
}

BENCHMARK(BM_quad)->Threads(1)->Setup(do_setup);
BENCHMARK(BM_window)->Threads(1)->Setup(do_setup);
BENCHMARK_MAIN();
