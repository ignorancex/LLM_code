#include <stdint.h>
#include "options.h"

// the only difference between this file and 'bench.c' is that this file benches times in sec and the other times in cpu cycles.
// everything else takes faaaar to loong
#ifndef ITERS
#define ITERS   1
#endif
#ifndef RUNS
#define RUNS    1
#endif

// if you uncomment this you need to comment out 'options.h'
/*#define TEST_BASE_LIST_SIZE_LOG (10u)
#define TEST_BASE_LIST_SIZE (1u << TEST_BASE_LIST_SIZE_LOG)
#define G_n 64
constexpr uint64_t w = 4;
constexpr uint64_t d = 20;
constexpr uint64_t r = 2;
constexpr uint64_t N = 20;
const uint64_t epsilon = 0;
const uint64_t size = (1u << 20u);
const uint64_t THRESHHOLD = 10;
*/

const bool find_all = false;

#include "windowed_nn_v2.h"
#include "indyk_motwani.h"

void log(const std::string &alg, const double time, const uint64_t errc){
	// normal human readable logging
	//std::cout << alg << " listsize: " << TEST_BASE_LIST_SIZE << ": w: " << w << " r: " << r << " N: " << N << " d: " << d << " time: " << time << "s err: " << errc << "=" << double(errc)/(double(ITERS*RUNS))*100.0 << "%" <<"\n";

	// csv logging.
	std::cout << alg << "," << G_n << ","  << TEST_BASE_LIST_SIZE << "," << w << "," << r << "," << N << "," << d << ","  << epsilon << "," << THRESHHOLD << "," << time << "," << uint64_t(double(errc)/(double(ITERS*RUNS))*100) <<"\n";
}

void bench_quad() {
	uint64_t errc = 0;
	double iter = 0;

	NNList L1{1}, L2{1};
	uint64_t pos1, pos2;

	for (int i = 0; i < ITERS; ++i) {
		double time = 0;

		NearestNeighbor::create_test_lists(L1, L2, TEST_BASE_LIST_SIZE, w, pos1, pos2);
		const NNContainer gold1 = L1[pos1];
		const NNContainer gold2 = L2[pos2];

		NearestNeighbor nn{L1, L2, w, r, N, d, find_all};

		for (int j = 0; j < RUNS; ++j) {
			double t0 = (double)clock()/CLOCKS_PER_SEC;
			nn.NN(gold1, gold2);
			/*uint64_t found = nn.NN();
			if (unlikely((found == 0) && (nn.sols_1.empty())) )
				errc += 1;
			*/
			time += ((double)clock()/CLOCKS_PER_SEC) - t0;
		}

		iter += time/RUNS;
	}

	log("Quad",  iter/ITERS, errc);
}

void bench_indyk() {
	uint64_t errc = 0;
	double iter = 0;

	NNList L1{1}, L2{1};
	uint64_t pos1, pos2;

	for (int i = 0; i < ITERS; ++i) {
		double time = 0;

		NearestNeighbor::create_test_lists(L1, L2, TEST_BASE_LIST_SIZE, w, pos1, pos2);
		const NNContainer gold1 = L1[pos1];
		const NNContainer gold2 = L2[pos2];


		for (int j = 0; j < RUNS; ++j) {
			IndykMontwani nn{L1, L2, w, TEST_BASE_LIST_SIZE_LOG, find_all};
			double t0 = (double)clock()/CLOCKS_PER_SEC;
			nn.NN(gold1, gold2);
			/*uint64_t found = nn.NN();
			if (unlikely((found == 0) && (nn.sols_1.empty())) )
				errc += 1;
			*/
			time += ((double)clock()/CLOCKS_PER_SEC) - t0;
		}

		iter += time/RUNS;
	}

	log("INDYC",  iter/ITERS, errc);
}

void bench_window2() {
	uint64_t errc = 0;
	double iter = 0;

	NNList L1{1}, L2{1};
	uint64_t pos1, pos2;

	for (int i = 0; i < ITERS; ++i) {
		double time = 0;

		NearestNeighbor::create_test_lists(L1, L2, TEST_BASE_LIST_SIZE, w, pos1, pos2, true, r);
		const NNContainer gold1 = L1[pos1];
		const NNContainer gold2 = L2[pos2];


		for (int j = 0; j < RUNS; ++j) {
			WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d, find_all, THRESHHOLD, epsilon};

			double t0 = (double)clock()/CLOCKS_PER_SEC;
			nn.NN(gold1, gold2);
			/*uint64_t found = nn.NN();
			if (unlikely((found == 0) && (nn.sols_1.empty())) )
				errc += 1;
			*/
			time += ((double)clock()/CLOCKS_PER_SEC) - t0;
		}

		iter += time/RUNS;
	}

	log("WIN2",  iter/ITERS, errc);
}

int main() {
	bench_quad();
	bench_indyk();
	bench_window2();
	return 0;
}
