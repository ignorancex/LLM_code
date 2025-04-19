#include <stdint.h>
#include <gtest/gtest.h>

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// This file will be automatically generated if you run `cd bench && python gen.py -o options2.h -b bench.csv`
#include "options.h"

#ifndef ITERS
#define ITERS   1
#endif
#ifndef RUNS
#define RUNS    1
#endif

// This can be set globally for this test. Because we want to find all solutions during the quadratic search. So we have
// an exact number of which we __MUST__ find a 'ratio'-fraction during the execution of our algorithm.
const bool find_all = true;

#include "windowed_nn_v2.h"

void log(const std::string &alg, const double time, const uint64_t errc){
	// normal human readable logging
	//std::cout << alg << " listsize: " << TEST_BASE_LIST_SIZE << ": w: " << w << " r: " << r << " N: " << N << " d: " << d << " time: " << time << "s err: " << errc << "=" << double(errc) <<"\n";

	// csv logging.
	std::cout << alg << "," << G_n << "," << TEST_BASE_LIST_SIZE << "," << w << "," << r << "," << N << "," << d << "," << THRESHHOLD << "," << epsilon << "," << gam << "," << time << "," << double(errc) <<"\n";
}

void bench_dist() {
	NNList L1{1}, L2{1};
	uint64_t pos1, pos2;

	double iter = 0.0;
	double sols = 0.0;

	/*
	NearestNeighbor nn_q{L1, L2, w, 1 , 1, 1, true};
	nn_q.NN(true);
	// EXPECT_EQ(nn_q.every_solution_unique(), true);
	const uint64_t nr_solutions = nn_q.sols_1.size();
	*/

	for (int i = 0; i < ITERS; ++i) {
		double time = 0.0;
		double inner_sols = 0.0;

		NearestNeighbor::create_test_lists_with_distribution(L1, L2, TEST_BASE_LIST_SIZE, gam, w, pos1, pos2, true, r);
		const NNContainer gold1 = L1[pos1];
		const NNContainer gold2 = L2[pos2];

		for (int j = 0; j < RUNS; ++j) {
			double t0 = (double)clock()/CLOCKS_PER_SEC;

			WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d, find_all, THRESHHOLD, epsilon};
			uint64_t found = nn.NN(gold1, gold2);
			time += ((double)clock()/CLOCKS_PER_SEC) - t0;

			EXPECT_EQ(gold1.is_equal(nn.sols_1[found]), true);
			EXPECT_EQ(gold2.is_equal(nn.sols_2[found]), true);

			inner_sols += nn.sols_1.size();
		}

		iter += time/RUNS;
		sols += inner_sols/RUNS;
	}

	log("dist",  iter/ITERS, sols/ITERS);
}

int main() {
	bench_dist();
	return 0;
}
