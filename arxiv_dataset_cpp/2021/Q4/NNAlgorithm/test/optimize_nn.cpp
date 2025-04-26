#include <gtest/gtest.h>
#include <cstdint>
#include <cmath>
#include <thread>

#define NN_CONFIG_SET
#define G_n     64


#include "bench.h"
#include "helper.h"
#include "container.h"
#include "list.h"
#include "windowed_nn_v2.h"

#ifndef ITERS
#define ITERS   2
#endif

#ifndef RUNS
#define RUNS   10
#endif

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(OptimizeWindowed2, Blub) {
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	// placeholders for the best parameters
	uint64_t b_r = 0;
	uint64_t b_d = 0;
	uint64_t b_N = 0;
	uint64_t b_e = 0;
	double b_time = 100000.0;

	// some unbenched parameters
	const uint64_t n = NNContainer::get_size();
	const bool find_all = false;    // this will allow the algorithms to early exit

	for (uint64_t lam_ = 20; lam_ < 21; ++lam_) {
		double lam = double(lam_)/double(G_n);
		uint64_t size = (1u << lam_);

		for (uint64_t w = 4; w >= 4; w--){
			int64_t r_ = uint64_t(lam*n/pow(log2(n),1)); // 2

			// TODO started Win2E listsize: 32768 w: 20 r: 10 N: 127 d: 32 epsilon: 1 schmiert ab n = 100
			for (uint64_t epsilon = 0; epsilon < 1; ++epsilon) {
				for (uint64_t threshhold = 10; threshhold < 60; threshhold += 10) {
					// r = 1 is not really good
					// be very careful by the choice of r
					for (uint64_t r = 2; r < 3; r += 5) {
						int64_t d_ = uint64_t(n * NearestNeighbor::H1(1. - double(lam)));

						// TODO w+5 ergibt allgemein keinen Sinn
						// started Win2E listsize: 32768 w: 40 r: 5 N: 458 d: 60 epsilon: 1 error
						for (uint64_t d = 18; d < 26; d += 1) {
							double q = NearestNeighbor::calc_q(w, d);
							int64_t N_ = n / q;
							for (uint64_t N = 20; N < 1000; N += 50) {
								// some logging of intermediate values
								// std::cout << "Win2E" << " listsize: " << size << " lam_: " << lam << " r_: " << r_ << " d_: " << d_ << " q: " << q << " N_: " << N_ << "\n";

								std::cout << "started Win2E" << " listsize: " << size << " w: " << w << " r: " << r
								          << " N: " << N << " d: " << d << " epsilon: " << epsilon << "\n"
								          << std::flush;

								// timing helper
								double iter = 0;

								// Now bench it
								for (int i = 0; i < ITERS; ++i) {
									double timing = 0;

									NearestNeighbor::create_test_lists(L1, L2, size, w, pos1, pos2, true, r);
									const NNContainer gold1 = L1[pos1];
									const NNContainer gold2 = L2[pos2];

									WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d, find_all, threshhold, epsilon};

									for (int j = 0; j < RUNS; ++j) {
										double t0 = (double) clock() / CLOCKS_PER_SEC;
										nn.NN(gold1, gold2);
										// nn.NN();
										timing += ((double) clock() / CLOCKS_PER_SEC) - t0;

										// std::cout << timing << "\n" << std::flush;
									}

									iter += timing / RUNS;
								}

								// some logging
								double timing = iter / ITERS;
								std::cout << "Win2E" << " listsize: " << size << " lam_: " << lam << ": w: " << w << "="
								          << double(w) / double(n) << " r: " << r << "=" << r_ << " N: " << N << "="
								          << N_ << " d: " << d << "=" << double(d) / double(n) << " epsilon: "
								          << epsilon << " timing: " << timing << "\n" << std::flush;

								// save the result if its the best.
								if (timing < b_time) {
									b_r = r;
									b_d = d;
									b_N = N;
									b_e = epsilon;
									b_time = timing;
								}
							}
						}
					}
				}
			}

			// 'optimal' parameter set. So theoretic 'optimal'
			uint64_t o_r = 0;
			uint64_t o_d = 0;
			uint64_t o_N = 0;
			NearestNeighbor::optimal_params(n, w, lam, &o_r, &o_N, &o_d);

			std::cout << "WIN2E best parameter set for n=" << n << " log(lam)=" << lam << " w=" << w << "\n";
			std::cout <<  " r: " << b_r << " N: " << b_N << " d: " << b_d << " epsilon: " << b_e << " time: " << b_time << "\n";
			std::cout << "WIN2E optimal parameter set for n=" << n << " log(lam)=" << lam << " w=" << w << "\n";
			std::cout <<  " r: " << o_r << " N: " << o_N << " d: " << o_d  << "\n";
		}
	}
}


// bsp um 64 bit zu optimieren.
TEST(OptimizeWindowed2, Blub2) {
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	// placeholders for the best parameters
	uint64_t b_r = 0;
	uint64_t b_d = 0;
	uint64_t b_N = 0;
	uint64_t b_e = 0;
	double b_time = 100000.0;
	double time = 100000.0;

	constexpr uint64_t lam = 20;
	constexpr uint64_t size = (1u << lam);
	constexpr bool find_all = true;
	constexpr uint64_t min_e = 0, max_e = 0;
	constexpr uint64_t min_t = 2, max_t = 1000;

	constexpr uint64_t min_w = 4, max_w = 4;
	constexpr uint64_t min_d = 18, max_d = 26;
	constexpr uint64_t min_r = 2, max_r = 2;
	constexpr uint64_t min_N = 20, max_N = 1000;

	constexpr double exit_time = 180.0;

	for (uint64_t w = min_w; w <= max_w; ++w) {
		for (uint64_t r = min_r; r <= max_r; ++r) {
			for (uint64_t epsilon = min_e; epsilon <= min_e; ++epsilon) {
				for (uint64_t THRESHHOLD = min_t; THRESHHOLD <= max_t; THRESHHOLD+=10) {
					for (uint64_t d = min_d; d <= max_d; ++d) {
						for (uint64_t N = min_N; N <= max_N; N += 20) {
							std::thread threado = std::thread(run, w, d, r, N, size, find_all, THRESHHOLD, epsilon,  std::string ("blub"), &time);
							threado.join();

							// std::cout << time << "\n" << std::flush;
							if(time < b_time) {
								b_r = r;
								b_d = d;
								b_N = N;
								b_e = epsilon;
								b_time = time;
							}
						}
					}
				}
			}
		}

		uint64_t o_r = 0;
		uint64_t o_d = 0;
		uint64_t o_N = 0;
		NearestNeighbor::optimal_params(G_n, w, lam, &o_r, &o_N, &o_d);

		std::cout << "WIN2E best parameter set for n=" << G_n << " log(lam)=" << lam << " w=" << w << "\n";
		std::cout <<  " r: " << b_r << " N: " << b_N << " d: " << b_d << " epsilon: " << b_e << " time: " << b_time << "\n";
		std::cout << "WIN2E optimal parameter set for n=" << G_n << " log(lam)=" << lam << " w=" << w << "\n";
		std::cout <<  " r: " << o_r << " N: " << o_N << " d: " << o_d  << "\n";
	}


}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
