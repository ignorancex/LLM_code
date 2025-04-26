#include <cstdint>
#include "options.h"

// everything else takes faaaar to loong
#ifndef ITERS
#define ITERS   1
#endif
#ifndef RUNS
#define RUNS    1
#endif

#include "windowed_nn_v2.h"
#include "indyk_motwani.h"

static __inline__ uint64_t ____rdtsc(void) {
    unsigned long long int x;
    __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
    // broken for 64-bit builds; don't copy this code
    return x;
}

uint64_t cycles(void) { // Access system counter for benchmarking
#if (OS_TARGET == OS_WIN) && (TARGET == TARGET_AMD64 || TARGET == TARGET_x86)

	return ____rdtsc();
#elif (OS_TARGET == OS_WIN) && (TARGET == TARGET_ARM)
	return __rdpmccntr64();
#elif (OS_TARGET == OS_LINUX) && (TARGET == TARGET_AMD64 || TARGET == TARGET_x86)
    unsigned int hi, lo;

    asm volatile ("rdtsc\n\t" : "=a" (lo), "=d"(hi));
    return ((uint64_t)lo) | (((uint64_t)hi) << 32);
#elif (OS_TARGET == OS_LINUX) && (TARGET == TARGET_ARM || TARGET == TARGET_ARM64)
    struct timespec time;

    clock_gettime(CLOCK_REALTIME, &time);
    return (uint64_t)(time.tv_sec*1e9 + time.tv_nsec);
#else
    return 0;
#endif
}

void log(const std::string &alg, const uint64_t time, const uint64_t errc){
	// normal human-readable logging
	//std::cout << alg << " listsize: " << TEST_BASE_LIST_SIZE << ": w: " << w << " r: " << r << " N: " << N << " d: " << d << " cycles: " << time << " err: " << errc << "=" << double(errc)/(double(ITERS*RUNS))*100.0 << "%" <<"\n";

	// csv logging.
	std::cout << alg << "," << TEST_BASE_LIST_SIZE << "," << w << "," << r << "," << N << "," << d << "," << epsilon << "," << time << "," << uint64_t(double(errc)/(double(ITERS*RUNS))*100) << "," << ITERS << "," << RUNS << "\n";
}

void bench_quad() {
	uint64_t errc = 0;
	uint64_t iter = 0;

	NNList L1{1}, L2{1};
	uint64_t pos1, pos2;

	for (int i = 0; i < ITERS; ++i) {
		uint64_t time = 0;
		NearestNeighbor::create_test_lists(L1, L2, TEST_BASE_LIST_SIZE, w, pos1, pos2);
		const NNContainer gold1 = L1[pos1];
		const NNContainer gold2 = L2[pos2];

		for (int j = 0; j < RUNS; ++j) {
			NearestNeighbor nn{L1, L2, w, r, N, d};

			uint64_t t0 = cycles();
			uint64_t found = nn.NN(gold1, gold2);
			time += cycles() - t0;

			if(!gold1.is_equal(nn.sols_1[found]) || !gold2.is_equal(nn.sols_2[found])){
				errc += 1;
			}
		}

		iter += time/RUNS;
	}

	log("Quad",  iter/ITERS, errc);
}

void bench_indyk() {
	uint64_t errc = 0;
	uint64_t iter = 0;

	NNList L1{1}, L2{1};
	uint64_t pos1, pos2;

	for (int i = 0; i < ITERS; ++i) {
		uint64_t time = 0;

		NearestNeighbor::create_test_lists(L1, L2, TEST_BASE_LIST_SIZE, w, pos1, pos2, true, r);

		const NNContainer gold1 = L1[pos1];
		const NNContainer gold2 = L2[pos2];

		for (int j = 0; j < RUNS; ++j) {
			IndykMontwani nn{L1, L2, w, TEST_BASE_LIST_SIZE_LOG};

			uint64_t t0 = cycles();
			uint64_t found = nn.indyk_NN(gold1, gold2);
			time += cycles() - t0;

			if(!gold1.is_equal(nn.sols_1[found]) || !gold2.is_equal(nn.sols_2[found])){
				errc += 1;
			}
		}

		iter += time/RUNS;
	}

	log("INDYC",  iter/ITERS, errc);
}

void bench_window2() {
	uint64_t errc = 0;
	uint64_t iter = 0;

	NNList L1{1}, L2{1};
	uint64_t pos1, pos2;

	for (int i = 0; i < ITERS; ++i) {
		uint64_t time = 0;

		NearestNeighbor::create_test_lists(L1, L2, TEST_BASE_LIST_SIZE, w, pos1, pos2, true, r);

		const NNContainer gold1 = L1[pos1];
		const NNContainer gold2 = L2[pos2];

		for (int j = 0; j < RUNS; ++j) {
			WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d};

			uint64_t t0 = cycles();
			uint64_t found = nn.NN(gold1, gold2);
			time += cycles() - t0;

			if(!gold1.is_equal(nn.sols_1[found]) || !gold2.is_equal(nn.sols_2[found])){
				errc += 1;
			}
		}

		iter += time/RUNS;
	}

	log("WIN2",  iter/ITERS, errc);
}

int main() {
	// Dont waste your time:
	// bench_quad();
	// bench_indyk();
	bench_window2();
	return 0;
}
