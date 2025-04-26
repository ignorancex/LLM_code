#include <stdint.h>

#define ITERS   500
#define RUNS    500

#define G_n 256
constexpr uint64_t w = 0.2*G_n;
constexpr uint64_t r = 10;

#include "container.h"
#include "helper.h"


// measure the time to generate and element with a given weight/weight per window
void bench_random_with_weight() {
	double iter_rand = 0, iter_rand_wind = 0;

	for (int i = 0; i < ITERS; ++i) {
		double time_rand = 0, time_rand_wind = 0;

		NNContainer test1, test2;
		test1.zero(); test2.zero();

		for (int j = 0; j < RUNS; ++j) {
			double t0 = (double)clock()/CLOCKS_PER_SEC;
			test1.random_with_weight(w);
			time_rand += ((double)clock()/CLOCKS_PER_SEC) - t0;

			t0 = (double)clock()/CLOCKS_PER_SEC;
			test2.random_with_weight_per_windows(w, r);
			time_rand_wind += ((double)clock()/CLOCKS_PER_SEC) - t0;
		}

		iter_rand += time_rand/RUNS;
		iter_rand_wind += time_rand_wind/RUNS;
	}

	double t_rand = iter_rand/ITERS;
	double t_rand_wind = iter_rand_wind/ITERS;

	std:: cout << "RAND " << t_rand << "\n";
	std:: cout << "WIND " << t_rand_wind << "\n";
	std:: cout << "COEFF" << t_rand_wind/t_rand << "\n";
}


int main() {
	bench_random_with_weight();
	return 0;
}