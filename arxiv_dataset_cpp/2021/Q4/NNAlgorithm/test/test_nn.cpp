#include <gtest/gtest.h>
#include <cstdint>

#define NN_CONFIG_SET
#define G_n                     100

#define SORT_INCREASING_ORDER
#define TEST_BASE_LIST_SIZE (1u << 10u)
//#define COMPRESS_OUTPUT

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

TEST(OptimalParameters, kk) {
	const uint64_t n    = 100;
	const double w_r    = 0.2;
	const uint64_t w    = w_r*n;
	const double lam_r  = 0.15;
	const uint64_t lam  = lam_r*n;

	uint64_t r, N, d;
	NearestNeighbor::optimal_params(n, w, lam_r, &r, &N, &d);
	std::cout << "optimal parameter set for n=" << n << " log(lam)=" << lam << " w=" << w << "\n";
	std::cout <<  " r: " << r << " N: " << N << " d: " << d  << "\n";
}

TEST(Container, random_with_weight_per_windows) {
	const uint64_t test_k = 4;
	const uint64_t test_w = G_n/test_k/2;
	NNContainer v;

	v.random_with_weight_per_windows(test_w, test_k);
	std::cout << "out: " << v << "\n";
}

TEST(List, create_test_lists_with_distribution) {
	const uint64_t gamma = 20;
	const uint64_t omega = 10;
	const uint64_t size = TEST_BASE_LIST_SIZE;
	uint64_t pos1, pos2;
	NNContainer tmp;
	NNList L1{1}, L2{1};

	NearestNeighbor::create_test_lists_with_distribution(L1, L2, size, gamma, omega, pos1, pos2);
	EXPECT_EQ(L1.size(), L2.size());


	for (uint64_t i = 0; i < L1.size(); ++i) {
		if (i == pos1){
			continue;
		}
		EXPECT_EQ(L1[i].weight(), gamma);
	}

	for (uint64_t j = 0; j < L2.size(); ++j) {
		if (j == pos2){
			continue;
		}
		EXPECT_EQ(L2[j].weight(), gamma);
	}

	uint64_t w = NNContainer::add(tmp, L1[pos1], L2[pos2]);
	EXPECT_EQ(w, omega);
	EXPECT_EQ(w, tmp.weight());

}

TEST(NearestNeighbor2, Windowed) {
	uint64_t pos1, pos2;
	const uint64_t w = 0.2*G_n;       // omega
	const uint64_t d = uint64_t(0.3*G_n);
	const uint64_t r = 2;
	const uint64_t N = 10000;
	const uint64_t size = TEST_BASE_LIST_SIZE;
	const bool find_all = true;
	const uint64_t tresh = 100;

	NNList L1{1}, L2{1};

	NearestNeighbor::create_test_lists(L1, L2, size, w, pos1, pos2, true, r);
	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];

	std::cout << "Solution should be: " << pos1 << " " << pos2 << "\n";
	std::cout << "first: " << L1[pos1] << " w:" << L1[pos1].weight() << "\n";
	std::cout << "second:" << L2[pos2] << " w:" << L2[pos2].weight() << "\n";
	std::cout << "List Size: " << L1.size() << " " << L2.size() << "\n";
	std::cout << "\n\n";

	WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d, find_all, tresh};
	uint64_t found = nn.NN();
	EXPECT_EQ((found > 0) || (!nn.sols_1.empty()), true);
	EXPECT_EQ(nn.print_result(gold1, gold2), true);


	// normal quadratic search
	NearestNeighbor nnq{L1, L2, w, r, N, d};
	nnq.NN();
	EXPECT_EQ(nnq.print_result(gold1, gold2), true);
}

TEST(NearestNeighbor2, WindowedWithEpsilon) {
	uint64_t pos1, pos2;
	const uint64_t w = 0.2*G_n;       // omega
	const uint64_t d = uint64_t(0.3*G_n);
	const uint64_t r = 2;
	const uint64_t N = 10000;
	const uint64_t epsilon = 1;
	const uint64_t size = TEST_BASE_LIST_SIZE;
	const bool find_all = true;
	const uint64_t tresh = 100;
	NNList L1{1}, L2{1};

	NearestNeighbor::create_test_lists(L1, L2, size, w, pos1, pos2, true, r);
	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];

	std::cout << "Solution should be: " << pos1 << " " << pos2 << "\n";
	std::cout << "first: " << L1[pos1] << " w:" << L1[pos1].weight() << "\n";
	std::cout << "second:" << L2[pos2] << " w:" << L2[pos2].weight() << "\n";
	std::cout << "List Size: " << L1.size() << " " << L2.size() << "\n";
	std::cout << "\n\n";

	WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d, find_all, tresh, epsilon};
	uint64_t found = nn.NN();
	EXPECT_EQ((found > 0) || (!nn.sols_1.empty()), true);
	EXPECT_EQ(nn.print_result(gold1, gold2), true);


	// normal quadratic search
	NearestNeighbor nnq{L1, L2, w, r, N, d};
	nnq.NN();
	EXPECT_EQ(nnq.print_result(gold1, gold2), true);

	// Do the stupid search:
	std::vector<std::pair<NNContainer, NNContainer>> test;
	std::vector<std::pair<uint64_t , uint64_t>> poss;
	NNContainer tmp;
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
