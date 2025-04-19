#include <gtest/gtest.h>
#include <cstdint>

#define NN_CONFIG_SET
#define G_n                     128
#define TEST_BASE_LIST_SIZE_LOG 10u
#define TEST_BASE_LIST_SIZE (1u << TEST_BASE_LIST_SIZE_LOG)
#define SOLUTION_LOGGING

#include "helper.h"
#include "container.h"
#include "list.h"
#include "windowed_nn_v2.h"
#include "indyk_motwani.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr uint64_t w = 4; // 0.2*G_n;// 0.2*G_n;       // omega
constexpr uint64_t d = uint64_t(0.3*G_n);
constexpr uint64_t r = 1;
constexpr uint64_t N = 1000;
constexpr uint64_t THRESHHOLD = 500;
constexpr bool find_all = true; // actually doesn't matter.
constexpr uint64_t size = TEST_BASE_LIST_SIZE;
constexpr uint64_t epsilon = 1;


TEST(GoldenNearestNeighbor2, Windowed) {
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	NearestNeighbor::create_test_lists(L1, L2, size, w, pos1, pos2, true, r);
	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];

	std::cout << "Solution should be: " << pos1 << " " << pos2 << "\n";
	std::cout << "first: " << L1[pos1] << " w:" << L1[pos1].weight() << "\n";
	std::cout << "second:" << L2[pos2] << " w:" << L2[pos2].weight() << "\n";
	std::cout << "List Size: " << L1.size() << " " << L2.size() << "\n";
	std::cout << "\n\n";

	WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d};
	uint64_t found = nn.NN(gold1, gold2);
	EXPECT_EQ(gold1.is_equal(nn.sols_1[found]), true);
	EXPECT_EQ(gold2.is_equal(nn.sols_2[found]), true);
	EXPECT_EQ(nn.print_result(gold1, gold2), true);
}

TEST(GoldenNearestNeighbor2, WindowedWithEpsilon) {
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	NearestNeighbor::create_test_lists(L1, L2, size, w, pos1, pos2, true, r);
	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];

	std::cout << "Solution should be: " << pos1 << " " << pos2 << "\n";
	std::cout << "first: " << L1[pos1] << " w:" << L1[pos1].weight() << "\n";
	std::cout << "second:" << L2[pos2] << " w:" << L2[pos2].weight() << "\n";
	std::cout << "List Size: " << L1.size() << " " << L2.size() << "\n";
	std::cout << "\n\n";

	WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d, find_all, THRESHHOLD, epsilon};
	uint64_t found = nn.NN(gold1, gold2);
	EXPECT_EQ(gold1.is_equal(nn.sols_1[found]), true);
	EXPECT_EQ(gold2.is_equal(nn.sols_2[found]), true);
	EXPECT_EQ(nn.print_result(gold1, gold2), true);
}

TEST(GoldenNearestNeighbor, IndykMontwani) {
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};

	NearestNeighbor::create_test_lists(L1, L2, size, w, pos1, pos2);
	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];

	std::cout << "Solution should be: " << pos1 << " " << pos2 << "\n";
	std::cout << "first: " << L1[pos1] << " w:" << L1[pos1].weight() << "\n";
	std::cout << "second:" << L2[pos2] << " w:" << L2[pos2].weight() << "\n";
	std::cout << "List Size: " << L1.size() << " " << L2.size() << "\n";
	std::cout << "\n\n";

	IndykMontwani nn{L1, L2, w, TEST_BASE_LIST_SIZE_LOG};
	uint64_t found = nn.indyk_NN(gold1, gold2);
	EXPECT_EQ(gold1.is_equal(nn.sols_1[found]), true);
	EXPECT_EQ(gold2.is_equal(nn.sols_2[found]), true);
	EXPECT_EQ(nn.print_result(gold1, gold2), true);
}

TEST(Dist_GoldenNearestNeighbor, Windowed) {
	uint64_t pos1, pos2;
	NNList L1{1}, L2{1};
	uint64_t found;

	constexpr uint64_t w = 24;
	constexpr uint64_t d = 40;
	constexpr uint64_t r = 2;
	constexpr uint64_t N = 132;
	constexpr uint64_t THRESHHOLD = 100;
	constexpr bool find_all = false; // actually doesnt matter.
	constexpr uint64_t size =  (1u << 10u);
	constexpr uint64_t gam = 0.1*G_n;

	NearestNeighbor::create_test_lists_with_distribution(L1, L2, size, gam, w, pos1, pos2);
	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];

	NearestNeighbor nn_{L1, L2, w, r, N, d, find_all, THRESHHOLD};
	found = nn_.NN(gold1, gold2);
	EXPECT_EQ(gold1.is_equal(nn_.sols_1[found]), true);
	EXPECT_EQ(gold2.is_equal(nn_.sols_2[found]), true);
	EXPECT_EQ(nn_.print_result(gold1, gold2), true);
	EXPECT_EQ(L1.size(), size);
	EXPECT_EQ(L2.size(), size);


	WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d, find_all, THRESHHOLD};
	found = nn.NN(gold1, gold2);
	EXPECT_EQ(gold1.is_equal(nn.sols_1[found]), true);
	EXPECT_EQ(gold2.is_equal(nn.sols_2[found]), true);
	EXPECT_EQ(nn.print_result(gold1, gold2), true);
	EXPECT_EQ(L1.size(), size);
	EXPECT_EQ(L2.size(), size);

}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
