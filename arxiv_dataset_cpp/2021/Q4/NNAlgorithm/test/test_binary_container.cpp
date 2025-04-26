#include <gtest/gtest.h>
#include <cstdint>
#include <random>
#include <vector>
#include <utility>

#define G_n 250
#include "container.h"

using namespace std;

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(Internals, access_pass_through) {
	BinaryContainer<G_n> b1;
	BinaryContainer<G_n> b2;

	b1.zero();
	b2.random();

	EXPECT_EQ(b1.size(), b2.size());
	for (int i = 0; i < b1.size(); ++i) {
		b2[i] = b1[i];
	}

	for (int i = 0; i < b1.size(); ++i) {
		EXPECT_EQ(b2[i], b1[i]);
	}
}

TEST(Internals, round_up_to_limb){
	BinaryContainer<64> b;

	EXPECT_EQ(b.round_up_to_limb(0), 1);
	EXPECT_EQ(b.round_up_to_limb(20), 1);
	EXPECT_EQ(b.round_up_to_limb(63), 1);
	EXPECT_EQ(b.round_up_to_limb(64), 2);
}

TEST(Internals, round_down_to_limb){
	BinaryContainer<64> b;

	EXPECT_EQ(b.round_down_to_limb(0), 0);
	EXPECT_EQ(b.round_down_to_limb(20), 0);
	EXPECT_EQ(b.round_down_to_limb(63), 0);
	EXPECT_EQ(b.round_down_to_limb(64), 1);
}

TEST(Internals, compute_limbs){
	BinaryContainer<63> b1;
	EXPECT_EQ(b1.compute_limbs(), 1);

	BinaryContainer<64> b2;
	EXPECT_EQ(b2.compute_limbs(), 1);

	BinaryContainer<65> b3;
	EXPECT_EQ(b3.compute_limbs(), 2);

	BinaryContainer<250> b4;
	EXPECT_EQ(b4.compute_limbs(), 4);
}

TEST(Internals, masks){
	BinaryContainer<G_n> b;

	EXPECT_EQ(b.mask(0), 1);
	EXPECT_EQ(b.mask(1), 2);
	EXPECT_EQ(b.mask(2), 4);
}



TEST(Static_Add, Probabilistic){
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);

	vector<pair<uint64_t, uint64_t>> boundsSet = {pair(0, 64), pair(0, 10), pair(2, 70), pair(64, 128), pair(0, 65), pair(3, 66)};

	for(auto bounds : boundsSet){
		uint64_t k_lower = bounds.first;
		uint64_t k_upper = bounds.second;

		for(uint64_t i = 0; i < 1; i++){
			uint64_t a = dis(gen);
			uint64_t b = dis(gen);
			uint64_t c = dis(gen);
			uint64_t d = dis(gen);
			uint64_t e = dis(gen);
			uint64_t f = dis(gen);

			BinaryContainer<128> b1;
			BinaryContainer<128> b2;
			BinaryContainer<128> b3;

			b1.data()[0] = a; b1.data()[1] = b;
			b2.data()[0] = c; b2.data()[1] = d;
			b3.data()[0] = e; b3.data()[1] = f;


			BinaryContainer<128>::add(b3, b2, b1, k_lower, k_upper);

			for(uint64_t k = 0; k < k_lower; k++){
				if(k < 64){
					EXPECT_EQ(b3.get_bit_shifted(k), (e>>k) & 1);
				}
				else {
					EXPECT_EQ(b3.get_bit_shifted(k), (f>>k) & 1);
				}
			}
			for(uint64_t k = k_lower; k < k_upper; k++){
				if(k < 64){
					EXPECT_EQ(b3.get_bit_shifted(k), ((a^c) >> k) & 1);
				}
				else {
					EXPECT_EQ(b3.get_bit_shifted(k), ((b^d) >> k) & 1);
				}
			}
			for(uint64_t k = k_upper; k < 128; k++){
				if(k < 64){
					EXPECT_EQ(b3.get_bit_shifted(k), (e>>k) & 1);
				}
				else {
					EXPECT_EQ(b3.get_bit_shifted(k), (f>>k) & 1);
				}
			}
		}
	}
}



TEST(Add, Full_Length_Zero) {
	BinaryContainer<G_n> b1;
	BinaryContainer<G_n> b2;
	BinaryContainer<G_n> b3;

	b1.zero(); b2.zero(); b3.zero();

	uint64_t w = BinaryContainer<G_n>::add(b3, b1, b2, 0, G_n);
	EXPECT_EQ(w, 0);
	EXPECT_EQ(w, b3.weight());
	EXPECT_EQ(w, b2.weight());
	EXPECT_EQ(w, b1.weight());

	for (int j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}
}

TEST(Add, Full_Length_One) {
	BinaryContainer<G_n> b1;
	BinaryContainer<G_n> b2;
	BinaryContainer<G_n> b3;
	uint64_t w;

	b1.zero(); b2.zero(); b3.zero();

	b1[0] = true;
	w = BinaryContainer<G_n>::add(b3, b1, b2, 0, G_n);
	EXPECT_EQ(w, 1);
	EXPECT_EQ(w, b3.weight());
	EXPECT_EQ(0, b2.weight());
	EXPECT_EQ(w, b1.weight());

	EXPECT_EQ(1, b3[0]);

	for (int j = 1; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}

	// 2. test.
	b1.one(); b2.zero(); b3.zero();

	w = BinaryContainer<G_n>::add(b3, b1, b2, 0, G_n);
	EXPECT_EQ(w, b1.size());
	EXPECT_EQ(w, b3.weight());
	EXPECT_EQ(0, b2.weight());
	EXPECT_EQ(w, b1.weight());

	for (int j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(true, b3[j]);
		EXPECT_EQ(1, b3[j]);
	}

	//3.test
	b1.one(); b2.one(); b3.zero();
	w = BinaryContainer<G_n>::add(b3, b1, b2, 0, G_n);

	EXPECT_EQ(w, 0);
	EXPECT_EQ(b2.size(), b2.weight());
	EXPECT_EQ(b1.size(), b1.weight());
	for (int j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(false, b3[j]);
		EXPECT_EQ(0, b3[j]);
	}
}

TEST(Add, OffByOne_Lower_One) {
	BinaryContainer<G_n> b1;
	BinaryContainer<G_n> b2;
	BinaryContainer<G_n> b3;
	uint64_t w;

	b1.zero(); b2.zero(); b3.zero();

	b1[0] = true;   // this should be ignored.
	w = BinaryContainer<G_n>::add(b3, b1, b2, 1, G_n);
	EXPECT_EQ(w, 0);
	EXPECT_EQ(b3.weight(), 0);
	EXPECT_EQ(b2.weight(), 0);
	EXPECT_EQ(b1.weight(), 1);

	for (int j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}

	// 2. test.
	b1.one(); b2.zero(); b3.zero();

	w = BinaryContainer<G_n>::add(b3, b1, b2, 1, G_n);
	EXPECT_EQ(b3.weight(), w);
	EXPECT_EQ(w, b3.size()-1);
	EXPECT_EQ(b2.weight(), 0);
	EXPECT_EQ(b1.weight(), b1.size());
	EXPECT_EQ(0, b3[0]);
	EXPECT_EQ(false, b3[0]);
	for (int j = 1; j < b3.size(); ++j) {
		EXPECT_EQ(true, b3[j]);
		EXPECT_EQ(1, b3[j]);
	}

	//3.test
	b1.one(); b2.one(); b3.zero();

	w = BinaryContainer<G_n>::add(b3, b1, b2, 1, G_n);
	EXPECT_EQ(b3.weight(), w);
	EXPECT_EQ(w, 0);
	EXPECT_EQ(b2.weight(), b2.size());
	EXPECT_EQ(b1.weight(), b1.size());

	EXPECT_EQ(0, b3[0]);
	EXPECT_EQ(false, b3[0]);

	for (int j = 1; j < b3.size(); ++j) {
		EXPECT_EQ(false, b3[j]);
		EXPECT_EQ(0, b3[j]);
	}


	//3.test
	b1.one(); b2.one(); b3.zero();
	b3[0] = 1;

	w = BinaryContainer<G_n>::add(b3, b1, b2, 1, G_n);
	EXPECT_EQ(b3.weight(), w+1);
	EXPECT_EQ(w, 0);
	EXPECT_EQ(b2.weight(), b2.size());
	EXPECT_EQ(b1.weight(), b1.size());

	EXPECT_EQ(1, b3[0]);
	EXPECT_EQ(true, b3[0]);

	for (int j = 1; j < b3.size(); ++j) {
		EXPECT_EQ(false, b3[j]);
		EXPECT_EQ(0, b3[j]);
	}

}

TEST(Add, OffByOne_Higher_One) {
	BinaryContainer<G_n> b1;
	BinaryContainer<G_n> b2;
	BinaryContainer<G_n> b3;
	uint64_t w ;

	b1.zero(); b2.zero(); b3.zero();

	b1[G_n-1] = true;   // this should be ignored.
	w = BinaryContainer<G_n>::add(b3, b1, b2, 0, G_n - 1);
	EXPECT_EQ(w, 0);
	EXPECT_EQ(b3.weight(), 0);

	for (int j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}

	// 2. test.
	b1.one(); b2.zero(); b3.zero();

	w = BinaryContainer<G_n>::add(b3, b1, b2, 0, G_n - 1);
	EXPECT_EQ(w, b3.size()-1);
	EXPECT_EQ(b3.weight(), w);

	EXPECT_EQ(0, b3[G_n-1]);
	EXPECT_EQ(false, b3[G_n-1]);
	for (int j = 0; j < b3.size() - 1; ++j) {
		EXPECT_EQ(true, b3[j]);
		EXPECT_EQ(1, b3[j]);
	}

	//3.test
	b1.one(); b2.one(); b3.zero();

	w = BinaryContainer<G_n>::add(b3, b1, b2, 0, G_n - 1);
	EXPECT_EQ(w, 0);
	EXPECT_EQ(b3.weight(), w);
	EXPECT_EQ(0, b3[G_n-1]);
	EXPECT_EQ(false, b3[G_n-1]);

	for (int j = 1; j < b3.size() - 1; ++j) {
		EXPECT_EQ(false, b3[j]);
		EXPECT_EQ(0, b3[j]);
	}


	//3.test
	b1.one(); b2.one(); b3.one();

	w = BinaryContainer<G_n>::add(b3, b1, b2, 0, G_n - 1);
	EXPECT_EQ(w, 0);
	EXPECT_EQ(b3.weight(), 1);
	EXPECT_EQ(1, b3[G_n-1]);
	EXPECT_EQ(true, b3[G_n-1]);

	for (int j = 1; j < b3.size() - 1; ++j) {
		EXPECT_EQ(false, b3[j]);
		EXPECT_EQ(0, b3[j]);
	}
}

TEST(Add, Complex_Ones) {
	BinaryContainer<G_n> b1;
	BinaryContainer<G_n> b2;
	BinaryContainer<G_n> b3;

	for (int k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (int k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			b1.one(); b2.zero(); b3.zero();


			uint64_t w = BinaryContainer<G_n>::add(b3, b1, b2, k_lower, k_higher);
			EXPECT_EQ(w, k_higher-k_lower);

			for (int j = 0; j < k_lower; ++j) {
				EXPECT_EQ(0, b3[j]);
			}
			for (int j = k_lower; j < k_higher; ++j) {
				EXPECT_EQ(1, b3[j]);
			}
			for (int j = k_higher; j < b1.size(); ++j) {
				EXPECT_EQ(0, b3[j]);
			}
		}
	}
}

TEST(Add, Complex_Ones2) {
	BinaryContainer<G_n> b1;
	BinaryContainer<G_n> b2;
	BinaryContainer<G_n> b3;

	for (int k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (int k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			b1.one(); b2.zero(); b3.one();


			uint64_t w = BinaryContainer<G_n>::add(b3, b1, b2, k_lower, k_higher);
			EXPECT_EQ(w, k_higher-k_lower);

			for (int j = 0; j < k_lower; ++j) {
				EXPECT_EQ(1, b3[j]);
			}
			for (int j = k_lower; j < k_higher; ++j) {
				EXPECT_EQ(1, b3[j]);
			}
			for (int j = k_higher; j < b1.size(); ++j) {
				EXPECT_EQ(1, b3[j]);
			}
		}
	}
}

TEST(Add_Full_Length, Probabilistic) {
	// Tests the full length add functions which returns also the weight of the element
	BinaryContainer<G_n> b1;
	BinaryContainer<G_n> b2;
	BinaryContainer<G_n> res;

	uint64_t weight;
	b1.zero(); b2.zero(); res.zero();
	for (int i = 0; i < b1.size(); ++i) {
		b1[i] = true;
		weight = BinaryContainer<G_n>::add(res, b1, b2);
		EXPECT_EQ(weight, i+1);
		EXPECT_EQ(true, b1.is_equal(res));
		EXPECT_EQ(false, b2.is_equal(res));

		res.zero();
	}
}

TEST(weight, Simple_Everything_True) {
	BinaryContainer<G_n> b1;
	b1.zero();

	for (int k_lower  = 1; k_lower < b1.size(); ++k_lower) {
		b1[k_lower-1] = true;
		EXPECT_EQ(k_lower, b1.weight());
		EXPECT_EQ(k_lower, b1.weight(0, k_lower));
		EXPECT_EQ(k_lower, b1.weight(0, b1.size()));

		if (k_lower + 1 < b1.size())
			EXPECT_EQ(0, b1.weight(k_lower +1, b1.size()));
	}
}


int main(int argc, char **argv) {
	srand(0);
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}