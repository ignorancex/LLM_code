#include <gtest/gtest.h>
#include <cstdint>
#include <random>
#include <vector>
#include <utility>

#define G_n 256
#include "container.h"

using namespace std;

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

#include "tests.c++"

int main(int argc, char **argv) {
	srand(0);
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
