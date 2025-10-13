#include <gtest/gtest.h>

#include <filesystem>

#include "instance.hpp"
#include "randomized_greedy.hpp"

namespace fs = std::filesystem;

class RandomizedGreedyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // This logic to find test instances seems standard for this project.
        fs::path currentPath = fs::current_path();
        // Heuristics to find the project root from the build directory
        while (currentPath.has_parent_path() && !fs::exists(currentPath / "cpp_src")) {
            currentPath = currentPath.parent_path();
        }
        cppSourceDir_ = currentPath / "cpp_src";
        examplePath_ = cppSourceDir_ / "tests" / "instances" / "example_1.txt";
        grf1Path_ = currentPath / "instances" / "instances_release_1" / "grf1.txt";
        grf4Path_ = currentPath / "instances" / "instances_release_1" / "grf4.txt";
    }

    fs::path cppSourceDir_;
    fs::path examplePath_;
    fs::path grf1Path_;
    fs::path grf4Path_;
};

TEST_F(RandomizedGreedyTest, SolvesExampleInstance) {
    auto inst = Instance::loadInstanceFromFile(examplePath_.string());
    ASSERT_NE(inst, nullptr);

    RandomizedGreedyConstruction greedy(*inst, 0.3);  // Use some randomization
    greedy.run();
    auto tour = greedy.getSolution();

    ASSERT_FALSE(tour.empty()) << "Heuristic failed to find a solution.";
    EXPECT_EQ(tour.size(), inst->getN());
    EXPECT_TRUE(inst->checkSolutionCorrectness(tour));
}

TEST_F(RandomizedGreedyTest, PureGreedyIsDeterministicAndCorrect) {
    auto inst = Instance::loadInstanceFromFile(examplePath_.string());
    ASSERT_NE(inst, nullptr);

    RandomizedGreedyConstruction greedy1(*inst, 0.0);
    greedy1.run();
    auto tour1 = greedy1.getSolution();

    ASSERT_FALSE(tour1.empty());
    EXPECT_TRUE(inst->checkSolutionCorrectness(tour1));

    RandomizedGreedyConstruction greedy2(*inst, 0.0);
    greedy2.run();
    auto tour2 = greedy2.getSolution();

    ASSERT_FALSE(tour2.empty());
    EXPECT_EQ(tour1, tour2) << "Pure greedy should be deterministic.";
}

TEST_F(RandomizedGreedyTest, HandlesGreedyTrapWithRandomization) {
    // A trap instance where pure greedy fails.
    boost::unordered_map<std::pair<int, int>, double> edges = {
        {{0, 1}, 10.0},
        {{1, 2}, 10.0},
        {{2, 3}, 10.0},
        {{2, 4}, 1.0},   // trap edge
        {{3, 4}, 10.0},  // correct edge
        {{4, 0}, 10.0},
    };
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations;
    Instance inst(5, edges, relations, "trap_test");

    // Pure greedy should fail.
    RandomizedGreedyConstruction greedy_fail(inst, 0.0);
    greedy_fail.run();
    auto tour_fail = greedy_fail.getSolution();
    EXPECT_TRUE(tour_fail.empty()) << "Pure greedy was expected to fail but found a solution.";

    // Randomized greedy should succeed.
    RandomizedGreedyConstruction greedy_succeed(inst, 0.8);
    greedy_succeed.run();
    auto tour_succeed = greedy_succeed.getSolution();
    ASSERT_FALSE(tour_succeed.empty()) << "Randomized greedy should have found a solution.";
    EXPECT_EQ(tour_succeed.size(), inst.getN());
    EXPECT_TRUE(inst.checkSolutionCorrectness(tour_succeed));
}

TEST_F(RandomizedGreedyTest, SolvesGRF1) {
    auto inst = Instance::loadInstanceFromFile(grf1Path_.string());
    ASSERT_NE(inst, nullptr);

    RandomizedGreedyConstruction greedy(*inst, 0.5);
    greedy.run();
    auto tour = greedy.getSolution();
    EXPECT_EQ(tour.size(), inst->getN());
    EXPECT_TRUE(inst->checkSolutionCorrectness(tour));
}

TEST_F(RandomizedGreedyTest, SolvesGRF4) {
    auto inst = Instance::loadInstanceFromFile(grf4Path_.string());
    ASSERT_NE(inst, nullptr);

    RandomizedGreedyConstruction greedy(*inst, 0.5);
    greedy.run();
    auto tour = greedy.getSolution();
    EXPECT_EQ(tour.size(), inst->getN());
    EXPECT_TRUE(inst->checkSolutionCorrectness(tour));
}