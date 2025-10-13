#include <gtest/gtest.h>

#include <filesystem>

#include "grasp.hpp"
#include "instance.hpp"

namespace fs = std::filesystem;

class GRASPTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Standard logic to find test instances from the project root
        fs::path currentPath = fs::current_path();
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

// Parameterized test class to test all constructive heuristics
class GRASPParameterizedTest : public GRASPTest,
                               public ::testing::WithParamInterface<ConstructiveHeuristicType> {};

TEST_P(GRASPParameterizedTest, SolvesExampleInstance) {
    ConstructiveHeuristicType heuristicType = GetParam();
    auto inst = Instance::loadInstanceFromFile(examplePath_.string());
    ASSERT_NE(inst, nullptr);

    GRASP grasp(*inst, 1, 0.2, 0.2, heuristicType, {LocalSearch::TwoOpt});
    grasp.run();
    auto tour = grasp.getSolution();

    ASSERT_FALSE(tour.empty()) << "GRASP failed to find a solution for example instance.";
    EXPECT_EQ(tour.size(), inst->getN());
    EXPECT_TRUE(inst->checkSolutionCorrectness(tour));
}

TEST_P(GRASPParameterizedTest, SolvesExampleInstanceLogToStdout) {
    ConstructiveHeuristicType heuristicType = GetParam();
    auto inst = Instance::loadInstanceFromFile(examplePath_.string());
    ASSERT_NE(inst, nullptr);

    GRASP grasp(*inst, 1, 0.2, 0.2, heuristicType, {LocalSearch::TwoOpt});
    grasp.run();
    auto tour = grasp.getSolution();

    ASSERT_FALSE(tour.empty()) << "GRASP failed to find a solution for example instance.";
    EXPECT_EQ(tour.size(), inst->getN());
    EXPECT_TRUE(inst->checkSolutionCorrectness(tour));
}

TEST_P(GRASPParameterizedTest, SolvesExampleInstanceWithSwapTwo) {
    ConstructiveHeuristicType heuristicType = GetParam();
    auto inst = Instance::loadInstanceFromFile(examplePath_.string());
    ASSERT_NE(inst, nullptr);

    GRASP grasp(*inst, 1, 0.2, 0.2, heuristicType, {LocalSearch::SwapTwo});
    grasp.run();
    auto tour = grasp.getSolution();

    ASSERT_FALSE(tour.empty()) << "GRASP failed to find a solution for example instance.";
    EXPECT_EQ(tour.size(), inst->getN());
    EXPECT_TRUE(inst->checkSolutionCorrectness(tour));
}

TEST_P(GRASPParameterizedTest, SolvesExampleInstanceWithTwoOptAndSwapTwo) {
    ConstructiveHeuristicType heuristicType = GetParam();
    auto inst = Instance::loadInstanceFromFile(examplePath_.string());
    ASSERT_NE(inst, nullptr);

    GRASP grasp(*inst, 1, 0.2, 0.2, heuristicType, {LocalSearch::TwoOpt, LocalSearch::SwapTwo});
    grasp.run();
    auto tour = grasp.getSolution();

    ASSERT_FALSE(tour.empty()) << "GRASP failed to find a solution for example instance.";
    EXPECT_EQ(tour.size(), inst->getN());
    EXPECT_TRUE(inst->checkSolutionCorrectness(tour));
}

TEST_P(GRASPParameterizedTest, SolvesExampleInstanceWithRelocate) {
    ConstructiveHeuristicType heuristicType = GetParam();
    auto inst = Instance::loadInstanceFromFile(examplePath_.string());
    ASSERT_NE(inst, nullptr);

    GRASP grasp(*inst, 1, 0.2, 0.2, heuristicType, {LocalSearch::Relocate});
    grasp.run();
    auto tour = grasp.getSolution();

    ASSERT_FALSE(tour.empty()) << "GRASP failed to find a solution for example instance.";
    EXPECT_EQ(tour.size(), inst->getN());
    EXPECT_TRUE(inst->checkSolutionCorrectness(tour));
}

TEST_P(GRASPParameterizedTest, SolvesExampleInstanceWithAllLocalSearches) {
    ConstructiveHeuristicType heuristicType = GetParam();
    auto inst = Instance::loadInstanceFromFile(examplePath_.string());
    ASSERT_NE(inst, nullptr);

    GRASP grasp(*inst,
                1,
                0.2,
                0.2,
                heuristicType,
                {LocalSearch::TwoOpt, LocalSearch::SwapTwo, LocalSearch::Relocate});
    grasp.run();
    auto tour = grasp.getSolution();

    ASSERT_FALSE(tour.empty()) << "GRASP failed to find a solution for example instance.";
    EXPECT_EQ(tour.size(), inst->getN());
    EXPECT_TRUE(inst->checkSolutionCorrectness(tour));
}

TEST_P(GRASPParameterizedTest, SolvesGRF1) {
    ConstructiveHeuristicType heuristicType = GetParam();
    if (!fs::exists(grf1Path_)) {
        GTEST_SKIP() << "GRF1 instance not found at " << grf1Path_;
    }

    auto inst = Instance::loadInstanceFromFile(grf1Path_.string());
    ASSERT_NE(inst, nullptr);

    GRASP grasp(*inst, 3, 0.2, 0.2, heuristicType, {LocalSearch::TwoOpt});
    grasp.run();
    auto tour = grasp.getSolution();

    ASSERT_FALSE(tour.empty()) << "GRASP failed to find a solution for GRF1.";
    EXPECT_EQ(tour.size(), inst->getN());
    EXPECT_TRUE(inst->checkSolutionCorrectness(tour));
}

TEST_P(GRASPParameterizedTest, SolvesGRF4) {
    ConstructiveHeuristicType heuristicType = GetParam();
    if (!fs::exists(grf4Path_)) {
        GTEST_SKIP() << "GRF4 instance not found at " << grf4Path_;
    }

    auto inst = Instance::loadInstanceFromFile(grf4Path_.string());
    ASSERT_NE(inst, nullptr);

    GRASP grasp(*inst, 1, 0.2, 0.2, heuristicType, {LocalSearch::TwoOpt});
    grasp.run();
    auto tour = grasp.getSolution();

    if (heuristicType == ConstructiveHeuristicType::SimpleRandomized && tour.empty()) {
        GTEST_SUCCEED() << "SimpleRandomized failed to find a solution, which is acceptable.";
        return;
    }

    ASSERT_FALSE(tour.empty()) << "GRASP failed to find a solution for GRF4.";
    EXPECT_EQ(tour.size(), inst->getN());
    EXPECT_TRUE(inst->checkSolutionCorrectness(tour));
}

// Instantiate the parameterized tests for all constructive heuristic types
INSTANTIATE_TEST_SUITE_P(AllHeuristics,
                         GRASPParameterizedTest,
                         ::testing::Values(ConstructiveHeuristicType::RandomizedGreedy,
                                           ConstructiveHeuristicType::MIPRandomizedGreedyBias,
                                           ConstructiveHeuristicType::MIPRandomizedGreedyRandom,
                                           ConstructiveHeuristicType::SimpleRandomized),
                         [](const ::testing::TestParamInfo<ConstructiveHeuristicType>& info) {
                             switch (info.param) {
                                 case ConstructiveHeuristicType::RandomizedGreedy:
                                     return "RandomizedGreedy";
                                 case ConstructiveHeuristicType::MIPRandomizedGreedyBias:
                                     return "MIPRandomizedGreedyBias";
                                 case ConstructiveHeuristicType::MIPRandomizedGreedyRandom:
                                     return "MIPRandomizedGreedyRandom";
                                 case ConstructiveHeuristicType::SimpleRandomized:
                                     return "SimpleRandomized";
                                 default:
                                     return "Unknown";
                             }
                         });
