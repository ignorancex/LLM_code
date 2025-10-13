#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <set>

#include "instance.hpp"
#include "mip_randomized_construction.hpp"

namespace fs = std::filesystem;

class MIPRandomizedConstructionTest : public ::testing::Test {
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
        example2Path_ = currentPath / "instances" / "examples" / "example_2.txt";
        grf1Path_ = currentPath / "instances" / "instances_release_1" / "grf1.txt";
        grf4Path_ = currentPath / "instances" / "instances_release_1" / "grf4.txt";
    }

    fs::path cppSourceDir_;
    fs::path examplePath_;
    fs::path example2Path_;
    fs::path grf1Path_;
    fs::path grf4Path_;
};

// Parameterized test class for testing both construction types
class MIPRandomizedConstructionParameterizedTest
    : public MIPRandomizedConstructionTest,
      public ::testing::WithParamInterface<ConstructionType> {};

TEST_F(MIPRandomizedConstructionTest, ComputeNodeDistances1) {
    boost::unordered_map<std::pair<int, int>, double> edges = {
        {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{2, 0}, 1.0}};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {};
    Instance inst(3, edges, relations, "test");

    MIPRandomizedConstruction mipRC(inst, 1, ConstructionType::Bias);
    std::vector<int> nodePriorities = {0, 1, 2};
    auto nodeDist = mipRC.computeNodeDist(nodePriorities);

    std::set<double> distValues;
    for (const auto& [key, value] : nodeDist) {
        distValues.insert(value);
    }

    EXPECT_EQ(distValues.size(), 1);
    EXPECT_EQ(*distValues.begin(), 1.0);
}

TEST_F(MIPRandomizedConstructionTest, ComputeNodeDistances2) {
    boost::unordered_map<std::pair<int, int>, double> edges = {
        {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{2, 3}, 1.0}, {{3, 4}, 1.0}, {{4, 0}, 1.0}};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {};
    Instance inst(5, edges, relations, "test");

    MIPRandomizedConstruction mipRC(inst, 1, ConstructionType::Bias);
    std::vector<int> nodePriorities = {0, 1, 2, 3, 4};
    auto nodeDist = mipRC.computeNodeDist(nodePriorities);

    std::set<double> distValues;
    for (const auto& [key, value] : nodeDist) {
        distValues.insert(value);
    }

    EXPECT_EQ(distValues.size(), 2);
    EXPECT_TRUE(distValues.count(1.0));
    EXPECT_TRUE(distValues.count(2.0));
    EXPECT_DOUBLE_EQ(nodeDist[std::make_pair(0, 4)], 1.0);
    EXPECT_DOUBLE_EQ(nodeDist[std::make_pair(1, 4)], 2.0);
}

TEST_F(MIPRandomizedConstructionTest, TSPSearchSolutionFindsOnlyFeasible) {
    boost::unordered_map<std::pair<int, int>, double> edges = {
        {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{2, 0}, 1.0}};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {};
    Instance inst(3, edges, relations, "test");

    MIPRandomizedConstruction mipRC(inst, 1, ConstructionType::Bias);
    TSPPrior tspPrior({0, 2, 1}, 0.5, 0.5);

    auto [tour, cost] = mipRC.evaluateIndividual(tspPrior, 5);

    // There is only one feasible tour
    EXPECT_EQ(tour, std::vector<int>({0, 1, 2}));
    EXPECT_DOUBLE_EQ(cost, 3.0);
}

TEST_F(MIPRandomizedConstructionTest, GetEdgesForTSPSearch1) {
    boost::unordered_map<std::pair<int, int>, double> edges = {
        {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{2, 0}, 1.0}};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {{{0, 1, 2, 0}, 0.0}};
    Instance inst(3, edges, relations, "test");

    MIPRandomizedConstruction mipRC(inst, 1, ConstructionType::Bias);
    TSPPrior tspPrior({0, 1, 2}, 0.5, 0.5);
    auto modifiedEdges = mipRC.getEdgesForTSPSearch(tspPrior);

    EXPECT_LT(modifiedEdges[std::make_pair(0, 1)], modifiedEdges[std::make_pair(1, 2)]);
    EXPECT_LT(modifiedEdges[std::make_pair(2, 0)], modifiedEdges[std::make_pair(1, 2)]);
}

TEST_F(MIPRandomizedConstructionTest, GetEdgesForTSPSearch2) {
    boost::unordered_map<std::pair<int, int>, double> edges = {{{0, 1}, 1.0},
                                                               {{0, 2}, 1.0},
                                                               {{1, 3}, 1.0},
                                                               {{2, 3}, 1.0},
                                                               {{3, 4}, 1.0},
                                                               {{1, 2}, 1.0},
                                                               {{2, 1}, 1.0},
                                                               {{4, 0}, 1.0}};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {{{0, 1, 3, 4}, -1.0},
                                                                              {{0, 2, 3, 4}, -1.0}};
    Instance inst(5, edges, relations, "test");

    MIPRandomizedConstruction mipRC(inst, 1, ConstructionType::Bias);
    std::vector<int> tour1 = {0, 1, 2, 3, 4};
    std::vector<int> tour2 = {0, 2, 1, 3, 4};

    auto dist1 = mipRC.computeNodeDist(tour1);
    auto dist2 = mipRC.computeNodeDist(tour2);

    std::set<double> distValues1, distValues2;
    for (const auto& [key, value] : dist1) {
        distValues1.insert(value);
    }
    for (const auto& [key, value] : dist2) {
        distValues2.insert(value);
    }

    EXPECT_EQ(distValues1, distValues2);
    EXPECT_DOUBLE_EQ(dist1[std::make_pair(0, 1)], dist1[std::make_pair(1, 2)]);
    EXPECT_DOUBLE_EQ(dist1[std::make_pair(1, 2)], dist1[std::make_pair(2, 3)]);
    EXPECT_DOUBLE_EQ(dist1[std::make_pair(2, 3)], dist1[std::make_pair(3, 4)]);
    EXPECT_DOUBLE_EQ(dist2[std::make_pair(0, 2)], dist2[std::make_pair(2, 1)]);
    EXPECT_DOUBLE_EQ(dist2[std::make_pair(2, 1)], dist2[std::make_pair(1, 3)]);
    EXPECT_DOUBLE_EQ(dist2[std::make_pair(1, 3)], dist2[std::make_pair(3, 4)]);
    EXPECT_DOUBLE_EQ(dist1[std::make_pair(1, 3)], dist2[std::make_pair(2, 3)]);
    EXPECT_DOUBLE_EQ(dist1[std::make_pair(1, 3)], 2.0);

    TSPPrior tspPrior1(tour1, 0.5, 0.5);
    TSPPrior tspPrior2(tour2, 0.5, 0.5);
    auto edges1 = mipRC.getEdgesForTSPSearch(tspPrior1);
    auto edges2 = mipRC.getEdgesForTSPSearch(tspPrior2);

    EXPECT_LT(edges1[std::make_pair(0, 1)], edges1[std::make_pair(0, 2)]);
    EXPECT_LT(edges2[std::make_pair(0, 2)], edges2[std::make_pair(0, 1)]);
    EXPECT_NE(edges1[std::make_pair(0, 1)], edges2[std::make_pair(0, 1)]);
}

TEST_F(MIPRandomizedConstructionTest, TSPSearch2) {
    boost::unordered_map<std::pair<int, int>, double> edges = {{{0, 1}, 1.0},
                                                               {{0, 2}, 1.0},
                                                               {{1, 3}, 1.0},
                                                               {{2, 3}, 1.0},
                                                               {{3, 4}, 1.0},
                                                               {{1, 2}, 1.0},
                                                               {{2, 1}, 1.0},
                                                               {{4, 0}, 1.0}};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {{{0, 1, 3, 4}, -1.0},
                                                                              {{0, 2, 3, 4}, -1.0}};
    Instance inst(5, edges, relations, "test");

    MIPRandomizedConstruction mipRC(inst, 1, ConstructionType::Bias);
    std::vector<int> tour1 = {0, 1, 2, 3, 4};
    std::vector<int> tour2 = {0, 2, 1, 3, 4};

    TSPPrior tspPrior1(tour1, 1.0, 0.5);
    TSPPrior tspPrior2(tour2, 1.0, 0.5);

    auto edges1 = mipRC.getEdgesForTSPSearch(tspPrior1);
    auto edges2 = mipRC.getEdgesForTSPSearch(tspPrior2);

    EXPECT_DOUBLE_EQ(edges1[std::make_pair(0, 1)], edges2[std::make_pair(0, 2)]);
    EXPECT_DOUBLE_EQ(edges2[std::make_pair(0, 1)], edges1[std::make_pair(0, 2)]);

    auto [compTour1, cost1] = mipRC.evaluateIndividual(tspPrior1, 5);
    auto [compTour2, cost2] = mipRC.evaluateIndividual(tspPrior2, 5);

    EXPECT_EQ(compTour1, tour1);
    EXPECT_EQ(compTour2, tour2);
}

TEST_F(MIPRandomizedConstructionTest, GenerateRandomPermutation) {
    boost::unordered_map<std::pair<int, int>, double> edges = {
        {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{2, 3}, 1.0}, {{3, 4}, 1.0}, {{4, 0}, 1.0}};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {};
    Instance inst(5, edges, relations, "test");

    MIPRandomizedConstruction mipRC(inst, 1, ConstructionType::Bias);

    // Test multiple random permutations
    std::map<std::pair<int, int>, int> positionCounts;  // (element, position) -> count

    for (int trial = 0; trial < 100; ++trial) {
        auto perm = mipRC.generateRandomPermutation();
        EXPECT_EQ(perm.size(), 5);
        EXPECT_EQ(perm[0], 0);  // First element should always be 0

        std::set<int> uniqueElements(perm.begin(), perm.end());
        EXPECT_EQ(uniqueElements.size(), 5);  // All elements should be unique

        for (int pos = 0; pos < 5; ++pos) {
            positionCounts[std::make_pair(perm[pos], pos)]++;
        }
    }

    // Check that elements are somewhat evenly distributed across positions
    // (This is a basic randomness check)
    for (int elem = 1; elem < 5; ++elem) {  // Skip element 0 since it's always at position 0
        for (int pos = 1; pos < 5; ++pos) {
            EXPECT_GT(positionCounts[std::make_pair(elem, pos)], 0);
        }
    }
}

TEST_F(MIPRandomizedConstructionTest, BestAmongMultipleTSPFeasibleSolutionsIsSelected) {
    if (!fs::exists(grf4Path_)) {
        GTEST_SKIP() << "GRF4 instance not found at " << grf4Path_;
    }

    auto inst = Instance::loadInstanceFromFile(grf4Path_.string());
    ASSERT_NE(inst, nullptr);

    MIPRandomizedConstruction mipRC(*inst, 10, ConstructionType::Bias);

    // Create a TSP prior with sequential priorities
    std::vector<int> priorities;
    for (int i = 0; i < inst->getN(); ++i) {
        priorities.push_back(i);
    }

    TSPPrior tspPrior(priorities, 0.0, 0.5);
    auto [bestTour, bestCost] = mipRC.evaluateIndividual(tspPrior, 10);

    EXPECT_FALSE(bestTour.empty());
    EXPECT_EQ(bestTour.size(), inst->getN());
    EXPECT_TRUE(inst->checkSolutionCorrectness(bestTour));
    EXPECT_DOUBLE_EQ(bestCost, inst->computeObjective(bestTour));
}

TEST_F(MIPRandomizedConstructionTest, ApplyAlphaRandomizationToEdges) {
    boost::unordered_map<std::pair<int, int>, double> edges = {
        {{0, 1}, 1.0}, {{1, 2}, 2.0}, {{2, 0}, 3.0}};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {};
    Instance inst(3, edges, relations, "test");

    MIPRandomizedConstruction mipRC(inst, 1, ConstructionType::Random);

    // Test with alpha = 0 (should not change edges)
    auto randomizedEdges = mipRC.applyAlphaRandomizationToEdges(0.0);
    EXPECT_DOUBLE_EQ(randomizedEdges[std::make_pair(0, 1)], 1.0);
    EXPECT_DOUBLE_EQ(randomizedEdges[std::make_pair(1, 2)], 2.0);
    EXPECT_DOUBLE_EQ(randomizedEdges[std::make_pair(2, 0)], 3.0);

    // Test with alpha > 0 (should change edges)
    randomizedEdges = mipRC.applyAlphaRandomizationToEdges(1.0);
    EXPECT_NE(randomizedEdges[std::make_pair(0, 1)], 1.0);
    EXPECT_NE(randomizedEdges[std::make_pair(1, 2)], 2.0);
    EXPECT_NE(randomizedEdges[std::make_pair(2, 0)], 3.0);
}

TEST_F(MIPRandomizedConstructionTest, OptionalAlphaParameterBias) {
    boost::unordered_map<std::pair<int, int>, double> edges = {
        {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{2, 0}, 1.0}};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {};
    Instance inst(3, edges, relations, "test");

    // Test with specific alpha value for Bias type
    MIPRandomizedConstruction mipRC(inst, 1, ConstructionType::Bias, 0.5);

    auto permutation = mipRC.generateRandomPermutation();
    TSPPrior tspPrior(permutation, 0.5, 0.5);  // Expected alpha should be 0.5

    // The constructor should use the provided alpha value
    // We can't directly test the stored alpha_ value since it's private,
    // but we can verify the functionality works
    EXPECT_NO_THROW(mipRC.run());
    auto tour = mipRC.getSolution();
    EXPECT_FALSE(tour.empty());
}

TEST_F(MIPRandomizedConstructionTest, OptionalAlphaParameterRandom) {
    boost::unordered_map<std::pair<int, int>, double> edges = {
        {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{2, 0}, 1.0}};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {};
    Instance inst(3, edges, relations, "test");

    // Test with specific alpha value for Random type
    MIPRandomizedConstruction mipRC(inst, 1, ConstructionType::Random, 0.1);

    // The constructor should use the provided alpha value
    EXPECT_NO_THROW(mipRC.run());
    auto tour = mipRC.getSolution();
    EXPECT_FALSE(tour.empty());
}

TEST_F(MIPRandomizedConstructionTest, OptionalAlphaAndBetaParametersBias) {
    boost::unordered_map<std::pair<int, int>, double> edges = {
        {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{2, 0}, 1.0}};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {};
    Instance inst(3, edges, relations, "test");

    // Test with specific alpha and beta values for Bias type
    MIPRandomizedConstruction mipRC(inst, 1, ConstructionType::Bias, 0.5, 1.0);

    EXPECT_NO_THROW(mipRC.run());
    auto tour = mipRC.getSolution();
    EXPECT_FALSE(tour.empty());
}

TEST_F(MIPRandomizedConstructionTest, DefaultParametersStillWork) {
    boost::unordered_map<std::pair<int, int>, double> edges = {
        {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{2, 0}, 1.0}};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {};
    Instance inst(3, edges, relations, "test");

    // Test that default parameters still work (no alpha/beta specified)
    MIPRandomizedConstruction mipRC1(inst, 1, ConstructionType::Bias);
    MIPRandomizedConstruction mipRC2(inst, 1, ConstructionType::Random);

    EXPECT_NO_THROW(mipRC1.run());
    EXPECT_NO_THROW(mipRC2.run());

    auto tour1 = mipRC1.getSolution();
    auto tour2 = mipRC2.getSolution();

    EXPECT_FALSE(tour1.empty());
    EXPECT_FALSE(tour2.empty());
}

TEST_F(MIPRandomizedConstructionTest, SolveRandomizedTSP) {
    boost::unordered_map<std::pair<int, int>, double> edges = {
        {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{2, 0}, 1.0}};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {};
    Instance inst(3, edges, relations, "test");

    MIPRandomizedConstruction mipRC(inst, 1, ConstructionType::Random);

    auto [tour, cost] = mipRC.solveRandomizedTSP(0.1, std::nullopt, 5);

    EXPECT_FALSE(tour.empty());
    EXPECT_EQ(tour.size(), 3);
    EXPECT_TRUE(inst.checkSolutionCorrectness(tour));
    EXPECT_GT(cost, 0.0);
}

TEST_F(MIPRandomizedConstructionTest, SolveRandomizedTSPWithBeta) {
    boost::unordered_map<std::pair<int, int>, double> edges = {
        {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{2, 0}, 1.0}};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {};
    Instance inst(3, edges, relations, "test");

    MIPRandomizedConstruction mipRC(inst, 1, ConstructionType::Random);

    auto [tour, cost] = mipRC.solveRandomizedTSP(std::nullopt, 0.5, 5);

    EXPECT_FALSE(tour.empty());
    EXPECT_EQ(tour.size(), 3);
    EXPECT_TRUE(inst.checkSolutionCorrectness(tour));
    EXPECT_GT(cost, 0.0);
}

TEST_F(MIPRandomizedConstructionTest, SolveRandomizedTSPErrorBothAlphaAndBeta) {
    boost::unordered_map<std::pair<int, int>, double> edges = {
        {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{2, 0}, 1.0}};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {};
    Instance inst(3, edges, relations, "test");

    MIPRandomizedConstruction mipRC(inst, 1, ConstructionType::Random);

    EXPECT_THROW(mipRC.solveRandomizedTSP(0.1, 0.5, 5), std::invalid_argument);
}

TEST_F(MIPRandomizedConstructionTest, SolveRandomizedTSPErrorNeitherAlphaNorBeta) {
    boost::unordered_map<std::pair<int, int>, double> edges = {
        {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{2, 0}, 1.0}};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {};
    Instance inst(3, edges, relations, "test");

    MIPRandomizedConstruction mipRC(inst, 1, ConstructionType::Random);

    EXPECT_THROW(mipRC.solveRandomizedTSP(std::nullopt, std::nullopt, 5), std::invalid_argument);
}

TEST_P(MIPRandomizedConstructionParameterizedTest, SolvesExampleInstance) {
    ConstructionType type = GetParam();
    auto inst = Instance::loadInstanceFromFile(examplePath_.string());
    ASSERT_NE(inst, nullptr);

    MIPRandomizedConstruction mipRC(*inst, 5, type);
    mipRC.run();
    auto tour = mipRC.getSolution();

    ASSERT_FALSE(tour.empty()) << "MIP randomized construction failed to find a solution for type "
                               << (type == ConstructionType::Bias ? "Bias" : "Random");
    EXPECT_EQ(tour.size(), inst->getN());
    EXPECT_TRUE(inst->checkSolutionCorrectness(tour));
}

TEST_P(MIPRandomizedConstructionParameterizedTest, ReturnsValidCost) {
    ConstructionType type = GetParam();
    auto inst = Instance::loadInstanceFromFile(examplePath_.string());
    ASSERT_NE(inst, nullptr);

    MIPRandomizedConstruction mipRC(*inst, 5, type);
    mipRC.run();
    auto tour = mipRC.getSolution();

    if (!tour.empty()) {
        double cost = inst->computeObjective(tour);
        EXPECT_GT(cost, 0.0);  // Cost should be positive
    }
}

TEST_P(MIPRandomizedConstructionParameterizedTest, SolvesGRF1) {
    ConstructionType type = GetParam();
    if (!fs::exists(grf1Path_)) {
        GTEST_SKIP() << "GRF1 instance not found at " << grf1Path_;
    }

    auto inst = Instance::loadInstanceFromFile(grf1Path_.string());
    ASSERT_NE(inst, nullptr);

    MIPRandomizedConstruction mipRC(*inst, 5, type);
    mipRC.run();
    auto tour = mipRC.getSolution();

    ASSERT_FALSE(tour.empty())
        << "MIP randomized construction failed to find a solution for GRF1 with type "
        << (type == ConstructionType::Bias ? "Bias" : "Random");
    EXPECT_EQ(tour.size(), inst->getN());
    EXPECT_TRUE(inst->checkSolutionCorrectness(tour));
}

TEST_P(MIPRandomizedConstructionParameterizedTest, SolvesGRF4) {
    ConstructionType type = GetParam();
    if (!fs::exists(grf4Path_)) {
        GTEST_SKIP() << "GRF4 instance not found at " << grf4Path_;
    }

    auto inst = Instance::loadInstanceFromFile(grf4Path_.string());
    ASSERT_NE(inst, nullptr);

    MIPRandomizedConstruction mipRC(*inst, 5, type);
    mipRC.run();
    auto tour = mipRC.getSolution();

    ASSERT_FALSE(tour.empty())
        << "MIP randomized construction failed to find a solution for GRF4 with type "
        << (type == ConstructionType::Bias ? "Bias" : "Random");
    EXPECT_EQ(tour.size(), inst->getN());
    EXPECT_TRUE(inst->checkSolutionCorrectness(tour));
}

// Instantiate the parameterized tests for both construction types
INSTANTIATE_TEST_SUITE_P(BothConstructionTypes,
                         MIPRandomizedConstructionParameterizedTest,
                         ::testing::Values(ConstructionType::Bias, ConstructionType::Random),
                         [](const ::testing::TestParamInfo<ConstructionType>& info) {
                             return info.param == ConstructionType::Bias ? "Bias" : "Random";
                         });