#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>

#include "../include/instance.hpp"

namespace fs = std::filesystem;

class InstanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Get the source directory path (two levels up from build directory)
        sourceDir_ = fs::current_path().parent_path();
        instancePath_ = sourceDir_ / "tests" / "instances" / "example_1.txt";
        solutionsDir_ = sourceDir_.parent_path() / "cpp_src" / "build" / "solutions";
    }

    void TearDown() override {
        // Clean up test files
        if (fs::exists(solutionsDir_ / "example_1.txt")) {
            fs::remove(solutionsDir_ / "example_1.txt");
        }
    }

    fs::path sourceDir_;
    fs::path instancePath_;
    fs::path solutionsDir_;
};

TEST_F(InstanceTest, LoadInstanceFromFile) {
    auto instance = Instance::loadInstanceFromFile(instancePath_.string());

    EXPECT_EQ(instance->getN(), 5);
    EXPECT_EQ(instance->getA(), 8);
    EXPECT_EQ(instance->getR(), 4);
    EXPECT_EQ(instance->getName(), "example_1.txt");
    EXPECT_EQ(instance->getEdges().size(), 8);
    EXPECT_EQ(instance->getRelations().size(), 4);
}

TEST_F(InstanceTest, SolutionCorrectness) {
    auto instance = Instance::loadInstanceFromFile(instancePath_.string());

    // Invalid because does not start at 0
    std::vector<int> invalidTour1 = {2, 1, 4, 3, 0};
    EXPECT_FALSE(instance->checkSolutionCorrectness(invalidTour1));

    // Invalid because of wrong length
    std::vector<int> invalidTour2 = {0, 2, 1, 4, 3, 2, 1};
    EXPECT_FALSE(instance->checkSolutionCorrectness(invalidTour2));

    // Invalid because of duplicate nodes
    std::vector<int> invalidTour3 = {0, 2, 1, 4, 3, 3};
    EXPECT_FALSE(instance->checkSolutionCorrectness(invalidTour3));

    // Invalid because of wrong node
    std::vector<int> invalidTour4 = {0, 2, 1, 4, 18};
    EXPECT_FALSE(instance->checkSolutionCorrectness(invalidTour4));
}

TEST_F(InstanceTest, WriteSolutionToFile) {
    auto instance = Instance::loadInstanceFromFile(instancePath_.string());

    // Test invalid tour
    std::vector<int> invalidTour = {2, 1, 4, 3};
    EXPECT_THROW(instance->saveSolution(invalidTour, 71.0), std::runtime_error);

    // Test valid tour
    std::vector<int> validTour = {0, 2, 1, 4, 3};
    instance->saveSolution(validTour, 71.0);

    // Verify file contents
    std::ifstream file(solutionsDir_ / "example_1.txt");
    std::string line;
    std::getline(file, line);
    EXPECT_TRUE(line.find("0,2,1,4,3 | 71 | ") == 0);
}

TEST_F(InstanceTest, MIPStartFromTsp1) {
    auto instance = Instance::loadInstanceFromFile(instancePath_.string());

    std::vector<int> tour;
    std::vector<int> partial_tour;
    std::vector<double> expected_partial_costs;

    // Case 1
    tour = {0, 2, 1, 4, 3, 0};
    partial_tour = {0};
    expected_partial_costs = {10, 25, 36, 56, 71};
    for (int i = 1; i < (int)tour.size(); ++i) {
        partial_tour.push_back(tour[i]);
        for (int j = 0; j < i; ++j) {  // j is the start index
            double cost = instance->computePartialTourCost(partial_tour, j);
            EXPECT_DOUBLE_EQ(cost,
                             j == 0
                                 ? expected_partial_costs[i - 1]
                                 : expected_partial_costs[i - 1] - expected_partial_costs[j - 1]);
        }
    }

    // Case 2
    tour = {0, 3, 2, 1, 4, 0};
    partial_tour = {0};
    expected_partial_costs = {10, 30, 45, 56, 62};
    for (int i = 1; i < (int)tour.size(); ++i) {
        partial_tour.push_back(tour[i]);
        for (int j = 0; j < i; ++j) {  // j is the start index
            double cost = instance->computePartialTourCost(partial_tour, j);
            EXPECT_DOUBLE_EQ(cost,
                             j == 0
                                 ? expected_partial_costs[i - 1]
                                 : expected_partial_costs[i - 1] - expected_partial_costs[j - 1]);
        }
    }
}

TEST_F(InstanceTest, TwoOptMethod) {
    /*
    Source: https://en.wikipedia.org/wiki/2-opt

    Here is an example of the above with arbitrary input:

    Example route: A → B → E → D → C → F → G → H → A
    Example parameters: v1=1, v2=4 (assuming starting index is 0)
    Contents of new_route by step:
    (A → B)
    A → B → (C → D → E)
    A → B → C → D → E → (F → G → H → A)
    */
    boost::unordered_map<std::pair<int, int>, double> edges = {{{0, 1}, 1.0},
                                                               {{1, 2}, 1.0},
                                                               {{2, 3}, 1.0},
                                                               {{3, 4}, 1.0},
                                                               {{4, 5}, 1.0},
                                                               {{5, 6}, 1.0},
                                                               {{6, 7}, 1.0},
                                                               {{7, 8}, 1.0}};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {};
    Instance inst(8, edges, relations, "test");

    std::vector<int> tour = {0, 1, 4, 3, 2, 5, 6, 7, 0};

    inst.get_two_opt_neighbor(tour, 1, 4);  // Apply 2-opt mutation

    // Check that the tour length remains the same
    EXPECT_EQ(tour.size(), 9);

    // check that the result is the expected one:
    std::vector<int> expected_tour = {0, 1, 2, 3, 4, 5, 6, 7, 0};
    EXPECT_EQ(tour, expected_tour);
}

TEST_F(InstanceTest, SwapTwoMethod) {
    boost::unordered_map<std::pair<int, int>, double> edges = {};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {};
    Instance inst(5, edges, relations, "test");

    std::vector<int> tour = {0, 1, 2, 3, 4};
    inst.get_swap_two_neighbor(tour, 1, 3);
    std::vector<int> expected_tour = {0, 3, 2, 1, 4};
    EXPECT_EQ(tour, expected_tour);
}

TEST_F(InstanceTest, RelocateMethod) {
    boost::unordered_map<std::pair<int, int>, double> edges = {};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {};
    Instance inst(5, edges, relations, "test");

    std::vector<int> tour = {0, 1, 2, 3, 4};
    inst.get_relocate_neighbor(tour, 1, 3);
    std::vector<int> expected_tour = {0, 2, 3, 1, 4};
    EXPECT_EQ(tour, expected_tour);
}

TEST_F(InstanceTest, AllTwoOptNeighbors) {
    auto instance = Instance::loadInstanceFromFile(instancePath_.string());
    int n = instance->getN();

    std::vector<int> original_tour = {0, 2, 1, 4, 3, 0};

    int count = 0;
    while (auto opt = instance->twoOptIteratorTracker_.next()) {
        std::vector<int> tour = original_tour;
        instance->get_two_opt_neighbor(tour, opt->first, opt->second);
        count++;
    }
    EXPECT_EQ(count, n * (n - 1) / 2);
}

TEST_F(InstanceTest, AllSwapTwoNeighbors) {
    auto instance = Instance::loadInstanceFromFile(instancePath_.string());
    int n = instance->getN();

    std::vector<int> original_tour = {0, 2, 1, 4, 3};

    int count = 0;
    instance->swapTwoIteratorTracker_.reset();
    while (auto opt = instance->swapTwoIteratorTracker_.next()) {
        std::vector<int> tour = original_tour;
        instance->get_swap_two_neighbor(tour, opt->first, opt->second);
        count++;
    }
    // n-1 nodes can be swapped (all but node 0)
    EXPECT_EQ(count, (n - 1) * (n - 2) / 2);
}

TEST_F(InstanceTest, AllRelocateNeighbors) {
    auto instance = Instance::loadInstanceFromFile(instancePath_.string());
    int n = instance->getN();

    std::vector<int> original_tour = {0, 2, 1, 4, 3};

    int count = 0;
    instance->relocateIteratorTracker_.reset();
    while (auto opt = instance->relocateIteratorTracker_.next()) {
        std::vector<int> tour = original_tour;
        instance->get_relocate_neighbor(tour, opt->first, opt->second);
        count++;
    }
    // n-1 nodes can be relocated to n-2 positions
    EXPECT_EQ(count, (n - 1) * (n - 2));
}

TEST_F(InstanceTest, TestSolutionCorrectness) {
    boost::unordered_map<std::pair<int, int>, double> edges = {
        {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{2, 0}, 1.0}};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations = {};
    Instance inst(3, edges, relations, "test");

    std::vector<int> tour = {0, 1, 2, 0};
    EXPECT_TRUE(inst.checkSolutionCorrectness(tour));
    tour.push_back(0);
    EXPECT_TRUE(inst.checkSolutionCorrectness(tour));

    tour = {0, 1, 2, 1};
    EXPECT_FALSE(inst.checkSolutionCorrectness(tour));
    tour.push_back(0);
    EXPECT_FALSE(inst.checkSolutionCorrectness(tour));

    tour = {0, 1, 2, 0, 1};
    EXPECT_FALSE(inst.checkSolutionCorrectness(tour));
    tour.push_back(0);
    EXPECT_FALSE(inst.checkSolutionCorrectness(tour));
}