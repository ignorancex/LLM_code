#include <gtest/gtest.h>

#include <filesystem>

#include "instance.hpp"
#include "tsp_model.hpp"

namespace fs = std::filesystem;

class TSPModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Get the source directory path (two levels up from build directory)
        sourceDir_ = fs::current_path().parent_path();
        examplePath_ = sourceDir_ / "tests" / "instances" / "example_1.txt";
        grf1Path_ = sourceDir_.parent_path() / "instances" / "instances_release_1" / "grf1.txt";
        grf4Path_ = sourceDir_.parent_path() / "instances" / "instances_release_1" / "grf4.txt";
    }

    fs::path sourceDir_;
    fs::path examplePath_;
    fs::path grf1Path_;
    fs::path grf4Path_;
};

TEST_F(TSPModelTest, DummyModel) {
    // Create a simple 3-node instance
    boost::unordered_map<std::pair<int, int>, double> edges = {
        {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{2, 0}, 1.0}};
    boost::unordered_map<std::tuple<int, int, int, int>, double> relations;
    Instance inst(3, edges, relations, "test");
    GurobiTSPModel model(inst);
    model.formulate();
    model.solveToFeasibleSolution();

    auto tour = model.getBestTour();
    std::vector<int> expected = {0, 1, 2};
    EXPECT_EQ(tour, expected);
}

TEST_F(TSPModelTest, ExampleInstance) {
    // Load example instance
    auto inst = Instance::loadInstanceFromFile(examplePath_.string());

    GurobiTSPModel model(*inst);
    model.formulate();
    model.solveToFeasibleSolution();

    auto tour = model.getBestTour();

    // Check that the path is feasible
    std::vector<std::pair<int, int>> edges_used;
    for (size_t i = 0; i < tour.size() - 1; ++i) {
        edges_used.emplace_back(tour[i], tour[i + 1]);
    }
    edges_used.emplace_back(tour.back(), tour.front());

    for (const auto& edge : edges_used) {
        EXPECT_TRUE(inst->getEdges().contains(edge));
    }
}

TEST_F(TSPModelTest, BigModel) {
    // Load a larger instance
    auto inst = Instance::loadInstanceFromFile(grf1Path_.string());

    GurobiTSPModel model(*inst);
    model.formulate();
    model.solveToOptimality(std::nullopt, false);

    auto tour = model.getBestTour();
    EXPECT_EQ(tour.size(), inst->getN());
}

TEST_F(TSPModelTest, GetBestNTours) {
    // Load instance
    auto inst = Instance::loadInstanceFromFile(grf4Path_.string());

    GurobiTSPModel model(*inst);
    model.formulate();
    model.solveToOptimality(std::nullopt, false);

    auto best_tour = model.getBestTour();
    auto tours = model.getBestNTours(5);

    EXPECT_EQ(tours.size(), 5);
    EXPECT_EQ(best_tour, tours[0]);

    for (const auto& tour : tours) {
        EXPECT_EQ(tour.size(), inst->getN());
    }

    // Check that all tours are different
    for (size_t i = 1; i < tours.size(); ++i) {
        EXPECT_NE(tours[0], tours[i]);
    }
}
