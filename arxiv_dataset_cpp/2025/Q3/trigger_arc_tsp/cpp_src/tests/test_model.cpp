#include <gtest/gtest.h>
#include <gurobi_c++.h>

#include <boost/unordered_map.hpp>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "instance.hpp"
#include "model.hpp"

// Test solving a non-formulated model
TEST(GurobiModelTest, SolvingNonFormulatedModel) {
    int N = 3;
    boost::unordered_map<std::pair<int, int>, double> edges;
    edges[{0, 1}] = 1;
    edges[{1, 2}] = 1;
    edges[{2, 0}] = 1;

    boost::unordered_map<std::tuple<int, int, int, int>, double> relations;
    Instance instance(N, edges, relations, "test");
    GurobiModel model(instance);

    EXPECT_THROW(model.solveModelWithParameters(), std::runtime_error);
}

// Test with a specific sample instance
TEST(GurobiModelTest, Sample1) {
    int N = 3;
    boost::unordered_map<std::pair<int, int>, double> edges;
    edges[{0, 1}] = 1;
    edges[{1, 2}] = 1;
    edges[{2, 0}] = 1;

    boost::unordered_map<std::tuple<int, int, int, int>, double> relations;
    Instance instance(N, edges, relations, "test");
    GurobiModel model(instance);
    model.formulate();
    model.solveModelWithParameters();

    EXPECT_EQ(model.getModel().get(GRB_IntAttr_Status), GRB_OPTIMAL);
    EXPECT_DOUBLE_EQ(model.getModel().get(GRB_DoubleAttr_ObjVal), 3.0);
    auto x = model.getX();
    EXPECT_DOUBLE_EQ(x.at({0, 1}).get(GRB_DoubleAttr_X), 1.0);
    EXPECT_DOUBLE_EQ(x.at({1, 2}).get(GRB_DoubleAttr_X), 1.0);
    EXPECT_DOUBLE_EQ(x.at({2, 0}).get(GRB_DoubleAttr_X), 1.0);

    std::vector<int> tour = {0, 1, 2, 0};
    EXPECT_DOUBLE_EQ(instance.computeObjective(tour), 3.0);
}

// Test with another sample instance
TEST(GurobiModelTest, Sample2) {
    int N = 4;
    boost::unordered_map<std::pair<int, int>, double> edges;
    edges[{0, 1}] = 1;
    edges[{1, 2}] = 1;
    edges[{2, 3}] = 1;
    edges[{3, 0}] = 1;

    boost::unordered_map<std::tuple<int, int, int, int>, double> relations;
    relations[{0, 1, 1, 2}] = 2;

    Instance instance(N, edges, relations, "test");
    GurobiModel model(instance);
    model.formulate();
    model.solveModelWithParameters();

    EXPECT_EQ(model.getModel().get(GRB_IntAttr_Status), GRB_OPTIMAL);
    auto x = model.getX();
    EXPECT_DOUBLE_EQ(x.at({0, 1}).get(GRB_DoubleAttr_X), 1.0);
    EXPECT_DOUBLE_EQ(x.at({1, 2}).get(GRB_DoubleAttr_X), 1.0);
    EXPECT_DOUBLE_EQ(x.at({2, 3}).get(GRB_DoubleAttr_X), 1.0);
    EXPECT_DOUBLE_EQ(x.at({3, 0}).get(GRB_DoubleAttr_X), 1.0);

    auto y = model.getY();
    EXPECT_DOUBLE_EQ(y.at({0, 1, 1, 2}).get(GRB_DoubleAttr_X), 1.0);

    auto [tour, cost] = model.getSolutionAndCost();
    EXPECT_DOUBLE_EQ(cost, 5.0);
    EXPECT_EQ(tour, (std::vector<int>{0, 1, 2, 3}));

    std::vector<int> full_tour = {0, 1, 2, 3, 0};
    EXPECT_DOUBLE_EQ(instance.computeObjective(full_tour), 5.0);
}

TEST(GurobiModelTest, Sample3) {
    int N = 4;
    boost::unordered_map<std::pair<int, int>, double> edges;
    edges[{0, 1}] = 1;
    edges[{1, 2}] = 1;
    edges[{2, 3}] = 1;
    edges[{3, 0}] = 1;

    boost::unordered_map<std::tuple<int, int, int, int>, double> relations;
    relations[{0, 1, 3, 0}] = 5;
    relations[{1, 2, 3, 0}] = 10;

    Instance instance(N, edges, relations, "test");
    GurobiModel model(instance);
    model.formulate();
    model.solveModelWithParameters();

    EXPECT_EQ(model.getModel().get(GRB_IntAttr_Status), GRB_OPTIMAL);
    auto [tour, cost] = model.getSolutionAndCost();
    EXPECT_EQ(tour, (std::vector<int>{0, 1, 2, 3}));
    EXPECT_DOUBLE_EQ(cost, 13.0);

    std::vector<int> full_tour = {0, 1, 2, 3, 0};
    EXPECT_DOUBLE_EQ(instance.computeObjective(full_tour), 13.0);
}

TEST(GurobiModelTest, Sample4) {
    int N = 4;
    boost::unordered_map<std::pair<int, int>, double> edges;
    edges[{0, 1}] = 1;
    edges[{1, 2}] = 1;
    edges[{2, 3}] = 1;
    edges[{3, 0}] = 1;

    boost::unordered_map<std::tuple<int, int, int, int>, double> relations;
    relations[{0, 1, 2, 3}] = 2;
    relations[{1, 2, 2, 3}] = 10;
    relations[{3, 0, 2, 3}] = 2;

    Instance instance(N, edges, relations, "test");
    GurobiModel model(instance);
    model.formulate();
    model.solveModelWithParameters();

    EXPECT_EQ(model.getModel().get(GRB_IntAttr_Status), GRB_OPTIMAL);
    auto [tour, cost] = model.getSolutionAndCost();
    EXPECT_EQ(tour, (std::vector<int>{0, 1, 2, 3}));
    EXPECT_DOUBLE_EQ(cost, 13.0);

    std::vector<int> full_tour = {0, 1, 2, 3, 0};
    EXPECT_DOUBLE_EQ(instance.computeObjective(full_tour), 13.0);
}

TEST(GurobiModelTest, Sample5) {
    int N = 2;
    boost::unordered_map<std::pair<int, int>, double> edges;
    edges[{0, 1}] = 1;
    edges[{1, 0}] = 1;

    boost::unordered_map<std::tuple<int, int, int, int>, double> relations;
    relations[{0, 1, 1, 0}] = 2;

    Instance instance(N, edges, relations, "test");
    GurobiModel model(instance);
    model.formulate();
    model.solveModelWithParameters();

    EXPECT_EQ(model.getModel().get(GRB_IntAttr_Status), GRB_OPTIMAL);
    auto [tour, cost] = model.getSolutionAndCost();
    EXPECT_EQ(tour, (std::vector<int>{0, 1}));
    EXPECT_DOUBLE_EQ(cost, 3.0);

    std::vector<int> full_tour = {0, 1, 0};
    EXPECT_DOUBLE_EQ(instance.computeObjective(full_tour), 3.0);
}

TEST(GurobiModelTest, Sample9) {
    int N = 5;
    boost::unordered_map<std::pair<int, int>, double> edges;
    // Add edges with cost 1
    edges[{0, 1}] = 1;
    edges[{1, 2}] = 1;
    edges[{2, 3}] = 1;
    edges[{3, 4}] = 1;
    edges[{4, 0}] = 1;
    // Add edges with cost 2
    edges[{0, 4}] = 2;
    edges[{4, 3}] = 2;
    edges[{3, 2}] = 2;
    edges[{2, 1}] = 2;
    edges[{1, 0}] = 2;

    boost::unordered_map<std::tuple<int, int, int, int>, double> relations;

    Instance instance(N, edges, relations, "test");
    GurobiModel model(instance);
    model.formulate();
    model.solveModelWithParameters();

    EXPECT_EQ(model.getModel().get(GRB_IntAttr_Status), GRB_OPTIMAL);
    auto [tour, cost] = model.getSolutionAndCost();
    EXPECT_EQ(tour, (std::vector<int>{0, 1, 2, 3, 4}));
    EXPECT_DOUBLE_EQ(cost, 5.0);

    std::vector<int> full_tour = {0, 1, 2, 3, 4, 0};
    EXPECT_DOUBLE_EQ(instance.computeObjective(full_tour), 5.0);
}

TEST(GurobiModelTest, Sample10) {
    // Get the source directory path (two levels up from build directory)
    auto sourceDir = std::filesystem::current_path().parent_path();
    auto instancePath = sourceDir / "tests" / "instances" / "example_1.txt";

    auto instance = Instance::loadInstanceFromFile(instancePath.string());
    GurobiModel model(*instance);
    model.formulate();
    model.solveModelWithParameters();

    EXPECT_EQ(model.getModel().get(GRB_IntAttr_Status), GRB_OPTIMAL);
    auto [tour, cost] = model.getSolutionAndCost();
    EXPECT_EQ(tour, (std::vector<int>{0, 3, 2, 1, 4}));
    EXPECT_DOUBLE_EQ(cost, 62.0);

    std::vector<int> full_tour = {0, 3, 2, 1, 4, 0};
    EXPECT_DOUBLE_EQ(instance->computeObjective(full_tour), 62.0);
}

TEST(GurobiModelTest, Sample11) {
    int N = 10;
    boost::unordered_map<std::pair<int, int>, double> edges;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                edges[{i, j}] = 1;
            }
        }
    }

    boost::unordered_map<std::tuple<int, int, int, int>, double> relations;
    relations[{0, 1, 1, 2}] = 0;

    Instance instance(N, edges, relations, "test");
    GurobiModel model(instance);
    model.formulate();
    model.solveModelWithParameters();

    EXPECT_EQ(model.getModel().get(GRB_IntAttr_Status), GRB_OPTIMAL);
    auto [tour, cost] = model.getSolutionAndCost();
    EXPECT_DOUBLE_EQ(cost, 9.0);

    auto x = model.getX();
    EXPECT_DOUBLE_EQ(x.at({0, 1}).get(GRB_DoubleAttr_X), 1.0);
    EXPECT_DOUBLE_EQ(x.at({1, 2}).get(GRB_DoubleAttr_X), 1.0);
}

TEST(GurobiModelTest, Sample12) {
    int N = 3;
    boost::unordered_map<std::pair<int, int>, double> edges;
    edges[{0, 1}] = 1;
    edges[{1, 0}] = 1;
    edges[{0, 2}] = 1;
    edges[{2, 0}] = 1;

    boost::unordered_map<std::tuple<int, int, int, int>, double> relations;
    relations[{0, 1, 1, 0}] = 0;
    relations[{0, 2, 1, 0}] = 0;

    Instance instance(N, edges, relations, "test");
    GurobiModel model(instance);
    model.formulate();

    EXPECT_THROW(model.solveModelWithParameters(), std::runtime_error);
    EXPECT_EQ(model.getModel().get(GRB_IntAttr_Status), GRB_INFEASIBLE);
}

TEST(GurobiModelTest, Sample13) {
    int N = 10;
    boost::unordered_map<std::pair<int, int>, double> edges;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                edges[{i, j}] = 1;
            }
        }
    }

    boost::unordered_map<std::tuple<int, int, int, int>, double> relations;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i != j && (i != 4 || j != 5)) {
                relations[{4, 5, i, j}] = 0;
            }
        }
    }

    Instance instance(N, edges, relations, "test");
    GurobiModel model(instance);
    model.formulate();
    model.solveModelWithParameters();

    EXPECT_EQ(model.getModel().get(GRB_IntAttr_Status), GRB_OPTIMAL);
    auto [tour, cost] = model.getSolutionAndCost();
    EXPECT_DOUBLE_EQ(cost, 2.0);
    EXPECT_EQ(tour[0], 0);
    EXPECT_EQ(tour[1], 4);
    EXPECT_EQ(tour[2], 5);
}

TEST(GurobiModelTest, Sample14) {
    int N = 5;
    boost::unordered_map<std::pair<int, int>, double> edges;
    edges[{0, 1}] = 1;
    edges[{1, 2}] = 1;
    edges[{1, 4}] = 1;
    edges[{2, 3}] = 1;
    edges[{3, 4}] = 1;
    edges[{4, 0}] = 1;

    boost::unordered_map<std::tuple<int, int, int, int>, double> relations;
    relations[{0, 1, 2, 3}] = 0;
    relations[{1, 4, 2, 3}] = 0;

    Instance instance(N, edges, relations, "test");
    GurobiModel model(instance);
    model.formulate();
    model.solveModelWithParameters();

    EXPECT_EQ(model.getModel().get(GRB_IntAttr_Status), GRB_OPTIMAL);
    auto [tour, cost] = model.getSolutionAndCost();
    EXPECT_EQ(tour, (std::vector<int>{0, 1, 2, 3, 4}));
    EXPECT_DOUBLE_EQ(cost, 4.0);

    auto y = model.getY();
    EXPECT_DOUBLE_EQ(y.at({0, 1, 2, 3}).get(GRB_DoubleAttr_X), 1.0);
}

TEST(GurobiModelTest, ArcAfterTargetDoesNotTrigger) {
    int N = 3;
    boost::unordered_map<std::pair<int, int>, double> edges;
    edges[{0, 1}] = 1;
    edges[{1, 2}] = 1;
    edges[{2, 0}] = 1;

    boost::unordered_map<std::tuple<int, int, int, int>, double> relations;
    relations[{1, 2, 0, 1}] = 0;

    Instance instance(N, edges, relations, "test");
    GurobiModel model(instance);
    model.formulate();
    model.solveModelWithParameters();

    EXPECT_EQ(model.getModel().get(GRB_IntAttr_Status), GRB_OPTIMAL);
    auto [tour, cost] = model.getSolutionAndCost();
    EXPECT_EQ(tour, (std::vector<int>{0, 1, 2}));
    EXPECT_DOUBLE_EQ(cost, 3.0);

    auto y = model.getY();
    EXPECT_DOUBLE_EQ(y.at({1, 2, 0, 1}).get(GRB_DoubleAttr_X), 0.0);
}

TEST(GurobiModelTest, SolveWithParameters) {
    // Get the source directory path (two levels up from build directory)
    auto sourceDir = std::filesystem::current_path().parent_path();
    auto instancePath = sourceDir / "tests" / "instances" / "example_1.txt";

    auto instance = Instance::loadInstanceFromFile(instancePath.string());
    GurobiModel model(*instance);
    model.formulate();

    // Test with time limit
    model.solveModelWithParameters(SolverParameters{.timeLimitSec = 1});
    EXPECT_LE(model.getModel().get(GRB_DoubleAttr_Runtime), 1.5);

    // Test with heuristic effort
    model.solveModelWithParameters(SolverParameters{.heuristicEffort = 0.658});
    EXPECT_DOUBLE_EQ(model.getModel().get(GRB_DoubleParam_Heuristics), 0.658);
}

TEST(GurobiModelTest, MIPStartFromTsp1) {
    auto sourceDir = std::filesystem::current_path().parent_path();
    auto instancePath = sourceDir / "tests" / "instances" / "example_1.txt";

    auto instance = Instance::loadInstanceFromFile(instancePath.string());
    GurobiModel model(*instance);
    model.formulate();
    model.solveModelWithParameters(SolverParameters{.mipStart = true});
}

TEST(GurobiModelTest, MIPStartFromTsp2) {
    auto rootDir = std::filesystem::current_path().parent_path().parent_path();
    auto instancePath = rootDir / "instances" / "instances_release_1" / "grf1.txt";

    auto instance = Instance::loadInstanceFromFile(instancePath.string());
    GurobiModel model(*instance);
    model.formulate();
    model.solveModelWithParameters(SolverParameters{.timeLimitSec = 3, .mipStart = true});
}

TEST(GurobiModelTest, MIPModelWarning) {
    auto rootDir = std::filesystem::current_path().parent_path().parent_path();
    auto instancePath = rootDir / "instances" / "instances_generic" / "decrease_n_10_r_200_1.txt";

    auto instance = Instance::loadInstanceFromFile(instancePath.string());
    GurobiModel model(*instance);
    model.formulate();
    model.solveModelWithParameters(SolverParameters{.timeLimitSec = 3});
    std::cout << "Model status: " << model.getModel().get(GRB_IntAttr_Status) << std::endl;
    auto [tour, cost] = model.getSolutionAndCost();
    auto other_cost = instance->computeObjective(tour);
    EXPECT_EQ(tour.size(), 10);
    EXPECT_TRUE(std::abs(cost - other_cost) < 1e-2);
}