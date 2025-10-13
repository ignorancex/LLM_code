#include "tsp_model.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "gurobi_c++.h"

GurobiTSPModel::GurobiTSPModel(const Instance& instance)
    : instance_(instance), env_(createSilentEnvironmentSilently()), model_(env_) {}

GRBEnv GurobiTSPModel::createSilentEnvironmentSilently() {
    GRBEnv env(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    return env;
}

void GurobiTSPModel::formulate() {
    formulated_ = true;

    // Create binary variables for edges
    for (const auto& [edge, cost] : instance_.getEdges()) {
        auto [i, j] = edge;
        std::string name = "x_" + std::to_string(i) + "_" + std::to_string(j);
        x_[edge] = model_.addVar(0.0, 1.0, 0.0, GRB_BINARY, name);
    }

    // Create continuous variables for node positions
    for (int i = 0; i < instance_.getN(); ++i) {
        std::string name = "u_" + std::to_string(i);
        double ub;
        if (i != 0) {
            ub = GRB_INFINITY;
        } else {
            ub = 0.0;  // u_0 is fixed to 0
        }
        u_[i] = model_.addVar(0.0, ub, 0.0, GRB_CONTINUOUS, name);
    }

    // Update model to integrate new variables
    model_.update();

    // Flow conservation constraints
    for (int i = 0; i < instance_.getN(); ++i) {
        GRBLinExpr outSum = 0;
        for (int j : instance_.getDeltaOut(i)) {
            outSum += x_[{i, j}];
        }
        model_.addConstr(outSum == 1, "flow_conservation_out_" + std::to_string(i));

        GRBLinExpr inSum = 0;
        for (int j : instance_.getDeltaIn(i)) {
            inSum += x_[{j, i}];
        }
        model_.addConstr(inSum == 1, "flow_conservation_in_" + std::to_string(i));
    }

    // Subtour elimination constraints
    for (const auto& [edge, _] : instance_.getEdges()) {
        auto [i, j] = edge;

        if (j != 0) {
            model_.addConstr(u_[i] - u_[j] + instance_.getN() * x_[{i, j}] <= instance_.getN() - 1,
                             "subtour_elimination_" + std::to_string(i) + "_" + std::to_string(j));
        }
    }
    // Set objective function
    GRBLinExpr objExpr = 0;
    for (const auto& [edge, cost] : instance_.getEdges()) {
        objExpr += x_[edge] * cost;
    }
    model_.setObjective(objExpr, GRB_MINIMIZE);
}

void GurobiTSPModel::solveToFeasibleSolution() {
    checkModelIsFormulated();

    model_.set(GRB_IntParam_SolutionLimit, 1);
    model_.optimize();

    checkModelStatus();
}

void GurobiTSPModel::solveToOptimality(std::optional<int> timeLimitSec, bool logs) {
    checkModelIsFormulated();

    if (!logs) {
        model_.set(GRB_IntParam_OutputFlag, 0);
    }

    model_.set(GRB_DoubleParam_TimeLimit, timeLimitSec.value_or(60));
    model_.set(GRB_IntParam_MIPFocus, 1);

    model_.optimize();

    checkModelStatus();
}

std::vector<int> GurobiTSPModel::getBestTour() const {
    const_cast<GRBModel&>(model_).update();

    std::vector<std::pair<int, double>> nodePositions;
    for (int i = 1; i < instance_.getN(); ++i) {
        nodePositions.emplace_back(i, u_.at(i).get(GRB_DoubleAttr_Xn));
    }

    std::sort(nodePositions.begin(), nodePositions.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    std::vector<int> tour = {0};
    for (const auto& [node, _] : nodePositions) {
        tour.push_back(node);
    }
    return tour;
}

std::vector<std::vector<int>> GurobiTSPModel::getBestNTours(int n) const {
    int nSolutions = model_.get(GRB_IntAttr_SolCount);
    std::vector<std::vector<int>> tours;

    for (int i = 0; i < std::min(n, nSolutions); ++i) {
        const_cast<GRBModel&>(model_).set(GRB_IntParam_SolutionNumber, i);
        const_cast<GRBModel&>(model_).update();
        tours.push_back(getBestTour());
    }
    return tours;
}

void GurobiTSPModel::checkModelIsFormulated() const {
    if (!formulated_) {
        throw std::runtime_error("Model is not formulated");
    }
}

void GurobiTSPModel::checkModelStatus() const {
    int status = model_.get(GRB_IntAttr_Status);
    if (status == GRB_INFEASIBLE) {
        const_cast<GRBModel&>(model_).computeIIS();
        const_cast<GRBModel&>(model_).write("model.ilp");
        throw std::runtime_error("Model is infeasible");
    }
}
