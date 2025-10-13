#include "model.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>

#include "instance.hpp"

GurobiModel::GurobiModel(const Instance& instance)
    : instance_(instance), env_(createSilentEnvironmentSilently()), model_(env_) {}

GRBEnv GurobiModel::createSilentEnvironmentSilently() {
    GRBEnv env(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    return env;
}

void GurobiModel::formulate() {
    addVariables();
    addConstraints();
    addObjective();
    formulated_ = true;
}

void GurobiModel::addVariables() {
    // Add edge variables (x)
    for (const auto& [edge, cost] : instance_.getEdges()) {
        auto [i, j] = edge;
        std::string name = "x_" + std::to_string(i) + "_" + std::to_string(j);
        x_[edge] = model_.addVar(0.0, 1.0, 0.0, GRB_BINARY, name);
    }

    // Add node position variables (u)
    for (int i = 0; i < instance_.getN(); ++i) {
        std::string name = "u_" + std::to_string(i);
        double ub = (i == 0) ? 0.0 : instance_.getN() - 1;
        u_[i] = model_.addVar(0.0, ub, 0.0, GRB_CONTINUOUS, name);
    }

    // Add relation variables (y)
    for (const auto& [relation, cost] : instance_.getRelations()) {
        auto [b0, b1, a0, a1] = relation;
        std::string name = "y_" + std::to_string(b0) + "_" + std::to_string(b1) + "_" +
                           std::to_string(a0) + "_" + std::to_string(a1);
        y_[relation] = model_.addVar(0.0, 1.0, 0.0, GRB_BINARY, name);
    }

    // Add precedence variables (z)
    for (const auto& [a1, a2, b1, b2] : instance_.getZVarIndices()) {
        std::string name = "z_" + std::to_string(a1) + "_" + std::to_string(a2) + "_" +
                           std::to_string(b1) + "_" + std::to_string(b2);
        z_[{a1, a2, b1, b2}] = model_.addVar(0.0, 1.0, 0.0, GRB_BINARY, name);
    }

    model_.update();
}

void GurobiModel::addConstraints() {
    // (1) Flow conservation constraints
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

    // (2) Subtour elimination constraints
    for (const auto& [edge, _] : instance_.getEdges()) {
        auto [i, j] = edge;
        if (j != 0) {
            model_.addConstr(u_[i] - u_[j] + instance_.getN() * x_[{i, j}] <= instance_.getN() - 1,
                             "subtour_elimination_" + std::to_string(i) + "_" + std::to_string(j));
        }
    }

    // (3) At most one relation can be active in R_a
    for (const auto& [a, relations] : instance_.getRA()) {
        GRBLinExpr sum = 0;
        for (const auto& b : relations) {
            sum += y_[{b.first, b.second, a.first, a.second}];
        }
        model_.addConstr(
            sum <= x_[a],
            "max_one_relation_" + std::to_string(a.first) + "_" + std::to_string(a.second));
    }

    // (4) Relation r=(b,a) is inactive if a or b are inactive
    for (const auto& [a, relations] : instance_.getRA()) {
        for (const auto& b : relations) {
            model_.addConstr(y_[{b.first, b.second, a.first, a.second}] <= x_[a],
                             "relation_inactive_if_target_inactive_1_" + std::to_string(b.first) +
                                 "_" + std::to_string(b.second) + "_" + std::to_string(a.first) +
                                 "_" + std::to_string(a.second));
            model_.addConstr(y_[{b.first, b.second, a.first, a.second}] <= x_[b],
                             "relation_inactive_if_target_inactive_2_" + std::to_string(b.first) +
                                 "_" + std::to_string(b.second) + "_" + std::to_string(a.first) +
                                 "_" + std::to_string(a.second));
        }
    }

    // (5) Relation r=(b,a) is inactive if b after a in the tour
    for (const auto& [a, relations] : instance_.getRA()) {
        for (const auto& b : relations) {
            model_.addConstr(y_[{b.first, b.second, a.first, a.second}] <=
                                 z_[{b.first, b.second, a.first, a.second}],
                             "trigger_before_arc_" + std::to_string(b.first) + "_" +
                                 std::to_string(b.second) + "_" + std::to_string(a.first) + "_" +
                                 std::to_string(a.second));
        }
    }

    // (6) At least one relation is active
    for (const auto& [a, relations] : instance_.getRA()) {
        for (const auto& b : relations) {
            GRBLinExpr sum = 0;
            for (const auto& c : relations) {
                sum += y_[{c.first, c.second, a.first, a.second}];
            }
            model_.addConstr(
                1 - z_[{a.first, a.second, b.first, b.second}] <= sum + (1 - x_[a]) + (1 - x_[b]),
                "force_relation_active_" + std::to_string(a.first) + "_" +
                    std::to_string(a.second) + "_" + std::to_string(b.first) + "_" +
                    std::to_string(b.second));
        }
    }

    // (7) Precedence on z variables
    for (const auto& [a1, a2, b1, b2] : instance_.getZVarIndices()) {
        if (a1 != b1 || a2 != b2) {
            model_.addConstr(u_[a1] <= u_[b1] + (instance_.getN() - 1) * (1 - z_[{a1, a2, b1, b2}]),
                             "model_z_variables_1_" + std::to_string(a1) + "_" +
                                 std::to_string(a2) + "_" + std::to_string(b1) + "_" +
                                 std::to_string(b2));
            model_.addConstr(z_[{a1, a2, b1, b2}] == 1 - z_[{b1, b2, a1, a2}],
                             "model_z_variables_2_" + std::to_string(a1) + "_" +
                                 std::to_string(a2) + "_" + std::to_string(b1) + "_" +
                                 std::to_string(b2));
        } else {
            model_.addConstr(z_[{a1, a2, b1, b2}] == z_[{b1, b2, a1, a2}],
                             "model_z_variables_3_" + std::to_string(a1) + "_" +
                                 std::to_string(a2) + "_" + std::to_string(b1) + "_" +
                                 std::to_string(b2));
        }
    }

    // Only last relation triggers
    for (const auto& [a, relations] : instance_.getRA()) {
        for (const auto& b : relations) {
            for (const auto& c : relations) {
                if (b != c) {
                    model_.addConstr(y_[{b.first, b.second, a.first, a.second}] <=
                                         z_[{c.first, c.second, b.first, b.second}] +
                                             z_[{a.first, a.second, c.first, c.second}] +
                                             (1 - x_[c]) + (1 - x_[b]) + (1 - x_[a]),
                                     "only_last_relation_triggers_" + std::to_string(b.first) +
                                         "_" + std::to_string(b.second) + "_" +
                                         std::to_string(a.first) + "_" + std::to_string(a.second) +
                                         "_" + std::to_string(c.first) + "_" +
                                         std::to_string(c.second));
                }
            }
        }
    }
}

void GurobiModel::addObjective() {
    GRBLinExpr obj = 0;

    // Edge costs
    for (const auto& [edge, cost] : instance_.getEdges()) {
        obj += x_[edge] * cost;
    }

    // Relation costs
    for (const auto& [relation, cost] : instance_.getRelations()) {
        obj += y_[relation] * cost;
    }

    model_.setObjective(obj, GRB_MINIMIZE);
}

void GurobiModel::solveModelWithParameters() {
    solveModelWithParameters(SolverParameters());
}

void GurobiModel::solveModelWithParameters(const SolverParameters& params) {
    checkModelIsFormulated();

    if (params.mipStart) {
        auto vars = instance_.getMipStart(true);
        provideMipStart(vars);
    }

    if (params.timeLimitSec > 0) {
        model_.set(GRB_DoubleParam_TimeLimit, params.timeLimitSec);
    }
    model_.set(GRB_DoubleParam_Heuristics, params.heuristicEffort);
    model_.set(GRB_IntParam_Presolve, params.presolve);

    if (!params.logs) {
        model_.set(GRB_IntParam_OutputFlag, 0);
    }

    model_.optimize();
    checkModelStatus();
}

std::pair<std::vector<int>, double> GurobiModel::getSolutionAndCost() const {
    const_cast<GRBModel&>(model_).update();

    // Get node positions
    std::vector<std::pair<int, double>> nodePositions;
    for (int i = 1; i < instance_.getN(); ++i) {
        nodePositions.emplace_back(i, u_.at(i).get(GRB_DoubleAttr_Xn));
    }

    // Sort nodes by position
    std::sort(nodePositions.begin(), nodePositions.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    // Build tour
    std::vector<int> tour = {0};
    for (const auto& [node, _] : nodePositions) {
        tour.push_back(node);
    }

    // Calculate cost
    double cost = 0;
    for (const auto& [edge, edgeCost] : instance_.getEdges()) {
        if (x_.at(edge).get(GRB_DoubleAttr_Xn) > 0.5) {
            cost += edgeCost;
        }
    }
    for (const auto& [relation, relationCost] : instance_.getRelations()) {
        if (y_.at(relation).get(GRB_DoubleAttr_Xn) > 0.5) {
            cost += relationCost;
        }
    }

    return {tour, cost};
}

void GurobiModel::checkModelIsFormulated() const {
    if (!formulated_) {
        throw std::runtime_error("Model is not formulated");
    }
}

void GurobiModel::checkModelStatus() const {
    int status = model_.get(GRB_IntAttr_Status);
    if (status == GRB_INFEASIBLE) {
        const_cast<GRBModel&>(model_).computeIIS();
        const_cast<GRBModel&>(model_).write("model.ilp");
        throw std::runtime_error("Model is infeasible");
    }
}

void GurobiModel::provideMipStart(
    const std::vector<boost::unordered_map<std::string, double>>& vars) const {
    assert(vars.size() == 4);

    // Set x variables (edge variables)
    for (const auto& [key, val] : vars[0]) {
        size_t pos = key.find(',');
        int i = std::stoi(key.substr(0, pos));
        int j = std::stoi(key.substr(pos + 1));
        const_cast<GRBVar&>(x_.at({i, j})).set(GRB_DoubleAttr_Start, val);
    }

    // Set y variables (relation variables)
    for (const auto& [key, val] : vars[1]) {
        size_t pos1 = key.find(',');
        size_t pos2 = key.find(',', pos1 + 1);
        size_t pos3 = key.find(',', pos2 + 1);
        int b0 = std::stoi(key.substr(0, pos1));
        int b1 = std::stoi(key.substr(pos1 + 1, pos2 - pos1 - 1));
        int a0 = std::stoi(key.substr(pos2 + 1, pos3 - pos2 - 1));
        int a1 = std::stoi(key.substr(pos3 + 1));
        const_cast<GRBVar&>(y_.at({b0, b1, a0, a1})).set(GRB_DoubleAttr_Start, val);
    }

    // Set u variables (node position variables)
    for (const auto& [key, val] : vars[2]) {
        int i = std::stoi(key);
        const_cast<GRBVar&>(u_.at(i)).set(GRB_DoubleAttr_Start, val);
    }

    // Set z variables (precedence variables)
    for (const auto& [key, val] : vars[3]) {
        size_t pos1 = key.find(',');
        size_t pos2 = key.find(',', pos1 + 1);
        size_t pos3 = key.find(',', pos2 + 1);
        int a1 = std::stoi(key.substr(0, pos1));
        int a2 = std::stoi(key.substr(pos1 + 1, pos2 - pos1 - 1));
        int b1 = std::stoi(key.substr(pos2 + 1, pos3 - pos2 - 1));
        int b2 = std::stoi(key.substr(pos3 + 1));
        const_cast<GRBVar&>(z_.at({a1, a2, b1, b2})).set(GRB_DoubleAttr_Start, val);
    }
}
