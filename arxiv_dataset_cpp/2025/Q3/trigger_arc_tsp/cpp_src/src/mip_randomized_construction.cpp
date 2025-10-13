#include "mip_randomized_construction.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>

MIPRandomizedConstruction::MIPRandomizedConstruction(const Instance& instance,
                                                     int timeLimitSec,
                                                     ConstructionType type,
                                                     std::optional<double> alpha,
                                                     std::optional<double> beta)
    : Method(instance),
      timeLimitSec_(timeLimitSec),
      type_(type),
      rng_(std::chrono::steady_clock::now().time_since_epoch().count()) {
    // Initialize alpha and beta based on construction type and provided values
    if (type_ == ConstructionType::Bias) {
        // For Bias type: alpha and beta default to [0.1, 3.0] range
        std::uniform_real_distribution<double> dist(0.1, 3.0);
        alpha_ = alpha.value_or(dist(rng_));
        beta_ = beta.value_or(dist(rng_));
    } else {  // type_ == ConstructionType::Random
        // For Random type: alpha defaults to [0.0, 1.0] range, beta is not used but set to 0.0
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        alpha_ = alpha.value_or(dist(rng_));
        beta_ = beta.value_or(0.0);  // Beta is not used in Random type
    }
}

void MIPRandomizedConstruction::run() {
    std::vector<int> tour;
    double cost;

    if (type_ == ConstructionType::Bias) {
        // Generate a random permutation and use stored alpha and beta
        auto permutation = generateRandomPermutation();
        TSPPrior tspPrior(permutation, alpha_, beta_);
        std::tie(tour, cost) = evaluateIndividual(tspPrior, timeLimitSec_);
    } else {  // type_ == ConstructionType::Random
        // Use stored alpha
        std::tie(tour, cost) = solveRandomizedTSP(alpha_, std::nullopt, timeLimitSec_);
    }

    assert(!tour.empty());
    bestTour_ = tour;
}

std::vector<int> MIPRandomizedConstruction::getSolution() const {
    return bestTour_;
}

std::pair<std::vector<int>, double> MIPRandomizedConstruction::solveRandomizedTSP(
    std::optional<double> alpha, std::optional<double> beta, int timeLimitSec) {
    // Error handling: exactly one of alpha or beta must be provided
    if (alpha.has_value() && beta.has_value()) {
        throw std::invalid_argument(
            "Both alpha and beta cannot be provided simultaneously in solveRandomizedTSP");
    }
    if (!alpha.has_value() && !beta.has_value()) {
        throw std::invalid_argument("Either alpha or beta must be provided in solveRandomizedTSP");
    }

    double bestCost = std::numeric_limits<double>::infinity();
    std::vector<int> bestTour = {};

    // Apply the appropriate randomization based on which parameter is provided
    boost::unordered_map<std::pair<int, int>, double> edges;
    if (alpha.has_value()) {
        // 1. apply alpha randomization to edges
        edges = applyAlphaRandomizationToEdges(alpha.value());
    } else {
        // 1. apply beta randomization to edges
        edges = applyBetaRandomizationToEdges(beta.value());
    }

    // 2. solve TSP with random edges
    auto tspInstance = createTSPInstance(edges);
    GurobiTSPModel model(*tspInstance);
    model.formulate();
    model.solveToOptimality(timeLimitSec, false);

    // 3. get best tour
    auto tours = model.getBestNTours(15);
    for (const auto& tour : tours) {
        double cost = instance_.computeObjective(tour);
        if (cost < bestCost) {
            bestTour = tour;
            bestCost = cost;
        }
    }

    return {bestTour, bestCost};
}

std::pair<std::vector<int>, double> MIPRandomizedConstruction::evaluateIndividual(
    TSPPrior& tspPrior, int timeLimitSec) {
    assert(tspPrior.priorities.size() == static_cast<size_t>(instance_.getN()));

    double bestCost = std::numeric_limits<double>::infinity();
    std::vector<int> bestTour = {};

    if (instance_.checkSolutionCorrectness(tspPrior.priorities)) {
        bestCost = instance_.computeObjective(tspPrior.priorities);
        bestTour = tspPrior.priorities;
    }

    // Continue with MIP-based search to find potentially better solutions
    auto tspEdges = getEdgesForTSPSearch(tspPrior);
    auto tspInstance = createTSPInstance(tspEdges);

    GurobiTSPModel model(*tspInstance);
    model.formulate();
    model.solveToOptimality(timeLimitSec, false);

    // Get multiple tours and find the best one for the original instance
    auto tours = model.getBestNTours(15);
    for (const auto& tour : tours) {
        double cost = instance_.computeObjective(tour);
        if (cost < bestCost) {
            bestTour = tour;
            bestCost = cost;
        }
    }

    tspPrior.cost = bestCost;
    tspPrior.bestTour = bestTour;
    tspPrior.relGap = model.getModel().get(GRB_DoubleAttr_MIPGap);

    return {bestTour, bestCost};
}

boost::unordered_map<std::pair<int, int>, double>
MIPRandomizedConstruction::applyAlphaRandomizationToEdges(double alpha) {
    // all edges become edge + alpha * random_uniform(-1, 1)
    auto newEdges = instance_.getEdges();
    for (const auto& [edge, cost] : newEdges) {
        newEdges[edge] = cost + alpha * ((rng_() / (double)rng_.max()) * 2 - 1);
    }
    return newEdges;
}

boost::unordered_map<std::pair<int, int>, double>
MIPRandomizedConstruction::applyBetaRandomizationToEdges(double beta) {
    // all edges become edge * beta * random_uniform(0, 1)
    auto newEdges = instance_.getEdges();
    for (const auto& [edge, cost] : newEdges) {
        newEdges[edge] = cost * (beta * (rng_() / (double)rng_.max()));
    }
    return newEdges;
}

boost::unordered_map<std::pair<int, int>, double> MIPRandomizedConstruction::getEdgesForTSPSearch(
    const TSPPrior& tspPrior) {
    // 1. Compute node distances based on priorities
    auto nodeDist = computeNodeDist(tspPrior.priorities);

    // 2. Estimate edge usage probability (inversely proportional to node distance)
    boost::unordered_map<std::pair<int, int>, double> edgeUsedProb;
    for (const auto& [edge, cost] : instance_.getEdges()) {
        edgeUsedProb[edge] = 1.0 / nodeDist[edge];
    }

    // 3. Estimate relation active probability
    boost::unordered_map<std::tuple<int, int, int, int>, double> relationActiveProb;
    for (const auto& [relation, cost] : instance_.getRelations()) {
        auto [trigger1, trigger2, target1, target2] = relation;
        std::pair<int, int> triggerEdge = {trigger1, trigger2};
        std::pair<int, int> targetEdge = {target1, target2};
        std::pair<int, int> connectionEdge = {trigger2, target1};

        double prob = edgeUsedProb[triggerEdge] * edgeUsedProb[targetEdge] *
                      (1.0 / std::pow(nodeDist[connectionEdge], tspPrior.beta));
        relationActiveProb[relation] = prob;
    }

    // 4. Modify edge costs based on relation probabilities using R_a structure
    auto edgesCost = instance_.getEdges();
    for (const auto& [edgeA, edgeList] : instance_.getRA()) {
        for (const auto& edgeB : edgeList) {
            // Create relation key (b, a) -> (edgeB, edgeA)
            std::tuple<int, int, int, int> relationKey = {
                edgeB.first, edgeB.second, edgeA.first, edgeA.second};

            if (relationActiveProb.find(relationKey) != relationActiveProb.end()) {
                auto relationIt = instance_.getRelations().find(relationKey);
                if (relationIt != instance_.getRelations().end()) {
                    double val =
                        tspPrior.alpha * relationActiveProb[relationKey] * relationIt->second;
                    edgesCost[edgeA] += val;
                    edgesCost[edgeB] += val;
                }
            }
        }
    }

    return edgesCost;
}

boost::unordered_map<std::pair<int, int>, double> MIPRandomizedConstruction::computeNodeDist(
    const std::vector<int>& nodePriorities) {
    boost::unordered_map<std::pair<int, int>, double> nodeDist;

    for (size_t i = 0; i < nodePriorities.size(); ++i) {
        for (size_t j = 0; j < nodePriorities.size(); ++j) {
            int nodeI = nodePriorities[i];
            int nodeJ = nodePriorities[j];

            if (i == j) {
                nodeDist[{nodeI, nodeJ}] = 1.0;
            } else {
                int diff = std::abs(static_cast<int>(i) - static_cast<int>(j));
                int cyclicDist = std::min(diff, static_cast<int>(nodePriorities.size()) - diff);
                nodeDist[{nodeI, nodeJ}] = static_cast<double>(cyclicDist);
            }
        }
    }

    return nodeDist;
}

std::vector<int> MIPRandomizedConstruction::generateRandomPermutation() {
    std::vector<int> permutation(instance_.getN());                  // size n
    std::iota(permutation.begin(), permutation.end(), 0);            // Fill with 0, 1, ..., n - 1
    std::shuffle(permutation.begin() + 1, permutation.end(), rng_);  // Shuffle 1 to n - 1
    return permutation;
}

std::vector<std::vector<int>> MIPRandomizedConstruction::generateNRandomPermutations(int n) {
    std::vector<std::vector<int>> permutations;
    permutations.reserve(n);

    for (int i = 0; i < n; ++i) {
        permutations.push_back(generateRandomPermutation());
    }

    return permutations;
}

std::unique_ptr<Instance> MIPRandomizedConstruction::createTSPInstance(
    const boost::unordered_map<std::pair<int, int>, double>& tspEdges) {
    // Create a TSP instance with modified edge costs and no relations
    boost::unordered_map<std::tuple<int, int, int, int>, double> emptyRelations;

    std::string tspName = instance_.getName();
    if (tspName.size() > 4 && tspName.substr(tspName.size() - 4) == ".txt") {
        tspName = tspName.substr(0, tspName.size() - 4);
    }
    tspName += "_tsp.txt";

    return std::make_unique<Instance>(instance_.getN(), tspEdges, emptyRelations, tspName);
}
