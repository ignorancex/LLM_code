#include "randomized_greedy.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>

#include "boost/unordered_set.hpp"

RandomizedGreedyConstruction::RandomizedGreedyConstruction(const Instance& instance, double alpha)
    : Method(instance),
      alpha_(alpha),
      bestCost_(std::numeric_limits<double>::infinity()),
      rng_(std::chrono::steady_clock::now().time_since_epoch().count()) {}

void RandomizedGreedyConstruction::run() {
    auto solution = constructSolution();
    if (solution.empty() || !instance_.checkSolutionCorrectness(solution)) {
        return;
    }

    double cost = instance_.computeObjective(solution);

    if (cost < bestCost_) {
        bestSolution_ = solution;
        bestCost_ = cost;
    }
}

std::vector<int> RandomizedGreedyConstruction::getSolution() const {
    return bestSolution_;
}

std::vector<int> RandomizedGreedyConstruction::constructSolution() {
    const size_t maxAttempts = (alpha_ > 0.0) ? 1000 : 1;
    for (size_t i = 0; i < maxAttempts; ++i) {
        std::vector<int> tour;
        tour.reserve(static_cast<size_t>(instance_.getN()));
        tour.push_back(0);

        boost::unordered_set<int> visitedNodes;
        visitedNodes.insert(0);

        bool stuck = false;
        while (tour.size() < static_cast<size_t>(instance_.getN())) {
            auto feasibleEdges = getFeasibleEdges(tour, visitedNodes);
            if (feasibleEdges.empty()) {
                stuck = true;
                break;
            }

            auto selectedEdge = selectRandomizedGreedyEdge(feasibleEdges, tour);
            tour.push_back(selectedEdge.second);
            visitedNodes.insert(selectedEdge.second);
        }

        if (!stuck) {
            return tour;
        }
    }
    return {};
}

std::vector<std::pair<int, int>> RandomizedGreedyConstruction::getFeasibleEdges(
    const std::vector<int>& partialTour, const boost::unordered_set<int>& visitedNodes) const {
    std::vector<std::pair<int, int>> feasibleEdges;
    const int currentNode = partialTour.back();
    const size_t expectedTourSize = static_cast<size_t>(instance_.getN());
    const bool isChoosingFinalNode = partialTour.size() == (expectedTourSize - 1);

    const auto& outgoingNodes = instance_.getDeltaOut(currentNode);
    for (const int nextNode : outgoingNodes) {
        if (visitedNodes.count(nextNode)) {
            continue;
        }

        if (isChoosingFinalNode) {
            if (instance_.getDeltaOut(nextNode).count(0)) {
                feasibleEdges.emplace_back(currentNode, nextNode);
            }
        } else {
            feasibleEdges.emplace_back(currentNode, nextNode);
        }
    }

    return feasibleEdges;
}

std::pair<int, int> RandomizedGreedyConstruction::selectRandomizedGreedyEdge(
    const std::vector<std::pair<int, int>>& feasibleEdges,
    const std::vector<int>& partialTour) const {
    assert(!feasibleEdges.empty());

    if (alpha_ == 0.0) {
        // Pure greedy: always select the edge with the minimum cost.
        double bestIncrementalCost = std::numeric_limits<double>::infinity();
        std::optional<std::pair<int, int>> bestEdge;

        for (const auto& edge : feasibleEdges) {
            auto extendedTour = partialTour;
            extendedTour.push_back(edge.second);
            // Compute only the incremental cost from the last edge position
            double incrementalCost =
                instance_.computePartialTourCost(extendedTour, partialTour.size() - 1);
            if (!bestEdge.has_value() || incrementalCost < bestIncrementalCost) {
                bestIncrementalCost = incrementalCost;
                bestEdge = edge;
            }
        }
        return bestEdge.value();
    }

    // Randomized selection
    std::vector<std::pair<double, std::pair<int, int>>> edgeCosts;
    edgeCosts.reserve(feasibleEdges.size());

    for (const auto& edge : feasibleEdges) {
        auto extendedTour = partialTour;
        extendedTour.push_back(edge.second);
        // Compute only the incremental cost from the last edge position
        double incrementalCost =
            instance_.computePartialTourCost(extendedTour, partialTour.size() - 1);
        edgeCosts.emplace_back(incrementalCost, edge);
    }

    std::sort(edgeCosts.begin(), edgeCosts.end());

    size_t rcl_pool_size = std::max(
        size_t(1), static_cast<size_t>(std::ceil(alpha_ * static_cast<double>(edgeCosts.size()))));
    const size_t rcl_size = std::min(rcl_pool_size, edgeCosts.size());

    std::uniform_int_distribution<size_t> dist(0, rcl_size - 1);
    const size_t selectedIdx = dist(rng_);

    return edgeCosts[selectedIdx].second;
}