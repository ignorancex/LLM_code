#include "simple_randomized_construction.hpp"

#include <algorithm>
#include <boost/unordered_set.hpp>
#include <numeric>
#include <random>
#include <vector>

SimpleRandomizedConstruction::SimpleRandomizedConstruction(const Instance& instance)
    : Method(instance), rng_(std::random_device{}()) {}

void SimpleRandomizedConstruction::run() {
    tour_ = constructSolution();
}

std::vector<int> SimpleRandomizedConstruction::getSolution() const {
    return tour_;
}

std::vector<int> SimpleRandomizedConstruction::constructSolution() {
    std::vector<int> tour;
    tour.reserve(instance_.getN());
    tour.push_back(0);

    boost::unordered_set<int> visited_nodes;
    visited_nodes.insert(0);

    while (tour.size() < static_cast<size_t>(instance_.getN())) {
        int last_node = tour.back();
        const size_t expectedTourSize = static_cast<size_t>(instance_.getN());
        const bool isChoosingFinalNode = tour.size() == (expectedTourSize - 1);

        std::vector<int> feasible_neighbors;
        const auto& neighbors = instance_.getDeltaOut(last_node);
        for (int neighbor : neighbors) {
            if (visited_nodes.find(neighbor) == visited_nodes.end()) {
                if (isChoosingFinalNode) {
                    // For the final node, ensure it can reach node 0
                    if (instance_.getDeltaOut(neighbor).count(0)) {
                        feasible_neighbors.push_back(neighbor);
                    }
                } else {
                    feasible_neighbors.push_back(neighbor);
                }
            }
        }

        if (feasible_neighbors.empty()) {
            return {};
        }

        std::uniform_int_distribution<int> dist(0, feasible_neighbors.size() - 1);
        int next_node = feasible_neighbors[dist(rng_)];

        tour.push_back(next_node);
        visited_nodes.insert(next_node);
    }

    return tour;
}