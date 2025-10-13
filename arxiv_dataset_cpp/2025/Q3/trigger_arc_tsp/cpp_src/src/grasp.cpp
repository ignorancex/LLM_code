#include "grasp.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "mip_randomized_construction.hpp"
#include "randomized_greedy.hpp"
#include "simple_randomized_construction.hpp"

GRASP::GRASP(const Instance& instance,
             int n_trials,
             double alpha,
             double beta,
             ConstructiveHeuristicType constructive_heuristic_type,
             const std::vector<LocalSearch>& local_searches,
             bool log_to_stdout,
             bool start_with_best_saved_solution,
             int time_limit_sec)
    : Method(instance),
      n_trials_(n_trials),
      alpha_(alpha),
      beta_(beta),
      constructive_heuristic_type_(constructive_heuristic_type),
      local_searches_(local_searches),
      log_to_stdout_(log_to_stdout),
      start_with_best_saved_solution_(start_with_best_saved_solution),
      time_limit_sec_(time_limit_sec) {
    switch (constructive_heuristic_type_) {
        case ConstructiveHeuristicType::RandomizedGreedy:
            constructive_heuristic_ =
                std::make_unique<RandomizedGreedyConstruction>(instance_, alpha_);
            break;
        case ConstructiveHeuristicType::MIPRandomizedGreedyBias:
            constructive_heuristic_ =
                std::make_unique<MIPRandomizedConstruction>(instance_, 10, ConstructionType::Bias);
            break;
        case ConstructiveHeuristicType::MIPRandomizedGreedyRandom:
            constructive_heuristic_ = std::make_unique<MIPRandomizedConstruction>(
                instance_, 10, ConstructionType::Random);
            break;
        case ConstructiveHeuristicType::SimpleRandomized:
            constructive_heuristic_ = std::make_unique<SimpleRandomizedConstruction>(instance_);
            break;
    }
}

std::optional<std::vector<int>> GRASP::localSearch(const std::vector<int>& tour) {
    instance_.resetTrackers();
    double current_cost = instance_.computeObjective(tour);

    auto is_local_search_type_enabled = [this](LocalSearch search_type) -> bool {
        return std::find(local_searches_.begin(), local_searches_.end(), search_type) !=
               local_searches_.end();
    };

    if (is_local_search_type_enabled(LocalSearch::TwoOpt)) {
        while (auto next_pair = instance_.twoOptIteratorTracker_.next()) {
            std::vector<int> new_tour = tour;
            instance_.get_two_opt_neighbor(new_tour, next_pair->first, next_pair->second);
            if (!instance_.checkSolutionCorrectness(new_tour)) {
                continue;
            }
            double new_cost = instance_.computeObjective(new_tour);
            if (new_cost < current_cost) {
                ;
                return new_tour;
            }
        }
    }

    if (is_local_search_type_enabled(LocalSearch::SwapTwo)) {
        while (auto next_pair = instance_.swapTwoIteratorTracker_.next()) {
            std::vector<int> new_tour = tour;
            instance_.get_swap_two_neighbor(new_tour, next_pair->first, next_pair->second);
            if (!instance_.checkSolutionCorrectness(new_tour)) {
                continue;
            }
            double new_cost = instance_.computeObjective(new_tour);
            if (new_cost < current_cost) {
                return new_tour;
            }
        }
    }
    if (is_local_search_type_enabled(LocalSearch::Relocate)) {
        while (auto next_pair = instance_.relocateIteratorTracker_.next()) {
            std::vector<int> new_tour = tour;
            instance_.get_relocate_neighbor(new_tour, next_pair->first, next_pair->second);
            if (!instance_.checkSolutionCorrectness(new_tour)) {
                continue;
            }
            double new_cost = instance_.computeObjective(new_tour);
            if (new_cost < current_cost) {
                return new_tour;
            }
        }
    }

    return std::nullopt;
}

void GRASP::run() {
    iteration_ = 0;
    best_cost_ = std::numeric_limits<double>::infinity();

    if (log_to_stdout_) {
        std::cout << "GRASP::run() called with n_trials=" << n_trials_ << ", alpha=" << alpha_
                  << ", beta=" << beta_ << std::endl;
    }

    if (start_with_best_saved_solution_) {
        auto best_known_solution = instance_.getBestKnownSolution();
        if (best_known_solution) {
            best_tour_ = *best_known_solution;
            best_cost_ = instance_.computeObjective(best_tour_);

            if (log_to_stdout_) {
                std::cout << "Loaded solution with objective " << best_cost_ << std::endl;
            }
        } else {
            throw std::runtime_error(
                "start_with_best_saved_solution_ is true, but no known solution found");
        }
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    auto get_time_elapsed = [start_time]() {
        // return the time elapsed in seconds
        return std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::high_resolution_clock::now() - start_time)
            .count();
    };

    while (iteration_ < n_trials_ && get_time_elapsed() < time_limit_sec_) {
        // 1. construct initial tour
        constructive_heuristic_->run();
        auto tour = constructive_heuristic_->getSolution();
        if (tour.empty() || !instance_.checkSolutionCorrectness(tour)) {
            // some heuristics can fail to construct a tour
            iteration_++;
            continue;
        }
        double current_cost = instance_.computeObjective(tour);
        if (current_cost < best_cost_) {
            log_iteration(current_cost, best_cost_, get_time_elapsed(), true);
            best_tour_ = tour;
            best_cost_ = current_cost;
        } else {
            log_iteration(current_cost, best_cost_, get_time_elapsed());
        }

        // 2. apply local search until no improvement is found
        while (true) {
            auto improved_tour = localSearch(tour);
            if (improved_tour) {
                tour = *improved_tour;
                current_cost = instance_.computeObjective(tour);
                log_iteration(current_cost, best_cost_, get_time_elapsed());
            } else {
                break;
            }
        }
        if (current_cost < best_cost_) {
            log_iteration(current_cost, best_cost_, get_time_elapsed(), true);
            best_tour_ = tour;
            best_cost_ = current_cost;
        }
        iteration_++;
    }
}

std::vector<int> GRASP::getSolution() const {
    return best_tour_;
}

void GRASP::log_iteration(double cost, double best_cost, double time_elapsed, bool improvement) {
    if (!log_to_stdout_) {
        return;
    }

    std::cout << "Iteration " << std::right << std::setw(3) << iteration_
              << " | Current cost: " << std::fixed << std::setprecision(2) << std::setw(8) << cost
              << " | Best cost: " << std::fixed << std::setprecision(2) << std::setw(8) << best_cost
              << " | Time elapsed: " << std::setw(4) << static_cast<long long>(time_elapsed) << "s";

    if (improvement) {
        std::cout << " ** Best updated **";
    }
    std::cout << std::endl;
}