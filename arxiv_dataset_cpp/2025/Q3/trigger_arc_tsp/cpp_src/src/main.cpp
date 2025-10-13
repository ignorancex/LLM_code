#include <boost/json/src.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "grasp.hpp"
#include "instance.hpp"
#include "mip_randomized_construction.hpp"
#include "model.hpp"
#include "randomized_greedy.hpp"
#include "simple_randomized_construction.hpp"

namespace po = boost::program_options;

std::string tourToString(const std::vector<int>& tour) {
    std::stringstream ss;
    for (size_t i = 0; i < tour.size(); ++i) {
        ss << tour[i];
        if (i < tour.size() - 1) {
            ss << ",";
        }
    }
    return ss.str();
}

int main(int argc, char* argv[]) {
    po::variables_map vm;
    po::options_description desc("Trigger Arc TSP Solver");

    desc.add_options()("help,h", "Show help message")(
        "instance-file,i", po::value<std::string>()->required(), "Path to the instance file")(
        "method",
        po::value<std::string>()->default_value("gurobi"),
        "Solution method (gurobi, randomized_greedy, mip_randomized_construction, "
        "mip_randomized_construction_random, grasp, or simple_randomized)")(
        "alpha",
        po::value<double>(),
        "Alpha parameter for randomized greedy and MIP randomized construction (0.0 to 1.0)")(
        "time-limit,t", po::value<int>()->default_value(60), "Time limit in seconds")(
        "heuristic-effort,e",
        po::value<double>()->default_value(0.05),
        "Heuristic effort (0.0 to 1.0)")(
        "presolve,p",
        po::value<int>()->default_value(-1),
        "Presolve level (-1: automatic, 0: off, 1: conservative, 2: aggressive)")(
        "mip-start,m", "Use MIP start if available")("logs,l", "Show solver logs")(
        "n-trials", po::value<int>()->default_value(10), "Number of trials for GRASP")(
        "beta", po::value<double>(), "Beta parameter for MIP randomized construction and GRASP")(
        "start-with-best", "Start GRASP with the best known solution")(
        "constructive-heuristic",
        po::value<std::string>()->default_value("RandomizedGreedy"),
        "Constructive heuristic for GRASP (RandomizedGreedy, MIPRandomizedGreedyBias, "
        "MIPRandomizedGreedyRandom, SimpleRandomized)")(
        "local-searches",
        po::value<std::vector<std::string>>()->multitoken(),
        "Local searches for GRASP (TwoOpt, SwapTwo, Relocate)")(
        "output-dir,o",
        po::value<std::string>()->default_value("output"),
        "Path to the output directory")("save-solutions", "Save solution files (default: false)");

    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << desc << "\n";
            return 0;
        }

        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error parsing command line: " << e.what() << "\n";
        std::cerr << "Use --help for usage information\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::stringstream param_ss;
    param_ss << "method=" << vm["method"].as<std::string>() << ";"
             << "time-limit=" << vm["time-limit"].as<int>() << ";"
             << "heuristic-effort=" << vm["heuristic-effort"].as<double>() << ";"
             << "presolve=" << vm["presolve"].as<int>() << ";"
             << "mip-start=" << (vm.count("mip-start") > 0) << ";"
             << "logs=" << (vm.count("logs") > 0) << ";"
             << "n-trials=" << vm["n-trials"].as<int>() << ";"
             << "start-with-best=" << (vm.count("start-with-best") > 0) << ";"
             << "constructive-heuristic=" << vm["constructive-heuristic"].as<std::string>() << ";";

    // Handle optional alpha and beta parameters
    if (vm.count("alpha")) {
        param_ss << "alpha=" << vm["alpha"].as<double>() << ";";
    }
    if (vm.count("beta")) {
        param_ss << "beta=" << vm["beta"].as<double>() << ";";
    }
    if (vm.count("local-searches")) {
        param_ss << "local-searches=";
        const auto& searches = vm["local-searches"].as<std::vector<std::string>>();
        for (size_t i = 0; i < searches.size(); ++i) {
            param_ss << searches[i] << (i < searches.size() - 1 ? "," : "");
        }
        param_ss << ";";
    }

    std::string param_string = param_ss.str();
    std::hash<std::string> hasher;
    size_t param_hash = hasher(param_string);

    std::cout << "Parameters hash: " << param_hash << std::endl;

    auto instance = Instance::loadInstanceFromFile(vm["instance-file"].as<std::string>());
    std::vector<int> tour;
    double cost = std::numeric_limits<double>::infinity();
    double mipGap = 0.0;
    double wallTime = 0.0;
    double objBound = 0.0;
    std::string methodName = vm["method"].as<std::string>();

    auto startTime = std::chrono::high_resolution_clock::now();

    if (methodName == "gurobi") {
        SolverParameters params = {.timeLimitSec = vm["time-limit"].as<int>(),
                                   .heuristicEffort = vm["heuristic-effort"].as<double>(),
                                   .presolve = vm["presolve"].as<int>(),
                                   .mipStart = vm.count("mip-start") > 0,
                                   .logs = vm.count("logs") > 0};

        GurobiModel model(*instance);
        model.formulate();
        model.solveModelWithParameters(params);

        std::tie(tour, cost) = model.getSolutionAndCost();
        mipGap = model.getMIPGap();
        wallTime = model.getWallTime();
        objBound = model.getObjBound();
    } else if (methodName == "randomized_greedy") {
        if (!vm.count("alpha")) {
            std::cerr << "Error: Alpha parameter is required for randomized_greedy method\n";
            return 1;
        }
        double alpha = vm["alpha"].as<double>();
        RandomizedGreedyConstruction greedy(*instance, alpha);
        greedy.run();
        tour = greedy.getSolution();
        if (!tour.empty()) {
            cost = instance->computeObjective(tour);
        } else {
            cost = std::numeric_limits<double>::infinity();
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        wallTime = std::chrono::duration<double>(endTime - startTime).count();
    } else if (methodName == "mip_randomized_construction") {
        int timeLimitSec = vm["time-limit"].as<int>();
        std::optional<double> alpha =
            vm.count("alpha") ? std::optional<double>(vm["alpha"].as<double>()) : std::nullopt;
        std::optional<double> beta =
            vm.count("beta") ? std::optional<double>(vm["beta"].as<double>()) : std::nullopt;

        MIPRandomizedConstruction mipRC(
            *instance, timeLimitSec, ConstructionType::Bias, alpha, beta);
        mipRC.run();
        tour = mipRC.getSolution();
        if (!tour.empty()) {
            cost = instance->computeObjective(tour);
        } else {
            cost = std::numeric_limits<double>::infinity();
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        wallTime = std::chrono::duration<double>(endTime - startTime).count();
    } else if (methodName == "mip_randomized_construction_random") {
        int timeLimitSec = vm["time-limit"].as<int>();
        std::optional<double> alpha =
            vm.count("alpha") ? std::optional<double>(vm["alpha"].as<double>()) : std::nullopt;
        std::optional<double> beta =
            vm.count("beta") ? std::optional<double>(vm["beta"].as<double>()) : std::nullopt;

        MIPRandomizedConstruction mipRC(
            *instance, timeLimitSec, ConstructionType::Random, alpha, beta);
        mipRC.run();
        tour = mipRC.getSolution();
        if (!tour.empty()) {
            cost = instance->computeObjective(tour);
        } else {
            cost = std::numeric_limits<double>::infinity();
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        wallTime = std::chrono::duration<double>(endTime - startTime).count();
    } else if (methodName == "simple_randomized") {
        SimpleRandomizedConstruction simple(*instance);
        simple.run();
        tour = simple.getSolution();
        if (!tour.empty()) {
            cost = instance->computeObjective(tour);
        } else {
            cost = std::numeric_limits<double>::infinity();
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        wallTime = std::chrono::duration<double>(endTime - startTime).count();
    } else if (methodName == "grasp") {
        int n_trials = vm["n-trials"].as<int>();
        if (!vm.count("alpha")) {
            std::cerr << "Error: Alpha parameter is required for GRASP method\n";
            return 1;
        }
        if (!vm.count("beta")) {
            std::cerr << "Error: Beta parameter is required for GRASP method\n";
            return 1;
        }
        double alpha = vm["alpha"].as<double>();
        double beta = vm["beta"].as<double>();
        bool start_with_best = vm.count("start-with-best") > 0;
        int time_limit_sec = vm["time-limit"].as<int>();

        std::string constructive_heuristic_str = vm["constructive-heuristic"].as<std::string>();
        ConstructiveHeuristicType constructive_heuristic;
        if (constructive_heuristic_str == "RandomizedGreedy") {
            constructive_heuristic = ConstructiveHeuristicType::RandomizedGreedy;
        } else if (constructive_heuristic_str == "MIPRandomizedGreedyBias") {
            constructive_heuristic = ConstructiveHeuristicType::MIPRandomizedGreedyBias;
        } else if (constructive_heuristic_str == "MIPRandomizedGreedyRandom") {
            constructive_heuristic = ConstructiveHeuristicType::MIPRandomizedGreedyRandom;
        } else if (constructive_heuristic_str == "SimpleRandomized") {
            constructive_heuristic = ConstructiveHeuristicType::SimpleRandomized;
        } else {
            std::cerr << "Error: Unknown constructive heuristic '" << constructive_heuristic_str
                      << "'\n";
            return 1;
        }

        std::vector<LocalSearch> local_searches;
        if (vm.count("local-searches")) {
            for (const auto& ls_str : vm["local-searches"].as<std::vector<std::string>>()) {
                if (ls_str == "TwoOpt") {
                    local_searches.push_back(LocalSearch::TwoOpt);
                } else if (ls_str == "SwapTwo") {
                    local_searches.push_back(LocalSearch::SwapTwo);
                } else if (ls_str == "Relocate") {
                    local_searches.push_back(LocalSearch::Relocate);
                } else {
                    std::cerr << "Error: Unknown local search '" << ls_str << "'\n";
                    return 1;
                }
            }
        }

        GRASP grasp(*instance,
                    n_trials,
                    alpha,
                    beta,
                    constructive_heuristic,
                    local_searches,
                    vm.count("logs") > 0,
                    start_with_best,
                    time_limit_sec);
        grasp.run();
        tour = grasp.getSolution();
        if (!tour.empty()) {
            cost = instance->computeObjective(tour);
        } else {
            cost = std::numeric_limits<double>::infinity();
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        wallTime = std::chrono::duration<double>(endTime - startTime).count();
    } else {
        std::cerr << "Error: Unknown method '" << methodName << "'\n";
        return 1;
    }

    if (!tour.empty() && vm.count("save-solutions")) {
        instance->saveSolution(tour, cost);
    }

    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    std::string timestamp = ss.str();

    boost::json::object json;
    json["parameters_hash"] = param_hash;
    json["instance_name"] = instance->getName();
    json["instance_file"] = vm["instance-file"].as<std::string>();
    json["method"] = methodName;

    if (methodName == "gurobi") {
        json["time_limit"] = vm["time-limit"].as<int>();
        json["heuristic_effort"] = vm["heuristic-effort"].as<double>();
        json["presolve"] = vm["presolve"].as<int>();
        json["mip_start"] = vm.count("mip-start") > 0 ? "true" : "false";
        json["mip_gap"] = mipGap;
        json["obj_bound"] = objBound;
    } else if (methodName == "randomized_greedy") {
        json["alpha"] = vm["alpha"].as<double>();
    } else if (methodName == "mip_randomized_construction") {
        json["time_limit"] = vm["time-limit"].as<int>();
        json["construction_type"] = "Bias";
        if (vm.count("alpha")) {
            json["alpha"] = vm["alpha"].as<double>();
        }
        if (vm.count("beta")) {
            json["beta"] = vm["beta"].as<double>();
        }
    } else if (methodName == "mip_randomized_construction_random") {
        json["time_limit"] = vm["time-limit"].as<int>();
        json["construction_type"] = "Random";
        if (vm.count("alpha")) {
            json["alpha"] = vm["alpha"].as<double>();
        }
        if (vm.count("beta")) {
            json["beta"] = vm["beta"].as<double>();
        }
    } else if (methodName == "simple_randomized") {
        // No parameters for simple randomized construction
    } else if (methodName == "grasp") {
        json["n_trials"] = vm["n-trials"].as<int>();
        json["alpha"] = vm["alpha"].as<double>();
        json["beta"] = vm["beta"].as<double>();
        json["start_with_best"] = vm.count("start-with-best") > 0 ? "true" : "false";
        json["time_limit"] = vm["time-limit"].as<int>();
        json["constructive_heuristic"] = vm["constructive-heuristic"].as<std::string>();
        if (vm.count("local-searches")) {
            boost::json::array local_searches_array;
            const auto& searches = vm["local-searches"].as<std::vector<std::string>>();
            for (const auto& search : searches) {
                local_searches_array.push_back(boost::json::value(search));
            }
            json["local_searches"] = local_searches_array;
        }
    }
    json["tour"] = tourToString(tour);
    json["cost"] = cost;
    json["wall_time"] = wallTime;
    json["timestamp"] = timestamp;

    std::string outputDir = vm["output-dir"].as<std::string>();
    if (!std::filesystem::exists(outputDir)) {
        std::filesystem::create_directories(outputDir);
    }

    std::string outputPath =
        outputDir + "/" + instance->getName() + "_" + std::to_string(param_hash) + ".json";
    std::ofstream outputFile;
    if (std::filesystem::exists(outputPath)) {
        outputFile.open(outputPath, std::ios::app);  // Open in append mode
    } else {
        outputFile.open(outputPath);  // Open in default mode (write)
    }
    if (!outputFile.is_open()) {
        std::cerr << "Error: Could not open output file: " << outputPath << std::endl;
        return 1;
    }

    outputFile << std::setw(4) << json << std::endl;
    outputFile.close();

    return 0;
}
