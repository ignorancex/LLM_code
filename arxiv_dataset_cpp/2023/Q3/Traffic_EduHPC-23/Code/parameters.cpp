/// @file parameters.cpp

#include "parameters.h"
#include <iostream>
#include <fstream>

/// @brief Return a set of default parameters
Parameters default_parameters()
{
    Parameters result;
    result.L = 500;
    result.T = 500;
    result.N = 300;
    result.p = 0.2;
    result.vmax = 2;
    result.seed = 13;
    result.per = 1;
    result.outputprefix = "traffic";
    return result;
}

/// @brief Read parameters from a file. Missing parameters are set to
///        default values using default_parameters.
Parameters read_parameters(const std::string& inifilename)
{
    Parameters result = default_parameters();
    std::ifstream infile(inifilename);
    std::string line;
    while (std::getline(infile, line)) {
        size_t equalsignpos = line.find('=');
        if (equalsignpos != std::string::npos) {
            std::string key = line.substr(0, equalsignpos);
            std::string value = line.substr(equalsignpos+1);
            if (key == "L") result.L = std::stoi(value);
            else if (key == "T") result.T = std::stoi(value);
            else if (key == "N") result.N = std::stoi(value);
            else if (key == "p") result.p = std::stod(value);
            else if (key == "vmax") result.vmax = std::stoi(value);
            else if (key == "seed") result.seed = std::stoi(value);
            else if (key == "per") result.per = std::stoi(value);
            else if (key == "outputprefix") result.outputprefix = value;
        }
    }
    if (result.N>result.L) {
        std::cerr << "Error: Too many cars!\n";
        exit(1);
    }
    return result;
}


/// @brief Print parameters
void print_parameters(const Parameters& params)
{
    std::cout << " L=" << params.L;
    std::cout << " T=" << params.T;
    std::cout << " N=" << params.N;
    std::cout << " p=" << params.p;
    std::cout << " vmax=" << params.vmax;
    std::cout << " seed=" << params.seed;
    std::cout << " per=" << params.per;
    std::cout << " outputprefix=" << params.outputprefix;
    std::cout << std::endl;
}

