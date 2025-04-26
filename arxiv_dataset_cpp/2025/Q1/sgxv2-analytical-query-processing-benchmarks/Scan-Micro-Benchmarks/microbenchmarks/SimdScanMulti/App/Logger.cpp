#include "Logger.hpp"

#include <iostream>
#include "gflags/gflags.h"

DECLARE_bool(debug);

void logger(const std::string &str, bool error) {
    if (FLAGS_debug && !error) {
        std::cout << str << std::endl;
    } else if (error) {
        std::cerr << str << std::endl;
    }
}

void error(const std::string &str) {
    logger(str, true);
}