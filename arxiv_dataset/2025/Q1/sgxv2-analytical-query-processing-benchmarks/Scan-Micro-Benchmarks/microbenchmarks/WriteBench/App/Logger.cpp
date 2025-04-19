#include "Logger.hpp"

#include <iostream>

void logger(const std::string &str, bool error) {
    if (!error) {
        std::cout << str << std::endl;
    } else if (error) {
        std::cerr << str << std::endl;
    }
}

void error(const std::string &str) {
    logger(str, true);
}

Logger::Logger(bool _debug) : debug{_debug} {}

void Logger::log(const std::string &str, bool error) const {
    if (debug && !error) {
        std::cout << str << std::endl;
    } else if (error) {
        std::cerr << str << std::endl;
    }
}

void Logger::err(const std::string &str) const {
    log(str, true);
}