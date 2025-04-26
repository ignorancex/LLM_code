#include "Logger.hpp"

#include <iostream>
#include <ctime>

std::string time_string() {
    const std::time_t time = std::time({});
    char timeString[std::size("[yyyy-mm-ddThh:mm:ss+0200]")];
    std::strftime(std::data(timeString), std::size(timeString),
                  "[%FT%T%z]", std::localtime(&time));
    return std::string {timeString};
}

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
        std::cout << time_string() << " " << str << std::endl;
    } else if (error) {
        std::cerr << time_string() << " " << str << std::endl;
    }
}

void Logger::err(const std::string &str) const {
    log(str, true);
}