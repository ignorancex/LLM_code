#include "Logger.hpp"
#include "rdtscpWrapper.h"
#include <cstdarg>
#include <cstdio>

#ifdef ENCLAVE
#include "EnclavePrint.hpp"
#else
#endif

constexpr uint64_t CPMS = CYCLES_PER_MICROSECOND;

unsigned long long start_time;

static const char* level_strings[] = {
        "INFO",
        "WARN",
        "ERROR",
        "ENCL",
        "DEBUG",
        "PCM"
};

static char const * get_log_color(LEVEL level) {
    switch (level) {
        case INFO:
            return ANSI_COLOR_GREEN;
        case WARN:
            return ANSI_COLOR_YELLOW;
        case ERROR:
            return ANSI_COLOR_RED;
        case DBG:
            return ANSI_COLOR_WHITE;
        case PCMLOG:
            return ANSI_COLOR_BLUE;
        default:
            return "";
    }
}

void initLogger(std::optional<uint64_t> start) {
    if (start.has_value()) {
        start_time = start.value();
    } else {
      start_time = rdtscp_s();
    }
}

uint64_t logger_get_start() {
    return start_time;
}

void logger(LEVEL level, const char *fmt, ...) {
    char buffer[BUFSIZ] = { '\0' };
    const char* color;
    va_list args;
    double time;

#ifndef DEBUG
    if (level == DBG) return;
#endif

    va_start(args, fmt);
    vsnprintf(buffer, BUFSIZ, fmt, args);

    color = get_log_color(level);

    auto current_time = rdtscp_s();
    time = static_cast<double>((current_time - start_time) / CPMS) / 1'000'000.0;

    printf("%s[%8.4f][%5s] %s" ANSI_COLOR_RESET "\n",
           color,
           time,
           level_strings[level],
           buffer);
}