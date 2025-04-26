#include "time_print.hpp"
#include "Logger.hpp"

#ifndef CYCLES_PER_MICROSECOND
#error "Define CYCLES_PER_MICROSECOND"
#else
constexpr uint64_t CPMS = CYCLES_PER_MICROSECOND;
#endif

TPCHTimers
timers_to_us(const TPCHTimers &src) {
    return TPCHTimers{src.selection_1 / CPMS, src.selection_2 / CPMS, src.selection_3 / CPMS, src.join_1 / CPMS,
                      src.join_2 / CPMS, src.join_3 / CPMS,
                      src.copy / CPMS, src.total / CPMS};
}

void
print_query_results(const TPCHTimers &timers, uint64_t numTuples) {
    logger(INFO, "QueryTimeTotal (us)         : %u", timers.total);
    auto selection_time = timers.selection_1 + timers.selection_2 + timers.selection_3;
    logger(INFO, "QueryTimeSelection (us)     : %u (%.2lf%%)", selection_time,
           (100 * (double) selection_time / (double) timers.total));
    logger(INFO, "QueryTimeSelection 1 (us)   : %u", timers.selection_1);
    logger(INFO, "QueryTimeSelection 2 (us)   : %u", timers.selection_2);
    logger(INFO, "QueryTimeSelection 3 (us)   : %u", timers.selection_3);
    auto join_time = timers.join_1 + timers.join_2 + timers.join_3;
    logger(INFO, "QueryTimeJoin (us)          : %u (%.2lf%%)", join_time,
           (100 * (double) join_time / (double) timers.total));
    logger(INFO, "QueryTimeCopy (us)          : %u (%.2lf%%)", timers.copy,
           (100 * (double) timers.copy / (double) timers.total));
    logger(INFO, "QueryTimeJoin 1 (us)         : %u", timers.join_1);
    logger(INFO, "QueryTimeJoin 2 (us)         : %u", timers.join_2);
    logger(INFO, "QueryTimeJoin 3 (us)         : %u", timers.join_3);
    logger(INFO, "QueryThroughput (M rec/s)   : %.4lf",
           static_cast<double>(numTuples) / static_cast<double>(timers.total));
}
