#include "psm/parallel_sortmerge_join.h"

#include "psm/parallel_sort.h"
#include <stdlib.h>
#include "psm/utility.h"

#include "Logger.hpp"
#ifdef ENCLAVE
#include "ocalls_t.h"
#else
#include "ocalls.hpp"
#endif

constexpr uint64_t CPMS = CYCLES_PER_MICROSECOND;

void print_timing(uint64_t total,
                  uint64_t join,
                  uint64_t partition,
                  uint64_t num_tuples,
                  int64_t matches,
                  uint64_t start,
                  uint64_t end) {
    double cycles_per_tuple = (double) total / (double) num_tuples;
    uint64_t time_usec_tsc = total / CPMS;
    uint64_t time_usec = end - start;
    double throughput = num_tuples / (double) time_usec_tsc;
    logger(INFO, "Total input tuples : %u", num_tuples);
    logger(INFO, "Result tuples : %lu", matches);
    logger(INFO, "Total Join Time (cycles)    : %lu", total);
    logger(INFO, "Partition Overall (cycles)  : %lu", partition);
    logger(INFO, "Build+Join Overall (cycles) : %lu", join);
    logger(INFO, "Cycles-per-tuple            : %.4lf", cycles_per_tuple);
    logger(INFO, "Cycles-per-tuple-partition  : %.4lf", (double) partition / num_tuples);
    logger(INFO, "Cycles-per-tuple-join      : %.4lf", (double) join / num_tuples);
    logger(INFO, "Total Runtime (us) : %lu ", time_usec);
    logger(INFO, "Total RT from TSC (us) : %lu ", time_usec_tsc);
    logger(INFO, "Throughput (M rec/sec) : %.2lf", throughput);
}

static int64_t merge(const relation_t *relR, const relation_t *relS)
{
    int64_t matches = 0;
    row_t *tr = relR->tuples;
    row_t *ts;
    row_t *gs = relS->tuples;
    std::size_t tr_s = 0;
    std::size_t gs_s = 0;
    logger(DBG, "Merge1");
    while ( tr_s < relR->num_tuples && gs_s < relS->num_tuples)
    {
        while (tr->key < gs->key)
        {
            tr++; tr_s++;
        }
        while (tr->key > gs->key)
        {
            gs++; gs_s++;
        }
        ts = gs;
        while (tr->key == gs->key)
        {
            ts = gs;
            while (ts->key == tr->key)
            {
                matches++;
                ts++;
            }
            tr++; tr_s++;
        }
        gs = ts;
    }
    logger(DBG, "Merge2");
    return matches;
}

result_t *PSM(const table_t *relR,  const table_t *relS, const joinconfig_t *config) {
//    bool is_sorted;
//    std::size_t position = 0L;
    auto nthreads = config->NTHREADS;
    int64_t matches = 0L;
    result_t *result = static_cast<result_t*> (malloc(sizeof(result_t)));
    uint64_t start = 0;
    uint64_t end = 0;
    uint64_t sort_timer = 0;
    uint64_t merge_timer = 0;
    uint64_t total_timer = 0;
    std::size_t sizeR = relR->num_tuples;
    std::size_t sizeS = relS->num_tuples;
    ocall_get_system_micros(&start);
    ocall_startTimer(&total_timer);

    /* SORT PHASE */
    ocall_startTimer(&sort_timer);
    if (!relR->sorted)
    {
        internal::parallel_sort(&relR->tuples[0], &relR->tuples[sizeR], nthreads);
    }
    logger(DBG, "R sorted");
    if (!relS->sorted)
    {
        internal::parallel_sort(&relS->tuples[0], &relS->tuples[sizeS], nthreads);
    }
    logger(DBG, "S sorted");
    ocall_stopTimer(&sort_timer);

    /* MERGE PHASE */
    ocall_startTimer(&merge_timer);

    matches = merge(relR, relS);

    ocall_stopTimer(&merge_timer);
    ocall_stopTimer(&total_timer);
    ocall_get_system_micros(&end);
    print_timing(total_timer, merge_timer, sort_timer, (uint64_t) (sizeR + sizeS), matches, start, end);
    result->totalresults = matches;
    result->nthreads = nthreads;
    return result;
}