#include "npj/no_partitioning_bucket_chaining_join.hpp"
#include <vector>
#include "Logger.hpp"

#include "rdtscpWrapper.h"

#ifndef ENCLAVE
#include <malloc.h>
#endif

#define HASH_BIT_MODULO(K, MASK, NBITS) (((K) & MASK) >> NBITS)

#ifndef NEXT_POW_2
/**
 *  compute the next number, greater than or equal to 32-bit unsigned v.
 *  taken from "bit twiddling hacks":
 *  http://graphics.stanford.edu/~seander/bithacks.html
 */
#define NEXT_POW_2(V)                           \
    do {                                        \
        V--;                                    \
        V |= V >> 1;                            \
        V |= V >> 2;                            \
        V |= V >> 4;                            \
        V |= V >> 8;                            \
        V |= V >> 16;                           \
        V++;                                    \
    } while(0)
#endif

constexpr uint64_t CPMS = CYCLES_PER_MICROSECOND;

/** print out the execution time statistics of the join */
static void
print_timing(uint64_t total, uint64_t build, uint64_t probe, uint64_t numtuples, int64_t result)
{
    double cyclestuple = (double) total / (double) numtuples;
    uint64_t time_usec_tsc = total / CPMS;
    double throughput =  (double) numtuples / (double) time_usec_tsc;
    logger(DBG, "Total input tuples : %lu", numtuples);
    logger(DBG, "Result tuples : %lu", result);
    logger(DBG, "Total Join Time (cycles)    : %lu", total);
    logger(DBG, "Build (cycles)              : %lu", build);
    logger(DBG, "Join (cycles)               : %lu", probe);
    logger(DBG, "Cycles-per-tuple : %.4lf", cyclestuple);
    logger(DBG, "Total RT from TSC (us) : %lu ", time_usec_tsc);
    logger(DBG, "Throughput (M rec/sec) : %.2lf", throughput);
}

result_t* NPBC_st(const table_t *relR, const table_t *relS, const joinconfig_t *config) {
    uint64_t probe_timer;
    uint64_t build_timer;
    uint64_t total_timer;

    const uint64_t numR = relR->num_tuples;
    uint32_t N = numR;
    NEXT_POW_2(N);

    int64_t matches = 0;
    const uint32_t MASK = N - 1;

    std::vector<uint32_t> next(numR);
    std::vector<uint32_t> bucket(N);

    uint64_t build_start_time = rdtscp_s();
#ifdef UNROLL
    uint32_t i = 0;
    while (i + 4 < numR) {
        uint32_t idx = HASH_BIT_MODULO(relR->tuples[i].key, MASK, 0);
        uint32_t idx1 = HASH_BIT_MODULO(relR->tuples[i+1].key, MASK, 0);
        uint32_t idx2 = HASH_BIT_MODULO(relR->tuples[i+2].key, MASK, 0);
        uint32_t idx3 = HASH_BIT_MODULO(relR->tuples[i+3].key, MASK, 0);
        next[i] = bucket[idx];
        bucket[idx] = ++i;     /* we start pos's from 1 instead of 0 */
        next[i] = bucket[idx1];
        bucket[idx1] = ++i;     /* we start pos's from 1 instead of 0 */
        next[i] = bucket[idx2];
        bucket[idx2] = ++i;     /* we start pos's from 1 instead of 0 */
        next[i] = bucket[idx3];
        bucket[idx3] = ++i;     /* we start pos's from 1 instead of 0 */
    }
    while (i < numR) {
        uint32_t idx = HASH_BIT_MODULO(relR->tuples[i].key, MASK, 0);
        next[i] = bucket[idx];
        bucket[idx] = ++i;     /* we start pos's from 1 instead of 0 */
    }
#else
    for (uint32_t i = 0; i < numR;) {
        uint32_t idx = HASH_BIT_MODULO(relR->tuples[i].key, MASK, 0);
        next[i] = bucket[idx];
        bucket[idx] = ++i;     /* we start pos's from 1 instead of 0 */
    }
#endif
    uint64_t in_between_time = rdtscp_s();
    build_timer = in_between_time - build_start_time;

    /* BUILD-LOOP END */
    /* PROBE-LOOP */
    // TODO: implement materialization
    const uint64_t numS = relS->num_tuples;
    const struct row_t *const Stuples = relS->tuples;
    const struct row_t *const Rtuples = relR->tuples;
    for (uint32_t i = 0; i < numS; i++) {
        uint32_t idx = HASH_BIT_MODULO(Stuples[i].key, MASK, 0);
        for (uint32_t hit = bucket[idx]; hit > 0; hit = next[hit - 1]) {
            if (Stuples[i].key == Rtuples[hit - 1].key) {
                matches++;
#ifdef JOIN_MATERIALIZE
                insert_output(output, Stuples[i].key, Rtuples[hit-1].payload, Stuples[i].payload);
#endif
            }
        }
    }
    /* PROBE-LOOP END  */
    auto end_time = rdtscp_s();
    probe_timer = end_time - in_between_time;
    total_timer = end_time - build_start_time;

    print_timing(total_timer, build_timer, probe_timer, relR->num_tuples + relS->num_tuples, matches);

    result_t * joinresult;
    joinresult = (result_t *) malloc(sizeof(result_t));
    joinresult->totalresults = matches;
    joinresult->nthreads = 1;
    return joinresult;
}