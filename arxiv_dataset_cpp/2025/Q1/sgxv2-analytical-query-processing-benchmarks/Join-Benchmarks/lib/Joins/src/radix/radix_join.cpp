#include "radix/radix_join.h"
#include "radix/prj_params.h"
#include "util.hpp"
#include "Logger.hpp"
#include "data-types.h"
#include "Barrier.hpp"
#include "rdtscpWrapper.h"
#include <boost/lockfree/queue.hpp>
#include <pthread.h>
#include <algorithm>

#ifndef ENCLAVE
#include <immintrin.h>
#include "ocalls.hpp"
#else

#include "ocalls_t.h"
#include "emmintrin.h"
#include "smmintrin.h"
#include "tmmintrin.h"
#include "xmmintrin.h"
#include "avxintrin.h"
#include "avx2intrin.h"
#include "avx512fintrin.h"
#include "avx512cdintrin.h"
#include "avx512vlintrin.h"
#include "avx512bwintrin.h"
#include "avx512cdintrin.h"
#include "avx512fintrin.h"
#include "avx512vbmiintrin.h"
#include "avx512vbmi2intrin.h"
#include "avx512dqintrin.h"
#include "avx512vldqintrin.h"

#endif

#ifdef CHUNKED_TABLE
#include "ChunkedTable.hpp"
#endif

#ifdef MUTEX_QUEUE

#include "task_queue.h"

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

#ifndef TUPLESPERCACHELINE
#define TUPLESPERCACHELINE (CACHE_LINE_SIZE/sizeof(tuple_t))
#endif

/**
 * @defgroup SoftwareManagedBuffer Optimized Partitioning Using SW-buffers
 * @{
 */
typedef union {
    struct {
        tuple_t tuples[CACHE_LINE_SIZE / sizeof(tuple_t)];
    } tuples;
    struct {
        tuple_t tuples[CACHE_LINE_SIZE / sizeof(tuple_t) - 1];
        int32_t slot;
    } data;
} cache_line_t;

#ifndef MUTEX_QUEUE
struct task_t {
    struct table_t relR;
    struct table_t tmpR;
    struct table_t relS;
    struct table_t tmpS;
};
#endif

struct radix_timers_t {
    uint64_t partitioning_total_timer;
    uint64_t partitioning_pass_1_timer;
    uint64_t partitioning_pass_1_r_timer;
    uint64_t partitioning_pass_1_s_timer;
    uint64_t partitioning_pass_1_hist_timer;
    uint64_t partitioning_pass_1_copy_timer;
    uint64_t partitioning_pass_2_timer;
    uint64_t partitioning_pass_2_hist_timer;
    uint64_t partitioning_pass_2_copy_timer;
    uint64_t join_total_timer;
    uint64_t build_in_depth_timer;
    uint64_t join_in_depth_timer;
    uint64_t total_timer;
};

using queue = boost::lockfree::queue<task_t, boost::lockfree::fixed_sized<true>>;

/** holds the arguments passed to each thread */
struct arg_t_radix {
    uint32_t **histR;
    const row_t *relR;
    row_t *tmpR;
    row_t *tmpR2;
    uint32_t **histS;
    const row_t *relS;
    row_t *tmpS;
    row_t *tmpS2;

    uint64_t numR;
    uint64_t numS;
    uint64_t totalR;
    uint64_t totalS;

    uint32_t num_radix_bits;
    uint32_t num_passes;

#ifndef MUTEX_QUEUE
    queue *join_queue;
    queue *part_queue;
#else
    task_queue_t *join_queue;
    task_queue_t *part_queue;
#endif
    Barrier *barrier;
    JoinFunction join_function;
    int64_t result;
    int32_t my_tid;
    int nthreads;

    /* stats about the thread */
    int32_t parts_joined;
    int32_t parts_partitioned;
    radix_timers_t timers;

    /* results of the thread */
    int materialize;
#ifdef CHUNKED_TABLE
    chunked_table_t *thread_result_table;
#else
    threadresult_t *threadresult;
#endif
} __attribute__((aligned(CACHE_LINE_SIZE)));

/** holds arguments passed for partitioning */
struct part_t {
    const row_t *rel;
    row_t *tmp;
    uint32_t **hist;
    uint32_t *output;
    arg_t_radix *thrargs;
    uint64_t num_tuples;
    uint64_t total_tuples;
    int32_t R;
    uint32_t D;
    int relidx;  /* 0: R, 1: S */
    uint32_t padding;
    uint64_t *hist_timer;
    uint64_t *copy_timer;
} __attribute__((aligned(CACHE_LINE_SIZE)));

static void *
alloc_aligned(size_t size) {
    void *ret = aligned_alloc(CACHE_LINE_SIZE, size);
    malloc_check(ret);
    return ret;
}

constexpr uint64_t CPMS = CYCLES_PER_MICROSECOND;

void print_timing(uint64_t join_runtime,
                  uint64_t join,
                  uint64_t partition,
                  uint64_t num_tuples,
                  int64_t result,
                  uint64_t start,
                  uint64_t end,
                  uint64_t pass1,
                  uint64_t pass2,
                  uint64_t build_in_depth,
                  uint64_t join_in_depth) {
    double cycles_per_tuple = (double) join_runtime / (double) num_tuples;
    uint64_t join_runtime_mus = join_runtime / CPMS;
    uint64_t total_runtime_mus = (end - start) / CPMS;
    double throughput = num_tuples / (double) join_runtime_mus;
    logger(INFO, "Total input tuples : %u", num_tuples);
    logger(INFO, "Result tuples : %lu", result);
    logger(INFO, "Total Join Time (cycles)    : %lu", join_runtime);
    logger(INFO, "Partition Overall (cycles)  : %lu", partition);
    logger(INFO, "Partition Pass One (cycles) : %lu", pass1);
    logger(INFO, "Partition Pass Two (cycles) : %lu", pass2);
    logger(INFO, "Build+Join Overall (cycles) : %lu", join);
    logger(INFO, "Build (cycles)              : %lu", build_in_depth);
    logger(INFO, "Join (cycles)               : %lu", join_in_depth);
    logger(INFO, "Cycles-per-tuple            : %.4lf", cycles_per_tuple);
    logger(INFO, "Cycles-per-tuple-partition  : %.4lf", (double) partition / num_tuples);
    logger(INFO, "Cycles-per-tuple-partitioning_pass_1_timer     : %.4lf", (double) pass1 / num_tuples);
    logger(INFO, "Cycles-per-tuple-partitioning_pass_2_timer     : %.4lf", (double) pass2 / num_tuples);
    logger(INFO, "Cycles-per-tuple-join      : %.4lf", (double) join / num_tuples);
    logger(INFO, "Pure Join Runtime (us) : %lu ", join_runtime_mus);
    logger(INFO, "Throughput (M rec/sec) : %.2lf", throughput);
    logger(INFO, "Total Runtime (us)     : %lu ", total_runtime_mus);
}

void print_timing(const radix_timers_t &timers,
                  uint64_t start,
                  uint64_t end,
                  uint64_t num_tuples,
                  int64_t result) {
    double cycles_per_tuple = (double) timers.total_timer / (double) num_tuples;
    uint64_t join_runtime_mus = timers.total_timer / CPMS;
    uint64_t total_runtime_mus = (end - start) / CPMS;
    double throughput = num_tuples / (double) join_runtime_mus;
    logger(INFO, "Total input tuples : %u", num_tuples);
    logger(INFO, "Result tuples : %lu", result);
    logger(INFO, "Total Join Time (cycles)    : %lu", timers.total_timer);
    logger(INFO, "Partition Overall (cycles)  : %lu", timers.partitioning_total_timer);
    logger(INFO, "Partition Pass One (cycles) : %lu", timers.partitioning_pass_1_timer);
    logger(INFO, "Partition R        (cycles) : %lu", timers.partitioning_pass_1_r_timer);
    logger(INFO, "Partition S        (cycles) : %lu", timers.partitioning_pass_1_s_timer);
    logger(INFO, "Partition Pass Two (cycles) : %lu", timers.partitioning_pass_2_timer);
    logger(INFO, "Partition Two Hist (cycles) : %lu", timers.partitioning_pass_2_hist_timer);
    logger(INFO, "Partition Two Copy (cycles) : %lu", timers.partitioning_pass_2_copy_timer);
    logger(INFO, "Build+Join Overall (cycles) : %lu", timers.join_total_timer);
    logger(INFO, "Build (cycles)              : %lu", timers.build_in_depth_timer);
    logger(INFO, "Join (cycles)               : %lu", timers.join_in_depth_timer);
    logger(INFO, "Cycles-per-tuple            : %.4lf", cycles_per_tuple);
    logger(INFO, "Cycles-per-tuple-partition  : %.4lf", (double) timers.partitioning_total_timer / num_tuples);
    logger(INFO, "Cycles-per-tuple-partitioning_pass_1_timer     : %.4lf",
           (double) timers.partitioning_pass_1_timer / num_tuples);
    logger(INFO, "Cycles-per-tuple-partitioning_pass_2_timer     : %.4lf",
           (double) timers.partitioning_pass_2_timer / num_tuples);
    logger(INFO, "Cycles-per-tuple-join      : %.4lf", (double) timers.join_total_timer / num_tuples);
    logger(INFO, "Pure Join Runtime (us) : %lu ", join_runtime_mus);
    logger(INFO, "Throughput (M rec/sec) : %.2lf", throughput);
    logger(INFO, "Total Runtime (us)     : %lu ", total_runtime_mus);
}

void print_timing(const radix_timers_t &timers,
                  uint64_t start,
                  uint64_t end,
                  uint64_t num_tuples,
                  int64_t result,
             uint64_t preparation_time, uint64_t join_time, uint64_t free_time) {
    double cycles_per_tuple = (double) timers.total_timer / (double) num_tuples;
    uint64_t join_runtime_mus = timers.total_timer / CPMS;
    uint64_t total_runtime_mus = (end - start) / CPMS;
    preparation_time /= CPMS;
    join_time /= CPMS;
    free_time /= CPMS;
    double throughput = num_tuples / (double) join_runtime_mus;
    logger(INFO, "Total input tuples : %u", num_tuples);
    logger(INFO, "Result tuples : %lu", result);
    logger(INFO, "Total Join Time (cycles)    : %lu", timers.total_timer);
    logger(INFO, "Partition Overall (cycles)  : %lu", timers.partitioning_total_timer);
    logger(INFO, "Partition Pass One (cycles) : %lu", timers.partitioning_pass_1_timer);
    logger(INFO, "Partition R        (cycles) : %lu", timers.partitioning_pass_1_r_timer);
    logger(INFO, "Partition S        (cycles) : %lu", timers.partitioning_pass_1_s_timer);
    logger(INFO, "Partition One Hist (cycles) : %lu", timers.partitioning_pass_1_hist_timer);
    logger(INFO, "Partition One Copy (cycles) : %lu", timers.partitioning_pass_1_copy_timer);
    logger(INFO, "Partition Pass Two (cycles) : %lu", timers.partitioning_pass_2_timer);
    logger(INFO, "Partition Two Hist (cycles) : %lu", timers.partitioning_pass_2_hist_timer);
    logger(INFO, "Partition Two Copy (cycles) : %lu", timers.partitioning_pass_2_copy_timer);
    logger(INFO, "Build+Join Overall (cycles) : %lu", timers.join_total_timer);
    logger(INFO, "Build (cycles)              : %lu", timers.build_in_depth_timer);
    logger(INFO, "Join (cycles)               : %lu", timers.join_in_depth_timer);
    logger(INFO, "Cycles-per-tuple            : %.4lf", cycles_per_tuple);
    logger(INFO, "Cycles-per-tuple-partition  : %.4lf", (double) timers.partitioning_total_timer / num_tuples);
    logger(INFO, "Cycles-per-tuple-partitioning_pass_1_timer     : %.4lf",
           (double) timers.partitioning_pass_1_timer / num_tuples);
    logger(INFO, "Cycles-per-tuple-partitioning_pass_2_timer     : %.4lf",
           (double) timers.partitioning_pass_2_timer / num_tuples);
    logger(INFO, "Cycles-per-tuple-join      : %.4lf", (double) timers.join_total_timer / num_tuples);
    logger(INFO, "Pure Join Runtime (us) : %lu ", join_runtime_mus);
    logger(INFO, "Preparation Time (us) : %lu ", preparation_time);
    logger(INFO, "Join time arg + join time (us) : %lu ", join_time);
    logger(INFO, "Free time (us) : %lu ", free_time);
    logger(INFO, "Throughput (M rec/sec) : %.2lf", throughput);
    logger(INFO, "Total Runtime (us)     : %lu ", total_runtime_mus);
}

[[nodiscard]] uint32_t calc_num_radix_bits(uint64_t num_r, uint64_t num_s, size_t nthreads) {
#ifndef MAX_PARTITIONS
    uint64_t total_tuples = num_r;
#ifdef CACHE_DIVISOR
    uint64_t max_tuples_in_cache = L2_CACHE_TUPLES / CACHE_DIVISOR;
#else
    uint64_t max_tuples_in_cache = L2_CACHE_TUPLES / 4; // 4 turned out to be best in most cases
#endif
    // Integer division rounding up
    uint64_t num_required_partitions = (total_tuples + max_tuples_in_cache - 1) / max_tuples_in_cache;
    num_required_partitions = std::max(num_required_partitions, nthreads);
#else
    // Create as many partitions as there are threads
    uint64_t num_required_partitions = nthreads;
#endif

    uint32_t radix_bits = 0;
    while ((1 << radix_bits) < num_required_partitions) {
        ++radix_bits;
    }

    return radix_bits;
}

[[nodiscard]] uint32_t calc_num_passes(uint32_t num_radix_bits) {
    // make sure that the histogram used for partitioning fits into the L1 cache
    constexpr auto space_in_l1 = 15 - 2; // 32 KB cache divided by 4 byte integers, if bigger integers are used to index
    // during partitioning, decrease this value

    if (num_radix_bits <= space_in_l1) {
        return 1;
    } else {
        return 2;
    }
}

[[nodiscard]] inline uint32_t fanout_pass_1(uint32_t radix_bits, uint32_t num_passes) {
    return 1 << (radix_bits / num_passes);
}

[[nodiscard]] inline uint32_t fanout_pass_2(uint32_t radix_bits, uint32_t num_passes) {
    return 1 << (radix_bits - radix_bits / num_passes);
}

[[nodiscard]] inline uint32_t padding_tuples(uint32_t radix_bits, uint32_t num_passes) {
    return SMALL_PADDING_TUPLES * (fanout_pass_2(radix_bits, num_passes) + 1);
}

[[nodiscard]] inline uint32_t relation_padding(uint32_t radix_bits, uint32_t num_passes) {
    return padding_tuples(radix_bits, num_passes) * fanout_pass_1(radix_bits, num_passes) * sizeof(struct row_t);
}

/**
 *  This algorithm builds the hashtable using the bucket chaining idea and used
 *  in PRO implementation. Join between given two relations is evaluated using
 *  the "bucket chaining" algorithm proposed by Manegold et al. It is used after
 *  the partitioning phase, which is common for all algorithms. Moreover, R and
 *  S typically fit into L2 or at least R and |R|*sizeof(int) fits into L2 cache.
 *
 * @param R input relation R (build)
 * @param S input relation S (probe)
 *
 * @return number of result tuples
 */
int64_t
bucket_chaining_join(const table_t *const R,
                     const table_t *const S,
                     table_t *const tmpR,
                     uint32_t num_radix_bits,
#ifdef CHUNKED_TABLE
                     chunked_table_t *output,
#else
                     output_list_t **output,
#endif
                     uint64_t *build_timer,
                     uint64_t *join_timer,
                     int materialize) {
    (void) (tmpR);
    const uint64_t numR = R->num_tuples;
    uint32_t N = numR;
    NEXT_POW_2(N);

    int64_t matches = 0;
    const uint32_t MASK = (N - 1) << num_radix_bits;

    auto next = (uint32_t *) malloc(sizeof(int) * numR);
    auto bucket = (uint32_t *) calloc(N, sizeof(int));

    /* BUILD-LOOP */
    uint64_t build_start_time = rdtscp_s();
#ifdef UNROLL
    uint32_t i = 0;
    while (i + 4 < numR) {
        uint32_t idx = HASH_BIT_MODULO(R->tuples[i].key, MASK, num_radix_bits);
        uint32_t idx1 = HASH_BIT_MODULO(R->tuples[i + 1].key, MASK, num_radix_bits);
        uint32_t idx2 = HASH_BIT_MODULO(R->tuples[i + 2].key, MASK, num_radix_bits);
        uint32_t idx3 = HASH_BIT_MODULO(R->tuples[i + 3].key, MASK, num_radix_bits);
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
        uint32_t idx = HASH_BIT_MODULO(R->tuples[i].key, MASK, num_radix_bits);
        next[i] = bucket[idx];
        bucket[idx] = ++i;     /* we start pos's from 1 instead of 0 */
    }
#else
    for (uint32_t i = 0; i < numR;) {
        uint32_t idx = HASH_BIT_MODULO(R->tuples[i].key, MASK, num_radix_bits);
        next[i] = bucket[idx];
        bucket[idx] = ++i;     /* we start pos's from 1 instead of 0 */
    }
#endif
    uint64_t in_between_time = rdtscp_s();
    if (build_timer != nullptr) {
        *build_timer += in_between_time - build_start_time;\

    }
    /* BUILD-LOOP END */


    /* Disable the following loop for no-probe for the break-down experiments */
    /* PROBE-LOOP */
    const uint64_t numS = S->num_tuples;
    const struct row_t *const Stuples = S->tuples;
    const struct row_t *const Rtuples = R->tuples;

    // materialize checked first to prevent additional branch in loop
    if (!materialize) {
        for (uint32_t i = 0; i < numS; i++) {
            uint32_t idx = HASH_BIT_MODULO(Stuples[i].key, MASK, num_radix_bits);
            for (uint32_t hit = bucket[idx]; hit > 0; hit = next[hit - 1]) {
                if (Stuples[i].key == Rtuples[hit - 1].key) {
                    matches++;
                }
            }
        }
    } else {
        for (uint32_t i = 0; i < numS; i++) {
            uint32_t idx = HASH_BIT_MODULO(Stuples[i].key, MASK, num_radix_bits);
            for (uint32_t hit = bucket[idx]; hit > 0; hit = next[hit - 1]) {
                if (Stuples[i].key == Rtuples[hit - 1].key) {
                    matches++;
                    insert_output(output, Stuples[i].key, Rtuples[hit - 1].payload, Stuples[i].payload);
                }
            }
        }
    }
    if (join_timer != nullptr) {
        *join_timer += rdtscp_s() - in_between_time;
    }
    /* PROBE-LOOP END  */

    /* clean up temp */
    free(bucket);
    free(next);

    return matches;
}

/** computes and returns the histogram size for join */
[[nodiscard]] inline uint32_t get_hist_size(uint32_t relSize) __attribute__((always_inline));

[[nodiscard]] inline uint32_t get_hist_size(uint32_t relSize) {
    NEXT_POW_2(relSize);
    relSize >>= 2;
    if (relSize < 4) relSize = 4;
    return relSize;
}

/**
 * Histogram-based hash table build method together with relation re-ordering as
 * described by Kim et al. It joins partitions Ri, Si of relations R & S.
 * This is version is not optimized with SIMD and prefetching. The parallel
 * radix join implementation using this function is PRH.
 */
int64_t
histogram_join(const table_t *const R,
               const table_t *const S,
               table_t *const tmpR,
               uint32_t num_radix_bits,
#ifdef CHUNKED_TABLE
                chunked_table_t *output,
#else
               output_list_t **output,
#endif
               uint64_t *build_timer,
               uint64_t *join_timer,
               int materialize) {
    const tuple_t *const Rtuples = R->tuples;
    const uint64_t numR = R->num_tuples;
    uint32_t Nhist = get_hist_size(numR);
    const uint32_t MASK = (Nhist - 1) << num_radix_bits;

    /* HISTOGRAM CREATION */
    uint64_t build_start_time = rdtscp_s();
    auto hist = (int32_t *) calloc(Nhist + 2, sizeof(int32_t));
#ifndef UNROLL
    for (uint32_t i = 0; i < numR; i++) {
        uint32_t idx = HASH_BIT_MODULO(Rtuples[i].key, MASK, num_radix_bits);
        ++hist[idx + 2];
    }
#else
    uint32_t i = 0;
    for (; i + 8 <= numR; i += 8) {
        uint32_t idx = HASH_BIT_MODULO(Rtuples[i].key, MASK, num_radix_bits);
        uint32_t idx1 = HASH_BIT_MODULO(Rtuples[i + 1].key, MASK, num_radix_bits);
        uint32_t idx2 = HASH_BIT_MODULO(Rtuples[i + 2].key, MASK, num_radix_bits);
        uint32_t idx3 = HASH_BIT_MODULO(Rtuples[i + 3].key, MASK, num_radix_bits);
        uint32_t idx4 = HASH_BIT_MODULO(Rtuples[i + 4].key, MASK, num_radix_bits);
        uint32_t idx5 = HASH_BIT_MODULO(Rtuples[i + 5].key, MASK, num_radix_bits);
        uint32_t idx6 = HASH_BIT_MODULO(Rtuples[i + 6].key, MASK, num_radix_bits);
        uint32_t idx7 = HASH_BIT_MODULO(Rtuples[i + 7].key, MASK, num_radix_bits);
        ++hist[idx + 2];
        ++hist[idx1 + 2];
        ++hist[idx2 + 2];
        ++hist[idx3 + 2];
        ++hist[idx4 + 2];
        ++hist[idx5 + 2];
        ++hist[idx6 + 2];
        ++hist[idx7 + 2];
    }
    for (; i < numR; ++i) {
        size_t idx = (Rtuples[i].key & MASK) >> num_radix_bits;
        ++hist[idx + 2];
    }
#endif

    /* prefix sum on histogram */
    for (uint32_t i = 2, sum = 0; i < Nhist + 2; i++) {
        sum += hist[i];
        hist[i] = sum;
    }

    /* BUILD PHASE */
    tuple_t *const tmpRtuples = tmpR->tuples;
    /* reorder tuples according to the prefix sum */
#ifndef UNROLL
    for (uint32_t i = 0; i < numR; i++) {
        uint32_t idx = HASH_BIT_MODULO(Rtuples[i].key, MASK, num_radix_bits) + 1;
        tmpRtuples[hist[idx]] = Rtuples[i];
        hist[idx]++;
    }
#else
    i = 0;
    for (; i + 4 <= numR; i += 4) {
        uint32_t idx = HASH_BIT_MODULO(Rtuples[i].key, MASK, num_radix_bits) + 1;
        uint32_t idx1 = HASH_BIT_MODULO(Rtuples[i + 1].key, MASK, num_radix_bits) + 1;
        uint32_t idx2 = HASH_BIT_MODULO(Rtuples[i + 2].key, MASK, num_radix_bits) + 1;
        uint32_t idx3 = HASH_BIT_MODULO(Rtuples[i + 3].key, MASK, num_radix_bits) + 1;

        tmpRtuples[hist[idx]] = Rtuples[i];
        ++hist[idx];
        tmpRtuples[hist[idx1]] = Rtuples[i + 1];
        ++hist[idx1];
        tmpRtuples[hist[idx2]] = Rtuples[i + 2];
        ++hist[idx2];
        tmpRtuples[hist[idx3]] = Rtuples[i + 3];
        ++hist[idx3];
    }
    for (; i < numR; ++i) {
        size_t idx = (Rtuples[i].key & MASK) >> num_radix_bits;
        tmpRtuples[hist[idx]] = Rtuples[i];
        ++hist[idx];
    }
#endif

    uint64_t in_between_time = rdtscp_s();
    if (build_timer != nullptr) {
        *build_timer += in_between_time - build_start_time;\

    }

    /* PROBE PHASE */
    int64_t match = 0;
    const uint64_t numS = S->num_tuples;
    const tuple_t *const Stuples = S->tuples;
    for (uint32_t i = 0; i < numS; ++i) {

        uint32_t idx = HASH_BIT_MODULO(Stuples[i].key, MASK, num_radix_bits);

        // TODO: implement prefetching
        // Here all data from j to end can probably be prefetched. This can probably also be speed up using
        // vectorization. At least if the histogram bins are large enough.
        int j = hist[idx];
        int end = hist[idx + 1];

        /* Scalar comparisons */
        if (materialize) {
            for (; j < end; j++) {
                if (Stuples[i].key == tmpRtuples[j].key) {
                    ++match;
                    insert_output(output, Stuples[i].key, tmpRtuples[j].payload, Stuples[i].payload);
                }
            }
        } else {
            for (; j < end; j++) {
                if (Stuples[i].key == tmpRtuples[j].key) {
                    ++match;
                }
            }
        }

    }
    if (join_timer != nullptr) {
        *join_timer += rdtscp_s() - in_between_time;
    }

    /* clean up */
    free(hist);

    return match;
}

void __attribute__((noinline)) partition_hist(const row_t *rel, uint32_t size, uint32_t *my_hist,
                                              uint32_t MASK, int32_t R);

void partition_hist(const row_t *rel, const uint32_t size, uint32_t *my_hist, const uint32_t MASK, const int32_t R) {
    for (uint32_t i = 0; i < size; ++i) {
        size_t idx = (rel[i].key & MASK) >> R;
        ++my_hist[idx];
    }
}

void __attribute__((noinline)) partition_hist_unrolled(const row_t *rel, uint32_t size, uint32_t *my_hist,
                                                       uint32_t MASK, int32_t R);

void partition_hist_unrolled(const row_t *rel, const uint32_t size, uint32_t *my_hist, const uint32_t MASK,
                             const int32_t R) {
    uint32_t i = 0;
    for (; i + 9 <= size; i += 9) {
        uint32_t idx = HASH_BIT_MODULO(rel[i].key, MASK, R);
        uint32_t idx1 = HASH_BIT_MODULO(rel[i + 1].key, MASK, R);
        uint32_t idx2 = HASH_BIT_MODULO(rel[i + 2].key, MASK, R);
        uint32_t idx3 = HASH_BIT_MODULO(rel[i + 3].key, MASK, R);
        uint32_t idx4 = HASH_BIT_MODULO(rel[i + 4].key, MASK, R);
        uint32_t idx5 = HASH_BIT_MODULO(rel[i + 5].key, MASK, R);
        uint32_t idx6 = HASH_BIT_MODULO(rel[i + 6].key, MASK, R);
        uint32_t idx7 = HASH_BIT_MODULO(rel[i + 7].key, MASK, R);
        uint32_t idx8 = HASH_BIT_MODULO(rel[i + 8].key, MASK, R);
        ++my_hist[idx];
        ++my_hist[idx1];
        ++my_hist[idx2];
        ++my_hist[idx3];
        ++my_hist[idx4];
        ++my_hist[idx5];
        ++my_hist[idx6];
        ++my_hist[idx7];
        ++my_hist[idx8];
    }
    for (; i < size; ++i) {
        size_t idx = (rel[i].key & MASK) >> R;
        ++my_hist[idx];
    }
}

void __attribute__((noinline))
partition_copy(const row_t *rel, uint32_t size, uint32_t *dst, row_t *tmp, uint32_t MASK, int32_t R);

void
partition_copy(const row_t *rel, const uint32_t size, uint32_t *dst, row_t *tmp, const uint32_t MASK, const int32_t R) {
    for (uint32_t i = 0; i < size; ++i) {
        size_t idx = (rel[i].key & MASK) >> R;
        tmp[dst[idx]] = rel[i];
        ++dst[idx];
    }
}

void __attribute__((noinline))
partition_copy_unrolled(const row_t *rel, uint32_t size, uint32_t *dst, row_t *tmp, uint32_t MASK, int32_t R);

void
partition_copy_unrolled(const row_t *rel, const uint32_t size, uint32_t *dst, row_t *tmp, const uint32_t MASK,
                        const int32_t R) {
    alignas(64) uint32_t partition_indexes[6];
    alignas(64) uint32_t copy_indexes[6];

    uint32_t i = 0;
    for (; i + 6 <= size; i += 6) {
        const auto current_data = rel + i;
        for (uint32_t j = 0; j < 6; ++j) {
            partition_indexes[j] = (current_data[j].key & MASK) >> R;
        }

        for (uint32_t j = 0; j < 6; ++j) {
            copy_indexes[j] = dst[partition_indexes[j]]++;
        }

        for (uint32_t j = 0; j < 6; ++j) {
            tmp[copy_indexes[j]] = current_data[j];
        }
    }
    for (; i < size; ++i) {
        const size_t idx = (rel[i].key & MASK) >> R;
        tmp[dst[idx]] = rel[i];
        ++dst[idx];
    }
}

/**
 * Radix clustering algorithm (originally described by Manegold et al)
 * The algorithm mimics the 2-pass radix clustering algorithm from
 * Kim et al. The difference is that it does not compute
 * prefix-sum, instead the sum (offset in the code) is computed iteratively.
 *
 * @warning This method puts padding between clusters, see
 * radix_cluster_nopadding for the one without padding.
 *
 * @param outRel [out] result of the partitioning
 * @param inRel [in] input relation
 * @param hist [out] number of tuples in each partition
 * @param R cluster bits
 * @param D radix bits per pass
 * @returns tuples per partition.
 */
void
radix_cluster(struct table_t *outRel,
              const struct table_t *inRel,
              uint32_t *hist,
              int R,
              int D,
              uint64_t *hist_timer,
              uint64_t *copy_timer) {
    uint32_t M = ((1U << D) - 1) << R;
    uint32_t offset;
    uint32_t fanOut = 1 << D;

    /* the following are fixed size when D is same for all the passes,
       and can be re-used from call to call. Allocating in this function
       just in case D differs from call to call. */
    uint32_t dst[fanOut];

    {
        rdtscpWrapper w{hist_timer};
#ifndef UNROLL
        partition_hist(inRel->tuples, inRel->num_tuples, hist, M, R);
#else
        partition_hist_unrolled(inRel->tuples, inRel->num_tuples, hist, M, R);
#endif
    }


    offset = 0;
    /* determine the start and end of each cluster depending on the counts. */
    for (uint64_t i = 0; i < fanOut; i++) {
        /* dst[i]      = outRel->tuples + offset; */
        /* determine the beginning of each partitioning by adding some
           padding to avoid L1 conflict misses during scatter. */
        dst[i] = (uint32_t) (offset + i * SMALL_PADDING_TUPLES);
        offset += hist[i];
    }

    {
        rdtscpWrapper w{copy_timer};
#ifndef UNROLL
        partition_copy(inRel->tuples, inRel->num_tuples, dst, outRel->tuples, M, R);
#else
        partition_copy_unrolled(inRel->tuples, inRel->num_tuples, dst, outRel->tuples, M, R);
#endif
    }

}


/**
 * This function implements the radix clustering of a given input
 * relations. The relations to be clustered are defined in task_t and after
 * clustering, each partition pair is added to the join_queue to be joined.
 *
 * @param task description of the relation to be partitioned
 * @param join_queue task queue to add join tasks after clustering
 */

void serial_radix_partition(task_t *const task,
#ifndef MUTEX_QUEUE
                            queue *join_queue,
#else
                            task_queue_t *join_queue,
#endif
                            const int R, const int D,
                            uint64_t *hist_timer,
                            uint64_t *copy_timer) {
    uint64_t offsetR = 0;
    uint64_t offsetS = 0;
    const int fanOut = 1 << D;
    uint32_t *outputR;
    uint32_t *outputS;

    outputR = (uint32_t *) calloc(fanOut + 1, sizeof(uint32_t));
    outputS = (uint32_t *) calloc(fanOut + 1, sizeof(uint32_t));


    radix_cluster(&task->tmpR, &task->relR, outputR, R, D, hist_timer, copy_timer);

    radix_cluster(&task->tmpS, &task->relS, outputS, R, D, hist_timer, copy_timer);

#ifdef MUTEX_QUEUE
    for (int i = 0; i < fanOut; i++) {
        if (outputR[i] > 0 && outputS[i] > 0) {
            task_t *t = task_queue_get_slot_atomic(join_queue);
            t->relR.num_tuples = outputR[i];
            t->relR.tuples = task->tmpR.tuples + offsetR + i * SMALL_PADDING_TUPLES;
            t->tmpR.tuples = task->relR.tuples + offsetR + i * SMALL_PADDING_TUPLES;
            offsetR += outputR[i];

            t->relS.num_tuples = outputS[i];
            t->relS.tuples = task->tmpS.tuples + offsetS + i * SMALL_PADDING_TUPLES;
            t->tmpS.tuples = task->relS.tuples + offsetS + i * SMALL_PADDING_TUPLES;
            offsetS += outputS[i];

            /* task_queue_copy_atomic(join_queue, &t); */
            task_queue_add_atomic(join_queue, t);
        } else {
            offsetR += outputR[i];
            offsetS += outputS[i];
        }
    }
#else
    task_t t {};
    for (int i = 0; i < fanOut; i++) {
        if (outputR[i] > 0 && outputS[i] > 0) {
            t.relR.num_tuples = outputR[i];
            t.relR.tuples = task->tmpR.tuples + offsetR + i * SMALL_PADDING_TUPLES;
            t.tmpR.tuples = task->relR.tuples + offsetR + i * SMALL_PADDING_TUPLES;
            offsetR += outputR[i];

            t.relS.num_tuples = outputS[i];
            t.relS.tuples = task->tmpS.tuples + offsetS + i * SMALL_PADDING_TUPLES;
            t.tmpS.tuples = task->relS.tuples + offsetS + i * SMALL_PADDING_TUPLES;
            offsetS += outputS[i];

            join_queue->bounded_push(t);
        } else {
            offsetR += outputR[i];
            offsetS += outputS[i];
        }
    }
#endif

    free(outputR);
    free(outputS);
}

/**
 * This function implements the parallel radix partitioning of a given input
 * relation. Parallel partitioning is done by histogram-based relation
 * re-ordering as described by Kim et al. Parallel partitioning method is
 * commonly used by all parallel radix join algorithms.
 *
 * @param part description of the relation to be partitioned
 */
void
parallel_radix_partition(part_t *const part) {
    const struct row_t *rel = part->rel;
    uint32_t **hist = part->hist;
    uint32_t *output = part->output;

    const uint32_t my_tid = part->thrargs->my_tid;
    const uint32_t nthreads = part->thrargs->nthreads;
    const uint32_t size = part->num_tuples;

    const int32_t R = part->R;
    const int32_t D = part->D;
    const uint32_t fanOut = 1 << D;
    const uint32_t MASK = (fanOut - 1) << R;
    const uint32_t padding = part->padding;

    if (my_tid == 0) {
        logger(INFO, "Radix partitioning. R=%d, D=%d, fanout=%d, MASK=%d",
               R, D, fanOut, MASK);
    }

    uint32_t dst[fanOut + 1];

    /* compute local histogram for the assigned region of rel */
    /* compute histogram */
    uint32_t *my_hist = hist[my_tid];

    uint64_t internal_hist_timer = rdtscp_s();

#ifndef UNROLL
    partition_hist(rel, size, my_hist, MASK, R);
#else
    partition_hist_unrolled(rel, size, my_hist, MASK, R);
#endif

    uint32_t sum = 0;
    /* compute local prefix sum on hist */
    for (uint32_t i = 0; i < fanOut; i++) {
        sum += my_hist[i];
        my_hist[i] = sum;
    }

    auto current_time = rdtscp_s();
    internal_hist_timer = current_time - internal_hist_timer;

    /* wait at a barrier until each thread complete histograms */
    part->thrargs->barrier->wait();
    /* barrier global sync point-1 */

    /* determine the start and end of each cluster */
    for (uint32_t i = 0; i < my_tid; i++) {
        for (uint32_t j = 0; j < fanOut; j++)
            output[j] += hist[i][j];
    }
    for (uint32_t i = my_tid; i < nthreads; i++) {
        for (uint32_t j = 1; j < fanOut; j++)
            output[j] += hist[i][j - 1];
    }

    /* Add padding tuples to output positions */
    for (uint32_t i = 0; i < fanOut; i++) {
        output[i] += i * padding; //PADDING_TUPLES;
        dst[i] = output[i];
    }
    output[fanOut] = part->total_tuples + fanOut * padding; //PADDING_TUPLES;

    current_time = rdtscp_s();
    uint64_t internal_copy_timer = current_time;
    struct row_t *tmp = part->tmp;

    /* Copy tuples to their corresponding clusters */
#ifndef UNROLL
    partition_copy(rel, size, dst, tmp, MASK, R);
#else
    partition_copy_unrolled(rel, size, dst, tmp, MASK, R);
#endif
    internal_copy_timer = rdtscp_s() - internal_copy_timer;

    *part->hist_timer += internal_hist_timer;
    *part->copy_timer += internal_copy_timer;
}

/**
 * Makes a non-temporal write of 64 bytes from src to dst.
 * Uses vectorized non-temporal stores if available, falls
 * back to assignment copy.
 *
 * @param dst
 * @param src
 *
 * @return
 */
static inline void
store_nontemp_64B(void *dst, void *src) {
    /* just copy with assignment */
    // *(cache_line_t *) dst = *(cache_line_t *) src;
    _mm512_stream_si512((__m512i *) dst, *(__m512i *) src);
}


/**
 * This function implements the parallel radix partitioning of a given input
 * relation. Parallel partitioning is done by histogram-based relation
 * re-ordering as described by Kim et al. Parallel partitioning method is
 * commonly used by all parallel radix join algorithms. However this
 * implementation is further optimized to benefit from write-combining and
 * non-temporal writes.
 *
 * @param part description of the relation to be partitioned
 */
void
parallel_radix_partition_optimized(part_t *const part) {
    const tuple_t *rel = part->rel;
    uint32_t **hist = part->hist;
    uint32_t *output = part->output;

    const uint32_t my_tid = part->thrargs->my_tid;
    const uint32_t nthreads = part->thrargs->nthreads;
    const uint32_t num_tuples = part->num_tuples;

    const int32_t R = part->R;
    const int32_t D = part->D;
    const uint32_t fanOut = 1 << D;
    const uint32_t MASK = (fanOut - 1) << R;
    const uint32_t padding = part->padding;

    /* compute local histogram for the assigned region of rel */
    /* compute histogram */
    uint32_t *my_hist = hist[my_tid];

    *part->hist_timer = rdtscp_s();

    partition_hist(rel, num_tuples, my_hist, MASK, R);

    /* compute local prefix sum on hist */
    uint32_t sum = 0;
    for (uint32_t i = 0; i < fanOut; i++) {
        sum += my_hist[i];
        my_hist[i] = sum;
    }

    /* wait at a barrier until each thread complete histograms */
    part->thrargs->barrier->wait();
    /* barrier global sync point-1 */

    /* determine the start and end of each cluster */
    for (uint32_t i = 0; i < my_tid; i++) {
        for (uint32_t j = 0; j < fanOut; j++)
            output[j] += hist[i][j];
    }
    for (uint32_t i = my_tid; i < nthreads; i++) {
        for (uint32_t j = 1; j < fanOut; j++)
            output[j] += hist[i][j - 1];
    }

    auto current_time = rdtscp_s();
    *part->hist_timer = current_time - *part->hist_timer;
    *part->copy_timer = current_time;

    /* uint32_t pre; /\* nr of tuples to cache-alignment *\/ */
    tuple_t *tmp = part->tmp;
    /* software write-combining buffer */
    cache_line_t buffer[fanOut] __attribute__((aligned(CACHE_LINE_SIZE)));

    for (uint32_t i = 0; i < fanOut; i++) {
        uint32_t off = output[i] + i * padding;
        /* pre        = (off + TUPLESPERCACHELINE) & ~(TUPLESPERCACHELINE-1); */
        /* pre       -= off; */
        output[i] = off;
        buffer[i].data.slot = off;
    }
    output[fanOut] = part->total_tuples + fanOut * padding;

    /* Copy tuples to their corresponding clusters */
    for (uint32_t i = 0; i < num_tuples; i++) {
        uint32_t idx = HASH_BIT_MODULO(rel[i].key, MASK, R);
        uint32_t slot = buffer[idx].data.slot;
        tuple_t *tup = (tuple_t *) (buffer + idx);
        uint32_t slotMod = (slot) & (TUPLESPERCACHELINE - 1);
        tup[slotMod] = rel[i];

        if (slotMod == (TUPLESPERCACHELINE - 1)) {
            /* write out 64-Bytes with non-temporal store */
            _mm512_stream_si512(reinterpret_cast<__m512i *>(tmp + slot - (TUPLESPERCACHELINE - 1)),
                                _mm512_stream_load_si512(buffer + idx));
            //memcpy(tmp + slot - (TUPLESPERCACHELINE - 1), buffer + idx, 64);
            /* writes += TUPLESPERCACHELINE; */
        }

        buffer[idx].data.slot = slot + 1;
    }
    _mm_sfence();

    /* write out the remainders in the buffer */
    for (uint32_t i = 0; i < fanOut; i++) {
        uint32_t slot = buffer[i].data.slot;
        uint32_t sz = (slot) & (TUPLESPERCACHELINE - 1);
        slot -= sz;
        for (uint32_t j = 0; j < sz; j++) {
            tmp[slot] = buffer[i].data.tuples[j];
            slot++;
        }
    }

    *part->copy_timer = rdtscp_s() - *part->copy_timer;
}

/**
 * The main thread of parallel radix join. It does partitioning in parallel with
 * other threads and during the join phase, picks up join tasks from the task
 * queue and calls appropriate JoinFunction to compute the join task.
 *
 * @param param
 *
 * @return
 */
void *
prj_thread(void *param) {
    auto args = (arg_t_radix *) param;
    int32_t my_tid = args->my_tid;

    // ocall_pin_thread(my_tid); // Pin each thread to its own core

    const int fanOut = 1 << (args->num_radix_bits / args->num_passes);
    const auto R = (args->num_radix_bits / args->num_passes);
    const auto D = (args->num_radix_bits - (args->num_radix_bits / args->num_passes));
    auto thresh1 = THRESHOLD1((unsigned long) args->nthreads) << std::max(D, R);
    uint32_t num_padding_tuples = padding_tuples(args->num_radix_bits, args->num_passes);

    if (args->my_tid == 0) {
        logger(INFO, "NUM_PASSES=%d, RADIX_BITS=%d", args->num_passes, args->num_radix_bits);
        logger(INFO, "fanOut = %d, R = %d, D = %d, thresh1 = %lu", fanOut, R, D, thresh1);
    }
    int64_t results = 0;

#ifndef MUTEX_QUEUE
    queue *part_queue = args->part_queue;
    queue *join_queue = args->join_queue;
#else
    task_queue_t *part_queue = args->part_queue;
    task_queue_t *join_queue = args->join_queue;
#endif

    auto outputR = (uint32_t *) calloc((fanOut + 1), sizeof(uint32_t));
    auto outputS = (uint32_t *) calloc((fanOut + 1), sizeof(uint32_t));
    malloc_check((void *) (outputR && outputS));

    args->histR[my_tid] = (uint32_t *) calloc(fanOut, sizeof(uint32_t));
    args->histS[my_tid] = (uint32_t *) calloc(fanOut, sizeof(uint32_t));

    /* in the first pass, partitioning is done together by all threads */

    args->parts_joined = 0;

    /* wait at a barrier until each thread starts and then start the timer */
    args->barrier->wait();
#ifndef RADIX_NO_TIMING
    uint64_t current_time = rdtscp_s();
    args->timers.total_timer = current_time;
    args->timers.partitioning_total_timer = current_time;
    args->timers.partitioning_pass_1_timer = current_time;
    args->timers.partitioning_pass_1_r_timer = current_time;
#endif
    /* if monitoring synchronization stats */

    /********** 1st pass of multi-pass partitioning ************/
    part_t part{};
    part.R = 0;
    part.D = args->num_radix_bits / args->num_passes;
    part.thrargs = args;
    part.padding = num_padding_tuples;

    /* 1. partitioning for relation R */
    part.rel = args->relR;
    part.tmp = args->tmpR;
    part.hist = args->histR;
    part.output = outputR;
    part.num_tuples = args->numR;
    part.total_tuples = args->totalR;
    part.relidx = 0;
    part.hist_timer = &args->timers.partitioning_pass_1_hist_timer;
    part.copy_timer = &args->timers.partitioning_pass_1_copy_timer;

#ifdef USE_SWWC_OPTIMIZED_PART
    logger(DBG, "Use SSWC optimized part");
    parallel_radix_partition_optimized(&part);
#else
    parallel_radix_partition(&part);
#endif

    args->barrier->wait();
    current_time = rdtscp_s();
    args->timers.partitioning_pass_1_r_timer = current_time - args->timers.partitioning_pass_1_r_timer;
    args->timers.partitioning_pass_1_s_timer = current_time;

    /* 2. partitioning for relation S */
    part.rel = args->relS;
    part.tmp = args->tmpS;
    part.hist = args->histS;
    part.output = outputS;
    part.num_tuples = args->numS;
    part.total_tuples = args->totalS;
    part.relidx = 1;
    part.hist_timer = &args->timers.partitioning_pass_1_hist_timer;
    part.copy_timer = &args->timers.partitioning_pass_1_copy_timer;

#ifdef USE_SWWC_OPTIMIZED_PART
    parallel_radix_partition_optimized(&part);
#else
    parallel_radix_partition(&part);
#endif


    /* wait at a barrier until each thread copies out */
    current_time = rdtscp_s();
    args->timers.partitioning_pass_1_s_timer = current_time - args->timers.partitioning_pass_1_s_timer;
    if (my_tid != 0) {
        args->timers.partitioning_pass_1_timer = current_time - args->timers.partitioning_pass_1_timer;
    }
    args->barrier->wait();
    /********** end of 1st partitioning phase ******************/

    /* 3. first thread creates partitioning tasks for 2nd pass */
    if (my_tid == 0) {
        uint64_t counter = 0;
        for (int i = 0; i < fanOut; i++) {
            int32_t ntupR = outputR[i + 1] - outputR[i] - num_padding_tuples;
            int32_t ntupS = outputS[i + 1] - outputS[i] - num_padding_tuples;

#ifdef MUTEX_QUEUE
            if (ntupR > 0 && ntupS > 0) {
                task_t *t = task_queue_get_slot(part_queue);

                t->relR.num_tuples = t->tmpR.num_tuples = ntupR;
                t->relR.tuples = args->tmpR + outputR[i];
                t->tmpR.tuples = args->tmpR2 + outputR[i];

                t->relS.num_tuples = t->tmpS.num_tuples = ntupS;
                t->relS.tuples = args->tmpS + outputS[i];
                t->tmpS.tuples = args->tmpS2 + outputS[i];

                task_queue_add(part_queue, t);
                counter++;
            }
#else
            if (ntupR > 0 && ntupS > 0) {
                task_t t {};

                t.relR.num_tuples = t.tmpR.num_tuples = ntupR;
                t.relR.tuples = args->tmpR + outputR[i];
                t.tmpR.tuples = args->tmpR2 + outputR[i];

                t.relS.num_tuples = t.tmpS.num_tuples = ntupS;
                t.relS.tuples = args->tmpS + outputS[i];
                t.tmpS.tuples = args->tmpS2 + outputS[i];

                //logger(WARN, "Partition size: %d %d", ntupR, ntupS);

                part_queue->push(t);
                counter++;
            }

#endif
        }
        /* debug partitioning task queue */
        if (args->num_passes == 2) {
            logger(INFO, "Pass-2: # partitioning tasks = %d", counter);
        } else {
            logger(INFO, "Pass-2: skipped. Join tasks = %d", counter);
        }
        args->timers.partitioning_pass_1_timer = current_time - args->timers.partitioning_pass_1_timer;
    }


    /* wait at a barrier until first thread adds all partitioning tasks */
    args->barrier->wait(
#ifndef RADIX_NO_TIMING
            [args]() {
                //ocall_stop_performance_counters(args->totalR + args->totalS, "Partition Pass 1");
                //ocall_start_performance_counters();
                return true;
            }
#endif
    );
    args->timers.partitioning_pass_2_timer = rdtscp_s();
    /* global barrier sync point-3 */

    /************ 2nd pass of multi-pass partitioning ********************/
    /* 4. now each thread further partitions and add to join task queue **/

    args->parts_partitioned = 0;
    if (args->num_passes == 1) {
        /* If the partitioning is single pass we directly add tasks from pass-1 */
#ifndef MUTEX_QUEUE
        queue *swap = join_queue;
#else
        task_queue_t *swap = join_queue;
#endif
        join_queue = part_queue;
        /* part_queue is used as a temporary queue for handling skewed parts */
        part_queue = swap;
    } else if (args->num_passes == 2) {
#ifdef MUTEX_QUEUE
        task_t *task;
        while ((task = task_queue_get_atomic(part_queue))) {
            serial_radix_partition(task, join_queue, R, D, &args->timers.partitioning_pass_2_hist_timer, &args->timers.partitioning_pass_1_copy_timer);
            args->parts_partitioned++;
        }
#else
        task_t task {};
        while (part_queue->pop(task)) {
            serial_radix_partition(&task, join_queue, R, D, &args->timers.partitioning_pass_2_hist_timer,
                &args->timers.partitioning_pass_2_copy_timer);
            args->parts_partitioned++;
        }
#endif
    } else {
        // Cannot happen since calc_num_passes can only return 1 or 2
    }

    free(outputR);
    free(outputS);

    /* wait at a barrier until all threads add all join tasks */
    current_time = rdtscp_s();
    args->timers.partitioning_pass_2_timer = current_time - args->timers.partitioning_pass_2_timer;
    args->timers.partitioning_total_timer =
            current_time - args->timers.partitioning_total_timer;/* partitioning finished */
    args->barrier->wait(
#ifndef RADIX_NO_TIMING
            [args, join_queue]() {
                //ocall_stop_performance_counters(args->totalR + args->totalS, "Partition Pass 2");
                //ocall_start_performance_counters();
#ifdef MUTEX_QUEUE
                logger(INFO, "Number of join tasks = %d", join_queue->count);
#endif
                return true;
            }
#endif
    );
    args->timers.join_total_timer = rdtscp_s();

#ifdef CHUNKED_TABLE
#ifndef CHUNKED_TABLE_PREALLOC
    if (args->materialize) {
        init_chunked_table(args->thread_result_table);
    }
#endif
#else
    output_list_t *output;
#endif

#ifdef MUTEX_QUEUE
    task_t *task;
    while ((task = task_queue_get_atomic(join_queue))) {
        /* do the actual join. join method differs for different algorithms,
           i.e. bucket chaining, histogram-based, histogram-based with simd &
           prefetching  */
#ifdef CHUNKED_TABLE
        results += args->join_function(&task->relR, &task->relS, &task->tmpR, args->num_radix_bits, args->thread_result_table,
                                       &args->build_in_depth_timer, &args->join_in_depth_timer, args->materialize);
#else
        results += args->join_function(&task->relR, &task->relS, &task->tmpR, args->num_radix_bits, &output,
                                       &args->timers.build_in_depth_timer, &args->timers.join_in_depth_timer, args->materialize);
#endif

        args->parts_joined++;
    }
#else
    task_t task {};
    while (join_queue->pop(task)) {
        /* do the actual join. join method differs for different algorithms,
           i.e. bucket chaining, histogram-based, histogram-based with simd &
           prefetching  */
#ifdef CHUNKED_TABLE
        results += args->join_function(&task.relR, &task.relS, &task.tmpR, args->num_radix_bits, args->thread_result_table,
                                       &args->timers.build_in_depth_timer, &args->timers.join_in_depth_timer, args->materialize);
#else
        results += args->join_function(&task.relR, &task.relS, &task.tmpR, args->num_radix_bits, &output,
                                       &args->timers.build_in_depth_timer, &args->timers.join_in_depth_timer,
                                       args->materialize);
#endif
        args->parts_joined++;
    }
#endif

    args->result = results;
    if (args->materialize) {
#ifdef CHUNKED_TABLE
        finish_chunked_table(args->thread_result_table);
#else
        args->threadresult->nresults = results;
        args->threadresult->threadid = args->my_tid;
        args->threadresult->results = output;
#endif
    }

    /* this thread is finished */

    /* this is for just reliable timing of finish time */
    current_time = rdtscp_s();
    args->timers.join_total_timer = current_time - args->timers.join_total_timer;/* build finished */
    args->timers.total_timer = current_time - args->timers.total_timer;/* probe finished */

    return nullptr;
}

/**
 * The template function for different joins: Basically each parallel radix join
 * has a initialization step, partitioning step and build-probe steps. All our
 * parallel radix implementations have exactly the same initialization and
 * partitioning steps. Difference is only in the build-probe step. Here are all
 * the parallel radix join implementations and their Join (build-probe) functions:
 *
 * - PRO,  Parallel Radix Join Optimized --> bucket_chaining_join()
 * - PRH,  Parallel Radix Join Histogram-based --> histogram_join()
 * - PRHO, Parallel Radix Histogram-based Optimized -> histogram_optimized_join()
 */
result_t *
join_init_run(const table_t *relR, const table_t *relS, JoinFunction jf, const joinconfig_t *config) {
    uint64_t start_time = rdtscp_s();
    auto nthreads = config->NTHREADS;
    Barrier barrier{static_cast<size_t>(nthreads)};
    pthread_t tid[nthreads];
    arg_t_radix args[nthreads];
    int64_t result = 0;

    ocall_pin_thread(config->ALLOC_CORE);

#ifndef CONSTANT_RADIX_BITS
    uint32_t num_radix_bits = calc_num_radix_bits(relR->num_tuples, relS->num_tuples, nthreads);
#else
    uint32_t num_radix_bits = 14;
    logger(INFO, "Forcing 14 radix bits.");
#endif
#ifndef FORCE_2_PHASES
    uint32_t num_passes = calc_num_passes(num_radix_bits);
#else
    uint32_t num_passes = 2;
    logger(INFO, "Forcing 2 join phases.");
#endif
    auto fanout_pass_1 = 1 << (num_radix_bits / num_passes);
    auto rel_padding = relation_padding(num_radix_bits, num_passes);

    logger(INFO, "Running RHO with %d passes and %d radix bits", num_passes,
           num_radix_bits);
    if (config->MATERIALIZE) {
        logger(INFO, "Materializing the output");
    }
#ifdef ENCLAVE
    logger(INFO, "Running in enclave mode");
#else
    logger(INFO, "Running in native mode");
#endif
#ifdef FULL_QUERY
    logger(INFO, "Running in full query mode");
#else
    logger(INFO, "Running in isolated mode");
#endif

#ifndef MUTEX_QUEUE
    queue part_queue {static_cast<queue::size_type>(fanout_pass_1 * 2)};
    queue join_queue {static_cast<queue::size_type>(1 << (num_radix_bits + 1))};
#else
    task_queue_t *part_queue = task_queue_init(fanout_pass_1);
    task_queue_t *join_queue = task_queue_init(1 << num_radix_bits);
#endif

    /* allocate temporary space for partitioning */

    uint64_t r_buf_size = relR->num_tuples * sizeof(struct row_t) + rel_padding;
    uint64_t s_buf_size = relS->num_tuples * sizeof(struct row_t) + rel_padding;

    auto tmpRelR = (struct row_t *) alloc_aligned(r_buf_size);
    auto tmpRelS = (struct row_t *) alloc_aligned(s_buf_size);

    // To achieve a fair comparison between enclave and native in single join benchmarks, we have to make sure that
    // the memory for the buffers is actually physically allocated
#ifndef ENCLAVE
#ifndef FULL_QUERY
    memset(tmpRelR, 42, r_buf_size);
    memset(tmpRelS, 42, s_buf_size);
#endif
#endif

    struct row_t *tmpRelR2 = nullptr;
    struct row_t *tmpRelS2 = nullptr;
    if (num_passes == 2 || jf == &histogram_join) {
        tmpRelR2 = (struct row_t *) alloc_aligned(r_buf_size);
        tmpRelS2 = (struct row_t *) alloc_aligned(s_buf_size);

        // To achieve a fair comparison between enclave and native in single join benchmarks, we have to make sure that
        // the memory for the buffers is actually physically allocated
#ifndef ENCLAVE
#ifndef FULL_QUERY
        memset(tmpRelR2, 42, r_buf_size);
        memset(tmpRelS2, 42, s_buf_size);
#endif
#endif
    }

    /* allocate histograms arrays, actual allocation is local to threads */
    auto histR = (uint32_t **) alloc_aligned(nthreads * sizeof(uint32_t *));
    auto histS = (uint32_t **) alloc_aligned(nthreads * sizeof(uint32_t *));

    /* first assign chunks of relR & relS for each thread */
    uint64_t numperthr[2];
    numperthr[0] = relR->num_tuples / nthreads;
    numperthr[1] = relS->num_tuples / nthreads;

    auto joinresult = (result_t *) malloc(sizeof(result_t));
    joinresult->nthreads = nthreads;
    joinresult->materialized = config->MATERIALIZE;

#ifdef CHUNKED_TABLE
    joinresult->result_type = 1;
    std::vector<chunked_table_t> thread_result_tables(nthreads);
#ifdef CHUNKED_TABLE_PREALLOC
    // Round up so that there will be enough chunks.
    // Add in 20 % more chunks for the case of light skew between the threads.
    size_t num_chunks_prealloc = (relS->num_tuples / TUPLES_PER_CHUNK + 1) * 6 / 5; // 20 % scew security
    size_t num_chunks_prealloc_per_thread = num_chunks_prealloc / nthreads + 1;
    for (auto & table : thread_result_tables) {
        init_chunked_table_prealloc(&table, num_chunks_prealloc_per_thread);
    }
#endif
#else
    joinresult->result_type = 0;
    joinresult->result = (threadresult_t *) malloc(sizeof(threadresult_t) * nthreads);
#endif

    // Allow as many cores as necessary
    ocall_set_mask(0, config->NTHREADS);

    auto preparation_time = rdtscp_s();

    for (int i = 0; i < nthreads; i++) {
        args[i].relR = relR->tuples + i * numperthr[0];
        args[i].tmpR = tmpRelR;
        args[i].tmpR2 = tmpRelR2;
        args[i].histR = histR;

        args[i].relS = relS->tuples + i * numperthr[1];
        args[i].tmpS = tmpRelS;
        args[i].tmpS2 = tmpRelS2;
        args[i].histS = histS;

        args[i].numR = (i == (nthreads - 1)) ? (relR->num_tuples - i * numperthr[0]) : numperthr[0];
        args[i].numS = (i == (nthreads - 1)) ? (relS->num_tuples - i * numperthr[1]) : numperthr[1];
        args[i].totalR = relR->num_tuples;
        args[i].totalS = relS->num_tuples;

        args[i].num_radix_bits = num_radix_bits;
        args[i].num_passes = num_passes;

        args[i].my_tid = i;
#ifndef MUTEX_QUEUE
        args[i].part_queue = &part_queue;
        args[i].join_queue = &join_queue;
#else
        args[i].part_queue = part_queue;
        args[i].join_queue = join_queue;
#endif
        args[i].barrier = &barrier;
        args[i].join_function = jf;
        args[i].nthreads = nthreads;
        args[i].timers = {};
        args[i].materialize = config->MATERIALIZE;

        args[i].parts_joined = 0;
        args[i].parts_partitioned = 0;
#ifdef CHUNKED_TABLE
        args[i].thread_result_table = &(thread_result_tables[i]);
#else
        args[i].threadresult = ((threadresult_t *) joinresult->result) + i;
#endif
    }

    logger(INFO, "Starting join threads");
    // Start the join threads backwards so that the thread pinned on core 0 does not influence the main thread.
    for (int i = nthreads - 1; i >= 1; --i) {
        int rv = pthread_create(&tid[i], nullptr, prj_thread, (void *) &args[i]);

        if (rv) {
            logger(ERROR, "return code from pthread_create() is %d\n", rv);
            ocall_exit(-1);
        }
    }

    prj_thread(args);

    /* wait for threads to finish */
    for (int i = nthreads - 1; i >= 1; --i) {
        pthread_join(tid[i], nullptr);
        result += args[i].result;
    }

    result += args[0].result;

    auto join_time = rdtscp_s();

    joinresult->totalresults = result;

#ifdef CHUNKED_TABLE
    // link results of all chunked tables together by copying the chunk pointers
    joinresult->result = concatenate(thread_result_tables);
#endif

    /* clean up */
    for (int i = 0; i < nthreads; i++) {
        free(histR[i]);
        free(histS[i]);
    }
    free(histR);
    free(histS);
#ifdef MUTEX_QUEUE
    task_queue_free(part_queue);
    task_queue_free(join_queue);
#endif
    free(tmpRelR);
    free(tmpRelS);

    if (num_passes == 2 || jf == &histogram_join) {
        free(tmpRelR2);
        free(tmpRelS2);
    }

    auto free_time = rdtscp_s();

#ifndef RADIX_NO_TIMING

    uint64_t end_time = rdtscp_s();
    radix_timers_t max_timers{};

    for (int i = 0; i < nthreads; ++i) {
        max_timers.total_timer = std::max(args[i].timers.total_timer, max_timers.total_timer);

        max_timers.partitioning_total_timer = std::max(args[i].timers.partitioning_total_timer,
                                                       max_timers.partitioning_total_timer);
        max_timers.partitioning_pass_1_timer = std::max(args[i].timers.partitioning_pass_1_timer,
                                                        max_timers.partitioning_pass_1_timer);
        max_timers.partitioning_pass_1_r_timer = std::max(args[i].timers.partitioning_pass_1_r_timer,
                                                          max_timers.partitioning_pass_1_r_timer);
        max_timers.partitioning_pass_1_s_timer =
                std::max(args[i].timers.partitioning_pass_1_s_timer, max_timers.partitioning_pass_1_s_timer);
        max_timers.partitioning_pass_1_hist_timer =
                std::max(args[i].timers.partitioning_pass_1_hist_timer, max_timers.partitioning_pass_1_hist_timer);
        max_timers.partitioning_pass_1_copy_timer =
                std::max(args[i].timers.partitioning_pass_1_copy_timer, max_timers.partitioning_pass_1_copy_timer);
        max_timers.partitioning_pass_2_timer =
                std::max(args[i].timers.partitioning_pass_2_timer, max_timers.partitioning_pass_2_timer);
        max_timers.partitioning_pass_2_hist_timer =
                std::max(args[i].timers.partitioning_pass_2_hist_timer, max_timers.partitioning_pass_2_hist_timer);
        max_timers.partitioning_pass_2_copy_timer =
                std::max(args[i].timers.partitioning_pass_2_copy_timer, max_timers.partitioning_pass_2_copy_timer);
        max_timers.join_total_timer = std::max(args[i].timers.join_total_timer, max_timers.join_total_timer);
        max_timers.build_in_depth_timer =
                std::max(args[i].timers.build_in_depth_timer, max_timers.build_in_depth_timer);
        max_timers.join_in_depth_timer = std::max(args[i].timers.join_in_depth_timer, max_timers.join_in_depth_timer);

        logger(INFO, "Thread %d time for partitioning pass 1: %d\n"
                     "          time for R: %d\n"
                     "          time for S: %d\n"
                     "          time for histogram: %d\n"
                     "          time for copy:      %d\n"
                     "          partitioned %d partitions in second pass. time: %d\n"
                     "          hist: %d copy: %d\n"
                     "          joined %d parts\n"
                     "          build: %d probe: %d",
                     i, args[i].timers.partitioning_pass_1_timer,
                     args[i].timers.partitioning_pass_1_r_timer,
                     args[i].timers.partitioning_pass_1_s_timer, args[i].timers.partitioning_pass_1_hist_timer,
                     args[i].timers.partitioning_pass_1_copy_timer,
                     args[i].parts_partitioned, args[i].timers.partitioning_pass_2_timer,
               args[i].timers.partitioning_pass_2_hist_timer, args[i].timers.partitioning_pass_2_copy_timer,
               args[i].parts_joined, args[i].timers.build_in_depth_timer, args[i].timers.join_in_depth_timer);
    }

    /* now print the timing results: */
    print_timing(max_timers,
                 start_time,
                 end_time,
                 relR->num_tuples + relS->num_tuples,
                 result, preparation_time - start_time, join_time - preparation_time, free_time - join_time);
#endif

    return joinresult;
}

result_t *
RHO(const table_t *relR, const table_t *relS, const joinconfig_t *config) {
    return join_init_run(relR, relS, bucket_chaining_join, config);
}

result_t *
RHT(const table_t *relR, const table_t *relS, const joinconfig_t *config) {
    return join_init_run(relR, relS, histogram_join, config);
}