#include "npj/no_partitioning_hash_join.hpp"
#include "npj/HashLinkTableCommon.hpp"
#include "pthread.h"
#include "Barrier.hpp"

#ifdef ENCLAVE
#include "ocalls_t.h"
#else
#include "ocalls.hpp"
#endif

// Unroll factor 10 was empirically determined to be optimal.
#ifndef PHT_UNROLL_FACTOR
#define PHT_UNROLL_FACTOR 10
#endif

struct hash_join_timers {
    uint64_t build;
    uint64_t probe;
    uint64_t total;
    uint64_t start;
    uint64_t end;
};

struct arg_t {
    int32_t tid;
    hashtable_t *ht;
    struct table_t relR;
    struct table_t relS;
    Barrier *barrier;
    int64_t num_results;
    uint64_t total_r_tuples;
    uint64_t total_s_tuples;
    threadresult_t *thread_result;
#ifndef NO_TIMING
    /* stats about the thread */
    hash_join_timers *timers;
    //    struct timeval start, end;
#endif
    int materialize;
};

/**
 * Multi-thread hashtable build method, ht is pre-allocated.
 * Writes to buckets are synchronized via latches.
 *
 * @param ht hash table to be built
 * @param rel the build relation
 * @param overflow_buffer pre-allocated chunk of buckets for overflow use.
 */
void build_hashtable_mt(hashtable_t *ht, const table_t *rel,
                        bucket_buffer_t **overflow_buffer) {
    const uint32_t hash_mask = ht->hash_mask;
    const uint32_t skip_bits = ht->skip_bits;

    for (uint64_t i = 0; i < rel->num_tuples; i++) {
        struct row_t *dest;

        int32_t idx = HASH(rel->tuples[i].key, hash_mask, skip_bits);
        /* copy the tuple to appropriate hash bucket */
        /* if full, follow nxt pointer to find correct place */
        bucket_t *curr = ht->buckets + idx;
        lock(&curr->latch);
        bucket_t *nxt = curr->next;

        if (curr->count == BUCKET_SIZE) {
            if (!nxt || nxt->count == BUCKET_SIZE) {
                bucket_t *b;
                /* b = (bucket_t*) calloc(1, sizeof(bucket_t)); */
                /* instead of calloc() everytime, we pre-allocate */
                get_new_bucket(&b, overflow_buffer);
                curr->next = b;
                b->next = nxt;
                b->count = 1;
                dest = b->tuples;
            } else {
                dest = nxt->tuples + nxt->count;
                nxt->count++;
            }
        } else {
            dest = curr->tuples + curr->count;
            curr->count++;
        }

        *dest = rel->tuples[i];
        unlock(&curr->latch);
    }
}

/**
 * Just a wrapper to call the build and probe for each thread.
 *
 * @param param the parameters of the thread, i.e. tid, ht, reln, ...
 * @return
 */
void *
npo_thread(void *param) {
    auto args = (arg_t *) param;

    /* allocate overflow buffer for each thread */
    bucket_buffer_t *overflow_buffer;
    init_bucket_buffer(&overflow_buffer);

    /* wait at a barrier until each thread starts and start timer */
    args->barrier->wait(
#ifndef NO_TIMING
        [&args]() {
            ocall_get_system_micros(&args->timers->start);
            //ocall_start_performance_counters();
            auto current_time = rdtscp_s();
            args->timers->total = current_time; /* no partitionig phase */
            args->timers->build = current_time;
            return true;
        }
#endif
    );

    /* insert tuples from the assigned part of relR to the ht */
    build_hashtable_mt(args->ht, &args->relR, &overflow_buffer);

    /* wait at a barrier until each thread completes build phase */
    args->barrier->wait(
#ifndef NO_TIMING
        [&args]() {
            auto current_time = rdtscp_s();
            args->timers->build = current_time - args->timers->build;
            //ocall_stop_performance_counters(args->total_r_tuples, "Build");
            //ocall_start_performance_counters();
            args->timers->probe = current_time;
            return true;
        }
#endif
    );

    /* probe for matching tuples from the assigned part of relS */
    output_list_t *output;
    args->num_results = probe_hashtable(args->ht, &args->relS, &output, args->materialize);

    /* for a reliable timing we have to wait until all finishes */
    args->barrier->wait(
#ifndef NO_TIMING
        [&args]() {
            auto current_time = rdtscp_s();
            //ocall_stop_performance_counters(args->total_s_tuples, "Probe");
            args->timers->probe = current_time - args->timers->probe;
            args->timers->total = current_time - args->timers->total;
            ocall_get_system_micros(&args->timers->end);
            return true;
        }
#endif
    );

    if (args->materialize) {
        args->thread_result->nresults = args->num_results;
        args->thread_result->threadid = args->tid;
        args->thread_result->results = output;
    }

    /* clean-up the overflow buffers */
    free_bucket_buffer(overflow_buffer);

    return nullptr;
}

result_t *
PHT_int(const table_t *relR, const table_t *relS, const joinconfig_t *config) {
    hashtable_t *ht;
    int64_t result = 0;
    int32_t numR;
    int32_t numS;
    int32_t numRthr;
    int32_t numSthr; /* total and per thread num */
    auto nthreads = config->NTHREADS;
    int i;
    int rv;
    //    cpu_set_t set;
    arg_t args[nthreads];
    pthread_t tid[nthreads];
    //    pthread_attr_t attr;
    Barrier barrier{static_cast<size_t>(nthreads)};

    uint32_t nbuckets = (relR->num_tuples / BUCKET_SIZE);
    allocate_hashtable(&ht, nbuckets);

    numR = relR->num_tuples;
    numS = relS->num_tuples;
    numRthr = numR / nthreads;
    numSthr = numS / nthreads;

    hash_join_timers timers{};

    auto joinresult = (result_t *) malloc(sizeof(result_t));
    joinresult->result = (threadresult_t *) malloc(sizeof(threadresult_t) * nthreads);
    joinresult->result_type = 0;
    joinresult->nthreads = nthreads;

    //    pthread_attr_init(&attr);
    for (i = 0; i < nthreads; i++) {
        args[i].tid = i;
        args[i].ht = ht;
        args[i].barrier = &barrier;

        /* passing part of the relR for next thread */
        args[i].relR.num_tuples = (i == (nthreads - 1)) ? numR : numRthr;
        args[i].relR.tuples = relR->tuples + numRthr * i;
        numR -= numRthr;

        /* passing part of the relS for next thread */
        args[i].relS.num_tuples = (i == (nthreads - 1)) ? numS : numSthr;
        args[i].relS.tuples = relS->tuples + numSthr * i;
        numS -= numSthr;

        args[i].total_r_tuples = relR->num_tuples;
        args[i].total_s_tuples = relS->num_tuples;

        args[i].timers = &timers;
        /*  The attr is not supported inside the Enclave, so the new thread will be
            created with PTHREAD_CREATE_JOINABLE.
        */
        args[i].thread_result = ((threadresult_t *) joinresult->result) + i;
        args[i].materialize = config->MATERIALIZE;
        rv = pthread_create(&tid[i], nullptr, npo_thread, (void *) &args[i]);

        if (rv) {
            logger(ERROR, "ERROR; return code from pthread_create() is %d\n", rv);
            ocall_exit(-1);
        }
    }

    for (i = 0; i < nthreads; i++) {
        pthread_join(tid[i], nullptr);
        /* sum up results */
        result += args[i].num_results;
    }

    joinresult->materialized = config->MATERIALIZE;
    joinresult->totalresults = result;


#ifndef NO_TIMING
    /* now print the timing results: */
    print_timing(timers.start, timers.end, timers.total, timers.build, timers.probe,
                 relR->num_tuples + relS->num_tuples, result);
#endif

    destroy_hashtable(ht);

    return joinresult;
}

[[gnu::always_inline]] inline void
save_in_bucket(bucket_t *bucket, const row_t &tuple) {
    lock(&bucket->latch);
    *(bucket->tuples + bucket->count++) = tuple;
    unlock(&bucket->latch);
}

/**
 * Thread-save hashtable build for a hashtable without overflow buckets.
 *
 * @param ht hash table to be built
 * @param rel the build relation
 */
void
build_hashtable_mt_no_overflow(hashtable_t *ht, const table_t *rel) {
    const uint32_t hash_mask = ht->hash_mask;

    for (uint64_t i = 0; i < rel->num_tuples; ++i) {
        bucket_t *bucket = ht->buckets + HASH(rel->tuples[i].key, hash_mask, 0);
        save_in_bucket(bucket, rel->tuples[i]);
    }
}

/**
 * Unrolled, thread-save hashtable build for a hashtable without overflow buckets.
 *
 * @param ht hash table to be built
 * @param rel the build relation
 */
template<int UNROLL_FACTOR=PHT_UNROLL_FACTOR>
void
build_hashtable_mt_no_overflow_unrolled_goto(hashtable_t *ht, const table_t *rel) {
    static_assert(UNROLL_FACTOR >= 1 && UNROLL_FACTOR <= 12, "Unroll factor must be between 1 and 12!");

    const uint32_t hash_mask = ht->hash_mask;

    bucket_t *bucket0, *bucket1, *bucket2, *bucket3, *bucket4, *bucket5, *bucket6, *bucket7, *bucket8, *bucket9, *
            bucket10, *bucket11;
    uint64_t i = 0;
    for (; i + UNROLL_FACTOR <= rel->num_tuples; i += UNROLL_FACTOR) {
        bucket0 = ht->buckets + HASH(rel->tuples[i].key, hash_mask, 0);
        if constexpr (UNROLL_FACTOR == 1) goto save;
        bucket1 = ht->buckets + HASH(rel->tuples[i + 1].key, hash_mask, 0);
        if constexpr (UNROLL_FACTOR == 2) goto save;
        bucket2 = ht->buckets + HASH(rel->tuples[i + 2].key, hash_mask, 0);
        if constexpr (UNROLL_FACTOR == 3) goto save;
        bucket3 = ht->buckets + HASH(rel->tuples[i + 3].key, hash_mask, 0);
        if constexpr (UNROLL_FACTOR == 4) goto save;
        bucket4 = ht->buckets + HASH(rel->tuples[i + 4].key, hash_mask, 0);
        if constexpr (UNROLL_FACTOR == 5) goto save;
        bucket5 = ht->buckets + HASH(rel->tuples[i + 5].key, hash_mask, 0);
        if constexpr (UNROLL_FACTOR == 6) goto save;
        bucket6 = ht->buckets + HASH(rel->tuples[i + 6].key, hash_mask, 0);
        if constexpr (UNROLL_FACTOR == 7) goto save;
        bucket7 = ht->buckets + HASH(rel->tuples[i + 7].key, hash_mask, 0);
        if constexpr (UNROLL_FACTOR == 8) goto save;
        bucket8 = ht->buckets + HASH(rel->tuples[i + 8].key, hash_mask, 0);
        if constexpr (UNROLL_FACTOR == 9) goto save;
        bucket9 = ht->buckets + HASH(rel->tuples[i + 9].key, hash_mask, 0);
        if constexpr (UNROLL_FACTOR == 10) goto save;
        bucket10 = ht->buckets + HASH(rel->tuples[i + 10].key, hash_mask, 0);
        if constexpr (UNROLL_FACTOR == 11) goto save;
        bucket11 = ht->buckets + HASH(rel->tuples[i + 11].key, hash_mask, 0);

    save:
        save_in_bucket(bucket0, rel->tuples[i]);
        if constexpr (UNROLL_FACTOR == 1) goto endloop;
        save_in_bucket(bucket1, rel->tuples[i + 1]);
        if constexpr (UNROLL_FACTOR == 2) goto endloop;
        save_in_bucket(bucket2, rel->tuples[i + 2]);
        if constexpr (UNROLL_FACTOR == 3) goto endloop;
        save_in_bucket(bucket3, rel->tuples[i + 3]);
        if constexpr (UNROLL_FACTOR == 4) goto endloop;
        save_in_bucket(bucket4, rel->tuples[i + 4]);
        if constexpr (UNROLL_FACTOR == 5) goto endloop;
        save_in_bucket(bucket5, rel->tuples[i + 5]);
        if constexpr (UNROLL_FACTOR == 6) goto endloop;
        save_in_bucket(bucket6, rel->tuples[i + 6]);
        if constexpr (UNROLL_FACTOR == 7) goto endloop;
        save_in_bucket(bucket7, rel->tuples[i + 7]);
        if constexpr (UNROLL_FACTOR == 8) goto endloop;
        save_in_bucket(bucket8, rel->tuples[i + 7]);
        if constexpr (UNROLL_FACTOR == 9) goto endloop;
        save_in_bucket(bucket9, rel->tuples[i + 7]);
        if constexpr (UNROLL_FACTOR == 10) goto endloop;
        save_in_bucket(bucket10, rel->tuples[i + 7]);
        if constexpr (UNROLL_FACTOR == 11) goto endloop;
        save_in_bucket(bucket11, rel->tuples[i + 7]);
    endloop:;
    }
    for (; i < rel->num_tuples; i++) {
        bucket_t *bucket = ht->buckets + HASH(rel->tuples[i].key, hash_mask, 0);
        save_in_bucket(bucket, rel->tuples[i]);
    }
}

using HashTableBuildFunction = void (*)(hashtable_t *, const table_t *);
using HashTableProbeFunction = int64_t (*)(const hashtable_t *, const table_t *);

/**
 * Just a wrapper to call the build and probe for each thread.
 *
 * @param param the parameters of the thread, i.e. tid, ht, reln, ...
 * @return
 */
template<HashTableBuildFunction BuildFunction, HashTableProbeFunction ProbeFunction>
void *
npo_unroll_thread(void *param) {
    auto args = (arg_t *) param;

    /* wait at a barrier until each thread starts and start timer */
    args->barrier->wait(
#ifndef NO_TIMING
        [&args]() {
            ocall_get_system_micros(&args->timers->start);
            //ocall_start_performance_counters();
            auto current_time = rdtscp_s();
            args->timers->total = current_time; /* no partitionig phase */
            args->timers->build = current_time;
            return true;
        }
#endif
    );

    /* insert tuples from the assigned part of relR to the ht */
    BuildFunction(args->ht, &args->relR);

    /* wait at a barrier until each thread completes build phase */
    args->barrier->wait(
#ifndef NO_TIMING
        [&args]() {
            auto current_time = rdtscp_s();
            args->timers->build = current_time - args->timers->build;
            //ocall_stop_performance_counters(args->total_r_tuples, "Build");
            //ocall_start_performance_counters();
            args->timers->probe = current_time;
            return true;
        }
#endif
    );

    /* probe for matching tuples from the assigned part of relS */
    args->num_results = ProbeFunction(args->ht, &args->relS);

    /* for a reliable timing we have to wait until all finishes */
    args->barrier->wait(
#ifndef NO_TIMING
        [&args]() {
            auto current_time = rdtscp_s();
            //ocall_stop_performance_counters(args->total_s_tuples, "Probe");
            args->timers->probe = current_time - args->timers->probe;
            args->timers->total = current_time - args->timers->total;
            ocall_get_system_micros(&args->timers->end);
            return true;
        }
#endif
    );

    if (args->materialize) {
        args->thread_result->nresults = args->num_results;
        args->thread_result->threadid = args->tid;
        args->thread_result->results = nullptr;
    }

    return nullptr;
}

template<HashTableBuildFunction BuildFunction, HashTableProbeFunction ProbeFunction>
result_t *
PHT_no_overflow_template(const table_t *relR, const table_t *relS, const joinconfig_t *config) {
    hashtable_t *ht;
    int64_t result = 0;
    int32_t numR;
    int32_t numS;
    int32_t numRthr;
    int32_t numSthr; /* total and per thread num */
    auto nthreads = config->NTHREADS;
    int i;
    int rv;
    //    cpu_set_t set;
    arg_t args[nthreads];
    pthread_t tid[nthreads];
    //    pthread_attr_t attr;
    Barrier barrier{static_cast<size_t>(nthreads)};

    uint32_t nbuckets = (relR->num_tuples / BUCKET_SIZE);
    allocate_hashtable(&ht, nbuckets);

    numR = relR->num_tuples;
    numS = relS->num_tuples;
    numRthr = numR / nthreads;
    numSthr = numS / nthreads;

    hash_join_timers timers{};

    auto joinresult = (result_t *) malloc(sizeof(result_t));
    joinresult->result = (threadresult_t *) malloc(sizeof(threadresult_t) * nthreads);
    joinresult->result_type = 0;
    joinresult->nthreads = nthreads;

    //    pthread_attr_init(&attr);
    for (i = 0; i < nthreads; i++) {
        args[i].tid = i;
        args[i].ht = ht;
        args[i].barrier = &barrier;

        /* passing part of the relR for next thread */
        args[i].relR.num_tuples = (i == (nthreads - 1)) ? numR : numRthr;
        args[i].relR.tuples = relR->tuples + numRthr * i;
        numR -= numRthr;

        /* passing part of the relS for next thread */
        args[i].relS.num_tuples = (i == (nthreads - 1)) ? numS : numSthr;
        args[i].relS.tuples = relS->tuples + numSthr * i;
        numS -= numSthr;

        args[i].total_r_tuples = relR->num_tuples;
        args[i].total_s_tuples = relS->num_tuples;

        args[i].timers = &timers;
        /*  The attr is not supported inside the Enclave, so the new thread will be
            created with PTHREAD_CREATE_JOINABLE.
        */
        args[i].thread_result = ((threadresult_t *) joinresult->result) + i;
        args[i].materialize = config->MATERIALIZE;
        rv = pthread_create(&tid[i], nullptr, npo_unroll_thread<BuildFunction, ProbeFunction>, (void *) &args[i]);

        if (rv) {
            logger(ERROR, "ERROR; return code from pthread_create() is %d\n", rv);
            ocall_exit(-1);
        }
    }

    for (i = 0; i < nthreads; i++) {
        pthread_join(tid[i], nullptr);
        /* sum up results */
        result += args[i].num_results;
    }

    joinresult->materialized = config->MATERIALIZE;
    joinresult->totalresults = result;


#ifndef NO_TIMING
    /* now print the timing results: */
    print_timing(timers.start, timers.end, timers.total, timers.build, timers.probe,
                 relR->num_tuples + relS->num_tuples, result);
#endif

    destroy_hashtable(ht);

    return joinresult;
}

result_t *
PHT(const table_t *relR, const table_t *relS, const joinconfig_t *config) {
#ifdef UNROLL
    return PHT_no_overflow_template<
        build_hashtable_mt_no_overflow_unrolled_goto,
        probe_hashtable_no_overflow_unrolled
    >(relR, relS,config);
#else
    return PHT_no_overflow_template<
        build_hashtable_mt_no_overflow,
        probe_hashtable_no_overflow
    >(relR, relS,config);
#endif
}

result_t *
PHT_no_overflow(const table_t *relR, const table_t *relS, const joinconfig_t *config) {
    return PHT_no_overflow_template<
        build_hashtable_mt_no_overflow,
        probe_hashtable_no_overflow
    >(relR, relS, config);
}

result_t *
PHT_unrolled(const table_t *relR, const table_t *relS, const joinconfig_t *config) {
    return PHT_no_overflow_template<
        build_hashtable_mt_no_overflow_unrolled_goto,
        probe_hashtable_no_overflow_unrolled
    >(relR, relS, config);
}

result_t *
PHT_overflow(const table_t *relR, const table_t *relS, const joinconfig_t *config) {
    return PHT_int(relR, relS, config);
}
