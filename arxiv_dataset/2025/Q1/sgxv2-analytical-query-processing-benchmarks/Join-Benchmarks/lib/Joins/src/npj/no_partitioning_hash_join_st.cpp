#include "npj/no_partitioning_hash_join_st.hpp"
#include "npj/HashLinkTableCommon.hpp"
#include <algorithm>

#ifdef ENCLAVE
#include "ocalls_t.h"
#else
#include "ocalls.hpp"
#endif

/**
 * Single-thread hashtable build method, ht is pre-allocated.
 *
 * @param ht hash table to be built
 * @param rel the build relation
 */
void
build_hashtable_st(hashtable_t *ht, const struct table_t *rel) {
    const uint32_t hash_mask = ht->hash_mask;
    const uint32_t skip_bits = ht->skip_bits;

    for (uint64_t i = 0; i < rel->num_tuples; i++) {
        struct row_t *dest;
        int64_t idx = HASH(rel->tuples[i].key, hash_mask, skip_bits);

        /* copy the tuple to appropriate hash bucket */
        /* if full, follow nxt pointer to find correct place */
        bucket_t *curr = ht->buckets + idx;
        bucket_t *nxt = curr->next;

        if (curr->count == BUCKET_SIZE) {
            if (!nxt || nxt->count == BUCKET_SIZE) {
                auto b = (bucket_t *) calloc(1, sizeof(bucket_t));
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
    }
}

/**
 * Single-thread hashtable build method, ht is pre-allocated. Uses the overflow buffer that was originally only used
 * for the mt version of hash table creation.
 *
 * @param ht hash table to be built
 * @param rel the build relation
 */
void
build_hashtable_st_opt(hashtable_t *ht, const struct table_t *rel, bucket_buffer_t **overflow_buffer) {
    const uint32_t hash_mask = ht->hash_mask;
    const uint32_t skip_bits = ht->skip_bits;

    for (uint64_t i = 0; i < rel->num_tuples; i++) {
        struct row_t *dest;
        int32_t idx = HASH(rel->tuples[i].key, hash_mask, skip_bits);

        /* copy the tuple to appropriate hash bucket */
        /* if full, follow nxt pointer to find correct place */
        bucket_t *curr = ht->buckets + idx;
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
    }
}

result_t *NPO_single_thread(const table_t *relR, const table_t *relS, const joinconfig_t *config) {
    hashtable_t *ht;
    int64_t result = 0;
#ifndef NO_TIMING
    uint64_t probe_timer;
    uint64_t build_timer;
    uint64_t total_timer;
    uint64_t start;
    uint64_t end;
#endif
    uint32_t nbuckets = (relR->num_tuples / BUCKET_SIZE);
    allocate_hashtable(&ht, nbuckets);

    /* allocate overflow buffer */
    bucket_buffer_t *overflow_buffer;
    init_bucket_buffer(&overflow_buffer);

#ifndef NO_TIMING
    ocall_get_system_micros(&start);
    ocall_start_performance_counters();
    auto current_time = rdtscp_s();
    build_timer = current_time;
    total_timer = current_time;
#endif


    build_hashtable_st_opt(ht, relR, &overflow_buffer);

#ifndef NO_TIMING
    current_time = rdtscp_s();
    build_timer = current_time - build_timer;
    ocall_stop_performance_counters(relR->num_tuples, "PHT build");
    ocall_start_performance_counters();
    probe_timer = rdtscp_s();
#endif

    output_list_t *output;
    result = probe_hashtable(ht, relS, &output, 0);

#ifndef NO_TIMING
    current_time = rdtscp_s();
    probe_timer = current_time - probe_timer;
    total_timer = current_time - total_timer;
    /* over all */
    ocall_stop_performance_counters(relS->num_tuples, "PHT probe");
    ocall_get_system_micros(&end);
    /* now print the timing results: */
    print_timing(start, end, total_timer, build_timer, probe_timer, relR->num_tuples + relS->num_tuples, result);
#endif

    destroy_hashtable(ht);
    free_bucket_buffer(overflow_buffer);

    result_t *joinresult;
    joinresult = (result_t *) malloc(sizeof(result_t));
    joinresult->totalresults = result;
    joinresult->nthreads = 1;
    return joinresult;
}

/**
 * Single-thread hashtable build method, ht is pre-allocated.
 *
 * @param ht hash table to be built
 * @param rel the build relation
 */
void
build_hashtable_no_overflow(hashtable_t *ht, const struct table_t *rel) {
    const uint32_t hash_mask = ht->hash_mask;

    for (uint64_t i = 0; i < rel->num_tuples; i++) {
        bucket_t *bucket = ht->buckets + HASH(rel->tuples[i].key, hash_mask, 0);
        *(bucket->tuples + bucket->count++) = rel->tuples[i];
    }
}

/**
 * Single-thread hashtable build method, ht is pre-allocated.
 *
 * @param ht hash table to be built
 * @param rel the build relation
 */
void
build_hashtable_no_overflow_unrolled(hashtable_t *ht, const struct table_t *rel) {
    const uint32_t hash_mask = ht->hash_mask;

    uint64_t i = 0;
    for (; i + 8 <= rel->num_tuples; i += 8) {
        bucket_t *bucket0 = ht->buckets + HASH(rel->tuples[i].key, hash_mask, 0);
        bucket_t *bucket1 = ht->buckets + HASH(rel->tuples[i + 1].key, hash_mask, 0);
        bucket_t *bucket2 = ht->buckets + HASH(rel->tuples[i + 2].key, hash_mask, 0);
        bucket_t *bucket3 = ht->buckets + HASH(rel->tuples[i + 3].key, hash_mask, 0);
        bucket_t *bucket4 = ht->buckets + HASH(rel->tuples[i + 4].key, hash_mask, 0);
        bucket_t *bucket5 = ht->buckets + HASH(rel->tuples[i + 5].key, hash_mask, 0);
        bucket_t *bucket6 = ht->buckets + HASH(rel->tuples[i + 6].key, hash_mask, 0);
        bucket_t *bucket7 = ht->buckets + HASH(rel->tuples[i + 7].key, hash_mask, 0);
        *(bucket0->tuples + bucket0->count++) = rel->tuples[i];
        *(bucket1->tuples + bucket1->count++) = rel->tuples[i + 1];
        *(bucket2->tuples + bucket2->count++) = rel->tuples[i + 2];
        *(bucket3->tuples + bucket3->count++) = rel->tuples[i + 3];
        *(bucket4->tuples + bucket4->count++) = rel->tuples[i + 4];
        *(bucket5->tuples + bucket5->count++) = rel->tuples[i + 5];
        *(bucket6->tuples + bucket6->count++) = rel->tuples[i + 6];
        *(bucket7->tuples + bucket7->count++) = rel->tuples[i + 7];
    }
    for (; i < rel->num_tuples; i++) {
        bucket_t *bucket = ht->buckets + HASH(rel->tuples[i].key, hash_mask, 0);
        *(bucket->tuples + bucket->count++) = rel->tuples[i];
    }
}

result_t *
NPO_no_overflow(const table_t *relR, const table_t *relS, const joinconfig_t *config) {
    hashtable_t *ht;
    int64_t result = 0;
#ifndef NO_TIMING
    uint64_t probe_timer;
    uint64_t build_timer;
    uint64_t total_timer;
    uint64_t start;
    uint64_t end;
#endif
    uint32_t nbuckets = (relR->num_tuples / BUCKET_SIZE);
    allocate_hashtable(&ht, nbuckets);

#ifndef NO_TIMING
    ocall_get_system_micros(&start);
    auto current_time = rdtscp_s();
    build_timer = current_time;
    total_timer = current_time;
#endif

#ifndef UNROLL
    build_hashtable_no_overflow(ht, relR);
#else
    build_hashtable_no_overflow_unrolled(ht, relR);
#endif

#ifndef NO_TIMING
    current_time = rdtscp_s();
    build_timer = current_time - build_timer;
    probe_timer = current_time;
#endif

#ifndef UNROLL
    result = probe_hashtable_no_overflow(ht, relS);
#else
    result = probe_hashtable_no_overflow_unrolled(ht, relS);
#endif

#ifndef NO_TIMING
    current_time = rdtscp_s();
    probe_timer = current_time - probe_timer;
    total_timer = current_time - total_timer;
    /* over all */
    ocall_get_system_micros(&end);
    /* now print the timing results: */
    print_timing(start, end, total_timer, build_timer, probe_timer, relR->num_tuples + relS->num_tuples, result);
#endif

    destroy_hashtable(ht);

    result_t *joinresult;
    joinresult = (result_t *) malloc(sizeof(result_t));
    joinresult->totalresults = result;
    joinresult->nthreads = 1;
    return joinresult;
}
