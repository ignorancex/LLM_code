#include "npj/HashLinkTableCommon.hpp"
#include "util.hpp"

#ifdef ENCLAVE
#include "ocalls_t.h"
#else
#include "ocalls.hpp"
#endif

constexpr uint64_t CPMS = CYCLES_PER_MICROSECOND;

/** print out the execution time statistics of the join */
void
print_timing(uint64_t start, uint64_t end, uint64_t total, uint64_t build, uint64_t probe,
             uint64_t numtuples, int64_t result) {
    double cyclestuple = (double) total / (double) numtuples;
    uint64_t time_usec = end - start;
    uint64_t time_usec_tsc = total / CPMS;
    double throughput = (double) numtuples / (double) time_usec_tsc;
    logger(INFO, "Total input tuples : %lu", numtuples);
    logger(INFO, "Result tuples : %lu", result);
    logger(INFO, "Total Join Time (cycles)    : %lu", total);
    logger(INFO, "Build (cycles)              : %lu", build);
    logger(INFO, "Join (cycles)               : %lu", probe);
    logger(INFO, "Cycles-per-tuple : %.4lf", cyclestuple);
    logger(INFO, "Total Runtime (us) : %lu ", time_usec);
    logger(INFO, "Total RT from TSC (us) : %lu ", time_usec_tsc);
    logger(INFO, "Throughput (M rec/sec) : %.2lf", throughput);
}

void allocate_hashtable(hashtable_t **ppht, uint32_t nbuckets) {
    auto ht = (hashtable_t *) malloc(sizeof(hashtable_t));
    ht->num_buckets = nbuckets;
    NEXT_POW_2((ht->num_buckets));

    /* allocate hashtable buckets cache line aligned */
    ht->buckets = (bucket_t *) aligned_alloc(CACHE_LINE_SIZE, ht->num_buckets * sizeof(bucket_t));
    if (!ht || !ht->buckets) {
        logger(ERROR, "Memory allocation for the hashtable failed!");
        ocall_exit(EXIT_FAILURE);
    }

    memset(ht->buckets, 0, ht->num_buckets * sizeof(bucket_t));
    ht->skip_bits = 0; /* the default for modulo hash */
    ht->hash_mask = (ht->num_buckets - 1) << ht->skip_bits;
    *ppht = ht;
}

void destroy_hashtable(hashtable_t *ht) {
    free(ht->buckets);
    free(ht);
}

int64_t probe_hashtable(const hashtable_t *ht, const struct table_t *rel, output_list_t **output, int materialize) {
    int64_t matches = 0;
    const uint32_t hash_mask = ht->hash_mask;
    const uint32_t skip_bits = ht->skip_bits;

    for (uint64_t i = 0; i < rel->num_tuples; i++) {
        type_key idx = HASH(rel->tuples[i].key, hash_mask, skip_bits);
        bucket_t *b = ht->buckets + idx;

        // Note, that it is the second bucket in the chain that might be under-full, so a simple optimization assuming
        // That all but the last bucket are full does not work, because it is the first/second bucket that could be
        // empty
        if (materialize) {
            do {
                for (uint64_t j = 0; j < b->count; j++) {
                    if (rel->tuples[i].key == b->tuples[j].key) {
                        insert_output(output, rel->tuples[i].key, b->tuples[j].payload, rel->tuples[i].payload);
                        matches++;
                    }
                }
                b = b->next;/* follow overflow pointer */
            } while (b);
        } else {
            do {
                for (uint64_t j = 0; j < b->count; j++) {
                    if (rel->tuples[i].key == b->tuples[j].key) {
                        matches++;
                    }
                }
                b = b->next;/* follow overflow pointer */
            } while (b);
        }
    }

    return matches;
}

[[nodiscard]] int64_t
probe_hashtable_no_overflow(const hashtable_t *ht, const struct table_t *rel) {
    int64_t matches = 0;
    const uint32_t hash_mask = ht->hash_mask;

    for (uint64_t i = 0; i < rel->num_tuples; i++) {
        auto key = rel->tuples[i].key;
        bucket_t const *bucket = ht->buckets + HASH(key, hash_mask, 0);
        for (uint64_t j = 0; j < bucket->count; ++j) {
            if (key == bucket->tuples[j].key) {
                matches++;
            }
        }
    }

    return matches;
}

[[nodiscard]] int64_t
probe_hashtable_no_overflow_unrolled(const hashtable_t *ht, const struct table_t *rel) {
    int64_t matches = 0;
    const uint32_t hash_mask = ht->hash_mask;

    uint64_t i = 0;
    for (; i + 4 <= rel->num_tuples; i += 4) {
        auto k0 = rel->tuples[i].key;
        auto k1 = rel->tuples[i + 1].key;
        auto k2 = rel->tuples[i + 2].key;
        auto k3 = rel->tuples[i + 3].key;

        bucket_t const *bucket0 = ht->buckets + HASH(k0, hash_mask, 0);
        bucket_t const *bucket1 = ht->buckets + HASH(k1, hash_mask, 0);
        bucket_t const *bucket2 = ht->buckets + HASH(k2, hash_mask, 0);
        bucket_t const *bucket3 = ht->buckets + HASH(k3, hash_mask, 0);

        matches += std::min(bucket0->count, static_cast<uint32_t>(k0 == bucket0->tuples[0].key) +
                                            static_cast<uint32_t>(k0 == bucket0->tuples[1].key));
        matches += std::min(bucket1->count, static_cast<uint32_t>(k1 == bucket1->tuples[0].key) +
                                            static_cast<uint32_t>(k1 == bucket1->tuples[1].key));
        matches += std::min(bucket2->count, static_cast<uint32_t>(k2 == bucket2->tuples[0].key) +
                                            static_cast<uint32_t>(k2 == bucket2->tuples[1].key));
        matches += std::min(bucket3->count, static_cast<uint32_t>(k3 == bucket3->tuples[0].key) +
                                            static_cast<uint32_t>(k3 == bucket3->tuples[1].key));
    }
    for (; i < rel->num_tuples; i++) {
        auto key = rel->tuples[i].key;
        bucket_t const *bucket = ht->buckets + HASH(key, hash_mask, 0);
        matches += std::min(bucket->count, static_cast<uint32_t>(key == bucket->tuples[0].key) +
                                           static_cast<uint32_t>(key == bucket->tuples[1].key));
    }

    return matches;
}

void init_bucket_buffer(bucket_buffer_t **ppbuf) {
    auto overflow_buffer = (bucket_buffer_t *) malloc(sizeof(bucket_buffer_t));
    if (!overflow_buffer) {
        logger(ERROR, "Memory alloc for overflobuf failed!");
        ocall_exit(EXIT_FAILURE);
    } else {
        overflow_buffer->count = 0;
        overflow_buffer->next = nullptr;

        *ppbuf = overflow_buffer;
    }
}

void get_new_bucket(bucket_t **result, bucket_buffer_t **buf) {
    if ((*buf)->count < OVERFLOW_BUF_SIZE) {
        *result = (*buf)->buf + (*buf)->count;
        (*buf)->count++;
    } else {
        /* need to allocate new buffer */
        auto new_buf = (bucket_buffer_t *) malloc(sizeof(bucket_buffer_t));
        if (!new_buf) {
            logger(ERROR, "Memory alloc for new_buf failed!");
            ocall_exit(EXIT_FAILURE);
        }
        new_buf->count = 1;
        new_buf->next = *buf;
        *buf = new_buf;
        *result = new_buf->buf;
    }
}

void free_bucket_buffer(bucket_buffer_t *buf) {
    do {
        bucket_buffer_t *tmp = buf->next;
        free(buf);
        buf = tmp;
    } while (buf);
}
