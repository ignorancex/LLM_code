#include "radix/prj_params.h"
#include "radix/radix_join.h"
#include "rdtscpWrapper.h"
#include "Logger.hpp"

#ifdef ENCLAVE
#include "ocalls_t.h"
#else
#include "ocalls.hpp"
#endif

#ifdef SIMD_SORT
#include "avx512-64bit-qsort.hpp"
#else
#include <algorithm>
#endif

void is_table_sorted(const struct table_t * const t)
{
    intkey_t curr = 0;

    for (uint64_t i = 0; i < t->num_tuples; i++)
    {
        intkey_t key = t->tuples[i].key;
        if (key < curr)
        {
            logger(ERROR, "ERROR! Table is not sorted!");
            ocall_exit(EXIT_FAILURE);
        }
        curr = key;
    }
}

static inline int __attribute__((always_inline))
compare(const void * k1, const void * k2)
{
    const type_key *r1 = (const type_key *) k1;
    const type_key *r2 = (const type_key *) k2;
    return (int) (*r1 - *r2);
}

inline
bool cmp(const row_t& a, const row_t& b) {
    return a.key < b.key;
}

static int64_t merge(const relation_t *relR, const relation_t *relS)
{
    int64_t matches = 0;
    const row_t *tr = relR->tuples;
    const row_t *ts;
    const row_t *gs = relS->tuples;
    int64_t tr_s = 0;
    int64_t gs_s = 0;
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
    return matches;
}


int64_t sortmerge_join(const struct table_t * const R,
                       const struct table_t * const S,
                       struct table_t * const tmpR,
                       uint32_t num_radix_bits,
                       #ifdef CHUNKED_TABLE
                       chunked_table_t *output,
                       #else
                       output_list_t **output,
                       #endif
                       uint64_t * build_timer,
                       uint64_t * join_timer,
                       int materialize)
{
    (void) (tmpR);

    const uint64_t numR = R->num_tuples;
    const uint64_t numS = S->num_tuples;

    uint64_t build_start_time = rdtscp_s();
    /* SORT PHASE */
    if (!R->sorted)
    {
#ifdef SIMD_SORT
        avx512_qsort<uint64_t>((uint64_t *) R->tuples, numR);
#else
        std::sort(R->tuples, R->tuples + numR, cmp);
#endif
        // qsort(R->tuples, numR, sizeof(tuple_t*), compare);
    }
    if (!S->sorted)
    {
#ifdef SIMD_SORT
        avx512_qsort<uint64_t>((uint64_t *) S->tuples, numS);
#else
        std::sort(S->tuples, S->tuples + numS, cmp);
#endif
        // qsort(S->tuples, numS, sizeof(tuple_t*), compare);
    }
    uint64_t in_between_time = rdtscp_s();
    if (build_timer != nullptr) {
        *build_timer += in_between_time - build_start_time;\
    }
    /* MERGE PHASE */
    auto matches = merge(R, S);

    if (join_timer != nullptr) {
        *join_timer += rdtscp_s() - in_between_time;
    }

    return matches;
}

result_t* RSM (const table_t * relR, const table_t * relS, const joinconfig_t *config)
{
    return join_init_run(relR, relS, sortmerge_join, config);
}
