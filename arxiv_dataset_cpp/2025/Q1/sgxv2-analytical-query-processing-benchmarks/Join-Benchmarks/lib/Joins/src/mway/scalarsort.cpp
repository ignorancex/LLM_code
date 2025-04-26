/** @version $Id$ */

#if defined(__cplusplus)
#include <algorithm>            /* sort() */
#else
#include <stdlib.h>             /* qsort() */
#endif

#include "mway/scalarsort.h"
// #include "avx512-64bit-qsort.hpp"

#if defined(__cplusplus)  /* C++ std::sort() */

void
scalarsort_int64(int64_t ** inputptr, int64_t ** outputptr, uint64_t nitems)
{
    int64_t * in = *inputptr;
    std::sort(in, in + nitems);

    *inputptr = *outputptr;
    *outputptr = in;
}

void
scalarsort_int32(int32_t ** inputptr, int32_t ** outputptr, uint64_t nitems)
{
    int32_t * in = *inputptr;
    std::sort(in, in + nitems);

    *inputptr = *outputptr;
    *outputptr = in;
}

/** for sorting tuples on key */
static inline int
compare_tuples(const tuple_t a, const tuple_t b)
{
    return (a.key < b.key);
}

void
scalarsort_tuples(tuple_t ** inputptr, tuple_t ** outputptr, uint64_t nitems)
{
    tuple_t * in  = *inputptr;
    tuple_t * out = *outputptr;

    std::sort(in, in + nitems, compare_tuples);

    *inputptr = out;
    *outputptr = in;
}

void simdsort_tuples(tuple_t ** inputptr, tuple_t ** outputptr, uint64_t nitems) {
    tuple_t * in  = *inputptr;

    // avx512_qsort<uint64_t>((uint64_t *) in, nitems);
    //qsort(in, nitems, sizeof(tuple_t), compare_tuples);

    *inputptr = *outputptr;
    *outputptr = in;
}

#else /* sorting with C qsort() */

static inline int __attribute__((always_inline))
compare_int64(const void * k1, const void * k2)
{
    int64_t val = *(int64_t *)k1 - *(int64_t *)k2;
    int ret = 0;
    if(val < 0)
        ret = -1;
    else if(val > 0)
        ret = 1;

    return ret;
}

static inline int __attribute__((always_inline))
compare_int32(const void * k1, const void * k2)
{
    int val = *(int32_t *)k1 - *(int32_t *)k2;
    return val;
}

static inline int __attribute__((always_inline))
compare_tuples(const void * k1, const void * k2)
{
    int val = ((tuple_t *)k1)->key - ((tuple_t *)k2)->key;
    return val;
}

void
scalarsort_int64(int64_t ** inputptr, int64_t ** outputptr, uint64_t nitems)
{
    int64_t * in  = *inputptr;
    qsort(in, nitems, sizeof(int64_t), compare_int64);

    *inputptr  = *outputptr;
    *outputptr = in;
}

void
scalarsort_int32(int32_t ** inputptr, int32_t ** outputptr, uint64_t nitems)
{
    int32_t * in  = *inputptr;
    qsort(in, nitems, sizeof(int32_t), compare_int32);

    *inputptr  = *outputptr;
    *outputptr = in;
}

void
scalarsort_tuples(tuple_t ** inputptr, tuple_t ** outputptr, uint64_t nitems)
{
    tuple_t * in  = *inputptr;

    qsort(in, nitems, sizeof(tuple_t), compare_tuples);

    *inputptr = *outputptr;
    *outputptr = in;
}

#endif
