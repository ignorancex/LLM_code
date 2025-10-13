#include "dgees.hpp"

// This implementation compute the real Schur decomposition of a real, non-symmetric matrix.

INTE_TYPE _dgees_select_nonzero(const REAL_TYPE *pr, const REAL_TYPE *pi)
{
    if (abs(*pr) + abs(*pi) > 1e-7)
        return 1;
    else
        return 0;
}

INTE_TYPE _dgees_select_nonreal(const REAL_TYPE *pr, const REAL_TYPE *pi)
{
    if (abs(*pi) > 1e-7)
        return 1;
    else
        return 0;
}

INTE_TYPE my_dgees(CHAR_TYPE JOBVS, CHAR_TYPE SORT, INTE_TYPE (*select)(const REAL_TYPE *, const REAL_TYPE *),
                   INTE_TYPE n, REAL_TYPE *a_arr, INTE_TYPE lda, INTE_TYPE *s_dim, REAL_TYPE *wr, REAL_TYPE *wi, REAL_TYPE *v_arr, INTE_TYPE ldv)
{
    return LAPACKE_dgees(CblasColMajor, JOBVS, SORT, select, n, a_arr, lda, s_dim, wr, wi, v_arr, ldv);
}