#include "dger.hpp"

// This implementation computes the vector outer product A = alpha x * y' + A.

void my_dger(INTE_TYPE m, INTE_TYPE n, REAL_TYPE alpha, REAL_TYPE *x, REAL_TYPE *y, REAL_TYPE *a, INTE_TYPE lda)
{
    cblas_dger(CblasColMajor, m, n, alpha, x, 1, y, 1, a, lda);
}

void my_dger(INTE_TYPE m, INTE_TYPE n, REAL_TYPE alpha, REAL_TYPE *x, REAL_TYPE *y, const View_ColMat<REAL_TYPE>& ViewA)
{
    cblas_dger(CblasColMajor, m, n, alpha, x, 1, y, 1, ViewA.v, ViewA.ld);
}

