#include "dgetrf.hpp"

INTE_TYPE my_dgetrf(INTE_TYPE m, INTE_TYPE n, REAL_TYPE *A, INTE_TYPE lda, INTE_TYPE* P)
{
    return LAPACKE_dgetrf(CblasColMajor, m, n, A, lda, P);
}
INTE_TYPE my_dgetrf(INTE_TYPE m, INTE_TYPE n, const View_ColMat<REAL_TYPE> &ViewA, INTE_TYPE* P)
{
    return LAPACKE_dgetrf(CblasColMajor, m, n, ViewA.v, ViewA.ld, P);
}