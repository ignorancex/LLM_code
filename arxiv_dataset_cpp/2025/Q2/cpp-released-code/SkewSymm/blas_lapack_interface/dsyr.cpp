#include "dsyr.hpp"

// This implementation computes the symmetric rank-1 update A = alpha x * x' + A, where A is stored as ColMat.
// For the same computation but on the lower vector of A stored as LowTriMat with low_col_traversal, use dspr.
// Only the lower part of A is computed and overwritten in this routine.

void my_dsyr(CBLAS_UPLO UPLO, INTE_TYPE n, REAL_TYPE alpha, REAL_TYPE *x, REAL_TYPE *a, INTE_TYPE lda)
{
    cblas_dsyr(CblasColMajor, UPLO, n, alpha, x, 1, a, lda);
}

void my_dsyr(INTE_TYPE n, REAL_TYPE alpha, REAL_TYPE *x, REAL_TYPE *a, INTE_TYPE lda)
{
    cblas_dsyr(CblasColMajor, CblasLower, n, alpha, x, 1, a, lda);
}


void my_dsyr(INTE_TYPE n, REAL_TYPE alpha, REAL_TYPE *x, REAL_TYPE *a)
{
    cblas_dsyr(CblasColMajor, CblasLower, n, alpha, x, 1, a, n);
}

void my_dsyr(REAL_TYPE alpha, REAL_TYPE *x, const View_ColMat<REAL_TYPE>& ViewA)
{
    cblas_dsyr(CblasColMajor, CblasLower, ViewA.r, alpha, x, 1, ViewA.v, ViewA.ld);
}