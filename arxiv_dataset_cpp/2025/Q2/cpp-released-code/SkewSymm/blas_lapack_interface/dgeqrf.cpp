#include "dgeqrf.hpp"

INTE_TYPE my_dgeqrf(INTE_TYPE lapack_layout, INTE_TYPE m, INTE_TYPE n, REAL_TYPE *A, INTE_TYPE lda, REAL_TYPE *tau)
{
	return LAPACKE_dgeqrf(lapack_layout, m, n, A, lda, tau);
};
INTE_TYPE my_dgeqrf(INTE_TYPE m, INTE_TYPE n, REAL_TYPE *A, INTE_TYPE lda, REAL_TYPE *tau)
{
	return LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, A, lda, tau);
};
INTE_TYPE my_dgeqrf(INTE_TYPE m, INTE_TYPE n, REAL_TYPE *A, REAL_TYPE *tau)
{
	return LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, A, m, tau);
};
INTE_TYPE my_dgeqrf(ColMat<REAL_TYPE> &A, REAL_TYPE *tau)
{
	return LAPACKE_dgeqrf(LAPACK_COL_MAJOR, A.r, A.c, A.v, A.r, tau);
};
INTE_TYPE my_dgeqrf(View_ColMat<REAL_TYPE> &A, REAL_TYPE *tau)
{
	return LAPACKE_dgeqrf(LAPACK_COL_MAJOR, A.r, A.c, A.v, A.ld, tau);
};