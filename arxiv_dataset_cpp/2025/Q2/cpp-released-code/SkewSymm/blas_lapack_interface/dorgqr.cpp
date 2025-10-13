#include "dorgqr.hpp"

INTE_TYPE my_dorgqr(INTE_TYPE lapack_layout, INTE_TYPE m, INTE_TYPE n, INTE_TYPE k, REAL_TYPE *HHF, INTE_TYPE ldh, REAL_TYPE *tau)
{
	return LAPACKE_dorgqr(lapack_layout, m, n, k, HHF, ldh, tau);
}
INTE_TYPE my_dorgqr(INTE_TYPE m, INTE_TYPE n, INTE_TYPE k, REAL_TYPE *HHF, INTE_TYPE ldh, REAL_TYPE *tau)
{
	return LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, k, HHF, ldh, tau);
}
INTE_TYPE my_dorgqr(INTE_TYPE m, INTE_TYPE n, INTE_TYPE k, REAL_TYPE *HHF, REAL_TYPE *tau)
{
	return LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, k, HHF, m, tau);
}
INTE_TYPE my_dorgqr(INTE_TYPE k, ColMat<REAL_TYPE> &HHF, REAL_TYPE *tau)
{
	return LAPACKE_dorgqr(LAPACK_COL_MAJOR, HHF.r, HHF.c, k, HHF.v, HHF.r, tau);
}
INTE_TYPE my_dorgqr(INTE_TYPE k, View_ColMat<REAL_TYPE> &HHF, REAL_TYPE *tau)
{
	return LAPACKE_dorgqr(LAPACK_COL_MAJOR, HHF.r, HHF.c, k, HHF.v, HHF.ld, tau);
}
INTE_TYPE my_dorgqr(ColMat<REAL_TYPE> &HHF, REAL_TYPE *tau)
{
	return LAPACKE_dorgqr(LAPACK_COL_MAJOR, HHF.r, HHF.c, HHF.c, HHF.v, HHF.r, tau);
}
INTE_TYPE my_dorgqr(View_ColMat<REAL_TYPE> &HHF, REAL_TYPE *tau)
{
	return LAPACKE_dorgqr(LAPACK_COL_MAJOR, HHF.r, HHF.c, HHF.c, HHF.v, HHF.ld, tau);
}
