#include "dormqr.hpp"

INTE_TYPE my_dormqr(INTE_TYPE lapack_layout, CHAR_TYPE lapack_side, CHAR_TYPE lapack_trans, INTE_TYPE m, INTE_TYPE n, INTE_TYPE k,
					const REAL_TYPE *HHF, INTE_TYPE ldh, const REAL_TYPE *tau, REAL_TYPE *C, INTE_TYPE ldc)
{
	return LAPACKE_dormqr(lapack_layout, lapack_side, lapack_trans, m, n, k, HHF, ldh, tau, C, ldc);
};
INTE_TYPE my_dormqr(CHAR_TYPE lapack_side, CHAR_TYPE lapack_trans, INTE_TYPE m, INTE_TYPE n, INTE_TYPE k,
					const REAL_TYPE *HHF, INTE_TYPE ldh, const REAL_TYPE *tau, REAL_TYPE *C, INTE_TYPE ldc)
{
	return LAPACKE_dormqr(LAPACK_COL_MAJOR, lapack_side, lapack_trans, m, n, k, HHF, ldh, tau, C, ldc);
};
INTE_TYPE my_dormqr(CHAR_TYPE lapack_side, CHAR_TYPE lapack_trans, const ColMat<REAL_TYPE> &HHF, const REAL_TYPE *tau, ColMat<REAL_TYPE> &C)
{
	return LAPACKE_dormqr(LAPACK_COL_MAJOR, lapack_side, lapack_trans, C.r, C.c, HHF.c, HHF.v, HHF.r, tau, C.v, C.r);
};
INTE_TYPE my_dormqr(CHAR_TYPE lapack_side, CHAR_TYPE lapack_trans, const View_ColMat<REAL_TYPE> &HHF, const REAL_TYPE *tau, View_ColMat<REAL_TYPE> &C)
{
	return LAPACKE_dormqr(LAPACK_COL_MAJOR, lapack_side, lapack_trans, C.r, C.c, HHF.c, HHF.v, HHF.ld, tau, C.v, C.ld);
};