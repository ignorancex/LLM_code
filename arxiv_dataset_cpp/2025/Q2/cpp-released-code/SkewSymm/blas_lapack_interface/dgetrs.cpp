#include "dgetrs.hpp"

INTE_TYPE my_dgetrs(char TRANSA, INTE_TYPE d, INTE_TYPE c, REAL_TYPE *LU, INTE_TYPE ldlu, INTE_TYPE *P, REAL_TYPE *X, INTE_TYPE ldx)
{
    return LAPACKE_dgetrs(CblasColMajor, TRANSA, d, c, LU, ldlu, P, X, ldx);
}
INTE_TYPE my_dgetrs(char TRANSA, INTE_TYPE d, REAL_TYPE *LU, INTE_TYPE ldlu, INTE_TYPE *P, const View_ColMat<REAL_TYPE> &ViewX)
{
    return LAPACKE_dgetrs(CblasColMajor, TRANSA, d, ViewX.c, LU, ldlu, P, ViewX.v, ViewX.ld);
}
INTE_TYPE my_dgetrs(char TRANSA, INTE_TYPE *P, const View_ColMat<REAL_TYPE> &ViewLU, const View_ColMat<REAL_TYPE> &ViewX)
{
    return LAPACKE_dgetrs(CblasColMajor, TRANSA, ViewLU.r, ViewX.c, ViewLU.v, ViewLU.ld, P, ViewX.v, ViewX.ld);
}