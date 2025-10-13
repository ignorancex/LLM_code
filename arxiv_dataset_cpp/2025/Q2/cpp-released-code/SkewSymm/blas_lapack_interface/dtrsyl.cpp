#include "dtrsyl.hpp"

INTE_TYPE my_dtrsyl(INTE_TYPE layout, CHAR_TYPE trana, CHAR_TYPE tranb, INTE_TYPE sgn, INTE_TYPE m, INTE_TYPE n,
                    REAL_TYPE *A, INTE_TYPE lda, REAL_TYPE *B, INTE_TYPE ldb, REAL_TYPE *C, INTE_TYPE ldc, REAL_TYPE *scale)
{
    return LAPACKE_dtrsyl(layout, trana, tranb, sgn, m, n, A, lda, B, ldb, C, ldc, scale);
}

INTE_TYPE my_dtrsyl(CHAR_TYPE trana, CHAR_TYPE tranb, INTE_TYPE sgn, INTE_TYPE m, INTE_TYPE n,
                    REAL_TYPE *A, INTE_TYPE lda, REAL_TYPE *B, INTE_TYPE ldb, REAL_TYPE *C, INTE_TYPE ldc, REAL_TYPE *scale)
{
    return LAPACKE_dtrsyl(CblasColMajor, trana, tranb, sgn, m, n, A, lda, B, ldb, C, ldc, scale);
}

INTE_TYPE my_dtrsyl(CHAR_TYPE trana, CHAR_TYPE tranb, INTE_TYPE sgn, const ColMat<REAL_TYPE> &A, const ColMat<REAL_TYPE> &B, ColMat<REAL_TYPE> &C)
{
    REAL_TYPE scale = 0.0;
    INTE_TYPE info;

    info = LAPACKE_dtrsyl(CblasColMajor, trana, tranb, sgn, C.r, C.c, A.v, A.r, B.v, B.r, C.v, C.r, &scale);
    scal(1.0 / scale, C);
    return info;
}
INTE_TYPE my_dtrsyl(CHAR_TYPE trana, CHAR_TYPE tranb, INTE_TYPE sgn, const View_ColMat<REAL_TYPE> &A, const View_ColMat<REAL_TYPE> &B, const View_ColMat<REAL_TYPE> &C)
{
    REAL_TYPE scale = 0.0;
    INTE_TYPE info;
    info = LAPACKE_dtrsyl(CblasColMajor, trana, tranb, sgn, C.r, C.c, A.v, A.ld, B.v, B.ld, C.v, C.ld, &scale);
    scal(1.0 / scale, C);
    return info;
}