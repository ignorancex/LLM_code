#include "sylvester.hpp"

INTE_TYPE symsyl(INTE_TYPE sgn, ColMat<REAL_TYPE> &X, const ColMat<REAL_TYPE> &A)
{
    return my_dtrsyl('N', 'N', sgn, A, A, X);
    // A is assumed to be quasi-Upper triangular.
}
INTE_TYPE symsyl(INTE_TYPE sgn, const View_ColMat<REAL_TYPE> &X, const View_ColMat<REAL_TYPE> &A)
{
    return my_dtrsyl('N', 'N', sgn, A, A, X);
    // A is assumed to be quasi-Upper triangular.
}
INTE_TYPE symsyl(INTE_TYPE sgn, ColMat<REAL_TYPE> &X, const ColMat<REAL_TYPE> &A, SchurCanonicalFactor &SCF, ColMat<REAL_TYPE> &W)
{
    INTE_TYPE info;
    SCF.factor(A);
    congruence(CblasTrans, X, SCF.svec, X, W);
    info = symsyl(sgn, X, SCF.uppt);
    congruence(CblasNoTrans, X, SCF.svec, X, W);
    return info;
}
INTE_TYPE symsyl(INTE_TYPE sgn, ColMat<REAL_TYPE> &X, const ColMat<REAL_TYPE> &A, const ColMat<REAL_TYPE> &C, SchurCanonicalFactor &SCF, ColMat<REAL_TYPE> &W)
{
    INTE_TYPE info;
    SCF.factor(A);
    congruence(CblasTrans, C, SCF.svec, X, W);
    info = symsyl(sgn, X, SCF.uppt);
    congruence(CblasNoTrans, X, SCF.svec, X, W);
    return info;
}

INTE_TYPE symsyl(INTE_TYPE sgn, const View_ColMat<REAL_TYPE> &X, const View_ColMat<REAL_TYPE> &A, const View_ColMat<REAL_TYPE> &C, SchurCanonicalFactor &SCF, const View_ColMat<REAL_TYPE> &W)
{
    INTE_TYPE info;
    SCF.factor(A);
    congruence(CblasTrans, C, SCF.svec, X, W);
    info = symsyl(sgn, X, SCF.uppt);
    congruence(CblasNoTrans, X, SCF.svec, X, W);
    return info;
}
