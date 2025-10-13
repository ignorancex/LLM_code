#include "dexpSemiSimple.hpp"

void dexpSemiSimple(REAL_TYPE *Y, INTE_TYPE ldy, REAL_TYPE *X, INTE_TYPE ldx, dexpSemiSimplePara &Para, BOOL_TYPE forward_action)
{
    INTE_TYPE d = Para.d;
    // CmatX for the complex storage of X, CmatS for the complex workspace of the similarity computation.
    auto workCmatX = Para.work;
    auto workCmatS = Para.work + (d * d);

    Para._assignComplex(workCmatX, d, X, ldx);
    Para._dexpSemiSimple_similarity(workCmatX, d, workCmatX, d, workCmatS, d, true);
    Para._dexpSemiSimple_core(workCmatX, d, workCmatX, d, forward_action);
    Para._dexpSemiSimple_similarity(workCmatX, d, workCmatX, d, workCmatS, d, false);
    Para._retrieveReal(Y, ldy, workCmatX, d);
}

void dexpSemiSimple(const View_ColMat<REAL_TYPE> &MatY, const View_ColMat<REAL_TYPE> &MatX, dexpSemiSimplePara &Para, BOOL_TYPE forward_action)
{
    dexpSemiSimple(MatY.v, MatY.ld, MatX.v, MatX.ld, Para, forward_action);
}
void dexpSemiSimple(SkewSymmMat &MatY, SkewSymmMat &MatX, dexpSemiSimplePara &Para, BOOL_TYPE forward_action)
{
    dexpSemiSimple(MatY.v, MatY.d, MatX.v, MatX.d, Para, forward_action);
    if (MatY.lv)
    {
        MatY.mat2vec();
        MatY.vec2mat();
    }
}
