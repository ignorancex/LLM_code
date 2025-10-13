#include "pivotedLUFac.hpp"

void solverPLU(REAL_TYPE *X, INTE_TYPE ldX, INTE_TYPE colX, const PivotedLUFactor &plu)
{
    // assert(r == c);
    // assert(f);
    my_dgetrs('N', plu.r, colX, plu.M.v, plu.r, plu.P, X, ldX);
}
void solverPLU(const View_ColMat<REAL_TYPE> ViewX, const PivotedLUFactor &plu)
{
    my_dgetrs('N', plu.r, ViewX.c, plu.M.v, plu.r, plu.P, ViewX.v, ViewX.ld);
}
