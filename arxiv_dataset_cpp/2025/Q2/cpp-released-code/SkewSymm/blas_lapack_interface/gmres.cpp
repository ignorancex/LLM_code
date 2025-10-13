#include "gmres.hpp"

REAL_TYPE *_GMRES_Work(INTE_TYPE n, INTE_TYPE m)
{
    return new REAL_TYPE[3 * m + 3 + 2 * n];
}

REAL_TYPE _GMRES_NormEst(REAL_TYPE alpha, INTE_TYPE k, ColMat<REAL_TYPE> &H, REAL_TYPE *s)
{
    REAL_TYPE normqr = alpha * alpha;
    for (INTE_TYPE i = 0; i < k; i++)
        normqr += pow(s[i] / H.fele(i, i), 2);
    return sqrt(normqr);
}

void _GMRES_Update(REAL_TYPE *x, INTE_TYPE n, INTE_TYPE k, ColMat<REAL_TYPE> &H, REAL_TYPE *s, ColMat<REAL_TYPE> &V, REAL_TYPE *y)
{
    // Fast column access is assumed in ViewH and ViewV.
    memcpy(y, s, sizeof(REAL_TYPE) * n);

    // Backsolve:
    for (INTE_TYPE i = k; i >= 0; i--)
    {
        y[i] /= H.fele(i, i);
        for (INTE_TYPE j = i - 1; j >= 0; j--)
            y[j] -= H.fele(j, i) * y[i];
    }

    for (INTE_TYPE j = 0; j <= k; j++)
        my_daxpy(n, y[j], V.fcol(j), x);
}

void _GMRES_Generate_Rotation(REAL_TYPE &dx, REAL_TYPE &dy, REAL_TYPE &cs, REAL_TYPE &sn)
{
    if (dy == 0.0)
    {
        cs = 1.0;
        sn = 0.0;
    }
    else if (abs(dy) > abs(dx))
    {
        REAL_TYPE temp = dx / dy;
        sn = 1.0 / sqrt(1.0 + temp * temp);
        cs = temp * sn;
    }
    else
    {
        REAL_TYPE temp = dy / dx;
        cs = 1.0 / sqrt(1.0 + temp * temp);
        sn = temp * cs;
    }
};

void _GMRES_Rotate(REAL_TYPE &dx, REAL_TYPE &dy, REAL_TYPE &cs, REAL_TYPE &sn)
{
    REAL_TYPE temp = cs * dx + sn * dy;
    dy = -sn * dx + cs * dy;
    dx = temp;
};