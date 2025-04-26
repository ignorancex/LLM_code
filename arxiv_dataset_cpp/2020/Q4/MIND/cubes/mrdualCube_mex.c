// H. Li - June 3, 2015
// This is a stand-alone file.

//  mex -Dchar16_t=UINT16_T mrdualCube_mex.c
//  mex -Dchar16_t=UINT16_T CFLAGS="\$CFLAGS -std=c99" mrdualCube_mex.c
#include "mex.h"
#include <math.h>

void mrdualCube(double*, int, int, const double*, const double*, const double*, int);

// u = mrdualCube_mex(m, n, coef, st, ed)
// it works only for 2d
// m:    number of rows of original space
// n:    number of columns of original space
// coef: coefficients
// st:   left end indices of input intervals, start from 0, 2-column vector (row-indices, column-indices)
// ed:   right end indices of input intervals, start from 0, 2-column vector (row-indices, column-indices)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int m        = (int)mxGetScalar(prhs[0]);
    int n        = (int)mxGetScalar(prhs[1]);
    double *coef = mxGetPr(prhs[2]);
    double *st   = mxGetPr(prhs[3]);
    double *ed   = mxGetPr(prhs[4]);
    int nCube    = mxGetM(prhs[3]);
        
    plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
    double *u = mxGetPr(plhs[0]);
    
    mrdualCube(u, m, n, coef, st, ed, nCube);
    
    return;
}

// multiscale coefficients
void mrdualCube(double *u, int m, int n,                       /* original space (output) */
                const double *coef,                            /* multiscale coefficients */
                const double *st, const double *ed, int nCube) /* specifying system of cubes */
{
    for (int k = 0; k < m*n; ++k) {
        u[k] = 0.;
    }
    double sc;
    for (int k = 0; k < nCube; ++k) {
        sc = coef[k] / sqrt((ed[k] - st[k] + 1)*(ed[k+nCube] - st[k+nCube] + 1));
        for (int i = (int)st[k]; i <= (int)ed[k]; ++i) {
            for (int j = (int)st[k+nCube]; j <= (int)ed[k+nCube]; ++j) {
                u[i+j*m] += sc;
            }
        }
    }
    return;
}