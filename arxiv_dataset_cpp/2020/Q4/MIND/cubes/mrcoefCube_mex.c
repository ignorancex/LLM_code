// H. Li - August 25, 2015
// This is a stand-alone file.

//  mex -Dchar16_t=UINT16_T mrcoefCube_mex.c
//  mex -Dchar16_t=UINT16_T CFLAGS="\$CFLAGS -std=c99" mrcoefCube_mex.c
#include "mex.h"
#include <math.h>

void mrcoefCube(double*, const double*, int, int, const double*, const double*, int);

// coef = mrcoefCube_mex(y, st, ed)
// it works only for 2d
// y:  data, a matrix
// st: upper left end indices of input cubes, start from 0, 2-column matrix (row-indices, column-indices)
// ed: lower right end indices of input cubes, start from 0, 2-column matrix (row-indices, column-indices)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *y  = mxGetPr(prhs[0]);
    int m      = mxGetM(prhs[0]);
    int n      = mxGetN(prhs[0]);
    double *st = mxGetPr(prhs[1]);
    double *ed = mxGetPr(prhs[2]);
    int nCube  = mxGetM(prhs[1]);

    plhs[0] = mxCreateDoubleMatrix(nCube, 1, mxREAL);
    double *coef = mxGetPr(plhs[0]);
    
    mrcoefCube(coef, y, m, n, st, ed, nCube);
    
    return;
}

// multiscale coefficients
void mrcoefCube(double *coef,                                  /* multiscale coefficients (output) */
                const double *y, int m, int n,                 /* whole data */
                const double *st, const double *ed, int nCube) /* specifying system of cubes */
{
    // cumulative sum along each column
    double *cs = malloc((m+1)*n*sizeof(double));
    for (int j = 0; j < n; ++j) {
        cs[j*(m+1)] = 0;
        for (int i = 0; i < m; ++i) {
            cs[(i+1)+j*(m+1)] = y[i+j*m] + cs[i+j*(m+1)];
        }
    }
    
    for (int i = 0; i < nCube; ++i) {
        coef[i] = 0;
        for (int j = (int)st[i+nCube]; j <= (int)ed[i+nCube] ; ++j) {
            coef[i] += cs[(int)ed[i]+1+j*(m+1)] - cs[(int)st[i]+j*(m+1)];
        }
        coef[i] = coef[i] / sqrt((ed[i] - st[i] + 1)*(ed[i+nCube] - st[i+nCube] + 1));
    }
    free(cs);
    
    return;
}