/*==========================================================
 * fastWht.c
 *
 * Calculates the Walsh-Hadamard transform (WHT) 
 * of 1xN vector (x)
 * and outputs a 1xN vector of coefficients (y)
 *
 * The calling syntax is:
 *
 *		y = fastWht(x)
 *
 * EXAMPLE:
 * 
 *      x = rand(1, 16);
 *      y = fastWht(x);
 *      xHat = fastWht(y); % the transform should reproduce x
 *
 * This is a MEX-file for MATLAB.
 *
 * Copyright (c) 2021, Tin Vlasic
 *
 *========================================================*/

#include "mex.h"
#include "stdio.h"
#include "stdlib.h"
#include "matrix.h"
#include "math.h"
#include "string.h"

// the computational routine (c function)
void fastWht(double *x,  double *y, int n){
    int L = 1;
    int K, J, M, j, nStage;
    double *tmp, *pResult, *xCpy;

    // allocate memory for middle stage results
    pResult = (double*) mxMalloc( n * sizeof(double));
    xCpy = (double*) mxMalloc( n * sizeof(double));
    
    // copy vector x to tmp memory location
    xCpy = memcpy(xCpy, x, n * sizeof(double));
    
    // first stage coefficients
    for (j = 0; j <= n-2; j += 2){
        xCpy[j] = xCpy[j] + xCpy[j+1];
        xCpy[j+1] = xCpy[j] - 2*xCpy[j+1];
    }

    // calculate coefficients for the ith stage specified by nStage
    for (nStage = 2; nStage<=log2(n); nStage++){
        K=1; J=0;
        M = (int) pow(2, L);
        while(K<n){
            for(j = J+1; j<=J+M-1; j += 2){
                pResult[K-1] = xCpy[j-1] + xCpy[j+M-1];
                pResult[K] = xCpy[j-1] - xCpy[j+M-1];
                pResult[K+1] = xCpy[j] - xCpy[j+M];
                pResult[K+2] = xCpy[j] + xCpy[j+M];
                K += 4;
            }
            J += 2 * M;
        }
        tmp = xCpy;
        xCpy = pResult;
        pResult = tmp;
        L += 1;
    }

    // scaling of coefficients
    for (j = 0; j < n ; j++) {
        y[j] = xCpy[j] / sqrt(n);
    }

    // free tmp memory
    mxFree(pResult);
    mxFree(xCpy);
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    double *x;      /* 1xN input matrix */
    size_t ncols;   /* size of matrix */
    double *y;      /* output matrix */
    
    /* check for proper number of arguments */
    if(nrhs!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","One input required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","One output required.");
    }
    /* make sure the second input argument is type double */
    if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Input vector must be type double.");
    }
    /* check that number of rows in second input argument is 1 */
    if(mxGetM(prhs[0])!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notRowVector","Input must be a row vector.");
    }
    
    /* create a pointer to the real data in the input matrix  */
    x = mxGetPr(prhs[0]);
    
    /* get dimensions of the input matrix */
    ncols = mxGetN(prhs[0]);
    
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(1,(mwSize)ncols, mxREAL);
    
    /* get a pointer to the real data in the output matrix */
    y = mxGetPr(plhs[0]);
    
    /* call the computational routine */
    fastWht(x, y, (mwSize)ncols);
}
