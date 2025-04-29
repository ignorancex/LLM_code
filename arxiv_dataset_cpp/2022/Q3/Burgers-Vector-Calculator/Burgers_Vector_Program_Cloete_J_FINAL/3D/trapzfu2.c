/* F=trapzfu2(X,Y) computes the integral of Y with respect to X using
the trapezoidal method. X and Y must be vectors of the same length.
For trapzf to work properly:
X must be a STRICTLY monotonically increasing vector.
X does not need to be evenly spaced.

L. Optican, NEI/NIH/DHHS

*/

#include "mex.h"

void trapzfu2( double *area, double *xv, double *yv, int N) {
register int i, j,r;

*area = 0;
i = 0;
r = (N - 1) % 4;
for(; i < r; i++) {
// *area += (xv[i+1] - xv[i]) * (yv[i] + yv[i+1]);
*area += (xv[i+1] - xv[i]) * (yv[i+1] + yv[i]);
}

for(; i <= (N-4); i += 4) {
*area += (xv[i+1] - xv[i]) * (yv[i+1] + yv[i]);
*area += (xv[i+2] - xv[i+1]) * (yv[i+2] + yv[i+1]);
*area += (xv[i+3] - xv[i+2]) * (yv[i+3] + yv[i+2]);
*area += (xv[i+4] - xv[i+3]) * (yv[i+4] + yv[i+3]);
}
*area = *area / 2;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
int N, N1, N2;
double *area;
double *xv, *yv ;

N1 = mxGetNumberOfElements(prhs[0]);
N2 = mxGetNumberOfElements(prhs[1]);
if (N1 != N2)
mexErrMsgTxt("Input x and y must have the same length");
else
N = N1;
if (N < 2)
mexErrMsgTxt("The minimum length is 2");

xv = mxGetPr(prhs[0]);
yv = mxGetPr(prhs[1]);
plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
area = mxGetPr(plhs[0]);

trapzfu2(area, xv, yv, N);
}