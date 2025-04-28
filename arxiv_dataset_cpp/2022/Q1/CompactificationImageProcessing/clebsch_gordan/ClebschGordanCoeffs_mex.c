/** 
 * Calculate the Clebsch-Gordan coefficients of order (l1, l2, l, m)
 * 
 * MATLAB call form:
 *      output = ClebschGordanCoeffs_mex(l1, l2, l, m)
 *  where
 *      l1, l2, l, m are integers satisfying
 *        l1>=0, l2>=0, abs(l1-l2)<=l<=l1+l2, and abs(m)<=l.
 * 
 * NOTES:
 *  (1) Code performs no input checks.
 *  (2) The coefficients are calcualted using  the method of [1].
 * 
 * REFERENCES:
 *  [1] Straub, W. O. (n.d.). Efficient Computation of Clebsch-Gordan Coefficients. Retrieved October 28, 2019, from http://vixra.org/abs/1403.0263
  */

#include <stdint.h>
#include "mex.h"
#include "clebsch_gordan_coefficients.h"


long l1, l2, l, m;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Obtain input
    l1 = mxGetScalar(prhs[0]);
    l2 = mxGetScalar(prhs[1]);
    l = mxGetScalar(prhs[2]);
    m = mxGetScalar(prhs[3]);

    // Generate output
    plhs[0] = mxCreateDoubleMatrix(cg_upper_bound(l1, l2, m) - cg_lower_bound(l1, l2, m) + 1, 1, mxREAL);
    cg_vector(l1, l2, l, m, mxGetDoubles(plhs[0]));
}
