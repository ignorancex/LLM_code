/** 
 * Calculate the bispectrum and its gradient.
 * 
 * MATLAB call form:
 *      p = powerSpectrum_mex(shcAbsSqr, bandlimit)
 *  where
 *      shcAbsSqr   column or row complex array of length (bandlimit+1)^2 of the squared absolute value of the spherical harmonics coefficients
 *      bandlimit   scalar, the bandlimit of the function represented by shc
 *      p           bandlimit+1 column array, the power spectrum of shc
 * 
 * NOTES:
 *  (1) The code performs no input checks.
 *  (2) This function is wrapped by powerSpectrum.m.
 * 
  */

#include <stdint.h>
#include "mex.h"


double *shcAbsSqr;
size_t bandlimit;

double *pow_spec;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // Obtain input
  shcAbsSqr = mxGetDoubles(prhs[0]);
  bandlimit = mxGetScalar(prhs[1]);
  
  // Create output
  plhs[0] = mxCreateDoubleMatrix(bandlimit+1, 1, mxREAL);
  pow_spec = mxGetDoubles(plhs[0]);

  // Compute output
  long current;
  for (long l = 0; l<=bandlimit; ++l)
  {
    current = l*(l+1);
    for (long m = -l; m<=l; ++m)
    {
      pow_spec[l] += shcAbsSqr[current + m];
    }
  }
}
