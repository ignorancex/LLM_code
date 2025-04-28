/** 
 * Calculate all Clebsch-Gordan coefficients for a given bandlimit and save them.
 * 
 * MATLAB call form:
 *      CGTable = buildCGTable_mex(bandlimit)
 *  where
 *    bandlimit is a positive integer.
 *    Format specification of CGTable is in buildCGTable MATLAB function.
 * 
 * NOTES:
 *  (1) Code performs no input checks.
 *  (2) The coefficients are calcualted using  the method of [1]
 * 
 * REFERENCES:
 *  [1] Straub, W. O. (n.d.). Efficient Computation of Clebsch-Gordan Coefficients. Retrieved October 28, 2019, from http://vixra.org/abs/1403.0263
 * 
  */


#include <stdint.h>
#include "mex.h"
#include "clebsch_gordan_coefficients.h"


size_t bandlimit;

mxArray *build_table(const size_t bl)
{
  long l_max;
  long m1min;
  long m1max;

  double *coeffs;

  mxArray *table = mxCreateCellMatrix(bandlimit+1, 1);
  mxArray *temp[4]; 

  for (long l1 = 0; l1<=bandlimit; ++l1)
  {
    temp[0] = mxCreateCellMatrix(l1+1, 1);
    mxSetCell(table, l1, temp[0]);

    for (long l2 = 0; l2<=l1; ++l2)
    {
      l_max = l1+l2>=bandlimit ? bandlimit : l1+l2;

      temp[1] = mxCreateCellMatrix(l_max - (l1-l2) + 1, 1);
      mxSetCell(temp[0], l2, temp[1]);

      for (long l = l1-l2; l<=l_max; ++l)
      {
        temp[2] = mxCreateCellMatrix(2*l+1, 1);
        mxSetCell(temp[1], l+l2-l1, temp[2]);

        for (long m = -l; m<=l; ++m)
        {
          m1min = cg_lower_bound(l1, l2, m);
          m1max = cg_upper_bound(l1, l2, m);

          temp[3] = mxCreateDoubleMatrix(m1max-m1min+1, 1, mxREAL);
          mxSetCell(temp[2], m+l, temp[3]);
          
          coeffs = mxGetDoubles(temp[3]);
          cg_vector(l1, l2, l, m, coeffs);
        }
      }
    }
  }

  return table;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // Obtain input
  bandlimit = mxGetScalar(prhs[0]);

  // Calculate output
  plhs[0] = build_table(bandlimit);
}
