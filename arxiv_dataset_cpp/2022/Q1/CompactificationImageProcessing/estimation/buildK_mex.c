/** 
 * Compute K. See buildK docs for better description of it.
 * 
 * MATLAB call form:
 *      K = buildK_mex(UUH, UUT, bandlimit)
 * where UUH = U*(U^*) and UUT = U*(U^T). U is described in
 * the docs of buildU.
 * 
 * NOTE:
 *  Code performs no input checks.
  */


#include <stdint.h>
#include "clebsch_gordan_mex.h"

/* Typedefs */
typedef enum PART {REAL_PART, IMAG_PART} PART;

typedef size_t **** bispectrum_lookup_table;
typedef bispectrum_lookup_table blt;


/* Input variables */
size_t bandlimit;
mxComplexDouble *UUH;
mxComplexDouble *UUT;

CGTable cgt;
const mxArray *CGs;

/* Output variables */
mxDouble *K;

/* Utility variables */
blt lookup;
size_t previous_bandlimit = 0;
bool first_run = true;


/* utility functions */
void build_bisp_lookup_table()
{
    size_t index = 0;

    lookup = malloc((bandlimit+1)*sizeof(size_t ***));

    for (long l1 = 0; l1<=bandlimit; ++l1)
    {
        lookup[l1] = malloc((l1+1)*sizeof(size_t **));
        for (long l2 = 0; l2<=l1; ++l2)
        {
            lookup[l1][l2] = malloc((((l1+l2)>=bandlimit ? bandlimit : l1+l2)  - (l1-l2) + 1)*sizeof(size_t *));
            for (long l = l1-l2; l<=l1+l2 && l<=bandlimit; ++l)
            {
                lookup[l1][l2][l - (l1-l2)] = malloc(2*sizeof(size_t));
                lookup[l1][l2][l - (l1-l2)][0] = index++;
                lookup[l1][l2][l - (l1-l2)][1] = index++;
            }
        }
    }
}


void destory_bisp_lookup_table()
{
    for (long l1 = 0; l1<=previous_bandlimit; ++l1)
    {
        for (long l2 = 0; l2<=l1; ++l2)
        {
            for (long l = l1-l2; l<=l1+l2 && l<=previous_bandlimit; ++l)
            {
                free(lookup[l1][l2][l - (l1-l2)]);
            }
            free(lookup[l1][l2]);
        }
        free(lookup[l1]);
    }
    free(lookup);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Get input variables
    UUH = mxGetComplexDoubles(prhs[0]);
    UUT = mxGetComplexDoubles(prhs[1]);
    bandlimit = (size_t) mxGetScalar(prhs[2]);

    // Handle first run
    if (first_run == true)
    {
        build_bisp_lookup_table();

        mxArray *inputMxArray[1];
        inputMxArray[0] = mxCreateDoubleScalar(bandlimit);
        mexCallMATLAB(0, NULL, 1, inputMxArray, "loadCGTable");

        CGs = mexGetVariablePtr("global", "CGs");
        create_CGTable(&cgt, &CGs, bandlimit);

        previous_bandlimit = bandlimit;
        first_run = false;
    }
    
    // Handle chagne in bandlimit between runs
    if (previous_bandlimit!=bandlimit)
    {
        mxArray *inputMxArray[1];
        inputMxArray[0] = mxCreateDoubleScalar(bandlimit);
        mexCallMATLAB(0, NULL, 1, inputMxArray, "loadCGTable");

        CGs = mexGetVariablePtr("global", "CGs");

        destory_bisp_lookup_table();
        destroy_CGTable(&cgt, previous_bandlimit);

        previous_bandlimit = bandlimit;

        build_bisp_lookup_table();
        create_CGTable(&cgt, &CGs, bandlimit);
    }

    // Size of list of spehrical harmonics coefficeints (complex) of bandlimit bandlimit
    long szC = (bandlimit+1)*(bandlimit+1);
    // Size of list of spehrical harmonics coefficeints (realified) of bandlimit bandlimit
    long szR = 2*(bandlimit+1)*(bandlimit+1); 

    // Generate output
    plhs[0] = mxCreateDoubleMatrix(szR, lookup[bandlimit][bandlimit][bandlimit][1]+1, mxREAL);
    K = mxGetDoubles(plhs[0]);
    
    // Build K
    double rp, ip; // real part and imaginary part
    for (long l1 = 0; l1<=bandlimit; ++l1)
    {
        for (long l2 = 0; l2<=l1; ++l2)
        {
            for (long l = l1-l2; l<=bandlimit && l<=l1+l2; ++l)
            {
                // K1
                for (long m1 = -l1; m1<=l1; ++m1)
                {
                    rp = 0;
                    ip = 0;

                    for (long m = (-l>=m1-l2) ? -l : m1-l2; m<=l && m<=m1+l2; ++m)
                    {
                        rp += get_cg(&cgt, l1, l2, l, m, m1)*UUH[szC*(l2*(l2+1) + (m-m1)) + (l*(l+1) + m)].real;
                        ip += get_cg(&cgt, l1, l2, l, m, m1)*UUH[szC*(l2*(l2+1) + (m-m1)) + (l*(l+1) + m)].imag;
                    }

                    // Real part bisp invariant
                    K[szR*lookup[l1][l2][l - (l1-l2)][REAL_PART] + 2*(l1*(l1+1) + m1)] += rp;    // Real part SHC
                    K[szR*lookup[l1][l2][l - (l1-l2)][REAL_PART] + 2*(l1*(l1+1) + m1)+1] += ip;  // Imag part SHC
                    
                    // Imag part bisp invariant
                    K[szR*lookup[l1][l2][l - (l1-l2)][IMAG_PART] + 2*(l1*(l1+1) + m1)] += ip;    // Real part SHC
                    K[szR*lookup[l1][l2][l - (l1-l2)][IMAG_PART] + 2*(l1*(l1+1) + m1)+1] -= rp;   // Imag part SHC
                }

                // K2
                for (long m2 = -l2; m2<=l2; ++m2)
                {
                    rp = 0;
                    ip = 0;

                    for (long m = (-l>=m2-l1) ? -l : m2-l1; m<=l && m<=m2+l1; ++m)
                    {
                        rp += get_cg(&cgt, l1, l2, l, m, m-m2)*UUH[szC*(l1*(l1+1) + (m-m2)) + (l*(l+1) + m)].real;
                        ip += get_cg(&cgt, l1, l2, l, m, m-m2)*UUH[szC*(l1*(l1+1) + (m-m2)) + (l*(l+1) + m)].imag;
                    }

                    // Real part bisp invariant
                    K[szR*lookup[l1][l2][l - (l1-l2)][REAL_PART] + 2*(l2*(l2+1) + m2)] += rp;    // Real part SHC
                    K[szR*lookup[l1][l2][l - (l1-l2)][REAL_PART] + 2*(l2*(l2+1) + m2)+1] += ip;  // Imag part SHC
                    
                    // Imag part bisp invariant
                    K[szR*lookup[l1][l2][l - (l1-l2)][IMAG_PART] + 2*(l2*(l2+1) + m2)] += ip;    // Real part SHC
                    K[szR*lookup[l1][l2][l - (l1-l2)][IMAG_PART] + 2*(l2*(l2+1) + m2)+1] -= rp;   // Imag part SHC
                }

                // K3
                for (long m = -l; m<=l; ++m)
                {
                    rp = 0;
                    ip = 0;

                    for (long m1 = (-l1>=m-l2) ? -l1 : m-l2; m1<=l1 && m1<=m+l2; ++m1)
                    {
                        rp += get_cg(&cgt, l1, l2, l, m, m1)*UUT[szC*(l2*(l2+1) + (m-m1)) + (l1*(l1+1) + m1)].real;
                        ip -= get_cg(&cgt, l1, l2, l, m, m1)*UUT[szC*(l2*(l2+1) + (m-m1)) + (l1*(l1+1) + m1)].imag;
                    }

                    // Real part bisp invariant
                    K[szR*lookup[l1][l2][l - (l1-l2)][REAL_PART] + 2*(l*(l+1) + m)] += rp;    // Real part SHC
                    K[szR*lookup[l1][l2][l - (l1-l2)][REAL_PART] + 2*(l*(l+1) + m)+1] -= ip;  // Imag part SHC
                    
                    // Imag part bisp invariant
                    K[szR*lookup[l1][l2][l - (l1-l2)][IMAG_PART] + 2*(l*(l+1) + m)] += ip;    // Real part SHC
                    K[szR*lookup[l1][l2][l - (l1-l2)][IMAG_PART] + 2*(l*(l+1) + m)+1] += rp;   // Imag part SHC
                }

            }
        }
    }

}