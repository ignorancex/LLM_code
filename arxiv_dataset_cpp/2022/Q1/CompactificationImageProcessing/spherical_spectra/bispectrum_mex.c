/** 
 * Calculate the bispectrum and its gradient.
 * 
 * MATLAB call form:
 *      b = bispectrum_mex(shc, bandlimit, CGs)
 *      [b, grad_i, grad_j, grad_vals, grad_nnz, grad_rows_no, grad_cols_no] = bispectrum_mex(shc, bandlimit, CGs)
 *      [b, grad_i, grad_j, grad_vals, grad_nnz, grad_rows_no, grad_cols_no] = bispectrum_mex(shc, bandlimit, CGs)
 *  where
 *      shc                           column or row complex array of length 
 *                                    (bandlimit+1)^2 of spherical harmonics coefficients
 *      bandlimit                     scalar, the bandlimit of the function represented 
 *                                    by shc
 *      CGs                           Clebsch-Gordan coefficients. Format documented in bispectrum MATLAB function.
 *      b                             bispectrum column vector, containing the bispectrum 
 *                                    invariants b_{l1,l2,l} for 
 *                                         0<=l1<=bandlimit, 0<=l2<=l1, abs(l1-l2)<=l<=min(bandlmit, l1+l2)
 *                                    separated to real and complex parts and ordered in lexigraphical order.
 *                                    That is, 
 *                                      b = [real(b_{0,0,0}); imag(b_{0,0,0}); ...
 *                                          real(b_{1,0,0}); imag(b_{1,0,0}); ...]
 *      grad_i, grad_j, grad_vals      grad_nnz x 1 arrays, such that the gardient of the bispectrum satisfies
 *                                        grad(grad_i(k), grad_j(k)) = grad_vals(k)
 *      grad_nnz                      number of non-zero elements in the gradient
 *      grad_rows_no, grad_cols_no    number of rows and columns of the gradient
 * 
 * 
 * NOTES:
 *  (1) The code performs no input checks.
 *  (2) This function is wrapped by bispectrum.m.
 * 
  */

#include <stdint.h>
#include "clebsch_gordan_mex.h"


/* Typedefs */
typedef enum PART {REAL_PART, IMAG_PART} PART;

typedef size_t **** bispectrum_lookup_table;
typedef bispectrum_lookup_table blt;


/* Input variables */
mxComplexDouble *shc;
size_t bandlimit;

CGTable cgt;

/* Output variables */
double *b;
double *grad_i;
double *grad_j;
double *grad_vals;
long grad_nnz; // Number of non-zero elements in the bispectrum gradient matrix
long grad_rows_no;
long grad_cols_no;


/* Utility variables */
blt lookup;
bool first_run = true;
size_t previous_bandlimit;

size_t nnz_ind;


/* utility functions */
void build_bisp_lookup_table()
{
    grad_nnz = 0;
    
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

                if (l1==l2 && l2==l)
                {
                    grad_nnz += 2*l+1;
                }
                else if (l1==l2)
                {
                    grad_nnz += 2*(l1 + l) + 2;
                }
                else if (l1==l)
                {
                    grad_nnz += 2*(l1 + l2) + 2;
                }
                else if (l2==l)
                {
                    grad_nnz += 2*(l2 + l1) + 2;
                }
                else
                {
                    grad_nnz += 2*(l1+l2+l) + 3;
                }
            }
        }
    }

    grad_nnz *= 4;
    grad_rows_no = index;
    grad_cols_no = 2*(bandlimit+1)*(bandlimit+1);
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

mxComplexDouble get_shc(const long l, const long m)
{
    return shc[l*(l+1)+m];
}

long bisp_lookup(const long l1, const long l2, const long l, const PART part)
{
    return lookup[l1][l2][l - (l1-l2)][part];
}


/* Functions calculating the bispectrum and its gradient */

long maximum(const long a, const long b)
{
    return (b>=a ? b : a);
}

long minimum(const long a, const long b)
{
    return (b>=a ? a : b);
}

void bisp()
{
    for (long l1 = 0; l1<=bandlimit; ++l1)
    {
        for (long l2 = 0; l2<=l1; ++l2)
        {
            for (long l = l1-l2; l<=l1+l2 && l<=bandlimit; ++l)
            {
                long ind_real = bisp_lookup(l1, l2, l, REAL_PART);
                long ind_imag = bisp_lookup(l1, l2, l, IMAG_PART);

                double b_real;
                double b_imag;
                for (long m = -l; m<=l; ++m)
                {
                    b_real = 0;
                    b_imag = 0;
                    
                    for (long m1 = maximum(-l1, m-l2); m1<=minimum(l1, m+l2); ++m1)
                    {
                        b_real += get_cg(&cgt, l1, l2, l, m, m1) 
                                    * (get_shc(l1, m1).real * get_shc(l2, m-m1).real 
                                        - get_shc(l1, m1).imag * get_shc(l2, m-m1).imag);
                        b_imag += - get_cg(&cgt, l1, l2, l, m, m1) 
                                    * (get_shc(l1, m1).real * get_shc(l2, m-m1).imag 
                                        + get_shc(l1, m1).imag * get_shc(l2, m-m1).real);
                    }
                    
                    b[ind_real] += get_shc(l, m).real * b_real - get_shc(l, m).imag * b_imag;
                    b[ind_imag] += get_shc(l, m).imag * b_real + get_shc(l, m).real * b_imag;
                }
            }
        }
    }
}


void calc_l_sum(const long l1, const long l2, const long l, const long n, double *l_sum_real, double *l_sum_imag)
{
    for (long m1 = maximum(-l1, n-l2); m1<=minimum(l1, n+l2); ++m1)
    {
        *l_sum_real += get_cg(&cgt, l1, l2, l, n, m1) *
                        (get_shc(l1, m1).real * get_shc(l2, n-m1).real 
                        - get_shc(l1, m1).imag * get_shc(l2, n-m1).imag);
        *l_sum_imag += -get_cg(&cgt, l1, l2, l, n, m1) *
                        (get_shc(l1, m1).imag * get_shc(l2, n-m1).real
                        + get_shc(l1, m1).real * get_shc(l2, n-m1).imag);
    }
}


void calc_l1_sum(const long l1, const long l2, const long l, const long n, double *l1_sum_real, double *l1_sum_imag)
{
    for (long m = maximum(-l, n-l2); m<=minimum(l, n+l2); ++m)
    {
        *l1_sum_real += get_cg(&cgt, l1, l2, l, m, n) *
                        (get_shc(l, m).real * get_shc(l2, m-n).real 
                        + get_shc(l, m).imag * get_shc(l2, m-n).imag);
        *l1_sum_imag += get_cg(&cgt, l1, l2, l, m, n) *
                        (get_shc(l, m).imag * get_shc(l2, m-n).real
                        - get_shc(l, m).real * get_shc(l2, m-n).imag);
    }
}


void calc_l2_sum(const long l1, const long l2, const long l, const long n, double *l2_sum_real, double *l2_sum_imag)
{
    for (long m = maximum(-l, n-l1); m<=minimum(l, n+l1); ++m)
    {
        *l2_sum_real += get_cg(&cgt, l1, l2, l, m, m-n) *
                        (get_shc(l, m).real * get_shc(l1, m-n).real 
                        + get_shc(l, m).imag * get_shc(l1, m-n).imag);
        *l2_sum_imag += get_cg(&cgt, l1, l2, l, m, m-n) *
                        (get_shc(l, m).imag * get_shc(l1, m-n).real
                        - get_shc(l, m).real * get_shc(l1, m-n).imag);
    }
}

void bisp_grad()
{
    nnz_ind = 0;

    long col_ind;

    double l1_sum_real, l1_sum_imag;
    double l2_sum_real, l2_sum_imag;
    double l_sum_real, l_sum_imag;

    for (long l1 = 0; l1<=bandlimit; ++l1)
    {
        for (long l2 = 0; l2<=l1; ++l2)
        {
            for (long l = l1-l2; l<=l1+l2 && l<=bandlimit; ++l)
            {
                if (l1==l2 && l==l1)  // if l1=l2=l
                {
                    // Calculate derivative of b_{l1,l2,l} w.r.t real(f_{l,n}) and imag(f_{l,n})
                    for (long n = -l; n<=l; ++n)
                    {
                        col_ind = 2*(l*(l+1) + n) + 1;
                        
                        l1_sum_real = 0; 
                        l1_sum_imag = 0;
                        l2_sum_real = 0; 
                        l2_sum_imag = 0;
                        l_sum_real = 0;
                        l_sum_imag = 0;

                        calc_l1_sum(l1, l2, l, n, &l1_sum_real, &l1_sum_imag);
                        calc_l2_sum(l1, l2, l, n, &l2_sum_real, &l2_sum_imag);
                        calc_l_sum(l1, l2, l, n, &l_sum_real, &l_sum_imag);

                         // Save the derivative of real(b_{l1,l2,l}) w.r.t real(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l_sum_real + l1_sum_real + l2_sum_real;

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t real(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l_sum_imag + l1_sum_imag + l2_sum_imag;

                        // Save the derivative of real(b_{l1,l2,l}) w.r.t imag(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = ++col_ind;
                        grad_vals[nnz_ind++] = -(l_sum_imag - l1_sum_imag - l2_sum_imag);

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t imag(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l_sum_real - l1_sum_real - l2_sum_real;
                        
                    }
                }
                else if (l1==l2)  // if l1=l2 and l!=l1
                {
                    // Calculate derivative of b_{l1,l2,l} w.r.t real(f_{l,n}) and imag(f_{l,n})
                    for (long n = -l; n<=l; ++n)
                    {
                        col_ind = 2*(l*(l+1) + n) + 1;
                        
                        l_sum_real = 0;
                        l_sum_imag = 0;

                        calc_l_sum(l1, l2, l, n, &l_sum_real, &l_sum_imag);

                         // Save the derivative of real(b_{l1,l2,l}) w.r.t real(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l_sum_real;

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t real(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l_sum_imag;

                        // Save the derivative of real(b_{l1,l2,l}) w.r.t imag(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = ++col_ind;
                        grad_vals[nnz_ind++] = -l_sum_imag;

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t imag(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l_sum_real;
                        
                    }

                    // Calculate derivative of b_{l1,l2,l} w.r.t real(f_{l1,n}) and imag(f_{l1,n})
                    for (long n = -l1; n<=l1; ++n)
                    {
                        col_ind = 2*(l1*(l1+1) + n) + 1;
                        
                        l1_sum_real = 0; 
                        l1_sum_imag = 0;
                        l2_sum_real = 0; 
                        l2_sum_imag = 0;

                        calc_l1_sum(l1, l2, l, n, &l1_sum_real, &l1_sum_imag);
                        calc_l2_sum(l1, l2, l, n, &l2_sum_real, &l2_sum_imag);

                         // Save the derivative of real(b_{l1,l2,l}) w.r.t real(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l1_sum_real + l2_sum_real;

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t real(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l1_sum_imag + l2_sum_imag;

                        // Save the derivative of real(b_{l1,l2,l}) w.r.t imag(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = ++col_ind;
                        grad_vals[nnz_ind++] = l1_sum_imag + l2_sum_imag;

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t imag(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = - l1_sum_real - l2_sum_real;
                        
                    }
                }
                else if (l==l1)  // if l1=l and l!=l2
                {
                    // Calculate derivative of b_{l1,l2,l} w.r.t real(f_{l2,n}) and imag(f_{l2,n})
                    for (long n = -l2; n<=l2; ++n)
                    {
                        col_ind = 2*(l2*(l2+1) + n) + 1;

                        l2_sum_real = 0; 
                        l2_sum_imag = 0;

                        calc_l2_sum(l1, l2, l, n, &l2_sum_real, &l2_sum_imag);

                         // Save the derivative of real(b_{l1,l2,l}) w.r.t real(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l2_sum_real;

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t real(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l2_sum_imag;

                        // Save the derivative of real(b_{l1,l2,l}) w.r.t imag(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = ++col_ind;
                        grad_vals[nnz_ind++] = l2_sum_imag;

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t imag(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = -l2_sum_real;
                        
                    }

                    // Calculate derivative of b_{l1,l2,l} w.r.t real(f_{l,n}) and imag(f_{l,n})
                    for (long n = -l; n<=l; ++n)
                    {
                        col_ind = 2*(l*(l+1) + n) + 1;
                        
                        l1_sum_real = 0; 
                        l1_sum_imag = 0;
                        l_sum_real = 0;
                        l_sum_imag = 0;

                        calc_l1_sum(l1, l2, l, n, &l1_sum_real, &l1_sum_imag);
                        calc_l_sum(l1, l2, l, n, &l_sum_real, &l_sum_imag);

                         // Save the derivative of real(b_{l1,l2,l}) w.r.t real(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l_sum_real + l1_sum_real;

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t real(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l_sum_imag + l1_sum_imag;

                        // Save the derivative of real(b_{l1,l2,l}) w.r.t imag(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = ++col_ind;
                        grad_vals[nnz_ind++] = -(l_sum_imag - l1_sum_imag);

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t imag(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l_sum_real - l1_sum_real;
                        
                    }
                }
                else if(l==l2)  // if l2=l and l!=l1
                {
                    // Calculate derivative of b_{l1,l2,l} w.r.t real(f_{l1,n}) and imag(f_{l1,n})
                    for (long n = -l1; n<=l1; ++n)
                    {
                        col_ind = 2*(l1*(l1+1) + n) + 1;
                        
                        l1_sum_real = 0; 
                        l1_sum_imag = 0;

                        calc_l1_sum(l1, l2, l, n, &l1_sum_real, &l1_sum_imag);

                         // Save the derivative of real(b_{l1,l2,l}) w.r.t real(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l1_sum_real;

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t real(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l1_sum_imag;

                        // Save the derivative of real(b_{l1,l2,l}) w.r.t imag(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = ++col_ind;
                        grad_vals[nnz_ind++] = l1_sum_imag;

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t imag(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = - l1_sum_real;
                        
                    }

                    // Calculate derivative of b_{l1,l2,l} w.r.t real(f_{l,n}) and imag(f_{l,n})
                    for (long n = -l; n<=l; ++n)
                    {
                        col_ind = 2*(l*(l+1) + n) + 1;
                        
                        l2_sum_real = 0; 
                        l2_sum_imag = 0;
                        l_sum_real = 0;
                        l_sum_imag = 0;

                        calc_l2_sum(l1, l2, l, n, &l2_sum_real, &l2_sum_imag);
                        calc_l_sum(l1, l2, l, n, &l_sum_real, &l_sum_imag);

                         // Save the derivative of real(b_{l1,l2,l}) w.r.t real(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l_sum_real + l2_sum_real;

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t real(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l_sum_imag + l2_sum_imag;

                        // Save the derivative of real(b_{l1,l2,l}) w.r.t imag(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = ++col_ind;
                        grad_vals[nnz_ind++] = -(l_sum_imag - l2_sum_imag);

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t imag(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l_sum_real - l2_sum_real;
                        
                    }

                }
                else   // if l1!=l2, l!=l1 and l!=l2
                {
                    // Calculate derivative of b_{l1,l2,l} w.r.t real(f_{l1,n}) and imag(f_{l1,n})
                    for (long n = -l1; n<=l1; ++n)
                    {
                        col_ind = 2*(l1*(l1+1) + n) + 1;
                        
                        l1_sum_real = 0; 
                        l1_sum_imag = 0;

                        calc_l1_sum(l1, l2, l, n, &l1_sum_real, &l1_sum_imag);

                         // Save the derivative of real(b_{l1,l2,l}) w.r.t real(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l1_sum_real;

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t real(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l1_sum_imag;

                        // Save the derivative of real(b_{l1,l2,l}) w.r.t imag(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = ++col_ind;
                        grad_vals[nnz_ind++] = l1_sum_imag;

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t imag(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = -l1_sum_real;
                        
                    }

                    // Calculate derivative of b_{l1,l2,l} w.r.t real(f_{l2,n}) and imag(f_{l2,n})
                    for (long n = -l2; n<=l2; ++n)
                    {
                        col_ind = 2*(l2*(l2+1) + n) + 1;
                        
                        l2_sum_real = 0; 
                        l2_sum_imag = 0;

                        calc_l2_sum(l1, l2, l, n, &l2_sum_real, &l2_sum_imag);

                         // Save the derivative of real(b_{l1,l2,l}) w.r.t real(f_{l2,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] =  l2_sum_real;

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t real(f_{l2,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l2_sum_imag;

                        // Save the derivative of real(b_{l1,l2,l}) w.r.t imag(f_{l2,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = ++col_ind;
                        grad_vals[nnz_ind++] = l2_sum_imag;

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t imag(f_{l2,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = -l2_sum_real;
                        
                    }

                    // Calculate derivative of b_{l1,l2,l} w.r.t real(f_{l,n}) and imag(f_{l,n})
                    for (long n = -l; n<=l; ++n)
                    {
                        col_ind = 2*(l*(l+1) + n) + 1;
                        
                        l_sum_real = 0;
                        l_sum_imag = 0;

                        calc_l_sum(l1, l2, l, n, &l_sum_real, &l_sum_imag);

                         // Save the derivative of real(b_{l1,l2,l}) w.r.t real(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l_sum_real;

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t real(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l_sum_imag;

                        // Save the derivative of real(b_{l1,l2,l}) w.r.t imag(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, REAL_PART) + 1;
                        grad_j[nnz_ind] = ++col_ind;
                        grad_vals[nnz_ind++] = -l_sum_imag;

                        // Save the derivative of imag(b_{l1,l2,l}) w.r.t imag(f_{l,n})
                        grad_i[nnz_ind] = bisp_lookup(l1, l2, l, IMAG_PART) + 1;
                        grad_j[nnz_ind] = col_ind;
                        grad_vals[nnz_ind++] = l_sum_real;
                        
                    }

                }

            }
        }
    }
}


/**
 * The MEX Function
  */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Obtain input
    shc = mxGetComplexDoubles(prhs[0]);
    bandlimit = mxGetScalar(prhs[1]);

    // Handle first run
    if (first_run==true)
    {
        build_bisp_lookup_table();
        
        // mxArray *inputMxArray[1];
        // inputMxArray[0] = mxCreateDoubleScalar(bandlimit);
        // mexCallMATLAB(0, NULL, 1, inputMxArray, "loadCGTable");
        // 
        // CGs = mexGetVariablePtr("global", "CGs");
        // if (CGs == NULL)
        // {
        //     mexErrMsgIdAndTxt("Bispectrum:CGsMissing", "Clebsch-Gordan coefficients need to be preloaded.");
        // }
        // 
        create_CGTable(&cgt, &prhs[2], bandlimit);

        previous_bandlimit = bandlimit;
        first_run = false;
    }
    
    // Handle change of bandlimit
    if (bandlimit!=previous_bandlimit)
    {
        // mxArray *inputMxArray[1];
        // inputMxArray[0] = mxCreateDoubleScalar(bandlimit);
        // mexCallMATLAB(0, NULL, 1, inputMxArray, "loadCGTable");
        // 
        // CGs = mexGetVariablePtr("global", "CGs");
        // if (CGs == NULL)
        // {
        //     mexErrMsgIdAndTxt("Bispectrum:CGsMissing", "Clebsch-Gordan coefficients need to be preloaded.");
        // }

        destory_bisp_lookup_table();
        destroy_CGTable(&cgt, previous_bandlimit);

        previous_bandlimit = bandlimit;

        build_bisp_lookup_table();
        create_CGTable(&cgt, &prhs[2], bandlimit);
    }

    // Create output
    plhs[0] = mxCreateDoubleMatrix(grad_rows_no, 1, mxREAL);
    b = mxGetDoubles(plhs[0]);
    if (nlhs>1)
    {
        plhs[1] = mxCreateDoubleMatrix(grad_nnz, 1, mxREAL);
        grad_i = mxGetDoubles(plhs[1]);

        plhs[2] = mxCreateDoubleMatrix(grad_nnz, 1, mxREAL);
        grad_j = mxGetDoubles(plhs[2]);

        plhs[3] = mxCreateDoubleMatrix(grad_nnz, 1, mxREAL);
        grad_vals = mxGetDoubles(plhs[3]);

        plhs[4] = mxCreateDoubleScalar((double) grad_nnz);
        plhs[5] = mxCreateDoubleScalar((double) grad_rows_no);
        plhs[6] = mxCreateDoubleScalar((double) grad_cols_no);
    }

    // Compute output
    bisp();

    if (nlhs>1)
    {
        bisp_grad();
    }
}
