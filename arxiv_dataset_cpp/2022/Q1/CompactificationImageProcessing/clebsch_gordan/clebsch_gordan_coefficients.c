/**
 * Functions relating to the computation of Clebsch-Gordan coefficients.
 * 
 * The method implemented here is suggested in [1].
 * 
 * REFERENCE:
 *  [1] Straub, W. O. (n.d.). Efficient Computation of Clebsch-Gordan Coefficients. Retrieved October 28, 2019, from http://vixra.org/abs/1403.0263
 */

#include <math.h>
#include "clebsch_gordan_coefficients.h"


/* 
Clebsch-Gordan utility functions. See [1] for details.
*/
long cg_lower_bound(const long l1, const long l2, const long m)
{
    /* Returns the lowest m-index of Clebsch-Gordan coefficients of order (l1, l2, l, m)  */

    return (m-l2<=-l1) ? -l1 : m-l2;
}


long cg_upper_bound(const long l1, const long l2, const long m)
{
    /* Returns the highest m-index of Clebsch-Gordan coefficients of order (l1, l2, l, m)  */

    return (m+l2<=l1) ? m+l2 : l1;
}


double backward_substitution_central_diagonal(const long l1, const long l2, const long l, const long m1, const long m)
{
    return l1*(l1+1) + l2*(l2+1) + 2*m1*(m-m1) - l*(l+1);
}

double backward_substitution_secondary_diagonal(const long l1, const long l2, const long m1, const long m)
{
    return sqrt(l1*(l1+1) - m1*(m1+1)) * sqrt(l2*(l2+1) - (m-m1)*(m-m1-1));
}


void cg_vector(const long l1, const long l2, const long l, const long m, double * restrict cg)
{
    /*  
    Calculate a vector containing <l1 m1 l2 (m-m1) | l m> for m1 satisfying
            max{-l1,m-l2}<=m1<=min{l1,m+l2}. 
    They are place in cg in ascending order of m1. In particular, I assume cg is an array of size (upper_bound - lower_bound + 1). 
    */

    long lower_bound = cg_lower_bound(l1, l2, m);
    long upper_bound = cg_upper_bound(l1, l2, m);

    cg[upper_bound - lower_bound] = 1.0/(upper_bound - lower_bound + 1);

    if (upper_bound > lower_bound)
    {   
        double norm = cg[upper_bound - lower_bound]*cg[upper_bound - lower_bound];

        // Using the recursion to calculate the Clebsch-Gordan coefficients
        cg[upper_bound - lower_bound - 1] 
            = -cg[upper_bound - lower_bound]*backward_substitution_central_diagonal(l1, l2, l, upper_bound, m)
                /backward_substitution_secondary_diagonal(l1, l2, upper_bound-1, m);
        norm += cg[upper_bound - lower_bound - 1]*cg[upper_bound - lower_bound - 1];

        for (long i = upper_bound - lower_bound - 2; i>=0; --i)
        {
            cg[i] = - (cg[i+1]*backward_substitution_central_diagonal(l1, l2, l, lower_bound+i+1, m)
                     + cg[i+2]*backward_substitution_secondary_diagonal(l1, l2, lower_bound+i+1, m))
                              /backward_substitution_secondary_diagonal(l1, l2, lower_bound+i, m);
            norm += cg[i]*cg[i];
        }
        
        // Normalize the solution so that the norm of the Clebsch-Gordan coefficients is 1
        norm = sqrt(norm);
        for (long i = 0; i<upper_bound-lower_bound+1; ++i)
        {
            cg[i] /= norm;
        }
    }
}


double get_cg(const cg_table *table, const long l1, const long l2, const long l, const long m, const long m1)
{
    /* Documentation of memory structure is in allocate_cg_table */
    
    return table->table[l1][l2][l-(l1>=l2 ? l1-l2 : l2-l1)][m+l][m1-cg_lower_bound(l1, l2, m)];
}
