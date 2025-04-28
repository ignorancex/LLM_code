#include "clebsch_gordan_mex.h"


void create_CGTable(CGTable * const cgt, const mxArray ** CGs, const size_t bl)
{
    mxArray *temp[3];

    (*cgt) = malloc((bl+1)*sizeof(double ****));

    for (long l1 = 0; l1<=bl; ++l1)
    {
        temp[0] = mxGetCell(*CGs, l1);
        (*cgt)[l1] = malloc((l1+1)*sizeof(double ***));

        for (long l2 = 0; l2<=l1; ++l2)
        {
            temp[1] = mxGetCell(temp[0], l2);
            (*cgt)[l1][l2] = malloc(((l1+l2>=bl ? bl : l1+l2) - (l1-l2) + 1)*sizeof(double **));

            for (long l = l1-l2; l<=l1+l2 && l<=bl; ++l)
            {
                temp[2] = mxGetCell(temp[1], l-(l1-l2));
                (*cgt)[l1][l2][l-(l1-l2)] = malloc((2*l+1)*sizeof(double *));

                for (long m = -l; m<=l; ++m)
                {
                    (*cgt)[l1][l2][l-(l1-l2)][m+l] = mxGetDoubles(mxGetCell(temp[2], m+l));
                }
            }
        }
    }
}

void destroy_CGTable(CGTable * const cgt, const size_t bl)
{
    for (long l1 = 0; l1<=bl; ++l1)
    {
        for (long l2 = 0; l2<=l1; ++l2)
        {
            for (long l = l1-l2; l<=l1+l2 && l<=bl; ++l)
            {
                free((*cgt)[l1][l2][l-(l1-l2)]);
            }
            free((*cgt)[l1][l2]);
        }
        free((*cgt)[l1]);
    }
    free(*cgt);
}

double get_cg(CGTable * const cgt, const long l1, const long l2, const long l, const long m, const long m1)
{
    return (*cgt)[l1][l2][l-(l1>=l2 ? l1-l2 : l2-l1)][m+l][m1-((m-l2<=-l1) ? -l1 : m-l2)];
}