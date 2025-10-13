#include "schAngFac.hpp"

void SchurAngularFactor::compute_SkewSymmMat(REAL_TYPE *MatA, INTE_TYPE lda)
{
    for (INTE_TYPE col_ind = 0; col_ind < d; col_ind++)
        memset(MatA + col_ind * lda, 0, d * sizeof(REAL_TYPE)); // Initialize S with 0
    for (INTE_TYPE ang_ind = 0; ang_ind < nzsize; ang_ind++)
        my_dger(d, d, a[ang_ind], svec._col_access[2 * ang_ind + 1], svec._col_access[2 * ang_ind], MatA, lda);

    for (INTE_TYPE col_ind = 0; col_ind < d; col_ind++)
    {
        MatA[col_ind + col_ind * lda] = 0.0;
        for (INTE_TYPE row_ind = col_ind + 1; row_ind < d; row_ind++)
        {
            MatA[row_ind + col_ind * lda] -= MatA[col_ind + row_ind * lda];
            MatA[col_ind + row_ind * lda] = -MatA[row_ind + col_ind * lda];
        }
    }
}

void SchurAngularFactor::compute_SpecOrthMat(REAL_TYPE *MatA, INTE_TYPE lda, REAL_TYPE *work)
{
    // The work is done on the lower triangular part and the upper triangular part is updated at the end.
    REAL_TYPE cval, sval;
    INTE_TYPE left_over_colum_num;

    // printf("Entering the recovery of the special orthogonal matrix from the Schur decomposition.\n");
    // printf("Angles:\t");
    // for (INTE_TYPE ang_ind = 0; ang_ind < nzsize; ang_ind++)
    //     printf("%2.2f\t", a[ang_ind]);

    for (INTE_TYPE col_ind = 0; col_ind < d; col_ind++) // Initialize S with 0.
        memset(MatA + col_ind * lda, 0, d * sizeof(REAL_TYPE));

    left_over_colum_num = d - 2 * nzsize;
    if (left_over_colum_num)
        my_dsyrk(d, left_over_colum_num, 1.0, svec._col_access[2 * nzsize], d, 0.0, work, d); // Initialize workspace and compute work = X*X'
    else
        memset(work, 0, sizeof(REAL_TYPE) * d * d); // Initialize workspace manually with 0.

    for (INTE_TYPE ang_ind = 0; ang_ind < nzsize; ang_ind++)
    {
        cval = cos(a[ang_ind]);
        sval = sin(a[ang_ind]);
        my_dger(d, d, sval, svec._col_access[2 * ang_ind + 1], svec._col_access[2 * ang_ind], MatA, lda); // Skew symmetric part B in A.
        my_dsyr(d, cval, svec._col_access[2 * ang_ind], work);                                            // Symmetric update in the workspace.
        my_dsyr(d, cval, svec._col_access[2 * ang_ind + 1], work);                                        // Symmetric update in the workspace.
    }

    // printf("The symmetric part of the Q = R e^D R^T:\n");
    // for (INTE_TYPE row_ind = 0; row_ind < d; row_ind++, printf("\n"))
    //     for (INTE_TYPE col_ind = 0; col_ind < row_ind; col_ind++)
    //         printf("%2.2f\t", work[row_ind + col_ind * d]);

    for (INTE_TYPE col_ind = 0; col_ind < d; col_ind++) // Compute lower triangular part of B - B'
        for (INTE_TYPE row_ind = col_ind; row_ind < d; row_ind++)
            MatA[row_ind + col_ind * lda] -= MatA[col_ind + row_ind * lda];

    // printf("The skew symmetric part of the Q = R e^D R^T:\n");
    // for (INTE_TYPE row_ind = 0; row_ind < d; row_ind++, printf("\n"))
    //     for (INTE_TYPE col_ind = 0; col_ind < row_ind; col_ind++)
    //         printf("%2.2f\t", MatA[row_ind + col_ind * d]);

    for (INTE_TYPE col_ind = 0; col_ind < d; col_ind++)
    {
        MatA[col_ind + col_ind * lda] = work[col_ind + col_ind * lda];
        for (INTE_TYPE row_ind = (col_ind + 1); row_ind < d; row_ind++)
        {
            MatA[col_ind + row_ind * lda] = work[row_ind + col_ind * d] - MatA[row_ind + col_ind * lda]; // Compute the upper triangular part.
            MatA[row_ind + col_ind * lda] += work[row_ind + col_ind * d];                                // Compute the lower triangular part.
        }
    }

    // printf("The recorved Q = R e^D R^T:\n");
    // for (INTE_TYPE row_ind = 0; row_ind < d; row_ind++, printf("\n"))
    //     for (INTE_TYPE col_ind = 0; col_ind < d; col_ind++)
    //         printf("%2.2f\t", MatA[row_ind + col_ind * d]);
    //     for (INTE_TYPE col_ind = 0; col_ind < d; col_ind++)
    //         for (INTE_TYPE row_ind = col_ind; row_ind < d; row_ind++)
    //         {
    //             MatA[row_ind + col_ind * lda] -= MatA[col_ind + row_ind * lda];
    //             MatA[col_ind + row_ind * lda] = -MatA[row_ind + col_ind * lda];
    //         }

    //     for (INTE_TYPE mat_ind = 0; mat_ind < d * d; mat_ind++)
    //         MatA[mat_ind] += work[mat_ind];
}

void SchurAngularFactor::SchurAngular_SpecOrth()
{
    auto wS = w;
    auto wr = wS + d * d;
    auto wi = wr + d;

    svec.fast_col_access();

    my_dgees('V', 'S', _dgees_select_nonreal, d, wS, d, &nzsize, wr, wi, v, d);
    nzsize = nzsize / 2;
    REAL_TYPE *top_left_ptr = wS;
    for (INTE_TYPE i = 0; i < nzsize; i++, top_left_ptr += (2 * d + 2))
        a[i] = atan2(*(top_left_ptr + 1), *top_left_ptr);
}