#include "lpm.h"

void solverPARDISO()
{
    MKL_INT n, idum, maxfct, mnum, mtype, phase, error, error1, msglvl, nrhs, iter;
    MKL_INT iparm[64];
    double ddum;

    void* pt[64]; /* Internal solver memory pointer */

    for (int i = 0; i < 64; i++)
        iparm[i] = 0;
    iparm[0] = 1;  /* No solver default */
    iparm[1] = 3;  /* The parallel (OpenMP) version of the nested dissection algorithm is used */
    iparm[3] = 0;  /* No iterative-direct algorithm */
    iparm[4] = 0;  /* No user fill-in reducing permutation */
    iparm[5] = 0;  /* Write solution into x */
    iparm[6] = 0;  /* Not in use */
    iparm[7] = 0;  /* Max numbers of iterative refinement steps */
    iparm[8] = 0;  /* Not in use */
    iparm[9] = 13; /* Perturb the pivot elements with 1E-13 */
    iparm[10] = 1; /* Use nonsymmetric permutation and scaling MPS */
    //iparm[11] = 0;        /* Not in use */
    iparm[12] = 0; /* Maximum weighted matching algorithm is switched-off (default for
                          symmetric). Try iparm[12] = 1 in case of inappropriate accuracy */
                          //iparm[13] = 0;        /* Output: Number of perturbed pivots */
                          //iparm[14] = 0;        /* Not in use */
                          //iparm[15] = 0;        /* Not in use */
                          //iparm[16] = 0;        /* Not in use */
                          //iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
                          //iparm[18] = -1;       /* Output: Mflops for LU factorization */
                          //iparm[19] = 0;        /* Output: Numbers of CG Iterations */

    n = dim * nparticle; /* Data number */
    maxfct = 1;          /* Maximum number of numerical factorizations */
    mnum = 1;            /* Which factorization to use */
    msglvl = 0;          /* 0, no print statistical info; 1, print statistical info */
    error = 0;           /* Initialize error flag */
    mtype = 2;           /* Real symmetric positive definite, -2: real+symmetric+indefinite */
    nrhs = 1;            /* Number of right hand sides */
    iter = 1;            /* Iteration number */

    for (int i = 0; i < 64; i++)
        pt[i] = 0; /* Initiliaze the internal solver memory pointer */

    /* Reordering and Symbolic Factorization. This step also allocates all memory that is  */
    /* necessary for the factorization */
    phase = 11;
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, K_global, IK, JK, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if (error != 0)
    {
        printf("\nERROR during symbolic factorization: %d", error);
        exit(1);
    }
    //printf("PARDISO: Size of factors(MB): %f", MAX(iparm[14], iparm[15] + iparm[16]) / 1000.0);
    printf("PARDISO: Size of factors(MB): %f", iparm[16] / 1000.0);

    /* Numerical factorization */
    phase = 22;
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, K_global, IK, JK, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if (error != 0)
    {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }

    /* Back substitution and iterative refinement */
    phase = 33;
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, K_global, IK, JK, &idum, &nrhs, iparm, &msglvl, residual, disp, &error);
    if (error != 0)
    {
        printf("\nERROR during solution: %d", error);
        exit(3);
    }
    printf("\nSolve completed at iteration: %d\n", iter);

    /* Termination and release of memory */
    phase = -1; /* Release internal memory. */
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, &ddum, IK, JK, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error1);
    if (error1 != 0)
    {
        printf("\nERROR on release stage: %d", error1);
        exit(4);
    }

    /* Components of position */
    for (int i = 0; i < nparticle; i++)
    {
        for (int j = 0; j < dim; j++)
            xyz[i][j] += disp[dim * i + j];
    }
}

#if 0
void solverCG()
{

    MKL_INT i, n, rci_request, itercount, mkl_disable_fast_mm;
    MKL_INT ipar[128];
    double dpar[128], * tmp;
    char tr = 'u';

    /* matrix descriptor */
    //struct matrix_descr descr;
    //double angle1 = 1.0, angle2 = 0.0;
    //descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    //descr.mode = SPARSE_FILL_MODE_LOWER;
    //descr.diag = SPARSE_DIAG_NON_UNIT;

    mkl_disable_fast_mm = 1; /* avoid memory leaks */

    /* initial setting */
    n = dim * nparticle; /* Data number */
    tmp = allocDouble1D(4 * n, 0);

    /* initial guess for the displacement vector */
    //for (i = 0; i < n; i++)
    //    disp[i] = -1.781401523368843e-019;
    memset(disp, 0, n * sizeof(double));

    /* initialize the solver */
    dcg_init(&n, disp, residual, &rci_request, ipar, dpar, tmp);

    if (rci_request != 0)
        goto failure;

    /* modify the initialized solver parameters */
    ipar[4] = dim * nparticle;
    ipar[8] = 1; /* default value is 0, does not perform the residual stopping test; otherwise, perform the test */
    ipar[9] = 0; /* default value is 1, perform user defined stopping test; otherwise, does not perform the test */
    //ipar[10] = 1; /* use the preconditioned version of the CG method */
    dpar[0] = 1e-20; /* specifies the relative tolerance, the default value is 1e-6 */
    dpar[1] = 1e-20; /* specifies the absolute tolerance, the default value is 0.0 */

    /* check the correctness and consistency of the newly set parameters */
    dcg_check(&n, disp, residual, &rci_request, ipar, dpar, tmp);
    if (rci_request != 0)
        goto failure;

    /* compute the solution by RCI (residual)CG solver */
    /* reverse Communications starts here */
rci:
    dcg(&n, disp, residual, &rci_request, ipar, dpar, tmp);
    /* if rci_request=0, then the solution was found according to the requested  */
    /* stopping tests. in this case, this means that it was found after 100      */
    /* iterations. */

    if (rci_request == 0)
        goto getsln;

    /* If rci_request=1, then compute the vector K_global*TMP[0]                  */
    /* and put the result in vector TMP[n]                                       */
    if (rci_request == 1)
    {
        mkl_dcsrsymv(&tr, &n, K_global, IK, JK, tmp, &tmp[n]);
        goto rci;
    }

    /* If rci_request=anything else, then dcg subroutine failed                  */
    /* to compute the solution vector: solution[n]                               */
    goto failure;
    /* Reverse Communication ends here                                           */
    /* Get the current iteration number into itercount                           */
getsln:
    dcg_get(&n, disp, residual, &rci_request, ipar, dpar, tmp, &itercount);
    printf("The system has been solved at the number of iterations: %d", itercount);
    goto success;

failure:
    printf("This example FAILED as the solver has returned the ERROR code %d", rci_request);

success:

    /* Components of position */
    for (int i = 0; i < nparticle; i++)
    {
        for (int j = 0; j < dim; j++)
            xyz[i][j] += disp[dim * i + j];
    }

    /* free memory */
    free(tmp);
    //printDouble(disp, 1, 20);
}
#endif

#if 1
void solverCG()
{
    MKL_INT n, rci_request, itercount, mkl_disable_fast_mm;
    MKL_INT ipar[128];
    double dpar[128], * tmp;

    /* matrix descriptor */
    struct matrix_descr descrA;
    sparse_matrix_t csrA;
    sparse_operation_t transA = SPARSE_OPERATION_NON_TRANSPOSE;
    descrA.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
    descrA.mode = SPARSE_FILL_MODE_UPPER;
    descrA.diag = SPARSE_DIAG_NON_UNIT;
    mkl_disable_fast_mm = 1; /* avoid memory leaks */

    /* initial setting */
    n = dim * nparticle; /* Data number */
    tmp = allocDouble1D(4 * n, 0);
    mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, n, n, IK, IK + 1, JK, K_global);

    /* initial guess for the displacement vector */
    memset(disp, 0, n * sizeof(double));

    /* initialize the solver */
    dcg_init(&n, disp, residual, &rci_request, ipar, dpar, tmp);
    if (rci_request != 0)
        goto failure;

    /* modify the initialized solver parameters */
    ipar[4] = dim * nparticle;
    ipar[8] = 1; /* default value is 0, does not perform the residual stopping test; otherwise, perform the test */
    ipar[9] = 0; /* default value is 1, perform user defined stopping test; otherwise, does not perform the test */
    //ipar[10] = 1; /* use the preconditioned version of the CG method */
    dpar[0] = 1e-8;  /* specifies the relative tolerance, the default value is 1e-6 */
    dpar[1] = 1e-12; /* specifies the absolute tolerance, the default value is 0.0 */

    /* check the correctness and consistency of the newly set parameters */
    dcg_check(&n, disp, residual, &rci_request, ipar, dpar, tmp);
    if (rci_request != 0)
        goto failure;

    /* compute the solution by RCI (residual)CG solver */
    /* reverse Communications starts here */
rci:
    dcg(&n, disp, residual, &rci_request, ipar, dpar, tmp);
    /* if rci_request=0, then the solution was found according to the requested  */
    /* stopping tests. in this case, this means that it was found after 100      */
    /* iterations. */
    if (rci_request == 0)
        goto getsln;

    /* If rci_request=1, then compute the vector K_global*TMP[0]                  */
    /* and put the result in vector TMP[n]                                       */
    if (rci_request == 1)
    {
        mkl_sparse_d_mv(transA, 1.0, csrA, descrA, tmp, 0.0, &tmp[n]);
        goto rci;
    }

    /* If rci_request=anything else, then dcg subroutine failed                  */
    /* to compute the solution vector: solution[n]                               */
    goto failure;
    /* Reverse Communication ends here                                           */
    /* Get the current iteration number into itercount                           */
getsln:
    dcg_get(&n, disp, residual, &rci_request, ipar, dpar, tmp, &itercount);
    printf("The system has been solved after %d iterations\n", itercount);
    goto success;

failure:
    printf("The computation FAILED as the solver has returned the ERROR code %d\n", rci_request);

success:

    /* update the position */
    for (int i = 0; i < nparticle; i++)
    {
        for (int j = 0; j < dim; j++)
            xyz[i][j] += disp[dim * i + j];
    }
    /* free memory */
    free(tmp);
}
#endif
