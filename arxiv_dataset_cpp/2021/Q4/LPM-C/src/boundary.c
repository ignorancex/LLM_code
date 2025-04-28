#include <stdio.h>
#include <math.h>
#include <memory.h>
#include <time.h>

#include "lpm.h"

/* For implicit integration, the incremental load should be small enough, otherwise the
   solution would not converge */

/* enforcing the displacement boundary conditions */
void setDispBC(int nboundDisp, struct dispBCPara *dBP)
{

    int sum_dispDoF = 0;
    for (int i = 0; i < nparticle; i++)
    {
        for (int n = 0; n < nboundDisp; n++)
        {
            if (type[i] == dBP[n].type)
            {
                sum_dispDoF++;
                if (dBP[n].flag == 'x')
                {
                    xyz[i][0] += dBP[n].step;
                    dispBC_index[dim * i] = 0;
                }
                if (dBP[n].flag == 'y')
                {
                    xyz[i][1] += dBP[n].step;
                    dispBC_index[dim * i + 1] = 0;
                }
                if (dBP[n].flag == 'z')
                {
                    /* this would not happen when dimension is 2 */
                    xyz[i][2] += dBP[n].step;
                    dispBC_index[dim * i + 2] = 0;
                }
            }
        }
    }

    /* allocate memory for reaction force vector */
    reaction_force = allocDouble1D(countNEqual(dispBC_index, nparticle * dim, 1), 0.);
}

/* enforcing the applied force boundary conditions */
void setForceBC(int nboundForce, struct forceBCPara *fBP)
{

    for (int n = 0; n < nboundForce; n++)
    {
        int sum_forceBC = 0;
        for (int i = 0; i < nparticle; i++)
        {
            if (type[i] == fBP[n].type)
                sum_forceBC++; /* count the number of force BC particles */
        }
        for (int i = 0; i < nparticle; i++)
        {
            if (type[i] == fBP[n].type)
            {
                Pex[dim * i] += fBP[n].step1 / sum_forceBC;     /* x component */
                Pex[dim * i + 1] += fBP[n].step2 / sum_forceBC; /* y component */
                if (dim == 3)
                    Pex[dim * i + 2] += fBP[n].step3 / sum_forceBC; /* z component */
            }
        }
    }
}

void setDispBC_stiffnessUpdate2D()
{
    /* Goal: update the stiffness matrix due to the apply of displacement boundary condition */
    /* change the DoF of disp BC into zero, insert the norm of the diagonal of the original
    stiffness matrix into the diagonal positions, 2D case */

    double *diag = allocDouble1D(nparticle * dim, 0.); /* diagonal vector of stiffness matrix */

    /* extract the diagonal vector of stiffness matrix */
    for (int i = 0; i < nparticle; i++)
    {
        diag[i * dim] = K_global[K_pointer[i][1]];
        diag[i * dim + 1] = K_global[K_pointer[i][1] + dim * (K_pointer[i][0])];
    }

    /* compute the norm of the diagonal */
    double norm_diag = cblas_dnrm2(dim * nparticle, diag, 1); /* Euclidean norm (L2 norm) */
    // double norm_diag = 1e4;

    /* update the stiffness matrix */
    for (int i = 0; i < nparticle; i++)
    {
        if (dispBC_index[dim * i] == 0 || fix_index[dim * i] == 0) /* x-disp BC */
        {
            for (int j = 0; j < nb_conn[i]; j++)
            {
                if (conn[i][j] > i)
                    continue;
                if (conn[i][j] == i)
                {
                    for (int kk = K_pointer[i][1]; kk <= K_pointer[i][1] + dim * K_pointer[i][0] - 1; kk++)
                        K_global[kk] = 0.0;
                    K_global[K_pointer[i][1]] = norm_diag;
                }
                else
                {
                    int num2 = 0;
                    for (int k = 0; k < nb_conn[conn[i][j]]; k++)
                    {
                        if (conn[conn[i][j]][k] <= conn[i][j])
                            continue;
                        num2++;
                        if (conn[conn[i][j]][k] == i)
                        {
                            K_global[K_pointer[conn[i][j]][1] + dim * num2] = 0.0;
                            K_global[K_pointer[conn[i][j]][1] + dim * K_pointer[conn[i][j]][0] + dim * num2 - 1] = 0.0;
                        }
                    }
                }
            }
            residual[dim * i] = 0.0;
        }

        if (dispBC_index[dim * i + 1] == 0 || fix_index[dim * i + 1] == 0) /* y-disp BC */
        {
            for (int j = 0; j < nb_conn[i]; j++)
            {
                if (conn[i][j] > i)
                    continue;
                if (conn[i][j] == i)
                {
                    K_global[K_pointer[i][1] + 1] = 0.0;
                    for (int kk = K_pointer[i][1] + dim * K_pointer[i][0]; kk <= K_pointer[i + 1][1] - 1; kk++)
                        K_global[kk] = 0.0;
                    K_global[K_pointer[i][1] + dim * K_pointer[i][0]] = norm_diag;
                }
                else
                {
                    int num2 = 0;
                    for (int k = 0; k < nb_conn[conn[i][j]]; k++)
                    {
                        if (conn[conn[i][j]][k] <= conn[i][j])
                            continue;
                        num2++;
                        if (conn[conn[i][j]][k] == i)
                        {
                            K_global[K_pointer[conn[i][j]][1] + dim * num2 + 1] = 0.0;
                            K_global[K_pointer[conn[i][j]][1] + dim * K_pointer[conn[i][j]][0] + dim * num2] = 0.0;
                        }
                    }
                }
            }
            residual[dim * i + 1] = 0.0;
        }
    }
}

void setDispBC_stiffnessUpdate3D()
{
    /* Goal: update the stiffness matrix due to the apply of displacement boundary condition */
    /* change the DoF of disp BC into zero, insert the norm of the diagonal of the original
    stiffness matrix into the diagonal positions */

    double *diag = allocDouble1D(nparticle * dim, 0.); /* diagonal vector of stiffness matrix */

    /* extract the diagonal vector of stiffness matrix */
    for (int i = 0; i < nparticle; i++)
    {
        diag[i * dim] = K_global[K_pointer[i][1]];
        diag[i * dim + 1] = K_global[K_pointer[i][1] + dim * (K_pointer[i][0])];
        diag[i * dim + 2] = K_global[K_pointer[i][1] + 2 * dim * (K_pointer[i][0]) - 1];
    }

    /* compute the norm of the diagonal */
    double norm_diag = cblas_dnrm2(dim * nparticle, diag, 1); /* Euclidean norm (L2 norm) */

    //printf("norm: %f\n", norm_diag);
    /* update the stiffness matrix */
    for (int i = 0; i < nparticle; i++)
    {
        if (dispBC_index[dim * i] == 0 || fix_index[dim * i] == 0) /* x-disp BC */
        {
            for (int j = 0; j < nb_conn[i]; j++)
            {
                if (conn[i][j] > i)
                    continue;
                if (conn[i][j] == i)
                {
                    for (int kk = K_pointer[i][1] + 1; kk <= K_pointer[i][1] + dim * K_pointer[i][0] - 1; kk++)
                        K_global[kk] = 0.0;
                    K_global[K_pointer[i][1]] = norm_diag;
                }
                else
                {
                    int num2 = 0;
                    for (int k = 0; k < nb_conn[conn[i][j]]; k++)
                    {
                        if (conn[conn[i][j]][k] <= conn[i][j])
                            continue;
                        num2++;
                        if (conn[conn[i][j]][k] == i)
                        {
                            K_global[K_pointer[conn[i][j]][1] + dim * num2] = 0.0;
                            K_global[K_pointer[conn[i][j]][1] + dim * K_pointer[conn[i][j]][0] + dim * num2 - 1] = 0.0;
                            K_global[K_pointer[conn[i][j]][1] + 2 * dim * K_pointer[conn[i][j]][0] + dim * num2 - 3] = 0.0;
                        }
                    }
                }
            }
            residual[dim * i] = 0.0;
        }

        if (dispBC_index[dim * i + 1] == 0 || fix_index[dim * i + 1] == 0) /* y-disp BC */
        {
            for (int j = 0; j < nb_conn[i]; j++)
            {
                if (conn[i][j] > i)
                    continue;
                if (conn[i][j] == i)
                {
                    K_global[K_pointer[i][1] + 1] = 0.0;
                    for (int kk = K_pointer[i][1] + dim * K_pointer[i][0] + 1; kk <= K_pointer[i][1] + 2 * dim * K_pointer[i][0] - 2; kk++)
                        K_global[kk] = 0.0;
                    K_global[K_pointer[i][1] + dim * K_pointer[i][0]] = norm_diag;
                }
                else
                {
                    int num2 = 0;
                    for (int k = 0; k < nb_conn[conn[i][j]]; k++)
                    {
                        if (conn[conn[i][j]][k] <= conn[i][j])
                            continue;
                        num2++;
                        if (conn[conn[i][j]][k] == i)
                        {
                            K_global[K_pointer[conn[i][j]][1] + dim * num2 + 1] = 0.0;
                            K_global[K_pointer[conn[i][j]][1] + dim * K_pointer[conn[i][j]][0] + dim * num2] = 0.0;
                            K_global[K_pointer[conn[i][j]][1] + 2 * dim * K_pointer[conn[i][j]][0] + dim * num2 - 2] = 0.0;
                        }
                    }
                }
            }
            residual[dim * i + 1] = 0.0;
        }

        if (dispBC_index[dim * i + 2] == 0 || fix_index[dim * i + 2] == 0) /* z-disp BC */
        {
            for (int j = 0; j < nb_conn[i]; j++)
            {
                if (conn[i][j] > i)
                    continue;
                if (conn[i][j] == i)
                {
                    K_global[K_pointer[i][1] + 2] = 0.0;
                    K_global[K_pointer[i][1] + dim * K_pointer[i][0] + 1] = 0.0;
                    for (int kk = K_pointer[i][1] + 2 * dim * K_pointer[i][0] - 1; kk <= K_pointer[i + 1][1] - 1; kk++)
                        K_global[kk] = 0.0;
                    K_global[K_pointer[i][1] + 2 * dim * K_pointer[i][0] - 1] = norm_diag;
                }
                else
                {
                    int num2 = 0;
                    for (int k = 0; k < nb_conn[conn[i][j]]; k++)
                    {
                        if (conn[conn[i][j]][k] <= conn[i][j])
                            continue;
                        num2++;
                        if (conn[conn[i][j]][k] == i)
                        {
                            K_global[K_pointer[conn[i][j]][1] + dim * num2 + 2] = 0.0;
                            K_global[K_pointer[conn[i][j]][1] + dim * K_pointer[conn[i][j]][0] + dim * num2 + 1] = 0.0;
                            K_global[K_pointer[conn[i][j]][1] + 2 * dim * K_pointer[conn[i][j]][0] + dim * num2 - 1] = 0.0;
                        }
                    }
                }
            }
            residual[dim * i + 2] = 0.0;
        }
    }
}
