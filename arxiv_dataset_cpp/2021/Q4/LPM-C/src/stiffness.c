#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <mkl.h>

#include "lpm.h"

void calcKnTv()
{
    /************************************************* Square *******************************************/
    if (lattice == 0)
    {
        KnTve = allocDouble2D(ntype, 3, 0);
        double mapping[3 * 3] =
            {1 / 2.0, -1 / 2.0, 0.0,
             0.0, 0.0, 1 / 2.0,
             0.0, 1.0 / 12.0, -1.0 / 12.0};

        for (int k = 0; k < ntype; k++)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 1, 3, 1.0, mapping, 3, Ce[k], 1, 0.0, KnTve[k], 1);

        for (int i = 0; i < nparticle; i++)
        {
            for (int j = 0; j < nb_initial[i]; j++)
            {
                /* assign properties for different layer of neighbors */
                if (nsign[i][j] == 0)
                {
                    Kn[i][j] = KnTve[type[i]][0];
                    Tv[i][j] = KnTve[type[i]][2];
                }
                else if (nsign[i][j] == 1)
                {
                    Kn[i][j] = KnTve[type[i]][1];
                    Tv[i][j] = KnTve[type[i]][2];
                }
            }
        }
    }

    /************************************************* Hexagon ******************************************/
    if (lattice == 1)
    {
        /**
         * for isotropic 2D materials
         */
        KnTve = allocDouble2D(ntype, 2, 0);
        double mapping[2 * 2] =
            {sqrt(3.0) / 12.0, -sqrt(3.0) / 12.0,
             -sqrt(3.0) / 144.0, sqrt(3.0) / 48.0};

        for (int k = 0; k < ntype; k++)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 1, 2, 1.0, mapping, 2, Ce[k], 1, 0.0, KnTve[k], 1);

        for (int i = 0; i < nparticle; i++)
        {
            for (int j = 0; j < nb_initial[i]; j++)
            {
                Kn[i][j] = KnTve[type[i]][0];
                Tv[i][j] = KnTve[type[i]][1];
            }
        }

        /**
         * for anisotropic 2D materials
         */
        // KnTve = allocDouble2D(ntype, 6, 0);

        // double mapping[6 * 6] =
        //     {sqrt(3.0) / 16.0, sqrt(3.0) / 16.0, -sqrt(3.0) / 12.0, 0.0, 0.0, -sqrt(3.0) / 8.0,
        //      0.0, 0.0, sqrt(3.0) / 6.0, 1.0 / 4, -1.0 / 4, 0.0,
        //      0.0, 0.0, sqrt(3.0) / 6.0, -1.0 / 4, 1.0 / 4, 0.0,
        //      -sqrt(3.0) / 24.0, sqrt(3.0) / 64.0, sqrt(3.0) / 24.0, 0.0, 0.0, sqrt(3.0) / 24.0,
        //      sqrt(3.0) / 48.0, -sqrt(3.0) / 144.0, -sqrt(3.0) / 24.0, -1 / 6.0, 0.0, 0.0,
        //      sqrt(3.0) / 48.0, -sqrt(3.0) / 144.0, -sqrt(3.0) / 24.0, 1 / 6.0, 0.0, 0.0};

        // /* since Ce is C11, C12, C44, it needs to be transformed to C11, C22, C66, C26, C16, C12 */
        // double tempCe[6] = {0.};

        // for (int k = 0; k < ntype; k++)
        // {
        //     tempCe[0] = Ce[k][0]; // C11
        //     tempCe[1] = Ce[k][0]; // C22
        //     tempCe[2] = Ce[k][2]; // C66
        //     tempCe[5] = Ce[k][1]; // C12

        //     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 6, 1, 6, 1.0, mapping, 6, tempCe, 1, 0.0, KnTve[k], 1);
        // }

        // for (int i = 0; i < nparticle; i++)
        // {
        //     for (int j = 0; j < nb_initial[i]; j++)
        //     {
        //         double dis = sqrt(pow((xyz[neighbors[i][j]][0] - xyz[i][0]), 2) + pow((xyz[neighbors[i][j]][1] - xyz[i][1]), 2));
        //         double l = (xyz[neighbors[i][j]][0] - xyz[i][0]) / dis;
        //         double m = (xyz[neighbors[i][j]][1] - xyz[i][1]) / dis;

        //         // layer-1
        //         /* spring property 1 */
        //         if ((fabs(l - cos(angle3)) < TOL && fabs(m - sin(angle3)) < TOL) || (fabs(l - cos(angle3 + PI)) < TOL && fabs(m - sin(angle3 + PI)) < TOL))
        //         {
        //             Kn[i][j] = KnTve[type[i]][0];
        //             Tv[i][j] = KnTve[type[i]][3];
        //         }
        //         /* spring property 2 */
        //         else if ((fabs(l - cos(angle3 + PI / 3.0)) < TOL && fabs(m - sin(angle3 + PI / 3.0)) < TOL) || (fabs(l - cos(angle3 + PI / 3.0 + PI)) < TOL && fabs(m - sin(angle3 + PI / 3.0 + PI)) < TOL))
        //         {
        //             Kn[i][j] = KnTve[type[i]][1];
        //             Tv[i][j] = KnTve[type[i]][4];
        //         }
        //         /* spring property 3 */
        //         else if ((fabs(l - cos(angle3 + 2.0 * PI / 3.0)) < TOL && fabs(m - sin(angle3 + 2.0 * PI / 3.0)) < TOL) || (fabs(l - cos(angle3 + 2.0 * PI / 3.0 + PI)) < TOL && fabs(m - sin(angle3 + 2.0 * PI / 3.0 + PI)) < TOL))
        //         {
        //             Kn[i][j] = KnTve[type[i]][2];
        //             Tv[i][j] = KnTve[type[i]][5];
        //         }

        //         // layer-2
        //         /* spring property 4 */
        //         if ((fabs(l - cos(angle3 + PI / 2.0)) < TOL && fabs(m - sin(angle3 + PI / 2.0)) < TOL) || (fabs(l - cos(angle3 + PI / 2.0 + PI)) < TOL && fabs(m - sin(angle3 + PI / 2.0 + PI)) < TOL))
        //         {
        //             Kn[i][j] = KnTve[type[i]][0];
        //             Tv[i][j] = KnTve[type[i]][3];
        //         }
        //         /* spring property 5 */
        //         else if ((fabs(l - cos(angle3 + 5 * PI / 6.0)) < TOL && fabs(m - sin(angle3 + 5 * PI / 6.0)) < TOL) || (fabs(l - cos(angle3 + 5 * PI / 6.0 + PI)) < TOL && fabs(m - sin(angle3 + 5 * PI / 6.0 + PI)) < TOL))
        //         {
        //             Kn[i][j] = KnTve[type[i]][1];
        //             Tv[i][j] = KnTve[type[i]][4];
        //         }
        //         /* spring property 6 */
        //         else if ((fabs(l - cos(angle3 + PI / 6.0)) < TOL && fabs(m - sin(angle3 + PI / 6.0)) < TOL) || (fabs(l - cos(angle3 + PI / 6.0 + PI)) < TOL && fabs(m - sin(angle3 + PI / 6.0 + PI)) < TOL))
        //         {
        //             Kn[i][j] = KnTve[type[i]][2];
        //             Tv[i][j] = KnTve[type[i]][5];
        //         }
        //     }
        // }
    }

    /********************************************** Simple cubic ****************************************/
    if (lattice == 2)
    {
        KnTve = allocDouble2D(ntype, 3, 0);

        /* orthotropic material
        double mapping[9 * 9] =
            {1, 0, 0, -1, 0, 0, -1, -1, 1,
             0, 1, 0, 0, -1, 0, 1, -1, -1,
             0, 0, 1, 0, 0, -1, -1, 1, -1,
             0, 0, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0,
             0, 0, 0, 29 / 378.0, 11 / 378.0, -61 / 378.0, -11 / 378.0, 61 / 378.0, -29 / 378.0,
             0, 0, 0, -61 / 378.0, 29 / 378.0, 11 / 378.0, -29 / 378.0, -11 / 378.0, 61 / 378.0,
             0, 0, 0, 11 / 378.0, -61 / 378.0, 29 / 378.0, 61 / 378.0, -29 / 378.0, -11 / 378.0};

        double tempCe[9] = {0.};

        for (int k = 0; k < ntype; k++)
        {
            tempCe[0] = Ce[k][0]; // C11
            tempCe[1] = Ce[k][0]; // C22
            tempCe[2] = Ce[k][0]; // C33
            tempCe[3] = Ce[k][2]; // C44
            tempCe[4] = Ce[k][2]; // C55
            tempCe[5] = Ce[k][2]; // C66
            tempCe[6] = Ce[k][1]; // C13
            tempCe[7] = Ce[k][1]; // C12
            tempCe[8] = Ce[k][1]; // C23

            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 9, 1, 9, 1.0, mapping, 9, tempCe, 1, 0.0, KnTve[k], 1);
        } */

        // cubic material
        double mapping[3 * 3] =
            {1, -1, -1,
             0, 0, 1,
             0, 1.0 / 18.0, -1.0 / 18.0};

        for (int k = 0; k < ntype; k++)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 1, 3, radius, mapping, 3, Ce[k], 1, 0.0, KnTve[k], 1);

        for (int i = 0; i < nparticle; i++)
        {
            for (int j = 0; j < nb_initial[i]; j++)
            {
                /* assign properties for different layer of neighbors */
                if (nsign[i][j] == 0)
                {
                    Kn[i][j] = 0.5 * (KnTve[type[i]][0] + KnTve[type[neighbors[i][j]]][0]);
                    Tv[i][j] = 0.5 * (KnTve[type[i]][2] + KnTve[type[neighbors[i][j]]][2]);
                }
                else if (nsign[i][j] == 1)
                {
                    Kn[i][j] = 0.5 * (KnTve[type[i]][1] + KnTve[type[neighbors[i][j]]][1]);
                    Tv[i][j] = 0.5 * (KnTve[type[i]][2] + KnTve[type[neighbors[i][j]]][2]);
                }
            }
        }
    }

    /************************************* Face centered cubic, FCC *************************************/
    if (lattice == 3)
    {
        KnTve = allocDouble2D(ntype, 3, 0);
        double mapping[3 * 3] =
            {0, 0, sqrt(2.0),
             sqrt(2.0) / 4.0, -sqrt(2.0) / 4.0, -sqrt(2.0) / 4.0,
             0, sqrt(2.0) / 24.0, -sqrt(2.0) / 24.0};

        for (int k = 0; k < ntype; k++)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 1, 3, radius, mapping, 3, Ce[k], 1, 0.0, KnTve[k], 1);

        for (int i = 0; i < nparticle; i++)
        {
            for (int j = 0; j < nb_initial[i]; j++)
            {
                /* assign properties for different layer of neighbors */
                if (nsign[i][j] == 0)
                {
                    Kn[i][j] = KnTve[type[i]][0];
                    Tv[i][j] = KnTve[type[i]][2];
                }
                else if (nsign[i][j] == 1)
                {
                    Kn[i][j] = KnTve[type[i]][1];
                    Tv[i][j] = KnTve[type[i]][2];
                }
            }
        }
    }

    /************************************* Body centered cubic, BCC *************************************/
    if (lattice == 4)
    {
        KnTve = allocDouble2D(ntype, 3, 0);

        double mapping[3 * 3] =
            {0., 0., sqrt(3.0),
             1. / sqrt(3.0), -1. / sqrt(3.0), 0.,
             0., sqrt(3.0) / 14.0, sqrt(3.0) / 14.0};

        for (int k = 0; k < ntype; k++)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 1, 3, radius, mapping, 3, Ce[k], 1, 0.0, KnTve[k], 1);

        for (int i = 0; i < nparticle; i++)
        {
            for (int j = 0; j < nb_initial[i]; j++)
            {
                /* assign properties for different layer of neighbors */
                if (nsign[i][j] == 0)
                {
                    Kn[i][j] = KnTve[type[i]][0];
                    Tv[i][j] = KnTve[type[i]][2];
                }
                else if (nsign[i][j] == 1)
                {
                    Kn[i][j] = KnTve[type[i]][1];
                    Tv[i][j] = KnTve[type[i]][2];
                }
            }
        }
    }
}

/* compute the stiffness matrix in 2D using finite difference */
void calcStiffness2DFiniteDifference(int plmode)
{
    memset(K_global, 0.0, K_pointer[nparticle][1] * sizeof(double)); /* initialization od stiffness matrix */

#pragma omp parallel for
    for (int i = 0; i < nparticle; i++)
    {
        double *K_local = allocDouble1D(dim * nb_conn[i] * dim, 0.);

        // compute the local stiffness matrix using finite difference method
        if (plmode == 6)
            computeBondForceElastic(i);
        double Fintemp[NDIM] = {Pin[NDIM * i], Pin[NDIM * i + 1], Pin[NDIM * i + 2]};
        for (int jID = 0; jID < nb_conn[i]; jID++)
        {
            for (int r = 0; r < dim; r++)
            {
                // central-difference
                // double xtemp = xyz[conn[i][jID]][r];
                // xyz[conn[i][jID]][r] -= EPS * radius;
                // if (plmode == 6)
                //     computeBondForceElastic(i);
                // Fintemp[0] = Pin[NDIM * i], Fintemp[1] = Pin[NDIM * i + 1], Fintemp[2] = Pin[NDIM * i + 2];
                // xyz[conn[i][jID]][r] = xtemp + EPS * radius;
                // if (plmode == 6)
                //     computeBondForceElastic(i);
                // xyz[conn[i][jID]][r] = xtemp; // restore to the original configuration
                // for (int s = 0; s < dim; s++)
                // {
                //     double Kvalue = 0.5 * (Pin[NDIM * i + s] - Fintemp[s]) / EPS / radius;
                //     K_local[dim * nb_conn[i] * r + dim * jID + s] = Kvalue;
                // }

                // forward-difference
                double xtemp = xyz[conn[i][jID]][r];
                xyz[conn[i][jID]][r] = xtemp + EPS * radius;
                if (plmode == 6)
                    computeBondForceElastic(i);
                xyz[conn[i][jID]][r] = xtemp; // restore to the original configuration
                for (int s = 0; s < dim; s++)
                {
                    double Kvalue = (Pin[NDIM * i + s] - Fintemp[s]) / EPS / radius;
                    K_local[dim * nb_conn[i] * r + dim * jID + s] = Kvalue;
                }
            }
        }

        // assemble K_local into the global stiffness matrix
        int num1 = 0;
        for (int jID = 0; jID < nb_conn[i]; jID++)
        {
            int jj = conn[i][jID];
            double K_locallocal[NDIM - 1][NDIM - 1] = {{0.0}};
            for (int r = 0; r < dim; r++)
                for (int s = 0; s < dim; s++)
                    K_locallocal[r][s] = K_local[dim * nb_conn[i] * s + dim * jID + r];

            if (jj == i)
            {
                K_global[K_pointer[i][1]] += K_locallocal[0][0];
                K_global[K_pointer[i][1] + 1] += K_locallocal[0][1];
                K_global[K_pointer[i][1] + dim * (K_pointer[i][0])] += K_locallocal[1][1];
            }
            else if (jj > i)
            {
                num1++;
                K_global[K_pointer[i][1] + dim * num1] += 0.5 * K_locallocal[0][0];
                K_global[K_pointer[i][1] + dim * num1 + 1] += 0.5 * K_locallocal[0][1];
                K_global[K_pointer[i][1] + dim * (K_pointer[i][0]) + dim * num1 - 1] += 0.5 * K_locallocal[1][0];
                K_global[K_pointer[i][1] + dim * (K_pointer[i][0]) + dim * num1] += 0.5 * K_locallocal[1][1];
            }
            else
            {
                int num2 = 0;
                for (int k = 0; k < nb_conn[jj]; k++)
                {
                    if (conn[jj][k] <= jj)
                        continue;
                    num2++;
                    if (conn[jj][k] == i)
                    {
                        K_global[K_pointer[jj][1] + dim * num2] += 0.5 * K_locallocal[0][0];
                        K_global[K_pointer[jj][1] + dim * num2 + 1] += 0.5 * K_locallocal[1][0];
                        K_global[K_pointer[jj][1] + dim * (K_pointer[jj][0]) + dim * num2 - 1] += 0.5 * K_locallocal[0][1];
                        K_global[K_pointer[jj][1] + dim * (K_pointer[jj][0]) + dim * num2] += 0.5 * K_locallocal[1][1];
                    }
                }
            }
            /* The JK array */
            if (jj == i)
            {
                JK[K_pointer[i][1]] = dim * (jj + 1) - 1;
                JK[K_pointer[i][1] + 1] = dim * (jj + 1);
                JK[K_pointer[i][1] + dim * (K_pointer[i][0])] = dim * (jj + 1);
            }
            else if (jj > i)
            {
                JK[K_pointer[i][1] + dim * num1] = dim * (jj + 1) - 1;
                JK[K_pointer[i][1] + dim * num1 + 1] = dim * (jj + 1);
                JK[K_pointer[i][1] + dim * (K_pointer[i][0]) + dim * num1 - 1] = dim * (jj + 1) - 1;
                JK[K_pointer[i][1] + dim * (K_pointer[i][0]) + dim * num1] = dim * (jj + 1);
            }
        }

        /* The IK array */
        IK[dim * i] = K_pointer[i][1] + 1;
        IK[dim * i + 1] = K_pointer[i][1] + dim * K_pointer[i][0] + 1;
        free(K_local);
    }
    IK[dim * nparticle] = K_pointer[nparticle][1] + 1;
}

/* compute the stiffness matrix in 3D using finite difference */
void calcStiffness3DFiniteDifference(int plmode)
{
    memset(K_global, 0.0, K_pointer[nparticle][1] * sizeof(double)); /* initialization od stiffness matrix */

#pragma omp parallel for
    for (int i = 0; i < nparticle; i++)
    {
        double *K_local = allocDouble1D(NDIM * nb_conn[i] * NDIM, 0.);

        // compute the local stiffness matrix using finite difference method
        if (plmode == 6)
            computeBondForceElastic(i);
        double Fintemp[NDIM] = {Pin[NDIM * i], Pin[NDIM * i + 1], Pin[NDIM * i + 2]};
        for (int jID = 0; jID < nb_conn[i]; jID++)
        {
            for (int r = 0; r < dim; r++)
            {
                // central-difference
                // double xtemp = xyz[conn[i][jID]][r];
                // xyz[conn[i][jID]][r] -= EPS * radius;
                // if (plmode == 6)
                //     computeBondForceElastic(i);
                // Fintemp[0] = Pin[NDIM * i], Fintemp[1] = Pin[NDIM * i + 1], Fintemp[2] = Pin[NDIM * i + 2];
                // xyz[conn[i][jID]][r] = xtemp + EPS * radius;
                // if (plmode == 6)
                //     computeBondForceElastic(i);
                // xyz[conn[i][jID]][r] = xtemp; // restore to the original configuration
                // for (int s = 0; s < dim; s++)
                // {
                //     double Kvalue = 0.5 * (Pin[NDIM * i + s] - Fintemp[s]) / EPS / radius;
                //     K_local[dim * nb_conn[i] * r + dim * jID + s] = Kvalue;
                // }

                // forward-difference
                double xtemp = xyz[conn[i][jID]][r];
                xyz[conn[i][jID]][r] = xtemp + EPS * radius;
                if (plmode == 6)
                    computeBondForceElastic(i);
                xyz[conn[i][jID]][r] = xtemp; // restore to the original configuration
                for (int s = 0; s < dim; s++)
                {
                    double Kvalue = (Pin[NDIM * i + s] - Fintemp[s]) / EPS / radius;
                    K_local[dim * nb_conn[i] * r + dim * jID + s] = Kvalue;
                }
            }
        }

        // assemble K_local into the global stiffness matrix
        int num1 = 0;
        for (int jID = 0; jID < nb_conn[i]; jID++)
        {
            int jj = conn[i][jID];
            double K_locallocal[NDIM][NDIM] = {{0.0}};
            for (int r = 0; r < NDIM; r++)
                for (int s = 0; s < NDIM; s++)
                    K_locallocal[r][s] = K_local[NDIM * nb_conn[i] * s + NDIM * jID + r];

            if (jj == i)
            {
                K_global[K_pointer[i][1]] += K_locallocal[0][0];
                K_global[K_pointer[i][1] + 1] += K_locallocal[0][1];
                K_global[K_pointer[i][1] + 2] += K_locallocal[0][2];
                K_global[K_pointer[i][1] + dim * (K_pointer[i][0])] += K_locallocal[1][1];
                K_global[K_pointer[i][1] + dim * (K_pointer[i][0]) + 1] += K_locallocal[1][2];
                K_global[K_pointer[i][1] + 2 * dim * (K_pointer[i][0]) - 1] += K_locallocal[2][2];
            }
            else if (jj > i)
            {
                num1++;
                K_global[K_pointer[i][1] + dim * num1] += 0.5 * K_locallocal[0][0];
                K_global[K_pointer[i][1] + dim * num1 + 1] += 0.5 * K_locallocal[0][1];
                K_global[K_pointer[i][1] + dim * num1 + 2] += 0.5 * K_locallocal[0][2];
                K_global[K_pointer[i][1] + dim * (K_pointer[i][0]) + dim * num1 - 1] += 0.5 * K_locallocal[1][0];
                K_global[K_pointer[i][1] + dim * (K_pointer[i][0]) + dim * num1] += 0.5 * K_locallocal[1][1];
                K_global[K_pointer[i][1] + dim * (K_pointer[i][0]) + dim * num1 + 1] += 0.5 * K_locallocal[1][2];
                K_global[K_pointer[i][1] + 2 * dim * (K_pointer[i][0]) + dim * num1 - 3] += 0.5 * K_locallocal[2][0];
                K_global[K_pointer[i][1] + 2 * dim * (K_pointer[i][0]) + dim * num1 - 2] += 0.5 * K_locallocal[2][1];
                K_global[K_pointer[i][1] + 2 * dim * (K_pointer[i][0]) + dim * num1 - 1] += 0.5 * K_locallocal[2][2];
            }
            else
            {
                int num2 = 0;
                for (int k = 0; k < nb_conn[jj]; k++)
                {
                    if (conn[jj][k] <= jj)
                        continue;
                    num2++;
                    if (conn[jj][k] == i)
                    {
                        K_global[K_pointer[jj][1] + dim * num2] += 0.5 * K_locallocal[0][0];
                        K_global[K_pointer[jj][1] + dim * num2 + 1] += 0.5 * K_locallocal[1][0];
                        K_global[K_pointer[jj][1] + dim * num2 + 2] += 0.5 * K_locallocal[2][0];
                        K_global[K_pointer[jj][1] + dim * (K_pointer[jj][0]) + dim * num2 - 1] += 0.5 * K_locallocal[0][1];
                        K_global[K_pointer[jj][1] + dim * (K_pointer[jj][0]) + dim * num2] += 0.5 * K_locallocal[1][1];
                        K_global[K_pointer[jj][1] + dim * (K_pointer[jj][0]) + dim * num2 + 1] += 0.5 * K_locallocal[2][1];
                        K_global[K_pointer[jj][1] + 2 * dim * (K_pointer[jj][0]) + dim * num2 - 3] += 0.5 * K_locallocal[0][2];
                        K_global[K_pointer[jj][1] + 2 * dim * (K_pointer[jj][0]) + dim * num2 - 2] += 0.5 * K_locallocal[1][2];
                        K_global[K_pointer[jj][1] + 2 * dim * (K_pointer[jj][0]) + dim * num2 - 1] += 0.5 * K_locallocal[2][2];
                    }
                }
            }
            /* The JK array */
            if (jj == i)
            {
                JK[K_pointer[i][1]] = dim * (jj + 1) - 2;
                JK[K_pointer[i][1] + 1] = dim * (jj + 1) - 1;
                JK[K_pointer[i][1] + 2] = dim * (jj + 1);
                JK[K_pointer[i][1] + dim * (K_pointer[i][0])] = dim * (jj + 1) - 1;
                JK[K_pointer[i][1] + dim * (K_pointer[i][0]) + 1] = dim * (jj + 1);
                JK[K_pointer[i][1] + 2 * dim * (K_pointer[i][0]) - 1] = dim * (jj + 1);
            }
            else if (jj > i)
            {
                JK[K_pointer[i][1] + dim * num1] = dim * (jj + 1) - 2;
                JK[K_pointer[i][1] + dim * num1 + 1] = dim * (jj + 1) - 1;
                JK[K_pointer[i][1] + dim * num1 + 2] = dim * (jj + 1);
                JK[K_pointer[i][1] + dim * (K_pointer[i][0]) + dim * num1 - 1] = dim * (jj + 1) - 2;
                JK[K_pointer[i][1] + dim * (K_pointer[i][0]) + dim * num1] = dim * (jj + 1) - 1;
                JK[K_pointer[i][1] + dim * (K_pointer[i][0]) + dim * num1 + 1] = dim * (jj + 1);
                JK[K_pointer[i][1] + 2 * dim * (K_pointer[i][0]) + dim * num1 - 3] = dim * (jj + 1) - 2;
                JK[K_pointer[i][1] + 2 * dim * (K_pointer[i][0]) + dim * num1 - 2] = dim * (jj + 1) - 1;
                JK[K_pointer[i][1] + 2 * dim * (K_pointer[i][0]) + dim * num1 - 1] = dim * (jj + 1);
            }
        }

        /* The IK array */
        IK[dim * i] = K_pointer[i][1] + 1;
        IK[dim * i + 1] = K_pointer[i][1] + dim * K_pointer[i][0] + 1;
        IK[dim * i + 2] = K_pointer[i][1] + 2 * dim * K_pointer[i][0];
        free(K_local);
    }
    IK[dim * nparticle] = K_pointer[nparticle][1] + 1;
}

/* compute the residual force vector using external/internal forces, residual = F_ex - F_in */
void updateRR()
{
    int ii = 0; /* index of reaction force */
    for (int i = 0; i < nparticle; i++)
    {
        for (int k = 0; k < dim; k++)
        {
            /* residual force (exclude the DoF that being applied to displacement BC) */
            residual[dim * i + k] = dispBC_index[dim * i + k] * (Pex[dim * i + k] - Pin[NDIM * i + k]);

            /* reaction force (DoF being applied to displacement BC) */
            if (dispBC_index[dim * i + k] == 0)
                reaction_force[ii++] = Pin[NDIM * i + k];
        }
    }
}
