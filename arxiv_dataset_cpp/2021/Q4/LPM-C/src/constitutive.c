#include <stdio.h>
#include <math.h>
#include <memory.h>
#include <mkl.h>
#include <omp.h>

#include "lpm.h"

/* switch the state variables from the converged one to the current one (flag=0) or vice versa (flag=1) or gather together (flag=2) */
void switchStateV(int conv_flag)
{

    if (conv_flag == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < nparticle; i++)
        {
            for (int j = 0; j < nneighbors; j++)
            {
                dLp[i][j][0] = dLp[i][j][1];
                damage_D[i][j][0] = damage_D[i][j][1];
            }
            for (int j = 0; j < 2 * NDIM; j++)
                J2_beta[i][j][0] = J2_beta[i][j][1];
            for (int j = 0; j < nslipSys; j++)
            {
                cp_gy[i][j][0] = cp_gy[i][j][1];
                cp_A_single[i][j][0] = cp_A_single[i][j][1];
            }

            J2_alpha[i][0] = J2_alpha[i][1];
            J2_beta_eq[i][0] = J2_beta_eq[i][1];
            damage_local[i][0] = damage_local[i][1];
            damage_nonlocal[i][0] = damage_nonlocal[i][1];
            cp_A[i][0] = cp_A[i][1];
        }
    }
    else if (conv_flag == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < nparticle; i++)
        {
            for (int j = 0; j < nneighbors; j++)
            {
                dLp[i][j][1] = dLp[i][j][0];
                damage_D[i][j][1] = damage_D[i][j][0];
            }
            for (int j = 0; j < 2 * NDIM; j++)
                J2_beta[i][j][1] = J2_beta[i][j][0];
            for (int j = 0; j < nslipSys; j++)
            {
                cp_gy[i][j][1] = cp_gy[i][j][0];
                cp_A_single[i][j][1] = cp_A_single[i][j][0];
            }

            J2_alpha[i][1] = J2_alpha[i][0];
            J2_beta_eq[i][1] = J2_beta_eq[i][0];
            damage_local[i][1] = damage_local[i][0];
            damage_nonlocal[i][1] = damage_nonlocal[i][0];
            cp_A[i][1] = cp_A[i][0];
        }
    }
    else if (conv_flag == 2)
    {
#pragma omp parallel for
        for (int i = 0; i < nparticle; i++)
        {
            for (int j = 0; j < nneighbors; j++)
            {
                dLp[i][j][0] = dLp[i][j][2];
            }
            for (int j = 0; j < 2 * NDIM; j++)
                J2_beta[i][j][0] = J2_beta[i][j][2];
            for (int j = 0; j < nslipSys; j++)
            {
                cp_gy[i][j][0] = cp_gy[i][j][2];
                cp_A_single[i][j][0] = cp_A_single[i][j][2];
            }

            J2_alpha[i][0] = J2_alpha[i][2];
            J2_beta_eq[i][0] = J2_beta_eq[i][2];
            cp_A[i][0] = cp_A[i][2];
        }
    }
}

/* compute the bond force and the state variables using constitutive relationship determined by plmode */
void computeBondForceGeneral(int plmode, int t)
{
    // double el_radius = 7.0;
    // double pc1[3] = {10.0, 13.0, 0.0}, pc2[3] = {10.0, 35.0, 0.0};

    if (plmode == 0)
    {
#pragma omp parallel for
        for (int iID = 0; iID < nparticle; iID++)
        {
            computeBondForceJ2mixedLinear3D(iID);
            // double disq1 = pow(xyz[iID][0] - pc1[0], 2.0) + pow(xyz[iID][1] - pc1[1], 2.0);
            // double disq2 = pow(xyz[iID][0] - pc2[0], 2.0) + pow(xyz[iID][1] - pc2[1], 2.0);
            // if (type[iID] >= 1 && type[iID] <= 6)
            //     computeBondForceElastic(iID);
            // else
            //     computeBondForceJ2mixedLinear3D(iID);

            // if ((xyz[iID][1] > 18 && xyz[iID][1] < 30) || xyz[iID][0] > 18)
            //     computeBondForceJ2mixedLinear3D(iID);
            // else
            //     computeBondForceElastic(iID);
        }
    }
    else if (plmode == 1)
    {
        memset(state_v, 0, nparticle * sizeof(int)); // initialize the state variables flag
        // #pragma omp parallel for
        for (int iID = 0; iID < nparticle; iID++)
            computeBondForceCPMiehe(iID);
    }
    else if (plmode == 3)
    {
#pragma omp parallel for
        for (int iID = 0; iID < nparticle; iID++)
            computeBondForceJ2energyReturnMap(iID, t);
    }
    else if (plmode == 4)
    {
#pragma omp parallel for
        for (int iID = 0; iID < nparticle; iID++)
            computeBondForceIncrementalUpdating(iID);
    }
    else if (plmode == 5)
    {
#pragma omp parallel for
        for (int iID = 0; iID < nparticle; iID++)
            computeBondForceJ2nonlinearIso(iID);
    }
    else if (plmode == 6)
    {
#pragma omp parallel for
        for (int iID = 0; iID < nparticle; iID++)
            computeBondForceElastic(iID);
    }

    computeStress();
    switchStateV(2); // gather together the pointwise state variables
}

/* update the damage variables */
int updateDamageGeneral(const char *dataName, int tstep, int plmode)
{
    int broken = 0;

    if (plmode == 0)
        broken = updateDuctileDamagePwiseNonlocal(dataName, tstep);
    // broken = updateDuctileDamagePwiseLocal(dataName, tstep);
    // broken = updateDuctileDamageBwiseNonlocal(dataName, tstep);
    // broken = updateDuctileDamageBwiseLocal(dataName, tstep);
    else if (plmode == 5)
        broken = updateDuctileDamageBwiseLocal(dataName, tstep);
    else if (plmode == 6)
        broken = updateBrittleDamage(dataName, tstep, nbreak);

    return broken;
}

/* Incremental updating (elastic) in general and for Wei's algorithm, trial */
void computeBondForceIncrementalUpdating(int ii)
{
    // create a temporary list to store particle ii and all of its neighbors ID
    int *temporary_nb = allocInt1D(nb[ii] + 1, ii);
    int s = 1;
    for (int k = 1; k < nneighbors + 1; k++)
    {
        if (damage_broken[ii][k - 1] > EPS && neighbors[ii][k - 1] != -1)
            temporary_nb[s++] = neighbors[ii][k - 1];
    }

    //#pragma omp parallel for
    // initialize the dilatational change for the above particle list
    for (int k = 0; k < nb[ii] + 1; k++)
    {
        int i = temporary_nb[k];
        ddL_total[i][0] = 0, TddL_total[i][0] = 0;
        ddL_total[i][1] = 0, TddL_total[i][1] = 0;

        for (int j = 0; j < nb_initial[i]; j++)
        {
            double dis0 = sqrt(pow(xyz_temp[i][0] - xyz_temp[neighbors[i][j]][0], 2) +
                               pow(xyz_temp[i][1] - xyz_temp[neighbors[i][j]][1], 2) +
                               pow(xyz_temp[i][2] - xyz_temp[neighbors[i][j]][2], 2));
            double dis1 = sqrt(pow(xyz[i][0] - xyz[neighbors[i][j]][0], 2) +
                               pow(xyz[i][1] - xyz[neighbors[i][j]][1], 2) +
                               pow(xyz[i][2] - xyz[neighbors[i][j]][2], 2));
            ddL[i][j] = damage_broken[i][j] * (dis1 - dis0);
            ddL_total[i][nsign[i][j]] += ddL[i][j];
            TddL_total[i][nsign[i][j]] += Tv[i][j] * ddL[i][j];
            // csx[i][j] = (xyz[i][0] - xyz[neighbors[i][j]][0]) / dis1;
            // csy[i][j] = (xyz[i][1] - xyz[neighbors[i][j]][1]) / dis1;
            // csz[i][j] = (xyz[i][2] - xyz[neighbors[i][j]][2]) / dis1;
        }
    }

    // update the bond force for the above particle list
    int i = ii;
    // memset(stress_tensor[i], 0.0, 2 * NDIM * sizeof(double));              // initialize the stress tensor
    Pin[i * NDIM] = 0.0, Pin[i * NDIM + 1] = 0.0, Pin[i * NDIM + 2] = 0.0; // initialize the internal force vector

    //#pragma omp parallel for
    for (int j = 0; j < nb_initial[i]; j++)
    {
        /* update Fij */
        F[i][j] = F_temp[i][j] + 2.0 * Kn[i][j] * ddL[i][j] + 0.5 * (TddL_total[i][nsign[i][j]] + TddL_total[neighbors[i][j]][nsign[i][j]]) + 0.5 * Tv[i][j] * (ddL_total[i][nsign[i][j]] + ddL_total[neighbors[i][j]][nsign[i][j]]);
        F[i][j] *= damage_broken[i][j];
        // F[i][j] *= damage_w[i][j];
        // F[i][j] *= (1.0 - damage_D[i][j][0]);

        /* compute internal forces */
        Pin[i * NDIM] += csx[i][j] * F[i][j];
        Pin[i * NDIM + 1] += csy[i][j] * F[i][j];
        Pin[i * NDIM + 2] += csz[i][j] * F[i][j]; // 0 for 2D
    }

    // free the array memory of particle ii
    free(temporary_nb);
}

/* Elastic material law */
void computeBondForceElastic(int ii)
{
    // create a temporary list to store particle ii and all of its neighbors ID
    int *temporary_nb = allocInt1D(nb[ii] + 1, ii);
    int s = 1;
    for (int k = 1; k < nneighbors + 1; k++)
    {
        if (damage_broken[ii][k - 1] > EPS && neighbors[ii][k - 1] != -1)
            temporary_nb[s++] = neighbors[ii][k - 1];
    }

    //#pragma omp parallel for
    // compute the dilatational change for the above particle list
    for (int k = 0; k < nb[ii] + 1; k++)
    {
        int i = temporary_nb[k];
        dL_total[i][0] = 0, TdL_total[i][0] = 0;
        dL_total[i][1] = 0, TdL_total[i][1] = 0;

        for (int j = 0; j < nb_initial[i]; j++)
        {
            double dis = sqrt(pow(xyz[i][0] - xyz[neighbors[i][j]][0], 2) +
                              pow(xyz[i][1] - xyz[neighbors[i][j]][1], 2) +
                              pow(xyz[i][2] - xyz[neighbors[i][j]][2], 2));
            dL[i][j] = dis - distance_initial[i][j];
            dL[i][j] -= dLp[i][j][0]; // elastic bond stretch
            dL[i][j] *= damage_broken[i][j];
            dL_total[i][nsign[i][j]] += dL[i][j];
            TdL_total[i][nsign[i][j]] += Tv[i][j] * dL[i][j];
            csx[i][j] = (xyz[i][0] - xyz[neighbors[i][j]][0]) / dis;
            csy[i][j] = (xyz[i][1] - xyz[neighbors[i][j]][1]) / dis;
            csz[i][j] = (xyz[i][2] - xyz[neighbors[i][j]][2]) / dis;
        }
    }

    // update the bond force
    int i = ii;
    // memset(stress_tensor[i], 0.0, 2 * NDIM * sizeof(double));              // initialize the stress tensor
    Pin[i * NDIM] = 0.0, Pin[i * NDIM + 1] = 0.0, Pin[i * NDIM + 2] = 0.0; // initialize the internal force vector
    for (int j = 0; j < nb_initial[i]; j++)
    {
        /* update Fij */
        F[i][j] = 2.0 * Kn[i][j] * dL[i][j] + 0.5 * (TdL_total[i][nsign[i][j]] + TdL_total[neighbors[i][j]][nsign[i][j]]) + 0.5 * Tv[i][j] * (dL_total[i][nsign[i][j]] + dL_total[neighbors[i][j]][nsign[i][j]]);
        F[i][j] *= damage_broken[i][j];
        // F[i][j] *= damage_w[i][j];
        // F[i][j] *= (1.0 - damage_D[i][j][0]);

        /* compute internal forces */
        Pin[i * NDIM] += csx[i][j] * F[i][j];
        Pin[i * NDIM + 1] += csy[i][j] * F[i][j];
        Pin[i * NDIM + 2] += csz[i][j] * F[i][j]; // 0 for 2D
    }

    // finish the loop for particle ii
    free(temporary_nb);
}

/* Energy-based return-mapping algorithm for J2 plasticity */
void computeBondForceJ2energyReturnMap(int ii, int load_indicator)
{
    // create a temporary list to store particle ii and all of its neighbors ID
    int *temporary_nb = allocInt1D(nb[ii] + 1, ii);
    int s = 1;
    for (int k = 1; k < nneighbors + 1; k++)
    {
        if (damage_broken[ii][k - 1] > EPS && neighbors[ii][k - 1] != -1)
            temporary_nb[s++] = neighbors[ii][k - 1];
    }

    // allocate temporary variables to store state variables in previous time step
    double *xJ2_alpha = allocDouble1D(nb[ii] + 1, 0.0);         /* accumulated equivalent plasticity strain */
    double *xJ2_beta_eq = allocDouble1D(nb[ii] + 1, 0.0);       /* equivilent back stress */
    double *xJ2_dlambda = allocDouble1D(nb[ii] + 1, 0.0);       /* plasticity multiplier */
    double **xdLp = allocDouble2D(nb[ii] + 1, nneighbors, 0.0); /* plastic bond stretch */

    // store the previous step state variables (history dependent) for the particles in above list
    for (int k = 0; k < nb[ii] + 1; k++)
    {
        int i = temporary_nb[k]; // global index for the particle
        for (int j = 0; j < nneighbors; j++)
            xdLp[k][j] = dLp[i][j][0];
        xJ2_beta_eq[k] = J2_beta_eq[i][0];
        xJ2_alpha[k] = J2_alpha[i][0];
    }

    // initialize the dilatational change for the above particle list
    for (int k = 0; k < nb[ii] + 1; k++)
    {
        int i = temporary_nb[k]; // global index for the particle
        dL_total[i][0] = 0, TdL_total[i][0] = 0;
        dL_total[i][1] = 0, TdL_total[i][1] = 0;

        for (int j = 0; j < nb_initial[i]; j++)
        {
            double dis = sqrt(pow(xyz[i][0] - xyz[neighbors[i][j]][0], 2) +
                              pow(xyz[i][1] - xyz[neighbors[i][j]][1], 2) +
                              pow(xyz[i][2] - xyz[neighbors[i][j]][2], 2));
            dL[i][j] = dis - distance_initial[i][j];
            dL[i][j] -= xdLp[k][j]; // elastic bond stretch
            dL[i][j] *= damage_broken[i][j];
            dL_total[i][nsign[i][j]] += dL[i][j];
            TdL_total[i][nsign[i][j]] += Tv[i][j] * dL[i][j];
            csx[i][j] = (xyz[i][0] - xyz[neighbors[i][j]][0]) / dis;
            csy[i][j] = (xyz[i][1] - xyz[neighbors[i][j]][1]) / dis;
            csz[i][j] = (xyz[i][2] - xyz[neighbors[i][j]][2]) / dis;
        }
    }

    for (int k = 0; k < nb[ii] + 1; k++)
    {
        int i = temporary_nb[k];
        int nb1 = countNEqual(neighbors1[i], nneighbors1, -1); /* number of neighbors layer-1 */
        int nb2 = nb[i] - nb1;                                 /* number of neighbors layer-2 */

        /* trial distortional energy U_d */
        double U_d = 0.0, J2_k = 0.0, J2_sigma = 0.0;
        double J2_V = particle_volume * nb[i] / nneighbors; // modified particle volume
        for (int j = 0; j < nb[i]; j++)
        {
            if (nsign[i][j] == 0)
            {
                U_d += 0.5 * Kn[i][j] * (dL[i][j] - dL_total[i][0] / nb1) * (dL[i][j] - dL_total[i][0] / nb1);
                J2_k = Kn[i][j]; // for calculation of equivalent plastic strain
            }
            else if (nsign[i][j] == 1)
            {
                U_d += 0.5 * Kn[i][j] * (dL[i][j] - dL_total[i][1] / nb2) * (dL[i][j] - dL_total[i][1] / nb2);
            }
        }
        J2_sigma = sqrt(6.0 * Ce[type[i]][2] * U_d / J2_V); // equivalent stress

        /* test the trial yield function */
        double dlambda = 0.0;
        double yield_func = fabs(load_indicator * J2_sigma - J2_beta_eq[i][0]) - (sigmay[i] + (1.0 - J2_xi) * J2_H * J2_alpha[i][0]);
        if (yield_func > 0.0)
        {
            pl_flag[i] = 1;

            /* solve the nonlinear equation to obtain lambda, using bisection, 0 < dlambda < 1 */
            double a = 0.0, b = 1.0;
            double ya = yield_func, yb = -1.0;
            while ((b - a) > TOLITER)
            {
                /* note the yield function below is at the current timestep */
                dlambda = (a + b) / 2.0;
                yield_func = fabs(load_indicator * J2_sigma / (1.0 + 0.5 * dlambda) - (J2_beta_eq[i][0] + load_indicator * J2_xi * J2_H * dlambda * J2_sigma * sqrt(radius / J2_k / Ce[type[i]][2]) / 6.0 / (1.0 + 0.5 * dlambda))) -
                             (sigmay[i] + (1.0 - J2_xi) * J2_H * J2_alpha[i][0] + (1.0 - J2_xi) * J2_H * dlambda * J2_sigma * sqrt(radius / J2_k / Ce[type[i]][2]) / 6.0 / (1.0 + 0.5 * dlambda));

                /* change the bisection */
                if (yield_func * ya < 0.0)
                {
                    b = dlambda;
                    yb = yield_func;
                }
                else
                {
                    a = dlambda;
                    ya = yield_func;
                }
            }
        }
        xJ2_dlambda[k] = dlambda;
        xJ2_alpha[k] += dlambda / (1.0 + 0.5 * dlambda) * sqrt(radius / 6.0 / J2_k * U_d / J2_V);                                                     // equivalent plastic strain
        xJ2_beta_eq[k] = (xJ2_beta_eq[k] + load_indicator * J2_xi * J2_H * dlambda / (1.0 + 0.5 * dlambda) * sqrt(radius / 6.0 / J2_k * U_d / J2_V)); // equivalent back stress

        /* incremental plastic bond stretch */
        double f_d = 0.0;
        for (int j = 0; j < nb[i]; j++)
        {
            if (nsign[i][j] == 0)
                f_d = 2.0 * Kn[i][j] / (1 + dlambda / 2.0) * (dL[i][j] - dL_total[i][nsign[i][j]] / nb1);
            if (nsign[i][j] == 1)
                f_d = 2.0 * Kn[i][j] / (1 + dlambda / 2.0) * (dL[i][j] - dL_total[i][nsign[i][j]] / nb2);
            ddLp[i][j] = dlambda * f_d / (4.0 * Kn[i][j]);
            ddLp[i][j] *= damage_broken[i][j];
            xdLp[k][j] += ddLp[i][j]; /* update the current plastic bond stretch */
        }
    }

    // update the dilatation terms
    for (int k = 0; k < nb[ii] + 1; k++)
    {
        int i = temporary_nb[k];
        dL_total[i][0] = 0, TdL_total[i][0] = 0;
        dL_total[i][1] = 0, TdL_total[i][1] = 0;

        for (int j = 0; j < nb_initial[i]; j++)
        {
            double dis = sqrt(pow(xyz[i][0] - xyz[neighbors[i][j]][0], 2) +
                              pow(xyz[i][1] - xyz[neighbors[i][j]][1], 2) +
                              pow(xyz[i][2] - xyz[neighbors[i][j]][2], 2));
            dL[i][j] = dis - distance_initial[i][j];
            dL[i][j] -= xdLp[k][j]; // elastic bond stretch
            dL[i][j] *= damage_broken[i][j];
            dL_total[i][nsign[i][j]] += dL[i][j];
            TdL_total[i][nsign[i][j]] += Tv[i][j] * dL[i][j];
            csx[i][j] = (xyz[i][0] - xyz[neighbors[i][j]][0]) / dis;
            csy[i][j] = (xyz[i][1] - xyz[neighbors[i][j]][1]) / dis;
            csz[i][j] = (xyz[i][2] - xyz[neighbors[i][j]][2]) / dis;
        }
    }

    // compute the average elastic bond stretch only for particle i, then compute the bond force, stress and internal force
    int i = ii;
    Pin[i * NDIM] = 0.0, Pin[i * NDIM + 1] = 0.0, Pin[i * NDIM + 2] = 0.0; // initialize the internal force vector
    for (int j = 0; j < nb_initial[i]; j++)
    {
        for (int jj = 0; jj < nneighbors; jj++)
        {
            if (neighbors[neighbors[i][j]][jj] == i) /* find the opposite particle */
                dL_ave[i][j] = 0.5 * (dL[i][j] + dL[neighbors[i][j]][jj]);
        }
        F[i][j] = 2.0 * Kn[i][j] * dL_ave[i][j] + 0.5 * (TdL_total[i][nsign[i][j]] + TdL_total[neighbors[i][j]][nsign[i][j]]) + 0.5 * Tv[i][j] * (dL_total[i][nsign[i][j]] + dL_total[neighbors[i][j]][nsign[i][j]]);
        //F[i][j] *= damage_w[i][j];
        F[i][j] *= (1.0 - damage_D[i][j][0]);

        /* compute internal forces */
        Pin[i * NDIM] += csx[i][j] * F[i][j];
        Pin[i * NDIM + 1] += csy[i][j] * F[i][j];
        Pin[i * NDIM + 2] += csz[i][j] * F[i][j]; // 0 for 2D
    }

    // store the state variables for the particle itself
    for (int j = 0; j < nneighbors; j++)
        dLp[i][j][2] = damage_broken[i][j] * xdLp[0][j];
    J2_beta_eq[i][2] = xJ2_beta_eq[0];
    J2_alpha[i][2] = xJ2_alpha[0];
    J2_dlambda[i] = xJ2_dlambda[0];

    // free the temporary arrays
    free(temporary_nb);
    free(xJ2_alpha);
    free(xJ2_dlambda);
    free(xJ2_beta_eq);
    freeDouble2D(xdLp, nb[ii] + 1);
}

/* Elastoplastic material with a mixed linear hardening law */
void computeBondForceJ2mixedLinear3D(int ii)
{
    // create a temporary list to store particle ii and all of its neighbors ID
    int *temporary_nb = allocInt1D(nb[ii] + 1, ii);
    int s = 1;
    for (int k = 1; k < nneighbors + 1; k++)
    {
        if (damage_broken[ii][k - 1] > EPS && neighbors[ii][k - 1] != -1)
            temporary_nb[s++] = neighbors[ii][k - 1];
    }

    // allocate temporary variables to store state variables in previous time step
    double *xJ2_alpha = allocDouble1D(nb[ii] + 1, 0.0);           /* accumulated equivalent plasticity strain */
    double **xdLp = allocDouble2D(nb[ii] + 1, nneighbors, 0.0);   /* plastic bond stretch */
    double **xJ2_beta = allocDouble2D(nb[ii] + 1, 2 * NDIM, 0.0); /* back stress */
    double *xJ2_dlambda = allocDouble1D(nb[ii] + 1, 0.0);         /* plasticity multiplier */

    // store the previous step state variables (history dependent) for the particles in above list
    for (int k = 0; k < nb[ii] + 1; k++)
    {
        int i = temporary_nb[k]; // global index for the particle
        for (int j = 0; j < nneighbors; j++)
            xdLp[k][j] = dLp[i][j][0];
        for (int j = 0; j < 2 * NDIM; j++)
            xJ2_beta[k][j] = J2_beta[i][j][0];
        xJ2_alpha[k] = J2_alpha[i][0];
    }

    // initialize the dilatational change for the above particle list
    for (int k = 0; k < nb[ii] + 1; k++)
    {
        int i = temporary_nb[k]; // global index for the particle
        dL_total[i][0] = 0, TdL_total[i][0] = 0;
        dL_total[i][1] = 0, TdL_total[i][1] = 0;

        for (int j = 0; j < nb_initial[i]; j++)
        {
            double dis = sqrt(pow(xyz[i][0] - xyz[neighbors[i][j]][0], 2) +
                              pow(xyz[i][1] - xyz[neighbors[i][j]][1], 2) +
                              pow(xyz[i][2] - xyz[neighbors[i][j]][2], 2));
            dL[i][j] = dis - distance_initial[i][j];
            dL[i][j] -= xdLp[k][j]; // elastic bond stretch
            dL[i][j] *= damage_broken[i][j];
            dL_total[i][nsign[i][j]] += dL[i][j];
            TdL_total[i][nsign[i][j]] += Tv[i][j] * dL[i][j];
            csx[i][j] = (xyz[i][0] - xyz[neighbors[i][j]][0]) / dis;
            csy[i][j] = (xyz[i][1] - xyz[neighbors[i][j]][1]) / dis;
            csz[i][j] = (xyz[i][2] - xyz[neighbors[i][j]][2]) / dis;
        }
    }

    // compute the trial bond force/stress
    for (int k = 0; k < nb[ii] + 1; k++)
    {
        // temporary variables
        double stress_local[2 * NDIM] = {0.0};    /* local stress tensor */
        double dplstrain_local[2 * NDIM] = {0.0}; /* delta plastic strain */

        int i = temporary_nb[k];

        for (int j = 0; j < nb_initial[i]; j++)
        {
            double Fij = 2.0 * Kn[i][j] * dL[i][j] + TdL_total[i][nsign[i][j]] + Tv[i][j] * dL_total[i][nsign[i][j]];
            Fij *= damage_w[i][j];
            // Fij *= (1.0 - damage_D[i][j][0]);

            // compute local stress tensor
            // stress_local[0] += 0.5 / particle_volume * distance_initial[i][j] * Fij * csx[i][j] * csx[i][j] * nneighbors / nb[i];
            // stress_local[1] += 0.5 / particle_volume * distance_initial[i][j] * Fij * csy[i][j] * csy[i][j] * nneighbors / nb[i];
            // stress_local[2] += 0.5 / particle_volume * distance_initial[i][j] * Fij * csz[i][j] * csz[i][j] * nneighbors / nb[i];
            // stress_local[3] += 0.5 / particle_volume * distance_initial[i][j] * Fij * csy[i][j] * csz[i][j] * nneighbors / nb[i];
            // stress_local[4] += 0.5 / particle_volume * distance_initial[i][j] * Fij * csx[i][j] * csz[i][j] * nneighbors / nb[i];
            // stress_local[5] += 0.5 / particle_volume * distance_initial[i][j] * Fij * csx[i][j] * csy[i][j] * nneighbors / nb[i];

            /* check if there are any opposite bonds, if yes then 1 */
            double opp_flag = 1.0;
            if (nb[i] == nneighbors)
                opp_flag = 0.5; // there are opposite bond
            else
            {
                for (int m = 0; m < nb_initial[i]; m++)
                {
                    if (fabs(csx_initial[i][m] + csx_initial[i][j]) < EPS &&
                        fabs(csy_initial[i][m] + csy_initial[i][j]) < EPS &&
                        fabs(csz_initial[i][m] + csz_initial[i][j]) < EPS)
                    {
                        if (damage_broken[i][m] <= EPS) // this is a broken bond
                            opp_flag = 1.0;
                        else
                            opp_flag = 0.5; // there are opposite bond
                        break;
                    }
                }
            }

            // compute local stress tensor
            stress_local[0] += opp_flag / particle_volume * distance_initial[i][j] * Fij * csx[i][j] * csx[i][j];
            stress_local[1] += opp_flag / particle_volume * distance_initial[i][j] * Fij * csy[i][j] * csy[i][j];
            stress_local[2] += opp_flag / particle_volume * distance_initial[i][j] * Fij * csz[i][j] * csz[i][j];
            stress_local[3] += opp_flag / particle_volume * distance_initial[i][j] * Fij * csy[i][j] * csz[i][j];
            stress_local[4] += opp_flag / particle_volume * distance_initial[i][j] * Fij * csx[i][j] * csz[i][j];
            stress_local[5] += opp_flag / particle_volume * distance_initial[i][j] * Fij * csx[i][j] * csy[i][j];
        }

        /* update stress tensor to be trial devitoric stress tensor */
        double temp = 1.0 / 3.0 * (stress_local[0] + stress_local[1] + stress_local[2]);
        for (int j = 0; j < NDIM; j++)
            stress_local[j] -= temp;

        /* substract trial stress tensor with the back stress tensor */
        for (int j = 0; j < 2 * NDIM; j++)
            stress_local[j] -= xJ2_beta[k][j];

        /* von Mises equivalent stress */
        double sigma_eq = 0.0;
        for (int j = 0; j < 2 * NDIM; j++)
        {
            if (j < NDIM)
                sigma_eq += stress_local[j] * stress_local[j]; // s11, s22, s33
            else
                sigma_eq += 2.0 * stress_local[j] * stress_local[j]; // s23, s13, s12
        }
        sigma_eq = sqrt(3.0 / 2.0 * sigma_eq);

        /* test the trial yield function */
        double yield_func = sigma_eq - (sigmay[i] + (1.0 - J2_xi) * J2_H * xJ2_alpha[k]);
        if (yield_func > 0.0)
        {
            pl_flag[i] = 1;
            xJ2_dlambda[k] = yield_func / (3 * Ce[type[i]][2] + J2_H);
        }

        xJ2_alpha[k] += xJ2_dlambda[k];

        /* incremental plastic strain tensor */
        for (int j = 0; j < 2 * NDIM; j++)
        {
            if (fabs(sigma_eq) > EPS)
            {
                dplstrain_local[j] = xJ2_dlambda[k] * 1.5 * stress_local[j] / sigma_eq;
                xJ2_beta[k][j] += 2. / 3. * J2_xi * J2_H * dplstrain_local[j];
            }
        }

        /* incremental plastic bond stretch */
        for (int j = 0; j < nb_initial[i]; j++)
        {
            ddLp[i][j] = distance_initial[i][j] * (dplstrain_local[0] * csx[i][j] * csx[i][j] +
                                                   dplstrain_local[1] * csy[i][j] * csy[i][j] +
                                                   dplstrain_local[2] * csz[i][j] * csz[i][j] +
                                                   2 * dplstrain_local[3] * csy[i][j] * csz[i][j] +
                                                   2 * dplstrain_local[4] * csx[i][j] * csz[i][j] +
                                                   2 * dplstrain_local[5] * csx[i][j] * csy[i][j]);
            ddLp[i][j] *= damage_broken[i][j];
            xdLp[k][j] += ddLp[i][j];
        }
    }

    // update the dilatation terms
    for (int k = 0; k < nb[ii] + 1; k++)
    {
        int i = temporary_nb[k];
        dL_total[i][0] = 0, TdL_total[i][0] = 0;
        dL_total[i][1] = 0, TdL_total[i][1] = 0;

        for (int j = 0; j < nb_initial[i]; j++)
        {
            double dis = sqrt(pow(xyz[i][0] - xyz[neighbors[i][j]][0], 2) +
                              pow(xyz[i][1] - xyz[neighbors[i][j]][1], 2) +
                              pow(xyz[i][2] - xyz[neighbors[i][j]][2], 2));
            dL[i][j] = dis - distance_initial[i][j];
            dL[i][j] -= xdLp[k][j]; // elastic bond stretch
            dL[i][j] *= damage_broken[i][j];
            dL_total[i][nsign[i][j]] += dL[i][j];
            TdL_total[i][nsign[i][j]] += Tv[i][j] * dL[i][j];
            csx[i][j] = (xyz[i][0] - xyz[neighbors[i][j]][0]) / dis;
            csy[i][j] = (xyz[i][1] - xyz[neighbors[i][j]][1]) / dis;
            csz[i][j] = (xyz[i][2] - xyz[neighbors[i][j]][2]) / dis;
        }
    }

    // compute the average elastic bond stretch only for particle i, then compute the bond force, stress and internal force
    int i = ii;
    memset(stress_tensor[i], 0.0, 2 * NDIM * sizeof(double));              // initialize the stress tensor
    Pin[i * NDIM] = 0.0, Pin[i * NDIM + 1] = 0.0, Pin[i * NDIM + 2] = 0.0; // initialize the internal force vector

    for (int j = 0; j < nb_initial[i]; j++)
    {
        for (int jj = 0; jj < nneighbors; jj++)
        {
            if (neighbors[neighbors[i][j]][jj] == i) /* find the opposite particle */
                dL_ave[i][j] = 0.5 * (dL[i][j] + dL[neighbors[i][j]][jj]);
        }
        F[i][j] = 2.0 * Kn[i][j] * dL_ave[i][j] + 0.5 * (TdL_total[i][nsign[i][j]] + TdL_total[neighbors[i][j]][nsign[i][j]]) + 0.5 * Tv[i][j] * (dL_total[i][nsign[i][j]] + dL_total[neighbors[i][j]][nsign[i][j]]);
        F[i][j] *= damage_w[i][j];
        // F[i][j] *= (1.0 - damage_D[i][j][0]);

        /* compute internal forces */
        Pin[i * NDIM] += csx[i][j] * F[i][j];
        Pin[i * NDIM + 1] += csy[i][j] * F[i][j];
        Pin[i * NDIM + 2] += csz[i][j] * F[i][j]; // 0 for 2D
    }

    // store the state variables for the particle itself
    for (int j = 0; j < nneighbors; j++)
        dLp[i][j][2] = damage_broken[i][j] * xdLp[0][j];
    for (int j = 0; j < 2 * NDIM; j++)
        J2_beta[i][j][2] = xJ2_beta[0][j];
    J2_alpha[i][2] = xJ2_alpha[0];
    J2_dlambda[i] = xJ2_dlambda[0];

    // if (J2_dlambda[i] > 0.0)
    //     printf("%d: %.4e, %.4e\n", i, J2_dlambda[i], J2_alpha[i][0]);

    // free the temporary arrays
    free(temporary_nb);
    free(xJ2_alpha);
    free(xJ2_dlambda);
    freeDouble2D(xdLp, nb[ii] + 1);
    freeDouble2D(xJ2_beta, nb[ii] + 1);
}

/* J2 plasticity using nonlinear isotropic hardening, then update damage (uncoupled) */
void computeBondForceJ2nonlinearIso(int ii)
{
    // create a temporary list to store particle ii and all of its neighbors ID
    int *temporary_nb = allocInt1D(nb[ii] + 1, ii);
    int s = 1;
    for (int k = 1; k < nneighbors + 1; k++)
    {
        if (damage_broken[ii][k - 1] > EPS && neighbors[ii][k - 1] != -1)
            temporary_nb[s++] = neighbors[ii][k - 1];
    }

    //#pragma omp parallel for
    // initialize the dilatational change for the above particle list
    for (int k = 0; k < nb[ii] + 1; k++)
    {
        int i = temporary_nb[k]; // global index for the particle
        dL_total[i][0] = 0, TdL_total[i][0] = 0;
        dL_total[i][1] = 0, TdL_total[i][1] = 0;

        for (int j = 0; j < nb_initial[i]; j++)
        {
            double dis = sqrt(pow(xyz[i][0] - xyz[neighbors[i][j]][0], 2) +
                              pow(xyz[i][1] - xyz[neighbors[i][j]][1], 2) +
                              pow(xyz[i][2] - xyz[neighbors[i][j]][2], 2));
            dL[i][j] = dis - distance_initial[i][j];
            dL[i][j] -= dLp[i][j][0]; // elastic bond stretch
            dL[i][j] *= damage_broken[i][j];
            dL_total[i][nsign[i][j]] += dL[i][j];
            TdL_total[i][nsign[i][j]] += Tv[i][j] * dL[i][j];
            csx[i][j] = (xyz[i][0] - xyz[neighbors[i][j]][0]) / dis;
            csy[i][j] = (xyz[i][1] - xyz[neighbors[i][j]][1]) / dis;
            csz[i][j] = (xyz[i][2] - xyz[neighbors[i][j]][2]) / dis;
        }
    }

    double **stress_local = allocDouble2D(nb[ii] + 1, 2 * NDIM, 0.);    /* local stress tensor */
    double **dplstrain_local = allocDouble2D(nb[ii] + 1, 2 * NDIM, 0.); /* delta plastic strain */

    //#pragma omp parallel for
    // compute the trial bond force/stress
    for (int k = 0; k < nb[ii] + 1; k++)
    {
        int i = temporary_nb[k];
        double J2_V = particle_volume * nb[i] / nneighbors; /* modified particle volume */
        for (int j = 0; j < nb_initial[i]; j++)
        {
            F[i][j] = 2.0 * Kn[i][j] * dL[i][j] + TdL_total[i][nsign[i][j]] + Tv[i][j] * dL_total[i][nsign[i][j]];
            F[i][j] *= damage_broken[i][j];
            stress_local[k][0] += 0.5 / J2_V * distance_initial[i][j] * F[i][j] * csx[i][j] * csx[i][j];
            stress_local[k][1] += 0.5 / J2_V * distance_initial[i][j] * F[i][j] * csy[i][j] * csy[i][j];
            stress_local[k][2] += 0.5 / J2_V * distance_initial[i][j] * F[i][j] * csz[i][j] * csz[i][j];
            stress_local[k][3] += 0.5 / J2_V * distance_initial[i][j] * F[i][j] * csy[i][j] * csz[i][j];
            stress_local[k][4] += 0.5 / J2_V * distance_initial[i][j] * F[i][j] * csx[i][j] * csz[i][j];
            stress_local[k][5] += 0.5 / J2_V * distance_initial[i][j] * F[i][j] * csx[i][j] * csy[i][j];
        }

        /* update stress tensor to be trial devitoric stress tensor */
        double temp = 1.0 / 3.0 * (stress_local[k][0] + stress_local[k][1] + stress_local[k][2]);
        for (int j = 0; j < NDIM; j++)
            stress_local[k][j] -= temp;

        /* substract trial stress tensor with the back stress tensor */
        for (int j = 0; j < 2 * NDIM; j++)
            stress_local[k][j] -= J2_beta[i][j][0];

        /* von Mises equivalent stress */
        double sigma_eq = 0.0;
        for (int j = 0; j < 2 * NDIM; j++)
        {
            if (j < NDIM)
                sigma_eq += stress_local[k][j] * stress_local[k][j]; // s11, s22, s33
            else
                sigma_eq += 2.0 * stress_local[k][j] * stress_local[k][j]; // s23, s13, s12
        }
        sigma_eq = sqrt(3.0 / 2.0 * sigma_eq);

        /* test the trial yield function */
        double dlambda = 0.0;
        double yield_func = sigma_eq - SY(J2_alpha[i][0]);
        if (yield_func > 0.0)
        {
            /* solve the nonlinear equation to obtain lambda, using bisection, 0 < dlambda < 1 */
            double a = 0.0, b = 1.0;
            double ya = yield_func, yb = -1.0;
            while ((b - a) > TOLITER)
            {
                /* note the yield function below is at the current timestep */
                dlambda = (a + b) / 2.0;
                yield_func = sigma_eq - 1.5 * dlambda * (2.0 * Ce[type[i]][2] + J2_C) - SY(J2_alpha[i][0] + dlambda);

                /* change the bisection */
                if (yield_func * ya < 0.0)
                {
                    b = dlambda;
                    yb = yield_func;
                }
                else
                {
                    a = dlambda;
                    ya = yield_func;
                }
            }
        }
        J2_dlambda[i] = dlambda;
        J2_alpha[i][0] += J2_dlambda[i]; /* accumulated equivalent plasticity strain */

        /* incremental plastic strain tensor */
        for (int j = 0; j < 2 * NDIM; j++)
        {
            dplstrain_local[k][j] = dlambda * 1.5 * stress_local[k][j] / sigma_eq;
            J2_beta[i][j][0] += J2_C * dplstrain_local[k][j];
        }

        /* incremental plastic bond stretch */
        for (int j = 0; j < nb_initial[i]; j++)
        {
            ddLp[i][j] = distance_initial[i][j] * (dplstrain_local[k][0] * csx[i][j] * csx[i][j] +
                                                   dplstrain_local[k][1] * csy[i][j] * csy[i][j] +
                                                   dplstrain_local[k][2] * csz[i][j] * csz[i][j] +
                                                   2 * dplstrain_local[k][3] * csy[i][j] * csz[i][j] +
                                                   2 * dplstrain_local[k][4] * csx[i][j] * csz[i][j] +
                                                   2 * dplstrain_local[k][5] * csx[i][j] * csy[i][j]);
            dLp[i][j][0] += ddLp[i][j];
        }
    }

    //#pragma omp parallel for
    // update the dilatation terms
    for (int k = 0; k < nb[ii] + 1; k++)
    {
        int i = temporary_nb[k];
        dL_total[i][0] = 0, TdL_total[i][0] = 0;
        dL_total[i][1] = 0, TdL_total[i][1] = 0;

        for (int j = 0; j < nb_initial[i]; j++)
        {
            double dis = sqrt(pow(xyz[i][0] - xyz[neighbors[i][j]][0], 2) +
                              pow(xyz[i][1] - xyz[neighbors[i][j]][1], 2) +
                              pow(xyz[i][2] - xyz[neighbors[i][j]][2], 2));
            dL[i][j] = dis - distance_initial[i][j];
            dL[i][j] -= dLp[i][j][0]; // elastic bond stretch
            dL[i][j] *= damage_broken[i][j];
            dL_total[i][nsign[i][j]] += dL[i][j];
            TdL_total[i][nsign[i][j]] += Tv[i][j] * dL[i][j];
        }
    }

    // compute the average elastic bond stretch only for particle i, then compute the bond force, stress and internal force
    int i = ii;
    memset(stress_tensor[i], 0.0, 2 * NDIM * sizeof(double));              // initialize the stress tensor
    Pin[i * NDIM] = 0.0, Pin[i * NDIM + 1] = 0.0, Pin[i * NDIM + 2] = 0.0; // initialize the internal force vector

    for (int j = 0; j < nb_initial[i]; j++)
    {
        for (int jj = 0; jj < nneighbors; jj++)
        {
            if (neighbors[neighbors[i][j]][jj] == i) /* find the opposite particle */
                dL_ave[i][j] = 0.5 * (dL[i][j] + dL[neighbors[i][j]][jj]);
        }

        F[i][j] = 2.0 * Kn[i][j] * dL_ave[i][j] + 0.5 * (TdL_total[i][nsign[i][j]] + TdL_total[neighbors[i][j]][nsign[i][j]]) + 0.5 * Tv[i][j] * (dL_total[i][nsign[i][j]] + dL_total[neighbors[i][j]][nsign[i][j]]);
        F[i][j] *= damage_w[i][j];
        // F[i][j] *= (1.0 - damage_D[i][j][0]);

        /* compute internal forces */
        Pin[i * NDIM] += csx[i][j] * F[i][j];
        Pin[i * NDIM + 1] += csy[i][j] * F[i][j];
        Pin[i * NDIM + 2] += csz[i][j] * F[i][j]; // 0 for 2D
    }

    // finish the loop for particle ii
    free(temporary_nb);
    freeDouble2D(stress_local, nb[ii] + 1);
    freeDouble2D(dplstrain_local, nb[ii] + 1);
}

/* Crystalline material with plasticity, using Miehe's algorithm */
void computeBondForceCPMiehe(int ii)
{
    // create a temporary list to store particle ii and all of its neighbors ID
    int *temporary_nb = allocInt1D(nb[ii] + 1, ii);
    int s = 1;
    for (int k = 1; k < nneighbors + 1; k++)
    {
        if (damage_broken[ii][k - 1] > EPS && neighbors[ii][k - 1] != -1)
            temporary_nb[s++] = neighbors[ii][k - 1];
    }

    // store state variables of previous time step
    double *xcp_A = allocDouble1D(nb[ii] + 1, 0.0);                   /* accumulated plastic slip */
    double **xcp_A_single = allocDouble2D(nb[ii] + 1, nslipSys, 0.0); /* accumulated plastic slip for individual slip systems */
    double **xdL = allocDouble2D(nb[ii] + 1, nneighbors, 0.0);        /* temporary bond stretch */
    double **xdL_total = allocDouble2D(nb[ii] + 1, 2, 0);
    double **xTdL_total = allocDouble2D(nb[ii] + 1, 2, 0);
    double **xdLp = allocDouble2D(nb[ii] + 1, nneighbors, 0.0); /* plastic bond stretch */
    double **xcp_gy = allocDouble2D(nb[ii] + 1, nslipSys, 0.0); /* yield stress */

    // temporary variables
    double *cp_gamma = allocDouble1D(nslipSys, 0.0);        /* plastic slip */
    double *cp_r = allocDouble1D(nslipSys, 0.0);            /* residual of RSS */
    double *cp_rrhs = allocDouble1D(nslipSys, 0.0);         /* residual of RSS, RHS calculation */
    double *cp_D = allocDouble1D(nslipSys * nslipSys, 0.0); /* Jacobian matrix for relative slip calculation */

    /* settings for solving linear systems of plastic slip */
    MKL_INT n = nslipSys, nrhs = 1, lda = nslipSys, ldb = 1, info;
    MKL_INT *ipiv = (MKL_INT *)malloc(sizeof(MKL_INT) * nslipSys);

    // store the (history dependent) state variables in previous step for the particles in above list
    for (int k = 0; k < nb[ii] + 1; k++)
    {
        int i = temporary_nb[k]; // global index for the particle
        for (int j = 0; j < nneighbors; j++)
            xdLp[k][j] = dLp[i][j][0];
        xcp_A[k] = cp_A[i][0];
        for (int j = 0; j < nslipSys; j++)
        {
            xcp_gy[k][j] = cp_gy[i][j][0];
            xcp_A_single[k][j] = cp_A_single[i][j][0];
        }
    }

    // initialize the dilatational change for the above particle list
    for (int k = 0; k < nb[ii] + 1; k++)
    {
        int i = temporary_nb[k]; // global index for the particle
        dL_total[i][0] = 0, TdL_total[i][0] = 0;
        dL_total[i][1] = 0, TdL_total[i][1] = 0;

        for (int j = 0; j < nb_initial[i]; j++)
        {
            double dis = sqrt(pow(xyz[i][0] - xyz[neighbors[i][j]][0], 2) +
                              pow(xyz[i][1] - xyz[neighbors[i][j]][1], 2) +
                              pow(xyz[i][2] - xyz[neighbors[i][j]][2], 2));
            dL[i][j] = dis - distance_initial[i][j];
            dL[i][j] -= xdLp[k][j]; // elastic bond stretch
            dL[i][j] *= damage_broken[i][j];
            dL_total[i][nsign[i][j]] += dL[i][j];
            TdL_total[i][nsign[i][j]] += Tv[i][j] * dL[i][j];
            csx[i][j] = (xyz[i][0] - xyz[neighbors[i][j]][0]) / dis;
            csy[i][j] = (xyz[i][1] - xyz[neighbors[i][j]][1]) / dis;
            csz[i][j] = (xyz[i][2] - xyz[neighbors[i][j]][2]) / dis;

            // store the geometric variables
            xdL[k][j] = dL[i][j];
            xdL_total[k][nsign[i][j]] = dL_total[i][nsign[i][j]];
            xTdL_total[k][nsign[i][j]] = TdL_total[i][nsign[i][j]];
        }
    }

    // compute the trial bond force/stress, and then do the plastic corrector
    for (int k = 0; k < nb[ii] + 1; k++)
    {
        // temporary variables
        double stress_local[2 * NDIM] = {0.0}; /* local stress tensor */
        double yield_func[MAXSLIPSYS] = {0.0}; /* yield function */

        int i = temporary_nb[k];
        if (state_v[i] == 1)
        { // i's state has been updated previously, we just collect them
            for (int j = 0; j < nb_initial[i]; j++)
                xdLp[k][j] += ddLp[i][j];
            xcp_A[k] += cp_dA[i];
            for (int j = 0; j < nslipSys; j++)
            {
                xcp_gy[k][j] += cp_dgy[i][j];
                xcp_A_single[k][j] += cp_dA_single[i][j];
            }
            continue;
        }
        else
            state_v[i] = 1;

        for (int j = 0; j < nb_initial[i]; j++)
        {
            double Fij = 2.0 * Kn[i][j] * xdL[k][j] + xTdL_total[k][nsign[i][j]] + Tv[i][j] * xdL_total[k][nsign[i][j]];
            Fij *= damage_w[i][j];
            // Fij *= (1.0 - damage_D[i][j][0]);

            /* check if there are any opposite bonds, if no then 1 */
            double opp_flag = 1.0;
            if (nb[i] == nneighbors)
                opp_flag = 0.5; // there are opposite bond
            else
            {
                for (int m = 0; m < nb_initial[i]; m++)
                {
                    if (fabs(csx_initial[i][m] + csx_initial[i][j]) < EPS &&
                        fabs(csy_initial[i][m] + csy_initial[i][j]) < EPS &&
                        fabs(csz_initial[i][m] + csz_initial[i][j]) < EPS)
                    {
                        if (damage_broken[i][m] <= EPS) // this is a broken bond
                            opp_flag = 1.0;
                        else
                            opp_flag = 0.5; // there are opposite bond
                        break;
                    }
                }
            }

            // compute local stress tensor
            stress_local[0] += opp_flag / particle_volume * distance_initial[i][j] * Fij * csx[i][j] * csx[i][j];
            stress_local[1] += opp_flag / particle_volume * distance_initial[i][j] * Fij * csy[i][j] * csy[i][j];
            stress_local[2] += opp_flag / particle_volume * distance_initial[i][j] * Fij * csz[i][j] * csz[i][j];
            stress_local[3] += opp_flag / particle_volume * distance_initial[i][j] * Fij * csy[i][j] * csz[i][j];
            stress_local[4] += opp_flag / particle_volume * distance_initial[i][j] * Fij * csx[i][j] * csz[i][j];
            stress_local[5] += opp_flag / particle_volume * distance_initial[i][j] * Fij * csx[i][j] * csy[i][j];
        }

        // compute trial resolved shear stress (RSS) and yield function, find max value
        double temp_max = 0.0;
        for (int m = 0; m < nslipSys; m++)
        {
            cp_RSS[i][m] = stress_local[0] * schmid_tensor[m][0] + stress_local[1] * schmid_tensor[m][1] +
                           stress_local[2] * schmid_tensor[m][2] + stress_local[3] * schmid_tensor[m][3] +
                           stress_local[4] * schmid_tensor[m][4] + stress_local[5] * schmid_tensor[m][5];
            yield_func[m] = cp_RSS[i][m] - cp_gy[i][m][0]; // previous yield point (haven't update yet)

            // find the maximum yield function
            if (yield_func[m] > temp_max)
                temp_max = yield_func[m];
        }

        if (temp_max <= EPS)
        { // elastic step
            for (int j = 0; j < nb_initial[i]; j++)
                ddLp[i][j] = 0.0;
            cp_dA[i] = 0.0;
            for (int m = 0; m < nslipSys; m++)
            {
                cp_Jact[i][m] = 0; // active slip system set is empty
                cp_dgy[i][m] = 0.0;
                cp_dA_single[i][m] = 0.0;
            }
        }
        else
        { // plastic slip occurs, determine the active slip systems below
            int niter_outer = 0;
            pl_flag[i] = 1;
            memset(cp_Jact[i], 0, nslipSys * sizeof(int));

            // printf("%d; ", i);
        label_outer:
            niter_outer++;
            // if (niter_outer == 2)
            //     memset(cp_Jact[i], 0, nslipSys * sizeof(int));

            // some definitions of temporary variables
            double norm_r = 1.0; // norm of residual
            memset(cp_gamma, 0.0, nslipSys * sizeof(double));
            memset(cp_r, 0.0, nslipSys * sizeof(double));
            memset(cp_rrhs, 0.0, nslipSys * sizeof(double));
            memset(cp_D, 0.0, (int)nslipSys * nslipSys * sizeof(double));

            int niter_inner = 0;
            do
            {
                niter_inner++;
                double dplstrain_local[2 * NDIM] = {0.0}; /* incremental plastic strain, from n to n+1 step */

                // note in this context, dplstrain is e11, e22, e33, 2e23, 2e13, 2e12
                for (int s = 0; s < nslipSys; s++)
                {
                    dplstrain_local[0] += cp_Jact[i][s] * cp_gamma[s] * schmid_tensor[s][0];
                    dplstrain_local[1] += cp_Jact[i][s] * cp_gamma[s] * schmid_tensor[s][1];
                    dplstrain_local[2] += cp_Jact[i][s] * cp_gamma[s] * schmid_tensor[s][2];
                    dplstrain_local[3] += cp_Jact[i][s] * cp_gamma[s] * schmid_tensor[s][3];
                    dplstrain_local[4] += cp_Jact[i][s] * cp_gamma[s] * schmid_tensor[s][4];
                    dplstrain_local[5] += cp_Jact[i][s] * cp_gamma[s] * schmid_tensor[s][5];
                }

                /* compute the updated elastic bond stretch */
                xdL_total[k][0] = 0, xTdL_total[k][0] = 0;
                xdL_total[k][1] = 0, xTdL_total[k][1] = 0;
                for (int j = 0; j < nb_initial[i]; j++)
                {
                    ddLp[i][j] = distance_initial[i][j] * (dplstrain_local[0] * csx[i][j] * csx[i][j] +
                                                           dplstrain_local[1] * csy[i][j] * csy[i][j] +
                                                           dplstrain_local[2] * csz[i][j] * csz[i][j] +
                                                           dplstrain_local[3] * csy[i][j] * csz[i][j] +
                                                           dplstrain_local[4] * csx[i][j] * csz[i][j] +
                                                           dplstrain_local[5] * csx[i][j] * csy[i][j]);
                    ddLp[i][j] *= damage_broken[i][j];

                    xdL[k][j] = dL[i][j] - ddLp[i][j]; // note dL is trial elastic deformation
                    xdL_total[k][nsign[i][j]] += xdL[k][j];
                    xTdL_total[k][nsign[i][j]] += Tv[i][j] * xdL[k][j];
                }

                // compute stress
                memset(stress_local, 0.0, 2 * NDIM * sizeof(double));
                for (int j = 0; j < nb_initial[i]; j++)
                {
                    // note xdL is modified elastic deformation
                    double Fij = 2.0 * Kn[i][j] * xdL[k][j] + xTdL_total[k][nsign[i][j]] + Tv[i][j] * xdL_total[k][nsign[i][j]];
                    Fij *= damage_w[i][j];
                    // Fij *= (1.0 - damage_D[i][j][0]);

                    /* check if there are any opposite bonds, if yes then 1 */
                    double opp_flag = 1.0;
                    if (nb[i] == nneighbors)
                        opp_flag = 0.5; // there are opposite bond
                    else
                    {
                        for (int m = 0; m < nb_initial[i]; m++)
                        {
                            if (fabs(csx_initial[i][m] + csx_initial[i][j]) < EPS &&
                                fabs(csy_initial[i][m] + csy_initial[i][j]) < EPS &&
                                fabs(csz_initial[i][m] + csz_initial[i][j]) < EPS)
                            {
                                if (damage_broken[i][m] <= EPS) // this is a broken bond
                                    opp_flag = 1.0;
                                else
                                    opp_flag = 0.5; // there are opposite bond
                                break;
                            }
                        }
                    }

                    // compute local stress tensor
                    stress_local[0] += opp_flag / particle_volume * distance_initial[i][j] * Fij * csx[i][j] * csx[i][j];
                    stress_local[1] += opp_flag / particle_volume * distance_initial[i][j] * Fij * csy[i][j] * csy[i][j];
                    stress_local[2] += opp_flag / particle_volume * distance_initial[i][j] * Fij * csz[i][j] * csz[i][j];
                    stress_local[3] += opp_flag / particle_volume * distance_initial[i][j] * Fij * csy[i][j] * csz[i][j];
                    stress_local[4] += opp_flag / particle_volume * distance_initial[i][j] * Fij * csx[i][j] * csz[i][j];
                    stress_local[5] += opp_flag / particle_volume * distance_initial[i][j] * Fij * csx[i][j] * csy[i][j];
                }

                // accumulated plastic slip
                cp_dA[i] = 0.0;
                for (int s = 0; s < nslipSys; s++)
                {
                    cp_dA[i] += cp_gamma[s];
                }
                xcp_A[k] = cp_A[i][0] + cp_dA[i];

                // hardening, h_hat, h_hatp
                double h_hat = cp_h0 / pow(cosh(cp_h0 * xcp_A[k] / (cp_taus[0] - cp_tau0[0])), 2.0);
                double h_hatp = -2.0 * cp_h0 * cp_h0 / (cp_taus[0] - cp_tau0[0]) * tanh(cp_h0 * xcp_A[k] / (cp_taus[0] - cp_tau0[0])) * h_hat;

                // renew the yield stress
                for (int a = 0; a < nslipSys; a++)
                {
                    double term1 = 0.0;
                    for (int b = 0; b < nslipSys; b++)
                    {
                        double hab = 0.0; // self or latent hardening
                        if (a == b)
                            hab = h_hat;
                        else
                            hab = cp_q * h_hat;
                        term1 += cp_Jact[i][b] * hab * cp_gamma[b];
                    }
                    cp_dgy[i][a] = cp_Jact[i][a] * term1;
                    xcp_gy[k][a] = cp_gy[i][a][0] + cp_dgy[i][a];
                }

                // compute resolved shear stress (RSS) and residual, right hand side
                for (int m = 0; m < nslipSys; m++)
                {
                    // compute RSS
                    double term1 = pow(1. + cp_gamma[m] * cp_eta / dtime, 1. / cp_p);
                    cp_RSS[i][m] = stress_local[0] * schmid_tensor[m][0] +
                                   stress_local[1] * schmid_tensor[m][1] +
                                   stress_local[2] * schmid_tensor[m][2] +
                                   stress_local[3] * schmid_tensor[m][3] +
                                   stress_local[4] * schmid_tensor[m][4] +
                                   stress_local[5] * schmid_tensor[m][5];

                    // residual
                    cp_r[m] = cp_Jact[i][m] * (cp_RSS[i][m] - xcp_gy[k][m] * term1);
                    cp_rrhs[m] = cp_r[m]; // use it in solving the linear system procedure
                    // if (ii == 0 && i == 0)
                    //     printf("%.3e; ", cp_rrhs[m]);
                }
                // if (ii == 0 && i == 0)
                //     printf("\n");

                // update the plastic shear
                /* cp_gamma calculation: left-hand-side Jacobian matrix */
                for (int m = 0; m < nslipSys; m++)
                {
                    for (int n = 0; n < nslipSys; n++)
                    {
                        if (cp_Jact[i][m] == 1 && cp_Jact[i][n] == 1)
                        {
                            // compute the h_star for m-n slip system (both should active)
                            double h_star = 0.0;
                            for (int d = 0; d < nslipSys; d++)
                            {
                                double hd = 0.0;
                                if (m == d && n == d)
                                    hd = h_hat + h_hatp * cp_gamma[d];
                                else if (m == d && n != d)
                                    hd = h_hatp * cp_gamma[d];
                                else if (m != d && n == d)
                                    hd = cp_q * (h_hat + h_hatp * cp_gamma[d]);
                                else if (m != d && n != d)
                                    hd = cp_q * h_hatp * cp_gamma[d];
                                h_star += cp_Jact[i][d] * hd;
                            }

                            /*
                             * cp_Cab is computed only once, before loading applies
                             * for the Dab matrix:
                             * if diagonal, the elements are (Cab + term1 + term2)
                             * if off-diagonal, the elements are (Cab + term2)
                             */
                            double term1 = xcp_gy[k][m] * (cp_eta / cp_p / dtime * pow(1. + cp_eta * cp_gamma[m] / dtime, (1. - cp_p) / cp_p));
                            double term2 = h_star * pow(1. + cp_eta * cp_gamma[m] / dtime, (1. / cp_p));
                            if (m == n)
                                cp_D[m * nslipSys + n] = cp_Cab[i][m * nslipSys + n] + term1 + term2;
                            else
                                cp_D[m * nslipSys + n] = cp_Cab[i][m * nslipSys + n] + term2;
                        }
                        else if (m == n)
                            cp_D[m * nslipSys + n] = 1.0;

                        //if (i == 288 && m == n)
                        //    printf("%.4e; ", cp_D[m * nslipSys + n]);
                    }
                }
                //printf("\n");

                /* compute incremental cp_gamma i.e. plastic slip */
                info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, cp_D, lda, ipiv, cp_rrhs, ldb);

                /* Check for the singularity */
                if (info > 0)
                {
                    printf("The diagonal element of the triangular factor of Jacobian matrix for cp_gamma \n");
                    printf("U(%i,%i) is zero, so that it is singular.\n", info, info);
                    exit(1);
                }

                /* update the cp_gamma, i.e. plastic slip, for active slip systems */
                for (int m = 0; m < nslipSys; m++)
                    cp_gamma[m] += cp_Jact[i][m] * cp_rrhs[m];

                norm_r = cblas_dnrm2(nslipSys, cp_r, 1);

                // if (i == 288)
                // {
                //     for (int m = 0; m < nslipSys; m++)
                //         printf("%.4e ", cp_D[m * nslipSys + m]);
                //     // for (int m = 0; m < nslipSys; m++)
                //     //     printf("%.4e ", cp_Cab[i][m * nslipSys + m]);
                //     // for (int m = 0; m < nslipSys; m++)
                //     //     printf("%.4e ", cp_Jact[i][m] * cp_D[m * nslipSys + m]);
                //      printf("\n");
                //      for (int m = 0; m < nslipSys; m++)
                //          printf("%.4e ", cp_r[m]);
                //      printf("\n");
                //      for (int m = 0; m < nslipSys; m++)
                //          printf("%.4e ", cp_gamma[m]);
                //     printf("\n");
                //     printf("%.4e %d\n", norm_r, niter_inner);

                // }

            } while (norm_r > TOLITER && niter_inner < MAXSMALL);

            /* update the active slip systems */
            int J_flag = 0;

            /* drop the minimum loaded slip system which have negative plastic slip */
            int minIndex = -1;     // index of minimum value yield function
            double minYield = 0.0; // min value of yield function
            for (int m = 0; m < nslipSys; m++)
            {
                yield_func[m] = cp_RSS[i][m] - xcp_gy[k][m];
                if (cp_Jact[i][m] == 1 && cp_gamma[m] <= 0.0)
                {
                    if (yield_func[m] < minYield)
                    {
                        minYield = yield_func[m];
                        minIndex = m;
                    }
                }
            }
            if (minIndex != -1)
            {
                cp_Jact[i][minIndex] = 0;
                J_flag = 1;
            }
            if (J_flag == 1)
            {
                // printf("%.4e; ", yield_func[minIndex]);
                goto label_outer;
            }

            /* add the maximum loaded slip system */
            int maxIndex = -1;     // index of maximum value yield function
            double maxYield = 0.0; // max value of yield function
            for (int m = 0; m < nslipSys; m++)
            {
                if (cp_Jact[i][m] == 0 && yield_func[m] > 0.0)
                {
                    if (yield_func[m] > maxYield)
                    {
                        maxYield = yield_func[m];
                        maxIndex = m;
                    }
                }
            }
            if (maxIndex != -1)
            {
                cp_Jact[i][maxIndex] = 1;
                J_flag = 2;
            }
            if (J_flag == 2 && niter_outer < cp_maxloop)
            {
                // printf("%.4e; ", yield_func[maxIndex]);
                goto label_outer;
            }
            else if (niter_outer >= cp_maxloop)
                goto label_outside;
        }

    label_outside:
        /* update the state variable arrays */
        for (int j = 0; j < nb_initial[i]; j++)
            xdLp[k][j] += ddLp[i][j];

        // update the plastic slip
        for (int s = 0; s < nslipSys; s++)
        {
            cp_dA_single[i][s] = cp_Jact[i][s] * cp_gamma[s];
            xcp_A_single[k][s] += cp_dA_single[i][s];
        }
    }

    // update the dilatation terms using the dLp obtained ubove
    for (int k = 0; k < nb[ii] + 1; k++)
    {
        int i = temporary_nb[k];
        dL_total[i][0] = 0, TdL_total[i][0] = 0;
        dL_total[i][1] = 0, TdL_total[i][1] = 0;

        for (int j = 0; j < nb_initial[i]; j++)
        {
            double dis = sqrt(pow(xyz[i][0] - xyz[neighbors[i][j]][0], 2) +
                              pow(xyz[i][1] - xyz[neighbors[i][j]][1], 2) +
                              pow(xyz[i][2] - xyz[neighbors[i][j]][2], 2));
            dL[i][j] = dis - distance_initial[i][j];
            dL[i][j] -= xdLp[k][j]; // elastic bond stretch
            dL[i][j] *= damage_broken[i][j];
            dL_total[i][nsign[i][j]] += dL[i][j];
            TdL_total[i][nsign[i][j]] += Tv[i][j] * dL[i][j];
            csx[i][j] = (xyz[i][0] - xyz[neighbors[i][j]][0]) / dis;
            csy[i][j] = (xyz[i][1] - xyz[neighbors[i][j]][1]) / dis;
            csz[i][j] = (xyz[i][2] - xyz[neighbors[i][j]][2]) / dis;
        }
    }

    // compute the average elastic bond stretch of particle i, then the bond force, stress and internal force
    int i = ii;
    // memset(stress_tensor[i], 0.0, 2 * NDIM * sizeof(double));              // initialize the stress tensor
    Pin[i * NDIM] = 0.0, Pin[i * NDIM + 1] = 0.0, Pin[i * NDIM + 2] = 0.0; // initialize the internal force vector

    for (int j = 0; j < nb_initial[i]; j++)
    {
        for (int jj = 0; jj < nneighbors; jj++)
        {
            if (neighbors[neighbors[i][j]][jj] == i) /* find the opposite particle */
                dL_ave[i][j] = 0.5 * (dL[i][j] + dL[neighbors[i][j]][jj]);
        }
        F[i][j] = 2.0 * Kn[i][j] * dL_ave[i][j] + 0.5 * (TdL_total[i][nsign[i][j]] + TdL_total[neighbors[i][j]][nsign[i][j]]) + 0.5 * Tv[i][j] * (dL_total[i][nsign[i][j]] + dL_total[neighbors[i][j]][nsign[i][j]]);
        F[i][j] *= damage_w[i][j];

        /* compute internal forces */
        Pin[i * NDIM] += csx[i][j] * F[i][j];
        Pin[i * NDIM + 1] += csy[i][j] * F[i][j];
        Pin[i * NDIM + 2] += csz[i][j] * F[i][j]; // 0 for 2D
    }

    // if (i == 3000)
    // {
    //     for (int m = 0; m < nslipSys; m++)
    //         printf("%.4e;", xcp_gy[0][m]);
    //     printf("\n");
    // }

    // store the state variables for the particle itself
    for (int j = 0; j < nneighbors; j++)
        dLp[i][j][2] = damage_broken[i][j] * xdLp[0][j];
    for (int m = 0; m < nslipSys; m++)
    {
        cp_gy[i][m][2] = xcp_gy[0][m];
        cp_A_single[i][m][2] = xcp_A_single[0][m];
    }
    cp_A[i][2] = xcp_A[0];

    // free the temporary arrays
    free(xcp_A);
    free(temporary_nb);
    freeDouble2D(xcp_A_single, nb[ii] + 1);
    freeDouble2D(xdL, nb[ii] + 1);
    freeDouble2D(xdLp, nb[ii] + 1);
    freeDouble2D(xdL_total, nb[ii] + 1);
    freeDouble2D(xTdL_total, nb[ii] + 1);
    freeDouble2D(xcp_gy, nb[ii] + 1);

    free(ipiv);
    free(cp_gamma);
    free(cp_r);
    free(cp_rrhs);
    free(cp_D);
}

/* update the damage state after bonds broken */
void updateCrack()
{
    // #pragma omp parallel for
    for (int i = 0; i < nparticle; i++)
    {
        nb[i] = nb_initial[i];
        damage_visual[i] = 0.0;
        Pin[i * NDIM] = 0.0, Pin[i * NDIM + 1] = 0.0, Pin[i * NDIM + 2] = 0.0; // initialize the internal force vector
        for (int j = 0; j < nb_initial[i]; j++)
        {
            if (damage_broken[i][j] <= EPS)
                nb[i] -= 1;

            damage_visual[i] += damage_broken[i][j];

            /* update the internal forces */
            F[i][j] *= damage_w[i][j];
            // F[i][j] *= (1.0 - damage_D[i][j][0]);
            // F[i][j] *= damage_broken[i][j];
            Pin[i * NDIM] += csx[i][j] * F[i][j];
            Pin[i * NDIM + 1] += csy[i][j] * F[i][j];
            Pin[i * NDIM + 2] += csz[i][j] * F[i][j]; // 0 for 2D
        }

        if (nb[i] < 1)
        {
            // fix this particle's position, if all bonds are broken
            fix_index[dim * i] = 0;
            fix_index[dim * i + 1] = 0;
            if (dim == 3)
                fix_index[dim * i + 2] = 0;
        }

        damage_visual[i] = 1 - damage_visual[i] / nb_initial[i];
    }
}

/* break the bond for elastic materials, when bond strain reaches a critical value (we limit the maximum broken number) */
int updateBrittleDamage(const char *dataName, int tstep, int nbreak)
{
    FILE *fpt;
    fpt = fopen(dataName, "a+");
    fprintf(fpt, "TIMESTEP ");
    fprintf(fpt, "%d\n", tstep);

    struct bondStrain b_cr[MAXSMALL * MAXSMALL]; // bonds of which the strain reach the critical value

    // initialization of struct
    for (int i = 0; i < MAXSMALL * MAXSMALL; i++)
    {
        b_cr[i].i_index = -1;
        b_cr[i].j_index = -1;
        b_cr[i].bstrain = 0.0;
    }

    // store bonds that have reached the critical bond strain, if no new crack, k = 0
    int k = 0;
    for (int i = 0; i < nparticle; i++)
    {
        for (int j = 0; j < nb_initial[i]; j++)
        {
            double ave_bstrain = dL[i][j] / distance_initial[i][j]; // note we already make dL to be 0 for broken bonds

            if (ave_bstrain >= critical_bstrain)
            {
                b_cr[k].i_index = i;
                b_cr[k].j_index = j;
                b_cr[k].bstrain = ave_bstrain;
                k++;
            }
        }
    }

    // delete the first nbreak bonds which have largest damage
    int broken_bonds = k;
    if (k > 0 && k <= nbreak)
    {
        for (int i = 0; i < k; i++)
        {
            damage_D[b_cr[i].i_index][b_cr[i].j_index][0] = 1.0;
            damage_w[b_cr[i].i_index][b_cr[i].j_index] = 0.0;
            damage_broken[b_cr[i].i_index][b_cr[i].j_index] = 0.0;
            fprintf(fpt, "%d %d \n", b_cr[i].i_index, neighbors[b_cr[i].i_index][b_cr[i].j_index]);
        }
    }
    else if (k > nbreak)
    {
        // sorting of the struct array, shell sort, from small to large
        int temp_i, temp_j;
        double temp_b;
        for (int r = k / 2; r >= 1; r = r / 2)
        {
            for (int i = r; i < k; ++i)
            {
                temp_i = b_cr[i].i_index;
                temp_j = b_cr[i].j_index;
                temp_b = b_cr[i].bstrain;
                int j = i - r;
                while (j >= 0 && b_cr[j].bstrain > temp_b)
                {
                    b_cr[j + r].bstrain = b_cr[j].bstrain;
                    b_cr[j + r].i_index = b_cr[j].i_index;
                    b_cr[j + r].j_index = b_cr[j].j_index;
                    j = j - r;
                }
                b_cr[j + r].bstrain = temp_b;
                b_cr[j + r].i_index = temp_i;
                b_cr[j + r].j_index = temp_j;
            }
        }

        // // for (int i = 0; i < k; i++)
        // //     printf("%d %d %f\n", b_cr[i].i_index, b_cr[i].j_index, b_cr[i].bstrain);

        // limit the maximum number of broken bonds
        for (int i = k - nbreak; i < k; i++)
        {
            damage_D[b_cr[i].i_index][b_cr[i].j_index][0] = 1.0;
            damage_w[b_cr[i].i_index][b_cr[i].j_index] = 0.0;
            damage_broken[b_cr[i].i_index][b_cr[i].j_index] = 0.0;
            fprintf(fpt, "%d %d \n", b_cr[i].i_index, neighbors[b_cr[i].i_index][b_cr[i].j_index]);
        }
    }

    fclose(fpt);

    return broken_bonds;
}

/* update the damage variables for elasto-plastic materials, using CDM, average bond damage, nonlocal */
int updateDuctileDamageBwiseNonlocal(const char *dataName, int tstep)
{
    // unlike brittle materials, we donnot limit the broken particle number
    FILE *fpt;
    fpt = fopen(dataName, "a+");
    fprintf(fpt, "TIMESTEP ");
    fprintf(fpt, "%d\n", tstep);

#pragma omp parallel for
    // nonlocal damage variable, only counts for neighboring particles
    for (int i = 0; i < nparticle; i++)
    {
        if (damage_nonlocal[i][0] > damage_threshold)
        {
            if (damage_nonlocal[i][0] > 1.0)
                damage_nonlocal[i][0] = 1.0;
            continue;
        }

        double DdotLocal = 0;
        double damage_f = (1.0 + damagec_A * J2_triaxiality[i]);
        if (damage_f > 0.0)
            DdotLocal = J2_dlambda[i] * (1.0 + damagec_A * J2_triaxiality[i]);

        double Ddot = DdotLocal * particle_volume;
        double A = particle_volume;
        for (int j = 0; j < nb_initial[i]; j++)
        {
            DdotLocal = 0;
            damage_f = (1.0 + damagec_A * J2_triaxiality[neighbors[i][j]]);
            if (damage_f > 0.0)
                DdotLocal = J2_dlambda[neighbors[i][j]] * (1.0 + damagec_A * J2_triaxiality[neighbors[i][j]]);
            // Ddot += damage_broken[i][j] * DdotLocal * DAM_PHI(distance_initial[i][j]) * particle_volume;
            // A += damage_broken[i][j] * DAM_PHI(distance_initial[i][j]) * particle_volume;
            Ddot += DdotLocal * DAM_PHI(distance_initial[i][j]) * particle_volume;
            A += DAM_PHI(distance_initial[i][j]) * particle_volume;
        }
        if (Ddot > 0.0)
            damage_nonlocal[i][0] += 1.0 / A * Ddot;
    }

    int k = 0; // broken number
    for (int i = 0; i < nparticle; i++)
    {
        for (int j = 0; j < nb_initial[i]; j++)
        {
            if (0.5 * (damage_nonlocal[i][0] + damage_nonlocal[neighbors[i][j]][0]) > damage_threshold)
            {
                // test if already broken
                if (fabs(damage_broken[i][j]) > EPS)
                {
                    damage_broken[i][j] = 0.0;
                    damage_D[i][j][0] = 1.0;

                    fprintf(fpt, "%d %d \n", i, neighbors[i][j]);
                    k++;
                }
            }
        }
    }

    for (int i = 0; i < nparticle; i++)
    {
        for (int j = 0; j < nb_initial[i]; j++)
        {
            if (fabs(damage_broken[i][j]) > EPS)
                damage_D[i][j][0] = 0.5 * (damage_nonlocal[i][0] + damage_nonlocal[neighbors[i][j]][0]);

            damage_w[i][j] = 1.0 - damage_D[i][j][0];
            // damage_w[i][j] = 1.0 - damage_D[i][j][0] / damage_threshold;
        }
    }
    fclose(fpt);

    return k;
}

/* update the damage variables for elasto-plastic materials, using CDM, average bond damage */
int updateDuctileDamageBwiseLocal(const char *dataName, int tstep)
{
    // unlike brittle materials, we donnot limit the broken particle number
    FILE *fpt;
    fpt = fopen(dataName, "a+");
    fprintf(fpt, "TIMESTEP ");
    fprintf(fpt, "%d\n", tstep);

    for (int i = 0; i < nparticle; i++)
    {
        // continuum damage mechanics approach
        double damage_f = (1.0 + damagec_A * J2_triaxiality[i]);
        if (damage_f > 0.0 && damage_local[i][0] <= damage_threshold)
            damage_local[i][0] += damage_f * J2_dlambda[i];
        else if (damage_local[i][0] > 1.0)
            damage_local[i][0] = 1.0;
    }

    int k = 0; // broken number
    for (int i = 0; i < nparticle; i++)
    {
        // compute weighted stretches and bond-wise damage
        // double stretch[50] = {0.0};
        // double max_stretch = 0.0;
        // for (int j = 0; j < nb_initial[i]; j++)
        // {
        //     double temp_str = dL_ave[i][j] / distance_initial[i][j];
        //     if (temp_str > EPS * EPS)
        //     {
        //         stretch[j] = temp_str;

        //         // find the maximum
        //         if (temp_str > max_stretch)
        //             max_stretch = temp_str;
        //     }
        // }

        nb[i] = nb_initial[i];
        for (int j = 0; j < nb_initial[i]; j++)
        {
            // stretch[j] = stretch[j] / max_stretch; // weighted stretch
            // damage_D[i][j][0] = stretch[j] * 0.5 * (damage_local[i][0] + damage_local[neighbors[i][j]][0]);
            damage_D[i][j][0] = 0.5 * (damage_local[i][0] + damage_local[neighbors[i][j]][0]);
            // damage_D[i][j][0] = MAX(damage_local[i][0], damage_local[neighbors[i][j]][0]);

            // if (damage_broken[i][j] <= EPS)
            //     damage_D[i][j][0] = 1.0;
            // else
            //     damage_D[i][j][0] = stretch[j] * MAX(damage_local[i][0], damage_local[neighbors[i][j]][0]);

            // not broken yet
            if (damage_D[i][j][0] > damage_threshold && damage_broken[i][j] > EPS)
            {
                damage_D[i][j][0] = 1.0;
                damage_broken[i][j] = 0.0;

                for (int jj = 0; jj < nneighbors; jj++)
                {
                    if (neighbors[neighbors[i][j]][jj] == i) /* find the opposite particle */
                    {
                        damage_D[neighbors[i][j]][jj][0] = 1.0;
                        damage_broken[neighbors[i][j]][jj] = 0.0;
                    }
                }

                fprintf(fpt, "%d %d \n", i, neighbors[i][j]);
                k++;
            }

            if (damage_broken[i][j] <= EPS)
                nb[i] -= 1;
        }
    }

    for (int i = 0; i < nparticle; i++)
    {
        for (int j = 0; j < nb_initial[i]; j++)
        {
            if (fabs(damage_broken[i][j]) < EPS || nb[i] == 0 || nb[neighbors[i][j]] == 0)
                damage_D[i][j][0] = 1.0;

            damage_w[i][j] = 1.0 - damage_D[i][j][0];
            // damage_w[i][j] = 1.0 - damage_D[i][j][0] / damage_threshold;
        }
    }
    fclose(fpt);

    return k;
}

/* update the damage variables for elasto-plastic materials, using damage accumulation laws, fix the broken particle */
int updateDuctileDamagePwiseLocal(const char *dataName, int tstep)
{
    // unlike brittle materials, we donnot limit the broken particle number
    FILE *fpt;
    fpt = fopen(dataName, "a+");
    fprintf(fpt, "TIMESTEP ");
    fprintf(fpt, "%d\n", tstep);

    // update the particle-wise damage
    int k = 0; // broken number
    for (int i = 0; i < nparticle; i++)
    {
        // continuum damage mechanics approach
        double damage_f = (1.0 + damagec_A * J2_triaxiality[i]);
        if (damage_f > 0.0 && damage_local[i][0] <= damage_threshold)
            damage_local[i][0] += damage_f * J2_dlambda[i];

        // fix the particles of which damage larger than threshold value
        if (damage_local[i][0] > damage_threshold && fabs(damage_local[i][0] - 1.0) > EPS)
        {
            damage_local[i][0] = 1.0; // totally broken
            for (int j = 0; j < nb_initial[i]; j++)
            {
                damage_broken[i][j] = 0.0;
                // damage_D[i][j][0] = damage_threshold;
                // damage_D[i][j][0] = 1.0;

                for (int jj = 0; jj < nneighbors; jj++)
                {
                    if (neighbors[neighbors[i][j]][jj] == i) /* find the opposite particle */
                    {
                        damage_broken[neighbors[i][j]][jj] = 0.0;
                        // damage_D[neighbors[i][j]][jj][0] = 1.0;
                    }
                }
            }

            fprintf(fpt, "%d \n", i);
            k++;
        }
    }

    // update the bond-wise damage
    for (int i = 0; i < nparticle; i++)
    {
        for (int j = 0; j < nb_initial[i]; j++)
        {
            damage_D[i][j][0] = MAX(damage_local[i][0], damage_local[neighbors[i][j]][0]);
            damage_w[i][j] = 1.0 - damage_D[i][j][0];
            // damage_w[i][j] = 1.0 - damage_D[i][j][0] / damage_threshold;
        }
    }
    fclose(fpt);

    return k;
}

/* update the damage variables for elasto-plastic materials, using damage accumulation laws, fix the broken particle */
/* nonlocal */
int updateDuctileDamagePwiseNonlocal(const char *dataName, int tstep)
{
    // unlike brittle materials, we donnot limit the broken particle number
    FILE *fpt;
    fpt = fopen(dataName, "a+");
    fprintf(fpt, "TIMESTEP ");
    fprintf(fpt, "%d\n", tstep);

    // #pragma omp parallel for
    //     // nonlocal damage variable, only counts for neighboring particles
    //     for (int i = 0; i < nparticle; i++)
    //     {
    //         if (damage_nonlocal[i][0] > damage_threshold)
    //         {
    //             if (damage_nonlocal[i][0] > 1.0)
    //                 damage_nonlocal[i][0] = 1.0;
    //             continue;
    //         }

    //         double DdotLocal = 0;
    //         double damage_f = (1.0 + damagec_A * J2_triaxiality[i]);
    //         if (damage_f > 0.0)
    //             DdotLocal = J2_dlambda[i] * (1.0 + damagec_A * J2_triaxiality[i]);

    //         double Ddot = DdotLocal * particle_volume;
    //         double A = particle_volume;
    //         for (int j = 0; j < nb_initial[i]; j++)
    //         {
    //             DdotLocal = 0;
    //             damage_f = (1.0 + damagec_A * J2_triaxiality[neighbors[i][j]]);
    //             if (damage_f > 0.0)
    //                 DdotLocal = J2_dlambda[neighbors[i][j]] * (1.0 + damagec_A * J2_triaxiality[neighbors[i][j]]);
    //             // Ddot += damage_broken[i][j] * DdotLocal * DAM_PHI(distance_initial[i][j]) * particle_volume;
    //             // A += damage_broken[i][j] * DAM_PHI(distance_initial[i][j]) * particle_volume;
    //             Ddot += DdotLocal * DAM_PHI(distance_initial[i][j]) * particle_volume;
    //             A += DAM_PHI(distance_initial[i][j]) * particle_volume;
    //         }
    //         if (Ddot > 0.0)
    //             damage_nonlocal[i][0] += 1.0 / A * Ddot;
    //     }

#pragma omp parallel for
    // nonlocal damage variables, counts for all particles
    for (int i = 0; i < nparticle; i++)
    {
        if (damage_nonlocal[i][0] > damage_threshold)
        {
            if (damage_nonlocal[i][0] > 1.0)
                damage_nonlocal[i][0] = 1.0;
            continue;
        }

        double Ddot = 0;
        double A = 0;
        for (int j = 0; j < nparticle; j++)
        {
            double DdotLocal = 0;
            double dis = sqrt(pow((xyz_initial[j][0] - xyz_initial[i][0]), 2) + pow((xyz_initial[j][1] - xyz_initial[i][1]), 2) + pow((xyz_initial[j][2] - xyz_initial[i][2]), 2));
            if (dis < 3 * damage_L)
            {
                double damage_f = (1.0 + damagec_A * J2_triaxiality[j]);
                if (damage_f > 0.0)
                    DdotLocal = J2_dlambda[j] * (1.0 + damagec_A * J2_triaxiality[j]);
                Ddot += DdotLocal * DAM_PHI(dis) * particle_volume;
                A += DAM_PHI(dis) * particle_volume;
            }
        }
        if (Ddot > 0.0)
            damage_nonlocal[i][0] += 1.0 / A * Ddot;
    }

    int k = 0; // broken number
    for (int i = 0; i < nparticle; i++)
    {
        for (int j = 0; j < nb_initial[i]; j++)
        {
            if (damage_nonlocal[i][0] > damage_threshold || damage_nonlocal[neighbors[i][j]][0] > damage_threshold)
            {
                // test if already broken
                if (fabs(damage_broken[i][j]) > EPS)
                {
                    damage_broken[i][j] = 0.0;
                    damage_D[i][j][0] = 1.0;

                    fprintf(fpt, "%d %d \n", i, neighbors[i][j]);
                    k++;
                }
            }
        }
    }

    // update the bond-wise damage
    for (int i = 0; i < nparticle; i++)
    {
        for (int j = 0; j < nb_initial[i]; j++)
        {
            if (fabs(damage_broken[i][j]) > EPS)
                damage_D[i][j][0] = MAX(damage_nonlocal[i][0], damage_nonlocal[neighbors[i][j]][0]);
            damage_w[i][j] = 1.0 - damage_D[i][j][0];
            // damage_w[i][j] = 1.0 - damage_D[i][j][0] / damage_threshold;
        }
    }
    fclose(fpt);

    return k;
}

void computeCab()
{
    /* compute the -dRSF/dgama */
#pragma omp parallel for
    for (int i = 0; i < nparticle; i++)
    {
        /* m is the alpha slip system */
        for (int m = 0; m < nslipSys; m++)
        {
            /* n is the beta slip system */
            for (int n = 0; n < nslipSys; n++)
            {
                /* compute LSum */
                double LSum[2] = {0, 0};
                for (int j = 0; j < nb_initial[i]; j++)
                {
                    LSum[nsign[i][j]] += Tv[i][j] * distance[i][j] * (csx[i][j] * csx[i][j] * schmid_tensor[n][0] + csy[i][j] * csy[i][j] * schmid_tensor[n][1] + csz[i][j] * csz[i][j] * schmid_tensor[n][2] + csy[i][j] * csz[i][j] * schmid_tensor[n][3] + csx[i][j] * csz[i][j] * schmid_tensor[n][4] + csx[i][j] * csy[i][j] * schmid_tensor[n][5]);
                }

                /* compute dRSFdgama */
                double dFdgama = 0;
                cp_Cab[i][m * nslipSys + n] = 0.;

                for (int j = 0; j < nb_initial[i]; j++)
                {
                    dFdgama = -2.0 * Kn[i][j] * distance[i][j] * (csx[i][j] * csx[i][j] * schmid_tensor[n][0] + csy[i][j] * csy[i][j] * schmid_tensor[n][1] + csz[i][j] * csz[i][j] * schmid_tensor[n][2] + csy[i][j] * csz[i][j] * schmid_tensor[n][3] + csx[i][j] * csz[i][j] * schmid_tensor[n][4] + csx[i][j] * csy[i][j] * schmid_tensor[n][5]) - 2.0 * LSum[nsign[i][j]];

                    /* check if there are any opposite bonds, if yes then 1 */
                    double opp_flag = 1.0;
                    if (nb[i] == nneighbors)
                        opp_flag = 0.5; // there are opposite bond
                    else
                    {
                        for (int m = 0; m < nb_initial[i]; m++)
                        {
                            if (fabs(csx_initial[i][m] + csx_initial[i][j]) < EPS &&
                                fabs(csy_initial[i][m] + csy_initial[i][j]) < EPS &&
                                fabs(csz_initial[i][m] + csz_initial[i][j]) < EPS)
                            {
                                if (damage_broken[i][m] <= EPS) // this is a broken bond
                                    opp_flag = 1.0;
                                else
                                    opp_flag = 0.5; // there are opposite bond
                                break;
                            }
                        }
                    }

                    cp_Cab[i][m * nslipSys + n] += -opp_flag / particle_volume * distance[i][j] * dFdgama * (csx[i][j] * csx[i][j] * schmid_tensor[m][0] + csy[i][j] * csy[i][j] * schmid_tensor[m][1] + csz[i][j] * csz[i][j] * schmid_tensor[m][2] + csy[i][j] * csz[i][j] * schmid_tensor[m][3] + csx[i][j] * csz[i][j] * schmid_tensor[m][4] + csx[i][j] * csy[i][j] * schmid_tensor[m][5]);
                }
            }
        }
    }
}
