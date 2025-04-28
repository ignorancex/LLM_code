#include <stdio.h>
#include <math.h>
#include <memory.h>

#include "lpm.h"

// void readLammps(const char *dataName, int skip)
// {
//     /* read the coordinate information of LAMMPS data file into the particle coordinate system */
//     FILE *fpt;
//     fpt = fopen(dataName, "r+"); /* read-only */

//     if (fpt == NULL)
//     {
//         printf("\'%s\' does not exist!\n", dataName);
//         exit(1);
//     }

//     char line[MAXLINE]; /* stores lines as they are read in */

//     /* count total particl e s */
//     nparticle = 0;
//     rewind(fpt);
//     for (int n = 0; n < skip; n++) /* skip beginning lines */
//         fgets(line, MAXLINE, fpt);

//     while (fgets(line, MAXLINE, fpt) != NULL)
//         nparticle++;

//     /* store into position variable xyz */
//     rewind(fpt);
//     for (int n = 0; n < skip; n++) /* skip beginning lines */
//         fgets(line, MAXLINE, fpt);

//     for (int n = 0; n < nparticle; n++)
//         fscanf(fpt, "%*d %*d %lf %lf %lf", &(xyz[n][0]), &(xyz[n][1]), &(xyz[n][2]));

//     fclose(fpt);
// }

/* write the particle-wise information into LAMMPS dump file type */
void writeDump(const char *dataName, int step, char flag)
{
    FILE *fpt;
    if (flag == 's')
        fpt = fopen(dataName, "w+");
    else
        fpt = fopen(dataName, "a+");

    fprintf(fpt, "ITEM: TIMESTEP\n");
    fprintf(fpt, "%d\n", step);
    fprintf(fpt, "ITEM: NUMBER OF ATOMS\n");
    fprintf(fpt, "%d\n", nparticle);
    fprintf(fpt, "ITEM: BOX BOUNDS pp pp pp\n");
    fprintf(fpt, "%8.8f %8.8f\n", box[0], box[1]);
    fprintf(fpt, "%8.8f %8.8f\n", box[2], box[3]);
    fprintf(fpt, "%8.8f %8.8f\n", box[4], box[5]);

    if (plmode == 1)
    {
        fprintf(fpt, "ITEM: ATOMS id type x y z dx dy dz s11 s22 s33 s23 s13 s12 gam11 gam12 gam13 gam14 gam21 gam22 gam23 gam24 pl gam_total\n");
        for (int i = 0; i < nparticle; i++)
        {
            fprintf(fpt, "%d %d %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %d %.4e\n", i, type[i],
                    xyz[i][0], xyz[i][1], xyz[i][2], xyz[i][0] - xyz_initial[i][0], xyz[i][1] - xyz_initial[i][1], xyz[i][2] - xyz_initial[i][2],
                    stress_tensor[i][0], stress_tensor[i][1], stress_tensor[i][2], stress_tensor[i][3], stress_tensor[i][4], stress_tensor[i][5],
                    cp_A_single[i][1][0], cp_A_single[i][4][0], cp_A_single[i][10][0], cp_A_single[i][19][0],
                    cp_A_single[i][9][0], cp_A_single[i][14][0], cp_A_single[i][17][0], cp_A_single[i][22][0], pl_flag[i], cp_A[i][0]);
        }
    }
    else
    {
        fprintf(fpt, "ITEM: ATOMS id type x y z dx dy dz s11 s22 s33 s23 s13 s12 stress_eq strain_eq stress_m stress_tri damage_local damage_nonlocal pl damage\n");
        for (int i = 0; i < nparticle; i++)
        {
            fprintf(fpt, "%d %d %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %d %.4e\n", i, type[i],
                    xyz[i][0], xyz[i][1], xyz[i][2], xyz[i][0] - xyz_initial[i][0], xyz[i][1] - xyz_initial[i][1], xyz[i][2] - xyz_initial[i][2],
                    stress_tensor[i][0], stress_tensor[i][1], stress_tensor[i][2], stress_tensor[i][3], stress_tensor[i][4], stress_tensor[i][5],
                    J2_stresseq[i], J2_alpha[i][0], J2_stressm[i], J2_triaxiality[i], damage_local[i][0], damage_nonlocal[i][0], pl_flag[i], damage_visual[i]);
        }
    }

    fclose(fpt);
}

/* store the slip information into file */
void writeRSS(const char *dataName, int step)
{
    FILE *fpt = fopen(dataName, "a+");
    fprintf(fpt, "TIMESTEP ");
    fprintf(fpt, "%d\n", step);
    fprintf(fpt, "id type ");
    for (int i = 0; i < nslipSys; i++)
        fprintf(fpt, "slip%u ", i + 1);

    fprintf(fpt, "\n");
    for (int i = 0; i < nparticle; i++)
    {
        fprintf(fpt, "%u %d ", i, type[i]);

        for (int j = 0; j < nslipSys; j++)
            fprintf(fpt, "%.6e ", cp_RSS[i][j]); // cp_RSS[i][j] OR dcp_RSS[i][j]

        fprintf(fpt, "\n");
    }
    fclose(fpt);
}

/* Store the stress information into file, for the particle type t1 */
void writeStress(const char *dataName, int t1, int tstep)
{
    FILE *fpt = fopen(dataName, "a+");

    // compute the average stress value for particle type t1
    int num = 0;
    double stress_ave[2 * NDIM] = {0.0};
    for (int i = 0; i < nparticle; i++)
    {
        if (type[i] == t1)
        {
            num++;
            for (int j = 0; j < 2 * NDIM; j++)
                stress_ave[j] += stress_tensor[i][j];
        }
    }

    // output the average stress value
    fprintf(fpt, "%d ", tstep);
    for (int j = 0; j < 2 * NDIM; j++)
        fprintf(fpt, "%.6e ", stress_ave[j] / num);
    fprintf(fpt, "\n");

    fclose(fpt);
}

/* Store the strain information into file, for the particle type t1 */
void writeStrain(const char *dataName, int t1, int tstep)
{
    FILE *fpt = fopen(dataName, "a+");

    // compute the average stress value for particle type t1
    int num = 0;
    double strain_ave[2 * NDIM] = {0.0};
    for (int i = 0; i < nparticle; i++)
    {
        if (type[i] == t1)
        {
            num++;
            for (int j = 0; j < 2 * NDIM; j++)
                strain_ave[j] += strain_tensor[i][j];
        }
    }

    // output the average stress value
    fprintf(fpt, "%d ", tstep);
    for (int j = 0; j < 2 * NDIM; j++)
        fprintf(fpt, "%.6e ", strain_ave[j] / num);
    fprintf(fpt, "\n");

    fclose(fpt);
}

// write out the Cab matrix for particle ii for a time step
void writeCab(const char *dataName, int ii)
{
    FILE *fpt;
    fpt = fopen(dataName, "a+");

    fprintf(fpt, "Particle ID: %d\n", ii);

    /* m is the alpha slip system */
    for (int m = 0; m < nslipSys; m++)
    {
        /* n is the beta slip system */
        for (int n = 0; n < nslipSys; n++)
        {
            fprintf(fpt, "%.4e ", cp_Cab[ii][m * nslipSys + n]);
        }
        fprintf(fpt, "\n");
    }
    fprintf(fpt, "\n");

    fclose(fpt);
}

void writeInternalForce(const char *dataName, int step)
{
    FILE *fpt = fopen(dataName, "a+");

    fprintf(fpt, "TIMESTEP ");
    fprintf(fpt, "%d\n", step);
    fprintf(fpt, "id type fx fy fz\n");

    for (int ii = 0; ii < nparticle; ii++)
    {
        /* output the internal force of particles */
        double fx = 0., fy = 0., fz = 0.;
        for (int j = 0; j < nb_initial[ii]; j++)
        {
            fx += csx[ii][j] * F[ii][j];
            fy += csy[ii][j] * F[ii][j];
            fz += csz[ii][j] * F[ii][j];
        }
        fprintf(fpt, "%u %d %.6e %.6e %.6e\n", ii, type[ii], fx, fy, fz);
    }

    fclose(fpt);
}

/* output the active slip systems */
void writeJact(const char *dataName, int step)
{
    FILE *fpt;
    fpt = fopen(dataName, "a+");

    fprintf(fpt, "timestep ");
    fprintf(fpt, "%d\n", step);
    fprintf(fpt, "id type actived_slip_systems\n");

    for (int ii = 0; ii < nparticle; ii++)
    {
        fprintf(fpt, "%u %d ", ii, type[ii]);
        for (int m = 0; m < nslipSys; m++)
        {
            if (cp_Jact[ii][m] == 1)
                fprintf(fpt, "%d ", m);
        }
        fprintf(fpt, "\n");
    }
    fclose(fpt);
}

/* store the bond length change information into file */
void writeBondstretch(const char *dataName, int step)
{
    FILE *fpt;
    fpt = fopen(dataName, "a+");
    fprintf(fpt, "TIMESTEP ");
    fprintf(fpt, "%d\n", step);

    // for (int i = 0; i < nparticle; i++)
    // {
    //     fprintf(fpt, "%d %d ", i, type[i]);
    //     for (int m = 0; m < nneighbors; m++)
    //         fprintf(fpt, "%.4e ", bond_stretch[i][m]);
    //     fprintf(fpt, "\n");
    // }
    for (int i = 0; i < nparticle; i++)
    {
        fprintf(fpt, "%d %d ", i, type[i]);
        for (int m = 0; m < nb_initial[i]; m++)
            fprintf(fpt, "%.4e ", dL[i][m] / distance_initial[i][m]);
        fprintf(fpt, "\n");
    }

    fclose(fpt);
}

/* store the bond force information into file */
void writeBondforce(const char *dataName, int step)
{
    FILE *fpt;
    fpt = fopen(dataName, "a+");
    fprintf(fpt, "TIMESTEP ");
    fprintf(fpt, "%d\n", step);

    for (int i = 0; i < nparticle; i++)
    {
        fprintf(fpt, "%d %d ", i, type[i]);
        for (int m = 0; m < nneighbors; m++)
            fprintf(fpt, "%.4e ", bond_force[i][m]);
        fprintf(fpt, "\n");
    }
    fclose(fpt);
}

/* output the absolute average displacement for a particular type of particle */
void writeDisp(const char *dataName, char c, int t1, int tstep)
{
    FILE *fpt;
    fpt = fopen(dataName, "a+");

    int flag;

    if (c == 'x')
        flag = 0;
    else if (c == 'y')
        flag = 1;
    else
        flag = 2;

    int num1 = 0;
    double p = 0, p0 = 0;
    for (int i = 0; i < nparticle; i++)
    {
        if (type[i] == t1)
        {
            p += xyz[i][flag];
            p0 += xyz_initial[i][flag];
            num1++;
        }
    }
    fprintf(fpt, "%d %8.8e\n", tstep, (p - p0) / num1);

    fclose(fpt);
}

/* output the reaction force for a particular type of particle */
void writeReaction(const char *dataName, char c, int t1, int tstep)
{
    FILE *fpt;
    fpt = fopen(dataName, "a+");

    /* calculate force of the domain */
    double force = 0.;

    for (int i = 0; i < nparticle; i++)
    {
        if (type[i] == t1)
        {
            for (int j = 0; j < nb_initial[i]; j++)
            {
                if (c == 'x')
                    force += -csx[i][j] * F[i][j];
                else if (c == 'y')
                    force += -csy[i][j] * F[i][j];
                else if (c == 'z')
                    force += -csz[i][j] * F[i][j];
            }
        }
    }

    fprintf(fpt, "%d %8.8e \n", tstep, force);
    fclose(fpt);
}

/* calculate the reaction force of the lower part (with the ratio p) of the body */
void writeForce(const char *dataname, char c, double p, int tstep)
{
    FILE *fpt;
    fpt = fopen(dataname, "a+");

    /* define the lower part of the particle system as a "boundary" */
    int *temp_BC = allocInt1D(nparticle, 0);

    int nbd = 0;
    for (int i = 0; i < nparticle; i++)
    {
        if (c == 'x')
        {
            if (xyz_initial[i][0] < box[0] + p * (box_x))
            {
                temp_BC[nbd] = i;
                nbd += 1;
            }
        }
        else if (c == 'y')
        {
            if (xyz_initial[i][1] < box[2] + p * (box_y))
            {
                temp_BC[nbd] = i;
                nbd += 1;
            }
        }
        else if (c == 'z')
        {
            if (xyz_initial[i][2] < box[4] + p * (box_z))
            {
                temp_BC[nbd] = i;
                nbd += 1;
            }
        }
    }

    /* calculate force of the domain */
    double force = 0.;

    for (int i = 0; i < nbd; i++)
    {
        for (int j = 0; j < nb_initial[temp_BC[i]]; j++)
        {
            if (c == 'x')
                force += -csx[temp_BC[i]][j] * F[temp_BC[i]][j];
            else if (c == 'y')
                force += -csy[temp_BC[i]][j] * F[temp_BC[i]][j];
            else if (c == 'z')
                force += -csz[temp_BC[i]][j] * F[temp_BC[i]][j];
        }
    }

    fprintf(fpt, "%d %8.8e \n", tstep, force);
    fclose(fpt);
}

/* write global stiffness matrix into file */
void writeK_global(const char *dataName, int l)
{
    FILE *fpt;
    fpt = fopen(dataName, "w+");
    for (int i = 0; i < l; i++)
    {
        fprintf(fpt, " %.5e\n", K_global[i]);
    }

    fclose(fpt);
}

/* write the plasticity multiplier array into file, from m to n (not included) indices */
void writeDlambda(const char *dataName, int m, int n, int globalStep, int iterStep)
{
    FILE *fpt;
    fpt = fopen(dataName, "a+");
    fprintf(fpt, "Step (Global-Iteration) ");
    fprintf(fpt, "%d-%d\n", globalStep, iterStep);

    for (int i = m; i < n; i++)
    {
        fprintf(fpt, "%d %.4e %.4e %.4e\n", i, J2_dlambda[i], J2_alpha[i][0], J2_alpha[i][1]);
    }

    fclose(fpt);
}

/* write the bond-wise damage value into file */
void writeDamage(const char *dataName, int step)
{
    FILE *fpt;
    fpt = fopen(dataName, "a+");
    fprintf(fpt, "TIMESTEP ");
    fprintf(fpt, "%d\n", step);

    for (int i = 0; i < nparticle; i++)
    {
        fprintf(fpt, "%d %d ", i, type[i]);
        for (int m = 0; m < nb_initial[i]; m++)
        {
            fprintf(fpt, "%.4e ", damage_D[i][m][0]);
        }
        fprintf(fpt, "\n");
    }
    fclose(fpt);
}

/* write the neighbor information of all particles into file */
void writeNeighbor(const char *dataName)
{
    FILE *fpt;
    fpt = fopen(dataName, "a+");

    for (int i = 0; i < nparticle; i++)
    {
        fprintf(fpt, "%d %d ", i, type[i]);
        for (int m = 0; m < nb_initial[i]; m++)
        {
            fprintf(fpt, "%d ", neighbors[i][m]);
        }
        fprintf(fpt, "\n");
    }
    fclose(fpt);
}