#include <stdio.h>
#include <memory.h>
#include <stdlib.h>

#include "lpm.h"

/* organize the bond force in order */
void computeBondForce()
{
    for (int i = 0; i < nparticle; i++)
    {
        for (int k = 0; k < nneighbors; k++)
        {
            for (int j = 0; j < nb_initial[i]; j++)
            {
                double p[3] = {csx[i][j] - bond_vector[k][0],
                               csy[i][j] - bond_vector[k][1],
                               csz[i][j] - bond_vector[k][2]};
                double pnrm = cblas_dnrm2(3, p, 1); // Euclidean norm
                if (pnrm < TOL)
                {
                    bond_force[i][k] = F[i][j];
                    break;
                }
            }
        }
    }
}

void computeBondStretch()
{
    for (int i = 0; i < nparticle; i++)
    {
        for (int k = 0; k < nneighbors; k++)
        {
            for (int j = 0; j < nb_initial[i]; j++)
            {
                double p[3] = {csx[i][j] - bond_vector[k][0],
                               csy[i][j] - bond_vector[k][1],
                               csz[i][j] - bond_vector[k][2]};
                double pnrm = cblas_dnrm2(3, p, 1); // Euclidean norm
                if (pnrm < TOL)
                {
                    bond_stretch[i][k] = dL[i][j] / distance_initial[i][j];
                    break;
                }
            }
        }
    }
}

/* compute the stress tensor (modified-2) */
void computeStress()
{
#pragma omp parallel for
    for (unsigned i = 0; i < (unsigned)nparticle; i++)
    {
        memset(stress_tensor[i], 0.0, 2 * NDIM * sizeof(double));

        /* compute stress tensor, s11, s22, s33, s23, s13, s12 */
        for (int j = 0; j < nb_initial[i]; j++)
        {
            /* compute stress tensor, s11, s22, s33, s23, s13, s12 */
            //stress_tensor[i][0] += 0.5 / particle_volume * distance_initial[i][j] * F[i][j] * csx[i][j] * csx[i][j] * nneighbors / nb[i];
            //stress_tensor[i][1] += 0.5 / particle_volume * distance_initial[i][j] * F[i][j] * csy[i][j] * csy[i][j] * nneighbors / nb[i];
            //stress_tensor[i][2] += 0.5 / particle_volume * distance_initial[i][j] * F[i][j] * csz[i][j] * csz[i][j] * nneighbors / nb[i];
            //stress_tensor[i][3] += 0.5 / particle_volume * distance_initial[i][j] * F[i][j] * csy[i][j] * csz[i][j] * nneighbors / nb[i];
            //stress_tensor[i][4] += 0.5 / particle_volume * distance_initial[i][j] * F[i][j] * csx[i][j] * csz[i][j] * nneighbors / nb[i];
            //stress_tensor[i][5] += 0.5 / particle_volume * distance_initial[i][j] * F[i][j] * csx[i][j] * csy[i][j] * nneighbors / nb[i];

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
            stress_tensor[i][0] += opp_flag / particle_volume * distance_initial[i][j] * F[i][j] * csx[i][j] * csx[i][j];
            stress_tensor[i][1] += opp_flag / particle_volume * distance_initial[i][j] * F[i][j] * csy[i][j] * csy[i][j];
            stress_tensor[i][2] += opp_flag / particle_volume * distance_initial[i][j] * F[i][j] * csz[i][j] * csz[i][j];
            stress_tensor[i][3] += opp_flag / particle_volume * distance_initial[i][j] * F[i][j] * csy[i][j] * csz[i][j];
            stress_tensor[i][4] += opp_flag / particle_volume * distance_initial[i][j] * F[i][j] * csx[i][j] * csz[i][j];
            stress_tensor[i][5] += opp_flag / particle_volume * distance_initial[i][j] * F[i][j] * csx[i][j] * csy[i][j];
        }

        J2_stresseq[i] = 0.0;
        for (int j = 0; j < 2 * NDIM; j++)
        {
            if (j < NDIM)
                J2_stresseq[i] += stress_tensor[i][j] * stress_tensor[i][j]; // s11, s22, s33
            else
                J2_stresseq[i] += 2.0 * stress_tensor[i][j] * stress_tensor[i][j]; // s23, s13, s12
        }
        J2_stresseq[i] = sqrt(3.0 / 2.0 * J2_stresseq[i]);
        J2_stressm[i] = 1.0 / 3.0 * (stress_tensor[i][0] + stress_tensor[i][1] + stress_tensor[i][2]);
        if (J2_stresseq[i] > EPS)
            J2_triaxiality[i] = J2_stressm[i] / J2_stresseq[i];

        /* compute bond stress (even the bond has broken) */
        for (int j = 0; j < nb_initial[i]; j++)
        {
            bond_stress[i][j] = (stress_tensor[i][0] * csx[i][j] * csx[i][j] +
                                 stress_tensor[i][1] * csy[i][j] * csy[i][j] +
                                 stress_tensor[i][2] * csz[i][j] * csz[i][j] +
                                 2 * stress_tensor[i][3] * csy[i][j] * csz[i][j] +
                                 2 * stress_tensor[i][4] * csx[i][j] * csz[i][j] +
                                 2 * stress_tensor[i][5] * csx[i][j] * csy[i][j]);
        }
    }
}

void computeStrain()
{
    /* compute the strain tensor using weighted least squre */
    double *matrix_A = allocDouble1D(3 * (dim - 1) * 3 * (dim - 1), 0);
    double *rhs_b = allocDouble1D(3 * (dim - 1), 0);

    double w1 = 0.1, w2 = 0.9; /* weights for the weighted least square method */

    for (int i = 0; i < nparticle; i++)
    {
        double r4t[NDIM][NDIM][NDIM][NDIM] = {0.};
        double r2t[NDIM][NDIM] = {0.}; /* in 2d, only 2 components are used */

        /* first nearst neighbors */
        for (int j = 0; j < nb_initial[i]; j++)
        {
            /* first nearst neighbors */
            if (nsign[i][j] == 0)
            {
                for (int k = 0; k < dim; k++)
                    for (int n = 0; n < dim; n++)
                        for (int m = 0; m < dim; m++)
                            for (int l = 0; l < dim; l++)
                                r4t[k][n][m][l] += w1 * (xyz_initial[neighbors[i][j]][k] - xyz_initial[i][k]) / distance_initial[i][j] *
                                                   (xyz_initial[neighbors[i][j]][n] - xyz_initial[i][n]) / distance_initial[i][j] *
                                                   (xyz_initial[neighbors[i][j]][m] - xyz_initial[i][m]) / distance_initial[i][j] *
                                                   (xyz_initial[neighbors[i][j]][l] - xyz_initial[i][l]) / distance_initial[i][j];
                for (int k = 0; k < dim; k++)
                    for (int n = 0; n < dim; n++)
                        r2t[k][n] += w1 * dL[i][j] / distance_initial[i][j] *
                                     (xyz_initial[neighbors[i][j]][k] - xyz_initial[i][k]) / distance_initial[i][j] *
                                     (xyz_initial[neighbors[i][j]][n] - xyz_initial[i][n]) / distance_initial[i][j];
            }
            /* second nearst neighbors */
            else if (nsign[i][j] == 1)
            {
                for (int k = 0; k < dim; k++)
                    for (int n = 0; n < dim; n++)
                        for (int m = 0; m < dim; m++)
                            for (int l = 0; l < dim; l++)
                                r4t[k][n][m][l] += w2 * (xyz_initial[neighbors[i][j]][k] - xyz_initial[i][k]) / distance_initial[i][j] *
                                                   (xyz_initial[neighbors[i][j]][n] - xyz_initial[i][n]) / distance_initial[i][j] *
                                                   (xyz_initial[neighbors[i][j]][m] - xyz_initial[i][m]) / distance_initial[i][j] *
                                                   (xyz_initial[neighbors[i][j]][l] - xyz_initial[i][l]) / distance_initial[i][j];
                for (int k = 0; k < dim; k++)
                    for (int n = 0; n < dim; n++)
                        r2t[k][n] += w2 * dL[i][j] / distance_initial[i][j] *
                                     (xyz_initial[neighbors[i][j]][k] - xyz_initial[i][k]) / distance_initial[i][j] *
                                     (xyz_initial[neighbors[i][j]][n] - xyz_initial[i][n]) / distance_initial[i][j];
            }
        }

        memset(matrix_A, 0, 3 * (dim - 1) * 3 * (dim - 1) * sizeof(double));
        memset(rhs_b, 0, 3 * (dim - 1) * sizeof(double));

        /* construct the linear system */
        int ii = 0;
        for (int j = 0; j < dim; j++)
            for (int k = 0; k < dim; k++)
                for (int m = 0; m < dim; m++)
                    for (int l = 0; l < dim; l++)
                    {
                        if (m <= l && j <= k)
                        {
                            matrix_A[ii] = r4t[m][l][j][k];
                            ii++;
                        }
                    }

        ii = 0;
        for (int j = 0; j < dim; j++)
            for (int k = 0; k < dim; k++)
            {
                if (j <= k)
                {
                    rhs_b[ii] = r2t[j][k];
                    ii++;
                }
            }

        /* compute the strain tensor e = rhs_b/matrix_A */
        MKL_INT n = 3 * (dim - 1),
                nrhs = 1, lda = 3 * (dim - 1), ldb = 1; /* ldb is 1 for ROW MAJOR layout */
        MKL_INT ipiv[2 * 3];                            /* max number of strain components is 6 */

        int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, matrix_A, lda, ipiv, rhs_b, ldb);

        /* Check for the exact singularity */
        // if (info > 0)
        // {
        //     printf("Zero diagonal component found in A[%i][%i]!\n", info, info);
        //     exit(1);
        // }
        // if this particle has some zero diagonal components, assign its strain as nearby particles, or 0
        if (info > 0 && i == 0)
            continue;
        if (info > 0 && i != 0)
        {
            for (int k = 0; k < 3 * (dim - 1); k++)
                strain_tensor[i][k] = strain_tensor[i - 1][k];
        }

        /* copy into real strain tensor array (there are 6 components in it) */
        if (dim == 2)
        {
            strain_tensor[i][0] = rhs_b[0]; // e11
            strain_tensor[i][5] = rhs_b[1]; // e12
            strain_tensor[i][1] = rhs_b[2]; // e22
        }
        else if (dim == 3)
        {
            strain_tensor[i][0] = rhs_b[0]; // e11
            strain_tensor[i][5] = rhs_b[1]; // e12
            strain_tensor[i][4] = rhs_b[2]; // e13
            strain_tensor[i][1] = rhs_b[3]; // e22
            strain_tensor[i][3] = rhs_b[4]; // e23
            strain_tensor[i][2] = rhs_b[5]; // e33
        }
    }

    free(matrix_A);
    free(rhs_b);
}

/* compute the bond length change under the small deformation assumption */
void computedL()
{
#pragma omp parallel for
    for (int i = 0; i < nparticle; i++)
    {
        dL_total[i][0] = 0, TdL_total[i][0] = 0; /* first layer of neighbors */
        dL_total[i][1] = 0, TdL_total[i][1] = 0; /* second layer of neighbors */

        // int num1 = 0, num2 = 0;
        for (int j = 0; j < nb_initial[i]; j++)
        {
            /* compute the bond stretch */
            // double p[3] = {0.}, pnrm = {0.};
            double xyz_j[3] = {0.}; /* temporary coordinates of neighbor j */

            xyz_j[0] = xyz[neighbors[i][j]][0];
            xyz_j[1] = xyz[neighbors[i][j]][1];
            xyz_j[2] = xyz[neighbors[i][j]][2];

            distance[i][j] = sqrt(pow(xyz[i][0] - xyz_j[0], 2) +
                                  pow(xyz[i][1] - xyz_j[1], 2) +
                                  pow(xyz[i][2] - xyz_j[2], 2));

            dL[i][j] = distance[i][j] - distance_initial[i][j];
            dL[i][j] -= dLp[i][j][0]; /* update the elastic bond stretch */

            dL_total[i][nsign[i][j]] += dL[i][j];
            TdL_total[i][nsign[i][j]] += Tv[i][j] * dL[i][j];

            csx[i][j] = (xyz[i][0] - xyz_j[0]) / distance[i][j];
            csy[i][j] = (xyz[i][1] - xyz_j[1]) / distance[i][j];
            csz[i][j] = (xyz[i][2] - xyz_j[2]) / distance[i][j];

            // // compute below information using initial configuration
            // csx[i][j] = csx_initial[i][j];
            // csy[i][j] = csy_initial[i][j];
            // csz[i][j] = csz_initial[i][j];
        }
    }
}

/* find the maximum value, and return index */
int findMaxDouble(double *arr, int len)
{
    double max = arr[0];

    int t = 0;
    for (int i = 0; i < len; i++)
    {
        if (arr[i] > max)
        {
            max = arr[i];
            t = i;
        }
    }

    return t;
}

int findMaxInt(int *arr, int len)
{

    int max = arr[0];

    int t;
    for (int i = 0; i < len; i++)
    {
        if (arr[i] > max)
        {
            max = arr[i];
            t = i;
        }
    }

    return t;
}

/* assign particle type in a rectangular region */
void setTypeRect(double r0, double r1, double r2, double r3, double r4, double r5, int t)
{
    for (int i = 0; i < nparticle; i++)
    {
        if (xyz[i][0] >= r0 && xyz[i][0] <= r1 && xyz[i][1] >= r2 && xyz[i][1] <= r3 && xyz[i][2] >= r4 && xyz[i][2] <= r5)
            type[i] = t;
    }
}

/* assign particle type in a circle region in xOy plane */
void setTypeCircle(double x, double y, double r, int t)
{
    for (int i = 0; i < nparticle; i++)
    {
        double disq = pow(xyz[i][0] - x, 2.0) + pow(xyz[i][1] - y, 2.0);
        if (disq <= r * r)
            type[i] = t;
    }
}

/* assign particle type as k if the neighbor list is full */
void setTypeFullNeighbor(int k)
{
    for (int i = 0; i < nparticle; i++)
    {
        if (nb_initial[i] == nneighbors)
            type[i] = k;
    }
}

/* allocate memory */
int *allocInt1D(int num, int a)
{
    int *res = (int *)malloc(sizeof(int) * num);

    for (int i = 0; i < num; i++)
        res[i] = a;

    return res;
}

double *allocDouble1D(int num, double a)
{
    double *res = (double *)malloc(sizeof(double) * num);

    for (int i = 0; i < num; i++)
        res[i] = a;

    return res;
}

int **allocInt2D(int row, int column, int a)
{
    int **res = (int **)malloc(sizeof(int *) * row);
    for (int i = 0; i < row; i++)
        res[i] = (int *)malloc(sizeof(int) * column);

    for (int i = 0; i < row; i++)
        for (int j = 0; j < column; j++)
            res[i][j] = a;

    return res;
}

int ***allocInt3D(int row, int column, int layer, int a)
{
    int i, j, k;

    int ***res = (int ***)malloc(sizeof(int **) * row);
    for (i = 0; i < row; i++)
    {
        res[i] = (int **)malloc(sizeof(int *) * column);
        for (j = 0; j < column; j++)
            res[i][j] = (int *)malloc(sizeof(int) * layer);
    }

    for (k = 0; k < layer; k++)
        for (i = 0; i < row; i++)
            for (j = 0; j < column; j++)
                res[i][j][k] = a;

    return res;
}

double **allocDouble2D(int row, int column, double a)
{
    double **res = (double **)malloc(sizeof(double *) * row);
    for (int i = 0; i < row; i++)
        res[i] = (double *)malloc(sizeof(double) * column);

    for (int i = 0; i < row; i++)
        for (int j = 0; j < column; j++)
            res[i][j] = a;

    return res;
}

double ***allocDouble3D(int row, int column, int layer, double a)
{
    int i, j, k;

    double ***res = (double ***)malloc(sizeof(double **) * row);
    for (i = 0; i < row; i++)
    {
        res[i] = (double **)malloc(sizeof(double *) * column);
        for (j = 0; j < column; j++)
            res[i][j] = (double *)malloc(sizeof(double) * layer);
    }

    for (k = 0; k < layer; k++)
        for (i = 0; i < row; i++)
            for (j = 0; j < column; j++)
                res[i][j][k] = a;

    return res;
}

/* free allocated memory */
void freeInt2D(int **arr, int row)
{
    for (int i = 0; i < row; i++)
        free((int *)arr[i]);

    free((int *)arr);
}

void freeDouble2D(double **arr, int row)
{
    for (int i = 0; i < row; i++)
        free((double *)arr[i]);

    free((double *)arr);
}

void freeDouble3D(double ***arr, int row, int column)
{
    for (int i = 0; i < row; i++)
        for (int j = 0; j < column; j++)
            free((void *)arr[i][j]);

    for (int i = 0; i < row; i++)
        free((void *)arr[i]);

    free((double *)arr);
}

/* shell sorting */
void sortInt(int *arr, int len)
{
    int i, j, temp, r;

    for (r = len / 2; r >= 1; r = r / 2)
    {
        for (i = r; i < len; ++i)
        {
            temp = arr[i];
            j = i - r;
            while (j >= 0 && arr[j] > temp)
            {
                arr[j + r] = arr[j];
                j = j - r;
            }
            arr[j + r] = temp;
        }
    }
}

/* read a line from a file, return length, and convert the lines beginning with # to all # */
int getLine(char *line, int max, FILE *fp)
{
    int c0, c, i;

    i = 0;
    while (--max > 0 && (c = fgetc(fp)) != EOF && c != '\n')
    {
        if (i == 0)
            c0 = c;
        if (c0 == '#')
            line[i++] = c0;
        else
            line[i++] = c;
    }
    if (c == '\n')
        line[i++] = c;
    line[i] = '\0';

    return i;
}

/* return the number of elements in arr which are equal to b */
int countEqual(int *arr, int len, int b)
{
    int num = 0;
    for (int i = 0; i < len; i++)
        if (arr[i] == b)
            num++;

    return num;
}

/* return the number of elements in arr which are not equal to b */
int countNEqual(int *arr, int len, int b)
{
    int num = 0;
    for (int i = 0; i < len; i++)
        if (arr[i] != b)
            num++;

    return num;
}

/* return the number of elements in arr which are larger than b */
int countLarger(int *arr, int len, int b)
{
    int num = 0;
    for (int i = 0; i < len; i++)
        if (arr[i] > b)
            num++;

    return num;
}

double sumDouble(double *arr, int len)
{
    double s = 0.0;
    for (int i = 0; i < len; i++)
        s += arr[i];

    return s;
}

int sumInt(int *arr, int len)
{
    int s = 0;
    for (int i = 0; i < len; i++)
        s += arr[i];

    return s;
}

void copyDouble1D(double *target, double *source, int row)
{
    /*int i;
    for (i = 0; i < row; i++)
    {
        target[i] = source[i];
    }*/
    memcpy(target, source, sizeof(double) * row);
}

/* Copy the first row*col items from source to target */
void copyDouble2D(double **target, double **source, int row, int col)
{
    int i;
    for (i = 0; i < row; i++)
    {
        /*for (j = 0; j < col; j++)
        {
            target[i][j] = source[i][j];
        }*/
        memcpy(target[i], source[i], sizeof(double) * col);
    }
}

void copyDouble3D(double ***target, double ***source, int row, int col, int deep)
{
    int i, j;
    for (i = 0; i < row; i++)
    {
        for (j = 0; j < col; j++)
        {
            /*for (k = 0; k < deep;k++)
                target[i][j][k] = source[i][j][k];*/
            memcpy(target[i][j], source[i][j], sizeof(double) * deep);
        }
    }
}

void copyInt2D(int **target, int **source, int row, int col)
{
    int i;
    for (i = 0; i < row; i++)
    {
        /*for (j = 0; j < col; j++)
        {
            target[i][j] = source[i][j];
        }*/
        memcpy(target[i], source[i], sizeof(int) * col);
    }
}

/* generate an array that contain the elements which are unequal to 'a',
return the length of the new array */
int findNElement(int *in, int *out, int len, int a)
{
    int k = 0;
    for (int i = 0; i < len; i++)
    {
        if (in[i] != a)
        {
            *(out + k) = in[i];
            k++;
        }
    }
    return k;
}

/* print from m-th to n-th items in the target double array */
void printDouble(double *target, int m, int n)
{
    int i;
    for (i = m - 1; i < n; i++)
    {
        printf("%8.8e \n", target[i]);
    }

    printf("\n");
}

/* print the first n items in the target int array */
void printInt(int *target, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        printf("%d ", target[i]);
    }
    printf("\n");
}