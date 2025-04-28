#include <math.h>
#include <mkl.h>

#include "lpm.h"

void createCuboid()
{
    int i, j, k, n, nparticle_t;
    double x, y, z, a;
    double **xyz_t, p[3] = {0};

    /* settings for matrix-vector product, BLAS */
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_TRANSPOSE trans = CblasNoTrans;

    int lda = 3, incx = 1, incy = 1;
    double blasAlpha = 1.0, blasBeta = 0.0;

    if (dim == 2)
        a = sqrt(pow(box_x, 2) + pow(box_y, 2));
    else
        a = sqrt(pow(box_x, 2) + pow(box_y, 2) + pow(box_z, 2));

    double box_t[6] = {-a, a, -a, a, -a, a};

    /* Below notations are referred to wiki: https://en.wikipedia.org/wiki/Euler_angles */
    /* 1, 2, 3 represent the rotation angles angle1, angle2, gamma */
    /* x, y, z represent fixed frames */
    /* X, Y, Z mean the rotation matrices that give elemental rotations about axies x, y, z */
    /* e.g. X1 means a rotation about x axis by an angle which is angle1 */

    /* Examples of elemental rotation matrices, c1 is cos_alpha, s2 is sin_beta, etc. */
    // X1 = [1   0   0 ]  Y2 = [c2    0   s2]  Z3 = [c3  -s3   0]
    //      [0  c1  -s1]       [ 0    1    0]       [s3   c3   0]
    //      [0  s1   c1]       [-s2   0   c2]       [ 0    0   1]

    // In 2D, only Z3 is used, therefore R = I-I-Z3 = Z3
    // R = [c3  -s3   0]  or R = [c3  -s3]
    //     [s3   c3   0]         [s3   c3]
    //     [0     0   1]

    if (eulerflag == 0)
    {
        /* R = X1-Y2-Z3; row major */
        // X1-Y2-Z3 = [c2c3          -c2s3          s2  ]
        //            [s1s2c3+c1s3   -s1s2s3+c1c3  -s1c2]
        //            [-c1s2c3+s1s3   c1s2s3+s1c3   c1c2]

        R_matrix[0] = cos(angle2) * cos(angle3);
        R_matrix[1] = -cos(angle2) * sin(angle3);
        R_matrix[2] = sin(angle2);
        R_matrix[3] = cos(angle1) * sin(angle3) + sin(angle1) * sin(angle2) * cos(angle3);
        R_matrix[4] = cos(angle1) * cos(angle3) - sin(angle1) * sin(angle2) * sin(angle3);
        R_matrix[5] = -sin(angle1) * cos(angle2);
        R_matrix[6] = sin(angle1) * sin(angle3) - cos(angle1) * sin(angle2) * cos(angle3);
        R_matrix[7] = sin(angle1) * cos(angle3) + cos(angle1) * sin(angle2) * sin(angle3);
        R_matrix[8] = cos(angle1) * cos(angle2);
    }
    else if (eulerflag == 1)
    {
        /* R = Z1-Y2-Z3, using Kocks convention, angle3 = pi - angle3' (angle3' is Kocks input); row major */
        // Z1-Y2-Z3 = [c1c2c3-s1s3  -c1c2s3-s1c3  s2c1]
        //            [s1c2c3+c1s3  -s1c2s3+c1c3  s2s1]
        //            [-s2c3         s2s3         c2  ]

        R_matrix[0] = -cos(angle1) * cos(angle2) * cos(angle3) - sin(angle1) * sin(angle3);
        R_matrix[1] = cos(angle3) * sin(angle1) - sin(angle3) * cos(angle2) * cos(angle1);
        R_matrix[2] = sin(angle2) * cos(angle1);
        R_matrix[3] = sin(angle3) * cos(angle1) - cos(angle3) * cos(angle2) * sin(angle1);
        R_matrix[4] = -cos(angle1) * cos(angle3) - sin(angle1) * cos(angle2) * sin(angle3);
        R_matrix[5] = sin(angle2) * sin(angle1);
        R_matrix[6] = cos(angle3) * sin(angle2);
        R_matrix[7] = sin(angle3) * sin(angle2);
        R_matrix[8] = cos(angle2);
    }
    else if (eulerflag == 2)
    {
        /* R = Z1-X2-Z3, using Bunge convention; row major */
        // Z1-X2-Z3 = [c1c3-s1c2s3  -c1s3-s1c2c3  s1s2]
        //            [c1c2s3+c3s1  c1c2c3-s1s3  -c1s2]
        //            [s2s3          s2c3         c2  ]

        R_matrix[0] = -sin(angle1) * cos(angle2) * sin(angle3) + cos(angle1) * cos(angle3);
        R_matrix[1] = -sin(angle3) * cos(angle1) - cos(angle3) * cos(angle2) * sin(angle1);
        R_matrix[2] = sin(angle1) * sin(angle2);

        R_matrix[3] = cos(angle3) * sin(angle1) + sin(angle3) * cos(angle2) * cos(angle1);
        R_matrix[4] = -sin(angle1) * sin(angle3) + cos(angle1) * cos(angle2) * cos(angle3);
        R_matrix[5] = -cos(angle1) * sin(angle2);

        R_matrix[6] = sin(angle2) * sin(angle3);
        R_matrix[7] = sin(angle2) * cos(angle3);
        R_matrix[8] = cos(angle2);
    }

    if (lattice == 0) /* 2D square lattice (double layer neighbor) */
    {
        /* model parameters */
        hx = 2 * radius;
        hy = hx;
        particles_first_row = floor((box_t[1] - box_t[0]) / hx);
        rows = floor((box_t[3] - box_t[2]) / hy);

        nneighbors = 8;
        nneighbors1 = 4;
        nneighbors2 = 4;
        neighbor1_cutoff = 2.0 * radius;
        neighbor2_cutoff = 2.0 * sqrt(2.0) * radius;
        nneighbors_AFEM = 16;
        nneighbors_AFEM1 = 12;
        nneighbors_AFEM2 = 12;
        particle_volume = 4 * pow(radius, 2) * box_z; /* volume of unit cell */

        nparticle = 0;
        nparticle_t = particles_first_row * rows;
        xyz_t = allocDouble2D(nparticle_t, NDIM, 0.);

        /* initialize the particle positions*/
        k = 0;
        for (j = 1; j <= rows; j++)
        {
            y = box_t[2] + hy * (j - 1) + radius;
            for (i = 1; i <= particles_first_row; i++, k++)
            {
                x = box_t[0] + hx * (i - 1) + radius;
                if (k < nparticle_t)
                {
                    xyz_t[k][0] = x;
                    xyz_t[k][1] = y;
                }
            }
        }

        /* rotate the system */
        for (i = 0; i < nparticle_t; i++)
        {
            cblas_dgemv(layout, trans, NDIM, NDIM, blasAlpha, R_matrix, lda, xyz_t[i], incx, blasBeta, p, incy);

            if (p[0] >= box[0] && p[0] <= box[1] &&
                p[1] >= box[2] && p[1] <= box[3])
                nparticle++;
        }

        xyz = allocDouble2D(nparticle, NDIM, 0.);

        k = 0;
        for (i = 0; i < nparticle_t; i++)
        {
            cblas_dgemv(layout, trans, NDIM, NDIM, blasAlpha, R_matrix, lda, xyz_t[i], incx, blasBeta, p, incy);

            if (p[0] >= box[0] && p[0] <= box[1] &&
                p[1] >= box[2] && p[1] <= box[3])
            {
                xyz[k][0] = p[0];
                xyz[k][1] = p[1];
                k++;
            }
        }

        freeDouble2D(xyz_t, nparticle_t);
    }

    if (lattice == 1) /* 2D hexagon lattice (double layer neighbor) */
    {
        /* model parameters */
        hx = 2 * radius;
        hy = hx * sqrt(3.0) / 2.0;
        particles_first_row = 1 + floor((box_t[1] - box_t[0]) / hx);
        rows = 1 + floor((box_t[3] - box_t[2]) / hy);

        nneighbors = 12;
        nneighbors1 = 6;
        nneighbors2 = 6;
        neighbor1_cutoff = 2.0 * radius;
        neighbor2_cutoff = 2.0 * sqrt(3.0) * radius;
        nneighbors_AFEM = 30;
        nneighbors_AFEM1 = 18;
        nneighbors_AFEM2 = 18;
        particle_volume = 2 * sqrt(3) * pow(radius, 2) * box_z; /* volume of unit cell */

        nparticle = 0;
        nparticle_t = particles_first_row * floor((rows + 1) / 2.0) + (particles_first_row - 1) * (rows - floor((rows + 1) / 2.0));
        xyz_t = allocDouble2D(nparticle_t, NDIM, 0.);

        /* initialize the particle positions*/
        k = 0;
        for (j = 1; j <= rows; j++)
        {
            y = box_t[2] + hy * (j - 1);
            if (j % 2 == 1)
                for (i = 1; i <= particles_first_row; i++, k++)
                {
                    x = box_t[0] + hx * (i - 1);
                    if (k < nparticle_t)
                    {
                        xyz_t[k][0] = x;
                        xyz_t[k][1] = y;
                    }
                }
            else
                for (i = 1; i <= particles_first_row - 1; i++, k++)
                {
                    x = box_t[0] + hx * (2 * i - 1) / 2.0;
                    if (k < nparticle_t)
                    {
                        xyz_t[k][0] = x;
                        xyz_t[k][1] = y;
                    }
                }
        }

        /* rotate the system */
        for (i = 0; i < nparticle_t; i++)
        {
            cblas_dgemv(layout, trans, NDIM, NDIM, blasAlpha, R_matrix, lda, xyz_t[i], incx, blasBeta, p, incy);

            if (p[0] >= box[0] && p[0] <= box[1] &&
                p[1] >= box[2] && p[1] <= box[3])
                nparticle++;
        }

        xyz = allocDouble2D(nparticle, NDIM, 0.);

        k = 0;
        for (i = 0; i < nparticle_t; i++)
        {
            cblas_dgemv(layout, trans, NDIM, NDIM, blasAlpha, R_matrix, lda, xyz_t[i], incx, blasBeta, p, incy);

            if (p[0] >= box[0] && p[0] <= box[1] &&
                p[1] >= box[2] && p[1] <= box[3])
            {
                xyz[k][0] = p[0];
                xyz[k][1] = p[1];
                k++;
            }
        }

        freeDouble2D(xyz_t, nparticle_t);
    }

    if (lattice == 2) /* simple cubic */
    {
        /* model parameters */
        hx = 2 * radius;
        hy = hx;
        hz = hx;
        particles_first_row = 1 + (int)floor((box_t[1] - box_t[0]) / hx);
        rows = 1 + (int)floor((box_t[3] - box_t[2]) / hy);
        layers = 1 + (int)floor((box_t[5] - box_t[4]) / hz);

        nneighbors = 18;
        nneighbors1 = 6;
        nneighbors2 = 12;
        neighbor1_cutoff = 2.0 * radius;
        neighbor2_cutoff = 2.0 * sqrt(2.0) * radius;
        nneighbors_AFEM = 60;
        nneighbors_AFEM1 = 24;
        nneighbors_AFEM2 = 54;
        particle_volume = pow(2 * radius, 3); /* volume of unit cell */

        nparticle = 0;
        nparticle_t = (int)particles_first_row * rows * layers;
        xyz_t = allocDouble2D(nparticle_t, NDIM, 0);

        /* initialize the particle xyz*/
        n = 0;
        for (k = 1; k <= layers; k++)
        {
            z = box_t[4] + hz * (k - 1);
            for (j = 1; j <= rows; j++)
            {
                y = box_t[2] + hy * (j - 1);
                for (i = 1; i <= particles_first_row; i++, n++)
                {
                    x = box[0] + hx * (i - 1);
                    if (n < nparticle_t)
                    {
                        xyz_t[n][0] = x;
                        xyz_t[n][1] = y;
                        xyz_t[n][2] = z;
                    }
                }
            }
        }

        /* rotate and specify the system region */
        for (i = 0; i < nparticle_t; i++)
        {
            cblas_dgemv(layout, trans, 3, 3, blasAlpha, R_matrix, lda, xyz_t[i], incx, blasBeta, p, incy);

            if (p[0] >= box[0] && p[0] <= box[1] &&
                p[1] >= box[2] && p[1] <= box[3] &&
                p[2] >= box[4] && p[2] <= box[5])
                nparticle++;
        }

        xyz = allocDouble2D(nparticle, NDIM, 0);

        k = 0;
        for (i = 0; i < nparticle_t; i++)
        {
            cblas_dgemv(layout, trans, 3, 3, blasAlpha, R_matrix, lda, xyz_t[i], incx, blasBeta, p, incy);

            if (p[0] >= box[0] && p[0] <= box[1] &&
                p[1] >= box[2] && p[1] <= box[3] &&
                p[2] >= box[4] && p[2] <= box[5])
            {
                xyz[k][0] = p[0];
                xyz[k][1] = p[1];
                xyz[k][2] = p[2];
                k++;
            }
        }

        freeDouble2D(xyz_t, nparticle_t);

        // define the bond vectors
        double ref_bond_vectorLocal[18][3] = {
            {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}, {1 / sqrt(2), 1 / sqrt(2), 0}, {1 / sqrt(2), 0, 1 / sqrt(2)}, {0, 1 / sqrt(2), 1 / sqrt(2)}, {-1 / sqrt(2), -1 / sqrt(2), 0}, {-1 / sqrt(2), 0, -1 / sqrt(2)}, {0, -1 / sqrt(2), -1 / sqrt(2)}, {1 / sqrt(2), -1 / sqrt(2), 0}, {1 / sqrt(2), 0, -1 / sqrt(2)}, {0, 1 / sqrt(2), -1 / sqrt(2)}, {-1 / sqrt(2), 1 / sqrt(2), 0}, {-1 / sqrt(2), 0, 1 / sqrt(2)}, {0, -1 / sqrt(2), 1 / sqrt(2)}};

        bond_vector = allocDouble2D(nneighbors, dim, 0);
        for (i = 0; i < nneighbors; i++)
            cblas_dgemv(layout, trans, 3, 3, blasAlpha, R_matrix, lda, ref_bond_vectorLocal[i], incx, blasBeta, bond_vector[i], incy);
    }

    if (lattice == 3) /* face centered cubic  */
    {
        /* model parameters */
        hx = 4 * radius / sqrt(2.0);
        hy = hx;
        hz = hx;
        particles_first_row = 1 + (int)floor((box_t[1] - box_t[0]) / hx);
        rows = 1 + (int)floor((box_t[3] - box_t[2]) / (hy / 2.0));
        layers = 1 + (int)floor((box_t[5] - box_t[4]) / (hz / 2.0));

        nneighbors = 18;
        nneighbors1 = 12;
        nneighbors2 = 6;
        neighbor1_cutoff = 2.0 * radius;
        neighbor2_cutoff = 2.0 * sqrt(2.0) * radius;
        nneighbors_AFEM = 60;
        nneighbors_AFEM1 = 54;
        nneighbors_AFEM2 = 24;
        /* particle_volume = 0.25 * (15 + 7 * sqrt(5.0))* pow(2.0 * radius / sqrt(3.0), 3); volume of regular dodecahedron */
        particle_volume = 4.0 * sqrt(2.0) * pow(radius, 3); /* volume of unit cell 1, rhombic dodecahedron */
        /* particle_volume1 = 4.0 * sqrt(2.0) * pow(radius, 3); volume of unit cell 1 */
        /* particle_volume2 = 16.0 * sqrt(2.0) * pow(radius, 3); volume of unit cell 2 */

        nparticle = 0;
        nparticle_t = (int)(floor((layers + 1) / 2.0) * (particles_first_row * floor((rows + 1) / 2.0) + (particles_first_row - 1) * floor(rows / 2.0)) +
                            floor(layers / 2.0) * ((particles_first_row - 1) * floor((rows + 1) / 2.0) + particles_first_row * floor(rows / 2.0)));
        xyz_t = allocDouble2D(nparticle_t, 3, 0);

        /* initialize the particle xyz*/
        /* raise the system by (radius, radius, radius) */
        n = 0;
        for (k = 1; k <= layers; k++)
        {
            z = box_t[4] + hz / 2.0 * (k - 1);
            if (k % 2 == 1)
            {
                for (j = 1; j <= rows; j++)
                {
                    y = box_t[2] + hy / 2.0 * (j - 1);
                    if (j % 2 == 1)
                    {
                        for (i = 1; i <= particles_first_row; i++, n++)
                        {
                            x = box_t[0] + hx * (i - 1);
                            if (n < nparticle_t)
                            {
                                xyz_t[n][0] = x;
                                xyz_t[n][1] = y;
                                xyz_t[n][2] = z;
                            }
                        }
                    }
                    else
                    {
                        for (i = 1; i <= particles_first_row - 1; i++, n++)
                        {
                            x = box_t[0] + hx * (i - 1) + hx / 2.0;
                            if (n < nparticle_t)
                            {
                                xyz_t[n][0] = x;
                                xyz_t[n][1] = y;
                                xyz_t[n][2] = z;
                            }
                        }
                    }
                }
            }
            else
            {
                for (j = 1; j <= rows; j++)
                {
                    y = box_t[2] + hy / 2.0 * (j - 1);
                    if (j % 2 == 1)
                    {
                        for (i = 1; i <= particles_first_row - 1; i++, n++)
                        {
                            x = box_t[0] + hx * (i - 1) + hx / 2.0;
                            if (n < nparticle_t)
                            {
                                xyz_t[n][0] = x;
                                xyz_t[n][1] = y;
                                xyz_t[n][2] = z;
                            }
                        }
                    }
                    else
                    {
                        for (i = 1; i <= particles_first_row; i++, n++)
                        {
                            x = box_t[0] + hx * (i - 1);
                            if (n < nparticle_t)
                            {
                                xyz_t[n][0] = x;
                                xyz_t[n][1] = y;
                                xyz_t[n][2] = z;
                            }
                        }
                    }
                }
            }
        }

        /* rotate and specify the system region */
        for (i = 0; i < nparticle_t; i++)
        {
            cblas_dgemv(layout, trans, 3, 3, blasAlpha, R_matrix, lda, xyz_t[i], incx, blasBeta, p, incy);

            if (p[0] >= box[0] && p[0] <= box[1] &&
                p[1] >= box[2] && p[1] <= box[3] &&
                p[2] >= box[4] && p[2] <= box[5])
                nparticle++;
        }

        xyz = allocDouble2D(nparticle, 3, 0);

        k = 0;
        for (i = 0; i < nparticle_t; i++)
        {
            cblas_dgemv(layout, trans, 3, 3, blasAlpha, R_matrix, lda, xyz_t[i], incx, blasBeta, p, incy);

            if (p[0] >= box[0] && p[0] <= box[1] &&
                p[1] >= box[2] && p[1] <= box[3] &&
                p[2] >= box[4] && p[2] <= box[5])
            {
                //printf("%f, %f, %f\n", p[0], p[1], p[2]);
                xyz[k][0] = p[0];
                xyz[k][1] = p[1];
                xyz[k][2] = p[2];
                k++;
            }
        }

        freeDouble2D(xyz_t, nparticle_t);
    }

    if (lattice == 4) /* body centered cubic  */
    {
        /* model parameters */
        hx = 4.0 / sqrt(3.0) * radius;
        hy = hx;
        hz = hx;
        particles_first_row = 1 + (int)floor((box_t[1] - box_t[0]) / hx);
        rows = 1 + (int)floor((box_t[3] - box_t[2]) / hy);
        layers = 1 + (int)floor((box_t[5] - box_t[4]) / (hz / 2.0));

        nneighbors = 14;
        nneighbors1 = 8;
        nneighbors2 = 6;
        neighbor1_cutoff = 2.0 * radius;
        neighbor2_cutoff = 4.0 / sqrt(3.0) * radius;
        nneighbors_AFEM = 40;
        nneighbors_AFEM1 = 34;
        nneighbors_AFEM2 = 24;
        particle_volume = 32.0 * sqrt(3.0) / 9.0 * pow(radius, 3); /* volume of unit cell 1, truncated octahedron */

        nparticle = 0;
        nparticle_t = (int)floor((layers + 1) / 2.0) * (particles_first_row * rows) + (int)floor(layers / 2.0) * ((particles_first_row - 1) * (rows - 1));
        //printf("%d\n", nparticle_t);
        xyz_t = allocDouble2D(nparticle_t, 3, 0);

        /* initialize the particle xyz*/
        n = 0;
        for (k = 1; k <= layers; k++)
        {
            z = box_t[4] + hz / 2.0 * (k - 1);
            if (k % 2 == 1)
            {
                for (j = 1; j <= rows; j++)
                {
                    y = box_t[2] + hy * (j - 1);
                    for (i = 1; i <= particles_first_row; i++, n++)
                    {
                        x = box_t[0] + hx * (i - 1);
                        if (n < nparticle_t)
                        {
                            xyz_t[n][0] = x;
                            xyz_t[n][1] = y;
                            xyz_t[n][2] = z;
                        }
                    }
                }
            }
            else
            {
                for (j = 1; j <= rows - 1; j++)
                {
                    y = box_t[2] + hy * (j - 1) + hy / 2.0;
                    for (i = 1; i <= particles_first_row - 1; i++, n++)
                    {
                        x = box_t[0] + hx * (i - 1) + hx / 2.0;
                        if (n < nparticle_t)
                        {
                            //printf("%f %f %f\n", x, y, z);
                            xyz_t[n][0] = x;
                            xyz_t[n][1] = y;
                            xyz_t[n][2] = z;
                        }
                    }
                }
            }
        }

        /* rotate and specify the system region */
        for (i = 0; i < nparticle_t; i++)
        {
            cblas_dgemv(layout, trans, 3, 3, blasAlpha, R_matrix, lda, xyz_t[i], incx, blasBeta, p, incy);

            if (p[0] >= box[0] && p[0] <= box[1] &&
                p[1] >= box[2] && p[1] <= box[3] &&
                p[2] >= box[4] && p[2] <= box[5])
                nparticle++;
        }

        xyz = allocDouble2D(nparticle, 3, 0);

        k = 0;
        for (i = 0; i < nparticle_t; i++)
        {
            cblas_dgemv(layout, trans, 3, 3, blasAlpha, R_matrix, lda, xyz_t[i], incx, blasBeta, p, incy);

            if (p[0] >= box[0] && p[0] <= box[1] &&
                p[1] >= box[2] && p[1] <= box[3] &&
                p[2] >= box[4] && p[2] <= box[5])
            {
                //printf("%f, %f, %f\n", p[0], p[1], p[2]);
                xyz[k][0] = p[0];
                xyz[k][1] = p[1];
                xyz[k][2] = p[2];
                k++;
            }
        }

        freeDouble2D(xyz_t, nparticle_t);
    }
}

/* define the slip systems for crystal plasticity calculations */
void slipSysDefine3D()
{
    if (lattice == 3) /* face centered cubic  */
    {
        nslip_face = 4;                                            /* number of crystal slip planes */
        max_nslip_vector = 6;                                      /* max number of slip vectors */
        nslipSys = 24;                                             /* total number of slip systems */
        cp_gy = allocDouble3D(nparticle, nslipSys, 3, cp_tau0[0]); /* FCC has 1 type of slip systems */
        nslip_vector = allocInt1D(nslip_face, 6);                  /* number of slip directions on slip plane */

        /* local normal vectors of crystal slip planes */
        double slip_normalLocal[4][3] = {
            {1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)}, {-1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)}, {1 / sqrt(3), -1 / sqrt(3), 1 / sqrt(3)}, {-1 / sqrt(3), -1 / sqrt(3), 1 / sqrt(3)}};

        /* local slip vectors of 4*6=24 slip systems */
        double slip_vectorLocal[4][6][3] = {
            /* 1 */
            {{1 / sqrt(2), -1 / sqrt(2), 0}, {-1 / sqrt(2), 1 / sqrt(2), 0}, {-1 / sqrt(2), 0, 1 / sqrt(2)}, {1 / sqrt(2), 0, -1 / sqrt(2)}, {0, 1 / sqrt(2), -1 / sqrt(2)}, {0, -1 / sqrt(2), 1 / sqrt(2)}}, /* 2 */
            {{1 / sqrt(2), 0, 1 / sqrt(2)}, {-1 / sqrt(2), 0, -1 / sqrt(2)}, {-1 / sqrt(2), -1 / sqrt(2), 0}, {1 / sqrt(2), 1 / sqrt(2), 0}, {0, 1 / sqrt(2), -1 / sqrt(2)}, {0, -1 / sqrt(2), 1 / sqrt(2)}}, /* 3 */
            {{-1 / sqrt(2), 0, 1 / sqrt(2)}, {1 / sqrt(2), 0, -1 / sqrt(2)}, {0, -1 / sqrt(2), -1 / sqrt(2)}, {0, 1 / sqrt(2), 1 / sqrt(2)}, {1 / sqrt(2), 1 / sqrt(2), 0}, {-1 / sqrt(2), -1 / sqrt(2), 0}}, /* 4 */
            {{-1 / sqrt(2), 1 / sqrt(2), 0}, {1 / sqrt(2), -1 / sqrt(2), 0}, {1 / sqrt(2), 0, 1 / sqrt(2)}, {-1 / sqrt(2), 0, -1 / sqrt(2)}, {0, -1 / sqrt(2), -1 / sqrt(2)}, {0, 1 / sqrt(2), 1 / sqrt(2)}}};

        // if don't consider opposite slip directions
        //double slip_vectorLocal[4][3][3] = {
        //    { { 1 / sqrt(2), -1 / sqrt(2), 0 }, { -1 / sqrt(2), 0, 1 / sqrt(2) }, { 0, 1 / sqrt(2), -1 / sqrt(2) } }, \
        //    {{ 1 / sqrt(2), 0, 1 / sqrt(2) }, { -1 / sqrt(2), -1 / sqrt(2), 0 }, { 0, 1 / sqrt(2), -1 / sqrt(2) }}, \
        //    {{ -1 / sqrt(2), 0, 1 / sqrt(2) }, { 0, -1 / sqrt(2), -1 / sqrt(2) }, { 1 / sqrt(2), 1 / sqrt(2), 0 } }, \
        //    {{ -1 / sqrt(2), 1 / sqrt(2), 0 }, { 1 / sqrt(2), 0, 1 / sqrt(2) }, { 0, -1 / sqrt(2), -1 / sqrt(2) }} };

        CBLAS_LAYOUT layout = CblasRowMajor;
        CBLAS_TRANSPOSE trans = CblasNoTrans;

        int lda = 3, incx = 1, incy = 1;
        double blasAlpha = 1.0, blasBeta = 0.0;

        /* rotated normal vectors of crystal slip planes */
        slip_normal = allocDouble2D(nslip_face, 3, 0);
        for (int i = 0; i < nslip_face; i++)
            cblas_dgemv(layout, trans, 3, 3, blasAlpha, R_matrix, lda, slip_normalLocal[i], incx, blasBeta, slip_normal[i], incy);

        /* rotate the slip vector, for total 12 or 24 slip vectors in FCC crystal system */
        slip_vector = allocDouble3D(nslip_face, max_nslip_vector, 3, 0);
        for (int i = 0; i < nslip_face; i++)
        {
            for (int j = 0; j < nslip_vector[i]; j++)
                cblas_dgemv(layout, trans, 3, 3, blasAlpha, R_matrix, lda, slip_vectorLocal[i][j], incx, blasBeta, slip_vector[i][j], incy);
        }

        /* calculate the local and rotated system's Schmid tensor */
        schmid_tensor_local = allocDouble2D(nslipSys, 2 * NDIM, 0);
        schmid_tensor = allocDouble2D(nslipSys, 2 * NDIM, 0);

        int numSchmid = 0;
        for (int n = 0; n < nslip_face; n++) /* slip faces loop */
        {
            for (int k = 0; k < nslip_vector[n]; k++) /* slip directions loop */
            {
                /* s11, s22, s33, s23, s13, s12 */
                schmid_tensor_local[numSchmid][0] = slip_normalLocal[n][0] * slip_vectorLocal[n][k][0];
                schmid_tensor_local[numSchmid][1] = slip_normalLocal[n][1] * slip_vectorLocal[n][k][1];
                schmid_tensor_local[numSchmid][2] = slip_normalLocal[n][2] * slip_vectorLocal[n][k][2];
                schmid_tensor_local[numSchmid][3] = slip_normalLocal[n][1] * slip_vectorLocal[n][k][2] + slip_normalLocal[n][2] * slip_vectorLocal[n][k][1];
                schmid_tensor_local[numSchmid][4] = slip_normalLocal[n][0] * slip_vectorLocal[n][k][2] + slip_normalLocal[n][2] * slip_vectorLocal[n][k][0];
                schmid_tensor_local[numSchmid][5] = slip_normalLocal[n][0] * slip_vectorLocal[n][k][1] + slip_normalLocal[n][1] * slip_vectorLocal[n][k][0];

                schmid_tensor[numSchmid][0] = slip_normal[n][0] * slip_vector[n][k][0];
                schmid_tensor[numSchmid][1] = slip_normal[n][1] * slip_vector[n][k][1];
                schmid_tensor[numSchmid][2] = slip_normal[n][2] * slip_vector[n][k][2];
                schmid_tensor[numSchmid][3] = slip_normal[n][1] * slip_vector[n][k][2] + slip_normal[n][2] * slip_vector[n][k][1];
                schmid_tensor[numSchmid][4] = slip_normal[n][0] * slip_vector[n][k][2] + slip_normal[n][2] * slip_vector[n][k][0];
                schmid_tensor[numSchmid][5] = slip_normal[n][0] * slip_vector[n][k][1] + slip_normal[n][1] * slip_vector[n][k][0];

                numSchmid++;
            }
        }
    }
    else if (lattice == 4) /* body centered cubic (1 type of slip systems) */
    {
        nslip_face = 6;                                            /* number of slip planes for one particle */
        nslipSys = 24;                                             /* the total number of slip systems */
        max_nslip_vector = 4;                                      /* maximum slip vector on one slip plane */
        cp_gy = allocDouble3D(nparticle, nslipSys, 3, cp_tau0[0]); /* BCC has 2 or more types of slip systems */
        nslip_vector = allocInt1D(nslip_face, 4);                  /* number of slip directions on one slip plane */

        /* local normal vectors of crystal slip planes */
        double slip_normalLocal[6][3] = {
            {1. / sqrt(2.), 1. / sqrt(2.), 0.},
            {1. / sqrt(2.), -1. / sqrt(2.), 0.},
            {1. / sqrt(2.), 0., 1. / sqrt(2.)},
            {1. / sqrt(2.), 0., -1. / sqrt(2.)},
            {0., 1. / sqrt(2.), 1. / sqrt(2.)},
            {0., 1. / sqrt(2.), -1. / sqrt(2.)}};

        /* local slip vectors for all slip systems */
        double slip_vectorLocal[6][4][3] = {
            /* 1 */
            {{-1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)}, {1. / sqrt(3.), -1. / sqrt(3.), -1. / sqrt(3.)}, {1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.)}, {-1. / sqrt(3.), 1. / sqrt(3.), -1. / sqrt(3.)}},
            /* 2 */
            {{1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)}, {-1. / sqrt(3.), -1. / sqrt(3.), -1. / sqrt(3.)}, {1. / sqrt(3.), 1. / sqrt(3.), -1. / sqrt(3.)}, {-1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.)}},
            /* 3 */
            {{-1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)}, {1. / sqrt(3.), -1. / sqrt(3.), -1. / sqrt(3.)}, {1. / sqrt(3.), 1. / sqrt(3.), -1. / sqrt(3.)}, {-1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.)}},
            /* 4 */
            {{1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)}, {-1. / sqrt(3.), -1. / sqrt(3.), -1. / sqrt(3.)}, {1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.)}, {-1. / sqrt(3.), 1. / sqrt(3.), -1. / sqrt(3.)}},
            /* 5 */
            {{1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.)}, {-1. / sqrt(3.), 1. / sqrt(3.), -1. / sqrt(3.)}, {1. / sqrt(3.), 1. / sqrt(3.), -1. / sqrt(3.)}, {-1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.)}},
            /* 6 */
            {{1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)}, {-1. / sqrt(3.), -1. / sqrt(3.), -1. / sqrt(3.)}, {-1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)}, {1. / sqrt(3.), -1. / sqrt(3.), -1. / sqrt(3.)}}};

        CBLAS_LAYOUT layout = CblasRowMajor;
        CBLAS_TRANSPOSE trans = CblasNoTrans;

        int lda = 3, incx = 1, incy = 1;
        double blasAlpha = 1.0, blasBeta = 0.0;

        /* rotated normal vectors of crystal slip planes */
        slip_normal = allocDouble2D(nslip_face, 3, 0);
        for (int i = 0; i < nslip_face; i++)
            cblas_dgemv(layout, trans, 3, 3, blasAlpha, R_matrix, lda, slip_normalLocal[i], incx, blasBeta, slip_normal[i], incy);

        /* rotate the slip vector, for all slip systems */
        slip_vector = allocDouble3D(nslip_face, max_nslip_vector, 3, 0.);
        for (int i = 0; i < nslip_face; i++)
        {
            for (int j = 0; j < nslip_vector[i]; j++)
                cblas_dgemv(layout, trans, 3, 3, blasAlpha, R_matrix, lda, slip_vectorLocal[i][j], incx, blasBeta, slip_vector[i][j], incy);
        }

        /* calculate the local and rotated system's Schmid tensor */
        schmid_tensor_local = allocDouble2D(nslipSys, 2 * 3, 0);
        schmid_tensor = allocDouble2D(nslipSys, 2 * 3, 0);

        int numSchmid = 0;
        for (int n = 0; n < nslip_face; n++) /* slip faces loop */
        {
            for (int k = 0; k < nslip_vector[n]; k++) /* slip directions loop */
            {
                /* s11, s22, s33, s23, s13, s12 */
                schmid_tensor_local[numSchmid][0] = slip_normalLocal[n][0] * slip_vectorLocal[n][k][0];
                schmid_tensor_local[numSchmid][1] = slip_normalLocal[n][1] * slip_vectorLocal[n][k][1];
                schmid_tensor_local[numSchmid][2] = slip_normalLocal[n][2] * slip_vectorLocal[n][k][2];
                schmid_tensor_local[numSchmid][3] = slip_normalLocal[n][1] * slip_vectorLocal[n][k][2] + slip_normalLocal[n][2] * slip_vectorLocal[n][k][1];
                schmid_tensor_local[numSchmid][4] = slip_normalLocal[n][0] * slip_vectorLocal[n][k][2] + slip_normalLocal[n][2] * slip_vectorLocal[n][k][0];
                schmid_tensor_local[numSchmid][5] = slip_normalLocal[n][0] * slip_vectorLocal[n][k][1] + slip_normalLocal[n][1] * slip_vectorLocal[n][k][0];

                schmid_tensor[numSchmid][0] = slip_normal[n][0] * slip_vector[n][k][0];
                schmid_tensor[numSchmid][1] = slip_normal[n][1] * slip_vector[n][k][1];
                schmid_tensor[numSchmid][2] = slip_normal[n][2] * slip_vector[n][k][2];
                schmid_tensor[numSchmid][3] = slip_normal[n][1] * slip_vector[n][k][2] + slip_normal[n][2] * slip_vector[n][k][1];
                schmid_tensor[numSchmid][4] = slip_normal[n][0] * slip_vector[n][k][2] + slip_normal[n][2] * slip_vector[n][k][0];
                schmid_tensor[numSchmid][5] = slip_normal[n][0] * slip_vector[n][k][1] + slip_normal[n][1] * slip_vector[n][k][0];

                numSchmid++;
            }
        }
    }
    else if (lattice == 5) /* body centered cubic (2 types of slip systems) */
    {
        nslip_face = 18;                                           /* number of slip planes for one particle */
        nslipSys = 48;                                             /* the total number of slip systems */
        max_nslip_vector = 4;                                      /* maximum slip vector on one slip plane */
        cp_gy = allocDouble3D(nparticle, nslipSys, 3, cp_tau0[0]); /* BCC has 2 or more types of slip systems */
        for (int i = 0; i < nparticle; i++)
        {
            for (int j = 24; j < nslipSys; j++)
            {
                cp_gy[i][j][0] = cp_tau0[1];
                cp_gy[i][j][1] = cp_tau0[1];
            }
        }

        nslip_vector = allocInt1D(nslip_face, 4); /* number of slip directions on one slip plane */
        for (int i = 6; i < nslip_face; i++)
            nslip_vector[i] = 2;

        /* local normal vectors of crystal slip planes */
        double slip_normalLocal[18][3] = {
            {1. / sqrt(2.), 1. / sqrt(2.), 0.}, {1. / sqrt(2.), -1. / sqrt(2.), 0.}, {1. / sqrt(2.), 0., 1. / sqrt(2.)}, {1. / sqrt(2.), 0., -1. / sqrt(2.)}, {0., 1. / sqrt(2.), 1. / sqrt(2.)}, {0., 1. / sqrt(2.), -1. / sqrt(2.)}, {2. / sqrt(6.), 1. / sqrt(6.), 1. / sqrt(6.)}, {2. / sqrt(6.), 1. / sqrt(6.), -1. / sqrt(6.)}, {2. / sqrt(6.), -1. / sqrt(6.), 1. / sqrt(6.)}, {1. / sqrt(6.), 2. / sqrt(6.), 1. / sqrt(6.)}, {1. / sqrt(6.), 2. / sqrt(6.), -1. / sqrt(6.)}, {1. / sqrt(6.), 1. / sqrt(6.), 2. / sqrt(6.)}, {1. / sqrt(6.), 1. / sqrt(6.), -2. / sqrt(6.)}, {1. / sqrt(6.), -1. / sqrt(6.), 2. / sqrt(6.)}, {1. / sqrt(6.), -2. / sqrt(6.), 1. / sqrt(6.)}, {-1. / sqrt(6.), 2. / sqrt(6.), 1. / sqrt(6.)}, {-1. / sqrt(6.), 1. / sqrt(6.), 2. / sqrt(6.)}, {-2. / sqrt(6.), 1. / sqrt(6.), 1. / sqrt(6.)}};

        /* local slip vectors for all slip systems */
        double slip_vectorLocal[18][4][3] = {
            /* 1 */
            {{-1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)}, {1. / sqrt(3.), -1. / sqrt(3.), -1. / sqrt(3.)}, {1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.)}, {-1. / sqrt(3.), 1. / sqrt(3.), -1. / sqrt(3.)}}, /* 2 */
            {{1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)}, {-1. / sqrt(3.), -1. / sqrt(3.), -1. / sqrt(3.)}, {1. / sqrt(3.), 1. / sqrt(3.), -1. / sqrt(3.)}, {-1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.)}}, /* 3 */
            {{-1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)}, {1. / sqrt(3.), -1. / sqrt(3.), -1. / sqrt(3.)}, {1. / sqrt(3.), 1. / sqrt(3.), -1. / sqrt(3.)}, {-1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.)}}, /* 4 */
            {{1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)}, {-1. / sqrt(3.), -1. / sqrt(3.), -1. / sqrt(3.)}, {1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.)}, {-1. / sqrt(3.), 1. / sqrt(3.), -1. / sqrt(3.)}}, /* 5 */
            {{1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.)}, {-1. / sqrt(3.), 1. / sqrt(3.), -1. / sqrt(3.)}, {1. / sqrt(3.), 1. / sqrt(3.), -1. / sqrt(3.)}, {-1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.)}}, /* 6 */
            {{1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)}, {-1. / sqrt(3.), -1. / sqrt(3.), -1. / sqrt(3.)}, {-1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)}, {1. / sqrt(3.), -1. / sqrt(3.), -1. / sqrt(3.)}}, /* 7 */
            {{-1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)}, {1. / sqrt(3.), -1. / sqrt(3.), -1. / sqrt(3.)}, {0, 0, 0}, {0, 0, 0}},                                                                            /* 8 */
            {{1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.)}, {-1. / sqrt(3.), 1. / sqrt(3.), -1. / sqrt(3.)}, {0, 0, 0}, {0, 0, 0}},                                                                            /* 9 */
            {{1. / sqrt(3.), 1. / sqrt(3.), -1. / sqrt(3.)}, {-1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.)}, {0, 0, 0}, {0, 0, 0}},                                                                            /* 10 */
            {{1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.)}, {-1. / sqrt(3.), 1. / sqrt(3.), -1. / sqrt(3.)}, {0, 0, 0}, {0, 0, 0}},                                                                            /* 11 */
            {{-1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)}, {1. / sqrt(3.), -1. / sqrt(3.), -1. / sqrt(3.)}, {0, 0, 0}, {0, 0, 0}},                                                                            /* 12 */
            {{1. / sqrt(3.), 1. / sqrt(3.), -1. / sqrt(3.)}, {-1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.)}, {0, 0, 0}, {0, 0, 0}},                                                                            /* 13 */
            {{1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)}, {-1. / sqrt(3.), -1. / sqrt(3.), -1. / sqrt(3.)}, {0, 0, 0}, {0, 0, 0}},                                                                            /* 14 */
            {{-1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)}, {1. / sqrt(3.), -1. / sqrt(3.), -1. / sqrt(3.)}, {0, 0, 0}, {0, 0, 0}},                                                                            /* 15 */
            {{1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)}, {-1. / sqrt(3.), -1. / sqrt(3.), -1. / sqrt(3.)}, {0, 0, 0}, {0, 0, 0}},                                                                            /* 16 */
            {{1. / sqrt(3.), 1. / sqrt(3.), -1. / sqrt(3.)}, {-1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.)}, {0, 0, 0}, {0, 0, 0}},                                                                            /* 17 */
            {{1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.)}, {-1. / sqrt(3.), 1. / sqrt(3.), -1. / sqrt(3.)}, {0, 0, 0}, {0, 0, 0}},                                                                            /* 18 */
            {{1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)}, {-1. / sqrt(3.), -1. / sqrt(3.), -1. / sqrt(3.)}, {0, 0, 0}, {0, 0, 0}}};

        CBLAS_LAYOUT layout = CblasRowMajor;
        CBLAS_TRANSPOSE trans = CblasNoTrans;

        int lda = 3, incx = 1, incy = 1;
        double blasAlpha = 1.0, blasBeta = 0.0;

        /* rotated normal vectors of crystal slip planes */
        slip_normal = allocDouble2D(nslip_face, 3, 0);
        for (int i = 0; i < nslip_face; i++)
            cblas_dgemv(layout, trans, 3, 3, blasAlpha, R_matrix, lda, slip_normalLocal[i], incx, blasBeta, slip_normal[i], incy);

        /* rotate the slip vector, for all slip systems */
        slip_vector = allocDouble3D(nslip_face, max_nslip_vector, 3, 0.);
        for (int i = 0; i < nslip_face; i++)
        {
            for (int j = 0; j < nslip_vector[i]; j++)
                cblas_dgemv(layout, trans, 3, 3, blasAlpha, R_matrix, lda, slip_vectorLocal[i][j], incx, blasBeta, slip_vector[i][j], incy);
        }

        /* calculate the local and rotated system's Schmid tensor */
        schmid_tensor_local = allocDouble2D(nslipSys, 2 * 3, 0);
        schmid_tensor = allocDouble2D(nslipSys, 2 * 3, 0);

        int numSchmid = 0;
        for (int n = 0; n < nslip_face; n++) /* slip faces loop */
        {
            for (int k = 0; k < nslip_vector[n]; k++) /* slip directions loop */
            {
                /* s11, s22, s33, s23, s13, s12 */
                schmid_tensor_local[numSchmid][0] = slip_normalLocal[n][0] * slip_vectorLocal[n][k][0];
                schmid_tensor_local[numSchmid][1] = slip_normalLocal[n][1] * slip_vectorLocal[n][k][1];
                schmid_tensor_local[numSchmid][2] = slip_normalLocal[n][2] * slip_vectorLocal[n][k][2];
                schmid_tensor_local[numSchmid][3] = slip_normalLocal[n][1] * slip_vectorLocal[n][k][2] + slip_normalLocal[n][2] * slip_vectorLocal[n][k][1];
                schmid_tensor_local[numSchmid][4] = slip_normalLocal[n][0] * slip_vectorLocal[n][k][2] + slip_normalLocal[n][2] * slip_vectorLocal[n][k][0];
                schmid_tensor_local[numSchmid][5] = slip_normalLocal[n][0] * slip_vectorLocal[n][k][1] + slip_normalLocal[n][1] * slip_vectorLocal[n][k][0];

                schmid_tensor[numSchmid][0] = slip_normal[n][0] * slip_vector[n][k][0];
                schmid_tensor[numSchmid][1] = slip_normal[n][1] * slip_vector[n][k][1];
                schmid_tensor[numSchmid][2] = slip_normal[n][2] * slip_vector[n][k][2];
                schmid_tensor[numSchmid][3] = slip_normal[n][1] * slip_vector[n][k][2] + slip_normal[n][2] * slip_vector[n][k][1];
                schmid_tensor[numSchmid][4] = slip_normal[n][0] * slip_vector[n][k][2] + slip_normal[n][2] * slip_vector[n][k][0];
                schmid_tensor[numSchmid][5] = slip_normal[n][0] * slip_vector[n][k][1] + slip_normal[n][1] * slip_vector[n][k][0];

                numSchmid++;
            }
        }
    }
    else
    {
        nslipSys = 0; /* total number of slip systems */
        cp_gy = allocDouble3D(nparticle, nslipSys, 3, 0.0);
    }

    /* allocate memory for general crystal plasticity simulation */
    cp_RSS = allocDouble2D(nparticle, nslipSys, 0.);
    cp_Jact = allocInt2D(nparticle, nslipSys, 0); /* active slip system */
    cp_A = allocDouble2D(nparticle, 3, 0.);       /* accumulated plastic slip for individual slip systems */
    cp_A_single = allocDouble3D(nparticle, nslipSys, 3, 0.);
    cp_Cab = allocDouble2D(nparticle, nslipSys * nslipSys, 0.);

    /* incremental state variables */
    cp_dA_single = allocDouble2D(nparticle, nslipSys, 0.0);
    cp_dA = allocDouble1D(nparticle, 0.0);
    cp_dgy = allocDouble2D(nparticle, nslipSys, 0.0);
}

/* apply a rigid translation to the particle system */
void moveParticle(double *movexyz)
{
    double **xyz_t = allocDouble2D(nparticle, 3, 0.);

    int nparticle_t = 0;
    for (int i = 0; i < nparticle; i++)
    {
        double temp[3] = {0.};

        for (int j = 0; j < 3; j++)
            temp[j] = xyz[i][j] + movexyz[j];

        if (temp[0] >= box[0] && temp[0] <= box[1] &&
            temp[1] >= box[2] && temp[1] <= box[3] &&
            temp[2] >= box[4] && temp[2] <= box[5])
        {
            for (int j = 0; j < 3; j++)
                xyz_t[nparticle_t][j] = temp[j];

            nparticle_t++;
        }
    }

    freeDouble2D(xyz, nparticle);
    xyz = allocDouble2D(nparticle_t, 3, 0.);

    /* transfer to new position array */
    for (int i = 0; i < nparticle_t; i++)
    {
        for (int j = 0; j < 3; j++)
            xyz[i][j] = xyz_t[i][j];
    }

    freeDouble2D(xyz_t, nparticle);

    nparticle = nparticle_t;
}

/* remove particles inside the circle through c (x, y or z) direction */
void removeCircle(double *pc, double ra, char c)
{
    double **xyz_t = allocDouble2D(nparticle, 3, 0.);

    int nparticle_t = 0;
    for (int i = 0; i < nparticle; i++)
    {
        double dis2 = 0.0;
        if (c == 'x')
            dis2 = pow(xyz[i][1] - pc[1], 2.0) + pow(xyz[i][2] - pc[2], 2.0);
        else if (c == 'y')
            dis2 = pow(xyz[i][0] - pc[0], 2.0) + pow(xyz[i][2] - pc[2], 2.0);
        else if (c == 'z')
            dis2 = pow(xyz[i][0] - pc[0], 2.0) + pow(xyz[i][1] - pc[1], 2.0);
        if (dis2 > ra * ra)
        {
            for (int j = 0; j < 3; j++)
                xyz_t[nparticle_t][j] = xyz[i][j];

            nparticle_t++;
        }
    }

    freeDouble2D(xyz, nparticle);
    xyz = allocDouble2D(nparticle_t, 3, 0.);

    /* transfer to new position array */
    for (int i = 0; i < nparticle_t; i++)
    {
        for (int j = 0; j < 3; j++)
            xyz[i][j] = xyz_t[i][j];
    }

    freeDouble2D(xyz_t, nparticle);

    nparticle = nparticle_t;
}

/* remove particles inside the circle part through z direction, theta1 < theta2 */
/* theta1 and theta2 should be both positive or both negative */
void removeCirclePartZ(double *pc, double ra, double theta1, double theta2)
{
    double **xyz_t = allocDouble2D(nparticle, 3, 0.);

    int nparticle_t = 0;
    for (int i = 0; i < nparticle; i++)
    {
        double dx = 0, dy = 0;
        dx = xyz[i][0] - pc[0];
        dy = xyz[i][1] - pc[1];
        if (fabs(dx) < EPS && fabs(dy) < EPS)
            continue;
        else
        {
            double theta = atan2(dy, dx);
            double dis2 = pow(xyz[i][0] - pc[0], 2.0) + pow(xyz[i][1] - pc[1], 2.0);
            if (dis2 > ra * ra || (theta >= theta1 && theta <= theta2))
            {
                for (int j = 0; j < 3; j++)
                    xyz_t[nparticle_t][j] = xyz[i][j];

                nparticle_t++;
            }
        }
    }

    freeDouble2D(xyz, nparticle);
    xyz = allocDouble2D(nparticle_t, 3, 0.);

    /* transfer to new position array */
    for (int i = 0; i < nparticle_t; i++)
    {
        for (int j = 0; j < 3; j++)
            xyz[i][j] = xyz_t[i][j];
    }

    freeDouble2D(xyz_t, nparticle);

    nparticle = nparticle_t;
}

// remove a block with size: xlo=r0, xhi=r1, ylo=r2, yhi=r3, zlo=r4, zhi=r5
void removeBlock(double r0, double r1, double r2, double r3, double r4, double r5)
{
    double **xyz_t = allocDouble2D(nparticle, 3, 0.);

    int nparticle_t = 0;
    for (int i = 0; i < nparticle; i++)
    {
        if (xyz[i][0] < r0 || xyz[i][0] > r1 || xyz[i][1] < r2 || xyz[i][1] > r3 || xyz[i][2] < r4 || xyz[i][2] > r5)
        {
            for (int j = 0; j < 3; j++)
                xyz_t[nparticle_t][j] = xyz[i][j];

            nparticle_t++;
        }
    }

    freeDouble2D(xyz, nparticle);
    xyz = allocDouble2D(nparticle_t, 3, 0.);

    /* transfer to new position array */
    for (int i = 0; i < nparticle_t; i++)
    {
        for (int j = 0; j < 3; j++)
            xyz[i][j] = xyz_t[i][j];
    }

    freeDouble2D(xyz_t, nparticle);

    nparticle = nparticle_t;
}

// create a cylinder from the original cuboid configuration
void createCylinderz(double *pc, double ra)
{
    double **xyz_t = allocDouble2D(nparticle, 3, 0.);

    int nparticle_t = 0;
    for (int i = 0; i < nparticle; i++)
    {
        if (pow(xyz[i][0] - pc[0], 2.0) + pow(xyz[i][1] - pc[1], 2.0) < ra * ra)
        {
            for (int j = 0; j < 3; j++)
                xyz_t[nparticle_t][j] = xyz[i][j];

            nparticle_t++;
        }
    }

    freeDouble2D(xyz, nparticle);
    xyz = allocDouble2D(nparticle_t, 3, 0.);

    /* transfer to new position array */
    for (int i = 0; i < nparticle_t; i++)
    {
        for (int j = 0; j < 3; j++)
            xyz[i][j] = xyz_t[i][j];
    }

    freeDouble2D(xyz_t, nparticle);

    nparticle = nparticle_t;
}

// remove a ring from the initial configuration, normal direction of the ring is along z direction
void removeRingz(double *pc, double R, double r)
{
    double **xyz_t = allocDouble2D(nparticle, 3, 0.);

    int nparticle_t = 0;
    for (int i = 0; i < nparticle; i++)
    {
        double h = xyz[i][2] - pc[2];
        if (fabs(h) < r)
        {
            if (pow(xyz[i][0] - pc[0], 2.0) + pow(xyz[i][1] - pc[1], 2.0) < pow(R - sqrt(r * r - h * h), 2.0))
            {
                for (int j = 0; j < 3; j++)
                    xyz_t[nparticle_t][j] = xyz[i][j];

                nparticle_t++;
            }
        }
        else
        {
            for (int j = 0; j < 3; j++)
                xyz_t[nparticle_t][j] = xyz[i][j];

            nparticle_t++;
        }
    }

    freeDouble2D(xyz, nparticle);
    xyz = allocDouble2D(nparticle_t, 3, 0.);

    /* transfer to new position array */
    for (int i = 0; i < nparticle_t; i++)
    {
        for (int j = 0; j < 3; j++)
            xyz[i][j] = xyz_t[i][j];
    }

    freeDouble2D(xyz_t, nparticle);

    nparticle = nparticle_t;
}

/* Create an initial crack which starts at edge, with length a, width 2w and height h, (perpendicular to xOy plane) */
void createCrack(double a1, double a2, double w, double h)
{
    double **xyz_t = allocDouble2D(nparticle, 3, 0.);

    int nparticle_t = 0;
    for (int i = 0; i < nparticle; i++)
    {
        if (xyz[i][0] < a1 || xyz[i][0] > a2 || xyz[i][1] > h + w || xyz[i][1] < h - w)
        {
            // for (int j = 0; j < 3; j++)
            //     xyz_t[nparticle_t][j] = xyz[i][j];

            // nparticle_t++;

            // with cycle
            double dis2 = pow(xyz[i][0] - a2, 2.0) + pow(xyz[i][1] - h, 2.0);
            if (dis2 > w * w)
            {
                for (int j = 0; j < 3; j++)
                    xyz_t[nparticle_t][j] = xyz[i][j];

                nparticle_t++;
            }
        }
    }
    freeDouble2D(xyz, nparticle);
    xyz = allocDouble2D(nparticle_t, 3, 0.);

    /* transfer to new position array */
    for (int i = 0; i < nparticle_t; i++)
    {
        for (int j = 0; j < 3; j++)
            xyz[i][j] = xyz_t[i][j];
    }
    freeDouble2D(xyz_t, nparticle);
    nparticle = nparticle_t;
}

/* define the initial crack, length=|a1-a2|, height=h */
void defineCrack(double a1, double a2, double h)
{
    for (int i = 0; i < nparticle; i++)
    {
        damage_visual[i] = 0.0;
        for (int j = 0; j < nb_initial[i]; j++)
        {
            if ((xyz[neighbors[i][j]][1] - h) * (xyz[i][1] - h) < 0 && xyz[i][0] > a1 && xyz[i][0] < a2 &&
                xyz[neighbors[i][j]][0] > a1 && xyz[neighbors[i][j]][0] < a2)
            {
                damage_D[i][j][0] = 1.0;
                damage_D[i][j][1] = 1.0;
                damage_broken[i][j] = 0.0;
                damage_w[i][j] = 0.0;
                nb[i] -= 1;
            }
            damage_visual[i] += damage_broken[i][j];
        }
        damage_visual[i] = 1 - damage_visual[i] / nb_initial[i];
    }
}

/* allocate memories for some global matrices */
void initMatrices()
{
    disp = allocDouble1D(dim * nparticle, 0);        /* global displacement vector */
    xyz_initial = allocDouble2D(nparticle, NDIM, 0); /* store coordinate information as 3D */
    xyz_temp = allocDouble2D(nparticle, NDIM, 0);

    distance = allocDouble2D(nparticle, nneighbors, 0.0);
    distance_initial = allocDouble2D(nparticle, nneighbors, 0.0);
    csx = allocDouble2D(nparticle, nneighbors, 0.0);
    csy = allocDouble2D(nparticle, nneighbors, 0.0);
    csz = allocDouble2D(nparticle, nneighbors, 0.0);
    csx_initial = allocDouble2D(nparticle, nneighbors, 0.0);
    csy_initial = allocDouble2D(nparticle, nneighbors, 0.0);
    csz_initial = allocDouble2D(nparticle, nneighbors, 0.0);

    // bond geometric measure
    dL = allocDouble2D(nparticle, nneighbors, 0.0); /* total bond stretch */
    dL_total = allocDouble2D(nparticle, 2, 0);
    TdL_total = allocDouble2D(nparticle, 2, 0);
    dL_ave = allocDouble2D(nparticle, nneighbors, 0.); /* average bond stretch */
    dLp = allocDouble3D(nparticle, nneighbors, 3, 0.); /* total plastic bond stretch */
    ddL = allocDouble2D(nparticle, nneighbors, 0.0);
    ddL_total = allocDouble2D(nparticle, 2, 0);
    TddL_total = allocDouble2D(nparticle, 2, 0);
    ddLp = allocDouble2D(nparticle, nneighbors, 0.);       /* incremental plastic bond stretch */
    bond_stress = allocDouble2D(nparticle, nneighbors, 0); /* bond stress, projection of stress tensor */
    bond_stretch = allocDouble2D(nparticle, nneighbors, 0);

    state_v = allocInt1D(nparticle, 0);
    pl_flag = allocInt1D(nparticle, 0); /* denote whether the plastic deformtion has been calculated, to avoid repetition */

    nb = allocInt1D(nparticle, -1);         /* number of normal bonds */
    nb_initial = allocInt1D(nparticle, -1); /* initial number of normal bonds */
    nb_conn = allocInt1D(nparticle, -1);    /* number of connections, including itself */
    neighbors = allocInt2D(nparticle, nneighbors, -1);
    neighbors1 = allocInt2D(nparticle, nneighbors1, -1);
    neighbors2 = allocInt2D(nparticle, nneighbors2, -1);
    nsign = allocInt2D(nparticle, nneighbors, -1);
    conn = allocInt2D(nparticle, nneighbors_AFEM + 1, -1); /* connection of AFEM particles */

    K_pointer = allocInt2D(nparticle + 1, 2, 0);

    residual = allocDouble1D(dim * nparticle, 0);  /* right hand side, residual */
    Pin = allocDouble1D(NDIM * nparticle, 0);      /* total internal force, fixed 3 dimension */
    Pex = allocDouble1D(dim * nparticle, 0);       /* total external force */
    Pex_temp = allocDouble1D(dim * nparticle, 0);  /* temp external force */
    dispBC_index = allocInt1D(dim * nparticle, 1); /* disp info for each degree of freedom, 0 as being applied disp BC */
    fix_index = allocInt1D(dim * nparticle, 1);    /* fix info for each degree of freedom, 0 as being fixed */

    Kn = allocDouble2D(nparticle, nneighbors, 0.0);
    Tv = allocDouble2D(nparticle, nneighbors, 0.0);

    stress_tensor = allocDouble2D(nparticle, 2 * NDIM, 0.);
    strain_tensor = allocDouble2D(nparticle, 2 * NDIM, 0.);

    // damage parameters
    damage_broken = allocDouble2D(nparticle, nneighbors, 1.); /* crack index parameter */
    damage_D = allocDouble3D(nparticle, nneighbors, 2, 0.);   /* bond-associated damage parameter, initially be 0, state variable */
    damage_w = allocDouble2D(nparticle, nneighbors, 1.);      /* intact parameter, apply on bond force */
    damage_visual = allocDouble1D(nparticle, 0.0);            /* damage indicator for visualization */
    damage_local = allocDouble2D(nparticle, 2, 0.0);          /* damage variable used in continuum damage mechanics */
    damage_nonlocal = allocDouble2D(nparticle, 2, 0.0);       /* nonlocal damage variable used in continuum damage mechanics */

    F = allocDouble2D(nparticle, nneighbors, 0.);          /* FIJ is total bond force between I and J */
    F_temp = allocDouble2D(nparticle, nneighbors, 0.);     /* F_tempIJ stores the bond force at last time step */
    bond_force = allocDouble2D(nparticle, nneighbors, 0.); /* Organized bond force in order*/

    J2_beta = allocDouble3D(nparticle, 2 * NDIM, 3, 0.); /* back stress tensor */
    J2_alpha = allocDouble2D(nparticle, 3, 0.);          /* accumulated equivalent plastic strain */
    J2_beta_eq = allocDouble2D(nparticle, 3, 0.);        /* equivalent back stress */
    J2_stresseq = allocDouble1D(nparticle, 0.);          /* von-Mises equivalent stress */
    J2_stressm = allocDouble1D(nparticle, 0.);           /* hydrostatic stress */
    J2_triaxiality = allocDouble1D(nparticle, 0.);       /* ratio of the hydrostatic stress to von-Mises equivalent stress */
    J2_dlambda = allocDouble1D(nparticle, 0.);           /* incremental equivalent plastic strain */
}
