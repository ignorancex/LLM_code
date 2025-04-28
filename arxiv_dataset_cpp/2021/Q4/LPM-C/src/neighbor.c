#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "lpm.h"

/* search the neighbor for the particle system, without a crack */
void searchNormalNeighbor()
{
#pragma omp parallel for
    for (int i = 0; i < nparticle; i++)
    {
        int index1 = 0, index2 = 0, index3 = 0;

        for (int j = 0; j < nparticle; j++)
        {
            double dis = sqrt(pow((xyz[j][0] - xyz[i][0]), 2) + pow((xyz[j][1] - xyz[i][1]), 2) + pow((xyz[j][2] - xyz[i][2]), 2));

            if ((dis < 1.01 * neighbor1_cutoff) && (j != i)) /* The first nearest neighbors */
            {
                neighbors1[i][index1++] = j;
                csx_initial[i][index3] = (xyz[i][0] - xyz[j][0]) / dis;
                csy_initial[i][index3] = (xyz[i][1] - xyz[j][1]) / dis;
                csz_initial[i][index3] = (xyz[i][2] - xyz[j][2]) / dis;
                distance_initial[i][index3] = dis;

                nsign[i][index3] = 0;
                neighbors[i][index3++] = j;
            }
            else if ((dis > 1.01 * neighbor1_cutoff) && (dis < 1.01 * neighbor2_cutoff)) /* The second nearest neighbors */
            {
                neighbors2[i][index2++] = j;
                csx_initial[i][index3] = (xyz[i][0] - xyz[j][0]) / dis;
                csy_initial[i][index3] = (xyz[i][1] - xyz[j][1]) / dis;
                csz_initial[i][index3] = (xyz[i][2] - xyz[j][2]) / dis;
                distance_initial[i][index3] = dis;

                nsign[i][index3] = 1;
                neighbors[i][index3++] = j;
            }
        }
        nb[i] = index3;
        nb_initial[i] = index3;
    }
}

/* search the AFEM neighbors, only once before the simulation starts */
int searchAFEMNeighbor()
{
    int* temp = allocInt1D(nneighbors_AFEM + 1, -1);
    int** collection = allocInt2D(nparticle + 1, nneighbors * nneighbors, -1);
    int** collectionInitial = allocInt2D(nparticle + 1, nneighbors * nneighbors, -1);

#pragma omp parallel for
    for (int i = 0; i < nparticle; i++) /* Collecting the afem neighbors for each particle */
    {
        int index2 = 0, index3 = 0, index4 = 0;
        for (int j = 0; j < nb[i]; j++)
        {
            if (nsign[i][j] == 0)
            {
                int nb2 = countNEqual(neighbors1[neighbors1[i][index3]], nneighbors1, -1);
                collection[i][index2++] = neighbors1[i][index3];
                for (int k = 0; k < nb2; k++)
                {
                    collection[i][index2++] = neighbors1[neighbors1[i][index3]][k];
                }
                index3++;
            }
            else if (nsign[i][j] == 1)
            {
                int nb2 = countNEqual(neighbors2[neighbors2[i][index4]], nneighbors2, -1);
                collection[i][index2++] = neighbors2[i][index4];
                for (int m = 0; m < nb2; m++)
                {
                    collection[i][index2++] = neighbors2[neighbors2[i][index4]][m];
                }
                index4++;
            }
        }
    }

#pragma omp parallel for
    for (int i = 0; i < nparticle; i++) /* Remove the duplicates in the collection matrix, obtain the conn matrix */
    {
        int nb1 = 1;
        conn[i][0] = collection[i][0];
        int nb2 = countNEqual(collection[i], nneighbors * nneighbors, -1);
        for (int j = 1; j < nb2; j++)
        {
            for (int k = 0; k < nb1; k++)
            {
                if (conn[i][k] == collection[i][j])
                    goto outer4;
            }
            conn[i][nb1++] = collection[i][j];
        outer4:;
        }
    }

    /* Sorting the conn matrix */
    for (int i = 0; i < nparticle; i++)
    {
        int nb1 = countNEqual(conn[i], nneighbors_AFEM + 1, -1);
        nb_conn[i] = nb1;
        for (int j = 0; j < nb1; j++)
            temp[j] = conn[i][j];
        sortInt(temp, nb1);
        for (int j = 0; j < nb1; j++)
            conn[i][j] = temp[j];
    }

    K_pointer[0][1] = 0;
    for (int i = 0; i < nparticle; i++)
    {
        int index1 = 0;
        for (int j = 0; j < nb_conn[i]; j++)
        {
            if (conn[i][j] >= i)
                index1++;
        }
        /* Numbers of conn larger than (or equal to) its own index for each particle */
        K_pointer[i][0] = index1;
        /* Start index for each particle in the global stiffness matrix */
        if (dim == 2)
            K_pointer[i + 1][1] = K_pointer[i][1] + dim * dim * index1 - 1;
        else if (dim == 3)
            K_pointer[i + 1][1] = K_pointer[i][1] + dim * dim * index1 - 3;
    }

    JK = allocInt1D(K_pointer[nparticle][1], -1);
    IK = allocInt1D(dim * nparticle + 1, -1);
    K_global = allocDouble1D(K_pointer[nparticle][1], -1);

    free(temp);
    freeInt2D(collection, nparticle + 1);
    freeInt2D(collectionInitial, nparticle + 1);

    return K_pointer[nparticle][1];
}
