#include "expmpa.hpp"


void matExpPadeApproximant::expPadeApprox(INTE_TYPE scaling_order)
{
    // Labelled as A, A2, A4, A6, W1, W2, Z1, Z2, W, U, V, R, W, where W is the workspace for getting powers in exp(A) = R^{2^s}

    // A is assumed to be given in Ms[0];

    s = scaling_order;
    INTE_TYPE scale = pow(2, s);

    const INTE_TYPE* bvec = _EXPM_PADE_APPROXIMANT;


    scal(1.0 / scale, *Ms[0]);

    REAL_TYPE x1, x2, x3;


    my_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *Ms[0], *Ms[0], 0.0, *Ms[1]);                     // Ms[1] <- A2 = Ms[0] * Ms[0]
    my_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *Ms[1], *Ms[1], 0.0, *Ms[2]);                     // Ms[2] <- A4 = Ms[1] * Ms[1]
    my_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *Ms[1], *Ms[2], 0.0, *Ms[3]);                     // Ms[3] <- A6 = Ms[1] * Ms[2]
    
    for (INTE_TYPE mat_ind = 0; mat_ind < d * d; mat_ind++)
    {
        x1 = Ms[1]->v[mat_ind];
        x2 = Ms[2]->v[mat_ind];
        x3 = Ms[3]->v[mat_ind];
        Ms[4]->v[mat_ind] = x1 * bvec[9] + x2 * bvec[11] + x3 * bvec[13];                       // Ms[4] <- W1 = b9A2 + b11A4 + b13A6
        Ms[5]->v[mat_ind] = x1 * bvec[3] + x2 * bvec[5] + x3 * bvec[7];                         // Ms[5] <- W2 = (b1I +) b3A2 + b5A4 + b7A6
        Ms[6]->v[mat_ind] = x1 * bvec[8] + x2 * bvec[10] + x3 * bvec[12];                       // Ms[6] <- Z1 = b8A2 + b10A4 + b12A6
        Ms[7]->v[mat_ind] = x1 * bvec[2] + x2 * bvec[4] + x3 * bvec[6];                         // Ms[7] <- Z2 = (b0I +) b2A2 + b4A4 + b6A6
    }
    for (INTE_TYPE col_ind = 0; col_ind < d; col_ind++)
    {
        Ms[5]->v[col_ind + col_ind * d] += bvec[1];
        Ms[7]->v[col_ind + col_ind * d] += bvec[0];
    }

    memcpy(Ms[8]->v, Ms[5]->v, sizeof(REAL_TYPE) * d * d);                                      // Ms[8] <- W = A6W1 + W2
    my_dgemm(CblasNoTrans, CblasNoTrans, 1.0, (*Ms[3]), (*Ms[4]), 1.0, (*Ms[8]));               // Ms[8] <- W = A6W1 + W2
    my_dgemm(CblasNoTrans, CblasNoTrans, 1.0, (*Ms[0]), (*Ms[8]), 0.0, (*Ms[9]));               // Ms[9] <- U = AW

    memcpy(Ms[10]->v, Ms[7]->v, sizeof(REAL_TYPE) * d * d);                                     // Ms[10] <- V = A6Z1 + Z2
    my_dgemm(CblasNoTrans, CblasNoTrans, 1.0, (*Ms[3]), (*Ms[6]), 1.0, (*Ms[10]));              // Ms[10] <- V = A6Z1 + Z2

    for (INTE_TYPE mat_ind = 0; mat_ind < d * d; mat_ind++)
        plu.M.v[mat_ind] = Ms[10]->v[mat_ind] - Ms[9]->v[mat_ind];                              // LU <- V - U
    plu.f = false;
    plu.computePLU();

    for (INTE_TYPE mat_ind = 0; mat_ind < d * d; mat_ind++)
        Ms[11]->v[mat_ind] = Ms[9]->v[mat_ind] + Ms[10]->v[mat_ind];                            // Ms[11] <- R = (V - U)^{-1}(U + V)
    solverPLU(Ms[11]->v, d, d, plu);                                                            // Ms[11] <- R = (V - U)^{-1}(U + V)



    // for (INTE_TYPE mat_ind = 0; mat_ind < d * d; mat_ind++)
    //     Ms[12]->v[mat_ind] = 0;                                                                 // Ms[12] <- Q = exp(A)
    // for (INTE_TYPE col_ind = 0; col_ind < d; col_ind++)
    //     Ms[12]->v[col_ind + col_ind * d] = 1.0;                                                 // Ms[12] <- Q = exp(A)
}