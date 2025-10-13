#include "dexpPade.hpp"

void dexpPadeApprox::_dexp_pade_low()
{
    _check_and_initialize(VecMatE, _vmsize, d);

    // For m = 3, 5, 7, 9
    // The required VecMatD sizes are 7, 8, 9, 10, stored as follows
    // E, E_2, E_4, ..., E_{m-1}
    // LW, LU, LV
    // LU-LV, L (V - U)L = LU + LV + (LU-LV)*Q
    // The respective VecMatM are
    // A, A_2, A_4, ..., A_{m-1}
    // W, U, V
    // U+V, Q

    const INTE_TYPE *bvec;

    INTE_TYPE k = (m - 1) / 2;
    switch (m)
    {
    case 3:
        bvec = _EXPM_PADE_APPROX_3;
        break;
    case 5:
        bvec = _EXPM_PADE_APPROX_5;
        break;
    case 7:
        bvec = _EXPM_PADE_APPROX_7;
        break;
    case 9:
        bvec = _EXPM_PADE_APPROX_9;
        break;
    default:
        bvec = _EXPM_PADE_APPROX_13;

        break;
    }

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatM[0]->v, d, VecMatE[0]->v, d, 0.0, VecMatE[1]->v, d);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatE[0]->v, d, VecMatM[0]->v, d, 1.0, VecMatE[1]->v, d);
    for (auto ind = 1; ind < k; ind++)
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatE[1]->v, d, VecMatM[ind]->v, d, 0.0, VecMatE[ind + 1]->v, d);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatM[1]->v, d, VecMatE[ind]->v, d, 1.0, VecMatE[ind + 1]->v, d);
    }

    // Getting the A's in the VecMatM[0 : k]
    VecMatE[k + 1]->Zero(); // W for odd terms
    VecMatE[k + 2]->Zero(); // U, the even terms
    for (auto ind = 0; ind < k; ind++)
    {
        cblas_daxpy(d * d, bvec[ind * 2 + 3], VecMatE[ind + 1]->v, 1, VecMatE[k + 1]->v, 1); // LW += b_{2i+1} * E_{2i}
        cblas_daxpy(d * d, bvec[ind * 2 + 2], VecMatE[ind + 1]->v, 1, VecMatE[k + 2]->v, 1); // LV += b_{2i} * E_{2i}
    }

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatM[0]->v, d, VecMatE[k + 1]->v, d, 0.0, VecMatM[k + 3]->v, d); // LU = A*LW
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatE[0]->v, d, VecMatM[k + 1]->v, d, 1.0, VecMatM[k + 3]->v, d); // LU = A*LW + E*W

    memcpy(VecMatE[k + 4]->v, VecMatE[k + 3]->v, sizeof(REAL_TYPE) * d * d);                                                                     // E[k+4] <- LU
    cblas_daxpy(d * d, -1.0, VecMatE[k + 2]->v, 1, VecMatE[k + 4]->v, 1);                                                                        // E[k+4] <- LU - LV
    memcpy(VecMatE[k + 5]->v, VecMatE[k + 3]->v, sizeof(REAL_TYPE) * d * d);                                                                     // E[k+5] <- LU
    cblas_daxpy(d * d, 1.0, VecMatE[k + 2]->v, 1, VecMatE[k + 5]->v, 1);                                                                         // E[k+5] <- LU + LV
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatE[k + 4]->v, d, VecMatM[k + 5]->v, d, 1.0, VecMatE[k + 5]->v, d); // Es[12] <- LU + LV + (LU - LV) * R
    PLU.Solve(VecMatE[k + 5]->v, d, d);
}

void dexpPadeApprox::_dexp_pade_13()
{
    _check_and_initialize(VecMatE, _vmsize, d);

    // ind:     0       1       2       3       4       5       6       7       8       9       10      11       12
    // Ms       A       A2      A4      A6      W1      W2      Z1      Z2      W       U       V       U + V    Q
    // Es       E       E2      E4      E6      LW1     LW2     LZ1     LZ2     LW      LU      LV      LU-LV    L in (V - U)L = LU + LV + (LU-LV)*Q

    // M is given before the entry.
    const INTE_TYPE *bvec = _EXPM_PADE_APPROX_13;
    REAL_TYPE x1, x2, x3;

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatM[0]->v, d, VecMatE[0]->v, d, 0.0, VecMatE[1]->v, d); // Ws[1] <- M2 = AM + MA
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatE[0]->v, d, VecMatM[0]->v, d, 1.0, VecMatE[1]->v, d); // Ws[1] <- M2 = AM + MA

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatM[1]->v, d, VecMatE[1]->v, d, 0.0, VecMatE[2]->v, d); // Ws[2] <- M4 = A2M2 + M2A2
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatE[1]->v, d, VecMatM[1]->v, d, 1.0, VecMatE[2]->v, d); // Ws[2] <- M4 = A2M2 + M2A2

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatM[2]->v, d, VecMatE[1]->v, d, 0.0, VecMatE[3]->v, d); // Ws[3] <- M6 = A4M2 + M4A2
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatE[2]->v, d, VecMatM[1]->v, d, 1.0, VecMatE[3]->v, d); // Ws[3] <- M6 = A4M2 + M4A2

    for (INTE_TYPE mat_ind = 0; mat_ind < d * d; mat_ind++)
    {
        x1 = VecMatE[1]->v[mat_ind];
        x2 = VecMatE[2]->v[mat_ind];
        x3 = VecMatE[3]->v[mat_ind];
        VecMatE[4]->v[mat_ind] = x1 * bvec[9] + x2 * bvec[11] + x3 * bvec[13]; // Ws[4] <- LW1 = b9M2 + b11M4 + b13M6
        VecMatE[5]->v[mat_ind] = x1 * bvec[3] + x2 * bvec[5] + x3 * bvec[7];   // Ws[5] <- LW2 = b3M2 + b5M4 + b7M6
        VecMatE[6]->v[mat_ind] = x1 * bvec[8] + x2 * bvec[10] + x3 * bvec[12]; // Ws[6] <- LZ1 = b8M2 + b10M4 + b12M6
        VecMatE[7]->v[mat_ind] = x1 * bvec[2] + x2 * bvec[4] + x3 * bvec[6];   // Ws[7] <- LZ2 = b2M2 + b4M4 + b6M6
    }

    memcpy(VecMatE[8]->v, VecMatE[5]->v, sizeof(REAL_TYPE) * d * d);                                                                 // Ws[8] <- LW = A6LW1 + M6W1 + LW2
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatE[3]->v, d, VecMatM[4]->v, d, 1.0, VecMatE[8]->v, d); // Ws[8] <- LW = A6LW1 + M6W1 + LW2
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatM[3]->v, d, VecMatE[4]->v, d, 1.0, VecMatE[8]->v, d); // Ws[8] <- LW = A6LW1 + M6W1 + LW2

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatE[0]->v, d, VecMatM[8]->v, d, 0.0, VecMatE[9]->v, d); // Ws[9] <- LU = A * LW + EW
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatM[0]->v, d, VecMatE[8]->v, d, 1.0, VecMatE[9]->v, d); // Ws[9] <- LU = A * LW + EW

    memcpy(VecMatE[10]->v, VecMatE[7]->v, sizeof(REAL_TYPE) * d * d);                                                                 // Ws[10] <- LV = A6LZ1 + E6Z1 + LZ2
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatE[3]->v, d, VecMatM[6]->v, d, 1.0, VecMatE[10]->v, d); // Ws[10] <- LV = A6LZ1 + E6Z1 + LZ2
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatM[3]->v, d, VecMatE[6]->v, d, 1.0, VecMatE[10]->v, d); // Ws[10] <- LV = A6LZ1 + E6Z1 + LZ2

    memcpy(VecMatE[11]->v, VecMatE[9]->v, sizeof(REAL_TYPE) * d * d);                                                                   // Es[11] <- LU
    cblas_daxpy(d * d, -1.0, VecMatE[10]->v, 1, VecMatE[11]->v, 1);                                                                     // Es[11] <- LU - LV
    memcpy(VecMatE[12]->v, VecMatE[9]->v, sizeof(REAL_TYPE) * d * d);                                                                   // Es[12] <- LU
    cblas_daxpy(d * d, 1.0, VecMatE[10]->v, 1, VecMatE[12]->v, 1);                                                                      // Es[12] <- LU + LV
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatE[11]->v, d, VecMatM[12]->v, d, 1.0, VecMatE[12]->v, d); // Es[12] <- LU + LV + (LU - LV) * R
    PLU.Solve(VecMatE[12]->v, d, d);
}