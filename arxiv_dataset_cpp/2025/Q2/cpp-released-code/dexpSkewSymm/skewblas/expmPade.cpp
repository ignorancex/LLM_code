#include "expmPade.hpp"

void expmPadeApprox::_expm_pade_low()
{
    // std::printf("Computing Pade Approximant of expm in order (%lld / %lld):\n", m, m);
    _check_and_initialize(VecMatM, _vmsize, d);

    // For m = 3, 5, 7, 9
    // The required VecMatM sizes are 7, 8, 9, 10, stored as follows
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

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatM[0]->v, d, VecMatM[0]->v, d, 0.0, VecMatM[1]->v, d);
    for (auto ind = 1; ind < k; ind++)
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatM[1]->v, d, VecMatM[ind]->v, d, 0.0, VecMatM[ind + 1]->v, d);

    // Getting the A's in the VecMatM[0 : k]
    VecMatM[k + 1]->Zero(); // W for odd terms
    VecMatM[k + 2]->Zero(); // U, the even terms
    for (auto ind = 0; ind < k; ind++)
    {
        cblas_daxpy(d * d, bvec[ind * 2 + 3], VecMatM[ind + 1]->v, 1, VecMatM[k + 1]->v, 1); // W += b_{2i+1} * A^{2i}
        cblas_daxpy(d * d, bvec[ind * 2 + 2], VecMatM[ind + 1]->v, 1, VecMatM[k + 2]->v, 1); // U += b_{2i} * A^{2i}
    }
    for (auto ind = 0; ind < d; ind++)
    {
        VecMatM[k + 1]->v[ind * (d + 1)] += bvec[1]; // W += b_{1} * I
        VecMatM[k + 2]->v[ind * (d + 1)] += bvec[0]; // U += b_{0} * I
    }

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatM[0]->v, d, VecMatM[k + 1]->v, d, 0.0, VecMatM[k + 3]->v, d); // V, the odd terms

    cblas_dcopy(d * d, VecMatM[k + 3]->v, 1, PLU.M.v, 1);
    cblas_daxpy(d * d, -1.0, VecMatM[k + 2]->v, 1, PLU.M.v, 1); // LU <- V - U
    PLU.f = false;
    PLU.Factor();

    cblas_dcopy(d * d, VecMatM[k + 3]->v, 1, VecMatM[k + 4]->v, 1);
    cblas_daxpy(d * d, 1.0, VecMatM[k + 2]->v, 1, VecMatM[k + 4]->v, 1); // V + U
    cblas_dcopy(d * d, VecMatM[k + 4]->v, 1, VecMatM[k + 5]->v, 1);      // Ms[12] <- Ms[11]
    PLU.Solve(VecMatM[k + 5]->v, d, d);                                  // Ms[11] <- R = (V - U)^{-1}(Ms[11])
}

void expmPadeApprox::_expm_pade_13()
{
    // Labelled as A, A2, A4, A6, W1, W2, Z1, Z2, W, U, V, R, W, where W is the workspace for getting powers in exp(A) = R^{2^s}

    // A is assumed to be given in Ms[0];

    // std::printf("Computing Pade Approximant of expm in order (%lld / %lld):\n", m, m);

    _check_and_initialize(VecMatM, _vmsize, d);

    const INTE_TYPE *bvec = _EXPM_PADE_APPROX_13;

    REAL_TYPE x1, x2, x3;

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatM[0]->v, d, VecMatM[0]->v, d, 0.0, VecMatM[1]->v, d); // Ms[1] <- A2 = Ms[0] * Ms[0]
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatM[1]->v, d, VecMatM[1]->v, d, 0.0, VecMatM[2]->v, d); // Ms[2] <- A4 = Ms[1] * Ms[1]
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatM[1]->v, d, VecMatM[2]->v, d, 0.0, VecMatM[3]->v, d); // Ms[3] <- A6 = Ms[1] * Ms[2]

    for (auto mat_ind = 0; mat_ind < d * d; mat_ind++)
    {
        x1 = VecMatM[1]->v[mat_ind];
        x2 = VecMatM[2]->v[mat_ind];
        x3 = VecMatM[3]->v[mat_ind];
        VecMatM[4]->v[mat_ind] = x1 * bvec[9] + x2 * bvec[11] + x3 * bvec[13]; // Ms[4] <- W1 = b9A2 + b11A4 + b13A6
        VecMatM[5]->v[mat_ind] = x1 * bvec[3] + x2 * bvec[5] + x3 * bvec[7];   // Ms[5] <- W2 = (b1I +) b3A2 + b5A4 + b7A6
        VecMatM[6]->v[mat_ind] = x1 * bvec[8] + x2 * bvec[10] + x3 * bvec[12]; // Ms[6] <- Z1 = b8A2 + b10A4 + b12A6
        VecMatM[7]->v[mat_ind] = x1 * bvec[2] + x2 * bvec[4] + x3 * bvec[6];   // Ms[7] <- Z2 = (b0I +) b2A2 + b4A4 + b6A6
    }
    for (auto col_ind = 0; col_ind < d; col_ind++)
    {
        VecMatM[5]->v[col_ind + col_ind * d] += bvec[1];
        VecMatM[7]->v[col_ind + col_ind * d] += bvec[0];
    }

    memcpy(VecMatM[8]->v, VecMatM[5]->v, sizeof(REAL_TYPE) * d * d);                                                                 // Ms[8] <- W = A6W1 + W2
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatM[3]->v, d, VecMatM[4]->v, d, 1.0, VecMatM[8]->v, d); // Ms[8] <- W = A6W1 + W2
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatM[0]->v, d, VecMatM[8]->v, d, 0.0, VecMatM[9]->v, d); // Ms[9] <- U = AW

    memcpy(VecMatM[10]->v, VecMatM[7]->v, sizeof(REAL_TYPE) * d * d);                                                                 // Ms[10] <- V = A6Z1 + Z2
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, VecMatM[3]->v, d, VecMatM[6]->v, d, 1.0, VecMatM[10]->v, d); // Ms[10] <- V = A6Z1 + Z2

    cblas_dcopy(d * d, VecMatM[10]->v, 1, PLU.M.v, 1);
    cblas_daxpy(d * d, -1.0, VecMatM[9]->v, 1, PLU.M.v, 1); // LU <- V - U
    PLU.f = false;
    PLU.Factor();

    cblas_dcopy(d * d, VecMatM[10]->v, 1, VecMatM[11]->v, 1);
    cblas_daxpy(d * d, 1.0, VecMatM[9]->v, 1, VecMatM[11]->v, 1); // Ms[11] <- V + U
    cblas_dcopy(d * d, VecMatM[11]->v, 1, VecMatM[12]->v, 1);     // Ms[12] <- Ms[11]
    PLU.Solve(VecMatM[12]->v, d, d);                              // Ms[11] <- R = (V - U)^{-1}(Ms[11])
}