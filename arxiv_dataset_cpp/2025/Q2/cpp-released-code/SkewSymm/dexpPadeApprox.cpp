#include "dexpPadeApprox.hpp"

void dexpPadeApproximant::setup_rs()
{
    if (!Rs[0])
        Rs[0] = new ColMat<REAL_TYPE>(d, d);

    Rs[0]->copy(*(empa->Ms[11]));

    ColMat<REAL_TYPE> **ptr = Rs;
    for (INTE_TYPE ind = 0; ind < (empa->s - 1); ind++, ptr++)
    {
        if (!*(ptr + 1))
            *(ptr + 1) = new ColMat<REAL_TYPE>(d, d);
        my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, (*ptr)->v, d, (*ptr)->v, d, 0.0, (*(ptr + 1))->v, d);
    }
}

void dexpPadeApproximant::_dexp_core()
{
    // ind:     0       1       2       3       4       5       6       7       8       9       10      11                                      12
    // Ms       A       A2      A4      A6      W1      W2      Z1      Z2      W       U       V       R                                       Workspace
    // Ws       M       M2      M4      M6      LW1     LW2     LZ1     LZ2     LW      LU      LV      L in (V - U)L = LU + LV + (LU-LV)*R     Workspace

    // M is given before the entry.
    const INTE_TYPE *bvec = _EXPM_PADE_APPROXIMANT;
    REAL_TYPE x1, x2, x3;

    my_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *Ms[0], *Ws[0], 0.0, *Ws[1]); // Ws[1] <- M2 = AM + MA
    my_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *Ws[0], *Ms[0], 1.0, *Ws[1]); // Ws[1] <- M2 = AM + MA

    my_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *Ms[1], *Ws[1], 0.0, *Ws[2]); // Ws[2] <- M4 = A2M2 + M2A2
    my_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *Ws[1], *Ms[1], 1.0, *Ws[2]); // Ws[2] <- M4 = A2M2 + M2A2

    my_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *Ms[2], *Ws[1], 0.0, *Ws[3]); // Ws[3] <- M6 = A4M2 + M4A2
    my_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *Ws[2], *Ms[1], 1.0, *Ws[3]); // Ws[3] <- M6 = A4M2 + M4A2

    for (INTE_TYPE mat_ind = 0; mat_ind < d * d; mat_ind++)
    {
        x1 = Ws[1]->v[mat_ind];
        x2 = Ws[2]->v[mat_ind];
        x3 = Ws[3]->v[mat_ind];
        Ws[4]->v[mat_ind] = x1 * bvec[9] + x2 * bvec[11] + x3 * bvec[13]; // Ws[4] <- LW1 = b9M2 + b11M4 + b13M6
        Ws[5]->v[mat_ind] = x1 * bvec[3] + x2 * bvec[5] + x3 * bvec[7];   // Ws[5] <- LW2 = b3M2 + b5M4 + b7M6
        Ws[6]->v[mat_ind] = x1 * bvec[8] + x2 * bvec[10] + x3 * bvec[12]; // Ws[6] <- LZ1 = b8M2 + b10M4 + b12M6
        Ws[7]->v[mat_ind] = x1 * bvec[2] + x2 * bvec[4] + x3 * bvec[6];   // Ws[7] <- LZ2 = b2M2 + b4M4 + b6M6
    }

    memcpy(Ws[8]->v, Ws[5]->v, sizeof(REAL_TYPE) * d * d);                  // Ws[8] <- LW = A6LW1 + M6W1 + LW2
    my_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *Ws[3], *Ms[4], 1.0, *Ws[8]); // Ws[8] <- LW = A6LW1 + M6W1 + LW2
    my_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *Ms[3], *Ws[4], 1.0, *Ws[8]); // Ws[8] <- LW = A6LW1 + M6W1 + LW2

    my_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *Ws[0], *Ms[8], 0.0, *Ws[9]); // Ws[9] <- LU = A * LW + EW
    my_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *Ms[0], *Ws[8], 1.0, *Ws[9]); // Ws[9] <- LU = A * LW + EW

    memcpy(Ws[10]->v, Ws[7]->v, sizeof(REAL_TYPE) * d * d);                  // Ws[10] <- LV = A6LZ1 + M6Z1 + LZ2
    my_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *Ws[3], *Ms[6], 1.0, *Ws[10]); // Ws[10] <- LV = A6LZ1 + M6Z1 + LZ2
    my_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *Ms[3], *Ws[6], 1.0, *Ws[10]); // Ws[10] <- LV = A6LZ1 + M6Z1 + LZ2

    // Using Ws[12] as workspace.
    Ws[12]->copy(*Ws[9]);                                                      // Ws[12] <- LU
    rsub(*Ws[12], *Ws[10]);                                                    // Ws[12] <- LU - LV
    Ws[11]->copy(*Ws[9]);                                                      // Ws[11] <- LU
    radd(*Ws[11], *Ws[10]);                                                    // Ws[11] <- LU + LV
    my_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *Ws[12], *Ms[11], 1.0, *Ws[11]); // Ws[11] <- LU + LV + (LU - LV) * R
    solverPLU(Ws[11]->v, d, d, empa->plu);                                     // Ws[11] <- L = (V - U)^{-1}(LU + LV + (LU - LV) * R)
}

void dexpPadeApproximant::_dexp_recover(ColMat<REAL_TYPE> &N)
{
    // Recursively do L_{i+1} = L_iR_i + R_iL_i
    using std::swap;
    ColMat<REAL_TYPE> *curr_L = Ws[12];
    ColMat<REAL_TYPE> *next_L = &N;
    ColMat<REAL_TYPE> **curr_R = Rs;
    if (empa->s % 2)
    {
        my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, Ws[11]->v, d, (*curr_R)->v, d, 0.0, next_L->v, d);
        my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, (*curr_R)->v, d, Ws[11]->v, d, 1.0, next_L->v, d);
        swap(curr_L, next_L);
        curr_R++;
        for (INTE_TYPE s_ind = 1; s_ind < empa->s; s_ind++, curr_R++, swap(curr_L, next_L))
        {
            my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, curr_L->v, d, (*curr_R)->v, d, 0.0, next_L->v, d);
            my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, (*curr_R)->v, d, curr_L->v, d, 1.0, next_L->v, d);
        }
    }
    else
    {
        next_L->copy(*Ws[11]);
        swap(curr_L, next_L);
        for (INTE_TYPE s_ind = 0; s_ind < empa->s; s_ind++, curr_R++, swap(curr_L, next_L))
        {
            my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, curr_L->v, d, (*curr_R)->v, d, 0.0, next_L->v, d);
            my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, (*curr_R)->v, d, curr_L->v, d, 1.0, next_L->v, d);
        }
    }
}

void dexpPadeApproximant::_dexp_recover(ColMat<REAL_TYPE> &N, ColMat<REAL_TYPE> &Q)
{
    // Recursively do L_{i+1} = L_iR_i + R_iL_i
    using std::swap;
    ColMat<REAL_TYPE> *curr_L = Ws[12];
    ColMat<REAL_TYPE> *next_L = &N;
    ColMat<REAL_TYPE> **curr_R = Rs;
    if (empa->s % 2)
    {
        my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, Ws[11]->v, d, (*curr_R)->v, d, 0.0, next_L->v, d);
        my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, (*curr_R)->v, d, Ws[11]->v, d, 1.0, next_L->v, d);
        swap(curr_L, next_L);
        curr_R++;
        for (INTE_TYPE s_ind = 1; s_ind < empa->s; s_ind++, curr_R++, swap(curr_L, next_L))
        {
            my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, curr_L->v, d, (*curr_R)->v, d, 0.0, next_L->v, d);
            my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, (*curr_R)->v, d, curr_L->v, d, 1.0, next_L->v, d);
        }
    }
    else
    {
        next_L->copy(*Ws[11]);
        swap(curr_L, next_L);
        for (INTE_TYPE s_ind = 0; s_ind < empa->s; s_ind++, curr_R++, swap(curr_L, next_L))
        {
            my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, curr_L->v, d, (*curr_R)->v, d, 0.0, next_L->v, d);
            my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, (*curr_R)->v, d, curr_L->v, d, 1.0, next_L->v, d);
        }
    }
    my_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, d, d, d, 1.0, Q.v, d, N.v, d, 0.0, Ws[12]->v, d);
    N.assign(*(Ws[12]));
}
