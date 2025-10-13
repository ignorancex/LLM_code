#include "skewSymmMat.hpp"

void SkewSymmMat::SchurAngular_SkewSymm()
{
    // The Schur decomposition of skew symmetric n x n matrices require 2 n workspace to store the complex (purely imaginary) eigenvalues.
    REAL_TYPE *wr = saf.w;
    REAL_TYPE *wi = wr + d;
    REAL_TYPE *wS = wr + 2 * d;
    memcpy(wS, v, sizeof(REAL_TYPE) * msize);

    this->initial_saf();
    saf.svec.fast_col_access();

    my_dgees('V', 'S', _dgees_select_nonzero, d, wS, d, &(saf.nzsize), wr, wi, saf.v, d);
    saf.nzsize = saf.nzsize / 2;
    REAL_TYPE *bot_left_ptr = wS + 1;
    for (INTE_TYPE i = 0; i < saf.nzsize; i++, bot_left_ptr += (2 * d + 2))
        saf.a[i] = *bot_left_ptr;
}

void SchurAngular_SkewSymm(SchurAngularFactor &saf, const SkewSymmMat &MatS)
{
    auto wr = saf.w;
    auto wi = wr + MatS.d;
    auto wS = wr + 2 * MatS.d;
    memcpy(wS, MatS.v, sizeof(REAL_TYPE) * MatS.msize);

    // saf.svec.fast_col_access();

    my_dgees('V', 'S', _dgees_select_nonzero, MatS.d, wS, MatS.d, &(saf.nzsize), wr, wi, saf.v, MatS.d);
    // my_dgees('V', 'N', nullptr, MatS.d, wS, MatS.d, &(saf.nzsize), wr, wi, saf.v, MatS.d);
    saf.nzsize = saf.nzsize / 2;
    for (INTE_TYPE i = 0; i < saf.nzsize; i++)
        saf.a[i] = wS[2 * i + 1 + 2 * i * MatS.d];
}

void SchurAngular_SkewSymm_Spectral(SchurAngularFactor &saf, const SkewSymmMat &MatS)
{
    auto wDiag = saf.w;
    auto wOffD = wDiag + MatS.d;
    // memcpy(wS, MatS.v, sizeof(REAL_TYPE) * MatS.msize);

    auto S_ptr = MatS.v;
    auto Z_ptr = saf.zvec.v;
    for (INTE_TYPE ind = 0; ind < MatS.msize; ind++, S_ptr++, Z_ptr++)
    {
        Z_ptr->real = 0;
        Z_ptr->imag = *S_ptr;
    }

    LAPACKE_zhetrd(LAPACK_COL_MAJOR, 'U', MatS.d, saf.zvec.v, MatS.d, wDiag, wOffD, saf.tau);
    LAPACKE_zungtr(LAPACK_COL_MAJOR, 'U', MatS.d, saf.zvec.v, MatS.d, saf.tau);
    LAPACKE_zsteqr(LAPACK_COL_MAJOR, 'V', MatS.d, wDiag, wOffD, saf.zvec.v, MatS.d);

    saf.nzsize = MatS.d / 2;

    CMPX_TYPE **z_neg_col = saf.zvec._col_access;
    REAL_TYPE **v_pos_col = saf.svec._col_access;
    REAL_TYPE **v_neg_col = v_pos_col + 1;

    REAL_TYPE sqrt2 = sqrt(2);
    for (INTE_TYPE i = 0; i < saf.nzsize; i++, z_neg_col++, v_pos_col += 2, v_neg_col += 2)
    {
        if (wDiag[i] > -1e-12)
            saf.a[i] = 0.0;
        else
            saf.a[i] = -wDiag[i];

        for (INTE_TYPE row_ind = 0; row_ind < MatS.d; row_ind++)
        {
            (*v_pos_col)[row_ind] = (*z_neg_col)[row_ind].real * sqrt2;
            (*v_neg_col)[row_ind] = -(*z_neg_col)[row_ind].imag * sqrt2;
        }
    }
    if (2 * saf.nzsize != MatS.d)
    {
        // z_neg_col = saf.zvec._col_access + saf.nzsize;
        // v_pos_col = saf.svec._col_access + (MatS.d - 1);
        for (INTE_TYPE row_ind = 0; row_ind < MatS.d; row_ind++)
            (*v_pos_col)[row_ind] = (*z_neg_col)[row_ind].real;
    }
}

void Spectral_SkewSymm(SpectralFactor &spf, const SkewSymmMat &MatS)
{
    auto wDiag = spf.w;
    auto wOffD = wDiag + MatS.d;
    auto tau = spf.val;
    // memcpy(wS, MatS.v, sizeof(REAL_TYPE) * MatS.msize);

    auto S_ptr = MatS.v;
    auto Z_ptr = spf.spv.v;
    for (INTE_TYPE ind = 0; ind < MatS.msize; ind++, S_ptr++, Z_ptr++)
    {
        Z_ptr->real = 0;
        Z_ptr->imag = *S_ptr;
    }

    Z_ptr = spf.spv.v;

    LAPACKE_zhetrd(LAPACK_COL_MAJOR, 'U', MatS.d, Z_ptr, MatS.d, wDiag, wOffD, tau);
    LAPACKE_zungtr(LAPACK_COL_MAJOR, 'U', MatS.d, Z_ptr, MatS.d, tau);
    LAPACKE_zsteqr(LAPACK_COL_MAJOR, 'V', MatS.d, wDiag, wOffD, Z_ptr, MatS.d);
    for (INTE_TYPE ind = 0; ind < MatS.d; ind++)
        assignCMPX(spf.val[ind], 0.0, wDiag[ind]);
}
