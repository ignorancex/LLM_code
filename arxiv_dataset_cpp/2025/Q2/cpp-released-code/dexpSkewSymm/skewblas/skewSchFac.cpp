#include "skewSchFac.hpp"

void SkewSchurFactor::_chessboard_scatter(INTE_TYPE dim, REAL_TYPE *board, INTE_TYPE ldb, REAL_TYPE *U, INTE_TYPE ldu, REAL_TYPE *Vt, INTE_TYPE ldvt)
{
    REAL_TYPE *ptrB = board;
    REAL_TYPE *ptrU = U;
    REAL_TYPE *ptrV = Vt;

    for (auto odd_ind = 0; odd_ind < k; odd_ind++, ptrV++, ptrB += 2 * ldb)
    {
        memset(ptrB, 0, sizeof(REAL_TYPE) * dim);
        cblas_dcopy(k, ptrV, ldvt, ptrB, 2);
    }
    ptrB = board + ldb;
    for (auto even_ind = 0; even_ind < m; even_ind++, ptrU += ldu, ptrB += 2 * ldb)
    {
        memset(ptrB, 0, sizeof(REAL_TYPE) * dim);
        cblas_dcopy(m, ptrU, 1, ptrB + 1, 2);
    }
};

void SkewSchurFactor::_givens_elimin_bidiag_extra_column(REAL_TYPE *_cos, REAL_TYPE *_sin, REAL_TYPE *diag, REAL_TYPE *offd, REAL_TYPE extra)
{
    REAL_TYPE temp = 0;
    REAL_TYPE diag_ele = 0;
    REAL_TYPE _c = 0;
    REAL_TYPE _s = 0;
    for (auto blk_ind = 0; blk_ind < m - 1; blk_ind++)
    {
        diag_ele = diag[m - 1 - blk_ind];
        temp = sqrt(extra * extra + diag_ele * diag_ele);
        _c = diag_ele / temp;
        _s = extra / temp;
        _cos[blk_ind] = _c;
        _sin[blk_ind] = _s;
        diag[m - 1 - blk_ind] = _c * diag_ele + _s * extra;
        extra = -_s * offd[m - 2 - blk_ind];
        offd[m - 2 - blk_ind] = _c * offd[m - 2 - blk_ind];
    }

    diag_ele = diag[0];
    temp = sqrt(extra * extra + diag_ele * diag_ele);
    _c = diag[0] / temp;
    _s = extra / temp;
    _cos[m - 1] = _c;
    _sin[m - 1] = _s;
    diag[0] = _c * diag_ele + _s * extra;
}

void SkewSchurFactor::_givens_rotate_cols(INTE_TYPE coli, INTE_TYPE colj, REAL_TYPE _cos, REAL_TYPE _sin, REAL_TYPE *MatVt, INTE_TYPE n)
{

    auto ptri = MatVt + coli * n;
    auto ptrj = MatVt + colj * n;
    REAL_TYPE temp = 0;
    for (auto row_ind = 0; row_ind < n; row_ind++)
    {
        temp = ptri[row_ind];
        ptri[row_ind] = _cos * ptri[row_ind] - _sin * ptrj[row_ind];
        ptrj[row_ind] = _sin * temp + _cos * ptrj[row_ind];
    }
}

void SkewSchurFactor::SchurAngular_SkewSymm()
{
    // S is stored at H.MatH;
    auto WorkS = H.MatH.v;
    auto WorkU = U.v;
    auto WorkVt = Vt.v;

    auto WorkD = A.v;
    auto WorkE = Work.v;
    auto WorkBDC = WorkE + (m - 1);
    auto WorkBDS = WorkBDC + m;

    // WorkE requries m - 1 space
    // WorkBCC requries m space
    // WorkBCS requries m space

    auto S_offdiag_pos = WorkS + 1;
    auto S_offdiag_neg = S_offdiag_pos + d + 1;

    auto VtRow = WorkVt;
    auto UCol = WorkU;
    auto scCol = R.v;

    H.SkewSymm_TriDiag_BLAS3();
    // _bdh.SkewSymm_TriDiag_BLAS2(WorkS, n);

    for (auto blk_ind = 0; blk_ind < m - 1; blk_ind++, S_offdiag_pos += 2 * d + 2, S_offdiag_neg += 2 * d + 2)
    {
        WorkD[blk_ind] = *S_offdiag_pos;
        WorkE[blk_ind] = -(*S_offdiag_neg);
    }
    WorkD[m - 1] = *S_offdiag_pos;

    if (m != k)
        _givens_elimin_bidiag_extra_column(WorkBDC, WorkBDS, WorkD, WorkE, -(*S_offdiag_neg));

    memset(WorkU, 0, sizeof(REAL_TYPE) * m * m);
    memset(WorkVt, 0, sizeof(REAL_TYPE) * k * k);
    // WorkVt[(m+1)*(m+1)-1] = 1.0;
    if (m != k)
        WorkVt[(m + 2) * m] = 1.0;

    LAPACKE_dbdsdc(LAPACK_COL_MAJOR, 'U', 'I', m, WorkD, WorkE, WorkU, m, WorkVt, k, nullptr, nullptr);

    a = 0;
    for (auto blk_ind = 0; blk_ind < m; blk_ind++)
        a += (abs(WorkD[blk_ind]) > 1e-12);

    if (m != k)
        for (auto blk_ind = m - 1; blk_ind >= 0; blk_ind--)
            _givens_rotate_cols(m - 1 - blk_ind, m, WorkBDC[blk_ind], WorkBDS[blk_ind], WorkVt, k);
}

void SkewSchurFactor::Explict_Vector(REAL_TYPE *MatR, INTE_TYPE ldr)
{

    _chessboard_scatter(d, MatR, ldr, U.v, m, Vt.v, k);

    // View_ColMat<REAL_TYPE>(MatR, ldr, d, d).printf("Scattered Chessboard:\n");

    H.Action(MatR, d, ldr);
}

void SkewSchurFactor::SkewCongruence_Explict(CBLAS_TRANSPOSE trans, REAL_TYPE *MatS, INTE_TYPE lds)
{
    // Assuming R is explicitly constructed

    if (trans == CblasNoTrans)
    {
        // Computing S <- RSR' in the lower triangular part.
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, d, d, d, 1.0, MatS, lds, R.v, d, 0.0, Work.v, d);
        // Work <- SR'
        skewblas_mmlt(CblasNoTrans, CblasNoTrans, d, d, d, 1.0, R.v, d, Work.v, d, 0.0, MatS, lds);
        // S <- R Work
    }
    else
    {
        // Computing S <- R'SR in the lower triangular part.
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, d, d, d, 1.0, R.v, d, MatS, lds, 0.0, Work.v, d);
        // Work <- R'SR
        skewblas_mmlt(CblasNoTrans, CblasNoTrans, d, d, d, 1.0, Work.v, d, R.v, d, 0.0, MatS, lds);
        // S <- Work R
    }
}
