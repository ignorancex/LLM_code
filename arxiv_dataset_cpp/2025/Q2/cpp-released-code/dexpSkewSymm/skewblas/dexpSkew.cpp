#include "dexpSkew.hpp"

void dexpSkewSymmPara::_setup_22_paras(REAL_TYPE *_forward, REAL_TYPE *_inverse, REAL_TYPE x, REAL_TYPE y)
{
    REAL_TYPE xpy = x + y;
    REAL_TYPE ymx = y - x;
    REAL_TYPE a = _dexpSkewSymm_sxdx(ymx);
    REAL_TYPE b = _dexpSkewSymm_cxdx(ymx);
    REAL_TYPE c = _dexpSkewSymm_sxdx(xpy);
    REAL_TYPE d = _dexpSkewSymm_cxdx(xpy);

    REAL_TYPE apco2 = (a + c) * 0.5;
    REAL_TYPE amco2 = (a - c) * 0.5;
    REAL_TYPE bpdo2 = (b + d) * 0.5;
    REAL_TYPE bmdo2 = (b - d) * 0.5;

    REAL_TYPE aprime = _dexpSkewSymm_xctx(ymx * 0.5);
    REAL_TYPE cprime = _dexpSkewSymm_xctx(xpy * 0.5);

    REAL_TYPE aprime_plus_cprime_over_2 = (aprime + cprime) * 0.5;
    REAL_TYPE aprime_minus_cprime_over_2 = (aprime - cprime) * 0.5;
    // REAL_TYPE bprime_plus_dprime_over_2 = x * 0.5;
    // REAL_TYPE bprime_minus_dprime_over_2 = -y * 0.5;

    _forward[0] = apco2;
    _forward[1] = bpdo2;
    _forward[2] = -bmdo2;
    _forward[3] = amco2;
    _forward[4] = -bpdo2;
    _forward[5] = apco2;
    _forward[6] = -amco2;
    _forward[7] = -bmdo2;
    _forward[8] = bmdo2;
    _forward[9] = -amco2;
    _forward[10] = apco2;
    _forward[11] = bpdo2;
    _forward[12] = amco2;
    _forward[13] = bmdo2;
    _forward[14] = -bpdo2;
    _forward[15] = apco2;

    _inverse[0] = aprime_plus_cprime_over_2;
    _inverse[1] = y * 0.5;
    _inverse[2] = x * 0.5;
    _inverse[3] = aprime_minus_cprime_over_2;
    _inverse[4] = -y * 0.5;
    _inverse[5] = aprime_plus_cprime_over_2;
    _inverse[6] = -aprime_minus_cprime_over_2;
    _inverse[7] = x * 0.5;
    _inverse[8] = -x * 0.5;
    _inverse[9] = -aprime_minus_cprime_over_2;
    _inverse[10] = aprime_plus_cprime_over_2;
    _inverse[11] = y * 0.5;
    _inverse[12] = aprime_minus_cprime_over_2;
    _inverse[13] = -x * 0.5;
    _inverse[14] = -y * 0.5;
    _inverse[15] = aprime_plus_cprime_over_2;
};

void dexpSkewSymmPara::_setup_12_paras(REAL_TYPE *_forward, REAL_TYPE *_inverse, REAL_TYPE x)
{
    REAL_TYPE e = _dexpSkewSymm_sxdx(x);
    REAL_TYPE f = _dexpSkewSymm_cxdx(x);
    REAL_TYPE ep = _dexpSkewSymm_xctx(x * 0.5);
    // REAL_TYPE fp = x * 0.5;

    _forward[0] = e;
    _forward[1] = f;
    _forward[2] = -f;
    _forward[3] = e;

    _inverse[0] = ep;
    _inverse[1] = x * 0.5;
    _inverse[2] = -x * 0.5;
    _inverse[3] = ep;
};

void dexpSkewSymmPara::_set_A(REAL_TYPE *VecA)
{
    // assert(forward != nullptr);
    // assert(a != nullptr);

    A.Assign(VecA);

    REAL_TYPE *_forward_current = _forward_para.v;
    REAL_TYPE *_inverse_current = _inverse_para.v;

    for (auto blk_col_index = 0; blk_col_index < a; blk_col_index++)
    {
        for (auto blk_row_index = blk_col_index + 1; blk_row_index < a; blk_row_index++)
        {
            _setup_22_paras(_forward_current, _inverse_current, A[blk_col_index], A[blk_row_index]);
            _forward_current += 16;
            _inverse_current += 16;
        }
    }

    for (auto blk_col_index = 0; blk_col_index < a; blk_col_index++)
    {
        _setup_12_paras(_forward_current, _inverse_current, A[blk_col_index]);
        _forward_current += 4;
        _inverse_current += 4;
    }
};

void dexpSkewSymmPara::_dexpSkewSymm_forward_core(REAL_TYPE *lvY, REAL_TYPE *lvX)
{
    // IMPORTANT : This routine can perform inplace update, i.e., it accepts lvX and lvY that are pointing to the same address space.
    // The input lvX and lvY must be in the lower-2x2-block-traversal-order defined by LowerTraversal.hpp::strict_lower_blk_traversal.

    // This routine computes the lower vector lvY of the skew symmetric Y from the lower vector lvX of the skew symmetric X subjected to:
    //              X = R' A R and Y = R' B R with Dexp_S[A] = exp(S)B with the real Schur decomposition S = R D R'.

    INTE_TYPE blk_index = 0;
    REAL_TYPE *A_cur = _forward_para.v;
    REAL_TYPE *x_cur = lvX;
    REAL_TYPE *y_cur = lvY;
    REAL_TYPE *w = Work.v;
    for (blk_index = 0; blk_index < _22_blk_size; blk_index++)
    {
        // Update on the strict lower 2 x 2 blocks.
        cblas_dgemv(CblasColMajor, CblasNoTrans, 4, 4, 1.0, A_cur, 4, x_cur, 1, 0.0, w, 1); // work = A * x;
        memcpy(y_cur, w, 4 * sizeof(REAL_TYPE));                                            // y = work;
        A_cur += 16;
        x_cur += 4;
        y_cur += 4;
    }

    if (d % 2)
    {
        // Odd number of the dimension implies a left-over row.

        // Update on 1 x 2 blocks
        for (blk_index = 0; blk_index < a; blk_index++)
        {
            cblas_dgemv(CblasColMajor, CblasNoTrans, 2, 2, 1.0, A_cur, 2, x_cur, 1, 0.0, w, 1); // work = A * x;
            memcpy(y_cur, w, 2 * sizeof(REAL_TYPE));                                            // y = work;
            A_cur += 4;
            x_cur += 2;
            y_cur += 2;
        }
    }

    // Update on the remaining bottom-left-corner of the diagonal 2 x 2 blocks.
    memcpy(y_cur, x_cur, a * sizeof(REAL_TYPE));
};

void dexpSkewSymmPara::_dexpSkewSymm_inverse_core(REAL_TYPE *lvY, REAL_TYPE *lvX)
{
    // IMPORTANT : This routine can perform inplace update, i.e., it accepts lvX and lvY that are pointing to the same address space.
    // The input lvX and lvY must be in the lower-2x2-block-traversal-order defined by LowerTraversal.hpp::strict_lower_blk_traversal.

    // This routine computes the lower vector lvY of the skew symmetric Y from the lower vector lvX of the skew symmetric X subjected to:
    //              X = R' A R and Y = R' B R with Dexp_S[A] = exp(S)B with the real Schur decomposition S = R D R'.

    INTE_TYPE blk_index = 0;
    REAL_TYPE *A_cur = _inverse_para.v;
    REAL_TYPE *x_cur = lvX;
    REAL_TYPE *y_cur = lvY;
    REAL_TYPE *w = Work.v;
    for (blk_index = 0; blk_index < _22_blk_size; blk_index++)
    {
        // Update on the strict lower 2 x 2 blocks.
        cblas_dgemv(CblasColMajor, CblasNoTrans, 4, 4, 1.0, A_cur, 4, x_cur, 1, 0.0, w, 1); // work = A * x;
        memcpy(y_cur, w, 4 * sizeof(REAL_TYPE));                                            // y = work;
        A_cur += 16;
        x_cur += 4;
        y_cur += 4;
    }

    if (d % 2)
    {
        // Odd number of the dimension implies a left-over row.

        // Update on 1 x 2 blocks
        for (blk_index = 0; blk_index < a; blk_index++)
        {
            cblas_dgemv(CblasColMajor, CblasNoTrans, 2, 2, 1.0, A_cur, 2, x_cur, 1, 0.0, w, 1); // work = A * x;
            memcpy(y_cur, w, 2 * sizeof(REAL_TYPE));                                            // y = work;
            A_cur += 4;
            x_cur += 2;
            y_cur += 2;
        }
    }

    // Update on the remaining bottom-left-corner of the diagonal 2 x 2 blocks.
    memcpy(y_cur, x_cur, a * sizeof(REAL_TYPE));
};

void dexpSkewSymmPara::_dexpSkewSymm_forward_core(SkewSymmMat &Y, SkewSymmMat &X)
{
    Y.mat2vec(_blk_tra);
    // View_ArrVec<REAL_TYPE>(Y.lv, Y.lsize).printf();s
    _dexpSkewSymm_forward_core(Y.lv, X.lv);
    // View_ArrVec<REAL_TYPE>(Y.lv, Y.lsize).printf();
    Y.vec2mat(_blk_tra);
};

void dexpSkewSymmPara::_dexpSkewSymm_inverse_core(SkewSymmMat &Y, SkewSymmMat &X)
{
    Y.mat2vec(_blk_tra);
    _dexpSkewSymm_inverse_core(Y.lv, X.lv);
    Y.vec2mat(_blk_tra);
};
