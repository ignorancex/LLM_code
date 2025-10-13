#include "hhMat.hpp"

REAL_TYPE HouseholderMatrix::col_elimin(REAL_TYPE *_vec, REAL_TYPE *_col, INTE_TYPE n)
{
    if (n <= 1)
    {
        *_vec = 1.0;
        return 0;
    }
    REAL_TYPE beta = 0, alpha = _col[0];
    REAL_TYPE tau = 0;
    if (alpha > 0)
        beta = -cblas_dnrm2(n, _col, 1);
    else
        beta = cblas_dnrm2(n, _col, 1);

    memcpy(_vec, _col, sizeof(REAL_TYPE) * n);
    memset(_col, 0, sizeof(REAL_TYPE) * n);

    tau = (beta - alpha) / beta;
    _col[0] = beta;
    _vec[0] = 1.0;

    for (INTE_TYPE ind = 1; ind < n; ind++)
        _vec[ind] /= (alpha - beta);

    // temp = my_dnrm2(n, _vec);
    // temp *= temp;
    // temp = 2.0 / temp;

    return tau;
};

// REAL_TYPE HouseholderMatrix::col_elimin(REAL_TYPE *_vec, INTE_TYPE n)
//{
//	REAL_TYPE vnorm = 0, temp = 0;
//	vnorm = my_dnrm2(n, _vec);
//	*_vec -= vnorm;

//	for (INTE_TYPE ind = 1; ind < n; ind++)
//		_vec[ind] /= _vec[0];
//	_vec[0] = 1.0;

//	temp = my_dnrm2(n, _vec);
//	temp *= temp;
//	temp = 2.0 / temp;

//	return temp;
//};

void HouseholderMatrix::_SkewSymm_Hessenberg_VW(REAL_TYPE *MatV, INTE_TYPE ldv, REAL_TYPE *MatW, INTE_TYPE ldw, REAL_TYPE *MatS, INTE_TYPE lds, REAL_TYPE *VecTau, INTE_TYPE reduced_dim, INTE_TYPE col_num)
{
    auto beginV = MatV;
    auto beginW = MatW;
    auto beginS = MatS;

    auto columnW = beginW;

    auto offdiagV = beginV + 1;
    auto offdiagW = beginW + 1;
    auto offdiagS = beginS + 1;
    auto nextdiag = offdiagS + lds;

    REAL_TYPE alpha = 0;

    for (INTE_TYPE col_ind = 0; col_ind < col_num; col_ind++, reduced_dim--, offdiagV += (ldv + 1), offdiagW += (ldw + 1), columnW += ldw, offdiagS += (lds + 1), nextdiag += (lds + 1))
    {
        // Skew-symm update the active column of S

        cblas_dgemv(CblasColMajor, CblasNoTrans, reduced_dim + 1, col_ind, 1.0, beginV + col_ind, ldv, beginW + col_ind, ldw, 1.0, offdiagS - 1, 1);
        cblas_dgemv(CblasColMajor, CblasNoTrans, reduced_dim + 1, col_ind, -1.0, beginW + col_ind, ldv, beginV + col_ind, ldv, 1.0, offdiagS - 1, 1);

        // Householder reflector that annihilates the last (reduced_dim - 1) entries.

        VecTau[col_ind] = col_elimin(offdiagV, offdiagS, reduced_dim);

        // Form the W matrix in Q S Q' = S + V W' - W V', where Q = H(1) ... H(d - 2)

        // w = S * v + (V * W' * v) + (W * V' * v);
        // y = v + (V * Y' * v) + (Y * V' * v);

        cblas_dgemv(CblasColMajor, CblasNoTrans, reduced_dim, reduced_dim, 1.0, nextdiag, lds, offdiagV, 1, 0.0, offdiagW, 1);
        cblas_dgemv(CblasColMajor, CblasTrans, reduced_dim, col_ind, 1.0, beginW + col_ind + 1, ldw, offdiagV, 1, 0.0, columnW, 1);
        cblas_dgemv(CblasColMajor, CblasNoTrans, reduced_dim, col_ind, 1.0, beginV + col_ind + 1, ldv, columnW, 1, 1.0, offdiagW, 1);
        cblas_dgemv(CblasColMajor, CblasTrans, reduced_dim, col_ind, 1.0, beginV + col_ind + 1, ldv, offdiagV, 1, 0.0, columnW, 1);
        cblas_dgemv(CblasColMajor, CblasNoTrans, reduced_dim, col_ind, -1.0, beginW + col_ind + 1, ldw, columnW, 1, 1.0, offdiagW, 1);
        cblas_dscal(reduced_dim, VecTau[col_ind], offdiagW, 1);
        alpha = -0.5 * VecTau[col_ind] * cblas_ddot(reduced_dim, offdiagV, 1, offdiagW, 1);
        cblas_daxpy(reduced_dim, alpha, offdiagV, 1, offdiagW, 1);
        memset(columnW, 0, sizeof(REAL_TYPE) * col_ind);
    }
}

void HouseholderMatrix::_SkewSymm_Hessenberg_VT(REAL_TYPE *MatV, INTE_TYPE ldv, REAL_TYPE *MatS, INTE_TYPE lds, INTE_TYPE col_num)
{
    INTE_TYPE reduced_dim = d - 1;
    REAL_TYPE alpha = 0;
    REAL_TYPE tau = 0;

    for (INTE_TYPE col_ind = 0; col_ind < col_num; col_ind++, MatS += lds + 1, MatV += ldv + 1, reduced_dim--)
    {
        tau = col_elimin(MatV + 1, MatS + 1, reduced_dim);
        if (abs(tau) > 1e-12)
        {
            cblas_dgemv(CblasColMajor, CblasNoTrans, reduced_dim, reduced_dim, tau, MatS + lds + 1, lds, MatV + 1, 1, 0.0, VecT + col_ind, 1);
            alpha = -0.5 * tau * cblas_ddot(reduced_dim, VecT + col_ind, 1, MatV + 1, 1);
            cblas_daxpy(reduced_dim, alpha, MatV + 1, 1, VecT + col_ind, 1);

            cblas_dger(CblasColMajor, reduced_dim, reduced_dim, 1.0, MatV + 1, 1, VecT + col_ind, 1, MatS + lds + 1, lds);
            cblas_dger(CblasColMajor, reduced_dim, reduced_dim, -1.0, VecT + col_ind, 1, MatV + 1, 1, MatS + lds + 1, lds);
        }
        VecT[col_ind] = tau;
    }
}

// REAL_TYPE HouseholderReflector::col_elimin(REAL_TYPE *vec, INTE_TYPE col, INTE_TYPE offset, INTE_TYPE length)
// {
//     memset(v + col * d, 0, sizeof(REAL_TYPE) * d);
//     REAL_TYPE vnorm = 0;
//     vnorm = cblas_dnrm2(length, vec, 1);
//     auto des = v + col * d + offset;
//     memcpy(des, vec, sizeof(REAL_TYPE) * length);
//     //*des += vnorm;
//     //*des -= (*v < 0) * 2 * vnorm;
//     // *des = (*v < 0) ? (*des) + vnorm : (*des) - vnorm;
//     *des -= vnorm;

//     for (INTE_TYPE ind = 1; ind < length; ind++)
//         des[ind] /= des[0];
//     des[0] = 1.0;

//     tau[col] = cblas_dnrm2(length, des, 1);
//     tau[col] *= tau[col];
//     tau[col] = 2.0 / tau[col];
//     // tau[col] *= des[0] * des[0];

//     return vnorm;
// }

// void HouseholderReflector::Action(CHAR_TYPE side, CHAR_TYPE trans, REAL_TYPE *MatQ, INTE_TYPE ldq)
// {
//     LAPACKE_dormqr_64(LAPACK_COL_MAJOR, side, trans, d, d, d, v, d, tau, MatQ, ldq);
// }
