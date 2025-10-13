#include "matOp.hpp"

void scal(INTE_TYPE n, INTE_TYPE k, REAL_TYPE s, REAL_TYPE *A, INTE_TYPE lda)
{
    REAL_TYPE *ptr = A;
    for (INTE_TYPE col_ind = 0; col_ind < k; col_ind++, ptr += lda)
        my_dscal(n, s, ptr);
}
void scal(REAL_TYPE s, ColMat<REAL_TYPE> &A)
{
    my_dscal(A.r * A.c, s, A.v);
}
void scal(REAL_TYPE s, const View_ColMat<REAL_TYPE> &A)
{
    REAL_TYPE *ptr = A.v;
    for (INTE_TYPE col_ind = 0; col_ind < A.c; col_ind++, ptr += A.ld)
        my_dscal(A.r, s, ptr);
}

void lmul(INTE_TYPE n, INTE_TYPE k, REAL_TYPE *A, INTE_TYPE lda, REAL_TYPE *B, INTE_TYPE ldb, REAL_TYPE *work)
{
    // A: n x n, B: n x k
    // work requires n x k space.
    my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, k, 1.0, A, lda, B, ldb, 0.0, work, n);
    REAL_TYPE *B_col_ptr = B;
    REAL_TYPE *w_col_ptr = work;

    for (INTE_TYPE col_ind = 0; col_ind < n; col_ind++, B_col_ptr += ldb, w_col_ptr += n)
        memcpy(B_col_ptr, w_col_ptr, sizeof(REAL_TYPE) * n);
}
void lmul(const ColMat<REAL_TYPE> &A, ColMat<REAL_TYPE> &B, REAL_TYPE *work)
{
    // A: n x n, B: n x k
    my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.r, A.c, B.c, 1.0, A.v, A.r, B.v, B.r, 0.0, work, A.r);
    memcpy(B.v, work, sizeof(REAL_TYPE) * B.r * B.c);
}

void rmul(INTE_TYPE n, INTE_TYPE k, REAL_TYPE *A, INTE_TYPE lda, REAL_TYPE *B, INTE_TYPE ldb, REAL_TYPE *work)
{
    // A: n x k, B: k x k
    // work requires n x k space

    my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, k, k, 1.0, A, lda, B, ldb, 0.0, work, n);
    REAL_TYPE *A_col_ptr = A;
    REAL_TYPE *w_col_ptr = work;

    for (INTE_TYPE col_ind = 0; col_ind < n; col_ind++, A_col_ptr += lda, w_col_ptr += n)
        memcpy(A_col_ptr, w_col_ptr, sizeof(REAL_TYPE) * n);
}
void rmul(ColMat<REAL_TYPE> &A, const ColMat<REAL_TYPE> &B, REAL_TYPE *work)
{
    // A: n x k, B: k x k
    // work requires n x k space
    my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.r, B.r, B.c, 1.0, A.v, A.r, B.v, B.r, 0.0, work, A.r);
    memcpy(A.v, work, sizeof(REAL_TYPE) * A.r * A.c);
}

void ladd(INTE_TYPE n, INTE_TYPE k, REAL_TYPE *A, INTE_TYPE lda, REAL_TYPE *B, INTE_TYPE ldb)
{
    REAL_TYPE *A_col_ptr = A;
    REAL_TYPE *B_col_ptr = B;

    for (INTE_TYPE col_ind = 0; col_ind < k; col_ind++, A_col_ptr += lda, B_col_ptr += ldb)
        for (INTE_TYPE row_ind = 0; row_ind < n; row_ind++)
            *(B_col_ptr + row_ind) += *(A_col_ptr + row_ind);
}
void ladd(const ColMat<REAL_TYPE> &A, ColMat<REAL_TYPE> &B)
{
    REAL_TYPE *A_ptr = A.v;
    REAL_TYPE *B_ptr = B.v;
    for (INTE_TYPE mat_ind = 0; mat_ind < A.r * A.c; mat_ind++, A_ptr++, B_ptr++)
        *B_ptr += *A_ptr;
}

void radd(INTE_TYPE n, INTE_TYPE k, REAL_TYPE *A, INTE_TYPE lda, REAL_TYPE *B, INTE_TYPE ldb)
{
    REAL_TYPE *A_col_ptr = A;
    REAL_TYPE *B_col_ptr = B;

    for (INTE_TYPE col_ind = 0; col_ind < k; col_ind++, A_col_ptr += lda, B_col_ptr += ldb)
        for (INTE_TYPE row_ind = 0; row_ind < n; row_ind++)
            *(A_col_ptr + row_ind) += *(B_col_ptr + row_ind);
}
void radd(ColMat<REAL_TYPE> &A, const ColMat<REAL_TYPE> &B)
{
    REAL_TYPE *A_ptr = A.v;
    REAL_TYPE *B_ptr = B.v;
    for (INTE_TYPE mat_ind = 0; mat_ind < A.r * A.c; mat_ind++, A_ptr++, B_ptr++)
        *A_ptr += *B_ptr;
}

void lsub(INTE_TYPE n, INTE_TYPE k, REAL_TYPE *A, INTE_TYPE lda, REAL_TYPE *B, INTE_TYPE ldb)
{
    REAL_TYPE *A_col_ptr = A;
    REAL_TYPE *B_col_ptr = B;
    for (INTE_TYPE col_ind = 0; col_ind < k; col_ind++, A_col_ptr += lda, B_col_ptr += ldb)
        for (INTE_TYPE row_ind = 0; row_ind < n; row_ind++)
            *(B_col_ptr + row_ind) = *(A_col_ptr + row_ind) - *(B_col_ptr + row_ind);
}
void lsub(const ColMat<REAL_TYPE> &A, ColMat<REAL_TYPE> &B)
{
    REAL_TYPE *A_ptr = A.v;
    REAL_TYPE *B_ptr = B.v;
    for (INTE_TYPE mat_ind = 0; mat_ind < A.r * A.c; mat_ind++, A_ptr++, B_ptr++)
        *B_ptr = *A_ptr - *B_ptr;
}

void rsub(INTE_TYPE n, INTE_TYPE k, REAL_TYPE *A, INTE_TYPE lda, REAL_TYPE *B, INTE_TYPE ldb)
{
    REAL_TYPE *A_col_ptr = A;
    REAL_TYPE *B_col_ptr = B;
    for (INTE_TYPE col_ind = 0; col_ind < k; col_ind++, A_col_ptr += lda, B_col_ptr += ldb)
        for (INTE_TYPE row_ind = 0; row_ind < n; row_ind++)
            *(A_col_ptr + row_ind) -= *(B_col_ptr + row_ind);
}
void rsub(ColMat<REAL_TYPE> &A, const ColMat<REAL_TYPE> &B)
{
    REAL_TYPE *A_ptr = A.v;
    REAL_TYPE *B_ptr = B.v;
    for (INTE_TYPE mat_ind = 0; mat_ind < A.r * A.c; mat_ind++, A_ptr++, B_ptr++)
        *A_ptr -= *B_ptr;
}

REAL_TYPE normsqFro(const ColMat<REAL_TYPE> &A)
{
    return my_dnrm2(A.r * A.c, A.v);
};
REAL_TYPE normsqFro(const View_ColMat<REAL_TYPE> &A)
{
    REAL_TYPE norm2 = 0.0;
    REAL_TYPE *ptr = A.v;
    for (INTE_TYPE col_ind = 0; col_ind < A.c; col_ind++, ptr += A.ld)
        norm2 += my_dnrm2(A.r, ptr);
    return norm2;
};
REAL_TYPE normFro(const ColMat<REAL_TYPE> &A)
{
    return sqrt(normsqFro(A));
};
REAL_TYPE normFro(const View_ColMat<REAL_TYPE> &A)
{
    return sqrt(normsqFro(A));
};

void congruence(CBLAS_TRANSPOSE tranc, CBLAS_TRANSPOSE tranm, INTE_TYPE m, INTE_TYPE n, REAL_TYPE *M, INTE_TYPE ldm, REAL_TYPE *C, INTE_TYPE ldc, REAL_TYPE *N, INTE_TYPE ldn, REAL_TYPE *W, INTE_TYPE ldw)
{
    if (tranc == CblasNoTrans)
    {
        my_dgemm(CblasColMajor, CblasNoTrans, tranm, n, m, m, 1.0, C, ldc, M, ldm, 0.0, W, ldw);
        my_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, m, m, 1.0, W, ldw, C, ldc, 0.0, N, ldn);
    }
    else
    {
        my_dgemm(CblasColMajor, CblasTrans, tranm, n, m, m, 1.0, C, ldc, M, ldm, 0.0, W, ldw);
        my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, m, 1.0, W, ldw, C, ldc, 0.0, N, ldn);
    }
}
void congruence(CBLAS_TRANSPOSE tranc, CBLAS_TRANSPOSE tranm, const ColMat<REAL_TYPE> &M, const ColMat<REAL_TYPE> &C, ColMat<REAL_TYPE> &N, ColMat<REAL_TYPE> &W)
{
    if (tranc == CblasNoTrans)
    {
        my_dgemm(CblasColMajor, CblasNoTrans, tranm, N.r, M.r, M.r, 1.0, C.v, C.r, M.v, M.r, 0.0, W.v, W.r);
        my_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, N.r, M.r, M.r, 1.0, W.v, W.r, C.v, C.r, 0.0, N.v, N.r);
    }
    else
    {
        my_dgemm(CblasColMajor, CblasTrans, tranm, N.r, M.r, M.r, 1.0, C.v, C.r, M.v, M.r, 0.0, W.v, W.r);
        my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N.r, M.r, M.r, 1.0, W.v, W.r, C.v, C.r, 0.0, N.v, N.r);
    }
}
void congruence(CBLAS_TRANSPOSE tranc, CBLAS_TRANSPOSE tranm, const View_ColMat<REAL_TYPE> &M, const View_ColMat<REAL_TYPE> &C, const View_ColMat<REAL_TYPE> &N, const View_ColMat<REAL_TYPE> &W)
{
    if (tranc == CblasNoTrans)
    {
        my_dgemm(CblasColMajor, CblasNoTrans, tranm, N.r, M.r, M.r, 1.0, C.v, C.ld, M.v, M.ld, 0.0, W.v, W.ld);
        my_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, N.r, M.r, M.r, 1.0, W.v, W.ld, C.v, C.ld, 0.0, N.v, N.ld);
    }
    else
    {
        my_dgemm(CblasColMajor, CblasTrans, tranm, N.r, M.r, M.r, 1.0, C.v, C.ld, M.v, M.ld, 0.0, W.v, W.ld);
        my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N.r, M.r, M.r, 1.0, W.v, W.ld, C.v, C.ld, 0.0, N.v, N.ld);
    }
}
void congruence(CBLAS_TRANSPOSE tranc, INTE_TYPE m, INTE_TYPE n, REAL_TYPE *M, INTE_TYPE ldm, REAL_TYPE *C, INTE_TYPE ldc, REAL_TYPE *N, INTE_TYPE ldn, REAL_TYPE *W, INTE_TYPE ldw)
{
    if (tranc == CblasNoTrans)
    {
        my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, m, 1.0, C, ldc, M, ldm, 0.0, W, ldw);
        my_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, m, m, 1.0, W, ldw, C, ldc, 0.0, N, ldn);
    }
    else
    {
        my_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, m, m, 1.0, C, ldc, M, ldm, 0.0, W, ldw);
        my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, m, 1.0, W, ldw, C, ldc, 0.0, N, ldn);
    }
}
void congruence(CBLAS_TRANSPOSE tranc, const ColMat<REAL_TYPE> &M, const ColMat<REAL_TYPE> &C, ColMat<REAL_TYPE> &N, ColMat<REAL_TYPE> &W)
{
    if (tranc == CblasNoTrans)
    {
        my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N.r, M.r, M.r, 1.0, C.v, C.r, M.v, M.r, 0.0, W.v, W.r);
        my_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, N.r, M.r, M.r, 1.0, W.v, W.r, C.v, C.r, 0.0, N.v, N.r);
    }
    else
    {
        my_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, N.r, M.r, M.r, 1.0, C.v, C.r, M.v, M.r, 0.0, W.v, W.r);
        my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N.r, M.r, M.r, 1.0, W.v, W.r, C.v, C.r, 0.0, N.v, N.r);
    }
}
void congruence(CBLAS_TRANSPOSE tranc, const View_ColMat<REAL_TYPE> &M, const View_ColMat<REAL_TYPE> &C, const View_ColMat<REAL_TYPE> &N, const View_ColMat<REAL_TYPE> &W)
{
    if (tranc == CblasNoTrans)
    {
        my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N.r, M.r, M.r, 1.0, C.v, C.ld, M.v, M.ld, 0.0, W.v, W.ld);
        my_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, N.r, M.r, M.r, 1.0, W.v, W.ld, C.v, C.ld, 0.0, N.v, N.ld);
    }
    else
    {
        my_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, N.r, M.r, M.r, 1.0, C.v, C.ld, M.v, M.ld, 0.0, W.v, W.ld);
        my_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N.r, M.r, M.r, 1.0, W.v, W.ld, C.v, C.ld, 0.0, N.v, N.ld);
    }
}
