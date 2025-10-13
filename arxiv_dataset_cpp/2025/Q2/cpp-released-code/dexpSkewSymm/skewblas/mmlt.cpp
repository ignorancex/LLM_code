#include "mmlt.hpp"

// void mmlt(CHAR_TYPE transa, CHAR_TYPE transb, INTE_TYPE m, INTE_TYPE n, INTE_TYPE k,
//           REAL_TYPE alpha,
//           REAL_TYPE *A, INTE_TYPE lda,
//           REAL_TYPE *B, INTE_TYPE ldb,
//           REAL_TYPE beta,
//           REAL_TYPE *C, INTE_TYPE ldc)
// {

//     if ((alpha == 0.0) && (beta == 1.0))
//         return;

//     // Initialize lower triangle of C to zero
//     for (auto j = 0; j < n; ++j)
//     {
//         auto i_start = j + 1;
//         auto i_end = m - 1;
//         auto length = i_end + 1 - i_start;
//         if (length > 0)
//             if (beta == 0)
//                 memset(C + i_start + j * ldc, 0, sizeof(length));
//             else
//                 for (auto i = i_start; i <= i_end; ++i)
//                     C[i + j * ldc] *= beta;
//     }

//     if (transa == 'N' && transb == 'N')
//     {
//         // C += A * B
//         for (int j = 0; j < n; ++j)
//         {
//             for (int p = 0; p < k; ++p)
//             {
//                 const double b_val = alpha * B[p + j * ldb];
//                 const double *a_col = A + p * lda;
//                 for (int i = j + 1; i < m; ++i)
//                 {
//                     C[i + j * ldc] += a_col[i] * b_val;
//                 }
//             }
//         }
//     }
//     else if (transa == 'N' && transb == 'T')
//     {
//         // C += A * B^T
//         for (int j = 0; j < n; ++j)
//         {
//             for (int p = 0; p < k; ++p)
//             {
//                 const double b_val = alpha * B[j + p * ldb];
//                 const double *a_col = A + p * lda;
//                 for (int i = j + 1; i < m; ++i)
//                 {
//                     C[i + j * ldc] += a_col[i] * b_val;
//                 }
//             }
//         }
//     }
//     else if (transa == 'T' && transb == 'N')
//     {
//         // C += A^T * B
//         for (int i = 0; i < m; ++i)
//         {
//             const int j_end = i < n - 1 ? i : n - 1;
//             for (int p = 0; p < k; ++p)
//             {
//                 const double a_val = alpha * A[p + i * lda];
//                 const double *b_col = B + p * ldb;
//                 for (int j = 0; j < j_end; ++j)
//                 {
//                     C[i + j * ldc] += a_val * b_col[j];
//                 }
//             }
//         }
//     }
//     else
//     {
//         // C += A^T * B^T
//         for (int i = 0; i < m; ++i)
//         {
//             const int j_end = i < n - 1 ? i : n - 1;
//             for (int p = 0; p < k; ++p)
//             {
//                 const double a_val = alpha * A[p + i * lda];
//                 const double *b_row = B + p * ldb;
//                 for (int j = 0; j < j_end; ++j)
//                 {
//                     C[i + j * ldc] += a_val * b_row[j];
//                 }
//             }
//         }
//     }
// }

INTE_TYPE _mmlt_blk_size(INTE_TYPE d)
{
    if (d > 600)
        return 60;
    else if (d > 20)
        return 6;
    else
        return d;
}

void skewblas_mmlt(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, INTE_TYPE m, INTE_TYPE n, INTE_TYPE k,
                   REAL_TYPE alpha,
                   REAL_TYPE *A, INTE_TYPE lda,
                   REAL_TYPE *B, INTE_TYPE ldb,
                   REAL_TYPE beta,
                   REAL_TYPE *C, INTE_TYPE ldc)
{
    INTE_TYPE d = m < n ? m : n;

    INTE_TYPE m_curr = m, blk_done = 0;
    REAL_TYPE *A_curr = A, *B_curr = B, *C_curr = C;

    INTE_TYPE blk_curr = _mmlt_blk_size(d - blk_done);

    while (blk_curr > 0)
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_curr, blk_curr, k, alpha, A_curr, lda, B_curr, ldb, beta, C_curr, ldc);
        blk_done += blk_curr;
        A_curr += blk_curr;
        B_curr += blk_curr * ldb;
        C_curr += blk_curr * (ldc + 1);

        m_curr -= blk_curr;
        blk_curr = _mmlt_blk_size(d - blk_done);
    }

    // if (d < 20)
    //     cblas_dgemm(CblasColMajor, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    // else
    // {

    // }
}

void skewblas_mmlt(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, INTE_TYPE m, INTE_TYPE n, INTE_TYPE k,
                   REAL_TYPE alpha,
                   REAL_TYPE *A, INTE_TYPE lda,
                   REAL_TYPE *B, INTE_TYPE ldb,
                   REAL_TYPE beta,
                   REAL_TYPE *C, INTE_TYPE ldc, INTE_TYPE blk)
{
    INTE_TYPE d = m < n ? m : n;

    INTE_TYPE m_curr = m, blk_done = 0;
    INTE_TYPE blk_curr = (d - blk_done > blk) ? blk : d - blk_done;
    REAL_TYPE *A_curr = A, *B_curr = B, *C_curr = C;
    while (blk_curr > 0)
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_curr, blk_curr, k, alpha, A_curr, lda, B_curr, ldb, beta, C_curr, ldc);
        blk_done += blk_curr;
        A_curr += blk_curr;
        B_curr += blk_curr * ldb;
        C_curr += blk_curr * (ldc + 1);

        m_curr -= blk_curr;
        blk_curr = (d - blk_done > blk) ? blk : d - blk_done;
    }
}