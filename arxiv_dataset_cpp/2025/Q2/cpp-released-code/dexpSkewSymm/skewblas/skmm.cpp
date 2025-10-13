#include "skmm.hpp"

void skewblas_skmm(CHAR_TYPE side, INTE_TYPE m, INTE_TYPE n, REAL_TYPE alpha, REAL_TYPE *A, INTE_TYPE lda, REAL_TYPE *B, INTE_TYPE ldb, REAL_TYPE beta, REAL_TYPE *C, INTE_TYPE ldc)
{

    if (m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0))
        return;

    if (alpha == 0.0)
    {
        if (ldc == m)
            if (beta == 0.0)
                memset(C, 0, sizeof(REAL_TYPE) * m * n);
            else
                cblas_dscal(m * n, beta, C, 1);
        else
        {
            REAL_TYPE *col_ptr = C;
            if (beta == 0.0)
                for (auto col_ind = 0; col_ind < n; col_ind++, col_ptr += ldc)
                    memset(col_ptr, 0, sizeof(REAL_TYPE) * m);
            else
                for (auto col_ind = 0; col_ind < n; col_ind++, col_ptr += ldc)
                    cblas_dscal(m, beta, col_ptr, 1);
        }

        return;
    }

    REAL_TYPE temp1, temp2;

    View_ColMat<REAL_TYPE> VC(C, ldc, m, n);

    if (side == 'L')
    {
        REAL_TYPE *b_col = B, *c_col = C, *a_col = A + (m - 1) * lda;
        for (auto col_ind = 0; col_ind < n; col_ind++, b_col += ldb, c_col += ldc, a_col = A + (m - 1) * lda)
            for (auto row_ind = m - 1; row_ind >= 0; row_ind--, a_col -= lda) // row_ind = m - 1 performs no update as A(m, m) = 0.
            {
                temp1 = alpha * b_col[row_ind];
                temp2 = 0.0;
                for (auto sum_ind = row_ind + 1; sum_ind < m; sum_ind++)
                {
                    c_col[sum_ind] += temp1 * a_col[sum_ind];
                    temp2 += b_col[sum_ind] * a_col[sum_ind];
                }
                if (beta == 0)
                    c_col[row_ind] = -alpha * temp2;
                else
                    c_col[row_ind] = beta * c_col[row_ind] - alpha * temp2;
            }
    }
    else
    {
        // 329           DO 170 j = 1,n
        // 340               DO 140 k = 1,j - 1
        // 344                   temp1 = alpha*a(j,k)
        // 346                   DO 130 i = 1,m
        // 347                       c(i,j) = c(i,j) + temp1*b(i,k)
        // 348   130             CONTINUE
        // 349   140         CONTINUE
        // 350               DO 160 k = j + 1,n
        // 354                   temp1 = alpha*a(k,j)
        // 356                   DO 150 i = 1,m
        // 357                       c(i,j) = c(i,j) + temp1*b(i,k)
        // 358   150             CONTINUE
        // 359   160         CONTINUE
        // 360   170     CONTINUE
        REAL_TYPE *b_col = B, *c_col = C, *a_col = A, *a_row = A;
        for (auto col_ind = 0; col_ind < n; col_ind++, a_col += lda, c_col += ldc, a_row += 1)
        {
            b_col = B;
            for (auto sum_ind = 0; sum_ind < col_ind; sum_ind++, b_col += ldb)
            {
                temp1 = alpha * a_row[sum_ind * lda];
                for (auto row_ind = 0; row_ind < m; row_ind++)
                    c_col[row_ind] -= temp1 * b_col[row_ind];
            }
            b_col += ldb;
            for (auto sum_ind = col_ind + 1; sum_ind < n; sum_ind++, b_col += ldb)
            {
                temp1 = alpha * a_col[sum_ind];
                for (auto row_ind = 0; row_ind < m; row_ind++)
                    c_col[row_ind] += temp1 * b_col[row_ind];
            }
        }
    }
}

void skewblas_skmm(CHAR_TYPE side, REAL_TYPE alpha, SkewSymmMat &A, ColMat<REAL_TYPE> &B, REAL_TYPE beta, ColMat<REAL_TYPE> &C)
{
    skewblas_skmm(side, C.r, C.c, alpha, A.v, A.r, B.v, B.r, beta, C.v, C.r);
}
void skewblas_skmm(CHAR_TYPE side, REAL_TYPE alpha, SkewSymmMat &A, View_ColMat<REAL_TYPE> &B, REAL_TYPE beta, ColMat<REAL_TYPE> &C)
{
    skewblas_skmm(side, C.r, C.c, alpha, A.v, A.r, B.v, B.ld, beta, C.v, C.r);
}
void skewblas_skmm(CHAR_TYPE side, REAL_TYPE alpha, SkewSymmMat &A, ColMat<REAL_TYPE> &B, REAL_TYPE beta, View_ColMat<REAL_TYPE> &C)
{
    skewblas_skmm(side, C.r, C.c, alpha, A.v, A.r, B.v, B.r, beta, C.v, C.ld);
}
void skewblas_skmm(CHAR_TYPE side, REAL_TYPE alpha, SkewSymmMat &A, View_ColMat<REAL_TYPE> &B, REAL_TYPE beta, View_ColMat<REAL_TYPE> &C)
{
    skewblas_skmm(side, C.r, C.c, alpha, A.v, A.r, B.v, B.ld, beta, C.v, C.ld);
}

void skewblas_skmm(CHAR_TYPE side, REAL_TYPE alpha, ColMat<REAL_TYPE> &A, ColMat<REAL_TYPE> &B, REAL_TYPE beta, ColMat<REAL_TYPE> &C)
{
    skewblas_skmm(side, C.r, C.c, alpha, A.v, A.r, B.v, B.r, beta, C.v, C.r);
}
void skewblas_skmm(CHAR_TYPE side, REAL_TYPE alpha, View_ColMat<REAL_TYPE> &A, ColMat<REAL_TYPE> &B, REAL_TYPE beta, ColMat<REAL_TYPE> &C)
{
    skewblas_skmm(side, C.r, C.c, alpha, A.v, A.ld, B.v, B.r, beta, C.v, C.r);
}
void skewblas_skmm(CHAR_TYPE side, REAL_TYPE alpha, ColMat<REAL_TYPE> &A, View_ColMat<REAL_TYPE> &B, REAL_TYPE beta, ColMat<REAL_TYPE> &C)
{
    skewblas_skmm(side, C.r, C.c, alpha, A.v, A.r, B.v, B.ld, beta, C.v, C.r);
}
void skewblas_skmm(CHAR_TYPE side, REAL_TYPE alpha, ColMat<REAL_TYPE> &A, ColMat<REAL_TYPE> &B, REAL_TYPE beta, View_ColMat<REAL_TYPE> &C)
{
    skewblas_skmm(side, C.r, C.c, alpha, A.v, A.r, B.v, B.r, beta, C.v, C.ld);
}
void skewblas_skmm(CHAR_TYPE side, REAL_TYPE alpha, View_ColMat<REAL_TYPE> &A, View_ColMat<REAL_TYPE> &B, REAL_TYPE beta, ColMat<REAL_TYPE> &C)
{
    skewblas_skmm(side, C.r, C.c, alpha, A.v, A.ld, B.v, B.ld, beta, C.v, C.r);
}
void skewblas_skmm(CHAR_TYPE side, REAL_TYPE alpha, View_ColMat<REAL_TYPE> &A, ColMat<REAL_TYPE> &B, REAL_TYPE beta, View_ColMat<REAL_TYPE> &C)
{
    skewblas_skmm(side, C.r, C.c, alpha, A.v, A.ld, B.v, B.r, beta, C.v, C.ld);
}
void skewblas_skmm(CHAR_TYPE side, REAL_TYPE alpha, ColMat<REAL_TYPE> &A, View_ColMat<REAL_TYPE> &B, REAL_TYPE beta, View_ColMat<REAL_TYPE> &C)
{
    skewblas_skmm(side, C.r, C.c, alpha, A.v, A.r, B.v, B.ld, beta, C.v, C.ld);
}
void skewblas_skmm(CHAR_TYPE side, REAL_TYPE alpha, View_ColMat<REAL_TYPE> &A, View_ColMat<REAL_TYPE> &B, REAL_TYPE beta, View_ColMat<REAL_TYPE> &C)
{
    skewblas_skmm(side, C.r, C.c, alpha, A.v, A.ld, B.v, B.ld, beta, C.v, C.ld);
}