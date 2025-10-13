#include "skmv.hpp"

void skewblas_skmv(INTE_TYPE n, REAL_TYPE alpha, REAL_TYPE *A, INTE_TYPE lda, REAL_TYPE *x, INTE_TYPE incx, REAL_TYPE beta, REAL_TYPE *y, INTE_TYPE incy)
{
    if (n <= 0)
        return;

    // Scale y by beta

    for (auto i = 0; i < n; i += incy)
        y[i] *= beta;

    if (alpha == 0.0)
        return;
    REAL_TYPE temp1, temp2;
    if ((incx == 1) || (incy == 1))
    {
        REAL_TYPE *col_ptr = A;
        for (auto col_ind = 0; col_ind < n; col_ind++, col_ptr += lda)
        {
            temp1 = alpha * x[col_ind];
            temp2 = 0;
            for (auto row_ind = col_ind + 1; row_ind < n; row_ind++)
            {
                y[row_ind] += temp1 * col_ptr[row_ind];
                temp2 -= col_ptr[row_ind] * x[row_ind];
            }
            y[col_ind] += alpha * temp2;
        }
    }
    else
    {
        REAL_TYPE *col_ptr = A;
        auto cx = 0, cy = 0, rx = 0, ry = 0;
        for (auto col_ind = 0; col_ind < n; col_ind++, col_ptr += lda, cx += incx, cy += incy)
        {
            temp1 = alpha * x[cx];
            temp2 = 0;
            rx = cx + incx;
            ry = cy + incy;
            for (auto row_ind = col_ind + 1; row_ind < n; row_ind++, rx += incx, ry += incy)
            {
                y[ry] += temp1 * col_ptr[row_ind];
                temp2 -= col_ptr[row_ind] * x[rx];
            }
            y[cy] += alpha * temp2;
        }
    }
};

void skewblas_skmv(REAL_TYPE alpha, SkewSymmMat &A, REAL_TYPE *x, INTE_TYPE incx, REAL_TYPE beta, REAL_TYPE *y, INTE_TYPE incy)
{
    skewblas_skmv(A.d, alpha, A.v, A.d, x, incx, beta, y, incy);
};
void skewblas_skmv(REAL_TYPE alpha, SkewSymmMat &A, ArrVec<REAL_TYPE> &x, REAL_TYPE beta, ArrVec<REAL_TYPE> &y)
{
    skewblas_skmv(A.d, alpha, A.v, A.d, x.v, 1, beta, y.v, 1);
};
void skewblas_skmv(REAL_TYPE alpha, SkewSymmMat &A, View_ArrVec<REAL_TYPE> &x, REAL_TYPE beta, View_ArrVec<REAL_TYPE> &y)
{
    skewblas_skmv(A.d, alpha, A.v, A.d, x.v, 1, beta, y.v, 1);
};
void skewblas_skmv(REAL_TYPE alpha, ColMat<REAL_TYPE> &A, ArrVec<REAL_TYPE> &x, REAL_TYPE beta, ArrVec<REAL_TYPE> &y)
{
    skewblas_skmv(A.r, alpha, A.v, A.r, x.v, 1, beta, y.v, 1);
};
void skewblas_skmv(REAL_TYPE alpha, ColMat<REAL_TYPE> &A, View_ArrVec<REAL_TYPE> &x, REAL_TYPE beta, View_ArrVec<REAL_TYPE> &y)
{
    skewblas_skmv(A.r, alpha, A.v, A.r, x.v, 1, beta, y.v, 1);
};