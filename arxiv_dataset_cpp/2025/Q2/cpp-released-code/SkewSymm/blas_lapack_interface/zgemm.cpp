#include "zgemm.hpp"


// This implementation computes real matrix C = alpha * A * B + beta * C
// using Intel(R) MKL function dgemm, where A, B, and  C are matrices and
// alpha and beta are double precision scalars. The matrices may be sent
// (1)  in the general array representation as follows:
//      (arr, row, col, leading_order, LAYOUT, TRANS);
// (2)  in the customized ColMat or View_ColMat in columnMajorMatrix.hpp

// Note that Users are respobible to manage the matrix size on their own,
// where A is in m x k, B is in k x n and C is in m x n. 

void my_zgemm(LAYOUT_FLAG LAYOUT, TRANSP_FLAG TRANSA, TRANSP_FLAG TRANSB, INTE_TYPE m, INTE_TYPE n, INTE_TYPE k, 
    const CMPX_TYPE &alpha, CMPX_TYPE* a_arr, INTE_TYPE lda, CMPX_TYPE* b_arr, INTE_TYPE ldb, 
    const CMPX_TYPE &beta, CMPX_TYPE* c_arr, INTE_TYPE ldc)
{
    cblas_zgemm(LAYOUT, TRANSA, TRANSB, m, n, k, &alpha, a_arr, lda, b_arr, ldb, &beta, c_arr, ldc);
}

void my_zgemm(TRANSP_FLAG TRANSA, TRANSP_FLAG TRANSB, const CMPX_TYPE &alpha, const View_ColMat<CMPX_TYPE> &A, const View_ColMat<CMPX_TYPE> &B, 
              const CMPX_TYPE &beta, const View_ColMat<CMPX_TYPE> &C)
{
    cblas_zgemm(CblasColMajor, TRANSA, TRANSB, A.r, B.c, A.c, &alpha, A.v, A.ld, B.v, B.ld, &beta, C.v, C.ld);
}