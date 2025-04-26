#include "test_matrix_util.h"
/*
int test_blas(int result){
   int  d = 5;
   int  k = 6;
   int  l = 5;
   double *A, *B, *C;
   A = malloc(sizeof(double) *d * k);
   B = malloc(sizeof(double) *k * l);
   C = malloc(sizeof(double) *d * l);
   double alpha  = 5.;
   double beta = 1;

   memset(A, 2., sizeof(double) * d * k);
   memset(B, 3., sizeof(double) * k * l);
   memset(C, 7, sizeof(double) * d * l);
   cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
       d, l, k, alpha, A, k, B, l, beta, C, l);

  for (int i = 0; i < d*l; i++) {
    assert(C[i] == (5*2*3 + 7.));
  }

   return result;

}
*/
int test_my_zgemm(int result){
    int M= 5;
    int N = 2;
    int K = 4;

    DCOMPLEX *A, *B, *Out;
    A = malloc(sizeof(DCOMPLEX) *M * K);
    B = malloc(sizeof(DCOMPLEX) *K * N);
    Out = malloc(sizeof(DCOMPLEX) *M * N);
    DCOMPLEX alpha  = 1;
    DCOMPLEX beta = 0;

    memset(A, 0, sizeof(DCOMPLEX) * M * K);
    memset(B, 0, sizeof(DCOMPLEX) * K * N);
    memset(Out, 0, sizeof(DCOMPLEX) * M * N);

    for (int i = 0; i < M*K; i++) {
      A[i] = i + i*I;
    }
    for (int i = 0; i < K*N; i++) {
      B[i] = i;
    }
    my_zgemm(M,N,K,alpha, A ,B,beta, Out);

    DCOMPLEX sum = 0;
    for (int k = 0; k < K; k++) {
      sum += A[k] + B[k*N];
    }
    assert (C[0] == sum);
    return result;
}


int test_inv2by2(int result){
    DCOMPLEX * A, *inv, *temp;
    A = malloc(sizeof(DCOMPLEX) *4);
    inv = malloc(sizeof(DCOMPLEX) *4);
    temp = malloc(sizeof(DCOMPLEX) *4);

    A[0] = 1;
    A[1] = 2;
    A[2] = 3;
    A[3] = 4;
    inv2by2(A, inv);
    my_zgemm(2,2,2, 1.0, A, inv, 1.0, temp);
    assert(temp[0] == 1);
    assert(temp[1] == 0);
    assert(temp[2] == 0);
    assert(temp[3] == 1);

    return result;
}

int test_inv2by2inv2by2_overwrite(int result){
    DCOMPLEX * A, *temp_A, *temp;
    A = malloc(sizeof(DCOMPLEX) *4);
    temp = malloc(sizeof(DCOMPLEX) *4);
    temp_A = malloc(sizeof(DCOMPLEX) *4);

    A[0] = 1;
    A[1] = 2;
    A[2] = 3;
    A[3] = 4;
    temp_A[0] = 1;
    temp_A[1] = 2;
    temp_A[2] = 3;
    temp_A[3] = 4;

    inv2by2_overwrite(A);
    my_zgemm(2,2,2, 1.0, A, temp_A, 0, temp);
    assert(temp[0] == 1);
    assert(temp[1] == 0);
    assert(temp[2] == 0);
    assert(temp[3] == 1);

    return result;
}


int test_outer(int result){
  int dim = 2;
  DCOMPLEX *v, *w, *mat;
  v = malloc(sizeof(DCOMPLEX) *dim);
  w = malloc(sizeof(DCOMPLEX) *dim);
  mat = malloc(sizeof(DCOMPLEX) *dim*dim);
  v[0] = 1;
  v[1] = 0;
  w[0] = 0;
  w[1] = 1 + 2*I;
  outer(dim, v,w, mat);
  assert(mat[0] == 0);
  assert(mat[1] == 1 - 2*I);
  assert(mat[2] == 0);
  assert(mat[3] == 0);
  return result;

}


int test_my_zaxpy(int result){
  int dim = 2;
  DCOMPLEX alpha = 3;
  DCOMPLEX *v, *w;
  v = malloc(sizeof(DCOMPLEX) *dim);
  w = malloc(sizeof(DCOMPLEX) *dim);
  v[0] = 1;
  v[1] = 0;
  w[0] = 0;
  w[1] = 1 + 2*I;
  my_zaxpy(dim, alpha, v,w);
  assert ( w[0] == 3);
  assert ( w[1] == 1 + 2*I);

  return result;
}



int test_my_zdot(int result){
  int dim = 2;
  DCOMPLEX *v, *w;
  v = malloc(sizeof(DCOMPLEX) *dim);
  w = malloc(sizeof(DCOMPLEX) *dim);
  v[0] = 1;
  v[1] = 2;
  w[0] = 0;
  w[1] = 1 + 2*I;
  my_zdot(dim, v, w);
  assert ( w[0] == 0);
  assert ( w[1] == 2 + 4*I);
  return result;
}
