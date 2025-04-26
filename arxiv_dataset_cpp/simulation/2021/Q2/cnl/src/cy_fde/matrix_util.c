#include "matrix_util.h"


void set_zval(DCOMPLEX *vector,  DCOMPLEX value, int d){
  for (size_t i = 0; i < d; i++, vector++) {
    *vector = value;
  }
}
/*
* A : M x K
* B : K x N
* C : M x N
*/
void my_zgemm(const int M, const int N, const int K, const DCOMPLEX alpha, const DCOMPLEX  *A, const  DCOMPLEX *B, const DCOMPLEX beta, DCOMPLEX * Out){
  if (beta == 0){
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n, ++Out){
        *Out =0;
        const DCOMPLEX *A_ptr = A + K*m;
        const DCOMPLEX *B_ptr = B + n;
        DCOMPLEX sum = 0;
        for (int  k = 0; k < K; ++k, ++A_ptr, B_ptr+= N) {
          sum+= *A_ptr * ( *B_ptr);
        }
        *Out += alpha*sum;
        if( isnanz(*Out) )printf("(my_zgemm) nan at (%i,%i)\n", m,n);
      }
    }
  }else{
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n, ++Out){
      *Out *= beta;
      const DCOMPLEX *A_ptr = A + K*m;
      const DCOMPLEX *B_ptr = B + n;
      DCOMPLEX sum = 0;
      for (int  k = 0; k < K; ++k, ++A_ptr, B_ptr+= N) {
        sum+= *A_ptr * ( *B_ptr);
        /*
        printf("m=%d, n=%d\n",m,n);
        printf("A;%e\n", abs(*A_ptr - A[K*m + k] ));
        printf("B;%e\n", abs(*B_ptr - B[k*N + n] ));
        */
      }
      *Out += alpha*sum;
      if( isnanz(*Out) )printf("(myzgemm) nan at (%i,%i)\n", m,n);
    }
  }
}
}


void my_zax(const int dim, const DCOMPLEX alpha, DCOMPLEX * o_vec){
for (int i = 0; i < dim; ++i, ++o_vec) {
  *o_vec *= alpha;
  }
}
void my_dax(const int dim, double alpha, double * o_vec){
for (int i = 0; i < dim; ++i, ++o_vec) {
  *o_vec *= alpha;
  }
}



void inv_ow(const int dim,  DCOMPLEX *o_vec){
  for (int i = 0; i < dim; ++i, ++o_vec) {
    *o_vec = 1./ *o_vec;
    }
}



void inv2by2(const DCOMPLEX *A, DCOMPLEX *o_inv){
  DCOMPLEX  det = 0;
  det = A[3]*A[0] - A[1]*A[2];
  if (det == 0){
    exit(EXIT_FAILURE);
  }
  o_inv[0] = A[3]/det;
  o_inv[1] = - A[1]/det;
  o_inv[2] = - A[2]/det;
  o_inv[3] = A[0]/det;
}



void inv2by2_overwrite(DCOMPLEX *o_A){
  DCOMPLEX  det =  o_A[3]*o_A[0] - o_A[1]*o_A[2];
  if (det == 0){
    exit(EXIT_FAILURE);
  }
  DCOMPLEX temp = o_A[0]/det;
  o_A[0] = o_A[3]/det;
  o_A[1] = - o_A[1]/det;
  o_A[2] = - o_A[2]/det;
  o_A[3] = temp;
}


void outer(const int dim, const DCOMPLEX * v , const DCOMPLEX  *w , DCOMPLEX *o_mat){
  for (int m = 0; m < dim; ++m, ++v) {
    const DCOMPLEX *w_ptr = w;
    for (int n = 0; n < dim; ++n, ++w_ptr, ++o_mat) {
      *o_mat = *v * conj(*w);
    }
  }
}



void my_zaxpy(const int dim, const DCOMPLEX alpha, const DCOMPLEX * v, DCOMPLEX *o_vec){
  for (int i = 0; i < dim; ++i, ++v, ++o_vec) {
    *o_vec +=  alpha*(*v);
  }
}

void my_zdot(const int dim, const DCOMPLEX *v , DCOMPLEX* o_vec){
  for (int i = 0; i < dim; ++i, ++v, ++o_vec) {
    *o_vec *=  *v;
  }
}


void z_isnan(const int dim, const DCOMPLEX *v){
for (int i = 0; i < dim; ++i) {
  if ( isnan(creal(v[i])) ||  isnan(cimag(v[i]) ) ){
  printf("(z_isnan)Nan Error at i = %i\n", i);
  assert(false);
  }
}
}
