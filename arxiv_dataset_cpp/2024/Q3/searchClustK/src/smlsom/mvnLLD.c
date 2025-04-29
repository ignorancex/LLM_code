#include "mvnLLD.h"

double ***A;
double **diagL;
int *flgs;

void init_params(int M, int p) {
  int i,m;

  A = (double ***)malloc(M * sizeof(double **)); //AはM*p*pの行列　中身なし
  for (m = 0; m < M; m++) {
    A[m] = (double **)malloc(p * sizeof(double *));
    for (i = 0; i < p; i++)
      A[m][i] = (double *)malloc(p * sizeof(double));
  }
  diagL = (double **)malloc(M * sizeof(double *)); //diagLはM*pの0行列 ノードと参照ベクトルの行列と思われる
  for (m = 0; m < M; m++) {
    diagL[m] = (double *)malloc(p * sizeof(double));
    for (i = 0; i < p; i++)
      diagL[m][i] = 0.0;
  }
  flgs = (int *)malloc(M * sizeof(int)); //M次元0ベクトル
  for (m = 0; m < M; m++)
    flgs[m] = 0;
}

void free_params(int M, int p) {
  int i,m;

  for (m = 0; m < M; m++) {
    free(diagL[m]);
    for (i = 0; i < p; i++)
      free(A[m][i]);
  }
  free(flgs);
}

double lld_mvnorm(double *xi, double *mu1, int m, int p) {
  int i,j;
  double tmp;
  double exp = 0.0;
  double det = 1.0;

  // calculate value of LLD
  // |A| = |LL^t| = |L|^2
  // |L| = prod(diag(L))
  // A was replaced in L^-1
  for (i = 0; i < p; i++) {
    tmp = 0.0;
    for (j = 0; j <= i; j++)
      tmp += A[m][i][j] * (xi[j] - mu1[j]); //L^-1 (x-mu)
    exp += tmp * tmp;
    det *= diagL[m][i];
  }

  return (log(det) + 0.5*exp);
}

int choldc(double **a, double *p, int n) {
  int i,j,k;
  double sum;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      for (sum = a[i][j], k = i-1; k >= 0; k--) sum -= a[i][k] * a[j][k];
      if (i == j) {
        if (sum <= 0) {
          return -1;
        }
        p[i] = sqrt(sum);
      } else {
        a[j][i] = sum / p[i];
      }
    }
  }

  return 0;
}

int updateA(double *mu1, double *mu2, int m, int p) {
  int i,j,k;
  int flg;

  // calculate matrix A that is moment estimator for Sigma
  for (j = 0; j < p; j++) {
    for (i = 0; i < p; i++) {
      A[m][i][j] = mu2[m*p*p+ i + j*p] - mu1[m*p + i] * mu1[m*p + j];
    }
  }

  // cholesky decomposition of A that is for calculating the likelihood
  // A = LL^t where L is a lower triangle matrix
  flg = choldc(A[m],diagL[m],p);

  // calculate inverse of L
  // A was replaced in L
  double sum;
  for (i = 0; i < p; i++) {
    A[m][i][i] = 1.0 / diagL[m][i];
    for (j = i+1; j < p; j++) {
      sum = 0.0;
      for (k = i; k < j; k++)
        sum -= A[m][j][k] * A[m][k][i];
      A[m][j][i] = sum / diagL[m][j];
    }
  }

  return flg;
}
