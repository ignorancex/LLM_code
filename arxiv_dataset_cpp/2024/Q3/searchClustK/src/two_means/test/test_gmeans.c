#include "gmeans.h"
#include "utils.h"


void test_first_prcomp(double *X, int *pp, int *ck, int *pnk, double *mu, double *vec1, double *vec2) {
  int i, j;
  int p = *pp, nk = *pnk;
  double lambda;

  double **cov = (double **)malloc(sizeof(double *) * p);
  for (i = 0; i < p; i++)
    cov[i] = (double *)malloc(sizeof(double) * p);

  covariance(X, p, ck, nk, mu, cov);

  printf("covariance:\n");
  for (i = 0; i < p; i++) {
    for (j = 0; j < p; j++)
      printf("%2.4f ", cov[i][j]);
    printf("\n");
  }

  lambda = first_prcomp(X, p, ck, nk, mu, cov, vec1, vec2);

  printf("lambda = %.5f\n", lambda);

  for (i = 0; i < p; i++) free(cov[i]);
  free(cov);
}

void test_normalCDF(double *value) {
  printf("cfd = %.5f\n", normalCDF(*value));
}

void test_ADstatic(double *x, int *pn, double *d) {
  int i, n = *pn;
  double *z;

  qsort(x, n, sizeof(double), asc_double);

  z = (double *)malloc(sizeof(double) * n);
  for (i = 0; i < n; i++) z[i] = normalCDF(x[i]);

  *d = ADstatic(z, n);

  free(z);
}

void test_projection(double *X, int *p, int *ck, int *nk, double *vec1, double *vec2, double *prj_x) {
  projection(X, *p, ck, *nk, vec1, vec2, prj_x);
}
