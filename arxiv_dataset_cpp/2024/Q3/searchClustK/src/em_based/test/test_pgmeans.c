#include "kstest.h"
#include "pgmeans.h"

void test_ksone(double *x, int *n, int *M, double *pi, double *mu, double *sigma) {
  int i;
  double d, prob;
  double *y = (double *)malloc(sizeof(double) * (*n));

  // calculate cdf with sorted inputs
  qsort(x, *n, sizeof(double), asc_double);
  for (i = 0; i < *n; i++)
    y[i] = pmixnorm(x[i], *M, pi, mu, sigma);

  // calculate ks stats and p-value
  ksone(y, *n, &d, &prob);

  // result
  printf("D = %.5f pval = %.5f\n", d, prob);

  free(y);
}

void test_create_projection(int *pp) {
  int j, p = *pp;
  double *P = (double *)malloc(sizeof(double) * p);

  RANDIN;

  create_projection(P, p);

  RANDOUT;

  printf("P = ");
  for (j = 0; j < p; j++)
    printf("%.5f ", P[j]);
  printf("\n");

  free(P);
}

void test_projection_mean(double *P, double *mus, double *prj_mus, int *p, int *M)
{
  projection_mean(P, mus, prj_mus, *p, *M);
}

void test_projection_fullcov(double *P, double *covs, double *prj_covs, int *p, int *M)
{
  projection_fullcov(P, covs, prj_covs, *p, *M);
}

void test_projection_diagcov(double *P, double *sigs, double *prj_sigs, int *p, int *M)
{
  projection_diagcov(P, sigs, prj_sigs, *p, *M);
}
void test_projection_x(double *P, double *data, double *prj_x, int *n, int *p)
{
  projection_x(P, data, prj_x, *n, *p);
}

void test_add_onecluster_props(double *props, double *new_props, int *M)
{
  add_onecluster_props(props, new_props, *M);
}

void test_add_onecluster_mean(double *xi, double *mus, double *new_mus, int *p, int *M)
{
  add_onecluster_mean(xi, mus, new_mus, *p, *M);
}

void test_add_onecluster_fullcov(double *covs, double *new_covs, int *p, int *M)
{
  add_onecluster_fullcov(covs, new_covs, *p, *M);
}

void test_add_onecluster_diagcov(double *sigs, double *new_sigs, int *p, int *M)
{
  add_onecluster_diagcov(sigs, new_sigs, *p, *M);
}

void test_copy_prop(double *new_props, double *props, int *M)
{
  copy_prop(new_props, props, *M);
}

void test_copy_mean(double *new_mus, double *mus, int *p, int *M)
{
  copy_mean(new_mus, mus, *p, *M);
}

void test_copy_fullcov(double *new_covs, double *covs, int *p, int *M)
{
  copy_fullcov(new_covs, covs, *p, *M);
}

void test_copy_diagcov(double *new_sigs, double *sigs, int *p, int *M)
{
  copy_diagcov(new_sigs, sigs, *p, *M);
}
