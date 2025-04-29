#include "utils.h"

void copy_prop(double *new_props, double *props, int M)
{
  int m;

  for (m = 0; m < M; m++)
    props[m] = new_props[m];
}

void copy_mean(double *new_mus, double *mus, int p, int M)

{
  int j, m;

  for (m = 0; m < M; m++) {
    for (j = 0; j < p; j++)
      mus[m*p + j] = new_mus[m*p + j];
  }
}

void copy_fullcov(double *new_covs, double *covs, int p, int M)
{
  int i, j, m;

  for (m = 0; m < M; m++) {
    for (i = 0; i < p; i++) {
      for (j = 0; j < p; j++)
        covs[m*p*p + i*p + j] = new_covs[m*p*p + i*p + j];
    }
  }
}

void copy_diagcov(double *new_sigs, double *sigs, int p, int M)
{
  int j, m;

  for (m = 0; m < M; m++) {
    for (j = 0; j < p; j++)
      sigs[m*p + j] = new_sigs[m*p + j];
  }
}

int asc_double( const void *x, const void *y )
{
  double tmp;
  tmp = *(double*)x - *(double *)y;
  if (tmp > 0)
    return 1;
  return -1;
}
