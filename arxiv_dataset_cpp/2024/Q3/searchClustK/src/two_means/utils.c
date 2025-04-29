#include "utils.h"

void euc_dists(double *X, int n, int p, double *D)
{
  int i1, i2, j;
  double d, tmp;

  for (i1 = 1; i1 < n; i1++) {
    for (i2 = 0; i2 < i1; i2++) {
      d = 0.0;
      for (j = 0; j < p; j++) {
        tmp = X[i1*p + j] - X[i2*p + j];
        d += tmp * tmp;
      }
      D[(int)(0.5*i1*(i1-1)) + i2] = sqrt(d);
    }
  }
}

int asc_double(const void *x, const void *y)
{
  double tmp;
  tmp = *(double*)x - *(double *)y;
  if (tmp > 0)
    return 1;
  return -1;
}

int asc_int(const void *x, const void *y)
{
  int tmp;
  tmp = *(int *)x - *(int *)y;
  if (tmp > 0)
    return 1;
  return -1;
}

void print_head(double *array)
{
  int i;

  printf("array : ");
  for (i = 0; i < 10; i++)
    printf("%.3f ",array[i]);
  printf("\n");
}
