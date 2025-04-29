#include "utils.h"
#include "dip.h"
#include "diptest.h"

void test_euc_dists(double *X, int *pn, int *pp, double *D) {
  int n = *pn, p = *pp;
  int i, j;
  double d;

  euc_dists(X, n, p, D);

  i = (int)(0.5 * n);

  printf("i=%d:\n", i);
  for (j = 0; j < i; j++) {
    d = D[(int)(0.5 * i * (i-1)) + j];
    printf(" %.4f", d);
  }
  for (j = i+1; j < n; j++) {
    d = D[(int)(0.5 * j * (j-1)) + i];
    printf( "%.4f ", d);
  } //d is distances from viewer ck[i]

  printf("\n");


}

void test_diptst(double *d, int *pn, double *dip) {
  int n = *pn;

  int low_hi[4];
  int ifault;
  int *gcm = (int *)malloc(sizeof(int) * (n));
  int *lcm = (int *)malloc(sizeof(int) * (n));
  int *mn = (int *)malloc(sizeof(int) * (n));
  int *mj = (int *)malloc(sizeof(int) * (n));
  const int min_is_0 = 0;
  const int debug_diptst = 0;

  diptst(d, pn, dip, low_hi, &ifault, gcm, lcm, mn, mj, &min_is_0, &debug_diptst);

  free(gcm); free(lcm); free(mn); free(mj);
}

void test_testCluster_unimodarity(double *X, int *pn, int *pp, int *c, double *pa, double *D) {
  int n = *pn, p = *pp;
  double a = *pa;

  euc_dists(X, n, p, D);

  testCluster_unimodality(D, n, c, a, 0.01, 1);
}
