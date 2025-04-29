#include "em_gmm.h"
#include "mixt4.h"

void test_estimate_mixprop(int *comp, double *props, double *n_m, double *dmover2, int *n, int *M) {
  int m;

  estimate_mixprop(*comp, props, *n_m, *dmover2, *n, *M);

  printf("props = ");
  for (m = 0; m < *M; m++)
    printf("%.5f ", props[m]);
  printf("\n");
}

void test_lshift_weights(double *weights, int *comp, int *n, int *M, int *Mmax) {
  lshift_weights(weights, *comp, *n, *M, *Mmax);
}
