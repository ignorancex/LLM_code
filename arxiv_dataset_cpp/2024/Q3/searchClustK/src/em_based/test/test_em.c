#include "em_gmm.h"

void test_list_fullcov(double *x, double *mus, double *covs, int *pM, int *pp) {
  int M = *pM, p = *pp;
  int m, i, j;
  params_link pl, params_fc;
  double loglik;

  params_fc = init_list_fullcov(mus, covs, M, p);

  pl = params_fc;
  for (m = 0; m < M; m++) {
    printf("%d: Mu = ", m);
    for (j = 0; j < p; j++)
      printf("%.3f ", pl->mu[j]);
    printf("\n");

    printf("%d : Cov\n", m);
    for (i = 0; i < p; i++) {
      for (j = 0; j < p; j++)
        printf("%.3f ", pl->cov[i * p + j]);
      printf("\n");
    }
    pl = pl->next;
  }

  pl = params_fc;
  for (m = 0; m < M; m++) {
    loglik = mvnorm_fullcov(x, pl, p);
    printf("loglik = %.5f\n", loglik);
    pl = pl->next;
  }

  free_list_fullcov(params_fc);
}

void test_list_diagcov(double *x, double *mus, double *sigs, int *pM, int *pp) {
  int M = *pM, p = *pp;
  int m, i, j;
  params_link pl, params_dc;
  double loglik;

  params_dc = init_list_diagcov(mus, sigs, M, p);

  pl = params_dc;
  for (m = 0; m < M; m++) {
    printf("%d: mus = ", m);
    for (j = 0; j < p; j++)
      printf("%.3f ", pl->mu[j]);
    printf("\n");

    printf("%d : sigs = ", m);
    for (j = 0; j < p; j++)
      printf("%.3f ", pl->sig[j]);
    printf("\n");

    pl = pl->next;
  }

  pl = params_dc;
  for (m = 0; m < M; m++) {
    loglik = mvnorm_diagcov(x, pl, p);
    printf("loglik = %.5f\n", loglik);
    pl = pl->next;
  }

  free_list_diagcov(params_dc);
}

void test_estep_fullcov(double *data, double *weights, double *props, double *mus, double *covs, int *pn, int *pM, int *pp) {
  int n = *pn, M = *pM, p = *pp;
  int m, i, j;
  double loglik;
  params_link params_fc;

  params_fc = init_list_fullcov(mus, covs, M, p);

  estep(params_fc, mvnorm_fullcov, data, weights, props, n, p, M);

  free_list_fullcov(params_fc);
}

void test_estep_diagcov(double *data, double *weights, double *props, double *mus, double *sigs, int *pn, int *pM, int *pp) {
  int n = *pn, M = *pM, p = *pp;
  int m, i, j;
  double loglik;
  params_link params_dc;

  params_dc = init_list_diagcov(mus, sigs, M, p);

  estep(params_dc, mvnorm_diagcov, data, weights, props, n, p, M);

  free_list_diagcov(params_dc);
}


void test_mstep_fullcov(double *data, double *weights, double *props, double *mus, double *covs, int *pn, int *pM, int *pp) {
  int n = *pn, M = *pM, p = *pp;
  int m, i, j;
  double loglik, sum;

  params_link pl, params_fc;

  params_fc = init_list_fullcov(mus, covs, M, p);

  estep(params_fc, mvnorm_fullcov, data, weights, props, n, p, M);

  pl = params_fc;
  for (m = 0; m < M; m++) {
    sum = mstep_fullcov(pl, data, &weights[m*n], n, p);
    props[m] = sum / n;
    pl = pl->next;
  }

  free_list_fullcov(params_fc);
}

void test_mstep_diagcov(double *data, double *weights, double *props, double *mus, double *covs, int *pn, int *pM, int *pp) {
  int n = *pn, M = *pM, p = *pp;
  int m, i, j;
  double loglik, sum;

  params_link pl, params_dc;

  params_dc = init_list_diagcov(mus, covs, M, p);

  estep(params_dc, mvnorm_diagcov, data, weights, props, n, p, M);

  pl = params_dc;
  for (m = 0; m < M; m++) {
    sum = mstep_diagcov(pl, data, &weights[m*n], n, p);
    props[m] = sum / n;
    pl = pl->next;
  }

  free_list_diagcov(params_dc);
}
