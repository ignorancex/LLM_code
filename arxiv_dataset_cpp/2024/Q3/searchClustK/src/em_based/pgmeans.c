#include "utils.h"
#include "kstest.h"
#include "em_gmm.h"
#include "pgmeans.h"

void create_projection(double *P, int p);
void projection_mean(double *P, double *mus, double *prj_mus, int p, int M);
void projection_fullcov(double *P, double *covs, double *prj_sigmas, int p, int M);
void projection_diagcov(double *P, double *sigs, double *prj_sigmas, int p, int M);
void projection_x(double *P, double *data, double *prj_x, int n, int p);
void add_onecluster_props(double *props, double *new_props, int M);
void add_onecluster_mean(double *xi, double *mus, double *new_mus, int p, int M);
void add_onecluster_fullcov(double *covs, double *new_covs, int p, int M);
void add_onecluster_diagcov(double *sigs, double *new_sigs, int p, int M);
double pmixnorm(double x, int M, double *pi, double *mu, double *sigma);

void pgmeans(double *data, double *weights, double *props, double *mus, double *covs,
             int *pcov_type, int *pn, int *pp, int *pM, int *pMmax,
             double *palpha, int *pnprj, int *pem_restart, int *pitmax, double *pth, int *pdebug)
{
  int i, j, k, m;
  double sum;

  /* from R */
  double alpha = *palpha;
  int nprj = *pnprj;
  int cov_type = *pcov_type;
  int n = *pn, p = *pp, M = *pM, Mmax = *pMmax;
  int itmax = *pitmax;
  double th = *pth;
  int debug = *pdebug;

  /* for adding new cluster */
  double *xi;
  double *new_covs;

  // switch based on covariance type.
  void (*projection_cov)(double *, double *, double *, int, int);
  void (*add_onecluster_cov)(double *, double *, int, int);
  void (*copy_cov)(double *, double *, int, int);

  if (cov_type == 0) {

    new_covs = (double *)malloc(sizeof(double) * Mmax * p * p);
    projection_cov = projection_fullcov;
    add_onecluster_cov = add_onecluster_fullcov;
    copy_cov = copy_fullcov;

  }
  else if(cov_type == 1) {

    new_covs = (double *)malloc(sizeof(double) * Mmax * p);
    projection_cov = projection_diagcov;
    add_onecluster_cov = add_onecluster_diagcov;
    copy_cov = copy_diagcov;

  } else {
    error("such cov_type=%d does not exist.\n", cov_type);
  }

  double *new_props = (double *)malloc(sizeof(double) * Mmax);
  double *new_mus = (double *)malloc(sizeof(double) * Mmax * p);

  /* for projection */
  double *P = (double *)malloc(sizeof(double) * p);
  double *prj_x = (double *)malloc(sizeof(double) * n);
  double *prj_mus = (double *)malloc(sizeof(double) * Mmax);
  double *prj_sigmas = (double *)malloc(sizeof(double) * Mmax);

  /* for ks-test */
  int test_flag = 0;
  double d, prob;
  double *y = (double *)malloc(sizeof(double) * n);

  /* for EM */
  int it, new_M;
  int em_restart = *pem_restart;
  double llik;
  double *logliks = (double *)malloc(sizeof(double) * itmax);
  int countf;
  int silent = 1; // EM debug flag. usually off.

  RANDIN;

  while (M < Mmax) {
    test_flag = 0;

    for (k = 0; k < nprj; k++) {

      // projection
      create_projection(P, p);

      projection_mean(P, mus, prj_mus, p, M);
      (*projection_cov)(P, covs, prj_sigmas, p, M);
      projection_x(P, data, prj_x, n, p);

      // KS test (one sample)
      qsort(prj_x, n, sizeof(double), asc_double);
      for (i = 0; i < n; i++)
        y[i] = pmixnorm(prj_x[i], M, props, prj_mus, prj_sigmas);
      ksone(y, n, &d, &prob);

      if (prob < alpha) {
        test_flag = 1;
        if (debug) {
          printf("M = %d: D = %.5f, prob = %.5f\n", M, d, prob);
          printf(" Null hypothesis was refused.\n");
        }
        break;
      }
    }

    if (test_flag == 0) {
      if (debug)
        printf("no projection refused null hypothesis.\n");
      break;
    }

    // new cluster added
    new_M = M + 1;

    llik = -DBL_MAX;
    for (it = 0; it < em_restart; it++) {
      i = (int)(n * UNIF);
      xi = &data[i*p];

      //add_onecluster(props, mus, covs, xi, new_props, new_mus, new_covs, p, M);
      add_onecluster_props(props, new_props, M);
      add_onecluster_mean(xi, mus, new_mus, p, M);
      add_onecluster_cov(covs, new_covs, p, M);

      em_gmm(data, weights, new_mus, new_covs, new_props, &cov_type, &n, &p, &new_M, pth, &countf, logliks, pitmax, &silent);

      if (llik < logliks[countf]) {
        llik = logliks[countf];

        copy_prop(new_props, props, new_M);

        copy_mean(new_mus, mus, p, new_M);

        (*copy_cov)(new_covs, covs, p, new_M);
      }
    }

    M = new_M;

    if (debug)
      printf("M = %d: loglik = %.5f\n", M, llik);
  }

  *pM = M;

  free(P);
  free(prj_x); free(prj_mus); free(prj_sigmas);
  free(y);
  free(new_props); free(new_mus); free(new_covs);
  free(logliks);

  RANDOUT;
}

void create_projection(double *P, int p)
{
  int j, k;
  double sum;

  // create random projection
  sum = 0.0;
  for (j = 0; j < p; j++) {
    P[j] = rnorm(0, 1.0/p);
    sum += P[j] * P[j];
  }

  // normalization so that ||P|| = 1.
  sum = sqrt(sum);
  for (j = 0; j < p; j++)
    P[j] /= sum;
}

void projection_mean(double *P, double *mus, double *prj_mus, int p, int M)
{
  int j, m;
  double *mu;

  for (m = 0; m < M; m++) {
    mu = &mus[m*p];

    prj_mus[m] = 0.0;
    for (j = 0; j < p; j++)
      prj_mus[m] += P[j] * mu[j];
  }
}


void projection_fullcov(double *P, double *covs, double *prj_sigmas, int p, int M)
{
  int i, j, m;
  double *cov;

  for (m = 0; m < M; m++) {
    cov = &covs[m*p*p];

    prj_sigmas[m] = 0.0;
    for (i = 0; i < p; i++) {
      for (j = 0; j < p; j++)
        prj_sigmas[m] += P[i] * P[j] * cov[i*p + j];
    }
  }
}

void projection_diagcov(double *P, double *sigs, double *prj_sigmas, int p, int M)
{
  int j, m;
  double *sig;

  for (m = 0; m < M; m++) {
    sig = &sigs[m*p];

    prj_sigmas[m] = 0.0;
    for (j = 0; j < p; j++)
      prj_sigmas[m] += P[j] * P[j] * sig[j];
  }
}

void projection_x(double *P, double *data, double *prj_x, int n, int p)
{
  int i, j;

  for (i = 0; i < n; i++) {
    prj_x[i] = 0.0;
    for (j = 0; j < p; j++)
      prj_x[i] += P[j] * data[i*p + j];
  }

}

void add_onecluster_props(double *props, double *new_props, int M)
{
  int m;
  double sum;

  sum = 0.0;
  for (m = 0; m < M; m++)
    sum += props[m];
  sum += 1.0/M;

  for (m = 0; m < M; m++) {
    new_props[m] = props[m] / sum;
  }
  new_props[M] = (1.0/M) / sum;
}

void add_onecluster_mean(double *xi, double *mus, double *new_mus, int p, int M)
{
  int j, m;

  for (m = 0; m < M; m++) {
    for (j = 0; j < p; j++)
      new_mus[m*p + j] = mus[m*p + j];
  }
  for (j = 0; j < p; j++)
    new_mus[M*p + j] = xi[j];
}

void add_onecluster_fullcov(double *covs, double *new_covs, int p, int M)
{
  int i, j, m;
  double sum;

  for (i = 0; i < p; i++) {
    for (j = 0; j < p; j++) {
      sum = 0.0;
      for (m = 0; m < M; m++) {
        new_covs[m*p*p + i*p + j] = covs[m*p*p + i*p + j];
        sum += covs[m*p*p + i*p + j];
      }
      new_covs[M*p*p + i*p + j] = sum / M;
    }
  }
}

void add_onecluster_diagcov(double *sigs, double *new_sigs, int p, int M)
{
  int j, m;
  double sum;

  for (j = 0; j < p; j++) {
    sum = 0.0;
    for (m = 0; m < M; m++) {
      new_sigs[m*p + j] = sigs[m*p + j];
      sum += sigs[m*p + j];
    }
    new_sigs[M*p + j] = sum / M;
  }
}

double pmixnorm(double x, int M, double *pi, double *mu, double *sigma) {
  double cdf = 0.0;
  int m;

  for (m = 0; m < M; m++)
    cdf += pi[m] * pnorm(x, mu[m], sqrt(sigma[m]), 1, 0);

  return(cdf);
}
