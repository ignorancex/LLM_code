#include "em_gmm.h"

params_link init_list_fullcov(double *mus, double *covs, int M, int p);
params_link init_list_diagcov(double *mus, double *sigs, int M, int p);
void free_list_fullcov(params_link params_fc);
void free_list_diagcov(params_link params_dc);

void estep(params_link params, double (*mvnorm)(double *, params_link, int),
           double *data, double *weights, double *props, int n, int p, int M);
double mixloglik(params_link params, double (*mvnorm)(double *, params_link, int),
                 double *data, double *props, int n, int p, int M);
double mstep_fullcov(params_link pl, double *data, double *weights, int n, int p);
double mstep_diagcov(params_link pl, double *data, double *weights, int n, int p);
double mvnorm_fullcov(double *xi, params_link pl, int p);
double mvnorm_diagcov(double *xi, params_link pl, int p);
void decomp_cov(double *cov, double *diagL, double *L, int p);
int choldc(double *a, double *p, int n);

void em_gmm(double *data, double *weights, double *mus, double *covs, double *props, int *pcov_type,
            int *pn, int *pp, int *pM, double *pth, int *pcountf, double *logliks, int *pitmax, int *psilent)
{
  int i, j, k, l, m, countf;
  int cov_type = *pcov_type;
  int n = *pn, p = *pp, M = *pM, itmax = *pitmax, silent = *psilent;
  double *mu, *cov;
  double sum, dl, n_m;
  double loglik;
  double th = *pth;

  params_link pl, params;

  params_link (*init_list)(double *, double *, int, int);
  void (*free_list)(params_link);
  double (*mvnorm)(double *, params_link, int);
  double (*mstep)(params_link, double *, double *, int, int);

  if (cov_type == 0) {

    init_list = init_list_fullcov;
    free_list = free_list_fullcov;
    mvnorm = mvnorm_fullcov;
    mstep = mstep_fullcov;

  } else if (cov_type == 1) {

    init_list = init_list_diagcov;
    free_list = free_list_diagcov;
    mvnorm = mvnorm_diagcov;
    mstep = mstep_diagcov;

  }

  // make list
  params = init_list(mus, covs, M, p);

  countf = 0;

  // initial loglikelihood
  logliks[countf] = mixloglik(params, mvnorm, data, props, n, p, M);
  if (silent == 0)
    printf("%3d : loglik = %.4f\n", countf, logliks[countf]);

  do {
    countf++;

    // Estep
    estep(params, mvnorm, data, weights, props, n, p, M);

    // Mstep
    pl = params;
    for (m = 0; m < M; m++) {
      n_m = mstep(pl, data, &weights[m*n], n, p);
      props[m] = n_m / n;
      pl = pl->next;
    }

    // calc current loglikelihood
    logliks[countf] = mixloglik(params, mvnorm, data, props, n, p, M);

    if (silent == 0)
      printf("%3d : loglik = %.4f\n", countf, logliks[countf]);

    dl = logliks[countf] - logliks[countf-1];
    dl = fabs(dl / logliks[countf-1]);

  } while(dl > th && countf < (itmax-1));

  *pcountf = countf;

  free_list(params);
}

void estep(params_link params, double (*mvnorm)(double *, params_link, int),
           double *data, double *weights, double *props, int n, int p, int M)
{
  int i, m;
  double sum;
  params_link pl;

  pl = params;
  for (m = 0; m < M; m++) {
    if (pl == NULL)
      error("params_link pl is empty!");
    for (i = 0; i < n; i++)
      weights[i + m*n] = props[m] * exp(mvnorm(&data[i*p], pl, p));
    pl = pl->next;
  }

  for (i = 0; i < n; i++) {
    sum = EPS;
    for (m = 0; m < M; m++)
      sum += weights[i + m*n];
    for (m = 0; m < M; m++)
      weights[i + m*n] = weights[i + m*n] / sum;
  }
}

double mixloglik(params_link params, double (*mvnorm)(double *, params_link, int),
                 double *data, double *props, int n, int p, int M)
// return sum of multivariate normal loglikehood
{
  int i, m;
  double sum, loglik = 0.0;
  params_link pl;

  for (i = 0; i < n; i++) {
    pl = params;
    sum=0.0;
    for (m = 0; m < M; m++) {
      if (pl == NULL)
        error("params_link pl is empty!");
      sum +=  props[m] * exp(mvnorm(&data[i*p], pl, p));
      pl = pl->next;
    }
    if (sum > EPS)
      loglik += log(sum);
    else
      loglik += log(EPS);
  }

  return loglik;
}


double mstep_fullcov(params_link pl, double *data, double *weights, int n, int p)
// return estimated sample size
{
  int i,j,l;
  double sum = EPS, tmp;

  int comp = pl->ind;
  double *mu = pl->mu;
  double *cov = pl->cov;
  double *chol = pl->chol;
  double *chol_diag = pl->chol_diag;

  // estimate sample size
  for (i = 0; i < n; i++)
    sum += weights[i];

  // estimate mean and covariance
  // preparation
  for (j = 0; j < p; j++) {
    mu[j] = 0.0;
    for (l = 0; l < p; l++)
      cov[j*p + l] = 0.0;
  }

  // estimation mean vector
  for (i = 0; i < n; i++) {
    for (j = 0; j < p; j++) {
      tmp = weights[i] * data[i*p + j];
      mu[j] += tmp;
    }
  }
  // normalize
  for (j = 0; j < p; j++)
    mu[j] /= sum;

  // estimation covariance
  for (i = 0; i < n; i++) {
    for (j = 0; j < p; j++) {
      tmp = weights[i] * (data[i*p + j] - mu[j]);
      for (l = 0; l < p; l++) {
        cov[j*p + l] += tmp * (data[i*p + l] - mu[l]);
      }
    }
  }
  // normalize
  for (j = 0; j < p; j++) {
    for (l = 0; l < p; l++)
      cov[j*p + l] /= sum;
  }

  decomp_cov(cov, chol_diag, chol, p);

  return sum;
}

double mstep_diagcov(params_link pl, double *data, double *weights, int n, int p)
// return estimated sample size
{
  int i,j,l;
  double sum = EPS, tmp;

  int comp = pl->ind;
  double *mu = pl->mu;
  double *sig = pl->sig;

  // estimate sample size
  for (i = 0; i < n; i++)
    sum += weights[i];

  // estimate mean and variances
  // preparation
  for (j = 0; j < p; j++) {
    mu[j] = 0.0;
    sig[j] = 0.0;
  }

  // estimation mean vector
  for (i = 0; i < n; i++) {
    for (j = 0; j < p; j++) {
      tmp = weights[i] * data[i*p + j];
      mu[j] += tmp;
    }
  }
  // normalize
  for (j = 0; j < p; j++)
    mu[j] /= sum;

  // estimation variances
  for (i = 0; i < n; i++) {
    for (j = 0; j < p; j++) {
      tmp = (data[i*p + j] - mu[j]);
      sig[j] += weights[i] * tmp * tmp;
    }
  }
  // normalize
  for (j = 0; j < p; j++)
    sig[j] /= sum;

  return sum;
}

double mvnorm_fullcov(double *xi, params_link pl, int p)
{
  int i,j;
  double tmp;
  double lexp = 0.0;
  double det = 1.0;

  double *mu = pl->mu;
  double *L = pl->chol;
  double *diagL = pl->chol_diag;

  // calculate value of loglikelihood
  // |A| = |LL^t| = |L|^2
  // |L| = prod(diag(L))
  for (i = 0; i < p; i++) {
    tmp = 0.0;
    for (j = 0; j <= i; j++)
      tmp += L[i*p + j] * (xi[j] - mu[j]); //L^-1 (x-mu)
    lexp += tmp * tmp;
    det *= diagL[i];
  }

  return (-0.5*p*log(2*M_PI) - log(det) - 0.5*lexp);
}

double mvnorm_diagcov(double *xi, params_link pl, int p)
{
  int j;
  double sum;
  double *mu = pl->mu;
  double *sig = pl->sig;

  sum = 0.0;
  for (j = 0; j < p; j++) {
    sum += -0.5 * log(sig[j]) - 0.5 * (xi[j] - mu[j]) * (xi[j] - mu[j]) / sig[j];
  }
  sum += -0.5*p*log(2*M_PI);

  return(sum);
}

void decomp_cov(double *cov, double *diagL, double *L, int p)
// cov : p-order covariance matrix
// index : the number of component
// p : dimension
{
  int i,j,k;
  int flg;
  double sum;

  for (i = 0; i < p; i++) {
    for (j = 0; j < p; j++)
      L[i*p + j] = cov[i*p + j];
  }

  // cholesky decomposition of A, for calculating the likelihood
  // A = LL^t where L is a lower triangle matrix
  flg = choldc(L, diagL, p);
  if (flg == -1)
    error("covariance become singular");

  // calculate inverse of L
  // A was replaced in L
  for (i = 0; i < p; i++) {
    L[i*p + i] = 1.0 / diagL[i];
    for (j = i+1; j < p; j++) {
      sum = 0.0;
      for (k = i; k < j; k++)
        sum -= L[j*p + k] * L[k*p + i];
      L[j*p + i] = sum / diagL[j];
    }
  }
}

int choldc(double *a, double *p, int n) {
  int i,j,k;
  double sum;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      for (sum = a[i*n + j], k = i-1; k >= 0; k--) sum -= a[i*n + k] * a[j*n + k];
      if (i == j) {
        if (sum <= 0) {
          return -1;
        }
        p[i] = sqrt(sum);
      } else {
        a[j*n + i] = sum / p[i];
      }
    }
  }

  return 0;
}

params_link init_list_fullcov(double *mus, double *covs, int M, int p)
{
  int m;
  params_link pl, params_fc;
  double *chol, *chol_diag;

  params_fc = NULL;
  for (m = M-1; m >= 0; m--) {
    pl = (params_link)malloc(sizeof(params_list));
    chol = (double *)malloc(sizeof(double) * p * p);
    chol_diag = (double *)malloc(sizeof(double) * p);

    pl->ind = m;
    pl->mu = &mus[m*p];
    pl->cov = &covs[m*p*p];
    pl->sig = NULL;

    decomp_cov(&covs[m*p*p], chol_diag, chol, p);
    pl->chol = chol;
    pl->chol_diag = chol_diag;

    pl->next = params_fc;

    params_fc = pl;
  }

  return params_fc;
}

params_link init_list_diagcov(double *mus, double *sigs, int M, int p)
{
  int m;
  params_link pl, params_dc;
  double *chol, *chol_diag;

  params_dc = NULL;
  for (m = M-1; m >= 0; m--) {
    pl = (params_link)malloc(sizeof(params_list));
    pl->chol = NULL;
    pl->chol_diag = NULL;

    pl->ind = m;
    pl->mu = &mus[m*p];
    pl->cov = NULL;
    pl->sig = &sigs[m*p];

    pl->next = params_dc;

    params_dc = pl;
  }

  return params_dc;
}

void free_list_fullcov(params_link params_fc)
{
  params_link pl = params_fc->next;
  params_link bl = params_fc;
  while (pl != NULL) {
    free(bl->chol);
    free(bl->chol_diag);
    free(bl);
    bl = pl;
    pl = pl->next;
  }
  free(bl->chol);
  free(bl->chol_diag);
  free(bl);
}

void free_list_diagcov(params_link params_dc)
{
  params_link pl = params_dc->next;
  params_link bl = params_dc;
  while (pl != NULL) {
    free(bl);
    bl = pl;
    pl = pl->next;
  }
  free(bl);
}
