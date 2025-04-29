#include "link.h"
#include "update.h"

updfunc matchUpdate(char **dtype) {
  if (!strcmp("pois",*dtype))
    return Update_mom1;
  else if (!strcmp("geom",*dtype))
    return Update_mom1;
  else if (!strcmp("nbin",*dtype))
    return Update_mom2;
  else if (!strcmp("norm",*dtype))
    return Update_mom2;
  else if (!strcmp("multinom",*dtype))
    return Update_multinom;
  else if (!strcmp("binary",*dtype))
    return Update_mom1;
  else {
    error("distance type '%s' not exist\n", *dtype);
  }
}

void Update_mom1(double *xi, double *moments, double *alpha, link neighs, int p, int K, int M)
{
  int j,m;
  double tmp;
  while((m = getTopFrom(neighs)) != -1) {
    for(j = 0; j < p; j++) {
      tmp = xi[j] - moments[m + j*M];
      moments[m + j*M] += tmp * alpha[0];
    }
  }
}

void Update_mom2(double *xi, double *moments, double *alpha, link neighs, int p, int K, int M)
{
  int j,m;
  double tmp, mmk;
  while((m = getTopFrom(neighs)) != -1) {
    for(j = 0; j < p; j++) {
      mmk = 1.0;
      mmk *= xi[j];
      tmp = mmk - moments[m*K + j*M*K];
      moments[m*K + j*M*K] += tmp * alpha[0];
      mmk *= xi[j];
      tmp = mmk - moments[m*K + 1 + j*M*K];
      moments[m*K + 1 + j*M*K] += tmp * alpha[1];
    }
  }
}

void Update_momK(double *xi, double *moments, double *alpha, link neighs, int p, int K, int M)
{
  int j,k,m;
  double tmp, mmk;
  while((m = getTopFrom(neighs)) != -1) {
    for(j = 0; j < p; j++) {
      mmk = 1.0;
      for (k = 0; k < K; k++) {
        mmk *= xi[j];
        tmp = mmk - moments[m*K + k + j*M*K];
        moments[m*K + k + j*M*K] += tmp * alpha[k];
      }
    }
  }
}

void Update_multinom(double *xi, double *theta, double *alpha, link neighs, int p, int K, int M)
{
  int j,m;
  double tmp;
  double sum = 0.0;

  for (j = 0; j < p; j++) {
    // Laplace smoothing
    sum += xi[j];
  }
  if (sum == 0.0)
    return;
  while((m = getTopFrom(neighs)) != -1) {
    for(j = 0; j < p; j++) {
      tmp = (xi[j]) / sum - theta[m + j*M];
      theta[m + j*M] += tmp * alpha[0];
    }
  }
}
