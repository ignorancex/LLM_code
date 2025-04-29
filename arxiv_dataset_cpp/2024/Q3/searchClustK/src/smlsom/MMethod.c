#include "link.h"
#include "mvnLLD.h"
#include "uLLD.h"
#include "MMethod.h"

void ubestMatchs(double *data, double *moments, distfunc lld, int n, int p, int M, int K,
                 int *cls, double *lliks, int *counts, double *consts, link avails)
{
  int i,j,m,nind,nearest;
  double dist,dm;
  link li;

  // set up 0
  for (m = 0; m < M; m++) counts[m] = 0;

  for (i = 0; i < n; i++) {
    // find the nearest code 'near'
    nind = 0; dm = DOUBLE_XMAX; nearest = -1;
    li = avails;
    while((li = li->next) != NULL) {
      m = li->node;
      dist = 0.0;
      for (j = 0; j < p; j++)
        dist += lld(data[i*p + j], &moments[m*K + j*M*K]);
      if (dist <= dm * (1 + EPS)) {
        if (dist < dm * (1 - EPS)) {
          nind = 0;
          nearest = m;
        } else {
          if(++nind * UNIF < 1.0) nearest = m;
        }
        dm = dist;
      }
    }
    if (nearest < 0)
      error("No nearest neighbour found...");
    counts[nearest]++;
    lliks[i] = -dm - consts[i];
    cls[i] = nearest;
  }
}

void ubatch(double *data, double *moments, distfunc lld, int n, int p, int M, int K,
            int *cls, double *lliks, int *counts, double *consts, link avails, char **dtype)
{
  // require cls and counts not empty
  // batch for nodes in `avails`

  int i,j,k,m;
  link li;
  double tmp, dist;

  //init
  li = avails;
  while ((li = li->next) != NULL) {
    m = li->node;
    for (j = 0; j < p; j++)
      for (k = 0; k < K; k++)
        moments[m*K + j*M*K + k] = 0.0;
  }

  //batch
  if (!strcmp("multinom",*dtype)) {
    double sum;
    for (i = 0; i < n; i++) {
      m = cls[i];
      sum = 0.0;
      for (j = 0; j < p; j++)
        sum += data[i*p + j];
      if (sum > 0) {
        for (j = 0; j < p; j++)
          moments[m + j*M] += data[i*p + j] / sum;
      }
    }
  }
  else {
    for (i = 0; i < n; i++) {
      m = cls[i];
      for (j = 0; j < p; j++) {
        tmp = 1.0;
        for (k = 0; k < K; k++) {
          tmp *= data[i*p + j];
          moments[m*K + j*M*K +k] += tmp;
        }
      }
    }
  }
  li = avails;
  while ((li = li->next) != NULL) {
    m = li->node;
    for (j = 0; j < p; j++) {
      for (k = 0; k < K; k++) {
        moments[m*K + j*M*K + k] /= (double)counts[m];
      }
    }
  }
  //log likelihood
  for (i = 0; i < n; i++)  {
    m = cls[i];
    dist = 0.0;
    for (j = 0; j < p; j++)
      dist += lld(data[i*p + j], &moments[m*K + j*M*K]);
    lliks[i] = -dist - consts[i];
  }
}

//df : |M|*K*p
double uMDL(double *lliks, int n, int p, int K, int nnodes, char **dtype) {
  int i;
  double mdl = 0.0;
  double df;
  if (!strcmp("multinom",*dtype))
    df = nnodes*K*(p-1);
  else
    df = nnodes*K*p;
  for (i = 0; i < n; i++)
    mdl += -lliks[i];
  mdl += df*0.5 * log(n)+ n * log(nnodes);
  return mdl;
}

void mvn_bestMatchs(double *data, double *mu1, double *mu2, int n, int p, int M, int *cls, double *lliks, int *counts, link avails) {
  int i,m,nind,nearest;
  double dist,dm;
  link li;

  double cnst = 0.5*p * log(2*M_PI);

  // set up 0
  for (m = 0; m < M; m++) counts[m] = 0;

  for (i = 0; i < n; i++) {
    // find the nearest code 'near'
    nind = 0; dm = DOUBLE_XMAX; nearest = -1;
    li = avails;
    while((li = li->next) != NULL) {
      m = li->node;
      if (flgs[m] == -1)
        continue;
      dist = lld_mvnorm(&data[i*p], &mu1[m*p], m, p);
      if (dist <= dm * (1 + EPS)) {
        if (dist < dm * (1 - EPS)) {
          nind = 0;
          nearest = m;
        } else {
          if(++nind * UNIF < 1.0) nearest = m;
        }
        dm = dist;
      }
    }
    if (nearest < 0)
      error("No nearest neighbour found...");
    counts[nearest]++;
    lliks[i] = -dm - cnst;
    cls[i] = nearest;
  }
}

void mvn_batch(double *data, double *mu1, double *mu2, int n, int p, int *cls, double *lliks, int *counts, link avails) {
  // require cls and counts not empty
  // batch for nodes in `avails`

  int i,j,k,m;
  link li;
  double cnst = 0.5*p * log(2*M_PI);

  //init
  li = avails;
  while ((li = li->next) != NULL) {
    m = li->node;
    for (k = 0; k < p; k++) {
      mu1[m*p + k] = 0.0;
      for (j = 0; j < p; j++)
        mu2[m*p*p + j + k*p] = 0.0;
    }
  }

  //batch
  for (i = 0; i < n; i++) {
    m = cls[i];
    for (j = 0; j < p; j++)
      mu1[m*p + j] += data[i*p + j];
    for (k = 0; k < p; k++) {
      for (j = 0; j < p; j++)
        mu2[m*p*p + j + k*p] += data[i*p + j] * data[i*p + k];
    }
  }
  li = avails;
  while ((li = li->next) != NULL) {
    m = li->node;
    for (j = 0; j < p; j++)
      mu1[m*p + j] = mu1[m*p + j] / (double)counts[m];
    for (k = 0; k < p; k++) {
      for (j = 0; j < p; j++)
        mu2[m*p*p + j + k*p] = mu2[m*p*p + j + k*p] / (double)counts[m];
    }
    //update A_m m in avails
    flgs[m] = updateA(mu1, mu2, m, p);
  }
  //log likelihood
  for (i = 0; i < n; i++)  {
    m = cls[i];
    lliks[i] = -lld_mvnorm(&data[i*p], &mu1[m*p], m, p) - cnst;
  }
}

//df : |P|p(p+3) / 2
double mvnMDL(double *lliks, int n, int p, int nnodes) {
  int i;
  double mdl = 0.0;
  double df = nnodes*p*(p+3)*0.5;
  for (i = 0; i < n; i++)
    mdl += -lliks[i];
  mdl += df*0.5 * log(n)+ n * log(nnodes);
  return mdl;
}
