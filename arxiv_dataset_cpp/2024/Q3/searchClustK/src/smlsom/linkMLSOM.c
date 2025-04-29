#include "base.h"
#include "link.h"
#include "searchGraph.h"
#include "mvnLLD.h"
#include "uLLD.h"
#include "linkMLSOM.h"

double calcr(int M) {
  link li = availNodes;
  int i = 0;
  double linkdist = 0.0;
  while((li = li->next) != NULL) {
    i++;
    linkdist += bfs_dist(li->node, M);
  }
  return (linkdist / i);
}

void ulinkMLSOM(double *data, double *moments, updfunc Update, distfunc lld,
                int *adjmatrix, double *alphas, double *radii, double *changes,
                int n, int p, int M, int K, int rlen, int chgbyns, double *consts)
{
  int m, i, j, k, l, t, nearest, niter, nind;
  double dm, dist, tmp, threshold;
  link li;
  link neighs = make_dummy();

  double *alpha = (double *)malloc(K*sizeof(double));

  niter = rlen * n;
  for (t = 0; t < niter; t++) {
    // pick a random data point
    i = (int)(n * UNIF);

    // find the nearest code 'near'
    nind = 0; dm = DOUBLE_XMAX; nearest = -1;
    // find 'winner' among available nodes
    li = availNodes;
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

    // update all moments within threshold of 'nearest'. Linear decrease
    // for both radius and learning parameter.
    threshold = radii[0] - (radii[0] - radii[1]) * (double)t/(double)niter;
    if (threshold < 1.0) threshold = 0.5;
    for (k = 0; k < K; k++) {
      //update the learning rate for kth moment
      alpha[k] = alphas[0 + 2*k] - (alphas[0 + 2*k] - alphas[1 + 2*k]) * (double)t/(double)niter;
    }

    l = (int)(t/chgbyns);

    // require `neighs` is empty
    bfs(nearest, (int)threshold, neighs, M);
    Update(&data[i*p], moments, alpha, neighs, p, K, M);
    //    for (j = 0; j < p; j++) {
    //      mmts[j*K] = data[i*p + j];
    //      for (k = 1; k < K; k++)
    //	mmts[k + j*K] = data[i*p + j] * mmts[k-1 + j*K];
    //    } // mmts : 1st-Kth moments of x_i
    //    while((m = getTopFrom(neighs)) != -1) {
    //      for(j = 0; j < p; j++) {
    //	for (k = 0; k < K; k++) {
    //    tmp = mmts[k + j*K] - moments[m*K+k + j*M*K];
    //    moments[m*K+k + j*M*K] += tmp * alpha[k];
    //	}
    //      }
    //    }

    // cumsum negative log-likelihood
    changes[l] += dm + consts[i];
  }
  free(neighs);
  //free(mmts);
  free(alpha);
}

void mvn_linkMLSOM(double *data, double *mu1, double *mu2, int *adjmatrix,
                   double *alphas, double *radii, double *changes,
                   int n, int p, int M, int rlen, int chgbyns)
{
  int m, i, j, k, l, t, nearest, niter, nind;
  double dm, dist, tmp, threshold, alpha;
  link li;
  link neighs = make_dummy();

  double cnst = 0.5*p * log(2*M_PI);

  niter = rlen * n;
  for (t = 0; t < niter; t++) {
    // pick a random data point
    i = (int)(n * UNIF);

    // find the nearest code 'near'
    nind = 0; dm = DOUBLE_XMAX; nearest = -1;
    // find 'winner' among available nodes
    li = availNodes;
    while((li = li->next) != NULL) {
      m = li->node;
      if (flgs[m] == -1) {
        continue;
      }
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

    // update all codes within threshold of 'nearest'. Linear decrease
    // for both radius and learning parameter.
    threshold = radii[0] - (radii[0] - radii[1]) * (double)t/(double)niter;
    if (threshold < 1.0) threshold = 0.5;
    alpha = alphas[0] - (alphas[0] - alphas[1]) * (double)t/(double)niter;

    l = (int)(t/chgbyns);

    // require `neighs` is empty
    bfs(nearest, (int)threshold, neighs, M);
    while((m = getTopFrom(neighs)) != -1) {
      // update mu1
      for (j = 0; j < p; j++) {
        tmp = data[i*p + j] - mu1[m*p + j];
        mu1[m*p + j] += tmp * alpha;
      }
      // update mu2
      for (k = 0; k < p; k++) {
        for (j = 0; j < p; j++) {
          tmp = data[i*p + j] * data[i*p + k] - mu2[m*p*p + j + k*p];
          mu2[m*p*p + j + k*p] += tmp * alpha;
        }
      }
      // update L
      flgs[m] = updateA(mu1, mu2, m, p);
    }

    // cumsum negative log-likelihood
    changes[l] += dm + cnst;
  }
  free(neighs);
}
