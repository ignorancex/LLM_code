#include "utils.h"
#include "dip.h"
#include "diptest.h"

#define B 1000

double testCluster_unimodality(double *D, int nk, int *ck, double a, double vthd, int debug)
// computing dip-dist and p-value. Then return eqn.(3) in Kalogeratos and Likas (2012)
// Input:
//  D : global distance matrix
//  nk : the nummber of samples in cluster k
//  ck : vector of indices of samples which belong to cluster k. THE VECTOR SHOULD BE SORTED.
//  a : significant level
//  vthd : threshold ratio of split viewers (see Kalogeratos and Likas (2012))
// Output
//  eqn.(3) in Kalogeratos and Likas (2012)
// Ref.:
//  Kalogeratos, Argyris, and Aristidis Likas. "Dip-means: an incremental clustering method for estimating the number of clusters." Advances in neural information processing systems. 2012.
{
  int i,j,r,t;
  int nd = nk-1;

  if (nk < 5)
    return 0.0;

  double *d = (double *)malloc(sizeof(double ) * nd);
  double *us = (double *)malloc(sizeof(double) * (nd));

  double pval;

  double dip, dips[B];
  int low_hi[4];
  int ifault;
  int *gcm = (int *)malloc(sizeof(int) * (nd));
  int *lcm = (int *)malloc(sizeof(int) * (nd));
  int *mn = (int *)malloc(sizeof(int) * (nd));
  int *mj = (int *)malloc(sizeof(int) * (nd));
  const int min_is_0 = 0;
  const int debug_diptst = 0;

  int sv_num = 0; // store the number of split viewers
  double score = 0.0;

  // calculate baseline for all viewers
  for (r = 0; r < B; r++) {
    for (t = 0; t < nd; t++)
      us[t] = UNIF;
    qsort(us, nd, sizeof(double), asc_double);

    diptst(us, &nd,  &dips[r], low_hi, &ifault, gcm, lcm, mn, mj, &min_is_0, &debug_diptst);
  }

  // for each viewer
  for (i = 0; i < nk; i++) {
    t = 0;
    for (j = 0; j < i; j++) {
      d[t] = D[(int)(0.5 * ck[i] * (ck[i]-1)) + ck[j]];
      t++;
    }
    for (j = i+1; j < nk; j++) {
      d[t] = D[(int)(0.5 * ck[j] * (ck[j]-1)) + ck[i]];
      t++;
    } //d is distances from viewer ck[i]

    // sort d for dip test
    qsort(d, nd, sizeof(double), asc_double);

    // calculate dip-static
    diptst(d, &nd, &dip, low_hi, &ifault, gcm, lcm, mn, mj, &min_is_0, &debug_diptst);

    // calculate score based on the pvalue which is estimated by a resampling method.
    // estimate p-value for the dip-static based on simulation
    pval = 0.0;
    for (r = 0; r < B; r++)
      if (dip < dips[r]) pval += 1.0;
    pval /= B;

    if (pval <= a) { // if the viewer is split viewer
      sv_num++;
      score += dip;
    }
  }

  free(d);
  free(gcm); free(lcm); free(mn); free(mj);
  free(us);

  if (debug) printf("split viewer ratio = %.4f\n", sv_num / (double)nk);

  if (sv_num / (double)nk < vthd)
    return 0;
  return (score / sv_num);
}

double calculate_score(double *D, int nk, int *ck, int nn_len, int *nn, double *qdips, double vthd, int debug) {
  int i,j,r,t;
  int nd = nk-1;
  double *d = (double *)malloc(sizeof(double ) * nd);

  double dip;
  int low_hi[4];
  int ifault;
  int *gcm = (int *)malloc(sizeof(int) * (nd));
  int *lcm = (int *)malloc(sizeof(int) * (nd));
  int *mn = (int *)malloc(sizeof(int) * (nd));
  int *mj = (int *)malloc(sizeof(int) * (nd));
  const int min_is_0 = 0;
  const int debug_diptst = 0;

  int sv_num = 0; // store the number of split viewers
  double score = 0.0;

  // find interval and calculate interpolation
  double basedip = 0.0;
  for (i = 0; nn[i] < nd; i++);
  if (i < nn_len) {
    basedip = (qdips[i] - qdips[i-1]) * nd + nn[i] * qdips[i-1] - nn[i-1] * qdips[i];
    basedip = basedip / (double)(nn[i]-nn[i-1]);
  } else {
    basedip = qdips[nn_len-1];
  }

  if (debug) {
    printf(" nd = %d: nn[i] = %d\n", nd, nn[i]);
    printf(" basedip = %.6f\n", basedip);
  }

  // for each viewer
  for (i = 0; i < nk; i++) {
    t = 0;
    for (j = 0; j < i; j++) {
      d[t] = D[(int)(0.5 * ck[i] * (ck[i]-1)) + ck[j]];
      t++;
    }
    for (j = i+1; j < nk; j++) {
      d[t] = D[(int)(0.5 * ck[j] * (ck[j]-1)) + ck[i]];
      t++;
    } //d is distances from viewer ck[i]

    // sort d for dip test
    qsort(d, nd, sizeof(double), asc_double);

    // calculate dip-static
    diptst(d, &nd, &dip, low_hi, &ifault, gcm, lcm, mn, mj, &min_is_0, &debug_diptst);

    if (dip > basedip) {
      sv_num++;
      score += dip;
    }
  }

  free(d);
  free(gcm); free(lcm); free(mn); free(mj);

  if (sv_num / (double)nk < vthd)
    return 0;
  return (score / sv_num);
}
