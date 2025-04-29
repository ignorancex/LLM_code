#include "twomeans.h"

void two_means(double *X, int p, int *ck, int nk, double *cnt1, double *cnt2, list ic1, list ic2)
{
  double tmp, s1, s2;
  int t, i, j, n1, n2;
  int it = 0;
  double chngs[2];

  chngs[1] = 0.0;
  //for (it = 0; it < MAXIT; it++) {
  while(1) {
    it++;

    chngs[0] = chngs[1];

    for (t = 0; t < nk; t++) {
      i = ck[t];

      s1 = 0.0; s2 = 0.0;
      chngs[1] = 0.0;

      // calculate distance
      for (j = 0; j < p; j++) {
        tmp = X[i*p + j] - cnt1[j];
        s1 += tmp * tmp;
        tmp = X[i*p + j] - cnt2[j];
        s2 += tmp * tmp;
      }

      if (s1 < s2) {
        pushTo(ic1, i);
        chngs[1] += s1;
      } else {
        pushTo(ic2, i);
        chngs[1] += s2;
      }
    }

    if (it == MAXIT) break;
    if (fabs(chngs[0] - chngs[1])/chngs[0] < EPS) break;

    for (j = 0; j < p; j++) {
      cnt1[j] = 0.0;
      cnt2[j] = 0.0;
    }

    // averaging cluster 2
    n1 = getDim(ic1);
    while((i = getTopFrom(ic1)) != -1) {
      for (j = 0; j < p; j++)
        cnt1[j] += X[i*p + j];
    }
    for (j = 0; j < p; j++)
      cnt1[j] /= n1;

    // averaging cluster 2
    n2 = getDim(ic2);
    while((i = getTopFrom(ic2)) != -1) {
      for (j = 0; j < p; j++)
        cnt2[j] += X[i*p + j];
    }
    for (j = 0; j < p; j++)
      cnt2[j] /= n2;
  }
}

void init_cluster_array(int *C, int n, int Kmax)
{
  int i, k;

  cluster_array = (list *)malloc(sizeof(list) * Kmax);
  for (k = 0; k < Kmax; k++)
    cluster_array[k] = make_dummy();
  for (i = 0; i < n; i++)
    pushTo(cluster_array[ C[i]-1 ], i);
}

void free_cluster_array(int Kmax)
{
  int k;

  for (k = 0; k < Kmax; k++) {
    if (cluster_array[k] != NULL)
      free_que(cluster_array[k]);
  }
  free(cluster_array);
}
