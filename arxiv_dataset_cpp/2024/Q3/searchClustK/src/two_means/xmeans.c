/*
  Implementation of `Pelleg, Dan, and Andrew W. Moore. "X-means: Extending k-means with efficient estimation of the number of clusters." The 17th International Conference on Machine Learning (2000): 727-734`
 */

#include "xmeans.h"
#include "twomeans.h"
#include "utils.h"

list *cluster_array;

boolean testCluster_bic(double *X, int p, int *ck, int nk,
                        double *parent_center, double *child1_center, double *child2_center,
                        list child1_array, list child2_array, double *bic_parent, double *bic_children);

void xmeans(double *X, int *pn, int *pp, double *M, int *pK,
            int *C, int *pKmax, int *pdebug)
{
  /*================= arguments from R ==================*/
  int n = *pn, p = *pp, K = *pK, Kmax = *pKmax;
  int debug = *pdebug;
  /*=====================================================*/
  int i, j, k;
  boolean test;
  double bic_p, bic_c;

  // for two means
  int nk;
  list cluster;
  double *vec1 = (double *)malloc(sizeof(double) * p); // left centroid
  double *vec2 = (double *)malloc(sizeof(double) * p); // right centroid
  int *ck = (int *)malloc(sizeof(int) * n); // sample indices
  list l, ic1, ic2;

  // initialize cluster index stack
  list stack;
  stack = make_dummy();
  for (k = K-1; k >= 0; k--)
    pushTo(stack, k);

  // cluster_array[k] is the list of indices of samples belong to cluster k.
  init_cluster_array(C, n, Kmax);

  RANDIN;

  k = getTopFrom(stack); // satisfy k is not -1
  do {
    // kth cluster is investigated whether splittd or not.
    if (debug) {
      printf("pick %d\n", k);
      printf("n = %d\n", getDim(cluster_array[k]));
    }

    nk = list2vec(ck, cluster_array[k]); // nk is the number of samples in cluster k.

    i = ck[(int)(nk * UNIF)];
    for (j = 0; j < p; j++) {
      vec2[j] = M[k*p + j] - (X[i*p + j] - M[k*p + j]);
      vec1[j] = X[i*p + j];
    }

    //compute new assignments
    ic1 = make_dummy();
    ic2 = make_dummy();
    two_means(X, p, ck, nk, vec1, vec2, ic1, ic2);

    // BIC comparison
    test = testCluster_bic(X, p, ck, nk, &M[k*p], vec1, vec2, ic1, ic2, &bic_p, &bic_c);

    if(debug) printf(" Parent's BIC = %.3f\n Children's BIC = %.3f\n", bic_p, bic_c);

    if (test) {
      if (debug) printf("   splitted\n");

      free_que(cluster_array[k]);
      cluster_array[k] = NULL;
      pushTo(stack, K); pushTo(stack, k);

      for (j = 0; j < p; j++) {
        M[k*p + j] = vec1[j];
        M[K*p + j] = vec2[j];
      }

      cluster_array[k] = ic1;
      cluster_array[K] = ic2;

      K = K+1;
    } else {
      if (debug) printf("   not splitted\n");
      if (ic1 != NULL) free_que(ic1);
      if (ic2 != NULL) free_que(ic2);
    }

    if (debug) print_que(stack);

    if (K >= Kmax)
      break;
  } while((k = getTopFrom(stack)) != -1);

  RANDOUT;

  if (debug) printf("K = %d\n", K);

  // convert list to array
  for (k = 0; k < K; k++) {
    l = cluster_array[k];
    if (l != NULL) {
      while((l = l->next) != NULL)
        C[l->ind] = k;
    }
  }

  *pK = K;

  free_cluster_array(Kmax);
  free(vec1); free(vec2);
  free_que(stack); free(ck);
}

double estimate_sigma(double *X, int p, int *c, int c_len, double *mu, int df) {
  int i, j, t;
  double sigma = 0.0, dist, tmp;

  for (t = 0; t < c_len; t++) {
    i = c[t];
    dist = 0.0;
    for (j = 0; j < p; j++) {
      tmp = X[i*p + j] - mu[j];
      dist += tmp * tmp;
    }
    sigma += dist / df;
  }

  return sigma;
}

double loglik(int n, int p, int k, double sigma)
{
  double l;

  l = - 0.5 * n * log(2*M_PI) - 0.5 * (n * p) * log(sigma) - 0.5 * (n-k);

  return l;
}

boolean testCluster_bic(double *X, int p, int *ck, int nk, double *mk, double *vec1, double *vec2, list ic1, list ic2, double *bic_p, double *bic_c)
// bic_p, bic_c: BIC of a parent, children respectively
{
  if (nk <= 5)
    return false;

  int n1, n2;
  double p1, p2;
  double sigma_p = 0.0, sigma_c = 0.0;
  double l_p, l_c; // loglikelihood of parent, children respectively

  int *c1 = (int *)malloc(sizeof(int) * nk);
  int *c2 = (int *)malloc(sizeof(int) * nk);

  // calculation of parent's BIC
  //  loglikelihood
  sigma_p = estimate_sigma(X, p, ck, nk, mk, nk-1);
  l_p = loglik(nk, p, 1, sigma_p);

  //  the number of parameters is p + 1
  //  p: one center, 1: sigma
  *bic_p = l_p - 0.5 * (p + 1) * log(nk);

  // calculation of two children's BIC
  n1 = list2vec(c1, ic1); // the number of samples belong to the first child cluster
  n2 = list2vec(c2, ic2); // the number of samples belong to the second child cluster

  //  mixising proportion
  p1 = n1 / (double)nk;
  p2 = n2 / (double)nk;

  //  loglikelihood
  sigma_c = estimate_sigma(X, p, c1, n1, vec1, nk-2) + estimate_sigma(X, p, c2, n2, vec2, nk-2);
  l_c = n1 * log(p1) + loglik(n1, p, 2, sigma_c);
  l_c += n2 * log(p2) + loglik(n2, p, 2, sigma_c);

  //  the number of parameters is 2p + 2
  //  2p: two centers, 1: sigma, 1: p1, p2 (p1+p2=1)
  *bic_c = l_c - 0.5 * (2*p + 2) * log(nk);

  free(c1); free(c2);

  if (*bic_p > *bic_c)
    return false;
  else
    return true;
}
