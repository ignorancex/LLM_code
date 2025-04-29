/*
  Implementation of `Kalogeratos, Argyris, and Aristidis Likas. "Dip-means: an incremental clustering method for estimating the number of clusters." Advances in neural information processing systems 25 (2012).`

  dipmeans(X,M,C)
    # X is input data matrix:
    #   X = {x_1, x_2, ..., x_n}, where x_i is p-dimentional real value vector
    # C = {c_1, c_2,.., c_K} where c_k is the set of sample index that belongs to cluster k
    # M is centroids:
    #   M = {m_1, m_2, ..., m_K}, where m_k is the centroid of cluster k

    {M,C} <- kmeans(X,M)
    D = [ d(x_i,x_j) ]_ij     # D is distances matrix where d is a distance function
    REPEAT
       # to do the Hartigan's dip test
       FOR k = 0 to K
         D_k = {D(i,j) | i,j in c_k}
         Hartigan's dip test on D_k and compute its dip's score
       ENDFOR
       maxscore = maximum value among K dip's scores
       k = the index of the cluster which has maxscore
       IF maxscore > 0.0
         X_k = {x_i | i in c_k}
         split X_k into two new clusters k1,k2
         ({m_k1,m_k2}, {c_k1, c_k2}) = Kmeans(X_k,{m_k1, m_k2})
       ENDIF
    UNTIL the number of clusters is not changed
 */

#include "base.h"
#include "time.h"
#include "utils.h"
#include "dip.h"
#include "diptest.h"
#include "twomeans.h"

list *cluster_array;

void dipmeans_sim(double *X, int *pn, int *pp, double *M, int *pK, int *C,
                  double *pa, double *pvthd, int *pKmax, int *pdebug)
{
  /*================= arguments from R ==================*/
  int n = *pn, p = *pp, K = *pK, Kmax = *pKmax;
  double a = *pa, vthd = *pvthd;
  int debug = *pdebug;
  /*=====================================================*/
  int i, j, k, nk, tg;

  // for two means
  list cluster;
  double *vec1 = (double *)malloc(sizeof(double) * p);
  double *vec2 = (double *)malloc(sizeof(double) * p);
  int *ck = (int *)malloc(sizeof(int) * n);

  // for dip test
  double maxscore;
  double *D = (double *)malloc(sizeof(double) * (int)(0.5 * n * (n-1))); // distances between samples, lower triangular matrix without diagonal
  list l, ic1, ic2;
  double tmp;
  clock_t t1, t2;

  init_cluster_array(C, n, Kmax);

  t1 = clock();
  euc_dists(X, n, p, D);
  t2 = clock();

  if (debug) printf("calculate dist matrix time = %g[s]\n", (double)(t2-t1)/CLOCKS_PER_SEC);

  RANDIN;

  while(1) {
    maxscore = 0.0;

    if (debug)
      printf("split viwer\n");

    for (k = 0; k < K; k++) {

      cluster = cluster_array[k];

      nk = list2vec(ck, cluster);

      qsort(ck, nk, sizeof(int), asc_int);

      if (debug) printf(" %d:", k);

      tmp = testCluster_unimodality(D, nk, ck, a, vthd, debug);

      if (debug) printf("\n");

      if (tmp > maxscore) {
        maxscore = tmp;
        tg = k;
      }
    }

    if (maxscore > 0.0) { // devide the target cluster into two new clusters
      k = tg;
      nk = list2vec(ck, cluster_array[k]);
      i = ck[(int)(nk * UNIF)];

      if (debug) {
        printf("split %d: score = %.8f\n", k, maxscore);
        printf("cluster size %d, sample i=%d\n", nk, i);
      }

      // Initialized two centers using the method recommended by the authors in section 3 of the paper
      for (j = 0; j < p; j++) {
        vec2[j] = M[k*p + j] - (X[i*p + j] - M[k*p + j]);
        vec1[j] = X[i*p + j];
      }

      //compute samples new assignment
      ic1 = make_dummy();
      ic2 = make_dummy();
      two_means(X, p, ck, nk, vec1, vec2, ic1, ic2);

      // change the data structure
      free_que(cluster_array[k]);

      cluster_array[k] = NULL;

      for (j = 0; j < p; j++) {
        M[k*p + j] = vec1[j];
        M[K*p + j] = vec2[j];
      }

      cluster_array[k] = ic1;
      cluster_array[K] = ic2;

      K = K+1;
    } else {
      if (debug) printf("   converged\n");
      break;
    }
    if (K >= Kmax)
      break;
  }

  // convert list to array
  if (debug) printf("k = %d\n",k);
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
  free(D); free(ck);

  RANDOUT;
}

void dipmeans_th(double *X, int *pn, int *pp, double *M, int *pK, int *C,
                 int *pnn_len, int *nn, double *qdips, double *pvthd, int *pKmax, int *pdebug)
{
  /*================= arguments from R ==================*/
  int n = *pn, p = *pp, K = *pK, Kmax = *pKmax, nn_len = *pnn_len;
  double vthd = *pvthd;
  int debug = *pdebug;
  /*=====================================================*/
  int i,j,k,nk,tg;
  list cluster;
  double maxscore;
  double *vec1 = (double *)malloc(sizeof(double) * p);
  double *vec2 = (double *)malloc(sizeof(double) * p);
  int *ck = (int *)malloc(sizeof(int) * n);
  double *D = (double *)malloc(sizeof(double) * (int)(0.5 * n * (n-1))); // distances between samples, lower triangular matrix without diagonal
  list l,ic1,ic2;
  double score;
  clock_t t1,t2;

  init_cluster_array(C, n, Kmax);

  t1 = clock();
  euc_dists(X, n, p, D);
  t2 = clock();

  if (debug) printf("calculate dist matrix time = %g[s]\n", (double)(t2-t1)/CLOCKS_PER_SEC);

  RANDIN;

  while(1) {
    maxscore = 0.0;

    for (k = 0; k < K; k++) {
      cluster = cluster_array[k];
      nk = list2vec(ck, cluster);
      qsort(ck, nk, sizeof(int), asc_int);
      score = calculate_score(D, nk, ck, nn_len, nn, qdips, vthd, debug);
      if (score > maxscore) {
        maxscore = score;
        tg = k;
      }
    }

    if (maxscore > 0.0) { // devide the target cluster into two new clusters
      k = tg;
      nk = list2vec(ck, cluster_array[k]);
      i = ck[(int)(nk * UNIF)];

      if (debug) {
        printf("split %d: score = %.8f\n", k, maxscore);
        printf("cluster size %d, sample i=%d\n", nk, i);
      }

      for (j = 0; j < p; j++) {
        vec2[j] = M[k*p + j] - (X[i*p + j] - M[k*p + j]);
        vec1[j] = X[i*p + j];
      }

      //compute samples new assignment
      ic1 = make_dummy();
      ic2 = make_dummy();
      two_means(X, p, ck, nk, vec1, vec2, ic1, ic2);

      // change the data structure
      free_que(cluster_array[k]);
      cluster_array[k] = NULL;
      for (j = 0; j < p; j++) {
        M[k*p + j] = vec1[j];
        M[K*p + j] = vec2[j];
      }

      cluster_array[k] = ic1;
      cluster_array[K] = ic2;

      K = K+1;
    } else {
      if (debug) printf("   converged\n");
      break;
    }
    if (K >= Kmax)
      break;
  }
  // convert list to array
  if (debug) printf("k = %d\n",k);
  for (k = 0; k < K; k++) {
    l = cluster_array[k];
    if (l != NULL) {
      while((l = l->next) != NULL)
        C[l->ind] = k;
    }
  }

  free_cluster_array(Kmax);
  free(vec1); free(vec2);
  free(D); free(ck);

  RANDOUT;
}
