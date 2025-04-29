/*
  Implementation of `Hamerly, Greg, and Charles Elkan. "Learning the k in k-means." Advances in neural information processing systems 16 (2003).`

  gmeans(X,M,C)
    # X is input data matrix:
    #   X = {x_1, x_2, ..., x_n}, where x_i is p-dimentional real value vector
    # C = {c_1, c_2,.., c_K} where c_k is the index set of samples that belong to cluster k
    # M is centroids:
    #   M = {m_1, m_2, ..., m_K}, where m_k is the centroid of cluster k
    REPEAT
       {M,C} <- kmeans(X,M)
       stack = NULL
       FOR k in 1:K
         push(stack, k)
       ENDFOR
       # to do the Anderson-Darling (AD) test
       FOR k = pop(stack)
         X_k = {x_i | i in c_k}
         s = the 1st principal component of X_k
         X'_k = projected X_k onto s
         Normalize X'_k so that its mean and variance are 0, 1 respectively.
         Do AD test for X'_k with the specified confidence level.
         IF the test concluded cluster k is not gaussian like
           split cluster k into two new clusters k1,k2
           push(stack,k1)
           push(stack,k2)
         ENDIF
       ENDFOR
    UNTIL stack is NULL

 */

#include "gmeans.h"
#include "twomeans.h"
#include "utils.h"

list *cluster_array;

boolean testCluster_ADtest(double *X, int p, int *ck, int nk, double *vec1, double *vec2, double *d, double cv);
double ADstatic(double *z, int n);

double first_prcomp(double *X, int p, int *ck, int nk, double *mk, double **cov, double *vec1, double *vec2);
double normalCDF(double value);


void gmeans(double *X, int *pn, int *pp, double *M, int *pK,
            int *C, double *pcv, int *pKmax, int *pdebug)
{
  /*================= arguments from R ==================*/
  int n = *pn, p = *pp, K = *pK, Kmax = *pKmax;
  double cv = *pcv;
  int debug = *pdebug;
  /*=====================================================*/
  int j, k;
  boolean test;
  double d;

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

  // for prcomp: initialize covariance matrix
  double lambda, tmp;
  double **cov = (double **)malloc(sizeof(double *) * p);
  for (j = 0; j < p; j++)
    cov[j] = (double *)malloc(sizeof(double) * p);

  // cluster_array[k] is the list of indices of samples belong to cluster k.
  init_cluster_array(C, n, Kmax);

  k = getTopFrom(stack); // satisfy k is not -1
  do {
    // kth cluster is investigated whether splittd or not.
    if (debug) {
      printf("pick %d\n", k);
      printf("n = %d\n", getDim(cluster_array[k]));
    }

    nk = list2vec(ck, cluster_array[k]); // nk is the number of samples in cluster k.

    // Initialized two centers using the principal components based method
    // recommended by the authors in section 2.1 of the paper

    // maximum eigen vector and value for initializing two new centers.
    lambda = first_prcomp(X, p, ck, nk, &M[k*p], cov, vec1, vec2);
    tmp = sqrt((2*lambda) / M_PI);

    for (j = 0; j < p; j++) {
      vec2[j] = M[k*p + j] - vec1[j]*tmp;
      vec1[j] = M[k*p + j] + vec1[j]*tmp;
    }

    //compute new assignments
    ic1 = make_dummy();
    ic2 = make_dummy();
    two_means(X, p, ck, nk, vec1, vec2, ic1, ic2);

    // AD test
    test = testCluster_ADtest(X, p, ck, nk, vec1, vec2, &d, cv);

    if (debug) printf("  D = %.5f\n", d);

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

  for (j = 0; j < p; j++) free(cov[j]);
  free(cov);

  free_cluster_array(Kmax);
  free(vec1); free(vec2);
  free_que(stack); free(ck);
}

void projection(double *X, int p, int *ck, int nk, double *vec1, double *vec2, double *prj_x) {

  int i, j, t;
  double v_norm, sum;
  double *v = (double *)malloc(sizeof(double) * p);

  // v = c1 - c2
  for (j = 0; j < p; j++)
    v[j] = vec1[j] - vec2[j];

  // ||v||^2
  v_norm = 0.0;
  for (j = 0; j < p; j++)
    v_norm += v[j] * v[j];

  // project X_k onto v
  for (t = 0; t < nk; t++) {
    i = ck[t];

    // xi_ = <xi, v> / ||v||^2
    prj_x[t] = 0.0;
    for (j = 0; j < p; j++)
      prj_x[t] += X[i*p + j] * v[j];

    prj_x[t] /= v_norm;
  }

  // normalize projected X_k
  //-- transform its mean 0
  sum = 0.0;

  for (t = 0; t < nk; t++)
    sum += prj_x[t];

  sum = sum / nk;

  for (t = 0; t < nk; t++)
    prj_x[t] = prj_x[t] - sum;

  //-- transform its variance 1
  sum = 0.0;

  for (t = 0; t < nk; t++)
    sum += prj_x[t]*prj_x[t];

  sum = sqrt(sum / (nk-1));

  for (t = 0; t < nk; t++)
    prj_x[t] = prj_x[t] / sum;

  free(v);
}

boolean testCluster_ADtest(double *X, int p, int *ck, int nk, double *vec1, double *vec2, double *d, double cv)
{
  if (nk <= 5)
    return false;

  int t;
  double *prj_x = (double *)malloc(sizeof(double) * nk);

  projection(X, p, ck, nk, vec1, vec2, prj_x);

  // sort
  qsort(prj_x, nk, sizeof(double), asc_double);

  // zi = F(xi_) where F is the cumulative distribution function (standard normal)
  for (t = 0; t < nk; t++)
    prj_x[t] = normalCDF(prj_x[t]);

  // Eq. 1 and Eq. 2
  *d = ADstatic(prj_x, nk);

  free(prj_x);

  if ((*d) > cv)
    return true; // it is not normal
  else
    return false; // it is normal
}

double ADstatic(double *z, int n)
{
  int i;
  double st = 0.0;

  // Eq. 1
  for (i = 0; i < n; i++)
    st += (2*(i+1)-1) * (log(z[i]) + log(1-z[n-1-i]));
  st = -st / n - n;

  // Eq. 2
  st = st * (1 + 4.0/n - 25.0/(n*n));

  return st;
}

void covariance(double *X, int p, int *ck, int nk, double *mk, double **cov)
{
  int i, j, t, ik;

  // clean up
  for (i = 0; i < p; i++) {
    for (j = 0; j < p; j++)
      cov[i][j] = 0;
  }

  // calculate sample covariance matrix
  for (ik = 0; ik < nk; ik++) {
    i = ck[ik];
    for (t = 0; t < p; t++) {
      for (j = 0; j < p; j++)
        cov[t][j] += (X[i*p + t] - mk[t]) * (X[i*p + j] - mk[j]);
    }
  }

  // normalize
  for (t = 0; t < p; t++) {
    for (j = 0; j < p; j++)
      cov[t][j] /= nk;
  }
}

double eigen_power(double **cov, int p, double *mk,double *vec1, double *vec2)
{
  int j,t,it;
  double sum = 0.0;
  double l1,l2 = 0.0;

  // normalize mk
  for (j = 0; j < p; j++)
    sum += mk[j] * mk[j];
  sum = sqrt(sum);
  for (j = 0; j < p; j++)
    vec1[j] = mk[j] / sum;

  it = 0;
  do {
    l1 = l2;
    // multiplying a vector by matrix
    for (t = 0; t < p; t++) {
      sum = 0.0;
      for (j = 0; j < p; j++)
        sum += cov[t][j] * vec1[j];
      vec2[t] = sum;
    }
    // calculate current estimation of eigen value
    l2 = 0.0;
    for (j = 0; j < p; j++)
      l2 += vec2[j] * vec1[j];
    // replace vec1 with normalized vec2 for next iteration
    sum = 0.0;
    for (j = 0; j < p; j++)
      sum += vec2[j] * vec2[j];
    sum = sqrt(sum);
    for (j = 0; j < p; j++)
      vec1[j] = vec2[j] / sum;
    it++;
  } while(fabs(l1 - l2) / l2 > EPS && it < MAXIT);

  return l2;
}

double first_prcomp(double *X, int p, int *ck, int nk, double *mk, double **cov, double *vec1, double *vec2)
{
  double lambda;

  // covariance
  covariance(X, p, ck, nk, mk, cov);

  // power method
  lambda = eigen_power(cov, p, mk, vec1, vec2);

  return lambda;
}

double normalCDF(double value)
{
  return 0.5 * erfc(-value * M_SQRT1_2);
}
