#include "xmeans.h"
#include "utils.h"
#include "list.h"
#include "twomeans.h"


void test_estimate_sigma(double *X, int *pp, int *c, int *pc_len, double *mu, int *pdf, double *sigma) {
  int p = *pp, c_len = *pc_len, df = *pdf;

  *sigma = estimate_sigma(X, p, c, c_len, mu, df);
}

void test_loglik(int *n, int *p, int *k, double *sigma, double *l)
{
  *l = loglik(*n, *p, *k, *sigma);
}


void test_testCluster_bic(double *X, int *pp, int *c, int *pn, double *mu, double *vec1, double *vec2, double *bic_p, double *bic_c)
{
  int p = *pp, n = *pn;
  list ic1, ic2;

  ic1 = make_dummy();
  ic2 = make_dummy();
  two_means(X, p, c, n, vec1, vec2, ic1, ic2);

  testCluster_bic(X, p, c, n, mu, vec1, vec2, ic1, ic2, bic_p, bic_c);

  free_que(ic1); free_que(ic2);
}
