#include "twomeans.h"

list *cluster_array;

void test_create_cluster_array(int *C, int *pn, int *pKmax) {
  int k;
  int n = *pn, Kmax = *pKmax;

  init_cluster_array(C, n, Kmax);

  for (k = 0; k < Kmax; k++)
    print_que(cluster_array[k]);

  free_cluster_array(Kmax);
}

void test_two_means(double *X, int *pp, int *c, int *pn, double *cnt1, double *cnt2)
{
  int p = *pp, n = *pn;

  list ic1 = make_dummy();
  list ic2 = make_dummy();

  two_means(X, p, c, n, cnt1, cnt2, ic1, ic2);

  print_que(ic1);
  print_que(ic2);

  free_que(ic1);
  free_que(ic2);
}
