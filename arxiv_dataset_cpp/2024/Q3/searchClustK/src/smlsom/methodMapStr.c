#include "link.h"
#include "mvnLLD.h"
#include "uLLD.h"
#include "MMethod.h"
#include "methodMapStr.h"

int uDeath(double *data, double *moments, distfunc lld, int n, int p, int M, int K, int *cls, double *lliks, int *counts, double *consts, double mdl, char **dtype)
{
  link li;
  int m,i,k,l;

  int marray_size = M*K*p;

  int nd = getDim(availNodes);
  int death = -1;

  link candidates = make_dummy();

  double mdl_c;
  int *counts_c = (int *)calloc(M,sizeof(int));
  int *cls_c = (int *)calloc(n,sizeof(int));
  double *lliks_c = (double *)calloc(n,sizeof(double));
  double *moms_c = (double *)malloc(marray_size * sizeof(double));

  double *moms_star = (double *)malloc(marray_size * sizeof(double));
  for (k = 0; k < marray_size; k++)
    moms_star[k] = moments[k];

  //
  li = availNodes;
  while ((li = li->next) != NULL) {
    m = li->node;
    pushTo(candidates, m);
  }

  li = availNodes;
  while ((li = li->next) != NULL) {
    for (k = 0; k < marray_size; k++)
      moms_c[k] = moments[k];
    m = li->node;
    getkFrom(candidates, m);
    // redistribution of samples
    ubestMatchs(data, moms_c, lld, n, p, M, K,cls_c, lliks_c, counts_c, consts, candidates);
    //-------------------debug---------------------//
    if (debug)
      printf("present map : death %d,mdl = %.1f\n",death,mdl);
    //---------------------------------------------//
    // batch for candidate map
    ubatch(data, moms_c, lld, n, p, M, K, cls_c, lliks_c, counts_c, consts, candidates, dtype);
    // calculate for candidate map
    mdl_c = uMDL(lliks_c, n, p, K, nd-1, dtype);
    //-------------------debug---------------------//
    if (debug)
      printf("  exclude node %d : mdl = %.1f\n",m,mdl_c);
    //---------------------------------------------//
    if (mdl_c < mdl) {
      mdl = mdl_c;
      for (k = 0; k < marray_size; k++)
        moms_star[k] = moms_c[k];
      death = m;
      for (i = 0; i < n; i++) {
        cls[i] = cls_c[i];
        lliks[i] = lliks_c[i];
      }
      for (l = 0; l < M; l++) counts[l] = counts_c[l];
    }
    pushTo(candidates,m);
  }
  for (k = 0; k < marray_size; k++)
    moments[k] = moms_star[k];

  if (death != -1) {
    //update map
    link fl;
    li = adjsNodes[death];
    while((li = li->next) != NULL) {
      fl = adjsNodes[death];
      while((fl = fl->next) != NULL) {
        if (li->node != fl->node) {
          if (!isIn(adjsNodes[li->node],fl->node)) {
            createEdge(li->node,fl->node);
            createEdge(fl->node,li->node);
          }
        }
      }
    }

    deleteNode(death);
    nd = nd - 1;
  }
  //-------------------debug---------------------//
  if (silent == 0)
    printf("death node %3d : nd = %3d, mdl = %.1f\n",death,nd,mdl);
  if (debug) print_que(availNodes);
  //---------------------------------------------//

  free_que(candidates);
  free(cls_c); free(counts_c); free(moms_c);
  free(moms_star);

  return death;
}

boolean uPrune(double *data, double *moments, distfunc lld, int n, int p, int M, int K, int *cls, double *consts, double beta)
{
  int i,j,m,l,ns;
  link li,smpl,adj,fadj;
  link *smplQ = (link *)malloc(M * sizeof(link));
  boolean ispr = false;
  double a = -DOUBLE_XMAX;
  double dist;
  double **D = (double **)calloc(M, sizeof(double *));
  for (m = 0; m < M; m++)
    D[m] = (double *)calloc(M, sizeof(double));

  li = availNodes;
  while ((li = li->next) != NULL)
    smplQ[li->node] = make_dummy();

  for (i = 0; i < n; i++) {
    m = cls[i];
    pushTo(smplQ[m],i);
  }

  //calculate KL-divergence between nodes
  li = availNodes;
  while ((li = li->next) != NULL) {
    m = li->node;
    ns = 0;
    smpl = smplQ[li->node];
    while ((smpl = smpl->next) != NULL) {
      ns++;
      i = smpl->node; //substitution node list for sample list
      dist = 0.0;
      for (j = 0; j < p; j++)
        dist += lld(data[i*p + j], &moments[m*K + j*M*K]);
      D[m][m] += -dist - consts[i]; // sum of loglikelihoods of m
      adj = adjsNodes[m];
      while ((adj = adj->next) != NULL) {
        l = adj->node;
        dist = 0.0;
        for (j = 0; j < p; j++)
          dist += lld(data[i*p + j], &moments[l*K + j*M*K]);
        D[m][l] += -dist - consts[i]; // sum of loglikelihoods of m->l
      }
    }
    D[m][m] = D[m][m] / ns;
    adj = adjsNodes[m];
    while ((adj = adj->next) != NULL) {
      l = adj->node;
      D[m][l] = D[m][m] - D[m][l] / ns;
    }
  }

  //---------debug---------//
  if (debug) {
    printf("print D\n");
    for (m = 0; m < M; m++) {
      if(!isIn(availNodes,m))
        continue;
      for (l = 0; l < M; l++) {
        if(!isIn(availNodes,l))
          continue;
        printf(" %6.1f ",D[m][l]);
      }
      printf("\n");
    }
  }
  //----------------------//

  //decision of edge deletion
  li = availNodes;
  while ((li = li->next) != NULL) {
    m = li->node;
    if (a < -D[m][m])
      a = -D[m][m];
  }

  // threshold
  a = beta*a;
  if (silent == 0)
    printf("     threshold = %.3f\n",a);
  li = availNodes;
  while ((li = li->next) != NULL) {
    m = li->node;
    adj = adjsNodes[m];
    while (adj->next != NULL) {
      fadj = adj->next;
      l = fadj->node;
      if ((0.5*D[m][l] + 0.5*D[l][m]) > a) {
        deleteEdge(l,m);
        adj->next = fadj->next;
        free(fadj);
        ispr = true;
        //-------------debug-------------------//
        if (debug)
          printf("  delete edge (%d, %d)\n",m,l);
        //print_list(M);
        //-------------------------------------//
      } else
        adj = adj->next;
    }
  }

  //free
  li = availNodes;
  while ((li = li->next) != NULL)
    free_que(smplQ[li->node]);
  free(smplQ);
  for (m = 0; m < M; m++)
    free(D[m]);
  free(D);

  return ispr;
}

int mvnDeath(double *data, double *mu1, double *mu2, int n, int p, int M, int *cls, double *lliks, int *counts, double mdl) {
  //mdl execpt each {m : dim(m) < 6}
  // mvn
  link li;
  int i,m,l;

  int nd = getDim(availNodes);
  int death = -1;

  link candidates = make_dummy();
  double mdl_c;
  int *counts_c = (int *)calloc(M,sizeof(int));
  int *cls_c = (int *)calloc(n,sizeof(int));
  int *cls_o = (int *)calloc(n,sizeof(int));
  int *counts_o = (int *)calloc(M,sizeof(int));

  // initialization
  for (i = 0; i < n; i++)
    cls_o[i] = cls[i];
  for (m = 0; m < M; m++)
    counts_o[m] = counts[m];

  // for all candidate node
  li = availNodes;
  while ((li = li->next) != NULL) {
    m = li->node;
    pushTo(candidates, m);
  }

  li = availNodes;
  while ((li = li->next) != NULL) {
    m = li->node;
    mvn_batch(data,mu1,mu2,n,p,cls_o,lliks,counts_o,availNodes);
    getkFrom(candidates, m);
    // redistribution of samples
    mvn_bestMatchs(data,mu1,mu2,n,p,M,cls_c,lliks,counts_c,candidates);
    //-------------------debug---------------------//
    if (debug)
      printf("present map : death %d,mdl = %.1f\n",death,mdl);
    //---------------------------------------------//
    // batch for candidate map
    // update mu1_c,mu2_c and A_m m in candidates
    mvn_batch(data,mu1,mu2,n,p,cls_c,lliks,counts_c,candidates);
    // calculate for candidate map
    mdl_c = mvnMDL(lliks, n, p, nd-1);
    // re
    //-------------------debug---------------------//
    if (debug)
      printf("  exclude node %d : mdl = %.1f\n",m,mdl_c);
    //---------------------------------------------//
    if (mdl_c < mdl) {
      mdl = mdl_c;
      death = m;
      for (i = 0; i < n; i++) cls[i] = cls_c[i];
      for (l = 0; l < M; l++) counts[l] = counts_c[l];
    }
    pushTo(candidates,m);
  }

  if (death != -1) {
    //update map
    link fl;

    li = adjsNodes[death];
    while((li = li->next) != NULL) {
      fl = adjsNodes[death];
      while((fl = fl->next) != NULL) {
        if (li->node != fl->node) {
          if (!isIn(adjsNodes[li->node],fl->node)) {
            createEdge(li->node,fl->node);
            createEdge(fl->node,li->node);
          }
        }
      }
    }

    deleteNode(death);
    nd = nd - 1;
  }

  mvn_batch(data, mu1, mu2, n, p, cls, lliks, counts, availNodes);
  mdl = mvnMDL(lliks, n, p, nd);
  //-------------------debug---------------------//
  if (silent == 0)
    printf("death node %3d : nd = %3d, mdl = %.1f\n",death,nd,mdl);
  if (debug) print_que(availNodes);
  //---------------------------------------------//

  free_que(candidates);
  //free(mu1_c); free(mu2_c);
  free(cls_c); free(counts_c);
  free(cls_o); free(counts_o);

  return death;
}

boolean mvnPrune(double *data, double *mu1, double *mu2, int n, int p, int M, int *cls, double beta) {
  int i,m,l,ns;
  link li,smpl,adj,fadj;
  link *smplQ = (link *)malloc(M * sizeof(link));
  boolean ispr = false;
  double cnst = 0.5*p * log(2*M_PI);
  double a = 0.0;
  double **D = (double **)calloc(M, sizeof(double *));
  for (m = 0; m < M; m++)
    D[m] = (double *)calloc(M, sizeof(double));

  li = availNodes;
  while ((li = li->next) != NULL)
    smplQ[li->node] = make_dummy();

  for (i = 0; i < n; i++) {
    m = cls[i];
    pushTo(smplQ[m],i);
  }

  //calculate KL-divergence between nodes
  li = availNodes;
  while ((li = li->next) != NULL) {
    m = li->node;
    ns = 0;
    smpl = smplQ[li->node];
    while ((smpl = smpl->next) != NULL) {
      ns++;
      i = smpl->node; //substitution node list for sample list
      D[m][m] += -lld_mvnorm(&data[i*p], &mu1[m*p], m, p) - cnst;
      adj = adjsNodes[m];
      while ((adj = adj->next) != NULL) {
        l = adj->node;
        D[m][l] += -lld_mvnorm(&data[i*p], &mu1[l*p], l, p) - cnst;
      }
    }
    D[m][m] = D[m][m] / ns;
    adj = adjsNodes[m];
    while ((adj = adj->next) != NULL) {
      l = adj->node;
      D[m][l] = D[m][m] - D[m][l] / ns;
    }
  }

  //---------debug---------//
  //  printf("print D\n");
  //  for (m = 0; m < M; m++) {
  //    for (l = 0; l < M; l++)
  //      printf(" %6.2f ",D[m][l]);
  //    printf("\n");
  //  }
  //----------------------//

  //decision of edge deletion
  li = availNodes;
  while ((li = li->next) != NULL) {
    m = li->node;
    if (a < -D[m][m])
      a = -D[m][m];
  }

  //a = 300*a; //????????
  a = beta*a;

  //-----------------debug---------------//
  if (silent == 0)
    printf("edge deletion : threshold = %.2f\n",a);
  //-------------------------------------//
  li = availNodes;
  while ((li = li->next) != NULL) {
    m = li->node;
    adj = adjsNodes[m];
    while (adj->next != NULL) {
      fadj = adj->next;
      l = fadj->node;
      if ((0.5*D[m][l] + 0.5*D[l][m]) > a) {
        deleteEdge(l,m);
        adj->next = fadj->next;
        free(fadj);
        ispr = true;
        //-------------debug-------------------//
        if (debug)
          printf("  delete edge (%d, %d)\n",m,l);
        //print_list(M);
        //-------------------------------------//
      } else
        adj = adj->next;
    }
  }

  //free
  li = availNodes;
  while ((li = li->next) != NULL)
    free_que(smplQ[li->node]);
  free(smplQ);
  for (m = 0; m < M; m++)
    free(D[m]);
  free(D);

  return ispr;
}
