/*
  Implementation of Mario A. T. Figueiredo and Anil K. Jain (2002).

  This C code is based on the MATLAB code developed by the authors, available at:
  http://www.lx.it.pt/~mtf/mixturecode2.zip

  Note that the MATLAB code and the paper's pseudo-code are slightly different. I adopted the implementation of MATLAB code. There is useful information on Cross Validated:
  https://stats.stackexchange.com/questions/423935/figueiredo-and-jains-gaussian-mixture-em-convergence-criterion

 */

#include "utils.h"
#include "em_gmm.h"

double mml(double *props, double loglik, double dmover2, int n, int M);
void estimate_mixprop(int comp, double *props, double n_m, double dmover2, int n, int M);
void lshift_props(double *props, int comp, int M);
void lshift_weights(double *weights, int comp, int n, int M, int Mmax);

params_link kill_comp(int comp, params_link pl);
void print_debug(double *props, double *weights, int n, int p, int M);

params_link params;

void mixtures4(double *data, double *likelihoods, double *mus, double *covs, double *props, int *pcov_type,
               int *pn, int *pp, double *pdmover2, int *pM, int *pMmin, double *pth, int *pcountf, double *dl,
               double *logliks, int *kappas, int *trans1, int *trans2, int *lives, double *pmindl,
               int *blives, double *bmus, double *bcovs, double *bprops, int *pitmax, int *psilent)
{
  /* from R */
  int cov_type = *pcov_type;
  int n = *pn, p = *pp, M = *pM, Mmin = *pMmin, itmax = *pitmax, silent = *psilent;
  double dmover2 = *pdmover2, th = *pth;

  int i, j, k, l, m, comp, min_comp, countf, Mmax;
  double *weights;
  double sum, tmp, dm, n_m;
  double mindl, loglik;
  int k_cont, killed;
  params_link pl, pl_b;

  params_link (*init_list)(double *, double *, int, int);
  void (*free_list)(params_link);
  double (*mvnorm)(double *, params_link, int);
  double (*mstep)(params_link, double *, double *, int, int);
  void (*copy_cov)(double *, double *, int, int);

  if (cov_type == 0) {

    init_list = init_list_fullcov;
    free_list = free_list_fullcov;
    mvnorm = mvnorm_fullcov;
    mstep = mstep_fullcov;
    copy_cov = copy_fullcov;

  } else if (cov_type == 1) {

    init_list = init_list_diagcov;
    free_list = free_list_diagcov;
    mvnorm = mvnorm_diagcov;
    mstep = mstep_diagcov;
    copy_cov = copy_diagcov;

  }

  // make list
  params = (*init_list)(mus, covs, M, p);

  logliks[countf] = mixloglik(params, mvnorm, data, props, n, p, M);
  if (silent == 0)
    printf("%03d : loglik = %.4f\n", countf, logliks[0]);

  // initial likelihood
  pl = params;
  for (m = 0; m < M; m++) {
    for (i = 0; i < n; i++)
      likelihoods[i + m*n] = exp((*mvnorm)(&data[i*p], pl, p));
    pl = pl->next;
  }

  // weights means the posterior probability of a component.
  // In the following procedure, in fact, it is sufficient to keep the posterior probability for a single component.
  weights = (double *)malloc(sizeof(double) * n);

  mindl = DOUBLE_XMAX; // \mathcal{L}_min = +\infty
  countf = 0;  // t = 0
  k_cont = 1;  // flag of knz > kmin
  Mmax = M;  // knz = kmax
  while(k_cont) {
    do {
      comp = 0;
      pl = params;
      while (comp < M) {
        // estimate weights of `comp`
        //   w_m^(i) = alpha_m u_m^(i) / (sum_{j=1}^{kmax} alpha_j u_j^(i)) for i = 1,2,..,n.
        n_m = 0.0;
        for (i = 0; i < n; i++) {
          sum = EPS;

          for (m = 0; m < M; m++)
            sum += props[m] * likelihoods[i + m*n];
          weights[i] = props[comp] * likelihoods[i + comp*n] / sum;

          n_m += weights[i];
        }

        // special part of M step for killing small component
        //  alpha_m = max {0, (sum_{i=1}^n w_m^(i))-N/2} / n
        //  alpha_m = alpha_m / (sum_{j=1}^kmax alpha_j) # renormalized
        estimate_mixprop(comp, props, n_m, dmover2, n, M);

        if (props[comp] == 0.0) {

          if (silent == 0)
            printf(" ---Delete comp = %d in M = %d\n", comp, M);

          trans1[countf] = 1;

          // kill the component
          pl = kill_comp(comp, pl);

          // left shift props and weights
          lshift_props(props, comp, M);
          lshift_weights(likelihoods, comp, n, M, Mmax);

          // decrease the number of components.
          M = M - 1;

        } else {
          // M-step for `comp`
          n_m = (*mstep)(pl, data, weights, n, p);

          // calculate likelihood of `comp`
          //  u_m^(i) = p(y^(i) | theta_m)
          for (i = 0; i < n; i++)
            likelihoods[i + comp*n] = exp((*mvnorm)(&data[i*p], pl, p));

          // The next component will be investigated.
          comp++;
          pl = pl->next;

        }
      } //while(M > Mmin && comp < M)

      countf++; // t = t + 1

      // update likelihoods
      pl = params;
      for (m = 0; m < M; m++) {
        for (i = 0; i < n; i++)
          likelihoods[i + m*n] = exp((*mvnorm)(&data[i*p], pl, p));
        pl = pl->next;
      }

      // calculate evaluation
      loglik = mixloglik(params, mvnorm, data, props, n, p, M);
      logliks[countf] = loglik;

      // calculate current model's MML
      dl[countf] = mml(props, loglik, dmover2, n, M);
      kappas[countf] = M;

      if (silent == 0)
        printf("%03d : M = %2d, loglik = %.3f, MML = %.3f\n", countf, M, loglik, dl[countf]);

      // validate termination rule
      tmp = logliks[countf] - logliks[countf-1];

      if (countf >= (itmax - 2)) {
        k_cont = 0;
        break;
      }

    } while(fabs(tmp / logliks[countf-1]) > th); // |L(t) - L(t-1)| < eps |L(t-1)|

    // check live components
    for (m = 0; m < Mmax; m++)
      lives[m] = 0;
    pl = params;
    for (m = 0; m < M; m++) {
      if (pl == NULL)
        error("params_link pl is empty!");
      lives[pl->ind] = 1;
      pl = pl->next;
    }

    // \mathcal{L}(\theta(t), Y) < \mathcal{L}_{min}
    if (dl[countf] < mindl) {
      mindl = dl[countf];
      for (m = 0; m < Mmax; m++) blives[m] = lives[m];
      copy_prop(props, bprops, Mmax);
      copy_mean(mus, bmus, p, Mmax);
      (*copy_cov)(covs, bcovs, p, Mmax);
    }

    // force to kill smallest component
    //  m* = argmin_m {alpha_m > 0}, alpha_m* = 0, knz = knz - 1
    //  {alpha_1, alpha_2, ..., alpha_kmax} = {alpha_1, ..., alpha_kmax} / (sum_{m=1}^kmax alpha_m)

    if (M > Mmin) {
      // find the smallest comp
      min_comp = -1;
      dm = DOUBLE_XMAX;
      for (m = 0; m < M; m++) {
        if (dm > props[m]) {
          min_comp = m;
          dm = props[m];
        }
      }

      if (min_comp == -1)
        error("can't find the smallest component");

      if (silent == 0)
        printf(" ---force to kill comp = %d, prop = %.5f\n", min_comp, dm);

      // kill the comp
      pl = params;
      for (m = 0; m < min_comp; m++) pl = pl->next;
      kill_comp(min_comp, pl);

      // flusleft
      lshift_props(props, min_comp, M);
      lshift_weights(likelihoods, min_comp, n, M, Mmax);

      // done delete
      M = M-1;

      // renormalize the mixing proportions
      sum = 0.0;
      for (m = 0; m < M; m++)
        sum += props[m];
      for (m = 0; m < M; m++)
        props[m] /= sum;

      trans2[countf] = 1;

      countf++;

      // calculate evaluation
      loglik = mixloglik(params, mvnorm, data, props, n, p, M);
      logliks[countf] = loglik;

      // MML
      dl[countf] = mml(props, loglik, dmover2, n, M);
      kappas[countf] = M;
    } else {
      k_cont = 0;
    }
    if (countf >= (itmax-1))
      k_cont = 0;
  }

  *pmindl = mindl;

  *pcountf = countf;
  free(weights);
  (*free_list)(params);
}

double mml(double *props, double loglik, double dmover2, int n, int M)
/*
  The formula is taken from the authors' code. Note that it differs slightly from Eq. (15).
 */
{
  int m;
  double sum = 0.0;

  for (m = 0; m < M; m++)
    sum += log(props[m]);

  return ( -loglik + dmover2 * sum + (dmover2 + 0.5) * M * log(n) );
}

void estimate_mixprop(int comp, double *props, double n_m, double dmover2, int n, int M)
/*
  Corresponds to Eq. (17). Note that the denominator is the data size (taken from MATLAB code).
*/
{
  int m;
  double sum;

  props[comp] = (n_m - dmover2) / n;
  if (props[comp] < 0)
    props[comp] = 0.0;

  sum = EPS;
  for (m = 0; m < M; m++)
    sum += props[m];
  for (m = 0; m < M; m++)
    props[m] /= sum;
}

void lshift_props(double *props, int comp, int M)
{
  int m;

  for (m = comp; m < M-1; m++)
    props[m] = props[m+1];
  props[m] = 0;
}

void lshift_weights(double *weights, int comp, int n, int M, int Mmax)
{
  int i, m;

  for (i = 0; i < n; i++ ) {
    for (m = comp; m < M-1; m++)
      weights[i + m*n] = weights[i + (m+1)*n];
    weights[i + m*n] = 0;
  }
}

params_link kill_comp(int comp, params_link pl)
{
  int m;
  params_link pl_b;

  if (comp == 0) {
    params = pl->next;
    return params;
  } else {
    pl_b = params;
    for (m = 0; m < (comp-1); m++) pl_b = pl_b->next;
    pl_b->next = pl->next;
  }
  free(pl);
  return pl_b->next;
}

void print_debug(double *props, double *weights, int n, int p, int M)
{
  int i,j,m,l;
  double *mu,*cov;
  params_link pl;

  pl = params;
  printf("list : ");
  do {
    printf("%d ",pl->ind);
  } while((pl = pl->next) != NULL);
  printf("\n");

  pl = params;
  for (m = 0; m < M; m++) {
    if (pl == NULL)
      error("error : pl is NULL inspite of m < M");
    l = pl->ind; mu = pl->mu; cov = pl->cov;
    printf("index : %d\n",l);
    printf(" mu :\n");
    printf("   ");
    for (j = 0; j < p; j++)
      printf("%.2f ",mu[j]);
    printf("\n");
    printf(" cov :\n");
    for (i = 0; i < p; i++) {
      printf("   ");
      for (j = 0; j < p; j++)
        printf("%.2f ",cov[i*p + j]);
      printf("\n");
    }
    pl = pl->next;
  }

  printf("props :\n");
  printf("  ");
  for (m = 0; m < M; m++)
    printf("%.2f ",props[m]);
  printf("\n");

  printf("weights :\n");
  for (i = 0; i < 6; i++) {
    printf(" ");
    for (m = 0; m < M; m++)
      printf("%2.2f ",weights[i*M + m]);
    printf("\n");
  }
}
