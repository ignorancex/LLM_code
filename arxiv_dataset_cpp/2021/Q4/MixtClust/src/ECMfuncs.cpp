#define ARMA_DONT_PRINT_ERRORS
#include <RcppArmadillo.h>
#include <RcppGSL.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>

using namespace Rcpp;


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]

// Make a vector into an array (in Armadillo speak, a cube) 
// [[Rcpp::export]]
arma::cube to_array(NumericVector sigmas, int K, int p) {
  NumericVector vSigmas(sigmas);
  arma::cube Sigmas(vSigmas.begin(), p, p, K, false);
  return Sigmas;
}


// Make a symmetric singular matrix non-singular by adding epsilon(tol) to the
// zero eigenvalues.
// [[Rcpp::export]]
arma::mat fix_var(arma::mat sigma, double tol = 1e-3) {
  int p = sigma.n_rows;
  arma::mat eigvec(p,p), ans(p,p);
  arma::vec eigval(p), failures(p); failures.zeros();
  arma::eig_sym(eigval, eigvec, sigma);
  double counter = 0.0;
  for (int j=0; j<p; j++) {
    bool fails = ((eigval(j) / eigval(p-1)) < tol);
    if (fails) {
      counter += 1.0;
      failures(j) = 1;
    }
  }
  if (counter > 0) {
    for (int j=0; j<p; j++) {
      if (failures(j) == 1) {
        eigval(j) = tol / (counter/p); 
      }
    }
    ans = eigvec * arma::diagmat(eigval) * eigvec.t();
  } else {
    ans = sigma;
  }
  return ans;
}


//' Mahalanobis distance
//'
//' @description Compute the squared Mahalanobis distance (fast implementation).
//'
//' @references Mahalanobis, Prasanta Chandra. "On the Generalised Distance in
//'   Statistics." Proceedings of the National Institute of Sciences of India 2
//'   (1936): 49-55.
//'
//' @param x Numeric. A vector (of length \eqn{p}) or matrix (with \eqn{p}
//'   columns).
//' @param mu Numeric. A vector of length \eqn{p}.
//' @param sigma Numeric. A \eqn{p \times p} non-negative definite matrix.
//' @param ischol Logical. Set to \code{TRUE} if \code{sigma} is provided as a
//'   Cholesky decomposition.
//'
//' @return The squared Mahalanobis distance for all rows in \code{x} and the
//'   mean vector \code{mu} with respect to covariance matrix \code{sigma},
//'   defined as \eqn{(x - \mu)' \Sigma^{-1}(x - \mu)}.
//'
//' @author Emily Goren, \email{emily.goren@gmail.com}
//'
//' @export
//' 
// [[Rcpp::export]]
arma::vec mahalanobis(arma::mat x, arma::vec mu, arma::mat sigma, bool ischol = false) {
  // Check inputs.
  if (mu.n_elem != sigma.n_cols) {
    Rcpp::stop("The supplied mean vector and covariance matrix have incompatible dimensions.");
  }
  if (x.n_cols != sigma.n_cols)  {
    Rcpp::stop("The supplied data matrix and covariance matrix have incompatible dimensions.");
  }
  // Cholesky decomp of covariance matrix -- take lower triangle.
  int n = x.n_rows, p = x.n_cols;
  arma::mat A(p,p);
  if (!ischol) {
    arma::mat Atmp(p,p);
    bool success = arma::chol(Atmp, sigma);
    if (!success) {
      Atmp = arma::chol(fix_var(sigma));
    }
    A = arma::trimatl(Atmp.t());
  } else {
    A = arma::trimatl(sigma.t());
  }
  arma::vec D = A.diag();
  // Solve linear system.
  arma::vec ans(n);
  for (int i = 0; i < n; i++) {
    arma::vec tmp(p);
    for (int j = 0; j < p; j++) {
      double s = 0.0;
      for (int k = 0; k < j; k++) {
        s += tmp(k) * A(j, k);
      }
      tmp(j) = ( x(i, j) - mu(j) - s ) / D(j);
    }
    ans.at(i) = sum(square(tmp)); 
  }
  return ans;
}

//' Multivariate t distribution
//'
//' @description Compute the density of the multivariate t distribution (fast
//'   implementation).
//'
//' @param x Numeric. A vector (of length \eqn{p}) or matrix (with \eqn{p}
//'   columns).
//' @param mu Numeric. A vector of length \eqn{p} representing the mean.
//' @param sigma Numeric. A \eqn{p \times p} non-negative definite matrix (or
//'   its Cholesky decomposition).
//' @param nu Numeric. A positive scalar representing the degrees of freedom.
//' @param logans Logical. If \code{TRUE}, the log density is returned.
//' @param ischol Logical. Set to \code{TRUE} if \code{sigma} is provided as a
//'   Cholesky decomposition.
//'
//' @return The multivariate t density for all rows in \code{x} using degrees of
//'   freedom \code{nu}, mean vector \code{mu}, and covariance matrix
//'   \code{sigma}.
//'
//' @author Emily Goren, \email{emily.goren@gmail.com}
//'
//' @export
//' 
// [[Rcpp::export]]
arma::vec dMVT(arma::mat x, arma::vec mu, arma::mat sigma, double nu, bool logans = false, bool ischol = false) {
  // Check inputs.
  if (mu.n_elem != sigma.n_cols) {
    Rcpp::stop("The supplied mean vector and covariance matrix have incompatible dimensions.");
  }
  if (x.n_cols != sigma.n_cols)  {
    Rcpp::stop("The supplied data matrix and covariance matrix have incompatible dimensions.");
  }
  int p = x.n_cols, n = x.n_rows;
  arma::mat A(p,p);
  if (!ischol) {
    bool success = arma::chol(A, sigma);
    if (!success) {
      A = arma::chol(fix_var(sigma));
    }
  } else {
    A = sigma;
  }
  arma::vec ans(n);
  arma::vec maha = mahalanobis(x, mu, A, true);
  double logDet = sum(arma::log(A.diag()));
  if (nu <= 0.0) { // MVN
    ans = -0.5*maha - (p/2.0)*std::log(2.0*M_PI) + logDet;
  } else {
    double c = lgamma((nu + p) / 2.0) - lgamma(nu / 2.0) - (p / 2.0) * std::log(nu*M_PI) - logDet;
    for (int i=0; i<n; i++) {
      ans(i) = c - 0.5*(nu+p) * log1p(maha(i)/nu);
    }
  }
  if (!logans) ans = exp(ans);
  return ans;
}


//' Multivariate t distribution with missing data
//'
//' @description Compute the marginal density of the multivariate t distribution
//'   for the observed coordinates.
//'
//' @param x Numeric. A vector (of length \eqn{p}) or matrix (with \eqn{p}
//'   columns). Missing values for row \eqn{i} correspondind to unique pattern
//'   of missingness \eqn{m} are indicated by the \eqn{m^{th}} element of the
//'   selection index list \code{D}.
//' @param mu Numeric. A vector of length \eqn{p} representing the mean.
//' @param sigma Numeric. A \eqn{p \times p} non-negative definite matrix (or
//'   its Cholesky decomposition).
//' @param nu Numeric. A postitive scalar.
//' @param grp Numeric. A vector of length \eqn{n} with elements indicating the
//'   which row of \code{Ru} to use as the observed coordinates for each of
//'   the \eqn{n} observations.
//' @param Ru Binary matrix. Each row corresponds to a unique pattern of missingness,
//' where \eqn{1} indicates observed and \eqn{0} indicates missing coordinates.
//'
//' @return The multivariate t density for all rows in \code{x} using degrees of
//'   freedom \code{nu}, mean vector \code{mu}, covariance matrix \code{sigma},
//'   and observed coordinates specified by rows in \code{Ru}.
//'
//' @author Emily Goren, \email{emily.goren@gmail.com}
//'
//' @export
//' 
// [[Rcpp::export]]
arma::vec h(arma::mat x, arma::vec mu, arma::mat sigma, double nu, arma::vec grp, arma::umat Ru) {
  int n = x.n_rows, M = Ru.n_rows;
  arma::vec ans(n);
  for (int m=0; m<M; m++) {
    int g = m+1;
    // get mu, cholesky decom of sigma for this missingness pattern
    arma::uvec oidx = arma::find(Ru.row(m) == 1);
    int pg = oidx.size();
    arma::vec mug = mu.elem(oidx);
    arma::mat sigmag = sigma.submat(oidx, oidx);
    arma::mat Rg(pg,pg);
    bool success = arma::chol(Rg, sigmag);
    if (!success) {
      Rg = arma::chol(fix_var(sigmag));
    }
    // get obs for this missingness pattern
    arma::uvec gidx = arma::find(grp == g);
    arma::mat xg = x.submat(gidx, oidx);
    ans(gidx) = dMVT(xg, mug, Rg, nu, false, true);
  }
  return ans;
}


// E-step: update Z.
// [[Rcpp::export]]
arma::mat up_Z(arma::mat x, arma::mat mus, NumericVector sigmas, arma::vec nus, arma::vec pis, arma::vec grp, arma::umat Ru) {
  int K = mus.n_rows, n = x.n_rows, p = x.n_cols;
  arma::cube Sigmas = to_array(sigmas, K, p);
  arma::mat ans(n, K);
  for (int k=0; k<K; k++) {
    ans.col(k) = pis(k) * h(x, mus.row(k).t(), Sigmas.slice(k), nus(k), grp, Ru);
  }
  for (int i=0; i<n; i++) {
    double rowsum = 0.0;
    for (int k=0; k<K; k++) {
      rowsum += ans(i,k);
    }
    ans.row(i) = ans.row(i) / rowsum;
  }
  return ans;
}


// E-step: update W.
// [[Rcpp::export]]
arma::mat up_W(arma::mat x, arma::mat mus, NumericVector sigmas, arma::vec nus, arma::vec grp, arma::umat Ru) {
  int K = mus.n_rows, n = x.n_rows, p = x.n_cols, M = Ru.n_rows;
  arma::cube Sigmas = to_array(sigmas, K, p);
  arma::mat ans(n, K);
  for (int m=0; m<M; m++) {
    int g = m+1;
    // get obs for this missingness pattern
    arma::uvec oidx = arma::find(Ru.row(m) == 1);
    arma::uvec gidx = arma::find(grp == g);
    int pg = oidx.size();
    int ng = gidx.size();
    arma::mat xg = x.submat(gidx, oidx);
    for (int k=0; k<K; k++) {
      arma::vec muk = mus.row(k).t();
      arma::mat sigmak = Sigmas.slice(k);
      // get mu, cholesky decom of sigma for this missingness pattern
      arma::vec mukg = muk.elem(oidx);
      arma::mat sigmakg = sigmak.submat(oidx, oidx);
      arma::mat Rkg(pg,pg);
      bool success = arma::chol(Rkg, sigmakg);
      if (!success) {
        Rkg = arma::chol(fix_var(sigmakg));
      }
      arma::vec maha = mahalanobis(xg, mukg, Rkg, true);
      for (int i=0; i<ng; i++) {
        ans(gidx(i), k) = ( nus(k) + pg ) / ( nus(k) + maha(i) );
      }
    }
  }
  return ans;
}


// CM-step 1: update pis.
// [[Rcpp::export]]
arma::vec up_pi(arma::mat z) {
  int n = z.n_rows, K = z.n_cols;
  arma::vec ans(K); ans.zeros();
  for (int k=0; k<K; k++) {
    for (int i=0; i<n; i++) {
      ans(k) += z(i,k);
    }
    ans(k) = ans(k) / n;
  }
  return ans;
}


// CM-step 2: update mus.
// [[Rcpp::export]]
arma::mat up_mu(arma::mat x, arma::mat z, arma::mat w, arma::mat A) {
  int p = x.n_cols, n = x.n_rows, K = z.n_cols;
  arma::mat ans(K, p);
  for (int k=0; k<K; k++) {
    arma::mat L(p,p); L.zeros();
    arma::vec R(p); R.zeros();
    for (int i=0; i<n; i++) {
      arma::mat dA = arma::diagmat(A.row(i));
      L += z(i,k) * w(i,k) * dA;
      R += z(i,k) * w(i,k) * dA * x.row(i).t();
    }
    ans.row(k) = arma::solve(L, R).t();
  }
  return ans;
}


// CM-step 2: update Sigmas.
// [[Rcpp::export]]
arma::cube up_Sigma(arma::mat x, arma::mat z, arma::mat w, arma::mat mus, arma::mat A, bool constr) {
  int p = x.n_cols, n = x.n_rows, K = z.n_cols;
  arma::cube Sigmas(p,p,K);
  for (int k=0; k<K; k++) {
    arma::mat L(p,p); L.zeros();
    arma::mat R(p,p); R.zeros();
    for (int i=0; i<n; i++) {
      arma::mat Ai = arma::diagmat(A.row(i));
      arma::vec u = x.row(i).t() - mus.row(k).t();
      L += z(i,k) * w(i,k) * Ai * u * u.t() * Ai;
      R += z(i,k) * A.row(i).t() * A.row(i);
    }
    Sigmas.slice(k) = L / R;
  }
  if (constr) {
    arma::vec pis = up_pi(z);
    arma::mat S(p,p); S.zeros();
    for (int k=0; k<K; k++) {
      S += pis(k) * Sigmas.slice(k);
    }
    for (int k=0; k<K; k++) {
      Sigmas.slice(k) = S;
    }
  }
  return Sigmas;
}


// CM-step 3: update nus -- helper functions for updating cluster-specifc
// degrees of freedom.
double approx_nu(NumericVector z, NumericVector w, double nu, NumericVector ps, int n) {
  double out, tmp = 0.0, nk = 0.0;
  for (int i=0; i<n; i++) {
    tmp += z[i] * (log(w[i]) - w[i] + R::digamma((nu + ps[i])/2.0) - log((nu + ps[i])/2.0));
    nk += z[i];
    }
  double cc = - 1.0 - tmp/nk;
  double num = - exp(cc) + 2.0*exp(cc) * (exp(R::digamma(nu/2.0)) - ((nu/2.0) - 0.5));
  double den = 1.0 - exp(cc);
  out = num / den;
  return out;
}
struct nu_pars {
  NumericVector z_k; 
  NumericVector w_k; 
  double nu_k; 
  NumericVector P;
  int n;
};
double objective_nu(double df, void *params) {
  struct nu_pars *pars;
  pars = (struct nu_pars *)params;
  int n = pars->n;
  NumericVector ps(n);
  ps = pars->P;
  double nu_k = pars->nu_k;
  NumericVector z_k(n);
  z_k = pars->z_k;
  NumericVector w_k(n);
  w_k = pars->w_k;
  double zsum = 0, asum = 0; 
  for (int i=0; i<n; i++) {
    asum += z_k[i] * (log(w_k[i])-w_k[i]+R::digamma((nu_k+ps[i])/2.0)-log((nu_k+ps[i])/2.0));
    zsum += z_k[i];
  }
  double ans = 1.0 - R::digamma(df/2.0) + log(df/2.0) + (asum/zsum);
  return ans;
}
double rootsolver_nu(NumericVector z_k, NumericVector w_k, double nu_k, NumericVector ps, int n, int iter_max = 1e6, double tol = 1e-3, double min_df = 1e-3, double max_df = 1e4) {
  int status, iter = 0;
  const gsl_root_fsolver_type *T;
  gsl_root_fsolver *solver;
  double ans;
  gsl_function F;
  struct nu_pars params = {z_k, w_k, nu_k, ps, n};
  F.function = &objective_nu;
  F.params = &params;
  T = gsl_root_fsolver_brent;
  solver = gsl_root_fsolver_alloc(T);
  gsl_root_fsolver_set (solver, &F, min_df, max_df);
  do
  {
    iter++;
    status = gsl_root_fsolver_iterate (solver);
    ans = gsl_root_fsolver_root (solver);
    min_df = gsl_root_fsolver_x_lower (solver);
    max_df = gsl_root_fsolver_x_upper (solver);
    status = gsl_root_test_interval (min_df, max_df, 0, tol);
  }
  while (status == GSL_CONTINUE && iter < iter_max);
  gsl_root_fsolver_free (solver);
  return ans;
}


// CM-step 3: update nus -- helper functions for updating constrained (same for
// all clusters) degrees of freedom.
double approx_nu_constr(NumericMatrix z, NumericMatrix w, double nu, NumericVector ps, int n, int K) {
  double out;
  double tmp = 0.0;
  for (int k=0; k<K; k++) {
    double tmpk = 0.0;
    for (int i=0; i<n; i++) {
      tmpk += z(i,k) * (log(w(i,k)) - w(i,k) + R::digamma((nu + ps[i])/2.0) - log((nu + ps[i])/2.0));
    }
    tmp += tmpk;
  }
  double cc = -1.0 - tmp/n;
  double num = -exp(cc) + 2.0*exp(cc) * (exp(R::digamma(nu/2.0)) - ((nu/2.0) - 0.5));
  double den = 1.0 - exp(cc);
  out = num / den;
  return out;
}

struct nu_constr_pars {
  NumericMatrix z; 
  NumericMatrix w;
  double nu; 
  NumericVector P;
  int K; 
  int n;
};
double objective_nu_constr(double df, void *params) {
  struct nu_constr_pars *pars;
  pars = (struct nu_constr_pars *)params;
  int n = pars->n;
  int K = pars->K;
  NumericVector ps(n);
  ps = pars->P;
  double nu = pars->nu;
  NumericMatrix z(n,K);
  z = pars->z;
  NumericMatrix w(n,K);
  w = pars->w;
  double asum = 0;
  for (int k=0; k<K; k++) {
    for (int i=0; i<n; i++) {
      asum += z(i,k) * (log(w(i,k))-w(i,k)+R::digamma((nu+ps[i])/2.0)-log((nu+ps[i])/2.0));
    }
  }
  double ans = 1.0 - R::digamma(df/2.0) + log(df/2.0) + (asum/n);
  return ans;
}
double rootsolver_nu_constr(NumericMatrix z, NumericMatrix w, double nu, NumericVector ps, int n, int K, int iter_max = 1e6, double tol = 1e-3, double min_df = 1e-3, double max_df = 1e4) {
  int status, iter = 0;
  const gsl_root_fsolver_type *T;
  gsl_root_fsolver *solver;
  double ans;
  gsl_function F;
  struct nu_constr_pars params = {z, w, nu, ps, K, n};
  F.function = &objective_nu_constr;
  F.params = &params;
  T = gsl_root_fsolver_brent;
  solver = gsl_root_fsolver_alloc (T);
  gsl_root_fsolver_set (solver, &F, min_df, max_df);
  do
  {
    iter++;
    status = gsl_root_fsolver_iterate (solver);
    ans = gsl_root_fsolver_root (solver);
    min_df = gsl_root_fsolver_x_lower (solver);
    max_df = gsl_root_fsolver_x_upper (solver);
    status = gsl_root_test_interval (min_df, max_df, 0, tol);
  }
  while (status == GSL_CONTINUE && iter < iter_max);
  gsl_root_fsolver_free (solver);
  return ans;
}


// CM-step 3: update nus.
// [[Rcpp::export]]
NumericVector up_nu(NumericMatrix z, NumericMatrix w, NumericVector nus, NumericVector ps, bool constr = false, bool approx = false) {
  int n = z.nrow(), K = z.ncol();
  NumericVector ans(K); 
  if (constr) {
    double tmp;
    if (approx) {
      tmp = approx_nu_constr(z, w, nus(0), ps, n, K);
    } else {
      tmp = rootsolver_nu_constr(z, w, nus(0), ps, n, K);
    }
    if (tmp < 3.0) {
      tmp = 3.0;
    }
    if (tmp > 200) {
      tmp = 200;
    }
    for (int k=0; k<K; k++) {
      ans(k) = tmp;
    }
  } else if (!constr) {
    for (int k=0; k<K; k++) {
      if (approx) {
        ans(k) = approx_nu(z(_,k), w(_,k), nus(k), ps, n);
      } else {
        ans(k) = rootsolver_nu(z(_,k), w(_,k), nus(k), ps, n);
      }
      if (ans(k) < 3.0) {
        ans(k) = 3.0;
      }
      if (ans(k) > 200) {
        ans(k) = 200;
      }
    }
  } else {
    Rcpp::stop("Degree of freedom constraint option must be boolean.");
  }
  return ans;
}


// Code to implement full EM (also including y.NA) per Lin's method
// For a given k, compute ES'inv(SES')S, 
// where S is the selection matrix based on the observed indices in D
// and E is the dispersion of cluster k.
// [[Rcpp::export]]
arma::cube SOiOEOOk(arma::mat sigma, arma::umat Ru) {
  int p = sigma.n_cols, M = Ru.n_rows;
  arma::cube ans(p,p,M);
  for (int m=0; m<M; m++) {
    arma::uvec oidx = arma::find(Ru.row(m) == 1);
    int pg = oidx.size();
    arma::mat Rg(pg,pg);
    arma::mat sigmag = sigma.submat(oidx, oidx);
    bool success = arma::inv(Rg, sigmag);
    if (!success) {
      Rg = arma::inv(fix_var(sigmag));
    }
      arma::mat R(p,p); R.zeros();
    R(oidx, oidx) = Rg;
    ans.slice(m) = sigma * R;
  }
  return ans;
}
// For a given k, compute xhat.
// [[Rcpp::export]]
arma::mat xhatk(arma::mat x, arma::vec mu, arma::vec grp, int M, NumericVector SOiOEOOk) {
  int p = x.n_cols, n = x.n_rows;
  arma::cube R = to_array(SOiOEOOk, M, p);
  arma::mat xhat(n,p);
  for (int i=0; i<n; i++) {
    int g = grp(i) - 1;
    arma::vec u = x.row(i).t() - mu;
    arma::vec xhati = mu + R.slice(g) * u;
    xhat.row(i) = xhati.t();
  }
  return xhat;
}
// Update mus per Lins method.
// [[Rcpp::export]]
arma::mat up_mu_Lin(int p, arma::mat z, arma::mat w, ListOf<NumericMatrix> xhat) {
  int n = z.n_rows, K = z.n_cols;
  arma::mat ans(K, p);
  for (int k=0; k<K; k++) {
    arma::vec num(p); num.zeros();
    double den = 0, wt = 0;
    NumericMatrix tmp = xhat[k];
    arma::mat xhatk(tmp.begin(), n, p, false);
    for (int i=0; i<n; i++) {
      wt = z(i,k) * w(i,k);
      num += wt*xhatk.row(i).t();
      den += wt;
    }
    ans.row(k) = num.t() / den;
  }
  return ans;
}

// Update Sigmas per Lins method (one k at a time)
// [[Rcpp::export]]
arma::mat up_Sigmak_Lin(int M, arma::vec zk, arma::vec wk, arma::vec mu, arma::mat sigma,
			arma::mat xhatk, arma::vec grp, NumericVector SOiOEOOk) {
  int n = xhatk.n_rows, p = xhatk.n_cols;
  arma::mat I = arma::eye(p,p);
  arma::mat num(p,p); num.zeros();
  double den = 0;
  arma::cube R = to_array(SOiOEOOk, M, p);
  for (int i=0; i<n; i++) {
    int g = grp(i) - 1;
    arma::vec u = xhatk.row(i).t() - mu;
    num += zk(i) * (wk(i) * u * u.t() + (I - R.slice(g)) * sigma);
    den += zk(i);
  }
  arma::mat ans = num / den;
  return ans;
}

// Compute the Q2 function
// [[Rcpp::export]]
double Q2(arma::mat x, arma::mat z, arma::mat w, NumericVector sigmas, arma::mat mus, arma::vec grp, arma::umat Ru) {
  int K = mus.n_rows, n = x.n_rows, p = x.n_cols, M = Ru.n_rows;
  arma::cube Sigmas = to_array(sigmas, K, p);
  arma::mat ans(n, K);
  for (int m=0; m<M; m++) {
    int g = m+1;
    // get obs for this missingness pattern
    arma::uvec oidx = arma::find(Ru.row(m) == 1);
    arma::uvec gidx = arma::find(grp == g);
    int pg = oidx.size();
    int ng = gidx.size();
    arma::mat xg = x.submat(gidx, oidx);
    for (int k=0; k<K; k++) {
      arma::vec muk = mus.row(k).t();
      arma::mat sigmak = Sigmas.slice(k);
      // get mu, cholesky decom of sigma for this missingness pattern
      arma::vec mukg = muk.elem(oidx);
      arma::mat sigmakg = sigmak.submat(oidx, oidx);
      arma::mat Rkg(pg,pg);
      bool success = arma::chol(Rkg, sigmakg);
      if (!success) {
        Rkg = arma::chol(fix_var(sigmakg));
      }
      double logDet = 2.0 * sum(arma::log(Rkg.diag()));
      arma::vec maha = mahalanobis(xg, mukg, Rkg, true);
      for (int i=0; i<ng; i++) {
        int idx = gidx(i);
        ans(idx, k) = z(idx,k) / 2.0 * ( - logDet - w(idx,k) * maha(i) );
      }
    }
  }
  double out = 0;
  for (int k=0; k<K; k++) {
    for (int i=0; i<n; i++) {
      out += ans(i,k);
    }
  }
  return out;
}