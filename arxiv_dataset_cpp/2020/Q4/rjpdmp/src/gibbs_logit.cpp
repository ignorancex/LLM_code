#include <RcppArmadillo.h>
#include <R_ext/Utils.h>
#include <iostream>
#include <exception>
#include "RNG.h"
#include "PolyaGamma.h"
// Rcpp::depends(RcppArmadillo)

using namespace Rcpp;

// Code from https://github.com/jgscott/helloPG following instructions for PG package.
// Slight adjustment for rpg
arma::vec rpg(arma::vec shape, arma::vec scale) {
  // C++-only interface to PolyaGamma class
  // draws random PG variates from arma::vectors of n's and psi's
  RNG r;
  PolyaGamma pg;
#ifdef USE_R
  GetRNGstate();
#endif
  int d = shape.n_elem;
  arma::vec result(d);
  for(int i=0; i<d; i++) {
    result[i] = pg.draw(shape(i), scale(i), r);
  }
#ifdef USE_R
  PutRNGstate();
#endif
  return result;
}

double g_fun(const arma::mat& dataX, arma::vec& beta, arma::vec& gamma, arma::vec& omega, arma::vec& kappa, double prior_sigma2){
  double s = sum(gamma);
  if( s < 1 ){
    return(0);
  }
  arma::uvec nzero = find(arma::abs(gamma) > 1e-10);
  arma::mat dataX_sub = dataX.cols(nzero);
  arma::mat V = inv(dataX_sub.t()*diagmat(omega)*dataX_sub + arma::eye(s,s)/prior_sigma2);
  arma::vec const_log = 0.5*kappa.t()*dataX_sub*V*dataX_sub.t()*kappa;
  double log_post = 0.5*log(arma::det(2*arma::datum::pi*V)) - (s/2.0)*log(2*arma::datum::pi*prior_sigma2) +const_log[0];
  return(log_post);
}

//' gibbs_logit
//'
//' Applies the Collapsed Gibbs Sampler to a Logistic regression with dirac spike and slab distribution, as detailed in Reversible Jump PDMP Samplers for Variable Selection, 2020.
//' For included variables an independent Gaussian prior is assumed with variance \code{prior_sigma2} and mean zero, variables are given prior probabilities of inclusion \code{ppi}.
//' Code makes use of the package set-up for Polya-Gamma simulation available at \code{https://github.com/jgscott/helloPG}.
//'
//' @param maxTime Maximum runtime (in Seconds) of the algorithm; will terminate the code after a given computation time or nmax iterations of the algorithm is reached.
//' @param dataX Matrix of all covariates where the i-th row corresponds to all p covariates x_{i,1}, ..., x_{i,p} of the i-th observation.
//' @param datay Vector of n observations of a {0, 1}-valued variable y.
//' @param prior_sigma2 Double for the prior variance for included variables. Default 10.
//' @param beta Initial position of the regression parameter
//' @param gamma Initial model for the sampler. Enteries should either be 1s or 0s.
//' @param ppi Double for the prior probability of inclusion (ppi) for each parameter.
//' @param nsamples Maximum number of samples. Default value is 10^5, lower values should be chosen for memory constraints if less samples are desired.
//' @return Returns a list with the following objects:
//' @return \code{beta}: Matrix of regression parameter samples, columns are samples.
//' @return \code{gamma}: Matrix of model parameter samples columns are samples.
//' @return \code{times}: computation times at sampled events - Useful for plotting computational efficiency.
//' @examples
//'
//' generate.logistic.data <- function(beta, n.obs, Sig) {
//' p <- length(beta)
//' dataX <- MASS::mvrnorm(n=n.obs,mu=rep(0,p),Sigma=Sig)
//' vals <- dataX %*% as.vector(beta)
//' generateY <- function(p) { rbinom(1, 1, p)}
//' dataY <- sapply(1/(1 + exp(-vals)), generateY)
//' return(list(dataX = dataX, dataY = dataY))
//' }
//'
//' n <- 15
//' p <- 25
//' beta <- c(1, rep(0, p-1))
//' Siginv <- diag(1,p,p)
//' Siginv[1,2] <- Siginv[2,1] <- 0.9
//' set.seed(1)
//' data <- generate.logistic.data(beta, n, solve(Siginv))
//' ppi <- 2/p
//'
//' zigzag_fit <- zigzag_logit(maxTime = 1, dataX = data$dataX,
//'                            datay = data$dataY, prior_sigma2 = 10,
//'                            theta0 = rep(0, p), x0 = rep(0, p), rj_val = 0.6,
//'                            ppi = ppi)
//'
//' gibbs_fit <- gibbs_logit(maxTime = 1, dataX = data$dataX, datay =data$dataY,
//'                          prior_sigma2 = 10,beta = rep(0,p), gamma =rep(0,p),
//'                          ppi = ppi)
//'\dontrun{
//' plot_pdmp(zigzag_fit, coords = 1:2, inds = 1:1e3,burn = .1,
//'           nsamples = 1e4, mcmc_samples =t(gibbs_fit$beta*gibbs_fit$gamma))
//'}
//' @export
// [[Rcpp::export]]
List gibbs_logit(const arma::mat& dataX, const arma::vec& datay, arma::vec beta, arma::vec gamma,
                 double ppi = 0.5, int nsamples = 1e5, double maxTime = 1e8, double prior_sigma2 = 10.0){
  arma::wall_clock timer;
  timer.tic();
  int n = datay.size();
  int p = dataX.n_cols;
  int sval = p;

  arma::vec omega2(n), gamma_change(p), times(nsamples), pgscale(n), pgshape(n);
  omega2.ones();  pgscale.ones();

  double ratio_val, p_gamma_change, g_fun_cashe, prior_diff_gamma;

  arma::vec kappa = datay -0.5*arma::ones(n,1);
  arma::mat beta_samples = arma::zeros(p,nsamples);
  arma::mat gamma_samples = arma::ones(p,nsamples);

  for( int i = 0; i < nsamples; i++){
    g_fun_cashe = g_fun(dataX,beta,gamma,omega2,kappa,prior_sigma2);
    // Sample gamma
    for(int j = 0; j < p; j++){
      gamma_change = gamma;
      if(gamma[j] > 0.0){
        gamma_change[j] = 0.0;
        prior_diff_gamma = -log(ppi) + log(1-ppi);
      } else {
        gamma_change[j] = 1.0;
        prior_diff_gamma = log(ppi) - log(1-ppi);
      }
      ratio_val = g_fun(dataX,beta,gamma_change,omega2,kappa,prior_sigma2) - g_fun_cashe;
      p_gamma_change = 1.0/(1.0+exp(-ratio_val-prior_diff_gamma));

      if(R::runif(0,1) < p_gamma_change){
        gamma = gamma_change;
        g_fun_cashe = ratio_val + g_fun_cashe;
      }
    }
    sval = sum(gamma);
    if( sval > 0 ){

      arma::uvec nzero = find(arma::abs(gamma) > 1e-10);
      arma::mat dataX_sub = dataX.cols(nzero);
      // Sample Beta
      arma::mat V_omega = arma::inv(dataX_sub.t()*diagmat(omega2)*dataX_sub + arma::eye(sval,sval)/prior_sigma2);
      arma::vec m_omega = V_omega*dataX_sub.t()*kappa;
      beta.elem(nzero) = m_omega + arma::chol(V_omega).t()*arma::randn(sval);
      // sample omega
      pgshape = dataX_sub*beta.elem(nzero);
      omega2 = rpg(pgscale, pgshape);
      //Rcout << "omega " << omega2;
    } else{
      // sample omega
      omega2 = rpg(pgscale,arma::zeros(n,1));
    }

    beta_samples.col(i) = beta;
    gamma_samples.col(i) = gamma;
    times(i) = timer.toc();
    if(timer.toc() > maxTime){
      beta_samples.shed_cols(i+1, nsamples-1);
      gamma_samples.shed_cols(i+1, nsamples-1);
      times.shed_rows(i+1, nsamples-1);
      break;
    }
  }
  List ret ;
  ret["beta"] = beta_samples ;
  ret["gamma"] = gamma_samples ;
  ret["times"] = times ;

  return(ret) ;
}
