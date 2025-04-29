#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include "distRcpp.h"
#include "SSRFun_cpp.h"
#include "dinvgamma_cpp.h"
#include "dmvnrm_arma_fast.h"

using namespace Rcpp;
// [[Rcpp::export]]

// likelihood function
Rcpp::List likelihoodFun_T_incr_cpp(arma::mat dist_mat, double upper_bound,
                                    Rcpp::List proposal_result, 
                                    String metric, Rcpp::List hyperparList,
                                    double n_incr){
  
  double n_obj = dist_mat.n_rows;
  int m = n_obj * (n_obj - 1) / 2;
  
  // extract hyperparameters
  //double a = hyperparList["a"];
  //double b = hyperparList["b"];
  //double alpha = hyperparList["alpha"];
  //arma::vec beta = hyperparList["beta"];
  //double df_nu = hyperparList["df"];
  
  // extract proposal results
  arma::mat x_mat = proposal_result["x"];
  double sigma2 = proposal_result["sigma2"];
  arma::mat lambda = proposal_result["lambda"];
  arma::mat g = proposal_result["g"];
  int p = x_mat.n_cols;
  
  // some transformations
  arma::colvec lambda_diag = lambda.diag();
  Rcpp::NumericVector lambda_diag_rcpp = NumericVector(lambda_diag.begin(),
                                                       lambda_diag.end());

  // calculate delta matrix and d matrix
  arma::mat d_mat = dist_mat;
  Rcpp::NumericMatrix delta_mat_rcpp = distRcpp(wrap(x_mat), metric);
  arma::mat delta_mat = Rcpp::as<Mat<double>>(delta_mat_rcpp);

  // calculate SSR
  double SSR = SSRFun_cpp(d_mat, delta_mat);

  // calculate SSR * g
  arma::uvec idx = arma::trimatu_ind(arma::size(delta_mat), 1);
  vec SSR_g_vec = g.elem(idx) % square(d_mat.elem(idx) - delta_mat.elem(idx));
  double SSR_g = sum(SSR_g_vec);
  // Rcout << "SSR_g: " << SSR_g;
  
  // calculate the term of sum over log of standard normal cdf
  arma::mat temp1 = (upper_bound-trimatu(delta_mat)) % sqrt(trimatu(g)) / sqrt(sigma2);
  arma::mat temp2 = -trimatu(delta_mat) % sqrt(trimatu(g)) / sqrt(sigma2);
  arma::uvec uw_idx = arma::trimatu_ind(arma::size(delta_mat), 1);
  //lw_idx.print();
  vec temp1_l = temp1.elem(uw_idx);
  vec temp2_l = temp2.elem(uw_idx);
  //x.print();
  int nn = temp1_l.size();
  arma::vec normal_cdf1(nn);
  arma::vec normal_cdf2(nn);
  for (int i=0; i<nn; i++) {
    normal_cdf1(i) = R::pnorm(temp1_l(i),0,1,true,false);
    normal_cdf2(i) = R::pnorm(temp2_l(i),0,1,true,false);
    //Rcout << "diff(normal_cdf): " << normal_cdf1(i)-normal_cdf2(i);
  }
  
  // log_normal_cdf.print();
  double sum_log_normal_cdf = sum(log(normal_cdf1 - normal_cdf2));
  // Rcout << "sum_log_normal_cdf: " << sum_log_normal_cdf;
  
  // calculate the log likelihood
  double loglikelihood = -(m/2)*log(sigma2) - SSR_g/(2*sigma2) - sum_log_normal_cdf +
    0.5*sum(log(g.elem(idx)));
  // Rcout << "loglikelihood: " << loglikelihood;

  // calculate the log priors
  // prior for X
  arma::vec x_logprior_temp;
  double n_diff = n_obj - n_incr;
  // Rcout << "n_diff: " << n_diff;
  arma::vec x_logprior(n_diff);
  arma::rowvec x_mean(p, fill::zeros);
  //x_logprior.print();
  for (int i = 0; i < (n_obj - n_incr); i++){
    x_logprior_temp = dmvnrm_arma_fast(x_mat.row(i), x_mean,
                                       lambda, true);
    // x_logprior_vec_temp.print();
    x_logprior[i] = as_scalar(x_logprior_temp);
    // x_logprior_vec.print();
  }
  // x_logprior.print();

  // prior for sigma2
  // Rcpp::NumericVector sigma2_vec = {sigma2};
  // Rcpp::NumericVector sigma2_logprior_vec = dinvgamma_cpp(sigma2_vec, a, 1.0/b, true);
  // double sigma2_logprior = sigma2_logprior_vec[0];
  // Rcout << "sigma2_logprior_vec: " << sigma2_logprior_vec;
  // Rcout << "sigma2_logprior: " << sigma2_logprior;

  // prior for lambda
  // arma::vec lambda_logprior(p);
  // for (int j = 0; j < p; j++){
  //   Rcpp::NumericVector temp = dinvgamma_cpp(lambda_diag_rcpp,
  //                                            alpha, 1.0/beta[j], true);
  //   // Rcout << "temp: " << temp;
  //   arma::vec temp_arma = as<arma::vec>(wrap(temp));
  //   //temp_arma.print();
  //   lambda_logprior[j] = temp_arma[j];
  // }
  // lambda_logprior.print();

  // prior for g
  // vec g_upper_part = g(idx);
  // vec g_logprior_vec(idx.size());
  // for (int i = 0; i < idx.size(); i++){
  //   double temp_g = R::dgamma(g_upper_part(i), df_nu, 1/df_nu, true);
  //   g_logprior_vec(i) = temp_g;
  // }

  double logprior = sum(x_logprior);
  // double logprior = sum(x_logprior) + sigma2_logprior + sum(lambda_logprior) +
  //   sum(g_logprior_vec);
  // Rcout << "logprior: " << logprior;

  // calculate log posterior
  // As the sum of the log-likelihood and the log-prior.
  /* This is equivalent to multiplying the probabilities and then taking the log
     logposterior <- loglikelihood + logprior */
  double logposterior = loglikelihood + logprior;
  // Rcout << "logposterior: " << logposterior;

  Rcpp::List output = Rcpp::List::create(Rcpp::Named("logposterior")=logposterior,
                                         Rcpp::Named("loglikelihood")=loglikelihood,
                                         Rcpp::Named("logprior")=logprior,
                                         Rcpp::Named("SSR")=SSR);

  return(output);

}


/*** R

# likelihoodFun_T_incr_cpp(dist_mat = dist.mat,
#                          proposal_result = currentVal,
#                          metric, hyperparList, n.incr)

*/
