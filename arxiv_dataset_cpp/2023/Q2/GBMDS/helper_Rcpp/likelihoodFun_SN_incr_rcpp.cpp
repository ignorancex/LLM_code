#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include "distRcpp.h"
#include "SSRFun_cpp.h"
#include "dinvgamma_cpp.h"
#include "dmvnrm_arma_fast.h"
#include "mypsnorm_cpp.h"

using namespace Rcpp;
// [[Rcpp::export]]

// likelihood function
Rcpp::List likelihoodFun_SN_incr_cpp(arma::mat dist_mat, double upper_bound,
                                     Rcpp::List proposal_result, 
                                     String metric, Rcpp::List hyperparList,
                                     double n_incr){
  
  double n_obj = dist_mat.n_rows;
  int m = n_obj * (n_obj - 1) / 2;
  
  // extract hyperparameters
  //double a = hyperparList["a"];
  //double b = hyperparList["b"];
  //double c = hyperparList["c"];
  //double d = hyperparList["d"];
  //double alpha = hyperparList["alpha"];
  //arma::vec beta = hyperparList["beta"];
  
  // extract proposal results
  arma::mat x_mat = proposal_result["x"];
  double sigma2 = proposal_result["sigma2"];
  arma::mat lambda = proposal_result["lambda"];
  double psi = proposal_result["psi"];
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
  
  // calculate the term of sum over log of standard normal cdf
  arma::mat temp = psi * (trimatl(d_mat) - trimatl(delta_mat)) / sqrt(sigma2);
  arma::uvec lw_idx = arma::trimatl_ind(arma::size(temp), -1);
  //temp.print();
  vec temp_l = temp.elem(lw_idx);
  int nn1 = temp_l.size();
  arma::vec log_normal_cdf(nn1);
  for (int i=0; i<nn1; i++) {
    log_normal_cdf(i) = R::pnorm(temp_l(i),0,1,true,true);
  }
  
  //Rcout << "log_normal_cdf(i): " << log_normal_cdf(50);
  //Rcout << "temp_l(i): " << temp_l(50);
  
  //log_normal_cdf.print();
  double sum_log_normal_cdf = sum(log_normal_cdf);
  //Rcout << "sum_log_normal_cdf: " << sum_log_normal_cdf;
  
  // calculate the term from truncated part
  arma::uvec delta_idx = arma::trimatl_ind(arma::size(delta_mat), -1);
  //delta_idx.print();
  vec delta_l = delta_mat.elem(delta_idx);
  // delta_l.print();
  int nn2 = delta_l.size();
  arma::vec truncated_term_vec(nn2);
  for (int i=0; i<nn2; i++) {
    // SEXP truncated_term1 = psnorm_cpp(1e10, delta_l(i), sqrt(sigma2), psi);
    // SEXP truncated_term2 = psnorm_cpp(0, delta_l(i), sqrt(sigma2), psi);
    // double truncated_term12 = *REAL(truncated_term1) - *REAL(truncated_term2);
    double truncated_term1 = mypsnorm_cpp(upper_bound, delta_l(i), sqrt(sigma2), psi);
    double truncated_term2 = mypsnorm_cpp(0, delta_l(i), sqrt(sigma2), psi);
    double truncated_term12 = truncated_term1 - truncated_term2;
    truncated_term_vec(i) = log(abs(truncated_term12));
    if (!arma::is_finite(truncated_term_vec(i))){
      truncated_term_vec(i) = 0;
    }
  }
  double truncated_term = sum(truncated_term_vec);
  //Rcout << "truncated_term: " << truncated_term;
  
  // calculate the log likelihood
  double loglikelihood = -(m/2)*log(sigma2) - truncated_term -
    SSR/(2*sigma2) + sum_log_normal_cdf;
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
  
  // prior for psi
  // double psi_logprior = R::dunif(psi,c,d,true);
  // Rcout << "psi_logprior: " << psi_logprior;
  
  double logprior = sum(x_logprior);
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

# likelihoodFun_SN_incr_cpp(dist_mat = dist.mat,
#                           proposal_result = currentVal,
#                           metric, hyperparList, n.incr)


*/
