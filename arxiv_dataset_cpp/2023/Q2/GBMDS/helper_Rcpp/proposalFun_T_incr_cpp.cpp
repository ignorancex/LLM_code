#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include "rmvnorm_arma.h"
#include "distRcpp.h"
#include "SSRFun_cpp.h"
#include "dinvgamma_cpp.h"
#include "dmvnrm_arma_fast.h"
#include "likelihoodFun_T_incr_cpp.h"
//#include "dproposalFun_cpp.h"
#include "logReferenceRatio_cpp.h"
using namespace Rcpp;
// [[Rcpp::export]]

// proposal function
Rcpp::List proposalFun_T_incr_cpp(arma::mat dist_mat, Rcpp::List currentVal, 
                                  arma::mat prevX, double annealingPar,
                                  String metric, Rcpp::List hyperparList,
                                  double n_incr, double upper_bound,
                                  arma::vec update_list, arma::vec update_list_x){
  
  double n_obj = dist_mat.n_rows;
  int m = n_obj * (n_obj - 1) / 2;
  
  // extract hyperparameters
  double a = hyperparList["a"];
  double b = hyperparList["b"];
  double alpha = hyperparList["alpha"];
  arma::vec beta = hyperparList["beta"];
  double constant_multiple = hyperparList["constant_multiple"];
  double df_nu = hyperparList["df"];
  
  // extract current results
  arma::mat x_cur = currentVal["x"];
  double sigma2_cur = currentVal["sigma2"];
  arma::mat lambda_cur = currentVal["lambda"];
  double SSR_cur = currentVal["SSR"];
  arma::mat g_cur = currentVal["g"];
  int p = x_cur.n_cols;
  
  // calculate delta matrix and d matrix
  arma::mat d_mat = dist_mat;
  Rcpp::NumericMatrix delta_mat_rcpp = distRcpp(wrap(x_cur), metric);
  arma::mat delta_mat = Rcpp::as<Mat<double>>(delta_mat_rcpp);
  
  // Propose new values for parameters
  // lambda_j
  // The full conditional posterior distrbuiton of lambda_j is the inverse Gamma distribution
  // lambda_j ~ IG(alpha + n/2 , beta_j + s_j/2)
  // where s_j/n is the sample variance of the jth coordinates of x_i's
  int cnt1 = std::count(update_list.begin(), update_list.end(), 1);
  arma::vec lambda_proposal_diag(p);
  // arma::mat lambda_initial(p, p, arma::fill::zeros);
  arma::mat lambda_proposal;
  if (cnt1 > 0){
    for (int j = 0; j < p; j++){
      // calculate shape parameter
      double lambda_shape = alpha + n_obj / 2;
      // calculate scale parameter
      double sj_n = var(x_cur.col(j)) * (n_obj - 1) / n_obj;
      double lambda_scale = beta[j] + sj_n * n_obj / 2;
      // Rcout << "lambda_shape: " << lambda_shape;
      // Rcout << "lambda_scale: " << lambda_scale;
      lambda_proposal_diag(j) = 1/arma::randg(arma::distr_param(lambda_shape,1/lambda_scale));
    }
    lambda_proposal = diagmat(lambda_proposal_diag);
  } else{
    lambda_proposal = lambda_cur;
  }
  
  // lambda_proposal.print();
  
  // x_i
  // A normal proposal density is used in the random walk Metropolis algorithm for
  // generation of x_i, i = 1, ... , n
  // Choose the variance of the normal proposal density to be a constant
  // multiple of sigma2/(n-1)
  int cnt2 = std::count(update_list.begin(), update_list.end(), 2);
  arma::mat x_proposal(n_obj, p, arma::fill::zeros);
  if (cnt2 > 0){
    // double x_var = constant_multiple * sigma2_cur / (n_obj - 1);
    arma::vec x_var(p, arma::fill::value(constant_multiple * sigma2_cur / (n_obj - 1)));
    arma::mat x_sigma = arma::diagmat(x_var);
    if (update_list_x.size() == n_obj){
      for (int i = 0; i < n_obj; i++){
        arma::rowvec temp0 = x_cur.row(i);
        NumericVector temp1 = NumericVector(temp0.begin(), temp0.end());
        x_proposal.row(i) = rmvnorm_arma(1, temp1, x_sigma);
      }
    } else{
      for (int i = 0; i < n_obj; i++){
        if (std::find(update_list_x.begin(), update_list_x.end(), i) != update_list_x.end()){
          arma::rowvec temp0 = x_cur.row(i);
          NumericVector temp1 = NumericVector(temp0.begin(), temp0.end());
          x_proposal.row(i) = rmvnorm_arma(1, temp1, x_sigma);
        } else{
          x_proposal.row(i) = x_cur.row(i);
        }
      }
    }
  } else{
    x_proposal = x_cur;
  }
  
  
  // sigma2
  // A normal proposal density is used in the random walk Metropolis algorithm for
  // generation of sigma2
  // Choose the variance of the normal proposal density to be proportional to the
  // variance of IG(m/2+a, SSR/2+b)
  int cnt3 = std::count(update_list.begin(), update_list.end(), 3);
  double sigma2_proposal;
  if (cnt3 > 0){
    double sigma2_sd = sqrt(constant_multiple * pow(SSR_cur/2 + b, 2)/(pow(m/2+a-1, 2)*(m/2+a-2)));
    sigma2_proposal = exp(R::rnorm(log(sigma2_cur), sigma2_sd));
  } else{
    sigma2_proposal = sigma2_cur;
  }
  // Rcout << "sigma2_proposal: " << sigma2_proposal;
  
  // g_ij
  // g_ij condition on other variables follows a Gamma distribution
  // g_ij ~ Gamma((1+df)/2 , (delta_ij-d_ij)^2/(2*sigma2) + df/4)
  int cnt4 = std::count(update_list.begin(), update_list.end(), 4);
  arma::mat g_proposal(n_obj, n_obj, fill::zeros);
  if (cnt4 > 0){
    double g_shape = (1+ df_nu) / 2;
    arma::uvec idx = arma::trimatu_ind(arma::size(delta_mat), 1);
    vec g_rate_vec = square(d_mat.elem(idx) - delta_mat.elem(idx)) / (2*sigma2_cur) + df_nu / 4;
    vec temp_g_vec(m);
    for (int i = 0; i < m; i++){
      double g_rate = g_rate_vec[i];
      // Rcout << "g_rate" << g_rate;
      NumericVector temp_g = rgamma(1, g_shape, 1/g_rate);
      // Rcout << "temp_g" << temp_g;
      arma::vec temp_gg = as<arma::vec>(wrap(temp_g));
      // double temp_g = rgamma(1, g_shape, g_rate);
      temp_g_vec(i) = temp_gg[0];
    }
    g_proposal.elem(idx) = temp_g_vec;
  } else{
    g_proposal = g_cur;
  }
  
  
  //g_proposal.print();
  
  Rcpp::List output;
  if (sigma2_proposal <=  1e-10 || isinf(sigma2_proposal)){
    output = Rcpp::List::create(Rcpp::Named("x")=x_cur,
                                Rcpp::Named("sigma2")=sigma2_cur,
                                Rcpp::Named("lambda")=lambda_cur,
                                Rcpp::Named("g")=g_cur);
  } else {
    
    Rcpp::List proposal = Rcpp::List::create(Rcpp::Named("x")=x_proposal,
                                             Rcpp::Named("sigma2")=sigma2_proposal,
                                             Rcpp::Named("lambda")=lambda_proposal,
                                             Rcpp::Named("g")=g_proposal);
    
    Rcpp::List result_new = likelihoodFun_T_incr_cpp(dist_mat, upper_bound, proposal,
                                                metric, hyperparList, n_incr);
    Rcpp::List result_cur = likelihoodFun_T_incr_cpp(dist_mat, upper_bound, currentVal, 
                                                metric, hyperparList, n_incr);
    
    // double dproposal_new = dproposalFun_cpp(dist_mat,
    //                                         proposal, currentVal,
    //                                         metric, hyperparList);
    // double dproposal_cur = dproposalFun_cpp(dist_mat,
    //                                         currentVal, proposal,
    //                                         metric, hyperparList);
    // Rcout << "dproposal_new: " << dproposal_new;
    // Rcout << "dproposal_cur: " << dproposal_cur;
    
    double logposterior_new = result_new["logposterior"];
    double logposterior_cur = result_cur["logposterior"];
    double log_reference_ratio = logReferenceRatio_cpp(x_proposal, x_cur, prevX);
    
    double probab = exp(//dproposal_new - dproposal_cur +
                        annealingPar * (logposterior_new - logposterior_cur) + 
                        (1-annealingPar) * log_reference_ratio);
    // Rcout << "probab: " << probab;
    
    // # accept or reject step
    double rand = R::runif(0,1);
    arma::mat x;
    double sigma2;
    arma::mat lambda;
    arma::mat g;
    double accept_count = 0;
    double accept_count_x = 0;
    if (rand < probab){
      // accept
      x = x_proposal;
      sigma2 = sigma2_proposal;
      lambda = lambda_proposal;
      g = g_proposal;
      accept_count = 1;
      if (cnt2 > 0){
        accept_count_x = 1;
      }
    } else {
      // reject
      x = x_cur;
      sigma2 = sigma2_cur;
      lambda = lambda_cur;
      g = g_cur;
      if (cnt2 > 0){
        accept_count_x = -1;
      }
    }
    
    
    
    output = Rcpp::List::create(Rcpp::Named("x")=x,
                                Rcpp::Named("accept_count") = accept_count,
                                Rcpp::Named("accept_count_x") = accept_count_x,
                                Rcpp::Named("sigma2")=sigma2,
                                Rcpp::Named("lambda")=lambda,
                                Rcpp::Named("g")=g);  
  }
  
  return(output);
  
}


/*** R

# proposalFun_T_cpp(dist_mat, currentVal, prevX, 0.3, metric, hyperparList)

# proposalFun_cpp(dist.mat, currentVal, prevX, tau[r], metric, hyperparList)

# prop.result <- proposalFun_T_incr_cpp(dist.mat, currentVal, prevX, tau[r], metric, hyperparList, n.incr)

*/
