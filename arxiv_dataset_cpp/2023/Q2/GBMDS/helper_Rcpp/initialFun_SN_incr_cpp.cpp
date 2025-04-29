#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include "reference_d_x_initial_cpp.h"
#include "rmvnorm_arma.h"
#include "distRcpp.h"
#include "SSRFun_cpp.h"
using namespace Rcpp;
// [[Rcpp::export]]

// Initialize parameters
Rcpp::List initialFun_SN_incr_cpp(arma::mat prev_result, arma::mat dist_mat, 
                                  String metric, Rcpp::List hyperparList){
  
  int n_obj = dist_mat.n_rows;
  int p = prev_result.n_cols;
  // int m = n_obj * (n_obj - 1) / 2;
  arma::mat reference_x_sd = hyperparList["reference_x_sd"];
  
  // initialize particles

  // extract hyperparameters
  double a = hyperparList["a"];
  double b = hyperparList["b"];
  double c = hyperparList["c"];
  double d = hyperparList["d"];
  double alpha = hyperparList["alpha"];
  arma::vec beta = hyperparList["beta"];
  
  // sigma2 (from prior distribution)
  double sigma2_initial = 1/arma::randg(distr_param(a,1/b));
  // Rcout << "sigma2_initial" << sigma2_initial;
  
  // lambda (from prior distribution)
  arma::vec lambda_initial_diag(p);
  // arma::mat lambda_initial(p, p, arma::fill::zeros);
  for (int i = 0; i < p; i++){
    lambda_initial_diag(i) = 1/arma::randg(distr_param(alpha,1/(beta[i])));
  }
  arma::mat lambda_initial = diagmat(lambda_initial_diag);
  
  // x_i (from reference distribution)
  arma::mat x_initial(n_obj, p, arma::fill::zeros);
  for (int i = 0; i < prev_result.n_rows; i++){
    arma::rowvec temp0 = prev_result.row(i);
    NumericVector temp1 = NumericVector(temp0.begin(), temp0.end());
    x_initial.row(i) = rmvnorm_arma(1, temp1, reference_x_sd);
    //x_initial.row(i).print();
  }
  // x_i (from prior distribution)
  for (int i = prev_result.n_rows; i < n_obj; i++){
    NumericVector temp2(p);
    x_initial.row(i) = rmvnorm_arma(1, temp2, lambda_initial);
  }
  
  // psi (from prior distribution)
  double psi_initial = R::runif(c,d);
  
  Rcpp::List output = Rcpp::List::create(Rcpp::Named("x")=x_initial,
                                         Rcpp::Named("sigma2")=sigma2_initial,
                                         Rcpp::Named("lambda")=lambda_initial,
                                         Rcpp::Named("psi")=psi_initial);
  
  return(output);
}

