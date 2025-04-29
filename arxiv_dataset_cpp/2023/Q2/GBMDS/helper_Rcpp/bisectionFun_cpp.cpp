#include <RcppArmadillo.h>
#include <iostream>
// [[Rcpp::depends(RcppArmadillo)]]
#include "rCESSFun_cpp.h"
using namespace Rcpp;

// [[Rcpp::export]]

// Calculate the next annealing parameter

double bisectionFun_cpp(double low, double high, 
                        arma::vec W, arma::vec logL, double phi){
  
  int n = 1000;
  double tol = 1e-7;
  double g_low = rCESSFun_cpp(W, logL, low, phi);
  double g_high = rCESSFun_cpp(W, logL, high, phi);
  double multiplier = g_low * g_high;
  
  if (!arma::is_finite(multiplier)){
    stop("Bisection flawed");
  }
  if (multiplier > 0){
    return (999e-3);
  }
  for (int i = 0; i <= n; i++){
    double mid = (low+ high) / 2;
    double g_mid = rCESSFun_cpp(W, logL, mid, phi);
    if ((g_mid == 0) || ((high - low) / 2) < tol){
      return (mid);
    } 
    if (arma::sign(g_mid) == arma::sign(g_low)){
      low = mid;
    } else{
      high = mid;
    }
  }
  
  Rcout << "Too many iterations";
}

