// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;

// Calculate test statistic using pairwise differences
// [[Rcpp::export]]
double ecf_reg_test_stat(const arma::vec& e1, const arma::vec& e01,
                         const arma::vec& e2, const arma::vec& e02) {
  int n1 = e1.n_elem;
  int n2 = e2.n_elem;
  double T1 = 0.0;
  double T2 = 0.0;
  double sigma = 1.0;
  
  // Group 1 calculations
  for(int i = 0; i < n1; i++) {
    for(int j = 0; j < n1; j++) {
      // e1_diffs
      double diff = e1(i) - e1(j);
      T1 += std::exp(-0.5 * std::pow(sigma * diff, 2));
      
      // e01_diffs
      diff = e01(i) - e01(j);
      T1 += std::exp(-0.5 * std::pow(sigma * diff, 2));
      
      // e1_e01_diffs
      diff = e1(i) - e01(j);
      T1 -= 2.0 * std::exp(-0.5 * std::pow(sigma * diff, 2));
    }
  }
  
  // Group 2 calculations
  for(int i = 0; i < n2; i++) {
    for(int j = 0; j < n2; j++) {
      // e2_diffs
      double diff = e2(i) - e2(j);
      T2 += std::exp(-0.5 * std::pow(sigma * diff, 2));
      
      // e02_diffs
      diff = e02(i) - e02(j);
      T2 += std::exp(-0.5 * std::pow(sigma * diff, 2));
      
      // e2_e02_diffs
      diff = e2(i) - e02(j);
      T2 -= 2.0 * std::exp(-0.5 * std::pow(sigma * diff, 2));
    }
  }
  
  // Combined statistic
  double T = (1.0/n1) * T1 + (1.0/n2) * T2;
  
  return T;
}