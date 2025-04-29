#include <Rcpp.h>
using namespace Rcpp;

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp 
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

// [[Rcpp::export]]
// Calculate relative effective sample size
double rESSFun_cpp(const std::vector<double>& logW) {
  
  Rcpp::NumericVector A;
  A = logW;
  
  int K = logW.size();
  // Rprintf("K: %i", K);
  
  // Find the maximum logW
  double logWmax = *std::max_element(logW.begin(), logW.end());
  // Rprintf("logWmax: %f", logWmax);
  
  // Calculate logRESS
  for (int i = 0; i < K; i++)
    A[i] -= logWmax;
  //Rf_PrintValue(A);
  
  for (int i = 0; i < K; i++)
    A[i] = std::exp(2*A[i]);
  //Rf_PrintValue(A);
  
  double logRESS = -(2 * logWmax + log((sum(A)))) - log(K);
  
  //Rprintf("logRESS: %f", logRESS);
  
  return exp(logRESS);
}

