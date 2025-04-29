#include <math.h>
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// int binom(int n, int k) {
//   if (k > n - k) {
//     k = n - k;
//   }
//
//   int out = 1;
//   for (int i = 0; i < k; ++i) {
//     out *= (n - i) / (i + 1);
//   }
//
//   return out;
// }

int binom(int n, int k);

//' Transition matrix according to the uncoupled model
//' 
//' Produces a transition matrix acording to the uncoupled (independent) model for a given vector of parameters.
//' @name uc_model
//' @param parameters vector with parameters of the uc model. 
//' @param l integer containing the number of signals (channels).
//' @return the transition matrix for a uc-model with the given parameters \eqn{\lambda}, \eqn{\eta}
//' @examples
//' # For l=2
//' q <- uc_model(c(0.9,0.8), 2)
//' q
//' @export

// [[Rcpp::export]]
NumericVector uc_model(NumericVector p, int l) {
  int n_levels = l + 1;
  NumericMatrix trans_matrix(n_levels, n_levels);
  //cout << p << std::endl;

  for (int i = 0; i < n_levels; i++) {
    for (int j = 0; j < n_levels; j++) {
      trans_matrix(i,j) = 0;
      for (int k = std::max(0, i-j); k < std::min(i, l-j)+1; k++) {
        trans_matrix(i,j) += (binom(i,k) * binom(l-i, j-i+k) *
                                  pow(p[1], i-k) * pow(1-p[1], k) *
                                  pow(p[0], l-j-k) * pow(1-p[0], j-i+k));
        //cout << i << " " << j << " " << k << " " << trans_matrix(i,j) << std::endl;
      }
    }
  }

  return trans_matrix;
}
