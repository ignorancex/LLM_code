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

//' Transition matrix according to the Chung-Kennedy model
//' 
//' Produces a transition matrix acording to the Chung-Kennedy (CK) model for a given vector of parameters.
//' @name ck_model
//' @param parameters vector with parameters of the ck model. 
//' @param l integer containing the number of signals (channels).
//' @return the transition matrix for a ck-model with the given parameters \eqn{\lambda}, \eqn{\eta}, \eqn{k}
//' @examples
//' # For l=2
//' q<-ck_model(c(0.9,0.8,0.5), 2)
//' q
//' @export

// [[Rcpp::export]]
NumericVector ck_model(NumericVector p, int l) {
  int n_levels = l + 1;
  NumericMatrix trans_matrix(n_levels, n_levels);
  //cout << p << std::endl;

  for (int i = 0; i < n_levels; i++) {
    for (int j = 0; j < n_levels; j++) {
      trans_matrix(i,j) = 0;
      if ((j == 0) || (j == l)) {
        trans_matrix(i,j) = 0.5*p[2];
      }

      for (int k = std::max(0, i-j); k < std::min(i, l-j)+1; k++) {
        trans_matrix(i,j) += ((1 - p[2]) * binom(i,k) * binom(l-i, j-i+k) *
          pow(p[1], i-k) * pow(1-p[1], k) *
          pow(p[0], l-j-k) * pow(1-p[0], j-i+k));
        //cout << i << " " << j << " " << k << " " << trans_matrix(i,j) << std::endl;
      }
    }
  }
  trans_matrix(0,0) += (p[0] - 0.5) * p[2];
  trans_matrix(0,l) += (0.5 - p[0]) * p[2];
  trans_matrix(l,0) += (0.5 - p[1]) * p[2];
  trans_matrix(l,l) += (p[1] - 0.5) * p[2];

  return trans_matrix;
}
