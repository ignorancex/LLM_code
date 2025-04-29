#include <math.h>
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

int binom(int n, int k) {
  if (k > n - k) {
    k = n - k;
  }
  
  int out = 1;
  for (int i = 0; i < k; ++i) {
    out *= (n - i) / (i + 1);
  }
  
  return out;
}

int binom(int n, int k);

//' Transition matrix according to the VND model
//' 
//' Produces a transition matrix acording to the VND model for a given vector of parameters.
//' @name vnd_model
//' @param parameters vector with parameters of the VND model \eqn{(\lambda_0,\eta_1,\lamda_1,\ldtos,\eta_\ell)}. 
//' @return the transition matrix for a VND model with the given parameters
//' @examples
//' # For l=2
//' q<-vnd_model(c(0.9,0.8,0.98,0.89))
//' q
//' @export

// [[Rcpp::export]]
NumericVector vnd_model(NumericVector parameters) {
  int l = parameters.length() / 2;
  int n_levels = l + 1;
  NumericMatrix trans_matrix(n_levels, n_levels);
  parameters.push_front(1.0);
  parameters.push_back(1.0);
  NumericMatrix p(2 , n_levels, parameters.begin());
  //cout << p << std::endl;
  
  for (int i = 0; i < n_levels; i++) {
    for (int j = 0; j < n_levels; j++) {
      trans_matrix(i,j) = 0;
      for (int k = std::max(0, i-j); k < std::min(i, l-j)+1; k++) {
        trans_matrix(i,j) += (binom(i,k) * binom(l-i, j-i+k) *
          pow(p(0,i), i-k) * pow(1-p(0,i), k) *
          pow(p(1,i), l-j-k) * pow(1-p(1,i), j-i+k));
        //cout << i << " " << j << " " << k << " " << trans_matrix(i,j) << std::endl;
      }
    }
  }
  
  return trans_matrix;
}