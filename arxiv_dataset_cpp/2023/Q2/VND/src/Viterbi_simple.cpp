#include <math.h>
#include <RcppArmadillo.h>


using namespace Rcpp;
using namespace arma;

//' Viterbi path
//' 
//' Produces a the estimated viterbi path for given data and parameters.
//' @name Viterbi_simple
//' @param y numeric vector of data
//' @param K integer containing the number of states
//' @param pi vector containing the initial probability of the system
//' @param P_trans KxK matrix of tranisitiong probabilities
//' @param mu vector containing the level for each state
//' @param var vector containing the variance for each state
//' @return vector of length equal to the length of the data containing the viterbi path
//' @export

// [[Rcpp::export]]
arma::vec Viterbi_simple(NumericVector y, int K, NumericVector pi, arma::mat P_trans, NumericVector mu, NumericVector var){
  int n = y.size();
  double pot;
  uvec indices;
  vec est_x(n), prob(K);
  mat delta(K,n), psi(K,n-1);
  NumericVector mydnorm(NumericVector x, NumericVector means, NumericVector sds);
  delta.col(0) = vec(mydnorm(rep(y(0),K),mu,sqrt(var))*pi);
  for( int i=0 ; i<n-1 ; i++ ){
    for( int j=0 ; j<K ; j++){
      prob = delta.col(i)%P_trans.col(j);
      indices = sort_index(prob, "descend");
      psi(j,i) = indices(0);
      delta(j,i+1) = R::dnorm(y(i+1), mu(j), sqrt(var(j)), 0) *prob(indices(0));
    }
    //small values
    if (max(delta.col(i+1)) == 0) {
      double tmp = min(mu - y(i+1));
      for(int j=0; j<K; j++){
        delta(j,i+1) = R::dnorm(y(i+1), mu(j), tmp/35, 0) * prob(indices(0));
      }
    }
    pot = floor(log10(max(delta.col(i+1))));
    delta.col(i+1) = delta.col(i+1)*pow(10,-pot);
  }
  //recursion
  indices = sort_index(delta.col(n-1),"descend");
  est_x(n-1) = indices(0);
  for( int i=n-2 ; i>-1 ; i--){
    est_x(i) = psi(est_x(i+1),i);
  }
  est_x = est_x +1;

  return est_x;
}
