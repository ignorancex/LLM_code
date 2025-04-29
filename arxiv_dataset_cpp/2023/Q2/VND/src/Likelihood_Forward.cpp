#include <math.h>
#include <RcppArmadillo.h>



using namespace Rcpp;
using namespace arma;


// [[Rcpp::export]]
List Log_Likelihood(NumericVector data, NumericVector init,
                NumericMatrix probabilities, NumericMatrix P_trans) {
  int n_data = data.size();
  int n_levels = init.size();
  vec prob;
  mat alpha(n_levels,n_data), probs = as<mat>(probabilities), P_tr = as<mat>(P_trans);
  double log_likelihood = 0;

  //values become very small
  prob = probs.col(0) % as<vec>(init);
  alpha.col(0) = prob / sum(prob);
  log_likelihood += log(sum(prob));
  for (int i=0; i<n_data-1; i++){
    //Forward algorithm
    //recursion time for beta
    int j = n_data-1-i;
    // transition probs are changing in each time step
    prob = (trans(P_tr)*alpha.col(i))%probs.col(i+1);
    alpha.col(i+1) = prob / sum(prob);
    log_likelihood += log(sum(prob));
  }

  return List::create(Named("lik") = log_likelihood);
}

