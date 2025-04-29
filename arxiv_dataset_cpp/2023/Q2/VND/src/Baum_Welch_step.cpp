
#include <math.h>
#include <RcppArmadillo.h>



using namespace Rcpp;
using namespace arma;


// [[Rcpp::export]]
List BW_step(NumericVector data, int n_levels,
             NumericMatrix probabilities, NumericMatrix P_trans_init ) {
  int n_data = data.size();
  uvec ind;
  vec prob, est_pi, current_error(1);
  mat alpha(n_levels,n_data), beta(n_levels,n_data), gamma(n_levels,n_data),
      y_gamma(1, n_levels), y_gamma2(1, n_levels), est_P_trans(n_levels,n_levels),
      P_trans = as<mat>(P_trans_init), probs = as<mat>(probabilities);
  NumericMatrix xi(n_levels,n_levels);
  cube epsilon = zeros<cube>(n_levels, n_levels, n_data-1);

  prob = probs.col(0);
  //values become very small
  alpha.col(0) = prob/sum(prob);
  beta.col(n_data-1) = ones<vec>(n_levels);
  for (int i=0; i<n_data-1; i++){
    //Forward-Backward algorithm
    //recursion time for beta
    int j = n_data-1-i;
    // transition probs are changing in each time step
    prob = P_trans*((beta.col(j))%probs.col(j));
    beta.col(j-1) = prob/sum(prob);

    prob = (trans(P_trans)*alpha.col(i))%probs.col(i+1);
    alpha.col(i+1) = prob/sum(prob);
  }
  //value of the likelihood stored in prob

  mat tmp = repmat(1/sum(alpha%beta),n_levels,1);
  gamma = trans(alpha%beta%tmp);
  y_gamma = sum(gamma%(repmat(vec(data),1,n_levels)));
  y_gamma2 = sum(gamma%(repmat(vec(data)%vec(data),1,n_levels)));
  //cout << sum(gamma) << std::endl;
  est_pi = trans(gamma.row(0)/sum(gamma.row(0)));

  est_P_trans.fill(0);
  for (int i=0; i<n_data-1; i++){
    epsilon.slice(i) = (repmat(alpha.col(i),1,n_levels)%P_trans)%repmat(trans((probs.col(i+1))%beta.col(i+1)),n_levels,1);
    est_P_trans = est_P_trans + epsilon.slice(i)/accu(epsilon.slice(i));
  }
  xi = clone(as<NumericMatrix>(wrap(est_P_trans)));
  est_P_trans = est_P_trans/repmat(sum(est_P_trans,1),1,n_levels);

  return List::create(Named("P_trans") = as<NumericMatrix>(wrap(est_P_trans)),
                      Named("xi") = xi,
                      Named("gamma") = as<NumericVector>(wrap(sum(gamma))),
                      Named("y_gamma") = as<NumericVector>(wrap(y_gamma)),
                      Named("y_gamma2") = as<NumericVector>(wrap(y_gamma2)),
                      Named("pi") = as<NumericVector>(wrap(est_pi)));
}

