#include <math.h>
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>


using namespace Rcpp;
using namespace arma;

//' Simulation of Markov chains
//' 
//' Simulates \code{n_data} data points from a Markov chain with given transition matrix and intial probability.
//' @name simulate_MC
//' @param n_data integer containing the number of data points to be simulated.
//' @param P_trans transition matrix to be simulated from. It should be a square matrix.
//' @param intitial vector of length equal to the number of rows of \code{P_trans} containing the intital probability distribution.
//' @return vector of length \code{n_data} containing the simulated markov chain with state space {0,1,...,\code{nrow(P_trans)}}.
//' @examples
//' q <- vnd_model(c(0.9,0.8,0.98,0.89))
//' mc <- simulate_MC(100, q, c(1/3,1/3,1/3))
//' plot(mc, type = "s")
//' @export

// [[Rcpp::export]]
NumericVector simulate_MC(int n_data, NumericMatrix P_trans, NumericVector initial) {
  int n_levels = P_trans.nrow();
  vec markov_chain(n_data);
  vec randoms = runif(n_data);
  markov_chain[0] = 0;
  double tmp = 0.;
  for (int j = 0; j < n_levels; j++) {
    tmp += initial[j];
    if (randoms[0] < tmp) {
      markov_chain[0] = j;
      break;
    }
  }
  for (int t=1; t<n_data; t++){
    int i = markov_chain[t-1];
    tmp = 0.;
    markov_chain[t] = 0;
    for (int j = 0; j < n_levels; j++) {
      tmp += P_trans(i,j);
      if (randoms[t] < tmp) {
        markov_chain[t] = j;
        break;
      }
    }
  }

  return as<NumericVector>(wrap(markov_chain));
}

