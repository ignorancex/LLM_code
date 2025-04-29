#include <Rcpp.h>
using namespace Rcpp;

// Functions in Rcpp for original variational mixture model and empirical mixture model

// [[Rcpp::export(name = ".rnkCalc")]]
NumericMatrix rnkCalc(NumericMatrix logrhonk, NumericVector lse, double N, double K) {
  NumericMatrix rnk(N, K );
  for (int n = 0; n < N; n++){
    for(int k = 0; k < K; k++){
      rnk(n, k) = exp(logrhonk(n, k) - lse(n));
    }
  }
  return rnk;
}


