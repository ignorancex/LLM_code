#include "CML.h"
using namespace Rcpp;

// [[Rcpp::export]]
arma::mat testArmaCor(arma::mat M){
  return arma::cor(M);
}

