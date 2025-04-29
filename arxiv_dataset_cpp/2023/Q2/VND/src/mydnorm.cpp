#include <math.h>
#include <RcppArmadillo.h>
//#include <typeinfo>


using namespace Rcpp;
using namespace arma;

//function for vectorial computation of normal density
// [[Rcpp::export]]
NumericVector mydnorm( NumericVector x, NumericVector means, NumericVector sds){
    int n = x.size() ;
    NumericVector res(n) ;
    for( int i=0; i<n; i++) res(i) = R::dnorm( x(i), means(i), sds(i),0) ;
    return res ;
}

// [[Rcpp::export]]
arma::vec density_pois( int x, NumericVector lambda){
  int n = lambda.size() ;
  NumericVector res(n) ;
  for( int i=0; i<n; i++) res(i) = R::dpois(x,lambda(i),0) ;
  return res ;
}

// [[Rcpp::export]]
arma::mat mydnorm2( arma::mat x, arma::mat means, arma::mat sds){
  mat res(x.n_rows,x.n_cols);
  for( int i=0; i<x.n_rows; i++) {
    for (int j=0; j<x.n_cols; j++){
      res(i,j) = R::dnorm( x(i,j), means(i,j), sds(i,j),0 ) ;
    }
  }
  return res ;
}

// [[Rcpp::export]]
arma::vec mydnorm3( NumericVector x, NumericVector means, NumericVector sds){
  int n = x.size() ;
  vec res(n) ;
  for( int i=0; i<n; i++) res(i) = R::dnorm( x(i), means(i), sds(i),0) ;
  return res ;
}

// [[Rcpp::export]]
arma::vec mydnorm4( arma::vec x, arma::vec means, arma::vec sds){
  int n = x.size() ;
  vec res(n) ;
  for( int i=0; i<n; i++) res(i) = R::dnorm( x(i), means(i), sds(i),0) ;
  return res ;
}

// [[Rcpp::export]]
arma::mat mydnorm5( double x, arma::vec means, arma::vec sds, int K, int r, arma::vec beta){
  mat res(K,pow(K,r-2)) ;
  for( int i=0; i<pow(K,r-1); i++) res(i) = R::dnorm( x, means(i), sds(i),0)*beta(i) ;
  return res ;
}

// [[Rcpp::export]]
arma::mat mydnorm6( double x, arma::vec means, arma::vec sds, int K){
  int KK = means.size(), KK_1 = KK/K ;
  mat res(KK_1,K) ;
  for( int i=0; i<KK; i++) res(i) = R::dnorm( x, means(i), sds(i),0) ;
  return res ;
}


// [[Rcpp::export]]
arma::vec mydnorm7( double x, arma::vec means, arma::vec sds, arma::vec beta){
  vec res(means.size()) ;
  for( int i=0; i<means.size(); i++) res(i) = R::dnorm( x, means(i), sds(i),0) * beta(i) ;
  return res ;
}
