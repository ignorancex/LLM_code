#include <Rcpp.h>
using namespace Rcpp;

// Functions in Rcpp for variational mixture model with variable selection

// [[Rcpp::export(name = ".cmatrixCalc")]]
NumericMatrix cmatrixCalc(NumericMatrix nullphi, NumericMatrix X, NumericVector c, double N, double D) {
  NumericMatrix cmatrix(N, D );
  for (int n = 0; n < N; n++){
    for(int d = 0; d < D; d++){
      cmatrix(n, d) = (1 - c(d)) * log(nullphi(d, X(n, d) - 1));
    }
  }
  return cmatrix;
}

// [[Rcpp::export(name = ".betaCalc")]]
NumericMatrix betaCalc(NumericVector priorbeta, NumericVector y, double K, double J, double N, NumericMatrix rnk){
  NumericMatrix v(K, J);
  for (int k = 0; k < K; k++){
    for (int j = 0; j < J; j++){
      double sum = 0; // Sum value
      for(int n = 0; n < N; n++){
        if(y(n) == j+1){
          sum += rnk(n, k);
        } 
      }
      v(k, j) = priorbeta(j) + sum;
    }
  }
  return v;
}


// [[Rcpp::export(name = ".CpriorbetaCalc")]]
NumericVector CpriorbetaCalc(NumericVector priorbeta, double K, double J){
  NumericVector v = NumericVector(Dimension(K));
  for (int k = 0; k < K; k++){
    double sum1 = 0; // Sum value
    double sum2 = 0;
    for(int j = 0; j < J; j++){
      sum1 += priorbeta(j);
      sum2 += lgamma(priorbeta(j));
    }
    v(k) = lgamma(sum1) - sum2;
  }
  return v;
}

// [[Rcpp::export(name = ".CpostbetaCalc")]]
NumericVector CpostbetaCalc(NumericMatrix beta, double K, double J){
  NumericVector v = NumericVector(Dimension(K));
  for (int k = 0; k < K; k++){
    double sum1 = 0; // Sum value
    double sum2 = 0;
    for(int j = 0; j < J; j++){
      sum1 += beta(k, j);
      sum2 += lgamma(beta(k, j));
    }
    v(k) = lgamma(sum1) - sum2;
  }
  return v;
}

// [[Rcpp::export(name = ".respthetaCalc")]]
NumericMatrix respthetaCalc(NumericMatrix Elogtheta, NumericMatrix rnk, NumericVector y, double N, double K){
  NumericMatrix v(N, K);
  for (int k = 0; k < K; k++){
    for (int n = 0; n < N; n++){
      v(n, k) = rnk(n, k) * Elogtheta(k, y(n) - 1);
    }
  }
  return v;
}

// [[Rcpp::export(name = ".nullphiCalc")]]
NumericMatrix nullphiCalc(NumericMatrix X, NumericVector nCat, double maxNCat, double D, double N) {
  NumericMatrix v(D, maxNCat);
  for (int d = 0; d < D; d++){
    double J = nCat[d];
    for (int j = 0; j < J; j++){
      double sum = 0; // Sum value
      for (int n = 0; n < N; n++){
        if(X(n,d) == j+1){
          sum += 1;
        } 
      }
      v(d, j) = sum / N;

    }
  }
  return v;
}

// [[Rcpp::export(name = ".firstbetaCalc")]]
NumericMatrix firstbetaCalc(NumericVector y, NumericVector priorbeta, double K, double J, double N, NumericVector clusterInit) {
  NumericMatrix v(K, J);
  for (int k = 0; k < K; k++){
    for (int j = 0; j < J; j++){
      double sum = 0; // Sum value
      for (int n = 0; n < N; n++){
        if(clusterInit(n) == k + 1 && y(n) == j+1){
          sum += 1;
        } 
      }
      v(k, j) = priorbeta(j) + sum;
    }
  }
  return v;
}

