#include <RcppArmadillo.h> // new 'lighter' header

// Functions in RcppArmadillo for variational mixture model with variable selection

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export(name = ".lognullphiCalc")]]
arma::cube lognullphiCalc(arma::mat nullphi, arma::mat X, double K, double D, double N){
  arma::cube v(N, D, K);
  for (int n = 0; n < N; n++){
    for (int k = 0; k < K; k++){
      for(int d = 0; d < D; d++){
        v(n, d, k) = log(nullphi(d, X(n, d) - 1));
      }
    }
  }
  return v;
}

// [[Rcpp::export(name = ".logrhonkCalcVarSel")]]
arma::mat logrhonkCalcVarSel(arma::vec Elogpi, arma::cube carray, arma::mat cmatrix, double K, double D, double N){
  arma::mat v(N, K);
  for (int n = 0; n < N; n++){
    for (int k = 0; k < K; k++){
      double sum1 = 0; // Sum value
      for(int d = 0; d < D; d++){
        sum1 += carray(k, d, n);
      }
      double sum2 = 0; // Sum value
      for(int d = 0; d < D; d++){
        sum2 += cmatrix(n, d);
      }
      v(n, k) = Elogpi(k) + sum1 + sum2;
    }
  }
  return v;
}

// [[Rcpp::export(name = ".epsCalcVarSel")]]
arma::cube epsCalcVarSel(double K, double maxNCat, double D, double N, arma::mat prioreps, arma::mat X, arma::mat rnk, arma::vec c){
  arma::cube v(K, maxNCat, D);
  for (int k = 0; k < K; k++){
    for (int d = 0; d < D; d++){
      for (int l = 0; l < maxNCat; l++){
        double sum = 0; // Sum value
        for(int n = 0; n < N; n++){
          if(X(n, d) == l+1){
            sum += rnk(n, k) * c(d);
          } 
        }
        v(k, l, d) = prioreps(d, l) + sum;
      }
    }
  }
  return v;
}

// [[Rcpp::export(name = ".CpostdeltaCalc")]]
arma::vec CpostdeltaCalc(arma::vec c, double a, double D){
  arma::vec v(D);
  for (int d = 0; d < D; d++){
    v(d) = lgamma(1 + 2*a) - lgamma(c(d) + a) - lgamma(1 - c(d) + a);
  }
  return v;
}

// [[Rcpp::export(name = ".logeta1Calc")]]
arma::vec logeta1Calc(arma::cube Elogphi, arma::mat rnk, arma::vec Elogdelta, double K, double D, double N){
  arma::vec v(D);
  for (int d = 0; d < D; d++){
    double sum = 0; // Sum value
    for (int n = 0; n < N; n++){
      for(int k = 0; k < K; k++){
        sum += Elogphi(k, d, n) * rnk(n, k);
      }
    }
    v(d) = sum + Elogdelta(d);
  }
  return v;
}

// [[Rcpp::export(name = ".logeta2Calc")]]
arma::vec logeta2Calc(arma::cube lognullphi, arma::mat rnk, arma::vec Elogminusdelta, double K, double D, double N){
  arma::vec v(D);
  for (int d = 0; d < D; d++){
    double sum = 0; // Sum value
    for (int n = 0; n < N; n++){
      for(int k = 0; k < K; k++){
        sum += lognullphi(n, d, k) * rnk(n, k);
      }
    }
    v(d) = sum + Elogminusdelta(d);
  }
  return v;
}

// [[Rcpp::export(name = ".cCalc")]]
arma::vec cCalc(arma::vec logeta1, arma::vec clse, double D){
  arma::vec v(D);
  for (int d = 0; d < D; d++){
    v(d) = exp(logeta1(d) - clse(d));
  }
  return v;
}

// [[Rcpp::export(name = ".sumDElogphiCalcVarSel")]]
arma::mat sumDElogphiCalcVarSel(arma::cube carray, arma::mat cmatrix, double K, double D, double N){
  arma::mat v(N, K);
  for (int n = 0; n < N; n++){
    for (int k = 0; k < K; k++){
      double sum = 0; // Sum value
      for(int d = 0; d < D; d++){
        sum += carray(k, d, n) + cmatrix(n, d);
      }
      v(n, k) = sum;
    }
  }
  return v;
}

// [[Rcpp::export(name = ".logrhonkCalcProfCat")]]
arma::mat logrhonkCalcProfCat(arma::vec Elogpi, arma::mat Elogtheta, arma::vec y, arma::cube carray, arma::mat cmatrix, double K, double D, double N){
  arma::mat v(N, K);
  for (int n = 0; n < N; n++){
    for (int k = 0; k < K; k++){
      double sum1 = 0; // Sum value
      for(int d = 0; d < D; d++){
        sum1 += carray(k, d, n);
      }
      double sum2 = 0; // Sum value
      for(int d = 0; d < D; d++){
        sum2 += cmatrix(n, d);
      }
      v(n, k) = Elogpi(k) + Elogtheta(k, y(n) - 1) + sum1 + sum2;
    }
  }
  return v;
}