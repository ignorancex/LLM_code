#include <Rcpp.h>
#include <stdio.h>
#include <string>

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::NumericVector qt7_cpp(Rcpp::NumericVector x, Rcpp::NumericVector probs) {
  // implementation of type 7
  // code from https://github.com/RcppCore/Rcpp/issues/967
  const size_t n=x.size(), np=probs.size();
  if (n==0) return x;
  if (np==0) return probs;
  Rcpp::NumericVector index = (n-1.)*probs, y=x.sort(), x_hi(np), qs(np);
  Rcpp::NumericVector lo = Rcpp::floor(index), hi = Rcpp::ceiling(index);

  for (size_t i=0; i<np; ++i) {
    qs[i] = y[lo[i]];
    x_hi[i] = y[hi[i]];
    if ((index[i]>lo[i]) && (x_hi[i] != qs[i])) {
      double h;
      h = index[i]-lo[i];
      qs[i] = (1.-h)*qs[i] + h*x_hi[i];
    }
  }
  return qs;
}

using namespace Rcpp;
// [[Rcpp::export]]
NumericVector outermin_cpp(NumericVector x, NumericVector y){
  int nx = x.size();
  int ny = y.size();
  NumericVector res(nx*ny); // store results here
  int inc = 0; // increment res vector
  for ( int xstep = 0; xstep < nx; xstep++ ) {
    for ( int ystep = 0; ystep < ny; ystep++ ) {
      res(inc) = x(xstep) - y(ystep);
      inc = inc + 1;
    }
  }
  return res;
}

// [[Rcpp::export]]
List apdinter_hdqt7_cpp(NumericMatrix x, int n, NumericMatrix w, int B, NumericVector probs){
  
  // Extract vectors
  NumericVector a = x(_, 0); // A1B1
  NumericVector b = x(_, 1); // A1B2
  NumericVector c = x(_, 2); // A2B1
  NumericVector d = x(_, 3); // A2B2
  
  // Compute distributions of all pairwise differences
  // NumericVector apd_ab = outermin_cpp(a, b);
  // NumericVector apd_cd = outermin_cpp(c, d);
  
  // Preallocate storage for deciles
  // NumericVector diff(9);
  
  // apd_ab = apd_ab.sort();
  // apd_cd = apd_cd.sort();
  
  // Calculate quantiles
  // for(int q = 0; q < 9; q++) {
  //   diff(q) = sum(apd_ab * w(q, _ )) - sum(apd_cd * w(q, _ ));
  // }

  // Preallocate storage for bootstrap results
  NumericMatrix bootABhd(B, 9);
  NumericMatrix bootABqt(B, 9);
  
  // Perform bootstrap 
  for(int i = 0; i < B; i++) {
    
    // Bootstrap: sample with replacement
    NumericVector boota = sample(a, n, true);
    NumericVector bootb = sample(b, n, true);
    NumericVector bootc = sample(c, n, true);
    NumericVector bootd = sample(d, n, true);
    
    NumericVector boot_apd_ab = outermin_cpp(boota, bootb);
    NumericVector boot_apd_cd = outermin_cpp(bootc, bootd);
    boot_apd_ab = boot_apd_ab.sort();
    boot_apd_cd = boot_apd_cd.sort();
    
    // Calculate quantiles: HD
    for(int q = 0; q < 9; q++) {
      bootABhd(i, q) = sum(boot_apd_ab * w(q, _ )) - sum(boot_apd_cd * w(q, _ ));
    }
    // Calculate quantiles: QT7
    bootABqt(i, _) = qt7_cpp(boot_apd_ab, probs) - qt7_cpp(boot_apd_cd, probs);
  }
  
  // Return results
  List res;
  res["bootABhd"] = bootABhd;
  res["bootABqt"] = bootABqt;
  return res;
}

