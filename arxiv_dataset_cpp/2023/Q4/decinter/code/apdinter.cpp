#include <Rcpp.h>
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
NumericMatrix apdinter_cpp(NumericMatrix x, int n, NumericMatrix w, int B){
  
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
  NumericMatrix bootdiff(B, 9);
  
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
    
    // Calculate quantiles
    for(int q = 0; q < 9; q++) {
      bootdiff(i, q) = sum(boot_apd_ab * w(q, _ )) - sum(boot_apd_cd * w(q, _ ));
    }
    
  }
  
  // Return results
  // List res;
  // res["diff"] = diff;
  // res["bootdiff"] = bootdiff;
  // return res;
  return bootdiff;
}

