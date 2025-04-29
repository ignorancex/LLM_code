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
List apdinter_b1b9_cpp(NumericMatrix x, int n, NumericMatrix w, int B){
  
  // Extract vectors
  NumericVector a = x(_, 0); // A1B1
  NumericVector b = x(_, 1); // A1B2
  NumericVector c = x(_, 2); // A2B1
  NumericVector d = x(_, 3); // A2B2
 
  // bootstrap boot 1 method
  NumericMatrix bootdiff_b1(B, 9);
  
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
      bootdiff_b1(i, q) = sum(boot_apd_ab * w(q, _ )) - sum(boot_apd_cd * w(q, _ ));
    }
    
  }

  // bootstrap boot 9 method
  NumericMatrix bootdiff_b9(B, 9);
    
    // Calculate quantiles
    for(int q = 0; q < 9; q++) {

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

          bootdiff_b9(i, q) = sum(boot_apd_ab * w(q, _ )) - sum(boot_apd_cd * w(q, _ ));

        }
        
    }
  
  // Return results
  List res;
  res["bootdiff_b9"] = bootdiff_b9;
  res["bootdiff_b1"] = bootdiff_b1;
  return res;
}

