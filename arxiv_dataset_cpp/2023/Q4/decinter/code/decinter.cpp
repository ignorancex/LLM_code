#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
List decinter_cpp(NumericMatrix x, int n, NumericMatrix w, int B){
  
  // Preallocate storage for deciles
  NumericVector ahd(9);
  NumericVector bhd(9);
  NumericVector chd(9);
  NumericVector dhd(9);
  NumericVector mainA(9);
  NumericVector mainB(9);
  NumericVector diff(9);
  
  NumericVector a = x(_, 0); // A1B1
  NumericVector b = x(_, 1); // A1B2
  NumericVector c = x(_, 2); // A2B1
  NumericVector d = x(_, 3); // A2B2
  
  a = a.sort();
  b = b.sort();
  c = c.sort();
  d = d.sort();
  
  // Calculate quantiles
  for(int q = 0; q < 9; q++) {
    ahd(q) = sum(a * w(q, _ ));
    bhd(q) = sum(b * w(q, _ ));
    chd(q) = sum(c * w(q, _ ));
    dhd(q) = sum(d * w(q, _ ));
  }
  
  diff = ahd - bhd - chd + dhd;
  mainA = ahd + bhd - chd - dhd;
  mainB = ahd - bhd + chd - dhd;
  
  // Preallocate storage for bootstrap results
  NumericMatrix bootdiff(B, 9);
  NumericMatrix bootmainA(B, 9);
  NumericMatrix bootmainB(B, 9);

  // Perform bootstrap 
  for(int i = 0; i < B; i++) {
    
    // Bootstrap: sample with replacement
    NumericVector boota = sample(a, n, true);
    NumericVector bootb = sample(b, n, true);
    NumericVector bootc = sample(c, n, true);
    NumericVector bootd = sample(d, n, true);
    boota = boota.sort();
    bootb = bootb.sort();
    bootc = bootc.sort();
    bootd = bootd.sort();
    
    // Calculate quantiles
    for(int q = 0; q < 9; q++) {
      // h1
      double tmp_a = sum(boota * w(q, _ ));
      double tmp_b = sum(bootb * w(q, _ ));
      double tmp_c = sum(bootc * w(q, _ ));
      double tmp_d = sum(bootd * w(q, _ ));
      bootdiff(i, q) = tmp_a - tmp_b - tmp_c + tmp_d;
      bootmainA(i, q) = tmp_a + tmp_b - tmp_c - tmp_d;
      bootmainB(i, q) = tmp_a - tmp_b + tmp_c - tmp_d;
    }
    
  }
  
  // Return results
  List res;
  res["diff"] = diff;
  res["mainA"] = mainA;
  res["mainB"] = mainB;
  res["bootdiff"] = bootdiff;
  res["bootmainA"] = bootmainA;
  res["bootmainB"] = bootmainB;
  return res;
}
