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

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
List dec_apd_all_cpp(List x, NumericMatrix wa, NumericMatrix wb, NumericMatrix wc, NumericMatrix wd, 
                     NumericMatrix wab_apd, NumericMatrix wcd_apd, int B){
  // List dec_apd_all_cpp(NumericMatrix x, int n, NumericMatrix w, NumericMatrix w_apd, int B){
  
  // =============================
  // COMPARE DECILES OF MARGINALS
  // =============================
  
  // Preallocate storage for deciles
  NumericVector ahd(9); 
  NumericVector bhd(9);
  NumericVector chd(9);
  NumericVector dhd(9);
  
  // NumericVector a = x(_, 0); // A1B1
  // NumericVector b = x(_, 1); // A1B2
  // NumericVector c = x(_, 2); // A2B1
  // NumericVector d = x(_, 3); // A2B2
  NumericVector a = x[0]; // A1B1
  NumericVector b = x[1]; // A1B2
  NumericVector c = x[2]; // A2B1
  NumericVector d = x[3]; // A2B2
  
  int na = a.length();
  int nb = b.length();
  int nc = c.length();
  int nd = d.length();
  
  a = a.sort();
  b = b.sort();
  c = c.sort();
  d = d.sort();
  
  // Calculate quantiles
  for(int q = 0; q < 9; q++) {
    ahd(q) = sum(a * wa(q, _ ));
    bhd(q) = sum(b * wb(q, _ ));
    chd(q) = sum(c * wc(q, _ ));
    dhd(q) = sum(d * wd(q, _ ));
    // diff(q) = xhd(q) - yhd(q);
  }
  
  // Preallocate storage for bootstrap results
  NumericMatrix boot_ahd(B, 9);
  NumericMatrix boot_bhd(B, 9);
  NumericMatrix boot_chd(B, 9);
  NumericMatrix boot_dhd(B, 9);
  
  NumericMatrix apd_ab_boot(B, 9);
  NumericMatrix apd_cd_boot(B, 9);
  
  // Perform bootstrap
  for(int i = 0; i < B; i++) {
    
    // Bootstrap: sample with replacement
    NumericVector boota = sample(a, na, true);
    NumericVector bootb = sample(b, nb, true);
    NumericVector bootc = sample(c, nc, true);
    NumericVector bootd = sample(d, nd, true);
    boota = boota.sort();
    bootb = bootb.sort();
    bootc = bootc.sort();
    bootd = bootd.sort();
    
    NumericVector boot_apd_ab = outermin_cpp(boota, bootb);
    NumericVector boot_apd_cd = outermin_cpp(bootc, bootd);
    boot_apd_ab = boot_apd_ab.sort();
    boot_apd_cd = boot_apd_cd.sort();
    
    // Calculate quantiles
    for(int q = 0; q < 9; q++) {
      boot_ahd(i, q) = sum(boota * wa(q, _ ));
      boot_bhd(i, q) = sum(bootb * wb(q, _ ));
      boot_chd(i, q) = sum(bootc * wc(q, _ ));
      boot_dhd(i, q) = sum(bootd * wd(q, _ ));
      
      apd_ab_boot(i, q) = sum(boot_apd_ab * wab_apd(q, _ ));
      apd_cd_boot(i, q) = sum(boot_apd_cd * wcd_apd(q, _ ));
    }
  }
  
  // ============================================================
  // COMPARE DECILES OF DISTRIBUTIONS OF ALL PAIRWISE DIFFERENCES
  // ============================================================
  
  // Compute distributions of all pairwise differences
  NumericVector apd_ab = outermin_cpp(a, b);
  NumericVector apd_cd = outermin_cpp(c, d);
  
  // Preallocate storage for deciles
  NumericVector apd_ab_hd(9); 
  NumericVector apd_cd_hd(9);
  //NumericVector apd_diff_hd(9);
  
  apd_ab = apd_ab.sort();
  apd_cd = apd_cd.sort();
  
  // Calculate quantiles
  for(int q = 0; q < 9; q++) {
    apd_ab_hd(q) = sum(apd_ab * wab_apd(q, _ ));
    apd_cd_hd(q) = sum(apd_cd * wcd_apd(q, _ ));
  }
  
  // Return results
  List res;
  res["ahd"] = ahd;
  res["bhd"] = bhd;
  res["chd"] = chd;
  res["dhd"] = dhd;
  res["boot_ahd"] = boot_ahd;
  res["boot_bhd"] = boot_bhd;
  res["boot_chd"] = boot_chd;
  res["boot_dhd"] = boot_dhd;
  res["apd_ab"] = apd_ab;
  res["apd_cd"] = apd_cd;
  res["apd_ab_hd"] = apd_ab_hd;
  res["apd_cd_hd"] = apd_cd_hd;
  res["apd_ab_boot"] = apd_ab_boot;
  res["apd_cd_boot"] = apd_cd_boot;
  return res;
}
