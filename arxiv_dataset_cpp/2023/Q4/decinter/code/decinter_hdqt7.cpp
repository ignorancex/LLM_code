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
  // Rcpp::NumericVector probs_names=Rcpp::round(probs*100., 6);
  // Rcpp::CharacterVector qs_names(np);
  // std::string _name, format;
  // char string[11], width[2], digits[2];
  // double probs_full, probs_trunc;
  
  for (size_t i=0; i<np; ++i) {
    // probs_full=static_cast<double>(probs_names[i]);
    // 
    // if(probs_full==100.) _name=std::string("100");
    // else if(probs_full==0.) _name=std::string("0");
    // else {
    //   int k=std::ceil(log10(probs_full));
    //   if(probs_full==std::floor(probs_full)) {
    //     sprintf(string, "%1$*2$.0f", probs_full, k);
    //     _name=string;
    //   }
    //   else for(size_t j=6; j>0; --j) {
    //     probs_trunc=(std::floor(probs_full*pow(10, j))/pow(10, j)); 
    //     if(probs_trunc==probs_full) {
    //       sprintf(width, "%1d", j+k+1);
    //       sprintf(digits, "%1d", j);
    //       format = std::string("%") + width +
    //         std::string(".") + digits + std::string("f");
    //       sprintf(string, format.c_str(), probs_trunc);
    //       _name=string;
    //       //_name=format;
    //     }
    //   }
    // }
    // 
    // qs_names[i] = _name + std::string("%");
    qs[i] = y[lo[i]];
    x_hi[i] = y[hi[i]];
    if ((index[i]>lo[i]) && (x_hi[i] != qs[i])) {
      double h;
      h = index[i]-lo[i];
      qs[i] = (1.-h)*qs[i] + h*x_hi[i];
    }
  }
  
  // qs.names()=qs_names;
  return qs;
}

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
List decinter_hdqt7_cpp(NumericMatrix x, int n, NumericMatrix w, int B, NumericVector probs){
  
  // Preallocate storage for deciles
  // HD
  NumericVector ahd(9);
  NumericVector bhd(9);
  NumericVector chd(9);
  NumericVector dhd(9);
  NumericVector Ahd(9);
  NumericVector Bhd(9);
  NumericVector ABhd(9);
  // QT7
  NumericVector aqt(9);
  NumericVector bqt(9);
  NumericVector cqt(9);
  NumericVector dqt(9);
  NumericVector Aqt(9);
  NumericVector Bqt(9);
  NumericVector ABqt(9);
  
  NumericVector a = x(_, 0); // A1B1
  NumericVector b = x(_, 1); // A1B2
  NumericVector c = x(_, 2); // A2B1
  NumericVector d = x(_, 3); // A2B2
  
  a = a.sort();
  b = b.sort();
  c = c.sort();
  d = d.sort();
  
  // Calculate quantiles: HD
  for(int q = 0; q < 9; q++) {
    ahd(q) = sum(a * w(q, _ ));
    bhd(q) = sum(b * w(q, _ ));
    chd(q) = sum(c * w(q, _ ));
    dhd(q) = sum(d * w(q, _ ));
  }
  
  ABhd = ahd - bhd - chd + dhd;
  Ahd = ahd + bhd - chd - dhd;
  Bhd = ahd - bhd + chd - dhd;

  // Calculate quantiles: QT7
  aqt = qt7_cpp(a, probs);
  bqt = qt7_cpp(b, probs);
  cqt = qt7_cpp(c, probs);
  dqt = qt7_cpp(d, probs);

  ABqt = aqt - bqt - cqt + dqt;
  Aqt = aqt + bqt - cqt - dqt;
  Bqt = aqt - bqt + cqt - dqt;
  
  // Preallocate storage for bootstrap results
  NumericMatrix bootABqt(B, 9);
  NumericMatrix bootAqt(B, 9);
  NumericMatrix bootBqt(B, 9);
  NumericMatrix bootABhd(B, 9);
  NumericMatrix bootAhd(B, 9);
  NumericMatrix bootBhd(B, 9);

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
    
    // Calculate quantiles: HD
    for(int q = 0; q < 9; q++) {
      double tmp_a = sum(boota * w(q, _ ));
      double tmp_b = sum(bootb * w(q, _ ));
      double tmp_c = sum(bootc * w(q, _ ));
      double tmp_d = sum(bootd * w(q, _ ));
      bootABhd(i, q) = tmp_a - tmp_b - tmp_c + tmp_d;
      bootAhd(i, q) = tmp_a + tmp_b - tmp_c - tmp_d;
      bootBhd(i, q) = tmp_a - tmp_b + tmp_c - tmp_d;
    }

    // Calculate quantiles: QT7
      NumericVector tmp_a = qt7_cpp(boota, probs);
      NumericVector tmp_b = qt7_cpp(bootb, probs);
      NumericVector tmp_c = qt7_cpp(bootc, probs);
      NumericVector tmp_d = qt7_cpp(bootd, probs);
      bootABqt(i, _) = tmp_a - tmp_b - tmp_c + tmp_d;
      bootAqt(i, _) = tmp_a + tmp_b - tmp_c - tmp_d;
      bootBqt(i, _) = tmp_a - tmp_b + tmp_c - tmp_d;

  }
  
  // Return results
  List res;
  // QT7
  res["ABqt"] = ABqt;
  res["Aqt"] = Aqt;
  res["Bqt"] = Bqt;
  res["bootABqt"] = bootABqt;
  res["bootAqt"] = bootAqt;
  res["bootBqt"] = bootBqt;
  // HD
  res["ABhd"] = ABhd;
  res["Ahd"] = Ahd;
  res["Bhd"] = Bhd;
  res["bootABhd"] = bootABhd;
  res["bootAhd"] = bootAhd;
  res["bootBhd"] = bootBhd;
  return res;
}
