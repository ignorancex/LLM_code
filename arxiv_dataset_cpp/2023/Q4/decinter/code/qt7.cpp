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
  Rcpp::NumericVector probs_names=Rcpp::round(probs*100., 6);
  Rcpp::CharacterVector qs_names(np);
  std::string _name, format;
  char string[11], width[2], digits[2];
  double probs_full, probs_trunc;
  
  for (size_t i=0; i<np; ++i) {
    probs_full=static_cast<double>(probs_names[i]);
    
    if(probs_full==100.) _name=std::string("100");
    else if(probs_full==0.) _name=std::string("0");
    else {
      int k=std::ceil(log10(probs_full));
      if(probs_full==std::floor(probs_full)) {
        sprintf(string, "%1$*2$.0f", probs_full, k);
        _name=string;
      }
      else for(size_t j=6; j>0; --j) {
        probs_trunc=(std::floor(probs_full*pow(10, j))/pow(10, j)); 
        if(probs_trunc==probs_full) {
          sprintf(width, "%1d", j+k+1);
          sprintf(digits, "%1d", j);
          format = std::string("%") + width +
            std::string(".") + digits + std::string("f");
          sprintf(string, format.c_str(), probs_trunc);
          _name=string;
          //_name=format;
        }
      }
    }
    
    qs_names[i] = _name + std::string("%");
    qs[i] = y[lo[i]];
    x_hi[i] = y[hi[i]];
    if ((index[i]>lo[i]) && (x_hi[i] != qs[i])) {
      double h;
      h = index[i]-lo[i];
      qs[i] = (1.-h)*qs[i] + h*x_hi[i];
    }
  }
  
  qs.names()=qs_names;
  return qs;
}