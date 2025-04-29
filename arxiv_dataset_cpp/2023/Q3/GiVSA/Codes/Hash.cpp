// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
// #include <unordered_map>
// #include <string>

using namespace Rcpp;

// [[Rcpp::export]]
static List mcmc(NumericVector start, Function run, int n, int bknot) {
  List samples(n-bknot);
  samples[0] = start;
  NumericVector foo = start;
  for(int i = 1; i < n; i++) {
    foo = run(foo, i/float(n));
    if (i >= bknot)
    samples[i-bknot] = foo;
  }
  return(samples);
}
// static long double hash_test(NumericVector model, Function posterior, StringVector temp_key) {
//   static std::unordered_map<std::string, long double > hashtable;
//   std::string key = Rcpp::as< std::string > (temp_key[0]);
//   // Rcout<<key<<std::endl;
//   long double temp;
//   if(hashtable.find(key) == hashtable.end()){
//     SEXP foo = posterior(model);
//     temp =  *REAL(foo);
//     hashtable[key] = temp;
//   }
//   else {
//     // Rcout<<"recycling is good"<<std::endl;
//     temp = hashtable[key];
//   }
//   return temp;
// }