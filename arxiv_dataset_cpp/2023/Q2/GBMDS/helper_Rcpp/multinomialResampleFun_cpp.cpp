#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]

// Obtain resampling index using multinomial resampling
Rcpp::NumericVector multinomialResampleFun_cpp(const std::vector<double>& W) {
  int N = W.size();
  //std::vector<int> index(N);
  
  // Obtaining namespace of Matrix package
  // Environment pkg = Environment::namespace_env("base");
  
  // Picking up Matrix() function from Matrix package
  // Function f = pkg["sample.int"];
  
  // calling rnorm()
  Function f("sample.int");   
  
  return f(Named("n", N), Named("size", N), Named("replace", true), Named("prob", W));
  // return index;
}


/*** R

#rESSFun(log(c(0.2, 0.4, 0.3, 0.1)))
#rESSFun_cpp(log(c(0.2, 0.4, 0.3, 0.1)))

*/


/*** R

# set.seed(1115)
# multinomialResampleFun(c(0.2, 0.4, 0.3, 0.1))
# set.seed(1115)
# multinomialResampleFun_cpp(c(0.2, 0.4, 0.3, 0.1))

*/
