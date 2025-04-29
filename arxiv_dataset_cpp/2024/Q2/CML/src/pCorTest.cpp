#include "pCorTest.h"

// Here, we are treating C as the correlation matrix
// [[Rcpp::export]]
double getPartialCorrelation(arma::mat C,size_t i,size_t j,arma::uvec k){
  if (i == j){
    return(1.0);
  }
  
  double pc;
  size_t k_size = k.size();
  
  if (k_size==0){
    return C(i,j);
  } else if (k_size==1){
    pc = (C(i,j)-C(i,k(0))*C(j,k(0))) / 
      sqrt( (1-pow(C(i,k(0)),2))*(1-pow(C(j,k(0)),2)));
  } else {
    arma::uvec indices(k_size+2);
    indices(0) = i;
    indices(1) = j;
    for (size_t l=0;l<k_size;++l){
      indices(l+2) = k(l);
    }
    arma::mat Cinv = arma::pinv(C(indices,indices));
    pc = -Cinv(0, 1)/sqrt(Cinv(0, 0) * Cinv(1, 1));
  }
  
  return pc;
}

double logPart(double r){
  double result;
  if (r == 1){ // Perfect correlation means this value will approach infinity
    result = 1000;
  } else {
    result = log1p((2*r)/(1-r));
  }
  return result; // returns log_e(1 + (2*r/(1-r)))
}

// [[Rcpp::export]]
double fisherZ(double pc,size_t n,size_t k_size){
  return sqrt(n - k_size - 3) * 0.5 * logPart(pc);
}

// [[Rcpp::export]]
List condIndTest(arma::mat &C,const size_t &i,const size_t &j,
                 const arma::uvec &k,const size_t &n,
                 const double &signif_level){
  double pc = getPartialCorrelation(C,i,j,k);
  double statistic = fisherZ(pc,n,k.size());

  bool lower = statistic < 0;
  
  double cutoff = R::qnorm(1-signif_level/2,0.0,1.0,true,false);
  
  bool accept_H0 = std::abs(statistic) <= cutoff;
  double pval=2*R::pnorm(statistic,0.0,1.0,lower,false);
  // The null hypothesis is accepted (p-value large) => H_0: r = 0 
  // => Conditional independence
  return List::create(
    _["result"]=accept_H0,
    _["statistic"]=statistic,
    _["pval"]=pval
  );
  
}


// [[Rcpp::export]]
List condIndTestPop(NumericMatrix G,const size_t &i,const size_t &j,
                    const arma::uvec &k){
  Function my_dsep("my_dsep");
  NumericVector tmp = my_dsep(G,i,j,k);
  double pval = tmp[0];
  bool accept_H0 = static_cast<bool>(pval);  

  // The null hypothesis is accepted (p-value large) => H_0: r = 0 
  // => Conditional independence
  return List::create(
    _["result"]=accept_H0, 
    _["pval"]=pval
  );
}
