#include "SharedFunctions.h"
using namespace Rcpp;

/*
 * This function generates all possible combinations of a vector x of length l and puts them in a matrix
 * It uses the R function combn to produce the results.
 * Tested: 12/16/20
 * Confirmed: 11/23/22
 */
// [[Rcpp::export]]
NumericMatrix combn_cpp(NumericVector x,size_t l){
  size_t xlength = x.length();
  NumericMatrix result;
  
  if (l > x.length()){
    stop("There aren't enough neighbors for the current value of l");
  }
  
  if (l==0){
    result = NumericMatrix(1,1);
    result(0,0) = NA_REAL;
  } else if (xlength==1 && l == 1){
    result = NumericMatrix(1,1);
    result(0,0) = x(0);
  } else if (l>=1 && xlength>1) {
    Function f("combn");
    result = f(Named("x")=x,_["m"]=l);
  }
  
  return result;
}

// Determines whether or not i is in vector x
// Tested: 11/23/22
// [[Rcpp::export]]
bool isMember(NumericVector x,const size_t &i){
  NumericVector::iterator it = x.begin();
  size_t j;
  
  for (;it!=x.end();++it){
    j = *it;
    if (i == j){
      return true;
    }
  }
  
  return false;
}


// PRINTING FUNCTIONS (DEBUGGING)

void printVecElements(NumericVector v,StringVector names,String opening,String closing){
  size_t ln = v.length();
  Rcout << opening.get_cstring();
  for (size_t i = 0;i<ln;++i) {
    Rcout << names(v(i));
    if (i < ln-1){
      Rcout << " ";
    }
  }
  Rcout << closing.get_cstring();
}


void printVecElementsNoNames(NumericVector v,String opening,
                             String closing,String sep){
  size_t l = v.length();
  Rcout << opening.get_cstring();
  for (size_t i = 0;i<l;++i) {
    Rcout << v(i);
    if (i<l-1){
      Rcout << sep.get_cstring();
    }
  }
  Rcout << closing.get_cstring();
}


void printMatrix(NumericMatrix m){
  size_t n = m.nrow();
  size_t p = m.ncol();
  String ending;
  for (size_t i = 0;i<n;++i){
    for (size_t j = 0;j<p;++j){
      if (j==p-1){
        ending = "\n";
      } else{
        ending = " ";
      }
      Rcout << m(i,j) << ending.get_cstring();
    }
  }
}

void makeNodeNames(int p,StringVector &node_names){
  for (int i=0;i<p;++i){
    String node("V");
    node += i;
    node_names.push_back(node);
  }
}

