#include <RcppArmadillo.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include<omp.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]

using namespace arma;

// [[Rcpp::export]]
  
void QiupC(mat &Qtemp, const mat &YG, const mat &YG2, const mat &le, const colvec sigmaSq, const double alphaf) {
  int p = YG.n_rows;
  colvec repal(p);
  repal.fill(1/alphaf);
  uvec ind = regspace<uvec>(1,  1,  p-1);
  for(int j = 0; j < p; j++){
    if(j > 0){ind(j-1) = 0;}
    vec  Qmean = zeros<vec>(p);
    Qmean(j) = 1;
    mat temp = -(Qtemp.cols(ind)*YG.rows(ind) - le);
    //rowvec YGj = YG.row(j);
    temp *= (YG.row(j)).t(); 
    colvec Rtemp =  sum(temp, 1);
    Rtemp = Rtemp/ sigmaSq + Qmean/alphaf;  
    vec varlami  = 1/(sum(YG2.row(j))/sigmaSq + repal);
    vec meanlami = varlami % Rtemp; //elementwise multi
    Qtemp.col(j) = meanlami + randn<vec>(p) % sqrt(varlami); //rnorm(p, meanlami, sqrt(varlami));
    if(j > 0){ind(j-1) = j;}
    //cout<< j<< "text10"<<  endl;
  }
  
}


// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R

#fit <- Qiup(1, Qlistmat)
#Qtemp    <- matrix(Qlistmat[, 1], p, p)
#mean((fit - Qtemp)^2)
*/
