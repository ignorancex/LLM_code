#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// [[Rcpp::export]]

List smdh_list_div_cpp(arma::mat X, arma::mat V, arma::vec b, arma::mat mean, arma::vec ss, arma::vec ssss, IntegerVector t_init, double hmult, double C, int t_max, double alpha, arma::vec clusters, arma::vec scl, arma::vec mu0, int scale, int t_init0, double q, double r){
  int n = arma::size(X)[0];
  int d = arma::size(X)[1];
  arma::vec xx(d);
  arma::vec x(d);
  int dp = arma::size(V)[1];
  IntegerVector ts(dp);
  double nv;
  arma::vec p(1);
  double h;
  double db;
  int index;
  int t;
  double div;
  //double hpow = .2; //1.0/(4.0+d);
  double denom;
  //Rprintf("%s \n", "tt");
  for(int tt = 0; tt < n; tt++){
    if(scale==1){
      t = tt + t_init0;
      if(t > t_max) t = t_max;
      mu0 = t/(t+1.0)*mu0 + X.row(tt).t()/(t+1.0);
      scl = t/(t+1.0)*scl + pow(X.row(tt).t(), 2)/(t+1.0);
      for(int j = 0; j < d; j++){
        div = scl[j] - mu0[j]*mu0[j];
        if(div > 0) xx[j] = (X(tt,j) - mu0[j])/pow(div, 0.5);
        else xx[j] = (X(tt,j) - mu0[j]);
      }
    }
    else{
      xx = X.row(tt).t();
    }
    index = 0;
    while(index < dp){
      clusters[tt] = index;
      ts[index] += 1;
      t = ts[index] + t_init[index];
      if(t > t_max) t = t_max;
      denom = pow(t+1.0,r);
      mean.col(index) = t/(t+1.0)*mean.col(index) + xx/(t+1.0);
      x = xx - mean.col(index);
      nv = norm(V.col(index), 2);
      V.col(index) /= nv;
      p = x.t()*V.col(index);
      ss[index] = ss[index]*t/(t+1.0) + p[0]*p[0]/(t+1.0);
      ssss[index] += arma::accu(pow(x, 2));
      h = pow(ss[index], .5)*hmult/pow(t+1.0, q);
      V.col(index) -= .3989*sqrt(d)*(b[index]-p[0])*exp(-pow(b[index]-p[0], 2)/2.0/h/h)*x/denom/h/h/h;
      db = -.3989*(b[index]-p[0])*exp(-pow(b[index]-p[0], 2)/2.0/h/h - log(denom) - 3*log(h));
      if(b[index] > (alpha*sqrt(ss[index]))) db += 2*C*(b[index]-alpha*sqrt(ss[index]))/denom;
      if(b[index] < (-alpha*sqrt(ss[index]))) db -= 2*C*(-alpha*sqrt(ss[index])-b[index])/denom;
      b[index] -= db;
      if(b[index] > (5*alpha*sqrt(ss[index]))) b[index] = 5*alpha*sqrt(ss[index]);
      if(b[index] < (-5*alpha*sqrt(ss[index]))) b[index] = -5*alpha*sqrt(ss[index]);
      p = x.t()*V.col(index)/norm(V.col(index), 2);
      if(p[0] < b[index]) index = 2*index+1;
      else index = 2*index+2;
    }
  }
  List ret(10);
  ret[0] = V;
  ret[1] = b;
  ret[2] = mean;
  ret[3] = ss;
  ret[4] = ssss;
  ret[5] = ts + t_init;
  ret[6] = clusters;
  ret[7] = scl;
  ret[8] = mu0;
  ret[9] = n + t_init0;
  return(ret);
}







// [[Rcpp::export]]

List smdh_list_div_pass_cpp(arma::mat X, arma::mat V, arma::vec b, arma::mat mean, arma::vec ssss, arma::vec ts, arma::vec scl, arma::vec mu0, int scale){
  int n = arma::size(X)[0];
  int d = arma::size(X)[1];
  arma::vec xx(d);
  arma::vec x(d);
  int dp = arma::size(V)[1];
  arma::vec p(1);
  int index;
  double div;
  arma::vec clusters(n);
  for(int tt = 0; tt < n; tt++){
    if(scale==1){
      for(int j = 0; j < d; j++){
        div = scl[j] - mu0[j]*mu0[j];
        if(div > 0) xx[j] = (X(tt,j) - mu0[j])/pow(div, 0.5);
        else xx[j] = (X(tt,j) - mu0[j]);
      }
    }
    else{
      xx = X.row(tt).t();
    }
    index = 0;
    while(index < dp){
      clusters[tt] = index;
      ts[index] += 1;
      x = xx - mean.col(index);
      p = x.t()*V.col(index);
      //if(prune == 1) ssss[index] += arma::accu(pow(x, 2));
      //else ssss[index] += arma::accu(pow(x, 2)/(mean2.col(index)-pow(mean.col(index), 2)));
      ssss[index] += arma::accu(pow(x, 2));
      if(p[0] < b[index]) index = 2*index+1;
      else index = 2*index+2;
    }
  }
  List ret(3);
  ret[0] = clusters;
  ret[1] = ssss;
  ret[2] = ts;
  return(ret);
}

