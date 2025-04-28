// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]



using namespace Rcpp;

arma::mat Pdist(arma::mat A, arma::mat B) {
  
  arma::colvec An =  sum(square(A),1);
  arma::colvec Bn =  sum(square(B),1);
  
  arma::mat C = -2 * (A * B.t());
  C.each_col() += An;
  C.each_row() += Bn.t();
  
  return C;
}

// [[Rcpp::export]]

NumericVector edfkmeans(arma::mat X, arma::mat M, arma::mat MU, IntegerVector asg, IntegerVector ns, double SS, int nclust, int n, int d, NumericVector sig, int nsig){
  arma::mat ds = Pdist(X, M);
  double a, b, c;
  double del1, del2, del;
  NumericVector edfs(nsig);
  double denom = pow(2*3.1416, 0.5);
  double jump;
  double offset;
  for(int i=0; i<n; i++){
    for(int j=0; j<d; j++){
      for(int cl=0; cl<nclust; cl++){
        if(cl!=asg[i]){
          a = 1.0-pow((ns[asg[i]]-1.0)/ns[asg[i]],2);
          b = 2.0*(X(i,j)-M(cl,j)-(X(i,j)-M(asg[i],j))*(ns[asg[i]]-1.0)/ns[asg[i]]);
          c = ds(i,cl)-ds(i,asg[i]);
          if((b*b-4*a*c)>=0){
            del1 = (-b+pow(b*b-4*a*c,0.5))/2.0/a;
            del2 = (-b-pow(b*b-4*a*c,0.5))/2.0/a;
            if(fabs(del1)<fabs(del2)) del = del1;
            else del = del2;
            if(del<0){
              jump = M(asg[i],j)-M(cl,j)*ns[cl]/(ns[cl]+1.0)-X(i,j)/(ns[cl]+1.0)+del*(ns[cl]-ns[asg[i]]+1.0)/(ns[asg[i]]*(ns[cl]+1.0));
              offset = pow(X(i,j)+del-MU(i,j),2);
              for(int ss=0; ss<nsig; ss++){
                edfs[ss] += jump*exp(-offset/sig[ss]/sig[ss]/2.0)/sig[ss]/denom;
              }
            }
            else{
              jump = -(M(asg[i],j)-M(cl,j)*ns[cl]/(ns[cl]+1.0)-X(i,j)/(ns[cl]+1.0)+del*(ns[cl]-ns[asg[i]]+1.0)/(ns[asg[i]]*(ns[cl]+1.0)));
              offset = pow(X(i,j)+del-MU(i,j),2);
              for(int ss=0; ss<nsig; ss++){
                edfs[ss] += jump*exp(-offset/sig[ss]/sig[ss]/2.0)/sig[ss]/denom;
              }
            }
          }
        }
      }
    }
  }
  for(int i=0; i<nsig; i++){
    edfs[i] += nclust*d;
  }
  return edfs;
}


// [[Rcpp::export]]

NumericVector edfkmeans_all(arma::mat X, arma::mat M, arma::mat MU, IntegerVector asg, IntegerVector ns, double SS, int nclust, int n, int d, NumericVector sig, int nsig){
  arma::mat ds = Pdist(X, M);
  double a, b, c;
  double del1, del2, del;
  NumericVector edfs(nsig);
  double denom = pow(2*3.1416, 0.5);
  double jump;
  double offset;
  for(int i=0; i<n; i++){
    for(int j=0; j<d; j++){
      for(int cl=0; cl<nclust; cl++){
        if(cl!=asg[i]){
          a = 1.0-pow((ns[asg[i]]-1.0)/ns[asg[i]],2);
          b = 2.0*(X(i,j)-M(cl,j)-(X(i,j)-M(asg[i],j))*(ns[asg[i]]-1.0)/ns[asg[i]]);
          c = ds(i,cl)-ds(i,asg[i]);
          if((b*b-4*a*c)>=0){
            del1 = (-b+pow(b*b-4*a*c,0.5))/2.0/a;
            del2 = (-b-pow(b*b-4*a*c,0.5))/2.0/a;
            if(fabs(del1)<fabs(del2)) del = del1;
            else del = del2;
            if(del<0){
              jump = M(asg[i],j)-M(cl,j)*ns[cl]/(ns[cl]+1.0)-X(i,j)/(ns[cl]+1.0)+del*(ns[cl]-ns[asg[i]]+1.0)/(ns[asg[i]]*(ns[cl]+1.0));
              for(int ss=0; ss<nsig; ss++){
                offset = pow(X(i,j)+del-MU(i,ss*d+j),2);
                edfs[ss] += jump*exp(-offset/sig[ss]/sig[ss]/2.0)/sig[ss]/denom;
              }
            }
            else{
              jump = -(M(asg[i],j)-M(cl,j)*ns[cl]/(ns[cl]+1.0)-X(i,j)/(ns[cl]+1.0)+del*(ns[cl]-ns[asg[i]]+1.0)/(ns[asg[i]]*(ns[cl]+1.0)));
              for(int ss=0; ss<nsig; ss++){
                offset = pow(X(i,j)+del-MU(i,ss*d+j),2);
                edfs[ss] += jump*exp(-offset/sig[ss]/sig[ss]/2.0)/sig[ss]/denom;
              }
            }
          }
        }
      }
    }
  }
  for(int i=0; i<nsig; i++){
    edfs[i] += nclust*d;
  }
  return edfs;
}


/*

// [[Rcpp::export]]

NumericMatrix jumps(arma::mat X, arma::mat M, arma::mat MU, IntegerVector asg, IntegerVector ns, double SS, int nclust, int n, int d){
arma::mat ds = Pdist(X, M);
double a, b, c;
double del1, del2, del;
NumericMatrix J(n, 2*d*nclust);
for(int i=0; i<n; i++){
for(int j=0; j<d; j++){
for(int cl=0; cl<nclust; cl++){
if(cl==asg[i]){
J(i,)
}
if(cl!=asg[i]){
a = 1.0-pow((ns[asg[i]]-1.0)/ns[asg[i]],2);
b = 2.0*(X(i,j)-M(cl,j)-(X(i,j)-M(asg[i],j))*(ns[asg[i]]-1.0)/ns[asg[i]]);
c = ds(i,cl)-ds(i,asg[i]);
if((b*b-4*a*c)>=0){
del1 = (-b+pow(b*b-4*a*c,0.5))/2.0/a;
del2 = (-b-pow(b*b-4*a*c,0.5))/2.0/a;
if(fabs(del1)<fabs(del2)) del = del1;
else del = del2;
if(del<0){
jump = M(asg[i],j)-M(cl,j)*ns[cl]/(ns[cl]+1.0)-X(i,j)/(ns[cl]+1.0)+del*(ns[cl]-ns[asg[i]]+1.0)/(ns[asg[i]]*(ns[cl]+1.0));
offset = pow(X(i,j)+del-MU(i,j),2);
for(int ss=0; ss<nsig; ss++){
edfs[ss] += jump*exp(-offset/sig[ss]/sig[ss]/2.0)/sig[ss]/denom;
}
}
else{
jump = -(M(asg[i],j)-M(cl,j)*ns[cl]/(ns[cl]+1.0)-X(i,j)/(ns[cl]+1.0)+del*(ns[cl]-ns[asg[i]]+1.0)/(ns[asg[i]]*(ns[cl]+1.0)));
offset = pow(X(i,j)+del-MU(i,j),2);
for(int ss=0; ss<nsig; ss++){
edfs[ss] += jump*exp(-offset/sig[ss]/sig[ss]/2.0)/sig[ss]/denom;
}
}
}
}
}
}
}
for(int i=0; i<nsig; i++){
edfs[i] += nclust*d;
}
return edfs;
}

*/

// [[Rcpp::export]]

NumericVector k_reg(NumericVector x, NumericVector y, double h, int n, int ord){ // x must be sorted in increasing order
  NumericMatrix L(ord + 1, n);
  NumericMatrix R(ord + 1, n);
  NumericMatrix Ly(ord + 1, n);
  NumericMatrix Ry(ord + 1, n);
  NumericVector coefs(ord + 1);
  coefs[0] = coefs[ord] = 1;
  if(ord>1){
    double num = 1;
    for(int j=2; j<=ord; j++) num *= j;
    double denom1 = 1;
    double denom2 = num/ord;
    for(int i=2; i<=ord; i++){
      coefs[i-1] = num/denom1/denom2;
      denom1 *= i;
      denom2 /= (ord-i+1);
    }
  }
  for(int i=0; i<=ord; i++){
    L(i,0) = pow(-x[0], i);
    Ly(i,0) = pow(-x[0], i)*y[0];
  } 
  for(int i=1; i<n; i++){
    for(int j=0; j<=ord; j++){
      L(j,i) = pow(-x[i],j) + exp((x[i-1]-x[i])/h)*L(j,i-1);
      R(j,n-i-1) = exp((x[n-i-1]-x[n-i])/h)*(pow(x[n-i],j)+R(j,n-i));
      Ly(j,i) = pow(-x[i],j)*y[i] + exp((x[i-1]-x[i])/h)*Ly(j,i-1);
      Ry(j,n-i-1) = exp((x[n-i-1]-x[n-i])/h)*(pow(x[n-i],j)*y[n-i]+Ry(j,n-i));
    }
  }
  NumericVector numerator(n);
  NumericVector denominator(n);
  for(int i=0; i<n; i++){
    for(int j=0; j<=ord; j++){
      denominator[i] += coefs[j]*(pow(x[i]+ord*h, ord-j)*L(j,i)+pow(ord*h-x[i],ord-j)*R(j,i));
      numerator[i] += coefs[j]*(pow(x[i]+ord*h, ord-j)*Ly(j,i)+pow(ord*h-x[i],ord-j)*Ry(j,i));
    }
  }
  NumericVector output(n);
  for(int i=0; i<n; i++) output[i] = numerator[i]/denominator[i];
  return output;
}



// [[Rcpp::export]]

NumericVector k_reg_loo(NumericVector x, NumericVector y, double h, int n, int ord){ // x must be sorted in increasing order
  NumericMatrix L(ord + 1, n);
  NumericMatrix R(ord + 1, n);
  NumericMatrix Ly(ord + 1, n);
  NumericMatrix Ry(ord + 1, n);
  NumericVector coefs(ord + 1);
  coefs[0] = coefs[ord] = 1;
  if(ord>1){
    double num = 1;
    for(int j=2; j<=ord; j++) num *= j;
    double denom1 = 1;
    double denom2 = num/ord;
    for(int i=2; i<=ord; i++){
      coefs[i-1] = num/denom1/denom2;
      denom1 *= i;
      denom2 /= (ord-i+1);
    }
  }
  for(int i=0; i<=ord; i++){
    L(i,0) = pow(-x[0], i);
    Ly(i,0) = pow(-x[0], i)*y[0];
  } 
  for(int i=1; i<n; i++){
    for(int j=0; j<=ord; j++){
      L(j,i) = pow(-x[i],j) + exp((x[i-1]-x[i])/h)*L(j,i-1);
      R(j,n-i-1) = exp((x[n-i-1]-x[n-i])/h)*(pow(x[n-i],j)+R(j,n-i));
      Ly(j,i) = pow(-x[i],j)*y[i] + exp((x[i-1]-x[i])/h)*Ly(j,i-1);
      Ry(j,n-i-1) = exp((x[n-i-1]-x[n-i])/h)*(pow(x[n-i],j)*y[n-i]+Ry(j,n-i));
    }
  }
  NumericVector numerator(n);
  NumericVector denominator(n);
  for(int i=0; i<n; i++){
    for(int j=0; j<=ord; j++){
      denominator[i] += coefs[j]*(pow(x[i]+ord*h, ord-j)*L(j,i)+pow(ord*h-x[i],ord-j)*R(j,i));
      numerator[i] += coefs[j]*(pow(x[i]+ord*h, ord-j)*Ly(j,i)+pow(ord*h-x[i],ord-j)*Ry(j,i));
    }
  }
  NumericVector output(n);
  for(int i=0; i<n; i++) output[i] = (numerator[i]-y[i])/(denominator[i]-1);
  return output;
}

