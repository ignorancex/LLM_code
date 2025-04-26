// [[Rcpp::depends(RcppArmadillo)]]

#include<RcppArmadillo.h>
using namespace arma;

vec fird(vec inner, int n, char loss) {
 vec d=ones(n);
 double u;
 switch(loss) {
  case 'd':
   for (int i=0; i<n; i++) {
    u=inner[i];
    if (u<-0.5) d[i]=1/(4*u*u);
   }
  break;
  case 'e': d=exp(inner/5)/5;
  break;
  case 'l':
   for (int i=0; i<n; i++) {
    u=exp(inner[i]/2);
    d[i]=u/(1+u)/2;
   }
 }
 return d;
}

vec secd(vec inner, int n, char loss) {
 vec d=zeros(n);
 double u;
 switch(loss) {
  case 'd':
   for (int i=0; i<n; i++) {
    u=inner[i];
    if (u<-0.5) d[i]=-1/(2*u*u*u);
   }
  break;
  case 'e': d=exp(inner/5)/25;
  break;
  case 'l':
   for (int i=0; i<n; i++) {
    u=exp(inner[i]/2);
    d[i]=u/(1+u)/(1+u)/4;
   }
 }
 return d;
}

mat update_alpha(mat A, mat K, mat P, mat W, mat ZB, mat wW, mat wWW, double nlrho, char loss) {

 int n=K.n_rows, p=K.n_cols, kminus=W.n_cols;
 double partial, secpartial, temp, diff, epsilon=0.0000001*p*kminus;
 mat oldA(p, kminus);
 vec u=sum(W%(K*A), 1);

 for (int iter=0; iter<100; iter++) {

  oldA=A;

  /* update alpha0 */
  for (int j=0; j<kminus; j++) {

   /* update alpha_0j */
   if (loss=='s') {

    partial=sum((1+u/2)%wW.col(j))+nlrho*A(0, j)+ZB(0, j);
    secpartial=sum(wWW.col(j))/2+nlrho;
    temp=partial/secpartial;
    A(0, j)-=temp;
    u=u-temp*W.col(j);

   } else {

    for (int i=0; i<10; i++) {
     partial=sum(fird(u, n, loss)%wW.col(j))+nlrho*A(0, j)+ZB(0, j);
     if (fabs(partial)<0.0000001) break;
     secpartial=sum(secd(u, n, loss)%wWW.col(j))+nlrho;
     if (fabs(secpartial)<0.001) secpartial=0.001;
     temp=partial/secpartial;
     A(0, j)-=temp;
     u=u-temp*W.col(j);
    } /* newton raphson for alpha_0j */

   }
  }

  /* update alpha */
  for (int j=0; j<kminus; j++) {
   for (int q=1; q<p; q++) {

    /* update alpha_qj */
    if (loss=='s') {

     partial=sum((1+u/2)%wW.col(j)%K.col(q))+sum(P.col(q)%(nlrho*A.col(j)+ZB.col(j)));
     secpartial=sum(wWW.col(j)%K.col(q)%K.col(q))/2+nlrho*P(q, q);
     temp=partial/secpartial;
     A(q, j)-=temp;
     u=u-temp*K.col(q)%W.col(j);

    } else {

     for (int i=0; i<10; i++) {
      partial=sum(fird(u, n, loss)%wW.col(j)%K.col(q))+sum(P.col(q)%(nlrho*A.col(j)+ZB.col(j)));
      if (fabs(partial)<0.0000001) break;
      secpartial=sum(secd(u, n, loss)%wWW.col(j)%K.col(q)%K.col(q))+nlrho*P(q, q);
      if (fabs(secpartial)<0.001) secpartial=0.001;
      temp=partial/secpartial;
      A(q, j)-=temp;
      u=u-temp*K.col(q)%W.col(j);
     } /* newton raphson for alpha_qj */

    }
   }
  }

  diff=sum(sum(pow(A-oldA, 2)));
  if (diff<epsilon) break;
 }

 return A;
}

vec update_gamma(vec gamma, mat K, mat WWK, vec ub, vec w) {

 int n=K.n_rows;
 vec oldgamma(n);
 double gamma_new, diff, epsilon=0.0000001*n;;

 for (int iter=0; iter<100; iter++) {

  oldgamma=gamma;

  /* update gamma */
  for (int l=0; l<n; l++) {
   gamma_new=gamma[l]-(sum(gamma%WWK.col(l))-ub[l])/(K(l, l+1)+1);
   if (gamma_new<0) {
    gamma[l]=0;
   } else if (gamma_new>w[l]) {
    gamma[l]=w[l];
   } else {
    gamma[l]=gamma_new;
   }
  }

 diff=sum(pow(gamma-oldgamma, 2));
 if (diff<epsilon) break;
 }

 return gamma;
}

double update_rho(double rho, double r1, double r2) {
 double r=r1/r2;
 if (r>100) {
  rho=2*rho;
 } else if (r<0.01) {
  rho=rho/2;
 }
 return rho;
}

// [[Rcpp::export]]
cube diffit_C(mat WWK, mat K, mat W, vec w, double cminus, vec lambda, char loss, double maxiter=100) {

 int n=K.n_rows, p=n+1, kminus=W.n_cols, m=lambda.size();
 mat A(p, kminus), ZB(p, kminus), wW(n, kminus), wWW(n, kminus), P(p, p);
 vec u(n);
 double nlambda;
 cube coef(p, kminus, m);

 for (int j=0; j<kminus; j++) {
  wW.col(j)=w%W.col(j);
  wWW.col(j)=wW.col(j)%W.col(j);
 }
 P.fill(0), A.fill(0), ZB.fill(0), u.fill(1);
 P(0, 0)=1;
 P.submat(1, 1, p-1, p-1)=K;
 K=join_rows(u, K);

 if (cminus>0) { /* fit for bent loss */

  mat B(p, kminus), Z(p, kminus), oldB(p, kminus), temp(p, kminus);
  vec b(n), ub(n), gamma(n);
  double nlrho, r1, r2, rho=1, epsilon=p*kminus*0.0000001;
  Z.fill(0), oldB.fill(0);

  /* iterate for lambda */
  for (int i=0; i<m; i++) {
 
   nlambda=n*lambda[i];

   /* ADMM algorithm start */
   for (int iter=0; iter<maxiter; iter++) {

    nlrho=nlambda+rho;

    /* update alpha */
    A=update_alpha(A, K, P, W, ZB, wW, wWW, nlrho, loss);
    u=sum(W%(K*A), 1);
    b=sum(W%(K*Z), 1);
    ub=(rho*u+b)/cminus;

    /* update beta */
    gamma=update_gamma(gamma, K, WWK, ub, w);
    temp.fill(0);
    for (int j=0; j<kminus; j++) {
     temp.submat(1, j, n, j)=gamma%W.col(j);
    }
    temp.row(0)=sum(temp);
    B=A-(cminus*temp-Z)/rho;

    /* check convergence */
    r1=0, r2=0;
    temp=A-B;
    for (int j=0; j<kminus; j++) {
     r1+=as_scalar(temp.col(j).t()*P*temp.col(j));
    }
    temp=B-oldB;
    for (int j=0; j<kminus; j++) {
     r2+=as_scalar(temp.col(j).t()*P*temp.col(j));
    }
    r2=rho*rho*r2;
    if ((r1<epsilon) & (r2<epsilon)) break;
    oldB=B;

    /* update z */
    ZB=Z-rho*B;
    Z=ZB+rho*A;

    /* update rho */
    rho=update_rho(rho, r1, r2);
   }

   coef.slice(i)=A;
  }

 } else { /* fit for common loss */

  /* iterate for lambda */
  for (int i=0; i<m; i++) {

   nlambda=n*lambda[i];
   A=update_alpha(A, K, P, W, ZB, wW, wWW, nlambda, loss);
   coef.slice(i)=A;

  }
 }

 return coef;
}