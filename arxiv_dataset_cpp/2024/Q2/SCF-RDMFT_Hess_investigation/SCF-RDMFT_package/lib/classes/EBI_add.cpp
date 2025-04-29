#include <stdio.h>
#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <tuple>
#include <iostream>

#include "1RDM_class.hpp"
#include "Functional_class.hpp"
#include "../tools.hpp" 
using namespace std;
using namespace Eigen;

const double SQRT_PI = 1.772453850905516027;

/* Inverse of the Error function */
double erfinv(const double x){
    //Correction for numerical errors/limits
    if (x<=-1.){
        return -INFINITY;
    }
    if(x>=1.){
        return INFINITY;
    }

    //from http://www.mimirgames.com/articles/programming/approximations-of-the-inverse-error-function/
    // Error of 1e-15 according to ref, tested of 1e-12 compared with Mathematica.
    double r;
    double a[] = {0.886226899, -1.645349621, 0.914624893, -0.140543331};
    double b[] = {1, -2.118377725, 1.442710462, -0.329097515, 0.012229801};
    double c[] = {-1.970840454, -1.62490649, 3.429567803, 1.641345311};
    double d[] = {1, 3.543889200, 1.637067800};
    double z = sign(x) * x;
    if(z<0.7){
        double x2 = z * z;
        r = z * (((a[3] * x2 + a[2]) * x2 + a[1]) * x2 + a[0]);
        r /= (((b[4] * x2 + b[3]) * x2 + b[2]) * x2 + b[1])* x2 + b[0];
    } else {
     double y = sqrt( -log((1 - z)/2));
     r = (((c[3] * y + c[2]) * y + c[1]) * y + c[0]);
     r /= ((d[2] * y + d[1]) * y + d[0]);
   }
   r = r * sign(x);
   z = z * sign(x);
   r -= (erf(r) - z)/(2/sqrt(M_PI) *exp(-r * r));
   r -= (erf(r) - z)/(2/sqrt(M_PI) *exp(-r * r)); //Comment out if you want single refinement
   return r;
}
double arcerf(const double x){ //alias for erfinv
    return erfinv(x);
}
/* Derivative of the Inverse Error funtion */
double derfinv(const double x){
    return SQRT_PI/2.*exp(pow(erfinv(x),2));
}

/* Derivative of the Error function */
double derf(const double x){
    if (abs(x)>DBL_MAX/2.){ //test x=inf with Ofast
        return 0.;
    }
    return 2./SQRT_PI*exp(-pow(x,2));
}
/* 2nd derivative of the Error function */
double dderf(const double x){
    if (abs(x)>DBL_MAX/2.){ //test x=inf with Ofast
        return 0.;
    }
    return -4./SQRT_PI*x*exp(-pow(x,2));
}
/* Auxiliary functions for solve_mu_aux */
tuple<double,double,double> compute_N (RDM1* gamma){
    double N =0; double dN =0; double ddN =0;
    for (int i=0;i<gamma->size();i++){
        double erf_i  = erf(gamma->x(i)+gamma->mu(0))+1.;
        double derf_i = derf(gamma->x(i)+gamma->mu(0));
        double dderf_i= dderf(gamma->x(i)+gamma->mu(0));  
        N  += pow(erf_i,2.)/2.;
        dN += erf_i*derf_i;
        ddN+= pow(derf_i,2.) + erf_i*dderf_i;
    }
    return make_tuple(N,dN,ddN);
}
double ne_cond (RDM1* gamma, double N){
    return pow(N - gamma->n_elec,2);
}
double dne_cond (RDM1* gamma, double N, double dN){
    return 2*dN*(N - gamma->n_elec);
}
double ddne_cond (RDM1* gamma, double N, double dN, double ddN){
    return 2*ddN*(N - gamma->n_elec) + 2*pow(dN,2);
}
/* Compute the value of mu (shared paramerter of EBI representation) from x */
void solve_mu_aux(RDM1* gamma){
    //from Supplementary material of YaoFang2022 (Algo2. Newton's method in 1D)
    gamma->mu(0) = 0;
    double F, dF, ddF; dF = 1; int k=0;
    while( (abs(dF)>1e-10) && (k<200) ){
        auto N = compute_N(gamma); k++;
        dF = dne_cond(gamma,get<0>(N),get<1>(N)); ddF = ddne_cond(gamma,get<0>(N),get<1>(N),get<2>(N));
        double ratio = abs(dF/ddF);
        if ( ratio>1.){
            gamma->mu(0) -= sign( get<0>(N)-gamma->n_elec ); //To prevant to large step
        }
        else{
            gamma->mu(0) -= sign( get<0>(N)-gamma->n_elec )*ratio;
        }  
    }
}

/* Auxiliary functions for solve_mu_subs_aux */
tuple<double,double,double> compute_N_subs (RDM1* gamma, vector<int> omega, int g){
    double N =0; double dN =0; double ddN =0;
    //TO DO
    return make_tuple(N,dN,ddN);
}
double ne_cond_subs (double N){
    return pow(N - 2.,2);
}
double dne_cond_subs (double N, double dN){
    return 2*dN*(N - 2.);
}
double ddne_cond_subs (double N, double dN, double ddN){
    return 2*ddN*(N - 2.) + 2*pow(dN,2);
}

/* Compute the value of mu (shared paramerter of EBI representation) for each subspace */
void solve_mu_subs_aux(RDM1* gamma){
    for (int i=0;i<gamma->omega.size();i++){
        vector<int> omega = gamma->omega[i];
        gamma->mu(i) = 0;
        double F, dF, ddF; dF = 1; int k=0;
        while( (abs(dF)>1e-10) && (k<200) ){
            auto N = compute_N_subs(gamma,omega,i); k++;
            dF = dne_cond_subs(get<0>(N),get<1>(N)); ddF = ddne_cond_subs(get<0>(N),get<1>(N),get<2>(N));
            double ratio = abs(dF/ddF);
            if( (abs(dF)<1e-10)&& (abs(ddF)<1e-10) ){
                break;
            }
            if ( ratio>1.){
                gamma->mu(i) -= sign( get<0>(N)-2. ); //To prevant to large step
            }
            else{
                gamma->mu(i) -= sign( get<0>(N)-2.)*ratio;
            }  
        }
    }

}