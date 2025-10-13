#include "def.hpp"
#include "tools.hpp"
#include "amplitudelib.hpp"
#include "fourier.h"
#include <cmath>
#include <gsl/gsl_sf_result.h>
#include <gsl/gsl_sf_bessel.h>
#include <fstream>
#include <iostream>

using namespace std;



double besselFunction(double nu, double x)
{   
    gsl_sf_result result;

    int gsl_status1 = gsl_sf_bessel_Knu_e(nu, x, &result);
    
    if (gsl_status1 != GSL_SUCCESS) 
    {
        return 0.0; 
    }
  
    return result.val;

}

// Function to be integrated without the dipole

double F(double m, double kp, double k, double l, double x, double qp)
{ 
 
    double z = kp / qp;  
    
    double abr = SQR(k) + SQR(l) - 2.*k*l*cos(x);

    //squares  
    double num1 = abr;  
    double num2 = SQR(k);
    double den1 = SQR(m)*SQR(m) + 2.*SQR(m)*abr + SQR(abr) ;
    double den2 = SQR(m)*SQR(m) + SQR(k)*SQR(k) + 2.*SQR(m)*SQR(k) ;
    
    
    //cross term
     double num3 = 2*(SQR(k)-l*k*cos(x));
     double den3 = SQR(m)*SQR(m) + SQR(m)*SQR(k) + SQR(m)*abr + abr*SQR(k);
     
     //first parenthesis
     double par1 = (1. / den1)+ (1./den2)- (2./den3);
     
     //second parenthesis 
     double par2 = (num1 / den1)+ (num2/den2)- (num3/den3);
     
     //function
     double fun =  (2*z*SQR(m) * par1) + (2*z*(SQR(z) + SQR(1.-z))*par2);  
     
     
     return l*fun;  //l from the 2D integral
}


//  Collinear approximation expression

double col_ap(double m, double kp, double k, double l, double qp)  
{
   double z = kp / qp;
   
   double num = 4*SQR(m)*SQR(k) + 2*(SQR(z) + SQR(1-z))*(SQR(m)*SQR(m)+SQR(k)*SQR(k));
   double den = SQR(k*k + m*m)*SQR(k*k + m*m);

   return l*l*l*(z*num/den);

}

// Photon flux including 0n emission probability

double modflux(double qp, double b, int Z, double S, double ss, double M, double A)
{
     double alpha =1./137.;
     double w = qp /sqrt(2.);
     
     //0n neutron emission
     double p = exp(-S/(b*b));
     double gamma= (ss/2.)/(M/A);
     //flux
     double flux = (1./qp)*(SQR(Z)*alpha * SQR(w) / (SQR(M_PI) * SQR(gamma)) ) * SQR(besselFunction(1,w*b*5.068/gamma)); 
     
     return 2.*M_PI*b*p*flux;    //2pi and b from the 2D integral 

}
     

double integrand(double* vec, size_t dim, void* p)
{
    parameters *par = ((parameters*)p);   

    double x =  vec[0];
    double l =  vec[1];
    double qp =  vec[2];
    double b =  vec[3];    
    
    //Dipole amplitudes
    double D = par -> dipole -> S_k( l, par -> xbj, FUNDAMENTAL, 1.0);
    double photflux = modflux(qp, b, par -> Z, par -> S, par -> ss, par ->M, par->A);
    double function = F(par -> mc, par -> kp, par -> kD / par -> zf, l, x, qp);
    
    double fun2 = col_ap(par -> mc, par -> kp, par -> kD/ par -> zf, l, qp);

    //cout << "D: " << D << ", photflux: " << photflux << ", function: " << function << endl;


     return D*photflux*function;
}



