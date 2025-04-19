//spartacous
/* numericx.c */

/*
A module that numerically solves x(a) = m_pacdr/T_pacdr(a) using Newton Raphson
following https://numericalmethodstutorials.readthedocs.io/en/latest/nr.html

Written by: Taewook Youn, 2022.
*/

#include "math.h"
#include "bessel.h"
#include "common.h"


//spartacous: numerically solve x using Newton Raphson

double xfunc( double x, double r_g, double alpha) {
  // Martin's rho-hat and p-hat
  return pow(x / alpha, 3.) - 1. - (r_g/8.)*pow(x, 3.)*bessk(3, x);
}

double dxfunc( double x, double r_g, double alpha) {
  return 3.*pow(x, 2.)/pow(alpha, 3.) + (r_g/8.)*pow(x, 3.)*bessk(2, x);
}

double ddxfunc( double x, double r_g, double alpha) {
  return 6.*x/pow(alpha, 3.) + (r_g*x/8.)*( x*bessk(0, x) - (x*x-2.)*bessk(1, x) );
}

double find_x( double x_g, double r_g, double alpha)
{
  double x0,x1,converge,error;
  double _error_tolerance_ = 1.e-3;
  int i=0;

  converge = (xfunc(x0,r_g,alpha)*ddxfunc(x0,r_g,alpha))/(dxfunc(x0,r_g,alpha)*dxfunc(x0,r_g,alpha));

  if(converge > 1.) {
    // printf("converge = %.4e\n", converge);// NOTE: for debugging purposes
    return x0;
  }

  x0 = x_g; //initial guess of x

  do{
          x1=x0-(xfunc(x0,r_g,alpha)/dxfunc(x0,r_g,alpha));
          if(xfunc(x1,r_g,alpha)==0.){
                  break;
          }
          error=fabs((x1-x0)/x0);
          x0=x1;
          i++;

          // printf("i=%i: x0 = %.4e, x1 = %.4e, error=%.4e\n", i,x0,x1,error);// NOTE: for debugging purposes

  }while(error > _error_tolerance_);

  return x1;
}
//spartacous
