//spartacous
/* sigmoid.c */

/*
A module that computes analytic sigmoids, useful to interpolate between the two different scalings of x(a) = m_pacdr/T_pacdr(a), and their derivatives. They are written in terms of alpha = a/a_tr.

Currently only two sigmoids are implemented: Tanh-based, and an asymmetric exponentials-based. The asymmetric one fits better, but its parameters have a slightly larger dependence on r_g.

These are analytical methods, and they are accurate down to ~O(1%) level, according to my tests.

The purpose of these methods is to serve as a good initial guess of x(a) for the full numerical solution, found in numericx.c

Written by: Manuel A. Buen-Abad, 2022.
*/

#include "math.h"



double tanh_sigmoid( double Delta,
                     double alpha0,
                     double alpha)
/* Tanh sigmoid. It fits rather well the step-like evolution of x(a)/a for SPARTACOUS. It is less good than the asymmetric sigmoid, but it has less spread in the bestfit parameters (Delta, alpha0) as a function of rg. */
{

  double ans;

  ans = tanh(Delta*log10(alpha/alpha0));

  return ans;
}



double d_tanh_sigmoid( double Delta,
                       double alpha0,
                       double alpha)
/* The derivative of the tanh sigmoid w.r.t. alpha. */
{

  double ans;

  ans = Delta*pow(cosh(Delta*log(alpha/alpha0)/
  log(10.)), -2.)/(alpha*log(10.));

  return ans;
}



double asym_sigmoid( double Delta,
                     double alpha0,
                     double powL,
                     double powR,
                     double alpha)
/* A particularly useful, asymmetric sigmoid. It fits rather well the step-like evolution of x(a)/a for SPARTACOUS. It is better than the Tanh sigmoid, but it has a larger spread in the bestfit parameters (Delta, alpha0, powL, powR) as a function of rg. */
{

  double left, right, ans;

  left = pow(1. + pow(alpha/alpha0, -Delta), -powL);
  right = pow(1. + pow(alpha/alpha0, Delta), -powR);

  ans = (left - right);

  return ans;
}



double d_asym_sigmoid( double Delta,
                       double alpha0,
                       double powL,
                       double powR,
                       double alpha)
/* The derivative of the asymmetric sigmoid w.r.t. alpha. */
{

  double pref, left, right, num, ans;

  pref = Delta/alpha/(1.+pow(alpha/alpha0, Delta));
  left = pow(1. + pow(alpha/alpha0, -Delta), -powL);
  right = pow(1. + pow(alpha/alpha0, Delta), -powR);

  num = powL*left + powR*right*pow(alpha/alpha0, Delta);

  ans = pref*num;

  return ans;
}



double x_over_alpha( double rg,
                     double sigm)

/* The step-like evolution of x(a)/a for SPARTACOUS. */

{

  double A, B, ans;

  A = (pow((1. + rg), 1./3.) - 1.)/2.;
  B = (pow((1. + rg), 1./3.) + 1.)/2.;

  ans = A*(-sigm) + B;

  return ans;
}



double dx_dalpha( double rg,
                  double dsigm)

/* The derivative of x(a)/a w.r.t. alpha. */

{

  double A, ans;

  A = (pow((1. + rg), 1./3.) - 1.)/2.;

  ans = -A*dsigm;

  return ans;
}
//spartacous
