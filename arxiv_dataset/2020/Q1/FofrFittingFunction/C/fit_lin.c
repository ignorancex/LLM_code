#include <math.h>
#include <stdio.h>
#include <stdlib.h>

const double fR0_fid = 1e-5;
double param[] = {
  3.1 , 2.34466 , -1.86362 , 34.4951 , 
  28.8637 , -13.1302 , 0.146542 , -0.0103024 , 
  -0.149446 , 1.62807 , 0.71291 , -1.41003};

//================================================================
// 
// This is a fitting formula for the enhancement:
//
// pofk_enhancement = P(k,z) / P_LCDM(k,z) 
//
// in Hu-Sawikcy f(R) gravity with n =1
//
// [k] is assumed to in units of [h/Mpc]
// 
// NB: If requesting a value outside of the regime we have
// fitted for we return the closest value without
// any warning
//
//================================================================

double pofk_enhancement_linear(double _z, double _fR0, double _k){
  double z = _z, fR0 = _fR0, k = _k;

  // The ranges we performed the fit for
  const double kmin   = 0.001;
  const double kmax   = 100.0;
  const double zmin   = 0.0;
  const double zmax   = 5.0;
  const double fR0min = 1e-7;
  const double fR0max = 1e-3;

  // Bound checks (by design ratio -> 1 as k -> 0 so no need to impose lower limit)
  if(fR0 < fR0min) fR0 = fR0min;
  if(fR0 > fR0max) fR0 = fR0max;
  
  // Transform to effective wave-number
  k = k * sqrt(fR0 / fR0_fid);
 
  // Bounds check
  if(k > kmax)     k   = kmax;
  if(z < zmin)     z   = zmin;
  if(z > zmax)     z   = zmax;

  // Avoid NaN
  if(abs(k-1) < 1e-10) k = 1.0 + 1e-10;
  if(k < 1e-10) return 1.0;

  double aminusone  = 1.0/(1.0+z) - 1.0;
  double aminusone2 = aminusone*aminusone;
  double b0 = (param[ 0]); 
  double b1 = (param[ 1]);
  double b2 = (param[ 2]);
  double c0 = (param[ 3]); 
  double c1 = (param[ 4]);
  double c2 = (param[ 5]);
  double d0 = (param[ 6]); 
  double d1 = (param[ 7]);
  double d2 = (param[ 8]);
  double e0 = (param[ 9]);
  double e1 = (param[10]);
  double e2 = (param[11]);

  double b = b0 + b1*aminusone + b2*aminusone2;
  double c = c0 + c1*aminusone + c2*aminusone2;
  double d = d0 + d1*aminusone + d2*aminusone2;
  double e = e0 + e1*aminusone + e2*aminusone2;
  
  double ratio = 1.0 + (b * k) * (b * k) / (1.0 + c * k * k) + fabs(log(k)*k/(k-1)) * d * atan(e * k);

  return ratio;
}

