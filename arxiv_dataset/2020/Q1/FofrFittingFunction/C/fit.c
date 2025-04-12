#include <math.h>
#include <stdio.h>
#include <stdlib.h>

const double fR0_low  = 5e-6;
double param_low[] = {0.768779 , -0.405375 , 0.0075176 , 0.0288574 , -0.0638224 , -0.401206 , 0.369507 , 0.109392 , -0.342089 , 0.226376 , -0.107105 , 0.0484649 , -0.024377 , -0.051962 , -0.0351849 , 0.147194 , 0.061761 , -0.131382 , 0.00759035 , -0.00101884 , 0.0118011 , -0.0296267 , 0.025968 , 0.076885 , 0.0312734 , 0.0293253 , -0.0141899 , 0.109011 , 0.0818948 , -0.0568241 , 0.120272 , 0.0249235 , -0.0298492 , 0.0354401 , -0.262769 , 0.230278 , -0.139116 , -0.132313 , 0.13132 , -0.0565551 , -0.0338864 , 0.0712653 , 0.20246 , -0.116113 , 0.102453 , 0.0632254 , 0.0694305 , 0.00296431 , 0.0522931 , 0.0780708 , -0.0977045};

const double fR0_mid  = 1e-5;
double param_mid[] = {0.936496 , -0.545832 , 0.634804 , -0.0290649 , -0.0954373 , -0.342491 , 0.491066 , 0.297816 , -0.287142 , -0.0399919 , 0.3037 , 0.360959 , 0.000615209 , -0.00941931 , -0.0181341 , 0.376297 , 0.486358 , 0.0349385 , 0.240066 , 0.188202 , 0.665834 , 0.0122249 , -0.0343399 , -0.0520361 , 0.261006 , 0.525633 , 0.266255 , 0.393546 , 0.29088 , -0.411491 , 0.776609 , 0.470777 , -0.681923 , -0.079589 , -0.282388 , 0.53954 , -0.0930797 , 0.0783781 , 0.194957 , 0.270378 , 0.370288 , 0.194857 , 0.318637 , 0.0457011 , 0.139237 , 0.033403 , 0.0762982 , -0.000104704 , -0.00275824 , 0.0461644 , 0.189897};

const double fR0_high = 5e-5;
double param_high[] = {0.572477 , 0.254686 , 1.21637 , 0.00046274 , -0.0901242 , -0.355849 , 2.31154 , 2.29822 , -0.483186 , 0.4988 , 0.36089 , 0.0703424 , 0.0257389 , 0.0168936 , -0.030697 , -0.206992 , 0.266084 , 0.603357 , 0.574264 , -0.30799 , 0.831644 , -0.00936448 , 0.00221153 , 0.0076829 , -0.650381 , 0.0179215 , 0.927038 , 0.77903 , 0.919643 , -0.936328 , 1.26756 , 1.44477 , -1.44129 , 0.219594 , 0.353883 , 1.02533 , -0.251705 , 0.124875 , 0.345995 , -0.146438 , 0.0200251 , 0.0892343 , 0.284755 , -0.158286 , 0.541178 , -0.0471913 , 0.139772 , -0.134888 , 0.0959162 , 0.368186 , -0.157828};

// General definition of the fitting formula
double ratio_by_param(double r, double aminusone, double k, double *param){
  double aminusone2 = aminusone*aminusone;
  double r2 = r*r;

  double b0 = (param[ 0]) + (param[ 9])*r + (param[18])*r2; 
  double b1 = (param[ 1]) + (param[10])*r + (param[19])*r2;
  double b2 = (param[ 2]) + (param[11])*r + (param[20])*r2;
  double c0 = (param[ 3]) + (param[12])*r + (param[21])*r2; 
  double c1 = (param[ 4]) + (param[13])*r + (param[22])*r2;
  double c2 = (param[ 5]) + (param[14])*r + (param[23])*r2;
  double d0 = (param[ 6]) + (param[15])*r + (param[24])*r2; 
  double d1 = (param[ 7]) + (param[16])*r + (param[25])*r2;
  double d2 = (param[ 8]) + (param[17])*r + (param[26])*r2;
  double e0 = (1.0      ) + (param[27])*r + (param[30])*r2;
  double e1 = (0.0      ) + (param[28])*r + (param[31])*r2;
  double e2 = (0.0      ) + (param[29])*r + (param[32])*r2;
  double f0 = (param[33]) + (param[36])*r + (param[39])*r2; 
  double f1 = (param[34]) + (param[37])*r + (param[40])*r2;
  double f2 = (param[35]) + (param[38])*r + (param[41])*r2;
  double g0 = (param[42]) + (param[45])*r + (param[48])*r2; 
  double g1 = (param[43]) + (param[46])*r + (param[49])*r2;
  double g2 = (param[44]) + (param[47])*r + (param[50])*r2;

  double b = b0 + b1*aminusone + b2*aminusone2;
  double c = c0 + c1*aminusone + c2*aminusone2;
  double d = d0 + d1*aminusone + d2*aminusone2;
  double e = e0 + e1*aminusone + e2*aminusone2;
  double f = f0 + f1*aminusone + f2*aminusone2;
  double g = g0 + g1*aminusone + g2*aminusone2;

  double ratio = 1.0 + b * ( 1.0 + c*k)/(1.0 + e*k) * pow( atan(d * k), 1.0 + f + g * k);

  return ratio;
}

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

double pofk_enhancement(double _z, double _fR0, double _k){
  double z = _z, fR0 = _fR0, k = _k;

  // The ranges we performed the fit for
  const double kmin   = 0.01;
  const double kmax   = 10.0;
  const double zmin   = 0.0;
  const double zmax   = 3.0;
  const double fR0min = 1e-7;
  const double fR0max = 1e-4;

  // Bound checks (by design ratio -> 1 as k -> 0 so no need to impose lower limit)
  if(k > kmax)     k   = kmax;
  if(z < zmin)     z   = zmin;
  if(z > zmax)     z   = zmax;
  if(fR0 < fR0min) fR0 = fR0min;
  if(fR0 > fR0max) fR0 = fR0max;
  
  double a = 1.0/(1.0+z);

  //==============================================================
  // The best-fit parameters for the range 1e-7 < fR0 < 5e-6
  //==============================================================
  double ratio_low  = ratio_by_param( log(fR0 / fR0_low ) , a-1 , k, param_low );

  //==============================================================
  // The best-fit parameters for the range 5e-6 < fR0 < 5e-5
  //==============================================================
  double ratio_mid  = ratio_by_param( log(fR0 / fR0_mid ) , a-1 , k, param_mid );

  //==============================================================
  // The best-fit parameters for the range 1e-5 < fR0 < 1e-4
  //==============================================================
  double ratio_high = ratio_by_param( log(fR0 / fR0_high) , a-1 , k, param_high);

  double ratio = 1.0;
  if(fR0 > 5e-5){
    ratio = ratio_high;
  } else if(fR0 < 5e-6) {
    ratio = ratio_low;
  } else if(fR0 > 1e-5) {
    ratio = ratio_mid + (ratio_high - ratio_mid) * (fR0 - 1e-5)/(5e-5 - 1e-5);
  } else {
    ratio = ratio_low + (ratio_mid  - ratio_low) * (fR0 - 5e-6)/(1e-5 - 5e-6);
  }
  return ratio > 1.0 ? ratio : 1.0;
}

