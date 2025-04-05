import numpy as np
  
fR0_fid  = 1e-5
param_fid  = [
 3.1       ,  2.34466 , -1.86362  ,  34.4951   ,
 28.8637   , -13.1302 ,  0.146542 , -0.0103024 ,
 -0.149446 ,  1.62807 ,  0.71291  , -1.41003]

"""
 This is a fitting formula for the linear enhancement:

 pofk_enhancement_linear = Plin(k,z) / Plin_LCDM(k,z) 

 in Hu-Sawikcy f(R) gravity with n =1

 [k] is assumed to in units of [h/Mpc]
 
 NB: If requesting a value outside of the regime we have
 fitted for we return the closest value without
 any warning
"""
def pofk_enhancement_linear(zin,fR0in,kin):
  z = zin
  fR0 = fR0in
  k = kin
  a = 1.0/(1.0+z)

  # The ranges we performed the fit for
  kmin = 0.001;
  kmax = 100.0;
  zmin = 0.0;
  zmax = 5.0;
  fR0min = 1e-7;
  fR0max = 1e-3;
  
  if(fR0 < fR0min): fR0 = fR0min;
  if(fR0 > fR0max): fR0 = fR0max;
  
  # Transform to effective k
  k = k * np.sqrt(fR0 / fR0_fid);

  # Bound checks
  if(k > kmax):     k   = kmax;
  if(z < zmin):     z   = zmin;
  if(z > zmax):     z   = zmax;
  
  # Avoid NaN
  if(abs(k-1) < 1e-10): k = 1.0 + 1e-10
  if(k < 1e-10): return 1.0

  aminusone  = 1.0/(1.0+z) - 1.0;
  aminusone2 = aminusone*aminusone;
  b0 = (param_fid[ 0]); 
  b1 = (param_fid[ 1]);
  b2 = (param_fid[ 2]);
  c0 = (param_fid[ 3]); 
  c1 = (param_fid[ 4]);
  c2 = (param_fid[ 5]);
  d0 = (param_fid[ 6]); 
  d1 = (param_fid[ 7]);
  d2 = (param_fid[ 8]);
  e0 = (param_fid[ 9]);
  e1 = (param_fid[10]);
  e2 = (param_fid[11]);

  b = b0 + b1*aminusone + b2*aminusone2;
  c = c0 + c1*aminusone + c2*aminusone2;
  d = d0 + d1*aminusone + d2*aminusone2;
  e = e0 + e1*aminusone + e2*aminusone2;

  ratio = 1.0 + (b * k) * (b * k) / (1.0 + c * k * k) + abs(np.log(k)*k/(k-1)) * d * np.arctan(e * k)

  return ratio

