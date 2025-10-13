from numpy import sqrt, log, sin, exp, power as pow
SQR, CUBE, POW4 = lambda x: x**2, lambda x: x**3, lambda x: x**4

def transferfunction_EisensteinHu(k,omega0hh,f_baryon,Tcmb):
  '''
  
  Eisenstein & Hu fit to the power spectrum (arXiv:astro-ph/9709112). This code
  is adapted from C code available on Wayne Hu's website at
  <https://background.uchicago.edu/~whu/transfer/power.c>.
  
  Parameters:
    
    k: float or array
      Wavenumber(s) in Mpc^-1
      
    omega0hh, f_baryon, Tcmb: floats
      Matter density parameter, baryon fraction, and CMB temperature,
      respectively.
  
  Returns:
    
    T_CDM, T_baryons, T_total: floats or arrays
      Transfer functions for CDM, baryons, and density-weighted total,
      respectively.
      
  '''
    
  omhh = omega0hh;
  obhh = omhh*f_baryon;
  if (Tcmb<=0.0):
    Tcmb=2.728;
  theta_cmb = Tcmb/2.7;

  z_equality = 2.50e4*omhh/POW4(theta_cmb);  # Really 1+z
  k_equality = 0.0746*omhh/SQR(theta_cmb);

  z_drag_b1 = 0.313*pow(omhh,-0.419)*(1+0.607*pow(omhh,0.674));
  z_drag_b2 = 0.238*pow(omhh,0.223);
  z_drag = 1291*pow(omhh,0.251)/(1+0.659*pow(omhh,0.828))*(1+z_drag_b1*pow(obhh,z_drag_b2));
  
  R_drag = 31.5*obhh/POW4(theta_cmb)*(1000/(1+z_drag));
  R_equality = 31.5*obhh/POW4(theta_cmb)*(1000/z_equality);

  sound_horizon = 2./3./k_equality*sqrt(6./R_equality)*log((sqrt(1+R_drag)+sqrt(R_drag+R_equality))/(1+sqrt(R_equality)));

  k_silk = 1.6*pow(obhh,0.52)*pow(omhh,0.73)*(1+pow(10.4*omhh,-0.95));

  alpha_c_a1 = pow(46.9*omhh,0.670)*(1+pow(32.1*omhh,-0.532));
  alpha_c_a2 = pow(12.0*omhh,0.424)*(1+pow(45.0*omhh,-0.582));
  alpha_c = pow(alpha_c_a1,-f_baryon)*pow(alpha_c_a2,-CUBE(f_baryon));
  
  beta_c_b1 = 0.944/(1+pow(458*omhh,-0.708));
  beta_c_b2 = pow(0.395*omhh, -0.0266);
  beta_c = 1.0/(1+beta_c_b1*(pow(1-f_baryon, beta_c_b2)-1));

  y = z_equality/(1+z_drag);
  alpha_b_G = y*(-6.*sqrt(1+y)+(2.+3.*y)*log((sqrt(1+y)+1)/(sqrt(1+y)-1)));
  alpha_b = 2.07*k_equality*sound_horizon*pow(1+R_drag,-0.75)*alpha_b_G;

  beta_node = 8.41*pow(omhh, 0.435);
  beta_b = 0.5+f_baryon+(3.-2.*f_baryon)*sqrt(pow(17.2*omhh,2.0)+1);

  k_peak = 2.5*3.14159*(1+0.217*omhh)/sound_horizon;
  sound_horizon_fit = 44.5*log(9.83/omhh)/sqrt(1+10.0*pow(obhh,0.75));

  alpha_gamma = 1-0.328*log(431.0*omhh)*f_baryon + 0.38*log(22.3*omhh)*SQR(f_baryon);

  q = k/13.41/k_equality;
  xx = k*sound_horizon;

  T_c_ln_beta = log(2.718282+1.8*beta_c*q);
  T_c_ln_nobeta = log(2.718282+1.8*q);
  T_c_C_alpha = 14.2/alpha_c + 386.0/(1+69.9*pow(q,1.08));
  T_c_C_noalpha = 14.2 + 386.0/(1+69.9*pow(q,1.08));

  T_c_f = 1.0/(1.0+POW4(xx/5.4));
  T_c = T_c_f*T_c_ln_beta/(T_c_ln_beta+T_c_C_noalpha*SQR(q)) + (1-T_c_f)*T_c_ln_beta/(T_c_ln_beta+T_c_C_alpha*SQR(q));
  
  s_tilde = sound_horizon*pow(1+CUBE(beta_node/xx),-1./3.);
  xx_tilde = k*s_tilde;

  T_b_T0 = T_c_ln_nobeta/(T_c_ln_nobeta+T_c_C_noalpha*SQR(q));
  T_b = sin(xx_tilde)/(xx_tilde)*(T_b_T0/(1+SQR(xx/5.2))+alpha_b/(1+CUBE(beta_b/xx))*exp(-pow(k/k_silk,1.4)));
  
  f_baryon = obhh/omhh;
  T_full = f_baryon*T_b + (1-f_baryon)*T_c;
  
  return T_c, T_b, T_full
