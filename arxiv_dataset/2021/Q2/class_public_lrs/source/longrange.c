/*
 * Numerical module that computes the long-range interaction integrals
 */
#include "longrange.h"

/* CLASS stores energy densities in units such that
   
   rho_class = 8piG/3 * rho_physical / Mpc^-2 = 1374.3 * rho_physical/eV^4
   
   Alternatively, 
   1374.3 = 8*pi*G/3 / (hbar^3*c^7) * Mpc_over_m^2 * eV^4

   This factor will be stored here
*/
const double _eV4_to_rho_class = 8*M_PI*_G_/3 / (pow(_h_P_/(2*_PI_), 3) * pow(_c_, 7)) * pow(_Mpc_over_m_, 2) * pow(_eV_, 4); // Multiply by this to convert eV^4 to CLASS units
const double _J4_to_rho_class = 8*M_PI*_G_/3 / (pow(_h_P_/(2*_PI_), 3) * pow(_c_, 7)) * pow(_Mpc_over_m_, 2); // Multiply by this to convert J^4 to CLASS units
const double _Mpc_times_eV = _eV_/(_h_P_/(2*_PI_) * _c_) * _Mpc_over_m_; // Multiply by this to convert eV to Mpc^-1

/**
 * This is the routine where the distribution function f0(q) of the
 * fermion undergoing long-range interactions is specified
 *
 * @param pbadist Input:  structure containing all parameters defining f0(q)
 * @param q       Input:  momentum
 * @param f0      Output: phase-space distribution
 */
int background_lrs_distribution(
                                 void * pbadist,
                                 double q,
                                 double * f0
                                 ) {
  struct background * pba;
  struct background_parameters_for_distributions * pbadist_local;

  /** - extract from the input structure pbadist all the relevant information */
  pbadist_local = pbadist;          /* restore actual format of pbadist */
  pba = pbadist_local->pba;         /* extract the background structure from it */

  *f0 = pba->lrs_g_F * 1.0/CUB(2*_PI_) * 1./(exp(q)+1.);
  // Fermi-Dirac distribution with g_F degrees of freedom

  return _SUCCESS_;
}

/**
 * This is the routine where the derivative of the
 * distribution function f0(q) is specified
 *
 * @param pbadist         Input:  structure containing all parameters defining f0(q)
 * @param q               Input:  momentum
 * @param dlnf0_dlnq      Output: derivative of log[phase-space distribution] wrt log[q]
 */
int background_lrs_derivative(
                                 void * pbadist,
                                 double q,
                                 double * dlnf0_dlnq
                                 ) {
  struct background * pba;
  struct background_parameters_for_distributions * pbadist_local;

  /** - extract from the input structure pbadist all the relevant information */
  pbadist_local = pbadist;          /* restore actual format of pbadist */
  pba = pbadist_local->pba;         /* extract the background structure from it */

  double f0 = 1./(exp(q)+1.);
  // Fermi-Dirac distribution with g_F degrees of freedom

  //Avoid underflow in extreme tail:
  if (fabs(f0)==0.)
    *dlnf0_dlnq = -q;
  else
    *dlnf0_dlnq = -q * exp(q)/(exp(q)+1.);

  return _SUCCESS_;
}

/**
 * This function is only used for the purpose of finding optimal
 * quadrature weights. The logic is: if we can accurately convolve
 * f0(q) with this function, then we can convolve it accurately with
 * any other relevant function.
 *
 * @param pbadist Input:  structure containing all background parameters
 * @param q       Input:  momentum
 * @param test    Output: value of the test function test(q)
 */

int background_lrs_test_function(
                                  void * pbadist,
                                  double q,
                                  double * test
                                  ) {

  double c = 2.0/(3.0*_zeta3_);
  double d = 120.0/(7.0*SQR(SQR(_PI_)));
  double e = 2.0/(45.0*_zeta5_);

  /** Using a + bq creates problems for otherwise acceptable distributions
      which diverges as \f$ 1/r \f$ or \f$ 1/r^2 \f$ for \f$ r\to 0 \f$*/
  *test = CUB(2.0*_PI_)/6.0*(c*q*q-d*q*q*q-e*q*q*q*q);

  return _SUCCESS_;
}


/**
 * This function finds optimal quadrature weights for the
 * long-range interaction integrals
 *
 * @param ppr Input: precision structure
 * @param pba Input/Output: background structure
 */

int background_lrs_init(
                         struct precision *ppr,
                         struct background *pba
                         ) {

  int index_q, tolexp,row,status,filenum;
  double q, dlnf0_dlnq;
  struct background_parameters_for_distributions pbadist;
  FILE *psdfile;

  pbadist.pba = pba;

  pbadist.q = NULL;
  pbadist.tablesize = 0;

  /* Handle perturbation qsampling: */
  if (pba->lrs_quadrature_strategy==qm_auto){
    /** Automatic q-sampling for this species */
    class_alloc(pba->q_lrs,_QUADRATURE_MAX_*sizeof(double),pba->error_message);
    class_alloc(pba->w_lrs,_QUADRATURE_MAX_*sizeof(double),pba->error_message);

    class_call(get_qsampling(pba->q_lrs,
			     pba->w_lrs,
			     &(pba->q_size_lrs),
			     _QUADRATURE_MAX_,
			     ppr->tol_lrs,
			     pbadist.q,
			     pbadist.tablesize,
			     background_lrs_test_function,
			     background_lrs_distribution,
			     &pbadist,
			     pba->error_message),
	       pba->error_message,
	       pba->error_message);
    pba->q_lrs=realloc(pba->q_lrs,pba->q_size_lrs*sizeof(double));
    pba->w_lrs=realloc(pba->w_lrs,pba->q_size_lrs*sizeof(double));


    if (pba->background_verbose > 0)
      printf("lrs species sampled with %d points for purpose of perturbation integration\n",
	     pba->q_size_lrs);

    /* Handle background q_sampling: */
    class_alloc(pba->q_lrs_bg,_QUADRATURE_MAX_BG_*sizeof(double),pba->error_message);
    class_alloc(pba->w_lrs_bg,_QUADRATURE_MAX_BG_*sizeof(double),pba->error_message);

    class_call(get_qsampling(pba->q_lrs_bg,
			     pba->w_lrs_bg,
			     &(pba->q_size_lrs_bg),
			     _QUADRATURE_MAX_BG_,
			     ppr->tol_lrs_bg,
			     pbadist.q,
			     pbadist.tablesize,
			     background_lrs_test_function,
			     background_lrs_distribution,
			     &pbadist,
			     pba->error_message),
	       pba->error_message,
	       pba->error_message);


    pba->q_lrs_bg=realloc(pba->q_lrs_bg,pba->q_size_lrs_bg*sizeof(double));
    pba->w_lrs_bg=realloc(pba->w_lrs_bg,pba->q_size_lrs_bg*sizeof(double));

    /** - in verbose mode, inform user of number of sampled momenta
	for background quantities */
    if (pba->background_verbose > 0)
      printf("lrs species sampled with %d points for purpose of background integration\n",
	     pba->q_size_lrs_bg);
  }
  else{
    /** Manual q-sampling for this species. Same sampling used for both perturbation and background sampling, since this will usually be a high precision setting anyway */
    pba->q_size_lrs_bg = pba->lrs_input_q_size;
    pba->q_size_lrs = pba->lrs_input_q_size;
    class_alloc(pba->q_lrs_bg,pba->q_size_lrs_bg*sizeof(double),pba->error_message);
    class_alloc(pba->w_lrs_bg,pba->q_size_lrs_bg*sizeof(double),pba->error_message);
    class_alloc(pba->q_lrs,pba->q_size_lrs*sizeof(double),pba->error_message);
    class_alloc(pba->w_lrs,pba->q_size_lrs*sizeof(double),pba->error_message);
    class_call(get_qsampling_manual(pba->q_lrs,
				    pba->w_lrs,
				    pba->q_size_lrs,
				    pba->lrs_qmax,
				    pba->lrs_quadrature_strategy,
				    pbadist.q,
				    pbadist.tablesize,
				    background_lrs_distribution,
				    &pbadist,
				    pba->error_message),
	       pba->error_message,
	       pba->error_message);
    for (index_q=0; index_q<pba->q_size_lrs; index_q++) {
      pba->q_lrs_bg[index_q] = pba->q_lrs[index_q];
      pba->w_lrs_bg[index_q] = pba->w_lrs[index_q];
    }
    /** - in verbose mode, inform user of number of sampled momenta
	for background quantities */
    if (pba->background_verbose > 0)
      printf("lrs species sampled with %d points for purpose of background andperturbation integration using the manual method\n",
	     pba->q_size_lrs);
  }

  class_alloc(pba->dlnf0_dlnq_lrs,
	      pba->q_size_lrs*sizeof(double),
	      pba->error_message);


  for (index_q=0; index_q<pba->q_size_lrs; index_q++) {
    q = pba->q_lrs[index_q];
    class_call(background_lrs_derivative(&pbadist,q,&dlnf0_dlnq),
	       pba->error_message,pba->error_message);
    pba->dlnf0_dlnq_lrs[index_q] = dlnf0_dlnq;
  }

  /* rho_class = 8piG/3 * rho_physical / Mpc^-2
     The 4pi (kB T0)^4 will appear in front of integrals*/
  pba->factor_lrs=4*_PI_*SQR(SQR(pba->T_cmb*pba->lrs_T_F*_k_B_)) * _J4_to_rho_class;

  return _SUCCESS_;
}


/**
 * For a given long-range interacting species: given the quadrature weights, the mass
 * and the redshift, find background *fermion* quantities by a quick weighted
 * sum over.  Input parameters passed as NULL pointers are not
 * evaluated for speed-up
 *
 * @param qvec             Input: sampled momenta
 * @param wvec             Input: quadrature weights
 * @param qsize            Input: number of momenta/weights
 * @param m_over_T0        Input: *effective* fermion mass divided by its *present day* temperature
 * @param factor           Input: normalization factor for the p.s.d.
 * @param z                Input: redshift
 * @param rho              Output: energy density
 * @param p                Output: pressure
 * @param I_MPhi           Output: \integ d^3q m/T 1/eps f(q)
                           where eps^2 = q^2 + (m/T)^2 and q=p/T
 * @param pseudo_p         Output: pseudo-pressure used in perturbation module for fluid approx
 * @param I1               Output: \integ d^3q m/T (2 eps^2 + (m/T)^2)/eps^3 f(q)
 *                         where eps^2 = q^2 + (m/T)^2 and q=p/T
 * @param I2               Output: \integ d^3q q^2/eps^3 f(q)
 *                         where eps^2 = q^2 + (m/T)^2 and q=p/T
 */

int background_lrs_momenta(
                            /* Only calculate for non-NULL pointers: */
                            double * qvec,
                            double * wvec,
                            int qsize,
                            double m_over_T0,
                            double factor,
                            double z,
                            double * rho, // density
                            double * p,   // pressure
                            double * I_Mphi,  
                            double * pseudo_p,  // pseudo-p used in ncdm fluid approx
			    double * I1,
			    double * I2
                            ) {

  int index_q;
  double epsilon;
  double q2;
  double factor2;
  /** Summary: */

  /** - rescale normalization at given redshift */
  factor2 = factor*SQR(SQR(1+z));

  /** - initialize quantities */
  if (rho!=NULL) *rho = 0.;
  if (p!=NULL) *p = 0.;
  if (I_Mphi!=NULL) *I_Mphi = 0.;
  if (pseudo_p!=NULL) *pseudo_p = 0.;
  if (I1!=NULL) *I1 = 0.;
  if (I2!=NULL) *I2 = 0.;

  /** - loop over momenta */
  for (index_q=0; index_q<qsize; index_q++) {

    /* squared momentum */
    q2 = qvec[index_q]*qvec[index_q];

    /* energy */
    epsilon = sqrt(q2+m_over_T0*m_over_T0/(1.+z)/(1.+z));

    /* integrand of the various quantities */
    if (rho!=NULL) *rho += q2 * epsilon*wvec[index_q];
    if (p!=NULL) *p += q2 * q2/3./epsilon*wvec[index_q];
    if (I_Mphi!=NULL) *I_Mphi += q2 * m_over_T0/(1.+z)/epsilon*wvec[index_q];
    if (pseudo_p!=NULL) *pseudo_p += q2 * SQR(q2)/CUB(epsilon)/3.0*wvec[index_q];
    if (I1!=NULL) *I1 += q2 * m_over_T0/(1.+z) * (2*SQR(epsilon) + SQR(m_over_T0/(1.+z))) / CUB(epsilon) * wvec[index_q];
    if (I2!=NULL) *I2 += q2 * q2/CUB(epsilon) * wvec[index_q];
  }

  /** - adjust normalization */
  if (rho!=NULL) *rho *= factor2;
  if (p!=NULL) *p *= factor2;
  if (I_Mphi!=NULL) *I_Mphi *= 4*_PI_;
  if (pseudo_p!=NULL) *pseudo_p *=factor2;
  if (I1!=NULL) *I1 *= 4*_PI_;
  if (I2!=NULL) *I2 *= 4*_PI_;

  return _SUCCESS_;
}

/**
 * Returns the *effective* fermion mass divided by its present day temperature
 */
inline double get_mT_over_T0_lrs(struct background * pba, double phi_M){
  // Convert phi_M to J and divide by kB*T0
  double phi_M_over_T0 = phi_M * _eV_ / (_k_B_*pba->lrs_T_F*pba->T_cmb);

  return pba->lrs_m_F_over_T0 + phi_M_over_T0 * pba->lrs_g_over_M;
}

/**
 * Function of the scalar field that is zero when the e.o.m. are satisfied
 * (it's the derivative of the potential with respect to the field)
 *
 * @param phi_M  Input:  Scalar field times its mass (eV^2)
 * @param param  Input:  Pointer to background_parameters_and_redshift struct
 * @param y      Output: If zero, the scalar field satisfies the e.o.m
 */
int potentialPrime(double phi_M, void *param, double *y, ErrorMsg error_message){
  struct background * pba;
  struct background_parameters_and_redshift * pbaz;
  pbaz = param;
  pba = pbaz->pba;

  // We first compute \integ d^3p m/E f(p), in eV^3
  double I_Mphi;
  class_call(background_lrs_momenta(pba->q_lrs_bg,
				    pba->w_lrs_bg,
				    pba->q_size_lrs_bg,
				    get_mT_over_T0_lrs(pba, phi_M),
				    pba->factor_lrs,
				    pbaz->z,
				    NULL,
				    NULL,
				    &I_Mphi,
				    NULL,
				    NULL,
				    NULL),
             error_message,
             error_message);
  I_Mphi *= CUB(pba->T_cmb*pba->lrs_T_F*_k_B_ * (1+pbaz->z)); // Multiply by T^3 (in J)
  I_Mphi /= CUB(_eV_); // Convert to eV^3
  
  *y = phi_M +  pba->lrs_g_over_M * I_Mphi;

  return _SUCCESS_;
}

/**
 * Returns the scalar field times its mass solving the equations of motion.
 * Hubble friction is neglected (i.e., the scalar mass is assumed to be much
 * heavier than the Hubble scale), and the resulting trascendental equation
 * is solved with the Ridder method
 *
 * @param pba    Input: pointer to background structure
 * @param z      Input: redshift
 * @param phi_M  Output: scalar field times mass (eV^2)
 */
int get_phi_M_lrs(struct background * pba, double z, double * phi_M){
  double m_F_over_T = pba->lrs_m_F_over_T0 / (1.+z);
  double T = pba->lrs_m_F / m_F_over_T; // Temperature (eV)
  if(m_F_over_T < 1e-5 ||
     m_F_over_T/(pba->lrs_g_F/24. * SQR(pba->lrs_g_over_M*T)) < 1e-5){
    // Analytic result in the ultrarelativistic regime
    *phi_M = - pba->lrs_g_F/24. * pba->lrs_g_over_M * m_F_over_T * CUB(T) /
      (1 + pba->lrs_g_F/24. * SQR(pba->lrs_g_over_M) * SQR(T));
    return _SUCCESS_;
  }
  
  double x2 = 0, x1=-pba->lrs_m_F/pba->lrs_g_over_M; // We know that -m0 <= g*phi <= 0. This ensures that the effective fermion mass will always be positive
  struct background_parameters_and_redshift bpaz;
  bpaz.pba = pba;
  bpaz.z = z;

  *phi_M = 0.;
  int fevals = 0;
  class_call(class_fzero_ridder(potentialPrime,
                                x1,
                                x2,
                                1e-5*MAX(fabs(x1),fabs(x2)),
                                &bpaz,
                                NULL,
                                NULL,
                                phi_M,
                                &fevals,
                                pba->error_message),
             pba->error_message,pba->error_message);

  if(pba->lrs_m_F + pba->lrs_g_over_M * (*phi_M) < 0) // mTilde < 0 because of round-off uncertainties
    *phi_M = -pba->lrs_m_F/pba->lrs_g_over_M * (1-1e-7);
      
  class_test(pba->lrs_m_F + pba->lrs_g_over_M * (*phi_M) < 0,
	     pba->error_message,
	     "Negative effective fermion mass (%.3e eV) for m_F/T_F=%.3e, g m_F/M =%.3e (phi M = %.3e eV). Check numerical precision",
	     pba->lrs_m_F + pba->lrs_g_over_M * (*phi_M),
	     pba->lrs_m_F_over_T0/(1+z),
	     pba->lrs_g_over_M * pba->lrs_m_F,
	     *phi_M);
  
  return _SUCCESS_;
}

/**
 * Returns the scale factor at adiabatic instability onset.
 * Makes use of previously tabulated results
 *
 * @param pba    Input: pointer to background structure
 * @param a_rel  Output: a/a_0
 */
int instabilityOnset_lrs(struct background * pba, double * a_rel){
  double mT_over_T;
  
  /* Interpolate to obtain mT/T at instability onset (mT is the effective fermion mass) */
  
  double min_g = 5, max_g = pow(10, 1.2),
    max_log10g = 7; // Limits of the interpolation arrays
  double min_log10g = log10(max_g);
  double mT_over_T_lin[100], mT_over_T_log[100];
  switch(pba->lrs_g_F){
  case 1: {
    double lin[100] = {-1.000000e+01, -1.000000e+01, -1.000000e+01, -1.000000e+01, -1.000000e+01, -1.000000e+01, -1.000000e+01, -1.000000e+01, -1.000000e+01, -1.000000e+01, -1.000000e+01, -1.000000e+01, -8.896817e+00, -5.362826e+00, -4.402384e+00, -3.884355e+00, -3.543687e+00, -3.296267e+00, -3.105417e+00, -2.952105e+00, -2.825294e+00, -2.718055e+00, -2.625783e+00, -2.545275e+00, -2.474219e+00, -2.410900e+00, -2.354011e+00, -2.302539e+00, -2.255681e+00, -2.212793e+00, -2.173351e+00, -2.136923e+00, -2.103150e+00, -2.071729e+00, -2.042405e+00, -2.014960e+00, -1.989205e+00, -1.964978e+00, -1.942138e+00, -1.920560e+00, -1.900136e+00, -1.880769e+00, -1.862375e+00, -1.844876e+00, -1.828205e+00, -1.812300e+00, -1.797108e+00, -1.782577e+00, -1.768664e+00, -1.755327e+00, -1.742529e+00, -1.730236e+00, -1.718417e+00, -1.707044e+00, -1.696090e+00, -1.685532e+00, -1.675346e+00, -1.665514e+00, -1.656014e+00, -1.646831e+00, -1.637948e+00, -1.629349e+00, -1.621020e+00, -1.612948e+00, -1.605121e+00, -1.597527e+00, -1.590156e+00, -1.582997e+00, -1.576040e+00, -1.569277e+00, -1.562700e+00, -1.556300e+00, -1.550071e+00, -1.544004e+00, -1.538094e+00, -1.532334e+00, -1.526718e+00, -1.521240e+00, -1.515896e+00, -1.510681e+00, 3.483726e+00, 3.295453e+00, 3.171182e+00, 3.075478e+00, 2.996803e+00, 2.929683e+00, 2.871017e+00, 2.818854e+00, 2.771873e+00, 2.729132e+00, 2.689934e+00, 2.653748e+00, 2.620154e+00, 2.588819e+00, 2.559471e+00, 2.531886e+00, 2.505876e+00, 2.481281e+00, 2.457967e+00, 2.435817e+00};
    double log[100] = {2.435817e+00, 2.130462e+00, 1.947601e+00, 1.821151e+00, 1.727097e+00, 1.653929e+00, 1.595240e+00, 1.547104e+00, 1.506950e+00, 1.473007e+00, 1.444006e+00, 1.419010e+00, 1.397308e+00, 1.378350e+00, 1.361703e+00, 1.347019e+00, 1.334017e+00, 1.322464e+00, 1.312170e+00, 1.302972e+00, 1.294736e+00, 1.287346e+00, 1.280703e+00, 1.274721e+00, 1.269328e+00, 1.264459e+00, 1.260058e+00, 1.256076e+00, 1.252470e+00, 1.249201e+00, 1.246236e+00, 1.243544e+00, 1.241099e+00, 1.238878e+00, 1.236857e+00, 1.235020e+00, 1.233347e+00, 1.231824e+00, 1.230438e+00, 1.229174e+00, 1.228023e+00, 1.226974e+00, 1.226017e+00, 1.225144e+00, 1.224348e+00, 1.223622e+00, 1.222960e+00, 1.222355e+00, 1.221804e+00, 1.221300e+00, 1.220840e+00, 1.220420e+00, 1.220037e+00, 1.219687e+00, 1.219367e+00, 1.219075e+00, 1.218809e+00, 1.218565e+00, 1.218343e+00, 1.218140e+00, 1.217954e+00, 1.217784e+00, 1.217629e+00, 1.217488e+00, 1.217358e+00, 1.217240e+00, 1.217132e+00, 1.217033e+00, 1.216943e+00, 1.216861e+00, 1.216786e+00, 1.216717e+00, 1.216654e+00, 1.216596e+00, 1.216544e+00, 1.216496e+00, 1.216452e+00, 1.216412e+00, 1.216375e+00, 1.216347e+00, 1.216311e+00, 1.216283e+00, 1.216258e+00, 1.216234e+00, 1.216213e+00, 1.216193e+00, 1.216176e+00, 1.216159e+00, 1.216144e+00, 1.216131e+00, 1.216118e+00, 1.216107e+00, 1.216097e+00, 1.216087e+00, 1.216078e+00, 1.216070e+00, 1.216063e+00, 1.216057e+00, 1.216051e+00, 1.216045e+00};
    memcpy(mT_over_T_lin, lin, sizeof(mT_over_T_lin));
    memcpy(mT_over_T_log, log, sizeof(mT_over_T_log));
    break; }
  case 2: {
    double lin[100] = {-2.966221e+00, -2.789834e+00, -2.649638e+00, -2.534669e+00, -2.438149e+00, -2.355622e+00, -1.000000e+01, -2.221136e+00, -2.165357e+00, -2.115454e+00, -2.070478e+00, -2.029683e+00, -1.992471e+00, -1.958358e+00, -1.926947e+00, -1.897909e+00, -1.870967e+00, -1.845888e+00, -1.822474e+00, -1.800554e+00, -1.779980e+00, -1.760626e+00, -1.742380e+00, -1.725144e+00, -1.708832e+00, -1.693366e+00, -1.678681e+00, -1.664713e+00, -1.651411e+00, -1.638724e+00, -1.626608e+00, -1.615025e+00, -1.603937e+00, -1.593312e+00, -1.583121e+00, -1.573335e+00, -1.563930e+00, -1.554884e+00, -1.546174e+00, -1.537782e+00, -1.529689e+00, -1.521880e+00, -1.514339e+00, 3.569901e+00, 3.277686e+00, 3.117022e+00, 3.000212e+00, 2.907280e+00, 2.829775e+00, 2.763202e+00, 2.704843e+00, 2.652911e+00, 2.606160e+00, 2.563685e+00, 2.524802e+00, 2.488984e+00, 2.455812e+00, 2.424949e+00, 2.396120e+00, 2.369095e+00, 2.343681e+00, 2.319717e+00, 2.297061e+00, 2.275594e+00, 2.255211e+00, 2.235820e+00, 2.217340e+00, 2.199700e+00, 2.182837e+00, 2.166694e+00, 2.151220e+00, 2.136370e+00, 2.122101e+00, 2.108377e+00, 2.095164e+00, 2.082429e+00, 2.070146e+00, 2.058288e+00, 2.046830e+00, 2.035752e+00, 2.025032e+00, 2.014652e+00, 2.004594e+00, 1.994842e+00, 1.985381e+00, 1.976197e+00, 1.967277e+00, 1.958609e+00, 1.950181e+00, 1.941983e+00, 1.934004e+00, 1.926236e+00, 1.918668e+00, 1.911294e+00, 1.904104e+00, 1.897092e+00, 1.890251e+00, 1.883574e+00, 1.877054e+00, 1.870686e+00};
    double log[100] = {1.870686e+00, 1.764551e+00, 1.683380e+00, 1.619043e+00, 1.566738e+00, 1.523400e+00, 1.486961e+00, 1.455962e+00, 1.429339e+00, 1.406293e+00, 1.386212e+00, 1.368617e+00, 1.353125e+00, 1.339429e+00, 1.327277e+00, 1.316462e+00, 1.306810e+00, 1.298175e+00, 1.290433e+00, 1.283479e+00, 1.277222e+00, 1.271584e+00, 1.266497e+00, 1.261900e+00, 1.257743e+00, 1.253980e+00, 1.250570e+00, 1.247478e+00, 1.244672e+00, 1.242124e+00, 1.239809e+00, 1.237704e+00, 1.235790e+00, 1.234049e+00, 1.232463e+00, 1.231019e+00, 1.229704e+00, 1.228506e+00, 1.227414e+00, 1.226418e+00, 1.225510e+00, 1.224682e+00, 1.223927e+00, 1.223238e+00, 1.222609e+00, 1.222035e+00, 1.221511e+00, 1.221033e+00, 1.220597e+00, 1.220198e+00, 1.219834e+00, 1.219502e+00, 1.219198e+00, 1.218921e+00, 1.218668e+00, 1.218436e+00, 1.218225e+00, 1.218032e+00, 1.217856e+00, 1.217694e+00, 1.217547e+00, 1.217413e+00, 1.217290e+00, 1.217178e+00, 1.217075e+00, 1.216981e+00, 1.216895e+00, 1.216817e+00, 1.216746e+00, 1.216680e+00, 1.216621e+00, 1.216566e+00, 1.216516e+00, 1.216470e+00, 1.216429e+00, 1.216391e+00, 1.216361e+00, 1.216324e+00, 1.216295e+00, 1.216268e+00, 1.216244e+00, 1.216222e+00, 1.216202e+00, 1.216183e+00, 1.216166e+00, 1.216151e+00, 1.216137e+00, 1.216124e+00, 1.216112e+00, 1.216101e+00, 1.216091e+00, 1.216082e+00, 1.216074e+00, 1.216066e+00, 1.216059e+00, 1.216053e+00, 1.216047e+00, 1.216042e+00, 1.216037e+00, 1.216033e+00};
    memcpy(mT_over_T_lin, lin, sizeof(mT_over_T_lin));
    memcpy(mT_over_T_log, log, sizeof(mT_over_T_log));
    break; }
  case 3: {
    double lin[100] = {-2.059731e+00, -2.011440e+00, -1.968163e+00, -1.929108e+00, -1.893648e+00, -1.861278e+00, -1.831586e+00, -1.804234e+00, -1.778939e+00, -1.755465e+00, -1.733611e+00, -1.713205e+00, -1.694101e+00, -1.676171e+00, -1.659305e+00, -1.643406e+00, -1.628388e+00, -1.614176e+00, -1.600705e+00, -1.587915e+00, -1.575752e+00, -1.564170e+00, -1.000000e+01, -1.542582e+00, -1.532503e+00, -1.522857e+00, -1.513616e+00, 3.444007e+00, 3.182213e+00, 3.024310e+00, 2.907559e+00, 2.814206e+00, 2.736271e+00, 2.669376e+00, 2.610824e+00, 2.558824e+00, 2.512116e+00, 2.469778e+00, 2.431113e+00, 2.395579e+00, 2.362748e+00, 2.332273e+00, 2.303870e+00, 2.277304e+00, 2.252377e+00, 2.228919e+00, 2.206789e+00, 2.185862e+00, 2.166030e+00, 2.147198e+00, 2.129285e+00, 2.112217e+00, 2.095929e+00, 2.080362e+00, 2.065465e+00, 2.051192e+00, 2.037499e+00, 2.024349e+00, 2.011707e+00, 1.999541e+00, 1.987823e+00, 1.976525e+00, 1.965624e+00, 1.955098e+00, 1.944925e+00, 1.935087e+00, 1.925566e+00, 1.916345e+00, 1.907411e+00, 1.898747e+00, 1.890342e+00, 1.882184e+00, 1.874260e+00, 1.866559e+00, 1.859073e+00, 1.851791e+00, 1.844705e+00, 1.837805e+00, 1.831086e+00, 1.824538e+00, 1.818155e+00, 1.811931e+00, 1.805859e+00, 1.799933e+00, 1.794148e+00, 1.788499e+00, 1.782980e+00, 1.777587e+00, 1.772315e+00, 1.767160e+00, 1.762118e+00, 1.757185e+00, 1.752357e+00, 1.747630e+00, 1.743002e+00, 1.738469e+00, 1.734029e+00, 1.729677e+00, 1.725411e+00, 1.721229e+00};
    double log[100] = {1.721229e+00, 1.649278e+00, 1.591459e+00, 1.543972e+00, 1.504317e+00, 1.470767e+00, 1.442083e+00, 1.417345e+00, 1.395857e+00, 1.377079e+00, 1.360584e+00, 1.346030e+00, 1.333139e+00, 1.321683e+00, 1.311473e+00, 1.302349e+00, 1.294177e+00, 1.286844e+00, 1.280251e+00, 1.274314e+00, 1.268961e+00, 1.264127e+00, 1.259758e+00, 1.255805e+00, 1.252224e+00, 1.248978e+00, 1.246033e+00, 1.243360e+00, 1.240932e+00, 1.238726e+00, 1.236719e+00, 1.234894e+00, 1.233233e+00, 1.231720e+00, 1.230343e+00, 1.229088e+00, 1.227944e+00, 1.226902e+00, 1.225951e+00, 1.225085e+00, 1.224294e+00, 1.223573e+00, 1.222915e+00, 1.222314e+00, 1.221766e+00, 1.221265e+00, 1.220809e+00, 1.220392e+00, 1.220011e+00, 1.219663e+00, 1.219346e+00, 1.219055e+00, 1.218791e+00, 1.218549e+00, 1.218328e+00, 1.218126e+00, 1.217941e+00, 1.217773e+00, 1.217619e+00, 1.217478e+00, 1.217350e+00, 1.217232e+00, 1.217125e+00, 1.217027e+00, 1.216937e+00, 1.216855e+00, 1.216780e+00, 1.216712e+00, 1.216650e+00, 1.216592e+00, 1.216540e+00, 1.216493e+00, 1.216449e+00, 1.216409e+00, 1.216373e+00, 1.216339e+00, 1.216309e+00, 1.216281e+00, 1.216256e+00, 1.216233e+00, 1.216211e+00, 1.216192e+00, 1.216174e+00, 1.216158e+00, 1.216143e+00, 1.216130e+00, 1.216117e+00, 1.216106e+00, 1.216096e+00, 1.216086e+00, 1.216078e+00, 1.216070e+00, 1.216063e+00, 1.216056e+00, 1.216050e+00, 1.216045e+00, 1.216040e+00, 1.216035e+00, 1.216031e+00, 1.216027e+00};
    memcpy(mT_over_T_lin, lin, sizeof(mT_over_T_lin));
    memcpy(mT_over_T_log, log, sizeof(mT_over_T_log));
    break; }
  case 4: {
    double lin[100] = {-1.802701e+00, -1.773790e+00, -1.747247e+00, -1.722777e+00, -1.700133e+00, -1.679107e+00, -1.659523e+00, -1.641231e+00, -1.624099e+00, -1.608016e+00, -1.592883e+00, -1.578614e+00, -1.565135e+00, -1.552377e+00, -1.540283e+00, -1.528798e+00, -1.517876e+00, 3.601566e+00, 3.213250e+00, 3.024606e+00, 2.892075e+00, 2.788875e+00, 2.704192e+00, 2.632417e+00, 2.570212e+00, 2.515414e+00, 2.466531e+00, 2.422485e+00, 2.382472e+00, 2.345873e+00, 2.312203e+00, 2.281072e+00, 2.252162e+00, 2.225212e+00, 2.200003e+00, 2.176351e+00, 2.154098e+00, 2.133109e+00, 2.113267e+00, 2.094471e+00, 2.076630e+00, 2.059667e+00, 2.043511e+00, 2.028101e+00, 2.013382e+00, 1.999303e+00, 1.985819e+00, 1.972892e+00, 1.960483e+00, 1.948560e+00, 1.937092e+00, 1.926051e+00, 1.915413e+00, 1.905154e+00, 1.895252e+00, 1.885687e+00, 1.876442e+00, 1.867499e+00, 1.858844e+00, 1.850460e+00, 1.842335e+00, 1.834456e+00, 1.826811e+00, 1.819390e+00, 1.812181e+00, 1.805176e+00, 1.798365e+00, 1.791740e+00, 1.785293e+00, 1.779016e+00, 1.772902e+00, 1.766945e+00, 1.761138e+00, 1.755475e+00, 1.749951e+00, 1.744560e+00, 1.739298e+00, 1.734159e+00, 1.729139e+00, 1.724233e+00, 1.719438e+00, 1.714749e+00, 1.710164e+00, 1.705677e+00, 1.701287e+00, 1.696989e+00, 1.692781e+00, 1.688660e+00, 1.684622e+00, 1.680666e+00, 1.676789e+00, 1.672987e+00, 1.669260e+00, 1.665604e+00, 1.662018e+00, 1.658499e+00, 1.655046e+00, 1.651656e+00, 1.648328e+00, 1.645060e+00};
    double log[100] = {1.645060e+00, 1.588025e+00, 1.541123e+00, 1.501920e+00, 1.468727e+00, 1.440329e+00, 1.415827e+00, 1.394534e+00, 1.375920e+00, 1.359563e+00, 1.345127e+00, 1.332338e+00, 1.320970e+00, 1.310836e+00, 1.301779e+00, 1.293666e+00, 1.286385e+00, 1.279838e+00, 1.273942e+00, 1.268625e+00, 1.263824e+00, 1.259484e+00, 1.255556e+00, 1.251999e+00, 1.248774e+00, 1.245848e+00, 1.243192e+00, 1.240779e+00, 1.238587e+00, 1.236593e+00, 1.234779e+00, 1.233128e+00, 1.231625e+00, 1.230256e+00, 1.229009e+00, 1.227872e+00, 1.226836e+00, 1.225891e+00, 1.225030e+00, 1.224244e+00, 1.223527e+00, 1.222873e+00, 1.222276e+00, 1.221731e+00, 1.221234e+00, 1.220780e+00, 1.220365e+00, 1.219987e+00, 1.219641e+00, 1.219325e+00, 1.219037e+00, 1.218774e+00, 1.218533e+00, 1.218314e+00, 1.218113e+00, 1.217930e+00, 1.217762e+00, 1.217609e+00, 1.217469e+00, 1.217341e+00, 1.217225e+00, 1.217118e+00, 1.217021e+00, 1.216931e+00, 1.216850e+00, 1.216776e+00, 1.216708e+00, 1.216646e+00, 1.216589e+00, 1.216537e+00, 1.216490e+00, 1.216446e+00, 1.216407e+00, 1.216370e+00, 1.216337e+00, 1.216307e+00, 1.216280e+00, 1.216254e+00, 1.216231e+00, 1.216210e+00, 1.216191e+00, 1.216173e+00, 1.216157e+00, 1.216142e+00, 1.216129e+00, 1.216117e+00, 1.216105e+00, 1.216095e+00, 1.216086e+00, 1.216077e+00, 1.216069e+00, 1.216062e+00, 1.216056e+00, 1.216050e+00, 1.216044e+00, 1.216039e+00, 1.216035e+00, 1.216031e+00, 1.216027e+00, 1.216023e+00};
    memcpy(mT_over_T_lin, lin, sizeof(mT_over_T_lin));
    memcpy(mT_over_T_log, log, sizeof(mT_over_T_log));
    break; }
  case 5: {
    double lin[100] = {-1.671396e+00, -1.650160e+00, -1.630453e+00, -1.612109e+00, -1.594983e+00, -1.578952e+00, -1.563910e+00, -1.549763e+00, -1.536430e+00, -1.523840e+00, -1.511930e+00, 3.296280e+00, 3.056096e+00, 2.901402e+00, 2.785367e+00, 2.692249e+00, 2.614526e+00, 2.547939e+00, 2.489815e+00, 2.438355e+00, 2.392283e+00, 2.350662e+00, 2.312778e+00, 2.278077e+00, 2.246120e+00, 2.216549e+00, 2.189073e+00, 2.163451e+00, 2.139479e+00, 2.116983e+00, 2.095818e+00, 2.075856e+00, 2.056987e+00, 2.039114e+00, 2.022153e+00, 2.006030e+00, 1.990678e+00, 1.976039e+00, 1.962058e+00, 1.948691e+00, 1.935892e+00, 1.923625e+00, 1.911853e+00, 1.900546e+00, 1.889674e+00, 1.879210e+00, 1.869131e+00, 1.859414e+00, 1.850038e+00, 1.840985e+00, 1.832237e+00, 1.823778e+00, 1.815592e+00, 1.807666e+00, 1.799987e+00, 1.792543e+00, 1.785322e+00, 1.778315e+00, 1.771510e+00, 1.764899e+00, 1.758473e+00, 1.752224e+00, 1.746145e+00, 1.740227e+00, 1.734465e+00, 1.728852e+00, 1.723382e+00, 1.718050e+00, 1.712849e+00, 1.707774e+00, 1.702822e+00, 1.697987e+00, 1.693265e+00, 1.688652e+00, 1.684143e+00, 1.679736e+00, 1.675426e+00, 1.671211e+00, 1.667087e+00, 1.663050e+00, 1.659099e+00, 1.655230e+00, 1.651440e+00, 1.647728e+00, 1.644089e+00, 1.640524e+00, 1.637028e+00, 1.633599e+00, 1.630237e+00, 1.626938e+00, 1.623702e+00, 1.620525e+00, 1.617407e+00, 1.614345e+00, 1.611338e+00, 1.608385e+00, 1.605484e+00, 1.602634e+00, 1.599833e+00, 1.597079e+00};
    double log[100] = {1.597079e+00, 1.548627e+00, 1.508229e+00, 1.474095e+00, 1.444940e+00, 1.419817e+00, 1.398011e+00, 1.378966e+00, 1.362245e+00, 1.347499e+00, 1.334442e+00, 1.322843e+00, 1.312507e+00, 1.303274e+00, 1.295007e+00, 1.287589e+00, 1.280921e+00, 1.274918e+00, 1.269506e+00, 1.264620e+00, 1.260203e+00, 1.256208e+00, 1.252589e+00, 1.249309e+00, 1.246334e+00, 1.243633e+00, 1.241180e+00, 1.238951e+00, 1.236924e+00, 1.235080e+00, 1.233403e+00, 1.231875e+00, 1.230484e+00, 1.229216e+00, 1.228061e+00, 1.227008e+00, 1.226049e+00, 1.225173e+00, 1.224375e+00, 1.223646e+00, 1.222982e+00, 1.222375e+00, 1.221822e+00, 1.221317e+00, 1.220855e+00, 1.220434e+00, 1.220050e+00, 1.219699e+00, 1.219378e+00, 1.219085e+00, 1.218818e+00, 1.218573e+00, 1.218350e+00, 1.218146e+00, 1.217960e+00, 1.217790e+00, 1.217634e+00, 1.217492e+00, 1.217363e+00, 1.217244e+00, 1.217136e+00, 1.217037e+00, 1.216946e+00, 1.216864e+00, 1.216788e+00, 1.216719e+00, 1.216656e+00, 1.216598e+00, 1.216546e+00, 1.216497e+00, 1.216453e+00, 1.216413e+00, 1.216376e+00, 1.216343e+00, 1.216312e+00, 1.216284e+00, 1.216259e+00, 1.216235e+00, 1.216214e+00, 1.216194e+00, 1.216176e+00, 1.216160e+00, 1.216145e+00, 1.216131e+00, 1.216119e+00, 1.216107e+00, 1.216097e+00, 1.216087e+00, 1.216079e+00, 1.216071e+00, 1.216063e+00, 1.216057e+00, 1.216051e+00, 1.216045e+00, 1.216040e+00, 1.216035e+00, 1.216031e+00, 1.216027e+00, 1.216024e+00, 1.216021e+00};
    memcpy(mT_over_T_lin, lin, sizeof(mT_over_T_lin));
    memcpy(mT_over_T_log, log, sizeof(mT_over_T_log));
    break; }
  case 6: {
    double lin[100] = {-1.589175e+00, -1.572062e+00, -1.556084e+00, -1.541126e+00, -1.527088e+00, -1.513885e+00, 3.320778e+00, 3.051891e+00, 2.886079e+00, 2.763858e+00, 2.666807e+00, 2.586410e+00, 2.517936e+00, 2.458452e+00, 2.406002e+00, 2.359212e+00, 2.317075e+00, 2.278830e+00, 2.243889e+00, 2.211785e+00, 2.182145e+00, 2.154660e+00, 2.129078e+00, 2.105185e+00, 2.082802e+00, 2.061775e+00, 2.041973e+00, 2.023282e+00, 2.005601e+00, 1.988844e+00, 1.972934e+00, 1.957802e+00, 1.943388e+00, 1.929639e+00, 1.916505e+00, 1.903943e+00, 1.891913e+00, 1.880380e+00, 1.869312e+00, 1.858678e+00, 1.848453e+00, 1.838611e+00, 1.829129e+00, 1.819987e+00, 1.811166e+00, 1.802649e+00, 1.794417e+00, 1.786458e+00, 1.778755e+00, 1.771298e+00, 1.764072e+00, 1.757067e+00, 1.750273e+00, 1.743679e+00, 1.737276e+00, 1.731056e+00, 1.725010e+00, 1.719131e+00, 1.713411e+00, 1.707844e+00, 1.702423e+00, 1.697143e+00, 1.691997e+00, 1.686981e+00, 1.682089e+00, 1.677316e+00, 1.672658e+00, 1.668111e+00, 1.663670e+00, 1.659332e+00, 1.655092e+00, 1.650948e+00, 1.646896e+00, 1.642932e+00, 1.639055e+00, 1.635260e+00, 1.631545e+00, 1.627908e+00, 1.624346e+00, 1.620856e+00, 1.617436e+00, 1.614085e+00, 1.610799e+00, 1.607577e+00, 1.604417e+00, 1.601318e+00, 1.598276e+00, 1.595291e+00, 1.592362e+00, 1.589485e+00, 1.586660e+00, 1.583886e+00, 1.581161e+00, 1.578483e+00, 1.575852e+00, 1.573266e+00, 1.570723e+00, 1.568224e+00, 1.565766e+00, 1.563348e+00};
    double log[100] = {1.563348e+00, 1.520567e+00, 1.484563e+00, 1.453910e+00, 1.427569e+00, 1.404755e+00, 1.384868e+00, 1.367435e+00, 1.352082e+00, 1.338505e+00, 1.326456e+00, 1.315730e+00, 1.306156e+00, 1.297589e+00, 1.289908e+00, 1.283007e+00, 1.276797e+00, 1.271200e+00, 1.266150e+00, 1.261587e+00, 1.257460e+00, 1.253723e+00, 1.250338e+00, 1.247267e+00, 1.244480e+00, 1.241950e+00, 1.239651e+00, 1.237561e+00, 1.235659e+00, 1.233929e+00, 1.232355e+00, 1.230921e+00, 1.229614e+00, 1.228424e+00, 1.227339e+00, 1.226350e+00, 1.225448e+00, 1.224626e+00, 1.223875e+00, 1.223191e+00, 1.222566e+00, 1.221996e+00, 1.221475e+00, 1.221000e+00, 1.220567e+00, 1.220171e+00, 1.219809e+00, 1.219479e+00, 1.219177e+00, 1.218902e+00, 1.218650e+00, 1.218420e+00, 1.218210e+00, 1.218019e+00, 1.217843e+00, 1.217683e+00, 1.217537e+00, 1.217403e+00, 1.217281e+00, 1.217170e+00, 1.217068e+00, 1.216975e+00, 1.216890e+00, 1.216812e+00, 1.216741e+00, 1.216676e+00, 1.216616e+00, 1.216562e+00, 1.216513e+00, 1.216467e+00, 1.216426e+00, 1.216388e+00, 1.216358e+00, 1.216322e+00, 1.216293e+00, 1.216267e+00, 1.216242e+00, 1.216220e+00, 1.216200e+00, 1.216182e+00, 1.216165e+00, 1.216150e+00, 1.216136e+00, 1.216123e+00, 1.216111e+00, 1.216100e+00, 1.216090e+00, 1.216081e+00, 1.216073e+00, 1.216066e+00, 1.216059e+00, 1.216053e+00, 1.216047e+00, 1.216042e+00, 1.216037e+00, 1.216033e+00, 1.216029e+00, 1.216025e+00, 1.216022e+00, 1.216019e+00,};
    memcpy(mT_over_T_lin, lin, sizeof(mT_over_T_lin));
    memcpy(mT_over_T_log, log, sizeof(mT_over_T_log));
    break; }
  default:
    class_test(_TRUE_,
	       pba->error_message,
	       "The instability onset temperature for g_F = %u is not encoded yet",
	       pba->lrs_g_F);
  }


  double g_m0_over_M = pba->lrs_g_over_M * pba->lrs_m_F;
  double log10_g_m0_over_M = log10(g_m0_over_M);
  if (g_m0_over_M < min_g){ // Always stable
    *a_rel = 1e99;
    return _SUCCESS_;
  }
  else if (g_m0_over_M < max_g){ // Interpolate linearly
    // Points just before and after g_m0_over_M
    int idx_low = 99 * (g_m0_over_M - min_g) / (max_g - min_g);
    int idx_high = idx_low + 1;
    double g_low = min_g + idx_low * (max_g - min_g)/99.;
    double g_high = min_g + idx_high * (max_g - min_g)/99.;
    
    double delta = (g_m0_over_M - g_low) / (g_high - g_low);
    mT_over_T = mT_over_T_lin[idx_low] + delta * (mT_over_T_lin[idx_high] - mT_over_T_lin[idx_low]);
  } else if (log10_g_m0_over_M < max_log10g){ // Interpolate linearly
    // Points just before and after log10_g_m0_over_M
    int idx_low = 99 * (log10_g_m0_over_M - min_log10g) / (max_log10g - min_log10g);
    int idx_high = idx_low + 1;
    double log10g_low = min_log10g + idx_low * (max_log10g - min_log10g)/99.;
    double log10g_high = min_log10g + idx_high * (max_log10g - min_log10g)/99.;
    
    double delta = (log10_g_m0_over_M - log10g_low) / (log10g_high - log10g_low);
    mT_over_T = mT_over_T_log[idx_low] + delta * (mT_over_T_log[idx_high] - mT_over_T_log[idx_low]);
  } else{
    mT_over_T = mT_over_T_log[99];
  }

  if (mT_over_T < 0){ // Always stable
    *a_rel = 1e99;
    return _SUCCESS_;
  }

  /* Convert mT_over_T to scale factor 
     To this purpose, we have to compute m0/T given mT/T as a = (m0/T) / (m0/T0)
     The relationship between m0 and mT is given by the following system of equations
      | M phi = -g/M T^3 I_Mphi
      | mT = m0 + g phi      
      M/g (mT-m0) = -g/M T^3 I_Mphi
      m0/T = mT/T + g^2/M^2 T^2 I_Mphi
      m0/T = mT/T + g^2 m0^2/M^2 (T/m0)^2 I_Mphi
     Let x = T/m0
      1 = mT/T x + g^2 m0^2/M^2 I_Mphi x^3
     Let A = mT/T, B = g^2 m0^2/M^2 I_Mphi
      x^3 + A/B x - 1/B = 0
     We have to solve this cubic equation     
  */

  double I_Mphi;
  class_call(background_lrs_momenta(pba->q_lrs_bg,
				    pba->w_lrs_bg,
				    pba->q_size_lrs_bg,
				    mT_over_T,
				    pba->factor_lrs,
				    0,
				    NULL,
				    NULL,
				    &I_Mphi,
				    NULL,
				    NULL,
				    NULL),
             pba->error_message,
             pba->error_message);
  
  double A = mT_over_T;
  double B = SQR(pba->lrs_g_over_M * pba->lrs_m_F) * I_Mphi;
  double T_over_m0 = cbrt(1/B) * (cbrt(0.5 + sqrt(0.25 + CUB(A/3)/B)) +
				cbrt(-CUB(A/3)/B / (0.5 + sqrt(0.25 + CUB(A/3)/B)))); // The second term is cbrt(0.5 - sqrt(0.25 + CUB(A/3)/B))
  
  *a_rel = 1./T_over_m0 / pba->lrs_m_F_over_T0;
  
  return _SUCCESS_;
}
