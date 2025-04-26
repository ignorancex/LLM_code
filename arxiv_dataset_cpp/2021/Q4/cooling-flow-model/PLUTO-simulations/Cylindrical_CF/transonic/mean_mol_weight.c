/* ///////////////////////////////////////////////////////////////////// */
/*! 
  \file  
  \brief Compute the mean molecular weight for Tabulated Cooling. Simple version.
*/
/* ///////////////////////////////////////////////////////////////////// */
#include "pluto.h"
#include "local_pluto.h"

/* ********************************************************************* */
/*Solar metallicities taken from https://ui.adsabs.harvard.edu/abs/2009ARA%26A..47..481A/abstract ("Present work") */
double X = 0.7154, Y = 0.2703, Z = 0.0143;

double MeanMolecularWeight(double *v)
/*!
 *
 * Return the mean molecular weight.
 * 
 * \param [in]  v  array of primitive variables (including ions)
 *
 *********************************************************************** */
{  
  double mu = 1/(2*X+0.75*Y+0.5625*Z);
  return mu;
}

double* MeanElIonWeight(double *v)
/*!
 *
 * Return the electron and molecular weight.
 * 
 * \param [in]  v  array of primitive variables (including ions)
 *
 *********************************************************************** */
{  
  static double muei[2];
  double mu = MeanMolecularWeight(v);
  muei[0] = 2./(1+X);
  muei[1] = 1/(1/mu-1/muei[0]);
  return muei;
}
  