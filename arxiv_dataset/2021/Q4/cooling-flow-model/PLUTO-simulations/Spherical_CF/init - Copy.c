/* ///////////////////////////////////////////////////////////////////// */
/*! 
  \file  
  \brief Contains basic functions for problem initialization.

  The init.c file collects most of the user-supplied functions useful 
  for problem configuration.
  It is automatically searched for by the makefile.

  \author A. Mignone (mignone@ph.unito.it)
  \date   March 5, 2017
*/
/* ///////////////////////////////////////////////////////////////////// */
#include "pluto.h"
#include <math.h>
#include "coolit.h"
#include <stdbool.h>
//#include <gsl.h>

double mu, Rcl;
double rho0, P0, cs0, v0, s0, cool_0, tcool0, r_crit;
  
int q = 2, counter = 0;
double T0 , Mach0, n0, chi, singular;

/* ********************************************************************* */
void Init (double *v, double x1, double x2, double x3)
/*! 
 * The Init() function can be used to assign initial conditions as
 * as a function of spatial position.
 *
 * \param [out] v   a pointer to a vector of primitive variables
 * \param [in] x1   coordinate point in the 1st dimension
 * \param [in] x2   coordinate point in the 2nd dimension
 * \param [in] x3   coordinate point in the 3rdt dimension
 *
 * The meaning of x1, x2 and x3 depends on the geometry:
 * \f[ \begin{array}{cccl}
 *    x_1  & x_2    & x_3  & \mathrm{Geometry}    \\ \noalign{\medskip}
 *     \hline
 *    x    &   y    &  z   & \mathrm{Cartesian}   \\ \noalign{\medskip}
 *    R    &   z    &  -   & \mathrm{cylindrical} \\ \noalign{\medskip}
 *    R    & \phi   &  z   & \mathrm{polar}       \\ \noalign{\medskip}
 *    r    & \theta & \phi & \mathrm{spherical} 
 *    \end{array}
 *  \f]
 *
 * Variable names are accessed by means of an index v[nv], where
 * nv = RHO is density, nv = PRS is pressure, nv = (VX1, VX2, VX3) are
 * the three components of velocity, and so forth.
 *
 *********************************************************************** */
{
  
  T0 = g_inputParam[T_CRIT]; //K
  Mach0 = g_inputParam[MACH_CRIT];
  n0 = g_inputParam[N_CRIT]; //cm^-3
  chi = g_inputParam[CHI]; 
  singular = g_inputParam[SINGULAR];

  mu = MeanMolecularWeight(v);
  rho0 = n0*mu*CONST_mp; //g cm^-3
  P0 = n0*CONST_kB*T0; //dyne cm^-2
  cs0 = sqrt(g_gamma*P0/rho0); //cm s^-1
  v0 = -Mach0*cs0; //cm s^-1
  s0 = P0/pow(rho0,g_gamma); //CGS
  cool_0 = lambda(T0); //CGS
  tcool0 = (1./(g_gamma-1))*CONST_kB*T0/(n0*cool_0); //s
  
  r_crit = -q*v0*g_gamma*tcool0; //cm
  Rcl = singular*r_crit/UNIT_LENGTH; // code units (pc usually)
  
  //double steep=20., f=0.007;
  //double shift = steep*atanh(2.*f*Tcl/(Tw-Tcl)-1);
  if (prank==0 && counter==0) printf("T0=%.1e\tM0=%.1f\tn0=%.2f\tchi=%d\tsingular=%.2f\n",T0,Mach0, n0, (int)chi, singular);
  if (prank==0 && counter==0) printf("rho0=%.1e\t\tP0=%.1e\t\tv0=%.2e\t\tr_crit=%.2e pc\tRcl=%.2e pc\t\ttcool0=%.2e\n",rho0/UNIT_DENSITY,P0/(UNIT_DENSITY*pow(UNIT_VELOCITY,2)), v0/UNIT_VELOCITY, r_crit/UNIT_LENGTH, Rcl, tcool0/(CONST_PI*1e12));
  if (prank==0 && counter==0) printf("x from %.2f to %.2f\n",g_domBeg[IDIR],g_domEnd[IDIR],x1);
  if (prank==0 && (x1<g_domBeg[IDIR] || x1>g_domEnd[IDIR])) printf("%.2f pc\n",x1);
  counter++;
  
  double T = (x1/Rcl<1.)?T0/chi:T0; //Tcl + (Tw-Tcl)/2. *(1+ tanh(x1/steep+shift));
  //T = Tcl + (chi-1.)*Tcl*(x1-g_domBeg[IDIR])/(g_domEnd[IDIR]-g_domBeg[IDIR]); //a linear temperature going from cooler to hotter
  v[RHO] = (x1/Rcl<1.)?chi*n0*mu*CONST_mp/UNIT_DENSITY:n0*mu*CONST_mp/UNIT_DENSITY;
  EXPAND(
  v[VX1] = 0.0;,
  v[VX2] = 0.0;,
  v[VX3] = 0.0;);
  
  //#if HAVE_ENERGY
  v[PRS] = P0/(UNIT_DENSITY*pow(UNIT_VELOCITY,2));
  //#endif
  v[TRC] = 1.0;
  //Pwind = P/(UNIT_DENSITY*UNIT_VELOCITY*UNIT_VELOCITY);
  //Dwind = ncl*mu*mp/chi/UNIT_DENSITY;

  #if PHYSICS == MHD || PHYSICS == RMHD
  v[BX1] = 0.0;
  v[BX2] = 0.0;
  v[BX3] = 0.0;

  v[AX1] = 0.0;
  v[AX2] = 0.0;
  v[AX3] = 0.0;
  #endif
}

/* ********************************************************************* */
void InitDomain (Data *d, Grid *grid)
/*! 
 * Assign initial condition by looping over the computational domain.
 * Called after the usual Init() function to assign initial conditions
 * on primitive variables.
 * Value assigned here will overwrite those prescribed during Init().
 *
 *
 *********************************************************************** */
{
}

/* ********************************************************************* */
void Analysis (const Data *d, Grid *grid)
/*! 
 *  Perform runtime data analysis.
 *
 * \param [in] d the PLUTO Data structure
 * \param [in] grid   pointer to array of Grid structures  
 *
 *********************************************************************** */
{
	/*
	int i, j, k;
	double *dx, *dy, *dz;*/
	/* ---- Parallel data reduction ---- */
	/*
	#ifdef PARALLEL
	int transfer_size = 3;
	int transfer = 0;
	double sendArray [transfer_size], recvArray[transfer_size];*/
	
	/* ---- Set pointer shortcuts ---- */
	/*
	dx = grid->dx[IDIR];
	dy = grid->dx[JDIR];
	dz = grid->dx[KDIR];*/

}
#if PHYSICS == MHD
/* ********************************************************************* */
void BackgroundField (double x1, double x2, double x3, double *B0)
/*!
 * Define the component of a static, curl-free background 
 * magnetic field.
 *
 * \param [in] x1  position in the 1st coordinate direction \f$x_1\f$
 * \param [in] x2  position in the 2nd coordinate direction \f$x_2\f$
 * \param [in] x3  position in the 3rd coordinate direction \f$x_3\f$
 * \param [out] B0 array containing the vector componens of the background
 *                 magnetic field
 *********************************************************************** */
{
   B0[0] = 0.0;
   B0[1] = 0.0;
   B0[2] = 0.0;
}
#endif

/* ********************************************************************* */
void UserDefBoundary (const Data *d, RBox *box, int side, Grid *grid) 
/*! 
 *  Assign user-defined boundary conditions.
 *
 * \param [in,out] d  pointer to the PLUTO data structure containing
 *                    cell-centered primitive quantities (d->Vc) and 
 *                    staggered magnetic fields (d->Vs, when used) to 
 *                    be filled.
 * \param [in] box    pointer to a RBox structure containing the lower
 *                    and upper indices of the ghost zone-centers/nodes
 *                    or edges at which data values should be assigned.
 * \param [in] side   specifies the boundary side where ghost zones need
 *                    to be filled. It can assume the following 
 *                    pre-definite values: X1_BEG, X1_END,
 *                                         X2_BEG, X2_END, 
 *                                         X3_BEG, X3_END.
 *                    The special value side == 0 is used to control
 *                    a region inside the computational domain.
 * \param [in] grid  pointer to an array of Grid structures.
 *
 *********************************************************************** */
{
  int   i, j, k, nv;
  double  *x1, *x2, *x3;

  x1 = grid->x[IDIR];
  x2 = grid->x[JDIR];
  x3 = grid->x[KDIR];
  
  if (side == X1_BEG) { 

    BOX_LOOP(box,k,j,i) {
		d->Vc[VX1][k][j][i] = d->Vc[VX1][k][j][IBEG];   //outflow velocity
		d->Vc[RHO][k][j][i] = 3.*rho0/UNIT_DENSITY;
		d->Vc[PRS][k][j][i] = 1.52*P0/(UNIT_DENSITY*pow(UNIT_VELOCITY,2));
		//d->Vc[VX1][k][j][i] = 0.14*sqrt(g_gamma*d->Vc[PRS][k][j][i]/d->Vc[RHO][k][j][i]);
	}

  }else if (side == X1_END){

    BOX_LOOP(box,k,j,i) {
		d->Vc[VX1][k][j][i] = d->Vc[VX1][k][j][IEND]; //outflow velocity 
		d->Vc[RHO][k][j][i] = 0.02*rho0/UNIT_DENSITY;
		d->Vc[PRS][k][j][i] = 1.12*P0/(UNIT_DENSITY*pow(UNIT_VELOCITY,2));
		//d->Vc[VX1][k][j][i] = 0.*sqrt(g_gamma*d->Vc[PRS][k][j][i]/d->Vc[RHO][k][j][i]);
	}		

  }

  /*
  if (side == 0) {  */  /* -- check solution inside domain -- */
  /* TOT_LOOP(k,j,i){
	
    if ((x1[i]*UNIT_LENGTH/r_crit>=(1.-1e-1)) && (x1[i]*UNIT_LENGTH/r_crit<=(1.+1e-1))) {
		d->Vc[RHO][k][j][i] = rho0/UNIT_DENSITY;
		d->Vc[VX1][k][j][i] = v0/UNIT_VELOCITY;
		d->Vc[PRS][k][j][i] = P0/(UNIT_DENSITY*pow(UNIT_VELOCITY,2));
	}/*
	if (x1[i]/Rcl<1.)
		d->Vc[VX1][k][j][i] = 0.; *//*
   }
  }*/
}

#if BODY_FORCE != NO
/* ********************************************************************* */
void BodyForceVector(double *v, double *g, double x1, double x2, double x3)
/*!
 * Prescribe the acceleration vector as a function of the coordinates
 * and the vector of primitive variables *v.
 *
 * \param [in] v  pointer to a cell-centered vector of primitive 
 *                variables
 * \param [out] g acceleration vector
 * \param [in] x1  position in the 1st coordinate direction \f$x_1\f$
 * \param [in] x2  position in the 2nd coordinate direction \f$x_2\f$
 * \param [in] x3  position in the 3rd coordinate direction \f$x_3\f$
 *
 *********************************************************************** */
{
  g[IDIR] = 0.0;
  g[JDIR] = 0.0;
  g[KDIR] = 0.0;
}
/* ********************************************************************* */
double BodyForcePotential(double x1, double x2, double x3)
/*!
 * Return the gravitational potential as function of the coordinates.
 *
 * \param [in] x1  position in the 1st coordinate direction \f$x_1\f$
 * \param [in] x2  position in the 2nd coordinate direction \f$x_2\f$
 * \param [in] x3  position in the 3rd coordinate direction \f$x_3\f$
 * 
 * \return The body force potential \f$ \Phi(x_1,x_2,x_3) \f$.
 *
 *********************************************************************** */
{
  return 0.0;
}
#endif
