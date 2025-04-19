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
#include "local_pluto.h"
#include "math.h"
#include <stdbool.h>
//#include "gsl.h"

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
  g_minCoolingTemp = 1.e4; /**< The minimum temperature (in K) below which cooling is arrested >**/
  g_maxCoolingRate = 0.01;  /**< The maximum fractional variation due to cooling >**/
  g_gamma = 1.6667;

  double nhot   = g_inputParam[N_HOT]; //Enter Ncl in cgs , i.e., in cm-3
  double Thot   = g_inputParam[T_HOT];  
  double Tcold  = g_inputParam[T_COLD];
  
  double mu    = MeanMolecularWeight((double*)d->Vc);
  double Prs   = nhot*CONST_kB*Thot;
  double ncold = Prs/(CONST_kB*Tcold);
  
  print("mu = %.3f\tmue = %.3f\tmui = %.3f\n",g_mu, g_mue, g_mui);
    
  int i, j, k;
  DIM_EXPAND(
  double *x = grid->x[IDIR];,
  double *y = grid->x[JDIR];,
  double *z = grid->x[KDIR];)
  DIM_EXPAND(
  double xmid = 0.5*(grid->xend_glob[0] + grid->xbeg_glob[0]);,
  double ymid = 0.5*(grid->xend_glob[1] + grid->xbeg_glob[1]);,
  double zmid = 0.5*(grid->xend_glob[2] + grid->xbeg_glob[2]);)
  double xbeg = grid->xbeg_glob[0]-GetNghost()*grid->dx[IDIR][0];
  double xend = grid->xend_glob[0]+GetNghost()*grid->dx[IDIR][0];
  
  double cs, tcool;
  double cstcool=1e100, cstcool_all;
  double tcoolmin=1e100,   tcoolmin_all;
  
#if COOLING == TOWNSEND
  double dummy0[NVAR], dummy1[NVAR];
  Radiat (dummy0, dummy1, 1);
#endif
  print("Starting initialization ... \n");
  TOT_LOOP(k,j,i){
    d->Vc[RHO][k][j][i] = nhot*g_mu*CONST_mp/UNIT_DENSITY;
    d->Vc[PRS][k][j][i] = Prs/(UNIT_DENSITY*pow(UNIT_VELOCITY,2));
    if (x[i]>=-50. && x[i]<=50.) {
      d->Vc[RHO][k][j][i] = ncold*g_mu*CONST_mp/UNIT_DENSITY;
    } 
#if GEOMETRY == CARTESIAN
    d->Vc[VX1][k][j][i]= 0.;  
#else
    d->Vc[iVR][k][j][i]= 0.;
#endif

#if COOLING !=  NO   
    cs = sqrt(g_gamma*Prs/(nhot*g_mu*CONST_mp)); //CGS
    tcool = (Prs/(g_gamma-1.))/((pow(nhot*g_mu,2)/(g_mue*g_mui))*lambda_interp(Prs/(nhot*CONST_kB))); //CGS
    if (cs*tcool<cstcool) cstcool = cs*tcool;
    if (tcool<tcoolmin) tcoolmin = tcool;
#endif
  }
#if COOLING !=  NO    
  print("Starting additional startup analysis ... \n");
#ifdef PARALLEL
  int transfer_size = 2;
  int transfer = 0;
  double sendArray [transfer_size], recvArray[transfer_size];
  sendArray[transfer++]=cstcool; sendArray[transfer++]=tcoolmin;
  MPI_Allreduce (sendArray, recvArray, transfer_size, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  transfer = 0;
  cstcool_all=recvArray[transfer++]; tcoolmin_all=recvArray[transfer++]; 
#else
  cstcool_all  = cstcool;
  tcoolmin_all = tcoolmin;
#endif
  print("Resolution needed %.2e (code units)\n",(cstcool_all/UNIT_LENGTH) );
  print("Initial minimum cooling time = %.2e (code units)\n",(tcoolmin_all/(UNIT_LENGTH/UNIT_VELOCITY)) ); 
#endif
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
  int i,j,k, nv;
  DIM_EXPAND(
  double *x  = grid->x[IDIR];,
  double *y  = grid->x[JDIR];,
  double *z  = grid->x[KDIR];)
  double *dx = grid->dx[IDIR];
  DIM_EXPAND(
  double xmid = 0.5*(grid->xend_glob[0] + grid->xbeg_glob[0]);,
  double ymid = 0.5*(grid->xend_glob[1] + grid->xbeg_glob[1]);,
  double zmid = 0.5*(grid->xend_glob[2] + grid->xbeg_glob[2]);)
  double xbeg = grid->xbeg_glob[0]-GetNghost()*grid->dx[IDIR][0];
  double xend = grid->xend_glob[0]+GetNghost()*grid->dx[IDIR][0];
    
  double nhot   = g_inputParam[N_HOT]; //Enter Ncl in cgs , i.e., in cm-3
  double Thot   = g_inputParam[T_HOT];  
  
  double Prs  = nhot*CONST_kB*Thot;
/* 
  if (side == X1_END) {
    //NVAR_LOOP(nv) OutflowBoundary(d->Vc[nv], box, side);
    BOX_LOOP(box,k,j,i) {
      if (x[i]>=(grid->xend_glob[0]-dx[0])) {
        d->Vc[RHO][k][j][i] = nhot*g_mu*CONST_mp/UNIT_DENSITY;
        d->Vc[PRS][k][j][i] = Prs/(UNIT_DENSITY*pow(UNIT_VELOCITY,2));
#if GEOMETRY == CARTESIAN
        d->Vc[VX1][k][j][i] = d->Vc[VX1][k][j][IEND];
#else
        d->Vc[iVR][k][j][i] = d->Vc[iVR][k][j][IEND];
#endif
      }
    }
  }

  if (side == 0) {
    RBox dom_box;
    TOT_LOOP(k,j,i){
      int convert_to_cons = 0;
      if (x[i]>=(grid->xend_glob[0]-dx[IEND-1])) {
        d->Vc[RHO][k][j][i] = nhot*g_mu*CONST_mp/UNIT_DENSITY;
        d->Vc[PRS][k][j][i] = Prs/(UNIT_DENSITY*pow(UNIT_VELOCITY,2));
        convert_to_cons = 1;
      }
      if (convert_to_cons) {
        RBoxDefine (i, i, j, j, k, k, CENTER, &dom_box);
        PrimToCons3D(d->Vc, d->Uc, &dom_box);
      }
    } 
  }
*/
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

