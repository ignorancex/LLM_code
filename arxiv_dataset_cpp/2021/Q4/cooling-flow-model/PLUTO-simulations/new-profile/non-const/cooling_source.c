/* ///////////////////////////////////////////////////////////////////// */
/*!
  \file
  \brief Integrate cooling and reaction source terms.

  Solve the system of ordinary differential equations describing
  optically thin cooling and ionization network (if any).
  We use an adaptive step size integration method and follow the 
  strategy outlined in section 5.1 of Tesileanu et al. (2008) for handling
  stiff cells.

  On output, a time-step estimate for the next time level is computed using
  the relative or absolute variation obtained during the integration of the ODE 
  (from t(n) --> t(n+1))  
  \f[
     \Delta t_c = \min_{ijk}\left[\frac{\Delta t^n M_R}{\epsilon}\right]
     \,\quad\rm{where}\quad
     \epsilon = \max\left(\left|\frac{p^{n+1}}{p^n} - 1\right|,\,
                |X^{n+1}-X^n|\right)
  \f]
  where \f$ M_R \f$ is the maximum cooling rate (defined by the global variable  
  ::g_maxCoolingRate) and X are the chemical species.
  
  \b References
     - "Simulating radiative astrophysical flows with the PLUTO code:
        a non-equilibrium, multi-species cooling function" \n
       Tesileanu, Mignone and Massaglia, A&A (2008) 488, 429

  \authors A. Mignone (mignone@ph.unito.it)\n
           O. Tesileanu
           B. Vaidya
  \date    April 22, 2014
*/
/* ///////////////////////////////////////////////////////////////////// */
#include "pluto.h"
#include "local_pluto.h"
double *g_invT_tab, *g_invY_tab, *g_Y_tab, *g_L_tab, *g_T_tab;
int g_ntab;
double interp1D(double* , double* , double , char* );

/* ********************************************************************* */
void CoolingSource (const Data *d, double dt, timeStep *Dts, Grid *GXYZ)
/*!
 * Integrate cooling and reaction source terms.
 *
 * \param [in,out]  d   pointer to Data structure
 * \param [in]     dt   the time step to be taken
 * \param [out]    Dts  pointer to the timeStep structure
 * \param [in]    GXYZ  pointer to an array of Grid structures
 *
 *********************************************************************** */
{
  int  nv, k, j, i, stiff, status;
  double err, scrh, min_tol = 2.e-5;
  double T0, T1, prs, tcool;
  double Tref = 1.e9, tcool_ref= 0.;
  double v0[NVAR], v1[NVAR], k1[NVAR];
  double maxrate, dt_sub;
  int n_sub_max = 50;
  intList var_list;
  
  
/* --------------------------------------------------------
    Set number and indices of the time-dependent variables
   -------------------------------------------------------- */
  
  var_list.nvar    = NIONS+1;
  var_list.indx[0] = PRS;
  for (nv = 0; nv < NIONS; nv++) var_list.indx[nv+1] = NFLX+nv;
  
/* -----------------------------------------------------
    Zero k1
   ----------------------------------------------------- */
  
  NVAR_LOOP(nv) k1[nv] = 0.0;  

  DOM_LOOP(k,j,i){  /* -- span the computational domain to find minimum tcool and global code step gets modified accordingly-- */
   NVAR_LOOP(nv) v0[nv] = v1[nv] = d->Vc[nv][k][j][i];
   //mu0 = MeanMolecularWeight(v0);
   T0  = v0[PRS]/v0[RHO]*KELVIN*g_mu;
  #if EOS == IDEAL
   v0[RHOE] = v1[RHOE] = v0[PRS]/(g_gamma-1.0);
  #else
   v1[RHOE] = v0[RHOE] = InternalEnergy(v0, T0);
  #endif
   Radiat(v0, k1, 0); //only run to populate tabulated arrays from storage if first time and get the cooling rate at every run 
   tcool = -v0[RHOE]/k1[RHOE]; //v0[RHOE]=(prs/(g_gamma-1.0)) for ideal gas and k1[RHOE]=-neniLAMBDA (See radiat.c)
   Dts->dt_cool = MIN(Dts->dt_cool, tcool);
   maxrate = GetMaxRate (v0, k1, T0);
  }
/* -----------------------------------------------------
 *     Zero reinitialization
 * ----------------------------------------------------- */
  NVAR_LOOP(nv) { k1[nv] = 0.0; v0[nv] = 0.0; v1[nv] = 0.0; }
 #ifdef PARALLEL
  double dt_cool_thisproc = Dts->dt_cool, dt_cool_allproc = 0.;
  MPI_Allreduce(&dt_cool_thisproc, &dt_cool_allproc, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  Dts->dt_cool = dt_cool_allproc;
 #endif
  
/*  ----------------------------------------------------------- 
                   Begin Integration 
    -----------------------------------------------------------  */
 
/* -----------------------------------------------------
 *     Subcycling starts here
 * ----------------------------------------------------- */
  dt_sub = 1.0/(1.0/(Dts->dt_cool/n_sub_max) + 2./dt); //Subcycle limit
  int sub_steps = (int)ceil(dt/dt_sub);
  dt_sub = dt/sub_steps;
  int n_sub = 0; 
  for (n_sub = 0; n_sub<sub_steps; n_sub++ ){
    DOM_LOOP(k,j,i){  /* -- span the computational domain -- */
      /* --------------------------------------------------
         Skip integration if cell has been tagged with 
         FLAG_INTERNAL_BOUNDARY or FLAG_SPLIT_CELL 
         (only for AMR)
         -------------------------------------------------- */ 
    #if INTERNAL_BOUNDARY == YES
     if (d->flag[k][j][i] & FLAG_INTERNAL_BOUNDARY) continue;
    #endif
     if (d->flag[k][j][i] & FLAG_SPLIT_CELL) continue;
     
     /* ----------------------------------------------
       Compute temperature and internal energy from
       density, pressure and concentrations.
       ---------------------------------------------- */
    
     NVAR_LOOP(nv) v0[nv] = v1[nv] = d->Vc[nv][k][j][i]; //Initialize cell fields
     prs = v0[PRS];
     //mu0 = MeanMolecularWeight(v0);
     T0  = v0[PRS]/v0[RHO]*KELVIN*g_mu;
    #if EOS == IDEAL
     v0[RHOE] = v1[RHOE] = prs/(g_gamma-1.0);
    #else
     v1[RHOE] = v0[RHOE] = InternalEnergy(v0, T0);
    #endif
     
     if (T0 <= 0.0){
       print ("! CoolingSource: negative initial temperature\n");
       print (" %12.6e  %12.6e\n",v0[RHOE], v0[RHO]);
       print (" at: %f %f\n",GXYZ[IDIR].x[i], GXYZ[JDIR].x[j]);
       QUIT_PLUTO(1);
     }
         
     Radiat(v0, k1, 0); //only run to populate tabulated arrays from storage if first time and get the cooling rate at every run
     tcool = -v0[RHOE]/k1[RHOE]; //v0[RHOE]=(prs/(g_gamma-1.0)) for ideal gas and k1[RHOE]=-neniLAMBDA (See radiat.c)
     tcool_ref = (1./(g_gamma-1))*(CONST_kB*Tref/(v0[RHO]*UNIT_DENSITY*lambda_interp(Tref)))*(g_mue*g_mui/g_mu)*CONST_mp;
     tcool_ref /= (UNIT_LENGTH/UNIT_VELOCITY);
     T1 = invY_interp( Y_interp(T0) + (T0/Tref)* (lambda_interp(Tref)/lambda_interp(T0))* (dt_sub/tcool));
     //mu1 = MeanMolecularWeight(v1);
    #if EOS == IDEAL
     prs  = T1*v0[RHO]/(KELVIN*g_mu);
     v1[RHOE] = prs/(g_gamma - 1.0);
		#else
     v1[RHOE] = InternalEnergy(v1, T1);
    #endif
     /* -- pressure must be positive -- */
     if (prs < 0.0) prs = g_smallPressure;
     /* ------------------------------------------------------
      *       As a safety measure, constrain the ions to
      *               lie in [0,1] and pressure to be positive.
      * ------------------------------------------------------ */
     NIONS_LOOP(nv){
       v1[nv] = MAX(v1[nv], 0.0);
       v1[nv] = MIN(v1[nv], 1.0);
     }
     /* ------------------------------------------------------
      *       If gas is initially or cooled below lower threshold, redefine
      *               pressure so that T = g_minCoolingTemp.
      * ------------------------------------------------------ */
     if (T1 < g_minCoolingTemp || T0 < g_minCoolingTemp)  prs = g_minCoolingTemp*v1[RHO]/(KELVIN*g_mu);
     
     err = fabs(prs/d->Vc[PRS][k][j][i] - 1.0);
    #if COOLING == MINEq
     for (nv = NFLX; nv < NFLX + NIONS - Fe_IONS; nv++) 
    #else
     NIONS_LOOP(nv)  err = MAX(err, fabs(d->Vc[nv][k][j][i] - v1[nv]));
    #endif
     err = MAX(err, fabs(d->Vc[nv][k][j][i] - v1[nv]));
     
     scrh = dt*g_maxCoolingRate/err; //****Check this out later (max change in P/ Current change in P)
     
     Dts->dt_cool = MIN(Dts->dt_cool, scrh);
     
     /* ------------------------------------------------------
      *      1H. Update main solution array
      * ------------------------------------------------------ */
     d->Vc[PRS][k][j][i] = prs;
     NIONS_LOOP(nv) d->Vc[nv][k][j][i] = v1[nv];
    } /* -- End DOM_LOOP(k,j,i) -- */
  //print("Temp %e     step %d     time %e     tcool %e\n",T1,n_sub,g_time,Dts->dt_cool);
  }   /* -- End Subcycle Loop   -- */
}

double lambda_interp(double temperature) {
#if HIGHT_FLAT != NO
  temperature = temperature<1.e9?temperature:1.e9;
#endif
  return interp1D(g_T_tab, g_L_tab, temperature, "lambda_interp");
}

double Y_interp(double temperature) {
#if HIGHT_FLAT != NO
  temperature = temperature<1.e9?temperature:1.e9;
#endif
  return interp1D(g_T_tab, g_Y_tab, temperature, "Y_interp");
}

double invY_interp(double townY) {
  return interp1D(g_invY_tab, g_invT_tab, townY, "invY_interp");
}

double interp1D(double* x_data, double* y_data, double x_request, char* msg) {

 /* ----------------------------------------------
  *         Table lookup by binary search used for interpolating y as a function of x. 
  *        x is assumed to be arranged in ascending order
  *  ---------------------------------------------- */
  int ntab = g_ntab; // number of entries maybe (int)(sizeof(x_data)/sizeof(x_data[0])) will work
  int klo = 0;
  int khi = ntab - 1;
  int kmid;
  double xmid, dx, scrh;

  if (x_request > x_data[khi] || x_request < x_data[klo]){
    print ("Called from %s\n",msg);
    print (" ! Requested value out of range   %12.6e\n",x_request);
    QUIT_PLUTO(1);
  }

  while (klo != (khi - 1)){
    kmid = (klo + khi)/2;
    xmid = x_data[kmid];
    if (x_request <= xmid){
      khi = kmid;
    }else if (x_request > xmid){
      klo = kmid;
    }
  }

/* -----------------------------------------------
 *     Compute r.h.s
 * ----------------------------------------------- */

  dx       = x_data[khi] - x_data[klo];
  scrh     = y_data[klo]*(x_data[khi] - x_request)/dx + y_data[khi]*(x_request - x_data[klo])/dx;

  return scrh;
}
