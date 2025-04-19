#define f_kin                       0.5  
#define eta_MLF                     1.0
#define eps_r                       0.1
#define eps_m			            0.001 
#define SIGR                        0.05
#define SIGTH                       0.05    
#define THJET                       30.0
#define ZJET                        10.0  

/* ///////////////////////////////////////////////////////////////////// */
/*! 
 *   \file  
 *     \brief Contains basic function for AGN jet implementation.
 *
 *       The source_jet.c file contains source function for jet implementation  
 *       useful for problem configuration.
 *       It is automatically searched for by the makefile.
 *
 *       \author Deovrat Prasad (deovrat@physics.iisc.ernet.in)
 *               \date   May 12, 2015
 *               */
/* ///////////////////////////////////////////////////////////////////// */
#include "pluto.h"
/* ********************************************************************* */
void Src_jet (const Data *d, Grid *grid)
{
    double M(double);
    double solver(double);
    double z=solver(g_time*UNIT_LENGTH/UNIT_VELOCITY);
    double M200=M(z);

    double RJETKPC=2.;  

    int i,j,k;
    double msub=0;
    double *x1, *x2, *x3, *dx1, *dx2, *dx3;
    double psi_jet, mdot_in,mdot_acc, darea, norm_jet, Srho, dr_cube, v_jet, Se;
    double sendarray[1], recvarray[1];
    double RJET, SIGRJ, THJ, d3, TinK, tau_dep=200.0*3.154e7*1.e6/(UNIT_LENGTH/UNIT_VELOCITY);
    double mu  = MeanMolecularWeight(d->Vc);

    // double M_BH=1.e9*CONST_Msun;
    // printf("%g \t",M_BH/CONST_Msun);
    double Mdot_edd=4*CONST_PI*CONST_G*M_BH*CONST_mp/(eps_r*CONST_sigmaT*CONST_c);
    // printf("%g \t",Mdot_edd/CONST_Msun*3.154e7);
    Mdot_edd=Mdot_edd/(UNIT_DENSITY*UNIT_LENGTH*UNIT_LENGTH*UNIT_VELOCITY);

    // x1=grid[IDIR].x; x2=grid[JDIR].x; x3=grid[KDIR].x;
    // dx1=grid[IDIR].dx; dx2=grid[JDIR].dx; dx3=grid[KDIR].dx;
    // x1=grid[IDIR].x;
    // dx1=grid[IDIR].dx;
    x1=grid->x[IDIR];
    dx1=grid->dx[IDIR]; 
    x2=grid->x[JDIR];
    dx2=grid->dx[JDIR];
    x3=grid->x[KDIR];
    dx3=grid->dx[KDIR];

    RJET = RJETKPC*CONST_pc*1.e3/UNIT_LENGTH; //jet region parameters, code unit 
    SIGRJ = SIGR*CONST_pc*1.e3/UNIT_LENGTH;
    THJ = THJET*CONST_PI/180.0;  

    dr_cube = pow(RJET,3.)-pow(g_domBeg[IDIR],3.);
    norm_jet = 3.0/(2.0*CONST_PI*dr_cube*(1.0-cos(THJ))); //inverse of volume of the jet region

/* ***************Applying the mass source************** */    
    mdot_acc =0.0;
    mdot_in=0.0;
    // printf("Working \n");
/* ************** Mass accretion through the inner radius*************** */

    double boundary=0.6*UNIT_LENGTH;
    double theta_bound=0.15;

    double *x1_glob=grid->x_glob[IDIR];
    double *x2_glob=grid->x_glob[JDIR];

    #ifdef PARALLEL
    if(grid->rank_coord[0]==0)
    {
    #endif
    KDOM_LOOP(k){
    JDOM_LOOP(j)
    {
            if(x2_glob[j]>theta_bound && x2_glob[j]<3.14-theta_bound)
            {
                int iboundary=IBEG;
                double mu  = MeanMolecularWeight(d->Vc);
                double T=d->Vc[PRS][k][j][iboundary]/d->Vc[RHO][k][j][iboundary]*KELVIN*mu;
                if(T<1.e5) mdot_in += x1[iboundary]*x1[iboundary]*sin(x2[j])*dx2[j]*dx3[k]*d->Vc[RHO][k][j][iboundary]*d->Vc[VX1][k][j][iboundary];
            }
      // }
     }}
     #ifdef PARALLEL 
     }

     sendarray[0] = mdot_in;
     MPI_Allreduce (sendarray, recvarray, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     mdot_in = recvarray[0];
     #endif

     mdot_acc=eps_m*mdot_in;
    // printf("%g \n", mdot_acc);
    //mdot_acc=-mdot_acc/tau_dep; 
    if (mdot_acc < 0.0){ //only inflowing mdot
      mdot_jet = -0.5*eta_MLF*mdot_acc; //single-jet mass loading, code units
      mdot_jet = MIN(mdot_jet,Mdot_edd); //limit with 0.1*MdotEdd for 1e9 BH
  }else{
      mdot_jet = 0.0;
    }
    //printf("%g\n",1.e26*0.5/(UNIT_DENSITY*UNIT_LENGTH*UNIT_LENGTH*UNIT_VELOCITY));

    mdot_acc = MAX(0.0, -mdot_acc); 
    
    mdot_acc = MIN(mdot_acc, Mdot_edd); //limit with 0.1*MdotEdd for 1e9 BH
    v_jet = sqrt(2.*eps_r*f_kin/eta_MLF)*CONST_c/UNIT_VELOCITY; //code units
    TEdot_jet = 0.5*(1.-f_kin)*eps_r*mdot_acc*(CONST_c/UNIT_VELOCITY)*(CONST_c/UNIT_VELOCITY); //code units, single-jet Edot_th
    // print("%20.10e %20.10e %20.10e %20.10e\n",v_jet,TEdot_jet, mdot_jet*v_jet*v_jet*(UNIT_DENSITY*UNIT_LENGTH*UNIT_LENGTH*UNIT_VELOCITY*UNIT_VELOCITY*UNIT_VELOCITY), EFF*mdot_acc*(UNIT_DENSITY*UNIT_LENGTH*UNIT_LENGTH*UNIT_VELOCITY)*CONST_c*CONST_c);

//     /* *********************************************************************** */
    
    M_BH = M_BH + mdot_acc*(UNIT_DENSITY*UNIT_LENGTH*UNIT_LENGTH*UNIT_VELOCITY)*g_dt*(UNIT_LENGTH/UNIT_VELOCITY);

    DOM_LOOP(k,j,i){
      psi_jet = 0.25*(2.0 + tanh((THJ -x2[j])/SIGTH) \
       + tanh((THJ + x2[j]- CONST_PI)/SIGTH)) \
       *(1.0 + tanh((RJET - x1[i])/SIGRJ));
      // psi_jet=1.;
       Srho = psi_jet*norm_jet*mdot_jet;
       Se = psi_jet*norm_jet*TEdot_jet;
     //  printf("%g \n",psi_jet);
      // printf("%g %g\n",d->Vc[PRS][k][j][i],g_dt*Se*(g_gamma-1.));
       d->Vc[RHO][k][j][i] += g_dt*Srho;
       d->Vc[PRS][k][j][i] += g_dt*Se*(g_gamma-1.);

       d->Vc[VX1][k][j][i] += g_dt*(v_jet-d->Vc[VX1][k][j][i])*Srho/d->Vc[RHO][k][j][i];

       d->Vc[VX2][k][j][i] -= g_dt*d->Vc[VX2][k][j][i]*Srho/d->Vc[RHO][k][j][i];

      /* d->Vc[VX3][k][j][i] -= g_dt*d->Vc[VX3][k][j][i]*Srho/d->Vc[RHO][k][j][i]; */ //uncomment for 3-D 

//       d->Vc[TRC][k][j][i] += g_dt*Srho/d->Vc[RHO][k][j][i];

      //  d->Vc[TRC][k][j][i] += g_dt*Srho*(ZJET-d->Vc[TRC][k][j][i])/d->Vc[RHO][k][j][i];

      //  TinK   = d->Vc[PRS][k][j][i]*mu*CONST_mp*UNIT_VELOCITY*UNIT_VELOCITY/(d->Vc[RHO][k][j][i]*CONST_kB);
      //  if (TinK < 5.e4 && d->Vc[RHO][k][j][i]*UNIT_DENSITY > 1.e-24){
      //    msub += x1[i]*x1[i]*sin(x2[j])*dx2[j]*dx3[k]*dx1[i]*d->Vc[RHO][k][j][i]*( 1.0- exp(-g_dt*UNIT_LENGTH/(tau_dep*UNIT_VELOCITY)) );
      //    d->Vc[RHO][k][j][i] = d->Vc[RHO][k][j][i]*exp(-g_dt/tau_dep);
      //  } 

    }

}
