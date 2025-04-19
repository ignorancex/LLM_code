#include "pluto.h"
#include <math.h>
#define pi 3.14159265
#define pc 3.086e18
#define sec_year 3.154e7
#define Msun 2.e33

int len=4000;
double Om=0.27;
double Ol=0.73;
double H0=70.e5/pc/1.e6;
double h=0.7;
double G=6.67e-8;
double peak_height=4.0;
double gam=4.0;
double bet=6.0;
double z_start=6.0;
double M0=1.e14*Msun;

float *zarray,*marray;

double Ez(double z)
{
    return sqrt( Om*pow((1.+z),3.)+(1.-Om) );
}

double Ezsqr(double z)
{
    return Om*pow((1.+z),3.)+(1.-Om);
}


double t(double z)
{
    double time=1./H0*2./3./sqrt(Ol)*log((sqrt(Ol/pow((1.+z),3.))+sqrt(Ol/pow(1.+z,3.)+Om))/sqrt(Om));
    return time;
}

double t_inv(double z,double t0)
{
    double t_start=t(z_start);
    return t(z)-t(z_start)-t0;
}

double z4()
{
    if(M0==1.e11*Msun) return 5.8332;
    else if(M0==1.e12*Msun) return 4.8747;
    else if(M0==1.e13*Msun) return 3.9257;
    else if(M0==1.e14*Msun) return 3.0139;
    else if(M0==5.e14*Msun) return 2.4212;
    else if(M0==1.e15*Msun) return 2.179;
}

double ct()
{
    double time=g_time*UNIT_LENGTH/UNIT_VELOCITY+t(z_start);
    double t04=t(z4());
    double c=4.*pow((1.+pow((time/3.4/t04),(6.5))),(1./8.));
    return c;
}

void init_array()
{
    FILE *f;
    int i,j;
    if(M0==1.e11*Msun) f=fopen("1e11_mah.dat","r");
    else if(M0==1.e12*Msun) f=fopen("1e12_mah.dat","r");
    else if(M0==1.e13*Msun) f=fopen("1e13_mah.dat","r");
    else if(M0==1.e14*Msun) f=fopen("1e14_mah.dat","r");
    else if(M0==5.e14*Msun) f=fopen("5e14_mah.dat","r");
    else if(M0==1.e15*Msun) f=fopen("1e15_mah.dat","r");
    float **array = (float **)malloc(len * sizeof(float *));
    for (i=0; i<len; i++)
    { 
         array[i] = (float *)malloc(8 * sizeof(float)); 
    } 
    
    for(i=0;i<len;i++)
    {
        for(j=0;j<8;j++)
        {
            fscanf(f, "%g", &array[i][j]);
        }
    }
    fclose(f);
    zarray=(float *)malloc(len*sizeof(float));
    marray=(float *)malloc(len*sizeof(float));
    for(i=0;i<len;i++)
    {
        zarray[i]=array[i][1];
        marray[i]=array[i][3];
    }
    free(array);
}

double M(double z)
{
    int low=0;
    int high=len-1;
    while(low!=high-1)
    {
        int mid=(low+high)/2;
        float zmid=zarray[mid];
        if(z<zmid) high=mid;
        else if(z>zmid) low=mid;
        else if(fabs(z-zmid)<1.e-5) break;
    }
    float dz=zarray[high]-zarray[low];
    float m=marray[low]*(zarray[high]-z)/dz+marray[high]*(z-zarray[low])/dz;
    return M0*pow(10.,m);
}

double solver(double t0)
{
    double left,right,mid,eps=0.001;
    double midval,rightval,root;
    left=0.;
    right=10.;
    do
    {
        mid=(left+right)/2.;
        rightval=t_inv(right,t0);
        midval=t_inv(mid,t0);
        if(rightval*midval>0) 
        {right=mid;}
        else {left=mid;}
    } while (right-left>eps);
    root=(right+left)/2.;
    return root;
}

double vir_radius()
{
    double z=solver(g_time*UNIT_LENGTH/UNIT_VELOCITY);
    double dcr0=3.*H0*H0/(8.*pi*G);
    double dcr=dcr0*Ezsqr(z);
    double M200=M(z);
    double dm=dcr0*Om*pow((1.+z),3.);
    double R200c = pow((3.*M200/(800.*pi*dcr)),(1./3.));
    return R200c;
}

double d_NFW(double r)
{
    double z=solver(g_time*UNIT_LENGTH/UNIT_VELOCITY);
    double dcr0=3.*H0*H0/(8.*pi*G);
    double dcr=dcr0*Ezsqr(z);
    double M200=M(z);
    double dm=dcr0*Om*pow((1.+z),3.);
    double R200c = pow((3.*M200/(800.*pi*dcr)),(1./3.));
    double R200m = pow((3.*M200/(800.*pi*dm)),(1./3.));
    double c=ct();
    double rs=R200c/c;
    return 200.*c*c*c*dcr/(3.*(log(1.+c)-c/(1.+c)))/(r/rs*pow((1.+r/rs),2.));
}

double f_tr(double r)
{
    double z=solver(g_time*UNIT_LENGTH/UNIT_VELOCITY);
    double dcr0=3.*H0*H0/(8.*pi*G);
    double dcr=dcr0*Ezsqr(z);
    double M200=M(z);
    double dm=dcr0*Om*pow((1.+z),3.);
    double R200c = pow((3.*M200/(800.*pi*dcr)),(1./3.));
    double R200m =pow((3.*M200/(800.*pi*dm)),(1./3.));
    double rt=(1.9-0.18*peak_height)*R200m; 
    return pow((1. + pow((r/rt),bet)),(-gam/bet));
}

double d_out(double r)
{
    double z=solver(g_time*UNIT_LENGTH/UNIT_VELOCITY);
    double se=1.5-z/6.*0.2;
    double be=1.2+z/6.*3.5;
    double dcr0=3.*H0*H0/(8.*pi*G);
    double dcr=dcr0*Ezsqr(z);
    double M200=M(z);
    double dm=dcr0*Om*pow((1.+z),3.);
    double R200c = pow((3.*M200/(800.*pi*dcr)),(1./3.));
    double R200m =pow((3.*M200/(800.*pi*dm)),(1./3.));
    return dm*(be*pow((r/(5.*R200m)),(-se)) + 1.);
}

double d_rho(double r)
{
    double z=solver(g_time*UNIT_LENGTH/UNIT_VELOCITY);
    double dcr0=3.*H0*H0/(8.*pi*G);
    double dcr=dcr0*Ezsqr(z);
    double dm=dcr0*Om*pow((1.+z),3.);
    return d_NFW(r)*f_tr(r)+d_out(r);
}

double phi_nfw(double r)
{
    double z=solver(g_time*UNIT_LENGTH/UNIT_VELOCITY);
    double dcr0=3.*H0*H0/(8.*pi*G);
    double dcr=dcr0*Ezsqr(z);
    double M200=M(z);
    double dm=dcr0*Om*pow((1.+z),3.);
    double R200c = pow((3.*M200/(800.*pi*dcr)),(1./3.));
    double R200m =pow((3.*M200/(800.*pi*dm)),(1./3.));
    double x=r/R200c;
    double c=ct();
    return -G*M200*log(1.+c*x)/(r*(log(1.+c)-c/(1.+c)));
}

double phi_outer(double r)
{
    double z=solver(g_time*UNIT_LENGTH/UNIT_VELOCITY);
    double se=1.5-z/6.*0.2;
    double be=1.2+z/6.*3.5;
    double dcr0=3.*H0*H0/(8.*pi*G);
    double dcr=dcr0*Ezsqr(z);
    double M200=M(z);
    double dm=dcr0*Om*pow((1.+z),3.);
    double R200c = pow((3.*M200/(800.*pi*dcr)),(1./3.));
    double R200m =pow((3.*M200/(800.*pi*dm)),(1./3.));
    return (4.*pi*G)*dm*((be*pow((5.*R200m),(se))*(pow(r,(2.-se))/((3.-se)*(2.-se)))+pow(r,2.)/6.));
}

double sigmoid(double x,double ce)
{
    return erf(x*ce);
}

double phi_full(double r)
{
    double z=solver(g_time*UNIT_LENGTH/UNIT_VELOCITY);
    double dcr0=3.*H0*H0/(8.*pi*G);
    double dcr=dcr0*Ezsqr(z);
    double M200=M(z);
    double dm=dcr0*Om*pow((1.+z),3.);
    double R200c = pow((3.*M200/(800.*pi*dcr)),(1./3.));
    double R200m =pow((3.*M200/(800.*pi*dm)),(1./3.));
    double rt=(1.9-0.18*peak_height)*R200m;
    double w1 =1.-(1.+sigmoid(log10(r/rt),1.4))/2.;
    double w2 =1.-w1;
    double phi_fit = phi_nfw(r)*w1+(phi_outer(r)-2./3.*pi*G*dm*r*r)*w2;
    return phi_fit;
}

double potential(double r)
{
    double z=solver(g_time*UNIT_LENGTH/UNIT_VELOCITY);
    double dcr0=3.*H0*H0/(8.*pi*G);
    double dcr=dcr0*Ezsqr(z);
    double M200=M(z);
    double dm=dcr0*Om*pow((1.+z),3.);
    double R200c = pow((3.*M200/(800.*pi*dcr)),(1./3.));

    double Omz=Om*pow((1.+z),3.)/Ezsqr(z);
    double phi_cosm=-H0*H0*Ezsqr(z)*(1.-3.*Omz/2.)*(r*r)/2.;
    double phi_grav=phi_full(r);
    return (phi_grav+phi_cosm);
}

double K()
{
    double ft=1.16e7;
    double K0=2.*pow((M(6.)/M0),2./3.);
    double mu=0.62;
    double mue=1.17;
    double res=K0*ft*CONST_kB/((mu*CONST_mp)*pow((mue*CONST_mp),g_gamma-1.));
    return res;
}

/* ********************************************************************* */
void Init (double *v, double x1, double x2, double x3)
/*
 *
 *********************************************************************** */
{
    init_array();
    M_BH=1.e6*CONST_Msun;

    double z=solver(g_time*UNIT_LENGTH/UNIT_VELOCITY);
    double dcr0=3.*H0*H0/(8.*pi*G);
    double dcr=dcr0*Ezsqr(z);
    double M200=M(z);
    double dm=dcr0*Om*pow((1.+z),3.);
    double R200c = pow((3.*M200/(800.*pi*dcr)),(1./3.));
    double xtrans=R200c/UNIT_LENGTH;

    double factor;
    if(x1<xtrans) factor=1;
    else factor=exp(-5*(x1/xtrans-1.));
    double T=1.e5+1.e7*factor;

    double cs=sqrt(g_gamma*CONST_kB*T/CONST_mp);
    double x1cgs=x1*UNIT_LENGTH;
    if(x1cgs<0) x1cgs=0.01*UNIT_LENGTH;  
    v[VX1]=(H0*Ez(z_start))*(x1cgs)/UNIT_VELOCITY;
    v[RHO]=0.2*d_rho(x1cgs)/UNIT_DENSITY;
    v[PRS]=v[RHO]*cs*cs/(UNIT_VELOCITY*UNIT_VELOCITY);
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
/* 
 *
 *
 *********************************************************************** */
{
  int i,j,k;
    double mdot_tot, mdot_cold, mtot, mcold, mwarm, mdot_acc, mdot_in,*dx1, *x1, *dx2, *x2, *dx3, *x3, dvol;
    double sendarray[6], recvarray[6];
    mdot_tot=0;
    mdot_cold=0;
    mtot=0;
    mcold=0;
    mwarm=0;
    mdot_acc=0.;
    mdot_in=0.;
    dx1 = grid->dx[IDIR];
    x1=grid->x[IDIR];
    dx2 = grid->dx[JDIR];
    x2=grid->x[JDIR];
    dx3 = grid->dx[KDIR];
    x3=grid->x[KDIR];
    
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
                mdot_tot += x1[iboundary]*x1[iboundary]*sin(x2[j])*dx2[j]*dx3[k]*d->Vc[RHO][k][j][iboundary]*d->Vc[VX1][k][j][iboundary];
                if(T<1.e5) mdot_cold += x1[iboundary]*x1[iboundary]*sin(x2[j])*dx2[j]*dx3[k]*d->Vc[RHO][k][j][iboundary]*d->Vc[VX1][k][j][iboundary];
                if(T<1.e5) mdot_in += x1[iboundary]*x1[iboundary]*sin(x2[j])*dx2[j]*dx3[k]*d->Vc[RHO][k][j][iboundary]*d->Vc[VX1][k][j][iboundary];  
            }
    }}

    #ifdef PARALLEL
    }
    #endif

    DOM_LOOP(k,j,i)
    {
        if(x2_glob[j]>theta_bound && x2_glob[j]<3.14-theta_bound)
        {
            double mu  = MeanMolecularWeight(d->Vc);
            double T=d->Vc[PRS][k][j][i]/d->Vc[RHO][k][j][i]*KELVIN*mu;
            dvol = x1[i]*x1[i]*sin(x2[j])*dx1[i]*dx2[j]*dx3[k];
            if(x1[i]*UNIT_LENGTH<5.e3*CONST_pc) mtot+=d->Vc[RHO][k][j][i]*dvol;
            if(T<1.e5 && x1[i]*UNIT_LENGTH<5.e3*CONST_pc) mcold+=d->Vc[RHO][k][j][i]*dvol;
            if(T>1.e5 && T<=1.e8 && x1[i]*UNIT_LENGTH<5.e3*CONST_pc) mwarm+=d->Vc[RHO][k][j][i]*dvol;
        }
    }
     #ifdef PARALLEL
     sendarray[0] = mdot_tot;
     sendarray[1] = mdot_cold;
     sendarray[2] = mtot;
     sendarray[3] = mcold;
     sendarray[4] = mwarm;
     sendarray[5] = mdot_in;
     MPI_Allreduce (sendarray, recvarray, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     mdot_tot = recvarray[0];
     mdot_cold = recvarray[1];
     mtot = recvarray[2];
     mcold = recvarray[3];
     mwarm = recvarray[4];
     mdot_in = recvarray[5];
     #endif

    double tau_dep=200.0*3.154e7*1.e6/(UNIT_LENGTH/UNIT_VELOCITY);
    double eps_m=0.001;
    double eps_r=0.1;
    
    mdot_acc=mdot_in*eps_m;
    mdot_acc = MAX(0.0, -mdot_acc); 
    double Mdot_edd=4*CONST_PI*CONST_G*M_BH*CONST_mp/(eps_r*CONST_sigmaT*CONST_c);
    Mdot_edd=Mdot_edd/(UNIT_DENSITY*UNIT_LENGTH*UNIT_LENGTH*UNIT_VELOCITY);
    mdot_acc = MIN(mdot_acc, Mdot_edd);


    if (prank == 0){
    char fname[512];
    static double tpos = -1.0;
    FILE *fp;
    sprintf (fname, "%s/analysis.dat",RuntimeGet()->output_dir);
    if (g_stepNumber == 0){ /* Open for writing only when we’re starting */
    fp = fopen(fname,"w"); /*
    from beginning */
    fprintf (fp,"# %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s\n", "t", "dt", "mdot_in", "mdot_acc", "mdot_edd", "mdot_tot", "mdot_cold", "m_tot", "m_cold", "m_warm", "mdot_jet", "TEdot_jet", "M_BH");
    }else{
    /* Append if this is not step 0 */
    if (tpos < 0.0){ /* Obtain time coordinate of to last written row */
    char
    sline[512];
    fp = fopen(fname,"r");
    while (fgets(sline, 512, fp)) {}
    sscanf(sline, "%lf\n",&tpos); /* tpos = time of the last written row */
    fclose(fp);
    }
    fp = fopen(fname,"a");
    }
    if (g_time > tpos){
    /* Write if current time if > tpos */
    fprintf (fp, "%12.6e %12.6e %12.6e %12.6e %12.6e %12.6e %12.6e %12.6e %12.6e %12.6e %12.6e %12.6e %12.6e \n",g_time, g_dt, mdot_in, mdot_acc, Mdot_edd, mdot_tot,mdot_cold, mtot, mcold, mwarm, mdot_jet, TEdot_jet, M_BH);
    }
    fclose(fp);
    }

}
/* ********************************************************************* */
void UserDefBoundary (const Data *d, RBox *box, int side, Grid *grid) 
/*
 *
 *********************************************************************** */
{ 
  int i,j,k;
  double mu  = MeanMolecularWeight(d->Vc);
  if (side == 0)
  {
    RBox dom_box;
    TOT_LOOP(k,j,i)
    {
      int convert_to_cons = 0;
      double T=d->Vc[PRS][k][j][i]/d->Vc[RHO][k][j][i]*KELVIN*mu;
      if(T<=2.e4)
      {
          T=2.e4;
          d->Vc[PRS][k][j][i]=T*d->Vc[RHO][k][j][i]/KELVIN/mu;
          convert_to_cons = 1;
      }
      else if(T>=1.e9)
      {
          T=1.e9;
          d->Vc[PRS][k][j][i]=T*d->Vc[RHO][k][j][i]/KELVIN/mu;
          convert_to_cons = 1;
      }
      if (convert_to_cons) 
      {
        RBoxDefine (i, i, j, j, k, k, CENTER, &dom_box);
        PrimToCons3D(d->Vc, d->Uc, &dom_box);
      }
    } /* DOM_LOOP() */
  } /* if (side == 0) */

    double  *x1, *x2, *x3, *dx1, *dx2, *dx3;
    double DM_POT[NX1_TOT];
    dx1 = grid->dx[IDIR];
    x1=grid->x[IDIR];
    dx2 = grid->dx[JDIR];
    x2=grid->x[JDIR];
    dx3 = grid->dx[KDIR];
    x3=grid->x[KDIR];
 
    ITOT_LOOP(i)
    {
        double r;
        if(x1[i]>0) r=x1[i];
        else r=0.01;
        DM_POT[i] = potential(r*UNIT_LENGTH)/(UNIT_VELOCITY*UNIT_VELOCITY);
    }
   

  if (side == X1_BEG)
  {  
    if (box->vpos == CENTER) 
    {
	#ifdef PARALLEL
            if (grid->rank_coord[0] == 0){
        #endif       
        BOX_LOOP(box,k,j,i)
        {
            d->Vc[RHO][k][j][i] = d->Vc[RHO][k][j][i+1];
            if( d->Vc[VX1][k][j][IBEG] < 0.0)
            {
            DIM_EXPAND( d->Vc[VX1][k][j][i] = d->Vc[VX1][k][j][i+1] ;  , 
                    d->Vc[VX2][k][j][i] = d->Vc[VX2][k][j][i+1] ;  ,
                    d->Vc[VX3][k][j][i] = d->Vc[VX3][k][j][i+1] ; )
            }
            else
            {
            DIM_EXPAND(d->Vc[VX1][k][j][i] = 0.0 ;,
                    d->Vc[VX2][k][j][i] = 0.0 ;,
                    d->Vc[VX3][k][j][i] = 0.0 ;)
            }
            d->Vc[PRS][k][j][i] = d->Vc[PRS][k][j][i+1] + (d->Vc[RHO][k][j][i+1])*(DM_POT[i+1]-DM_POT[i]);
        }
	#ifdef PARALLEL
        }
        #endif
    }
  }

  if (side == X1_END)
  {  
    if (box->vpos == CENTER) 
    {
	#ifdef PARALLEL
            if (grid->rank_coord[0] == grid->nproc[IDIR]-1){
        #endif 
        double z=solver(g_time*UNIT_LENGTH/UNIT_VELOCITY);
        BOX_LOOP(box,k,j,i)
        {
            d->Vc[RHO][k][j][i] = d->Vc[RHO][k][j][i-1];
            d->Vc[PRS][k][j][i] = d->Vc[PRS][k][j][i-1];
        }
	#ifdef PARALLEL
        }
        #endif
    }
  }
}


void BodyForceVector(double *v, double *g, double x1, double x2, double x3)
{
    double dr=pc;
    double x1cgs=x1*UNIT_LENGTH;
    double z=solver(g_time*UNIT_LENGTH/UNIT_VELOCITY);
    double dcr0=3.*H0*H0/(8.*pi*G);
    double dcr=dcr0*Ezsqr(z);
    double M200=M(z);
    double dm=dcr0*Om*pow((1.+z),3.);
    double R200c = pow((3.*M200/(800.*pi*dcr)),(1./3.));
    double R200m =pow((3.*M200/(800.*pi*dm)),(1./3.));
    g[IDIR]=-((potential(x1cgs+dr)-potential(x1cgs))/dr)/(UNIT_VELOCITY*UNIT_VELOCITY/UNIT_LENGTH);
    g[JDIR]=0;
    g[KDIR]=0;  
}
