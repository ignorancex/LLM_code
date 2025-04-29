//
//  Estimation of incubation time
//
//
//  Created by Piet Groeneboom on March-16-2021.
//  Copyright (c) 2020 Piet. All rights reserved.
//

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <time.h>
#include <random>
#include <fstream>
#include <string.h>
#include <Rcpp.h>

using namespace std;
using namespace Rcpp;

#define SQR(x) ((x)*(x))


typedef struct
{
    int ind;
    double alpha;
    double pp;
}
data_object;

int     compare (const void * a, const void * b);
int     compare2(const void *a, const void *b);
void    sort_index(int m, int ind[], double alpha[], double pp[]);
double  bdf(double A, double B, int m, double t[], double p[], double u, double h);
double  KK(double x);
double  K(double x);
double  dens_estimate(double A, double B,  int njumps, double jumploc[], double p[], double h, double u);

int     LUPDecompose(double **A, int N, double Tol, int *P);
void    LUPSolve(double **A, int *P, double *b, int N, double *x);

void    MLE_IQM(int ndata, int *m1, int ind[], int ngrid, double **data, double alpha[],
             double alpha0[], double pp[], double *phi_IQM, int *NumIt);
double  psi(double *data_point, double alpha);
void    Compute_nabla(int m, int ndata, double **data, double alpha[], double g[], double nabla[]);
int     fenchelviol(double tol, double min_nabla, double norm);
void    Compute_g(int m, int ndata, double pp[], double **data, double alpha[], double g[]);
void    Compute_DF(int m, double pp[], double DF[]);
double  phi_crit(int m, int ndata, double pp[], double g[]);
void    Lin_Comb(int ndata, double l, double g1[], double g2[], double g[]);
void    IterationLoop_IQM(int *m1, int *indexmin1, int *iteration1, double *minimum1, int ndata,
                          int ngrid, int ind[],double pp[], double pp0[], double **data,
                          double g[], double alpha[], double alpha0[], double w[]);
void    CheckPositivity_IQM(int *m1, int ndata, int ind[],
                                double pp[], double pp1[], double **data, double alpha[], double w[]);
                                
void    ComputeWeights_IQM(int m, int ndata, double **data, double alpha[], double pp[], double w[]);
void    Compute_critarray(int ndata, int ngrid, double **data, double g[], double alpha0[], double w[], double critarray[]);
double  Compute_minimum_IQM(int ndata, int ngrid, int *index,
                                    double **data, double g[], double alpha0[], double w[]);
void    Compute_W(int m, int ndata, double **data, double alpha[], double w[], double **W);
void    Compute_pp_IQM(int m, int ndata, double pp[],
                                    double **data, double alpha[], double w[]);

int     armijoviol1(int m, double eps, double phi_old, double phi_new,
                double nabla_old[], double p_old[], double p_new[]);
int     armijoviol2(int m, double eps, double phi_old, double phi_new,
                double nabla_old[], double p_old[], double p_new[]);
void    transfer(int n, double a[], double b[]);


// [[Rcpp::export]]

// input should contain columns V1, V2, V3
// h is the smoother bandwidth, default 4.0

List MLE(DataFrame input, double h)
{
    int     n,m,i,ngrid,*ind;
    int     iterations;
    double  *tt,**data;
    double  phi_IQM;
    double  *pp,*grid,*SMLE,*dens,*MLE;
    double  M1;
    
    DataFrame DF = Rcpp::DataFrame(input);
    NumericVector data1 = DF["V1"];
    NumericVector data2 = DF["V2"];
    NumericVector data3 = DF["V3"];
    
    // determine the sample size
    
    n = (int)data1.size();
     
    // M1 is upper bound (assumed) of incubation time
    // One can also make M1 dependent on computation of MLE
    // Only used in the computatiion of SMLE
    
    M1=20;
    
    // h is the bandwidth for either SMLE of density
    // h=4;
    
    Rcpp::Rcout << "bandwidth = " << h << endl << endl;
    Rcpp::Rcout << "Sample size n = " << n << endl << endl;
    
    // ngrid give number of points where confidence intervals are computed
    
    ngrid=200;
    
    // points where condidence intervals are computed
    grid = new double[ngrid+1];
    for (i=0;i<=ngrid;i++)
        grid[i] = 20*i*1.0/ngrid;
    
    // index array, used for indexing the generators of the solution
    ind = new int[ngrid+1];
    
    // value of criterion function
    phi_IQM=0;
    
    // pp is array containng the masses of the MLE to be computer, updated during the iterations
    pp = new double[ngrid+1];
    
    // tt is going to contain the points of mass
    tt = new double[ngrid+1];
    
    tt[0]=0;
    
    // arrays containing the SMLE and density
    SMLE = new double[ngrid+1];
    dens = new double[ngrid+1];
    
    // the data matrix contains three columns, corresponding to E, S_L and S_R
    
    data = new double *[n];
    for (i=0;i<n;i++)
        data[i]= new double[3];
    
    for (i=0;i<n;i++)
    {
        data[i][0]=(double)data1[i];
        data[i][1]=(double)data2[i];
        data[i][2]=(double)data3[i];
    }
    
    MLE = new double[ngrid+1];
    
    MLE[0]=pp[0]=0;
    
    //start looking for generators of the solution with just one generator
    m = 1;

    // IQM algorithm
    
    MLE_IQM(n,&m,ind,ngrid,data,tt,grid,pp,&phi_IQM,&iterations);
    
    Compute_DF(m,pp,MLE);
    
    NumericMatrix out1 = NumericMatrix(m+1,2);
    
    for (i=0;i<=m;i++)
    {
        out1(i,0)=tt[i];
        out1(i,1)=MLE[i];
    }
    
    NumericMatrix out2 = NumericMatrix(m+1,2);
    
    for (i=0;i<=m;i++)
    {
        out2(i,0)=tt[i];
        out2(i,1)=pp[i];
    }
    
    // SMLE is an smooth estimate of the distribution function
    // dens is an estimate of the density w.r.t. Lebesgue measure
    
    for (i=0;i<=ngrid;i++)
    {
        SMLE[i] = bdf(0,M1,m,tt,pp,grid[i],4);
        dens[i] = dens_estimate(0.0,M1,m,tt,pp,grid[i],h);
    }
    
    NumericMatrix out3 = NumericMatrix(ngrid+1,2);
    for (i=0;i<=ngrid;i++)
    {
        out3(i,0)=grid[i];
        out3(i,1)=SMLE[i];
    }
    
    NumericMatrix out4 = NumericMatrix(ngrid+1,2);
    for (i=0;i<=ngrid;i++)
    {
        out4(i,0)=grid[i];
        out4(i,1)=dens[i];
    }
    
    // make the list for the output
    
    List out = List::create(Rcpp::Named("MLE")=out1,Rcpp::Named("masses")=out2,Rcpp::Named("SMLE")=out3,Rcpp::Named("density")=out4);
       
    // free memory
    
    delete[] grid; delete[] SMLE; delete[] dens;
    
    for (i=0;i<n;i++) delete[] data[i];
    delete[] data;
    
    delete[] pp; delete[] tt;  delete[] MLE;
    
    return out;
}

void    IterationLoop_IQM(int *m1, int *indexmin1, int *iteration1, double *minimum1, int ndata,
                          int ngrid, int ind[],double pp[], double pp0[], double **data,
                          double g[], double alpha[], double alpha0[], double w[])
{
    int       m,indexmin,iteration;
    double    minimum;
    
    m=*m1;
    indexmin= *indexmin1;
    iteration=*iteration1;
    minimum = *minimum1;
    
    m+=1;
    iteration+=1;
    
    ind[m] = indexmin;
    alpha[m]=alpha0[indexmin];
    
    if (m>1)
    {
        pp[m]=0;
        sort_index(m,ind,alpha,pp);
    }
    
    Compute_pp_IQM(m,ndata,pp0,data,alpha,w);
    
    if (m==1)
        pp[1]=pp0[1];
    else
        CheckPositivity_IQM(&m,ndata,ind,pp,pp0,data,alpha,w);
    
    /* a "feasible" pp is returned from CheckPositivity_IQM */
    
    Compute_g(m,ndata,pp,data,alpha,g);
    
    minimum = Compute_minimum_IQM(ndata,ngrid,&indexmin,data,g,alpha0,w);
    
    *m1=m;
    *indexmin1=indexmin;
    *iteration1=iteration;
    *minimum1=minimum;
}

int armijoviol1(int m, double eps, double phi_old, double phi_new,
                double nabla_old[], double p_old[], double p_new[])
{
    int    i;
    double inprod;
    int    armijovioltemp;
    
    
    inprod = 0;
    for (i=1;i<=m;i++)
        inprod = inprod + nabla_old[i]*(p_new[i] - p_old[i]);
    
    armijovioltemp = (phi_new < phi_old + (1 - eps)*inprod);
    return    armijovioltemp;
}

int armijoviol2(int m, double eps, double phi_old, double phi_new,
                double nabla_old[], double p_old[], double p_new[])
{
    int    i;
    double inprod;
    int    armijovioltemp;
    
    inprod = 0;
    for (i=1;i<=m;i++)
        inprod = inprod + nabla_old[i]*(p_new[i]-p_old[i]);
    
    armijovioltemp = (phi_new > phi_old + eps*inprod);
    return    armijovioltemp;
}


int fenchelviol(double tol, double min_nabla, double norm)
{
    int fenchelvioltemp=0;
    
    if (min_nabla < -tol || norm>tol) fenchelvioltemp = 1;
    
    return fenchelvioltemp;
}



void    Compute_g(int m, int ndata, double pp[], double **data, double alpha[], double g[])
{
    int i,k;
    double tol=1.0e-10;
    
    for (i=0;i<ndata;i++)
        g[i] = 0;
    
    for (i=0;i<ndata;i++)
    {
        if (m>0)
        {
            for (k=1;k<=m;k++)
                g[i] += pp[k]*psi(data[i],alpha[k]);
        }
        else
            g[i] = fmax(tol,psi(data[i],0.5));
    }
}


void    Compute_DF(int m, double pp[], double DF[])
{
    int i;
    
    DF[0]=0;
    
    for (i=1;i<=m;i++)
        DF[i]= DF[i-1]+pp[i];
}

void    Compute_nabla(int m, int ndata, double **data, double alpha[], double g[], double nabla[])
{
    int j,k;
    double sum;
    double tol=1.0e-10;
    
    for (k=1;k<=m;k++)
    {
        {
            sum = 0;
            for (j=0;j<ndata;j++)
                sum+= psi(data[j],alpha[k])/fmax(tol,g[j]);
            nabla[k]=1.0-sum/ndata;
        }
    }
}


double phi_crit(int m, int ndata, double pp[], double g[])
{
    int        i;
    double    sum,sum1,tol=1.0e-10;
    
    sum1=sum=0;
    
    for (i=1;i<=m;i++)
        sum1 += pp[i];
    
    for (i=0;i<ndata;i++)
        sum += log(fmax(g[i],tol))/ndata;
    
    return sum1-sum;
}

void    Lin_Comb(int ndata, double l, double g1[], double g2[], double g[])
{
    int i;
    
    for (i=0;i<ndata;i++)
        g[i]=l*g1[i]+(1-l)*g2[i];
}

// data_point is a vector with the 3 elemants E, S_L and S_R, alpha is a possible location of mass

double psi(double *data_point, double alpha)
{
    double y;
    
    y=0;
    
    if (alpha<data_point[2])
        y += (data_point[2]-alpha);
    if (alpha<data_point[2]-data_point[0])
        y -= (data_point[2]-data_point[0]-alpha);
    if (alpha< data_point[1])
        y -= (data_point[1]-alpha);
    if (alpha < data_point[1]-data_point[0])
        y += (data_point[1]-data_point[0]-alpha);
    
    return y;
}

void    CheckPositivity_IQM(int *m1, int ndata, int ind[],
                            double pp[], double pp1[], double **data, double alpha[], double w[])
{
    double a,min_nabla,min1,crit,crit0,tol;
    int m,i,k,imin;
    
    tol=0;
    
    m=*m1;
    
    min_nabla=tol;
    min1=1.0;
    imin=1;
    
    for (i=1;i<=m;i++)
    {
        if (pp1[i]<tol)
        {
            crit = pp[i]/(pp[i]-pp1[i]);
            crit0=pp1[i];
            if (crit0<min_nabla)
                min_nabla=crit0;
            if (crit<min1)
            {
                min1 = crit;
                imin = i;
            }
        }
    }
    
    if (min_nabla<tol)
    {
        while (min_nabla<tol)
        {
            a = min1;
            for (k=1;k<=m;k++)
                pp1[k]=pp[k]+ a*(pp1[k]-pp[k]);
            if (imin<m)
            {
                for (k= imin;k<m;k++)
                {
                    alpha[k]=alpha[k+1];
                    pp1[k]=pp1[k+1];
                    ind[k]=ind[k+1];
                }
            }
            
            m -=1;
            
            transfer(m,pp1,pp);
            Compute_pp_IQM(m,ndata,pp1,data,alpha,w);
            
            min1 = 1.0;
            min_nabla = tol;
            
            for (i=1;i<=m;i++)
            {
                if (pp1[i]<tol)
                {
                    crit = pp[i]/(pp[i]-pp1[i]);
                    crit0=pp1[i];
                    if (crit0<min_nabla)
                        min_nabla=crit0;
                    if (crit<min1)
                    {
                        min1 = crit;
                        imin = i;
                    }
                }
            }
        }
        transfer(m,pp1,pp);
    }
    else transfer(m,pp1,pp);
    
    *m1=m;
}


void ComputeWeights_IQM(int m, int ndata, double **data, double alpha[], double pp[], double w[])
{
    int        i,j;
    double    sum,tol=1.0e-10;
    
    /* vector data contains the data points, vector alpha contains the (preliminary) locations of the masses */
    
    for (j=0;j<ndata;j++)
    {
        sum=0;
        for (i=1;i<=m;i++)
            sum += pp[i]*psi(data[j],alpha[i]);
        
        w[j] = 1.0/fmax(tol,sum);
        
    }
}


void Compute_critarray(int ndata, int ngrid, double **data, double g[], double alpha0[], double w[], double critarray[])
{
    int j,k;
    double sum,sum1,sum2;
    
    /* g contains the preliminary solution for the density of the mixing distribution at the observation points */
    
    for (k=1;k<=ngrid;k++)
    {
        sum1 =sum2 = 0;
        for (j=0;j<ndata;j++)
        {
            sum1 -= 2*w[j]*psi(data[j],alpha0[k]);
            sum2 += g[j]*psi(data[j],alpha0[k])*SQR(w[j]);
        }
        sum = ndata + sum1 + sum2;
        
        critarray[k]=sum;
    }
}


double Compute_minimum_IQM(int ndata, int ngrid, int *index,
                           double **data, double g[], double alpha0[], double w[])
{
    int j,k;
    double sum,sum1,sum2,sum3,minimum;
    
    minimum=0;
    
    /* g contains the preliminary solution for the density of the mixing distribution at the observation points */
    
    for (k=1;k<=ngrid;k++)
    {
        sum1 =sum2 = sum3 = 0;
        for (j=0;j<ndata;j++)
        {
            sum1 -= 2*w[j]*psi(data[j],alpha0[k]);
            sum2 += g[j]*psi(data[j],alpha0[k])*SQR(w[j]);
            sum3 += SQR(psi(data[j],alpha0[k])*w[j]);
        }
        sum = (ndata + sum1 + sum2)/sqrt(sum3);
        
        if (sum<minimum)
        {
            minimum=sum;
            *index=k;
        }
    }
    
    return minimum;
}


void Compute_W(int m, int ndata, double **data, double alpha[], double w[], double **W)
{
    int i,j,k;
    
    for (i=1;i<=m;i++)
    {
        for (j=1;j<=m;j++)
            W[i][j]=0;
    }
    
    //* data[] contains the data points, alpha[] contains the theta[i]'s in Geurt's notation */
    
    for (j=0;j<ndata;j++)
    {
        for (i=1;i<=m;i++)
        {
            for (k=1;k<=m;k++)
                W[i][k] += psi(data[j],alpha[i])*psi(data[j],alpha[k])*SQR(w[j])/ndata;
        }
    }
}

/*    The following procedure computes the coefficients pp[i].
 At each iteration a linear system of equations is solved
 */

void Compute_pp_IQM(int m, int ndata, double pp[], double **data, double alpha[], double w[])
{
    int i,j,*P;
    double **a,*b,*b1,*b2,**S,tol=1.0e-10;
    
    S = new double *[m+1];
    for (i=0;i<m+1;i++)
        S[i] = new double [m+1];
    
    b1 = new double[m+1];
    b2 = new double[m+1];
    P = new int[m+1];
    
    for (i=0;i<m+1;i++)
    {
        for (j=0;j<m+1;j++)
            S[i][j]=0;
    }
    
    a = new double *[m+2];
    for (i=0;i<m+2;i++)
        a[i] = new double[m+2];
    
    b = new double[m+2];
    
    for (i=1;i<m+2;i++)
        b[i]=0;
    
    Compute_W(m,ndata,data,alpha,w,a);
    
    if (m>=1)
    {
        for (i=1;i<=m;i++)
        {
            for (j=0;j<ndata;j++)
                b[i] += 2*psi(data[j],alpha[i])*w[j]/ndata;
        }
        
        for (i=1;i<=m;i++)
            b[i] -= 1;
        
        b[m+1]=1;
        
        for (i=1;i<=m;i++)
            a[i][m+1]=a[m+1][i]=1;
        
        a[m+1][m+1]=0;
        
        for (i=0;i<m+1;i++)
        {
            for (j=0;j<m+1;j++)
                S[i][j] =a[i+1][j+1];
        }
        
        for (i=0;i<m+1;i++)
            b1[i]=b[i+1];
        
        LUPDecompose(S,m,tol,P);
        LUPSolve(S,P,b1,m,b2);
        
        for (i=1;i<=m;i++)
            b[i]=b2[i-1];
        
        transfer(m,b,pp);
    }
    
    delete[] b; delete[] b1; delete[] b2; delete[] P;
    
    for (i=0;i<m+1;i++)
        delete[] S[i];
    delete[] S;
    
    for (i=0;i<m+2;i++)
        delete[] a[i];
    delete[] a;
    
}

void sort_index(int m, int ind[], double alpha[], double pp[])
{
    int i;
    data_object *obs;
    
    obs= new data_object[m];
    
    for (i=0;i<m;i++)
    {
        obs[i].ind=ind[i+1];
        obs[i].alpha=alpha[i+1];
        obs[i].pp=pp[i+1];
    }
    
    qsort(obs,m,sizeof(data_object),compare2);
    
    for (i=0;i<m;i++)
    {
        ind[i+1]=obs[i].ind;
        alpha[i+1]=obs[i].alpha;
        pp[i+1]=obs[i].pp;
    }
    
    delete[] obs;
}

void transfer(int n, double a[], double b[])
{
    int    i;
    for (i = 1; i<= n;i++)    b[i] = a[i];
}


void MLE_IQM(int ndata, int *m1, int ind[], int ngrid, double **data, double alpha[],
             double alpha0[], double pp[], double *phi_IQM, int *NumIt)
{
    int            i,m,iteration,iteration2,NumIterations,indexmin;
    double        min_nabla,minimum,*g,*g1,*g2,*w;
    double        norm,phi_old,phi_new,tol=1.0e-10,eps=0.1;
    double        *pp0,*pp1,*pp2,*pp3,*nabla,*nabla1,*nabla2;
    double        l,s;
    
    iteration=iteration2=0;
    NumIterations=1000;
    m=1;
    
    g = new double[ndata+1];
    g1 = new double[ndata+1];
    g2 = new double[ndata+1];
    
    nabla = new double[ngrid+1];
    nabla1 = new double[ngrid+1];
    nabla2 = new double[ngrid+1];
    pp0 = new double[ngrid+1];
    pp1 = new double[ngrid+1];
    pp2 = new double[ngrid+1];
    pp3 = new double[ngrid+1];
    w = new double[ndata+1];
    
    for (i=1;i<=ngrid;i++)
    {
        pp1[i]=pp2[i]=pp3[i]=1.0/ngrid;
        pp[i]=pp0[i]=0;
        ind[i]=i;
    }
    
    Compute_g(ngrid,ndata,pp1,data,alpha0,g1);
    ComputeWeights_IQM(ngrid,ndata,data,alpha0,pp1,w);
    
    pp[1]=1;
    ind[1]=ngrid/2;
    alpha[1]=alpha0[ngrid/2];
    Compute_g(1,ndata,pp,data,alpha,g);
    
    minimum=Compute_minimum_IQM(ndata,ngrid,&indexmin,data,g,alpha0,w);
    
    while (minimum < -tol && iteration2<=NumIterations)
        IterationLoop_IQM(&m,&indexmin,&iteration2,&minimum,ndata,ngrid,ind,
                          pp,pp0,data,g,alpha,alpha0,w);
    
    /*    Interpretation of the following 3 nabla vectors:
     1.    nabla is the vector of derivatives of the criterion function for strictly positive pp_i's.
     2.    nabla1 is the vector of derivatives of the criterion function  in all directions, if one uses pp1[]
     to compute the vector g1[] and from this the vector of derivatives.
     The vector pp1[] is all the time used to generate the weights for the quadratic
     minimization and the outer iterations proceed by adjusting the vector pp1[].
     3.    nabla2 is the vector of derivatives of the criterion function  in all directions, if one uses pp[]
     to compute the vector g[] and from this the vector of derivatives. */
    
    Compute_nabla(m,ndata,data,alpha,g,nabla);
    Compute_nabla(ngrid,ndata,data,alpha0,g1,nabla1);
    Compute_nabla(ngrid,ndata,data,alpha0,g,nabla2);
    
    //for (i=1;i<=ndata;i++)
    //printf("%15.10f\n",g1[i]);
    
    phi_new=phi_crit(ngrid,ndata,pp1,g1);
    
    min_nabla=0;
    for (i=1;i<=ngrid;i++)
        if (nabla2[i] < min_nabla) min_nabla = nabla2[i];
    
    
    norm=0;
    
    for (i=1;i<=m;i++)
    {
        if (fabs(nabla[i])>norm)
            norm = fabs(nabla[i]);
    }
    
    
    /*    If you want to see the iterations, "decomment" the following three lines. */
    
    //printf("\n");
    //printf("Iteration       Criterion            Fenchel                norm            # of generators\n\n");
    //printf("%5d        %12.10f       %12.10f         %13.10f     %8d\n",iteration,phi_new,min_nabla,norm,m);
    
    /* iterations continue until the Fenchel duality conditions
     are satisfied or until maximum of iterations is attained. */
    
    while (iteration<=NumIterations && fenchelviol(tol,min_nabla,norm))
    {
        iteration ++;
        iteration2 =0;
        
        phi_old=phi_new;
        
        minimum=Compute_minimum_IQM(ndata,ngrid,&indexmin,data,g,alpha0,w);
        
        while (minimum < -tol && iteration2<=NumIterations)
            IterationLoop_IQM(&m,&indexmin,&iteration2,&minimum,ndata,ngrid,ind,
                              pp,pp0,data,g,alpha,alpha0,w);
        
        phi_new=phi_crit(m,ndata,pp,g);
        
        
        for (i=1;i<=ngrid;i++)
            pp2[i]=0;
        
        for (i=1;i<=m;i++)
            pp2[ind[i]]=pp[i];
        
        if (!armijoviol2(ngrid, eps, phi_old, phi_new, nabla1, pp1,pp2))
        {
            for (i=1;i<=ngrid;i++)
                pp1[i]=0.1*pp1[i]+0.9*pp2[i];
            Lin_Comb(ndata,0.1,g1,g,g);
            for (i=0;i<ndata;i++)
                g1[i]=g[i];
        }
        else
        {
            transfer(ngrid,pp2,pp3);
            for (i=0;i<ndata;i++)
                g2[i]=g[i];
            
            l= 1.0;
            s= 0.5;
            while  ((s>1.0e-10) & (armijoviol1(ngrid, eps, phi_old, phi_new, nabla1, pp1,pp2)
                                   || armijoviol2(ngrid, eps, phi_old, phi_new, nabla1, pp1,pp2)))
            {
                if (armijoviol1(ngrid, eps, phi_old, phi_new, nabla1, pp1,pp2) && (l+s<=1))
                    l= l + s;
                else
                    l= l - s;
                for (i= 1;i<= ngrid;i++)
                    pp2[i]= (1-l)*pp1[i] + l*pp3[i];
                Lin_Comb(ndata,1-l,g1,g,g);
                phi_new=phi_crit(ngrid,ndata,pp2,g);
                s= s/2;
            }
            transfer(ngrid,pp2,pp1);
            for (i=0;i<ndata;i++)
                g1[i]=g[i];
        }
        
        phi_new=phi_crit(ngrid,ndata,pp1,g1);
        ComputeWeights_IQM(ngrid,ndata,data,alpha0,pp1,w);
        
        Compute_pp_IQM(m,ndata,pp0,data,alpha,w);
        
        CheckPositivity_IQM(&m,ndata,ind,pp,pp0,data,alpha,w);
        Compute_g(m,ndata,pp,data,alpha,g);
        
        Compute_nabla(m,ndata,data,alpha,g,nabla);
        Compute_nabla(ngrid,ndata,data,alpha0,g1,nabla1);
        Compute_nabla(ngrid,ndata,data,alpha0,g,nabla2);
        
        for (i=1,min_nabla=0;i<=ngrid;i++)
            if (nabla2[i] < min_nabla) min_nabla = nabla2[i];
        
        for (i=1,norm=0;i<=m;i++)
        {
            if (fabs(nabla[i])>norm)
                norm = fabs(nabla[i]);
        }
        
        /* If you want to see the iterations, "decomment" the following line. */
        
        //printf("%5d        %12.10f       %12.10f         %13.10f     %8d\n",iteration,phi_new,min_nabla,norm,m);
    }
    
    if (fenchelviol(tol,min_nabla,norm))
        printf("Maximum number of iteration exceeded before criterion was reached");
    
    phi_new=phi_crit(m,ndata,pp,g);
    
    /* If you want to see the final iteration with the real phi_new, "decomment" the following lines. */
    
    //printf("%5d        %12.10f       %12.10f         %13.10f     %8d\n",iteration,phi_new,min_nabla,norm,m);
    
    *m1=m;
    *phi_IQM=phi_new;
    *NumIt=iteration;
    
    delete[] g; delete[] g1; delete[] g2; delete[] nabla; delete[] nabla1; delete[] nabla2;
    delete[] pp0; delete[] pp1; delete[] pp2; delete[] pp3; delete[] w;
    
}

int LUPDecompose(double **A, int N, double Tol, int *P)
{
    int i, j, k, imax;
    double maxA, *ptr, absA;
    
    for (i = 0; i <= N; i++)
        P[i] = i; //Unit permutation matrix, P[N] initialized with N
    
    for (i = 0; i < N; i++)
    {
        maxA = 0.0;
        imax = i;
        
        for (k = i; k < N; k++)
            if ((absA = fabs(A[k][i])) > maxA)
            {
                maxA = absA;
                imax = k;
            }
        
        if (maxA < Tol) return 0; //failure, matrix is degenerate
        
        if (imax != i) {
            //pivoting P
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;
            
            //pivoting rows of A
            ptr = A[i];
            A[i] = A[imax];
            A[imax] = ptr;
            
            //counting pivots starting from N (for determinant)
            P[N]++;
        }
        
        for (j = i + 1; j < N; j++)
        {
            A[j][i] /= A[i][i];
            
            for (k = i + 1; k < N; k++)
                A[j][k] -= A[j][i] * A[i][k];
        }
    }
    
    return 1;  //decomposition done
}

// INPUT: A,P filled in LUPDecompose; b - rhs vector; N - dimension
// OUTPUT: x - solution vector of A*x=b

void LUPSolve(double **A, int *P, double *b, int N, double *x)
{
    for (int i = 0; i < N; i++)
    {
        x[i] = b[P[i]];
        
        for (int k = 0; k < i; k++)
            x[i] -= A[i][k] * x[k];
    }
    
    for (int i = N - 1; i >= 0; i--)
    {
        for (int k = i + 1; k < N; k++)
            x[i] -= A[i][k] * x[k];
        
        x[i] = x[i] / A[i][i];
    }
}

double bdf(double A, double B, int m, double t[], double p[], double u, double h)
{
    int            k;
    double        t1,t2,t3,sum;
    
    
    sum=0;
    for (k=1;k<=m;k++)
    {
        t1=(u-t[k])/h;
        t2=(u+t[k]-2*A)/h;
        t3=(2*B-u-t[k])/h;
        sum+= (KK(t1)+KK(t2)-KK(t3))*p[k];
        //sum+= KK(t1)*p[k];
    }
    return sum;
}

double KK(double x)
{
    double u,y;
    
    u=x*x;
    
    if (u<=1)
        y = (16.0 + 35*x - 35*pow(x,3) + 21*pow(x,5) - 5*pow(x,7))/32.0;
    else
    {
        if (x>1)
            y=1;
        else
            y=0;
        
    }
    
    return y;
}

double K(double x)
{
    double u,y;
    
    u=x*x;
    
    if (u<=1)
        y=(35.0/32)*pow(1-u,3);
    else
        y=0.0;
    
    return y;
}

double dens_estimate(double A, double B,  int m, double t[], double p[], double u, double h)
{
    int k;
    double      t1,sum;
    
    sum=0;
    
    for (k=1;k<=m;k++)
    {
        t1=(u-t[k])/h;
        sum += K(t1)*p[k]/h;
    }
    
    return fmax(0,sum);
}

int compare(const void *a, const void *b)
{
    double x = *(double*)a;
    double y = *(double*)b;
    
    if (x < y)
        return -1;
    if (x > y)
        return 1;
    return 0;
}

int compare2(const void *a, const void *b)
{
    if ((*(data_object *) a).ind < (*(data_object *) b).ind)
        return -1;
    if ((*(data_object *) a).ind > (*(data_object *) b).ind)
        return 1;
    return 0;
}

