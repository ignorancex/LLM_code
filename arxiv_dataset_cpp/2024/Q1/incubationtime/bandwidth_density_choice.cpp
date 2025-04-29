//
//  Computation of the NPMLE for
//  the distribution of the incubation time
//
//  Created by Piet Groeneboom on Novemeber 5, 2020.
//  Copyright (c) 2020 Piet Groeneboom. All rights reserved.


#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <random>
#include <Rcpp.h>

using namespace std;
using namespace Rcpp;

#define SQR(x) ((x)*(x))

typedef struct
{
    double x;
    int index;
}
data0_object;


typedef struct
{
    double x;
    double y;
}
data_object;

typedef struct
{
    int i;
    int j;
}
index_object;

typedef struct
{
  double alpha, beta;
}
weight_t;

int compute_mle(int n, double **data, double F[], double tt[], double pp[]);
void sort_data0(int m, double data0[], int index0[]);
void sort_index(int m, int index1[]);
double bdf(double A, double B, int m, double t[], double p[], double u, double h);
double KK(double x);
double K(double x);
int    compare(const void * a, const void * b);
int    compare2(const void * a, const void * b);
double golden(int n1, int **N, int **ind_second, int *index_end, double F[], double yy[], double yy_new[], double (*f)(int,int**,int**,int*,double*,double*,double*,double));
double  golden(double A, double B, int m, double t[], double p[], double v, double h,
               double (*f)(double,double,int,double*,double*,double,double,double));
double f_alpha(int n1, int **N, int **ind_second, int *index_end, double F[], double yy[], double yy_new[], double alpha);
int fenchelviol(int n1, double yy[], double grad[], double tol, double *inprod, double *partsum);
void transfer(int first, int last, double a[], double b[]);
double criterion(int n1, int **N, int **ind_second, int *index_end, double yy[]);
double  criterion2(double A, double B, int n1, double tt[], double pp[], double u, double v, double h);
void gradient(int n1, int **N, int **ind_second, int *index_end, double yy[], double grad[]);
void weights(int n1, int **N, int **ind_second, int *index_end, double yy[], double w[]);
void cumsum(int n1, double yy[], double cumw[], double cs[], double grad[], double w[]);
void convexminorant(int n1, double cumw[], double cs[], double yy[]);
void isoreg(int n1, int **N, int **ind_second, int *index_end, double F[]);
void data_bootstrap(int n, int m, double M1, double data_exit[], double **bootstrap_data,double tt[], double pp[], double h, int seed);
double data_smooth(int m, double M1, double max_dens, double tt[], double pp[], double h);
weight_t weight(double x);
double bdens(double B, int m, double t[], double p[], double u, double h);
double MSE_dens(int n, int m, int ngrid, double grid[], double M1, int B, double **data, double data3[], double **bootstrap_data, double tt[], double pp[], double tt_bootstrap[], double pp_bootstrap[], double F_bootstrap[], double dens1[], double h, double h0, int seed);;

// [[Rcpp::export]]

List Compute_bandwidth(DataFrame input)
{
    int     B,i,m,n,ngrid,NumIt,iter,seed;
    double  **data,**bootstrap_data,*data3,*tt,*pp,*F,*h,h0,*MSE;
    double  *F_bootstrap,*pp_bootstrap,*tt_bootstrap;
    double  M1,*dens,*grid;
    double  min,h_min;
    
    DataFrame DF = Rcpp::DataFrame(input);
    NumericVector data01 = DF["V1"];
    NumericVector data02 = DF["V2"];
    
    // determine the sample size
    n = (int)data01.size();
    
    // Number of bandwidths tried
    NumIt = 20;
    
    // contains the (bootstrap) MSE's for different bandwidths
    MSE =  new double[NumIt];
    
    // the array h contains the different bandwidths tried
    h =  new double[NumIt];
    
    // upper bound incubation time
    M1 = 20;
    seed=1;
        
    // h0 is the bandwidth of the (oversmoothed) estimtate of the density
    // with which the bootstrap estimates are compared
    //and from which the bootstrap samples are generated.
    
    h0 = 6*pow(n,-1.0/11);
    
    data = new double *[n];
    for (i=0;i<n;i++)
        data[i]= new double[2];
    
    bootstrap_data = new double *[n];
    for (i=0;i<n;i++)
      bootstrap_data[i]= new double[2];
    
    // data3 is exposure time
    data3 = new double[n];
    
    for (i=0;i<n;i++)
    {
        data[i][0]=(double)data01[i];
        data[i][1]=(double)data02[i];
        data3[i]=data[i][1]-data[i][0];
    }
    
    ngrid = 100;
    grid = new double[ngrid+1];
                                  
    for (i=0;i<=ngrid;i++)
        grid[i] = M1*i/ngrid;

    // number of bootstrap samples in the estimates of the MSE
    B=1000;

    // F and F_bootstrap are the discrete MLEs

    F =  new double[2*n+2];
    F_bootstrap =  new double[2*n+2];
    
    // dens is the (oversmoothed) estimate of the density with which
    // the bootstrap estimates dens_bootstrap are compared
    // and from which the bootstrap samples are generated.
    
    dens =  new double[ngrid+1];
    
    // masses and location of masses of the MLE
    pp = new double[2*n+3];
    tt = new double[2*n+3];
    pp_bootstrap = new double[2*n+3];
    tt_bootstrap = new double[2*n+3];
    
    // m is the number of masses of the (discrete) MLE
    m = compute_mle(n,data,F,tt,pp);
                    
    // bdens is an estimate of the density using boundary kernels, dens is the pilot density
    for (i=0;i<=ngrid;i++)
        dens[i]= bdens(M1,m,tt,pp,grid[i],h0);
    
    min=1000;
    h_min = 10;
    
    Rcout << "20 bandwidths and MSE's from 1000 bootstrap samples for each bandwidth:" << std::endl << std::endl;
            
    for (iter=0;iter<NumIt;iter++)
    {
        seed ++;
        h[iter] = 3+iter*0.025*M1*pow(n,-1.0/7);
        
        // MSE is computed via the smooth bootstrap with bootstrap  samples
        // generated from the (oversmoothed) density estimate dens    
    
        MSE[iter] = MSE_dens(n,m,ngrid,grid,M1,B,data,data3,bootstrap_data,tt,pp, tt_bootstrap,pp_bootstrap,F_bootstrap,dens,h[iter],h0,seed);
        
        if (MSE[iter]<min)
        {
            min = MSE[iter];
            h_min = h[iter];
        }
                
        Rcout  << setw(10) << iter+1 << setprecision(6) <<  setw(15) << h[iter] << setprecision(6) <<  setw(15) << MSE[iter]  << std::endl;
    }
    
    Rcout << std::endl;
            
    for (i=0;i<=ngrid;i++)
        dens[i]= bdens(M1,m,tt,pp,grid[i],h_min);
                                  
    NumericMatrix out1 = NumericMatrix(m+1,2);
    
    for (i=0;i<=m;i++)
    {
        out1(i,0)=tt[i];
        out1(i,1)=F[i];
    }
    
    double out2 = h_min;
    
    NumericMatrix out3 = NumericMatrix(ngrid+1,2);
    
    for (i=0;i<=ngrid;i++)
    {
        out3(i,0)=grid[i];
        out3(i,1)=dens[i];
    }
    
    NumericMatrix out4 = NumericMatrix(NumIt,2);
    
    for (i=0;i<NumIt;i++)
    {
        out4(i,0)=h[i];
        out4(i,1)=MSE[i];
    }
    
    // make the list for the output
    
    List out = List::create(Named("MLE")=out1,Named("bandwidth")=out2,
                            Named("density")=out3,Named("MSE")=out4);

    // free memory
    
    for (i=0;i<n;i++)
        delete data[i];
    delete[] data;
    
    for (i=0;i<n;i++)
        delete bootstrap_data[i];
    delete[] bootstrap_data;
    
    delete[] F; delete[] F_bootstrap;  delete[] data3;
    delete[] tt; delete[] pp; delete[] tt_bootstrap; delete[] pp_bootstrap;
    delete[] grid; delete[] dens;
    delete[] h; delete[] MSE;
    
    return out;
}

int compute_mle(int n, double **data, double F[], double tt[], double pp[])
{
  int i,j,k,m,n1,**N,*index_end,**ind_second,**N1;
  int *index0,*ind,*ind1;
  double min,max_obs1,min_obs2,*data0;
  double *F1,*tt1;
  
  data0 = new double[2*n];
  
  for (i=0;i<n;i++)
  {
    data0[i]=data[i][0];
    data0[n+i]=data[i][1];
  }
  
  ind = new int[2*n+1];
  index0 = new int[2*n+1];
  ind1 = new int[2*n+1];
  
  sort_data0(2*n,data0,index0);
  
  sort_index(2*n,index0);
  
  // index0 maps indices of data1 and data2 to indices of data0
  // The indices of data2 are shifted to n+i
  // We have: data0[index0[i]]=data1[i] and data0[index0[n+i]]=data2[i], for (i=0,..,n.
  
  
  min_obs2= 1.0e10;
  for (i=0;i<n;i++)
  {
    if (data[i][1]<min_obs2)
      min_obs2 = data[i][1];
  }
  
  max_obs1= 0;
  for (i=0;i<n;i++)
  {
    if (data[i][0]>max_obs1)
      max_obs1 = data[i][0];
  }
  
  //printf("minimum and maximum are: %15.10f %15.10f\n",min_obs2,max_obs1);
  
  // ind maps the indices ind0 to the indices of the array tt
  
  for (i=0;i<2*n;i++)
  {
    if (data0[i]<min_obs2)
      ind[i]=0;
  }
  
  tt[0]=0;
  ind[0]=0;
  
  j=0;
  
  for (i=0;i<2*n;i++)
  {
    if (data0[i]<min_obs2)
    {
      ind[i]=0;
    }
    else
    {
      if (data0[i]>=min_obs2 && data0[i]<=max_obs1)
      {
        if (i>0 && data0[i]>data0[i-1])
          j++;
        ind[i]=j;
        tt[j]=data0[i];
      }
    }
  }
  
  n1=j;
  
  min=data0[2*n-1];
  
  for (i=2*n-1;i>=0;i--)
  {
    if (data0[i]>max_obs1)
    {
      ind[i]=n1+1;
      if (data0[i]<min)
        min=data0[i];
    }
  }
  
  tt[n1+1]=min;
  
  for (i=0;i<2*n;i++)
    ind1[i]=ind[index0[i]];
  
  N = new int *[n1+1];
  for (i=0;i<n1+1;i++)
    N[i]= new int[n1+2];
  
  for (i=0;i<=n1;i++)
  {
    for (j=0;j<=n1+1;j++)
      N[i][j]=0;
  }
  
  for (i=0;i<n;i++)
  {
    if (ind1[n+i]<=n1)
    {
      if (data[i][0]<min_obs2)
        N[0][ind1[n+i]]++;
      else
      {
        if (data[i][0]>=min_obs2 && data[i][1]<=max_obs1)
          N[ind1[i]][ind1[n+i]]++;
      }
    }
    else
    {
      if (data[i][0]>=min_obs2)
        N[ind1[i]][n1+1]++;
    }
  }
  
  index_end = new int[n1+1];
  N1        = new int*[n1+1];
  ind_second = new int*[n1+1];
  
  for (i=0;i<=n1;i++)
  {
    index_end[i]=0;
    for (j=i+1;j<=n1+1;j++)
    {
      if (N[i][j]>0)
        index_end[i]++;
    }
  }
  
  for (i=0;i<=n1;i++)
  {
    ind_second[i] = new int[index_end[i]+1];
    N1[i] = new int[index_end[i]+1];
  }
  
  for (i=0;i<=n1;i++)
  {
    k=0;
    for (j=i+1;j<=n1+1;j++)
    {
      if (N[i][j]>0)
      {
        k++;
        N1[i][k]=N[i][j];
        ind_second[i][k]=j;
      }
    }
  }

  F[0]=0.0;
  for (i=1;i<=n1;i++)
    F[i]=i*1.0/(n1+1);
  
  F[n1+1]=1;
  
  isoreg(n1,N1,ind_second,index_end,F);
  
  F1= new double[n1+2];
  tt1= new double[n1+2];
  
  F1[0]=0;
  
  j=0;
  
  for (i=1;i<=n1+1;i++)
  {
    if (F[i]>F[i-1])
    {
      j++;
      F1[j]=F[i];
      tt1[j]=tt[i];
    }
  }
  
  m=j;
  
  
  for (i=1;i<=m;i++)
  {
    F[i] = F1[i];
    tt[i]=tt1[i];
    pp[i] = F[i]-F[i-1];
  }
  
  for (i=1;i<=m;i++)
    pp[i]= F[i]-F[i-1];
  
  for (i=0;i<n1+1;i++)
    delete[] ind_second[i];
  delete[] ind_second;
  
  for (i=0;i<n1+1;i++)
    delete[] N[i];
  delete[] N;
  
  for (i=0;i<n1+1;i++)
    delete[] N1[i];
  delete[] N1;
  
  delete[]  index_end;
  
  delete[] ind;
  delete[] index0;
  delete[] ind1;
  
  delete[] data0;
  delete[] F1;
  delete[] tt1;
  
  return m;
}

// compare integers
int compare(const void * a, const void * b)
{
  return ( *(int*)a - *(int*)b );
}

// compare doubles
int compare2(const void *a, const void *b)
{
    double x = *(double*)a;
    double y = *(double*)b;
    
    if (x < y)
        return -1;
    if (x > y)
        return 1;
    return 0;
}


void sort_data0(int m, double data0[], int index0[])
{
    int i;
    data0_object *obs;
    
    obs = new data0_object[m];
    
    for (i=0;i<m;i++)
    {
        obs[i].x = data0[i];
        obs[i].index = i;
    }
    
    qsort(obs,m,sizeof(data0_object),compare2);
    
    for (i=0;i<m;i++)
    {
        data0[i]=obs[i].x;
        index0[i]=obs[i].index;
    }
    
    delete[] obs;
}

void sort_data(int m, double data1[], double data2[])
{
    int i;
    data_object *obs;
    
    obs = new data_object[m];
    
    for (i=0;i<m;i++)
    {
        obs[i].x = data1[i];
        obs[i].y = data2[i];
    }
    
    qsort(obs,m,sizeof(data_object),compare2);
    
    for (i=0;i<m;i++)
    {
        data1[i]=obs[i].x;
        data2[i]=obs[i].y;
    }
    
    delete[] obs;
}

void sort_index(int m, int index1[])
{
    int i;
    index_object *obs;
    
    obs = new index_object[m];
    
    for (i=0;i<m;i++)
    {
        obs[i].i = index1[i];
        obs[i].j = i;
    }
    
    qsort(obs,m,sizeof(index_object),compare);
    
    for (i=0;i<m;i++)
        index1[i]=obs[i].j;
    
    delete[] obs;
}

double golden(int n1, int **N, int **ind_second, int *index_end, double F[], double yy[], double yy_new[], double (*f)(int,int**,int**,int*,double*,double*,double*,double))
{
    double a,b,eps=1.0e-5;
    
    a=0;
    b=1;
    
    double k = (sqrt(5.0) - 1.0) / 2;
    double xL = b - k*(b - a);
    double xR = a + k*(b - a);
    
    while (b-a>eps)
    {
        if ((*f)(n1,N,ind_second,index_end,F,yy,yy_new,xL)<(*f)(n1,N,ind_second,index_end,F,yy,yy_new,xR))
        {
            b = xR;
            xR = xL;
            xL = b - k*(b - a);
        }
        else
        {
            a = xL;
            xL = xR;
            xR = a + k * (b - a);
        }
    }
    return (a+b)/2;
    
}

double f_alpha(int n1, int **N, int **ind_second, int *index_end, double F[], double yy[], double yy_new[], double alpha)
{
    int i;
    
    for (i=1;i<=n1;i++)
        yy_new[i]=(1-alpha)*F[i]+alpha*yy[i];

    return criterion(n1,N,ind_second,index_end,yy_new);
}


int fenchelviol(int n1, double yy[], double grad[], double tol, double *inprod, double *partsum)
{
    double    sum,sum2;
    int    i;
    int    fenchelvioltemp;
    
    fenchelvioltemp = 0;
    
    sum=0;
    sum2=0;
    
    for (i=1;i<=n1;i++)
    {
        sum += grad[i];
        if (sum < sum2)
            sum2 = sum;
    }
    
    sum=0;
    for (i=1;i<=n1;i++)
        sum += grad[i]*yy[i];
    
    *inprod = sum;
    *partsum = sum2;
    
    if ((fabs(sum) > tol) || (sum2 < -tol) ) fenchelvioltemp = 1;
    
    return fenchelvioltemp;
}

void transfer(int first, int last, double a[], double b[])
{
    int    i;
    for (i = first; i<= last;i++)    b[i] = a[i];
}

double criterion(int n1, int **N, int **ind_second, int *index_end, double yy[])
{
    int i,j;
    double sum=0;
    
    for (i=0;i<=n1;i++)
    {
        for (j=1;j<=index_end[i];j++)
            sum -= N[i][j]*log(yy[ind_second[i][j]]-yy[i]);
    }
    
    return sum;
}

void gradient(int n1, int **N, int **ind_second, int *index_end, double yy[], double grad[])
{
    int i,j;
    
    for (i=1;i<=n1;i++)
        grad[i]=0;
    
    
    for (i=0;i<=n1;i++)
    {
        for (j=1;j<=index_end[i];j++)
        {
            grad[i] -= N[i][j]/(yy[ind_second[i][j]]-yy[i]);
            grad[ind_second[i][j]] += N[i][j]/(yy[ind_second[i][j]]-yy[i]);
        }
    }
}


void weights(int n1, int **N, int **ind_second, int *index_end, double yy[], double w[])
{
    int i,j;
    
    for (j=1;j<=n1;j++)
        w[j]=0;
    
    
    for (i=0;i<=n1;i++)
    {
        for (j=1;j<=index_end[i];j++)
        {
            w[i] += N[i][j]/SQR(yy[ind_second[i][j]]-yy[i]);
            w[ind_second[i][j]] += N[i][j]/SQR(yy[ind_second[i][j]]-yy[i]);
        }
    }
}


void cumsum(int n1, double yy[], double cumw[], double cs[], double grad[], double w[])
{
    int    j;
    
    cumw[0]=0;
    cs[0]=0;
    
    for (j=1;j<=n1;j++)
    {
        cumw[j] = cumw[j-1]+w[j];
        cs[j]   = cs[j-1]+yy[j]*w[j]+grad[j];
    }
    
}

void convexminorant(int n1, double cumw[], double cs[], double yy[])
{
    int    i,j,m;
    
    yy[1] = cs[1]/cumw[1];
    for (i=1;i<=n1;i++)
    {
        yy[i] = (cs[i]-cs[i-1])/(cumw[i]-cumw[i-1]);
        if (yy[i-1]>yy[i])
        {
            j = i;
            while ((yy[j-1] > yy[i]) && (j>1))
            {
                j--;
                yy[i] = (cs[i]-cs[j-1])/(cumw[i]-cumw[j-1]);
                for (m=j;m<i;m++)
                    yy[m] = yy[i];
            }
        }
    }
    
    for (i=1;i<=n1;i++)
    {
        if (yy[i]<=0)
            yy[i]=0;
        if (yy[i]>=1)
            yy[i]=1;
    }
}

void isoreg(int n1, int **N, int **ind_second, int *index_end, double F[])
{
    int i,iter;
    double *yy,*yy_new,alpha,inprod,partsum;
    double *w,*cs,*cumw,*grad;
    double tol=1.0e-6;
    
    yy = new double[n1+2];
    yy_new = new double[n1+2];
    
    yy[0]=0.0;
    yy[n1+1]=1;
    
    cs   = new double[n1+2];
    cumw    = new double[n1+2];
    grad    = new double[n1+2];
    w    = new double[n1+2];
            
    gradient(n1,N,ind_second,index_end,F,grad);
    
    iter=0;
    while (fenchelviol(n1,F,grad,tol,&inprod,&partsum) && iter<1000)
    {
        iter++;
        transfer(0,n1,F,yy);
        gradient(n1,N,ind_second,index_end,yy,grad);
        weights(n1,N,ind_second,index_end,yy,w);
        cumsum(n1,yy,cumw,cs,grad,w);
        convexminorant(n1,cumw,cs,yy);

        alpha=golden(n1,N,ind_second,index_end,F,yy,yy_new,f_alpha);
        
        for (i=1;i<=n1+1;i++)
            F[i] = alpha*yy[i]+(1-alpha)*F[i];
    }
    
    delete[] yy; delete[] yy_new;
    delete[] cs; delete[] cumw; delete[] grad; delete[] w;
    
    //printf("Number of iterations: %d\n\n",iter);
    //printf("Fenchel duality criteria: %15.10f     %15.10f\n\n",inprod,partsum);
}

double criterion2(double A, double B, int n1, double tt[], double pp[], double u, double v, double h)
{
    return fabs(bdf(A,B,n1,tt,pp,u,h)-v);
}

double golden(double A, double B, int m, double t[], double p[], double v, double h,
              double (*f)(double,double,int,double*,double*,double,double,double))
{
    double a,b,eps=1.0e-6;
    
    a=A;
    b=B;
    
    double k = (sqrt(5.0) - 1.0) / 2;
    double xL = b - k*(b - a);
    double xR = a + k*(b - a);
    
    while (b-a>eps)
    {
        if ((*f)(A,B,m,t,p,xL,v,h)<(*f)(A,B,m,t,p,xR,v,h))
        {
            b = xR;
            xR = xL;
            xL = b - k*(b - a);
        }
        else
        {
            a = xL;
            xL = xR;
            xR = a + k * (b - a);
        }
    }
    return (a+b)/2;
    
}

double bdf(double A, double B, int m, double t[], double p[], double u, double h)
{
    int       k;
    double    t1,t2,t3,sum;
    
    sum=0;
    
    for (k=1;k<=m;k++)
    {
        t1=(u-t[k])/h;
        t2=(u+t[k]-2*A)/h;
        t3=(2*B-u-t[k])/h;
        sum+= (KK(t1)+KK(t2)-KK(t3))*p[k];
    }
    return  fmax(sum,0);
}

double MSE_dens(int n, int m, int ngrid, double grid[], double M1, int B, double **data, double data3[], double **bootstrap_data, double tt[], double pp[], double tt_bootstrap[], double pp_bootstrap[], double F_bootstrap[], double dens1[], double h, double h0, int seed)
{
  int i,iter,m_bootstrap;
  double MSE;
  
  MSE=0;
  for (iter=1;iter<=B;iter++)
  {
    data_bootstrap(n,m,M1,data3,bootstrap_data,tt,pp,h0,seed);
    
    m_bootstrap = compute_mle(n,bootstrap_data,F_bootstrap,
                              tt_bootstrap,pp_bootstrap);
    
    for (i=1;i<ngrid;i++)
      MSE += SQR(bdens(M1,m_bootstrap,tt_bootstrap,pp_bootstrap,grid[i],h)-dens1[i]);
  }
  
  return MSE/B;
}


double bdens(double B, int m, double t[], double p[], double u, double h)
{
    int             k;
    double          rho,sum,x;
    weight_t        weights;
    
    sum=0;
    if (u-h>=0 && u+h<=B)
    {
        for (k=1;k<=m;k++)
        {
            x=(u-t[k])/h;
            sum+= K(x)*p[k]/h;
        }
    }
    else
    {
        if (u<h)
        {
            rho = u/h;
            weights=weight(rho);
            for (k=1;k<=m;k++)
            {
                x=(u-t[k])/h;
                sum += (K(x)*weights.alpha + x*K(x)*weights.beta)*p[k]/h;
            }
            
        }
        else
        {
            if (u>B-h)
            {
                rho = (B-u)/h;
                weights=weight(rho);
                for (k=1;k<=m;k++)
                {
                    x=(u-t[k])/h;
                    sum += (K(x)*weights.alpha - x*K(x)*weights.beta)*p[k]/h;
                }
            }
        }
    }
     
    return fmax(0,sum);
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

void data_bootstrap(int n, int m, double M1, double data_exit[], double **bootstrap_data,double tt[], double pp[], double h, int seed)
{
    int    i;
    double a,u,v,max_dens;
    
    max_dens=0;
    for (i=1;i<=100;i++)
    {
        a = bdens(M1,m,tt,pp,i*0.01*M1,h);
        if (a>max_dens)
            max_dens = a;
    }
    
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dis_unif(0,1);
                
    for (i=0;i<n;i++)
    {
        u=dis_unif(gen);
        v = data_exit[i]*u+data_smooth(m,M1,max_dens,tt,pp,h);
        
        if (v > data_exit[i])
        {
            bootstrap_data[i][0]= v-data_exit[i];
            bootstrap_data[i][1]=v;
        }
        else
        {
            bootstrap_data[i][0]=0;
            bootstrap_data[i][1]=v;
        }
    }
}

double data_smooth(int m, double M1, double max_dens, double tt[], double pp[], double h)
{
    int j,seed;
    double c,v,w;
    
    w=1;
    seed = rand();
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dis_unif(0.0,1.0);
    
    j=0;
    while (j<1)
    {
        w=M1*dis_unif(gen);
        v=dis_unif(gen);
        c = bdens(M1,m,tt,pp,w,h);
        if (v<c/(max_dens*(M1+1)))
            j++;
    }
    
    return w;
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

weight_t weight(double x)
{
  short i;
  double y1, y2, y3, y4;
  double *xx;
  weight_t temp;
  
  xx= new double[10];
  xx[1] = x;
  for (i=2;i<=9;i++)
    xx[i] = x * xx[i - 1];
  
  y1 = 0.5 + (35.0*xx[1])/32.0-(35.0*xx[3])/32.0+(21.0*xx[5])/32.0-(5.0*xx[7])/32.0;
  y2 = -35.0/256.0+(35.0 * xx[2])/64.0-(105.0*xx[4])/128.0+(35.0*xx[6])/64.0-(35.0*xx[8])/256.0;
  y3 = 1.0/18 + 25.0*xx[3]/96 - 21.0*xx[5]/32 + 15.0*xx[7]/32 - 35.0*xx[9]/288;
  
  
  y4 = y1 * y3 - (y2*y2);
  temp.alpha = y3 / y4;
  temp.beta = -y2 / y4;
  
  delete[] xx;
  
  return temp;
}

