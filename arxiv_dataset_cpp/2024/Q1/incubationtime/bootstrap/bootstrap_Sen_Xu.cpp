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

int     compare (const void * a, const void * b);
int     compare2(const void *a, const void *b);

double  criterion(int n1, double yy[], int index_end[], int **sec_index);
double  f_alpha(int n1, double alpha, double F[], double yy[], double yy_new[], int **sec_index, int index_end[]);
double  f_alpha_prime(int n1, double alpha, double F[], double grad[], double yy[], double yy_new[]);
int     fenchelviol(int n1, double b[], double nabla[], double tol, double *inprod, double *partsum);

void    isoreg(int n1, double F[], int **sec_index,
               int index_end[], double cumw[], double grad[]);
void    cumsum(int n1, double yy[], double cumw[], double cs[], double grad[], double w[]);
void    weights(int n1, double yy[], double w[], int **sec_index, int index_end[]);
void    gradient(int n1, double yy[], double grad[],int **sec_index, int index_end[]);
void    gradient(int n1, int **N, double yy[], double grad[]);
void    transfer(int first, int last, double a[], double b[]);

int     compute_mle(int n, double **data, double F[], double tt[], double pp[]);
void    sort_data(int n, double data1[], double data2[]);
void    sort_data0(int m, double data0[], int index0[]);
void    sort_index(int n, int index1[]);
double  criterion2(double A, double B, int n1, double tt[], double pp[], double u, double v, double h);
double Weibull_df(double a, double b, double M1, double x);
double  golden(double A, double B, int m, double t[], double p[], double v, double h,
              double (*f)(double,double,int,double*,double*,double,double,double));
double  golden(int n1, double F[], double yy[], double yy_new[],
               int **sec_index, int *index_end,
               double (*f)(int,double,double*,double*,double*,int**,int*));
double  bdf(double A, double B, int m, double t[], double p[], double u, double h);
double  bdf_conv(double B, int m, double t[], double p[], double u, double h);
double  KK(double x);
double  KK2(double x);
void    data_bootstrap(int n, int n1, double M1, double data3[], double **bootstrap_data,double tt[], double pp[], double h, int seed);
double data_smooth(int m, double M1, double tt[], double F[], double h);
double K(double x);


// [[Rcpp::export]]

List ComputeIntervals_df(int N, double M0, int ngrid0, NumericMatrix dat0)
{
    int     i,j,k,m,m_bootstrap,n,ngrid,NumIt,iter,percentile1,percentile2,*percentage,seed;
    double  a,b,**data,**bootstrap_data,*data3,*tt,*pp,*F,h;
    double  *MLE,*SMLE,*F_bootstrap,*MLE_bootstrap,*pp_bootstrap,*tt_bootstrap;
    double  M1,*grid,*lowbound,*upbound,**f3,*f4;
    
    
    M1 = (double)M0;
    seed=1;
    h=5;
    
    a=3.035140901;
    b=0.002619475;
    // determine the sample size
    
    n = (int)N;
    
    data = new double *[n];
    for (i=0;i<n;i++)
        data[i]= new double[2];
    
    for (i=0;i<n;i++)
    {
        for (j=0;j<2;j++)
            data[i][j]=(double)dat0(i,j);
    }
    
    // Number of bootstrap samples
    NumIt = 1000;
    
    percentile1=round(0.025*NumIt);
    percentile2=round(0.975*NumIt);
    
    bootstrap_data = new double *[n];
    for (i=0;i<n;i++)
      bootstrap_data[i]= new double[2];
    
    
    data3 = new double[n];
    
    for (i=0;i<n;i++)
        data3[i]=data[i][1]-data[i][0];
    
    ngrid = (int)ngrid0;
    percentage = new int[ngrid+1];
    grid = new double[ngrid+1];
    
    for (i=0;i<=ngrid;i++)
      percentage[i]=0;
                                  
    for (i=0;i<=ngrid;i++)
        grid[i] = M1*i/ngrid;
    
    // tt will contain the points of mass
    // pp will contain the masses
    
    tt = new double[2*n+2];
    pp = new double[2*n+2];
                                  
    pp_bootstrap = new double[2*n+2];
    tt_bootstrap = new double[2*n+2];
                                  
    // F will be the array containing the values of the MLE
    F =  new double[2*n+2];
    F_bootstrap =  new double[2*n+2];
    
    // m is the number of masses
    
    m = compute_mle(n,data,F,tt,pp);
    
    MLE =  new double[ngrid+1];
    
    for (j=0;j<=ngrid;j++)
    {
      for (k=1;k<=m;k++)
      {
        if (tt[k-1]<=grid[j] && grid[j]<tt[k])
          MLE[j] = F[k-1];
      }
      if (grid[j]>=tt[m])
        MLE[j]=F[m];
    }
                                  
    SMLE =  new double[ngrid+1];
    MLE_bootstrap =  new double[ngrid+1];
                                  
    for (i=0;i<=ngrid;i++)
        SMLE[i]= bdf(0.0,M1,m,tt,pp,grid[i],h);
                                  
    f3 = new double*[NumIt+1];
    for (iter=0;iter<NumIt+1;iter++)
         f3[iter] = new double[ngrid+1];
                                      
    f4 = new double[NumIt+1];
                                  
    lowbound = new double[ngrid+1];
    upbound  = new double[ngrid+1];
                                  
    for (iter=0;iter<NumIt;iter++)
    {
        seed++;
        data_bootstrap(n,m,M1,data3,bootstrap_data,tt,F,h,seed);
        m_bootstrap = compute_mle(n,bootstrap_data,F_bootstrap,tt_bootstrap,pp_bootstrap);
                                  
        for (j=0;j<=ngrid;j++)
        {
            for (k=1;k<=m_bootstrap;k++)
            {
                if (tt_bootstrap[k-1]<=grid[j] && grid[j]<tt_bootstrap[k])
                    MLE_bootstrap[j] = F_bootstrap[k-1];
            }
            if (grid[j]>=tt_bootstrap[m_bootstrap])
                MLE_bootstrap[j]=F_bootstrap[m_bootstrap];
        }
                                      
        for (i=0;i<=ngrid;i++)
            f3[iter][i]= MLE_bootstrap[i]-SMLE[i];
        
        //Rcout << iter+1 << endl;
     }
                                      
                                  
    for (i=0;i<=ngrid;i++)
    {
        for (iter=0;iter<NumIt;iter++)
            f4[iter]=f3[iter][i];
                                      
        qsort(f4,NumIt,sizeof(double),compare2);
                                      
        lowbound[i] = fmax(0,MLE[i]-f4[percentile2-1]);
        upbound[i]  = fmax(0,fmin(1,MLE[i]-f4[percentile1-1]));
        
        if (Weibull_df(a,b,M1,grid[i])<lowbound[i] || Weibull_df(a,b,M1,grid[i])>upbound[i])
            percentage[i]++;
    }
                                  
    
    NumericMatrix out1 = NumericMatrix(m+1,2);
    
    for (i=0;i<=m;i++)
    {
        out1(i,0)=tt[i];
        out1(i,1)=F[i];
    }
    
    int out2 = m;
    
    NumericMatrix out3 = NumericMatrix(ngrid+1,4);
    
    for (i=0;i<=ngrid;i++)
    {
        out3(i,0)=grid[i];
        out3(i,1)=MLE[i];
        out3(i,2)=lowbound[i];
        out3(i,3)=upbound[i];
    }
    
    NumericMatrix out4 = NumericMatrix(ngrid,1);
    for (i=0;i<ngrid;i++)
        out4(i,0)=percentage[i+1];
    
    NumericMatrix out5 = NumericMatrix(ngrid,1);
    for (i=0;i<ngrid;i++)
        out5(i,0)=upbound[i+1]-lowbound[i+1];
    // make the list for the output
    
    List out = List::create(Named("MLE")=out1,Named("m")=out2,Named("CI_df")=out3,Named("percentages")=out4,Named("length_interval")=out5);
    // free memory
    
    for (i=0;i<n;i++)
        delete data[i];
    delete[] data;
    
    for (i=0;i<n;i++)
        delete bootstrap_data[i];
    delete[] bootstrap_data;
    
    for (i=0;i<NumIt+1;i++)
        delete f3[i];
    delete[] f3;
    
    delete[] f4; delete[] lowbound; delete[] upbound;
    
    delete[] F; delete[] data3; delete[] percentage;
    delete[] tt; delete[] pp; delete[] tt_bootstrap; delete[] pp_bootstrap;
    delete[] grid; delete[] F_bootstrap; delete[] MLE; delete[] SMLE; delete[] MLE_bootstrap;
    
    return out;
}

double Weibull_df(double a, double b, double M1, double x)
{
    if (x>0 && x<= M1)
        return (1-exp(-b*pow(x,a)))/(1-exp(-b*pow(M1,a)));
    else
    {
        if (x>M1)
            return 1;
        else
            return 0;
    }
}

int compute_mle(int n, double **data, double F[], double tt[], double pp[])
{
    int i,j,k,m,n1;
    int *index0,*ind,*ind1,*index_end,**sec_index;
    double min,max_obs1,min_obs2,*data0;
    double *F1,*tt1,*grad,*cumw;
    
    data0 = new double[2*n];
    
    for (i=0;i<n;i++)
    {
        data0[i]=data[i][0];
        data0[n+i]=data[i][1];
    }
    
    ind = new int[2*n+1];
    index0 = new int[2*n+1];
    ind1 = new int[2*n+1];
    
    grad= new double[2*n+1];
    cumw= new double[2*n+1];
    
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
    
    vector<vector<int> >index(n1+2);
    
    for (i=0;i<n;i++)
    {
        if (ind1[n+i]<=n1)
        {
            if (data[i][0]<min_obs2)
                index[ind1[i]].push_back(ind1[n+i]);
            else
            {
                if (data[i][0]>=min_obs2 && data[i][1]<=max_obs1)
                {
                    index[ind1[i]].push_back(ind1[n+i]);
                    index[ind1[n+i]].push_back(ind1[i]);
                }
            }
        }
        else
        {
            if (data[i][0]>=min_obs2)
                index[ind1[n+i]].push_back(ind1[i]);
        }
    }
    
    sec_index = new int *[n1+2];
    index_end = new int[n1+2];
    
    for (i=0;i<=n1+1;i++)
    {
        k = (int)index[i].size();
        index_end[i]=k;
        sec_index[i] = new int[k];
        for (j=0;j<k;j++)
            sec_index[i][j] = index[i][j];
    }
    
    
    F[0]=0.0;
    for (i=1;i<=n1;i++)
         F[i]=i*1.0/(n1+1);
     
    F[n1+1]=1;
         
    isoreg(n1,F,sec_index,index_end,cumw,grad);
    
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

    for (i=0;i<n1+2;i++)
        delete[] sec_index[i];
    delete[] sec_index;
    
    delete[] ind;
    delete[] index0;
    delete[] index_end;
    delete[] ind1;
    
    delete[] data0;
    delete[] F1;
    delete[] cumw;
    delete[] grad;
    delete[] tt1;
        
    return m;
}
                                  
double data_smooth(int m, double M1, double tt[], double F[], double h)
{
    int i,j,seed;
    double x,u,v,w;
    
    w=1;
    seed = rand();
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dis_unif(0.0,1.0);

    //v = dis_unif(gen);
    //w = golden(0,tt[m]+h,m,tt,pp,v,h,criterion2);
    
    u = dis_unif(gen);
    for (i=1;i<=m;i++)
    {
        if (F[i-1]< u && u<= F[i])
            x=tt[i];
    }
    
    if (u>= F[m])
        x=tt[m];
    

    j=0;
    while (j<1)
    {
        w= 2*dis_unif(gen)-1;
        v=dis_unif(gen);
        if (v<(16.0/35)*K(w))
            j++;
    }
    
    v = x + h*w;
    
    if (v < 0)
      v = -v;

    if (v>M1)
      v = 2*M1-v;
    return v;
}


void data_bootstrap(int n, int n1, double M1, double data3[], double **bootstrap_data,double tt[], double FF[], double h, int seed)
{
    int    i;
    double u,v;
    
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dis_unif(0,1);
                
    for (i=0;i<n;i++)
    {
        u=dis_unif(gen);
        v = data3[i]*u+data_smooth(n1,M1,tt,FF,h);
        
        if (v > data3[i])
        {
            bootstrap_data[i][0]= v-data3[i];
            bootstrap_data[i][1]=v;
        }
        else
        {
            bootstrap_data[i][0]=0;
            bootstrap_data[i][1]=v;
        }
    }
}

double bdf_conv(double B, int m, double data[], double p[], double u, double h)
{
  int            i;
  double        t1,sum;
  
  sum=0;
  
  for (i=1;i<=m;i++)
  {
    t1=(u-data[i])/h;
    sum += KK2(t1)*p[i];
  }
  return fmax(0,sum);
}

int compare(const void * a, const void * b)
{
  return ( *(int*)a - *(int*)b );
}

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

double golden(int n1, double F[], double yy[], double yy_new[],
              int **sec_index, int *index_end,
              double (*f)(int,double,double*,double*,double*,int**,int*))
{
    double a,b,eps=1.0e-10;
    
    a=0;
    b=1;
    
    double k = (sqrt(5.0) - 1.0) / 2;
    double xL = b - k*(b - a);
    double xR = a + k*(b - a);
    
    while (b-a>eps)
    {
        if ((*f)(n1,xL,F,yy,yy_new,sec_index,index_end)<(*f)(n1,xR,F,yy,yy_new,sec_index,index_end))
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

double f_alpha(int n1, double alpha, double F[], double yy[], double yy_new[], int **sec_index, int index_end[])
{
    int i;
    
    for (i=1;i<=n1;i++)
        yy_new[i]=(1-alpha)*F[i]+alpha*yy[i];

    return criterion(n1,yy_new,index_end,sec_index);
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

double criterion(int n1, double yy[], int index_end[], int **sec_index)
{
    int i,j;
    double sum=0;
    
    for (j=0;j<index_end[0];j++)
        sum -= log(yy[sec_index[0][j]]);
    
    for (i=1;i<=n1;i++)
    {
        for (j=0;j<index_end[i];j++)
            sum -= log(fabs(yy[i]-yy[sec_index[i][j]]))/2;
    }
    
    for (j=0;j<index_end[n1+1];j++)
        sum -= log(1-yy[sec_index[n1+1][j]]);
    
    return sum;
}

void gradient(int n1, double yy[], double grad[], int **sec_index, int index_end[])
{
    int i,j;
    
    for (i=0;i<=n1;i++)
        grad[i]=0;
    
    for (j=0;j<index_end[0];j++)
        grad[sec_index[0][j]] += 1.0/yy[sec_index[0][j]];
    
    
    for (i=1;i<=n1;i++)
    {
        for (j=0;j<index_end[i];j++)
            grad[i] += 1.0/(yy[i]-yy[sec_index[i][j]]);
    }
    
    for (j=0;j<index_end[n1+1];j++)
            grad[sec_index[n1+1][j]] -= 1.0/(1-yy[sec_index[n1+1][j]]);
        
}


void weights(int n1, double yy[], double w[], int **sec_index, int index_end[])
{
    int i,j;
    
    for (j=1;j<=n1;j++)
        w[j]=0;
    
    for (j=0;j<index_end[0];j++)
        w[sec_index[0][j]] += 1.0/SQR(yy[sec_index[0][j]]);
    
    for (i=1;i<=n1;i++)
    {
        for (j=0;j<index_end[i];j++)
            w[i] += 1.0/SQR(yy[i]-yy[sec_index[i][j]]);
    }
    
    for (j=0;j<index_end[n1+1];j++)
        w[sec_index[n1+1][j]] += 1.0/SQR(1-yy[sec_index[n1+1][j]]);
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
    for (i=2;i<=n1;i++)
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

void isoreg(int n1, double F[], int **sec_index, int index_end[], double cumw[], double grad[])
{
    int i,iter;
    double *yy,*yy_new,alpha,inprod,partsum;
    double *w,*cs;
    double tol=1.0e-10;
    
    yy = new double[n1+2];
    yy_new = new double[n1+2];
    
    yy[0]=0.0;
    yy[n1+1]=1;
    
    cs   = new double[n1+2];
    w    = new double[n1+2];
            
    gradient(n1,F,grad,sec_index,index_end);
    
    iter=0;
    while (fenchelviol(n1,F,grad,tol,&inprod,&partsum) && iter<5000)
    {
        iter++;
        transfer(1,n1,F,yy);
        gradient(n1,yy,grad,sec_index,index_end);
        weights(n1,yy,w,sec_index,index_end);
        cumsum(n1,yy,cumw,cs,grad,w);
        convexminorant(n1,cumw,cs,yy);

        alpha=golden(n1,F,yy,yy_new,sec_index,index_end,f_alpha);
        
        for (i=1;i<=n1+1;i++)
            F[i] = alpha*yy[i]+(1-alpha)*F[i];
    }
    
    delete[] yy; delete[] yy_new;
    delete[] cs; delete[] w;
    
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
    double    t1,sum;
    
    sum=0;
    
    for (k=1;k<=m;k++)
    {
        t1=(u-t[k])/h;
        sum+= KK(t1)*p[k];
    }
    return  fmax(sum,0);
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

double KK2(double x)
{
  double y;
  
  y=0;
  
  if (x<=-2)
    y=0;
  
  if (x>=2)
    y=1;
  
  if (x>-2 && x<0)
    y=pow(2.0 + x,8)*(6864 - 16256*x + 16976*SQR(x) - 9440*pow(x,3) + 2690*pow(x,4) - 400*pow(x,5) + 25*pow(x,6))/3514368.0;
  
  
  if (x>=0 && x<2)
    y = 0.5 + 350*x/429.0 - 35*pow(x,3)/66.0 + 7*pow(x,5)/24.0 - 5*pow(x,7)/32.0 + 35*pow(x,8)/512.0 - 7*pow(x,10)/1536.0 + 35*pow(x,12)/135168.0 - 25*pow(x,14)/3514368.0;
  
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

