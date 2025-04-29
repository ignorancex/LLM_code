//
//  The Weibull maximum likelihood method for estimating the distribution of the incubation time
//
//  Created by Piet Groeneboom on 16/07/2020.
//  Copyright (c) 2020 Piet Groeneboom. All rights reserved.

//

#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <time.h>
#include <fstream>
#include <string.h>
#include <Rcpp.h>

using namespace std;
using namespace Rcpp;

#define SQR(x) ((x)*(x))

int n,m,nloc,*loc;
double *alpha_init,*alpha;
int *data1,*data2,*tt;
double  *dens,*SMLE,*pp,*grid,*F,h;

void EM(int m, double alpha_init[], double alpha[]);
double bdf(double A, double B, int m, int t[], double p[], double u, double h);
double dens_estimate(double A, double B,  int m, int t[], double p[], double u, double h);
double KK(double x);
double K(double x);

// [[Rcpp::export]]

List NPMLE(DataFrame input)
{
    int i,j,ngrid,NumIt;
    int endpoint;
    double *grid;
    
    NumIt=10000;
    
    endpoint=140;
    
    DataFrame DF = Rcpp::DataFrame(input);
    IntegerVector data01 = DF["V1"];
    IntegerVector data02 = DF["V2"];
    
    // determine the sample size
    
    n = (int)data01.size();
    
    data1 = new int[n];
    data2 = new int[n];
    
    for (i=0;i<n;i++)
    {
        data1[i]=data01[i];
        data2[i]=data02[i];
    }
    
    // number of parameters to be estimated

    m=14;
    
    tt = new int[m+1];
    pp = new double[m+1];
    F = new double[m+1];
    
    for (i=0;i<=m;i++)
        tt[i]=i;
    
    // number of points on grid
    
    ngrid=140;

    grid = new double[ngrid+1];
    SMLE = new double[ngrid+1];
    dens = new double[ngrid+1];
    
    for (i=0;i<=ngrid;i++)
        grid[i] = i*0.1;
    
    // parameters for EM

    alpha       = new double[m+1];
    alpha_init  = new double[m+1];
    
    alpha_init[0]=alpha_init[1]=alpha_init[2]=0;
    
    for (i=3;i<=m;i++)
        alpha_init[i] = 1.0/(m-2);
    
    for (j=1;j<=NumIt;j++)
    {
        EM(m,alpha_init,alpha);
        for (i=1;i<=m;i++)
            alpha_init[i]=alpha[i];
    }
    
    F[0]=0;
    for (i=1;i<=m;i++)
        F[i] = F[i-1]+alpha[i];
    
    pp[0]=0;
    for (i=1;i<=m;i++)
        pp[i] = F[i]-F[i-1];
    
    for (i=0;i<=ngrid;i++)
    {
        SMLE[i] = bdf(0,14,m,tt,pp,grid[i],3.6);
        dens[i]=dens_estimate(0.0,14.0,m,tt,pp,grid[i],4.6);
    }
    
    NumericMatrix out1 = NumericMatrix(15,2);
    
    for (i=0;i<=14;i++)
    {
        out1(i,0)=i;
        out1(i,1)=F[i];
    }
    
    NumericMatrix out2 = NumericMatrix(ngrid+1,2);
    
    for (i=0;i<=ngrid;i++)
    {
        out2(i,0)=grid[i];
        out2(i,1)=SMLE[i];
    }
    
    NumericMatrix out3 = NumericMatrix(ngrid+1,2);
     
     for (i=0;i<=ngrid;i++)
     {
         out3(i,0)=grid[i];
         out3(i,1)=dens[i];
     }

    
    // make the list for the output, containing the two estimates and the log likelihood
    
    List out = List::create(Rcpp::Named("MLE")=out1,Rcpp::Named("SMLE")=out2,Rcpp::Named("dens")=out3);

    // free memory
    
    delete[] data1; delete[] data2; delete[] alpha; delete[] alpha_init; delete[] grid;
    delete[] SMLE; delete[] dens; delete[] tt; delete[] pp; delete[] F;
    return out;
}

void EM(int m, double alpha_init[], double alpha[])
{
    int i,j,k;
    double sum;
    
    for (i=1;i<=m;i++)
    {
        alpha[i]=0;
        for (j=0;j<n;j++)
        {
            if (data1[j]<i && i<=data2[j])
            {
                sum=0;
                for (k=1;k<=m;k++)
                {
                    if (data1[j]<k && k<=data2[j])
                        sum += alpha_init[k];
                }
                if (sum>0)
                    alpha[i] +=alpha_init[i]/(n*sum);
            }
        }
    }
}

double bdf(double A, double B, int m, int t[], double p[], double u, double h)
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

double dens_estimate(double A, double B,  int m, int t[], double p[], double u, double h)
{
    int k;
    double sum,t1,t2,t3;

    
    sum=0;
    
    for (k=1;k<=m;k++)
    {
        t1=(u-t[k])/h;
        t2=(u+t[k]-2*A)/h;
        t3=(2*B-u-t[k])/h;
        sum+= (K(t1)+K(t2)+K(t3))*p[k]/h;
    }
    return fmax(0,sum);
}
