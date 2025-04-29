//
//  The Weibull maximum likelihood method for estimating the distribution of the incubation time
//
//  Created by Piet Groeneboom on 16/07/2020.
//  Copyright (c) 2020 Piet Groeneboom. All rights reserved.
//
//  The Hooke-Jeeves algorithm is a codification of toms178.cpp:
//  Version: 12 February 2008
//  Author:
//    The ALGOL original is by Arthur Kaupe.
//    C version by Mark Johnson
//    C++ version by John Burkardt
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
int *data1,*data2;

double  criterion(int m, double alpha[]);
int     hooke(int m, double startpt[], double endpt[], double rho, double eps,
                int itermax, double f(int m, double alpha[]));
double  best_nearby (int m, double delta[], double point[], double prevbest,
                    double f(int m, double alpha[]), int *funevals);

int gelimd(double **a,double *b,double *x, int m);
int gelimd2(double **a,double *b,double *x, int m);
void f(double *x,double *fv,int m);

// [[Rcpp::export]]

List Weibull(DataFrame input)
{
    int ngrid,i,iter,nIterations;
    double rho,eps;
    double *grid;
    
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

    m=2;
    
    // number of points on grid
    
    ngrid=140;
    
    grid= new double[ngrid+1];
    
    for (i=0;i<=ngrid;i++)
        grid[i] = i*0.1;
    
    // parameters for Hooke-Jeeves
    
    rho=0.5;
    eps=1.0e-10;
    nIterations=1000;

    alpha       = new double[m];
    alpha_init  = new double[m];
    
    alpha_init[0]=3.0;
    alpha_init[1]=0.02;
    
    iter=hooke(m,alpha_init,alpha,rho,eps,nIterations,criterion);
        
    NumericMatrix out1 = NumericMatrix(ngrid+1,2);
    
    for (i=0;i<=ngrid;i++)
    {
        out1(i,0)=grid[i];
        out1(i,1)=1-exp(-alpha[1]*pow(grid[i],alpha[0]));
    }
    
    NumericMatrix out2 = NumericMatrix(ngrid+1,2);
    
    for (i=0;i<=ngrid;i++)
    {
         out2(i,0)=grid[i];
         out2(i,1)= alpha[0]*alpha[1]*exp(-alpha[1]*pow(grid[i],alpha[0]))*pow(grid[i],alpha[0]-1);
     }
    
    NumericVector out3 = NumericVector(2);
    
    out3(0)=alpha[0];
    out3(1)=alpha[1];
    
    // make the list for the output, containing the two estimates and the log likelihood
    
    List out = List::create(Rcpp::Named("df")=out1,Rcpp::Named("dens")=out2,Rcpp::Named("parameters")=out3);

    // free memory
    
    delete[] data1; delete[] data2; delete[] alpha; delete[] alpha_init; delete[] grid;
    return out;
}

double criterion(int m, double alpha[])
{
    int i;
    double sum1,sum;
    
    sum=0.0;
    
    for (i=0;i<n;i++)
    {
        sum1 = exp(-alpha[1]*pow(data1[i],alpha[0]))-exp(-alpha[1]*pow(data2[i],alpha[0]));
        sum -= log(sum1);
    }
    return sum;
}

double best_nearby(int m, double delta[], double point[], double prevbest,
                    double f(int m, double alpha[]), int *funevals)
{
    double ftmp,minf,*z;
    int i;
    
    z = new double[m];
    
    minf = prevbest;
    
    for ( i = 0; i < m; i++ )
        z[i] = point[i];
    
    for ( i = 0; i < m; i++ )
    {
        z[i] = point[i] + delta[i];
        
        ftmp = f(m,z);
        *funevals = *funevals + 1;
        
        if ( ftmp < minf )
            minf = ftmp;
        else
        {
            delta[i] = - delta[i];
            z[i] = point[i] + delta[i];
            ftmp = f(m,z);
            *funevals = *funevals + 1;
            
            if ( ftmp < minf )
                minf = ftmp;
            else
                z[i] = point[i];
        }
    }
    
    for ( i = 0; i < m; i++ )
        point[i] = z[i];
    
    delete [] z;
    
    return minf;
}

int hooke(int m, double startpt[], double endpt[], double rho, double eps,
          int itermax, double f(int m, double alpha[]))
{
    double *delta,fbefore;
    int i,iters,keep,funevals,count;
    double newf,*newx,steplength,tmp;
    bool verbose = false;
    double *xbefore;
    
    delta = new double[m];
    newx = new double[m];
    xbefore = new double[m];
    
    for ( i = 0; i < m; i++ )
        xbefore[i] = newx[i] = startpt[i];
    
    for ( i = 0; i < m; i++ )
    {
        if ( startpt[i] == 0.0 )
            delta[i] = rho;
        else
            delta[i] = rho*fabs(startpt[i]);
    }
    
    funevals = 0;
    steplength = rho;
    iters = 0;
    
    
    fbefore = f(m,newx);
    funevals = funevals + 1;
    newf = fbefore;
    
    while ( iters < itermax && eps < steplength )
    {
        iters = iters + 1;
        
        if (verbose)
        {
            cout << "\n";
            cout << "  FUNEVALS, = " << funevals
            << "  F(X) = " << fbefore << "\n";
            
            for ( i = 0; i < m; i++ )
            {
                cout << "  " << i + 1
                << "  " << xbefore[i] << "\n";
            }
        }
        //
        //  Find best new alpha, one coordinate at a time.
        //
        for ( i = 0; i < m; i++ )
            newx[i] = xbefore[i];
        
        
        
        newf = best_nearby(m,delta,newx,fbefore,f,&funevals);
        //
        //  If we made some improvements, pursue that direction.
        //
        keep = 1;
        count=0;
        
        while (newf<fbefore && keep == 1 && count<=100)
        {
            count++;
            for ( i = 0; i < m; i++ )
            {
                //
                //  Arrange the sign of DELTA.
                //
                if ( newx[i] <= xbefore[i] )
                    delta[i] = - fabs(delta[i]);
                else
                    delta[i] = fabs(delta[i]);
                //
                //  Now, move further in this direction.
                //
                tmp = xbefore[i];
                xbefore[i] = newx[i];
                newx[i] = newx[i] + newx[i] - tmp;
            }
            
            fbefore = newf;
            
            newf = best_nearby(m,delta,newx,fbefore,f,&funevals);
            //
            //  If the further (optimistic) move was bad...
            //
            if (fbefore <= newf)
                break;
            //
            //  Make sure that the differences between the new and the old points
            //  are due to actual displacements; beware of roundoff errors that
            //  might cause NEWF < FBEFORE.
            //
            keep = 0;
            
            for ( i = 0; i < m; i++ )
            {
                if ( 0.5 * fabs(delta[i]) < fabs(newx[i]-xbefore[i]))
                {
                    keep = 1;
                    break;
                }
            }
        }
        
        if (eps <= steplength && fbefore <= newf)
        {
            steplength = steplength * rho;
            for ( i = 0; i < m; i++ )
                delta[i] = delta[i] * rho;
        }
        
    }
    
    for ( i = 0; i < m; i++ )
        endpt[i] = xbefore[i];
    
    delete [] delta;
    delete [] newx;
    delete [] xbefore;
    
    return iters;
}



