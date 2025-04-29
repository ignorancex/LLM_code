//
//  The support reduction algorithm for computing the MLE for estimating
//  the distribution of the incubation time. Confidence intervals
//
//  Created by Piet Groeneboom on 20/09/2023.
//  Copyright (c) 2023 Piet Groeneboom. All rights reserved.


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
  int ind;
  double alpha;
  double pp;
}
data_object;

void   data_exp_double(int n, int ngrid0, double **data, double cum_Weibull[], int seed);
void    IterationLoop_IQM(int *m1, int *indexmin1, int *iteration1, double *minimum1, int ndata,
 int ngrid, int ind[], double pp[], double pp0[], double **data, double g[], double tt[],
  double grid[], double w[]);
int armijoviol1(int m, double eps, double phi_old, double phi_new,
                double nabla_old[], double p_old[], double p_new[]);
int armijoviol2(int m, double eps, double phi_old, double phi_new,
                double nabla_old[], double p_old[], double p_new[]);
int fenchelviol(double tol, double min_nabla, double norm);
double G(int m, double tt[], double pp[], double *data);
void    Compute_g(int m, int ndata, double pp[], double **data, double tt[], double g[]);
void    Compute_DF(int m, double pp[], double DF[]);
void    Compute_nabla(int m, int ndata, double **data, double tt[], double g[], double nabla[]);
double phi_crit(int m, int ndata, double pp[], double g[]);
void    Lin_Comb(int ndata, double l, double g1[], double g2[], double g[]);
double psi(double *data_point, double tt);
void    CheckPositivity_IQM(int *m1, int ndata, int ind[],
                            double pp[], double pp1[], double **data, double tt[], double w[]);
void ComputeWeights_IQM(int m, int ndata, double **data, double tt[], double pp[], double w[]);
void Compute_critarray(int ndata, int ngrid, double **data, double g[], double grid[],
         double w[], double critarray[]);
double Compute_minimum_IQM(int ndata, int ngrid, int *index,
                           double **data, double g[], double grid[], double w[]);
void Compute_W1(int m, int ndata, double **data, double tt[], double g[], double **W);
void Compute_W(int m, int ndata, double **data, double tt[], double w[], double **W);
void Compute_pp_IQM(int m, int ndata, double pp[], double **data, double tt[], double w[]);
void MLE_IQM(int ndata, int *m1, int ind[], int ngrid, double **data, double tt[],double grid[], double pp[], double MLE[], double **W, double *phi_IQM, int *NumIt);

void Compute_covar(int ngrid, int NumIt, double **Fvalue, double **covar);
void compute_sigma(int m, int ngrid, double **Fisher_matrix, double *MLE, double *F, double *tt, double *grid, double *sigma);
void MLE_to_F(int m, int ngrid0, double grid[], double tt[], double MLE[], double F[]);
int compare(const void *a, const void *b);
void sort_index(int m, int ind[], double alpha[], double pp[]);
double Weibull_inv(double a, double b, double M1, double u);
void  transfer(int n, double a[], double b[]);
int   LUPDecompose(double **A, int N, double Tol, int *P);
void  LUPInvert(double **A, int *P, int N, double **IA);
void  LUPSolve(double **A, int *P, double *b, int N, double *x);

// [[Rcpp::export]]

List CI_NPMLE()
{
    int     i,n,m,ngrid,ngrid0,*ind;
    int     *percentage,iterations,iter,NumIt,seed;
    double  *tt,*pp,*F;
    double  **data;
    double  phi_IQM,*var_array;
    double  *grid,**Fisher_matrix;
    double  *sigma,**F_value,**covar;
    double  *MLE,*mean_diag,**diag_Fisher;
    double  *Weibull_array,*cumulative_Weibull;
    double  *lowbound,*upbound;
    
    // determine the sample size
    
    n=1000;
    NumIt = 1000;
    
    seed=1;
    
    ngrid0=16;
    ngrid=34;
    
    lowbound = new double[ngrid0+1];
    upbound  = new double[ngrid0+1];
    percentage = new int[ngrid0+1];
    
    for (i=0;i<=ngrid0;i++)
      percentage[i]=0;
    
    ind = new int[ngrid+1];
    
    data = new double *[n];
    for (i=0;i<n;i++)
        data[i]= new double[3];
    
    var_array = new double[ngrid+1];
    
    // means of diagonals inverse Fisher matrix
    var_array[0]=0;
    var_array[1]=0.2980044352;
    var_array[2]=0.2972111370;
    var_array[3]=0.3889178792;
    var_array[4]= 1.0493557668;
    var_array[5]=2.0455598422;
    var_array[6]=3.0001495297;
    var_array[7]=3.2479146132;
    var_array[8]=2.6016648955;
    var_array[9]=1.6304075616;
    var_array[10]=0.7956524055;
    var_array[11]= 0.3148103904;
    var_array[12]=0.1046048985;
    var_array[13]=0.0360410435;
    var_array[14]=0.0076358554;
    var_array[15]=0.0012257086;
    
    Weibull_array = new double[ngrid+1];
    cumulative_Weibull = new double[ngrid+1];
    
    // integrals over intervals f length 1
    
    Weibull_array[0]=0;
    Weibull_array[1]=0.00064872;
    Weibull_array[2]=0.00992978;
    Weibull_array[3]=0.0429498;
    Weibull_array[4]=0.112548;
    Weibull_array[5]=0.224047;
    Weibull_array[6]=0.371218;
    Weibull_array[7]=0.535908;
    Weibull_array[8]=0.693386;
    Weibull_array[9]=0.821808;
    Weibull_array[10]=0.910489;
    Weibull_array[11]=0.96182;
    Weibull_array[12]=0.98643;
    Weibull_array[13]=0.996074;
    Weibull_array[14]=0.99912;
    Weibull_array[15]=0.999884;
    
    for (i=16;i<ngrid+1;i++)
      Weibull_array[i]=1;
    
    cumulative_Weibull[0]=0;
    for (i=1;i<ngrid+1;i++)
      cumulative_Weibull[i]=cumulative_Weibull[i-1]+Weibull_array[i];

    F =  new double[ngrid+1];
    MLE = new double[ngrid+1];
    
    F_value = new double *[NumIt+1];
    for (i=0;i<NumIt+1;i++)
      F_value[i]= new double[ngrid+1];
    
    // pp is array containng the masses of the MLE to be computed, updated during the iterations
    pp = new double[ngrid+1];
    
    // tt is going to contain the points of mass
    tt = new double[ngrid+1];

    grid = new double[ngrid+1];
    
    for (i=0;i<=ngrid;i++)
        grid[i] = (double)i;
    
    sigma = new double [ngrid+1];
        
    Fisher_matrix = new double *[ngrid+1];
    for (i=0;i<ngrid+1;i++)
        Fisher_matrix[i]= new double[ngrid+1];
    
    diag_Fisher = new double *[NumIt+1];
    for (i=0;i<NumIt+1;i++)
      diag_Fisher[i]= new double[ngrid+1];
    
    mean_diag = new double[ngrid+1];
    
    covar = new double *[ngrid+1];
    for (i=0;i<ngrid+1;i++)
      covar[i]= new double[ngrid+1];
    
    
    m=1;
    
    tt[0]=pp[0]=F[0]=MLE[0]=0;
    
    for (iter=0;iter<NumIt;iter++)
    {
        seed++;
        
        data_exp_double(n,ngrid0,data,cumulative_Weibull,seed);
        
        MLE_IQM(n,&m,ind,ngrid,data,tt,grid,pp,MLE,Fisher_matrix,&phi_IQM,&iterations);
        
        compute_sigma(m,ngrid,Fisher_matrix,MLE,F,tt,grid,sigma);
        
        for (i=1;i<ngrid0;i++)
            F_value[iter][i]=F[i];
        
        for (i=1;i<ngrid0;i++)
        {
            lowbound[i]=F[i]-1.96*sqrt(var_array[i])/sqrt(n);
            upbound[i]=F[i]+1.96*sqrt(var_array[i])/sqrt(n);
            
            diag_Fisher[iter][i]=SQR(sigma[i]);
    
            if (Weibull_array[i]<lowbound[i] || Weibull_array[i]>upbound[i])
                    percentage[i]++;
        }
        
        Rcout << iter+1<< endl;
    }
    
    for (i=1;i<ngrid0;i++)
      mean_diag[i] = 0;
    
    for (iter=0;iter<NumIt;iter++)
    {
        for (i=1;i<ngrid0;i++)
            mean_diag[i] += diag_Fisher[iter][i];
    }
    
    for (i=1;i<ngrid;i++)
      mean_diag[i] = 0;
    
    for (iter=0;iter<NumIt;iter++)
    {
        for (i=1;i<ngrid0;i++)
            mean_diag[i] += diag_Fisher[iter][i];
    }
    
    for (i=1;i<ngrid0;i++)
      mean_diag[i] /= NumIt;
    
    Compute_covar(ngrid0,NumIt,F_value,covar);
        
    NumericMatrix out1 = NumericMatrix(m,2);
    
    for (i=0;i<m;i++)
    {
        out1(i,0)=tt[i+1];
        out1(i,1)=MLE[i+1];
    }
    
    NumericMatrix out2 = NumericMatrix(8,5);
     
    for (i=0;i<=7;i++)
    {
        lowbound[i]=F[i+3]-1.96*sigma[i+3]/sqrt(n);
        upbound[i]=F[i+3]+1.96*sigma[i+3]/sqrt(n);

        out2(i,0)=grid[i+3];
        out2(i,1)=Weibull_array[i+3];
        out2(i,2)=F[i+3];
        out2(i,3)=lowbound[i+3];
        out2(i,4)=upbound[i+3];
    }
    
    NumericMatrix out3 = NumericMatrix(8,2);
    
    for (i=0;i<=7;i++)
    {
        out3(i,0)=grid[i+3];
        out3(i,1)=1-percentage[i+3]*1.0/NumIt;
    }
    
    NumericMatrix out4 = NumericMatrix(ngrid0-1,2);
    
    for (i=0;i<ngrid0-1;i++)
    {
      out4(i,0) = n*covar[i+1][i+1];
      out4(i,1) = mean_diag[i+1];
    }

    // make the list for the output
    
    List out = List::create(Named("MLE")=out1,Named("CI_MLE")=out2,
                            Named("percentages")=out3,Named("Variances")=out4);

    // free memory
    
    for (i=0;i<n;i++)
        delete[] data[i];
    delete[] data;
    
    delete[] pp; delete[] tt; delete[] grid; delete[] F;  delete[] MLE;
    delete[] Weibull_array; delete[] cumulative_Weibull;
    delete[] sigma; delete[] mean_diag;
    delete[] lowbound; delete[] upbound; delete[] percentage;
    delete[] ind;
    
    for (i=0;i<NumIt+1;i++)
        delete[] F_value[i];
    delete[] F_value;
 
    for (i=0;i<ngrid+1;i++)
        delete[] Fisher_matrix[i];
    delete[] Fisher_matrix;
    for (i=0;i<NumIt+1;i++)
        delete[] diag_Fisher[i];
    delete[] diag_Fisher;
    for (i=0;i<ngrid+1;i++)
        delete[] covar[i];
    delete[] covar;
    
    return out;
}

void   data_exp_double(int n, int ngrid0, double **data, double cum_Weibull[], int seed)
{
    int    i,j,m;
    double s,v;
    
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dis_unif(0,1);
    std::uniform_int_distribution<int> dis_int(0,3);
    std::uniform_int_distribution<int> dis1(1,ngrid0-1);
    
    s=0;
    for (i=0;i<n;i++)
    {
        m=dis1(gen);
        v = dis_unif(gen)*m;
        
        for (j=1;j<=ngrid0+m;j++)
        {
            if (j-m<=0)
            {
                if (cum_Weibull[j-1]<v && v<=cum_Weibull[j])
                    s = (double)(j);
            }
            else
            {
                if (j-m>0)
                {
                    if (cum_Weibull[j-1]-cum_Weibull[j-1-m]< v && v<=cum_Weibull[j]-cum_Weibull[j-m])
                        s = (double)(j);
                }
            }
        }
        
        data[i][2] = s+1+dis_int(gen);
        data[i][1] = fmax(0,s-dis_int(gen));
        data[i][0] = m;
    }
}

void    IterationLoop_IQM(int *m1, int *indexmin1, int *iteration1, double *minimum1, int ndata, int ngrid, int ind[],double pp[], double pp0[], double **data, double g[], double tt[], double grid[], double w[])
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
    tt[m]=grid[indexmin];
    
    if (m>1)
    {
        pp[m]=0;
        sort_index(m,ind,tt,pp);
    }
    
    Compute_pp_IQM(m,ndata,pp0,data,tt,w);
    
    if (m==1)
        pp[1]=pp0[1];
    else
        CheckPositivity_IQM(&m,ndata,ind,pp,pp0,data,tt,w);
    
    /* a "feasible" pp is returned from CheckPositivity_IQM */
    
    Compute_g(m,ndata,pp,data,tt,g);
    
    minimum = Compute_minimum_IQM(ndata,ngrid,&indexmin,data,g,grid,w);
    
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

double G(int m, double tt[], double pp[], double *data)
{
    int k;
    double result;
    
    result=0;

    for (k=1;k<=m;k++)
        result += psi(data,tt[k])*pp[k];
    
    return result;
}

double score_numerator(int m, double tt[], double aa[], double *data)
{
    int k;
    double result;
    
    result=0;

    for (k=1;k<=m;k++)
        result += psi(data,tt[k])*aa[k];
    
    return result;
}

double  Compute_process(int ndata, double t, double **data,  double g[])
{
    int i;
    double sum=0;
    
    for (i=0;i<ndata;i++)
    {
        if (g[i]>0)
            sum += psi(data[i],t)/g[i]-1;
    }
    return sum/ndata;
}


void    Compute_g(int m, int ndata, double pp[], double **data, double tt[], double g[])
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
                g[i] += pp[k]*psi(data[i],tt[k]);
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

void    Compute_nabla(int m, int ndata, double **data, double tt[], double g[], double nabla[])
{
    int j,k;
    double sum;
    double tol=1.0e-10;
    
    for (k=1;k<=m;k++)
    {
        {
            sum = 0;
            for (j=0;j<ndata;j++)
                sum+= psi(data[j],tt[k])/fmax(tol,g[j]);
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

// data_point is a vector with the 3 elemants E, S_L and S_R, tt is a possible location of mass

double psi(double *data_point, double tt)
{
    double y;
    
    y=0;
    
    if (tt<data_point[2])
        y += (data_point[2]-tt);
    if (tt<data_point[2]-data_point[0])
        y -= (data_point[2]-data_point[0]-tt);
    if (tt< data_point[1])
        y -= (data_point[1]-tt);
    if (tt < data_point[1]-data_point[0])
        y += (data_point[1]-data_point[0]-tt);
    
    return y;
}

void    CheckPositivity_IQM(int *m1, int ndata, int ind[],
                            double pp[], double pp1[], double **data, double tt[], double w[])
{
    double a,min_p,min1,crit,crit0;
    int m,i,k,imin;
    
    m=*m1;
    
    min_p=0;
    min1=1.0;
    imin=1;
    
    for (i=1;i<=m;i++)
    {
        if (pp1[i]<0)
        {
            crit = pp[i]/(pp[i]-pp1[i]);
            crit0=pp1[i];
            if (crit0<min_p)
                min_p=crit0;
            if (crit<min1)
            {
                min1 = crit;
                imin = i;
            }
        }
    }
    
    if (min_p<0)
    {
        while (min_p<0)
        {
            a = min1;
            for (k=1;k<=m;k++)
                pp1[k]=pp[k]+ a*(pp1[k]-pp[k]);
            if (imin<m)
            {
                for (k= imin;k<m;k++)
                {
                    tt[k]=tt[k+1];
                    pp1[k]=pp1[k+1];
                    ind[k]=ind[k+1];
                }
            }
            
            m -=1;
            
            transfer(m,pp1,pp);
            Compute_pp_IQM(m,ndata,pp1,data,tt,w);
            
            min1 = 1.0;
            min_p = 0;
            
            for (i=1;i<=m;i++)
            {
                if (pp1[i]<0)
                {
                    crit = pp[i]/(pp[i]-pp1[i]);
                    crit0=pp1[i];
                    if (crit0<min_p)
                        min_p=crit0;
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


void ComputeWeights_IQM(int m, int ndata, double **data, double tt[], double pp[], double w[])
{
    int        i,j;
    double    sum,tol=1.0e-10;
    
    /* vector data contains the data points, vector tt contains the (preliminary) locations of the masses */
    
    for (j=0;j<ndata;j++)
    {
        sum=0;
        for (i=1;i<=m;i++)
            sum += pp[i]*psi(data[j],tt[i]);
        
        w[j] = 1.0/fmax(tol,sum);
    }
}


void Compute_critarray(int ndata, int ngrid, double **data, double g[], double grid[], double w[], double critarray[])
{
    int j,k;
    double sum,sum1,sum2;
    
    /* g contains the preliminary solution for the density of the mixing distribution at the observation points */
    
    for (k=1;k<=ngrid;k++)
    {
        sum1 =sum2 = 0;
        for (j=0;j<ndata;j++)
        {
            sum1 -= 2*w[j]*psi(data[j],grid[k]);
            sum2 += g[j]*psi(data[j],grid[k])*SQR(w[j]);
        }
        sum = ndata + sum1 + sum2;
        
        critarray[k]=sum;
    }
}


double Compute_minimum_IQM(int ndata, int ngrid, int *index,
                           double **data, double g[], double grid[], double w[])
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
            sum1 -= 2*w[j]*psi(data[j],grid[k]);
            sum2 += g[j]*psi(data[j],grid[k])*SQR(w[j]);
            sum3 += SQR(psi(data[j],grid[k])*w[j]);
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

void Compute_W1(int m, int ndata, double **data, double tt[], double g[], double **W)
{
    int i,j,k;
    double tol=1.0e-10;
    
    for (i=1;i<=m;i++)
    {
        for (j=1;j<=m;j++)
            W[i][j]=0;
    }
    
    //* data[] contains the data points, tt[] contains the theta[i]'s in Geurt's notation */
    
    for (j=0;j<ndata;j++)
    {
        if (g[j]>tol)
        {
            for (i=1;i<m;i++)
            {
                for (k=1;k<m;k++)
                    W[i][k] += (psi(data[j],tt[i])-psi(data[j],tt[m]))*(psi(data[j],tt[k])-psi(data[j],tt[m]))/(SQR(g[j])*ndata);
            }
        }
    }
}

void Compute_W(int m, int ndata, double **data, double tt[], double w[], double **W)
{
    int i,j,k;
    
    for (i=1;i<=m;i++)
    {
        for (j=1;j<=m;j++)
            W[i][j]=0;
    }
    
    //* data[] contains the data points, tt[] contains the theta[i]'s in Geurt's notation */
    
    for (j=0;j<ndata;j++)
    {
        for (i=1;i<=m;i++)
        {
            for (k=1;k<=m;k++)
                W[i][k] += psi(data[j],tt[i])*psi(data[j],tt[k])*SQR(w[j])/ndata;
        }
    }
}

/*    The following procedure computes the coefficients pp[i].
 At each iteration a linear system of equations is solved
 */

void Compute_pp_IQM(int m, int ndata, double pp[], double **data, double tt[], double w[])
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
    
    Compute_W(m,ndata,data,tt,w,a);
    
    if (m>=1)
    {
        for (i=1;i<=m;i++)
        {
            for (j=0;j<ndata;j++)
                b[i] += 2*psi(data[j],tt[i])*w[j]/ndata;
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


void MLE_IQM(int ndata, int *m1, int ind[], int ngrid, double **data, double tt[],double grid[], double pp[], double MLE[], double **W, double *phi_IQM, int *NumIt)
{
    int           i,m,iteration,iteration2,NumIterations,indexmin;
    double        min_nabla,minimum,*g1,*g2,*g,*w;
    double        norm,phi_old,phi_new,tol=1.0e-10,eps=0.1;
    double        *pp0,*pp1,*pp2,*pp3,*nabla,*nabla1,*nabla2;
    double        l,s;
    
    iteration=iteration2=0;
    NumIterations=1000;
    m=1;
    
    g1 = new double[ndata+1];
    g2 = new double[ndata+1];
    g = new double[ndata+1];
    
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
    
    Compute_g(ngrid,ndata,pp1,data,grid,g1);
    ComputeWeights_IQM(ngrid,ndata,data,grid,pp1,w);
    
    pp[1]=1;
    ind[1]=ngrid/2;
    tt[1]=grid[ngrid/2];
    Compute_g(1,ndata,pp,data,tt,g);

    minimum=Compute_minimum_IQM(ndata,ngrid,&indexmin,data,g,grid,w);

    while (minimum < -tol && iteration2<=NumIterations)
        IterationLoop_IQM(&m,&indexmin,&iteration2,&minimum,ndata,ngrid,ind,
                          pp,pp0,data,g,tt,grid,w);
    
    /*    Interpretation of the following 3 nabla vectors:
     1.    nabla is the vector of derivatives of the criterion function for strictly positive pp_i's.
     2.    nabla1 is the vector of derivatives of the criterion function  in all directions, if one uses pp1[]
     to compute the vector g1[] and from this the vector of derivatives.
     The vector pp1[] is all the time used to generate the weights for the quadratic
     minimization and the outer iterations proceed by adjusting the vector pp1[].
     3.    nabla2 is the vector of derivatives of the criterion function  in all directions, if one uses pp[]
     to compute the vector g[] and from this the vector of derivatives. */
    
    Compute_nabla(m,ndata,data,tt,g,nabla);
    Compute_nabla(ngrid,ndata,data,grid,g1,nabla1);
    Compute_nabla(ngrid,ndata,data,grid,g,nabla2);
    
    phi_new=phi_crit(ngrid,ndata,pp1,g1);
    
    min_nabla=0;
    for (i=1;i<=ngrid;i++)
        if (nabla2[i] < min_nabla) min_nabla = nabla2[i];
    
    
    norm=0;
    
    /*for (i=1;i<=m;i++)
    {
        if (fabs(nabla[i])>norm)
            norm = fabs(nabla[i]);
    }*/
    
    for (i=1;i<=m;i++)
        norm += pp[i]*nabla[i];
        
    norm = fabs(norm);

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
        
        minimum=Compute_minimum_IQM(ndata,ngrid,&indexmin,data,g,grid,w);
        
        while (minimum < -tol && iteration2<=NumIterations)
            IterationLoop_IQM(&m,&indexmin,&iteration2,&minimum,ndata,ngrid,ind,
                              pp,pp0,data,g,tt,grid,w);
        
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
        ComputeWeights_IQM(ngrid,ndata,data,grid,pp1,w);
        
        Compute_pp_IQM(m,ndata,pp0,data,tt,w);
        
        CheckPositivity_IQM(&m,ndata,ind,pp,pp0,data,tt,w);
        Compute_g(m,ndata,pp,data,tt,g);
        
        Compute_nabla(m,ndata,data,tt,g,nabla);
        Compute_nabla(ngrid,ndata,data,grid,g1,nabla1);
        Compute_nabla(ngrid,ndata,data,grid,g,nabla2);
        
        for (i=1,min_nabla=0;i<=ngrid;i++)
            if (nabla2[i] < min_nabla)
                min_nabla = nabla2[i];
        
        /*norm=0;
        for (i=1,norm=0;i<=m;i++)
        {
            if (fabs(nabla[i])>norm)
                norm = fabs(nabla[i]);
        }*/
        
        norm=0;
        for (i=1;i<=m;i++)
            norm += pp[i]*nabla[i];
            
        norm = fabs(norm);
        
        /* If you want to see the iterations, "decomment" the following line. */
        
        //printf("%5d        %12.10f       %12.10f         %13.10f     %8d\n",iteration,phi_new,min_nabla,norm,m);
    }
    
    if (fenchelviol(tol,min_nabla,norm))
        printf("Maximum number of iteration exceeded before criterion was reached");
    
    phi_new=phi_crit(m,ndata,pp,g);
    
    MLE[0]=0;
    
    for (i=1;i<=m;i++)
        MLE[i]=MLE[i-1]+pp[i];
    
    Compute_g(m,ndata,pp,data,tt,g);
    Compute_W1(m,ndata,data,tt,g,W);
    
    /* If you want to see the final iteration with the real phi_new, "decomment" the following lines. */
    
    //printf("%5d        %12.10f       %12.10f         %13.10f     %8d\n",iteration,phi_new,min_nabla,norm,m);
    
    *m1=m;
    *phi_IQM=phi_new;
    *NumIt=iteration;
    
    delete[] g1; delete[] g2; delete[] nabla; delete[] nabla1; delete[] nabla2;
    delete[] pp0; delete[] pp1; delete[] pp2; delete[] pp3; delete[] w;
    
}

void Compute_covar(int ngrid, int NumIt, double **Fvalue, double **covar)
{
  int i,j,iter;
  double *mean;
  
  mean = new double[ngrid];
  
  for (i=1;i<ngrid;i++)
  {
    mean[i]=0;
    for (iter=0;iter<NumIt;iter++)
      mean[i] += Fvalue[iter][i];
    mean[i]/=NumIt;
  }
  
  for (i=1;i<ngrid;i++)
  {
    for (j=1;j<ngrid;j++)
    {
      covar[i][j]=0;
      for (iter=0;iter<NumIt;iter++)
        covar[i][j] += Fvalue[iter][i]*Fvalue[iter][j]-mean[i]*mean[j];
      covar[i][j]/=NumIt;
    }
  }
  delete[] mean;
}

void compute_sigma(int m, int ngrid, double **Fisher_matrix, double *MLE, double *F, double *tt, double *grid, double *sigma)
{
    int i,j,k,m1,**A,*P;
    double tol=1.0e-10;
    double **S,**IS,**B,**B1;
    
    m1=m-1;
    
    S = new double *[ngrid+1];
    for (i=0;i<ngrid+1;i++)
               S[i] = new double [ngrid+1];
        
    IS = new double *[ngrid+1];
    for (i=0;i<ngrid+1;i++)
                   IS[i] = new double [ngrid+1];
    A = new int *[ngrid+1];
        for (i=0;i<ngrid+1;i++)
            A[i] = new int [ngrid+1];
    
    B = new double *[ngrid+1];
        for (i=0;i<ngrid+1;i++)
            B[i] = new double [ngrid+1];
        
    B1 = new double *[ngrid+1];
        for (i=0;i<ngrid+1;i++)
            B1[i] = new double [ngrid+1];
        
    for (i=0;i<=ngrid;i++)
    {
        for (j=0;j<=i;j++)
            A[i][j]=1;
        if (i<ngrid)
        {
            for (j=i+1;j<=ngrid;j++)
                A[i][j]=0;
        }
    }
        
    P = new int[ngrid+1];
    
    for (i=0;i<m1;i++)
    {
        for (j=0;j<m1;j++)
            S[i][j] = Fisher_matrix[i+1][j+1];
    }
    
    LUPDecompose(S,m1,tol,P);
    LUPInvert(S,P,m1,IS);
    
    for (i=0;i<m1;i++)
    {
        for (j=0;j<m1;j++)
            B[i][j]=B1[i][j]=0;
    }
    
    for (i=0;i<m1;i++)
    {
        for (j=0;j<m1;j++)
            for (k=0;k<m1;k++)
                B1[i][j] += A[i][k]*IS[k][j];
    }
    
    for (i=0;i<m1;i++)
    {
        for (j=0;j<m1;j++)
            for (k=0;k<m1;k++)
                B[i][j] += B1[i][k]*A[j][k];
    }
    
    F[0]=0;
    for (i=0;i<=ngrid;i++)
    {
        //if (grid[i]<tt[1])
            sigma[i]=0;
        for (k=1;k<=m;k++)
        {
            if (tt[k-1]<=grid[i] && grid[i]<tt[k])
            {
                F[i] = MLE[k-1];
                if (k>1)
                    sigma[i]=sqrt(B[k-2][k-2]);
            }
        }
        if (grid[i]>=tt[m])
        {
            F[i]=1;
            sigma[i]=0;
        }
    }
    
    for (i=0;i<=ngrid;i++)
    {
        if (grid[i]<tt[1])
            sigma[i]=sqrt(B[0][0]);
    }
    
    for (i=0;i<ngrid+1;i++)
        delete[] A[i];
    delete[] A;
    for (i=0;i<ngrid+1;i++)
        delete[] B[i];
    delete[] B;
    for (i=0;i<ngrid+1;i++)
        delete[] B1[i];
    delete[] B1;
    for (i=0;i<ngrid+1;i++)
        delete[] S[i];
    delete[] S;
    for (i=0;i<ngrid+1;i++)
        delete[] IS[i];
    delete[] IS;
    delete[] P;
}

void MLE_to_F(int m, int ngrid0, double grid[], double tt[], double MLE[], double F[])
{
    int i,k;
    
    F[0]=0;
    for (i=1;i<=ngrid0;i++)
    {
        for (k=1;k<=m;k++)
        {
            if (tt[k-1]<=grid[i] && grid[i]<tt[k])
                F[i] = MLE[k-1];
        }
        if (grid[i]>=tt[m])
            F[i]=1;
    }
}
    
int compare(const void *a, const void *b)
{
    if ((*(data_object *) a).ind < (*(data_object *) b).ind)
        return -1;
    if ((*(data_object *) a).ind > (*(data_object *) b).ind)
        return 1;
    return 0;
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
    
    qsort(obs,m,sizeof(data_object),compare);
    
    for (i=0;i<m;i++)
    {
        ind[i+1]=obs[i].ind;
        alpha[i+1]=obs[i].alpha;
        pp[i+1]=obs[i].pp;
    }
    
    delete[] obs;
}

double Weibull_inv(double a, double b, double M1, double u)
{
  double c,v;
  
  c= 1-exp(-b*pow(M1,a));
  
  if (u>=1)
    return M1;
  if (u<=0)
    return 0;
  
  v = pow(b,-1.0/a)*pow(log(1/(1-c*u)),1.0/a);
  
  return v;
}

void transfer(int n, double a[], double b[])
{
  int    i;
  for (i = 1; i<= n;i++)    b[i] = a[i];
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

void LUPInvert(double **A, int *P, int N, double **IA)
{
  for (int j = 0; j < N; j++)
  {
    for (int i = 0; i < N; i++)
    {
      if (P[i] == j)
        IA[i][j] = 1.0;
      else
        IA[i][j] = 0.0;
      
      for (int k = 0; k < i; k++)
        IA[i][j] -= A[i][k] * IA[k][j];
    }
    for (int i = N - 1; i >= 0; i--)
    {
      for (int k = i + 1; k < N; k++)
        IA[i][j] -= A[i][k] * IA[k][j];
      
      IA[i][j] = IA[i][j] / A[i][i];
    }
  }
  
}      

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











