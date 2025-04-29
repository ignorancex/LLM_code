/* The code is taken from http://www.cs.utexas.edu/~cjhsieh/nmf/
 * Latest updated by LTK Hien, July 2020
 *
 *  Usage: [W H objKL timeKL] = ccd_KL(V, k, max_iter, Winit, Hinit, trace);
 *
 * Given the nonnegative input matrix V, this code solves the following KL-NMF problem to find the low-rank approximation WH for V. 
 *
 *  min_{W>=0,H>=0} sum_{i,j} V_{ij}*log(V_{ij}/(WH)_{ij})
 *
 *  Input arguments
 *  	V: n by m nonnegative input matrix.
 *  	k: rank of output matrices W and H. 
 *  	max_iter: maximum iteration. 
 *  	Winit: k by n initial matrix for W. 
 *  	Hinit: k by m initial matrix for H. 
 *  	trace: 1: compute objective value per iteration. 
 *  		   0: do not compute objective value per iteration. (default)
 *
 *  Output arguments
 *  	W: k by n dense matrix.
 *  	H: k by m dense matrix.
 *  	objKL: objective values.
 *  	timeKL: time taken by this algorithm. 
 *
 */

#include "math.h"
#include "mex.h" 
#include <time.h>

double obj(int n, int m, double *V, double *WH)
{
	double total = 0;
	for ( int i=0 ; i<n*m ; i++ )
		total = total + V[i]*log((V[i]+1e-5)/(WH[i]+1e-15))-V[i]+WH[i];
	return (total);
}

void update(int m, int k, double *Wt, double *WHt, double *Vt, double *H)
{
	int maxinner = 2;
	for ( int q=0 ; q<k ; q++ )
	{
		for (int inneriter =0 ; inneriter<maxinner ; inneriter++)
		{
			double g=0, h=0, tmp, s, oldW, newW, diff;
			for (int j=0, hind=q ; j<m ; j++, hind+=k )
			{	
				tmp = (Vt[j])/(WHt[j]+1e-10);
				g = g + H[hind]*(1-tmp); // 1-V/WH
				h = h + tmp*H[hind]*H[hind]/(WHt[j]+1e-10);    //V/WH^2
			}
			s = -g/h;
			oldW = Wt[q];
			newW = Wt[q]+s;
			if ( newW < 1e-15)
				newW = 1e-15;
			diff = newW-oldW;
			Wt[q] = newW;
			for ( int j=0 ; j<m ; j++)
            {	WHt[j] = WHt[j]+diff*H[j*k+q];
                 if (WHt[j]< 1e-16) // added by Hien to avoid numerical error when WHt[j] <0
                           WHt[j]=1e-16;
            }

			if ( fabs(diff) < fabs(oldW)*0.5 )
				break;
		}
	}
}

int newKL(int n, int m, int k, int maxiter, double maxtime, double *V, double *W, double *H, int trace, double *objlist, double *timelist)
{ // Hien added a time constraint to exit the program when max time is reached. 
  // and the function returns the real number of iterations run within the time constraint 
	
    char matlab_output[1024];
	double total = 0, begin;
	double *WH = (double *)malloc(sizeof(double)*n*m);

	// temp arrays when updating variables in W (since V and WH are stored in column format)
	double *WHt = (double *)malloc(sizeof(double)*m); 
	double *Vt = (double *)malloc(sizeof(double)*m);
    int reallength=0;
	begin = clock();
	for ( int i=0, ind=0 ; i<m ; i++ )
		for ( int j=0 ; j<n ; j++, ind++ )
		{
			WH[ind] = 0;
			int indw = j*k, indh = i*k;
			for (int r=0 ; r<k ; r++ )
				WH[ind] += W[indw+r]*H[indh+r];
		}
	total = (clock()-begin)/CLOCKS_PER_SEC;
    // Hien added some lines to calculate the objective value of the initial point
   	timelist[0] = total;
    objlist[0] = obj(n,m,V,WH);
    
     
	for ( int iter=1 ; iter<(maxiter+1) ; iter++) // counter is set from 1
	{   reallength +=1;
		double begin = clock(); // for counting the running time

		// Update W
		for ( int i=0 ; i<n ; i++)
		{
			double  *Wt = &(W[i*k]);
			for ( int j=0 ; j<m ; j++ )
			{
				WHt[j] = WH[j*n+i];
				Vt[j] = V[j*n+i];
			}
			update(m, k, Wt, WHt, Vt, H);
			for ( int j=0 ; j<m ; j++ )
				WH[j*n+i] = WHt[j];
		}

		// Update H
		for ( int i=0 ; i<m ; i++ )
		{
			double *Ht = &(H[i*k]);
			double *wht = &(WH[i*n]);
			double *vt = &(V[i*n]);

			update(n,k,Ht,wht,vt,W);
		}

		if ( trace == 1 ) 
		{
			total += (clock()-begin)/CLOCKS_PER_SEC;
			timelist[iter] = total;
            
            objlist[iter] = obj(n,m,V,WH);
            
			// printf will not flush the output buffer
            sprintf(matlab_output, "display('Klnmf:Iteration %d Objective: %lf Time taken: %lf');", iter, objlist[iter], timelist[iter]);
			mexEvalString(matlab_output);
          
		}
		else {
			sprintf(matlab_output, "display('Iteration %d')", iter);
			mexEvalString(matlab_output);
		}
        if (total>maxtime)
         break;
         
	}

	free(WH);
	free(WHt);
	free(Vt);
    return (reallength+1);
}

void usage()
{
	printf("Error calling KL_NMF.\n");
	printf("Usage: [W H objKL timeKL] = ccd_KL(V, k, max_iter, Winit, Hinit, trace=0)\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *xValues;
	int i,j, reallength;
	double avg;
	double *V, *W, *H;
	// double *timelist = NULL, *objlist = NULL;
    double *time1=NULL, *obj1=NULL; 
    
	int n,m, k, maxiter, maxtime;
	int trace = 0;
	double *outArray;

	// Check input/output number of arguments
	if ( nlhs > 8 )
	{
		usage();
		printf("Number of input or output arguments are not correct.\n");
		return;
	}

	V = mxGetPr(prhs[0]);
	n = mxGetM(prhs[0]);
	m = mxGetN(prhs[0]);

	k = mxGetScalar(prhs[1]);
	maxiter = mxGetScalar(prhs[2]);
    maxtime =  mxGetScalar(prhs[3]);
    
	W = mxGetPr(prhs[4]);
	if ( mxGetM(prhs[4]) != k || mxGetN(prhs[4])!=n ) {
		usage();
		printf("Error: Winit should be a %d by %d matrix. \n", k, n);
		return;
	}

	H = mxGetPr(prhs[5]);
	if ( mxGetM(prhs[5]) != k || mxGetN(prhs[5])!=m ) {
		usage();
		printf("Error: Hinit should be a %d by %d matrix. \n", k, m);
		return;
	}
	// use clone matrices such that CPP does not change the initial value, this initial is used for other algorithms
    double *W_clone = (double *)malloc(sizeof(double)*k*n); 
    double *H_clone = (double *)malloc(sizeof(double)*k*m);
    
    for(int idx=0; idx<k*n;idx++)
        W_clone[idx]=W[idx];
    for(int idx=0; idx<k*m;idx++)
        H_clone[idx]=H[idx];
    
	if ( nrhs>6 )
		trace = mxGetScalar(prhs[6]);

	if ( trace==0 && nlhs >2 )
	{
		usage();
		printf("Error: only 2 output matrices (W, H) when trace = 0.\n");
		return;
	}
    
    
  double *objlist=(double *)malloc(sizeof(double)*maxiter); 
  double *timelist=(double *)malloc(sizeof(double)*maxiter);

 reallength=newKL(n,m, k,maxiter,maxtime,V,W_clone,H_clone, trace, objlist, timelist);
 
 if ( trace==1 )
	{
		plhs[2] = mxCreateDoubleMatrix(1,reallength,mxREAL); // real number of iterations
		obj1=mxGetPr(plhs[2]);
        for (int cnt=0; cnt<reallength;cnt++)
           obj1[cnt]=objlist[cnt];
        free(objlist);
        
		plhs[3] = mxCreateDoubleMatrix(1,reallength,mxREAL);
		time1 = mxGetPr(plhs[3]);
        for (int cnt=0; cnt<reallength;cnt++)
           time1[cnt]=timelist[cnt];
        free(timelist);
	}

	plhs[0] = mxCreateDoubleMatrix(k,n,mxREAL);
	outArray=mxGetPr(plhs[0]);
	for ( i=0 ; i<k*n ; i++ )
		outArray[i] = W_clone[i];
    
	plhs[1] = mxCreateDoubleMatrix(k,m,mxREAL);
	outArray=mxGetPr(plhs[1]);
	for ( i=0 ; i<k*m ; i++ )
		outArray[i] = H_clone[i];
   
    free(W_clone);
    free(H_clone);
	return;
}
