#include <math.h>
#include <string.h>
#include "mex.h"
#include "blas.h"
#include "lapack.h"
const ptrdiff_t ione = 1;
const double fone = 1.0;
const double fzero = 0.0;
void EIQP(double *Q, double *c, double *A, double *b, double epsilon, ptrdiff_t nc, ptrdiff_t nb, double *z, double *y, double *tau, double *v, double *w, double *kappa);
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	ptrdiff_t i, j, k, nc, nb;
	double *Q, *c, *A, *b, epsilon, *Q_copy, *c_copy, *A_copy, *b_copy;
	double *z, *y, *tau, *v, *w, *kappa;
	Q = mxGetPr(prhs[0]);
    c = mxGetPr(prhs[1]);
    nc = mxGetM(prhs[0]); 
    A = mxGetPr(prhs[2]);
    b = mxGetPr(prhs[3]);
    nb = mxGetM(prhs[2]);
    epsilon = mxGetScalar(prhs[4]);
	plhs[0] = mxCreateDoubleMatrix(nc,1,mxREAL);
    z = mxGetPr(plhs[0]);
    y = malloc(sizeof(double)*nb);
    tau = malloc(sizeof(double)*1);
    v = malloc(sizeof(double)*nc);
    w = malloc(sizeof(double)*nb);
    kappa = malloc(sizeof(double)*1);
    Q_copy = malloc(sizeof(double)*nc*nc);
    c_copy = malloc(sizeof(double)*nc);
    A_copy = malloc(sizeof(double)*nb*nc);
    b_copy = malloc(sizeof(double)*nb);
    memcpy(Q_copy,Q,sizeof(double)*nc*nc);
    memcpy(c_copy,c,sizeof(double)*nc);
    memcpy(A_copy,A,sizeof(double)*nb*nc);
    memcpy(b_copy,b,sizeof(double)*nb);
    for(i=0;i<nc;i++)
    {
    	z[i] = 1.0;
    	v[i] = 1.0;
    }
    for(i=0;i<nb;i++)
    {
    	y[i] = 1.0;
    	w[i] = 1.0;
    }
    tau[0] = 1.0; 
    kappa[0] = 1.0;
    /*===========================================*/
    EIQP(Q_copy,c_copy,A_copy,b_copy,epsilon,nc,nb,z,y,tau,v,w,kappa);
    /*===========================================*/
    if(kappa[0]<tau[0])
    {
    	plhs[1] = mxCreateString("Optimal");
    	for(i=0;i<nc;i++)
    		z[i] = z[i]/tau[0];
    }
    else
    {
    	plhs[1] = mxCreateString("Infeasible");
    	for(i=0;i<nc;i++)
    		z[i] = 0.0;
    }
    free(y); free(tau); free(v); free(w); free(kappa); free(Q_copy); free(c_copy); free(A_copy); free(b_copy);
    return;
}

void EIQP(double *Q, double *c, double *A, double *b, double epsilon, ptrdiff_t nc, ptrdiff_t nb, double *z, double *y, double *tau, double *v, double *w, double *kappa)
{
	ptrdiff_t i, j, iter, max_iter, info, n, nc_nc, nb_nc, *ipiv;
	double eta, gamma;
	double *Qz, *ATy, *Az, zTc, yTb, zTQz, sigma, tmp, *r_z, *r_y, r_tau, *M0, *M, mu_bar, *u;
	n = nc+nb+1;
	nc_nc = nc*nc;
	nb_nc = nb*nc;
    ipiv = malloc(sizeof(ptrdiff_t)*n);
	Qz = malloc(sizeof(double)*nc);
	ATy = malloc(sizeof(double)*nc);
	Az = malloc(sizeof(double)*nb);
	r_z = malloc(sizeof(double)*nc);
	r_y = malloc(sizeof(double)*nb);
	M0 = malloc(sizeof(double)*n*n);
	M = malloc(sizeof(double)*n*n);
	u = malloc(sizeof(double)*n);
	eta = 0.414213/sqrt(n);
	gamma = 1.0-eta;
	dgemv("n",&nc,&nc,&fone,Q,&nc,z,&ione,&fzero,Qz,&ione); // Qz = Q*z;
	dgemv("t",&nb,&nc,&fone,A,&nb,y,&ione,&fzero,ATy,&ione); // ATy = A'*y;
	dgemv("n",&nb,&nc,&fone,A,&nb,z,&ione,&fzero,Az,&ione); // Az = A*z;
	zTc = ddot(&nc,z,&ione,c,&ione); // zTc = z'*c;
	yTb = ddot(&nb,y,&ione,b,&ione); // yTb = y'*b;
	zTQz = ddot(&nc,z,&ione,Qz,&ione); // zTQz = z'*Q*z;
	sigma = (1.0>(-zTQz-zTc+yTb))?1.0:(-zTQz-zTc+yTb);
	for(i=0;i<nc;i++)
		sigma = (sigma>(Qz[i]-ATy[i]+c[i]))?sigma:(Qz[i]-ATy[i]+c[i]);
	for(i=0;i<nb;i++)
		sigma = (sigma>(Az[i]-b[i]))?sigma:(Az[i]-b[i]);
	tmp = 1.0/sigma;
	dscal(&nc_nc,&tmp,Q,&ione);
	dscal(&nc,&tmp,c,&ione);
	dscal(&nb_nc,&tmp,A,&ione);
	dscal(&nb,&tmp,b,&ione);
	dscal(&nc,&tmp,Qz,&ione);
	dscal(&nc,&tmp,ATy,&ione);
	dscal(&nb,&tmp,Az,&ione);
	zTc = tmp*zTc;
	yTb = tmp*yTb;
	zTQz = tmp*zTQz;
	max_iter = (int)ceil(log(n/epsilon)/(-log(gamma)));
	/*========= Start: assign M0 ===========================================*/
	// M0 = [Q, -A', c;
	//       A, zeros(nb,nb), -b;
	//       -c', b', 0];
	memset(M0,0.0,sizeof(double)*n*n);
	for(i=0;i<nc;i++)
	{
		for(j=0;j<nc;j++)
			M0[i*n+j] = Q[i*nc+j];
		for(j=nc;j<nc+nb;j++)
			M0[i*n+j] = A[i*nb+(j-nc)];
		M0[i*n+(nc+nb)] = -c[i];
	}
	for(i=nc;i<nc+nb;i++)
	{
		for(j=0;j<nc;j++)
			M0[i*n+j] = -A[j*nb+(i-nc)];
		M0[i*n+(nc+nb)] = b[i-nc];
	}
	for(j=0;j<nc;j++)
		M0[(nc+nb)*n+j] = c[j];
	for(j=nc;j<nc+nb;j++)
		M0[(nc+nb)*n+j] = -b[j-nc];
	/*========= End: assign M0 ===========================================*/
	for(i=0;i<nc;i++)
		r_z[i] = v[i] - Qz[i] + ATy[i] - tau[0]*c[i];
	for(i=0;i<nb;i++)
		r_y[i] = w[i] - Az[i] + tau[0]*b[i];
	r_tau = kappa[0] + zTQz/tau[0] + zTc - yTb;
	for(iter=1;iter<=max_iter;iter++)
    {
    	mu_bar = (tau[0]*kappa[0] + ddot(&nc,z,&ione,v,&ione) + ddot(&nb,y,&ione,w,&ione))/n; // mu = (z'*v+y'*w+tau*kappa)/(n+1)
    	for(i=0;i<nc;i++)
    		u[i] = gamma*mu_bar/z[i] - v[i] + eta*r_z[i];
    	for(i=0;i<nb;i++)
    		u[i+nc] = gamma*mu_bar/y[i] - w[i] + eta*r_y[i];
    	u[nc+nb] = gamma*mu_bar/tau[0] - kappa[0] + eta*r_tau;
    	/*========= Start: assign M ===========================================*/
    	memcpy(M,M0,sizeof(double)*n*n);
    	for(i=0;i<nc;i++)
    		M[i*n+i] += v[i]/z[i];
    	for(i=nc;i<nc+nb;i++)
    		M[i*n+i] += w[i-nc]/y[i-nc];
    	M[(nc+nb)*n+(nc+nb)] += (zTQz/(tau[0]*tau[0]) + kappa[0]/tau[0]);
    	for(i=0;i<nc;i++)
    		M[i*n+(nc+nb)] -= 2.0*Qz[i]/tau[0];
    	/*========= End: assign M ===========================================*/
    	// solve the linear Newton system
    	dgesv(&n,&ione,M,&n,ipiv,u,&n,&info);
    	for(i=0;i<nc;i++)
    		z[i] += u[i];
    	for(i=0;i<nb;i++)
    		y[i] += u[i+nc];
    	tau[0] += u[nc+nb];        
	    dgemv("n",&nc,&nc,&fone,Q,&nc,z,&ione,&fzero,Qz,&ione); // Qz = Q*z;
	    dgemv("t",&nb,&nc,&fone,A,&nb,y,&ione,&fzero,ATy,&ione); // ATy = A'*y;
	    dgemv("n",&nb,&nc,&fone,A,&nb,z,&ione,&fzero,Az,&ione); // Az = A*z;
	    zTc = ddot(&nc,z,&ione,c,&ione); // zTc = z'*c;
	    yTb = ddot(&nb,y,&ione,b,&ione); // yTb = y'*b;
	    zTQz = ddot(&nc,z,&ione,Qz,&ione); // zTQz = z'*Q*z;        
		for(i=0;i<nc;i++)
			v[i] = Qz[i] - ATy[i] +tau[0]*c[i] + gamma*r_z[i];
		for(j=0;j<nb;j++)
			w[j] = Az[j] - tau[0]*b[j] + gamma*r_y[j];
		kappa[0] = -zTQz/tau[0] - zTc + yTb + gamma*r_tau;      
		dscal(&nc,&gamma,r_z,&ione);
		dscal(&nb,&gamma,r_y,&ione);
		r_tau = gamma*r_tau;
    }
    free(M0); free(M);
    return;
}
