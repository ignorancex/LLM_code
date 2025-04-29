#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cblas.h>
#include <lapacke.h>

const int ione = 1;
const double fone = 1.0;
const double fzero = 0.0;

void EIQP(double *Q_copy, double *c_copy, double *A_copy, double *b_copy, double epsilon, int nc, int nb, double *z, char *status)
{
    int i, j, k, iter, max_iter, info, n, nc_nc, nb_nc, *ipiv;
	double *Q, *c, *A, *b;
    double *y, tau, *v, *w, kappa;
    double eta, gamma;
	double *Qz, *ATy, *Az, zTc, yTb, zTQz, sigma, tmp, *r_z, *r_y, r_tau, *M0, *M, mu_bar, *u;

	n = nc+nb+1;
	nc_nc = nc*nc;
	nb_nc = nb*nc;

	Q = malloc(sizeof(double)*nc*nc);
	c = malloc(sizeof(double)*nc);
	A = malloc(sizeof(double)*nb*nc);
	b = malloc(sizeof(double)*nb);
    y = malloc(sizeof(double)*nb);
    v = malloc(sizeof(double)*nc);
    w = malloc(sizeof(double)*nb);

    ipiv = malloc(sizeof(double)*n);
	Qz = malloc(sizeof(double)*nc);
	ATy = malloc(sizeof(double)*nc);
	Az = malloc(sizeof(double)*nb);
	r_z = malloc(sizeof(double)*nc);
	r_y = malloc(sizeof(double)*nb);
	M0 = malloc(sizeof(double)*n*n);
	M = malloc(sizeof(double)*n*n);
	u = malloc(sizeof(double)*n);

	memcpy(Q,Q_copy,sizeof(double)*nc*nc);
	memcpy(c,c_copy,sizeof(double)*nc);
	memcpy(A,A_copy,sizeof(double)*nb*nc);
	memcpy(b,b_copy,sizeof(double)*nb);
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
    tau = 1.0; 
    kappa = 1.0;
    eta = 0.414213/sqrt(n);
	gamma = 1.0-eta;

	cblas_dgemv(CblasColMajor,CblasNoTrans,nc,nc,fone,Q,nc,z,ione,fzero,Qz,ione); // dgemv("n",&nc,&nc,&fone,Q,&nc,z,&ione,&fzero,Qz,&ione); // Qz = Q*z;
	cblas_dgemv(CblasColMajor,CblasTrans,nb,nc,fone,A,nb,y,ione,fzero,ATy,ione); // dgemv("t",&nb,&nc,&fone,A,&nb,y,&ione,&fzero,ATy,&ione);  // ATy = A'*y;
	cblas_dgemv(CblasColMajor,CblasNoTrans,nb,nc,fone,A,nb,z,ione,fzero,Az,ione); // dgemv("n",&nb,&nc,&fone,A,&nb,z,&ione,&fzero,Az,&ione); // Az = A*z;
	zTc = cblas_ddot(nc,z,ione,c,ione); // zTc = ddot(&nc,z,&ione,c,&ione);  // zTc = z'*c;
	yTb = cblas_ddot(nb,y,ione,b,ione); // yTb = ddot(&nb,y,&ione,b,&ione); // yTb = y'*b;
	zTQz = cblas_ddot(nc,z,ione,Qz,ione); // zTQz = ddot(&nc,z,&ione,Qz,&ione); // zTQz = z'*Q*z;
	sigma = (1.0>(-zTQz-zTc+yTb))?1.0:(-zTQz-zTc+yTb);
	for(i=0;i<nc;i++)
		sigma = (sigma>(Qz[i]-ATy[i]+c[i]))?sigma:(Qz[i]-ATy[i]+c[i]);
	for(i=0;i<nb;i++)
		sigma = (sigma>(Az[i]-b[i]))?sigma:(Az[i]-b[i]);

	tmp = 1.0/sigma;
	cblas_dscal(nc_nc,tmp,Q,ione); // dscal(&nc_nc,&tmp,Q,&ione);
	cblas_dscal(nc,tmp,c,ione); // dscal(&nc,&tmp,c,&ione);
	cblas_dscal(nb_nc,tmp,A,ione); // dscal(&nb_nc,&tmp,A,&ione);
	cblas_dscal(nb,tmp,b,ione); // dscal(&nb,&tmp,b,&ione);
	cblas_dscal(nc,tmp,Qz,ione); // dscal(&nc,&tmp,Qz,&ione);
	cblas_dscal(nc,tmp,ATy,ione); // dscal(&nc,&tmp,ATy,&ione);
	cblas_dscal(nb,tmp,Az,ione); // dscal(&nb,&tmp,Az,&ione);
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
		r_z[i] = v[i] - Qz[i] + ATy[i] - tau*c[i];
	for(i=0;i<nb;i++)
		r_y[i] = w[i] - Az[i] + tau*b[i];
	r_tau = kappa + zTQz/tau + zTc - yTb;
	for(iter=1;iter<=max_iter;iter++)
    {
		mu_bar = (tau*kappa + cblas_ddot(nc,z,ione,v,ione) + cblas_ddot(nb,y,ione,w,ione))/n; // mu_bar = (tau[0]*kappa[0] + ddot(&nc,z,&ione,v,&ione) + ddot(&nb,y,&ione,w,&ione))/n; // mu = (z'*v+y'*w+tau*kappa)/(n+1)
    	for(i=0;i<nc;i++)
    		u[i] = gamma*mu_bar/z[i] - v[i] + eta*r_z[i];
    	for(i=0;i<nb;i++)
    		u[i+nc] = gamma*mu_bar/y[i] - w[i] + eta*r_y[i];
    	u[nc+nb] = gamma*mu_bar/tau - kappa + eta*r_tau;

    	/*========= Start: assign M ===========================================*/
    	memcpy(M,M0,sizeof(double)*n*n);
    	for(i=0;i<nc;i++)
    		M[i*n+i] += v[i]/z[i];
    	for(i=nc;i<nc+nb;i++)
    		M[i*n+i] += w[i-nc]/y[i-nc];
    	M[(nc+nb)*n+(nc+nb)] += (zTQz/(tau*tau) + kappa/tau);
    	for(i=0;i<nc;i++)
    		M[i*n+(nc+nb)] -= 2.0*Qz[i]/tau;
    	/*========= Start: assign M ===========================================*/

    	// solve the linear Newton system
    	info = LAPACKE_dgesv(LAPACK_COL_MAJOR,n,ione,M,n,ipiv,u,n); // dgesv(&n,&ione,M,&n,ipiv,u,&n,&info);

    	for(i=0;i<nc;i++)
    		z[i] += u[i];
    	for(i=0;i<nb;i++)
    		y[i] += u[i+nc];
    	tau += u[nc+nb];
        
	    cblas_dgemv(CblasColMajor,CblasNoTrans,nc,nc,fone,Q,nc,z,ione,fzero,Qz,ione); // dgemv("n",&nc,&nc,&fone,Q,&nc,z,&ione,&fzero,Qz,&ione); // Qz = Q*z;
	    cblas_dgemv(CblasColMajor,CblasTrans,nb,nc,fone,A,nb,y,ione,fzero,ATy,ione); // dgemv("t",&nb,&nc,&fone,A,&nb,y,&ione,&fzero,ATy,&ione); // ATy = A'*y;
		cblas_dgemv(CblasColMajor,CblasNoTrans,nb,nc,fone,A,nb,z,ione,fzero,Az,ione); // dgemv("n",&nb,&nc,&fone,A,&nb,z,&ione,&fzero,Az,&ione); // Az = A*z;
		zTc = cblas_ddot(nc,z,ione,c,ione); // zTc = ddot(&nc,z,&ione,c,&ione);  // zTc = z'*c;
		yTb = cblas_ddot(nb,y,ione,b,ione); // yTb = ddot(&nb,y,&ione,b,&ione);  // yTb = y'*b;
		zTQz = cblas_ddot(nc,z,ione,Qz,ione); // zTQz = ddot(&nc,z,&ione,Qz,&ione); // zTQz = z'*Q*z;
        
		for(i=0;i<nc;i++)
			v[i] = Qz[i] - ATy[i] +tau*c[i] + gamma*r_z[i];
		for(j=0;j<nb;j++)
			w[j] = Az[j] - tau*b[j] + gamma*r_y[j];
		kappa = -zTQz/tau - zTc + yTb + gamma*r_tau;
        
		cblas_dscal(nc,gamma,r_z,ione); // dscal(&nc,&gamma,r_z,&ione);
		cblas_dscal(nb,gamma,r_y,ione); // dscal(&nb,&gamma,r_y,&ione);
		r_tau = gamma*r_tau;
    }

    if(kappa<tau)
    {
        strcpy(status, "Optimal");
    	for(i=0;i<nc;i++)
    		z[i] = z[i]/tau;
    }
    else
    {
        strcpy(status, "Infeasibile");
    	for(i=0;i<nc;i++)
    		z[i] = 0.0；
    }
    free(M0); free(M);
    return;
}
