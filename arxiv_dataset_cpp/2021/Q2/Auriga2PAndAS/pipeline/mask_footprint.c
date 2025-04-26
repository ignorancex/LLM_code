#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#define TAILLEMAX 1000

// To compile gcc -fPIC -shared mask_footprint.c -o mask_footprint.so -lm


/******************************************************
 ** Code to know if you are inside a polygone or not **
 ******************************************************/
double fimag(double x0, double xs, double xe, double y0, double ys, double ye){
	double top,bot;
	top= -(xe-x0) * (ys-y0) + (ye-y0) * (xs-x0);
	bot=  (xe-x0) * (xs-x0) + (ye-y0) * (ys-y0);
	return atan2(top,bot);
}


int in_poly(double x, double y, int np, double xp[], double yp[]){
     

	int j; 
	double tiny,xs,xe,ys,ye;
	double simag;

	tiny=1.e-5;
	simag=0.0;

	for(j=0;j<np;j++){
		if (j<np-1){
			xe=xp[j+1];
			xs=xp[j];
			ye=yp[j+1];
			ys=yp[j];
		}else{
			xe=xp[0];
			xs=xp[j];
			ye=yp[0];
			ys=yp[j];
		}
        simag+=fimag(x,xs,xe,y,ys,ye);
	}

	if (fabs(simag)>tiny){
		return 1;
	}else{
		return 0;
	}
}






/*****************************
 *** MAIN PART OF THE CODE ***
 *****************************/
double * mask(double *data_xki, double *data_eta, double *xki1, double *eta1, double *xki2, double *eta2, double *xki3, double *eta3, double *xki4, double *eta4, long *nb_field, double *lim_g, double *lim_i, long len_data, long len_field, long len_lim){

	long i,j,tmp;
	double * out;
	double xki[4],eta[4];
	out=(double *)malloc(len_data*3*sizeof(double));

	#pragma omp parallel for private(j,xki,eta,tmp)
	for(i=0;i<len_data;i++){ // For each particules
		out[i]=0;
		out[i+len_data]=0;
		out[i+2*len_data]=0;
		for(j=0;j<len_field;j++){ // For each CCD i neach field
			xki[0]=xki1[j]; eta[0]=eta1[j];
			xki[1]=xki2[j]; eta[1]=eta2[j];
			xki[2]=xki3[j]; eta[2]=eta3[j];
			xki[3]=xki4[j]; eta[3]=eta4[j];
			tmp=in_poly(data_xki[i], data_eta[i], 4, xki, eta);
			if(tmp==1){ // If inside the polygone
				out[i]=nb_field[j];
				out[i+len_data]=lim_g[nb_field[j]-1]; // Because starts a zeros not one
				out[i+2*len_data]=lim_i[nb_field[j]-1];
				j=len_field+10;
			}
		}
	}


	return out;
}
