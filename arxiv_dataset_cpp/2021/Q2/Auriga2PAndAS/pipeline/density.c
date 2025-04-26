#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#define TAILLEMAX 1000


// Saturation for region with more than 30 stars per arcsec2
// So we remove region with stars closer than sqrt(30)arcmin


long * density(double *xki, double *eta, double *g, long length){

	long i,j;
	long * flag;
	double R; // distance between two stars
	double sat_len=1.0/(60.0*sqrt(30.0)); // Saturation len


	flag=(long *)malloc(length*sizeof(long));

	#pragma omp parallel for
	for(i=0;i<length;i++) flag[i]=1; // Set flag at 1

	#pragma omp parallel for private(j,R)
	for(i=0;i<length;i++){
		for(j=(i+1);j<length;j++){
			if(xki[j]<=xki[i]+sat_len){
				R=sqrt(pow(xki[j]-xki[i],2.0)+pow(eta[j]-eta[i],2.0));
				if(R<=sat_len){ // If saturation remove the less bright stars in g
					if(g[i]<g[j]) flag[j]=0; else flag[i]=0;	
				}
			}else j=length+10;
		}
	}

	
	return flag;
}
