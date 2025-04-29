#include <stdio.h>
#include <math.h>
#include <complex.h>

#define PI 3.142857

void fft_new(float *x, Int32 nfft)
{
	Int32 N;
	complex double *y;
	complex double temp;
	int k,n;

	N = sizeof(x)/sizeof(float);

	for(k = 0; k < nfft; k++)
	{
		for(n = 0; n < N; n++)
		{
			temp = exp(-2*I*PI*(k-1)*(n-1)/nfft)
			creal(y[k]) = creal(y[k]) + x[k]*creal(temp);
			cimag(y[k]) = cimag(y[k]) + x[k]*cimag(temp);
		}
	}
	return (y);
}	

