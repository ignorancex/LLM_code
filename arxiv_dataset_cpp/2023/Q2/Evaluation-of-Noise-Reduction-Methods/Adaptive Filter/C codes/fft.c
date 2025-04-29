#include <stdio.h>
#include <math.h>

#define PI 3.142857

struct complex
{
  int real, img;
};

struct complex *fft(float *x, int nfft)
{
	int N;
	struct complex temp;
	struct complex *y;
	int k,n;

	N = sizeof(x)/sizeof(float);

	for(k = 0; k < nfft; k++)
	{	
		(y[k]).real=0;
		(y[k]).img=0;
		for(n = 0; n < N; n++)
		{
			//temp = exp(-2*I*PI*(k-1)*(n-1)/nfft)
			temp.real = cos(-2*PI*(k-1)*(n-1)/nfft);
			temp.img = sin(-2*PI*(k-1)*(n-1)/nfft);
			(y[k]).real = (y[k]).real + x[k]*(temp).real;
			(y[k]).img = (y[k]).img + x[k]*(temp).img;
			//printf("Sum = %d + %di", (y[k]).real, (y[k]).img);
		}
	}
	return y;
}	
