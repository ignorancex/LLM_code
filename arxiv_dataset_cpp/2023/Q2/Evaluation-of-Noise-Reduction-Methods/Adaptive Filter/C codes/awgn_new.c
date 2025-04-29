#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define LRAND_MAX 0x7FFFFFFF

float *AWGN_NEW( float *sig, float reqSNR)
{
    int signal_length = sizeof(sig)/sizeof(float);
	float sum = 0;
	float *y;

    for(int i = 0;i < signal_length; i++)
    {
        sum = sum + sig[i]*sig[i];
    }

    float sigPower = sum/signal_length;
    sigPower = 10 * log10(sigPower);
    float p = sigPower - reqSNR;
    float a = p/10;
    float noisePower = pow(10,a);

    for (int i = 0; i < signal_length; i++)
    {
        double random_num = rand() / RAND_MAX;
        y[i] = sqrt(noisePower)*random_num; 
        printf("%f\n", y[i]);
    }
	return y;
}
