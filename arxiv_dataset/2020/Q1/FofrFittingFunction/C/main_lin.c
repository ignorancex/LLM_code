#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double pofk_enhancement_linear(double _z, double _fR0, double _k);

int main(int argc, char **argv){
  if(argc < 3){
    printf("Run as ./test fofr0 redshift\n");
    exit(1);
  }
  double fofr0 = atof(argv[1]);
  double z    = atof(argv[2]);

  int nk = 30;
  double kmin = 0.01;
  double kmax = 10.0;
  printf("# For fR0 = %e and z = %f we have the following (k, ratio(k)): \n",fofr0,z);
  for(int i = 0; i < nk; i++){
    double k = exp(log(kmin) + log(kmax/kmin)*i/(double)(nk-1));
    printf("%lf    %lf\n", k, pofk_enhancement_linear(z, fofr0, k));
  }
}




