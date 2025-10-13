#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "SSm0.h"

int main (int argc, char **argv)
{
  double beta, cosT;

  /* Commandline argumets */
  if (argc == 3)
    {
      beta = atof(argv[1]);
      cosT = atof(argv[2]);
    }
  /* Random values */
  else
    {
      time_t t;
      srand((unsigned) time(&t));
      /* beta       = [0,1] */
      beta = rand() / ((double)RAND_MAX);
      /* cos(theta) = [-1,1] */
      cosT = (rand() / ((double)RAND_MAX))*2 - 1.0;
    }

  printf("beta       = %12.8lf\ncos(theta) = %12.8lf\n", beta, cosT);
  printf("                       \\tilde{I}(quarks)                 \\tilde{S}(gluons)\n");
  for (int i = -3; i <= 1; i++)
    printf("ep^%2d                  %16.10lf                  %16.10lf\n", i,
           SSm0_tldI(i, beta, cosT), SSm0_tldS(i, beta, cosT));

  return 0;
}
