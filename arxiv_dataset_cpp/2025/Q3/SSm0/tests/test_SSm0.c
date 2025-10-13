#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "SSm0.h"

int main (int argc, char **argv)
{

  double sigma = 1.0e-9;

  for (int i = -2; i <= 1; i++)
    {
      char fname[100];
      sprintf(fname, "SijEp%d.dat", i);
      FILE* finp = fopen(fname, "r");

      char data_buffer[1000];
      double x,y,val;
      if (finp == 0)
        {
          fprintf(stderr, "Can not open input file \"%s\"\n", fname);
          exit( EXIT_FAILURE );
        }
      else
        {
          printf( "Input from file \"%s\"\n", fname);
          int n_points_ok = 0;
          int n_points_bad = 0;
          while (!feof(finp))
            {
              if (data_buffer != fgets(data_buffer, 1000, finp))
                break;
              if (sscanf(data_buffer, "%lf\t%lf\t%lf", &x, &y, &val))
                {
                  double res = SSm0_tldS(i, x, y);
                  if (fabs((res-val)/res) > sigma)
                    {
                      printf("CONTROL: x=%12.8f, y=%12.8f, res=%20.12f, refval=%20.12f, err=%1.20f\n", x, y, res, val, fabs(res-val));
                      n_points_bad += 1;
                    }
                  else
                    n_points_ok += 1;
                }
            }

          printf("-----------------------------\n");
          printf("points tested: %10d, with sigma > %.3e: %10d\n", n_points_ok + n_points_bad, sigma, n_points_bad);

        }
    }


    for (int i = -2; i <= 1; i++)
    {
      char fname[100];
      sprintf(fname, "IijEp%d.dat", i);
      FILE* finp = fopen(fname, "r");

      char data_buffer[1000];
      double x,y,val;
      if (finp == 0)
        {
          fprintf(stderr, "Can not open input file \"%s\"\n", fname);
          exit( EXIT_FAILURE );
        }
      else
        {
          printf( "Input from file \"%s\"\n", fname);
          int n_points_ok = 0;
          int n_points_bad = 0;
          while (!feof(finp))
            {
              if (data_buffer != fgets(data_buffer, 1000, finp))
                break;
              if (sscanf(data_buffer, "%lf\t%lf\t%lf", &x, &y, &val))
                {
                  double res = SSm0_tldI(i, x, y);
                  if (fabs((res-val)/res) > sigma)
                    {
                      printf("CONTROL: x=%12.8f, y=%12.8f, res=%20.12f, refval=%20.12f, err=%1.20f\n", x, y, res, val, fabs(res-val));
                      n_points_bad += 1;
                    }
                  else
                    n_points_ok += 1;
                }
            }

          printf("-----------------------------\n");
          printf("points tested: %10d, with sigma > %.3e: %10d\n", n_points_ok + n_points_bad, sigma, n_points_bad);

        }
    }

  return 0;
}
