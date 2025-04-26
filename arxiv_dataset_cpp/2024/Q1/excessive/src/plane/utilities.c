/*----------------------------------------------------------------------------*/
/*--------------------------------  UTILITIES  -------------------------------*/
/*----------------------------------------------------------------------------*/

#include "plane.h"

/*------------------------------------------------------------------------------*/

void format_time(char *time_str, time_t t)
{
  if (t < 60)
  {
    sprintf(time_str,"%dsec",(int) t);
  }
  else if (t < 3600)
  {
    int min = t/60;
    int sec = t - 60*min;
    sprintf(time_str,"%dmin %dsec",min,sec);
  }
  else
  {
    int hrs = t/3600;
    int min = (t - 3600*hrs)/60;
    int sec = t - 3600*hrs - 60*min;
    sprintf(time_str,"%dh %dmin %dsec",hrs,min,sec);
  }
}

/*------------------------------------------------------------------------------*/

void start_and_stop_work(uint64_t *start, uint64_t *stop,
			 uint64_t total_work, int num_jobs, int job)
{
  if ((job < 0) || (job >= num_jobs))
  {
    fprintf(stderr,"job = %d is invalid; must be in the range [0,%d)\n",job,num_jobs);
    exit(1);
  }
  if (num_jobs < 1)
  {
    fprintf(stderr,"num_jobs = %d is invalid; must be a positive integer\n",num_jobs);
    exit(1);
  }
  uint64_t r = total_work % num_jobs;
  uint64_t q = (total_work-r)/num_jobs;
  if (job < r)
  {
    *start = (q+1)*job;
    *stop = (q+1)*(job+1);
  }
  else
  {
    *start = (q+1)*r + q*(job-r);
    *stop = (q+1)*r + q*(job-r+1);    
  }
}

