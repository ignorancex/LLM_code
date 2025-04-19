#include <stdio.h>
#include <time.h>

void format_time(char *time_str, time_t t)
{
  if (t < 60)
  {
    sprintf(time_str,"%dsec",(int) t);
  }
  else if (t < 3600)
  {
    int min = (long) (t/60);
    sprintf(time_str,"%dmin %dsec",min,(int) (t - 60*min));
  }
  else
  {
    int hrs = (long) (t/3600);
    int min = (long) ((t - 3600*hrs)/60);    
    sprintf(time_str,"%dh %dmin %dsec",hrs,min,(int) (t - 3600*hrs - 60*min));    
  }
}
