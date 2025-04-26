/*----------------------------------------------------------------------------*/
/*--------------------------------  UTILITIES  -------------------------------*/
/*----------------------------------------------------------------------------*/

#include "quartic.h"

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
    *start = q*job + r;
    *stop = q*(job+1) + r;
  }
}

/*------------------------------------------------------------------------------*/

void loop_printer(uint64_t loop_idx, uint64_t start, uint64_t stop,
		  uint64_t steps_until_print, int *num_prints, time_t *loop_timer)
{
  if (loop_idx % steps_until_print == 0)
  {
    fprintf(stdout,"%5.2f%% ",(loop_idx-start)*100.0/(stop-start));
    fflush(stdout);
    (*num_prints)++;
    if (*num_prints==10)
    {
      char time_str[128];
      *num_prints = 0;
      format_time(time_str,time(NULL)-*loop_timer);
      fprintf(stdout,"- Time: %s\n",time_str);
      *loop_timer = time(NULL);
    }
  }
}
  
/*----------------------------------------------------------------------------*/

void start_vector(uint16_t *vv, int len_vv, int q, int start_idx)
{
  int i;
  for (i=0; i<len_vv; i++) vv[i]=0;
  i = len_vv-1;
  while (start_idx > 0)
  {
    vv[i] = start_idx % q;
    start_idx = (start_idx-vv[i])/q;
    i--;
  }
}

/*----------------------------------------------------------------------------*/

int next_vector(uint16_t *vv, int len_vv, int q)
{
  // Find right-most entry that is < q-1
  int i = len_vv-1;
  while ((*(vv+i) == q-1) && i>=0) i--;
  if (i==-1) return 0;
  (*(vv+i))++;
  memset(vv+i+1,0,(len_vv-i-1)*sizeof(uint16_t)); // Zero out i+1, ..., len_vv - 1 
  // int j; for (j=i+1; j<len_vv; j++) *(vv+j) = 0;
  return 1;
}

/*----------------------------------------------------------------------------*/

int next_projective_vector(uint16_t *vv, int len_vv, int q)
{
  int j = len_vv-1;
  while (*(vv+j) == q-1 && j>=0) j--;
  // The j-th entry is the rightmost entry not equal to q-1.  
  if (j==-1) return 0; // vv = (1,q-1,...,q-1)
  
  uint16_t i = 0;
  while (*(vv+i) == 0) i++;
  // The i-th entry is the first nonzero, which is equal to 1 by hypothesis.
  if (i < j)
  {
    // vv = (0,...,0,1,*,...,*,q-1,...,q-1)
    (*(vv+j))++;
    memset(vv+j+1,0,(len_vv-j-1)*sizeof(uint16_t)); // Zero out j+1, ..., len_vv - 1 
    // for (i=j+1; i<len_vv; i++) *(vv+i) = 0;
    return 1;
  }
  // Now vv = (0,...,0,1,q-1,...,q-1); note i=j if q-1 > 1 and i=j+1 if q-1=1
  if (i==0) return 0;
  *(vv+i-1) = 1;
  memset(vv+i,0,(len_vv-i)*sizeof(int16_t));  // Zero out i, ..., len_vv - 1 
  // for (j=i; j<len_vv; j++) *(vv+j) = 0;
  return 1;
}

/*----------------------------------------------------------------------------*/

void test_next_projective_vector(int m, int q)
{
  uint16_t ww[m];
  memset(ww,0,m*sizeof(int16_t));
  ww[m-1] = 1;
  uint16_t i;
  while (1)
  {
    fprintf(stdout,"(%d",ww[0]);
    for (i=1; i<m; i++) fprintf(stdout,", %d",ww[i]);
    fprintf(stdout,")\n");    
    if (!next_projective_vector(ww,m,q)) break;
  }
}

/*----------------------------------------------------------------------------*/

void qset_initialize(qset_t *S, int q)
{
  size_t num_entries = (uint64_t) 1 << SETBITS;
  int64_t *ptr = (int64_t*) malloc(num_entries*sizeof(int64_t));
  if (ptr == NULL) MEMORY_ERROR;
  memset(ptr,0xFF,8*num_entries); // Set all entries to -1
  S->members = ptr;
  S->num_members = num_entries;
  S->mask = ((int64_t) 1 << SETBITS) - 1;
  S->q = q;
}

/*----------------------------------------------------------------------------*/

// Private function: Return the integer given by the base-q representation of
// the quartic Q in our preferred basis, dropping the first entry of Q since
// it's always 1. The values will be at most q^10 <= 2^50 if q <= 32.
static int64_t quartic_key(uint16_t *Q, int q)
{
  int i;
  uint64_t rop = (uint64_t) Q[1];
  uint64_t pow = q;
  for (i=2; i<11; i++)
  {
    rop = rop + pow * (uint64_t)Q[i];
    pow *= q;
  }
  return rop;
}

/*----------------------------------------------------------------------------*/

void qset_add_item(qset_t *S, uint16_t *Q)
{
  int64_t key = quartic_key(Q,S->q);
  int64_t idx = key & (S->mask); // Reduce mod 2^SETBITS
  size_t i;
  for (i=idx; i<S->num_members; i++)
  {
    if (S->members[i] == key) return; // Q already in the set
    if (S->members[i] == -1)
    {
      S->members[i] = key; // Add Q to the set
      return;
    }
  }
  // Loop around to start of the array
  for (i=0; i<idx; i++)
  {
    if (S->members[i] == key) return; // Q already in the set
    if (S->members[i] == -1)
    {
      S->members[i] = key; // Add Q to the set
      return;
    }
  }
  // At this point, there's no space left in the set
  fprintf(stderr,"ERROR: qset structure is full, cannot place quartic with key %lld"
	  " and index %lld\n",key,idx);
  exit(1);
}

/*----------------------------------------------------------------------------*/

int qset_is_member(qset_t *S, uint16_t *Q)
{
  int64_t key = quartic_key(Q,S->q);
  int64_t idx = key & (S->mask); // Reduce mod 2^SETBITS
  size_t i;
  for (i=idx; i<S->num_members; i++)
  {
    if (S->members[i] == -1) return 0;  // Q not in the set
    if (S->members[i] == key) return 1; // Q in the set
  }
  for (i=0; i<idx; i++)
  {
    if (S->members[i] == -1) return 0;  // Q not in the set
    if (S->members[i] == key) return 1; // Q in the set
  }
  return 0; // Looked at the whole array and didn't find Q
}
