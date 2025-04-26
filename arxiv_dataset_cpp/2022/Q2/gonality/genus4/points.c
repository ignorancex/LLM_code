/*----------------------------------------------------------------------------*/
/*      		 Code for working with point_t objects                */
/*----------------------------------------------------------------------------*/

#include "genus4.h"

/*----------------------------------------------------------------------------*/

void point_init(point_t *P, fq_ctx_t F)
{
  fq_init(P->x,F);
  fq_init(P->y,F);
  fq_init(P->z,F);
  fq_init(P->w,F);
}

/*----------------------------------------------------------------------------*/

void point_clear(point_t *P, fq_ctx_t F)
{
  fq_clear(P->x,F);
  fq_clear(P->y,F);
  fq_clear(P->z,F);
  fq_clear(P->w,F);
}

/*----------------------------------------------------------------------------*/

void point_set(point_t *P, point_t *Q, fq_ctx_t F)
{
  fq_set(P->x,Q->x,F);
  fq_set(P->y,Q->y,F);
  fq_set(P->z,Q->z,F);
  fq_set(P->w,Q->w,F);
}

/*----------------------------------------------------------------------------*/

void point_pretty_print(point_t *P, fq_ctx_t F)
{
  fprintf(stdout,"(%s, %s, %s, %s)",  
	  fq_get_str_pretty(P->x,F),
	  fq_get_str_pretty(P->y,F),
	  fq_get_str_pretty(P->z,F),
	  fq_get_str_pretty(P->w,F));
}

/*----------------------------------------------------------------------------*/

void point_normalize(point_t *P, fq_ctx_t F)
{
  if(!fq_is_zero(P->x,F))
  {
    if(fq_is_one(P->x,F)) return;
    fq_div(P->y,P->y,P->x,F);
    fq_div(P->z,P->z,P->x,F);
    fq_div(P->w,P->w,P->x,F);
    fq_one(P->x,F);
    return;
  }
  if(!fq_is_zero(P->y,F))
  {
    if(fq_is_one(P->y,F)) return;
    fq_div(P->z,P->z,P->y,F);
    fq_div(P->w,P->w,P->y,F);
    fq_one(P->y,F);
    return;
  }
  if(!fq_is_zero(P->z,F))
  {
    if(fq_is_one(P->z,F)) return;
    fq_div(P->w,P->w,P->z,F);
    fq_one(P->z,F);
    return;
  }
  // Now P = (0,0,0,*)
  if(fq_is_one(P->w,F)) return;
  fq_one(P->w,F);
  return;
}

/*----------------------------------------------------------------------------*/

int point_eq_raw(const point_t *P, const point_t *Q, fq_ctx_t F)
{
  if (!fq_equal(P->x,Q->x,F)) return 0;
  if (!fq_equal(P->y,Q->y,F)) return 0;
  if (!fq_equal(P->z,Q->z,F)) return 0;
  if (!fq_equal(P->w,Q->w,F)) return 0;
  return 1;
}

/*----------------------------------------------------------------------------*/

int point_eq(point_t *P, point_t *Q, fq_ctx_t F)
{
  point_normalize(P,F);
  point_normalize(Q,F);
  return point_eq_raw(P,Q,F);
}

/*----------------------------------------------------------------------------*/

void point_frobenius(point_t *Q, const point_t *P, long d, fq_ctx_t F)
{
  fq_frobenius(Q->x,P->x,d,F);
  fq_frobenius(Q->y,P->y,d,F);
  fq_frobenius(Q->z,P->z,d,F);
  fq_frobenius(Q->w,P->w,d,F);
}

/*----------------------------------------------------------------------------*/

// Set Q = f(P), where f is the q-Frobenius map on F = GF(q^2)
void point_quadratic_frobenius(point_t *Q, const point_t *P, fq_ctx_t GFq_squared)
{
  long p = fmpz_get_ui(fq_ctx_prime(GFq_squared));
  long degree = fq_ctx_degree(GFq_squared);
  if (degree == 2)
  {
    point_frobenius(Q,P,1,GFq_squared);
  }
  else if (degree == 4)
  {
    point_frobenius(Q,P,2,GFq_squared);
  }
  else
  {
    fprintf(stderr,"ERROR: Cannot handle case p = %ld, degree = %ld\n",p,degree);
    exit(1);
  }
}

/*----------------------------------------------------------------------------*/

int next_vector(int8_t *vv, int8_t len_vv, int8_t bound)
{
  // Find right-most entry that is < bound
  int8_t i = len_vv-1;
  while (*(vv+i) == bound) i--;
  if (i==-1) return 0;
  (*(vv+i))++;
  int8_t j;
  for (j=i+1; j<len_vv; j++) *(vv+j) = 0;
  return 1;
}

/*----------------------------------------------------------------------------*/

int next_projective_vector(int8_t *vv, int8_t len_vv, int8_t bound)
{
  int8_t j = len_vv-1;
  while (*(vv+j) == bound) j--;
  // The j-th entry is the rightmost entry not equal to bound.  
  if (j==-1) return 0; // vv = (1,bound,...,bound)
  
  int8_t i = 0;
  while (*(vv+i) == 0) i++;
  // The i-th entry is the first nonzero, which is equal to 1 by hypothesis.
  if (i < j)
  {
    // vv = (0,...,0,1,*,...,*,bound,...,bound)
    (*(vv+j))++;
    memset(vv+j+1,0,(len_vv-j-1)*sizeof(int8_t)); // Zero out j+1, ..., len_vv - 1 
    // for (i=j+1; i<len_vv; i++) *(vv+i) = 0;
    return 1;
  }
  // Now vv = (0,...,0,1,bound,...,bound); note i=j if bound > 1 and i=j+1 if bound=1
  if (i==0) return 0;
  *(vv+i-1) = 1;
  memset(vv+i,0,(len_vv-i)*sizeof(int8_t));  // Zero out i, ..., len_vv - 1 
  // for (j=i; j<len_vv; j++) *(vv+j) = 0;
  return 1;
}

/*----------------------------------------------------------------------------*/

void test_next_projective_vector(int8_t m, int8_t n)
{
  int8_t i;
  int8_t ww[m];
  for (i=0; i<m-1; i++) ww[i] = 0;
  ww[m-1] = 1;
  while (1)
  {
    fprintf(stdout,"(%d",ww[0]);
    for (i=1; i<m; i++) fprintf(stdout,", %d",ww[i]);
    fprintf(stdout,")\n");    
    if (!next_projective_vector(ww,m,n)) break;
  }
}
    
