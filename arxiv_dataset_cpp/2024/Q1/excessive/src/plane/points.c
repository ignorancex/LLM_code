/*----------------------------------------------------------------------------*/
/*---------------------------------  POINTS  ---------------------------------*/
/*----------------------------------------------------------------------------*/

#include "plane.h"
#include <arith.h>

/*----------------------------------------------------------------------------*/

void point_init(point_t *P, fq_ctx_t FF)
{
  fq_init(P->x,FF);
  fq_init(P->y,FF);
  fq_init(P->z,FF);
}

void point_clear(point_t *P, fq_ctx_t FF)
{
  fq_clear(P->x,FF);
  fq_clear(P->y,FF);
  fq_clear(P->z,FF);
}

/*----------------------------------------------------------------------------*/

void point_set(point_t *P, point_t *Q, fq_ctx_t FF)
{
  fq_set(P->x,Q->x,FF);
  fq_set(P->y,Q->y,FF);
  fq_set(P->z,Q->z,FF);
}

/*----------------------------------------------------------------------------*/

void point_pretty_print(point_t *P, fq_ctx_t FF)
{
  fprintf(stdout,"(%s, %s, %s)",  
	  fq_get_str_pretty(P->x,FF),
	  fq_get_str_pretty(P->y,FF),
	  fq_get_str_pretty(P->z,FF));
}

/*----------------------------------------------------------------------------*/

void point_normalize(point_t *P, fq_ctx_t FF)
{
  if(!fq_is_zero(P->x,FF))
  {
    if(fq_is_one(P->x,FF)) return;
    fq_div(P->y,P->y,P->x,FF);
    fq_div(P->z,P->z,P->x,FF);
    fq_one(P->x,FF);
    return;
  }
  if(!fq_is_zero(P->y,FF))
  {
    if(fq_is_one(P->y,FF)) return;
    fq_div(P->z,P->z,P->y,FF);
    fq_one(P->y,FF);
    return;
  }
  // Now P = (0,0,*)
  if(fq_is_one(P->z,FF)) return;
  fq_one(P->z,FF);
  return;
}

/*----------------------------------------------------------------------------*/

int point_eq_raw(const point_t *P, const point_t *Q, fq_ctx_t FF)
{
  if (!fq_equal(P->x,Q->x,FF)) return 0;
  if (!fq_equal(P->y,Q->y,FF)) return 0;
  if (!fq_equal(P->z,Q->z,FF)) return 0;
  return 1;
}

/*----------------------------------------------------------------------------*/

int point_eq(point_t *P, point_t *Q, fq_ctx_t FF)
{
  point_normalize(P,FF);
  point_normalize(Q,FF);
  return point_eq_raw(P,Q,FF);
}

/*----------------------------------------------------------------------------*/

void point_frobenius(point_t *Q, const point_t *P, fq_ctx_t FF)
{
  fq_frobenius(Q->x,P->x,1,FF);
  fq_frobenius(Q->y,P->y,1,FF);
  fq_frobenius(Q->z,P->z,1,FF);
}

/*----------------------------------------------------------------------------*/

// Private utility for checking if normalized point P is in the array pts
static int point_in_array(point_t *P, point_t *pts, int len_pts, fq_ctx_t FF)
{
  int i;
  for (i=0; i<len_pts; i++)
  {
    if (point_eq_raw(P,pts+i,FF)) return 1;
  }
  return 0;
}

int galois_orbits(point_t **orbit_reps, fq_ctx_t FF, fq_t *elts, int rat_pts)
{
  int p = fmpz_get_ui(fq_ctx_prime(FF));
  int r = fq_ctx_degree(FF);
  int q = field_cardinality_to_int(FF);
  int i, j;

  // Initialize array to keep track of points of PP^2(FF) encountered
  // while constructing Galois orbits
  int num_pts = 1 + q + q*q;
  point_t seen_pts[num_pts];
  for (i=0; i<num_pts; i++) point_init(seen_pts+i,FF);
  int num_seen = 0;

  // Compute expected number of orbits
  int num_orbits = 0;
  int d, e;
  fmpz_t tmp;
  fmpz_init(tmp);
  for (d=1; d<=r; d++)
  {
    // Count closed points of degree d
    if (r % d != 0) continue;    
    // #{points of degree d} = \sum_{e=1}^d \mu(d/e) \#PP^2(FF_2^{e})
    if (d==1 && !rat_pts) continue;    
    int num_d_points = 0;
    for (e=1; e<=d; e++)
    {
      if (d % e != 0) continue;
      q = 1;
      for (i=1; i<=e; i++) q *= p; // q = p^e
      fmpz_set_ui(tmp,d/e);
      num_d_points += arith_moebius_mu(tmp) * (1 + q + q*q);
    }
    num_orbits += num_d_points/d;
  }  
  *orbit_reps = (point_t *) malloc(num_orbits*sizeof(point_t));
  if (*orbit_reps == NULL) MEMORY_ERROR;
  for (i=0; i<num_orbits; i++) point_init(*orbit_reps+i,FF);
  int orbit_ind = 0;

  // Reset q = p^r
  q = field_cardinality_to_int(FF);  
  
  // Loop over normalized points of PP^2(FF).  Apply Frobenius to each point and
  // store it in seen_pts. Start with points of the form (1,*,*).
  point_t P, Q;
  point_init(&P,FF);
  point_init(&Q,FF);    
  fq_one(P.x,FF);
  for (i=0; i<q; i++)
  {
    fq_set(P.y,elts[i],FF);
    for (j=0; j<q; j++)
    {
      fq_set(P.z,elts[j],FF);
      if (point_in_array(&P,seen_pts,num_seen,FF)) continue;
      // Haven't seen this point. Put it in the list of orbit reps
      // and put its whole orbit in list of seen points.
      if (!rat_pts)
      {
	point_frobenius(&Q,&P,FF);
	if (point_eq_raw(&Q,&P,FF))
	{
	  point_set(seen_pts+num_seen,&P,FF);
	  num_seen++;
	  continue;
	}
      }
      point_set(*orbit_reps+orbit_ind,&P,FF);
      orbit_ind++;
      point_set(&Q,&P,FF);
      for (d=0; d<r; d++)
      {
	point_set(seen_pts+num_seen,&Q,FF);
	num_seen++;
	point_frobenius(&Q,&Q,FF);
	if (point_eq_raw(&P,&Q,FF)) break;
      }
    }
  }

  // Points of the form (0,1,*)
  fq_zero(P.x,FF);
  fq_one(P.y,FF);
  for (i=0; i<q; i++)
  {
    fq_set(P.z,elts[i],FF);
    if (point_in_array(&P,seen_pts,num_seen,FF)) continue;
    // Haven't seen this point. Put it in the list of orbit reps
    // and put its whole orbit in list of seen points.
    if (!rat_pts)
    {
      point_frobenius(&Q,&P,FF);
      if (point_eq_raw(&Q,&P,FF))
      {
	point_set(seen_pts+num_seen,&P,FF);
	num_seen++;
	continue;
      }
    }   
    point_set(*orbit_reps+orbit_ind,&P,FF);
    orbit_ind++;
    point_set(&Q,&P,FF);
    for (d=0; d<r; d++)
    {
      point_set(seen_pts+num_seen,&Q,FF);
      num_seen++;
      point_frobenius(&Q,&Q,FF);
      if (point_eq_raw(&P,&Q,FF)) break;      
    }
  }

  // Add the point (0,0,1) if appropriate
  if (rat_pts)
  {
    fq_zero(P.x,FF);
    fq_zero(P.y,FF);
    fq_one(P.z,FF);
    point_set(*orbit_reps+orbit_ind,&P,FF);
    orbit_ind++;
  }
  
#if DEBUG
  if (num_orbits != orbit_ind)
  {
    fprintf(stderr,"Only found %d orbits ... expected to find %d\n",orbit_ind,num_orbits);
    exit(1);
  }
  fprintf(stdout, "Number of Galois orbits of points = %d\n",num_orbits);
  fprintf(stdout,"\nOrbit representatives:\n");
  for (i=0; i<num_orbits; i++)
  {
    point_pretty_print(*orbit_reps+i,FF);
    fprintf(stdout,"\n");
  }
#endif /* DEBUG */
  
  point_clear(&P,FF);
  point_clear(&Q,FF);  
  fmpz_clear(tmp);
  for (i=0; i<num_pts; i++) point_clear(seen_pts+i,FF);  
  return num_orbits;
}

