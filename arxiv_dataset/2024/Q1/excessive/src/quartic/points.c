/*----------------------------------------------------------------------------*/
/*---------------------------------  POINTS  ---------------------------------*/
/*----------------------------------------------------------------------------*/

#include "quartic.h"
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

void point_pretty_print(point_t *P, fq_ctx_t FF)
{
  fprintf(stdout,"(%s, %s, %s)",  
	  fq_get_str_pretty(P->x,FF),
	  fq_get_str_pretty(P->y,FF),
	  fq_get_str_pretty(P->z,FF));
}

/*----------------------------------------------------------------------------*/

int point_eq_raw(const point_t *P, const point_t *Q, fq_ctx_t F)
{
  if (!fq_equal(P->x,Q->x,F)) return 0;
  if (!fq_equal(P->y,Q->y,F)) return 0;
  if (!fq_equal(P->z,Q->z,F)) return 0;
  return 1;
}

/*----------------------------------------------------------------------------*/

// Private function: set Q = f(P), where f is the p^d-Frobenius map on F = GF(p^*)
static void point_frobenius(point_t *Q, const point_t *P, long d, fq_ctx_t F)
{
  fq_frobenius(Q->x,P->x,d,F);
  fq_frobenius(Q->y,P->y,d,F);
  fq_frobenius(Q->z,P->z,d,F);
}

// Set Q = f(P), where f is the q-Frobenius map on FF = GF(q^2)
void point_quadratic_frobenius(point_t *Q, const point_t *P, fq_ctx_t FF)
{
  long d = fq_ctx_degree(FF);  // q^2 = p^d, so d must be even.
  long p = fmpz_get_ui(fq_ctx_prime(FF));  
  if (d % 2 != 0)
  {
    fprintf(stderr,"ERROR: FF has odd degree %ld over GF(%ld)\n",d,p);
    exit(1);
  }
  point_frobenius(Q,P,d/2,FF);
}

/*----------------------------------------------------------------------------*/

// Construct four arrays of rational points for the different phases of testing
// xpts are the   q points on the line x=0 not equal to (0:0:1)
// ypts are the   q points on the line y=0 not equal to (0:0:1)
// zpts are the q-1 points on the line z=0 not equal to (1:0:0) or (0:1:0)
// genpts are the remaining (q-1)^2 points not on a coordinate line
void construct_rational_point_arrays(point_t **xpts, point_t **ypts, point_t **zpts,
				     point_t **genpts, fq_t *elts,
				     fq_ctx_t GFq_squared)
{
  int q = get_q(GFq_squared);
  int i, j, num_pts;
  point_t *P;

  // Build xpts -- (0:1:*)
  num_pts = q;
  P = (point_t *) malloc(num_pts*sizeof(point_t));
  if (P==NULL) MEMORY_ERROR;
  *xpts = P;
  for (i=0; i<num_pts; i++) point_init(P+i,GFq_squared);
  for (i=0; i<num_pts; i++)
  {
    fq_zero((P+i)->x,GFq_squared);
    fq_one((P+i)->y,GFq_squared);
    fq_set((P+i)->z,elts[i],GFq_squared);
  }

  // Build ypts -- (1:0:*)
  num_pts = q;
  P = (point_t *) malloc(num_pts*sizeof(point_t));
  if (P==NULL) MEMORY_ERROR;
  *ypts = P;
  for (i=0; i<num_pts; i++) point_init(P+i,GFq_squared);
  for (i=0; i<num_pts; i++)
  {
    fq_one((P+i)->x,GFq_squared);
    fq_zero((P+i)->y,GFq_squared);
    fq_set((P+i)->z,elts[i],GFq_squared);
  }

  // Build zpts -- (*:1:0), skipping (0:1:0)
  num_pts = q-1;
  P = (point_t *) malloc(num_pts*sizeof(point_t));
  if (P==NULL) MEMORY_ERROR;
  *zpts = P;
  for (i=0; i<num_pts; i++) point_init(P+i,GFq_squared);
  for (i=0; i<num_pts; i++) 
  {
    fq_set((P+i)->x,elts[i+1],GFq_squared); // i+1 to skip over element 0
    fq_one((P+i)->y,GFq_squared);
    fq_zero((P+i)->z,GFq_squared);
  }

  // Build genpts
  num_pts = (q-1)*(q-1);
  P = (point_t *) malloc(num_pts*sizeof(point_t));
  if (P==NULL) MEMORY_ERROR;
  *genpts = P;
  int idx = 0;
  for (i=0; i<num_pts; i++) point_init(P+i,GFq_squared);
  for (i=1; i<q; i++)
  {
    for (j=1; j<q; j++)
    {
      fq_set((P+idx)->x,elts[i],GFq_squared); 
      fq_set((P+idx)->y,elts[j],GFq_squared);
      fq_one((P+idx)->z,GFq_squared);
      idx++;
    }
  }
}

/*----------------------------------------------------------------------------*/

// Private function: return 1 if pt is in the list pts, and 0 otherwise. 
static int point_in_list(point_t *pt, point_t *pts, int list_len, fq_ctx_t FF)
{
  int i;
  for (i=0; i<list_len; i++)
  {
    if (point_eq_raw(pt,pts+i,FF)) return 1;
  }
  return 0;
}

void construct_quadratic_points(point_t **pts, fq_t *elts, fq_ctx_t GFq_squared)
{
  int q = get_q(GFq_squared);
  int num_places = ((q*q*q*q)-q) / 2; // q^4 - q non-rational quadratic points
  point_t *P = (point_t *) malloc(num_places*sizeof(point_t));
  if (P==NULL) MEMORY_ERROR;
  *pts = P;
  
  int i;
  for (i=0; i<num_places; i++) point_init(P+i,GFq_squared);

  // Temp point for testing if Frobenius is already on the list
  point_t Q;
  point_init(&Q,GFq_squared);

  int j, idx = 0;
  // set P_{idx} = (1, elts_i, elts_j)
  for (i=0; i<q*q; i++)
  {
    for (j=0; j<q*q; j++)
    {
      fq_one((P+idx)->x,GFq_squared);
      fq_set((P+idx)->y,elts[i],GFq_squared);
      fq_set((P+idx)->z,elts[j],GFq_squared);
      point_quadratic_frobenius(&Q,P+idx,GFq_squared);
      if (point_eq_raw(&Q,P+idx,GFq_squared)) continue; // Point is rational
      if (point_in_list(&Q,P,idx,GFq_squared)) continue; // Seen place already
      idx++;
    }
  }

  // // set P_{idx} = (0, 1, elts_i)
  for (i=0; i<q*q; i++) 
  {
    fq_zero((P+idx)->x,GFq_squared);
    fq_one((P+idx)->y,GFq_squared);
    fq_set((P+idx)->z,elts[i],GFq_squared);
    point_quadratic_frobenius(&Q,P+idx,GFq_squared);
      if (point_eq_raw(&Q,P+idx,GFq_squared)) continue; // Point is rational    
    if (point_in_list(&Q,P,idx,GFq_squared)) continue;
    idx++;
    if (idx == num_places) break;
  }

  if (idx != num_places)
  {
    fprintf(stderr,"ERROR: Got %d  quadratic places, expected %d\n",idx,num_places);
    exit(1);
  }

  point_clear(&Q,GFq_squared);
}

/*----------------------------------------------------------------------------*/

void quad_point_indices(fq_idx_vec_t **point_inds, point_t *quad_pts,
			fq_t *fld_elts, fq_ctx_t GFq_squared)
{
  int q = get_q(GFq_squared);
  int num_plane_quads = (q*q*q*q-q)/2;
  *point_inds = (fq_idx_vec_t *) malloc(num_plane_quads*sizeof(fq_idx_vec_t));
  if (*point_inds==NULL) MEMORY_ERROR;
  fq_idx_vec_t *P = *point_inds;
  
  int i;
  for (i=0; i<num_plane_quads; i++)
  {
    fq_idx_vec_init(P+i,3);
    P[i][0] = field_element_to_index((quad_pts+i)->x,fld_elts,GFq_squared);
    P[i][1] = field_element_to_index((quad_pts+i)->y,fld_elts,GFq_squared);
    P[i][2] = field_element_to_index((quad_pts+i)->z,fld_elts,GFq_squared);    
  }
}

/*----------------------------------------------------------------------------*/

int quad_places_in_general_position(fq_idx_vec_t P, fq_idx_vec_t Q,
				    ops_store_t *tables)
{
  // If P = (a,b,c) and Q = (x,y,z), then the cross product
  // (bz - cy, cx - az, ay-bx) is orthogonal to both.

  uint16_t a, b, c, x, y, z, s, t, u, tmp;
  int q = tables->q;

  a = P[0];
  b = P[1];
  c = P[2];
  x = Q[0];
  y = Q[1];
  z = Q[2];
  
  s = fq_idx_mul(&b,&z,tables);
  tmp = fq_idx_mul(&c,&y,tables);
  s = fq_idx_sub(&s,&tmp,tables);

  t = fq_idx_mul(&c,&x,tables);
  tmp = fq_idx_mul(&a,&z,tables);
  t = fq_idx_sub(&t,&tmp,tables);

  u = fq_idx_mul(&a,&y,tables);
  tmp = fq_idx_mul(&b,&x,tables);
  u = fq_idx_sub(&u,&tmp,tables);

  if (s != 0)
  {
    // Normalize to (1, *, *)
    tmp = fq_idx_inv(&s,tables);
    t = fq_idx_mul(&t,&tmp,tables);
    if (t >= q) return 1;
    u = fq_idx_mul(&u,&tmp,tables);
    if (u >= q) return 1;
    return 0;
  }

  if (t != 0)
  {
    // Normalize to (0,1,*)
    tmp = fq_idx_inv(&t,tables);
    u = fq_idx_mul(&u,&tmp,tables);
    if (u >= q) return 1;
    return 0;
  }

  // Now the line normalizes to (0,0,1)
  return 0; 
}
