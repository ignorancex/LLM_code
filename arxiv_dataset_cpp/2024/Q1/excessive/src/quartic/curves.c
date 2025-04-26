/*----------------------------------------------------------------------------*/
/*		      Genus 3 Excessive Curve Search Code                     */
/*----------------------------------------------------------------------------*/

#include "quartic.h"

/*----------------------------------------------------------------------------*/

void quartic_basis_vec_init(quartic_basis_vec_t *vec, fq_ctx_t FF)
{
  *vec = _fq_vec_init(11,FF);
}

void quartic_basis_vec_clear(quartic_basis_vec_t *vec, fq_ctx_t FF)
{
  _fq_vec_clear(*vec,11,FF);
}

/*----------------------------------------------------------------------------*/

void quartic_basis_vec_pretty_print(quartic_basis_vec_t vv, fq_ctx_t FF)
{
  
  fprintf(stdout,"(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
	  fq_get_str_pretty(vv+0,FF),
	  fq_get_str_pretty(vv+1,FF),
	  fq_get_str_pretty(vv+2,FF),
	  fq_get_str_pretty(vv+3,FF),
	  fq_get_str_pretty(vv+4,FF),
	  fq_get_str_pretty(vv+5,FF),
	  fq_get_str_pretty(vv+6,FF),
	  fq_get_str_pretty(vv+7,FF),
	  fq_get_str_pretty(vv+8,FF),
	  fq_get_str_pretty(vv+9,FF),	  
	  fq_get_str_pretty(vv+10,FF));
}

/*----------------------------------------------------------------------------*/

void quartic_basis_vec_pretty_print_file(FILE *fp, quartic_basis_vec_t vv, fq_ctx_t FF)
{
  
  fprintf(fp,"(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
	  fq_get_str_pretty(vv+0,FF),
	  fq_get_str_pretty(vv+1,FF),
	  fq_get_str_pretty(vv+2,FF),
	  fq_get_str_pretty(vv+3,FF),
	  fq_get_str_pretty(vv+4,FF),
	  fq_get_str_pretty(vv+5,FF),
	  fq_get_str_pretty(vv+6,FF),
	  fq_get_str_pretty(vv+7,FF),
	  fq_get_str_pretty(vv+8,FF),
	  fq_get_str_pretty(vv+9,FF),	  
	  fq_get_str_pretty(vv+10,FF));
}

/*----------------------------------------------------------------------------*/

// private function: translate a quartic_basis_vec_t object into a vector of
// indices, given by the ordering in fld_elts. This is not a particularly fast
// implementation, so it shouldn't be used in the inner loop.
static void quartic_basis_vec_to_index_vec(uint16_t *ind_vec,
					   quartic_basis_vec_t cc_vec,
					   fq_t *fld_elts, fq_ctx_t FF)
{
  int i;
  for (i=0; i<11; i++)
  {
    *(ind_vec+i) = field_element_to_index(cc_vec+i,fld_elts,FF);
  }
}

/*----------------------------------------------------------------------------*/

// Private function to evaluate each basis vector at point P and store in
// vv. The element a is expected to be chosen so that T^2 + aT + 1 is irreducible.
static void quartic_basis_vec_eval(quartic_basis_vec_t vv, point_t *P, fq_t a,
				    fq_ctx_t FF)
{
  fq_struct *x, *y, *z, *w;  
  x = P->x;
  y = P->y;
  z = P->z;

  // temp variables
  fq_t s, t;
  fq_init(s,FF);
  fq_init(t,FF);

  // z^4 + x^2*z^2 + a*x*z^3 + y^2*z^2 + a*y*z^3
  w = vv;
  fq_mul(t,z,z,FF);  // t = z^2
  fq_set(w,t,FF);    // w = z^2
  fq_mul(s,x,x,FF);  // s = x^2
  fq_add(w,w,s,FF);  // w = z^2 + x^2
  fq_add(s,x,y,FF);  // s = x+y
  fq_mul(s,a,s,FF);  // s = a*(x+y)
  fq_mul(s,s,z,FF);  // s = a*x*z + a*y*z
  fq_add(w,w,s,FF);  // w = z^2 + x^2 + a*x*z + a*y*z
  fq_mul(s,y,y,FF);  // s = y^2
  fq_add(w,w,s,FF);  // w = z^2 + x^2 + a*x*z + y^2 + a*y*z
  fq_mul(w,w,t,FF);  // w = z^4 + x^2*z^2 + a*x*z^3 + y^2*z^2 + a*y*z^3

  // x^4 + a*x^3*z + x^2*z^2
  w = vv+1;
  fq_mul(t,x,x,FF);  // t = x^2
  fq_set(w,t,FF);    // w = x^2
  fq_mul(s,a,x,FF);  // s = a*x
  fq_mul(s,s,z,FF);  // s = a*x*z
  fq_add(w,w,s,FF);  // w = x^2 + a*x*z
  fq_mul(s,z,z,FF);  // s = z^2
  fq_add(w,w,s,FF);  // w = x^2 + a*x*z + z^2
  fq_mul(w,w,t,FF);  // w = x^2(x^2 + a*x*z + z^2)
  
  // x^3*z + a*x^2*z^2 + x*z^3
  w = vv+2;
  fq_mul(t,x,z,FF);  // t = x*z
  fq_mul(w,x,x,FF);  // w = x^2
  fq_mul(s,a,x,FF);  // s = a*x
  fq_mul(s,s,z,FF);  // s = a*x*z
  fq_add(w,w,s,FF);  // w = x^2 + a*x*z
  fq_mul(s,z,z,FF);  // s = z^2
  fq_add(w,w,s,FF);  // w = x^2 + a*x*z + z^2
  fq_mul(w,w,t,FF);  // w = x*z*(x^2 + a*x*z + z^2)

  // y^4 + a*y^3*z + y^2*z^2
  w = vv+3;
  fq_mul(t,y,y,FF);  // t = y^2
  fq_set(w,t,FF);    // w = y^2
  fq_mul(s,a,y,FF);  // s = a*y
  fq_mul(s,s,z,FF);  // s = a*y*z
  fq_add(w,w,s,FF);  // w = y^2 + a*y*z
  fq_mul(s,z,z,FF);  // s = z^2
  fq_add(w,w,s,FF);  // w = y^2 + a*y*z + z^2
  fq_mul(w,w,t,FF);  // w = y^2(y^2 + a*y*z + z^2)
  
  // y^3*z + a*y^2*z^2 + y*z^3
  w = vv+4;
  fq_mul(t,y,z,FF);  // t = y*z
  fq_mul(w,y,y,FF);  // w = y^2
  fq_mul(s,a,y,FF);  // s = a*y
  fq_mul(s,s,z,FF);  // s = a*y*z
  fq_add(w,w,s,FF);  // w = y^2 + a*y*z
  fq_mul(s,z,z,FF);  // s = z^2
  fq_add(w,w,s,FF);  // w = y^2 + a*y*z + z^2
  fq_mul(w,w,t,FF);  // w = y*z*(y^2 + a*y*z + z^2)

  // x^3*y
  w = vv+5;
  fq_pow_ui(w,x,3,FF);
  fq_mul(w,w,y,FF);

  // x^2*y^2
  w = vv+6;
  fq_mul(w,x,y,FF);
  fq_mul(w,w,w,FF);

  // x*y^3
  w = vv+7;
  fq_pow_ui(w,y,3,FF);
  fq_mul(w,x,w,FF);

  // x^2*y*z
  w = vv+8;
  fq_pow_ui(w,x,2,FF);
  fq_mul(w,w,y,FF);
  fq_mul(w,w,z,FF);  

  // x*y^2*z
  w = vv+9;
  fq_pow_ui(w,y,2,FF);
  fq_mul(w,w,x,FF);
  fq_mul(w,w,z,FF);  

  // x*y*z^2  
  w = vv+10;
  fq_pow_ui(w,z,2,FF);
  fq_mul(w,w,x,FF);
  fq_mul(w,w,y,FF);  

  fq_clear(s,FF);
  fq_clear(t,FF);
}

/*----------------------------------------------------------------------------*/

void quartic_basis_pol_evaluations(fq_idx_vec_t **vecs, point_t *pts, int num_pts,
				   fq_t *elts, fq_t a, fq_ctx_t FF)
{
  *vecs = (fq_idx_vec_t*) malloc(num_pts*sizeof(fq_idx_vec_t));
  if (*vecs == NULL) MEMORY_ERROR;
  fq_idx_vec_t *vv = *vecs;
  quartic_basis_vec_t VV;
  quartic_basis_vec_init(&VV,FF);

  // Initialize and populate quartic basis vectors
  int i;
  for (i=0; i<num_pts; i++)
  {
    quartic_basis_vec_eval(VV,pts+i,a,FF);
    fq_idx_vec_init(vv+i,11);
    quartic_basis_vec_to_index_vec(*(vv+i),VV,elts,FF);
  }
  quartic_basis_vec_clear(&VV,FF);
}

/*----------------------------------------------------------------------------*/

// Private function to avoid code repetition
void _write_point_inds(FILE *fp, fq_t *elts, point_t *pts, int num_pts, fq_ctx_t FF)
{
  // points, represented as a space-separated triple of indices i j k  
  uint16_t x,y,z;
  int i;
  for (i=0; i<num_pts; i++)
  {
    x = field_element_to_index(pts[i].x,elts,FF);
    y = field_element_to_index(pts[i].y,elts,FF);
    z = field_element_to_index(pts[i].z,elts,FF);    
    fprintf(fp,"%d %d %d\n",x,y,z);
  }
}

void _write_point_evals(FILE *fp, fq_idx_vec_t *pt_data, int num_pts, fq_ctx_t FF)
{
  // vectors of quartic basis polynomial evaluations for each point,
  // as a space-separated triple of indices
  int i;
  for (i=0; i<num_pts; i++) 
  {
    write_index_vector_to_file(fp,pt_data[i],11);
  }
}

void point_data_test(FILE *fp, fq_t *elts,
		     point_t *xpts, point_t *ypts, point_t *zpts, point_t *genpts,
		     fq_idx_vec_t *xpt_data, fq_idx_vec_t *ypt_data,
		     fq_idx_vec_t *zpt_data, fq_idx_vec_t *genpt_data,
		     point_t *quadpts, fq_idx_vec_t *quadpt_data, 
		     fq_t a, fq_ctx_t GFq_squared)
{
  // Field cardinality
  int q = get_q(GFq_squared);
  fprintf(fp,"%d\n",q);

  // Field elements in order  
  int i;
  for (i=0; i<q*q; i++)
  {
    fprintf(fp,"%s\n",fq_get_str_pretty(elts[i],GFq_squared));
  }

  // Element a such that T^2 + aT + 1 is irreducible
  fprintf(fp,"%s\n",fq_get_str_pretty(a,GFq_squared));

  // Write points to file
  _write_point_inds(fp,elts,xpts,q,GFq_squared);
  _write_point_inds(fp,elts,ypts,q,GFq_squared);
  _write_point_inds(fp,elts,zpts,q-1,GFq_squared);
  _write_point_inds(fp,elts,genpts,(q-1)*(q-1),GFq_squared);  

  // Write quartic basis polynomial point evaluation data
  _write_point_evals(fp,xpt_data,q,GFq_squared);
  _write_point_evals(fp,ypt_data,q,GFq_squared);
  _write_point_evals(fp,zpt_data,q-1,GFq_squared);
  _write_point_evals(fp,genpt_data,(q-1)*(q-1),GFq_squared);

  // Write quadratic places to file
  _write_point_inds(fp,elts,quadpts,(q*q*q*q-q)/2,GFq_squared);
  
  // Write quartic basis polynomial point evaluation data
  _write_point_evals(fp,quadpt_data,(q*q*q*q-q)/2,GFq_squared);
}

/*----------------------------------------------------------------------------*/

int quartic_has_rational_point(uint16_t *Q, fq_idx_vec_t *pt_data, int num_pts,
			       ops_store_t *tables)
{
  int i;
  for (i=0; i<num_pts; i++)
  {
    if (fq_idx_vec_dot(Q,pt_data[i],11,tables)==0) return 1;
  }
  return 0;
}

/*----------------------------------------------------------------------------*/

void quadratic_places_on_quartic(int *Qquad_inds, int *num_places,
				 uint16_t * Q, fq_idx_vec_t *quadpt_data,
				 fq_idx_vec_t *quadpt_inds, ops_store_t *tables)
{
  int i;
  int q = tables->q;
  int num_plane_quads = ((q*q*q*q) - q)/2;
  int cnt = 0;
  for (i=0; i<num_plane_quads; i++)
  {
    if (fq_idx_vec_dot(Q,quadpt_data[i],11,tables) != 0) continue;
    Qquad_inds[cnt] = i; // 
    cnt++;
  }
  *num_places = cnt;
}
