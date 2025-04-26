/*----------------------------------------------------------------------------*/
/*		      Genus-4 Excessive Curve Search Code                     */
/*----------------------------------------------------------------------------*/

#include "genus4.h"

/*----------------------------------------------------------------------------*/

void cubic_coef_vec_init(cubic_coef_vec_t *vec, fq_ctx_t GFq)
{
  *vec = _fq_vec_init(20,GFq);
}

void cubic_coef_vec_clear(cubic_coef_vec_t *vec, fq_ctx_t GFq)
{
  _fq_vec_clear(*vec,20,GFq);
}

/*----------------------------------------------------------------------------*/

// Print vv to standard out
void cubic_coef_vec_pretty_print(cubic_coef_vec_t vv, fq_ctx_t GFq)
{
  
  fprintf(stdout,"(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
	  fq_get_str_pretty(vv+0,GFq),
	  fq_get_str_pretty(vv+1,GFq),
	  fq_get_str_pretty(vv+2,GFq),
	  fq_get_str_pretty(vv+3,GFq),
	  fq_get_str_pretty(vv+4,GFq),
	  fq_get_str_pretty(vv+5,GFq),
	  fq_get_str_pretty(vv+6,GFq),
	  fq_get_str_pretty(vv+7,GFq),
	  fq_get_str_pretty(vv+8,GFq),
	  fq_get_str_pretty(vv+9,GFq),	  
	  fq_get_str_pretty(vv+10,GFq),
	  fq_get_str_pretty(vv+11,GFq),
	  fq_get_str_pretty(vv+12,GFq),
	  fq_get_str_pretty(vv+13,GFq),
	  fq_get_str_pretty(vv+14,GFq),
	  fq_get_str_pretty(vv+15,GFq),
	  fq_get_str_pretty(vv+16,GFq),
	  fq_get_str_pretty(vv+17,GFq),
	  fq_get_str_pretty(vv+18,GFq),
	  fq_get_str_pretty(vv+19,GFq)); 
}

/*----------------------------------------------------------------------------*/

// Evaluate the initialized cubic monomial vector at point P.
void cubic_coef_vec_eval(cubic_coef_vec_t vv, point_t *P, fq_ctx_t GFq)
{
  fq_struct *x, *y, *z, *w, *val;  
  x = P->x;
  y = P->y;
  z = P->z;
  w = P->w;

  // x^3
  val = vv;
  fq_mul(val,x,x,GFq);
  fq_mul(val,val,x,GFq);

  // x^2*y
  val = vv+1;
  fq_mul(val,x,x,GFq);
  fq_mul(val,val,y,GFq);

  // x^2*z
  val = vv+2;
  fq_mul(val,x,x,GFq);  
  fq_mul(val,val,z,GFq);

  // x^2*w
  val = vv+3;
  fq_mul(val,x,x,GFq);    
  fq_mul(val,val,w,GFq);

  // x*y^2
  val = vv+4;
  fq_mul(val,x,y,GFq);
  fq_mul(val,val,y,GFq);

  // x*y*z
  val = vv+5;
  fq_mul(val,x,y,GFq);
  fq_mul(val,val,z,GFq);

  // x*y*w
  val = vv+6;
  fq_mul(val,x,y,GFq);
  fq_mul(val,val,w,GFq);

  // x*z^2
  val = vv+7;
  fq_mul(val,x,z,GFq);  
  fq_mul(val,val,z,GFq);
  
  // x*z*w
  val = vv+8;
  fq_mul(val,x,z,GFq);
  fq_mul(val,val,w,GFq);
  
  // x*w^2
  val = vv+9;
  fq_mul(val,x,w,GFq);
  fq_mul(val,val,w,GFq);
  
  // y^3
  val = vv+10;
  fq_mul(val,y,y,GFq);
  fq_mul(val,val,y,GFq);
  
  // y^2*z
  val = vv+11;
  fq_mul(val,y,y,GFq);
  fq_mul(val,val,z,GFq);
  
  // y^2*w
  val = vv+12;
  fq_mul(val,y,y,GFq);
  fq_mul(val,val,w,GFq);
  
  // y*z^2
  val = vv+13;
  fq_mul(val,y,z,GFq);
  fq_mul(val,val,z,GFq);
  
  // y*z*w
  val = vv+14;
  fq_mul(val,y,z,GFq);
  fq_mul(val,val,w,GFq);
  
  // y*w^2
  val = vv+15;
  fq_mul(val,y,w,GFq);
  fq_mul(val,val,w,GFq);
  
  // z^3
  val = vv+16;
  fq_mul(val,z,z,GFq);
  fq_mul(val,val,z,GFq);
  
  // z^2*w
  val = vv+17;
  fq_mul(val,z,z,GFq);
  fq_mul(val,val,w,GFq);
  
  // z*w^2
  val = vv+18;
  fq_mul(val,z,w,GFq);
  fq_mul(val,val,w,GFq);
    
  // w^3
  val = vv+19;  
  fq_mul(val,w,w,GFq);
  fq_mul(val,val,w,GFq);
}

/*----------------------------------------------------------------------------*/

// Allocate memory and store the vector of cubic monomial evaluations
// v_P = (m(P) : m \in monomials),
// where P is a point in representing an element of PP^3, and where
// monomials is given by
// [x^3, x^2*y, x^2*z, x^2*w, x*y^2, x*y*z, x*y*w, x*z^2, x*z*w, x*w^2,
//  y^3, y^2*z, y^2*w, y*z^2, y*z*w, y*w^2, z^3, z^2*w, z*w^2, w^3].
// pts -- pointer to an array of points over GFq
// num_pts -- the number of points in pts
// field_elts -- pointer to an array of elements of GFq
void cubic_monomial_evaluations(cubic_coef_vec_t **vecs, point_t *pts, long num_pts,
			       fq_t *field_elts, fq_ctx_t GFq)
{
  *vecs = (cubic_coef_vec_t*) malloc(num_pts*sizeof(cubic_coef_vec_t));
  if (*vecs == NULL) MEMORY_ERROR;
  cubic_coef_vec_t *vv = *vecs;

  // Initialize and populate cubic coefficient vectors
  long i;
  for (i=0; i<num_pts; i++)
  {
    cubic_coef_vec_init(vv+i,GFq);
    cubic_coef_vec_eval(*(vv+i),pts+i,GFq);
    // // DEBUG
    // fprintf(stdout,"point %ld: ",i);
    // cubic_coef_vec_pretty_print(*(vv+i),GFq);
    // fprintf(stdout,"\n");
    // // END DEBUG
  }
}

/*----------------------------------------------------------------------------*/

void print_cubic_to_file_old(fq_struct *cubic, FILE *fp, fq_ctx_t F)
{
  char s[1024];
  sprintf(s,"(%s)*x^3 + (%s)*x^2*y + (%s)*x^2*z + (%s)*x^2*w + (%s)*x*y^2 + (%s)*x*y*z + (%s)*x*y*w + (%s)*x*z^2 + (%s)*x*z*w + (%s)*x*w^2 + (%s)*y^3 + (%s)*y^2*z + (%s)*y^2*w + (%s)*y*z^2 + (%s)*y*z*w +(%s)*y*w^2 + (%s)*z^3 + (%s)*z^2*w + (%s)*z*w^2 + (%s)*w^3\n",
	  fq_get_str_pretty(cubic+0,F),fq_get_str_pretty(cubic+1,F),
	  fq_get_str_pretty(cubic+2,F),fq_get_str_pretty(cubic+3,F),
	  fq_get_str_pretty(cubic+4,F),fq_get_str_pretty(cubic+5,F),
	  fq_get_str_pretty(cubic+6,F),fq_get_str_pretty(cubic+7,F),
	  fq_get_str_pretty(cubic+8,F),fq_get_str_pretty(cubic+9,F),
	  fq_get_str_pretty(cubic+10,F),fq_get_str_pretty(cubic+11,F),
	  fq_get_str_pretty(cubic+12,F),fq_get_str_pretty(cubic+13,F),
	  fq_get_str_pretty(cubic+14,F),fq_get_str_pretty(cubic+15,F),
	  fq_get_str_pretty(cubic+16,F),fq_get_str_pretty(cubic+17,F),
	  fq_get_str_pretty(cubic+18,F),fq_get_str_pretty(cubic+19,F));
  fputs(s,fp);
}

/*----------------------------------------------------------------------------*/

void print_cubic_to_file(fq_struct *cubic, FILE *fp, fq_ctx_t F)
{
  char s[1024];
  char tmp[128];
  sprintf(s,"(%s)*x^3",fq_get_str_pretty(cubic+0,F));
  if (!fq_is_zero(cubic+1,F))
  {
    sprintf(tmp," + (%s)*x^2*y",fq_get_str_pretty(cubic+1,F));
    strcat(s,tmp);
  }
  if (!fq_is_zero(cubic+2,F))
  {
    sprintf(tmp," + (%s)*x^2*z",fq_get_str_pretty(cubic+2,F));
    strcat(s,tmp);
  }
  if (!fq_is_zero(cubic+3,F))
  {
    sprintf(tmp," + (%s)*x^2*w",fq_get_str_pretty(cubic+3,F));
    strcat(s,tmp);
  }
  if (!fq_is_zero(cubic+4,F))
  {
    sprintf(tmp," + (%s)*x*y^2",fq_get_str_pretty(cubic+4,F));
    strcat(s,tmp);
  }
  if (!fq_is_zero(cubic+5,F))
  {
    sprintf(tmp," + (%s)*x*y*z",fq_get_str_pretty(cubic+5,F));
    strcat(s,tmp);
  }
  if (!fq_is_zero(cubic+6,F))
  {
    sprintf(tmp," + (%s)*x*y*w",fq_get_str_pretty(cubic+6,F));
    strcat(s,tmp);
  }
  if (!fq_is_zero(cubic+7,F))
  {
    sprintf(tmp," + (%s)*x*z^2",fq_get_str_pretty(cubic+7,F));
    strcat(s,tmp);
  }
  if (!fq_is_zero(cubic+8,F))
  {
    sprintf(tmp," + (%s)*x*z*w",fq_get_str_pretty(cubic+8,F));
    strcat(s,tmp);
  }
  if (!fq_is_zero(cubic+9,F))
  {
    sprintf(tmp," + (%s)*x*w^2",fq_get_str_pretty(cubic+9,F));
    strcat(s,tmp);
  }
  if (!fq_is_zero(cubic+10,F))
  {
    sprintf(tmp," + (%s)*y^3",fq_get_str_pretty(cubic+10,F));
    strcat(s,tmp);
  }
  if (!fq_is_zero(cubic+11,F))
  {
    sprintf(tmp," + (%s)*y^2*z",fq_get_str_pretty(cubic+11,F));
    strcat(s,tmp);
  }
  if (!fq_is_zero(cubic+12,F))
  {
    sprintf(tmp," + (%s)*y^2*w",fq_get_str_pretty(cubic+12,F));
    strcat(s,tmp);
  }
  if (!fq_is_zero(cubic+13,F))
  {
    sprintf(tmp," + (%s)*y*z^2",fq_get_str_pretty(cubic+13,F));
    strcat(s,tmp);
  }
  if (!fq_is_zero(cubic+14,F))
  {
    sprintf(tmp," + (%s)*y*z*w",fq_get_str_pretty(cubic+14,F));
    strcat(s,tmp);
  }
  if (!fq_is_zero(cubic+15,F))
  {
    sprintf(tmp," + (%s)*y*w^2",fq_get_str_pretty(cubic+15,F));
    strcat(s,tmp);
  }
  if (!fq_is_zero(cubic+16,F))
  {
    sprintf(tmp," + (%s)*z^3",fq_get_str_pretty(cubic+16,F));
    strcat(s,tmp);
  }
  if (!fq_is_zero(cubic+17,F))
  {
    sprintf(tmp," + (%s)*z^2*w",fq_get_str_pretty(cubic+17,F));
    strcat(s,tmp);
  }
  if (!fq_is_zero(cubic+18,F))
  {
    sprintf(tmp," + (%s)*z*w^2",fq_get_str_pretty(cubic+18,F));
    strcat(s,tmp);
  }
  if (!fq_is_zero(cubic+19,F))
  {
    sprintf(tmp," + (%s)*w^3",fq_get_str_pretty(cubic+19,F));
    strcat(s,tmp);
  }
  strcat(s,"\n");
  fputs(s,fp);
}

/*----------------------------------------------------------------------------*/

void print_cubic_to_screen(fq_struct *cubic, fq_ctx_t F)
{
  char s[1024];
  sprintf(s,"(%s)*x^3 + (%s)*x^2*y + (%s)*x^2*z + (%s)*x^2*w + (%s)*x*y^2 + (%s)*x*y*z + (%s)*x*y*w + (%s)*x*z^2 + (%s)*x*z*w + (%s)*x*w^2 + (%s)*y^3 + (%s)*y^2*z + (%s)*y^2*w + (%s)*y*z^2 + (%s)*y*z*w +(%s)*y*w^2 + (%s)*z^3 + (%s)*z^2*w + (%s)*z*w^2 + (%s)*w^3\n",
	  fq_get_str_pretty(cubic+0,F),fq_get_str_pretty(cubic+1,F),
	  fq_get_str_pretty(cubic+2,F),fq_get_str_pretty(cubic+3,F),
	  fq_get_str_pretty(cubic+4,F),fq_get_str_pretty(cubic+5,F),
	  fq_get_str_pretty(cubic+6,F),fq_get_str_pretty(cubic+7,F),
	  fq_get_str_pretty(cubic+8,F),fq_get_str_pretty(cubic+9,F),
	  fq_get_str_pretty(cubic+10,F),fq_get_str_pretty(cubic+11,F),
	  fq_get_str_pretty(cubic+12,F),fq_get_str_pretty(cubic+13,F),
	  fq_get_str_pretty(cubic+14,F),fq_get_str_pretty(cubic+15,F),
	  fq_get_str_pretty(cubic+16,F),fq_get_str_pretty(cubic+17,F),
	  fq_get_str_pretty(cubic+18,F),fq_get_str_pretty(cubic+19,F));
  fprintf(stdout,"%s",s);
}

/*----------------------------------------------------------------------------*/

long nonsplit_quadric_surface_points(long q, point_t **quadric_pts,
				     fq_t *field_elts, fq_ctx_t GFq_squared)
{
  long i, num_pts, pt_ind;
  
  // There are q^2 + 1 rational points on xy + N(z,w) = 0 over GF(q).
  // This surface splits over GF(q)^2, and the number of points on
  // xy + zw = 0 over GF(q) is (q+1)^2. So we get (q^2+1)^2 quadratic points,
  // which break into q^2 + 1 + (q^2)(q^2+1)/2 orbits. 
  num_pts = (q*q + 1) + (q*q)*(q*q+1)/2;
  *quadric_pts = (point_t*) malloc(num_pts*sizeof(point_t));
  point_t* surface_pts = *quadric_pts;
  if (surface_pts == NULL) MEMORY_ERROR;
  for (i=0; i<num_pts; i++) point_init(surface_pts+i,GFq_squared);

  // Define a, the coefficient on the quadric surface that depends on q
  fq_t a;
  fq_init(a,GFq_squared);
  if (q == 2)
  {
    fq_one(a, GFq_squared);
  }
  else if (q == 3)
  {
    fq_zero(a, GFq_squared);
  }
  else if (q == 4)
  {
    fq_t gen;
    fq_init(gen,GFq_squared);
    fq_gen(gen,GFq_squared); // gen = T
    fq_set(a, gen, GFq_squared); // a = T
    fq_mul(a,a,a,GFq_squared); // a = T^2
    fq_add(a,a,gen,GFq_squared); // a = T^2 + T
    fq_clear(gen,GFq_squared);    
  }
  else
  {
    fprintf(stderr,"ERROR: No support for q = %ld\n",q);
    exit(1);
  }
  
  fprintf(stdout,"\nNonsplit smooth quadric surface: xy + z^2 + (%s)zw + w^2\n",fq_get_str_pretty(a,GFq_squared));
  fprintf(stdout,"Searching for %ld Galois orbits of quadratic points on the surface ... ",num_pts);

  
  // Search for points on the surface
  pt_ind = 0;
  fq_t u, v;
  fq_struct *w, *x, *y, *z;
  fq_init(u,GFq_squared);
  fq_init(v,GFq_squared);
  point_t conj, tmp;
  point_init(&conj,GFq_squared);
  point_init(&tmp,GFq_squared);  
  int8_t vec_ind[4];
  for (i=0;i<3;i++) vec_ind[i] = 0;
  vec_ind[3] = 1;
  while(1)
  {
    x = *(field_elts+vec_ind[0]);
    y = *(field_elts+vec_ind[1]);
    z = *(field_elts+vec_ind[2]);
    w = *(field_elts+vec_ind[3]);
    fq_mul(u,x,y,GFq_squared);  // u = xy
    fq_mul(v,z,z,GFq_squared);  // v = z^2
    fq_add(u,u,v,GFq_squared);  // u = xy + z^2
    fq_mul(v,a,z,GFq_squared);  // v = az
    fq_mul(v,v,w,GFq_squared);  // v = azw
    fq_add(u,u,v,GFq_squared);  // u = xy + z^2 + azw
    fq_mul(v,w,w,GFq_squared);  // v = w^2
    fq_add(u,u,v,GFq_squared);  // u = xy + z^2 + azw + w^2
    if (fq_is_zero(u,GFq_squared))
    {
      // Check if we've seen the conjugate point already
      fq_set(tmp.x,x,GFq_squared);
      fq_set(tmp.y,y,GFq_squared);
      fq_set(tmp.z,z,GFq_squared);
      fq_set(tmp.w,w,GFq_squared);    
      point_quadratic_frobenius(&conj,&tmp,GFq_squared);
      int seen_it = 0;
      for (i=0; i<pt_ind; i++)
      {
	if (point_eq_raw(surface_pts+i,&conj,GFq_squared))
	{
	  seen_it = 1;
	  break;
	}
      }
      if (!seen_it)
      {
	// We haven't seen this point before. Copy it into our list and up the index.
	point_set(surface_pts+pt_ind,&tmp,GFq_squared);
	pt_ind++;
      }
    }
    if (!next_projective_vector(vec_ind,4,q*q-1)) break;
  }
  
  // // DEBUG: Print the pointsnn
  // for (i=0; i<num_pts; i++)
  // {
  //   fprintf(stdout,"point %ld: ",i);
  //   point_pretty_print(surface_pts+i,GFq_squared);
  //   fprintf(stdout,"\n");
  // }
  // // END DEBUG

  // DEBUG: Write points to file for testing in Sage
  // FILE *fp = fopen("pts_test","w");
  // char str[128];
  // for (i=0; i<num_pts; i++)
  // {
  //   sprintf(str,"%s, %s, %s, %s\n",
  // 	    fq_get_str_pretty((surface_pts+i)->x,GFq_squared),
  // 	    fq_get_str_pretty((surface_pts+i)->y,GFq_squared),
  // 	    fq_get_str_pretty((surface_pts+i)->z,GFq_squared),
  // 	    fq_get_str_pretty((surface_pts+i)->w,GFq_squared));
  //   fputs(str,fp);
  // }
  // fclose(fp);
  // // END DEBUG
    
  fq_clear(u,GFq_squared);
  fq_clear(v,GFq_squared);
  point_clear(&tmp,GFq_squared);  
  point_clear(&conj,GFq_squared);
  fq_clear(a,GFq_squared);
  fprintf(stdout,"found %ld.\n",pt_ind);
  return(pt_ind);
}

/*----------------------------------------------------------------------------*/

// translate a cubic_coef_vec_t object into a vector of indices,
// given by the ordering in fld_elts. This is not a particularly fast
// implementation, so it shouldn't be used in the inner loop. 
void cubic_coef_vec_to_index_vec(int8_t *ind_vec, cubic_coef_vec_t cc_vec,
				 fq_t *fld_elts, fq_ctx_t GFq)
{
  long p, q, d, i, j;
  p = fmpz_get_si(fq_ctx_prime(GFq));
  q = 1;
  d = fq_ctx_degree(GFq);
  for (i=0; i<d; i++) q *= p;
  
  for (i=0; i<20; i++)
  {
    for (j=0; j<q; j++)
    {
      if(fq_equal(*(fld_elts+j),cc_vec+i,GFq))
      {
	*(ind_vec+i) = j;
	break;
      }
    }
  }
}

