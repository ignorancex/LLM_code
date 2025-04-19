/*----------------------------------------------------------------------------*/
/*    		      Find Quadratic Points on a Quartic                      */
/*----------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------

Code to test the quadratic point search functions as well as the matrix
construction functions affiliated with them. 

------------------------------------------------------------------------------*/

#include "quartic.h"

/*----------------------------------------------------------------------------*/


int main(int argc, char **argv)
{
  int q = 3; 
  uint16_t Q[11] = {1,2,1,2,1,0,1,0,0,1,0}; // -x^4 + x^2*y^2 - y^4 + x^3*z + x*y^2*z + y^3*z + x*z^3 + y*z^3 + z^4

  fprintf(stdout, "\nSearching for quadratic points on a quartic plane curve over GF(%d)\n"
	          "and determining which pairs are in general position.\n",q);

  // Construct field and element arrays
  fq_ctx_t FF;
  fq_t *elts;
  construct_quadratic_extension(FF,&elts,q);

  // Construct the minimal polynomial x^2 + ax + 1 for a quadratic extension
  fq_t a;
  fq_init(a,FF);
  irreducible_quadratic(a,elts,FF);

  // Evaluate quartic basis polynomials at quadratic places and store point indices
  point_t *quadpts;
  fq_idx_vec_t *quadpt_data;
  fq_idx_vec_t *quadpt_inds;
  int num_quad_pts = (q*q*q*q-q)/2;
  construct_quadratic_points(&quadpts,elts,FF);
  quartic_basis_pol_evaluations(&quadpt_data,quadpts,num_quad_pts,elts,a,FF);
  quad_point_indices(&quadpt_inds,quadpts,elts,FF);

  // Construct fast arithmetic tables
  ops_store_t tables;
  construct_ops_store(&tables,a,elts,FF);

  uint16_t c = tables.root;
  fprintf(stdout, "q = %d, a = %s, c = %s\n",q,fq_get_str_pretty(a,FF),fq_get_str_pretty(elts[c],FF));

  // Find quadratic points
  int i, j, num_places;
  int Qquad_inds[QUADBOUND];
  quadratic_places_on_quartic(Qquad_inds,&num_places,Q,quadpt_data,quadpt_inds,&tables);
  fprintf(stdout,"Found %d quadratic points on Q:\n",num_places);
  for (i=0; i<num_places; i++)
  {
    point_pretty_print(quadpts+Qquad_inds[i],FF);
    fprintf(stdout,"\n");
  }

// EXPECT this:
// pts = [(1, 0, t+1),
//        (1, 0, t+2),
//        (1, t, t),
//        (1, t+1, 0),
//        (1, t+1, 2),
//        (1, t+2, 2),
//        (0, 1, t+1),
//        (0, 1, t+2)]
  

  // Determine which pairs are in general position
  fq_idx_vec_t pt1, pt2;
  fprintf(stdout,"\nFinding pairs *not* in general position:\n");
  for (i=0; i<num_places; i++)
  {
    for (j=i+1; j<num_places; j++)
    {
      pt1 = quadpt_inds[Qquad_inds[i]];
      pt2 = quadpt_inds[Qquad_inds[j]];
      if (quad_places_in_general_position(pt1,pt2,&tables)) continue;
      fprintf(stdout,"%d, %d\n",i,j);
    }
  }  

// EXPECT this:
// 0, 1
// 4, 5
// 6, 7

  // Determine image of Q under a matrix that moves a pair of quadratic points
  // to (c:0:1) and (0:c:1)
  uint16_t mm[9];
  uint16_t newQ[11] = {1,0,0,0,0,0,0,0,0,0,0}; // Never modify the 0-th entry.
  uint16_t lc;
  fprintf(stdout,"\nFinding matrices that preserve special form of our quartic\n");
  for (i=0; i<num_places; i++)
  {
    for (j=i+1; j<num_places; j++)
    {
      pt1 = quadpt_inds[Qquad_inds[i]];
      pt2 = quadpt_inds[Qquad_inds[j]];
      if (!quad_places_in_general_position(pt1,pt2,&tables)) continue;
      mm[0] = quad_transmat_entry00(pt1,pt2,&tables);
      mm[1] = quad_transmat_entry01(pt1,pt2,&tables);
      mm[2] = quad_transmat_entry02(pt1,pt2,&tables);
      mm[3] = quad_transmat_entry10(pt1,pt2,&tables);
      mm[4] = quad_transmat_entry11(pt1,pt2,&tables);
      mm[5] = quad_transmat_entry12(pt1,pt2,&tables);
      mm[6] = quad_transmat_entry20(pt1,pt2,&tables);
      mm[7] = quad_transmat_entry21(pt1,pt2,&tables);
      mm[8] = quad_transmat_entry22(pt1,pt2,&tables);

      fprintf(stdout,"pt1 = ");
      point_pretty_print(quadpts+Qquad_inds[i],FF);
      fprintf(stdout,", pt2 = ");
      point_pretty_print(quadpts+Qquad_inds[j],FF);
      fprintf(stdout,"\n");
      fprintf(stdout,"Matrix coefficients: ");
      write_index_vector_to_stdout(mm,9);
      fprintf(stdout,"\n\n");

      // FIXME: Remove these checks when happy with code --- they just slow it down
      // Expect x^2*z^2, y^2*z^2, x*z^3, y*z^3 to vanish
      if (xxzz_coef(Q,mm,&tables) || yyzz_coef(Q,mm,&tables))
      {
	fprintf(stderr,"ERROR: x^2*z^2 or y^2*z^2 coefficients don't vanish\n");
	exit(1);
      }
      if (xzzz_coef(Q,mm,&tables) || yzzz_coef(Q,mm,&tables))
      {
	fprintf(stderr,"ERROR: x*z^3 or y*z^3 coefficients don't vanish\n");
	exit(1);	
      }

      // Normalize by dividing by the z^4 coefficient
      lc = zzzz_coef(Q,mm,&tables);
      lc = tables.invs[lc]; // Get inverse
      if (lc==0)
      {
	fprintf(stderr,"ERROR: z^4 coefficient vanishes\n");
	exit(1);	
      }
      newQ[1]  = xxxx_coef(Q,mm,&tables);
      newQ[2]  = xxxz_coef(Q,mm,&tables);
      newQ[3]  = yyyy_coef(Q,mm,&tables);
      newQ[4]  = yyyz_coef(Q,mm,&tables);
      newQ[5]  = xxxy_coef(Q,mm,&tables);
      newQ[6]  = xxyy_coef(Q,mm,&tables);
      newQ[7]  = xyyy_coef(Q,mm,&tables);
      newQ[8]  = xxyz_coef(Q,mm,&tables);
      newQ[9]  = xyyz_coef(Q,mm,&tables);    
      newQ[10] = xyzz_coef(Q,mm,&tables);
  
      newQ[1]  = fq_idx_mul(&lc,newQ+1,&tables);
      newQ[2]  = fq_idx_mul(&lc,newQ+2,&tables);
      newQ[3]  = fq_idx_mul(&lc,newQ+3,&tables);
      newQ[4]  = fq_idx_mul(&lc,newQ+4,&tables);
      newQ[5]  = fq_idx_mul(&lc,newQ+5,&tables);
      newQ[6]  = fq_idx_mul(&lc,newQ+6,&tables);
      newQ[7]  = fq_idx_mul(&lc,newQ+7,&tables);
      newQ[8]  = fq_idx_mul(&lc,newQ+8,&tables);
      newQ[9]  = fq_idx_mul(&lc,newQ+9,&tables);
      newQ[10] = fq_idx_mul(&lc,newQ+10,&tables);
    }
  }
  
  // Clean up!
  clear_ops_store(&tables);
  for (i=0; i<num_quad_pts; i++) point_clear(quadpts+i,FF);    
  for (i=0; i<num_quad_pts; i++) fq_idx_vec_clear(quadpt_data[i]);
  for (i=0; i<num_quad_pts; i++) fq_idx_vec_clear(quadpt_inds[i]);
  free(quadpts);
  free(quadpt_data);
  free(quadpt_inds);
  fq_clear(a,FF);
  clear_field(FF,elts);
  return(0);
}
