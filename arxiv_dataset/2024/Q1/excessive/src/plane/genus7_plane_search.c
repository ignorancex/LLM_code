/*----------------------------------------------------------------------------*/
/*                 Genus-7 Excessive Plane Curve Search Code                  */
/*----------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------

Search for excessive curves of genus 7 over GF(2) with a cubic point, 
which can be represented as (singular) plane curves of degree 9. There are 
three cases to consider: 
  Case 1: triple cubic point on a line
  Case 2: Double and single points on a line
  Case 3: Cubic point on two distinct lines

Write each resulting polynomial to the file fp as an integer whose string of 0's
and 1's describes the coefficients in the basis given by
point_data_for_degree9_plane_curves.

The search is broken into num_jobs jobs, and a call to this function will only
do the tile of work corresponding to the index job. This tile is approximately 
1 / num_jobs of the total work. We must have num_jobs > 0 and 
0 \le job < num_jobs.

------------------------------------------------------------------------------*/

#include "plane.h"

/*------------------------------------------------------------------------------*/

// Struct for polynomial basis data for excessive genus-7 curves. 
typedef struct {
  int        case_num; // which case of Prop. 6.4 are we in?
  int        num_pols; // number of basis polynomials
  int         dims[5]; // dimensions of five types of basis polynomials
  coef_vec_t pols[49]; // The theory says we need 49 in case 3.
} basis_pol_data_t;

// Set the basis polynomial coefficient vectors, set the number of basis
// polynomials, and set the dimensions of the number of basis polynomials of
// each type. The bases are explicitly constructed following
// Proposition 6.5 and Appendix A in our paper on excessive curves.
// The types of the basis polynomials are given by:
//   Type 1: polynomials that don't vanish at (1,0,0), (0,1,0), or (0,0,1)
//   Type 2: polynomials that do not vanish at (1,1,0)
//   Type 3: polynomials that do not vanish at (1,0,1)
//   Type 4: polynomials that do not vanish at (0,1,1)
//   Type 5: polynomials that do not vanish at (1,1,1)
// For example, for the usual genus-g search over degree-d polynomials
// (as executed in plane_search.c), these dimensions are 3, d-1, d-1, d-1,
// (d-1)(d-2)/2.
void new_basis_data(basis_pol_data_t *pol_data, int case_num)
{
  coef_vec_t *pols = pol_data->pols;
  int *dims = pol_data->dims;
  
  if (case_num < 1 || case_num > 3)
  {
    fprintf(stderr,"ERROR: Invalid case_num %d\n",case_num);
    exit(1);
  }
  pol_data->case_num = case_num;

  // Write down the coefficient vector representation for each of the basis
  // polynomials in terms of the standard basis of monomials of degree 9.
  // These were generated using the script basis_pols.sage, though I edited
  // the comments slightly to make the factorizations shorter. 
  switch (case_num)
  {
    case 1:
      pol_data->num_pols = 37;
      dims[0] =  2;
      dims[1] =  8;
      dims[2] =  8;
      dims[3] =  0;
      dims[4] = 19;
      pols[ 0] = 0x1;               // x^9
      pols[ 1] = 0x3980006;         // (y^3 + y^2*z + z^3)^3
      pols[ 2] = 0x8;               // x^8*y
      pols[ 3] = 0x10;              // x^7*y^2
      pols[ 4] = 0x20;              // x^6*y^3
      pols[ 5] = 0x40;              // x^5*y^4
      pols[ 6] = 0x80;              // x^4*y^5
      pols[ 7] = 0x100;             // x^3*y^6
      pols[ 8] = 0xa000000200;      // x^2*y^3*(y^3 + y^2*z + z^3)
      pols[ 9] = 0x110000400;       // x * y^2 * (y^3 + y^2*z + z^3)^2
      pols[10] = 0x800;             // x^8*z
      pols[11] = 0x1000;            // x^7*z^2
      pols[12] = 0x2000;            // x^6*z^3
      pols[13] = 0x4000;            // x^5*z^4
      pols[14] = 0x8000;            // x^4*z^5
      pols[15] = 0x10000;           // x^3*z^6
      pols[16] = 0xa000020000;      // x^2 * z * (y^3 + y^2*z + z^3)^2
      pols[17] = 0x140040000;       // x * z^2 * (y^3 + y^2*z + z^3)^2
      pols[18] = 0x288000000;       // x * y * z * (y^3 + y^2*z + z^3)^2
      pols[19] = 0x8c00000000;      // x^2 * y * z * (y^5 + y*z^4 + z^5)
      pols[20] = 0x6800000000;      // x^2 * y^2 * z^2 * (y^3 + y^2*z + z^3)
      pols[21] = 0xd000000000;      // x^2 * y^3 * z * (y^3 + y^2*z + z^3)
      pols[22] = 0x10000000000;     // x^3*y*z^5
      pols[23] = 0x20000000000;     // x^3*y^2*z^4
      pols[24] = 0x40000000000;     // x^3*y^3*z^3
      pols[25] = 0x80000000000;     // x^3*y^4*z^2
      pols[26] = 0x100000000000;    // x^3*y^5*z
      pols[27] = 0x200000000000;    // x^4*y*z^4
      pols[28] = 0x400000000000;    // x^4*y^2*z^3
      pols[29] = 0x800000000000;    // x^4*y^3*z^2
      pols[30] = 0x1000000000000;   // x^4*y^4*z
      pols[31] = 0x2000000000000;   // x^5*y*z^3
      pols[32] = 0x4000000000000;   // x^5*y^2*z^2
      pols[33] = 0x8000000000000;   // x^5*y^3*z
      pols[34] = 0x10000000000000;  // x^6*y*z^2
      pols[35] = 0x20000000000000;  // x^6*y^2*z
      pols[36] = 0x40000000000000;  // x^7*y*z
      break;

    case 2:
      pol_data->num_pols = 43;
      dims[0] =  2;
      dims[1] =  8;
      dims[2] =  8;
      dims[3] =  0;
      dims[4] = 25;
      pols[ 0] = 0x1;               // x^9
      pols[ 1] = 0x5e00006;         // (y^3 + y*z^2 + z^3) * (y^3 + y^2*z + z^3)^2
      pols[ 2] = 0x8;               // x^8*y
      pols[ 3] = 0x10;              // x^7*y^2
      pols[ 4] = 0x20;              // x^6*y^3
      pols[ 5] = 0x40;              // x^5*y^4
      pols[ 6] = 0x80;              // x^4*y^5
      pols[ 7] = 0x100;             // x^3*y^6
      pols[ 8] = 0x200;             // x^2*y^7
      pols[ 9] = 0x280000400;       // x * y^5 * (y^3 + y^2*z + z^3)
      pols[10] = 0x800;             // x^8*z
      pols[11] = 0x1000;            // x^7*z^2
      pols[12] = 0x2000;            // x^6*z^3
      pols[13] = 0x4000;            // x^5*z^4
      pols[14] = 0x8000;            // x^4*z^5
      pols[15] = 0x10000;           // x^3*z^6
      pols[16] = 0x20000;           // x^2*z^7
      pols[17] = 0x140040000;       // x * z^2 * (y^3 + y^2*z + z^3)^2
      pols[18] = 0x288000000;       // x * y * z * (y^3 + y^2*z + z^3)^2
      pols[19] = 0x230000000;       // x * y^2 * z * (y^5 + y*z^4 + z^5)
      pols[20] = 0x1a0000000;       // x * y^3 * z^2 * (y^3 + y^2*z + z^3)
      pols[21] = 0x340000000;       // x * y^4 * z * (y^3 + y^2*z + z^3)
      pols[22] = 0x400000000;       // x^2*y*z^6
      pols[23] = 0x800000000;       // x^2*y^2*z^5
      pols[24] = 0x1000000000;      // x^2*y^3*z^4
      pols[25] = 0x2000000000;      // x^2*y^4*z^3
      pols[26] = 0x4000000000;      // x^2*y^5*z^2
      pols[27] = 0x8000000000;      // x^2*y^6*z
      pols[28] = 0x10000000000;     // x^3*y*z^5
      pols[29] = 0x20000000000;     // x^3*y^2*z^4
      pols[30] = 0x40000000000;     // x^3*y^3*z^3
      pols[31] = 0x80000000000;     // x^3*y^4*z^2
      pols[32] = 0x100000000000;    // x^3*y^5*z
      pols[33] = 0x200000000000;    // x^4*y*z^4
      pols[34] = 0x400000000000;    // x^4*y^2*z^3
      pols[35] = 0x800000000000;    // x^4*y^3*z^2
      pols[36] = 0x1000000000000;   // x^4*y^4*z
      pols[37] = 0x2000000000000;   // x^5*y*z^3
      pols[38] = 0x4000000000000;   // x^5*y^2*z^2
      pols[39] = 0x8000000000000;   // x^5*y^3*z
      pols[40] = 0x10000000000000;  // x^6*y*z^2
      pols[41] = 0x20000000000000;  // x^6*y^2*z
      pols[42] = 0x40000000000000;  // x^7*y*z
      break;

    case 3:
      pol_data->num_pols = 49;      
      dims[0] =  3;
      dims[1] =  8;
      dims[2] =  5;
      dims[3] =  5;
      dims[4] = 28;
      pols[ 0] = 0x20001;           // x^2 * (x^7 + z^7)
      pols[ 1] = 0x2000002;         // y^2 * (y^7 + z^7)
      pols[ 2] = 0x101004;          // z^2 * (x^7 + y^7 + z^7)
      pols[ 3] = 0x8;               // x^8*y
      pols[ 4] = 0x10;              // x^7*y^2
      pols[ 5] = 0x20;              // x^6*y^3
      pols[ 6] = 0x40;              // x^5*y^4
      pols[ 7] = 0x80;              // x^4*y^5
      pols[ 8] = 0x100;             // x^3*y^6
      pols[ 9] = 0x200;             // x^2*y^7
      pols[10] = 0x400;             // x*y^8
      pols[11] = 0x22800;           // x^2 * z * (x^3 + x^2*z + z^3)^2
      pols[12] = 0x31000;           // x^2 * z^2 * (x^5 + x*z^4 + z^5)
      pols[13] = 0x62000;           // x * z^3 + (x^5 + x*z^4 + z^5)
      pols[14] = 0x16000;           // x^3 * z^3 * (x^3 + x^2*z + z^3)
      pols[15] = 0x58000;           // x * z^5 * (x^3 + x^2*z + z^3)
      pols[16] = 0x2280000;         // y^2 * z * (y^3 + y^2*z + z^3)^2
      pols[17] = 0x3100000;         // y^2 * z^2 * (y^5 + y*z^4 + z^5)
      pols[18] = 0x6200000;         // y * z^3 * (y^5 + y*z^4 + z^5)
      pols[19] = 0x1600000;         // y^3 * z^3 * (y^3 + y^2*z + z^3)
      pols[20] = 0x5800000;         // y * z^5 * (y^3 + y^2*z + z^3)
      pols[21] = 0x8000000;         // x*y*z^7
      pols[22] = 0x10000000;        // x*y^2*z^6
      pols[23] = 0x20000000;        // x*y^3*z^5
      pols[24] = 0x40000000;        // x*y^4*z^4
      pols[25] = 0x80000000;        // x*y^5*z^3
      pols[26] = 0x100000000;       // x*y^6*z^2
      pols[27] = 0x200000000;       // x*y^7*z
      pols[28] = 0x400000000;       // x^2*y*z^6
      pols[29] = 0x800000000;       // x^2*y^2*z^5
      pols[30] = 0x1000000000;      // x^2*y^3*z^4
      pols[31] = 0x2000000000;      // x^2*y^4*z^3
      pols[32] = 0x4000000000;      // x^2*y^5*z^2
      pols[33] = 0x8000000000;      // x^2*y^6*z
      pols[34] = 0x10000000000;     // x^3*y*z^5
      pols[35] = 0x20000000000;     // x^3*y^2*z^4
      pols[36] = 0x40000000000;     // x^3*y^3*z^3
      pols[37] = 0x80000000000;     // x^3*y^4*z^2
      pols[38] = 0x100000000000;    // x^3*y^5*z
      pols[39] = 0x200000000000;    // x^4*y*z^4
      pols[40] = 0x400000000000;    // x^4*y^2*z^3
      pols[41] = 0x800000000000;    // x^4*y^3*z^2
      pols[42] = 0x1000000000000;   // x^4*y^4*z
      pols[43] = 0x2000000000000;   // x^5*y*z^3
      pols[44] = 0x4000000000000;   // x^5*y^2*z^2
      pols[45] = 0x8000000000000;   // x^5*y^3*z
      pols[46] = 0x10000000000000;  // x^6*y*z^2
      pols[47] = 0x20000000000000;  // x^6*y^2*z
      pols[48] = 0x40000000000000;  // x^7*y*z      
      break;
  }
  
}

/*------------------------------------------------------------------------------*/

// Construct point data for the polynomials in the bases for our search of
// degree-9 plane curves with special cubic point configurations. This
// function allocates space for the point data, and then computes it using the
// point data for the individual monomials.
//
//   Case 1: triple cubic point on a line
//   Case 2: Double and single points on a line
//   Case 3: Cubic point on two distinct lines
void point_data_for_degree9_plane_curves(point_data_t **basis_pt_data,
					 basis_pol_data_t *basis_data,
					 point_data_t *monomial_pt_data,
					 int num_pts, table_t *add_table)
{
  point_data_t *data = (point_data_t *) malloc(num_pts*sizeof(point_data_t));
  if (data==NULL) MEMORY_ERROR;

  int i, j, num_monomials;
  point_data_t *P, *Q;
  coef_vec_t pol;
  coef_vec_t *pols = basis_data->pols;
  int num_basis_pols = basis_data->num_pols;

  for (i=0; i<num_pts; i++)
  {
    P = data+i; // New point data
    point_data_init(P,num_basis_pols);
    Q = monomial_pt_data+i; // Point data for monomials
    num_monomials = Q->len; // Should be the same for each i
    P->pt = Q->pt;
    P->x = Q->x;
    P->y = Q->y;
    P->z = Q->z;
    for (j=0; j<num_basis_pols; j++)
    {
      pol = pols[j];
      (P->pt_vals)[j] = fq_sum_terms_table(pol,Q->pt_vals,num_monomials,add_table);
      (P->x_deriv)[j] = fq_sum_terms_table(pol,Q->x_deriv,num_monomials,add_table);
      (P->y_deriv)[j] = fq_sum_terms_table(pol,Q->y_deriv,num_monomials,add_table);
      (P->z_deriv)[j] = fq_sum_terms_table(pol,Q->z_deriv,num_monomials,add_table);
    }
  }
  *basis_pt_data = data;
}

/*----------------------------------------------------------------------------*/

// Rearrange the list of 70 quartic points so the quadratic ones comes first.
// There are no rational points on this list. 
void put_quadratic_first(point_t * pts, int num_pts, fq_ctx_t FF)
{
  point_t tmp;
  point_init(&tmp,FF);
  int i,j=0;
  for (i=0; i<num_pts; i++)
  {
    // Swap the i-th point and j-th point if i-th is quadratic
    point_frobenius(&tmp, pts+i, FF);
    point_frobenius(&tmp, &tmp, FF);
    if (point_eq_raw(&tmp,pts+i,FF))
    {
      point_set(pts+i,pts+j,FF);
      point_set(pts+j,&tmp,FF);
      j++;
    }
  }
  point_clear(&tmp,FF);
}

/*----------------------------------------------------------------------------*/

// If pol is an integer representation of a polynomial in our special basis,
// convert it to the integer representation of a polynomial in the standard
// monomial basis.
coef_vec_t convert_to_std_basis(coef_vec_t pol, basis_pol_data_t *pol_data)
{
  coef_vec_t new_pol = 0;
  int i = 0;
  int64_t ibit;
  while (pol)
  {
    // Determine if bit i of pol is set.
    // If so, unset it and xor in the appropriate value to new_pol
    ibit = 1LL<<i;
    if (pol & ibit) // Nonzero iff bit i of pol is set
    {
      pol ^= ibit;
      new_pol ^= (pol_data->pols)[i];
    }
    i++;
  }
  return new_pol;
}

/*----------------------------------------------------------------------------*/

int main(int argc, char **argv)
{
  // Read arguments
  if (argc != 5)
  {
    fprintf(stderr,"Usage: genus7_plane_search [case] [filename] [num_jobs] [job]\n");
    return(1);
  }
  
  int case_num = strtol(argv[1],NULL,10);
  if (case_num < 1 || case_num > 3)
  {
    fprintf(stderr,"ERROR: Invalid case_num %d\n",case_num);
    exit(1);
  }
  
  FILE *fp = fopen(argv[2],"w");
  if (fp==NULL)
  {
    fprintf(stderr,"ERROR: Cannot open file %s\n",argv[2]);
    exit(1);
  }
  
  int num_jobs = strtol(argv[3],NULL,10);
  int job = strtol(argv[4],NULL,10);
  // The function start_and_stop_work checks validity of these arguments.

  fprintf(stdout,"\nSearching for excessive plane curves of genus 7"
	         " and degree 9\nover GF(2) with ");
  switch (case_num)
  {
    case 1:
      fprintf(stdout,"a triple cubic point on the line x = 0\n\n");
      break;
    case 2:
      fprintf(stdout,"a simple and a double cubic point on the line x = 0\n\n");
      break;
    case 3:
      fprintf(stdout,"simple cubic points on the lines x = 0 and y = 0\n\n");
      break;
  }
  fprintf(stdout, " Job index: %d\nTotal jobs: %d\n\n",job,num_jobs);

  // GF(16) arithmetic
  fq_ctx_t EE;
  fq_t *EE_elts;
  construct_field(EE,&EE_elts,4); // excessive => no quintic point
  table_t EE_add_table;
  table_t EE_mul_table;
  construct_op_table(&EE_add_table,EE_elts,EE,1); // Addition table
  construct_op_table(&EE_mul_table,EE_elts,EE,0); // Multiplication table
  
  // GF(32) arithmetic
  fq_ctx_t FF; 
  fq_t *FF_elts;
  construct_field(FF,&FF_elts,5); // excessive => no quintic point
  table_t FF_add_table;
  table_t FF_mul_table;
  construct_op_table(&FF_add_table,FF_elts,FF,1); // Addition table
  construct_op_table(&FF_mul_table,FF_elts,FF,0); // Multiplication table

  // Find Galois orbits of points
  point_t *EE_reps, *FF_reps;
  int num_EE_reps = galois_orbits(&EE_reps,EE,EE_elts,0);  
  int num_FF_reps = galois_orbits(&FF_reps,FF,FF_elts,0);
  put_quadratic_first(EE_reps,num_EE_reps,EE);


  // Construct monomials of degree 9 and the polynomial basis for this case
  monomial_t *terms;
  int d = 9; // polynomial degree
  int num_terms = construct_monomials(&terms,d);
  basis_pol_data_t basis_data;
  new_basis_data(&basis_data, case_num);

  // GF(16) table evaluations of the basis vectors
  point_data_t *EE_monomial_pt_data, *EE_basis_pt_data;  
  point_data_populate_all(&EE_monomial_pt_data,EE_reps,num_EE_reps,EE_elts,EE,&EE_mul_table,terms,num_terms);
  point_data_for_degree9_plane_curves(&EE_basis_pt_data,&basis_data,
				      EE_monomial_pt_data,num_EE_reps,&EE_add_table);

  // GF(32) table evaluations of the basis vectors  
  point_data_t *FF_monomial_pt_data, *FF_basis_pt_data;
  point_data_populate_all(&FF_monomial_pt_data,FF_reps,num_FF_reps,FF_elts,FF,&FF_mul_table,terms,num_terms);
  point_data_for_degree9_plane_curves(&FF_basis_pt_data,&basis_data,
				      FF_monomial_pt_data,num_FF_reps,&FF_add_table);

  
#if DEBUG
  // This next call is for sanity checking the precomputations.
  fprintf(fp,"%d\n",case_num);
  point_data_test_full(fp,EE_monomial_pt_data,num_EE_reps,EE,EE_elts);
  point_data_test_short(fp,EE_basis_pt_data,num_EE_reps);
  point_data_test_full(fp,FF_monomial_pt_data,num_FF_reps,FF,FF_elts);
  point_data_test_short(fp,FF_basis_pt_data,num_FF_reps);
  return 0;
#endif /* DEBUG */

  // Initialize the vector cc of coefficients of our polynomial
  uint64_t cc, cc_std;
  if (case_num == 3) cc = 7; // Set first 3 bits of cc
  else cc = 3;               // Set first 2 bits of cc
  int *dims = basis_data.dims;
  int  xy_dim = dims[1];
  int  xz_dim = dims[2];
  int  yz_dim = dims[3];
  int xyz_dim = dims[4];
  int  xy_offset = dims[0];
  int  xz_offset = xy_offset + xy_dim;
  int  yz_offset = xz_offset + xz_dim;
  int xyz_offset = yz_offset + yz_dim;
  int rv_xy;
  int rv_xz; 
  int rv_yz;
  int rv_xyz;

  print_queue_t queue;
  print_queue_init(&queue);

  // Job distribution setup: steps through outer loop
  uint64_t i;
  uint64_t loop_work = 1UL << (xyz_dim-1); // -1 because only odd weight vectors used
  uint64_t start, stop;
  start_and_stop_work(&start,&stop,loop_work,num_jobs,job);
  
  
  time_t begin = time(NULL);
  char time_str[128];
  fprintf(stdout,"Passes through outer loop: %llu\n",(unsigned long long) (stop-start));
  uint64_t steps_until_print = (stop-start) / 100;
  if (steps_until_print == 0) steps_until_print = 1;
  int num_prints = 0;
  time_t loop_timer = time(NULL);  

  int j, is_good;
  uint64_t curve_count = 0;
  odd_weight_vector_with_offset_and_index(&cc,xyz_dim,xyz_offset,start);
  for (i=start; i<stop; i++)
  {
    odd_weight_vector_with_offset_and_index(&cc,xy_dim,xy_offset,0);
    while (1)
    {
      odd_weight_vector_with_offset_and_index(&cc,xz_dim,xz_offset,0);
      while (1)
      {
	odd_weight_vector_with_offset_and_index(&cc,yz_dim,yz_offset,0);
  	while (1)
  	{
	  is_good = 1;
	  // Test for smooth quartic points on the curve
	  for (j=0; j<num_EE_reps; j++)
	  {
	    if (point_is_smooth_on_curve(EE_basis_pt_data+j,cc,&EE_add_table))
	    {
	      is_good = 0;
	      break;
	    }
	  }
	  if (is_good)
	  {
	    // Test for smooth quintic points on the curve
	    for (j=0; j<num_FF_reps; j++)
	    {
	      if (point_is_smooth_on_curve(FF_basis_pt_data+j,cc,&FF_add_table))
	      {
		is_good = 0;
		break;
	      }
	    }
	  }
	  if (is_good)
	  {
	    cc_std = convert_to_std_basis(cc,&basis_data);
	    print_queue_write(&queue,cc_std,fp);
	    curve_count++;
	  }
  	  rv_yz = next_odd_weight_vector_with_offset(&cc,yz_dim,yz_offset);
  	  if (rv_yz==0) break;
  	}
  	rv_xz = next_odd_weight_vector_with_offset(&cc,xz_dim,xz_offset);
  	if (rv_xz==0) break;
      }
      rv_xy = next_odd_weight_vector_with_offset(&cc,xy_dim,xy_offset);
      if (rv_xy==0) break;
    }
    rv_xyz = next_odd_weight_vector_with_offset(&cc,xyz_dim,xyz_offset);
    if ((rv_xyz==0) && (i != stop-1))
    {
      fprintf(stderr,"ERROR: Unexpected end to iteration.\n");
      exit(1);
    }
    // Loop is finished: print something useful sometimes
    if (i % steps_until_print == 0)
    {
      fprintf(stdout,"%5.2f%% ",(i-start)*100.0/(stop-start));
      fflush(stdout);
      num_prints++;
      if (num_prints==10)
      {
    	num_prints = 0;
    	format_time(time_str,time(NULL)-loop_timer);
    	fprintf(stdout,"- Time: %s\n",time_str);
    	loop_timer = time(NULL);
      }     
    }	      
  }
  print_queue_purge(&queue,fp);  
  print_queue_clear(&queue);
  fprintf(stdout,"\nFound %lld curves\n",(unsigned long long) curve_count);
  format_time(time_str,time(NULL)-begin);  
  fprintf(stdout,"Total time: %s\n",time_str);

  // Clean up!
  for (i=0; i<num_EE_reps; i++)
  {    
    point_data_clear(EE_monomial_pt_data+i);
    point_data_clear(EE_basis_pt_data+i);
    point_clear(EE_reps+i,EE);
  }
  free(EE_reps);  
  free(EE_monomial_pt_data);
  free(EE_basis_pt_data);
  free(EE_add_table.entries);
  free(EE_mul_table.entries);
  clear_field(EE,EE_elts);
  
  for (i=0; i<num_FF_reps; i++)
  {
    point_data_clear(FF_monomial_pt_data+i);
    point_data_clear(FF_basis_pt_data+i);
    point_clear(FF_reps+i,FF);
  }
  free(FF_reps);  
  free(FF_monomial_pt_data);
  free(FF_basis_pt_data);
  free(FF_add_table.entries);
  free(FF_mul_table.entries);
  clear_field(FF,FF_elts);

  free(terms);
  fclose(fp);
  return(0);
}
