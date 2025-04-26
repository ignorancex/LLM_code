/*----------------------------------------------------------------------------*/
/*		      Smooth Pointless Quartic Search Code                    */
/*----------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------

Search for excessive curves of genus 3 over GF(q). If q \ge 5, then any such
curve can be represented as a smooth pointless plane quartic that vanishes at 2
special quadratic points. This means the dimension of the search is 11.

The search is broken into num_jobs jobs, and a call to this function will only
do the tile of work corresponding to the index job. This tile is approximately 1
/ num_jobs of the total work. We must have num_jobs > 0 and 0 \le job <
num_jobs.

------------------------------------------------------------------------------*/

#include "quartic.h"

/*----------------------------------------------------------------------------*/


int main(int argc, char **argv)
{
  // Read arguments
  if (argc != 5)
  {
    fprintf(stderr,"Usage: quartic_search [q] [filename] [num_jobs] [job]\n");
    return(1);
  }
  int q = strtol(argv[1],NULL,10);
  FILE *fp = fopen(argv[2],"w");
  if (fp==NULL)
  {
    fprintf(stderr,"ERROR: Cannot open file %s\n",argv[2]);
    return(1);
  }
  int num_jobs = strtol(argv[3],NULL,10);
  int job = strtol(argv[4],NULL,10);
  // The function start_and_stop_work checks validity of these arguments.

  fprintf(stdout, "\nSearching for excessive quartic plane curves over GF(%d)\n"
	          "through two quadratic points in general position.\n",q);

  fprintf(stdout, " Job index: %d\nTotal jobs: %d\n\n",job,num_jobs);

  // Construct field and element arrays
  fq_ctx_t FF;
  fq_t *elts;
  construct_quadratic_extension(FF,&elts,q);

  // Construct the minimal polynomial x^2 + ax + 1 for a quadratic extension
  fq_t a;
  fq_init(a,FF);
  irreducible_quadratic(a,elts,FF);

  // Construct special basis of polynomials and evaluate at all rational points
  point_t *xpts, *ypts, *zpts, *genpts;
  fq_idx_vec_t *xpt_data, *ypt_data, *zpt_data, *genpt_data;
  construct_rational_point_arrays(&xpts,&ypts,&zpts,&genpts,elts,FF);
  quartic_basis_pol_evaluations(&xpt_data,xpts,q,elts,a,FF);
  quartic_basis_pol_evaluations(&ypt_data,ypts,q,elts,a,FF);
  quartic_basis_pol_evaluations(&zpt_data,zpts,q-1,elts,a,FF);
  quartic_basis_pol_evaluations(&genpt_data,genpts,(q-1)*(q-1),elts,a,FF);

  // Evaluate quartic basis polynomials at quadratic places and store point indices
  point_t *quadpts;
  fq_idx_vec_t *quadpt_data;
  fq_idx_vec_t *quadpt_inds;
  int num_quad_pts = (q*q*q*q-q)/2;
  construct_quadratic_points(&quadpts,elts,FF);
  quartic_basis_pol_evaluations(&quadpt_data,quadpts,num_quad_pts,elts,a,FF);
  quad_point_indices(&quadpt_inds,quadpts,elts,FF);

#if DEBUG
  point_data_test(fp,elts,xpts,ypts,zpts,genpts,
		  xpt_data,ypt_data,zpt_data,genpt_data,
		  quadpts,quadpt_data,a,FF);
  return 0;
#endif /* DEBUG */

  // Construct fast arithmetic tables
  ops_store_t tables;
  construct_ops_store(&tables,a,elts,FF);
  // DEBUG
  // print_ops_store(&tables);
  // pgl_tests(&tables);
  // return 0;
  // END DEBUG

  // Clean up all memory that Flint allocated (Before allocating space for hash table)
  int i;
  for (i=0; i<q; i++) point_clear(xpts+i,FF);
  for (i=0; i<q; i++) point_clear(ypts+i,FF);
  for (i=0; i<q-1; i++) point_clear(zpts+i,FF);
  for (i=0; i<(q-1)*(q-1); i++) point_clear(genpts+i,FF);
  for (i=0; i<num_quad_pts; i++) point_clear(quadpts+i,FF);    
  free(xpts);
  free(ypts);
  free(zpts);
  free(genpts);
  free(quadpts);
  fq_clear(a,FF);
  clear_field(FF,elts);
  
  // Initialize set for tracking pointless quartics
  qset_t seen;
  qset_initialize(&seen,q);

  // Job distribution and timer setup
  uint64_t start, stop;
  uint64_t total_work = q*q-q; // Only outer-most loop to be used for job distribution
  start_and_stop_work(&start,&stop,total_work,num_jobs,job);
  time_t begin = time(NULL);
  fprintf(stdout,"Passes through outer loop: %lld\n",(unsigned long long) (stop-start));
  uint64_t steps_until_print = (stop-start) / 100;
  if (steps_until_print == 0) steps_until_print = 1;
  int num_prints = 0;
  time_t loop_timer = time(NULL);

  // Search! 
  // Initialize quartic:
  //   z^4 monomial has coefficient 1 (else (0:0:1) on the curve)
  //   x^4 monomial has nonzero coefficient (else (1:0:0) on the curve)
  uint16_t Q[11] = {1,0,0,0,0,0,0,0,0,0,0};
  start_vector(Q+1,2,q,start+q); // index start+q so that x^4 coefficient is nonzero
  int num_curves = 0;
  int j,k,l;
  // Compute a few constants to avoid doing so in the inner loop
  int qsquared = q*q;
  int qcubed = q*q*q;
  int qminus1 = q-1;
  int qminus1squared = (q-1)*(q-1);
  int satisfies_weil_bound;
  // Loop over Q[1], Q[2]  
  for (i=start; i<stop; i++)
  {
    loop_printer(i,start,stop,steps_until_print,&num_prints,&loop_timer);
    if (quartic_has_rational_point(Q,ypt_data,q,&tables))
    {
      next_vector(Q+1,2,q);
      continue;
    }
    // Loop over Q[3], Q[4]
    start_vector(Q+3,2,q,0);
    for (j=0; j<qsquared; j++)
    {
      // xy-symmetry allows us to assume y^4 coef is lex \ge than x^4 coef
      if (Q[3] < Q[1] || quartic_has_rational_point(Q,xpt_data,q,&tables))
      {
	next_vector(Q+3,2,q);
	continue;
      }
      // Loop over Q[5], Q[6], Q[7]
      start_vector(Q+5,3,q,0);      
      for (k=0; k<qcubed; k++)
      {
	if (quartic_has_rational_point(Q,zpt_data,qminus1,&tables))
	{
	  next_vector(Q+5,3,q);
	  continue;
	}
	start_vector(Q+8,3,q,0);
	for (l=0; l<qcubed; l++)
	{
	  // Quick abort if we've seen this one.
	  if (qset_is_member(&seen,Q))
	  {
	    next_vector(Q+8,3,q);
	    continue; 
	  }
	  if (quartic_has_rational_point(Q,genpt_data,qminus1squared,&tables))
	  {
	    next_vector(Q+8,3,q);
	    continue;
	  }
	  satisfies_weil_bound = mark_good_translates_smart(&seen,Q,quadpt_data,quadpt_inds,&tables);
	  if (satisfies_weil_bound)
	  {
	    write_index_vector_to_file(fp,Q,11);
	    num_curves++;
	  }
	  next_vector(Q+8,3,q);
	}
	next_vector(Q+5,3,q);
      }
      next_vector(Q+3,2,q);
    }
    next_vector(Q+1,2,q);
  }
  fprintf(stdout,"\nFound %d curves\n",num_curves);
  char time_str[128];
  format_time(time_str,time(NULL)-begin);  
  fprintf(stdout,"Total time: %s\n",time_str);

  // Clean up!
  qset_clear(seen);
  clear_ops_store(&tables);
  for (i=0; i<q; i++) fq_idx_vec_clear(xpt_data[i]);
  for (i=0; i<q; i++) fq_idx_vec_clear(ypt_data[i]);
  for (i=0; i<q-1; i++) fq_idx_vec_clear(zpt_data[i]);
  for (i=0; i<(q-1)*(q-1); i++) fq_idx_vec_clear(genpt_data[i]);
  for (i=0; i<num_quad_pts; i++) fq_idx_vec_clear(quadpt_data[i]);
  for (i=0; i<num_quad_pts; i++) fq_idx_vec_clear(quadpt_inds[i]);
  free(xpt_data);
  free(ypt_data);
  free(zpt_data);
  free(genpt_data);
  free(quadpt_data);
  free(quadpt_inds);
  fclose(fp);  
  return(0);
}
