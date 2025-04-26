/*----------------------------------------------------------------------------*/
/*                  General Excessive Plane Curve Search Code                 */
/*----------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------

Search for excessive curves of genus g over GF(2) with a divisor of
degree g-3, which can be represented as (singular) plane curves of degree g+1.
If the genus is 5, 6, or 7, we also require that the curve pass through a
special point of degree 2, 3, or 4.  Write each resulting polynomial to the file
fp as an integer whose string of 0's and 1's gives the coefficients of the
polynomial in the monomial basis given by construct_monomials.

The search is broken into num_jobs jobs, and a call to this function will only
do the tile of work corresponding to the index job. This tile is approximately 
1 / num_jobs of the total work. We must have num_jobs > 0 and 
0 \le job < num_jobs.

------------------------------------------------------------------------------*/

#include "plane.h"

/*------------------------------------------------------------------------------*/

// Function to construct special points that we insist our curve pass
// through. One search algorithm we are using insists that an excessive curve of
// genus g has an effective divisor of degree g-3. For genus 5, this means our
// curve has a quadratic point. Up to the PGL_3-action, we may take this point
// to be (0,1,t) where t^2 + t + 1 = 0. For genus 6, it means that our curve has
// a cubic point, which we may take to be (0,1,t) or (1,t,t^2), where t^3 + t +
// 1 = 0. For genus 7, it means that our curve has a quadratic point or a
// quartic point, which we may take as (0,1, s^2+s), (0,1,s), or (1,s,s^2),
// where s^4 + s + 1 = 0.
//
// This function allocates space for the special points, constructs them, and
// populates the appropriate addition table that will be needed to work with
// them. It returns the number of special points constructed.
int construct_special_points(point_data_t **specials, int genus, table_t *add_table,
			       monomial_t *terms, int num_terms)
{
  if (genus < 5 || genus > 7)
  {
    *specials = NULL;
    add_table->entries = NULL;
    return 0;
  }
  
  fq_ctx_t EE;
  fq_t *elts; 
  construct_field(EE,&elts,genus-3); // gen = elts[2]
  table_t mul_table;
  construct_op_table( add_table, elts, EE, 1);
  construct_op_table(&mul_table, elts, EE, 0);

  point_t pts[3]; // Need at most three of these
  int i, num_pts, max_pts = 3;
  for (i=0; i<max_pts; i++) point_init(pts+i,EE);

  point_t *P;
  fq_t *gen = elts+2;
  if (genus == 5)
  {
    P = pts;
    fq_zero(P->x,EE);
    fq_one(P->y,EE);
    fq_set(P->z,*gen,EE);
    num_pts = 1;
  }
  else if (genus == 6)
  {
    P = pts; // (0,1,t)
    fq_zero(P->x,EE);
    fq_one(P->y,EE);
    fq_set(P->z,*gen,EE);
    P = pts+1; // (1,t,t^2)
    fq_one(P->x,EE);
    fq_set(P->y,*gen,EE);
    fq_mul(P->z,*gen,*gen,EE);
    num_pts = 2;
  }
  else
  {
    P = pts; // (0,1,s^2+s), because (s^2+s)^2 + (s^2+s) + 1 = 0
    fq_zero(P->x,EE);
    fq_one(P->y,EE);
    fq_mul(P->z,*gen,*gen,EE); // z = s^2
    fq_add(P->z,P->z,*gen,EE); // z = s^2 + s
    P = pts+1; // (0,1,s)
    fq_zero(P->x,EE);
    fq_one(P->y,EE);
    fq_set(P->z,*gen,EE);
    P = pts+2; // (1,s,s^2)
    fq_one(P->x,EE);
    fq_set(P->y,*gen,EE);
    fq_mul(P->z,*gen,*gen,EE);
    num_pts = 3;
  }

  point_data_populate_all(specials,pts,num_pts,elts,EE,&mul_table,terms,num_terms);

  // Clean up
  for (i=0; i<max_pts; i++) point_clear(pts+i,EE);  
  clear_field(EE,elts);
  free(mul_table.entries);
  return num_pts;
}

/*----------------------------------------------------------------------------*/

// Return 1 if no special point is designated, or if genus is 5,6,7 and curve
// has a special point. Return 0 otherwise.
int has_special_point(coef_vec_t coefs, point_data_t *specials, int num_specials,
		       table_t *special_add_table)
{
  if (num_specials == 0) return 1;
  int j;
  point_data_t *P = specials; // abbreviate
  int64_t len = P->len;
  for (j = 0; j<num_specials; j++)
  {
    if (fq_sum_terms_table(coefs,(P+j)->pt_vals,len,special_add_table)==0) return 1;    
  }
  return 0;
}

/*----------------------------------------------------------------------------*/

int main(int argc, char **argv)
{
  // Read arguments
  if (argc != 5)
  {
    fprintf(stderr,"Usage: plane_search [genus] [filename] [num_jobs] [job]\n");
    return(1);
  }
  int g = strtol(argv[1],NULL,10);
  if (g < 3)
  {
    fprintf(stderr,"ERROR: genus must be at least 4\n");
    return(1);
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

  fprintf(stdout, "\nSearching for excessive curves of genus "
	          "%d over GF(2)\nwith an effective divisor of degree "
                  "%d\n\n", g, g-3);

  fprintf(stdout, " Job index: %d\nTotal jobs: %d\n\n",job,num_jobs);

  fq_ctx_t FF;
  fq_t *elts;
  construct_field(FF,&elts,g-2);
  table_t add_table;
  table_t mul_table;
  construct_op_table(&add_table,elts,FF,1); // Addition table
  construct_op_table(&mul_table,elts,FF,0); // Multiplication table

  point_t *reps;  
  int num_reps = galois_orbits(&reps,FF,elts,0);

  monomial_t *terms;
  int d = g+1; // polynomial degree
  int num_terms = construct_monomials(&terms,d);

  point_data_t *data;
  point_data_populate_all(&data,reps,num_reps,elts,FF,&mul_table,terms,num_terms);

 // Construct list of special points for g = 5, 6, 7
 point_data_t *specials;
 table_t special_add_table;
 int num_specials = construct_special_points(&specials,g,&special_add_table,terms,num_terms);

#if DEBUG
  // This next call is for sanity checking the precomputations.
  point_data_test_full(fp,data,num_reps,FF,elts);
  if (g >= 5 && g <= 7) point_data_test_short(fp,specials,num_specials);
  return 0;
#endif /* DEBUG */

  // Initialize the vector cc of coefficients of our polynomial
  uint64_t cc = 7; // Set first three bits of cc
  int  xy_offset = 3;
  int  xz_offset = 3 + d-1;
  int  yz_offset = 3 + 2*(d-1);
  int xyz_offset = 3 + 3*(d-1);  
  int bideg = d-1;
  int trideg = (d-1)*(d-2)/2;
  int rv_xy;
  int rv_xz; 
  int rv_yz;
  int rv_xyz;

  print_queue_t queue;
  print_queue_init(&queue);

  // Job distribution setup: steps through outer loop  
  uint64_t i;
  uint64_t loop_work = 1UL << (trideg-1); // -1 because only odd weight vectors used
  uint64_t start, stop;
  start_and_stop_work(&start,&stop,loop_work,num_jobs,job);
  
  
  time_t begin = time(NULL);
  char time_str[128];
  fprintf(stdout,"Passes through outer loop: %lld\n",(unsigned long long) (stop-start));
  uint64_t steps_until_print = (stop-start) / 100;
  if (steps_until_print == 0) steps_until_print = 1;
  int num_prints = 0;
  time_t loop_timer = time(NULL);  

  int is_good, j;
  uint64_t curve_count = 0;
  odd_weight_vector_with_offset_and_index(&cc,trideg,xyz_offset,start);
  for (i=start; i<stop; i++)
  {
    odd_weight_vector_with_offset_and_index(&cc,bideg,xy_offset,0);
    while (1)
    {
      odd_weight_vector_with_offset_and_index(&cc,bideg,xz_offset,0);
      while (1)
      {
	odd_weight_vector_with_offset_and_index(&cc,bideg,yz_offset,0);
  	while (1)
  	{	  
	  is_good = has_special_point(cc,specials,num_specials,&special_add_table);
	  if (is_good)
	  {
	    for (j=0; j<num_reps; j++)
	    {
	      if (point_is_smooth_on_curve(data+j,cc,&add_table))
	      {
		is_good = 0;
		break;
	      }
	    }
	  }
	  if (is_good)
	  {
	    print_queue_write(&queue,cc,fp);
	    curve_count++;
	  }
  	  rv_yz = next_odd_weight_vector_with_offset(&cc,bideg,yz_offset);
  	  if (rv_yz==0) break;
  	}
  	rv_xz = next_odd_weight_vector_with_offset(&cc,bideg,xz_offset);
  	if (rv_xz==0) break;
      }
      rv_xy = next_odd_weight_vector_with_offset(&cc,bideg,xy_offset);
      if (rv_xy==0) break;
    }
    rv_xyz = next_odd_weight_vector_with_offset(&cc,trideg,xyz_offset);
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
  for (j=0; j<num_reps; j++)
  {    
    point_data_clear(data+j);
    point_clear(reps+j,FF);
  }
  for (j=0; j<num_specials; j++) point_data_clear(specials+j);
  free(reps);
  free(terms);
  free(data);
  free(specials);
  free(add_table.entries);
  free(mul_table.entries);
  free(special_add_table.entries);
  clear_field(FF,elts);
  fclose(fp);
  return(0);
}
