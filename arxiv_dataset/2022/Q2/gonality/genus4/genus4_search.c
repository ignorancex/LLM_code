/*----------------------------------------------------------------------------*/
/*		      Genus-4 Excessive Curve Search Code                     */
/*----------------------------------------------------------------------------*/

#include "genus4.h"
#define WITH_FLD_ELTS 0

/*----------------------------------------------------------------------------*/

int main(int argc, char **argv)
{
  long i, j, k, q;
  time_t begin = time(NULL);
  char time_str[128];

  // Get command line arguments
  if (argc != 3)
  {
    fprintf(stdout,"Usage: genus4_search [field order] [filename]\n");
    return(1);
  }
  q = strtol(argv[1],NULL,10);
  fprintf(stdout, "\nSearching for excessive genus-4 curve candidates over GF(%ld)\n",q);
  FILE *fp = fopen(argv[2],"w");

  // Construct GF(q) (rat_elts) and GF(q^2) (quad_elts)
  // Write FF for the ambient quadratic field in which all computations occur.
  fq_ctx_t FF;
  fq_t *quad_elts;
  int8_t *add_table, *mul_table;
  construct_fields(q, FF, &quad_elts);
  construct_add_table(&add_table,quad_elts,FF);
  construct_mul_table(&mul_table,quad_elts,FF);

  // Construct quadratic points on surface, evaluate cubic monomials,
  // and convert vectors of cubic monomials to index vectors
  point_t *quadric_pts;
  long num_pts = nonsplit_quadric_surface_points(q,&quadric_pts,quad_elts,FF);
  cubic_coef_vec_t *xx; // one monomial vector for each quadric pt
  cubic_monomial_evaluations(&xx,quadric_pts,num_pts,quad_elts,FF);
  int8_t* xx_vecs[num_pts]; // Array of pointers, which will store vectors of length 20
  for (i=0; i<num_pts; i++)
  {
     xx_vecs[i] = (int8_t *)malloc(20*sizeof(int8_t));
     cubic_coef_vec_to_index_vec(xx_vecs[i],*(xx+i),quad_elts,FF);
  }

  // // DEBUG: Print the points and cubic monomial valuations
  // for (i=0; i<num_pts; i++)
  // {    
  //   fprintf(stdout,"point %ld: ",i+1);
  //   point_pretty_print(quadric_pts+i,FF);
  //   fprintf(stdout,", ");
  //   for (j=0; j<20; j++) fprintf(stdout, "%d, ",*(xx_vecs[i]+j));
  //   // cubic_coef_vec_pretty_print(*(xx+i),FF);
  //   fprintf(stdout,"\n");
  // }
  // // END DEBUG  

  // Count cubics to test: only need them up to scalar multiple, and if the
  // x^3- or y^3-coefficient vanishes, then the point (1,0,0,0) or (0,1,0,0)
  // lies on it. We can also cancel the x^2y, xy^2, z^3, and w^3 terms by
  // adding an appropriate product of xy + z^2 + azw + w^2 with a linear form.
  // Consequently, there are q^14 (q-1) to test. 
  long total_cubics = q-1;
  for (i=0; i<14; i++) total_cubics *= q;
  fprintf(stdout,"Testing %ld cubics ...\n",total_cubics);

  // Iterate over cubics: we start with the coefficient vector
  //   (1,0,...,0,1,0,...0),
  // where the 0-th and 10-th entries are 1. (See above about x^3
  // and y^3 coefficient.) 

  cubic_coef_vec_t cubic;
  cubic_coef_vec_init(&cubic,FF);
  int8_t ind_vec[20];
  for (i=0; i<20; i++) ind_vec[i] = 0;
  ind_vec[0] = 1;
  ind_vec[10] = 1;
  fq_t dot;
  fq_init(dot,FF);
  long pointless, count=0;
  int8_t dot_ind = 1;

  long steps_until_print = total_cubics / 100;
  long num_prints = 0;
  time_t loop_timer = time(NULL);
  long pts_tested = 0;
  long qq = q*q;
  for (k=0; k<total_cubics; k++)
  {
    pointless = 1;
    // Set coefficients of the k-th cubic (GF(q) elements come first in quad_elts) 
    // for (i=0; i<20; i++) fq_set(cubic+i,quad_elts[ind_vec[i]],FF);
    for (j=0; j<num_pts; j++)
    {
      pts_tested += 1;
#if WITH_FLD_ELTS
      for (i=0; i<20; i++) fq_set(cubic+i,quad_elts[ind_vec[i]],FF);
      _fq_vec_dot(dot,*(xx+j),cubic,20,FF);
      if (!fq_is_zero(dot,FF)) continue;
#else      
      dot_ind = fq_vec_dot_table(xx_vecs[j],ind_vec,20,add_table,mul_table,qq);
      if (dot_ind != 0) continue;
#endif /* WITH_FLD_ELTS */
      pointless = 0;
      break;
    }
    if (pointless)
    {
#if !WITH_FLD_ELTS
      for (i=0; i<20; i++) fq_set(cubic+i,quad_elts[ind_vec[i]],FF);
#endif /* WITH_FLD_ELTS */
      print_cubic_to_file(cubic,fp,FF);
      // print_cubic_to_screen(cubic,FF); // DEBUG
      count++;
    }
    while(next_projective_vector(ind_vec,20,q-1)!=0)
    {
      next_projective_vector(ind_vec,20,q-1);
      if (ind_vec[0] && ind_vec[10] && (ind_vec[1]==0) && (ind_vec[4]==0) &&
	  (ind_vec[16]==0) && (ind_vec[19]==0)) break;
    }
    if ((k+1) % steps_until_print == 0)
    {
      fprintf(stdout,"%5.2f%% ",k*100.0/total_cubics);
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
  if (next_projective_vector(ind_vec,20,q-1))
  {
    fprintf(stderr,"ERROR: Did not finish iterating over cubics!\n");
    exit(1);
  }
  fprintf(stdout,"Average points tested per cubic: %.2f\n",((float) pts_tested) / total_cubics);
  format_time(time_str,time(NULL)-begin);
  fprintf(stdout,"Found %ld excessive curve candidates\n",count);
  fprintf(stdout,"Total time: %s\n",time_str);
    

  // Clean up and return
  fclose(fp);
  for (i=0; i < num_pts; i++) point_clear(quadric_pts+i,FF);
  free(quadric_pts);
  for (i=0; i < qq; i++) fq_clear(quad_elts[i],FF);
  free(quad_elts);
  for (i=0; i < num_pts; i++) cubic_coef_vec_clear(xx+i,FF);
  free(xx);
  cubic_coef_vec_clear(&cubic,FF);
  fq_clear(dot,FF);
  fq_ctx_clear(FF);
  return(0);
}
