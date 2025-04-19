/*----------------------------------------------------------------------------*/
/*------------------------------  PLANE CURVES  ------------------------------*/
/*----------------------------------------------------------------------------*/

#include "plane.h"

/*----------------------------------------------------------------------------*/

int construct_monomials(monomial_t **terms, int deg)
{
  int d = deg;
  int num_terms = (d+2)*(d+1)/2;
  
  *terms = (monomial_t *) malloc(num_terms*sizeof(monomial_t));
  if (*terms == NULL) MEMORY_ERROR;

  int i, j;
  monomial_t *term = *terms;
  for (i=0; i<num_terms; i++)
  {
    (term+i)->a = 0;
    (term+i)->b = 0;
    (term+i)->c = 0;    
  }

  // Set univariate terms
  (term+0)->a = d;
  (term+1)->b = d;
  (term+2)->c = d;

  // Set the bivariate terms
  int ind = 3;
  for (i=1; i<d; i++)
  {
    (term+ind)->a = d-i;
    (term+ind)->b = i;
    (term+ind+d-1)->a = d-i;
    (term+ind+d-1)->c = i;
    (term+ind+2*(d-1))->b = d-i;
    (term+ind+2*(d-1))->c = i;
    ind++;
  }

  // Set the general terms
  ind = 3*d;
  for (i=1; i<d-1;i++)
  {
    for (j=1; j<d-i; j++)
    {
      (term+ind)->a = i;
      (term+ind)->b = j;
      (term+ind)->c = d-i-j;
      ind++;
    }
  }
  
#if DEBUG
  if (ind != num_terms)
  {
    fprintf(stderr,"Wrong number of monomials. Expected %d, got %d.\n",num_terms,ind);
    exit(1);
  }
  fprintf(stdout,"\nMonomials of degree %d:\n",d);
  for (i=0; i<num_terms; i++)
  {
    if ((term+i)->a != 0)
    {
      fprintf(stdout,"x^%d",(term+i)->a);
    }
    if ((term+i)->b != 0)
    {
      fprintf(stdout,"y^%d",(term+i)->b);
    }
    if ((term+i)->c != 0)
    {
      fprintf(stdout,"z^%d",(term+i)->c);
    }
    fprintf(stdout,"\n");
  }
# endif /* DEBUG */
  return num_terms;
}

/*----------------------------------------------------------------------------*/

void point_data_init(point_data_t *data, int num_terms)
{
  data->len = num_terms;
  table_vec_init(&(data->pt_vals),num_terms);
  table_vec_init(&(data->x_deriv),num_terms);
  table_vec_init(&(data->y_deriv),num_terms);
  table_vec_init(&(data->z_deriv),num_terms);
}
 
/*----------------------------------------------------------------------------*/

void point_data_clear(point_data_t *data)
{
  table_vec_clear(data->pt_vals);
  table_vec_clear(data->x_deriv);
  table_vec_clear(data->y_deriv);
  table_vec_clear(data->z_deriv);
}

/*----------------------------------------------------------------------------*/

// Private function: Return the table value of monomial "term" evaluated at
// the point with table indices x, y, z
static uint16_t evaluate_monomial(monomial_t *term, uint16_t x, uint16_t y,
				  uint16_t z, table_t *mul_table)
{
  // fprintf(stdout,"exps = (%d, %d, %d)\n",term->a,term->b,term->c);
  // fprintf(stdout,"ins = (%d, %d, %d)\n",x,y,z);
  uint16_t rv = 1; // NB: 1 is the index of the element 1
  int i;
  for (i=0; i<term->a; i++)
  {
    rv = fq_op_table(&rv,&x,mul_table);
    // fprintf(stdout,"rv = %d\n",rv);
  }
  for (i=0; i<term->b; i++)
  {
    rv = fq_op_table(&rv,&y,mul_table);
  }
  for (i=0; i<term->c; i++)
  {
    rv = fq_op_table(&rv,&z,mul_table);
  }
  return rv;
}

// Private function: Return the table value of the partial derivative of
// monomial "term" with respect to "var" evaluated at the table point with
// indices x, y, z. We are using the fact that our base field has characteristic
// 2: (d/dt) t^n = 0 if n is even and t^(n-1) if n is odd. 
static uint16_t evaluate_derivative(monomial_t *term, uint16_t x, uint16_t y,
				    uint16_t z, char var, table_t *mul_table)
{
  monomial_t deriv = *term; // This is a copy!
  switch (var)
  {
  case 'x':
    if (deriv.a % 2 == 0) return 0;
    deriv.a--;
    return evaluate_monomial(&deriv,x,y,z,mul_table);
  case 'y':
    if (deriv.b % 2 == 0) return 0;
    deriv.b--;
    return evaluate_monomial(&deriv,x,y,z,mul_table);
  case 'z':
    if (deriv.c % 2 == 0) return 0;
    deriv.c--;
    return evaluate_monomial(&deriv,x,y,z,mul_table);
  default:
    fprintf(stderr,"ERROR: invalid derivative %c\n",var);
    exit(1);
  }
}

/*----------------------------------------------------------------------------*/

// Private function: populate all fields of data; this will allocate space for
// the fields that require it.
static void point_data_populate(point_data_t *data, point_t *P, fq_t *elts,
				fq_ctx_t FF, table_t *mul_table,
				monomial_t *terms, int num_terms)
{
  int i;
  int q = field_cardinality_to_int(FF);
  
  point_data_init(data,num_terms);
  data->pt = P;
  for (i=0; i<q; i++)
  {
    if (fq_equal(P->x,elts[i],FF))
    {
      data->x = (uint16_t) i;
    }
    if (fq_equal(P->y,elts[i],FF))
    {
      data->y = (uint16_t) i;
    }
    if (fq_equal(P->z,elts[i],FF))
    {
      data->z = (uint16_t) i;
    }
  }

  uint16_t x = data->x;
  uint16_t y = data->y;
  uint16_t z = data->z;
  for (i=0; i<num_terms; i++)
  {
    (data->pt_vals)[i] = evaluate_monomial(terms+i,x,y,z,mul_table);
    (data->x_deriv)[i] = evaluate_derivative(terms+i,x,y,z,'x',mul_table);
    (data->y_deriv)[i] = evaluate_derivative(terms+i,x,y,z,'y',mul_table);
    (data->z_deriv)[i] = evaluate_derivative(terms+i,x,y,z,'z',mul_table);
  }  
}

/*----------------------------------------------------------------------------*/

void point_data_populate_all(point_data_t **data, point_t *pts, int num_pts,
			     fq_t *elts, fq_ctx_t FF, table_t *mul_table,
			     monomial_t *terms, int num_terms)
{
  *data = (point_data_t *) malloc(num_pts*sizeof(point_data_t));
  if (*data==NULL) MEMORY_ERROR;
  
  int i;
  for (i=0; i<num_pts; i++)
  {
    point_data_populate(*data+i,pts+i,elts,FF,mul_table,terms,num_terms);
  }

#if DEBUG
  fprintf(stdout,"\nPoint-to-Index Conversion:\n");
  int x,y,z;
  for (i=0; i<num_pts; i++)
  {
    x = (*data+i)->x;
    y = (*data+i)->y;
    z = (*data+i)->z;    
    point_pretty_print(pts+i,FF);
    fprintf(stdout," = ");    
    fprintf(stdout,"(%d, %d, %d)",x,y,z);
    fprintf(stdout,"\n");
  }
  fprintf(stdout,"\n");
#endif /* DEBUG */
}

/*----------------------------------------------------------------------------*/

// Private function to write table vector to file. Each vector is written
// as a space-separated sequence of indices followed by a newline. 
static void write_table_vector_to_file(FILE *fp, table_vec_t vv, int len)
{
  char tmp[8];  
  char s[1024];
  int j;
  sprintf(s,"%d",vv[0]);
  for (j=1; j<len; j++)
  {
    sprintf(tmp," %d",vv[j]);
    strcat(s,tmp);
    }
  strcat(s,"\n");
  fputs(s,fp);
}

void point_data_test_short(FILE *fp, point_data_t *data, int num_pts)
{
  // Number of polynomials
  int len = data->len;
  fprintf(fp,"%d\n",len);

  // points, represented as a space-separated triple of indices i j k
  point_data_t *P;
  int i;
  for (i=0; i<num_pts; i++)
  {
    P = data+i;
    fprintf(fp,"%d %d %d\n",P->x,P->y,P->z);
  }

  // For each point (in same order as above):
  // vector of polynomial evluations of point, as a space-separated tuple of indices
  // vector of x-derivatives of polynomial evaluations of point
  // vector of y-derivatives of polynomial evaluations of point
  // vector of z-derivatives of polynomial evaluations of point
  for (i=0; i<num_pts; i++)
  {
    P = data+i;
    write_table_vector_to_file(fp,P->pt_vals,len);
    write_table_vector_to_file(fp,P->x_deriv,len);
    write_table_vector_to_file(fp,P->y_deriv,len);
    write_table_vector_to_file(fp,P->z_deriv,len);    
  }
}


void point_data_test_full(FILE *fp, point_data_t *data, int num_pts,
			  fq_ctx_t FF, fq_t *elts)
{

  // Field degree
  int r = fq_ctx_degree(FF);
  fprintf(fp,"%d\n",r);

  // Field elements in order
  int i;
  int q = field_cardinality_to_int(FF);
  for (i=0; i<q; i++)
  {
    fprintf(fp,"%s\n",fq_get_str_pretty(elts[i],FF));
  }

  // Number of orbit representatives
  fprintf(fp,"%d\n",num_pts);

  point_data_test_short(fp,data,num_pts);
}

/*----------------------------------------------------------------------------*/

int point_is_smooth_on_curve(point_data_t *P, coef_vec_t pol, table_t *add_table)
{
  int len = P->len;
  
  // Does P lie on the curve?
  if (fq_sum_terms_table(pol,P->pt_vals,len,add_table) !=0) return 0;  

  // P is on the curve. Are any of the partial derivatives nonzero?
  if (fq_sum_terms_table(pol,P->x_deriv,len,add_table) !=0) return 1;
  if (fq_sum_terms_table(pol,P->y_deriv,len,add_table) !=0) return 1;
  if (fq_sum_terms_table(pol,P->z_deriv,len,add_table) !=0) return 1;

  // Now P lies on the curve, but it's a singular point. 
  return 0;
}

/*----------------------------------------------------------------------------*/
