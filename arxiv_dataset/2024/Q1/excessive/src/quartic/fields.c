/*----------------------------------------------------------------------------*/
/*                   FIELD ARITHMETIC AND VECTOR FUNCTIONS                    */
/*----------------------------------------------------------------------------*/

#include "quartic.h"

/*----------------------------------------------------------------------------*/

int field_cardinality_to_int(fq_ctx_t FF)
{
  int p = fmpz_get_si(fq_ctx_prime(FF));
  int q = 1;
  int d = fq_ctx_degree(FF);
  int i;
  for (i=0; i<d; i++) q *= p;
  return q;
}

int get_q(fq_ctx_t GFq_squared)
{
  int p = fmpz_get_si(fq_ctx_prime(GFq_squared));
  int d = fq_ctx_degree(GFq_squared);
  if (d % 2 != 0)
  {
    fprintf(stderr,"ERROR: GFq_squared has odd degree %d over GF(%d)\n",d,p);
    exit(1);
  }
  int i;
  int q = 1;
  for (i=0; i<d/2; i++) q *= p;
  return q;
}

/*----------------------------------------------------------------------------*/

void quadratic_frobenius(fq_t rop, fq_t op, fq_ctx_t GFq_squared)
{
  long d = fq_ctx_degree(GFq_squared);  // q^2 = p^d, so d must be even.
  if (d % 2 != 0)
  {
    long p = fmpz_get_ui(fq_ctx_prime(GFq_squared));  
    fprintf(stderr,"ERROR: GFq_squared has odd degree %ld over GF(%ld)\n",d,p);
    exit(1);
  }
  fq_frobenius(rop,op,d/2,GFq_squared);
}

/*----------------------------------------------------------------------------*/

int fq_is_rational(fq_t op, fq_ctx_t GFq_squared)
{
  int rv;
  fq_t tmp;
  fq_init(tmp,GFq_squared);
  quadratic_frobenius(tmp,op,GFq_squared);
  if (fq_equal(tmp,op,GFq_squared))
  {
    rv = 1;
  }
  else
  {
    rv = 0;
  }
  fq_clear(tmp,GFq_squared);
  return rv;
}

/*----------------------------------------------------------------------------*/  

void field_elements_array(fq_t **elts, fq_ctx_t GFq_squared)
{

  int i, j, r, q, p;
  fq_t tmp, x, power, gen;
  fq_init(tmp,GFq_squared);
  fq_init(x,GFq_squared);  
  fq_init(power,GFq_squared);
  fq_init(gen,GFq_squared);

  // Get prime, degree, and cardinality of field extension
  p = fmpz_get_ui(fq_ctx_prime(GFq_squared));
  q = get_q(GFq_squared); // Throws an error if field cardinality is not a square.

#if DEBUG
  int degree = fq_ctx_degree(GFq_squared);
  fprintf(stdout,"Field elements array: p = %d, q = %d, degree = %d\n",p,q,degree/2);
#endif /* DEBUG */
  
  // Allocate space for the field elements and init
  *elts = (fq_t*) malloc(q*q*sizeof(fq_t));
  fq_t* elts_ptr = *elts;  
  if (elts_ptr==NULL) MEMORY_ERROR;
  for (i=0; i<q*q; i++) fq_init(elts_ptr[i],GFq_squared);

  // Populate the array with finite field elements
  fq_gen(gen,GFq_squared);
  int rat_idx = 0; 
  int quad_idx = q; // First q elements; start quadratic ones at index q
  for (i=0; i<q*q; i++)
  {
    j = i;
    fq_one(power,GFq_squared);
    fq_zero(x,GFq_squared);
    // Write i = j_0 + j_1*p + j_2*p^2 + ...
    // and set elt[i] = j_0 + j_1*gen + j_2*gen^2 + ...
    while (j > 0)
    {
      r = j % p;
      fq_mul_ui(tmp,power,r,GFq_squared);
      fq_add(x,x,tmp,GFq_squared);
      fq_mul(power,power,gen,GFq_squared);
      j = (j - r) / p;
    }
    if (fq_is_rational(x,GFq_squared))
    {
      fq_set(elts_ptr[rat_idx],x,GFq_squared);
      rat_idx++;
    }
    else
    {
      fq_set(elts_ptr[quad_idx],x,GFq_squared);
      quad_idx++;
    }
  }

  if (rat_idx - q + quad_idx != q*q)
  {
    fprintf(stderr,"ERROR: Recorded %d field elements\n",rat_idx - q + quad_idx);
  }

#if DEBUG
  fprintf(stdout,"Field elements: ");
  for (i=0; i<q*q; i++)
  {
    fprintf(stdout,"%s ",fq_get_str_pretty(elts_ptr[i],GFq_squared));
  }
  fprintf(stdout,"\n");
#endif /* DEBUG */

  // Verify that the first p elements are 0, 1, ..., p-1
  for (i=0; i<p; i++)
  {
    fq_set_si(x,i,GFq_squared);
    if (!fq_equal(x,elts_ptr[i],GFq_squared))
    {
      char elt_str[64];
      strcpy(elt_str,fq_get_str_pretty(elts_ptr[i],GFq_squared));
      fprintf(stderr,"Element %d in the array is %s, not %d\n",i,elt_str,i);
    }
  }

  fq_clear(tmp,GFq_squared);  
  fq_clear(x,GFq_squared);
  fq_clear(power,GFq_squared);
  fq_clear(gen,GFq_squared);
}

/*----------------------------------------------------------------------------*/

void construct_quadratic_extension(fq_ctx_t GFq_squared, fq_t **elts, int q)
{
  int p, r, done;
  done = 0;
  for (p=2; p<=q; p++)
  {
    for (r=1; r<=5; r++)
    {
      if ((int) pow(p,r) == q)
      {
	done=1;
	break;
      }
    }
    if (done) break;
  }
  if (!done)
  {
    fprintf(stderr, "ERROR: Not able to write q = %d = p^r with p < 32 and r < 6\n",q);
    exit(1);
  }
  
  // Construct finite field and array of elements
  fmpz_t Zp;
  fmpz_init(Zp);
  fmpz_set_ui(Zp,p);
  if (!fmpz_is_prime(Zp))
  {
    fprintf(stderr, "ERROR: q = %d is not a prime power\n",q);
    exit(1);
  }
  if (!_fq_ctx_init_conway(GFq_squared,Zp,2*r,"t"))
  {
    fprintf(stderr,"ERROR: No Conway polynomial for degree = %d over GF(%d)\n",2*r,p);
    exit(1);
  }
  field_elements_array(elts,GFq_squared);
  fmpz_clear(Zp);
}

void clear_field(fq_ctx_t FF, fq_t *elts)
{
  int i;
  int q = field_cardinality_to_int(FF);  
  for (i=0; i<q; i++) fq_clear(elts[i],FF);
  free(elts);
  fq_ctx_clear(FF);
}

/*----------------------------------------------------------------------------*/

void irreducible_quadratic(fq_t a, fq_t *elts, fq_ctx_t GFq_squared)
{
  int q = get_q(GFq_squared);
  int i, j, has_root;
  fq_t rop;
  fq_init(rop,GFq_squared);
  for (i=0; i<q; i++)
  {
    fq_set(a,elts[i],GFq_squared);
    // Check if x^2 + ax + 1 has a root
    has_root = 0;
    for (j=0; j<q; j++)
    {
      fq_add(rop,elts[j],a,GFq_squared);   //  x + a
      fq_mul(rop,rop,elts[j],GFq_squared); //  x(x+a)
      fq_neg(rop,rop,GFq_squared);           // -x(x+a)
      if (fq_is_one(rop,GFq_squared))        // -x(x+a) = 1 ?
      {
	has_root = 1;
	break;
      }
    }
    if (!has_root) break;
  }
  fq_clear(rop,GFq_squared);
}

/*----------------------------------------------------------------------------*/

uint16_t field_element_to_index(fq_t elt, fq_t *fld_elts, fq_ctx_t FF)
{
  int j;
  int q = field_cardinality_to_int(FF);
  fq_reduce(elt,FF); // For safety: fq_inv does not canonicalize its output (!?!)
  for (j=0; j<q; j++)
  {
    if (fq_equal(elt,fld_elts[j],FF)) return j;
  }
  fprintf(stdout,"%s does not appear to have an index\n",fq_get_str_pretty(elt,FF));
  exit(1);
}
  
/*----------------------------------------------------------------------------*/

void construct_ops_store(ops_store_t *tables, fq_t a,
			 fq_t *fld_elts, fq_ctx_t GFq_squared)
{
  int i, j, p, q;
  uint16_t *entries;
  q = get_q(GFq_squared);
  p = fmpz_get_ui(fq_ctx_prime(GFq_squared));  
  tables->q = q;
  tables->qq = q*q;
  tables->p = p;
  tables->a = field_element_to_index(a,fld_elts,GFq_squared);  

  fq_t tmp;
  fq_init(tmp,GFq_squared);

  // Root of T^2 + aT + 1
  for (i=q; i<q*q; i++) // Start at q to skip over GF(q) elements
  {
    fq_add(tmp,fld_elts[i],a,GFq_squared);    // T+a
    fq_mul(tmp,fld_elts[i],tmp,GFq_squared);  // T(T+a)
    fq_neg(tmp,tmp,GFq_squared);              // -T(T+a)
    fq_sub_one(tmp,tmp,GFq_squared);          // -T(T+a) - 1
    fq_reduce(tmp,GFq_squared); // Flint does not reduce for equality checking
    if (!fq_is_zero(tmp,GFq_squared)) continue;
    tables->root = i;
    break;
  }
  
  // Addition table
  entries = (uint16_t *) malloc(q*q*q*q*sizeof(uint16_t));
  if (entries==NULL) MEMORY_ERROR;
  tables->add_table = entries;
  for (i=0; i<q*q; i++)
  {
    for (j=0; j<q*q; j++)
    {
      fq_add(tmp,fld_elts[i],fld_elts[j],GFq_squared);
      entries[i+q*q*j] = field_element_to_index(tmp,fld_elts,GFq_squared);
    }
  }

  // Subtraction table
  entries = (uint16_t *) malloc(q*q*q*q*sizeof(uint16_t));
  if (entries==NULL) MEMORY_ERROR;
  tables->sub_table = entries;
  for (i=0; i<q*q; i++)
  {
    for (j=0; j<q*q; j++)
    {
      fq_sub(tmp,fld_elts[i],fld_elts[j],GFq_squared);
      entries[i+q*q*j] = field_element_to_index(tmp,fld_elts,GFq_squared);
    }
  }
  
  // Multiplication table
  entries = (uint16_t *) malloc(q*q*q*q*sizeof(uint16_t));
  if (entries==NULL) MEMORY_ERROR;
  tables->mul_table = entries;
  for (i=0; i<q*q; i++)
  {
    for (j=0; j<q*q; j++)
    {
      fq_mul(tmp,fld_elts[i],fld_elts[j],GFq_squared);	
      entries[i+q*q*j] = field_element_to_index(tmp,fld_elts,GFq_squared);
    }
  }

  // Conjugates array (image of quadratic Frobenius)
  entries = (uint16_t *) malloc(q*q*sizeof(uint16_t));
  if (entries==NULL) MEMORY_ERROR;
  tables->conjs = entries;
  for (i=0; i<q*q; i++)
  {
    quadratic_frobenius(tmp,fld_elts[i],GFq_squared);
    entries[i] = field_element_to_index(tmp,fld_elts,GFq_squared);
  }

  // Inverse array
  entries = (uint16_t *) malloc(q*q*sizeof(uint16_t));
  if (entries==NULL) MEMORY_ERROR;
  tables->invs = entries;
  *entries = 0; // Store 0 at 0-th position
  for (i=1; i<q*q; i++)
  {
    fq_inv(tmp,fld_elts[i],GFq_squared);
    entries[i] = field_element_to_index(tmp,fld_elts,GFq_squared);
  }
  
  // Squares, Cubes, Quads arrays
  tables->squares = (uint16_t *) malloc(q*q*sizeof(uint16_t));
  tables->cubes   = (uint16_t *) malloc(q*q*sizeof(uint16_t));
  tables->quads   = (uint16_t *) malloc(q*q*sizeof(uint16_t));  
  if (tables->squares==NULL) MEMORY_ERROR;
  if (tables->cubes==NULL)   MEMORY_ERROR;
  if (tables->quads==NULL)   MEMORY_ERROR;  
  for (i=0; i<q*q; i++)
  {
    fq_mul(tmp,*(fld_elts+i),*(fld_elts+i),GFq_squared);	
    tables->squares[i] = field_element_to_index(tmp,fld_elts,GFq_squared);
    fq_mul(tmp,tmp,*(fld_elts+i),GFq_squared);
    tables->cubes[i] = field_element_to_index(tmp,fld_elts,GFq_squared);
    fq_mul(tmp,tmp,*(fld_elts+i),GFq_squared);
    tables->quads[i]= field_element_to_index(tmp,fld_elts,GFq_squared);
  }
  
  fq_clear(tmp,GFq_squared);  
}

void clear_ops_store(ops_store_t *tables)
{
  free(tables->add_table);
  free(tables->sub_table);  
  free(tables->mul_table);
  free(tables->conjs);  
  free(tables->invs);
  free(tables->squares);
  free(tables->cubes);
  free(tables->quads);    
}

/*----------------------------------------------------------------------------*/

void print_ops_store(ops_store_t *tables)
{
  int q = tables->q;
  int qq = tables->qq;
  int i, j;

  fprintf(stdout,"\nq = %d, p = %d, a = %d\n",q,tables->p,tables->a);
  fprintf(stdout,"Root of T^2 + aT + 1: %d\n",tables->root);

  fprintf(stdout,"\nAddition table:\n");
  uint16_t *entries = tables->add_table;
  for (i=0; i<qq; i++)
  {
    for (j=0; j<qq; j++)
    {
      fprintf(stdout,"%2d ",entries[i+qq*j]);      
    }
    fprintf(stdout,"\n");
  }    

  fprintf(stdout,"\nSubtraction table:\n");
  entries = tables->sub_table;
  for (i=0; i<qq; i++)
  {
    for (j=0; j<qq; j++)
    {
      fprintf(stdout,"%2d ",entries[i+qq*j]);      
    }
    fprintf(stdout,"\n");
  }    
  
  fprintf(stdout,"\nMultiplication table:\n");
  entries = tables->mul_table;
  for (i=0; i<qq; i++)
  {
    for (j=0; j<qq; j++)
    {
      fprintf(stdout,"%2d ",entries[i+qq*j]);
    }
    fprintf(stdout,"\n");
  }    

  fprintf(stdout,"\nConjugates Array: ");
  entries = tables->conjs;
  for (i=0; i<qq; i++)
  {
    fprintf(stdout,"%2d ",entries[i]);
  }    

  fprintf(stdout,"\n  Inverses Array: ");
  entries = tables->invs;
  for (i=0; i<qq; i++)
  {
    fprintf(stdout,"%2d ",entries[i]);
  }    

  fprintf(stdout,"\n   Squares Array: ");
  entries = tables->squares;
  for (i=0; i<qq; i++)
  {
    fprintf(stdout,"%2d ",entries[i]);
  }    

  fprintf(stdout,"\n     Cubes Array: ");
  entries = tables->cubes;
  for (i=0; i<qq; i++)
  {
    fprintf(stdout,"%2d ",entries[i]);    
  }    

  fprintf(stdout,"\n     Quads Array: ");
  entries = tables->quads;
  for (i=0; i<qq; i++)
  {
    fprintf(stdout,"%2d ",entries[i]);
  }
  fprintf(stdout,"\n\n");
}

/*----------------------------------------------------------------------------*/

// FIXME: Macro-ify these! See if it speeds things up. 
uint16_t fq_idx_add(uint16_t *op1, uint16_t *op2, ops_store_t *tables)
{
  return (tables->add_table)[*op1 + (*op2)*(tables->qq)];
}

uint16_t fq_idx_sub(uint16_t *op1, uint16_t *op2, ops_store_t *tables)
{
  return (tables->sub_table)[*op1 + (*op2)*(tables->qq)];
}

uint16_t fq_idx_mul(uint16_t *op1, uint16_t *op2, ops_store_t *tables)
{
  return (tables->mul_table)[*op1 + (*op2)*(tables->qq)];  
}

uint16_t fq_idx_mul_ui(uint16_t *op1, int16_t op2, ops_store_t *tables)
{
  return (tables->mul_table)[*op1 + (op2 % (tables->p))*(tables->qq)];  
}

uint16_t fq_idx_conj(uint16_t *op, ops_store_t *tables)
{
  return (tables->conjs)[*op];
}

uint16_t fq_idx_inv(uint16_t *op, ops_store_t *tables)
{
  return (tables->invs)[*op];
}

uint16_t fq_idx_square(uint16_t *op, ops_store_t *tables)
{
  return (tables->squares)[*op];
}

uint16_t fq_idx_cube(uint16_t *op, ops_store_t *tables)
{
  return (tables->cubes)[*op];
}

uint16_t fq_idx_quad(uint16_t *op, ops_store_t *tables)
{
  return (tables->quads)[*op];
}
  
/*----------------------------------------------------------------------------*/

void fq_idx_vec_init(fq_idx_vec_t *vec, int num_entries)
{
  *vec = (fq_idx_vec_t) malloc(num_entries*sizeof(uint16_t));
  if (*vec == NULL) MEMORY_ERROR;
}

/*----------------------------------------------------------------------------*/

uint16_t fq_idx_vec_dot(fq_idx_vec_t op1, fq_idx_vec_t op2, int len,
			ops_store_t *tables)
{
  uint16_t rop = 0;
  uint16_t tmp;
  int i;
  for (i=0; i<len; i++)
  {
    if ((*(op1+i) == 0) || (*(op2+i) == 0)) continue;
    tmp = fq_idx_mul(op1+i,op2+i,tables);
    rop = fq_idx_add(&rop,&tmp,tables);
  }
  return rop;
}

/*----------------------------------------------------------------------------*/

void write_index_vector_to_file(FILE *fp, fq_idx_vec_t vv, int len)
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

void write_index_vector_to_stdout(fq_idx_vec_t vv, int len)
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
  fputs(s,stdout);
}

