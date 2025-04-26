/*----------------------------------------------------------------------------*/
/*-----------------  FIELD ARITHMETIC AND VECTOR FUNCTIONS  ------------------*/
/*----------------------------------------------------------------------------*/

#include "plane.h"

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

void field_elements_array(fq_t **elts, fq_ctx_t FF)
{

  int i, j, r, card, p, degree;
  fq_t x, power, gen;
  fq_init(x,FF);
  fq_init(power,FF);
  fq_init(gen,FF);

  // Get prime, degree, and cardinality of field extension
  p = fmpz_get_ui(fq_ctx_prime(FF));
  card = field_cardinality_to_int(FF);
  degree = fq_ctx_degree(FF);

#if DEBUG
  fprintf(stdout,"p = %d, q = %d, degree = %d\n",p,card,degree);
#endif /* DEBUG */
  
  // Allocate space for the field elements and init
  *elts = (fq_t*) malloc(card*sizeof(fq_t));
  fq_t* elts_ptr = *elts;  
  if (elts_ptr==NULL) MEMORY_ERROR;
  for (i=0; i<card; i++) fq_init(elts_ptr[i],FF);

  // Populate the array with finite field elements
  if (degree==1)
  {
    fq_one(gen,FF);
  }
  else
  {
    fq_gen(gen,FF);
  }
  for (i=0; i<card; i++)
  {
    j = i;
    fq_one(power,FF);
    // Write i = j_0 + j_1*p + j_2*p^2 + ...
    // and set elt[i] = j_0 + j_1*gen + j_2*gen^2 + ...
    while (j > 0)
    {
      r = j % p;
      fq_mul_ui(x,power,r,FF);
      fq_add(elts_ptr[i],elts_ptr[i],x,FF);
      fq_mul(power,power,gen,FF);
      j = (j - r) / p;
    }
  }

#if DEBUG
  fprintf(stdout,"Field elements: ");
  for (i=0; i<card; i++)
  {
    fprintf(stdout,"%s ",fq_get_str_pretty(elts_ptr[i],FF));
  }
  fprintf(stdout,"\n");
#endif /* DEBUG */

  // Verify that the first two elements are 0 and 1.
  if (!fq_is_zero(elts_ptr[0],FF))
  {
    char elt_str[64];
    fq_get_str_pretty(elts_ptr[0],FF);
    fprintf(stderr,"Element 0 in the array is %s, not 0\n",elt_str);
    exit(1);
  }
  if (!fq_is_one(elts_ptr[1],FF))
  {
    char elt_str[64];
    fq_get_str_pretty(elts_ptr[1],FF);
    fprintf(stderr,"Element 1 in the array is %s, not 1\n",elt_str);
    exit(1);
  }
    
  fq_clear(x,FF);
  fq_clear(power,FF);
  fq_clear(gen,FF);
}

/*----------------------------------------------------------------------------*/

void construct_field(fq_ctx_t FF, fq_t **elts, int r)
{
  fmpz_t Zq;
  fmpz_init(Zq);
  fmpz_set_ui(Zq,2);

  if (r == 1)
  {
    fq_ctx_init(FF,Zq,1,"t");
    field_elements_array(elts,FF);
  }
  else
  {
    if (!_fq_ctx_init_conway(FF,Zq,r,"t"))
    {
      fprintf(stderr,"No Conway polynomial for r = %d over GF(2)\n",r);
      exit(1);
    }
    field_elements_array(elts, FF);
  }  
  
  fmpz_clear(Zq);
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

void construct_op_table(table_t *table, fq_t *fld_elts, fq_ctx_t FF, int is_add)
{
  int i, j, k;
  int q = field_cardinality_to_int(FF);

  uint16_t *entries = (uint16_t *) malloc(q*q*sizeof(uint16_t));
  if (entries==NULL) MEMORY_ERROR;
  table->entries = entries;  
  table->width = q;

  fq_t tmp;
  fq_init(tmp,FF);
  for (i=0; i<q; i++)
  {
    for (j=0; j<q; j++)
    {
      if (is_add)
      {
	fq_add(tmp,*(fld_elts+i),*(fld_elts+j),FF);
      }
      else
      {
	fq_mul(tmp,*(fld_elts+i),*(fld_elts+j),FF);	
      }
      for (k=0; k<q; k++)
      {
	if (fq_equal(tmp,*(fld_elts+k),FF))
	{
	  *(entries + i + q*j) = (uint16_t) k;
	  break;
	}
      }
    }
  }
  fq_clear(tmp,FF);
}

/*----------------------------------------------------------------------------*/

void fq_print_table(table_t *table)
{
  int q = table->width;
  int i, j;
  uint16_t *entries = table->entries;
  for (i=0; i<q; i++)
  {
    for (j=0; j<q; j++)
    {
      fprintf(stdout,"%d ",*(entries+i+q*j));
    }
    fprintf(stdout,"\n");
  }    
}

/*----------------------------------------------------------------------------*/

uint16_t fq_op_table(uint16_t *op1, uint16_t *op2, table_t *table)
{
  int q = table->width;
  uint16_t *entries = table->entries;
  return *(entries + *op1 + (*op2)*q);
}

/*----------------------------------------------------------------------------*/

uint16_t fq_sum_terms_table(uint64_t mask, uint16_t *inds, int len, table_t *add_table)
{
  // Sum only the terms specified by the bits in mask
  uint16_t rop = 0;
  int i,j = 0;
  while (mask)
  {
    i = __builtin_ctzll(mask);
    rop = fq_op_table(&rop,inds+i+j,add_table);
    j += (i+1);
    mask >>= (i+1);
  }
  return rop;
}

/*----------------------------------------------------------------------------*/

void table_vec_init(table_vec_t *vec, int num_entries)
{
  *vec = (table_vec_t) malloc(num_entries*sizeof(uint16_t));
  if (*vec == NULL) MEMORY_ERROR;
}

