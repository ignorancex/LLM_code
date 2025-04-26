/*----------------------------------------------------------------------------*/
/*		      Genus-4 Excessive Curve Search Code                     */
/*----------------------------------------------------------------------------*/

#include "genus4.h"

/*----------------------------------------------------------------------------*/

void field_elements_array(fq_t **elts, fq_ctx_t GFq)
{

  long i, j, r, card, p, degree;
  fmpz_t tmp;
  fq_t x, power, gen;
  fmpz_init(tmp);
  fq_init(x,GFq);
  fq_init(power,GFq);
  fq_init(gen,GFq);

  // Get prime, degree, and cardinality of field extension
  p = fmpz_get_ui(fq_ctx_prime(GFq));
  fq_ctx_order(tmp,GFq);
  card = fmpz_get_ui(tmp);
  degree = fq_ctx_degree(GFq);

  // fprintf(stdout,"p = %ld, q = %ld, degree = %ld\n",p,card,degree);

  // Allocate space for the field elements and init
  *elts = (fq_t*) malloc(card*sizeof(fq_t));
  fq_t* elts_ptr = *elts;  
  if (elts_ptr==NULL) MEMORY_ERROR;
  for (i=0; i<card; i++) fq_init(elts_ptr[i],GFq);

  // Populate the array with finite field elements
  if (degree==1)
  {
    fq_one(gen,GFq);
  }
  else
  {
    fq_gen(gen,GFq);
  }
  for (i=0; i<card; i++)
  {
    j = i;
    fq_one(power,GFq);
    // Write i = j_0 + j_1*p + j_2*p^2 + ...
    // and set elt[i] = j_0 + j_1*gen + j_2*gen^2 + ...    
    while (j > 0)
    {
      r = j % p;
      fq_mul_ui(x,power,r,GFq);
      fq_add(elts_ptr[i],elts_ptr[i],x,GFq);
      fq_mul(power,power,gen,GFq);
      j = (j - r) / p;
    }
  }

  // DEBUG
  // fprintf(stdout,"Field elements: ");
  // for (i=0; i<card; i++)
  // {
  //   fprintf(stdout,"%s ",fq_get_str_pretty(elts_ptr[i],GFq));
  // }
  // fprintf(stdout,"\n");
  // END DEBUG
    
  fq_clear(x,GFq);
  fq_clear(power,GFq);
  fq_clear(gen,GFq);
  fmpz_clear(tmp);
}

/*----------------------------------------------------------------------------*/

// Make a list of the elements in GFq that are fixed by x -> x^(p^d),
// where q = p^degree. (allocate memory too)
void subfield_elements_array(fq_t **elts, fq_ctx_t GFq, long d)
{
  long i, j, k, r, card, p, degree, subfld_card;
  fmpz_t tmp;
  fq_t x, power, gen;
  fmpz_init(tmp);
  fq_init(x,GFq);
  fq_init(power,GFq);
  fq_init(gen,GFq);

  // Get prime, degree, and cardinality of field extension
  p = fmpz_get_ui(fq_ctx_prime(GFq));
  fq_ctx_order(tmp,GFq);
  card = fmpz_get_ui(tmp);
  degree = fq_ctx_degree(GFq);

  if (degree % d != 0)
  {
    fprintf(stderr,"d = %ld does not describe a subfield\n",d);
    exit(1);
  }

  // Allocate space for the field elements and init
  subfld_card = 1;
  for (i=0; i<d; i++) subfld_card *= p;    
  *elts = (fq_t*) malloc(subfld_card*sizeof(fq_t));
  fq_t* elts_ptr = *elts;
  if (elts_ptr==NULL) MEMORY_ERROR;
  for (i=0; i<subfld_card; i++) fq_init(elts_ptr[i],GFq);

  // Populate the array with finite field elements, indexed by k
  if (degree==1)
  {
    fq_one(gen,GFq);
  }
  else
  {
    fq_gen(gen,GFq);
  }
  k = 0;  
  for (i=0; i<card; i++)
  {
    j = i;
    fq_one(power,GFq);
    fq_zero(elts_ptr[k],GFq);
    while (j > 0)
    {
      r = j % p;
      fq_mul_ui(x,power,r,GFq);
      fq_add(elts_ptr[k],elts_ptr[k],x,GFq);
      fq_mul(power,power,gen,GFq);
      j = (j - r) / p;
    }
    fq_frobenius(x,elts_ptr[k],d,GFq);
    
    if (fq_equal(x,elts_ptr[k],GFq)) k++;
    if (k == subfld_card) break;
  }
  // DEBUG
  // fprintf(stdout,"Subfield elements: ");
  // for (i=0; i<subfld_card; i++)
  // {
  //   fprintf(stdout,"%s ",fq_get_str_pretty(elts_ptr[i],GFq));
  // }
  // fprintf(stdout,"\n");
  // END DEBUG
    
  fq_clear(x,GFq);
  fq_clear(power,GFq);
  fq_clear(gen,GFq);
  fmpz_clear(tmp);
}

/*----------------------------------------------------------------------------*/

void construct_fields(long q, fq_ctx_t GFq_squared, fq_t **quad_elts)
{
  fmpz_t Zq;
  fmpz_init(Zq);

  fq_t *rat_elts;

  if (q == 2)
  {
    fmpz_set_ui(Zq,q);
    if (!_fq_ctx_init_conway(GFq_squared,Zq,2,"t")) // Conway polynomial: t^2 + t + 1
    {
      fprintf(stderr,"No Conway polynomial for q = %ld, d = 2\n",q);
      exit(1);
    }
    field_elements_array(quad_elts, GFq_squared);
    subfield_elements_array(&rat_elts,GFq_squared,1);
  }
  else if (q == 3)
  {
    fmpz_set_ui(Zq,q);
    if (!_fq_ctx_init_conway(GFq_squared,Zq,2,"t")) // Conway polynomial: t^3 + 2t + 2
    {
      fprintf(stderr,"No Conway polynomial for q = %ld, d = 2\n",q);
      exit(1);
    }
    field_elements_array(quad_elts, GFq_squared);
    subfield_elements_array(&rat_elts,GFq_squared,1);
  }
  else if (q == 4)
  {
    fmpz_set_ui(Zq,2);
    if (!_fq_ctx_init_conway(GFq_squared,Zq,4,"T")) // Conway polynomial: T^4 + T + 1
    {
      fprintf(stderr,"No Conway polynomial for q = %ld, d = 4\n",q);
      exit(1);
    }
    field_elements_array(quad_elts, GFq_squared);
    subfield_elements_array(&rat_elts,GFq_squared,2);
  }
  else
  {
    fprintf(stderr,"ERROR: No support for q = %ld\n",q);
    exit(1);
  }

  // Now move the elements of GF(q) to the beginning of quad_elts
  fq_t tmp[q*q];
  long seen;
  long i, j, k;
  fq_t *quad_ptr = *quad_elts; // Alias
  for (i=0; i<q*q; i++)
  {
    fq_init(tmp[i],GFq_squared);
    if (i < q)
    {
      fq_set(tmp[i],*(rat_elts+i),GFq_squared);
      continue;
    }
    // Now i >= q. Walk quad_elts to find the next one we haven't seen.
    for (j=0; j<q*q; j++)
    {
      seen = 0;      
      for (k=0; k<i; k++)
      {
	if (fq_equal(tmp[k],*(quad_ptr+j),GFq_squared))
	{
	  seen = 1;
	  break;
	}
      }
      if (!seen)
      {
	fq_set(tmp[i],*(quad_ptr+j),GFq_squared);
	break;
      }
    }
  }
  for (i=0; i<q*q; i++) fq_set(*(quad_ptr+i),tmp[i],GFq_squared);

  // DEBUG
  // fprintf(stdout,"Reordered field elements: ");
  // for (i=0; i<q*q; i++)
  // {
  //   fprintf(stdout,"%s ",fq_get_str_pretty(*(quad_ptr+i),GFq_squared));
  // }
  // fprintf(stdout,"\n");
  // END DEBUG
  
  
  fmpz_clear(Zq);
  for (i=0; i<q; i++) fq_clear(*(rat_elts+i),GFq_squared);
  for (i=0; i<q; i++) fq_clear(tmp[i],GFq_squared);  
}

/*----------------------------------------------------------------------------*/

void construct_add_table(int8_t **add_table, fq_t *fld_elts, fq_ctx_t GFq)
{
  long p, q, d, i, j, k;
  p = fmpz_get_si(fq_ctx_prime(GFq));
  q = 1;
  d = fq_ctx_degree(GFq);
  for (i=0; i<d; i++) q *= p;

  *add_table = (int8_t*) malloc(q*q*sizeof(int8_t));

  fq_t tmp;
  fq_init(tmp,GFq);
  for (i=0; i<q; i++)
  {
    for (j=0; j<q; j++)
    {
      fq_add(tmp,*(fld_elts+i),*(fld_elts+j),GFq);
      for (k=0; k<q; k++)
      {
	if (fq_equal(tmp,*(fld_elts+k),GFq))
	{
	  *(*add_table + i + q*j) = k;
	  break;
	}
      }
    }
  }
  fq_clear(tmp, GFq);
}

/*----------------------------------------------------------------------------*/

int8_t fq_add_table(int8_t *op1, int8_t *op2, int8_t *add_table, long q)
{
  return *(add_table + *op1 + (*op2)*q);
}

/*----------------------------------------------------------------------------*/

void construct_mul_table(int8_t **mul_table, fq_t *fld_elts, fq_ctx_t GFq)
{
  long p, q, d, i, j, k;
  p = fmpz_get_si(fq_ctx_prime(GFq));
  q = 1;
  d = fq_ctx_degree(GFq);
  for (i=0; i<d; i++) q *= p;

  *mul_table = (int8_t*) malloc(q*q*sizeof(int8_t));

  fq_t tmp;
  fq_init(tmp,GFq);
  for (i=0; i<q; i++)
  {
    for (j=0; j<q; j++)
    {
      fq_mul(tmp,*(fld_elts+i),*(fld_elts+j),GFq);
      for (k=0; k<q; k++)
      {
	if (fq_equal(tmp,*(fld_elts+k),GFq))
	{
	  *(*mul_table + i + q*j) = k;
	  break;
	}
      }
    }
  }
  fq_clear(tmp,GFq);
}

/*----------------------------------------------------------------------------*/

int8_t fq_mul_table(int8_t *op1, int8_t *op2, int8_t *mul_table, long q)
{
  return *(mul_table + *op1 + (*op2)*q);
}

/*----------------------------------------------------------------------------*/

int8_t fq_vec_dot_table(int8_t *op1, int8_t *op2, long len,
		      int8_t *add_table, int8_t *mul_table, long q)
{
  int8_t rop = fq_mul_table(op1,op2,mul_table,q);
  long i;
  int8_t tmp;
  for (i=1; i<len; i++)
  {
    if ((*(op1+i) == 0) || (*(op2+i) == 0)) continue;
    tmp = fq_mul_table(op1+i,op2+i,mul_table,q);
    rop = fq_add_table(&rop,&tmp,add_table,q);
  }
  return rop;
}
