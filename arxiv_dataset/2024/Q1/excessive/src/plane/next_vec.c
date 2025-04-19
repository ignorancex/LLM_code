/*----------------------------------------------------------------------------*/
/*--------------------------  BIT VECTOR ITERATORS  --------------------------*/
/*----------------------------------------------------------------------------*/

#include "plane.h"

int next_lex_vector_of_same_weight_with_offset(uint64_t *vv, int len,
					       int offset)
{
  if (len == 0) return 0;
  
  // Extract the part at desired offset.
  uint64_t x = *vv;
  x >>= offset; 
  uint64_t mask = (1UL<<len) - 1;
  x = (x & mask);

  // Vector of all 0's ==> no next vector of same weight
  if (x == 0) return 0;

  // Vector of all 1's then 0's  ==> no next vector of same weight
  if ( ( (x|(x-1))+1 == 1UL<<len) ) return 0;

  // Zero out the part we're updating
  *vv ^= x << offset;

  // Update the desired part
  uint64_t s, r;
  s = x & (-x);
  r = s + x;
  x = r | ( (x ^ r) >> (2 + __builtin_ctz(s)) );
  
  // xor the updated part back in
  *vv ^= x << offset;  
  return 1;
}

/*----------------------------------------------------------------------------*/

int next_odd_weight_vector_with_offset(uint64_t *vv, int len, int offset)
{
  if (next_lex_vector_of_same_weight_with_offset(vv,len,offset)) return 1;
  if (len <= 0) return 0;

  // Extract the part at desired offset.
  uint64_t x = *vv;
  x >>= offset; 
  uint64_t mask = (1UL<<len) - 1;
  x = (x & mask);

  // x should be the vector of all 1's then all 0's
  if ( ( (x|(x-1))+1 != 1UL<<len) )
  {
    fprintf(stderr,"ERROR: vv = %llu, len = %d, offset = %d\n",(unsigned long long)*vv,len,offset);
    fprintf(stderr,"ERROR: substring is not of the form 1...10...0\n");
    exit(1);
  }

  // Determine the weight and increase by 2 if possible.
  int wt = len - __builtin_ctz(x);
  if (wt >= len-1) return 0;
  wt += 2;
  *vv ^= x << offset;  // Zero out the part we're about to update
  x = (1UL<<wt)-1;
  *vv ^= x << offset;  // xor the updated part back in
  return 1;
}

/*----------------------------------------------------------------------------*/

void print_bits(uint64_t n, int min_bits)
{
  int print = 0;
  int i;
  min_bits = (min_bits < 64) ? min_bits : 64;
  for (i=63; i>=0; i--)
  {
    if (i+1 == min_bits) print = 1;
    if ((n>>i) & 1)
    {
      fprintf(stdout,"1");
      print = 1;
    }
    else if (print)
    {
      fprintf(stdout,"0");
    }
  }
  if (!print)
  {
    fprintf(stdout,"0");
  }
  fprintf(stdout,"\n");
}

/*----------------------------------------------------------------------------*/

void set_first_odd_weight_vector_with_offset(uint64_t *vv, int len, int offset)
{
  uint64_t mask = (1UL<<len) - 1; 
  uint64_t x = (*vv>>offset) & mask; // Current contents of active field
  // *vv ^= (x << offset);           // Zero out the active field
  // *vv ^= (1UL << offset);         // Set active field to 0...01
  *vv ^= (x^1UL) << offset;          // Combines previous two statements
}

/*----------------------------------------------------------------------------*/

void test_next_odd_weight_vector_with_offset(int total_len, int len,
					     int offset)
{
  // RAND_MAX = 0x7fffffff = 2^{31} - 1
  if (RAND_MAX < 0x7fffffff)
  {
    fprintf(stderr,"Unexpectedly small value of RAND_MAX: %ld\n", (long) RAND_MAX);
    exit(1);
  }
  if (total_len > 31)
  {
    fprintf(stderr,"Test code only guaranteed to work up to total length 31\n");
    exit(1);
  }

  // Initialize vv
  srand(time(0)); // Random seed initializer
  uint64_t mask = (1UL<<total_len) - 1; 
  uint64_t vv = rand() & mask;      // random bit string of length total_len
  set_first_odd_weight_vector_with_offset(&vv,len,offset);

  while (1)
  {
    print_bits(vv,total_len);
    if (!next_odd_weight_vector_with_offset(&vv,len,offset)) break;
  }
}

/*----------------------------------------------------------------------------*/

int odd_weight_vector_with_offset_and_index(uint64_t *vv, int len,
					    int offset, uint64_t index)
{
  if (len <= 0) return 0;
  if (index < 0) return 0;

  // Get vector with given index first
  uint64_t x = 1;
  uint64_t j;
  int success = 1;
  for (j=0; j<index; j++)
  {
    success = next_odd_weight_vector_with_offset(&x,len,0);
  }
  if (!success) return 0;

  // Swap in the new vector at the correct offset
  uint64_t mask = (1UL<<len) - 1; 
  uint64_t field = ((*vv>>offset) & mask); // Current contents of active field
  // *vv ^= (field << offset); // Zero out the active field
  // *vv ^= (x << offset); // Set active field to x
  *vv ^= (field^x)<<offset; // Combines previous two statements (xor is associative)
  return 1;
}

/*----------------------------------------------------------------------------*/
