/* bit_ops.c */

#include <stdlib.h> /* malloc */
#include <bit_ops.h>



/* number of leading zeros and number of trailing zeros */
#if defined __GNUC__
/* using built-in gcc functions*/
#  define bo_nlz_32(x) (uint32_t)__builtin_clz(x)
#  define bo_ntz_32(x) (uint32_t)__builtin_ctz(x)
#else
/* using table lookup methods */
static inline uint32_t bo_nlz_32(uint32_t word);
static inline uint32_t bo_ntz_32(uint32_t word);
#endif



/* number of leading zeros */
static inline uint32_t bo_nlz_64(uint64_t word);
/* number of trailing zeros */
static inline uint32_t bo_ntz_64(uint64_t word);



/* most significant bit set or BO_NULL_BIT_POS if input is 0 */
inline uint32_t bo_msb(uint32_t word) {
  return (word == 0) ? BO_NULL_BIT_POS : BO_WORD_SIZE - 1 - bo_nlz_32(word);
}

/* least significant bit set or BO_NULL_BIT_POS if word is 0 */
inline uint32_t bo_lsb(uint32_t word) {
  return (word == 0) ? BO_NULL_BIT_POS : bo_ntz_32(word);
}

/* nearest bit set to 1 that is to the left of given bit or 
 * BO_NULL_BIT_POS if there is no such set bit */
inline uint32_t bo_nbl(uint32_t word, uint32_t index) {
  if (index == BO_WORD_SIZE - 1) {
    return BO_NULL_BIT_POS;
  }
  uint32_t mask = (uint32_t) ~0 << (index + 1);
  return bo_lsb(word & mask);
}

/* nearest bit set to 1 that is to the right of given bit or
 * BO_NULL_BIT_POS if there is no such set bit */
inline uint32_t bo_nbr(uint32_t word, uint32_t index) {
  if (index == 0) {
    return BO_NULL_BIT_POS;
  }
  uint32_t mask = (uint32_t) ~0 >> (BO_WORD_SIZE - index);
  return bo_msb(word & mask);
}

/* ceiling of the log (base 2) of given integer */
inline uint32_t bo_ceil_log(uint32_t word) {
  return BO_WORD_SIZE - bo_nlz_32(word - 1);
}

/* number of significant bits = msb + 1 */
inline uint32_t bo_bits_in_32(uint32_t word) {
  return BO_WORD_SIZE - bo_nlz_32(word);
}

inline uint32_t bo_bits_in_64(uint64_t word) {
  return 2 * BO_WORD_SIZE - bo_nlz_64(word);
}

/* bit value at bit position in given word, wiht indices numbered
 * right-to-left */
inline bool bo_bit(uint32_t word, uint32_t index) {
  uint32_t isolated_bit = word & (1 << index);
  return isolated_bit != 0;
}

/* toggle bit value at bit postion in given word, with indices numbered
 * right-to-left */
inline uint32_t bo_toggle(uint32_t word, uint32_t index) {
  return word ^ (1 << index);
}

/* Return binary string for the representation of n where b is a string of
   length BO_WORD_SIZE, i.e. an array of BO_WORD_SIZE + ! characters. */
inline void bo_binary_repr(uint32_t n, char* b) {
  uint32_t i;
  for (i = 0; i < BO_WORD_SIZE; i++)
    b[BO_WORD_SIZE - 1 - i] = (n & (1 << i)) == 0 ? '0' : '1';
  b[BO_WORD_SIZE] = '\0';
}

/* number of leading zeros, linear scan */
static inline uint32_t bo_nlz_64(uint64_t word) {
  uint32_t n = 2 * BO_WORD_SIZE;
  while (word != 0) {
    word >>= 1;
    n--;
  }
  return n;
}

/* number of leading zeros
   Harley's algorithm, based on table lookup, code from Hacker's Delight, 
   requires 32-bit ints. */
#ifndef __GNUC__
static inline uint32_t bo_nlz_32(uint32_t x) {
#define u 99
  static uint32_t table[64] = {
    32, 31, u, 16, u, 30, 3, u,  15, u, u, u, 29, 10, 2, u, 
    u, u, 12, 14, 21, u, 19, u,  u, 28, u, 25, u, 9, 1, u, 
    17, u, 4, u, u, u, 11 ,u,    13, 22, 20, u, 26, u, u, 18,
    5, u, u, 23, u, 27, u, 6,    u, 24, 7, u, 8, u, 0, u };
#undef u
  x = x | (x >> 1);
  x = x | (x >> 2);
  x = x | (x >> 4);
  x = x | (x >> 8);
  x = x | (x >>16);
  x = x * 0x06EB14F9;  /* multiplier is 7 * 255**3 */
  return table[x >> 26];
}
#endif

/* number of trailing zeros, linear scan */
static inline uint32_t bo_ntz_64(uint64_t word) {
  uint32_t n = 0;
  while ((word & 1) == 0) {
    word >>= 1;
    n++;
  }
  return n;
}

/* number of trailing zeros
   Table lookup using a de Bruinj cycle, code from Hacker's Delight,
   requires 32-bit ints. */
#ifndef __GNUC__
static inline uint32_t bo_ntz_32(uint32_t x) {
  static const uint32_t table[32] 
    = { 0, 1, 2, 24, 3, 19, 6, 25, 22, 4, 20, 10, 16, 7, 12, 26,
    31, 23, 18, 5, 21, 9, 15, 11, 30, 17, 8, 14, 29, 13, 28, 27 };

  if (x == 0) {
    return 32;
  }
  x = (x & -x) * 0x04D7651F;
  return table[x >> 27];
}
#endif
