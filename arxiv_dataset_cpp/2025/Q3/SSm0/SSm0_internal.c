#include "SSm0_internal.h"

double rat(long int a, long int b) {
  return ((double)a)/((double)b);
}

void check_range(int cond, const char* msg)
{
  if (cond)
    {
      fprintf(stderr, "Not valid region: %s, exiting!\n", msg);
      exit( EXIT_FAILURE );
    }

}

void region_unimplemented(const char* erstr)
{
  fprintf(stderr, "Unimplemented case: \"%s\", exiting!\n", erstr);
  exit( EXIT_FAILURE );
}


/* All parameters are collected here */
/* const struct PolylogConstants li_constants = */
/*   { */
/*     .epsdif = 5e-14,            /\* WTFepsdif *\/ */
/*     /\* Li22 x->1 *\/ */
/*     .li22x1c1 = -2., */
/*     .li22x1c2 = 2., */
/*     .li22x1c3 = 0.5, */
/*     .li22x1c4 = -0.166666666666666666667, */
/*     .li22x1c5 = 0.333333333333333333333, */
/*     .li22x1c6 = 1.64493406684822643647,  // Pi^2/6 */
/*     .li22x1c7 = -2.40411380631918857080, // -2*Zeta(3) */
/*     .li22x1c8 = 2.16464646742227638303,  // Pi^4/45 */

/*     /\* f1/f2 splitting for Li22 *\/ */
/*     .xi0 = 1, */
/*     .ccxzero = 0.36787944117144232159552377016146, /\* exp(-xi0); *\/ */

/*     .Pi2 = 9.8696044010893586188, /\* Pi^2 *\/ */
/*     .Pi4 = 97.409091034002437236, /\* Pi^4 *\/ */

/*     /\* Recursion limits *\/ */
/*     .LI2LOGA0MAX = 8, */
/*     .LINLOGA0MAX = 18, */
/*     .LINLOGA1MAX = 8 */
/*   }; */


