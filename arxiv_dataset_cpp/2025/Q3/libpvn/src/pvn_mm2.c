#include "pvn.h"

#ifdef PVN_TEST
int main(int argc, char *argv[])
{
  if (1 < argc) {
    (void)fprintf(stderr, "%s takes no arguments\n", *argv);
    return EXIT_FAILURE;
  }
  /* A =
  (-1.0, 1.0) ( 2.0, 2.0)
  ( 1.0,-1.0) (-2.0,-2.0)
  */
  const double
    a11r = -1.0, a11i = 1.0, a21r = 1.0, a21i = -1.0, a12r = 2.0, a12i = 2.0, a22r = -2.0, a22i = -2.0;
  double
    c11r = 0.0, c11i = 0.0, c21r = 0.0, c21i = 0.0, c12r = 0.0, c12i = 0.0, c22r = 0.0, c22i = 0.0;
  /* C = 0 + A * A */
  pvn_zmma2(&c11r, &c11i, &c21r, &c21i, &c12r, &c12i, &c22r, &c22i, a11r, a11i, a21r, a21i, a12r, a12i, a22r, a22i, a11r, a11i, a21r, a21i, a12r, a12i, a22r, a22i);
  /* expected output:
  ( 4.0,-2.0) (-4.0,-8.0)
  (-4.0, 2.0) ( 4.0, 8.0)
  */
  (void)printf("(%# 4.1F,%# 4.1F) (%# 4.1F,%# 4.1F)\n", c11r, c11i, c12r, c12i);
  (void)printf("(%# 4.1F,%# 4.1F) (%# 4.1F,%# 4.1F)\n", c21r, c21i, c22r, c22i);
  pvn_zmm2(&c11r, &c11i, &c21r, &c21i, &c12r, &c12i, &c22r, &c22i, a11r, a11i, a21r, a21i, a12r, a12i, a22r, a22i, a11r, a11i, a21r, a21i, a12r, a12i, a22r, a22i);
  (void)printf("(%# 4.1F,%# 4.1F) (%# 4.1F,%# 4.1F)\n", c11r, c11i, c12r, c12i);
  (void)printf("(%# 4.1F,%# 4.1F) (%# 4.1F,%# 4.1F)\n", c21r, c21i, c22r, c22i);
  return EXIT_SUCCESS;
}
#else /* !PVN_TEST */
#endif /* ?PVN_TEST */
