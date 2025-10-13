#include "pvn.h"

#ifdef PVN_TEST
int main(int argc, char *argv[])
{
  if (1 < argc) {
    (void)fprintf(stderr, "%s takes no arguments\n", *argv);
    return EXIT_FAILURE;
  }
#ifdef PVN_CR_MATH
  /* check the linkage */
  (void)printf("cr_hypotf =%18p\n", cr_hypotf);
  (void)printf("cr_powf   =%18p\n", cr_powf);
  (void)printf("cr_rsqrtf =%18p\n", cr_rsqrtf);
  (void)printf("cr_sincosf=%18p\n", cr_sincosf);
  (void)printf("cr_hypot  =%18p\n", cr_hypot);
  (void)printf("cr_pow    =%18p\n", cr_pow);
  (void)printf("cr_rsqrt  =%18p\n", cr_rsqrt);
  (void)printf("cr_sincos =%18p\n", cr_sincos);
  (void)printf("cr_hypotl =%18p\n", cr_hypotl);
  (void)printf("cr_powl   =%18p\n", cr_powl);
  (void)printf("cr_rsqrtl =%18p\n", cr_rsqrtl);
#if (defined(PVN_QUADMATH) || (defined(__PPC64__) && defined(__LITTLE_ENDIAN__) && defined(_ARCH_PWR9)))
  (void)printf("cr_hypotq =%18p\n", cr_hypotq);
  (void)printf("cr_rsqrtq =%18p\n", cr_rsqrtq);
  (void)printf("cr_sqrtq  =%18p\n", cr_sqrtq);
#endif /* PVN_QUADMATH */
#endif /* PVN_CR_MATH */
  return EXIT_SUCCESS;
}
#else /* !PVN_TEST */
float rhypotf(float x, float y)
{
  return (1.0f / hypotf(x, y));
}

double rhypot(double x, double y)
{
  return (1.0 / hypot(x, y));
}

long double rhypotl(long double x, long double y)
{
  return (1.0L / hypotl(x, y));
}

#ifdef PVN_QUADMATH
__float128 rhypotq(__float128 x, __float128 y)
{
  return (1.0q / hypotq(x, y));
}
#endif /* PVN_QUADMATH */
#endif /* ?PVN_TEST */
