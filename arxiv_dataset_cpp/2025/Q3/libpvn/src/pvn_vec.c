#include "pvn.h"

#ifdef PVN_TEST
int main(int argc, char *argv[])
{
  if (1 < argc) {
    (void)fprintf(stderr, "%s takes no arguments\n", *argv);
    return EXIT_FAILURE;
  }
  (void)printf("PVN_VECLEN = %u\n", PVN_FABI(pvn_vec_len,PVN_VEC_LEN)());
  (void)printf("PVN_SAFELEN(float) = %zu\n", PVN_SAFELEN(float));
  (void)printf("PVN_SAFELEN(double) = %zu\n", PVN_SAFELEN(double));
  return EXIT_SUCCESS;
}
#else /* !PVN_TEST */
unsigned PVN_FABI(pvn_vec_len,PVN_VEC_LEN)()
{
  return (unsigned)(PVN_VECLEN);
}
#endif /* ?PVN_TEST */
