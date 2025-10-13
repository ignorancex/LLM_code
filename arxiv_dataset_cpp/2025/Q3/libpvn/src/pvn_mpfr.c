#include "pvn.h"

#ifdef PVN_TEST
int main(int argc, char *argv[])
{
  if (1 < argc) {
    (void)fprintf(stderr, "%s takes no arguments\n", *argv);
    return EXIT_FAILURE;
  }
  mpfr_rnd_t rnd = __MPFR_RND_INVALID;
  mpfr_prec_t prec = __MPFR_PREC_INVALID;
  mpfr_exp_t emin = __MPFR_EXP_INVALID, emax = __MPFR_EXP_INVALID;
  const int r = PVN_FABI(pvn_mpfr_start,PVN_MPFR_START)(&rnd, &prec, &emin, &emax);
  (void)printf("PVN_MPFR_START=%d\n", r);
  (void)printf("rnd=%d\n", (int)rnd);
  (void)printf("prec=%ld\n", (long)prec);
  (void)printf("emin=%ld\n", (long)emin);
  (void)printf("emax=%ld\n", (long)emax);
  return (r ? EXIT_FAILURE : PVN_FABI(pvn_mpfr_stop,PVN_MPFR_STOP)());
}
#else /* !PVN_TEST */
int PVN_FABI(pvn_mpfr_start,PVN_MPFR_START)(mpfr_rnd_t *const rnd, mpfr_prec_t *const prec, mpfr_exp_t *const emin, mpfr_exp_t *const emax)
{
  PVN_ASSERT(rnd);
  PVN_ASSERT(prec);
  PVN_ASSERT(emin);
  PVN_ASSERT(emax);
  if (*rnd == __MPFR_RND_INVALID)
    *rnd = mpfr_get_default_rounding_mode();
  else if ((*rnd < MPFR_RNDN) || (*rnd > MPFR_RNDF))
    return -1;
  else
    mpfr_set_default_rounding_mode(*rnd);
  if (*prec == __MPFR_PREC_INVALID)
    *prec = mpfr_get_default_prec();
  else if ((*prec < MPFR_PREC_MIN) || (*prec > MPFR_PREC_MAX))
    return -2;
  else
    mpfr_set_default_prec(*prec);
  if (*emin == __MPFR_EXP_INVALID)
    *emin = mpfr_get_emin_min();
  if (mpfr_set_emin(*emin))
    return 1;
  if (*emax == __MPFR_EXP_INVALID)
    *emax = mpfr_get_emax_max();
  if (mpfr_set_emax(*emax))
    return 2;
  return 0;
}

int PVN_FABI(pvn_mpfr_stop,PVN_MPFR_STOP)()
{
  mpfr_free_cache();
  return 0;
}
#endif /* ?PVN_TEST */
