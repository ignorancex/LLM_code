#include "pvn.h"

#ifdef PVN_TEST
int main(int argc, char *argv[])
{
  if ((argc != 5) && (argc != 7)) {
    (void)fprintf(stderr, "%s ar ai br bi [cr ci]\n", *argv);
    return EXIT_FAILURE;
  }
  const double ar = atof(argv[1]);
  const double ai = atof(argv[2]);
  const double br = atof(argv[3]);
  const double bi = atof(argv[4]);
  char s[26] = { '\0' };
  if (argc == 7) {
    const double cr = atof(argv[5]);
    const double ci = atof(argv[6]);
    double dr = 0.0, di = 0.0;
    pvn_zfma(&dr, &di, ar, ai, br, bi, cr, ci);
    (void)printf("pvn_zfma=(%s,", pvn_dtoa(s, dr));
    (void)printf("%s)\n", pvn_dtoa(s, di));
  }
  else {
    double cr = 0.0, ci = 0.0;
    pvn_zmul(&cr, &ci, ar, ai, br, bi);
    (void)printf("pvn_zmul=(%s,", pvn_dtoa(s, cr));
    (void)printf("%s)\n", pvn_dtoa(s, ci));
  }
  return EXIT_SUCCESS;
}
#else /* !PVN_TEST */
void PVN_FABI(pvn_cmul,PVN_CMUL)(float *const cr, float *const ci, const float *const ar, const float *const ai, const float *const br, const float *const bi)
{
  PVN_ASSERT(ar);
  PVN_ASSERT(ai);
  PVN_ASSERT(br);
  PVN_ASSERT(bi);
  pvn_cmul(cr, ci, *ar, *ai, *br, *bi);
}

void PVN_FABI(pvn_cfma,PVN_CFMA)(float *const dr, float *const di, const float *const ar, const float *const ai, const float *const br, const float *const bi, const float *const cr, const float *const ci)
{
  PVN_ASSERT(ar);
  PVN_ASSERT(ai);
  PVN_ASSERT(br);
  PVN_ASSERT(bi);
  PVN_ASSERT(cr);
  PVN_ASSERT(ci);
  pvn_cfma(dr, di, *ar, *ai, *br, *bi, *cr, *ci);
}

void PVN_FABI(pvn_zmul,PVN_ZMUL)(double *const cr, double *const ci, const double *const ar, const double *const ai, const double *const br, const double *const bi)
{
  PVN_ASSERT(ar);
  PVN_ASSERT(ai);
  PVN_ASSERT(br);
  PVN_ASSERT(bi);
  pvn_zmul(cr, ci, *ar, *ai, *br, *bi);
}

void PVN_FABI(pvn_zfma,PVN_ZFMA)(double *const dr, double *const di, const double *const ar, const double *const ai, const double *const br, const double *const bi, const double *const cr, const double *const ci)
{
  PVN_ASSERT(ar);
  PVN_ASSERT(ai);
  PVN_ASSERT(br);
  PVN_ASSERT(bi);
  PVN_ASSERT(cr);
  PVN_ASSERT(ci);
  pvn_zfma(dr, di, *ar, *ai, *br, *bi, *cr, *ci);
}

void PVN_FABI(pvn_wmul,PVN_WMUL)(long double *const cr, long double *const ci, const long double *const ar, const long double *const ai, const long double *const br, const long double *const bi)
{
  PVN_ASSERT(ar);
  PVN_ASSERT(ai);
  PVN_ASSERT(br);
  PVN_ASSERT(bi);
  pvn_wmul(cr, ci, *ar, *ai, *br, *bi);
}

void PVN_FABI(pvn_wfma,PVN_WFMA)(long double *const dr, long double *const di, const long double *const ar, const long double *const ai, const long double *const br, const long double *const bi, const long double *const cr, const long double *const ci)
{
  PVN_ASSERT(ar);
  PVN_ASSERT(ai);
  PVN_ASSERT(br);
  PVN_ASSERT(bi);
  PVN_ASSERT(cr);
  PVN_ASSERT(ci);
  pvn_wfma(dr, di, *ar, *ai, *br, *bi, *cr, *ci);
}

#ifdef PVN_QUADMATH
void PVN_FABI(pvn_ymul,PVN_YMUL)(__float128 *const cr, __float128 *const ci, const __float128 *const ar, const __float128 *const ai, const __float128 *const br, const __float128 *const bi)
{
  PVN_ASSERT(ar);
  PVN_ASSERT(ai);
  PVN_ASSERT(br);
  PVN_ASSERT(bi);
  pvn_ymul(cr, ci, *ar, *ai, *br, *bi);
}

void PVN_FABI(pvn_yfma,PVN_YFMA)(__float128 *const dr, __float128 *const di, const __float128 *const ar, const __float128 *const ai, const __float128 *const br, const __float128 *const bi, const __float128 *const cr, const __float128 *const ci)
{
  PVN_ASSERT(ar);
  PVN_ASSERT(ai);
  PVN_ASSERT(br);
  PVN_ASSERT(bi);
  PVN_ASSERT(cr);
  PVN_ASSERT(ci);
  pvn_yfma(dr, di, *ar, *ai, *br, *bi, *cr, *ci);
}
#else /* !PVN_QUADMATH */
void PVN_FABI(pvn_ymul,PVN_YMUL)(long double *const cr, long double *const ci, const long double *const ar, const long double *const ai, const long double *const br, const long double *const bi)
{
  PVN_ASSERT(ar);
  PVN_ASSERT(ai);
  PVN_ASSERT(br);
  PVN_ASSERT(bi);
  pvn_wmul(cr, ci, *ar, *ai, *br, *bi);
}

void PVN_FABI(pvn_yfma,PVN_YFMA)(long double *const dr, long double *const di, const long double *const ar, const long double *const ai, const long double *const br, const long double *const bi, const long double *const cr, const long double *const ci)
{
  PVN_ASSERT(ar);
  PVN_ASSERT(ai);
  PVN_ASSERT(br);
  PVN_ASSERT(bi);
  PVN_ASSERT(cr);
  PVN_ASSERT(ci);
  pvn_wfma(dr, di, *ar, *ai, *br, *bi, *cr, *ci);
}
#endif /* ?PVN_QUADMATH */
#endif /* ?PVN_TEST */
