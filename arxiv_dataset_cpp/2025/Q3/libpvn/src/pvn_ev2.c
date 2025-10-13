#include "pvn.h"

/* a simplification of the expressions for Λ inspired by Vjeran Hari */

#ifdef FLT_BIG_EXP
#error FLT_BIG_EXP already defined
#else /* !FLT_BIG_EXP */
#define FLT_BIG_EXP (FLT_MAX_EXP - 2)
#endif /* ?FLT_BIG_EXP */

#ifdef DBL_BIG_EXP
#error DBL_BIG_EXP already defined
#else /* !DBL_BIG_EXP */
#define DBL_BIG_EXP (DBL_MAX_EXP - 2)
#endif /* ?DBL_BIG_EXP */

#ifdef LDBL_BIG_EXP
#error LDBL_BIG_EXP already defined
#else /* !LDBL_BIG_EXP */
#define LDBL_BIG_EXP (LDBL_MAX_EXP - 2)
#endif /* ?LDBL_BIG_EXP */

#ifdef PVN_TEST
int main(int argc, char *argv[])
{
  if ((argc < 4) || (argc > 5)) {
    (void)fprintf(stderr, "%s a11 a22 a21r [a21i]\n", *argv);
    return EXIT_FAILURE;
  }
  const double a11 = atof(argv[1]);
  const double a22 = atof(argv[2]);
  const double a21r = atof(argv[3]);
  const double a21i = ((5 == argc) ? atof(argv[4]) : 0.0);
  double cs = 1.0, snr = 0.0, sni = 0.0, l1 = 0.0, l2 = 0.0;
  int es = 0, lc = 0;
  char s[26u] = { '\0' };
  if (4 == argc) {
    lc = PVN_FABI(pvn_dljev2,PVN_DLJEV2)(&a11, &a22, &a21r, &cs, &snr, &l1, &l2, &es);
    (void)printf("pvn_dljev2 = %d, es = %d\n", lc, es);
    sni = 0.0;
  }
  else {
    lc = PVN_FABI(pvn_zljev2,PVN_ZLJEV2)(&a11, &a22, &a21r, &a21i, &cs, &snr, &sni, &l1, &l2, &es);
    (void)printf("pvn_zljev2 = %d, es = %d\n", lc, es);
  }
  (void)printf("a11  = %s\n", pvn_dtoa(s, a11));
  (void)printf("a22  = %s\n", pvn_dtoa(s, a22));
  (void)printf("a21r = %s\n", pvn_dtoa(s, a21r));
  (void)printf("a21i = %s\n", pvn_dtoa(s, a21i));
  (void)printf("cs   = %s\n", pvn_dtoa(s, cs));
  (void)printf("snr  = %s\n", pvn_dtoa(s, snr));
  (void)printf("sni  = %s\n", pvn_dtoa(s, sni));
  (void)printf("l1   = %s\n", pvn_dtoa(s, l1));
  (void)printf("l2   = %s\n", pvn_dtoa(s, l2));
  if (es) {
    l1 = scalbn(l1, es);
    l2 = scalbn(l2, es);
    (void)printf("l11  = %s\n", pvn_dtoa(s, l1));
    (void)printf("l22  = %s\n", pvn_dtoa(s, l2));
  }
  return EXIT_SUCCESS;
}
#else /* !PVN_TEST */
int PVN_FABI(pvn_sljeu2,PVN_SLJEU2)(const float *const a11, const float *const a22, const float *const a21, float *const cs, float *const sn, float *const tg, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21);
  PVN_ASSERT(cs);
  PVN_ASSERT(sn);
  PVN_ASSERT(tg);
  PVN_ASSERT(es);
  float a1 = *a11;
  if (!isfinite(a1))
    return -1;
  float a2 = *a22;
  if (!isfinite(a2))
    return -2;
  float ar = *a21;
  if (!isfinite(ar))
    return -3;
  const int wt = (*es ? 1 : 0);
  int
    e1 = (a1 != 0.0f),
    e2 = (a2 != 0.0f),
    er = (ar != 0.0f);
  *es = (e1 | (e2 << 1) | (er << 2));
  if (*es) {
    (void)frexpf(fmaxf(fabsf(a1), FLT_TRUE_MIN), &e1);
    (void)frexpf(fmaxf(fabsf(a2), FLT_TRUE_MIN), &e2);
    (void)frexpf(fmaxf(fabsf(ar), FLT_TRUE_MIN), &er);
    e1 = pvn_imax3(e1, e2, er);
    er = *es;
    *es = (FLT_BIG_EXP - e1);
    a1 = scalbnf(a1, *es);
    a2 = scalbnf(a2, *es);
    ar = scalbnf(ar, *es);
    *es = -*es;
  }
  const float
    aa = fabsf(ar);
  /* a non-zero element underflows due to scaling */
  e1 = ((((er & 1) && (fabsf(a1) < FLT_MIN)) || ((er & 2) && (fabsf(a2) < FLT_MIN)) || ((er & 4) && (aa < FLT_MIN))) << 1);
  const float
    as = copysignf(1.0f, ar),
    an = (aa * 2.0f),
    ad = (a1 - a2),
    t2 = copysignf(fminf(fmaxf(an / fabsf(ad), 0.0f), FLT_MAX), ad),
    t1 = (t2 / (1.0f + hypotf(t2, 1.0f))),
    sc = hypotf(t1, 1.0f),
    c1 = (1.0f / sc);
  *cs = c1;
  *tg = t1;
  if (wt)
    *sn = (as * t1);
  else {
    const float s1 = (t1 / sc);
    *sn = (as * s1);
  }
  /* sine/tangent underflows with a non-zero aa */
  e2 = (((aa != 0.0f) && (fabsf(*sn) < FLT_MIN)) << 2);
  return (wt | e1 | e2);
}

int PVN_FABI(pvn_sljev2,PVN_SLJEV2)(const float *const a11, const float *const a22, const float *const a21, float *const cs, float *const sn, float *const l1, float *const l2, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21);
  PVN_ASSERT(cs);
  PVN_ASSERT(sn);
  PVN_ASSERT(l1);
  PVN_ASSERT(l2);
  PVN_ASSERT(es);
  float a1 = *a11;
  if (!isfinite(a1))
    return -1;
  float a2 = *a22;
  if (!isfinite(a2))
    return -2;
  float ar = *a21;
  if (!isfinite(ar))
    return -3;
  const int wt = (*es ? 1 : 0);
  int
    e1 = (a1 != 0.0f),
    e2 = (a2 != 0.0f),
    er = (ar != 0.0f);
  *es = (e1 | (e2 << 1) | (er << 2));
  if (*es) {
    (void)frexpf(fmaxf(fabsf(a1), FLT_TRUE_MIN), &e1);
    (void)frexpf(fmaxf(fabsf(a2), FLT_TRUE_MIN), &e2);
    (void)frexpf(fmaxf(fabsf(ar), FLT_TRUE_MIN), &er);
    e1 = pvn_imax3(e1, e2, er);
    er = *es;
    *es = (FLT_BIG_EXP - e1);
    a1 = scalbnf(a1, *es);
    a2 = scalbnf(a2, *es);
    ar = scalbnf(ar, *es);
    *es = -*es;
  }
  const float
    aa = fabsf(ar);
  /* a non-zero element underflows due to scaling */
  e1 = ((((er & 1) && (fabsf(a1) < FLT_MIN)) || ((er & 2) && (fabsf(a2) < FLT_MIN)) || ((er & 4) && (aa < FLT_MIN))) << 1);
  const float
    as = copysignf(1.0f, ar),
    an = (aa * 2.0f),
    ad = (a1 - a2),
    t2 = copysignf(fminf(fmaxf(an / fabsf(ad), 0.0f), FLT_MAX), ad),
    t1 = (t2 / (1.0f + hypotf(t2, 1.0f))),
    sc = hypotf(t1, 1.0f),
    c1 = (1.0f / sc);
  *cs = c1;
  if (wt)
    *sn = (as * t1);
  else {
    const float s1 = (t1 / sc);
    *sn = (as * s1);
  }
  /* sine/tangent underflows with a non-zero aa */
  e2 = (((aa != 0.0f) && (fabsf(*sn) < FLT_MIN)) << 2);
  *l1 = fmaf( t1, aa, a1);
  *l2 = fmaf(-t1, aa, a2);
  /* a non-zero matrix and the scaled eigenvalue with the smaller magnitude underflows */
  er = ((er && (fminf(fabsf(*l1), fabsf(*l2)) < FLT_MIN)) << 3);
  return (wt | e1 | e2 | er);
}

int PVN_FABI(pvn_dljeu2,PVN_DLJEU2)(const double *const a11, const double *const a22, const double *const a21, double *const cs, double *const sn, double *const tg, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21);
  PVN_ASSERT(cs);
  PVN_ASSERT(sn);
  PVN_ASSERT(tg);
  PVN_ASSERT(es);
  double a1 = *a11;
  if (!isfinite(a1))
    return -1;
  double a2 = *a22;
  if (!isfinite(a2))
    return -2;
  double ar = *a21;
  if (!isfinite(ar))
    return -3;
  const int wt = (*es ? 1 : 0);
  int
    e1 = (a1 != 0.0),
    e2 = (a2 != 0.0),
    er = (ar != 0.0);
  *es = (e1 | (e2 << 1) | (er << 2));
  if (*es) {
    (void)frexp(fmax(fabs(a1), DBL_TRUE_MIN), &e1);
    (void)frexp(fmax(fabs(a2), DBL_TRUE_MIN), &e2);
    (void)frexp(fmax(fabs(ar), DBL_TRUE_MIN), &er);
    e1 = pvn_imax3(e1, e2, er);
    er = *es;
    *es = (DBL_BIG_EXP - e1);
    a1 = scalbn(a1, *es);
    a2 = scalbn(a2, *es);
    ar = scalbn(ar, *es);
    *es = -*es;
  }
  const double
    aa = fabs(ar);
  /* a non-zero element underflows due to scaling */
  e1 = ((((er & 1) && (fabs(a1) < DBL_MIN)) || ((er & 2) && (fabs(a2) < DBL_MIN)) || ((er & 4) && (aa < DBL_MIN))) << 1);
  const double
    as = copysign(1.0, ar),
    an = (aa * 2.0),
    ad = (a1 - a2),
    t2 = copysign(fmin(fmax(an / fabs(ad), 0.0), DBL_MAX), ad),
    t1 = (t2 / (1.0 + hypot(t2, 1.0))),
    sc = hypot(t1, 1.0),
    c1 = (1.0 / sc);
  *cs = c1;
  *tg = t1;
  if (wt)
    *sn = (as * t1);
  else {
    const double s1 = (t1 / sc);
    *sn = (as * s1);
  }
  /* sine/tangent underflows with a non-zero aa */
  e2 = (((aa != 0.0) && (fabs(*sn) < DBL_MIN)) << 2);
  return (wt | e1 | e2);
}

int PVN_FABI(pvn_dljev2,PVN_DLJEV2)(const double *const a11, const double *const a22, const double *const a21, double *const cs, double *const sn, double *const l1, double *const l2, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21);
  PVN_ASSERT(cs);
  PVN_ASSERT(sn);
  PVN_ASSERT(l1);
  PVN_ASSERT(l2);
  PVN_ASSERT(es);
  double a1 = *a11;
  if (!isfinite(a1))
    return -1;
  double a2 = *a22;
  if (!isfinite(a2))
    return -2;
  double ar = *a21;
  if (!isfinite(ar))
    return -3;
  const int wt = (*es ? 1 : 0);
  int
    e1 = (a1 != 0.0),
    e2 = (a2 != 0.0),
    er = (ar != 0.0);
  *es = (e1 | (e2 << 1) | (er << 2));
  if (*es) {
    (void)frexp(fmax(fabs(a1), DBL_TRUE_MIN), &e1);
    (void)frexp(fmax(fabs(a2), DBL_TRUE_MIN), &e2);
    (void)frexp(fmax(fabs(ar), DBL_TRUE_MIN), &er);
    e1 = pvn_imax3(e1, e2, er);
    er = *es;
    *es = (DBL_BIG_EXP - e1);
    a1 = scalbn(a1, *es);
    a2 = scalbn(a2, *es);
    ar = scalbn(ar, *es);
    *es = -*es;
  }
  const double
    aa = fabs(ar);
  /* a non-zero element underflows due to scaling */
  e1 = ((((er & 1) && (fabs(a1) < DBL_MIN)) || ((er & 2) && (fabs(a2) < DBL_MIN)) || ((er & 4) && (aa < DBL_MIN))) << 1);
  const double
    as = copysign(1.0, ar),
    an = (aa * 2.0),
    ad = (a1 - a2),
    t2 = copysign(fmin(fmax(an / fabs(ad), 0.0), DBL_MAX), ad),
    t1 = (t2 / (1.0 + hypot(t2, 1.0))),
    sc = hypot(t1, 1.0),
    c1 = (1.0 / sc);
  *cs = c1;
  if (wt)
    *sn = (as * t1);
  else {
    const double s1 = (t1 / sc);
    *sn = (as * s1);
  }
  /* sine/tangent underflows with a non-zero aa */
  e2 = (((aa != 0.0) && (fabs(*sn) < DBL_MIN)) << 2);
  *l1 = fma( t1, aa, a1);
  *l2 = fma(-t1, aa, a2);
  /* a non-zero matrix and the scaled eigenvalue with the smaller magnitude underflows */
  er = ((er && (fmin(fabs(*l1), fabs(*l2)) < DBL_MIN)) << 3);
  return (wt | e1 | e2 | er);
}

int PVN_FABI(pvn_cljeu2,PVN_CLJEU2)(const float *const a11, const float *const a22, const float *const a21r, const float *const a21i, float *const cs, float *const snr, float *const sni, float *const tg, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21r);
  PVN_ASSERT(a21i);
  PVN_ASSERT(cs);
  PVN_ASSERT(snr);
  PVN_ASSERT(sni);
  PVN_ASSERT(tg);
  PVN_ASSERT(es);
  float a1 = *a11;
  if (!isfinite(a1))
    return -1;
  float a2 = *a22;
  if (!isfinite(a2))
    return -2;
  float ar = *a21r;
  if (!isfinite(ar))
    return -3;
  float ai = *a21i;
  if (!isfinite(ai))
    return -4;
  const int wt = (*es ? 1 : 0);
  int
    e1 = (a1 != 0.0f),
    e2 = (a2 != 0.0f),
    er = (ar != 0.0f),
    ei = (ai != 0.0f);
  *es = (e1 | (e2 << 1) | (er << 2) | (ei << 3));
  if (*es) {
    (void)frexpf(fmaxf(fabsf(a1), FLT_TRUE_MIN), &e1);
    (void)frexpf(fmaxf(fabsf(a2), FLT_TRUE_MIN), &e2);
    (void)frexpf(fmaxf(fabsf(ar), FLT_TRUE_MIN), &er);
    (void)frexpf(fmaxf(fabsf(ai), FLT_TRUE_MIN), &ei);
    e1 = pvn_imax4(e1, e2, er, ei);
    ei = *es;
    *es = (FLT_BIG_EXP - e1);
    a1 = scalbnf(a1, *es);
    a2 = scalbnf(a2, *es);
    ar = scalbnf(ar, *es);
    ai = scalbnf(ai, *es);
    *es = -*es;
  }
  const float
    ar_ = fabsf(ar),
    ai_ = fabsf(ai),
    aa = hypotf(ar_, ai_);
  /* a non-zero element underflows due to scaling */
  e1 = ((((ei & 1) && (fabsf(a1) < FLT_MIN)) || ((ei & 2) && (fabsf(a2) < FLT_MIN)) || ((ei & 4) && (ar_ < FLT_MIN)) || ((ei & 8) && (ai_ < FLT_MIN))) << 1);
  ar = copysignf(fminf(ar_ / aa, 1.0f), ar);
  ai = ai / fmaxf(aa, FLT_TRUE_MIN);
  const float
    an = (aa * 2.0f),
    ad = (a1 - a2),
    t2 = copysignf(fminf(fmaxf(an / fabsf(ad), 0.0f), FLT_MAX), ad),
    t1 = (t2 / (1.0f + hypotf(t2, 1.0f))),
    sc = hypotf(t1, 1.0f),
    c1 = (1.0f / sc);
  *cs = c1;
  *tg = t1;
  if (wt) {
    *snr = (ar * t1);
    *sni = (ai * t1);
  }
  else {
    const float s1 = (t1 / sc);
    *snr = (ar * s1);
    *sni = (ai * s1);
  }
  /* sine/tangent underflows with a non-zero aa */
  e2 = (((ar_ != 0.0f) && (fabsf(*snr) < FLT_MIN)) << 2);
  er = (((ai_ != 0.0f) && (fabsf(*sni) < FLT_MIN)) << 3);
  return (wt | e1 | e2 | er);
}

int PVN_FABI(pvn_cljev2,PVN_CLJEV2)(const float *const a11, const float *const a22, const float *const a21r, const float *const a21i, float *const cs, float *const snr, float *const sni, float *const l1, float *const l2, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21r);
  PVN_ASSERT(a21i);
  PVN_ASSERT(cs);
  PVN_ASSERT(snr);
  PVN_ASSERT(sni);
  PVN_ASSERT(l1);
  PVN_ASSERT(l2);
  PVN_ASSERT(es);
  float a1 = *a11;
  if (!isfinite(a1))
    return -1;
  float a2 = *a22;
  if (!isfinite(a2))
    return -2;
  float ar = *a21r;
  if (!isfinite(ar))
    return -3;
  float ai = *a21i;
  if (!isfinite(ai))
    return -4;
  const int wt = (*es ? 1 : 0);
  int
    e1 = (a1 != 0.0f),
    e2 = (a2 != 0.0f),
    er = (ar != 0.0f),
    ei = (ai != 0.0f);
  *es = (e1 | (e2 << 1) | (er << 2) | (ei << 3));
  if (*es) {
    (void)frexpf(fmaxf(fabsf(a1), FLT_TRUE_MIN), &e1);
    (void)frexpf(fmaxf(fabsf(a2), FLT_TRUE_MIN), &e2);
    (void)frexpf(fmaxf(fabsf(ar), FLT_TRUE_MIN), &er);
    (void)frexpf(fmaxf(fabsf(ai), FLT_TRUE_MIN), &ei);
    e1 = pvn_imax4(e1, e2, er, ei);
    ei = *es;
    *es = (FLT_BIG_EXP - e1);
    a1 = scalbnf(a1, *es);
    a2 = scalbnf(a2, *es);
    ar = scalbnf(ar, *es);
    ai = scalbnf(ai, *es);
    *es = -*es;
  }
  const float
    ar_ = fabsf(ar),
    ai_ = fabsf(ai),
    aa = hypotf(ar_, ai_);
  /* a non-zero element underflows due to scaling */
  e1 = ((((ei & 1) && (fabsf(a1) < FLT_MIN)) || ((ei & 2) && (fabsf(a2) < FLT_MIN)) || ((ei & 4) && (ar_ < FLT_MIN)) || ((ei & 8) && (ai_ < FLT_MIN))) << 1);
  ar = copysignf(fminf(ar_ / aa, 1.0f), ar);
  ai = ai / fmaxf(aa, FLT_TRUE_MIN);
  const float
    an = (aa * 2.0f),
    ad = (a1 - a2),
    t2 = copysignf(fminf(fmaxf(an / fabsf(ad), 0.0f), FLT_MAX), ad),
    t1 = (t2 / (1.0f + hypotf(t2, 1.0f))),
    sc = hypotf(t1, 1.0f),
    c1 = (1.0f / sc);
  *cs = c1;
  if (wt) {
    *snr = (ar * t1);
    *sni = (ai * t1);
  }
  else {
    const float s1 = (t1 / sc);
    *snr = (ar * s1);
    *sni = (ai * s1);
  }
  /* sine/tangent underflows with a non-zero aa */
  e2 = (((ar_ != 0.0f) && (fabsf(*snr) < FLT_MIN)) << 2);
  er = (((ai_ != 0.0f) && (fabsf(*sni) < FLT_MIN)) << 3);
  *l1 = fmaf( t1, aa, a1);
  *l2 = fmaf(-t1, aa, a2);
  /* a non-zero matrix and the scaled eigenvalue with the smaller magnitude underflows */
  ei = ((ei && (fminf(fabsf(*l1), fabsf(*l2)) < FLT_MIN)) << 4);
  return (wt | e1 | e2 | er | ei);
}

int PVN_FABI(pvn_zljeu2,PVN_ZLJEU2)(const double *const a11, const double *const a22, const double *const a21r, const double *const a21i, double *const cs, double *const snr, double *const sni, double *const tg, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21r);
  PVN_ASSERT(a21i);
  PVN_ASSERT(cs);
  PVN_ASSERT(snr);
  PVN_ASSERT(sni);
  PVN_ASSERT(tg);
  PVN_ASSERT(es);
  double a1 = *a11;
  if (!isfinite(a1))
    return -1;
  double a2 = *a22;
  if (!isfinite(a2))
    return -2;
  double ar = *a21r;
  if (!isfinite(ar))
    return -3;
  double ai = *a21i;
  if (!isfinite(ai))
    return -4;
  const int wt = (*es ? 1 : 0);
  int
    e1 = (a1 != 0.0),
    e2 = (a2 != 0.0),
    er = (ar != 0.0),
    ei = (ai != 0.0);
  *es = (e1 | (e2 << 1) | (er << 2) | (ei << 3));
  if (*es) {
    (void)frexp(fmax(fabs(a1), DBL_TRUE_MIN), &e1);
    (void)frexp(fmax(fabs(a2), DBL_TRUE_MIN), &e2);
    (void)frexp(fmax(fabs(ar), DBL_TRUE_MIN), &er);
    (void)frexp(fmax(fabs(ai), DBL_TRUE_MIN), &ei);
    e1 = pvn_imax4(e1, e2, er, ei);
    ei = *es;
    *es = (DBL_BIG_EXP - e1);
    a1 = scalbn(a1, *es);
    a2 = scalbn(a2, *es);
    ar = scalbn(ar, *es);
    ai = scalbn(ai, *es);
    *es = -*es;
  }
  const double
    ar_ = fabs(ar),
    ai_ = fabs(ai),
    aa = hypot(ar_, ai_);
  /* a non-zero element underflows due to scaling */
  e1 = ((((ei & 1) && (fabs(a1) < DBL_MIN)) || ((ei & 2) && (fabs(a2) < DBL_MIN)) || ((ei & 4) && (ar_ < DBL_MIN)) || ((ei & 8) && (ai_ < DBL_MIN))) << 1);
  ar = copysign(fmin(ar_ / aa, 1.0), ar);
  ai = ai / fmax(aa, DBL_TRUE_MIN);
  const double
    an = (aa * 2.0),
    ad = (a1 - a2),
    t2 = copysign(fmin(fmax(an / fabs(ad), 0.0), DBL_MAX), ad),
    t1 = (t2 / (1.0 + hypot(t2, 1.0))),
    sc = hypot(t1, 1.0),
    c1 = (1.0 / sc);
  *cs = c1;
  *tg = t1;
  if (wt) {
    *snr = (ar * t1);
    *sni = (ai * t1);
  }
  else {
    const double s1 = (t1 / sc);
    *snr = (ar * s1);
    *sni = (ai * s1);
  }
  /* sine/tangent underflows with a non-zero aa */
  e2 = (((ar_ != 0.0) && (fabs(*snr) < DBL_MIN)) << 2);
  er = (((ai_ != 0.0) && (fabs(*sni) < DBL_MIN)) << 3);
  return (wt | e1 | e2 | er);
}

int PVN_FABI(pvn_zljev2,PVN_ZLJEV2)(const double *const a11, const double *const a22, const double *const a21r, const double *const a21i, double *const cs, double *const snr, double *const sni, double *const l1, double *const l2, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21r);
  PVN_ASSERT(a21i);
  PVN_ASSERT(cs);
  PVN_ASSERT(snr);
  PVN_ASSERT(sni);
  PVN_ASSERT(l1);
  PVN_ASSERT(l2);
  PVN_ASSERT(es);
  double a1 = *a11;
  if (!isfinite(a1))
    return -1;
  double a2 = *a22;
  if (!isfinite(a2))
    return -2;
  double ar = *a21r;
  if (!isfinite(ar))
    return -3;
  double ai = *a21i;
  if (!isfinite(ai))
    return -4;
  const int wt = (*es ? 1 : 0);
  int
    e1 = (a1 != 0.0),
    e2 = (a2 != 0.0),
    er = (ar != 0.0),
    ei = (ai != 0.0);
  *es = (e1 | (e2 << 1) | (er << 2) | (ei << 3));
  if (*es) {
    (void)frexp(fmax(fabs(a1), DBL_TRUE_MIN), &e1);
    (void)frexp(fmax(fabs(a2), DBL_TRUE_MIN), &e2);
    (void)frexp(fmax(fabs(ar), DBL_TRUE_MIN), &er);
    (void)frexp(fmax(fabs(ai), DBL_TRUE_MIN), &ei);
    e1 = pvn_imax4(e1, e2, er, ei);
    ei = *es;
    *es = (DBL_BIG_EXP - e1);
    a1 = scalbn(a1, *es);
    a2 = scalbn(a2, *es);
    ar = scalbn(ar, *es);
    ai = scalbn(ai, *es);
    *es = -*es;
  }
  const double
    ar_ = fabs(ar),
    ai_ = fabs(ai),
    aa = hypot(ar_, ai_);
  /* a non-zero element underflows due to scaling */
  e1 = ((((ei & 1) && (fabs(a1) < DBL_MIN)) || ((ei & 2) && (fabs(a2) < DBL_MIN)) || ((ei & 4) && (ar_ < DBL_MIN)) || ((ei & 8) && (ai_ < DBL_MIN))) << 1);
  ar = copysign(fmin(ar_ / aa, 1.0), ar);
  ai = ai / fmax(aa, DBL_TRUE_MIN);
  const double
    an = (aa * 2.0),
    ad = (a1 - a2),
    t2 = copysign(fmin(fmax(an / fabs(ad), 0.0), DBL_MAX), ad),
    t1 = (t2 / (1.0 + hypot(t2, 1.0))),
    sc = hypot(t1, 1.0),
    c1 = (1.0 / sc);
  *cs = c1;
  if (wt) {
    *snr = (ar * t1);
    *sni = (ai * t1);
  }
  else {
    const double s1 = (t1 / sc);
    *snr = (ar * s1);
    *sni = (ai * s1);
  }
  /* sine/tangent underflows with a non-zero aa */
  e2 = (((ar_ != 0.0) && (fabs(*snr) < DBL_MIN)) << 2);
  er = (((ai_ != 0.0) && (fabs(*sni) < DBL_MIN)) << 3);
  *l1 = fma( t1, aa, a1);
  *l2 = fma(-t1, aa, a2);
  /* a non-zero matrix and the scaled eigenvalue with the smaller magnitude underflows */
  ei = ((ei && (fmin(fabs(*l1), fabs(*l2)) < DBL_MIN)) << 4);
  return (wt | e1 | e2 | er | ei);
}

int PVN_FABI(pvn_xljeu2,PVN_XLJEU2)(const long double *const a11, const long double *const a22, const long double *const a21, long double *const cs, long double *const sn, long double *const tg, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21);
  PVN_ASSERT(cs);
  PVN_ASSERT(sn);
  PVN_ASSERT(tg);
  PVN_ASSERT(es);
  long double a1 = *a11;
  if (!isfinite(a1))
    return -1;
  long double a2 = *a22;
  if (!isfinite(a2))
    return -2;
  long double ar = *a21;
  if (!isfinite(ar))
    return -3;
  const int wt = (*es ? 1 : 0);
  int
    e1 = (a1 != 0.0L),
    e2 = (a2 != 0.0L),
    er = (ar != 0.0L);
  *es = (e1 | (e2 << 1) | (er << 2));
  if (*es) {
    (void)frexpl(fmaxl(fabsl(a1), LDBL_TRUE_MIN), &e1);
    (void)frexpl(fmaxl(fabsl(a2), LDBL_TRUE_MIN), &e2);
    (void)frexpl(fmaxl(fabsl(ar), LDBL_TRUE_MIN), &er);
    e1 = pvn_imax3(e1, e2, er);
    er = *es;
    *es = (LDBL_BIG_EXP - e1);
    a1 = scalbnl(a1, *es);
    a2 = scalbnl(a2, *es);
    ar = scalbnl(ar, *es);
    *es = -*es;
  }
  const long double
    aa = fabsl(ar);
  /* a non-zero element underflows due to scaling */
  e1 = ((((er & 1) && (fabsl(a1) < LDBL_MIN)) || ((er & 2) && (fabsl(a2) < LDBL_MIN)) || ((er & 4) && (aa < LDBL_MIN))) << 1);
  const long double
    as = copysignl(1.0L, ar),
    an = (aa * 2.0L),
    ad = (a1 - a2),
    t2 = copysignl(fminl(fmaxl(an / fabsl(ad), 0.0L), LDBL_MAX), ad),
    t1 = (t2 / (1.0L + hypotl(t2, 1.0L))),
    sc = hypotl(t1, 1.0L),
    c1 = (1.0L / sc);
  *cs = c1;
  *tg = t1;
  if (wt)
    *sn = (as * t1);
  else {
    const long double s1 = (t1 / sc);
    *sn = (as * s1);
  }
  /* sine/tangent underflows with a non-zero aa */
  e2 = (((aa != 0.0L) && (fabsl(*sn) < LDBL_MIN)) << 2);
  return (wt | e1 | e2);
}

int PVN_FABI(pvn_xljev2,PVN_XLJEV2)(const long double *const a11, const long double *const a22, const long double *const a21, long double *const cs, long double *const sn, long double *const l1, long double *const l2, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21);
  PVN_ASSERT(cs);
  PVN_ASSERT(sn);
  PVN_ASSERT(l1);
  PVN_ASSERT(l2);
  PVN_ASSERT(es);
  long double a1 = *a11;
  if (!isfinite(a1))
    return -1;
  long double a2 = *a22;
  if (!isfinite(a2))
    return -2;
  long double ar = *a21;
  if (!isfinite(ar))
    return -3;
  const int wt = (*es ? 1 : 0);
  int
    e1 = (a1 != 0.0L),
    e2 = (a2 != 0.0L),
    er = (ar != 0.0L);
  *es = (e1 | (e2 << 1) | (er << 2));
  if (*es) {
    (void)frexpl(fmaxl(fabsl(a1), LDBL_TRUE_MIN), &e1);
    (void)frexpl(fmaxl(fabsl(a2), LDBL_TRUE_MIN), &e2);
    (void)frexpl(fmaxl(fabsl(ar), LDBL_TRUE_MIN), &er);
    e1 = pvn_imax3(e1, e2, er);
    er = *es;
    *es = (LDBL_BIG_EXP - e1);
    a1 = scalbnl(a1, *es);
    a2 = scalbnl(a2, *es);
    ar = scalbnl(ar, *es);
    *es = -*es;
  }
  const long double
    aa = fabsl(ar);
  /* a non-zero element underflows due to scaling */
  e1 = ((((er & 1) && (fabsl(a1) < LDBL_MIN)) || ((er & 2) && (fabsl(a2) < LDBL_MIN)) || ((er & 4) && (aa < LDBL_MIN))) << 1);
  const long double
    as = copysignl(1.0L, ar),
    an = (aa * 2.0L),
    ad = (a1 - a2),
    t2 = copysignl(fminl(fmaxl(an / fabsl(ad), 0.0L), LDBL_MAX), ad),
    t1 = (t2 / (1.0L + hypotl(t2, 1.0L))),
    sc = hypotl(t1, 1.0L),
    c1 = (1.0L / sc);
  *cs = c1;
  if (wt)
    *sn = (as * t1);
  else {
    const long double s1 = (t1 / sc);
    *sn = (as * s1);
  }
  /* sine/tangent underflows with a non-zero aa */
  e2 = (((aa != 0.0L) && (fabsl(*sn) < LDBL_MIN)) << 2);
  *l1 = fmal( t1, aa, a1);
  *l2 = fmal(-t1, aa, a2);
  /* a non-zero matrix and the scaled eigenvalue with the smaller magnitude underflows */
  er = ((er && (fminl(fabsl(*l1), fabsl(*l2)) < LDBL_MIN)) << 3);
  return (wt | e1 | e2 | er);
}

int PVN_FABI(pvn_wljeu2,PVN_WLJEU2)(const long double *const a11, const long double *const a22, const long double *const a21r, const long double *const a21i, long double *const cs, long double *const snr, long double *const sni, long double *const tg, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21r);
  PVN_ASSERT(a21i);
  PVN_ASSERT(cs);
  PVN_ASSERT(snr);
  PVN_ASSERT(sni);
  PVN_ASSERT(tg);
  PVN_ASSERT(es);
  long double a1 = *a11;
  if (!isfinite(a1))
    return -1;
  long double a2 = *a22;
  if (!isfinite(a2))
    return -2;
  long double ar = *a21r;
  if (!isfinite(ar))
    return -3;
  long double ai = *a21i;
  if (!isfinite(ai))
    return -4;
  const int wt = (*es ? 1 : 0);
  int
    e1 = (a1 != 0.0L),
    e2 = (a2 != 0.0L),
    er = (ar != 0.0L),
    ei = (ai != 0.0L);
  *es = (e1 | (e2 << 1) | (er << 2) | (ei << 3));
  if (*es) {
    (void)frexpl(fmaxl(fabsl(a1), LDBL_TRUE_MIN), &e1);
    (void)frexpl(fmaxl(fabsl(a2), LDBL_TRUE_MIN), &e2);
    (void)frexpl(fmaxl(fabsl(ar), LDBL_TRUE_MIN), &er);
    (void)frexpl(fmaxl(fabsl(ai), LDBL_TRUE_MIN), &ei);
    e1 = pvn_imax4(e1, e2, er, ei);
    ei = *es;
    *es = (LDBL_BIG_EXP - e1);
    a1 = scalbnl(a1, *es);
    a2 = scalbnl(a2, *es);
    ar = scalbnl(ar, *es);
    ai = scalbnl(ai, *es);
    *es = -*es;
  }
  const long double
    ar_ = fabsl(ar),
    ai_ = fabsl(ai),
    aa = hypotl(ar_, ai_);
  /* a non-zero element underflows due to scaling */
  e1 = ((((ei & 1) && (fabsl(a1) < LDBL_MIN)) || ((ei & 2) && (fabsl(a2) < LDBL_MIN)) || ((ei & 4) && (ar_ < LDBL_MIN)) || ((ei & 8) && (ai_ < LDBL_MIN))) << 1);
  ar = copysignl(fminl(ar_ / aa, 1.0L), ar);
  ai = ai / fmaxl(aa, LDBL_TRUE_MIN);
  const long double
    an = (aa * 2.0L),
    ad = (a1 - a2),
    t2 = copysignl(fminl(fmaxl(an / fabsl(ad), 0.0L), LDBL_MAX), ad),
    t1 = (t2 / (1.0L + hypotl(t2, 1.0L))),
    sc = hypotl(t1, 1.0L),
    c1 = (1.0L / sc);
  *cs = c1;
  *tg = t1;
  if (wt) {
    *snr = (ar * t1);
    *sni = (ai * t1);
  }
  else {
    const long double s1 = (t1 / sc);
    *snr = (ar * s1);
    *sni = (ai * s1);
  }
  /* sine/tangent underflows with a non-zero aa */
  e2 = (((ar_ != 0.0L) && (fabsl(*snr) < LDBL_MIN)) << 2);
  er = (((ai_ != 0.0L) && (fabsl(*sni) < LDBL_MIN)) << 3);
  return (wt | e1 | e2 | er);
}

int PVN_FABI(pvn_wljev2,PVN_WLJEV2)(const long double *const a11, const long double *const a22, const long double *const a21r, const long double *const a21i, long double *const cs, long double *const snr, long double *const sni, long double *const l1, long double *const l2, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21r);
  PVN_ASSERT(a21i);
  PVN_ASSERT(cs);
  PVN_ASSERT(snr);
  PVN_ASSERT(sni);
  PVN_ASSERT(l1);
  PVN_ASSERT(l2);
  PVN_ASSERT(es);
  long double a1 = *a11;
  if (!isfinite(a1))
    return -1;
  long double a2 = *a22;
  if (!isfinite(a2))
    return -2;
  long double ar = *a21r;
  if (!isfinite(ar))
    return -3;
  long double ai = *a21i;
  if (!isfinite(ai))
    return -4;
  const int wt = (*es ? 1 : 0);
  int
    e1 = (a1 != 0.0L),
    e2 = (a2 != 0.0L),
    er = (ar != 0.0L),
    ei = (ai != 0.0L);
  *es = (e1 | (e2 << 1) | (er << 2) | (ei << 3));
  if (*es) {
    (void)frexpl(fmaxl(fabsl(a1), LDBL_TRUE_MIN), &e1);
    (void)frexpl(fmaxl(fabsl(a2), LDBL_TRUE_MIN), &e2);
    (void)frexpl(fmaxl(fabsl(ar), LDBL_TRUE_MIN), &er);
    (void)frexpl(fmaxl(fabsl(ai), LDBL_TRUE_MIN), &ei);
    e1 = pvn_imax4(e1, e2, er, ei);
    ei = *es;
    *es = (LDBL_BIG_EXP - e1);
    a1 = scalbnl(a1, *es);
    a2 = scalbnl(a2, *es);
    ar = scalbnl(ar, *es);
    ai = scalbnl(ai, *es);
    *es = -*es;
  }
  const long double
    ar_ = fabsl(ar),
    ai_ = fabsl(ai),
    aa = hypotl(ar_, ai_);
  /* a non-zero element underflows due to scaling */
  e1 = ((((ei & 1) && (fabsl(a1) < LDBL_MIN)) || ((ei & 2) && (fabsl(a2) < LDBL_MIN)) || ((ei & 4) && (ar_ < LDBL_MIN)) || ((ei & 8) && (ai_ < LDBL_MIN))) << 1);
  ar = copysignl(fminl(ar_ / aa, 1.0L), ar);
  ai = ai / fmaxl(aa, LDBL_TRUE_MIN);
  const long double
    an = (aa * 2.0L),
    ad = (a1 - a2),
    t2 = copysignl(fminl(fmaxl(an / fabsl(ad), 0.0L), LDBL_MAX), ad),
    t1 = (t2 / (1.0L + hypotl(t2, 1.0L))),
    sc = hypotl(t1, 1.0L),
    c1 = (1.0L / sc);
  *cs = c1;
  if (wt) {
    *snr = (ar * t1);
    *sni = (ai * t1);
  }
  else {
    const long double s1 = (t1 / sc);
    *snr = (ar * s1);
    *sni = (ai * s1);
  }
  /* sine/tangent underflows with a non-zero aa */
  e2 = (((ar_ != 0.0L) && (fabsl(*snr) < LDBL_MIN)) << 2);
  er = (((ai_ != 0.0L) && (fabsl(*sni) < LDBL_MIN)) << 3);
  *l1 = fmal( t1, aa, a1);
  *l2 = fmal(-t1, aa, a2);
  /* a non-zero matrix and the scaled eigenvalue with the smaller magnitude underflows */
  ei = ((ei && (fminl(fabsl(*l1), fabsl(*l2)) < LDBL_MIN)) << 4);
  return (wt | e1 | e2 | er | ei);
}

#ifdef PVN_QUADMATH
#ifdef FLT128_BIG_EXP
#error FLT128_BIG_EXP already defined
#else /* !FLT128_BIG_EXP */
#define FLT128_BIG_EXP (FLT128_MAX_EXP - 2)
#endif /* ?FLT128_BIG_EXP */

int PVN_FABI(pvn_qljeu2,PVN_QLJEU2)(const __float128 *const a11, const __float128 *const a22, const __float128 *const a21, __float128 *const cs, __float128 *const sn, __float128 *const tg, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21);
  PVN_ASSERT(cs);
  PVN_ASSERT(sn);
  PVN_ASSERT(tg);
  PVN_ASSERT(es);
  __float128 a1 = *a11;
  if (!isfiniteq(a1))
    return -1;
  __float128 a2 = *a22;
  if (!isfiniteq(a2))
    return -2;
  __float128 ar = *a21;
  if (!isfiniteq(ar))
    return -3;
  const int wt = (*es ? 1 : 0);
  int
    e1 = (a1 != 0.0q),
    e2 = (a2 != 0.0q),
    er = (ar != 0.0q);
  *es = (e1 | (e2 << 1) | (er << 2));
  if (*es) {
    (void)frexpq(fmaxq(fabsq(a1), FLT128_TRUE_MIN), &e1);
    (void)frexpq(fmaxq(fabsq(a2), FLT128_TRUE_MIN), &e2);
    (void)frexpq(fmaxq(fabsq(ar), FLT128_TRUE_MIN), &er);
    e1 = pvn_imax3(e1, e2, er);
    er = *es;
    *es = (FLT128_BIG_EXP - e1);
    a1 = scalbnq(a1, *es);
    a2 = scalbnq(a2, *es);
    ar = scalbnq(ar, *es);
    *es = -*es;
  }
  const __float128
    aa = fabsq(ar);
  /* a non-zero element underflows due to scaling */
  e1 = ((((er & 1) && (fabsq(a1) < FLT128_MIN)) || ((er & 2) && (fabsq(a2) < FLT128_MIN)) || ((er & 4) && (aa < FLT128_MIN))) << 1);
  const __float128
    as = copysignq(1.0q, ar),
    an = (aa * 2.0q),
    ad = (a1 - a2),
    t2 = copysignq(fminq(fmaxq(an / fabsq(ad), 0.0q), FLT128_MAX), ad),
    t1 = (t2 / (1.0q + hypotq(t2, 1.0q))),
    sc = hypotq(t1, 1.0q),
    c1 = (1.0q / sc);
  *cs = c1;
  *tg = t1;
  if (wt)
    *sn = (as * t1);
  else {
    const __float128 s1 = (t1 / sc);
    *sn = (as * s1);
  }
  /* sine/tangent underflows with a non-zero aa */
  e2 = (((aa != 0.0q) && (fabsq(*sn) < FLT128_MIN)) << 2);
  return (wt | e1 | e2);
}

int PVN_FABI(pvn_qljev2,PVN_QLJEV2)(const __float128 *const a11, const __float128 *const a22, const __float128 *const a21, __float128 *const cs, __float128 *const sn, __float128 *const l1, __float128 *const l2, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21);
  PVN_ASSERT(cs);
  PVN_ASSERT(sn);
  PVN_ASSERT(l1);
  PVN_ASSERT(l2);
  PVN_ASSERT(es);
  __float128 a1 = *a11;
  if (!isfiniteq(a1))
    return -1;
  __float128 a2 = *a22;
  if (!isfiniteq(a2))
    return -2;
  __float128 ar = *a21;
  if (!isfiniteq(ar))
    return -3;
  const int wt = (*es ? 1 : 0);
  int
    e1 = (a1 != 0.0q),
    e2 = (a2 != 0.0q),
    er = (ar != 0.0q);
  *es = (e1 | (e2 << 1) | (er << 2));
  if (*es) {
    (void)frexpq(fmaxq(fabsq(a1), FLT128_TRUE_MIN), &e1);
    (void)frexpq(fmaxq(fabsq(a2), FLT128_TRUE_MIN), &e2);
    (void)frexpq(fmaxq(fabsq(ar), FLT128_TRUE_MIN), &er);
    e1 = pvn_imax3(e1, e2, er);
    er = *es;
    *es = (FLT128_BIG_EXP - e1);
    a1 = scalbnq(a1, *es);
    a2 = scalbnq(a2, *es);
    ar = scalbnq(ar, *es);
    *es = -*es;
  }
  const __float128
    aa = fabsq(ar);
  /* a non-zero element underflows due to scaling */
  e1 = ((((er & 1) && (fabsq(a1) < FLT128_MIN)) || ((er & 2) && (fabsq(a2) < FLT128_MIN)) || ((er & 4) && (aa < FLT128_MIN))) << 1);
  const __float128
    as = copysignq(1.0q, ar),
    an = (aa * 2.0q),
    ad = (a1 - a2),
    t2 = copysignq(fminq(fmaxq(an / fabsq(ad), 0.0q), FLT128_MAX), ad),
    t1 = (t2 / (1.0q + hypotq(t2, 1.0q))),
    sc = hypotq(t1, 1.0q),
    c1 = (1.0q / sc);
  *cs = c1;
  if (wt)
    *sn = (as * t1);
  else {
    const __float128 s1 = (t1 / sc);
    *sn = (as * s1);
  }
  /* sine/tangent underflows with a non-zero aa */
  e2 = (((aa != 0.0q) && (fabsq(*sn) < FLT128_MIN)) << 2);
  *l1 = fmaq( t1, aa, a1);
  *l2 = fmaq(-t1, aa, a2);
  /* a non-zero matrix and the scaled eigenvalue with the smaller magnitude underflows */
  er = ((er && (fminq(fabsq(*l1), fabsq(*l2)) < FLT128_MIN)) << 3);
  return (wt | e1 | e2 | er);
}

int PVN_FABI(pvn_yljeu2,PVN_YLJEU2)(const __float128 *const a11, const __float128 *const a22, const __float128 *const a21r, const __float128 *const a21i, __float128 *const cs, __float128 *const snr, __float128 *const sni, __float128 *const tg, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21r);
  PVN_ASSERT(a21i);
  PVN_ASSERT(cs);
  PVN_ASSERT(snr);
  PVN_ASSERT(sni);
  PVN_ASSERT(tg);
  PVN_ASSERT(es);
  __float128 a1 = *a11;
  if (!isfiniteq(a1))
    return -1;
  __float128 a2 = *a22;
  if (!isfiniteq(a2))
    return -2;
  __float128 ar = *a21r;
  if (!isfiniteq(ar))
    return -3;
  __float128 ai = *a21i;
  if (!isfiniteq(ai))
    return -4;
  const int wt = (*es ? 1 : 0);
  int
    e1 = (a1 != 0.0q),
    e2 = (a2 != 0.0q),
    er = (ar != 0.0q),
    ei = (ai != 0.0q);
  *es = (e1 | (e2 << 1) | (er << 2) | (ei << 3));
  if (*es) {
    (void)frexpq(fmaxq(fabsq(a1), FLT128_TRUE_MIN), &e1);
    (void)frexpq(fmaxq(fabsq(a2), FLT128_TRUE_MIN), &e2);
    (void)frexpq(fmaxq(fabsq(ar), FLT128_TRUE_MIN), &er);
    (void)frexpq(fmaxq(fabsq(ai), FLT128_TRUE_MIN), &ei);
    e1 = pvn_imax4(e1, e2, er, ei);
    ei = *es;
    *es = (FLT128_BIG_EXP - e1);
    a1 = scalbnq(a1, *es);
    a2 = scalbnq(a2, *es);
    ar = scalbnq(ar, *es);
    ai = scalbnq(ai, *es);
    *es = -*es;
  }
  const __float128
    ar_ = fabsq(ar),
    ai_ = fabsq(ai),
    aa = hypotq(ar_, ai_);
  if ((aa <= 0.0q) || !isfiniteq(aa)) {
    *cs = aa;
    return -9;
  }
  /* a non-zero element underflows due to scaling */
  e1 = ((((ei & 1) && (fabsq(a1) < FLT128_MIN)) || ((ei & 2) && (fabsq(a2) < FLT128_MIN)) || ((ei & 4) && (ar_ < FLT128_MIN)) || ((ei & 8) && (ai_ < FLT128_MIN))) << 1);
  ar = copysignq(fminq(ar_ / aa, 1.0q), ar);
  ai = ai / fmaxq(aa, FLT128_TRUE_MIN);
  const __float128
    an = (aa * 2.0q),
    ad = (a1 - a2),
    t2 = copysignq(fminq(fmaxq(an / fabsq(ad), 0.0q), FLT128_MAX), ad),
    t1 = (t2 / (1.0q + hypotq(t2, 1.0q))),
    sc = hypotq(t1, 1.0q),
    c1 = (1.0q / sc);
  *cs = c1;
  *tg = t1;
  if (wt) {
    *snr = (ar * t1);
    *sni = (ai * t1);
  }
  else {
    const __float128 s1 = (t1 / sc);
    *snr = (ar * s1);
    *sni = (ai * s1);
  }
  /* sine/tangent underflows with a non-zero aa */
  e2 = (((ar_ != 0.0q) && (fabsq(*snr) < FLT128_MIN)) << 2);
  er = (((ai_ != 0.0q) && (fabsq(*sni) < FLT128_MIN)) << 3);
  return (wt | e1 | e2 | er);
}

int PVN_FABI(pvn_yljev2,PVN_YLJEV2)(const __float128 *const a11, const __float128 *const a22, const __float128 *const a21r, const __float128 *const a21i, __float128 *const cs, __float128 *const snr, __float128 *const sni, __float128 *const l1, __float128 *const l2, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21r);
  PVN_ASSERT(a21i);
  PVN_ASSERT(cs);
  PVN_ASSERT(snr);
  PVN_ASSERT(sni);
  PVN_ASSERT(l1);
  PVN_ASSERT(l2);
  PVN_ASSERT(es);
  __float128 a1 = *a11;
  if (!isfiniteq(a1))
    return -1;
  __float128 a2 = *a22;
  if (!isfiniteq(a2))
    return -2;
  __float128 ar = *a21r;
  if (!isfiniteq(ar))
    return -3;
  __float128 ai = *a21i;
  if (!isfiniteq(ai))
    return -4;
  const int wt = (*es ? 1 : 0);
  int
    e1 = (a1 != 0.0q),
    e2 = (a2 != 0.0q),
    er = (ar != 0.0q),
    ei = (ai != 0.0q);
  *es = (e1 | (e2 << 1) | (er << 2) | (ei << 3));
  if (*es) {
    (void)frexpq(fmaxq(fabsq(a1), FLT128_TRUE_MIN), &e1);
    (void)frexpq(fmaxq(fabsq(a2), FLT128_TRUE_MIN), &e2);
    (void)frexpq(fmaxq(fabsq(ar), FLT128_TRUE_MIN), &er);
    (void)frexpq(fmaxq(fabsq(ai), FLT128_TRUE_MIN), &ei);
    e1 = pvn_imax4(e1, e2, er, ei);
    ei = *es;
    *es = (FLT128_BIG_EXP - e1);
    a1 = scalbnq(a1, *es);
    a2 = scalbnq(a2, *es);
    ar = scalbnq(ar, *es);
    ai = scalbnq(ai, *es);
    *es = -*es;
  }
  const __float128
    ar_ = fabsq(ar),
    ai_ = fabsq(ai),
    aa = hypotq(ar_, ai_);
  /* a non-zero element underflows due to scaling */
  e1 = ((((ei & 1) && (fabsq(a1) < FLT128_MIN)) || ((ei & 2) && (fabsq(a2) < FLT128_MIN)) || ((ei & 4) && (ar_ < FLT128_MIN)) || ((ei & 8) && (ai_ < FLT128_MIN))) << 1);
  ar = copysignq(fminq(ar_ / aa, 1.0q), ar);
  ai = ai / fmaxq(aa, FLT128_TRUE_MIN);
  const __float128
    an = (aa * 2.0q),
    ad = (a1 - a2),
    t2 = copysignq(fminq(fmaxq(an / fabsq(ad), 0.0q), FLT128_MAX), ad),
    t1 = (t2 / (1.0q + hypotq(t2, 1.0q))),
    sc = hypotq(t1, 1.0q),
    c1 = (1.0q / sc);
  *cs = c1;
  if (wt) {
    *snr = (ar * t1);
    *sni = (ai * t1);
  }
  else {
    const __float128 s1 = (t1 / sc);
    *snr = (ar * s1);
    *sni = (ai * s1);
  }
  /* sine/tangent underflows with a non-zero aa */
  e2 = (((ar_ != 0.0q) && (fabsq(*snr) < FLT128_MIN)) << 2);
  er = (((ai_ != 0.0q) && (fabsq(*sni) < FLT128_MIN)) << 3);
  *l1 = fmaq( t1, aa, a1);
  *l2 = fmaq(-t1, aa, a2);
  /* a non-zero matrix and the scaled eigenvalue with the smaller magnitude underflows */
  ei = ((ei && (fminq(fabsq(*l1), fabsq(*l2)) < FLT128_MIN)) << 4);
  return (wt | e1 | e2 | er | ei);
}
#else /* !PVN_QUADMATH */
int PVN_FABI(pvn_qljeu2,PVN_QLJEU2)(const long double *const a11, const long double *const a22, const long double *const a21, long double *const cs, long double *const sn, long double *const tg, int *const es)
{
  return PVN_FABI(pvn_xljeu2,PVN_XLJEU2)(a11, a22, a21, cs, sn, tg, es);
}

int PVN_FABI(pvn_qljev2,PVN_QLJEV2)(const long double *const a11, const long double *const a22, const long double *const a21, long double *const cs, long double *const sn, long double *const l1, long double *const l2, int *const es)
{
  return PVN_FABI(pvn_xljev2,PVN_XLJEV2)(a11, a22, a21, cs, sn, l1, l2, es);
}

int PVN_FABI(pvn_yljeu2,PVN_YLJEU2)(const long double *const a11, const long double *const a22, const long double *const a21r, const long double *const a21i, long double *const cs, long double *const snr, long double *const sni, long double *const tg, int *const es)
{
  return PVN_FABI(pvn_wljeu2,PVN_WLJEU2)(a11, a22, a21r, a21i, cs, snr, sni, tg, es);
}

int PVN_FABI(pvn_yljev2,PVN_YLJEV2)(const long double *const a11, const long double *const a22, const long double *const a21r, const long double *const a21i, long double *const cs, long double *const snr, long double *const sni, long double *const l1, long double *const l2, int *const es)
{
  return PVN_FABI(pvn_wljev2,PVN_WLJEV2)(a11, a22, a21r, a21i, cs, snr, sni, l1, l2, es);
}
#endif /* ?PVN_QUADMATH */
#endif /* ?PVN_TEST */
