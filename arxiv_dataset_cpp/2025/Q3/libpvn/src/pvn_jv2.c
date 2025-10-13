#include "pvn.h"

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
  double
    a11 = atof(argv[1]),
    a22 = atof(argv[2]),
    a21r = atof(argv[3]),
    a21i = ((5 == argc) ? atof(argv[4]) : 0.0),
    ch = 1.0,
    shr = 0.0,
    shi = 0.0,
    th = (4.0 / 5.0);
  int es = 0, lc = 0;
  if (4 == argc) {
    lc = PVN_FABI(pvn_dljv2,PVN_DLJV2)(&a11, &a22, &a21r, &ch, &shr, &th, &es);
    (void)printf("pvn_dljv2 = %d, es = %d\n", lc, es);
  }
  else {
    lc = PVN_FABI(pvn_zljv2,PVN_ZLJV2)(&a11, &a22, &a21r, &a21i, &ch, &shr, &shi, &th, &es);
    (void)printf("pvn_zljv2 = %d, es = %d\n", lc, es);
  }
  if (lc < 0)
    return EXIT_FAILURE;
  char s[46u] = { '\0' };
  if (4 == argc) {
#ifdef PVN_QUADMATH
    __float128 c[2][2];
#else /* !PVN_QUADMATH */
    long double c[2][2];
#endif /* ?PVN_QUADMATH */
    (void)printf("A =\n");
    (void)printf("[%s,", pvn_dtoa(s, a11));
    (void)printf("%s]\n", pvn_dtoa(s, a21r));
    (void)printf("[%s,", pvn_dtoa(s, a21r));
    (void)printf("%s]\n", pvn_dtoa(s, a22));
    (void)printf("V =\n");
    (void)printf("[%s,", pvn_dtoa(s, ch));
    (void)printf("%s]\n", pvn_dtoa(s, shr));
    (void)printf("[%s,", pvn_dtoa(s, shr));
    (void)printf("%s]\n", pvn_dtoa(s, ch));
#if (!defined(_WIN32) || defined(_DLL))
    (void)printf("V^T A V =\n");
    pvn_qmm2(&(c[0][0]), &(c[1][0]), &(c[0][1]), &(c[1][1]), ch, shr, shr, ch, a11, a21r, a21r, a22);
    pvn_qmm2(&(c[0][0]), &(c[1][0]), &(c[0][1]), &(c[1][1]), c[0][0], c[1][0], c[0][1], c[1][1], ch, shr, shr, ch);
    (void)printf("[%s,", pvn_qtoa(s, c[0][0]));
    (void)printf("%s]\n", pvn_qtoa(s, c[0][1]));
    (void)printf("[%s,", pvn_qtoa(s, c[1][0]));
    (void)printf("%s]\n", pvn_qtoa(s, c[1][1]));
    c[0][0] = -1.0;
    c[0][1] =  0.0;
    c[1][0] =  0.0;
    c[1][1] =  1.0;
    (void)printf("V^T J V - J =\n");
    pvn_qmma2(&(c[0][0]), &(c[1][0]), &(c[0][1]), &(c[1][1]), ch, shr, -shr, -ch, ch, shr, shr, ch);
    (void)printf("[%s,", pvn_qtoa(s, c[0][0]));
    (void)printf("%s]\n", pvn_qtoa(s, c[0][1]));
    (void)printf("[%s,", pvn_qtoa(s, c[1][0]));
    (void)printf("%s]\n", pvn_qtoa(s, c[1][1]));
#endif /* !_WIN32 || _DLL */
  }
  else {
#ifdef PVN_QUADMATH
    __float128 c[2][4];
#else /* !PVN_QUADMATH */
    long double c[2][4];
#endif /* ?PVN_QUADMATH */
    (void)printf("A =\n");
    (void)printf("[(%s,", pvn_dtoa(s, a11));
    (void)printf("%s),", pvn_dtoa(s, 0.0));
    (void)printf("(%s,", pvn_dtoa(s, a21r));
    (void)printf("%s)]\n", pvn_dtoa(s, -a21i));
    (void)printf("[(%s,", pvn_dtoa(s, a21r));
    (void)printf("%s),", pvn_dtoa(s, a21i));
    (void)printf("(%s,", pvn_dtoa(s, a22));
    (void)printf("%s)]\n", pvn_dtoa(s, 0.0));
    (void)printf("V =\n");
    (void)printf("[(%s,", pvn_dtoa(s, ch));
    (void)printf("%s),", pvn_dtoa(s, 0.0));
    (void)printf("(%s,", pvn_dtoa(s, shr));
    (void)printf("%s)]\n", pvn_dtoa(s, -shi));
    (void)printf("[(%s,", pvn_dtoa(s, shr));
    (void)printf("%s),", pvn_dtoa(s, shi));
    (void)printf("(%s,", pvn_dtoa(s, ch));
    (void)printf("%s)]\n", pvn_dtoa(s, 0.0));
#if (!defined(_WIN32) || defined(_DLL))
    pvn_ymm2(&(c[0][0]), &(c[0][1]), &(c[1][0]), &(c[1][1]), &(c[0][2]), &(c[0][3]), &(c[1][2]), &(c[1][3]), ch, 0.0, shr, shi, shr, -shi, ch, 0.0, a11, 0.0, a21r, a21i, a21r, -a21i, a22, 0.0);
    pvn_ymm2(&(c[0][0]), &(c[0][1]), &(c[1][0]), &(c[1][1]), &(c[0][2]), &(c[0][3]), &(c[1][2]), &(c[1][3]), c[0][0], c[0][1], c[1][0], c[1][1], c[0][2], c[0][3], c[1][2], c[1][3], ch, 0.0, shr, shi, shr, -shi, ch, 0.0);
    (void)printf("V^H A V =\n");
    (void)printf("[(%s,", pvn_qtoa(s, c[0][0]));
    (void)printf("%s),", pvn_qtoa(s, c[0][1]));
    (void)printf("(%s,", pvn_qtoa(s, c[0][2]));
    (void)printf("%s)]\n", pvn_qtoa(s, c[0][3]));
    (void)printf("[(%s,", pvn_qtoa(s, c[1][0]));
    (void)printf("%s),", pvn_qtoa(s, c[1][1]));
    (void)printf("(%s,", pvn_qtoa(s, c[1][2]));
    (void)printf("%s)]\n", pvn_qtoa(s, c[1][3]));
    c[0][0] = -1.0;
    c[0][1] =  0.0;
    c[0][2] =  0.0;
    c[0][3] =  0.0;
    c[1][0] =  0.0;
    c[1][1] =  0.0;
    c[1][2] =  1.0;
    c[1][3] =  0.0;
    pvn_ymma2(&(c[0][0]), &(c[0][1]), &(c[1][0]), &(c[1][1]), &(c[0][2]), &(c[0][3]), &(c[1][2]), &(c[1][3]), ch, 0.0, shr, shi, -shr, shi, -ch, -0.0, ch, 0.0, shr, shi, shr, -shi, ch, 0.0);
    (void)printf("V^H J V - J =\n");
    (void)printf("[(%s,", pvn_qtoa(s, c[0][0]));
    (void)printf("%s),", pvn_qtoa(s, c[0][1]));
    (void)printf("(%s,", pvn_qtoa(s, c[0][2]));
    (void)printf("%s)]\n", pvn_qtoa(s, c[0][3]));
    (void)printf("[(%s,", pvn_qtoa(s, c[1][0]));
    (void)printf("%s),", pvn_qtoa(s, c[1][1]));
    (void)printf("(%s,", pvn_qtoa(s, c[1][2]));
    (void)printf("%s)]\n", pvn_qtoa(s, c[1][3]));
#endif /* !_WIN32 || _DLL */
  }
  return EXIT_SUCCESS;
}
#else /* !PVN_TEST */
int PVN_FABI(pvn_sljv2,PVN_SLJV2)(const float *const a11, const float *const a22, const float *const a21, float *const ch, float *const sh, float *const th, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21);
  PVN_ASSERT(ch);
  PVN_ASSERT(sh);
  PVN_ASSERT(th);
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
  int wt = 0, bt = 0;
  switch (*es) {
  case -2:
    wt = 1;
    break;
  case -1:
    wt = 1;
    bt = 1;
    break;
  case 0:
    break;
  case 1:
    bt = 1;
    break;
  default:
    return -6;
  }
  int
    e1 = (a1 != 0.0f),
    e2 = (a2 != 0.0f),
    er = (ar != 0.0f);
  float t1 = 0.0f, aa = 0.0f;
  if (er) {
    *es = (e1 | (e2 << 1) | (er << 2));
    if (*es) {
      (void)frexpf(fmaxf(a1, FLT_TRUE_MIN), &e1);
      (void)frexpf(fmaxf(a2, FLT_TRUE_MIN), &e2);
      (void)frexpf(fmaxf(fabsf(ar), FLT_TRUE_MIN), &er);
      e1 = pvn_imax3(e1, e2, er);
      e2 = *es;
      *es = (FLT_BIG_EXP - e1);
      a1 = scalbnf(a1, *es);
      a2 = scalbnf(a2, *es);
      ar = scalbnf(ar, *es);
      er = (ar != 0.0f);
      *es = -*es;
    }
    if (er) {
      aa = fabsf(ar);
      /* a non-zero element underflows due to scaling */
      e1 = ((((e2 & 1) && (a1 < FLT_MIN)) || ((e2 & 2) && (a2 < FLT_MIN)) || ((e2 & 4) && (aa < FLT_MIN))) << 2);
      ar = copysignf(1.0f, ar);
      const float
        an = (2.0f * aa),
        ad = (a1 + a2),
        t2 = -((an >= ad) ? 1.0f : (an / ad));
      t1 = (t2 / (1.0f + sqrtf(fmaf(-t2, t2, 1.0f))));
      if (bt && (fabsf(t1) > *th)) {
        t1 = copysignf(*th, t1);
        bt = -1;
      }
      /* tangent underflows */
      e2 = ((fabsf(t1) < FLT_MIN) << 3);
    }
    else
      e2 = e1 = 0;
  }
  else
    *es = e2 = e1 = 0;
  *ch = rsqrtf(fmaf(-t1, t1, 1.0f));
  *sh = (wt ? t1 : (*ch * t1)) * ar;
  /* sine/tangent underflows with a non-zero aa */
  er = ((er && (fabsf(*sh) < FLT_MIN)) << 4);
  *th = t1;
  bt = ((bt == -1) << 1);
  return (wt | bt | e1 | e2 | er);
}

int PVN_FABI(pvn_dljv2,PVN_DLJV2)(const double *const a11, const double *const a22, const double *const a21, double *const ch, double *const sh, double *const th, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21);
  PVN_ASSERT(ch);
  PVN_ASSERT(sh);
  PVN_ASSERT(th);
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
  int wt = 0, bt = 0;
  switch (*es) {
  case -2:
    wt = 1;
    break;
  case -1:
    wt = 1;
    bt = 1;
    break;
  case 0:
    break;
  case 1:
    bt = 1;
    break;
  default:
    return -6;
  }
  int
    e1 = (a1 != 0.0),
    e2 = (a2 != 0.0),
    er = (ar != 0.0);
  double t1 = 0.0, aa = 0.0;
  if (er) {
    *es = (e1 | (e2 << 1) | (er << 2));
    if (*es) {
      (void)frexp(fmax(a1, DBL_TRUE_MIN), &e1);
      (void)frexp(fmax(a2, DBL_TRUE_MIN), &e2);
      (void)frexp(fmax(fabs(ar), DBL_TRUE_MIN), &er);
      e1 = pvn_imax3(e1, e2, er);
      e2 = *es;
      *es = (DBL_BIG_EXP - e1);
      a1 = scalbn(a1, *es);
      a2 = scalbn(a2, *es);
      ar = scalbn(ar, *es);
      er = (ar != 0.0);
      *es = -*es;
    }
    if (er) {
      aa = fabs(ar);
      /* a non-zero element underflows due to scaling */
      e1 = ((((e2 & 1) && (a1 < DBL_MIN)) || ((e2 & 2) && (a2 < DBL_MIN)) || ((e2 & 4) && (aa < DBL_MIN))) << 2);
      ar = copysign(1.0, ar);
      const double
        an = (2.0 * aa),
        ad = (a1 + a2),
        t2 = -((an >= ad) ? 1.0 : (an / ad));
      t1 = (t2 / (1.0 + sqrt(fma(-t2, t2, 1.0))));
      if (bt && (fabs(t1) > *th)) {
        t1 = copysign(*th, t1);
        bt = -1;
      }
      /* tangent underflows */
      e2 = ((fabs(t1) < DBL_MIN) << 3);
    }
    else
      e2 = e1 = 0;
  }
  else
    *es = e2 = e1 = 0;
  *ch = rsqrt(fma(-t1, t1, 1.0));
  *sh = (wt ? t1 : (*ch * t1)) * ar;
  /* sine/tangent underflows with a non-zero aa */
  er = ((er && (fabs(*sh) < DBL_MIN)) << 4);
  *th = t1;
  bt = ((bt == -1) << 1);
  return (wt | bt | e1 | e2 | er);
}

int PVN_FABI(pvn_cljv2,PVN_CLJV2)(const float *const a11, const float *const a22, const float *const a21r, const float *const a21i, float *const ch, float *const shr, float *const shi, float *const th, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21r);
  PVN_ASSERT(a21i);
  PVN_ASSERT(ch);
  PVN_ASSERT(shr);
  PVN_ASSERT(shi);
  PVN_ASSERT(th);
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
  int wt = 0, bt = 0;
  switch (*es) {
  case -2:
    wt = 1;
    break;
  case -1:
    wt = 1;
    bt = 1;
    break;
  case 0:
    break;
  case 1:
    bt = 1;
    break;
  default:
    return -8;
  }
  int
    e1 = (a1 != 0.0f),
    e2 = (a2 != 0.0f),
    er = (ar != 0.0f),
    ei = (ai != 0.0f);
  float t1 = 0.0f, aa = 0.0f;
  if (er || ei) {
    *es = (e1 | (e2 << 1) | (er << 2) | (ei << 3));
    if (*es) {
      (void)frexpf(fmaxf(a1, FLT_TRUE_MIN), &e1);
      (void)frexpf(fmaxf(a2, FLT_TRUE_MIN), &e2);
      (void)frexpf(fmaxf(fabsf(ar), FLT_TRUE_MIN), &er);
      (void)frexpf(fmaxf(fabsf(ai), FLT_TRUE_MIN), &ei);
      e1 = pvn_imax4(e1, e2, er, ei);
      e2 = *es;
      *es = (FLT_BIG_EXP - e1);
      a1 = scalbnf(a1, *es);
      a2 = scalbnf(a2, *es);
      ar = scalbnf(ar, *es);
      ai = scalbnf(ai, *es);
      er = (ar != 0.0f);
      ei = (ai != 0.0f);
      *es = -*es;
    }
    if (er || ei) {
      const float
        ar_ = fabsf(ar),
        ai_ = fabsf(ai);
      aa = hypotf(ar_, ai_); /* aa cannot be zero here if hypotf is correctly rounded */
      /* a non-zero element underflows due to scaling */
      e1 = ((((e2 & 1) && (a1 < FLT_MIN)) || ((e2 & 2) && (a2 < FLT_MIN)) || ((e2 & 4) && (ar_ < FLT_MIN)) || ((e2 & 8) && (ai_ < FLT_MIN))) << 2);
      ar = (ar / aa);
      ai = (ai / aa);
      const float
        an = (2.0f * aa),
        ad = (a1 + a2),
        t2 = -((an >= ad) ? 1.0f : (an / ad));
      t1 = (t2 / (1.0f + sqrtf(fmaf(-t2, t2, 1.0f))));
      if (bt && (fabsf(t1) > *th)) {
        t1 = copysignf(*th, t1);
        bt = -1;
      }
      /* tangent underflows */
      e2 = ((fabsf(t1) < FLT_MIN) << 3);
    }
    else
      e2 = e1 = 0;
  }
  else
    *es = e2 = e1 = 0;
  *ch = rsqrtf(fmaf(-t1, t1, 1.0f));
  *shi = (wt ? t1 : (*ch * t1));
  *shr = (*shi * ar);
  *shi *= ai;
  /* sine/tangent underflows with a non-zero aa */
  er = ((er && (fabsf(*shr) < FLT_MIN)) << 4);
  ei = ((ei && (fabsf(*shi) < FLT_MIN)) << 5);
  *th = t1;
  bt = ((bt == -1) << 1);
  return (wt | bt | e1 | e2 | er | ei);
}

int PVN_FABI(pvn_zljv2,PVN_ZLJV2)(const double *const a11, const double *const a22, const double *const a21r, const double *const a21i, double *const ch, double *const shr, double *const shi, double *const th, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21r);
  PVN_ASSERT(a21i);
  PVN_ASSERT(ch);
  PVN_ASSERT(shr);
  PVN_ASSERT(shi);
  PVN_ASSERT(th);
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
  int wt = 0, bt = 0;
  switch (*es) {
  case -2:
    wt = 1;
    break;
  case -1:
    wt = 1;
    bt = 1;
    break;
  case 0:
    break;
  case 1:
    bt = 1;
    break;
  default:
    return -8;
  }
  int
    e1 = (a1 != 0.0),
    e2 = (a2 != 0.0),
    er = (ar != 0.0),
    ei = (ai != 0.0);
  double t1 = 0.0, aa = 0.0;
  if (er || ei) {
    *es = (e1 | (e2 << 1) | (er << 2) | (ei << 3));
    if (*es) {
      (void)frexp(fmax(a1, DBL_TRUE_MIN), &e1);
      (void)frexp(fmax(a2, DBL_TRUE_MIN), &e2);
      (void)frexp(fmax(fabs(ar), DBL_TRUE_MIN), &er);
      (void)frexp(fmax(fabs(ai), DBL_TRUE_MIN), &ei);
      e1 = pvn_imax4(e1, e2, er, ei);
      e2 = *es;
      *es = (DBL_BIG_EXP - e1);
      a1 = scalbn(a1, *es);
      a2 = scalbn(a2, *es);
      ar = scalbn(ar, *es);
      ai = scalbn(ai, *es);
      er = (ar != 0.0);
      ei = (ai != 0.0);
      *es = -*es;
    }
    if (er || ei) {
      const double
        ar_ = fabs(ar),
        ai_ = fabs(ai);
      aa = hypot(ar_, ai_); /* aa cannot be zero here if hypot is correctly rounded */
      /* a non-zero element underflows due to scaling */
      e1 = ((((e2 & 1) && (a1 < DBL_MIN)) || ((e2 & 2) && (a2 < DBL_MIN)) || ((e2 & 4) && (ar_ < DBL_MIN)) || ((e2 & 8) && (ai_ < DBL_MIN))) << 2);
      ar = (ar / aa);
      ai = (ai / aa);
      const double
        an = (2.0 * aa),
        ad = (a1 + a2),
        t2 = -((an >= ad) ? 1.0 : (an / ad));
      t1 = (t2 / (1.0 + sqrt(fma(-t2, t2, 1.0))));
      if (bt && (fabs(t1) > *th)) {
        t1 = copysign(*th, t1);
        bt = -1;
      }
      /* tangent underflows */
      e2 = ((fabs(t1) < DBL_MIN) << 3);
    }
    else
      e2 = e1 = 0;
  }
  else
    *es = e2 = e1 = 0;
  *ch = rsqrt(fma(-t1, t1, 1.0));
  *shi = (wt ? t1 : (*ch * t1));
  *shr = (*shi * ar);
  *shi *= ai;
  /* sine/tangent underflows with a non-zero aa */
  er = ((er && (fabs(*shr) < DBL_MIN)) << 4);
  ei = ((ei && (fabs(*shi) < DBL_MIN)) << 5);
  *th = t1;
  bt = ((bt == -1) << 1);
  return (wt | bt | e1 | e2 | er | ei);
}

int PVN_FABI(pvn_xljv2,PVN_XLJV2)(const long double *const a11, const long double *const a22, const long double *const a21, long double *const ch, long double *const sh, long double *const th, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21);
  PVN_ASSERT(ch);
  PVN_ASSERT(sh);
  PVN_ASSERT(th);
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
  int wt = 0, bt = 0;
  switch (*es) {
  case -2:
    wt = 1;
    break;
  case -1:
    wt = 1;
    bt = 1;
    break;
  case 0:
    break;
  case 1:
    bt = 1;
    break;
  default:
    return -6;
  }
  int
    e1 = (a1 != 0.0L),
    e2 = (a2 != 0.0L),
    er = (ar != 0.0L);
  long double t1 = 0.0L, aa = 0.0L;
  if (er) {
    *es = (e1 | (e2 << 1) | (er << 2));
    if (*es) {
      (void)frexpl(fmaxl(a1, LDBL_TRUE_MIN), &e1);
      (void)frexpl(fmaxl(a2, LDBL_TRUE_MIN), &e2);
      (void)frexpl(fmaxl(fabsl(ar), LDBL_TRUE_MIN), &er);
      e1 = pvn_imax3(e1, e2, er);
      e2 = *es;
      *es = (LDBL_BIG_EXP - e1);
      a1 = scalbnl(a1, *es);
      a2 = scalbnl(a2, *es);
      ar = scalbnl(ar, *es);
      er = (ar != 0.0L);
      *es = -*es;
    }
    if (er) {
      aa = fabsl(ar);
      /* a non-zero element underflows due to scaling */
      e1 = ((((e2 & 1) && (a1 < LDBL_MIN)) || ((e2 & 2) && (a2 < LDBL_MIN)) || ((e2 & 4) && (aa < LDBL_MIN))) << 2);
      ar = copysignl(1.0L, ar);
      const long double
        an = (2.0L * aa),
        ad = (a1 + a2),
        t2 = -((an >= ad) ? 1.0L : (an / ad));
      t1 = (t2 / (1.0L + sqrtl(fmal(-t2, t2, 1.0L))));
      if (bt && (fabsl(t1) > *th)) {
        t1 = copysignl(*th, t1);
        bt = -1;
      }
      /* tangent underflows */
      e2 = ((fabsl(t1) < LDBL_MIN) << 3);
    }
    else
      e2 = e1 = 0;
  }
  else
    *es = e2 = e1 = 0;
  *ch = rsqrtl(fmal(-t1, t1, 1.0L));
  *sh = (wt ? t1 : (*ch * t1)) * ar;
  /* sine/tangent underflows with a non-zero aa */
  er = ((er && (fabsl(*sh) < LDBL_MIN)) << 4);
  *th = t1;
  bt = ((bt == -1) << 1);
  return (wt | bt | e1 | e2 | er);
}

int PVN_FABI(pvn_wljv2,PVN_WLJV2)(const long double *const a11, const long double *const a22, const long double *const a21r, const long double *const a21i, long double *const ch, long double *const shr, long double *const shi, long double *const th, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21r);
  PVN_ASSERT(a21i);
  PVN_ASSERT(ch);
  PVN_ASSERT(shr);
  PVN_ASSERT(shi);
  PVN_ASSERT(th);
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
  int wt = 0, bt = 0;
  switch (*es) {
  case -2:
    wt = 1;
    break;
  case -1:
    wt = 1;
    bt = 1;
    break;
  case 0:
    break;
  case 1:
    bt = 1;
    break;
  default:
    return -8;
  }
  int
    e1 = (a1 != 0.0L),
    e2 = (a2 != 0.0L),
    er = (ar != 0.0L),
    ei = (ai != 0.0L);
  long double t1 = 0.0L, aa = 0.0L;
  if (er || ei) {
    *es = (e1 | (e2 << 1) | (er << 2) | (ei << 3));
    if (*es) {
      (void)frexpl(fmaxl(a1, LDBL_TRUE_MIN), &e1);
      (void)frexpl(fmaxl(a2, LDBL_TRUE_MIN), &e2);
      (void)frexpl(fmaxl(fabsl(ar), LDBL_TRUE_MIN), &er);
      (void)frexpl(fmaxl(fabsl(ai), LDBL_TRUE_MIN), &ei);
      e1 = pvn_imax4(e1, e2, er, ei);
      e2 = *es;
      *es = (LDBL_BIG_EXP - e1);
      a1 = scalbnl(a1, *es);
      a2 = scalbnl(a2, *es);
      ar = scalbnl(ar, *es);
      ai = scalbnl(ai, *es);
      er = (ar != 0.0L);
      ei = (ai != 0.0L);
      *es = -*es;
    }
    if (er || ei) {
      const long double
        ar_ = fabsl(ar),
        ai_ = fabsl(ai);
      aa = hypotl(ar_, ai_); /* aa cannot be zero here if hypotl is correctly rounded */
      /* a non-zero element underflows due to scaling */
      e1 = ((((e2 & 1) && (a1 < LDBL_MIN)) || ((e2 & 2) && (a2 < LDBL_MIN)) || ((e2 & 4) && (ar_ < LDBL_MIN)) || ((e2 & 8) && (ai_ < LDBL_MIN))) << 2);
      ar = (ar / aa);
      ai = (ai / aa);
      const long double
        an = (2.0L * aa),
        ad = (a1 + a2),
        t2 = -((an >= ad) ? 1.0L : (an / ad));
      t1 = (t2 / (1.0L + sqrtl(fmal(-t2, t2, 1.0L))));
      if (bt && (fabsl(t1) > *th)) {
        t1 = copysignl(*th, t1);
        bt = -1;
      }
      /* tangent underflows */
      e2 = ((fabsl(t1) < LDBL_MIN) << 3);
    }
    else
      e2 = e1 = 0;
  }
  else
    *es = e2 = e1 = 0;
  *ch = rsqrtl(fmal(-t1, t1, 1.0L));
  *shi = (wt ? t1 : (*ch * t1));
  *shr = (*shi * ar);
  *shi *= ai;
  /* sine/tangent underflows with a non-zero aa */
  er = ((er && (fabsl(*shr) < LDBL_MIN)) << 4);
  ei = ((ei && (fabsl(*shi) < LDBL_MIN)) << 5);
  *th = t1;
  bt = ((bt == -1) << 1);
  return (wt | bt | e1 | e2 | er | ei);
}

#ifdef PVN_QUADMATH
#ifdef FLT128_BIG_EXP
#error FLT128_BIG_EXP already defined
#else /* !FLT128_BIG_EXP */
#define FLT128_BIG_EXP (FLT128_MAX_EXP - 2)
#endif /* ?FLT128_BIG_EXP */

int PVN_FABI(pvn_qljv2,PVN_QLJV2)(const __float128 *const a11, const __float128 *const a22, const __float128 *const a21, __float128 *const ch, __float128 *const sh, __float128 *const th, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21);
  PVN_ASSERT(ch);
  PVN_ASSERT(sh);
  PVN_ASSERT(th);
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
  int wt = 0, bt = 0;
  switch (*es) {
  case -2:
    wt = 1;
    break;
  case -1:
    wt = 1;
    bt = 1;
    break;
  case 0:
    break;
  case 1:
    bt = 1;
    break;
  default:
    return -6;
  }
  int
    e1 = (a1 != 0.0q),
    e2 = (a2 != 0.0q),
    er = (ar != 0.0q);
  __float128 t1 = 0.0q, aa = 0.0q;
  if (er) {
    *es = (e1 | (e2 << 1) | (er << 2));
    if (*es) {
      (void)frexpq(fmaxq(a1, FLT128_TRUE_MIN), &e1);
      (void)frexpq(fmaxq(a2, FLT128_TRUE_MIN), &e2);
      (void)frexpq(fmaxq(fabsq(ar), FLT128_TRUE_MIN), &er);
      e1 = pvn_imax3(e1, e2, er);
      e2 = *es;
      *es = (FLT128_BIG_EXP - e1);
      a1 = scalbnq(a1, *es);
      a2 = scalbnq(a2, *es);
      ar = scalbnq(ar, *es);
      er = (ar != 0.0q);
      *es = -*es;
    }
    if (er) {
      aa = fabsq(ar);
      /* a non-zero element underflows due to scaling */
      e1 = ((((e2 & 1) && (a1 < FLT128_MIN)) || ((e2 & 2) && (a2 < FLT128_MIN)) || ((e2 & 4) && (aa < FLT128_MIN))) << 2);
      ar = copysignq(1.0q, ar);
      const __float128
        an = (2.0q * aa),
        ad = (a1 + a2),
        t2 = -((an >= ad) ? 1.0q : (an / ad));
      t1 = (t2 / (1.0q + sqrtq(fmaq(-t2, t2, 1.0q))));
      if (bt && (fabsq(t1) > *th)) {
        t1 = copysignq(*th, t1);
        bt = -1;
      }
      /* tangent underflows */
      e2 = ((fabsq(t1) < FLT128_MIN) << 3);
    }
    else
      e2 = e1 = 0;
  }
  else
    *es = e2 = e1 = 0;
  *ch = rsqrtq(fmaq(-t1, t1, 1.0q));
  *sh = (wt ? t1 : (*ch * t1)) * ar;
  /* sine/tangent underflows with a non-zero aa */
  er = ((er && (fabsq(*sh) < FLT128_MIN)) << 4);
  *th = t1;
  bt = ((bt == -1) << 1);
  return (wt | bt | e1 | e2 | er);
}

int PVN_FABI(pvn_yljv2,PVN_YLJV2)(const __float128 *const a11, const __float128 *const a22, const __float128 *const a21r, const __float128 *const a21i, __float128 *const ch, __float128 *const shr, __float128 *const shi, __float128 *const th, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a22);
  PVN_ASSERT(a21r);
  PVN_ASSERT(a21i);
  PVN_ASSERT(ch);
  PVN_ASSERT(shr);
  PVN_ASSERT(shi);
  PVN_ASSERT(th);
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
  int wt = 0, bt = 0;
  switch (*es) {
  case -2:
    wt = 1;
    break;
  case -1:
    wt = 1;
    bt = 1;
    break;
  case 0:
    break;
  case 1:
    bt = 1;
    break;
  default:
    return -8;
  }
  int
    e1 = (a1 != 0.0q),
    e2 = (a2 != 0.0q),
    er = (ar != 0.0q),
    ei = (ai != 0.0q);
  __float128 t1 = 0.0q, aa = 0.0q;
  if (er || ei) {
    *es = (e1 | (e2 << 1) | (er << 2) | (ei << 3));
    if (*es) {
      (void)frexpq(fmaxq(a1, FLT128_TRUE_MIN), &e1);
      (void)frexpq(fmaxq(a2, FLT128_TRUE_MIN), &e2);
      (void)frexpq(fmaxq(fabsq(ar), FLT128_TRUE_MIN), &er);
      (void)frexpq(fmaxq(fabsq(ai), FLT128_TRUE_MIN), &ei);
      e1 = pvn_imax4(e1, e2, er, ei);
      e2 = *es;
      *es = (FLT128_BIG_EXP - e1);
      a1 = scalbnq(a1, *es);
      a2 = scalbnq(a2, *es);
      ar = scalbnq(ar, *es);
      ai = scalbnq(ai, *es);
      er = (ar != 0.0q);
      ei = (ai != 0.0q);
      *es = -*es;
    }
    if (er || ei) {
      const __float128
        ar_ = fabsq(ar),
        ai_ = fabsq(ai);
      aa = hypotq(ar_, ai_);
      if ((aa <= 0.0q) || !isfiniteq(aa)) {
        *ch = aa;
        return -9;
      }
      /* a non-zero element underflows due to scaling */
      e1 = ((((e2 & 1) && (a1 < FLT128_MIN)) || ((e2 & 2) && (a2 < FLT128_MIN)) || ((e2 & 4) && (ar_ < FLT128_MIN)) || ((e2 & 8) && (ai_ < FLT128_MIN))) << 2);
      ar = (ar / aa);
      ai = (ai / aa);
      const __float128
        an = (2.0q * aa),
        ad = (a1 + a2),
        t2 = -((an >= ad) ? 1.0q : (an / ad));
      t1 = (t2 / (1.0q + sqrtq(fmaq(-t2, t2, 1.0q))));
      if (bt && (fabsq(t1) > *th)) {
        t1 = copysignq(*th, t1);
        bt = -1;
      }
      /* tangent underflows */
      e2 = ((fabsq(t1) < FLT128_MIN) << 3);
    }
    else
      e2 = e1 = 0;
  }
  else
    *es = e2 = e1 = 0;
  *ch = rsqrtq(fmaq(-t1, t1, 1.0q));
  *shi = (wt ? t1 : (*ch * t1));
  *shr = (*shi * ar);
  *shi *= ai;
  /* sine/tangent underflows with a non-zero aa */
  er = ((er && (fabsq(*shr) < FLT128_MIN)) << 4);
  ei = ((ei && (fabsq(*shi) < FLT128_MIN)) << 5);
  *th = t1;
  bt = ((bt == -1) << 1);
  return (wt | bt | e1 | e2 | er | ei);
}
#else /* !PVN_QUADMATH */
int PVN_FABI(pvn_qljv2,PVN_QLJV2)(const long double *const a11, const long double *const a22, const long double *const a21, long double *const ch, long double *const sh, long double *const th, int *const es)
{
  return PVN_FABI(pvn_xljv2,PVN_XLJV2)(a11, a22, a21, ch, sh, th, es);
}

int PVN_FABI(pvn_yljv2,PVN_YLJV2)(const long double *const a11, const long double *const a22, const long double *const a21r, const long double *const a21i, long double *const ch, long double *const shr, long double *const shi, long double *const th, int *const es)
{
  return PVN_FABI(pvn_wljv2,PVN_WLJV2)(a11, a22, a21r, a21i, ch, shr, shi, th, es);
}
#endif /* ?PVN_QUADMATH */
#endif /* ?PVN_TEST */
