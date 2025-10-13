#include "pvn.h"

#ifdef PVN_TEST
int main(int argc, char *argv[])
{
  if (1 < argc) {
    (void)fprintf(stderr, "%s takes no arguments\n", *argv);
    return EXIT_FAILURE;
  }
  const double x[4][2] = {{-1.0, 1.0}, { 1.0,-1.0}, { 2.0, 2.0}, {-2.0,-2.0}};
  const size_t n = (size_t)4u;
  /* expected output: 4.472136 */
  (void)printf("%#F\n", PVN_FABI(pvn_znrm2,PVN_ZNRM2)(&n, &(x[0][0])));
  return EXIT_SUCCESS;
}
#else /* !PVN_TEST */
#ifdef PVN_MPFR
float PVN_FABI(pvn_mps_nrmp,PVN_MPS_NRMP)(const float *const p, const size_t *const n, const float *const x)
{
  if (!p || !(*p > 0.0f))
    return -1.0f;
  if (!n)
    return -2.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -3.0f;
  const size_t m = *n;
  float f = 0.0f;
  mpfr_t mf, mx;
  (void)mpfr_init_set_d(mf, 0.0, MPFR_RNDN);
  (void)mpfr_init_set_d(mx, 0.0, MPFR_RNDN);
  if (*p == 1.0f) {
    for (size_t i = 0u; i < m; ++i) {
      (void)mpfr_set_flt(mx, x[i], MPFR_RNDN);
      (void)mpfr_abs(mx, mx, MPFR_RNDN);
      (void)mpfr_add(mf, mf, mx, MPFR_RNDN);
    }
  }
  else if (*p == 2.0f) {
    for (size_t i = 0u; i < m; ++i) {
      (void)mpfr_set_flt(mx, x[i], MPFR_RNDN);
      (void)mpfr_fma(mf, mx, mx, mf, MPFR_RNDN);
    }
    (void)mpfr_sqrt(mf, mf, MPFR_RNDN);
  }
  else if (isinf(*p)) {
    for (size_t i = 0u; i < m; ++i) {
      (void)mpfr_set_flt(mx, x[i], MPFR_RNDN);
      (void)mpfr_abs(mx, mx, MPFR_RNDN);
      (void)mpfr_max(mf, mf, mx, MPFR_RNDN);
    }
  }
  else {
    mpfr_t mp;
    (void)mpfr_init_set_d(mp, 0.0, MPFR_RNDN);
    (void)mpfr_set_flt(mf, *p, MPFR_RNDN);
    for (size_t i = 0u; i < m; ++i) {
      (void)mpfr_set_flt(mx, x[i], MPFR_RNDN);
      (void)mpfr_abs(mx, mx, MPFR_RNDN);
      (void)mpfr_pow(mx, mx, mp, MPFR_RNDN);
      (void)mpfr_add(mf, mf, mx, MPFR_RNDN);
    }
    (void)mpfr_d_div(mp, 1.0, mp, MPFR_RNDN);
    (void)mpfr_pow(mf, mf, mp, MPFR_RNDN);
    mpfr_clear(mp);
  }
  f = mpfr_get_flt(mf, MPFR_RNDN);
  mpfr_clear(mx);
  mpfr_clear(mf);
  return f;
}

double PVN_FABI(pvn_mpd_nrmp,PVN_MPD_NRMP)(const double *const p, const size_t *const n, const double *const x)
{
  if (!p || !(*p > 0.0))
    return -1.0;
  if (!n)
    return -2.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -3.0;
  const size_t m = *n;
  double f = 0.0;
  mpfr_t mf, mx;
  (void)mpfr_init_set_d(mf, 0.0, MPFR_RNDN);
  (void)mpfr_init_set_d(mx, 0.0, MPFR_RNDN);
  if (*p == 1.0) {
    for (size_t i = 0u; i < m; ++i) {
      (void)mpfr_set_d(mx, x[i], MPFR_RNDN);
      (void)mpfr_abs(mx, mx, MPFR_RNDN);
      (void)mpfr_add(mf, mf, mx, MPFR_RNDN);
    }
  }
  else if (*p == 2.0) {
    for (size_t i = 0u; i < m; ++i) {
      (void)mpfr_set_d(mx, x[i], MPFR_RNDN);
      (void)mpfr_fma(mf, mx, mx, mf, MPFR_RNDN);
    }
    (void)mpfr_sqrt(mf, mf, MPFR_RNDN);
  }
  else if (isinf(*p)) {
    for (size_t i = 0u; i < m; ++i) {
      (void)mpfr_set_d(mx, x[i], MPFR_RNDN);
      (void)mpfr_abs(mx, mx, MPFR_RNDN);
      (void)mpfr_max(mf, mf, mx, MPFR_RNDN);
    }
  }
  else {
    mpfr_t mp;
    (void)mpfr_init_set_d(mp, *p, MPFR_RNDN);
    for (size_t i = 0u; i < m; ++i) {
      (void)mpfr_set_d(mx, x[i], MPFR_RNDN);
      (void)mpfr_abs(mx, mx, MPFR_RNDN);
      (void)mpfr_pow(mx, mx, mp, MPFR_RNDN);
      (void)mpfr_add(mf, mf, mx, MPFR_RNDN);
    }
    (void)mpfr_d_div(mp, 1.0, mp, MPFR_RNDN);
    (void)mpfr_pow(mf, mf, mp, MPFR_RNDN);
    mpfr_clear(mp);
  }
  f = mpfr_get_d(mf, MPFR_RNDN);
  mpfr_clear(mx);
  mpfr_clear(mf);
  return f;
}

long double PVN_FABI(pvn_mpx_nrmp,PVN_MPX_NRMP)(const long double *const p, const size_t *const n, const long double *const x)
{
  if (!p || !(*p > 0.0L))
    return -1.0L;
  if (!n)
    return -2.0L;
  if (!*n)
    return -0.0L;
  if (!x)
    return -3.0L;
  const size_t m = *n;
  long double f = 0.0L;
  mpfr_t mf, mx;
  (void)mpfr_init_set_ld(mf, 0.0L, MPFR_RNDN);
  (void)mpfr_init_set_ld(mx, 0.0L, MPFR_RNDN);
  if (*p == 1.0L) {
    for (size_t i = 0u; i < m; ++i) {
      (void)mpfr_set_ld(mx, x[i], MPFR_RNDN);
      (void)mpfr_abs(mx, mx, MPFR_RNDN);
      (void)mpfr_add(mf, mf, mx, MPFR_RNDN);
    }
  }
  else if (*p == 2.0L) {
    for (size_t i = 0u; i < m; ++i) {
      (void)mpfr_set_ld(mx, x[i], MPFR_RNDN);
      (void)mpfr_fma(mf, mx, mx, mf, MPFR_RNDN);
    }
    (void)mpfr_sqrt(mf, mf, MPFR_RNDN);
  }
  else if (isinf(*p)) {
    for (size_t i = 0u; i < m; ++i) {
      (void)mpfr_set_ld(mx, x[i], MPFR_RNDN);
      (void)mpfr_abs(mx, mx, MPFR_RNDN);
      (void)mpfr_max(mf, mf, mx, MPFR_RNDN);
    }
  }
  else {
    mpfr_t mp;
    (void)mpfr_init_set_ld(mp, *p, MPFR_RNDN);
    for (size_t i = 0u; i < m; ++i) {
      (void)mpfr_set_ld(mx, x[i], MPFR_RNDN);
      (void)mpfr_abs(mx, mx, MPFR_RNDN);
      (void)mpfr_pow(mx, mx, mp, MPFR_RNDN);
      (void)mpfr_add(mf, mf, mx, MPFR_RNDN);
    }
    (void)mpfr_d_div(mp, 1.0, mp, MPFR_RNDN);
    (void)mpfr_pow(mf, mf, mp, MPFR_RNDN);
    mpfr_clear(mp);
  }
  f = mpfr_get_ld(mf, MPFR_RNDN);
  mpfr_clear(mx);
  mpfr_clear(mf);
  return f;
}
#endif /* PVN_MPFR */

float PVN_FABI(pvn_res_nrmf,PVN_RES_NRMF)(const size_t *const n, const float *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -2.0f;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return __builtin_fabsf(*x);
  if (*n == (size_t)2u)
    return hypotf(x[0u], x[1u]);
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  float fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_res_nrmf,PVN_RES_NRMF)(&nl, x);
    fr = PVN_FABI(pvn_res_nrmf,PVN_RES_NRMF)(&nr, (x + nl));
  }
  return hypotf(fl, fr);
}

double PVN_FABI(pvn_red_nrmf,PVN_RED_NRMF)(const size_t *const n, const double *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -2.0;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return __builtin_fabs(*x);
  if (*n == (size_t)2u)
    return hypot(x[0u], x[1u]);
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  double fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_red_nrmf,PVN_RED_NRMF)(&nl, x);
    fr = PVN_FABI(pvn_red_nrmf,PVN_RED_NRMF)(&nr, (x + nl));
  }
  return hypot(fl, fr);
}

long double PVN_FABI(pvn_rex_nrmf,PVN_REX_NRMF)(const size_t *const n, const long double *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0L;
  if (!*n)
    return -0.0L;
  if (!x)
    return -2.0L;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return __builtin_fabsl(*x);
  if (*n == (size_t)2u)
    return hypotl(x[0u], x[1u]);
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  long double fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_rex_nrmf,PVN_REX_NRMF)(&nl, x);
    fr = PVN_FABI(pvn_rex_nrmf,PVN_REX_NRMF)(&nr, (x + nl));
  }
  return hypotl(fl, fr);
}

float PVN_FABI(pvn_res_nrm1,PVN_RES_NRM1)(const size_t *const n, const float *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -2.0f;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return __builtin_fabsf(*x);
  if (*n == (size_t)2u)
    return (__builtin_fabsf(x[0u]) + __builtin_fabsf(x[1u]));
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  float fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_res_nrm1,PVN_RES_NRM1)(&nl, x);
    fr = PVN_FABI(pvn_res_nrm1,PVN_RES_NRM1)(&nr, (x + nl));
  }
  return (fl + fr);
}

double PVN_FABI(pvn_red_nrm1,PVN_RED_NRM1)(const size_t *const n, const double *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -2.0;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return __builtin_fabs(*x);
  if (*n == (size_t)2u)
    return (__builtin_fabs(x[0u]) + __builtin_fabs(x[1u]));
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  double fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_red_nrm1,PVN_RED_NRM1)(&nl, x);
    fr = PVN_FABI(pvn_red_nrm1,PVN_RED_NRM1)(&nr, (x + nl));
  }
  return (fl + fr);
}

long double PVN_FABI(pvn_rex_nrm1,PVN_REX_NRM1)(const size_t *const n, const long double *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0L;
  if (!*n)
    return -0.0L;
  if (!x)
    return -2.0L;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return __builtin_fabsl(*x);
  if (*n == (size_t)2u)
    return (__builtin_fabsl(x[0u]) + __builtin_fabsl(x[1u]));
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  long double fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_rex_nrm1,PVN_REX_NRM1)(&nl, x);
    fr = PVN_FABI(pvn_rex_nrm1,PVN_REX_NRM1)(&nr, (x + nl));
  }
  return (fl + fr);
}

float PVN_FABI(pvn_res_nrmi,PVN_RES_NRMI)(const size_t *const n, const float *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -2.0f;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return __builtin_fabsf(*x);
  if (*n == (size_t)2u)
    return __builtin_fmaxf(__builtin_fabsf(x[0u]), __builtin_fabsf(x[1u]));
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  float fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_res_nrmi,PVN_RES_NRMI)(&nl, x);
    fr = PVN_FABI(pvn_res_nrmi,PVN_RES_NRMI)(&nr, (x + nl));
  }
  return __builtin_fmaxf(fl, fr);
}

double PVN_FABI(pvn_red_nrmi,PVN_RED_NRMI)(const size_t *const n, const double *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -2.0;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return __builtin_fabs(*x);
  if (*n == (size_t)2u)
    return __builtin_fmax(__builtin_fabs(x[0u]), __builtin_fabs(x[1u]));
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  double fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_red_nrmi,PVN_RED_NRMI)(&nl, x);
    fr = PVN_FABI(pvn_red_nrmi,PVN_RED_NRMI)(&nr, (x + nl));
  }
  return __builtin_fmax(fl, fr);
}

long double PVN_FABI(pvn_rex_nrmi,PVN_REX_NRMI)(const size_t *const n, const long double *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0L;
  if (!*n)
    return -0.0L;
  if (!x)
    return -2.0L;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return __builtin_fabsl(*x);
  if (*n == (size_t)2u)
    return __builtin_fmaxl(__builtin_fabsl(x[0u]), __builtin_fabsl(x[1u]));
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  long double fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_rex_nrmi,PVN_REX_NRMI)(&nl, x);
    fr = PVN_FABI(pvn_rex_nrmi,PVN_REX_NRMI)(&nr, (x + nl));
  }
  return __builtin_fmaxl(fl, fr);
}

float PVN_FABI(pvn_res_nrmp,PVN_RES_NRMP)(const float *const p, const size_t *const n, const float *const x)
{
#ifndef NDEBUG
  if (!p || !(*p > 0.0f))
    return -1.0f;
  if (!n)
    return -2.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -3.0f;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return __builtin_fabsf(*x);
  if (*n == (size_t)2u)
    return pvn_v1s_lp(*p, x[0u], x[1u]);
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  float fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_res_nrmp,PVN_RES_NRMP)(p, &nl, x);
    fr = PVN_FABI(pvn_res_nrmp,PVN_RES_NRMP)(p, &nr, (x + nl));
  }
  return pvn_v1s_lp(*p, fl, fr);
}

double PVN_FABI(pvn_red_nrmp,PVN_RED_NRMP)(const double *const p, const size_t *const n, const double *const x)
{
#ifndef NDEBUG
  if (!p || !(*p > 0.0))
    return -1.0;
  if (!n)
    return -2.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -3.0;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return __builtin_fabs(*x);
  if (*n == (size_t)2u)
    return pvn_v1d_lp(*p, x[0u], x[1u]);
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  double fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_red_nrmp,PVN_RED_NRMP)(p, &nl, x);
    fr = PVN_FABI(pvn_red_nrmp,PVN_RED_NRMP)(p, &nr, (x + nl));
  }
  return pvn_v1d_lp(*p, fl, fr);
}

long double PVN_FABI(pvn_rex_nrmp,PVN_REX_NRMP)(const long double *const p, const size_t *const n, const long double *const x)
{
#ifndef NDEBUG
  if (!p || !(*p > 0.0L))
    return -1.0L;
  if (!n)
    return -2.0L;
  if (!*n)
    return -0.0L;
  if (!x)
    return -3.0L;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return __builtin_fabsl(*x);
  if (*n == (size_t)2u)
    return pvn_v1x_lp(*p, x[0u], x[1u]);
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  long double fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_rex_nrmp,PVN_REX_NRMP)(p, &nl, x);
    fr = PVN_FABI(pvn_rex_nrmp,PVN_REX_NRMP)(p, &nr, (x + nl));
  }
  return pvn_v1x_lp(*p, fl, fr);
}

float PVN_FABI(pvn_rfs_nrmf,PVN_RFS_NRMF)(const size_t *const n, const float *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -2.0f;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return __builtin_fabsf(*x);
  if (*n == (size_t)2u)
    return __builtin_hypotf(x[0u], x[1u]);
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  float fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_rfs_nrmf,PVN_RFS_NRMF)(&nl, x);
    fr = PVN_FABI(pvn_rfs_nrmf,PVN_RFS_NRMF)(&nr, (x + nl));
  }
  return __builtin_hypotf(fl, fr);
}

double PVN_FABI(pvn_rfd_nrmf,PVN_RFD_NRMF)(const size_t *const n, const double *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -2.0;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return __builtin_fabs(*x);
  if (*n == (size_t)2u)
    return __builtin_hypot(x[0u], x[1u]);
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  double fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_rfd_nrmf,PVN_RFD_NRMF)(&nl, x);
    fr = PVN_FABI(pvn_rfd_nrmf,PVN_RFD_NRMF)(&nr, (x + nl));
  }
  return __builtin_hypot(fl, fr);
}

long double PVN_FABI(pvn_rfx_nrmf,PVN_RFX_NRMF)(const size_t *const n, const long double *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0L;
  if (!*n)
    return -0.0L;
  if (!x)
    return -2.0L;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return __builtin_fabsl(*x);
  if (*n == (size_t)2u)
    return __builtin_hypotl(x[0u], x[1u]);
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  long double fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_rfx_nrmf,PVN_RFX_NRMF)(&nl, x);
    fr = PVN_FABI(pvn_rfx_nrmf,PVN_RFX_NRMF)(&nr, (x + nl));
  }
  return __builtin_hypotl(fl, fr);
}

float PVN_FABI(pvn_rhs_nrmf,PVN_RHS_NRMF)(const size_t *const n, const float *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -2.0f;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return __builtin_fabsf(*x);
  if (*n == (size_t)2u)
    return pvn_v1s_hypot(x[0u], x[1u]);
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  float fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_rhs_nrmf,PVN_RHS_NRMF)(&nl, x);
    fr = PVN_FABI(pvn_rhs_nrmf,PVN_RHS_NRMF)(&nr, (x + nl));
  }
  return pvn_v1s_hypot(fl, fr);
}

double PVN_FABI(pvn_rhd_nrmf,PVN_RHD_NRMF)(const size_t *const n, const double *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -2.0;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return __builtin_fabs(*x);
  if (*n == (size_t)2u)
    return pvn_v1d_hypot(x[0u], x[1u]);
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  double fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_rhd_nrmf,PVN_RHD_NRMF)(&nl, x);
    fr = PVN_FABI(pvn_rhd_nrmf,PVN_RHD_NRMF)(&nr, (x + nl));
  }
  return pvn_v1d_hypot(fl, fr);
}

long double PVN_FABI(pvn_rhx_nrmf,PVN_RHX_NRMF)(const size_t *const n, const long double *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0L;
  if (!*n)
    return -0.0L;
  if (!x)
    return -2.0L;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return __builtin_fabsl(*x);
  if (*n == (size_t)2u)
    return pvn_v1x_hypot(x[0u], x[1u]);
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  long double fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_rhx_nrmf,PVN_RHX_NRMF)(&nl, x);
    fr = PVN_FABI(pvn_rhx_nrmf,PVN_RHX_NRMF)(&nr, (x + nl));
  }
  return pvn_v1x_hypot(fl, fr);
}

float PVN_FABI(pvn_snrm2,PVN_SNRM2)(const size_t *const n, const float *const x)
{
  if (!n)
    return -1.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -2.0f;
#ifdef _OPENMP
  const size_t mt = (size_t)omp_get_max_threads();
  float p[mt];
  for (size_t i = 0u; i < mt; ++i)
    p[i] = 0.0f;
#pragma omp parallel default(none) shared(n,x,p)
  {
    const size_t nt = (size_t)omp_get_num_threads();
    const size_t tn = (size_t)omp_get_thread_num();
    lldiv_t qr = lldiv((long long)*n, (long long)nt);
    const size_t q = (size_t)(qr.quot);
    const size_t r = (size_t)(qr.rem);
    size_t m = q, o = (tn * q);
    if (r) {
      if (tn < r) {
        ++m;
        o += tn;
      }
      else
        o += r;
    }
    if (m)
#ifdef PVN_NRM_SAFE
      p[tn] = PVN_FABI(pvn_res_nrmf,PVN_RES_NRMF)(&m, (x + o));
#else /* !PVN_NRM_SAFE */
      p[tn] = PVN_FABI(pvn_rfs_nrmf,PVN_RFS_NRMF)(&m, (x + o));
#endif /* ?PVN_NRM_SAFE */
  }
#ifdef PVN_NRM_SAFE
  return PVN_FABI(pvn_res_nrmf,PVN_RES_NRMF)(&mt, p);
#else /* !PVN_NRM_SAFE */
  return PVN_FABI(pvn_rfs_nrmf,PVN_RFS_NRMF)(&mt, p);
#endif /* ?PVN_NRM_SAFE */
#else /* !_OPENMP */
#ifdef PVN_NRM_SAFE
  return PVN_FABI(pvn_res_nrmf,PVN_RES_NRMF)(n, x);
#else /* !PVN_NRM_SAFE */
  return PVN_FABI(pvn_rfs_nrmf,PVN_RFS_NRMF)(n, x);
#endif /* ?PVN_NRM_SAFE */
#endif /* _OPENMP */
}

float PVN_FABI(pvn_cnrm2,PVN_CNRM2)(const size_t *const n, const float *const x)
{
  const size_t m = (n ? (*n << 1u) : (size_t)0u);
  return PVN_FABI(pvn_snrm2,PVN_SNRM2)(&m, x);
}

double PVN_FABI(pvn_dnrm2,PVN_DNRM2)(const size_t *const n, const double *const x)
{
  if (!n)
    return -1.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -2.0;
#ifdef _OPENMP
  const size_t mt = (size_t)omp_get_max_threads();
  double p[mt];
  for (size_t i = 0u; i < mt; ++i)
    p[i] = 0.0;
#pragma omp parallel default(none) shared(n,x,p)
  {
    const size_t nt = (size_t)omp_get_num_threads();
    const size_t tn = (size_t)omp_get_thread_num();
    lldiv_t qr = lldiv((long long)*n, (long long)nt);
    const size_t q = (size_t)(qr.quot);
    const size_t r = (size_t)(qr.rem);
    size_t m = q, o = (tn * q);
    if (r) {
      if (tn < r) {
        ++m;
        o += tn;
      }
      else
        o += r;
    }
    if (m)
#ifdef PVN_NRM_SAFE
      p[tn] = PVN_FABI(pvn_red_nrmf,PVN_RED_NRMF)(&m, (x + o));
#else /* !PVN_NRM_SAFE */
      p[tn] = PVN_FABI(pvn_rfd_nrmf,PVN_RFD_NRMF)(&m, (x + o));
#endif /* ?PVN_NRM_SAFE */
  }
#ifdef PVN_NRM_SAFE
  return PVN_FABI(pvn_red_nrmf,PVN_RED_NRMF)(&mt, p);
#else /* !PVN_NRM_SAFE */
  return PVN_FABI(pvn_rfd_nrmf,PVN_RFD_NRMF)(&mt, p);
#endif /* ?PVN_NRM_SAFE */
#else /* !_OPENMP */
#ifdef PVN_NRM_SAFE
  return PVN_FABI(pvn_red_nrmf,PVN_RED_NRMF)(n, x);
#else /* !PVN_NRM_SAFE */
  return PVN_FABI(pvn_rfd_nrmf,PVN_RFD_NRMF)(n, x);
#endif /* ?PVN_NRM_SAFE */
#endif /* ?_OPENMP */
}

double PVN_FABI(pvn_znrm2,PVN_ZNRM2)(const size_t *const n, const double *const x)
{
  const size_t m = (n ? (*n << 1u) : (size_t)0u);
  return PVN_FABI(pvn_dnrm2,PVN_DNRM2)(&m, x);
}

long double PVN_FABI(pvn_xnrm2,PVN_XNRM2)(const size_t *const n, const long double *const x)
{
  if (!n)
    return -1.0L;
  if (!*n)
    return -0.0L;
  if (!x)
    return -2.0L;
#ifdef _OPENMP
  const size_t mt = (size_t)omp_get_max_threads();
  long double p[mt];
  for (size_t i = 0u; i < mt; ++i)
    p[i] = 0.0L;
#pragma omp parallel default(none) shared(n,x,p)
  {
    const size_t nt = (size_t)omp_get_num_threads();
    const size_t tn = (size_t)omp_get_thread_num();
    lldiv_t qr = lldiv((long long)*n, (long long)nt);
    const size_t q = (size_t)(qr.quot);
    const size_t r = (size_t)(qr.rem);
    size_t m = q, o = (tn * q);
    if (r) {
      if (tn < r) {
        ++m;
        o += tn;
      }
      else
        o += r;
    }
    if (m)
#ifdef PVN_NRM_SAFE
      p[tn] = PVN_FABI(pvn_rex_nrmf,PVN_REX_NRMF)(&m, (x + o));
#else /* !PVN_NRM_SAFE */
      p[tn] = PVN_FABI(pvn_rfx_nrmf,PVN_RFX_NRMF)(&m, (x + o));
#endif /* ?PVN_NRM_SAFE */
  }
#ifdef PVN_NRM_SAFE
  return PVN_FABI(pvn_rex_nrmf,PVN_REX_NRMF)(&mt, p);
#else /* !PVN_NRM_SAFE */
  return PVN_FABI(pvn_rfx_nrmf,PVN_RFX_NRMF)(&mt, p);
#endif /* ?PVN_NRM_SAFE */
#else /* !_OPENMP */
#ifdef PVN_NRM_SAFE
  return PVN_FABI(pvn_rex_nrmf,PVN_REX_NRMF)(n, x);
#else /* !PVN_NRM_SAFE */
  return PVN_FABI(pvn_rfx_nrmf,PVN_RFX_NRMF)(n, x);
#endif /* ?PVN_NRM_SAFE */
#endif /* ?_OPENMP */
}

long double PVN_FABI(pvn_wnrm2,PVN_WNRM2)(const size_t *const n, const long double *const x)
{
  const size_t m = (n ? (*n << 1u) : (size_t)0u);
  return PVN_FABI(pvn_xnrm2,PVN_XNRM2)(&m, x);
}

float PVN_FABI(pvn_snrm1,PVN_SNRM1)(const size_t *const n, const float *const x)
{
  if (!n)
    return -1.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -2.0f;
#ifdef _OPENMP
  const size_t mt = (size_t)omp_get_max_threads();
  float p[mt];
  for (size_t i = 0u; i < mt; ++i)
    p[i] = 0.0f;
#pragma omp parallel default(none) shared(n,x,p)
  {
    const size_t nt = (size_t)omp_get_num_threads();
    const size_t tn = (size_t)omp_get_thread_num();
    lldiv_t qr = lldiv((long long)*n, (long long)nt);
    const size_t q = (size_t)(qr.quot);
    const size_t r = (size_t)(qr.rem);
    size_t m = q, o = (tn * q);
    if (r) {
      if (tn < r) {
        ++m;
        o += tn;
      }
      else
        o += r;
    }
    if (m)
      p[tn] = PVN_FABI(pvn_res_nrm1,PVN_RES_NRM1)(&m, (x + o));
  }
  return PVN_FABI(pvn_res_nrm1,PVN_RES_NRM1)(&mt, p);
#else /* !_OPENMP */
  return PVN_FABI(pvn_res_nrm1,PVN_RES_NRM1)(n, x);
#endif /* _OPENMP */
}

double PVN_FABI(pvn_dnrm1,PVN_DNRM1)(const size_t *const n, const double *const x)
{
  if (!n)
    return -1.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -2.0;
#ifdef _OPENMP
  const size_t mt = (size_t)omp_get_max_threads();
  double p[mt];
  for (size_t i = 0u; i < mt; ++i)
    p[i] = 0.0;
#pragma omp parallel default(none) shared(n,x,p)
  {
    const size_t nt = (size_t)omp_get_num_threads();
    const size_t tn = (size_t)omp_get_thread_num();
    lldiv_t qr = lldiv((long long)*n, (long long)nt);
    const size_t q = (size_t)(qr.quot);
    const size_t r = (size_t)(qr.rem);
    size_t m = q, o = (tn * q);
    if (r) {
      if (tn < r) {
        ++m;
        o += tn;
      }
      else
        o += r;
    }
    if (m)
      p[tn] = PVN_FABI(pvn_red_nrm1,PVN_RED_NRM1)(&m, (x + o));
  }
  return PVN_FABI(pvn_red_nrm1,PVN_RED_NRM1)(&mt, p);
#else /* !_OPENMP */
  return PVN_FABI(pvn_red_nrm1,PVN_RED_NRM1)(n, x);
#endif /* ?_OPENMP */
}

long double PVN_FABI(pvn_xnrm1,PVN_XNRM1)(const size_t *const n, const long double *const x)
{
  if (!n)
    return -1.0L;
  if (!*n)
    return -0.0L;
  if (!x)
    return -2.0L;
#ifdef _OPENMP
  const size_t mt = (size_t)omp_get_max_threads();
  long double p[mt];
  for (size_t i = 0u; i < mt; ++i)
    p[i] = 0.0L;
#pragma omp parallel default(none) shared(n,x,p)
  {
    const size_t nt = (size_t)omp_get_num_threads();
    const size_t tn = (size_t)omp_get_thread_num();
    lldiv_t qr = lldiv((long long)*n, (long long)nt);
    const size_t q = (size_t)(qr.quot);
    const size_t r = (size_t)(qr.rem);
    size_t m = q, o = (tn * q);
    if (r) {
      if (tn < r) {
        ++m;
        o += tn;
      }
      else
        o += r;
    }
    if (m)
      p[tn] = PVN_FABI(pvn_rex_nrm1,PVN_REX_NRM1)(&m, (x + o));
  }
  return PVN_FABI(pvn_rex_nrm1,PVN_REX_NRM1)(&mt, p);
#else /* !_OPENMP */
  return PVN_FABI(pvn_rex_nrm1,PVN_REX_NRM1)(n, x);
#endif /* ?_OPENMP */
}

float PVN_FABI(pvn_snrmi,PVN_SNRMI)(const size_t *const n, const float *const x)
{
  if (!n)
    return -1.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -2.0f;
#ifdef _OPENMP
  const size_t mt = (size_t)omp_get_max_threads();
  float p[mt];
  for (size_t i = 0u; i < mt; ++i)
    p[i] = 0.0f;
#pragma omp parallel default(none) shared(n,x,p)
  {
    const size_t nt = (size_t)omp_get_num_threads();
    const size_t tn = (size_t)omp_get_thread_num();
    lldiv_t qr = lldiv((long long)*n, (long long)nt);
    const size_t q = (size_t)(qr.quot);
    const size_t r = (size_t)(qr.rem);
    size_t m = q, o = (tn * q);
    if (r) {
      if (tn < r) {
        ++m;
        o += tn;
      }
      else
        o += r;
    }
    if (m)
      p[tn] = PVN_FABI(pvn_res_nrmi,PVN_RES_NRMI)(&m, (x + o));
  }
  return PVN_FABI(pvn_res_nrmi,PVN_RES_NRMI)(&mt, p);
#else /* !_OPENMP */
  return PVN_FABI(pvn_res_nrmi,PVN_RES_NRMI)(n, x);
#endif /* _OPENMP */
}

double PVN_FABI(pvn_dnrmi,PVN_DNRMI)(const size_t *const n, const double *const x)
{
  if (!n)
    return -1.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -2.0;
#ifdef _OPENMP
  const size_t mt = (size_t)omp_get_max_threads();
  double p[mt];
  for (size_t i = 0u; i < mt; ++i)
    p[i] = 0.0;
#pragma omp parallel default(none) shared(n,x,p)
  {
    const size_t nt = (size_t)omp_get_num_threads();
    const size_t tn = (size_t)omp_get_thread_num();
    lldiv_t qr = lldiv((long long)*n, (long long)nt);
    const size_t q = (size_t)(qr.quot);
    const size_t r = (size_t)(qr.rem);
    size_t m = q, o = (tn * q);
    if (r) {
      if (tn < r) {
        ++m;
        o += tn;
      }
      else
        o += r;
    }
    if (m)
      p[tn] = PVN_FABI(pvn_red_nrmi,PVN_RED_NRMI)(&m, (x + o));
  }
  return PVN_FABI(pvn_red_nrmi,PVN_RED_NRMI)(&mt, p);
#else /* !_OPENMP */
  return PVN_FABI(pvn_red_nrmi,PVN_RED_NRMI)(n, x);
#endif /* ?_OPENMP */
}

long double PVN_FABI(pvn_xnrmi,PVN_XNRMI)(const size_t *const n, const long double *const x)
{
  if (!n)
    return -1.0L;
  if (!*n)
    return -0.0L;
  if (!x)
    return -2.0L;
#ifdef _OPENMP
  const size_t mt = (size_t)omp_get_max_threads();
  long double p[mt];
  for (size_t i = 0u; i < mt; ++i)
    p[i] = 0.0L;
#pragma omp parallel default(none) shared(n,x,p)
  {
    const size_t nt = (size_t)omp_get_num_threads();
    const size_t tn = (size_t)omp_get_thread_num();
    lldiv_t qr = lldiv((long long)*n, (long long)nt);
    const size_t q = (size_t)(qr.quot);
    const size_t r = (size_t)(qr.rem);
    size_t m = q, o = (tn * q);
    if (r) {
      if (tn < r) {
        ++m;
        o += tn;
      }
      else
        o += r;
    }
    if (m)
      p[tn] = PVN_FABI(pvn_rex_nrmi,PVN_REX_NRMI)(&m, (x + o));
  }
  return PVN_FABI(pvn_rex_nrmi,PVN_REX_NRMI)(&mt, p);
#else /* !_OPENMP */
  return PVN_FABI(pvn_rex_nrmi,PVN_REX_NRMI)(n, x);
#endif /* ?_OPENMP */
}

float PVN_FABI(pvn_snrmp,PVN_SNRMP)(const float *const p, const size_t *const n, const float *const x)
{
  if (!p || !(*p > 0.0f))
    return -1.0f;
  if (!n)
    return -2.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -3.0f;
  if (*p == 1.0f)
    return PVN_FABI(pvn_snrm1,PVN_SNRM1)(n, x);
  if (*p == 2.0f)
    return PVN_FABI(pvn_snrm2,PVN_SNRM2)(n, x);
  if (isinf(*p))
    return PVN_FABI(pvn_snrmi,PVN_SNRMI)(n, x);
#ifdef _OPENMP
  const size_t mt = (size_t)omp_get_max_threads();
  float _p[mt];
  for (size_t i = 0u; i < mt; ++i)
    _p[i] = 0.0f;
#pragma omp parallel default(none) shared(n,x,_p)
  {
    const size_t nt = (size_t)omp_get_num_threads();
    const size_t tn = (size_t)omp_get_thread_num();
    lldiv_t qr = lldiv((long long)*n, (long long)nt);
    const size_t q = (size_t)(qr.quot);
    const size_t r = (size_t)(qr.rem);
    size_t m = q, o = (tn * q);
    if (r) {
      if (tn < r) {
        ++m;
        o += tn;
      }
      else
        o += r;
    }
    if (m)
      _p[tn] = PVN_FABI(pvn_res_nrmp,PVN_RES_NRMP)(p, &m, (x + o));
  }
  return PVN_FABI(pvn_res_nrmp,PVN_RES_NRMP)(p, &mt, _p);
#else /* !_OPENMP */
  return PVN_FABI(pvn_res_nrmp,PVN_RES_NRMP)(p, n, x);
#endif /* _OPENMP */
}

double PVN_FABI(pvn_dnrmp,PVN_DNRMP)(const double *const p, const size_t *const n, const double *const x)
{
  if (!p || !(*p > 0.0))
    return -1.0;
  if (!n)
    return -2.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -3.0;
  if (*p == 1.0)
    return PVN_FABI(pvn_dnrm1,PVN_DNRM1)(n, x);
  if (*p == 2.0)
    return PVN_FABI(pvn_dnrm2,PVN_DNRM2)(n, x);
  if (isinf(*p))
    return PVN_FABI(pvn_dnrmi,PVN_DNRMI)(n, x);
#ifdef _OPENMP
  const size_t mt = (size_t)omp_get_max_threads();
  double _p[mt];
  for (size_t i = 0u; i < mt; ++i)
    _p[i] = 0.0;
#pragma omp parallel default(none) shared(n,x,_p)
  {
    const size_t nt = (size_t)omp_get_num_threads();
    const size_t tn = (size_t)omp_get_thread_num();
    lldiv_t qr = lldiv((long long)*n, (long long)nt);
    const size_t q = (size_t)(qr.quot);
    const size_t r = (size_t)(qr.rem);
    size_t m = q, o = (tn * q);
    if (r) {
      if (tn < r) {
        ++m;
        o += tn;
      }
      else
        o += r;
    }
    if (m)
      _p[tn] = PVN_FABI(pvn_red_nrmp,PVN_RED_NRMP)(p, &m, (x + o));
  }
  return PVN_FABI(pvn_red_nrmp,PVN_RED_NRMP)(p, &mt, _p);
#else /* !_OPENMP */
  return PVN_FABI(pvn_red_nrmp,PVN_RED_NRMP)(p, n, x);
#endif /* ?_OPENMP */
}

long double PVN_FABI(pvn_xnrmp,PVN_XNRMP)(const long double *const p, const size_t *const n, const long double *const x)
{
  if (!p || !(*p > 0.0L))
    return -1.0L;
  if (!n)
    return -2.0L;
  if (!*n)
    return -0.0L;
  if (!x)
    return -3.0L;
  if (*p == 1.0L)
    return PVN_FABI(pvn_xnrm1,PVN_XNRM1)(n, x);
  if (*p == 2.0L)
    return PVN_FABI(pvn_xnrm2,PVN_XNRM2)(n, x);
  if (isinf(*p))
    return PVN_FABI(pvn_xnrmi,PVN_XNRMI)(n, x);
#ifdef _OPENMP
  const size_t mt = (size_t)omp_get_max_threads();
  long double _p[mt];
  for (size_t i = 0u; i < mt; ++i)
    _p[i] = 0.0L;
#pragma omp parallel default(none) shared(n,x,_p)
  {
    const size_t nt = (size_t)omp_get_num_threads();
    const size_t tn = (size_t)omp_get_thread_num();
    lldiv_t qr = lldiv((long long)*n, (long long)nt);
    const size_t q = (size_t)(qr.quot);
    const size_t r = (size_t)(qr.rem);
    size_t m = q, o = (tn * q);
    if (r) {
      if (tn < r) {
        ++m;
        o += tn;
      }
      else
        o += r;
    }
    if (m)
      _p[tn] = PVN_FABI(pvn_rex_nrmp,PVN_REX_NRMP)(p, &m, (x + o));
  }
  return PVN_FABI(pvn_rex_nrmp,PVN_REX_NRMP)(p, &mt, _p);
#else /* !_OPENMP */
  return PVN_FABI(pvn_rex_nrmp,PVN_REX_NRMP)(p, n, x);
#endif /* ?_OPENMP */
}

#ifdef PVN_QUADMATH
#ifdef PVN_MPFR
__float128 PVN_FABI(pvn_mpq_nrmp,PVN_MPQ_NRMP)(const __float128 *const p, const size_t *const n, const __float128 *const x)
{
  if (!p || !(*p > 0.0q))
    return -1.0q;
  if (!n)
    return -2.0q;
  if (!*n)
    return -0.0q;
  if (!x)
    return -3.0q;
  const size_t m = *n;
  __float128 f = 0.0q;
  mpfr_t mf, mx;
  (void)mpfr_init_set_ld(mf, 0.0L, MPFR_RNDN);
  (void)mpfr_init_set_ld(mx, 0.0L, MPFR_RNDN);
  if (*p == 1.0q) {
    for (size_t i = 0u; i < m; ++i) {
      (void)mpfr_set_float128(mx, x[i], MPFR_RNDN);
      (void)mpfr_abs(mx, mx, MPFR_RNDN);
      (void)mpfr_add(mf, mf, mx, MPFR_RNDN);
    }
  }
  else if (*p == 2.0q) {
    for (size_t i = 0u; i < m; ++i) {
      (void)mpfr_set_float128(mx, x[i], MPFR_RNDN);
      (void)mpfr_fma(mf, mx, mx, mf, MPFR_RNDN);
    }
    (void)mpfr_sqrt(mf, mf, MPFR_RNDN);
  }
  else if (!isfiniteq(*p)) {
    for (size_t i = 0u; i < m; ++i) {
      (void)mpfr_set_float128(mx, x[i], MPFR_RNDN);
      (void)mpfr_abs(mx, mx, MPFR_RNDN);
      (void)mpfr_max(mf, mf, mx, MPFR_RNDN);
    }
  }
  else {
    mpfr_t mp;
    (void)mpfr_init_set_ld(mp, 0.0L, MPFR_RNDN);
    (void)mpfr_set_float128(mf, *p, MPFR_RNDN);
    for (size_t i = 0u; i < m; ++i) {
      (void)mpfr_set_float128(mx, x[i], MPFR_RNDN);
      (void)mpfr_abs(mx, mx, MPFR_RNDN);
      (void)mpfr_pow(mx, mx, mp, MPFR_RNDN);
      (void)mpfr_add(mf, mf, mx, MPFR_RNDN);
    }
    (void)mpfr_d_div(mp, 1.0, mp, MPFR_RNDN);
    (void)mpfr_pow(mf, mf, mp, MPFR_RNDN);
    mpfr_clear(mp);
  }
  f = mpfr_get_float128(mf, MPFR_RNDN);
  mpfr_clear(mx);
  mpfr_clear(mf);
  return f;
}
#endif /* PVN_MPFR */

__float128 PVN_FABI(pvn_req_nrmf,PVN_REQ_NRMF)(const size_t *const n, const __float128 *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0q;
  if (!*n)
    return -0.0q;
  if (!x)
    return -2.0q;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return fabsq(*x);
  if (*n == (size_t)2u)
    return hypotq(x[0], x[1]);
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  __float128 fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_req_nrmf,PVN_REQ_NRMF)(&nl, x);
    fr = PVN_FABI(pvn_req_nrmf,PVN_REQ_NRMF)(&nr, (x + nl));
  }
  return hypotq(fl, fr);
}

__float128 PVN_FABI(pvn_req_nrm1,PVN_REQ_NRM1)(const size_t *const n, const __float128 *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0q;
  if (!*n)
    return -0.0q;
  if (!x)
    return -2.0q;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return fabsq(*x);
  if (*n == (size_t)2u)
    return (fabsq(x[0]) + fabsq(x[1]));
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  __float128 fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_req_nrm1,PVN_REQ_NRM1)(&nl, x);
    fr = PVN_FABI(pvn_req_nrm1,PVN_REQ_NRM1)(&nr, (x + nl));
  }
  return (fl + fr);
}

__float128 PVN_FABI(pvn_req_nrmi,PVN_REQ_NRMI)(const size_t *const n, const __float128 *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0q;
  if (!*n)
    return -0.0q;
  if (!x)
    return -2.0q;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return fabsq(*x);
  if (*n == (size_t)2u)
    return fmaxq(fabsq(x[0]), fabsq(x[1]));
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  __float128 fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_req_nrmi,PVN_REQ_NRMI)(&nl, x);
    fr = PVN_FABI(pvn_req_nrmi,PVN_REQ_NRMI)(&nr, (x + nl));
  }
  return fmaxq(fl, fr);
}

__float128 PVN_FABI(pvn_rfq_nrmf,PVN_RFQ_NRMF)(const size_t *const n, const __float128 *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0q;
  if (!*n)
    return -0.0q;
  if (!x)
    return -2.0q;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return fabsq(*x);
  if (*n == (size_t)2u)
#ifdef __MATHIMF_H_INCLUDED
    return __hypotq(x[0], x[1]);
#else /* !__MATHIMF_H_INCLUDED */
#ifdef _GNU_SOURCE
    return __builtin_hypotf128(x[0], x[1]);
#else /* !_GNU_SOURCE */
    return hypotq(x[0], x[1]);
#endif /* ?_GNU_SOURCE */
#endif /* ?__MATHIMF_H_INCLUDED */
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  __float128 fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_rfq_nrmf,PVN_RFQ_NRMF)(&nl, x);
    fr = PVN_FABI(pvn_rfq_nrmf,PVN_RFQ_NRMF)(&nr, (x + nl));
  }
#ifdef __MATHIMF_H_INCLUDED
  return __hypotq(fl, fr);
#else /* !__MATHIMF_H_INCLUDED */
#ifdef _GNU_SOURCE
  return __builtin_hypotf128(fl, fr);
#else /* !_GNU_SOURCE */
  return hypotq(fl, fr);
#endif /* ?_GNU_SOURCE */
#endif /* ?__MATHIMF_H_INCLUDED */
}

__float128 PVN_FABI(pvn_rhq_nrmf,PVN_RHQ_NRMF)(const size_t *const n, const __float128 *const x)
{
#ifndef NDEBUG
  if (!n)
    return -1.0q;
  if (!*n)
    return -0.0q;
  if (!x)
    return -2.0q;
#endif /* !NDEBUG */
  if (*n == (size_t)1u)
    return fabsq(*x);
  if (*n == (size_t)2u)
    return pvn_v1q_hypot(x[0], x[1]);
  const size_t nl = ((*n >> 1u) + (*n & (size_t)1u));
  const size_t nr = (*n - nl);
  __float128 fl, fr;
  cilk_scope {
    fl = cilk_spawn PVN_FABI(pvn_rhq_nrmf,PVN_RHQ_NRMF)(&nl, x);
    fr = PVN_FABI(pvn_rhq_nrmf,PVN_RHQ_NRMF)(&nr, (x + nl));
  }
  return pvn_v1q_hypot(fl, fr);
}

__float128 PVN_FABI(pvn_qnrm2,PVN_QNRM2)(const size_t *const n, const __float128 *const x)
{
  if (!n)
    return -1.0q;
  if (!*n)
    return -0.0q;
  if (!x)
    return -2.0q;
#ifdef _OPENMP
  const size_t mt = (size_t)omp_get_max_threads();
  __float128 p[mt];
  for (size_t i = 0u; i < mt; ++i)
    p[i] = 0.0q;
#pragma omp parallel default(none) shared(n,x,p)
  {
    const size_t nt = (size_t)omp_get_num_threads();
    const size_t tn = (size_t)omp_get_thread_num();
    lldiv_t qr = lldiv((long long)*n, (long long)nt);
    const size_t q = (size_t)(qr.quot);
    const size_t r = (size_t)(qr.rem);
    size_t m = q, o = (tn * q);
    if (r) {
      if (tn < r) {
        ++m;
        o += tn;
      }
      else
        o += r;
    }
    if (m)
#ifdef PVN_NRM_SAFE
      p[tn] = PVN_FABI(pvn_req_nrmf,PVN_REQ_NRMF)(&m, (x + o));
#else /* !PVN_NRM_SAFE */
      p[tn] = PVN_FABI(pvn_rfq_nrmf,PVN_RFQ_NRMF)(&m, (x + o));
#endif /* ?PVN_NRM_SAFE */
  }
#ifdef PVN_NRM_SAFE
  return PVN_FABI(pvn_req_nrmf,PVN_REQ_NRMF)(&mt, p);
#else /* !PVN_NRM_SAFE */
  return PVN_FABI(pvn_rfq_nrmf,PVN_RFQ_NRMF)(&mt, p);
#endif /* ?PVN_NRM_SAFE */
#else /* !_OPENMP */
#ifdef PVN_NRM_SAFE
  return PVN_FABI(pvn_req_nrmf,PVN_REQ_NRMF)(n, x);
#else /* !PVN_NRM_SAFE */
  return PVN_FABI(pvn_rfq_nrmf,PVN_RFQ_NRMF)(n, x);
#endif /* ?PVN_NRM_SAFE */
#endif /* ?_OPENMP */
}

__float128 PVN_FABI(pvn_ynrm2,PVN_YNRM2)(const size_t *const n, const __float128 *const x)
{
  const size_t m = (n ? (*n << 1u) : (size_t)0u);
  return PVN_FABI(pvn_qnrm2,PVN_QNRM2)(&m, x);
}

__float128 PVN_FABI(pvn_qnrm1,PVN_QNRM1)(const size_t *const n, const __float128 *const x)
{
  if (!n)
    return -1.0q;
  if (!*n)
    return -0.0q;
  if (!x)
    return -2.0q;
#ifdef _OPENMP
  const size_t mt = (size_t)omp_get_max_threads();
  __float128 p[mt];
  for (size_t i = 0u; i < mt; ++i)
    p[i] = 0.0q;
#pragma omp parallel default(none) shared(n,x,p)
  {
    const size_t nt = (size_t)omp_get_num_threads();
    const size_t tn = (size_t)omp_get_thread_num();
    lldiv_t qr = lldiv((long long)*n, (long long)nt);
    const size_t q = (size_t)(qr.quot);
    const size_t r = (size_t)(qr.rem);
    size_t m = q, o = (tn * q);
    if (r) {
      if (tn < r) {
        ++m;
        o += tn;
      }
      else
        o += r;
    }
    if (m)
      p[tn] = PVN_FABI(pvn_req_nrm1,PVN_REQ_NRM1)(&m, (x + o));
  }
  return PVN_FABI(pvn_req_nrm1,PVN_REQ_NRM1)(&mt, p);
#else /* !_OPENMP */
  return PVN_FABI(pvn_req_nrm1,PVN_REQ_NRM1)(n, x);
#endif /* ?_OPENMP */
}

__float128 PVN_FABI(pvn_qnrmi,PVN_QNRMI)(const size_t *const n, const __float128 *const x)
{
  if (!n)
    return -1.0q;
  if (!*n)
    return -0.0q;
  if (!x)
    return -2.0q;
#ifdef _OPENMP
  const size_t mt = (size_t)omp_get_max_threads();
  __float128 p[mt];
  for (size_t i = 0u; i < mt; ++i)
    p[i] = 0.0q;
#pragma omp parallel default(none) shared(n,x,p)
  {
    const size_t nt = (size_t)omp_get_num_threads();
    const size_t tn = (size_t)omp_get_thread_num();
    lldiv_t qr = lldiv((long long)*n, (long long)nt);
    const size_t q = (size_t)(qr.quot);
    const size_t r = (size_t)(qr.rem);
    size_t m = q, o = (tn * q);
    if (r) {
      if (tn < r) {
        ++m;
        o += tn;
      }
      else
        o += r;
    }
    if (m)
      p[tn] = PVN_FABI(pvn_req_nrmi,PVN_REQ_NRMI)(&m, (x + o));
  }
  return PVN_FABI(pvn_req_nrmi,PVN_REQ_NRMI)(&mt, p);
#else /* !_OPENMP */
  return PVN_FABI(pvn_req_nrmi,PVN_REQ_NRMI)(n, x);
#endif /* ?_OPENMP */
}
#else /* !PVN_QUADMATH */
#ifdef PVN_MPFR
long double PVN_FABI(pvn_mpq_nrmp,PVN_MPQ_NRMP)(const long double *const p, const size_t *const n, const long double *const x)
{
  return PVN_FABI(pvn_mpx_nrmp,PVN_MPX_NRMP)(p, n, x);
}
#endif /* PVN_MPFR */
long double PVN_FABI(pvn_req_nrmf,PVN_REQ_NRMF)(const size_t *const n, const long double *const x)
{
  return PVN_FABI(pvn_rex_nrmf,PVN_REX_NRMF)(n, x);
}
long double PVN_FABI(pvn_req_nrm1,PVN_REQ_NRM1)(const size_t *const n, const long double *const x)
{
  return PVN_FABI(pvn_rex_nrm1,PVN_REX_NRM1)(n, x);
}
long double PVN_FABI(pvn_req_nrmi,PVN_REQ_NRMI)(const size_t *const n, const long double *const x)
{
  return PVN_FABI(pvn_rex_nrmi,PVN_REX_NRMI)(n, x);
}
long double PVN_FABI(pvn_rfq_nrmf,PVN_RFQ_NRMF)(const size_t *const n, const long double *const x)
{
  return PVN_FABI(pvn_rfx_nrmf,PVN_RFX_NRMF)(n, x);
}
long double PVN_FABI(pvn_rhq_nrmf,PVN_RHQ_NRMF)(const size_t *const n, const long double *const x)
{
  return PVN_FABI(pvn_rhx_nrmf,PVN_RHX_NRMF)(n, x);
}
long double PVN_FABI(pvn_qnrm2,PVN_QNRM2)(const size_t *const n, const long double *const x)
{
  return PVN_FABI(pvn_xnrm2,PVN_XNRM2)(n, x);
}
long double PVN_FABI(pvn_ynrm2,PVN_YNRM2)(const size_t *const n, const long double *const x)
{
  return PVN_FABI(pvn_wnrm2,PVN_WNRM2)(n, x);
}
long double PVN_FABI(pvn_qnrm1,PVN_QNRM1)(const size_t *const n, const long double *const x)
{
  return PVN_FABI(pvn_xnrm1,PVN_XNRM1)(n, x);
}
long double PVN_FABI(pvn_qnrmi,PVN_QNRMI)(const size_t *const n, const long double *const x)
{
  return PVN_FABI(pvn_xnrmi,PVN_XNRMI)(n, x);
}
#endif /* ?PVN_QUADMATH */

#if (defined(__AVX__) && defined(__FMA__))
static __m128 rxs_nrmf(const size_t n, const float *const x)
{
  register const __m128 z = _mm_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)3u),
    m = ((n >> 2u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm_andnot_ps(z, _mm_load_ps(x));
    case 1u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, -0.0f, -0.0f, x[0u]));
    case 2u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, -0.0f, x[1u], x[0u]));
    case 3u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, x[2u], x[1u], x[0u]));
    default: /* should never happen */
      return z;
    }
  }
  register __m128 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm_load_ps(x);
    fr = (r ? rxs_nrmf(r, (x + 4u)) : _mm_load_ps(x + 4u));
    return pvn_v4s_hypot(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 2u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rxs_nrmf(nl, x);
    fr = rxs_nrmf(nr, (x + nl));
  }
  return pvn_v4s_hypot(fl, fr);
}

static __m128 rxsunrmf(const size_t n, const float *const x)
{
  register const __m128 z = _mm_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)3u),
    m = ((n >> 2u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm_andnot_ps(z, _mm_loadu_ps(x));
    case 1u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, -0.0f, -0.0f, x[0u]));
    case 2u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, -0.0f, x[1u], x[0u]));
    case 3u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, x[2u], x[1u], x[0u]));
    default: /* should never happen */
      return z;
    }
  }
  register __m128 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm_loadu_ps(x);
    fr = (r ? rxsunrmf(r, (x + 4u)) : _mm_loadu_ps(x + 4u));
    return pvn_v4s_hypot(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 2u);
  const size_t nr = (n - nl);

  cilk_scope {
    fl = cilk_spawn rxsunrmf(nl, x);
    fr = rxsunrmf(nr, (x + nl));
  }
  return pvn_v4s_hypot(fl, fr);
}

float PVN_FABI(pvn_rxs_nrmf,PVN_RXS_NRMF)(const size_t *const n, const float *const x)
{
  if (!n)
    return -1.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -2.0f;
  register const __m128 r = (((uintptr_t)x & (uintptr_t)0x0Fu) ? rxsunrmf(*n, x) : rxs_nrmf(*n, x));
#ifdef PVN_NRM_SAFE
  alignas(16) float f[4u];
  _mm_store_ps(f, r);
  const size_t m = (size_t)4u;
  return PVN_FABI(pvn_res_nrmf,PVN_RES_NRMF)(&m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v4s_hypot_red(r);
#endif /* ?PVN_NRM_SAFE */
}

static __m128d rxd_nrmf(const size_t n, const double *const x)
{
  register const __m128d z = _mm_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)1u),
    m = ((n >> 1u) + r);
  if (m == (size_t)1u) {
    if (r)
      return _mm_andnot_pd(z, _mm_set_pd(-0.0, *x));
    else
      return _mm_andnot_pd(z, _mm_load_pd(x));
  }
  register __m128d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm_load_pd(x);
    fr = (r ? rxd_nrmf(r, (x + 2u)) : _mm_load_pd(x + 2u));
    return pvn_v2d_hypot(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 1u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rxd_nrmf(nl, x);
    fr = rxd_nrmf(nr, (x + nl));
  }
  return pvn_v2d_hypot(fl, fr);
}

static __m128d rxdunrmf(const size_t n, const double *const x)
{
  register const __m128d z = _mm_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)1u),
    m = ((n >> 1u) + r);
  if (m == (size_t)1u) {
    if (r)
      return _mm_andnot_pd(z, _mm_set_pd(-0.0, *x));
    else
      return _mm_andnot_pd(z, _mm_loadu_pd(x));
  }
  register __m128d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm_loadu_pd(x);
    fr = (r ? rxdunrmf(r, (x + 2u)) : _mm_loadu_pd(x + 2u));
    return pvn_v2d_hypot(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 1u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rxdunrmf(nl, x);
    fr = rxdunrmf(nr, (x + nl));
  }
  return pvn_v2d_hypot(fl, fr);
}

double PVN_FABI(pvn_rxd_nrmf,PVN_RXD_NRMF)(const size_t *const n, const double *const x)
{
  if (!n)
    return -1.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -2.0;
  register const __m128d r = (((uintptr_t)x & (uintptr_t)0x0Fu) ? rxdunrmf(*n, x) : rxd_nrmf(*n, x));
#ifdef PVN_NRM_SAFE
  alignas(16) double f[2u];
  _mm_store_pd(f, r);
  return hypot(f[0u], f[1u]);
#else /* !PVN_NRM_SAFE */
  return pvn_v2d_hypot_red(r);
#endif /* ?PVN_NRM_SAFE */
}

static __m128 rxs_nrm1(const size_t n, const float *const x)
{
  register const __m128 z = _mm_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)3u),
    m = ((n >> 2u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm_andnot_ps(z, _mm_load_ps(x));
    case 1u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, -0.0f, -0.0f, x[0u]));
    case 2u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, -0.0f, x[1u], x[0u]));
    case 3u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, x[2u], x[1u], x[0u]));
    default: /* should never happen */
      return z;
    }
  }
  register __m128 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm_load_ps(x);
    fr = (r ? rxs_nrm1(r, (x + 4u)) : _mm_load_ps(x + 4u));
    return pvn_v4s_add_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 2u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rxs_nrm1(nl, x);
    fr = rxs_nrm1(nr, (x + nl));
  }
  return pvn_v4s_add_abs(fl, fr);
}

static __m128 rxsunrm1(const size_t n, const float *const x)
{
  register const __m128 z = _mm_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)3u),
    m = ((n >> 2u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm_andnot_ps(z, _mm_loadu_ps(x));
    case 1u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, -0.0f, -0.0f, x[0u]));
    case 2u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, -0.0f, x[1u], x[0u]));
    case 3u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, x[2u], x[1u], x[0u]));
    default: /* should never happen */
      return z;
    }
  }
  register __m128 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm_loadu_ps(x);
    fr = (r ? rxsunrm1(r, (x + 4u)) : _mm_loadu_ps(x + 4u));
    return pvn_v4s_add_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 2u);
  const size_t nr = (n - nl);

  cilk_scope {
    fl = cilk_spawn rxsunrm1(nl, x);
    fr = rxsunrm1(nr, (x + nl));
  }
  return pvn_v4s_add_abs(fl, fr);
}

float PVN_FABI(pvn_rxs_nrm1,PVN_RXS_NRM1)(const size_t *const n, const float *const x)
{
  if (!n)
    return -1.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -2.0f;
  register const __m128 r = (((uintptr_t)x & (uintptr_t)0x0Fu) ? rxsunrm1(*n, x) : rxs_nrm1(*n, x));
#ifdef PVN_NRM_SAFE
  alignas(16) float f[4u];
  _mm_store_ps(f, r);
  const size_t m = (size_t)4u;
  return PVN_FABI(pvn_res_nrm1,PVN_RES_NRM1)(&m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v4s_add_red(r);
#endif /* ?PVN_NRM_SAFE */
}

static __m128d rxd_nrm1(const size_t n, const double *const x)
{
  register const __m128d z = _mm_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)1u),
    m = ((n >> 1u) + r);
  if (m == (size_t)1u) {
    if (r)
      return _mm_andnot_pd(z, _mm_set_pd(-0.0, *x));
    else
      return _mm_andnot_pd(z, _mm_load_pd(x));
  }
  register __m128d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm_load_pd(x);
    fr = (r ? rxd_nrm1(r, (x + 2u)) : _mm_load_pd(x + 2u));
    return pvn_v2d_add_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 1u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rxd_nrm1(nl, x);
    fr = rxd_nrm1(nr, (x + nl));
  }
  return pvn_v2d_add_abs(fl, fr);
}

static __m128d rxdunrm1(const size_t n, const double *const x)
{
  register const __m128d z = _mm_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)1u),
    m = ((n >> 1u) + r);
  if (m == (size_t)1u) {
    if (r)
      return _mm_andnot_pd(z, _mm_set_pd(-0.0, *x));
    else
      return _mm_andnot_pd(z, _mm_loadu_pd(x));
  }
  register __m128d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm_loadu_pd(x);
    fr = (r ? rxdunrm1(r, (x + 2u)) : _mm_loadu_pd(x + 2u));
    return pvn_v2d_add_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 1u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rxdunrm1(nl, x);
    fr = rxdunrm1(nr, (x + nl));
  }
  return pvn_v2d_add_abs(fl, fr);
}

double PVN_FABI(pvn_rxd_nrm1,PVN_RXD_NRM1)(const size_t *const n, const double *const x)
{
  if (!n)
    return -1.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -2.0;
  register const __m128d r = (((uintptr_t)x & (uintptr_t)0x0Fu) ? rxdunrm1(*n, x) : rxd_nrm1(*n, x));
#ifdef PVN_NRM_SAFE
  alignas(16) double f[2u];
  _mm_store_pd(f, r);
  return pvn_v1d_add_abs(f[0u], f[1u]);
#else /* !PVN_NRM_SAFE */
  return pvn_v2d_add_red(r);
#endif /* ?PVN_NRM_SAFE */
}

static __m128 rxs_nrmi(const size_t n, const float *const x)
{
  register const __m128 z = _mm_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)3u),
    m = ((n >> 2u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm_andnot_ps(z, _mm_load_ps(x));
    case 1u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, -0.0f, -0.0f, x[0u]));
    case 2u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, -0.0f, x[1u], x[0u]));
    case 3u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, x[2u], x[1u], x[0u]));
    default: /* should never happen */
      return z;
    }
  }
  register __m128 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm_load_ps(x);
    fr = (r ? rxs_nrmi(r, (x + 4u)) : _mm_load_ps(x + 4u));
    return pvn_v4s_max_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 2u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rxs_nrmi(nl, x);
    fr = rxs_nrmi(nr, (x + nl));
  }
  return pvn_v4s_max_abs(fl, fr);
}

static __m128 rxsunrmi(const size_t n, const float *const x)
{
  register const __m128 z = _mm_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)3u),
    m = ((n >> 2u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm_andnot_ps(z, _mm_loadu_ps(x));
    case 1u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, -0.0f, -0.0f, x[0u]));
    case 2u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, -0.0f, x[1u], x[0u]));
    case 3u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, x[2u], x[1u], x[0u]));
    default: /* should never happen */
      return z;
    }
  }
  register __m128 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm_loadu_ps(x);
    fr = (r ? rxsunrmi(r, (x + 4u)) : _mm_loadu_ps(x + 4u));
    return pvn_v4s_max_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 2u);
  const size_t nr = (n - nl);

  cilk_scope {
    fl = cilk_spawn rxsunrmi(nl, x);
    fr = rxsunrmi(nr, (x + nl));
  }
  return pvn_v4s_max_abs(fl, fr);
}

float PVN_FABI(pvn_rxs_nrmi,PVN_RXS_NRMI)(const size_t *const n, const float *const x)
{
  if (!n)
    return -1.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -2.0f;
  register const __m128 r = (((uintptr_t)x & (uintptr_t)0x0Fu) ? rxsunrmi(*n, x) : rxs_nrmi(*n, x));
#ifdef PVN_NRM_SAFE
  alignas(16) float f[4u];
  _mm_store_ps(f, r);
  const size_t m = (size_t)4u;
  return PVN_FABI(pvn_res_nrmi,PVN_RES_NRMI)(&m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v4s_max_red(r);
#endif /* ?PVN_NRM_SAFE */
}

static __m128d rxd_nrmi(const size_t n, const double *const x)
{
  register const __m128d z = _mm_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)1u),
    m = ((n >> 1u) + r);
  if (m == (size_t)1u) {
    if (r)
      return _mm_andnot_pd(z, _mm_set_pd(-0.0, *x));
    else
      return _mm_andnot_pd(z, _mm_load_pd(x));
  }
  register __m128d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm_load_pd(x);
    fr = (r ? rxd_nrmi(r, (x + 2u)) : _mm_load_pd(x + 2u));
    return pvn_v2d_max_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 1u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rxd_nrmi(nl, x);
    fr = rxd_nrmi(nr, (x + nl));
  }
  return pvn_v2d_max_abs(fl, fr);
}

static __m128d rxdunrmi(const size_t n, const double *const x)
{
  register const __m128d z = _mm_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)1u),
    m = ((n >> 1u) + r);
  if (m == (size_t)1u) {
    if (r)
      return _mm_andnot_pd(z, _mm_set_pd(-0.0, *x));
    else
      return _mm_andnot_pd(z, _mm_loadu_pd(x));
  }
  register __m128d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm_loadu_pd(x);
    fr = (r ? rxdunrmi(r, (x + 2u)) : _mm_loadu_pd(x + 2u));
    return pvn_v2d_max_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 1u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rxdunrmi(nl, x);
    fr = rxdunrmi(nr, (x + nl));
  }
  return pvn_v2d_max_abs(fl, fr);
}

double PVN_FABI(pvn_rxd_nrmi,PVN_RXD_NRMI)(const size_t *const n, const double *const x)
{
  if (!n)
    return -1.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -2.0;
  register const __m128d r = (((uintptr_t)x & (uintptr_t)0x0Fu) ? rxdunrmi(*n, x) : rxd_nrmi(*n, x));
#ifdef PVN_NRM_SAFE
  alignas(16) double f[2u];
  _mm_store_pd(f, r);
  return pvn_v1d_max_abs(f[0u], f[1u]);
#else /* !PVN_NRM_SAFE */
  return pvn_v2d_max_red(r);
#endif /* ?PVN_NRM_SAFE */
}

static __m256 rys_nrmf(const size_t n, const float *const x)
{
  register const __m256 z = _mm256_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)7u),
    m = ((n >> 3u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm256_andnot_ps(z, _mm256_load_ps(x));
    case 1u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, -0.0f, -0.0f, x[0u]), 0));
    case 2u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, -0.0f, x[1u], x[0u]), 0));
    case 3u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, x[2u], x[1u], x[0u]), 0));
    case 4u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_load_ps(x), 0));
    case 5u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, -0.0f, -0.0f, x[4u]), _mm_load_ps(x)));
    case 6u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, -0.0f, x[5u], x[4u]), _mm_load_ps(x)));
    case 7u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, x[6u], x[5u], x[4u]), _mm_load_ps(x)));
    default: /* should never happen */
      return z;
    }
  }
  register __m256 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm256_load_ps(x);
    fr = (r ? rys_nrmf(r, (x + 8u)) : _mm256_load_ps(x + 8u));
    return pvn_v8s_hypot(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 3u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rys_nrmf(nl, x);
    fr = rys_nrmf(nr, (x + nl));
  }
  return pvn_v8s_hypot(fl, fr);
}

static __m256 rysunrmf(const size_t n, const float *const x)
{
  register const __m256 z = _mm256_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)7u),
    m = ((n >> 3u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm256_andnot_ps(z, _mm256_loadu_ps(x));
    case 1u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, -0.0f, -0.0f, x[0u]), 0));
    case 2u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, -0.0f, x[1u], x[0u]), 0));
    case 3u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, x[2u], x[1u], x[0u]), 0));
    case 4u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_loadu_ps(x), 0));
    case 5u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, -0.0f, -0.0f, x[4u]), _mm_loadu_ps(x)));
    case 6u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, -0.0f, x[5u], x[4u]), _mm_loadu_ps(x)));
    case 7u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, x[6u], x[5u], x[4u]), _mm_loadu_ps(x)));
    default: /* should never happen */
      return z;
    }
  }
  register __m256 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm256_loadu_ps(x);
    fr = (r ? rysunrmf(r, (x + 8u)) : _mm256_loadu_ps(x + 8u));
    return pvn_v8s_hypot(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 3u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rysunrmf(nl, x);
    fr = rysunrmf(nr, (x + nl));
  }
  return pvn_v8s_hypot(fl, fr);
}

float PVN_FABI(pvn_rys_nrmf,PVN_RYS_NRMF)(const size_t *const n, const float *const x)
{
  if (!n)
    return -1.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -2.0f;
  register const __m256 r = (((uintptr_t)x & (uintptr_t)0x1Fu) ? rysunrmf(*n, x) : rys_nrmf(*n, x));
#ifdef PVN_NRM_SAFE
  alignas(32) float f[8u];
  _mm256_store_ps(f, r);
  const size_t m = (size_t)8u;
  return PVN_FABI(pvn_res_nrmf,PVN_RES_NRMF)(&m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v8s_hypot_red(r);
#endif /* ?PVN_NRM_SAFE */
}

static __m256d ryd_nrmf(const size_t n, const double *const x)
{
  register const __m256d z = _mm256_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)3u),
    m = ((n >> 2u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm256_andnot_pd(z, _mm256_load_pd(x));
    case 1u:
      return _mm256_andnot_pd(z, _mm256_insertf128_pd(z, _mm_set_pd(-0.0, x[0u]), 0));
    case 2u:
      return _mm256_andnot_pd(z, _mm256_insertf128_pd(z, _mm_load_pd(x), 0));
    case 3u:
      return _mm256_andnot_pd(z, _mm256_set_m128d(_mm_set_pd(-0.0, x[2u]), _mm_load_pd(x)));
    default: /* should never happen */
      return z;
    }
  }
  register __m256d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm256_load_pd(x);
    fr = (r ? ryd_nrmf(r, (x + 4u)) : _mm256_load_pd(x + 4u));
    return pvn_v4d_hypot(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 2u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn ryd_nrmf(nl, x);
    fr = ryd_nrmf(nr, (x + nl));
  }
  return pvn_v4d_hypot(fl, fr);
}

static __m256d rydunrmf(const size_t n, const double *const x)
{
  register const __m256d z = _mm256_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)3u),
    m = ((n >> 2u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm256_andnot_pd(z, _mm256_loadu_pd(x));
    case 1u:
      return _mm256_andnot_pd(z, _mm256_insertf128_pd(z, _mm_set_pd(-0.0, x[0u]), 0));
    case 2u:
      return _mm256_andnot_pd(z, _mm256_insertf128_pd(z, _mm_loadu_pd(x), 0));
    case 3u:
      return _mm256_andnot_pd(z, _mm256_set_m128d(_mm_set_pd(-0.0, x[2u]), _mm_loadu_pd(x)));
    default: /* should never happen */
      return z;
    }
  }
  register __m256d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm256_loadu_pd(x);
    fr = (r ? rydunrmf(r, (x + 4u)) : _mm256_loadu_pd(x + 4u));
    return pvn_v4d_hypot(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 2u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rydunrmf(nl, x);
    fr = rydunrmf(nr, (x + nl));
  }
  return pvn_v4d_hypot(fl, fr);
}

double PVN_FABI(pvn_ryd_nrmf,PVN_RYD_NRMF)(const size_t *const n, const double *const x)
{
  if (!n)
    return -1.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -2.0;
  register const __m256d r = (((uintptr_t)x & (uintptr_t)0x1Fu) ? rydunrmf(*n, x) : ryd_nrmf(*n, x));
#ifdef PVN_NRM_SAFE
  alignas(32) double f[4u];
  _mm256_store_pd(f, r);
  const size_t m = (size_t)4u;
  return PVN_FABI(pvn_red_nrmf,PVN_RED_NRMF)(&m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v4d_hypot_red(r);
#endif /* ?PVN_NRM_SAFE */
}

static __m256 rys_nrm1(const size_t n, const float *const x)
{
  register const __m256 z = _mm256_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)7u),
    m = ((n >> 3u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm256_andnot_ps(z, _mm256_load_ps(x));
    case 1u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, -0.0f, -0.0f, x[0u]), 0));
    case 2u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, -0.0f, x[1u], x[0u]), 0));
    case 3u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, x[2u], x[1u], x[0u]), 0));
    case 4u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_load_ps(x), 0));
    case 5u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, -0.0f, -0.0f, x[4u]), _mm_load_ps(x)));
    case 6u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, -0.0f, x[5u], x[4u]), _mm_load_ps(x)));
    case 7u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, x[6u], x[5u], x[4u]), _mm_load_ps(x)));
    default: /* should never happen */
      return z;
    }
  }
  register __m256 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm256_load_ps(x);
    fr = (r ? rys_nrm1(r, (x + 8u)) : _mm256_load_ps(x + 8u));
    return pvn_v8s_add_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 3u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rys_nrm1(nl, x);
    fr = rys_nrm1(nr, (x + nl));
  }
  return pvn_v8s_add_abs(fl, fr);
}

static __m256 rysunrm1(const size_t n, const float *const x)
{
  register const __m256 z = _mm256_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)7u),
    m = ((n >> 3u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm256_andnot_ps(z, _mm256_loadu_ps(x));
    case 1u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, -0.0f, -0.0f, x[0u]), 0));
    case 2u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, -0.0f, x[1u], x[0u]), 0));
    case 3u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, x[2u], x[1u], x[0u]), 0));
    case 4u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_loadu_ps(x), 0));
    case 5u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, -0.0f, -0.0f, x[4u]), _mm_loadu_ps(x)));
    case 6u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, -0.0f, x[5u], x[4u]), _mm_loadu_ps(x)));
    case 7u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, x[6u], x[5u], x[4u]), _mm_loadu_ps(x)));
    default: /* should never happen */
      return z;
    }
  }
  register __m256 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm256_loadu_ps(x);
    fr = (r ? rysunrm1(r, (x + 8u)) : _mm256_loadu_ps(x + 8u));
    return pvn_v8s_add_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 3u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rysunrm1(nl, x);
    fr = rysunrm1(nr, (x + nl));
  }
  return pvn_v8s_add_abs(fl, fr);
}

float PVN_FABI(pvn_rys_nrm1,PVN_RYS_NRM1)(const size_t *const n, const float *const x)
{
  if (!n)
    return -1.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -2.0f;
  register const __m256 r = (((uintptr_t)x & (uintptr_t)0x1Fu) ? rysunrm1(*n, x) : rys_nrm1(*n, x));
#ifdef PVN_NRM_SAFE
  alignas(32) float f[8u];
  _mm256_store_ps(f, r);
  const size_t m = (size_t)8u;
  return PVN_FABI(pvn_res_nrm1,PVN_RES_NRM1)(&m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v8s_add_red(r);
#endif /* ?PVN_NRM_SAFE */
}

static __m256d ryd_nrm1(const size_t n, const double *const x)
{
  register const __m256d z = _mm256_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)3u),
    m = ((n >> 2u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm256_andnot_pd(z, _mm256_load_pd(x));
    case 1u:
      return _mm256_andnot_pd(z, _mm256_insertf128_pd(z, _mm_set_pd(-0.0, x[0u]), 0));
    case 2u:
      return _mm256_andnot_pd(z, _mm256_insertf128_pd(z, _mm_load_pd(x), 0));
    case 3u:
      return _mm256_andnot_pd(z, _mm256_set_m128d(_mm_set_pd(-0.0, x[2u]), _mm_load_pd(x)));
    default: /* should never happen */
      return z;
    }
  }
  register __m256d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm256_load_pd(x);
    fr = (r ? ryd_nrm1(r, (x + 4u)) : _mm256_load_pd(x + 4u));
    return pvn_v4d_add_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 2u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn ryd_nrm1(nl, x);
    fr = ryd_nrm1(nr, (x + nl));
  }
  return pvn_v4d_add_abs(fl, fr);
}

static __m256d rydunrm1(const size_t n, const double *const x)
{
  register const __m256d z = _mm256_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)3u),
    m = ((n >> 2u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm256_andnot_pd(z, _mm256_loadu_pd(x));
    case 1u:
      return _mm256_andnot_pd(z, _mm256_insertf128_pd(z, _mm_set_pd(-0.0, x[0u]), 0));
    case 2u:
      return _mm256_andnot_pd(z, _mm256_insertf128_pd(z, _mm_loadu_pd(x), 0));
    case 3u:
      return _mm256_andnot_pd(z, _mm256_set_m128d(_mm_set_pd(-0.0, x[2u]), _mm_loadu_pd(x)));
    default: /* should never happen */
      return z;
    }
  }
  register __m256d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm256_loadu_pd(x);
    fr = (r ? rydunrm1(r, (x + 4u)) : _mm256_loadu_pd(x + 4u));
    return pvn_v4d_add_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 2u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rydunrm1(nl, x);
    fr = rydunrm1(nr, (x + nl));
  }
  return pvn_v4d_add_abs(fl, fr);
}

double PVN_FABI(pvn_ryd_nrm1,PVN_RYD_NRM1)(const size_t *const n, const double *const x)
{
  if (!n)
    return -1.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -2.0;
  register const __m256d r = (((uintptr_t)x & (uintptr_t)0x1Fu) ? rydunrm1(*n, x) : ryd_nrm1(*n, x));
#ifdef PVN_NRM_SAFE
  alignas(32) double f[4u];
  _mm256_store_pd(f, r);
  const size_t m = (size_t)4u;
  return PVN_FABI(pvn_red_nrm1,PVN_RED_NRM1)(&m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v4d_add_red(r);
#endif /* ?PVN_NRM_SAFE */
}

static __m256 rys_nrmi(const size_t n, const float *const x)
{
  register const __m256 z = _mm256_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)7u),
    m = ((n >> 3u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm256_andnot_ps(z, _mm256_load_ps(x));
    case 1u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, -0.0f, -0.0f, x[0u]), 0));
    case 2u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, -0.0f, x[1u], x[0u]), 0));
    case 3u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, x[2u], x[1u], x[0u]), 0));
    case 4u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_load_ps(x), 0));
    case 5u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, -0.0f, -0.0f, x[4u]), _mm_load_ps(x)));
    case 6u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, -0.0f, x[5u], x[4u]), _mm_load_ps(x)));
    case 7u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, x[6u], x[5u], x[4u]), _mm_load_ps(x)));
    default: /* should never happen */
      return z;
    }
  }
  register __m256 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm256_load_ps(x);
    fr = (r ? rys_nrmi(r, (x + 8u)) : _mm256_load_ps(x + 8u));
    return pvn_v8s_max_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 3u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rys_nrmi(nl, x);
    fr = rys_nrmi(nr, (x + nl));
  }
  return pvn_v8s_max_abs(fl, fr);
}

static __m256 rysunrmi(const size_t n, const float *const x)
{
  register const __m256 z = _mm256_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)7u),
    m = ((n >> 3u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm256_andnot_ps(z, _mm256_loadu_ps(x));
    case 1u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, -0.0f, -0.0f, x[0u]), 0));
    case 2u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, -0.0f, x[1u], x[0u]), 0));
    case 3u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, x[2u], x[1u], x[0u]), 0));
    case 4u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_loadu_ps(x), 0));
    case 5u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, -0.0f, -0.0f, x[4u]), _mm_loadu_ps(x)));
    case 6u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, -0.0f, x[5u], x[4u]), _mm_loadu_ps(x)));
    case 7u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, x[6u], x[5u], x[4u]), _mm_loadu_ps(x)));
    default: /* should never happen */
      return z;
    }
  }
  register __m256 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm256_loadu_ps(x);
    fr = (r ? rysunrmi(r, (x + 8u)) : _mm256_loadu_ps(x + 8u));
    return pvn_v8s_max_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 3u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rysunrmi(nl, x);
    fr = rysunrmi(nr, (x + nl));
  }
  return pvn_v8s_max_abs(fl, fr);
}

float PVN_FABI(pvn_rys_nrmi,PVN_RYS_NRMI)(const size_t *const n, const float *const x)
{
  if (!n)
    return -1.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -2.0f;
  register const __m256 r = (((uintptr_t)x & (uintptr_t)0x1Fu) ? rysunrmi(*n, x) : rys_nrmi(*n, x));
#ifdef PVN_NRM_SAFE
  alignas(32) float f[8u];
  _mm256_store_ps(f, r);
  const size_t m = (size_t)8u;
  return PVN_FABI(pvn_res_nrmi,PVN_RES_NRMI)(&m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v8s_max_red(r);
#endif /* ?PVN_NRM_SAFE */
}

static __m256d ryd_nrmi(const size_t n, const double *const x)
{
  register const __m256d z = _mm256_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)3u),
    m = ((n >> 2u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm256_andnot_pd(z, _mm256_load_pd(x));
    case 1u:
      return _mm256_andnot_pd(z, _mm256_insertf128_pd(z, _mm_set_pd(-0.0, x[0u]), 0));
    case 2u:
      return _mm256_andnot_pd(z, _mm256_insertf128_pd(z, _mm_load_pd(x), 0));
    case 3u:
      return _mm256_andnot_pd(z, _mm256_set_m128d(_mm_set_pd(-0.0, x[2u]), _mm_load_pd(x)));
    default: /* should never happen */
      return z;
    }
  }
  register __m256d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm256_load_pd(x);
    fr = (r ? ryd_nrmi(r, (x + 4u)) : _mm256_load_pd(x + 4u));
    return pvn_v4d_max_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 2u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn ryd_nrmi(nl, x);
    fr = ryd_nrmi(nr, (x + nl));
  }
  return pvn_v4d_max_abs(fl, fr);
}

static __m256d rydunrmi(const size_t n, const double *const x)
{
  register const __m256d z = _mm256_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)3u),
    m = ((n >> 2u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm256_andnot_pd(z, _mm256_loadu_pd(x));
    case 1u:
      return _mm256_andnot_pd(z, _mm256_insertf128_pd(z, _mm_set_pd(-0.0, x[0u]), 0));
    case 2u:
      return _mm256_andnot_pd(z, _mm256_insertf128_pd(z, _mm_loadu_pd(x), 0));
    case 3u:
      return _mm256_andnot_pd(z, _mm256_set_m128d(_mm_set_pd(-0.0, x[2u]), _mm_loadu_pd(x)));
    default: /* should never happen */
      return z;
    }
  }
  register __m256d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm256_loadu_pd(x);
    fr = (r ? rydunrmi(r, (x + 4u)) : _mm256_loadu_pd(x + 4u));
    return pvn_v4d_max_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 2u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rydunrmi(nl, x);
    fr = rydunrmi(nr, (x + nl));
  }
  return pvn_v4d_max_abs(fl, fr);
}

double PVN_FABI(pvn_ryd_nrmi,PVN_RYD_NRMI)(const size_t *const n, const double *const x)
{
  if (!n)
    return -1.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -2.0;
  register const __m256d r = (((uintptr_t)x & (uintptr_t)0x1Fu) ? rydunrmi(*n, x) : ryd_nrmi(*n, x));
#ifdef PVN_NRM_SAFE
  alignas(32) double f[4u];
  _mm256_store_pd(f, r);
  const size_t m = (size_t)4u;
  return PVN_FABI(pvn_red_nrmi,PVN_RED_NRMI)(&m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v4d_max_red(r);
#endif /* ?PVN_NRM_SAFE */
}
#ifdef __AVX512F__
static __m512 rzs_nrmf(const size_t n, const float *const x)
{
  register const __m512 z = _mm512_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)15u),
    m = ((n >> 4u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case  0u:
      return _mm512_andnot_ps(z, _mm512_load_ps(x));
    case  1u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,     (__mmask16)1u, x));
    case  2u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,     (__mmask16)3u, x));
    case  3u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,     (__mmask16)7u, x));
    case  4u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,    (__mmask16)15u, x));
    case  5u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,    (__mmask16)31u, x));
    case  6u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,    (__mmask16)63u, x));
    case  7u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,   (__mmask16)127u, x));
    case  8u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,   (__mmask16)255u, x));
    case  9u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,   (__mmask16)511u, x));
    case 10u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,  (__mmask16)1023u, x));
    case 11u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,  (__mmask16)2047u, x));
    case 12u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,  (__mmask16)4095u, x));
    case 13u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,  (__mmask16)8191u, x));
    case 14u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z, (__mmask16)16383u, x));
    case 15u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z, (__mmask16)32767u, x));
    default: /* should never happen */
      return z;
    }
  }
  register __m512 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm512_load_ps(x);
    fr = (r ? rzs_nrmf(r, (x + 16u)) : _mm512_load_ps(x + 16u));
    return pvn_v16s_hypot(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 4u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rzs_nrmf(nl, x);
    fr = rzs_nrmf(nr, (x + nl));
  }
  return pvn_v16s_hypot(fl, fr);
}

static __m512 rzsunrmf(const size_t n, const float *const x)
{
  register const __m512 z = _mm512_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)15u),
    m = ((n >> 4u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case  0u:
      return _mm512_andnot_ps(z, _mm512_loadu_ps(x));
    case  1u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,     (__mmask16)1u, x));
    case  2u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,     (__mmask16)3u, x));
    case  3u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,     (__mmask16)7u, x));
    case  4u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,    (__mmask16)15u, x));
    case  5u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,    (__mmask16)31u, x));
    case  6u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,    (__mmask16)63u, x));
    case  7u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,   (__mmask16)127u, x));
    case  8u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,   (__mmask16)255u, x));
    case  9u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,   (__mmask16)511u, x));
    case 10u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,  (__mmask16)1023u, x));
    case 11u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,  (__mmask16)2047u, x));
    case 12u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,  (__mmask16)4095u, x));
    case 13u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,  (__mmask16)8191u, x));
    case 14u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z, (__mmask16)16383u, x));
    case 15u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z, (__mmask16)32767u, x));
    default: /* should never happen */
      return z;
    }
  }
  register __m512 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm512_loadu_ps(x);
    fr = (r ? rzsunrmf(r, (x + 16u)) : _mm512_loadu_ps(x + 16u));
    return pvn_v16s_hypot(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 4u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rzsunrmf(nl, x);
    fr = rzsunrmf(nr, (x + nl));
  }
  return pvn_v16s_hypot(fl, fr);
}

float PVN_FABI(pvn_rzs_nrmf,PVN_RZS_NRMF)(const size_t *const n, const float *const x)
{
  if (!n)
    return -1.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -2.0f;
  register const __m512 r = (((uintptr_t)x & (uintptr_t)0x3Fu) ? rzsunrmf(*n, x) : rzs_nrmf(*n, x));
#ifdef PVN_NRM_SAFE
  alignas(64) float f[16u];
  _mm512_store_ps(f, r);
  const size_t m = (size_t)16u;
  return PVN_FABI(pvn_res_nrmf,PVN_RES_NRMF)(&m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v16s_hypot_red(r);
#endif /* ?PVN_NRM_SAFE */
}

static __m512d rzd_nrmf(const size_t n, const double *const x)
{
  register const __m512d z = _mm512_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)7u),
    m = ((n >> 3u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm512_andnot_pd(z, _mm512_load_pd(x));
    case 1u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,   (__mmask8)1u, x));
    case 2u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,   (__mmask8)3u, x));
    case 3u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,   (__mmask8)7u, x));
    case 4u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,  (__mmask8)15u, x));
    case 5u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,  (__mmask8)31u, x));
    case 6u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,  (__mmask8)63u, x));
    case 7u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z, (__mmask8)127u, x));
    default: /* should never happen */
      return z;
    }
  }
  register __m512d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm512_load_pd(x);
    fr = (r ? rzd_nrmf(r, (x + 8u)) : _mm512_load_pd(x + 8u));
    return pvn_v8d_hypot(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 3u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rzd_nrmf(nl, x);
    fr = rzd_nrmf(nr, (x + nl));
  }
  return pvn_v8d_hypot(fl, fr);
}

static __m512d rzdunrmf(const size_t n, const double *const x)
{
  register const __m512d z = _mm512_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  size_t
    r = (n & (size_t)7u),
    m = ((n >> 3u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm512_andnot_pd(z, _mm512_loadu_pd(x));
    case 1u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,   (__mmask8)1u, x));
    case 2u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,   (__mmask8)3u, x));
    case 3u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,   (__mmask8)7u, x));
    case 4u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,  (__mmask8)15u, x));
    case 5u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,  (__mmask8)31u, x));
    case 6u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,  (__mmask8)63u, x));
    case 7u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z, (__mmask8)127u, x));
    default: /* should never happen */
      return z;
    }
  }
  register __m512d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm512_loadu_pd(x);
    fr = (r ? rzdunrmf(r, (x + 8u)) : _mm512_loadu_pd(x + 8u));
    return pvn_v8d_hypot(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 3u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rzdunrmf(nl, x);
    fr = rzdunrmf(nr, (x + nl));
  }
  return pvn_v8d_hypot(fl, fr);
}

double PVN_FABI(pvn_rzd_nrmf,PVN_RZD_NRMF)(const size_t *const n, const double *const x)
{
  if (!n)
    return -1.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -2.0;
  register
#if (!defined(PVN_NRM_SAFE) || ((PVN_NRM_SAFE & 4) == 0))
    const
#endif /* ?PVN_NRM_SAFE */
    __m512d r = (((uintptr_t)x & (uintptr_t)0x3Fu) ? rzdunrmf(*n, x) : rzd_nrmf(*n, x));
#ifdef PVN_NRM_SAFE
#if ((PVN_NRM_SAFE & 4) != 0)
  PVN_V8D_SORT(r);
#endif /* ?PVN_NRM_SAFE */
  alignas(64) double f[8u];
  _mm512_store_pd(f, r);
  const size_t m = (size_t)8u;
  return PVN_FABI(pvn_red_nrmf,PVN_RED_NRMF)(&m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v8d_hypot_red(r);
#endif /* ?PVN_NRM_SAFE */
}

static __m512 rzs_nrm1(const size_t n, const float *const x)
{
  register const __m512 z = _mm512_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)15u),
    m = ((n >> 4u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case  0u:
      return _mm512_andnot_ps(z, _mm512_load_ps(x));
    case  1u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,     (__mmask16)1u, x));
    case  2u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,     (__mmask16)3u, x));
    case  3u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,     (__mmask16)7u, x));
    case  4u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,    (__mmask16)15u, x));
    case  5u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,    (__mmask16)31u, x));
    case  6u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,    (__mmask16)63u, x));
    case  7u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,   (__mmask16)127u, x));
    case  8u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,   (__mmask16)255u, x));
    case  9u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,   (__mmask16)511u, x));
    case 10u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,  (__mmask16)1023u, x));
    case 11u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,  (__mmask16)2047u, x));
    case 12u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,  (__mmask16)4095u, x));
    case 13u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,  (__mmask16)8191u, x));
    case 14u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z, (__mmask16)16383u, x));
    case 15u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z, (__mmask16)32767u, x));
    default: /* should never happen */
      return z;
    }
  }
  register __m512 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm512_load_ps(x);
    fr = (r ? rzs_nrm1(r, (x + 16u)) : _mm512_load_ps(x + 16u));
    return pvn_v16s_add_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 4u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rzs_nrm1(nl, x);
    fr = rzs_nrm1(nr, (x + nl));
  }
  return pvn_v16s_add_abs(fl, fr);
}

static __m512 rzsunrm1(const size_t n, const float *const x)
{
  register const __m512 z = _mm512_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)15u),
    m = ((n >> 4u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case  0u:
      return _mm512_andnot_ps(z, _mm512_loadu_ps(x));
    case  1u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,     (__mmask16)1u, x));
    case  2u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,     (__mmask16)3u, x));
    case  3u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,     (__mmask16)7u, x));
    case  4u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,    (__mmask16)15u, x));
    case  5u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,    (__mmask16)31u, x));
    case  6u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,    (__mmask16)63u, x));
    case  7u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,   (__mmask16)127u, x));
    case  8u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,   (__mmask16)255u, x));
    case  9u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,   (__mmask16)511u, x));
    case 10u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,  (__mmask16)1023u, x));
    case 11u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,  (__mmask16)2047u, x));
    case 12u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,  (__mmask16)4095u, x));
    case 13u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,  (__mmask16)8191u, x));
    case 14u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z, (__mmask16)16383u, x));
    case 15u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z, (__mmask16)32767u, x));
    default: /* should never happen */
      return z;
    }
  }
  register __m512 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm512_loadu_ps(x);
    fr = (r ? rzsunrm1(r, (x + 16u)) : _mm512_loadu_ps(x + 16u));
    return pvn_v16s_add_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 4u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rzsunrm1(nl, x);
    fr = rzsunrm1(nr, (x + nl));
  }
  return pvn_v16s_add_abs(fl, fr);
}

float PVN_FABI(pvn_rzs_nrm1,PVN_RZS_NRM1)(const size_t *const n, const float *const x)
{
  if (!n)
    return -1.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -2.0f;
  register const __m512 r = (((uintptr_t)x & (uintptr_t)0x3Fu) ? rzsunrm1(*n, x) : rzs_nrm1(*n, x));
#ifdef PVN_NRM_SAFE
  alignas(64) float f[16u];
  _mm512_store_ps(f, r);
  const size_t m = (size_t)16u;
  return PVN_FABI(pvn_res_nrm1,PVN_RES_NRM1)(&m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v16s_add_red(r);
#endif /* ?PVN_NRM_SAFE */
}

static __m512d rzd_nrm1(const size_t n, const double *const x)
{
  register const __m512d z = _mm512_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)7u),
    m = ((n >> 3u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm512_andnot_pd(z, _mm512_load_pd(x));
    case 1u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,   (__mmask8)1u, x));
    case 2u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,   (__mmask8)3u, x));
    case 3u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,   (__mmask8)7u, x));
    case 4u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,  (__mmask8)15u, x));
    case 5u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,  (__mmask8)31u, x));
    case 6u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,  (__mmask8)63u, x));
    case 7u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z, (__mmask8)127u, x));
    default: /* should never happen */
      return z;
    }
  }
  register __m512d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm512_load_pd(x);
    fr = (r ? rzd_nrm1(r, (x + 8u)) : _mm512_load_pd(x + 8u));
    return pvn_v8d_add_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 3u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rzd_nrm1(nl, x);
    fr = rzd_nrm1(nr, (x + nl));
  }
  return pvn_v8d_add_abs(fl, fr);
}

static __m512d rzdunrm1(const size_t n, const double *const x)
{
  register const __m512d z = _mm512_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  size_t
    r = (n & (size_t)7u),
    m = ((n >> 3u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm512_andnot_pd(z, _mm512_loadu_pd(x));
    case 1u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,   (__mmask8)1u, x));
    case 2u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,   (__mmask8)3u, x));
    case 3u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,   (__mmask8)7u, x));
    case 4u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,  (__mmask8)15u, x));
    case 5u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,  (__mmask8)31u, x));
    case 6u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,  (__mmask8)63u, x));
    case 7u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z, (__mmask8)127u, x));
    default: /* should never happen */
      return z;
    }
  }
  register __m512d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm512_loadu_pd(x);
    fr = (r ? rzdunrm1(r, (x + 8u)) : _mm512_loadu_pd(x + 8u));
    return pvn_v8d_add_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 3u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rzdunrm1(nl, x);
    fr = rzdunrm1(nr, (x + nl));
  }
  return pvn_v8d_add_abs(fl, fr);
}

double PVN_FABI(pvn_rzd_nrm1,PVN_RZD_NRM1)(const size_t *const n, const double *const x)
{
  if (!n)
    return -1.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -2.0;
  register
#if (!defined(PVN_NRM_SAFE) || ((PVN_NRM_SAFE & 2) == 0))
    const
#endif /* ?PVN_NRM_SAFE */
    __m512d r = (((uintptr_t)x & (uintptr_t)0x3Fu) ? rzdunrm1(*n, x) : rzd_nrm1(*n, x));
#ifdef PVN_NRM_SAFE
#if ((PVN_NRM_SAFE & 2) != 0)
  PVN_V8D_SORT(r);
#endif /* ?PVN_NRM_SAFE */
  alignas(64) double f[8u];
  _mm512_store_pd(f, r);
  const size_t m = (size_t)8u;
  return PVN_FABI(pvn_red_nrm1,PVN_RED_NRM1)(&m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v8d_add_red(r);
#endif /* ?PVN_NRM_SAFE */
}

static __m512 rzs_nrmi(const size_t n, const float *const x)
{
  register const __m512 z = _mm512_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)15u),
    m = ((n >> 4u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case  0u:
      return _mm512_andnot_ps(z, _mm512_load_ps(x));
    case  1u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,     (__mmask16)1u, x));
    case  2u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,     (__mmask16)3u, x));
    case  3u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,     (__mmask16)7u, x));
    case  4u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,    (__mmask16)15u, x));
    case  5u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,    (__mmask16)31u, x));
    case  6u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,    (__mmask16)63u, x));
    case  7u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,   (__mmask16)127u, x));
    case  8u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,   (__mmask16)255u, x));
    case  9u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,   (__mmask16)511u, x));
    case 10u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,  (__mmask16)1023u, x));
    case 11u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,  (__mmask16)2047u, x));
    case 12u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,  (__mmask16)4095u, x));
    case 13u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,  (__mmask16)8191u, x));
    case 14u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z, (__mmask16)16383u, x));
    case 15u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z, (__mmask16)32767u, x));
    default: /* should never happen */
      return z;
    }
  }
  register __m512 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm512_load_ps(x);
    fr = (r ? rzs_nrmi(r, (x + 16u)) : _mm512_load_ps(x + 16u));
    return pvn_v16s_max_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 4u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rzs_nrmi(nl, x);
    fr = rzs_nrmi(nr, (x + nl));
  }
  return pvn_v16s_max_abs(fl, fr);
}

static __m512 rzsunrmi(const size_t n, const float *const x)
{
  register const __m512 z = _mm512_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)15u),
    m = ((n >> 4u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case  0u:
      return _mm512_andnot_ps(z, _mm512_loadu_ps(x));
    case  1u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,     (__mmask16)1u, x));
    case  2u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,     (__mmask16)3u, x));
    case  3u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,     (__mmask16)7u, x));
    case  4u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,    (__mmask16)15u, x));
    case  5u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,    (__mmask16)31u, x));
    case  6u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,    (__mmask16)63u, x));
    case  7u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,   (__mmask16)127u, x));
    case  8u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,   (__mmask16)255u, x));
    case  9u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,   (__mmask16)511u, x));
    case 10u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,  (__mmask16)1023u, x));
    case 11u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,  (__mmask16)2047u, x));
    case 12u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,  (__mmask16)4095u, x));
    case 13u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,  (__mmask16)8191u, x));
    case 14u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z, (__mmask16)16383u, x));
    case 15u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z, (__mmask16)32767u, x));
    default: /* should never happen */
      return z;
    }
  }
  register __m512 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm512_loadu_ps(x);
    fr = (r ? rzsunrmi(r, (x + 16u)) : _mm512_loadu_ps(x + 16u));
    return pvn_v16s_max_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 4u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rzsunrmi(nl, x);
    fr = rzsunrmi(nr, (x + nl));
  }
  return pvn_v16s_max_abs(fl, fr);
}

float PVN_FABI(pvn_rzs_nrmi,PVN_RZS_NRMI)(const size_t *const n, const float *const x)
{
  if (!n)
    return -1.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -2.0f;
  register const __m512 r = (((uintptr_t)x & (uintptr_t)0x3Fu) ? rzsunrmi(*n, x) : rzs_nrmi(*n, x));
#ifdef PVN_NRM_SAFE
  alignas(64) float f[16u];
  _mm512_store_ps(f, r);
  const size_t m = (size_t)16u;
  return PVN_FABI(pvn_res_nrmi,PVN_RES_NRMI)(&m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v16s_max_red(r);
#endif /* ?PVN_NRM_SAFE */
}

static __m512d rzd_nrmi(const size_t n, const double *const x)
{
  register const __m512d z = _mm512_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)7u),
    m = ((n >> 3u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm512_andnot_pd(z, _mm512_load_pd(x));
    case 1u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,   (__mmask8)1u, x));
    case 2u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,   (__mmask8)3u, x));
    case 3u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,   (__mmask8)7u, x));
    case 4u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,  (__mmask8)15u, x));
    case 5u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,  (__mmask8)31u, x));
    case 6u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,  (__mmask8)63u, x));
    case 7u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z, (__mmask8)127u, x));
    default: /* should never happen */
      return z;
    }
  }
  register __m512d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm512_load_pd(x);
    fr = (r ? rzd_nrmi(r, (x + 8u)) : _mm512_load_pd(x + 8u));
    return pvn_v8d_max_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 3u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rzd_nrmi(nl, x);
    fr = rzd_nrmi(nr, (x + nl));
  }
  return pvn_v8d_max_abs(fl, fr);
}

static __m512d rzdunrmi(const size_t n, const double *const x)
{
  register const __m512d z = _mm512_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  size_t
    r = (n & (size_t)7u),
    m = ((n >> 3u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm512_andnot_pd(z, _mm512_loadu_pd(x));
    case 1u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,   (__mmask8)1u, x));
    case 2u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,   (__mmask8)3u, x));
    case 3u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,   (__mmask8)7u, x));
    case 4u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,  (__mmask8)15u, x));
    case 5u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,  (__mmask8)31u, x));
    case 6u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,  (__mmask8)63u, x));
    case 7u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z, (__mmask8)127u, x));
    default: /* should never happen */
      return z;
    }
  }
  register __m512d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm512_loadu_pd(x);
    fr = (r ? rzdunrmi(r, (x + 8u)) : _mm512_loadu_pd(x + 8u));
    return pvn_v8d_max_abs(fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 3u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rzdunrmi(nl, x);
    fr = rzdunrmi(nr, (x + nl));
  }
  return pvn_v8d_max_abs(fl, fr);
}

double PVN_FABI(pvn_rzd_nrmi,PVN_RZD_NRMI)(const size_t *const n, const double *const x)
{
  if (!n)
    return -1.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -2.0;
  register const __m512d r = (((uintptr_t)x & (uintptr_t)0x3Fu) ? rzdunrmi(*n, x) : rzd_nrmi(*n, x));
#ifdef PVN_NRM_SAFE
  alignas(64) double f[8u];
  _mm512_store_pd(f, r);
  const size_t m = (size_t)8u;
  return PVN_FABI(pvn_red_nrmi,PVN_RED_NRMI)(&m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v8d_max_red(r);
#endif /* ?PVN_NRM_SAFE */
}
#endif /* __AVX512F__ */

#if (defined(PVN_USE_INTEL) && (PVN_USE_INTEL > 1) && (defined(__INTEL_CLANG_COMPILER) || defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)))
static __m128 rxs_nrmp(const float p, const size_t n, const float *const x)
{
  register const __m128 z = _mm_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)3u),
    m = ((n >> 2u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm_andnot_ps(z, _mm_load_ps(x));
    case 1u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, -0.0f, -0.0f, x[0u]));
    case 2u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, -0.0f, x[1u], x[0u]));
    case 3u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, x[2u], x[1u], x[0u]));
    default: /* should never happen */
      return z;
    }
  }
  register __m128 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm_load_ps(x);
    fr = (r ? rxs_nrmp(p, r, (x + 4u)) : _mm_load_ps(x + 4u));
    return pvn_v4s_lp(p, fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 2u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rxs_nrmp(p, nl, x);
    fr = rxs_nrmp(p, nr, (x + nl));
  }
  return pvn_v4s_lp(p, fl, fr);
}

static __m128 rxsunrmp(const float p, const size_t n, const float *const x)
{
  register const __m128 z = _mm_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)3u),
    m = ((n >> 2u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm_andnot_ps(z, _mm_loadu_ps(x));
    case 1u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, -0.0f, -0.0f, x[0u]));
    case 2u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, -0.0f, x[1u], x[0u]));
    case 3u:
      return _mm_andnot_ps(z, _mm_set_ps(-0.0f, x[2u], x[1u], x[0u]));
    default: /* should never happen */
      return z;
    }
  }
  register __m128 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm_loadu_ps(x);
    fr = (r ? rxsunrmp(p, r, (x + 4u)) : _mm_loadu_ps(x + 4u));
    return pvn_v4s_lp(p, fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 2u);
  const size_t nr = (n - nl);

  cilk_scope {
    fl = cilk_spawn rxsunrmp(p, nl, x);
    fr = rxsunrmp(p, nr, (x + nl));
  }
  return pvn_v4s_lp(p, fl, fr);
}

float PVN_FABI(pvn_rxs_nrmp,PVN_RXS_NRMP)(const float *const p, const size_t *const n, const float *const x)
{
  if (!p || !(*p > 0.0f))
    return -1.0f;
  if (!n)
    return -2.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -3.0f;
  if (*p == 1.0f)
    return PVN_FABI(pvn_rxs_nrm1,PVN_RXS_NRM1)(n, x);
  if (*p == 2.0f)
    return PVN_FABI(pvn_rxs_nrmf,PVN_RXS_NRMF)(n, x);
  if (isinf(*p))
    return PVN_FABI(pvn_rxs_nrmi,PVN_RXS_NRMI)(n, x);
  register const __m128 r = (((uintptr_t)x & (uintptr_t)0x0Fu) ? rxsunrmp(*p, *n, x) : rxs_nrmp(*p, *n, x));
#ifdef PVN_NRM_SAFE
  alignas(16) float f[4u];
  _mm_store_ps(f, r);
  const size_t m = (size_t)4u;
  return PVN_FABI(pvn_res_nrmp,PVN_RES_NRMP)(p, &m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v4s_lp_red(*p, r);
#endif /* ?PVN_NRM_SAFE */
}

static __m128d rxd_nrmp(const double p, const size_t n, const double *const x)
{
  register const __m128d z = _mm_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)1u),
    m = ((n >> 1u) + r);
  if (m == (size_t)1u) {
    if (r)
      return _mm_andnot_pd(z, _mm_set_pd(-0.0, *x));
    else
      return _mm_andnot_pd(z, _mm_load_pd(x));
  }
  register __m128d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm_load_pd(x);
    fr = (r ? rxd_nrmp(p, r, (x + 2u)) : _mm_load_pd(x + 2u));
    return pvn_v2d_lp(p, fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 1u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rxd_nrmp(p, nl, x);
    fr = rxd_nrmp(p, nr, (x + nl));
  }
  return pvn_v2d_lp(p, fl, fr);
}

static __m128d rxdunrmp(const double p, const size_t n, const double *const x)
{
  register const __m128d z = _mm_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)1u),
    m = ((n >> 1u) + r);
  if (m == (size_t)1u) {
    if (r)
      return _mm_andnot_pd(z, _mm_set_pd(-0.0, *x));
    else
      return _mm_andnot_pd(z, _mm_loadu_pd(x));
  }
  register __m128d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm_loadu_pd(x);
    fr = (r ? rxdunrmp(p, r, (x + 2u)) : _mm_loadu_pd(x + 2u));
    return pvn_v2d_lp(p, fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 1u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rxdunrmp(p, nl, x);
    fr = rxdunrmp(p, nr, (x + nl));
  }
  return pvn_v2d_lp(p, fl, fr);
}

double PVN_FABI(pvn_rxd_nrmp,PVN_RXD_NRMP)(const double *const p, const size_t *const n, const double *const x)
{
  if (!p || !(*p > 0.0))
    return -1.0;
  if (!n)
    return -2.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -3.0;
  if (*p == 1.0)
    return PVN_FABI(pvn_rxd_nrm1,PVN_RXD_NRM1)(n, x);
  if (*p == 2.0)
    return PVN_FABI(pvn_rxd_nrmf,PVN_RXD_NRMF)(n, x);
  if (isinf(*p))
    return PVN_FABI(pvn_rxd_nrmi,PVN_RXD_NRMI)(n, x);
  register const __m128d r = (((uintptr_t)x & (uintptr_t)0x0Fu) ? rxdunrmp(*p, *n, x) : rxd_nrmp(*p, *n, x));
#ifdef PVN_NRM_SAFE
  alignas(16) double f[2u];
  _mm_store_pd(f, r);
  return pvn_v1d_lp(*p, f[0u], f[1u]);
#else /* !PVN_NRM_SAFE */
  return pvn_v2d_lp_red(*p, r);
#endif /* ?PVN_NRM_SAFE */
}

static __m256 rys_nrmp(const float p, const size_t n, const float *const x)
{
  register const __m256 z = _mm256_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)7u),
    m = ((n >> 3u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm256_andnot_ps(z, _mm256_load_ps(x));
    case 1u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, -0.0f, -0.0f, x[0u]), 0));
    case 2u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, -0.0f, x[1u], x[0u]), 0));
    case 3u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, x[2u], x[1u], x[0u]), 0));
    case 4u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_load_ps(x), 0));
    case 5u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, -0.0f, -0.0f, x[4u]), _mm_load_ps(x)));
    case 6u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, -0.0f, x[5u], x[4u]), _mm_load_ps(x)));
    case 7u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, x[6u], x[5u], x[4u]), _mm_load_ps(x)));
    default: /* should never happen */
      return z;
    }
  }
  register __m256 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm256_load_ps(x);
    fr = (r ? rys_nrmp(p, r, (x + 8u)) : _mm256_load_ps(x + 8u));
    return pvn_v8s_lp(p, fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 3u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rys_nrmp(p, nl, x);
    fr = rys_nrmp(p, nr, (x + nl));
  }
  return pvn_v8s_lp(p, fl, fr);
}

static __m256 rysunrmp(const float p, const size_t n, const float *const x)
{
  register const __m256 z = _mm256_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)7u),
    m = ((n >> 3u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm256_andnot_ps(z, _mm256_loadu_ps(x));
    case 1u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, -0.0f, -0.0f, x[0u]), 0));
    case 2u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, -0.0f, x[1u], x[0u]), 0));
    case 3u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_set_ps(-0.0f, x[2u], x[1u], x[0u]), 0));
    case 4u:
      return _mm256_andnot_ps(z, _mm256_insertf128_ps(z, _mm_loadu_ps(x), 0));
    case 5u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, -0.0f, -0.0f, x[4u]), _mm_loadu_ps(x)));
    case 6u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, -0.0f, x[5u], x[4u]), _mm_loadu_ps(x)));
    case 7u:
      return _mm256_andnot_ps(z, _mm256_set_m128(_mm_set_ps(-0.0f, x[6u], x[5u], x[4u]), _mm_loadu_ps(x)));
    default: /* should never happen */
      return z;
    }
  }
  register __m256 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm256_loadu_ps(x);
    fr = (r ? rysunrmp(p, r, (x + 8u)) : _mm256_loadu_ps(x + 8u));
    return pvn_v8s_lp(p, fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 3u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rysunrmp(p, nl, x);
    fr = rysunrmp(p, nr, (x + nl));
  }
  return pvn_v8s_lp(p, fl, fr);
}

float PVN_FABI(pvn_rys_nrmp,PVN_RYS_NRMP)(const float *const p, const size_t *const n, const float *const x)
{
  if (!p || !(*p > 0.0f))
    return -1.0f;
  if (!n)
    return -2.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -3.0f;
  if (*p == 1.0f)
    return PVN_FABI(pvn_rys_nrm1,PVN_RYS_NRM1)(n, x);
  if (*p == 2.0f)
    return PVN_FABI(pvn_rys_nrmf,PVN_RYS_NRMF)(n, x);
  if (isinf(*p))
    return PVN_FABI(pvn_rys_nrmi,PVN_RYS_NRMI)(n, x);
  register const __m256 r = (((uintptr_t)x & (uintptr_t)0x1Fu) ? rysunrmp(*p, *n, x) : rys_nrmp(*p, *n, x));
#ifdef PVN_NRM_SAFE
  alignas(32) float f[8u];
  _mm256_store_ps(f, r);
  const size_t m = (size_t)8u;
  return PVN_FABI(pvn_res_nrmp,PVN_RES_NRMP)(p, &m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v8s_lp_red(r);
#endif /* ?PVN_NRM_SAFE */
}

static __m256d ryd_nrmp(const double p, const size_t n, const double *const x)
{
  register const __m256d z = _mm256_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)3u),
    m = ((n >> 2u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm256_andnot_pd(z, _mm256_load_pd(x));
    case 1u:
      return _mm256_andnot_pd(z, _mm256_insertf128_pd(z, _mm_set_pd(-0.0, x[0u]), 0));
    case 2u:
      return _mm256_andnot_pd(z, _mm256_insertf128_pd(z, _mm_load_pd(x), 0));
    case 3u:
      return _mm256_andnot_pd(z, _mm256_set_m128d(_mm_set_pd(-0.0, x[2u]), _mm_load_pd(x)));
    default: /* should never happen */
      return z;
    }
  }
  register __m256d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm256_load_pd(x);
    fr = (r ? ryd_nrmp(p, r, (x + 4u)) : _mm256_load_pd(x + 4u));
    return pvn_v4d_lp(p, fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 2u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn ryd_nrmp(p, nl, x);
    fr = ryd_nrmp(p, nr, (x + nl));
  }
  return pvn_v4d_lp(p, fl, fr);
}

static __m256d rydunrmp(const double p, const size_t n, const double *const x)
{
  register const __m256d z = _mm256_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)3u),
    m = ((n >> 2u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm256_andnot_pd(z, _mm256_loadu_pd(x));
    case 1u:
      return _mm256_andnot_pd(z, _mm256_insertf128_pd(z, _mm_set_pd(-0.0, x[0u]), 0));
    case 2u:
      return _mm256_andnot_pd(z, _mm256_insertf128_pd(z, _mm_loadu_pd(x), 0));
    case 3u:
      return _mm256_andnot_pd(z, _mm256_set_m128d(_mm_set_pd(-0.0, x[2u]), _mm_loadu_pd(x)));
    default: /* should never happen */
      return z;
    }
  }
  register __m256d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm256_loadu_pd(x);
    fr = (r ? rydunrmp(p, r, (x + 4u)) : _mm256_loadu_pd(x + 4u));
    return pvn_v4d_lp(p, fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 2u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rydunrmp(p, nl, x);
    fr = rydunrmp(p, nr, (x + nl));
  }
  return pvn_v4d_lp(p, fl, fr);
}

double PVN_FABI(pvn_ryd_nrmp,PVN_RYD_NRMP)(const double *const p, const size_t *const n, const double *const x)
{
  if (!p || !(*p > 0.0))
    return -1.0;
  if (!n)
    return -2.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -3.0;
  if (*p == 1.0)
    return PVN_FABI(pvn_ryd_nrm1,PVN_RYD_NRM1)(n, x);
  if (*p == 2.0)
    return PVN_FABI(pvn_ryd_nrmf,PVN_RYD_NRMF)(n, x);
  if (isinf(*p))
    return PVN_FABI(pvn_ryd_nrmi,PVN_RYD_NRMI)(n, x);
  register const __m256d r = (((uintptr_t)x & (uintptr_t)0x1Fu) ? rydunrmp(*p, *n, x) : ryd_nrmp(*p, *n, x));
#ifdef PVN_NRM_SAFE
  alignas(32) double f[4u];
  _mm256_store_pd(f, r);
  const size_t m = (size_t)4u;
  return PVN_FABI(pvn_red_nrmp,PVN_RED_NRMP)(p, &m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v4d_lp_red(*p, r);
#endif /* ?PVN_NRM_SAFE */
}
#ifdef __AVX512F__
static __m512 rzs_nrmp(const float p, const size_t n, const float *const x)
{
  register const __m512 z = _mm512_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)15u),
    m = ((n >> 4u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case  0u:
      return _mm512_andnot_ps(z, _mm512_load_ps(x));
    case  1u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,     (__mmask16)1u, x));
    case  2u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,     (__mmask16)3u, x));
    case  3u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,     (__mmask16)7u, x));
    case  4u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,    (__mmask16)15u, x));
    case  5u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,    (__mmask16)31u, x));
    case  6u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,    (__mmask16)63u, x));
    case  7u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,   (__mmask16)127u, x));
    case  8u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,   (__mmask16)255u, x));
    case  9u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,   (__mmask16)511u, x));
    case 10u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,  (__mmask16)1023u, x));
    case 11u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,  (__mmask16)2047u, x));
    case 12u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,  (__mmask16)4095u, x));
    case 13u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z,  (__mmask16)8191u, x));
    case 14u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z, (__mmask16)16383u, x));
    case 15u:
      return _mm512_andnot_ps(z, _mm512_mask_load_ps(z, (__mmask16)32767u, x));
    default: /* should never happen */
      return z;
    }
  }
  register __m512 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm512_load_ps(x);
    fr = (r ? rzs_nrmp(p, r, (x + 16u)) : _mm512_load_ps(x + 16u));
    return pvn_v16s_lp(p, fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 4u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rzs_nrmp(p, nl, x);
    fr = rzs_nrmp(p, nr, (x + nl));
  }
  return pvn_v16s_lp(p, fl, fr);
}

static __m512 rzsunrmp(const float p, const size_t n, const float *const x)
{
  register const __m512 z = _mm512_set1_ps(-0.0f);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)15u),
    m = ((n >> 4u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case  0u:
      return _mm512_andnot_ps(z, _mm512_loadu_ps(x));
    case  1u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,     (__mmask16)1u, x));
    case  2u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,     (__mmask16)3u, x));
    case  3u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,     (__mmask16)7u, x));
    case  4u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,    (__mmask16)15u, x));
    case  5u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,    (__mmask16)31u, x));
    case  6u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,    (__mmask16)63u, x));
    case  7u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,   (__mmask16)127u, x));
    case  8u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,   (__mmask16)255u, x));
    case  9u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,   (__mmask16)511u, x));
    case 10u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,  (__mmask16)1023u, x));
    case 11u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,  (__mmask16)2047u, x));
    case 12u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,  (__mmask16)4095u, x));
    case 13u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z,  (__mmask16)8191u, x));
    case 14u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z, (__mmask16)16383u, x));
    case 15u:
      return _mm512_andnot_ps(z, _mm512_mask_loadu_ps(z, (__mmask16)32767u, x));
    default: /* should never happen */
      return z;
    }
  }
  register __m512 fl, fr;
  if (m == (size_t)2u) {
    fl = _mm512_loadu_ps(x);
    fr = (r ? rzsunrmp(p, r, (x + 16u)) : _mm512_loadu_ps(x + 16u));
    return pvn_v16s_lp(p, fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 4u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rzsunrmp(p, nl, x);
    fr = rzsunrmp(p, nr, (x + nl));
  }
  return pvn_v16s_lp(p, fl, fr);
}

float PVN_FABI(pvn_rzs_nrmp,PVN_RZS_NRMP)(const float *const p, const size_t *const n, const float *const x)
{
  if (!p || !(*p > 0.0f))
    return -1.0f;
  if (!n)
    return -2.0f;
  if (!*n)
    return -0.0f;
  if (!x)
    return -3.0f;
  if (*p == 1.0f)
    return PVN_FABI(pvn_rzs_nrm1,PVN_RZS_NRM1)(n, x);
  if (*p == 2.0f)
    return PVN_FABI(pvn_rzs_nrmf,PVN_RZS_NRMF)(n, x);
  if (isinf(*p))
    return PVN_FABI(pvn_rzs_nrmi,PVN_RZS_NRMI)(n, x);
  register const __m512 r = (((uintptr_t)x & (uintptr_t)0x3Fu) ? rzsunrmp(*p, *n, x) : rzs_nrmp(*p, *n, x));
#ifdef PVN_NRM_SAFE
  alignas(64) float f[16u];
  _mm512_store_ps(f, r);
  const size_t m = (size_t)16u;
  return PVN_FABI(pvn_res_nrmp,PVN_RES_NRMP)(p, &m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v16s_lp_red(*p, r);
#endif /* ?PVN_NRM_SAFE */
}

static __m512d rzd_nrmp(const double p, const size_t n, const double *const x)
{
  register const __m512d z = _mm512_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  const size_t
    r = (n & (size_t)7u),
    m = ((n >> 3u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm512_andnot_pd(z, _mm512_load_pd(x));
    case 1u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,   (__mmask8)1u, x));
    case 2u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,   (__mmask8)3u, x));
    case 3u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,   (__mmask8)7u, x));
    case 4u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,  (__mmask8)15u, x));
    case 5u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,  (__mmask8)31u, x));
    case 6u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z,  (__mmask8)63u, x));
    case 7u:
      return _mm512_andnot_pd(z, _mm512_mask_load_pd(z, (__mmask8)127u, x));
    default: /* should never happen */
      return z;
    }
  }
  register __m512d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm512_load_pd(x);
    fr = (r ? rzd_nrmp(p, r, (x + 8u)) : _mm512_load_pd(x + 8u));
    return pvn_v8d_lp(p, fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 3u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rzd_nrmp(p, nl, x);
    fr = rzd_nrmp(p, nr, (x + nl));
  }
  return pvn_v8d_lp(p, fl, fr);
}

static __m512d rzdunrmp(const double p, const size_t n, const double *const x)
{
  register const __m512d z = _mm512_set1_pd(-0.0);
#ifndef NDEBUG
  if (!n)
    return z;
#endif /* !NDEBUG */
  size_t
    r = (n & (size_t)7u),
    m = ((n >> 3u) + (r != (size_t)0u));
  if (m == (size_t)1u) {
    switch ((unsigned)r) {
    case 0u:
      return _mm512_andnot_pd(z, _mm512_loadu_pd(x));
    case 1u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,   (__mmask8)1u, x));
    case 2u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,   (__mmask8)3u, x));
    case 3u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,   (__mmask8)7u, x));
    case 4u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,  (__mmask8)15u, x));
    case 5u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,  (__mmask8)31u, x));
    case 6u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z,  (__mmask8)63u, x));
    case 7u:
      return _mm512_andnot_pd(z, _mm512_mask_loadu_pd(z, (__mmask8)127u, x));
    default: /* should never happen */
      return z;
    }
  }
  register __m512d fl, fr;
  if (m == (size_t)2u) {
    fl = _mm512_loadu_pd(x);
    fr = (r ? rzdunrmp(p, r, (x + 8u)) : _mm512_loadu_pd(x + 8u));
    return pvn_v8d_lp(p, fl, fr);
  }
  const size_t nl = (((m >> 1u) + (m & (size_t)1u)) << 3u);
  const size_t nr = (n - nl);
  cilk_scope {
    fl = cilk_spawn rzdunrmp(p, nl, x);
    fr = rzdunrmp(p, nr, (x + nl));
  }
  return pvn_v8d_lp(p, fl, fr);
}

double PVN_FABI(pvn_rzd_nrmp,PVN_RZD_NRMP)(const double *const p, const size_t *const n, const double *const x)
{
  if (!p || !(*p > 0.0))
    return -1.0;
  if (!n)
    return -2.0;
  if (!*n)
    return -0.0;
  if (!x)
    return -3.0;
  if (*p == 1.0)
    return PVN_FABI(pvn_rzd_nrm1,PVN_RZD_NRM1)(n, x);
  if (*p == 2.0)
    return PVN_FABI(pvn_rzd_nrmf,PVN_RZD_NRMF)(n, x);
  if (isinf(*p))
    return PVN_FABI(pvn_rzd_nrmi,PVN_RZD_NRMI)(n, x);
  register
#if (!defined(PVN_NRM_SAFE) || ((PVN_NRM_SAFE & 8) == 0))
    const
#endif /* ?PVN_NRM_SAFE */
    __m512d r = (((uintptr_t)x & (uintptr_t)0x3Fu) ? rzdunrmp(*p, *n, x) : rzd_nrmp(*p, *n, x));
#ifdef PVN_NRM_SAFE
#if ((PVN_NRM_SAFE & 8) != 0)
  PVN_V8D_SORT(r);
#endif /* ?PVN_NRM_SAFE */
  alignas(64) double f[8u];
  _mm512_store_pd(f, r);
  const size_t m = (size_t)8u;
  return PVN_FABI(pvn_red_nrmp,PVN_RED_NRMP)(p, &m, f);
#else /* !PVN_NRM_SAFE */
  return pvn_v8d_lp_red(*p, r);
#endif /* ?PVN_NRM_SAFE */
}
#endif /* __AVX512F__ */
#endif /* Intel */
#endif /* __AVX__ && __FMA__ */
#endif /* ?PVN_TEST */
