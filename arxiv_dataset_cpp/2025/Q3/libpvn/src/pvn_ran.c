#include "pvn.h"

#if (defined(__RDRND__) && !defined(__NVCOMPILER))
#include <immintrin.h>
#endif /* __RDRND__ */

#ifdef PVN_TEST
int main(int argc, char *argv[])
{
  if (1 < argc) {
    (void)fprintf(stderr, "%s takes no arguments\n", *argv);
    return EXIT_FAILURE;
  }
  const int u = PVN_FABI(pvn_ran_open,PVN_RAN_OPEN)();
  (void)printf("open=%d\n", u);
  char s[46] = { '\0' };
  (void)printf("safe_f=%s\n", pvn_stoa(s, PVN_FABI(pvn_ran_safe_f,PVN_RAN_SAFE_F)(&u, (const int*)NULL)));
  (void)printf("safe_d=%s\n", pvn_dtoa(s, PVN_FABI(pvn_ran_safe,PVN_RAN_SAFE)(&u, (const int*)NULL)));
#if (!defined(_WIN32) || defined(_DLL))
  (void)printf("safe_l=%s\n", pvn_xtoa(s, PVN_FABI(pvn_ran_safe_l,PVN_RAN_SAFE_L)(&u, (const int*)NULL)));
  (void)printf("safe_q=%s\n", pvn_qtoa(s, PVN_FABI(pvn_ran_safe_q,PVN_RAN_SAFE_Q)(&u, (const int*)NULL)));
#endif /* !_WIN32 || _DLL */
  (void)printf("close=%d\n", PVN_FABI(pvn_ran_close,PVN_RAN_CLOSE)(&u));
  return EXIT_SUCCESS;
}
#else /* !PVN_TEST */
int PVN_FABI(pvn_ran_open,PVN_RAN_OPEN)()
{
#if (defined(__RDRND__) && !defined(__NVCOMPILER))
  return __RDRND__;
#else /* !__RDRND__ */
  return open("/dev/random", (O_RDONLY | PVN_LF64));
#endif /* ?__RDRND__ */
}

int PVN_FABI(pvn_ran_close,PVN_RAN_CLOSE)(const int *const u)
{
  PVN_ASSERT(u);
#if (defined(__RDRND__) && !defined(__NVCOMPILER))
  return ((*u < 0) ? -1 : 0);
#else /* !__RDRND__ */
  return close(*u);
#endif /* ?__RDRND__ */
}

int PVN_FABI(pvn_ran_16,PVN_RAN_16)(const int *const u, uint16_t *const r)
{
  PVN_ASSERT(u);
  PVN_ASSERT(r);
#if (defined(__RDRND__) && !defined(__NVCOMPILER))
  return _rdrand16_step(r);
#else /* !__RDRND__ */
  return (read(*u, r, sizeof(*r)) == (ssize_t)(sizeof(*r)));
#endif /* ?__RDRND__ */
}

int PVN_FABI(pvn_ran_32,PVN_RAN_32)(const int *const u, uint32_t *const r)
{
  PVN_ASSERT(u);
  PVN_ASSERT(r);
#if (defined(__RDRND__) && !defined(__NVCOMPILER))
  return _rdrand32_step(r);
#else /* !__RDRND__ */
  return (read(*u, r, sizeof(*r)) == (ssize_t)(sizeof(*r)));
#endif /* ?__RDRND__ */
}

int PVN_FABI(pvn_ran_64,PVN_RAN_64)(const int *const u, uint64_t *const r)
{
  PVN_ASSERT(u);
  PVN_ASSERT(r);
#if (defined(__RDRND__) && !defined(__NVCOMPILER))
  return _rdrand64_step((unsigned long long*)r);
#else /* !__RDRND__ */
  return (read(*u, r, sizeof(*r)) == (ssize_t)(sizeof(*r)));
#endif /* ?__RDRND__ */
}

float PVN_FABI(pvn_ran_safe_f,PVN_RAN_SAFE_F)(const int *const u, const int *const p)
{
  PVN_ASSERT(u);
  const float rmin = (p ? scalbnf(FLT_MIN, *p) : FLT_MIN);
  const float rmax = (FLT_MAX * 0.25f);
  float a = FLT_MAX, r = 0.0f;
  if (*u < 0)
    return r;
  while (!(a >= rmin) || !(a <= rmax)) {
#if (defined(__RDRND__) && !defined(__NVCOMPILER))
    while (!_rdrand32_step((unsigned*)&r)) /**/;
    a = fabsf(r);
#else /* !__RDRND__ */
    ssize_t s = 0;
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
    {
      s = read(*u, &r, sizeof(r));
    }
    if (s == (ssize_t)sizeof(r))
      a = fabsf(r);
    else if (s < 0)
      return -0.0f;
    else /* s >= 0 */
      return 0.0f;
#endif /* ?__RDRND__ */
  }
  return r;
}

float PVN_FABI(pvn_ran_f,PVN_RAN_F)(const int *const u)
{
  PVN_ASSERT(u);
  float r = ((*u < 0) ? __builtin_nanf("") : 0.0f);
#if (defined(__RDRND__) && !defined(__NVCOMPILER))
  while (!_rdrand32_step((unsigned*)&r)) /**/;
#else /* !__RDRND__ */
  ssize_t s = 0;
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
  {
    s = read(*u, &r, sizeof(r));
  }
  if (s != (ssize_t)sizeof(r))
    r = __builtin_nanf("");
#endif /* ?__RDRND__ */
  return r;
}

double PVN_FABI(pvn_ran_safe,PVN_RAN_SAFE)(const int *const u, const int *const p)
{
  PVN_ASSERT(u);
  const double rmin = (p ? scalbn(DBL_MIN, *p) : DBL_MIN);
  const double rmax = (DBL_MAX * 0.25);
  double a = DBL_MAX, r = 0.0;
  if (*u < 0)
    return r;
  while (!(a >= rmin) || !(a <= rmax)) {
#if (defined(__RDRND__) && !defined(__NVCOMPILER))
    while (!_rdrand64_step((unsigned long long*)&r)) /**/;
    a = fabs(r);
#else /* !__RDRND__ */
    ssize_t s = 0;
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
    {
      s = read(*u, &r, sizeof(r));
    }
    if (s == (ssize_t)sizeof(r))
      a = fabs(r);
    else if (s < 0)
      return -0.0;
    else /* s >= 0 */
      return 0.0;
#endif /* ?__RDRND__ */
  }
  return r;
}

double PVN_FABI(pvn_ran,PVN_RAN)(const int *const u)
{
  PVN_ASSERT(u);
  double r = ((*u < 0) ? __builtin_nan("") : 0.0);
#if (defined(__RDRND__) && !defined(__NVCOMPILER))
  while (!_rdrand64_step((unsigned long long*)&r)) /**/;
#else /* !__RDRND__ */
  ssize_t s = 0;
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
  {
    s = read(*u, &r, sizeof(r));
  }
  if (s != (ssize_t)sizeof(r))
    r = __builtin_nan("");
#endif /* ?__RDRND__ */
  return r;
}

long double PVN_FABI(pvn_ran_safe_l,PVN_RAN_SAFE_L)(const int *const u, const int *const p)
{
  PVN_ASSERT(u);
  const long double rmin = (p ? scalbnl(LDBL_MIN, *p) : LDBL_MIN);
  const long double rmax = (LDBL_MAX * 0.25L);
  long double a = LDBL_MAX, r = 0.0L;
  if (*u < 0)
    return r;
  while (!(a >= rmin) || !(a <= rmax)) {
#if (defined(__RDRND__) && !defined(__NVCOMPILER))
    while (!_rdrand64_step((unsigned long long*)&r)) /**/;
    while (!_rdrand16_step((unsigned short*)((unsigned long long*)&r + 1))) /**/;
    a = fabsl(r);
#else /* !__RDRND__ */
    ssize_t s = 0;
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
    {
#ifdef __x86_64__
      s = read(*u, &r, (size_t)10u);
#else /* !__x86_64__ */
      s = read(*u, &r, sizeof(r));
#endif /* ?__x86_64__ */
    }
#ifdef __x86_64__
    if (s == (ssize_t)10)
#else /* !__x86_64__ */
    if (s == (ssize_t)sizeof(r))
#endif /* ?__x86_64__ */
      a = fabsl(r);
    else if (s < 0)
      return -0.0L;
    else /* s >= 0 */
      return 0.0L;
#endif /* ?__RDRND__ */
  }
  return r;
}

long double PVN_FABI(pvn_ran_l,PVN_RAN_L)(const int *const u)
{
  PVN_ASSERT(u);
  long double r = ((*u < 0) ? __builtin_nanl("") : 0.0L);
#if (defined(__RDRND__) && !defined(__NVCOMPILER))
  while (!_rdrand64_step((unsigned long long*)&r)) /**/;
  while (!_rdrand16_step((unsigned short*)((unsigned long long*)&r + 1))) /**/;
#else /* !__RDRND__ */
  ssize_t s = 0;
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
  {
#ifdef __x86_64__
      s = read(*u, &r, (size_t)10u);
#else /* !__x86_64__ */
      s = read(*u, &r, sizeof(r));
#endif /* ?__x86_64__ */
  }
#ifdef __x86_64__
  if (s != (ssize_t)10)
#else /* !__x86_64__ */
  if (s != (ssize_t)sizeof(r))
#endif /* ?__x86_64__ */
    r = __builtin_nanl("");
#endif /* ?__RDRND__ */
  return r;
}

#ifdef PVN_QUADMATH
__float128 PVN_FABI(pvn_ran_safe_q,PVN_RAN_SAFE_Q)(const int *const u, const int *const p)
{
  PVN_ASSERT(u);
  const __float128 rmin = (p ? scalbnq(FLT128_MIN, *p) : FLT128_MIN);
  const __float128 rmax = (FLT128_MAX * 0.25q);
  __float128 a = FLT128_MAX, r = 0.0q;
  if (*u < 0)
    return r;
  while (!(a >= rmin) || !(a <= rmax)) {
#if (defined(__RDRND__) && !defined(__NVCOMPILER))
    while (!_rdrand64_step((unsigned long long*)&r)) /**/;
    while (!_rdrand64_step((unsigned long long*)&r + 1)) /**/;
    a = fabsq(r);
#else /* !__RDRND__ */
    ssize_t s = 0;
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
    {
      s = read(*u, &r, sizeof(r));
    }
    if (s == (ssize_t)sizeof(r))
      a = fabsq(r);
    else if (s < 0)
      return -0.0q;
    else /* s >= 0 */
      return 0.0q;
#endif /* ?__RDRND__ */
  }
  return r;
}

__float128 PVN_FABI(pvn_ran_q,PVN_RAN_Q)(const int *const u)
{
  PVN_ASSERT(u);
  __float128 r = ((*u < 0) ? (0.0q / 0.0q) : 0.0q);
#if (defined(__RDRND__) && !defined(__NVCOMPILER))
  while (!_rdrand64_step((unsigned long long*)&r)) /**/;
  while (!_rdrand64_step((unsigned long long*)&r + 1)) /**/;
#else /* !__RDRND__ */
  ssize_t s = 0;
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
  {
    s = read(*u, &r, sizeof(r));
  }
  if (s != (ssize_t)sizeof(r))
    r = (0.0q / 0.0q);
#endif /* ?__RDRND__ */
  return r;
}
#else /* !PVN_QUADMATH */
long double PVN_FABI(pvn_ran_safe_q,PVN_RAN_SAFE_Q)(const int *const u, const int *const p)
{
  return PVN_FABI(pvn_ran_safe_l,PVN_RAN_SAFE_L)(u, p);
}

long double PVN_FABI(pvn_ran_q,PVN_RAN_Q)(const int *const u)
{
  return PVN_FABI(pvn_ran_l,PVN_RAN_L)(u);
}
#endif /* ?PVN_QUADMATH */
#endif /* ?PVN_TEST */
