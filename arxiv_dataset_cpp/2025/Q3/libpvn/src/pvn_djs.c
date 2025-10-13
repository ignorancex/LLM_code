#include "pvn.h"

/* !!! LITTLE ENDIAN ONLY !!! */

#ifdef PVN_TEST
int main(int argc, char *argv[])
{
  if (argc != 3) {
    (void)fprintf(stderr, "%s p q\n", argv[0]);
    return EXIT_FAILURE;
  }
  const unsigned p = (unsigned)atoi(argv[1]);
  const unsigned q = (unsigned)atoi(argv[2]);
  const float f = 0.0f;
  double d = 0.0;
  PVN_FABI(pvn_djs_denc,PVN_DJS_DENC)(&d, &f, &p, &q);
  char s[46] = { '\0' };
  (void)fprintf(stdout, "d=%s\n", pvn_dtoa(s, d));
#if (!defined(_WIN32) || defined(_DLL))
  d = 0.0;
#ifdef __x86_64__
  long double x = 0.0L;
  PVN_FABI(pvn_djs_xenc,PVN_DJS_XENC)(&x, &d, &p, &q);
  (void)fprintf(stdout, "x=%s\n", pvn_xtoa(s, x));
#endif /* __x86_64__ */
#ifdef PVN_QUADMATH
  __float128 e = 0.0q;
#else /* !PVN_QUADMATH */
  long double e = 0.0L;
#endif /* ?PVN_QUADMATH */
  PVN_FABI(pvn_djs_qenc,PVN_DJS_QENC)(&e, &d, &p, &q);
  (void)fprintf(stdout, "q=%s\n", pvn_qtoa(s, e));
#endif /* !_WIN32 || _DLL */
  return EXIT_SUCCESS;
}
#else /* !PVN_TEST */
void PVN_FABI(pvn_djs_denc,PVN_DJS_DENC)(double *const e, const float *const f, const unsigned *const p, const unsigned *const q)
{
  PVN_ASSERT(e);
  PVN_ASSERT(f);
  PVN_ASSERT(p);
  PVN_ASSERT(q);
  PVN_ASSERT(*p);
  PVN_ASSERT(*q);
  *e = (double)*f;
#ifdef NDEBUG
  *(unsigned long long*)e = (*(const unsigned long long*)e | ((unsigned long long)(*p - 1u) << 14u) | (unsigned long long)(*q - 1u));
#else /* !NDEBUG */
  *(unsigned long long*)e = (*(const unsigned long long*)e | ((unsigned long long)((*p - 1u) & 16383u) << 14u) | (unsigned long long)((*q - 1u) & 16383u));
#endif /* ?NDEBUG */
}

void PVN_FABI(pvn_djs_ddec,PVN_DJS_DDEC)(const double *const e, unsigned *const p, unsigned *const q)
{
  PVN_ASSERT(e);
  PVN_ASSERT(p);
  PVN_ASSERT(q);
  unsigned long long u = *(const unsigned long long*)e;
  *q = ((unsigned)(u & 16383ull) + 1u);
  u >>= 14u;
  *p = ((unsigned)(u & 16383ull) + 1u);
}

void PVN_FABI(pvn_djs_xenc,PVN_DJS_XENC)(long double *const e, const double *const d, const unsigned *const p, const unsigned *const q)
{
  PVN_ASSERT(e);
  PVN_ASSERT(d);
  PVN_ASSERT(p);
  PVN_ASSERT(q);
  PVN_ASSERT(*p);
  PVN_ASSERT(*q);
  *e = (long double)*d;
#ifdef __x86_64__
#ifdef NDEBUG
  *(unsigned long long*)e = (*(const unsigned long long*)e | ((unsigned long long)(*p - 1u) << 5u) | (unsigned long long)(*q - 1u));
#else /* !NDEBUG */
  *(unsigned long long*)e = (*(const unsigned long long*)e | ((unsigned long long)((*p - 1u) & 31u) << 5u) | (unsigned long long)((*q - 1u) & 31u));
#endif /* ?NDEBUG */
#else /* !__x86_64__ */
#ifdef NDEBUG
  *(unsigned long long*)e = (*(const unsigned long long*)e | ((unsigned long long)(*p - 1u) << 30u) | (unsigned long long)(*q - 1u));
#else /* !NDEBUG */
  *(unsigned long long*)e = (*(const unsigned long long*)e | ((unsigned long long)((*p - 1u) & 1073741823u) << 30u) | (unsigned long long)((*q - 1u) & 1073741823u));
#endif /* ?NDEBUG */
#endif /* ?__x86_64__ */
}

void PVN_FABI(pvn_djs_xdec,PVN_DJS_XDEC)(const long double *const e, unsigned *const p, unsigned *const q)
{
  PVN_ASSERT(e);
  PVN_ASSERT(p);
  PVN_ASSERT(q);
  unsigned long long u = *(const unsigned long long*)e;
#ifdef __x86_64__
  *q = ((unsigned)(u & 31ull) + 1u);
  u >>= 5u;
  *p = ((unsigned)(u & 31ull) + 1u);
#else /* !__x86_64__ */
  *q = ((unsigned)(u & 1073741823ull) + 1u);
  u >>= 30u;
  *p = ((unsigned)(u & 1073741823ull) + 1u);
#endif /* ?__x86_64__ */
}

#ifdef PVN_QUADMATH
void PVN_FABI(pvn_djs_qenc,PVN_DJS_QENC)(__float128 *const e, const double *const d, const unsigned *const p, const unsigned *const q)
{
  PVN_ASSERT(e);
  PVN_ASSERT(d);
  PVN_ASSERT(p);
  PVN_ASSERT(q);
  PVN_ASSERT(*p);
  PVN_ASSERT(*q);
  *e = (__float128)*d;
#ifdef NDEBUG
  *(unsigned long long*)e = (*(const unsigned long long*)e | ((unsigned long long)(*p - 1u) << 30u) | (unsigned long long)(*q - 1u));
#else /* !NDEBUG */
  *(unsigned long long*)e = (*(const unsigned long long*)e | ((unsigned long long)((*p - 1u) & 1073741823u) << 30u) | (unsigned long long)((*q - 1u) & 1073741823u));
#endif /* ?NDEBUG */
}

void PVN_FABI(pvn_djs_qdec,PVN_DJS_QDEC)(const __float128 *const e, unsigned *const p, unsigned *const q)
{
  PVN_ASSERT(e);
  PVN_ASSERT(p);
  PVN_ASSERT(q);
  unsigned long long u = *(const unsigned long long*)e;
  *q = ((unsigned)(u & 1073741823ull) + 1u);
  u >>= 30u;
  *p = ((unsigned)(u & 1073741823ull) + 1u);
}
#else /* !PVN_QUADMATH */
void PVN_FABI(pvn_djs_qenc,PVN_DJS_QENC)(long double *const e, const double *const d, const unsigned *const p, const unsigned *const q)
{
  PVN_FABI(pvn_djs_xenc,PVN_DJS_XENC)(e, d, p, q);
}

void PVN_FABI(pvn_djs_qdec,PVN_DJS_QDEC)(const long double *const e, unsigned *const p, unsigned *const q)
{
  PVN_FABI(pvn_djs_xdec,PVN_DJS_XDEC)(e, p, q);
}
#endif /* ?PVN_QUADMATH */
#endif /* ?PVN_TEST */
