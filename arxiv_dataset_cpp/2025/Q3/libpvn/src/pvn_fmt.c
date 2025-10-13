#include "pvn.h"

#ifdef PVN_TEST
int main(int argc, char *argv[])
{
  if (1 < argc) {
    (void)fprintf(stderr, "%s takes no arguments\n", *argv);
    return EXIT_FAILURE;
  }
  char s[46u] = { '\0' };
  (void)printf(" FLT_TRUE_MIN=%s\n", pvn_stoa(s, FLT_TRUE_MIN));
  (void)printf(" FLT_MIN     =%s\n", pvn_stoa(s, FLT_MIN));
  (void)printf(" FLT_MIN_EXP =%6d\n", FLT_MIN_EXP);
  (void)printf(" FLT_EPSILON =%s\n", pvn_stoa(s, FLT_EPSILON));
  (void)printf(" FLT_MAX     =%s\n", pvn_stoa(s, FLT_MAX));
  (void)printf(" FLT_MAX_EXP =%6d\n", FLT_MAX_EXP);
  (void)printf(" DBL_TRUE_MIN=%s\n", pvn_dtoa(s, DBL_TRUE_MIN));
  (void)printf(" DBL_MIN     =%s\n", pvn_dtoa(s, DBL_MIN));
  (void)printf(" DBL_MIN_EXP =%6d\n", DBL_MIN_EXP);
  (void)printf(" DBL_EPSILON =%s\n", pvn_dtoa(s, DBL_EPSILON));
  (void)printf(" DBL_MAX     =%s\n", pvn_dtoa(s, DBL_MAX));
  (void)printf(" DBL_MAX_EXP =%6d\n", DBL_MAX_EXP);
#if (!defined(_WIN32) || defined(_DLL))
  (void)printf("LDBL_TRUE_MIN=%s\n", pvn_xtoa(s, LDBL_TRUE_MIN));
  (void)printf("LDBL_MIN     =%s\n", pvn_xtoa(s, LDBL_MIN));
  (void)printf("LDBL_MIN_EXP =%6d\n", LDBL_MIN_EXP);
  (void)printf("LDBL_EPSILON =%s\n", pvn_xtoa(s, LDBL_EPSILON));
  (void)printf("LDBL_MAX     =%s\n", pvn_xtoa(s, LDBL_MAX));
  (void)printf("LDBL_MAX_EXP =%6d\n", LDBL_MAX_EXP);
#ifdef PVN_QUADMATH
  (void)printf("FLT128_TRUE_MIN=%s\n", pvn_qtoa(s, FLT128_TRUE_MIN));
  (void)printf("FLT128_MIN     =%s\n", pvn_qtoa(s, FLT128_MIN));
  (void)printf("FLT128_MIN_EXP =%6d\n", FLT128_MIN_EXP);
  (void)printf("FLT128_EPSILON =%s\n", pvn_qtoa(s, FLT128_EPSILON));
  (void)printf("FLT128_MAX     =%s\n", pvn_qtoa(s, FLT128_MAX));
  (void)printf("FLT128_MAX_EXP =%6d\n", FLT128_MAX_EXP);
#endif /* PVN_QUADMATH */
#endif /* !_WIN32 || _DLL */
  /* assume Windows Terminal on Windows */
  (void)printf(PVN_ANSI_BLACK "Black" PVN_ANSI_CLEAR "\n");
  (void)printf(PVN_ANSI_RED "Red" PVN_ANSI_CLEAR "\n");
  (void)printf(PVN_ANSI_GREEN "Green" PVN_ANSI_CLEAR "\n");
  (void)printf(PVN_ANSI_YELLOW "Yellow" PVN_ANSI_CLEAR "\n");
  (void)printf(PVN_ANSI_BLUE "Blue" PVN_ANSI_CLEAR "\n");
  (void)printf(PVN_ANSI_MAGENTA "Magenta" PVN_ANSI_CLEAR "\n");
  (void)printf(PVN_ANSI_CYAN "Cyan" PVN_ANSI_CLEAR "\n");
  (void)printf(PVN_ANSI_WHITE "White" PVN_ANSI_CLEAR "\n");
  (void)printf("Foreground\n");
  return EXIT_SUCCESS;
}
#else /* !PVN_TEST */
size_t pvn_atoz(const char *const s)
{
  char *e = (char*)NULL;
  const size_t z = ((s && *s) ? (size_t)strtoull(s, &e, 0) : (size_t)0u);
  return ((e && *e) ? (size_t)0u : z);
}

char *pvn_stoa(char *const s, const float x)
{
  if (s) {
    int l = sprintf((char*)memset(s, 0, (size_t)17u), "%# -16.9E", (double)x);
    if (l <= 0)
      return (char*)NULL;
    char *d = s + 16;
    char *e = strrchr(s, 'E');
    if (e) {
      for (--d; isblank(*d); --d)
        *d = '\0';
      e += 2;
      l = (int)(strchr(e, '\0') - e);
      if (l >= 2)
        return s;
      d = s + 16;
      e += l;
      for (int i = 0; i < l; ++i)
        *--d = *--e;
      for (--d; isdigit(*d); --d)
        *d = '0';
    }
    else
      for (--d; !*d; --d)
        *d = ' ';
  }
  return s;
}

char *pvn_dtoa(char *const s, const double x)
{
  if (s) {
    int l = sprintf((char*)memset(s, 0, (size_t)26u), "%# -25.17E", x);
    if (l <= 0)
      return (char*)NULL;
    char *d = s + 25;
    char *e = strrchr(s, 'E');
    if (e) {
      for (--d; isblank(*d); --d)
        *d = '\0';
      e += 2;
      l = (int)(strchr(e, '\0') - e);
      if (l >= 3)
        return s;
      d = s + 25;
      e += l;
      for (int i = 0; i < l; ++i)
        *--d = *--e;
      for (--d; isdigit(*d); --d)
        *d = '0';
    }
    else
      for (--d; !*d; --d)
        *d = ' ';
  }
  return s;
}

#if (!defined(_WIN32) || defined(_DLL))
#ifdef _WIN32
long double pvn_atox(const char *const s)
{
  char *e = (char*)NULL;
  const __float128 x = ((s && *s) ? strtoflt128(s, &e) : 0.0q);
  return ((e && *e) ? 0.0L : (long double)x);
}

char *pvn_xtoa(char *const s, const long double x)
{
  if (s) {
    const __float128 x_ = (__float128)x;
    int l = quadmath_snprintf((char*)memset(s, 0, (size_t)31u), (size_t)31u, "%# -30.21QE", x_);
    if (l <= 0)
      return (char*)NULL;
    char *d = s + 30;
    char *e = strrchr(s, 'E');
    if (e) {
      for (--d; isblank(*d); --d)
        *d = '\0';
      e += 2;
      l = (int)(strchr(e, '\0') - e);
      if (l >= 4)
        return s;
      d = s + 30;
      e += l;
      for (int i = 0; i < l; ++i)
        *--d = *--e;
      for (--d; isdigit(*d); --d)
        *d = '0';
    }
    else
      for (--d; !*d; --d)
        *d = ' ';
  }
  return s;
}
#else /* !_WIN32 */
long double pvn_atox(const char *const s)
{
  char *e = (char*)NULL;
  const long double x = ((s && *s) ? strtold(s, &e) : 0.0L);
  return ((e && *e) ? 0.0L : x);
}

char *pvn_xtoa(char *const s, const long double x)
{
  if (s) {
#ifdef __x86_64__
    int l = sprintf((char*)memset(s, 0, (size_t)31u), "%# -30.21LE", x);
    if (l <= 0)
      return (char*)NULL;
    char *d = s + 30;
    char *e = strrchr(s, 'E');
    if (e) {
      for (--d; isblank(*d); --d)
        *d = '\0';
      e += 2;
      l = (int)(strchr(e, '\0') - e);
      if (l >= 4)
        return s;
      d = s + 30;
      e += l;
      for (int i = 0; i < l; ++i)
        *--d = *--e;
      for (--d; isdigit(*d); --d)
        *d = '0';
    }
#else /* !__x86_64__ */
    int l = sprintf((char*)memset(s, 0, (size_t)46u), "%# -45.36LE", x);
    if (l <= 0)
      return (char*)NULL;
    char *d = s + 45;
    char *e = strrchr(s, 'E');
    if (e) {
      for (--d; isblank(*d); --d)
        *d = '\0';
      e += 2;
      l = (int)(strchr(e, '\0') - e);
      if (l >= 4)
        return s;
      d = s + 45;
      e += l;
      for (int i = 0; i < l; ++i)
        *--d = *--e;
      for (--d; isdigit(*d); --d)
        *d = '0';
    }
#endif /* ?__x86_64__ */
    else
      for (--d; !*d; --d)
        *d = ' ';
  }
  return s;
}
#endif /* ?_WIN32 */
#ifdef PVN_QUADMATH
__float128 pvn_atoq(const char *const s)
{
  char *e = (char*)NULL;
  const __float128 x = ((s && *s) ? strtoflt128(s, &e) : 0.0q);
  return ((e && *e) ? 0.0q : x);
}

char *pvn_qtoa(char *const s, const __float128 x)
{
  if (s) {
    int l = quadmath_snprintf((char*)memset(s, 0, (size_t)46u), (size_t)46u, "%# -45.36QE", x);
    if (l <= 0)
      return (char*)NULL;
    char *d = s + 45;
    char *e = strrchr(s, 'E');
    if (e) {
      for (--d; isblank(*d); --d)
        *d = '\0';
      e += 2;
      l = (int)(strchr(e, '\0') - e);
      if (l >= 4)
        return s;
      d = s + 45;
      e += l;
      for (int i = 0; i < l; ++i)
        *--d = *--e;
      for (--d; isdigit(*d); --d)
        *d = '0';
    }
    else
      for (--d; !*d; --d)
        *d = ' ';
  }
  return s;
}
#else /* !PVN_QUADMATH */
long double pvn_atoq(const char *const s)
{
  return pvn_atox(s);
}

char *pvn_qtoa(char *const s, const long double x)
{
  return pvn_xtoa(s, x);
}
#endif /* ?PVN_QUADMATH */
#endif /* !_WIN32 || _DLL */
#endif /* ?PVN_TEST */
