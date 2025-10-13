#ifdef PVN_SV2_SAFE
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif /* __GNUC__ && !__clang__ */
#pragma STDC FENV_ACCESS ON
#endif /* PVN_SV2_SAFE */

#include "pvn.h"

#ifdef PVN_TEST
int main(int argc, char *argv[])
{
  if ((argc != 3) && (argc != 4)) {
    (void)fprintf(stderr, "%s (S|D|C|Z) N [P]\n", *argv);
    return EXIT_FAILURE;
  }
  const char t = (char)toupper(*(argv[1]));
  const char T = (((t == 'C') || (t == 'Z')) ? 'H' : 'T');
  int n = atoi(argv[2]);
  const int upper = (n < 0);
  n = abs(n);
  const int p = ((argc == 4) ? atoi(argv[3]) : 0);
  const int *const pp = (p ? &p : (const int*)NULL);
  char s[46] = { '\0' };
  if (!n) {
#ifdef PVN_QUADMATH
    __float128 E[4] = { 0.0q, 0.0q, 0.0q, 0.0q };
#else /* !PVN_QUADMATH */
    long double E[4] = { 0.0L, 0.0L, 0.0L, 0.0L };
#endif /* ?PVN_QUADMATH */
    if (t == 'S') {
      float a11 = 0.0f, a21 = 0.0f, a12 = 0.0f, a22 = 0.0f;
      (void)printf("G(1,1) = ");
      if (scanf("%e", &a11) != 1)
        return EXIT_FAILURE;
      (void)printf("G(2,1) = ");
      if (scanf("%e", &a21) != 1)
        return EXIT_FAILURE;
      (void)printf("G(1,2) = ");
      if (scanf("%e", &a12) != 1)
        return EXIT_FAILURE;
      (void)printf("G(2,2) = ");
      if (scanf("%e", &a22) != 1)
        return EXIT_FAILURE;
      float u11 = -0.0f, u21 = -0.0f, u12 = -0.0f, u22 = -0.0f, v11 = -0.0f, v21 = -0.0f, v12 = -0.0f, v22 = -0.0f, s1 = -0.0f, s2 = -0.0f;
      int es[3] = { 0, 0, 0 };
      const int knd = PVN_FABI(pvn_sljsv2,PVN_SLJSV2)(&a11, &a21, &a12, &a22, &u11, &u21, &u12, &u22, &v11, &v21, &v12, &v22, &s1, &s2, es);
      (void)printf("knd=%d, es={%d,%d,%d}\n\tG =\n", knd, es[0], es[1], es[2]);
      (void)printf("%s ", pvn_stoa(s, a11));
      (void)printf("%s\n", pvn_stoa(s, a12));
      (void)printf("%s ", pvn_stoa(s, a21));
      (void)printf("%s\n\tU =\n", pvn_stoa(s, a22));
      (void)printf("%s ", pvn_stoa(s, u11));
      (void)printf("%s\n", pvn_stoa(s, u12));
      (void)printf("%s ", pvn_stoa(s, u21));
      (void)printf("%s\n\tV =\n", pvn_stoa(s, u22));
      (void)printf("%s ", pvn_stoa(s, v11));
      (void)printf("%s\n", pvn_stoa(s, v12));
      (void)printf("%s ", pvn_stoa(s, v21));
      (void)printf("%s\n\ts =\n", pvn_stoa(s, v22));
      (void)printf("%s ", pvn_stoa(s, s1));
      (void)printf("%s\n\tS =\n", pvn_stoa(s, s2));
      PVN_FABI(pvn_sqljr2,PVN_SQLJR2)(&a11, &a21, &a12, &a22, &u11, &u21, &u12, &u22, &v11, &v21, &v12, &v22, &s1, &s2, es, E);
      s1 = scalbnf(s1, (es[1] - es[0]));
      s2 = scalbnf(s2, (es[2] - es[0]));
      (void)printf("%s ", pvn_stoa(s, s1));
      (void)printf("%s\n", pvn_stoa(s, s2));
    }
    else if (t == 'D') {
      double a11 = 0.0, a21 = 0.0, a12 = 0.0, a22 = 0.0;
      (void)printf("G(1,1) = ");
      if (scanf("%le", &a11) != 1)
        return EXIT_FAILURE;
      (void)printf("G(2,1) = ");
      if (scanf("%le", &a21) != 1)
        return EXIT_FAILURE;
      (void)printf("G(1,2) = ");
      if (scanf("%le", &a12) != 1)
        return EXIT_FAILURE;
      (void)printf("G(2,2) = ");
      if (scanf("%le", &a22) != 1)
        return EXIT_FAILURE;
      double u11 = -0.0, u21 = -0.0, u12 = -0.0, u22 = -0.0, v11 = -0.0, v21 = -0.0, v12 = -0.0, v22 = -0.0, s1 = -0.0, s2 = -0.0;
      int es[3] = { 0, 0, 0 };
      const int knd = PVN_FABI(pvn_dljsv2,PVN_DLJSV2)(&a11, &a21, &a12, &a22, &u11, &u21, &u12, &u22, &v11, &v21, &v12, &v22, &s1, &s2, es);
      (void)printf("knd=%d, es={%d,%d,%d}\n\tG =\n", knd, es[0], es[1], es[2]);
      (void)printf("%s ", pvn_dtoa(s, a11));
      (void)printf("%s\n", pvn_dtoa(s, a12));
      (void)printf("%s ", pvn_dtoa(s, a21));
      (void)printf("%s\n\tU =\n", pvn_dtoa(s, a22));
      (void)printf("%s ", pvn_dtoa(s, u11));
      (void)printf("%s\n", pvn_dtoa(s, u12));
      (void)printf("%s ", pvn_dtoa(s, u21));
      (void)printf("%s\n\tV =\n", pvn_dtoa(s, u22));
      (void)printf("%s ", pvn_dtoa(s, v11));
      (void)printf("%s\n", pvn_dtoa(s, v12));
      (void)printf("%s ", pvn_dtoa(s, v21));
      (void)printf("%s\n\ts =\n", pvn_dtoa(s, v22));
      (void)printf("%s ", pvn_dtoa(s, s1));
      (void)printf("%s\n\tS =\n", pvn_dtoa(s, s2));
      PVN_FABI(pvn_dqljr2,PVN_DQLJR2)(&a11, &a21, &a12, &a22, &u11, &u21, &u12, &u22, &v11, &v21, &v12, &v22, &s1, &s2, es, E);
      s1 = scalbnf(s1, (es[1] - es[0]));
      s2 = scalbnf(s2, (es[2] - es[0]));
      (void)printf("%s ", pvn_dtoa(s, s1));
      (void)printf("%s\n", pvn_dtoa(s, s2));
    }
    else if (t == 'C') {
      float a11r = 0.0f, a11i = 0.0f, a21r = 0.0f, a21i = 0.0f, a12r = 0.0f, a12i = 0.0f, a22r = 0.0f, a22i = 0.0f;
      (void)printf("G(1,1) = ");
      if (scanf(" (%e,%e)", &a11r, &a11i) != 2)
        return EXIT_FAILURE;
      (void)printf("G(2,1) = ");
      if (scanf(" (%e,%e)", &a21r, &a21i) != 2)
        return EXIT_FAILURE;
      (void)printf("G(1,2) = ");
      if (scanf(" (%e,%e)", &a12r, &a12i) != 2)
        return EXIT_FAILURE;
      (void)printf("G(2,2) = ");
      if (scanf(" (%e,%e)", &a22r, &a22i) != 2)
        return EXIT_FAILURE;
      float
        u11r = -0.0f, u11i = -0.0f, u21r = -0.0f, u21i = -0.0f, u12r = -0.0f, u12i = -0.0f, u22r = -0.0f, u22i = -0.0f,
        v11r = -0.0f, v11i = -0.0f, v21r = -0.0f, v21i = -0.0f, v12r = -0.0f, v12i = -0.0f, v22r = -0.0f, v22i = -0.0f,
        s1 = -0.0f, s2 = -0.0f;
      int es[3] = { 0, 0, 0 };
      const int knd = PVN_FABI(pvn_cljsv2,PVN_CLJSV2)(&a11r, &a11i, &a21r, &a21i, &a12r, &a12i, &a22r, &a22i,
                                  &u11r, &u11i, &u21r, &u21i, &u12r, &u12i, &u22r, &u22i,
                                  &v11r, &v11i, &v21r, &v21i, &v12r, &v12i, &v22r, &v22i,
                                  &s1, &s2, es);
      (void)printf("knd=%d, es={%d,%d,%d}\n", knd, es[0], es[1], es[2]);
      (void)printf("\tG =\n");
      (void)printf("(%s,", pvn_stoa(s, a11r));
      (void)printf("%s) ", pvn_stoa(s, a11i));
      (void)printf("(%s,", pvn_stoa(s, a12r));
      (void)printf("%s)\n", pvn_stoa(s, a12i));
      (void)printf("(%s,", pvn_stoa(s, a21r));
      (void)printf("%s) ", pvn_stoa(s, a21i));
      (void)printf("(%s,", pvn_stoa(s, a22r));
      (void)printf("%s)\n", pvn_stoa(s, a22i));
      (void)printf("\tU =\n");
      (void)printf("(%s,", pvn_stoa(s, u11r));
      (void)printf("%s) ", pvn_stoa(s, u11i));
      (void)printf("(%s,", pvn_stoa(s, u12r));
      (void)printf("%s)\n", pvn_stoa(s, u12i));
      (void)printf("(%s,", pvn_stoa(s, u21r));
      (void)printf("%s) ", pvn_stoa(s, u21i));
      (void)printf("(%s,", pvn_stoa(s, u22r));
      (void)printf("%s)\n", pvn_stoa(s, u22i));
      (void)printf("\tV =\n");
      (void)printf("(%s,", pvn_stoa(s, v11r));
      (void)printf("%s) ", pvn_stoa(s, v11i));
      (void)printf("(%s,", pvn_stoa(s, v12r));
      (void)printf("%s)\n", pvn_stoa(s, v12i));
      (void)printf("(%s,", pvn_stoa(s, v21r));
      (void)printf("%s) ", pvn_stoa(s, v21i));
      (void)printf("(%s,", pvn_stoa(s, v22r));
      (void)printf("%s)\n", pvn_stoa(s, v22i));
      (void)printf("\ts =\n");
      (void)printf("%s ", pvn_stoa(s, s1));
      (void)printf("%s\n\tS =\n", pvn_stoa(s, s2));
      PVN_FABI(pvn_cyljr2,PVN_CYLJR2)(&a11r, &a11i, &a21r, &a21i, &a12r, &a12i, &a22r, &a22i,
                  &u11r, &u11i, &u21r, &u21i, &u12r, &u12i, &u22r, &u22i,
                  &v11r, &v11i, &v21r, &v21i, &v12r, &v12i, &v22r, &v22i,
                  &s1, &s2, es, E);
      s1 = scalbnf(s1, (es[1] - es[0]));
      s2 = scalbnf(s2, (es[2] - es[0]));
      (void)printf("%s ", pvn_stoa(s, s1));
      (void)printf("%s\n", pvn_stoa(s, s2));
    }
    else if (t == 'Z') {
      double a11r = 0.0, a11i = 0.0, a21r = 0.0, a21i = 0.0, a12r = 0.0, a12i = 0.0, a22r = 0.0, a22i = 0.0;
      (void)printf("G(1,1) = ");
      if (scanf(" (%le,%le)", &a11r, &a11i) != 2)
        return EXIT_FAILURE;
      (void)printf("G(2,1) = ");
      if (scanf(" (%le,%le)", &a21r, &a21i) != 2)
        return EXIT_FAILURE;
      (void)printf("G(1,2) = ");
      if (scanf(" (%le,%le)", &a12r, &a12i) != 2)
        return EXIT_FAILURE;
      (void)printf("G(2,2) = ");
      if (scanf(" (%le,%le)", &a22r, &a22i) != 2)
        return EXIT_FAILURE;
      double
        u11r = -0.0, u11i = -0.0, u21r = -0.0, u21i = -0.0, u12r = -0.0, u12i = -0.0, u22r = -0.0, u22i = -0.0,
        v11r = -0.0, v11i = -0.0, v21r = -0.0, v21i = -0.0, v12r = -0.0, v12i = -0.0, v22r = -0.0, v22i = -0.0,
        s1 = -0.0, s2 = -0.0;
      int es[3] = { 0, 0, 0 };
      const int knd = PVN_FABI(pvn_zljsv2,PVN_ZLJSV2)(&a11r, &a11i, &a21r, &a21i, &a12r, &a12i, &a22r, &a22i,
                                  &u11r, &u11i, &u21r, &u21i, &u12r, &u12i, &u22r, &u22i,
                                  &v11r, &v11i, &v21r, &v21i, &v12r, &v12i, &v22r, &v22i,
                                  &s1, &s2, es);
      (void)printf("knd=%d, es={%d,%d,%d}\n", knd, es[0], es[1], es[2]);
      (void)printf("\tG =\n");
      (void)printf("(%s,", pvn_dtoa(s, a11r));
      (void)printf("%s) ", pvn_dtoa(s, a11i));
      (void)printf("(%s,", pvn_dtoa(s, a12r));
      (void)printf("%s)\n", pvn_dtoa(s, a12i));
      (void)printf("(%s,", pvn_dtoa(s, a21r));
      (void)printf("%s) ", pvn_dtoa(s, a21i));
      (void)printf("(%s,", pvn_dtoa(s, a22r));
      (void)printf("%s)\n", pvn_dtoa(s, a22i));
      (void)printf("\tU =\n");
      (void)printf("(%s,", pvn_dtoa(s, u11r));
      (void)printf("%s) ", pvn_dtoa(s, u11i));
      (void)printf("(%s,", pvn_dtoa(s, u12r));
      (void)printf("%s)\n", pvn_dtoa(s, u12i));
      (void)printf("(%s,", pvn_dtoa(s, u21r));
      (void)printf("%s) ", pvn_dtoa(s, u21i));
      (void)printf("(%s,", pvn_dtoa(s, u22r));
      (void)printf("%s)\n", pvn_dtoa(s, u22i));
      (void)printf("\tV =\n");
      (void)printf("(%s,", pvn_dtoa(s, v11r));
      (void)printf("%s) ", pvn_dtoa(s, v11i));
      (void)printf("(%s,", pvn_dtoa(s, v12r));
      (void)printf("%s)\n", pvn_dtoa(s, v12i));
      (void)printf("(%s,", pvn_dtoa(s, v21r));
      (void)printf("%s) ", pvn_dtoa(s, v21i));
      (void)printf("(%s,", pvn_dtoa(s, v22r));
      (void)printf("%s)\n", pvn_dtoa(s, v22i));
      (void)printf("\ts =\n");
      (void)printf("%s ", pvn_dtoa(s, s1));
      (void)printf("%s\n\tS =\n", pvn_dtoa(s, s2));
      PVN_FABI(pvn_zyljr2,PVN_ZYLJR2)(&a11r, &a11i, &a21r, &a21i, &a12r, &a12i, &a22r, &a22i,
                  &u11r, &u11i, &u21r, &u21i, &u12r, &u12i, &u22r, &u22i,
                  &v11r, &v11i, &v21r, &v21i, &v12r, &v12i, &v22r, &v22i,
                  &s1, &s2, es, E);
      s1 = scalbn(s1, (es[1] - es[0]));
      s2 = scalbn(s2, (es[2] - es[0]));
      (void)printf("%s ", pvn_dtoa(s, s1));
      (void)printf("%s\n", pvn_dtoa(s, s2));
    }
    else
      return EXIT_FAILURE;
#if (!defined(_WIN32) || defined(_DLL))
    (void)printf("cond_2(G) =%s\n", pvn_qtoa(s, E[0]));
    (void)printf("|| U^%c U - I ||_F =%s\n", T, pvn_qtoa(s, E[1]));
    (void)printf("|| V^%c V - I ||_F =%s\n", T, pvn_qtoa(s, E[2]));
    (void)printf("|| U Σ V^%c - G ||_F / || G ||_F =%s\n", T, pvn_qtoa(s, E[3]));
#endif /* !_WIN32 || _DLL */
  }
  else {
    int u = PVN_FABI(pvn_ran_open,PVN_RAN_OPEN)();
    if (u < 0) {
      (void)fprintf(stderr, "open(/dev/random): %d\n", u);
      return EXIT_FAILURE;
    }
#ifdef PVN_QUADMATH
    __float128 EC = 0.0q, EU = 0.0q, EV = 0.0q, EG = 0.0q;
#else /* !PVN_QUADMATH */
    long double EC = 0.0L, EU = 0.0L, EV = 0.0L, EG = 0.0L;
#endif /* ?PVN_QUADMATH */
    if (t == 'S') {
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(n,s,u,upper,T,pp) reduction(max:EC,EU,EV,EG)
#endif /* _OPENMP */
      for (int i = 0u; i < n; ++i) {
        const float a11 = PVN_FABI(pvn_ran_safe_f,PVN_RAN_SAFE_F)(&u, pp);
        if (!(a11 != 0.0f))
          PVN_STOP("G(1,1)");
        const float a21 = (upper ? 0.0f : PVN_FABI(pvn_ran_safe_f,PVN_RAN_SAFE_F)(&u, pp));
        if (!upper && !(a21 != 0.0f))
          PVN_STOP("G(2,1)");
        const float a12 = PVN_FABI(pvn_ran_safe_f,PVN_RAN_SAFE_F)(&u, pp);
        if (!(a12 != 0.0f))
          PVN_STOP("G(1,2)");
        const float a22 = PVN_FABI(pvn_ran_safe_f,PVN_RAN_SAFE_F)(&u, pp);
        if (!(a22 != 0.0f))
          PVN_STOP("G(2,2)");
        float u11 = -0.0f, u21 = -0.0f, u12 = -0.0f, u22 = -0.0f, v11 = -0.0f, v21 = -0.0f, v12 = -0.0f, v22 = -0.0f, s1 = -0.0f, s2 = -0.0f;
        int es[3] = { 0, 0, 0 };
        const int knd = PVN_FABI(pvn_sljsv2,PVN_SLJSV2)(&a11, &a21, &a12, &a22, &u11, &u21, &u12, &u22, &v11, &v21, &v12, &v22, &s1, &s2, es);
        if ((knd < 0) || ((knd != 13) && (knd != 15)))
          PVN_STOP("pvn_sljsv2");
#ifdef PVN_QUADMATH
        __float128 E[4] = { 0.0q, 0.0q, 0.0q, 0.0q };
#else /* !PVN_QUADMATH */
        long double E[4] = { 0.0L, 0.0L, 0.0L, 0.0L };
#endif /* ?PVN_QUADMATH */
        PVN_FABI(pvn_sqljr2,PVN_SQLJR2)(&a11, &a21, &a12, &a22, &u11, &u21, &u12, &u22, &v11, &v21, &v12, &v22, &s1, &s2, es, E);
#ifdef PVN_QUADMATH
        EC = fmaxq(EC, E[0]);
#else /* !PVN_QUADMATH */
        EC = fmaxl(EC, E[0]);
#endif /* ?PVN_QUADMATH */
#if (!defined(_WIN32) || defined(_DLL))
#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
        {
          (void)printf("%10d: cond_2(G) =%s\n", i, pvn_qtoa(s, E[0]));
        }
#endif /* !NDEBUG */
#endif /* !_WIN32 || _DLL */
#ifdef PVN_QUADMATH
        EU = fmaxq(EU, E[1]);
#else /* !PVN_QUADMATH */
        EU = fmaxl(EU, E[1]);
#endif /* ?PVN_QUADMATH */
#if (!defined(_WIN32) || defined(_DLL))
#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
        {
          (void)printf("%10d: || U^%c U - I ||_F =%s\n", i, T, pvn_qtoa(s, E[1]));
        }
#endif /* !NDEBUG */
#endif /* !_WIN32 || _DLL */
#ifdef PVN_QUADMATH
        EV = fmaxq(EV, E[2]);
#else /* !PVN_QUADMATH */
        EV = fmaxl(EV, E[2]);
#endif /* ?PVN_QUADMATH */
#if (!defined(_WIN32) || defined(_DLL))
#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
        {
          (void)printf("%10d: || V^%c V - I ||_F =%s\n", i, T, pvn_qtoa(s, E[2]));
        }
#endif /* !NDEBUG */
#endif /* !_WIN32 || _DLL */
#ifdef PVN_QUADMATH
        EG = fmaxq(EG, E[3]);
#else /* !PVN_QUADMATH */
        EG = fmaxl(EG, E[3]);
#endif /* ?PVN_QUADMATH */
#if (!defined(_WIN32) || defined(_DLL))
#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
        {
          (void)printf("%10d: || U Σ V^%c - G ||_F / || G ||_F =%s\n", i, T, pvn_qtoa(s, E[3]));
        }
#endif /* !NDEBUG */
#endif /* !_WIN32 || _DLL */
      }
    }
    else if (t == 'D') {
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(n,s,u,upper,T,pp) reduction(max:EC,EU,EV,EG)
#endif /* _OPENMP */
      for (int i = 0u; i < n; ++i) {
        const double a11 = PVN_FABI(pvn_ran_safe,PVN_RAN_SAFE)(&u, pp);
        if (!(a11 != 0.0))
          PVN_STOP("G(1,1)");
        const double a21 = (upper ? 0.0 : PVN_FABI(pvn_ran_safe,PVN_RAN_SAFE)(&u, pp));
        if (!upper && !(a21 != 0.0))
          PVN_STOP("G(2,1)");
        const double a12 = PVN_FABI(pvn_ran_safe,PVN_RAN_SAFE)(&u, pp);
        if (!(a12 != 0.0))
          PVN_STOP("G(1,2)");
        const double a22 = PVN_FABI(pvn_ran_safe,PVN_RAN_SAFE)(&u, pp);
        if (!(a22 != 0.0))
          PVN_STOP("G(2,2)");
        double u11 = -0.0, u21 = -0.0, u12 = -0.0, u22 = -0.0, v11 = -0.0, v21 = -0.0, v12 = -0.0, v22 = -0.0, s1 = -0.0, s2 = -0.0;
        int es[3] = { 0, 0, 0 };
        const int knd = PVN_FABI(pvn_dljsv2,PVN_DLJSV2)(&a11, &a21, &a12, &a22, &u11, &u21, &u12, &u22, &v11, &v21, &v12, &v22, &s1, &s2, es);
        if ((knd < 0) || ((knd != 13) && (knd != 15)))
          PVN_STOP("pvn_dljsv2");
#ifdef PVN_QUADMATH
        __float128 E[4] = { 0.0q, 0.0q, 0.0q, 0.0q };
#else /* !PVN_QUADMATH */
        long double E[4] = { 0.0L, 0.0L, 0.0L, 0.0L };
#endif /* ?PVN_QUADMATH */
        PVN_FABI(pvn_dqljr2,PVN_DQLJR2)(&a11, &a21, &a12, &a22, &u11, &u21, &u12, &u22, &v11, &v21, &v12, &v22, &s1, &s2, es, E);
#ifdef PVN_QUADMATH
        EC = fmaxq(EC, E[0]);
#else /* !PVN_QUADMATH */
        EC = fmaxl(EC, E[0]);
#endif /* ?PVN_QUADMATH */
#if (!defined(_WIN32) || defined(_DLL))
#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
        {
          (void)printf("%10d: cond_2(G) =%s\n", i, pvn_qtoa(s, E[0]));
        }
#endif /* !NDEBUG */
#endif /* !_WIN32 || _DLL */
#ifdef PVN_QUADMATH
        EU = fmaxq(EU, E[1]);
#else /* !PVN_QUADMATH */
        EU = fmaxl(EU, E[1]);
#endif /* ?PVN_QUADMATH */
#if (!defined(_WIN32) || defined(_DLL))
#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
        {
          (void)printf("%10d: || U^%c U - I ||_F =%s\n", i, T, pvn_qtoa(s, E[1]));
        }
#endif /* !NDEBUG */
#endif /* !_WIN32 || _DLL */
#ifdef PVN_QUADMATH
        EV = fmaxq(EV, E[2]);
#else /* !PVN_QUADMATH */
        EV = fmaxl(EV, E[2]);
#endif /* ?PVN_QUADMATH */
#if (!defined(_WIN32) || defined(_DLL))
#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
        {
          (void)printf("%10d: || V^%c V - I ||_F =%s\n", i, T, pvn_qtoa(s, E[2]));
        }
#endif /* !NDEBUG */
#endif /* !_WIN32 || _DLL */
#ifdef PVN_QUADMATH
        EG = fmaxq(EG, E[3]);
#else /* !PVN_QUADMATH */
        EG = fmaxl(EG, E[3]);
#endif /* ?PVN_QUADMATH */
#if (!defined(_WIN32) || defined(_DLL))
#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
        {
          (void)printf("%10d: || U Σ V^%c - G ||_F / || G ||_F =%s\n", i, T, pvn_qtoa(s, E[3]));
        }
#endif /* !NDEBUG */
#endif /* !_WIN32 || _DLL */
      }
    }
    else if (t == 'C') {
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(n,s,u,upper,T,pp) reduction(max:EC,EU,EV,EG)
#endif /* _OPENMP */
      for (int i = 0u; i < n; ++i) {
        const float a11r = PVN_FABI(pvn_ran_safe_f,PVN_RAN_SAFE_F)(&u, pp);
        if (!(a11r != 0.0f))
          PVN_STOP("G(1,1)");
        const float a11i = PVN_FABI(pvn_ran_safe_f,PVN_RAN_SAFE_F)(&u, pp);
        if (!(a11i != 0.0f))
          PVN_STOP("G(1,1)");
        const float a21r = (upper ? 0.0f : PVN_FABI(pvn_ran_safe_f,PVN_RAN_SAFE_F)(&u, pp));
        if (!upper && !(a21r != 0.0f))
          PVN_STOP("G(2,1)");
        const float a21i = (upper ? 0.0f : PVN_FABI(pvn_ran_safe_f,PVN_RAN_SAFE_F)(&u, pp));
        if (!upper && !(a21i != 0.0f))
          PVN_STOP("G(2,1)");
        const float a12r = PVN_FABI(pvn_ran_safe_f,PVN_RAN_SAFE_F)(&u, pp);
        if (!(a12r != 0.0f))
          PVN_STOP("G(1,2)");
        const float a12i = PVN_FABI(pvn_ran_safe_f,PVN_RAN_SAFE_F)(&u, pp);
        if (!(a12i != 0.0f))
          PVN_STOP("G(1,2)");
        const float a22r = PVN_FABI(pvn_ran_safe_f,PVN_RAN_SAFE_F)(&u, pp);
        if (!(a22r != 0.0f))
          PVN_STOP("G(2,2)");
        const float a22i = PVN_FABI(pvn_ran_safe_f,PVN_RAN_SAFE_F)(&u, pp);
        if (!(a22i != 0.0f))
          PVN_STOP("G(2,2)");
        float
          u11r = -0.0f, u11i = -0.0f, u21r = -0.0f, u21i = -0.0f, u12r = -0.0f, u12i = -0.0f, u22r = -0.0f, u22i = -0.0f,
          v11r = -0.0f, v11i = -0.0f, v21r = -0.0f, v21i = -0.0f, v12r = -0.0f, v12i = -0.0f, v22r = -0.0f, v22i = -0.0f,
          s1 = -0.0f, s2 = -0.0f;
        int es[3] = { 0, 0, 0 };
        const int knd = PVN_FABI(pvn_cljsv2,PVN_CLJSV2)(&a11r, &a11i, &a21r, &a21i, &a12r, &a12i, &a22r, &a22i,
                                    &u11r, &u11i, &u21r, &u21i, &u12r, &u12i, &u22r, &u22i,
                                    &v11r, &v11i, &v21r, &v21i, &v12r, &v12i, &v22r, &v22i,
                                    &s1, &s2, es);
        if ((knd < 0) || ((knd != 13) && (knd != 15)))
          PVN_STOP("pvn_cljsv2");
#ifdef PVN_QUADMATH
        __float128 E[4] = { 0.0q, 0.0q, 0.0q, 0.0q };
#else /* !PVN_QUADMATH */
        long double E[4] = { 0.0L, 0.0L, 0.0L, 0.0L };
#endif /* ?PVN_QUADMATH */
        PVN_FABI(pvn_cyljr2,PVN_CYLJR2)(&a11r, &a11i, &a21r, &a21i, &a12r, &a12i, &a22r, &a22i,
                    &u11r, &u11i, &u21r, &u21i, &u12r, &u12i, &u22r, &u22i,
                    &v11r, &v11i, &v21r, &v21i, &v12r, &v12i, &v22r, &v22i,
                    &s1, &s2, es, E);
#ifdef PVN_QUADMATH
        EC = fmaxq(EC, E[0]);
#else /* !PVN_QUADMATH */
        EC = fmaxl(EC, E[0]);
#endif /* ?PVN_QUADMATH */
#if (!defined(_WIN32) || defined(_DLL))
#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
        {
          (void)printf("%10d: cond_2(G) =%s\n", i, pvn_qtoa(s, E[0]));
        }
#endif /* !NDEBUG */
#endif /* !_WIN32 || _DLL */
#ifdef PVN_QUADMATH
        EU = fmaxq(EU, E[1]);
#else /* !PVN_QUADMATH */
        EU = fmaxl(EU, E[1]);
#endif /* ?PVN_QUADMATH */
#if (!defined(_WIN32) || defined(_DLL))
#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
        {
          (void)printf("%10d: || U^%c U - I ||_F =%s\n", i, T, pvn_qtoa(s, E[1]));
        }
#endif /* !NDEBUG */
#endif /* !_WIN32 || _DLL */
#ifdef PVN_QUADMATH
        EV = fmaxq(EV, E[2]);
#else /* !PVN_QUADMATH */
        EV = fmaxl(EV, E[2]);
#endif /* ?PVN_QUADMATH */
#if (!defined(_WIN32) || defined(_DLL))
#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
        {
          (void)printf("%10d: || V^%c V - I ||_F =%s\n", i, T, pvn_qtoa(s, E[2]));
        }
#endif /* !NDEBUG */
#endif /* !_WIN32 || _DLL */
#ifdef PVN_QUADMATH
        EG = fmaxq(EG, E[3]);
#else /* !PVN_QUADMATH */
        EG = fmaxl(EG, E[3]);
#endif /* ?PVN_QUADMATH */
#if (!defined(_WIN32) || defined(_DLL))
#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
        {
          (void)printf("%10d: || U Σ V^%c - G ||_F / || G ||_F =%s\n", i, T, pvn_qtoa(s, E[3]));
        }
#endif /* !NDEBUG */
#endif /* !_WIN32 || _DLL */
      }
    }
    else if (t == 'Z') {
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(n,s,u,upper,T,pp) reduction(max:EC,EU,EV,EG)
#endif /* _OPENMP */
      for (int i = 0u; i < n; ++i) {
        const double a11r = PVN_FABI(pvn_ran_safe,PVN_RAN_SAFE)(&u, pp);
        if (!(a11r != 0.0))
          PVN_STOP("G(1,1)");
        const double a11i = PVN_FABI(pvn_ran_safe,PVN_RAN_SAFE)(&u, pp);
        if (!(a11i != 0.0))
          PVN_STOP("G(1,1)");
        const double a21r = (upper ? 0.0 : PVN_FABI(pvn_ran_safe,PVN_RAN_SAFE)(&u, pp));
        if (!upper && !(a21r != 0.0))
          PVN_STOP("G(2,1)");
        const double a21i = (upper ? 0.0 : PVN_FABI(pvn_ran_safe,PVN_RAN_SAFE)(&u, pp));
        if (!upper && !(a21i != 0.0))
          PVN_STOP("G(2,1)");
        const double a12r = PVN_FABI(pvn_ran_safe,PVN_RAN_SAFE)(&u, pp);
        if (!(a12r != 0.0))
          PVN_STOP("G(1,2)");
        const double a12i = PVN_FABI(pvn_ran_safe,PVN_RAN_SAFE)(&u, pp);
        if (!(a12i != 0.0))
          PVN_STOP("G(1,2)");
        const double a22r = PVN_FABI(pvn_ran_safe,PVN_RAN_SAFE)(&u, pp);
        if (!(a22r != 0.0))
          PVN_STOP("G(2,2)");
        const double a22i = PVN_FABI(pvn_ran_safe,PVN_RAN_SAFE)(&u, pp);
        if (!(a22i != 0.0))
          PVN_STOP("G(2,2)");
        double
          u11r = -0.0, u11i = -0.0, u21r = -0.0, u21i = -0.0, u12r = -0.0, u12i = -0.0, u22r = -0.0, u22i = -0.0,
          v11r = -0.0, v11i = -0.0, v21r = -0.0, v21i = -0.0, v12r = -0.0, v12i = -0.0, v22r = -0.0, v22i = -0.0,
          s1 = -0.0, s2 = -0.0;
        int es[3] = { 0, 0, 0 };
        const int knd = PVN_FABI(pvn_zljsv2,PVN_ZLJSV2)(&a11r, &a11i, &a21r, &a21i, &a12r, &a12i, &a22r, &a22i,
                                    &u11r, &u11i, &u21r, &u21i, &u12r, &u12i, &u22r, &u22i,
                                    &v11r, &v11i, &v21r, &v21i, &v12r, &v12i, &v22r, &v22i,
                                    &s1, &s2, es);
        if ((knd < 0) || ((knd != 13) && (knd != 15)))
          PVN_STOP("pvn_zljsv2");
#ifdef PVN_QUADMATH
        __float128 E[4] = { 0.0q, 0.0q, 0.0q, 0.0q };
#else /* !PVN_QUADMATH */
        long double E[4] = { 0.0L, 0.0L, 0.0L, 0.0L };
#endif /* ?PVN_QUADMATH */
        PVN_FABI(pvn_zyljr2,PVN_ZYLJR2)(&a11r, &a11i, &a21r, &a21i, &a12r, &a12i, &a22r, &a22i,
                    &u11r, &u11i, &u21r, &u21i, &u12r, &u12i, &u22r, &u22i,
                    &v11r, &v11i, &v21r, &v21i, &v12r, &v12i, &v22r, &v22i,
                    &s1, &s2, es, E);
#ifdef PVN_QUADMATH
        EC = fmaxq(EC, E[0]);
#else /* !PVN_QUADMATH */
        EC = fmaxl(EC, E[0]);
#endif /* ?PVN_QUADMATH */
#if (!defined(_WIN32) || defined(_DLL))
#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
        {
          (void)printf("%10d: cond_2(G) =%s\n", i, pvn_qtoa(s, E[0]));
        }
#endif /* !NDEBUG */
#endif /* !_WIN32 || _DLL */
#ifdef PVN_QUADMATH
        EU = fmaxq(EU, E[1]);
#else /* !PVN_QUADMATH */
        EU = fmaxl(EU, E[1]);
#endif /* ?PVN_QUADMATH */
#if (!defined(_WIN32) || defined(_DLL))
#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
        {
          (void)printf("%10d: || U^%c U - I ||_F =%s\n", i, T, pvn_qtoa(s, E[1]));
        }
#endif /* !NDEBUG */
#endif /* !_WIN32 || _DLL */
#ifdef PVN_QUADMATH
        EV = fmaxq(EV, E[2]);
#else /* !PVN_QUADMATH */
        EV = fmaxl(EV, E[2]);
#endif /* ?PVN_QUADMATH */
#if (!defined(_WIN32) || defined(_DLL))
#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
        {
          (void)printf("%10d: || V^%c V - I ||_F =%s\n", i, T, pvn_qtoa(s, E[2]));
        }
#endif /* !NDEBUG */
#endif /* !_WIN32 || _DLL */
#ifdef PVN_QUADMATH
        EG = fmaxq(EG, E[3]);
#else /* !PVN_QUADMATH */
        EG = fmaxl(EG, E[3]);
#endif /* ?PVN_QUADMATH */
#if (!defined(_WIN32) || defined(_DLL))
#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp critical
#endif /* _OPENMP */
        {
          (void)printf("%10d: || U Σ V^%c - G ||_F / || G ||_F =%s\n", i, T, pvn_qtoa(s, E[3]));
        }
#endif /* !NDEBUG */
#endif /* !_WIN32 || _DLL */
      }
    }
    else
      return EXIT_FAILURE;
    u = PVN_FABI(pvn_ran_close,PVN_RAN_CLOSE)(&u);
#if (!defined(_WIN32) || defined(_DLL))
#ifndef NDEBUG
    (void)printf("max(cond_2(G))=");
#endif /* !NDEBUG */
    (void)printf("%s", pvn_qtoa(s, EC));
#ifndef NDEBUG
    (void)printf("\nmax(|| U^%c U - I ||_F)=", T);
#else /* NDEBUG */
    (void)putchar(',');
#endif /* ?NDEBUG */
    (void)printf("%s", pvn_qtoa(s, EU));
#ifndef NDEBUG
    (void)printf("\nmax(|| V^%c V - I ||_F)=", T);
#else /* NDEBUG */
    (void)putchar(',');
#endif /* ?NDEBUG */
    (void)printf("%s", pvn_qtoa(s, EV));
#ifndef NDEBUG
    (void)printf("\nmax(|| U Σ V^%c - G ||_F / || G ||_F)=", T);
#else /* NDEBUG */
    (void)putchar(',');
#endif /* ?NDEBUG */
    (void)printf("%s\n", pvn_qtoa(s, EG));
#endif /* !_WIN32 || _DLL */
    if (u != 0) {
      (void)fprintf(stderr, "close(/dev/random): %d\n", u);
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}
#else /* !PVN_TEST */
static inline void ef_mulf(int *const e, float *const f, const int e1, const float f1, const int e2, const float f2)
{
  PVN_ASSERT(e);
  PVN_ASSERT(f);
  *f = (f1 * f2);
  *f = frexpf(*f, e);
  *e += (e1 + e2);
}

static inline void ef_divf(int *const e, float *const f, const int e1, const float f1, const int e2, const float f2)
{
  PVN_ASSERT(e);
  PVN_ASSERT(f);
  *f = (f1 / f2);
  *f = frexpf(*f, e);
  *e += (e1 - e2);
}

/* f1, f2 >= 0 */
static inline int ef_cmpf(const int e1, const float f1, const int e2, const float f2)
{
  if (f2 == 0.0f)
    return (f1 != 0.0f);
  if (f1 == 0.0f)
    return -1;
  if (e1 < e2)
    return -1;
  if (e2 < e1)
    return 1;
  if (f1 < f2)
    return -1;
  if (f2 < f1)
    return 1;
  return 0;
}

/* TODO: the Kahan's algorithm for determinants of order two, with a power-of-two prescaling, might have been used instead */
static float sQR(float *const A11, float *const A21, float *const A12, float *const A22, float *const w1)
{
  PVN_ASSERT(A11);
  PVN_ASSERT(A21);
  PVN_ASSERT(A12);
  PVN_ASSERT(A22);
  PVN_ASSERT(w1);
#ifdef PVN_SV2_SAFE
  if (feclearexcept(FE_INEXACT))
    return __builtin_nanf("");
#endif /* PVN_SV2_SAFE */
  const float t = (*A21 / *A11);
#ifdef PVN_SV2_SAFE
  if (fetestexcept(FE_INEXACT) & FE_INEXACT) {
    const double d = (double)*A11;
    *A11 = *w1;
    *w1 = hypotf(t, 1.0f);
    if (copysignf(1.0f, *A12) == copysignf(1.0f, *A22)) {
      const float a12 = fmaf(*A22, t, *A12);
      const double
        a = ((double)*A22 * d),
        b = ((double)*A12 * (double)*A21),
        c = (a - b);
      *A22 = (float)(c / d);
      if (*w1 == 1.0f)
        *A12 = a12;
      else {
        *A12 = (a12 / *w1);
        *A22 = (*A22 / *w1);
      }
      *A21 = 1.0f;
    }
    else {
      const float a22 = fmaf(-*A12, t, *A22);
      const double
        a = ((double)*A12 * d),
        b = ((double)*A22 * (double)*A21),
        c = (a + b);
      *A12 = (float)(c / d);
      if (*w1 == 1.0f)
        *A22 = a22;
      else {
        *A12 = (*A12 / *w1);
        *A22 = (a22 / *w1);
      }
      *A21 = -1.0f;
    }
    return t;
  }
  else /* tan exact */
    *A21 = -0.0f;
#else /* !PVN_SV2_SAFE */
  *A21 = 0.0f;
#endif /* ?PVN_SV2_SAFE */
  *A11 = *w1;
  *w1 = hypotf(t, 1.0f);
  const float a12 = fmaf( *A22, t, *A12);
  const float a22 = fmaf(-*A12, t, *A22);
  if (*w1 == 1.0f) {
    *A12 = a12;
    *A22 = a22;
  }
  else {
    *A12 = (a12 / *w1);
    *A22 = (a22 / *w1);
  }
  return t;
}

static void slpsv2(const float A11, const float A12, const float A22, float *const tf, float *const cf, float *const sf, float *const tp, float *const cp, float *const sp, float *const s1, float *const s2, int *const es
#ifdef PVN_PRINTOUT
                   , char *const s
#endif /* PVN_PRINTOUT */
                   )
{
  PVN_ASSERT(tf);
  PVN_ASSERT(cf);
  PVN_ASSERT(sf);
  PVN_ASSERT(tp);
  PVN_ASSERT(cp);
  PVN_ASSERT(sp);
  PVN_ASSERT(s1);
  PVN_ASSERT(s2);
  PVN_ASSERT(es);
  int ae = 0, be = 0, abe = 0, a_be = 0, de = 0, ne = 0, t2e = 0;
  float af = 0.0f, bf = 0.0f, abf = 0.0f, a_bf = 0.0f, df = 0.0f, nf = 0.0f, t2f = 0.0f, t2 = 0.0f;

  /* tan(2φ) */
  if (A11 == A22)
    t2 = ((2.0f * A22) / A12);
  else if (A12 == A22) {
    /* (2 * A12 * A22) / (A11 * A11) */
    af = frexpf(A11, &ae);
    abf = frexpf(A12, &abe);
    bf = frexpf(A22, &be);
    ef_mulf(&ne, &nf, abe, abf, be, bf);
    ++ne;
    ef_mulf(&de, &df, ae, af, ae, af);
    ef_divf(&t2e, &t2f, ne, nf, de, df);
    t2 = scalbnf(t2f, t2e);
  }
  else if ((A11 / A12) < (FLT_EPSILON * 0.5f))
    t2 = ((2.0f * A22) / A12);
#ifdef PVN_SV2_SAFE
  else {
    /* should never overflow */
    const float a = hypotf(A11, A12);
    af = frexpf(a, &ae);
    const float b = A22;
    bf = frexpf(b, &be);
    abf = (a + b);
    if (!isfinite(abf)) {
      abf = ((0.5f * a) + (0.5f * b));
      de = 1;
    }
    abf = frexpf(abf, &abe);
    abe += de;
    a_bf = (a - b);
    if (a == b)
      de = 0;
    else if (a_bf >= FLT_MIN) {
      a_bf = frexpf(a_bf, &a_be);
      de = 1;
    }
    else {
      de = ((FLT_MIN_EXP + FLT_MANT_DIG) - pvn_imin(ae, be));
      a_bf = (scalbnf(af, (ae + de)) - scalbnf(bf, (be + de)));
      a_bf = frexpf(a_bf, &a_be);
      a_be -= de;
      de = -1;
    }
    if (de)
      ef_mulf(&de, &df, a_be, a_bf, abe, abf);
    af = frexpf(A12, &ae);
    ef_mulf(&ne, &nf, ae, af, be, bf);
    ++ne;
    ef_divf(&t2e, &t2f, ne, nf, de, df);
    if (isfinite(t2f))
      t2 = scalbnf(t2f, t2e);
    else
      t2 = t2f;
  }
#else /* !PVN_SV2_SAFE */
  else if (A11 > A12) {
    const float x = (A12 / A11);
    const float y = (A22 / A11);
    t2 = (((2.0f * x) * y) / fmaxf(fmaf((x - y), (x + y), 1.0f), 0.0f));
  }
  else if (A12 > A11) {
    const float x = (A11 / A12);
    const float y = (A22 / A12);
    t2 = ((2.0f * y) / fmaxf(fmaf((x - y), (x + y), 1.0f), 0.0f));
  }
  else {
    const float y = (A22 / A12);
    t2 = ((2.0f * y) / fmaxf(fmaf((1.0f - y), (1.0f + y), 1.0f), 0.0f));
  }
#endif /* ?PVN_SV2_SAFE */

  /* tan(φ) */
  if (isfinite(t2))
    *tf = (t2 / (1.0f + hypotf(t2, 1.0f)));
  else
    *tf = 1.0f;
#ifdef PVN_PRINTOUT
  if (s) (void)dprintf(PVN_PRINTOUT, "tan(φ)=%s, ", pvn_stoa(s, *tf));
#endif /* PVN_PRINTOUT */
  *cf = hypotf(*tf, 1.0f); /* sec(φ) */
  *sf = (*tf / *cf); /* sin(φ) */
  nf = frexpf(*cf, &ne);
  *cf = (1.0f / *cf); /* cos(φ) */

  *sp = fmaf(*tf, A22, A12);
  *tp = (*sp / A11); /* tan(ψ) */
#ifdef PVN_PRINTOUT
  if (s) (void)dprintf(PVN_PRINTOUT, "tan(ψ)=%s\n", pvn_stoa(s, *tp));
#endif /* PVN_PRINTOUT */

  af = frexpf(A11, &ae);
  bf = frexpf(A22, &be);

  if (isfinite(*tp)) {
    *cp = hypotf(*tp, 1.0f); /* sec(ψ) */
    *sp = (*tp / *cp); /* sin(ψ) */
    df = frexpf(*cp, &de);
    *cp = (1.0f / *cp); /* cos(ψ) */
    ef_divf(&t2e, &t2f, ne, nf, de, df);
    /* s2 = A22 * (sec(φ) / sec(ψ)) */
    ef_mulf(&abe, &abf, be, bf, t2e, t2f);
    /* s1 = A11 * (sec(ψ) / sec(φ)) */
    ef_divf(&a_be, &a_bf, ae, af, t2e, t2f);
  }
  else {
    /* s1 ≈ A12 + tan(φ) * A22 */
    a_bf = frexpf(*sp, &a_be);
    /* tan(ψ) so large that sec(ψ) ≈ tan(ψ) */
    ef_divf(&t2e, &t2f, ae, af, a_be, a_bf);
    *sp = 1.0f; /* sin(ψ) */
    *cp = scalbnf(t2f, t2e); /* cos(ψ) */
    /* s2 ≈ A22 * cos(ψ) */
    ef_mulf(&abe, &abf, be, bf, t2e, t2f);
  }

#ifdef PVN_PRINTOUT
  if (s) {
    (void)dprintf(PVN_PRINTOUT, "cos(φ)=%s, ", pvn_stoa(s, *cf));
    (void)dprintf(PVN_PRINTOUT, "sin(φ)=%s\n", pvn_stoa(s, *sf));
    (void)dprintf(PVN_PRINTOUT, "cos(ψ)=%s, ", pvn_stoa(s, *cf));
    (void)dprintf(PVN_PRINTOUT, "sin(ψ)=%s\n", pvn_stoa(s, *sf));
  }
#endif /* PVN_PRINTOUT */

  *s1 = a_bf;
  es[1] = a_be;
  *s2 = abf;
  es[2] = abe;
}

int PVN_FABI(pvn_sljsv2,PVN_SLJSV2)
(const float *const a11, const float *const a21, const float *const a12, const float *const a22,
 float *const u11, float *const u21, float *const u12, float *const u22,
 float *const v11, float *const v21, float *const v12, float *const v22,
 float *const s1, float *const s2, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a21);
  PVN_ASSERT(a12);
  PVN_ASSERT(a22);
  PVN_ASSERT(u11);
  PVN_ASSERT(u21);
  PVN_ASSERT(u12);
  PVN_ASSERT(u22);
  PVN_ASSERT(v11);
  PVN_ASSERT(v21);
  PVN_ASSERT(v12);
  PVN_ASSERT(v22);
  PVN_ASSERT(s1);
  PVN_ASSERT(s2);
  PVN_ASSERT(es);
  if (!isfinite(*a11))
    return -1;
  if (!isfinite(*a21))
    return -2;
  if (!isfinite(*a12))
    return -3;
  if (!isfinite(*a22))
    return -4;
  /* should not fail beyond this point when *es == 0 */
  int knd = 0, mxe = INT_MIN, e = INT_MIN;
  if (*a11 != 0.0f) {
    knd |= 1;
    (void)frexpf(*a11, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a21 != 0.0f) {
    knd |= 2;
    (void)frexpf(*a21, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a12 != 0.0f) {
    knd |= 4;
    (void)frexpf(*a12, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a22 != 0.0f) {
    knd |= 8;
    (void)frexpf(*a22, &e);
    if (mxe < e)
      mxe = e;
  }
  switch (knd) {
  case  0:
  case  1:
  case  2:
  case  4:
  case  6:
  case  8:
  case  9:
    e = 0;
    break;
  case  3:
  case  5:
  case  7:
  case 10:
  case 11:
  case 12:
  case 13:
  case 14:
    e = (FLT_MAX_EXP - mxe - 1);
    break;
  case 15:
    e = (FLT_MAX_EXP - mxe - 2);
    break;
  default:
    return INT_MIN;
  }
  mxe = *es;
  if (!*es)
    *es = e;
  else if (*es < 0)
    ++*es;
  es[2] = es[1] = 0;

  /* scaling by 2^(*es), can only fail when mxe != 0 */
  float A11 = *a11, A21 = *a21, A12 = *a12, A22 = *a22;
  if (*es) {
    A11 = scalbnf(A11, *es);
    A21 = scalbnf(A21, *es);
    A12 = scalbnf(A12, *es);
    A22 = scalbnf(A22, *es);
    if (mxe) {
      if (!isfinite(A11))
        return -5;
      if (!isfinite(A21))
        return -6;
      if (!isfinite(A12))
        return -7;
      if (!isfinite(A22))
        return -8;
    }
    if (*es < 0) {
      knd = 0;
      if (A11 != 0.0f)
        knd |= 1;
      if (A21 != 0.0f)
        knd |= 2;
      if (A12 != 0.0f)
        knd |= 4;
      if (A22 != 0.0f)
        knd |= 8;
    }
  }

  *u11 = 1.0f;
  *u21 = 0.0f;
  *u12 = 0.0f;
  *u22 = 1.0f;
  *v11 = 1.0f;
  *v21 = 0.0f;
  *v12 = 0.0f;
  *v22 = 1.0f;
  *s1 = 0.0f;
  *s2 = 0.0f;

  /* simplify the form of A */
  switch (knd) {
  case  0:
    /* [ 0 0 ] */
    /* [ 0 0 ] */
    *u11 = copysignf(1.0f, A11);
    *u22 = copysignf(1.0f, A22);
    A22 = A12 = A21 = A11 = 0.0f;
    break;
  case  1:
    /* [ * 0 ] */
    /* [ 0 0 ] */
    if (A11 < 0.0f) {
      *u11 = -1.0f;
      A11 = -A11;
    }
    *u22 = copysignf(1.0f, A22);
    A22 = A12 = A21 = 0.0f;
    *s1 = A11;
    break;
  case  2:
    /* [ 0 0 ] */
    /* [ * 0 ] */
    *u11 = 0.0f;
    *u22 = 0.0f;
    A11 = A21;
    A22 = A12;
    if (A11 < 0.0f) {
      *u12 = -1.0f;
      A11 = -A11;
    }
    else
      *u12 = 1.0f;
    *u21 = copysignf(1.0f, A12);
    A22 = A12 = A21 = 0.0f;
    *s1 = A11;
    break;
  case  3:
    /* [ * 0 ] */
    /* [ * 0 ] */
    A12 = fabsf(A11);
    A22 = fabsf(A21);
    if (A12 < A22) {
      *u11 = 0.0f;
      *u21 = copysignf(1.0f, A11);
      *u12 = copysignf(1.0f, A21);
      *u22 = 0.0f;
      A11 = A22;
      A21 = A12;
    }
    else {
      *u11 = copysignf(1.0f, A11);
      *u22 = copysignf(1.0f, A21);
      A11 = A12;
      A21 = A22;
    }
    A22 = A12 = 0.0f;
    *s1 = hypotf(A11, A21);
    e = 3;
    break;
  case  4:
    /* [ 0 * ] */
    /* [ 0 0 ] */
    *u11 = copysignf(1.0f, A12);
    *u22 = copysignf(1.0f, A21);
    A11 = fabsf(A12);
    A22 = A12 = A21 = 0.0f;
    *v11 = 0.0f;
    *v21 = 1.0f;
    *v12 = 1.0f;
    *v22 = 0.0f;
    *s1 = A11;
    break;
  case  5:
    /* [ * * ] */
    /* [ 0 0 ] */
    A21 = fabsf(A11);
    A22 = fabsf(A12);
    if (A21 < A22) {
      *v11 = 0.0f;
      *v21 = copysignf(1.0f, A12);
      *v12 = copysignf(1.0f, A11);
      *v22 = 0.0f;
      A11 = A22;
      A12 = A21;
    }
    else {
      *v11 = copysignf(1.0f, A11);
      *v22 = copysignf(1.0f, A12);
      A11 = A21;
      A12 = A22;
    }
    A22 = A21 = 0.0f;
    *s1 = hypotf(A11, A12);
    e = 5;
    break;
  case  6:
    /* [ 0 * ] */
    /* [ * 0 ] */
    *u11 = copysignf(1.0f, A12);
    *u22 = copysignf(1.0f, A21);
    A11 = fabsf(A12);
    A22 = fabsf(A21);
    A12 = A21 = 0.0f;
    *v11 = 0.0f;
    *v21 = 1.0f;
    *v12 = 1.0f;
    *v22 = 0.0f;
    *s1 = A11;
    *s2 = A22;
    break;
  case  7:
    /* [ * * ] */
    /* [ * 0 ] */
    A22 = A11;
    A11 = A12;
    A12 = A22;
    A22 = A21;
    A21 = 0.0f;
    *v11 = 0.0f;
    *v22 = 0.0f;
    if (A11 < 0.0f) {
      A11 = -A11;
      *v21 = -1.0f;
    }
    else
      *v21 = 1.0f;
    if (A12 < 0.0f) {
      A12 = -A12;
      A22 = -A22;
      *v12 = -1.0f;
    }
    else
      *v12 = 1.0f;
    if (A22 < 0.0f) {
      *u22 = -1.0f;
      A22 = -A22;
    }
    e = 13;
    break;
  case  8:
    /* [ 0 0 ] */
    /* [ 0 * ] */
    *u11 = 0.0f;
    *u21 = copysignf(1.0f, A11);
    *u12 = copysignf(1.0f, A22);
    *u22 = 0.0f;
    A11 = fabsf(A22);
    A22 = A12 = A21 = 0.0f;
    *v11 = 0.0f;
    *v21 = 1.0f;
    *v12 = 1.0f;
    *v22 = 0.0f;
    *s1 = A11;
    break;
  case  9:
    /* [ * 0 ] */
    /* [ 0 * ] */
    if (A11 < 0.0f) {
      *u11 = -1.0f;
      A11 = -A11;
    }
    if (A22 < 0.0f) {
      *u22 = -1.0f;
      A22 = -A22;
    }
    A12 = A21 = 0.0f;
    *s1 = A11;
    *s2 = A22;
    break;
  case 10:
    /* [ 0 0 ] */
    /* [ * * ] */
    *u11 = 0.0f;
    *u21 = 1.0f;
    *u12 = 1.0f;
    *u22 = 0.0f;
    A11 = A21;
    A12 = A22;
    A21 = fabsf(A11);
    A22 = fabsf(A12);
    if (A21 < A22) {
      *v11 = 0.0f;
      *v21 = copysignf(1.0f, A12);
      *v12 = copysignf(1.0f, A11);
      *v22 = 0.0f;
      A11 = A22;
      A12 = A21;
    }
    else {
      *v11 = copysignf(1.0f, A11);
      *v22 = copysignf(1.0f, A12);
      A11 = A21;
      A12 = A22;
    }
    A22 = A21 = 0.0f;
    *s1 = hypotf(A11, A12);
    e = 5;
    break;
  case 11:
    /* [ * 0 ] */
    /* [ * * ] */
    *u11 = 0.0f;
    *u12 = 1.0f;
    *u22 = 0.0f;
    A12 = A11;
    A11 = A22;
    A22 = A12;
    A12 = A21;
    A21 = 0.0f;
    *v11 = 0.0f;
    *v22 = 0.0f;
    if (A11 < 0.0f) {
      A11 = -A11;
      *v21 = -1.0f;
    }
    else
      *v21 = 1.0f;
    if (A12 < 0.0f) {
      A12 = -A12;
      A22 = -A22;
      *v12 = -1.0f;
    }
    else
      *v12 = 1.0f;
    if (A22 < 0.0f) {
      *u21 = -1.0f;
      A22 = -A22;
    }
    else
      *u21 = 1.0f;
    e = 13;
    break;
  case 12:
    /* [ 0 * ] */
    /* [ 0 * ] */
    A11 = A12;
    A21 = A22;
    *v11 = 0.0f;
    *v21 = 1.0f;
    *v12 = 1.0f;
    *v22 = 0.0f;
    A12 = fabsf(A11);
    A22 = fabsf(A21);
    if (A12 < A22) {
      *u11 = 0.0f;
      *u21 = copysignf(1.0f, A11);
      *u12 = copysignf(1.0f, A21);
      *u22 = 0.0f;
      A11 = A22;
      A21 = A12;
    }
    else {
      *u11 = copysignf(1.0f, A11);
      *u22 = copysignf(1.0f, A21);
      A11 = A12;
      A21 = A22;
    }
    A22 = A12 = 0.0f;
    *s1 = hypotf(A11, A21);
    e = 3;
    break;
  case 13:
    /* [ * * ] */
    /* [ 0 * ] */
    if (A11 < 0.0f) {
      A11 = -A11;
      *v11 = -1.0f;
    }
    if (A12 < 0.0f) {
      A12 = -A12;
      A22 = -A22;
      *v22 = -1.0f;
    }
    if (A22 < 0.0f) {
      *u22 = -1.0f;
      A22 = -A22;
    }
    A21 = 0.0f;
    e = 13;
    break;
  case 14:
    /* [ 0 * ] */
    /* [ * * ] */
    *u11 = 0.0f;
    *u12 = 1.0f;
    *u22 = 0.0f;
    A11 = A12;
    A12 = A22;
    A22 = A11;
    A11 = A21;
    A21 = 0.0f;
    if (A11 < 0.0f) {
      A11 = -A11;
      *v11 = -1.0f;
    }
    if (A12 < 0.0f) {
      A12 = -A12;
      A22 = -A22;
      *v22 = -1.0f;
    }
    if (A22 < 0.0f) {
      *u21 = -1.0f;
      A22 = -A22;
    }
    else
      *u21 = 1.0f;
    e = 13;
    break;
  case 15:
    /* [ * * ] */
    /* [ * * ] */
    e = 15;
    break;
  default:
    return INT_MIN;
  }

  if ((e == 13) && (A11 < A22))
    e = -13;

#ifdef PVN_PRINTOUT
  char s[17] = { '\0' };
  (void)dprintf(PVN_PRINTOUT, "\tA[knd=%d,e=%d] * 2^%d =\n", knd, e, *es);
  (void)dprintf(PVN_PRINTOUT, "%s ", pvn_stoa(s, A11));
  (void)dprintf(PVN_PRINTOUT, "%s\n", pvn_stoa(s, A12));
  (void)dprintf(PVN_PRINTOUT, "%s ", pvn_stoa(s, A21));
  (void)dprintf(PVN_PRINTOUT, "%s\n", pvn_stoa(s, A22));
#endif /* PVN_PRINTOUT */

  float tt = 0.0f, ct = 1.0f, st = 0.0f;

  if (e == 15) {
    /* [ * * ] */
    /* [ * * ] */
    *s1 = hypotf(A11, A21);
    *s2 = hypotf(A12, A22);
    if (*s1 < *s2) {
      pvn_fswp(&A11, &A12);
      pvn_fswp(&A21, &A22);
      pvn_fswp(v11, v12);
      pvn_fswp(v21, v22);
      pvn_fswp(s1, s2);
    }
    if (copysignf(1.0f, A11) != 1.0f) {
      *u11 = -*u11;
      *u12 = -*u12;
      A11 = -A11;
      A12 = -A12;
    }
    if (copysignf(1.0f, A21) != 1.0f) {
      *u21 = -*u21;
      *u22 = -*u22;
      A21 = -A21;
      A22 = -A22;
    }
    if (A11 < A21) {
      pvn_fswp(u11, u21);
      pvn_fswp(u12, u22);
      pvn_fswp(&A11, &A21);
      pvn_fswp(&A12, &A22);
    }
    /* U^T(ϑ):
        cos(ϑ)  sin(ϑ)
       -sin(ϑ)  cos(ϑ)
    */
    ct = *s1;
    /* apply the left Givens rotation to A (and maybe to U) */
    tt = sQR(&A11, &A21, &A12, &A22, &ct);
    st = -tt;
    A21 = A12;
    if (ct == 1.0f) {
      if ((A12 == 0.0f) || (A22 == 0.0f)) {
        A21 = *u11;
        *u11 = fmaf(tt, *u21, *u11);
        *u21 = fmaf(st,  A21, *u21);
        A21 = *u12;
        *u12 = fmaf(tt, *u22, *u12);
        *u22 = fmaf(st,  A21, *u22);
      }
      st = tt;
    }
    else {
      if ((A12 == 0.0f) || (A22 == 0.0f)) {
        A21 = *u11;
        *u11 = (fmaf(tt, *u21, *u11) / ct);
        *u21 = (fmaf(st,  A21, *u21) / ct);
        A21 = *u12;
        *u12 = (fmaf(tt, *u22, *u12) / ct);
        *u22 = (fmaf(st,  A21, *u22) / ct);
      }
      st = (tt / ct);
      ct = (1.0f / ct);
    }
    A21 = 0.0f;
    if (A12 == 0.0f) {
      A12 = 0.0f;
      if (copysignf(1.0f, A22) != 1.0f) {
        *u21 = -*u21;
        *u22 = -*u22;
        A22 = -A22;
      }
      e = 0;
    }
    else if (A22 == 0.0f) {
      if (copysignf(1.0f, A12) != 1.0f) {
        A12 = -A12;
        A22 = -A22;
        *v12 = -*v12;
        *v22 = -*v22;
      }
      if (copysignf(1.0f, A22) != 1.0f) {
        *u21 = -*u21;
        *u22 = -*u22;
        A22 = 0.0f;
      }
      *s1 = hypotf(*s1, *s2);
      *s2 = 0.0f;
      e = 5;
    }
    else
      e = 13;
    if (A12 < 0.0f) {
      A12 = -A12;
      A22 = -A22;
      *v12 = -*v12;
      *v22 = -*v22;
    }
    if (A22 < 0.0f) {
      A22 = -A22;
      /* sin(ϑ) is always non-negative by construction */
      /* this is just an extra bit of info, used later */
      st = -st;
    }
    if ((e == 13) && (A11 < A22))
      e = -13;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(ϑ)=%s, ", pvn_stoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(ϑ)=%s, ", pvn_stoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(ϑ)=%s\n", pvn_stoa(s, st));
#endif /* PVN_PRINTOUT */
  }

  if (e == 3) {
    /* [ X 0 ] */
    /* [ x 0 ] */
    /* U^T(θ):
        cos(θ)  sin(θ)
       -sin(θ)  cos(θ)
    */
    tt = (A21 / A11);
    /* 1 / cos */
    ct = hypotf(tt, 1.0f);
    /* apply the left Givens rotation to U */
    st = -tt;
    A21 = *u11;
    if (ct == 1.0f) {
      *u11 = fmaf(tt, *u21, *u11);
      *u21 = fmaf(st,  A21, *u21);
      A21 = *u12;
      *u12 = fmaf(tt, *u22, *u12);
      *u22 = fmaf(st,  A21, *u22);
      st = tt;
    }
    else {
      *u11 = (fmaf(tt, *u21, *u11) / ct);
      *u21 = (fmaf(st,  A21, *u21) / ct);
      A21 = *u12;
      *u12 = (fmaf(tt, *u22, *u12) / ct);
      *u22 = (fmaf(st,  A21, *u22) / ct);
      st = (tt / ct);
      ct = (1.0f / ct);
    }
    A11 = *s1;
    A21 = 0.0f;
    e = 0;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(ϑ)=%s, ", pvn_stoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(ϑ)=%s, ", pvn_stoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(ϑ)=%s\n", pvn_stoa(s, st));
#endif /* PVN_PRINTOUT */
  }

  if (e == 5) {
    /* [ X x ] */
    /* [ 0 0 ] */
    /* V(θ):
       cos(θ) -sin(θ)
       sin(θ)  cos(θ)
    */
    tt = (A12 / A11);
    /* 1 / cos */
    ct = hypotf(tt, 1.0f);
    /* apply the right Givens rotation to V */
    st = -tt;
    A12 = *v11;
    if (ct == 1.0f) {
      *v11 = fmaf(tt, *v12, *v11);
      *v12 = fmaf(st,  A12, *v12);
      A12 = *v21;
      *v21 = fmaf(tt, *v22, *v21);
      *v22 = fmaf(st,  A12, *v22);
      st = tt;
    }
    else {
      *v11 = (fmaf(tt, *v12, *v11) / ct);
      *v12 = (fmaf(st,  A12, *v12) / ct);
      A12 = *v21;
      *v21 = (fmaf(tt, *v22, *v21) / ct);
      *v22 = (fmaf(st,  A12, *v22) / ct);
      st = (tt / ct);
      ct = (1.0f / ct);
    }
    A11 = *s1;
    A12 = 0.0f;
    e = 0;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(θ)=%s, ", pvn_stoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(θ)=%s, ", pvn_stoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(θ)=%s\n", pvn_stoa(s, st));
#endif /* PVN_PRINTOUT */
  }

#ifdef PVN_PRINTOUT
  (void)dprintf(PVN_PRINTOUT, "\tA =\n");
  (void)dprintf(PVN_PRINTOUT, "%s ", pvn_stoa(s, scalbnf(A11, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s\n", pvn_stoa(s, scalbnf(A12, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s ", pvn_stoa(s, scalbnf(A21, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s\n", pvn_stoa(s, scalbnf(A22, -*es)));
#endif /* PVN_PRINTOUT */

  if (abs(e) == 13) {
    /* [ x y ] */
    /* [ 0 z ] */

    float tf = 0.0f, cf = 1.0f, sf = 0.0f, tp = 0.0f, cp = 1.0f, sp = 0.0f;
    if (e == -13) {
      float tf_ = 0.0f, cf_ = 1.0f, sf_ = 0.0f, tp_ = 0.0f, cp_ = 1.0f, sp_ = 0.0f;
      slpsv2(A22, A12, A11, &tf_, &cf_, &sf_, &tp_, &cp_, &sp_, s1, s2, es
#ifdef PVN_PRINTOUT
             , s
#endif /* PVN_PRINTOUT */
             );
      tf = (1.0f / tp_);
      cf = sp_;
      sf = cp_;
      tp = (1.0f / tf_);
      cp = sf_;
      sp = cf_;
    }
    else
      slpsv2(A11, A12, A22, &tf, &cf, &sf, &tp, &cp, &sp, s1, s2, es
#ifdef PVN_PRINTOUT
             , s
#endif /* PVN_PRINTOUT */
             );

    /* update U */
    if (copysignf(1.0f, st) != 1.0f) {
      /* U^T(φ) * diag(1, -1) * U^T(ϑ):
          cos(φ - ϑ) -sin(φ - ϑ)
         -sin(φ - ϑ) -cos(φ - ϑ)
      */
      st = -st;
      float tf_t = (tf - tt), cf_t = 1.0f, sf_t = 0.0f;
      if (tf_t != 0.0f) {
        tf_t /= fmaf(tf, tt, 1.0f);
        if (isfinite(tf_t)) {
          /* 1 / cos */
          cf_t = hypotf(tf_t, 1.0f);
          sf_t = (tf_t / cf_t);
          cf_t = (1.0f / cf_t);
        }
        else {
          sf_t = copysignf(1.0f, tf_t);
          cf_t = 0.0f;
        }
      }
      else
        sf_t = tf_t;
      const float _sf_t = -sf_t;
      A21 = *u11;
      *u11 = (_sf_t * *u21 + cf_t * *u11);
      *u21 = (_sf_t *  A21 - cf_t * *u21);
      A21 = *u12;
      *u12 = (_sf_t * *u22 + cf_t * *u12);
      *u22 = (_sf_t *  A21 - cf_t * *u22);
      A21 = -1.0f;
    }
    else if (tt != 0.0f) {
      /* U^T(φ) * U^T(ϑ) = U^T(φ + ϑ):
          cos(φ + ϑ)  sin(φ + ϑ)
         -sin(φ + ϑ)  cos(φ + ϑ)
      */
      float tft = (tf + tt), cft = 1.0f, sft = 0.0f;
      if (tft != 0.0f) {
        tft /= fmaf(-tf, tt, 1.0f);
        if (isfinite(tft)) {
          /* 1 / cos */
          cft = hypotf(tft, 1.0f);
          sft = (tft / cft);
          cft = (1.0f / cft);
        }
        else {
          sft = copysignf(1.0f, tft);
          cft = 0.0f;
        }
      }
      else
        sft = tft;
      A21 = *u11;
      *u11 = (cft * *u11 + sft * *u21);
      *u21 = (cft * *u21 - sft *  A21);
      A21 = *u12;
      *u12 = (cft * *u12 + sft * *u22);
      *u22 = (cft * *u22 - sft *  A21);
      A21 = 1.0f;
    }
    else if (tf != 0.0f) {
      /* U^T(φ):
          cos(φ)  sin(φ)
         -sin(φ)  cos(φ)
       */
      A21 = *u11;
      *u11 = (cf * *u11 + sf * *u21);
      *u21 = (cf * *u21 - sf *  A21);
      A21 = *u12;
      *u12 = (cf * *u12 + sf * *u22);
      *u22 = (cf * *u22 - sf *  A21);
      A21 = -0.0f;
    }
    else /* U^T(φ) = I */
      A21 = 0.0f;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "U operation=%s\n", pvn_stoa(s, A21));
#endif /* PVN_PRINTOUT */

    /* update V */
    if (tp != 0.0f) {
      /* V(ψ):
         cos(ψ) -sin(ψ)
         sin(ψ)  cos(ψ)
      */
      A21 = *v11;
      *v11 = (*v11 * cp + *v12 * sp);
      *v12 = (*v12 * cp -  A21 * sp);
      A21 = *v21;
      *v21 = (*v21 * cp + *v22 * sp);
      *v22 = (*v22 * cp -  A21 * sp);
      A21 = -0.0f;
    }
    else /* V(ψ) = I */
      A21 = 0.0f;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "V operation=%s\n", pvn_stoa(s, A21));
#endif /* PVN_PRINTOUT */

    if (e == -13) {
      *u21 = -*u21;
      *u22 = -*u22;
      *v12 = -*v12;
      *v22 = -*v22;
    }
  }

  if (ef_cmpf(es[1], *s1, es[2], *s2) < 0) {
    pvn_fswp(u11, u21);
    pvn_fswp(u12, u22);
    pvn_fswp(v11, v12);
    pvn_fswp(v21, v22);
    pvn_fswp(s1, s2);
  }
  pvn_fswp(u21, u12);
  return knd;
}

static float cpolarf(const float r, const float i, float *const c, float *const s)
{
  PVN_ASSERT(c);
  PVN_ASSERT(s);
  const float a = hypotf(r, i);
  *c = copysignf(fminf((fabsf(r) / a), 1.0f), r);
  *s = (i / fmaxf(a, FLT_TRUE_MIN));
  return a;
}

int PVN_FABI(pvn_cljsv2,PVN_CLJSV2)
(const float *const a11r, const float *const a11i, const float *const a21r, const float *const a21i, const float *const a12r, const float *const a12i, const float *const a22r, const float *const a22i,
 float *const u11r, float *const u11i, float *const u21r, float *const u21i, float *const u12r, float *const u12i, float *const u22r, float *const u22i,
 float *const v11r, float *const v11i, float *const v21r, float *const v21i, float *const v12r, float *const v12i, float *const v22r, float *const v22i,
 float *const s1, float *const s2, int *const es)
{
  PVN_ASSERT(a11r); PVN_ASSERT(a11i);
  PVN_ASSERT(a21r); PVN_ASSERT(a21i);
  PVN_ASSERT(a12r); PVN_ASSERT(a12i);
  PVN_ASSERT(a22r); PVN_ASSERT(a22i);
  PVN_ASSERT(u11r); PVN_ASSERT(u11i);
  PVN_ASSERT(u21r); PVN_ASSERT(u21i);
  PVN_ASSERT(u12r); PVN_ASSERT(u12i);
  PVN_ASSERT(u22r); PVN_ASSERT(u22i);
  PVN_ASSERT(v11r); PVN_ASSERT(v11i);
  PVN_ASSERT(v21r); PVN_ASSERT(v21i);
  PVN_ASSERT(v12r); PVN_ASSERT(v12i);
  PVN_ASSERT(v22r); PVN_ASSERT(v22i);
  PVN_ASSERT(s1);
  PVN_ASSERT(s2);
  PVN_ASSERT(es);
  if (!isfinite(*a11r))
    return -1;
  if (!isfinite(*a11i))
    return -2;
  if (!isfinite(*a21r))
    return -3;
  if (!isfinite(*a21i))
    return -4;
  if (!isfinite(*a12r))
    return -5;
  if (!isfinite(*a12i))
    return -6;
  if (!isfinite(*a22r))
    return -7;
  if (!isfinite(*a22i))
    return -8;
  /* should not fail beyond this point when *es == 0 */
  int kndi = 0, mxe = INT_MIN, e = 0;
  if (*a11i != 0.0f) {
    kndi |= 1;
    (void)frexpf(*a11i, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a21i != 0.0f) {
    kndi |= 2;
    (void)frexpf(*a21i, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a12i != 0.0f) {
    kndi |= 4;
    (void)frexpf(*a12i, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a22i != 0.0f) {
    kndi |= 8;
    (void)frexpf(*a22i, &e);
    if (mxe < e)
      mxe = e;
  }
  *u11i = 0.0f;
  *u21i = 0.0f;
  *u12i = 0.0f;
  *u22i = 0.0f;
  *v11i = 0.0f;
  *v21i = 0.0f;
  *v12i = 0.0f;
  *v22i = 0.0f;
  if (!kndi)
    return PVN_FABI(pvn_sljsv2,PVN_SLJSV2)(a11r, a21r, a12r, a22r, u11r, u21r, u12r, u22r, v11r, v21r, v12r, v22r, s1, s2, es);
  int kndr = 0;
  if (*a11r != 0.0f) {
    kndr |= 1;
    (void)frexpf(*a11r, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a21r != 0.0f) {
    kndr |= 2;
    (void)frexpf(*a21r, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a12r != 0.0f) {
    kndr |= 4;
    (void)frexpf(*a12r, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a22r != 0.0f) {
    kndr |= 8;
    (void)frexpf(*a22r, &e);
    if (mxe < e)
      mxe = e;
  }
  if (!kndr) {
    *u11r = 0.0f;
    *u21r = 0.0f;
    *u12r = 0.0f;
    *u22r = 0.0f;
    return PVN_FABI(pvn_sljsv2,PVN_SLJSV2)(a11i, a21i, a12i, a22i, u11i, u21i, u12i, u22i, v11r, v21r, v12r, v22r, s1, s2, es);
  }

  int knd = (kndr | kndi);
  switch (knd) {
  case  1:
  case  2:
  case  4:
  case  6:
  case  8:
  case  9:
    e = (FLT_MAX_EXP - mxe - 1);
    break;
  case  3:
  case  5:
  case  7:
  case 10:
  case 11:
  case 12:
  case 13:
  case 14:
  case 15:
    e = (FLT_MAX_EXP - mxe - 2);
    break;
  default:
    return INT_MIN;
  }
  mxe = *es;
  if (!*es)
    *es = e;
  else if (*es < 0)
    ++*es;
  es[2] = es[1] = 0;

  /* scaling by 2^(*es), can only fail when mxe != 0 */
  float A11r = *a11r, A11i = *a11i, A21r = *a21r, A21i = *a21i, A12r = *a12r, A12i = *a12i, A22r = *a22r, A22i = *a22i;
  if (*es) {
    A11r = scalbnf(A11r, *es);
    A11i = scalbnf(A11i, *es);
    A21r = scalbnf(A21r, *es);
    A21i = scalbnf(A21i, *es);
    A12r = scalbnf(A12r, *es);
    A12i = scalbnf(A12i, *es);
    A22r = scalbnf(A22r, *es);
    A22i = scalbnf(A22i, *es);
    if (mxe) {
      if (!isfinite(A11r))
        return -9;
      if (!isfinite(A11i))
        return -10;
      if (!isfinite(A21r))
        return -11;
      if (!isfinite(A21i))
        return -12;
      if (!isfinite(A12r))
        return -13;
      if (!isfinite(A12i))
        return -14;
      if (!isfinite(A22r))
        return -15;
      if (!isfinite(A22i))
        return -16;
    }
    if (*es < 0) {
      kndr = 0;
      if (A11r != 0.0f)
        kndr |= 1;
      if (A21r != 0.0f)
        kndr |= 2;
      if (A12r != 0.0f)
        kndr |= 4;
      if (A22r != 0.0f)
        kndr |= 8;
      kndi = 0;
      if (A11i != 0.0f)
        kndi |= 1;
      if (A21i != 0.0f)
        kndi |= 2;
      if (A12i != 0.0f)
        kndi |= 4;
      if (A22i != 0.0f)
        kndi |= 8;
      knd = (kndr | kndi);
    }
  }

  *u11r = 1.0f;
  *u21r = 0.0f;
  *u12r = 0.0f;
  *u22r = 1.0f;
  *v11r = 1.0f;
  *v21r = 0.0f;
  *v12r = 0.0f;
  *v22r = 1.0f;
  *s1 = 0.0f;
  *s2 = 0.0f;

  /* simplify the form of A */
  switch (knd) {
  case  0:
  case  1:
  case  8:
  case  9:
    /* [ ? 0 ] */
    /* [ 0 ? ] */
    *s1 = cpolarf(A11r, A11i, u11r, u11i); *u11i = -*u11i;
    A11r = *s1; A11i = 0.0f;
    *s2 = cpolarf(A22r, A22i, u22r, u22i); *u22i = -*u22i;
    A22r = *s2; A22i = 0.0f;
    A12i = A12r = A21i = A21r = 0.0f;
    e = 0;
    break;
  case  2:
  case  4:
  case  6:
    /* [ 0 ? ] */
    /* [ ? 0 ] */
    *u11r = 0.0f;
    *u22r = 0.0f;
    A11r = A21r; A11i = A21i;
    A22r = A12r; A22i = A12i;
    *s1 = cpolarf(A11r, A11i, u12r, u12i); *u12i = -*u12i;
    A11r = *s1; A11i = 0.0f;
    *s2 = cpolarf(A22r, A22i, u21r, u21i); *u21i = -*u21i;
    A22r = *s2; A22i = 0.0f;
    A12i = A12r = A21i = A21r = 0.0f;
    e = 0;
    break;
  case 12:
    /* [ 0 * ] */
    /* [ 0 * ] */
    *v11r = 0.0f;
    *v21r = 1.0f;
    *v12r = 1.0f;
    *v22r = 0.0f;
    A11r = A12r; A11i = A12i;
    A21r = A22r; A21i = A22i;
    /* FALLTHRU */
  case  3:
    /* [ * 0 ] */
    /* [ * 0 ] */
    *s1 = cpolarf(A11r, A11i, u11r, u11i); *u11i = -*u11i;
    A11r = *s1; A11i = 0.0f;
    *s2 = cpolarf(A21r, A21i, u22r, u22i); *u22i = -*u22i;
    A21r = *s2; A21i = 0.0f;
    if (A11r < A21r) {
      pvn_fswp(u11r, u21r);
      pvn_fswp(u11i, u21i);
      pvn_fswp(u22r, u12r);
      pvn_fswp(u22i, u12i);
      pvn_fswp(&A11r, &A21r);
    }
    A22i = A22r = A12i = A12r = 0.0f;
    *s1 = hypotf(A11r, A21r);
    *s2 = 0.0f;
    e = 3;
    break;
  case 10:
    /* [ 0 0 ] */
    /* [ * * ] */
    *u11r = 0.0f;
    *u21r = 1.0f;
    *u12r = 1.0f;
    *u22r = 0.0f;
    A11r = A21r; A11i = A21i;
    A12r = A22r; A12i = A22i;
    /* FALLTHRU */
  case  5:
    /* [ * * ] */
    /* [ 0 0 ] */
    *s1 = cpolarf(A11r, A11i, v11r, v11i); *v11i = -*v11i;
    A11r = *s1; A11i = 0.0f;
    *s2 = cpolarf(A12r, A12i, v22r, v22i); *v22i = -*v22i;
    A12r = *s2; A12i = 0.0f;
    if (A11r < A12r) {
      pvn_fswp(&A11r, &A12r);
      pvn_fswp(v11r, v12r);
      pvn_fswp(v11i, v12i);
      pvn_fswp(v22r, v21r);
      pvn_fswp(v22i, v21i);
    }
    A22i = A22r = A21i = A21r = 0.0f;
    *s1 = hypotf(A11r, A12r);
    *s2 = 0.0f;
    e = 5;
    break;
  case  7:
    /* [ * * ] */
    /* [ * 0 ] */
    pvn_fswp(&A11r, &A12r);
    pvn_fswp(&A11i, &A12i);
    A22r = A21r; A22i = A21i;
    *v11r = 0.0f;
    *v22r = 0.0f;
    *s1 = cpolarf(A11r, A11i, v21r, v21i); *v21i = -*v21i;
    A11r = *s1; A11i = 0.0f;
    *s1 = cpolarf(A12r, A12i, v12r, v12i); *v12i = -*v12i;
    A12r = *s1; A12i = 0.0f;
    pvn_cmul(&A21r, &A21i, A22r, A22i, *v12r, *v12i);
    A22r = A21r; A22i = A21i;
    *s2 = cpolarf(A22r, A22i, u22r, u22i); *u22i = -*u22i;
    A22r = *s2; A22i = 0.0f;
    *s2 = *s1 = A21i = A21r = 0.0f;
    e = 13;
    break;
  case 11:
    /* [ * 0 ] */
    /* [ * * ] */
    *u11r = 0.0f;
    *u12r = 1.0f;
    *u22r = 0.0f;
    pvn_fswp(&A11r, &A22r);
    pvn_fswp(&A11i, &A22i);
    A12r = A21r; A12i = A21i;
    *v11r = 0.0f;
    *v22r = 0.0f;
    *s1 = cpolarf(A11r, A11i, v21r, v21i); *v21i = -*v21i;
    A11r = *s1; A11i = 0.0f;
    *s1 = cpolarf(A12r, A12i, v12r, v12i); *v12i = -*v12i;
    A12r = *s1; A12i = 0.0f;
    pvn_cmul(&A21r, &A21i, A22r, A22i, *v12r, *v12i);
    A22r = A21r; A22i = A21i;
    *s2 = cpolarf(A22r, A22i, u21r, u21i); *u21i = -*u21i;
    A22r = *s2; A22i = 0.0f;
    *s2 = *s1 = A21i = A21r = 0.0f;
    e = 13;
    break;
  case 13:
    /* [ * * ] */
    /* [ 0 * ] */
    *s1 = cpolarf(A11r, A11i, v11r, v11i); *v11i = -*v11i;
    A11r = *s1; A11i = 0.0f;
    *s1 = cpolarf(A12r, A12i, v22r, v22i); *v22i = -*v22i;
    A12r = *s1; A12i = 0.0f;
    pvn_cmul(&A21r, &A21i, A22r, A22i, *v22r, *v22i);
    A22r = A21r; A22i = A21i;
    *s2 = cpolarf(A22r, A22i, u22r, u22i); *u22i = -*u22i;
    A22r = *s2; A22i = 0.0f;
    *s2 = *s1 = A21i = A21r = 0.0f;
    e = 13;
    break;
  case 14:
    /* [ 0 * ] */
    /* [ * * ] */
    *u11r = 0.0f;
    *u12r = 1.0f;
    *u22r = 0.0f;
    A11r = A21r; A11i = A21i;
    pvn_fswp(&A12r, &A22r);
    pvn_fswp(&A12i, &A22i);
    *s1 = cpolarf(A11r, A11i, v11r, v11i); *v11i = -*v11i;
    A11r = *s1; A11i = 0.0f;
    *s1 = cpolarf(A12r, A12i, v22r, v22i); *v22i = -*v22i;
    A12r = *s1; A12i = 0.0f;
    pvn_cmul(&A21r, &A21i, A22r, A22i, *v22r, *v22i);
    A22r = A21r; A22i = A21i;
    *s2 = cpolarf(A22r, A22i, u21r, u21i); *u21i = -*u21i;
    A22r = *s2; A22i = 0.0f;
    *s2 = *s1 = A21i = A21r = 0.0f;
    e = 13;
    break;
  case 15:
    /* [ * * ] */
    /* [ * * ] */
    e = 15;
    break;
  default:
    return INT_MIN;
  }

  if ((e == 13) && (A11r < A22r))
    e = -13;

#ifdef PVN_PRINTOUT
  char s[17] = { '\0' };
#endif /* PVN_PRINTOUT */
  float tt = 0.0f, ct = 1.0f, st = 0.0f;

  if (e == 15) {
    float B11, B21, B12, B22, C11, C21, C12, C22, S11, S21, S12, S22;
    B11 = cpolarf(A11r, A11i, &C11, &S11); S11 = -S11;
    B21 = cpolarf(A21r, A21i, &C21, &S21); S21 = -S21;
    B12 = cpolarf(A12r, A12i, &C12, &S12); S12 = -S12;
    B22 = cpolarf(A22r, A22i, &C22, &S22); S22 = -S22;
    *s1 = hypotf(B11, B21);
    *s2 = hypotf(B12, B22);
    if (*s1 < *s2) {
      pvn_fswp(&A11r, &A12r);
      pvn_fswp(&A11i, &A12i);
      pvn_fswp(&A21r, &A22r);
      pvn_fswp(&A21i, &A22i);
      pvn_fswp(&B11, &B12);
      pvn_fswp(&B21, &B22);
      pvn_fswp(&C11, &C12);
      pvn_fswp(&C21, &C22);
      pvn_fswp(&S11, &S12);
      pvn_fswp(&S21, &S22);
      pvn_fswp(s1, s2);
      pvn_fswp(v11r, v12r);
      pvn_fswp(v11i, v12i);
      pvn_fswp(v21r, v22r);
      pvn_fswp(v21i, v22i);
    }
    if (B11 < B21) {
      pvn_fswp(u11r, u21r);
      pvn_fswp(u11i, u21i);
      pvn_fswp(u12r, u22r);
      pvn_fswp(u12i, u22i);
      pvn_fswp(&A11r, &A21r);
      pvn_fswp(&A11i, &A21i);
      pvn_fswp(&A12r, &A22r);
      pvn_fswp(&A12i, &A22i);
      pvn_fswp(&B11, &B21);
      pvn_fswp(&B12, &B22);
      pvn_fswp(&C11, &C21);
      pvn_fswp(&C12, &C22);
      pvn_fswp(&S11, &S21);
      pvn_fswp(&S12, &S22);
    }
    pvn_cmul(&ct, &st, C11, S11, *u11r, *u11i);
    *u11r = ct; *u11i = st;
    pvn_cmul(&ct, &st, C11, S11, *u12r, *u12i);
    *u12r = ct; *u12i = st;
    A11r = B11; A11i = 0.0f;
    pvn_cmul(&ct, &st, C11, S11, A12r, A12i);
    A12r = ct; A12i = st;
    pvn_cmul(&ct, &st, C21, S21, *u21r, *u21i);
    *u21r = ct; *u21i = st;
    pvn_cmul(&ct, &st, C21, S21, *u22r, *u22i);
    *u22r = ct; *u22i = st;
    A21r = B21; A21i = 0.0f;
    pvn_cmul(&ct, &st, C21, S21, A22r, A22i);
    A22r = ct; A22i = st;
    /* rotate */
    tt = (A21r / A11r);
    /* 1 / cos */
    ct = hypotf(tt, 1.0f);
    /* apply the left Givens rotation to A and U */
    st = -tt;
    A21r = A12r;
    A21i = A12i;
    if (ct == 1.0f) {
      A12r = fmaf(tt, A22r, A12r);
      A12i = fmaf(tt, A22i, A12i);
      A22r = fmaf(st, A21r, A22r);
      A22i = fmaf(st, A21i, A22i);
      A21r = *u11r;
      A21i = *u11i;
      *u11r = fmaf(tt, *u21r, *u11r);
      *u11i = fmaf(tt, *u21i, *u11i);
      *u21r = fmaf(st,  A21r, *u21r);
      *u21i = fmaf(st,  A21i, *u21i);
      A21r = *u12r;
      A21i = *u12i;
      *u12r = fmaf(tt, *u22r, *u12r);
      *u12i = fmaf(tt, *u22i, *u12i);
      *u22r = fmaf(st,  A21r, *u22r);
      *u22i = fmaf(st,  A21i, *u22i);
      st = tt;
    }
    else {
      A12r = (fmaf(tt, A22r, A12r) / ct);
      A12i = (fmaf(tt, A22i, A12i) / ct);
      A22r = (fmaf(st, A21r, A22r) / ct);
      A22i = (fmaf(st, A21i, A22i) / ct);
      A21r = *u11r;
      A21i = *u11i;
      *u11r = (fmaf(tt, *u21r, *u11r) / ct);
      *u11i = (fmaf(tt, *u21i, *u11i) / ct);
      *u21r = (fmaf(st,  A21r, *u21r) / ct);
      *u21i = (fmaf(st,  A21i, *u21i) / ct);
      A21r = *u12r;
      A21i = *u12i;
      *u12r = (fmaf(tt, *u22r, *u12r) / ct);
      *u12i = (fmaf(tt, *u22i, *u12i) / ct);
      *u22r = (fmaf(st,  A21r, *u22r) / ct);
      *u22i = (fmaf(st,  A21i, *u22i) / ct);
      st = (tt / ct);
      ct = (1.0f / ct);
    }
    A11r = *s1;
    B12 = cpolarf(A12r, A12i, &C12, &S12); S12 = -S12;
    A12r = B12; A12i = 0.0f;
    pvn_cmul(&A21r, &A21i, A22r, A22i, C12, S12);
    A22r = A21r; A22i = A21i;
    pvn_cmul(&A21r, &A21i, *v12r, *v12i, C12, S12);
    *v12r = A21r; *v12i = A21i;
    pvn_cmul(&A21r, &A21i, *v22r, *v22i, C12, S12);
    *v22r = A21r; *v22i = A21i;
    B22 = cpolarf(A22r, A22i, &C22, &S22); S22 = -S22;
    A22r = B22; A22i = 0.0f;
    pvn_cmul(&A21r, &A21i, C22, S22, *u21r, *u21i);
    *u21r = A21r; *u21i = A21i;
    pvn_cmul(&A21r, &A21i, C22, S22, *u22r, *u22i);
    *u22r = A21r; *u22i = A21i;
    A21i = A21r = 0.0f;
    if (A12r == 0.0f)
      e = 0;
    else if (A22r == 0.0f) {
      *s1 = hypotf(*s1, *s2);
      *s2 = 0.0f;
      e = 5;
    }
    else if (A11r < A22r)
      e = -13;
    else
      e = 13;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(ϑ)=%s, ", pvn_stoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(ϑ)=%s, ", pvn_stoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(ϑ)=%s\n", pvn_stoa(s, st));
#endif /* PVN_PRINTOUT */
  }

  if (e == 3) {
    tt = (A21r / A11r);
    /* 1 / cos */
    ct = hypotf(tt, 1.0f);
    /* apply the left Givens rotation to U */
    st = -tt;
    A21r = *u11r;
    A21i = *u11i;
    if (ct == 1.0f) {
      *u11r = fmaf(tt, *u21r, *u11r);
      *u11i = fmaf(tt, *u21i, *u11i);
      *u21r = fmaf(st,  A21r, *u21r);
      *u21i = fmaf(st,  A21i, *u21i);
      A21r = *u12r;
      A21i = *u12i;
      *u12r = fmaf(tt, *u22r, *u12r);
      *u12i = fmaf(tt, *u22i, *u12i);
      *u22r = fmaf(st,  A21r, *u22r);
      *u22i = fmaf(st,  A21i, *u22i);
      st = tt;
    }
    else {
      *u11r = (fmaf(tt, *u21r, *u11r) / ct);
      *u11i = (fmaf(tt, *u21i, *u11i) / ct);
      *u21r = (fmaf(st,  A21r, *u21r) / ct);
      *u21i = (fmaf(st,  A21i, *u21i) / ct);
      A21r = *u12r;
      A21i = *u12i;
      *u12r = (fmaf(tt, *u22r, *u12r) / ct);
      *u12i = (fmaf(tt, *u22i, *u12i) / ct);
      *u22r = (fmaf(st,  A21r, *u22r) / ct);
      *u22i = (fmaf(st,  A21i, *u22i) / ct);
      st = (tt / ct);
      ct = (1.0f / ct);
    }
    A11r = *s1;
    A21i = A21r = 0.0f;
    e = 0;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(ϑ)=%s, ", pvn_stoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(ϑ)=%s, ", pvn_stoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(ϑ)=%s\n", pvn_stoa(s, st));
#endif /* PVN_PRINTOUT */
  }

  if (e == 5) {
    tt = (A12r / A11r);
    /* 1 / cos */
    ct = hypotf(tt, 1.0f);
    /* apply the right Givens rotation to V */
    st = -tt;
    A12r = *v11r;
    A12i = *v11i;
    if (ct == 1.0f) {
      *v11r = fmaf(tt, *v12r, *v11r);
      *v11i = fmaf(tt, *v12i, *v11i);
      *v12r = fmaf(st,  A12r, *v12r);
      *v12i = fmaf(st,  A12i, *v12i);
      A12r = *v21r;
      A12i = *v21i;
      *v21r = fmaf(tt, *v22r, *v21r);
      *v21i = fmaf(tt, *v22i, *v21i);
      *v22r = fmaf(st,  A12r, *v22r);
      *v22i = fmaf(st,  A12i, *v22i);
      st = tt;
    }
    else {
      *v11r = (fmaf(tt, *v12r, *v11r) / ct);
      *v11i = (fmaf(tt, *v12i, *v11i) / ct);
      *v12r = (fmaf(st,  A12r, *v12r) / ct);
      *v12i = (fmaf(st,  A12i, *v12i) / ct);
      A12r = *v21r;
      A12i = *v21i;
      *v21r = (fmaf(tt, *v22r, *v21r) / ct);
      *v21i = (fmaf(tt, *v22i, *v21i) / ct);
      *v22r = (fmaf(st,  A12r, *v22r) / ct);
      *v22i = (fmaf(st,  A12i, *v22i) / ct);
      st = (tt / ct);
      ct = (1.0f / ct);
    }
    A11r = *s1;
    A12i = A12r = 0.0f;
    e = 0;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(θ)=%s, ", pvn_stoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(θ)=%s, ", pvn_stoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(θ)=%s\n", pvn_stoa(s, st));
#endif /* PVN_PRINTOUT */
  }

#ifdef PVN_PRINTOUT
  (void)dprintf(PVN_PRINTOUT, "\tA =\n");
  (void)dprintf(PVN_PRINTOUT, "(%s,", pvn_stoa(s, scalbnf(A11r, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s) ", pvn_stoa(s, scalbnf(A11i, -*es)));
  (void)dprintf(PVN_PRINTOUT, "(%s,", pvn_stoa(s, scalbnf(A12r, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s)\n", pvn_stoa(s, scalbnf(A12i, -*es)));
  (void)dprintf(PVN_PRINTOUT, "(%s,", pvn_stoa(s, scalbnf(A21r, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s) ", pvn_stoa(s, scalbnf(A21i, -*es)));
  (void)dprintf(PVN_PRINTOUT, "(%s,", pvn_stoa(s, scalbnf(A22r, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s)\n", pvn_stoa(s, scalbnf(A22i, -*es)));
#endif /* PVN_PRINTOUT */

  if (abs(e) == 13) {
    float tf = 0.0f, cf = 1.0f, sf = 0.0f, tp = 0.0f, cp = 1.0f, sp = 0.0f;
    if (e == -13) {
      float tf_ = 0.0f, cf_ = 1.0f, sf_ = 0.0f, tp_ = 0.0f, cp_ = 1.0f, sp_ = 0.0f;
      slpsv2(A22r, A12r, A11r, &tf_, &cf_, &sf_, &tp_, &cp_, &sp_, s1, s2, es
#ifdef PVN_PRINTOUT
             , s
#endif /* PVN_PRINTOUT */
             );
      tf = (1.0f / tp_);
      cf = sp_;
      sf = cp_;
      tp = (1.0f / tf_);
      cp = sf_;
      sp = cf_;
    }
    else
      slpsv2(A11r, A12r, A22r, &tf, &cf, &sf, &tp, &cp, &sp, s1, s2, es
#ifdef PVN_PRINTOUT
             , s
#endif /* PVN_PRINTOUT */
             );

    if (tf != 0.0f) {
      A21r = *u11r; A21i = *u11i;
      *u11r = (cf * *u11r + sf * *u21r);
      *u11i = (cf * *u11i + sf * *u21i);
      *u21r = (cf * *u21r - sf *  A21r);
      *u21i = (cf * *u21i - sf *  A21i);
      A21r = *u12r; A21i = *u12i;
      *u12r = (cf * *u12r + sf * *u22r);
      *u12i = (cf * *u12i + sf * *u22i);
      *u22r = (cf * *u22r - sf *  A21r);
      *u22i = (cf * *u22i - sf *  A21i);
      A21r = -0.0f;
      A21i = ((e == 13) ? 0.0f : -0.0f);
    }
    else {
      A21r = 0.0f;
      A21i = ((e == 13) ? 0.0f : -0.0f);
    }

    if (tp != 0.0f) {
      A21r = *v11r; A21i = *v11i;
      *v11r = (*v11r * cp + *v12r * sp);
      *v11i = (*v11i * cp + *v12i * sp);
      *v12r = (*v12r * cp -  A21r * sp);
      *v12i = (*v12i * cp -  A21i * sp);
      A21r = *v21r; A21i = *v21i;
      *v21r = (*v21r * cp + *v22r * sp);
      *v21i = (*v21i * cp + *v22i * sp);
      *v22r = (*v22r * cp -  A21r * sp);
      *v22i = (*v22i * cp -  A21i * sp);
      A21r = -0.0f;
      A21i = ((e == 13) ? 0.0f : -0.0f);
    }
    else {
      A21r = 0.0f;
      A21i = ((e == 13) ? 0.0f : -0.0f);
    }

    if (e == -13) {
      *u21r = -*u21r;
      *u21i = -*u21i;
      *u22r = -*u22r;
      *u22i = -*u22i;
      *v12r = -*v12r;
      *v12i = -*v12i;
      *v22r = -*v22r;
      *v22i = -*v22i;
    }
  }

  if (ef_cmpf(es[1], *s1, es[2], *s2) < 0) {
    pvn_fswp(u11r, u21r);
    pvn_fswp(u11i, u21i);
    pvn_fswp(u12r, u22r);
    pvn_fswp(u12i, u22i);
    pvn_fswp(v11r, v12r);
    pvn_fswp(v11i, v12i);
    pvn_fswp(v21r, v22r);
    pvn_fswp(v21i, v22i);
    pvn_fswp(s1, s2);
  }
  pvn_fswp(u21r, u12r);
  pvn_fswp(u21i, u12i);
  *u11i = -*u11i;
  *u21i = -*u21i;
  *u12i = -*u12i;
  *u22i = -*u22i;
  return knd;
}

static inline void ef_mul(int *const e, double *const f, const int e1, const double f1, const int e2, const double f2)
{
  PVN_ASSERT(e);
  PVN_ASSERT(f);
  *f = (f1 * f2);
  *f = frexp(*f, e);
  *e += (e1 + e2);
}

static inline void ef_div(int *const e, double *const f, const int e1, const double f1, const int e2, const double f2)
{
  PVN_ASSERT(e);
  PVN_ASSERT(f);
  *f = (f1 / f2);
  *f = frexp(*f, e);
  *e += (e1 - e2);
}

/* f1, f2 >= 0 */
static inline int ef_cmp(const int e1, const double f1, const int e2, const double f2)
{
  if (f2 == 0.0)
    return (f1 != 0.0);
  if (f1 == 0.0)
    return -1;
  if (e1 < e2)
    return -1;
  if (e2 < e1)
    return 1;
  if (f1 < f2)
    return -1;
  if (f2 < f1)
    return 1;
  return 0;
}

/* TODO: the Kahan's algorithm for determinants of order two, with a power-of-two prescaling, might have been used instead */
static double dQR(double *const A11, double *const A21, double *const A12, double *const A22, double *const w1)
{
  PVN_ASSERT(A11);
  PVN_ASSERT(A21);
  PVN_ASSERT(A12);
  PVN_ASSERT(A22);
  PVN_ASSERT(w1);
#ifdef PVN_SV2_SAFE
  if (feclearexcept(FE_INEXACT))
    return __builtin_nan("");
#endif /* PVN_SV2_SAFE */
  const double t = (*A21 / *A11);
#ifdef PVN_SV2_SAFE
  if (fetestexcept(FE_INEXACT) & FE_INEXACT) {
#ifndef T
#ifdef PVN_QUADMATH
#define T __float128
#else /* !PVN_QUADMATH */
#define T long double
#endif /* ?PVN_QUADMATH */
#else /* T */
#error T already defined
#endif /* ?T */
    const T d = (T)*A11;
    *A11 = *w1;
    *w1 = hypot(t, 1.0);
    if (copysign(1.0, *A12) == copysign(1.0, *A22)) {
      const double a12 = fma(*A22, t, *A12);
      const T
        a = ((T)*A22 * d),
        b = ((T)*A12 * (T)*A21),
        c = (a - b);
      *A22 = (double)(c / d);
      if (*w1 == 1.0)
        *A12 = a12;
      else {
        *A12 = (a12 / *w1);
        *A22 = (*A22 / *w1);
      }
      *A21 = 1.0;
    }
    else {
      const double a22 = fma(-*A12, t, *A22);
      const T
        a = ((T)*A12 * d),
        b = ((T)*A22 * (T)*A21),
        c = (a + b);
      *A12 = (double)(c / d);
      if (*w1 == 1.0)
        *A22 = a22;
      else {
        *A12 = (*A12 / *w1);
        *A22 = (a22 / *w1);
      }
      *A21 = -1.0;
    }
#undef T
    return t;
  }
  else /* tan exact */
    *A21 = -0.0;
#else /* !PVN_SV2_SAFE */
  *A21 = 0.0;
#endif /* ?PVN_SV2_SAFE */
  *A11 = *w1;
  *w1 = hypot(t, 1.0);
  const double a12 = fma( *A22, t, *A12);
  const double a22 = fma(-*A12, t, *A22);
  if (*w1 == 1.0) {
    *A12 = a12;
    *A22 = a22;
  }
  else {
    *A12 = (a12 / *w1);
    *A22 = (a22 / *w1);
  }
  return t;
}

static void dlpsv2(const double A11, const double A12, const double A22, double *const tf, double *const cf, double *const sf, double *const tp, double *const cp, double *const sp, double *const s1, double *const s2, int *const es
#ifdef PVN_PRINTOUT
                   , char *const s
#endif /* PVN_PRINTOUT */
                   )
{
  PVN_ASSERT(tf);
  PVN_ASSERT(cf);
  PVN_ASSERT(sf);
  PVN_ASSERT(tp);
  PVN_ASSERT(cp);
  PVN_ASSERT(sp);
  PVN_ASSERT(s1);
  PVN_ASSERT(s2);
  PVN_ASSERT(es);
  int ae = 0, be = 0, abe = 0, a_be = 0, de = 0, ne = 0, t2e = 0;
  double af = 0.0, bf = 0.0, abf = 0.0, a_bf = 0.0, df = 0.0, nf = 0.0, t2f = 0.0, t2 = 0.0;

  /* tan(2φ) */
  if (A11 == A22)
    t2 = ((2.0 * A22) / A12);
  else if (A12 == A22) {
    /* (2 * A12 * A22) / (A11 * A11) */
    af = frexp(A11, &ae);
    abf = frexp(A12, &abe);
    bf = frexp(A22, &be);
    ef_mul(&ne, &nf, abe, abf, be, bf);
    ++ne;
    ef_mul(&de, &df, ae, af, ae, af);
    ef_div(&t2e, &t2f, ne, nf, de, df);
    t2 = scalbn(t2f, t2e);
  }
  else if ((A11 / A12) < (DBL_EPSILON * 0.5))
    t2 = ((2.0 * A22) / A12);
#ifdef PVN_SV2_SAFE
  else {
    /* should never overflow */
    const double a = hypot(A11, A12);
    af = frexp(a, &ae);
    const double b = A22;
    bf = frexp(b, &be);
    abf = (a + b);
    if (!isfinite(abf)) {
      abf = ((0.5 * a) + (0.5 * b));
      de = 1;
    }
    abf = frexp(abf, &abe);
    abe += de;
    a_bf = (a - b);
    if (a == b)
      de = 0;
    else if (a_bf >= DBL_MIN) {
      a_bf = frexp(a_bf, &a_be);
      de = 1;
    }
    else {
      de = ((DBL_MIN_EXP + DBL_MANT_DIG) - pvn_imin(ae, be));
      a_bf = (scalbn(af, (ae + de)) - scalbn(bf, (be + de)));
      a_bf = frexp(a_bf, &a_be);
      a_be -= de;
      de = -1;
    }
    if (de)
      ef_mul(&de, &df, a_be, a_bf, abe, abf);
    af = frexp(A12, &ae);
    ef_mul(&ne, &nf, ae, af, be, bf);
    ++ne;
    ef_div(&t2e, &t2f, ne, nf, de, df);
    if (isfinite(t2f))
      t2 = scalbn(t2f, t2e);
    else
      t2 = t2f;
  }
#else /* !PVN_SV2_SAFE */
  else if (A11 > A12) {
    const double x = (A12 / A11);
    const double y = (A22 / A11);
    t2 = (((2.0 * x) * y) / fmax(fma((x - y), (x + y), 1.0), 0.0));
  }
  else if (A12 > A11) {
    const double x = (A11 / A12);
    const double y = (A22 / A12);
    t2 = ((2.0 * y) / fmax(fma((x - y), (x + y), 1.0), 0.0));
  }
  else {
    const double y = (A22 / A12);
    t2 = ((2.0 * y) / fmax(fma((1.0 - y), (1.0 + y), 1.0), 0.0));
  }
#endif /* ?PVN_SV2_SAFE */

  /* tan(φ) */
  if (isfinite(t2))
    *tf = (t2 / (1.0 + hypot(t2, 1.0)));
  else
    *tf = 1.0;
#ifdef PVN_PRINTOUT
  if (s) (void)dprintf(PVN_PRINTOUT, "tan(φ)=%s, ", pvn_dtoa(s, *tf));
#endif /* PVN_PRINTOUT */
  *cf = hypot(*tf, 1.0); /* sec(φ) */
  *sf = (*tf / *cf); /* sin(φ) */
  nf = frexp(*cf, &ne);
  *cf = (1.0 / *cf); /* cos(φ) */

  *sp = fma(*tf, A22, A12);
  *tp = (*sp / A11); /* tan(ψ) */
#ifdef PVN_PRINTOUT
  if (s) (void)dprintf(PVN_PRINTOUT, "tan(ψ)=%s\n", pvn_dtoa(s, *tp));
#endif /* PVN_PRINTOUT */

  af = frexp(A11, &ae);
  bf = frexp(A22, &be);

  if (isfinite(*tp)) {
    *cp = hypot(*tp, 1.0); /* sec(ψ) */
    *sp = (*tp / *cp); /* sin(ψ) */
    df = frexp(*cp, &de);
    *cp = (1.0 / *cp); /* cos(ψ) */
    ef_div(&t2e, &t2f, ne, nf, de, df);
    /* s2 = A22 * (sec(φ) / sec(ψ)) */
    ef_mul(&abe, &abf, be, bf, t2e, t2f);
    /* s1 = A11 * (sec(ψ) / sec(φ)) */
    ef_div(&a_be, &a_bf, ae, af, t2e, t2f);
  }
  else {
    /* s1 ≈ A12 + tan(φ) * A22 */
    a_bf = frexp(*sp, &a_be);
    /* tan(ψ) so large that sec(ψ) ≈ tan(ψ) */
    ef_div(&t2e, &t2f, ae, af, a_be, a_bf);
    *sp = 1.0; /* sin(ψ) */
    *cp = scalbn(t2f, t2e); /* cos(ψ) */
    /* s2 ≈ A22 * cos(ψ) */
    ef_mul(&abe, &abf, be, bf, t2e, t2f);
  }

#ifdef PVN_PRINTOUT
  if (s) {
    (void)dprintf(PVN_PRINTOUT, "cos(φ)=%s, ", pvn_dtoa(s, *cf));
    (void)dprintf(PVN_PRINTOUT, "sin(φ)=%s\n", pvn_dtoa(s, *sf));
    (void)dprintf(PVN_PRINTOUT, "cos(ψ)=%s, ", pvn_dtoa(s, *cf));
    (void)dprintf(PVN_PRINTOUT, "sin(ψ)=%s\n", pvn_dtoa(s, *sf));
  }
#endif /* PVN_PRINTOUT */

  *s1 = a_bf;
  es[1] = a_be;
  *s2 = abf;
  es[2] = abe;
}

int PVN_FABI(pvn_dljsv2,PVN_DLJSV2)
(const double *const a11, const double *const a21, const double *const a12, const double *const a22,
 double *const u11, double *const u21, double *const u12, double *const u22,
 double *const v11, double *const v21, double *const v12, double *const v22,
 double *const s1, double *const s2, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a21);
  PVN_ASSERT(a12);
  PVN_ASSERT(a22);
  PVN_ASSERT(u11);
  PVN_ASSERT(u21);
  PVN_ASSERT(u12);
  PVN_ASSERT(u22);
  PVN_ASSERT(v11);
  PVN_ASSERT(v21);
  PVN_ASSERT(v12);
  PVN_ASSERT(v22);
  PVN_ASSERT(s1);
  PVN_ASSERT(s2);
  PVN_ASSERT(es);
  if (!isfinite(*a11))
    return -1;
  if (!isfinite(*a21))
    return -2;
  if (!isfinite(*a12))
    return -3;
  if (!isfinite(*a22))
    return -4;
  /* should not fail beyond this point when *es == 0 */
  int knd = 0, mxe = INT_MIN, e = INT_MIN;
  if (*a11 != 0.0) {
    knd |= 1;
    (void)frexp(*a11, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a21 != 0.0) {
    knd |= 2;
    (void)frexp(*a21, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a12 != 0.0) {
    knd |= 4;
    (void)frexp(*a12, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a22 != 0.0) {
    knd |= 8;
    (void)frexp(*a22, &e);
    if (mxe < e)
      mxe = e;
  }
  switch (knd) {
  case  0:
  case  1:
  case  2:
  case  4:
  case  6:
  case  8:
  case  9:
    e = 0;
    break;
  case  3:
  case  5:
  case  7:
  case 10:
  case 11:
  case 12:
  case 13:
  case 14:
    e = (DBL_MAX_EXP - mxe - 1);
    break;
  case 15:
    e = (DBL_MAX_EXP - mxe - 2);
    break;
  default:
    return INT_MIN;
  }
  mxe = *es;
  if (!*es)
    *es = e;
  else if (*es < 0)
    ++*es;
  es[2] = es[1] = 0;

  /* scaling by 2^(*es), can only fail when mxe != 0 */
  double A11 = *a11, A21 = *a21, A12 = *a12, A22 = *a22;
  if (*es) {
    A11 = scalbn(A11, *es);
    A21 = scalbn(A21, *es);
    A12 = scalbn(A12, *es);
    A22 = scalbn(A22, *es);
    if (mxe) {
      if (!isfinite(A11))
        return -5;
      if (!isfinite(A21))
        return -6;
      if (!isfinite(A12))
        return -7;
      if (!isfinite(A22))
        return -8;
    }
    if (*es < 0) {
      knd = 0;
      if (A11 != 0.0)
        knd |= 1;
      if (A21 != 0.0)
        knd |= 2;
      if (A12 != 0.0)
        knd |= 4;
      if (A22 != 0.0)
        knd |= 8;
    }
  }

  *u11 = 1.0;
  *u21 = 0.0;
  *u12 = 0.0;
  *u22 = 1.0;
  *v11 = 1.0;
  *v21 = 0.0;
  *v12 = 0.0;
  *v22 = 1.0;
  *s1 = 0.0;
  *s2 = 0.0;

  /* simplify the form of A */
  switch (knd) {
  case  0:
    /* [ 0 0 ] */
    /* [ 0 0 ] */
    *u11 = copysign(1.0, A11);
    *u22 = copysign(1.0, A22);
    A22 = A12 = A21 = A11 = 0.0;
    break;
  case  1:
    /* [ * 0 ] */
    /* [ 0 0 ] */
    if (A11 < 0.0) {
      *u11 = -1.0;
      A11 = -A11;
    }
    *u22 = copysign(1.0, A22);
    A22 = A12 = A21 = 0.0;
    *s1 = A11;
    break;
  case  2:
    /* [ 0 0 ] */
    /* [ * 0 ] */
    *u11 = 0.0;
    *u22 = 0.0;
    A11 = A21;
    A22 = A12;
    if (A11 < 0.0) {
      *u12 = -1.0;
      A11 = -A11;
    }
    else
      *u12 = 1.0;
    *u21 = copysign(1.0, A12);
    A22 = A12 = A21 = 0.0;
    *s1 = A11;
    break;
  case  3:
    /* [ * 0 ] */
    /* [ * 0 ] */
    A12 = fabs(A11);
    A22 = fabs(A21);
    if (A12 < A22) {
      *u11 = 0.0;
      *u21 = copysign(1.0, A11);
      *u12 = copysign(1.0, A21);
      *u22 = 0.0;
      A11 = A22;
      A21 = A12;
    }
    else {
      *u11 = copysign(1.0, A11);
      *u22 = copysign(1.0, A21);
      A11 = A12;
      A21 = A22;
    }
    A22 = A12 = 0.0;
    *s1 = hypot(A11, A21);
    e = 3;
    break;
  case  4:
    /* [ 0 * ] */
    /* [ 0 0 ] */
    *u11 = copysign(1.0, A12);
    *u22 = copysign(1.0, A21);
    A11 = fabs(A12);
    A22 = A12 = A21 = 0.0;
    *v11 = 0.0;
    *v21 = 1.0;
    *v12 = 1.0;
    *v22 = 0.0;
    *s1 = A11;
    break;
  case  5:
    /* [ * * ] */
    /* [ 0 0 ] */
    A21 = fabs(A11);
    A22 = fabs(A12);
    if (A21 < A22) {
      *v11 = 0.0;
      *v21 = copysign(1.0, A12);
      *v12 = copysign(1.0, A11);
      *v22 = 0.0;
      A11 = A22;
      A12 = A21;
    }
    else {
      *v11 = copysign(1.0, A11);
      *v22 = copysign(1.0, A12);
      A11 = A21;
      A12 = A22;
    }
    A22 = A21 = 0.0;
    *s1 = hypot(A11, A12);
    e = 5;
    break;
  case  6:
    /* [ 0 * ] */
    /* [ * 0 ] */
    *u11 = copysign(1.0, A12);
    *u22 = copysign(1.0, A21);
    A11 = fabs(A12);
    A22 = fabs(A21);
    A12 = A21 = 0.0;
    *v11 = 0.0;
    *v21 = 1.0;
    *v12 = 1.0;
    *v22 = 0.0;
    *s1 = A11;
    *s2 = A22;
    break;
  case  7:
    /* [ * * ] */
    /* [ * 0 ] */
    A22 = A11;
    A11 = A12;
    A12 = A22;
    A22 = A21;
    A21 = 0.0;
    *v11 = 0.0;
    *v22 = 0.0;
    if (A11 < 0.0) {
      A11 = -A11;
      *v21 = -1.0;
    }
    else
      *v21 = 1.0;
    if (A12 < 0.0) {
      A12 = -A12;
      A22 = -A22;
      *v12 = -1.0;
    }
    else
      *v12 = 1.0;
    if (A22 < 0.0) {
      *u22 = -1.0;
      A22 = -A22;
    }
    e = 13;
    break;
  case  8:
    /* [ 0 0 ] */
    /* [ 0 * ] */
    *u11 = 0.0;
    *u21 = copysign(1.0, A11);
    *u12 = copysign(1.0, A22);
    *u22 = 0.0;
    A11 = fabs(A22);
    A22 = A12 = A21 = 0.0;
    *v11 = 0.0;
    *v21 = 1.0;
    *v12 = 1.0;
    *v22 = 0.0;
    *s1 = A11;
    break;
  case  9:
    /* [ * 0 ] */
    /* [ 0 * ] */
    if (A11 < 0.0) {
      *u11 = -1.0;
      A11 = -A11;
    }
    if (A22 < 0.0) {
      *u22 = -1.0;
      A22 = -A22;
    }
    A12 = A21 = 0.0;
    *s1 = A11;
    *s2 = A22;
    break;
  case 10:
    /* [ 0 0 ] */
    /* [ * * ] */
    *u11 = 0.0;
    *u21 = 1.0;
    *u12 = 1.0;
    *u22 = 0.0;
    A11 = A21;
    A12 = A22;
    A21 = fabs(A11);
    A22 = fabs(A12);
    if (A21 < A22) {
      *v11 = 0.0;
      *v21 = copysign(1.0, A12);
      *v12 = copysign(1.0, A11);
      *v22 = 0.0;
      A11 = A22;
      A12 = A21;
    }
    else {
      *v11 = copysign(1.0, A11);
      *v22 = copysign(1.0, A12);
      A11 = A21;
      A12 = A22;
    }
    A22 = A21 = 0.0;
    *s1 = hypot(A11, A12);
    e = 5;
    break;
  case 11:
    /* [ * 0 ] */
    /* [ * * ] */
    *u11 = 0.0;
    *u12 = 1.0;
    *u22 = 0.0;
    A12 = A11;
    A11 = A22;
    A22 = A12;
    A12 = A21;
    A21 = 0.0;
    *v11 = 0.0;
    *v22 = 0.0;
    if (A11 < 0.0) {
      A11 = -A11;
      *v21 = -1.0;
    }
    else
      *v21 = 1.0;
    if (A12 < 0.0) {
      A12 = -A12;
      A22 = -A22;
      *v12 = -1.0;
    }
    else
      *v12 = 1.0;
    if (A22 < 0.0) {
      *u21 = -1.0;
      A22 = -A22;
    }
    else
      *u21 = 1.0;
    e = 13;
    break;
  case 12:
    /* [ 0 * ] */
    /* [ 0 * ] */
    A11 = A12;
    A21 = A22;
    *v11 = 0.0;
    *v21 = 1.0;
    *v12 = 1.0;
    *v22 = 0.0;
    A12 = fabs(A11);
    A22 = fabs(A21);
    if (A12 < A22) {
      *u11 = 0.0;
      *u21 = copysign(1.0, A11);
      *u12 = copysign(1.0, A21);
      *u22 = 0.0;
      A11 = A22;
      A21 = A12;
    }
    else {
      *u11 = copysign(1.0, A11);
      *u22 = copysign(1.0, A21);
      A11 = A12;
      A21 = A22;
    }
    A22 = A12 = 0.0;
    *s1 = hypot(A11, A21);
    e = 3;
    break;
  case 13:
    /* [ * * ] */
    /* [ 0 * ] */
    if (A11 < 0.0) {
      A11 = -A11;
      *v11 = -1.0;
    }
    if (A12 < 0.0) {
      A12 = -A12;
      A22 = -A22;
      *v22 = -1.0;
    }
    if (A22 < 0.0) {
      *u22 = -1.0;
      A22 = -A22;
    }
    A21 = 0.0;
    e = 13;
    break;
  case 14:
    /* [ 0 * ] */
    /* [ * * ] */
    *u11 = 0.0;
    *u12 = 1.0;
    *u22 = 0.0;
    A11 = A12;
    A12 = A22;
    A22 = A11;
    A11 = A21;
    A21 = 0.0;
    if (A11 < 0.0) {
      A11 = -A11;
      *v11 = -1.0;
    }
    if (A12 < 0.0) {
      A12 = -A12;
      A22 = -A22;
      *v22 = -1.0;
    }
    if (A22 < 0.0) {
      *u21 = -1.0;
      A22 = -A22;
    }
    else
      *u21 = 1.0;
    e = 13;
    break;
  case 15:
    /* [ * * ] */
    /* [ * * ] */
    e = 15;
    break;
  default:
    return INT_MIN;
  }

  if ((e == 13) && (A11 < A22))
    e = -13;

#ifdef PVN_PRINTOUT
  char s[26] = { '\0' };
  (void)dprintf(PVN_PRINTOUT, "\tA[knd=%d,e=%d] * 2^%d =\n", knd, e, *es);
  (void)dprintf(PVN_PRINTOUT, "%s ", pvn_dtoa(s, A11));
  (void)dprintf(PVN_PRINTOUT, "%s\n", pvn_dtoa(s, A12));
  (void)dprintf(PVN_PRINTOUT, "%s ", pvn_dtoa(s, A21));
  (void)dprintf(PVN_PRINTOUT, "%s\n", pvn_dtoa(s, A22));
#endif /* PVN_PRINTOUT */

  double tt = 0.0, ct = 1.0, st = 0.0;

  if (e == 15) {
    /* [ * * ] */
    /* [ * * ] */
    *s1 = hypot(A11, A21);
    *s2 = hypot(A12, A22);
    if (*s1 < *s2) {
      pvn_dswp(&A11, &A12);
      pvn_dswp(&A21, &A22);
      pvn_dswp(v11, v12);
      pvn_dswp(v21, v22);
      pvn_dswp(s1, s2);
    }
    if (copysign(1.0, A11) != 1.0) {
      *u11 = -*u11;
      *u12 = -*u12;
      A11 = -A11;
      A12 = -A12;
    }
    if (copysign(1.0, A21) != 1.0) {
      *u21 = -*u21;
      *u22 = -*u22;
      A21 = -A21;
      A22 = -A22;
    }
    if (A11 < A21) {
      pvn_dswp(u11, u21);
      pvn_dswp(u12, u22);
      pvn_dswp(&A11, &A21);
      pvn_dswp(&A12, &A22);
    }
    /* U^T(ϑ):
        cos(ϑ)  sin(ϑ)
       -sin(ϑ)  cos(ϑ)
    */
    ct = *s1;
    /* apply the left Givens rotation to A (and maybe to U) */
    tt = dQR(&A11, &A21, &A12, &A22, &ct);
    st = -tt;
    A21 = A12;
    if (ct == 1.0) {
      if ((A12 == 0.0) || (A22 == 0.0)) {
        A21 = *u11;
        *u11 = fma(tt, *u21, *u11);
        *u21 = fma(st,  A21, *u21);
        A21 = *u12;
        *u12 = fma(tt, *u22, *u12);
        *u22 = fma(st,  A21, *u22);
      }
      st = tt;
    }
    else {
      if ((A12 == 0.0) || (A22 == 0.0)) {
        A21 = *u11;
        *u11 = (fma(tt, *u21, *u11) / ct);
        *u21 = (fma(st,  A21, *u21) / ct);
        A21 = *u12;
        *u12 = (fma(tt, *u22, *u12) / ct);
        *u22 = (fma(st,  A21, *u22) / ct);
      }
      st = (tt / ct);
      ct = (1.0 / ct);
    }
    A21 = 0.0;
    if (A12 == 0.0) {
      A12 = 0.0;
      if (copysign(1.0, A22) != 1.0) {
        *u21 = -*u21;
        *u22 = -*u22;
        A22 = -A22;
      }
      e = 0;
    }
    else if (A22 == 0.0) {
      if (copysign(1.0, A12) != 1.0) {
        A12 = -A12;
        A22 = -A22;
        *v12 = -*v12;
        *v22 = -*v22;
      }
      if (copysign(1.0, A22) != 1.0) {
        *u21 = -*u21;
        *u22 = -*u22;
        A22 = 0.0;
      }
      *s1 = hypot(*s1, *s2);
      *s2 = 0.0;
      e = 5;
    }
    else
      e = 13;
    if (A12 < 0.0) {
      A12 = -A12;
      A22 = -A22;
      *v12 = -*v12;
      *v22 = -*v22;
    }
    if (A22 < 0.0) {
      A22 = -A22;
      /* sin(ϑ) is always non-negative by construction */
      /* this is just an extra bit of info, used later */
      st = -st;
    }
    if ((e == 13) && (A11 < A22))
      e = -13;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(ϑ)=%s, ", pvn_dtoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(ϑ)=%s, ", pvn_dtoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(ϑ)=%s\n", pvn_dtoa(s, st));
#endif /* PVN_PRINTOUT */
  }

  if (e == 3) {
    /* [ X 0 ] */
    /* [ x 0 ] */
    /* U^T(θ):
        cos(θ)  sin(θ)
       -sin(θ)  cos(θ)
    */
    tt = (A21 / A11);
    /* 1 / cos */
    ct = hypot(tt, 1.0);
    /* apply the left Givens rotation to U */
    st = -tt;
    A21 = *u11;
    if (ct == 1.0) {
      *u11 = fma(tt, *u21, *u11);
      *u21 = fma(st,  A21, *u21);
      A21 = *u12;
      *u12 = fma(tt, *u22, *u12);
      *u22 = fma(st,  A21, *u22);
      st = tt;
    }
    else {
      *u11 = (fma(tt, *u21, *u11) / ct);
      *u21 = (fma(st,  A21, *u21) / ct);
      A21 = *u12;
      *u12 = (fma(tt, *u22, *u12) / ct);
      *u22 = (fma(st,  A21, *u22) / ct);
      st = (tt / ct);
      ct = (1.0 / ct);
    }
    A11 = *s1;
    A21 = 0.0;
    e = 0;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(ϑ)=%s, ", pvn_dtoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(ϑ)=%s, ", pvn_dtoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(ϑ)=%s\n", pvn_dtoa(s, st));
#endif /* PVN_PRINTOUT */
  }

  if (e == 5) {
    /* [ X x ] */
    /* [ 0 0 ] */
    /* V(θ):
       cos(θ) -sin(θ)
       sin(θ)  cos(θ)
    */
    tt = (A12 / A11);
    /* 1 / cos */
    ct = hypot(tt, 1.0);
    /* apply the right Givens rotation to V */
    st = -tt;
    A12 = *v11;
    if (ct == 1.0) {
      *v11 = fma(tt, *v12, *v11);
      *v12 = fma(st,  A12, *v12);
      A12 = *v21;
      *v21 = fma(tt, *v22, *v21);
      *v22 = fma(st,  A12, *v22);
      st = tt;
    }
    else {
      *v11 = (fma(tt, *v12, *v11) / ct);
      *v12 = (fma(st,  A12, *v12) / ct);
      A12 = *v21;
      *v21 = (fma(tt, *v22, *v21) / ct);
      *v22 = (fma(st,  A12, *v22) / ct);
      st = (tt / ct);
      ct = (1.0 / ct);
    }
    A11 = *s1;
    A12 = 0.0;
    e = 0;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(θ)=%s, ", pvn_dtoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(θ)=%s, ", pvn_dtoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(θ)=%s\n", pvn_dtoa(s, st));
#endif /* PVN_PRINTOUT */
  }

#ifdef PVN_PRINTOUT
  (void)dprintf(PVN_PRINTOUT, "\tA =\n");
  (void)dprintf(PVN_PRINTOUT, "%s ", pvn_dtoa(s, scalbn(A11, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s\n", pvn_dtoa(s, scalbn(A12, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s ", pvn_dtoa(s, scalbn(A21, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s\n", pvn_dtoa(s, scalbn(A22, -*es)));
#endif /* PVN_PRINTOUT */

  if (abs(e) == 13) {
    /* [ x y ] */
    /* [ 0 z ] */

    double tf = 0.0, cf = 1.0, sf = 0.0, tp = 0.0, cp = 1.0, sp = 0.0;
    if (e == -13) {
      double tf_ = 0.0, cf_ = 1.0, sf_ = 0.0, tp_ = 0.0, cp_ = 1.0, sp_ = 0.0;
      dlpsv2(A22, A12, A11, &tf_, &cf_, &sf_, &tp_, &cp_, &sp_, s1, s2, es
#ifdef PVN_PRINTOUT
             , s
#endif /* PVN_PRINTOUT */
             );
      tf = (1.0 / tp_);
      cf = sp_;
      sf = cp_;
      tp = (1.0 / tf_);
      cp = sf_;
      sp = cf_;
    }
    else
      dlpsv2(A11, A12, A22, &tf, &cf, &sf, &tp, &cp, &sp, s1, s2, es
#ifdef PVN_PRINTOUT
             , s
#endif /* PVN_PRINTOUT */
             );

    /* update U */
    if (copysign(1.0, st) != 1.0) {
      /* U^T(φ) * diag(1, -1) * U^T(ϑ):
          cos(φ - ϑ) -sin(φ - ϑ)
         -sin(φ - ϑ) -cos(φ - ϑ)
      */
      st = -st;
      double tf_t = (tf - tt), cf_t = 1.0, sf_t = 0.0;
      if (tf_t != 0.0) {
        tf_t /= fma(tf, tt, 1.0);
        if (isfinite(tf_t)) {
          /* 1 / cos */
          cf_t = hypot(tf_t, 1.0);
          sf_t = (tf_t / cf_t);
          cf_t = (1.0 / cf_t);
        }
        else {
          sf_t = copysign(1.0, tf_t);
          cf_t = 0.0;
        }
      }
      else
        sf_t = tf_t;
      const double _sf_t = -sf_t;
      A21 = *u11;
      *u11 = (_sf_t * *u21 + cf_t * *u11);
      *u21 = (_sf_t *  A21 - cf_t * *u21);
      A21 = *u12;
      *u12 = (_sf_t * *u22 + cf_t * *u12);
      *u22 = (_sf_t *  A21 - cf_t * *u22);
      A21 = -1.0;
    }
    else if (tt != 0.0) {
      /* U^T(φ) * U^T(ϑ) = U^T(φ + ϑ):
          cos(φ + ϑ)  sin(φ + ϑ)
         -sin(φ + ϑ)  cos(φ + ϑ)
      */
      double tft = (tf + tt), cft = 1.0, sft = 0.0;
      if (tft != 0.0) {
        tft /= fma(-tf, tt, 1.0);
        if (isfinite(tft)) {
          /* 1 / cos */
          cft = hypot(tft, 1.0);
          sft = (tft / cft);
          cft = (1.0 / cft);
        }
        else {
          sft = copysign(1.0, tft);
          cft = 0.0;
        }
      }
      else
        sft = tft;
      A21 = *u11;
      *u11 = (cft * *u11 + sft * *u21);
      *u21 = (cft * *u21 - sft *  A21);
      A21 = *u12;
      *u12 = (cft * *u12 + sft * *u22);
      *u22 = (cft * *u22 - sft *  A21);
      A21 = 1.0;
    }
    else if (tf != 0.0) {
      /* U^T(φ):
          cos(φ)  sin(φ)
         -sin(φ)  cos(φ)
       */
      A21 = *u11;
      *u11 = (cf * *u11 + sf * *u21);
      *u21 = (cf * *u21 - sf *  A21);
      A21 = *u12;
      *u12 = (cf * *u12 + sf * *u22);
      *u22 = (cf * *u22 - sf *  A21);
      A21 = -0.0;
    }
    else /* U^T(φ) = I */
      A21 = 0.0;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "U operation=%s\n", pvn_dtoa(s, A21));
#endif /* PVN_PRINTOUT */

    /* update V */
    if (tp != 0.0) {
      /* V(ψ):
         cos(ψ) -sin(ψ)
         sin(ψ)  cos(ψ)
      */
      A21 = *v11;
      *v11 = (*v11 * cp + *v12 * sp);
      *v12 = (*v12 * cp -  A21 * sp);
      A21 = *v21;
      *v21 = (*v21 * cp + *v22 * sp);
      *v22 = (*v22 * cp -  A21 * sp);
      A21 = -0.0;
    }
    else /* V(ψ) = I */
      A21 = 0.0;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "V operation=%s\n", pvn_dtoa(s, A21));
#endif /* PVN_PRINTOUT */

    if (e == -13) {
      *u21 = -*u21;
      *u22 = -*u22;
      *v12 = -*v12;
      *v22 = -*v22;
    }
  }

  if (ef_cmp(es[1], *s1, es[2], *s2) < 0) {
    pvn_dswp(u11, u21);
    pvn_dswp(u12, u22);
    pvn_dswp(v11, v12);
    pvn_dswp(v21, v22);
    pvn_dswp(s1, s2);
  }
  pvn_dswp(u21, u12);
  return knd;
}

static double cpolar(const double r, const double i, double *const c, double *const s)
{
  PVN_ASSERT(c);
  PVN_ASSERT(s);
  const double a = hypot(r, i);
  *c = copysign(fmin((fabs(r) / a), 1.0), r);
  *s = (i / fmax(a, DBL_TRUE_MIN));
  return a;
}

int PVN_FABI(pvn_zljsv2,PVN_ZLJSV2)
(const double *const a11r, const double *const a11i, const double *const a21r, const double *const a21i, const double *const a12r, const double *const a12i, const double *const a22r, const double *const a22i,
 double *const u11r, double *const u11i, double *const u21r, double *const u21i, double *const u12r, double *const u12i, double *const u22r, double *const u22i,
 double *const v11r, double *const v11i, double *const v21r, double *const v21i, double *const v12r, double *const v12i, double *const v22r, double *const v22i,
 double *const s1, double *const s2, int *const es)
{
  PVN_ASSERT(a11r); PVN_ASSERT(a11i);
  PVN_ASSERT(a21r); PVN_ASSERT(a21i);
  PVN_ASSERT(a12r); PVN_ASSERT(a12i);
  PVN_ASSERT(a22r); PVN_ASSERT(a22i);
  PVN_ASSERT(u11r); PVN_ASSERT(u11i);
  PVN_ASSERT(u21r); PVN_ASSERT(u21i);
  PVN_ASSERT(u12r); PVN_ASSERT(u12i);
  PVN_ASSERT(u22r); PVN_ASSERT(u22i);
  PVN_ASSERT(v11r); PVN_ASSERT(v11i);
  PVN_ASSERT(v21r); PVN_ASSERT(v21i);
  PVN_ASSERT(v12r); PVN_ASSERT(v12i);
  PVN_ASSERT(v22r); PVN_ASSERT(v22i);
  PVN_ASSERT(s1);
  PVN_ASSERT(s2);
  PVN_ASSERT(es);
  if (!isfinite(*a11r))
    return -1;
  if (!isfinite(*a11i))
    return -2;
  if (!isfinite(*a21r))
    return -3;
  if (!isfinite(*a21i))
    return -4;
  if (!isfinite(*a12r))
    return -5;
  if (!isfinite(*a12i))
    return -6;
  if (!isfinite(*a22r))
    return -7;
  if (!isfinite(*a22i))
    return -8;
  /* should not fail beyond this point when *es == 0 */
  int kndi = 0, mxe = INT_MIN, e = 0;
  if (*a11i != 0.0) {
    kndi |= 1;
    (void)frexp(*a11i, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a21i != 0.0) {
    kndi |= 2;
    (void)frexp(*a21i, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a12i != 0.0) {
    kndi |= 4;
    (void)frexp(*a12i, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a22i != 0.0) {
    kndi |= 8;
    (void)frexp(*a22i, &e);
    if (mxe < e)
      mxe = e;
  }
  *u11i = 0.0;
  *u21i = 0.0;
  *u12i = 0.0;
  *u22i = 0.0;
  *v11i = 0.0;
  *v21i = 0.0;
  *v12i = 0.0;
  *v22i = 0.0;
  if (!kndi)
    return PVN_FABI(pvn_dljsv2,PVN_DLJSV2)(a11r, a21r, a12r, a22r, u11r, u21r, u12r, u22r, v11r, v21r, v12r, v22r, s1, s2, es);
  int kndr = 0;
  if (*a11r != 0.0) {
    kndr |= 1;
    (void)frexp(*a11r, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a21r != 0.0) {
    kndr |= 2;
    (void)frexp(*a21r, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a12r != 0.0) {
    kndr |= 4;
    (void)frexp(*a12r, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a22r != 0.0) {
    kndr |= 8;
    (void)frexp(*a22r, &e);
    if (mxe < e)
      mxe = e;
  }
  if (!kndr) {
    *u11r = 0.0;
    *u21r = 0.0;
    *u12r = 0.0;
    *u22r = 0.0;
    return PVN_FABI(pvn_dljsv2,PVN_DLJSV2)(a11i, a21i, a12i, a22i, u11i, u21i, u12i, u22i, v11r, v21r, v12r, v22r, s1, s2, es);
  }
  int knd = (kndr | kndi);
  switch (knd) {
  case  1:
  case  2:
  case  4:
  case  6:
  case  8:
  case  9:
    e = (DBL_MAX_EXP - mxe - 1);
    break;
  case  3:
  case  5:
  case  7:
  case 10:
  case 11:
  case 12:
  case 13:
  case 14:
  case 15:
    e = (DBL_MAX_EXP - mxe - 2);
    break;
  default:
    return INT_MIN;
  }
  mxe = *es;
  if (!*es)
    *es = e;
  else if (*es < 0)
    ++*es;
  es[2] = es[1] = 0;

  /* scaling by 2^(*es), can only fail when mxe != 0 */
  double A11r = *a11r, A11i = *a11i, A21r = *a21r, A21i = *a21i, A12r = *a12r, A12i = *a12i, A22r = *a22r, A22i = *a22i;
  if (*es) {
    A11r = scalbn(A11r, *es);
    A11i = scalbn(A11i, *es);
    A21r = scalbn(A21r, *es);
    A21i = scalbn(A21i, *es);
    A12r = scalbn(A12r, *es);
    A12i = scalbn(A12i, *es);
    A22r = scalbn(A22r, *es);
    A22i = scalbn(A22i, *es);
    if (mxe) {
      if (!isfinite(A11r))
        return -9;
      if (!isfinite(A11i))
        return -10;
      if (!isfinite(A21r))
        return -11;
      if (!isfinite(A21i))
        return -12;
      if (!isfinite(A12r))
        return -13;
      if (!isfinite(A12i))
        return -14;
      if (!isfinite(A22r))
        return -15;
      if (!isfinite(A22i))
        return -16;
    }
    if (*es < 0) {
      kndr = 0;
      if (A11r != 0.0)
        kndr |= 1;
      if (A21r != 0.0)
        kndr |= 2;
      if (A12r != 0.0)
        kndr |= 4;
      if (A22r != 0.0)
        kndr |= 8;
      kndi = 0;
      if (A11i != 0.0)
        kndi |= 1;
      if (A21i != 0.0)
        kndi |= 2;
      if (A12i != 0.0)
        kndi |= 4;
      if (A22i != 0.0)
        kndi |= 8;
      knd = (kndr | kndi);
    }
  }

  *u11r = 1.0;
  *u21r = 0.0;
  *u12r = 0.0;
  *u22r = 1.0;
  *v11r = 1.0;
  *v21r = 0.0;
  *v12r = 0.0;
  *v22r = 1.0;
  *s1 = 0.0;
  *s2 = 0.0;

  /* simplify the form of A */
  switch (knd) {
  case  0:
  case  1:
  case  8:
  case  9:
    /* [ ? 0 ] */
    /* [ 0 ? ] */
    *s1 = cpolar(A11r, A11i, u11r, u11i); *u11i = -*u11i;
    A11r = *s1; A11i = 0.0;
    *s2 = cpolar(A22r, A22i, u22r, u22i); *u22i = -*u22i;
    A22r = *s2; A22i = 0.0;
    A12i = A12r = A21i = A21r = 0.0;
    e = 0;
    break;
  case  2:
  case  4:
  case  6:
    /* [ 0 ? ] */
    /* [ ? 0 ] */
    *u11r = 0.0;
    *u22r = 0.0;
    A11r = A21r; A11i = A21i;
    A22r = A12r; A22i = A12i;
    *s1 = cpolar(A11r, A11i, u12r, u12i); *u12i = -*u12i;
    A11r = *s1; A11i = 0.0;
    *s2 = cpolar(A22r, A22i, u21r, u21i); *u21i = -*u21i;
    A22r = *s2; A22i = 0.0;
    A12i = A12r = A21i = A21r = 0.0;
    e = 0;
    break;
  case 12:
    /* [ 0 * ] */
    /* [ 0 * ] */
    *v11r = 0.0;
    *v21r = 1.0;
    *v12r = 1.0;
    *v22r = 0.0;
    A11r = A12r; A11i = A12i;
    A21r = A22r; A21i = A22i;
    /* FALLTHRU */
  case  3:
    /* [ * 0 ] */
    /* [ * 0 ] */
    *s1 = cpolar(A11r, A11i, u11r, u11i); *u11i = -*u11i;
    A11r = *s1; A11i = 0.0;
    *s2 = cpolar(A21r, A21i, u22r, u22i); *u22i = -*u22i;
    A21r = *s2; A21i = 0.0;
    if (A11r < A21r) {
      pvn_dswp(u11r, u21r);
      pvn_dswp(u11i, u21i);
      pvn_dswp(u22r, u12r);
      pvn_dswp(u22i, u12i);
      pvn_dswp(&A11r, &A21r);
    }
    A22i = A22r = A12i = A12r = 0.0;
    *s1 = hypot(A11r, A21r);
    *s2 = 0.0;
    e = 3;
    break;
  case 10:
    /* [ 0 0 ] */
    /* [ * * ] */
    *u11r = 0.0;
    *u21r = 1.0;
    *u12r = 1.0;
    *u22r = 0.0;
    A11r = A21r; A11i = A21i;
    A12r = A22r; A12i = A22i;
    /* FALLTHRU */
  case  5:
    /* [ * * ] */
    /* [ 0 0 ] */
    *s1 = cpolar(A11r, A11i, v11r, v11i); *v11i = -*v11i;
    A11r = *s1; A11i = 0.0;
    *s2 = cpolar(A12r, A12i, v22r, v22i); *v22i = -*v22i;
    A12r = *s2; A12i = 0.0;
    if (A11r < A12r) {
      pvn_dswp(&A11r, &A12r);
      pvn_dswp(v11r, v12r);
      pvn_dswp(v11i, v12i);
      pvn_dswp(v22r, v21r);
      pvn_dswp(v22i, v21i);
    }
    A22i = A22r = A21i = A21r = 0.0;
    *s1 = hypot(A11r, A12r);
    *s2 = 0.0;
    e = 5;
    break;
  case  7:
    /* [ * * ] */
    /* [ * 0 ] */
    pvn_dswp(&A11r, &A12r);
    pvn_dswp(&A11i, &A12i);
    A22r = A21r; A22i = A21i;
    *v11r = 0.0;
    *v22r = 0.0;
    *s1 = cpolar(A11r, A11i, v21r, v21i); *v21i = -*v21i;
    A11r = *s1; A11i = 0.0;
    *s1 = cpolar(A12r, A12i, v12r, v12i); *v12i = -*v12i;
    A12r = *s1; A12i = 0.0;
    pvn_zmul(&A21r, &A21i, A22r, A22i, *v12r, *v12i);
    A22r = A21r; A22i = A21i;
    *s2 = cpolar(A22r, A22i, u22r, u22i); *u22i = -*u22i;
    A22r = *s2; A22i = 0.0;
    *s2 = *s1 = A21i = A21r = 0.0;
    e = 13;
    break;
  case 11:
    /* [ * 0 ] */
    /* [ * * ] */
    *u11r = 0.0;
    *u12r = 1.0;
    *u22r = 0.0;
    pvn_dswp(&A11r, &A22r);
    pvn_dswp(&A11i, &A22i);
    A12r = A21r; A12i = A21i;
    *v11r = 0.0;
    *v22r = 0.0;
    *s1 = cpolar(A11r, A11i, v21r, v21i); *v21i = -*v21i;
    A11r = *s1; A11i = 0.0;
    *s1 = cpolar(A12r, A12i, v12r, v12i); *v12i = -*v12i;
    A12r = *s1; A12i = 0.0;
    pvn_zmul(&A21r, &A21i, A22r, A22i, *v12r, *v12i);
    A22r = A21r; A22i = A21i;
    *s2 = cpolar(A22r, A22i, u21r, u21i); *u21i = -*u21i;
    A22r = *s2; A22i = 0.0;
    *s2 = *s1 = A21i = A21r = 0.0;
    e = 13;
    break;
  case 13:
    /* [ * * ] */
    /* [ 0 * ] */
    *s1 = cpolar(A11r, A11i, v11r, v11i); *v11i = -*v11i;
    A11r = *s1; A11i = 0.0;
    *s1 = cpolar(A12r, A12i, v22r, v22i); *v22i = -*v22i;
    A12r = *s1; A12i = 0.0;
    pvn_zmul(&A21r, &A21i, A22r, A22i, *v22r, *v22i);
    A22r = A21r; A22i = A21i;
    *s2 = cpolar(A22r, A22i, u22r, u22i); *u22i = -*u22i;
    A22r = *s2; A22i = 0.0;
    *s2 = *s1 = A21i = A21r = 0.0;
    e = 13;
    break;
  case 14:
    /* [ 0 * ] */
    /* [ * * ] */
    *u11r = 0.0;
    *u12r = 1.0;
    *u22r = 0.0;
    A11r = A21r; A11i = A21i;
    pvn_dswp(&A12r, &A22r);
    pvn_dswp(&A12i, &A22i);
    *s1 = cpolar(A11r, A11i, v11r, v11i); *v11i = -*v11i;
    A11r = *s1; A11i = 0.0;
    *s1 = cpolar(A12r, A12i, v22r, v22i); *v22i = -*v22i;
    A12r = *s1; A12i = 0.0;
    pvn_zmul(&A21r, &A21i, A22r, A22i, *v22r, *v22i);
    A22r = A21r; A22i = A21i;
    *s2 = cpolar(A22r, A22i, u21r, u21i); *u21i = -*u21i;
    A22r = *s2; A22i = 0.0;
    *s2 = *s1 = A21i = A21r = 0.0;
    e = 13;
    break;
  case 15:
    /* [ * * ] */
    /* [ * * ] */
    e = 15;
    break;
  default:
    return INT_MIN;
  }

  if ((e == 13) && (A11r < A22r))
    e = -13;

#ifdef PVN_PRINTOUT
  char s[26] = { '\0' };
#endif /* PVN_PRINTOUT */
  double tt = 0.0, ct = 1.0, st = 0.0;

  if (e == 15) {
    double B11, B21, B12, B22, C11, C21, C12, C22, S11, S21, S12, S22;
    B11 = cpolar(A11r, A11i, &C11, &S11); S11 = -S11;
    B21 = cpolar(A21r, A21i, &C21, &S21); S21 = -S21;
    B12 = cpolar(A12r, A12i, &C12, &S12); S12 = -S12;
    B22 = cpolar(A22r, A22i, &C22, &S22); S22 = -S22;
    *s1 = hypot(B11, B21);
    *s2 = hypot(B12, B22);
    if (*s1 < *s2) {
      pvn_dswp(&A11r, &A12r);
      pvn_dswp(&A11i, &A12i);
      pvn_dswp(&A21r, &A22r);
      pvn_dswp(&A21i, &A22i);
      pvn_dswp(&B11, &B12);
      pvn_dswp(&B21, &B22);
      pvn_dswp(&C11, &C12);
      pvn_dswp(&C21, &C22);
      pvn_dswp(&S11, &S12);
      pvn_dswp(&S21, &S22);
      pvn_dswp(s1, s2);
      pvn_dswp(v11r, v12r);
      pvn_dswp(v11i, v12i);
      pvn_dswp(v21r, v22r);
      pvn_dswp(v21i, v22i);
    }
    if (B11 < B21) {
      pvn_dswp(u11r, u21r);
      pvn_dswp(u11i, u21i);
      pvn_dswp(u12r, u22r);
      pvn_dswp(u12i, u22i);
      pvn_dswp(&A11r, &A21r);
      pvn_dswp(&A11i, &A21i);
      pvn_dswp(&A12r, &A22r);
      pvn_dswp(&A12i, &A22i);
      pvn_dswp(&B11, &B21);
      pvn_dswp(&B12, &B22);
      pvn_dswp(&C11, &C21);
      pvn_dswp(&C12, &C22);
      pvn_dswp(&S11, &S21);
      pvn_dswp(&S12, &S22);
    }
    pvn_zmul(&ct, &st, C11, S11, *u11r, *u11i);
    *u11r = ct; *u11i = st;
    pvn_zmul(&ct, &st, C11, S11, *u12r, *u12i);
    *u12r = ct; *u12i = st;
    A11r = B11; A11i = 0.0;
    pvn_zmul(&ct, &st, C11, S11, A12r, A12i);
    A12r = ct; A12i = st;
    pvn_zmul(&ct, &st, C21, S21, *u21r, *u21i);
    *u21r = ct; *u21i = st;
    pvn_zmul(&ct, &st, C21, S21, *u22r, *u22i);
    *u22r = ct; *u22i = st;
    A21r = B21; A21i = 0.0;
    pvn_zmul(&ct, &st, C21, S21, A22r, A22i);
    A22r = ct; A22i = st;
    /* rotate */
    tt = (A21r / A11r);
    /* 1 / cos */
    ct = hypot(tt, 1.0);
    /* apply the left Givens rotation to A and U */
    st = -tt;
    A21r = A12r;
    A21i = A12i;
    if (ct == 1.0) {
      A12r = fma(tt, A22r, A12r);
      A12i = fma(tt, A22i, A12i);
      A22r = fma(st, A21r, A22r);
      A22i = fma(st, A21i, A22i);
      A21r = *u11r;
      A21i = *u11i;
      *u11r = fma(tt, *u21r, *u11r);
      *u11i = fma(tt, *u21i, *u11i);
      *u21r = fma(st,  A21r, *u21r);
      *u21i = fma(st,  A21i, *u21i);
      A21r = *u12r;
      A21i = *u12i;
      *u12r = fma(tt, *u22r, *u12r);
      *u12i = fma(tt, *u22i, *u12i);
      *u22r = fma(st,  A21r, *u22r);
      *u22i = fma(st,  A21i, *u22i);
      st = tt;
    }
    else {
      A12r = (fma(tt, A22r, A12r) / ct);
      A12i = (fma(tt, A22i, A12i) / ct);
      A22r = (fma(st, A21r, A22r) / ct);
      A22i = (fma(st, A21i, A22i) / ct);
      A21r = *u11r;
      A21i = *u11i;
      *u11r = (fma(tt, *u21r, *u11r) / ct);
      *u11i = (fma(tt, *u21i, *u11i) / ct);
      *u21r = (fma(st,  A21r, *u21r) / ct);
      *u21i = (fma(st,  A21i, *u21i) / ct);
      A21r = *u12r;
      A21i = *u12i;
      *u12r = (fma(tt, *u22r, *u12r) / ct);
      *u12i = (fma(tt, *u22i, *u12i) / ct);
      *u22r = (fma(st,  A21r, *u22r) / ct);
      *u22i = (fma(st,  A21i, *u22i) / ct);
      st = (tt / ct);
      ct = (1.0 / ct);
    }
    A11r = *s1;
    B12 = cpolar(A12r, A12i, &C12, &S12); S12 = -S12;
    A12r = B12; A12i = 0.0;
    pvn_zmul(&A21r, &A21i, A22r, A22i, C12, S12);
    A22r = A21r; A22i = A21i;
    pvn_zmul(&A21r, &A21i, *v12r, *v12i, C12, S12);
    *v12r = A21r; *v12i = A21i;
    pvn_zmul(&A21r, &A21i, *v22r, *v22i, C12, S12);
    *v22r = A21r; *v22i = A21i;
    B22 = cpolar(A22r, A22i, &C22, &S22); S22 = -S22;
    A22r = B22; A22i = 0.0;
    pvn_zmul(&A21r, &A21i, C22, S22, *u21r, *u21i);
    *u21r = A21r; *u21i = A21i;
    pvn_zmul(&A21r, &A21i, C22, S22, *u22r, *u22i);
    *u22r = A21r; *u22i = A21i;
    A21i = A21r = 0.0;
    if (A12r == 0.0)
      e = 0;
    else if (A22r == 0.0) {
      *s1 = hypot(*s1, *s2);
      *s2 = 0.0;
      e = 5;
    }
    else if (A11r < A22r)
      e = -13;
    else
      e = 13;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(ϑ)=%s, ", pvn_dtoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(ϑ)=%s, ", pvn_dtoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(ϑ)=%s\n", pvn_dtoa(s, st));
#endif /* PVN_PRINTOUT */
  }

  if (e == 3) {
    tt = (A21r / A11r);
    /* 1 / cos */
    ct = hypot(tt, 1.0);
    /* apply the left Givens rotation to U */
    st = -tt;
    A21r = *u11r;
    A21i = *u11i;
    if (ct == 1.0) {
      *u11r = fma(tt, *u21r, *u11r);
      *u11i = fma(tt, *u21i, *u11i);
      *u21r = fma(st,  A21r, *u21r);
      *u21i = fma(st,  A21i, *u21i);
      A21r = *u12r;
      A21i = *u12i;
      *u12r = fma(tt, *u22r, *u12r);
      *u12i = fma(tt, *u22i, *u12i);
      *u22r = fma(st,  A21r, *u22r);
      *u22i = fma(st,  A21i, *u22i);
      st = tt;
    }
    else {
      *u11r = (fma(tt, *u21r, *u11r) / ct);
      *u11i = (fma(tt, *u21i, *u11i) / ct);
      *u21r = (fma(st,  A21r, *u21r) / ct);
      *u21i = (fma(st,  A21i, *u21i) / ct);
      A21r = *u12r;
      A21i = *u12i;
      *u12r = (fma(tt, *u22r, *u12r) / ct);
      *u12i = (fma(tt, *u22i, *u12i) / ct);
      *u22r = (fma(st,  A21r, *u22r) / ct);
      *u22i = (fma(st,  A21i, *u22i) / ct);
      st = (tt / ct);
      ct = (1.0 / ct);
    }
    A11r = *s1;
    A21i = A21r = 0.0;
    e = 0;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(ϑ)=%s, ", pvn_dtoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(ϑ)=%s, ", pvn_dtoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(ϑ)=%s\n", pvn_dtoa(s, st));
#endif /* PVN_PRINTOUT */
  }

  if (e == 5) {
    tt = (A12r / A11r);
    /* 1 / cos */
    ct = hypot(tt, 1.0);
    /* apply the right Givens rotation to V */
    st = -tt;
    A12r = *v11r;
    A12i = *v11i;
    if (ct == 1.0) {
      *v11r = fma(tt, *v12r, *v11r);
      *v11i = fma(tt, *v12i, *v11i);
      *v12r = fma(st,  A12r, *v12r);
      *v12i = fma(st,  A12i, *v12i);
      A12r = *v21r;
      A12i = *v21i;
      *v21r = fma(tt, *v22r, *v21r);
      *v21i = fma(tt, *v22i, *v21i);
      *v22r = fma(st,  A12r, *v22r);
      *v22i = fma(st,  A12i, *v22i);
      st = tt;
    }
    else {
      *v11r = (fma(tt, *v12r, *v11r) / ct);
      *v11i = (fma(tt, *v12i, *v11i) / ct);
      *v12r = (fma(st,  A12r, *v12r) / ct);
      *v12i = (fma(st,  A12i, *v12i) / ct);
      A12r = *v21r;
      A12i = *v21i;
      *v21r = (fma(tt, *v22r, *v21r) / ct);
      *v21i = (fma(tt, *v22i, *v21i) / ct);
      *v22r = (fma(st,  A12r, *v22r) / ct);
      *v22i = (fma(st,  A12i, *v22i) / ct);
      st = (tt / ct);
      ct = (1.0 / ct);
    }
    A11r = *s1;
    A12i = A12r = 0.0;
    e = 0;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(θ)=%s, ", pvn_dtoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(θ)=%s, ", pvn_dtoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(θ)=%s\n", pvn_dtoa(s, st));
#endif /* PVN_PRINTOUT */
  }

#ifdef PVN_PRINTOUT
  (void)dprintf(PVN_PRINTOUT, "\tA =\n");
  (void)dprintf(PVN_PRINTOUT, "(%s,", pvn_dtoa(s, scalbn(A11r, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s) ", pvn_dtoa(s, scalbn(A11i, -*es)));
  (void)dprintf(PVN_PRINTOUT, "(%s,", pvn_dtoa(s, scalbn(A12r, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s)\n", pvn_dtoa(s, scalbn(A12i, -*es)));
  (void)dprintf(PVN_PRINTOUT, "(%s,", pvn_dtoa(s, scalbn(A21r, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s) ", pvn_dtoa(s, scalbn(A21i, -*es)));
  (void)dprintf(PVN_PRINTOUT, "(%s,", pvn_dtoa(s, scalbn(A22r, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s)\n", pvn_dtoa(s, scalbn(A22i, -*es)));
#endif /* PVN_PRINTOUT */

  if (abs(e) == 13) {
    double tf = 0.0, cf = 1.0, sf = 0.0, tp = 0.0, cp = 1.0, sp = 0.0;
    if (e == -13) {
      double tf_ = 0.0, cf_ = 1.0, sf_ = 0.0, tp_ = 0.0, cp_ = 1.0, sp_ = 0.0;
      dlpsv2(A22r, A12r, A11r, &tf_, &cf_, &sf_, &tp_, &cp_, &sp_, s1, s2, es
#ifdef PVN_PRINTOUT
             , s
#endif /* PVN_PRINTOUT */
             );
      tf = (1.0 / tp_);
      cf = sp_;
      sf = cp_;
      tp = (1.0 / tf_);
      cp = sf_;
      sp = cf_;
    }
    else
      dlpsv2(A11r, A12r, A22r, &tf, &cf, &sf, &tp, &cp, &sp, s1, s2, es
#ifdef PVN_PRINTOUT
             , s
#endif /* PVN_PRINTOUT */
             );

    if (tf != 0.0) {
      A21r = *u11r; A21i = *u11i;
      *u11r = (cf * *u11r + sf * *u21r);
      *u11i = (cf * *u11i + sf * *u21i);
      *u21r = (cf * *u21r - sf *  A21r);
      *u21i = (cf * *u21i - sf *  A21i);
      A21r = *u12r; A21i = *u12i;
      *u12r = (cf * *u12r + sf * *u22r);
      *u12i = (cf * *u12i + sf * *u22i);
      *u22r = (cf * *u22r - sf *  A21r);
      *u22i = (cf * *u22i - sf *  A21i);
      A21r = -0.0;
      A21i = ((e == 13) ? 0.0 : -0.0);
    }
    else {
      A21r = 0.0;
      A21i = ((e == 13) ? 0.0 : -0.0);
    }

    if (tp != 0.0) {
      A21r = *v11r; A21i = *v11i;
      *v11r = (*v11r * cp + *v12r * sp);
      *v11i = (*v11i * cp + *v12i * sp);
      *v12r = (*v12r * cp -  A21r * sp);
      *v12i = (*v12i * cp -  A21i * sp);
      A21r = *v21r; A21i = *v21i;
      *v21r = (*v21r * cp + *v22r * sp);
      *v21i = (*v21i * cp + *v22i * sp);
      *v22r = (*v22r * cp -  A21r * sp);
      *v22i = (*v22i * cp -  A21i * sp);
      A21r = -0.0;
      A21i = ((e == 13) ? 0.0 : -0.0);
    }
    else {
      A21r = 0.0;
      A21i = ((e == 13) ? 0.0 : -0.0);
    }

    if (e == -13) {
      *u21r = -*u21r;
      *u21i = -*u21i;
      *u22r = -*u22r;
      *u22i = -*u22i;
      *v12r = -*v12r;
      *v12i = -*v12i;
      *v22r = -*v22r;
      *v22i = -*v22i;
    }
  }

  if (ef_cmp(es[1], *s1, es[2], *s2) < 0) {
    pvn_dswp(u11r, u21r);
    pvn_dswp(u11i, u21i);
    pvn_dswp(u12r, u22r);
    pvn_dswp(u12i, u22i);
    pvn_dswp(v11r, v12r);
    pvn_dswp(v11i, v12i);
    pvn_dswp(v21r, v22r);
    pvn_dswp(v21i, v22i);
    pvn_dswp(s1, s2);
  }
  pvn_dswp(u21r, u12r);
  pvn_dswp(u21i, u12i);
  *u11i = -*u11i;
  *u21i = -*u21i;
  *u12i = -*u12i;
  *u22i = -*u22i;
  return knd;
}

#ifndef XLJR2
#define XLJR2                                                   \
  PVN_ASSERT(a11);                                              \
  PVN_ASSERT(a21);                                              \
  PVN_ASSERT(a12);                                              \
  PVN_ASSERT(a22);                                              \
  PVN_ASSERT(u11);                                              \
  PVN_ASSERT(u21);                                              \
  PVN_ASSERT(u12);                                              \
  PVN_ASSERT(u22);                                              \
  PVN_ASSERT(v11);                                              \
  PVN_ASSERT(v21);                                              \
  PVN_ASSERT(v12);                                              \
  PVN_ASSERT(v22);                                              \
  PVN_ASSERT(s1);                                               \
  PVN_ASSERT(s2);                                               \
  PVN_ASSERT(es);                                               \
  PVN_ASSERT(E);                                                \
  long double                                                   \
    u11l = *u11, u21l = *u21, u12l = *u12, u22l = *u22,         \
    v11l = *v11, v21l = *v21, v12l = *v12, v22l = *v22,         \
    s1l = scalbnl(*s1, es[1] - es[0]),                          \
    s2l = scalbnl(*s2, es[2] - es[0]);                          \
  /* cond_2(G) */                                               \
  E[0] = fminl((s1l / s2l), __builtin_infl());                  \
  /* U^T U - I */                                               \
  long double T11 = fmal(u11l, u11l, fmal(u21l, u21l, -1.0L));  \
  long double T21 = fmal(u12l, u11l, u22l * u21l);              \
  long double T12 = T21;                                        \
  long double T22 = fmal(u12l, u12l, fmal(u22l, u22l, -1.0L));  \
  E[1] = hypotl(hypotl(T11, T21), hypotl(T12, T22));            \
  /* V^T V - I */                                               \
  T11 = fmal(v11l, v11l, fmal(v21l, v21l, -1.0L));              \
  T21 = fmal(v12l, v11l, v22l * v21l);                          \
  T12 = T21;                                                    \
  T22 = fmal(v12l, v12l, fmal(v22l, v22l, -1.0L));              \
  E[2] = hypotl(hypotl(T11, T21), hypotl(T12, T22));            \
  /* U Σ V^T - G */                                             \
  u11l *= s1l;                                                  \
  u21l *= s1l;                                                  \
  u12l *= s2l;                                                  \
  u22l *= s2l;                                                  \
  E[3] = hypotl(hypotl(*a11, *a21), hypotl(*a12, *a22));        \
  T11 = fmal(u11l, v11l, fmal(u12l, v12l, -*a11));              \
  T21 = fmal(u21l, v11l, fmal(u22l, v12l, -*a21));              \
  T12 = fmal(u11l, v21l, fmal(u12l, v22l, -*a12));              \
  T22 = fmal(u21l, v21l, fmal(u22l, v22l, -*a22));              \
  E[3] = (hypotl(hypotl(T11, T21), hypotl(T12, T22)) / E[3])
#else /* XLJR2 */
#error XLJR2 already defined
#endif /* ?XLJR2 */

void PVN_FABI(pvn_sxljr2,PVN_SXLJR2)
(const float *const a11, const float *const a21, const float *const a12, const float *const a22,
 const float *const u11, const float *const u21, const float *const u12, const float *const u22,
 const float *const v11, const float *const v21, const float *const v12, const float *const v22,
 const float *const s1, const float *const s2, const int *const es, long double *const E)
{
  XLJR2;
}

void PVN_FABI(pvn_dxljr2,PVN_DXLJR2)
(const double *const a11, const double *const a21, const double *const a12, const double *const a22,
 const double *const u11, const double *const u21, const double *const u12, const double *const u22,
 const double *const v11, const double *const v21, const double *const v12, const double *const v22,
 const double *const s1, const double *const s2, const int *const es, long double *const E)
{
  XLJR2;
}

#ifndef WLJR2
#define WLJR2                                                        \
  PVN_ASSERT(a11r); PVN_ASSERT(a11i);                                \
  PVN_ASSERT(a21r); PVN_ASSERT(a21i);                                \
  PVN_ASSERT(a12r); PVN_ASSERT(a12i);                                \
  PVN_ASSERT(a22r); PVN_ASSERT(a22i);                                \
  PVN_ASSERT(u11r); PVN_ASSERT(u11i);                                \
  PVN_ASSERT(u21r); PVN_ASSERT(u21i);                                \
  PVN_ASSERT(u12r); PVN_ASSERT(u12i);                                \
  PVN_ASSERT(u22r); PVN_ASSERT(u22i);                                \
  PVN_ASSERT(v11r); PVN_ASSERT(v11i);                                \
  PVN_ASSERT(v21r); PVN_ASSERT(v21i);                                \
  PVN_ASSERT(v12r); PVN_ASSERT(v12i);                                \
  PVN_ASSERT(v22r); PVN_ASSERT(v22i);                                \
  PVN_ASSERT(s1);                                                    \
  PVN_ASSERT(s2);                                                    \
  PVN_ASSERT(es);                                                    \
  PVN_ASSERT(E);                                                     \
  long double                                                        \
    U11r = *u11r, U11i = *u11i, U21r = *u21r, U21i = *u21i,          \
    U12r = *u12r, U12i = *u12i, U22r = *u22r, U22i = *u22i,          \
    V11r = *v11r, V11i = *v11i, V21r = *v21r, V21i = *v21i,          \
    V12r = *v12r, V12i = *v12i, V22r = *v22r, V22i = *v22i,          \
    S1 = scalbnl(*s1, es[1] - es[0]),                                \
    S2 = scalbnl(*s2, es[2] - es[0]);                                \
  /* cond_2(G) */                                                    \
  E[0] = fminl((S1 / S2), __builtin_infl());                         \
  /* U^H U - I */                                                    \
  long double T11r = -1.0L, T11i = 0.0L;                             \
  pvn_wfma(&T11r, &T11i, U21r, -U21i, U21r, U21i, T11r, T11i);       \
  pvn_wfma(&T11r, &T11i, U11r, -U11i, U11r, U11i, T11r, T11i);       \
  long double T21r = 0.0L, T21i = 0.0L;                              \
  pvn_wmul(&T21r, &T21i, U22r, -U22i, U21r, U21i);                   \
  pvn_wfma(&T21r, &T21i, U12r, -U12i, U11r, U11i, T21r, T21i);       \
  long double T12r = 0.0L, T12i = 0.0L;                              \
  pvn_wmul(&T12r, &T12i, U21r, -U21i, U22r, U22i);                   \
  pvn_wfma(&T12r, &T12i, U11r, -U11i, U12r, U12i, T12r, T12i);       \
  long double T22r = -1.0L, T22i = 0.0L;                             \
  pvn_wfma(&T22r, &T22i, U22r, -U22i, U22r, U22i, T22r, T22i);       \
  pvn_wfma(&T22r, &T22i, U12r, -U12i, U12r, U12i, T22r, T22i);       \
  E[1] = hypotl(hypotl(hypotl(T11r, T11i), hypotl(T21r, T21i)),      \
                hypotl(hypotl(T12r, T12i), hypotl(T22r, T22i)));     \
  /* V^H V -I */                                                     \
  pvn_wfma(&T11r, &T11i, V21r, -V21i, V21r, V21i, -1.0L, 0.0L);      \
  pvn_wfma(&T11r, &T11i, V11r, -V11i, V11r, V11i, T11r, T11i);       \
  pvn_wmul(&T21r, &T21i, V22r, -V22i, V21r, V21i);                   \
  pvn_wfma(&T21r, &T21i, V12r, -V12i, V11r, V11i, T21r, T21i);       \
  pvn_wmul(&T12r, &T12i, V21r, -V21i, V22r, V22i);                   \
  pvn_wfma(&T12r, &T12i, V11r, -V11i, V12r, V12i, T12r, T12i);       \
  pvn_wfma(&T22r, &T22i, V22r, -V22i, V22r, V22i, -1.0L, 0.0L);      \
  pvn_wfma(&T22r, &T22i, V12r, -V12i, V12r, V12i, T22r, T22i);       \
  E[2] = hypotl(hypotl(hypotl(T11r, T11i), hypotl(T21r, T21i)),      \
                hypotl(hypotl(T12r, T12i), hypotl(T22r, T22i)));     \
  /* U Σ V^H - G */                                                  \
  U11r *= S1; U11i *= S1;                                            \
  U21r *= S1; U21i *= S1;                                            \
  U12r *= S2; U12i *= S2;                                            \
  U22r *= S2; U22i *= S2;                                            \
  V11i = -V11i; V21i = -V21i; V12i = -V12i; V22i = -V22i;            \
  E[3] = hypotl(hypotl(hypotl(*a11r, *a11i), hypotl(*a21r, *a21i)),  \
                hypotl(hypotl(*a12r, *a12i), hypotl(*a22r, *a22i))); \
  pvn_wfma(&T11r, &T11i, U12r, U12i, V12r, V12i, -*a11r, -*a11i);    \
  pvn_wfma(&T11r, &T11i, U11r, U11i, V11r, V11i, T11r, T11i);        \
  pvn_wfma(&T21r, &T21i, U22r, U22i, V12r, V12i, -*a21r, -*a21i);    \
  pvn_wfma(&T21r, &T21i, U21r, U21i, V11r, V11i, T21r, T21i);        \
  pvn_wfma(&T12r, &T12i, U12r, U12i, V22r, V22i, -*a12r, -*a12i);    \
  pvn_wfma(&T12r, &T12i, U11r, U11i, V21r, V21i, T12r, T12i);        \
  pvn_wfma(&T22r, &T22i, U22r, U22i, V22r, V22i, -*a22r, -*a22i);    \
  pvn_wfma(&T22r, &T22i, U21r, U21i, V21r, V21i, T22r, T22i);        \
  E[3] = (hypotl(hypotl(hypotl(T11r, T11i), hypotl(T21r, T21i)),     \
                 hypotl(hypotl(T12r, T12i), hypotl(T22r, T22i))) / E[3])
#else /* WLJR2 */
#error WLJR2 already defined
#endif /* ?WLJR2 */

void PVN_FABI(pvn_cwljr2,PVN_CWLJR2)
(const float *const a11r, const float *const a11i, const float *const a21r, const float *const a21i,
 const float *const a12r, const float *const a12i, const float *const a22r, const float *const a22i,
 const float *const u11r, const float *const u11i, const float *const u21r, const float *const u21i,
 const float *const u12r, const float *const u12i, const float *const u22r, const float *const u22i,
 const float *const v11r, const float *const v11i, const float *const v21r, const float *const v21i,
 const float *const v12r, const float *const v12i, const float *const v22r, const float *const v22i,
 const float *const s1, const float *const s2, const int *const es, long double *const E)
{
  WLJR2;
}

void PVN_FABI(pvn_zwljr2,PVN_ZWLJR2)
(const double *const a11r, const double *const a11i, const double *const a21r, const double *const a21i,
 const double *const a12r, const double *const a12i, const double *const a22r, const double *const a22i,
 const double *const u11r, const double *const u11i, const double *const u21r, const double *const u21i,
 const double *const u12r, const double *const u12i, const double *const u22r, const double *const u22i,
 const double *const v11r, const double *const v11i, const double *const v21r, const double *const v21i,
 const double *const v12r, const double *const v12i, const double *const v22r, const double *const v22i,
 const double *const s1, const double *const s2, const int *const es, long double *const E)
{
  WLJR2;
}

static inline void ef_mull(int *const e, long double *const f, const int e1, const long double f1, const int e2, const long double f2)
{
  PVN_ASSERT(e);
  PVN_ASSERT(f);
  *f = (f1 * f2);
  *f = frexpl(*f, e);
  *e += (e1 + e2);
}

static inline void ef_divl(int *const e, long double *const f, const int e1, const long double f1, const int e2, const long double f2)
{
  PVN_ASSERT(e);
  PVN_ASSERT(f);
  *f = (f1 / f2);
  *f = frexpl(*f, e);
  *e += (e1 - e2);
}

/* f1, f2 >= 0 */
static inline int ef_cmpl(const int e1, const long double f1, const int e2, const long double f2)
{
  if (f2 == 0.0L)
    return (f1 != 0.0L);
  if (f1 == 0.0L)
    return -1;
  if (e1 < e2)
    return -1;
  if (e2 < e1)
    return 1;
  if (f1 < f2)
    return -1;
  if (f2 < f1)
    return 1;
  return 0;
}

static void xlpsv2(const long double A11, const long double A12, const long double A22, long double *const tf, long double *const cf, long double *const sf, long double *const tp, long double *const cp, long double *const sp, long double *const s1, long double *const s2, int *const es
#ifdef PVN_PRINTOUT
                   , char *const s
#endif /* PVN_PRINTOUT */
                   )
{
  PVN_ASSERT(tf);
  PVN_ASSERT(cf);
  PVN_ASSERT(sf);
  PVN_ASSERT(tp);
  PVN_ASSERT(cp);
  PVN_ASSERT(sp);
  PVN_ASSERT(s1);
  PVN_ASSERT(s2);
  PVN_ASSERT(es);
  int ae = 0, be = 0, abe = 0, a_be = 0, de = 0, ne = 0, t2e = 0;
  long double af = 0.0L, bf = 0.0L, abf = 0.0L, a_bf = 0.0L, df = 0.0L, nf = 0.0L, t2f = 0.0L, t2 = 0.0L;

  /* tan(2φ) */
  if (A11 == A22)
    t2 = ((2.0L * A22) / A12);
  else if (A12 == A22) {
    /* (2 * A12 * A22) / (A11 * A11) */
    af = frexpl(A11, &ae);
    abf = frexpl(A12, &abe);
    bf = frexpl(A22, &be);
    ef_mull(&ne, &nf, abe, abf, be, bf);
    ++ne;
    ef_mull(&de, &df, ae, af, ae, af);
    ef_divl(&t2e, &t2f, ne, nf, de, df);
    t2 = scalbnl(t2f, t2e);
  }
  else if ((A11 / A12) < (LDBL_EPSILON * 0.5L))
    t2 = ((2.0 * A22) / A12);
#ifdef PVN_SV2_SAFE
  else {
    /* should never overflow */
    const long double a = hypotl(A11, A12);
    af = frexpl(a, &ae);
    const long double b = A22;
    bf = frexpl(b, &be);
    abf = (a + b);
    if (!isfinite(abf)) {
      abf = ((0.5L * a) + (0.5L * b));
      de = 1;
    }
    abf = frexpl(abf, &abe);
    abe += de;
    a_bf = (a - b);
    if (a == b)
      de = 0;
    else if (a_bf >= LDBL_MIN) {
      a_bf = frexpl(a_bf, &a_be);
      de = 1;
    }
    else {
      de = ((LDBL_MIN_EXP + LDBL_MANT_DIG) - pvn_imin(ae, be));
      a_bf = (scalbnl(af, (ae + de)) - scalbnl(bf, (be + de)));
      a_bf = frexpl(a_bf, &a_be);
      a_be -= de;
      de = -1;
    }
    if (de)
      ef_mull(&de, &df, a_be, a_bf, abe, abf);
    af = frexpl(A12, &ae);
    ef_mull(&ne, &nf, ae, af, be, bf);
    ++ne;
    ef_divl(&t2e, &t2f, ne, nf, de, df);
    if (isfinite(t2f))
      t2 = scalbnl(t2f, t2e);
    else
      t2 = t2f;
  }
#else /* !PVN_SV2_SAFE */
  else if (A11 > A12) {
    const long double x = (A12 / A11);
    const long double y = (A22 / A11);
    t2 = (((2.0L * x) * y) / fmaxl(fmal((x - y), (x + y), 1.0L), 0.0L));
  }
  else if (A12 > A11) {
    const long double x = (A11 / A12);
    const long double y = (A22 / A12);
    t2 = ((2.0L * y) / fmaxl(fmal((x - y), (x + y), 1.0L), 0.0L));
  }
  else {
    const long double y = (A22 / A12);
    t2 = ((2.0L * y) / fmaxl(fmal((1.0L - y), (1.0L + y), 1.0L), 0.0L));
  }
#endif /* ?PVN_SV2_SAFE */

  /* tan(φ) */
  if (isfinite(t2))
    *tf = (t2 / (1.0L + hypotl(t2, 1.0L)));
  else
    *tf = 1.0L;
#ifdef PVN_PRINTOUT
  if (s) (void)dprintf(PVN_PRINTOUT, "tan(φ)=%s, ", pvn_xtoa(s, *tf));
#endif /* PVN_PRINTOUT */
  *cf = hypotl(*tf, 1.0L); /* sec(φ) */
  *sf = (*tf / *cf); /* sin(φ) */
  nf = frexpl(*cf, &ne);
  *cf = (1.0L / *cf); /* cos(φ) */

  *sp = fmal(*tf, A22, A12);
  *tp = (*sp / A11); /* tan(ψ) */
#ifdef PVN_PRINTOUT
  if (s) (void)dprintf(PVN_PRINTOUT, "tan(ψ)=%s\n", pvn_dtoa(s, *tp));
#endif /* PVN_PRINTOUT */

  af = frexpl(A11, &ae);
  bf = frexpl(A22, &be);

  if (isfinite(*tp)) {
    *cp = hypotl(*tp, 1.0L); /* sec(ψ) */
    *sp = (*tp / *cp); /* sin(ψ) */
    df = frexpl(*cp, &de);
    *cp = (1.0L / *cp); /* cos(ψ) */
    ef_divl(&t2e, &t2f, ne, nf, de, df);
    /* s2 = A22 * (sec(φ) / sec(ψ)) */
    ef_mull(&abe, &abf, be, bf, t2e, t2f);
    /* s1 = A11 * (sec(ψ) / sec(φ)) */
    ef_divl(&a_be, &a_bf, ae, af, t2e, t2f);
  }
  else {
    /* s1 ≈ A12 + tan(φ) * A22 */
    a_bf = frexpl(*sp, &a_be);
    /* tan(ψ) so large that sec(ψ) ≈ tan(ψ) */
    ef_divl(&t2e, &t2f, ae, af, a_be, a_bf);
    *sp = 1.0L; /* sin(ψ) */
    *cp = scalbnl(t2f, t2e); /* cos(ψ) */
    /* s2 ≈ A22 * cos(ψ) */
    ef_mull(&abe, &abf, be, bf, t2e, t2f);
  }

#ifdef PVN_PRINTOUT
  if (s) {
    (void)dprintf(PVN_PRINTOUT, "cos(φ)=%s, ", pvn_xtoa(s, *cf));
    (void)dprintf(PVN_PRINTOUT, "sin(φ)=%s\n", pvn_xtoa(s, *sf));
    (void)dprintf(PVN_PRINTOUT, "cos(ψ)=%s, ", pvn_xtoa(s, *cf));
    (void)dprintf(PVN_PRINTOUT, "sin(ψ)=%s\n", pvn_xtoa(s, *sf));
  }
#endif /* PVN_PRINTOUT */

  *s1 = a_bf;
  es[1] = a_be;
  *s2 = abf;
  es[2] = abe;
}

int PVN_FABI(pvn_xljsv2,PVN_XLJSV2)
(const long double *const a11, const long double *const a21, const long double *const a12, const long double *const a22,
 long double *const u11, long double *const u21, long double *const u12, long double *const u22,
 long double *const v11, long double *const v21, long double *const v12, long double *const v22,
 long double *const s1, long double *const s2, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a21);
  PVN_ASSERT(a12);
  PVN_ASSERT(a22);
  PVN_ASSERT(u11);
  PVN_ASSERT(u21);
  PVN_ASSERT(u12);
  PVN_ASSERT(u22);
  PVN_ASSERT(v11);
  PVN_ASSERT(v21);
  PVN_ASSERT(v12);
  PVN_ASSERT(v22);
  PVN_ASSERT(s1);
  PVN_ASSERT(s2);
  PVN_ASSERT(es);
  if (!isfinite(*a11))
    return -1;
  if (!isfinite(*a21))
    return -2;
  if (!isfinite(*a12))
    return -3;
  if (!isfinite(*a22))
    return -4;
  int knd = 0, mxe = INT_MIN, e = INT_MIN;
  if (*a11 != 0.0L) {
    knd |= 1;
    (void)frexpl(*a11, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a21 != 0.0L) {
    knd |= 2;
    (void)frexpl(*a21, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a12 != 0.0L) {
    knd |= 4;
    (void)frexpl(*a12, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a22 != 0.0L) {
    knd |= 8;
    (void)frexpl(*a22, &e);
    if (mxe < e)
      mxe = e;
  }
  switch (knd) {
  case  0:
  case  1:
  case  2:
  case  4:
  case  6:
  case  8:
  case  9:
    e = 0;
    break;
  case  3:
  case  5:
  case  7:
  case 10:
  case 11:
  case 12:
  case 13:
  case 14:
    e = (LDBL_MAX_EXP - mxe - 1);
    break;
  case 15:
    e = (LDBL_MAX_EXP - mxe - 2);
    break;
  default:
    return INT_MIN;
  }
  mxe = *es;
  if (!*es)
    *es = e;
  else if (*es < 0)
    ++*es;
  es[2] = es[1] = 0;

  /* scaling by 2^(*es), can only fail when mxe != 0 */
  long double A11 = *a11, A21 = *a21, A12 = *a12, A22 = *a22;
  if (*es) {
    A11 = scalbnl(A11, *es);
    A21 = scalbnl(A21, *es);
    A12 = scalbnl(A12, *es);
    A22 = scalbnl(A22, *es);
    if (mxe) {
      if (!isfinite(A11))
        return -5;
      if (!isfinite(A21))
        return -6;
      if (!isfinite(A12))
        return -7;
      if (!isfinite(A22))
        return -8;
    }
    if (*es < 0) {
      knd = 0;
      if (A11 != 0.0L)
        knd |= 1;
      if (A21 != 0.0L)
        knd |= 2;
      if (A12 != 0.0L)
        knd |= 4;
      if (A22 != 0.0L)
        knd |= 8;
    }
  }

  *u11 = 1.0L;
  *u21 = 0.0L;
  *u12 = 0.0L;
  *u22 = 1.0L;
  *v11 = 1.0L;
  *v21 = 0.0L;
  *v12 = 0.0L;
  *v22 = 1.0L;
  *s1 = 0.0L;
  *s2 = 0.0L;

  /* simplify the form of A */
  switch (knd) {
  case  0:
    /* [ 0 0 ] */
    /* [ 0 0 ] */
    *u11 = copysignl(1.0L, A11);
    *u22 = copysignl(1.0L, A22);
    A22 = A12 = A21 = A11 = 0.0L;
    break;
  case  1:
    /* [ * 0 ] */
    /* [ 0 0 ] */
    if (A11 < 0.0L) {
      *u11 = -1.0L;
      A11 = -A11;
    }
    *u22 = copysignl(1.0L, A22);
    A22 = A12 = A21 = 0.0L;
    *s1 = A11;
    break;
  case  2:
    /* [ 0 0 ] */
    /* [ * 0 ] */
    *u11 = 0.0L;
    *u22 = 0.0L;
    A11 = A21;
    A22 = A12;
    if (A11 < 0.0L) {
      *u12 = -1.0L;
      A11 = -A11;
    }
    else
      *u12 = 1.0L;
    *u21 = copysignl(1.0L, A12);
    A22 = A12 = A21 = 0.0L;
    *s1 = A11;
    break;
  case  3:
    /* [ * 0 ] */
    /* [ * 0 ] */
    A12 = fabsl(A11);
    A22 = fabsl(A21);
    if (A12 < A22) {
      *u11 = 0.0L;
      *u21 = copysignl(1.0L, A11);
      *u12 = copysignl(1.0L, A21);
      *u22 = 0.0L;
      A11 = A22;
      A21 = A12;
    }
    else {
      *u11 = copysignl(1.0L, A11);
      *u22 = copysignl(1.0L, A21);
      A11 = A12;
      A21 = A22;
    }
    A22 = A12 = 0.0L;
    *s1 = hypotl(A11, A21);
    e = 3;
    break;
  case  4:
    /* [ 0 * ] */
    /* [ 0 0 ] */
    *u11 = copysignl(1.0L, A12);
    *u22 = copysignl(1.0L, A21);
    A11 = fabsl(A12);
    A22 = A12 = A21 = 0.0L;
    *v11 = 0.0L;
    *v21 = 1.0L;
    *v12 = 1.0L;
    *v22 = 0.0L;
    *s1 = A11;
    break;
  case  5:
    /* [ * * ] */
    /* [ 0 0 ] */
    A21 = fabsl(A11);
    A22 = fabsl(A12);
    if (A21 < A22) {
      *v11 = 0.0L;
      *v21 = copysignl(1.0L, A12);
      *v12 = copysignl(1.0L, A11);
      *v22 = 0.0L;
      A11 = A22;
      A12 = A21;
    }
    else {
      *v11 = copysignl(1.0L, A11);
      *v22 = copysignl(1.0L, A12);
      A11 = A21;
      A12 = A22;
    }
    A22 = A21 = 0.0L;
    *s1 = hypotl(A11, A12);
    e = 5;
    break;
  case  6:
    /* [ 0 * ] */
    /* [ * 0 ] */
    *u11 = copysignl(1.0L, A12);
    *u22 = copysignl(1.0L, A21);
    A11 = fabsl(A12);
    A22 = fabsl(A21);
    A12 = A21 = 0.0L;
    *v11 = 0.0L;
    *v21 = 1.0L;
    *v12 = 1.0L;
    *v22 = 0.0L;
    *s1 = A11;
    *s2 = A22;
    break;
  case  7:
    /* [ * * ] */
    /* [ * 0 ] */
    A22 = A11;
    A11 = A12;
    A12 = A22;
    A22 = A21;
    A21 = 0.0L;
    *v11 = 0.0L;
    *v22 = 0.0L;
    if (A11 < 0.0L) {
      A11 = -A11;
      *v21 = -1.0L;
    }
    else
      *v21 = 1.0L;
    if (A12 < 0.0L) {
      A12 = -A12;
      A22 = -A22;
      *v12 = -1.0L;
    }
    else
      *v12 = 1.0L;
    if (A22 < 0.0L) {
      *u22 = -1.0L;
      A22 = -A22;
    }
    e = 13;
    break;
  case  8:
    /* [ 0 0 ] */
    /* [ 0 * ] */
    *u11 = 0.0L;
    *u21 = copysignl(1.0L, A11);
    *u12 = copysignl(1.0L, A22);
    *u22 = 0.0L;
    A11 = fabsl(A22);
    A22 = A12 = A21 = 0.0L;
    *v11 = 0.0L;
    *v21 = 1.0L;
    *v12 = 1.0L;
    *v22 = 0.0L;
    *s1 = A11;
    break;
  case  9:
    /* [ * 0 ] */
    /* [ 0 * ] */
    if (A11 < 0.0L) {
      *u11 = -1.0L;
      A11 = -A11;
    }
    if (A22 < 0.0L) {
      *u22 = -1.0L;
      A22 = -A22;
    }
    A12 = A21 = 0.0L;
    *s1 = A11;
    *s2 = A22;
    break;
  case 10:
    /* [ 0 0 ] */
    /* [ * * ] */
    *u11 = 0.0L;
    *u21 = 1.0L;
    *u12 = 1.0L;
    *u22 = 0.0L;
    A11 = A21;
    A12 = A22;
    A21 = fabsl(A11);
    A22 = fabsl(A12);
    if (A21 < A22) {
      *v11 = 0.0L;
      *v21 = copysignl(1.0L, A12);
      *v12 = copysignl(1.0L, A11);
      *v22 = 0.0L;
      A11 = A22;
      A12 = A21;
    }
    else {
      *v11 = copysignl(1.0L, A11);
      *v22 = copysignl(1.0L, A12);
      A11 = A21;
      A12 = A22;
    }
    A22 = A21 = 0.0L;
    *s1 = hypotl(A11, A12);
    e = 5;
    break;
  case 11:
    /* [ * 0 ] */
    /* [ * * ] */
    *u11 = 0.0L;
    *u12 = 1.0L;
    *u22 = 0.0L;
    A12 = A11;
    A11 = A22;
    A22 = A12;
    A12 = A21;
    A21 = 0.0L;
    *v11 = 0.0L;
    *v22 = 0.0L;
    if (A11 < 0.0L) {
      A11 = -A11;
      *v21 = -1.0L;
    }
    else
      *v21 = 1.0L;
    if (A12 < 0.0L) {
      A12 = -A12;
      A22 = -A22;
      *v12 = -1.0L;
    }
    else
      *v12 = 1.0L;
    if (A22 < 0.0L) {
      *u21 = -1.0L;
      A22 = -A22;
    }
    else
      *u21 = 1.0L;
    e = 13;
    break;
  case 12:
    /* [ 0 * ] */
    /* [ 0 * ] */
    A11 = A12;
    A21 = A22;
    *v11 = 0.0L;
    *v21 = 1.0L;
    *v12 = 1.0L;
    *v22 = 0.0L;
    A12 = fabsl(A11);
    A22 = fabsl(A21);
    if (A12 < A22) {
      *u11 = 0.0L;
      *u21 = copysignl(1.0L, A11);
      *u12 = copysignl(1.0L, A21);
      *u22 = 0.0L;
      A11 = A22;
      A21 = A12;
    }
    else {
      *u11 = copysignl(1.0L, A11);
      *u22 = copysignl(1.0L, A21);
      A11 = A12;
      A21 = A22;
    }
    A22 = A12 = 0.0L;
    *s1 = hypotl(A11, A21);
    e = 3;
    break;
  case 13:
    /* [ * * ] */
    /* [ 0 * ] */
    if (A11 < 0.0L) {
      A11 = -A11;
      *v11 = -1.0L;
    }
    if (A12 < 0.0L) {
      A12 = -A12;
      A22 = -A22;
      *v22 = -1.0L;
    }
    if (A22 < 0.0L) {
      *u22 = -1.0L;
      A22 = -A22;
    }
    A21 = 0.0L;
    e = 13;
    break;
  case 14:
    /* [ 0 * ] */
    /* [ * * ] */
    *u11 = 0.0L;
    *u12 = 1.0L;
    *u22 = 0.0L;
    A11 = A12;
    A12 = A22;
    A22 = A11;
    A11 = A21;
    A21 = 0.0L;
    if (A11 < 0.0L) {
      A11 = -A11;
      *v11 = -1.0L;
    }
    if (A12 < 0.0L) {
      A12 = -A12;
      A22 = -A22;
      *v22 = -1.0L;
    }
    if (A22 < 0.0L) {
      *u21 = -1.0L;
      A22 = -A22;
    }
    else
      *u21 = 1.0L;
    e = 13;
    break;
  case 15:
    /* [ * * ] */
    /* [ * * ] */
    e = 15;
    break;
  default:
    return INT_MIN;
  }

  if ((e == 13) && (A11 < A22))
    e = -13;

#ifdef PVN_PRINTOUT
#ifdef __x86_64__
  char s[31] = { '\0' };
#else /* !__x86_64__ */
  char s[46] = { '\0' };
#endif /* ?__x86_64__ */
  (void)dprintf(PVN_PRINTOUT, "\tA[knd=%d,e=%d] * 2^%d =\n", knd, e, *es);
  (void)dprintf(PVN_PRINTOUT, "%s ", pvn_xtoa(s, A11));
  (void)dprintf(PVN_PRINTOUT, "%s\n", pvn_xtoa(s, A12));
  (void)dprintf(PVN_PRINTOUT, "%s ", pvn_xtoa(s, A21));
  (void)dprintf(PVN_PRINTOUT, "%s\n", pvn_xtoa(s, A22));
#endif /* PVN_PRINTOUT */

  long double tt = 0.0L, ct = 1.0L, st = 0.0L;

  if (e == 15) {
    /* [ * * ] */
    /* [ * * ] */
    *s1 = hypotl(A11, A21);
    *s2 = hypotl(A12, A22);
    if (*s1 < *s2) {
      pvn_Lswp(&A11, &A12);
      pvn_Lswp(&A21, &A22);
      pvn_Lswp(v11, v12);
      pvn_Lswp(v21, v22);
      pvn_Lswp(s1, s2);
    }
    if (copysignl(1.0L, A11) != 1.0L) {
      *u11 = -*u11;
      *u12 = -*u12;
      A11 = -A11;
      A12 = -A12;
    }
    if (copysignl(1.0L, A21) != 1.0L) {
      *u21 = -*u21;
      *u22 = -*u22;
      A21 = -A21;
      A22 = -A22;
    }
    if (A11 < A21) {
      pvn_Lswp(u11, u21);
      pvn_Lswp(u12, u22);
      pvn_Lswp(&A11, &A21);
      pvn_Lswp(&A12, &A22);
    }
    /* U^T(ϑ):
        cos(ϑ)  sin(ϑ)
       -sin(ϑ)  cos(ϑ)
    */
    tt = (A21 / A11);
    /* 1 / cos */
    ct = hypotl(tt, 1.0L);
    /* apply the left Givens rotation to A (and maybe to U) */
    st = -tt;
    A21 = A12;
    if (ct == 1.0L) {
      A12 = fmal(tt, A22, A12);
      A22 = fmal(st, A21, A22);
      if ((A12 == 0.0L) || (A22 == 0.0L)) {
        A21 = *u11;
        *u11 = fmal(tt, *u21, *u11);
        *u21 = fmal(st,  A21, *u21);
        A21 = *u12;
        *u12 = fmal(tt, *u22, *u12);
        *u22 = fmal(st,  A21, *u22);
      }
      st = tt;
    }
    else {
      A12 = (fmal(tt, A22, A12) / ct);
      A22 = (fmal(st, A21, A22) / ct);
      if ((A12 == 0.0L) || (A22 == 0.0L)) {
        A21 = *u11;
        *u11 = (fmal(tt, *u21, *u11) / ct);
        *u21 = (fmal(st,  A21, *u21) / ct);
        A21 = *u12;
        *u12 = (fmal(tt, *u22, *u12) / ct);
        *u22 = (fmal(st,  A21, *u22) / ct);
      }
      st = (tt / ct);
      ct = (1.0L / ct);
    }
    A11 = *s1;
    A21 = 0.0L;
    if (A12 == 0.0L) {
      A12 = 0.0L;
      if (copysignl(1.0L, A22) != 1.0L) {
        *u21 = -*u21;
        *u22 = -*u22;
        A22 = -A22;
      }
      e = 0;
    }
    else if (A22 == 0.0L) {
      if (copysignl(1.0L, A12) != 1.0L) {
        A12 = -A12;
        A22 = -A22;
        *v12 = -*v12;
        *v22 = -*v22;
      }
      if (copysignl(1.0L, A22) != 1.0L) {
        *u21 = -*u21;
        *u22 = -*u22;
        A22 = 0.0L;
      }
      *s1 = hypotl(*s1, *s2);
      *s2 = 0.0L;
      e = 5;
    }
    else
      e = 13;
    if (A12 < 0.0L) {
      A12 = -A12;
      A22 = -A22;
      *v12 = -*v12;
      *v22 = -*v22;
    }
    if (A22 < 0.0L) {
      A22 = -A22;
      /* sin(ϑ) is always non-negative by construction */
      /* this is just an extra bit of info, used later */
      st = -st;
    }
    if ((e == 13) && (A11 < A22))
      e = -13;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(ϑ)=%s, ", pvn_xtoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(ϑ)=%s, ", pvn_xtoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(ϑ)=%s\n", pvn_xtoa(s, st));
#endif /* PVN_PRINTOUT */
  }

  if (e == 3) {
    /* [ X 0 ] */
    /* [ x 0 ] */
    /* U^T(θ):
        cos(θ)  sin(θ)
       -sin(θ)  cos(θ)
    */
    tt = (A21 / A11);
    /* 1 / cos */
    ct = hypotl(tt, 1.0L);
    /* apply the left Givens rotation to U */
    st = -tt;
    A21 = *u11;
    if (ct == 1.0L) {
      *u11 = fmal(tt, *u21, *u11);
      *u21 = fmal(st,  A21, *u21);
      A21 = *u12;
      *u12 = fmal(tt, *u22, *u12);
      *u22 = fmal(st,  A21, *u22);
      st = tt;
    }
    else {
      *u11 = (fmal(tt, *u21, *u11) / ct);
      *u21 = (fmal(st,  A21, *u21) / ct);
      A21 = *u12;
      *u12 = (fmal(tt, *u22, *u12) / ct);
      *u22 = (fmal(st,  A21, *u22) / ct);
      st = (tt / ct);
      ct = (1.0L / ct);
    }
    A11 = *s1;
    A21 = 0.0L;
    e = 0;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(ϑ)=%s, ", pvn_xtoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(ϑ)=%s, ", pvn_xtoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(ϑ)=%s\n", pvn_xtoa(s, st));
#endif /* PVN_PRINTOUT */
  }

  if (e == 5) {
    /* [ X x ] */
    /* [ 0 0 ] */
    /* V(θ):
       cos(θ) -sin(θ)
       sin(θ)  cos(θ)
    */
    tt = (A12 / A11);
    /* 1 / cos */
    ct = hypotl(tt, 1.0L);
    /* apply the right Givens rotation to V */
    st = -tt;
    A12 = *v11;
    if (ct == 1.0L) {
      *v11 = fmal(tt, *v12, *v11);
      *v12 = fmal(st,  A12, *v12);
      A12 = *v21;
      *v21 = fmal(tt, *v22, *v21);
      *v22 = fmal(st,  A12, *v22);
      st = tt;
    }
    else {
      *v11 = (fmal(tt, *v12, *v11) / ct);
      *v12 = (fmal(st,  A12, *v12) / ct);
      A12 = *v21;
      *v21 = (fmal(tt, *v22, *v21) / ct);
      *v22 = (fmal(st,  A12, *v22) / ct);
      st = (tt / ct);
      ct = (1.0L / ct);
    }
    A11 = *s1;
    A12 = 0.0L;
    e = 0;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(θ)=%s, ", pvn_xtoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(θ)=%s, ", pvn_xtoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(θ)=%s\n", pvn_xtoa(s, st));
#endif /* PVN_PRINTOUT */
  }

#ifdef PVN_PRINTOUT
  (void)dprintf(PVN_PRINTOUT, "\tA =\n");
  (void)dprintf(PVN_PRINTOUT, "%s ", pvn_xtoa(s, scalbnl(A11, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s\n", pvn_xtoa(s, scalbnl(A12, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s ", pvn_xtoa(s, scalbnl(A21, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s\n", pvn_xtoa(s, scalbnl(A22, -*es)));
#endif /* PVN_PRINTOUT */

  if (abs(e) == 13) {
    /* [ x y ] */
    /* [ 0 z ] */

    long double tf = 0.0L, cf = 1.0L, sf = 0.0L, tp = 0.0L, cp = 1.0L, sp = 0.0L;
    if (e == -13) {
      long double tf_ = 0.0L, cf_ = 1.0L, sf_ = 0.0L, tp_ = 0.0L, cp_ = 1.0L, sp_ = 0.0L;
      xlpsv2(A22, A12, A11, &tf_, &cf_, &sf_, &tp_, &cp_, &sp_, s1, s2, es
#ifdef PVN_PRINTOUT
             , s
#endif /* PVN_PRINTOUT */
             );
      tf = (1.0L / tp_);
      cf = sp_;
      sf = cp_;
      tp = (1.0L / tf_);
      cp = sf_;
      sp = cf_;
    }
    else
      xlpsv2(A11, A12, A22, &tf, &cf, &sf, &tp, &cp, &sp, s1, s2, es
#ifdef PVN_PRINTOUT
             , s
#endif /* PVN_PRINTOUT */
             );

    /* update U */
    if (copysignl(1.0L, st) != 1.0L) {
      /* U^T(φ) * diag(1, -1) * U^T(ϑ):
          cos(φ - ϑ) -sin(φ - ϑ)
         -sin(φ - ϑ) -cos(φ - ϑ)
      */
      st = -st;
      long double tf_t = (tf - tt), cf_t = 1.0L, sf_t = 0.0L;
      if (tf_t != 0.0L) {
        tf_t /= fmal(tf, tt, 1.0L);
        if (isfinite(tf_t)) {
          /* 1 / cos */
          cf_t = hypotl(tf_t, 1.0L);
          sf_t = (tf_t / cf_t);
          cf_t = (1.0L / cf_t);
        }
        else {
          sf_t = copysignl(1.0L, tf_t);
          cf_t = 0.0L;
        }
      }
      else
        sf_t = tf_t;
      const long double _sf_t = -sf_t;
      A21 = *u11;
      *u11 = (_sf_t * *u21 + cf_t * *u11);
      *u21 = (_sf_t *  A21 - cf_t * *u21);
      A21 = *u12;
      *u12 = (_sf_t * *u22 + cf_t * *u12);
      *u22 = (_sf_t *  A21 - cf_t * *u22);
      A21 = -1.0L;
    }
    else if (tt != 0.0L) {
      /* U^T(φ) * U^T(ϑ) = U^T(φ + ϑ):
          cos(φ + ϑ)  sin(φ + ϑ)
         -sin(φ + ϑ)  cos(φ + ϑ)
      */
      long double tft = (tf + tt), cft = 1.0L, sft = 0.0L;
      if (tft != 0.0L) {
        tft /= fmal(-tf, tt, 1.0L);
        if (isfinite(tft)) {
          /* 1 / cos */
          cft = hypotl(tft, 1.0L);
          sft = (tft / cft);
          cft = (1.0L / cft);
        }
        else {
          sft = copysignl(1.0L, tft);
          cft = 0.0L;
        }
      }
      else
        sft = tft;
      A21 = *u11;
      *u11 = (cft * *u11 + sft * *u21);
      *u21 = (cft * *u21 - sft *  A21);
      A21 = *u12;
      *u12 = (cft * *u12 + sft * *u22);
      *u22 = (cft * *u22 - sft *  A21);
      A21 = 1.0L;
    }
    else if (tf != 0.0L) {
      /* U^T(φ):
          cos(φ)  sin(φ)
         -sin(φ)  cos(φ)
       */
      A21 = *u11;
      *u11 = (cf * *u11 + sf * *u21);
      *u21 = (cf * *u21 - sf *  A21);
      A21 = *u12;
      *u12 = (cf * *u12 + sf * *u22);
      *u22 = (cf * *u22 - sf *  A21);
      A21 = -0.0L;
    }
    else /* U^T(φ) = I */
      A21 = 0.0L;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "U operation=%s\n", pvn_xtoa(s, A21));
#endif /* PVN_PRINTOUT */

    /* update V */
    if (tp != 0.0L) {
      /* V(ψ):
         cos(ψ) -sin(ψ)
         sin(ψ)  cos(ψ)
      */
      A21 = *v11;
      *v11 = (*v11 * cp + *v12 * sp);
      *v12 = (*v12 * cp -  A21 * sp);
      A21 = *v21;
      *v21 = (*v21 * cp + *v22 * sp);
      *v22 = (*v22 * cp -  A21 * sp);
      A21 = -0.0L;
    }
    else /* V(ψ) = I */
      A21 = 0.0L;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "V operation=%s\n", pvn_xtoa(s, A21));
#endif /* PVN_PRINTOUT */

    if (e == -13) {
      *u21 = -*u21;
      *u22 = -*u22;
      *v12 = -*v12;
      *v22 = -*v22;
    }
  }

  if (ef_cmpl(es[1], *s1, es[2], *s2) < 0) {
    pvn_Lswp(u11, u21);
    pvn_Lswp(u12, u22);
    pvn_Lswp(v11, v12);
    pvn_Lswp(v21, v22);
    pvn_Lswp(s1, s2);
  }
  pvn_Lswp(u21, u12);
  return knd;
}

static long double cpolarl(const long double r, const long double i, long double *const c, long double *const s)
{
  PVN_ASSERT(c);
  PVN_ASSERT(s);
  const long double a = hypotl(r, i);
  *c = copysignl(fminl((fabsl(r) / a), 1.0L), r);
  *s = (i / fmaxl(a, LDBL_TRUE_MIN));
  return a;
}

int PVN_FABI(pvn_wljsv2,PVN_WLJSV2)
(const long double *const a11r, const long double *const a11i, const long double *const a21r, const long double *const a21i, const long double *const a12r, const long double *const a12i, const long double *const a22r, const long double *const a22i,
 long double *const u11r, long double *const u11i, long double *const u21r, long double *const u21i, long double *const u12r, long double *const u12i, long double *const u22r, long double *const u22i,
 long double *const v11r, long double *const v11i, long double *const v21r, long double *const v21i, long double *const v12r, long double *const v12i, long double *const v22r, long double *const v22i,
 long double *const s1, long double *const s2, int *const es)
{
  PVN_ASSERT(a11r); PVN_ASSERT(a11i);
  PVN_ASSERT(a21r); PVN_ASSERT(a21i);
  PVN_ASSERT(a12r); PVN_ASSERT(a12i);
  PVN_ASSERT(a22r); PVN_ASSERT(a22i);
  PVN_ASSERT(u11r); PVN_ASSERT(u11i);
  PVN_ASSERT(u21r); PVN_ASSERT(u21i);
  PVN_ASSERT(u12r); PVN_ASSERT(u12i);
  PVN_ASSERT(u22r); PVN_ASSERT(u22i);
  PVN_ASSERT(v11r); PVN_ASSERT(v11i);
  PVN_ASSERT(v21r); PVN_ASSERT(v21i);
  PVN_ASSERT(v12r); PVN_ASSERT(v12i);
  PVN_ASSERT(v22r); PVN_ASSERT(v22i);
  PVN_ASSERT(s1);
  PVN_ASSERT(s2);
  PVN_ASSERT(es);
  if (!isfinite(*a11r))
    return -1;
  if (!isfinite(*a11i))
    return -2;
  if (!isfinite(*a21r))
    return -3;
  if (!isfinite(*a21i))
    return -4;
  if (!isfinite(*a12r))
    return -5;
  if (!isfinite(*a12i))
    return -6;
  if (!isfinite(*a22r))
    return -7;
  if (!isfinite(*a22i))
    return -8;
  int kndi = 0, mxe = INT_MIN, e = 0;
  if (*a11i != 0.0L) {
    kndi |= 1;
    (void)frexpl(*a11i, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a21i != 0.0L) {
    kndi |= 2;
    (void)frexpl(*a21i, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a12i != 0.0L) {
    kndi |= 4;
    (void)frexpl(*a12i, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a22i != 0.0L) {
    kndi |= 8;
    (void)frexpl(*a22i, &e);
    if (mxe < e)
      mxe = e;
  }
  *u11i = 0.0L;
  *u21i = 0.0L;
  *u12i = 0.0L;
  *u22i = 0.0L;
  *v11i = 0.0L;
  *v21i = 0.0L;
  *v12i = 0.0L;
  *v22i = 0.0L;
  if (!kndi)
    return PVN_FABI(pvn_xljsv2,PVN_XLJSV2)(a11r, a21r, a12r, a22r, u11r, u21r, u12r, u22r, v11r, v21r, v12r, v22r, s1, s2, es);
  int kndr = 0;
  if (*a11r != 0.0L) {
    kndr |= 1;
    (void)frexpl(*a11r, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a21r != 0.0L) {
    kndr |= 2;
    (void)frexpl(*a21r, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a12r != 0.0L) {
    kndr |= 4;
    (void)frexpl(*a12r, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a22r != 0.0L) {
    kndr |= 8;
    (void)frexpl(*a22r, &e);
    if (mxe < e)
      mxe = e;
  }
  if (!kndr) {
    *u11r = 0.0L;
    *u21r = 0.0L;
    *u12r = 0.0L;
    *u22r = 0.0L;
    return PVN_FABI(pvn_xljsv2,PVN_XLJSV2)(a11i, a21i, a12i, a22i, u11i, u21i, u12i, u22i, v11r, v21r, v12r, v22r, s1, s2, es);
  }
  int knd = (kndr | kndi);
  switch (knd) {
  case  1:
  case  2:
  case  4:
  case  6:
  case  8:
  case  9:
    e = (LDBL_MAX_EXP - mxe - 1);
    break;
  case  3:
  case  5:
  case  7:
  case 10:
  case 11:
  case 12:
  case 13:
  case 14:
  case 15:
    e = (LDBL_MAX_EXP - mxe - 2);
    break;
  default:
    return INT_MIN;
  }
  mxe = *es;
  if (!*es)
    *es = e;
  else if (*es < 0)
    ++*es;
  es[2] = es[1] = 0;

  /* scaling by 2^(*es), can only fail when mxe != 0 */
  long double A11r = *a11r, A11i = *a11i, A21r = *a21r, A21i = *a21i, A12r = *a12r, A12i = *a12i, A22r = *a22r, A22i = *a22i;
  if (*es) {
    A11r = scalbnl(A11r, *es);
    A11i = scalbnl(A11i, *es);
    A21r = scalbnl(A21r, *es);
    A21i = scalbnl(A21i, *es);
    A12r = scalbnl(A12r, *es);
    A12i = scalbnl(A12i, *es);
    A22r = scalbnl(A22r, *es);
    A22i = scalbnl(A22i, *es);
    if (mxe) {
      if (!isfinite(A11r))
        return -9;
      if (!isfinite(A11i))
        return -10;
      if (!isfinite(A21r))
        return -11;
      if (!isfinite(A21i))
        return -12;
      if (!isfinite(A12r))
        return -13;
      if (!isfinite(A12i))
        return -14;
      if (!isfinite(A22r))
        return -15;
      if (!isfinite(A22i))
        return -16;
    }
    if (*es < 0) {
      kndr = 0;
      if (A11r != 0.0L)
        kndr |= 1;
      if (A21r != 0.0L)
        kndr |= 2;
      if (A12r != 0.0L)
        kndr |= 4;
      if (A22r != 0.0L)
        kndr |= 8;
      kndi = 0;
      if (A11i != 0.0L)
        kndi |= 1;
      if (A21i != 0.0L)
        kndi |= 2;
      if (A12i != 0.0L)
        kndi |= 4;
      if (A22i != 0.0L)
        kndi |= 8;
      knd = (kndr | kndi);
    }
  }

  *u11r = 1.0L;
  *u21r = 0.0L;
  *u12r = 0.0L;
  *u22r = 1.0L;
  *v11r = 1.0L;
  *v21r = 0.0L;
  *v12r = 0.0L;
  *v22r = 1.0L;
  *s1 = 0.0L;
  *s2 = 0.0L;

  /* simplify the form of A */
  switch (knd) {
  case  0:
  case  1:
  case  8:
  case  9:
    /* [ ? 0 ] */
    /* [ 0 ? ] */
    *s1 = cpolarl(A11r, A11i, u11r, u11i); *u11i = -*u11i;
    A11r = *s1; A11i = 0.0L;
    *s2 = cpolarl(A22r, A22i, u22r, u22i); *u22i = -*u22i;
    A22r = *s2; A22i = 0.0L;
    A12i = A12r = A21i = A21r = 0.0L;
    e = 0;
    break;
  case  2:
  case  4:
  case  6:
    /* [ 0 ? ] */
    /* [ ? 0 ] */
    *u11r = 0.0L;
    *u22r = 0.0L;
    A11r = A21r; A11i = A21i;
    A22r = A12r; A22i = A12i;
    *s1 = cpolarl(A11r, A11i, u12r, u12i); *u12i = -*u12i;
    A11r = *s1; A11i = 0.0L;
    *s2 = cpolarl(A22r, A22i, u21r, u21i); *u21i = -*u21i;
    A22r = *s2; A22i = 0.0L;
    A12i = A12r = A21i = A21r = 0.0L;
    e = 0;
    break;
  case 12:
    /* [ 0 * ] */
    /* [ 0 * ] */
    *v11r = 0.0L;
    *v21r = 1.0L;
    *v12r = 1.0L;
    *v22r = 0.0L;
    A11r = A12r; A11i = A12i;
    A21r = A22r; A21i = A22i;
    /* FALLTHRU */
  case  3:
    /* [ * 0 ] */
    /* [ * 0 ] */
    *s1 = cpolarl(A11r, A11i, u11r, u11i); *u11i = -*u11i;
    A11r = *s1; A11i = 0.0L;
    *s2 = cpolarl(A21r, A21i, u22r, u22i); *u22i = -*u22i;
    A21r = *s2; A21i = 0.0L;
    if (A11r < A21r) {
      pvn_Lswp(u11r, u21r);
      pvn_Lswp(u11i, u21i);
      pvn_Lswp(u22r, u12r);
      pvn_Lswp(u22i, u12i);
      pvn_Lswp(&A11r, &A21r);
    }
    A22i = A22r = A12i = A12r = 0.0L;
    *s1 = hypotl(A11r, A21r);
    *s2 = 0.0L;
    e = 3;
    break;
  case 10:
    /* [ 0 0 ] */
    /* [ * * ] */
    *u11r = 0.0L;
    *u21r = 1.0L;
    *u12r = 1.0L;
    *u22r = 0.0L;
    A11r = A21r; A11i = A21i;
    A12r = A22r; A12i = A22i;
    /* FALLTHRU */
  case  5:
    /* [ * * ] */
    /* [ 0 0 ] */
    *s1 = cpolarl(A11r, A11i, v11r, v11i); *v11i = -*v11i;
    A11r = *s1; A11i = 0.0L;
    *s2 = cpolarl(A12r, A12i, v22r, v22i); *v22i = -*v22i;
    A12r = *s2; A12i = 0.0L;
    if (A11r < A12r) {
      pvn_Lswp(&A11r, &A12r);
      pvn_Lswp(v11r, v12r);
      pvn_Lswp(v11i, v12i);
      pvn_Lswp(v22r, v21r);
      pvn_Lswp(v22i, v21i);
    }
    A22i = A22r = A21i = A21r = 0.0L;
    *s1 = hypotl(A11r, A12r);
    *s2 = 0.0L;
    e = 5;
    break;
  case  7:
    /* [ * * ] */
    /* [ * 0 ] */
    pvn_Lswp(&A11r, &A12r);
    pvn_Lswp(&A11i, &A12i);
    A22r = A21r; A22i = A21i;
    *v11r = 0.0L;
    *v22r = 0.0L;
    *s1 = cpolarl(A11r, A11i, v21r, v21i); *v21i = -*v21i;
    A11r = *s1; A11i = 0.0L;
    *s1 = cpolarl(A12r, A12i, v12r, v12i); *v12i = -*v12i;
    A12r = *s1; A12i = 0.0L;
    pvn_wmul(&A21r, &A21i, A22r, A22i, *v12r, *v12i);
    A22r = A21r; A22i = A21i;
    *s2 = cpolarl(A22r, A22i, u22r, u22i); *u22i = -*u22i;
    A22r = *s2; A22i = 0.0L;
    *s2 = *s1 = A21i = A21r = 0.0L;
    e = 13;
    break;
  case 11:
    /* [ * 0 ] */
    /* [ * * ] */
    *u11r = 0.0L;
    *u12r = 1.0L;
    *u22r = 0.0L;
    pvn_Lswp(&A11r, &A22r);
    pvn_Lswp(&A11i, &A22i);
    A12r = A21r; A12i = A21i;
    *v11r = 0.0L;
    *v22r = 0.0L;
    *s1 = cpolarl(A11r, A11i, v21r, v21i); *v21i = -*v21i;
    A11r = *s1; A11i = 0.0L;
    *s1 = cpolarl(A12r, A12i, v12r, v12i); *v12i = -*v12i;
    A12r = *s1; A12i = 0.0L;
    pvn_wmul(&A21r, &A21i, A22r, A22i, *v12r, *v12i);
    A22r = A21r; A22i = A21i;
    *s2 = cpolarl(A22r, A22i, u21r, u21i); *u21i = -*u21i;
    A22r = *s2; A22i = 0.0L;
    *s2 = *s1 = A21i = A21r = 0.0L;
    e = 13;
    break;
  case 13:
    /* [ * * ] */
    /* [ 0 * ] */
    *s1 = cpolarl(A11r, A11i, v11r, v11i); *v11i = -*v11i;
    A11r = *s1; A11i = 0.0L;
    *s1 = cpolarl(A12r, A12i, v22r, v22i); *v22i = -*v22i;
    A12r = *s1; A12i = 0.0L;
    pvn_wmul(&A21r, &A21i, A22r, A22i, *v22r, *v22i);
    A22r = A21r; A22i = A21i;
    *s2 = cpolarl(A22r, A22i, u22r, u22i); *u22i = -*u22i;
    A22r = *s2; A22i = 0.0L;
    *s2 = *s1 = A21i = A21r = 0.0L;
    e = 13;
    break;
  case 14:
    /* [ 0 * ] */
    /* [ * * ] */
    *u11r = 0.0L;
    *u12r = 1.0L;
    *u22r = 0.0L;
    A11r = A21r; A11i = A21i;
    pvn_Lswp(&A12r, &A22r);
    pvn_Lswp(&A12i, &A22i);
    *s1 = cpolarl(A11r, A11i, v11r, v11i); *v11i = -*v11i;
    A11r = *s1; A11i = 0.0L;
    *s1 = cpolarl(A12r, A12i, v22r, v22i); *v22i = -*v22i;
    A12r = *s1; A12i = 0.0L;
    pvn_wmul(&A21r, &A21i, A22r, A22i, *v22r, *v22i);
    A22r = A21r; A22i = A21i;
    *s2 = cpolarl(A22r, A22i, u21r, u21i); *u21i = -*u21i;
    A22r = *s2; A22i = 0.0L;
    *s2 = *s1 = A21i = A21r = 0.0L;
    e = 13;
    break;
  case 15:
    /* [ * * ] */
    /* [ * * ] */
    e = 15;
    break;
  default:
    return INT_MIN;
  }

  if ((e == 13) && (A11r < A22r))
    e = -13;

#ifdef PVN_PRINTOUT
#ifdef __x86_64__
    char s[31] = { '\0' };
#else /* !__x86_64__ */
    char s[46] = { '\0' };
#endif /* ?__x86_64__ */
#endif /* PVN_PRINTOUT */
  long double tt = 0.0L, ct = 1.0L, st = 0.0L;

  if (e == 15) {
    long double B11, B21, B12, B22, C11, C21, C12, C22, S11, S21, S12, S22;
    B11 = cpolarl(A11r, A11i, &C11, &S11); S11 = -S11;
    B21 = cpolarl(A21r, A21i, &C21, &S21); S21 = -S21;
    B12 = cpolarl(A12r, A12i, &C12, &S12); S12 = -S12;
    B22 = cpolarl(A22r, A22i, &C22, &S22); S22 = -S22;
    *s1 = hypotl(B11, B21);
    *s2 = hypotl(B12, B22);
    if (*s1 < *s2) {
      pvn_Lswp(&A11r, &A12r);
      pvn_Lswp(&A11i, &A12i);
      pvn_Lswp(&A21r, &A22r);
      pvn_Lswp(&A21i, &A22i);
      pvn_Lswp(&B11, &B12);
      pvn_Lswp(&B21, &B22);
      pvn_Lswp(&C11, &C12);
      pvn_Lswp(&C21, &C22);
      pvn_Lswp(&S11, &S12);
      pvn_Lswp(&S21, &S22);
      pvn_Lswp(s1, s2);
      pvn_Lswp(v11r, v12r);
      pvn_Lswp(v11i, v12i);
      pvn_Lswp(v21r, v22r);
      pvn_Lswp(v21i, v22i);
    }
    if (B11 < B21) {
      pvn_Lswp(u11r, u21r);
      pvn_Lswp(u11i, u21i);
      pvn_Lswp(u12r, u22r);
      pvn_Lswp(u12i, u22i);
      pvn_Lswp(&A11r, &A21r);
      pvn_Lswp(&A11i, &A21i);
      pvn_Lswp(&A12r, &A22r);
      pvn_Lswp(&A12i, &A22i);
      pvn_Lswp(&B11, &B21);
      pvn_Lswp(&B12, &B22);
      pvn_Lswp(&C11, &C21);
      pvn_Lswp(&C12, &C22);
      pvn_Lswp(&S11, &S21);
      pvn_Lswp(&S12, &S22);
    }
    pvn_wmul(&ct, &st, C11, S11, *u11r, *u11i);
    *u11r = ct; *u11i = st;
    pvn_wmul(&ct, &st, C11, S11, *u12r, *u12i);
    *u12r = ct; *u12i = st;
    A11r = B11; A11i = 0.0L;
    pvn_wmul(&ct, &st, C11, S11, A12r, A12i);
    A12r = ct; A12i = st;
    pvn_wmul(&ct, &st, C21, S21, *u21r, *u21i);
    *u21r = ct; *u21i = st;
    pvn_wmul(&ct, &st, C21, S21, *u22r, *u22i);
    *u22r = ct; *u22i = st;
    A21r = B21; A21i = 0.0L;
    pvn_wmul(&ct, &st, C21, S21, A22r, A22i);
    A22r = ct; A22i = st;
    /* rotate */
    tt = (A21r / A11r);
    /* 1 / cos */
    ct = hypotl(tt, 1.0L);
    /* apply the left Givens rotation to A and U */
    st = -tt;
    A21r = A12r;
    A21i = A12i;
    if (ct == 1.0L) {
      A12r = fmal(tt, A22r, A12r);
      A12i = fmal(tt, A22i, A12i);
      A22r = fmal(st, A21r, A22r);
      A22i = fmal(st, A21i, A22i);
      A21r = *u11r;
      A21i = *u11i;
      *u11r = fmal(tt, *u21r, *u11r);
      *u11i = fmal(tt, *u21i, *u11i);
      *u21r = fmal(st,  A21r, *u21r);
      *u21i = fmal(st,  A21i, *u21i);
      A21r = *u12r;
      A21i = *u12i;
      *u12r = fmal(tt, *u22r, *u12r);
      *u12i = fmal(tt, *u22i, *u12i);
      *u22r = fmal(st,  A21r, *u22r);
      *u22i = fmal(st,  A21i, *u22i);
      st = tt;
    }
    else {
      A12r = (fmal(tt, A22r, A12r) / ct);
      A12i = (fmal(tt, A22i, A12i) / ct);
      A22r = (fmal(st, A21r, A22r) / ct);
      A22i = (fmal(st, A21i, A22i) / ct);
      A21r = *u11r;
      A21i = *u11i;
      *u11r = (fmal(tt, *u21r, *u11r) / ct);
      *u11i = (fmal(tt, *u21i, *u11i) / ct);
      *u21r = (fmal(st,  A21r, *u21r) / ct);
      *u21i = (fmal(st,  A21i, *u21i) / ct);
      A21r = *u12r;
      A21i = *u12i;
      *u12r = (fmal(tt, *u22r, *u12r) / ct);
      *u12i = (fmal(tt, *u22i, *u12i) / ct);
      *u22r = (fmal(st,  A21r, *u22r) / ct);
      *u22i = (fmal(st,  A21i, *u22i) / ct);
      st = (tt / ct);
      ct = (1.0L / ct);
    }
    A11r = *s1;
    B12 = cpolarl(A12r, A12i, &C12, &S12); S12 = -S12;
    A12r = B12; A12i = 0.0L;
    pvn_wmul(&A21r, &A21i, A22r, A22i, C12, S12);
    A22r = A21r; A22i = A21i;
    pvn_wmul(&A21r, &A21i, *v12r, *v12i, C12, S12);
    *v12r = A21r; *v12i = A21i;
    pvn_wmul(&A21r, &A21i, *v22r, *v22i, C12, S12);
    *v22r = A21r; *v22i = A21i;
    B22 = cpolarl(A22r, A22i, &C22, &S22); S22 = -S22;
    A22r = B22; A22i = 0.0L;
    pvn_wmul(&A21r, &A21i, C22, S22, *u21r, *u21i);
    *u21r = A21r; *u21i = A21i;
    pvn_wmul(&A21r, &A21i, C22, S22, *u22r, *u22i);
    *u22r = A21r; *u22i = A21i;
    A21i = A21r = 0.0L;
    if (A12r == 0.0L)
      e = 0;
    else if (A22r == 0.0L) {
      *s1 = hypotl(*s1, *s2);
      *s2 = 0.0L;
      e = 5;
    }
    else if (A11r < A22r)
      e = -13;
    else
      e = 13;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(ϑ)=%s, ", pvn_xtoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(ϑ)=%s, ", pvn_xtoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(ϑ)=%s\n", pvn_xtoa(s, st));
#endif /* PVN_PRINTOUT */
  }

  if (e == 3) {
    tt = (A21r / A11r);
    /* 1 / cos */
    ct = hypotl(tt, 1.0L);
    /* apply the left Givens rotation to U */
    st = -tt;
    A21r = *u11r;
    A21i = *u11i;
    if (ct == 1.0L) {
      *u11r = fmal(tt, *u21r, *u11r);
      *u11i = fmal(tt, *u21i, *u11i);
      *u21r = fmal(st,  A21r, *u21r);
      *u21i = fmal(st,  A21i, *u21i);
      A21r = *u12r;
      A21i = *u12i;
      *u12r = fmal(tt, *u22r, *u12r);
      *u12i = fmal(tt, *u22i, *u12i);
      *u22r = fmal(st,  A21r, *u22r);
      *u22i = fmal(st,  A21i, *u22i);
      st = tt;
    }
    else {
      *u11r = (fmal(tt, *u21r, *u11r) / ct);
      *u11i = (fmal(tt, *u21i, *u11i) / ct);
      *u21r = (fmal(st,  A21r, *u21r) / ct);
      *u21i = (fmal(st,  A21i, *u21i) / ct);
      A21r = *u12r;
      A21i = *u12i;
      *u12r = (fmal(tt, *u22r, *u12r) / ct);
      *u12i = (fmal(tt, *u22i, *u12i) / ct);
      *u22r = (fmal(st,  A21r, *u22r) / ct);
      *u22i = (fmal(st,  A21i, *u22i) / ct);
      st = (tt / ct);
      ct = (1.0L / ct);
    }
    A11r = *s1;
    A21i = A21r = 0.0L;
    e = 0;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(ϑ)=%s, ", pvn_xtoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(ϑ)=%s, ", pvn_xtoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(ϑ)=%s\n", pvn_xtoa(s, st));
#endif /* PVN_PRINTOUT */
  }

  if (e == 5) {
    tt = (A12r / A11r);
    /* 1 / cos */
    ct = hypotl(tt, 1.0L);
    /* apply the right Givens rotation to V */
    st = -tt;
    A12r = *v11r;
    A12i = *v11i;
    if (ct == 1.0L) {
      *v11r = fmal(tt, *v12r, *v11r);
      *v11i = fmal(tt, *v12i, *v11i);
      *v12r = fmal(st,  A12r, *v12r);
      *v12i = fmal(st,  A12i, *v12i);
      A12r = *v21r;
      A12i = *v21i;
      *v21r = fmal(tt, *v22r, *v21r);
      *v21i = fmal(tt, *v22i, *v21i);
      *v22r = fmal(st,  A12r, *v22r);
      *v22i = fmal(st,  A12i, *v22i);
      st = tt;
    }
    else {
      *v11r = (fmal(tt, *v12r, *v11r) / ct);
      *v11i = (fmal(tt, *v12i, *v11i) / ct);
      *v12r = (fmal(st,  A12r, *v12r) / ct);
      *v12i = (fmal(st,  A12i, *v12i) / ct);
      A12r = *v21r;
      A12i = *v21i;
      *v21r = (fmal(tt, *v22r, *v21r) / ct);
      *v21i = (fmal(tt, *v22i, *v21i) / ct);
      *v22r = (fmal(st,  A12r, *v22r) / ct);
      *v22i = (fmal(st,  A12i, *v22i) / ct);
      st = (tt / ct);
      ct = (1.0L / ct);
    }
    A11r = *s1;
    A12i = A12r = 0.0L;
    e = 0;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(θ)=%s, ", pvn_xtoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(θ)=%s, ", pvn_xtoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(θ)=%s\n", pvn_xtoa(s, st));
#endif /* PVN_PRINTOUT */
  }

#ifdef PVN_PRINTOUT
  (void)dprintf(PVN_PRINTOUT, "\tA =\n");
  (void)dprintf(PVN_PRINTOUT, "(%s,", pvn_xtoa(s, scalbnl(A11r, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s) ", pvn_xtoa(s, scalbnl(A11i, -*es)));
  (void)dprintf(PVN_PRINTOUT, "(%s,", pvn_xtoa(s, scalbnl(A12r, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s)\n", pvn_xtoa(s, scalbnl(A12i, -*es)));
  (void)dprintf(PVN_PRINTOUT, "(%s,", pvn_xtoa(s, scalbnl(A21r, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s) ", pvn_xtoa(s, scalbnl(A21i, -*es)));
  (void)dprintf(PVN_PRINTOUT, "(%s,", pvn_xtoa(s, scalbnl(A22r, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s)\n", pvn_xtoa(s, scalbnl(A22i, -*es)));
#endif /* PVN_PRINTOUT */

  if (abs(e) == 13) {
    long double tf = 0.0L, cf = 1.0L, sf = 0.0L, tp = 0.0L, cp = 1.0L, sp = 0.0L;
    if (e == -13) {
      long double tf_ = 0.0L, cf_ = 1.0L, sf_ = 0.0L, tp_ = 0.0L, cp_ = 1.0L, sp_ = 0.0L;
      xlpsv2(A22r, A12r, A11r, &tf_, &cf_, &sf_, &tp_, &cp_, &sp_, s1, s2, es
#ifdef PVN_PRINTOUT
             , s
#endif /* PVN_PRINTOUT */
             );
      tf = (1.0L / tp_);
      cf = sp_;
      sf = cp_;
      tp = (1.0L / tf_);
      cp = sf_;
      sp = cf_;
    }
    else
      xlpsv2(A11r, A12r, A22r, &tf, &cf, &sf, &tp, &cp, &sp, s1, s2, es
#ifdef PVN_PRINTOUT
             , s
#endif /* PVN_PRINTOUT */
             );

    if (tf != 0.0L) {
      A21r = *u11r; A21i = *u11i;
      *u11r = (cf * *u11r + sf * *u21r);
      *u11i = (cf * *u11i + sf * *u21i);
      *u21r = (cf * *u21r - sf *  A21r);
      *u21i = (cf * *u21i - sf *  A21i);
      A21r = *u12r; A21i = *u12i;
      *u12r = (cf * *u12r + sf * *u22r);
      *u12i = (cf * *u12i + sf * *u22i);
      *u22r = (cf * *u22r - sf *  A21r);
      *u22i = (cf * *u22i - sf *  A21i);
      A21r = -0.0L;
      A21i = ((e == 13) ? 0.0L : -0.0L);
    }
    else {
      A21r = 0.0L;
      A21i = ((e == 13) ? 0.0L : -0.0L);
    }

    if (tp != 0.0L) {
      A21r = *v11r; A21i = *v11i;
      *v11r = (*v11r * cp + *v12r * sp);
      *v11i = (*v11i * cp + *v12i * sp);
      *v12r = (*v12r * cp -  A21r * sp);
      *v12i = (*v12i * cp -  A21i * sp);
      A21r = *v21r; A21i = *v21i;
      *v21r = (*v21r * cp + *v22r * sp);
      *v21i = (*v21i * cp + *v22i * sp);
      *v22r = (*v22r * cp -  A21r * sp);
      *v22i = (*v22i * cp -  A21i * sp);
      A21r = -0.0L;
      A21i = ((e == 13) ? 0.0L : -0.0L);
    }
    else {
      A21r = 0.0L;
      A21i = ((e == 13) ? 0.0L : -0.0L);
    }

    if (e == -13) {
      *u21r = -*u21r;
      *u21i = -*u21i;
      *u22r = -*u22r;
      *u22i = -*u22i;
      *v12r = -*v12r;
      *v12i = -*v12i;
      *v22r = -*v22r;
      *v22i = -*v22i;
    }
  }

  if (ef_cmpl(es[1], *s1, es[2], *s2) < 0) {
    pvn_Lswp(u11r, u21r);
    pvn_Lswp(u11i, u21i);
    pvn_Lswp(u12r, u22r);
    pvn_Lswp(u12i, u22i);
    pvn_Lswp(v11r, v12r);
    pvn_Lswp(v11i, v12i);
    pvn_Lswp(v21r, v22r);
    pvn_Lswp(v21i, v22i);
    pvn_Lswp(s1, s2);
  }
  pvn_Lswp(u21r, u12r);
  pvn_Lswp(u21i, u12i);
  *u11i = -*u11i;
  *u21i = -*u21i;
  *u12i = -*u12i;
  *u22i = -*u22i;
  return knd;
}

#ifdef PVN_QUADMATH
#ifndef QLJR2
#define QLJR2                                                   \
  PVN_ASSERT(a11);                                              \
  PVN_ASSERT(a21);                                              \
  PVN_ASSERT(a12);                                              \
  PVN_ASSERT(a22);                                              \
  PVN_ASSERT(u11);                                              \
  PVN_ASSERT(u21);                                              \
  PVN_ASSERT(u12);                                              \
  PVN_ASSERT(u22);                                              \
  PVN_ASSERT(v11);                                              \
  PVN_ASSERT(v21);                                              \
  PVN_ASSERT(v12);                                              \
  PVN_ASSERT(v22);                                              \
  PVN_ASSERT(s1);                                               \
  PVN_ASSERT(s2);                                               \
  PVN_ASSERT(es);                                               \
  PVN_ASSERT(E);                                                \
  __float128                                                    \
    u11l = *u11, u21l = *u21, u12l = *u12, u22l = *u22,         \
    v11l = *v11, v21l = *v21, v12l = *v12, v22l = *v22,         \
    s1l = scalbnq(*s1, es[1] - es[0]),                          \
    s2l = scalbnq(*s2, es[2] - es[0]);                          \
  /* cond_2(G) */                                               \
  E[0] = fminq((s1l / s2l), (1.0q / 0.0q));                     \
  /* U^T U - I */                                               \
  __float128 T11 = fmaq(u11l, u11l, fmaq(u21l, u21l, -1.0q));   \
  __float128 T21 = fmaq(u12l, u11l, u22l * u21l);               \
  __float128 T12 = T21;                                         \
  __float128 T22 = fmaq(u12l, u12l, fmaq(u22l, u22l, -1.0q));   \
  E[1] = hypotq(hypotq(T11, T21), hypotq(T12, T22));            \
  T11 = fmaq(v11l, v11l, fmaq(v21l, v21l, -1.0q));              \
  T21 = fmaq(v12l, v11l, v22l * v21l);                          \
  T12 = T21;                                                    \
  T22 = fmaq(v12l, v12l, fmaq(v22l, v22l, -1.0q));              \
  E[2] = hypotq(hypotq(T11, T21), hypotq(T12, T22));            \
  u11l *= s1l;                                                  \
  u21l *= s1l;                                                  \
  u12l *= s2l;                                                  \
  u22l *= s2l;                                                  \
  E[3] = hypotq(hypotq(*a11, *a21), hypotq(*a12, *a22));        \
  T11 = fmaq(u11l, v11l, fmaq(u12l, v12l, -*a11));              \
  T21 = fmaq(u21l, v11l, fmaq(u22l, v12l, -*a21));              \
  T12 = fmaq(u11l, v21l, fmaq(u12l, v22l, -*a12));              \
  T22 = fmaq(u21l, v21l, fmaq(u22l, v22l, -*a22));              \
  E[3] = (hypotq(hypotq(T11, T21), hypotq(T12, T22)) / E[3])
#else /* QLJR2 */
#error QLJR2 already defined
#endif /* ?QLJR2 */

void PVN_FABI(pvn_sqljr2,PVN_SQLJR2)
(const float *const a11, const float *const a21, const float *const a12, const float *const a22,
 const float *const u11, const float *const u21, const float *const u12, const float *const u22,
 const float *const v11, const float *const v21, const float *const v12, const float *const v22,
 const float *const s1, const float *const s2, const int *const es, __float128 *const E)
{
  QLJR2;
}

void PVN_FABI(pvn_dqljr2,PVN_DQLJR2)
(const double *const a11, const double *const a21, const double *const a12, const double *const a22,
 const double *const u11, const double *const u21, const double *const u12, const double *const u22,
 const double *const v11, const double *const v21, const double *const v12, const double *const v22,
 const double *const s1, const double *const s2, const int *const es, __float128 *const E)
{
  QLJR2;
}

void PVN_FABI(pvn_xqljr2,PVN_XQLJR2)
(const long double *const a11, const long double *const a21, const long double *const a12, const long double *const a22,
 const long double *const u11, const long double *const u21, const long double *const u12, const long double *const u22,
 const long double *const v11, const long double *const v21, const long double *const v12, const long double *const v22,
 const long double *const s1, const long double *const s2, const int *const es, __float128 *const E)
{
  QLJR2;
}

#ifndef YLJR2
#define YLJR2                                                        \
  PVN_ASSERT(a11r); PVN_ASSERT(a11i);                                \
  PVN_ASSERT(a21r); PVN_ASSERT(a21i);                                \
  PVN_ASSERT(a12r); PVN_ASSERT(a12i);                                \
  PVN_ASSERT(a22r); PVN_ASSERT(a22i);                                \
  PVN_ASSERT(u11r); PVN_ASSERT(u11i);                                \
  PVN_ASSERT(u21r); PVN_ASSERT(u21i);                                \
  PVN_ASSERT(u12r); PVN_ASSERT(u12i);                                \
  PVN_ASSERT(u22r); PVN_ASSERT(u22i);                                \
  PVN_ASSERT(v11r); PVN_ASSERT(v11i);                                \
  PVN_ASSERT(v21r); PVN_ASSERT(v21i);                                \
  PVN_ASSERT(v12r); PVN_ASSERT(v12i);                                \
  PVN_ASSERT(v22r); PVN_ASSERT(v22i);                                \
  PVN_ASSERT(s1);                                                    \
  PVN_ASSERT(s2);                                                    \
  PVN_ASSERT(es);                                                    \
  PVN_ASSERT(E);                                                     \
  __float128                                                         \
    U11r = *u11r, U11i = *u11i, U21r = *u21r, U21i = *u21i,          \
    U12r = *u12r, U12i = *u12i, U22r = *u22r, U22i = *u22i,          \
    V11r = *v11r, V11i = *v11i, V21r = *v21r, V21i = *v21i,          \
    V12r = *v12r, V12i = *v12i, V22r = *v22r, V22i = *v22i,          \
    S1 = scalbnq(*s1, es[1] - es[0]),                                \
    S2 = scalbnq(*s2, es[2] - es[0]);                                \
  /* cond_2(G) */                                                    \
  E[0] = fminq((S1 / S2), (1.0q / 0.0q));                            \
  /* U^H U - I */                                                    \
  __float128 T11r = -1.0q, T11i = 0.0q;                              \
  pvn_yfma(&T11r, &T11i, U21r, -U21i, U21r, U21i, T11r, T11i);       \
  pvn_yfma(&T11r, &T11i, U11r, -U11i, U11r, U11i, T11r, T11i);       \
  __float128 T21r = 0.0q, T21i = 0.0q;                               \
  pvn_ymul(&T21r, &T21i, U22r, -U22i, U21r, U21i);                   \
  pvn_yfma(&T21r, &T21i, U12r, -U12i, U11r, U11i, T21r, T21i);       \
  __float128 T12r = 0.0q, T12i = 0.0q;                               \
  pvn_ymul(&T12r, &T12i, U21r, -U21i, U22r, U22i);                   \
  pvn_yfma(&T12r, &T12i, U11r, -U11i, U12r, U12i, T12r, T12i);       \
  __float128 T22r = -1.0q, T22i = 0.0q;                              \
  pvn_yfma(&T22r, &T22i, U22r, -U22i, U22r, U22i, T22r, T22i);       \
  pvn_yfma(&T22r, &T22i, U12r, -U12i, U12r, U12i, T22r, T22i);       \
  E[1] = hypotq(hypotq(hypotq(T11r, T11i), hypotq(T21r, T21i)),      \
                hypotq(hypotq(T12r, T12i), hypotq(T22r, T22i)));     \
  /* V^H V -I */                                                     \
  pvn_yfma(&T11r, &T11i, V21r, -V21i, V21r, V21i, -1.0q, 0.0q);      \
  pvn_yfma(&T11r, &T11i, V11r, -V11i, V11r, V11i, T11r, T11i);       \
  pvn_ymul(&T21r, &T21i, V22r, -V22i, V21r, V21i);                   \
  pvn_yfma(&T21r, &T21i, V12r, -V12i, V11r, V11i, T21r, T21i);       \
  pvn_ymul(&T12r, &T12i, V21r, -V21i, V22r, V22i);                   \
  pvn_yfma(&T12r, &T12i, V11r, -V11i, V12r, V12i, T12r, T12i);       \
  pvn_yfma(&T22r, &T22i, V22r, -V22i, V22r, V22i, -1.0q, 0.0q);      \
  pvn_yfma(&T22r, &T22i, V12r, -V12i, V12r, V12i, T22r, T22i);       \
  E[2] = hypotq(hypotq(hypotq(T11r, T11i), hypotq(T21r, T21i)),      \
                hypotq(hypotq(T12r, T12i), hypotq(T22r, T22i)));     \
  /* U Σ V^H - G */                                                  \
  U11r *= S1; U11i *= S1;                                            \
  U21r *= S1; U21i *= S1;                                            \
  U12r *= S2; U12i *= S2;                                            \
  U22r *= S2; U22i *= S2;                                            \
  V11i = -V11i; V21i = -V21i; V12i = -V12i; V22i = -V22i;            \
  E[3] = hypotq(hypotq(hypotq(*a11r, *a11i), hypotq(*a21r, *a21i)),  \
                hypotq(hypotq(*a12r, *a12i), hypotq(*a22r, *a22i))); \
  pvn_yfma(&T11r, &T11i, U12r, U12i, V12r, V12i, -*a11r, -*a11i);    \
  pvn_yfma(&T11r, &T11i, U11r, U11i, V11r, V11i, T11r, T11i);        \
  pvn_yfma(&T21r, &T21i, U22r, U22i, V12r, V12i, -*a21r, -*a21i);    \
  pvn_yfma(&T21r, &T21i, U21r, U21i, V11r, V11i, T21r, T21i);        \
  pvn_yfma(&T12r, &T12i, U12r, U12i, V22r, V22i, -*a12r, -*a12i);    \
  pvn_yfma(&T12r, &T12i, U11r, U11i, V21r, V21i, T12r, T12i);        \
  pvn_yfma(&T22r, &T22i, U22r, U22i, V22r, V22i, -*a22r, -*a22i);    \
  pvn_yfma(&T22r, &T22i, U21r, U21i, V21r, V21i, T22r, T22i);        \
  E[3] = (hypotq(hypotq(hypotq(T11r, T11i), hypotq(T21r, T21i)),     \
                 hypotq(hypotq(T12r, T12i), hypotq(T22r, T22i))) / E[3])
#else /* YLJR2 */
#error YLJR2 already defined
#endif /* ?YLJR2 */

void PVN_FABI(pvn_cyljr2,PVN_CYLJR2)
(const float *const a11r, const float *const a11i, const float *const a21r, const float *const a21i,
 const float *const a12r, const float *const a12i, const float *const a22r, const float *const a22i,
 const float *const u11r, const float *const u11i, const float *const u21r, const float *const u21i,
 const float *const u12r, const float *const u12i, const float *const u22r, const float *const u22i,
 const float *const v11r, const float *const v11i, const float *const v21r, const float *const v21i,
 const float *const v12r, const float *const v12i, const float *const v22r, const float *const v22i,
 const float *const s1, const float *const s2, const int *const es, __float128 *const E)
{
  YLJR2;
}

void PVN_FABI(pvn_zyljr2,PVN_ZYLJR2)
(const double *const a11r, const double *const a11i, const double *const a21r, const double *const a21i,
 const double *const a12r, const double *const a12i, const double *const a22r, const double *const a22i,
 const double *const u11r, const double *const u11i, const double *const u21r, const double *const u21i,
 const double *const u12r, const double *const u12i, const double *const u22r, const double *const u22i,
 const double *const v11r, const double *const v11i, const double *const v21r, const double *const v21i,
 const double *const v12r, const double *const v12i, const double *const v22r, const double *const v22i,
 const double *const s1, const double *const s2, const int *const es, __float128 *const E)
{
  YLJR2;
}

void PVN_FABI(pvn_wyljr2,PVN_WYLJR2)
(const long double *const a11r, const long double *const a11i, const long double *const a21r, const long double *const a21i,
 const long double *const a12r, const long double *const a12i, const long double *const a22r, const long double *const a22i,
 const long double *const u11r, const long double *const u11i, const long double *const u21r, const long double *const u21i,
 const long double *const u12r, const long double *const u12i, const long double *const u22r, const long double *const u22i,
 const long double *const v11r, const long double *const v11i, const long double *const v21r, const long double *const v21i,
 const long double *const v12r, const long double *const v12i, const long double *const v22r, const long double *const v22i,
 const long double *const s1, const long double *const s2, const int *const es, __float128 *const E)
{
  YLJR2;
}

static inline void ef_mulq(int *const e, __float128 *const f, const int e1, const __float128 f1, const int e2, const __float128 f2)
{
  PVN_ASSERT(e);
  PVN_ASSERT(f);
  *f = (f1 * f2);
  *f = frexpq(*f, e);
  *e += (e1 + e2);
}

static inline void ef_divq(int *const e, __float128 *const f, const int e1, const __float128 f1, const int e2, const __float128 f2)
{
  PVN_ASSERT(e);
  PVN_ASSERT(f);
  *f = (f1 / f2);
  *f = frexpq(*f, e);
  *e += (e1 - e2);
}

/* f1, f2 >= 0 */
static inline int ef_cmpq(const int e1, const __float128 f1, const int e2, const __float128 f2)
{
  if (f2 == 0.0q)
    return (f1 != 0.0q);
  if (f1 == 0.0q)
    return -1;
  if (e1 < e2)
    return -1;
  if (e2 < e1)
    return 1;
  if (f1 < f2)
    return -1;
  if (f2 < f1)
    return 1;
  return 0;
}

static void qlpsv2(const __float128 A11, const __float128 A12, const __float128 A22, __float128 *const tf, __float128 *const cf, __float128 *const sf, __float128 *const tp, __float128 *const cp, __float128 *const sp, __float128 *const s1, __float128 *const s2, int *const es
#ifdef PVN_PRINTOUT
                   , char *const s
#endif /* PVN_PRINTOUT */
                   )
{
  PVN_ASSERT(tf);
  PVN_ASSERT(cf);
  PVN_ASSERT(sf);
  PVN_ASSERT(tp);
  PVN_ASSERT(cp);
  PVN_ASSERT(sp);
  PVN_ASSERT(s1);
  PVN_ASSERT(s2);
  PVN_ASSERT(es);
  int ae = 0, be = 0, abe = 0, a_be = 0, de = 0, ne = 0, t2e = 0;
  __float128 af = 0.0q, bf = 0.0q, abf = 0.0q, a_bf = 0.0q, df = 0.0q, nf = 0.0q, t2f = 0.0q, t2 = 0.0q;

  /* tan(2φ) */
  if (A11 == A22)
    t2 = ((2.0q * A22) / A12);
  else if (A12 == A22) {
    /* (2 * A12 * A22) / (A11 * A11) */
    af = frexpq(A11, &ae);
    abf = frexpq(A12, &abe);
    bf = frexpq(A22, &be);
    ef_mulq(&ne, &nf, abe, abf, be, bf);
    ++ne;
    ef_mulq(&de, &df, ae, af, ae, af);
    ef_divq(&t2e, &t2f, ne, nf, de, df);
    t2 = scalbnq(t2f, t2e);
  }
  else if ((A11 / A12) < (FLT128_EPSILON * 0.5q))
    t2 = ((2.0q * A22) / A12);
#ifdef PVN_SV2_SAFE
  else {
    /* should never overflow */
    const __float128 a = hypotq(A11, A12);
    af = frexpq(a, &ae);
    const __float128 b = A22;
    bf = frexpq(b, &be);
    abf = (a + b);
    if (!isfiniteq(abf)) {
      abf = ((0.5q * a) + (0.5q * b));
      de = 1;
    }
    abf = frexpq(abf, &abe);
    abe += de;
    a_bf = (a - b);
    if (a == b)
      de = 0;
    else if (a_bf >= FLT128_MIN) {
      a_bf = frexpq(a_bf, &a_be);
      de = 1;
    }
    else {
      de = ((FLT128_MIN_EXP + FLT128_MANT_DIG) - pvn_imin(ae, be));
      a_bf = (scalbnq(af, (ae + de)) - scalbnq(bf, (be + de)));
      a_bf = frexpq(a_bf, &a_be);
      a_be -= de;
      de = -1;
    }
    if (de)
      ef_mulq(&de, &df, a_be, a_bf, abe, abf);
    af = frexpq(A12, &ae);
    ef_mulq(&ne, &nf, ae, af, be, bf);
    ++ne;
    ef_divq(&t2e, &t2f, ne, nf, de, df);
    if (isfiniteq(t2f))
      t2 = scalbnq(t2f, t2e);
    else
      t2 = t2f;
  }
#else /* !PVN_SV2_SAFE */
  else if (A11 > A12) {
    const __float128 x = (A12 / A11);
    const __float128 y = (A22 / A11);
    t2 = (((2.0q * x) * y) / fmaxq(fmaq((x - y), (x + y), 1.0q), 0.0q));
  }
  else if (A12 > A11) {
    const __float128 x = (A11 / A12);
    const __float128 y = (A22 / A12);
    t2 = ((2.0q * y) / fmaxq(fmaq((x - y), (x + y), 1.0q), 0.0q));
  }
  else {
    const __float128 y = (A22 / A12);
    t2 = ((2.0q * y) / fmaxq(fmaq((1.0q - y), (1.0q + y), 1.0q), 0.0q));
  }
#endif /* ?PVN_SV2_SAFE */

  /* tan(φ) */
  if (isfiniteq(t2))
    *tf = (t2 / (1.0q + hypotq(t2, 1.0q)));
  else
    *tf = 1.0q;
#ifdef PVN_PRINTOUT
  if (s) (void)dprintf(PVN_PRINTOUT, "tan(φ)=%s, ", pvn_qtoa(s, *tf));
#endif /* PVN_PRINTOUT */
  *cf = hypotq(*tf, 1.0q); /* sec(φ) */
  *sf = (*tf / *cf); /* sin(φ) */
  nf = frexpq(*cf, &ne);
  *cf = (1.0q / *cf); /* cos(φ) */

  *sp = fmaq(*tf, A22, A12);
  *tp = (*sp / A11); /* tan(ψ) */
#ifdef PVN_PRINTOUT
  if (s) (void)dprintf(PVN_PRINTOUT, "tan(ψ)=%s\n", pvn_qtoa(s, *tp));
#endif /* PVN_PRINTOUT */

  af = frexpq(A11, &ae);
  bf = frexpq(A22, &be);

  if (isfiniteq(*tp)) {
    *cp = hypotq(*tp, 1.0q); /* sec(ψ) */
    *sp = (*tp / *cp); /* sin(ψ) */
    df = frexpq(*cp, &de);
    *cp = (1.0q / *cp); /* cos(ψ) */
    ef_divq(&t2e, &t2f, ne, nf, de, df);
    /* s2 = A22 * (sec(φ) / sec(ψ)) */
    ef_mulq(&abe, &abf, be, bf, t2e, t2f);
    /* s1 = A11 * (sec(ψ) / sec(φ)) */
    ef_divq(&a_be, &a_bf, ae, af, t2e, t2f);
  }
  else {
    /* s1 ≈ A12 + tan(φ) * A22 */
    a_bf = frexpq(*sp, &a_be);
    /* tan(ψ) so large that sec(ψ) ≈ tan(ψ) */
    ef_divq(&t2e, &t2f, ae, af, a_be, a_bf);
    *sp = 1.0q; /* sin(ψ) */
    *cp = scalbnq(t2f, t2e); /* cos(ψ) */
    /* s2 ≈ A22 * cos(ψ) */
    ef_mulq(&abe, &abf, be, bf, t2e, t2f);
  }

#ifdef PVN_PRINTOUT
  if (s) {
    (void)dprintf(PVN_PRINTOUT, "cos(φ)=%s, ", pvn_qtoa(s, *cf));
    (void)dprintf(PVN_PRINTOUT, "sin(φ)=%s\n", pvn_qtoa(s, *sf));
    (void)dprintf(PVN_PRINTOUT, "cos(ψ)=%s, ", pvn_qtoa(s, *cf));
    (void)dprintf(PVN_PRINTOUT, "sin(ψ)=%s\n", pvn_qtoa(s, *sf));
  }
#endif /* PVN_PRINTOUT */

  *s1 = a_bf;
  es[1] = a_be;
  *s2 = abf;
  es[2] = abe;
}

int PVN_FABI(pvn_qljsv2,PVN_QLJSV2)
(const __float128 *const a11, const __float128 *const a21, const __float128 *const a12, const __float128 *const a22,
 __float128 *const u11, __float128 *const u21, __float128 *const u12, __float128 *const u22,
 __float128 *const v11, __float128 *const v21, __float128 *const v12, __float128 *const v22,
 __float128 *const s1, __float128 *const s2, int *const es)
{
  PVN_ASSERT(a11);
  PVN_ASSERT(a21);
  PVN_ASSERT(a12);
  PVN_ASSERT(a22);
  PVN_ASSERT(u11);
  PVN_ASSERT(u21);
  PVN_ASSERT(u12);
  PVN_ASSERT(u22);
  PVN_ASSERT(v11);
  PVN_ASSERT(v21);
  PVN_ASSERT(v12);
  PVN_ASSERT(v22);
  PVN_ASSERT(s1);
  PVN_ASSERT(s2);
  PVN_ASSERT(es);
  if (!isfiniteq(*a11))
    return -1;
  if (!isfiniteq(*a21))
    return -2;
  if (!isfiniteq(*a12))
    return -3;
  if (!isfiniteq(*a22))
    return -4;
  int knd = 0, mxe = INT_MIN, e = INT_MIN;
  if (*a11 != 0.0q) {
    knd |= 1;
    (void)frexpq(*a11, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a21 != 0.0q) {
    knd |= 2;
    (void)frexpq(*a21, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a12 != 0.0q) {
    knd |= 4;
    (void)frexpq(*a12, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a22 != 0.0q) {
    knd |= 8;
    (void)frexpq(*a22, &e);
    if (mxe < e)
      mxe = e;
  }
  switch (knd) {
  case  0:
  case  1:
  case  2:
  case  4:
  case  6:
  case  8:
  case  9:
    e = 0;
    break;
  case  3:
  case  5:
  case  7:
  case 10:
  case 11:
  case 12:
  case 13:
  case 14:
    e = (FLT128_MAX_EXP - mxe - 1);
    break;
  case 15:
    e = (FLT128_MAX_EXP - mxe - 2);
    break;
  default:
    return INT_MIN;
  }
  mxe = *es;
  if (!*es)
    *es = e;
  else if (*es < 0)
    ++*es;
  es[2] = es[1] = 0;

  /* scaling by 2^(*es), can only fail when mxe != 0 */
  __float128 A11 = *a11, A21 = *a21, A12 = *a12, A22 = *a22;
  if (*es) {
    A11 = scalbnq(A11, *es);
    A21 = scalbnq(A21, *es);
    A12 = scalbnq(A12, *es);
    A22 = scalbnq(A22, *es);
    if (mxe) {
      if (!isfiniteq(A11))
        return -5;
      if (!isfiniteq(A21))
        return -6;
      if (!isfiniteq(A12))
        return -7;
      if (!isfiniteq(A22))
        return -8;
    }
    if (*es < 0) {
      knd = 0;
      if (A11 != 0.0q)
        knd |= 1;
      if (A21 != 0.0q)
        knd |= 2;
      if (A12 != 0.0q)
        knd |= 4;
      if (A22 != 0.0q)
        knd |= 8;
    }
  }

  *u11 = 1.0q;
  *u21 = 0.0q;
  *u12 = 0.0q;
  *u22 = 1.0q;
  *v11 = 1.0q;
  *v21 = 0.0q;
  *v12 = 0.0q;
  *v22 = 1.0q;
  *s1 = 0.0q;
  *s2 = 0.0q;

  /* simplify the form of A */
  switch (knd) {
  case  0:
    /* [ 0 0 ] */
    /* [ 0 0 ] */
    *u11 = copysignq(1.0q, A11);
    *u22 = copysignq(1.0q, A22);
    A22 = A12 = A21 = A11 = 0.0q;
    break;
  case  1:
    /* [ * 0 ] */
    /* [ 0 0 ] */
    if (A11 < 0.0q) {
      *u11 = -1.0q;
      A11 = -A11;
    }
    *u22 = copysignq(1.0q, A22);
    A22 = A12 = A21 = 0.0q;
    *s1 = A11;
    break;
  case  2:
    /* [ 0 0 ] */
    /* [ * 0 ] */
    *u11 = 0.0q;
    *u22 = 0.0q;
    A11 = A21;
    A22 = A12;
    if (A11 < 0.0q) {
      *u12 = -1.0q;
      A11 = -A11;
    }
    else
      *u12 = 1.0q;
    *u21 = copysignq(1.0q, A12);
    A22 = A12 = A21 = 0.0q;
    *s1 = A11;
    break;
  case  3:
    /* [ * 0 ] */
    /* [ * 0 ] */
    A12 = fabsq(A11);
    A22 = fabsq(A21);
    if (A12 < A22) {
      *u11 = 0.0q;
      *u21 = copysignq(1.0q, A11);
      *u12 = copysignq(1.0q, A21);
      *u22 = 0.0q;
      A11 = A22;
      A21 = A12;
    }
    else {
      *u11 = copysignq(1.0q, A11);
      *u22 = copysignq(1.0q, A21);
      A11 = A12;
      A21 = A22;
    }
    A22 = A12 = 0.0q;
    *s1 = hypotq(A11, A21);
    e = 3;
    break;
  case  4:
    /* [ 0 * ] */
    /* [ 0 0 ] */
    *u11 = copysignq(1.0q, A12);
    *u22 = copysignq(1.0q, A21);
    A11 = fabsq(A12);
    A22 = A12 = A21 = 0.0q;
    *v11 = 0.0q;
    *v21 = 1.0q;
    *v12 = 1.0q;
    *v22 = 0.0q;
    *s1 = A11;
    break;
  case  5:
    /* [ * * ] */
    /* [ 0 0 ] */
    A21 = fabsq(A11);
    A22 = fabsq(A12);
    if (A21 < A22) {
      *v11 = 0.0q;
      *v21 = copysignq(1.0q, A12);
      *v12 = copysignq(1.0q, A11);
      *v22 = 0.0q;
      A11 = A22;
      A12 = A21;
    }
    else {
      *v11 = copysignq(1.0q, A11);
      *v22 = copysignq(1.0q, A12);
      A11 = A21;
      A12 = A22;
    }
    A22 = A21 = 0.0q;
    *s1 = hypotq(A11, A12);
    e = 5;
    break;
  case  6:
    /* [ 0 * ] */
    /* [ * 0 ] */
    *u11 = copysignq(1.0q, A12);
    *u22 = copysignq(1.0q, A21);
    A11 = fabsq(A12);
    A22 = fabsq(A21);
    A12 = A21 = 0.0q;
    *v11 = 0.0q;
    *v21 = 1.0q;
    *v12 = 1.0q;
    *v22 = 0.0q;
    *s1 = A11;
    *s2 = A22;
    break;
  case  7:
    /* [ * * ] */
    /* [ * 0 ] */
    A22 = A11;
    A11 = A12;
    A12 = A22;
    A22 = A21;
    A21 = 0.0q;
    *v11 = 0.0q;
    *v22 = 0.0q;
    if (A11 < 0.0q) {
      A11 = -A11;
      *v21 = -1.0q;
    }
    else
      *v21 = 1.0q;
    if (A12 < 0.0q) {
      A12 = -A12;
      A22 = -A22;
      *v12 = -1.0q;
    }
    else
      *v12 = 1.0q;
    if (A22 < 0.0q) {
      *u22 = -1.0q;
      A22 = -A22;
    }
    e = 13;
    break;
  case  8:
    /* [ 0 0 ] */
    /* [ 0 * ] */
    *u11 = 0.0q;
    *u21 = copysignq(1.0q, A11);
    *u12 = copysignq(1.0q, A22);
    *u22 = 0.0q;
    A11 = fabsq(A22);
    A22 = A12 = A21 = 0.0q;
    *v11 = 0.0q;
    *v21 = 1.0q;
    *v12 = 1.0q;
    *v22 = 0.0q;
    *s1 = A11;
    break;
  case  9:
    /* [ * 0 ] */
    /* [ 0 * ] */
    if (A11 < 0.0q) {
      *u11 = -1.0q;
      A11 = -A11;
    }
    if (A22 < 0.0q) {
      *u22 = -1.0q;
      A22 = -A22;
    }
    A12 = A21 = 0.0q;
    *s1 = A11;
    *s2 = A22;
    break;
  case 10:
    /* [ 0 0 ] */
    /* [ * * ] */
    *u11 = 0.0q;
    *u21 = 1.0q;
    *u12 = 1.0q;
    *u22 = 0.0q;
    A11 = A21;
    A12 = A22;
    A21 = fabsq(A11);
    A22 = fabsq(A12);
    if (A21 < A22) {
      *v11 = 0.0q;
      *v21 = copysignq(1.0q, A12);
      *v12 = copysignq(1.0q, A11);
      *v22 = 0.0q;
      A11 = A22;
      A12 = A21;
    }
    else {
      *v11 = copysignq(1.0q, A11);
      *v22 = copysignq(1.0q, A12);
      A11 = A21;
      A12 = A22;
    }
    A22 = A21 = 0.0q;
    *s1 = hypotq(A11, A12);
    e = 5;
    break;
  case 11:
    /* [ * 0 ] */
    /* [ * * ] */
    *u11 = 0.0q;
    *u12 = 1.0q;
    *u22 = 0.0q;
    A12 = A11;
    A11 = A22;
    A22 = A12;
    A12 = A21;
    A21 = 0.0q;
    *v11 = 0.0q;
    *v22 = 0.0q;
    if (A11 < 0.0q) {
      A11 = -A11;
      *v21 = -1.0q;
    }
    else
      *v21 = 1.0q;
    if (A12 < 0.0q) {
      A12 = -A12;
      A22 = -A22;
      *v12 = -1.0q;
    }
    else
      *v12 = 1.0q;
    if (A22 < 0.0q) {
      *u21 = -1.0q;
      A22 = -A22;
    }
    else
      *u21 = 1.0q;
    e = 13;
    break;
  case 12:
    /* [ 0 * ] */
    /* [ 0 * ] */
    A11 = A12;
    A21 = A22;
    *v11 = 0.0q;
    *v21 = 1.0q;
    *v12 = 1.0q;
    *v22 = 0.0q;
    A12 = fabsq(A11);
    A22 = fabsq(A21);
    if (A12 < A22) {
      *u11 = 0.0q;
      *u21 = copysignq(1.0q, A11);
      *u12 = copysignq(1.0q, A21);
      *u22 = 0.0q;
      A11 = A22;
      A21 = A12;
    }
    else {
      *u11 = copysignq(1.0q, A11);
      *u22 = copysignq(1.0q, A21);
      A11 = A12;
      A21 = A22;
    }
    A22 = A12 = 0.0q;
    *s1 = hypotq(A11, A21);
    e = 3;
    break;
  case 13:
    /* [ * * ] */
    /* [ 0 * ] */
    if (A11 < 0.0q) {
      A11 = -A11;
      *v11 = -1.0q;
    }
    if (A12 < 0.0q) {
      A12 = -A12;
      A22 = -A22;
      *v22 = -1.0q;
    }
    if (A22 < 0.0q) {
      *u22 = -1.0q;
      A22 = -A22;
    }
    A21 = 0.0q;
    e = 13;
    break;
  case 14:
    /* [ 0 * ] */
    /* [ * * ] */
    *u11 = 0.0q;
    *u12 = 1.0q;
    *u22 = 0.0q;
    A11 = A12;
    A12 = A22;
    A22 = A11;
    A11 = A21;
    A21 = 0.0q;
    if (A11 < 0.0q) {
      A11 = -A11;
      *v11 = -1.0q;
    }
    if (A12 < 0.0q) {
      A12 = -A12;
      A22 = -A22;
      *v22 = -1.0q;
    }
    if (A22 < 0.0q) {
      *u21 = -1.0q;
      A22 = -A22;
    }
    else
      *u21 = 1.0q;
    e = 13;
    break;
  case 15:
    /* [ * * ] */
    /* [ * * ] */
    e = 15;
    break;
  default:
    return INT_MIN;
  }

  if ((e == 13) && (A11 < A22))
    e = -13;

#ifdef PVN_PRINTOUT
  char s[46] = { '\0' };
  (void)dprintf(PVN_PRINTOUT, "\tA[knd=%d,e=%d] * 2^%d =\n", knd, e, *es);
  (void)dprintf(PVN_PRINTOUT, "%s ", pvn_qtoa(s, A11));
  (void)dprintf(PVN_PRINTOUT, "%s\n", pvn_qtoa(s, A12));
  (void)dprintf(PVN_PRINTOUT, "%s ", pvn_qtoa(s, A21));
  (void)dprintf(PVN_PRINTOUT, "%s\n", pvn_qtoa(s, A22));
#endif /* PVN_PRINTOUT */

  __float128 tt = 0.0q, ct = 1.0q, st = 0.0q;

  if (e == 15) {
    /* [ * * ] */
    /* [ * * ] */
    *s1 = hypotq(A11, A21);
    *s2 = hypotq(A12, A22);
    if (*s1 < *s2) {
      pvn_qswp(&A11, &A12);
      pvn_qswp(&A21, &A22);
      pvn_qswp(v11, v12);
      pvn_qswp(v21, v22);
      pvn_qswp(s1, s2);
    }
    if (copysignq(1.0q, A11) != 1.0q) {
      *u11 = -*u11;
      *u12 = -*u12;
      A11 = -A11;
      A12 = -A12;
    }
    if (copysignq(1.0q, A21) != 1.0q) {
      *u21 = -*u21;
      *u22 = -*u22;
      A21 = -A21;
      A22 = -A22;
    }
    if (A11 < A21) {
      pvn_qswp(u11, u21);
      pvn_qswp(u12, u22);
      pvn_qswp(&A11, &A21);
      pvn_qswp(&A12, &A22);
    }
    /* U^T(ϑ):
        cos(ϑ)  sin(ϑ)
       -sin(ϑ)  cos(ϑ)
    */
    tt = (A21 / A11);
    /* 1 / cos */
    ct = hypotq(tt, 1.0q);
    /* apply the left Givens rotation to A (and maybe to U) */
    st = -tt;
    A21 = A12;
    if (ct == 1.0q) {
      A12 = fmaq(tt, A22, A12);
      A22 = fmaq(st, A21, A22);
      if ((A12 == 0.0q) || (A22 == 0.0q)) {
        A21 = *u11;
        *u11 = fmaq(tt, *u21, *u11);
        *u21 = fmaq(st,  A21, *u21);
        A21 = *u12;
        *u12 = fmaq(tt, *u22, *u12);
        *u22 = fmaq(st,  A21, *u22);
      }
      st = tt;
    }
    else {
      A12 = (fmaq(tt, A22, A12) / ct);
      A22 = (fmaq(st, A21, A22) / ct);
      if ((A12 == 0.0q) || (A22 == 0.0q)) {
        A21 = *u11;
        *u11 = (fmaq(tt, *u21, *u11) / ct);
        *u21 = (fmaq(st,  A21, *u21) / ct);
        A21 = *u12;
        *u12 = (fmaq(tt, *u22, *u12) / ct);
        *u22 = (fmaq(st,  A21, *u22) / ct);
      }
      st = (tt / ct);
      ct = (1.0q / ct);
    }
    A11 = *s1;
    A21 = 0.0q;
    if (A12 == 0.0q) {
      A12 = 0.0q;
      if (copysignq(1.0q, A22) != 1.0q) {
        *u21 = -*u21;
        *u22 = -*u22;
        A22 = -A22;
      }
      e = 0;
    }
    else if (A22 == 0.0q) {
      if (copysignq(1.0q, A12) != 1.0q) {
        A12 = -A12;
        A22 = -A22;
        *v12 = -*v12;
        *v22 = -*v22;
      }
      if (copysignq(1.0q, A22) != 1.0q) {
        *u21 = -*u21;
        *u22 = -*u22;
        A22 = 0.0q;
      }
      *s1 = hypotq(*s1, *s2);
      *s2 = 0.0q;
      e = 5;
    }
    else
      e = 13;
    if (A12 < 0.0q) {
      A12 = -A12;
      A22 = -A22;
      *v12 = -*v12;
      *v22 = -*v22;
    }
    if (A22 < 0.0q) {
      A22 = -A22;
      /* sin(ϑ) is always non-negative by construction */
      /* this is just an extra bit of info, used later */
      st = -st;
    }
    if ((e == 13) && (A11 < A22))
      e = -13;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(ϑ)=%s, ", pvn_qtoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(ϑ)=%s, ", pvn_qtoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(ϑ)=%s\n", pvn_qtoa(s, st));
#endif /* PVN_PRINTOUT */
  }

  if (e == 3) {
    /* [ X 0 ] */
    /* [ x 0 ] */
    /* U^T(θ):
        cos(θ)  sin(θ)
       -sin(θ)  cos(θ)
    */
    tt = (A21 / A11);
    /* 1 / cos */
    ct = hypotq(tt, 1.0q);
    /* apply the left Givens rotation to U */
    st = -tt;
    A21 = *u11;
    if (ct == 1.0q) {
      *u11 = fmaq(tt, *u21, *u11);
      *u21 = fmaq(st,  A21, *u21);
      A21 = *u12;
      *u12 = fmaq(tt, *u22, *u12);
      *u22 = fmaq(st,  A21, *u22);
      st = tt;
    }
    else {
      *u11 = (fmaq(tt, *u21, *u11) / ct);
      *u21 = (fmaq(st,  A21, *u21) / ct);
      A21 = *u12;
      *u12 = (fmaq(tt, *u22, *u12) / ct);
      *u22 = (fmaq(st,  A21, *u22) / ct);
      st = (tt / ct);
      ct = (1.0q / ct);
    }
    A11 = *s1;
    A21 = 0.0q;
    e = 0;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(ϑ)=%s, ", pvn_qtoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(ϑ)=%s, ", pvn_qtoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(ϑ)=%s\n", pvn_qtoa(s, st));
#endif /* PVN_PRINTOUT */
  }

  if (e == 5) {
    /* [ X x ] */
    /* [ 0 0 ] */
    /* V(θ):
       cos(θ) -sin(θ)
       sin(θ)  cos(θ)
    */
    tt = (A12 / A11);
    /* 1 / cos */
    ct = hypotq(tt, 1.0q);
    /* apply the right Givens rotation to V */
    st = -tt;
    A12 = *v11;
    if (ct == 1.0q) {
      *v11 = fmaq(tt, *v12, *v11);
      *v12 = fmaq(st,  A12, *v12);
      A12 = *v21;
      *v21 = fmaq(tt, *v22, *v21);
      *v22 = fmaq(st,  A12, *v22);
      st = tt;
    }
    else {
      *v11 = (fmaq(tt, *v12, *v11) / ct);
      *v12 = (fmaq(st,  A12, *v12) / ct);
      A12 = *v21;
      *v21 = (fmaq(tt, *v22, *v21) / ct);
      *v22 = (fmaq(st,  A12, *v22) / ct);
      st = (tt / ct);
      ct = (1.0q / ct);
    }
    A11 = *s1;
    A12 = 0.0q;
    e = 0;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(θ)=%s, ", pvn_qtoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(θ)=%s, ", pvn_qtoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(θ)=%s\n", pvn_qtoa(s, st));
#endif /* PVN_PRINTOUT */
  }

#ifdef PVN_PRINTOUT
  (void)dprintf(PVN_PRINTOUT, "\tA =\n");
  (void)dprintf(PVN_PRINTOUT, "%s ", pvn_qtoa(s, scalbnq(A11, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s\n", pvn_qtoa(s, scalbnq(A12, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s ", pvn_qtoa(s, scalbnq(A21, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s\n", pvn_qtoa(s, scalbnq(A22, -*es)));
#endif /* PVN_PRINTOUT */

  if (abs(e) == 13) {
    /* [ x y ] */
    /* [ 0 z ] */

    __float128 tf = 0.0q, cf = 1.0q, sf = 0.0q, tp = 0.0q, cp = 1.0q, sp = 0.0q;
    if (e == -13) {
      __float128 tf_ = 0.0q, cf_ = 1.0q, sf_ = 0.0q, tp_ = 0.0q, cp_ = 1.0q, sp_ = 0.0q;
      qlpsv2(A22, A12, A11, &tf_, &cf_, &sf_, &tp_, &cp_, &sp_, s1, s2, es
#ifdef PVN_PRINTOUT
             , s
#endif /* PVN_PRINTOUT */
             );
      tf = (1.0q / tp_);
      cf = sp_;
      sf = cp_;
      tp = (1.0q / tf_);
      cp = sf_;
      sp = cf_;
    }
    else
      qlpsv2(A11, A12, A22, &tf, &cf, &sf, &tp, &cp, &sp, s1, s2, es
#ifdef PVN_PRINTOUT
             , s
#endif /* PVN_PRINTOUT */
             );

    /* update U */
    if (copysignq(1.0q, st) != 1.0q) {
      /* U^T(φ) * diag(1, -1) * U^T(ϑ):
          cos(φ - ϑ) -sin(φ - ϑ)
         -sin(φ - ϑ) -cos(φ - ϑ)
      */
      st = -st;
      __float128 tf_t = (tf - tt), cf_t = 1.0q, sf_t = 0.0q;
      if (tf_t != 0.0q) {
        tf_t /= fmaq(tf, tt, 1.0q);
        if (isfiniteq(tf_t)) {
          /* 1 / cos */
          cf_t = hypotq(tf_t, 1.0q);
          sf_t = (tf_t / cf_t);
          cf_t = (1.0q / cf_t);
        }
        else {
          sf_t = copysignq(1.0q, tf_t);
          cf_t = 0.0q;
        }
      }
      else
        sf_t = tf_t;
      const __float128 _sf_t = -sf_t;
      A21 = *u11;
      *u11 = (_sf_t * *u21 + cf_t * *u11);
      *u21 = (_sf_t *  A21 - cf_t * *u21);
      A21 = *u12;
      *u12 = (_sf_t * *u22 + cf_t * *u12);
      *u22 = (_sf_t *  A21 - cf_t * *u22);
      A21 = -1.0q;
    }
    else if (tt != 0.0q) {
      /* U^T(φ) * U^T(ϑ) = U^T(φ + ϑ):
          cos(φ + ϑ)  sin(φ + ϑ)
         -sin(φ + ϑ)  cos(φ + ϑ)
      */
      __float128 tft = (tf + tt), cft = 1.0q, sft = 0.0q;
      if (tft != 0.0q) {
        tft /= fmaq(-tf, tt, 1.0q);
        if (isfiniteq(tft)) {
          /* 1 / cos */
          cft = hypotq(tft, 1.0q);
          sft = (tft / cft);
          cft = (1.0q / cft);
        }
        else {
          sft = copysignq(1.0q, tft);
          cft = 0.0q;
        }
      }
      else
        sft = tft;
      A21 = *u11;
      *u11 = (cft * *u11 + sft * *u21);
      *u21 = (cft * *u21 - sft *  A21);
      A21 = *u12;
      *u12 = (cft * *u12 + sft * *u22);
      *u22 = (cft * *u22 - sft *  A21);
      A21 = 1.0q;
    }
    else if (tf != 0.0q) {
      /* U^T(φ):
          cos(φ)  sin(φ)
         -sin(φ)  cos(φ)
       */
      A21 = *u11;
      *u11 = (cf * *u11 + sf * *u21);
      *u21 = (cf * *u21 - sf *  A21);
      A21 = *u12;
      *u12 = (cf * *u12 + sf * *u22);
      *u22 = (cf * *u22 - sf *  A21);
      A21 = -0.0q;
    }
    else /* U^T(φ) = I */
      A21 = 0.0q;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "U operation=%s\n", pvn_qtoa(s, A21));
#endif /* PVN_PRINTOUT */

    /* update V */
    if (tp != 0.0q) {
      /* V(ψ):
         cos(ψ) -sin(ψ)
         sin(ψ)  cos(ψ)
      */
      A21 = *v11;
      *v11 = (*v11 * cp + *v12 * sp);
      *v12 = (*v12 * cp -  A21 * sp);
      A21 = *v21;
      *v21 = (*v21 * cp + *v22 * sp);
      *v22 = (*v22 * cp -  A21 * sp);
      A21 = -0.0q;
    }
    else /* V(ψ) = I */
      A21 = 0.0q;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "V operation=%s\n", pvn_qtoa(s, A21));
#endif /* PVN_PRINTOUT */

    if (e == -13) {
      *u21 = -*u21;
      *u22 = -*u22;
      *v12 = -*v12;
      *v22 = -*v22;
    }
  }

  if (ef_cmpq(es[1], *s1, es[2], *s2) < 0) {
    pvn_qswp(u11, u21);
    pvn_qswp(u12, u22);
    pvn_qswp(v11, v12);
    pvn_qswp(v21, v22);
    pvn_qswp(s1, s2);
  }
  pvn_qswp(u21, u12);
  return knd;
}

static __float128 cpolarq(const __float128 r, const __float128 i, __float128 *const c, __float128 *const s)
{
  PVN_ASSERT(c);
  PVN_ASSERT(s);
  const __float128 a = hypotq(r, i);
  *c = copysignq(fminq((fabsq(r) / a), 1.0q), r);
  *s = (i / fmaxq(a, FLT128_TRUE_MIN));
  return a;
}

int PVN_FABI(pvn_yljsv2,PVN_YLJSV2)
(const __float128 *const a11r, const __float128 *const a11i, const __float128 *const a21r, const __float128 *const a21i, const __float128 *const a12r, const __float128 *const a12i, const __float128 *const a22r, const __float128 *const a22i,
 __float128 *const u11r, __float128 *const u11i, __float128 *const u21r, __float128 *const u21i, __float128 *const u12r, __float128 *const u12i, __float128 *const u22r, __float128 *const u22i,
 __float128 *const v11r, __float128 *const v11i, __float128 *const v21r, __float128 *const v21i, __float128 *const v12r, __float128 *const v12i, __float128 *const v22r, __float128 *const v22i,
 __float128 *const s1, __float128 *const s2, int *const es)
{
  PVN_ASSERT(a11r); PVN_ASSERT(a11i);
  PVN_ASSERT(a21r); PVN_ASSERT(a21i);
  PVN_ASSERT(a12r); PVN_ASSERT(a12i);
  PVN_ASSERT(a22r); PVN_ASSERT(a22i);
  PVN_ASSERT(u11r); PVN_ASSERT(u11i);
  PVN_ASSERT(u21r); PVN_ASSERT(u21i);
  PVN_ASSERT(u12r); PVN_ASSERT(u12i);
  PVN_ASSERT(u22r); PVN_ASSERT(u22i);
  PVN_ASSERT(v11r); PVN_ASSERT(v11i);
  PVN_ASSERT(v21r); PVN_ASSERT(v21i);
  PVN_ASSERT(v12r); PVN_ASSERT(v12i);
  PVN_ASSERT(v22r); PVN_ASSERT(v22i);
  PVN_ASSERT(s1);
  PVN_ASSERT(s2);
  PVN_ASSERT(es);
  if (!isfiniteq(*a11r))
    return -1;
  if (!isfiniteq(*a11i))
    return -2;
  if (!isfiniteq(*a21r))
    return -3;
  if (!isfiniteq(*a21i))
    return -4;
  if (!isfiniteq(*a12r))
    return -5;
  if (!isfiniteq(*a12i))
    return -6;
  if (!isfiniteq(*a22r))
    return -7;
  if (!isfiniteq(*a22i))
    return -8;
  int kndi = 0, mxe = INT_MIN, e = 0;
  if (*a11i != 0.0q) {
    kndi |= 1;
    (void)frexpq(*a11i, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a21i != 0.0q) {
    kndi |= 2;
    (void)frexpq(*a21i, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a12i != 0.0q) {
    kndi |= 4;
    (void)frexpq(*a12i, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a22i != 0.0q) {
    kndi |= 8;
    (void)frexpq(*a22i, &e);
    if (mxe < e)
      mxe = e;
  }
  *u11i = 0.0q;
  *u21i = 0.0q;
  *u12i = 0.0q;
  *u22i = 0.0q;
  *v11i = 0.0q;
  *v21i = 0.0q;
  *v12i = 0.0q;
  *v22i = 0.0q;
  if (!kndi)
    return PVN_FABI(pvn_qljsv2,PVN_QLJSV2)(a11r, a21r, a12r, a22r, u11r, u21r, u12r, u22r, v11r, v21r, v12r, v22r, s1, s2, es);
  int kndr = 0;
  if (*a11r != 0.0q) {
    kndr |= 1;
    (void)frexpq(*a11r, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a21r != 0.0q) {
    kndr |= 2;
    (void)frexpq(*a21r, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a12r != 0.0q) {
    kndr |= 4;
    (void)frexpq(*a12r, &e);
    if (mxe < e)
      mxe = e;
  }
  if (*a22r != 0.0q) {
    kndr |= 8;
    (void)frexpq(*a22r, &e);
    if (mxe < e)
      mxe = e;
  }
  if (!kndr) {
    *u11r = 0.0q;
    *u21r = 0.0q;
    *u12r = 0.0q;
    *u22r = 0.0q;
    return PVN_FABI(pvn_qljsv2,PVN_QLJSV2)(a11i, a21i, a12i, a22i, u11i, u21i, u12i, u22i, v11r, v21r, v12r, v22r, s1, s2, es);
  }
  int knd = (kndr | kndi);
  switch (knd) {
  case  1:
  case  2:
  case  4:
  case  6:
  case  8:
  case  9:
    e = (FLT128_MAX_EXP - mxe - 1);
    break;
  case  3:
  case  5:
  case  7:
  case 10:
  case 11:
  case 12:
  case 13:
  case 14:
  case 15:
    e = (FLT128_MAX_EXP - mxe - 2);
    break;
  default:
    return INT_MIN;
  }
  mxe = *es;
  if (!*es)
    *es = e;
  else if (*es < 0)
    ++*es;
  es[2] = es[1] = 0;

  /* scaling by 2^(*es), can only fail when mxe != 0 */
  __float128 A11r = *a11r, A11i = *a11i, A21r = *a21r, A21i = *a21i, A12r = *a12r, A12i = *a12i, A22r = *a22r, A22i = *a22i;
  if (*es) {
    A11r = scalbnq(A11r, *es);
    A11i = scalbnq(A11i, *es);
    A21r = scalbnq(A21r, *es);
    A21i = scalbnq(A21i, *es);
    A12r = scalbnq(A12r, *es);
    A12i = scalbnq(A12i, *es);
    A22r = scalbnq(A22r, *es);
    A22i = scalbnq(A22i, *es);
    if (mxe) {
      if (!isfiniteq(A11r))
        return -9;
      if (!isfiniteq(A11i))
        return -10;
      if (!isfiniteq(A21r))
        return -11;
      if (!isfiniteq(A21i))
        return -12;
      if (!isfiniteq(A12r))
        return -13;
      if (!isfiniteq(A12i))
        return -14;
      if (!isfiniteq(A22r))
        return -15;
      if (!isfiniteq(A22i))
        return -16;
    }
    if (*es < 0) {
      kndr = 0;
      if (A11r != 0.0q)
        kndr |= 1;
      if (A21r != 0.0q)
        kndr |= 2;
      if (A12r != 0.0q)
        kndr |= 4;
      if (A22r != 0.0q)
        kndr |= 8;
      kndi = 0;
      if (A11i != 0.0q)
        kndi |= 1;
      if (A21i != 0.0q)
        kndi |= 2;
      if (A12i != 0.0q)
        kndi |= 4;
      if (A22i != 0.0q)
        kndi |= 8;
      knd = (kndr | kndi);
    }
  }

  *u11r = 1.0q;
  *u21r = 0.0q;
  *u12r = 0.0q;
  *u22r = 1.0q;
  *v11r = 1.0q;
  *v21r = 0.0q;
  *v12r = 0.0q;
  *v22r = 1.0q;
  *s1 = 0.0q;
  *s2 = 0.0q;

  /* simplify the form of A */
  switch (knd) {
  case  0:
  case  1:
  case  8:
  case  9:
    /* [ ? 0 ] */
    /* [ 0 ? ] */
    *s1 = cpolarq(A11r, A11i, u11r, u11i); *u11i = -*u11i;
    A11r = *s1; A11i = 0.0q;
    *s2 = cpolarq(A22r, A22i, u22r, u22i); *u22i = -*u22i;
    A22r = *s2; A22i = 0.0q;
    A12i = A12r = A21i = A21r = 0.0q;
    e = 0;
    break;
  case  2:
  case  4:
  case  6:
    /* [ 0 ? ] */
    /* [ ? 0 ] */
    *u11r = 0.0q;
    *u22r = 0.0q;
    A11r = A21r; A11i = A21i;
    A22r = A12r; A22i = A12i;
    *s1 = cpolarq(A11r, A11i, u12r, u12i); *u12i = -*u12i;
    A11r = *s1; A11i = 0.0q;
    *s2 = cpolarq(A22r, A22i, u21r, u21i); *u21i = -*u21i;
    A22r = *s2; A22i = 0.0q;
    A12i = A12r = A21i = A21r = 0.0q;
    e = 0;
    break;
  case 12:
    /* [ 0 * ] */
    /* [ 0 * ] */
    *v11r = 0.0q;
    *v21r = 1.0q;
    *v12r = 1.0q;
    *v22r = 0.0q;
    A11r = A12r; A11i = A12i;
    A21r = A22r; A21i = A22i;
    /* FALLTHRU */
  case  3:
    /* [ * 0 ] */
    /* [ * 0 ] */
    *s1 = cpolarq(A11r, A11i, u11r, u11i); *u11i = -*u11i;
    A11r = *s1; A11i = 0.0q;
    *s2 = cpolarq(A21r, A21i, u22r, u22i); *u22i = -*u22i;
    A21r = *s2; A21i = 0.0q;
    if (A11r < A21r) {
      pvn_qswp(u11r, u21r);
      pvn_qswp(u11i, u21i);
      pvn_qswp(u22r, u12r);
      pvn_qswp(u22i, u12i);
      pvn_qswp(&A11r, &A21r);
    }
    A22i = A22r = A12i = A12r = 0.0q;
    *s1 = hypotq(A11r, A21r);
    *s2 = 0.0q;
    e = 3;
    break;
  case 10:
    /* [ 0 0 ] */
    /* [ * * ] */
    *u11r = 0.0q;
    *u21r = 1.0q;
    *u12r = 1.0q;
    *u22r = 0.0q;
    A11r = A21r; A11i = A21i;
    A12r = A22r; A12i = A22i;
    /* FALLTHRU */
  case  5:
    /* [ * * ] */
    /* [ 0 0 ] */
    *s1 = cpolarq(A11r, A11i, v11r, v11i); *v11i = -*v11i;
    A11r = *s1; A11i = 0.0q;
    *s2 = cpolarq(A12r, A12i, v22r, v22i); *v22i = -*v22i;
    A12r = *s2; A12i = 0.0q;
    if (A11r < A12r) {
      pvn_qswp(&A11r, &A12r);
      pvn_qswp(v11r, v12r);
      pvn_qswp(v11i, v12i);
      pvn_qswp(v22r, v21r);
      pvn_qswp(v22i, v21i);
    }
    A22i = A22r = A21i = A21r = 0.0q;
    *s1 = hypotq(A11r, A12r);
    *s2 = 0.0q;
    e = 5;
    break;
  case  7:
    /* [ * * ] */
    /* [ * 0 ] */
    pvn_qswp(&A11r, &A12r);
    pvn_qswp(&A11i, &A12i);
    A22r = A21r; A22i = A21i;
    *v11r = 0.0q;
    *v22r = 0.0q;
    *s1 = cpolarq(A11r, A11i, v21r, v21i); *v21i = -*v21i;
    A11r = *s1; A11i = 0.0q;
    *s1 = cpolarq(A12r, A12i, v12r, v12i); *v12i = -*v12i;
    A12r = *s1; A12i = 0.0q;
    pvn_ymul(&A21r, &A21i, A22r, A22i, *v12r, *v12i);
    A22r = A21r; A22i = A21i;
    *s2 = cpolarq(A22r, A22i, u22r, u22i); *u22i = -*u22i;
    A22r = *s2; A22i = 0.0q;
    *s2 = *s1 = A21i = A21r = 0.0q;
    e = 13;
    break;
  case 11:
    /* [ * 0 ] */
    /* [ * * ] */
    *u11r = 0.0q;
    *u12r = 1.0q;
    *u22r = 0.0q;
    pvn_qswp(&A11r, &A22r);
    pvn_qswp(&A11i, &A22i);
    A12r = A21r; A12i = A21i;
    *v11r = 0.0q;
    *v22r = 0.0q;
    *s1 = cpolarq(A11r, A11i, v21r, v21i); *v21i = -*v21i;
    A11r = *s1; A11i = 0.0q;
    *s1 = cpolarq(A12r, A12i, v12r, v12i); *v12i = -*v12i;
    A12r = *s1; A12i = 0.0q;
    pvn_ymul(&A21r, &A21i, A22r, A22i, *v12r, *v12i);
    A22r = A21r; A22i = A21i;
    *s2 = cpolarq(A22r, A22i, u21r, u21i); *u21i = -*u21i;
    A22r = *s2; A22i = 0.0q;
    *s2 = *s1 = A21i = A21r = 0.0q;
    e = 13;
    break;
  case 13:
    /* [ * * ] */
    /* [ 0 * ] */
    *s1 = cpolarq(A11r, A11i, v11r, v11i); *v11i = -*v11i;
    A11r = *s1; A11i = 0.0q;
    *s1 = cpolarq(A12r, A12i, v22r, v22i); *v22i = -*v22i;
    A12r = *s1; A12i = 0.0q;
    pvn_ymul(&A21r, &A21i, A22r, A22i, *v22r, *v22i);
    A22r = A21r; A22i = A21i;
    *s2 = cpolarq(A22r, A22i, u22r, u22i); *u22i = -*u22i;
    A22r = *s2; A22i = 0.0q;
    *s2 = *s1 = A21i = A21r = 0.0q;
    e = 13;
    break;
  case 14:
    /* [ 0 * ] */
    /* [ * * ] */
    *u11r = 0.0q;
    *u12r = 1.0q;
    *u22r = 0.0q;
    A11r = A21r; A11i = A21i;
    pvn_qswp(&A12r, &A22r);
    pvn_qswp(&A12i, &A22i);
    *s1 = cpolarq(A11r, A11i, v11r, v11i); *v11i = -*v11i;
    A11r = *s1; A11i = 0.0q;
    *s1 = cpolarq(A12r, A12i, v22r, v22i); *v22i = -*v22i;
    A12r = *s1; A12i = 0.0q;
    pvn_ymul(&A21r, &A21i, A22r, A22i, *v22r, *v22i);
    A22r = A21r; A22i = A21i;
    *s2 = cpolarq(A22r, A22i, u21r, u21i); *u21i = -*u21i;
    A22r = *s2; A22i = 0.0q;
    *s2 = *s1 = A21i = A21r = 0.0q;
    e = 13;
    break;
  case 15:
    /* [ * * ] */
    /* [ * * ] */
    e = 15;
    break;
  default:
    return INT_MIN;
  }

  if ((e == 13) && (A11r < A22r))
    e = -13;

#ifdef PVN_PRINTOUT
  char s[46] = { '\0' };
#endif /* PVN_PRINTOUT */
  __float128 tt = 0.0q, ct = 1.0q, st = 0.0q;

  if (e == 15) {
    __float128 B11, B21, B12, B22, C11, C21, C12, C22, S11, S21, S12, S22;
    B11 = cpolarq(A11r, A11i, &C11, &S11); S11 = -S11;
    B21 = cpolarq(A21r, A21i, &C21, &S21); S21 = -S21;
    B12 = cpolarq(A12r, A12i, &C12, &S12); S12 = -S12;
    B22 = cpolarq(A22r, A22i, &C22, &S22); S22 = -S22;
    *s1 = hypotq(B11, B21);
    *s2 = hypotq(B12, B22);
    if (*s1 < *s2) {
      pvn_qswp(&A11r, &A12r);
      pvn_qswp(&A11i, &A12i);
      pvn_qswp(&A21r, &A22r);
      pvn_qswp(&A21i, &A22i);
      pvn_qswp(&B11, &B12);
      pvn_qswp(&B21, &B22);
      pvn_qswp(&C11, &C12);
      pvn_qswp(&C21, &C22);
      pvn_qswp(&S11, &S12);
      pvn_qswp(&S21, &S22);
      pvn_qswp(s1, s2);
      pvn_qswp(v11r, v12r);
      pvn_qswp(v11i, v12i);
      pvn_qswp(v21r, v22r);
      pvn_qswp(v21i, v22i);
    }
    if (B11 < B21) {
      pvn_qswp(u11r, u21r);
      pvn_qswp(u11i, u21i);
      pvn_qswp(u12r, u22r);
      pvn_qswp(u12i, u22i);
      pvn_qswp(&A11r, &A21r);
      pvn_qswp(&A11i, &A21i);
      pvn_qswp(&A12r, &A22r);
      pvn_qswp(&A12i, &A22i);
      pvn_qswp(&B11, &B21);
      pvn_qswp(&B12, &B22);
      pvn_qswp(&C11, &C21);
      pvn_qswp(&C12, &C22);
      pvn_qswp(&S11, &S21);
      pvn_qswp(&S12, &S22);
    }
    pvn_ymul(&ct, &st, C11, S11, *u11r, *u11i);
    *u11r = ct; *u11i = st;
    pvn_ymul(&ct, &st, C11, S11, *u12r, *u12i);
    *u12r = ct; *u12i = st;
    A11r = B11; A11i = 0.0q;
    pvn_ymul(&ct, &st, C11, S11, A12r, A12i);
    A12r = ct; A12i = st;
    pvn_ymul(&ct, &st, C21, S21, *u21r, *u21i);
    *u21r = ct; *u21i = st;
    pvn_ymul(&ct, &st, C21, S21, *u22r, *u22i);
    *u22r = ct; *u22i = st;
    A21r = B21; A21i = 0.0q;
    pvn_ymul(&ct, &st, C21, S21, A22r, A22i);
    A22r = ct; A22i = st;
    /* rotate */
    tt = (A21r / A11r);
    /* 1 / cos */
    ct = hypotq(tt, 1.0q);
    /* apply the left Givens rotation to A and U */
    st = -tt;
    A21r = A12r;
    A21i = A12i;
    if (ct == 1.0q) {
      A12r = fmaq(tt, A22r, A12r);
      A12i = fmaq(tt, A22i, A12i);
      A22r = fmaq(st, A21r, A22r);
      A22i = fmaq(st, A21i, A22i);
      A21r = *u11r;
      A21i = *u11i;
      *u11r = fmaq(tt, *u21r, *u11r);
      *u11i = fmaq(tt, *u21i, *u11i);
      *u21r = fmaq(st,  A21r, *u21r);
      *u21i = fmaq(st,  A21i, *u21i);
      A21r = *u12r;
      A21i = *u12i;
      *u12r = fmaq(tt, *u22r, *u12r);
      *u12i = fmaq(tt, *u22i, *u12i);
      *u22r = fmaq(st,  A21r, *u22r);
      *u22i = fmaq(st,  A21i, *u22i);
      st = tt;
    }
    else {
      A12r = (fmaq(tt, A22r, A12r) / ct);
      A12i = (fmaq(tt, A22i, A12i) / ct);
      A22r = (fmaq(st, A21r, A22r) / ct);
      A22i = (fmaq(st, A21i, A22i) / ct);
      A21r = *u11r;
      A21i = *u11i;
      *u11r = (fmaq(tt, *u21r, *u11r) / ct);
      *u11i = (fmaq(tt, *u21i, *u11i) / ct);
      *u21r = (fmaq(st,  A21r, *u21r) / ct);
      *u21i = (fmaq(st,  A21i, *u21i) / ct);
      A21r = *u12r;
      A21i = *u12i;
      *u12r = (fmaq(tt, *u22r, *u12r) / ct);
      *u12i = (fmaq(tt, *u22i, *u12i) / ct);
      *u22r = (fmaq(st,  A21r, *u22r) / ct);
      *u22i = (fmaq(st,  A21i, *u22i) / ct);
      st = (tt / ct);
      ct = (1.0q / ct);
    }
    A11r = *s1;
    B12 = cpolarq(A12r, A12i, &C12, &S12); S12 = -S12;
    A12r = B12; A12i = 0.0q;
    pvn_ymul(&A21r, &A21i, A22r, A22i, C12, S12);
    A22r = A21r; A22i = A21i;
    pvn_ymul(&A21r, &A21i, *v12r, *v12i, C12, S12);
    *v12r = A21r; *v12i = A21i;
    pvn_ymul(&A21r, &A21i, *v22r, *v22i, C12, S12);
    *v22r = A21r; *v22i = A21i;
    B22 = cpolarq(A22r, A22i, &C22, &S22); S22 = -S22;
    A22r = B22; A22i = 0.0q;
    pvn_ymul(&A21r, &A21i, C22, S22, *u21r, *u21i);
    *u21r = A21r; *u21i = A21i;
    pvn_ymul(&A21r, &A21i, C22, S22, *u22r, *u22i);
    *u22r = A21r; *u22i = A21i;
    A21i = A21r = 0.0q;
    if (A12r == 0.0q)
      e = 0;
    else if (A22r == 0.0q) {
      *s1 = hypotq(*s1, *s2);
      *s2 = 0.0q;
      e = 5;
    }
    else if (A11r < A22r)
      e = -13;
    else
      e = 13;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(ϑ)=%s, ", pvn_qtoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(ϑ)=%s, ", pvn_qtoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(ϑ)=%s\n", pvn_qtoa(s, st));
#endif /* PVN_PRINTOUT */
  }

  if (e == 3) {
    tt = (A21r / A11r);
    /* 1 / cos */
    ct = hypotq(tt, 1.0q);
    /* apply the left Givens rotation to U */
    st = -tt;
    A21r = *u11r;
    A21i = *u11i;
    if (ct == 1.0q) {
      *u11r = fmaq(tt, *u21r, *u11r);
      *u11i = fmaq(tt, *u21i, *u11i);
      *u21r = fmaq(st,  A21r, *u21r);
      *u21i = fmaq(st,  A21i, *u21i);
      A21r = *u12r;
      A21i = *u12i;
      *u12r = fmaq(tt, *u22r, *u12r);
      *u12i = fmaq(tt, *u22i, *u12i);
      *u22r = fmaq(st,  A21r, *u22r);
      *u22i = fmaq(st,  A21i, *u22i);
      st = tt;
    }
    else {
      *u11r = (fmaq(tt, *u21r, *u11r) / ct);
      *u11i = (fmaq(tt, *u21i, *u11i) / ct);
      *u21r = (fmaq(st,  A21r, *u21r) / ct);
      *u21i = (fmaq(st,  A21i, *u21i) / ct);
      A21r = *u12r;
      A21i = *u12i;
      *u12r = (fmaq(tt, *u22r, *u12r) / ct);
      *u12i = (fmaq(tt, *u22i, *u12i) / ct);
      *u22r = (fmaq(st,  A21r, *u22r) / ct);
      *u22i = (fmaq(st,  A21i, *u22i) / ct);
      st = (tt / ct);
      ct = (1.0q / ct);
    }
    A11r = *s1;
    A21i = A21r = 0.0q;
    e = 0;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(ϑ)=%s, ", pvn_qtoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(ϑ)=%s, ", pvn_qtoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(ϑ)=%s\n", pvn_qtoa(s, st));
#endif /* PVN_PRINTOUT */
  }

  if (e == 5) {
    tt = (A12r / A11r);
    /* 1 / cos */
    ct = hypotq(tt, 1.0q);
    /* apply the right Givens rotation to V */
    st = -tt;
    A12r = *v11r;
    A12i = *v11i;
    if (ct == 1.0q) {
      *v11r = fmaq(tt, *v12r, *v11r);
      *v11i = fmaq(tt, *v12i, *v11i);
      *v12r = fmaq(st,  A12r, *v12r);
      *v12i = fmaq(st,  A12i, *v12i);
      A12r = *v21r;
      A12i = *v21i;
      *v21r = fmaq(tt, *v22r, *v21r);
      *v21i = fmaq(tt, *v22i, *v21i);
      *v22r = fmaq(st,  A12r, *v22r);
      *v22i = fmaq(st,  A12i, *v22i);
      st = tt;
    }
    else {
      *v11r = (fmaq(tt, *v12r, *v11r) / ct);
      *v11i = (fmaq(tt, *v12i, *v11i) / ct);
      *v12r = (fmaq(st,  A12r, *v12r) / ct);
      *v12i = (fmaq(st,  A12i, *v12i) / ct);
      A12r = *v21r;
      A12i = *v21i;
      *v21r = (fmaq(tt, *v22r, *v21r) / ct);
      *v21i = (fmaq(tt, *v22i, *v21i) / ct);
      *v22r = (fmaq(st,  A12r, *v22r) / ct);
      *v22i = (fmaq(st,  A12i, *v22i) / ct);
      st = (tt / ct);
      ct = (1.0q / ct);
    }
    A11r = *s1;
    A12i = A12r = 0.0q;
    e = 0;
#ifdef PVN_PRINTOUT
    (void)dprintf(PVN_PRINTOUT, "tan(θ)=%s, ", pvn_qtoa(s, tt));
    (void)dprintf(PVN_PRINTOUT, "cos(θ)=%s, ", pvn_qtoa(s, ct));
    (void)dprintf(PVN_PRINTOUT, "sin(θ)=%s\n", pvn_qtoa(s, st));
#endif /* PVN_PRINTOUT */
  }

#ifdef PVN_PRINTOUT
  (void)dprintf(PVN_PRINTOUT, "\tA =\n");
  (void)dprintf(PVN_PRINTOUT, "(%s,", pvn_qtoa(s, scalbnq(A11r, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s) ", pvn_qtoa(s, scalbnq(A11i, -*es)));
  (void)dprintf(PVN_PRINTOUT, "(%s,", pvn_qtoa(s, scalbnq(A12r, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s)\n", pvn_qtoa(s, scalbnq(A12i, -*es)));
  (void)dprintf(PVN_PRINTOUT, "(%s,", pvn_qtoa(s, scalbnq(A21r, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s) ", pvn_qtoa(s, scalbnq(A21i, -*es)));
  (void)dprintf(PVN_PRINTOUT, "(%s,", pvn_qtoa(s, scalbnq(A22r, -*es)));
  (void)dprintf(PVN_PRINTOUT, "%s)\n", pvn_qtoa(s, scalbnq(A22i, -*es)));
#endif /* PVN_PRINTOUT */

  if (abs(e) == 13) {
    __float128 tf = 0.0q, cf = 1.0q, sf = 0.0q, tp = 0.0q, cp = 1.0q, sp = 0.0q;
    if (e == -13) {
      __float128 tf_ = 0.0q, cf_ = 1.0q, sf_ = 0.0q, tp_ = 0.0q, cp_ = 1.0q, sp_ = 0.0q;
      qlpsv2(A22r, A12r, A11r, &tf_, &cf_, &sf_, &tp_, &cp_, &sp_, s1, s2, es
#ifdef PVN_PRINTOUT
             , s
#endif /* PVN_PRINTOUT */
             );
      tf = (1.0q / tp_);
      cf = sp_;
      sf = cp_;
      tp = (1.0q / tf_);
      cp = sf_;
      sp = cf_;
    }
    else
      qlpsv2(A11r, A12r, A22r, &tf, &cf, &sf, &tp, &cp, &sp, s1, s2, es
#ifdef PVN_PRINTOUT
             , s
#endif /* PVN_PRINTOUT */
             );

    if (tf != 0.0q) {
      A21r = *u11r; A21i = *u11i;
      *u11r = (cf * *u11r + sf * *u21r);
      *u11i = (cf * *u11i + sf * *u21i);
      *u21r = (cf * *u21r - sf *  A21r);
      *u21i = (cf * *u21i - sf *  A21i);
      A21r = *u12r; A21i = *u12i;
      *u12r = (cf * *u12r + sf * *u22r);
      *u12i = (cf * *u12i + sf * *u22i);
      *u22r = (cf * *u22r - sf *  A21r);
      *u22i = (cf * *u22i - sf *  A21i);
      A21r = -0.0q;
      A21i = ((e == 13) ? 0.0q : -0.0q);
    }
    else {
      A21r = 0.0q;
      A21i = ((e == 13) ? 0.0q : -0.0q);
    }

    if (tp != 0.0q) {
      A21r = *v11r; A21i = *v11i;
      *v11r = (*v11r * cp + *v12r * sp);
      *v11i = (*v11i * cp + *v12i * sp);
      *v12r = (*v12r * cp -  A21r * sp);
      *v12i = (*v12i * cp -  A21i * sp);
      A21r = *v21r; A21i = *v21i;
      *v21r = (*v21r * cp + *v22r * sp);
      *v21i = (*v21i * cp + *v22i * sp);
      *v22r = (*v22r * cp -  A21r * sp);
      *v22i = (*v22i * cp -  A21i * sp);
      A21r = -0.0q;
      A21i = ((e == 13) ? 0.0q : -0.0q);
    }
    else {
      A21r = 0.0q;
      A21i = ((e == 13) ? 0.0q : -0.0q);
    }

    if (e == -13) {
      *u21r = -*u21r;
      *u21i = -*u21i;
      *u22r = -*u22r;
      *u22i = -*u22i;
      *v12r = -*v12r;
      *v12i = -*v12i;
      *v22r = -*v22r;
      *v22i = -*v22i;
    }
  }

  if (ef_cmpq(es[1], *s1, es[2], *s2) < 0) {
    pvn_qswp(u11r, u21r);
    pvn_qswp(u11i, u21i);
    pvn_qswp(u12r, u22r);
    pvn_qswp(u12i, u22i);
    pvn_qswp(v11r, v12r);
    pvn_qswp(v11i, v12i);
    pvn_qswp(v21r, v22r);
    pvn_qswp(v21i, v22i);
    pvn_qswp(s1, s2);
  }
  pvn_qswp(u21r, u12r);
  pvn_qswp(u21i, u12i);
  *u11i = -*u11i;
  *u21i = -*u21i;
  *u12i = -*u12i;
  *u22i = -*u22i;
  return knd;
}
#else /* !PVN_QUADMATH */
void PVN_FABI(pvn_sqljr2,PVN_SQLJR2)
(const float *const a11, const float *const a21, const float *const a12, const float *const a22,
 const float *const u11, const float *const u21, const float *const u12, const float *const u22,
 const float *const v11, const float *const v21, const float *const v12, const float *const v22,
 const float *const s1, const float *const s2, const int *const es, long double *const E)
{
  XLJR2;
}

void PVN_FABI(pvn_dqljr2,PVN_DQLJR2)
(const double *const a11, const double *const a21, const double *const a12, const double *const a22,
 const double *const u11, const double *const u21, const double *const u12, const double *const u22,
 const double *const v11, const double *const v21, const double *const v12, const double *const v22,
 const double *const s1, const double *const s2, const int *const es, long double *const E)
{
  XLJR2;
}

void PVN_FABI(pvn_xqljr2,PVN_XQLJR2)
(const long double *const a11, const long double *const a21, const long double *const a12, const long double *const a22,
 const long double *const u11, const long double *const u21, const long double *const u12, const long double *const u22,
 const long double *const v11, const long double *const v21, const long double *const v12, const long double *const v22,
 const long double *const s1, const long double *const s2, const int *const es, long double *const E)
{
  XLJR2;
}

void PVN_FABI(pvn_cyljr2,PVN_CYLJR2)
(const float *const a11r, const float *const a11i, const float *const a21r, const float *const a21i,
 const float *const a12r, const float *const a12i, const float *const a22r, const float *const a22i,
 const float *const u11r, const float *const u11i, const float *const u21r, const float *const u21i,
 const float *const u12r, const float *const u12i, const float *const u22r, const float *const u22i,
 const float *const v11r, const float *const v11i, const float *const v21r, const float *const v21i,
 const float *const v12r, const float *const v12i, const float *const v22r, const float *const v22i,
 const float *const s1, const float *const s2, const int *const es, long double *const E)
{
  WLJR2;
}

void PVN_FABI(pvn_zyljr2,PVN_ZYLJR2)
(const double *const a11r, const double *const a11i, const double *const a21r, const double *const a21i,
 const double *const a12r, const double *const a12i, const double *const a22r, const double *const a22i,
 const double *const u11r, const double *const u11i, const double *const u21r, const double *const u21i,
 const double *const u12r, const double *const u12i, const double *const u22r, const double *const u22i,
 const double *const v11r, const double *const v11i, const double *const v21r, const double *const v21i,
 const double *const v12r, const double *const v12i, const double *const v22r, const double *const v22i,
 const double *const s1, const double *const s2, const int *const es, long double *const E)
{
  WLJR2;
}

void PVN_FABI(pvn_wyljr2,PVN_WYLJR2)
(const long double *const a11r, const long double *const a11i, const long double *const a21r, const long double *const a21i,
 const long double *const a12r, const long double *const a12i, const long double *const a22r, const long double *const a22i,
 const long double *const u11r, const long double *const u11i, const long double *const u21r, const long double *const u21i,
 const long double *const u12r, const long double *const u12i, const long double *const u22r, const long double *const u22i,
 const long double *const v11r, const long double *const v11i, const long double *const v21r, const long double *const v21i,
 const long double *const v12r, const long double *const v12i, const long double *const v22r, const long double *const v22i,
 const long double *const s1, const long double *const s2, const int *const es, long double *const E)
{
  WLJR2;
}

int PVN_FABI(pvn_qljsv2,PVN_QLJSV2)
(const long double *const a11, const long double *const a21, const long double *const a12, const long double *const a22,
 long double *const u11, long double *const u21, long double *const u12, long double *const u22,
 long double *const v11, long double *const v21, long double *const v12, long double *const v22,
 long double *const s1, long double *const s2, int *const es)
{
  return PVN_FABI(pvn_xljsv2,PVN_XLJSV2)(a11, a21, a12, a22, u11, u21, u12, u22, v11, v21, v12, v22, s1, s2, es);
}

int PVN_FABI(pvn_yljsv2,PVN_YLJSV2)
(const long double *const a11r, const long double *const a11i, const long double *const a21r, const long double *const a21i, const long double *const a12r, const long double *const a12i, const long double *const a22r, const long double *const a22i,
 long double *const u11r, long double *const u11i, long double *const u21r, long double *const u21i, long double *const u12r, long double *const u12i, long double *const u22r, long double *const u22i,
 long double *const v11r, long double *const v11i, long double *const v21r, long double *const v21i, long double *const v12r, long double *const v12i, long double *const v22r, long double *const v22i,
 long double *const s1, long double *const s2, int *const es)
{
  return PVN_FABI(pvn_wljsv2,PVN_WLJSV2)(a11r, a11i, a21r, a21i, a12r, a12i, a22r, a22i, u11r, u11i, u21r, u21i, u12r, u12i, u22r, u22i, v11r, v11i, v21r, v21i, v12r, v12i, v22r, v22i, s1, s2, es);
}
#endif /* ?PVN_QUADMATH */
#endif /* ?PVN_TEST */
