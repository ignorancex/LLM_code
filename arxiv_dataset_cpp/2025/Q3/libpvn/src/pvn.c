#include "pvn.h"

#ifdef PVN_TEST
int main(int argc, char *argv[])
{
  if (argc == 1) {
    (void)printf("built on        : %s\n", __DATE__);
    (void)printf("debug           : ");
#ifdef NDEBUG
    (void)printf("false\n");
#else /* !NDEBUG */
    (void)printf("true\n");
#endif /* ?NDEBUG */
#ifdef PVN_CILK
    (void)printf("OpenCilk        : %d\n", PVN_CILK);
#else /* !PVN_CILK */
#ifdef _OPENMP
    (void)printf("OpenMP          : %d\n", _OPENMP);
#endif /* ?_OPENMP */
#endif /* ?PVN_CILK */
    (void)printf("endianness      : %s\n", (pvn_le() ? "little" : "big"));
    (void)printf("math_errhandling: %d\n", PVN_FABI(c_math_err,C_MATH_ERR)());
#ifdef PVN_CR_MATH
    (void)printf("PVN_CR_MATH     : %d\n", PVN_CR_MATH);
#endif /* PVN_CR_MATH */
#ifdef PVN_QUADMATH
    (void)printf("PVN_QUADMATH    : %d\n", PVN_QUADMATH);
#endif /* PVN_QUADMATH */
    (void)printf("PVN_CC          : %s\n", PVN_CC);
    (void)printf("PVN_CPPFLAGS    : %s\n", PVN_CPPFLAGS);
    (void)printf("PVN_CFLAGS      : %s\n", PVN_CFLAGS);
    (void)printf("PVN_LIBS        : %s\n", PVN_LIBS);
    (void)printf("PVN_LDFLAGS     : %s\n", PVN_LDFLAGS);
  }
  else {
    struct option longopts[] = {
      { "cc",       no_argument, (int*)NULL, 'C' },
      { "ldflags",  no_argument, (int*)NULL, 'L' },
      { "cflags",   no_argument, (int*)NULL, 'c' },
      { "dynamic",  no_argument, (int*)NULL, 'd' },
      { "cppflags", no_argument, (int*)NULL, 'i' },
      { "libs",     no_argument, (int*)NULL, 'l' },
      { "parallel", no_argument, (int*)NULL, 'p' },
      { (char*)NULL, 0, (int*)NULL, 0 }
    };
    for (int ch = 0; (ch = getopt_long(argc, argv, "CLcdilp", longopts, (int*)NULL)) != -1; ) {
      switch ((char)ch) {
      case 'C':
        (void)printf("%s\n", PVN_CC);
        break;
      case 'L':
        (void)printf("%s\n", PVN_LDFLAGS);
        break;
      case 'c':
        (void)printf("%s\n", PVN_CFLAGS);
        break;
      case 'd':
#if (defined(PVN_DYNAMIC) || (defined(_WIN32) && defined(_DLL)))
        (void)printf("true\n");
#else /* !DYNAMIC */
        (void)printf("false\n");
#endif /* ?DYNAMIC */
        break;
      case 'i':
        (void)printf("%s\n", PVN_CPPFLAGS);
        break;
      case 'l':
        (void)printf("%s\n", PVN_LIBS);
        break;
      case 'p':
#ifdef _OPENMP
        (void)printf("OpenMP=%d\n", _OPENMP);
#else /* !_OPENMP */
#ifdef PVN_CILK
        (void)printf("OpenCilk=%d\n", PVN_CILK);
#else /* !PVN_CILK */
        (void)printf("simd\n");
#endif /* ?PVN_CILK */
#endif /* ?_OPENMP */
        break;
      default:
        goto err;
      }
    }
    argv += optind;
    argc -= optind;
    if (argc)
      goto err;
  }
  return EXIT_SUCCESS;
 err:
  (void)fprintf(stderr, "pvn.exe [-C|--cc] [-L|--ldflags] [-c|--cflags] [-d|--dynamic] [-i|--cppflags] [-l|--libs] [-p|--parallel]\n");
  return EXIT_FAILURE;
}
#else /* !PVN_TEST */
int PVN_FABI(pvn_le,PVN_LE)()
{
  return pvn_le();
}

int PVN_FABI(pvn_omp,PVN_OMP)()
{
  return pvn_omp();
}

int PVN_FABI(c_math_err,C_MATH_ERR)()
{
  return math_errhandling;
}
#endif /* ?PVN_TEST */
