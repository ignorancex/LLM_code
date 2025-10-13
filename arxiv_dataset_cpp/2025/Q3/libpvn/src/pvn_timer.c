#include "pvn.h"

#ifdef PVN_TEST
int main(int argc, char *argv[])
{
  if (1 < argc) {
    (void)fprintf(stderr, "%s takes no arguments\n", *argv);
    return EXIT_FAILURE;
  }
#ifdef _WIN32
  (void)printf("QueryPerformanceFrequency: %ld Hz.\n", (long)PVN_FABI(pvn_time_mono_freq,PVN_TIME_MONO_FREQ)());
#else /* !_WIN32 */
  (void)printf("%s resolution: %lld ns.\n", ((PVN_CLOCK_MONOTONIC == CLOCK_MONOTONIC) ? "CLOCK_MONOTONIC" : "CLOCK_MONOTONIC_RAW"), pvn_time_mono_res());
#endif /* ?_WIN32 */
  return EXIT_SUCCESS;
}
#else /* !PVN_TEST */
#ifdef _WIN32
#include <Windows.h>

long long PVN_FABI(pvn_time_mono_ticks,PVN_TIME_MONO_TICKS)()
{
  long long r = 0ll;
  return (QueryPerformanceCounter((LARGE_INTEGER*)&r) ? r : -1ll);
}

long long PVN_FABI(pvn_time_mono_freq,PVN_TIME_MONO_FREQ)()
{
  long long r = 0ll;
  return (QueryPerformanceFrequency((LARGE_INTEGER*)&r) ? r : -1ll);
}
#else /* !_WIN32 */
long long PVN_FABI(pvn_time_real_us,PVN_TIME_REAL_US)()
{
  return pvn_time_real_us();
}

long long PVN_FABI(pvn_time_thread_ns,PVN_TIME_THREAD_NS)()
{
  return pvn_time_thread_ns();
}

long long PVN_FABI(pvn_time_thread_res,PVN_TIME_THREAD_RES)()
{
  return pvn_time_thread_res();
}

long long PVN_FABI(pvn_time_mono_ns,PVN_TIME_MONO_NS)()
{
  return pvn_time_mono_ns();
}

long long PVN_FABI(pvn_time_mono_res,PVN_TIME_MONO_RES)()
{
  return pvn_time_mono_res();
}

long long PVN_FABI(pvn_time_mono_ticks,PVN_TIME_MONO_TICKS)()
{
  return pvn_time_mono_ns();
}

long long PVN_FABI(pvn_time_mono_freq,PVN_TIME_MONO_FREQ)()
{
  return 1000000000ll;
}
#endif /* ?_WIN32 */
#endif /* ?PVN_TEST */
