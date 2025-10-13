#include "pvn.h"

#ifdef PVN_TEST
int main(int argc, char *argv[])
{
  if (1 < argc) {
    (void)fprintf(stderr, "%s takes no arguments\n", *argv);
    return EXIT_FAILURE;
  }
  size_t a = (size_t)0u;
  (void)printf("sizeof(pvn_lock_t)=%zu\n", PVN_FABI(pvn_lock_size,PVN_LOCK_SIZE)(&a));
  (void)printf("alignof(pvn_lock_t)=%zu\n", a);
  pvn_lock_t lock;
  (void)printf("pvn_lock_init=%d\n", PVN_FABI(pvn_lock_init,PVN_LOCK_INIT)(&lock));
  (void)printf("pvn_lock=%d\n", PVN_FABI(pvn_lock,PVN_LOCK)(&lock));
  (void)printf("pvn_unlock=%d\n", PVN_FABI(pvn_unlock,PVN_UNLOCK)(&lock));
  (void)printf("pvn_lock_destroy=%d\n", PVN_FABI(pvn_lock_destroy,PVN_LOCK_DESTROY)(&lock));
  return EXIT_SUCCESS;
}
#else /* !PVN_TEST */
#ifdef _WIN32
#include <Windows.h>
static_assert(sizeof(pvn_lock_t) == sizeof(HANDLE), "sizeof(pvn_lock_t) != sizeof(HANDLE)");
#endif /* _WIN32 */

size_t PVN_FABI(pvn_lock_size,PVN_LOCK_SIZE)(size_t *const a)
{
  if (a)
    *a = alignof(pvn_lock_t);
  return sizeof(pvn_lock_t);
}

int PVN_FABI(pvn_lock_init,PVN_LOCK_INIT)(pvn_lock_t *const lock)
{
  PVN_ASSERT(lock);
#ifdef _WIN32
  *lock = CreateMutexA((LPSECURITY_ATTRIBUTES)NULL, FALSE, (LPCSTR)NULL);
  return (*lock != NULL);
#else /* !_WIN32 */
  pthread_mutexattr_t attr;
  PVN_SYSI_CALL(pthread_mutexattr_init(&attr));
  PVN_SYSI_CALL(pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK));
  PVN_SYSI_CALL(pthread_mutex_init(lock, &attr));
  PVN_SYSI_CALL(pthread_mutexattr_destroy(&attr));
  return PTHREAD_MUTEX_ERRORCHECK;
#endif /* ?_WIN32 */
}

int PVN_FABI(pvn_lock_destroy,PVN_LOCK_DESTROY)(pvn_lock_t *const lock)
{
  PVN_ASSERT(lock);
#ifdef _WIN32
  return !CloseHandle(*lock);
#else /* !_WIN32 */
  return pthread_mutex_destroy(lock);
#endif /* ?_WIN32 */
}

int PVN_FABI(pvn_lock,PVN_LOCK)(pvn_lock_t *const lock)
{
  PVN_ASSERT(lock);
#ifdef _WIN32
  switch (WaitForSingleObject(*lock, INFINITE)) {
  case WAIT_OBJECT_0:
    return 0;
  case WAIT_ABANDONED:
    return 1;
  default:
    return -1;
  }
#else /* !_WIN32 */
  return pthread_mutex_lock(lock);
#endif /* ?_WIN32 */
}

int PVN_FABI(pvn_unlock,PVN_UNLOCK)(pvn_lock_t *const lock)
{
  PVN_ASSERT(lock);
#ifdef _WIN32
  return !ReleaseMutex(*lock);
#else /* !_WIN32 */
  return pthread_mutex_unlock(lock);
#endif /* ?_WIN32 */
}
#endif /* ?PVN_TEST */
