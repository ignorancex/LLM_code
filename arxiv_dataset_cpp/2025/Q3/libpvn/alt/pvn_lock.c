#include "pvn.h"

#ifdef PVN_TEST
int main(/* int argc, char *argv[] */)
{
  pvn_lock_t lock;
  pvn_lock_init(&lock, true);
  (void)printf("1st   lock = %ld\n", (long)pvn_lock(&lock));
  (void)printf("2nd   lock = %ld\n", (long)pvn_lock(&lock));
  (void)printf("1st unlock = %ld\n", (long)pvn_unlock(&lock));
  (void)printf("3rd   lock = %ld\n", (long)pvn_lock(&lock));
  pvn_lock_destroy(&lock);
  return EXIT_SUCCESS;
}
#else /* !PVN_TEST */
void pvn_lock_init(pvn_lock_t *const lock, const bool recursive)
{
  PVN_SYSP_CALL(lock);
  lock->mutex_type = (recursive ? PTHREAD_MUTEX_RECURSIVE : PTHREAD_MUTEX_ERRORCHECK);
  lock->count = 0;
  pthread_mutexattr_t attr;
  PVN_SYSI_CALL(pthread_mutexattr_init(&attr));
  PVN_SYSI_CALL(pthread_mutexattr_settype(&attr, lock->mutex_type));
  PVN_SYSI_CALL(pthread_mutex_init(&(lock->mutex), &attr));
  PVN_SYSI_CALL(pthread_mutexattr_destroy(&attr));
}

void pvn_lock_destroy(pvn_lock_t *const lock)
{
  sig_atomic_t c = pvn_lock(lock);
  if (c <= 0)
    return;
  lock->count = -1;
  for (sig_atomic_t j, i = (c - 1); i >= 0; --i)
    if ((j = pvn_unlock(lock)) != -1)
      PVN_SYSI_CALL(j);
  while (true) {
    const int e = pthread_mutex_destroy(&(lock->mutex));
    switch (e) {
    case 0:
      return;
    case EBUSY:
      break;
    default:
      PVN_SYSI_CALL(e);
    }
  }
}

sig_atomic_t pvn_lock(pvn_lock_t *const lock)
{
  if (!lock)
    return -2;
  int e = pthread_mutex_lock(&(lock->mutex));
  switch (e) {
  case 0:
    break;
  case EINVAL:
    return -3;
  case EDEADLK:
    return -4;
  default:
    PVN_SYSI_CALL(e);
  }
  if (lock->count >= 0)
    return ++(lock->count);
  /* else this lock is about to die */
  if ((e = pthread_mutex_unlock(&(lock->mutex))))
    PVN_SYSI_CALL(e);
  return -1;
}

sig_atomic_t pvn_unlock(pvn_lock_t *const lock)
{
  if (!lock)
    return -2;
  if (lock->count > 0)
    --(lock->count);
  else if (!(lock->count))
    return -3;
  const int e = pthread_mutex_unlock(&(lock->mutex));
  switch (e) {
  case 0:
    break;
  case EINVAL:
    if (lock->count >= 0)
      ++(lock->count);
    return -4;
  case EPERM:
    if (lock->count >= 0)
      ++(lock->count);
    return -5;
  default:
    if (lock->count >= 0)
      ++(lock->count);
    PVN_SYSI_CALL(e);
  }
  return (lock->count);
}
#endif /* ?PVN_TEST */
