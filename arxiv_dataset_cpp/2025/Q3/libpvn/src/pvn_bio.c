#include "pvn.h"

#ifdef PVN_TEST
int main(int argc, char *argv[])
{
  if (argc != 3) {
    (void)fprintf(stderr, "%s fn sz\n", *argv);
    return EXIT_FAILURE;
  }
  off_t sz = (off_t)atol(argv[2]);
  int fd = PVN_FABI(pvn_bopen_wo,PVN_BOPEN_WO)(&sz, argv[1]);
  (void)printf("pvn_bopen_wo=%d, sz=%ld\n", fd, (long)sz);
  if (fd >= 0)
    (void)printf("pvn_bclose=%d\n", (fd = PVN_FABI(pvn_bclose,PVN_BCLOSE)(&fd)));
  return ((fd >= 0) ? EXIT_SUCCESS : EXIT_FAILURE);
}
#else /* !PVN_TEST */
static const size_t GiB = (size_t)(1ull << 30u);
#ifdef _WIN32
#include <Windows.h>
static const int perms = (_S_IREAD | _S_IWRITE);
#else /* !_WIN32 */
static const mode_t perms = (S_IRUSR | S_IWUSR);
#endif /* ?_WIN32 */

int PVN_FABI(pvn_bopen_ro,PVN_BOPEN_RO)(off_t *const sz, const char *const fn, ...)
{
#ifdef _WIN32
  const int fd = (fn ? open(fn, _O_RDONLY) : -2);
#else /* !_WIN32 */
  const int fd = (fn ? open(fn, (O_RDONLY | PVN_LF64)) : -2);
#endif /* ?_WIN32 */
  if (fd >= 0) {
    if (sz) {
      struct stat buf;
      if (fstat(fd, &buf) < 0) {
        *sz = close(fd);
        return -1;
      }
      *sz = buf.st_size;
    }
  }
  return fd;
}

int PVN_FABI(pvn_bopen_rw,PVN_BOPEN_RW)(off_t *const sz, const char *const fn, ...)
{
#ifdef _WIN32
  const int fd = (fn ? open(fn, (_O_RDWR | _O_CREAT), perms) : -2);
#else /* !_WIN32 */
  const int fd = (fn ? open(fn, (O_RDWR | O_CREAT | PVN_LF64), perms) : -2);
#endif /* ?_WIN32 */
  if (fd >= 0) {
    if (sz) {
      if (*sz >= 0) {
        if (/* Windows: max 2^31 - 1 bytes */
#ifdef _WIN32
            _chsize(fd, (long)*sz) < 0
#else /* !_WIN32 */
            (ftruncate(fd, *sz) < 0) || (fsync(fd) < 0)
#endif /* ?_WIN32 */
            ) {
          *sz = close(fd);
          return -1;
        }
      }
      else {
        struct stat buf;
        if (fstat(fd, &buf) < 0) {
          *sz = close(fd);
          return -1;
        }
        *sz = buf.st_size;
      }
    }
  }
  return fd;
}

int PVN_FABI(pvn_bopen_wo,PVN_BOPEN_WO)(off_t *const sz, const char *const fn, ...)
{
#ifdef _WIN32
  const int fd = (fn ? open(fn, (_O_WRONLY | _O_CREAT), perms) : -2);
#else /* !_WIN32 */
  const int fd = (fn ? open(fn, (O_WRONLY | O_CREAT | PVN_LF64), perms) : -2);
#endif /* ?_WIN32 */
  if (fd >= 0) {
    if (sz) {
      if (*sz >= 0) {
        if ( /* Windows: max 2^31 - 1 bytes */
#ifdef _WIN32
            _chsize(fd, (long)*sz) < 0
#else /* !_WIN32 */
            (ftruncate(fd, *sz) < 0) || (fsync(fd) < 0)
#endif /* ?_WIN32 */
            ) {
          *sz = close(fd);
          return -1;
        }
      }
      else {
        struct stat buf;
        if (fstat(fd, &buf) < 0) {
          *sz = close(fd);
          return -1;
        }
        *sz = buf.st_size;
      }
    }
  }
  return fd;
}

int PVN_FABI(pvn_bclose,PVN_BCLOSE)(const int *const fd)
{
  return (fd ? ((*fd < 0) ? -3 : close(*fd)) : -2);
}

ssize_t PVN_FABI(pvn_bwrite,PVN_BWRITE)(const int *const fd, const void *const buf, const size_t *const nB, const off_t *const off)
{
  if (!fd || (*fd < 0))
    return -1;
  if (!nB)
    return -3;
  if (*nB && !buf)
    return -2;
  if (!off || (*off < 0))
    return -4;
  if (!*nB)
    return 0;

  const void *_buf = buf;
  size_t _nB = *nB;
  off_t _off = *off;
  const size_t qGiB = (_nB >> 30u);
  ssize_t ret = 0;
#ifdef _WIN32
  const HANDLE h = (HANDLE)_get_osfhandle(*fd);
  if (h == INVALID_HANDLE_VALUE)
    return -1;
  OVERLAPPED o;
  (void)memset(&o, 0, sizeof(o));
#endif /* _WIN32 */
  for (size_t i = 0u; i < qGiB; ++i) {
#ifdef _WIN32
    ssize_t ret1 = 0ll;
    *(off_t*)&(o.Offset) = _off;
    if (!WriteFile(h, _buf, (DWORD)GiB, (LPDWORD)&ret1, &o))
      return -5;
#else /* !_WIN32 */
    const ssize_t ret1 = pwrite(*fd, _buf, GiB, _off);
#endif /* ?_WIN32 */
    if ((ret1 <= 0) || ((size_t)ret1 != GiB))
      return -5;
#ifndef _WIN32
    if (fsync(*fd) < 0)
      return -6;
#endif /* !_WIN32 */
    _buf = (const void*)(((const char*)_buf) + ret1);
    _nB -= ret1;
    _off += ret1;
    ret += ret1;
  }
#ifdef _WIN32
  ssize_t ret1 = 0ll;
  *(off_t*)&(o.Offset) = _off;
  if (!WriteFile(h, _buf, (DWORD)_nB, (LPDWORD)&ret1, &o))
    return -7;
#else /* !_WIN32 */
  const ssize_t ret1 = pwrite(*fd, _buf, _nB, _off);
#endif /* ?_WIN32 */
  if ((ret1 <= 0) || ((size_t)ret1 != _nB))
    return -7;
#ifndef _WIN32
  if (fsync(*fd) < 0)
    return -8;
#endif /* !_WIN32 */
  ret += ret1;
  return ret;
}

ssize_t PVN_FABI(pvn_bread,PVN_BREAD)(const int *const fd, void *const buf, const size_t *const nB, const off_t *const off)
{
  if (!fd || (*fd < 0))
    return -1;
  if (!nB)
    return -3;
  if (*nB && !buf)
    return -2;
  if (!off || (*off < 0))
    return -4;
  if (!*nB)
    return 0;

  void *_buf = buf;
  size_t _nB = *nB;
  off_t _off = *off;
  const size_t qGiB = (_nB >> 30u);
  ssize_t ret = 0;
#ifdef _WIN32
  const HANDLE h = (HANDLE)_get_osfhandle(*fd);
  if (h == INVALID_HANDLE_VALUE)
    return -1;
  OVERLAPPED o;
  (void)memset(&o, 0, sizeof(o));
#endif /* _WIN32 */
  for (size_t i = 0u; i < qGiB; ++i) {
#ifdef _WIN32
    ssize_t ret1 = 0ll;
    *(off_t*)&(o.Offset) = _off;
    if (!ReadFile(h, _buf, (DWORD)GiB, (LPDWORD)&ret1, &o))
      return -5;
#else /* !_WIN32 */
    const ssize_t ret1 = pread(*fd, _buf, GiB, _off);
#endif /* ?_WIN32 */
    if ((ret1 <= 0) || ((size_t)ret1 != GiB))
      return -5;
    _buf = (void*)(((char*)_buf) + ret1);
    _nB -= ret1;
    _off += ret1;
    ret += ret1;
  }
#ifdef _WIN32
  ssize_t ret1 = 0ll;
  *(off_t*)&(o.Offset) = _off;
  if (!ReadFile(h, _buf, (DWORD)_nB, (LPDWORD)&ret1, &o))
    return -7;
#else /* !_WIN32 */
  const ssize_t ret1 = pread(*fd, _buf, _nB, _off);
#endif /* ?_WIN32 */
  if ((ret1 <= 0) || ((size_t)ret1 != _nB))
    return -6;
  ret += ret1;
  return ret;
}

#ifdef __x86_64__
ssize_t PVN_FABI(pvn_bwrite80,PVN_BWRITE80)(const int *const fd, const long double *const buf, const size_t *const n, const off_t *const off)
{
  if (!fd || (*fd < 0))
    return -1;
  if (!n)
    return -3;
  if (*n && !buf)
    return -2;
  if (!off || (*off < 0))
    return -4;
  if (!*n)
    return 0;

  const long double *_buf = buf;
  size_t _n = *n;
  off_t _off = *off;
  ssize_t ret = 0;
#ifdef _WIN32
  const HANDLE h = (HANDLE)_get_osfhandle(*fd);
  if (h == INVALID_HANDLE_VALUE)
    return -1;
  OVERLAPPED o;
  (void)memset(&o, 0, sizeof(o));
#endif /* _WIN32 */
  for (size_t i = 0u; i < _n; ++i) {
#ifdef _WIN32
    ssize_t ret1 = 0ll;
    *(off_t*)&(o.Offset) = _off;
    if (!WriteFile(h, _buf++, (DWORD)10u, (LPDWORD)&ret1, &o))
      return -5;
#else /* !_WIN32 */
    const ssize_t ret1 = pwrite(*fd, _buf++, (size_t)10u, _off);
#endif /* ?_WIN32 */
    if (ret1 != (ssize_t)10)
      return -5;
#ifndef NDEBUG
#ifndef _WIN32
    if (fsync(*fd) < 0)
      return -6;
#endif /* !_WIN32 */
#endif /* !NDEBUG */
    _off += (off_t)ret1;
    ret += ret1;
  }
#ifndef _WIN32
  if (fsync(*fd) < 0)
    return -8;
#endif /* !_WIN32 */
  return ret;
}

ssize_t PVN_FABI(pvn_bread80,PVN_BREAD80)(const int *const fd, long double *const buf, const size_t *const n, const off_t *const off)
{
  if (!fd || (*fd < 0))
    return -1;
  if (!n)
    return -3;
  if (*n && !buf)
    return -2;
  if (!off || (*off < 0))
    return -4;
  if (!*n)
    return 0;

  long double *_buf = buf;
  size_t _n = *n;
  off_t _off = *off;
  ssize_t ret = 0;
#ifdef _WIN32
  const HANDLE h = (HANDLE)_get_osfhandle(*fd);
  if (h == INVALID_HANDLE_VALUE)
    return -1;
  OVERLAPPED o;
  (void)memset(&o, 0, sizeof(o));
#endif /* _WIN32 */
  for (size_t i = 0u; i < _n; ++i) {
#ifdef _WIN32
    ssize_t ret1 = 0ll;
    *(off_t*)&(o.Offset) = _off;
    if (!ReadFile(h, _buf++, (DWORD)10u, (LPDWORD)&ret1, &o))
      return -5;
#else /* !_WIN32 */
    const ssize_t ret1 = pread(*fd, _buf++, (size_t)10u, _off);
#endif /* ?_WIN32 */
    if (ret1 != (ssize_t)10)
      return -5;
    _off += (off_t)ret1;
    ret += ret1;
  }
  return ret;
}
#endif /* __x86_64__ */
#endif /* ?PVN_TEST */
