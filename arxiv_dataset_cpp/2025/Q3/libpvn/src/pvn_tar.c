#include "pvn.h"

#ifdef PVN_TEST
int main(int argc, char *argv[])
{
  if (1 < argc) {
    (void)fprintf(stderr, "%s takes no arguments\n", *argv);
    return EXIT_FAILURE;
  }
#ifdef _WIN32
  (void)fprintf(stderr, "A tar stream is to be written to stdout.\n");
  FILE *const p = stdout;
#else /* !_WIN32 */
  (void)fprintf(stderr, "A tar.gz stream is to be written to stdout.\n");
  FILE *const p = popen("gzip -9cq -", "w");
#endif /* ?_WIN32 */
  if (!p)
    return EXIT_FAILURE;
  const int fd = fileno(p);
  if (fd < 0)
    return EXIT_FAILURE;
  const unsigned sz = 4u;
  int r = EXIT_SUCCESS, ret = 0;
  if ((ret = PVN_FABI(pvn_tar_add_file,PVN_TAR_ADD_FILE)(&fd, "foo", &sz, "bar\n")) < 0)
    r = EXIT_FAILURE;
  (void)fprintf(stderr, "pvn_tar_add_file_=%d\n", ret);
  if ((ret = PVN_FABI(pvn_tar_add_dir,PVN_TAR_ADD_DIR)(&fd, "dir")) < 0)
    r = EXIT_FAILURE;
  (void)fprintf(stderr, "pvn_tar_add_dir_=%d\n", ret);
  if ((ret = PVN_FABI(pvn_tar_add_file,PVN_TAR_ADD_FILE)(&fd, "dir/FOO", &sz, "BAR\n")) < 0)
    r = EXIT_FAILURE;
  (void)fprintf(stderr, "pvn_tar_add_file_=%d\n", ret);
  if ((ret = PVN_FABI(pvn_tar_terminate,PVN_TAR_TERMINATE)(&fd)) < 0)
    r = EXIT_FAILURE;
  (void)fprintf(stderr, "pvn_tar_terminate_=%d\n", ret);
#ifndef _WIN32
  if (pclose(p) < 0)
    r = EXIT_FAILURE;
#endif /* !_WIN32 */
  return r;
}
#else /* !PVN_TEST */
static_assert(sizeof(header_posix_ustar) == 512u, "sizeof(header_posix_ustar) != 512");

int PVN_FABI(pvn_tar_add_file,PVN_TAR_ADD_FILE)(const int *const fd, const char *const fn, const unsigned *const sz, const void *const buf)
{
  if (!sz)
    return -3;
  if (*sz && !buf)
    return -4;
  if (!fd || (*fd < 0))
    return -1;
  unsigned fnl = (fn ? (unsigned)strlen(fn) : 0u);
  if (!fnl || (fnl > 256u))
    return -2;
  unsigned i = fnl;
  while (i) {
    if (fn[--i] == '/')
      --fnl;
    else
      break;
  }
  if (!fnl)
    return -2;
  if ((fnl == 256u) && (fn[155u] != '/'))
    return -2;
  if (((fnl == 1u) && (fn[0u] == '.')) || ((fnl == 2u) && (fn[0u] == '.') && (fn[1u] == '.')))
    return -2;
  /* directory? */
  const int de = (!*sz && buf);
  header_posix_ustar hdr;
  (void)memset(&hdr, 0, sizeof(hdr));
  unsigned ni = 0u;
  for (i = fnl; i; ) {
    if (fn[--i] == '/') {
      ni = (i ? 1u : 0u);
      if (((fnl - i) <= (100u + ni)) && (i <= 155u)) {
        ni += i;
        (void)strncpy(hdr.name, fn + ni, fnl - ni);
        (void)strncpy(hdr.prefix, fn, i);
        ni = (fnl - ni);
        break;
      }
      ni = 0u;
    }
    else if (!i && (fnl <= 100u)) {
      (void)strncpy(hdr.name, fn, fnl);
      ni = fnl;
    }
  }
  if (!ni)
    return -2;
  if (de)
    (hdr.name)[ni] = '/';
  /* never allow setuid/setgid or executable files */
  (void)strcpy(hdr.mode, (de ? "0000777" : "0000666"));
  (void)strcpy(hdr.gid, strcpy(hdr.uid, "0000000"));
  if (sprintf(hdr.size, "%011o", *sz) != 11)
    return -3;
  if (sprintf(hdr.mtime, "%011o", (unsigned)time((time_t*)NULL)) != 11)
    return -5;
  (hdr.typeflag)[0u] = (de ? '5' : '0');
  (void)strcpy(hdr.magic, "ustar");
  (hdr.version)[1u] = (hdr.version)[0u] = '0';
  /* never disclose the real username */
  (void)strcpy(hdr.gname, strcpy(hdr.uname, "root"));
  for (i = 0u; i < 8u; ++i)
    (hdr.checksum)[i] = ' ';
  const unsigned char *const u = (const unsigned char*)&hdr;
  unsigned c = 0u;
  for (i = 0u; i < (unsigned)sizeof(hdr); ++i)
    c += u[i];
  if (sprintf(hdr.checksum, "%06o", c) != 6)
    return -6;
  if (write(*fd, &hdr, sizeof(hdr)) != (ssize_t)sizeof(hdr))
    return -7;
  int ret = 1;
  for (i = 0u; i < *sz; i += (unsigned)sizeof(hdr)) {
    if (write(*fd, memcpy(memset(&hdr, 0, sizeof(hdr)), (const char*)buf + i, pvn_umin(*sz - i, (unsigned)sizeof(hdr))), sizeof(hdr)) != (ssize_t)sizeof(hdr))
      return -(7 + ret);
    ++ret;
  }
  return ret;
}

int PVN_FABI(pvn_tar_add_dir,PVN_TAR_ADD_DIR)(const int *const fd, const char *const dn)
{
  const unsigned sz = 0u;
  return PVN_FABI(pvn_tar_add_file,PVN_TAR_ADD_FILE)(fd, dn, &sz, &sz);
}

int PVN_FABI(pvn_tar_terminate,PVN_TAR_TERMINATE)(const int *const fd)
{
  if (!fd || (*fd < 0))
    return -1;
  header_posix_ustar hdr;
  (void)memset(&hdr, 0, sizeof(hdr));
  if (write(*fd, &hdr, sizeof(hdr)) != (ssize_t)sizeof(hdr))
    return -2;
  if (write(*fd, &hdr, sizeof(hdr)) != (ssize_t)sizeof(hdr))
    return -3;
  return 2;
}
#endif /* ?PVN_TEST */
