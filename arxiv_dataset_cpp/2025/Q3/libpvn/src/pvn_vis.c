#include "pvn.h"

#ifndef OPEN_RW
#ifdef _WIN32
#define OPEN_RW (_O_RDWR | _O_CREAT | _O_TRUNC)
#else /* !_WIN32 */
#define OPEN_RW (O_RDWR | O_CREAT | O_TRUNC | PVN_LF64)
#endif /* ?_WIN32 */
#else /* OPEN_RW */
#error OPEN_RW already defined
#endif /* ?OPEN_RW */
#ifndef PERM_RW
#ifdef _WIN32
#define PERM_RW (_S_IREAD | S_IWRITE)
#else /* !_WIN32 */
#define PERM_RW (S_IRUSR | S_IWUSR)
#endif /* ?_WIN32 */
#else /* PERM_RW */
#error PERM_RW already defined
#endif /* ?PERM_RW */

#ifdef PVN_TEST
int main(int argc, char *argv[])
{
#ifdef _WIN32
  if (1 <= argc) {
    (void)fprintf(stderr, "%s cannot be run on Windows\n", *argv);
    return EXIT_FAILURE;
  }
#else /* !_WIN32 */
  if (3 != argc) {
    (void)fprintf(stderr, "%s cnt bpp\n", *argv);
    return EXIT_FAILURE;
  }
#endif /* ?_WIN32 */
  const unsigned m = 200u;
  const unsigned n = 320u;
  const unsigned sx = 2u;
  const unsigned sy = 2u;
  int c = atoi(argv[1u]);
  if (c <= 0)
    return EXIT_FAILURE;
  const unsigned cnt = (unsigned)c;
  c = atoi(argv[2u]);
  if ((c != 0) && (c != 1) && (c != 4) && (c != 8))
    return EXIT_FAILURE;
  const unsigned bpp = (unsigned)(c ? c : 1);
  const size_t ldA = (size_t)m;
  double *const A = (double*)malloc(ldA * (n * sizeof(double)));
  if (!A)
    return EXIT_FAILURE;
  (void)system("rm -f ../etc/*.bmp ../etc/*.dat ../etc/*.gif ../etc/*.out ../etc/*.png");
  (void)fprintf(stderr, "pvn_rvis_start... ");
  pvn_rvis_ctx ctx;
  const char *dat = (const char*)NULL;
  if (c == 0)
    dat = "../etc/pvn_vis_r0.dat";
  else if (c == 1)
    dat = "../etc/pvn_vis_r1.dat";
  else if (c == 4)
    dat = "../etc/pvn_vis_r4.dat";
  else if (c == 8)
    dat = "../etc/pvn_vis_r8.dat";
  else
    return EXIT_FAILURE;
  if (pvn_rvis_start(&ctx, m, n, pvn_rop_id, dat)) {
    (void)fprintf(stderr, "ERROR\n");
    return EXIT_FAILURE;
  }
  (void)fprintf(stderr, "OK\n");
  for (unsigned k = 0u; k < cnt; ++k) {
    for (unsigned j = 0u; j < n; ++j) {
      double *const cA = A + j * ldA;
      for (unsigned i = 0u; i < m; ++i)
        cA[i] = (i + j + k) % 254u;
    }
    (void)fprintf(stderr, "pvn_rvis_frame %u... ", k);
    if (pvn_rvis_frame(&ctx, A, ldA)) {
      (void)fprintf(stderr, "ERROR\n");
      return EXIT_FAILURE;
    }
    (void)fprintf(stderr, "OK\n");
  }
  free(A);
  (void)fprintf(stderr, "pvn_rvis_stop... ");
  int r = 0;
  if (c == 0)
    r = pvn_rvis_stop(&ctx, sx, sy, bpp, "../etc/pvn_vis_r0");
  else if (c == 1)
    r = pvn_rvis_stop(&ctx, sx, sy, bpp, "../etc/pvn_vis_r1");
  else if (c == 4)
    r = pvn_rvis_stop(&ctx, sx, sy, bpp, "../etc/pvn_vis_r4");
  else if (c == 8)
    r = pvn_rvis_stop(&ctx, sx, sy, bpp, "../etc/pvn_vis_r8");
  else
    r = 1;
  (void)fprintf(stderr, "%d\n", r);
  if (r)
    return EXIT_FAILURE;
  (void)fprintf(stderr, "Converting the BMPs to the PNGs with ImageMagick...\n");
  (void)system("for B in ../etc/*.bmp; do magick $B -quality 90 ../etc/`basename $B bmp`png; done");
  (void)fprintf(stderr, "Assembling the APNG animation from the PNGs with apngasm...\n");
  if (c == 0)
    (void)system("apngasm -F -o ../etc/pvn_vis_r0.png -d 1:6 ../etc/*-*.png");
  else if (c == 1)
    (void)system("apngasm -F -o ../etc/pvn_vis_r1.png -d 1:6 ../etc/*-*.png");
  else if (c == 4)
    (void)system("apngasm -F -o ../etc/pvn_vis_r4.png -d 1:6 ../etc/*-*.png");
  else if (c == 8)
    (void)system("apngasm -F -o ../etc/pvn_vis_r8.png -d 1:6 ../etc/*-*.png");
  else
    return EXIT_FAILURE;
  (void)fprintf(stderr, "Converting the APNG to the animated GIF with apng2gif...\n");
  if (c == 0)
    (void)system("apng2gif ../etc/pvn_vis_r0.png ../etc/pvn_vis_r0.gif");
  else if (c == 1)
    (void)system("apng2gif ../etc/pvn_vis_r1.png ../etc/pvn_vis_r1.gif");
  else if (c == 4)
    (void)system("apng2gif ../etc/pvn_vis_r4.png ../etc/pvn_vis_r4.gif");
  else if (c == 8)
    (void)system("apng2gif ../etc/pvn_vis_r8.png ../etc/pvn_vis_r8.gif");
  else
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
#else /* !PVN_TEST */
int pvn_rvis_start_f(pvn_rvis_ctx_f *const ctx, const unsigned m, const unsigned n, const pvn_rop_f op, const char *const fnB)
{
  if (!ctx)
    return -1;
  if (!m)
    return -2;
  if (!n)
    return -3;
  if (!op)
    return -4;
  if (!fnB)
    return -5;
  if (!*fnB)
    return -5;

  ctx->minB =  __builtin_inff();
  ctx->maxB = -__builtin_inff();
  ctx->op = op;
  ctx->B = (float*)NULL;
  ctx->m = m;
  ctx->n = n;
  ctx->cnt = 0u;
  ctx->fdB = -1;

  ctx->B = (float*)malloc(m * (n * sizeof(float)));
  if (ctx->B) {
    ctx->fdB = open(fnB, OPEN_RW, PERM_RW);
    if (-1 == ctx->fdB) {
      free(ctx->B);
      ctx->B = (float*)NULL;
      return 2;
    }
    return 0;
  }
  return 1;
}

pvn_rvis_ctx_f* PVN_FABI(pvn_rvis_start_f,PVN_RVIS_START_F)(const unsigned *const m, const unsigned *const n, const unsigned *const fop, const char *const fnB, ...)
{
  PVN_ASSERT(m);
  PVN_ASSERT(n);
  PVN_ASSERT(fop);
  PVN_ASSERT(fnB);
  pvn_rvis_ctx_f *ret = (pvn_rvis_ctx_f*)NULL;
  pvn_rop_f op = (pvn_rop_f)NULL;
  switch (*fop) {
  case 0u:
    op = pvn_rop_idf;
    break;
  case 1u:
    op = pvn_rop_absf;
    break;
  case 2u:
    op = pvn_rop_lgabsf;
    break;
  case 3u:
    op = pvn_rop_logabsf;
    break;
  default:
    return ret;
  }
  ret = (pvn_rvis_ctx_f*)malloc(sizeof(pvn_rvis_ctx_f));
  if (ret && pvn_rvis_start_f(ret, *m, *n, op, fnB)) {
    free(ret);
    ret = (pvn_rvis_ctx_f*)NULL;
  }
  return ret;
}

int pvn_rvis_start(pvn_rvis_ctx *const ctx, const unsigned m, const unsigned n, const pvn_rop op, const char *const fnB)
{
  if (!ctx)
    return -1;
  if (!m)
    return -2;
  if (!n)
    return -3;
  if (!op)
    return -4;
  if (!fnB)
    return -5;
  if (!*fnB)
    return -5;

  ctx->minB =  __builtin_inf();
  ctx->maxB = -__builtin_inf();
  ctx->op = op;
  ctx->B = (double*)NULL;
  ctx->m = m;
  ctx->n = n;
  ctx->cnt = 0u;
  ctx->fdB = -1;

  ctx->B = (double*)malloc(m * (n * sizeof(double)));
  if (ctx->B) {
    ctx->fdB = open(fnB, OPEN_RW, PERM_RW);
    if (-1 == ctx->fdB) {
      free(ctx->B);
      ctx->B = (double*)NULL;
      return 2;
    }
    return 0;
  }
  return 1;
}

pvn_rvis_ctx* PVN_FABI(pvn_rvis_start,PVN_RVIS_START)(const unsigned *const m, const unsigned *const n, const unsigned *const fop, const char *const fnB, ...)
{
  PVN_ASSERT(m);
  PVN_ASSERT(n);
  PVN_ASSERT(fop);
  PVN_ASSERT(fnB);
  pvn_rvis_ctx *ret = (pvn_rvis_ctx*)NULL;
  pvn_rop op = (pvn_rop)NULL;
  switch (*fop) {
  case 0u:
    op = pvn_rop_id;
    break;
  case 1u:
    op = pvn_rop_abs;
    break;
  case 2u:
    op = pvn_rop_lgabs;
    break;
  case 3u:
    op = pvn_rop_logabs;
    break;
  default:
    return ret;
  }
  ret = (pvn_rvis_ctx*)malloc(sizeof(pvn_rvis_ctx));
  if (ret && pvn_rvis_start(ret, *m, *n, op, fnB)) {
    free(ret);
    ret = (pvn_rvis_ctx*)NULL;
  }
  return ret;
}

#if (!defined(_WIN32) || defined(_DLL))
int pvn_rvis_start_l(pvn_rvis_ctx_l *const ctx, const unsigned m, const unsigned n, const pvn_rop_l op, const char *const fnB)
{
  if (!ctx)
    return -1;
  if (!m)
    return -2;
  if (!n)
    return -3;
  if (!op)
    return -4;
  if (!fnB)
    return -5;
  if (!*fnB)
    return -5;

  ctx->minB =  __builtin_infl();
  ctx->maxB = -__builtin_infl();
  ctx->op = op;
  ctx->B = (long double*)NULL;
  ctx->m = m;
  ctx->n = n;
  ctx->cnt = 0u;
  ctx->fdB = -1;

  ctx->B = (long double*)malloc(m * (n * sizeof(long double)));
  if (ctx->B) {
    ctx->fdB = open(fnB, OPEN_RW, PERM_RW);
    if (-1 == ctx->fdB) {
      free(ctx->B);
      ctx->B = (long double*)NULL;
      return 2;
    }
    return 0;
  }
  return 1;
}

pvn_rvis_ctx_l* PVN_FABI(pvn_rvis_start_l,PVN_RVIS_START_L)(const unsigned *const m, const unsigned *const n, const unsigned *const fop, const char *const fnB, ...)
{
  PVN_ASSERT(m);
  PVN_ASSERT(n);
  PVN_ASSERT(fop);
  PVN_ASSERT(fnB);
  pvn_rvis_ctx_l *ret = (pvn_rvis_ctx_l*)NULL;
  pvn_rop_l op = (pvn_rop_l)NULL;
  switch (*fop) {
  case 0u:
    op = pvn_rop_idl;
    break;
  case 1u:
    op = pvn_rop_absl;
    break;
  case 2u:
    op = pvn_rop_lgabsl;
    break;
  case 3u:
    op = pvn_rop_logabsl;
    break;
  default:
    return ret;
  }
  ret = (pvn_rvis_ctx_l*)malloc(sizeof(pvn_rvis_ctx_l));
  if (ret && pvn_rvis_start_l(ret, *m, *n, op, fnB)) {
    free(ret);
    ret = (pvn_rvis_ctx_l*)NULL;
  }
  return ret;
}
#endif /* !_WIN32 || _DLL */

int pvn_rvis_frame_f(pvn_rvis_ctx_f *const ctx, const float *const restrict A, const size_t ldA)
{
  if (!ctx)
    return -1;
  if (!A)
    return -2;
  if (!ldA)
    return -3;
  size_t sz = (size_t)(ctx->m);
  int ret = (ctx->op)(ctx->m, ctx->n, A, ldA, ctx->B, sz, &(ctx->minB), &(ctx->maxB));
  if (ret)
    return ret;
  sz *= ((ctx->n) * sizeof(float));
  if ((ssize_t)sz != write(ctx->fdB, ctx->B, sz))
    ret = 2;
  else
    ++(ctx->cnt);
  return ret;
}

int PVN_FABI(pvn_rvis_frame_f,PVN_RVIS_FRAME_F)(pvn_rvis_ctx_f *const *const ctx, const float *const restrict A, const size_t *const ldA)
{
  PVN_ASSERT(ctx);
  PVN_ASSERT(A);
  PVN_ASSERT(ldA);
  return pvn_rvis_frame_f(*ctx, A, *ldA);
}

int pvn_rvis_frame(pvn_rvis_ctx *const ctx, const double *const restrict A, const size_t ldA)
{
  if (!ctx)
    return -1;
  if (!A)
    return -2;
  if (!ldA)
    return -3;
  size_t sz = (size_t)(ctx->m);
  int ret = (ctx->op)(ctx->m, ctx->n, A, ldA, ctx->B, sz, &(ctx->minB), &(ctx->maxB));
  if (ret)
    return ret;
  sz *= ((ctx->n) * sizeof(double));
  if ((ssize_t)sz != write(ctx->fdB, ctx->B, sz))
    ret = 2;
  else
    ++(ctx->cnt);
  return ret;
}

int PVN_FABI(pvn_rvis_frame,PVN_RVIS_FRAME)(pvn_rvis_ctx *const *const ctx, const double *const restrict A, const size_t *const ldA)
{
  PVN_ASSERT(ctx);
  PVN_ASSERT(A);
  PVN_ASSERT(ldA);
  return pvn_rvis_frame(*ctx, A, *ldA);
}

#if (!defined(_WIN32) || defined(_DLL))
int pvn_rvis_frame_l(pvn_rvis_ctx_l *const ctx, const long double *const restrict A, const size_t ldA)
{
  if (!ctx)
    return -1;
  if (!A)
    return -2;
  if (!ldA)
    return -3;
  size_t sz = (size_t)(ctx->m);
  int ret = (ctx->op)(ctx->m, ctx->n, A, ldA, ctx->B, sz, &(ctx->minB), &(ctx->maxB));
  if (ret)
    return ret;
  sz *= ((ctx->n) * sizeof(long double));
  if ((ssize_t)sz != write(ctx->fdB, ctx->B, sz))
    ret = 2;
  else
    ++(ctx->cnt);
  return ret;
}

int PVN_FABI(pvn_rvis_frame_l,PVN_RVIS_FRAME_L)(pvn_rvis_ctx_l *const *const ctx, const long double *const restrict A, const size_t *const ldA)
{
  PVN_ASSERT(ctx);
  PVN_ASSERT(A);
  PVN_ASSERT(ldA);
  return pvn_rvis_frame_l(*ctx, A, *ldA);
}
#endif /* !_WIN32 || _DLL */

int pvn_rvis_stop_f(pvn_rvis_ctx_f *const ctx, const unsigned sx, const unsigned sy, const unsigned bppB, const char *const bnB)
{
  if (!ctx)
    return -1;
  if (!sx)
    return -2;
  if (!sy)
    return -3;
  if (!bnB)
    return -5;
  if (!*bnB)
    return -5;

  float ubc, mid, shf;
  if (bppB == 1u) {
    ubc = 1.0f;
    mid = 0.0f;
    shf = 0.0f;
  }
  else if (bppB == 4u) {
    ubc = 15.0f;
    mid = 7.0f;
    shf = 13.0f;
  }
  else if (bppB == 8u) {
    ubc = 255.0f;
    mid = 127.0f;
    shf = 253.0f;
  }
  else
    return -4;
  const float wid = ctx->maxB - ctx->minB;

  const char *fmt = "-%010u.bmp";
  if (ctx->cnt <= 10u)
    fmt = "-%1u.bmp";
  else if (ctx->cnt <= 100u)
    fmt = "-%02u.bmp";
  else if (ctx->cnt <= 1000u)
    fmt = "-%03u.bmp";
  else if (ctx->cnt <= 10000u)
    fmt = "-%04u.bmp";
  else if (ctx->cnt <= 100000u)
    fmt = "-%05u.bmp";
  else if (ctx->cnt <= 1000000u)
    fmt = "-%06u.bmp";
  else if (ctx->cnt <= 10000000u)
    fmt = "-%07u.bmp";
  else if (ctx->cnt <= 100000000u)
    fmt = "-%08u.bmp";
  else if (ctx->cnt <= 1000000000u)
    fmt = "-%09u.bmp";

  const size_t bnl = strlen(bnB);
  char fn[bnl + 16u];
  strcpy(fn, bnB)[bnl] = '.';
  fn[bnl + 1u] = 'c';
  fn[bnl + 2u] = 's';
  fn[bnl + 3u] = 'v';
  fn[bnl + 4u] = '\0';

  pvn_bmp_t bmp = (pvn_bmp_t)NULL;
  pvn_bmp_pixel_setter_t ps = (pvn_bmp_pixel_setter_t)NULL;
  const size_t ldB = (size_t)(ctx->m);
  const size_t sz = ldB * ((ctx->n) * sizeof(float));
  int ret = 0;

  if (pvn_bmp_create(&bmp, (ctx->n * sx), ((int)(ctx->m) * -(int)sy), bppB)) {
    ret = 2;
    goto end;
  }
  if (pvn_bmp_read_cmap(bmp, fn)) {
    ret = 3;
    goto end;
  }
  if (!(ps = pvn_bmp_get_pixel_setter(bmp))) {
    ret = 4;
    goto end;
  }
  if (lseek(ctx->fdB, 0, SEEK_SET)) {
    ret = 5;
    goto end;
  }
  for (unsigned f = 0u; f < ctx->cnt; ++f) {
    if ((ssize_t)sz != read(ctx->fdB, ctx->B, sz)) {
      ret = 6;
      break;
    }
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(ctx,ldB,bppB,ubc,wid,mid,shf)
#endif /* _OPENMP */
    for (unsigned j = 0u; j < ctx->n; ++j) {
      float *const cB = ctx->B + j * ldB;
      for (unsigned i = 0u; i < ctx->m; ++i) {
        float *const b = cB + i;
        if (bppB == 1u) {
          if (*b == 0.0f)
            *b = 0.0f;
          else
            *b = 1.0f;
        }
        else if (!isfinite(*b))
          *b = (PVN_FABI(pvn_signbitf,PVN_SIGNBITF)(b) ? 0.0f : ubc);
        else if (wid == 0.0f)
          *b = mid;
        else
          *b = roundf(fmaf((*b - ctx->minB) / wid, shf, 1.0f));
      }
    }
    for (unsigned j = 0u; j < ctx->n; ++j) {
      const float *const cB = ctx->B + j * ldB;
      const unsigned jsx = j * sx;
      for (unsigned i = 0u; i < ctx->m; ++i) {
        const unsigned isy = i * sy;
        const unsigned c = (unsigned)(cB[i]);
        for (unsigned ii = 0u; ii < sy; ++ii) {
          const unsigned iisy = isy + ii;
          for (unsigned jj = 0u; jj < sx; ++jj)
            ps(bmp, (jsx + jj), iisy, c);
        }
      }
    }
    if (sprintf((fn + bnl), fmt, f) <= 0) {
      ret = 7;
      break;
    }
    if (pvn_bmp_fwrite(bmp, fn)) {
      ret = 8;
      break;
    }
  }
 end:
  if (bmp)
    pvn_bmp_destroy(bmp);

  if (-1 != ctx->fdB) {
    if (close(ctx->fdB))
      ret = 9;
    ctx->fdB = -1;
  }
  if (ctx->B) {
    free(ctx->B);
    ctx->B = (float*)NULL;
  }
  if (ret)
    return ret;
  fn[bnl] = '.';
  fn[bnl + 1u] = 'o';
  fn[bnl + 2u] = 'u';
  fn[bnl + 3u] = 't';
  fn[bnl + 4u] = '\0';
  FILE *const outf = fopen(fn, "w");
  if (!outf)
    return (ret = 10);
  if (fprintf(outf, "m: %u\nn: %u\nsx: %u\nsy: %u\nbpp: %u\ncnt: %u\nmin: %s\nmax: ", ctx->m, ctx->n, sx, sy, bppB, ctx->cnt, pvn_stoa(fn, ctx->minB)) <= 0)
    ret = 11;
  if (fprintf(outf, "%s\n", pvn_stoa(fn, ctx->maxB)) <= 0)
    ret = 12;
  if (fclose(outf))
    ret = 13;
  return ret;
}

int PVN_FABI(pvn_rvis_stop_f,PVN_RVIS_STOP_F)(pvn_rvis_ctx_f *const *const ctx, const unsigned *const sx, const unsigned *const sy, const unsigned *const bppB, const char *const bnB, ...)
{
  PVN_ASSERT(ctx);
  PVN_ASSERT(sx);
  PVN_ASSERT(sy);
  PVN_ASSERT(bppB);
  PVN_ASSERT(bnB);
  const int r = pvn_rvis_stop_f(*ctx, *sx, *sy, *bppB, bnB);
  if (*ctx)
    free(*ctx);
  return r;
}

int pvn_rvis_stop(pvn_rvis_ctx *const ctx, const unsigned sx, const unsigned sy, const unsigned bppB, const char *const bnB)
{
  if (!ctx)
    return -1;
  if (!sx)
    return -2;
  if (!sy)
    return -3;
  if (!bnB)
    return -5;
  if (!*bnB)
    return -5;

  double ubc, mid, shf;
  if (bppB == 1u) {
    ubc = 1.0;
    mid = 0.0;
    shf = 0.0;
  }
  else if (bppB == 4u) {
    ubc = 15.0;
    mid = 7.0;
    shf = 13.0;
  }
  else if (bppB == 8u) {
    ubc = 255.0;
    mid = 127.0;
    shf = 253.0;
  }
  else
    return -4;
  const double wid = ctx->maxB - ctx->minB;

  const char *fmt = "-%010u.bmp";
  if (ctx->cnt <= 10u)
    fmt = "-%1u.bmp";
  else if (ctx->cnt <= 100u)
    fmt = "-%02u.bmp";
  else if (ctx->cnt <= 1000u)
    fmt = "-%03u.bmp";
  else if (ctx->cnt <= 10000u)
    fmt = "-%04u.bmp";
  else if (ctx->cnt <= 100000u)
    fmt = "-%05u.bmp";
  else if (ctx->cnt <= 1000000u)
    fmt = "-%06u.bmp";
  else if (ctx->cnt <= 10000000u)
    fmt = "-%07u.bmp";
  else if (ctx->cnt <= 100000000u)
    fmt = "-%08u.bmp";
  else if (ctx->cnt <= 1000000000u)
    fmt = "-%09u.bmp";

  const size_t bnl = strlen(bnB);
  char fn[bnl + 25u];
  strcpy(fn, bnB)[bnl] = '.';
  fn[bnl + 1u] = 'c';
  fn[bnl + 2u] = 's';
  fn[bnl + 3u] = 'v';
  fn[bnl + 4u] = '\0';

  pvn_bmp_t bmp = (pvn_bmp_t)NULL;
  pvn_bmp_pixel_setter_t ps = (pvn_bmp_pixel_setter_t)NULL;
  const size_t ldB = (size_t)(ctx->m);
  const size_t sz = ldB * ((ctx->n) * sizeof(double));
  int ret = 0;

  if (pvn_bmp_create(&bmp, (ctx->n * sx), ((int)(ctx->m) * -(int)sy), bppB)) {
    ret = 2;
    goto end;
  }
  if (pvn_bmp_read_cmap(bmp, fn)) {
    ret = 3;
    goto end;
  }
  if (!(ps = pvn_bmp_get_pixel_setter(bmp))) {
    ret = 4;
    goto end;
  }
  if (lseek(ctx->fdB, 0, SEEK_SET)) {
    ret = 5;
    goto end;
  }
  for (unsigned f = 0u; f < ctx->cnt; ++f) {
    if ((ssize_t)sz != read(ctx->fdB, ctx->B, sz)) {
      ret = 6;
      break;
    }
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(ctx,ldB,bppB,ubc,wid,mid,shf)
#endif /* _OPENMP */
    for (unsigned j = 0u; j < ctx->n; ++j) {
      double *const cB = ctx->B + j * ldB;
      for (unsigned i = 0u; i < ctx->m; ++i) {
        double *const b = cB + i;
        if (bppB == 1u) {
          if (*b == 0.0)
            *b = 0.0;
          else
            *b = 1.0;
        }
        else if (!isfinite(*b))
          *b = (PVN_FABI(pvn_signbit,PVN_SIGNBIT)(b) ? 0.0 : ubc);
        else if (wid == 0.0)
          *b = mid;
        else
          *b = round(fma((*b - ctx->minB) / wid, shf, 1.0));
      }
    }
    for (unsigned j = 0u; j < ctx->n; ++j) {
      const double *const cB = ctx->B + j * ldB;
      const unsigned jsx = j * sx;
      for (unsigned i = 0u; i < ctx->m; ++i) {
        const unsigned isy = i * sy;
        const unsigned c = (unsigned)(cB[i]);
        for (unsigned ii = 0u; ii < sy; ++ii) {
          const unsigned iisy = isy + ii;
          for (unsigned jj = 0u; jj < sx; ++jj)
            ps(bmp, (jsx + jj), iisy, c);
        }
      }
    }
    if (sprintf((fn + bnl), fmt, f) <= 0) {
      ret = 7;
      break;
    }
    if (pvn_bmp_fwrite(bmp, fn)) {
      ret = 8;
      break;
    }
  }
 end:
  if (bmp)
    pvn_bmp_destroy(bmp);

  if (-1 != ctx->fdB) {
    if (close(ctx->fdB))
      ret = 9;
    ctx->fdB = -1;
  }
  if (ctx->B) {
    free(ctx->B);
    ctx->B = (double*)NULL;
  }
  if (ret)
    return ret;
  fn[bnl] = '.';
  fn[bnl + 1u] = 'o';
  fn[bnl + 2u] = 'u';
  fn[bnl + 3u] = 't';
  fn[bnl + 4u] = '\0';
  FILE *const outf = fopen(fn, "w");
  if (!outf)
    return (ret = 10);
  if (fprintf(outf, "m: %u\nn: %u\nsx: %u\nsy: %u\nbpp: %u\ncnt: %u\nmin: %s\nmax: ", ctx->m, ctx->n, sx, sy, bppB, ctx->cnt, pvn_dtoa(fn, ctx->minB)) <= 0)
    ret = 11;
  if (fprintf(outf, "%s\n", pvn_dtoa(fn, ctx->maxB)) <= 0)
    ret = 12;
  if (fclose(outf))
    ret = 13;
  return ret;
}

int PVN_FABI(pvn_rvis_stop,PVN_RVIS_STOP)(pvn_rvis_ctx *const *const ctx, const unsigned *const sx, const unsigned *const sy, const unsigned *const bppB, const char *const bnB, ...)
{
  PVN_ASSERT(ctx);
  PVN_ASSERT(sx);
  PVN_ASSERT(sy);
  PVN_ASSERT(bppB);
  PVN_ASSERT(bnB);
  const int r = pvn_rvis_stop(*ctx, *sx, *sy, *bppB, bnB);
  if (*ctx)
    free(*ctx);
  return r;
}

#if (!defined(_WIN32) || defined(_DLL))
int pvn_rvis_stop_l(pvn_rvis_ctx_l *const ctx, const unsigned sx, const unsigned sy, const unsigned bppB, const char *const bnB)
{
  if (!ctx)
    return -1;
  if (!sx)
    return -2;
  if (!sy)
    return -3;
  if (!bnB)
    return -5;
  if (!*bnB)
    return -5;

  long double ubc, mid, shf;
  if (bppB == 1u) {
    ubc = 1.0L;
    mid = 0.0L;
    shf = 0.0L;
  }
  else if (bppB == 4u) {
    ubc = 15.0L;
    mid = 7.0L;
    shf = 13.0L;
  }
  else if (bppB == 8u) {
    ubc = 255.0L;
    mid = 127.0L;
    shf = 253.0L;
  }
  else
    return -4;
  const long double wid = ctx->maxB - ctx->minB;

  const char *fmt = "-%010u.bmp";
  if (ctx->cnt <= 10u)
    fmt = "-%1u.bmp";
  else if (ctx->cnt <= 100u)
    fmt = "-%02u.bmp";
  else if (ctx->cnt <= 1000u)
    fmt = "-%03u.bmp";
  else if (ctx->cnt <= 10000u)
    fmt = "-%04u.bmp";
  else if (ctx->cnt <= 100000u)
    fmt = "-%05u.bmp";
  else if (ctx->cnt <= 1000000u)
    fmt = "-%06u.bmp";
  else if (ctx->cnt <= 10000000u)
    fmt = "-%07u.bmp";
  else if (ctx->cnt <= 100000000u)
    fmt = "-%08u.bmp";
  else if (ctx->cnt <= 1000000000u)
    fmt = "-%09u.bmp";

  const size_t bnl = strlen(bnB);
  char fn[bnl +
#ifdef __x86_64__
          30u
#else /* !__x86_64__ */
          45
#endif /* ?__x86_64__ */
          ];
  strcpy(fn, bnB)[bnl] = '.';
  fn[bnl + 1u] = 'c';
  fn[bnl + 2u] = 's';
  fn[bnl + 3u] = 'v';
  fn[bnl + 4u] = '\0';

  pvn_bmp_t bmp = (pvn_bmp_t)NULL;
  pvn_bmp_pixel_setter_t ps = (pvn_bmp_pixel_setter_t)NULL;
  const size_t ldB = (size_t)(ctx->m);
  const size_t sz = ldB * ((ctx->n) * sizeof(long double));
  int ret = 0;

  if (pvn_bmp_create(&bmp, (ctx->n * sx), ((int)(ctx->m) * -(int)sy), bppB)) {
    ret = 2;
    goto err;
  }
  if (pvn_bmp_read_cmap(bmp, fn)) {
    ret = 3;
    goto err;
  }
  if (!(ps = pvn_bmp_get_pixel_setter(bmp))) {
    ret = 4;
    goto err;
  }
  if (lseek(ctx->fdB, 0, SEEK_SET)) {
    ret = 5;
    goto err;
  }
  for (unsigned f = 0u; f < ctx->cnt; ++f) {
    if ((ssize_t)sz != read(ctx->fdB, ctx->B, sz)) {
      ret = 6;
      break;
    }
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(ctx,ldB,bppB,ubc,wid,mid,shf)
#endif /* _OPENMP */
    for (unsigned j = 0u; j < ctx->n; ++j) {
      long double *const cB = ctx->B + j * ldB;
      for (unsigned i = 0u; i < ctx->m; ++i) {
        long double *const b = cB + i;
        if (bppB == 1u) {
          if (*b == 0.0L)
            *b = 0.0L;
          else
            *b = 1.0L;
        }
        else if (!isfinite(*b))
          *b = (PVN_FABI(pvn_signbitl,PVN_SIGNBITL)(b) ? 0.0L : ubc);
        else if (wid == 0.0L)
          *b = mid;
        else
          *b = roundl(fmal((*b - ctx->minB) / wid, shf, 1.0L));
      }
    }
    for (unsigned j = 0u; j < ctx->n; ++j) {
      const long double *const cB = ctx->B + j * ldB;
      const unsigned jsx = j * sx;
      for (unsigned i = 0u; i < ctx->m; ++i) {
        const unsigned isy = i * sy;
        const unsigned c = (unsigned)(cB[i]);
        for (unsigned ii = 0u; ii < sy; ++ii) {
          const unsigned iisy = isy + ii;
          for (unsigned jj = 0u; jj < sx; ++jj)
            ps(bmp, (jsx + jj), iisy, c);
        }
      }
    }
    if (sprintf((fn + bnl), fmt, f) <= 0) {
      ret = 7;
      break;
    }
    if (pvn_bmp_fwrite(bmp, fn)) {
      ret = 8;
      break;
    }
  }
 err:
  if (bmp)
    pvn_bmp_destroy(bmp);

  if (-1 != ctx->fdB) {
    if (close(ctx->fdB))
      ret = 9;
    ctx->fdB = -1;
  }
  if (ctx->B) {
    free(ctx->B);
    ctx->B = (long double*)NULL;
  }
  if (ret)
    return ret;
  fn[bnl] = '.';
  fn[bnl + 1u] = 'o';
  fn[bnl + 2u] = 'u';
  fn[bnl + 3u] = 't';
  fn[bnl + 4u] = '\0';
  FILE *const outf = fopen(fn, "w");
  if (!outf)
    return (ret = 10);
  if (fprintf(outf, "m: %u\nn: %u\nsx: %u\nsy: %u\nbpp: %u\ncnt: %u\nmin: %s\nmax: ", ctx->m, ctx->n, sx, sy, bppB, ctx->cnt, pvn_xtoa(fn, ctx->minB)) <= 0)
    ret = 11;
  if (fprintf(outf, "%s\n", pvn_xtoa(fn, ctx->maxB)) <= 0)
    ret = 12;
  if (fclose(outf))
    ret = 13;
  return ret;
}

int PVN_FABI(pvn_rvis_stop_l,PVN_RVIS_STOP_L)(pvn_rvis_ctx_l *const *const ctx, const unsigned *const sx, const unsigned *const sy, const unsigned *const bppB, const char *const bnB, ...)
{
  PVN_ASSERT(ctx);
  PVN_ASSERT(sx);
  PVN_ASSERT(sy);
  PVN_ASSERT(bppB);
  PVN_ASSERT(bnB);
  const int r = pvn_rvis_stop_l(*ctx, *sx, *sy, *bppB, bnB);
  if (*ctx)
    free(*ctx);
  return r;
}
#endif /* !_WIN32 || _DLL */

int pvn_cvis_start_f(pvn_cvis_ctx_f *const ctx, const unsigned m, const unsigned n, const pvn_cop_f op, const char *const fnB, const char *const fnC)
{
  if (!ctx)
    return -1;
  if (!m)
    return -2;
  if (!n)
    return -3;
  if (!op)
    return -4;
  if (!fnB)
    return -5;
  if (!*fnB)
    return -5;
  if (!fnC)
    return -6;
  if (!*fnC)
    return -6;

  ctx->minC = ctx->minB =  __builtin_inff();
  ctx->maxC = ctx->maxB = -__builtin_inff();
  ctx->op = op;
  ctx->C = ctx->B = (float*)NULL;
  ctx->m = m;
  ctx->n = n;
  ctx->cnt = 0u;
  ctx->fdC = ctx->fdB = -1;
  ctx->err = 0;

  const size_t sz = m * (n * sizeof(float));
  ctx->B = (float*)malloc(sz);
  if (!(ctx->B))
    return 1;
  ctx->C = (float*)malloc(sz);
  if (!(ctx->C)) {
    free(ctx->B);
    ctx->B = (float*)NULL;
    return 2;
  }

  ctx->fdB = open(fnB, OPEN_RW, PERM_RW);
  if (-1 == ctx->fdB) {
    free(ctx->C);
    free(ctx->B);
    ctx->C = ctx->B = (float*)NULL;
    return 3;
  }
  ctx->fdC = open(fnC, OPEN_RW, PERM_RW);
  if (-1 == ctx->fdC) {
    ctx->err = close(ctx->fdB);
    ctx->fdB = -1;
    free(ctx->C);
    free(ctx->B);
    ctx->C = ctx->B = (float*)NULL;
    return (ctx->err ? 5 : 4);
  }
  return 0;
}

pvn_cvis_ctx_f* PVN_FABI(pvn_cvis_start_f,PVN_CVIS_START_F)(const unsigned *const m, const unsigned *const n, const unsigned *const fop, const char *const fnB, const char *const fnC, ...)
{
  pvn_cvis_ctx_f *ret = (pvn_cvis_ctx_f*)NULL;
  if (!m)
    return ret;
  if (!n)
    return ret;
  if (!fop)
    return ret;
  if (!fnB)
    return ret;
  if (!fnC)
    return ret;
  pvn_cop_f op = (pvn_cop_f)NULL;
  switch (*fop) {
  case 0u:
    op = pvn_cop_idf;
    break;
  case 1u:
    op = pvn_cop_absf;
    break;
  case 2u:
    op = pvn_cop_lgabsf;
    break;
  case 3u:
    op = pvn_cop_logabsf;
    break;
  default:
    return ret;
  }
  ret = (pvn_cvis_ctx_f*)malloc(sizeof(pvn_cvis_ctx_f));
  if (ret && pvn_cvis_start_f(ret, *m, *n, op, fnB, fnC)) {
    free(ret);
    ret = (pvn_cvis_ctx_f*)NULL;
  }
  return ret;
}

int pvn_cvis_start(pvn_cvis_ctx *const ctx, const unsigned m, const unsigned n, const pvn_cop op, const char *const fnB, const char *const fnC)
{
  if (!ctx)
    return -1;
  if (!m)
    return -2;
  if (!n)
    return -3;
  if (!op)
    return -4;
  if (!fnB)
    return -5;
  if (!*fnB)
    return -5;
  if (!fnC)
    return -6;
  if (!*fnC)
    return -6;

  ctx->minC = ctx->minB =  __builtin_inf();
  ctx->maxC = ctx->maxB = -__builtin_inf();
  ctx->op = op;
  ctx->C = ctx->B = (double*)NULL;
  ctx->m = m;
  ctx->n = n;
  ctx->cnt = 0u;
  ctx->fdC = ctx->fdB = -1;
  ctx->err = 0;

  const size_t sz = m * (n * sizeof(double));
  ctx->B = (double*)malloc(sz);
  if (!(ctx->B))
    return 1;
  ctx->C = (double*)malloc(sz);
  if (!(ctx->C)) {
    free(ctx->B);
    ctx->B = (double*)NULL;
    return 2;
  }

  ctx->fdB = open(fnB, OPEN_RW, PERM_RW);
  if (-1 == ctx->fdB) {
    free(ctx->C);
    free(ctx->B);
    ctx->C = ctx->B = (double*)NULL;
    return 3;
  }
  ctx->fdC = open(fnC, OPEN_RW, PERM_RW);
  if (-1 == ctx->fdC) {
    ctx->err = close(ctx->fdB);
    ctx->fdB = -1;
    free(ctx->C);
    free(ctx->B);
    ctx->C = ctx->B = (double*)NULL;
    return (ctx->err ? 5 : 4);
  }
  return 0;
}

pvn_cvis_ctx* PVN_FABI(pvn_cvis_start,PVN_CVIS_START)(const unsigned *const m, const unsigned *const n, const unsigned *const fop, const char *const fnB, const char *const fnC, ...)
{
  pvn_cvis_ctx *ret = (pvn_cvis_ctx*)NULL;
  if (!m)
    return ret;
  if (!n)
    return ret;
  if (!fop)
    return ret;
  if (!fnB)
    return ret;
  if (!fnC)
    return ret;
  pvn_cop op = (pvn_cop)NULL;
  switch (*fop) {
  case 0u:
    op = pvn_cop_id;
    break;
  case 1u:
    op = pvn_cop_abs;
    break;
  case 2u:
    op = pvn_cop_lgabs;
    break;
  case 3u:
    op = pvn_cop_logabs;
    break;
  default:
    return ret;
  }
  ret = (pvn_cvis_ctx*)malloc(sizeof(pvn_cvis_ctx));
  if (ret && pvn_cvis_start(ret, *m, *n, op, fnB, fnC)) {
    free(ret);
    ret = (pvn_cvis_ctx*)NULL;
  }
  return ret;
}

#if (!defined(_WIN32) || defined(_DLL))
int pvn_cvis_start_l(pvn_cvis_ctx_l *const ctx, const unsigned m, const unsigned n, const pvn_cop_l op, const char *const fnB, const char *const fnC)
{
  if (!ctx)
    return -1;
  if (!m)
    return -2;
  if (!n)
    return -3;
  if (!op)
    return -4;
  if (!fnB)
    return -5;
  if (!*fnB)
    return -5;
  if (!fnC)
    return -6;
  if (!*fnC)
    return -6;

  ctx->minC = ctx->minB =  __builtin_infl();
  ctx->maxC = ctx->maxB = -__builtin_infl();
  ctx->op = op;
  ctx->C = ctx->B = (long double*)NULL;
  ctx->m = m;
  ctx->n = n;
  ctx->cnt = 0u;
  ctx->fdC = ctx->fdB = -1;
  ctx->err = 0;

  const size_t sz = m * (n * sizeof(long double));
  ctx->B = (long double*)malloc(sz);
  if (!(ctx->B))
    return 1;
  ctx->C = (long double*)malloc(sz);
  if (!(ctx->C)) {
    free(ctx->B);
    ctx->B = (long double*)NULL;
    return 2;
  }

  ctx->fdB = open(fnB, OPEN_RW, PERM_RW);
  if (-1 == ctx->fdB) {
    free(ctx->C);
    free(ctx->B);
    ctx->C = ctx->B = (long double*)NULL;
    return 3;
  }
  ctx->fdC = open(fnC, OPEN_RW, PERM_RW);
  if (-1 == ctx->fdC) {
    ctx->err = close(ctx->fdB);
    ctx->fdB = -1;
    free(ctx->C);
    free(ctx->B);
    ctx->C = ctx->B = (long double*)NULL;
    return (ctx->err ? 5 : 4);
  }
  return 0;
}

pvn_cvis_ctx_l* PVN_FABI(pvn_cvis_start_l,PVN_CVIS_START_L)(const unsigned *const m, const unsigned *const n, const unsigned *const fop, const char *const fnB, const char *const fnC, ...)
{
  pvn_cvis_ctx_l *ret = (pvn_cvis_ctx_l*)NULL;
  if (!m)
    return ret;
  if (!n)
    return ret;
  if (!fop)
    return ret;
  if (!fnB)
    return ret;
  if (!fnC)
    return ret;
  pvn_cop_l op = (pvn_cop_l)NULL;
  switch (*fop) {
  case 0u:
    op = pvn_cop_idl;
    break;
  case 1u:
    op = pvn_cop_absl;
    break;
  case 2u:
    op = pvn_cop_lgabsl;
    break;
  case 3u:
    op = pvn_cop_logabsl;
    break;
  default:
    return ret;
  }
  ret = (pvn_cvis_ctx_l*)malloc(sizeof(pvn_cvis_ctx_l));
  if (ret && pvn_cvis_start_l(ret, *m, *n, op, fnB, fnC)) {
    free(ret);
    ret = (pvn_cvis_ctx_l*)NULL;
  }
  return ret;
}
#endif /* !_WIN32 || _DLL */

int pvn_cvis_frame_f(pvn_cvis_ctx_f *const ctx, const float complex *const restrict A, const size_t ldA)
{
  if (!ctx)
    return -1;
  if (!A)
    return -2;
  if (!ldA)
    return -3;
  size_t sz = (size_t)(ctx->m);
  int ret = (ctx->op)(ctx->m, ctx->n, A, ldA, ctx->B, sz, ctx->C, sz, &(ctx->minB), &(ctx->maxB), &(ctx->minC), &(ctx->maxC));
  if (ret < 0)
    return ret;
  sz *= ((ctx->n) * sizeof(float));
  if ((ssize_t)sz != write(ctx->fdB, ctx->B, sz))
    ret |= 2;
  if ((ssize_t)sz != write(ctx->fdC, ctx->C, sz))
    ret |= 4;
  if (ret <= 1)
    ++(ctx->cnt);
  ctx->err = (ret & 1);
  return ret;
}

int PVN_FABI(pvn_cvis_frame_f,PVN_CVIS_FRAME_F)(pvn_cvis_ctx_f *const *const ctx, const float complex *const restrict A, const size_t *const ldA)
{
  PVN_ASSERT(ctx);
  PVN_ASSERT(A);
  PVN_ASSERT(ldA);
  return pvn_cvis_frame_f(*ctx, A, *ldA);
}

int pvn_cvis_frame(pvn_cvis_ctx *const ctx, const double complex *const restrict A, const size_t ldA)
{
  if (!ctx)
    return -1;
  if (!A)
    return -2;
  if (!ldA)
    return -3;
  size_t sz = (size_t)(ctx->m);
  int ret = (ctx->op)(ctx->m, ctx->n, A, ldA, ctx->B, sz, ctx->C, sz, &(ctx->minB), &(ctx->maxB), &(ctx->minC), &(ctx->maxC));
  if (ret < 0)
    return ret;
  sz *= ((ctx->n) * sizeof(double));
  if ((ssize_t)sz != write(ctx->fdB, ctx->B, sz))
    ret |= 2;
  if ((ssize_t)sz != write(ctx->fdC, ctx->C, sz))
    ret |= 4;
  if (ret <= 1)
    ++(ctx->cnt);
  ctx->err = (ret & 1);
  return ret;
}

int PVN_FABI(pvn_cvis_frame,PVN_CVIS_FRAME)(pvn_cvis_ctx *const *const ctx, const double complex *const restrict A, const size_t *const ldA)
{
  PVN_ASSERT(ctx);
  PVN_ASSERT(A);
  PVN_ASSERT(ldA);
  return pvn_cvis_frame(*ctx, A, *ldA);
}

#if (!defined(_WIN32) || defined(_DLL))
int pvn_cvis_frame_l(pvn_cvis_ctx_l *const ctx, const long double complex *const restrict A, const size_t ldA)
{
  if (!ctx)
    return -1;
  if (!A)
    return -2;
  if (!ldA)
    return -3;
  size_t sz = (size_t)(ctx->m);
  int ret = (ctx->op)(ctx->m, ctx->n, A, ldA, ctx->B, sz, ctx->C, sz, &(ctx->minB), &(ctx->maxB), &(ctx->minC), &(ctx->maxC));
  if (ret < 0)
    return ret;
  sz *= ((ctx->n) * sizeof(long double));
  if ((ssize_t)sz != write(ctx->fdB, ctx->B, sz))
    ret |= 2;
  if ((ssize_t)sz != write(ctx->fdC, ctx->C, sz))
    ret |= 4;
  if (ret <= 1)
    ++(ctx->cnt);
  ctx->err = (ret & 1);
  return ret;
}

int PVN_FABI(pvn_cvis_frame_l,PVN_CVIS_FRAME_L)(pvn_cvis_ctx_l *const *const ctx, const long double complex *const restrict A, const size_t *const ldA)
{
  PVN_ASSERT(ctx);
  PVN_ASSERT(A);
  PVN_ASSERT(ldA);
  return pvn_cvis_frame_l(*ctx, A, *ldA);
}
#endif /* !_WIN32 || _DLL */

int pvn_cvis_stop_f(pvn_cvis_ctx_f *const ctx, const unsigned sx, const unsigned sy, const unsigned bppB, const char *const bnB, const unsigned bppC, const char *const bnC)
{
  if (!ctx)
    return -1;
  if (!sx)
    return -2;
  if (!sy)
    return -3;
  if (!bnB)
    return -5;
  if (!*bnB)
    return -5;
  if (!bnC)
    return -7;
  if (!*bnC)
    return -7;

  int ret = 0;
  pvn_rvis_ctx_f r;
  r.minB = ctx->minB;
  r.maxB = ctx->maxB;
  r.op = (pvn_rop_f)NULL;
  r.B = ctx->B;
  r.m = ctx->m;
  r.n = ctx->n;
  r.cnt = ctx->cnt;
  r.fdB = ctx->fdB;
  if (pvn_rvis_stop_f(&r, sx, sy, bppB, bnB))
    ret |= 2;
  ctx->B = (float*)NULL;
  ctx->fdB = -1;
  r.minB = ctx->minC;
  r.maxB = ctx->maxC;
  r.B = ctx->C;
  r.fdB = ctx->fdC;
  if (pvn_rvis_stop_f(&r, sx, sy, bppC, bnC))
    ret |= 4;
  ctx->C = (float*)NULL;
  ctx->fdC = -1;
  return (ret | ctx->err);
}

int PVN_FABI(pvn_cvis_stop_f,PVN_CVIS_STOP_F)(pvn_cvis_ctx_f *const *const ctx, const unsigned *const sx, const unsigned *const sy, const unsigned *const bppB, const char *const bnB, const unsigned *const bppC, const char *const bnC, ...)
{
  PVN_ASSERT(ctx);
  PVN_ASSERT(sx);
  PVN_ASSERT(sy);
  PVN_ASSERT(bppB);
  PVN_ASSERT(bnB);
  PVN_ASSERT(bppC);
  PVN_ASSERT(bnC);
  const int r = pvn_cvis_stop_f(*ctx, *sx, *sy, *bppB, bnB, *bppC, bnC);
  if (*ctx)
    free(*ctx);
  return r;
}

int pvn_cvis_stop(pvn_cvis_ctx *const ctx, const unsigned sx, const unsigned sy, const unsigned bppB, const char *const bnB, const unsigned bppC, const char *const bnC)
{
  if (!ctx)
    return -1;
  if (!sx)
    return -2;
  if (!sy)
    return -3;
  if (!bnB)
    return -5;
  if (!*bnB)
    return -5;
  if (!bnC)
    return -7;
  if (!*bnC)
    return -7;

  int ret = 0;
  pvn_rvis_ctx r;
  r.minB = ctx->minB;
  r.maxB = ctx->maxB;
  r.op = (pvn_rop)NULL;
  r.B = ctx->B;
  r.m = ctx->m;
  r.n = ctx->n;
  r.cnt = ctx->cnt;
  r.fdB = ctx->fdB;
  if (pvn_rvis_stop(&r, sx, sy, bppB, bnB))
    ret |= 2;
  ctx->B = (double*)NULL;
  ctx->fdB = -1;
  r.minB = ctx->minC;
  r.maxB = ctx->maxC;
  r.B = ctx->C;
  r.fdB = ctx->fdC;
  if (pvn_rvis_stop(&r, sx, sy, bppC, bnC))
    ret |= 4;
  ctx->C = (double*)NULL;
  ctx->fdC = -1;
  return (ret | ctx->err);
}

int PVN_FABI(pvn_cvis_stop,PVN_CVIS_STOP)(pvn_cvis_ctx *const *const ctx, const unsigned *const sx, const unsigned *const sy, const unsigned *const bppB, const char *const bnB, const unsigned *const bppC, const char *const bnC, ...)
{
  PVN_ASSERT(ctx);
  PVN_ASSERT(sx);
  PVN_ASSERT(sy);
  PVN_ASSERT(bppB);
  PVN_ASSERT(bnB);
  PVN_ASSERT(bppC);
  PVN_ASSERT(bnC);
  const int r = pvn_cvis_stop(*ctx, *sx, *sy, *bppB, bnB, *bppC, bnC);
  if (*ctx)
    free(*ctx);
  return r;
}

#if (!defined(_WIN32) || defined(_DLL))
int pvn_cvis_stop_l(pvn_cvis_ctx_l *const ctx, const unsigned sx, const unsigned sy, const unsigned bppB, const char *const bnB, const unsigned bppC, const char *const bnC)
{
  if (!ctx)
    return -1;
  if (!sx)
    return -2;
  if (!sy)
    return -3;
  if (!bnB)
    return -5;
  if (!*bnB)
    return -5;
  if (!bnC)
    return -7;
  if (!*bnC)
    return -7;

  int ret = 0;
  pvn_rvis_ctx_l r;
  r.minB = ctx->minB;
  r.maxB = ctx->maxB;
  r.op = (pvn_rop_l)NULL;
  r.B = ctx->B;
  r.m = ctx->m;
  r.n = ctx->n;
  r.cnt = ctx->cnt;
  r.fdB = ctx->fdB;
  if (pvn_rvis_stop_l(&r, sx, sy, bppB, bnB))
    ret |= 2;
  ctx->B = (long double*)NULL;
  ctx->fdB = -1;
  r.minB = ctx->minC;
  r.maxB = ctx->maxC;
  r.B = ctx->C;
  r.fdB = ctx->fdC;
  if (pvn_rvis_stop_l(&r, sx, sy, bppC, bnC))
    ret |= 4;
  ctx->C = (long double*)NULL;
  ctx->fdC = -1;
  return (ret | ctx->err);
}

int PVN_FABI(pvn_cvis_stop_l,PVN_CVIS_STOP_L)(pvn_cvis_ctx_l *const *const ctx, const unsigned *const sx, const unsigned *const sy, const unsigned *const bppB, const char *const bnB, const unsigned *const bppC, const char *const bnC, ...)
{
  PVN_ASSERT(ctx);
  PVN_ASSERT(sx);
  PVN_ASSERT(sy);
  PVN_ASSERT(bppB);
  PVN_ASSERT(bnB);
  PVN_ASSERT(bppC);
  PVN_ASSERT(bnC);
  const int r = pvn_cvis_stop_l(*ctx, *sx, *sy, *bppB, bnB, *bppC, bnC);
  if (*ctx)
    free(*ctx);
  return r;
}
#endif /* !_WIN32 || _DLL */

int pvn_rop_idf(const unsigned m, const unsigned n, const float *const restrict A, const size_t ldA, float *const restrict B, const size_t ldB, float *const restrict minB, float *const restrict maxB)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!minB)
    return -7;
  if (!maxB)
    return -8;
  if (m > pvn_umin(ldA, ldB))
    return -9;
  float mB = __builtin_inff(), MB = __builtin_inff();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB) reduction(min:mB,MB)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const float *const cA = A + j * ldA;
    float *const cB = B + j * ldB;
    for (unsigned i = 0u; i < m; ++i) {
      const float x = (cB[i] = cA[i]);
      if (isfinite(x)) {
        mB = fminf(mB,  x);
        MB = fminf(MB, -x);
      }
    }
  }
  MB = -MB;
  if (isfinite(mB))
    *minB = fminf(*minB, mB);
  if (isfinite(MB))
    *maxB = fmaxf(*maxB, MB);
  return 0;
}

int pvn_rop_id(const unsigned m, const unsigned n, const double *const restrict A, const size_t ldA, double *const restrict B, const size_t ldB, double *const restrict minB, double *const restrict maxB)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!minB)
    return -7;
  if (!maxB)
    return -8;
  if (m > pvn_umin(ldA, ldB))
    return -9;
  double mB = __builtin_inf(), MB = __builtin_inf();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB) reduction(min:mB,MB)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const double *const cA = A + j * ldA;
    double *const cB = B + j * ldB;
    for (unsigned i = 0u; i < m; ++i) {
      const double x = (cB[i] = cA[i]);
      if (isfinite(x)) {
        mB = fmin(mB,  x);
        MB = fmin(MB, -x);
      }
    }
  }
  MB = -MB;
  if (isfinite(mB))
    *minB = fmin(*minB, mB);
  if (isfinite(MB))
    *maxB = fmax(*maxB, MB);
  return 0;
}

int pvn_rop_idl(const unsigned m, const unsigned n, const long double *const restrict A, const size_t ldA, long double *const restrict B, const size_t ldB, long double *const restrict minB, long double *const restrict maxB)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!minB)
    return -7;
  if (!maxB)
    return -8;
  if (m > pvn_umin(ldA, ldB))
    return -9;
  long double mB = __builtin_infl(), MB = __builtin_infl();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB) reduction(min:mB,MB)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const long double *const cA = A + j * ldA;
    long double *const cB = B + j * ldB;
    for (unsigned i = 0u; i < m; ++i) {
      const long double x = (cB[i] = cA[i]);
      if (isfinite(x)) {
        mB = fminl(mB,  x);
        MB = fminl(MB, -x);
      }
    }
  }
  MB = -MB;
  if (isfinite(mB))
    *minB = fminl(*minB, mB);
  if (isfinite(MB))
    *maxB = fmaxl(*maxB, MB);
  return 0;
}

int pvn_rop_absf(const unsigned m, const unsigned n, const float *const restrict A, const size_t ldA, float *const restrict B, const size_t ldB, float *const restrict minB, float *const restrict maxB)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!minB)
    return -7;
  if (!maxB)
    return -8;
  if (m > pvn_umin(ldA, ldB))
    return -9;
  float mB = __builtin_inff(), MB = __builtin_inff();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB) reduction(min:mB,MB)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const float *const cA = A + j * ldA;
    float *const cB = B + j * ldB;
    for (unsigned i = 0u; i < m; ++i) {
      const float x = (cB[i] = fabsf(cA[i]));
      if (isfinite(x)) {
        mB = fminf(mB,  x);
        MB = fminf(MB, -x);
      }
    }
  }
  MB = -MB;
  if (isfinite(mB))
    *minB = fminf(*minB, mB);
  if (isfinite(MB))
    *maxB = fmaxf(*maxB, MB);
  return 0;
}

int pvn_rop_abs(const unsigned m, const unsigned n, const double *const restrict A, const size_t ldA, double *const restrict B, const size_t ldB, double *const restrict minB, double *const restrict maxB)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!minB)
    return -7;
  if (!maxB)
    return -8;
  if (m > pvn_umin(ldA, ldB))
    return -9;
  double mB = __builtin_inf(), MB = __builtin_inf();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB) reduction(min:mB,MB)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const double *const cA = A + j * ldA;
    double *const cB = B + j * ldB;
    for (unsigned i = 0u; i < m; ++i) {
      const double x = (cB[i] = fabs(cA[i]));
      if (isfinite(x)) {
        mB = fmin(mB,  x);
        MB = fmin(MB, -x);
      }
    }
  }
  MB = -MB;
  if (isfinite(mB))
    *minB = fmin(*minB, mB);
  if (isfinite(MB))
    *maxB = fmax(*maxB, MB);
  return 0;
}

int pvn_rop_absl(const unsigned m, const unsigned n, const long double *const restrict A, const size_t ldA, long double *const restrict B, const size_t ldB, long double *const restrict minB, long double *const restrict maxB)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!minB)
    return -7;
  if (!maxB)
    return -8;
  if (m > pvn_umin(ldA, ldB))
    return -9;
  long double mB = __builtin_infl(), MB = __builtin_infl();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB) reduction(min:mB,MB)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const long double *const cA = A + j * ldA;
    long double *const cB = B + j * ldB;
    for (unsigned i = 0u; i < m; ++i) {
      const long double x = (cB[i] = fabsl(cA[i]));
      if (isfinite(x)) {
        mB = fminl(mB,  x);
        MB = fminl(MB, -x);
      }
    }
  }
  MB = -MB;
  if (isfinite(mB))
    *minB = fminl(*minB, mB);
  if (isfinite(MB))
    *maxB = fmaxl(*maxB, MB);
  return 0;
}

int pvn_rop_lgabsf(const unsigned m, const unsigned n, const float *const restrict A, const size_t ldA, float *const restrict B, const size_t ldB, float *const restrict minB, float *const restrict maxB)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!minB)
    return -7;
  if (!maxB)
    return -8;
  if (m > pvn_umin(ldA, ldB))
    return -9;
  float mB = __builtin_inff(), MB = __builtin_inff();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB) reduction(min:mB,MB)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const float *const cA = A + j * ldA;
    float *const cB = B + j * ldB;
    for (unsigned i = 0u; i < m; ++i) {
      const float x = (cB[i] = log2f(fabsf(cA[i])));
      if (isfinite(x)) {
        mB = fminf(mB,  x);
        MB = fminf(MB, -x);
      }
    }
  }
  MB = -MB;
  if (isfinite(mB))
    *minB = fminf(*minB, mB);
  if (isfinite(MB))
    *maxB = fmaxf(*maxB, MB);
  return 0;
}

int pvn_rop_lgabs(const unsigned m, const unsigned n, const double *const restrict A, const size_t ldA, double *const restrict B, const size_t ldB, double *const restrict minB, double *const restrict maxB)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!minB)
    return -7;
  if (!maxB)
    return -8;
  if (m > pvn_umin(ldA, ldB))
    return -9;
  double mB = __builtin_inf(), MB = __builtin_inf();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB) reduction(min:mB,MB)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const double *const cA = A + j * ldA;
    double *const cB = B + j * ldB;
    for (unsigned i = 0u; i < m; ++i) {
      const double x = (cB[i] = log2(fabs(cA[i])));
      if (isfinite(x)) {
        mB = fmin(mB,  x);
        MB = fmin(MB, -x);
      }
    }
  }
  MB = -MB;
  if (isfinite(mB))
    *minB = fmin(*minB, mB);
  if (isfinite(MB))
    *maxB = fmax(*maxB, MB);
  return 0;
}

int pvn_rop_lgabsl(const unsigned m, const unsigned n, const long double *const restrict A, const size_t ldA, long double *const restrict B, const size_t ldB, long double *const restrict minB, long double *const restrict maxB)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!minB)
    return -7;
  if (!maxB)
    return -8;
  if (m > pvn_umin(ldA, ldB))
    return -9;
  long double mB = __builtin_infl(), MB = __builtin_infl();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB) reduction(min:mB,MB)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const long double *const cA = A + j * ldA;
    long double *const cB = B + j * ldB;
    for (unsigned i = 0u; i < m; ++i) {
      const long double x = (cB[i] = log2l(fabsl(cA[i])));
      if (isfinite(x)) {
        mB = fminl(mB,  x);
        MB = fminl(MB, -x);
      }
    }
  }
  MB = -MB;
  if (isfinite(mB))
    *minB = fminl(*minB, mB);
  if (isfinite(MB))
    *maxB = fmaxl(*maxB, MB);
  return 0;
}

int pvn_rop_logabsf(const unsigned m, const unsigned n, const float *const restrict A, const size_t ldA, float *const restrict B, const size_t ldB, float *const restrict minB, float *const restrict maxB)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!minB)
    return -7;
  if (!maxB)
    return -8;
  if (m > pvn_umin(ldA, ldB))
    return -9;
  float mB = __builtin_inff(), MB = __builtin_inff();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB) reduction(min:mB,MB)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const float *const cA = A + j * ldA;
    float *const cB = B + j * ldB;
    for (unsigned i = 0u; i < m; ++i) {
      const float x = (cB[i] = log10f(fabsf(cA[i])));
      if (isfinite(x)) {
        mB = fminf(mB,  x);
        MB = fminf(MB, -x);
      }
    }
  }
  MB = -MB;
  if (isfinite(mB))
    *minB = fminf(*minB, mB);
  if (isfinite(MB))
    *maxB = fmaxf(*maxB, MB);
  return 0;
}

int pvn_rop_logabs(const unsigned m, const unsigned n, const double *const restrict A, const size_t ldA, double *const restrict B, const size_t ldB, double *const restrict minB, double *const restrict maxB)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!minB)
    return -7;
  if (!maxB)
    return -8;
  if (m > pvn_umin(ldA, ldB))
    return -9;
  double mB = __builtin_inf(), MB = __builtin_inf();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB) reduction(min:mB,MB)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const double *const cA = A + j * ldA;
    double *const cB = B + j * ldB;
    for (unsigned i = 0u; i < m; ++i) {
      const double x = (cB[i] = log10(fabs(cA[i])));
      if (isfinite(x)) {
        mB = fmin(mB,  x);
        MB = fmin(MB, -x);
      }
    }
  }
  MB = -MB;
  if (isfinite(mB))
    *minB = fmin(*minB, mB);
  if (isfinite(MB))
    *maxB = fmax(*maxB, MB);
  return 0;
}

int pvn_rop_logabsl(const unsigned m, const unsigned n, const long double *const restrict A, const size_t ldA, long double *const restrict B, const size_t ldB, long double *const restrict minB, long double *const restrict maxB)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!minB)
    return -7;
  if (!maxB)
    return -8;
  if (m > pvn_umin(ldA, ldB))
    return -9;
  long double mB = __builtin_infl(), MB = __builtin_infl();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB) reduction(min:mB,MB)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const long double *const cA = A + j * ldA;
    long double *const cB = B + j * ldB;
    for (unsigned i = 0u; i < m; ++i) {
      const long double x = (cB[i] = log10l(fabsl(cA[i])));
      if (isfinite(x)) {
        mB = fminl(mB,  x);
        MB = fminl(MB, -x);
      }
    }
  }
  MB = -MB;
  if (isfinite(mB))
    *minB = fminl(*minB, mB);
  if (isfinite(MB))
    *maxB = fmaxl(*maxB, MB);
  return 0;
}

int pvn_cop_idf(const unsigned m, const unsigned n, const float complex *const restrict A, const size_t ldA, float *const restrict B, const size_t ldB, float *const restrict C, const size_t ldC, float *const restrict minB, float *const restrict maxB, float *const restrict minC, float *const restrict maxC)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!C)
    return -7;
  if (!ldC)
    return -8;
  if (!minB)
    return -9;
  if (!maxB)
    return -10;
  if (!minC)
    return -11;
  if (!maxC)
    return -12;
  if (m > pvn_umin(pvn_umin(ldA, ldB), ldC))
    return -13;
  float mB = __builtin_inff(), MB = __builtin_inff(), mC = __builtin_inff(), MC = __builtin_inff();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB,C,ldC) reduction(min:mB,MB,mC,MC)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const float complex *const cA = A + j * ldA;
    float *const cB = B + j * ldB;
    float *const cC = C + j * ldC;
    for (unsigned i = 0u; i < m; ++i) {
      const float complex z = cA[i];
      const float x = (cB[i] = crealf(z));
      const float y = (cC[i] = cimagf(z));
      if (isfinite(x)) {
        mB = fminf(mB,  x);
        MB = fminf(MB, -x);
      }
      if (isfinite(y)) {
        mC = fminf(mC,  y);
        MC = fminf(MC, -y);
      }
    }
  }
  MB = -MB;
  MC = -MC;
  if (isfinite(mB))
    *minB = fminf(*minB, mB);
  if (isfinite(MB))
    *maxB = fmaxf(*maxB, MB);
  if (isfinite(mC))
    *minC = fminf(*minC, mC);
  if (isfinite(MC))
    *maxC = fmaxf(*maxC, MC);
  return 0;
}

int pvn_cop_id(const unsigned m, const unsigned n, const double complex *const restrict A, const size_t ldA, double *const restrict B, const size_t ldB, double *const restrict C, const size_t ldC, double *const restrict minB, double *const restrict maxB, double *const restrict minC, double *const restrict maxC)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!C)
    return -7;
  if (!ldC)
    return -8;
  if (!minB)
    return -9;
  if (!maxB)
    return -10;
  if (!minC)
    return -11;
  if (!maxC)
    return -12;
  if (m > pvn_umin(pvn_umin(ldA, ldB), ldC))
    return -13;
  double mB = __builtin_inf(), MB = __builtin_inf(), mC = __builtin_inf(), MC = __builtin_inf();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB,C,ldC) reduction(min:mB,MB,mC,MC)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const double complex *const cA = A + j * ldA;
    double *const cB = B + j * ldB;
    double *const cC = C + j * ldC;
    for (unsigned i = 0u; i < m; ++i) {
      const double complex z = cA[i];
      const double x = (cB[i] = creal(z));
      const double y = (cC[i] = cimag(z));
      if (isfinite(x)) {
        mB = fmin(mB,  x);
        MB = fmin(MB, -x);
      }
      if (isfinite(y)) {
        mC = fmin(mC,  y);
        MC = fmin(MC, -y);
      }
    }
  }
  MB = -MB;
  MC = -MC;
  if (isfinite(mB))
    *minB = fmin(*minB, mB);
  if (isfinite(MB))
    *maxB = fmax(*maxB, MB);
  if (isfinite(mC))
    *minC = fmin(*minC, mC);
  if (isfinite(MC))
    *maxC = fmax(*maxC, MC);
  return 0;
}

int pvn_cop_idl(const unsigned m, const unsigned n, const long double complex *const restrict A, const size_t ldA, long double *const restrict B, const size_t ldB, long double *const restrict C, const size_t ldC, long double *const restrict minB, long double *const restrict maxB, long double *const restrict minC, long double *const restrict maxC)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!C)
    return -7;
  if (!ldC)
    return -8;
  if (!minB)
    return -9;
  if (!maxB)
    return -10;
  if (!minC)
    return -11;
  if (!maxC)
    return -12;
  if (m > pvn_umin(pvn_umin(ldA, ldB), ldC))
    return -13;
  long double mB = __builtin_infl(), MB = __builtin_infl(), mC = __builtin_infl(), MC = __builtin_infl();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB,C,ldC) reduction(min:mB,MB,mC,MC)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const long double complex *const cA = A + j * ldA;
    long double *const cB = B + j * ldB;
    long double *const cC = C + j * ldC;
    for (unsigned i = 0u; i < m; ++i) {
      const long double complex z = cA[i];
      const long double x = (cB[i] = creall(z));
      const long double y = (cC[i] = cimagl(z));
      if (isfinite(x)) {
        mB = fminl(mB,  x);
        MB = fminl(MB, -x);
      }
      if (isfinite(y)) {
        mC = fminl(mC,  y);
        MC = fminl(MC, -y);
      }
    }
  }
  MB = -MB;
  MC = -MC;
  if (isfinite(mB))
    *minB = fminl(*minB, mB);
  if (isfinite(MB))
    *maxB = fmaxl(*maxB, MB);
  if (isfinite(mC))
    *minC = fminl(*minC, mC);
  if (isfinite(MC))
    *maxC = fmaxl(*maxC, MC);
  return 0;
}

int pvn_cop_absf(const unsigned m, const unsigned n, const float complex *const restrict A, const size_t ldA, float *const restrict B, const size_t ldB, float *const restrict C, const size_t ldC, float *const restrict minB, float *const restrict maxB, float *const restrict minC, float *const restrict maxC)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!C)
    return -7;
  if (!ldC)
    return -8;
  if (!minB)
    return -9;
  if (!maxB)
    return -10;
  if (!minC)
    return -11;
  if (!maxC)
    return -12;
  if (m > pvn_umin(pvn_umin(ldA, ldB), ldC))
    return -13;
  float mB = __builtin_inff(), MB = __builtin_inff(), mC = __builtin_inff(), MC = __builtin_inff();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB,C,ldC) reduction(min:mB,MB,mC,MC)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const float complex *const cA = A + j * ldA;
    float *const cB = B + j * ldB;
    float *const cC = C + j * ldC;
    for (unsigned i = 0u; i < m; ++i) {
      const float complex z = cA[i];
      const float x = (cB[i] = cabsf(z));
      const float y = (cC[i] = cargf(z));
      if (isfinite(x)) {
        mB = fminf(mB,  x);
        MB = fminf(MB, -x);
      }
      if (isfinite(y)) {
        mC = fminf(mC,  y);
        MC = fminf(MC, -y);
      }
    }
  }
  MB = -MB;
  MC = -MC;
  if (isfinite(mB))
    *minB = fminf(*minB, mB);
  if (isfinite(MB))
    *maxB = fmaxf(*maxB, MB);
  if (isfinite(mC))
    *minC = fminf(*minC, mC);
  if (isfinite(MC))
    *maxC = fmaxf(*maxC, MC);
  /* check for overflow of |z| */
  return (MB == __builtin_inff());
}

int pvn_cop_abs(const unsigned m, const unsigned n, const double complex *const restrict A, const size_t ldA, double *const restrict B, const size_t ldB, double *const restrict C, const size_t ldC, double *const restrict minB, double *const restrict maxB, double *const restrict minC, double *const restrict maxC)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!C)
    return -7;
  if (!ldC)
    return -8;
  if (!minB)
    return -9;
  if (!maxB)
    return -10;
  if (!minC)
    return -11;
  if (!maxC)
    return -12;
  if (m > pvn_umin(pvn_umin(ldA, ldB), ldC))
    return -13;
  double mB = __builtin_inf(), MB = __builtin_inf(), mC = __builtin_inf(), MC = __builtin_inf();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB,C,ldC) reduction(min:mB,MB,mC,MC)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const double complex *const cA = A + j * ldA;
    double *const cB = B + j * ldB;
    double *const cC = C + j * ldC;
    for (unsigned i = 0u; i < m; ++i) {
      const double complex z = cA[i];
      const double x = (cB[i] = cabs(z));
      const double y = (cC[i] = carg(z));
      if (isfinite(x)) {
        mB = fmin(mB,  x);
        MB = fmin(MB, -x);
      }
      if (isfinite(y)) {
        mC = fmin(mC,  y);
        MC = fmin(MC, -y);
      }
    }
  }
  MB = -MB;
  MC = -MC;
  if (isfinite(mB))
    *minB = fmin(*minB, mB);
  if (isfinite(MB))
    *maxB = fmax(*maxB, MB);
  if (isfinite(mC))
    *minC = fmin(*minC, mC);
  if (isfinite(MC))
    *maxC = fmax(*maxC, MC);
  /* check for overflow of |z| */
  return (MB == __builtin_inf());
}

int pvn_cop_absl(const unsigned m, const unsigned n, const long double complex *const restrict A, const size_t ldA, long double *const restrict B, const size_t ldB, long double *const restrict C, const size_t ldC, long double *const restrict minB, long double *const restrict maxB, long double *const restrict minC, long double *const restrict maxC)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!C)
    return -7;
  if (!ldC)
    return -8;
  if (!minB)
    return -9;
  if (!maxB)
    return -10;
  if (!minC)
    return -11;
  if (!maxC)
    return -12;
  if (m > pvn_umin(pvn_umin(ldA, ldB), ldC))
    return -13;
  long double mB = __builtin_infl(), MB = __builtin_infl(), mC = __builtin_infl(), MC = __builtin_infl();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB,C,ldC) reduction(min:mB,MB,mC,MC)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const long double complex *const cA = A + j * ldA;
    long double *const cB = B + j * ldB;
    long double *const cC = C + j * ldC;
    for (unsigned i = 0u; i < m; ++i) {
      const long double complex z = cA[i];
      const long double x = (cB[i] = cabsl(z));
      const long double y = (cC[i] = cargl(z));
      if (isfinite(x)) {
        mB = fminl(mB,  x);
        MB = fminl(MB, -x);
      }
      if (isfinite(y)) {
        mC = fminl(mC,  y);
        MC = fminl(MC, -y);
      }
    }
  }
  MB = -MB;
  MC = -MC;
  if (isfinite(mB))
    *minB = fminl(*minB, mB);
  if (isfinite(MB))
    *maxB = fmaxl(*maxB, MB);
  if (isfinite(mC))
    *minC = fminl(*minC, mC);
  if (isfinite(MC))
    *maxC = fmaxl(*maxC, MC);
  /* check for overflow of |z| */
  return (MB == __builtin_infl());
}

int pvn_cop_lgabsf(const unsigned m, const unsigned n, const float complex *const restrict A, const size_t ldA, float *const restrict B, const size_t ldB, float *const restrict C, const size_t ldC, float *const restrict minB, float *const restrict maxB, float *const restrict minC, float *const restrict maxC)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!C)
    return -7;
  if (!ldC)
    return -8;
  if (!minB)
    return -9;
  if (!maxB)
    return -10;
  if (!minC)
    return -11;
  if (!maxC)
    return -12;
  if (m > pvn_umin(pvn_umin(ldA, ldB), ldC))
    return -13;
  float mB = __builtin_inff(), MB = __builtin_inff(), mC = __builtin_inff(), MC = __builtin_inff();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB,C,ldC) reduction(min:mB,MB,mC,MC)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const float complex *const cA = A + j * ldA;
    float *const cB = B + j * ldB;
    float *const cC = C + j * ldC;
    for (unsigned i = 0u; i < m; ++i) {
      const float complex z = cA[i];
      const float x = (cB[i] = log2f(cabsf(z)));
      const float y = (cC[i] = cargf(z));
      if (isfinite(x)) {
        mB = fminf(mB,  x);
        MB = fminf(MB, -x);
      }
      if (isfinite(y)) {
        mC = fminf(mC,  y);
        MC = fminf(MC, -y);
      }
    }
  }
  MB = -MB;
  MC = -MC;
  if (isfinite(mB))
    *minB = fminf(*minB, mB);
  if (isfinite(MB))
    *maxB = fmaxf(*maxB, MB);
  if (isfinite(mC))
    *minC = fminf(*minC, mC);
  if (isfinite(MC))
    *maxC = fmaxf(*maxC, MC);
  /* check for overflow of |z| */
  return (MB == __builtin_inff());
}

int pvn_cop_lgabs(const unsigned m, const unsigned n, const double complex *const restrict A, const size_t ldA, double *const restrict B, const size_t ldB, double *const restrict C, const size_t ldC, double *const restrict minB, double *const restrict maxB, double *const restrict minC, double *const restrict maxC)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!C)
    return -7;
  if (!ldC)
    return -8;
  if (!minB)
    return -9;
  if (!maxB)
    return -10;
  if (!minC)
    return -11;
  if (!maxC)
    return -12;
  if (m > pvn_umin(pvn_umin(ldA, ldB), ldC))
    return -13;
  double mB = __builtin_inf(), MB = __builtin_inf(), mC = __builtin_inf(), MC = __builtin_inf();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB,C,ldC) reduction(min:mB,MB,mC,MC)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const double complex *const cA = A + j * ldA;
    double *const cB = B + j * ldB;
    double *const cC = C + j * ldC;
    for (unsigned i = 0u; i < m; ++i) {
      const double complex z = cA[i];
      const double x = (cB[i] = log2(cabs(z)));
      const double y = (cC[i] = carg(z));
      if (isfinite(x)) {
        mB = fmin(mB,  x);
        MB = fmin(MB, -x);
      }
      if (isfinite(y)) {
        mC = fmin(mC,  y);
        MC = fmin(MC, -y);
      }
    }
  }
  MB = -MB;
  MC = -MC;
  if (isfinite(mB))
    *minB = fmin(*minB, mB);
  if (isfinite(MB))
    *maxB = fmax(*maxB, MB);
  if (isfinite(mC))
    *minC = fmin(*minC, mC);
  if (isfinite(MC))
    *maxC = fmax(*maxC, MC);
  /* check for overflow of |z| */
  return (MB == __builtin_inf());
}

int pvn_cop_lgabsl(const unsigned m, const unsigned n, const long double complex *const restrict A, const size_t ldA, long double *const restrict B, const size_t ldB, long double *const restrict C, const size_t ldC, long double *const restrict minB, long double *const restrict maxB, long double *const restrict minC, long double *const restrict maxC)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!C)
    return -7;
  if (!ldC)
    return -8;
  if (!minB)
    return -9;
  if (!maxB)
    return -10;
  if (!minC)
    return -11;
  if (!maxC)
    return -12;
  if (m > pvn_umin(pvn_umin(ldA, ldB), ldC))
    return -13;
  long double mB = __builtin_infl(), MB = __builtin_infl(), mC = __builtin_infl(), MC = __builtin_infl();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB,C,ldC) reduction(min:mB,MB,mC,MC)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const long double complex *const cA = A + j * ldA;
    long double *const cB = B + j * ldB;
    long double *const cC = C + j * ldC;
    for (unsigned i = 0u; i < m; ++i) {
      const long double complex z = cA[i];
      const long double x = (cB[i] = log2l(cabsl(z)));
      const long double y = (cC[i] = cargl(z));
      if (isfinite(x)) {
        mB = fminl(mB,  x);
        MB = fminl(MB, -x);
      }
      if (isfinite(y)) {
        mC = fminl(mC,  y);
        MC = fminl(MC, -y);
      }
    }
  }
  MB = -MB;
  MC = -MC;
  if (isfinite(mB))
    *minB = fminl(*minB, mB);
  if (isfinite(MB))
    *maxB = fmaxl(*maxB, MB);
  if (isfinite(mC))
    *minC = fminl(*minC, mC);
  if (isfinite(MC))
    *maxC = fmaxl(*maxC, MC);
  /* check for overflow of |z| */
  return (MB == __builtin_infl());
}

int pvn_cop_logabsf(const unsigned m, const unsigned n, const float complex *const restrict A, const size_t ldA, float *const restrict B, const size_t ldB, float *const restrict C, const size_t ldC, float *const restrict minB, float *const restrict maxB, float *const restrict minC, float *const restrict maxC)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!C)
    return -7;
  if (!ldC)
    return -8;
  if (!minB)
    return -9;
  if (!maxB)
    return -10;
  if (!minC)
    return -11;
  if (!maxC)
    return -12;
  if (m > pvn_umin(pvn_umin(ldA, ldB), ldC))
    return -13;
  float mB = __builtin_inff(), MB = __builtin_inff(), mC = __builtin_inff(), MC = __builtin_inff();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB,C,ldC) reduction(min:mB,MB,mC,MC)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const float complex *const cA = A + j * ldA;
    float *const cB = B + j * ldB;
    float *const cC = C + j * ldC;
    for (unsigned i = 0u; i < m; ++i) {
      const float complex z = cA[i];
      const float x = (cB[i] = log10f(cabsf(z)));
      const float y = (cC[i] = cargf(z));
      if (isfinite(x)) {
        mB = fminf(mB,  x);
        MB = fminf(MB, -x);
      }
      if (isfinite(y)) {
        mC = fminf(mC,  y);
        MC = fminf(MC, -y);
      }
    }
  }
  MB = -MB;
  MC = -MC;
  if (isfinite(mB))
    *minB = fminf(*minB, mB);
  if (isfinite(MB))
    *maxB = fmaxf(*maxB, MB);
  if (isfinite(mC))
    *minC = fminf(*minC, mC);
  if (isfinite(MC))
    *maxC = fmaxf(*maxC, MC);
  /* check for overflow of |z| */
  return (MB == __builtin_inff());
}

int pvn_cop_logabs(const unsigned m, const unsigned n, const double complex *const restrict A, const size_t ldA, double *const restrict B, const size_t ldB, double *const restrict C, const size_t ldC, double *const restrict minB, double *const restrict maxB, double *const restrict minC, double *const restrict maxC)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!C)
    return -7;
  if (!ldC)
    return -8;
  if (!minB)
    return -9;
  if (!maxB)
    return -10;
  if (!minC)
    return -11;
  if (!maxC)
    return -12;
  if (m > pvn_umin(pvn_umin(ldA, ldB), ldC))
    return -13;
  double mB = __builtin_inf(), MB = __builtin_inf(), mC = __builtin_inf(), MC = __builtin_inf();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB,C,ldC) reduction(min:mB,MB,mC,MC)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const double complex *const cA = A + j * ldA;
    double *const cB = B + j * ldB;
    double *const cC = C + j * ldC;
    for (unsigned i = 0u; i < m; ++i) {
      const double complex z = cA[i];
      const double x = (cB[i] = log10(cabs(z)));
      const double y = (cC[i] = carg(z));
      if (isfinite(x)) {
        mB = fmin(mB,  x);
        MB = fmin(MB, -x);
      }
      if (isfinite(y)) {
        mC = fmin(mC,  y);
        MC = fmin(MC, -y);
      }
    }
  }
  MB = -MB;
  MC = -MC;
  if (isfinite(mB))
    *minB = fmin(*minB, mB);
  if (isfinite(MB))
    *maxB = fmax(*maxB, MB);
  if (isfinite(mC))
    *minC = fmin(*minC, mC);
  if (isfinite(MC))
    *maxC = fmax(*maxC, MC);
  /* check for overflow of |z| */
  return (MB == __builtin_inf());
}

int pvn_cop_logabsl(const unsigned m, const unsigned n, const long double complex *const restrict A, const size_t ldA, long double *const restrict B, const size_t ldB, long double *const restrict C, const size_t ldC, long double *const restrict minB, long double *const restrict maxB, long double *const restrict minC, long double *const restrict maxC)
{
  if (!m)
    return -1;
  if (!n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (!B)
    return -5;
  if (!ldB)
    return -6;
  if (!C)
    return -7;
  if (!ldC)
    return -8;
  if (!minB)
    return -9;
  if (!maxB)
    return -10;
  if (!minC)
    return -11;
  if (!maxC)
    return -12;
  if (m > pvn_umin(pvn_umin(ldA, ldB), ldC))
    return -13;
  long double mB = __builtin_infl(), MB = __builtin_infl(), mC = __builtin_infl(), MC = __builtin_infl();
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(m,n,A,ldA,B,ldB,C,ldC) reduction(min:mB,MB,mC,MC)
#endif /* _OPENMP */
  for (unsigned j = 0u; j < n; ++j) {
    const long double complex *const cA = A + j * ldA;
    long double *const cB = B + j * ldB;
    long double *const cC = C + j * ldC;
    for (unsigned i = 0u; i < m; ++i) {
      const long double complex z = cA[i];
      const long double x = (cB[i] = log10l(cabsl(z)));
      const long double y = (cC[i] = cargl(z));
      if (isfinite(x)) {
        mB = fminl(mB,  x);
        MB = fminl(MB, -x);
      }
      if (isfinite(y)) {
        mC = fminl(mC,  y);
        MC = fminl(MC, -y);
      }
    }
  }
  MB = -MB;
  MC = -MC;
  if (isfinite(mB))
    *minB = fminl(*minB, mB);
  if (isfinite(MB))
    *maxB = fmaxl(*maxB, MB);
  if (isfinite(mC))
    *minC = fminl(*minC, mC);
  if (isfinite(MC))
    *maxC = fmaxl(*maxC, MC);
  /* check for overflow of |z| */
  return (MB == __builtin_infl());
}
#endif /* ?PVN_TEST */
