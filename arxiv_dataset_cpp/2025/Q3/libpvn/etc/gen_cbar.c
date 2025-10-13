#include "pvn.h"

/* definitions:
   255 = 3*5*17 colors shown
   1 pixel line per color
   17 bars
   3*5 = 15 pixel lines per a bar
   + 1 pixel line of the color 0 at the bottom for delineation
   = 16 pixel lines per bar
   + 1 pixel line at the top of the colorbar
   = 16*17 + 1 = 273 pixel lines
   2 pixels of the color 0 at each side of bars
   the width of each bar is 15+2 = 17 pixels
   + 1 pixel to separate it from the text

   the text next to each bar is height-centered;
   the lowest bar has the ">=MIN_VAL",
   the highest one the "<=MAX_VAL",
   and the rest have the "=MID-VAL" type of description,
   where MID-VAL is the value of the middle color in each bar

   the palette color 0 (representing -INFINITY or NaN) replaced by white,
   the palette color 255 (representing +INFINITY or NaN) replaced by black;
   (expected to have a white background and a black text in bitmaps)
   demonstrated by a tiny black line at the top of colorbars
*/

typedef long double (*pfn)(long double);

static long double id(long double x)
{
  return x;
}

#if (!defined(_GNU_SOURCE) && !defined(__INTEL_COMPILER))
static long double exp10l(long double x)
{
  return powl(10.0L, x);
}
#endif /* !_GNU_SOURCE && !__INTEL_COMPILER */

int main(int argc, char *argv[])
{
  if ((argc < 8) || (argc > 9)) {
    /* min_val, max_val: from the output of a visualization run */
    /* multiplier: the scale factor of a bitmap to be generated */
    /* text_width:
       if < 0, add |text_width| pixels;
       if > 0, text_width *= max_text_len * multiplier;
       if = 0, text_width  = max_text_len * multiplier * 8.
    */
    /* plt: palette CSV */
    /* bmp: the full name of a bitmap to be generated */
    /* id|lg|log: the transformation function from the run */
    /* fmt: a C-style format, with the leading % */
    (void)fprintf(stderr, "%s min_val max_val multiplier text_width plt bmp id|lg|log [fmt]\n", argv[0]);
    return EXIT_FAILURE;
  }
#ifdef __x86_64__
  char s[31] = { '\0' };
#else /* !__x86_64__ */
  char s[46] = { '\0' };
#endif /* ?__x86_64__ */
  char *endptr = (char*)NULL;
  const long double min_val = strtold(argv[1], &endptr);
#ifndef NDEBUG
  (void)fprintf(stdout, "min_val = %s\n", pvn_xtoa(s, min_val));
#endif /* !NDEBUG */
  if (*endptr || !isfinite(min_val))
    return EXIT_FAILURE;
  const long double max_val = strtold(argv[2], &endptr);
#ifndef NDEBUG
  (void)fprintf(stdout, "max_val = %s\n", pvn_xtoa(s, max_val));
#endif /* !NDEBUG */
  if (*endptr || !isfinite(max_val))
    return EXIT_FAILURE;
  if (max_val < min_val) {
    (void)fprintf(stderr, "max_val < min_val\n");
    return EXIT_FAILURE;
  }
  const unsigned n_colors = ((max_val == min_val) ? 1u : 254u);
#ifndef NDEBUG
  (void)fprintf(stdout, "n_colors = %u\n", n_colors);
#endif /* !NDEBUG */
  if (n_colors != 254u) {
    (void)fprintf(stderr, "n_colors != 254\n");
    return EXIT_FAILURE;
  }
  int tmp = 0;
  const unsigned M = (((tmp = atoi(argv[3])) > 0) ? (unsigned)tmp : 0u);
  if (!M) {
    (void)fprintf(stderr, "M <= 0\n");
    return EXIT_FAILURE;
  }
  const unsigned ptsize = 14u * M;
#ifndef NDEBUG
  (void)fprintf(stdout, "ptsize = %u\n", ptsize);
#endif /* !NDEBUG */
  const char *const fmt = ((argc == 9) ? argv[8] : (char*)NULL);
  if (fmt)
    tmp = sprintf(s, fmt, min_val) + 1;
  else
    tmp = (int)strlen(pvn_xtoa(s, min_val)) + 1;
  unsigned text_width = (unsigned)tmp * M;
  tmp = atoi(argv[4]);
  if (tmp < 0)
    text_width = (unsigned)-tmp;
  else if (tmp > 0)
    text_width *= (unsigned)tmp;
  else /* tmp == 0 */
    text_width *= 8u;
#ifndef NDEBUG
  (void)fprintf(stdout, "text_width = %u\n", text_width);
#endif /* !NDEBUG */
  const char *const plt = argv[5];
#ifndef NDEBUG
  (void)fprintf(stdout, "plt = %s\n", plt);
#endif /* !NDEBUG */
  const char *const bmp = argv[6];
#ifndef NDEBUG
  (void)fprintf(stdout, "bmp = %s\n", bmp);
#endif /* !NDEBUG */
  pfn fn = id;
  if (!strcmp(argv[7], "lg"))
    fn = exp2l;
  else if (!strcmp(argv[7], "log"))
    fn = exp10l;
  else if (strcmp(argv[7], "id")) {
    (void)fprintf(stderr, "Scale not one of id, lg (base 2), log (base 10)!\n");
    return EXIT_FAILURE;
  }

  const unsigned spc_x = 18u * M;
  const unsigned width = spc_x + text_width;
  const unsigned height = 273u * M;
  const unsigned spc_by = 16u * M;
  pvn_bmp_t cbar = (pvn_bmp_t)NULL;
  if (pvn_bmp_create(&cbar, width, height, 8u))
    return EXIT_FAILURE;
  if (pvn_bmp_read_cmap(cbar, plt))
    return EXIT_FAILURE;
  if (pvn_bmp_set_palette_color(cbar, 0u, ~0u))
    return EXIT_FAILURE;
  if (pvn_bmp_set_palette_color(cbar, 255u, 0u))
    return EXIT_FAILURE;
  const pvn_bmp_pixel_setter_t ps = pvn_bmp_get_pixel_setter(cbar);
  if (!ps)
    return EXIT_FAILURE;

  /* top delineation */
  unsigned y = 0u;
  for (; y < M; ++y)
    for (unsigned x = 0u; x < (17u * M); ++x)
      ps(cbar, x, y, 0u);
  /* bars */
  for (unsigned c = 255u; c; --c) {
    for (unsigned i = 0u; i < M; ++i, ++y) {
      unsigned x = 0u;
      for (; x < M; ++x)
        ps(cbar, x, y, 0u);
      for (; x < (16u * M); ++x)
        ps(cbar, x, y, c);
      for (; x < (17u * M); ++x)
        ps(cbar, x, y, 0u);
    }
    /* bottom delineation */
    if (!((256u - c) % 15u))
      for (unsigned i = 0u; i < M; ++i, ++y)
        for (unsigned x = 0u; x < (17u * M); ++x)
          ps(cbar, x, y, 0u);
  }
  (void)fprintf(stdout, "#!/bin/sh\nmagick %s -font %s ", bmp,
                /* Font name is Courier-Bold on Linux, Courier-New-Bold on macOS. */
#ifdef _GNU_SOURCE
                "Courier-Bold"
#else /* macOS */
                "Courier-New-Bold"
#endif /* ?_GNU_SOURCE */
                );
  (void)fprintf(stdout, "-pointsize %u ", ptsize);
  (void)fprintf(stdout, "-annotate +%u+%u \'≤", spc_x, (spc_by - 1u));
  (void)(fmt ? fprintf(stdout, fmt, fn(max_val)) : fprintf(stdout, "%s", pvn_xtoa(s, fn(max_val))));
  (void)fprintf(stdout, "\' ");
  const long double wid = max_val - min_val;
  for (unsigned b = 16u; b >= 2u; --b) {
    const unsigned c = (b - 1u) * 15u + 8u;
    const long double c_1 = (long double)(c - 1u);
    const long double val = fn(fmal(wid, (c_1 / 253.0L), min_val));
    (void)fprintf(stdout, "-annotate +%u+%u \'=", spc_x, (18u - b) * spc_by - 1u);
    (void)(fmt ? fprintf(stdout, fmt, val) : fprintf(stdout, "%s", pvn_xtoa(s, val)));
    (void)fprintf(stdout, "\' ");
  }
  (void)fprintf(stdout, "-annotate +%u+%u \'≥", spc_x, (17u * spc_by - 1u));
  (void)(fmt ? fprintf(stdout, fmt, fn(min_val)) : fprintf(stdout, "%s", pvn_xtoa(s, fn(min_val))));
  (void)fprintf(stdout, "\' .%s\n", bmp);
  (void)fprintf(stdout, "mv -fv .%s %s\n", bmp, bmp);
  (void)fprintf(stdout, "magick %s -colors 256 -quality 90 `basename %s bmp`png\n", bmp, bmp);
  (void)fprintf(stdout, "rm -fv %s\n", bmp);
  if (pvn_bmp_fwrite(cbar, bmp))
    return EXIT_FAILURE;
  pvn_bmp_destroy(cbar);
  return EXIT_SUCCESS;
}
