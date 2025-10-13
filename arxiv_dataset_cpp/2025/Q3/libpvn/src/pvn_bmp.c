#include "pvn.h"

#ifdef PVN_TEST
int main(int argc, char *argv[])
{
  if (argc != 6) {
    fprintf(stderr, "%s FileName Width Height BitsPerPixel Colormap\n", *argv);
    return EXIT_FAILURE;
  }

  const char *const fn = argv[1];
  const uint32_t w = (uint32_t)atoi(argv[2]);
  const int32_t h = atoi(argv[3]);
  const uint16_t bpp = (uint16_t)atoi(argv[4]);
  const char *const cmap = argv[5];

  uint32_t nc = 0u;
  switch (bpp) {
  case 1u:
    nc = 2u;
    break;
  case 4u:
    nc = 16u;
    break;
  case 8u:
    nc = 256u;
    break;
  default:
    return EXIT_FAILURE;
  }
  const uint32_t tw = w * nc;

  pvn_bmp_t bmp = (pvn_bmp_t)NULL;
  if (pvn_bmp_create(&bmp, tw, h, bpp))
    return EXIT_FAILURE;
  if (pvn_bmp_read_cmap(bmp, cmap))
    return EXIT_FAILURE;
  const pvn_bmp_pixel_setter_t ps = pvn_bmp_get_pixel_setter(bmp);
  if (!ps)
    return EXIT_FAILURE;

  for (uint32_t c = 0u, x = 0u; c < nc; ++c)
    for (uint32_t i = 0u; i < w; ++i, ++x)
      for (uint32_t j = 0u; j < (uint32_t)h; ++j)
        ps(bmp, x, j, c);

  if (pvn_bmp_fwrite(bmp, fn))
    return EXIT_FAILURE;
  pvn_bmp_destroy(bmp);
  return EXIT_SUCCESS;
}
#else /* !PVN_TEST */
int pvn_bmp_create(pvn_bmp_t *const bmp, const uint32_t width, const int32_t height, const uint16_t bpp)
{
  void *hdr_end;
  uint32_t c_palette;
  uint32_t size_palette;
  uint32_t size_prolog;
  uint32_t lda;
  uint32_t size_pixel_row;
  uint32_t size_pixel_array;
  uint32_t size_file;

  if (bmp)
    *bmp = (pvn_bmp_t)NULL;
  else
    return -1;

  if (!width)
    return -2;
  if (!height)
    return -3;

  switch (bpp) {
  case  1u:
  case  4u:
  case  8u:
    c_palette = 1u << bpp;
    break;
  case 24u:
    c_palette = 0u;
    break;
  default:
    return -4;
  }

  size_palette = c_palette * (uint32_t)sizeof(pvn_bmp_palette_entry_t);
  size_prolog = 2u /* BM */ + (uint32_t)sizeof(pvn_bmp_header_t) + size_palette + 2u /* GAP1 */;

  lda = (width * bpp + 31u) >> 5u;
  size_pixel_row = lda * (uint32_t)sizeof(uint32_t);
  size_pixel_array = size_pixel_row * (uint32_t)abs(height);
  size_file = size_prolog + size_pixel_array;

  *bmp = (pvn_bmp_t)calloc((size_t)((size_file >> 2u) + 6u), sizeof(uint32_t));
  if (!*bmp)
    return 1;

  hdr_end = (void*)((uint8_t*)&((*bmp)->header) + sizeof(pvn_bmp_header_t));
  (*bmp)->palette = (pvn_bmp_palette_entry_t*)(c_palette ? hdr_end : NULL);
  (*bmp)->image = (uint8_t*)(c_palette ? (void*)((uint8_t*)((*bmp)->palette) + size_palette) : hdr_end);

  (*bmp)->i_ldaB = size_pixel_row;

  (*bmp)->header.f_size = size_file;
  (*bmp)->header.i_offset = size_prolog;
  (*bmp)->header.d_size = 40u;
  (*bmp)->header.width = width;
  (*bmp)->header.height = height;
  (*bmp)->header.planes = (uint16_t)1u;
  (*bmp)->header.bpp = bpp;
  (*bmp)->header.i_size = size_pixel_array;
  (*bmp)->header.c_palette = c_palette;

  return 0;
}

void pvn_bmp_destroy(const pvn_bmp_t bmp)
{
  if (bmp)
    free(bmp);
}

int pvn_bmp_set_palette_color(const pvn_bmp_t bmp, const uint32_t ix, const uint32_t c)
{
  if (bmp && bmp->palette) {
    if (ix < bmp->header.c_palette) {
      bmp->palette[ix].c = c;
      return 0;
    }
    return -2;
  }
  return -1;
}

int pvn_bmp_get_palette_color(const pvn_bmp_t bmp, const uint32_t ix, uint32_t *const c)
{
  if (bmp && bmp->palette) {
    if (ix < bmp->header.c_palette) {
      if (c) {
        *c = bmp->palette[ix].c;
        return 0;
      }
      return -3;
    }
    return -2;
  }
  return -1;
}

static void bmp_set_pixel_01(const pvn_bmp_t bmp, const uint32_t x, const uint32_t y, const uint32_t c)
{
  uint8_t *const row = bmp->image + y * bmp->i_ldaB;
  const uint32_t ix = x >> 3u;
  const uint32_t bit_ix = 7u - (x & 7u);
  const uint32_t mask = 1u << bit_ix;
  const uint32_t byte = (uint32_t)(row[ix]);
  row[ix] = (uint8_t)(c ? (byte | mask) : (byte & ~mask));
}

static void bmp_set_pixel_01_c(const pvn_bmp_t bmp, const uint32_t x, const uint32_t y, const uint32_t c)
{
  bmp_set_pixel_01(bmp, x, (uint32_t)(bmp->header.height) - y - 1u, c);
}

static void bmp_set_pixel_04(const pvn_bmp_t bmp, const uint32_t x, const uint32_t y, const uint32_t c)
{
  uint8_t *const row = bmp->image + y * bmp->i_ldaB;
  const uint32_t ix = x >> 1u;
  const uint32_t byte = (uint32_t)(row[ix]);
  row[ix] = (uint8_t)((x & 1u) ? ((byte & 0xF0u) | c) : ((byte & 0x0Fu) | (c << 4u)));
}

static void bmp_set_pixel_04_c(const pvn_bmp_t bmp, const uint32_t x, const uint32_t y, const uint32_t c)
{
  bmp_set_pixel_04(bmp, x, (uint32_t)(bmp->header.height) - y - 1u, c);
}

static void bmp_set_pixel_08(const pvn_bmp_t bmp, const uint32_t x, const uint32_t y, const uint32_t c)
{
  uint8_t *const row = bmp->image + y * bmp->i_ldaB;
  row[x] = (uint8_t)c;
}

static void bmp_set_pixel_08_c(const pvn_bmp_t bmp, const uint32_t x, const uint32_t y, const uint32_t c)
{
  bmp_set_pixel_08(bmp, x, (uint32_t)(bmp->header.height) - y - 1u, c);
}

static void bmp_set_pixel_24(const pvn_bmp_t bmp, const uint32_t x, const uint32_t y, const uint32_t c)
{
  union { uint32_t word; uint8_t bytes[4]; } z;
  uint8_t *const row = bmp->image + y * bmp->i_ldaB;
  uint32_t i = x * 3u;
  z.word = c;
  row[i++] = z.bytes[0];
  row[i++] = z.bytes[1];
  row[i] = z.bytes[2];
}

static void bmp_set_pixel_24_c(const pvn_bmp_t bmp, const uint32_t x, const uint32_t y, const uint32_t c)
{
  bmp_set_pixel_24(bmp, x, (uint32_t)(bmp->header.height) - y - 1u, c);
}

pvn_bmp_pixel_setter_t pvn_bmp_get_pixel_setter(const pvn_bmp_t bmp)
{
  if (!bmp)
    return (pvn_bmp_pixel_setter_t)NULL;
  switch (bmp->header.bpp) {
  case  1u:
    return ((bmp->header.height < 0) ? bmp_set_pixel_01 : bmp_set_pixel_01_c);
  case  4u:
    return ((bmp->header.height < 0) ? bmp_set_pixel_04 : bmp_set_pixel_04_c);
  case  8u:
    return ((bmp->header.height < 0) ? bmp_set_pixel_08 : bmp_set_pixel_08_c);
  case 24u:
    return ((bmp->header.height < 0) ? bmp_set_pixel_24 : bmp_set_pixel_24_c);
  default:
    return (pvn_bmp_pixel_setter_t)NULL;
  }
}

static uint32_t bmp_get_pixel_01(const pvn_bmp_t bmp, const uint32_t x, const uint32_t y)
{
  const uint8_t *const row = bmp->image + y * bmp->i_ldaB;
  const uint32_t ix = x >> 3u;
  const uint32_t bit_ix = 7u - (x & 7u);
  const uint32_t mask = 1u << bit_ix;
  const uint32_t byte = (uint32_t)(row[ix]);
  return ((byte & mask) >> bit_ix);
}

static uint32_t bmp_get_pixel_01_c(const pvn_bmp_t bmp, const uint32_t x, const uint32_t y)
{
  return bmp_get_pixel_01(bmp, x, (uint32_t)(bmp->header.height) - y - 1u);
}

static uint32_t bmp_get_pixel_04(const pvn_bmp_t bmp, const uint32_t x, const uint32_t y)
{
  const uint8_t *const row = bmp->image + y * bmp->i_ldaB;
  const uint32_t ix = x >> 1u;
  const uint32_t byte = (uint32_t)(row[ix]);
  return ((x & 1u) ? (byte & 0x0Fu) : ((byte & 0xF0u) >> 4u));
}

static uint32_t bmp_get_pixel_04_c(const pvn_bmp_t bmp, const uint32_t x, const uint32_t y)
{
  return bmp_get_pixel_04(bmp, x, (uint32_t)(bmp->header.height) - y - 1u);
}

static uint32_t bmp_get_pixel_08(const pvn_bmp_t bmp, const uint32_t x, const uint32_t y)
{
  const uint8_t *const row = bmp->image + y * bmp->i_ldaB;
  return (uint32_t)(row[x]);
}

static uint32_t bmp_get_pixel_08_c(const pvn_bmp_t bmp, const uint32_t x, const uint32_t y)
{
  return bmp_get_pixel_08(bmp, x, (uint32_t)(bmp->header.height) - y - 1u);
}

static uint32_t bmp_get_pixel_24(const pvn_bmp_t bmp, const uint32_t x, const uint32_t y)
{
  union { uint32_t word; uint8_t bytes[4]; } z;
  const uint8_t *const row = bmp->image + y * bmp->i_ldaB;
  uint32_t i = x * 3u;
  z.bytes[0] = row[i++];
  z.bytes[1] = row[i++];
  z.bytes[2] = row[i];
  z.bytes[3] = 0u;
  return z.word;
}

static uint32_t bmp_get_pixel_24_c(const pvn_bmp_t bmp, const uint32_t x, const uint32_t y)
{
  return bmp_get_pixel_24(bmp, x, (uint32_t)(bmp->header.height) - y - 1u);
}

pvn_bmp_pixel_getter_t pvn_bmp_get_pixel_getter(const pvn_bmp_t bmp)
{
  if (!bmp)
    return (pvn_bmp_pixel_getter_t)NULL;
  switch (bmp->header.bpp) {
  case  1u:
    return ((bmp->header.height < 0) ? bmp_get_pixel_01 : bmp_get_pixel_01_c);
  case  4u:
    return ((bmp->header.height < 0) ? bmp_get_pixel_04 : bmp_get_pixel_04_c);
  case  8u:
    return ((bmp->header.height < 0) ? bmp_get_pixel_08 : bmp_get_pixel_08_c);
  case 24u:
    return ((bmp->header.height < 0) ? bmp_get_pixel_24 : bmp_get_pixel_24_c);
  default:
    return (pvn_bmp_pixel_getter_t)NULL;
  }
}

int pvn_bmp_fwrite(const pvn_bmp_t bmp, const char *const fn)
{
  if (!bmp)
    return -1;
  if (!(fn && *fn))
    return -2;

  FILE *const f = fopen(fn, "wb");
  if (!f)
    return 1;

  const size_t h = (size_t)(uint32_t)abs(bmp->header.height);
  int ret = 0;

  if (fwrite("BM", sizeof(uint8_t), (size_t)2u, f) < (size_t)2u) {
    ret = 2;
    goto end;
  }
  if (fwrite(&(bmp->header), sizeof(pvn_bmp_header_t), (size_t)1u, f) < (size_t)1u) {
    ret = 3;
    goto end;
  }
  if (bmp->palette && (fwrite(bmp->palette, sizeof(pvn_bmp_palette_entry_t), (size_t)(bmp->header.c_palette), f) < (size_t)(bmp->header.c_palette))) {
    ret = 4;
    goto end;
  }
  /* mark the bitmap as made by libpvn by writing VN into the gap */
  if (fwrite("VN", sizeof(uint8_t), (size_t)2u, f) < (size_t)2u) {
    ret = 5;
    goto end;
  }
  if (fwrite(bmp->image, (sizeof(uint8_t) * bmp->i_ldaB), h, f) < h) {
    ret = 6;
    goto end;
  }
 end:
  if (fclose(f) && !ret)
    ret = 7;

  return ret;
}

int pvn_bmp_read_cmap(const pvn_bmp_t bmp, const char *const fn)
{
  if (!(bmp && bmp->palette))
    return -1;
  if (!(fn && *fn))
    return -2;

  FILE *const f = fopen(fn, "r");
  if (!f)
    return 1;

  double dr, dg, db;
  int ret = 0;
  for (uint32_t ix = 0u; ix < bmp->header.c_palette; ++ix) {
    if (fscanf(f, " %lE, %lE, %lE", &dr, &dg, &db) < 3) {
      ret = 2;
      break;
    }
    bmp->palette[ix].chan.b = (uint8_t)lround(db * 255.0);
    bmp->palette[ix].chan.g = (uint8_t)lround(dg * 255.0);
    bmp->palette[ix].chan.r = (uint8_t)lround(dr * 255.0);
    bmp->palette[ix].chan.a = 0u;
  }

  if (fclose(f) && !ret)
    ret = 3;

  return ret;
}
#endif /* ?PVN_TEST */
