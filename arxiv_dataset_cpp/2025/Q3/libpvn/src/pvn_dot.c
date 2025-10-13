#include "pvn.h"

#ifdef PVN_TEST
int main(int argc, char *argv[])
{
  if (1 < argc) {
    (void)fprintf(stderr, "%s takes no arguments\n", *argv);
    return EXIT_FAILURE;
  }
  const double x[4][2] = {{-1.0, 1.0}, { 1.0,-1.0}, { 2.0, 2.0}, {-2.0,-2.0}};
  double zr = 0.0, zi = 0.0;
  const int m = 4;
  const unsigned i = 2u;
  PVN_FABI(pvn_zdot,PVN_ZDOT)(&zr, &zi, &m, &(x[0][0]), &i, &(x[0][1]), &i, &(x[0][0]), &i, &(x[0][1]), &i);
  /* expected output: ( 20.0,  0.0) */
  (void)printf("(%# 5.1F,%# 5.1F)\n", zr, zi);
  return EXIT_SUCCESS;
}
#else /* !PVN_TEST */
void PVN_FABI(pvn_sdot,PVN_SDOT)(float *const d, const int *const m, const float *const x, const unsigned *const ix, const float *const y, const unsigned *const iy)
{
  PVN_ASSERT(d);
  PVN_ASSERT(m);
  PVN_ASSERT(x);
  PVN_ASSERT(ix);
  PVN_ASSERT(y);
  PVN_ASSERT(iy);
  unsigned n = 0u;
  if (*m >= 0) {
    *d = 0.0f;
    n = (unsigned)*m;
  }
  else
    n = (unsigned)-*m;
  if (!n)
    return;
  const float *x_ = x;
  const float *y_ = y;
  for (unsigned i = 0u; i < n; ++i) {
    *d = fmaf(*x_, *y_, *d);
    x_ += *ix;
    y_ += *iy;
  }
}

void PVN_FABI(pvn_cdot,PVN_CDOT)(float *const zr, float *const zi, const int *const m, const float *const xr, const unsigned *const ixr, const float *const xi, const unsigned *const ixi, const float *const yr, const unsigned *const iyr, const float *const yi, const unsigned *const iyi)
{
  PVN_ASSERT(zr);
  PVN_ASSERT(zi);
  PVN_ASSERT(m);
  PVN_ASSERT(xr);
  PVN_ASSERT(ixr);
  PVN_ASSERT(xi);
  PVN_ASSERT(ixi);
  PVN_ASSERT(yr);
  PVN_ASSERT(iyr);
  PVN_ASSERT(yi);
  PVN_ASSERT(iyi);
  unsigned n = 0u;
  if (*m >= 0) {
    *zr = 0.0f;
    *zi = 0.0f;
    n = (unsigned)*m;
  }
  else
    n = (unsigned)-*m;
  if (!n)
    return;
  if (*ixi) {
    if (*iyi) {
      const float *xr_ = xr;
      const float *xi_ = xi;
      const float *yr_ = yr;
      const float *yi_ = yi;
      for (unsigned i = 0u; i < n; ++i) {
        pvn_cfma(zr, zi, *xr_, -*xi_, *yr_, *yi_, *zr, *zi);
        xr_ += *ixr;
        xi_ += *ixi;
        yr_ += *iyr;
        yi_ += *iyi;
      }
    }
    else {
      const float *xr_ = xr;
      const float *xi_ = xi;
      const float *yr_ = yr;
      const float *yi_ = (yi + 1);
      for (unsigned i = 0u; i < n; ++i) {
        pvn_cfma(zr, zi, *xr_, -*xi_, *yr_, *yi_, *zr, *zi);
        xr_ += *ixr;
        xi_ += *ixi;
        yr_ += *iyr;
        yi_ += *iyr;
      }
    }
  }
  else {
    if (*iyi) {
      const float *xr_ = xr;
      const float *xi_ = (xi + 1);
      const float *yr_ = yr;
      const float *yi_ = yi;
      for (unsigned i = 0u; i < n; ++i) {
        pvn_cfma(zr, zi, *xr_, -*xi_, *yr_, *yi_, *zr, *zi);
        xr_ += *ixr;
        xi_ += *ixr;
        yr_ += *iyr;
        yi_ += *iyi;
      }
    }
    else {
      const float *xr_ = xr;
      const float *xi_ = (xi + 1);
      const float *yr_ = yr;
      const float *yi_ = (yi + 1);
      for (unsigned i = 0u; i < n; ++i) {
        pvn_cfma(zr, zi, *xr_, -*xi_, *yr_, *yi_, *zr, *zi);
        xr_ += *ixr;
        xi_ += *ixr;
        yr_ += *iyr;
        yi_ += *iyr;
      }
    }
  }
}

void PVN_FABI(pvn_ddot,PVN_DDOT)(double *const d, const int *const m, const double *const x, const unsigned *const ix, const double *const y, const unsigned *const iy)
{
  PVN_ASSERT(d);
  PVN_ASSERT(m);
  PVN_ASSERT(x);
  PVN_ASSERT(ix);
  PVN_ASSERT(y);
  PVN_ASSERT(iy);
  unsigned n = 0u;
  if (*m >= 0) {
    *d = 0.0;
    n = (unsigned)*m;
  }
  else
    n = (unsigned)-*m;
  if (!n)
    return;
  const double *x_ = x;
  const double *y_ = y;
  for (unsigned i = 0u; i < n; ++i) {
    *d = fma(*x_, *y_, *d);
    x_ += *ix;
    y_ += *iy;
  }
}

void PVN_FABI(pvn_zdot,PVN_ZDOT)(double *const zr, double *const zi, const int *const m, const double *const xr, const unsigned *const ixr, const double *const xi, const unsigned *const ixi, const double *const yr, const unsigned *const iyr, const double *const yi, const unsigned *const iyi)
{
  PVN_ASSERT(zr);
  PVN_ASSERT(zi);
  PVN_ASSERT(m);
  PVN_ASSERT(xr);
  PVN_ASSERT(ixr);
  PVN_ASSERT(xi);
  PVN_ASSERT(ixi);
  PVN_ASSERT(yr);
  PVN_ASSERT(iyr);
  PVN_ASSERT(yi);
  PVN_ASSERT(iyi);
  unsigned n = 0u;
  if (*m >= 0) {
    *zr = 0.0;
    *zi = 0.0;
    n = (unsigned)*m;
  }
  else
    n = (unsigned)-*m;
  if (!n)
    return;
  if (*ixi) {
    if (*iyi) {
      const double *xr_ = xr;
      const double *xi_ = xi;
      const double *yr_ = yr;
      const double *yi_ = yi;
      for (unsigned i = 0u; i < n; ++i) {
        pvn_zfma(zr, zi, *xr_, -*xi_, *yr_, *yi_, *zr, *zi);
        xr_ += *ixr;
        xi_ += *ixi;
        yr_ += *iyr;
        yi_ += *iyi;
      }
    }
    else {
      const double *xr_ = xr;
      const double *xi_ = xi;
      const double *yr_ = yr;
      const double *yi_ = (yi + 1);
      for (unsigned i = 0u; i < n; ++i) {
        pvn_zfma(zr, zi, *xr_, -*xi_, *yr_, *yi_, *zr, *zi);
        xr_ += *ixr;
        xi_ += *ixi;
        yr_ += *iyr;
        yi_ += *iyr;
      }
    }
  }
  else {
    if (*iyi) {
      const double *xr_ = xr;
      const double *xi_ = (xi + 1);
      const double *yr_ = yr;
      const double *yi_ = yi;
      for (unsigned i = 0u; i < n; ++i) {
        pvn_zfma(zr, zi, *xr_, -*xi_, *yr_, *yi_, *zr, *zi);
        xr_ += *ixr;
        xi_ += *ixr;
        yr_ += *iyr;
        yi_ += *iyi;
      }
    }
    else {
      const double *xr_ = xr;
      const double *xi_ = (xi + 1);
      const double *yr_ = yr;
      const double *yi_ = (yi + 1);
      for (unsigned i = 0u; i < n; ++i) {
        pvn_zfma(zr, zi, *xr_, -*xi_, *yr_, *yi_, *zr, *zi);
        xr_ += *ixr;
        xi_ += *ixr;
        yr_ += *iyr;
        yi_ += *iyr;
      }
    }
  }
}

void PVN_FABI(pvn_xdot,PVN_XDOT)(long double *const d, const int *const m, const long double *const x, const unsigned *const ix, const long double *const y, const unsigned *const iy)
{
  PVN_ASSERT(d);
  PVN_ASSERT(m);
  PVN_ASSERT(x);
  PVN_ASSERT(ix);
  PVN_ASSERT(y);
  PVN_ASSERT(iy);
  unsigned n = 0u;
  if (*m >= 0) {
    *d = 0.0L;
    n = (unsigned)*m;
  }
  else
    n = (unsigned)-*m;
  if (!n)
    return;
  const long double *x_ = x;
  const long double *y_ = y;
  for (unsigned i = 0u; i < n; ++i) {
    *d = fmal(*x_, *y_, *d);
    x_ += *ix;
    y_ += *iy;
  }
}

void PVN_FABI(pvn_wdot,PVN_WDOT)(long double *const zr, long double *const zi, const int *const m, const long double *const xr, const unsigned *const ixr, const long double *const xi, const unsigned *const ixi, const long double *const yr, const unsigned *const iyr, const long double *const yi, const unsigned *const iyi)
{
  PVN_ASSERT(zr);
  PVN_ASSERT(zi);
  PVN_ASSERT(m);
  PVN_ASSERT(xr);
  PVN_ASSERT(ixr);
  PVN_ASSERT(xi);
  PVN_ASSERT(ixi);
  PVN_ASSERT(yr);
  PVN_ASSERT(iyr);
  PVN_ASSERT(yi);
  PVN_ASSERT(iyi);
  unsigned n = 0u;
  if (*m >= 0) {
    *zr = 0.0L;
    *zi = 0.0L;
    n = (unsigned)*m;
  }
  else
    n = (unsigned)-*m;
  if (!n)
    return;
  if (*ixi) {
    if (*iyi) {
      const long double *xr_ = xr;
      const long double *xi_ = xi;
      const long double *yr_ = yr;
      const long double *yi_ = yi;
      for (unsigned i = 0u; i < n; ++i) {
        pvn_wfma(zr, zi, *xr_, -*xi_, *yr_, *yi_, *zr, *zi);
        xr_ += *ixr;
        xi_ += *ixi;
        yr_ += *iyr;
        yi_ += *iyi;
      }
    }
    else {
      const long double *xr_ = xr;
      const long double *xi_ = xi;
      const long double *yr_ = yr;
      const long double *yi_ = (yi + 1);
      for (unsigned i = 0u; i < n; ++i) {
        pvn_wfma(zr, zi, *xr_, -*xi_, *yr_, *yi_, *zr, *zi);
        xr_ += *ixr;
        xi_ += *ixi;
        yr_ += *iyr;
        yi_ += *iyr;
      }
    }
  }
  else {
    if (*iyi) {
      const long double *xr_ = xr;
      const long double *xi_ = (xi + 1);
      const long double *yr_ = yr;
      const long double *yi_ = yi;
      for (unsigned i = 0u; i < n; ++i) {
        pvn_wfma(zr, zi, *xr_, -*xi_, *yr_, *yi_, *zr, *zi);
        xr_ += *ixr;
        xi_ += *ixr;
        yr_ += *iyr;
        yi_ += *iyi;
      }
    }
    else {
      const long double *xr_ = xr;
      const long double *xi_ = (xi + 1);
      const long double *yr_ = yr;
      const long double *yi_ = (yi + 1);
      for (unsigned i = 0u; i < n; ++i) {
        pvn_wfma(zr, zi, *xr_, -*xi_, *yr_, *yi_, *zr, *zi);
        xr_ += *ixr;
        xi_ += *ixr;
        yr_ += *iyr;
        yi_ += *iyr;
      }
    }
  }
}

#ifdef PVN_QUADMATH
void PVN_FABI(pvn_qdot,PVN_QDOT)(__float128 *const d, const int *const m, const __float128 *const x, const unsigned *const ix, const __float128 *const y, const unsigned *const iy)
{
  PVN_ASSERT(d);
  PVN_ASSERT(m);
  PVN_ASSERT(x);
  PVN_ASSERT(ix);
  PVN_ASSERT(y);
  PVN_ASSERT(iy);
  unsigned n = 0u;
  if (*m >= 0) {
    *d = 0.0q;
    n = (unsigned)*m;
  }
  else
    n = (unsigned)-*m;
  if (!n)
    return;
  const __float128 *x_ = x;
  const __float128 *y_ = y;
  for (unsigned i = 0u; i < n; ++i) {
    *d = fmaq(*x_, *y_, *d);
    x_ += *ix;
    y_ += *iy;
  }
}

void PVN_FABI(pvn_ydot,PVN_YDOT)(__float128 *const zr, __float128 *const zi, const int *const m, const __float128 *const xr, const unsigned *const ixr, const __float128 *const xi, const unsigned *const ixi, const __float128 *const yr, const unsigned *const iyr, const __float128 *const yi, const unsigned *const iyi)
{
  PVN_ASSERT(zr);
  PVN_ASSERT(zi);
  PVN_ASSERT(m);
  PVN_ASSERT(xr);
  PVN_ASSERT(ixr);
  PVN_ASSERT(xi);
  PVN_ASSERT(ixi);
  PVN_ASSERT(yr);
  PVN_ASSERT(iyr);
  PVN_ASSERT(yi);
  PVN_ASSERT(iyi);
  unsigned n = 0u;
  if (*m >= 0) {
    *zr = 0.0q;
    *zi = 0.0q;
    n = (unsigned)*m;
  }
  else
    n = (unsigned)-*m;
  if (!n)
    return;
  if (*ixi) {
    if (*iyi) {
      const __float128 *xr_ = xr;
      const __float128 *xi_ = xi;
      const __float128 *yr_ = yr;
      const __float128 *yi_ = yi;
      for (unsigned i = 0u; i < n; ++i) {
        pvn_yfma(zr, zi, *xr_, -*xi_, *yr_, *yi_, *zr, *zi);
        xr_ += *ixr;
        xi_ += *ixi;
        yr_ += *iyr;
        yi_ += *iyi;
      }
    }
    else {
      const __float128 *xr_ = xr;
      const __float128 *xi_ = xi;
      const __float128 *yr_ = yr;
      const __float128 *yi_ = (yi + 1);
      for (unsigned i = 0u; i < n; ++i) {
        pvn_yfma(zr, zi, *xr_, -*xi_, *yr_, *yi_, *zr, *zi);
        xr_ += *ixr;
        xi_ += *ixi;
        yr_ += *iyr;
        yi_ += *iyr;
      }
    }
  }
  else {
    if (*iyi) {
      const __float128 *xr_ = xr;
      const __float128 *xi_ = (xi + 1);
      const __float128 *yr_ = yr;
      const __float128 *yi_ = yi;
      for (unsigned i = 0u; i < n; ++i) {
        pvn_yfma(zr, zi, *xr_, -*xi_, *yr_, *yi_, *zr, *zi);
        xr_ += *ixr;
        xi_ += *ixr;
        yr_ += *iyr;
        yi_ += *iyi;
      }
    }
    else {
      const __float128 *xr_ = xr;
      const __float128 *xi_ = (xi + 1);
      const __float128 *yr_ = yr;
      const __float128 *yi_ = (yi + 1);
      for (unsigned i = 0u; i < n; ++i) {
        pvn_yfma(zr, zi, *xr_, -*xi_, *yr_, *yi_, *zr, *zi);
        xr_ += *ixr;
        xi_ += *ixr;
        yr_ += *iyr;
        yi_ += *iyr;
      }
    }
  }
}
#else /* !PVN_QUADMATH */
void PVN_FABI(pvn_qdot,PVN_QDOT)(long double *const d, const int *const m, const long double *const x, const unsigned *const ix, const long double *const y, const unsigned *const iy)
{
  PVN_FABI(pvn_xdot,PVN_XDOT)(d, m, x, ix, y, iy);
}

void PVN_FABI(pvn_ydot,PVN_YDOT)(long double *const zr, long double *const zi, const int *const m, const long double *const xr, const unsigned *const ixr, const long double *const xi, const unsigned *const ixi, const long double *const yr, const unsigned *const iyr, const long double *const yi, const unsigned *const iyi)
{
  PVN_FABI(pvn_wdot,PVN_WDOT)(zr, zi, m, xr, ixr, xi, ixi, yr, iyr, yi, iyi);
}
#endif /* ?PVN_QUADMATH */
#endif /* ?PVN_TEST */
