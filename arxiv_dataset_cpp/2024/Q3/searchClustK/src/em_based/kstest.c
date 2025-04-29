#include "kstest.h"

void ksone(double *y, int n, double *d, double *prob);
double probks(double alam);

void ksone(double *y, int n, double *d, double *prob) {
  /*
    Based on `ksone` from `William H. Press, Saul A. Teukolsky, William T. Vetterling, and Brian P. Flannery.
    Numerical recipes in C. Cambridge University Press 1992`.

    y: sorted data's cdf.
    n: The length of data.
    d: KS statistic
    prob: p-value
   */

  int i;
  double dt, en, ff, fn, fo;

  *d = 0.0; en = (double)n;
  for (i = 0; i < n; i++) {
    fn = i/en;
    fo = (i+1)/en;
    ff = y[i];
    dt = fmax(ff-fn, fo-ff);  // left is Dn+, right is Dn-
    if (dt > *d) *d = dt;
  }
  en = sqrt(en);
  *prob = probks((en + 0.12 + 0.11 / en) * (*d));
}

double probks(double alam) {
  /*
    `probks` from `William H. Press, Saul A. Teukolsky, William T. Vetterling, and Brian P. Flannery. Numerical recipes in C. Cambridge University Press 1992`.
  */

  int j;
  double a2,fac=2.0;
  double sum=0.0,term,termbf=0.0;

  a2 = -2.0*alam*alam;
  for (j=1; j<=100; j++) {
    term = fac * exp(a2*j*j);
    sum += term;
    if (fabs(term) <= EPS1*termbf || fabs(term) <= EPS2*sum) return sum;
    fac = -fac;
    termbf=fabs(term);
  }
  return 1.0;
}
