/*

  Implementation of the Li22(x,y) function for real arguments.

  Based on the code provided with the paper
  by Hjalte Frellesvig, Damiano Tommasini, and Christopher Wever [ArXiv:1601.02649]

  For x~1, y~1 and y<0 we make use of alternative expansion in log(-y)
  with expansion coefficients provided in the file SSm0_Li22log1yNegTab.h

 */

#include "SSm0_internal.h"
/* static const tables with coefficients */
#include "SSm0_Li22data.h"
#include "SSm0_Li22log1yNegTab.h"


/* fast */
double li22re_rec(double next, double extra, int nc, double  a, double ab, int ncmax)
{
  if (nc < ncmax)
    return next + li22re_rec(WTFccli221[nc]*a*next + WTFccli222[nc]*extra, extra*ab, nc+1, a, ab, ncmax);
  else
    return next + WTFccli221[nc]*a*next + WTFccli222[nc]*extra;
}
double li22re_fast(double a, double b, int ncmax)
{
  double ab = a*b;
  return li22re_rec(0.25*a*ab, a*ab*ab, 0, a, ab, ncmax);
}

/* smalla */
double li22re_diagonalind(double iab, int nmo, double logomx, double litx)
{
  double res = 0.;
  double iabp = 1.;
  int i;

  for (i = 0; i <= nmo; ++i)
    {
      res += WTFccli22diagonalpow[nmo][i]*iabp;
      iabp *= iab;
    }

  res += WTFccli22diagonallog[nmo]*logomx*(1.-iabp);
  res += WTFccli22diagonallit[nmo]*litx*(1.+iabp);

  return res;
}

/* Make explicit REAL */
double li22re_diagonal(double a, double b, char ncmax)
{
  double res = 0;
  double ab = a*b;
  check_range(ab > 1, "diagonal, a*b > 1");
  double iab = 1./ab;
  double logomx = log(1.-ab);
  double litx = li2re(ab);
  double ap = 1.;
  int i;

  for (i = 0; i < ncmax; ++i)
    {
      ap *= a;
      res += ap*li22re_diagonalind(iab,i,logomx,litx);
    }
  return res;
}


// This is Li22(x/2,y)
double li22re_holderf1rec(double part1, double part2, double x, double xy, int n, int nn)
{
  if (n < nn)
    return part1 +
      li22re_holderf1rec(part1*x*WTFccholder11[n]+part2, part2*xy*WTFccholder12[n], x, xy, n+1, nn);
  else
    return part1;
}
double li22re_holderf1(double x, double xy, int nn)
{
  return li22re_holderf1rec(x*xy*WTFccholderc11, x*xy*xy*WTFccholderc12, x, xy, 0, nn);
}


// This is Li1111(x/2, 1/x, z, 1/z)
double li22re_holderf2rec(double part1, double part2, double part3, double part4,
                          double x, double z, int n, int nn)
{
    if (n < nn)
      return part1 +
        li22re_holderf2rec(part1*x*WTFccholderfo1[n]+part2,
                           part2*WTFccholderfo2[n]+part3,
                           part3*z*WTFccholderfo3[n]+part4,
                           part4*WTFccholderfo4[n], x, z, n+1, nn);
    else
      return part1;
}
double li22re_holderf2(double x, double z, int nn)
{
  return li22re_holderf2rec(0., 0., z*x*WTFccholderc21, z*x*WTFccholderc22, x, z, 0, nn);
}


// This is Li12(x/2,y)
double li22re_holderf3rec(double part1, double part2, double x, double xy, int n, int nn)
{
  if (n < nn)
    return part1 +
      li22re_holderf3rec(part1*x*WTFccholderfo1[n]+part2, part2*xy*WTFccholder32[n], x, xy, n+1, nn);
  else
    return part1;
}
double li22re_holderf3(double x, double xy, int nn)
{
  return li22re_holderf3rec(x*xy*WTFccholderc31, x*xy*xy*WTFccholderc32, x, xy, 0, nn);
}


// This is Li11(x/2, 1/x)
double li22re_holderf4rec(double part1, double part2, double x, int n, int nn)
{
  if (n < nn)
    return part1 +
      li22re_holderf4rec(part1*x*WTFccholderfo1[n]+part2,
                         part2*WTFccholderfo2[n], x, n+1, nn);
  else
    return part1;
}
double li22re_holderf4(double x, int nn)
{
  return li22re_holderf4rec(x*WTFccholderc41, x*WTFccholderc42, x, 0, nn);
}


// This is -Li111(1/2,x,1/x)
double li22re_holderf5rec(double part1, double part2, double part3, double y, int n, int nn)
{
  if (n < nn)
    return part1 +
      li22re_holderf5rec(part1*WTFccholderfo1[n]+part2,
                         y*part2*WTFccholderfo2[n]+part3,
                         part3*WTFccholderfo3[n], y, n+1, nn);
  else
    return part1;
}
double li22re_holderf5(double y, int nn)
{
  return -li22re_holderf5rec(0., y*WTFccholderc51, y*WTFccholderc52, y, 0, nn);
}


// This is the Holder relation with q=2 used for othervise slowly convergent regions.
double li22re_li22holder(double x, double y, int ncmax)
{
  double xy = x*y;
  double r1 = xy/(xy-1.);
  double r2 = x/(x-1.);
  double xyh = xy*0.5;
  return
    li22re_holderf1(x,xy,ncmax) +
    li22re_holderf2(r1,r2,ncmax) +
    WTFccholderclog2*li22re_holderf3(x,xy,ncmax) -
    li22re_holderf4(r2,ncmax)*SSm0_li2logA0(xyh) -
    li22re_holderf5(r2,ncmax)*log(1.-xyh);
}

double li22re_smalla(double a, double b)
{
  double absa = fabs(a);
  if (absa < 0.1)
    return li22re_diagonal(a,b,18);
  if (absa < 0.2)
    return li22re_diagonal(a,b,25);
  if (absa < 0.3)
    return li22re_diagonal(a,b,33);
  if (absa < 0.4)
    return li22re_diagonal(a,b,43);
  if (a < 0 && a*b < 0)
    return li22re_li22holder(a,b,95);
  if (absa < 0.5)
    return li22re_diagonal(a,b,56);
  if (absa < 0.6)
    return li22re_diagonal(a,b,77);
  if (absa <= 0.7)
    return li22re_diagonal(a,b,112);

  check_range(absa > 0.7, "smalla a*b > 0.7");
  return 0./0.;
}


/* B Stuffle */
double li22re_stufflerec(double next, double extra, int nc, double b, double ab, int ncmax)
{
  if (nc < ncmax)
    return next +
      li22re_stufflerec(WTFccli22stuffle1[nc]*b*next +
                        WTFccli22stuffle2[nc]*extra, extra*ab, nc+1, b, ab, ncmax);
  else
    return next + WTFccli22stuffle1[nc]*b*next + WTFccli22stuffle2[nc]*extra;
}

double li22re_stuffle(double a, double b, int ncmax)
{
  /* check_range(a > 1.0, "li22re_stuffle a > 1"); */
  double ab = a*b;
  return -li22re_stufflerec(ab-b*li2re(a), ab*ab, 0, b, ab, ncmax);
}

/* Log A */
double WTFli22logA1recint(double next, int nc, int nco, double b, double logb, double logbxi, int ncmax)
{
  if (nc < ncmax)
    return next*(WTFccli22logA1k1[nc][nco] +
                 WTFccli22logA1k2[nc][nco]*logb +
                 WTFccli22logA1k3[nc][nco]*logbxi) +
      WTFli22logA1recint(next*b, nc+1, nco, b, logb, logbxi,ncmax);
  else
    return next*(WTFccli22logA1k1[nc][nco] +
                 WTFccli22logA1k2[nc][nco]*logb +
                 WTFccli22logA1k3[nc][nco]*logbxi);
}

double li22re_logA1rec(double next, int nc,
                       double a, double b, double logb, double logbxi, double logabdif, int ncmax)
{
  if (nc < ncmax)
    return next*(WTFccli22logA1k4[nc] + b*WTFccli22logA1k5[nc])*logabdif +
      WTFli22logA1recint(next, 0, nc, b, logb, logbxi, ncmax) +
      li22re_logA1rec(next*a, nc+1, a, b, logb, logbxi, logabdif, ncmax);
  else
    return next*(WTFccli22logA1k4[nc] + b*WTFccli22logA1k5[nc])*logabdif +
      WTFli22logA1recint(next, 0, nc, b, logb, logbxi, ncmax);
}

/* Version for 0 < y < 1 */
double li22re_logA1ff2(double x, double y, int ncmax)
{
  check_range(y < 0 || x*y < 0, "logA1ff2: y < 0 || x*y < 0");
  double a = log(y);
  double b = -log(x*y);
  double apb = a+b;
  double logb = log(b);
  double logbxi = log(b + li_constants.xi0);
  double logabdif = log(apb) - log(apb + li_constants.xi0);
  double li2a1 = -(b + li_constants.xi0)/a;
  double li2a2 = -b/a;

  return li22re_logA1rec(1.,0,a,b,logb,logbxi,logabdif, ncmax)  +
    apb*a*( li2re(li2a1) - li2re(li2a2) +
            logbxi*log(fabs((apb + li_constants.xi0)/a)) -
            logb*log(fabs(apb/a)));
}

/* Version for y < 0 */
double WTFli22logA1recint_yNeg(double next, int nc, int nco,
                              double b, double logb, double logbxi, int ncmax)
{
  if (nc < ncmax)
    return next*(K_A_IJ_yNeg[nc][nco] +
                 K_B_IJ_yNeg[nc][nco]*logb +
                 K_C_IJ_yNeg[nc][nco]*logbxi) +
      WTFli22logA1recint_yNeg(next*b, nc+1, nco, b, logb, logbxi,ncmax);
  else
    return next*(K_A_IJ_yNeg[nc][nco] +
                 K_B_IJ_yNeg[nc][nco]*logb +
                 K_C_IJ_yNeg[nc][nco]*logbxi);
}

double li22re_logA1rec_yNeg(double next, int nc,
                           double a, double b, double logb, double logbxi, int ncmax)
{
  if (nc < ncmax)
    return
      WTFli22logA1recint_yNeg(next, 0, nc, b, logb, logbxi, ncmax) +
      li22re_logA1rec_yNeg(next*a, nc+1, a, b, logb, logbxi, ncmax);
  else
    return WTFli22logA1recint_yNeg(next, 0, nc, b, logb, logbxi, ncmax);
}

/*  y < 0,  x*y > 0  */
double li22re_logA1ff2_yNeg(double x, double y, int ncmax)
{
  check_range(y > 0 || x*y < 0, "logA1ff2_yNeg: y > 0 || x*y < 0");
  double a = log(-y);
  double b = -log(x*y);
  double logb = log(b);
  double logbxi = log(b + li_constants.xi0);

  return li22re_logA1rec_yNeg(1.,0,a,b,logb,logbxi, ncmax);
}


double li22re_crli12rec(double part1, double part2, double x, double xy, int n, int nn)
{
  if (n < nn)
    return
      part1 +
      li22re_crli12rec(part1*x*WTFcclogA1ff11[n]+part2,
                       part2*xy*WTFcclogA1ff12[n], x, xy, n+1, nn);
  else
    return part1;
}

double li22re_crli12(double x, double y, int nn)
{
  double xy = x*y;
  return li22re_crli12rec(x*xy/2., x*xy*xy/12., x, xy, 0, nn);
}

double li22re_logA1ff1(double x, double y)
{
  double xp = x*li_constants.ccxzero;
  int nmax = 90;
  return li22re_fast(xp,y,nmax) + li_constants.xi0*li22re_crli12(xp,y,100);
}

// This is the log expansion of li22 usable around (1., 1.) .
double li22re_logA1(double x, double y, int ncmax)
{
  double f1_f2 = li22re_logA1ff1(x,y);
  if (y < 0 && x*y > 0)
    f1_f2 += li22re_logA1ff2_yNeg(x,y,ncmax);
  else
    f1_f2 += li22re_logA1ff2(x,y,ncmax);
  return f1_f2;
}

/* Make explicit REAL!!! */
double li22re_li22holderstuffle(double x, double y, int ncmax)
{
  check_range(x > 1.0 && y > 1.0, "li22holderstuffle: x>1 && y>1");
  double xy = x*y;
  return li2re(x)*li2re(y) -li22re_li22holder(y,x,ncmax)-li4re(xy);
}

/* REAL version!!!  */
//This is Li22(x,1)
double li22re_li22x1(double x)
{
  check_range(x > 1.0, "li22x1: x>1");
  double omx = 1.-x;
  double logomx = log(omx);
  double logomxsq = logomx*logomx;
  double arg1 = 1./omx;
  double arg2 = -x/omx;
  double li2x = li2re(x);
  if (x < 0)
    return li_constants.li22x1c1*(li4re(arg1)+li4re(arg2)+li4re(x))
      + li_constants.li22x1c2*li3re(x)*logomx
      + li_constants.li22x1c3*li2x*li2x
      + logomxsq*( li_constants.li22x1c4*logomxsq + li_constants.li22x1c5*logomx*log(-x) + li_constants.li22x1c6)
      + li_constants.li22x1c7*logomx + li_constants.li22x1c8;
  else
    return li_constants.li22x1c1*(li4re(arg1)+li4re(arg2)+li4re(x))
      + li_constants.li22x1c2*li3re(x)*logomx
      + li_constants.li22x1c3*li2x*li2x
      + logomxsq*( li_constants.li22x1c4*logomxsq + li_constants.li22x1c5*logomx*log(x) + li_constants.li22x1c6)
      + li_constants.li22x1c7*logomx + li_constants.li22x1c8;
}

double li22re_li22smallastuffle(double a, double b)
{
  double ab = a*b;
  check_range(a > 1.0 && b > 1.0, "li22re_li22smallastuffle: a > 1.0 && b > 1.0");
  return li2re(a)*li2re(b)-li4re(ab)-li22re_smalla(b,a);
}

double li22re_li22xx(double x)
{
  check_range(x > 1.0, "li22re_li22xx: x > 1.0");
  double xsq = x*x;
  double li2c = li2re(x);
  return 0.5*(li2c*li2c - li4re(xsq));
}

//This is Li22(1/y,y)
static const double WTFli22iyyc1 = 9.86960440108935861883;  // Pi^2
static const double WTFli22iyyc2 = -1.08232323371113819152;  // -Pi^4/90
double li22re_li22iyy(double y)
{
  check_range(y > 0 && y<1, "li22re_li22iyy: y > 0 && y < 1");
  if (y < 0)
    {
      double logc = log(-y);
      double li2c = li2re(y);
      return 3.*li4re(y) - 0.5*li2c*(li2c+logc*logc+WTFli22iyyc1) + WTFli22iyyc2;
    }
  else
    {
      double logy = log(y);
      double logysq = logy*logy;
      double li2y = li2re(y);
      return 3.*li4re(y) - 0.5*( logysq*( WTFli22iyyc1 + li2y) + li2y*li2y) + WTFli22iyyc2 ;
    }
}


//This is Li22(1,y)
static const double WTFli221yc1 = 2.;
static const double WTFli221yc2 = -2.;
static const double WTFli221yc3 = -0.5;
static const double WTFli221yc4 = 0.166666666666666666667;
static const double WTFli221yc5 = -0.333333333333333333333;
static const double WTFli221yc6 = -1.64493406684822643647;  // -Pi^2/6
static const double WTFli221yc7 = 2.40411380631918857080; // 2*Zeta(3)
static const double WTFli221yc8 = -2.16464646742227638303;  // -Pi^4/45
static const double WTFli221yc9 = 1.64493406684822643647;  // Pi^2/6
double li22re_li221y(double y)
{
  check_range(y > 1.0, "li22_li221y: y > 1.0");
  double omy = 1.-y;
  double logomy = log(omy);
  double logomysq = logomy*logomy;
  double arg1 = 1./omy;
  double arg2 = -y/omy;
  double li2y = li2re(y);
  return WTFli221yc1*(li4re(arg1)+li4re(arg2)) + li4re(y) +
    WTFli221yc2*li3re(y)*logomy +
    (WTFli221yc9 + WTFli221yc3*li2y)*li2y +
    logomysq*( WTFli221yc4*logomysq + WTFli221yc5*logomy*log(y) + WTFli221yc6) +
    WTFli221yc7*logomy + WTFli221yc8;
}





double li22re(double x, double y)
{
  /* printf("Li22(x=%f,y=%f), x*y = %f\n",x,y, x*y);  */
  double absxy = fabs(x*y);
  double absx = fabs(x);
  double absy = fabs(y);
  int ncmax;

  /* Regions A/B */
  if (absxy < 0.7)
    {
      if      (absxy < 0.3) ncmax = 20;
      else if (absxy < 0.5) ncmax = 35;
      else                  ncmax = 65;
      if (absy > 1.15 || absx < 0.25)
        return li22re_fast(x,y,ncmax); /* A (Li22fast) */
      else
        return li22re_stuffle(x,y,ncmax); /* B (Stuffle) */
    }

  /* Special cases X */
  else if (fabs(y - x)    < li_constants.epsdif)
    return li22re_li22xx(x);     /* X (x=y) */
  else if (fabs(y - 1./x) < li_constants.epsdif)
    return li22re_li22iyy(y);    /* X (y=1/x) */
  else if (fabs(y - 1.)   < li_constants.epsdif)
    return li22re_li22x1(x);     /* X (y=1) */
  else if (fabs(x - 1.)   < li_constants.epsdif)
    return li22re_li221y(y);     /* X (x=1) */

  /* This case is not needed */
  else if (absxy > 1.42857)   region_unimplemented("C,D");

  /* 0.7 < |x*y| < 1.42857 */
  else
    {
      if ( absx < 0.7)
        return li22re_smalla(x,y); /* Ea (diagonal) */
      else if (absy < 0.7)    region_unimplemented("Eb (diagonal+stuffle)");
      else if (absx > 10./7.) region_unimplemented("Ec (diagonal+inversion)");
      else if (absy > 10./7.) region_unimplemented("Ed (diagonal+inv+st)");

      /* |x,y|>=7/10  || |x,y|<=10/7  */
      else
        {

          if (fabs(x-y) < 1e-14) region_unimplemented("X (x=y (internal))");
          if (x < 0 && x*y < 0)
            return li22re_li22holder(x,y,100); /* F (Holder) */
          if (x*y < 0)
            return li22re_li22holderstuffle(x,y,100); /* Fb (Holder+stuffle) */
          if (x > 0 && x*y > 0)
            return li22re_logA1(x,y,40); /* H (LogA1 center) */
          if(x > 0 || x*y > 0)
              return li22re_logA1(x,y,60); /* H (LogA1 edge) */
          fprintf(stderr, "No implementation for Li22(x=%lf,y=%lf), x*y = %lf, exiting!\n", x,y, x*y);
          exit( EXIT_FAILURE );
        }
    }

  return 0.0/0.0;
}
