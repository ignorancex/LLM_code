#include "SSm0.h"
#include "SSm0_internal.h"

/* finite part gluons */
double tldS0li22(double, double);
double tldS0liRe(double, double);
double tldS0liMR(double, double);
/* finite part quarks */
double tldI0liRe(double, double);
double tldI0liMR(double, double);

/* same file */
double tldSm1(double, double);
double tldIm1(double, double);

const double LOG2  = 0.69314718055994530942;
const double ZETA3 = 1.20205690315959428539;

double SSm0_tldS(int epord, double x, double y)
{
  if (x < 0 || x > 1 || y < -1 || y > 1)
    {
      fprintf(stderr, "ERROR: input parameters out of range be=[0,1], cos(del)=[-1,1]\n");
      exit( EXIT_FAILURE );
    }

  switch (epord)
    {
    case -3:
      return -1./4.;

    case -2:
      return 1./24. + log((1-x*y)*(1-x*y)/(1-x*x))/2. ;

    case -1:
      return 119./36. - M_PI*M_PI/24. - (11.*LOG2)/6. - log((1 + x)/(1-x))/x +
        (13*(log(1 - x) + log(1 + x) - 2*log(1 - x*y)))/12. +
        (pow(log(1 - x),2) - 2*log(1 - x)*log(1 + x) - pow(log(1 + x),2) +
         4*log(1 + x)*log(1 - x*y) - 2*pow(log(1 - x*y),2) -
         4*li2re((x*(1 + y))/(1 + x)) + 4*li2re((x*(1 - y))/(1 - x*y)))/2.;

    case 0:
      return tldSm1(x, y);

    case 1:
      return tldS0li22(x, y) + tldS0liRe(x, y) + tldS0liMR(x, y);

    default:
      fprintf(stderr, "Unimplemented order ep^%d, choose one from ep^-3...ep^1\n", epord);
      exit( EXIT_FAILURE );
    }

}


double SSm0_tldI(int epord, double x, double y)
{
  if (x < 0 || x > 1 || y < -1 || y > 1)
    {
      fprintf(stderr, "ERROR: input parameters out of range be = [0,1], cos(del) = [-1,1]\n");
      exit( EXIT_FAILURE );
    }

  switch (epord)
    {
    case -3:
      return 0;

    case -2:
      return -1./12.;

    case -1:
      return 25./72. - LOG2/3. -log(1 - x)/6. - log(1 + x)/6. + log(1 - x*y)/3.;

    case 0:
      return tldIm1(x, y);

    case 1:
      return tldI0liRe(x, y) + tldI0liMR(x, y);

    default:
      fprintf(stderr, "Unimplemented order ep^%d, choose one from ep^-3...ep^1\n", epord);
      exit( EXIT_FAILURE );
    }

}



double tldSm1(double x, double y)
{
  /* polynomials from Li arguments */
  double p1 = 1 - x;
  double p3 = 1 + x;
  double p4 = 1 - y;
  double p5 = 1 + y;
  double p6 = 1 - x*y;

  double inv2 = 1/((double)2);

  double PI = M_PI;

  double w[68];

    w[1] = li2re(pow(p3,-1));
    w[2] = log(p3);
    w[3] = log(p6);
    w[4] = li2re(inv2*p4);
    w[5] = log(p1);
    w[6] = li2re(p1*pow(p6,-1));
    w[7] = li2re(inv2*p1);
    w[8] = pow(x,-1);
    w[9] = li2re( - x);
    w[10] = li2re(x*p4*pow(p6,-1));
    w[11] = rat(-10,3);
    w[12] = li2re(x*p5*pow(p3,-1));
    w[13] = rat(10,3);
    w[14] = li2re(x);
    w[15] = li2re(p1);
    w[16] = li3re(pow(p3,-1)*p6);
    w[17] = li3re(inv2*p4);
    w[18] = li3re(inv2*p3*p4*pow(p6,-1));
    w[19] = li3re(inv2*p5);
    w[20] = li3re(p1*pow(p3,-1));
    w[21] = li3re(p1*pow(p6,-1));
    w[22] = li3re(inv2*p1*p5*pow(p6,-1));
    w[23] = li3re(x*p4*pow(p6,-1));
    w[24] = li3re(x*p5*pow(p3,-1));
    w[25] = rat(-1,3);
    w[26] = log(p4);
    w[27] = rat(-13,12);
    w[28] = log(p5);
    w[29] = rat(13,6);
    w[30] = rat(-19,6);
    w[31] = rat(1,4);
    w[32] = rat(31,9);
    w[33] = rat(19,6);
    w[34] = rat(3,2);
    w[35] = rat(7,12);
    w[36] = rat(-11,12);
    w[37] = rat(-7,3);
    w[38] = rat(1,2);
    w[39] = rat(5,3);
    w[40] = rat(-62,9);
    w[41] = rat(-13,6);
    w[42] = log(2);
    w[43] = rat(-11,3);
    w[44] = rat(22,3);
    w[45] = rat(11,6);
    w[46] = rat(269,36);
    w[47] = rat(-589,108);
    w[48] = rat(1,3);
    w[49] = rat(1,12);
    w[50] = rat(69,8);
   w[51] = 4*w[26];
   w[52] = w[51] + 4*w[28];
   w[53] =  - w[43] + 2*w[42];
   w[54] = 4*w[8] - w[53] + w[52];
   w[54] = w[42]*w[54];
   w[55] = w[6] - w[15];
   w[56] = 4*w[7];
   w[57] =  - w[56] + 4*w[9];
   w[58] = 4*w[4];
   w[59] = w[51]*w[28];
   w[60] = pow(PI,2);
   w[61] = w[31]*w[60];
   w[62] = w[8]*w[30];
   w[63] = w[5]*w[25];
   w[63] = w[63] - w[8] + w[27] + 2*w[26];
   w[63] = w[5]*w[63];
   w[54] = w[63] + w[54] + w[62] - w[59] - 4*w[12] + w[61] - 2*w[14] + 
   w[32] - w[58] - w[57] + 6*w[55];
   w[54] = w[5]*w[54];
   w[55] = 4*w[42];
   w[61] =  - w[55] + 7*w[28];
   w[62] = w[8]*w[34];
   w[63] = w[2]*w[33];
   w[62] = w[63] - 3*w[5] + w[62] + w[35] - w[61];
   w[62] = w[2]*w[62];
   w[63] = 2*w[8];
   w[64] = w[5] - w[63] + w[29] - w[51];
   w[64] = w[5]*w[64];
   w[65] = 4*w[14];
   w[66] = w[36]*w[60];
   w[67] = w[8]*w[33];
   w[53] =  - w[42]*w[53];
   w[53] = w[62] + w[64] + w[53] + w[67] + 6*w[12] + w[66] + 7*w[9] + 
   w[65] - 4*w[6] - w[56] + 8*w[15] - 3*w[1] + w[32] + w[58];
   w[53] = w[2]*w[53];
   w[56] = w[2] - w[26];
   w[58] = w[8]*w[38];
   w[62] = w[3]*w[37];
   w[56] = w[62] + 5*w[5] + w[58] + w[39] - w[61] - 3*w[56];
   w[56] = w[3]*w[56];
   w[52] = w[55] + w[44] - w[52];
   w[52] = w[42]*w[52];
   w[51] =  - w[2] + 8*w[5] - w[8] + w[11] + 14*w[28] - w[55] + w[51];
   w[51] = w[2]*w[51];
   w[58] = w[14] + w[15];
   w[61] = w[9] - w[1];
   w[62] = w[41]*w[60];
   w[55] =  - 2*w[5] - 5*w[26] - w[55];
   w[55] = w[5]*w[55];
   w[51] = w[56] + w[51] + w[55] + w[52] + w[59] - 2*w[12] + w[62] - 7*
   w[6] - 5*w[10] + w[40] - 14*w[61] + 9*w[58];
   w[51] = w[3]*w[51];
   w[52] = w[48]*w[60];
   w[52] = w[12] + w[52] - w[65] + w[10] + w[57];
   w[52] = w[8]*w[52];
   w[55] =  - w[17] + w[18] - w[19] - w[20] + w[21] + w[22];
   w[56] = w[23] - w[24];
   w[57] = w[50]*ZETA3;
   w[58] = w[10]*w[11];
   w[59] = w[49]*w[60];
   w[60] = w[12]*w[13];
   w[61] = w[45] - w[63];
   w[61] = w[42]*w[61];
   w[61] = w[46] + w[61];
   w[61] = w[42]*w[61];

   w[0] =  - 10*w[16] + w[47] + w[51] + w[52] + w[53] + w[54] -
     4*w[55] + 6*w[56] + w[57] + w[58] + w[59] + w[60] + w[61];

   return w[0];
}

double tldIm1(double x, double y)
{

  /* polynomials from Li arguments */
  double p2 = 1 + x;
  double p3 = 1 - y;
  double p4 = 1 + y;
  double p5 = 1 - x*y;

  double w[25];

  w[1] = li2re(x*p3*pow(p5,-1));
  w[2] = rat(2,3);
  w[3] = li2re(x*p4*pow(p2,-1));
  w[4] = rat(-2,3);
  w[5] = log(1 - x);
  w[6] = rat(1,6);
  w[7] = rat(1,3);
  w[8] = pow(x,-1);
  w[9] = rat(37,36);
  w[10] = log(p2);
  w[11] = rat(-1,3);
  w[12] = rat(-1,6);
  w[13] = log(p5);
  w[14] = rat(-37,18);
  w[15] = log(2);
  w[16] = rat(4,3);
  w[17] = rat(59,36);
  w[18] = rat(-131,216);
  w[19] = w[15]*w[4];
  w[19] = w[19] + w[9];
  w[20] = w[5] + w[8];
  w[20] = w[11]*w[20];
  w[21] = w[13]*w[2];
  w[22] = w[10]*w[12];
  w[20] = w[22] + w[21] + w[20] + w[19];
  w[20] = w[10]*w[20];
  w[21] = w[7]*w[8];
  w[22] = w[5]*w[6];
  w[19] = w[22] + w[21] + w[19];
  w[19] = w[5]*w[19];
  w[21] = w[13]*w[16];
  w[22] = w[15]*w[7];
  w[21] = w[22] + w[17] + w[21];
  w[21] = w[15]*w[21];
  w[22] = w[2]*w[1];
  w[23] = w[4]*w[3];
  w[24] = w[13]*w[11];
  w[24] = w[14] + w[24];
  w[24] = w[13]*w[24];

  w[0] = w[18] + w[19] + w[20] + w[21] + w[22] + w[23] + w[24];

  return w[0];
}
