#include "SSm0_internal.h"

double tldI0liMR(double x, double y)
{

  double sqrOneMy2 = sqrt(1-y*y);

  /* polynomials from Li arguments */
  double p1 = 1 - x;
  double p3 = 1 + x;
  double p4 = x - y;
  double p5 = 1 - y;
  double p7 = 1 + y;
  double p8 = 1 + x*x - 2*x*y;
  double p9 = 1 - x*y;
  double p10 = sqrOneMy2;
  double p11 = 1 - x*x - 2*x*y + 2*x*x*y*y;

  double inv2 = 1/((double)2);

  double PI = M_PI;

  double wIliMR[63];

  wIliMR[1] = log(x);
  wIliMR[2] = atan2( - p10,p4);
  wIliMR[3] = rat(1,3);
  wIliMR[4] = rat(4,3);
  wIliMR[5] = atan2(p10,y);
  wIliMR[6] = rat(-4,3);
  wIliMR[7] = atan2(p10,p4);
  wIliMR[8] = rat(-8,3);
  wIliMR[9] = rat(-2,3);
  wIliMR[10] = log(p1);
  wIliMR[11] = li2re2(p3*p4*pow(p8,-1), - p10*p3*pow(p8,-1));
  wIliMR[12] = li2re2(p3*p4*pow(p8,-1),p10*p3*pow(p8,-1));
  wIliMR[13] = li2re2(inv2*p1, - inv2*p7*pow(p10,-1)*p1);
  wIliMR[14] = li2re2(inv2*p1,inv2*p7*pow(p10,-1)*p1);
  wIliMR[15] = li2re2( - x*pow(p1,-1)*p5, - x*p10*pow(p1,-1));
  wIliMR[16] = li2re2( - x*pow(p1,-1)*p5,x*p10*pow(p1,-1));
  wIliMR[17] = li2re2(1, - p7*pow(p10,-1));
  wIliMR[18] = li2re2(1,p7*pow(p10,-1));
  wIliMR[19] = rat(-1,3);
  wIliMR[20] = rat(-7,3);
  wIliMR[21] = rat(2,3);
  wIliMR[22] = log(p3);
  wIliMR[23] = log(p8);
  wIliMR[24] = rat(-1,6);
  wIliMR[25] = li3re2(inv2, - inv2*p7*pow(p10,-1));
  wIliMR[26] = rat(8,3);
  wIliMR[27] = li3re2(inv2,inv2*p7*pow(p10,-1));
  wIliMR[28] = li3re2(pow(p1,-1)*p11*pow(p3,-1), - 2*x*p10*pow(p1,-1)*pow(p3,-1)*p9);
  wIliMR[29] = li3re2(pow(p1,-1)*p11*pow(p3,-1),2*x*p10*pow(p1,-1)*pow(p3,-1)*p9);
  wIliMR[30] = li3re2(inv2*p1, - inv2*p7*pow(p10,-1)*p1);
  wIliMR[31] = li3re2(inv2*p1,inv2*p7*pow(p10,-1)*p1);
  wIliMR[32] = li3re2( - x*pow(p1,-1)*p5, - x*p10*pow(p1,-1));
  wIliMR[33] = li3re2( - x*pow(p1,-1)*p5,x*p10*pow(p1,-1));
  wIliMR[34] = li3re2(x*p4*pow(p8,-1), - x*p10*pow(p8,-1));
  wIliMR[35] = li3re2(x*p4*pow(p8,-1),x*p10*pow(p8,-1));
  wIliMR[36] = 1/(1 - 4*x*y + 2*pow(x,2) + 4*pow(x,2)*pow(y,2) - 4*pow(x,3)*y + pow(x,4));
  wIliMR[37] = li2im2(inv2, - inv2*p7*pow(p10,-1));
  wIliMR[38] = li2im2(inv2,inv2*p7*pow(p10,-1));
  wIliMR[39] = li2im2(pow(p1,-1)*p11*pow(p3,-1), - 2*x*p10*pow(p1,-1)*pow(p3,-1)*p9);
  wIliMR[40] = li2im2(inv2*p1, - inv2*p7*pow(p10,-1)*p1);
  wIliMR[41] = li2im2(inv2*p1,inv2*p7*pow(p10,-1)*p1);
  wIliMR[42] = li2im2( - x*pow(p1,-1)*p5, - x*p10*pow(p1,-1));
  wIliMR[43] = li2im2( - x*pow(p1,-1)*p5,x*p10*pow(p1,-1));
  wIliMR[44] = 1/(2 - 8*x*y + 4*pow(x,2) + 8*pow(x,2)*pow(y,2) - 8*pow(x,3)*y + 2*pow(x,4));
  wIliMR[45] = li2im2(pow(p1,-1)*p11*pow(p3,-1),2*x*p10*pow(p1,-1)*pow(p3,-1)*p9);
  wIliMR[46] = 1/(3 - 12*x*y + 6*pow(x,2) + 12*pow(x,2)*pow(y,2) - 12*pow(x,3)*y + 3*pow(x,4));
  wIliMR[47] = 1/(6 - 24*x*y + 12*pow(x,2) + 24*pow(x,2)*pow(y,2) - 24*pow(x,3)*y + 6*pow(x,4));
  wIliMR[48] = wIliMR[23]*wIliMR[21];
  wIliMR[49] = wIliMR[22]*wIliMR[6];
  wIliMR[50] = wIliMR[1]*wIliMR[4];
  wIliMR[51] = wIliMR[10]*wIliMR[20];
  wIliMR[48] = wIliMR[51] + wIliMR[50] + wIliMR[48] + wIliMR[49];
  wIliMR[48] = wIliMR[5]*wIliMR[48];
  wIliMR[49] = PI*wIliMR[8];
  wIliMR[50] = wIliMR[4]*wIliMR[7];
  wIliMR[51] = wIliMR[2]*wIliMR[6];
  wIliMR[49] = wIliMR[51] + wIliMR[49] + wIliMR[50];
  wIliMR[49] = wIliMR[1]*wIliMR[49];
  wIliMR[50] = wIliMR[2] + PI;
  wIliMR[50] = wIliMR[50]*wIliMR[21];
  wIliMR[51] = wIliMR[7]*wIliMR[9];
  wIliMR[50] = wIliMR[51] + wIliMR[50];
  wIliMR[50] = wIliMR[22]*wIliMR[50];
  wIliMR[52] = wIliMR[7]*wIliMR[6];
  wIliMR[53] = wIliMR[2]*wIliMR[4];
  wIliMR[52] = wIliMR[53] + 4*PI + wIliMR[52];
  wIliMR[52] = wIliMR[10]*wIliMR[52];
  wIliMR[48] = wIliMR[48] + wIliMR[52] + wIliMR[49] + wIliMR[50];
  wIliMR[48] = wIliMR[5]*wIliMR[48];
  wIliMR[49] = wIliMR[6]*PI;
  wIliMR[50] = wIliMR[7]*wIliMR[21];
  wIliMR[52] = wIliMR[2]*wIliMR[19];
  wIliMR[50] = wIliMR[52] + wIliMR[49] + wIliMR[50];
  wIliMR[50] = wIliMR[2]*wIliMR[50];
  wIliMR[52] = wIliMR[15] + wIliMR[16];
  wIliMR[53] = wIliMR[7]*PI;
  wIliMR[53] = wIliMR[53] + wIliMR[11] + wIliMR[12] + wIliMR[14] + wIliMR[13] + wIliMR[52];
  wIliMR[53] = wIliMR[4]*wIliMR[53];
  wIliMR[54] = wIliMR[17] + wIliMR[18];
  wIliMR[55] = wIliMR[6]*wIliMR[54];
  wIliMR[56] = wIliMR[19]*pow(wIliMR[7],2);
  wIliMR[50] = wIliMR[50] + wIliMR[53] + wIliMR[55] + wIliMR[56];
  wIliMR[50] = wIliMR[10]*wIliMR[50];
  wIliMR[53] = wIliMR[37] - wIliMR[38] - wIliMR[40] + wIliMR[41] + wIliMR[42] - wIliMR[43];
  wIliMR[55] = wIliMR[2] - wIliMR[7];
  wIliMR[56] = wIliMR[55]*wIliMR[1];
  wIliMR[56] = wIliMR[53] + wIliMR[56];
  wIliMR[57] = wIliMR[46]*wIliMR[56];
  wIliMR[58] = 2*wIliMR[1];
  wIliMR[59] =  - wIliMR[58] + 2*wIliMR[10] + wIliMR[22] - wIliMR[23];
  wIliMR[59] = wIliMR[59]*wIliMR[5];
  wIliMR[60] = wIliMR[46]*wIliMR[59];
  wIliMR[61] = wIliMR[39] - wIliMR[45];
  wIliMR[62] =  - wIliMR[47]*wIliMR[61];
  wIliMR[57] = wIliMR[60] + wIliMR[57] + wIliMR[62];
  wIliMR[57] = wIliMR[57]*pow(x,2);
  wIliMR[56] =  - wIliMR[59] - wIliMR[56];
  wIliMR[56] = wIliMR[36]*wIliMR[56];
  wIliMR[59] = wIliMR[44]*wIliMR[61];
  wIliMR[56] = wIliMR[57] + wIliMR[59] + wIliMR[56];
  wIliMR[56] = x*y*wIliMR[56];
  wIliMR[55] = wIliMR[55]*wIliMR[58];
  wIliMR[53] = wIliMR[55] + wIliMR[45] + 2*wIliMR[53];
  wIliMR[53] = wIliMR[46]*wIliMR[53];
  wIliMR[55] = wIliMR[36]*wIliMR[19]*wIliMR[39];
  wIliMR[53] = wIliMR[56] + 2*wIliMR[60] + wIliMR[55] + wIliMR[53];
  wIliMR[53] = sqrOneMy2*x*wIliMR[53];
  wIliMR[55] = wIliMR[23]*PI;
  wIliMR[56] = wIliMR[9]*wIliMR[55];
  wIliMR[57] = wIliMR[7]*wIliMR[23];
  wIliMR[58] = wIliMR[3]*wIliMR[57];
  wIliMR[59] = wIliMR[2]*wIliMR[23]*wIliMR[24];
  wIliMR[56] = wIliMR[59] + wIliMR[56] + wIliMR[58];
  wIliMR[56] = wIliMR[2]*wIliMR[56];
  wIliMR[58] = wIliMR[4]*PI;
  wIliMR[59] = wIliMR[2]*wIliMR[3];
  wIliMR[51] = wIliMR[59] + wIliMR[51] + wIliMR[58];
  wIliMR[51] = wIliMR[2]*wIliMR[51];
  wIliMR[58] = wIliMR[7]*wIliMR[3];
  wIliMR[49] = wIliMR[49] + wIliMR[58];
  wIliMR[49] = wIliMR[7]*wIliMR[49];
  wIliMR[49] = wIliMR[49] + wIliMR[51];
  wIliMR[49] = wIliMR[1]*wIliMR[49];
  wIliMR[51] = wIliMR[21]*wIliMR[55];
  wIliMR[55] = wIliMR[24]*wIliMR[57];
  wIliMR[51] = wIliMR[51] + wIliMR[55];
  wIliMR[51] = wIliMR[7]*wIliMR[51];
  wIliMR[52] = wIliMR[52] + wIliMR[54];
  wIliMR[52] = wIliMR[4]*wIliMR[22]*wIliMR[52];
  wIliMR[54] = wIliMR[27] + wIliMR[25];
  wIliMR[54] = wIliMR[26]*wIliMR[54];
  wIliMR[55] = wIliMR[29] + wIliMR[28];
  wIliMR[55] = wIliMR[19]*wIliMR[55];
  wIliMR[57] = wIliMR[30] + wIliMR[31] + wIliMR[33] + wIliMR[32];
  wIliMR[57] = wIliMR[8]*wIliMR[57];
  wIliMR[58] = wIliMR[35] + wIliMR[34];
  wIliMR[58] = wIliMR[6]*wIliMR[58];

  wIliMR[0] = wIliMR[48] + wIliMR[49] + wIliMR[50] + wIliMR[51] + 
    wIliMR[52] + wIliMR[53] + wIliMR[54] + wIliMR[55] + wIliMR[56] + 
    wIliMR[57] + wIliMR[58];

  return wIliMR[0];
}
