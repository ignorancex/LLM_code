/*

  Implementation of the Re[Li2(x)] and Im[Li2(x)] functions for real and complex arguments.

  Based on the code provided with the paper
  by Hjalte Frellesvig, Damiano Tommasini, and Christopher Wever [ArXiv:1601.02649]

*/

#include "SSm0_internal.h"

/* BernoulliB[2 m]/Factorial[2 m + 1] */
static const double ccli2logA0[100] =
  {
    1.,
    0.0277777777777777777778,
    -0.000277777777777777777778,
    4.72411186696900982615e-6,
    -9.18577307466196355085e-8,
    1.8978869988970999072e-9,
    -4.06476164514422552681e-11,
    8.92169102045645255522e-13,
    -1.99392958607210756872e-14,
    4.51898002961991819165e-16,
    -1.03565176121812470145e-17,
    2.39521862102618674574e-19,
    -5.58178587432500933628e-21,
    1.30915075541832128581e-22,
    -3.08741980242674029324e-24,
    7.31597565270220342036e-26,
    -1.74084565723400074099e-27,
    4.15763564461389971962e-29,
    -9.96214848828462210319e-31,
    2.39403442489616530052e-32,
    -5.76834735536739008429e-34,
    1.39317947964700797783e-35,
    -3.37212196548508947047e-37,
    8.17820877756210262176e-39,
    -1.98701083115238592556e-40,
    4.83577851804055089629e-42,
    -1.17869372487183843267e-43,
    2.877096408117257145e-45,
    -7.03205909815602801496e-47,
    1.72086031450331462909e-48,
    -4.21607239056044549168e-50,
    1.03404064051330395739e-51,
    -2.53866306259946531616e-53,
    6.23855317692459088784e-55,
    -1.53443980691346503917e-56,
    3.777294635578550234e-58,
    -9.30586212480468658839e-60,
    2.29434368222418732072e-61,
    -5.66068873941414784857e-63,
    1.39756872198540085454e-64,
    -3.45267347330633889778e-66,
    8.53498373696321710664e-68,
    -2.11106753916379301954e-69,
    5.22446788920815644493e-71,
    -1.29363445303169121491e-72,
    3.20479645174984764253e-74,
    -7.94326694999713716406e-76,
    1.96969401238294242761e-77,
    -4.88642119356886536649e-79,
    1.2127399993287380323e-80,
    -3.01107647674682414162e-82,
    7.47904589777194783774e-84,
    -1.85837941991417435344e-85,
    4.6193425842611789485e-87,
    -1.1486235467130135708e-88,
    2.8570740556428067213e-90,
    -7.10896369091622518708e-92,
    1.76940464275317093111e-93,
    -4.40533971491237081972e-95,
    1.09713120606298044082e-96,
    -2.73313083816123243674e-98,
    6.81053053662527748214e-100,
    -1.69752549739977101562e-101,
    4.23216763434037815455e-103,
    -1.05540011102109065058e-104,
    2.63254505953694582769e-106,
    -6.56803912905139478216e-108,
    1.6390562839944812655e-109,
    -4.09116816995417822688e-111,
    1.02139414028717896929e-112,
    -2.55052340302039905225e-114,
    6.3701938923623999053e-116,
    -1.59133256352353330242e-117,
    3.97605039816258639081e-119,
    -9.93626602127902345807e-121,
    2.48354935273027793979e-122,
    -6.20866996208690084797e-124,
    1.55238189965646443514e-125,
    -3.88213719560680557378e-127,
    9.70987570883174305432e-129,
    -2.42898695457098901872e-130,
    6.07720263182787102327e-132,
    -1.52071433809476079929e-133,
    3.8058825087439879074e-135,
    -9.52632528675517774988e-137,
    2.38482361980845745438e-138,
    -5.97099262734982093322e-140,
    1.49518472872052581311e-141,
    -3.74455227876193378591e-143,
    9.37908338404674472152e-145,
    -2.34949819836149205611e-146,
    5.88630640090960269216e-148,
    -1.47489970707240971116e-149,
    3.696007761196264006e-151,
    -9.26302721807284788094e-153,
    2.32178307158320397184e-154,
    -5.82020071424204647756e-156,
    1.45915330388174209927e-157,
    -3.65855485706352121659e-159,
    9.17408974648312441359e-161
  };

static const double ccli2logA1[100] =
  {
    -1.,
    0.013888888888888888889,
    -0.000069444444444444444444,
    7.8735197782816830436e-7,
    -1.1482216343327454439e-8,
    1.8978869988970999072e-10,
    -3.3873013709535212723e-12,
    6.3726364431831803966e-14,
    -1.2462059912950672305e-15,
    2.5105444608999545509e-17,
    -5.1782588060906235072e-19,
    1.0887357368300848844e-20,
    -2.3257441143020872235e-22,
    5.0351952131473895608e-24,
    -1.1026499294381215333e-25,
    2.4386585509007344735e-27,
    -5.4401426788562523156e-29,
    1.2228340131217352117e-30,
    -2.7672634689679505842e-32,
    6.3000905918320139487e-34,
    -1.4420868388418475211e-35,
    3.3170939991595428044e-37,
    -7.6639135579206578874e-39,
    1.7778714733830657873e-40,
    -4.1396058982341373449e-42,
    9.6715570360811017926e-44,
    -2.2667187016766123705e-45,
    5.3279563113282539722e-47,
    -1.2557248389564335741e-48,
    2.9670005422470941881e-50,
    -7.0267873176007424861e-52,
    1.6678074846988773506e-53,
    -3.9666610353116645565e-55,
    9.4523532983705922543e-57,
    -2.2565291278139191752e-58,
    5.3961351936836431914e-60,
    -1.2924808506673175817e-61,
    3.1004644354380909739e-63,
    -7.4482746571238787481e-65,
    1.7917547717761549417e-66,
    -4.3158418416329236222e-68,
    1.0408516752394167203e-69,
    -2.5131756418616583566e-71,
    6.0749626618699493546e-73,
    -1.4700391511723763806e-74,
    3.5608849463887196028e-76,
    -8.6339858152142795261e-78,
    2.0954191621095132209e-79,
    -5.0900220766342347568e-81,
    1.2374897952334061554e-82,
    -3.0110764767468241416e-84,
    7.3323979389921057233e-86,
    -1.7869032883790138014e-87,
    4.3578703625105461778e-89,
    -1.0635403210305681211e-90,
    2.5973400505843697466e-92,
    -6.3472890097466296313e-94,
    1.5521093357483955536e-95,
    -3.7977066507865265687e-97,
    9.2977220852794952612e-99,
    -2.2776090318010270306e-100,
    5.5824020792010471165e-102,
    -1.3689721753223959803e-103,
    3.3588632018574429798e-105,
    -8.2453133673522707077e-107,
    2.0250346611822660213e-108,
    -4.9757872189783293804e-110,
    1.2231763313391651235e-111,
    -3.0082118896721898727e-113,
    7.4014068136752099224e-115,
    -1.8218024307288564659e-116,
    4.4860520368749295108e-118,
    -1.1050920580024536822e-119,
    2.7233221905223194458e-121,
    -6.713693257620961796e-123,
    1.6556995684868519599e-124,
    -4.0846512908466452947e-126,
    1.0080401945821197631e-127,
    -2.4885494843633369063e-129,
    6.1454909549567994015e-131,
    -1.5181168466068681367e-132,
    3.7513596492764635946e-134,
    -9.272648403016834142e-136,
    2.2927003064722818719e-137,
    -5.6704317183066534225e-139,
    1.402837423416739679e-140,
    -3.4715073414824540309e-142,
    8.5930156823018724891e-144,
    -2.1275865220238260147e-145,
    5.2691479685655869222e-147,
    -1.3052767768674955867e-148,
    3.2342342862140674133e-150,
    -8.0157592775674440824e-152,
    1.9871009468797118312e-153,
    -4.927142137272791426e-155,
    1.2219910903069494589e-156,
    -3.0313545386677325404e-158,
    7.5214087828955778313e-160,
    -1.8666096209507761309e-161,
    4.633378659839961825e-163
  };



static const double WTFccli2logA10 =  1.6449340668482264365;
static const double WTFccli2logA12 =  -0.25;

static const double WTFccli2inv0 = -1.6449340668482264365;
static const double WTFccli2inv2 = -0.5;

static const double WTFccli2inone = 1.64493406684822643647;

double SSm0_li2logA0rec(double next, int nc, double lsq)
{
  if (nc < li_constants.LI2LOGA0MAX)
    {
      return next*ccli2logA0[nc] + SSm0_li2logA0rec(next*lsq, nc+1, lsq);
    }
  return next*ccli2logA0[nc];
}

double SSm0_li2logA0(double x)
{
  double mlog = -log(1.0-x);
  double mlsq = mlog*mlog;
  return mlog - 0.25*mlsq + SSm0_li2logA0rec(mlog*mlsq, 1, mlsq);
}

double SSm0_li2logA1rec(double next, int nc, double xsq)
{
  if (nc < li_constants.LINLOGA1MAX)
    {
      return ccli2logA1[nc]*next + SSm0_li2logA1rec(next*xsq, nc+1, xsq);
    }
  return ccli2logA1[nc]*next;
}

/* REAL VERSION */
double SSm0_li2logA1(double a)
{
  double x = -log(a);
  double xsq = x*x;
  return WTFccli2logA10 + WTFccli2logA12*xsq + x*log(fabs(x)) + SSm0_li2logA1rec(x, 0, xsq);
}

/* REAL version */
double SSm0_li2inv(double x)
{
  double ix = 1.0/x;
  if ( x > 0)
    {
      double logx2 = log(x)*log(x);
      double lsq = logx2 - li_constants.Pi2;
      return -SSm0_li2logA0(ix) + WTFccli2inv0 + WTFccli2inv2*lsq;
    }
  else
    {
      double logmx = log(-x);
      return -SSm0_li2logA0(ix) + WTFccli2inv0 + WTFccli2inv2*logmx*logmx;
    }
}

/* real function of real argument */
double li2re(double x)
{
  if (x < 0.5)
    {
      if (fabs(x) < 1.0)
        return SSm0_li2logA0(x);
      else
        return SSm0_li2inv(x);
    }
  else if (fabs(x-1.0) < 1.0)
    {
      if(x == 1.0)
        return WTFccli2inone;
      else
        return SSm0_li2logA1(x);
    }
  else
    return SSm0_li2inv(x);
}



double li2logA0re2_rec(double next, double cos_prev, double cos_prev_prev, int nc, double rsq, double cphi)
{
  double cnphi = 2*cphi*cos_prev - cos_prev_prev; /* cos(n*phi)     */
  double cnnphi = 2*cphi*cnphi - cos_prev;        /* cos((n+1)*phi) */

  if (nc < li_constants.LI2LOGA0MAX)
    return next*cnphi*ccli2logA0[nc] + li2logA0re2_rec(next*rsq, cnnphi, cnphi, nc+1, rsq, cphi);
  else
    return next*cnphi*ccli2logA0[nc];
}


double li2logA0re2(double rex, double imx)
{
  double reAl = - log((1 - rex)*(1 - rex) + imx*imx)/2.;
  double imAl = - atan2(-imx,1 - rex);
  double absAl = hypot(reAl, imAl); /* |al| */
  double argAl = atan2(imAl, reAl); /* arg(al) */
  double absAl2 = absAl*absAl;
  double cAA = cos(argAl);

  double res = reAl - 0.25*(reAl*reAl-imAl*imAl);
  /* for (size_t m = 1; m <= li_constants.LI2LOGA0MAX; m++) */
  /*   { */
  /*     res += pow(absAl,2*m+1.)*cos((2.*m+1.)*argAl)*ccli2logA0[m]; */
  /*   } */
  res += li2logA0re2_rec(absAl*absAl2, cos(2*argAl), cAA, 1, absAl2, cAA);

  return res;
}


double li2logA0im2_rec(double next, double sin_prev, double sin_prev_prev, int nc, double rsq, double cphi)
{
  double snphi = 2*cphi*sin_prev - sin_prev_prev; /* sin(n*phi)     */
  double snnphi = 2*cphi*snphi - sin_prev;        /* sin((n+1)*phi) */

  if (nc < li_constants.LI2LOGA0MAX)
    return next*snphi*ccli2logA0[nc] + li2logA0re2_rec(next*rsq, snnphi, snphi, nc+1, rsq, cphi);
  else
    return next*snphi*ccli2logA0[nc];
}


double li2logA0im2(double rex, double imx)
{
  double reAl = - log((1 - rex)*(1 - rex) + imx*imx)/2.;
  double imAl = - atan2(-imx,1 - rex);
  double absAl = hypot(reAl, imAl); /* |al| */
  double argAl = atan2(imAl, reAl); /* arg(al) */
  double absAl2 = absAl*absAl;

  double res = imAl - 0.5*reAl*imAl;
  /* for (size_t m = 1; m <= li_constants.LI2LOGA0MAX; m++) */
  /*   { */
  /*     res += pow(absAl,2*m+1.)*sin((2.*m+1.)*argAl)*ccli2logA0[m]; */
  /*   } */

  res += li2logA0im2_rec(absAl*absAl2, sin(2*argAl), sin(argAl), 1, absAl2, cos(argAl));

  return res;
}

double li2logA1re2_rec(double next, double cos_prev, double cos_prev_prev, int nc, double rsq, double cphi)
{
  double cnphi = 2*cphi*cos_prev - cos_prev_prev; /* cos(n*phi)     */
  double cnnphi = 2*cphi*cnphi - cos_prev;        /* cos((n+1)*phi) */

  if (nc < li_constants.LINLOGA1MAX)
    return next*cnphi*ccli2logA1[nc] + li2logA1re2_rec(next*rsq, cnnphi, cnphi, nc+1, rsq, cphi);
  else
    return next*cnphi*ccli2logA1[nc];
}


double li2logA1re2(double rex, double imx)
{
  double reAl = - log(rex*rex + imx*imx)/2.;
  double imAl = - atan2(imx, rex);
  double absAl = hypot(reAl, imAl); /* |al| */
  double argAl = atan2(imAl, reAl); /* arg(al) */
  double absx  = hypot(rex, imx); /* |x| */
  double argx  = atan2(imx, rex); /* arg(x) */
  double absAl2 = absAl*absAl;
  double cAA = cos(argAl);
  /* printf("Re[al]  = %f,  Im[al] = %f\n", reAl, imAl); */
  /* printf("Abs[al] = %f, Arg[al] = %f\n", absAl, argAl); */

  double AlLogAl = argAl*argx - log(absx)*log(absAl); /* Re[al*log(al)] */
  double logax = log(absx) ;
  double al2     = logax*logax - argx*argx;
  double res = WTFccli2logA10 + WTFccli2logA12*al2 + AlLogAl;

  /* for (size_t m = 0; m <= li_constants.LINLOGA1MAX; m++) */
  /*   { */
  /*     res += pow(absAl,2*m+1.)*cos((2.*m+1.)*argAl)*ccli2logA1[m]; */
  /*   } */
  res += absAl*cAA*ccli2logA1[0];
  res += li2logA1re2_rec(absAl*absAl2, cos(2*argAl), cAA, 1, absAl2, cAA);

  return res;
}

double li2logA1im2_rec(double next, double sin_prev, double sin_prev_prev, int nc, double rsq, double cphi)
{
  double snphi = 2*cphi*sin_prev - sin_prev_prev; /* sin(n*phi)     */
  double snnphi = 2*cphi*snphi - sin_prev;        /* sin((n+1)*phi) */

  if (nc < li_constants.LINLOGA1MAX)
    return next*snphi*ccli2logA1[nc] + li2logA1re2_rec(next*rsq, snnphi, snphi, nc+1, rsq, cphi);
  else
    return next*snphi*ccli2logA1[nc];
}


double li2logA1im2(double rex, double imx)
{
  double reAl = - log(rex*rex + imx*imx)/2.;
  double imAl = - atan2(imx, rex);
  double absAl = hypot(reAl, imAl); /* |al| */
  double argAl = atan2(imAl, reAl); /* arg(al) */
  double absx  = hypot(rex, imx); /* |x| */
  double argx  = atan2(imx, rex); /* arg(x) */
  double absAl2 = absAl*absAl;
  double sAA = sin(argAl);

  double AlLogAl = -argAl*log(absx) - argx*log(absAl); /* Im[al*log(al)] */
  double logax = log(absx);
  double al2     = 2*argx*logax; /* Im[al^2] */
  double res =  WTFccli2logA12*al2 + AlLogAl;

  /* for (size_t m = 0; m <= li_constants.LINLOGA1MAX; m++) */
  /*   { */
  /*     res += pow(absAl,2*m+1.)*sin((2.*m+1.)*argAl)*ccli2logA1[m]; */
  /*   } */
  res += absAl*sAA*ccli2logA1[0];
  res += li2logA1im2_rec(absAl*absAl2, sin(2*argAl), sAA, 1, absAl2, cos(argAl));

  return res;
}


double li2invRe2(double rex, double imx)
{
  double absx2 = rex*rex + imx*imx;
  double reix = rex/absx2;
  double imix = -imx/absx2;

  double logax = log(absx2);
  double lsq = logax*logax;

  double argmx = atan2(-imx, -rex);

  return -li2logA0re2(reix, imix) + WTFccli2inv0 + argmx*argmx/2. - 1./8.*lsq;
}

double li2invIm2(double rex, double imx)
{
  double absx2 = rex*rex + imx*imx;
  double reix = rex/absx2;
  double imix = -imx/absx2;

  double argmx = atan2(-imx, -rex);

  return -li2logA0im2(reix, imix) - argmx*log(absx2)/2.;
}

/* real part of Li2(rex + I*imx) */
double li2re2(double rex, double imx)
{
  double ax  = hypot(rex, imx);      /* |x|   */
  double axb = hypot(1 - rex, -imx); /* |1-x| */
  if (rex < 0.5)
    {
      if (ax < 1.0)
        return li2logA0re2(rex,imx);
      else
        return li2invRe2(rex, imx);
    }
  else if (axb < 1.0)
    {
      if (rex == 1.0 && imx == 0.0)
        return WTFccli2inone;
      else
        return li2logA1re2(rex,imx);
    }
  else
    return li2invRe2(rex,imx);
}

/* imaginary part of Li2(rex + I*imx) */
double li2im2(double rex, double imx)
{
  double ax  = hypot(rex, imx);      /* |x|   */
  double axb = hypot(1 - rex, -imx); /* |1-x| */
  if (rex < 0.5)
    {
      if (ax < 1.0)
        return li2logA0im2(rex,imx);
      else
        return li2invIm2(rex, imx);
    }
  else if (axb < 1.0)
    {
      if (rex == 1.0 && imx == 0.0)
        return WTFccli2inone;
      else
        return li2logA1im2(rex,imx);
    }
  else
    return li2invIm2(rex,imx);
}
