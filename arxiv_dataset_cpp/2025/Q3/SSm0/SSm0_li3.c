/*

  Implementation of the Re[Li3(x)] function for real and complex arguments.

  Based on the code provided with the paper
  by Hjalte Frellesvig, Damiano Tommasini, and Christopher Wever [ArXiv:1601.02649]

*/

#include "SSm0_internal.h"

static const double ccli3logA0[100] =
  {
    1.,
    -0.375,
    0.0787037037037037037037,
    -0.00868055555555555555556,
    0.00012962962962962962963,
    0.0000810185185185185185185,
    -3.41935716085375949322e-6,
    -1.32865646258503401361e-6,
    8.66087175610985134795e-8,
    2.52608759553203997648e-8,
    -2.14469446836406476093e-9,
    -5.14011062201297891534e-10,
    5.24958211460082943639e-11,
    1.08877544066363183754e-11,
    -1.27793960944936953056e-12,
    -2.36982417730874520998e-13,
    3.10435788796546229428e-14,
    5.26175862991250608413e-15,
    -7.53847954994926536599e-16,
    -1.18623225777522852531e-16,
    1.83169799654913833821e-17,
    2.70681710318373501515e-18,
    -4.45543389782963882643e-19,
    -6.23754849225569465037e-20,
    1.08515215348745349131e-20,
    1.44911748660360819307e-21,
    -2.64663397544589903347e-22,
    -3.38976534885101047219e-23,
    6.46404773360331088903e-24,
    7.97583448960241242421e-25,
    -1.58091787902874833559e-25,
    -1.88614997296228681931e-26,
    3.8715536638418473304e-27,
    4.48011750023456073049e-28,
    -9.49303387191183612642e-29,
    -1.0682813809077381224e-29,
    2.33044789361030518601e-30,
    2.55607757265197540806e-31,
    -5.72742160613725968447e-32,
    -6.13471321379642358259e-33,
    1.40908086040689448401e-33,
    1.47642223976665341443e-34,
    -3.47010516489959160555e-35,
    -3.56210662409746357967e-36,
    8.55369656823692105755e-37,
    8.61357241183691332131e-38,
    -2.11030731665291450123e-38,
    -2.0871470317736616066e-39,
    5.21069983399591768727e-40,
    5.06687761943858411019e-41,
    -1.28760990549822459562e-41,
    -1.232193564693024988e-42,
    3.18413433597311711513e-43,
    3.00130346610884786096e-44,
    -7.87951702827426663048e-45,
    -7.32118282422694898505e-46,
    1.95115865383390980642e-46,
    1.78832537410002436587e-47,
    -4.83454010781168240474e-48,
    -4.37385329936259134894e-49,
    1.19859725604737009748e-49,
    1.0710183895375325241e-50,
    -2.97326667639271410471e-51,
    -2.62549381380331082931e-52,
    7.3794635575180183245e-53,
    6.44281913614258243117e-54,
    -1.8324611992094396787e-54,
    -1.58257415149925283552e-55,
    4.55253558045741042317e-56,
    3.89090093895914349218e-57,
    -1.131542025142378846e-57,
    -9.57439265268174191257e-59,
    2.81369922984216864689e-59,
    2.35790425459578207476e-60,
    -6.99947980008426417277e-61,
    -5.81133103721257972681e-62,
    1.74191509334634560741e-62,
    1.43331541799268487191e-63,
    -4.33664474404944952592e-64,
    -3.53759582752554591305e-65,
    1.08003876644682829566e-65,
    8.73694811842762586939e-67,
    -2.69077194932209436453e-67,
    -2.15914767155319480376e-68,
    6.70594984010431247385e-69,
    5.3390370903270346715e-70,
    -1.67179779623437813933e-70,
    -1.32095921062365319204e-71,
    4.16909460365277804311e-72,
    3.27002042294121945992e-73,
    -1.03998992874654137524e-73,
    -8.09907798947651714009e-75,
    2.59502183718779187431e-75,
    2.00694244747268093241e-76,
    -6.47698343789186455602e-77,
    -4.9755291458631618614e-78,
    1.61703745561942237663e-78,
    1.23407065857989201858e-79,
    -4.03811576717266187617e-80,
    -3.06216849830506353157e-81
  };

static const double ccli3logA1[100] =
  {
    0.75,
    -0.0034722222222222222222,
    0.000011574074074074074074,
    -9.8418997228521038045e-8,
    1.1482216343327454439e-9,
    -1.5815724990809165893e-11,
    2.4195009792525151945e-13,
    -3.9828977769894877479e-15,
    6.9233666183059290581e-17,
    -1.2552722304499772755e-18,
    2.3537540027684652306e-20,
    -4.5363989034586870184e-22,
    8.9451696703926431671e-24,
    -1.7982840046954962717e-25,
    3.6754997647937384443e-27,
    -7.6208079715647952295e-29,
    1.6000419643694859752e-30,
    -3.3967611475603755879e-32,
    7.2822722867577646953e-34,
    -1.5750226479580034872e-35,
    3.4335400924805893359e-37,
    -7.5388499980898700099e-39,
    1.6660681647653604103e-40,
    -3.703898902881387057e-42,
    8.2792117964682746899e-44,
    -1.859914814630981114e-45,
    4.1976272253270599454e-47,
    -9.5142076988004535218e-49,
    2.1650428257869544381e-50,
    -4.9450009037451569801e-52,
    1.1333527931614100784e-53,
    -2.6059491948419958604e-55,
    6.0100924777449462977e-57,
    -1.390051955642734155e-58,
    3.2236130397341702504e-60,
    -7.4946322134495044325e-62,
    1.7465957441450237591e-63,
    -4.0795584676816986499e-65,
    9.5490700732357419848e-67,
    -2.2396934647201936771e-68,
    5.2632217580889312466e-70,
    -1.2391091371897818099e-71,
    2.9222972579786725077e-73,
    -6.9033666612158515393e-75,
    1.6333768346359737562e-76,
    -3.8705271156399126118e-78,
    9.1850912927811484321e-80,
    -2.1827282938640762717e-81,
    5.1939000781981987314e-83,
    -1.2374897952334061554e-84,
    2.9520357615164942565e-86,
    -7.0503826336462555032e-88,
    1.6857578192254847183e-89,
    -4.0350651504727279424e-91,
    9.6685483730051647373e-93,
    -2.3190536165931872738e-94,
    5.5677973769707277468e-96,
    -1.338025289438272029e-97,
    3.2183954667682428548e-99,
    -7.7481017377329127177e-101,
    1.866892649017235271e-102,
    -4.5019371606460057391e-104,
    1.0864858534304730003e-105,
    -2.624111876451127328e-107,
    6.3425487441171313136e-109,
    -1.5341171675623227434e-110,
    3.7132740440136786421e-112,
    -8.9939436127879788493e-114,
    2.1798636881682535309e-115,
    -5.2867191526251499446e-117,
    1.2829594582597580746e-118,
    -3.1153139144964788269e-120,
    7.569123684948312892e-122,
    -1.8400825611637293552e-123,
    4.4757955050806411973e-125,
    -1.0892760318992447104e-126,
    2.6523709680822372044e-128,
    -6.4617961191161523274e-130,
    1.5750313192173018394e-131,
    -3.8409318468479996259e-133,
    9.3710916457214082513e-135,
    -2.2874144202905265821e-136,
    5.5859327729017073145e-138,
    -1.3647025633763582571e-139,
    3.3355480695921490721e-141,
    -8.1560315314926725526e-143,
    1.9951191617715253051e-144,
    -4.8823952740351548234e-146,
    1.1952733269796775364e-147,
    -2.9273044269808816234e-149,
    7.1718504223488768502e-151,
    -1.7577360251163409855e-152,
    4.3095479986921742378e-154,
    -1.0569685887658041655e-155,
    2.5932327038277849611e-157,
    -6.3645369286820284316e-159,
    1.5625538859112023404e-160,
    -3.8374534606610090976e-162,
    9.4273213179332127824e-164,
    -2.3166893299199809125e-165
  };

static const double WTFccli3logA10 =  1.2020569031595942854;
static const double WTFccli3logA11 =  -1.6449340668482264365;
static const double WTFccli3logA13 =  0.083333333333333333333;
static const double WTFccli3logA1log =  -0.5;

static const double WTFccli3inv1 = -1.6449340668482264365;
static const double WTFccli3inv3 = -0.16666666666666666667;

static const double WTFccli3inone = 1.2020569031595942854;


double SSm0_li3logA0rec(double next, int nc, double x)
{
  if (nc < li_constants.LINLOGA0MAX)
    return next*ccli3logA0[nc] + SSm0_li3logA0rec(next*x, nc+1, x);
  else
    return next*ccli3logA0[nc];
}

double SSm0_li3logA0(double x)
{
  double mlog = -log(1.0-x);
  return mlog + SSm0_li3logA0rec(mlog*mlog, 1, mlog);
}

double SSm0_li3logA1rec(double next, int nc, double xsq)
{
  if (nc < li_constants.LINLOGA1MAX)
    return ccli3logA1[nc]*next + SSm0_li3logA1rec(next*xsq, nc+1, xsq);
  else
    return ccli3logA1[nc]*next;
}


double SSm0_li3logA1(double a)
{
  double x = -log(a);
  double xsq = x*x;
  return WTFccli3logA10 +
    x*(WTFccli3logA11 + xsq*WTFccli3logA13) +
    WTFccli3logA1log*xsq*log(fabs(x)) +
    SSm0_li3logA1rec(xsq, 0, xsq);
}

/* REAL version */
double SSm0_li3inv(double x)
{
  double ix = 1.0/x;
  if ( x > 0)
    {
      double logx2 = log(x)*log(x);
      double lsq = logx2 - 3*li_constants.Pi2;
      return SSm0_li3logA0(ix) + log(x)*(WTFccli3inv1 + lsq*WTFccli3inv3);
    }
  else
    {
      double logmx = log(-x);
      return SSm0_li3logA0(ix) + logmx*(WTFccli3inv1 + logmx*logmx*WTFccli3inv3);
    }
}

/* Explicitly real PolyLog[3,x] */
double li3re(double x)
{
  if (x < 0.5)
    {
      if (fabs(x) < 1.0)
        return SSm0_li3logA0(x);
      else
        return SSm0_li3inv(x);
    }
  if (fabs(x-1.0) < 1.0)
    {
      if (x == 1.0)
        return WTFccli3inone;
      else
        return SSm0_li3logA1(x);
    }
  else
    return SSm0_li3inv(x);
}

double li3logA0re2_rec(double next, double cos_prev, double cos_prev_prev, int nc, double r, double cphi)
{
  double cnphi = 2*cphi*cos_prev - cos_prev_prev; /* cos(n*phi)     */

  if (nc < li_constants.LINLOGA0MAX)
    return next*cnphi*ccli3logA0[nc] + li3logA0re2_rec(next*r, cnphi, cos_prev, nc+1, r, cphi);
  else
    return next*cnphi*ccli3logA0[nc];
}


double li3logA0re2(double rex, double imx)
{
  double reAl = - log((1 - rex)*(1 - rex) + imx*imx)/2.;
  double imAl = - atan2(-imx,1 - rex);
  double absAl = hypot(reAl, imAl); /* |al| */
  double argAl = atan2(imAl, reAl); /* arg(al) */
  double cAA = cos(argAl);

  double res = reAl;
  /* for (size_t j = 1; j <= li_constants.LINLOGA0MAX; j++) */
  /*   { */
  /*     res += pow(absAl,j+1)*cos((j+1)*argAl)*ccli3logA0[j]; */
  /*   } */

  res += li3logA0re2_rec(absAl*absAl, cAA, 1, 1, absAl, cAA);
  return res;
}

double li3logA1re2_rec(double next, double cos_prev, double cos_prev_prev, int nc, double rsq, double cphi)
{
  double cnphi = 2*cphi*cos_prev - cos_prev_prev; /* cos(n*phi)     */
  double cnnphi = 2*cphi*cnphi - cos_prev;        /* cos((n+1)*phi) */

  if (nc < li_constants.LINLOGA1MAX)
    return next*cnphi*ccli3logA1[nc] + li3logA1re2_rec(next*rsq, cnnphi, cnphi, nc+1, rsq, cphi);
  else
    return next*cnphi*ccli3logA1[nc];
}

double li3logA1re2(double rex, double imx)
{
  double reAl = - log(rex*rex + imx*imx)/2.;
  double imAl = - atan2(imx, rex);
  double absAl = hypot(reAl, imAl); /* |al| */
  double argAl = atan2(imAl, reAl); /* arg(al) */
  double absx  = hypot(rex, imx); /* |x| */
  double argx  = atan2(imx, rex); /* arg(x) */
  double logax = log(absx) ;
  double al2     = logax*logax - argx*argx;
  double absAl2 = absAl*absAl;
  double cAA = cos(argAl);

  double res = WTFccli3logA10 - logax*WTFccli3logA11 +
    (3*logax*argx*argx - logax*logax*logax)*WTFccli3logA13 +
    WTFccli3logA1log*(al2*log(absAl) - 2*argAl*argx*logax);

  /* for (size_t n = 0; n <= li_constants.LINLOGA1MAX; n++) */
  /*   { */
  /*     res += pow(absAl,2*n+2)*cos((2*n+2)*argAl)*ccli3logA1[n]; */
  /*   } */
  res += li3logA1re2_rec(absAl2, cAA, 1, 0, absAl2, cAA);

  return res;
}

double li3invRe2(double rex, double imx)
{
  double absx2 = rex*rex + imx*imx;
  double reix = rex/absx2;
  double imix = -imx/absx2;
  double logax = log(absx2);
  double argmx = atan2(-imx, -rex);

  return li3logA0re2(reix, imix) + 0.25*argmx*argmx*logax - logax*logax*logax/48. + logax*WTFccli3inv1/2.;
}


/* real part of Li3(rex + I*imx) */
double li3re2(double rex, double imx)
{
  double ax  = hypot(rex, imx);      /* |x|   */
  double axb = hypot(1 - rex, -imx); /* |1-x| */
  if (rex < 0.5)
    {
      if (ax < 1.0)
        return li3logA0re2(rex,imx);
      else
        return li3invRe2(rex, imx);
    }
  if (axb < 1.0)
    {
      if (rex == 1.0 && imx == 0.0)
        return WTFccli3inone;
      else
        return li3logA1re2(rex,imx);
    }
  return li3invRe2(rex,imx);
}
