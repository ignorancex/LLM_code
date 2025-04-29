#include "uLLD.h"
#define LOGEPS 300*log(10)
#define LGAMMAEPS lgamma(1e-300)
#define Eps1 0.9999

// function of "dtype" that is strings to specify model name
// return the LLD function correspoinding to the specified model.
distfunc match_dist(char **dtype) {
  if (!strcmp("pois",*dtype))
    return plldist;
  else if (!strcmp("geom",*dtype))
    return glldist;
  else if (!strcmp("nbin",*dtype))
    return nblldist;
  else if (!strcmp("norm",*dtype))
    return nlldist;
  else if (!strcmp("multinom",*dtype))
    return mltlldist;
  else if (!strcmp("binary",*dtype))
    return blldist;
  else {
    error("distance type '%s' not exist\n", *dtype);
  }
}

/* follows are LLD functions.
   It is a function of a data and 1st-Kth moments and
   return the log-likelihood dissimilarity. */

/*
  f ~ Po(lambda)
  LLD = mu - f * log(mu)
  if |mu|~0 then -f*log(eps) where eps = 1e-300 in here
*/
double plldist(double f, double *pmu) {
  double mu = *pmu;
  if (mu <= DBL_MIN) {
    if (mu < -EPS) {
      return HUGE_VAL;
    } else {
      if (f >= 1)
        return LOGEPS;
      else
        return 0;
    }
  }
  return (mu-f * log(mu));
}

/*mu : mean
  p = 1/(mu+1): MLE
  -log(p) - f*log(1-p) = -log(1/(mu+1)) - f * log(mu/(mu+1))
  = log(mu+1) - f*(log(mu) - log(mu+1))
  = log1p(mu) - f*(log(mu) - log1p(mu))
*/
/*
  mu -> infty => p = 0 => -log(0)
  |mu| -> 0 => p = 1 => - f*log(0)
  if |mu|~0 then -f*log(eps) where eps = 1e-300 in here
*/
double glldist(double f, double *pmu) {
  double mu = *pmu;
  if (mu <= DBL_MIN) {
    if (mu < -EPS) {
      return HUGE_VAL;
    } else {
      if (f >= 1)
        return LOGEPS;
      else
        return 0;
    }
  }
  return (log1p(mu) - f * (log(mu) - log1p(mu)));
}

/* p : multinomial distribution
   f (x | p) = lgamma(sum_j(xj) + 1) - sum_j(lgamma(xj+1)) + sum_j (xj log(pj))
*/
double mltlldist(double f, double *pp) {
  double p = *pp;
  if (p <= DBL_MIN) {
    if (p < -EPS)
      return HUGE_VAL;
    p = 1e-300;
  }
  return (- f * log(p));
}

/* mu1 = r*(1-p)/p, mu2 = r*(1-p)*(r*(1-p)+1)/p^2
   r = p*mu1/(1-p) <=> mu2 = mu1*(p*mu1+1)/p = mu1^2 + mu1/p
   <=> mu2 - mu1^2 = mu1/p
   <=> p = mu1 / (mu2-mu1^2)
   => r = mu1^2 / (mu2 - mu1^2 - mu1)
   LLD(x,(p,r)) = lgamma(r) - lgamma(x+r) -r*log(p) - x*log(1-p)
   mu1 -> infty => p = 0 (because mu2 > mu1 > 0)
   |mu1| -> 0 => p,r -> 0
   r*log(p) -> 0, f*log(1-p) -> 0
   lgamma(r) - lgamma(f+r) if f=0 then 0 else lgamma(eps) - lgamma(f)
*/
double nblldist(double f, double *pmu) {
  double mu1 = pmu[0], mu2 = pmu[1];
  double p,r,tmp;

  if (mu1 <= DBL_MIN) {
    if (mu1 < -EPS)
      return HUGE_VAL;
    if (f==0)
      return 0;
    else
      return LGAMMAEPS-lgamma(f);
  }
  tmp = mu2 - mu1*mu1;
  if (tmp <= DBL_MIN) {
    //printf("       %2.1f,%2.1f\n",mu2,mu1*mu1);
    if (tmp < -EPS)
      return HUGE_VAL;
    tmp = EPS;
  }
  p = mu1 / tmp;
  if(p >= 1)
    p = 1-EPS;
  else if (p <= 0)
    p = EPS;
  r = (p*mu1)/(1-p);
  return (lgamma(r) - lgamma(f+r) -r*log(p) - f*log(1-p));
}

double nlldist(double x, double *pmu) {
  double mu1 = pmu[0], mu2 = pmu[1];
  double sig;
  if (mu2 <= DBL_MIN)
    return HUGE_VAL;
  sig = mu2 - mu1*mu1;
  if(sig <= EPS) {
    if (sig >= -EPS)
      sig = EPS;
    else
      return HUGE_VAL;
  }
  return (0.5 * log(sig) + (x-mu1)*(x-mu1)/(2.0 * sig));
}

double blldist(double f, double *pp) {
  double p = *pp;
  if (p <= DBL_MIN) {
    if (p < -EPS)
      return HUGE_VAL;
    p = 1e-300;
  }
  if (p > Eps1)
    p = 1-EPS;
  return (-f*log(p) - (1-f)*log(1-p));
}
