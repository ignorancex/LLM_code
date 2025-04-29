//========================================================================== */
//                                                                            */
//   TruncNorm.c                                                              */
//   (c) 2011 Alan Lenarcic                                                   */
//                                                                            */
//   A slice sampler based approach to get long tailed truncated T draws      */
//  in one or a few slice samples.  This is not the most elegant or quickest  */
//  solution to the challenging truncated T problem, but it works well enough */
//  for infrequent need of Robit data.                                        */
//                                                                            */
// ========================================================================== */
/******************************************************************************/
//// LICENSE INFO: C CODE
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  A copy of the GNU General Public License is available at
//  https://www.R-project.org/Licenses/
//
/******************************************************************************/
#ifndef TRUNCNORMH
  #include "TruncNorm.h"
  #define TRUNCNORMH 0
#endif

#define Something 0


//// What is going on?

double rTruncNorm(double mu, double sig, double L, double U) {
 double NR = rTruncNorm2( (L - mu) / sig, (U-mu) / sig);
 return(  NR * sig + mu);
}

extern
 "C"
{
  SEXP rTNorm(SEXP sRet, SEXP smu, SEXP ssig, SEXP sL, SEXP sU) {
    double mu, sig, L, U;
    int ii;
    if (Rf_isNull(smu)) { mu = 0;} else { mu = REAL(smu)[0]; }
    if (Rf_isNull(ssig)) { sig = 1.0;} else { sig = REAL(ssig)[0] ;}
    L = REAL(sL)[0];  U = REAL(sU)[0];
    for (ii = 0; ii < Rf_length(sRet); ii++) {
      if(Rf_length(smu) > ii) {  mu = REAL(smu)[ii]; }  
      if (Rf_length(ssig) > ii) { sig = REAL(ssig)[ii]; }
      if (Rf_length(sL) > ii) { L = REAL(sL)[ii]; }
      if (Rf_length(sU) > ii) { U = REAL(sU)[ii]; }   
      REAL(sRet)[ii] = rTruncNorm(mu, sig, L, U);
    }
    return(sRet);
  }
  SEXP rTT(SEXP sRet, SEXP snu, SEXP smu, SEXP ssig, SEXP sL, SEXP sU) {
    double mu, sig, L, U, nu;
    GetRNGstate();
    int ii;
    nu = -1; sig = 1; mu = 0.0;
    if (Rf_isNull(smu)) { mu = 0;} else { mu = REAL(smu)[0]; }
    if (Rf_isNull(ssig)) { sig = 1.0;} else { sig = REAL(ssig)[0]; }
    if (Rf_isNull(snu)) { nu = 1.0; } else { nu = REAL(snu)[0]; }
    if (Rf_isNull(sL)) { L = R_NegInf; } else {
      L = REAL(sL)[0]; 
    }
    if (Rf_isNull(sU)) {  U = R_PosInf; } else {
      U = REAL(sU)[0];
    }
    for (ii = 0; ii < Rf_length(sRet); ii++) {
      //Rprintf("rTT: on ii = %d\n", ii); R_FlushConsole(); 
      if(!Rf_isNull(smu) && Rf_length(smu) > ii) {  mu = REAL(smu)[ii]; }  
      if (!Rf_isNull(ssig) && Rf_length(ssig) > ii) { sig = REAL(ssig)[ii]; }
      if (!Rf_isNull(sL) && Rf_length(sL) > ii) { L = REAL(sL)[ii]; }
      if (!Rf_isNull(sU) && Rf_length(sU) > ii) { U = REAL(sU)[ii]; } 
      if (!Rf_isNull(snu) && Rf_length(snu) > ii) { nu = REAL(snu)[ii]; }  
      if (nu <= 0  || !R_finite(nu) || R_isnancpp(nu)) {
        REAL(sRet)[ii] = rTruncNorm(mu, sig, L, U);
      } else {
        REAL(sRet)[ii] = mu+rTruncT2(nu, (L-mu) / sqrt(sig), (U-mu)/sqrt(sig));
      }
    }
    PutRNGstate();
    return(sRet);
  }
}


double rTruncNorm2(double L, double U) {
  int ii; double pQ;
  //Rprintf("rTruncNorm2: start with L = %f, U = %f\n"); R_FlushConsole();
  if ((!R_finite(U) || R_isnancpp(U)) && L < 0) {
    //Rprintf("rTruncNorm2: Starting notfiniteU, L < 0"); R_FlushConsole();
    for (ii = 0; ii < MaxTruncTrieds; ii++) {
      pQ = Rf_rnorm(0.0,1.0);
      if (pQ > L) { return(pQ); }
    }
    return(pQ);
  } else if ((!R_finite(L) || R_isnancpp(L)) && U > 0) {
    //Rprintf("rTruncNorm2: Starting notfinit L, U > 0"); R_FlushCosnole();
    for (ii = 0; ii < MaxTruncTrieds; ii++) {
       pQ = Rf_rnorm(0.0,1.0);
       if (pQ < U) { return(pQ);}
    } return(pQ);
  } else if ((!R_finite(U)|| R_isnancpp(U)) && L > 0) {
    //Rprintf("rTruncNorm2: L > 0 but U is not finite \n"); R_FlushConsole();
    return(rTailTruncHalfNorm(L, U));
  } else if (L < 0 && U > 0) {
    //Rprintf("rTruncNorm2")
    double PL = Rf_pnorm5(L,0.0,1.0,0,0);  
    double PU = Rf_pnorm5(-U,0.0,1.0,0,0);
    int S  = 0;
    if (Rf_runif(0.0,1.0) >  PU / ( PL + PU)) { S = 1; }
    if (S == 1) {
      return(rInsideHalfNorm(U));
    } else {
      return(-rInsideHalfNorm(-L));
    }
  } else if (L > 0) {
    return(rTailTruncHalfNorm(L, U));
  } else if (L == 0) {
    return(rInsideHalfNorm(U));
  } else if (U == 0) {
    return(-rInsideHalfNorm(-L));
  } else {
    return(-rTailTruncHalfNorm(-U,-L));
  }
  return(0.0);
}
double rTruncT2(double nu, double L, double U) {
  int ii; double pQ;
  // Rprintf("rTruncT2: STarint with nu = %f, L = %f, U = %f \n", nu, L, U);
  //R_FlushConsole();
  if ((!R_finite(U) || R_isnancpp(U)) && L < 0) {
    //Rprintf("rTruncT2:  U is not finite, but L = %f < 0\n", L);
    //R_FlushConsole();
    for (ii = 0; ii < MaxTruncTrieds; ii++) {
      //Rprintf("rTruncT2: Start = ii = %d, Rf_rnorm(0.0,1.0) = %f\n",
      //  ii, Rf_rnorm(0.0,1.0));  R_FlushConsole();
      //Rprintf("rTruncT2: on ii = %d, ");  R_FlushConsole();
      //Rprintf("  rchisq(nu) = %f\n",
      // rchisq(nu) ); R_FlushConsole();
      pQ = Rf_rnorm(0.0,1.0) / sqrt( rchisq(nu) / nu);
      if (!R_finite(pQ)) {
        Rf_error("Fuck, pQ was sample not finite!"); R_FlushConsole();
        
      }
      //Rprintf("rTruncT2:  Tried with pQ = %f\n", pQ);  R_FlushConsole();
      if (pQ > L) { return(pQ); }
    }
    return(pQ);
  } else if (( !R_finite(L)||R_isnancpp(L)) && U > 0) {
    //Rprintf("rTruncT2:  L is not finite, but U = %f > 0\n", U);
    R_FlushConsole();
    for (ii = 0; ii < MaxTruncTrieds; ii++) {
       pQ = Rf_rnorm(0.0,1.0) / sqrt(2.0 *Rf_rgamma(nu / 2.0, 1.0) / nu);
       if (pQ < U) { return(pQ);}
    } return(pQ);
  } else if ((!R_finite(U) || R_isnancpp(U)) && L > 0) {
    //Rprintf("rTruncT2:  U is not finite, L > 0 = %f \n", L);
    return(rTailTruncHalfT(nu,L, U));
  } else if (L < 0 && U > 0) {
    double PL = Rf_pnorm5(L,0.0,1.0,0,0);  
    double PU = Rf_pnorm5(-U,0.0,1.0,0,0);
    int S  = 0;
    if (Rf_runif(0.0,1.0) >  PU / ( PL + PU)) { S = 1; }
    if (S == 1) {
      return(rInsideHalfT(nu, U));
    } else {
      return(-rInsideHalfT(nu,-L));
    }
  } else if (L > 0) {
    return(rTailTruncHalfT(nu,L, U));
  } else if (L == 0) {
    return(rInsideHalfT(nu,U));
  } else if (U == 0) {
    return(-rInsideHalfT(nu,-L));
  } else {
    return(-rTailTruncHalfT(nu, -U,-L));
  }
  return(0.0);
}


inline double rInsideHalfT(double nu, double U) {
  int ii;  double pQ;  //double ProbKeep;
  if (U < CutCut) {
    for (ii = 0; ii < MaxTruncTrieds; ii++) {
      pQ = runif(0.0,U);
      if ( log(Rf_runif(0,1.0)) <
        log(nu + pQ*pQ*.5) *(nu+1) * .5  - log(nu) * (nu+1) * .5) {
        return(pQ);
      } 
    }
    return(pQ);
  } else {
    for (ii = 0; ii < MaxTruncTrieds; ii++) {
      pQ = fabs(Rf_rnorm(0.0,1.0) / sqrt( rchisq(nu) / nu));
      if (pQ < U) { return(pQ); }
    } return(pQ);
  }         
}


inline double rInsideHalfNorm(double U) {
  int ii;  double pQ;  //double ProbKeep;
  if (U < CutCut) {
    for (ii = 0; ii < MaxTruncTrieds; ii++) {
      pQ = runif(0.0, U);
      if (runif(0.0,1.0) < exp(-pQ * pQ *.5)) {
        return(pQ);
      } 
    }
    return(pQ);
  } else {
    for (ii = 0; ii < MaxTruncTrieds; ii++) {
      pQ = fabs(Rf_rnorm(0.0,1.0));
      if (pQ < U) { return(pQ); }
    } return(pQ);
  }         
}


//  P = lambda / x^nu;  P > L suggests,  x > pow( lambda/L, 1/nu)
//  P > U suggests, x < pow( lambda/U, 1/nu)
inline double rTruncPareto(double nu, double lambda, double L, double U) {
  double LBound = 0.0;
  if (L < lambda) {
    LBound = 1.0;
  } else {
    LBound = powl(lambda/L, nu);
  }
  if (U <= lambda) {
    return(lambda);
  }
  double UBound = 0.0;
  if (R_finite(U)) {
    UBound = powl(lambda / U, nu);
  }
  double rU = Rf_runif(UBound, LBound);
  if (rU == 0.0) { return(R_PosInf); }
  return( lambda / powl(rU, 1.0/nu));
}

inline double rTruncExpo(double lambda, double L, double U) {
  double mm = exp(-(U-L) * lambda);
  if (!R_finite(U)) { mm = 0.0; }
  double rU = runif(mm,1.0);
  return(L + - log(rU) / lambda);
}
inline double rTailTruncHalfT(double nu, double L, double U) {
  int ii;   double pQ;  double ProbKeep;
  //Rprintf("rTailTruncHalfT, Complete Start\n", L); R_FlushConsole(); 
    double Move =  nu/2;  if (nu <= 2) { Move = 1.5; }
  if (!R_finite(U)) {
    //Rprintf("rTailTruncHalfT, U is not finite, but L = %f \n", L); R_FlushConsole(); 

    if (L < .4) {
      for (ii = 0; ii < MaxTruncTrieds; ii++) {
       pQ = fabs(Rf_rnorm(0.0,1.0)/ sqrt( Rf_rchisq(nu)/ nu));
       if (pQ > L) { return(pQ); }
      }  
    } else if (L >= Move) {
      for (ii = 0; ii < MaxTruncTrieds; ii++) {
        pQ = rTruncPareto(nu, L, L, U);
        ProbKeep =  log(nu + pQ*pQ)* (nu+1.0)*.5  - log(pQ)*(nu+1);
        if (log(Rf_runif(0,1.0)) < -ProbKeep) {
          return(pQ);
        }
      } return(pQ);      
    } else {
      pQ = L; double NQ = 0;
      //Rprintf("Start rTailTruncHalfT\n"); R_FlushConsole(); 
      double Multer = 1.0;
      for (ii = 0; ii < MaxSliceLoop; ii++) {
        //Rprintf("S1 = 1: On ii = %d, pQ = %f, NQ = %f, Find NQ = %f\n", ii, pQ, NQ, (FindTLevel(NQ,nu))); R_FlushConsole();     
        Multer =  1.0;
        if (ii < MaxSliceLoop / 2) { Multer=(1.0 + 2*(MaxSliceLoop /2-ii)/MaxSliceLoop) ; }
        NQ =  Multer * log(runif(0.0,1.0)) - (nu+1.0) *.5 * log(nu+pQ*pQ);
        //Rprintf("S2 = 2: On ii = %d, pQ = %f, NQ = %f, Find NQ = %f\n", ii, pQ, NQ, (FindTLevel(NQ,nu))); R_FlushConsole(); 
        pQ = runif(L, (FindTLevel(NQ,nu)) );
        //Rprintf("S3 = 3: On ii = %d, pQ = %f, NQ = %f, Find NQ = %f\n", ii, pQ, NQ, (FindTLevel(NQ,nu))); R_FlushConsole(); 
      }
      return(pQ);
    }
  }
  double lM2CqPi = log(nu+L*L) * (nu+1) *.5;
  if (U < CutCut) {
    for (ii = 0; ii < MaxTruncTrieds; ii++) {
      pQ = runif(L,U);
      if (log(runif(0,1.0)) < log(nu +pQ * pQ) * (nu+1) * .5 -  
        lM2CqPi) { return(pQ); }
    }
  }
  if (U > CutCut && L < Move) {
    pQ = L; double NQ = 0; double CP = 0.0;
    double Multer = 1.0;
    for (ii = 0; ii < MaxSliceLoop; ii++) {
      Multer =  1.0;
      if (ii < MaxSliceLoop / 2) { Multer = (1.0 + 2*(MaxSliceLoop /2-ii)/MaxSliceLoop) ; }
      NQ = Multer *
        log(runif(0.0,1.0)) - (nu+1) *.5 * log(nu+pQ*pQ);
      CP = FindTLevel(NQ, nu);  if (CP > U) { CP = U; }
      pQ = runif(L,CP);
    }
    return(pQ);
  }
  for (ii = 0; ii < MaxTruncTrieds; ii++) {
    pQ = rTruncPareto(nu, L, L, U);
    ProbKeep =  log(nu + pQ*pQ )* (nu+1)*.5 - log(pQ)*(nu+1);
    if (log(Rf_runif(0,1.0)) < ProbKeep) {
      return(pQ);
    }
  } return(pQ);
}

inline double rTailTruncHalfNorm(double L, double U) {
  int ii;   double pQ;  double ProbKeep;
  double lM2CqPi = L*L * .5;
  if (U < CutCut) {
    for (ii = 0; ii < MaxTruncTrieds; ii++) {
      pQ = runif(L,U);
      if (runif(0,1.0) < exp(-pQ * pQ * .5 + L * L * .5)) { return(pQ); }
    }
  }
  if (U > CutCut && L < .25) {
    for (ii = 0; ii < MaxTruncTrieds; ii++) {
       pQ = fabs(Rf_rnorm(0.0,1.0));
       if (pQ > L && pQ < U) { return(pQ); }
    }
  }
  for (ii = 0; ii < MaxTruncTrieds; ii++) {
    pQ = rTruncExpo(L, L, U);
    ProbKeep = exp(-pQ * pQ * .5 - lM2CqPi + L * pQ);
    if (runif(0,1.0) < ProbKeep) {
      return(pQ);
    }
  } return(pQ);
}

inline double FindTLevel(double lfdens, double nu) {
  return(sqrt(exp(-lfdens *2 / (nu+1))-nu));
}
