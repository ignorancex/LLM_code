/* ========================================================================== */
/*                                                                            */
/*   BayesSpikeSplicSampling.cpp                                              */
/*   (c) 2011 Alan Lenarcic                                                   */
/*                                                                            */
/*   Modified Splice Sampling For BayesSpikeCL  RCpp class                    */
/*   This is a demonstration of the Switching Algorithm as implemented        */
/*    which allows for a sparse Tau Vector and thus Group Sparse Sampling     */
/*                                                                            */
//  The Bulk of this file is designed to maximize, slice sample, and
//  eventually determine the integrated area of a f_2(tau_k) group parameter.
//  The Df3, and Df4 are the upper and lower limit functions bounding f_2(tau_k),
//  We need to determine maximum of f_2(tau) to find best df_3, df_4 densities.
//  Then we can take a sample of f_2 through a certain amount of slice sampling,
//  Finally we make a bounds determination of whether this draw of f_2 suggests
//  That this group parameter is worthy.
//
//  Note that functions lOfX and the derivatives are coded in
//  "BayesSpikeHelpers.cc", and that many of the functions here serve as Rcpp
//  wrappers to generate draws of f_2 and investigate their quality.
/* ========================================================================== */
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
#ifndef BAYESSPIKECPPH
  #include "BayesSpikeCpp.h"
  #define BAYESSPIKECPPH 0
#endif
 
// Degrees of Freedom of lower bound function q3(x)
double BayesSpikeCL::get_Df3() {
 return(REAL(MT->staupriordf)[0] + (double) MT->NumJ + 1);
}
//Degrees of Freedom of upper bound function q4(x)
double BayesSpikeCL::get_Df4() {
 return(REAL(MT->staupriordf)[0] + (double) MT->NumJ-1);
}
  
////////////////////////////////////////////////////////////////////////////////
//  double BayesSpikeCL::ValAtMaximum()
//
//  Finds the maximum for the current OnTauIndex
//     
double BayesSpikeCL::ValAtMaximum() {
  double MyMax = MaximizeOn();
  
  SetupMTForOnii(OnTauIndex);
  double Returner = (lFOfX(MyMax, MT));  
  int One = 1;
  F77_CALL(dscal)(&MT->NumJ, REAL(sOnSigma), MT->D, &One);
  ScaleOfEigenVectors *= REAL(sOnSigma)[0];
  return(Returner);
}
double BayesSpikeCL::MaximizeOn() {
  if (MT == NULL) {
    Rf_error("MaxmimizeOn: MT was nver setup!");
  }
  if (MT->R == NULL) {
    Rf_error("MaximizeOn: MT->R is not set up");
  }

  SetupMTForOnii(OnTauIndex);
  double Returner = MaximizeMe(1.0, MT, MaximizeMeCauchyTotalIters, CauchyEpsilon, 
    Verbose);
  int One = 1;
  F77_CALL(dscal)(&MT->NumJ, REAL(sOnSigma), MT->D, &One);
  ScaleOfEigenVectors *= REAL(sOnSigma)[0];
  return(Returner);
}
////////////////////////////////////////////////////////////////////
// double GiveLOfX(double SampleStart);
//
// LOfX is the log of posterior probability for the current Ontau on Onii setup
//  using SetupMT
//
double BayesSpikeCL::GiveLOfX(double SampleStart) {
  //  Note that we must descale Eigenvalues of XtX for the calculations:
  SetupMTForOnii(OnTauIndex);
  double TempTemperature = MT->Temperature;  MT->Temperature = 1.0;
  double returner = (lFOfX(SampleStart, MT));
  int One = 1;
  F77_CALL(dscal)(&MT->NumJ, REAL(sOnSigma), MT->D, &One);
  ScaleOfEigenVectors *= REAL(sOnSigma)[0];
  MT->Temperature = TempTemperature;
  return(returner);
}
//  Version of GiveLOfX for a SEXP vector SampleStart
SEXP BayesSpikeCL::sLOfX(SEXP SampleStart) {
  SEXP sOut = R_NilValue;
  Rf_protect(sOut = Rf_allocVector(REALSXP, Rf_length(SampleStart)));
  int ii;
  //  Note that we must descale Eigenvalues of XtX for the calculations:
  SetupMTForOnii(OnTauIndex);
  double TempTemperature = MT->Temperature;  MT->Temperature = 1.0;
  for (ii = 0; ii < Rf_length(SampleStart); ii++) {
    REAL(sOut)[ii] = (*lFOfX)(REAL(SampleStart)[ii],MT);
  }
  MT->Temperature = TempTemperature;
  //  Re-unscale things
  int One = 1;
  F77_CALL(dscal)(&MT->NumJ, REAL(sOnSigma), MT->D, &One);
  ScaleOfEigenVectors *= REAL(sOnSigma)[0];
  Rf_unprotect(1);
  return(sOut);
}


// Test for random Uniforms
SEXP BayesSpikeCL::GRunif(int Counts) {
  SEXP sOut = R_NilValue;
  if (Counts <= 0) { return(R_NilValue); }
  Rf_protect(sOut = Rf_allocVector(REALSXP, Counts));
  for (int ii = 0; ii < Counts; ii++) {
    REAL(sOut)[ii] = Rf_runif(0.0,1.0);
  } 
  Rf_unprotect(1);
  return(sOut);
}
double BayesSpikeCL::get_MaxX() {
  double MaxX = -1;
  double TempTemperature = MT->Temperature;  MT->Temperature = 1.0;
  if (MT->sPriorOftau != NULL && !Rf_isNull(MT->sPriorOftau) && Rf_length(MT->sPriorOftau) > 1) {
    MaxX = Rf_length(MT->sPriorOftau) * REAL(MT->sPriorXScaling)[0] +
         REAL(MT->sPriorXScaling)[2];
    if (REAL(MT->sPriorXScaling)[1] == 1) {
    } else if (REAL(MT->sPriorXScaling)[1] == 2) {
      MaxX = MaxX * MaxX;
    } else if (REAL(MT->sPriorXScaling)[1] == -1) {
      MaxX = exp(MaxX);
    }
  }
  MT->Temperature = TempTemperature;
  return(MaxX);
}

// Run a binomial search for the maximizer of LOfX
double BayesSpikeCL::BinFindLOfX(double StartX, double GoalVal) {
  double MaxX = get_MaxX();
 double TempTemperature = MT->Temperature;  MT->Temperature = 1.0; 
 SetupMTForOnii(OnTauIndex);

 double returner = (BinFindX( lFOfX, StartX, GoalVal, 
    0.0, MaxX, 100, -SpliceMove, MT, CauchyEpsilon)); 
 int One = 1;  
 F77_CALL(dscal)(&MT->NumJ, REAL(sOnSigma), MT->D, &One); 
 ScaleOfEigenVectors *= REAL(sOnSigma)[0];
 MT->Temperature = TempTemperature;
 return(returner); 
}

SEXP BayesSpikeCL::BAllLOfX(SEXP SampleStart) {
SEXP sOut = R_NilValue;
  Rf_protect(sOut = Rf_allocVector(REALSXP, Rf_length(SampleStart)));
  int ii;
  double TempTemperature = MT->Temperature; MT->Temperature = 1.0;

  //  Note that we must descale Eigenvalues of XtX for the calculations:
  SetupMTForOnii(OnTauIndex); int One = 1;
  for (ii = 0; ii < Rf_length(SampleStart); ii++) {
    REAL(sOut)[ii] = (*lFOfX)(REAL(SampleStart)[ii],MT);
    REAL(sOut)[ii] += GivePriorOfX(REAL(SampleStart)[ii], MT);
  }
  //  Re-unscale things
  F77_CALL(dscal)(&MT->NumJ, REAL(sOnSigma), MT->D, &One);
  ScaleOfEigenVectors *= REAL(sOnSigma)[0];
  MT->Temperature = TempTemperature;
  Rf_unprotect(1); 
  return(sOut);
}

// We may consider non-parametric priors for f_2(tau_k).  However,
// We can only hope to numerically integrate these, and we will never be confident
// We are covering the whole space. 
SEXP BayesSpikeCL::BPriorLOfX(SEXP SampleStart) {
SEXP sOut = R_NilValue;
  Rf_protect(sOut = Rf_allocVector(REALSXP, Rf_length(SampleStart)));
  int ii;
  double TempTemperature = MT->Temperature; MT->Temperature = 1.0;

  //  Note that we must descale Eigenvalues of XtX for the calculations:
  SetupMTForOnii(OnTauIndex); int One = 1;
  for (ii = 0; ii < Rf_length(SampleStart); ii++) {
    REAL(sOut)[ii] = GivePriorOfX(REAL(SampleStart)[ii], MT);
  }
  //  Re-unscale things
  F77_CALL(dscal)(&MT->NumJ, REAL(sOnSigma), MT->D, &One);
  ScaleOfEigenVectors *= REAL(sOnSigma)[0];
  MT->Temperature = TempTemperature;
  Rf_unprotect(1); 
  return(sOut);
}

double BayesSpikeCL::BinFindROfX(double StartX, double GoalVal) {
  double MaxX = get_MaxX();
  
 SetupMTForOnii(OnTauIndex); int One = 1;
 
 double returner = (BinFindX( lFOfX, StartX, GoalVal, 
    0.0, MaxX, 100, SpliceMove, MT, CauchyEpsilon)); 
 
 F77_CALL(dscal)(&MT->NumJ, REAL(sOnSigma), MT->D, &One); 
 ScaleOfEigenVectors *= REAL(sOnSigma)[0];
 return(returner); 
}

// This is useful for investigating whether the histogram of SampleTauSpliced
//  actually resembles the target density.
SEXP BayesSpikeCL::r2(int reps) {
 SEXP sOut = R_NilValue;
 if (reps <= 0) { return(R_NilValue); }
 
 Rf_protect(sOut = Rf_allocVector(REALSXP, reps));
  
 SetupMTForOnii(OnTauIndex);  int One = 1;
 double MaxTauEst = 2;
 MaxTauEst = MaximizeMe(MaxTauEst, MT,
    MaximizeMeCauchyTotalIters, CauchyEpsilon, Verbose -2);
 double FakeOuttau;
 for (int ii = 0; ii < reps; ii++) {
   FakeOuttau = SampleTauSpliced(MaxTauEst, MT, 
        MaximizeMeCauchyTotalIters, MaxIters, CauchyEpsilon,
        SpliceMove, NumSpliceSampleReps, DoMax, Verbose-2, 
        REAL(sOnTau)[MT->Onii]);
   REAL(sOut)[ii] = FakeOuttau;
 }
 F77_CALL(dscal)(&MT->NumJ, &REAL(sOnSigma)[0], MT->D, &One);
 ScaleOfEigenVectors *= REAL(sOnSigma)[0];
 Rf_unprotect(1);
 return(sOut);
}

// Just samples a single tau_k from f_2
SEXP BayesSpikeCL::r2o() {

 
SetupMTForOnii(OnTauIndex);  int One = 1;
 
 double  MaxTauEst = MaximizeMe(2, MT,
    MaximizeMeCauchyTotalIters, CauchyEpsilon, Verbose -2);
  double FakeOuttau = SampleTauSpliced(MaxTauEst, MT, 
        MaximizeMeCauchyTotalIters, MaxIters, CauchyEpsilon,
        SpliceMove, NumSpliceSampleReps, DoMax, Verbose-2, REAL(sOnTau)[MT->Onii]);
 F77_CALL(dscal)(&MT->NumJ, &REAL(sOnSigma)[0], MT->D, &One);
 ScaleOfEigenVectors *= REAL(sOnSigma)[0];
 SEXP sOut = R_NilValue;
 Rf_protect(sOut = Rf_allocVector(REALSXP, 1));
 REAL(sOut)[0] = FakeOuttau;
 Rf_unprotect(1);
 return(sOut);
}

// Most of these BayesSpikeCL call functions from BayesSpikeHelpers.cc related
// to f_2(tau_k) and it's derivatives 
//
// Further inspection of quality of f_2 into its derviatives allowed bug detection
// and investigation into correct formula.
double BayesSpikeCL::BDerivlF(double Outtau) {
 SetupMTForOnii(OnTauIndex); int One = 1;
 double returner = (DerivF(MT->NumJ, Outtau, -2.0, 0.0, MT->D,MT->R));
 F77_CALL(dscal)(&MT->NumJ, &REAL(sOnSigma)[0], MT->D, &One);
 ScaleOfEigenVectors *= REAL(sOnSigma)[0];
 return(returner);
}
double BayesSpikeCL::BDerivPrior(double Outtau) {
 SetupMTForOnii(OnTauIndex); 
 double returner = GiveDerivPriorOfX(Outtau, MT);
 int One = 1;
 F77_CALL(dscal)(&MT->NumJ, &REAL(sOnSigma)[0], MT->D, &One);
 ScaleOfEigenVectors *= REAL(sOnSigma)[0];
 return(returner);

}
double BayesSpikeCL::BSDerivlF(double Outtau) {
 SetupMTForOnii(OnTauIndex); 
 double returner = SDerivF(MT->NumJ, Outtau, -2.0, 0.0, MT->D,MT->R);
 int One = 1;
 F77_CALL(dscal)(&MT->NumJ, &REAL(sOnSigma)[0], MT->D, &One);
 ScaleOfEigenVectors *= REAL(sOnSigma)[0];
 return(returner);
}
double BayesSpikeCL::BSDerivPrior(double Outtau) {
 SetupMTForOnii(OnTauIndex); int One = 1;
 double returner = GiveSDerivPriorOfX(Outtau, MT);
 F77_CALL(dscal)(&MT->NumJ, &REAL(sOnSigma)[0], MT->D, &One);
 ScaleOfEigenVectors *= REAL(sOnSigma)[0];;
 return(returner);
}

////////////////////////////////////////////////////////////////////////
//
SEXP BayesSpikeCL::BAllDeriv(SEXP sIn) {
 SEXP sOut = R_NilValue;
 Rf_protect(sOut = Rf_allocVector(REALSXP, Rf_length(sIn)));
 SetupMTForOnii(OnTauIndex); int One = 1;
  for (int ii = 0; ii < Rf_length(sIn); ii++) {
    REAL(sOut)[ii] = GiveDerivPriorOfX(REAL(sIn)[ii], MT) +
      (DerivF(MT->NumJ, REAL(sIn)[ii], -2.0, 0.0, MT->D,MT->R));
  }
 F77_CALL(dscal)(&MT->NumJ, &REAL(sOnSigma)[0], MT->D, &One);
 ScaleOfEigenVectors *= REAL(sOnSigma)[0];

 Rf_unprotect(1);
 return(sOut);

}
    
    
    
SEXP BayesSpikeCL::BAllSDeriv(SEXP sIn) {
 SEXP sOut = R_NilValue;
 Rf_protect(sOut = Rf_allocVector(REALSXP, Rf_length(sIn)));
 SetupMTForOnii(OnTauIndex);  int One = 1;
  for (int ii = 0; ii < Rf_length(sIn); ii++) {
    REAL(sOut)[ii] = GiveSDerivPriorOfX(REAL(sIn)[ii], MT) +
      (SDerivF(MT->NumJ, REAL(sIn)[ii], -2.0, 0.0, MT->D,MT->R));
  }
 F77_CALL(dscal)(&MT->NumJ, &REAL(sOnSigma)[0], MT->D, &One);
 ScaleOfEigenVectors *= REAL(sOnSigma)[0];

 Rf_unprotect(1);
 return(sOut);

}
      
// 1/f2(X_m) * 1/(Up(U1)-Low(U1))        
////////////////////////////////////////////////////////////////////////////////
//  MaximizeMe()
//
//   Given MT TauOfFContainer we have a function "f_2(x)" encoded
//    This uses Newton Raphson to get the maximal point of f_2(x)
//
//   This is useful both in AlsSwitchingSampler and in SampleTauSpliced
//
//   We will usually start SampleTauSpliced from the maximal point and sample
//    Down.  After 5 slice samples one can usually get a very good sample from
//    a tough distribution
double MaximizeMe(double StartTau, TauOfFContainer *MT,
  int MaximizeMeCauchyTotalIters, double CauchyEpsilon, int Verbose) {
  int ii;
  
  double Outtau = 0;
  if (Verbose > 1 ) {
    Rprintf("MaximizeMe: Creating MT\n"); R_FlushConsole();
  }
  
  double ExpectMax = 1.0;
  if (StartTau > 0) {
    ExpectMax = StartTau;
  } else {
   double RBar = 0.0;
   double DBar = 0.0;
   for (ii = 0; ii < MT->NumJ; ii++) {
     RBar += MT->R[ii];
   }
   
   DBar = DBar + REAL(MT->staubarnu)[0];
   for (ii = 0; ii < MT->NumJ; ii++) {
     DBar += MT->D[ii]; 
   }
   if (REAL(MT->staubarnu)[0] > 0) {
     RBar /= (MT->NumJ + REAL(MT->staubarnu)[0]);
     DBar /= (MT->NumJ + REAL(MT->staubarnu)[0]);
   } else {
     RBar /= MT->NumJ;
     DBar /= MT->NumJ;
   }
   if (Verbose > 1 ) {
     Rprintf("MaximizeMe: RBar = %f, DBar = %f\n", RBar, DBar); R_FlushConsole();
   }
   ExpectMax =  (RBar + DBar -2) / (DBar*DBar);
   if (ExpectMax <= 0) { ExpectMax = 1.0;}
  }

  double MaxX = -1;
  if (MT->sPriorOftau != NULL && !Rf_isNull(MT->sPriorOftau) && Rf_length(MT->sPriorOftau) > 1 && 
    Rf_length(MT->sPriorXScaling) >= 3) {
    MaxX = Rf_length(MT->sPriorOftau) * REAL(MT->sPriorXScaling)[0] +
         REAL(MT->sPriorXScaling)[2];
    if (REAL(MT->sPriorXScaling)[1] == 1) {
    } else if (REAL(MT->sPriorXScaling)[1] == 2) {
      MaxX = MaxX * MaxX;
    } else if (REAL(MT->sPriorXScaling)[1] == -1) {
      MaxX = exp(MaxX);
    }
  }
    
  if (Verbose > 2 ) {
    Rprintf("MaximizeMe: \n"); R_FlushConsole();
    Rprintf("Let's get a look at things:\n");
    Rprintf(" lFOfX(%f,MT) = %f\n", ExpectMax,
      lFOfX(ExpectMax, MT));
    if (MT->sPriorOftau == NULL || Rf_isNull(MT->sPriorOftau)) {
      Rprintf("MT->sPriorOftau is NULL\n");
    }
    Rprintf("nu in MT = %f, taubarnu = %f\n", REAL(MT->staupriordf)[0],
     REAL(MT->staubarnu)[0]);
    Rprintf(" Prior at %f = %f\n", ExpectMax, GivePriorOfX(ExpectMax, MT));
    Rprintf("logF(%f) = %f for J = %d\n", ExpectMax, 
      logF(MT->NumJ, ExpectMax, REAL(MT->staubarnu)[0], REAL(MT->staubarnu)[0],
      MT->D, MT->R), MT->NumJ); R_FlushConsole();
    Rprintf("Deriv here is %f\n",
      DerivF(MT->NumJ, ExpectMax, REAL(MT->staubarnu)[0], 
      REAL(MT->staubarnu)[0], MT->D, MT->R));
    Rprintf("SecDeriv here is %f\n",
    SDerivF(MT->NumJ, ExpectMax, REAL(MT->staubarnu)[0], 
      REAL(MT->staubarnu)[0], MT->D, MT->R)); R_FlushConsole();
  }  
 
  Outtau = GetSomeMax( &lFOfX, ExpectMax,
    0.0, MaxX, (int) (ceil((double)MaximizeMeCauchyTotalIters/4.0) + 1), 
    .9 * ExpectMax, (void *) MT, 
    CauchyEpsilon); 
  if (Verbose > 2 ) {
    Rprintf("MaximizeMe:  After GetSomeMax, went to Outtau = %f\n", Outtau); R_FlushConsole();
    Rprintf("Let's get a look at things:\n");
    Rprintf(" lFOfX(%f,MT) = %f\n", Outtau,
      lFOfX(Outtau, MT));
    if (MT->sPriorOftau == NULL || Rf_isNull(MT->sPriorOftau)) {
      Rprintf("MT->sPriorOftau is NULL\n");
    }
    Rprintf("nu in MT = %f, taubarnu = %f\n", REAL(MT->staupriordf)[0],
     REAL(MT->staubarnu)[0]);
    Rprintf(" Prior at %f = %f\n", Outtau, GivePriorOfX(Outtau, MT));
    Rprintf("logF(%f) = %f for J = %d\n", ExpectMax, 
      logF(MT->NumJ, Outtau, REAL(MT->staubarnu)[0], REAL(MT->staubarnu)[0],
      MT->D, MT->R), MT->NumJ); R_FlushConsole();
    Rprintf("Deriv here is %f, with Prior = \n",
      DerivF(MT->NumJ, Outtau, REAL(MT->staubarnu)[0]-2, 
      REAL(MT->staubarnu)[0], MT->D, MT->R),
      DerivF(MT->NumJ, Outtau, REAL(MT->staubarnu)[0]-2, 
      REAL(MT->staubarnu)[0], MT->D, MT->R) +
        GiveDerivPriorOfX( Outtau, MT));
    Rprintf("SecDeriv here is %f\n",
    SDerivF(MT->NumJ, Outtau, REAL(MT->staubarnu)[0], 
      REAL(MT->staubarnu)[0]-2, MT->D, MT->R)); R_FlushConsole();
    if (Verbose > 3) {
      Rprintf("More Get Some Max Fun, Start at %f \n", ExpectMax); R_FlushConsole();
      double MoveDist = .9 * ExpectMax;
      double OnX =  ExpectMax; 
      double PropXUp = ExpectMax; double PropFOfXUp;
      double PropXDown = ExpectMax; double PropFOfXDown;
      double MinX = 0.0;  double MaxX = -1;
      double CurFofX  =lFOfX(ExpectMax, MT);
      int MaxInt = 10;
      
       int ii;
       //Rprintf("Get Some Max, starting at %f, with %f, MoveDist = %f, MaxInt = %d\n",
       //  OnX, CurFofX, MoveDist, MaxInt); R_FlushConsole();
       
       for (ii = 0; ii < MaxInt; ii++) {
         PropXUp = OnX + MoveDist;
         PropXDown = OnX - MoveDist;
         if (PropXDown <= MinX) {
           PropXDown = MinX;
         } 
         if (MaxX > MinX && PropXUp >= MaxX) {
           PropXUp = MaxX;
         }
         PropFOfXUp=lFOfX(PropXUp, MT);
         PropFOfXDown=lFOfX(PropXDown, MT);
         if (PropFOfXUp > PropFOfXDown && PropFOfXUp > CurFofX) {
           OnX = PropXUp; CurFofX = PropFOfXUp;
         } else if (PropFOfXDown > CurFofX &&
           PropFOfXDown > PropFOfXUp) {
           OnX = PropXDown; CurFofX = PropFOfXDown;  
         } else {
           MoveDist = MoveDist / 2;
         }
         Rprintf("Get Some Max %d, starting at %f, with %f, MoveDist = %f\n",
           ii, OnX, CurFofX, MoveDist); R_FlushConsole();
       } 
    }
  }
 
  double XOfOn = 0.0;
  double DerivOfXOn = 0.0;  
  double SDerivOfXOn = 0.0;
  double OldXOfOn = lFOfX(Outtau, MT);
  //double OldXOfOn =  logF(J, REAL(Outtau)[0], REAL(snu)[0], 
  //   REAL(staubarnu)[0], REAL(sD), REAL(sRVals));
  double Oldtau = 0.0;
  int MaxLeft = (int) ceil(3.0*(double)MaximizeMeCauchyTotalIters/4.0 );
  if (Verbose > 4)  {
    if (REAL(MT->staubarnu)[0] >=0  &&
      Rf_isNull(MT->sPriorXScaling)) {
      for (ii = 0; ii < MaxLeft; ii++) {
        DerivOfXOn = DerivF(MT->NumJ, Outtau, -2.0, 
        0.0, MT->D, MT->R) +
        GiveDerivPriorOfX( Outtau, MT);
       SDerivOfXOn = SDerivF(MT->NumJ, Outtau, -2.0, 
        0.0, MT->D, MT->R )+
        GiveSDerivPriorOfX( Outtau, MT);
        Outtau = Outtau - DerivOfXOn / SDerivOfXOn;
        XOfOn = lFOfX( Outtau, MT); 
        Rprintf("MaximizeMe Working: Deriv = %f, SDeriv = %f, Outtau = %f, XOfOn = %f \n",
          DerivOfXOn, SDerivOfXOn, Outtau, XOfOn);    
        if (fabs(Oldtau-Outtau) <= CauchyEpsilon) {
          return(Outtau); 
        }
        OldXOfOn = XOfOn;  Oldtau = Outtau;
      }    
    }  
  
  }
  if (REAL(MT->staubarnu)[0] >=0  &&
    Rf_isNull(MT->sPriorXScaling)) {
    for (ii = 0; ii < MaxLeft; ii++) {
      DerivOfXOn = DerivF(MT->NumJ, Outtau, -2.0, 
        0.0, MT->D, MT->R) +
        GiveDerivPriorOfX( Outtau, MT);
      SDerivOfXOn = SDerivF(MT->NumJ, Outtau, -2.0, 
        0.0, MT->D, MT->R )+
        GiveSDerivPriorOfX( Outtau, MT);
      Outtau = Outtau - DerivOfXOn / SDerivOfXOn;
      XOfOn = lFOfX(Outtau,MT);
      if (fabs(Outtau-Oldtau) <= CauchyEpsilon) {
        return(Outtau); 
      }
      OldXOfOn = XOfOn;  Oldtau = Outtau;
    }    
  } else {
    for (ii = 0; ii < MaxLeft; ii++) {
      DerivOfXOn = DerivF(MT->NumJ, Outtau, -2.0, 
        0.0, MT->D, MT->R);
      DerivOfXOn += GiveDerivPriorOfX( Outtau, MT);
      SDerivOfXOn = SDerivF(MT->NumJ, Outtau, -2.0, 
        0.0, MT->D, MT->R );
      SDerivOfXOn += GiveSDerivPriorOfX( Outtau, MT);
      Outtau = Outtau - DerivOfXOn / SDerivOfXOn;
      XOfOn = logF(MT->NumJ, Outtau, -2.0, 
        0.0, MT->D, MT->R);
      XOfOn += GivePriorOfX( Outtau, MT);   
      if (fabs(OldXOfOn - XOfOn) <= CauchyEpsilon ||
        fabs(Oldtau - Outtau) <= CauchyEpsilon ) {
        if(Verbose > 0) {
         Rprintf("MaximizeMe: Less than Epsilon = %f\n", Outtau); R_FlushConsole();
        } 
        return(Outtau); 
      }
      OldXOfOn = XOfOn;  Oldtau = Outtau;
    }  
  }
 if(Verbose > 0) {
   Rprintf("MaximizeMe: Hit End, freeing MT\n"); R_FlushConsole();
 }
 return(Outtau);
}

////////////////////////////////////////////////////////////////////////////////
//   SampleTauSpliced()
//
//    Sample a tau_k from f_2(x)  (the active probability of tau_k)
//      using slice sampling. 
//
double SampleTauSpliced(double MaxTauEst,
  TauOfFContainer *MT, int MaximizeMeCauchyTotalIters, 
  int MaxIters, double CauchyEpsilon,
  double SpliceMove,  int NumSamsPer, int DoMax, 
  int Verbose, double OldTau) {
  
  double Maxtau = 1.0;

  if (DoMax >= 2) {
    if (Verbose > 1) {
      Rprintf("DoMax > 2, SampleTauSpliced Running MaximizeMe\n"); R_FlushConsole();
    }
  
    Maxtau = MaximizeMe(MaxTauEst, MT,
      MaximizeMeCauchyTotalIters,
      CauchyEpsilon, Verbose);
  }  else if (MaxTauEst > 0) {
    Maxtau = MaxTauEst;
  } else{
    Maxtau = Rf_runif(0.0,20.0);
  }
  if (OldTau > 0.0) {
  double OldTauVal = lFOfX(OldTau, MT);
  if (OldTauVal > lFOfX(Maxtau, MT)) {
    Maxtau = OldTau;
  }
  }

  double Unif1 = Rf_runif(0.0,1.0), Unif2 = Rf_runif(0.0,1.0);
  double MaxX = -1;
  
  //if (Verbose > 7) {
  //  Rprintf("SampleTauSpliced first U1 = %f, U2 = %f \n", Unif1, Unif2); R_FlushConsole();
  //}

  if (Verbose > 1) {
    Rprintf("SampleTauSpliced: checking sPriorOftau \n"); R_FlushConsole();
  }  
  //  MaxX is the MaximumX contained in sPriorOftau
  //  We will assume prior is perfectly zero outside of this so that simulation
  //   Is impossible!
  if (MT->sPriorOftau != NULL && !Rf_isNull(MT->sPriorOftau) && Rf_length(MT->sPriorOftau) > 1) {
    MaxX = Rf_length(MT->sPriorOftau) * REAL(MT->sPriorXScaling)[0] +
         REAL(MT->sPriorXScaling)[2];
    if (REAL(MT->sPriorXScaling)[1] == 1) {
    } else if (REAL(MT->sPriorXScaling)[1] == 2) {
      MaxX = MaxX * MaxX;
    } else if (REAL(MT->sPriorXScaling)[1] == -1) {
      MaxX = exp(MaxX);
    }
  }
  int jj = 0;
  
  if (Verbose > 3) {
    Rprintf("SampleTauSpliced: FirstTauSpliced\n"); R_FlushConsole();
  }
  Unif1 = Rf_runif(0.0,1.0); Unif2 = Rf_runif(0.0,1.0); 
  if (Verbose > 7) {
    Rprintf("SampleTauSpliced second U1 = %f, U2 = %f, Maxtau = %f\n", Unif1, Unif2, Maxtau); R_FlushConsole();
  }
  
  double Outtau = 0.0;
  if (Verbose > 7) {
    Rprintf("SampleTauSpliced second U1 = %f, U2 = %f, Maxtau = %f, Current Level = %f, Goal Level = %f\n", 
      Unif1, Unif2, Maxtau, lFOfX(Maxtau,MT), lFOfX(Maxtau, MT)+ log(Unif1)); R_FlushConsole();
  }
  Outtau = SpliceSample(lFOfX, Maxtau, 
    SpliceMove, 0.0, MaxX, Unif1, Unif2, MaxIters, 
    (void*)MT, CauchyEpsilon);
  if (Verbose > 7) {
    Rprintf("SampleTauSpliced After first adventure U1 = %f, U2 = %f, Outtau = %f, Current Level = %f\n", 
      Unif1, Unif2, Outtau, lFOfX(Outtau,MT), lFOfX(Maxtau, MT)+ log(Unif1)); R_FlushConsole();
    Rprintf(" Let's do a manual repeat: MaxIters = %d, SpliceMove= %f \n", MaxIters,SpliceMove);
    Rprintf("  We'll have Min = 0.0, MaxX= %f, CauchyEpislon = %f \n", MaxX,CauchyEpsilon);
    double CurFofX = lFOfX(Maxtau, MT);
    double GoalVal = CurFofX + log(Unif1);
    double Low = BinFindX( lFOfX, Maxtau, GoalVal, 
      0.0, MaxX, MaxIters, -SpliceMove, MT, CauchyEpsilon);
    double High = BinFindX( lFOfX, Maxtau, GoalVal, 
      0.0, MaxX, MaxIters, SpliceMove, MT, CauchyEpsilon); 
    Rprintf("  So we have CurFofX = %f,GoalVal= %f, Low= %f, High = %f, so answer = %f\n",
      CurFofX, GoalVal, Low,High, Low + (High-Low) * Unif2); R_FlushConsole();   
  }
  if (Verbose > 1) {
    Rprintf("SampleTauSpliced: sNumSamsPer = %d\n",
      NumSamsPer); R_FlushConsole();
  }
  if (NumSamsPer >1) {
    for (jj = 1; jj < NumSamsPer; jj++) {
      Unif1 = Rf_runif(0.0,1.0); Unif2 = Rf_runif(0.0,1.0); 
      if (Verbose > 7) {
        Rprintf("SampleTauSpliced jj = %d, U1 = %f, U2 = %f, On Outtau = %f, Current level = %f, GoalVal = %f\n", 
          jj,Unif1, Unif2, Outtau, lFOfX(Outtau, MT), lFOfX(Outtau, MT) + log(Unif1)); R_FlushConsole();
      }
      Outtau= SpliceSample(lFOfX, Outtau, 
        SpliceMove, 0.0, MaxX, Unif1, Unif2, 
        MaxIters, (void*)MT, CauchyEpsilon);
    }  
  }
  if (Verbose > 7) {
    Rprintf("SampleTauSpliced at end jj = %d, U1 = %f, U2 = %f, On Outtau = %f, level = %f\n", 
          jj,Unif1, Unif2, Outtau, lFOfX(Outtau, MT)); R_FlushConsole();
  }
  if (Outtau > 100000.0) {
  Rprintf("SampleTauSpliced Onii=%d, we started with a strange error, Outtau = %f\n", MT->Onii, Outtau);
    Rprintf("  We had thought that MaxTau = %f, but lFofX(Outtau) is %f.\n", 
      Maxtau, lFOfX(Maxtau, MT));
    
    int TooManyMax = 8;  Maxtau = 10000.0; double TryMaxtau = 100000.0;
    int TryMax = 0;
    while (Outtau > 100000.0)  {
      Maxtau = GetSomeMax(lFOfX, TryMaxtau,
        0.0, Outtau * 2, 200, .9 * Outtau,
        MT, CauchyEpsilon);
        TryMax++;
    
      if (Verbose >= 5) {
        Rprintf("   (Onii=%d) But after more GetSomeMax, ntimes = %d, TryMaxtau = %f, MaxTau = %f, with val = %f \n", 
          MT->Onii, TryMax, TryMaxtau, Maxtau, lFOfX(Maxtau, MT)); R_FlushConsole();
      }
      if (Maxtau > MAXTAU) {
        if (Verbose >= 5) { 
          Rprintf("  (Onii=%d) Could mean that there is no max, coming back \n", MT->Onii); R_FlushConsole();
        }
        if (TryMax > TooManyMax)  {
          return(Maxtau);
        }
      } else {
        Outtau = Maxtau;
      }
      TryMaxtau = TryMaxtau  * .5;
      Outtau = Maxtau;
      double FTT = lFOfX(Outtau, MT);
      if (R_isnancpp(FTT) || FTT != FTT || !R_finite(FTT)) {
        Rprintf("  Uh Oh, we're nan or something!"); R_FlushConsole();
        Rprintf("\n MT->R = "); PrintVector(MT->R, MT->NumJ); R_FlushConsole();
        Rprintf("\n MT->D = "); PrintVector(MT->D, MT->NumJ); R_FlushConsole();
        Rprintf("  Onii = %d, length %d\n", MT->Onii, MT->NumJ);  R_FlushConsole();
        Rprintf("  Anyway we might have failed?  \n"); R_FlushConsole();
        return(-100);
      }
    }
    NumSamsPer = 5;
    Outtau = Maxtau;
    for (jj = 0; jj < NumSamsPer; jj++) {
      Unif1 = Rf_runif(0.0,1.0); Unif2 = Rf_runif(0.0,1.0); 
      if (Verbose > 7) {
        Rprintf("SampleTauSpliced jj = %d, U1 = %f, U2 = %f, On Outtau = %f, Current level = %f, GoalVal = %f\n", 
          jj,Unif1, Unif2, Outtau, lFOfX(Outtau, MT), lFOfX(Outtau, MT) + log(Unif1)); R_FlushConsole();
      }
      Outtau= SpliceSample(lFOfX, Outtau, 
        SpliceMove, 0.0, MaxX, Unif1, Unif2, 
        MaxIters, (void*)MT, CauchyEpsilon);
    }    
  }
  return(Outtau);
}

/////////////////////////////////////////////////////////////////////////////
//  Prototype function that practices SliceSampling
//
double SpliceSample( double(*GiveMeFOfX)(double, void *), double StartX,
  double SpliceMove, double MinX, double MaxX, double Unif1, double Unif2,
  int MaxInt, void *HelpStruct, double Epsilon) {
  double CurFofX = (*GiveMeFOfX)(StartX, HelpStruct);
  double GoalVal = CurFofX + log(Unif1);
  double Low = BinFindX( GiveMeFOfX, StartX, GoalVal, 
    MinX, MaxX, MaxInt, -SpliceMove, HelpStruct, Epsilon);
  double High = BinFindX( GiveMeFOfX, StartX, GoalVal, 
    MinX, MaxX, MaxInt, SpliceMove, HelpStruct, Epsilon);    
  return( Unif2 *(High-Low) + Low);
}

double BinFindX(double (*GiveMeFOfX)(double, void*), double StartX,
 double GoalF, double MinX, double MaxX, int MaxInt, double MoveDist,
 void *HelpStruct, double Epsilon) {
 double CurFofX = (*GiveMeFOfX)(StartX, HelpStruct);
 double OnX = StartX;
 double PropX = StartX; double PropFOfX;
 int ii;
 if (CurFofX < GoalF) {
   Rprintf("BinFindX Problem: GoalF = %f > StartF!\n", GoalF, CurFofX);
   R_FlushConsole();
   return(CurFofX);
 }     
 
 //Rprintf("CurFOfX = %f at %f, but GoalF = %f, with MoveDist = %f\n",
 //  CurFofX, OnX, GoalF,MoveDist);
 for (ii = 0; ii < MaxInt; ii++) {
   PropX = OnX + MoveDist;
   if (PropX < MinX) {
     MoveDist = MoveDist / 2;
   } else if (MaxX > MinX && PropX > MaxX) {
     MoveDist = MoveDist / 2;
   } else {
     PropFOfX = (*GiveMeFOfX)(PropX, HelpStruct);
     if (fabs(PropFOfX -GoalF) <= Epsilon) {
       return(PropX);
     }
     if (PropFOfX < GoalF) {
       MoveDist = MoveDist * .5;
     } else {
       OnX = PropX;
       CurFofX = PropFOfX;
       MoveDist *=2;
     }
   }
   //Rprintf("CurFOfX = %f at %f, but GoalF = %f, with MoveDist = %f\n",
   //  CurFofX, OnX, GoalF,MoveDist); R_FlushConsole();
 }
 return(OnX);
}



////////////////////////////////////////////////////////////////////////////////
//  inline int InsertSmallResid(int n, int p, ... )
//
//   Inline function that would put XtX into doing a SamplePartBeta
//
//
inline int InsertSmallResid(int n, int p,
  double *XtResid, double **pXtX, int *XLC, SEXP sBeta, int iFirstRandom, 
  SEXP sOnTau, SEXP tauEndList, int iOnii, double *SmallXtResid,
  SEXP MyEigenValues, SEXP MyEigenVectors, int Verbose) {
 
  if (Verbose > 4) {
    Rprintf("InsertSmallXtResid: Starting \n"); R_FlushConsole();
  }
  int One = 1;  
  int St = 0;
  
  if ( iOnii == 0 ) {
    St = iFirstRandom;
  } else {
    St = (ANINT( (tauEndList), ( iOnii - 1) ) ) + 1;
  }
  int End = (ANINT( (tauEndList), iOnii ) );

  int k = End - St + 1;
  char Trans = 'N'; double OneD=1.0;  int jj;
  int NotIn = 1;
  for (jj = 0; jj < k; jj++) {
    if (XLC[jj+St] < 0) {  NotIn = 0; break; }
  }
  double AbsSum = F77_CALL(dasum)(&k, REAL(sBeta) + St, &One);
  if (NotIn == 0 && AbsSum > 0) {
    if (Verbose > 4) {
      Rprintf("InsertSmallXtResid: Using EigenValues \n");
      Rprintf("  Note XtResid[%d:%d] = ", St, End);
      for (jj = St; jj < End; jj++) { Rprintf("%f,", XtResid[jj]); }
      Rprintf(" \n"); R_FlushConsole();
    }
    //F77_CALL(dscal)(&k, &ZeroD, SmallXtResid, &One);
    double ZeroD = 0.0;
    Trans = 'T';
    F77_CALL(dgemv)(&Trans, &k, &k,&OneD, REAL(MyEigenVectors), &k,
      REAL(sBeta) + St, &One, &ZeroD, SmallXtResid, &One);

    for (jj = 0; jj < k; jj++) {
     (SmallXtResid)[jj] *= REAL(MyEigenValues)[jj];
    }
    Trans = 'N';
   // Rprintf("Attemptying dgemv from in to out!"); R_FlushConsole();
    F77_CALL(dgemv)(&Trans, &k, &k, &OneD, REAL(MyEigenVectors), &k,
      SmallXtResid, &One, &ZeroD, SmallXtResid, &One);
    
    F77_CALL(daxpy)(&k, &OneD, XtResid + St, &One, SmallXtResid, &One);
    return(2);
  
  }
  F77_CALL(dcopy)(&k, XtResid + St, &One, SmallXtResid, &One); 
  if (AbsSum == 0.0) {
    //Rprintf("InsertSmall Resid: Sorry, no Non-ZeroBeta Here!\n");
    return(0); 
  }

  //Rprintf("Insert Small Resid, found some beta!\n"); R_FlushConsole();
  int jY = 1;
  for (jj = 0; jj < k-1; jj++) {
    if (XLC[St+jj] +1 != XLC[St+jj+1]) {
      jY = 0;
      break;
    }
  }
  if (jY == 0) {
    if (Verbose > 4) {
      Rprintf("InsertSmallXtResid: jY = 0, doing partial \n");
      Rprintf("  Note XtResid[%d:%d] = ", St, End);
      for (jj = St; jj < End; jj++) { Rprintf("%f, ", XtResid[jj]); }
      Rprintf(" \n"); R_FlushConsole();
      Rprintf("  That XLC: ");
      for (jj = St; jj < End; jj++) { Rprintf("%d, ", XLC[jj]);}
      Rprintf(" \n"); R_FlushConsole();
    }
    for (jj = 0; jj < k; jj++) {
      if (XLC[jj+St] >= 0) {
        F77_CALL(daxpy)(&k, REAL(sBeta)+jj+St, pXtX[XLC[jj+St]] + St, &One,
          SmallXtResid, &One);
      } else {
        Rprintf("Hey, Doh, un matched issue with XLC here! \n"); R_FlushConsole();
        // If we'rehere, it's probably not important;

        Rprintf("   coordinate %d is not in databse, Beta = %f \n", jj + St,
         REAL(sBeta)[jj+St]);
        Rprintf("   St = %d, jj = %d, XLC[jj+St=%d] = %d\n",
          St, jj, jj+St, XLC[jj+St]);
        Rprintf("   REAL(sBeta)[jj+St=%d] = %f, length(sBeta)=%d\n",
          jj+St, REAL(sBeta)[jj+St], Rf_length(sBeta));
        //Rprintf("Ordered Active[%d]] = ", NumActive); 
        //PrintVector(OrderedActive, NumActive);
        Rprintf("\n   REAL(sBeta) = "); PrintVector(REAL(sBeta), Rf_length(sBeta));
        Rprintf("\n   XLC = "); PrintVector(XLC, p);
        Rprintf("\n   Go inspect error!\n");
        Rf_error("Bad Stuff on XLC, go inspect.\n");
      }                                                          
    }
    return(1);
  }
  if (Verbose > 4) {
    Rprintf("InsertSmallXtResid: jY = 1, doing outside \n");
    Rprintf("  Note XtResid[%d:%d] = ", St, End);
    for (jj = St; jj < End; jj++) { Rprintf("%f, ", XtResid[jj]); }
    Rprintf(" \n"); R_FlushConsole();
    Rprintf("  That XLC: ");
    for (jj = St; jj < End; jj++) { Rprintf("%d, ", XLC[jj]);}
     Rprintf(" \n"); R_FlushConsole();
  }
  for (jj = 0; jj < k; jj++) {
    if (XLC[jj+St] >= 0) {
        F77_CALL(daxpy)(&k, REAL(sBeta)+jj+St, pXtX[XLC[jj+St]] + St, &One,
          SmallXtResid, &One);
    } else {
      Rprintf("Hey, Doh, issue with XLC here, coordinate %d is not in database, %f\n",
           jj+St, REAL(sBeta)[jj+St]); R_FlushConsole();
        
        // If we'rehere, it's probably not important;
    }                                                          
  }
  if (Verbose > 4) {
    Rprintf(" InsertSmallXtResid: at end the output is: \n");
    for (jj = 0; jj < k; jj++) { Rprintf("%f, ", SmallXtResid[jj]); }
    R_FlushConsole();
  }
  return(1);
  Trans = 'N';
  F77_CALL(dgemv)(&Trans, &k, &k, &OneD, pXtX[XLC[St]]  + St, 
    &p, REAL(sBeta) + St, &One,
    &OneD, SmallXtResid, &One);
  return(1);
}

////////////////////////////////////////////////////////////////////////////////
//  double FunInt()
//
//     does the integration used in FullIntegrate()
//
double FunInt(double (*GiveMelFOfX)(double, void*), 
  TauOfFContainer  *MT, int MaxN, double Tweak, double Stop) {
  //Rprintf("Really Starting Fun Int\n!"); R_FlushConsole();
  double Start = 0;
  int TypeInt = 1;
  //Rprintf("Starting Fun Int\n!"); R_FlushConsole();
  double df = 100.0/ MaxN; 
  if (!Rf_isNull(MT->sPriorOftau) && MaxN > Rf_length(MT->sPriorOftau)) {
    MaxN = Rf_length(MT->sPriorOftau);
  }
  if (!Rf_isNull(MT->sPriorXScaling) && Rf_length(MT->sPriorXScaling) > 3) {
    Start = REAL(MT->sPriorXScaling)[2]; df = REAL(MT->sPriorXScaling)[0];
  }

  double RetAns = 0;
  int ii = 0;
  if (TypeInt == 1) {
    double OnF = Start;
    //Rprintf("Starting Type 1, Tweak = %f, df = %f\n!", Tweak, df); R_FlushConsole();
    for (ii = 0; ii < MaxN; ii++) {
      RetAns += exp((*GiveMelFOfX)(OnF, MT)-Tweak);
      if (RetAns != RetAns) {
        Rprintf("Well, ii = %d, OnF = %f, but RetAns is already %f \n",
          ii, OnF, RetAns);
        Rprintf("the lFOfX is %f\n", (*GiveMelFOfX)(OnF, MT));
        R_FlushConsole(); Rf_error("Oops!");
      } else if (Stop > 0 && RetAns * df > Stop) {
        return(OnF);
      }
      OnF+=df;
    } 
    //Rprintf(" Well, after the integration, RetAns = %f, ii = %d, OnF = %f\n ", 
    //  RetAns, ii, OnF);
    //Rprintf(" If we calculate the output, we get exp(Tweak) = %f\n", exp(Tweak));
    return(RetAns * df);
  } else if (TypeInt == 2) {
    double OnF = Start;
    if (Stop <= 0) {
    for (ii = 0; ii < MaxN; ii++) {
      RetAns += exp((*GiveMelFOfX)(OnF*OnF, MT)-Tweak) * (2*OnF *df);
      OnF += df;
    }
    } else {
    for (ii = 0; ii < MaxN; ii++) {
      RetAns += exp((*GiveMelFOfX)(OnF*OnF, MT)-Tweak) * (2*OnF *df);
      if (RetAns * (1 + df * df) > Stop) {
        return(OnF);
      }
      OnF += df;
    }    
    }
    return(RetAns*(1+df*df));
  } else if (TypeInt == -1) {
    double OnF = Start;
    double eOnF = 0;
    if (Stop <= 0)  {
    for (ii = 0; ii < MaxN; ii++) {
      eOnF = exp(OnF);
      RetAns += exp((*GiveMelFOfX)(eOnF, MT)-Tweak) * eOnF;
      OnF += df;
    }
    } else {
    for (ii = 0; ii < MaxN; ii++) {
      eOnF = exp(OnF);
      RetAns += exp((*GiveMelFOfX)(eOnF, MT)-Tweak) * eOnF;
      if (RetAns * df > Stop) {
        return(eOnF);
      }
      OnF += df;
    }    
    }
    return(RetAns*df);
  } 
  return(RetAns * df);
}


////////////////////////////////////////////////////////////////////////////////
//  int FullIntegrate (double (*lFOfX)(double, void*))
//
//    This is the function if HowSample set to 2 or 1 that is run that
//   does a numerical integration of "f_2(x)" which is the "active" density
//   for the current "iOnii" tau value.  
//      Aka f_2(x) = P( tau_k = x | Y, Beta(not in group k)
//
//    The way to integrate this is encoded in differentials in MT
//
//
int FullIntegrate (double (*lFOfX)(double, void*), TauOfFContainer* MT, 
  int NumIntegrands, int iOnii, double *ProbList, double OnPiA, double TweakCons,
  double *pIntegrateDensity, int Verbose, double *PlaceMe, double *pProbScore) {
  double F2 = 0.0;
  if (Verbose > 1) {
    Rprintf("Inserted Integration is Done!\n", MT->NumJ); R_FlushConsole();
  } 
  double Stop = -1.0;
  F2 = FunInt(lFOfX, MT,NumIntegrands, TweakCons,Stop); 
  if (Verbose > 1) {
    Rprintf(" After integration, F2 = %f, but TweakCons = %f\n", F2, TweakCons); R_FlushConsole(); 
  }
  if (F2 != F2) {
    Rprintf("Uh Oh, F2 is an NAN, go look at your terrible code!\n");
    Rf_error("Bad Error!\n");
  }
  int cMoves = 0;
  while (!R_finite(F2) && F2 > 0) {
    TweakCons += 2;
    F2 = FunInt(lFOfX,  MT,NumIntegrands,  TweakCons,Stop); 
    cMoves++;
    if (cMoves > 40) {
      break;
    } 
  }
  while (!R_finite(F2) && F2 < 0) {
    TweakCons -= 2;
    F2 = FunInt(lFOfX, MT,NumIntegrands,  TweakCons, Stop); 
    cMoves++;
    if (cMoves > 40) {
      break;
    }  
  }
  if (!R_finite(F2) && F2 > 0) {
    F2 = (double) 10000000;
  }
  if (!R_finite(F2) && F2 < 0) {
    F2 = (double) -10000000;
  }
  //Rprintf("F2 = %f, TweakCons = %f, NumIntegrands = %d\n", 
  //  F2, TweakCons, NumIntegrands);
  double logOddsOne = log(F2) + TweakCons + 
    log( OnPiA) - log(1.0 - OnPiA ) ;  
  double ProbScore = 1.0;
  
  pIntegrateDensity[0] = F2*exp(TweakCons);
  if (Verbose > 4)  {
    Rprintf("HowSample, we get pIntegratedDensity[0] = %f \n", pIntegrateDensity[0]);
  }
  if (logOddsOne > 30) {
    ProbScore = 1.0;
  } else if (logOddsOne < -15) {
    ProbScore = exp(logOddsOne);
  } else {
    ProbScore = exp(logOddsOne) / (1.0 + exp(logOddsOne));
  }
  pProbScore[0] = logOddsOne;

  //Rprintf("** Writing ProbList maybe? \n");
  if (ProbList != NULL) {
    //Rprintf("** Wrote to ProbList maybe? iOnii = %d\n", iOnii);  R_FlushConsole();
    ProbList[iOnii] = logOddsOne;
  } else {
    //Rprintf("** Not Writing to ProbList maybe? \n");
    if (Verbose > 3) {
      Rprintf(" We're could not write on ProbList\n", iOnii); R_FlushConsole();
    }  
  }
  if (Verbose > 2) {
    Rprintf("SampleANewTau: logOddsOne = %f, ProbScore = %f\n", logOddsOne, ProbScore); R_FlushConsole();
  }
  if (PlaceMe == NULL) {
    if (Rf_runif(0.0,1.0) >= ProbScore) {
      return((int) 0);
    }  else {
      return((int) 1);
    }
  } else {
    if (Rf_runif(0.0,1.0) >= ProbScore) {
      PlaceMe[0] = 0.0;
      return((int) 0);
    }
    double NewStop = Rf_runif(0.0,1.0) * F2;
    PlaceMe[0] = (double) FunInt(lFOfX, MT,NumIntegrands, TweakCons,NewStop); 
    return((int) PlaceMe[0]);
  }
}

//double SampleANewTau(SEXP YResidSq, SEXP XTResid, SEXP XtY, SEXP XtX, 
//  SEXP sBeta, SEXP FirstRandom, SEXP sOnTau, SEXP tauEndList, SEXP Onii,
//  SEXP OutYResidSq, SEXP SmallXtResid,
//  SEXP AllEigenVectors, SEXP AllEigenValues, SEXP diagARASq,
//  SEXP sOnSigma, SEXP sBSMT,
//  SEXP sMaximizeMeCauchyTotalIters, SEXP sCauchyEpsilon, SEXP sSpliceMove, SEXP NumSpliceSampleReps, 
//  SEXP sOnPiA, SEXP DoMax, SEXP DoRNGState, SEXP FakeOuttau,  SEXP MaxTauEst,
//  SEXP DependenciesList, SEXP BFixed, SEXP ProbList, SEXP sHowSample,
//  SEXP sVerbose) {

////////////////////////////////////////////////////////////////////////////////
//  int BayesSpikeCL::SetupMTForOnii(int iOnii)
//
//   MT has valuable information for using SampleTauSpliced and AlsSwitchingSampler
//
//   One has the Eigenvectors and Eigenvalues of the columns of each group
//       If X(k) are vectors of X in group k and X(k)^T X(k) is a 
//       n_k * n_k matrix, and Gamma are the eigen vectos and D_k are the eigenvalues 
//   MT has to calculate    Gamma XTResid = R, With D and R Tau's density is
//     calculated in O(k) time. 
//  
int BayesSpikeCL::SetupMTForOnii(int iOnii) {
int One = 1;  
  St = 0; 
  if (iFirstRandom < 0 || iFirstRandom > p){
    Rprintf("SetupMTForOnii: FirstRandom not setup, FirstRandom = %d, p = %d\n",
      iFirstRandom, p);  R_FlushConsole();
    if (sX == NULL || Rf_isNull(sX)) {
     Rf_error("Also, X is Null dammit!");
    }
    if (MyEigenVectorList == NULL || Rf_isNull(MyEigenVectorList)) {
      Rf_error("MyEigenVector List is NULL dammit");
    }
    if (!Rf_isNull(sX) && !Rf_isNull(Rf_getAttrib(sX, R_DimSymbol))
      && Rf_length(Rf_getAttrib(sX, R_DimSymbol)) == 2) {
      Rprintf(" And Dim X = (%d, %d)", 
        INTEGER(Rf_getAttrib(sX, R_DimSymbol))[0],
        INTEGER(Rf_getAttrib(sX, R_DimSymbol))[1]); R_FlushConsole();
    }
    Rf_error("  SetupMTForOnii: Quit and Broken\n");
  } 

  if (Rf_isNull(tauEndList) || tauEndList == NULL || Rf_length(tauEndList) < 1) {
    Rf_error("SetupMTForOnii(%d): tauEndList is NULL\n", iOnii);
  }    
  if (Rf_length(tauEndList) <= iOnii) {
    Rf_error("SetupMTForOnii(%d): Rf_length(tauEndList) = %d, but iOnii = %d",
      iOnii, Rf_length(tauEndList), iOnii);                                  
  }
  if (sBeta == NULL || Rf_isNull(sBeta) || Rf_length(sBeta) <0) {
    Rf_error("SetupMTForOnii(%d): sBeta not of a Rf_length.\n", iOnii);
  }
                                                                
  //if (p != 
  //  (ANINT(tauEndList, Rf_length(tauEndList)-1)) +1) {
  //  Rf_error("sBeta has Rf_length %d, but tauEndList last is %d!\n",
  //    Rf_length(sBeta), (ANINT(tauEndList, Rf_length(tauEndList)-1)) +1
  //  );
  //}

  if (iOnii >= Rf_length(tauEndList)   || iOnii < 0) {
    Rf_error("Hey, we don't accept iOnii =%d, when l(tauEndList) = %d\n",
      iOnii, Rf_length(tauEndList));
  } 
  OnTauIndex = iOnii;

  if ( iOnii == 0 ) {
    St = iFirstRandom;
  } else {
    St = (ANINT( (tauEndList), (iOnii - 1) ) ) + 1;
  }

  k = (ANINT( (tauEndList), (iOnii) )) - St + 1;
  if (Verbose > 4) {
    Rprintf("SetupMTForOnii(%d): k = %d, St = %d \n", iOnii, k, St); R_FlushConsole();
  }

  if (SmallXtResid == NULL) {
    Rf_error("SetupMTForOnii(%d): SmallXtResid not setup", iOnii);
  }
  if (SmallRVec == NULL) {
    Rf_error("SetupMTForOnii(%d): SmallRVec is NULL\n", iOnii);
  }

  if (AllEigenValues == NULL || Rf_isNull(AllEigenValues)) {
    Rf_error("SetupMTForOnii(%d): AllEigenValues is NULL!", iOnii);
  }
  if (AllEigenVectors == NULL) {
    Rf_error("SetupMTForOnii(%d): AllEigenVectors has Null pointer!", iOnii);
  }
  if (AllEigenVectors == NULL || Rf_isNull(AllEigenVectors)) {
    Rf_error("SetupMTForOnii(%d): AllEigenVectors is Nill Object!", iOnii);
  }
  if (Rf_length(AllEigenValues) <= iOnii) {
    Rf_error(" Error: SetupMTForOnii, iOnii = %d, but EigenValues List has length %d",
      iOnii, Rf_length(AllEigenValues));
  }
  if (Rf_length(AllEigenVectors) <= iOnii) {
    Rf_error(" Error: SetupMTForOnii, iOnii = %d, but EigenVectors List has length %d",
      iOnii, Rf_length(AllEigenVectors));
  }
  if (ScaleOfEigenVectors >= 1.05 || ScaleOfEigenVectors < .95) {
    Rprintf("Error:, SetupMTForOnii[%d], Scale of AllEigenVectors is %f before move to iOnii!\n", iOnii, ScaleOfEigenVectors);
    Rf_error("SetupMTForOnii(%d): Go Check for this!\n", iOnii);
  }
  MyEigenValueList = VECTOR_ELT(AllEigenValues, iOnii ); 
  MyEigenVectorList = VECTOR_ELT(AllEigenVectors, iOnii );
  
  ScaleOfEigenVectors = 1.0;
  if (MyEigenValueList == NULL || Rf_isNull(MyEigenValueList)) {
    Rf_error("SetupMTForOnii(%d): MyEigenValueList is NULL!", iOnii);
  }
  if (MyEigenVectorList == NULL || Rf_isNull(MyEigenVectorList)) {
    Rf_error("SetupMTForOnii(%d): MyEigenVectorList is NULL!", iOnii);
  }
  
  if (sOnSigma == NULL || Rf_isNull(sOnSigma) || !Rf_isReal(sOnSigma)) {
    Rf_error("SetupMTForOnii(%d): Bad Sigma Input!", iOnii);
  }
    
  // Scaling of Eigenvalues and EigenVectors by iSigmaSq
  //  This eliminates noise from the puzzle
  if (Verbose > 5) {
    Rprintf("SetupMTForOnii: Inserting SmallXtResid, FirstRandom = %d\n", 
      iOnii, iFirstRandom ); R_FlushConsole();
  }  

  //Rprintf("Inserting the Small Resid\n"); R_FlushConsole();  
  InsertSmallResid(n,p,XtResid, pXtX, XLC, sBeta, iFirstRandom, 
    sOnTau, tauEndList, iOnii, SmallXtResid,
      MyEigenValueList, MyEigenVectorList, Verbose); 
  ScaleOfSmallXtResid = 1.0;
  
  if (Verbose > 5) {
    Rprintf("SetupMTForOnii(%d): Before Scaling: SmallXtResid = ", iOnii);
    PrintVector(SmallXtResid, k); Rprintf("\n"); R_FlushConsole();
  }
  //Rprintf("Finished Inserting the Small Resid\n"); R_FlushConsole();
  double iOnSigma = 1.0 / REAL(sOnSigma)[0];

  F77_CALL(dscal)(&k, &iOnSigma, SmallXtResid, &One);
  ScaleOfSmallXtResid = iOnSigma;
  
  int DLen = Rf_length(MyEigenValueList);
  if (DLen != k) {
    Rf_error("SetupMTForOnii(%d): dLen=%d does not equal k=%d, iOnii = %d", iOnii, DLen, k, iOnii);
  }
  if (Verbose >= 5) {
    Rprintf("SetupMTForOnii(%d): scaling by Sigma\n", iOnii); R_FlushConsole();
  } 
  F77_CALL(dscal)(&k, &iOnSigma, REAL(MyEigenValueList), &One);
  ScaleOfEigenVectors = iOnSigma;

  if (Verbose >= 4) {
    Rprintf("SetupMTForOnii(%d): Looking at dims of EigenVectorList, k = %d\n",iOnii, k); R_FlushConsole();
  }
  if (Rf_isNull(MyEigenVectorList)) {
    Rf_error("SetupMTForOnii(%d): MyEigenVector List is a Null!", iOnii);
  }
  if (Rf_length(MyEigenVectorList) <= 0) {
    Rf_error("SetupMTForOnii(%d): MyEigenVector List has length 0\n", iOnii);
  }
  if (Verbose >= 4) {
    Rprintf("Let's print EigenVectors Like they're vectors! \n"); R_FlushConsole();
    PrintVector(REAL(MyEigenVectorList), Rf_length(MyEigenVectorList)); 
     R_FlushConsole();
  }
  
  SEXP DimEigenVectors = Rf_getAttrib(MyEigenVectorList, R_DimSymbol);
  if (Verbose >= 4) {
    Rprintf("SetupMTForOnii(%d): We Got DimEigenVectors Out of it \n", iOnii); R_FlushConsole();
  }
  if (DimEigenVectors == NULL || Rf_isNull(DimEigenVectors) ||
    Rf_length(DimEigenVectors) != 2) {
    Rf_error("SetupMTForOnii(%d): No dimension  to MyEigenVectorList %d \n", iOnii);
  }     
  
  //Rf_getAttrib(ReturnMat, R_DimSymbol)
  //Rprintf("Looking at MyEigenVectorList\n"); R_FlushConsole();
  if (INTEGER(DimEigenVectors)[0] != k ||
    INTEGER(DimEigenVectors)[1] != k) {
    Rf_error("SetupMTForOnii(%d), MyEigenVectorList only has dim(%d,%d) relative to %d",
      iOnii, INTEGER(DimEigenVectors)[0],
      INTEGER(DimEigenVectors)[1], k);
  }
  if (SmallRVec == NULL) {
    Rf_error("SetupMTForOnii(%d): SmallRVec is NULL", iOnii);
  } 
  
  char Trans = 'T';
  int ii = 0;
  double ZeroD = 0.0;  double OneD = 1.0;
  if (Verbose >= 5) {
    Rprintf("SetupMTForOnii(%d): Dimension of MyEigenVectorList is (%d,%d) to k = %d\n",
      iOnii, INTEGER(DimEigenVectors)[0],
      INTEGER(DimEigenVectors)[1], k);
    R_FlushConsole();
    Rprintf(" OneD = %f, One = %d, ZeroD = %f\n", OneD, One, ZeroD);
    Rprintf(" MaxTauList = %d \n", MaxTauList);
    R_FlushConsole();
  }
  if (MaxTauList < k) {
    Rf_error("SetupMTForOnii(%d): diagARASq[%d] is too short! for k= %d", 
      iOnii, MaxTauList, k);
  }
  
  //Rprintf("Multiplying EigenValue list by SmallXtResid\n"); R_FlushConsole();
  F77_CALL(dgemv)(&Trans, &k, &k, &OneD, REAL(MyEigenVectorList), &k,
    SmallXtResid, &One, &ZeroD, SmallRVec, &One);
  if (Verbose >= 5) {
    Rprintf("SetupMTForOnii(%d): SmallRVec after multiplication is ", iOnii); R_FlushConsole();
    PrintVector(SmallRVec, k); R_FlushConsole();
    Rprintf("   SetupMTForOnii(%d): About to square now!\n", iOnii); R_FlushConsole();
  }
  if (k > MaxTauList) {
    Rf_error("MaxTauList = %d, but k = %d, SmallRVec is too small!\n",
      k, MaxTauList);
  }
  for (ii = 0; ii < k; ii++) {
    SmallRVec[ii] *= SmallRVec[ii];
  }
  if (Verbose >= 5) {
    Rprintf("SampleANewTau: SmallRVec = "); 
      PrintVector(SmallRVec, k); Rprintf("\n"); R_FlushConsole();
  }    
  if (Verbose >= 4) {
    Rprintf("Setting up MT with D,R, NumJ = %d\n", 
      Rf_length(MyEigenValueList)); R_FlushConsole();
  }
  if (Verbose >= 5) {
    Rprintf("SmallRVec = ");  R_FlushConsole();
    PrintVector(SmallRVec, k); R_FlushConsole();
  }
  if (Verbose >= 5) {
    Rprintf("EigenValues = ");  R_FlushConsole(); 
      PrintVector(REAL(MyEigenValueList), k); R_FlushConsole();
  }
  if (!Rf_isReal(MyEigenValueList)) {
    Rf_error("MyEigenValueList is Not Real!");
  }
  MT->Onii = iOnii;
  MT->NumJ = Rf_length(MyEigenValueList);
  if (MT->NumJ != k) {
    Rf_error("MT->NumJ = %d, but k = %d\n", MT->NumJ, k);
  }
  MT->D = REAL(MyEigenValueList);  MT->R = SmallRVec;

  TweakCons = 0;
  for (ii = 0; ii < k; ii++) {
    TweakCons += -.5 * log(REAL(MyEigenValueList)[ii] + 1) + 
      .5 * SmallRVec[ii] / (REAL(MyEigenValueList)[ii] + 1);
  }
  if (Verbose >= 5) {
    Rprintf("  Now TweakCons = %f \n", TweakCons);  R_FlushConsole(); 
  }
  if (Verbose >= 5) {
    Rprintf("MT created, sD, sR, TweakCons\n", Rf_length(MyEigenValueList)); R_FlushConsole();
    Rprintf("Hey, what's going to happen?  TweakCons = %f, insert 100K\n", TweakCons);
    R_FlushConsole();
    if (!Rf_isNull(MT->sPriorXScaling)) {
      Rprintf(" The MT->sPriorXScaling is ");
      PrintVector(REAL(MT->sPriorXScaling), Rf_length(MT->sPriorXScaling));
      Rprintf("\n"); R_FlushConsole();
    }
    if (!Rf_isNull(MT->sPriorOftau)) {
      Rprintf(" Warning, the prior is not null!\n"); R_FlushConsole();
    } else {
      Rprintf(" Well we have null MT->sPriorOftau!\n"); R_FlushConsole();
    }
    Rprintf(" Our diagARASq = ");PrintVector(SmallRVec, MaxTauList);
    R_FlushConsole();
    Rprintf("\n  Our D = "); PrintVector(REAL(MyEigenValueList),k);
    Rprintf(" Our Resids by the way were \n");
    PrintVector(SmallXtResid,k); Rprintf("\n"); R_FlushConsole();
    Rprintf("Printed That Vector!\n");
  }  
  //Rprintf("Before Integrate sR = "); PrintVector(REAL(diagARASq), k);
  //R_FlushConsole();
  if (HowSample < 0) {
    Rf_error("HowSample is Less than zero = %d!\n", HowSample);
  }
  if (Verbose >= 4) {
    Rprintf("Now we return SampleANewTau(%d)\n", iOnii); R_FlushConsole();
  }
  return(1);
}

////////////////////////////////////////////////////////////////////////////////
//
//  double BayesSpikeCL::SampleANewTau(int iOnii)
//
//   Sample tau for Group iOnii, which could be zero
//
//   One needs to calculate F2 = int f_2(x)dx because probability of activation is
//       P(Active | Current State) = F2 / (F2+F1)
//   Where F1 is the posterior level of when Tau is off
//   F2 is the is the integration of all posterior levels over all values of Tau
//
//   AlsSwitchingSampler is a fast MCMC method for getting this draw
//     If HowSample = 3, AlsSwitchingSampler is performed
//   Numeric integration is achieved using FullIntegrate()
//
//   SampleTauSpliced samples tau from the active distribution 
//      using a Slice-sample
//
//
double BayesSpikeCL::SampleANewTau(int iOnii) {
  int Verbose = this->Verbose - 3;
  double *pProbScore;  double pPlace=0.0;
  if (ProbTau != NULL) {
      pProbScore = ProbTau + iOnii;
  } else {
     pProbScore = &pPlace;
  }
      
  if (iOnii == Rf_length(sOnTau)) {
    Rprintf("SampleANewTau: iOnii submit = %d = length(sOnTau).  On later note, please submit on 0...L-1 scale!", iOnii);
    iOnii = Rf_length(sOnTau)-1;
  } else if (iOnii < 0) {
    Rprintf("SampleANewTau: iOnii submit = %d <0.  On later note, please submit on 0...L-1 scale!", iOnii);
    iOnii = 0;
  }  else if (iOnii > Rf_length(sOnTau)) {
    Rf_error("SampleANewTau, %d iOnii is invalid relative to length = %d.",
      iOnii, Rf_length(sOnTau));
  }
  //Verbose = 8;
  if (Verbose > 0) {
    Rprintf("-------------------------------------------------------------------------\n"); R_FlushConsole();
    Rprintf("-- BayesSpikeSliceSampling.cpp:: SampleANewTau(%d):  Starting iOnii = %d\n", 
      iOnii, iOnii); R_FlushConsole();
  }
  if (Verbose > 1) {
    Rprintf("-- BayesSpikeSliceSampling.cpp:: SampleANewTau(%d): FirstRandom = %d\n", 
      iOnii, iFirstRandom); R_FlushConsole();
  }
  if (Verbose >1) {
    Rprintf("-- BayesSpikeSliceSampling.cpp:: SampleANewTau(%d):  HowSample[%d] is: %d\n ", iOnii, iOnii, HowSample);
    if (ProbFixed == NULL) {
      Rprintf("-- Sorry, no ProbFixed\n"); R_FlushConsole();
    } else {
      Rprintf("-- ProbList[0:(%d-1)] is: ", iFirstRandom); 
      PrintVector(ProbFixed, iFirstRandom); R_FlushConsole();
    }
    if (BFixed == NULL && iFirstRandom > 0) {
        Rf_error("BayesSpikeCL:  Error BFixed is Null\n");
    } else if (BFixed != NULL) {
      Rprintf("-- BFixed[0:(%d-1)] is: ", iFirstRandom); 
      PrintVector(BFixed, iFirstRandom); R_FlushConsole();  
      Rprintf("\n");
    } else {
      Rprintf("-- No Fixed Coordinates. \n");
    }
    if (sOnTau == NULL || Rf_isNull(sOnTau) ||
      Rf_length(sOnTau) != Rf_length(tauEndList)) {
      Rf_error("-- BayesSpikeSliceSampling.cpp:: SampleANewTau(%d), error, sOnTau is Null!\n", iOnii);
    }
    Rprintf("-- OnTau = "); PrintVector(
      REAL(sOnTau), Rf_length(tauEndList));  
    Rprintf("\n"); 
    R_FlushConsole();
  }
  if (Verbose > 1) {
    Rprintf("-- ProbTauList has Rf_length(%d) is : ", Rf_length(tauEndList)); 
    if (ProbTau == NULL) {
      Rprintf(":  I lied, it's null :"); R_FlushConsole();
    } else {
      PrintVector( ProbTau, Rf_length(tauEndList));
    }
    Rprintf("\n"); R_FlushConsole();
  }
  
  if (DependenciesTau != NULL && !Rf_isNull(DependenciesTau) && 
      Rf_length(DependenciesTau) >= iOnii) {
    if (Verbose > -4) {
      Rprintf("-- SampleANewTau:  We're going to check Dependencies \n"); R_FlushConsole();
      Rprintf("-- Length DependenciesTau == %d \n", Rf_length(DependenciesTau));
      R_FlushConsole();
    }
    if (Rf_isReal(DependenciesTau)) {
      Rprintf("-- SampleaNewTau:  Dependencies is Real?\n");
      Rprintf("--  DependenciesTau = "); R_FlushConsole();
      PrintVector(REAL(DependenciesTau), Rf_length(DependenciesTau)); R_FlushConsole();
    } else if (Rf_isInteger(DependenciesTau)) {
      Rprintf("-- DependenciesList: is an Integer?\n");
    } else if (!Rf_isNull(DependenciesTau) && 
      CheckDependencies(
      VECTOR_ELT(DependenciesTau, iOnii), iFirstRandom, BFixed, sOnTau, iOnii) == 0) {
      //REAL(sOnTau)[iOnii] = 0;
      if (Verbose > 1) {
        Rprintf("-- BayesSpikeSliceSampling.cpp:: SampleANewTau(%d): We'll downgrade %d automatically on Dependencies\n", iOnii, iOnii); R_FlushConsole();
      }
      return(0);
    }
  }
  OnTauIndex = iOnii;
  if (ScaleOfEigenVectors >= 1.05 || ScaleOfEigenVectors < .95) {    
    Rprintf("Error:, SetupMT[%d], Scale of AllEigenVectors is %f before move to iOnii!\n");
    Rf_error("SetupMT: Go Check for this!\n");
  }
  #ifdef DOTIMEH
     TauClock1 = clock();
  #endif
  SetupMTForOnii(iOnii);   // Warning, MyEigenList is Scaled after this
  #ifdef DOTIMEH
     TauClock2 = clock();
     if (RsTimePartsList != NULL) {
       REAL(RsTimePartsList->asSexp())[22] += ((double)(TauClock2-TauClock1) / CLOCKS_PER_SEC);
     }
  #endif

  double UsePiA;  double OnMe = -1;
  if (rPriorProbTau != NULL && Rf_length(rPriorProbTau->asSexp()) == Rf_length(tauEndList)) {
    OnMe = REAL(rPriorProbTau->asSexp())[iOnii];
  } else if (rPriorProbTau != NULL && Rf_length(rPriorProbTau->asSexp()) == 2*Rf_length(tauEndList)) {
    OnMe = REAL(rPriorProbTau->asSexp())[iOnii*2];
  } 
  if (R_isnancpp(OnMe) || OnMe < 0.0 || OnMe > 1.0 || !R_finite(OnMe) ||
    R_IsNA(OnMe) || R_IsNaN(OnMe) || OnMe < 0.0) {
    OnMe = -1.0; 
  }
  if (OnMe < 0.0) {
    if (Rf_length(sOnPiA) >= 2) {
      UsePiA = REAL(sOnPiA)[1];
    } else {
      UsePiA = REAL(sOnPiA)[0];
    }
  } else {
    UsePiA = OnMe;
  }
  if (UsePiA == 1.0) {
    if (ProbTau != NULL) {
      ProbTau[iOnii] = 999.0;
    }
    if (Verbose >= 3) {
      Rprintf("--  BayesSpikeSliceSampling.cpp:: SampleANewTau(%d) - UsePiA=%f, We automatically keep this one.",
        iOnii, UsePiA); R_FlushConsole();
    }
    //if (RunProbVector != NULL &&  tt > StartRunProbVector) {
    //   RunProbVector[iFirstRandom+iOnii] += 1.0;
    //   TotEveryProbVector[iFirstRandom+iOnii] += 1;
    //} 
    #ifdef DOTIMEH
     TauClock1 = clock();
    #endif
    double NewFindTau = SampleTauSpliced(MaxTauEst, MT, 
        MaximizeMeCauchyTotalIters, MaxIters, CauchyEpsilon,
        SpliceMove, NumSpliceSampleReps, DoMax, Verbose-2, REAL(sOnTau)[iOnii]);
    #ifdef DOTIMEH
     TauClock2 = clock();
     if (RsTimePartsList != NULL) {
       REAL(RsTimePartsList->asSexp())[23] += ((double)(TauClock2-TauClock1) / CLOCKS_PER_SEC);
     }
    #endif
    if (Verbose > 2) {
      Rprintf("-- BayesSpikeSliceSampling.cpp:: SampleANewTau(%d)", iOnii);
      Rprintf(", autuomatic fixed: de-scaling by Sigma\n"); R_FlushConsole();
    }  
    int One = 1;
    F77_CALL(dscal)(&k, REAL(sOnSigma), REAL(MyEigenValueList), &One); 
    ScaleOfEigenVectors *= REAL(sOnSigma)[0];
    if (Verbose > 2) {
      Rprintf("-- BayesSpikeSliceSampling.cpp:: SampleANewTau(%d)", iOnii);
      Rprintf("  The End, FakeOutTau = %f\n", FakeOuttau); R_FlushConsole();
    } 
    return(NewFindTau);
  }  else if (UsePiA == 0.0) {
    if (ProbTau != NULL) {
      ProbTau[iOnii] = -999.0;
    }
    if (Verbose >= 3) {
      Rprintf("--  BayesSpikeSliceSampling.cpp:: SampleANewTau(%d) - UsePiA=%f, We automatically delete this one.",
        iOnii, UsePiA); R_FlushConsole();
    }
    //if (RunProbVector != NULL &&  tt > StartRunProbVector) {
    //   RunProbVector[iFirstRandom+iOnii] += 0.0;
    //   TotEveryProbVector[iFirstRandom+iOnii] += 1;
    //} 
    if (Verbose > 2) {
      Rprintf("-- BayesSpikeSliceSampling.cpp:: SampleANewTau(%d)", iOnii);
      Rprintf(" autuomatic ignore, de-scaling by Sigma\n"); R_FlushConsole();
    }  
    int One = 1;
    F77_CALL(dscal)(&k, REAL(sOnSigma), REAL(MyEigenValueList), &One); 
    ScaleOfEigenVectors *= REAL(sOnSigma)[0];
    if (Verbose > 2) {
      Rprintf("-- BayesSpikeSliceSampling.cpp:: SampleANewTau(%d)", iOnii);
      Rprintf("  The End, FakeOutTau = %f\n", FakeOuttau); R_FlushConsole();
    } 
    return(0.0);
  }
  if (UsePiA < 0.0) {
    Rprintf("--------------------------------------------------------------------------------\n");
    Rprintf("--  BayesSpikeSliceSampling.cpp:: SampleANewTau(%d) Error, we have iOnii = %d, UsePiA = %f!\n",
     iOnii, iOnii, UsePiA);
    R_FlushConsole();
    Rf_error("-- Seek source of error! \n"); R_FlushConsole();
  }
  if (HowSample == 0) {
    if (Verbose > 2) {
      Rprintf("-- BayesSpikeSliceSampling.cpp:: SampleANewTau(%d) Will do by integration. \n",
        iOnii); R_FlushConsole();
    }
    double NewFindTau = 0.0;
    #ifdef DOTIMEH
      TauClock1 = clock();
    #endif


    FakeOuttau = FullIntegrate( &lFOfX, MT, 
      NumIntegrations, OnTauIndex, ProbTau, UsePiA, TweakCons, 
      &IntegratedDensity, Verbose, &NewFindTau, pProbScore);
    #ifdef DOTIMEH
     TauClock2 = clock();
     if (RsTimePartsList != NULL) {
       REAL(RsTimePartsList->asSexp())[24] += ((double)(TauClock2-TauClock1) / CLOCKS_PER_SEC);
     }
    #endif
    if (Verbose > 2) {
      Rprintf("-- BayesSpikeSliceSampling.cpp:: SampleANewTau(%d)", iOnii);
      Rprintf(" We found by integration (at beginning), NewFind = %f, FakeOuttau = %f, IntegratedDensity = %f\n",
        NewFindTau, FakeOuttau, IntegratedDensity); R_FlushConsole();
    }
    if (NewFindTau > 0.0 && NewFindTau < ThisMaxTau) {
      FakeOuttau = NewFindTau;
    }
    if (ProbTau != NULL) {
      ProbTau[iOnii] = -999.0;
    }
    //if (RunProbVector != NULL && tt > StartRunProbVector) {
    //  RunProbVector[iOnii + iFirstRandom] += IntegratedDensity / (1.0 + IntegratedDensity);
    //  TotEveryProbVector[iFirstRandom+iOnii]++;
    //}
    if (FakeOuttau > 0) { 
      if (Verbose > 2) {
        Rprintf("-- BayesSpikeSliceSampling.cpp:: SampleANewTau(%d)", iOnii);
        Rprintf(": Integration victory, FakeOuttau = %f \n", FakeOuttau); R_FlushConsole();
      }
      #ifdef DOTIMEH
        TauClock1 = clock();
      #endif
      FakeOuttau = SampleTauSpliced(MaxTauEst, MT, MaximizeMeCauchyTotalIters,
       MaxIters, CauchyEpsilon,
       SpliceMove,  NumSpliceSampleReps, DoMax, Verbose -4, REAL(sOnTau)[iOnii]);
      #ifdef DOTIMEH
        TauClock2 = clock();
        if (RsTimePartsList != NULL) {
         REAL(RsTimePartsList->asSexp())[25] += ((double)(TauClock2-TauClock1) / CLOCKS_PER_SEC);
        }
      #endif
      if (FakeOuttau > ThisMaxTau) {
       if (Verbose > -1) {
        Rprintf("-- BayesSpikeSliceSampling.cpp:: SampleANewTau(%d)", iOnii);
         Rprintf(":: Bad SampleTauSpliced, Trying to Sample new Tau using Full Integrate");
       }
       double NewTau = 0.0;
       #ifdef DOTIMEH
         TauClock1 = clock();
       #endif
       FakeOuttau = FullIntegrate( &lFOfX, MT, 
         NumIntegrations, iOnii, ProbTau, UsePiA, TweakCons, 
         &IntegratedDensity, Verbose, &NewTau, pProbScore);
       #ifdef DOTIMEH
         TauClock2 = clock();
         if (RsTimePartsList != NULL) {
          REAL(RsTimePartsList->asSexp())[30] += ((double)(TauClock2-TauClock1) / CLOCKS_PER_SEC);
         }
       #endif
       FakeOuttau = NewTau;    
      }
    }
    if (GoBackAfter > 0 && NumFails >= MaxAlsErrors) {
      GoBackAfter--;
    } else if (GoBackAfter <= 0 && NumFails >= MaxAlsErrors) {
      NumFails = 0;
      GoBackAfter = 0;  HowSample = 3;
    }
    if (GoBackAfterlA > 0 && NumlAZero >= MaxlAErrors) {
      GoBackAfterlA--;
    } else if (GoBackAfterlA <= 0 && NumlAZero >= MaxlAErrors) {
      NumlAZero = 0;
      GoBackAfterlA = 0;  HowSample = 3;
    }
  }  else {
    if (Verbose > 2) {
     Rprintf("-- BayesSpikeSliceSampling.cpp:: SampleANewTau(%d)", iOnii);
     Rprintf(":: We will run switching sampler, FakeOuttau originally = %f\n", 
       FakeOuttau);
      R_FlushConsole();
    } 

      #ifdef DOTIMEH
        unsigned long AlTauClock1 = clock();    
      #endif
    int Out = (int) AlsSwitchingSampler(&lFOfX, iOnii);
      #ifdef DOTIMEH
        unsigned long AlTauClock2 = clock();
        if (RsTimePartsList != NULL) {
         REAL(RsTimePartsList->asSexp())[26] += ((double)(AlTauClock2-AlTauClock1) / CLOCKS_PER_SEC);
        }
      #endif
    if (Verbose  > 1) {
      Rprintf("-- BayesSpikeSliceSampling.cpp:: SampleANewTau(%d):: ", iOnii);
      Rprintf("After AlsSwitchingSampler: Out = %d: %f\n", Out, FakeOuttau);
      if (ProbTau != NULL) {
        Rprintf("   The ProbTau[%d] = %f \n", iOnii, ProbTau[iOnii]);
      }
      R_FlushConsole();
    }
    if (Out == 0) {
      REAL(sOnTau)[iOnii] = 0.0;  FakeOuttau = 0.0;
      //if (RunProbVector != NULL &&  tt > StartRunProbVector) {
      //   RunProbVector[iFirstRandom+iOnii] += 0.0;
      //   TotEveryProbVector[iFirstRandom+iOnii] += 1;
      //} 
    }
    if (FakeOuttau > ThisMaxTau || (REAL(sOnTau)[iOnii] != 0.0 &&
      FakeOuttau > 20 * REAL(sOnTau)[iOnii]) ) {
      double NewTau2 = 1.0;
      if (Verbose > -1) {
        Rprintf("-- BayesSpikeSliceSampling.cpp:: SampleANewTau(%d):: ", iOnii);
        Rprintf(" After Als Sampler: we're trying to draw a smaller Tau"); R_FlushConsole();
      }
      #ifdef DOTIMEH
        TauClock1 = clock();
      #endif
      FakeOuttau = FullIntegrate( &lFOfX, MT, 
        NumIntegrations, iOnii, ProbTau, UsePiA, TweakCons, 
        &IntegratedDensity, Verbose, &NewTau2, pProbScore);
      #ifdef DOTIMEH
          TauClock2 = clock();
        if (RsTimePartsList != NULL) {
         REAL(RsTimePartsList->asSexp())[31] += ((double)(TauClock2-TauClock1) / CLOCKS_PER_SEC);
        }
      #endif
      if (Verbose > -5) {
        Rprintf("Als Sampler: We drew a smaller Tau eventually: %f \n", NewTau2); R_FlushConsole();
      }
      FakeOuttau = NewTau2;    
      //if (RunProbVector != NULL &&  tt > StartRunProbVector) {
      //   RunProbVector[iFirstRandom+iOnii] += 1.0;
      //   TotEveryProbVector[iFirstRandom+iOnii] += 1;
      //} 
    }
  }
  
  if (Verbose > 2) {
    Rprintf("-- BayesSpikeSliceSampling.cpp:: SampleANewTau(%d)", iOnii);
    Rprintf(" de-scaling by Sigma\n"); R_FlushConsole();
  }  
  int One = 1;
  F77_CALL(dscal)(&k, REAL(sOnSigma), REAL(MyEigenValueList), &One); 
  ScaleOfEigenVectors *= REAL(sOnSigma)[0];
  if (Verbose > 2) {
    Rprintf("-- BayesSpikeSliceSampling.cpp:: SampleANewTau(%d)", iOnii);
    Rprintf("  The End, FakeOutTau = %f\n", FakeOuttau); R_FlushConsole();
    Rprintf("--------------------------------------------------------------------------\n"); 
    R_FlushConsole();
  } 
  //REAL(sOnTau)[iOnii] = FakeOuttau;
  //this->Verbose = 0;
  return(FakeOuttau);
}     

////////////////////////////////////////////////////////////////////////////
//  Some values that might help SliceSampling for AlsSwitchingSampler
//
//
//
double BayesSpikeCL::get_xM3() {
  double Df3 = get_Df3();
  return(1.0 / (Df3 + 2.0));
}
double BayesSpikeCL::get_xM4() {
  double Df4 = get_Df4();
  return(1.0/(Df4 + 2.0));
}  

double BayesSpikeCL::get_lM3xM3() {
  double Df3 = get_Df3();   double xM3 = get_xM3();
  return(Rf_dchisq( 1.0 / xM3 , Df3, 1) - 2 * log (xM3));
}
double BayesSpikeCL::get_lM4xM4() {
  double Df4 = get_Df4();   double xM4 = get_xM4();
  return(Rf_dchisq( 1.0 / xM4 , Df4, 1) - 2 * log (xM4));
}

double BayesSpikeCL::get_lUB32() {
  double xM3 = get_xM3();
  double lM3xM3 = get_lM3xM3();  double lM2xM2 = ValAtMaximum()+get_UsePrior();
  double xM2 = MaximizeOn();
  return(lM3xM3 - lM2xM2 + log(xM3) - log(xM2));
}
double BayesSpikeCL::get_lLB42() {
  double xM4 = get_xM4();
  double lM4xM4 = get_lM4xM4();  double lM2xM2 = ValAtMaximum()+get_UsePrior();
  double xM2 = MaximizeOn();
  return(lM4xM4 - lM2xM2 + log(xM4) - log(xM2));
}

double BayesSpikeCL::get_NewX4() {
  double xM2 = MaximizeOn();     double xM4 = get_xM4();
  double Df4 = get_Df4();
  ItsNewX4 = 1;
  return(( xM2 / xM4 ) / Rf_rchisq( Df4 ));
}   
double BayesSpikeCL::get_q4X4() {
  double xM2 = MaximizeOn();     double xM4 = get_xM4();
  double Df4 = get_Df4();
  q4X4 = Rf_dchisq( xM2 / (xM4 * X4) , Df4, 1) - 2  * log (X4) - log(xM4) + log(xM2);
  return(q4X4);
}  

// Get f2X4, getter from the MT X4 distribution 
double BayesSpikeCL::get_f2X4() {
 // double xM2 = MaximizeOn();     double xM4 = get_xM4();
 //  double Df4 = get_Df4();
  double (iOnSigma) = 1.0/REAL(sOnSigma)[0];
  int One = 1;
  F77_CALL(dscal)(&MT->NumJ, &iOnSigma, MT->D, &One);
  f2X4 = (*lFOfX)(X4,MT) + get_UsePrior();  
  F77_CALL(dscal)(&MT->NumJ, REAL(sOnSigma), MT->D, &One);
  
  return(f2X4);
}   

double BayesSpikeCL::get_UsePrior() {
  double UsePrior = 0.0;
  int iOnii = OnTauIndex;
  if (!Rf_isNull(sPriorProbTau)) {
    if (REAL(sPriorProbTau)[iOnii] > 0.0 &&   REAL(sPriorProbTau)[iOnii] < 1.0) {
      UsePrior = log(REAL(sPriorProbTau)[iOnii]) -log (1.0 - REAL(sPriorProbTau)[iOnii]); 
    } else if (REAL(sPriorProbTau)[iOnii] > 1.0 ||
      REAL(sPriorProbTau)[iOnii] < 0.0  || 
      R_isnancpp(REAL(sPriorProbTau)[iOnii]) || !R_finite(REAL(sPriorProbTau)[iOnii]) ||
      R_IsNA(REAL(sPriorProbTau)[iOnii]) || R_IsNaN(REAL(sPriorProbTau)[iOnii]) ) {
      if (Rf_length(sOnPiA) >= 2) {
        UsePrior = log(REAL(sOnPiA)[1]) -log (1.0 - REAL(sOnPiA)[1]); 
      } else {
        UsePrior = log(REAL(sOnPiA)[0]) -log (1.0 - REAL(sOnPiA)[0]); 
      }
    } else if (REAL(sPriorProbTau)[iOnii] == 1.0) {
      UsePrior = log(.999999) - log(.000001);
    } else {
      UsePrior = log(.000001) - log(.999999);
    }
  } else if (Rf_length(sOnPiA) >= 2) {
    UsePrior = log(REAL(sOnPiA)[1]) -log (1.0 - REAL(sOnPiA)[1]);   
  } else {
    UsePrior = log(REAL(sOnPiA)[0]) -log (1.0 - REAL(sOnPiA)[0]);     
  }
  return(UsePrior);
}

double BayesSpikeCL::get_InitlA() {
  return(get_f2X4() - get_q4X4());
}
double BayesSpikeCL::get_lMyHLB() {
  double lUB32 = get_lUB32();
  double lMyHLB = -10;
  if (lUB32 > 0) { 
    if (lUB32 < 10) { 
      lMyHLB = log(1.0 - exp(-lUB32)); 
    } else {
      lMyHLB = -exp(-lUB32);
    }
  } 
  return(lMyHLB);
}


double BayesSpikeCL::get_NewlA() {
  X4 = get_NewX4();
  double lA = get_InitlA();
  double lLB42 = get_lLB42();
  double lUB32 = get_lUB32(); 
  double lMyHLB = get_lMyHLB();
  if (lA + lLB42 -.5 > 0) {
    lA = 0; 
    //if (NumFails >= MaxAlsErrors) {HowSample = 0; GoBackAfter=10 * Rf_length(tauEndList);}
  } else if (lUB32 > 0 && lA + lLB42  - .5< lMyHLB) {
    lA = lMyHLB ; 
  } else {
    lA += lLB42-.5;
  }
  return(lA);
}

// getter for "lA", (log A(t, iOnii)), from last draw of tau(iOnii)
double BayesSpikeCL::get_lA() {
  return(lA);
  /*
  double lA = get_InitlA();
  double lLB42 = get_lLB42();
  double lUB32 = get_lUB32(); 
  double lMyHLB = get_lMyHLB();
  if (lA + lLB42 -.5 > 0) {
    lA = 0; 
    //if (NumFails >= MaxAlsErrors) {HowSample = 0; GoBackAfter=10 * Rf_length(tauEndList);}
  } else if (lUB32 > 0 && lA + lLB42  - .5< lMyHLB) {
    lA = lMyHLB ; 
  } else {
    lA += lLB42-.5;
  }
  return(lA);
  */
}


  

//////////////////////////////////////////////////////////////////////////////
//  int BayesSpikeCL::AlsSwitchingSampler -- SampleANewTau
// 
//  This is key accelerating function for Group exclusive selection.
//
//  Let f2(X) be the target, unnormalized density, int f2(X) dx = F2 is the
//   target odds ratio
//
//  Let q4(X) be a density upper bound for f(2)X with lLB42 being the log
//   lowest upper bound. 
//
//  Meanwhile q3(X) is density lower bound for f2(X)  with lUB32 being the
//   highest lower bound.
//
//  If we are in off state, lA represents probability of remaining at zero.
//  If we are in on state, lB represents probability of going to zero.
//
//  Functional relationship of lA,lB to each other, and draws from q2(X) 
//   target distribution are related.
//
//  The transition density is of form:
//
//    P(Active |  Current State) = [  A    1-  A  | On  ]
//                                 [  B    1 - B  | Off ]
//
//  But A, B are random functions of draws X2 (from q2) and X3 (from q3)
//
//
int BayesSpikeCL::AlsSwitchingSampler(double(*lFOfX)(double, void*), int iOnii) {  
  int Verbose = this->Verbose -2;
  if (Verbose > 4) {
    Rprintf("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"); R_FlushConsole();
    Rprintf("aaa BayesSpikeSliceSampling.cpp:: AlsSwitchingSampler(%d): Start %d, CurrentTau = %f\n", 
      iOnii, iOnii, REAL(sOnTau)[iOnii]); R_FlushConsole();
  }
  
  double UsePrior = 0.0;
  if (!Rf_isNull(sPriorProbTau)) {
     if (REAL(sPriorProbTau)[iOnii] > 0.0 && REAL(sPriorProbTau)[iOnii] < 1.0) {
       UsePrior = log(REAL(sPriorProbTau)[OnTauIndex]) -log (1.0 - REAL(sPriorProbTau)[OnTauIndex]); 
     } else if (REAL(sPriorProbTau)[iOnii] > 1.0 ||
      REAL(sPriorProbTau)[iOnii] < 0.0  || 
      R_isnancpp(REAL(sPriorProbTau)[iOnii]) || !R_finite(REAL(sPriorProbTau)[iOnii]) ||
      R_IsNA(REAL(sPriorProbTau)[iOnii]) || R_IsNaN(REAL(sPriorProbTau)[iOnii])) {
        if (Rf_length(sOnPiA) >= 2) {
          UsePrior = log(REAL(sOnPiA)[1]) -log (1.0 - REAL(sOnPiA)[1]);   
        } else {
          UsePrior = log(REAL(sOnPiA)[0]) -log (1.0 - REAL(sOnPiA)[0]);     
        }
     } else if (REAL(sPriorProbTau)[iOnii] == 0.0) {
       UsePrior = log(.9999999)-log(.0000001); 
     } else {
       UsePrior = log(.0000001)-log(.9999999); 
     }  
  } else if (Rf_length(sOnPiA) >= 2) {
    UsePrior = log(REAL(sOnPiA)[1]) -log (1.0 - REAL(sOnPiA)[1]);   
  } else {
    UsePrior = log(REAL(sOnPiA)[0]) -log (1.0 - REAL(sOnPiA)[0]);     
  }
  
  if (MT == NULL) {
    Rf_error("aa MT is NULL\n");
  }
  if (MT->staubarnu == NULL || Rf_isNull(MT->staubarnu)) {
    Rf_error("aa Error: MY->staubarnu is NULL\n");
  }
  if (MT->NumJ <= 0) {
    Rf_error("aa Error: MT->NumJ is <= 0\n");
  }
  //double iNumJ = 1.0 / MT->NumJ;
  //Rprintf("AlsSwitchingSampler To Get Dbar, Rbar, NumJ = %d\n",
  //  MT->NumJ); R_FlushConsole();
  //if (MT->NumJ != Rf_length(MT->sD)) {
  //  Rf_error("AlsSwitchingSampler: MT->NumJ does not equal Rf_length of sD!");
  //}
  //if (MT->NumJ > Rf_length(MT->sR)) {
  //  Rf_error("AlsSwitchingSmapler: MT->NumJ longer than sR!");
  //}
  if (MT->R == NULL || MT->D == NULL) {
    Rf_error("aa AlsSwitchingSampler: give nonNull D, R\n");
  }
  //double Dbar = F77_CALL(dasum)(&(MT->NumJ), REAL(MT->sD), &One) *iNumJ;  
  //Rprintf("AlsSwitchingSampler To Get Rbar = %d\n",
  //  MT->NumJ); R_FlushConsole();
  //double Rbar = F77_CALL(dasum)(&(MT->NumJ), MT->R, &One) * iNumJ;
  if (Rf_isNull(MT->staubarnu)|| !Rf_isReal(MT->staubarnu)) {
    Rf_error("AlsSwitchingSampler: staubarnu is not REAL");
  }
  if (Rf_isNull(MT->staupriordf)||!Rf_isReal(MT->staupriordf)) {
    Rf_error("AlsSwitchingSampler: staupriordf is not REAL");
  }
  

  //REAL(FakeOuttau)[0] = 2;
  double MaxTauEst = 2.0;
  if (REAL(sOnTau)[iOnii] > 0.0 && lFOfX(REAL(sOnTau)[iOnii], MT) >
    lFOfX(MaxTauEst, MT) ) {
    MaxTauEst = REAL(sOnTau)[iOnii];  
  }
  #ifdef DOTIMEH
        TauClock1 = clock();
  #endif
  MaxTauEst = MaximizeMe(MaxTauEst, MT,
    MaximizeMeCauchyTotalIters, CauchyEpsilon, Verbose -2);
  #ifdef DOTIMEH
        TauClock2 = clock();
        if (RsTimePartsList != NULL) {
         REAL(RsTimePartsList->asSexp())[27] += ((double)(TauClock2-TauClock1) / CLOCKS_PER_SEC);
        }
  #endif
  if (Verbose > 2) {
    Rprintf("*aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n");
    Rprintf("aa BayesSpikeSliceSampling.cpp:: AlsSwitchingSampler(%d), Start!  \n", iOnii);
    Rprintf("aa Al's SwitchingSampler[%d]: MaxTauEst = %f\n", iOnii,
      MaxTauEst);
  }
  if (MaxTauEst <= 0) {
    if (Verbose > 1) {
      Rprintf("aa MaxTauEst Error, went less than zero!\n"); R_FlushConsole();
    }
    MaxTauEst = .01;
  }

  ///////////////////////////////////////////////////////////////////////
  // First we locate the ostensible maximum of the unknown distribution
  //   It is expected that that distribution has tails that are
  //   much like  1/(tau^2)^((NumJ+Prior)/2+1)
  //     Or Inv-Chisquared of NumJ + Prior df.
  //
  // We expect that DF4 to have longer tails (dominate), and DF3 to have shorter tails
  //
  //  By finding the ratio of their relative maximums, and scaling each density to locate
  //    Its maximum at xM2 = MaxTauEst, we should have two densities, DF3 acting as
  //    lower bound and Df4 acting as upper bound, up to scalar concepts.
  double xM2 = MaxTauEst;
  double lM2xM2 = lFOfX(xM2,MT) + UsePrior;
  
  if (Verbose > 3) {
    Rprintf("aa Al's SwitchingSampler[%d]: MaxTauEst = %f, lM2xM2 = %f, -.1 = %f, +.1 = %f \n",
      iOnii, xM2, lM2xM2,
        lFOfX( xM2-.1 > .000001 ? xM2-.1 : .000001, MT ) + UsePrior, 
        lFOfX(xM2+.1, MT)+UsePrior); R_FlushConsole();
  }
  
  double Df3 = REAL(MT->staupriordf)[0] + (double) MT->NumJ + 1;
  
  
  double Df4 = REAL(MT->staupriordf)[0] + (double) MT->NumJ-1;
  //double SDeriv = SDerivF(MT->NumJ, xM2, -2.0, 
  //  0.0, MT->D, MT->R) +   GiveSDerivPriorOfX(xM2, MT);
  //double TryAlpha = DeriveAlpha(SDeriv, xM2);
  //if (TryAlpha > 0.0 && TryAlpha  < Df4 -.5) {
  //   Df4 = TryAlpha;
  //} else { 
  //  double iMaxTauEst = 1.0/MaxTauEst;
  //  for (int kk = 0; kk < MT->NumJ; kk++) {
  //    Df4 += MT->R[kk] / (MT->D[kk] + iMaxTauEst);
  //  }
  //  if (Df4 > REAL(MT->staupriordf)[0] + (double) MT->NumJ-1) {
  //    Df4 = REAL(MT->staupriordf)[0] + (double) MT->NumJ-1; 
  //  } else {
  //    Df4 = .65 * Df4;
  //  }
  //}
  //REAL(MT->staupriordf)[0] + (double) MT->NumJ -1; 
  double xM3 = 1.0 / (Df3 + 2.0);  double xM4 = 1.0/(Df4 + 2.0);
  double lM3xM3 = Rf_dchisq( 1.0 / xM3 , Df3, 1) - 2 * log (xM3);
  double lM4xM4 = Rf_dchisq( 1.0 / xM4 , Df4, 1) - 2 * log (xM4);
 
  if (Verbose > 3) {
    Rprintf("Df3 = %f, xM3 = %f, Max of that dens is lM3xM3 = %f \n",
      Df3, xM3, lM3xM3); R_FlushConsole();
    Rprintf("Df4 = %f, xM4 = %f, Max of that dens is lM4xM4 = %f \n",
      Df4, xM4, lM4xM4); R_FlushConsole();
    Rprintf(" Let's try a little off angle lFOfX(xM2 + .1) = %f\n", lFOfX(xM2+ .1, MT));
    Rprintf(" And Beside at the constant is %f \n", 
      Rf_dchisq( xM2 / ((xM2+.1)*xM4),
      Df4,1) - 2 * log(xM4) + log(xM4) - log(xM2)); R_FlushConsole();
  }
  
  double lUB32 = lM3xM3 - lM2xM2 + log(xM3) - log(xM2);
  double lLB42 = lM4xM4 - lM2xM2 + log(xM4) - log(xM2);
  
  MT->lLB42 = lLB42;  MT->lUB32 = lUB32;
  MT->xM3 = xM3;  MT->xM2 = xM2;  MT->xM4 = xM4;
  MT->lM3xM3 = lM3xM3;  MT->lM4xM4 = lM4xM4;
  
  if (Verbose > 3) {
    Rprintf("lUB32 is supposedly %f > lLB42 is supposedly %f  \n",
      lUB32, lLB42); R_FlushConsole();
  }
    
  //double lW =lM2xM2 + log(xM2) - log(xM2) - lM3xM3;
  
  //double MAll  = 
  //  MT->NumJ * (Dbar + Rbar - 2.0) / ( Dbar * Dbar) +
  //  REAL(MT->staupriordf)[0] * REAL(MT->staubarnu)[0];


    
  X4 = ( xM2 / xM4 ) / Rf_rchisq( Df4 );  ItsNewX4  = 1;
  
  q4X4 = Rf_dchisq( xM2 / (xM4 * X4) , Df4, 1) - 2  * log (X4) - log(xM4) + log(xM2);   
  f2X4 = (*lFOfX)(X4,MT) + UsePrior;

  MT->q4X4 = q4X4;  MT->f2X4 = f2X4;  MT->X4 = X4; MT->UsePrior = UsePrior;

  if (Verbose > 3) {
    Rprintf("---------------------------------------------------------------------\n");
    Rprintf("We drew X4 = %f from xm2 / xm4 * r_4(X) distribution \n", X4);
    Rprintf("  We'll claim at q4X4 = %f, f2X4 = %f\n", q4X4, f2X4);
    Rprintf("  We would like that  (lLB42=%f)+ (f2X4=%f) - (q4X4=%f) = %f < 0 \n",
      lLB42, f2X4, q4X4, lLB42 +f2X4 - q4X4);
    Rprintf(" q4X4 = Rf_dchisq( %f / ( %f * %f ), %f, 1) - 2 * log(%f) - log (%f) + log(%f)\n",
      xM2, xM4, X4, Df4, X4, xM4, xM2);
    Rprintf(" = %f - %f - %f + %f = %f\n", Rf_dchisq( xM2 / ( xM4 * X4), Df4,1),
      2 * log(X4), log(xM4),log(xM2),
       Rf_dchisq( xM2 / ( xM4 * X4), Df4,1)-
         2 * log(X4) - log(xM4) + log(xM2)
      );  R_FlushConsole();
    Rprintf("  But at maximum, this is scaled Rf(max, Df4) - 2l(max) = %f \n",
      Rf_dchisq( 1/(xM4), Df4,1) - 2 * log(xM4) + log(xM4) - log(xM2));
    Rprintf("  Compared to (lM4xM4 = %f) + (log(xM4)=%f) - (log(xM2)=%f = %f \n",
       lM4xM4, log(xM4), log(xM2), lM4xM4 + log(xM4) - log(xM2));
    Rprintf("  At this point q4X4 or probability of that from 4 distribution is %f \n", q4X4);
    Rprintf("  Alternate probability of drawing this if we had the X2 density is %f \n", f2X4);
    R_FlushConsole();
  }

  lA = f2X4 - q4X4;  InitlA = lA; 

  
  double lMyHLB = -10;
  if (lUB32 > 0) { 
    if (lUB32 < 10) { 
      lMyHLB = log(1.0 - exp(-lUB32)); 
    } else {
      lMyHLB = -exp(-lUB32);
    }
  }


  if (Verbose > 3) {
    Rprintf("We drew X4 = %f from xm2 / xm4 * r_4(X) distribution \n", X4);
    Rprintf("  At this point q4X4 or probability of that from 4 distribution is %f \n", q4X4);
    Rprintf("  Alternate probability of drawing this if we had the X2 density is %f \n", f2X4);
    Rprintf("  Hence lA will be f2X4 - q4x4 = %f \n", lA);
    R_FlushConsole();
    if (lA + lLB42 > 0) {
      Rprintf("However, the lower bound by our calculation is supposed to be %f \n",
       lLB42); R_FlushConsole();
      Rprintf("  And hence this did not succeed as an an invert! \n"); R_FlushConsole();
    }
  }

  if (lA + lLB42 -.5 > 0) {
    lA = 0; 
    //if (NumFails >= MaxAlsErrors) {HowSample = 0; GoBackAfter=10 * Rf_length(tauEndList);}
  } else if (lUB32 > 0 && lA + lLB42  - .5< lMyHLB) {
    lA = lMyHLB ; 
  } else {
    lA += lLB42-.5;
  }
  
  if (lA < 0.0 && lMyHLB < 0.0 && lA < lMyHLB) {
    lA = lMyHLB;
  }
  ItsNewX4 = 0;
  MT->lA = lA; MT->f2X4 = f2X4;

  if (Verbose >= 5) {
    Rprintf("lA = %f; lMyHLB = %f; lLB42 = %f; lUB32 = %f; Df3 = %f; Df4 = %f\n",
      lA, lMyHLB, lLB42, lUB32, Df3, Df4);
    Rprintf("X4 = %f; f2X4 = %f; q4X4 = %f;\n", X4, f2X4, q4X4);
    Rprintf("xM3 = %f; lM3xM3 = %f; xM4 = %f; lM4xM4 = %f;\n", xM3,lM3xM3,  xM4, lM4xM4);
    Rprintf("xM2 = %f; lM2xM2 = %f\n", xM2, lM2xM2);
    Rprintf("Sigma = %f \n", REAL(sOnSigma)[0]);
    Rprintf("TweakCons = %f\n", TweakCons);
    R_FlushConsole();
  }
  /*
  if (sHowSample == NULL || Rf_isNull(sHowSample) || !Rf_isReal(sHowSample)) {
    Rf_error("AlsSwitchingSampler: sHowSample is not REAL");
  }
  double MMove = 0.0;
  if (Rf_isNull(sHowSample)|| Rf_length(sHowSample) == 1 && REAL(sHowSample)[0] == 1) {
    lA -= TweakCons;   MMove = -TweakCons;
  } else if (Rf_length(sHowSample) == 1 && REAL(sHowSample)[0] == 2) {
    lA += log( 1-REAL(ProbList)[iOnii]);   MMove = log(1-REAL(ProbList)[iOnii]);
  }  else if (REAL(sHowSample)[0] == 4) { lA -= lW; 
  } else if (Rf_length(sHowSample) >= Rf_length(ProbList)) {
    lA -= REAL(sHowSample)[iOnii + 1]; MMove = -REAL(sHowSample)[iOnii+1];
  }
  */
  //Verbose = 3;
  if (Verbose > 1) {
     Rprintf("AlsSwitchingSampler: %d, lA = %f, X4 = %f, q4X4 = %f, f2X4 = %f, X4 = %f, Df4 = %f, lLB42 = %f, lUB32 = %f\n",
       iOnii, lA, X4, q4X4, f2X4, X4,Df4, lLB42, lUB32); R_FlushConsole();
  }
  MT->rUnif1 = 0.0;
  //if (lA > 0) { REAL(FakeOuttau)[0] = 0.0; return(0); }
  if (REAL(sOnTau)[iOnii] == 0.0) {
    if (this->ProbTau != NULL) {
      if (Verbose > 1) {
        Rprintf(" AlsSwitching Sampler, save %d, lA = %f, so ProbTau[%d] should become %f \n",
          iOnii, lA, iOnii, 1.0 -exp(lA)); R_FlushConsole();
      }
      if (R_finite(lA)) {
        if (lA >= 0.0) {
          this->ProbTau[iOnii] = -100;
        }  else {
          // D = 1.0-exp(LA),  log(D/(1-D))  = log(1.0-exp(lA)) - log(exp(LA))
          this->ProbTau[iOnii] = log(1.0-exp(lA)) - lA;
        }
      } else {
        if (lA < 0) {
          this->ProbTau[iOnii] = 100; 
        } else {
          this->ProbTau[iOnii] = -100;
        }
      }
    } else {
      if (Verbose > 1) {
        Rprintf("  AlsSwitchingSampler:ProbTau is NULL?\n"); R_FlushConsole();
      }
    }
    MT->rUnif1 = Rf_runif(0.0,1.0);
    //Rprintf(" Since CurrentTau = 0.0, lA = %f, is enough to quit \n", lA); R_FlushConsole();

    if (lA >= 0.0) {
      FakeOuttau = 0.0;   lB = -100.0;
      return(0);    
    } else if (MT->rUnif1 >= exp(lA)) {
      if (Verbose > 1) {
        Rprintf("AlsSwitchingSampler: Escape 0, lA = %f, rUnif1 = %f\n", lA, MT->rUnif1);R_FlushConsole();
      }
      #ifdef DOTIMEH
        TauClock1 = clock();
      #endif
      FakeOuttau = SampleTauSpliced(MaxTauEst, MT, 
        MaximizeMeCauchyTotalIters, MaxIters, CauchyEpsilon,
        SpliceMove, NumSpliceSampleReps, DoMax, Verbose-2, REAL(sOnTau)[iOnii]);
      #ifdef DOTIMEH
        TauClock2 = clock();
        if (RsTimePartsList != NULL) {
         REAL(RsTimePartsList->asSexp())[28] += ((double)(TauClock2-TauClock1) / CLOCKS_PER_SEC);
        }
     #endif
      if (FakeOuttau == -100 ||  R_isnancpp(FakeOuttau)) {
         Rprintf("Error, we got a nan return from SampleTauSpliced "); R_FlushConsole();
         Rprintf("CurrentSigma = %f \n", get_OnSigma());
         Rprintf("\nXtResid = "); PrintVector(XtResid, p);
         Rprintf("\nBeta = "); PrintVector(REAL(sBeta), p);
         Rprintf("\nY = "); PrintVector(REAL(sY), n);
         Rprintf("\n  Current sOnTau = "); PrintVector(REAL(sOnTau), Rf_length(sOnTau));
         Rprintf("tauEnd List is ");
         int jj;
         for (jj = 0; jj < Rf_length(tauEndList); jj++) {
           Rprintf(" %d, ", ANINT((tauEndList), (jj) )); 
         }
         Rprintf("\n"); R_FlushConsole();
         //if (Rf_isInteger(tauEndList)) {
         //  PrintVector("\n  CurrenttauEndList = "); PrintVector(INTEGER(tauEndList), Rf_length(tauEndList));
         //}  else {
         //  PrintVector("\n CurrenttauEndList = "); PrintVector(REAL(tauEndList), Rf_length(tauEndList));
         //}
         Rprintf("\n  SmallXtResid = ");  PrintVector(SmallXtResid, MT->NumJ);
         Rprintf(" \n Good Luck Finding the error ! \n");
         Rf_error("Nan FakeOuttau: go look"); 
         SEXP sTry = R_NilValue;
         Rf_protect(sTry = Rf_allocVector(REALSXP, p));         
         int One = 1;  double ZeroD = 0.0;  
         F77_CALL(dscal)(&p, &ZeroD, REAL(sTry), &One);
         set_Beta(sTry);
         Rf_unprotect(1);   
      }
      REAL(sOnTau)[iOnii] = FakeOuttau;
      //if (RunProbVector != NULL && tt > StartRunProbVector) {
      //  RunProbVector[iOnii + iFirstRandom] += 1.0 - exp(lA);
      //  TotEveryProbVector[iFirstRandom+iOnii]++;
      //}
      lB = 999;
      return(1);
    } else {
      //if (RunProbVector != NULL && tt > StartRunProbVector) {
      //  RunProbVector[iOnii + iFirstRandom] += 1.0 - exp(lA);
      //  TotEveryProbVector[iFirstRandom+iOnii]++;
      //}
      FakeOuttau = 0.0;  lB = 999;
      return(0);
    }
  }
  if (Verbose > 1) {
    Rprintf("AlsSwitchingSampler: About to SampleTauSpliced\n"); R_FlushConsole();
  }
 
  //REAL(FakeOuttau)[0] = xM2/ (xM3*rchisq(Df3));
  #ifdef DOTIMEH
     TauClock1 = clock();
  #endif
  FakeOuttau = SampleTauSpliced(MaxTauEst, MT, 
        MaximizeMeCauchyTotalIters, MaxIters, CauchyEpsilon,
        SpliceMove, NumSpliceSampleReps, DoMax, Verbose-2, REAL(sOnTau)[iOnii]);
  #ifdef DOTIMEH
     TauClock2 = clock();
     if (RsTimePartsList != NULL) {
        REAL(RsTimePartsList->asSexp())[28] += ((double)(TauClock2-TauClock1) / CLOCKS_PER_SEC);
     }
  #endif
  if (FakeOuttau == -100 ||  R_isnancpp(FakeOuttau)) {
    Rprintf("Error, we got a nan return from SampleTauSpliced "); R_FlushConsole();
    Rprintf("CurrentSigma = %f \n", get_OnSigma());
    Rprintf("\nXtResid = "); PrintVector(XtResid, p);
    Rprintf("\nBeta = "); PrintVector(REAL(sBeta), p);
    Rprintf("\nY = "); PrintVector(REAL(sY), n);
         Rprintf("\n  Current sOnTau = "); PrintVector(REAL(sOnTau), Rf_length(sOnTau));
         Rprintf("tauEnd List is ");
         int jj;
         for (jj = 0; jj < Rf_length(tauEndList); jj++) {
           Rprintf(" %d, ", ANINT((tauEndList), (jj) )); 
         }
         Rprintf("\n"); R_FlushConsole();
         //if (Rf_isInteger(tauEndList)) {
         //  PrintVector("\n  CurrenttauEndList = "); PrintVector(INTEGER(tauEndList), Rf_length(tauEndList));
         //}  else {
         //  PrintVector("\n CurrenttauEndList = "); PrintVector(REAL(tauEndList), Rf_length(tauEndList));
         //}
         Rf_error("Nan FakeOuttau: go look"); 
         Rprintf("\n  SmallXtResid = ");  PrintVector(SmallXtResid, MT->NumJ);
         Rprintf(" \n Good Luck Finding the error ! \n");
         SEXP sTry = R_NilValue;
         Rf_protect(sTry = Rf_allocVector(REALSXP, p));         
         int One = 1;  double ZeroD = 0.0;  
         F77_CALL(dscal)(&p, &ZeroD, REAL(sTry), &One);
         set_Beta(sTry);
         Rf_unprotect(1); 
      
      return((int) 0); 
  }  
  if (Verbose > 1) {
    Rprintf("Al's Switching Sampler: after SampleTauSpliced FakeOuttau = %f\n",
      FakeOuttau); R_FlushConsole();
  }      
  double X2 = FakeOuttau;
  double q3X2 = Rf_dchisq( xM2/(xM3* X2), Df3, 1 ) -  log(xM3) + log(xM2)  - 2  * log (X2); 
  double f2X2 = (*lFOfX)(X2,MT) + UsePrior;   
  
  MT->X2 = X2;  MT->q3X2 = q3X2;  MT->f2X2 = f2X2;  
  
  if (lA > 0) { lA = 0; 
    NumlAZero++; TotalLAZero++;
    if (NumlAZero > MaxlAErrors) {
      HowSample = 0;  GoBackAfterlA = NumlAZero;
    }
  }
  lB = log(1.0 - exp(lA)) + q3X2  - f2X2; 
  MT->lB=lB;
  
  if (ProbTau != NULL) {
    if (Verbose > 1) {
        Rprintf("  AlsSwitchingSampler: iOnii = %d, lB = %f, so ProTau[%d] should be %f \n",
          iOnii, lB, iOnii, 1.0 - exp(lB)); R_FlushConsole();
    }
    if (R_finite(lB)) {
      if (lB >= 0) {
        ProbTau[iOnii] = -100;
      } else {
        ProbTau[iOnii] = log(1.0 - exp(lB)) - lB;
      }
    } else {
      if (lB >= 0) {
         ProbTau[iOnii] = -100;
      } else {
         ProbTau[iOnii] = 100;
      }
    }
  } else {
    if (Verbose > 0) {
      Rprintf("  AlsSwitchingSampler: ProbTau is Null, alright? \n"); R_FlushConsole();
    }
  }
  if (lB > 0) { 
    NumFails++; TotalFails++; 
    if (Verbose > -1) {
     Rprintf("AlsSwitchingSampler: --TrueFailure --  %d, lA = %f, lB = %f,  X2 = %f, q3X2 = %f, f2X2 = %f, lUB32 = %f\n",
       iOnii, lA, lB, X2, q3X2, f2X2, lUB32); R_FlushConsole();
    }
    if (NumFails >= MaxAlsErrors) {HowSample = 3; GoBackAfter=10 * Rf_length(tauEndList);}
  }  
  if (Verbose > 1) {
     Rprintf("AlsSwitchingSampler:  %d, lA = %f, lB = %f,  X2 = %f, q3X2 = %f, f2X2 = %f, lUB32 = %f\n",
       iOnii, lA, lB, X2, q3X2, f2X2, lUB32); R_FlushConsole();
  }

  if (lB > 0) {   

    FakeOuttau = 0.0;
    REAL(sOnTau)[iOnii] = 0.0; 
    return(0);}
  if (Rf_runif(0.0,1.0) <= exp(lB)) {
    //if (RunProbVector != NULL && tt > StartRunProbVector) {
    //  RunProbVector[iOnii + iFirstRandom] += 1.0 - exp(lB);
    //  TotEveryProbVector[iFirstRandom+iOnii]++;
    //}
    if (ProbTau != NULL) {
      if (lB < 0) {
        ProbTau[iOnii] = log(1.0 - exp(lB)) - lB;
      } else {
        ProbTau[iOnii] = -100;
      }
    }
    FakeOuttau = 0.0;
    REAL(sOnTau)[iOnii] = 0.0;
    return(0);
  } else {
    if (ProbTau != NULL) {
      if (lB < 0) {
        ProbTau[iOnii] =  log(1.0 - exp(lB)) - lB;
      } else {
        ProbTau[iOnii] = -100;
      }
    }
    //if (RunProbVector != NULL && tt > StartRunProbVector) {
    //  RunProbVector[iOnii + iFirstRandom] += 1.0 - exp(lB);
    //  TotEveryProbVector[iFirstRandom+iOnii]++;
    //}
    FakeOuttau = X2;
    REAL(sOnTau)[iOnii] = X2; 
    return(1);
  } 
}


double GetSomeMax(double (*GiveMeFOfX)(double, void*), double StartX,
 double MinX, double MaxX, int MaxInt, double MoveDist,
 void *HelpStruct, double Epsilon) {
 double CurFofX = (*GiveMeFOfX)(StartX, HelpStruct);
 double OnX = StartX;
 double PropXUp = StartX; double PropFOfXUp;
 double PropXDown = StartX; double PropFOfXDown;
 int ii;
 //Rprintf("Get Some Max, starting at %f, with %f, MoveDist = %f, MaxInt = %d\n",
 //  OnX, CurFofX, MoveDist, MaxInt); R_FlushConsole();
 
 for (ii = 0; ii < MaxInt; ii++) {
   PropXUp = OnX + MoveDist;
   PropXDown = OnX - MoveDist;
   if (PropXDown <= MinX) {
     PropXDown = MinX;
   } 
   if (MaxX > MinX && PropXUp >= MaxX) {
     PropXUp = MaxX;
   }
   int CountUps = 0;
   PropFOfXUp=(*GiveMeFOfX)(PropXUp, HelpStruct);
   PropFOfXDown=(*GiveMeFOfX)(PropXDown, HelpStruct);
   if (PropFOfXUp > PropFOfXDown && PropFOfXUp > CurFofX) {
     OnX = PropXUp; CurFofX = PropFOfXUp;
     CountUps++;  
     if (CountUps > 4) {
       MoveDist *= 1.5;  CountUps = (int) 0;
     }
   } else if (PropFOfXDown > CurFofX &&
     PropFOfXDown > PropFOfXUp) {
     OnX = PropXDown; CurFofX = PropFOfXDown; CountUps = 0;  
   } else {
     MoveDist = MoveDist / 2.0;
   }
   //Rprintf("Get Some Max %d, starting at %f, with %f, MoveDist = %f\n",
   //  ii, OnX, CurFofX, MoveDist); R_FlushConsole();
 } 
 //Rprintf("Get Some Max, returning at %f, with %f \n",
 //  OnX, CurFofX); R_FlushConsole();
 return(OnX);

}

inline double DeriveAlpha(double D, double xm2) {
  if (D > 0) { return(0.0); }
  if (xm2 < 0) { return(0.0); }
  double xalpha = pow( D / xm2, .25) - 2.0;
  int jj = 0; double FFunc;  double DDeriv;
  for (jj = 0; jj < 15; jj++) {
    FFunc = pow(xalpha +2.0,3.0) * ( .5 - xm2 * xalpha - 2.0 * xm2);
    DDeriv = -(  (xalpha +2.0) * (xalpha+2.0)* ( 4.0* xm2 * xalpha + 8.0 * xm2 -1.5 ));
    xalpha = xalpha - FFunc / DDeriv;
  }
  if (xalpha > 0.0) { return(xalpha); }
  return(0.0);
}

double BayesSpikeCL::SolveAlpha() {
    double iSigma = 1.0 / REAL(sOnSigma)[0];
    int One=1;
    F77_CALL(dscal)(&MT->NumJ, &iSigma, MT->D, &One);
    double xM2 = MaximizeOn();
    double SDeriv = SDerivF(MT->NumJ, xM2, -2.0, 
      0.0, MT->D, MT->R) +   GiveSDerivPriorOfX(xM2, MT);
    double TryAlpha = DeriveAlpha(SDeriv, xM2);
    F77_CALL(dscal)(&MT->NumJ, REAL(sOnSigma), MT->D, &One);
    return(TryAlpha);
  }
double BayesSpikeCL::get_d4() {
    double iSigma = 1.0 / REAL(sOnSigma)[0];
    int One=1;
    F77_CALL(dscal)(&MT->NumJ, &iSigma, MT->D, &One);
    double xM2 = MaximizeOn();
    double Df4 = REAL(MT->staupriordf)[0] + (double) MT->NumJ-1;
    double SDeriv = SDerivF(MT->NumJ, xM2, -2.0, 
      0.0, MT->D, MT->R) +   GiveSDerivPriorOfX(xM2, MT);
    double TryAlpha = DeriveAlpha(SDeriv, xM2);
    if (TryAlpha > 0.0 && TryAlpha  < Df4 -.5) {
       Df4 = TryAlpha;
    } else { 
      double iMaxTauEst = 1.0/MaxTauEst;
      for (int kk = 0; kk < MT->NumJ; kk++) {
        Df4 += MT->R[kk] / (MT->D[kk] + iMaxTauEst);
      }
      if (Df4 > REAL(MT->staupriordf)[0] + (double) MT->NumJ-1) {
        Df4 = REAL(MT->staupriordf)[0] + (double) MT->NumJ-1; 
      } else {
       Df4 = .65 * Df4;
      }
    }
    F77_CALL(dscal)(&MT->NumJ, REAL(sOnSigma), MT->D, &One);
  //REAL(MT->staupriordf)[0] + (double) MT->NumJ -1;   
    return(Df4);
  }
