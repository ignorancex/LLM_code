/* ========================================================================== */
/*                                                                            */
/*   BayesSpikeCpp.cpp                                                        */
/*   (c) 2011-2019 Alan Lenarcic                                              */
/*    Coded for papers on Diallel at lab of William Valdar, UNC.              */
/*    An implementation of "GroupBayes" estimator as described in methods     */
/*    paper from Lenarcic and Valdar coming to Arxiv.                         */
/*   This is code that has been used in papers related to package             */
/*    BayesDiallel and papers in Genetics: 2013-2014                          */
/*    and supplied to users of BayesDiallel.                                  */
/*                                                                            */
/*                                                                            */
/*   Description                                                              */
/*                                                                            */
/*   This is the main Rcpp style .cppp file for BayesSpike                    */
/*   At the bottom of this file is the  "RCPP_MODULE(modBayesSpikeCL)"        */
/*   module definition which externs to the R front end as a class.           */
/*                                                                            */
/*   Most functions here are stated in "BayesSpike.h"                         */
/*   These are mostly helper functions for the package                        */
/*   They are predominantly commited to Coordinate Descent Operations         */
/*   such as growing and shrinking models and the "XtResid" matrix.           */
/*                                                                            */
/*   "BayesSpikeGibbs.cpp" contains the main "RunAlgorithm" definition        */
/*   "BayesSpike.h" contains the class description and constructor/destructor */
/*    draw.                                                                   */
/*   "BayesSpikeSliceSampling.cpp" conducts slice sampling for Tau Draws      */
/*                                                                            */
/*                                                                            */
/*                                                                            */
/*                                                                            */
/*                                                                            */
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



//////////////////////////////////////////////////////////////////
//  Resize(double Factor)
//
//   Attempts to Resize XtX, kXFinder, iChol, etc to new max memory size
//
//  10/12/2013: We have introduced a mechanism to try to kill 
//     rows out of XtX if it gets too large
//
//  4/9/2014: Resize Checked for dealing with "weighted Robit/iiWeight Data" 
int BayesSpikeCL::Resize(double Factor) {
  int Verbose = this->Verbose -1;
  //Verbose = 5;
  //Rprintf("Running Resize now, Factor = %f, tt = %d\n", Factor, tt); R_FlushConsole();
  int OldKappaMem = OnKappaMem;
  if (OnKappaMem <= 0) {
    Rprintf("Error: Resize, Factor = %f, but OnKappaMem is way off == %d",
      Factor, OnKappaMem); R_FlushConsole();
    Rprintf("Quit on Error!\n"); R_FlushConsole();
    Rf_error("Resize:  Error on KappaMem!"); R_FlushConsole();
  }
  if ( Factor * (  (double) OnKappaMem) / ((double) 500000) >= ((double) 10000)) {
    Rprintf("**********************************************************************************\n");
    Rprintf("** BayesSpikeCL: Resize: Uhoh, OnKappaMem=%d, p=%d, this is going to be memory heavy!\n");
    Rprintf("** Unbridgeable distance!\n");
    R_FlushConsole();
  }
  if (Verbose > 1 || (Verbose >= 0 && p >= 10000)) {
  Rprintf("************************************************ \n");
  R_FlushConsole();
  Rprintf("** BayesSpikeCL:  Resize(tt=%d), OnKappaMem = %d, new factor = %f \n",
    tt, OnKappaMem, Factor);
  Rprintf("** Thus target new OnKappaMem = %d \n", (int) ceil(OnKappaMem * Factor));
  Rprintf("** Current OnKappaS = %d\n", OnKappaS);
  Verbose = 3;
  R_FlushConsole();
  } 
  if (OnKappaMem > p) {
    if (Verbose > 1 || (Verbose >= 1 && p >= 10000)) {
      Rprintf("** Resize:  Cannot add more memory than p, returning \n");
      R_FlushConsole();
    }
    return(-1);
  }
  
  //if (p >= 100000) {
    //Rprintf("** Resize: Checking pXtX order and correctness before we start. \n"); R_FlushConsole();
    //CheckpXtX();
    //Rprintf("** Resize: Before Start, pXtX Resize is correctly ordered.\n"); R_FlushConsole();
  //}
  
  int TryKappaMem =  (int) ceil(OnKappaMem * Factor);
  if (TryKappaMem <= OnKappaMem) {
    TryKappaMem = OnKappaMem + 10;
  }
  if (TryKappaMem  >= MaximumAllocation) { TryKappaMem = MaximumAllocation; }
  if (TryKappaMem > p) { TryKappaMem = p;}
  int OnKSqP = (int) ceil( (TryKappaMem * (TryKappaMem+1)) / 2.0) + 2; 
  if ( ((double)OnKSqP / (double) 50000) > (double) 50000) {
    Rprintf("** Weird, OnKSqP = %d, at this level, this will not be good! ", OnKSqP); R_FlushConsole();
  }
  if (TryKappaMem < OnKappaMem) {
    Rprintf("************************************************************************\n");
    Rprintf("** Resize(Factor=%f), tt = %d", Factor, tt); R_FlushConsole();
    Rprintf("** Error, OnKappaMem = %d, MaximumAllocation = %d, we cannot shrink to TryKappaMem = %d");
    Rprintf("**   We just don't like to shrink, sorry\n"); R_FlushConsole();
    return(-1);
  }
  //FullCrazyTestiWeightedXtX("Resize Start");
  //CheckkXToXLC("Resize Start");
  //CheckkXFinderError("Resize Start");
  
  /*int CountBad = QuickCheckWeightXtX();
  if (CountBad > 0) {
    Rprintf("*** Resize(Factor=%f) Error at beginning, tt=%d, we got QuickCheckWeightXtX, CountBad = %d \n", Factor, tt, CountBad); R_FlushConsole();
    Rf_error("***  Resize QuickCheckWeightXtX Fail. \n");
  } */
  /*
  if (OnKappaMem < MaxSquaresAllocation -1 &&
    ceil(OnKappaMem * Factor) >= MaxSquaresAllocation) {
    TryKappaMem = MaxSquaresAllocation-1;
  } else if (OnKappaMem >= MaxSquaresAllocation -1) {
    TryKappaMem = MaxSquaresAllocation+1;
  } else {
    TryKappaMem = MaxSquaresAllocation-1; 
  }
  */
  int OldMaxLargeSizeSquares = MaxLargeSizeSquares;

  if (TryKappaMem >= MaxSquaresAllocation) {
    Rprintf("*******************************************************\n");
    Rprintf("** BayesSpikeCpp.cpp:Resize()");
    Rprintf("** Memory Goes over MaxSquaresAllocation!  tt=%d/%d, for New OnKappaMem = %d, NumActive = \n", 
      tt, MaxGibbsIters, OnKappaMem, NumActive); R_FlushConsole();
    Rprintf("** This is too large an active set size for full matrix inversion.  \n"); R_FlushConsole();
    Rprintf("** We will simply Set Our MaxLargeSizeSquares =%d to desired %d\n",  MaxLargeSizeSquares,
      MaxSquaresAllocation);

    if (OnKappaMem >= MaxSquaresAllocation) {
      MaxLargeSizeSquares = MaxSquaresAllocation;
      OnKSqP = (int) (ceil(( MaxLargeSizeSquares*(MaxLargeSizeSquares+1)) / 2.0) )+2; 
    }
    OnKappaMem = TryKappaMem;
  }  else {
    OnKappaMem = TryKappaMem;
    MaxLargeSizeSquares = TryKappaMem;  
    OnKSqP = TryKappaMem * (TryKappaMem+1) /2 + 1;
  }
  //int OnKSq = OnKappaMem * OnKappaMem;
  double TestCount;
  int WasResized = 0;
  
  if ( ((double) OnKSqP) / ((double) 50000) > 50000) {
    Rprintf("** RMemReallocD, Uh Oh, we will not succeed in ReAllocating OnKSqP=%d!\n", OnKSqP); R_FlushConsole();
  }
  if (MaxLargeSizeSquares > OldMaxLargeSizeSquares) {
  if (p >= 200000 && Verbose >= 0) {
    Rprintf("** RMemReallocD: start with squares an OnKSqP = %d, RMemReallocD rIChol\n", OnKSqP); R_FlushConsole();
  }
  if (rIChol == NULL) {
    RMemGetD(rIChol, "rIChol", OnKSqP);
  } else {
    if (Verbose >= 0 && p >= 200000) {
      Rprintf("** Test Integrity of rIChol now of size %d, gointg to %d \n",
        OldMaxLargeSizeSquares, MaxLargeSizeSquares); R_FlushConsole();
      TestCount = 0.0;
      for (int iti = 0; iti < OldMaxLargeSizeSquares; iti++) {
        TestCount += rIChol[iti];
      }
      Rprintf("** Integrity of rIChol for first %d points is calced to %f\n",
        OldMaxLargeSizeSquares, TestCount);
      TestCount = 0.0;
      for (int iti = OldMaxLargeSizeSquares; iti < OldMaxLargeSizeSquares*
        (OldMaxLargeSizeSquares+1)/ 2; iti++) {
        TestCount += rIChol[iti];
      }
      Rprintf("** Integrity of rIChol, for last %d points is %f, we will now Realloc.\n",
        OldMaxLargeSizeSquares*(OldMaxLargeSizeSquares+1)/2, TestCount);
      R_FlushConsole();
    }
    RMemReallocD( rIChol, ("rIChol") , OnKSqP );
    if (Verbose >= 0 && p >= 200000) {
      Rprintf("** RMemReallocD succeeded in realloc rIChol.\n");
    }
    for (int iti = OldMaxLargeSizeSquares*(OldMaxLargeSizeSquares+1)/2; 
      iti < OnKSqP; iti++) {
      rIChol[iti] = 0.0;
    }
  }

  if (p >= 200000 && Verbose >= 0) {
    Rprintf("** RMemReallocD: start with squares an OnKSqP = %d, RMemReallocD SmallXtX\n", OnKSqP); R_FlushConsole();
  }
  if (rSmallXtX == NULL) {
    RMemGetD(rSmallXtX, "rSmallXtX", OnKSqP);
  } else {
    if (Verbose >= 0 && p >= 200000) {
      Rprintf("** Test Integrity of rSmallXtX now of size %d, gointg to %d \n",
        OldMaxLargeSizeSquares, MaxLargeSizeSquares); R_FlushConsole();
      TestCount = 0.0;
      for (int iti = 0; iti < OldMaxLargeSizeSquares; iti++) {
        TestCount += rSmallXtX[iti];
      }
      Rprintf("**  Integrity of rSmallXtX for first %d points is calced to %f\n",
        OldMaxLargeSizeSquares, TestCount);
      TestCount = 0.0;
      for (int iti = OldMaxLargeSizeSquares; iti < OldMaxLargeSizeSquares*
        (OldMaxLargeSizeSquares+1)/ 2; iti++) {
        TestCount += rSmallXtX[iti];
      }
      Rprintf("** Integrity of rSmallXtX, for last %d points is %f \n",
        OldMaxLargeSizeSquares*(OldMaxLargeSizeSquares+1)/2, TestCount);
      R_FlushConsole();
    }
    RMemReallocD(rSmallXtX, ("rSmallXtX") , OnKSqP );
    for (int iti = OldMaxLargeSizeSquares*(OldMaxLargeSizeSquares+1)/2; 
      iti < OnKSqP; iti++) {
      rSmallXtX[iti] = 0.0;
    }
  }
  if (p >= 200000 && Verbose >= 0) {
    Rprintf("** RMemReallocD: start with squares an OnKSqP = %d, RMemReallocD rI\n", OnKSqP); R_FlushConsole();
  }
  if (rI == NULL) {
    RMemGetD(rI, "rI", OnKSqP);
  } else {
    RMemReallocD( rI, ("rI") , OnKSqP );
    for (int iti = OldMaxLargeSizeSquares*(OldMaxLargeSizeSquares+1)/2; 
      iti < OnKSqP; iti++) {
      rI[iti] = 0.0;
    }
  }
  if (rICholBack != NULL) {
    RMemReallocD( rICholBack, "rICholBack", OnKSqP );
  }
  } else {
    if (Verbose >= 1 || (p >= 200000 && Verbose >= 0)) {
      Rprintf("** Because  MaxLargeSizeSquares == OldMaxLargeSizeSquares== %d, no RMemRealloc\n",
        MaxLargeSizeSquares);   R_FlushConsole();
    }
  }
  if (p >= 200000 && Verbose >= 0) {
    Rprintf("RMemReallocD: Starting PropBeta, cXtY, smXtY to OnKappaMem = %d\n", OnKappaMem); R_FlushConsole();
  }  
  if (PropBeta == NULL) {
    RMemGetD(PropBeta, "PropBeta", OnKappaMem);
  } else {
    RMemReallocD(PropBeta, "PropBeta", OnKappaMem);
  }
  if (cXtY == NULL) {  
    RMemGetD(cXtY, "cXtY", OnKappaMem);
  } else {
    RMemReallocD(cXtY, "cXtY", OnKappaMem);  
  }
  if (cXtY == NULL) {  
    RMemGetD(smXtY, "smXtY", OnKappaMem);
  } else {
    RMemReallocD(smXtY, "smXtY", OnKappaMem); 
  }

  if (kXFinder == NULL) {  
    RMemGetI(kXFinder, "kXFinder", OnKappaMem);
  } 
  int *NewkXFinder;
  RMemGetI(NewkXFinder, "NewkXFinder", OnKappaMem);
  for (int iti = 0; iti < OnKappaMem; iti++) {
    NewkXFinder[iti] = -1;
  }
  //RMemReallocI(XLC, "XLC", p);
  if (OrderedActive == NULL) {
    RMemGetI(OrderedActive, "OrderedActive", OnKappaMem);
  } else {  RMemReallocI(OrderedActive, "OrderedActive", OnKappaMem); }
 
  if (Verbose >= 3) {
    Rprintf(" ** BayesSpikeCpp.cpp::Resize() We are about to resize iWeightedXtX \n");
    R_FlushConsole();
  }
  short int *NewWeightedXtX = NULL;
  if (iWeightedXtX != NULL) {
    //RMemReallocS(iWeightedXtX, "iWeightedXtX", OnKappaMem); 
    //for (int iti = OnKappaS; iti < OnKappaMem; iti++) {
    //  iWeightedXtX[iti] = 0;
    //}
    NewWeightedXtX = NULL;
    RMemGetS(NewWeightedXtX, "NewWeightedXtX", OnKappaMem);
    for (int iti = 0; iti < OnKappaMem; iti++) {
      NewWeightedXtX[iti] = 0;
    }
  }

  if (Verbose >= 3) {
    Rprintf(" ** BayesSpikeCpp.cpp::Resize() We successfully resized iWeightedXtX \n");
    R_FlushConsole();
  }
  if ( (( (long) p ) * (long(OnKappaMem))) > ((long) 33554432) * ((long) 54) - (long) 1) {
    Rprintf("RMemReallocD(tt=%d): Uh Oh, p=%d, OnKappaMem=%d, larger than usual memory block!\n", 
      tt, p, OnKappaMem);  R_FlushConsole();
  }
  if (p >= 200000 && Verbose >= 0) {
    Rprintf("RMemReallocD: Starting to work on a large pXtX allocation.\n"); R_FlushConsole();
  } 
  
  if (OnKappaMem >= MaxSquaresAllocation) {
    Rprintf("** Hey: Resize XtX, MaxSquaresAllocation is %d, but OnKappaMem is %d, this is not going to be good!\n",
      OnKappaMem); R_FlushConsole();
  }
  double **NewpXtX = (double**) Calloc(OnKappaMem, double*);
  for (int iti = 0; iti < OnKappaMem; iti++) {
    NewpXtX[iti] = NULL;
  }
  if (p >= 200000 && Verbose >= 0) {
    Rprintf("RMemReallocD: NewpXtX is allocated OnKappaMem = %d.\n", OnKappaMem); R_FlushConsole();
  } 
  if (Verbose > 1) {
    Rprintf("** About to move around XLC, pXtX,  kXFinder, iWeightedXtX \n"); R_FlushConsole();
  }
  int ii; //int One = 1; 
  int TotMoved = 0;     double MyZero = 0.0;   //int OldXLC;
  if (OnKappaS > 0 && iiWeight != NULL && iWeightedXtX != NULL) {
    //int CountQuick = QuickCheckWeightXtX();
    //if (CountQuick > 0) {
    //  Rprintf("ERROR ERROR ERROR: BayesSpikeCpp.cpp:::Resize(), though iiWeight is Not Null, QuickCheckWeightXtX got %d fails!\n", CountQuick);
    //  R_FlushConsole();
    //  Rf_error("BayesSpikeCpp.cpp:::Resize(), though iiWeight is Not Null, QuickCheckWeightXtX got %d fails\n", CountQuick);
    //}
  }
  if (OldKappaMem > 0) {
  if (OnKappaS > 0) {
  for (ii = 0; ii < p; ii++) {
    if (XLC[ii] >= 0) {
      WasResized = 0;
      if (XLC[ii] >= OnKappaS) {
        Rprintf("BayesSpikeCpp.cpp: Resize ain't happening ii=%d, XLC[%d] = %d >= OnKappaS=%d\n",
          ii, ii, XLC[ii], OnKappaS); R_FlushConsole();
        Rf_error("BayesSpikeCpp.cpp: Resize, XLC[%d] = %d is wrong versus OnKappaS!",
          ii, XLC[ii], OnKappaS);
      }
      if (iiWeight != NULL && iWeightedXtX != NULL && XLC[ii] >= 0 && XLC[ii] < OnKappaS && (int) iWeightedXtX[XLC[ii]] == 0) {
        WasResized = 1;
        ReweightedOneXtX(ii);
        if (XLC[ii] < 0 || XLC[ii] >= OnKappaS) {
        } else if (iWeightedXtX[XLC[ii]] == 0) {
            Rf_error("BayesSpikeCpp.cpp:::Resize we tried to Update XLC[ii=%d]=%d, but got %d!\n",
              ii, XLC[ii], (int) iWeightedXtX[XLC[ii]]);
        }
      }
      if (XLC[ii] >= OnKappaS || XLC[ii] < 0 || kXFinder[XLC[ii]] != ii) {
        Rprintf("BayesSpikeCpp.cpp: During Resize, bad issues rising.  XLC[ii=%d]=%d, kXFinder[XLC[ii=%d]=%d]=%d, not good\n",
          ii, XLC[ii], ii, XLC[ii], XLC[ii] < OnKappaMem ? kXFinder[XLC[ii]] : -999); R_FlushConsole();
        Rprintf("   New kXFinder was "); if (TotMoved > 0) {
          PrintVector(NewkXFinder, TotMoved);
        }
        Rprintf("\n"); R_FlushConsole();
        Rprintf(" -- Error at this stage. \n");
      }
      NewpXtX[TotMoved] = pXtX[XLC[ii]];
      //F77_CALL(dcopy)(&p, XtX + XLC[ii] * p, &One, NewXtX + p * TotMoved, &One);
      if (iiWeight != NULL && NewWeightedXtX != NULL && XLC[ii] >= 0 && XLC[ii] <= OnKappaS) { NewWeightedXtX[TotMoved] = iWeightedXtX[XLC[ii]]; }
      int One = 1;
      if (ii != 0) {
        if (iiWeight == NULL) {
          MyZero = F77_CALL(ddot)(&n, REAL(sX), &One, REAL(sX) + ii*n, &One);
        } else {
          if (WW == NULL) {
            RMemGetD(WW, "WW", n);
          }
          for (int iti = 0; iti < n; iti++) {
            WW[iti] = iiWeight[iti] * REAL(sX)[iti];
          }
          MyZero = F77_CALL(ddot)(&n, WW, &One, REAL(sX) + ii*n, &One);
        }
        if (fabs(MyZero - *(NewpXtX[TotMoved]+0)) > .0001) {
          Rprintf("ERROR: BayesSpikeCpp.cpp::Resize() ii=%d/%d, TotMoved=%d, MyZero=%f, *NewpXtX[%d]+0 = %f, *pXtX[XLC[%d]=%d]+0=%f \n",
            ii, p, TotMoved, MyZero, TotMoved, *(NewpXtX[TotMoved]+0),
            ii, XLC[ii], *(pXtX[XLC[ii]]+0));
          if (iiWeight != NULL && iWeightedXtX != NULL && XLC[ii] >= 0 && XLC[ii] < OnKappaS) {
            Rprintf("------: Note That iWeightedXtX[XLC[ii=%d]=%d]=%d \n",
              ii, XLC[ii], (int) iWeightedXtX[XLC[ii]]); R_FlushConsole();
          }
          Rprintf("----- Had we performed a Resize? %d\n", WasResized); R_FlushConsole();
          if (WasResized == 0) {
            Rprintf("----- This means we didn't feel we had to do a resize. \n"); R_FlushConsole();
          } else {
            Rprintf("----- We did manually perform a resize. \n"); R_FlushConsole();
          }
          if (REAL(sBeta)[ii] == 0.0) {
            Rprintf("----- Probably we didn't perform a resize since Beta[ii=%d] = 0.0!\n", ii); R_FlushConsole();
          } else {
            Rprintf("----- We probably should have performed a resize since Beta[ii=%d] = %f \n", ii, REAL(sBeta)[ii]); R_FlushConsole();
          }
          Rprintf("ERROR: BayesSpikeCpp.cpp::Resize() We're going to take this as a Resize Error!\n"); R_FlushConsole();
          Rf_error("ERROR IN RESIZE We can't reswap pXtX correctly!\n");
        }
      } else {
        if (iiWeight == NULL) {
          MyZero = F77_CALL(ddot)(&n, REAL(sX), &One, REAL(sX) + 1*n, &One);
        } else {
          if (WW == NULL) {
            RMemGetD(WW, "WW", n);
          }
          for (int iti = 0; iti < n; iti++) {
            WW[iti] = iiWeight[iti] * REAL(sX)[iti];
          }
          MyZero = F77_CALL(ddot)(&n, WW, &One, REAL(sX) + 1*n, &One);
        }
        if (fabs(MyZero - *(NewpXtX[TotMoved]+1)) > .0001) {
          Rprintf("ERROR: BayesSpikeCpp.cpp::Resize() 1 Version since ii=%d/%d, TotMoved=%d, MyZero=%f, *NewpXtX[%d]+1 = %f, *pXtX[XLC[%d]=%d]+1=%f \n",
            ii, p, TotMoved, MyZero, TotMoved, *(NewpXtX[TotMoved]+1),
            ii, XLC[ii], *(pXtX[XLC[ii]]+1));
          if (iiWeight != NULL && iWeightedXtX != NULL && XLC[ii] >= 0 && XLC[ii] < OnKappaS) {
            Rprintf("------: Note That iWeightedXtX[XLC[ii=%d]=%d]=%d",
              ii, XLC[ii], (int) iWeightedXtX[XLC[ii]]); R_FlushConsole();
          } else if (iiWeight == NULL) {
            Rprintf("------:  iiWeight is not null, iiWeight[0] = %f \n", iiWeight[0]); R_FlushConsole();
          } else {
            Rprintf("-----: iiWeight was NULL. \n"); R_FlushConsole();
          }
          Rprintf("ERROR: BayesSpikeCpp.cpp::Resize() We're going to take this as a Resize Error!\n");  R_FlushConsole();
          Rf_error("ERROR IN RESIZE We can't reswap pXtX correctly!\n");
        }      
      }
      if (TotMoved >= OnKappaMem) {
        Rf_error("ERROR: BayesSpikeCpp.cpp::Resize(): TotMoved=%d is beyond OnKappamem=%d, OnKappaS=%d\n",
          TotMoved, OnKappaMem, OnKappaS);
      }
      NewkXFinder[TotMoved] = ii;
      if (iWeightedXtX != NULL && NewWeightedXtX != NULL) {
        NewWeightedXtX[TotMoved] = iWeightedXtX[XLC[ii]];
      }
      XLC[ii] = TotMoved;
      TotMoved++;

      if (TotMoved >= OnKappaS) {
        break;
      }
    }
  }
  }
  }
  if (OldKappaMem > 0 && TotMoved !=  OnKappaS) {
    Rprintf("WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW\n");
    Rprintf("WWWW BayesSpikeCpp.cpp: Resize, weird, OnKappaS=%d, TotMoved=%d!\n", 
      TotMoved, OnKappaS);
    Rprintf("WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW\n");
    R_FlushConsole();
  }
  double *Block = NULL;
  if (Verbose >= 3) {
    Rprintf("Now Allocating for BayesSpikeCpp. Block.\n"); R_FlushConsole();
  }
  if (OldKappaMem == 0 || this->pXtX == NULL) {
    Block = NULL;
    if (Block == NULL) {
       
    }
    //Block = (double *) Calloc(p*OnKappaMem, double);
    for (int ii = 0; ii < OnKappaMem; ii++) {
      this->pXtX[ii] = (double *) Calloc(p, double);
    }
    for (int ii = 0; ii < p; ii++) {
      *(this->pXtX[0] + ii) = 0.0;
    }
     int CnError3 = 0;
      for (int ii = 0; ii < OnKappaMem; ii++) {
        if (this->pXtX[ii] == NULL) {
          Rprintf("ERROR: Resize, ii = %d, this->pXtX[ii=%d] = NULL on Block Allocte.\n", 
            ii, ii);
          Rprintf("    We have OnKappaS=%d, OldKappaMem=%d, TotMoved=%d, OnKappeMem=%d\n",
            OnKappaS, OldKappaMem, TotMoved, OnKappaMem); R_FlushConsole();
          CnError3++;
        }
      }  
      if (CnError3 > 0) {
        Rf_error("  Error:Resize() On Block Allocate CnError3 = %d \n", CnError3);
      } else {
        if (Verbose >= 1) {
          Rprintf("** Resize: We have that tt = %d/%d, OnKappaMem = %d, Resize has good CnError3 = %d\n",
            tt, MaxGibbsIters, OnKappaMem, CnError3);  R_FlushConsole();
        }
      }
  } else if (OnKappaMem > TotMoved) {
    if ( (p > 100000 && (OnKappaMem - TotMoved)  > 50) ||
      ( ((double) p) / ((double)524288) * ((double) (OnKappaMem-TotMoved)) > 
      4096.0 / 32.0 ) ) {
      for (int ii = TotMoved; ii < OnKappaMem; ii++) {
        NewpXtX[ii] = (double *) Calloc(p, double);
      }  
      int CnError1 = 0;
      for (int ii = 0; ii < OnKappaMem; ii++) {
        if (NewpXtX[ii] == NULL) {
          Rprintf("ERROR: Resize, ii = %d, NewpXtX[ii=%d] = NULL on Block Allocte.\n", 
            ii, ii);
          Rprintf("** We have OnKappaS=%d, OldKappaMem=%d, TotMoved=%d, OnKappeMem=%d\n",
            OnKappaS, OldKappaMem, TotMoved, OnKappaMem); R_FlushConsole();
          CnError1++;
        }
      }  
      if (CnError1 > 0) {
        Rf_error("** Error:Resize() On Block Allocate CnError1 = %d \n", CnError1);
      } else {
        if (Verbose >= 1) {
          Rprintf("** Resize: We have that tt = %d/%d, OnKappaMem = %d, Resize has good CnError1 = %d\n",
            tt, MaxGibbsIters, OnKappaMem, CnError1);  R_FlushConsole();
        }
      }
    } else {
      //Block = (double *) Calloc( (OnKappaMem-TotMoved) * p, double);
      for (int ii = TotMoved; ii < OnKappaMem; ii++) {
        NewpXtX[ii] = (double *) Calloc(p, double);
        //Block + (ii-TotMoved) * p;
      }
      Block = NULL;
      int CnError2 = 0;
      for (int ii = 0; ii < OnKappaMem; ii++) {
        if (NewpXtX[ii] == NULL) {
          Rprintf("ERROR: Resize, ii = %d, NewpXtX[ii=%d] = NULL on Block Allocte.\n", 
            ii, ii);
          Rprintf("** We have OnKappaS=%d, OldKappaMem=%d, TotMoved=%d, OnKappeMem=%d\n",
            OnKappaS, OldKappaMem, TotMoved, OnKappaMem); R_FlushConsole();
          CnError2++;
        }
      }  
      if (CnError2 > 0) {
        Rf_error("** Error:Resize() On Block Allocate CnError2 = %d \n", CnError2);
      } else {
        if (Verbose >= 1) {
          Rprintf("** Resize: We have that tt = %d/%d, OnKappaMem = %d, Resize has good CnError2 = %d\n",
          tt, MaxGibbsIters, OnKappaMem, CnError2);  R_FlushConsole();
        }
      }
    }
  }
  if (Verbose > 1 || (p >= 200000 && Verbose >= 0)) {
    Rprintf("** BayesSpikeCpp.cpp: Resize, lets see our allocation of NewpXtX[OnKappaMem-1]\n");
    R_FlushConsole();
    for (int aii = 0; aii < p; aii++) {
      *(NewpXtX[OnKappaMem-1]+aii) = 0.0;
    }
    Rprintf("**   Hey, we finished NewpXtX, OnKappaMem=%d.\n",
      OnKappaMem); R_FlushConsole();
  }
  if (Verbose > 1 || (p >= 200000 && Verbose >= 0)) {
    Rprintf("** BayesSpikeCpp.cpp: Resize:  Filled NewpXtX up to TotMoved %d, p = %d\n",
      TotMoved,  p); R_FlushConsole();
  }
  double **TemppXtX = pXtX;
  this->pXtX = NewpXtX;  TrueOrder = 1;
  if (Verbose >= 2) {
    Rprintf("** BayesSpikeCpp.cpp: Resize(tt=%d), now Freeing TemppXtX, OnKappaMem=%d\n",
      tt, OnKappaMem); R_FlushConsole();
  }
  Free(TemppXtX);  
      double CTn = 0.0;
  if (Verbose >= 2) {
    Rprintf(" **>> BayesSpikeCpp:cpp::Resize() now Testing pXtX for NULLS \n"); R_FlushConsole();
    for (int iti = 0; iti < OnKappaMem; iti++) {
      if (this->pXtX[iti] == NULL) {
        Rprintf("ERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERROR");
        Rprintf("ERROR: BayesSpikeCpp::Resize() faile for iti=%d/%d, OldKappaMem=%d, OnKappaS=%d",
          iti, OnKappaMem, OldKappaMem, OnKappaS); R_FlushConsole();
        Rf_error("  Resize: not good because iti=%d/%d is NULL!  OldKappaMem=%d, OnKappaS=%d\n",
          iti, OnKappaMem, OldKappaMem, OnKappaS);
      }
    }
    Rprintf(" **>> BayesSpikeCpp.cpp: Seriously None of the %d members of pXtX are NULL. \n", OnKappaMem); R_FlushConsole();
    if (OldKappaMem >= OnKappaMem) {
      Rprintf(" **>> BayesSpikeCpp.cpp, weird error, we have that OldKappaMem=%d, OnKappaMem = %d!\n",
        OldKappaMem, OnKappaMem); R_FlushConsole();
    } else {
      Rprintf(" **>> BayesSpikeCpp.cpp Test pXtX[OldKappaMem=%d].\n", OldKappaMem); R_FlushConsole();
      Rprintf(" **>> pXtX[OldKappaMem=%d][0-2] = %.3f, %.3f, %.3f, and Total is",
       OldKappaMem, *(this->pXtX[OldKappaMem]+0),
       p > 2 ? *(this->pXtX[OldKappaMem]+1) : -666,
       p > 3 ? *(this->pXtX[OldKappaMem]+2) : -666      
      ); R_FlushConsole();
      for (int iti = 0; iti < p; iti++) {
        CTn += *(this->pXtX[OldKappaMem]+iti);
      }
      Rprintf(" %f \n"); R_FlushConsole();
    }
    if (OnKappaS >= 1) {
      Rprintf(" **>> pXtX[OnKappaS-1=%d-1] goes to be \n", OnKappaS); R_FlushConsole();
      Rprintf(" **>> pXtX[OnKappaS-1=%d-1][0-2] = %.3f, %.3f, %.3f, and Total is",
        OnKappaS, *(this->pXtX[OnKappaS-1]+0),
        p > 2 ? *(this->pXtX[OnKappaS-1]+1) : -666,
        p > 3 ? *(this->pXtX[OnKappaS-1]+2) : -666      
      ); R_FlushConsole();
      for (int iti = 0; iti < p; iti++) {
        CTn += *(this->pXtX[OnKappaS-1]+iti);
      }
      Rprintf(" %f \n"); R_FlushConsole();
    } else {
      Rprintf(" **>> and OnKappaS = %d, so can't maximize. \n", OnKappaS); R_FlushConsole();
    }
    Rprintf(" **>> FurtherMore, here is a blanking out of all *pXtX \n");
    for (int iiti = OldKappaMem; iiti < OnKappaMem; iiti++) {
      for (int jjti = 0; jjti < p; jjti++) {
        *(this->pXtX[iiti] + jjti) = 0.0;
      }
    }
    for (int iiti = OldKappaMem; iiti < OnKappaMem; iiti++) {
      if (this->pXtX[iiti] ==NULL) {
        Rprintf(" **>> Resize check! Bad Blank Out pXtX error iiti = !\n", iiti); R_FlushConsole();
        Rf_error("  **>> Do Not Pass go");
      }
    }
    Rprintf(" **>> Blanked out pXtX, this should have worked!\n"); R_FlushConsole();
  }
  FFree(kXFinder, "kXFinder");
  kXFinder = NewkXFinder;

  //FullCrazyTestiWeightedXtX("Resize End");
  if (iWeightedXtX != NULL && NewWeightedXtX != NULL) {
    if (Verbose >= 3) {
      Rprintf("About to Free iWeightedXtX \n"); R_FlushConsole();
    }
    FFree(iWeightedXtX, "iWeightedXtX");
    iWeightedXtX = NewWeightedXtX;  NewWeightedXtX = NULL;
  }
  if (Verbose >= 3) {
    Rprintf("Successfully Freed iWeightedXtX\n"); R_FlushConsole();
  }
  //CheckkXToXLC((char*) "Resize Start");
  //CheckkXFinderError((char*)"Blank out pXtX");
  //CountBad = QuickCheckWeightXtX();
  //if (CountBad > 0) {
  //  Rprintf("*** Resize(Factor=%f) Error at end, tt=%d, we got QuickCheckWeightXtX, CountBad = %d \n", Factor, tt, CountBad); R_FlushConsole();
  //  Rf_error("***  Resize QuickCheckWeightXtX Fail. \n");
  //}
  if (Verbose >= 0 && p >= 200000) {
    CheckResizeIntegrity();
    //Rprintf(" **>> After CheckResizeIntegrity, we launch CheckpXtX()!\n");
    //CheckpXtX();
  }

  //Rprintf("** BayesSpikeCpp.cpp:::Resize(tt=%d); OnKappaMem=%d/OldKappaMem=%d, OnKappaS=%d, Now check pXtX \n", 
  //  tt, OnKappaMem, OldKappaMem, OnKappaS); R_FlushConsole();
  //CheckpXtX();
  if (Verbose > 1 || MaxLargeSizeSquares != OnKappaMem) {
    Rprintf("** BayesSpikeCpp.cpp: Resize(tt=%d):  Now, complete, Factor=%f, OnKappaMem=%d, \n",
      tt, Factor, OnKappaMem);
    Rprintf("**       MaxLargeSizeSquares=%d, MaxSquaresAllocation=%d, NumActive=%d, OnKappaS=%d.\n",
       (int) MaxLargeSizeSquares, (int) MaxSquaresAllocation, (int) NumActive, (int) OnKappaS); 
    Rprintf("**       We exit with a success. \n"); R_FlushConsole();  
    Rprintf("*********************************************************************\n");
    R_FlushConsole();
  } 
  return(1);
}

//////////////////////////////////////////////////////////////////////////////////
// Check integrity of pXtX storage of squared covaraince for active coefficients.
int BayesSpikeCL::CheckpXtX() {
  double *Goes = NULL;
  if (OnKappaS <= 0) {
    return(1);
  }
  int CountBad = 0;
  Goes = Calloc(p, double);
  double dTotal;
  if (Goes == NULL) {
    Rf_error("CheckpXtX, we fail to allocate Check Vector\n");
  }
  char Trans = 'T'; double OneD = 1.0;  double ZeroD = 0.0;
  int One = 1;
  double MaxBad = 0.0;
  Rprintf("**>> BayesSpikeCpp.cpp::CheckpXtX(tt=%d/%d) launch. \n", tt, MaxGibbsIters); R_FlushConsole();
  if (iiWeight == 0) {
  for (int ii = 0; ii < OnKappaS; ii++) {
    F77_CALL(dgemv)(&Trans, &n, &p, &OneD, REAL(sX), &n, 
      REAL(sX) + n * kXFinder[ii], &One,
      &ZeroD, Goes, &One);
    dTotal = 0.0;
    for (int jj = 0; jj < p; jj++) {
      dTotal += fabs(Goes[jj] - *(pXtX[ii]+jj));
    }
    if (dTotal > p * .00001) {
       Rprintf("**>> BayesSpikeCpp.cpp::CheckpXtX(): Nope, not good here dTotal=%f, ii=%d, kXFinder[%d]=%d \n",
         dTotal, ii, ii, kXFinder[ii]); R_FlushConsole();
       CountBad++;
    }
    if (MaxBad < dTotal) { MaxBad = dTotal; }
  }
  } else {
    if (WW == NULL) {
      RMemGetD(WW, "WW", NULL);
    }
  for (int ii = 0; ii < OnKappaS; ii++) {
    if (iWeightedXtX[ii] == 0) {
      ReweightedOneXtX(kXFinder[ii]);
    }
    for (int iti = 0; iti < n; iti++) {
      WW[iti] = iiWeight[iti] * REAL(sX)[iti + n * kXFinder[ii]];
    }
    F77_CALL(dgemv)(&Trans, &n, &p, &OneD, REAL(sX), &n, 
      WW, &One,
      &ZeroD, Goes, &One);
    dTotal = 0.0;
    for (int jj = 0; jj < p; jj++) {
      dTotal += fabs(Goes[jj] - *(pXtX[ii]+jj));
    }
    if (dTotal > p * .00001) {
       Rprintf("**>> BayesSpikeCpp.cpp::CheckpXtX(): Nope, not good here dTotal=%f, ii=%d, kXFinder[%d]=%d \n",
         dTotal, ii, ii, kXFinder[ii]); R_FlushConsole();
       CountBad++;
    }
    if (MaxBad < dTotal) { MaxBad = dTotal; }
  }  
  }
  if (CountBad >= 1) {
    Free(Goes); Goes = NULL;
    Rprintf("ERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERROR");
    Rprintf("ERROR: BayesSpikeCpp.cpp::CheckpXtX, CountBad = %d, we fail.\n", CountBad);
    Rprintf("ERROR: MaxBad = %f \n", MaxBad); R_FlushConsole();
    Rf_error("BayesSpikeCpp.cpp::CheckpXtX: CountBad = %d, and we fail. \n", CountBad);
  }
  Rprintf("**>> BayesSpikeCpp.cpp::CheckpXtX(tt=%d/%d): we have a pass, CountBad=%d.  Now for Hard Check.\n", tt, MaxGibbsIters, CountBad); R_FlushConsole();
  Free(Goes); Goes = NULL;
  double *ReplaceXtX = NULL;
  One = 1;
  for (int iti = 0; iti < OnKappaMem; iti++) {
    ReplaceXtX = Calloc(p, double);
    F77_CALL(dcopy)(&p, pXtX[iti], &One, ReplaceXtX, &One);
    Free(pXtX[iti]);  pXtX[iti] = ReplaceXtX;
  }
  Rprintf("**>> BayesSpikeCpp.cpp::CheckpXtX(tt=%d/%d): we have a hard check pass, CountBad=%d.  Now for Hard Check.\n",tt, MaxGibbsIters, CountBad); R_FlushConsole();

  return(1);
}

////////////////////////////////////////////////////////////////////////////////
// Check that on resizing of pXtX, kXFinder, etc that memory is intact and that
//  values that should sum to each other (even with weightw Wi) are still stable.
int BayesSpikeCL::CheckResizeIntegrity() {
  Rprintf("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
  Rprintf("<<< CheckResizeIntegrity(tt=%d): About to do.\n", tt); R_FlushConsole();
  if (Rf_length(sOnPiA) == 2) {
    Rprintf("<<< OnPiA=%f,%f, and OnKappaS/p = %f, is this good?\n",
      REAL(sOnPiA)[0], REAL(sOnPiA)[1], ((double)OnKappaS) / ((double) p) );
    R_FlushConsole();
  } else {
    Rprintf("<<< OnPiA=%f, and OnKappaS/p = %f, is this good?\n",
      REAL(sOnPiA)[1], ((double)OnKappaS) / ((double) p) );
    R_FlushConsole();  
  }
  Rprintf("<<< Checking Order of kXFinder. \n");    R_FlushConsole();
  CheckkXFinderError((char*)"CheckResizeIntegrity");
  double Checks;  int Total;  int ii, jj;
  Rprintf("<<< Check smXtY, length OnKappaMem = %d\n", OnKappaMem);
  R_FlushConsole();
  for (ii = 0; ii < OnKappaMem; ii++) {
    smXtY[ii] = 0.0;
  }
  Rprintf("<<< CheckResizeIntegrity, we smXtY set to all zero \n"); R_FlushConsole();
  Rprintf("<<< CheckResizeIntegrity, Now configure cXtY \n"); R_FlushConsole();
  Checks = 0.0;
  for (ii = 0; ii < OnKappaMem; ii++) {
    Checks += cXtY[ii];
  }
  Rprintf("<<< cXtY totals to %f \n", cXtY); R_FlushConsole();
  Rprintf("<<< Check pXtX up to OnKappaS=%d/%d \n", OnKappaS, OnKappaMem);
  Checks = 0.0;
  for (ii = 0; ii < OnKappaS; ii++) {
    if (pXtX[ii] == NULL) {
      Rprintf("<<< Pre OnKappaS=%d/%d, pXtX[ii=%d] integrity has failed!\n", OnKappaS, OnKappaMem, ii); R_FlushConsole();
      Rprintf("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>> FAIL!\n");   R_FlushConsole();
      Rf_error("  Error in pXtX[ii=%d] being NULL!\n", ii);
    }
    for (jj = 0; jj < p; jj++) {
      Checks += *(pXtX[ii] + jj);
    }
    if (pXtX[ii] == NULL) {
      Rprintf("<<< Pre WOAH AFTER OnKappaS=%d/%d, pXtX[ii=%d] integrity has failed!\n", OnKappaS, OnKappaMem, ii); R_FlushConsole();
      Rprintf("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>> FAIL!\n");   R_FlushConsole();
      Rf_error("  Error in pXtX[ii=%d] being NULL!\n", ii);
    }
  }
  Rprintf("<<< Onto OnKappaS pXtX Integrity sums to %f \n", Checks); R_FlushConsole();
  for (ii = OnKappaS; ii < OnKappaMem; ii++) {
    if (pXtX[ii] == NULL) {
      Rprintf("<<< Post OnKappaS=%d/%d, pXtX[ii=%d] integrity fails!\n", OnKappaS, OnKappaMem, ii); R_FlushConsole();
      Rprintf("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>> FAIL!\n");   R_FlushConsole();
      Rf_error("  Error in pXtX[ii=%d] being NULL!\n", ii);
    }
    for (jj = 0; jj < p; jj++) {
      *(pXtX[ii] + jj) = 0.0;
    }
    if (pXtX[ii] == NULL) {
      Rprintf("<<< Post OnKappaS=%d/%d (WOAH AFTER), pXtX[ii=%d] integrity fails!\n", OnKappaS, OnKappaMem, ii); R_FlushConsole();
      Rprintf("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>> FAIL!\n");   R_FlushConsole();
      Rf_error("  Error in pXtX[ii=%d] being NULL!\n", ii);
    }
  }
  Rprintf("<<< pXtX is summed back to zero for rest integrity \n"); R_FlushConsole();
  Rprintf("<<< Turn PropBeta to zero. \n"); R_FlushConsole();
  for (ii = 0; ii < OnKappaMem; ii++) {
    PropBeta[ii] = 0.0;
  }
  Rprintf("<<< PropBeta Turned to Zero \n", cXtY); R_FlushConsole();
  Rprintf("<<< Zero Check rIChol.  \n"); R_FlushConsole();
  Total = MaxLargeSizeSquares*(MaxLargeSizeSquares+1)/2;
  for (ii = 0; ii < Total; ii++) {
    rIChol[ii] = 0.0;
  }
  Rprintf("<<< rIChol checks out. \n"); R_FlushConsole();
  Rprintf("<<< Zero Check rI.  \n"); R_FlushConsole();
  Total = MaxLargeSizeSquares*(MaxLargeSizeSquares+1)/2;
  for (ii = 0; ii < Total; ii++) {
    rI[ii] = 0.0;
  }
  Rprintf("<<< rI checks out. \n"); R_FlushConsole();
  Rprintf("<<< Zero Check rSmallXtX.  \n"); R_FlushConsole();
  Total = MaxLargeSizeSquares*(MaxLargeSizeSquares+1)/2;
  for (ii = 0; ii < Total; ii++) {
    rSmallXtX[ii] = 0.0;
  }
  Rprintf("<<< rSmallXtX checks out. \n"); R_FlushConsole();  
  if (rICholBack != NULL) {
  Rprintf("<<< Zero Check rICholBack.  \n"); R_FlushConsole();
  Total = MaxLargeSizeSquares*(MaxLargeSizeSquares+1)/2;
  for (ii = 0; ii < Total; ii++) {
    rICholBack[ii] = 0.0;
  }
  Rprintf("<<< rICholBack checks out. \n"); R_FlushConsole();    
  }
  Rprintf("<<< CheckOrderedActive. \n"); R_FlushConsole();
  for (ii = 0; ii < OnKappaMem; ii++) {
    OrderedActive[ii] = -1;
  }
  Rprintf("<<< OrderedActive Checks Out. \n"); R_FlushConsole();
  Rprintf("<<< Checking kXFinder with OnKappaS=%d, OnKappaMem=%d\n",
    OnKappaS, OnKappaMem); R_FlushConsole();
  for (ii = 0; ii < OnKappaS; ii++) {
    if (ii < 0 || ii >= OnKappaS) {
      Rprintf("<<< This doesn't make sense, what have I done to ii=%d, OnKappaS=%d? \n", ii, OnKappaS);
      R_FlushConsole();
    } else if (kXFinder[ii] < 0 || kXFinder[ii] >= p) {
      Rprintf("<<<  On Check, kXFinder, strange ii=%d produces %d, OnKappaS=%d\n", ii, kXFinder[ii], OnKappaS); R_FlushConsole();
    }
  }
  double d0ii0 = 0.0;
  int One = 1;
  if (OnKappaS >= 1) {
  if (kXFinder[0] != 0) {
    if (iiWeight == NULL) {
    d0ii0 = F77_CALL(ddot)(&n, REAL(sX), &One, REAL(sX) + n * kXFinder[0], &One);
    } else {
      if (iWeightedXtX != NULL && iWeightedXtX[0] == 0) {
        ReweightedOneXtX(kXFinder[0]);
      }
      if (WW == NULL) { RMemGetD(WW, "WW", n);}
      for (int iti = 0; iti < n; iti++) {
        WW[iti] = REAL(sX)[iti + n * kXFinder[0]] * iiWeight[iti]; 
      }
      d0ii0 = F77_CALL(ddot)(&n, REAL(sX), &One, WW, &One);      
    }
  
  if (fabs(d0ii0 - *(pXtX[0] + 0)) > .0001) {
    Rprintf("<<<  Oooh, bad OnCheck, kXFinder[0] = %d, but *(pXtX[0]+0) = %f and sX[,0] * sX[,%d] = %f\n",
      kXFinder[0], *(pXtX[0] + 0), kXFinder[0], d0ii0); R_FlushConsole();
    Rf_error("<<< Bad Check Of pXtX!\n");
  } else {
    Rprintf("<<<  Successful OnCheck, kXFinder[0] = %d, and *(pXtX[0]+0) = %f and sX[,0] * sX[,%d] = %f\n",
      kXFinder[0], *(pXtX[0] + 0), kXFinder[0], d0ii0); R_FlushConsole();
  }
  double d0iiOnKappaSm1 = 0.0;
    if (OnKappaS > 0) {
    if (iiWeight == NULL) {
      d0iiOnKappaSm1 = F77_CALL(ddot)(&n, REAL(sX), &One, REAL(sX) + n * kXFinder[OnKappaS-1], &One);
    } else {
      if (OnKappaS > 0 && iWeightedXtX != NULL && iWeightedXtX[OnKappaS-1] == 0) {
        ReweightedOneXtX(kXFinder[OnKappaS-1]);
      }
      if (WW == NULL) { RMemGetD(WW, "WW", n);}
      for (int iti = 0; iti < n; iti++) {
        WW[iti] = REAL(sX)[iti + n * kXFinder[OnKappaS-1]] * iiWeight[iti]; 
      }
      d0iiOnKappaSm1 = F77_CALL(ddot)(&n, REAL(sX), &One, WW, &One);      
    }
    if (fabs(d0iiOnKappaSm1 - *(pXtX[OnKappaS-1] + 0)) > .0001) {
      Rprintf("<<<  Oooh, bad OnCheck, kXFinder[%d-1=%d] = %d, but *(pXtX[%d-1=%d]+0) = %f and sX[,0] * sX[,%d] = %f\n",
       OnKappaS, OnKappaS-1, kXFinder[0], OnKappaS, OnKappaS-1, *(pXtX[OnKappaS-1] + 0), kXFinder[OnKappaS-1], d0iiOnKappaSm1); R_FlushConsole();
     Rf_error("<<< Bad Check Of pXtX!\n");
    } else {
      Rprintf("<<<  Oooh, Successful OnCheck, kXFinder[%d-1=%d] = %d, nad *(pXtX[%d-1=%d]+0) = %f and sX[,0] * sX[,%d] = %f\n",
        OnKappaS, OnKappaS-1, kXFinder[0], OnKappaS, OnKappaS-1, *(pXtX[OnKappaS-1] + 0), kXFinder[OnKappaS-1], d0iiOnKappaSm1); R_FlushConsole();
    }
    }
  } else if (OnKappaS > 0 && kXFinder[0] == 0) {
    d0ii0 = 0.0; 
    if (iiWeight == NULL) {
    for (int iti = 0; iti < n; iti++) {
      d0ii0 += REAL(sX)[iti] * REAL(sX)[iti];
    }
    } else {
      if (iWeightedXtX  != NULL && iWeightedXtX[0] == 0) {
        ReweightedOneXtX(0);
      }
      for (int iti = 0; iti < n; iti++) {
        d0ii0 += REAL(sX)[iti] * REAL(sX)[iti]*iiWeight[iti];
      }      
    }
    if (fabs(d0ii0 - *(pXtX[0] + 0)) > .0001) {
      Rprintf("<<<  Oooh, bad OnCheck, kXFinder[0] = %d, but *(pXtX[0]+0) = %f and sX[,0] * sX[,%d] = %f\n",
        kXFinder[0], *(pXtX[0] + 0), kXFinder[0], d0ii0); R_FlushConsole();
      Rf_error("<<< Bad Check Of pXtX!\n");
    } else {
      Rprintf("<<<  Successful OnCheck, kXFinder[0] = %d, and *(pXtX[0]+0) = %f and sX[,0] * sX[,%d] = %f\n",
        kXFinder[0], *(pXtX[0] + 0), kXFinder[0], d0ii0); R_FlushConsole();
    }
  }
  }
  Rprintf("<<< On ii=%d, OnKappaS=%d, kXFinder Old checks out \n", ii, OnKappaS);
  for (ii = OnKappaS; ii < OnKappaMem; ii++) {
    kXFinder[ii] = -1;
  }
  Rprintf("<<< kXFinder All Checks Out. \n"); R_FlushConsole();  
  Total = 0;
  int AnError = 0;
  Rprintf("<<< CheckOut XLC\n"); R_FlushConsole();
  for (ii = 0; ii < p; ii++) {
    if (XLC[ii] < 0) {
    } else if (XLC[ii] >= 0 && XLC[ii] < OnKappaS) {
      if (kXFinder[XLC[ii]] != ii) {
        Rprintf("<<< Hey, XLC[ii=%d] and kXFinder[%d] = %d, not good!\n",
          ii, kXFinder[XLC[ii]]); R_FlushConsole();
        AnError++;
      } else {
        Total++;
      }
    } else {
      Rprintf("<<< Hey, OnKappaS=%d, but XLC[ii=%d] is %d!\n",
        OnKappaS, ii, XLC[ii]);
      AnError++;
    }
  }
  if (AnError > 0) {
    Rprintf("XLC Check produced %d errors, revise and go back!\n", AnError);
    Rf_error("XLC: Not good on Integrity Check\n");
  }
  Rprintf("<<< tt=%d, Pass Integrity Check OnKappaS=%d/%d \n", tt, OnKappaS, OnKappaMem);
  return(1); 
}


////////////////////////////////////////////////////////////////////////////////
//    FillsBetaFromPropBetaAndCompute()
//
//       PropBeta, which is only of length Active coefficients, is stuffed into
//   a vector to replace all p active coefficients.  We don't normally need
//   to do multiplications from the Beta vector, but this is helpful for XtResid
//   update and T noise update.
int BayesSpikeCL::FillsBetaFromPropBetaAndCompute() {
  int ii;
  if (NumActive > p) { 
    Rf_error("FillsBetaFromPropBetAndCompute: NumActive is Bad! = %d\n",
      NumActive, p);
  }
  
  //if (tt <= 1) {
  //  Rprintf("** BayesSpikeCpp.cpp:::FillsBetaFromPropBetaAndCompute(tt=%d):  Note Testing TestOrderedActive at the beginning!\n", tt); R_FlushConsole();
  //}
  //int ACount = TestOrderedActive();
  //if (ACount >= 1) {
  // Rprintf("** ERRROR IN TestOrderedActive beginning FillsBetaFromPropBetaAndCompute. ACount = %d, tt=%d\n", ACount, tt); R_FlushConsole();
  //  Rf_error("** Go Catch this error!\n");
  //}
  
  
  if (NumActive <= 0) {
    return(1);
  }
  for (ii = 0; ii < NumActive; ii++) {
    if ( R_isnancpp(PropBeta[ii]) || PropBeta[ii] != PropBeta[ii] )  {
      PropBeta[ii] = 0; }
    //if (fabs(PropBeta[ii]) > 5000) {
    //  Rprintf("Ooh, error on PropBeta[%d] = %f\n", ii, PropBeta[ii]);
    //  Rf_error("Go Look for the error");
    //}
    REAL(sBeta)[OrderedActive[ii]] = PropBeta[ii];
  } 
  if (dfRobit >= 0.0) {
    RobitReplace();
  } else if (dfTNoise >= 0.0) {
    UpdateTNoise();
  } else {
    UpdateFreshXtResid();
  }
  return(1);
}

////////////////////////////////////////////////////////////////////////////
//  GiveItAChanceAndKillUselessCoeficients()
//
//     Tries to kill off a bunch of coefficients that are too small to matter,
//  and/or which might have only been present in the model on a rare
//  occasion, many iterations ago.
//
int BayesSpikeCL::GiveItAChanceAndKillUselessCoeficients(int PleaseKill) {
  int *RankMyUselessCoefficients = NULL;
  if (PleaseKill > OnKappaS) {
    PleaseKill = OnKappaS;
  }
  ItersSinceRemoval = 0;
  double *MIPCoefficients = NULL;
  if (Verbose >= 0) {
    Rprintf("--------------------------------------------------------------------------------------\n");
    Rprintf("--- BayesSpikeCpp.cpp:: GiveItAChanceAndKillUselessCoeficients(tt=%d/%d): We're going to have\n", tt, MaxGibbsIters);
    Rprintf("--- to kill coefficients since OnKappaMem = %d/%d.\n", OnKappaMem, p); 
    Rprintf("---  You gave us PleaseKill = %d, KillPct=%.3f.\n", 
      PleaseKill, KillPct); R_FlushConsole();
  }
  MIPCoefficients = Calloc(OnKappaS+1, double);
  //RankMyUselessCoefficicents = Calloc(OnKappaS, int);
  if (MIPCoefficients == NULL) {
    Rprintf("--- GiveItAChanceAndKillUselessCoefficients(): couldn't allocate \n");
    Rprintf("--- OnKappaS=%d, p=%d MIPCoeficients, error!!!! \n", OnKappaS, p);
    Rf_error("--- BayesSpikeCpp: GiveItAChanceAndKillUselessCoefficients(): Allocation Error!\n");
  }
  //if (RankMyUselessCoefficicents == NULL) {
  //  Rprintf("GiveItAChanceAndKillUselessCoefficients: couldn't allocate OnKappaS =%d, p=%d RankMyUselessCoefficicents!\n", OnKappaS, p);
  //  FFree(MIPCoefficients, "MIPCoefficients");
  //   Rf_error("Allocation Error!\n");
  //}
  for (int iti = 0; iti < OnKappaS; iti++) {
    MIPCoefficients[iti] = 0.0;
    //RankMyUselessCoefficients[iti] = iti;
  }
  int* GroupsOfMine  = NULL; 
  if (RunProbVector != NULL && TotEveryProbVector != NULL) {
    GroupsOfMine = Calloc(OnKappaS+1, int);
    if (Verbose>= 3) {
      Rprintf("--- Hey, filling GroupsOfMine! iFirstRandom =%d\n", iFirstRandom); R_FlushConsole();
      if (Rf_length(tauEndList) >= 5) {
      Rprintf("--- First EndGroups are %d, %d, %d, %d, %d\n",
        ANINT(tauEndList,0), ANINT(tauEndList,1), ANINT(tauEndList,2),
        ANINT(tauEndList,3), ANINT(tauEndList,4));
      }
    }
    int CountOnjj;  int Stjj;
    for (int iti = 0; iti < OnKappaS; iti++) {
      if (tauEndList == NULL || Rf_isNull(tauEndList) || Rf_length(tauEndList) == 0 ||
        iFirstRandom < 0 || iFirstRandom >= p || kXFinder[iti] < iFirstRandom) {
        GroupsOfMine[iti] = kXFinder[iti];
      }  else {
       if (iti >= 1 && kXFinder[iti] > iFirstRandom && 
         kXFinder[iti] > kXFinder[iti-1] && GroupsOfMine[iti-1] >= iFirstRandom) {
         Stjj = GroupsOfMine[iti-1] - iFirstRandom;
       } else {
         Stjj = 0;
       }
       if (Stjj >= Rf_length(tauEndList)) {
         Rf_error("--- Error in fill GroupsOfMine, Stjj = %d>%d, what?", Stjj,
           Rf_length(tauEndList)); R_FlushConsole();
       }
       for (CountOnjj = Stjj; CountOnjj < Rf_length(tauEndList); CountOnjj++) {
         if (CountOnjj < Rf_length(tauEndList) && 
           ANINT(tauEndList, CountOnjj) >= kXFinder[iti]) { 
             //if (kXFinder[iti] == 10 || kXFinder[iti] == 14) {
             //  Rprintf(" We just got to kXFinder[%d] = %d, CountOnjj = %d, Stjj = %d, end is %d\n",
             //    iti, kXFinder[iti], CountOnjj, Stjj, 
             //    (int) ANINT(tauEndList, CountOnjj)); R_FlushConsole();
             //}
           GroupsOfMine[iti] = CountOnjj+iFirstRandom;
           CountOnjj = Rf_length(tauEndList)+5;
           break;
         }
       }
      }
    }
    if (Verbose >= 3) {
      Rprintf("--- GroupsOfMine is filled, lets see it now!\n---- "); R_FlushConsole();
      int intwenty = 0;
      int MaxGroupsOfMine = 0;
      for (int iti = 0; iti < OnKappaS; iti++) {
        if (MaxGroupsOfMine < GroupsOfMine[iti]) {
          MaxGroupsOfMine = GroupsOfMine[iti];
        }
        if (iti == OnKappaS-1) {
          Rprintf("%d:%d\n", kXFinder[iti], GroupsOfMine[iti]);  R_FlushConsole();
        } else {
          Rprintf("%d:%d,  ", kXFinder[iti], GroupsOfMine[iti]);
          if (intwenty >= 20) {
            Rprintf("\n---- "); R_FlushConsole();   intwenty=0;
          }
        }
        intwenty++;
      }
      Rprintf("--- About to fill MIPCoefficients! MaxGroupsOfMine = %d\n", 
        MaxGroupsOfMine); R_FlushConsole();
    }
    for (int iti = 0; iti < OnKappaS; iti++) {
      if (kXFinder[iti] >= p || 
        (tauEndList != NULL && !Rf_isNull(tauEndList) &&
          Rf_length(tauEndList) >= 0 &&  iFirstRandom >= 0 &&
          GroupsOfMine[iti] >= iFirstRandom + Rf_length(tauEndList))) {
        Rprintf("---  ERROR-ERROR-ERROR BayesSpikeCpp.cpp:: GiveItAChanceAndKillUselessCoeficients() \n");
        Rprintf("--- An issue for kXFinder[iti=%d], iti=%d, \n", iti,
          kXFinder[iti], iFirstRandom);
        if (tauEndList == NULL || Rf_isNull(tauEndList)) {
          Rprintf("---  And NULL tauEndList, iFirsRandom = %d, GroupsOfMine[iti=%d]=%d\n",
            Rf_length(tauEndList), iFirstRandom, iti, 
            GroupsOfMine[iti]); R_FlushConsole();
        } else if (Rf_length(tauEndList) == 0) {
          Rprintf("---  And Zero length tauEndList=%d, iFirstRandom=%d, GroupsOfMine[iti=%d]=%d\n",
            Rf_length(tauEndList), iFirstRandom, iti, 
            GroupsOfMine[iti]); R_FlushConsole();        
        } else {
          Rprintf("---  And length tauEndList=%d, iFirstRandom=%d, GroupsOfMine[iti=%d]=%d\n",
            Rf_length(tauEndList), iFirstRandom, iti, 
            GroupsOfMine[iti]); R_FlushConsole();        
        }
        Rf_error("---- What happened here!\n");
      }
      //if (Verbose >= 1) {
      //  Rprintf(" %d, ", iti); R_FlushConsole();
      //  if ( iti > 0 && ((int)iti / (int) 20)  * 20 == iti) {
      //    Rprintf("\n  "); R_FlushConsole();
      //  }
      //}
      if (iti < OnKappaS) {
        MIPCoefficients[iti] = ((double) RunProbVector[GroupsOfMine[iti]]) / 
         ((double)TotEveryProbVector[GroupsOfMine[iti]]);
        if (kXFinder[iti] >= 0 && kXFinder[iti] < this->p && REAL(sBeta)[kXFinder[iti]] != 0.0) {
          MIPCoefficients[iti] = 1.0;
        }
      }
    }
    if (Verbose >= 0) {
      Rprintf("--- Finished Filling MIPCoefficients but about ");
      Rprintf("going to quit and free GroupsOfMine. \n"); R_FlushConsole();
    }
  } else {
    double Maxie = 0.0;
    for (int iti = 0; iti < OnKappaS; iti++) {
      if (REAL(sBeta)[kXFinder[iti]] != 0.0) {
        MIPCoefficients[iti] = -1.00;
      } else {
        MIPCoefficients[iti] = fabs(XtResid[kXFinder[iti]] + XjSq[kXFinder[iti]] * REAL(sBeta)[kXFinder[iti]]);
        if (MIPCoefficients[iti] > Maxie) { Maxie = MIPCoefficients[iti]; }
      }
    }
    for (int iti = 0; iti < OnKappaS; iti++) {
      if (MIPCoefficients[iti] < 0) { MIPCoefficients[iti] = Maxie+1.0; }
    }
  }
  if (Verbose >= 0) {
    Rprintf("--- HeapSortAll: about to sort MIPCoefficients\n"); R_FlushConsole();
  }
  RankMyUselessCoefficients = HeapSortAll(OnKappaS, MIPCoefficients);
  int LeftRight = 0;
  if (Verbose >= 0) {
    Rprintf("--- Now We're looking at MIPCoefficients as we have sorted!\n"); R_FlushConsole();
  }
  if (MIPCoefficients[RankMyUselessCoefficients[0]] < MIPCoefficients[RankMyUselessCoefficients[OnKappaS-1]]) {
    LeftRight = 1;
    if (Verbose >= 2) {
      Rprintf("--- Oops: LeftRight = 1, MIPCoefficients[Rank[0] = %d] = %.4f, and MIP[Rank[OnKappaS]=%d] = %.4f\n",
        RankMyUselessCoefficients[0], MIPCoefficients[RankMyUselessCoefficients[0]], 
        RankMyUselessCoefficients[OnKappaS-1], 
        MIPCoefficients[RankMyUselessCoefficients[OnKappaS-1]]);
    }
  }
  if (Verbose >= 2) {
    Rprintf("--- Here they are in order of sort!  LeftRight = %d\n", LeftRight);
    int intwenty = 0;
    int MinRank = RankMyUselessCoefficients[0];
    int MaxRank = RankMyUselessCoefficients[0];    
    for (int iti = 0; iti < OnKappaS; iti++) {
       if (RankMyUselessCoefficients[iti] < MinRank) {
         MinRank = RankMyUselessCoefficients[iti];
       }
       if (RankMyUselessCoefficients[iti] > MaxRank) {
         MaxRank = RankMyUselessCoefficients[iti];
       }
       if (iti == OnKappaS-1) {
         if (MIPCoefficients[RankMyUselessCoefficients[iti]] > .0001) {
          Rprintf("%d:%d:MIP=%.4f:Bt=%.2f\n--- ", iti, kXFinder[RankMyUselessCoefficients[iti]],
            MIPCoefficients[RankMyUselessCoefficients[iti]],
            REAL(sBeta)[kXFinder[RankMyUselessCoefficients[iti]]]); R_FlushConsole();
         } else {
          Rprintf("%d:%d:MIP=%.2e:Bt=%.2f\n--- ", iti, kXFinder[RankMyUselessCoefficients[iti]],
            MIPCoefficients[RankMyUselessCoefficients[iti]],
            REAL(sBeta)[kXFinder[RankMyUselessCoefficients[iti]]]); R_FlushConsole();         
         }
       } else {
        if (MIPCoefficients[RankMyUselessCoefficients[iti]] > .0001) {
           Rprintf("%d:%d:MIP=%.4f:Bt=%.2f,", iti, kXFinder[RankMyUselessCoefficients[iti]],
             MIPCoefficients[RankMyUselessCoefficients[iti]],
             REAL(sBeta)[kXFinder[RankMyUselessCoefficients[iti]]]);
        } else {
           Rprintf("%d:%d:MIP=%.2e:Bt=%.2f,", iti, kXFinder[RankMyUselessCoefficients[iti]],
             MIPCoefficients[RankMyUselessCoefficients[iti]],
             REAL(sBeta)[kXFinder[RankMyUselessCoefficients[iti]]]);        
        }
         if (intwenty == 6) {
           Rprintf("\n--- "); R_FlushConsole();
           intwenty = 0;
         }
       }
       intwenty++;
    }
    Rprintf("--- MinRank = %d, MaxRank = %d \n", MinRank, MaxRank);
    if (MinRank != 0) {
      Rprintf("--- Error Issue, MinRank = %d, bad sort? \n", MinRank);  R_FlushConsole();
    }
    if (MaxRank != OnKappaS-1) {
      Rprintf("--- Error Issue MaxRank = %d, bad sort? \n", MaxRank); R_FlushConsole();
    }
  }
  int FindPivot = 0;
  if (RunProbVector != NULL) {
    if (Verbose >= 2) {
      Rprintf("--- RunProbVector != NULL, looking for Pivot if Leftright = %d\n", LeftRight); R_FlushConsole();
    }
    if (LeftRight == 1) {
      if (Verbose >= 0) {
        Rprintf("--- GiveAChance to Delete, we are going to try to delete low probability members with MIP less than %f\n",
          MinimumProbKeeper);
      }
      for (FindPivot = 0; FindPivot < OnKappaS; FindPivot++) {
        if (REAL(sBeta)[kXFinder[RankMyUselessCoefficients[FindPivot]]] != 0.0 || 
          (PleaseKill > 0 && FindPivot > PleaseKill &&
           MIPCoefficients[RankMyUselessCoefficients[FindPivot]] > MinimumProbKeeper) ||
          (PleaseKill < 0 && MIPCoefficients[RankMyUselessCoefficients[FindPivot]] > MinimumProbKeeper)) {
          break;
        }
      }
      if (Verbose >= 0) {
        if (FindPivot > OnKappaS) { FindPivot = OnKappaS-1; }
        Rprintf("--- GiveAChanceToDelete() we found FindPivot=%d/%d had Splitting MIP\n", FindPivot, OnKappaS);
        Rprintf("--- Here's a slight sample of smallest: \n--- ");
        int OgoMin = 5;  if (FindPivot < 5) { OgoMin = FindPivot; }
        for (int ii = 0; ii < OgoMin; ii++) {
          if (ii == OgoMin-1) {
            if (MIPCoefficients[RankMyUselessCoefficients[ii]] > .0001) {
              Rprintf("%d:%d:MIP=%.4f:Bt=%.2f\n", ii, RankMyUselessCoefficients[ii], 
                MIPCoefficients[RankMyUselessCoefficients[ii]],
                REAL(sBeta)[kXFinder[RankMyUselessCoefficients[ii]]] ); R_FlushConsole();
            } else {
              Rprintf("%d:%d:MIP=%.2e:Bt=%.2f\n", ii, RankMyUselessCoefficients[ii], 
                MIPCoefficients[RankMyUselessCoefficients[ii]],
                REAL(sBeta)[kXFinder[RankMyUselessCoefficients[ii]]] ); R_FlushConsole();
            }
          } else {
            if (MIPCoefficients[RankMyUselessCoefficients[ii]] > .0001) {
              Rprintf("%d:%d:MIP=%.2f:Bt=%.2f, ", ii, RankMyUselessCoefficients[ii], 
              MIPCoefficients[RankMyUselessCoefficients[ii]],
              REAL(sBeta)[kXFinder[RankMyUselessCoefficients[ii]]]); R_FlushConsole();
            } else {
              Rprintf("%d:%d:MIP=%.2e:Bt=%.2f, ", ii, RankMyUselessCoefficients[ii], 
              MIPCoefficients[RankMyUselessCoefficients[ii]],
              REAL(sBeta)[kXFinder[RankMyUselessCoefficients[ii]]]); R_FlushConsole();            
            }
          }
        }
        double CountPct = 0.0;
        for (int ii = 0; ii < FindPivot; ii++) {
          CountPct += MIPCoefficients[RankMyUselessCoefficients[ii]];
        }
        Rprintf("--- Rank[FindPivot-1 = %d-1]=%d and its MIP =%.4f, Bt=%.2f, Rank[FindPivot = %d]=%d and its MIP =%.4f, Bt=%.2f\n",
          FindPivot, RankMyUselessCoefficients[FindPivot-1], MIPCoefficients[RankMyUselessCoefficients[FindPivot-1]],
          REAL(sBeta)[RankMyUselessCoefficients[FindPivot-1]],
          FindPivot, RankMyUselessCoefficients[FindPivot], MIPCoefficients[RankMyUselessCoefficients[FindPivot]],
          REAL(sBeta)[RankMyUselessCoefficients[FindPivot-1]]
          ); 
        R_FlushConsole();
        Rprintf("--- The average inclusion of these %d MIP are now %f\n", 
          FindPivot, CountPct/FindPivot); R_FlushConsole();
        Rprintf("--- The maximum of MIPCoefficients is at %d with %f\n",
          RankMyUselessCoefficients[OnKappaS-1], MIPCoefficients[RankMyUselessCoefficients[OnKappaS-1]]); R_FlushConsole();
        if (CountPct / FindPivot < KillPct/2) {
         Rprintf("--- We'll try to preserve some but why are there so many small probability coefficients? KillPct=%f\n",
           KillPct);
         R_FlushConsole();
        }
      }
      if (PleaseKill > 0 && FindPivot < PleaseKill) {
        Rprintf("------------------------------------------------------------------------------------\n");
        Rprintf("--- tt=%d, MIP Based elimination, too many active coefficients to nuke PleaseKill=%d\n",
          tt, PleaseKill);
        Rprintf("--- Not good, OnKappaS=%d, MIPCoefficients[FindPivot =%d] = %f, Beta[kXFinder[RankMyUseless[%d]=%d] = %d] = %f\n",
          OnKappaS, FindPivot, FindPivot, RankMyUselessCoefficients[FindPivot],
          kXFinder[RankMyUselessCoefficients[FindPivot]], REAL(sBeta)[
          kXFinder[RankMyUselessCoefficients[FindPivot]]]); R_FlushConsole();
        Rprintf("---  This will probably trigger an error. \n");
        Rprintf("------------------------------------------------------------------------------------\n");
        R_FlushConsole();
        FFree(GroupsOfMine, "GroupsOfMine"); 
        FFree(MIPCoefficients, "MIPCoefficients");
        FFree(RankMyUselessCoefficients, "RankMyUselessCoefficients");
        return(-1);
      } else if (PleaseKill > 0 && PleaseKill < FindPivot && FindPivot >= OnKappaS) {
        FindPivot = PleaseKill;
      } else if (FindPivot >= OnKappaS) {
        FindPivot = (int) (((double) OnKappaS)*(1.0-KillPct));
      }
      if (FindPivot > KillPct * OnKappaS) {
        if ((1.0-KillPct) * FindPivot > KillPct*OnKappaS) {
          FindPivot = (int) round((1.0-KillPct) * ((double)FindPivot));
        } else {
          FindPivot = FindPivot;
        }
      }
      if (FindPivot < (int) (((double)OnKappaS) * KillPct)) {
        if (REAL(sBeta)[kXFinder[RankMyUselessCoefficients[0]]] != 0.0) {
          Rprintf("--- KillCoefficients, all beta are not zero, this sucks, no ability to kill coefficients here!\n");
          R_FlushConsole();
          FindPivot = -1;
          FFree(GroupsOfMine, "GroupsOfMine"); 
          FFree(MIPCoefficients, "MIPCoefficients");
          FFree(RankMyUselessCoefficients, "RankMyUselessCoefficients");
          return(-1);
        }
        if (REAL(sBeta)[kXFinder[RankMyUselessCoefficients[FindPivot]]] != 0.0) {
        } else {
          for (; FindPivot < OnKappaS; FindPivot++) {
            if (REAL(sBeta)[kXFinder[RankMyUselessCoefficients[FindPivot]]] != 0.0 ||
              FindPivot >= (int) (OnKappaS * KillPct)) {
              break;
            }
          }
        }
      }
    } else {
      for (FindPivot = OnKappaS-1; FindPivot >= 0; FindPivot--) {
        if (REAL(sBeta)[kXFinder[RankMyUselessCoefficients[FindPivot]]] != 0.0 ||
          MIPCoefficients[RankMyUselessCoefficients[FindPivot]] > MinimumProbKeeper) {
          break;
        }      
      }
      if (REAL(sBeta)[kXFinder[RankMyUselessCoefficients[OnKappaS-1]]] != 0.0) {
          Rprintf("--- KillCoefficients, all beta are not zero, this sucks, no ability to kill coefficients here!\n");
          R_FlushConsole();
          FindPivot = -1;
          FFree(GroupsOfMine, "GroupsOfMine"); 
          FFree(MIPCoefficients, "MIPCoefficients");  MIPCoefficients=NULL;
          FFree(RankMyUselessCoefficients, "RankMyUselessCoefficients");
          RankMyUselessCoefficients = NULL;
          return(-1);
      }
      if (OnKappaS-FindPivot < (int) (OnKappaS * KillPct)) {
        for (; FindPivot >= 0; FindPivot--) {
          if (REAL(sBeta)[kXFinder[RankMyUselessCoefficients[FindPivot]]] != 0.0 ||  
            (OnKappaS-FindPivot) > (int) (OnKappaS*(1.0-KillPct))) {
            break;
          }
        }
      }
    }
  } else {
    if (LeftRight == 1) {
      FindPivot = (int) (OnKappaS * KillPct);
      if (REAL(sBeta)[kXFinder[RankMyUselessCoefficients[FindPivot]]] != 0.0 && FindPivot != 0) {
        for (; FindPivot > 0; FindPivot--) {
           if (FindPivot > 0 && REAL(sBeta)[kXFinder[RankMyUselessCoefficients[FindPivot-1]]] != 0.0) {
             break;
           }
        }
      } else if (REAL(sBeta)[kXFinder[RankMyUselessCoefficients[0]]] != 0.0) {
         Rprintf("--- KillCoefficients, all beta are not zero, this sucks, no ability to kill coefficients here!\n");
          R_FlushConsole();
          FindPivot = -1;
          FFree(MIPCoefficients, "MIPCoefficients");
          FFree(RankMyUselessCoefficients, "RankMyUselessCoefficients");
          FFree(GroupsOfMine, "GroupsOfMine"); 
          return(-1);
      }
    } else {
      FindPivot = (int) (OnKappaS * (1.0-KillPct));
      if (REAL(sBeta)[kXFinder[RankMyUselessCoefficients[FindPivot]]] != 0.0 && FindPivot != OnKappaS-1) {
        for (; FindPivot < OnKappaS-1; FindPivot++) {
           if (REAL(sBeta)[kXFinder[RankMyUselessCoefficients[FindPivot+1]]] != 0.0) {
             break;
           }
        }
      } else if (REAL(sBeta)[kXFinder[RankMyUselessCoefficients[OnKappaS-1]]] != 0.0) {
         Rprintf("--- KillCoefficients, all beta are not zero, this sucks, no ability to kill coefficients here!\n");
          R_FlushConsole();
          FindPivot = -1;
          FFree(MIPCoefficients, "MIPCoefficients"); MIPCoefficients = NULL;
          FFree(RankMyUselessCoefficients, "RankMyUselessCoefficients");
          FFree(GroupsOfMine, "GroupsOfMine"); 
          RankMyUselessCoefficients = NULL;
          return(-1);
      }
    }
  }
  if (FindPivot < 0 || FindPivot >= OnKappaS) {
    Rprintf("--------------------------------------------------------------------------\n");
    Rprintf("--- Give Eliminate, we have an error, OnKappaS = %d, KillPct=%f, FindPivot=%d!\n",
      OnKappaS, KillPct, FindPivot);  R_FlushConsole();
    Rprintf("--- LeftRight = %d. \n", LeftRight);  R_FlushConsole();
    if (RunProbVector != NULL) {
      Rprintf("--- RunProbVector is not null.\n");
    } else {
      Rprintf("--- RunProbVector is null.\n");
    }
    Rprintf("--- MIPCoefficients is at an error!\n");
    R_FlushConsole();
    Rf_error("Error: FindPivot is bad.\n");
  }
  if (Verbose >= 1) {
    Rprintf("----------------------------------------------------------------\n");
    Rprintf("--- Hey, We are doing Give Eliminate. LeftRight = %d.\n", LeftRight);
    Rprintf("--- FindPivot was chosen = %d, OnKappaS = %d\n", FindPivot, OnKappaS); R_FlushConsole();
    if (RunProbVector != NULL) {
      Rprintf("--- The RunProbVector is nonzero, MIPCoefficients is probs\n");
      if (FindPivot >= 0 && FindPivot < OnKappaS) {
        Rprintf("---  At FindPivot=%d, the prob = %f \n", FindPivot,   
          MIPCoefficients[RankMyUselessCoefficients[FindPivot]]);
      }
    } else {
      if (FindPivot >= 0 && FindPivot < OnKappaS) {
        Rprintf("**  MIPCoefficients[FindPivot=%d] = %f, is just rank order\n",
          FindPivot, MIPCoefficients[RankMyUselessCoefficients[FindPivot]]); 
      }                     
    }
    R_FlushConsole();
    Rprintf("--- Now we need to do a kill reallocation.\n"); R_FlushConsole();
  }
  int OldKappaS = OnKappaS;
  int OldKappaMem = OnKappaMem;
  if (Verbose >= 2) {
    Rprintf("--- About to DoTheKills about to kill.  ");
    Rprintf("--- Here's the coefficients we want killed: \n  ");   R_FlushConsole();
    int St = FindPivot;   int End = OnKappaS;
    if (LeftRight == 1) {
      St = 0;  End = FindPivot;
    }
    int OnO = 0;
    for (int ii2 = St; ii2 < End; ii2++) {
      if (ii2 >= End-1) {
        Rprintf("%d:%f\n", RankMyUselessCoefficients[ii2], 
          MIPCoefficients[RankMyUselessCoefficients[ii2]]);
        R_FlushConsole();
      } else {
         Rprintf("%d:%f, ", RankMyUselessCoefficients[ii2], 
          MIPCoefficients[RankMyUselessCoefficients[ii2]]);
         if (OnO == 15) {
            Rprintf("\n   "); R_FlushConsole();
            OnO= 0;
         }
      }
      OnO++;
    }
    Rprintf("--- Go and Do your Kills \n"); R_FlushConsole();
  }
  if (GroupsOfMine != NULL) {
    int BK;
    if (LeftRight == 1) {
      int BK;
      for (int ii = 0; ii < FindPivot; ii++) {
        if (iFirstRandom >= 0 && GroupsOfMine[RankMyUselessCoefficients[ii]] >= iFirstRandom) {
          BK = GroupsOfMine[RankMyUselessCoefficients[ii]] - iFirstRandom;
          if (REAL(sOnTau)[BK] != 0.0) {
            REAL(sOnTau)[BK] = 0.0;
          }
        }
      }
    } else {
      for (int ii = OnKappaS-1; ii > FindPivot; ii--) {
        if (iFirstRandom >= 0 && GroupsOfMine[RankMyUselessCoefficients[ii]] >= iFirstRandom) {
          BK = GroupsOfMine[RankMyUselessCoefficients[ii]] - iFirstRandom;
          if (REAL(sOnTau)[BK] != 0.0) {
            REAL(sOnTau)[BK] = 0.0;  
          }   
        }
      }    
    }
  }
  if (Verbose >= 1) {
    Rprintf("--- About to Do Kills: Note FindPivot=%d/%d, LeftRight=%d \n",
      FindPivot, OnKappaS,LeftRight);
    Rprintf("--- Going in One Order:");
    for (int jpj = 0; jpj< 10; jpj++) {
      Rprintf("-(jj=%d/OKS%d:%d/%d:%f)",
        jpj, OnKappaS, RankMyUselessCoefficients[jpj], p, 
        GroupsOfMine[RankMyUselessCoefficients[jpj]]);
      if (jpj >= p-1 || jpj >9) { Rprintf("\n"); break; } else { Rprintf("-"); }
    }
    Rprintf("--- Going down Order:");
    for (int jj = OnKappaS-1; jj>= 0 && jj > OnKappaS-10; jj++) {
      Rprintf("-(jj=%d/OKS%d:%d/%d:%f)",
        jj, OnKappaS, RankMyUselessCoefficients[jj], p, 
        GroupsOfMine[RankMyUselessCoefficients[jj]]);
      if (jj <= 0 || jj <= OnKappaS-9) { Rprintf("\n"); break; } else { Rprintf("-"); }
    }
    R_FlushConsole();
    Rprintf("--- I wonder how DoTheKills will work? \n"); R_FlushConsole();
  }
  DoTheKills(RankMyUselessCoefficients, FindPivot, LeftRight);
  if (Verbose >= 1) {
    Rprintf("--- Do Deletes, we have done all of our Eliminate.  We have now\n");
    Rprintf("--- KillPct = %f, OnKappaS=%d, OldKappaS=%d, FindPivot = %d\n", KillPct, OnKappaS, OldKappaS, FindPivot);
    Rprintf("--- OnKappaMem=%d, OldKappaMem = %d\n", OnKappaMem, OldKappaMem); R_FlushConsole();
    Rprintf("--- About to delete the MIPCoefficients and RankMyUselessCoefficients.\n"); R_FlushConsole();

  }
  if (GroupsOfMine != NULL) { FFree(GroupsOfMine, "GroupsOfMine"); }
  FFree(MIPCoefficients, "MIPCoefficients");  MIPCoefficients = NULL;
  FFree(RankMyUselessCoefficients, "RankMyUselessCoefficients");
  RankMyUselessCoefficients=NULL;
  if (Verbose >= 0) {
    Rprintf("--- BayesSpikeCpp,cpp:::GiveItAChanceAndKillUselessCoeficients(tt=%d/%d): We have finished.\n", tt, MaxGibbsIters);
    Rprintf("---    We have finished eliminating OldKappaS=%d, OnKappaS=%d, %d elements.        \n",
      OldKappaS, OnKappaS, OldKappaS-OnKappaS); R_FlushConsole();
    Rprintf("-----------------------------------------------------------------------------------\n");
    R_FlushConsole();
  }
  if (p >= 100000) { 
    Rprintf(" **>> BayesSpikeCpp:: GiveItAChanceAndKillUselessCoeficients, now CheckpXtX is going on.\n"); R_FlushConsole();
    CheckpXtX();
    Rprintf(" **>> BayesSpikeCpp:: GiveItAChanceAndKillUselessCoeficients, Checked significantly!  \n"); R_FlushConsole();
  }
  return(1);
}

///////////////////////////////////////////////////////////////////////////////
//  PrintCharIntVector()
//
//    Some of these are unnecessary print equations.
//
void PrintCharIntVector(int *Go, char *AGo, int Length, int ATen) {
  Rprintf("Here is %s which has length %d\n  ", AGo, Length);
  if (Length <= 0) { return; }
  int pp = 0;
  for (int ii = 0; ii < Length; ii++){
    if (ii == Length-1) {
      Rprintf("%d\n", Go[ii]); R_FlushConsole();
    } else {
      Rprintf("%d, ", Go[ii]); 
    }
    if (ATen <= pp) {
      Rprintf("\n"); R_FlushConsole();  pp = 0;
    }      
    pp++;
  }
}
void PrintOrderCharIntVector(int *Go, int *Order, char *AGo, 
  char*AOrderGo, int Length, int ATen) {
  Rprintf("Here is %s which has length %d, Order is \n  ", AGo, Length, AOrderGo);  R_FlushConsole();
  if (Length <= 0) { return; }
  int pp = 0;
  for (int ii = 0; ii < Length; ii++){
    if (Order[ii] < 0 || Order[ii] > Length) {
      Rprintf("PrintOrderCharInVector, AOrder = %s, we have ii=%d, is %d but Length=%d\n",
        AOrderGo, ii, Order[ii], Length); R_FlushConsole();
    } else {
    if (ii == Length-1) {
      Rprintf("%d\n", Go[Order[ii]]); R_FlushConsole();
    } else {
      Rprintf("%d, ", Go[Order[ii]]); 
    }
    if (ATen <= pp) {
      Rprintf("\n"); R_FlushConsole(); pp = 0;
    } 
    }     
    pp++;
  }
}

#define MyPrintDoTheKillsMacro()                                          \
        PrintCharIntVector(KeepInOrder,(char*)"KeepInOrder", jj2, 10);          \
        PrintCharIntVector(AllIntsInRank, (char*)"AllIntsInRank", jj2, 10);    \
        PrintCharIntVector(RankMyUseless, (char*)"RankMyUseless", OnKappaS, 10);\
        PrintCharIntVector(kXFinder, (char*)"kXFinder", OnKappaS, 10);          \
        Rprintf("The following should be AllIntsInRank in Order: \n");    \
        R_FlushConsole();                                                 \
        PrintOrderCharIntVector(AllIntsInRank,                            \
           KeepInOrder, (char*)"AllIntsInRank",                           \
            (char*)"KeepInOrder", jj2, 10)

int BayesSpikeCL::DoTheKills(int *RankMyUseless, int FindPivot, int LeftRight) {
  int *AllIntsInRank = NULL;
  int *KeepInOrder = NULL;  int jj2 = 0;
  if (Verbose >= 2) {
    Rprintf("DoTheKills about to get rid of RankMyUseless!\n");  
    R_FlushConsole();
  }
  if (FindPivot < 0 || FindPivot >= OnKappaS) {
    Rf_error("DoTheKills: No way, FindPivot = %d, OnKappaS = %d\n",
      FindPivot, OnKappaS); R_FlushConsole();
  }

  if (LeftRight == 1) {
    jj2 = 0;
    AllIntsInRank = Calloc(OnKappaS-FindPivot+2, int);
    for (int ii2 = FindPivot; ii2 < OnKappaS; ii2++) {
      if (RankMyUseless[ii2] < 0 || RankMyUseless[ii2] >= OnKappaS) {
        Rf_error("DoTheKills, ii2=%d, FindPivot=%d, but RankMyUseless[%d]=%d, and OnKappaS=%d!\n",
         ii2, FindPivot, ii2, RankMyUseless[ii2], OnKappaS);
      }
      AllIntsInRank[jj2] = kXFinder[RankMyUseless[ii2]];
      jj2++;
    }
    KeepInOrder = HeapSortAll(jj2, AllIntsInRank);
  } else {
     AllIntsInRank = Calloc(FindPivot+2, int);
     jj2 = 0;
     for (int ii2 = 0; ii2 < FindPivot; ii2++) {
       if (RankMyUseless[ii2] < 0 || RankMyUseless[ii2] >= OnKappaS) {
        Rf_error("DoTheKills, LeftRight=0 error, ii2=%d, FindPivot=%d, but RankMyUseless[%d]=%d, and OnKappaS=%d!\n",
         ii2, FindPivot, ii2, RankMyUseless[ii2], OnKappaS);
       }
       AllIntsInRank[jj2] = kXFinder[RankMyUseless[ii2]];
       jj2++;
     }
     KeepInOrder = HeapSortAll(jj2, AllIntsInRank);
  }
  int Verbose = this->Verbose - 2;
  int ii2;   int intwenty=0;
  if (Verbose > 2) {
    Rprintf("BayesSpikeCL: We did our KeepInOrder reorder RunRecordXtX \n"); R_FlushConsole();
    if (Verbose >= 2) {
      Rprintf("  Here's all keeper AllIntsInRank in order  \n  ");
      intwenty =0;
      for (ii2 = 0; ii2 < jj2; ii2++) {
        if (intwenty==20) {
          Rprintf("\n"); R_FlushConsole();   intwenty=0;
        }
        if (KeepInOrder[ii2] < 0 || KeepInOrder[ii2] >= jj2) {
          Rf_error("Error In trying to cout KeepInOrder, ii2=%d, KeepInOrder[%d]=%d!\n",
            ii2, ii2, KeepInOrder[ii2]);
        }
        if (ii2 == jj2-1) {
          Rprintf("%d.\n", AllIntsInRank[KeepInOrder[ii2]]); R_FlushConsole();
        } else {
          Rprintf("%d, ", AllIntsInRank[KeepInOrder[ii2]]); R_FlushConsole();
        }
        intwenty++;
      }
      Rprintf("Well Hopefully that's a winner!\n"); R_FlushConsole();
    }
  }
  //if (TrueOrder == 1) { return(0); }

  int XtXDim = p * OnKappaMem;
  if (Verbose >= 2) {
    Rprintf("   DoKill: Allocate NewXtX of dim %d*%d= %d\n",
      p, OnKappaMem, XtXDim);
    R_FlushConsole();
  }
  //double **NewpXtX = (double**) Calloc(OnKappaMem, double*);  
  //for (int iti = 0; iti < OnKappaMem; iti++) {
  //  NewpXtX[iti] = NULL;
  //}

  short int *NewiWeightedXtX = NULL;
  if (iWeightedXtX != NULL) {
    if (Verbose >= 2) {
      Rprintf("   BayesSpikeCpp.cpp:::DoKill() Allocating NewiWeightedXtX\n"); R_FlushConsole();
    }
    RMemGetS(NewiWeightedXtX, "NewiWeightedXtX", OnKappaMem); 
  }
  int One = 1; 
  int GoP = 0;
  if (Verbose >= 2) {
    Rprintf("   DoKill: About to do copying of XColumns, LeftRight = %d, One = %d.\n", LeftRight, One);
    R_FlushConsole();
  }
  int ComingIn = 0; int PositionComingIn=0; int GoingOut = 0;
  double *pGoingOut = NULL;
  if (AllIntsInRank[KeepInOrder[0]] > AllIntsInRank[KeepInOrder[jj2-1]]) {
    GoP = jj2-1;
    One = 1;
    for (ii2 = 0; ii2 < jj2; ii2++) {
      if (KeepInOrder[ii2] < 0 || KeepInOrder[ii2] >= OnKappaS) {
        Rprintf("BayesSpikeCpp.cpp::DoTheKills: OnDownOrder: We have below or above on KeepInOrder, OnKappaS Error.  When ii2=%d\n", ii2);
        Rprintf("AllIntsInrank has a problem, KeepInOrder[%d]=%d, AllInts=%d, OnKappaS = %d\n",
          ii2, KeepInOrder[ii2], AllIntsInRank[KeepInOrder[ii2]], OnKappaS); R_FlushConsole();
        MyPrintDoTheKillsMacro();
        Rf_error("This is not good.\n");
      }
      if (AllIntsInRank[KeepInOrder[ii2]] <  0 || 
        AllIntsInRank[KeepInOrder[ii2]] >= p) {
        Rprintf("BayesSpikeCpp.cpp::DoTheKills: OnDownOrder: Error in DoTheKills, ii2 = %d, jj2 = %d, GoP=%d\n",
          ii2, jj2, GoP);
        MyPrintDoTheKillsMacro();
        Rf_error("DoKill: This is bad and doesn't work!\n");
      }
      if (GoP < 0 || GoP >= OnKappaS) {
        Rprintf("BayesSpikeCpp.cpp::DoTheKills: OnDownOrder: DoKill, GoP = %d, OnKappaS=%d\n", GoP, OnKappaS); R_FlushConsole();
        MyPrintDoTheKillsMacro();
        Rf_error("DoKill! We can't copy this in if GoP = %d!\n", GoP);
      }
      if (XLC[AllIntsInRank[KeepInOrder[ii2]]] < 0 &&
        XLC[AllIntsInRank[KeepInOrder[ii2]]] >= OnKappaS) {
        Rprintf("BayesSpikeCpp.cpp::DoTheKills: OnDownOrder, OnKappaS=%d/%d, ii2=%d, KeepInOrder[ii2=%d]=%d, AllIntsInRank[%d]=%d, XLC[%d]=%d", 
          OnKappaS, OnKappaMem, ii2, ii2, KeepInOrder[ii2], KeepInOrder[ii2],
          AllIntsInRank[KeepInOrder[ii2]], AllIntsInRank[KeepInOrder[ii2]],
          XLC[AllIntsInRank[KeepInOrder[ii2]]]); R_FlushConsole();  
        Rprintf("BayesSpikeCpp.cpp::DoTheKills: OnDownOrder, this shouldn't have happened, so an error!\n");
        R_FlushConsole();
        Rf_error("No, this OnKappaS=%d, ii2=%d, not working!\n");
      } 
      ComingIn = AllIntsInRank[KeepInOrder[ii2]];
      PositionComingIn = XLC[ComingIn];
      GoingOut = kXFinder[GoP];
      pGoingOut = pXtX[GoP];
      pXtX[GoP] = pXtX[PositionComingIn];
      kXFinder[GoP] = ComingIn;
      XLC[ComingIn] = GoP;
      pXtX[PositionComingIn] = pGoingOut;
      kXFinder[PositionComingIn] = GoingOut;
      XLC[GoingOut] = PositionComingIn;
      //NewpXtX[GoP] = pXtX[XLC[AllIntsInRank[KeepInOrder[ii2]]]];
      //F77_CALL(dcopy)(&p, pXtX[XLC[AllIntsInRank[KeepInOrder[ii2]]] * p, &One, NewXtX + p * GoP, &One);
      if (iWeightedXtX != NULL && NewiWeightedXtX != NULL) {
        if (GoP < 0 ||  GoP >= OnKappaMem) {
          Rf_error("DoKill: Oh no NewiWeighteXtX, GoP=%d, OnKappaMem=%d\n", GoP, OnKappaMem);
        }
        NewiWeightedXtX[GoP] = iWeightedXtX[AllIntsInRank[KeepInOrder[ii2]]];
      }
      GoP--;
      if (GoP < 0) {
        break;
      }
    }
  } else {
    GoP = 0;
    for (ii2 = 0; ii2 < jj2; ii2++) {
      if (KeepInOrder[ii2] < 0 || KeepInOrder[ii2] >= OnKappaS) {
        Rprintf("DoKill: OnUpOrder: We have below or above on KeepInOrder, OnKappaS Error.  When ii2=%d\n", ii2);
        Rprintf("DoKill: OnUpOrder: AllIntsInrank has a problem, KeepInOrder[%d]=%d, AllInts=%d, OnKappaS = %d\n",
          ii2, KeepInOrder[ii2], AllIntsInRank[KeepInOrder[ii2]], OnKappaS); R_FlushConsole();
        MyPrintDoTheKillsMacro();
        Rf_error("DoKill: This is not good.\n");
      }
      if (AllIntsInRank[KeepInOrder[ii2]] <  0 || 
        AllIntsInRank[KeepInOrder[ii2]] >= p) {
        Rprintf("DoKill: OnUpOrder: Error in up ReOrder, ii2 = %d, jj2 = %d, GoP=%d\n",
          ii2, jj2, GoP);
        Rprintf("DoKill: OnUpOrder: We KeepInOrder[ii2=%d] = %d, AllIntsInRank[%d] = %d\n",
          ii2, KeepInOrder[ii2], AllIntsInRank[KeepInOrder[ii2]]);
        MyPrintDoTheKillsMacro();
        Rf_error("DoKill: This is bad and doesn't work!\n");
      }
      if (XLC[AllIntsInRank[KeepInOrder[ii2]]] < 0) {
        Rprintf("DoKill: OnUpOrder: We KeepInOrder have an error, ii2 = %d!\n", ii2);
        R_FlushConsole();
        Rprintf("DoKill: OnUpOrder: KeepInOrder[ii2=%d] = %d, AllIntsInRank[KeepInOrder%d] = %d, XLC[AllIntsInRank%d] = %d\n",
          ii2, KeepInOrder[ii2], AllIntsInRank[KeepInOrder[ii2]], 
          XLC[AllIntsInRank[KeepInOrder[ii2]]]);
        MyPrintDoTheKillsMacro();
        Rf_error("AllIntsInRank. \n");
      }
      ComingIn = AllIntsInRank[KeepInOrder[ii2]];
      PositionComingIn = XLC[ComingIn];
      GoingOut = kXFinder[GoP];
      pGoingOut = pXtX[GoP];
      pXtX[GoP] = pXtX[PositionComingIn];
      kXFinder[GoP] = ComingIn;
      XLC[ComingIn] = GoP;
      pXtX[PositionComingIn] = pGoingOut;
      kXFinder[PositionComingIn] = GoingOut;
      XLC[GoingOut] = PositionComingIn;
      //NewpXtX[GoP] = pXtX[XLC[AllIntsInRank[KeepInOrder[ii2]]]];
      //F77_CALL(dcopy)(&p, XtX + XLC[AllIntsInRank[KeepInOrder[ii2]]] * p, &One, NewXtX + p * GoP, &One);
      if (NewiWeightedXtX != NULL && iWeightedXtX != NULL) {
        if (GoP < 0 || GoP >= OnKappaMem) {
          Rf_error("DoKill: Oh no, here GoP = %d can't fill NewiWeightedXtX \n", GoP); 
        }
        NewiWeightedXtX[GoP] = iWeightedXtX[AllIntsInRank[KeepInOrder[ii2]]];
      }
      //kXFinder[GoP] = AllIntsInRank[KeepInOrder[ii2]];
      GoP++;
      if (GoP < 0) {
        break;
      }
    }  
  }
  if (Verbose >= 2) {
    Rprintf("   DoKill: Blanking out additional kXFinder\n"); R_FlushConsole();
  }
  if (jj2 < OnKappaS) {
    for (ii2 = jj2; ii2 < OnKappaS; ii2++) {
      kXFinder[ii2]  = -1;
    }
  }
  for (ii2 = 0; ii2 < p; ii2++) {
    XLC[ii2] = -1;
  }
  for (ii2 = 0; ii2 < jj2; ii2++) {
    if (ii2 < OnKappaS && kXFinder[ii2] >= 0) {
      XLC[kXFinder[ii2]] = ii2;
    }
  }

  //double **TemppXtX = pXtX;
  if (Verbose >= -1) {
    Rprintf("  GiveTheKills we did all of the killing of columns hoped this worked!\n"); R_FlushConsole();
  }
  if (Verbose >= -1) {
    Rprintf("  GiveTheKills, about to Free\n"); R_FlushConsole();
  }
  //this->pXtX = NewpXtX;
  for (int iti = 0; iti < OnKappaMem; iti++) {
    if (this->pXtX[iti] == NULL) {
      Rf_error("  GiveTheKills: not good because iti=%d/%d is NULL!\n",
        iti, OnKappaMem);
    }
  }
  OnKappaS = jj2;
  if (iWeightedXtX != NULL) {
    if (Verbose >= 2) {
      Rprintf("   BayesSpikeCpp.cpp:::GiveTheKills()  Free iWeightedXtX \n"); R_FlushConsole();
    }
    FFree(iWeightedXtX, "iWeightedXtX"); iWeightedXtX = NULL;
    if (NewiWeightedXtX != NULL) {
      iWeightedXtX = NewiWeightedXtX;
      FullCrazyTestiWeightedXtX((char*) "After Give The Kills");
    }
  }
  //FFree(TemppXtX, "TemppXtX");           TemppXtX = NULL;
  FFree(KeepInOrder, "KeepInOrder");     KeepInOrder = NULL;
  FFree(AllIntsInRank, "AllIntsInRank"); AllIntsInRank = NULL;
  TrueOrder = 1;    
  if (Verbose >= -1) {
    Rprintf("  GiveTheKills, all done.\n"); R_FlushConsole();
  }
  CheckkXFinderError((char*)"GiveTheKills");
  if (Verbose >= -1) {
    Rprintf("  End for GiveTheKills: CheckkXFinder. \n"); R_FlushConsole();
  }
  return(1);

}
int BayesSpikeCL::ReorderXtX() {
  int Verbose = this->Verbose - 2;
  if (Verbose >= 2) {
    Rprintf("BayesSpikeCL(tt=%d): RunRecordXtX (OnKappaS=%d/%d/%d)\n", tt, OnKappaS, OnKappaMem, p); R_FlushConsole();
  }
  if (Verbose >= 3) {
    Rprintf("BayesSpikeCL:ReorderXtX, OnKappaS=%d/OnKappaMem=%d\n", OnKappaS, OnKappaMem); R_FlushConsole();
  }
  
  //return(6);
  //if (TrueOrder == 1) { return(0); }
  //int XtXDim = p * OnKappaMem;
  //double **NewpXtX = (double**) Calloc(OnKappaMem, double*);  
  double *pGoingOut=NULL;
  int GoingOut=0;  int LocGoingOut=0;
  int ComingIn=0;  int LocComingIn=0; 
  
  if (Verbose >= 6) {
    Rprintf("%d%d%d%d%d%d\n", GoingOut, LocGoingOut, ComingIn, LocComingIn);
    if (pGoingOut == NULL) {
     Rprintf("pGoingOut is not NULL\n"); R_FlushConsole();
    }
  }
  int ii=0; //int One = 1; 
  int TotMoved = 0;
  short int *NewWeightedXtX = NULL;
  int *NewkXFinder = NULL;
  double **NewpXtX = NULL;
  
  //NewpXtX = (double**) Calloc(OnKappaMem, double*);  
  //for (int iti = 0; iti < OnKappaMem; iti++) {
  //  NewpXtX[iti] = pXtX[iti];
  //}
  //Rprintf("BayesSpikeCpp.cpp::ReOrder, we have pXtX refilled \n"); R_FlushConsole();
  //FFree(pXtX, "pXtX"); this->pXtX = NewpXtX;
  //NewWeightedXtX = (short int*) Calloc(OnKappaMem, short int);
  //for (int iti = 0; iti < OnKappaMem; iti++) {
  //  NewWeightedXtX[iti] = iWeightedXtX[iti];
  //}
  //FFree(iWeightedXtX, "iWeightedXtX"); iWeightedXtX = NewWeightedXtX;
  //Rprintf("BayesSpikeCpp.cpp::ReOrder, we replaced iWeightedXtX\n"); R_FlushConsole();
  
  //NewkXFinder = (int*) Calloc(OnKappaMem, int);
  //for (int iti = 0; iti < OnKappaMem; iti++) {
  //  NewkXFinder[iti] = kXFinder[iti];
  //}
  //FFree(kXFinder, "kXFinder"); kXFinder = NewkXFinder;
  //Rprintf("BayesSpikeCpp.cpp::ReOrder, we replaced kXFinder\n"); R_FlushConsole();
  //return(6);
  
  
  if (OnKappaS <= 0) {
    return(6);
  }
  if (iiWeight != NULL && iWeightedXtX != NULL) {
    RMemGetS(NewWeightedXtX, "NewWeightedXtX", OnKappaMem);
  }
  RMemGetI(NewkXFinder, "NewkXFinder", OnKappaMem+1);
  if (OnKappaS < OnKappaMem) {
    for (ii = OnKappaS; ii < OnKappaMem; ii++) {
      NewkXFinder[ii] = -1;
    }
  }
  for (ii = 0; ii < p; ii++) {
    if (XLC[ii] >= 0 && XLC[ii] < OnKappaS) {
      if (TotMoved >= OnKappaS) {
        Rf_error("** BayesSpike.cpp::ReOrderXtX, TotMoved=%d, OnKappaS=%d/%d what happened for XLC?\n",
          TotMoved, OnKappaS, OnKappaMem);
      }
      NewkXFinder[TotMoved] = ii;  TotMoved++;
    }
  }
  if (OnKappaS != TotMoved) {
    Rprintf("** BayesSpikeCpp.cpp::ReOrderXtX, well OnKappaS=%d/%d, TotMoved=%d, probably a disasterous error!\n",
      OnKappaS, OnKappaMem, TotMoved);
  }
  if (NewWeightedXtX != NULL) {
  for (ii = 0; ii < OnKappaS; ii++) {
     if (XLC[NewkXFinder[ii]] < 0) {
       Rprintf("** BayesSpikeCpp.cpp::ReOrderXtX(ii=%d/%d=OnKppaS), why is XLC[NewkXFinder[ii=%d]=%d]=%d?\n",
         ii, NewkXFinder[ii], XLC[NewkXFinder[ii]]); R_FlushConsole();
     } else {
       NewWeightedXtX[ii] = iWeightedXtX[XLC[NewkXFinder[ii]]];
     }
  }
  }
  NewpXtX = (double **) Calloc(OnKappaMem, double*);
  for (ii = 0; ii < OnKappaS; ii++) {
    NewpXtX[ii] = NULL;
  }
  for (ii = 0; ii < OnKappaS; ii++) {
    if (XLC[NewkXFinder[ii]] < 0) {
       Rprintf("** BayesSpikeCpp.cpp::ReOrderXtX(ii=%d/%d=OnKappaS), why is XLC[NewkXFinder[ii=%d]=%d]=%d?\n",
         ii, NewkXFinder[ii], XLC[NewkXFinder[ii]]); R_FlushConsole();
    } else {
      if (pXtX[XLC[NewkXFinder[ii]]] == NULL) {
        Rprintf("** BayesSpikeCpp.cpp::ReOrderXtX(ii=%d/%d=OnKappaS),pXtX will have problems, XLC[NewkXFinder[ii=%d]=%d] = %d but pXtX[%d]=NULL!\n",
          ii, OnKappaS, NewkXFinder[ii], XLC[NewkXFinder[ii]], XLC[NewkXFinder[ii]], pXtX[XLC[NewkXFinder[ii]]]);
      } else {
         NewpXtX[ii] = (double *) pXtX[XLC[NewkXFinder[ii]]];
      }
    }
  }
  if (OnKappaS < OnKappaMem) {
    for (ii = OnKappaS; ii < OnKappaMem; ii++) {
      NewpXtX[ii] = pXtX[ii];
    }
  }
  for (ii = 0; ii < OnKappaS; ii++) {
    if (NewpXtX[ii] == NULL) {
      Rprintf("BayesSpikeCpp.cpp::ReOrderXtX(ii=%d/%d OnKappaS), no we are screwed! NewpXtX[ii=%d]=NULL\n", ii, OnKappaS, ii);
      R_FlushConsole();
    }
  }
  //FFree(NewpXtX, "NewpXtX"); FFree(NewkXFinder, "NewkXFinder");
  //FFree(NewWeightedXtX, "NewWeightedXtX");  
  //Rprintf("BayesSpikeCpp.cpp::ReOrder, early Free, that's it. \n"); R_FlushConsole();
  //return(6);
  
  for (ii = 0; ii < OnKappaS; ii++) {
    if (NewkXFinder[ii] < 0 || NewkXFinder[ii] >= p) {
       Rprintf("** BayesSpikeCpp.cpp::ReOrderXtX(ii=%d/%d=OnKppaS), why is NewkXFinder[ii=%d]=%d?\n",
         ii, NewkXFinder[ii]); R_FlushConsole();
    } else {
       XLC[NewkXFinder[ii]] = ii;
    }
  }
  if (Verbose >= 2) {
    Rprintf("BayesSpikeCpp.cpp::ReorderXtX, after we're freeing. pXtX \n");  R_FlushConsole();
  }
  FFree(pXtX, "pXtX");  pXtX = NewpXtX; NewpXtX = NULL;
  
  /*
  Rprintf("BayesSpikeCpp.cpp: Now test reordered NewpXtX \n"); R_FlushConsole();
  for (ii = 0; ii < OnKappaMem; ii++) {
    if (pXtX[ii] == NULL) {
      pXtX[ii] = NULL;
    } else {
      pXtX[ii] = pXtX[ii] + 0;
      for (int jj = 0; jj < p; jj++) {
        *(pXtX[ii] + jj) = *(pXtX[ii] + jj) + 0.0;
      }
    }
  } */
  //Rprintf("BayesSpikeCpp.cpp: Well, we did check of pXtX"); R_FlushConsole();
  
  if (Verbose >= 2) {
    Rprintf("BayesSpikeCpp.cpp::ReorderXtX, freeing iWeightedXtX \n");
  }
  if (iWeightedXtX != NULL && NewWeightedXtX != NULL) {
    FFree(iWeightedXtX, "iWeightedXtX"); iWeightedXtX = NewWeightedXtX;
    //Rprintf("BayesSpikeCpp.cpp: Test NewWeightedXtX\n");  R_FlushConsole();
    //for (ii = 0; ii < OnKappaMem; ii++) {
    //  iWeightedXtX[ii] = iWeightedXtX[ii]+0;
    //}
    //Rprintf("BayesSpikeCpp.cpp:ReOrder, iWeightedXtX tested\n");R_FlushConsole();
  }
  if (Verbose >= 2) {
    Rprintf("BayesSpikeCpp.cpp::ReorderXtX freeing kXFinder");
  }
  FFree(kXFinder, "kXFinder");  kXFinder = NewkXFinder;
  //Rprintf("BayesSpikeCpp.cpp: Test kXFinder\n");
  //for (ii = 0; ii < OnKappaS; ii++) {
  //  if (kXFinder[ii] < 0 || kXFinder[ii] >= p) {
  //    Rprintf("NewkXFinder updated, no way, kXFinder[ii=%d/%d] = %d? No p=%d!\n", ii, OnKappaS, kXFinder[ii], p); R_FlushConsole();
  //  } else if (XLC[kXFinder[ii]] != ii) {
  //    Rprintf("NewkXFinder updated, no way, XLC[kXFinder[ii=%d/%d]=%d]=%d, No p=%d!\n", ii, OnKappaS, kXFinder[ii],
  //      XLC[kXFinder[ii]], p); R_FlushConsole();
  //  }
  //}
  if (OnKappaS < OnKappaMem) {
  for (ii = OnKappaS; ii < OnKappaMem; ii++) {
     kXFinder[ii] = -1;
  }
  }
  return(1);  
  if (Verbose >= -1) {
    Rprintf("BayesSpikeCpp.cpp::ReorderXtX, did this work? \n"); R_FlushConsole();
  }
  //this->Verbose = 5;
  return(1);
  for (ii = 0; ii < p; ii++) {
    if (XLC[ii] >= 0) {
      if (XLC[ii] >= OnKappaS) {
        Rprintf("** BayesSpikeCpp.cpp::ReorderXtX: Bad, ii=%d, XLC[ii=%d]=%d, OnKappaS=%d/%d, bad!\n",
         ii, ii, XLC[ii], OnKappaS, OnKappaMem); R_FlushConsole();
        Rf_error("** BayesSpikeCpp.cpp::ReorderXtX Error: XLC goes over the top. \n");
      }
      ComingIn = ii;  LocComingIn = XLC[ii];
      GoingOut = kXFinder[TotMoved];  
      LocGoingOut = TotMoved;
      pGoingOut = pXtX[TotMoved];
      pXtX[LocGoingOut] = pXtX[LocComingIn];
      pXtX[LocComingIn] = pGoingOut;
      XLC[ComingIn] = LocGoingOut;
      XLC[GoingOut] = LocComingIn;
      if (NewWeightedXtX != NULL) {
        NewWeightedXtX[TotMoved] = iWeightedXtX[LocComingIn];
      }
      kXFinder[LocGoingOut] = ComingIn;
      kXFinder[LocComingIn] = GoingOut;
      //NewpXtX[TotMoved] = pXtX[XLC[ii]];
      //F77_CALL(dcopy)(&p, XtX + XLC[ii] * p, &One, NewXtX + p * TotMoved, &One);
      //kXFinder[TotMoved] = ii; XLC[ii] = TotMoved;
      TotMoved++;
      if (TotMoved >= OnKappaS) {
        break;
      }
    }
  }
  if (NewWeightedXtX != NULL) {
    if (iWeightedXtX != NULL) {
      FFree(iWeightedXtX, "iWeightedXtX");
    }
    iWeightedXtX = NewWeightedXtX; NewWeightedXtX = NULL;
  }
  //double **TemppXtX = this->pXtX;
  //this->pXtX = NewpXtX;
  //FFree(TemppXtX, "TemppXtX"); 
  int CountNulls = 0;
  for (ii = 0; ii < OnKappaMem; ii++) {
    if (this->pXtX[ii] == NULL) {
      Rprintf("** BayesSpikeCpp.cpp::ReorderxtX(tt=%d)  Bad pXtX[ii=%d/%d] is NULL!\n",
       tt, ii, OnKappaMem); R_FlushConsole(); CountNulls++;
    }
  } 
  if (CountNulls > 0) {
     Rprintf("ERRORERRROERRROERRORERRORERRORERRROERRROERRORERRORERRORERRROERRROERRORERROR\n");
     Rprintf("** BayesSpikeCpp.cpp::ReorderxtX(tt=%d) OnKappaMem=%d, OnKappaS=%d, We counted %d NULLs!\n",
       tt, ii, OnKappaMem); R_FlushConsole(); CountNulls++;  
  }
  TrueOrder = 1;  
  if (p >= 100000) {  
    //Rprintf(" **>> BayesSpikeCpp:: ReOrder, now CheckpXtX is going on.\n"); R_FlushConsole();
    //CheckpXtX();
    //Rprintf(" **>> BayesSpikeCpp:: ReOrder, Checked significantly!  \n"); R_FlushConsole();
  }
  return(1);
}

int BayesSpikeCL::QuickCheckWeightXtX() {
  if (OnKappaS <= 0) { return(0); }
  if (iiWeight == NULL) {
    return(0);
    //Rf_error("QuickCheckWeightXtX() Why ask if iiWeight is NULL?\n");
  }
  if (WW == NULL) {
    RMemGetD(WW, "WW", n);
  }
  double MyDot = 0.0;
  int One = 1;  int jj; 
  int CountRejects=0;
  for (jj = 0; jj < OnKappaS; jj++) {
    if ((int) iWeightedXtX[jj] != (short int) 0) {
      for (int iti = 0; iti < n; iti++) {
        WW[iti] = iiWeight[iti] * REAL(sX)[iti + n * kXFinder[jj]];
      }
      MyDot = F77_CALL(ddot)(&n, WW, &One, REAL(sX), &One);
      if (fabs(*(pXtX[jj]+0)-MyDot) > .0001) {
         CountRejects++;
      }
    }
  }
  if (CountRejects == 0) { return(0); }
  int tNR = 0;
  Rprintf("BayesSpikeCpp.cpp:::QuickCheckWeightXtX, we have %d/%d fails!\n", CountRejects, OnKappaS);
  R_FlushConsole();
  int No10 = 0;
  for (jj = 0; jj < OnKappaS; jj++) {
    if ((int) iWeightedXtX[jj] != (short int) 0)  {
      for (int iti = 0; iti < n; iti++) {
        WW[iti] = iiWeight[iti] * REAL(sX)[iti + n * kXFinder[jj]];
      }
      MyDot = F77_CALL(ddot)(&n, WW, &One, REAL(sX), &One);
      if (fabs(*(pXtX[jj]+0)-MyDot) > .0001) {
         if (No10 >= 2) { Rprintf("\n");  No10= 0; R_FlushConsole();}
         Rprintf("%d/%d: Coord %d has %d/%d:iiWeight=%d but %.5f!=%.5f, ", tNR, CountRejects, kXFinder[jj], jj, OnKappaS, (int) iWeightedXtX[jj],
           *(pXtX[jj]+0), MyDot); R_FlushConsole();  
         if (REAL(sBeta)[kXFinder[jj]] == 0.0) {
           Rprintf("  ---  Don't know why iWeighted[jj=%d] = %d, since Beta[kXFinder[jj=%d]=%d]=%f? \n", jj, (int) iWeightedXtX[jj], 
             jj, kXFinder[jj], REAL(sBeta)[kXFinder[jj]]); R_FlushConsole();
         } else {
           Rprintf("  ---  Note, Valid to be on since Beta[kXFinder[jj=%d]=%d]=%f? \n", jj, (int) iWeightedXtX[jj], 
             jj, kXFinder[jj], REAL(sBeta)[kXFinder[jj]]); R_FlushConsole();         
         }          
         No10++;  
         tNR++;        
      }
    }
  }
  return(CountRejects);
}
////////////////////////////////////////////////////////////////////////
//  Reweight only XtX which come from active, non-zero coefficients;
int BayesSpikeCL::ReweightOnlyActiveXtX() {
  int jj;   int One = 1;  int ii;
  if (Verbose >= 5) {
    Rprintf("BayesSpikeCpp.cpp():::ReweightOnlyActiveXtX \n"); R_FlushConsole();
  }
  int CountBad = 0;

  if (OnKappaS <= 0) {
    return(6);
  }
  if (OnKappaS >= 0) {
  for (jj = 0; jj < OnKappaS; jj++) {
    iWeightedXtX[jj] = (short int) 0;
  }
  }
  int TFixed = p;
  if (iFirstRandom == 0 || BFixed == NULL) {
    TFixed = 0;
  } else if (iFirstRandom > 0 && iFirstRandom < p && BFixed != NULL) {
    TFixed = iFirstRandom;
  }
  if (TFixed > 0) {
    for (jj = 0; jj < OnKappaS; jj++) {
      if (kXFinder[jj] < TFixed && 
        (REAL(sBeta)[kXFinder[jj]] != 0.0 || BFixed[jj] == 1) && iWeightedXtX[jj] == 0)  {
        ReweightedOneXtX(kXFinder[jj]);
        iWeightedXtX[jj] = 1; 
        XjSq[kXFinder[jj]] = *(pXtX[jj] + kXFinder[jj]);
      }
    }
  }
  int xJ; int St = 0;
  if (tauEndList != NULL && !Rf_isNull(tauEndList) && Rf_length(tauEndList) >= 1 &&
    sOnTau != NULL && !Rf_isNull(sOnTau) ) {
    for (jj = 0; jj < Rf_length(sOnTau); jj++) {
      if (REAL(sOnTau)[jj] > 0.0) {
         if (jj == 0) { St = iFirstRandom;} else {
           St = ANINT(tauEndList, jj-1);
         }
         for (xJ = St; xJ < ANINT(tauEndList, jj); xJ++) {
           if (XLC[xJ] >= 0 && XLC[xJ] < OnKappaS && iWeightedXtX[XLC[xJ]] == 0) {
             ReweightedOneXtX(xJ);
             iWeightedXtX[XLC[xJ]] = 1; 
             XjSq[xJ] = *(pXtX[XLC[xJ]] + xJ);
           }
         }
      }
    }  
  }
  //return(1);  
  
  //for (jj = 0; jj < OnKappaS; jj++) {
  //  if (REAL(sBeta)[kXFinder[jj]] == 0.0) {
  //    //ReweightedOneXtX(kXFinder[jj]);
  //    iWeightedXtX[jj] = 0;
  //  } else {
  //    ReweightedOneXtX(kXFinder[jj]);
  //  }
  //}
  //}
  if (iiWeight != NULL && dfRobit > 0.0) {
   CountBad = QuickCheckWeightXtX();
  }
  if (CountBad > 0) {
    Rprintf("BayesSpikeCpp.cpp():::ReweightOnlyActiveXtX Immediately in on tt=%d/%d we got CountBad = %d, AFTER we Reweight.   Not Good I quit!\n",
      tt, MaxGibbsIters, CountBad); R_FlushConsole();
    Rf_error("BayesSpikeCpp.cpp():::ReweightOnlyActiveXtX  Immediately in fail at end. \n"); R_FlushConsole();
  }
  if (Verbose >= 5) {
    Rprintf("BayesSpikeCpp.cpp():::  Done ReweigtedOneXtX now reweight XjSq \n"); R_FlushConsole();
  }
  if (iiWeight == NULL) {
    Rf_error("BayesSpikeCpp.cpp()::: No weight, iiWeight is NULL!\n");
  }
  if (WW == NULL) {
    RMemGetD(WW, "WW", n);
  }
  for (jj = 0; jj < p; jj++) {
    if (XLC[jj] >= OnKappaS) {
      Rprintf("BayesSpikeCpp.cpp: Error In ReWeightOnlyActive, XLC[jj=%d]=%d, but OnKappaS=%d/%d!\n",
        jj, XLC[jj], OnKappaS, OnKappaMem); R_FlushConsole();
      Rf_error("--  Error on XLC[jj=%d]=%d, OnKappaS=%d!\n", jj, XLC[jj], OnKappaS);
    } else if (XLC[jj] >= 0 && XLC[jj] < OnKappaS && iWeightedXtX[XLC[jj]] == 1) {
      XjSq[jj] = *(pXtX[XLC[jj]] + jj); //[jj + XLC[jj] * p];
    } else {
      for (ii = 0; ii < n; ii++) {
        WW[ii] = iiWeight[ii] * REAL(sX)[ii + jj * n];
      }
      XjSq[jj] = F77_CALL(ddot)(&n, WW, &One, REAL(sX) + jj * n, &One);
    }
  }
  return(1);
}

////////////////////////////////////////////////////////////////////////
//  int BayesSpikeCL::BackReweightOnlyActiveXtX()
//
//  Reweight only XtX which come from active, non-zero coefficients;
int BayesSpikeCL::BackReweightOnlyActiveXtX() {
  if (iWeightedXtX == NULL) {
    Rf_error("Error: BackReWeightOnlyActiveXtX, can't do!\n");
  }
  int jj;   //int One = 1;  int ii;
  if (OnKappaS <= 0) {
    return(1);
  }
  int TFixed = p;
  if (iFirstRandom == 0 || BFixed == NULL) {
    TFixed = 0;
  } else if (iFirstRandom > 0 && iFirstRandom < p && BFixed != NULL) {
    TFixed = iFirstRandom;
  }
  //int *IReweighted = Calloc(p,int);
  //int cR = 0;
  if (TFixed > 0) {
    for (jj = 0; jj < OnKappaS; jj++) {
      if (kXFinder[jj] < TFixed && 
        (REAL(sBeta)[kXFinder[jj]] != 0.0 || BFixed[jj] == 1) && iWeightedXtX[jj] == 0)  {
        ReweightedOneXtX(kXFinder[jj]);      //IReweighted[cR] = kXFinder[jj]; cR++;
        iWeightedXtX[jj] = 1; 
        XjSq[kXFinder[jj]] = *(pXtX[jj] + kXFinder[jj]);
      }
    }
  }
  int xJ; int St = 0;
  if (tauEndList != NULL && !Rf_isNull(tauEndList) && Rf_length(tauEndList) >= 1 &&
    sOnTau != NULL && !Rf_isNull(sOnTau) ) {
    for (jj = 0; jj < Rf_length(sOnTau); jj++) {
      if (REAL(sOnTau)[jj] > 0.0) {
         if (jj == 0) { St = iFirstRandom;} else {
           St = ANINT(tauEndList, jj-1)+1;
         }
         for (xJ = St; xJ <= ANINT(tauEndList, jj); xJ++) {
           if (XLC[xJ] >= 0 && XLC[xJ] < OnKappaS && iWeightedXtX[XLC[xJ]] == 0) {
             ReweightedOneXtX(xJ);      //IReweighted[cR] = xJ; cR++;
             iWeightedXtX[XLC[xJ]] = 1; 
             XjSq[xJ] = *(pXtX[XLC[xJ]] + xJ);
           }
         }
      }
    }  
  }
  if (Verbose >= 3) {
    Rprintf("BackReweightedOneXtX here is the reweighted coords:");
    //Rprintf("\nRW <- c("); 
    //for (jj = 0; jj < cR; jj++) {
    //  Rprintf("IReweighted[%d]",jj); if (jj < cR-1) { Rprintf(", ");}
    //}
    //Rprintf(")\n"); R_FlushConsole();
  }
  //Free(IReweighted);  IReweighted = NULL;
  return(1);
  for (jj = 0; jj < OnKappaS; jj++) {
    if (REAL(sBeta)[kXFinder[jj]] != 0.0 && iWeightedXtX[jj] == 0) {
      ReweightedOneXtX(kXFinder[jj]);
      iWeightedXtX[jj] = 1;
      XjSq[jj] = *(pXtX[jj] + kXFinder[jj]);
    } else if (REAL(sBeta)[kXFinder[jj]] == 0.0 && iWeightedXtX[jj] == 1) {
      iWeightedXtX[jj] = 0;
    }
  }
  return(1);
}

/////////////////////////////////////////////////////////////////////////
// Reweight XtX only for requested coefficient iOneX
int BayesSpikeCL::ReweightedOneXtX(int iOneX) {
  if (iOneX < 0 || iOneX >= p) {
    Rf_error("Error: ReweightedOneXtX, we cannot update iOneX = %d\n", iOneX);
  }
  int jj = XLC[iOneX]; int ii;  double Factor = 0.0;    char Trans = 'T';
  if (kXFinder[jj] != iOneX) {
    Rf_error("Error: ReweightedOneXtX: you asked to update iOneX=%d, but kXFinder[jj=%d]=%d!\n",
      iOneX, jj, kXFinder[jj]);
  }

  if (jj < 0 || jj >= OnKappaS) {
    Rf_error("Error: BayesSpikeCpp():::ReweightedOneXtX cannot free jj = %d\n", jj);
  }
  iWeightedXtX[jj] = 0;
  int One = 1;  double ZeroD = 0.0; double OneD = 1.0;
  int GI = kXFinder[jj] * n;
  if (n < p) {
     //jj = XLC[iOneX];
     if (jj < 0) {
       Rf_error("ReweightedOneXtX, asked to update iOneX = %d, but jj = %d\n",
         iOneX, jj);
     }
     F77_CALL(dscal)(&p, &ZeroD, pXtX[jj], &One);
     for (ii = 0; ii < n; ii++) {
       Factor = iiWeight[ii] * REAL(sX)[ii + GI];
       if (Factor != 0) {
         F77_CALL(daxpy)(&p, &Factor, REAL(sX) + ii, &n,
           pXtX[jj], &One);
       }        
     }
     iWeightedXtX[jj] = 1;
     return(1);
   }

   if (WW == NULL) {
     RMemGetD(WW,"WW", n);
   }      

   for (ii = 0; ii < n; ii++) {
     WW[ii] = REAL(sX)[GI] * iiWeight[ii];  GI++;
   }
   F77_CALL(dgemv)(&Trans, &n, &p, &OneD, REAL(sX), &n, WW, &One,
    &ZeroD, pXtX[jj], &One); 
   iWeightedXtX[jj] = 1;
   return(1); 
}

///////////////////////////////////////////////////////////////////////////////
//   int BayesSpikeCL::ReweightEigenvalues()
//
//     When we are performing a weighted regression (T-noise or robit)
//       we must reweight the eigenvalues and eigenvectors of groups
//       to perform group selection.
//
int BayesSpikeCL::ReweightEigenvalues() {
  SEXP sEigenVectorMatrix;
  if (Verbose >= 4) {
    Rprintf("BayesSpikeCL: ReweightEigenvalues start\n"); R_FlushConsole();
  }
  int Start; int One = 1; int Length = 0;
  double ZeroD = 0.0; 
  double OneD = 1.0;

  int Info = 0;
  char upperlower = 'U';   char Jobz = 'V';  char Trans = 'T';
  int LWork = 0; int MaxLWork = 0;  if (WW != NULL) { MaxLWork = n; }
  int ii, jj,ke;
  for (ke = 0; ke< Rf_length(tauEndList); ke++)  {

    if (ke == 0) {
        Start = iFirstRandom;
    }  else {
        Start = ANINT(tauEndList, (ke-1)) +1;
    }
    Length = ANINT(tauEndList, (ke))- Start + 1;
    if (Start + Length > p) {
      Rf_error("BayesSpikeCL: ReweightEigenValues On tt = %d/%d: Error Start=%d, Length=%d, p=%d\n",
        ke, Rf_length(tauEndList), Start, Length, p);
    }
    if (Verbose >= 5) {
      Rprintf("BayesSpikeCL: ReweightEigenvalues On tt = %d/%d:  Start=%d, Length=%d\n", 
        ke, Rf_length(tauEndList), Start, Length); R_FlushConsole();
    }
    sEigenVectorMatrix = VECTOR_ELT(AllEigenVectors, ke);
    if (sEigenVectorMatrix == NULL) {
      Rf_error("sEigenVectorMatrix: Error, sEigenVectorMatrix for ke=%d/%d is NULL!\n", ke, Rf_length(tauEndList));
    } else if (!Rf_isReal(sEigenVectorMatrix)) {
      Rf_error("sEigenVectorMatrix: Error, sEigenVectorMatrix for ke=%d/%d is not REAL!\n", ke, Rf_length(tauEndList));
    } else if (Rf_length(sEigenVectorMatrix) != Length*Length) {
      Rf_error("sEigenVectorMatrix: Error: sEigenVector matrix for ke=%d/%d is length %d but Length is %d!\n",
        ke, Rf_length(tauEndList), Rf_length(sEigenVectorMatrix), Length);
    }
    if (REAL(sOnTau)[ke] > 0 && XLC[Start] >= 0) {
      if (Verbose >= 6) {
        Rprintf("BayesSpikeCL: we have XLC[Start=%d]=%d so we will try dcopy method. \n",
          Start, XLC[Start]);  R_FlushConsole();
      }
    
      for (jj = 0; jj < Length; jj++) {
        if (XLC[jj+Start] < 0 || XLC[jj+Start] >= OnKappaMem) {
          //Rf_error("BayesSpikeCL: we have fail because XLC[jj+Start]=%d but OnKappaMem=%d? \n",
          //  XLC[jj+Start], OnKappaMem);
          for (int kk = 0; kk < Length; kk++) {
            REAL(sEigenVectorMatrix)[Length *jj + kk] = 0;
            for (int jtj = 0; jtj < n; jtj++) {
              REAL(sEigenVectorMatrix)[Length *jj + kk] += iiWeight[jtj] * 
                REAL(sX)[(jj+Start)*n + jtj]*REAL(sX)[(k+Start)*n + jtj];
            }
          }
        } else if (pXtX[XLC[jj+Start]] == NULL) {
          Rf_error("BayesSpikeCL: we have fail because XLC[jj+Start]=%d gets null pXtX!\n",
            XLC[jj+Start]);
        } else {
          F77_CALL(dcopy)(&Length, pXtX[XLC[jj+Start]] + Start, &One, REAL(sEigenVectorMatrix) +
            jj * Length, &One);
        }
      }
      if (Verbose >= 6) {
        Rprintf("BayesSpikeCL:  Successful on dcopy for ke=%d/%d. \n", ke, Rf_length(tauEndList)); R_FlushConsole();
      }
      // Rf_error("Well, taking from XtX, now inspect the answer for ke = %d \n", ke); R_FlushConsole(); 
    } else {
      if (WW == NULL) {
        RMemGetD(WW,"WW", n);  MaxLWork = n;
      } 
      if (Verbose >= 6) {
        Rprintf("BayesSpikeCL: XLC[Start=%d] not present so going to (Start*n)=%d to Length Multiply \n",
          Start, Start*n); R_FlushConsole();
      }
      for (jj = 0; jj < Length; jj++) {
        for (ii = 0; ii < n; ii++) {
          WW[ii] = REAL(sX)[n *(jj+Start) + ii] * iiWeight[ii];
        }
        F77_CALL(dgemv)(&Trans, &n,&Length, &OneD, REAL(sX) + (Start) * n, &n, WW, &One,
          &ZeroD, REAL(sEigenVectorMatrix) + jj *Length, &One);
      }  
      //Rf_error("Well now inspect the answer for ke = %d \n", ke); R_FlushConsole();    
    }
    LWork = Length * 3;
    if (LWork > n && MaxLWork < LWork) {
      if (Verbose > 1) {
        Rprintf("BayesSpikeCL: ReweightEigenvalues, we have to resize WW matrix from %d to $d!\n", MaxLWork, LWork); R_FlushConsole();
      }    
      if (WW != NULL)  {FFree(WW, "WW");}
      RMemGetD(WW, "WW-LWork", LWork);
      MaxLWork = LWork;
    }
    if (Verbose >= 6) {
      Rprintf("BayesSpikeCL: ReweightEigenValues: ke=%d/%d now for dsyev. \n",
        ke, Rf_length(tauEndList)); R_FlushConsole();
    }
    F77_CALL(dsyev)(&Jobz, &upperlower, &Length, REAL(sEigenVectorMatrix), &Length, 
      REAL(VECTOR_ELT(AllEigenValues, ke)), WW, &LWork, &Info );
      
    if (Verbose >= 6) {
      Rprintf("BayesSpikeCL: ReweightEigenValues: ke=%d/%d we successfully dsyev fill in now!\n",
        ke, Rf_length(tauEndList));
    }
    for (jj=0;jj < (int) Length /2; jj++) {
      for (ii = 0; ii < Length; ii++) {
        REAL(sEigenVectorMatrix)[jj*Length+ii] +=
          REAL(sEigenVectorMatrix)[(Length-jj-1)*Length+ii];
        REAL(sEigenVectorMatrix)[(Length-jj-1)*Length+ii] =
          REAL(sEigenVectorMatrix)[jj*Length+ii]-
          REAL(sEigenVectorMatrix)[(Length-jj-1)*Length+ii];
        REAL(sEigenVectorMatrix)[(jj)*Length+ii] =
          REAL(sEigenVectorMatrix)[jj*Length+ii]-
          REAL(sEigenVectorMatrix)[(Length-jj-1)*Length+ii];
      }
      double Swap = REAL(VECTOR_ELT(AllEigenValues,ke))[jj];
      REAL(VECTOR_ELT(AllEigenValues,ke))[jj] =
        REAL(VECTOR_ELT(AllEigenValues,ke))[Length-jj-1];
      REAL(VECTOR_ELT(AllEigenValues,ke))[Length-jj-1] = Swap;
    }
    if (Verbose >= 5) {
      Rprintf("BayesSpikeCpp:: ReweightEigenValues %d/%d we got a swap done for Length=%d. \n", ke, Rf_length(tauEndList), Length);
      R_FlushConsole();
    }
    if (Info != 0) {
      if (Verbose >= -2) {
        Rprintf("BayesSpikeCL:: Reweight Eigenvalues had a fail on ke = %d\n", ke);R_FlushConsole();
      }
    } 
  }
  if (Verbose >= 5) {
    Rprintf("BayesSpikeCL:: Successful Reweight of Eigenvalues. \n"); R_FlushConsole();
  }
  return(1);
}

// RefillXtX, if there is new weighting, all values of XtX are filled. 
int BayesSpikeCL::RefillXtX()  {
  int One = 1; double OneD = 1.0;  double ZeroD = 0.0;
  char Trans = 'T';
  if (Verbose >= 3) {
    Rprintf("BayesSpikeCpp.cpp:::RefillXtX, tt = %d/%d", tt, MaxGibbsIters); R_FlushConsole();
    Rprintf("   Checking integrity of pXtX"); R_FlushConsole();
    for (int iti = 0; iti < OnKappaMem; iti++) {
      *(pXtX[iti]) = 0.0;
      *(pXtX[iti] + p -1) = 0.0;
    }
    Rprintf("   Integrity Pass, OnKappaS=%d/%d\n", OnKappaS, OnKappaMem); R_FlushConsole();
  }
  if (OnKappaS <= 0) {
    if (Verbose >= 3) {
      Rprintf("BayesSpikeCpp.cpp:RefillXtX, OnKappaS=%d, no vectors. \n"); R_FlushConsole();
    }
    return(1);
  }
  int jj;  int ii = 0;
  if (iiWeight == NULL) {
    if (Verbose >= 3) {
      Rprintf("BayesSpikeCpp.cpp:::RefillXtX() doing iiWeight is null version. \n"); R_FlushConsole();
    }
    for (jj = 0; jj < OnKappaS; jj++) {
       if (kXFinder[jj] >= OnKappaS || kXFinder[jj] < 0) {
         Rf_error("RefillX: kXFinder is off mark \n"); R_FlushConsole();
       }
       F77_CALL(dgemv)(&Trans, &n, &p, &OneD, REAL(sX), &n, REAL(sX) +
         kXFinder[jj] * n, &One, &ZeroD, pXtX[jj], &One);
    }
  } else {
    double Factor = 0.0;  One = 1;
    for (ii = 0; ii < OnKappaS; ii++) {
      F77_CALL(dscal)(&p, &ZeroD, pXtX[ii], &One);
    }
    if (n < p) {
      if (Verbose >= 3) {
        Rprintf("BayesSpikeCpp.cpp:::RefillXtX() Refresh iiWeight, n < p version. \n"); R_FlushConsole();
      }
      One = 1;
      for (jj = 0; jj < OnKappaS; jj++) {
        for (int ii = 0; ii < n; ii++) {
          Factor = iiWeight[ii] * REAL(sX)[ii + kXFinder[jj]* n];
          if (Factor != 0) {
            F77_CALL(daxpy)(&p, &Factor, REAL(sX) + ii, &n,
              pXtX[jj], &One);
          }        
        }
        iWeightedXtX[jj] = 1;
      }
      return(2);
    } else {
      if (Verbose >= 3) {
        Rprintf("BayesSpikeCpp.cpp:::RefillXtX() doing iiWeight version, p > n \n"); R_FlushConsole();
      }  
      if (WW == NULL) {
        RMemGetD(WW,"WW", n);
      }      
      if (Verbose >= 3) {
        Rprintf("BayesSpikeCpp.cpp:::RefillXtX() doing iiWeight WW acquired. \n"); R_FlushConsole();
      }
      if (iWeightedXtX == NULL) {
        Rprintf("BayesSpikeCpp.cpp::iWeightedXtX is NULL Strange!\n"); R_FlushConsole();
        iWeightedXtX = (short int *) Calloc(OnKappaMem, short int);
      }
      if (Verbose >= 3) {
        Rprintf("BayesSpikeCpp.cpp:: Check Integrity of iWeightedXtX -- "); R_FlushConsole();
        for (int iti = 0; iti < OnKappaMem; iti++) { iWeightedXtX[ii] = (short int) 0; }
        Rprintf("  PASS. \n"); R_FlushConsole();
        Rprintf("BayesSpikeCpp.cpp:: Check Integrity of WW -- "); R_FlushConsole();
        for (int iti = 0; iti < n; iti++) { WW[ii] = 0.0; }
        Rprintf("  PASS. \n"); R_FlushConsole();
      }
      One = 1;
      for (jj = 0; jj < OnKappaS; jj++) {
        int GI = kXFinder[jj] * n;
        for (ii = 0; ii < n; ii++) {
          WW[ii] = REAL(sX)[GI] * iiWeight[ii]; GI++;
        }
        Trans = 'T';
        F77_CALL(dgemv)(&Trans, &n, &p, &OneD, REAL(sX), &n, WW, &One,
          &ZeroD, pXtX[jj], &One); 
        iWeightedXtX[jj] = 1;
      }
      return(3);
    }
  }
  return(1);
}

int BayesSpikeCL::UpdateNonFreshXtResid() {
  int Verbose = this->Verbose -2;
  int ii;
  int One = 1;
  int CNZero = 0;
  for (ii = 0; ii < p; ii++) {
    if (REAL(sBeta)[ii] != 0) {
      CNZero++;
    }
  }

  double NegOne = -1.0;  double NegCons = 0.0;
  char Trans = 'N'; double OneD = 1.0; double ZeroD = 0.0;
  
  if (iiWeight == NULL) {
    if (n < CNZero) {
      if (WW == NULL) {
         RMemGetD(WW, "WW", n);
      } 
      F77_CALL(dcopy)(&n, REAL(sY), &One, WW, &One);
      for (ii = 0; ii < p; ii++) {
        if (REAL(sBeta)[ii] != 0) {
          NegCons = REAL(sBeta)[ii] * -1.0;
          F77_CALL(daxpy)(&n, &NegCons, REAL(sX) + ii *n, &One, WW, &One);
        }
      }
      Trans = 'T';
      F77_CALL(dgemv)(&Trans, &n, &p, &OneD, REAL(sX), &n, WW, &One, &ZeroD,
        XtResid, &One); 
      return(4);   
    }
    F77_CALL(dcopy)(&p, XtY, &One, XtResid, &One);
    for (ii = 0; ii < p; ii++) {
      if (REAL(sBeta)[ii] != 0) {
        if (XLC[ii] >= 0 && XLC[ii] < p) {
          NegCons = -1.0 * REAL(sBeta)[ii];
          F77_CALL(daxpy)(&p, &NegCons, pXtX[XLC[ii]], &One, XtResid, &One);        
        } else {
          if (WW == NULL) {
            RMemGetD(WW, "WW", n);
          }
          Trans = 'N';
          F77_CALL(dscal)(&n, &ZeroD, WW, &One);
          F77_CALL(daxpy)(&n, REAL(sBeta) + ii, REAL(sX) + n * ii, 
            &One, WW, &One);
          Trans = 'T';
          NegOne = -1.0;
          F77_CALL(dgemv)(&Trans, &n, &p, &NegOne, REAL(sX), &n, WW, &One,
            &OneD, XtResid, &One);
        }
      }
    } 
  } else {
    if (n < CNZero) {
      if (WW == NULL) {
         RMemGetD(WW, "WW", n);
      } 
      for (ii = 0; ii < n; ii++) {
        WW[ii] = iiWeight[ii] * REAL(sY)[ii];
      }
      for (ii = 0; ii < p; ii++) {
        if (REAL(sBeta)[ii] != 0) {
          NegCons = REAL(sBeta)[ii] * -1.0;
          F77_CALL(daxpy)(&n, &NegCons, REAL(sX) + ii *n, &One, WW, &One);
        }
      }
      Trans = 'T';
      F77_CALL(dgemv)(&Trans, &n, &p, &OneD, REAL(sX), &n, WW, &One, &ZeroD,
        XtResid, &One); 
      return(4);   
    }
    if (iWeightedXtX == NULL && OnKappaS > 0) {
      Rf_error("UpdateNonFresh: Uh Oh, iWeightedXtX is Null, not helpful.");
    }
    F77_CALL(dcopy)(&p, XtY, &One, XtResid, &One);
    for (ii = 0; ii < p; ii++) {
      if (REAL(sBeta)[ii] != 0) {
        if (XLC[ii] >= 0 && XLC[ii] < OnKappaS) {
          if (iWeightedXtX[XLC[ii]] == 0) {
            ReweightedOneXtX(ii);
            if (iWeightedXtX[XLC[ii]] == 0) {
              Rf_error("BayesSpikeCpp.cpp:::UpdateNonFresh() we tried to Update XLC[ii=%d]=%d, but got %d!\n",
                ii, XLC[ii], (int) iWeightedXtX[XLC[ii]]);
            }
          }
          NegCons = -1.0 * REAL(sBeta)[ii];
          F77_CALL(daxpy)(&p, &NegCons, pXtX[XLC[ii]], &One, XtResid, &One);        
        } else {
          if (WW == NULL) {
            RMemGetD(WW, "WW", n);
          }
          //Trans = 'N';
          //F77_CALL(dscal)(&n, &ZeroD, WW, &One);
          for (int jj = 0; jj < n; jj++) {
            WW[jj] = REAL(sX)[jj + ii * n] * iiWeight[jj];
          }
          Trans = 'T';
          NegCons = - 1.0 * REAL(sBeta)[ii];
          F77_CALL(dgemv)(&Trans, &n, &p, &NegCons, REAL(sX), &n, WW, &One,
            &OneD, XtResid, &One);
        }
      }
    }   
  }
  return(1);
}

/////////////////////////////////////////////////////////////////////////////
//  UpdateFreshXtResid()
//
//    Given that XtResid needs to be updated again, uses data in XtX currently 
//   to get there
//
//  UpdateNonFreshXtResid is called if there are members of Beta we don't need
//   in memory that are nonzero
//
//
int BayesSpikeCL::UpdateFreshXtResid() {
  int Verbose = this->Verbose - 3;
  //Verbose = 10;
  if (Verbose > 2) {
    Rprintf("UpdateFreshXtResid: Starting \n"); R_FlushConsole();
  }
  if (dfTNoise > 0 && Verbose >= 1) {
    Rprintf("UpdateFreshXtResid: we have a dfTNoise =%f update. \n", dfTNoise);
    R_FlushConsole();
  }
  if (iiWeight != NULL) {
    if ((Verbose >= 2) || (dfTNoise > 0 && Verbose >= 1)) {
      Rprintf("UpdateFreshXtResid: we have to back reweight XtX\n");
    }
    BackReweightOnlyActiveXtX();
  }
  int ContiguousEnd = 0; int ContiguousStart = 0; //int Distance = 0;
  //int Start = 0;
  //double OneD = 1.0;
  if (dfTNoise > 0 && Verbose >= 1) {
    Rprintf("UpdateFreshXtResid: we are updating XtY into XtResid. \n"); R_FlushConsole();
  }
  int One = 1;
  F77_CALL(dcopy)(&p, XtY, &One, XtResid, &One);
  if (OnKappaS == 0 && NumActive <= 0) {
    return(0);
  }
  //char Trans = 'N';
  //double NegOneD = -1.0; 
  if (NumActive <= 0) {
    return(1); 
  }
  double Factor = 0;
  int ii = 0;  int jj = 0;
  if (dfTNoise > 0 && Verbose >= 1) {
    Rprintf("UpdateFreshXtResid: Going to sum through NumActive = %d. \n", NumActive); R_FlushConsole();
  }
  for (int ii = 0; ii < NumActive; ii++) {
    if (REAL(sBeta)[OrderedActive[ii]] != 0.0 && 
      !R_isnancpp(REAL(sBeta)[OrderedActive[ii]]) &&
      R_finite(REAL(sBeta)[OrderedActive[ii]])) {
      Factor = -1.0 * REAL(sBeta)[OrderedActive[ii]];
      if (XLC[OrderedActive[ii]] < 0) {
        Rprintf("** ERROR UpdateFreshxtResid, XLC[OrderedActive[ii=%d]=%d] = %d!\n",
          ii, OrderedActive[ii], XLC[OrderedActive[ii]]); R_FlushConsole(); 
        for (int kk = 0; kk < p; kk++) {
          for (int iti = 0; iti < n; iti++) {
            if (iiWeight == NULL) {
              XtResid[kk] -= REAL(sBeta)[OrderedActive[ii]] *
              REAL(sX)[OrderedActive[ii]*n + iti] *
              REAL(sX)[kk*n + iti]; 
            } else {
              XtResid[kk] -= REAL(sBeta)[OrderedActive[ii]] *
              REAL(sX)[OrderedActive[ii]*n + iti] *
              iiWeight[iti] *
              REAL(sX)[kk*n + iti]; 
            }
          }
        }
        //Rf_error("ERROR UpdateFreshXtResid XLC < 0\n");
      } else {
        F77_CALL(daxpy)(&p, &Factor, pXtX[XLC[OrderedActive[ii]]], 
          &One, XtResid, &One);
      }
    } else {
        REAL(sBeta)[OrderedActive[ii]] = 0.0;
    }
  }
  
  //if (tt <= 5) {
    //Rprintf("*** UpdateFreshXtResid: We are testing!, tt = %d\n", tt);
    //int ACount = TestXtResid();
    //if (ACount > 0) {
    //  Rprintf("*** BayesSpikeCpp.cpp(): ERROR, UpdateFrestXtResid generates an error = %d.  Seriously?\n", ACount);
    //  Rf_error("  UpdateFreshXtResid Fail. \n");
    //}
    //Rprintf("*** UpdateFreshXtResid:: Pass, tt = %d\n", tt); R_FlushConsole();
  //}
  if (Verbose > 2) {
    Rprintf("UpdateFreshXtResid: Finished  \n"); R_FlushConsole();
  }
  return(1);
  // None of this is a good idea, Alan.
  //
  //  Especially now that pXtX is broken, we do not allow this code to be run.
  for (ii = 0; ii < NumActive; ii++) {
    if (Verbose > 2) {
      Rprintf("UpdateFreshXtResid: ii = %d, OnKappaS = %d \n", ii, OnKappaS); R_FlushConsole();
    }
    ContiguousStart = kXFinder[ii];
    if (kXFinder[ii] < 0 || kXFinder[ii] >= p) {
    } else if (REAL(sBeta)[kXFinder[ii]] == 0.0 ||
      REAL(sBeta)[kXFinder[ii]] != REAL(sBeta)[kXFinder[ii]]) {
    } else if (ii < OnKappaS-1) {
      ContiguousEnd = ContiguousStart+1; 
      jj = ii + 1;
      if (ii+1 < OnKappaS-1) {
        for (jj = ii+1; jj < OnKappaS-1; jj++) {
          if (REAL(sBeta)[kXFinder[jj+1]] == 0.0 ||
            (kXFinder[jj+1] != ContiguousEnd+1) ) {
            break;
          } else {
            ContiguousEnd++;
          }
        }
      }
      //Rprintf("Contiguous part found between ii =%d, jj = %d, CS = %d, CE = %d\n",
      //  ii, jj, ContiguousStart, ContiguousEnd); R_FlushConsole();
      //Start = ii * p;  //Distance = ContiguousEnd - ContiguousStart +1;
      // Bad Contiguous code!  XtX is not contiguous.
      //F77_CALL(dgemv)(&Trans, &p, &Distance, &NegOneD, pXtX[Start], &p,
      //  REAL(sBeta) + ContiguousStart, &One, &OneD, XtResid, &One); 
      ii = jj;
    } else {
      Factor = -1.0 * REAL(sBeta)[kXFinder[ii]];
      F77_CALL(daxpy)(&p, &Factor, pXtX[ii], &One, XtResid, &One);
    }                            
  }
  if (Verbose > 2) {
    Rprintf("UpdateFreshXtResid: Finished  \n"); R_FlushConsole();
  }
  return(1);
}

int BayesSpikeCL::SetupBFixedAndFriends() {
  int Verbose = this->Verbose -1;
  if(Verbose > 2) {
    Rprintf("SetupBFixedAndFriends():  Starting, free OrderedActive\n");
  }
  FFree(OrderedActive, "OrderedActive");
  if(Verbose > 2) {
    Rprintf("SetupBFixedAndFriends():  Starting, free BFixed\n");
  }
  FFree(BFixed, "BFixed"); 
  int TiFirstRandom  = this->iFirstRandom;
  if (tauEndList == NULL || Rf_isNull(tauEndList) || Rf_length(tauEndList) <= 0
    || this->iFirstRandom < 0) {
    if (Verbose >= 1) {
      Rprintf("SETUP B Fixed And Friends Issue!, tauEndList == NULL!\n");
      R_FlushConsole();
    }
    TiFirstRandom = p;
  } else if (this->iFirstRandom < 0) {
    TiFirstRandom = p;  
  }
  if(Verbose > 2) {
    Rprintf("SetupBFixedAndFriends(): GetMemory\n");
  }
  RMemGetS(BFixed, "BFixed",TiFirstRandom);
  RMemGetI(OrderedActive, "OrderedActive", OnKappaMem+1);
  int ii;
  
  for (ii = 0; ii < TiFirstRandom; ii++) {
    if (REAL(sBeta)[ii] != 0 && !R_isnancpp(REAL(sBeta)[ii])) {
      BFixed[ii] = 1;  NewFixedCoords++;
    } else {BFixed[ii] = 0;}
  }
  if(Verbose > 2) {
    Rprintf("SetupBFixedAndFriends(): StartRefreshOrderedActive\n");
  }
  if (NewFixedCoords > 0) {
    AddAllNewCoords();
  }
  RefreshOrderedActive(0);
  return(1);
}

////////////////////////////////////////////////////////////////////////////////
//  int BayesSpikeCL::UpdateBFixed()
//
//    Operation where Fixed Parameters are chosen for inclusion
//    However, it depends on tail properties of the indicators.
//
//    I don't think this gets used.
int BayesSpikeCL::UpdateBFixed() {
  int ii = 0;
  int EndList = iFirstRandom;
  if (sOnTau == NULL || Rf_isNull(sOnTau) || Rf_length(sOnTau) <= 0 ||
    tauEndList == NULL || Rf_isNull(tauEndList) || Rf_length(tauEndList) <= 0) {
    EndList = p;  
  } else if (iFirstRandom < 0) {
    EndList = p;
  }
  if (EndList > 0) {
    for (ii = 0; ii < EndList; ii++) {
      if (REAL(sBeta)[ii] != 0.0) {
        BFixed[ii] = 1;
      } else {
        BFixed[ii] = 0;
      }
    }
  }
  return(1);
}
int BayesSpikeCL::RefreshOrderedActive( int ForceFill ) {
  int Verbose = this->Verbose - 2;
  int ii = 0;  int COA = 0; 
  int RefreshNumberNew = 0;  
  if (Verbose > 1) {
    Rprintf("BayesSpikeCL:  Refressing OrderedActive, Force = %d \n ", ForceFill); R_FlushConsole();
  }
  //if (tt <= 1) {
  //  Rprintf("BayesSpikeCpp.cpp::RefreshOrderedActive(tt=%d), we will always test first!\n", tt); 
  //  R_FlushConsole();
  //}
  //if (NumActive >= 1 && OrderedActive != NULL  && ForceFill <= 1) {
    //int ATest = TestOrderedActive();
    //if (ATest > 0) {
    //  Rprintf("BayesSpikeCpp.cpp:RefreshOrderedActive(tt=%d) ERROR, we have ATest on test is %d!\n", tt, ATest);
    //  Rf_error(" **  Bad RefreshOrderedActive Start!\n"); 
    //}
  //}
  if (OrderedActive == NULL) {
    RMemGetI(OrderedActive, "OrderedActive", OnKappaMem+1);
  }
  
  
  int EndList = iFirstRandom;
  if (tauEndList == NULL || Rf_isNull(tauEndList) || Rf_length(tauEndList) <= 0 ||
    sOnTau == NULL || Rf_isNull(sOnTau) || Rf_length(sOnTau) <= 0) {
    EndList = p;
  } else {
    EndList = iFirstRandom;
  }
  if (Verbose > 1) {
    Rprintf("RefreshOrdered Active: BFixed = ");
    if (EndList > 0) {
    PrintVector(BFixed, EndList);
    R_FlushConsole();
    } else {
      Rprintf("  No BFixed ! \n"); R_FlushConsole();
    }
    if (!Rf_isNull(sOnTau)  && !Rf_isNull(tauEndList) && Rf_length(tauEndList) >= 1) {
      Rprintf("   And sOnTau = "); 
      PrintVector(REAL(sOnTau), Rf_length(tauEndList));
    } else {
      Rprintf("  But sOnTau is Null "); R_FlushConsole();
    }
    Rprintf("\n"); R_FlushConsole();
  }
  if (ForceFill == 0 && EndList > 0) {
    for (ii = 0; ii < EndList; ii++) {
       if (BFixed[ii] > 0 && XLC[ii] >= 0) {
         if (COA >= OnKappaMem) {
           Rf_error("RefreshOrderedActive: on BFixed, COA = %d, OnKappaMem = %d, not possible",
             COA, OnKappaMem);
         }
         if (OrderedActive[COA] != ii) {
           RefreshNumberNew++;
         }
         OrderedActive[COA] = ii; COA++;
       }
    }
    NumActiveFixed = COA;  NumActiveTau = 0;
    NewFixedCoords = 0;  AllNewCoords = 0;
    if (!Rf_isNull(tauEndList) && Rf_length(tauEndList) >= 1) {
      int jj;  int St = 0;
      for (jj = 0; jj < Rf_length(sOnTau); jj++ ) {
        if (jj == 0) {
          St = iFirstRandom;
        } else { St = ANINT(tauEndList, (jj-1) )+1; }
        if (REAL(sOnTau)[jj] > 0 && XLC[St] >= 0) {
          NumActiveTau++;
          for (ii = St;  ii <= ANINT(tauEndList, jj); ii++) {
            if (COA >= OnKappaMem) {
              Rf_error("RefreshOrderedActive: on Tau, COA = %d, OnKappaMem = %d, not possible",
                COA, OnKappaMem);
            }
            if (OrderedActive[COA] != ii) {
              RefreshNumberNew++;
            }
            OrderedActive[COA] = ii;  COA++;
          }
        }
      }
    }
  } else {
    NewFixedCoords = 0; AllNewCoords = 0;
    if (EndList > 0) {
    for (ii = 0; ii < EndList; ii++) {
      if (BFixed[ii] > 0 && XLC[ii] < 0) {
        NewFixedCoords++;  AllNewCoords++;
      }
    } }
    if (!Rf_isNull(tauEndList) && Rf_length(tauEndList) > 0) {
      int St = iFirstRandom;
      for (int jj = 0; jj < Rf_length(tauEndList); jj++) {
        if (REAL(sOnTau)[jj] > 0 && XLC[St] < 0) {
          AllNewCoords += ANINT(tauEndList, jj)+1- St;
        }
        St = ANINT(tauEndList, jj)+1;
      }
    }
    if (AllNewCoords > 0) {
      Rprintf("Inside RefreshOrderedActive, tt = %d, we still found %d Fixed and %d total coords to add, why wasn't Add all Called?\n",
        tt, NewFixedCoords, AllNewCoords);
      AddAllNewCoords();
    }
    if (EndList > 0) {
    for (ii = 0; ii < EndList; ii++) {
       if (BFixed[ii] > 0) {
         if (XLC[ii] < 0) { AddCoord(ii); }
         if (COA >= OnKappaMem) {
           Rf_error("RefreshOrderedActive: on BFixed, COA = %d, OnKappaMem = %d, not possible",
             COA, OnKappaMem);
         }
         if (OrderedActive[COA] != ii) {
           RefreshNumberNew++;
         }
         OrderedActive[COA] = ii;  COA++;
       }
    }}
    NumActiveFixed = COA; NumActiveTau = 0;
    if (!Rf_isNull(tauEndList)  && Rf_length(tauEndList) >= 1) {
      int jj;  int St = 0;
      for (jj = 0; jj < Rf_length(sOnTau); jj++ ) {
        if (jj == 0) {
          St = iFirstRandom;
        } else { St = ANINT(tauEndList, (jj-1) )+1; }
          if (REAL(sOnTau)[jj] > 0) {
            NumActiveTau++;
            for (ii = St;  ii <= ANINT(tauEndList, jj); ii++) {
              if (XLC[ii] < 0) {
                Rprintf("RefreshOrderedActive: tt=%d/%d, We are adding XLC[ii=%d], which really shouldn't happen here\n",
                  tt, MaxGibbsIters, ii);  R_FlushConsole();
              }
              if (XLC[ii] < 0) { AddCoord(ii); }
              if (COA >= OnKappaMem) {
                Rf_error("RefreshOrderedActive: on Tau, COA = %d, OnKappaMem = %d, not possible",
                  COA, OnKappaMem);
              }
              if (OrderedActive[COA] != ii) {
                RefreshNumberNew++;
              }
              OrderedActive[COA] = ii; COA++;
          }
        }
      }
    }  
  
  }
  NumActive = COA;
  if (Verbose > 1) {
    int OtherCount = 0;
    int jj; int St;
    if (EndList >= 1) {
    for (ii = 0; ii < EndList; ii++) {
      if (BFixed[ii] > 0) { OtherCount++;}
    }
    }
    int OnTauCount = 0;
    if (tauEndList != NULL && Rf_length(tauEndList) > 0 &&
      !Rf_isNull(tauEndList)) {
      for (jj = 0; jj < Rf_length(tauEndList); jj++) {
        if (REAL(sOnTau)[jj] > 0) {
          if (jj == 0) { St = EndList; } else {
            St = ANINT(tauEndList, jj-1) + 1;
          }
          OnTauCount += ANINT(tauEndList, jj) - St + 1;
        }
      }
    }
    Rprintf("After Refresh Ordered Active, COA = %d, NumActiveFixed = %d \n", COA, NumActiveFixed);
    Rprintf("  However, OtherCount = %d, OnTauCount = %d \n", OtherCount, OnTauCount);
    if (ForceFill > 0 && COA != OtherCount + OnTauCount) {
      Rf_error("Error: So something went wrong!");
    }
    R_FlushConsole();  
  }
  if (RefreshNumberNew > 0 && NumberNew <= 0) {
    NumberNew = RefreshNumberNew;
  }
  if (ForceFill == 1 && NumActive >= 1) {
    double TestT = 0;   int One = 1;
    if (iiWeight == NULL) {
      TestT =  F77_CALL(ddot)(&n,  REAL(sX), &One,
        REAL(sX) + n * kXFinder[0], &One);
      if (fabs(TestT - *(pXtX[0]+0)) > .0001) {
        Rprintf("Hmm: TestT after RefreshOrderedActive suggests that pXtX is not configured, resetting!\n"); R_FlushConsole();
      }
      char Trans = 'T'; double OneD = 1.0;  double ZeroD = 0.0;
      for (ii = 0; ii < NumActive; ii++) {
        F77_CALL(dgemv)(&Trans, &n, &p, &OneD, REAL(sX), &n,
          REAL(sX) + kXFinder[ii] * n, &One, &ZeroD, pXtX[ii], &One);
      }
    } else {
      TestT = 0; 
      for (ii = 0; ii < n; ii++) {
        TestT+= REAL(sX)[ii] * iiWeight[ii] * REAL(sX)[ii+ n * kXFinder[0]];
      }
      if (fabs(TestT- *(pXtX[0]+0)) > .0001) {
        Rprintf("Hmm: Weighted RefreshOrderedActive on ForceFill suggests pXtX is not configured. resetting!.\n"); R_FlushConsole();
      }
      BackReweightOnlyActiveXtX();
    }
  }
  //if (tt <= 2) {
  //  Rprintf("** Note RefreshOrderedActive, tt = %d, we are testing OrderedActive afterward!\n", tt); R_FlushConsole();
  //}
  //int ACount = TestOrderedActive();
  //if (ACount >= 1) {
  //  Rprintf("** ERRROR IN TestOrderedActive after RefreshOrderedActive. ACount = %d, tt=%d\n", ACount, tt); R_FlushConsole();
  //  Rf_error("** Go Catch this error!\n");
  //}
  return(NumActive);
}

int BayesSpikeCL::TestOrderedActive() {
  int OnGo = 0;
  int TotalAdds = 0;
  for (int ii = 0; ii < p; ii++) {
    while (OnGo < NumActive && OrderedActive[OnGo] < ii) { OnGo++; }
    if (REAL(sBeta)[ii] != 0.0 && (OnGo >= NumActive || OrderedActive[OnGo] > ii)) {
      TotalAdds++;
    }
  }

  int Will10 = 0;
  if (TotalAdds > 0)  {
  Rprintf("BayesSpikeCpp.cpp:::TestOrderedACtive: We Got TotalAdds = %d, and NumActive = %d, lets figure it out!\n", TotalAdds, NumActive);
  R_FlushConsole();
  OnGo = 0;
  Will10 = 0;
  for (int ii = 0; ii < p; ii++) {
    while (OnGo < NumActive && OrderedActive[OnGo] < ii) { OnGo++; }
    if (REAL(sBeta)[ii] != 0.0 && (OnGo >= NumActive || OrderedActive[OnGo] > ii)) {
      if (Will10 >= 10) { Rprintf("\n");  Will10 = 0; }
      Rprintf("%d:%.4f, ", ii, REAL(sBeta)[ii]); R_FlushConsole(); Will10++;
    }
  }
  Rprintf("\n");
  Rprintf("** You Should Seek out this Error!\n"); 
  Rprintf("Error Error On BayesSpikeCpp.cpp:::TestOrderedActive: A Fail!\n");  R_FlushConsole();
  return(TotalAdds);
  }                   
  if (RsOnTau == NULL ||  Rf_isNull(tauEndList) || !(Rf_isReal(tauEndList) ||
    Rf_isInteger(tauEndList)) || Rf_length(tauEndList) <= 0 ||
    !Rf_isReal(sOnTau) || Rf_length(sOnTau) != Rf_length(tauEndList)) {
    return(0);
  }
  int St = iFirstRandom;
  TotalAdds = 0;
  for (int ii = 0; ii < Rf_length(tauEndList); ii++) {
    if (REAL(sOnTau)[ii] == 0.0) {
    for (int jSt = St; jSt <= ANINT(tauEndList, ii); jSt++) {
      if (REAL(sBeta)[jSt] != 0.0) {
         TotalAdds++;
      }
    }
    }
    St = ANINT(tauEndList,ii)+1;
  }
  if (TotalAdds <= 0) {
    return(0);
  }
  Rprintf("ERROR: TestOrderedActive, it seems that sOnTau is zero but has non zero elements for %d elements!\n", TotalAdds);
  for (int ii = 0; ii < Rf_length(tauEndList); ii++) {
    if (REAL(sOnTau)[ii] == 0.0) {
    for (int jSt = St; jSt <= ANINT(tauEndList, ii); jSt++) {
      if (REAL(sBeta)[jSt] != 0.0) {
         if (Will10 >= 10) { Rprintf("\n"); R_FlushConsole(); Will10++; }
         Rprintf("OnT=%d:%d:%f, ", ii, jSt, REAL(sBeta)[jSt]); R_FlushConsole();
      }
    }
    }
    St = ANINT(tauEndList,ii)+1;
  }
  Rprintf("ERROR: TestOrderedActive(): Investigate this Crime!\n"); R_FlushConsole();
  return(TotalAdds);
}
int BayesSpikeCL::AddCoord(int NewCoord) {
  int Verbose = this->Verbose -4;
  //if (tt == 1) { Rprintf("+"); R_FlushConsole(); }
  if (this->pXtX == NULL) {
    Rf_error("BayesSpikeCL:AddCoord this->pXtX is NULL!\n");
  }
  if (NewCoord < 0 || NewCoord >= p) {
    Rf_error("BayesSpikeCL: Can't Add Coord be cause it is wrong! %d\n", NewCoord);
  }
  if (XLC[NewCoord] >= 0) {
    Rf_error("BayesSpikeCL: Can't AddCoord that's already filled: %d, %d",
      NewCoord, XLC[NewCoord]);
  }

  double Factor = 0;
  if (OnKappaS >= OnKappaMem && OnKappaMem < p) {
     if (OnKappaMem * 2.0 > (double) p) {
       Factor = ((double) p+1) / ( (double) OnKappaMem);
     } else {
       Factor = 2.0;
     }
     if (Verbose >= 0 && p >= 100000) {
       Rprintf("** BayesSpikeCpp::AddCoord, tt=%d, adding %d, but OnKappaS =%d, OnKappaMem = %d, we are going to have to Resize.\n",
         tt, NewCoord, OnKappaS, OnKappaMem); R_FlushConsole();
     }
     Resize(Factor);
     if (Verbose >= 0 && p >= 100000) {
       Rprintf("\n*** BayesSpikeCpp.cpp::AddCoord, just resized to OnKappaMem=%d, OnKappaS=%d, Adding NewCoord=%d\n", 
         OnKappaMem, OnKappaS, AllNewCoords);
     }
  }
  //if (tt == 1) { Rprintf("m"); R_FlushConsole(); }
  if (OnKappaS >= OnKappaMem || OnKappaS >= p) {
    Rprintf("Add Coord Error: Note, we tried to do an add but we failed to AddCoord. \n"); R_FlushConsole();
    Rprintf("   OnKappaMem = %d, OnKappaS = %d, p=%d. \n", OnKappaMem, OnKappaS, p); R_FlushConsole();
    Rf_error("AddCoord: We fail badly. \n");
  }

  XLC[NewCoord] = OnKappaS;
  kXFinder[OnKappaS] = NewCoord;
  if (Verbose > 2) {
    Rprintf("AddCoord: About to Add coordinate %d, OnKappaS = %d\n", 
      NewCoord, OnKappaS); R_FlushConsole();
  }
  double ZeroD = 0.0; double OneD = 1.0; int One = 1;
  //OneD = 0.0;  // Temporary error fix
  //if (tt == 1) { Rprintf("b"); R_FlushConsole(); }
  char Trans = 'T';
  if (NewCoord*n >= Rf_length(sX) || NewCoord * n < 0) {
    Rf_error("AddCoord, NewCoord=%d, n=%d, length sX=%d, not going!\n",
      NewCoord, n, Rf_length(sX));
  }
  /*
  if (tt == 1) {
    Rprintf(" --- Check for AddCoord Fail, tt = %d, Newcoord=%d/%d, OnKappaS=%d/%d \n",
      tt, NewCoord, p, OnKappaS, OnKappaMem); R_FlushConsole();
    Rprintf(" --- NewCoord=%d, n = %d, NewCoord * n = %d \n", NewCoord, n, NewCoord*n);
    R_FlushConsole();
    Rprintf(" --- First Members of REAL(sX)[0-3] = "); R_FlushConsole();
    Rprintf(" %.3f, %.3f, %.3f, %.3f \n", REAL(sX)[0], n > 1 ? REAL(sX)[1] : 666.0, 
      n > 2 ? REAL(sX)[2]: 666.0 , n > 3 ? REAL(sX)[3] : 666.0); R_FlushConsole();
    Rprintf(" --- First Members of REAL(sX)[n*NewCoord:%d+0-3] = ", n*NewCoord); R_FlushConsole();
    if (n + n *NewCoord < n * p) {
    Rprintf(" %.3f, %.3f, %.3f, %.3f \n", REAL(sX)[n *NewCoord+0], n > 1 ? REAL(sX)[n *NewCoord+1] : 666.0, 
      n > 2 ? REAL(sX)[n *NewCoord+2]: 666.0 , n > 3 ? REAL(sX)[n *NewCoord+3] : 666.0); R_FlushConsole();
    }
    int COffs = 0;
    for (int iti = 0; iti < OnKappaMem; iti++) {
      if (pXtX[iti] == NULL) {
        Rprintf(" --- Oh Now on AddCoord(NewCoord=%d), we have null pXtX[iti=%d], OnKappaS=%d/%d\n",
          iti, OnKappaS, OnKappaMem); R_FlushConsole();
        COffs++;
      } 
    }
    if (COffs >= 1) {
      Rprintf("ERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERROR\n");
      Rprintf(" -- Error AddCoord(NewCoord=%d), OnKappaS=%d/%d, we have %d null pXtX!\n",
        NewCoord, OnKappaS, OnKappaMem, COffs);
      Rprintf("ERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERROR\n");
        R_FlushConsole();
      Rf_error(" --- AddCoord(NewCoord=%d) error found OnKappaS=%d/%d, COffs = %d!\n",
        NewCoord, OnKappaS, OnKappaMem, COffs);
    }
    if (OnKappaS >= 0) {
    Rprintf(" --- Okay lets check out pXtX[OnKappaS=%d/%d]", OnKappaS, OnKappaMem); R_FlushConsole();
    if (this->pXtX[OnKappaS] == NULL) {
      Rf_error("Error: pXtX[%d], this is not good we have NULL XtX!\n", OnKappaS);
    }
    Rprintf(" --- pXtX[OnKappaS=%d][0] is going to be ", OnKappaS); R_FlushConsole();
    Rprintf(" %f \n", *(pXtX[OnKappaS]+0)); R_FlushConsole();
    Rprintf(" --- pXtX[OnKappaS=%d][0-3] = ", OnKappaS); R_FlushConsole();
    Rprintf(" %.3f, %.3f, %.3f, %.3f \n",
      *(pXtX[OnKappaS]+0), n > 2 ? *(pXtX[OnKappaS]+1) : -666,
      n > 3 ? *(pXtX[OnKappaS]+2) : -666, 
      n > 4 ? *(pXtX[OnKappaS]+3) : -666);      
    R_FlushConsole();
    Rprintf(" --- Did We die?  here is pXtX[%d][p-1=%d-1=%d: ", OnKappaS, p, p-1); R_FlushConsole();
    Rprintf(" %.5f \n", *(pXtX[OnKappaS]+p-1)); R_FlushConsole();
    Rprintf(" --- Did We all Die?  If Not, Trans=%c, n=%d, p=%d, OneD=%f, ZeroD=%f, One=%d\n",
      Trans, n, p, OneD, ZeroD, One); R_FlushConsole();
    }
    Rprintf(" --- If we Survive we will now dgemv. \n"); R_FlushConsole();
  }
  */
  if (iiWeight == NULL) {
    F77_CALL(dgemv)(&Trans, &n, &p, &OneD, REAL(sX), &n,
      REAL(sX) + NewCoord * n, &One, &ZeroD, pXtX[OnKappaS], &One);
    //if (tt == 1) {
    //  Rprintf("w"); R_FlushConsole();
    //}
  } else {
    int ii;
    F77_CALL(dscal)(&p, &ZeroD, pXtX[OnKappaS], &One);
    double Factor = 0.0;
    for (ii = 0; ii < n; ii++) {
      Factor = iiWeight[ii] * REAL(sX)[ii + n * NewCoord];
      F77_CALL(daxpy)(&p, &Factor, REAL(sX) + ii, &n,
        pXtX[OnKappaS], &One);  
    }
  }
  OnKappaS++;
  if (OnKappaS > p) {
    Rf_error("AddCoord:  Somehow accidentally added more than p=%d coordinates!\n",
      p);
  }
  
  if (tt < 10 || Verbose >= 2) {
    char MyGo[250];
    sprintf(MyGo, "AddCoordEnd: tt=%d/%d, NewCoord=%d, OnKappaS=%d, iiWeight is %s", tt, MaxGibbsIters, NewCoord, OnKappaS, iiWeight==NULL? "NULL" : "Not Null");
    CheckkXFinderError(MyGo);
  }
  //if (tt == 1) { Rprintf("c"); R_FlushConsole(); }
   if (Verbose > 2) {
     Rprintf("   Finished Adding %d, OnKappaS = %d \n", NewCoord, OnKappaS);
   }
  return(1);
}
int BayesSpikeCL::SetXtResidNoXtX() {
  UpdateNonFreshXtResid();
  int Verbose = this->Verbose -2;
  return(1);
  if (Verbose > 1) {
    Rprintf("SetXtResidNoXtX \n"); R_FlushConsole();
  }
  if (XtResid == NULL) {
    RMemGetD(XtResid, "XtResid", p);
  }
  double ZeroD = 0.0; int One = 1;  int ii;
  F77_CALL(dscal)(&p, &ZeroD, XtResid, &One);
  //double OneD = 1.0;
  double Residy = 0.0;
  if (iiWeight == NULL) {
    if (Verbose > 1) {
      Rprintf("SetXtResidNoXtX iiWeight = NULL\n"); R_FlushConsole();
    }
    for (ii = 0; ii < n; ii++) {
      Residy = REAL(sY)[ii] - 
        F77_CALL(ddot)(&p, REAL(sX) + ii, &n, REAL(sBeta), &One);
      F77_CALL(daxpy)(&p, &Residy, REAL(sX) + ii, &n, XtResid, &One); 
    }
    if (Verbose > 1) {
      Rprintf("SetXtResidNoXtX with iiWeight = NULL\n"); R_FlushConsole();
    }
    return(1);
  } else {
    if (Verbose > 1) {
      Rprintf("SetXtResidNoXtX with iiWeight Active\n"); R_FlushConsole();
    }
    for (ii = 0; ii < n; ii++) {
      Residy = iiWeight[ii]*(REAL(sY)[ii] - 
        F77_CALL(ddot)(&p, REAL(sX) + ii, &n, REAL(sBeta), &One));
      F77_CALL(daxpy)(&p, &Residy, REAL(sX) + ii, &n, XtResid, &One); 
    }
    return(1);
  }
}

/////////////////////////////////////////////////////////////////////////
//  RecordCoda()
//
//  RecordCoda records onto Codajj/CodaBeta buffers the identity of non-zero
//   coordinates in the draws as well as their beta magnitudes.  
//
//  Also activates ProbCoda/ICoda buffers if those are being written to a file
int BayesSpikeCL::RecordCoda() {
  RanRecordCoda++;
  if (Codajj == NULL || CodaBeta == NULL) {
    return(1);
  }
  int ii = 0;
  int AT = 0;
  if (MaxCODABeta < p + OnCoda) {
    AT = WriteCoda();
    if (AT < 0) {
      Rprintf("*******************************************************\n");
      Rprintf("** Error in RecordCoda, we ran WriteCoda got AT = %d \n", AT);
      Rprintf("** Don't know the error, but we quit too! \n");
      R_FlushConsole();
      return(-1);
    }
  }
  int AnyNonZero = 0;
  for (ii = 0; ii < p; ii++) {
    if (REAL(sBeta)[ii] != 0.0) { AnyNonZero = 1;  break;}
  }
  if (AnyNonZero == 0) { 
    Codajj[OnCoda] = -(tt+1); 
    CodaBeta[OnCoda] = 0;
    OnCoda++;
    if (ICoda != NULL && ProbCoda != NULL) {
      if (iiOnILocProb > LengthProbCoda-1) {
        iiOnILocProb = LengthProbCoda;
         WritePICoda();  iiOnILocProb = 0;
      }
    }
    if (ICoda == NULL) {
      Rprintf("ICoda is not set up!\n"); R_FlushConsole();
    }
    ICoda[iiOnILocProb] = (bufferILocType) LocationOfBetaWrite;
    if (dfRobit < 0) {
      CurrentProb = ProbPosterior();
    }
    ProbCoda[iiOnILocProb] = CurrentProb;
    LocationOfBetaWrite++;    iiOnILocProb++;
    return(0); 
  }
  Codajj[OnCoda] = -(tt+1); 
  CodaBeta[OnCoda] = AnyNonZero;
  int SCoda = OnCoda;
  OnCoda++;    AnyNonZero = 0;
  if (ICoda != NULL && ProbCoda != NULL) {
    if (iiOnILocProb > LengthProbCoda-1) {
      iiOnILocProb = LengthProbCoda;
       WritePICoda();  iiOnILocProb = 0;
    }
    ICoda[iiOnILocProb] = (long) LocationOfBetaWrite;
    if (dfRobit < 0) {
      CurrentProb = ProbPosterior();
    }
    ProbCoda[iiOnILocProb] = CurrentProb;
    //Rprintf("Wrote to iiOnILocProb[%d], LocationOfBetaWrite = %d, tt = %d\n", 
    //  iiOnILocProb, LocationOfBetaWrite, tt); R_FlushConsole();
    iiOnILocProb++;

  }
  LocationOfBetaWrite++;
  AnyNonZero = 0;
  for (ii = 0; ii < p; ii++) {
    if (REAL(sBeta)[ii] != 0.0) {
      Codajj[OnCoda] = ii;
      CodaBeta[OnCoda] = REAL(sBeta)[ii];
      AnyNonZero++;
      OnCoda++; LocationOfBetaWrite++;
    }  
  }
  CodaBeta[SCoda] = AnyNonZero;
  int RecordCodaFlag = 0;
  if (RsCodaBetaAllDrawBuffer != NULL && BetaAllDrawBuffer != NULL) {
    RecordCodaFlag = WriteToBetaAllDrawBuffer();
    if (RecordCodaFlag < 0) {
      Rprintf("BayesSpikeCpp.cpp:::RecordCoda() error, WRiteToBetaAllDrawBuffer returned %d\n", RecordCodaFlag);
      Rprintf("-------------------------------  returning an error from RecordCoda(). \n");
      R_FlushConsole(); return(-1);
    }    
  }
  return(1);
}

/////////////////////////////////////////////////////////////////
//  WritePICoda
//
//  ProbCoda and ICoda Buffers are written to files stored in RsCodaILocFile/RsCodaProbFile
//
//   ProbCoda is a list of overall Posterior Probability of Draws
//   ICoda is an integer pointer to the location of Betas in the binary d file
//
//
int BayesSpikeCL::WritePICoda() {
  FILE *pFile = NULL; FILE *iFile = NULL;
  if (Verbose >= 2) { 
    Rprintf("WritePICoda Start. \n"); R_FlushConsole();
  }
  if (RsCodaILocFile == NULL || Rf_isNull(RsCodaILocFile->asSexp() )) {
    Rprintf("WritePICoda: Your RsCodaILocFile has NULL Pointer! \n");  R_FlushConsole();
    
    return(-5);
  }
  if (RsCodaProbFile == NULL || Rf_isNull(RsCodaProbFile->asSexp()) ||
    Rf_isNull(STRING_ELT(RsCodaProbFile->asSexp(),0)))  {
    Rprintf("WritePICoda:  Your RsCodaProbFile has NULL Pointer \n"); R_FlushConsole();
    return(-4);  
  }
  if (Verbose >= 2) {
    Rprintf(" About to Write PI Coda!\n"); R_FlushConsole();
    Rprintf("  Rf_isString(sCodaIFile) = %d\n", Rf_isString(RsCodaILocFile->asSexp()));
    Rprintf("  RsCodaILocFile = %s \n",  CHAR(STRING_ELT(RsCodaILocFile->asSexp(), 0)) ); R_FlushConsole();
  }
  if (!Rf_isString(RsCodaProbFile->asSexp()) || !Rf_isString(RsCodaILocFile->asSexp())) {
     Rprintf("Error RsCodaProbFile File is not String! \n"); R_FlushConsole();
     Rprintf("What are we going to do about this string? : ");
     Rprintf(" It is %s \n", CHAR(STRING_ELT(RsCodaProbFile->asSexp(), 0)) ); R_FlushConsole();
  }
  if (ICoda == NULL) {
    Rprintf(" RecordCoda: Can't do it because there's no ICoda\n"); R_FlushConsole();
    return(-1);
  }
  if (ProbCoda == NULL) {
    Rprintf(" RecordCoda: Can't do it because there's no Prob Coda\n"); R_FlushConsole();
    return(-1);
  }
  int NewWrite = this->NewWrite;
  if (ICoda[0] == 0) { NewWrite = 1; }
  if (NewWrite == 1)  {
    if (Verbose > 1) {
      Rprintf("About to Newwrite to ICoda and ProbCoda, iiOn-%d, sizeof(double) = %d, sizeof(int) = %d\n",
        iiOnILocProb, sizeof(double), sizeof(int) ); R_FlushConsole();
    }
    iFile = fopen(CHAR(STRING_ELT(RsCodaILocFile->asSexp(), 0)), "wb");
    pFile = fopen(CHAR(STRING_ELT(RsCodaProbFile->asSexp(), 0)), "wb");  
    if (iFile != NULL) { TotalOpenedFiles++; }
    if (pFile != NULL) { TotalOpenedFiles++; }
  } else {
    if (Verbose > 1) {
      Rprintf("About to append to ICoda and ProbCoda, iiOn-%d, sizeof(double) = %d, sizeof(int) = %d\n",
        iiOnILocProb, sizeof(double), sizeof(int) ); R_FlushConsole();
    }
    iFile = fopen(CHAR(STRING_ELT(RsCodaILocFile->asSexp(), 0)), "ab");
    pFile = fopen(CHAR(STRING_ELT(RsCodaProbFile->asSexp(), 0)), "ab");
    if (iFile != NULL) { TotalOpenedFiles++; }
    if (pFile != NULL) { TotalOpenedFiles++; }
  }
  if (iFile == NULL || pFile == NULL) {
    Rprintf("*****************************************************************\n");
    Rprintf("** WritePICoda: Something bad happened during RecordCoda and we have file error \n");
    Rprintf("** Try to force open again?");
    if (iFile == NULL)  {
      Rprintf("** Trying to force open iFile = %s\n ", CHAR(STRING_ELT(RsCodaILocFile->asSexp(), 0)));
      iFile = fopen(CHAR(STRING_ELT(RsCodaILocFile->asSexp(), 0)), "ab");
      if (iFile != NULL) { TotalOpenedFiles++; }
      if (iFile == NULL) {
        Rprintf("** Nope that iFile failed sadly! \n"); R_FlushConsole();
      }
    }
    if (pFile == NULL)  {
      Rprintf("** Trying to force open pFile = %s\n ", CHAR(STRING_ELT(RsCodaProbFile->asSexp(), 0)));
      pFile = fopen(CHAR(STRING_ELT(RsCodaProbFile->asSexp(), 0)), "ab");
      if (pFile != NULL) { TotalOpenedFiles++; }
      if (pFile == NULL) {
        Rprintf("** Nope that pFile failed sadly! \n"); R_FlushConsole();
      }
    }
    if (iFile == NULL || pFile == NULL) {
      Rprintf("** WritePICoda: Fail to open iFile or pFile! \n"); R_FlushConsole();
      if (iFile != NULL) {fclose(iFile); TotalClosedFiles++; }
      if (pFile != NULL) { fclose(pFile); TotalClosedFiles++; }
      return(-1);
      Rf_error("WritePICoda: Could Not open PrintFiles!");
    }
  }
  if (Verbose > 1) {
    Rprintf("About to write to ICoda and ProbCoda, iiOn-%d, sizeof(double) = %d, sizeof(int) = %d\n",
      iiOnILocProb, sizeof(double), sizeof(int) ); R_FlushConsole();
  }
  fwrite( ICoda, sizeof(bufferILocType), iiOnILocProb, iFile );
  fwrite( ProbCoda, sizeof(double), iiOnILocProb, pFile );  
  fclose(iFile); fclose(pFile);  pFile = NULL; iFile = NULL;
  TotalClosedFiles+=2;
  iiOnILocProb = 0;
  if (Verbose >= 2) { 
    Rprintf("WritePICoda finished. \n"); R_FlushConsole();
  }
  return(1);
}

int BayesSpikeCL::WriteToBetaAllDrawBuffer() {
  if (Verbose >=3) { 
    Rprintf("WriteToBetaAllDrawBuffer Start. \n"); R_FlushConsole();
  }
  if (RsCodaBetaAllDrawBuffer == NULL) {
    Rprintf("WriteToBetaAllDrawBuffer: Why do it if BetaAllFile name is null!\n");
    R_FlushConsole();
    return(-1);
  }
  if (BetaAllDrawBuffer == NULL) {
    Rprintf("WriteToBetaAllDrawBuffer: No Beta is NULL!\n");
  }
 
  int ABad = 0;
  if (LengthWrittenBetaAllDrawBuffer+p+1>= LengthBetaAllDrawBuffer) {
    ABad = this->ClearToBetaAllDrawBuffer();
    if (ABad < 0) {
      Rprintf("BayesSpikeCpp.cpp::: Error WriteToBetAllDrawBuffer: ClearToBetaAllDrawBuffer returned %d \n", ABad);
      Rprintf("--------------------   We shall return an error as well. \n"); R_FlushConsole();
      return(-1);
    }
  }
  int OnActive = 0;
  int Onjj = 0;   double A;  double B;
  if (XjSq == NULL) {
    Rf_error("WriteToBetaAllDrawBuffer: No XjSq is null!\n");
  }
  int OnRandomTau = 0;
  double SampleABeta;
  BetaAllDrawBuffer[LengthWrittenBetaAllDrawBuffer] = tt;
  /*
   int One = 1;
  Rprintf("LetsTryQuitting\n"); R_FlushConsole();
    F77_CALL(dcopy)(&p, REAL(sBeta),&One, BetaAllDrawBuffer, &One);
  LengthWrittenBetaAllDrawBuffer += p+1;
  LengthTotalWrittenBetaAllDrawBuffer += p+1;
  if (tt >= 3) {
    Rprintf("WriteToBetaAllDrawBuffer: Quit after initial error \n");
    Rf_error("Inserted Error");
  }
  return(1);
  */
  for (Onjj = 0; Onjj < p; Onjj++) {
    while(OnActive < NumActive && OrderedActive[OnActive] < Onjj) {
      OnActive++;
    }
    if (OnActive < NumActive && OrderedActive[OnActive] == Onjj) {
      BetaAllDrawBuffer[LengthWrittenBetaAllDrawBuffer+1+Onjj] =
        REAL(sBeta)[Onjj];
    } else {
      // (1/2sigma^2) * (  Y-XBetanj - XjBetaj)^2 == 
      //  -(1/2sigma^2) * (   -2BetajXj(Y-XBetanj) + Xj^2Betaj^2) + Betaj^2/(2*tauSq)
      A = XtResid[Onjj];  
      B = XjSq[Onjj];
      if (iFirstRandom < 0 || 
         (iFirstRandom > 0 && iFirstRandom > Onjj)) {
        if (!Rf_isNull(tauFixed)) {
          if (Rf_length(tauFixed) > Onjj) {
            if (REAL(tauFixed)[Onjj] > 0.0) {
              B += REAL(sOnSigma)[0] / REAL(tauFixed)[Onjj];
            } else {
              B += 1.0 / fabs(REAL(tauFixed)[Onjj]); 
            }
          } else {
            if (tauFixed == NULL || Rf_isNull(tauFixed) || Rf_length(tauFixed) <= 0) {
              Rprintf("BayesSpikeCpp.cpp:: tauFixed is zero length here?\n"); R_FlushConsole();
            } else if (REAL(tauFixed)[Rf_length(tauFixed)-1] > 0.0) {
              B += REAL(sOnSigma)[0] / REAL(tauFixed)[Rf_length(tauFixed)-1];
            } else {
              B += 1.0 / fabs(REAL(tauFixed)[Rf_length(tauFixed)-1]);            
            }
          }
        } 
      } else {
        if (!Rf_isNull(tauEndList) && !Rf_isNull(sOnTau) &&
          tauEndList != NULL && sOnTau != NULL &&
          Rf_length(tauEndList) >= 1) {
        if (OnRandomTau < Rf_length(tauEndList) &&
          Onjj < ANINT(tauEndList, OnRandomTau)) {
          if (REAL(sOnTau)[OnRandomTau] > 0) {
            B += REAL(sOnSigma)[0]/REAL(sOnTau)[OnRandomTau]; 
          } 
        } else if (OnRandomTau < Rf_length(tauEndList) &&
          Onjj == ANINT(tauEndList, OnRandomTau)) {
          if (REAL(sOnTau)[OnRandomTau] > 0) {
            B += REAL(sOnSigma)[0]/REAL(sOnTau)[OnRandomTau]; 
          }
          OnRandomTau++;    
        } else {
          while(OnRandomTau < Rf_length(tauEndList) &&
             Onjj > ANINT(tauEndList, OnRandomTau)) {
            OnRandomTau++;   
          }
          if (OnRandomTau < Rf_length(tauEndList) &&
            Onjj <= ANINT(tauEndList, OnRandomTau)) {
            if (REAL(sOnTau)[OnRandomTau] > 0) {
              B += REAL(sOnSigma)[0]/REAL(sOnTau)[OnRandomTau];
            }
          }
        }
        }
      }
      //  Betaj^2 * B - 2 Betaj * A +  A^2/B = B( Betaj - A/B)^2
      SampleABeta = A/B +  sqrt(REAL(sOnSigma)[0]) /sqrt(B) * Rf_rnorm(0.0, 1.0);
      BetaAllDrawBuffer[LengthWrittenBetaAllDrawBuffer+1+Onjj] = SampleABeta;
    }
  }
  LengthWrittenBetaAllDrawBuffer += (p+1);
  LengthTotalWrittenBetaAllDrawBuffer += (p+1);

  return(1);
}


////////////////////////////////////////////////////////////////////////////////
//  ClearToBetaAllDrawBuffer
//
//    Write all data specified in file named "RsCodaBetaAllDrawBuffer" 
//     the element BetaAllDrawBuffer.
int BayesSpikeCL::ClearToBetaAllDrawBuffer() {
  FILE *pFile = NULL;
  if (Verbose >= 2) { 
    Rprintf("ClearToBetaAllDrawBuffer Start. \n"); R_FlushConsole();
  }
  if (RsCodaBetaAllDrawBuffer == NULL) {
    Rprintf("ClearToBetaAllDrawBuffer: Your RsCodaBetaAllDrawBuffer has NULL Pointer! \n");  R_FlushConsole();
    
    return(-5);
  }
  if (RsCodaBetaAllDrawBuffer == NULL || Rf_isNull(RsCodaBetaAllDrawBuffer->asSexp()) ||
    Rf_isNull(STRING_ELT(RsCodaBetaAllDrawBuffer->asSexp(),0)))  {
    Rprintf("ClearToBetaAllDrawBuffer:  Your ClearToBetaAllDrawBuffer has NULL Pointer \n"); R_FlushConsole();
    return(-4);  
  }
  if (Verbose >= 2) {
    Rprintf(" About to Write CodaAllDrawBuffer Coda!\n"); R_FlushConsole();
    Rprintf("  Rf_isString(RsCodaBetaAllDrawBuffer) = %d\n", Rf_isString(RsCodaBetaAllDrawBuffer->asSexp()));
    R_FlushConsole();
  }
  if (!Rf_isString(RsCodaBetaAllDrawBuffer->asSexp()) || !Rf_isString(RsCodaBetaAllDrawBuffer->asSexp())) {
     Rprintf("Error RsCodaBetaAllDrawBuffer File is not String! \n"); R_FlushConsole();
     Rprintf("What are we going to do about this string? : ");
     Rprintf(" It is %s \n", CHAR(STRING_ELT(RsCodaBetaAllDrawBuffer->asSexp(), 0)) ); R_FlushConsole();
  }
  if (BetaAllDrawBuffer == NULL) {
    Rprintf(" RecordCoda: Can't do it because there's no BetaAllDrawBuffer\n"); R_FlushConsole();
    return(-1);
  }
  if (LengthTotalWrittenBetaAllDrawBuffer == 
    LengthWrittenBetaAllDrawBuffer) {
    pFile = fopen(CHAR(STRING_ELT(RsCodaBetaAllDrawBuffer->asSexp(), 0)), "wb");
  } else {
    pFile = fopen(CHAR(STRING_ELT(RsCodaBetaAllDrawBuffer->asSexp(), 0)), "ab");
  }
  if (pFile != NULL) {TotalOpenedFiles++; }
  
  if (pFile == NULL) {
    Rprintf("****************************************************************\n");
    Rprintf("** WriteRsCodaBetallDrawBuffer: no open of pFile = %s \n",
      CHAR(STRING_ELT(RsCodaBetaAllDrawBuffer->asSexp(), 0)));
    Rprintf("**  One more chance! \n"); R_FlushConsole(); 
    pFile = fopen(CHAR(STRING_ELT(RsCodaBetaAllDrawBuffer->asSexp(), 0)), "wb");
    if (pFile != NULL) { TotalOpenedFiles++; }
    if (pFile == NULL) {
      Rprintf("** No still could not open pFile! \n");
      return(-1);
      Rf_error("WriteRsCodaBetaAllDrawBuffer: Could Not open PrintFiles!");
    }
  }
  if (Verbose > 1) {
    Rprintf("About to write to RsCodaBetaAllDrawBuffer,  sizeof(double) = %d, sizeof(int) = %d\n",
      sizeof(double), sizeof(int) ); R_FlushConsole();
  }
  fwrite( BetaAllDrawBuffer, sizeof(double), LengthWrittenBetaAllDrawBuffer, pFile );  
  LengthWrittenBetaAllDrawBuffer = 0;
  fclose(pFile);  pFile = NULL; TotalClosedFiles++;
  if (Verbose >= 2) { 
    Rprintf("WriteRsCodaBetaAllDrawBuffer finished. \n"); R_FlushConsole();
  }
  return(1);
}  

////////////////////////////////////////////////////////////////////////////////
//   int BayesSpikeCL::WriteCoda() {
//
//    WriteCoda will write to the two files CodaIFile (integers) and 
//  CodaJFile (doubles), a compressed string of numbers corresponding to
//  the active (non-zero) Beta coefficients in simulation
//
//  CodaIFile contains the coordinate identitifes of non-zero Beta Coefficients.
//   A negative number at the beginning of each CodaIFile contains
//   the count of non-zero coefficients. 
//  CodaJFile contains the values of the Beta Coefficients.
//
int BayesSpikeCL::WriteCoda() {
  FILE *iFile = NULL;  FILE *dFile = NULL;

  if (Rf_isNull(sCodaIFile) || !Rf_isString(sCodaIFile)) {
    Rprintf("----------------------------------------------------------\n");
    Rprintf("Right at the beginning of WriteCoda!!\n");
    Rf_error("WriteCoda: bad error, CodaIFile is not a string!\n");
  }
  if (Rf_isNull(sCodaJFile) || !Rf_isString(sCodaJFile)) {
    Rprintf("----------------------------------------------------------\n");
    Rf_error("WriteCoda: bad error, CodaJFile is not a string!\n");
  }
  if (Verbose >= 2) {
    Rprintf(" About to Write Coda!\n"); R_FlushConsole();
    if (Rf_isNull(sCodaIFile) || !Rf_isString(sCodaIFile)) {
      Rprintf("---------------------------------------------------------\n");
      Rf_error("WriteCoda: bad error, CodaIFile is not a string!\n");
    }
    Rprintf("  Rf_isString(sCodaIFile) = %d\n", Rf_isString(sCodaIFile));
    Rprintf("  sCodaIFile = %s \n",  CHAR(STRING_ELT(sCodaIFile, 0)) ); R_FlushConsole();
  }
  if (!Rf_isString(sCodaIFile) || !Rf_isString(sCodaJFile)) {
     Rprintf("Error sCodaIFile is not String!\n"); R_FlushConsole();
     Rprintf("What are we going to do about this string?: ");
     Rprintf(" it is %s \n", CHAR(STRING_ELT(sCodaIFile, 0)) ); R_FlushConsole();
  }
  if (NewWrite == 1)  {
    iFile = fopen(CHAR(STRING_ELT(sCodaIFile, 0)), "wb");
    dFile = fopen(CHAR(STRING_ELT(sCodaJFile, 0)), "wb");  
    if (iFile != NULL) { TotalOpenedFiles++; }
    if (dFile != NULL) { TotalOpenedFiles++; }
  } else {
    iFile = fopen(CHAR(STRING_ELT(sCodaIFile, 0)), "ab");
    dFile = fopen(CHAR(STRING_ELT(sCodaJFile, 0)), "ab");
    if (iFile != NULL) { TotalOpenedFiles++; }
    if (dFile != NULL) { TotalOpenedFiles++; }
  }
  if (iFile == NULL || dFile == NULL) {
    Rprintf("**************************************************************\n");
    Rprintf("** WriteCoda Failure to open iFile or dFile !  \n");
    if (iFile == NULL) {
      Rprintf("**  Second shot at: iFile: %s \n", CHAR(STRING_ELT(sCodaIFile, 0)));
      iFile = fopen(CHAR(STRING_ELT(sCodaIFile, 0)), "wb");
      if (iFile != NULL) { TotalOpenedFiles++; }
      if (iFile == NULL) {
        Rprintf("** No failed on second shot iFile to read %s \n",
          CHAR(STRING_ELT(sCodaIFile, 0)));
      }
      R_FlushConsole(); 
    }
    if (dFile == NULL) {
      Rprintf("**  Second shot at: dFile: %s \n", CHAR(STRING_ELT(sCodaJFile, 0)));
      dFile = fopen(CHAR(STRING_ELT(sCodaJFile, 0)), "wb");
      if (dFile != NULL) { TotalOpenedFiles++; }
      if (dFile == NULL) {
        Rprintf("** No failed on second shot dFile to read %s \n",
          CHAR(STRING_ELT(sCodaJFile, 0)));
      }
      R_FlushConsole(); 
    }
    if (iFile == NULL || dFile == NULL)  {
      Rprintf("**  WriteCoda, early exit  \n");
      if (iFile != NULL) { fclose(iFile);  TotalClosedFiles++; iFile = NULL; }
      if (dFile != NULL) { fclose(dFile);  TotalClosedFiles++; dFile = NULL; }
      Rprintf("** We're going to close buffers and prevent WriteCoda rewrite! \n"); R_FlushConsole();
      FFree(Codajj, "Codajj"); Codajj = NULL; FFree(CodaBeta, "CodaBeta");
      Rprintf("Codajj, CodaBeta are cleared! \n"); R_FlushConsole();
      DDelete(RsCodaIFile, "RscodaIFile"); RsCodaIFile = NULL; sCodaIFile = R_NilValue;
      DDelete(RsCodaJFile, "RscodaJFile"); RsCodaJFile = NULL; sCodaJFile = R_NilValue;  
      OnCoda = 0; NewWrite = 0;
      Rprintf("** WriteCoda: We have cleared buffers and this should no longer happen! \n"); R_FlushConsole();    
      return(-1);
      Rf_error("WriteCoda: Could Not open PrintFiles!");
    }
  }
  fwrite( Codajj, sizeof(int), OnCoda, iFile );
  fwrite( CodaBeta, sizeof(double), OnCoda, dFile );
  OnCoda = 0;  
  NewWrite = 0;
  int weClosed = 0;
  weClosed = fclose(iFile);  iFile = NULL;  TotalClosedFiles++;
  if (weClosed != 0) {
    Rprintf("WriteCoda: Well, we failed to close sCodaIFile !\n");  R_FlushConsole();
  }
  weClosed = fclose(dFile);  dFile = NULL; TotalClosedFiles++;
  if (weClosed != 0) {
    Rprintf("WriteCoda: Well, we failed to close sCodaJFile !\n");  R_FlushConsole();
  }
  return(1);
}

////////////////////////////////////////////////////////////////////////////////
//  int BayesSpikeCL::RecordHistory()
//
//     When the Gibbs Sampler finishes a step, the Beta coordinates that are 
//   active and other important data are recorded to buffers.  Note that
//   these are compressed buffers and not conventional CODA-mcmc objects.  
//   This way the buffers can be stored in .bin files and reloaded when
//   necessary.  
//     However, for small datasets, regular package CODA-mcmc objects can 
//   be used.  This function, RecordHistory, records data to a Coda MCMC object
//   (data matrix of doubles).
//
int BayesSpikeCL::RecordHistory() {
  if (this->Verbose > 3) {
    Rprintf("Recording History\n"); R_FlushConsole();
  }
  if (this->OnCodaTable < 0) {
    Rprintf("RecordHistory Error, OnCodaTable < 0"); R_FlushConsole();
    return(-2);
  }
  if (RCodaList == NULL || Rf_isNull(RCodaList->asSexp())) {
    Rprintf("RCodaList:  Error RCodaList is Null\n"); R_FlushConsole();
    return(-2);
  }
  int CTL = (((VECSEXP2) (this->RCodaList->asSexp()))->vecsxp.length);
  if (CTL <= this->OnCodaTable) {
    Rf_error("RCodaList:  Error, RCodaList has length %d, but OnCodaTable = %d\n",
      CTL, this->OnCodaTable); 
  }
  int TiFirstRandom = iFirstRandom;
  if (sOnTau == NULL || Rf_isNull(sOnTau) || Rf_length(sOnTau) <= 0 ||
    tauEndList == NULL || Rf_isNull(tauEndList) || Rf_length(tauEndList) <= 0) {
    TiFirstRandom = p;
  }
  SEXP CodaTable = VECTOR_ELT(this->RCodaList->asSexp(), 
    this->OnCodaTable);
  if (CodaTable == NULL && !Rf_isNull(this->DoRecord)) {
    Rprintf("BayesSpikeCL Error, give me a non null CodaTable!\n"); 
    R_FlushConsole();
  }
  if (CodaTable == NULL || Rf_isNull(CodaTable)) { return(-1); }
  if (Rf_isNull(Rf_getAttrib(CodaTable, R_DimSymbol))) {
    Rf_error("RecordHistory: CodaTable does not have a Dim Object!\n");
  }
  if (!Rf_isInteger(Rf_getAttrib(CodaTable, R_DimSymbol))) {
    Rf_error("RecordHistory: CodaTable Dimension!\n");
  }
  if (tt >= INTEGER(Rf_getAttrib(CodaTable, R_DimSymbol))[0]) {
    Rf_error("Cannot Record History Past End of Matrix!");
  }  
  int LDA = INTEGER(Rf_getAttrib(CodaTable, R_DimSymbol))[0];
  int D = INTEGER(Rf_getAttrib(CodaTable, R_DimSymbol))[1];
  int iP = 0;
  if (!Rf_isReal(sBeta)) {
    Rf_error("RecordHistory: Hey, Beta is not Real!\n");
  }
  if (!Rf_isReal(CodaTable)) {
    Rf_error("RecordHistory: Hey CodaTable is not Real!\n");
  }
  if (DoRecord == NULL) {
    Rprintf("RecordHistory: Hey, Do Record is Null pointer!\n");
    return(-1);
  }
  if (Rf_isNull(DoRecord)) {
    Rprintf("RecordHistory: Hey Do Record is Null SEXP\n");
    return(-1);
  }
  if (!Rf_isInteger(DoRecord) && !Rf_isReal(DoRecord)) {
    Rprintf("RecordHistory: Hey, Do Record is not an integer!\n");
    return(-1);
  }

  int CLen = 0; int One = 1;
  if (!Rf_isNull(sBeta) && !Rf_isNull(DoRecord) && Rf_length(DoRecord) > 0 && (ANINT(DoRecord, 0)) == 1) {
    CLen = p;
    if (iP + p > D) {
      Rf_error("Not Enough room for Beta!\n");
    }
    F77_CALL(dcopy)(&CLen, REAL(sBeta), &One, 
      REAL(CodaTable) + iP*LDA + tt, &LDA);
    iP += CLen;
  }
  if (CLen > D) {
    Rf_error("RecordHistory: Error at beginning, CLen = %d, D = %d\n",
      CLen, D);
  }

  if (sOnTau != NULL && !Rf_isNull(sOnTau)  && Rf_length(sOnTau) > 0
   && !Rf_isNull(DoRecord) && Rf_length(DoRecord) >= 2 && (ANINT(DoRecord,1)) == 1) {
    CLen = Rf_length(sOnTau);
    if (iP + Rf_length(sOnTau) > D) {
      Rf_error("Not enough room for Tau!");
    }
    F77_CALL(dcopy)(&CLen, REAL(sOnTau), &One, 
      REAL(CodaTable) + iP*LDA + tt, &LDA);
    iP += CLen;
  }
  if (!Rf_isNull(sOnSigma)  && Rf_length(sOnSigma) > 0 &&
    !Rf_isNull(DoRecord) && Rf_length(DoRecord) >= 3 && (ANINT(DoRecord,2) == 1)) {
    CLen = Rf_length(sOnSigma);
    if (iP + Rf_length(sOnSigma) > D) {
      Rf_error("Not enough room for CurrentSigma!");
    }
    F77_CALL(dcopy)(&CLen, REAL(sOnSigma), &One, 
      REAL(CodaTable) + iP*LDA + tt, &LDA);
    iP += CLen;
  }
  if (!Rf_isNull(sOnPiA)  && Rf_length(sOnPiA) > 0 &&
    !Rf_isNull(DoRecord) && Rf_length(DoRecord) >= 4 && (ANINT(DoRecord,3) == 1)) {
    CLen = Rf_length(sOnPiA);
    if (iP + Rf_length(sOnPiA) > D) {
      Rf_error("Not enough room for OnPiA!");
    }
    F77_CALL(dcopy)(&CLen, REAL(sOnPiA), &One, 
      REAL(CodaTable) + iP*LDA + tt, &LDA);
    iP += CLen;
  }
  if (tauFixed != NULL && !Rf_isNull(tauFixed) && Rf_length(tauFixed) > 0 &&
    !Rf_isNull(DoRecord) && Rf_length(DoRecord) >=5 && (ANINT(DoRecord,4) == 1)) {
    CLen = Rf_length(tauFixed);
    if (iP + CLen > D) {
      Rf_error("Record History: Not enough room for tauFixed!");
    }
    F77_CALL(dcopy)(&CLen, REAL(tauFixed), &One, 
      REAL(CodaTable) + iP*LDA + tt, &LDA);
    iP += CLen;
  }
  if (ProbFixed != NULL && 
    !Rf_isNull(DoRecord) && Rf_length(DoRecord) >= 6 && (ANINT(DoRecord,5) == 1)) {
    if (TiFirstRandom > 0 && TiFirstRandom <= p) {
      CLen = TiFirstRandom;
      if (iP + TiFirstRandom > D) {
        Rf_error("Not enough room for TauProbList!");
      }
      F77_CALL(dcopy)(&CLen, ProbFixed, &One, 
        REAL(CodaTable) + iP*LDA + tt, &LDA);
      iP += CLen;
    }
  }
  if (ProbTau != NULL && 
    !Rf_isNull(DoRecord) && Rf_length(DoRecord) >= 7 && (ANINT(DoRecord,6) == 1)) {
    if (!Rf_isNull(sOnTau) && Rf_length(sOnTau) > 0) {
      CLen = Rf_length(sOnTau);
      if (iP + Rf_length(sOnTau) > D) {
        Rf_error("Not enough room for TauProbList!");
      }
      F77_CALL(dcopy)(&CLen, ProbTau, &One, 
        REAL(CodaTable) + iP*LDA + tt, &LDA);
      iP += CLen;
    }
  }
  if (iP > CountRecord() || iP > D) {
    Rf_error("Error in recording, iP = %d, but CR() = %d, D = %d\n",
      iP, CountRecord(), D);
  }
  if (this->Verbose > 3) {
    Rprintf("Recorded down to iP = %d\n", iP); R_FlushConsole();
  }
  return(1);  
}


////////////////////////////////////////////////////////////////////////
//  UpdateXtY()
//
//     UpdateXtY will reweight XtY  sum( X_ij w_i Y_i ) where
//   The vector iiWeight are the weights for every datapoint Y_i.
//   This occurs in t-noise data and in GLM distributions.
//
int BayesSpikeCL::UpdateXtY() {
    int One = 1; char Trans = 'T'; double OneD = 1.0;
    double ZeroD = 0.0;
    int ii;
    if (iiWeight == NULL) {
      F77_CALL(dgemv)(&Trans, &n, &p, &OneD, REAL(sX), &n, REAL(sY), &One,
        &ZeroD, XtY, &One);
    } else {
      if (n < p) {
        F77_CALL(dscal)(&p, &ZeroD, XtY, &One);
        double Factor = 0.0;
        for (ii = 0; ii < n; ii++) {
          Factor = iiWeight[ii] * REAL(sY)[ii];
          F77_CALL(daxpy)(&p, &Factor, REAL(sX) + ii, &n,  XtY, &One);        
        }
      } else {
        if (WW == NULL) {
          RMemGetD(WW, "WW", n);
        }
        for (ii = 0; ii < n; ii++) {WW[ii] = REAL(sY)[ii] * iiWeight[ii];}
        F77_CALL(dgemv)(&Trans, &n, &p, &OneD, REAL(sX), &n, WW, &One,
          &ZeroD, XtY, &One);
      }
    }
    return(1);
}

////////////////////////////////////////////////////////////////////////////////
//  FillSmallXtResid()
//
//
//  cXtY are set to parts of t(X) %*% (Y-X Beta) related to 
//     the OrderedActive Coordinates
int BayesSpikeCL::FillSmallXtResid() {
  if (OrderedActive == NULL) {
    if (AllNewCoords > 0) {
      AddAllNewCoords();
    }
    RefreshOrderedActive(0);
  }
  int ii;
  for (ii = 0; ii < NumActive; ii++) {
    cXtY[ii] = XtResid[OrderedActive[ii]];
  }
  for (ii = 0; ii < NumActive; ii++) {
    PropBeta[ii] = REAL(sBeta)[OrderedActive[ii]];
  }
  return(1);
}
////////////////////////////////////////////////////////////////////////////////
//  SmallXtX are parts of t(X) %*% X related to the OrderedActive Coordinates
//
//   Note "OrderedActive" is a vector of length NumActive which has coordinates
//   in increasing order.
//
//
int BayesSpikeCL::FillSmallXtX() {
  if (OrderedActive == NULL) {
    if (AllNewCoords > 0) {
      AddAllNewCoords();
    }
    RefreshOrderedActive(0);
  }
  int OnKSqP;
  if (rSmallXtX == NULL || rIChol == NULL || rI == NULL) {
    if (OnKappaMem  < MaxSquaresAllocation) {
      if (MaxLargeSizeSquares <= 0 || OnKappaMem > MaxLargeSizeSquares ||
        MaxLargeSizeSquares > MaxSquaresAllocation) {
        MaxLargeSizeSquares = OnKappaMem;  
      } else {
        MaxLargeSizeSquares = MaxLargeSizeSquares;
      }
    } else {
      MaxLargeSizeSquares = MaxSquaresAllocation;
    }
    OnKSqP = (MaxLargeSizeSquares*(MaxLargeSizeSquares+1) /2) + 2;
    if (rIChol == NULL) {
      RMemGetD(rIChol, "rIChol", OnKSqP);
    }
    if (rI == NULL) {
      RMemGetD(rI, "rI", OnKSqP);
    }
    if (rSmallXtX == NULL) {
      RMemGetD(rSmallXtX, "rSmallXtX", OnKSqP);
    }
  }
  int ii, jj;  int C1 = 0;
  //int Of1; 
  double *WantO;
  if (NumActive  >  MaxLargeSizeSquares) {
    Rf_error("WantO filling, can't finish NumActive=%d versus Max Squares=%d \n",
      NumActive, MaxLargeSizeSquares);
  }
  if (NumActive == 1) {
    if (OrderedActive == NULL) {
      Rf_error("PrepareForRegression: BayesSpike: Error: OrderedActive is NULL!. \n");
    }
    if (XLC[OrderedActive[0]] < 0 || XLC[OrderedActive[0]] >= OnKappaMem) {
      Rf_error("PrepareForRegression: BayesSpikeError: XLC[OrderedActive[0] =%d] = %d but OnKappaMem = %d!\n",
        OrderedActive[0], XLC[OrderedActive[0]]);
    }
    if (pXtX[XLC[OrderedActive[0]]] == NULL) {
      Rf_error("PrepareForRegression: Error XtX[XLC[OrderedActive[0]=%d]=%d]=NULL!.\n",
        OrderedActive[0], XLC[OrderedActive[0]]);
    }
    rSmallXtX[0] = *(pXtX[XLC[OrderedActive[0]]] + OrderedActive[0]);
    cXtY[0] = XtY[OrderedActive[0]];
    return(1);
  }
  for (ii = 0; ii < NumActive; ii++) {
    //Of1 = p * XLC[OrderedActive[ii]];
    if (XLC[OrderedActive[ii]] < 0 || 
      XLC[OrderedActive[ii]] >= OnKappaS) {
      Rf_error("Error, can't make WantO for ii=%d, OnKappaS=%d/%d, OrderedActive[ii=%d]=%d, XLC[%d]=%d\n",
        ii, OnKappaS, OnKappaMem, ii, OrderedActive[ii], OrderedActive[ii],
        XLC[OrderedActive[ii]]);  
    }
    WantO = pXtX[XLC[OrderedActive[ii]]];
    for (jj = ii; jj < NumActive; jj++) {
      rSmallXtX[C1++] = WantO[OrderedActive[jj]];
    }
  }
  for (ii = 0; ii < NumActive; ii++) {
    cXtY[ii] = XtY[OrderedActive[ii]];
  }
  return(1);
}

// SEtupXjSq
// This is just \sum_i X_ij*X_ij for each column.
// This determines some critical information for Coordinate Descent signals.
int BayesSpikeCL::SetupXjSq() {
  if (XjSq == NULL) {
    RMemGetD(XjSq, "XjSq", p);
  }
  int ii = 0;
  int jj = 0;
  int CI = 0;
  for (jj = 0; jj < p; jj++) {
    for (ii = 0; ii < n; ii++) {
       XjSq[jj] += REAL(sX)[CI] * REAL(sX)[CI]; CI++;
    }
  }
  //F77_CALL(dscal)(&p, &ZeroD, XjSq, &One);
  //char Trans = 'T';  
  //for (ii = 0; ii < n; ii++) {
  //  F77_CALL(dgemv)(&Trans, &n, &p, &OneD, REAL(sX), &n, REAL(sX) + ii * n,
  //    &One, &OneD, XjSq, &One);
  //}
  return(1);
}

int BayesSpikeCL::SetupOrderAttack() {
  int EndAmount = iFirstRandom;
  if (sOnTau == NULL || Rf_isNull(sOnTau) || Rf_length(sOnTau) <= 0 ||
    tauEndList == NULL || Rf_isNull(sOnTau) || Rf_length(sOnTau) <= 0) {
    EndAmount = p;  
  }
  if (EndAmount  <= 0) {
    Rf_error("SetupOrderAttack: We need fixed effects, setup FirstRandom");
  }
  if (OrderAttack == NULL) {
    RMemGetI(OrderAttack, "OrderAttack", EndAmount);
  }
  HeapSortGot(EndAmount, OrderAttack, XtResid);
  ReverseString(EndAmount, OrderAttack);
  return(1);
}

///////////////////////////////////////////////////////////////////////////////
// get_MT()
//
//  MT is composed of all pieces of TauOfFContainer
//    These are pieces of information run in SampleANewTau(iOnii)
//    And are set for the group iOnii just previously run
//  This simple function returns a SEXP with all of the state information.
SEXP BayesSpikeCL::get_MT() {
  SEXP oD = R_NilValue;
  SEXP oR = R_NilValue;
  SEXP oOnii = R_NilValue;
  SEXP sNumJ = R_NilValue;
  SEXP slogOddsPi = R_NilValue;
  SEXP FalseNil = R_NilValue;
  Rf_protect(oD = Rf_allocVector(REALSXP, MT->NumJ));
  Rf_protect(oR = Rf_allocVector(REALSXP, MT->NumJ));
  Rf_protect(oOnii = Rf_allocVector(INTSXP,1));
  Rf_protect(sNumJ = Rf_allocVector(INTSXP,1));
  Rf_protect(slogOddsPi = Rf_allocVector(REALSXP,1));
  Rf_protect(FalseNil = Rf_allocVector(INTSXP,1));
  INTEGER(FalseNil)[0] = -999;
  INTEGER(sNumJ)[0] = MT->NumJ;
  INTEGER(oOnii)[0] = MT->Onii;
  REAL(slogOddsPi)[0] = MT->logOddsPi;
  int One = 1;
  if (MT->D != NULL) {
    F77_CALL(dcopy)(&MT->NumJ, MT->D, &One, REAL(oD), &One);
  }
  double iOnSigma = 1.0 / REAL(sOnSigma)[0];
  SEXP aMTsPriorOftau = MT->sPriorOftau;
  if (Rf_isNull(aMTsPriorOftau)) {aMTsPriorOftau = FalseNil; }
  SEXP asPriorXScaling = MT->sPriorXScaling;
  if (Rf_isNull(asPriorXScaling)) { asPriorXScaling = FalseNil; }
  ////////////////////////////////////////////////////////////
  //  Note that we must re-descale X matrix by current iOnSigma,
  //    which means taking the Diagonals down a notch
  F77_CALL(dscal)(&MT->NumJ, &iOnSigma, REAL(oD), &One);
  F77_CALL(dcopy)(&MT->NumJ, MT->R, &One, REAL(oR), &One);
  SEXP All = R_NilValue;
  Rprintf("Making MT \n"); R_FlushConsole();
  Rf_protect(All = Rf_allocVector(VECSXP, 10));
  SET_VECTOR_ELT(All, 0, oOnii);
  SET_VECTOR_ELT(All, 1, sNumJ);
  SET_VECTOR_ELT(All, 2, oD);        
  SET_VECTOR_ELT(All, 3, oR);
  SET_VECTOR_ELT(All, 4, aMTsPriorOftau);
  SET_VECTOR_ELT(All, 5, MT->staubarnu);
  SET_VECTOR_ELT(All, 6, MT->staupriordf);
  SET_VECTOR_ELT(All, 7, asPriorXScaling);
  SET_VECTOR_ELT(All, 8, slogOddsPi);
  SEXP AllOtherDoubles = R_NilValue;
  Rf_protect(AllOtherDoubles = Rf_allocVector(REALSXP, 16));
  SET_VECTOR_ELT(All, 9, AllOtherDoubles);

  SEXP sStrings = R_NilValue;
  Rf_protect(sStrings = Rf_allocVector(STRSXP, 10));                         
  SET_STRING_ELT(sStrings, 0, Rf_mkChar("Onii integer For this Tau"));
  SET_STRING_ELT(sStrings, 1, Rf_mkChar("NumJ"));
  SET_STRING_ELT(sStrings, 2, Rf_mkChar("D"));
  SET_STRING_ELT(sStrings, 3, Rf_mkChar("R"));
  SET_STRING_ELT(sStrings, 4, Rf_mkChar("PriorOftau"));
  SET_STRING_ELT(sStrings, 5, Rf_mkChar("staubarnuu"));
  SET_STRING_ELT(sStrings, 6, Rf_mkChar("staupriordf"));
  SET_STRING_ELT(sStrings, 7, Rf_mkChar("sPriorXScaling"));
  SET_STRING_ELT(sStrings, 8, Rf_mkChar("slogOddsPi")); 
  SET_STRING_ELT(sStrings, 9, Rf_mkChar("sAllOtherDoubles")); 
  
  REAL(AllOtherDoubles)[0] = MT->lLB42;
  REAL(AllOtherDoubles)[1] = MT->lUB32;
  REAL(AllOtherDoubles)[2] = MT->xM2;
  REAL(AllOtherDoubles)[3] = MT->xM3;
  REAL(AllOtherDoubles)[4] = MT->xM4;
  REAL(AllOtherDoubles)[5] = MT->lM3xM3;
  REAL(AllOtherDoubles)[6] = MT->lM4xM4;
  REAL(AllOtherDoubles)[7] = MT->q4X4;
  REAL(AllOtherDoubles)[8] = MT->f2X4;
  REAL(AllOtherDoubles)[9] = MT->X4;
  REAL(AllOtherDoubles)[10] = MT->UsePrior;
  REAL(AllOtherDoubles)[11] = MT->rUnif1;
  REAL(AllOtherDoubles)[12] = MT->X2;
  REAL(AllOtherDoubles)[13] = MT->q3X2;
  REAL(AllOtherDoubles)[14] = MT->f2X2;
  REAL(AllOtherDoubles)[15] = MT->lB;
  SEXP sStringOfAllOtherDoubles = R_NilValue;
  Rf_protect(sStringOfAllOtherDoubles = Rf_allocVector(STRSXP, 16));                         
  SET_STRING_ELT(sStringOfAllOtherDoubles, 0, Rf_mkChar("lLB42"));
  SET_STRING_ELT(sStringOfAllOtherDoubles, 1, Rf_mkChar("lUB32"));
  SET_STRING_ELT(sStringOfAllOtherDoubles, 2, Rf_mkChar("xM2"));
  SET_STRING_ELT(sStringOfAllOtherDoubles, 3, Rf_mkChar("xM3"));
  SET_STRING_ELT(sStringOfAllOtherDoubles, 4, Rf_mkChar("xM4"));
  SET_STRING_ELT(sStringOfAllOtherDoubles, 5, Rf_mkChar("lM3xM3"));
  SET_STRING_ELT(sStringOfAllOtherDoubles, 6, Rf_mkChar("lM4xM4"));
  SET_STRING_ELT(sStringOfAllOtherDoubles, 7, Rf_mkChar("q4X4"));
  SET_STRING_ELT(sStringOfAllOtherDoubles, 8, Rf_mkChar("f2X4"));
  SET_STRING_ELT(sStringOfAllOtherDoubles, 9, Rf_mkChar("X4"));
  SET_STRING_ELT(sStringOfAllOtherDoubles, 10, Rf_mkChar("UsePrior"));
  SET_STRING_ELT(sStringOfAllOtherDoubles, 11, Rf_mkChar("rUnif1")); 
  SET_STRING_ELT(sStringOfAllOtherDoubles, 12, Rf_mkChar("X2"));
  SET_STRING_ELT(sStringOfAllOtherDoubles, 13, Rf_mkChar("q3X2"));
  SET_STRING_ELT(sStringOfAllOtherDoubles, 14, Rf_mkChar("f2X2"));
  SET_STRING_ELT(sStringOfAllOtherDoubles, 15, Rf_mkChar("lB")); 
  Rf_setAttrib(AllOtherDoubles, R_NamesSymbol, sStringOfAllOtherDoubles );  
  
   Rprintf(" Length All = %d \n",
     Rf_length(All));  R_FlushConsole();
   Rprintf("get_MT: Length(sStrings) = %d\n", Rf_length(sStrings)); R_FlushConsole();
   Rprintf("get_MT: Finished Assigning, now setting attribute.\n"); R_FlushConsole();
   Rprintf("get_MT: We'll just return ALL right now.\n");
   Rf_setAttrib(All, R_NamesSymbol, sStrings );   
   Rprintf("SetupMT: Finished Assigning, now Unprotecting"); R_FlushConsole(); 
   Rf_unprotect(10); return(All);

}


////////////////////////////////////////////////////////////////////////////////
//  DeleteMe()
//
//   The deleting function that performs garbage collection.
//
//   Ideally this ensures all attached SEXPs are detached, all memory is freed
//   from HEAP, and if there is a memory leak, this will usually initiate a segfault.
void BayesSpikeCL::DeleteMe() {
////////////////////////////////////////////////////////////
  //  Re-enter R and Kill TBSRoo, TBSR5
  //Verbose = 8;
  if (Verbose > 0) {
    Rprintf("---------------------------------------------------\n");
    Rprintf("-  Call to DeleteMe,  we're removing the whole shebang\n");
    if (RsSaveDir == NULL || Rf_isNull(sSaveDir)) {
      Rprintf("-  But there is only null SaveDir! \n"); R_FlushConsole();
    } else if (!Rf_isString(sSaveDir)) {
      Rprintf("- But SaveDir is not a satisfactory string! \n"); 
    } else {
      Rprintf("-  Our Save Dir is %s.\n",  CHAR(STRING_ELT(sSaveDir, 0)) ); 
    }
    Rprintf("------------------------------------------------------\n");
    R_FlushConsole();
  }
  if (Verbose > 0) {
    Rprintf("DeleteMe:  We'll call to kill TBSR5, TBSRoo \n"); R_FlushConsole();
  }
  if (AmDestroying > 0) {
    if (Verbose >= -1) {
      Rprintf("DeleteMe: Begin Destroying, won't do it twice \n"); R_FlushConsole();
    }
    return;
  }
  SEXP FreeTBSRoo = R_NilValue;  SEXP FreeTBSR5 = R_NilValue;
  SEXP sVerbose = R_NilValue;  SEXP call = R_NilValue;
  SEXP sNInstance = R_NilValue;
  SEXP RTM = R_NilValue;
  AmDestroying = 1; BeingDestroyed = 1;
  int PTECTED = 0;
  
  if (BayesSpikeNameSpace == R_NilValue) {
    if (Verbose > 0) {
      Rprintf("DeleteMe:  Cannot delete TBSRoo because BayesSpikeNameSpace is Empty\n");
    }
  } else {  
    Rf_protect(FreeTBSR5 = Rf_findVarInFrame( BayesSpikeNameSpace, Rf_install("FreeTBSR5")));
    PTECTED++;
    if (Rf_isNull(FreeTBSR5)) {
      Rprintf("BayesSpikeCL: Error: no FreeTBSR5 -- We will fail to FreeTBSR5. \n");
      Rf_unprotect(PTECTED); PTECTED = 0;
    }  else {
      if (Verbose >= 2) {
        Rprintf("DDeleteMe: BayesSpikeCL: Freeing TBSR5 since it was found. \n"); R_FlushConsole();
      }
      sNInstance = R_NilValue;
      Rf_protect(sNInstance = Rf_allocVector(INTSXP, 1)); PTECTED++;
      sVerbose = R_NilValue;
      Rf_protect(sVerbose = Rf_allocVector(INTSXP,1));  PTECTED++;
      INTEGER(sVerbose)[0] = Verbose;
      INTEGER(sNInstance)[0] = NInstance;
      call = R_NilValue;
      Rf_protect(call = Rf_lcons( FreeTBSR5, 
        Rf_cons(sNInstance, 
          Rf_cons(sVerbose, R_NilValue)))); PTECTED++;
      if (Rf_isNull(call)) {
        Rf_unprotect(PTECTED); PTECTED = 0;
        Rf_error("BayesSpikeCL Destroy: Call is a null to delete TBSR5\n");
      }
      RTM = R_NilValue;
      if (Verbose > 0) {
        Rprintf("DeleteMe: We've got the call to delete TBSR5, now getting to work\n "); R_FlushConsole();
      }
      Rf_protect(RTM = Rf_eval(call, get_BayesSpikeNameSpace())); PTECTED++;
      if (Verbose > 0 ) {
        Rprintf("DDeleteMe: We have run Free TBSR5. We've gotten the Free object\n"); R_FlushConsole();
      }
      if (Verbose > 0) {
        Rprintf("DeleteMe: We finished running program FreeTBSR5 \n"); R_FlushConsole();
      }
      Rf_unprotect(PTECTED); PTECTED = 0;
      if (Verbose > 0) {
        Rprintf("DeleteMe: Now consider delete TBRS5 we have attached. \n"); R_FlushConsole();
      }
      Rf_protect(FreeTBSR5 = Rf_findVarInFrame( BayesSpikeNameSpace, Rf_install("KillFreeTBSR5")));
      PTECTED++;
      call = R_NilValue;
      SEXP TBSR5 = get_TBSR5();
      if (!Rf_isNull(TBSR5)) {
        if (Verbose > 0) {
          Rprintf("DeleteMe: We still managed to find TBSR5 to try to kill!\n");
        }
        Rf_protect(call = Rf_lcons( FreeTBSR5, 
          Rf_cons(get_TBSR5(), 
            Rf_cons(sVerbose, R_NilValue)))); PTECTED++;
        if (Rf_isNull(call)) {
          Rf_unprotect(PTECTED); PTECTED = 0;
          Rf_error("BayesSpikeCL Destroy: Call is a null to delete TBSR5\n");
        }
        RTM = R_NilValue;
        if (Verbose > 0) {
          Rprintf("DeleteMe: We've got the call to delete TBSR5, now getting to work\n "); R_FlushConsole();
        }
        Rf_protect(RTM = Rf_eval(call, get_BayesSpikeNameSpace())); PTECTED++;
        if (Verbose > 0) {
          Rprintf("DeleteMe: Just conducted second TBSR5 KillFree. \n");
        }
      }
      
      Rf_unprotect(PTECTED); PTECTED = 0;
    }
  
    Rf_protect(FreeTBSRoo = Rf_findVarInFrame( get_BayesSpikeNameSpace(), Rf_install("FreeTBSRoo")));
    PTECTED++;
    if (Rf_isNull(FreeTBSRoo)) {
      Rprintf("BayesSpikeCL: Error: no FreeTBSRoo\n");
      Rf_unprotect(PTECTED); PTECTED = 0;
    } else {
      sNInstance = R_NilValue;
      Rf_protect(sNInstance = Rf_allocVector(INTSXP, 1)); PTECTED++;
      sVerbose = R_NilValue;
      Rf_protect(sVerbose = Rf_allocVector(INTSXP,1));  PTECTED++;
      INTEGER(sVerbose)[0] = Verbose;
      INTEGER(sNInstance)[0] = NInstance;
      call = R_NilValue;
      Rf_protect(call = Rf_lcons( FreeTBSRoo, 
        Rf_cons(sNInstance, 
          Rf_cons(sVerbose, R_NilValue)))); PTECTED++;
      if (Rf_isNull(call)) {
        Rf_unprotect(PTECTED); PTECTED = 0;
        Rf_error("BayesSpikeCL Destroy: Call is a null to delete TBSRoo\n");
      }
      RTM = R_NilValue;
      if (Verbose > 0) {
        Rprintf("DeleteMe: We've got the call to delete TBSRoo, now getting to work\n "); R_FlushConsole();
      }
      Rf_protect(RTM = Rf_eval(call, get_BayesSpikeNameSpace())); PTECTED++;
      if (Verbose > 0 ) {
        Rprintf("We've gotten the Free object\n"); R_FlushConsole();
      }
      if (Verbose > 0) {
        Rprintf("DeleteMe: We got program FreeTBSRoo \n"); R_FlushConsole();
      }
      Rf_unprotect(PTECTED); PTECTED = 0;
    }
    if (PTECTED > 0) {
      Rf_unprotect(PTECTED); PTECTED = 0;
    }
  }
  if (RTBSR5 != NULL) {
    if (Verbose >= -1) {
      Rprintf(" Final Deleting RTBSR5: good luck, in position %d of .AccpList \n", RTBSR5->getInVectorLoc());
      R_FlushConsole();
    }
    DDelete(RTBSR5, "RTBSR5");
  }
  if (Verbose >= -1) {
    Rprintf("BayesSpikeCpp.cpp:DeleteMe(): Deleted TBSR5: we are onto the rest of members"); R_FlushConsole();
  }
  if (RAFD == NULL) {
    if (Verbose >= -1) {
      Rprintf("BayesSpikeCpp.cpp:DeleteMe(): RAFD is NULL. Don't have to delete. \n"); R_FlushConsole();
    }
  } else if (RAFD!=NULL) {
    if (Verbose > 1) {
      Rprintf(" Deleting RAFD  \n"); R_FlushConsole();
    }
    SEXP FreeAFD = R_NilValue;
    Rf_protect(FreeAFD = Rf_findVarInFrame( get_BayesSpikeNameSpace(),
      Rf_install("FinalizeCAFD"))); PTECTED++;
    SEXP sAFD = RAFD->asSexp();
    if (Rf_isNull(sAFD)) {
      Rprintf("DDeleteMe: RAFD delete, Ooh, weird, sAFD was already NULL \n"); R_FlushConsole();
      Rf_unprotect(PTECTED); PTECTED = 0;
    } else {
    SEXP sVerbose2 = R_NilValue;
    Rf_protect(sVerbose2 = Rf_allocVector(INTSXP,1)); PTECTED++; 
    INTEGER(sVerbose2)[0] = Verbose;
    SEXP call2 = R_NilValue;
    Rf_protect(call2 = Rf_lcons( FreeAFD, 
      Rf_cons(sAFD,
        Rf_cons(sVerbose2, R_NilValue)))); PTECTED++;
    SEXP RTM2 = R_NilValue;
    if (Verbose > 0) {
      Rprintf("DeleteMe: Now Called to delete RAFD\n "); R_FlushConsole();
    }
    Rf_protect(RTM2 = Rf_eval(call2, get_BayesSpikeNameSpace())); PTECTED++;
    if (Verbose > 0) {
      Rprintf("DeleteMe: Well, at least we finished deleting, I think\n "); R_FlushConsole();
    }
    }
     Rf_unprotect(PTECTED); PTECTED = 0;
     DDelete(RAFD, "RAFD");   AFD = R_NilValue;
  }

  if (Verbose >= -1) {
    Rprintf("BayesSpikeCpp.h:DeleteMe(), AFT Deleted. getting away with DDeleting RObjects\n"); R_FlushConsole();
  }
  
       AmDestroying = 1;
     BeingDestroyed = 1;
     
     DDelete(RsBeta, "RsBeta");   DDelete(RsRandomInfoList, "RsRandomInfoList");
     
     DDelete(RtauEndList, "RtauEndList"); tauEndList = R_NilValue;
     DDelete(RsY, "RsY"); DDelete(RsX, "RsX");  sY = R_NilValue; sX = R_NilValue;
     DDelete(RtauFixed, "RtauFixed");  DDelete(RsOnTau, "RsOnTau"); sOnTau = R_NilValue;
     DDelete(RsOnSigma, "RsOnSigma"); sOnSigma = R_NilValue; DDelete(RSigmaPrior, "RSigmaPrior");
     DDelete(RInitFlags, "RInitFlags"); DDelete(RsOnPiA, "RsOnPiA"); sOnPiA = R_NilValue;
     DDelete(RPiAPrior, "RPiAPrior");    PiAPrior = R_NilValue;
     
     DDelete(RDependenciesFixed, "RDependenciesFixed"); DependenciesFixed = R_NilValue; 
     DDelete(RDependenciesTau, "RDependenciesTau"); DependenciesTau = R_NilValue; 
     if (Verbose >= -1) {
       Rprintf("BayesSpikeCpp.h:DeleteMe(): About to free MergeActive and MergePropBeta. \n"); R_FlushConsole();
     }
  
      if (MergeActive != NULL) {
       LengthMergeActive = 0; FFree(MergeActive, "MergeActive"); 
       FFree(MergePropBeta, MergePropBeta);
     }
     
     if (Verbose >= -1) {
       Rprintf("BayesSpikeCpp.cpp:DeleteMe(): Onto EigenVectors and Values\n"); R_FlushConsole();
     }
     DDelete(RAllEigenVectors, "RAllEigenVectors");
     DDelete(RAllEigenValues, "RAllEigenValues");  
     AllEigenVectors = R_NilValue; AllEigenValues = R_NilValue; 
     DDelete(Rstaupriordf, "Rstaupriordf"); DDelete(Rstaubarnu, "Rstaubarnu");  
     staubarnu = R_NilValue; staupriordf = R_NilValue;
     
     if (Verbose >= -1) {
       Rprintf("BayesSpikeCpp.cpp:DeleteMe(): Onto Delete RsPriorXScaling. \n"); R_FlushConsole();
     }
     DDelete(RsPriorXScaling, "RsPriorXScaling"); DDelete(RsPriorOftau, "RsPriorOftau"); 
     sPriorXScaling = R_NilValue; sPriorOftau = R_NilValue;     
     
     if (Verbose >= -1) {
       Rprintf("BayesSpikeCpp:DeleteMe(): Deleting RCodaTable. \n"); R_FlushConsole();
     }
     DDelete(RCodaTable, "RCodaTable");  CodaTable = R_NilValue;        
     DDelete(RDoRecord, "RDoRecord");  
     DDelete(RCodaList, "RCodaList");  CodaList = R_NilValue;
     DDelete(RAFD, "RAFD");   AFD = R_NilValue;  
     
     
      if (Verbose >= -1) {
       Rprintf("BayesSpikeCpp:DeleteMe(): Deleting CodaFiles. \n"); R_FlushConsole();
     }
     DDelete(RsCodaIFile, "RsCodaIFile"); 
     DDelete(RsCodaJFile, "RsCodaJFile"); 
     DDelete(RsCodadTFile, "RsCodadTFile"); 
     DDelete(RsCodaiTFile, "RsCodaiTFile"); 
     DDelete(RsCodaPFile, "RsCodaPFile");
     DDelete(RsCodaProbFile, "RsCodaProbFile"); 
     DDelete(RsCodaILocFile, "RsCodaILocFile");
     DDelete(RsCodaOldIFile, "RsCodaOldIFile");
     DDelete(RsCodaOldJFile, "RsCodaOldJFile");
     DDelete(RsOldCodaILocFile, "RsOldCodaILocFile");
     DDelete(RsOldCodaProbFile, "RsOldCodaProbFile");
     DDelete(RsSaveDir, "RsSaveDir");
    
     if (Verbose >= -1) {
       Rprintf("BayesSpikeCpp:DeleteMe(): Deleting PiACodaFile and Buffer. \n"); R_FlushConsole();
     } 
     DDelete(RsPiACodaFile, "RsPiACodaFile");  
     FFree(PiACodaBuffer, "PiACodaBuffer"); 
     DDelete(RsSigCodaFile, "RsCodaSigFile");
     FFree(SigCodaBuffer, "SigCodaBuffer");
     
     DDelete(RsPostProbBufferFile, "RsPostProbBufferFile");  
     FFree(PostProbCodaBuffer, "ProbCodaBuffer");
      
      
     if (Verbose >= -1) {
       Rprintf("BayesSpikeCpp:DeleteMe(): Deleting YFile and such. \n"); R_FlushConsole();
     }
     DDelete(RsYFile, "RsYFile"); 
       FFree(YBuffer, "YBuffer");
     DDelete(RsWeightFile, "RsWeightFile"); 
       FFree(WeightBuffer, "WeightBuffer");  
     DDelete(RsYCodaList, "RsYCodaList");  
     DDelete(RsWeightCodaList, "RsWeightCodaList"); 
     DDelete(RsPiACodaList, "RsPiACodaList");  
     DDelete(RsSigCodaList, "RsSigCodaList"); 
     
     DDelete(RsCodaiTLocFile, "RsCodaiTLocFile");
     
     DDelete(RsAlterWeightFile, "RsAlterWeightFile"); 
     FFree(AlterWeightBuffer, "AlterWeightBuffer");   
     DDelete(RsAlterWeightCodaList, "RsAlterWeightCodaList");      
              
     if (Verbose >= -1) {
       Rprintf("BayesSpikeCpp:DeleteMe(): Delete RunProbVector and Such. \n"); R_FlushConsole();
     }
     FFree(RunProbVector, "RunProbVector"); FFree(TotEveryProbVector, "TotEveryProbVector");
     FFree(PropBeta, ("PropBeta" ));  FFree(rI, "rI");  FFree(rIChol, "rIChol");
     FFree(rSmallXtX, "SmallXtX" );  FFree(XLC, "XLC"); FFree(cXtY, "cXtY"); 
     FFree(kXFinder, "kXFinder" ); FFree(iiWeight, "iiWeight");
     FFree(CodaBeta, "CodaBeta"); FFree(Codajj, "Codajj");
     FFree(XtResid, "XtResid");  
     if (pXtX != NULL) {
       if (Verbose >= -1) {
         Rprintf("In DDelete: Free pXtX\n"); R_FlushConsole();
       }
     
       for (int ii = 0; ii < OnKappaMem; ii++) {
         if (pXtX[ii] != NULL) {
           if (Verbose >= 6) {
             Rprintf("       Delete pXtX[ii=%d] \n", ii); R_FlushConsole();
           }
           Free(pXtX[ii]);  pXtX[ii] = NULL;
         }
       }
       if (Verbose >= 3) {
         Rprintf("Delete pXtX, delete whole vector\n"); R_FlushConsole();
       }
       Free(pXtX); pXtX = NULL;
     }
     //FFree(XtX, "XtX");
     FFree(XtY, "XtY"); FFree(WW, "WW"); 
     
     FFree(ProbFixed, "ProbFixed");
     FFree(ProbTau, "ProbTau"); 
     DDelete(rPriorProbFixed, "rPriorProbFixed"); sPriorProbTau = R_NilValue;
     DDelete(rPriorProbTau, "rPriorProbTau"); sPriorProbFixed = R_NilValue;
     
     ffree(MT, "MT");  FFree(OrderAttack, "OrderAttack");
     FFree(SmallXtResid, "SmallXtResid");  FFree(SmallRVec, "SmallRVec");
     FFree(cXtY, "cXtY");  FFree(XjSq, "XjSq");  
     
     if (Verbose >= -1) {
       Rprintf("DDeleteMe(): Freeing some Coda. \n");
     }
     FFree(CodaP, "CodaP");  FFree(CodaTau, "CodaTau");  CodaTau = NULL; 
     FFree(CodaTjj, "CodaTjj"); CodaTjj = NULL;
     FFree(CodaTLocjj, "CodaTLocjj");
     FFree(NoShrinkFixed, "NoShrinkFixed");
     FFree(NoShrinkFixedPrior, "NoShrinkFixedPrior");
     FFree(NoShrinkRandom, "NoShrinkRandom");
     FFree(NoShrinkRandomPrior, "NoShrinkRandomPrior");
     FFree(XtXRecorded, "XtXRecorded");
     FFree(XjSq, "XjSq"); FFree(BFixed, "BFixed");  FFree(iWeightedXtX, "iWeightedXtX");
     FFree(smXtY, "smXtY"); FFree(rICholBack, "rICholBack");
     
     
     if (Verbose >= 1) {
       Rprintf("DDeleteMe(): Freeing Temperature Information. \n"); R_FlushConsole();
     }
     FFree(LengthTempDraws, "LengthTempDraws"); FFree(NumMerges, "NumMerges");
     FFree(TemperatureList, "TemperatureList");
     
     if (Verbose >= 1) {
       Rprintf("DDeleteMe(): Freeing important old Coda lists. \n"); R_FlushConsole();
     }
     FFree(ProbCoda, "ProbCoda");  FFree(ProbOldCoda, "OldProbCoda");
     FFree(IOldCoda, "IOldCoda");  FFree(ICoda, "ICoda");
     
     FFree(PctBetas, "PctBetas");  FFree(PctTaus, "PctTaus");
     FFree(intY, "intY");
     
     DDelete(InitiateTime, "InitiateTime");
     DDelete(SecondInitiateTime, "SecondInitiateTime");
     DDelete(CompleteTime, "CompleteTime");
     DDelete(RInitFlags, "RInitFlags");
     DDelete(RsTimePartsList, "RsTimePartsList");
     DDelete(RsBeta, "RsBeta");   DDelete(RsRandomInfoList, "RsRandomInfoList");
     DDelete(RBayesSpikeNameSpace, "RBayesSpikeNameSpace");
     BayesSpikeNameSpace = R_NilValue;
     DDelete(RtauEndList, "RtauEndList"); DDelete(RsY, "RsY");
     DDelete(RsX, "RsX"); sX = NULL;
     DDelete(RtauFixed, "RtauFixed");  tauFixed=R_NilValue;
     DDelete(RsOnTau, "RsOnTau"); 
     DDelete(RsOnSigma, "RsOnSigma"); DDelete(RSigmaPrior, "RSigmaPrior");
     DDelete(RInitFlags, "RInitFlags"); DDelete(RsOnPiA, "RsOnPiA");
     DDelete(RPiAPrior, "RPiAPrior"); 
     this->sX = R_NilValue;  this->sY = R_NilValue;
     this->PiAPrior = R_NilValue;  this->SigmaPrior = R_NilValue;
     this->tauEndList = R_NilValue;  this->sOnTau = R_NilValue;
     this->CodaTable = R_NilValue;  this->CodaList = R_NilValue;
     this->sBeta = R_NilValue;  


     DDelete(RsCodaIFile, "RsCodaIFile"); 
     DDelete(RsCodaJFile, "RsCodaJFile"); 
     DDelete(RsCodadTFile, "RsCodadTFile"); 
     DDelete(RsCodaiTFile, "RsCodaiTFile"); 
     DDelete(RsCodaPFile, "RsCodaPFile");
     DDelete(RsCodaProbFile, "RsCodaProbFile");
     DDelete(RsCodaILocFile, "RsCodaILocFile");
     FFree(NumMerges, "NumMerges");
     
   DDelete(RAFD, "RAFD");
     AFD = R_NilValue; 
   sCodaIFile = R_NilValue;  sCodaJFile = R_NilValue;  
    sCodadTFile = R_NilValue; sCodaiTFile = R_NilValue; sCodaPFile = R_NilValue;
   sPriorXScaling = R_NilValue; sPriorOftau = R_NilValue; 
    DependenciesList = R_NilValue;  tauFixed = R_NilValue;
  ffree(this->MT, "MT");
  FFree(this->XtResid, "XtResid");  
  if (this->pXtX != NULL) {
    if (Verbose >= 3) {
      Rprintf("DeleteMe:  Deleting pXtX\n"); R_FlushConsole();
    }
    for (int ii = 0; ii < OnKappaMem; ii++) {
      if (this->pXtX[ii] != NULL) {
        if (Verbose >= 3) {
          Rprintf("       Deleting pXtX[ii=%d/%d]\n", ii, OnKappaMem); R_FlushConsole();
        }
        Free(this->pXtX[ii]);  this->pXtX[ii] = NULL;
      }
    }
    if (Verbose >= 3) {
      Rprintf("DeleteMe:  Really Deleting final pXtX vector\n"); R_FlushConsole();
    }
    Free(this->pXtX);  this->pXtX = NULL;
  }
  //FFree(this->pXtX, "XtX"); 
  FFree(XLC, "XLC");

  if (Verbose >= -1) {
    Rprintf("DeleteMe: Get rid of access to NamesSpace\n"); R_FlushConsole();
  } 
  DDelete(RBayesSpikeNameSpace, "RBayesSpikeNameSpace");
  RBayesSpikeNameSpace = NULL;
  BayesSpikeNameSpace = R_NilValue;
  if (PTECTED > 0) {
    Rf_unprotect(PTECTED);
  }
  if (Verbose >= -1) {
    Rprintf("DeleteMe: Concluded, Let's return.\n"); R_FlushConsole();
    Rprintf("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD\n");
    R_FlushConsole();
  }
  return;

}

// Call DeleteMe()
void DDeleteStuff(BayesSpikeCL *AB) {
  AB->DeleteMe();
}

///////////////////////////////////////////////////////////////////////////////
//  "FinalizerForBayesSpikeCL()"
//
//  A Finalizer is an operation that R garbage collection will call to
//   delete a Module object never determined.
//  Rooting out and fully deleting BayesSpike is the hardest challenge.
//  However, typically, once DeleteMe has been called, most of the memory
//  trapped in the BayesSpike Object has been deleted.
//
void FinalizerForBayesSpikeCL(BayesSpikeCL *AcL) {
  if (AcL->BeingDestroyed == 1) {
    Rprintf("FinalizerForBayesSpikeCL:  Big Bad issue, you're already destroying it!"); 
    R_FlushConsole();
  }
  ////////////////////////////////////////////////////////////
  //  Re-enter R and Kill TBSRoo
  AcL->DeleteMe();
  return;
}

double BayesSpikeCL::ProbLDraw() {
  double On = 0.0;  int ii = 0;
  int EndList = iFirstRandom;
  if (tauEndList == NULL || Rf_isNull(tauEndList) || Rf_length(tauEndList) <= 0 ||
    sOnTau == NULL || Rf_isNull(sOnTau) || Rf_length(sOnTau) <= 0) {
    EndList = p;  
  }
  if (EndList > 0) {
    for (ii = 0; ii < EndList; ii++){
      On += ProbFixed[ii];
    }
  }
  if (sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) > 0 &&
    tauEndList != NULL && !Rf_isNull(tauEndList) && Rf_length(tauEndList) > 0
    && !(ProbTau == NULL)) {
    for (ii = 0; ii < EndList; ii++) {
      On += ProbTau[ii];
    }
  }
  return(On);
}
double BayesSpikeCL::TestPnorm(SEXP sP, SEXP sMu, SEXP sSig, SEXP sLTF, SEXP sLog) {

double PP = 0.0;
if (!Rf_isNull(sP) && (Rf_isReal(sP))) {
  PP = REAL(sP)[0];
}
double Mu = 0.0;
if (!Rf_isNull(sMu) && (Rf_isReal(sMu))) {
  Mu = REAL(sMu)[0];
}
double Sig = 1.0;
if (!Rf_isNull(sSig) && (Rf_isReal(sSig))) {
  Sig = REAL(sSig)[0];
}
int doLeft = 1;
if (!Rf_isNull(sLTF) && (Rf_isReal(sLTF) || Rf_isInteger(sLTF))) {
  doLeft = GetFirstInteger(sLTF);
}
int doLog = 0;
if (!Rf_isNull(sLog) && (Rf_isReal(sLog) || Rf_isInteger(sLog))) {
  doLog = GetFirstInteger(sLog);
}
return(Rf_pnorm5(PP, Mu, Sig, doLeft, doLog));
}

double BayesSpikeCL::TestPt(SEXP sP, SEXP sDf, SEXP sLTF, SEXP sLog) {

double PP = 0.0;
if (!Rf_isNull(sP) && (Rf_isReal(sP))) {
  PP = REAL(sP)[0];
}

double Df = 6.0;
if (!Rf_isNull(sDf) && (Rf_isReal(sDf))) {
  Df = REAL(sDf)[0];
}
int doLeft = 1;
if (!Rf_isNull(sLTF) && (Rf_isReal(sLTF) || Rf_isInteger(sLTF))) {
  doLeft = GetFirstInteger(sLTF);
}
int doLog = 0;
if (!Rf_isNull(sLog) && (Rf_isReal(sLog) || Rf_isInteger(sLog))) {
  doLog = GetFirstInteger(sLog);
}
return(Rf_pt(PP, Df, doLeft, doLog));
}

////////////////////////////////////////////////////////////////////////////////
//  ProbPosterior()
//
//     Function for assembling the current Posterior functions.
//
//     This includes the unnormalized Posterior value  P(Theta|Y) for a 
//  MCMC sample of Theta, useful in importance sampling.
//
double BayesSpikeCL::ProbPosterior() {
  double On1 = 0.0; int ii = 0;  double On2 = 0.0;  double On = 0.0;
  int SC = 0;
  if (Verbose > 5) {
  if (rPriorProbFixed == NULL) {
    Rprintf("ProbPosterior: rPriorProbFixed is NULL\n"); R_FlushConsole();
  } else if (Rf_isNull(rPriorProbFixed->asSexp())) {
    Rprintf("ProbPosterior: rPriorProbFixed->asSexp() is a null value\n"); R_FlushConsole();
  }
  if (rPriorProbTau == NULL) {
    Rprintf("ProbPosterior: rPriorProbTau is NULL\n"); R_FlushConsole();
  } else if (Rf_isNull(rPriorProbTau->asSexp())) {
    Rprintf("ProbPosterior: rPriorProbTau->asSexp() is a null value\n"); R_FlushConsole();
  }
  }
  int EndList = iFirstRandom;
  if (sOnTau == NULL || Rf_isNull(sOnTau) || Rf_length(sOnTau) <= 0 ||
    tauEndList == NULL || Rf_isNull(tauEndList) || Rf_length(tauEndList) <= 0) {
    EndList = p;  
  } 
  int Stt = iFirstRandom; int JT = 0;
  if (!Rf_isNull(tauFixedPrior) && Rf_length(tauFixedPrior) >= 2) {
    for (ii = 0; ii < Rf_length(tauFixed);ii++) {
      if (Rf_length(tauFixedPrior) >= 2+2* ii) {
        On1 += REAL(tauFixedPrior)[2*ii] *(log(REAL(tauFixedPrior)[2*ii+1]) - 
          log(fabs(REAL(tauFixed)[ii]))) - REAL(tauFixedPrior)[2*ii+1]/fabs(REAL(tauFixed)[ii]);
      } else {
        On1 += REAL(tauFixedPrior)[0] *(log(REAL(tauFixedPrior)[1]) - 
          log(fabs(REAL(tauFixed)[ii]))) - REAL(tauFixedPrior)[1]/fabs(REAL(tauFixed)[ii]);
      }
    }
  }
  if (EndList > 0) {
    if (rPriorProbFixed != NULL && !Rf_isNull(rPriorProbFixed->asSexp())) {
      if (Verbose > 5) {
        Rprintf("ProbPosterior: Fixed issues Gone into using rPriorProbFixed!\n"); R_FlushConsole();
      }
      if (RtauFixed == NULL || tauFixed == NULL || Rf_isNull(tauFixed)  ||
        Rf_length(tauFixed) <= 0) {
        Rf_error("ProbPosterior: Why is tauFixed of NULL Length?\n");  
      }

      for (ii = 0; ii < EndList;ii++) {
        if (BFixed[ii] == 1 && REAL(rPriorProbFixed->asSexp())[ii] > 0.0 && 
          REAL(rPriorProbFixed->asSexp())[ii] < 1.0) {
           On1 += log(REAL(rPriorProbFixed->asSexp())[ii]) - 
             log(1.0 - REAL(rPriorProbFixed->asSexp())[ii]);
           if (TypePrior <= 1) {
             if (Rf_length(tauFixed) > ii) {
              if (REAL(tauFixed)[ii] > 0.0) {
               On1 += -.5 * REAL(sBeta)[ii] * REAL(sBeta)[ii] / REAL(tauFixed)[ii]-
                 .5 * log(REAL(tauFixed)[ii])-logSqrt2PI;
              } else {
               On1 += -.5 * REAL(sBeta)[ii] * REAL(sBeta)[ii] / (fabs(REAL(tauFixed)[ii]) * REAL(sOnSigma)[0])-
                 .5 * log(fabs(REAL(tauFixed)[ii]) * REAL(sOnSigma)[0])-logSqrt2PI;           
              }
             } else {
               if (REAL(tauFixed)[0] > 0.0) {
                 On1 += -.5 * REAL(sBeta)[ii] * REAL(sBeta)[ii] / REAL(tauFixed)[0]-
                   .5 * log(REAL(tauFixed)[0])-logSqrt2PI;
               } else {
                 On1 += -.5 * REAL(sBeta)[ii] * REAL(sBeta)[ii] / (REAL(sOnSigma)[0] * fabs(REAL(tauFixed)[0]))-
                   .5 * log(REAL(sOnSigma)[0] * fabs(REAL(tauFixed)[0]))-logSqrt2PI;               
               }
             }
          } else if (TypePrior == 3) {
            if (Rf_length(tauFixed) > ii) {
              if (REAL(tauFixed)[ii] > 0.0) {
              On1 += - fabs(REAL(sBeta)[ii] * REAL(tauFixed)[ii]) +
                log(REAL(tauFixed)[ii]);
              } else {
              On1 += - fabs(REAL(sBeta)[ii] * REAL(sOnSigma)[0] * fabs(REAL(tauFixed)[ii])) +
                log(REAL(sOnSigma)[0] * fabs(REAL(tauFixed)[ii]));              
              }
            } else {
              if (REAL(tauFixed)[0] > 0.0) {
              On1 += - fabs(REAL(sBeta)[ii] *  REAL(tauFixed)[0] ) +
                log(REAL(tauFixed)[0]);
              } else {
              On1 += - fabs(REAL(sBeta)[ii] *  REAL(sOnSigma)[0] * fabs(REAL(tauFixed)[0]) ) +
                log(REAL(sOnSigma)[0] * fabs(REAL(tauFixed)[0]));              
              }
            }          
          } else if (TypePrior == 4) {
            On1 -= log( REAL(tauFixed)[1] - REAL(tauFixed)[0]);
          }
        }
      }
    } else {
      if (Verbose > 5) {
        Rprintf("ProbPosterior: Fixed issues using PiAPrior!\n"); 
        R_FlushConsole();
      }
      SC = 0;
      for (ii = 0; ii < EndList; ii++) {
        if (BFixed[ii] == 1) { SC++; 
          if (TypePrior <= 1) {
            if (Rf_length(tauFixed) > ii) {
              if (REAL(tauFixed)[ii] > 0.0) {
              On1 += -.5 * REAL(sBeta)[ii] * REAL(sBeta)[ii] / REAL(tauFixed)[ii]-
                .5 * log(REAL(tauFixed)[ii])-logSqrt2PI;
              } else {
              On1 += -.5 * REAL(sBeta)[ii] * REAL(sBeta)[ii] / (REAL(sOnSigma)[0] * fabs(REAL(tauFixed)[ii]))-
                .5 * log(REAL(sOnSigma)[0] * fabs(REAL(tauFixed)[ii]))-logSqrt2PI;              
              }
            } else {
              if (REAL(tauFixed)[0] > 0.0) {
              On += -.5 * REAL(sBeta)[ii] * REAL(sBeta)[ii] / REAL(tauFixed)[0]-
                .5 * log(REAL(tauFixed)[0])-logSqrt2PI;
              } else {
              On += -.5 * REAL(sBeta)[ii] * REAL(sBeta)[ii] / (REAL(sOnSigma)[0] * fabs(REAL(tauFixed)[0]))-
                .5 * log(REAL(sOnSigma)[0] * fabs(REAL(tauFixed)[0]))-logSqrt2PI;
              }
            }
          } else if (TypePrior == 3) {
            if (Rf_length(tauFixed) > 3 && Rf_length(tauFixed) > ii) {
              if (REAL(tauFixed)[ii] > 0.0) {
              On1 += - fabs(REAL(sBeta)[ii] * REAL(tauFixed)[ii] )+
                 log(REAL(tauFixed)[ii]);
              } else {
              On1 += - fabs(REAL(sBeta)[ii] * REAL(sOnSigma)[0] * fabs(REAL(tauFixed)[ii]) )+
                 log(REAL(sOnSigma)[0] * fabs(REAL(tauFixed)[ii]));              
              }
            } else {
              if (REAL(tauFixed)[0] > 0.0) {
              On1 += - fabs(REAL(sBeta)[ii] *  REAL(tauFixed)[0] ) +
                 log(REAL(tauFixed)[0]);
              } else {
              On1 += - fabs(REAL(sBeta)[ii] *  REAL(sOnSigma)[0] * fabs(REAL(tauFixed)[0]) ) +
                 log(REAL(sOnSigma)[0] * fabs(REAL(tauFixed)[0]));              
              }
            }          
          } else if (TypePrior == 4) {
            On1 -= log( REAL(tauFixed)[1] - REAL(tauFixed)[0]);
          }
        }
      }
      if (!Rf_isNull(sOnPiA) && Rf_length(sOnPiA) >= 1) {
        On1 += ((double) SC) * ( log( REAL(sOnPiA)[0] ) - log(1.0 - REAL(sOnPiA)[0] )) ;
      }
      //if (RPiAPrior != NULL && !Rf_isNull(PiAPrior)  && Rf_length(PiAPrior) >= 2 &&  
      //    Rf_isReal(PiAPrior) && REAL(PiAPrior)[0] > 0) {
      //    On1 += REAL(PiAPrior)[0] * log( REAL(sOnPiA)[0] ) +
      //    REAL(PiAPrior)[1] * log(1.0 - REAL(sOnPiA) [0] );
      //}  
    }
    
  }
           
       

  if (sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) > 0) {
    if (rPriorProbTau != NULL && !Rf_isNull(rPriorProbTau->asSexp())) {
      if (Verbose > 5) {
        Rprintf("ProbPosterior: Random issues using PriorProbTau!\n"); R_FlushConsole();
      }
      Stt = iFirstRandom;
      for (ii = 0; ii < Rf_length(sOnTau); ii++) {
        if (REAL(sOnTau)[ii] > 0.0) {
           if (REAL(rPriorProbTau->asSexp())[ii] > 0.0 
            && REAL(rPriorProbTau->asSexp())[ii] < 1.0) {
              On2 += log(REAL(rPriorProbTau->asSexp())[ii]) - 
               log(1.0 - REAL(rPriorProbTau->asSexp())[ii]);           
            } else if (REAL(rPriorProbTau->asSexp())[ii] >= 1.0) {
            
            } else if (REAL(rPriorProbTau->asSexp())[ii] < 0.0) {
              if (Rf_length(sOnPiA) == 2) {
                On2 += log(REAL(sOnPiA)[1]) - log(1.0-REAL(sOnPiA)[1]);
              } else {
                On2 += log(REAL(sOnPiA)[0]) - log(1.0-REAL(sOnPiA)[0]);           
              }            
            }
            On2 +=  GivePriorOfX(REAL(sOnTau)[ii], MT); 
            if (ii > 0) { Stt = ANINT(tauEndList, (ii-1)) +1; } else {
              Stt = iFirstRandom;
            }  
            JT = ANINT(tauEndList,ii);
            for (int jj = Stt; jj <= JT; jj++) {
             On2 += -REAL(sBeta)[jj]  * REAL(sBeta)[jj] * .5 / REAL(sOnTau)[ii] - 
               .5 * log(REAL(sOnTau)[ii])-logSqrt2PI;            
            }    
        
        }
        if (R_isnancpp(On2)) {
          Rf_error("Using PriorProbTau: We hit a nan on On2, ii = %d,  Stt = %d, end = %d",
            ii, Stt, ANINT(tauEndList, ii));
        }
      }
    } else {
      if (Verbose > 5) {
        Rprintf("ProbPosterior: Random issues using PriorPiA!\n"); R_FlushConsole();
      }
      SC = 0;

      if (tauEndList == NULL || Rf_isNull(tauEndList)  || Rf_length(tauEndList) <= 0) {
      
      } else {
        JT = (ANINT(tauEndList, (0)));
        for (ii = 0; ii < Rf_length(sOnTau); ii++) {
          if (REAL(sOnTau)[ii] > 0.0) { SC++; 
            JT = (ANINT(tauEndList, (ii)));
            On2 +=  GivePriorOfX(REAL(sOnTau)[ii], MT);
            if (ii > 0) { Stt = ANINT(tauEndList, (ii-1)) + 1; }
            for (int jj = Stt; jj <= JT; jj++) {
              On2 += -REAL(sBeta)[jj]  * REAL(sBeta)[jj] * .5 / REAL(sOnTau)[ii] - 
                .5 * log(REAL(sOnTau)[ii]) - logSqrt2PI;
            }
            if (R_isnancpp(On2)) {
              Rf_error("Using OnPiA on Tau: We hit a nan on On2, ii = %d,  Stt = %d, end = %d",
                ii, Stt, ANINT(tauEndList, ii));
            }
          }
        }
      }
      double APiA;
      if (Rf_length(sOnPiA) == 2) {
        APiA = REAL(sOnPiA)[1];
      } else {
        APiA = REAL(sOnPiA)[0];
      }
    
      On2 += SC * ( log( APiA ) - log(1.0 - APiA ) );
    }
  }
  if (RPiAPrior != NULL && !Rf_isNull(PiAPrior) && 
    Rf_isReal(PiAPrior) && REAL(PiAPrior)[0] > 0) {
    if (Rf_length(sOnPiA) >= 1) {
      if (rPriorProbFixed != NULL && !Rf_isNull(rPriorProbFixed->asSexp())) {
      } else {
        if (Verbose > 5) {
          Rprintf("ProbPosterior: Using OnPiA Prior because of null PriorProbFixed\n");
        }
        On += (REAL(PiAPrior)[0]-1) * log(REAL(sOnPiA)[0] ) +
              (REAL(PiAPrior)[1]-1) * log( 1.0 - REAL(sOnPiA)[0]);
      }  
    } else if (Rf_length(sOnPiA)  >= 2) {
      if (rPriorProbTau != NULL && !Rf_isNull(rPriorProbTau->asSexp())) {
      } else {
        if (Verbose > 5) {
          Rprintf("ProbPosterior: Using OnPiA Prior because of null PriorProbTau\n");
        }
        if (RPiAPrior != NULL && !Rf_isNull(PiAPrior) && Rf_isReal(PiAPrior)) {
        if (Rf_length(PiAPrior) == 4  && REAL(PiAPrior)[2] > 0) {
          On += (REAL(PiAPrior)[2]-1) * log(REAL(sOnPiA)[1]) +
            (REAL(PiAPrior)[3]-1) * log(1.0 - REAL(sOnPiA)[1]);
        } else if (!Rf_isNull(PiAPrior) && Rf_length(PiAPrior) >= 2 &&
          REAL(PiAPrior)[0] > 0) {
          On += (REAL(PiAPrior)[0]-1) * log(REAL(sOnPiA)[1]) +
            (REAL(PiAPrior)[1]-1) * log(1.0 - REAL(sOnPiA)[1]);        
        }
        }
      }
    }
  }
     
  if(Verbose > 5) {
    Rprintf("Off to look at final On effects! \n");
  }
  double isqSigma = 1/ sqrt(REAL(sOnSigma)[0]);
  if (dfRobit > 0) {
    for (ii = 0; ii < n; ii++) {
      if (intY[ii] == 1) {
        On += Rf_pt(REAL(sY)[ii]*isqSigma, dfRobit, 1, 1);
      } else {
        On += Rf_pt(REAL(sY)[ii]*isqSigma, dfRobit, 0, 1);
      }
    }
  } else if (dfRobit == 0.0) {
    if (iiWeight != NULL && AlterWeightdfRobit > 0.0) {
    double MyCons;
    for (ii = 0; ii < n; ii++) {
      MyCons = sqrt( iiWeight[ii] / REAL(sOnSigma)[0]);
      if (intY[ii] == 1) {
        On += Rf_pnorm5(REAL(sY)[ii]*MyCons, 0.0, 1.0, 1,1);
      } else {
        On += Rf_pnorm5(REAL(sY)[ii]*MyCons, 0.0, 1.0, 0,1);
      }
    }    
    } else {
    for (ii = 0; ii < n; ii++) {
      if (intY[ii] == 1) {
        On += Rf_pnorm5(REAL(sY)[ii]*isqSigma, 0.0, 1.0, 1,1);
      } else {
        On += Rf_pnorm5(REAL(sY)[ii]*isqSigma, 0.0, 1.0, 0,1);
      }
    }
    }
  } else if (dfTNoise > 0) {
    On += -.5 * YResidSq/ REAL(sOnSigma)[0]   - n * .5  * log(REAL(sOnSigma)[0]);
    if (RSigmaPrior != NULL && !Rf_isNull(SigmaPrior)  && Rf_isReal(SigmaPrior) &&
      Rf_length(SigmaPrior) >= 2) {
       On += -.5 *REAL(SigmaPrior)[1] * REAL(SigmaPrior)[0]   / REAL(sOnSigma)[0]
         - .5 * ( log(REAL(sOnSigma)[0]) * ( REAL(SigmaPrior)[0] + 2.0));
    }  
    int One = 1;
    On -= .5 * F77_CALL(dasum)(&n, iiWeight, &One);
    for (ii = 0; ii < n; ii++) {
      On+=  .5 * (log(iiWeight[ii]) * (dfTNoise + 2.0));
    }
  } else {
    On += -.5 * YResidSq/ REAL(sOnSigma)[0] - n * .5  * log(REAL(sOnSigma)[0]);
    if (RSigmaPrior != NULL && !Rf_isNull(SigmaPrior)  && 
      Rf_isReal(SigmaPrior) && Rf_length(SigmaPrior) >= 2) {
       On += -.5 *REAL(SigmaPrior)[1] * REAL(SigmaPrior)[0]  / REAL(sOnSigma)[0]
         - .5 * ( log(REAL(sOnSigma)[0]) * ( REAL(SigmaPrior)[0] + 2.0));
    }
  }

  if (R_isnancpp(On1) || R_isnancpp(On2) || R_isnancpp(On)) {
    if (R_isnancpp(On1)) {
      Rprintf("*************************************************************\n");
      Rprintf("ProbPosterior: Error, On1 is NAN! \n"); R_FlushConsole();
      Rprintf("Let's look into this:   ");
      if (rPriorProbFixed == NULL ||
        Rf_isNull(rPriorProbFixed->asSexp()) ||
        rPriorProbFixed->asSexp() == NULL) {
        Rprintf("Uh Oh rPriorProbFixed is NULL!\n"); R_FlushConsole();
      } else if (!Rf_isReal(rPriorProbFixed->asSexp())) {
        Rprintf("Uh Oh rPriorProbFixed is not real!\n"); R_FlushConsole();
      } else {
        SEXP sPPF = rPriorProbFixed->asSexp();
        Rprintf("rPriorProbFixed is length $d \n", Rf_length(sPPF)); R_FlushConsole();
        int ATK = 0;
        Rprintf("rPriorProbFixed = c( ");
        for (int jjt = 0; jjt < Rf_length(sPPF); jjt++) {
           if (jjt >= Rf_length(sPPF)-1) {
             Rprintf(" %4f);\n", REAL(sPPF)[jjt]);   R_FlushConsole();
             R_FlushConsole();
           } else if (ATK >= 8) {
             ATK = 0;
             Rprintf("\n                  %.4f,",  REAL(sPPF)[jjt]);
             R_FlushConsole();
           } else {
             Rprintf(" %.4f,", REAL(sPPF)[jjt]);  R_FlushConsole();
           }
        }
      }
        if (sOnPiA == NULL || Rf_isNull(sOnPiA)) {
          Rprintf("sOnPiA not set!\n"); R_FlushConsole();
        } else if (!Rf_isReal(sOnPiA)) {
          Rprintf("sOnPiA not real!\n"); R_FlushConsole();
        } else {
           Rprintf("REAL(sOnPiA)[0] = %.5f\n", REAL(sOnPiA)[0]); R_FlushConsole();
        }
        if (RPiAPrior == NULL || Rf_isNull(PiAPrior)) {
          Rprintf("PiAPrior is null, which is okay.\n"); R_FlushConsole();
        } else {
          Rprintf("PiAPrior = %.4f, %.4f\n", REAL(PiAPrior)[0], REAL(PiAPrior)[1]);
          R_FlushConsole();
        }
        Rprintf("SC = %d \n", SC); R_FlushConsole();
        Rprintf("TypePrior = %d \n", TypePrior); R_FlushConsole();
        if (RtauFixed == NULL || tauFixed==NULL || Rf_isNull(tauFixed) || Rf_length(tauFixed) <= 0) {
          Rprintf("tauFixed is NULL!\n"); R_FlushConsole();
        } else if (!Rf_isReal(tauFixed)) { Rprintf("tauFixed is not REAL!\n"); R_FlushConsole();
        } else {
          Rprintf("tauFixed is length %d \n", Rf_length(tauFixed));
          int LTF = 0;
          Rprintf("tauFixed = c( ");
          for (int jjt = 0; jjt < Rf_length(tauFixed); jjt++) {
             if (jjt >= Rf_length(tauFixed)-1) {
               Rprintf(" %4f);\n", REAL(tauFixed)[jjt]);   R_FlushConsole();
               LTF++;
               R_FlushConsole();
             } else if (LTF >= 8) {
               LTF = 0;
               Rprintf("\n                  %.4f,",  REAL(tauFixed)[jjt]);
               R_FlushConsole();
             } else {
               Rprintf(" %.4f,", REAL(tauFixed)[jjt]);  R_FlushConsole();
             }
          }
        }
        if (sOnTau == NULL || Rf_isNull(sOnTau)) {
          Rprintf("sOnTau is NULL, I guess okay? \n "); R_FlushConsole();
        } else if (Rf_isReal(sOnTau)) {
          Rprintf("sOnTau is length %d \n", Rf_length(sOnTau)); R_FlushConsole();
          int LsT = 0;
          Rprintf("sOnTau = c( ");
          for (int jjt = 0; jjt < Rf_length(sOnTau); jjt++) {
             if (jjt >= Rf_length(sOnTau)-1) {
               Rprintf(" %4f);\n", REAL(sOnTau)[jjt]);   R_FlushConsole();
               R_FlushConsole();    LsT++;
             } else if (LsT >= 8) {
               LsT = 0;
               Rprintf("\n                  %.4f,",  REAL(sOnTau)[jjt]);
               R_FlushConsole();
             } else {
               Rprintf(" %.4f,", REAL(sOnTau)[jjt]);  R_FlushConsole();
             }
          }
        }
        if (sBeta == NULL || Rf_isNull(sBeta)) {
          Rprintf("sBeta is NULL not going forward!\n"); R_FlushConsole();
          Rf_error("F Me, sBeta is NULL!\n");
        } else if (!Rf_isReal(sBeta)) {
          Rf_error("sBeta isn't even real, not good!\n"); R_FlushConsole();
        } else {
          Rprintf("sBeta is length %d \n", Rf_length(sBeta));
          int LsB = 0;
          Rprintf("sBeta = c( ");
          for (int jjt = 0; jjt < Rf_length(sBeta); jjt++) {
             if (jjt >= Rf_length(sBeta)-1) {
               Rprintf(" %4f);\n", REAL(sBeta)[jjt]);   R_FlushConsole();
               R_FlushConsole();    LsB++;
             } else if (LsB >= 8) {
               LsB = 0;
               Rprintf("\n                  %.4f,",  REAL(sBeta)[jjt]);
               R_FlushConsole();
             } else {
               Rprintf(" %.4f,", REAL(sBeta)[jjt]);  R_FlushConsole();
             }
          }
        }

    }

    if (R_isnancpp(On2)) {
      Rprintf("PropPosterior: Error, On2 is NAN! \n"); R_FlushConsole();
    }
    if (R_isnancpp(On)) {
      Rprintf("PropPosterior: Error, On is NAN! \n"); R_FlushConsole();
    }
    Rf_error("ProbPosterior Error:  On1 = %f, On2 = %f, On = %f \n", On1, On2, On);
  }
  CurrentProb = On + On1+On2;
  return(On+ On1 + On2);
}

// Get_RegionMIP  We get RegionMIP from ProbVector Only. 
SEXP BayesSpikeCL::get_RegionMIP() {
  if (RMIP == NULL || Rf_isNull(RMIP->asSexp()) ||
      Rf_length(RMIP->asSexp()) <= 0) {
      int PTECTED = 0;
      if (RTBSR5 == NULL) {
        Rprintf("BayesSpikeCL:get_PostProbCodaList, where is TBSR5?\n");
        R_FlushConsole();
        return(R_NilValue);
      }
      SEXP MIPFunction;
      Rf_protect(MIPFunction = 
         Rf_findVarInFrame( get_BayesSpikeNameSpace(), 
         Rf_install("ATryFillMIP")));
      PTECTED++;
      if (Rf_isNull(MIPFunction)) {
        Rprintf("BayesSpikeCL: Error: no ATryFillMIP.\n");
        Rf_unprotect(PTECTED); PTECTED = 0;
        return(R_NilValue);
      }
      SEXP call;
      Rf_protect(call = Rf_lcons( MIPFunction, 
        Rf_cons(RTBSR5->asSexp(), R_NilValue))); 
      PTECTED++;
      if (Rf_isNull(call)) {
        Rprintf("BayesSpikeCL: MIPFunction, Error, could not genreate call.\n");
        Rf_unprotect(PTECTED); return(R_NilValue);
      }
      SEXP RTM = R_NilValue;
      if (Verbose > 1) {
        Rprintf("get_MIP: Starting to call ATryFillMIP.\n "); R_FlushConsole();
      }
      Rf_protect(RTM = Rf_eval(call, get_BayesSpikeNameSpace())); PTECTED++;
      if (RMIP == NULL) {
        Rprintf("BayesSpikeCL, get_MIP,after all that, still Null List!\n");
        if (!Rf_isNull(RTM)) {
          RMIP = new AObject(RTM);
          Rf_unprotect(PTECTED); PTECTED = 0;
          return(RTM);
        } else {
          Rf_unprotect(PTECTED); PTECTED = 0;
          return(RTM);
        }
      } else {
         Rf_unprotect(PTECTED); PTECTED = 0;
         if (RMIP == NULL) { return(R_NilValue); }
         return(RMIP->asSexp());
      }
      return(R_NilValue);
    }
    if (RMIP == NULL) { return(R_NilValue); }
    return(RMIP->asSexp());  
  }

////////////////////////////////////////////////////////////////////////////////
// CheckkXFinder(): 
//
//  This code writes to and attempts to find numerical inconsistencies or 
// memory leaks inside the kXFinder.
//  
int BayesSpikeCL::CheckkXFinderError(char* aS) {
  if (Verbose >= 0) {
    Rprintf("--- BayesSpikeCpp.cpp:::CheckkXFinderError(tt=%d/%d: OnKappaS=%d/%d): let's check performance after %s \n",
      tt, MaxGibbsIters, OnKappaS, this->p, aS);  R_FlushConsole();
  }
  if (OnKappaS <= 1) { return(1); }
  int LookFor = 0; int ii = 0;
  for (ii = 0; ii < OnKappaS-1; ii++) {
    LookFor = kXFinder[ii];
    for (int jj = ii+1; jj < OnKappaS; jj++) {
      if (kXFinder[jj] == LookFor) {
        Rprintf("CHECKkXFinderERROR: Oh no, kXFinder[%d] = kXFinder[%d] = %f!\n",
          ii, jj, kXFinder[ii]);
        Rprintf("### This is a bad error. During %s.\n", aS);
        Rf_error("ERROR CheckkXFinder after %s!\n", aS);
      }
    }
  }
  if (Verbose >= 2) {
    Rprintf("--- BayesSpikeCpp.cpp:::CheckkXFinderError(tt=%d/%d): Very least, kXFinder is non duplicated. \n", tt, MaxGibbsIters);
    R_FlushConsole();
  }
  int nErrors = 0;
  for (ii = 0; ii < this->p; ii++) {
    if (XLC[ii] >= 0 && kXFinder[XLC[ii]] != ii) {
      Rprintf("ERROR CheckkXFinderError(tt=%d/%d: OnKappaS=%d/%d): XLC[ii=%d]=%d, kXFinder[%d]=%d!\n",
        tt, MaxGibbsIters, OnKappaS, this->p, ii, XLC[ii], XLC[ii], kXFinder[XLC[ii]]); R_FlushConsole();
      nErrors++;
    }
    if (nErrors > 20) {
      Rprintf("CheckkXFinderError[tt=%d/%d] aS=%s, ouch, too many errors on XLC kXFinder issue!\n",
        tt, MaxGibbsIters, aS);
      break;
    }
  }
  double sLast; double sMe;     int jj = 0;   int LN, MN;
  if (Verbose >= 2) {
    Rprintf("--- CheckkXFinderError(tt=%d/%d): We are about to check pXtX. \n", tt, MaxGibbsIters);
  }
  for (ii = 0; ii < OnKappaS; ii++) {
    if (pXtX[ii] == NULL) {
      Rprintf("ERROR: CheckkXFinderError(tt=%d/%d: OnKappaS=%d/%d, pXtX[ii=%d/%d] is NULL!\n");
      nErrors++; R_FlushConsole();
    } else {
      sLast = 0.0; sMe = 0.0;
      LN = n * (p-1);  MN = kXFinder[ii] * n;
      for (jj = 0; jj < n; jj++) {
        sLast += REAL(sX)[LN] * REAL(sX)[MN];
        sMe += REAL(sX)[MN] * REAL(sX)[MN];  LN++; MN++;
      }
      if (fabs(sMe - *(pXtX[ii] + kXFinder[ii])) >= .0001) {
        Rprintf("BayesSpikeCpp:CheckkXFinder(): ERROR (tt=%d/%d) (ii=%d/%d for coord %d/%d, sMe=%f but *(pXFinder[%d]+%d) = %f \n",
          tt, MaxGibbsIters, ii, OnKappaS, kXFinder[ii], p, sMe, ii, kXFinder[ii],
           *(pXtX[ii] + kXFinder[ii]));  nErrors++; R_FlushConsole();
      } else {
        *(pXtX[ii] + kXFinder[ii]) = sMe;
      }
      if (fabs(sLast - *(pXtX[ii] + p-1)) >= .0001) {
        Rprintf("Error CheckkXFinderError(tt=%d/%d) LAST (ii=%d/%d for coord %d/%d, sMe=%f but *(pXFinder[%d]+%d) = %f \n",
          tt, MaxGibbsIters, ii, OnKappaS, kXFinder[ii], p, sMe, ii, p-1,
           *(pXtX[ii] + p-1));  nErrors++; R_FlushConsole();
      }  else {
         *(pXtX[ii] + p-1) = sLast;
      }
      if (nErrors > 50) {
        Rprintf("Error CheckkXFinderError Too many errors break!\n"); R_FlushConsole();
        break;
      }
    }
  }
  if (nErrors > 0) {
    Rprintf("ERROR CheckkXFinder(tt=%d/%d) at end nErrors was %d, too many it is an error!\n",
      tt, MaxGibbsIters, nErrors); R_FlushConsole();
    Rf_error("CheckkXFinder: Sorry, function %s that ran me generated mistakes!\n", aS);
  }
  if (Verbose >= 2) {
    Rprintf("--- CheckkXFinder(tt=%d/%d), at End, we expect success after %s \n", tt,
      MaxGibbsIters, aS);
  }
  return(1);
}
  

////////////////////////////////////////////////////////////////////////////
// modBayesSpikeCL:: Rcpp functions for BayesSpike.cc
//
//  These functions, fields, and methods are attached and callable, to BayesSpikeCL object 
//  (usually termed "MBS" in the R Code).
//
extern "C" {
RCPP_MODULE(modBayesSpikeCL) {
  using namespace Rcpp;
  
  class_<BayesSpikeCL>( "BayesSpikeCL" )
  .constructor< SEXP, SEXP, SEXP, SEXP, SEXP, SEXP>()
  //.finalizer(&DDeleteStuff)
  .field_readonly("IamBayesSpike", &BayesSpikeCL::IamBayesSpike)
  .field_readonly("NInstance", &BayesSpikeCL::NInstance )
  .field_readonly("fd", &BayesSpikeCL::fd)
  .field_readonly("fd1", &BayesSpikeCL::fd1)
  .field_readonly("fd2", &BayesSpikeCL::fd2)  
  .field_readonly("RanRecordCoda", &BayesSpikeCL::RanRecordCoda)    
  .field_readonly("RanFillTau", &BayesSpikeCL::RanFillTau)  
  .field_readonly("WroteTotalTau", &BayesSpikeCL::WroteTotalTau) 
  .field_readonly("WroteAllTau", &BayesSpikeCL::WroteAllTau)   
  .field("HowSample", &BayesSpikeCL::HowSample)
  .field("NumIntegrations", &BayesSpikeCL::NumIntegrations)
  .field("FakeOuttau", &BayesSpikeCL::FakeOuttau)
  .field("IntegratedDensity", &BayesSpikeCL::IntegratedDensity)
  .field("MaxIters", &BayesSpikeCL::MaxIters)
  .field("CauchyEpsilon", &BayesSpikeCL::CauchyEpsilon)
  .field("MaxTau", &BayesSpikeCL::ThisMaxTau)
  .field("EEMergeEvery", &BayesSpikeCL::EEMergeEvery)
  .property("SampleTausOnly", &BayesSpikeCL::get_SampleTausOnly, &BayesSpikeCL::set_SampleTausOnly, "Do not Sample Beta during SampleFixedB step.")
  .field("DoSampleBFixedii", &BayesSpikeCL::DoSampleBFixedii)
  .field("burnin", &BayesSpikeCL::burnin)
  .field("KillPct", &BayesSpikeCL::KillPct)
  .field("MinimumProbKeeper", &BayesSpikeCL::MinimumProbKeeper)
  .field_readonly("OnKSqPBack", &BayesSpikeCL::OnKSqPBack)
  .field_readonly("NumActiveBack", &BayesSpikeCL::NumActiveBack)
  .field_readonly("WillAddNewTau", &BayesSpikeCL::WillAddNewTau)
  .field_readonly("AllNewCoords", &BayesSpikeCL::AllNewCoords)  
  .field_readonly("NewFixedCoords", &BayesSpikeCL::NewFixedCoords)
  .field_readonly("BackSigma", &BayesSpikeCL::BackSigma)

  .field_readonly("LenRandom", &BayesSpikeCL::LenRandom)
  .field_readonly("LenFixed", &BayesSpikeCL::LenFixed)
  .field_readonly("CountFreeFixed", &BayesSpikeCL::CountFreeFixed)
  .field_readonly("CountFreeRandom", &BayesSpikeCL::CountFreeRandom)  
  .field_readonly("OnShrinkFixed", &BayesSpikeCL::OnShrinkFixed)
  .field_readonly("OnShrinkRandom", &BayesSpikeCL::OnShrinkRandom)
  .property("AlterInitRun", &BayesSpikeCL::get_AlterInitRun, &BayesSpikeCL::set_AlterInitRun,
     "AlterInitRun")
  .method("BlankAllNewCoords", &BayesSpikeCL::BlankAllNewCoords)  
  .method("SetAllNewCoords", &BayesSpikeCL::SetAllNewCoords)
  .method("CountAllNewCoords", &BayesSpikeCL::CountAllNewCoords, "Count he Previously Unlisted New Coordinates to add to XtX")
  .property("MaximumAllocation", &BayesSpikeCL::get_MaximumAllocation,
    &BayesSpikeCL::set_MaximumAllocation, "Maximum Memory Value allowed")
  .field("StartRemoval", &BayesSpikeCL::StartRemoval)
  .field("CheckRemovable", &BayesSpikeCL::CheckRemovable)
  .field_readonly("ItersSinceRemoval", &BayesSpikeCL::ItersSinceRemoval)  
  .property("rICholBack", &BayesSpikeCL::get_rICholBack, "Backup Matrix of rIChol")
  .property("get_rICholPart2", &BayesSpikeCL::get_rICholPart2, "get rIChol at stage of Part2 departure")
  .field("DoMax", &BayesSpikeCL::DoMax) 
  .method("KillNanBeta", &BayesSpikeCL::KillNanBeta, "Remove Nans from Beta")
  .property("TemperatureDecreasingRate", &BayesSpikeCL::get_TemperatureDecreasingRate,
    &BayesSpikeCL::set_TemperatureDecreasingRate,
     "For higher temperature levles, how many fewer cells should be sampled?")
  .property("AlterWeightdfTNoise", &BayesSpikeCL::get_AlterWeightdfTNoise,
    &BayesSpikeCL::set_AlterWeightdfTNoise, "Alternative dfTNoise for weight function")
  .property("AlterdfTNoise", &BayesSpikeCL::get_AlterWeightdfTNoise,
    &BayesSpikeCL::set_AlterWeightdfTNoise, "Alternative dfTNoise for weight function")
  .property("AlterWeightdfRobit", &BayesSpikeCL::get_AlterWeightdfRobit,
    &BayesSpikeCL::set_AlterWeightdfRobit, "Alternative df Robit for weight function")
  .property("AlterdfRobit", &BayesSpikeCL::get_AlterWeightdfRobit,
    &BayesSpikeCL::set_AlterWeightdfRobit, "Alternative dfT Robit for weight function")   
  .property("AlterdfTRobit", &BayesSpikeCL::get_AlterWeightdfRobit,
    &BayesSpikeCL::set_AlterWeightdfRobit, "Alternative dfT Robit for weight function")    
  .property("CurrentMaxGibbsIters", &BayesSpikeCL::get_CurrentMaxGibbsIters,
    &BayesSpikeCL::set_CurrentMaxGibbsIters, 
    "Current number of Gibbs iters for current Temperature.") 
  .field("NumSpliceSampleReps", &BayesSpikeCL::NumSpliceSampleReps)
  .field("NumSteps", &BayesSpikeCL::NumSteps)
  .method("CheckpXtX", &BayesSpikeCL::CheckpXtX, "Check whether pXtX was improperly swapped.")
  .property("smXtY", &BayesSpikeCL::get_smXtY, "smXtY backup vector")
  .property("tt", &BayesSpikeCL::get_tt, &BayesSpikeCL::set_tt, "Iteration of Algorithm")
  .field("TypeFixedPrior", &BayesSpikeCL::TypePrior)
  .field("NumberNew", &BayesSpikeCL::NumberNew)
  .field("MaxTauEst", &BayesSpikeCL::MaxTauEst)
  .field("NewWrite", &BayesSpikeCL::NewWrite)
  .field("NewTWrite", &BayesSpikeCL::NewTWrite)
  .property("CodaTau", &BayesSpikeCL::get_CodaTau, "Coda Buffer of Tau Saves")
  .property("CodaTjj", &BayesSpikeCL::get_CodaTjj, "Coda Buffer of Active Tau locations") 
  .field_readonly("LengthWrittenTauCodaBuffer", &BayesSpikeCL::LengthWrittenTauCodaBuffer)
  .field_readonly("LengthTotalWrittenTauCodaBuffer", &BayesSpikeCL::LengthTotalWrittenTauCodaBuffer) 
  .field("MaximizeMeCauchyTotalIters", &BayesSpikeCL::MaximizeMeCauchyTotalIters)
  .property("MaxInvert", &BayesSpikeCL::get_MaxInvert, &BayesSpikeCL::set_MaxInvert, "Maximum Matrix size to invert whole")
  .field("EarlyEndtt", &BayesSpikeCL::EarlyEndtt)
  .field("TotalFails", &BayesSpikeCL::TotalFails)
  .field("NumFails", &BayesSpikeCL::NumFails)  
  .field("MaxAlsErrors", &BayesSpikeCL::MaxAlsErrors)  
  .field("GoBackAfter", &BayesSpikeCL::GoBackAfter)  
  .field("EarlyEndStep", &BayesSpikeCL::EarlyEndStep)  
  .field_readonly("NukeBetaTau", &BayesSpikeCL::NukeBetaTau)  
  .field_readonly("NukeBetaNotTau", &BayesSpikeCL::NukeBetaNotTau)  
  .method("FillRunProbVector", &BayesSpikeCL::FillRunProbVector,
    "A Method to try to fill ProbVector")
  .method("TestOrderedActive", &BayesSpikeCL::TestOrderedActive,
    "Test Ordered Active")
  .method("CheckkXToXLC", &BayesSpikeCL::DoCheckkXToXLC, "CheckkXToXLC")
  .property("StartRunProbVector", &BayesSpikeCL::get_StartRunProbVector, 
    "First Iter to start recording ProbVector")
  .property("TotEveryProbVector", &BayesSpikeCL::GetTotEveryProbVector,
    "How much samples are put into every ProbVector positions.")
  .property("DoShortMIP", &BayesSpikeCL::get_DoShortMIP, &BayesSpikeCL::set_DoShortMIP,
    "Do use RunProbVector for MIP (If p is large)")
  .property("DoShort", &BayesSpikeCL::get_DoShortMIP, &BayesSpikeCL::set_DoShortMIP,
    "Do use RunProbVector for MIP (If p is large)")
  .property("RunProbVector", &BayesSpikeCL::GetRunProbVector,
    "Running Totals in ProbVector")
  .property("TotInProbVector", &BayesSpikeCL::get_TotInProbVector,
    "How many Values are put in ProbVector")
  .field("PrintIter", &BayesSpikeCL::PrintIter)
  .field("ReorderIter", &BayesSpikeCL::ReorderIter) 
  .field("NumlAZero", &BayesSpikeCL::NumlAZero) 
  .field("TotalLAZero", &BayesSpikeCL::TotalLAZero) 
  .field("GoBackAfterlA", &BayesSpikeCL::GoBackAfterlA) 
  .field("MaxlAErrors", &BayesSpikeCL::MaxlAErrors)   
  .field("StartIter", &BayesSpikeCL::StartIter)
  .field("EndIter", &BayesSpikeCL::EndIter)
  .field("ZeroOutBeforeSample", &BayesSpikeCL::ZeroOutBeforeSample)
  .field("RevertTemperatureEvery", &BayesSpikeCL::RevertTemperatureEvery)
  .property("MIP", &BayesSpikeCL::get_MIP, 
    &BayesSpikeCL::set_MIP, "Average Model Inclusion Probabilities")
  .property("RegionMIP", &BayesSpikeCL::get_RegionMIP, "Region Model Inclusion Probabilities")
  .property("ProbCodaList", &BayesSpikeCL::get_PostProbCodaList, 
    &BayesSpikeCL::set_PostProbCodaList, "A Posterior Request for Probability chain")
  .property("PostProbCoda", &BayesSpikeCL::get_PostProbCodaList, 
    &BayesSpikeCL::set_PostProbCodaList, "A Posterior Request for Probability chain")
  .property("PostProbCodaList", &BayesSpikeCL::get_PostProbCodaList, 
    &BayesSpikeCL::set_PostProbCodaList, "A Posterior Request for Probability chain")
  .property("PostProbBufferFile", &BayesSpikeCL::get_sPostProbBufferFile, 
    "Get PostProbBuffer File")
  .property("sPostProbBufferFile", &BayesSpikeCL::get_sPostProbBufferFile, 
    "Get PostProbBuffer File")
  .property("RsPostProbBufferFile", &BayesSpikeCL::get_sPostProbBufferFile, 
    "Get PostProbBuffer File")
  .property("TooMuchBuffer", &BayesSpikeCL::get_TooMuchBuffer)
  .property("InitiateTime", &BayesSpikeCL::get_InitiateTime, 
    &BayesSpikeCL::set_InitiateTime, "Initiating time")
  .property("SecondInitiateTime", &BayesSpikeCL::get_SecondInitiateTime, 
    &BayesSpikeCL::set_SecondInitiateTime, "SecondInitiateTime")    
  .property("CompleteTime", &BayesSpikeCL::get_CompleteTime, 
    &BayesSpikeCL::set_CompleteTime, "CompleteTime")    
  .property("PostBufferFile", &BayesSpikeCL::get_sPostProbBufferFile, 
    "Get PostProbBuffer File")
  .property("ProbBufferFile", &BayesSpikeCL::get_sPostProbBufferFile, 
    "Get PostProbBuffer File")
  .method("SetupProbFixed", &BayesSpikeCL::SetupProbFixed, "Setup ProbFixed")
  .method("SetupProbTau", &BayesSpikeCL::SetupProbTau, "Setup Prob Tau")
  .method("AssignTau", &BayesSpikeCL::AssignTau, "Setup Prob Tau")
  .method("SetupProbBufferFile", &BayesSpikeCL::set_RsProbBufferFile, "Setup ProbBuffer to contain ProbFixed, ProbTau")
  .property("CurrentPostProbCodaBuffer", 
    &BayesSpikeCL::get_CurrentPostProbCodaBuffer, "Current Filling of PostProbCodaBuffer")
  .method("WritePostProbBuffer", &BayesSpikeCL::WritePostProbBuffer, "Write To PostProbBuffer")
  .property("AllPostProbCodaBuffer", &BayesSpikeCL::get_PostProbCodaBuffer, "Entirerty of Posterior Prior Buffer")
  .property("CurrentPostProbCodaBuffer", &BayesSpikeCL::get_CurrentPostProbCodaBuffer, "Entirerty of Posterior Prior Buffer")
  .property("PostProbCodaBuffer", &BayesSpikeCL::get_PostProbCodaBuffer, "Entirerty of Posterior Prior Buffer")
  .property("LengthWrittenPostProb", &BayesSpikeCL::get_LengthWrittenPostProb, "Current Filling of PostProbCodaBuffer")
  .property("LengthTotalWrittenPostProb", &BayesSpikeCL::get_LengthTotalWrittenPostProb, "Current Filling of PostProbCodaBuffer into file")
  .property("NewPostProbWrite", &BayesSpikeCL::get_NewPostProbWrite, &BayesSpikeCL::set_NewPostProbWrite,
    "NewPostProbWrite: whether to make file for post write.")
  .property("LengthPostProb", &BayesSpikeCL::get_LengthPostProb, "Number of Posterior Probability elements in ProbFixed+ProbTau")
  .property("LengthPostProbCodaBuffer", &BayesSpikeCL::get_LengthPostProbCodaBuffer, "Current Filling of PostProbCodaBuffer")
  .property("MaxLengthOldIDFiles", &BayesSpikeCL::get_MaxLengthOldIDFiles, "Max Length old I and D Files")
  .method("SetCodaOldLocAndProbFile", &BayesSpikeCL::set_sCodaOldLocAndProbFile,
    "SetCodaOldILoc and Prob Files for Temperature Sampling, supply file names, start iter, last iter, and id of chain")
  .property("LengthTempDraws", &BayesSpikeCL::get_LengthTempDraws, "Length Draws in chains per Temperature")
  .property("CurrentProb", &BayesSpikeCL::get_CurrentProb, "CurrentProbability")
  .property("iiOnILocProb", &BayesSpikeCL::get_iiOnILocProb, &BayesSpikeCL::set_iiOnILocProb, "Set  iiOnILocProb" )
  .property("LengthOldCoda", &BayesSpikeCL::get_LengthOldCoda, "Length of Old Coda")
  .property("LengthProbCoda", &BayesSpikeCL::get_LengthProbCoda, "Length of ProbCoda")
  .property("IOldCoda", &BayesSpikeCL::get_IOldCoda)
  .method("GoFindInOld", &BayesSpikeCL::GoFindInOld, "Find an Energy in BayesSpikeCL::ProbOldCoda")
  .method("SetupIntY", &BayesSpikeCL::SetupIntY, "Setup Int Y with dfRobit for Robit regression")
  .property("dfRobit", &BayesSpikeCL::get_dfRobit, 
     &BayesSpikeCL::set_dfRobit, "Get df for Robit regression")
  .property("intY", &BayesSpikeCL::get_intY, "Integer Y for Robit Regression (Y itself is imputed robit draws) ")
  .property("sCodaiTFile", &BayesSpikeCL::get_sCodaiTFile, "sCodaiTFile") 
  .property("sCodadTFile", &BayesSpikeCL::get_sCodadTFile, "sCodadTFile")
  .property("sCodaiTLocFile", &BayesSpikeCL::get_RsCodaiTLocFile, "Location of Tau saved paramters")
  .property("CodaiTLocFile", &BayesSpikeCL::get_RsCodaiTLocFile, "Location of Tau saved paramters")
  .property("sCodaIFile", &BayesSpikeCL::get_sCodaIFile, "sCodaIFile")
  .property("sCodaDFile", &BayesSpikeCL::get_sCodaJFile, "sCodaDFile")
  .property("CodaJFile", &BayesSpikeCL::get_sCodaJFile, "sCodaDFile")
  .property("CodaiTFile", &BayesSpikeCL::get_sCodaiTFile, "sCodaiTFile") 
  .property("CodadTFile", &BayesSpikeCL::get_sCodadTFile, "sCodadTFile")
  .property("CodaIFile", &BayesSpikeCL::get_sCodaIFile, "sCodaIFile")
  .property("CodaDFile", &BayesSpikeCL::get_sCodaJFile, "sCodaDFile")
  .property("CodaJFile", &BayesSpikeCL::get_sCodaJFile, "sCodaDFile")
  .property("ProbCoda", &BayesSpikeCL::get_ProbCoda)
  .property("ICoda", &BayesSpikeCL::get_ICoda)
  .property("SaveDir", &BayesSpikeCL::get_SaveDir, &BayesSpikeCL::set_SaveDir, 
    "Saved Directory Name")
  .property("ProbOldCoda", &BayesSpikeCL::get_ProbOldCoda)
  .property("sCodaOldIFile", &BayesSpikeCL::get_sCodaOldIFile, &BayesSpikeCL::set_sCodaOldIFile,
    "Old File -I- for Coda chains")
  .property("sCodaOldJFile", &BayesSpikeCL::get_sCodaOldJFile, &BayesSpikeCL::set_sCodaOldJFile,
    "Old File -J- for Coda chains")
  .property("CodaOldIFile", &BayesSpikeCL::get_sCodaOldIFile, &BayesSpikeCL::set_sCodaOldIFile,
    "Old File -I- for Coda chains")
  .property("CodaOldJFile", &BayesSpikeCL::get_sCodaOldJFile, &BayesSpikeCL::set_sCodaOldJFile,
    "Old File -J- for Coda chains")
  .property("sCodaOldDFile", &BayesSpikeCL::get_sCodaOldJFile, &BayesSpikeCL::set_sCodaOldJFile,
    "Old File -J- for Coda chains")
  .field("RobitNotReplaceFlag", &BayesSpikeCL::RobitNotReplaceFlag, "Flag to not Replace Robit during RobitReplace")
  .property("OnTauii",  &BayesSpikeCL::get_OnTauIndex,
    "Index of current setup OnTau")
  .property("lB", &BayesSpikeCL::get_lB, "Log of B Probability")
  .property("OnTauIndex",  &BayesSpikeCL::get_OnTauIndex,
    "Index of current setup OnTau")
  .property("ontauii",  &BayesSpikeCL::get_OnTauIndex,
    "Index of current setup OnTau")
  .property("TauIndex",  &BayesSpikeCL::get_OnTauIndex,
    "Index of current setup OnTau")
  .property("Tauii",  &BayesSpikeCL::get_OnTauIndex,
    "Index of current setup OnTau")
  .property("NewWriteTLoc",  &BayesSpikeCL::get_NewWriteTLoc, "Whether To Write New File for Tau Loc")
  .property("NewWriteT",  &BayesSpikeCL::get_NewWriteT, &BayesSpikeCL::set_NewWriteT,
    "Whether To Write New File For Tau")
  .property("OnCodaT",  &BayesSpikeCL::get_OnCodaT, "Location in OnCodaTau File writing to")
  .method("WriteTau", &BayesSpikeCL::WriteTau, "WriteTau Buffer to File now")
  .property("LengthCodaTjj",  &BayesSpikeCL::get_LengthCodaTjj, "Length CodaTau Buffer")
  .property("LengthCodaTLoc",  &BayesSpikeCL::get_LengthCodaTLoc, "Length CodaTLoc Buffer")
  .property("TotOnCodaT",  &BayesSpikeCL::get_TotCodaT, "Location in OnCodaTau File writing to")
  .property("TotCodaTLocjj",  &BayesSpikeCL::get_TotCodaTLocjj, "Location in OnCodaTau File writing to")
  .property("TotOnCodaTLocjj", &BayesSpikeCL::get_TotCodaTLocjj, "Location in OnCodaTau File writing to")
  .property("OnCodaTLocjj",  &BayesSpikeCL::get_OnCodaTLocjj, "Location in OnCodaTau File writing to")
  .method("GiveItAChanceAndKillUselessCoeficients", &BayesSpikeCL::GiveItAChanceAndKillUselessCoeficients, 
    "GiveItAChanceAndKillUselessCoeficients kill")
  .method("RunAlgorithm", &BayesSpikeCL::RunAlgorithm, "Run The BayesSpike Regression Algorithm (BayesSpikeGibbs.cpp)")
  .method("EESamplerMerge", &BayesSpikeCL::EESamplerMerge, "Merge EE Sampler")
  .method("TestFindInMe", &BayesSpikeCL::TestFindInMe, "Test FindInMe Program, what do you want to find in OldProbCoda?")
  .method("LoadProbOldCodaICoda", &BayesSpikeCL::LoadProbOldCodaICoda,
    "Loads ProbOldCoda and IOldCoda from respective files, input is start and end")
  .property("gPrior", &BayesSpikeCL::get_gPrior, &BayesSpikeCL::set_gPrior, "Zellner g Prior weights XtX in exponential density")
  .property("ProbVector", &BayesSpikeCL::GetProbVector, "Posterior Probability of Inclusion")
  .method("SetupProbCoda", &BayesSpikeCL::SetupProbCoda, "Prob Coda, Buffer vector of current Probabilities, supply int")
  .method("SetupRunProbVector", &BayesSpikeCL::SetupRunProbVector, "Setup Probability of Inclusion Vector")
  .property("RunProbRegionVector", &BayesSpikeCL::get_RunProbRegionVector, "Running ProbRegionVector if RegionWidth > 1")
  .method("SetRegionWidthAndRunProbRegionVector", &BayesSpikeCL::SetRegionWidthAndRunProbRegionVector,
    "Set up and Fill RunProbRegionVector from Saved Information ")
  .method("SetupRunLessZeroMoreZeroSumBeta", &BayesSpikeCL::SetupRunLessZeroMoreZeroSumBeta,
    "Set up the Run Less Zero and More Zero ")
  .property("RunMoreZero", &BayesSpikeCL::get_RunMoreZero, "Total Draws more than zero for each beta")
  .property("RunLessZero", &BayesSpikeCL::get_RunLessZero, "Total Draws less than zero for each beta")
  .property("RunSumBeta", &BayesSpikeCL::get_RunSumBeta, "Running Total of Beta Draws")
  .property("QMedianBeta", &BayesSpikeCL::get_QMedianBeta, "Quick Calculated Posterior Median (Means for non zero)")
  .property("RegionWidth", &BayesSpikeCL::get_RegionWidth, &BayesSpikeCL::set_RegionWidth, "Region Width for Windowed RMIP")
  .method("ShowRegionNeed", &BayesSpikeCL::ShowRegionNeed, "Show a RegionNeed(point, width, total length)")
  .method("RefreshOrderedActive", &BayesSpikeCL::RefreshOrderedActive, 
    "Refresh Ordered Active List")
  .method("SetXtResidNoXtX", &BayesSpikeCL::SetXtResidNoXtX, "Re setup XtResid")
  .field_readonly("nWeight", &BayesSpikeCL::nWeight)
  .field_readonly("BeingDestroyed", &BayesSpikeCL::BeingDestroyed)
  .property("NoShrinkFixed", &BayesSpikeCL::get_NoShrinkFixed, 
   &BayesSpikeCL::set_NoShrinkFixed, "Any Fixed variables which are fixed into the model?")
  .property("CNoShrinkFixed", &BayesSpikeCL::get_CNoShrinkFixed,
    "No Shrink Fixed")
  .property("LengthNoShrinkFixed", &BayesSpikeCL::get_LengthNoShrinkFixed, 
     "How many Fixed variables which are fixed into the model?")
  .property("NoShrinkRandom", &BayesSpikeCL::get_NoShrinkRandom, 
   &BayesSpikeCL::set_NoShrinkRandom, "Any Random groups which are fixed into the model?")
  .property("NoShrinkFixedPrior", &BayesSpikeCL::get_NoShrinkFixedPrior, 
   &BayesSpikeCL::set_NoShrinkFixedPrior, "Prior Spread for unsrunk fixed effects in model.")
  .property("NoShrinkRandomPrior", &BayesSpikeCL::get_NoShrinkRandomPrior, 
   &BayesSpikeCL::set_NoShrinkRandomPrior, "Prior Spread for unsrunk random effects groups in model.")
  .property("Onii", &BayesSpikeCL::get_Onii, "Onii, index of tau currently operating on")
  .property("CodaList", &BayesSpikeCL::get_CodaList, &BayesSpikeCL::set_CodaList,
    "CodaList:  List of all CodaTables")    
  .property("OtherNameCodaList", &BayesSpikeCL::get_OtherNameCodaList,
    &BayesSpikeCL::set_OtherNameCodaList, "Rename Coda Table names")
  .property("OtherCodaNames", &BayesSpikeCL::get_OtherNameCodaList,
    &BayesSpikeCL::set_OtherNameCodaList, "Rename Coda Table names")
  .property("OldCodaNames", &BayesSpikeCL::get_OldCodaNames,
    &BayesSpikeCL::set_OldCodaNames, "Old Coda Table names")
  .property("OtherDeCenterNamesCodaList", &BayesSpikeCL::get_OtherNameDeCenterCodaList,
    &BayesSpikeCL::set_OtherNameDeCenterCodaList, "Old Coda Table names")
  .property("OldDeCenterCodaNames", &BayesSpikeCL::get_OldDeCenterCodaNames,
    &BayesSpikeCL::set_OldDeCenterCodaNames, "DeCenter CodaNames")
  .property("OtherDeCenterCodaNames", &BayesSpikeCL::get_OtherNameDeCenterCodaList,
    &BayesSpikeCL::set_OtherNameDeCenterCodaList, "Old Coda Table names")
  .property("AllTempCodaLists", &BayesSpikeCL::get_AllTempCodaLists, &BayesSpikeCL::set_AllTempCodaLists,
    "CodaList: for All Temperatures")
  .property("AllCodaLists", &BayesSpikeCL::get_AllTempCodaLists, &BayesSpikeCL::set_AllTempCodaLists,
    "CodaList: for All Temperatures")
  .property("AllCodaList", &BayesSpikeCL::get_AllTempCodaLists, &BayesSpikeCL::set_AllTempCodaLists,
    "CodaList: for All Temperatures")
  .property("TotalOpenedFiles", &BayesSpikeCL::get_TotalOpenedFiles, "Total Files we've opened")
  .property("TotalClosedFiles", &BayesSpikeCL::get_TotalClosedFiles, "Total Files we've closed")
  .property("TotalR5OpenedFiles", &BayesSpikeCL::get_TotalR5OpenedFiles, 
    &BayesSpikeCL::set_TotalR5OpenedFiles, "Total Files we've opened in R5")
  .property("TotalR5ClosedFiles", &BayesSpikeCL::get_TotalR5ClosedFiles, 
    &BayesSpikeCL::set_TotalR5ClosedFiles, "Total Files we've closed in R5")
  .property("SolveAlpha", &BayesSpikeCL::SolveAlpha, "Derive an A degrees of freedom that might be good")
  .property("CodaChains", &BayesSpikeCL::get_CodaList, &BayesSpikeCL::set_CodaList,
    "CodaList:  List of all CodaTables") 
  .property("BayesSpikeNameSpace", &BayesSpikeCL::get_BayesSpikeNameSpace,
    &BayesSpikeCL::set_BayesSpikeNameSpace, "Namespace in R for BayesSpike objects")
  .property("XtXRecorded", &BayesSpikeCL::get_XtXRecorded, "XtXRecorded ")
  .property("CodaChain", &BayesSpikeCL::get_CodaTable, &BayesSpikeCL::set_CodaTable,
    "CodaList:  Get current CodaTable") 
  .property("TBSR5", &BayesSpikeCL::get_TBSR5, &BayesSpikeCL::set_TBSR5,
    "TBSR5:  R5 object for this sampler")
  .property("TBSRoo", &BayesSpikeCL::get_TBSRoo, &BayesSpikeCL::set_TBSRoo,
    "TBSRoo:  R.oo object for this sampler")
  .field_readonly("TypePrior", &BayesSpikeCL::TypePrior, "Type of Fixed Prior")
  .field_readonly("LastStF", &BayesSpikeCL::LastStF, "LastStF from last SamplePartBeta2")
  .field_readonly("LastGoFor", &BayesSpikeCL::LastGoFor, "LastGoFor from last SamplePartBeta2")   
  .property("SPrIChol", &BayesSpikeCL::get_SPrIChol, "Last rIChol from last SamplePartBeta2")
  .property("SPrI", &BayesSpikeCL::get_SPrI, "Last rI from last SamplePartBeta2")  
  .property("MaxLargeSizeSquares", &BayesSpikeCL::get_MaxLargeSizeSquares, "MaxLargeSizeSquares: maximum size of rI, smXtX, rIChol, signal for PartBeta2 regression")
  .method("CheckX", &BayesSpikeCL::CheckX, "Verify that X pointer has not moved")
  .method("CheckTBSRoo", &BayesSpikeCL::CheckTBSRoo, "Verify that TBSRoo pointer has not moved")
  .method("CheckTBSR5", &BayesSpikeCL::CheckTBSR5, "Verify that TBSR5 pointer has not moved")
  .method("CheckY", &BayesSpikeCL::CheckY, "Verify that Y pointer has not moved")
  .method("CheckBeta", &BayesSpikeCL::CheckBeta, "Verify that Beta pointer has not moved")
  .method("RenameCodaList", 
    &BayesSpikeCL::RenameCodaList, "Rename Coda List with New Dimnames")        
  .property( "tauEndList", &BayesSpikeCL::get_tauEndList, 
     &BayesSpikeCL::set_tauEndList, "tauEndList, Indice of Last Random Effects")
  .property("p", &BayesSpikeCL::get_p, "p: count of coefficients")
  .property("n", &BayesSpikeCL::get_n, "n: count of data")
  .property( "OnTau", &BayesSpikeCL::get_OnTau, 
     &BayesSpikeCL::set_OnTau, "OnTau, Current Spread of Random Effects")
  .property( "sOnTau", &BayesSpikeCL::get_OnTau, 
     &BayesSpikeCL::set_OnTau, "OnTau, Current Spread of Random Effects")
  .property( "tau", &BayesSpikeCL::get_OnTau, 
     &BayesSpikeCL::set_OnTau, "OnTau, Current Spread of Random Effects")
  .property( "NInstance", &BayesSpikeCL::get_NInstance, &BayesSpikeCL::set_NInstance, 
    "NInstance, Location of TBSRoo in ListBayesSpikeOb")
  .field_readonly("DoingEEProbSort", &BayesSpikeCL::DoingEEProbSort)
  .property("FileCodaChainIter", &BayesSpikeCL::get_FileCodaChainIter, "CodaChain Iter we are on")
  .property("OnChain", &BayesSpikeCL::get_FileCodaChainIter, "CodaChain Iter we are on")
  .property("OnChainIter", &BayesSpikeCL::get_FileCodaChainIter, "CodaChain Iter we are on")
  .property("EEProbSortWidth", &BayesSpikeCL::get_EEProbSortWidth, &BayesSpikeCL::set_EEProbSortWidth,
    "EEProbSortWidth:  Width of EE blocks if choosing in this manner")
  .property( "SubCodaList", &BayesSpikeCL::get_SubCodaList, 
    &BayesSpikeCL::set_SubCodaList, "CodaList from save file that is subset of total")
  .property( "SubCodaVector", &BayesSpikeCL::get_SubCodaSubSetCoords, 
    &BayesSpikeCL::set_SubCodaSubSetCoords, "Indices of desired coefficients of Beta for SubCodaList")  
  .property("SubCodaSubSetCoords", &BayesSpikeCL::get_SubCodaSubSetCoords, 
    &BayesSpikeCL::set_SubCodaSubSetCoords, "Indices of desired coefficients of Beta for SubCodaList")  
  .property("SubSetCoords", &BayesSpikeCL::get_SubCodaSubSetCoords, 
    &BayesSpikeCL::set_SubCodaSubSetCoords, "Indices of desired coefficients of Beta for SubCodaList")      
  .property("CSubSetCoords", &BayesSpikeCL::get_CSubSetCoords, 
    &BayesSpikeCL::set_CSubSetCoords, "Indices of desired coefficients of Beta for SubCodaList")   
  .property("SubSetTau", &BayesSpikeCL::get_SubSetTau, 
    &BayesSpikeCL::set_SubSetTau, "Indices of desired Tau dispersion parameters (Group indices) for TauCodaList, a subset of all taus")     
  .property("SubSetTau", &BayesSpikeCL::get_CSubSetTau, 
    &BayesSpikeCL::set_CSubSetTau, 
      "Indices of desired Tau dispersion parameters (Group indices) for TauCodaList, a subset of all taus")     
  .property( "PctBetas", &BayesSpikeCL::get_PctBetas, "Posterior Percentage On for Betas")
  .method("ClearPctBeta", &BayesSpikeCL::ClearPctBetas, "Posterior Percentage On for Betas")
  .method("ClearPctBetas", &BayesSpikeCL::ClearPctBetas, "Posterior Percentage On for Betas")
  .property( "PctTaus", &BayesSpikeCL::get_PctTaus, "Posterior Percentage On for Taus")
  .property( "Verbose", &BayesSpikeCL::get_Verbose, &BayesSpikeCL::set_Verbose, "Verbose")
  .property( "verbose", &BayesSpikeCL::get_Verbose, &BayesSpikeCL::set_Verbose, "Verbose")
  .property( "X", &BayesSpikeCL::get_X, &BayesSpikeCL::set_X, "X")
  .property( "Y", &BayesSpikeCL::get_Y, &BayesSpikeCL::set_Y, "Y")
  .property( "DoRecord", &BayesSpikeCL::get_DoRecord, &BayesSpikeCL::set_DoRecord, "DoRecord")
  .property( "cXtY", &BayesSpikeCL::get_cXtY,  "Compact XtY")
  .property( "Beta", &BayesSpikeCL::get_Beta, &BayesSpikeCL::set_Beta, "Beta")
  .method("SetNamesBeta", &BayesSpikeCL::set_NamesBeta, "Set the names of current Beta")
  .property( "sBeta", &BayesSpikeCL::get_Beta, &BayesSpikeCL::set_Beta, "Beta")
  .property( "beta", &BayesSpikeCL::get_Beta, &BayesSpikeCL::set_Beta, "Beta")
  .property( "CodaBeta", &BayesSpikeCL::get_CodaBeta, "Collapsed OutPut CodaBeta")
  .property( "Codajj", &BayesSpikeCL::get_Codajj, "Collapsed OutPut Codajj")
  .property( "XtY", &BayesSpikeCL::get_XtY,  "t(X) %*% Y")
  .property( "XtX", &BayesSpikeCL::get_XtX,  "t(X) %*%XY")
  .method("SubXtX", &BayesSpikeCL::get_SubXtX,  "Integer list subset of t(X) %*%XY" )
  .method("subXtX", &BayesSpikeCL::get_SubXtX,  "Integer list subset of t(X) %*%XY" )
  .property( "pXtX", &BayesSpikeCL::get_XtX,  "t(X) %*%XY")
  .property("NumMerges", &BayesSpikeCL::get_NumMerges, "Number of Merges from EE sampler")
  .property( "XLC", &BayesSpikeCL::get_XLC,  "XLC, Loctation Of X")
  .property( "NumActiveFixed", &BayesSpikeCL::get_NumActiveFixed,  "Number of Active Fixed Factors")
  .property( "NumActiveTau", &BayesSpikeCL::get_NumActiveTau,  "Number of Active Random Groups")
  .property( "tauFixed", &BayesSpikeCL::get_tauFixed, 
    &BayesSpikeCL::set_tauFixed, "tauFixed, Spread locations for fixed coefficients")
  .property( "tauFixedPrior", &BayesSpikeCL::get_tauFixedPrior, 
    &BayesSpikeCL::set_tauFixedPrior, "tauFixedPrior, Spread locations for fixed coefficients")
  .property( "ProbFixed", &BayesSpikeCL::get_ProbFixed,  "Probability of Fixed switching per factor")
  .property( "ProbTau", &BayesSpikeCL::get_ProbTau,  "Probability of tau switching per group")
  .property( "OrderedActive", &BayesSpikeCL::get_OrderedActive,  "OrderedActive, Active Members Of Beta")
  .property( "OldOrderedActive", &BayesSpikeCL::get_OldOrderedActive,  "OldOrderedActive, Active Members Of Beta")
  .property( "XtResid", &BayesSpikeCL::get_XtResid,  "t(X) %*% ( Y - X %*% Beta) ")
  .property( "kXFinder", &BayesSpikeCL::get_kXFinder,  "kXFinder, Order of active coefficients")  
  .property( "iiWeight", &BayesSpikeCL::get_iiWeight, 
     &BayesSpikeCL::set_iiWeight,"iiWeight, Weighting Matrix")
  .property("iWeightedXtX", &BayesSpikeCL::get_iWeightedXtX, "Whether pXtX has a column correctly weighted")
  .method("SetupSumiiWeight", &BayesSpikeCL::SetupSumiiWeight, "Setup Sum iiWeight")
  .property("AverageiiWeight", &BayesSpikeCL::GetAverageiiWeight, "Average ii Weight")
  .property("CountiiWeight", &BayesSpikeCL::get_CountiiWeight, "Count of ii Weights used")
  .property( "AllEigenVectors", &BayesSpikeCL::get_AllEigenVectors, 
     &BayesSpikeCL::set_AllEigenVectors,"AllEigenValues, EigenValues of Random Effects Groups")
  .property( "AllEigenValues", &BayesSpikeCL::get_AllEigenValues, 
     &BayesSpikeCL::set_AllEigenValues,"AllEigenVectors, Weighting Matrix")
  .property("WW", &BayesSpikeCL::get_WW, "WW weighting vector")
  .property("intY", &BayesSpikeCL::get_intY, "Integer Binary Y")
  .method("RobitReplace", &BayesSpikeCL::RobitReplace, "ReplaceYWith Robit Draws")
  .property("dfRobit", &BayesSpikeCL::get_dfRobit, "df of T distribution for Robit regression")
  .property( "OnPiA", &BayesSpikeCL::get_OnPiA, 
     &BayesSpikeCL::set_OnPiA,"OnPiA, BayesB Prior Probabilities of Fixed and Random Effects")
  .property( "sOnPiA", &BayesSpikeCL::get_OnPiA, 
     &BayesSpikeCL::set_OnPiA,"OnPiA, BayesB Prior Probabilities of Fixed and Random Effects")
  .property( "PiA", &BayesSpikeCL::get_OnPiA, 
     &BayesSpikeCL::set_OnPiA,"OnPiA, BayesB Prior Probabilities of Fixed and Random Effects")
  .property( "piA", &BayesSpikeCL::get_OnPiA, 
     &BayesSpikeCL::set_OnPiA,"OnPiA, BayesB Prior Probabilities of Fixed and Random Effects")
  .property( "FirstRandom", &BayesSpikeCL::get_FirstRandom, 
     &BayesSpikeCL::set_FirstRandom,"Indice of First Random Effect")
  .property( "RawiFirstRandom", &BayesSpikeCL::get_RawiFirstRandom, 
     &BayesSpikeCL::set_RawiFirstRandom,"Indice of First Random Effect (Raw i)")
  .property( "iFirstRandom", &BayesSpikeCL::get_RawiFirstRandom, 
     &BayesSpikeCL::set_RawiFirstRandom,"Indice of First Random Effect (Raw i)")
  .property("InitFlags", &BayesSpikeCL::get_InitFlags,
    "Initial Flags given to instantiation of BayesSpikeCL")
  .property( "CFirstRandom", &BayesSpikeCL::get_CFirstRandom, 
     &BayesSpikeCL::set_CFirstRandom,
     "Indice of First Random Effect (C scale from 0:(n-1))")
  .property( "PiAPrior", &BayesSpikeCL::get_PiAPrior, 
     &BayesSpikeCL::set_PiAPrior,"PiAPrior, Prior on OnPiA")
  .property( "OnSigma", &BayesSpikeCL::get_OnSigma, 
     &BayesSpikeCL::set_OnSigma,"OnSigma, Estimate of Gaussian Noise")
  .property( "Sigma", &BayesSpikeCL::get_OnSigma, 
     &BayesSpikeCL::set_OnSigma,"OnSigma, Estimate of Gaussian Noise")
  .property( "SigmaPrior", &BayesSpikeCL::get_SigmaPrior, 
     &BayesSpikeCL::set_SigmaPrior,"SigmaPrior, Weighting Matrix")
  .property("LengthSigmaPrior", &BayesSpikeCL::getLSigmaPrior, "SigmaPrior: length")
  .method("TestSigma", &BayesSpikeCL::TestSigma, "Test for Memory Integrity of Sigma and Sigma Prior")
  .property( "dfTNoise", &BayesSpikeCL::get_dfTNoise, 
     &BayesSpikeCL::set_dfTNoise,"dfTNoise, Degree of freedom for T Noise")
  .property( "tauPriordf", &BayesSpikeCL::get_staupriordf, 
     &BayesSpikeCL::set_staupriordf,"tau Prior DF, Degree of freedom for Tau Random Effects")
  .property( "tauPriorMean", &BayesSpikeCL::get_staubarnu, 
     &BayesSpikeCL::set_staubarnu,"tau Prior mean, Prior Mean for Tau Random Effects") 
  .property( "PriorXScaling", &BayesSpikeCL::get_sPriorXScaling, 
     &BayesSpikeCL::set_sPriorXScaling,"PriorXScaling, Prior Info for Tau Random Effects")          
  .property( "PriorOfTau", &BayesSpikeCL::get_sPriorOftau, 
     &BayesSpikeCL::set_sPriorOftau,"PriorOfTau, Density of Prior for Tau Random Effects")  
  .property( "OrderAttack", &BayesSpikeCL::get_OrderAttack, 
     "OrderAttack, Order to attack Fixed Effects") 
  .method("FillsBetaFromPropBetaAndCompute", &BayesSpikeCL::FillsBetaFromPropBetaAndCompute,
    "FillssBetafromPropBeta and computes issues")
  .property( "CodaTable", &BayesSpikeCL::get_CodaTable, 
     &BayesSpikeCL::set_CodaTable,
     "CodaTable, MCMC Output")
  .property("StartRecordingiiWeight", &BayesSpikeCL::get_StartRecordingiiWeight, "StartRecordingiiWeight")
  .property("copy", &BayesSpikeCL::get_Copy, "A Full BayesSpike Object regenerated copy of current object, new allocation")
  .property("DoLogitNonePostPreProb", &BayesSpikeCL::get_DoLogitNonePostPreProb, &BayesSpikeCL::set_DoLogitNonePostPreProb,
    "DoLogitNonePostPreProb")
  .property("DoLogitPostProb", &BayesSpikeCL::get_DoLogitNonePostPreProb, &BayesSpikeCL::set_DoLogitNonePostPreProb,
    "DoLogitPostProb")
  .method("ReorderXtX", &BayesSpikeCL::ReorderXtX, "ReorderXtX")
  .method("SetupTimePartsList", &BayesSpikeCL::SetupRsTimePartsList, "Setup Computation Time of Various functions")
  .property("TauMaximium", &BayesSpikeCL::MaximizeOn, "Value for which f2(tau^2) is maximized.")
  .property("RsTimePartsList", &BayesSpikeCL::get_RsTimePartsList, "Computation Time of Various functions")
  .property("TimePartsList", &BayesSpikeCL::get_RsTimePartsList, &BayesSpikeCL::set_RsTimePartsList,  "Computation Time of Various functions")
  .property("sTimePartsList", &BayesSpikeCL::get_RsTimePartsList, &BayesSpikeCL::set_RsTimePartsList, "Computation Time of Various functions")
  .property("TimePartsList", &BayesSpikeCL::get_RsTimePartsList, &BayesSpikeCL::set_RsTimePartsList, "Computation Time of Various functions")
  .property("ValTauMaximium", &BayesSpikeCL::ValAtMaximum, "Value for which f2(tau^2) is maximized.")
  .property("xM2", &BayesSpikeCL::MaximizeOn, "Value for which f2(tau^2) is maximized.")
  .property("lM2xM2", &BayesSpikeCL::ValAtMaximum, "Value for which f2(tau^2) is maximized.")  
  .property("Df3", &BayesSpikeCL::get_Df3, "Degrees of Freedom of Upper Bound density q3(x).")
  .property("Df4", &BayesSpikeCL::get_Df4, "Degrees of Freedom of Lower Bound density q4(x).")
  .property("xM3", &BayesSpikeCL::get_xM3, "Maximizer of Upper Bound density q3(x).")
  .property("xM4", &BayesSpikeCL::get_xM4, "Maximizer of  Lower Bound density q4(x).")
  .property("lM3xM3", &BayesSpikeCL::get_lM3xM3, "Value at Maximum xM3 for q3(x).")
  .property("lM4xM4", &BayesSpikeCL::get_lM4xM4, "Value at Maximum xM4 for q4(x).")
  .property("lUB32", &BayesSpikeCL::get_lUB32, "Upper Bound value F_3>2")
  .property("lLB42", &BayesSpikeCL::get_lLB42, "Lower Bound value F_4<2")
  .property("ScaleOfSmallXtResid", &BayesSpikeCL::get_ScaleOfSmallXtResid, "Scaling factor for ScaleOfSmallXtResid.")
  .property("ScaleOfEigenVectors", &BayesSpikeCL::get_ScaleOfEigenVectors, "Scaling factor for ScaleOfEigenVectors.")  
  .property("NewlA", &BayesSpikeCL::get_NewlA, "Draw a new lA value.")  
  .property("lA", &BayesSpikeCL::get_lA, "Give current a lA value for current X4.")
  .property("NewX4", &BayesSpikeCL::get_NewX4, "Give current a lA value for current X4.")
  .property("UsePrior", &BayesSpikeCL::get_UsePrior, "Value of Prior Logit used in SampleATau.")
  .property("InitlA", &BayesSpikeCL::get_InitlA, "log A(t) before sample is taken")
  .property("lMyHLB", &BayesSpikeCL::get_lMyHLB, "log(1-exp(-lUB32)) approx, minimum value for log(A(t))")
  .property("f2X4", &BayesSpikeCL::get_f2X4, "Value of f2 measured at random point X4")  
  .property("q4X4", &BayesSpikeCL::get_q4X4, "Value of q4 measured at random point X4")  
  .property("X4", &BayesSpikeCL::get_OldX4, "X4 is a random point last generated from density q4(x)") 
  .property("NewX4", &BayesSpikeCL::get_NewX4, "X4 is a new random point generated from density q4(x)") 
  .property("Codatjj", &BayesSpikeCL::get_CodaTjj, "A Buffer with locations of Tau on File")
  .property("CodaTLocjj", &BayesSpikeCL::get_CodaTLocjj, "A Buffer with locations of Tau on File")
  .property("CodaILocFile", &BayesSpikeCL::get_CodaILocFile,
     "CodaILocFile: file recording Location in this List of former temperature draws")
  .property("OldCodaILocFile", &BayesSpikeCL::get_OldCodaILocFile,
     "OldCodaILocFile: file recording Location in Previous List of former temperature draws")
  .property("NoSave", &BayesSpikeCL::get_NoSave, &BayesSpikeCL::set_NoSave,
    "A Flag that kills the save of effects")
  .property("DoSave", &BayesSpikeCL::get_DoSave, &BayesSpikeCL::set_DoSave,
    "A Flag setting whether to save to Hard-disk")
  .property("CodaProbFile", &BayesSpikeCL::get_CodaProbFile,
     &BayesSpikeCL::set_CodaProbFile,
     "CodaProbFile: file recording Likelihood (Temp =1) in this list of temperature draws")
  .property("OldCodaProbFile", &BayesSpikeCL::get_OldCodaProbFile,
     "OldCodaProbFile: file recording Likelihood (Temp=1) in Previous List of former temperature draws")
  .property("Tempii", &BayesSpikeCL::get_Tempii, &BayesSpikeCL::set_Tempii, 
    "Move to iith position in Temperature List")
  .property("TemperatureList", &BayesSpikeCL::get_TemperatureList, &BayesSpikeCL::set_TemperatureList,
    "Set TemperatureList for System")
  .property("Temperature", &BayesSpikeCL::get_Temperature, "Temperature of system")
  .property("invTemperature", &BayesSpikeCL::get_invTemperature, "inverse Temperature of system (Beta)")
  .property("OldTemperature", &BayesSpikeCL::get_OldTemperature, &BayesSpikeCL::set_OldTemperature, "Old Temperature of Old Chain")
  .property("invOldTemperature", &BayesSpikeCL::get_invOldTemperature, "inverse Old Temperature of system")
  .property("AFD", &BayesSpikeCL::get_AFD, &BayesSpikeCL::set_AFD, 
    "AFD, Sexp Pointer to Diallel Object for use") 
  .property( "WeightedXtX", &BayesSpikeCL::get_WeightedXtX, "Which XtX are currently weighted?")
  .property( "NeedRecord", &BayesSpikeCL::CountRecord, "Needed Dim to Coda Table")
  .property( "MaxGibbsIters", &BayesSpikeCL::get_MaxGibbsIters, 
     &BayesSpikeCL::set_MaxGibbsIters,"MaxGibbsIters, Number of Iters in one MCMC chain")     
  .property( "NumSamples", &BayesSpikeCL::get_MaxGibbsIters, 
     &BayesSpikeCL::set_MaxGibbsIters,"MaxGibbsIters, Number of Iters in one MCMC chain")
  .property( "BFixed", &BayesSpikeCL::get_BFixed,  "BFixed, 1 or zero for active Fixed Variables") 
  .property( "PropBeta", &BayesSpikeCL::get_PropBeta,  "PropBeta, PropsedValues for Partial Vec")  
  .property( "MergeActive", &BayesSpikeCL::get_MergeActive,  "Last active set pulled in EEsamplerMerge") 
  .property( "MergePropBeta", &BayesSpikeCL::get_MergePropBeta,  "Beta of last active set in EE sampler Merge")
  .property( "LengthMergeActive", &BayesSpikeCL::get_LengthMergeActive,  "Length of MergeActive") 
  .property( "MaxLengthMergeActive", &BayesSpikeCL::get_MaxLengthMergeActive,  "Merge Active Allocation Size")  
  .property( "OnKappaS", &BayesSpikeCL::get_OnKappaS,  "OnKappaS")
  .property( "OnKappaMem", &BayesSpikeCL::get_OnKappaMem,  "OnKappaMem")  
  .property( "YSq", &BayesSpikeCL::get_YSq,  "YSq") 
  .property( "rI", &BayesSpikeCL::get_rI,  "reduced Inverse Matrix")
  .property( "SmallXtX", &BayesSpikeCL::get_SmallXtX,  "SmallXtX") 
  .property( "smXtX", &BayesSpikeCL::get_SmallXtX,  "SmallXtX") 
  .property( "XjSq", &BayesSpikeCL::get_XjSq,  "Squared XtX diagonal") 
  .property( "rIChol", &BayesSpikeCL::get_rIChol,  "rIChol")  
  .property( "rICholSym", &BayesSpikeCL::get_rICholSym,  "rICholSym")  
  .property( "NumActive", &BayesSpikeCL::get_NumActive,  "Number of total Active")  
  .property( "YResidSq", &BayesSpikeCL::get_YResidSq,  "YResidSq")
  .property( "D", &BayesSpikeCL::get_D,  "D Vector")
  .property( "R", &BayesSpikeCL::get_R,  "R Vector")
  .property( "TestMemAllocateFailure", &BayesSpikeCL::TestMemAllocateFailure, "TestMemAllocateFailure")
  .property("TweakCons", &BayesSpikeCL::get_TweakCons, "Divisor Tweak that helps to computationally improve sOnL Of X.")
  .property("HadSetPriorProbTau", &BayesSpikeCL::get_HadSetPriorProbTau, "Have we set PriorProbTau?")
  .property("HadSetPriorProbFixed", &BayesSpikeCL::get_HadSetPriorProbFixed, "Have we set PriorProbFixed?")  
  .property("PriorProbs", &BayesSpikeCL::get_PriorProbs, "getPriorProbs")
  .method("setCodaTFile", &BayesSpikeCL::set_sCodaTFile, "Set Files for Record Taus")
  .property("CodaTisSetup", &BayesSpikeCL::get_CodaTisSetup, "Is CodaTau Setup?")
  .property("LengthWrittenBetaAllDrawBuffer", &BayesSpikeCL::get_LengthWrittenBetaAllDrawBuffer, 
    "Length Written to BetaAllDrawBuffer")
  .property("LengthTotalWrittenBetaAllDrawBuffer", &BayesSpikeCL::get_LengthTotalWrittenBetaAllDrawBuffer,
    "Length of everything written to BetaAllDrawBufer and cleared")
  .property("BetaAllDrawBuffer", &BayesSpikeCL::get_BetaAllDrawBuffer,
    "BetaAllDrawBuffer")
  .property("AllBetaAllDrawBuffer", &BayesSpikeCL::get_AllBetaAllDrawBuffer,
    "All BetaAllDrawBuffer")
  .property("LengthBetaAllDrawBuffer", &BayesSpikeCL::get_LengthBetaAllDrawBuffer,
    "Length of BetaAllDrawBufer")
  .method("SetupRsCodaBetaAllDrawBuffer", &BayesSpikeCL::set_RsCodaBetaAllDrawBuffer,
    "Setup a Buffer drawing non zero values for all Beta")
  .property("RsCodaBetaAllDrawBuffer", &BayesSpikeCL::get_RsCodaBetaAllDrawBuffer,
    "File to Save CodaBetaAllDrawBuffer to")
  .property("FileBetaAllDrawBuffer", &BayesSpikeCL::get_RsCodaBetaAllDrawBuffer,
    "File to Save CodaBetaAllDrawBuffer to")
  .property("SubCodaLongList", &BayesSpikeCL::get_SubCodaLongList,
    &BayesSpikeCL::set_SubCodaLongList, "A Coda object of non-shrunk to zero Beta Draws")
  .property("CodaPFile", &BayesSpikeCL::get_sCodaPFile, 
    &BayesSpikeCL::set_sCodaPFile, "Set File for Record Prob")
  .property("CodaiTFile", &BayesSpikeCL::get_sCodaiTFile, 
     "File for Integers of Tau")
  .property("FullFileiT", &BayesSpikeCL::get_sCodaiTFile, 
     "File for Integers of Tau")
  .property("d3", &BayesSpikeCL::get_d3, 
    "Degrees of freedom for 'd3' lower bound density, Alsswitching")
  .property("d4", &BayesSpikeCL::get_d4, 
    "Degrees of freedom for 'd4' lower bound density, Alsswitching")
  .property("CodaiDFile", &BayesSpikeCL::get_sCodadTFile, 
     "File for Doubles of Tau")
  .property("CodadTFile", &BayesSpikeCL::get_sCodadTFile, 
     "File for Doubles of Tau")
  .property("FullFiledT", &BayesSpikeCL::get_sCodadTFile, 
     "File for Doubles of Tau")
  .property("MT", &BayesSpikeCL::get_MT, "MT TauOfFContainer sub object")
  //.property("ChainIter", &BayesSpikeCL::get_ChainIter, &BayesSpikeCL::set_ChainIter,
  //  "Chain Iteration for CodaTable")
  .property("OnCodaTable", &BayesSpikeCL::get_OnCodaTable, &BayesSpikeCL::set_OnCodaTable,
    "OnCodaTable number of table in CodaList to check up on")
  .method("SampleANewTau", &BayesSpikeCL::SampleANewTau, "Sample A New Tau")
  .method("DeleteMe", &BayesSpikeCL::DeleteMe, "Delete Me")
  .method("SetupMT", &BayesSpikeCL::SetupMT, "Setup the MT")
  .method("SetupMTForOnii",&BayesSpikeCL::SetupMTForOnii, "SetupMTForOnii")
  .method("MaximizeOn", &BayesSpikeCL::MaximizeOn, "MaximizeOn")
  .method("sLOfX", &BayesSpikeCL::sLOfX, " Log Tau Likelihood of X ")
  .method("RICholSubDim", &BayesSpikeCL::RICholSubDim, "Give Sub equation of rIChol, supply dim (For partBeta)")
  .method("RISubDim", &BayesSpikeCL::RISubDim, "Give Sub equation of rI, supply dim (For partBeta)")
  .method("BAllDeriv", &BayesSpikeCL::BAllDeriv, "Derivative of all members")
  .method("BAllSDeriv", &BayesSpikeCL::BAllSDeriv, "Second Derivative of all members")
  .property( "DependenciesFixed", &BayesSpikeCL::get_DependenciesFixed, 
     &BayesSpikeCL::set_DependenciesFixed,
     "DependenciesFixed:  Fixed Dependencies on nearby QTL")
  .method("SetCodaILocAndProbFile", &BayesSpikeCL::set_sCodaILocAndProbFile, "Sets ILoc and ProbFile")
  .method("FillPropBetaFromBeta", &BayesSpikeCL::FillPropBetaFromBeta, "Fills PropBeta from Beta")
  .method("GRunif", &BayesSpikeCL::GRunif, "Get some random uniforms hopefully") 
  .property( "MaxX", &BayesSpikeCL::get_MaxX, "Max for SpliceSample")  
  .property( "DependenciesTau", &BayesSpikeCL::get_DependenciesTau, 
     &BayesSpikeCL::set_DependenciesTau,
     "DependenciesTau:  Fixed Dependencies on nearby QTL") 
  .method("BPriorLOfX", &BayesSpikeCL::BPriorLOfX, "B Prior of Value")
  .property("TauCodaList", &BayesSpikeCL::get_TauCodaList, &BayesSpikeCL::set_TauCodaList, "TauCodaList - Coda MCMC list of Tau draws - set SubSetTau to choose vectors to keep")
  .property("SubTauList", &BayesSpikeCL::get_TauCodaList, &BayesSpikeCL::set_TauCodaList, "TauCodaList - Coda MCMC list of Tau draws - set SubSetTau to choose vectors to keep.  Note this is same as TauCodaList.")
  .property("tauCodaList", &BayesSpikeCL::get_TauCodaList,&BayesSpikeCL::set_TauCodaList, "TauCodaList -- alias ")
  .property("CodaTauList", &BayesSpikeCL::get_TauCodaList,&BayesSpikeCL::set_TauCodaList, "TauCodaList -- alias")  
  .property("SetupYSq", &BayesSpikeCL::SetupYSq, "Refreshes YSq")
  .property("YSq", &BayesSpikeCL::SetupYSq, "Refreshes YSq")
  .property("RNGState", &BayesSpikeCL::get_RNGState, &BayesSpikeCL::set_RNGState, "RNGState")
  .property("PriorProbFixed", &BayesSpikeCL::get_PriorProbFixed, &BayesSpikeCL::set_PriorProbFixed, "PriorProbFixed")
  .property("PriorProbTau", &BayesSpikeCL::get_PriorProbTau, 
    &BayesSpikeCL::set_PriorProbTau, "PriorProbTau")
  .method("CheckkXFinderError", &BayesSpikeCL::DoCheckkXFinderError, "Check for kXFinder")
  .method("Resize", &BayesSpikeCL::Resize, "Resize Memory larger")
  .method("UpdateFreshXtResid", &BayesSpikeCL::UpdateFreshXtResid, "Update Fresh XtResid")
  .method("TestXtResid", &BayesSpikeCL::TestXtResid, "Update Fresh XtResid")
  .method("SetupYResidSq", &BayesSpikeCL::SetupYResidSq, "Setup YResidSq Calculation")
  .method("RecordHistory", &BayesSpikeCL::RecordHistory, "Record the History")
  .method("WriteCoda", &BayesSpikeCL::WriteCoda, "Write Coda")
  .method("RecordCoda", &BayesSpikeCL::RecordCoda, "Record to Coda")
  .method("WritePICoda", &BayesSpikeCL::WritePICoda, "Write PI to Coda")
  .method("BDerivPrior", &BayesSpikeCL::BDerivPrior, "BDerivPrior")
  .method("ReweightEigenvalues", &BayesSpikeCL::ReweightEigenvalues, "ReweightEigenValues")
  .method("BSDerivlF", &BayesSpikeCL::BSDerivlF, "BSDerivlF")
  .method("r2o", &BayesSpikeCL::r2o, "Random Draw from Unknown Tau distribution")
  .method("r2", &BayesSpikeCL::r2, "A Number of Random Draws from Unknown Tau distribution")
  .method("BSDerivPrior", &BayesSpikeCL::BSDerivPrior, "B Second derivative of DerivPrior")
  .method("RefreshsBeta", &BayesSpikeCL::RefreshsBeta, "RefreshsBeta with new vector")
  .method("RefreshBeta", &BayesSpikeCL::RefreshsBeta, "RefreshBeta with new vector")
  .method("PutRNGState", &BayesSpikeCL::rPutRNGstate, "End Randomization for C")
  .method("ReweightedOneXtX", &BayesSpikeCL::ReweightedOneXtX, "ReweightXtX")
  .method("QuickCheckWeightXtX", &BayesSpikeCL::QuickCheckWeightXtX, "Have we Reweighted XtX correctly?")
  .field("DontPartBetaNoise", &BayesSpikeCL::DontPartBetaNoise, " Flag whether to eliminate noise to test SamplePartBeta2")
  .method("UpdateSigma", &BayesSpikeCL::UpdateSigma, "Update OnSigmaSq")
  .method("SampleFixedB", &BayesSpikeCL::SampleFixedB, " Sample New Fixed B")
  .method("SetupRICholBack", &BayesSpikeCL::SetupRICholBack, "SetupRICholBack")
  .method("UpdatePiA", &BayesSpikeCL::UpdatePiA, "Update PiA estimation")
  .method("ReadFileCodas", &BayesSpikeCL::ReadFileCodas, "Read Info Saved to HardDrive from Chains") 
  .method("SampleNewTaus", &BayesSpikeCL::SampleNewTaus, " Sample Tau parameters for random effects")
  .method("PrepareForRegression",  &BayesSpikeCL::PrepareForRegression, "Prepare Regression Step")
  .method("SamplePropBeta", &BayesSpikeCL::SamplePropBeta, " Sample PropBeta from Gibbs Step")
  .method("ReadSetPctBetas", &BayesSpikeCL::ReadSetPctBetas, " Read from Files and compile On Pct of Betas, input StartIter, EndIter")
  .method("ReweightOnlyActiveXtX", &BayesSpikeCL::ReweightOnlyActiveXtX, "Reweight XtX on Active cells")
  .method("UpdateNonFreshXtResid", &BayesSpikeCL::UpdateNonFreshXtResid, "Update XtResid without altering XtX")
  .method("CopyIn", &BayesSpikeCL::CopyIn, "Copy into rIChol, the part of rICholBack from StF to StF + GoFor (input StF, GoFor)")
  .method("ProbPosterior", &BayesSpikeCL::ProbPosterior, "Calculate Likelihood probability of current sample")
  .property("JustDidPartBeta", &BayesSpikeCL::get_JustDidPartBeta, "Did we just do PartBeta last tick?")
  .property("JustDidPartBFixed", &BayesSpikeCL::get_JustDidPartBFixed, "Did we just take Betas from Part BFixed?  What was first indice?")
  .method("SamplePartBeta", &BayesSpikeCL::SamplePartBeta, 
    "Sample only part of active beta from StF, to StF+GoWant, recursively subsamples if necesssary Call FillSmallXtX")  
    .method("FillSmallXtX", &BayesSpikeCL::FillSmallXtX, 
    "  Fills SmallXtX with relevant columns from XtX")
  .method("FillSmallXtResid", &BayesSpikeCL::FillSmallXtResid, 
    "  Fills smXtY with XtResid, useful in SamplePartBeta2")
  .method("CopyIn2", &BayesSpikeCL::CopyIn2, 
    "  Fills smrIChol with small part of XtX, useful in SamplePartBeta2")
  .method("SamplePartBeta2", &BayesSpikeCL::SamplePartBeta2, 
    "Sample only part of active beta from StF, to StF+GoWant, recursively subsamples if necesssary Call FillSmallXtResid first")     
  .method("SetupBFixedAndFriends", &BayesSpikeCL::SetupBFixedAndFriends, "Sets up BFixed")
  .method("SetupYBuffer", &BayesSpikeCL::SetupYBuffer, "Record Y values from Gibbs Sampler to File")
  .property("LengthYBuffer", &BayesSpikeCL::get_LengthYBuffer, "Lenth Of YBuffer if setup")
  .property("LengthWrittenYBuffer", &BayesSpikeCL::get_LengthWrittenYBuffer, "Lenth Of YBuffer if setup")
  .property("LocationOfBetaWrite", &BayesSpikeCL::get_LocationOfBetaWrite, "Location of Beta")
  .method("FillYBuffer", &BayesSpikeCL::FillYBuffer, "Record Y values from Gibbs Sampler to File")
  .method("WriteYBuffer", &BayesSpikeCL::WriteYBuffer, "Record Y values from Gibbs Sampler to File")  
  .property("YBuffer", &BayesSpikeCL::get_CurrentYBuffer, "Matrix of Current Y Buffer")
  .property("DeCenteredCodaList", &BayesSpikeCL::get_RsDeCenteredCodaList,
    &BayesSpikeCL::set_RsDeCenteredCodaList, "Centered Version of Coda List")
  .property("CenteredColumns", &BayesSpikeCL::get_CenteredColumns, &BayesSpikeCL::set_CenteredColumns, "What columns have been centered in Coda List")
  .method("UnSave", &BayesSpikeCL::UnSave, "Remove all saves for BayesSpikeCL object (Must supply confirm flag!)")
  .method("unsave", &BayesSpikeCL::UnSave, "Remove all saves for BayesSpikeCL object (Must supply confirm flag!)")
  .method("Unsave", &BayesSpikeCL::UnSave, "Remove all saves for BayesSpikeCL object (Must supply confirm flag!)")
  .method("Save", &BayesSpikeCL::SSave, "Save the current BayesSpikeCL object")
  .method("save", &BayesSpikeCL::SSave, "Save the current BayesSpikeCL object")
  .method("ssave", &BayesSpikeCL::SSave, "Save the current BayesSpikeCL object")
  .method("SSave", &BayesSpikeCL::SSave, "Save the current BayesSpikeCL object")  
  .method("RunRegression", &BayesSpikeCL::RunRegression, "Complete Running Of Bayesian Regression") 
  .property("CurrentYBuffer", &BayesSpikeCL::get_CurrentYBuffer, "Matrix of Current Y Buffer")
  .property("AllYBuffer", &BayesSpikeCL::get_AllYBuffer, "Matrix of Total YBuffer")
  .property("NewYBufferWrite", &BayesSpikeCL::get_NewYBufferWrite,
    &BayesSpikeCL::set_NewYBufferWrite, "Whether to overwrite file on next flush of YBuffer")
  .property("LengthTotalWrittenYBuffer", &BayesSpikeCL::get_LengthTotalWrittenYBuffer, "Total Amount of YBuffer written to File")
  .property("TotalWrittenYBuffer", &BayesSpikeCL::get_LengthTotalWrittenYBuffer, "Total Amount of YBuffer written to File")
  .method("SetupWeightBuffer", &BayesSpikeCL::SetupWeightBuffer, "Record Weight values from Gibbs Sampler to File")
  .property("LengthWeightBuffer", &BayesSpikeCL::get_LengthWeightBuffer, "Length Of WeightBuffer if setup")
  .property("LengthWrittenWeightBuffer", &BayesSpikeCL::get_LengthWrittenWeightBuffer, "Length Of WeightBuffer if setup")
  .method("FillWeightBuffer", &BayesSpikeCL::FillWeightBuffer, "Record Weight values from Gibbs Sampler to File")
  .method("WriteWeightBuffer", &BayesSpikeCL::WriteWeightBuffer, "Record Weight values from Gibbs Sampler to File")  
  .property("WeightBuffer", &BayesSpikeCL::get_CurrentWeightBuffer, "Matrix of Current Weight Buffer")
  .property("CurrentWeightBuffer", &BayesSpikeCL::get_CurrentWeightBuffer, "Matrix of Current Weight Buffer")
  .property("AllWeightBuffer", &BayesSpikeCL::get_AllWeightBuffer, "Matrix of Total WeightBuffer")
  .property("LengthTotalWrittenWeightBuffer", &BayesSpikeCL::get_LengthTotalWrittenWeightBuffer, "Total Amount of WeightBuffer written to File")
  .property("TotalWrittenWeightBuffer", &BayesSpikeCL::get_LengthTotalWrittenWeightBuffer, "Total Amount of WeightBuffer written to File")
  .property("NewWeightBufferWrite", &BayesSpikeCL::get_NewWeightBufferWrite,
    &BayesSpikeCL::set_NewWeightBufferWrite, "Whether to overwrite file on next flush of WeightBuffer")
  .method("TestPt", &BayesSpikeCL::TestPt, "Test Pt Function")
  .method("TestPnorm", &BayesSpikeCL::TestPnorm, "Test Pnorm Function")
  .property("MaxCODABeta", &BayesSpikeCL::get_MaxCODABeta, "Length of Coda Tau sequence")
  .property("MaxCODA", &BayesSpikeCL::get_MaxCODABeta, "Length of Coda Tau sequence")  
  .property("WeightCodaList", &BayesSpikeCL::get_WeightCodaList,
    &BayesSpikeCL::set_WeightCodaList, "Coda mcmc.list of Weights for t noise draws.")
  .property("YCodaList", &BayesSpikeCL::get_YCodaList,
    &BayesSpikeCL::set_YCodaList, "Coda mcmc.list of Y latent space draws for Probit/Robit regressions.") 
  .property("PiACodaList", &BayesSpikeCL::get_PiACodaList,
    &BayesSpikeCL::set_PiACodaList, "Coda mcmc.list of PiA draws.")
  .method("SetupAlterWeightBuffer", &BayesSpikeCL::SetupAlterWeightBuffer,
    "Setup an alternative weight buffer for Regression")
  .property("AlterWeightCodaList", &BayesSpikeCL::get_AlterWeightCodaList,
    "Coda mcmc.list of AlterWeight draws.")
  .method("DeliberatelySetAlterWeightCodaList", &BayesSpikeCL::DeliberatelySetAlterWeightCodaList,
    "Put in the mcmc.list")
  .property("AlterWeightList", &BayesSpikeCL::get_AlterWeightCodaList,
    &BayesSpikeCL::DeliberatelySetAlterWeightCodaList, "Coda mcmc.list of AlterWeight draws.")
  .property("SigCodaList", &BayesSpikeCL::get_SigCodaList,
    &BayesSpikeCL::set_SigCodaList, "Coda mcmc.list of Sig aka sigma^2 draws.")   
  .property("SigmaCodaList", &BayesSpikeCL::get_SigCodaList,
    &BayesSpikeCL::set_SigCodaList, "Coda mcmc.list of Sigma aka sigma^2 draws.")   
  .property("sigmaCodaList", &BayesSpikeCL::get_SigCodaList,
    &BayesSpikeCL::set_SigCodaList, "Coda mcmc.list of Sigma aka sigma^2 draws.")   
  .property("AlterWeightFile", &BayesSpikeCL::get_RsAlterWeightFile,  "File that Alternative Weights will be Stored in.")    
  .property("RsAlterWeightFile", &BayesSpikeCL::get_RsAlterWeightFile,  "File that Alternative Weights will be Stored in.")  
  .property("YFile", &BayesSpikeCL::get_RsYFile,  "File that Y Buffer will be Stored in.")    
  .property("RsYFile", &BayesSpikeCL::get_RsYFile,  "File that Y Buffer will be Stored in.")    
  .property("WeightFile", &BayesSpikeCL::get_RsWeightFile,  "File that Weight Buffer will be Stored in.")    
  .property("RsWeightFile", &BayesSpikeCL::get_RsWeightFile,  "File that Weight Buffer will be Stored in.")   
  .property("SigCodaFile", &BayesSpikeCL::get_RsSigCodaFile,  "File that Y Buffer will be Stored in.")    
  .property("RsSigCodaFile", &BayesSpikeCL::get_RsSigCodaFile,  "File that Y Buffer will be Stored in.")    
  .property("PiACodaFile", &BayesSpikeCL::get_RsPiACodaFile,  "File that Y Buffer will be Stored in.")    
  .property("RsPiACodaFile", &BayesSpikeCL::get_RsPiACodaFile,  "File that Y Buffer will be Stored in.")    
  .method("SetupSigCodaBuffer", &BayesSpikeCL::SetupSigCodaBuffer, "Record Y values from Gibbs Sampler to File")
  .property("LengthSigCodaBuffer", &BayesSpikeCL::get_LengthSigCodaBuffer, "Lenth Of SigCodaBuffer if setup")
  .property("LengthWrittenSigCodaBuffer", &BayesSpikeCL::get_LengthWrittenSigCodaBuffer, "Lenth Of SigCodaBuffer if setup")
  .property("LocationOfBetaWrite", &BayesSpikeCL::get_LocationOfBetaWrite, "Location of Beta")
  .method("AddAllNewCoords", &BayesSpikeCL::AddAllNewCoords, "Location of Beta")
  .property("DoAddCoordsOnSetBeta", &BayesSpikeCL::get_DoAddCoordsOnSetBeta, 
    &BayesSpikeCL::set_DoAddCoordsOnSetBeta, "Flag to add coords to setBeta")
  .method("FillPostProbBuffer", &BayesSpikeCL::FillPostProbBuffer, "Record PostProb values from Gibbs Sampler to File")
  .method("FillSigCodaBuffer", &BayesSpikeCL::FillSigCodaBuffer, "Record Y values from Gibbs Sampler to File")
  .method("WriteSigCodaBuffer", &BayesSpikeCL::WriteSigCodaBuffer, "Record Y values from Gibbs Sampler to File")  
  .property("SigCodaBuffer", &BayesSpikeCL::get_CurrentSigCodaBuffer, "Matrix of Current Y Buffer")
  .property("CurrentSigCodaBuffer", &BayesSpikeCL::get_CurrentSigCodaBuffer, "Matrix of Current Y Buffer")
  .method("DeriveAlterProbability", &BayesSpikeCL::DeriveAlterProbability, "Function Derives Alternative Probabilities for logit/T noise from other draws of Y")
  .property("AllSigCodaBuffer", &BayesSpikeCL::get_AllSigCodaBuffer, "Matrix of Total SigCodaBuffer")
  .property("NewSigCodaBufferWrite", &BayesSpikeCL::get_NewSigCodaBufferWrite,
    &BayesSpikeCL::set_NewSigCodaBufferWrite, "Whether to overwrite file on next flush of SigCodaBuffer")
  .property("NewWriteSigCodaBuffer", &BayesSpikeCL::get_NewSigCodaBufferWrite,
    &BayesSpikeCL::set_NewSigCodaBufferWrite, "Whether to overwrite file on next flush of SigCodaBuffer")
  .property("LengthTotalWrittenSigCodaBuffer", &BayesSpikeCL::get_LengthTotalWrittenSigCodaBuffer, "Total Amount of SigCodaBuffer written to File")
  .property("TotalWrittenSigCodaBuffer", &BayesSpikeCL::get_LengthTotalWrittenSigCodaBuffer, "Total Amount of SigCodaBuffer written to File")      
 .method("SetupPiACodaBuffer", &BayesSpikeCL::SetupPiACodaBuffer, "Record Y values from Gibbs Sampler to File")
  .property("LengthPiACodaBuffer", &BayesSpikeCL::get_LengthPiACodaBuffer, "Lenth Of PiACodaBuffer if setup")
  .property("LengthWrittenPiACodaBuffer", &BayesSpikeCL::get_LengthWrittenPiACodaBuffer, "Lenth Of PiACodaBuffer if setup")
  .property("LocationOfBetaWrite", &BayesSpikeCL::get_LocationOfBetaWrite, "Location of Beta")
  .property("AlterWeightTemperature", &BayesSpikeCL::get_AlterWeightTemperature,
    &BayesSpikeCL::set_AlterWeightTemperature, "Alternate Temperature for Weight Function.")
  .method("FillPiACodaBuffer", &BayesSpikeCL::FillPiACodaBuffer, "Record Y values from Gibbs Sampler to File")
  .method("UpdateRunMoreZero", &BayesSpikeCL::UpdateRunMoreZero, "count how many are above zero")
  .method("AddToRunProbRegionVector", &BayesSpikeCL::AddToRunProbRegionVector, "Try and find a running Probability Region")
  .method("WritePiACodaBuffer", &BayesSpikeCL::WritePiACodaBuffer, "Record Y values from Gibbs Sampler to File")  
  .property("PiACodaBuffer", &BayesSpikeCL::get_CurrentPiACodaBuffer, "Matrix of Current Y Buffer")
  .property("CurrentPiACodaBuffer", &BayesSpikeCL::get_CurrentPiACodaBuffer, "Matrix of Current Y Buffer")
  .property("AllPiACodaBuffer", &BayesSpikeCL::get_AllPiACodaBuffer, "Matrix of Total PiACodaBuffer")
  .property("NewPiACodaBufferWrite", &BayesSpikeCL::get_NewPiACodaBufferWrite,
    &BayesSpikeCL::set_NewPiACodaBufferWrite, "Whether to overwrite file on next flush of PiACodaBuffer")
  .property("NewWritePiACodaBuffer", &BayesSpikeCL::get_NewPiACodaBufferWrite,
    &BayesSpikeCL::set_NewPiACodaBufferWrite, "Whether to overwrite file on next flush of PiACodaBuffer")
  .property("LengthTotalWrittenPiACodaBuffer", &BayesSpikeCL::get_LengthTotalWrittenPiACodaBuffer, "Total Amount of PiACodaBuffer written to File")
  .property("TotalWrittenPiACodaBuffer", &BayesSpikeCL::get_LengthTotalWrittenPiACodaBuffer, "Total Amount of PiACodaBuffer written to File")
  .method("SetCodaFileNames", &BayesSpikeCL::SetCodaFileNames, "Set Names to Save Coda Files");  
}
}
