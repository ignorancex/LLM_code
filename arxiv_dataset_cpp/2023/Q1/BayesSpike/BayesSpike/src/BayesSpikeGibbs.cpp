/* ========================================================================== */
/*                                                                            */
/*   BayesSpikeGibbs.cpp                                                      */
/*   (c) 2011 Alan Lenarcic                                                   */
/*                                                                            */
/*   Description                                                              */
/*                                                                            */
/* This is actual code for running overall BayesSpike Algorithm               */
/*  RunAlgorithm is algorithm that runs code and should be noted as directing */
/*  All chief algorithms related to BayesSpike.                               */
/*                                                                            */
/*  SampleANewTau, however, from BayesSpikeSliceSampling.cc                   */
/*              covers group selection                                        */
/*                                                                            */
/* Order of the algorithm is encoded by "MRF()" calls, which declare when     */
/*  and what function was ran last and what part of standard order            */
/*  the function exists in.                                                   */
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

#ifndef UNISTDH
  #include <unistd.h>
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <sys/types.h>
  #define UNISTDH 0
#endif


#ifndef TRUNCNORMH
  #include "TruncNorm.h"
  #define TRUNCNORMH 0
#endif

#ifndef HEAPSORTH
  #include "HeapSort.h"
  #define HEAPSORTH
#endif

#ifndef AKILLDERIVEFILE
  #define AKILLDERIVEFILE(X) \
    if (fd1 != -1  && fCodaIFile == NULL) { close(fd1); fd1 = -1; TotalClosedFiles++;}   \
    if (fd2 != -1 && fCodaDFile == NULL) { close(fd2); fd2 = -1; TotalClosedFiles++;}    \
    if (fd3 != -1 && fCodaiTFile == NULL ) { close(fd3); fd3 = -1; TotalClosedFiles++;}  \
    if (fd4 != -1 && fCodadTFile == NULL) { close(fd4); fd4 = -1; TotalClosedFiles++;}   \
    if (fCodaIFile != NULL) { fclose(fCodaIFile); fCodaIFile= NULL; TotalClosedFiles++; } \
    if (fCodaDFile != NULL) { fclose(fCodaDFile); fCodaDFile= NULL; TotalClosedFiles++; } \
    if (fCodaiTFile != NULL) { fclose(fCodaIFile); fCodaiTFile= NULL; TotalClosedFiles++; } \
    if (fCodadTFile != NULL) { fclose(fCodaIFile); fCodadTFile= NULL; TotalClosedFiles++; } \
    if (fd1 != -1) { fd1 = -1; TotalClosedFiles++;}                              \
    if (fd2 != -1) { fd2 = -1; TotalClosedFiles++;}                              \
    if (fd3 != -1) { fd3 = -1; TotalClosedFiles++;}                              \
    if (fd4 != -1) { fd4 = -1; TotalClosedFiles++;}                              \
    if (fdILoc != -1 && fCodaILocFile == NULL) { close(fdILoc); fdILoc = -1; TotalClosedFiles++;}  \
    if (fdProb != -1 && fCodaProbFile == NULL) { close(fdProb); fdProb = -1; TotalClosedFiles++;}   \
    if (fCodaILocFile != NULL) { fclose(fCodaILocFile); fCodaILocFile = NULL;  TotalClosedFiles++; } \
    if (fCodaProbFile != NULL) { fclose(fCodaProbFile); fCodaProbFile = NULL;  TotalClosedFiles++; } \
    if (fdILoc != -1) { fdILoc = -1; TotalClosedFiles++;}     \
    if (fdProb != -1) { fdProb = -1; TotalClosedFiles++;}     \
    do {                                                      \
    } while(FALSE)
#endif

SEXP BayesSpikeCL::get_RsTimePartsList() {
  if (RsTimePartsList != NULL) {
    return(RsTimePartsList->asSexp());
  }
  return(R_NilValue);
}
// SetupRsTimePartsList()
// 
//   This coveniently names all of the functions we want to install
//   to access by .Call interface.
//
int BayesSpikeCL::SetupRsTimePartsList() {
  if (RsTimePartsList != NULL) {
    DDelete(RsTimePartsList, "RsTimePartsList");
  }
  RsTimePartsList  = new AObject(Rf_allocVector(REALSXP, 35));
  for (int ii = 0; ii < 35; ii++) {
    REAL(RsTimePartsList->asSexp())[ii] = 0.0;
  }
  SEXP sStrings = R_NilValue;
  Rf_protect(sStrings = Rf_allocVector(STRSXP, 35));   
  SET_STRING_ELT(sStrings, 0, Rf_mkChar("SampleFixedB"));
  SET_STRING_ELT(sStrings, 1, Rf_mkChar("SampleNewTaus"));
  SET_STRING_ELT(sStrings, 2, Rf_mkChar("AddAllNewCoords")); 
  SET_STRING_ELT(sStrings, 3, Rf_mkChar("UpdatePiA"));
  SET_STRING_ELT(sStrings, 4, Rf_mkChar("RefreshOrderedActive"));
  SET_STRING_ELT(sStrings, 5, Rf_mkChar("PrepareForRegression"));
  SET_STRING_ELT(sStrings, 6, Rf_mkChar("SamplePropBeta"));
  SET_STRING_ELT(sStrings, 7, Rf_mkChar("FillsBetaFromPropBetaAndCompute"));
  SET_STRING_ELT(sStrings, 8, Rf_mkChar("RobitReplace"));
  SET_STRING_ELT(sStrings, 9, Rf_mkChar("UpdateTNoise"));
  SET_STRING_ELT(sStrings, 10, Rf_mkChar("UpdateSigma"));
  SET_STRING_ELT(sStrings, 11, Rf_mkChar("RecordHistory"));
  SET_STRING_ELT(sStrings, 12, Rf_mkChar("RecordCoda"));  
  SET_STRING_ELT(sStrings, 13, Rf_mkChar("FillPostProbBuffer"));  
  SET_STRING_ELT(sStrings, 14, Rf_mkChar("AllTimesCombined"));  
  SET_STRING_ELT(sStrings, 15, Rf_mkChar("FillTauBuffer"));  
  SET_STRING_ELT(sStrings, 16, Rf_mkChar("FillYBuffer"));
  SET_STRING_ELT(sStrings, 17, Rf_mkChar("FillPiACodaBuffer"));
  SET_STRING_ELT(sStrings, 19, Rf_mkChar("FillSigCodaBuffer"));    
  SET_STRING_ELT(sStrings, 20, Rf_mkChar("FullIntegrate"));    
  SET_STRING_ELT(sStrings, 21, Rf_mkChar("AfterTauSampledSetupBuffer"));    
  SET_STRING_ELT(sStrings, 22, Rf_mkChar("SetupMTForTauii"));    
  SET_STRING_ELT(sStrings, 23, Rf_mkChar("SampleATauSpliced"));    
  SET_STRING_ELT(sStrings, 24, Rf_mkChar("FullIntegrateHowSample=0"));  
  SET_STRING_ELT(sStrings, 25, Rf_mkChar("SampleTauSplicedAfterIntegrate=0"));  
  SET_STRING_ELT(sStrings, 26, Rf_mkChar("AlsSwitchingSampler")); 
  SET_STRING_ELT(sStrings, 27, Rf_mkChar("MaximizeMe"));
  SET_STRING_ELT(sStrings, 28, Rf_mkChar("SampleTauSplicedInAlsSwitching"));      
  SET_STRING_ELT(sStrings, 29, Rf_mkChar("SampleANewTau"));  
  SET_STRING_ELT(sStrings, 30, Rf_mkChar("LastHowSample1FullIntegrate"));  
  SET_STRING_ELT(sStrings, 31, Rf_mkChar("LastHowSample3FullIntegrate"));  
  SET_STRING_ELT(sStrings, 32, Rf_mkChar("AllSampleANewTau"));  
  SET_STRING_ELT(sStrings, 33, Rf_mkChar("Add new Coordinates"));  
  SET_STRING_ELT(sStrings, 34, Rf_mkChar("NoWriteTimesCombined"));
  Rf_setAttrib(RsTimePartsList->asSexp(), R_NamesSymbol, sStrings );   
  Rf_unprotect(1);
  
  return(1);
}

void BayesSpikeCL::set_RsTimePartsList(SEXP iIn) {
  if (Rf_isNull(iIn)) {
    return;
  }
  if (!Rf_isReal(iIn)) {
    Rprintf("Hey: set_RsTimePartsList: supply a real vector.\n");
    return;
  } 
  if (Rf_length(iIn) != 20) {
    Rprintf("Hey: set_RsTimePartsList: You gave bad input length %d, not %d!\n",
      Rf_length(iIn), 20);
    return;
  }
  if (RsTimePartsList == NULL) {
    SetupRsTimePartsList();
  }
  int iti = 0;
  for (iti = 0; iti < 20; iti++) {
    REAL(RsTimePartsList->asSexp())[iti] = REAL(iIn)[iti];
  }
}
/////////////////////////////////////////////////////////////////////
//  RunAlgorithm()
//
//  Chief BayesSpike algorithm running off of a BayesSpikeCL Class
//
//  This runs gibbs sampler loop throughout data.
//
//  Note we call MRF() macro defined in BayesSpike.h
//   MRF(functionname, "characterstring", step, Loc
//  If tt = EarlyEndtt and this function is EarlyEndStep=step then Algorithm stops
//  here for model checking.
//
//  Loop through all "MRF()" calls to track Gibbs Sampler progression in order.
//
int BayesSpikeCL::RunAlgorithm() {
  int oVerbose = this->Verbose -1;
  if (oVerbose > 0) {
    Rprintf("BayesSpikeCL:  Starting Algorithm, tt = %d, Max = %d\n",
      tt, MaxGibbsIters); R_FlushConsole();
  } 
  int MaxGibbsIters = this->MaxGibbsIters;  
  if (this->CurrentMaxGibbsIters < 0) {
  } else {
    MaxGibbsIters = this->CurrentMaxGibbsIters;
  }
  
  if (this->tt >= MaxGibbsIters) {
    Rprintf("BayesSpikeCL::RunAlgorithm():: I've got this->MaxGibbsIters = %d \n", MaxGibbsIters);
    Rprintf(" ------  But tt = %d now ! \n", this->tt); R_FlushConsole();
    Rf_error("BayesSpikeCL::RunAlgorithm(): Cannot run tt=%d, past MaxGibbsIters = %d", tt, MaxGibbsIters);
  } else {
    tt = this->tt;
  }

  if (MT == NULL && !Rf_isNull(tauEndList)) {
    Rf_error("RunAlgorithm: Setup MT First!\n");
  }
  if ((tauEndList != NULL && !Rf_isNull(tauEndList)) 
    && (iFirstRandom < 0 || iFirstRandom > p) ) {
    Rf_error("RunAlgorithm: Error, iFirstRandom not setup correctly = %d\n", 
      iFirstRandom);
  }
  if (sX != RsX->asSexp()) {
    Rprintf("sX is no longer asSexp of RsX!"); R_FlushConsole();
    sX = RsX->asSexp();
  }
  //if (R_isnancpp(REAL(RsX->asSexp())[0])) {
  //   Rprintf("Error, first member of sX is a nan!"); R_FlushConsole();
  //  Rf_error("  RsX contains NANs, not good "); R_FlushConsole();
  //}
  //if (sY != RsY->asSexp()) {
  //  Rprintf("sY is no longer asSexp of RsY!"); R_FlushConsole();
  //  sY = RsY->asSexp();
  //}
  //int jti = 0;
  //for (jti = 0; jti < Rf_length(RsY->asSexp()); jti++) {
  //  if (R_isnancpp(REAL(RsY->asSexp())[jti])) {
  //    Rprintf("Error, first member of sY is a nan!"); R_FlushConsole();
  //    Rf_error("  RsY contains NANs, not good "); R_FlushConsole();
  //  }
  //}
  KillNanBeta(); 
  UpdateNonFreshXtResid();
  int SF;
  if (oVerbose > 0) {
    Rprintf("BayesSpikeCL:  Update Sigma First as Bonus\n"); R_FlushConsole();
  }
  
  /*
  if (tt <= 2) {
      Rprintf("*** BayesSpikeGibbs.cpp:::RunAlgorithm(),At Start will do a QuickCheckWeightXtX(), tt=%d\n",tt);
      R_FlushConsole();
  }
  if (iiWeight != NULL && iWeightedXtX != NULL) {
    int CountBad2 = QuickCheckWeightXtX();
    if (CountBad2 > 0) {
      Rprintf("*** BayesSpikeGibbs.cpp:::RunAlgorithm At Raw Beginning AfterReweightOnlyActive, tt=%d, we got QuickCheckWeightXtX, CountBad = %d \n", tt, CountBad2); R_FlushConsole();
      Rf_error("***  Resize QuickCheckWeightXtX Fail. \n");
    }
  }
  */
  //if (dfRobit >= 0 && AlterWeightdfRobit >= 0) {
  //  dfRobit = -1;
  //}

  if (DoLogitNonePostPreProb >= 2) {
    SEXP sRobit = R_NilValue;  
    Rf_protect(sRobit = Rf_allocVector(REALSXP, 1));  
    REAL(sRobit)[0] = 9.0;
    set_dfRobit(sRobit);
    Rf_unprotect(1);
  } else if (DoLogitNonePostPreProb == 1) {
    if (AlterInitRun == 1) {
      dfRobit = 9.0;
      AlterWeightdfRobit = -9.0;
    } else {
      dfRobit = 0.0;  AlterWeightdfRobit = 9.0;
    }
  } else if (dfRobit < 0.0 && AlterWeightdfRobit >= 0.0) {
    if (AlterInitRun == 1) {
      dfRobit = AlterWeightdfRobit;
    } else {
      dfRobit = 0.0;
    }
  } else if (dfTNoise < 0.0 && AlterWeightdfTNoise > 0.0) {
    if (AlterInitRun == 1) {
      dfTNoise = AlterWeightdfTNoise;
    } else {
      dfTNoise = 0.0;
    }
  } else if (dfRobit < 0 && DoLogitNonePostPreProb <= 0) {
    if (RSigmaPrior != NULL && !Rf_isNull(SigmaPrior) && Rf_isReal(SigmaPrior)
      && Rf_length(SigmaPrior) >=2 && REAL(SigmaPrior)[0] >= 0.0) {
      MRF(UpdateSigma, "UpdateSigma", 10,10);
    }
  }
  
  int SumDoRecord = 0;
  if (!Rf_isNull(this->DoRecord) && Rf_length(this->DoRecord) >= 1) {
    if (Rf_isInteger(this->DoRecord)) {
      for (int jjj = 0; jjj < Rf_length(this->DoRecord); jjj++) {
        SumDoRecord+= INTEGER(this->DoRecord)[jjj];
      }
    } else if (Rf_isReal(this->DoRecord)) {
       for (int jjj = 0; jjj < Rf_length(this->DoRecord); jjj++) {
        SumDoRecord+= (int) REAL(this->DoRecord)[jjj];
      }   
    }
  }
  if (SumDoRecord > 0 && (CodaTable == NULL || Rf_isNull(CodaTable) ||
    Rf_length(CodaTable) <= 0)) {
    Rprintf("BayesSpikeCL: RunAlgorithm: note, SumDoRecord=%d,  but CodaTable is not setup trigger an error!\n", SumDoRecord);
    R_FlushConsole();
    if (Rf_isNull(this->DoRecord) || Rf_length(this->DoRecord) <= 0) {
      Rprintf("  -----   DoRecord is NULL \n");
    } else if (Rf_length(this->DoRecord) <= 0) {
      Rprintf("  -----   DoRecord is length 0 \n");
    } else {
      if (Rf_isInteger(DoRecord)) {
        Rprintf("DoRecord is an Integer vector length %d \n", Rf_length(DoRecord)); R_FlushConsole();
      } else {
        Rprintf("DoRecord is a Real vector length %d. \n", Rf_length(DoRecord)); R_FlushConsole();
      }
      Rprintf("Here is DoRecord: ("); R_FlushConsole();
      for (int iti = 0; iti < Rf_length(DoRecord); iti++) {
        if (iti == Rf_length(DoRecord)-1) {
          Rprintf("%d)\n", ANINT(DoRecord, iti)); R_FlushConsole();
        } else {
         Rprintf("%d, ", ANINT(DoRecord, iti)); R_FlushConsole();
        }
      }
    }
    Rf_error("BayesSpikeCL: RunAlgorithm: Won't run if CodaTable is not setup!\n");
  }

  if (EarlyEndtt >= 0 && EarlyEndStep >= 0) {
    Rprintf("BayesSpikeCL: RunAlgorithm, note EarlyEndtt=%d, and EarlyEndStep=%d, and at start tt = %d\n",
      EarlyEndtt, EarlyEndStep, tt); R_FlushConsole();
  }
  if (oVerbose > 0) {
    Rprintf("BayesSpikeCL:  Let the Sampler Begin\n"); R_FlushConsole();
  } 
  if (EarlyEndtt == 0 && EarlyEndStep == 666 ) {
    Rprintf("BayesSpikeCL:BayesSpikeCL:BayesSpikeCL:BayesSpikeCL:BayesSpikeCL:BayesSpikeCL:BayesSpikeCL\n");
    Rprintf("BayesSpikeCL: EarlyEndtt = %d, EarlyEndStep = %d, and tt = %d, we are doing Early Quit to give MBS", 
      EarlyEndtt, EarlyEndStep, tt); 
    R_FlushConsole();
    Rf_error("BayesSpike: Deliberate RunAlgorithm Early End Error\n");
  } 
  int EveryGo = 0;
  for (; tt < MaxGibbsIters; tt++) {
    if (EveryGo >= 100) {
      Rprintf(" ** BayesSpike tt = %d/%d: Verbose = %d\n", tt, MaxGibbsIters, Verbose); R_FlushConsole();
      EveryGo = 0;
    }
    EveryGo++;
    if (TemperatureList != NULL &&  RevertTemperatureEvery > 0 && LengthTemperatureList > 1 && 
      LengthTemperatureList-1 > Tempii &&  tt > 0 && (tt % RevertTemperatureEvery) == 0) {
      if (Temperature == 1.0) {
        Temperature = TemperatureList[Tempii]; invTemperature = 1.0 / Temperature;
      } else if (fabs(Temperature - TemperatureList[Tempii+1]) < .0001) {
        Temperature = TemperatureList[Tempii]; invTemperature = 1.0 / Temperature;
      } else if (Tempii >= LengthTemperatureList-2) {
        Temperature = 1.0; invTemperature = 1.0;
      } else {
        Temperature = TemperatureList[Tempii+1]; invTemperature = 1.0 / TemperatureList[Tempii+1];
      }
    } 
    #ifdef DOTIMEH
      AllRect1 = clock();
    #endif
    if (ProbOldCoda != NULL && tt > 0 && EEMergeEvery > 0 && tt % EEMergeEvery == 0)  {
      if (Verbose >= 4) {
        Rprintf("RunAlgorithm[tt=%d]  EE Sampler Merge. \n"); R_FlushConsole();
      }
      CurrentProb = ProbPosterior();
      EESamplerMerge();  // Merge Equi-Energy operation
    }
    if (Verbose >= 3) {
      Rprintf("RunAlgorithm[tt] = %d: Start\n",tt); R_FlushConsole();
    }
    AllNewCoords = 0;
    if (tauEndList == NULL || Rf_isNull(tauEndList) || Rf_length(tauEndList) == 0 || iFirstRandom != 0) {
      MRF(SampleFixedB, "SampleFixedB", 0,0);     // SampleFixed Parameters
    }   
    if (!Rf_isNull(tauEndList)) {
      MRF(SampleNewTaus, "SampleNewTaus",1,1);    // Sample On-Off for all Taus
    }
    //if (tt <= 1) {
    //  Rprintf("*** Note RunAlgorithm(%d), after SampleNewTaus testing XtResid and OrderedActive \n", tt); R_FlushConsole();
    //}
    //TestXtResid();
    //int OSNT = TestOrderedActive();
    //if (OSNT >= 1) {
    //  Rf_error("BayesSpikeGibbs.cpp::RunAlgorithm(tt=%d), fail OSNT = %d from SampleNewTaus", tt, OSNT);
    //}
    MRF(AddAllNewCoords, "AddAllNewCoords",2,2);// Add New Coordinates to memory
    if (RPiAPrior != NULL && !Rf_isNull(PiAPrior) && 
      Rf_isReal(PiAPrior) && Rf_length(PiAPrior) >= 2) {
      MRF(UpdatePiA, "UpdatePiA",3,3);    // UpdatePiA probability of activation
    }
    
    // The following begin sampling of new sparse set Beta
    if (Verbose >= 3) {
      Rprintf("RunAlgorithm[tt] = %d: RefreshOrderedActive\n", tt);
    }
    MRFRefreshOrderedActive(1, "RefreshOrderedActive", 4, 4);
    
    //if (tt <= 1) {
    //  Rprintf("***** NOTE WE WILL RUN TESTORDEREDACTIVE AFTER EVERY RUN BEFORE");
    //  Rprintf(" YOU REMOVE THIS CODE!\n"); R_FlushConsole();
    //}
    //TestOrderedActive();
    
    MRF(PrepareForRegression, "PrepareForRegression",5,5);
    MRF(SamplePropBeta,"SamplePropBeta",6,6);
    MRF(FillsBetaFromPropBetaAndCompute,"FillsBetaFromPropBetaAndCompute",7,7);  
    
    //if (tt <= 1) {
    //  Rprintf("***** RunAlgorithm() NOTE WE WILL RUN TESTORDEREDACTIVE and XTRESID AFTER EVERY RUN BEFORE");
    //  Rprintf(" YOU REMOVE THIS CODE!\n"); R_FlushConsole();
    //}
    //int OTT = TestOrderedActive(); 
    //TestXtResid();
    //if (OTT >= 1) {
    //  Rf_error("RunAlgorithm(tt=%d), we Got TestOrderedActive returns %d, not good!\n", tt, OTT);
    //}
         
    if (dfRobit >= 0) {
      MRF(RobitReplace, "RobitReplace", 8,8);
    } else {
      if (RSigmaPrior != NULL && !Rf_isNull(SigmaPrior) &&
        Rf_isReal(SigmaPrior) && Rf_length(SigmaPrior) >= 2 && REAL(SigmaPrior)[0] >= 0.0) {
        MRF(UpdateSigma, "UpdateSigma",10,10);
      }
      if (dfTNoise > 0.0) {
      // T-Noise update Error residuals if T-Noise is going on.
        MRF(UpdateTNoise, "UpdateTNoise", 9,9);
      }
    }
    /*
    if (tt <= 2) {
      Rprintf("*** BayesSpikeGibbs.cpp:::RunAlgorithm(), Right After RobitReplace do QuickCheckWeightXtX(), tt=%d\n",tt);
      R_FlushConsole();
    }
    int CountBad = QuickCheckWeightXtX();
    if (CountBad > 0) {
      Rprintf("*** BayesSpikeGibbs.cpp:::RunAlgorithm Right After RobitReplace, tt=%d, we got QuickCheckWeightXtX, CountBad = %d \n", tt, CountBad); R_FlushConsole();
      Rf_error("***  Resize QuickCheckWeightXtX Fail. \n");
    } */   

    #ifdef DOTIMEH
      AllRect2 = clock();
      if (RsTimePartsList != NULL) { 
        REAL(RsTimePartsList->asSexp())[12] += ((double)(AllRect2-AllRect1) / CLOCKS_PER_SEC);  
      }
    #endif
    if ((NoSave == 0 || NoSave == -1) && RCodaList != NULL  && DoRecord != NULL  && !Rf_isNull(DoRecord)) {
      MRF(RecordHistory, "RecordHistory",11, 11);   // Records to CodaList if present
    }
    if (NoSave == 0  && DoSave == 1) {
      MRF(RecordCoda, "RecordCoda",12, 12); // Records to Coda Files, 
                                            //  which are compressed binaries
    }
    if (NoSave == 0 && DoSave == 1 && RsCodaiTFile == NULL) {
      if (sOnTau == NULL || Rf_isNull(sOnTau) || Rf_length(sOnTau) <= 0 ||
        tauEndList == NULL || Rf_isNull(tauEndList) || 
        Rf_length(tauEndList) <= 0) {
      
      } else {
        Rprintf("--- Hey RecordCoda we have some issues. \n");
      }
    }
    if (NoSave == 0 && DoSave == 1 && RsCodaiTFile != NULL) {
      //Rprintf("  Hey Looks like its time to Run FillTau!\n"); R_FlushConsole();
      MRF(FillTau, "FillTauBuffer", 15,15);
      if (RanFillTau != RanRecordCoda) {
        Rprintf("--- Hey This is an issue RanFillTau=%d, RanRecordCoda=%d why different? \n",
          RanFillTau, RanRecordCoda);
        Rf_error("RanFillTau=%d, RanRecordCoda=%d \n", RanFillTau, RanRecordCoda);
      }
    }  else {
      //Rprintf(" No running FillTau!\n"); R_FlushConsole();
    }
    if (NoSave == 0 && DoSave == 1 && RsPostProbBufferFile != NULL) {
      MRF(FillPostProbBuffer, "FillPostProbBuffer", 13,13);
      // Fill Posterior Probability buffer if we are saving that.
    }
    if (NoSave == 0 && DoSave == 1 && RsWeightFile != NULL) {
      MRF(FillWeightBuffer, "FillWeightBuffer", 14,14);
      // Fill Weight buffer if we are saving to that.
    }

    if (NoSave == 0 && DoSave == 1 && RsYFile != NULL) {
      MRF(FillYBuffer, "FillYBuffer", 16,16);
    }
    if (NoSave == 0 && DoSave == 1 && PiACodaBuffer != NULL) {
      MRF(FillPiACodaBuffer, "FillPiACodaBuffer", 17,17);
    }
    if (NoSave == 0 && DoSave == 1 && SigCodaBuffer != NULL) {
      MRF(FillSigCodaBuffer, "FillSigCodaBuffer", 19,19);
    }
    if (tt % ReorderIter == 0) {
      // -- Reorder the XtX matrix
      /*if (tt <= 2) {
        Rprintf("*** BayesSpikeGibbs.cpp:::RunAlgorithm(), Before Reorder(), tt=%d\n",tt);
        R_FlushConsole();
      }
      int CountBad4 = QuickCheckWeightXtX();
      if (CountBad4 > 0) {
        Rprintf("*** BayesSpikeGibbs.cpp:::RunAlgorithm() Before ReOrderXtX, tt=%d, we got QuickCheckWeightXtX, CountBad = %d, Before Reorder\n", tt, CountBad4); R_FlushConsole();
        Rf_error("***  Resize QuickCheckWeightXtX Fail. \n");
      } */
      ReorderXtX();
      /*if (tt <= 2) {
        Rprintf("*** BayesSpikeGibbs.cpp:::RunAlgorithm(), Afer ReorderXtX(), tt=%d\n",tt);
        R_FlushConsole();
      }
      int CountBad3 = QuickCheckWeightXtX();
      if (CountBad3 > 0) {
        Rprintf("*** BayesSpikeGibbs.cpp:::RunAlgorithm() After ReOrderXtX, tt=%d, we got QuickCheckWeightXtX, CountBad = %d \n", tt, CountBad3); R_FlushConsole();
        Rf_error("***  Resize QuickCheckWeightXtX Fail. \n");
      } */
    }
    if (NoSave == 0 && DoSave == 1 && RunSumBeta != NULL && tt > StartRunProbVector) {
      UpdateRunMoreZero();
      if (RunProbRegionVector != NULL && RegionWidth > 1) {
        AddToRunProbRegionVector();
      }
    }    
    if (oVerbose >= 0 && tt % PrintIter == 0) {
      Rprintf("RunAlgorithm(tt = %d) \n", tt+1); R_FlushConsole(); R_ProcessEvents();
    }
    #ifdef DOTIMEH
      AllRect2 = clock();
      if (RsTimePartsList != NULL) { 
        REAL(RsTimePartsList->asSexp())[14] +=((double)(AllRect2-AllRect1) / CLOCKS_PER_SEC);;  
      }
    #endif
    

  }
  //Rprintf("Here I am at the end of the Algorithm!\n"); R_FlushConsole();
  oVerbose = this->Verbose;
  if (oVerbose > 0) {
    Rprintf("RunAlgorithm, finished with tt = %d, Final Record\n", tt); R_FlushConsole();
  }
  if (oVerbose >= 3) {
    Rprintf("Hey we are all at the end, we should write a Tau soon!  NoSave=%d, DoSave = %d\n",
      NoSave, DoSave); R_FlushConsole();
  }
  //this->Verbose = 4;

  if (NoSave == 0  && DoSave == 1  && !Rf_isNull(sCodaIFile)) {
    if (!Rf_isString(sCodaIFile) || !Rf_isString(sCodaJFile)) {
      Rprintf("Run Algorithm inside: NoSave == 0!\n");
      Rprintf("However, we're still running WriteCoda on NULL Strings?\n");
      Rprintf("----------------------------------------\n");
      Rprintf("RunAlgorithm: We've hit a dumb error and we shouldn't have but I'll continue!\n");
      R_FlushConsole();
    } else {
      if (Verbose >= 1) {
        Rprintf("RunAlgorithm(tt=%d) about to WriteCoda. \n", tt);
      }
      MRF(WriteCoda, "WriteCoda", 19,9);
    }
  }
  if (NoSave == 0 && DoSave == 1 && RsCodaiTFile != NULL) {
    if (oVerbose >= 1) {
      Rprintf("At end of everything,  with OnCodaT = %d!, NewTWrite = %d, execute now!\n",
        OnCodaT, NewTWrite); R_FlushConsole();
    }
    if (sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) >= 1) {
      MRF(WriteTau, "WriteTau Buffer", 20,9);
    }
  }
  if (NoSave == 0 && DoSave == 1 && 
    (RsPostProbBufferFile != NULL || RsWeightFile != NULL || RsYFile != NULL)) {
    if (Verbose >= 1) {
      Rprintf("NoSave: About to write PI Codas for tt=%d. \n", tt); R_FlushConsole();
    }
    MRF(WritePICoda, "WritePICoda", 21, -1);
    if (CodaTable != NULL  && !Rf_isNull(CodaTable)) {
      MRF(RecordPandT, "RecordPandT", 103, -1);
    }
    if (sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) >= 1) {
      MRF(WriteTau, "WriteTau Buffer", 12, -1);
    }
    if (RsPostProbBufferFile != NULL)  {
      MRF(WritePostProbBuffer, "WritePostProbBuffer", 104, -1);
    }
    if (RsWeightFile != NULL) {
      MRF(WriteWeightBuffer, "WriteWeightBuffer", 105, -1);
    }
    if (RsYFile != NULL) {
      MRF(WriteYBuffer, "WriteYBuffer", 106, -1);
    }
  }
  if (RsCodaBetaAllDrawBuffer != NULL && LengthWrittenBetaAllDrawBuffer > 0) {
    if (oVerbose > 0) {
      Rprintf("RunAlgorithm: tt=%d, about to ClearBetaAllDrawBuffer. \n",tt); R_FlushConsole();
    }
    MRF(ClearToBetaAllDrawBuffer, "ClearToBetaAllDrawBuffer", 106, -1);
  }
  //this->Verbose = oVerbose;
  if (oVerbose > 0) {
    Rprintf("RunAlgorithm, finished with tt = %d, Returning\n", tt); R_FlushConsole();
  }
  return(1);
}

////////////////////////////////////////////////////////////////////////////////
//  int SamplePartBeta(int StF, MaxDo)
//
//   A "PartBeta" sampling algorithm older than BayesSpike 10/10/2013
//
//   If there are too many coordinates (default +400) to do matrix 
//     cholesky and inversion in the active set, then this will do
//     the matrix in parts by coordinates by coordinate/
//   This Starts at active set coordinate StF and builds a sub-matrix
//     of max size MaxDo, and then serves the regression doing cXtY in order
//   This maximization uses CopyIn2 to copy material inside.
//
//   We probably shouldn't do this anymore since it is less efficient than
//    SamplePartBeta2
int BayesSpikeCL::SamplePartBeta(int StF, int MaxDo) {
  if (StF >= NumActive)  {
    return(1);
  }
  if (Verbose >= 3) {
    Rprintf("SamplePartBeta: Initializing, StF=%d, MaxDo = %d\n", StF, MaxDo);
  }
  int GoFor = MaxDo;
  if (StF + MaxDo > NumActive) { GoFor = NumActive-StF; }
  CopyIn(StF, GoFor);
  char MUpLo = 'L';
  int Info = 0;  
  char Diag = 'N'; char Trans = 'T';
  int OnkSqP = GoFor * (GoFor+1) / 2;
  if (OnkSqP > MaxLargeSizeSquares*(MaxLargeSizeSquares+1)/2) {
    Rprintf(" --- Uh Oh BayesSpikeGibbs.cpp, SamplePartBeta: GoFor=%d, OnkSqP=%d, MaxLargeSizeSquares=%d, not good!\n",
      GoFor, OnkSqP, MaxLargeSizeSquares); R_FlushConsole();
    Rf_error(" --- Error on SamplePartBeta \n");
  }
  if (rIChol == NULL) {
    Rf_error("SamplePartBeta: We Shouldn't run this. \n"); R_FlushConsole();
  }
  int One = 1;
  int jj;
  double OneD = 1.0;   double ZeroD = 0.0; int PickMe;
  if (GoFor > 1) {
    F77_CALL(dpptrf)(&MUpLo, &GoFor, rIChol, &Info);
  } else {
    rIChol[0] = sqrt(rIChol[0]);  Info = 0;
  }  
  if (Info < 0) {
    if (GoFor == 1) {
      Rprintf("SamplePartBeta: No way, GoFor == 1, still no invert!");
      Rf_error("Take a look at quantities");
    }
    int AstF = StF;
    int NSub = (int) ceil(GoFor / 2);
    if (NSub <= 0) { NSub = 1; }
    int Good = 1;
    while(AstF < StF +GoFor) {
      if (AstF + NSub > StF+GoFor) {
        NSub = StF+GoFor-AstF;
      }
      JustDidPartBeta = 1;
      Good = SamplePartBeta(AstF, NSub);
      if (Good < 0) {
        Rprintf("BayesSpike:SamplePartBeta:  Impossible at AstF = %d, NSub = %d", 
          AstF, NSub); R_FlushConsole();
        Rf_error("Take a look at quantiles");
      }
      AstF += NSub;
    }
    return(1);
  }
  if (GoFor > 1) {
    F77_CALL(dcopy)(&OnkSqP, rIChol, &One, rI, &One);
    F77_CALL(dpptri)(&MUpLo, &GoFor, rI, &Info);
    if (Info < 0) {
      Rprintf("Error: rIChol as input had an illegal value \n"); R_FlushConsole();
    } else if (Info > 0) {
      Rprintf("Sample Small Beta dpptri error, not positive definite \n"); R_FlushConsole();
      Rf_error("Temporary Freedom");
    }
    F77_CALL(dtptri)(&MUpLo, &Diag, &GoFor, rIChol, &Info);
  } else {
    rI[0] =  1.0/ (rIChol[0] * rIChol[0]);
    rIChol[0] =  1.0 / rIChol[0];
    Info = (int) 0;
  }

  // Q = LL'
  // Q^(-1) = L'^(-1) %*% L^(-1)
  double SqrtScal = sqrt(REAL(sOnSigma)[0]);
  int ii;
  if (smXtY == NULL) {
    RMemGetD(smXtY, "smXtY", OnKappaMem);
  }
  JustDidPartBeta = 0;
  if (TypePrior != 3) {
    for (ii = 0; ii < GoFor; ii++) { PropBeta[StF + ii] = Rf_rnorm(0.0,1.0); }
    if (Temperature != 1.0) {
      double sQ = sqrt(Temperature);
      F77_CALL(dscal)(&GoFor, &sQ, PropBeta + StF, &One);
    }
    F77_CALL(dscal)(&GoFor, &SqrtScal, PropBeta + StF, &One);
    F77_CALL(dtpmv)(&MUpLo, &Trans, &Diag, &GoFor, rIChol, 
      PropBeta + StF, &One);
    F77_CALL(dspmv)(&MUpLo, &GoFor, &OneD, rI, smXtY, 
      &One, &OneD, PropBeta + StF , &One);
  } else {
    F77_CALL(dspmv)(&MUpLo, &GoFor, &OneD, rI, smXtY, 
      &One, &ZeroD, PropBeta+StF, &One);
    PickMe = 0; double Low = 0;  double High = 0;
    double TFL = REAL(tauFixed)[0]; double TFU = REAL(tauFixed)[p];
    double Z=0.0;
    for (ii = StF; ii < NumActive; ii++) {
      if (Rf_length(tauFixed) >= p * 2) {
        TFL = REAL(tauFixed)[OrderedActive[ii]];
        TFU = REAL(tauFixed)[OrderedActive[ii]];
      }
      //    TFL =  PBo + FC * Low <  PBo + FC * High = TFU
      if (Temperature == 1.0) {
        Low = (TFL - PropBeta[ii]) / (SqrtScal *rIChol[PickMe]);
        High = (TFU - PropBeta[ii]) / (SqrtScal*rIChol[PickMe]);
        Z = rTruncNorm2(Low, High);
        for (jj = ii; jj < GoFor; jj++) {
          PropBeta[jj] += Z * SqrtScal * rIChol[PickMe];  PickMe++;
        }
      } else {
        double SqT = sqrt(Temperature);
        Low = (TFL - PropBeta[ii]) / (SqT * SqrtScal * rIChol[PickMe]);
        High = (TFU - PropBeta[ii]) / (SqT * SqrtScal * rIChol[PickMe]);
        Z = rTruncNorm2(Low, High);
        for (jj = ii; jj < GoFor; jj++) {
          PropBeta[jj] += Z * SqrtScal * SqT * rIChol[PickMe];  PickMe++;
        }  
      }
    }    
  } 

  return(1);
}


////////////////////////////////////////////////////////////////////////////////
//  SamplePartBeta2(int StF, MaxDo)
//
//   A "PartBeta" sampling algorithm new to BayesSpike 10/10/2013
//
//   If there are too many coordinates (default +400) to do matrix 
//     cholesky and inversion in the active set, then this will do
//     the matrix in parts by coordinates by coordinate/
//   This Starts at active set coordinate StF and builds a sub-matrix
//     of max size MaxDo, and then serves the regression keeping the
//     "cXtY" partial XtResid = t(X) %*% (Y-X %*% Beta)
//   This maximization uses CopyIn2 to copy material inside.
//
//   This works by sampling some blocks Beta by the conditional posterior given other Beta
//   We continue until all Beta have received at least an update.
int BayesSpikeCL::SamplePartBeta2(int StF, int MaxDo) {
  if (JustDidPartBeta < 0) {
    JustDidPartBeta = 1;
  }
  if (StF >= NumActive)  {
    return(1);
  }
  if (StF < 0) {
    Rf_error("BayesSpikeGibbs.cpp:::SamplePartBeta2(tt=%d): Won't do if StF given =%d\n", tt, StF);
  }
  if (Verbose >= 3) {
    Rprintf("SamplePartBeta2: Initializing, StF=%d, MaxDo = %d\n", StF, MaxDo);
  }
  int GoFor = MaxDo;
  if (StF + MaxDo > NumActive) { GoFor = NumActive-StF; }
  if (GoFor <= 0) {
    return(0);
  }
  if (rIChol == NULL) {
    RMemGetD(rIChol, "rIChol", MaxLargeSizeSquares*(MaxLargeSizeSquares+1)/2); 
  }
  if (rSmallXtX == NULL) {
    RMemGetD(rSmallXtX, "rSmallXtX", MaxLargeSizeSquares*(MaxLargeSizeSquares+1)/2); 
  }
  
  // CopyIn2, copies information starting at StF of length GoFor into
  //          Both smXtY (from cXtY) and rIChol (from pXtX)
  CopyIn2(StF, GoFor);
  if (Verbose >= 3) {
    Rprintf("SamplePartBeta2: CopyIn2 is finished.\n"); R_FlushConsole();
  }
  char MUpLo = 'L';
  int Info = 0;  
  char Diag = 'N'; char Trans = 'T';
  int OnkSqP = GoFor * (GoFor+1) / 2;
  int One = 1;
  int jj;
  double OneD = 1.0;   double ZeroD = 0.0; int PickMe;

  if (GoFor > MaxLargeSizeSquares) {
    Rprintf(" --- BayesSpikeGibbs.cpp:: SamplePartBeta2: No, this doesn't work GoFor=%d, MaxLargeSizeSquares=%d\n",
      GoFor, MaxLargeSizeSquares); R_FlushConsole();
    Rf_error(" --- BayesSpikeGibbs.cpp:: SamplePartBeta2: this is the a GoFor SamplePartBeta2 error \n");
  }
  if (MaxLargeSizeSquares <= 0) {
    Rf_error(" --- BayesSpikeGibbs.cpp:: MaxLargeSizeSquares = %d \n", MaxLargeSizeSquares); 
  }
  LastStF = StF;  LastGoFor = GoFor;

  // SamplePartBeta2()
  //
  // dpptrf, LAPACK call, cholesky of packed rIChol
  Info = 0;
  if (GoFor > 1) {
    F77_CALL(dpptrf)(&MUpLo, &GoFor, rIChol, &Info);
  } else {
    rIChol[0] = sqrt(rIChol[0]);  Info = 0;
  }  
  if (Verbose >= 3) {
    Rprintf(" --- BayesSpikeGibbs.cpp:: SamplePartBeta2: dpptrf run for dimension GoFor =%d, Info = %d\n", 
      GoFor, Info); R_FlushConsole();
    if (Info == 0) {
      Rprintf(" --- BayesSpikeGibbs.cpp:: My guess is that Info worked out well!\n", Info);
    }
  }
  One = 1;
  if (Info < 0) {
    if (Verbose >= -1) {
      Rprintf(" -- BayesSpikeGibbs::SamplePartBeta(StF=%d, GoFor =%d), we have to half again because Info=%d!\n",
        StF, GoFor, Info); R_FlushConsole();
    }
    JustDidPartBeta++;
    if (GoFor == 1) {
      Rprintf("SamplePartBeta2: No way, GoFor == 1, still no invert!\n"); R_FlushConsole();
      Rf_error("Take a look at quantities");
    }
    int AstF = StF;
    int NSub = (int) ceil(GoFor / 2);
    if (NSub <= 0) { NSub = 1; }
    int Good = 1;
    if (Verbose >= 3) {
      Rprintf("SamplePartBeta2: Oops dpptrf had info = %d, doing a partial inverse now with dim %d instead of %d",
        Info, NSub, GoFor); R_FlushConsole();
    }
    while(AstF < StF +GoFor) {
      if (AstF + NSub > StF+GoFor) {
        NSub = StF+GoFor-AstF;
      }
      Good = SamplePartBeta2(AstF, NSub);
      if (Good < 0) {
        Rprintf("BayesSpike:SamplePartBeta2:  Impossible at AstF = %d, NSub = %d", 
          AstF, NSub); R_FlushConsole();
        Rf_error("Take a look at quantiles");
      }
      AstF += NSub;
    }
    return(1);
  }
  if (rI == NULL) {
    RMemGetD(rI, "rI", MaxLargeSizeSquares*(MaxLargeSizeSquares+1)/2); 
  }
  if (GoFor > 1) {
    if (Verbose >= 3)  {
      Rprintf("BayesSpike:SamplePartBeta2: we're copying succesful dpptrf rIChol.\n"); 
        R_FlushConsole();
    }
    F77_CALL(dcopy)(&OnkSqP, rIChol, &One, rI, &One);
    if (Verbose >= 3) {
      Rprintf("BayesSpike:SamplePartBeta2: we're dpptri. rI from cholesky to inverse\n");
        R_FlushConsole();
    }
    F77_CALL(dpptri)(&MUpLo, &GoFor, rI, &Info);
    if (Info < 0) {
      Rprintf("Error: rIChol as input had an illegal value, Info=%d \n", Info); 
      R_FlushConsole();
    } else if (Info > 0) {
      Rprintf("Sample Small Beta dpptri error, Info=%d, not positive definite \n", Info); 
      R_FlushConsole();
      Rf_error("Temporary Freedom");
    }
    if (Verbose >= 3) {
      Rprintf("BayesSpike:SamplePartBeta2: we're dtptri. Cholesky to Cholesy inverse.\n"); R_FlushConsole();
    }
    F77_CALL(dtptri)(&MUpLo, &Diag, &GoFor, rIChol, &Info);
  } else {
    if (Verbose >= 3) {
      Rprintf("BayesSpike: SamplePartBeta2: Do inverse and multiplication for one value. \n"); 
        R_FlushConsole();
    }
    rI[0] =  1.0/ (rIChol[0] * rIChol[0]);
    rIChol[0] =  1.0 / rIChol[0];
    Info = (int) 0;
  }

  int sVerbose = Verbose;
  //sVerbose = 3;
  // Q = LL'
  // Q^(-1) = L'^(-1) %*% L^(-1)
  double SqrtScal = sqrt(REAL(sOnSigma)[0]);
  int ii;
  if (Verbose >= 3) {
    Rprintf("BayesSpike: SamplePartBeta2: About to fill PartBeta, GoFor=%d, StF=%d\n", 
      GoFor, StF); R_FlushConsole();
  }
  if (TypePrior != 3) {
    One = 1;
    if (GoFor + StF > NumActive) {
      if (Verbose >= 2) {
        Rprintf("TypePrior=%d, Warning Bad, GoFor = %d, StF=%d, NumActive=%d, OnKappaS=%d/%d, not goot tt=%d!\n",
         TypePrior, GoFor, StF, NumActive, OnKappaS, OnKappaMem, tt); R_FlushConsole();
      }
    }
    if (GoFor > MaxLargeSizeSquares) {
      Rprintf("Oh No! Bad, TypePrior = %d, GoFor = %d, StF=%d, NumActive=%d, OnKappaS=%d/%d and LargeSizeSquares=%d! tt =%d",
        TypePrior, GoFor, StF, NumActive, OnKappaS, OnKappaMem, MaxLargeSizeSquares);
      R_FlushConsole();
      Rf_error("Error in SamplePartBeta2, GoFor=%d > LargeSizeSquares=%d!\n",
        GoFor, MaxLargeSizeSquares);  
    }
    for (ii = 0; ii < GoFor; ii++) { PropBeta[StF + ii] = Rf_rnorm(0.0,1.0); }
    if (sVerbose >= 3) {
      Rprintf("BayesSpike:[tt=%d], SamplePartBeta for TypePrior=%d, going to dscal\n", tt, TypePrior);
    }
    if (Temperature != 1.0) {
      double sQ = sqrt(Temperature) * SqrtScal;
      if (GoFor > 1) {
        F77_CALL(dscal)(&GoFor, &sQ, PropBeta + StF, &One);
      } else {
        PropBeta[StF] = PropBeta[StF] * sQ;
      }
    }  else {
      if (GoFor > 1) {
        F77_CALL(dscal)(&GoFor,&SqrtScal, PropBeta + StF, &One);
      } else {
        PropBeta[StF] = PropBeta[StF] * SqrtScal;
      }
    }
    if (sVerbose >= 3) {
      Rprintf("BayesSpike:[tt=%d] SamplePartBeta2: About to dtpmv size = %d,%d against PropBeta+StF=%d\n",
        tt, GoFor, GoFor, StF); R_FlushConsole();
    }
    if (GoFor > 1) {
      F77_CALL(dtpmv)(&MUpLo,&Trans,&Diag,&GoFor,rIChol,PropBeta + StF,&One);
    } else {
      PropBeta[StF] = rIChol[0] * PropBeta[StF];
    }
    if (sVerbose >= 3) {
      Rprintf("BayesSpike: SamplePartBeta2: About to dpsmv\n"); R_FlushConsole();
    }
    if (DontPartBetaNoise == 1) {
      if (GoFor > 1) {
        F77_CALL(dscal)(&GoFor, &ZeroD, PropBeta+StF, &One);
      } else {
        PropBeta[StF] = 0.0;
      }
    }
    if (GoFor > 1) {
      F77_CALL(dspmv)(&MUpLo, &GoFor, &OneD, rI, smXtY, 
        &One, &OneD, PropBeta + StF , &One);
    } else {
      PropBeta[StF] = PropBeta[StF] + rI[0] * smXtY[0];
    }
    if (sVerbose >= 3) {
      Rprintf("BayesSpike: SamplePartBeta2: Completed dpsmv\n"); 
      R_FlushConsole();
    }
  } else {
    F77_CALL(dspmv)(&MUpLo, &GoFor, &OneD, rI, smXtY, 
      &One, &ZeroD, PropBeta+StF, &One);
    PickMe = 0; double Low = 0;  double High = 0;
    double TFL = REAL(tauFixed)[0]; double TFU = REAL(tauFixed)[p];
    double Z=0.0;
    for (ii = StF; ii < GoFor+StF; ii++) {
      if (Rf_length(tauFixed) >= p * 2) {
        TFL = REAL(tauFixed)[OrderedActive[ii]];
        TFU = REAL(tauFixed)[OrderedActive[ii]];
      }
      //    TFL =  PBo + FC * Low <  PBo + FC * High = TFU
      if (Temperature == 1.0) {
      Low = (TFL - PropBeta[ii]) / (SqrtScal *rIChol[PickMe]);
      High = (TFU - PropBeta[ii]) / (SqrtScal*rIChol[PickMe]);
      Z = rTruncNorm2(Low, High);
      for (jj = ii; jj < GoFor; jj++) {
        PropBeta[jj] += Z * SqrtScal * rIChol[PickMe];  PickMe++;
      }
      } else {
        double SqT = sqrt(Temperature);
        Low = (TFL - PropBeta[ii]) / (SqT*SqrtScal *rIChol[PickMe]);
        High = (TFU - PropBeta[ii]) / (SqT*SqrtScal*rIChol[PickMe]);
        Z = rTruncNorm2(Low, High);
        for (jj = ii; jj < GoFor; jj++) {
          PropBeta[jj] += Z * SqrtScal * SqT* rIChol[PickMe];  PickMe++;
        }  
      }
    }    
  }
  int Onjk = 0;   double DiffBeta;
  if (GoFor + StF < NumActive) {
    for (jj = 0; jj < GoFor; jj++) {
      if (XLC[OrderedActive[StF+jj]] < 0) {
        Rprintf("Uh Oh StF=%d, jj=%d, For OrderedCActive[%d] = %d, XLC[%d] = %d\n",
          StF, jj, StF+jj, OrderedActive[StF+jj], XLC[OrderedActive[StF+jj]]); 
        Rf_error("This ain't going to work!\n");
      }
    }
    for (jj = 0; jj < GoFor; jj++) {
      DiffBeta = PropBeta[StF+jj] - REAL(sBeta)[OrderedActive[StF+jj]];
      if (XLC[OrderedActive[StF+jj]] < 0 || XLC[OrderedActive[StF+jj]] >= OnKappaS) {
        Rprintf("BayesSpikeGibbs.cpp:: Error XLC[OrderedActive[%d+%d=%d]=%d] = %d, OnKappaS=%d/%d!\n",
          StF, jj, StF+jj, OrderedActive[StF+jj], XLC[OrderedActive[StF+jj]],
          OnKappaS, OnKappaMem); R_FlushConsole();
        Rf_error("BayesSpikeGibbs.cpp:: Error XLC[OrderedActive[%d+%d=%d]=%d] = %d, OnKappaS=%d/%d!\n",
          StF, jj, StF+jj, OrderedActive[StF+jj], XLC[OrderedActive[StF+jj]],
          OnKappaS, OnKappaMem);
      }
      if (GoFor + StF > OnKappaMem) {
        Rf_error("BayesSpikeGibbs.cpp:: Error, GoFor=%d, StF=%d, OnKappaMem=%d\n",
          GoFor, StF, OnKappaMem);
      }
      if (NumActive > OnKappaMem) {
        Rf_error("BayesSpikeGibbs.cpp:: Error, GoFor=%d, StF=%d, NumActive=%d, OnKappaMem=%d\n",
          GoFor, StF, NumActive, OnKappaMem);
      }
      for(Onjk = GoFor + StF; Onjk < NumActive; Onjk++) {
        if (pXtX[XLC[OrderedActive[StF+jj]]] == NULL) {
          Rprintf(" --- BayesSpikeGibbs.cpp::SamplePartBeta2(tt=%d) DiffBeta Error on smXtY GoFor=%d, StF=%d, Onjk=%d\n",
            tt, GoFor, StF, Onjk);
          Rprintf(" --- OrderedActive[%d+%d] = %d, XLC[%d]=%d \n",
            StF, jj, OrderedActive[StF+jj], OrderedActive[StF+jj],
            XLC[OrderedActive[StF+jj]]); R_FlushConsole();
          Rf_error(" --- BayesSpikeGibbs.cpp::SamplePartBeta2() NULL pXtX is Not good.");
        }
        cXtY[Onjk] -= DiffBeta * 
          *(pXtX[XLC[OrderedActive[StF+jj]]] + OrderedActive[Onjk]);  
      }
    } 
  }
  return(1);
}


////////////////////////////////////////////////////////////////////////////////
// ProbOldCoda, ICoda
//  
//   The posterior probability of draws at higher temperatures
//  allows us to determine Merge samples.  By sorting by increasing energy
//  we are able to identify all samples from the previous temperature within
//  the same energy slice as our target draw.  
//
int BayesSpikeCL::LoadProbOldCodaICoda(int DoEEProbSort, int StartIter, int LastIter) {
  FILE *fPOldCoda = NULL;
  FILE *fOldILocCoda = NULL;
  if (Verbose > 2) {
    Rprintf("LoadProbOldCodaICoda: About to Start!"); R_FlushConsole();
  }
  if (RsOldCodaILocFile == NULL || Rf_isNull(RsOldCodaILocFile->asSexp())) {
    Rprintf("Error:RsOldCodaILocFile is NULL\n"); R_FlushConsole();
    Rf_error("LoadProbOldCodaICodaError.");
  }
  if (!Rf_isString(RsOldCodaILocFile->asSexp())) {
    Rf_error("LoadProbOldCodaICoda:  Error, RsOldCodaILoc has no string \n");
    R_FlushConsole();
  }
  if (RsOldCodaProbFile == NULL || Rf_isNull(RsOldCodaProbFile->asSexp())) {
    Rprintf("LoadProbOldCodaICoda: Error: RsOldCodaProbFile is NULL\n"); R_FlushConsole();
    Rf_error("LoadProbOldCodaICoda:  Error.");
  }
  if (!Rf_isString(RsOldCodaProbFile->asSexp())) {
    Rf_error("LoadProbOldCodaICoda:  Error, RsOldCodaProbFile has no string \n");
    R_FlushConsole();
  }
    if (RsCodaOldIFile == NULL || Rf_isNull(RsCodaOldIFile->asSexp())) {
    Rprintf("LoadProbOldCodaICoda: Error: RsCodaOldIFile is NULL\n"); R_FlushConsole();
    Rf_error("LoadProbOldCodaICodaError.");
  }
  if (!Rf_isString(RsOldCodaProbFile->asSexp())) {
    Rf_error("LoadProbOldCodaICoda:  Error, RsCodaOldIFile has no string \n");
    R_FlushConsole();
  }

 
  if (Verbose > 1) {
    Rprintf("LoadProbOldCodaICoda: fOldILocCoda, fPOldCoda first measure size "); R_FlushConsole();
  }


  //FILE *fPOldICoda;
  long file_size;
  //char *buffer;
  struct stat stbuf;  
  
  ////////////////////////////////////////////////////////////////////////////
  //  Note: Use of open(), fdopen, fstat
  //    are methods of reading length of files
  //    this is somewhat inconvenient, but safer than fseek
  fd = -1;
  fd = open(CHAR(STRING_ELT(RsCodaOldIFile->asSexp(),0)), O_RDONLY);
    TotalOpenedFiles++;
  if (fd == -1) {
    Rf_error("Setup fPOldILocCoda: We can't open fPOldICoda"); 
  } 
  //fPOldICoda = fdopen(fd, "rb");
  if (fstat(fd, &stbuf) == -1) {
    Rf_error("Setup fPOldILocCoda: could not use fstat");
  }
  file_size = stbuf.st_size;
  MaxLengthOldIDFiles = file_size / sizeof(long int);
  //fdclose(fPOldICoda);
  close(fd);  TotalClosedFiles++;
  
  struct stat stbuf1;
  fd1 = -1; fd2 = -1;
  fd1 = open(CHAR(STRING_ELT(RsOldCodaILocFile->asSexp(), 0)), O_RDONLY);
  TotalOpenedFiles++;
  if (fd1 == -1) {
    Rprintf("Error: LoadProbOldCodaICoda: Cannot Open: fPOldILocCoda = %s.\n",
      CHAR(STRING_ELT(RsOldCodaILocFile->asSexp(), 0)));
    Rf_error("Setup fPOldILocCoda: We can't open fPOldILocCoda"); 
  }
  //fOldILocCoda = fdopen(fd1, "rb");
  if (fstat(fd1, &stbuf1) == -1) {
    Rprintf("Error: LoadProbOldCodaICoda: FStat Fail on fPOldILocCoda = %s.\n",
      CHAR(STRING_ELT(RsOldCodaILocFile->asSexp(), 0)));
    Rf_error("Setup fPOldILocCoda: could not use fstat");
  } 
  file_size = stbuf1.st_size;
  long int lSizeI = file_size / sizeof(long int);
  close(fd1);  TotalClosedFiles++;
  
  struct stat stbuf2;
  fd2 = open(CHAR(STRING_ELT(RsOldCodaProbFile->asSexp(), 0)), O_RDONLY);
  TotalOpenedFiles++;
  if (fd2 == -1) {
    Rprintf("Error: LoadProbOldCodaICoda: Cannot Open: fPOldProbCoda = %s.\n",
      CHAR(STRING_ELT(RsOldCodaProbFile->asSexp(), 0)));
    Rf_error("Setup fPOldILocCoda: fd2 error, We can't open fPOldProbCoda"); 
  }
  //fPOldCoda = fdopen(fd2, "rb");
  if (fstat(fd2, &stbuf2) == -1) {
    Rprintf("Error: LoadProbOldCodaICoda: Cannot stat: fPOldProbCoda = %s.\n",
      CHAR(STRING_ELT(RsOldCodaProbFile->asSexp(), 0)));
    Rf_error("Setup fPOldILocCoda: could not use fstat on ProbOldCoda");
  } 
  file_size = stbuf2.st_size;
  long int lSizeP = file_size / sizeof(double);
  close(fd2); TotalClosedFiles++;
  
  fOldILocCoda = fopen(CHAR(STRING_ELT(RsOldCodaILocFile->asSexp(), 0)), "rb");
  fPOldCoda = fopen(CHAR(STRING_ELT(RsOldCodaProbFile->asSexp(), 0)), "rb");
  if (fOldILocCoda != NULL) { TotalOpenedFiles++; }
  if (fPOldCoda != NULL) { TotalOpenedFiles++; }
    
  if (lSizeI != lSizeP || Verbose > 1) {
    Rprintf("LoadProbOldCodaICoda: Size Error with fPOldProbCoda = %s.\n",
      CHAR(STRING_ELT(RsOldCodaProbFile->asSexp(), 0)));
    Rprintf("LoadOldLocandProbfiles, lSizeP = %d, but lSizeI = %d \n",
      lSizeP, lSizeI); R_FlushConsole();
  }

  if (LastIter <= 0) {
    LastIter = lSizeI-1;
  }
  if (LastIter > lSizeI-1) { LastIter = lSizeI-1; }
  if (LastIter > lSizeP-1) { LastIter = lSizeP-1; }
  if (StartIter <= 0 || StartIter >= LastIter) {
    StartIter = 0;
  }
  if (fOldILocCoda == NULL) {
    if (fPOldCoda != NULL) { fclose(fPOldCoda); fPOldCoda = NULL; TotalClosedFiles++; }
    Rf_error("LoadProbOldCodaICoda: Failed to open RsOldCodaILocFile with old I Location Info!");
  }  
  if (fPOldCoda == NULL) {
    if (fOldILocCoda != NULL) { fclose(fOldILocCoda); fOldILocCoda = NULL; TotalClosedFiles++;}
    Rf_error("LoadProbOldCodaICoda: Failed to open RsOldCodaProbFile with old Prob Location Info!");
  }
  if (Verbose > 0) {
    Rprintf("LoadOldLocAndProbFiles:  FFreeing materials \n");  R_FlushConsole();
  }
  FFree(ProbOldCoda, "ProbOldCoda");  FFree(IOldCoda, "IOldCoda");
  ProbOldCoda = (double*) Calloc(LastIter - StartIter +2, double);
  IOldCoda = (long int*) Calloc(LastIter - StartIter +2, long int);
  //int ii;
  int result;
  fseek(fPOldCoda, StartIter * sizeof(double),  SEEK_SET);
  fseek(fOldILocCoda, StartIter * sizeof(long int), SEEK_SET);
  if (Verbose > 1) {
    Rprintf("LoadOldLocAndProbFiles:  Loading IOldCoda \n");
    R_FlushConsole();
  }
  result = fread(IOldCoda, sizeof(int), LastIter - StartIter+1, fOldILocCoda);
  if (Verbose > 1) {
    Rprintf("LoadOldLocAndProbFiles:  Loading ProbOldCoda \n");
    R_FlushConsole();
  }
  result = fread(ProbOldCoda, sizeof(double), LastIter - StartIter+1, fPOldCoda); 
  if (Verbose >= 5) {
    Rprintf("LoadOldLocAndProbFiles: on fread result is %d \n", result); R_FlushConsole();
  }
  if (Verbose >= 2) {
    Rprintf("LoadOldLocAndProbFiles: Readin finished, doing Read In!\n");
    R_FlushConsole();
  }
  fclose(fOldILocCoda); fOldILocCoda = NULL; fclose(fPOldCoda); fPOldCoda = NULL;
  TotalClosedFiles+=2;
  if (Verbose > 1) {
    Rprintf("LoadOldLocAndProbFiles:  Close and Go Home\n"); R_FlushConsole();
  }
  LengthOldCoda = LastIter - StartIter +1;
  if (LengthOldCoda > MaxGibbsIters) {
    Rf_error("Error: LengthOldCoda = %d is greater than MaxGibbsIters = %d, not good\n",
      LengthOldCoda, MaxGibbsIters);
  }
  if (DoEEProbSort > 0 || DoingEEProbSort > 0) {
    DoingEEProbSort = 1;
    if (Verbose >= 3) {
      Rprintf("LoadOldLocAndProbFiles: Doing a Sort of the loaded in File!\n");
      R_FlushConsole();
    }
    int *MyNewSort = HeapSortAll(LengthOldCoda, ProbOldCoda);
    bufferILocType *NewIOldCoda = (bufferILocType*) Calloc(LengthOldCoda, bufferILocType);
    double *NewProbCoda = (double*) Calloc(LengthOldCoda, double);
    int itt = 0;
    for (itt =0; itt < LengthOldCoda; itt++) {
      if (MyNewSort[itt] <  0 || MyNewSort[itt] >= LengthOldCoda) {
        Rprintf("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE\n");
        Rprintf("EE Error, MyNewSort has members itt=%d/%d of %d relative to %d \n",
          itt, LengthOldCoda, itt, LengthOldCoda);
        Rprintf("EE Looks like sort Failure. \n");
        FFree(MyNewSort, "MyNewSort");  FFree(NewIOldCoda, "NewIOldCoda");
        FFree(NewProbCoda, "NewProbCoda");
        FFree(IOldCoda, "IOldCoda");  FFree(ProbOldCoda, "ProbOldCoda");
        Rf_error("EE: MyNewSort in gen IOldCoda/ProbOld coda is invalid. \n");
      }
      NewIOldCoda[itt] = IOldCoda[MyNewSort[itt]];
      NewProbCoda[itt] = ProbOldCoda[MyNewSort[itt]];
    }
    FFree(MyNewSort, "MyNewSort");
    FFree(IOldCoda, "IOldCoda"); IOldCoda = NewIOldCoda;
    FFree(ProbOldCoda, "ProbOldCoda"); ProbOldCoda = NewProbCoda;
  } else {
    DoingEEProbSort = 0;
  }
  if (Verbose >= 2) {
    Rprintf("LoadOldLocAndProbFiles: Successful!\n");R_FlushConsole();
  }
  return(1);

}

///////////////////////////////////////////////////////////////////////////////
// WriteProstProbBuffer()
//                
//   Write Gibbs Samples from PostProbBuffer to disk.
int BayesSpikeCL::WritePostProbBuffer() {
  if (NoSave != 0) { return(-1); }
  FILE *PostProbFile = NULL;
  if (LengthWrittenPostProb <= 0 ) {
    return(0);
  }
  if (Verbose >= 5) {
    Rprintf("Starting Writing PostProbBuffer tt=%d, =LengthWrittenPostProb=%d/%d, Total Written now = %d, LengthTotalWrittenPostProb\n",
      tt, LengthWrittenPostProb, LengthPostProbCodaBuffer); R_FlushConsole();
  }
  if (RsPostProbBufferFile == NULL || Rf_isNull(RsPostProbBufferFile->asSexp()) ||
    !Rf_isString(RsPostProbBufferFile->asSexp())) {
    Rprintf("WritePostProbBuffer: Why, RsPostProbBufferFile is not a string!\n");
    if (RsPostProbBufferFile == NULL) {
      Rprintf("WritePostProbBuffer: RsPostProbBufferFile is Null Pointer\n");
    } else if (Rf_isNull(RsPostProbBufferFile->asSexp())) {
      Rprintf("WritePostProbBuffer: RsPostProbBufferFile is Null SEXP\n");
    } else if (!Rf_isString(RsPostProbBufferFile->asSexp())) {
      Rprintf("WritePostProbBuffer: RsPostProbBufferFile is not a String Object.\n");
    }
    R_FlushConsole(); return(-1);
  }
  if (Verbose >= 2) {
    printf("-- Note WritePostProbBuffer: we have written %d to %s, add %d \n",
      LengthTotalWrittenPostProb, CHAR(STRING_ELT(RsPostProbBufferFile->asSexp(), 0)),
      LengthWrittenPostProb); R_FlushConsole();
  }
  if (NewPostProbWrite == 1)  {
    PostProbFile = fopen(CHAR(STRING_ELT(RsPostProbBufferFile->asSexp(), 0)), "wb");
    NewPostProbWrite = 0;
    if (PostProbFile == NULL) {
      Rprintf("****************************************************************\n");
      Rprintf("** WritePostProbBuffer: Error Trying to open a new version of : %s \n",
        CHAR(STRING_ELT(RsPostProbBufferFile->asSexp(), 0))); R_FlushConsole();
      Rprintf("** Give up and Quit! \n"); R_FlushConsole();
      Rprintf("** Clearing PostProbBuffer ! \n"); R_FlushConsole();
      DDelete(RsPostProbBufferFile, "RsPostProbBufferFile"); RsPostProbBufferFile = NULL;
      FFree(PostProbCodaBuffer, "PostProbCodaBuffer"); PostProbCodaBuffer = NULL;
      LengthWrittenPostProb = 0;  LengthPostProb = 0; LengthTotalWrittenPostProb = 0;
      Rprintf("** Cleared PostProbBuffer from sweep! \n");
      return(-1);
    }
    TotalOpenedFiles++;
  } else {
    PostProbFile = fopen(CHAR(STRING_ELT(RsPostProbBufferFile->asSexp(), 0)), "ab");
    // int fSeekFail = fseek (PostProbFile , ((long) sizeof(double) * 
    //  (long) LengthTotalWrittenPostProb * LengthProbCoda), SEEK_SET);
    if (PostProbFile == NULL) {
      Rprintf("** WritePostProbBuffer: Error to append new version of : %s \n",
        CHAR(STRING_ELT(RsPostProbBufferFile->asSexp(), 0))); R_FlushConsole();
      Rprintf("** Give up and Quit! \n"); R_FlushConsole();
      Rprintf("** Clearing PostProbBuffer ! \n"); R_FlushConsole();
      DDelete(RsPostProbBufferFile, "RsPostProbBufferFile"); RsPostProbBufferFile = NULL;
      FFree(PostProbCodaBuffer, "PostProbCodaBuffer"); PostProbCodaBuffer = NULL;
      LengthWrittenPostProb = 0;  LengthPostProb = 0; LengthTotalWrittenPostProb = 0;
      Rprintf("** Cleared PostProbBuffer from sweep! \n");
      return(-1);
    }
    TotalOpenedFiles++;

  }
  if (PostProbFile == NULL) {
    Rprintf("****************************************************************\n");
    Rprintf("** WritePostProbBuffer: Error opening: %s \n",
      CHAR(STRING_ELT(RsPostProbBufferFile->asSexp(), 0))); R_FlushConsole();
    Rprintf("** We give up and quit! \n");
    Rprintf("** Clearing PostProbBuffer ! \n"); R_FlushConsole();
      DDelete(RsPostProbBufferFile, "RsPostProbBufferFile"); RsPostProbBufferFile = NULL;
      FFree(PostProbCodaBuffer, "PostProbCodaBuffer"); PostProbCodaBuffer = NULL;
      LengthWrittenPostProb = 0;  LengthPostProb = 0; LengthTotalWrittenPostProb = 0;
      Rprintf("** Cleared PostProbBuffer from sweep! \n");
    return(-1);
  }
  int MyWrite = -1; 
  if (Verbose >= 3) {  
    Rprintf("writeFileCoda, writing PostProbCodaBuffer, LWPP = %d, LPC = %d \n",
      LengthWrittenPostProb, LengthPostProb); R_FlushConsole();
  }
  MyWrite = fwrite( (void*) PostProbCodaBuffer, 
    sizeof(double), LengthWrittenPostProb*LengthPostProb, PostProbFile ); 
  if (MyWrite != LengthWrittenPostProb * LengthPostProb) {
    Rprintf("WritePostProbFile: we got MyWrite = %d, but LengthWritten=%d, LengthPostProb = %d!\n",
      MyWrite, LengthWrittenPostProb, LengthPostProb);
    R_FlushConsole();
  }
  if (Verbose >= 3) {  
    Rprintf("writeFileCoda, wrote PostProbCodaBuffer, LWPP = %d, LPostProb = %d, MyWrite = %d \n",
      LengthWrittenPostProb, LengthPostProb, MyWrite); R_FlushConsole();
  }
  LengthTotalWrittenPostProb +=  LengthWrittenPostProb;
  fclose(PostProbFile);   TotalClosedFiles++;
  LengthWrittenPostProb = 0;
  if (Verbose >= 5) {
    Rprintf("Finished Writing PostProbBuffer!\n"); R_FlushConsole();
  }
  return(1);
}
int BayesSpikeCL::FillPostProbBuffer() {
  if (NoSave != 0) {
    return(-1);
  }
  if (RsPostProbBufferFile == NULL) {
    return(0);
  }
  if (Verbose >= 3) {
    Rprintf("FillPostProbBuffer:  Filling LWPP = %d/%d\n",
      LengthWrittenPostProb,LengthPostProbCodaBuffer); R_FlushConsole();
  }
  if (PostProbCodaBuffer == NULL) {
    Rf_error("FillPostProbBuffer: Hey, RsProbBufferFile is not null but unsatisfactory  PostCodaBuffer!\n");
  } else if (LengthPostProb <= 0) {
    Rf_error("FillPostProbBuffer: Hey, RsProbBufferFile is not null but  LengthPostProb = %d!\n", LengthPostProb);
  } else if  (LengthPostProbCodaBuffer <= 0) {
    Rf_error("FillPostProbBuffer: Hey, RsProbBufferFile is not null but LengthPostProbCodaBuffer = %d!\n", LengthPostProbCodaBuffer);
  }
  if (LengthPostProb  <= 0) {
    Rprintf("FillPostProbBuffer: Real Weird, LengthPostProb = %d!\n", LengthPostProb);
    R_FlushConsole();
    Rf_error("Error on Post ProbBuffer!\n");
  }
  if (LengthWrittenPostProb >= LengthPostProbCodaBuffer-1) {
    if (Verbose >= 2) {
      printf("FillPostProbBuffer: about to write now with LengthWrittenPostProb=%d. \n",
      LengthWrittenPostProb);
    }
    int Out = WritePostProbBuffer();
    if (Out < 0) {
      Rprintf("FillPostProbBuffer: ERROR, WritePostProbBuffer returns %d \n", Out);
      R_FlushConsole();
      return(Out);
    }
    LengthWrittenPostProb=0;
  }

  if (iFirstRandom != 0 && ProbFixed == NULL) {
    Rf_error("FillPostProbBuffer: Error: ProbFixed is NULL!\n");
  }
  if (sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) > 0 ) {
    if (ProbTau == NULL) {
      Rf_error("FillPostProbBuffer: ProbTau is a NULL but it shouldn't be!\n");
    }
  }
  int One = 1;
  if (sOnTau == NULL || Rf_isNull(sOnTau) || iFirstRandom < 0 || Rf_length(sOnTau) <= 0 ||
    tauEndList == NULL || Rf_isNull(tauEndList) || Rf_length(sOnTau) <= 0) {
    F77_CALL(dcopy)(&p, ProbFixed, &One, PostProbCodaBuffer +  LengthWrittenPostProb * LengthPostProb, &One);
  } else if (iFirstRandom > 0) {
    F77_CALL(dcopy)(&iFirstRandom, ProbFixed, &One, PostProbCodaBuffer +  LengthWrittenPostProb * LengthPostProb, &One);
    int lSOnTau = Rf_length(sOnTau);
    F77_CALL(dcopy)(&lSOnTau, ProbTau, &One,PostProbCodaBuffer +  LengthWrittenPostProb * LengthPostProb + iFirstRandom, &One);
  } else {
    int lSOnTau2 = Rf_length(sOnTau);
    F77_CALL(dcopy)(&lSOnTau2, ProbTau, &One,PostProbCodaBuffer +  LengthWrittenPostProb * LengthPostProb, &One);
  }
  LengthWrittenPostProb++;
  if (LengthWrittenPostProb >= LengthPostProbCodaBuffer)  {
     int Out2 = WritePostProbBuffer();
     if (Out2 < 0) {
       Rprintf("FillPostProbBuffer: ERROR, WritePostProbBuffer returns %d \n", Out2);
       return(Out2);
     }
     LengthWrittenPostProb=0;
  }
  if (Verbose >= 5) {
    Rprintf("FillPostProbBuffer: Finished with LengthWrittenPostProb=%d/%d\n",
      LengthWrittenPostProb,LengthPostProbCodaBuffer);
  }
  return(1);
}                              
int BayesSpikeCL::GoFindInOld(double AnEnergy) {
  if (ProbOldCoda == NULL) {
    Rf_error("GoFindInOld:  Sorry, can't find anything when ProbOldCoda is NULL!");
  }
  return(FindInMe(LengthOldCoda, ProbOldCoda, AnEnergy));
}
int BayesSpikeCL::TestFindInMe(SEXP FindProb) {
  if (Rf_isNull(FindProb) || Rf_length(FindProb) <= 0) {
    Rprintf("BayesSpikeCL: TestFindInMe: NULL FindProb. \n");
    return(-1);
  }
  double dFindProb = 0.0;
  if (Rf_isReal(FindProb)) {
    dFindProb = REAL(FindProb)[0];
  } else if (Rf_isInteger(FindProb)) {
    dFindProb = INTEGER(FindProb)[0];
  }  else {
    Rprintf("TestFindInMe Please Improve FindProb Input!\n");
    return(-1);
  }
   if (ProbOldCoda == NULL) { Rprintf("TestFindInMe: No, ProbOldCoda is NULL. \n"); return(-1); }
   return(FindInMe(LengthOldCoda, ProbOldCoda, dFindProb));
}

////////////////////////////////////////////////////////////////////////////////
// EESamplerMerge()
//
//    As described in Section 2.3 and used in the Simulations, this
//  merge step pulls draws from a previous higher temperature within an energy
//  window of our current draw at local temperature.  This allows the sampler
//  to quickly draw into a new sampler territory.
//
//  We use "FindInMe" to find closest sample within the sorted ProbOldCoda.
//   We then sample amongth similar Posterior values in there.
int BayesSpikeCL::EESamplerMerge() {
  if (ProbOldCoda == NULL) { return(-1); }
  if (Verbose > 2) {
    Rprintf("EESamplerMerge tt = %d/%d, conducting merge. \n", tt, MaxGibbsIters); R_FlushConsole();
  }
  int MyPull = 0;  double LPP = 0.0;  double NewProb = 0.0; if (Tempii == 0) { return(1); }
  if (DoingEEProbSort  == 1 && EEProbSortWidth > 0) {
    int MyFindMin = FindInMe(LengthOldCoda, ProbOldCoda, CurrentProb-EEProbSortWidth);
    if (Verbose >= 2) {
      Rprintf("EESamplerMerge: CurrentProb=%f, EEPSW=%f, found MFM=%d for Prob=%f, to right it is %f. \n",
        CurrentProb, EEProbSortWidth, MyFindMin, ProbOldCoda[MyFindMin],
        (MyFindMin >= 0 && MyFindMin) < LengthOldCoda-1 ? ProbOldCoda[MyFindMin+1] : (0.0/0.0));
      R_FlushConsole();
    }
    if (MyFindMin < LengthOldCoda-1 && 
      ProbOldCoda[MyFindMin] < CurrentProb-EEProbSortWidth) {
      MyFindMin++;
    }
    int MyFindMax = FindInMe(LengthOldCoda, ProbOldCoda, CurrentProb+EEProbSortWidth);
    if (MyFindMax > 0 && 
      ProbOldCoda[MyFindMax] < CurrentProb+EEProbSortWidth) {
      MyFindMax--;
    }
    if (MyFindMin < 0) { MyFindMin = 0; }
    if (MyFindMax >= LengthOldCoda) {MyFindMax = LengthOldCoda-1; }
    if (Verbose > 5) {
      Rprintf("--------------------------------------------------------------\n");
      Rprintf("  EESamplerMerge: Sorted version --- \n");
      Rprintf("  Look for CurrentProb = %f +- %f My FindMin = %d,  MyFindMax = %d \n",
        CurrentProb, EEProbSortWidth, MyFindMin, MyFindMax);
      Rprintf("  Probability MyFindMin[%d] = %f \n", MyFindMin, ProbOldCoda[MyFindMin]);
      Rprintf("  Probability MyFindMax[%d] = %f \n", MyFindMax, ProbOldCoda[MyFindMax]);
      R_FlushConsole();
    }
    if (MyFindMin >= MyFindMax) {
      MyFindMin = MyFindMax;
      MyPull = MyFindMax;
    } else {
      MyPull =(int) (MyFindMin + 
        (int) floor( (MyFindMax - MyFindMin+1.0) * Rf_runif(0.0,1.0)));
    }
    if (MyPull >= MyFindMax) { MyPull = MyFindMax; }
    NewProb = ProbOldCoda[MyPull];
    int WouldFindMin = FindInMe(LengthOldCoda, ProbOldCoda, NewProb - EEProbSortWidth);
    int WouldFindMax = FindInMe(LengthOldCoda, ProbOldCoda, NewProb + EEProbSortWidth);
    if (WouldFindMax > 0 && 
      ProbOldCoda[WouldFindMax] > NewProb+EEProbSortWidth) {
      WouldFindMax--;
    }
    if (WouldFindMin < LengthOldCoda-1 && 
      ProbOldCoda[WouldFindMin] < NewProb-EEProbSortWidth) {
      WouldFindMin++;
    }
    if (WouldFindMax >= LengthOldCoda) {
      WouldFindMax = LengthOldCoda -1;
    }
    if (WouldFindMin < 0) {
      WouldFindMin = 0;
    }
    
    if (Verbose > 5) {
      Rprintf("   We chose MyPull = %d, between [%d and %d] with prob = %f \n", MyFindMin, MyFindMax, MyPull, NewProb);
      Rprintf("   WouldFindMin = %d, with prob %f \n",  WouldFindMin, ProbOldCoda[WouldFindMin]);
      Rprintf("   WouldFindMax = %d, with prob %f \n", WouldFindMax, ProbOldCoda[WouldFindMax]);
      R_FlushConsole();
    }
    if (WouldFindMin > WouldFindMax) {
      LPP = -1;
    }  else if (WouldFindMin == WouldFindMax && CurrentProb !=  ProbOldCoda[WouldFindMax]) {
      LPP = -1;
    } else if (CurrentProb > ProbOldCoda[WouldFindMax] || CurrentProb < ProbOldCoda[WouldFindMin])  {
      LPP = -1;
    } else {
      LPP = NewProb * invTemperature - CurrentProb * invTemperature - 
        log(1.0 / ( MyFindMax - MyFindMin +1.0)) +
        log(1.0 / (WouldFindMax - WouldFindMin + 1.0));
      // Want Probability of Acceptance be p(Accept) = min(1,exp(LPP))
      //                          Accept if a random uniform is less than min(1,exp(LPP)).
    }
    if (Verbose > 5) {
      Rprintf("  invTemperature = %f, As a result, LPP = %f \n", invTemperature, LPP);
      Rprintf("---------------------------------------------------------------\n");
      R_FlushConsole();
    }
    //double DMin = fabs(log(Rf_runif(0.0,1.0)));
    //int AFindMin = FindInMe(LengthOldCoda, ProbOldCoda, CurrentProb -DMin );
  } else {
    MyPull = (int) floor( LengthOldCoda* Rf_runif(0.0,1.0));
    if (MyPull >= LengthOldCoda) { MyPull = 0; }
    NewProb = ProbOldCoda[MyPull];  
    LPP = NewProb * invTemperature + CurrentProb * invOldTemperature -
     CurrentProb * invTemperature - NewProb *invOldTemperature;
  }

  if (MyPull < 0 || MyPull >= LengthOldCoda) {
    Rf_error("EESAmplerMerge: Clearly an error in MyPull is %d, LengthOldCoda=%d \n",
      MyPull, LengthOldCoda);
  }

  if (Verbose > 2) {
    Rprintf("EESamplerMerge: We're starting\n"); R_FlushConsole();
  }
  size_t result = 0;  

  if (log(Rf_runif(0.0,1.0)) > LPP) {
     LengthMergeActive = -1;
     if (Verbose > 1) {
       Rprintf("----------------------------------------------------------\n");
       Rprintf("EESamplerMerge: We chose not to go, LPP = %f \n", LPP);
       Rprintf("CurrentProb = %f, but NewProb[%d] was %f \n", CurrentProb, 
         MyPull, NewProb);
       Rprintf("----------------------------------------------------------\n");
       R_FlushConsole();
       return(1);
     }
     return(1);
  }
  if (Verbose > 1) {
       Rprintf("----------------------------------------------------------\n");
       Rprintf("EESamplerMerge: We Choose to take new prob LPP = %f \n", LPP);
       Rprintf("CurrentProb = %f, but NewProb[%d] was %f \n", CurrentProb, 
         MyPull, NewProb);
       Rprintf("----------------------------------------------------------\n");
       R_FlushConsole();
  }
  if (Verbose > 1) {
    Rprintf("EESamplerMerge: Chose to Go - open files\n"); R_FlushConsole();
  }
  if (NumMerges != NULL) {
    NumMerges[Tempii]++;
  }
  
  if (RsCodaOldIFile == NULL || Rf_isNull(RsCodaOldIFile->asSexp())) {
    Rprintf("EESamplerMerge Error: RsCodaOldIFile is a NULL!\n"); 
    Rf_error("RsCodaOldIFile is Null cannot do EESamplerMerge");
  }
  if (!Rf_isString(RsCodaOldIFile->asSexp())) {
    Rprintf("EESamplerMerge Error: RsCodaOldIFile is not a String!\n"); 
    Rf_error("RsCodaOldIFile is Not string, annot do EESamplerMerge");
  }
  if (RsCodaOldJFile == NULL || Rf_isNull(RsCodaOldJFile->asSexp())) {
    Rf_error("RsCodaOldJFile is Null cannot do EESamplerMerge");
  }
  if (!Rf_isString(RsCodaOldJFile->asSexp())) {
    Rf_error("RsCodaOldJFile is Not string, annot do EESamplerMerge");
  } 
  if (Verbose > 1) {
    Rprintf("EESamplerMerge: Get File sizes \n"); R_FlushConsole();
  } 
  
  FILE *fCodaIFile = NULL, *fCodaDFile = NULL;
  
  int weClosed = 1;
  long file_size;
  struct stat stbuf;
  fd1 = open(CHAR(STRING_ELT(RsCodaOldIFile->asSexp(), 0)), O_RDONLY);
    TotalOpenedFiles++;
  if (fd1 == -1)  {
    Rprintf("EESamplerMerge: Error, Could not Open RsCodaOldIFile = %s\n",
      CHAR(STRING_ELT(RsCodaOldIFile->asSexp(), 0)));
    Rf_error("  Trying to open RsCodaOldIFile for length failed at fd1 ");
  }
  //fCodaIFile = fdopen(fd1, "rb");
  if (fstat(fd1, &stbuf) == -1) {
    Rprintf("EESamplerMerge: Error, Could not run fstat on RsCodaOldIFile = %s\n",
      CHAR(STRING_ELT(RsCodaOldIFile->asSexp(), 0)));
    Rf_error("Setup RsCodaOldIFile: could not use fstat");
  }
  file_size = stbuf.st_size;
  long int lSizeI = file_size / sizeof(int);
  //fdclose(fPOldICoda);
  weClosed = close(fd1);  
  if (weClosed != 0) {
    Rprintf("EESamplerMerge: Error, Could not close corrctly RsCodaOldIFile = %s\n",
      CHAR(STRING_ELT(RsCodaOldIFile->asSexp(), 0)));
    Rprintf("EEMergeEvery  Well, we failed to close RsCodaOldIFile !\n");  R_FlushConsole();
  }
  TotalClosedFiles++;
  fCodaIFile = NULL;
  struct stat stbuf2;
  fd2 = open(CHAR(STRING_ELT(RsCodaOldJFile->asSexp(), 0)), O_RDONLY);
    TotalOpenedFiles++;
  if (fd2 == -1)  {
    Rprintf("EESamplerMerge: Error, Could not Open RsCodaOldDFile = %s\n",
      CHAR(STRING_ELT(RsCodaOldJFile->asSexp(), 0)));
    Rf_error("  Trying to open RsCodaOldDFile for length failed at fd2 ");
  }
  //fCodaDFile = fdopen(fd2, "rb");
  if (fstat(fd2, &stbuf2) == -1) {
    Rprintf("EESamplerMerge: Error, Could not Open fstat RsCodaOldDFile = %s\n",
      CHAR(STRING_ELT(RsCodaOldJFile->asSexp(), 0)));
    Rf_error("Setup RsCodaOldIFile: could not use fstat");
  }
  file_size = stbuf2.st_size;
  long int lSizeD = file_size / sizeof(double);
  if (lSizeD != lSizeI) {
    Rprintf("EESamplerMerge: Lengths MISSMATCH = %s\n",
      CHAR(STRING_ELT(RsCodaOldJFile->asSexp(), 0)));
    Rprintf("EESampleMerge: Warning lSizeD = %d, lSizeI = %d\n", lSizeD, lSizeI);
  }
  //fdclose(fPOldICoda);
  weClosed = close(fd2);   
  if (weClosed != 0) {
    Rprintf("EESamplerMerge: Error, Could not Close RsCodaOldDFile = %s\n",
      CHAR(STRING_ELT(RsCodaOldJFile->asSexp(), 0)));
    Rprintf("  Well, we failed to close RsCodaOldJFile !\n");  R_FlushConsole();
  }
  TotalClosedFiles++;
  fCodaDFile = NULL;
  if (Verbose > 3) {
    Rprintf("EESampleMerge: Found ends of lSizeI = %d, lSizeD = %d unnormalized \n", (int) lSizeI, (int) lSizeD);
    R_FlushConsole();  
  }
     
  if (Verbose > 1) {
    Rprintf("EESamplerMerge: Opening RsCodaOldIFile = %s \n",
      CHAR(STRING_ELT(RsCodaOldIFile->asSexp(),0)) ); R_FlushConsole();
  }   
  fCodaIFile = fopen( CHAR(STRING_ELT(RsCodaOldIFile->asSexp(), 0)), "rb");
  
  
  fd1 = -1;  fd2 = -2;
  if (fCodaIFile==NULL) {
    for (int iti = 0; iti < 100; iti++) {
      R_FlushConsole();
    }
    fCodaIFile = fopen( CHAR(STRING_ELT(RsCodaOldIFile->asSexp(), 0)), "rb");
    if (fCodaIFile == NULL) {
      Rprintf("Double Null Fail on fCodaIFile in EESamplerMerge \n");
      Rprintf("  TotalOpenedFiles = %d, TotalClosedFiles = %d \n",
        TotalOpenedFiles, TotalClosedFiles);
      fd1 = open( CHAR(STRING_ELT(RsCodaOldIFile->asSexp(), 0)), O_RDONLY);
      if (fd1 == -1) {
        Rprintf("We were not able to open with open I File either! \n"); R_FlushConsole();      
      }
      TotalOpenedFiles++;
      Rprintf("EESamplerMerge: DOUBLE NULL FAIL Error, Could not Open RsCodaOldIFile = %s\n",
        CHAR(STRING_ELT(RsCodaOldIFile->asSexp(), 0)));
      Rf_error("EESampleMerge: sCodaI Old File failed to open");
    }
  } 
  TotalOpenedFiles++;
  if (Verbose > 1) {
    Rprintf("EESamplerMerge: Opening RsCodaOldJFile = %s \n",
      CHAR(STRING_ELT(RsCodaOldJFile->asSexp(),0)) ); R_FlushConsole();
  } 
  fCodaDFile = fopen( CHAR(STRING_ELT(RsCodaOldJFile->asSexp(), 0)), "rb");   
  if (fCodaDFile==NULL) {
    for (int iti = 0; iti < 100; iti++) {
      R_FlushConsole();
    }
    fCodaDFile = fopen( CHAR(STRING_ELT(RsCodaOldJFile->asSexp(), 0)), "rb");
    if (fCodaDFile == NULL) {
      fclose(fCodaIFile);  TotalClosedFiles++;
      fd2 = open( CHAR(STRING_ELT(RsCodaOldJFile->asSexp(), 0)), O_RDONLY);

      if (fd2 == -1) {
        Rprintf("We were not able to open with open J File either! \n"); R_FlushConsole();
      }
      TotalOpenedFiles++;
      Rf_error("EESampleMerge: sCodaD Old File failed to open");
    }
  } 
  TotalOpenedFiles++;
  int SeekSize = 0;


  
  if ( (long int) (IOldCoda[MyPull]  ) >= lSizeI) {
    Rprintf("EESampleMerge Error: IOldCoda[MyPull] = %d, goes beyond lOldSizeI = %d\n",
      IOldCoda[MyPull] * sizeof(int), lSizeI); 
    fclose(fCodaIFile);  fclose(fCodaDFile);   TotalClosedFiles+=2;
    Rf_error("EESampleMerge Error.");
  }
  if ( (long int) (IOldCoda[MyPull]) >= lSizeD) {
    Rprintf("EESampleMerge Error: IOldCoda[MyPull] = %d, goes beyond lOldSizeD = %d\n",
      IOldCoda[MyPull] * sizeof(double), lSizeD); 
    fclose(fCodaIFile); fclose(fCodaDFile);   TotalClosedFiles+=2;
    Rf_error("EESampleMerge Error.");
  }
  // obtain file size:
  
  if (Verbose > 3) {
    Rprintf("EESampleMerge: Seeking to MyPull=%d from file to %d \n",
      MyPull, IOldCoda[MyPull]); R_FlushConsole();
  } 
  int fSeekFail;
  fSeekFail = fseek (fCodaIFile , ((long) sizeof(int) *  (long) IOldCoda[MyPull]), SEEK_SET);
  if (fSeekFail!=0) {
    fclose(fCodaIFile);  fclose(fCodaDFile);  TotalClosedFiles+=2;
    Rf_error("EESampleMerge: fseek fails for fCodaIFile \n");
  }
  fSeekFail = fseek (fCodaDFile , ((long) sizeof(double) * (long) IOldCoda[MyPull]), SEEK_SET);  
  if (fSeekFail != 0) {
    fclose(fCodaIFile); fclose(fCodaDFile);   TotalClosedFiles+=2;
    Rf_error("EESampleMerge: fseek fails for fCodaDFile \n");
  }
  
  if (Verbose > 3) {
    Rprintf("EESamplerMerge: Going to read from fCodaIFile \n"); R_FlushConsole();
  }
  int Checker[3];  double CheckerD[3];
  result = fread(Checker, sizeof(int), 1, fCodaIFile);
  if (result <= 0) {
    fclose(fCodaIFile);  fclose(fCodaDFile);  TotalClosedFiles+=2;
    Rf_error("EESampleMerge: failed to read from fCodaIFile pointer!");
  }
  if (Verbose > 3) {
    Rprintf("Successfully read Checker[0] = %d, on to Checker D \n", Checker[0]); R_FlushConsole();
  }
  result = fread(CheckerD, sizeof(double), 1, fCodaDFile);
  if (result <= 0) {
    Rf_error("EESampleMerge: failed to read from fCodaJFile pointer");
  }
  if (Verbose > 3) {
    Rprintf("Successfuly read CheckerD[0] = %f, check how good it is \n", CheckerD[0]); R_FlushConsole();
  }
  int SetBack = -3;
  if (CheckerD[0] <= 0.0) {
    Rprintf("CheckerD = %f <= 0 error \n", CheckerD[0]); R_FlushConsole();
    Rprintf("Error: We read CheckerD from fCodaDFile at %d, file %s, got %f \n",
      IOldCoda[MyPull], CHAR(STRING_ELT(RsCodaOldJFile->asSexp(),0)), CheckerD[0]);
    R_FlushConsole();
    
    if (IOldCoda[MyPull] < 3) {
      SetBack = -IOldCoda[MyPull];
    }
    int GoForward = 8; 
    if (IOldCoda[MyPull] + SetBack + GoForward > MaxLengthOldIDFiles)  {
      GoForward = MaxLengthOldIDFiles - IOldCoda[MyPull] + SetBack + GoForward -1;
    }
    fseek (fCodaDFile , ((long) sizeof(double) * 
      (long) (IOldCoda[MyPull] + SetBack)), SEEK_SET); 
    double MyReadChecker[15];
    result = fread(MyReadChecker, sizeof(double), GoForward, fCodaDFile);
    Rprintf(" Here's a heads up from around fCodaDFile: "); R_FlushConsole();
    PrintVector(MyReadChecker, GoForward); R_FlushConsole();
    Rprintf(" IOldCoda[%d] was %d \n", MyPull, IOldCoda[MyPull]); R_FlushConsole();
    Rprintf("  Checker[0] = %d \n", Checker[0]); R_FlushConsole(); 
    fseek (fCodaIFile , ((long) sizeof(int) * 
      (long) (IOldCoda[MyPull] + SetBack)), SEEK_SET); 
    int MyiReadChecker[15];
    result = fread(MyiReadChecker, sizeof(int), GoForward, fCodaIFile); 
    Rprintf(" Here's a heads up from around fCodaIFile: "); R_FlushConsole();
    PrintVector(MyiReadChecker, GoForward); R_FlushConsole();
    fclose(fCodaIFile); fclose(fCodaDFile);  TotalClosedFiles+=2;      
    Rf_error("Well we quit because CheckerD was <= 0 = %f \n", CheckerD[0]);
  }
  if ( fabs((float)  (( (int) ( CheckerD[0] * 10) ) / 10 ) - CheckerD[0]) > .01) {
    fclose(fCodaIFile); fclose(fCodaDFile);   TotalClosedFiles+=2;
    Rf_error("EESampleMerge Well CheckerD[0] = %f, we might have an error \n",
      CheckerD[0]);
  }
  if (Verbose > 3) {
    Rprintf(" Well CheckerD doesn't seem automaticall bad!\n"); R_FlushConsole();
  }

  if (DoingEEProbSort <= 0) {
    if (MyPull == LengthOldCoda -1) {
      SeekSize = (int) (((long)MaxLengthOldIDFiles) - ((long)IOldCoda[MyPull] -1));
    } else {
      SeekSize = (int) (((long)IOldCoda[MyPull+1]) - ((long)IOldCoda[MyPull] -1));
    }
    if (Verbose > 1) {
      Rprintf("EESampleMerge: MyPull = %d, We're going to take a SeekSize = %d \n", 
        MyPull, SeekSize); R_FlushConsole();
    }
    
  }
  if (DoingEEProbSort == 1) {
    SeekSize = (int) round(CheckerD[0]);
  } else if (SeekSize != (int) round( CheckerD[0])) {
    Rprintf("Error: EESampleMerge, We got CheckerD[0] = %f, but we were wrong about length %d\n",
      CheckerD[0], SeekSize);
    Rprintf("We have MyPull = %d, IOldCoda[MyPull] = %d \n", MyPull, IOldCoda[MyPull]); R_FlushConsole();
    if (MyPull == LengthOldCoda-1) {
      Rprintf("MyPull = LengthOldCoda-1, so MaxLengthOldIDFiles = %d \n", MaxLengthOldIDFiles); R_FlushConsole();
    } else {
      Rprintf(" And ICoda[MyPull+1] = %d\n", IOldCoda[MyPull+1]); R_FlushConsole();
    }
    SeekSize = (int) round(CheckerD[0]);
    Rprintf("Just Live again !"); R_FlushConsole();
  }
  
  if (OnKappaMem < SeekSize) {
      if (Verbose > 3)  {
        Rprintf("EESampleMerge: Seeker = %d, OnKappaMem = %d, we're going to have to resize\n");
        R_FlushConsole();
      }
      double Factor = (SeekSize+2.0) / OnKappaMem;
      Resize(Factor);
       if (p >= 100000) {
         Rprintf("*** BayesSpikeGibbs.cpp:: EESampleMerge, just resized to OnKappaMem=%d, OnKappaS=%d\n", 
           OnKappaMem, OnKappaS);
       }
      if (SeekSize >= OnKappaMem) {
        Rf_error("*** BayesSpikeGibbs.cpp:: EESampleMerge: SeekSize=%d, OnKappaMem=%d after resize!\n",
          SeekSize, OnKappaMem);
      }
  }
  
  if (Checker[0] > 0) {
    Rprintf("Error: EESampleMerge Checker[0] = %d, not good thats a beta coordinate!\n", Checker[0]);
    Rf_error("EESampleMerge: Error");
  }
  if (Verbose > 3 && Checker[0] < 0) {
    Rprintf("Well Checker[0] Looks good, let's as about it !\n");  R_FlushConsole();
    Rprintf("Error: We read Checker from CodaIFile at %d, file %s, got %d \n",
      IOldCoda[MyPull], CHAR(STRING_ELT(RsCodaOldIFile->asSexp(),0)), Checker[0]);
    R_FlushConsole();
  }
    
  if (NumActive > 0) {
    for (int jj = 0; jj < NumActive; jj++) {
      REAL(sBeta)[OrderedActive[jj]] = 0.0;
      BFixed[OrderedActive[jj]] = 0;
      OrderedActive[jj] = 0;
    }
  }
  if (Verbose > 3) {
    Rprintf("EESampleMerge: Okay, finally reading in info\n");   R_FlushConsole();
  }
  
  result=fread(OrderedActive, sizeof(int), SeekSize, fCodaIFile);
  result=fread(PropBeta, sizeof(double), SeekSize, fCodaDFile);
  if (Verbose > 2) {
    Rprintf("--------------------------------------------------------------\n");
    R_FlushConsole();
    Rprintf("EESampleMerge: Here we go, well SeekSize = %d, and we got:  IVec=\n"); R_FlushConsole();
    PrintVector( OrderedActive, SeekSize); R_FlushConsole();
    Rprintf("\n PropBeta="); R_FlushConsole();
    PrintVector( PropBeta, SeekSize); R_FlushConsole();
    Rprintf("\n Hopefully this is useful.");
    Rprintf(" ----------------------------------------------------------- \n");
    R_FlushConsole();
  }
  if (Verbose > 3) {
    Rprintf("EESampleMerge: Read Info, now trying to add coordinates that aren't in XtX \n");   R_FlushConsole();
  }
  NumActive = SeekSize;
  for (int ii = 0; ii < SeekSize; ii++) {
    if (Verbose > 4) {
      Rprintf("Should we add Coordinate ii = %d,", ii); R_FlushConsole();
      Rprintf(" or PropBeta = %f\n", PropBeta[ii] ); R_FlushConsole();
      Rprintf("  OrderedActive[ii] = %d \n", OrderedActive[ii]); R_FlushConsole();
      Rprintf("  And XLC[OA[ii]] = %d \n", XLC[OrderedActive[ii]]); R_FlushConsole();
    }
    if (OrderedActive[ii] < 0 || OrderedActive[ii] >= p) {
      Rprintf("EESampleMerge: Well we've now seen a problem, seekSize = %d and OrderedActive = \n", SeekSize);
      PrintVector(OrderedActive, SeekSize);
      Rprintf("IOldCoda[MyPull = %d] = %d, SetBack = %d \n",
         MyPull, (int) IOldCoda[MyPull], SetBack);   R_FlushConsole();
      Rf_error(" Hey: Looks like a file error, you searched for coordinate %d, ii = %d/%d, but no dice! \n",
        OrderedActive[ii], ii, NumActive);
       R_FlushConsole();
       
    }
     if (PropBeta[ii] != 0.0 && XLC[OrderedActive[ii]] < 0) {
       AddCoord(OrderedActive[ii]);       
     } 
     if (PropBeta[ii] != 0.0) {
       BFixed[OrderedActive[ii]] = 1;
     }
  }
  NumActive = SeekSize;
  
  if (MaxLengthMergeActive < NumActive) {
    if (MergeActive != NULL) {
      FFree(MergeActive, "MergeActive");
      FFree(MergePropBeta, "MergePropBeta");
    }
    MergeActive = Calloc(NumActive, int);
    MergePropBeta = Calloc(NumActive, double);
    MaxLengthMergeActive = NumActive;
  }
  LengthMergeActive = NumActive;
  for (int ii = 0; ii < NumActive; ii++) {
    MergeActive[ii] = OrderedActive[ii];
    MergePropBeta[ii] = PropBeta[ii];
  }
  
  if (Verbose > 3) {
    Rprintf("EESampleMerge: Closing I and D Files \n"); R_FlushConsole();
  }
  
  weClosed = fclose(fCodaIFile); 
  if (weClosed != 0) {
    Rprintf("EESampleMerge: Failed to close CodaIFile\n"); R_FlushConsole(); 
  } 
  TotalClosedFiles++;
  fCodaIFile = NULL; fclose(fCodaDFile); 
  if (weClosed != 0) {
    Rprintf("EESampleMerge: Failed to close CodaDFile\n"); R_FlushConsole();
  }
  TotalClosedFiles++;
  fCodaDFile = NULL;
  if (Verbose > 1) {
    Rprintf("EESampler:  Done with Coordinates Added, just update other statistics.\n"); 
    R_FlushConsole();
  }
  int SF = 0;
  MRF(FillsBetaFromPropBetaAndCompute,"FillsBetaFromPropBetaAndCompute",5, -1);
  
  if (Verbose >= 1) {
    Rprintf("EESampler: Testing XtResid first time\n");  R_FlushConsole();
    //TestXtResid();
    Rprintf("EESampler -- Did we succeed? \n"); R_FlushConsole();
  } 
  if (Verbose >= 1) {
    printf("EESampler: We have Merged a Beta in. \n"); R_FlushConsole();
  } 
  if (dfRobit >= 0.0) {
    MRF(RobitReplace, "RobitReplace", 8, -1);
  }
  MRF(UpdatePiA, "UpdatePiA",2, -1);
  if (Verbose > 3) {
    Rprintf("EESampler, we finished! \n"); R_FlushConsole();
  }
  //if (RSigmaPrior != NULL && !Rf_isNull(SigmaPrior) && Rf_isReal(SigmaPrior) &&
  //  Rf_length(SigmaPrior) >= 2) {     
    MRF(UpdateSigma, "UpdateSigma",10, -1);
  //}
  if (dfTNoise > 0.0) {
    MRF(UpdateTNoise, "UpdateTNoise", 19, -1);
  }
  if (iFirstRandom != 0) {
    int BackDoSampleBFixedii = DoSampleBFixedii;
    DoSampleBFixedii = -1;
    int BackVerbose = Verbose; Verbose = Verbose-3;
    MRF(SampleFixedB, "SampleFixedB", 11, -1);  // Update the fixed coordinates  
    DoSampleBFixedii = BackDoSampleBFixedii;
    Verbose = BackVerbose;
    
    if (Verbose >= 1) {
      Rprintf("EESampler: We resampled FixedB did XtResid work out?\n");  R_FlushConsole();
      //TestXtResid();
      Rprintf("EESampler -- Did we succeed? \n"); R_FlushConsole();
    }   
  }   
  if (!Rf_isNull(tauEndList)) {
    MRF(SampleNewTaus, "SampleNewTaus",3, -1);
  }

  return(1);
}

int BayesSpikeCL::RobitReplace() {
  int One = 1;  double ZeroD = 0.0; 
  double OneD = 1.0;
  char Trans = 'N';
  if (dfRobit  < 0) {
    Rprintf("RobitReplace: Why did we call this? dfRobit = %d \n", dfRobit);
    return(-1);
  }
  if (Verbose >= 4) {
    Rprintf("RobitReplace: Starting, dfRobit = %f\n", dfRobit); R_FlushConsole();
  }
  if (iiWeight == NULL && dfRobit > 0.0) {
    if (Verbose >= 2) {
      Rprintf("RobitReplace: Hey, dfRobit = %f, but iiWeight is NULL, why?\n", dfRobit);
    }
    RMemGetD(iiWeight, "iiWeight", n);
    for (int iti = 0; iti < n; iti++) {
      iiWeight[iti] = 1.0;
    }
  }
  if (iWeightedXtX == NULL) {
    if (Verbose >= 2) {
      Rprintf("RobitReplace: Hey dfRobit=%f, we will have to allocate iWeightedXtX. \n");
    }
    RMemReallocS(iWeightedXtX, "iWeightedXtX", OnKappaMem);
    for (int iti = 0; iti < OnKappaMem; iti++) {
       iWeightedXtX[iti] = 0;
    } 
    if (Verbose >= 2) {
      printf("-- Just Allocated iWeightedXtX\n");
    }
  }
  if (Verbose >= 4) {
    Rprintf("RobitReplace(): Guaranteed that iiWeight and iWeightedXtX allocated. \n");
  }
  if (dfRobit > 0.0 && Verbose >= 4) {
    Rprintf("RobitReplace(): For good old sakes we are checking WeightXtX. \n");
    int CountBad = QuickCheckWeightXtX();
    if (CountBad > 0) {
      Rprintf("BayesSpikeCpp.cpp():::RobitReplace on tt=%d/%d we got CountBad = %d, BEFORE we Reweight.   Not Good I quit!\n",
        tt, MaxGibbsIters, CountBad); R_FlushConsole();
      Rf_error("BayesSpikeCpp.cpp():::RobitReplace fail at beginning. \n"); R_FlushConsole();
    }
  }
  if (Verbose >= 4) {
    Rprintf("RobitReplace(): About to make decision because NumActive=%d, p=%d to rescale Y. \n",
      NumActive, p);
  }
  if (NumActive > 0 && NumActive < p/2) {
    F77_CALL(dscal)(&n, &ZeroD, REAL(sY), &One);
    for (int iti = 0; iti < NumActive; iti++ ) {
      F77_CALL(daxpy)(&n, REAL(sBeta) + OrderedActive[iti],
        REAL(sX) + OrderedActive[iti] *n, &One, REAL(sY), &One);
    }
  } else {

    F77_CALL(dgemv)(&Trans, &n, &p, &OneD, REAL(sX), &n, REAL(sBeta), &One,
      &ZeroD, REAL(sY),&One);
  }
  if (Verbose >= 4) {
    Rprintf("RobitReplace(): We have successfully reloaded Y note that RobitNotReplaceFlag=%d \n", RobitNotReplaceFlag);
  }
  if (Verbose >= 5) {
    Rprintf("RobitReplace: After loading in we fill EY: \n");
    PrintVector(REAL(sY), n);
    Rprintf(" \n"); R_FlushConsole();
  }
  if (RobitNotReplaceFlag == 1) {
    if (Verbose >= 5) {
      Rprintf("BayesSpikeGibbs.cpp:::RobitReplace(): RobitNotReplaceFlag = 1, Just puting Expected Y into mix\n"); R_FlushConsole();
    }
    Trans = 'T';
    ZeroD = 0.0;
    F77_CALL(dgemv)(&Trans, &n, &p, &OneD, REAL(sX),&n, REAL(sY), &One, 
      &ZeroD, XtY, &One);
      
    if (Verbose >= 4) {
      Rprintf("BayesSpikeGibbs.cpp: RobitReplace(): Update sOnSigma if necessary. \n");
    }
    if (DoLogitNonePostPreProb <= 0) {
      REAL(sOnSigma)[0] = 1.0;                       
    } else if (DoLogitNonePostPreProb >= 1) {
      //REAL(sOnSigma)[0] = 7/9 * 3.141593  *  3.141593 /3;
    }
    UpdateXtY();    YSq = 0.0;
    for (int tt = 0; tt < n; tt++) {
      YSq += REAL(sY)[tt] * REAL(sY)[tt];
    }
  
    UpdateFreshXtResid();
    if (Verbose >= 5) {
      Rprintf("BayesSpikeGibbs.cpp:::RobitReplace() Early Quit, Updated YSq, YSq = %f \n", YSq); R_FlushConsole();
    }
    CurrentProb = ProbPosterior();
    return(1);
  }
  if (Verbose >= 5) {
      Rprintf("BayesSpikeGibbs.cpp:::RobitReplace(), Calculate CurrentProb from ProbPosterior()\n", Temperature);
      R_FlushConsole();
  }
  CurrentProb = ProbPosterior();
  double PSig = 0.0, iPSig = 0.0;
  double iSigma = sqrt(1.0 / REAL(sOnSigma)[0]);
  double sqSigma = sqrt(REAL(sOnSigma)[0]);
  if (dfRobit == 0.0) {
    if (Verbose >= 5) {
      Rprintf("BayesSpikeGibbs.cpp:::RobitReplace(), Drawing Truncated Norm Draws, Temp = %f\n", Temperature);
      R_FlushConsole();
    }
    if (Temperature == 1.0) {
    if (iiWeight == NULL || (AlterWeightdfRobit < 0.0 && dfRobit < 0.0)) {
    for (int iti = 0; iti < n; iti++) {
      if (intY[iti] == 1) {
        REAL(sY)[iti] = REAL(sY)[iti] + rTruncNorm2( -REAL(sY)[iti]*iSigma, R_PosInf) * sqSigma; 
      } else {                                                   
        REAL(sY)[iti] = REAL(sY)[iti] + rTruncNorm2( R_NegInf, -REAL(sY)[iti] * iSigma) * sqSigma; 
      }
    } 
    } else {
    for (int iti = 0; iti < n; iti++) {
      PSig = sqrt( REAL(sOnSigma)[0] * Temperature / iiWeight[iti]); 
      //PSig = sqrt(REAL(sOnSigma)[0] * Temperature);
      iPSig = 1.0 / PSig;    
      if (intY[iti] == 1) {
        REAL(sY)[iti] = REAL(sY)[iti] + rTruncNorm2( -REAL(sY)[iti] * iPSig, R_PosInf) * PSig; 
      } else {                                                   
        REAL(sY)[iti] = REAL(sY)[iti] + rTruncNorm2( R_NegInf, -REAL(sY)[iti] * iPSig) * PSig; 
      }
    }     
    }
    } else {
    //double iSqrtT = sqrt(invTemperature);
    //F77_CALL(dscal)(&n, &iSqrtT, REAL(sY), &One);
    if (iiWeight == NULL || AlterWeightdfRobit < 0.0) {
     sqSigma = sqrt( REAL(sOnSigma)[0] * Temperature );
     iSigma = 1.0 / sqSigma; 
     for (int iti = 0; iti < n; iti++) {
       if (intY[iti] == 1) {
         REAL(sY)[iti] = REAL(sY)[iti] + rTruncNorm2( -REAL(sY)[iti] * iSigma, R_PosInf) * sqSigma; 
       } else {                                                   
         REAL(sY)[iti] = REAL(sY)[iti] + rTruncNorm2( R_NegInf, -REAL(sY)[iti]*iSigma) * sqSigma; 
       }
     }
    } else {
    // (Yi-barY)^2 wi = (Y-barY)^2/sigma^2
    for (int iti = 0; iti < n; iti++) {
      PSig = sqrt(1.0/iiWeight[iti]  * REAL(sOnSigma)[0] * Temperature);
      //PSig = sqrt(REAL(sOnSigma)[0] * Temperature);
      iPSig = 1.0 / PSig;
      //PSig = REAL(sOnSigma)[0];
      if (intY[iti] == 1) {
        REAL(sY)[iti] = REAL(sY)[iti] + PSig * rTruncNorm2( -REAL(sY)[iti] * iPSig, R_PosInf); 
      } else {                                                   
        REAL(sY)[iti] = REAL(sY)[iti] + PSig * rTruncNorm2( R_NegInf, -REAL(sY)[iti] * iPSig); 
      }
    }    
    }     
    }
  } else {
    if (Verbose >= 5) {
      Rprintf("BayesSpikeGibbs.cpp:::RobitReplace() Drawing dfRobit = %f draws, Temperature = %f\n", 
        dfRobit, Temperature);
      R_FlushConsole();
    }

    if (Temperature == 1.0) {
    for (int iti = 0; iti < n; iti++) {
      if (intY[iti] == 1) {
        REAL(sY)[iti] = REAL(sY)[iti] + rTruncT2(dfRobit, -REAL(sY)[iti] * iSigma, R_PosInf) * sqSigma; 
      } else {                                                   
        REAL(sY)[iti] = REAL(sY)[iti] + rTruncT2(dfRobit, R_NegInf, -REAL(sY)[iti] * iSigma) * sqSigma; 
      }
    } 
    } else {
    double dTemp = invTemperature * dfRobit;
    //sqSigma = sqrt(Temperature) * sqSigma;
    //iSigma = 1.0 / sqSigma;
    for (int iti = 0; iti < n; iti++) {
      if (intY[iti] == 1) {
        REAL(sY)[iti] = REAL(sY)[iti] + rTruncT2(dTemp, -REAL(sY)[iti]*iSigma, R_PosInf) * sqSigma; 
      } else {                                                   
        REAL(sY)[iti] = REAL(sY)[iti] + rTruncT2(dTemp, R_NegInf, -REAL(sY)[iti]*iSigma) * sqSigma; 
      }
    }     
    }
  } 

  if (DoLogitNonePostPreProb <= 0) {
    REAL(sOnSigma)[0] = 1.0;
  }
  if (dfRobit == 0.0) {
    //dfTNoise = 0.0;
    UpdateXtY();
    if (DoLogitNonePostPreProb <= 0) {
      REAL(sOnSigma)[0] = 1.0;
    }
    YSq = 0.0;
    if (Verbose >= 5) {
      Rprintf("BayesSpikeGibbs.cpp:::RobitReplace() Updating YSq \n"); R_FlushConsole();
    }
    for (int iti = 0; iti < n; iti++) {
      YSq += REAL(sY)[iti] * REAL(sY)[iti];
    }
    if (Verbose >= 5) {
      Rprintf("BayesSpikeGibbs.cpp:::RobitReplace(): Updated YSq, YSq = %f \n", YSq); R_FlushConsole();
    }
    UpdateFreshXtResid();
    if (Verbose >= 5) {
      Rprintf("BayesSpikeGibbs.cpp:::RobitReplace(): UpdatedFreshXtResid, now YSq = %f \n", YSq); R_FlushConsole();
    }
  } else if (dfRobit > 0.0) {
    dfTNoise = dfRobit;
    if (Verbose >= 5) {
      Rprintf("  BayesSpikeGibbs.cpp:::RobitReplace(): tt = %d/%d, About to run UpdateTNoise. \n", 
        tt, MaxGibbsIters);  R_FlushConsole();
    }
    if (DoLogitNonePostPreProb <= 0) {
      REAL(sOnSigma)[0] = 1.0;
    }
    UpdateTNoise(); 
    /*
    if (tt == 0) {
      Rprintf("  BayesSpikeGibbs.cpp::RobitReplace(): tt = %d/%d, testing XtResid.\n", tt, MaxGibbsIters);  R_FlushConsole();
      int ACount = TestXtResid();
      if (ACount == -1) {
        Rprintf("  BayesSpikeGibbs.cpp::RobitReplace(): tt = %d/%d ERROR, TestxtResid is bad after UpdateTNoise.\n"); R_FlushConsole();
        Rf_error("  BayesSpikeGibbs.cpp::RobitReplace() Early Fail.\n", tt, MaxGibbsIters); 
      }
    }    
    int CountBad = QuickCheckWeightXtX();
    if (CountBad > 0) {
      Rprintf("BayesSpikeCpp.cpp():::RobitReplace on tt=%d/%d we got CountBad = %d, After UpdateTNoise we Reweight.   Not Good I quit!\n",
        tt, MaxGibbsIters, CountBad); R_FlushConsole();
      Rf_error("BayesSpikeCpp.cpp():::RobitReplace after UpdateTNoise. \n"); R_FlushConsole();
    } */
    dfTNoise = -1;
  } 
  

  //dfTNoise = dfRobit;
  

  //REAL(sOnSigma)[0] = 1.0;
  return(1);
}

///////////////////////////////////////////////////////////////////////////////
//  CopyIn
//     This version works if Small sXTX, CopyIn 2 works from raw and 
//   has to do harder work.
//
int BayesSpikeCL::CopyIn(int StF, int GoFor) {
  int One = 1;
  int att= 0; int GoWant = NumActive - (StF + GoFor);
  if (Verbose > 2) {
    Rprintf("CopyIn: Start by copying in smXtY. \n"); R_FlushConsole();
  }
  if (smXtY == NULL) {
    if (Verbose >= 1) {
      Rprintf("CopyIn1: Allocate smXtY for first time to length %d\n", OnKappaMem);
      R_FlushConsole();
    }
    RMemGetD(smXtY, "smXtY", OnKappaMem);
  }
  F77_CALL(dcopy)(&GoFor, cXtY, &One, smXtY, &One);
  if (Verbose >= 1) {
    Rprintf("CopyIn1: After the first copy, smXtY = ");  R_FlushConsole();
      PrintVector( smXtY, GoFor); Rprintf("\n"); R_FlushConsole();
  }
  int OnMark = 0; 
  double ISubtract = 0.0;
  int kk;
  if (Verbose > 1) {
    Rprintf("CopyIn1: Now goes to the SmallXtX subtraction. \n"); R_FlushConsole();
  }
  if (StF > 0) {
    for (kk = 0; kk < StF; kk++) {
      ISubtract = -1.0 * PropBeta[kk];
      F77_CALL(daxpy)(&GoFor, &ISubtract, rSmallXtX + OnMark + (StF-kk), &One, smXtY, &One);
      OnMark += NumActive - kk; 
    }
  }
  if (Verbose >= 1) {
    Rprintf("  AFter first part of action = ");
      PrintVector( smXtY, GoFor); Rprintf("\n"); R_FlushConsole();
  }
  if (Verbose >= 1) {
    Rprintf("Copy in middle part \n"); R_FlushConsole();
  }
  int OnStart = 0;
  GoWant = GoFor;
  int StartMark = OnMark;
  for (kk = StF; kk < StF + GoFor; kk++) {   
     if (OnMark + GoWant > (NumActive *(NumActive +1)) / 2) {
       Rf_error("CopyIn Middle Error:  OnMark = %d, GoWant = %d, OnkSq = %d", 
         OnMark, GoWant, NumActive * (NumActive+1) /2 );
     }
     F77_CALL(dcopy)(&GoWant, rICholBack + OnMark, &One, rIChol + OnStart, &One );
     OnStart += GoWant;
     GoWant--;
     OnMark += NumActive - kk; 
  }  
  if (Verbose >= 1) {
    Rprintf("  Mid way, we have smXtY = ");
      PrintVector( smXtY, GoFor); Rprintf("\n"); R_FlushConsole();
  }
  if (Verbose >= 1) {
    Rprintf("Okay: rIChol copied partly, now for last part\n"); R_FlushConsole();
    Rprintf("  We have StartMark = %d, StF = %d, GoFor = %d \n",
      StartMark, StF, GoFor); R_FlushConsole();
    Rprintf("  NumActive = %d, StF + GoFor = %d \n", NumActive, StF, GoFor);
    R_FlushConsole();
  }       
  if (StF + GoFor < NumActive) {
    GoWant = NumActive - (StF + GoFor);
    for (att = 0; att < GoFor; att++) {
       if (StartMark+GoFor-att + GoWant > NumActive * (NumActive+1)/2 ) {
         Rprintf("Uh Oh, StartMark + GoFor + GoWant = %d, ", 
           StartMark + GoFor + GoWant);
         Rf_error(" we're not going to print over the level!");
       } 
       if ( StF + GoFor + GoWant > NumActive) {
         Rf_error("  Uh oh, StF = %d, GoFor = %d, GoWant = %d, NumActive = %d, not good",
           StF, GoWant, GoFor, NumActive);
       }
       if (Verbose >= 4) {
         Rprintf("Copy, we start copying from rICholBack: %d\n", StartMark + GoFor);
         R_FlushConsole();
       }
       smXtY[att] -= F77_CALL(ddot)(&GoWant, rICholBack + StartMark + GoFor-att, &One,
         PropBeta + StF + GoFor, &One);
       StartMark += NumActive - (StF + att);

    }
  }
  for (att = 0; att < GoFor; att++) {
    if (!R_finite(smXtY[att]) || R_isnancpp(smXtY[att])) { smXtY[att] = 0; }
  }
  if (Verbose >= 1) {
    Rprintf("CopyIn1: Finished.\n"); R_FlushConsole();
  }
  return(1);
}


///////////////////////////////////////////////////////////////////////////////
//  CopyIn2
//     This assumes rSmallXtX is not large enough to store all of the 
//   information it needs, thus we have to pull from XtX has to do harder work.

int BayesSpikeCL::CopyIn2(int StF, int GoFor) {
  int One = 1;
  int att= 0; int GoWant = NumActive - (StF + GoFor);
  if (GoWant < 0) {
    GoFor = NumActive - StF;
    GoWant = 0;
  }
  double Factor = 1.0;
  if (Verbose > 1) {
    Rprintf("CopyIn2: Copying smXtY in. \n"); R_FlushConsole();
  }
  if (smXtY == NULL) {
    RMemGetD(smXtY, "smXtY", OnKappaMem);
  }
  if (GoFor > OnKappaMem) {
    Rf_error("Hey: CopyIn2, you gave us GoFor = %d, OnKappaMem=%d, but no way are we making a copy for smXtX\n",
      GoFor, OnKappaMem);
  }
  F77_CALL(dcopy)(&GoFor, cXtY + StF, &One, smXtY, &One);
  if (Verbose > 1) {
    Rprintf("  After the first copy, smXtY = "); R_FlushConsole();
    PrintVector( smXtY, GoFor); Rprintf("\n"); R_FlushConsole();
  }
  //int OnMark = 0; 
  int kk;
  if (Verbose > 1) {
    Rprintf("CopyIn2: Now first part. \n"); R_FlushConsole();
  }
  int OnCoordii;
  /*
  if (StF > 0) {
    for (kk = 0; kk < StF; kk++) {
      ISubtract = -1.0 * PropBeta[kk];
      OnCoordii = OrderedActive[kk];
      if (XLC[OnCoordii] < 0) {
        Rf_error("Error: CopyIn2, how could OnCoordii =%d have XLC =%d?, StF=%d, GoFor=%d\n",
           OnCoordii, XLC[OnCoordii], StF, GoFor);
      }
      OnMark = XLC[OnCoordii] * p;
      for (int jtj = 0; jtj < GoFor; jtj++) {
        smXtY[jtj] += ISubract * XtX[OnMark + OrderedActive[jtj+ StF];
      }
    }
  } */
  //OnMark = 0;
  int CpMe = 0; 
  int OnShrinkFixed = 0;
  if (GoFor > OnKappaMem || GoFor >= MaxLargeSizeSquares) {
    Rf_error("Hey: BayesSpikeGibbs.cpp: We can't write from cXtY or smXtY if GoFor=%d, OnKappaMem=%d\n",
      GoFor, OnKappaMem);
  }
  for (kk = 0; kk < GoFor; kk++) {
    OnCoordii = OrderedActive[kk+StF];
    if (XLC[OnCoordii] < 0 || XLC[OnCoordii] >= OnKappaS) {
      Rprintf("---  Uh Oh, BayesSpikeGibbs.cpp, XLC[OnCoordii=%d] = %d!",
        OnCoordii, XLC[OnCoordii]);  R_FlushConsole();
      Rprintf("kk=%d, StF=%d, OrderedActive[%d+%d=%d]=%d, OnKappaS=%d/%d\n",
        kk, StF, kk, StF, kk+StF, OrderedActive[kk+StF],
        OnKappaS, OnKappaMem);  R_FlushConsole();
      Rprintf("---  Gotta be an error\n"); R_FlushConsole();
      Rf_error("XLC[OnCoordii=%d] = %d.\n", OnCoordii, XLC[OnCoordii]);
    }
    //OnMark = XLC[OnCoordii] * p;
    //smXtY[kk] = cXtY[kk+StF];
    if (GoFor == 1) {
      rIChol[CpMe] = *(pXtX[XLC[OnCoordii]] + OnCoordii);
      smXtY[0] += rIChol[CpMe] * REAL(sBeta)[OnCoordii];
    } else {
    for (int jtj = kk; jtj < GoFor; jtj++) {
      if (pXtX[XLC[OnCoordii]] == NULL) {
          Rprintf(" --- BayesSpikeGibbs.cpp::CopyIn2(tt=%d) pXtX for XLC[OnCoordii=%d] NULL\n",
            tt, OnCoordii);
          Rprintf(" --- OnCoordii=%d, XLC[OnCoordii] = %d \n",
            OnCoordii, XLC[OnCoordii]); R_FlushConsole();
          Rf_error(" --- BayesSpikeGibbs.cpp::CpyIn2() NULL pXtX is Not good.");
       }
       rIChol[CpMe] = *(pXtX[XLC[OnCoordii]] + OrderedActive[jtj+StF]);
       smXtY[kk] += rIChol[CpMe] *
          REAL(sBeta)[OrderedActive[jtj+StF]];   
       if (jtj > kk) {
         smXtY[jtj] += rIChol[CpMe] *
          REAL(sBeta)[OrderedActive[kk+StF]]; 
         //smXtY[kk] += rIChol[CpMe] *
         //  REAL(sBeta)[OrderedActive[jtj+StF]];
       }      
       CpMe++; 
    }
  }
  }
  NumActiveBack =GoFor;
  if (Verbose > 1) {
    Rprintf("CopyIn2: We Have Now filled rIChol and updated smXtY.\n"); R_FlushConsole();
  }
  CpMe = 0;
  int StR = 0;
  for (kk = 0; kk < GoFor; kk++) {
    if (iFirstRandom < 0 || Rf_isNull(sOnTau) || Rf_length(sOnTau) <= 0 || OrderedActive[kk+StF] < iFirstRandom) {
      Factor = -1.0;
      if (LengthNoShrinkFixed > 0 && NoShrinkFixed != NULL && NoShrinkFixedPrior != NULL) {
        while(OnShrinkFixed < LengthNoShrinkFixed && NoShrinkFixed[OnShrinkFixed] < OrderedActive[StF+kk]) {
          OnShrinkFixed++;
        }
        if (OnShrinkFixed < LengthNoShrinkFixed && NoShrinkFixed[OnShrinkFixed] == OrderedActive[StF+kk]) {
          Factor= REAL(sOnSigma)[0] / NoShrinkFixedPrior[OnShrinkFixed];
        }
      }
      if (Factor < 0.0) {
      if (Rf_length(tauFixed) == 1 || Rf_length(tauFixed) < OrderedActive[kk+StF]) {
        if (REAL(tauFixed)[0] > 0.0) {
          Factor = fabs(REAL(sOnSigma)[0] / REAL(tauFixed)[0]);
        } else {
          Factor = fabs(1.0 / REAL(tauFixed)[0]);
        }
      } else if (Rf_length(tauFixed) >= OrderedActive[kk+StF]) {
        if (REAL(tauFixed)[OrderedActive[kk+StF]]) {
          Factor = fabs(REAL(sOnSigma)[0] / REAL(tauFixed)[ OrderedActive[kk+StF]]);
        } else {
          Factor = fabs(1.0 / REAL(tauFixed)[ OrderedActive[kk+StF]]);
        }
      }
      }
      rIChol[CpMe] += Factor;
    } else {
      if (kk + StF > 0) {
        if (OrderedActive[kk+StF-1] > OrderedActive[kk+StF])  {
          Rf_error("Error in set Tau that OrderedActive is ordered fails for kk=%d, StF=%d!\n",
            kk, StF);
        }
      }
      while(StR < Rf_length(tauEndList) && ANINT(tauEndList, StR) < OrderedActive[kk+StF]) {
        StR++;
      }
      if (StR < Rf_length(tauEndList)) {
        rIChol[CpMe] +=  REAL(sOnSigma)[0] / REAL(sOnTau)[StR];
      }
    }
    CpMe += GoFor-kk;
  }
  if (Verbose > 1) {
    Rprintf("  CopyIn2 = smXtY is "); R_FlushConsole();
      PrintVector( smXtY, GoFor); Rprintf("\n"); R_FlushConsole();
  }
  if (Verbose > 1) {
    Rprintf("CopyIn2:  We have Packed rIChol = "); R_FlushConsole();
      PrintVector( rIChol, GoFor); Rprintf("\n"); R_FlushConsole();
    CpMe = 0;
    Rprintf("CopyIn2:   UnPacked (full Dim is %d) that is \n  ", GoFor);  R_FlushConsole();
    int GoDo = GoFor;
    if (GoFor >= 10) { GoDo = 10; }
    //  0
    //  1  6
    //  2  7  11
    //  3  8  12  15
    //  4  9  13  16  18
    //  5 10  14  17  19 20
    for (int jj = 0; jj < GoDo; jj++) {
      CpMe = jj;
      if (jj >= 1) {
        for (int kk = 0; kk < jj; kk++) {
           Rprintf("%f, ", rIChol[CpMe]); R_FlushConsole();
           CpMe += GoFor-kk-1;
        }
      }
      Rprintf("%f\n", rIChol[CpMe]); R_FlushConsole();
    }
    Rprintf("$$ Does this Look good to you for OrderedActive ="); R_FlushConsole();
    PrintVector(OrderedActive+StF, GoFor);
    Rprintf("\n$$  CopyIn2:  Good luck, go looking!\n"); R_FlushConsole();
  }
  for (att = 0; att < GoFor; att++) {
    if (!R_finite(smXtY[att]) || R_isnancpp(smXtY[att])) { smXtY[att] = 0; }
  }
  if (Verbose >= 1) {
    Rprintf("$$ CopyIn2: All Complete, integrity of smXtY is Good.\n"); R_FlushConsole();
  }
  return(1);
}

int BayesSpikeCL::SetupRICholBack() {
   if (OnKappaMem > MaxSquaresAllocation || OnKappaMem > MaxLargeSizeSquares) {
      return(3);
   }
    int OnKSqP = (int) ceil( (OnKappaMem * (OnKappaMem+1)) / 2.0) + 2;
    int One = 1;
    if (OnKSqPBack < OnKSqP) {
      FFree(rICholBack, "rICholBack");
      OnKSqPBack = OnKSqP;
      RMemGetD(rICholBack, "rICholBack", OnKSqPBack);
    }

    if (smXtY == NULL) {
        RMemGetD(smXtY, "smXtY", OnKappaMem);
    }
    if (rIChol != NULL) {
      F77_CALL(dcopy)(&OnKSqP, rIChol, &One, rICholBack, &One);
    }
    return(1);
}
int BayesSpikeCL::SamplePropBeta() {
  int oVerbose = this->Verbose -2;
  if (oVerbose > 2) {
    Rprintf("BayesSpikeCL::BayesSpikeGibbs.cpp:::SamplePropBeta(NumActive=%d) Sampling Beta Now!\n",
      NumActive); R_FlushConsole();
  }
  if (NumActive <= 0) {
    return(0);
  }
  char MUpLo = 'L';
  int Info = 0;  
  char Diag = 'N'; char Trans = 'T';
  int One = 1;
  int jj;
  double OneD = 1.0;   double ZeroD = 0.0; int PickMe;
  Info = 0;
  if (rIChol == NULL) {
    Rf_error("Error: rIChol is Now Null, it very much shouldn't be now!\n");
  }
  if (oVerbose > 3)  {
     Rprintf("SamplePropBeta: About to start, NumberNew=%d, NumActive=%d, MaxInvert=%d \n",
       NumberNew, NumActive, MaxInvert); R_FlushConsole();
  }
  if (NumberNew >= 0 && NumActive <= MaxInvert && NumActive < MaxLargeSizeSquares) {
    if (oVerbose > 2) {
      Rprintf("SamplePropBeta: NumberNew = %d, NumActive=%d, MaxInvert=%d, MaxLargeSizeSquares=%d, Conducting dpptrf Cholesky\n",
        NumberNew, NumActive, MaxInvert, MaxLargeSizeSquares); R_FlushConsole();
    }
    F77_CALL(dpptrf)(&MUpLo, &NumActive, rIChol, &Info);
  }
  if (Info < 0) {
    Rprintf("Error: rIChol as input had an illegal value, tt=%d, NumActive=%d, Info=%d \n", tt, NumActive, Info); R_FlushConsole();
    R_FlushConsole();
    if (NumActive < MaxLargeSizeSquares) {
      Rprintf("rIChol is: \n");
      PrintVector(rIChol, NumActive*(NumActive+1)/2);
      R_FlushConsole();
    } else {
      Rprintf("rIChol is: \n");    R_FlushConsole();
      PrintVector(rIChol, MaxLargeSizeSquares*(MaxLargeSizeSquares+1)/2);
      R_FlushConsole();
    }
  } 
  
  if (Info > 0) {
    if (Verbose > 0) {
      Rprintf("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n");
      Rprintf("III: BayesSpikeGibbs.cpp:::SamplePropBeta(), tt = %d, dpptrf error, not positive ", this->tt);
      Rprintf("III: definite go to small Prop Beta.\n", this->tt); R_FlushConsole();
    }
    NumberNew = 2;
    MaxInvert = NumActive -1;
    PrepareForRegression();
    SetupRICholBack();
    //FillPropBetaFromBeta();
    if (Verbose > 0) {
      Rprintf("|||: I hope we are ready to SamplePropBeta2. \n"); R_FlushConsole();
    }
    // If Info > 0 this means we actually have to do MaxInvert;
  }
  if (Info > 0 || NumActive > MaxInvert || NumActive >= MaxLargeSizeSquares) {    
      // If we can't invert whole matrix, split matrix into parts and do
      //  conditional regressions using SamplePartBeta
      //   SamplePartBeta samples from Beta_Part | Y - X * Beta_NotPart
      //  Hopefully SamplePartBeta does the trick;
      //
      //  There are Two levels of SamplePartBeta
      //   At one level we can fill the entire smXtX and we will invert parts of this
      //   At the next level-2 we only fill pieces of smXtX at a time into RIChol
      //
      //  Because second level is probably the fastest we will eventually change on to-2
      //    exclusively once testing is complete.  AL 10/15/2013
      //
      int StF = 0;  int Move = (int) ceil(NumActive/2);
      if (Move * 2 < NumActive) { Move = Move + 1;}
      if (Move > MaxInvert) { Move = MaxInvert; }
      if (Move <= 0) {
        Rf_error("Error: SamplePartBeta: We can never use zero Move! \n");
      }
      int OldDVerbose = this->Verbose;
      //this->Verbose = 5;
      if (this->Verbose >= 2) {
        Rprintf("*********************************************************************************\n");
        Rprintf("** SamplePartBeta: Large Sample, or Error Sample if Info = %d\n", Info); R_FlushConsole();
        Rprintf("** Note NumActive = %d, MaxInvert=%d, MaxSquaresAllocation=%d\n",
          NumActive, MaxInvert, MaxSquaresAllocation); R_FlushConsole();
      }
      if (OnKappaMem == MaxSquaresAllocation) {
        if (this->Verbose >= 2) {
          Rprintf("** We have NumActive=%d, MaxSqures=%d, OnKappaMem=%d, and we'll do SamplePart 2\n",
            NumActive, MaxSquaresAllocation, OnKappaMem); R_FlushConsole();
        }
        while(StF < NumActive) {
          if (this->Verbose > 2) {
            Rprintf(" Ontt = %d, Running SamplePartBeta(%d,%d) \n", tt, StF, Move); R_FlushConsole();
          }
          if (Move+StF > NumActive) { Move = NumActive - StF; }
          SamplePartBeta2(StF,  Move );
          StF += Move;
        }
        Rprintf("  Ontt = %d, finished SamplePartBeta, we'll quit now. \n", tt); R_FlushConsole();
      } else {
        if (this->Verbose >= 2) {
          Rprintf("** We have NumActive=%d, MaxSqures=%d but OnKappaMem=%d, and we'll do SamplePart 2\n",
            NumActive, MaxSquaresAllocation, OnKappaMem); R_FlushConsole();
        }
        while(StF < NumActive) {
          if (this->Verbose > 2) {
            Rprintf(" Ontt = %d, Running SamplePartBeta2(%d,%d) \n", tt, StF, Move); R_FlushConsole();
          } 
          if (Move+StF > NumActive) { Move = NumActive - StF; }
          SamplePartBeta2(StF,  Move );
          StF += Move;
        }
        if (this->Verbose >= 2) {      
          Rprintf("  Ontt = %d, finished SamplePartBeta2, we'll quit now. \n", tt); R_FlushConsole();
        }
      }
      if (this->Verbose >= 2) {
        Rprintf("  Beta = \n  "); R_FlushConsole();
      }
      int intwenty = 0;
      if (this->Verbose >= 2) {
      for (int ii = 0; ii < NumActive; ii++) {
        if (ii == NumActive -1) { Rprintf("%d:%f\n", OrderedActive[ii], PropBeta[ii]); } else{
          Rprintf("%d:%f, ", OrderedActive[ii], PropBeta[ii]);
        }
        if (intwenty >= 12) {
          Rprintf("\n  "); R_FlushConsole();  intwenty = 0;
        }
        intwenty++;
      }
      }
      int CountNanBeta = 0;  int CountBadPropBeta = 0;
      if (this->Verbose >= 2) {
      for (int ii = 0; ii < NumActive; ii++) {
        if (R_isnancpp(PropBeta[ii]) || !R_finite(PropBeta[ii])) {
          Rprintf("Not Good, there is a nan at %d for coord=%d, PropBeta[%d] = %f\n",
            ii, OrderedActive[ii], ii, PropBeta[ii]); R_FlushConsole();
          CountNanBeta++;
        } else if (fabs(PropBeta[ii]) >= 999999) {
          Rprintf("Weird, we have PropBeta[ii=%d] for OrderedActive[%d]=%d, is %f?\n",
            ii, ii, OrderedActive[ii], PropBeta[ii]); R_FlushConsole();
          CountBadPropBeta++;
        }
      }
      
      if (CountNanBeta > 0 || CountBadPropBeta > 0) {
        Rf_error("SamplePartBeta, we got bad Part Betas and you should investigate!\n"); R_FlushConsole();
      }
      Rprintf("$$ SamplePartBeta2: No Infinites?  Does this look good? \n"); R_FlushConsole();
      }
      this->Verbose = OldDVerbose;
      return(1);   
  }
  int OnKSqP = NumActive*(NumActive+1)/2;
  if (rIChol == NULL) {
    Rf_error("SamplePartBeta: rIChol is NULL!\n");
  }
  if (NumberNew >= 0) {
    F77_CALL(dcopy)(&OnKSqP, rIChol, &One, rI, &One);
    F77_CALL(dpptri)(&MUpLo, &NumActive, rI, &Info);
    if (Info < 0) {
      Rprintf("Error: rIChol as input had an illegal value \n"); R_FlushConsole();
    } else if (Info > 0) {
      Rprintf("SamplePropBetaSmall dpptri error, not positive definite \n"); R_FlushConsole();
      Rf_error("Temporary Freedom");
    }
    F77_CALL(dtptri)(&MUpLo, &Diag, &NumActive, rIChol, &Info);
  }
  // Q = LL'
  // Q^(-1) = L'^(-1) %*% L^(-1)
  double SqrtScal = sqrt(REAL(sOnSigma)[0]);
  int ii;
  double SqT = 1.0;
  //Rprintf("SampleProprBeta: Here we are about to add some noise, TypePrior=%d\n", TypePrior);
  //////////////////////////////////////////////////////////////////////////////
  //  TypePrior =3  is a uniform like prior that bounds the value
  //  It is also implemented in both SamplePartBetas.
  //  Usual operation is TypePrior <=2 which is unbounded
  //  Usual operation is achived by getting inverse of  (XtX+D)  into rI
  //   The contents of rIChol can be used as the noise distribution.
  //
  if (TypePrior != 3) {
    for (ii = 0; ii < NumActive; ii++) { PropBeta[ii] = Rf_rnorm(0.0,1.0); }
    if (Temperature != 1.0) {
    
      SqT = SqrtScal*sqrt(Temperature);
      //Rprintf("Temperature not 1.0 applying Noise SqT = %f, SqrtScal=%f\n", SqT, SqrtScal);   R_FlushConsole();
      F77_CALL(dscal)(&NumActive, &SqT, PropBeta, &One);
    }  else {
      //Rprintf("Temperature is 1.0 applying Noise SqrtScal=%f\n", SqrtScal);   R_FlushConsole();
      F77_CALL(dscal)(&NumActive,&SqrtScal, PropBeta, &One);
    }
    F77_CALL(dtpmv)(&MUpLo,&Trans,&Diag,&NumActive,rIChol,PropBeta,&One);
    F77_CALL(dspmv)(&MUpLo, &NumActive, &OneD, rI, cXtY, 
      &One, &OneD, PropBeta, &One);
  } else {
    F77_CALL(dspmv)(&MUpLo, &NumActive, &OneD, rI, cXtY, 
      &One, &ZeroD, PropBeta, &One);
    PickMe = 0; double Low = 0;  double High = 0;
    double TFL = REAL(tauFixed)[0]; double TFU = REAL(tauFixed)[p];
    double Z=0.0;
    for (ii = 0; ii < NumActive; ii++) {
      if (Rf_length(tauFixed) >= p * 2) {
        TFL = REAL(tauFixed)[OrderedActive[ii]];
        TFU = REAL(tauFixed)[OrderedActive[ii]];
      }
      if (PickMe >= NumActive*(NumActive+1)/2) {
        Rf_error("Error: NumActive=%d, PickMe=%d!\n", NumActive, PickMe);
      }
      if (Temperature == 1.0) {
        //    TFL =  PBo + FC * Low <  PBo + FC * High = TFU
        Low = (TFL - PropBeta[ii]) / (SqrtScal *rIChol[PickMe]);
        High = (TFU - PropBeta[ii]) / (SqrtScal*rIChol[PickMe]);
        Z = rTruncNorm2(Low, High);
        for (jj = ii; jj < NumActive; jj++) {
          PropBeta[jj] += Z * SqrtScal * rIChol[PickMe];  PickMe++;
        }
      } else {
        SqT = sqrt(Temperature);
        //    TFL =  PBo + FC * Low <  PBo + FC * High = TFU
        Low = (TFL - PropBeta[ii]) / (SqrtScal *SqT* rIChol[PickMe]);
        High = (TFU - PropBeta[ii]) / (SqrtScal*SqT*rIChol[PickMe]);
        Z = rTruncNorm2(Low, High);
        for (jj = ii; jj < NumActive; jj++) {
          if (PickMe >= NumActive*(NumActive+1)/2) {
            Rf_error("PickMe: No, we can't do this!\n");
          }
          PropBeta[jj] += Z * SqT*SqrtScal * rIChol[PickMe];  PickMe++;
        }
      }
    }    
  }  
  return(1);
}

int BayesSpikeCL::FillPropBetaFromBeta() {
  int ii;
  for (ii = 0; ii < NumActive; ii++) {
    PropBeta[ii] = REAL(sBeta)[OrderedActive[ii]];
  }
  return(1);
}

int BayesSpikeCL::PrepareForRegression() {
   int oVerbose = this->Verbose - 3;
   if (oVerbose > 1) {
     Rprintf("-------------------------------------------------------------------\n");
     Rprintf("-- BayesSpikeGibbs.cpp::PrepareForRegression(). \n");
     Rprintf("-- PrepareForRegression, NumberNew = %d, NumActive=%d, Making a decision.\n", NumberNew, NumActive);
     R_FlushConsole();
     Rprintf("  --  We have NumActive = %d, NumberNew=%d, MaxInvert, MaxLargeSizeSquares = %d. \n",
       NumActive, NumberNew, MaxInvert, MaxLargeSizeSquares); R_FlushConsole();
   }
   if (NumActive > MaxLargeSizeSquares && Verbose >= 1) {
     Rprintf("Uh, oh, NumActive = %d, But MaxLargeSizeSquares=%d, we will have to do something drastic.\n",
       NumActive, MaxLargeSizeSquares); R_FlushConsole();
   }
   if (NumActive <= 0) {
     return(0);
   }
   if ((rSmallXtX == NULL || NumberNew > 0 || dfTNoise > 0) && NumActive <= MaxInvert && NumActive <= MaxLargeSizeSquares) {
     if (oVerbose> 1) {
       Rprintf("    PrepareForRegression(): Will Fill Small XtX with NumberNew = %d, dfTNoise=%d, NumActive=%d\n",
         NumberNew, dfTNoise, NumActive); R_FlushConsole();
       if (rSmallXtX == NULL) {
       Rprintf("     PrepareForRegression(): We will also have to allocate SmallXtX. Will we do it? \n"); R_FlushConsole();
       }
     }
     if (MaxLargeSizeSquares >= OnKappaMem) {
       FillSmallXtX(); 
     }
     if (oVerbose> 1) {
       Rprintf("    PrepareForRegression: We filled SmallXtX. \n",
         NumberNew, dfTNoise);  R_FlushConsole();
     }
   } else if  (NumActive > MaxLargeSizeSquares  || NumActive > MaxInvert) {
     if (oVerbose > 1) {
       Rprintf("    PrepareForRegression (Because NumActive = %d < MLSS=%d or MaxInvert=%d: now fill SmallXtResid for partial move. \n",
         NumActive, MaxLargeSizeSquares, MaxInvert); R_FlushConsole();
     }
     FillSmallXtResid();
     if (oVerbose > 1) {
       Rprintf("    PrepareForRegression: Complete FillSmallXtResid() -- Ready to do MaxInvert version now returning. \n"); R_FlushConsole();
     }
     return(3);
   }
   if (oVerbose > 1) {
     Rprintf("    Prepare For regression: We have NumActive less than resid so we are doing typical inverse. \n"); R_FlushConsole();
   }
   int OnRC = 0;
   double Factor;
   if (MaxLargeSizeSquares >= NumActive && (NumberNew >= 0 || dfTNoise > 0)) {
     if (oVerbose > 1) {
       Rprintf("PrepareForRegression, iFirstRandom = %d, go to town\n", iFirstRandom);
       Rprintf("PrepareForRegression, length of tauFixed = %d\n", 
         Rf_length(tauFixed));
       R_FlushConsole();
     }
     int FillCount = NumActive * (NumActive+1) / 2;
     if (NumActive > MaxLargeSizeSquares) {
       Rprintf("**  PrepareForRegression: Well we should be doing CopyIn2 version. NumActive = %d, but MaxLargeSizeSquares=%d\n",
         NumActive, MaxLargeSizeSquares);
       Rprintf("**   This is going to cause this function to fail to fill. \n");
       Rprintf("**   We really should solve this issue. \n"); R_FlushConsole();
       Rprintf("**   But It isn't solved yet, and you should come here.\n");
       R_FlushConsole();
       //Rf_error("**  MaxLargeSizeSquares Problem is not yet Solved.\n"); 
     }
     int One = 1;
     if (NumActive == 1) {
       if (oVerbose >= 1) {
         Rprintf("**    PrepareForRegression(): there is only one Numactive, filling from Small to rIChol. \n");
         R_FlushConsole();
       }
       if (rIChol == NULL) {
         Rprintf("**     ERROR ISSUE: rIChol is NULL!. \n"); R_FlushConsole(); 
       }
       if (rSmallXtX == NULL) {
         Rprintf("**      Error issue: rSmallXtX is NULL! \n"); R_FlushConsole();
       }
       if (oVerbose >= 1)  {
         Rprintf("**    PrepareForRegression(): note: rSmallXtX[0] is %f. \n", rSmallXtX[0]); R_FlushConsole();
         Rprintf("**    PrepareForRegression(): note: rIChol[0] is %f. \n", rIChol[0]); R_FlushConsole();
       }
       rIChol[0] = rSmallXtX[0];
       if (oVerbose >= 1) {
         Rprintf("**    PrepareForRegression(): One Element of rIChol[0] is filled. \n");
         R_FlushConsole();
       }       
     } else if (NumActive <= MaxLargeSizeSquares) {
       F77_CALL(dcopy)(&FillCount, rSmallXtX, &One, rIChol, &One);
     }
     int CI = 0; OnRC = 0;
     if ((iFirstRandom > 0 || (
       sOnTau == NULL || Rf_isNull(sOnTau) || Rf_length(sOnTau) <= 0)) &&
       NumActiveFixed > 0) {
       int OnShrinkFixed = 0;
       if (oVerbose >= 1) {
         Rprintf("**    PrepareForRegression(): About to loop through NumActiveFixed=%d, Start OnShrinkFixed=%d/%d\n", 
           NumActiveFixed, OnShrinkFixed, LengthNoShrinkFixed);
         R_FlushConsole();
       }
       for (CI = 0; CI < NumActiveFixed; CI++) {
         Factor = -1.0;
         if (oVerbose >= 1) {
            Rprintf("**       ** PrepareForRegression(): and CI=%d/%d to test LNSF. \n",
              CI, NumActiveFixed); R_FlushConsole();
         }
         if (LengthNoShrinkFixed > 0 && NoShrinkFixed != NULL && NoShrinkFixedPrior != NULL)  {
           if (oVerbose >= 1) {
             Rprintf("PrepareForRegression: We are on CI=%d, looking through NoShink with OnShrinkFixed=%d/%d. OrderedActive[%d]=%d\n",
               CI, OnShrinkFixed, LengthNoShrinkFixed, CI, OrderedActive[CI]);  R_FlushConsole();
           }
           while(OnShrinkFixed < LengthNoShrinkFixed &&
             NoShrinkFixed[OnShrinkFixed] < OrderedActive[CI]) {
             if (oVerbose >= 1) {
               Rprintf("*************************  Zoom OnShrinkFixed Forward +1. from %d/%d. \n",
                  OnShrinkFixed, LengthNoShrinkFixed);  R_FlushConsole();
             }
             OnShrinkFixed++;  
           }
           if (OnShrinkFixed < LengthNoShrinkFixed &&
             NoShrinkFixed[OnShrinkFixed] == OrderedActive[CI]
           )  {
             if (NoShrinkFixedPrior == NULL) {
               Rf_error("UhOh: Never Set NoShrinkFixed Prior!\n"); R_FlushConsole();
             }
             if (oVerbose >= 1) {
               Rprintf("*************************  We hit OnShrinkFixed=%d/%d now to create Factor from NoShrinkFixedPrior. \n",
                  OnShrinkFixed, LengthNoShrinkFixed);  R_FlushConsole();
             }
             Factor= REAL(sOnSigma)[0] / NoShrinkFixedPrior[OnShrinkFixed];
             if (oVerbose >= 1) {
               Rprintf("************************ PrepareForRegression() after OnShrinkFixed: Factor set to %f. \n", Factor); R_FlushConsole();
             }
           } 
        }
        if (oVerbose >= 1) {
          Rprintf("**        ** PrepareForRegression: We fixed all of LengthNoShrinkFixed: At This point Factor=%f \n", Factor);
          R_FlushConsole();
        }
        if (Factor < 0.0) {
           if (Rf_length(tauFixed) == 1  || 
            Rf_length(tauFixed) < OrderedActive[CI]+1) {
            if (REAL(tauFixed)[0] > 0.0) {
            Factor = fabs(REAL(sOnSigma)[0] / REAL(tauFixed)[0]);
            } else {
            Factor = fabs(1.0 / REAL(tauFixed)[0]);
            }
           }  else if (OrderedActive[CI] < Rf_length(tauFixed)) {
            if (REAL(tauFixed)[OrderedActive[CI]] > 0.0) {
              Factor = fabs(REAL(sOnSigma)[0] / REAL(tauFixed)[ OrderedActive[CI]]);
            } else {
              Factor = fabs(1.0 / REAL(tauFixed)[OrderedActive[CI]]);
            }
           }
        }
        if (oVerbose >= 1) {
          Rprintf("**        ** PrepareForRegression: After less than zero step, Factor set to %f. \n", Factor);
        }
          if (OnRC >= FillCount || OnRC >= 
            NumActiveFixed * ( NumActiveFixed+1) / 2 + 
            (NumActive-NumActiveFixed) * NumActive) {
            Rf_error("]] BayesSpikeGibbs.cpp:::PrepareForRegression(): on fixed: OnRC = %d, FillCount = %d, NumActive = %d, CI = %d, OKM = %d\n",
               OnRC, FillCount, NumActive, CI, OnKappaMem);
          }
          if (OnRC >= NumActive*(NumActive+1)/2) {
            Rprintf("BayesSpikeGibbs() Error: OnRC=%d, CI=%d/%d  and NumActive=%d, but we're going to create a bug. \n",
              OnRC, CI, NumActiveFixed, NumActive); R_FlushConsole();
          }
          rIChol[OnRC] += Factor;
          OnRC += NumActive - CI;
       }
      }
      int St = iFirstRandom;
      int jj = 0; int ii = 0; int k=0;
      if (oVerbose > 1) {
        Rprintf("]] BayesSpikeGibbs.cpp:::PrepareForRegression(): Onto Random variables\n"); R_FlushConsole();
      }
      CI = NumActiveFixed; 
      if (NumActiveFixed < NumActive && tauEndList != NULL && !Rf_isNull(tauEndList) && 
        Rf_length(tauEndList) > 0  && sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) > 0) {
         for(jj = 0; jj < Rf_length(tauEndList); jj++) {
           if (jj > 0) {St =ANINT(tauEndList, jj-1)+1;}
           if (REAL(sOnTau)[jj] > 0) {
              k = ANINT(tauEndList, jj) - St + 1;
              if (REAL(sOnTau)[jj] > ThisMaxTau) {  
                REAL(sOnTau)[jj] = ThisMaxTau;
              }
              Factor = REAL(sOnSigma)[0] / REAL(sOnTau)[jj];
              for (ii = 0; ii < k; ii++) {
                if (OnRC >= FillCount) {
                 Rprintf("]] BayesSpikeGibbs.cpp:::PrepareForRegression() Well, we can't be doing this OnRC = %d \n", OnRC); R_FlushConsole();
                 Rf_error("]] PrepareForRegression: on tau: OnRC = %d, FillCount = %d, NumActive = %d, CI = %d, OKM = %d, jj = %d, ii = %d\n",
                   OnRC, FillCount, NumActive, CI, OnKappaMem, jj, ii);
                }
                rIChol[OnRC] += Factor;
                OnRC += NumActive - CI; CI++;
              }        
           }
         }     
      }
      NumberNew = 1;
   }
   if (oVerbose > 1) {
     Rprintf("BayesSpikeGibbs.cpp:::PrepareForRegression(): Now we are ready to do giant inverse, NumActive=%d. MaxInvert=%d\n",
       NumActive, MaxInvert); R_FlushConsole();
   }
   return(1);
}

///////////////////////////////////////////
//  int BayesSpikeCL::UpdateTNoise()
//
//  Tries to calculate t-weights for each datapoint if one believes noise
//   distribution is t-distributed
//
//  Consider that  Epsilon_i ~ Z / sqrt(V)*sigma for  Z rnorm(0,1)
//    and V has inverse chi squred density 
//    
//
//  f(Epsilon|V, sigma) * p(V) propto  exp( - Epsilon_i^2 / (2 sigma^2 V)) / sqrt( 2 * sigma^2 * V) * V^(-TDF/2-1) * exp(-1/2V)
//    Thus V has a posterior given Epsilon  V sim  (1 + Epsilon^2/sigma^2) * Inverse-ChiSquared(1 + TDF)
//    
//
int BayesSpikeCL::UpdateTNoise() {
  int ii;
  int One = 1;
  double Resid = 0.0;
  double iSigmaSq;
  if (Verbose >= 5) {
    Rprintf("BayesSpikeGibbs.cpp:::UpdateTNoise() Updating with dfTNoise = %f \n", dfTNoise); R_FlushConsole();
  }
  if (R_isnancpp(REAL(sOnSigma)[0]) || REAL(sOnSigma)[0] <= 0) {
    iSigmaSq = 1.0;
  } else {
    iSigmaSq = 1.0 / REAL(sOnSigma)[0];
  }
  double dfTNoise = this->dfTNoise;
  if (dfTNoise <= 0.0) {
    Rf_error("UpdateTNoise: Hey, We shouldn't be here with a negative number!");
  }
  if (iiWeight  == NULL) {
    RMemGetD(iiWeight, "iiWeight", n);
  }
  if (iWeightedXtX == NULL) {
    if (Verbose >= 5) {
      Rprintf("BayesSpikeGibbs.cpp:::UpdateTNoise() iWeightedXtX Is NULL Allocating length %d=OnKappaMem\n", OnKappaMem); R_FlushConsole();
    }
    RMemGetS(iWeightedXtX, "iWeightedXtX", OnKappaMem);
    if (Verbose >= 5) {
      Rprintf("BayesSpikeGibbs.cpp:::UpdateTNoise() Finished Allocating iWeightedXtX length %d=OnKappaMem\n", OnKappaMem); R_FlushConsole();
    }
    for (int iti = 0; iti < OnKappaMem; iti++) {
      iWeightedXtX[iti] = (short int) 0;
    }
  }
  double OneD = 1.0;   char Trans = 'N';
  double NegOneD = -1.0;
  F77_CALL(dcopy)(&n, REAL(sY), &One, iiWeight, &One);
  F77_CALL(dgemv)(&Trans, &n, &p, &NegOneD, 
    REAL(sX), &n, REAL(sBeta), &One, &OneD, iiWeight, &One);
  if (Temperature == 1.0) {
  for (ii = 0; ii < n; ii++) {
    Resid =  iiWeight[ii] * iiWeight[ii] * iSigmaSq;
    iiWeight[ii] = Rf_rgamma( (dfTNoise + 1) * .5,1.0) / 
      (( Resid + dfTNoise ) * .5);  
    if (R_isnancpp(iiWeight[ii])) {
      iiWeight[ii] = 1.0;
    }
  }  
  } else {
    for (ii = 0; ii < n; ii++) {
      Resid =  iiWeight[ii] * iiWeight[ii] * iSigmaSq;
      iiWeight[ii] = Rf_rgamma( (dfTNoise + 1) * .5 * invTemperature,1.0) / 
        (( Resid + dfTNoise ) * .5 * invTemperature);  
      if (R_isnancpp(iiWeight[ii])) {
        iiWeight[ii] = 1.0;
      }
  }    
  }
  if (Verbose >= 5) {
    Rprintf("BayesSpikeGibbs.cpp:::UpdateTNoise()  We just updated iiWeight \n");
    R_FlushConsole();
  }
  nWeight = F77_CALL(dasum)(&n, iiWeight, &One);
  if (Verbose >= 5) {
    Rprintf("BayesSpikeGibbs.cpp:::UpdateTNoise() Will ReweightOnlyActiveXtX. \n");
    R_FlushConsole();
  }
  if (OnKappaS >= 1) {
    if (iWeightedXtX != NULL && OnKappaS >= 1) {
      for (int jj = 0; jj < OnKappaS; jj++) {
        iWeightedXtX[jj] = (short int) 0;
      }
    }
  }
  if (OnKappaS >= 1) {
    for (int jj = 0; jj < OnKappaS; jj++) {
      if (iWeightedXtX[jj] != (short int) 0) {
        Rprintf("BayesSpikeGibbs.cpp:::UpdateTNoise: What up, iWeightedXtX[jj=%d] = %d why?\n",
          jj, iWeightedXtX[jj]); R_FlushConsole();
      }
    }
  }
  ReweightOnlyActiveXtX();
  /*
  if (tt <= 2) {
    Rprintf("BayesSpikeGibbs.cpp:::UpdateTNoise, warning doing QuickCheck every step. (tt=%d)\n",tt);
  }
  int CountBad = QuickCheckWeightXtX();
  if (CountBad > 0) {
    Rprintf("*** BayesSpikeGibbs.cpp:::UpdateTNoise Error AfterReweightOnlyActive, tt=%d, we got QuickCheckWeightXtX, CountBad = %d \n", tt, CountBad); R_FlushConsole();
    Rprintf("*** Seriously, we just ran UpdateTNoise, this is damn weird!  OnKappaS = %d", OnKappaS); R_FlushConsole();
    Rf_error("***  Resize QuickCheckWeightXtX Fail. \n");
  } */
  if (Verbose >= 5) {
    Rprintf("BayesSpikeGibbs.cpp:::UpdateTNoise() Completed ReweightOnlyActiveXtX. \n");
    R_FlushConsole();
  }
  if (tauEndList != NULL && !Rf_isNull(tauEndList) && Rf_length(tauEndList)) {
    if (Verbose >= 5) {
      Rprintf("BayesSpikeGibbs.cpp:::UpdateTNoise() ReweightEigenValues. \n");
      R_FlushConsole();
    }
    ReweightEigenvalues();
  }
  if (Verbose >= 4) {
    Rprintf("BayesSpikeGibbs.cpp:::UpdateTNoise() UpdateXtY, YSq and FreshXtResid \n");
    R_FlushConsole();
  }

  UpdateXtY();
  SetupYSq();
  if (Verbose >= 4) {
    Rprintf("BayesSpikeGibbs.cpp:::UpdateTNoise()  UpdatingFreshXtResid.\n");
    R_FlushConsole();
  }
  UpdateFreshXtResid();
  if (Verbose >= 4) {
    Rprintf("BayesSpikeGibbs.cpp:::UpdateTNoise() We supposedly Updated FreshXtResid. \n");
    R_FlushConsole();
  }
  if (StartRecordingiiWeight >= 0 && tt >= StartRecordingiiWeight && SumiiWeight != NULL) {
    for (int iti = 0; iti < n; iti++) {
      SumiiWeight[iti] += iiWeight[iti];  
    }
    CountiiWeight++;
  }
  return(1);
}


//////////////////////////////////////////////////
//  "DoRecord"
//    This is an instruction vector with 7 positions,
//   This controls whether the CODA chains recorded by sampler
//   record various items
//    Pos 1 : Record Fixed Coefficients
//    Pos 2 : Record Random Coefficients (those attached to tau_k parameters)
//    Pos 3 : Record tau_k parameters
//    Pos 4 : Record Sigma:1 noise value
//    Pos 5 :  Record PiA estimates (for either Fixed/Random or both)
//    Pos 6 : Record posterior probability of Fixed coefficients in model
//    Pos 7 : Record posterior probability of Random coefficients in model
void BayesSpikeCL::set_DoRecord(SEXP DoRecord_) {
  if (Rf_isNull(DoRecord_) || Rf_length(DoRecord_) != 7) {
    Rf_error("set_DoRecord: Has length %d, but must be 7", Rf_length(DoRecord_));
  }
  //if (!Rf_isNull(sOnTau)  && iFirstRandom < 0) {
  //  Rf_error("set_DoRecord: Must set FirstRandom first!");
  //}
  if (ANINT(DoRecord_, (1)) == 1 && 
    (tauEndList == NULL || Rf_isNull(tauEndList) || Rf_length(tauEndList) <= 0)) {
    
    Rprintf("set_DoRecord: Please, to record random Must not be set to Null, we're blanking out.\n");
    if (Rf_isReal(DoRecord_)) {
      REAL(DoRecord_)[1] = 0.0;
      REAL(DoRecord_)[6] = 0.0;
    }
    if (Rf_isInteger(DoRecord_)) {
      INTEGER(DoRecord_)[1] = 0;  INTEGER(DoRecord_)[6] = 0;
    } else if (Rf_isReal(DoRecord_)) {
      REAL(DoRecord_)[1] = 0.0;  REAL(DoRecord_)[6] = 0.0;
    }
  }
  if (Verbose >= 2) {
    Rprintf("set_DoRecord, setting DoRecord to a vector\n"); R_FlushConsole();
  }
  DDelete(RDoRecord, "RDoRecord");  DoRecord = R_NilValue;
  if (Rf_isInteger(DoRecord_)) {
    RDoRecord = new AObject(DoRecord_);
    DoRecord = RDoRecord->asSexp();
  } else if (Rf_isReal(DoRecord_)) {
    RDoRecord = new AObject(Rf_allocVector(INTSXP, Rf_length(DoRecord_)));
    DoRecord = RDoRecord->asSexp();
    for (int ii = 0; ii < Rf_length(DoRecord); ii++) {
      INTEGER(DoRecord)[ii] = ((int) REAL(DoRecord_)[ii]); 
    }
  }
  int NeedRecord = CountRecord();
  if (CodaTable == NULL || Rf_isNull(CodaTable) || Rf_length(CodaTable) <= 0) {
    if (Verbose >= 2) {
      Rprintf("set_DoRecord: CodaTable Not Setup.\n"); R_FlushConsole();
    }
  }
  if (CodaTable != NULL && !Rf_isNull(CodaTable) &&  Rf_length(CodaTable) >= 1 &&
    Rf_length(Rf_getAttrib(CodaTable, R_DimSymbol)) == 2 &&
    INTEGER(Rf_getAttrib(CodaTable, R_DimSymbol))[1] != NeedRecord) {
    Rf_error("set_DoRecord: You're going to have to readjust Codatable to dim %d.\n",
      NeedRecord);  
  }
  if (Verbose >= 2) {
    Rprintf("set_DoRecord: Done setting up.\n"); R_FlushConsole();
  }
}


////////////////////////////////////////
//  CountRecord calculates the width of CODA MCMC output chains
//    given the many options selected by DoRecord vector
//
//
int BayesSpikeCL::CountRecord() {
  if (Rf_isNull(DoRecord) || DoRecord == NULL) {
    Rprintf("CountRecord, error, DoRecord is NULL!\n"); R_FlushConsole();
    Rf_error("Count Record: ERROR!\n");
  }
  if (!Rf_isInteger(DoRecord)) {
    Rprintf("CountRecord, error, Do Record is not INTEGER!\n"); R_FlushConsole();
    Rf_error("Count Record: ERROR!\n");
  }
  if (Rf_isNull(DoRecord)|| Rf_length(DoRecord) != 7) {
    return(p);
  }
  int Count = 0;
  if (ANINT(DoRecord, 0) > 0) {
    Count += p;
  }
  if (Verbose >= 2) {
    Rprintf("CountRecord: checking out sOnTau.\n");
  }
  if (ANINT(DoRecord, 1) > 0) {
    if (sOnTau == NULL || tauEndList == NULL ||
      Rf_isNull(sOnTau) || Rf_length(sOnTau) <= 0 ||
      Rf_isNull(tauEndList) || Rf_length(tauEndList) <= 0) {
      Count += 0;
      if (Rf_isReal(DoRecord)) {
        REAL(DoRecord)[1] = 0.0;
      } else if (Rf_isInteger(DoRecord)) {
        INTEGER(DoRecord)[1] = 0;
      }
    } else if (Rf_isNull(tauEndList)) {
      Rf_error("CountRecord: DoRecord[2] is 1 but there is no tauEndList");
    } else {
      Count += Rf_length(tauEndList);
    }
  }
  if (ANINT(DoRecord, 2) > 0) {
    if (Rf_isNull(sOnSigma)) {
      Rf_error("CountRecord: DoRecord[4] is 1 but there is no sOnSigma");
    }
    Count += Rf_length(sOnSigma);
  }  
  if (ANINT(DoRecord, 3) > 0) {
    if (Rf_isNull(sOnPiA)) {
      Rf_error("CountRecord: DoRecord[3] is 1 but there is no sOnPiA");
    }
    Count += Rf_length(sOnPiA);
  }  
  if (Verbose >= 3) {
    Rprintf("CountRecord: About to setup tauFixed\n"); R_FlushConsole();
  }
  if (ANINT(DoRecord, 4) > 0) {
    if (Rf_isNull(tauFixed)) {
      Rf_error("CountRecord: DoRecord[5] is 1 but there is no tauFixed");
    }
    Count += Rf_length(tauFixed);     // tauFixed[0:iFirstRandom]
  } 
  if (ANINT(DoRecord, 5) > 0) {
    if (sOnTau == NULL || tauEndList == NULL || 
      Rf_isNull(sOnTau) || Rf_length(sOnTau) <= 0 || Rf_isNull(tauEndList)) {
      Count += p;
    } else if (iFirstRandom < 0 || iFirstRandom > p) {
      Rf_error("CountRecord: DoRecord[6] is 1 but there is no iFirstRandom");
    } else if (iFirstRandom > 0 && iFirstRandom <= p) {
      Count += iFirstRandom;  // ProbFixed
    } else {
      Rf_error("CountRecord: Weird:, what's going on for DoRecord[5]?\n");
    }
  } 
  if (ANINT(DoRecord, 6) > 0) {
    if (sOnTau == NULL || tauEndList == NULL || 
      Rf_isNull(sOnTau) || Rf_length(sOnTau) <= 0 ||
      Rf_isNull(tauEndList) || Rf_length(tauEndList) <= 0) {
      if (Rf_isInteger(DoRecord)) {
        INTEGER(DoRecord)[6] = 0;
      } else if (Rf_isReal(DoRecord)) {
        REAL(DoRecord)[6] = 0.0;
      }
      //Rf_error("CountRecord: DoRecord[7] is 1 but there is no tauEndList");
    } else if (Rf_isNull(sOnTau) || Rf_isNull(tauEndList) || Rf_length(sOnTau) <= 0 ||
      Rf_length(tauEndList) <= 0) {
    } else if (!Rf_isNull(tauEndList) && Rf_length(tauEndList) > 0) {
      Count += Rf_length(tauEndList);  // ProbFixed // ProbTau
    }
  } 
  return(Count);
} 


//////////////////////////////////////
//  Update estimate of new Sigma^2 parameter ("sOnSigma")
//   This represents the noise distribution if Gaussian
//
int BayesSpikeCL::UpdateSigma() {  
  int oVerbose = this->Verbose -1;
  BackSigma = REAL(sOnSigma)[0];
  // sum( (Y-Xb)^2 ) = sum(Y(Y-Xb)) -  sum(bX^T(Y-Xb))
  if (oVerbose > 3) {
    Rprintf("  On UpdateSigma \n"); R_FlushConsole();
  }
  SetupYResidSq();
  //if (!R_finite(YResidSq) || YResidSq > 4 * YSq) {
  //  Rprintf("  Messed up, YResidSq = %f, but YSq = %f \n", YResidSq, YSq);
  //  YResidSq = YSq;
  //}
  if (RsOnSigma == NULL || Rf_isNull(sOnSigma)) {
    Rf_error("UpdateSigma: Sigma is null!");
  }

  if (Rf_length(SigmaPrior) != 2 && Verbose > 3) {
    Rprintf("Length of Sigma is %d\n", Rf_length(SigmaPrior));
  }
  if (oVerbose > 3) {
    Rprintf("SigmaPrior = "); R_FlushConsole();
    int ii;
    for (ii = 0; ii < Rf_length(SigmaPrior); ii++) {
      Rprintf(" %f, ", REAL(SigmaPrior)[ii]); R_FlushConsole();
    }
    Rprintf("\n and REAL(sOnSigma)[0] = %f\n", REAL(sOnSigma)[0]); R_FlushConsole();
  }
  
  if (RSigmaPrior == NULL || SigmaPrior == NULL || 
    Rf_isNull(SigmaPrior) || Rf_length(SigmaPrior) < 2 ||
    REAL(SigmaPrior)[1] < 0)  {
    if (Verbose >= 4) {
      Rprintf("BayesSpikeGibbs.cpp:::UpdateSigma(tt=%d), leave without update. \n",tt);
      if (RSigmaPrior == NULL  || SigmaPrior == NULL || 
        Rf_isNull(SigmaPrior)) {
        Rprintf("    SigmaPrior is NULL\n");
      } else if (Rf_length(SigmaPrior) < 2) {
        Rprintf("   Length SigmaPrior is %d\n", Rf_length(SigmaPrior));
      } else {
        Rprintf("   SigmaPrior[1] = %f\n", REAL(SigmaPrior)[1]);
      }
      R_FlushConsole();
    }
    return(1);
  }
  if (this->nWeight <= 0 || this->iiWeight == NULL) {
    this->nWeight = (double) this->n;
  }
  if (Verbose >= 5) {
    Rprintf("BayesSpikeGibbs.cpp::UpdateSigma(tt=%d): YResidSq=%.3f, nWeight=%.3f, SigmaPrior[0]=%.4f, SigmaPrior[1]=%.4f.\n",
      tt, YResidSq, (double) nWeight, REAL(SigmaPrior)[0], REAL(SigmaPrior)[1]);
  } 
  if (Temperature == 1.0) {
  REAL(sOnSigma)[0] = (YResidSq + 
    REAL(SigmaPrior)[1] * REAL(SigmaPrior)[0]) /
    Rf_rchisq(this->nWeight + REAL(SigmaPrior)[0]);
  } else {
    REAL(sOnSigma)[0] = (YResidSq + 
      REAL(SigmaPrior)[1] * REAL(SigmaPrior)[0]) * .5 * invTemperature /
      Rf_rgamma( ((1.0-Temperature+ .5*(this->nWeight + REAL(SigmaPrior)[0])) * invTemperature), 1.0 );  
  }
  if (R_isnancpp(REAL(sOnSigma)[0])) {
    REAL(sOnSigma)[0] = 1.0;
    if (oVerbose > -1) {
      Rprintf("Error: we samplesOnSigma NAN, YResidSq = %f \n", YResidSq);
      R_FlushConsole();
    }
  }
  if (!R_finite(REAL(sOnSigma)[0])) {
    Rprintf("Error UpdateSigma: We Got infinite OnSigma = %f, Temperature=%f, YResidSq=%f\n", 
      REAL(sOnSigma)[0], Temperature, YResidSq); R_FlushConsole();
    Rprintf("Error UpdateSigma: We have nWeight = %f, SigmaPrior[0] = %f, SigmaPrior = [1]. \n",
      nWeight, REAL(SigmaPrior)[0], REAL(SigmaPrior)[1]); R_FlushConsole();
    int CountNanSigma = 0;  int CountHugeSigma = 0;
    REAL(sOnSigma)[0] = BackSigma;
    for (int ii = 0; ii < p; ii++) {
      if (R_isnancpp(REAL(sBeta)[ii]) || !R_finite(REAL(sBeta)[ii])) {
         CountNanSigma++;  
         Rprintf("Error UpdateSigma: We have Beta[%d] = %f!\n", ii, REAL(sBeta)[ii]); R_FlushConsole();
      } else if (fabs(REAL(sBeta)[ii]) >= 999999) {
         CountHugeSigma++;
         Rprintf("Weird Updatesigma, has Beta[%d] = %f?\n", ii, REAL(sBeta)[ii]); R_FlushConsole();
      }
    }
    if (CountNanSigma  > 0 || CountHugeSigma > 0) {
      Rf_error("Error UpdateSigma: Investigate the problem with Beta\n");
    }
    double totalSigma = 0.0;  double OnMe;  double EstMe;  int One = 1; int TotalCount = 0;
    double tryTotalSigma = 0.0;
    int LogFlag = 0;
    for (int ii = 0; ii < n; ii++) {
      OnMe = REAL(sY)[ii];
      tryTotalSigma = -1.0;
      if (R_isnancpp(OnMe) || !R_finite(OnMe)) {
      } else {
        EstMe = F77_CALL(ddot)(&p, REAL(sX) + ii, &n, REAL(sBeta), &One);
        if (R_isnancpp(EstMe) || !R_finite(EstMe)) {
          EstMe = 0.0;
        } 
        if (!R_isnancpp(OnMe-EstMe) && R_finite(OnMe-EstMe)) {
          OnMe = (OnMe-EstMe) * (OnMe-EstMe);
          if (!R_finite(OnMe+totalSigma)) {
            if (LogFlag == 0) {
              tryTotalSigma = log(OnMe) + log(totalSigma/OnMe + 1.0);
              LogFlag = 1;
            } else {
              tryTotalSigma = log(OnMe) + log(exp(totalSigma) / OnMe+1.0);
            }
          } else {
            tryTotalSigma = totalSigma+ OnMe;
          }
        }
      }
      if (R_isnancpp(tryTotalSigma) || !R_finite(tryTotalSigma)) {
        EstMe = -1;
        Rprintf("-- Oops on ii = %d, totalSigma was %f, tryTotalSigma = %f, REAL(sY)[%d]=%f, EstMe=%f\n",
          ii, totalSigma, tryTotalSigma, ii, REAL(sY)[ii], EstMe); R_FlushConsole();
      } else if (tryTotalSigma < 0) {
      } else {
        totalSigma = tryTotalSigma;  TotalCount++;
      }
    }
    Rprintf("Updatesigma: We did retotals, totalSigma = %f, TotalCount = %d\n",
      totalSigma, TotalCount); R_FlushConsole();
    if (LogFlag == 1) {
      REAL(sOnSigma)[0] = exp(totalSigma/TotalCount);
    } else {
      REAL(sOnSigma)[0] = totalSigma / TotalCount;
    }
    if (!R_finite(REAL(sOnSigma)[0])) {
      Rprintf("UpdateSigma: Still Infinite Sigma!\n"); R_FlushConsole();
      REAL(sOnSigma)[0] = 1.0;
    }
  }
  if (oVerbose > 3) {
    Rprintf(" Finishing UpdateSigma, with Sigma = %f\n", REAL(sOnSigma)[0]); R_FlushConsole();
  }
  return(1);
}


int BayesSpikeCL::UpdatePiA() {
  if (PiAPrior == NULL || Rf_isNull(PiAPrior) || Rf_length(PiAPrior) < 2) {
    return(-1);  
  }
  int isFixed = -1;  int isRandom = -1;
  LenFixed = -1;  LenRandom = -1; 
  if (sOnTau == NULL || Rf_isNull(sOnTau) || Rf_length(sOnTau) <= 0 || 
    tauEndList == NULL || Rf_isNull(tauEndList) || Rf_length(tauEndList) <= 0) {
    isFixed = 1; LenFixed = p;
  } else if (iFirstRandom > 0) {
    isFixed = 1; LenFixed = iFirstRandom; 
  } else { isFixed = 0; LenFixed = 0; }
  if (sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) > 0 &&
    tauEndList != NULL && !Rf_isNull(tauEndList) && Rf_length(tauEndList) > 0) {
    isRandom = 1;  LenRandom = Rf_length(sOnTau);
  } else { isRandom = 0; LenRandom = 0; }
  
  CountFreeFixed = 0; 
  CountFreeRandom = 0;
  OnShrinkFixed = 0;
  OnShrinkRandom = 0;  

  if (Verbose >= 8) {
    Rprintf("uuu  BayesSpikeGibbs.cpp:::UpdatePiA(tt=%d) Start, LenFixed = %d, LenRandom=%d\n", tt, LenFixed, LenRandom); R_FlushConsole();
  }
  double OnMe = 0.0;
        
  int AOkProbFixed = 1;  int AOkProbRandom = 1;
  if (rPriorProbFixed == NULL || Rf_isNull(rPriorProbFixed->asSexp()) || 
    Rf_length(rPriorProbFixed->asSexp()) > 0) {
    if (LengthNoShrinkFixed >= LenFixed) {
      AOkProbFixed = 0; CountFreeFixed = 0;
    } else if (LengthNoShrinkFixed > 1) {
      AOkProbFixed = 1; CountFreeFixed = LenFixed-LengthNoShrinkFixed;
    } else {
      AOkProbFixed = 1;   CountFreeFixed = LenFixed;
    }
  } else if (rPriorProbFixed != NULL && !Rf_isNull(rPriorProbFixed->asSexp() )) {
    if (Rf_length(rPriorProbFixed->asSexp()) == LenFixed) {
     CountFreeFixed = 0; 
     for (int iti = 0; iti < LenFixed; iti++) {
       OnMe = REAL(rPriorProbFixed->asSexp())[iti];
       while(LengthNoShrinkFixed > 0 && OnShrinkFixed < LengthNoShrinkFixed && 
         NoShrinkFixed[OnShrinkFixed] < iti) {
          OnShrinkFixed++;
       }
       if (LengthNoShrinkFixed > 0 && OnShrinkFixed < LengthNoShrinkFixed &&
         NoShrinkFixed[OnShrinkFixed] == iti) {
       } else if (R_isnancpp(OnMe) || R_IsNA(OnMe)  || !R_finite(OnMe) || OnMe < 0.0 || OnMe > 1.0) {
         CountFreeFixed++;
       } 
     }
    } else if (Rf_length(rPriorProbFixed->asSexp()) == 2*LenFixed) {
     CountFreeFixed = 0; 
     for (int iti = 0; iti < LenFixed; iti++) {
       OnMe = REAL(rPriorProbFixed->asSexp())[iti*2];
       while(LengthNoShrinkFixed > 0 && OnShrinkFixed < LengthNoShrinkFixed && 
         NoShrinkFixed[OnShrinkFixed] < iti) {
          OnShrinkFixed++;
       }
       if (LengthNoShrinkFixed > 0 && OnShrinkFixed < LengthNoShrinkFixed &&
         NoShrinkFixed[OnShrinkFixed] == iti) {
       } else if (R_isnancpp(OnMe) || R_IsNA(OnMe)  || !R_finite(OnMe) || OnMe < 0.0 || OnMe > 1.0) {
         CountFreeFixed++;
       } 
     }
    }
    if (CountFreeFixed <= 0) { AOkProbFixed = 0;}
  }   
  if (Verbose >= 8) {
    Rprintf("UpdatePiA: isFixed=%d, isRandom=%d.\n", isFixed, isRandom); R_FlushConsole();
  }
  if (LenRandom <= 0) {
    CountFreeRandom = 0; AOkProbRandom = 1;
  } else if (rPriorProbTau == NULL || Rf_isNull(rPriorProbTau->asSexp())) {
    if (LengthNoShrinkRandom == Rf_length(sOnTau)) {
      AOkProbRandom = 0;  CountFreeRandom = 0; 
    } else if (LengthNoShrinkRandom > 0) {
      AOkProbRandom = 1;
      CountFreeRandom = LenRandom - LengthNoShrinkRandom;
    } else {
      AOkProbRandom = 1;
      CountFreeRandom = LenRandom;
    }
  } else if (rPriorProbTau != NULL && !Rf_isNull(rPriorProbTau->asSexp() ) &&
    Rf_length(rPriorProbTau->asSexp()) ) {
    CountFreeRandom = 0;
    if (Rf_length(rPriorProbTau->asSexp()) == Rf_length(sOnTau)) {
      for (int iti = 0; iti < Rf_length(rPriorProbTau->asSexp()); iti++) {
        OnMe = REAL(rPriorProbTau->asSexp())[iti];
        while(LengthNoShrinkRandom > 0 && OnShrinkRandom < LengthNoShrinkRandom && 
          NoShrinkRandom[OnShrinkRandom] < iti) {
          OnShrinkRandom++;
        }
        if (LengthNoShrinkRandom > 0 && OnShrinkRandom < LengthNoShrinkRandom &&
          NoShrinkRandom[OnShrinkRandom] == iti) {
        } else if (R_isnancpp(OnMe) || R_IsNA(OnMe)  || !R_finite(OnMe) || OnMe < 0.0 || OnMe > 1.0) {
           CountFreeRandom++;
        }  
      }
    } else if (Rf_length(rPriorProbTau->asSexp()) == 2*Rf_length(sOnTau)) {
      for (int iti = 0; iti < Rf_length(rPriorProbTau->asSexp()); iti++) {
        OnMe = REAL(rPriorProbTau->asSexp())[2*iti];
        while(LengthNoShrinkRandom > 0 && OnShrinkRandom < LengthNoShrinkRandom && 
          NoShrinkRandom[OnShrinkRandom] < iti) {
          OnShrinkRandom++;
        }
        if (LengthNoShrinkRandom > 0 && OnShrinkRandom < LengthNoShrinkRandom &&
          NoShrinkRandom[OnShrinkRandom] == iti) {
        } else if (R_isnancpp(OnMe) || R_IsNA(OnMe)  || !R_finite(OnMe) || OnMe < 0.0 || OnMe > 1.0) {
          CountFreeRandom++;
        }  
      }    
    } 
    if (CountFreeRandom <= 0) {AOkProbRandom = 0; }
  }
  if ((LenFixed <= 0 || AOkProbFixed == 0) && (LenRandom <= 0 || AOkProbRandom == 0)) {
    if (Verbose >= 8) {
      Rprintf("uuu  BayesSpikeGibbs.cpp:::UpdatePiA(tt=%d) NoOK Return, LenFixed = %d, LenRandom=%d, AOKProbFixed = %d, AOkProbRandom=%d\n", 
        tt, LenFixed, LenRandom, AOkProbFixed, AOkProbRandom); R_FlushConsole();
      Rprintf("uuu   Does LenFixed less than zero?  AOkProbFixed=0:%d?  Okay, LenRandom <= 0:%d?  and AOkProbRandom == 0:%d?", 
        ((int)LenFixed <= 0), ((int)AOkProbFixed == 0), ((int)LenRandom <= 0), ((int) AOkProbRandom==0));
    }
    return(1);
  }
  //int oVerbose = this->Verbose - 2;
  int CountFixedOn = 0;  int CountRandomOn = 0;
  int ii;
  OnShrinkRandom = 0;  OnShrinkFixed = 0;
  if (sOnTau == NULL || Rf_isNull(sOnTau) || Rf_length(sOnTau) <= 0 ||
    tauEndList == NULL || Rf_isNull(tauEndList) || Rf_length(tauEndList) <= 0) {
    if (AOkProbFixed == 0) {
    } else if (rPriorProbFixed == NULL || Rf_isNull(rPriorProbFixed->asSexp())) {
      for (ii = 0; ii < p; ii++) {
         while (LengthNoShrinkFixed > 0 && OnShrinkFixed < LengthNoShrinkFixed &&
           NoShrinkFixed[OnShrinkFixed] < ii) {
           OnShrinkFixed++;  
         }
         if (LengthNoShrinkFixed > 0 && OnShrinkFixed < LengthNoShrinkFixed &&
           NoShrinkFixed[OnShrinkFixed] == ii) {
         } else if (REAL(sBeta)[ii] != 0.0) { CountFixedOn++; }
      }
    } else if (Rf_length(rPriorProbFixed->asSexp()) == LenFixed) {
       for (ii = 0; ii < p; ii++) {
         OnMe = REAL(rPriorProbFixed->asSexp())[ii];
         while (LengthNoShrinkFixed > 0 && OnShrinkFixed < LengthNoShrinkFixed &&
           NoShrinkFixed[OnShrinkFixed] < ii) {
           OnShrinkFixed++;  
         }
         if (LengthNoShrinkFixed > 0 && OnShrinkFixed < LengthNoShrinkFixed &&
           NoShrinkFixed[OnShrinkFixed] == ii) {
         } else if (R_isnancpp(OnMe) || R_IsNA(OnMe)  || !R_finite(OnMe) || OnMe < 0.0 || OnMe > 1.0) {
           if(REAL(sBeta)[ii] != 0.0) {
             CountFixedOn++;
           }
         }
       }  
    } else if (Rf_length(rPriorProbFixed->asSexp()) == 2*LenFixed) {
       for (ii = 0; ii < p; ii++) {
         OnMe = REAL(rPriorProbFixed->asSexp())[2*ii];
         while (LengthNoShrinkFixed > 0 && OnShrinkFixed < LengthNoShrinkFixed &&
           NoShrinkFixed[OnShrinkFixed] < ii) {
           OnShrinkFixed++;  
         }
         if (LengthNoShrinkFixed > 0 && OnShrinkFixed < LengthNoShrinkFixed &&
           NoShrinkFixed[OnShrinkFixed] == ii) {
         } else if (R_isnancpp(OnMe) || R_IsNA(OnMe)  || !R_finite(OnMe) || OnMe < 0.0 || OnMe > 1.0) {
           if(REAL(sBeta)[ii] != 0.0) {
             CountFixedOn++;
           }
         }
       }  
    }
    if (Verbose >= 8) {
      Rprintf("uuu  BayesSpikeGibbs.cpp:::UpdatePiA(tt=%d) About to set only fixed sOnPiA \n", 
        tt, LenFixed, LenRandom); R_FlushConsole();
    }
    if (Temperature == 1.0) {
      REAL(sOnPiA)[0] = Rf_rbeta( CountFixedOn  + REAL(PiAPrior)[0],
        CountFreeFixed-CountFixedOn  + REAL(PiAPrior)[1]);
    } else {
      REAL(sOnPiA)[0] = Rf_rbeta( 
        ((CountFixedOn +  REAL(PiAPrior)[0]-1.0 + Temperature) * invTemperature),
        ((CountFreeFixed-CountFixedOn + REAL(PiAPrior)[1]-1.0 + Temperature) * invTemperature) );    
    }
    if (MT != NULL) {
      MT->logOddsPi = log(REAL(sOnPiA)[0]) - log(1.0 - REAL(sOnPiA)[0]);
    } 
    return(1); 
  } else {
    OnShrinkFixed = 0;  OnShrinkRandom = 0;
    CountFixedOn = 0;
    if (AOkProbFixed  == 0) {
    } else if (rPriorProbFixed == NULL || Rf_isNull(rPriorProbFixed->asSexp())) {
      for (ii = 0; ii < iFirstRandom; ii++) {
         while (LengthNoShrinkFixed > 0 && OnShrinkFixed < LengthNoShrinkFixed &&
           NoShrinkFixed[OnShrinkFixed] < ii) {
           OnShrinkFixed++;  
         }
         if (LengthNoShrinkFixed > 0 && OnShrinkFixed < LengthNoShrinkFixed &&
           NoShrinkFixed[OnShrinkFixed] == ii) {
         } else if (REAL(sBeta)[ii] != 0.0) { CountFixedOn++; }
      }
    } else if (Rf_length(rPriorProbFixed->asSexp()) == LenFixed) {
       for (ii = 0; ii < iFirstRandom; ii++) {
         OnMe = REAL(rPriorProbFixed->asSexp())[ii];
         while (LengthNoShrinkFixed > 0 && OnShrinkFixed < LengthNoShrinkFixed &&
           NoShrinkFixed[OnShrinkFixed] < ii) {
           OnShrinkFixed++;  
         }
         if (LengthNoShrinkFixed > 0 && OnShrinkFixed < LengthNoShrinkFixed &&
           NoShrinkFixed[OnShrinkFixed] == ii) {
         } else if (R_isnancpp(OnMe) || R_IsNA(OnMe)  || !R_finite(OnMe) || OnMe < 0.0 || OnMe > 1.0) {
           if(REAL(sBeta)[ii] != 0.0) {
             CountFixedOn++;
           }
         }
       }  
    } else if (Rf_length(rPriorProbFixed->asSexp()) == 2*LenFixed) {
       for (ii = 0; ii < iFirstRandom; ii++) {
         OnMe = REAL(rPriorProbFixed->asSexp())[2*ii];
         while (LengthNoShrinkFixed > 0 && OnShrinkFixed < LengthNoShrinkFixed &&
           NoShrinkFixed[OnShrinkFixed] < ii) {
           OnShrinkFixed++;  
         }
         if (LengthNoShrinkFixed > 0 && OnShrinkFixed < LengthNoShrinkFixed &&
           NoShrinkFixed[OnShrinkFixed] == ii) {
         } else if (R_isnancpp(OnMe) || R_IsNA(OnMe)  || !R_finite(OnMe) || OnMe < 0.0 || OnMe > 1.0) {
           if(REAL(sBeta)[ii] != 0.0) {
             CountFixedOn++;
           }
         }
       }  
    }
    CountRandomOn = 0;
    OnShrinkRandom = 0;
    if (AOkProbRandom == 0) {
    } else if (rPriorProbTau == NULL || Rf_isNull(rPriorProbTau->asSexp())) {
      for (ii = 0; ii < LenRandom; ii++) {
        while (LengthNoShrinkRandom > 0 && OnShrinkRandom < LengthNoShrinkRandom &&
          NoShrinkRandom[OnShrinkRandom] < ii) {
          OnShrinkRandom++;
        }
        if (LengthNoShrinkRandom > 0 && OnShrinkRandom < LengthNoShrinkRandom &&
          NoShrinkRandom[OnShrinkRandom] == ii) {
        } else if (REAL(sOnTau)[ii] > 0) {  CountRandomOn++; }
      }
    } else if (Rf_length(rPriorProbTau->asSexp()) == Rf_length(sOnTau)) {
      for (ii = 0; ii < LenRandom; ii++) {
        OnMe = REAL(rPriorProbTau->asSexp())[ii];
        while (LengthNoShrinkRandom > 0 && OnShrinkRandom < LengthNoShrinkRandom &&
          NoShrinkRandom[OnShrinkRandom] < ii) {
          OnShrinkRandom++;
        }
        if (LengthNoShrinkRandom > 0 && OnShrinkRandom < LengthNoShrinkRandom &&
          NoShrinkRandom[OnShrinkRandom] == ii) {
        } else if (R_isnancpp(OnMe) || R_IsNA(OnMe)  || !R_finite(OnMe) || OnMe < 0.0 || OnMe > 1.0) {
           if(REAL(sOnTau)[ii] != 0.0) {
             CountRandomOn++;
           }
        }
      }    
    } else {
      for (ii = 0; ii < LenRandom; ii++) {
        OnMe = REAL(rPriorProbTau->asSexp())[2*ii];
        while (LengthNoShrinkRandom > 0 && OnShrinkRandom < LengthNoShrinkRandom &&
          NoShrinkRandom[OnShrinkRandom] < ii) {
          OnShrinkRandom++;
        }
        if (LengthNoShrinkRandom > 0 && OnShrinkRandom < LengthNoShrinkRandom &&
          NoShrinkRandom[OnShrinkRandom] == ii) {
        } else if (R_isnancpp(OnMe) || R_IsNA(OnMe)  || !R_finite(OnMe) || OnMe < 0.0 || OnMe > 1.0) {
           if(REAL(sOnTau)[ii] != 0.0) {
             CountRandomOn++;
           }
        }
      }     
    }    
  }

  if (Verbose >= 8) {
      Rprintf("uuu  BayesSpikeGibbs.cpp:::UpdatePiA(tt=%d) Boosted to CountRandomOn = %d.  LengthNoShrinkRandom=%d\n", 
        tt, CountRandomOn, LengthNoShrinkRandom); R_FlushConsole();
  }
  if (Rf_length(PiAPrior) == 2) {
    if (Temperature == 1.0) {
    REAL(sOnPiA)[0] = Rf_rbeta( CountFixedOn + CountRandomOn + REAL(PiAPrior)[0],
      LenFixed -CountFixedOn+ LenRandom -CountRandomOn + REAL(PiAPrior)[1]);
    } else {
      REAL(sOnPiA)[0] = Rf_rbeta( 
        (CountFixedOn + CountRandomOn + Temperature - 1.0 + REAL(PiAPrior)[0]) * invTemperature,
        (LenFixed + LenRandom -CountFixedOn-CountRandomOn + Temperature - 1.0 + REAL(PiAPrior)[1]) * invTemperature);    
    }
    if (Verbose >= 8) {
      Rprintf("uuu  BayesSpikeGibbs.cpp:::UpdatePiA(tt=%d) One way sOnPiA updated to %.6f\n", 
        tt, REAL(sOnPiA)[0]); R_FlushConsole();
    }
    if (MT != NULL) {
      MT->logOddsPi = log(REAL(sOnPiA)[0]) - log(1.0 - REAL(sOnPiA)[0]);
    }  
  } else if (Rf_length(PiAPrior) == 4) {
    if (Rf_length(sOnPiA) < 2) {
      Rf_error("Can't separate priors for random and fixed if OnPiA = 2");
    }
    if (Temperature == 1.0) {
      REAL(sOnPiA)[0] = Rf_rbeta( CountFixedOn + REAL(PiAPrior)[0],
        LenFixed + REAL(PiAPrior)[1]);  
      REAL(sOnPiA)[1] = Rf_rbeta(  CountRandomOn + REAL(PiAPrior)[2],
        LenRandom + REAL(PiAPrior)[2]);
    } else {
      REAL(sOnPiA)[0] = Rf_rbeta( (CountFixedOn + Temperature - 1.0 + REAL(PiAPrior)[0]) * invTemperature,
        (LenFixed - CountFixedOn + Temperature - 1.0 + REAL(PiAPrior)[1]) * invTemperature);  
      REAL(sOnPiA)[1] = Rf_rbeta(  (CountRandomOn + Temperature - 1.0 + REAL(PiAPrior)[2]) * invTemperature,
       ( LenRandom -CountRandomOn + Temperature - 1.0+ REAL(PiAPrior)[2]) * invTemperature );   
    }
    if (Verbose >= 8) {
      Rprintf("uuu  BayesSpikeGibbs.cpp:::UpdatePiA(tt=%d) Two way sOnPiA updated to %.6f, %.6f\n", 
        tt, REAL(sOnPiA)[0], REAL(sOnPiA)[1]); R_FlushConsole();
    }
    if (MT != NULL) {
      MT->logOddsPi = log(REAL(sOnPiA)[1]) - log(1.0 - REAL(sOnPiA)[1]);
    }  
  }
  return(1);
}

////////////////////////////////////////////////////////////////////////////////
//  AssignTau
//
//    Quickly assigns values to tau and activates necessary variables.
int BayesSpikeCL::AssignTau(SEXP sAssTau) {
  if (tauEndList == NULL || Rf_isNull(tauEndList) || Rf_isNull(sOnTau) ||
    sOnTau == NULL || Rf_length(tauEndList) <= 0 || Rf_length(sOnTau) <= 0) {
    Rf_error("AssignTau: No way we'll do this, tauEndList is null or sOnTau is NULL!\n"); 
  }
  if (Rf_isNull(sAssTau) || Rf_length(sAssTau) != Rf_length(sOnTau)) {
    Rf_error("AssignTau: You did not supply a valid sAssignTau.\n");
  }
  int St = iFirstRandom;
  for (int iOnii = 0; iOnii < Rf_length(tauEndList);iOnii++) {
    if (Rf_length(sAssTau) > iOnii && REAL(sAssTau)[iOnii] > 0.0) {
      if (XLC[St] < 0) {
        if (Verbose >= 1) {
          Rprintf("     $$ BayesSpikeGibbs.cpp:: AssignTau(): Looks like we'll add new Nonzero for iOnii=%d/%d, St=%d\n",
            iOnii, Rf_length(tauEndList), St); R_FlushConsole(); 
        }
        AllNewCoords += ANINT(tauEndList, iOnii)+1-St;
      }
      //for (int ii = St; ii <= ANINT(tauEndList, iOnii); ii++) {
      //  if (XLC[ii] < 0) {
      //    AllNewCood
      //    //AddCoord(ii);
      //  }
      //}
    }
    REAL(sOnTau)[iOnii] = REAL(sAssTau)[iOnii];
    St = ANINT(tauEndList, iOnii)+1;
  }
  if (AllNewCoords >= 1 && DoAddCoordsOnSetBeta > 0) {
      if (Verbose >= 1) {
        Rprintf("     $$ BayesSpikeGibbs.cpp:: AssignTau(): We should add at total of %d New Coords, %d Fixed, OnKappaS=%d/%d\n",
          AllNewCoords, NewFixedCoords, OnKappaS, OnKappaMem); R_FlushConsole();
      }
      AddAllNewCoords();
      if (Verbose >= 1) {
        Rprintf("     $$ BayesSpikeGibbs.cpp:: AssignTau(): We have added all of those coords,  OnKappaS=%d/%d",
          OnKappaS, OnKappaMem); R_FlushConsole();
      }
      RefreshOrderedActive(1);
  }  
  SampleTausOnly = 0;
  return(1);
}

int BayesSpikeCL::SampleNewTaus()  {
  if (sOnTau == NULL || tauEndList == NULL ||
    Rf_isNull(tauEndList) || Rf_isNull(sOnTau)) {
    Rf_error(" SampleNewTaus: Tau is not Setup! ");
  }
  WillAddNewTau = 0;
  int oVerbose = this->Verbose - 3;
  double PropTau = 0.0;
  int ii = 0;
  int St = iFirstRandom;
  if (RunProbVector != NULL  && tt >= StartRunProbVector  && iFirstRandom == 0) {
    TotInProbVector++;
  }
  
  //Verbose = 10;
  if (oVerbose > 0) {
    Rprintf(" SampleANewTaus Start: Verbose=%d, FirstRandom = %d.\n", 
      Verbose, iFirstRandom); R_FlushConsole();
  }
  double IPT = 0.0;
  if (oVerbose >= 10) {
    Rprintf(" SampleANewTaus: IPT = %d.\n", IPT); R_FlushConsole();
  }
  int iOnii;  
  
  //if (tt <= 1) {
  //  Rprintf("  SampleANewTaus(tt=%d):  Note Testing TestOrderedActive at the beginning!", tt); R_FlushConsole();
  //}
  //int ACount = TestOrderedActive();
  //if (ACount >= 1) {
  //  Rprintf("** ERRROR IN TestOrderedActive after SamnpleANewTaus. ACount = %d, tt=%d\n", ACount, tt); R_FlushConsole();
  //  Rf_error("** Go Catch this error!\n");
  //}
  
  double SumSBeta = 0.0;       int One = 1;
  double UsePiA;
  double TweakCons = 1.0;
  int NumIntegrations = 100000;
  double NewTau = 1.0;  double OnMe;
  #ifdef DOTIMEH
    unsigned long SampleNewClock1, SampleNewClock2;
  #endif 
  #ifdef DOTIMEH
    unsigned long FlipSampleAllTau1, FlipSampleAllTau2;
    FlipSampleAllTau1 = clock();
  #endif  
  double *pPutProb, PutProb = 0.0;     
  NukeBetaTau=0; NukeBetaNotTau=0;
  for (iOnii = 0; iOnii < Rf_length(sOnTau); iOnii++) {  
    if (oVerbose > 0) {
      Rprintf("SampleNewTaus: On ii = %d\n", iOnii);
    }
    #ifdef DOTIMEH
      SampleNewClock1 = clock();
    #endif
    PropTau = SampleANewTau(iOnii);
    #ifdef DOTIMEH
      SampleNewClock2 = clock();  
      REAL(RsTimePartsList->asSexp())[29] += ((double)(SampleNewClock2-SampleNewClock1) / CLOCKS_PER_SEC);  
    #endif
    //if (Verbose > -10) {
    //  Rprintf("SampleNewTaus: PropTau = %f, after SampleANewTau, FakeOutTau = %f\n",
    //    PropTau, FakeOuttau);
    //}
    PropTau = FakeOuttau;
    if (PropTau > ThisMaxTau) {
      Rprintf("Error: We wound up Sampling PropTau = %f > ThisMaxTau = %f \n",
        PropTau, ThisMaxTau); R_FlushConsole();
     OnMe = -1.0;
     if (!Rf_isNull(sPriorProbTau)) {
       OnMe = REAL(sPriorProbTau)[iOnii];
     } 
     if (R_isnancpp(OnMe) || OnMe > 1.0 || !R_finite(OnMe) ||
       R_IsNA(OnMe) || R_IsNaN(OnMe)) {
       OnMe = -1.0; 
     }
     if (OnMe < 0.0) { 
       if (Rf_length(sOnPiA) >= 2) {
         UsePiA = REAL(sOnPiA)[1];
       } else {
         UsePiA = REAL(sOnPiA)[0];
       }
       if (fabs(OnMe + 1.0) >= .0001) {
          UsePiA = UsePiA * fabs(OnMe)  / ( UsePiA * fabs(OnMe) + 1.0-UsePiA); // A CHANGE
       }
      } else if (OnMe == 0.0) {
        UsePiA = 0.0; 
      } else if (OnMe >= 1.0) {
        UsePiA = 1.0;
      } else {
        UsePiA = .5;
      }
      if (UsePiA == 0.0) {
        PropTau = 0.0;  FakeOuttau = 0.0;
      }  else {
        #ifdef DOTIMEH
          TauClock1 = clock();
        #endif
        if (ProbTau != NULL) {
          pPutProb = ProbTau + iOnii;
        } else {
          pPutProb = &PutProb;
        }
        FullIntegrate( &lFOfX, MT, 
          NumIntegrations, iOnii, ProbTau, UsePiA, TweakCons, 
          &IntegratedDensity, Verbose, &NewTau, pPutProb);
        #ifdef DOTIMEH 
          TauClock2 = clock();
          if (RsTimePartsList != NULL) {
            REAL(RsTimePartsList->asSexp())[20] +=((double)(TauClock2-TauClock1) / CLOCKS_PER_SEC); 
          }
        #endif
        Rprintf("Well, NewTau from FullIntegrate = %f \n", NewTau); R_FlushConsole();
        PropTau = NewTau;  FakeOuttau = NewTau;
      }
    }
    if (RunProbVector != NULL  && tt >= StartRunProbVector) {
      //double IPT;
      if (ProbTau[iOnii] <= -10) {
        IPT = exp(ProbTau[iOnii]);
      } else if (ProbTau[iOnii] >= 10) {
        IPT = 1.0 - exp(-ProbTau[iOnii]);
      } else {
        IPT = exp(ProbTau[iOnii]) /(1.0 + exp(ProbTau[iOnii]));
      }
      RunProbVector[iFirstRandom+iOnii]+=IPT;  TotEveryProbVector[iFirstRandom+iOnii]++;
    }
    #ifdef DOTIMEH
          TauClock1 = clock();
    #endif
    St = iFirstRandom; 
    if (iOnii  > 0) {
      St = ANINT(tauEndList, (iOnii-1)) +1;
    }
    int k = ANINT( tauEndList, iOnii ) - St +1;
    One = 1;
    SumSBeta = F77_CALL(dasum)(&k, REAL(sBeta) + St, &One);
    if (SampleTausOnly <= 0 && PropTau != 0.0 && REAL(sOnTau)[iOnii] == 0) {
      if (XLC[St] < 0) {
         NumberNew += ANINT(tauEndList, iOnii) - St+1;
         //AllNewCoords += ANINT(tauEndList, iOnii) - St+1;
         WillAddNewTau = 1;
      }
      REAL(sOnTau)[iOnii] = PropTau;
    } else if (SampleTausOnly <= 0 && PropTau == 0 && (((REAL(sOnTau)[iOnii] > 0 || SumSBeta > 0.0)) ||
      (REAL(sOnTau)[iOnii] == 0 && SumSBeta > 0.0))  ) {
      //Rprintf("Weird SumS Beta = %f for iOnii = %d\n", SumSBeta, iOnii);
      //Rprintf(" sBeta %d through %d are \n", St, St + k);
      //PrintVector(REAL(sBeta)+St, k);
      //Rprintf("\n  What's wrong? \n"); R_FlushConsole();
      //Rf_error("Hey, why are we refreshing SumSBeta when we clearly had them set to zero already!");
      NumberNew++;
      double ZeroD = 0.0;  double OneD = 1.0;
      char Trans = 'T';
      NukeBetaTau++;

      if (XLC[St] >= 0) {
        //char Trans = 'N';  double OneD = 1.0; 
        int Tester =  0;
        if (Verbose >= 5) {
          Rprintf("BayesSpikeGibbs.cpp::SampleNewTaus() Tester = %d \n", Tester);
          R_FlushConsole();
        }
        //Tester = TestXtResid();
        //if (Tester > 0) {
        //  Rprintf("** SampleNewTaus():  Well we just failed as before we add known St = %d, iOnii = %d, tt = %d\n", St, iOnii, tt);
        //  R_FlushConsole();
        //  Rf_error("** SampleNewTaus():  Not Good!\n");
        //}
        for (int jSt = St; jSt <= ANINT(tauEndList, iOnii); jSt++) {
          if (XLC[jSt] >= 0) {
            //Tester = TestXtResid();
            //if (Tester > 0) {
            //  Rprintf("** SampleNewTaus():  Well we just failed as before add jSt = %d, iOnii = %d, tt = %d\n", jSt, iOnii, tt);
            //  R_FlushConsole();
            //  Rf_error("** SampleNewTaus():  Not Good!\n");
            //}
            F77_CALL(daxpy)(&p, REAL(sBeta)+jSt, pXtX[XLC[jSt]], &One, XtResid, &One);
            REAL(sBeta)[jSt] = 0.0;
            //Tester = TestXtResid();
            //if (Tester > 0) {                                                                                       
            //  Rprintf("** SampleNewTaus():  Well we just failed right at end of Tester jSt = %d, iOnii = %d, tt = %d, XLC[jSt=%d] = %d\n", jSt, iOnii, tt,
            //    jSt, XLC[jSt]);
            //  R_FlushConsole();
            //  Rf_error("** SampleNewTaus():  Not Good!\n");
            //}
          } else {
            Rprintf("** Hey: Assumption on XLC[St=%d] is false for jSt=%d, this is not good, iOnii = %d!  PropTau = %f!\n",
              St, jSt, iOnii, PropTau); R_FlushConsole();
            Rprintf("** Error in BayesSpikeGibbs.cpp:::SampleNewTaus(); \n");   R_FlushConsole();
            if (WW == NULL) { RMemGetD(WW, "WW", n); }
            F77_CALL(dcopy)(&n, REAL(sX) + n * jSt, &One, WW, &One);
            Trans = 'T';
            F77_CALL(dgemv)(&Trans, &n, &p, REAL(sBeta)+jSt, REAL(sX), &n, WW, &One,
              &OneD, XtResid, &One);
            REAL(sBeta)[jSt] = 0.0;
            NukeBetaNotTau++;
            //Rf_error("Error: SampleNewTaus: jSt error\n", jSt); 
          }
        }
        //F77_CALL(dgemv)(&Trans, &p, &k, &OneD, pXtX[XLC[St]], &p, 
        //  REAL(sBeta)+St, &One, &OneD, XtResid, &One);
        //ZeroD = 0;
        //F77_CALL(dscal)(&k, &ZeroD, REAL(sBeta) + St, &One); 
        //Tester = TestXtResid();
        //if (Tester > 0) {
        //  Rprintf("** SampleNewTaus():  Well we just failed at end all at end as we added known St = %d, iOnii = %d, tt = %d\n", St, iOnii, tt);
        //  R_FlushConsole();
        //  Rf_error("** SampleNewTaus():  Not Good!\n");
        //}
      } else if (iiWeight == NULL) {
        if (WW == NULL) { RMemGetD(WW, "WW", n); }
        Trans = 'N';  
        F77_CALL(dgemv)(&Trans, &n, &k, &OneD, REAL(sX) + St * n, &n, 
          REAL(sBeta) + St, &One, &ZeroD, WW, &One);
        Trans = 'T';
        F77_CALL(dgemv)(&Trans, &n, &p, &OneD, REAL(sX), &n,
          WW, &One, &OneD, XtResid, &One);
        NukeBetaNotTau++;
        for (int ij = 0; ij < k; ij++) {
          REAL(sBeta)[ij+k] = 0.0;
        }
      }  else {
        if (WW == NULL) { RMemGetD(WW, "WW", n); } 
        for (int ij = 0; ij < k; ij++) {
           for (int jk = 0; jk < n; jk++) {
             WW[jk] = iiWeight[jk] * REAL(sX)[( St + ij) * n + jk];
           }
           F77_CALL(dgemv)(&Trans, &n, &k, REAL(sBeta) + St + ij, REAL(sX) + St *n, &n,
             WW, &One, &OneD, XtResid, &One); 
        }   
        F77_CALL(dscal)(&k, &ZeroD, REAL(sBeta) + St, &One);
      }
      REAL(sOnTau)[iOnii] = PropTau;
    }  else if (SampleTausOnly <= 0 && PropTau > 0 && XLC[St] < 0 && REAL(sOnTau)[iOnii] > 0) {
      //for (ii = St; ii <= ANINT(tauEndList, iOnii); ii++) {
      //  if (XLC[ii] < 0) {
      //    //AddCoord(ii);
      //    AllNewCoords+=ANINT(tauEndList, iOnii)+1-St;
      //    break;
      //  }
      //}
      REAL(sOnTau)[iOnii] = PropTau;
    } else {
      REAL(sOnTau)[iOnii] = PropTau;
    }
    #ifdef DOTIMEH
      TauClock2 = clock();
      if (RsTimePartsList != NULL) {
        REAL(RsTimePartsList->asSexp())[33] +=((double)(TauClock2-TauClock1) / CLOCKS_PER_SEC); 
      }
    #endif

  }

  #ifdef DOTIMEH
    FlipSampleAllTau2 = clock();
    if (RsTimePartsList != NULL) {
      REAL(RsTimePartsList->asSexp())[32] +=((double)(FlipSampleAllTau2-FlipSampleAllTau1) / CLOCKS_PER_SEC); 
    }
  #endif
  NumActiveTau = 0;
  St = iFirstRandom;
  for (ii = 0; ii < Rf_length(tauEndList); ii++) {
    if (REAL(sOnTau)[ii] > 0) {
      if (ii > 0) {
        St = ANINT(tauEndList, (ii-1) )+1;
      } else {
        St = iFirstRandom;
      }
      NumActiveTau++;
      if (XLC[St] < 0) {
        AllNewCoords += ANINT(tauEndList,ii)+1-St;
      }
    }
    //St = ANINT(tauEndList,ii)+1;
  }
  if (Verbose >= 4 || (AllNewCoords > 0  && Verbose >= 2)) {
    Rprintf("---- On tt = %d, NumActiveTau now %d, AllNewCoords = %d\n", tt, NumActiveTau, AllNewCoords);
    R_FlushConsole();
  }
  #ifdef DOTIMEH 
    TauClock2 = clock();
    if (RsTimePartsList != NULL) {
      REAL(RsTimePartsList->asSexp())[21] +=((double)(TauClock2-TauClock1) / CLOCKS_PER_SEC); 
    }
  #endif
  //if (tt <= 1) {
  // Rprintf("  SampleANewTaus(tt=%d):  Note Testing TestOrderedActive at the end!", tt); R_FlushConsole();
  //}
  //ACount = TestOrderedActive();
  //if (ACount >= 1) {
  //  Rprintf("** ERROR BayesSpikeGibbs.cpp:::SampleANewTaus(), we have a TestOrderedActive Fail!\n");
  //  Rprintf("** ERRROR IN TestOrderedActive after SamnpleANewTaus. ACount = %d, tt=%d\n", ACount, tt); R_FlushConsole();
  //  Rf_error("** Go Catch this error!\n");
  // }             
  return(1);  
}


////////////////////////////////////////////////////////////////////////////////
//  int BayesSpikeCL::SampleFixedB();
//
//   Sample Indicators of activation of "fixed parameters" which are parameters
//    not associated to groups.
//
//   "PriorProbFixed" can give different prior information for each coordinate
//   However, "PiA" can be unfixed and supply modification to this value
//    depending on length of PriorProbFixed.
//   
//   If we turn an previously off coordinate on, we must sample a value
//    immediately for that coordinate and update XtResid.
//
//
//
//
#define hL2PI 0.9189385
int BayesSpikeCL::SampleFixedB() {
  int oVerbose= this->Verbose - 3;
  
  
  JustDidPartBFixed = -1;
  int ii;
  double CondResid;   NewFixedCoords = 0;
  double LogOddsProb = 0.0;
  double PropTau = 0.0;
  //int Locii = 0;
  //if (Verbose >= 9) {
  //  Rprintf("SampleFixedB: Locii = %d \n", Locii); R_FlushConsole();
  //}
  double Prob = 1.0;                                                     
  SEXP ADependencyFixed;
  int jj; int pd;
  double iPropTau = 1.0;
  double iSigmaSq = 1.0/ REAL(sOnSigma)[0];
  double iA = 0.0;  double LeftBound = 0.0, RightBound = 0.0;
  
  double NewBeta = 0.0;
  
  int EndFixedB = iFirstRandom;
  if (iFirstRandom == 0) { 
    Rprintf("SampleFixedB, can't do it, iFirstRandom = 0!\n"); R_FlushConsole();
    return(-1); 
  }
  if (sOnTau == NULL || Rf_isNull(sOnTau) || Rf_length(sOnTau) <= 0 ||
    tauEndList == NULL || Rf_isNull(tauEndList) || Rf_length(tauEndList) <= 0) {
    EndFixedB = p;  
  }
  if (EndFixedB <= 0) {
    return(1);
  }

  NumActiveFixed = 0;

  
  if (BFixed == NULL) {
    Rf_error("SampleFixedB: Allocate BFixed first!");
  }
  int One = 1;
  if (Verbose > 1) {
    Rprintf("SampleFixedB: Starting\n"); R_FlushConsole();
  }
  double XtXj;
  double negsBeta;
  int OnShrinkFixed = 0;
  if (RunProbVector  != NULL  &&  tt > StartRunProbVector  &&
   (Rf_isNull(sOnTau) || Rf_length(sOnTau) <= 0 || iFirstRandom > 0)) {
    TotInProbVector++;
  }
  double PPF1 = -1;  double PPF2 = -1;
  if (tt == -1) {
    Rprintf("SampleFixedB:  Checking XtXj"); R_FlushConsole();
    XtXj = 0.0;
    for (ii = 0; ii < EndFixedB; ii++) {
      XtXj += XjSq[ii];
    }
    Rprintf(" --- XjSq pans out! sum is %f\n", XtXj); R_FlushConsole();
    Rprintf("SampleFixedB:  Checking XtResid"); R_FlushConsole();
    XtXj = 0.0;
    for (ii = 0; ii < EndFixedB; ii++) {
      XtXj += XtResid[ii];
    }
    Rprintf(" --- XtResid pans out! sum is %f\n", XtXj); R_FlushConsole();
    Rprintf(" --- Count BFixed \n"); R_FlushConsole();
    int CountOn = 0;  double PlusMetal = 0.0;
    for (ii = 0; ii < EndFixedB; ii++) {
      CountOn += BFixed[ii];
    }
    Rprintf(" --- BFixed counted up to ii=%d, and CountOn = %d\n", ii, CountOn);
    Rprintf(" --- Check Beta \n"); R_FlushConsole();
    XtXj = 0.0;
    for (ii = 0; ii < EndFixedB; ii++) {
      XtXj += REAL(sBeta)[ii];
    }
    Rprintf(" --- Beta, ii = %d, sum is %f\n", ii, XtXj);
    R_FlushConsole();
    Rprintf(" --- Check XLC \n"); R_FlushConsole();
    CountOn = 0;
    for (ii = 0; ii < p; ii++) {
      CountOn += XLC[ii];
    }
    Rprintf(" --- XLC, ii = %d, sum is %d\n", ii, CountOn);
    R_FlushConsole();
    if (OnKappaMem > 0) {
    Rprintf(" --- Check pXtX[0][0] = %f, OnKappaS=%d/%d\n", *(pXtX[0]+0), OnKappaS, OnKappaMem); R_FlushConsole();
    } else {
      Rprintf(" --- That's not good, OnKappaMem = 0!\n");
    }
    if (OnKappaS > 0) {
      Rprintf(" --- Check pXtX[OnKappaS-1][0] = %f \n", *(pXtX[OnKappaS-1]+0)); R_FlushConsole();
      Rprintf(" --- Check pXtX[OnKappaS-1][p-1] = %f \n", *(pXtX[OnKappaS-1]+p-1)); R_FlushConsole();
    } else {
      Rprintf(" -- Note, OnKappaS = %d, this won't work. \n", OnKappaS);
      Rprintf(" --- Check pXtX[OnKappaMem-1][0] = %f \n", *(pXtX[OnKappaMem-1]+0)); R_FlushConsole();
      Rprintf(" --- Check pXtX[OnKappaMem-1][p-1] = %f \n", *(pXtX[OnKappaMem-1]+p-1)); R_FlushConsole();
    }
    Rprintf(" --- TypePrior = %d \n", TypePrior); R_FlushConsole();
    if (ProbFixed != NULL) {
      Rprintf(" --- ProbFixed is Not Null \n"); R_FlushConsole();
      Rprintf(" --- And ProbFixed[EndFixedB-1] is %f\n", ProbFixed[EndFixedB-1]); R_FlushConsole();
    }
    if (tauFixed != NULL && !Rf_isNull(tauFixed) && Rf_length(tauFixed) >= 1) {
      Rprintf(" --- tauFixed is Not Null, we'll explore, its lengthis %d\n", Rf_length(tauFixed)); R_FlushConsole();
      for (ii = 0; ii < Rf_length(tauFixed); ii++) {
        if (ii < Rf_length(tauFixed)) {
          PlusMetal += REAL(tauFixed)[ii];
        } 
      }
      Rprintf(" --- tauFixed[%d-1] = Last Element is %f , PlusMetal = %f\n",
        Rf_length(tauFixed), Rf_length(tauFixed) > 0 ? REAL(tauFixed)[Rf_length(tauFixed)-1] : 0.0, PlusMetal);
      R_FlushConsole();
    }
    if (RunProbVector != NULL) {
      Rprintf(" --- RunProbVector is not null.\n"); R_FlushConsole();
      Rprintf(" --- RunProbVector[EndFixedB-1 = %d-1 = %d] = %f.\n",
        EndFixedB, EndFixedB-1, RunProbVector[EndFixedB-1]); R_FlushConsole();
    }
    Rprintf(" --- Check iiWeight \n"); R_FlushConsole();
    if (iiWeight != NULL) {
      Rprintf(" --- iiWeight is not null, for iiWeight \n"); R_FlushConsole();
      XtXj = 0.0;
      for (ii = 0; ii < EndFixedB;ii++) {
        XtXj += iiWeight[ii];
      }
      Rprintf(" --- iiWeight, XtXj is  %f \n", XtXj); R_FlushConsole();
    }
    Rprintf(" --- Check rPriorProbFixed \n"); R_FlushConsole();
    if (rPriorProbFixed != NULL) {
      Rprintf(" --- rPriorProbFixed is Not all NULL\n"); R_FlushConsole();
    } else if (rPriorProbFixed == NULL) {
      Rprintf(" --- rPriorProbFixed is Definitely a NULL!\n"); R_FlushConsole();
    }
    if (NoShrinkFixed != NULL && LengthNoShrinkFixed > 0) {
      Rprintf(" --- NoShrinkFixed is Not NULL!\n");  R_FlushConsole();
      Rprintf(" --- Last NoShrinkFixed is %f \n", NoShrinkFixed[LengthNoShrinkFixed-1]);
      R_FlushConsole();
    } else {
      Rprintf(" --- NoShrinkFixed is NULL, no issue!\n"); R_FlushConsole();
    }
    
    if (rPriorProbFixed == NULL) {
    } else if (rPriorProbFixed != NULL && rPriorProbFixed->asSexp() == NULL) {
      Rprintf(" --- rPriorProbFixed is a NULL object \n"); R_FlushConsole();
    } else if (rPriorProbFixed != NULL && Rf_isNull(rPriorProbFixed->asSexp())) {
      Rprintf(" --- rPriorProbFixed->asSexp() is a NULL object \n"); R_FlushConsole();
    } else if (rPriorProbFixed != NULL && !Rf_isNull(rPriorProbFixed->asSexp())) {
      Rprintf(" --- rPriorProbFixed->asSexp() is not a null object. \n"); R_FlushConsole();
      if (Rf_length(rPriorProbFixed->asSexp()) <= 0) {
        Rprintf(" --- rPriorProbFixed->asSexp() looks like length is %d \n",
          Rf_length(rPriorProbFixed->asSexp())); R_FlushConsole();
      } else {
        Rprintf(" --- rPriorProbFixed->asSexp() has length %d \n",
          Rf_length(rPriorProbFixed->asSexp())); R_FlushConsole();
      }
    }
    if (rPriorProbFixed == NULL) {
      Rprintf(" --- Again rPriorProbFixed is NULL!\n"); R_FlushConsole();
    } else if (rPriorProbFixed != NULL && rPriorProbFixed->asSexp() != NULL &&
      !Rf_isNull(rPriorProbFixed->asSexp())) {
      Rprintf(" --- We have rPriorProbFixed is not null!\n");  R_FlushConsole();
      if (Rf_length(rPriorProbFixed->asSexp()) == EndFixedB) {
        Rprintf("rPriorProbFixed has same length of EndFixedB\n"); 
        R_FlushConsole();
      }
    } else if (rPriorProbFixed != NULL && Rf_length(rPriorProbFixed->asSexp()) == 2*EndFixedB) {
      Rprintf(" --- rPriorProbFixed has 2 * EndFixedB\n");  R_FlushConsole();
    } else if (rPriorProbFixed != NULL && Rf_length(rPriorProbFixed->asSexp()) == 1) {
      Rprintf(" --- rPriorProbFixed has length 1\n"); R_FlushConsole();
    } else if (rPriorProbFixed != NULL && Rf_length(rPriorProbFixed->asSexp()) == 2) {
      Rprintf(" --- rPriorProbFixed has length 2\n"); R_FlushConsole();
    }
    Rprintf(" --- More Extended Test of pXtX \n"); R_FlushConsole();
    XtXj = 0.0;
    int CountNulls = 0;
    for (ii = 0; ii < OnKappaMem; ii++) {
      if (this->pXtX[ii] == NULL) {
        Rprintf(" --- ERROR Extended Test of pXtX.  Nope, pXtX[%d/%d] is NULL!\n", ii, OnKappaMem);
        R_FlushConsole();
        CountNulls++;
      }
      for (int jj = 0; jj < p; jj++) {
        XtXj += *(pXtX[ii] + jj);
      }
    }
    if (ProbFixed != NULL) {
      Rprintf(" --- Check ProbFixed Now. \n"); R_FlushConsole();
      for (int jj = 0; jj <EndFixedB; jj++) {
        ProbFixed[jj] = 0.0;
      }
      Rprintf(" --- ProbFixed Cleared.\n"); R_FlushConsole();
    }
    if (CountNulls > 0) {
      Rprintf(" --- We will quit with an error because Extended Test of pXtX shows CountNulls %d\n",
        CountNulls);
      Rf_error("Error, Locate the NULLS in pXtX\n");
    }
    Rprintf(" --- We completed extended test of pXtX, as far as we know integrity is strong! Now SampleBFixed\n"); 
    R_FlushConsole();
  }

  int jii; double TLogOddsProb = 0.0;
  for (jii = 0; jii < EndFixedB; jii++) {
    if ( (tt % 2) == 1) {
      ii = EndFixedB - jii-1;
      //ii = jii;
      if (ii < 0 || ii >= EndFixedB) {
        Rf_error("Error in your formula, ii=%d, jii=%d/%d, EndFixedB=%d.  No!\n",ii, jii, EndFixedB, EndFixedB);
      }
    } else {
      ii = jii;
    }
    /*if (tt == 1 && AB == 25 && ii == EndFixedB-1) {
      Rprintf("%d\n", ii); R_FlushConsole(); 
    } else if (tt == 1 && AB == 25) {
      Rprintf("%d,\n", ii); R_FlushConsole();   AB = 0;  
    } else if (tt == 1) {
      Rprintf("%d, ", ii); R_FlushConsole(); 
      AB++;
    } */
    PPF1 = -1.0;  PPF2 = -1.0;
    if (rPriorProbFixed != NULL && !Rf_isNull(rPriorProbFixed->asSexp())) {
      if (Rf_length(rPriorProbFixed->asSexp()) == EndFixedB) {
        PPF1 = REAL(rPriorProbFixed->asSexp())[ii];   PPF2 = -1.0;
        if (R_IsNA(PPF1) || R_IsNaN(PPF1) || !R_finite(PPF1) || R_isnancpp(PPF1) 
          || PPF1 < 0.0 || PPF1 > 1.0) {
          PPF1 = -1.0;
        } 
      } else if (Rf_length(rPriorProbFixed->asSexp()) == 2*EndFixedB) {
        PPF1 = REAL(rPriorProbFixed->asSexp())[ii*2];   
        PPF2 = REAL(rPriorProbFixed->asSexp())[ii*2+1]; 
        if (R_IsNA(PPF1) || R_IsNaN(PPF1) || !R_finite(PPF1) || R_isnancpp(PPF1) || PPF1 < 0.0 || PPF1 > 1.0) {
          PPF1 = -1.0; PPF2 = -1.0;
        } 
      } else if (Rf_length(rPriorProbFixed->asSexp()) == 1) {
        PPF1 = REAL(rPriorProbFixed->asSexp())[0]; PPF2 = -1.0;
      } else if (Rf_length(rPriorProbFixed->asSexp()) == 2) {
        PPF1 = REAL(rPriorProbFixed->asSexp())[0];
        PPF2 = REAL(rPriorProbFixed->asSexp())[1];
      }
    } else { PPF1 = -1.0;  PPF2 = -1.0;}
    if (this->Verbose > 3) {
      Rprintf("BayesSpikeGibbs.cpp:SampleFixedB: On ii = %d/%d.\n", ii, EndFixedB); R_FlushConsole();
    }
    if (oVerbose > 1) {
      Rprintf("BayesSpikeGibbs:SampleFixedB: On Coordinate %d,  OnKappaS=%d/%d\n", ii, OnKappaS, OnKappaMem); R_FlushConsole();
    }
    if (ii < 0 || ii >= p) {
      Rf_error("Error in SampleBFixed, right before ConResid, tt = %d, ii = %d, Not Good!\n",tt,ii); 
    }
    /*if (tt == 1 && (ii ==  10168 || ii == 548 || ii == 74)) {
      Rprintf("\n ---  Break, tt = %d, ii = %d.\n", tt, ii);   R_FlushConsole();
      Rprintf("\n ---  XtResid[ii=%d] = %f, XtXj[ii=%d] = %f.\n", ii, XtResid[ii], ii, XjSq[ii]);
      R_FlushConsole();
    } */
    CondResid = (double) XtResid[ii]; 
    XtXj = (double) XjSq[ii];
    if (REAL(sBeta)[ii] != 0.0) {
      CondResid += (double) XtXj * REAL(sBeta)[ii];
    }
    //if (REAL(sBeta)[ii] != 0.0 && (XLC[ii] < 0 || XLC[ii] >= OnKappaS)) {
    //   Locii = 0;
    //} else if (REAL(sBeta)[ii] != 0.0) {
    //   Locii = ii + p * XLC[ii];
    //}
    LogOddsProb = 100.00;
    if (NoShrinkFixed != NULL && LengthNoShrinkFixed > 0) {
      if (this->Verbose > 4) {                                               
        Rprintf("SampleFixedB:  Checking out NonNull NoShrink Fixed!\n"); R_FlushConsole();
        Rprintf("LengthNoShrinkVariables = %d, OnShrinkV = %d \n", 
          LengthNoShrinkFixed, OnShrinkFixed); R_FlushConsole();
      }
      while(OnShrinkFixed < LengthNoShrinkFixed &&
        NoShrinkFixed[OnShrinkFixed] < ii) {
        OnShrinkFixed++;  
      }
    }
    if (NoShrinkFixed != NULL && OnShrinkFixed < LengthNoShrinkFixed &&
      NoShrinkFixed[OnShrinkFixed] == ii) {
      BFixed[ii] = 1;
      if (this->Verbose > 4) {
        Rprintf("NoShrinkFixed: Not Shrinking ii=%d, NoShrinkFixed[%d]=%d\n",
          ii, OnShrinkFixed, NoShrinkFixed[OnShrinkFixed]); R_FlushConsole();
      }
      if (ii >= p || (iFirstRandom >= 0 && iFirstRandom <= ii)) {
        Rprintf("NoShrinkFixed: Issue: ii =%d/%d, jii=%d, iFirstRandom = %d and OnShrinkFixed=%d, LengthNoShrinkFixed=%d? \n",
           ii, p, jii, iFirstRandom, OnShrinkFixed, LengthNoShrinkFixed);
        Rprintf("NoShrinkFixed[%d]=%d, and ii=%d so this should be good but it is not?\n",
          OnShrinkFixed, NoShrinkFixed[OnShrinkFixed], ii);
        Rf_error("NoShrinkFixed: About to save inapropriate location for ii. \n");
      }
      if (ProbFixed != NULL) {
            ProbFixed[ii] = 101;
        } else if (Verbose > 2) {
          Rprintf("SampleFixedB:  Is it okay, NoShrink says don't shrink at all, but ProbFixed NULL?");
          R_FlushConsole();
      }
      LogOddsProb = 101.0;
    } else if (PPF1 >= .9999999) {
        BFixed[ii] = 1;
        if (ProbFixed != NULL) {
          ProbFixed[ii] = 100;
        } else if (Verbose > 2) {
          Rprintf("SampleFixedB:  Is it okay, PPF1 says don't shrink at all, but ProbFixed NULL?");
          R_FlushConsole();
        }
        LogOddsProb = 100.0;
    }  else if (PPF1 == 0.0) {
        BFixed[ii] = 0;
        if (ProbFixed != NULL) {
          ProbFixed[ii] = -1000.0;
        } else if (Verbose > 2) {
          Rprintf("SampleFixedB:  Is it okay, PPF1 says kill, but ProbFixed NULL?");
          R_FlushConsole();
        }
        LogOddsProb = -100.0;
    } else if (TypePrior <= 1) {
       //  SampleFixedB: TypePrior = 1
       //
       //
       //  This is the most basic type of prior, Gaussian around zero:
       //     P(Beta_j | B_j = 1) propto  exp(- .5 Beta_j^2 / tau^2_j)
       //
       //  The simple posterior is of form exp(-.5( A x^2 -2 B x))
       //    Where B = X^T_j(Y-X^T_{not j} Beta_{not j)) = CondResid
       //     And A =   X_ij^2 / sigma^2 + 1 / tau^2_j
       //
       //   tau^2_j is either REAL(tauFixed)[0] if tauFixed is of short length
       //    or it is set to tau^2_j = REAL(tauFixed)[j] if tauFixed is long enough
       if (oVerbose > 2) {
         Rprintf("  For ii = %d,about to do Type Prior = 1\n", ii ); R_FlushConsole();
       }
       if (Rf_length(tauFixed) > 1 && ii >= 0 && ii < p && Rf_length(tauFixed) > ii) {
         if (REAL(tauFixed)[ii] > 0.0) {    
           PropTau = REAL(tauFixed)[ii];
         } else {
           PropTau = fabs(REAL(tauFixed)[ii] * REAL(sOnSigma)[0]);
         }
       } else if (Rf_length(tauFixed) == 1) {
         if (REAL(tauFixed)[0] > 0.0) {
           PropTau = REAL(tauFixed)[0];
         } else {
           PropTau = fabs(REAL(tauFixed)[0] * REAL(sOnSigma)[0]);
         }
       }  else {
         Rf_error("BayesSpikeGibbs.cpp(): SampleFixedB, why is tauFixed length %d?",
           Rf_length(tauFixed));
       }
       if (PropTau > 0.0) {
         iPropTau = 1.0 / PropTau;
       } else {
         iPropTau = 0.0;
       }
       iA = 1.0 / ( XtXj * iSigmaSq + iPropTau);
       //LogOddsProb = -.5 * log( PropTau ) + .5 * log(iA) +
        // .5 * iSigmaSq  * iSigmaSq * (CondResid *CondResid) *iA;
       LogOddsProb = -.5 * log( PropTau*XtXj*iSigmaSq + 1.0 )  +
         .5 * iSigmaSq  * iSigmaSq * (CondResid *CondResid) *iA;
       if (oVerbose > 2) {
         Rprintf("  For ii=%d, PropTau=%f; iA=%f; CondResid=%f; XtXj = %f; LogOddsProb=%f; PPF1=%f; PPF2 = %f; PiA=%f; iSigmaSq=%f\n",
           ii, PropTau, iA, CondResid, XtXj, LogOddsProb, PPF1,PPF2,REAL(sOnPiA)[0], iSigmaSq); R_FlushConsole();
       }
       if (PPF1 > 0.0 && PPF2 > 0.0  && PPF2 < 1.0) {
         LogOddsProb += log(PPF1) - log(1.0 - PPF2);       
       }  else if (PPF1 == 0.0) {
         LogOddsProb = LogOddsProb - 9999999.0;     
       } else if (PPF1 == 1.0) {
          LogOddsProb += 999999999.0;
       } else if (PPF1 > 0.0 && PPF1 < 1.0) {
         LogOddsProb += log(PPF1) - log( 1.0 - PPF1);
       } else {
         if (sOnPiA == NULL || Rf_isNull(sOnPiA) || !Rf_isReal(sOnPiA) || Rf_length(sOnPiA) < 1) {
           Rf_error("Error: BayesSpikeGibbs.cpp SampleBFixed, sOnPiA is defective \n");
         }
         LogOddsProb += log(REAL(sOnPiA)[0]) - log( 1.0 - REAL(sOnPiA)[0]);
       }
       //if (tt == 1) {
       //  Rprintf("."); R_FlushConsole();
       //}
       if (oVerbose > 2) {
         Rprintf("  For ii = %d, LogOddsProb =  %f\n", ii, LogOddsProb); R_FlushConsole();
       }
     } else if (TypePrior == 4) {
       // TypePrior == 4
       //  P(Beta_j)*sqrt(2pitau^2) = .5 * exp( -.5( Beta_j - tau^2)^2/tau^2 | Beta_j > 0) + .5 (-.5 (Beta_j+tau^2)^2/ tau^2 | Beta_j < 0)
       //    = .5 exp(-Beta_j^2/tau^2-tau^2) *(  exp(-.5 Beta / tau)+ exp(.5 Beta/tau) )
       //
       //
       if (oVerbose > 2) {
         Rprintf("  For ii = %d, about to do Type Prior = 4\n",ii ); R_FlushConsole();
       }
       if (Rf_length(tauFixed) > 3 && Rf_length(tauFixed) > ii) {
         if (REAL(tauFixed)[ii] > 0.0) {
           PropTau = REAL(tauFixed)[ii];
         } else {
           PropTau = fabs(REAL(tauFixed)[ii] * REAL(sOnSigma)[0]);
         }
       } else if (tauFixed != NULL && !Rf_isNull(tauFixed) && Rf_length(tauFixed) >= 1) {
         if (REAL(tauFixed)[0] > 0.0) {
           PropTau = REAL(tauFixed)[0];
         } else {
           PropTau = fabs(REAL(tauFixed)[ii] * REAL(sOnSigma)[0]);
         }
       } 
       iPropTau = 1.0 / PropTau;
       iA = 1.0 / ( XtXj * iSigmaSq);
       double Bp = CondResid - PropTau;
       double Bn = CondResid + PropTau;
       LogOddsProb =   -log( PropTau) + .5 * log(iA) +
         ( .5 * iSigmaSq  * iSigmaSq * (Bp *Bp) *iA +
           Rf_pnorm5( -Bp * iA, 0.0,1.0,1,0) ) +
         (.5 * iSigmaSq * iSigmaSq * (Bn * Bn) * iA +
           Rf_pnorm5( - Bn * iA, 0.0,1.0,1,0) );
       if (PPF1 >= 0.0 && PPF2 >= 0.0) {
         LogOddsProb += log(PPF1) - log(1.0 - PPF2);       
       }  else if (PPF1 == 0.0) {
         LogOddsProb = LogOddsProb - 9999999.0;     
       } else if (PPF1 == 1.0) {
          LogOddsProb += 999999999.0;
       } else if (PPF1 > 0.0 && PPF1 < 1.0) {
         LogOddsProb += log(PPF1) - log( 1.0 - PPF1);
       } else {
         LogOddsProb += log(REAL(sOnPiA)[0]) - log( 1.0 - REAL(sOnPiA)[0]);
       }
        
       if (oVerbose > 2) {
         Rprintf("  For ii = %d,LogOddsProb =  %f\n", ii, LogOddsProb); R_FlushConsole();
       }
     } else if (TypePrior == 2) {
      if (tauFixed != NULL && !Rf_isNull(tauFixed) && Rf_length(tauFixed) < EndFixedB +1) {
        Rf_error("Cannot do abs mean fitting without room at end of tauFixed for mean");
      }
      if (tauFixed != NULL && !Rf_isNull(tauFixed) && Rf_length(tauFixed) > ii) {
        if (REAL(tauFixed)[ii] > 0.0) {
          PropTau = REAL(tauFixed)[ii];
        } else {
          PropTau = fabs(REAL(tauFixed)[ii] * REAL(sOnSigma)[0]);
        }
      } else if (tauFixed != NULL && !Rf_isNull(tauFixed) && Rf_length(tauFixed) >= 1) {
        if (REAL(tauFixed)[0] > 0.0) {
          PropTau = REAL(tauFixed)[0];
        } else {
          PropTau = fabs(REAL(tauFixed)[0] * REAL(sOnSigma)[0]);
        }
      } 
      iPropTau = 1.0 / PropTau; 
      if (Rf_isNull(Rf_getAttrib(tauFixed, R_DimSymbol)) ||
        Rf_length(Rf_getAttrib(tauFixed, R_DimSymbol)) <= 0 ||
        Rf_length(tauFixed) <= INTEGER(Rf_getAttrib(tauFixed, R_DimSymbol))[0] + ii) {
        Rf_error("Error SampleFixedB: Cannot do TypePrior == 4\n");
      }         
      if (tauFixed == NULL || Rf_isNull(tauFixed)) {
        Rprintf("BayesSpikeCpp.cpp:::SampleFixedB:: Why is tauFixed NULL Here?\n");
        R_FlushConsole();
      }
      SEXP DimtauFixed = Rf_getAttrib(tauFixed, R_DimSymbol);
      double PropDoubleMean  = 0.0;
      if (Rf_isNull(DimtauFixed) || Rf_length(DimtauFixed) <= 1) {
        if (Rf_length(tauFixed) > ii) {
          PropDoubleMean = REAL(tauFixed)[ii];
        } else {
          PropDoubleMean = REAL(tauFixed)[0];
        }
      } else {
        PropDoubleMean = REAL(tauFixed)[
        INTEGER(DimtauFixed)[0] + ii];
      }
      iA = 1.0 / ( XtXj * iSigmaSq + iPropTau);
      LogOddsProb = -.5 * log( PropTau) - hL2PI + .5 * log(iA) +
        .5 * iSigmaSq * iSigmaSq * (CondResid *CondResid) *iA +
        .5 * PropDoubleMean* PropDoubleMean  * 
        (iA - iPropTau * iPropTau) +
        log(
           .5 * exp(-CondResid* PropDoubleMean * .5 * iSigmaSq * iPropTau * iA) +
           .5 * exp( CondResid * PropDoubleMean * .5 * iSigmaSq * iPropTau * iA));
       if (PPF1 >= 0.0 && PPF2 >= 0.0) {
         LogOddsProb += log(PPF1) - log(1.0 - PPF2);       
       }  else if (PPF1 == 0.0) {
         LogOddsProb = LogOddsProb - 9999999.0;     
       } else if (PPF1 == 1.0) {
          LogOddsProb += 999999999.0;
       } else if (PPF1 > 0.0 && PPF1 < 1.0) {
         LogOddsProb += log(PPF1) - log( 1.0 - PPF1);
       } else {
         LogOddsProb += log(REAL(sOnPiA)[0]) - log( 1.0 - REAL(sOnPiA)[0]);
       }
     } else if (TypePrior == 3) {
       if (Rf_length(tauFixed) < 2 * EndFixedB) {
         Rf_error("If you're going to tell me a flat prior, supply bounds!");
       }
       LeftBound = REAL(tauFixed)[ii];
       RightBound = REAL(tauFixed)[
         INTEGER(Rf_getAttrib(tauFixed, R_DimSymbol))[0] + ii];
       if (RightBound < LeftBound) {
         Rf_error("For Prior Type 3, give Left < Right Bound");
       }
       iA = 1.0 / ( XtXj * iSigmaSq);
       LogOddsProb = .5 * log(iA) + log(
         Rf_pnorm5(RightBound, iA * CondResid * iSigmaSq, sqrt( iA ), 1,0) -
         Rf_pnorm5(LeftBound, iA * CondResid * iSigmaSq, sqrt( iA ), 1,0));
       if (PPF1 >= 0.0 && PPF2 >= 0.0) {
         LogOddsProb += log(PPF1) - log(1.0 - PPF2);       
       }  else if (PPF1 == 0.0) {
         LogOddsProb = LogOddsProb - 9999999.0;     
       } else if (PPF1 == 1.0) {
          LogOddsProb += 999999999.0;
       } else if (PPF1 > 0.0 && PPF1 < 1.0) {
         LogOddsProb += log(PPF1) - log( 1.0 - PPF1);
       } else {
         LogOddsProb += log(REAL(sOnPiA)[0]) - log( 1.0 - REAL(sOnPiA)[0]);
       }
     }
     if (Temperature != 1.0) { 
       TLogOddsProb = LogOddsProb * invTemperature;
       if (TLogOddsProb > 10.0) {
         Prob = 1.0-exp(-TLogOddsProb);
       } else if (TLogOddsProb < -10.0) {
         Prob = exp(TLogOddsProb);
       } else { 
         Prob = exp(TLogOddsProb) / (1.0 + exp(TLogOddsProb)); 
       }
     } else {
       if (LogOddsProb > 10.0) {
         Prob = 1.0-exp(-LogOddsProb);
       } else if (LogOddsProb < -10.0) {
         Prob = exp(LogOddsProb);
       } else { 
         Prob = exp(LogOddsProb) / (1.0 + exp(LogOddsProb)); 
       }
     }
     if (RunProbVector  != NULL  &&  tt > StartRunProbVector) {
       RunProbVector[ii] += Prob;   TotEveryProbVector[ii]++;
     }
     //if (tt == 1) {
     //    Rprintf("."); R_FlushConsole();
     //}
     if (Rf_runif(0.0,1.0) > Prob) {
        if (oVerbose > 3) {
         Rprintf("  For ii = %d, Prob = %f not gonna keep\n",ii, Prob); R_FlushConsole();
       }
       BFixed[ii] =((short int) 0);
     } else {
       if (oVerbose > 3) {
         Rprintf("  For ii = %d, = %f, gonna keep\n",ii, Prob); R_FlushConsole();
       }
       BFixed[ii] = ((short int) 1);
     }
     if (ProbFixed != NULL) {
       ProbFixed[ii] = LogOddsProb;
     }
     if (oVerbose > 2) {
       Rprintf("  For ii = %d, Check Dependencies fixed\n", ii); R_FlushConsole();
     }
     if (DependenciesFixed != NULL && !Rf_isNull(DependenciesFixed) &&
       Rf_length(DependenciesFixed) > ii)  {
       Rf_error("Sample BFixed: No Way are we doing Dependencies!");
       ADependencyFixed = VECTOR_ELT(DependenciesFixed, ii);
       for (jj = 0; jj < Rf_length(ADependencyFixed); jj++) {
         pd = ANINT( ADependencyFixed, jj);
         if (pd < ii && pd >= 0 && 
           BFixed[pd] > 0) {
           BFixed[ii] = 0;   
           break; 
         }
       }
     }
    if (oVerbose > 2) {
      Rprintf("  For ii = %d, Think about what to Add\n"); R_FlushConsole();
    }
    //if (tt == 1) {
    // Rprintf("."); R_FlushConsole();
    //}
    if (ii < 0 || ii >= p) {
      Rf_error(" -- Woah, BayesSpikeGibbs.pp, here ii = %d! and p=%d!\n", ii, p);
    }

    if (SampleTausOnly <= 0 && (int) BFixed[ii] != 0 && REAL(sBeta)[ii] == 0.0) {      
      //if (tt == 1) {
      //  Rprintf("g"); R_FlushConsole();
      //}
      if ( XLC[ii] < 0 && XLC[ii] < OnKappaS) { 
        if (OnKappaS + 1 < StartRemoval) {
          AddCoord(ii); 
          //if (REAL(sBeta)[ii] != 0.0) {
          //  F77_CALL(daxpy)(&p, REAL(sBeta)+ii, pXtX[XLC[ii]], &One, XtResid, &One);
          //}
        } else {
          AllNewCoords++;
        }
        NumberNew++;
      } else if (REAL(sBeta)[ii] != 0.0) {
        //if (REAL(sBeta)[ii] != 0.0) {
        //  F77_CALL(daxpy)(&p, REAL(sBeta)+ii, pXtX[XLC[ii]], &One, XtResid, &One);
        //}
      }
    } else if (BFixed[ii] == 0 && REAL(sBeta)[ii] != 0.0) {
      //if (tt == 1) {
      //  Rprintf("n"); R_FlushConsole();
      //}
      NumberNew++;
      One = 1;
      if (XLC[ii] >= 0 && XLC[ii] < OnKappaS) {  // Consider Changing XtResid Right Here!
        if (pXtX[XLC[ii]] == NULL) {
          Rprintf(" --- BayesSpikeGibbs.cpp::SampleFixedB() daxpy Error on XtResid Change, XLC[ii]=%d, pXtX[%d] is NULL!\n",
            ii, XLC[ii]);
          Rf_error(" --- BayesSpikeGibbs.cpp::SampleFixedB() NULL pXtX is Not good for daxpy");
        }
        F77_CALL(daxpy)(&p, REAL(sBeta)+ii, pXtX[XLC[ii]], &One, XtResid, &One);
      } else if (iiWeight == NULL) {
        AllNewCoords++;  NewFixedCoords++;
        char Trans = 'T';  double OneD = 1.0;  
         negsBeta = -1.0 * REAL(sBeta)[ii];  One = 1;
        if ( (( (double) n ) / ((double )8192.0) ) * ((double)ii) >= 4096.0) {
          double *MyPoint = REAL(sX);
          for (int iti = 0; iti < ii; iti++) {
            MyPoint = (double*) ((double*) MyPoint + n);
          }
          F77_CALL(dgemv)(&Trans, &n, &p, &negsBeta, REAL(sX), &n, MyPoint, &One,
            &OneD, XtResid, &One);
        } else {
          F77_CALL(dgemv)(&Trans, &n, &p, &negsBeta, REAL(sX), &n, REAL(sX) + ii*n, &One,
            &OneD, XtResid, &One);
        }
        
      } else {
        AllNewCoords++;  NewFixedCoords++;
        for (int jk = 0; jk < n; jk++) {
          double Factor = REAL(sBeta)[ii] * iiWeight[jk];
          F77_CALL(daxpy)(&p, &Factor, REAL(sX) + jk, &n, XtResid, &One);
        }
      }
      REAL(sBeta)[ii] = 0.0;
      if (iWeightedXtX != NULL && XLC[ii] >= 0  && XLC[ii] < OnKappaS) {
        iWeightedXtX[XLC[ii]] = (short int) 0;
      }
    } else if (SampleTausOnly <= 0 && BFixed[ii] == 1 && 
      REAL(sBeta)[ii] != 0.0 && (XLC[ii] < 0 || XLC[ii] >= OnKappaS)) {
      //if (tt == 1) {
      //  Rprintf("a"); R_FlushConsole();
      //}
      if (OnKappaS+1 < StartRemoval) {
        AddCoord(ii); 
      } else {
        AllNewCoords++;  NewFixedCoords++;
      }
    }
    
    //  For highly correlated datasets it might be useful to update Betas such
    //   that they enter during BFixed and eliminated correlation.
    //
    //  If we are
    if (BFixed[ii] == 1 && SampleTausOnly <= 0 && DoSampleBFixedii >= 1 && BFixed[ii] == 1 && XLC[ii] >= 0 && XLC[ii] < p) {
      if (JustDidPartBFixed < 0) {
        JustDidPartBFixed = ii;
      }
      double d0ii;
      if (iiWeight != NULL && iWeightedXtX != NULL && XLC[ii] >= 0 && XLC[ii] < OnKappaS &&
        iWeightedXtX[XLC[ii]] == 0) {
        ReweightedOneXtX(ii);
      }
      if (iiWeight == NULL) {
         d0ii = F77_CALL(ddot)(&n, REAL(sX), &One, REAL(sX) + n * ii, &One);
      } else {
        if (WW == NULL) {
          RMemGetD(WW,"WW", n);
        } 
        if (iWeightedXtX != NULL && XLC[ii] >= 0 && XLC[ii] < OnKappaS && iWeightedXtX[XLC[ii]] == 0) {
          ReweightedOneXtX(ii);
        }
        for (int iti = 0; iti < n; iti++) {
          WW[iti] = iiWeight[iti] * REAL(sX)[iti];
        }
        d0ii = F77_CALL(ddot)(&n, WW, &One, REAL(sX) + n * ii, &One);
      }
      if (fabs(*(pXtX[XLC[ii]]+0) - d0ii) > .0001) {
        Rprintf("SetupBFixed, so we want to add ii = %d, XLC[ii=%d]=%d, but pXtX[%d] = %f, and sX[,0] * sX[,%d] = %f\n",
          ii, ii, XLC[ii], XLC[ii], *(pXtX[ii]+0), ii, d0ii); R_FlushConsole();
        Rprintf("Here is pXtX[%d]: ",ii); R_FlushConsole();
        for (int iti = 0; iti < 5; iti++) {
          if (iti < p) {
            Rprintf("%.4f, ", *(pXtX[ii]+iti)); R_FlushConsole();
          }
          Rprintf("\n"); R_FlushConsole();
        }
        Rf_error("SetupBFixed, I think there is a swap error !\n"); R_FlushConsole();
      }
      double diiii;
      if (iiWeight == NULL) {
         diiii = F77_CALL(ddot)(&n, REAL(sX)+n*ii, &One, REAL(sX) + n * ii, &One);
      } else {
        if (WW == NULL) {
          RMemGetD(WW,"WW", n);
        } 
        for (int iti = 0; iti < n; iti++) {
          WW[iti] = iiWeight[iti] * REAL(sX)[iti + n * ii];
        }
        diiii = F77_CALL(ddot)(&n, WW, &One, REAL(sX) + n * ii, &One);
      }
      if (fabs(*(pXtX[XLC[ii]]+ii) - diiii) > .0001) {
        Rprintf("SetupBFixed, so we want to add ii = %d, XLC[ii=%d]=%d, but pXtX[%d]+%d = %f, and sX[,%d] * sX[,%d] = %f\n",
          ii, ii, XLC[ii], XLC[ii], ii, *(pXtX[ii]+ii), ii, diiii); R_FlushConsole();
        Rf_error("SetupBFixed, I think there is a swap error !\n"); R_FlushConsole();
      }
    
      if (TypePrior == 1 || TypePrior == 2) {
       ///  iSigmaSq * Xj^2 - 2 * CondResid Xj + Xj^2/iPropTau
       
       if (Temperature == 1.0) {
         NewBeta = iA * CondResid * iSigmaSq + sqrt(iA) * Rf_rnorm(0.0,1.0); 
       } else {
         NewBeta = iA * CondResid * iSigmaSq + sqrt(Temperature*iA) * Rf_rnorm(0.0,1.0); 
       } 
       //CondResid = XtResid[ii]; iA = 1.0 / ( (*(pXtX[XLC[ii]] + ii)) + REAL(sOnSigma)[0]/ PropTau);  
       //NewBeta =  CondResid * iA + sqrt(iA * Temperature * REAL(sOnSigma)[0]) * Rf_rnorm(0.0,1.0);
       if (R_isnancpp(NewBeta)) {
       } else if (NewBeta != REAL(sBeta)[ii]) {
         negsBeta = (REAL(sBeta)[ii]-NewBeta); 
         F77_CALL(daxpy)(&p, &negsBeta, pXtX[XLC[ii]], &One, XtResid, &One);
         REAL(sBeta)[ii] = NewBeta;
       }
      } else if (TypePrior == 3) {
        NewBeta = iA * CondResid;  double Low, High;   double Z;
        //double Low = (LeftBound-NewBeta) / sqrt(iA);
        //double High = (RightBound-NewBeta) / sqrt(iA);
        if (Temperature == 1.0) {
          //    TFL =  PBo + FC * Low <  PBo + FC * High = TFU
          Low = (LeftBound - NewBeta) / sqrt(iA);
          High = (RightBound - NewBeta) / sqrt(iA);
          Z = rTruncNorm2(Low, High);    
          NewBeta =  NewBeta + Z * sqrt(iA);
        }  else {
          double SqT = sqrt(Temperature);
          Low = (LeftBound - NewBeta) / sqrt(iA);
          High = (RightBound - NewBeta) / sqrt(iA);
          Z = rTruncNorm2(Low, High);    
          NewBeta =  NewBeta + Z * SqT*sqrt(iA);
        }
        if (R_isnancpp(NewBeta)) {
        } else if (NewBeta != REAL(sBeta)[ii]) {
           negsBeta = (REAL(sBeta)[ii]-NewBeta); 
           F77_CALL(daxpy)(&p, &negsBeta, pXtX[XLC[ii]], &One, XtResid, &One);
           REAL(sBeta)[ii] = NewBeta;
        }
      }
    }
  }
  if (tt <= -1) {
    Rprintf("BayesSpikeGibbs.cpp:SampleFixedB(): After SampleFixedB ii to %d, we check ProbFixed. \n", EndFixedB); R_FlushConsole();
    for (ii = 0; ii < EndFixedB; ii++) {
      ProbFixed[ii] = ProbFixed[ii]+1.0;
      ProbFixed[ii] = ProbFixed[ii]-1.0;
    }
    Rprintf("BayesSpikeGibbs.cpp:SampleFixedB(): After SampleFixedB ProbFixed passes. \n");R_FlushConsole();
  }
  for (ii = 0; ii < EndFixedB; ii++) {
    NumActiveFixed += BFixed[ii];
  }
  int tauFixedNoSigma = 1;
  if (REAL(tauFixed)[0] < 0.0) {
    tauFixedNoSigma = 0;
  }
  if (!Rf_isReal(tauFixed)) {
    Rf_error("BayesSpikeGibbs:SampleFixedB(): Late in the game to discover tauFixed is not Real!\n");
  }
  if (tauFixed == NULL || Rf_isNull(tauFixed) || Rf_length(tauFixed) <= 0) {
  } else if (RtauFixedPrior == NULL || tauFixedPrior == NULL || Rf_isNull(tauFixedPrior) || Rf_length(tauFixedPrior) <= 0) {
  } else if (Rf_length(tauFixed) == 1 && !Rf_isNull(tauFixedPrior) && Rf_length(tauFixedPrior) == 2 && TypePrior == 1) {
     double dS = 0.0;
     for (ii = 0; ii < EndFixedB; ii++) {
       if (BFixed[ii] > 0 && REAL(sBeta)[ii] != 0) {
         dS += REAL(sBeta)[ii] * REAL(sBeta)[ii];
       }
     }
     if (tauFixedNoSigma == 0) {
       dS = dS / REAL(sOnSigma)[0];
     }
     if (Temperature == 1.0) {
       if (.5 *(NumActiveFixed + REAL(tauFixedPrior)[0]) <= 0) {
         REAL(tauFixed)[0] = .5 *( dS + 1) /
           Rf_rgamma(1.0,1.0);
       } else {
         REAL(tauFixed)[0] = .5 *( dS + REAL(tauFixedPrior)[1] * REAL(tauFixedPrior)[0]) /
           Rf_rgamma(.5 *(NumActiveFixed + REAL(tauFixedPrior)[0]),1.0);
       }
     } else {
       REAL(tauFixed)[0] = .5 *( dS + REAL(tauFixedPrior)[1] * REAL(tauFixedPrior)[0]) * invTemperature /
         Rf_rgamma( ( 1.0 - Temperature + .5 *(NumActiveFixed + REAL(tauFixedPrior)[0] )) * invTemperature,1.0);     
     }
     if (tauFixedNoSigma == 0) { REAL(tauFixed)[0] = -1.0 * REAL(tauFixed)[0]; }
  } else if (Rf_length(tauFixed) == EndFixedB+1  && !Rf_isNull(tauFixedPrior) && Rf_length(tauFixedPrior) ==  5 && TypePrior == 1) {
    if (tauFixedNoSigma == 1) {
     if (Temperature == 1.0) {
       for (ii = 0; ii < EndFixedB; ii++) {
         if (Rf_length(tauFixed) < EndFixedB+1) {
           Rprintf("BayesSpikeGibbs.cpp:::Real Tau Hey: tauFixed Error issue here. \n"); R_FlushConsole();
         } else if ( REAL(tauFixed)[EndFixedB] + .5 <= 0) {
           REAL(tauFixed)[ii] =  ( 0 + 
               .5 * REAL(sBeta)[ii] * REAL(sBeta)[ii] ) / Rf_rgamma(  .5, 1.0);
         } else {
         REAL(tauFixed)[ii] = 
           ( REAL(tauFixed)[EndFixedB] * REAL(tauFixedPrior)[0] + 
             .5 * REAL(sBeta)[ii] * REAL(sBeta)[ii] ) /
             Rf_rgamma( REAL(tauFixedPrior)[0] + .5, 1.0);
         }
       }
     } else {
       for (ii = 0; ii < EndFixedB; ii++) {
         REAL(tauFixed)[ii] = 
           (( REAL(tauFixed)[EndFixedB] * REAL(tauFixedPrior)[0] + 
             .5 * REAL(sBeta)[ii] * REAL(sBeta)[ii]) * invTemperature ) /
             Rf_rgamma(( 1.0 - Temperature + REAL(tauFixedPrior)[0] + .5)* invTemperature, 1.0);
       }     
     }
    } else {
     double iSigma = 1.0 / REAL(sOnSigma)[0];
     if (Temperature == 1.0) {
       for (ii = 0; ii < EndFixedB; ii++) {
         if (Rf_length(tauFixed) < EndFixedB+1) {
           Rprintf("BayesSpikeGibbs.cpp:::Real Tau Hey: tauFixed Error issue here. \n"); R_FlushConsole();
         } else if ( REAL(tauFixed)[EndFixedB] + .5 <= 0) {
           REAL(tauFixed)[ii] =  -( 0 + 
               .5 * REAL(sBeta)[ii] * REAL(sBeta)[ii] * iSigma) / Rf_rgamma(  .5, 1.0);
         } else {
         REAL(tauFixed)[ii] =  
           -1.0 * ( REAL(tauFixed)[EndFixedB] * REAL(tauFixedPrior)[0] + 
             .5 * REAL(sBeta)[ii] * REAL(sBeta)[ii]*iSigma ) /
             Rf_rgamma( REAL(tauFixedPrior)[0] + .5, 1.0);
         }
       }
     } else {
       for (ii = 0; ii < EndFixedB; ii++) {
         REAL(tauFixed)[ii] = 
           -1.0 * (( REAL(tauFixed)[EndFixedB] * REAL(tauFixedPrior)[0] + 
             .5 * REAL(sBeta)[ii] * REAL(sBeta)[ii]* iSigma) * invTemperature ) /
             Rf_rgamma(( 1.0 - Temperature + REAL(tauFixedPrior)[0] + .5)* invTemperature, 1.0);
       }     
     }     
    }
  } else if (TypePrior == 1 && Rf_length(tauFixed) == 2*EndFixedB+1 && Rf_length(tauFixedPrior) == 6) {
     double dS = 0.0;
     for (ii = 0; ii < EndFixedB; ii++) {
       if (BFixed[ii] > 0 && REAL(sBeta)[ii] != 0) {
         dS += REAL(sBeta)[ii] * REAL(sBeta)[ii] / fabs(REAL(tauFixed)[EndFixedB+ii]);
       }
     }
    if (tauFixedNoSigma == 0) { dS = dS / REAL(sOnSigma)[0];}
    if (Rf_length(tauFixed) >= 1 + 2*EndFixedB) {
      if (Temperature== 1.0) {
        if (.5 *(NumActiveFixed + REAL(tauFixedPrior)[3]) <= 0) {
          REAL(tauFixed)[2*EndFixedB] = .5 *( dS + .5) /
            Rf_rgamma(.5 ,1.0);
        } else {
          REAL(tauFixed)[2*EndFixedB] = .5 *( dS + REAL(tauFixedPrior)[3] * 
            REAL(tauFixedPrior)[4]) /
            Rf_rgamma(.5 *(NumActiveFixed + REAL(tauFixedPrior)[3]),1.0);
        }
      } else if (Rf_length(tauFixed) >= 1 + 2*EndFixedB) {
        REAL(tauFixed)[EndFixedB*2] = .5 *( dS + REAL(tauFixedPrior)[3] * 
          REAL(tauFixedPrior)[4]) * invTemperature /
          Rf_rgamma( (Temperature - 1.0 + .5 *( + NumActiveFixed + REAL(tauFixedPrior)[3])) * invTemperature,1.0);     
      }
      if (tauFixedNoSigma == 0) {
        REAL(tauFixed)[EndFixedB*2] = -1.0 * REAL(tauFixed)[EndFixedB*2];
      }
    } 
    if (Rf_length(tauFixed) >= 2*EndFixedB + 1) {
      for (ii = 0; ii < EndFixedB; ii++) {
        REAL(tauFixed)[ii] = fabs(REAL(tauFixed)[2*EndFixedB] * REAL(tauFixed)[EndFixedB + ii]) *
          (2.0*tauFixedNoSigma - 1.0);
      }
    }
  }
  if (Rf_length(tauFixed) == 1 && !Rf_isNull(tauFixedPrior) && Rf_length(tauFixedPrior) >= 2 && TypePrior == 4) {
    double dS = F77_CALL(dasum)(&EndFixedB, REAL(sBeta), &One);
    if (Temperature == 1.0) {
      if (NumActiveFixed + REAL(tauFixedPrior)[0] <= 0) {
      REAL(tauFixed)[0] = ( dS+.5) /
        Rf_rgamma( (NumActiveFixed + .5),1.0);
      } else {
      REAL(tauFixed)[0] = ( dS + REAL(tauFixedPrior)[1] * REAL(tauFixedPrior)[0]) /
        Rf_rgamma( (NumActiveFixed + REAL(tauFixedPrior)[0]),1.0);
      }
    } else {
      REAL(tauFixed)[0] = ( dS + REAL(tauFixedPrior)[1] * REAL(tauFixedPrior)[0]) * invTemperature /
        Rf_rgamma( (NumActiveFixed + REAL(tauFixedPrior)[0]) * invTemperature,1.0);     
    }
  }

  //if (tt == 1) {
  //  Rprintf("BayesSpikeGibbs.cpp: SampleFixedB(tt=%d), all finished.\n", tt); R_FlushConsole();
  //}
  return(1);
}

int BayesSpikeCL::WriteP(int sP, int eP) {
  FILE *pFile = NULL;
  if (Verbose >= 4) {
    Rprintf(" About to Write CodaP!\n"); R_FlushConsole();
    Rprintf("  Rf_isString(sCodaiPFile) = %d\n", Rf_isString(sCodaPFile)); R_FlushConsole();
    Rprintf("  sCodaiPFile = %s \n",  CHAR(STRING_ELT(sCodaPFile, 0)) ); R_FlushConsole();
  }
  if (!Rf_isString(sCodaPFile)) {
     Rprintf("Error sCodaPFile is not String!: \n"); R_FlushConsole();
     Rprintf("What are we going to do about this string?\n ");
     Rprintf(" it is %s \n", CHAR(STRING_ELT(sCodaPFile, 0)) ); R_FlushConsole();
  }

  if (CodaTable == NULL || Rf_isNull(CodaTable) || Rf_length(CodaTable) <= 0) {
    Rf_error("RecordTau, CodaTable was Null, doh!");
  }
  if (this->Verbose > 3) {
    Rprintf("WriteP: Get Dim of sDCodaTable \n"); R_FlushConsole();
    R_FlushConsole();
  }
  SEXP sdCodaTable = Rf_getAttrib(CodaTable, R_DimSymbol);
  if (Rf_isNull(sdCodaTable) || Rf_length(sdCodaTable) < 2) {
    Rf_error("RecordTau: CodaTable was not right dim");
  }
  int LDA = ANINT(sdCodaTable, 0);
  int TLen = ANINT(sdCodaTable, 1);
  if (TLen < sP || sP + eP > TLen) {
    Rf_error("WriteP: CodaTable not big enough, TLen = %d, but sP = %d, eP = %d\n",
      TLen, sP, eP);
  }
  if (this->Verbose > 3) {
    Rprintf("WriteP: Opening pFile \n"); R_FlushConsole();
    R_FlushConsole();
  }
  if (NewPWrite > 0) {
    pFile = fopen(CHAR(STRING_ELT(sCodaPFile, 0)), "wb");    
  }  else {
    pFile = fopen(CHAR(STRING_ELT(sCodaPFile, 0)), "ab"); 
  }
  if (pFile != NULL) { TotalOpenedFiles++; }
  if (pFile == NULL) {
    Rf_error("WriteP: pFile is Null, could not open!");
  }
  FFree(CodaP, "CodaP");
  int TotalP = eP * LDA; 
  if (this->Verbose > 3) {
    Rprintf("WriteP: Allocating %d TotalP for CodaP \n", TotalP); R_FlushConsole();
    R_FlushConsole();
  }
  RMemGetD(CodaP, "CodaP", TotalP);
  int tt1 = 0;
  int t2P = 0; int nn1;
  int no = 0;
  if (this->Verbose > 3) {
    Rprintf("WriteP: Putting %d info into CodaP \n", TotalP); R_FlushConsole();
    R_FlushConsole();
  }
  for (tt1 = 0; tt1 < LDA; tt1++) {
    nn1 = tt1 + LDA * sP;
    for (t2P = sP; t2P < sP + eP; t2P++) {
      CodaP[no] = REAL(CodaTable)[nn1];
      nn1 += LDA;  no++;
    }
  }
  if (this->Verbose > 3) {
    Rprintf("WriteP: Now Writing to pFile \n"); R_FlushConsole();
    R_FlushConsole();
  }
 fwrite( CodaP, sizeof(double), TotalP, pFile ); 
 int weClosed = 0;
 weClosed = fclose(pFile);  
 if (weClosed != 0) {
   Rprintf("WriteP: Well, we failed to close sCodaPFile !\n");  R_FlushConsole();
 }
 TotalClosedFiles++;
 pFile = NULL;
 FFree(CodaP, "CodaP");
 NewPWrite = 0;
 if (this->Verbose > 3) {
    Rprintf("WriteP: Finished \n"); R_FlushConsole();
    R_FlushConsole();
 }
 return(1);
}

//////////////////////////////////////////////////////////////////////////////
//  FillTau
//
//    Writes Tau Columns to a file
//     These will be double/integer files, such that only positive hits are
//     recorded.
//
//
int BayesSpikeCL::FillTau()  {


  if (Verbose >= 2) {
    Rprintf("FillTau: Starting to Fill Tau Buffer\n"); R_FlushConsole();
  }
  if (CodaTjj == NULL || CodaTau == NULL) {
    if (Verbose >= 2) {
      Rprintf("No Fill Tau if CodaTjj and CodaTau not set up!\n");
      R_FlushConsole();
    }
    return(-1);
  }
  if (sOnTau == NULL || Rf_isNull(sOnTau) || Rf_length(sOnTau) <= 0) {
    Rf_error("No Fill Tau if there is no OnTau!\n");
  }
  if (RsCodaiTLocFile == NULL || Rf_isNull(RsCodaiTLocFile->asSexp())) {
    Rprintf("FillTau: shouldn't happen because RsCodiTLocFile is set to NULL!\n");
  }
  if (sCodaiTFile == NULL || sCodadTFile == NULL) {
    if (Verbose >= 2) {
      Rprintf("No Fill Tau if sCodaiTFile aren't setup!\n");
      R_FlushConsole();
    }
    return(-1);
  }
  int AT = 0;

  int TotalOnTaus = 0;
  for (int taut = 0; taut < Rf_length(sOnTau); taut++) {
    if (REAL(sOnTau)[taut] > 0.0) {
      TotalOnTaus++;
    }
  }
  if (OnCodaT + TotalOnTaus+1 >= LengthCodaTjj) {
    AT = WriteTau();
    if (AT < 0) {
      Rprintf("********************************************************\n");
      Rprintf("** FillTau: we get an AT = %d, after running WriteTau \n", AT);
      Rprintf("** Sign of failure, we return an error too! \n");
      return(-1);
    }
    OnCodaT = 0;
  }
  int OldWriteTLoc = NewWriteTLoc;
  
  RanFillTau++;
  if (CodaTLocjj == NULL) {
    Rf_error("FillTau: Why do this with NULL CodaTLocjj, you'll never get anywhere!\n");
  }
  if (CodaTLocjj != NULL) {
    if (Verbose >= 2) {
      Rprintf("  FillTau: CodaTLocjj  is not NULL!\n"); R_FlushConsole();
    }
    if (OnCodaTLocjj +1 >= LengthCodaTLoc) {
      FILE *iTFile=NULL;
      if (NewWriteTLoc == 1) {
        iTFile = fopen(CHAR(STRING_ELT(RsCodaiTLocFile->asSexp(), 0)), "wb");
        if (iTFile != NULL) { TotalOpenedFiles++; }   
        if (iTFile == NULL) {
          Rprintf("******************************************************\n");
          Rprintf("** FillTau: We fail on try to open iTFile first time!\n");
          R_FlushConsole();
        }
        NewWriteTLoc = 0;    WroteTotalTau = 0;
      } else {
        iTFile = fopen(CHAR(STRING_ELT(RsCodaiTLocFile->asSexp(), 0)), "ab"); 
        if (iTFile != NULL) { TotalOpenedFiles++; }       
        if (iTFile == NULL) {
          Rprintf("******************************************************\n");
          Rprintf("** FillTau: We fail on try to open on an ab!\n");
          R_FlushConsole();
        }
      }
      if (iTFile == NULL) {
        Rprintf("FillTau:******************************************************\n");
        Rprintf("** Error on try to read iTFile, NewWriteTLoc = %d, OldWriteTLoc = %d\n",
          NewWriteTLoc, OldWriteTLoc); R_FlushConsole();
        Rprintf("** iTFile name is %s\n",
          CHAR(STRING_ELT(RsCodaiTLocFile->asSexp(), 0)));
        Rprintf("** FillTau: Error, We got a NULL on fopen of iTFile cannot open CodaiTLocFile!\n");
        Rprintf("** FillTau: Only possibility is to quit now! \n"); R_FlushConsole();
        Rprintf("** Clearing OnCodaTLocjj buffer ! \n");
        FFree(CodaTLocjj, "CodaTLocjj"); CodaTLocjj = NULL;
        OnCodaTLocjj = 0; TotCodaTLocjj = 0;
        DDelete(RsCodaiTLocFile, "RsCodaiTLocFile"); RsCodaiTLocFile = NULL;
        Rprintf("** FillTau: Finished deleting this failure. \n");
        R_FlushConsole();
        return(-1);
      }
      WroteTotalTau += OnCodaTLocjj;
      WroteAllTau += OnCodaTLocjj;
      if (OnCodaTLocjj <= 0) {
        Rf_error("FillTau: How is this, OnCodaTLocjj = %d we can't fill!\n", OnCodaTLocjj);
      }
      fwrite( CodaTLocjj, sizeof(bufferILocType), OnCodaTLocjj, iTFile );
      fclose(iTFile); TotalClosedFiles++;
      OnCodaTLocjj = 0;
    }
  }
  if (CodaTLocjj != NULL) {
    CodaTLocjj[OnCodaTLocjj] = TotOnCodaT;   OnCodaTLocjj++;  TotCodaTLocjj++;
  }
  if (Verbose >= 2) {
    Rprintf("  FillCodaTau: put into position %d TotOnCodaT = %d\n",
      OnCodaT, TotOnCodaT); R_FlushConsole();
  }
  CodaTjj[OnCodaT] = -(tt+1);

  if (Verbose >= 2) {
    Rprintf("  FillCodaTau: put into position %d on CodaTau TotalOnTaus = %d\n",
      OnCodaT, TotalOnTaus); R_FlushConsole();
  }
  if (CodaTau == NULL) {
    Rf_error("FillCodatau: Won't happen here, CodaTau = NULL!\n"); R_FlushConsole();
  }
  if (OnCodaT+TotalOnTaus+1 >= LengthCodaTjj) {
    Rf_error("FillCodatau: Error, we have LengthCodaTjj = %d, OnCodaT = %d at TotalOnTaus =%d, Write will fail!\n",
      LengthCodaTjj, OnCodaT, TotalOnTaus);
  }
  CodaTau[OnCodaT] = (double) TotalOnTaus;
  OnCodaT++;  TotOnCodaT++;  
  TotalOnTaus = 0;
  for (int taut = 0; taut < Rf_length(sOnTau); taut++) {
    if (REAL(sOnTau)[taut] > 0.0) {
      //if (OnCodaT >= LengthCodaTjj) {
      //  Rf_error("FillCodatau: Error, we have LengthCodaTjj = %d, OnCodaT = %d at tau=%d, Real=%f Write, TotOnCodAT=%d, TotalOnTaus=%d\n",
      //    , OnCodaT, taut, REAL(sOnTau)[taut],
      //    TotOnCodaT, TotalOnTaus);
      //}
      CodaTjj[OnCodaT] = taut;
      CodaTau[OnCodaT] = REAL(sOnTau)[taut];
      OnCodaT++;  TotOnCodaT++;
      TotalOnTaus++;
    }
  }  
  LengthWrittenTauCodaBuffer += 1+TotalOnTaus;
  
  if (OnCodaT + TotalOnTaus+1 >= LengthCodaTjj) {
    if (Verbose >= 3) {
      Rprintf("FillTau Send to WriteTau at end, OnCodaT=%d, TotalOnTaus=%d, LengthCodaTjj=%d\n");
      R_FlushConsole();
    }
    AT = WriteTau();
    if (AT < 0) {
      Rprintf("********************************************************\n");
      Rprintf("** FillTau (At End): we get an AT = %d, after running WriteTau \n", AT);
      Rprintf("** Sign of failure, we return an error too! \n");
      return(-1);
    }
    OnCodaT = 0;
  }
  
  if (Verbose >= 4) {
    Rprintf("FillTau: Filled Buffer to length %d/%d to %d for iter %d\n",
      OnCodaT, LengthCodaTjj);
  }
  return(1);
}
int BayesSpikeCL::WriteTau() {
  if (Verbose >= 3) {
    Rprintf("WriteTau: Write Buffer[%d] to File: %s\n", OnCodaT,
      CHAR(STRING_ELT(RsCodaiTLocFile->asSexp(),0))); R_FlushConsole();
  }
   if (CodaTjj == NULL || CodaTau == NULL) {
    if (Verbose >= 2) {
      Rprintf("No Fill Tau if CodaTjj and CodaTau not set up!\n");
      R_FlushConsole();
    }
    return(-1);
  }
  
  if (OnCodaT <= 0) {
    if (Verbose >= 1) {
      Rprintf("WriteTau: No sense continuing OnCodaT = %d!\n", OnCodaT);
      R_FlushConsole();
    }
    return(0);
  }
  if (sOnTau == NULL || Rf_isNull(sOnTau) || Rf_length(sOnTau) <= 0) {
    Rf_error("No Fill Tau if there is no OnTau!\n");
  }
  if (sCodaiTFile == NULL || sCodadTFile == NULL) {
    if (Verbose >= 1) {
      Rprintf("No Fill Tau if sCodaiTFile aren't setup!\n");
      R_FlushConsole();
    }
    return(-1);
  }
  if (OnCodaT >= LengthCodaTjj) {
    Rprintf("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE\n");
    Rprintf("EE WriteTau() weird, OnCodaT=%d, LengthCodaTjj = %d.\n",
      OnCodaT, LengthCodaTjj); R_FlushConsole();
  }
  if (CodaTLocjj != NULL) {
      FILE *iTFile=NULL;
      if (NewWriteTLoc == 1) {
        iTFile = fopen(CHAR(STRING_ELT(RsCodaiTLocFile->asSexp(), 0)), "wb");
        if (iTFile != NULL) { TotalOpenedFiles++; }
        if (iTFile == NULL) {
          Rprintf("********************************************************\n");
          Rprintf("**  WriteTau: NewWriteTLoc=1, first time open fail!\n");
          Rprintf("** Tried to open %s \n",
            CHAR(STRING_ELT(RsCodaiTLocFile->asSexp(), 0))
            );
          Rprintf("**  Did not succeeed!\n"); R_FlushConsole();
          Rprintf("** I'll force this once more time!  \n");
          iTFile = fopen(CHAR(STRING_ELT(RsCodaiTLocFile->asSexp(), 0)), "wb");
          if (iTFile != NULL)  { TotalOpenedFiles++; }
          if (iTFile == NULL) {
            Rprintf("**  NOPE Unable to open up that file! \n"); R_FlushConsole();
          }
        }
        NewWriteTLoc = 0;
        WroteTotalTau = 0;
      } else {
        iTFile = fopen(CHAR(STRING_ELT(RsCodaiTLocFile->asSexp(), 0)), "ab");
        if (iTFile != NULL) { TotalOpenedFiles++; }
        if (iTFile == NULL) {
          Rprintf("********************************************************\n");
          Rprintf("**  WriteTau: NewWriteTLoc=1, second time open fail!\n");
          Rprintf("** Tried to open %s \n",
            CHAR(STRING_ELT(RsCodaiTLocFile->asSexp(), 0))
            );
          Rprintf("**  Did not succeeed!\n"); R_FlushConsole();
          Rprintf("** I'll force this once more time!  \n");
          iTFile = fopen(CHAR(STRING_ELT(RsCodaiTLocFile->asSexp(), 0)), "wb");
          if (iTFile != NULL) {TotalOpenedFiles++; }
          if (iTFile == NULL) {
            Rprintf("**  NOPE Unable to open up that file! \n"); R_FlushConsole();
          }
        }
      }
      if (iTFile == NULL) {
        Rprintf("***********************************************************\n");
        Rprintf("***********************************************************\n");
        Rprintf("** Look this is a dumb error but likely computer specific \n");
        Rprintf("** WriteTau: Error, cannot open CodaiTLocFile = %s!\n",
          CHAR(STRING_ELT(RsCodaiTLocFile->asSexp(), 0)));
        Rprintf("** We need to clear buffer and remove ! \n");
        FFree(CodaTLocjj, "CodaTLocjj"); CodaTLocjj = NULL;
        OnCodaTLocjj = 0; 
        DDelete(RsCodaiTLocFile, "RsCodaiTLocFile");  RsCodaiTLocFile = NULL;
        Rprintf("** WriteTau: RsCodaiTLocFile buffer is cleared ! \n"); R_FlushConsole();
        return(-1);
      }
      fwrite( CodaTLocjj, sizeof(bufferILocType), OnCodaTLocjj, iTFile );
      fclose(iTFile);   TotalClosedFiles++;  iTFile=NULL;
      WroteTotalTau += OnCodaTLocjj;  WroteAllTau += OnCodaTLocjj; 
      OnCodaTLocjj = 0;
  }
  FILE *iFile = NULL;  FILE *dFile = NULL;
  int OldTWrite = NewTWrite;
  if (NewTWrite == 1)  {
    iFile = fopen(CHAR(STRING_ELT(sCodaiTFile, 0)), "wb");
    dFile = fopen(CHAR(STRING_ELT(sCodadTFile, 0)), "wb");  
    NewTWrite=0;
  } else {
    iFile = fopen(CHAR(STRING_ELT(sCodaiTFile, 0)), "ab");
    dFile = fopen(CHAR(STRING_ELT(sCodadTFile, 0)), "ab");
  }
  if (dFile != NULL) { TotalOpenedFiles++;}
  if (iFile != NULL) { TotalOpenedFiles++; }
  if (iFile != NULL) {
    fwrite( CodaTjj, sizeof(int), OnCodaT, iFile );
    fclose(iFile);  TotalClosedFiles++;
  } else {
    Rprintf("***************************************************************\n");
    Rprintf("** WriteTau: Error ! \n"); R_FlushConsole();
    Rprintf("** WriteTau:  Uh oh, we get iTFile open faile on OldTWrite = %d \n",
      OldTWrite);
    Rprintf("** Try one more time to open  %s \n", CHAR(STRING_ELT(sCodaiTFile, 0)));
    if (OldTWrite == 1) {
      iFile = fopen(CHAR(STRING_ELT(sCodaiTFile, 0)), "wb");
    } else {
      iFile = fopen(CHAR(STRING_ELT(sCodaiTFile, 0)), "ab");
    }
    if (iFile != NULL) { TotalOpenedFiles++; }
    if (iFile == NULL) {
      Rprintf("**  Nope on second WriteTau: still a fail.  Clear iFile! \n");
      if (Verbose >= 2) {
        Rprintf("Freeing CodaTjj \n"); R_FlushConsole();
      }
      FFree(CodaTjj, "CodaTjj"); CodaTjj = NULL; OnCodaT = 0;
      if (Verbose >= 2) {
        Rprintf("Freeing RsCodaiTFile ! \n"); R_FlushConsole();
      }
      DDelete(RsCodaiTFile, "RsCodaiTFile"); RsCodaiTFile = NULL;
      Rprintf("** Everything is cleared returning from WriteTau Fail! \n");R_FlushConsole();
      sCodaiTFile = R_NilValue;
      LengthTotalWrittenTauCodaBuffer = 0;  LengthWrittenTauCodaBuffer = 0;
      if (dFile != NULL) { fclose(dFile); TotalClosedFiles++; dFile = NULL; }
      Rprintf("** clear dT information! \n");
      DDelete(RsCodadTFile, "sCodadTFile"); RsCodadTFile = NULL;
      sCodadTFile = R_NilValue; FFree(CodaTau, "CodaTau"); CodaTau = NULL;
      Rprintf("** Okay now quit WriteTau! \n"); R_FlushConsole();
      return(-1);
    } else {
      fwrite( CodaTjj, sizeof(int), OnCodaT, iFile );
      fclose(iFile); TotalClosedFiles++;
    }
  }
  if (dFile != NULL) {
    if (CodaTau != NULL) {
      fwrite( CodaTau, sizeof(double), OnCodaT, dFile );
    }
    fclose(dFile); TotalClosedFiles++;
  } else {
    Rprintf("***************************************************************\n");
    Rprintf("** WriteTau: Error ! \n"); R_FlushConsole();
    Rprintf("** WriteTau:  Uh oh, we get dTFile open faile on OldTWrite = %d \n",
      OldTWrite);
    Rprintf("** Try one more time to open  %s \n", CHAR(STRING_ELT(sCodadTFile, 0)));
    if (OldTWrite == 1) {
      dFile = fopen(CHAR(STRING_ELT(sCodadTFile, 0)), "wb");
    } else {
      dFile = fopen(CHAR(STRING_ELT(sCodadTFile, 0)), "ab");
    }
    if (dFile != NULL) { TotalOpenedFiles++;}
    if (dFile == NULL) {
      Rprintf("**  Nope on second WriteTau: still a fail.  Clear iFile! \n");
      Rprintf("** Clear CodaTau! \n"); R_FlushConsole();
      FFree(CodaTau, "CodaTau"); CodaTau = NULL; OnCodaT = 0;
      Rprintf("** Clear RsCodadTFile! \n"); R_FlushConsole();
      DDelete(RsCodadTFile, "RsCodadTFile"); RsCodadTFile = NULL;
      sCodaiTFile = R_NilValue;
      LengthTotalWrittenTauCodaBuffer = 0;  LengthWrittenTauCodaBuffer = 0;
      if (iFile != NULL) { fclose(iFile); iFile = NULL; TotalClosedFiles++; }
      if (Verbose >= 2) {
        Rprintf("Freeing CodaTjj \n"); R_FlushConsole();
      }
      FFree(CodaTjj, "CodaTjj"); CodaTjj = NULL; OnCodaT = 0;
      if (Verbose >= 2) {
        Rprintf("Freeing RsCodaiTFile ! \n"); R_FlushConsole();
      }
      DDelete(RsCodaiTFile, "RsCodaiTFile"); RsCodaiTFile = NULL;
      Rprintf("** Okay, WriteTau: everything cleared. \n"); R_FlushConsole();      
      return(-1);
    } else {
      fwrite( CodaTau, sizeof(double), OnCodaT, dFile );
      fclose(dFile);  TotalClosedFiles++;
    }
  }
  LengthTotalWrittenTauCodaBuffer +=  LengthWrittenTauCodaBuffer;
  LengthWrittenTauCodaBuffer = 0; 
  OnCodaT = 0;
  if (Verbose >= 2) {
    Rprintf("Successfully Wrote Tau Buffer to File!\n");
  }
  return(1);
}

int BayesSpikeCL::WriteTauFromTable(int sT, int eT) {
  FILE *iFile = NULL;  FILE *dFile = NULL;
  int oVerbose = Verbose-2;
  //oVerbose = 4;
  if (oVerbose >= 1) {
    Rprintf(" About to Write CodaT!\n"); R_FlushConsole();
    Rprintf("  Rf_isString(sCodaiTFile) = %d\n", Rf_isString(sCodaiTFile));
    Rprintf("  sCodaiTFile = %s. \n",  CHAR(STRING_ELT(sCodaiTFile, 0)) ); R_FlushConsole();
  }
  if (!Rf_isString(sCodaiTFile) || !Rf_isString(sCodadTFile)) {
     Rprintf("Error sCodaiTFile is not String!\n"); R_FlushConsole();
     Rprintf("What are we going to do about this string? : ");
     Rprintf(" it is %s \n", CHAR(STRING_ELT(sCodaiTFile, 0)) ); R_FlushConsole();
  }
  FILE *iTLocFile = NULL;
  if (NewTWrite == 1)  {
    if (Verbose >= 2) {
      Rprintf("WriteCodaT: Starting to open because NewTWrite = 1!\n");
    }
    iFile = fopen(CHAR(STRING_ELT(sCodaiTFile, 0)), "wb");
    dFile = fopen(CHAR(STRING_ELT(sCodadTFile, 0)), "wb"); 
    if (iFile != NULL) { TotalOpenedFiles++; } 
    if (dFile != NULL) { TotalOpenedFiles++; }
    if (CodaTLocjj != NULL && RsCodaiTLocFile != NULL) {
      iTLocFile = fopen(CHAR(STRING_ELT(RsCodaiTLocFile->asSexp(), 0)), "wb"); 
      if (iTLocFile != NULL) { TotalOpenedFiles++; }
      if (iTLocFile == NULL) {
        Rprintf("*********************************************************\n");
        Rprintf("**  WriteTauFromTable: woah, tried to open iTLoc first time\n");
        Rprintf("**  Got Fail Message!\n");
        FFree(CodaTLocjj, "CodaTLocjj"); CodaTLocjj = NULL;
        DDelete(RsCodaiTLocFile, "RsCodaiTLocFile"); RsCodaiTLocFile = NULL;
        if (iFile != NULL) { fclose(iFile); TotalClosedFiles++; }
        if (dFile != NULL) { fclose(dFile); TotalClosedFiles++;}
        return(-1);
      }
      NewWriteTLoc = 0;
    }
    if (Verbose >= 1) {
      Rprintf("WriteTauFromTable: Write to sCodaiTFile=%s for first time! \n", 
        CHAR(STRING_ELT(RsCodaiTFile->asSexp(), 0))); R_FlushConsole();
    }
    NewTWrite=0;
  } else {
    iFile = fopen(CHAR(STRING_ELT(sCodaiTFile, 0)), "ab");
    dFile = fopen(CHAR(STRING_ELT(sCodadTFile, 0)), "ab");
    if (iFile != NULL) {TotalOpenedFiles++; }
    if (dFile != NULL) { TotalOpenedFiles++; }
    if (CodaTLocjj != NULL && RsCodaiTLocFile != NULL) {
      if (NewWriteTLoc == 0) {
        iTLocFile = fopen(CHAR(STRING_ELT(RsCodaiTLocFile->asSexp(), 0)), "ab");  
      } else {
        iTLocFile = fopen(CHAR(STRING_ELT(RsCodaiTLocFile->asSexp(), 0)), "wb");
        NewWriteTLoc = 0;        
      }
      if (iTLocFile != NULL) { TotalOpenedFiles++; }
      if (iTLocFile == NULL) {
        Rprintf("*********************************************************\n");
        Rprintf("**  WriteTauFromTable: woah, tried to open iTLoc first time\n");
        Rprintf("**  Got Fail Message!\n");
        Rprintf("**  We'll still remove contents! \n"); R_FlushConsole();
        DDelete(RsCodaiTLocFile, "RsCodaiTLocFile"); RsCodaiTLocFile = NULL;
        FFree(CodaTLocjj, "CodaTLocjj"); CodaTLocjj  = NULL;
        if (iFile != NULL) { fclose(iFile); TotalClosedFiles++; }
        if (dFile != NULL) { fclose(dFile); TotalClosedFiles++; }
        return(-1);
      }
    }
  }
  if (iFile == NULL || dFile == NULL) {
    Rprintf("*************************************************************\n");
    Rprintf("** RecordCodaT:  Well, iFile is NULL, dFile is NULL             \n");
    Rprintf("** Seems like we face a file opening error.  Time to deal with it!\n");
    if (iFile != NULL) {
      fclose(iFile);  iFile = NULL;   TotalClosedFiles++; 
    }
    if (dFile != NULL) {
      fclose(dFile);   TotalClosedFiles++; 
    }
    if (iTLocFile != NULL) {
      fclose(iTLocFile);  iTLocFile = NULL;      TotalClosedFiles++; 
    }
    Rprintf("** We're going to quit, delete buffers, and run another day! \n");
    FFree(CodaTjj, "CodaTjj"); CodaTjj = NULL;  FFree(CodaTau, "CodaTau");
    DDelete(RsCodaiTFile, "RsCodaiTFile"); RsCodaiTFile = NULL;
    sCodaiTFile = R_NilValue;
    DDelete(RsCodadTFile, "sCodadTFile"); RsCodadTFile = NULL;
    sCodadTFile = R_NilValue;
    return(-1);
    Rf_error("RecordCodaT: Could Not open PrintFiles!");
  }
  if (CodaTjj != NULL && CodaTau != NULL && OnCodaT > 0) {
    if (oVerbose > 0) {
      Rprintf("RecordCodaT: Writing preexisting CodaTjj, CodaTau\n");R_FlushConsole();
    }
    fwrite( CodaTjj, sizeof(int), OnCodaT, iFile );
    fwrite( CodaTau, sizeof(double), OnCodaT, dFile ); 
    OnCodaT = 0;  NewTWrite = 0;
    fclose(iFile); fclose(dFile);   TotalClosedFiles+=2;
    if (RsCodaiTLocFile!=NULL && iTLocFile != NULL) { fclose(iTLocFile);  TotalClosedFiles++;  iTLocFile=NULL; }
    return(2);
  }
  if (sT <= 0 || eT <= 0) {
    if (iFile != NULL) { fclose(iFile);  iFile=NULL; TotalClosedFiles++; }
    if (dFile != NULL) { fclose(dFile); dFile=NULL; TotalClosedFiles++; }
    if (iTLocFile != NULL) { fclose(iTLocFile); iTLocFile=NULL; TotalClosedFiles++; }
    Rf_error("RecordCodaTau, sT and eT are Zero, sorry!");
  }
  if (CodaTable == NULL || Rf_isNull(CodaTable)) {
    if (iFile != NULL) { fclose(iFile);  iFile=NULL; TotalClosedFiles++; }
    if (dFile != NULL) { fclose(dFile); dFile=NULL; TotalClosedFiles++; }
    if (iTLocFile != NULL) { fclose(iTLocFile); iTLocFile=NULL; TotalClosedFiles++; }
    Rf_error("RecordCodaTau, CodaTable was Null, doh!");
  }
  SEXP sdCodaTable = Rf_getAttrib(CodaTable, R_DimSymbol);
  if (Rf_isNull(sdCodaTable) || Rf_length(sdCodaTable) < 2) {
    if (iFile != NULL) { fclose(iFile);  iFile=NULL; TotalClosedFiles++; }
    if (dFile != NULL) { fclose(dFile); dFile=NULL; TotalClosedFiles++; }
    if (iTLocFile != NULL) { fclose(iTLocFile); iTLocFile=NULL; TotalClosedFiles++; }
    Rf_error("RecordTau: CodaTable was not right dim");
  }
  int LDA = ANINT(sdCodaTable, 0);
  int TLen = ANINT(sdCodaTable, 1);
  if (sT > TLen || sT + eT > TLen) {
    if (iFile != NULL) { fclose(iFile);  iFile=NULL; TotalClosedFiles++; }
    if (dFile != NULL) { fclose(dFile); dFile=NULL; TotalClosedFiles++; }
    if (iTLocFile != NULL) { fclose(iTLocFile); iTLocFile=NULL; TotalClosedFiles++; }
    Rf_error("RecordTau: second dim of table is only %d, for sT = %d, eT = %d\n",
      TLen, sT, eT);
  }
  int tt1 = 0; int tt2P = 0; int nn1 = 0;
  int CNonZero = 0;  int ANew = 0;  int AllNew = 0;
  if (oVerbose > 3) {
    Rprintf("RecordCodaT: Counting number of nonzero Coefficients, sT = %d, eT = %d\n",
      sT, eT); R_FlushConsole();
    R_FlushConsole();
  }
  for (tt1 = 0; tt1 < LDA; tt1++) {
    nn1 = tt1 + sT * LDA; 
    ANew = 0;
    for (tt2P = sT; tt2P < sT + eT; tt2P++) {
      if (REAL(CodaTable)[nn1] > 0) {
        CNonZero++;  ANew++;
      }      
      nn1+=LDA;  
    }
    if (ANew > 0) {AllNew++;}
  }
  if (oVerbose > 3) {
    Rprintf("RecordCodaT: Counted %d non zeroes, AllNew = %d, nn1 = %d, now make Mem\n", 
      CNonZero, AllNew, nn1); R_FlushConsole();
    R_FlushConsole();
  }
  if (CNonZero == 0) {
    if (oVerbose > 1) {
      Rprintf("Warning, there are no nonzero Tau in this CodaTable!\n");
      int jj, kk;
      for (jj = 0; jj < 1; jj++) {
        Rprintf("%d : ");
        for (kk = 0; kk < eT; kk++) {
          Rprintf(" %.4f", REAL(CodaTable)[jj + (kk+sT)*LDA]);
          if (kk < eT-1) { Rprintf(","); }
        }
        Rprintf("\n"); R_FlushConsole();
      }
    }
    if (iFile != NULL) { fclose(iFile);  iFile=NULL; TotalClosedFiles++; }
    if (dFile != NULL) { fclose(dFile); dFile=NULL; TotalClosedFiles++; }
    if (iTLocFile != NULL) { fclose(iTLocFile); iTLocFile=NULL; TotalClosedFiles++; }
    return(-1);
  }
  RMemGetI(CodaTjj, "CodaTjj", CNonZero+1+AllNew);
  RMemGetD(CodaTau, "CodaTau", CNonZero+1+AllNew);
  int OnPrint = 0;
  if (oVerbose > 3) {
    Rprintf("RecordCodaT, CNonZero = %d, AllNew = %d \n", CNonZero, AllNew);
    Rprintf("RecordCodaT: Now Filling CodaTjj, CodaTau\n"); R_FlushConsole();
    R_FlushConsole();
  }
  for (tt1 = 0; tt1 < LDA; tt1++) {
    nn1 = tt1 + sT * LDA;
    ANew = 0;
    for (tt2P = sT; tt2P < sT + eT; tt2P++) {
      if (REAL(CodaTable)[nn1] > 0) { ANew++; break;}
      nn1 += LDA;
    }
    if (ANew > 0) {
      CodaTjj[OnPrint] = -(tt1+1);
      CodaTau[OnPrint] = -9.999;  OnPrint++;
    }
    nn1 = tt1 + sT * LDA;
    for (tt2P = sT; tt2P < sT + eT; tt2P++) {
      if (REAL(CodaTable)[nn1] > 0) {
        if (OnPrint >= CNonZero + AllNew) {
          Rprintf("Error: We're going to write something bad to CodaTjj at %d \n");
          Rprintf(" OnPrint = %d, CNonZero = %d, AllNew = %d, REAL(CodaTable)[%d] = %f",
            OnPrint, CNonZero, AllNew, nn1, REAL(CodaTable)[nn1]);
          R_FlushConsole();
        }
        CodaTjj[OnPrint] = tt2P-sT;
        CodaTau[OnPrint] = REAL(CodaTable)[nn1]; 
        OnPrint++;
      }   
      nn1+=LDA;
    }
  }  
  if (oVerbose > 3) {
    Rprintf("RecordCodaT: Now Writing CodaTjj, CodaTau\n"); R_FlushConsole();
    R_FlushConsole();
  }
  if (iFile != NULL) {
    fwrite( CodaTjj, sizeof(int), CNonZero+AllNew, iFile );
  } else {
    Rprintf("RecordCodaT:  Uh oh, iFile is null at this stage! \n"); R_FlushConsole();
    R_FlushConsole();
  }
  if (dFile != NULL) {
    fwrite( CodaTau, sizeof(double), CNonZero+AllNew, dFile ); 
  } else {
    Rprintf("RecordCodaT:  Uh oh, dFile is NULL at this stage! \n"); R_FlushConsole();
  }
  OnCodaT = 0;  NewTWrite = 0;
    if (iFile != NULL) { fclose(iFile);  iFile=NULL; TotalClosedFiles++; }
    if (dFile != NULL) { fclose(dFile); dFile=NULL; TotalClosedFiles++; }
    if (iTLocFile != NULL) { fclose(iTLocFile); iTLocFile=NULL; TotalClosedFiles++; }
  iFile = NULL; dFile = NULL;
  if (oVerbose > 2) {
    Rprintf("RecordCodaT: Freeing CodaTjj, CodaTau \n"); R_FlushConsole();
  }
  FFree(CodaTjj, "CodaTjj"); FFree(CodaTau, "CodaTau");
  OnCodaT = 0; 
  NewTWrite = 0;
  if (oVerbose > 3) {
    Rprintf("RecordCodaT: All Done, quit\n"); R_FlushConsole();
    R_FlushConsole();
  }
  return(1);
}


int BayesSpikeCL::RecordPandT () {
  int OldVerbose = this->Verbose;
  if (OldVerbose >= 6) {
    Rprintf("RecordPandT: and OldVerbose = %d\n", OldVerbose); R_FlushConsole();
  }
  //this->Verbose= 5;
  if (this->Verbose > -1) {
    Rprintf("RecordingPandT to P, T Files\n"); R_FlushConsole();
  }
  if (CodaTable == NULL || Rf_isNull(CodaTable)) { return(-1); }
  if (tt > INTEGER(Rf_getAttrib(CodaTable, R_DimSymbol))[0]) {
    Rf_error("RecordingPandT Cannot Record History Past End of Matrix!");
  }  
  //int LDA = INTEGER(Rf_getAttrib(CodaTable, R_DimSymbol))[0];
  int D = INTEGER(Rf_getAttrib(CodaTable, R_DimSymbol))[1];
  int sT = 0;
  int sP = 0; int eP = 0;
  int CLen = 0;
  int TryToWriteTau;
  if (!Rf_isNull(sBeta) && (ANINT(DoRecord, 0)) == 1) {
    CLen = p;
    if (sP + p > D) {
      Rf_error("RecordPAndT: Not Enough Room for Beta!\n");
    }
    sT += CLen;  sP += CLen;
  }
  if (CLen > D) {
    Rf_error("RecordHistory: Error at sBeta, CLen = %d, D = %d\n",
      CLen, D);
  }

  if (sOnTau != NULL && !Rf_isNull(sOnTau)  && Rf_length(sOnTau) > 0 &&
   tauEndList != NULL && !Rf_isNull(tauEndList) && Rf_length(tauEndList) > 0
   && (ANINT(DoRecord,1)) == 1) {
    CLen = Rf_length(sOnTau);
    if (sT + Rf_length(sOnTau) > D) {
      Rf_error("Not enough room for Tau!");
    }
    if (Rf_isNull(sCodaiTFile)) {
      if (Verbose > 0) {
        Rprintf("RecordHistory Issue: sCodaiTFile is a NULL, not recording Tau!\n"); R_FlushConsole();
      }
    } else if (!Rf_isString(sCodaiTFile)) {
      if (Verbose > 0) {
        Rprintf("RecordHistory Issue: sCodaiTFile is not a String, not recording Tau!\n"); R_FlushConsole();
      }
    } else if (Rf_isNull(sCodadTFile)) {
      if (Verbose > 0) {
        Rprintf("RecordHistory Issue: sCodadTFile is a NULL, not recording Tau!\n"); R_FlushConsole();
      }
    } else if (!Rf_isString(sCodadTFile)) {
      if (Verbose > 0) {
        Rprintf("RecordHistory Issue: sCodadTFile is not a String, not recording Tau!\n"); R_FlushConsole();
      }
    } else {
      if (sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) >= 1) {
        TryToWriteTau = WriteTau();
        if (TryToWriteTau < 0) {
          Rprintf("***************************************************************\n");
          Rprintf("** Record P & T, I don't know why, but we could not write Tau \n");
          return(TryToWriteTau);
        }
      }
    }
  } else {
    CLen = 0;
  }
  sP += CLen;
    
  if (!Rf_isNull(sOnSigma)  && Rf_length(sOnSigma) > 0 &&
    (ANINT(DoRecord,2) == 1)) {
    CLen = Rf_length(sOnSigma);
    if (sP + Rf_length(sOnSigma) > D) {
      Rf_error("Not enough room for CurrentSigma!");
    }
    eP += CLen;
  }
  if (!Rf_isNull(sOnPiA)  && Rf_length(sOnPiA) > 0 &&
    (ANINT(DoRecord,3) == 1)) {
    CLen = Rf_length(sOnPiA);
    if (sP + Rf_length(sOnPiA) > D) {
      Rf_error("Not enough room for OnPiA!");
    }
    eP += CLen;
  }
  if (Rf_isNull(sCodaPFile) || !Rf_isString(sCodaPFile)) {
    if (Verbose > 0) {
      Rprintf("Error: sCodaPFile is not String!\n"); R_FlushConsole();
      Rprintf("  Not writing it to File \n"); R_FlushConsole();
    }
  } else if (eP > 0) {
    WriteP(sP, eP);
  }
  if (this->Verbose > 2) {
    Rprintf("Record PandT, finished, now quitting\n"); R_FlushConsole();
  }
  //this->Verbose = OldVerbose;
  return(1);
}

SEXP BayesSpikeCL::ReadSetPctBetas(SEXP sStartIter, SEXP sEndIter) {
  if (Verbose > 0) {
  Rprintf("BayesSpikeCL:: Verbose happens now to be %d, but we'll set it up \n", Verbose);
  R_FlushConsole();  //Verbose = 4;
  }
  if (Verbose > 0) {
    Rprintf("BayesSpikeCL::ReadSetPctBetas, Starting\n"); R_FlushConsole();
  }
  
  int LoadStartIter = 0;
  int LoadEndIter =0;
  if (Rf_isNull(sStartIter)) {
    Rf_protect(sStartIter = Rf_allocVector(INTSXP,1));
    if (Rf_isReal(sStartIter)) {
      REAL(sStartIter)[0] = StartIter;
    } else {
      INTEGER(sStartIter)[0] = StartIter; 
    }
    LoadStartIter = 1;
  }
  if (Rf_isNull(sEndIter)) {
    Rf_protect(sEndIter = Rf_allocVector(INTSXP,1));
    INTEGER(sEndIter)[0] = EndIter; LoadEndIter = 1;
  }
  if (Verbose > 0) {
    Rprintf("BayesSpikeCL::ReadSetPctBetas, Got StartIter = %d, sEndIter = %d\n",
      ANINT(sStartIter,0), ANINT(sEndIter,0)); R_FlushConsole();
  }
  SEXP sReturnNewCountBeta = R_NilValue;
  if (Verbose > 0) {
      Rprintf("BayesSpikeCL:: Trying to findVar in Frame\n",
      ANINT(sStartIter,0), ANINT(sEndIter,0)); R_FlushConsole();
  }
  Rf_protect(sReturnNewCountBeta = Rf_findVarInFrame( BayesSpikeNameSpace, 
  Rf_install("ReturnNewCountBeta")));

  if (Verbose > 0) {
      Rprintf("BayesSpikeCL:: We got the function we think \n",
      ANINT(sStartIter,0), ANINT(sEndIter,0)); R_FlushConsole();
  }
  SEXP sVerbose = R_NilValue;
  Rf_protect(sVerbose = Rf_allocVector(INTSXP, 1));
  INTEGER(sVerbose)[0]  = this->Verbose;
  if (Rf_isNull(sReturnNewCountBeta)) {
    Rf_unprotect(2 + LoadEndIter + LoadStartIter);
    Rf_error("BayesSpikeCL: Error: no sReturnNewCountBeta");
  }
 //##ChainIters = NULL, StartIter = 0, EndIter = 2000,
  // ##Verbose = FALSE, SubSetCoords = NULL, TotalNumDraws = -1
  SEXP call = R_NilValue;
  if (Verbose > 1) {
    Rprintf("About to call RTM if we try\n"); R_FlushConsole();
  }
  SEXP sNInstance = R_NilValue;
  Rf_protect(sNInstance = Rf_allocVector(INTSXP, 1));
  INTEGER(sNInstance)[0] = NInstance;
  Rf_protect(call = Rf_lcons( sReturnNewCountBeta, 
    Rf_cons(TBSR5,
      Rf_cons(sStartIter, 
        Rf_cons(sEndIter, 
          Rf_cons(sVerbose, R_NilValue))))));
  if (Verbose > 1) {
    Rprintf("Call has been created, now Eval the Call\n"); R_FlushConsole();
  }        
  SEXP RTM = R_NilValue;
  Rf_protect(RTM = Rf_eval(call, BayesSpikeNameSpace));
  if (Rf_isNull(RTM)) {
    Rf_unprotect(5 + LoadEndIter + LoadStartIter);
    Rf_error("BayesSpikeCL Destroy: Failed to Get Sub Coda From Files");
  }

  if (Verbose > 1) {
    Rprintf("GetPctBetas: Well, we returned to C++, checking out the vector\n"); R_FlushConsole();
    if (Rf_isNull(RTM)) {
      Rf_unprotect(5 + LoadEndIter + LoadStartIter);
      Rf_error("Doh, RTM is a NULL");
    }
    if (Rf_isVector(RTM)) {
      Rprintf("GetPctBetas:  RTM is a vector! \n"); R_FlushConsole();
    }
    Rprintf("  Supposed Length of RTM is %d \n", Rf_length(RTM));
  }     
  int One = 1;
  if (Rf_isNull(RTM)) {
    Rf_unprotect(5 + LoadEndIter + LoadStartIter);
    Rf_error("BayesSpikeCL:: GetPctBetas, RTM was returned NULL");
  }
  if (Rf_length(RTM) <= 0 || Rf_isNull(VECTOR_ELT(RTM,0)) ||
    !Rf_isReal(VECTOR_ELT(RTM,0))) {
    Rf_unprotect(5+ LoadEndIter + LoadStartIter);
    Rf_error("BayesSpikeCL:: GetPctBetas, RTM did not have any Betas");  
  } 
  double *AP = REAL(VECTOR_ELT(RTM,0));

  RMemGetD(PctBetas, "PctBetas", p);
  F77_CALL(dcopy)(&p, AP, &One, PctBetas, &One);
  if (Verbose > 0) {
    Rprintf("BayesSpikeCL:: GetPctBetas, Got Betas!");
  }
  if (Rf_length(RTM) >= 2 && !Rf_isNull(VECTOR_ELT(RTM,1)) &&
    Rf_isReal(VECTOR_ELT(RTM,1))) {
    int k = Rf_length(sOnTau);
    RMemGetD(PctTaus, "PctTaus", k);
    F77_CALL(dcopy)(&k, REAL(VECTOR_ELT(RTM,1)), &One,
      PctTaus, &One);  
  }  
  Rf_unprotect(5+LoadEndIter + LoadStartIter);    
  return(R_NilValue);
}

SEXP BayesSpikeCL::ReadFileCodas(SEXP StartIter, SEXP EndIter,
  SEXP SubSetCoords) {
  if (Verbose > 0) {
  Rprintf("BayesSpikeCL:: Verbose happens now to be %d, but we'll set it up \n", Verbose);
  R_FlushConsole();  //Verbose = 4;
  }
  
  if (Verbose > 0) {
    Rprintf("BayesSpikeCL::ReadFileCodas, Starting\n"); R_FlushConsole();
  }
  SEXP sReturnSubCodaFromFiles = R_NilValue;
  Rf_protect(sReturnSubCodaFromFiles = Rf_findVarInFrame( BayesSpikeNameSpace, 
  Rf_install("ReturnSubCodaFromFiles")));
  SEXP sVerbose = R_NilValue;
  Rf_protect(sVerbose = Rf_allocVector(INTSXP, 1));
  INTEGER(sVerbose)[0]  = this->Verbose;
  if (Rf_isNull(sReturnSubCodaFromFiles)) {
    Rf_error("BayesSpikeCL: Error: no sReturnSubCodaFromFiles");
  }
  if (Verbose > 0) {
    Rprintf("BayesSpikeCL::Well, we have a nonnull sReturnSubCodaFromFiles\n"); R_FlushConsole();
  }
  //##ChainIters = NULL, StartIter = 0, EndIter = 2000,
  // ##Verbose = FALSE, SubSetCoords = NULL, TotalNumDraws = -1
  SEXP call = R_NilValue;
  if (Verbose > 1) {
    Rprintf("About to call RTM if we try\n"); R_FlushConsole();
  }

  Rf_protect(call = Rf_lcons( sReturnSubCodaFromFiles, 
    Rf_cons(TBSRoo,
      Rf_cons(StartIter, 
        Rf_cons(EndIter, 
          Rf_cons(sVerbose, 
            Rf_cons(SubSetCoords, R_NilValue)))))));
  SEXP RTM = R_NilValue;
  Rf_protect(RTM = Rf_eval(call, BayesSpikeNameSpace));
  if (Rf_isNull(RTM)) {
    Rf_error("BayesSpikeCL Destroy: Failed to Get Sub Coda From Files");
  }
  Rf_unprotect(4);
  return(RTM);
}


int BayesSpikeCL::TestMemAllocateFailure() {
  Rprintf("*****************************************************************\n");
  Rprintf("** BayesSpikeGibbs.cpp:: TestMemAllocateFailure has been called. \n");
  Rprintf("** First check pXtX. \n");  R_FlushConsole();
  CheckpXtX();
  Rprintf("**    And Also TestMemAllocateFailure. \n"); R_FlushConsole();
  int COn = 0, ii;
  double CTotalOnXtResid = 0.0;  double MinOnXtResid = 0.0;  double MaxOnXtResid = 0.0;
  double sdOnXtResid = 0.0; double MinOffXtResid = 0;  double MaxOffXtResid = 0.0;
  double sdOffXtResid = 0.0;   double WorkIt = 0.0;
  double CTotalOffXtResid = 0.0;
  double CTotalBeta = 0.0;  int CountNonZeroBeta = 0;  double MinBeta=0.0; double MaxBeta = 0.0;
  double sdBeta = 0.0;
  if (Verbose > 3.0) {
    Rprintf("***TestMemAllocateFailure, sdBeta = %f, CTotalOffXtResid=%f, CTotalBeta=%f\n", 
      sdBeta, CTotalOffXtResid, CTotalBeta); 
  }
  
  double CTotalOnProbFixed = 0.0;  double MinOnProbFixed = -666.0;
  double MaxOnProbFixed = -666.0;  double sdOnProbFixed = 0.0;
  double CTotalOffProbFixed = 0.0;  double MinOffProbFixed = -666.0;
  double MaxOffProbFixed = -666.0;  double sdOffProbFixed = 0.0;  
  int GoStP = 0;
  if (tauEndList == NULL || Rf_isNull(tauEndList) || Rf_length(tauEndList) == 0 ||
    sOnTau == NULL || Rf_isNull(sOnTau) || Rf_length(sOnTau) == 0) {
    GoStP = p;
  } else if (iFirstRandom > 0) {
    GoStP = iFirstRandom;
  }
  if (GoStP > 0) {
    for (ii = 0; ii < GoStP; ii++) {
      if (BFixed[ii] == 1) {
        COn++;
        if (REAL(sBeta)[ii] != 0.0) {
          WorkIt = XtResid[ii] + XjSq[ii] * REAL(sBeta)[ii];
          CTotalBeta = fabs(REAL(sBeta)[ii]);  CountNonZeroBeta++;
          if (MinBeta <= 0.0) {
            CTotalBeta = fabs(REAL(sBeta)[ii]);
          } else if (MinBeta  > fabs(REAL(sBeta)[ii])) {
            MinBeta = fabs(REAL(sBeta)[ii]);
          }
          if (MaxBeta < fabs(REAL(sBeta)[ii])) {
            MaxBeta = fabs(REAL(sBeta)[ii]);
          }
        } else {WorkIt = XtResid[ii]; }
          if (MinOnXtResid <= 0.0) {
            MinOnXtResid = fabs(WorkIt);
          } else if (MinOnXtResid < fabs(WorkIt)) {
            MinOnXtResid = fabs(WorkIt);
          }
          if (MaxOnXtResid < fabs(WorkIt)) { MaxOnXtResid = fabs(WorkIt); }
          CTotalOnXtResid += fabs(WorkIt);
          sdOnXtResid += WorkIt*WorkIt;
          if (ProbFixed != NULL) {
            CTotalOnProbFixed += ProbFixed[ii];  
            if (MinOnProbFixed <= -665.0) {  MinOnProbFixed = ProbFixed[ii];
            } else if (MinOnProbFixed > ProbFixed[ii]) {
              MinOnProbFixed = ProbFixed[ii];
            }
            if (MaxOnProbFixed <= -665.0) {  MaxOnProbFixed = ProbFixed[ii];
            } else if (MaxOnProbFixed < ProbFixed[ii]) {
              MaxOnProbFixed = ProbFixed[ii];
            }
            sdOnProbFixed += ProbFixed[ii]*ProbFixed[ii];
          }
      } else {
        WorkIt = fabs(XtResid[ii]);
        if (MinOffXtResid == 0.0) { MinOffXtResid = WorkIt;
        } else if (MinOffXtResid < fabs(WorkIt)) {
          MinOffXtResid = fabs(WorkIt);
        }
        if (MaxOffXtResid < fabs(WorkIt)) {
          MaxOffXtResid = fabs(WorkIt);
        }
        CTotalOffXtResid += fabs(WorkIt);
        sdOffXtResid = WorkIt*WorkIt;
          if (ProbFixed != NULL) {
            CTotalOffProbFixed += ProbFixed[ii];  
            if (MinOffProbFixed <= -665.0) {  MinOffProbFixed = ProbFixed[ii];
            } else if (MinOffProbFixed > ProbFixed[ii]) {
              MinOffProbFixed = ProbFixed[ii];
            }
            if (MaxOffProbFixed <= -665.0) {  MaxOffProbFixed = ProbFixed[ii];
            } else if (MaxOffProbFixed < ProbFixed[ii]) {
              MaxOffProbFixed = ProbFixed[ii];
            }
            sdOffProbFixed += ProbFixed[ii]*ProbFixed[ii];
          }
      }
    }
  } 
  Rprintf("**  Number of Active Fixed Effects are %d/%d, p Total = %d\n", COn, GoStP, p);
  Rprintf("**  Mean OnXtResid = %.4f, sd OnXtResid = %.4f, min-max = %.4f,%.4f \n",
    CTotalOnXtResid / COn, sqrt(sdOnXtResid / (COn-1)), MinOnXtResid, MaxOnXtResid);
  R_FlushConsole();
  Rprintf("**  Number of InActive Fixed Effects are %d/%d, p Total = %d", GoStP-COn, GoStP, p); R_FlushConsole();
  Rprintf("**  Mean OffXtResid = %.4f, sd OffXtResid = %.4f, min-max = %.4f,%.4f \n",
    CTotalOffXtResid / (GoStP-COn), sqrt(sdOffXtResid / (GoStP-COn-1)), MinOffXtResid, MaxOffXtResid);
  R_FlushConsole();   
  Rprintf("**  Mean OnProbResid = %.4f, sdOnProbFixed = %.4f, min-max = %.4f,%.4f \n",
    CTotalOnProbFixed / (GoStP), sqrt(sdOnProbFixed / (GoStP-1)), MinOnProbFixed, MaxOnProbFixed);
  Rprintf("**  Note OnPiA[0] = %f, Sigma = %f \n", REAL(sOnPiA)[0], REAL(sOnSigma)[0]);
  R_FlushConsole();
  Rprintf("**  We Should Check XtX now. \n"); R_FlushConsole();
  if (RPiAPrior == NULL || Rf_isNull(RPiAPrior->asSexp())) {
    Rprintf("**  There is No RPiAPrior");
  } else if (Rf_length(RPiAPrior->asSexp()) == 2) {
    Rprintf("**  And PiAPrior = %f,%f \n", REAL(RPiAPrior->asSexp())[0],
      REAL(RPiAPrior->asSexp())[1]);
  } else if (Rf_length(RPiAPrior->asSexp()) == 2) {
    Rprintf("**  And PiAPrior = %f,%f and %f,%f \n", REAL(RPiAPrior->asSexp())[0],
      REAL(RPiAPrior->asSexp())[1], REAL(RPiAPrior->asSexp())[2],
      REAL(RPiAPrior->asSexp())[3]);
  }
  R_FlushConsole();
  Rprintf("**  Mean OffProbResid = %.4f, sdOffProbFixed = %.4f, min-max = %.4f,%.4f \n",
    CTotalOffProbFixed / (GoStP-COn), sqrt(sdOffProbFixed / (GoStP-COn-1)), MinOffProbFixed, MaxOffProbFixed);
  Rprintf("** We will now test XtResid.\n"); R_FlushConsole();
  TestXtResid();
  Rprintf("** Passed TestXtResid() \n");
  Rprintf("** Why do you think this messed up?\n");  R_FlushConsole();
  Rprintf("**************************************************************************\n");
  return(1);
  
}
int BayesSpikeCL::TestXtResid() {
  if (XtResid == NULL) {
    Rf_error("BayesSpikeGibbs.cpp:::TestXtResid(): XtResid does not exist!\n");
  }
  if (Verbose >= 3) {
    Rprintf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    Rprintf(">> Start TestXtResid. OnKappaS=%d/%d, tt=%d/%d\n", OnKappaS, p, tt, MaxGibbsIters); R_FlushConsole();
  }
  int kk = 0;
  double MyOn;  int AScore = 0;  int AScore2=0; int AScore3= 0;
  double TryTwo;
  double SumTheProd = 0.0;
  if (Verbose >= 3) {
      Rprintf(">> Allocate Vectors to store fails. \n"); R_FlushConsole();
  }
  SEXP sFails = R_NilValue;
  Rf_protect(sFails = Rf_allocVector(REALSXP, p));
  SEXP sFails2 = R_NilValue;
  Rf_protect(sFails2 = Rf_allocVector(REALSXP, p));
  SEXP sFails3 = R_NilValue;
  Rf_protect(sFails3 = Rf_allocVector(REALSXP, p));
  if (Verbose >= 3) {
      Rprintf(">> Allocated Vectors to store fails. \n"); R_FlushConsole();
  }
  if (iiWeight != NULL) {
    if (Verbose >= 3) {
      Rprintf(">> TestXtResid - iiWeight is Not Null. \n"); R_FlushConsole();
    }
    if (WW == NULL) {
      RMemGetD(WW, "WW", n);
    }
  }
  if (p <= 0) {
    Rf_error("BayesSpikeGibbs.cpp:TestXtResid(), no way, p = %d \n", p);
  }
  if (sX == NULL || Rf_isNull(sX) || Rf_length(sX) != n *p) {
    Rf_error("BayesSpikeGibs.cpp::TestXtResid(): Error, won't do, sX is bad Dim! %d but not %d,%d\n",
     sX != NULL && !Rf_isNull(sX) ? Rf_length(sX): 0, n, p);
  }
  if (sY == NULL || Rf_isNull(sY) || Rf_length(sY) < n) {
    Rf_error("BayesSpikeGibs.cpp::TestXtResid(): Error, won't do, sY is bad Dim!\n");
  }
  if (XtResid == NULL) {
    Rf_error("BayesSpikeGibbs.cpp::TestXtResid(): No XtResid is NULL!\n");
  }
  if (XtY == NULL) {
    Rf_error("BayesSpikeGibbs.cpp::TestXtResid(), no XtY is NULL \n");
  }
  if (iiWeight == NULL && Verbose >= 3) {
    Rprintf("BayesSpikeGibbs.cpp::TestXtResid(): Test iiWeight is definitely null version. \n"); R_FlushConsole();
  }
  if (iiWeight != NULL && Verbose >= 3) {
    Rprintf("BayesSpikeGibbs.cpp::TestXtResid(tt=%d/%d): Test iiWeight is not at all null version. \n", tt, MaxGibbsIters); R_FlushConsole();
  }
  if (Verbose >= 3) {
    Rprintf("BayesSpikeGibbs.cpp:: Force CheckpXtX"); R_FlushConsole();
    CheckpXtX();
  }
  
  if (Verbose >= 3) {
    Rprintf(">> BayesSpikeGibbs.cpp::TestXtResid Integrity. \n"); R_FlushConsole();
  }
  for (int ii = 0; ii < p; ii++) {
    XtResid[ii] = XtResid[ii] + 0.0;
  }
  if (Verbose >= 3) {
    Rprintf(">> BayesSpikeGibbs.cpp::TestXtResid, Test XtY Integrity. \n"); R_FlushConsole();
  }
  for (int ii = 0; ii < p; ii++) {
    XtY[ii] = XtY[ii] + 0.0;
  }
  if (Verbose >= 3) {
    Rprintf(">> BayesSpikeGibbs.cpp::TestXtResid, Onto the test the level. \n"); R_FlushConsole();
  }
  int XtYFail = 0;
  for (int ii = 0; ii < p; ii++) {
    if (Verbose >= 5) {
      Rprintf("BayesSpikeGibbs.cpp::TestXtResid On ii=%d/%d: ", ii, p); R_FlushConsole();
    }
    MyOn = 0.0;
    if (iiWeight == NULL) {
    for (int jj = 0; jj < n; jj++) {
      MyOn+=REAL(sX)[ii*n+jj] * REAL(sY)[jj];
    }
    } else {
      for (int jj = 0; jj < n; jj++) {
        MyOn+=REAL(sX)[ii*n+jj] * REAL(sY)[jj] * iiWeight[jj];
      }    
    }
    if (fabs(MyOn - XtY[ii]) >= .00001) {
      XtYFail++;
    }
    if (Verbose >= 5) {
      Rprintf("XtY Pass:"); R_FlushConsole();
    }
    if (OnKappaS > 0) {
    for (int kk = 0; kk < OnKappaS; kk++) {
      if (kXFinder[kk] < 0 || kXFinder[kk] >= p) {
        Rf_error("BayesSpikeGibbs.cpp:Test XtResid, why is kXFinder[kk=%d/%d] = %d? \n",
         kk, OnKappaS, kXFinder[kk]); R_FlushConsole();
      } else if (REAL(sBeta)[kXFinder[kk]] != 0.0) {
        SumTheProd = 0.0;
        if (iiWeight == NULL) {
        for (int jj = 0; jj < n; jj++) {
          SumTheProd += REAL(sX)[ii*n+jj] * REAL(sX)[kXFinder[kk] * n +jj];
        }
        } else {
        for (int jj = 0; jj < n; jj++) {
          SumTheProd += REAL(sX)[ii*n+jj] * REAL(sX)[kXFinder[kk] * n +jj]*iiWeight[jj];
        }        
        }
        MyOn -= SumTheProd * REAL(sBeta)[kXFinder[kk]];
      }
    }
    }
    if (Verbose >= 5) {
      Rprintf("XtR[ii=%d/%d]=%f right: ", ii, p, MyOn); R_FlushConsole();
    }
    REAL(sFails)[ii] = fabs(MyOn-XtResid[ii]);
    if (fabs(MyOn - XtResid[ii]) >= .0001) {
       AScore++;
    }
    TryTwo = XtY[ii];
    if (iWeightedXtX == NULL && iiWeight != NULL) {
      Rf_error("BayesSpikeGibbs.cpp:::TestXtResid() Error: We do not have iWeightedXtX\n");
    }
    if (OnKappaS > 0) {
    for (int kk = 0; kk < OnKappaS; kk++) {
      if (REAL(sBeta)[kXFinder[kk]] != 0.0) {
        if (iiWeight != NULL && iWeightedXtX != NULL && iWeightedXtX[kk] == 0) {
          if (WW == NULL) {
            RMemGetD(WW, "WW", n);
          }
          for (int iti = 0; iti < n; iti++) {
            WW[iti] = iiWeight[iti] * REAL(sX)[kXFinder[kk]*n+ii];
          }
          int MyOne = 1;
          TryTwo-= F77_CALL(ddot)(&n, WW, &MyOne, REAL(sX)+ii*n, &MyOne) * REAL(sBeta)[kXFinder[kk]];
        } else {
          if (ii >= p || ii < 0) {
            Rprintf("BayesSpikeGibbs.cpp:::TestXtXResid: we're not doing right. ii=%d/%d\n", ii, OnKappaS); R_FlushConsole();
          }
          TryTwo -= (*(pXtX[kk]+ii)) * REAL(sBeta)[kXFinder[kk]];
        }
      }
    }
    }
    if (Verbose >= 5) {
      Rprintf("Calced TryTwo = %f \n", TryTwo); R_FlushConsole();
    }
    REAL(sFails2)[ii] = fabs(TryTwo-XtResid[ii]);
    REAL(sFails3)[ii] = fabs(TryTwo-MyOn);
    if (fabs(TryTwo- XtResid[ii]) >= .0001) {
      AScore2++;
    }
    if (fabs(TryTwo- MyOn) >= .0001) {
      AScore3++;
    }
  }
  if (AScore <= 0) {
    if (Verbose >= 3) {
      Rprintf("TestXtResid: Test shows that all XtResid are Okay! kk = %d\n", kk); R_FlushConsole();
    }  
    Rf_unprotect(3);
    return(0);
  }
  Rprintf("ERRORERRROERRRORERRORERRORERRORERRRORERRORERRRORERRROR\n"); R_FlushConsole();
  Rprintf("****************************************************************************\n");
  Rprintf("**  ERRORS BayesSpikeGibbs.cpp:::TestXtResid Errors!  And tt=%d/%d, OnKappaS=%d/%d/p\n", tt, MaxGibbsIters, OnKappaS, OnKappaMem, p);
  if (dfRobit > 0 || dfTNoise > 0) {
    Rprintf("**  dfRobit = %f and dfTNoise = %f \n", dfRobit, dfTNoise);
    R_FlushConsole();
    int iiWeightNo = 0;
    if (OnKappaS > 0 && iWeightedXtX != NULL) {
      for (int iti = 0; iti < OnKappaS; iti++)  {
        if (iWeightedXtX[iti] == 0) {
          iiWeightNo++;
        }
      }
    }
    Rprintf("**  out of OnKappaS=%d/%d, we have there are %d unweighted\n", OnKappaS, OnKappaMem, iiWeightNo);
    Rprintf("** We must note that XtYFail = %d, so this is bad. \n", XtYFail); R_FlushConsole();
  } else {                                                                  
    Rprintf("** No dfRobits are set \n");
  }
  Rprintf("**  TestXtResid: TestXtResid we have %d fails. \n", AScore);
  Rprintf("**  Also, when we count XtR composed Xt(Y-XBeta) we get %d between brute.\n",
    AScore2);
  Rprintf("**    This Xt(Y-XBeta) matrix form is %d different from XtResid. \n", AScore3);
 
   Rprintf("**    Print XtResid At sFails, which are %d \n", AScore);
  R_FlushConsole();
  int OneTwenty = 0;
  for (int ii = 0; ii < p; ii++) {
     if (OneTwenty >= 20) { Rprintf("\n"); ii = p+1; OneTwenty= 0; break;}
     if (REAL(sFails)[ii]  >= .0001) {
       Rprintf("%d:%.3f, ", ii, XtResid[ii]); R_FlushConsole();
       OneTwenty++;
     }
  }
   
  Rprintf("**    Print XtResid sFails, there are %d \n", AScore);
  R_FlushConsole();
  OneTwenty = 0;
  for (int ii = 0; ii < p; ii++) {
     if (OneTwenty >= 20) { Rprintf("\n"); ii = p+1; OneTwenty= 0; break;}
     if (REAL(sFails)[ii]  >= .0001) {
       Rprintf("%d:%.3f, ", ii, REAL(sFails)[ii]); R_FlushConsole();
       OneTwenty++;
     }
  }
  Rprintf("**    There were %d  This Xt(Y-XBeta) matrix form is different from XtResid. \n", AScore);
  Rprintf("**\n**\n");
  Rprintf("**    Now here are those AScore2=%d sFails2 where Xt(Y-XBeta) using F77_CALL is different from XtResid[ii] \n", AScore2);
  R_FlushConsole();
  OneTwenty = 0;
  for (int ii = 0; ii < p; ii++) {
     if (OneTwenty >= 20) { Rprintf("\n"); OneTwenty= 0; ii = p+1;  break;}
     if (REAL(sFails2)[ii]  >= .0001) {
       Rprintf("%d:%.3f, ", ii, REAL(sFails2)[ii]); R_FlushConsole();
       OneTwenty++;
     }
  }
  Rprintf("**\n**\n");
  Rprintf("** Now here are those sFails3 where Xt(Y-XBeta) using F77_CALL is different from Xt(Y-XBeta) slow. \n",
    AScore3);
  R_FlushConsole();
  OneTwenty = 0;
  for (int ii = 0; ii < p; ii++) {
     if (OneTwenty >= 20) { Rprintf("\n"); OneTwenty= 0; ii = p+1;  break;}
     if (REAL(sFails3)[ii]  >= .0001) {
       Rprintf("%d:%.3f, ", ii, REAL(sFails3)[ii]); R_FlushConsole();
       OneTwenty++;
     }
  }
  Rf_unprotect(3);
  Rprintf("**  Remember that OnKappaS=%d/%d/%d\n", OnKappaS, OnKappaMem, p); R_FlushConsole();
  Rprintf("************************************************************************************\n");
  return(AScore);
}

void BayesSpikeCL::RenameCodaList(SEXP dimnames) {
  int ii;
  SEXP sOut;
  if (CodaList == NULL ||
    Rf_isNull(CodaList) || Rf_length(CodaList) <= 0) {
    if (Verbose >= 0) {
      Rprintf("Cannot Rename Coda list because it is null!\n");
      R_FlushConsole();
    }
    return;
  }
  if (Verbose >= 5) {
    Rprintf("BayesSpikeCL:: Attempting to RenameCodaList.\n"); R_FlushConsole();
  }
  if (Rf_length(dimnames) != 2) {
    Rprintf("BayesSpikeCL: Hey, that's weird, ");
    Rprintf(" dimnames does not have length 2 is %d\n",
      Rf_length(dimnames)); R_FlushConsole();
  }
  for (ii = 0; ii < Rf_length(CodaList); ii++) {
     sOut = VECTOR_ELT(CodaList, ii);
     if (!Rf_isNull(sOut)) {
       Rf_setAttrib(sOut, R_DimNamesSymbol, dimnames);
     }
  }
  if (Verbose >= 2) {
    Rprintf("BayesSpikeCL:: Done trying to set CodaList.\n");
    R_FlushConsole();
  }
  return;
}
int BayesSpikeCL::CountAllNewCoords() {
  if ( (iFirstRandom > 0 || tauEndList==NULL || 
    Rf_isNull(tauEndList) || Rf_length(tauEndList) <= 0) && BFixed == NULL) {
    Rf_error("CountAllNewCoords: BFixed is not set up!\n");
  }
  if ( (iFirstRandom > 0 || tauEndList==NULL || 
    Rf_isNull(tauEndList) || Rf_length(tauEndList) <= 0) && 
      (sBeta == NULL || Rf_isNull(sBeta) || Rf_length(sBeta) <= 0)) {
    Rf_error("CountAllNewCoords: Beta is not set up!\n");
  }
  int TestNewFixed = 0;  int TestAllNew = 0;
  int ii, St;
  if (tauEndList != NULL && !Rf_isNull(tauEndList) && Rf_length(tauEndList) >= 1 && 
    iFirstRandom > 0  && iFirstRandom < p && BFixed != NULL) {
    if (BFixed == NULL) {
      Rprintf("BayesSpikeGibbs.cpp::: CountAllNewCoords: Error \n"); R_FlushConsole();
      Rf_error("CountAllNewCoords: BFixed is NULL, cannot do this!");
    }
    for (ii = 0; ii < iFirstRandom; ii++) {
      if (BFixed[ii] > 0 && XLC[ii] < 0) { TestNewFixed++; TestAllNew++; }
    }
  } else if (tauEndList == NULL || Rf_isNull(tauEndList) || Rf_length(tauEndList) <= 0) {
    if (BFixed == NULL) {
      Rprintf("BayesSpikeGibbs.cpp::: CountAllNewCoords: Error \n"); R_FlushConsole();
      Rf_error("CountAllNewCoords: BFixed is NULL, cannot do this!");
    }
    for (ii = 0; ii < p; ii++) {
      if ((REAL(sBeta)[ii] != 0.0 || BFixed[ii] > 0) && XLC[ii] < 0) { TestNewFixed++; TestAllNew++; }
    }
  } else  {
    //for (ii = 0; ii < p; ii++) {
    //  if ((REAL(sBeta)[ii] != 0.0 || BFixed[ii] > 0) && XLC[ii] < 0) { TestNewFixed++; TestAllNew++; }
    //}  
  }
  if (tauEndList != NULL && !Rf_isNull(tauEndList) && Rf_length(tauEndList) >= 1 &&
    sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) == Rf_length(tauEndList)) {
    for (ii = 0; ii < Rf_length(tauEndList);ii++) {
      if (ii == 0) { St = iFirstRandom; } else {
        St = ANINT(tauEndList, ii-1)+1;
      }
      if (REAL(sOnTau)[ii] > 0.0) {
        if (XLC[St] < 0) {
          TestAllNew += ANINT(tauEndList,ii)+1-St;
        }
      } else {
        for (int jj = St; jj <= ANINT(tauEndList,ii); jj++) {
          if (REAL(sBeta)[jj] != 0.0) {
            TestAllNew += ANINT(tauEndList,ii)+1-St;
            jj = ANINT(tauEndList,ii)+1;
            break;
          }
        }
      }
    }  
  }
  AllNewCoords = TestAllNew;
  NewFixedCoords =  TestNewFixed;
  return(AllNewCoords);
}
int BayesSpikeCL::TestAllNewCoords() {
  if ( (iFirstRandom > 0 || tauEndList==NULL || 
    Rf_isNull(tauEndList) || Rf_length(tauEndList) <= 0) && BFixed == NULL) {
    Rf_error("TestAllNewCoords: BFixed is not set up!\n");
  }
  
  int TestNewFixed = 0;  int TestAllNew = 0;
  int ii, St;
  if (Verbose >= 3) {
    Rprintf("tttt  BayesSpikeGibbs.cpp:::TestAllNewCoords()  we are starting with Fixed. \n"); R_FlushConsole();
  }
  if (tauEndList != NULL && !Rf_isNull(tauEndList) 
    && Rf_length(tauEndList) >= 1 
    && iFirstRandom > 0  && iFirstRandom < p) {
    if (BFixed == NULL) {
      Rprintf("tttt ERROR:: BayesSpikeGibbs.cpp:::TestAllNewCoords(), we have NULL BFixed. \n");
      R_FlushConsole();
      Rf_error("ERROR : No BFixed. \n");
    }
    for (ii = 0; ii < iFirstRandom; ii++) {
      if (BFixed[ii] > 0 && XLC[ii] < 0) { TestNewFixed++; TestAllNew++; }
    }
  } else if (tauEndList == NULL || Rf_isNull(tauEndList) || Rf_length(tauEndList) <= 0) {
     if (BFixed == NULL) {
      Rprintf("tttt ERROR:: BayesSpikeGibbs.cpp:::TestAllNewCoords(), we have NULL BFixed. tauEndList Also NULL.\n");
      R_FlushConsole();
      Rf_error("ERROR : No BFixed. \n");
    }
    for (ii = 0; ii < p; ii++) {
      if ((REAL(sBeta)[ii] != 0.0 || BFixed[ii] > 0) && XLC[ii] < 0) { TestNewFixed++; TestAllNew++; }
    }
  } else {
    //for (ii = 0; ii < p; ii++) {
    //  if ((REAL(sBeta)[ii] != 0.0 || BFixed[ii] > 0) && XLC[ii] < 0) { TestNewFixed++; TestAllNew++; }
    //}  
  }
  if (Verbose >= 3) {
    Rprintf("tttt  BayesSpikeGibbs.cpp:::TestAllNewCoords(): On to tauEndList. \n"); R_FlushConsole();
  }
  //int jj;
  if (tauEndList != NULL && !Rf_isNull(tauEndList) && Rf_length(tauEndList) >= 1 &&
    sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) == Rf_length(tauEndList)) {
    for (ii = 0; ii < Rf_length(tauEndList);ii++) {
      if (REAL(sOnTau)[ii] > 0.0) {
        if (ii == 0) { St = iFirstRandom; } else {
          St = ANINT(tauEndList, ii-1)+1;
        }
        if (XLC[St] < 0) {
          TestAllNew += ANINT(tauEndList,ii)+1-St;
        }
      } 
    }  
  }
  if (Verbose >= 3) {
    Rprintf("tttt  BayesSpikeGibbs.cpp:::TestAllNewCoords(): After Random Test, TestAllNew = %d. \n", TestAllNew); R_FlushConsole();
  }
  if (TestNewFixed != NewFixedCoords || TestAllNew != AllNewCoords) {
    Rprintf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    Rprintf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    Rprintf("xx TestAllFixed: On tt = %d, we get an issue, TestNewFixed=%d, \n", tt, TestNewFixed);
    Rprintf("xx   and NewFixedCoords=%d, AllNewCoords=%d but TestAllNew = %d.\n",
       NewFixedCoords, AllNewCoords, TestAllNew); R_FlushConsole();
    Rprintf("xx  Why is this? \n");
    if (AllNewCoords < TestAllNew) {
      Rprintf("xx Since AllNewCoords=%d is less, we replace with TestAllNew = %d\n", AllNewCoords, TestAllNew); R_FlushConsole();
      NewFixedCoords = TestNewFixed;
      AllNewCoords = TestAllNew; 
    }
    return(-1);  
  }
  return(1);
}
int BayesSpikeCL::AddAllNewCoords() {
   if (Verbose >= 4) {
     Rprintf("**  BayesSpikeGibbs.cpp:::AddAllNewCoords(tt=%d): We're about TestAllNewCoords. \n",tt); R_FlushConsole();
   }
   TestAllNewCoords();
   if (Verbose >= 4) {
     Rprintf("**  BayesSpikeGibbs.cpp:::AddAllNewCoords(tt=%d): Finished with TestAllNewCoords. \n", tt); R_FlushConsole();
   }
   //CheckkXToXLC((char*) "AddAllNewCoords Start");
   //if (tt <= 1) {
   //  Rprintf("** NOTE: BayesSpikeGibbs.cpp::AddAllNewCoords(tt=%d), we will be testing OrderedActive here!\n", tt); R_FlushConsole();
   //}
   //int ACountTA = TestOrderedActive();
   //if (ACountTA >= 1) {
   //  Rprintf("**  Uh Oh Error, BayesSpikeGibbs.cpp:AddAllNewCoords(tt=%d) at top, we have ACountTA=%d, fail for TestOrderedActive()", tt, ACountTA);
   //  Rf_error("**  TestOrderedActive inside AddAllNewCoords start has ACountTA = %d!\n", ACountTA);
   //}

   if (Verbose >= 2) {
     Rprintf("** AddAllNewCoords(tt=%d): We have succeeded \n", tt); R_FlushConsole();
   }
   //int CountXtResid = TestXtResid();
   //
   //if (CountXtResid > 0) {
   //    Rprintf("**  AddAllNewCoords error, at begining, we gout CountXtResid = %d, bad count!\n", CountXtResid);
   //    Rprintf("**  tt = %d, AllNewCoords=%d, OnKappaS = %d, bad situation!\n", tt, AllNewCoords, OnKappaS);
   //    Rf_error("BayesSpikeGibbs.cpp:: AddAllNewCoords XtResid is bad at beginning. \n"); R_FlushConsole();
   //}
   if (Verbose >= 2) {
     Rprintf("** AddAllNewCoords(tt=%d): OnKappaS=%d/%d add %d: Note that we pass XtResid path!\n", tt, OnKappaS, OnKappaMem, AllNewCoords);
   }

   if ( AllNewCoords <= 0 && NewFixedCoords <= 0) {
     if (Verbose >= 5) {
       Rprintf("*****************************************************************************************************\n");
       Rprintf("** BayesSpikeGibbs.cpp: AddAllNewCoords() tt=%d, No Coordinates to add!  ItersSinceRemoval = %d\n", 
         tt, ItersSinceRemoval); R_FlushConsole();
     }
     if (StartRemoval <= OnKappaS) {
       ItersSinceRemoval++;
     }
     AllNewCoords = 0;  NewFixedCoords = 0;
     return(5);
   }
   int OldKappaS = OnKappaS;
   int OldKappaMem = OnKappaMem;
   if (Verbose >= 4) {
       Rprintf("** BayesSpikeGibbs.cpp: AddAllNewCoords() tt=%d,  Will Add %d Coordinates, %d new fixed, ItersSinceRemoval = %d\n", 
         tt, AllNewCoords, NewFixedCoords, ItersSinceRemoval); R_FlushConsole();
   }
   if (StartRemoval  <= OnKappaS + AllNewCoords) {
     if (AllNewCoords + OnKappaS >= MaximumAllocation ||
       ItersSinceRemoval >= CheckRemovable) {
       Rprintf("**************************************************************\n");
       Rprintf("**************************************************************\n");
       Rprintf("** BayesSpikeGibbs.cpp: Well we're going to have to do an interesting delete attempt\n");
       Rprintf("** Potentially we will have to add memory. \n");
       Rprintf("**  AllNewcoords = %d OnKappaS = %d, MaximumAllocation=%d, StartRemoval=%d\n",
         AllNewCoords, OnKappaS, MaximumAllocation, StartRemoval); R_FlushConsole();
       Rprintf("**  ItemsSinceRemoval=%d, CheckRemovable=%d\n", 
         ItersSinceRemoval, CheckRemovable);  R_FlushConsole();
       Rprintf("**  We had OldKappaMem =%d, needed to add %d AllNewCoords\n", 
         OldKappaMem, AllNewCoords); R_FlushConsole();
       Rprintf("**  After Resize we have OnKappaS = %d, and OldKappaS = %d\n", 
         OnKappaS, OldKappaS); R_FlushConsole();
       Rprintf("** Note that sOnPiA = %f, %f\n", REAL(sOnPiA)[0], Rf_length(sOnPiA) >= 2 ?
           REAL(sOnPiA)[1] : -1); R_FlushConsole();
       if (!Rf_isNull(PiAPrior) && Rf_length(PiAPrior) >= 2) {
           Rprintf("** Note that PiAPrior = %f,%f,%f,%f\n", REAL(PiAPrior)[0],
           REAL(PiAPrior)[1], Rf_length(PiAPrior) >= 4 ? REAL(PiAPrior)[2] : -1,
           Rf_length(PiAPrior) >= 4 ? REAL(PiAPrior)[3] : -1);
           R_FlushConsole();     
        }
        Rprintf("** The Number Of Active are %d/%d\n", NumActive, p);
        Rprintf("*************************************************************\n");
        Rprintf("*************************************************************\n");
        Rprintf("** AddAllNewCoords(): Let's count those active\n");
        int CActive = 0;  int CNan = 0;  
        for (int ii = 0; ii < p; ii ++) {
          if (R_isnancpp(REAL(sBeta)[ii])) {
            CNan++;
          } else if (REAL(sBeta)[ii] != 0.0) {
            CActive++;
          } 
        }
        Rprintf("** There are %d active coefficients and %d NAN coefficients out of %d\n",
          CActive, CNan, p); R_FlushConsole();
        int CsOnTau = 0;  int CsNanTau = 0;
        for (int ii = 0; ii < Rf_length(sOnTau); ii++) {
          if (R_isnancpp(REAL(sOnTau)[ii])) { CsNanTau++;
          } else if (REAL(sOnTau)[ii] > 0) { CsOnTau++; }
        }
        Rprintf("** BayesSpikeGibbs.cpp: AddAllNewCoords(): There are %d nonzero sOnTau and %d NAN sOnTau out of %d\n", 
          CsOnTau, CsNanTau, Rf_length(sOnTau));
        Rprintf("** BayesSpikeGibbs.cpp: AddAllNewCoords(): Note tauPriorMean (taupriornu) = %f  and df (taupriordf) = %f\n", 
          get_staubarnu(), get_staupriordf());
        R_FlushConsole();
        Rprintf("** Perhaps one needs a stronger GroupPrior Mean.\n"); R_FlushConsole();
        Rprintf("** You've got more than OnKappaMem = %d, p = %d\n", OnKappaMem, p);
        Rprintf("** We have MaxSquaresAllocation = %d and MaximumAllocation=%d for safety\n", 
          MaxSquaresAllocation, MaximumAllocation); R_FlushConsole();  
        Rprintf("** Hopefully GiveAChanceDelete does a good job \n"); R_FlushConsole();
        Rprintf("**************************************************************\n");
        Rprintf("**************************************************************\n"); R_FlushConsole();
     }
     if (AllNewCoords + OnKappaS >= MaximumAllocation)  {
       if (Verbose >= 1) {
         Rprintf("** BayesSpikeGibbs.cpp: AddAllNewCoords(): we have AllNewCoords=%d, OnKappaS=%d, but MaximumAllocation=%d\n",
           AllNewCoords, OnKappaS, MaximumAllocation); R_FlushConsole();
         Rprintf("** We Must Eliminate %d+%d-%d=%d Coordinates\n",
           AllNewCoords, OnKappaS, MaximumAllocation = AllNewCoords+OnKappaS-MaximumAllocation); R_FlushConsole();
       }
       GiveItAChanceAndKillUselessCoeficients(AllNewCoords+OnKappaS-MaximumAllocation);
       ItersSinceRemoval = 0;
       /*if (Verbose >= 1) {
         Rprintf("** BayesSpikeGibbs.cpp: AddAllNewCoords(): we have AllNewCoords=%d, OldKappaS=%d, OnKappaS=%d, MaximumAllocation=%d\n",
           AllNewCoords, OldKappaS, OnKappaS, MaximumAllocation); R_FlushConsole();
       }
       int CountBad = QuickCheckWeightXtX();
        if (CountBad > 0) {
          Rprintf("*** AddAllNewCoords(tt=%d,%d): After Max Out GiveItAChanceAndKill, we get CountBad = %d\n", tt, AllNewCoords, CountBad); R_FlushConsole();
          Rf_error("***  Resize QuickCheckWeightXtX Fail. \n");
        }*/
     } else if (ItersSinceRemoval >= CheckRemovable) {
       if (Verbose >= 1) {
         Rprintf("** BayesSpikeGibbs.cpp: AddAllNewCoords(): we have AllNewCoords=%d, OnKappaS=%d, < MaximumAllocation=%d\n",
           AllNewCoords, OnKappaS, MaximumAllocation); R_FlushConsole();
         Rprintf("** But ItersSinceRemoval =%d, tt = %d, CheckRemovable=%d, time for a Kill\n",
           ItersSinceRemoval, tt, CheckRemovable); R_FlushConsole();
       } 
       GiveItAChanceAndKillUselessCoeficients(-1);
       /*CheckkXToXLC((char*) "AddAllNewCoords after GiveItAChanceAndKill");
       int CountBad = QuickCheckWeightXtX();
       if (CountBad > 0) {
         Rprintf("*** AddAllNewCoords(tt=%d,%d): After CheckRemovable Out GiveItAChanceAndKill, we get CountBad = %d\n", tt, AllNewCoords, CountBad); R_FlushConsole();
         Rf_error("***  Resize QuickCheckWeightXtX Fail. \n");
       } */
       ItersSinceRemoval = 0;
       if (Verbose >= 1) {
         Rprintf("**************************************************************\n");
         Rprintf("**************************************************************\n");
         Rprintf("** BayesSpikeGibbs.cpp: AddAllNewCoords(): After GiveItAChance on tt=%d.  \n", tt);
         Rprintf("** we have AllNewCoords=%d, OldKappaS=%d, OnKappaS=%d, MaximumAllocation=%d.\n",
           AllNewCoords, OldKappaS, OnKappaS, MaximumAllocation); R_FlushConsole();
       }    
     } else {
       ItersSinceRemoval++;
     }
   }
   if (AllNewCoords+OnKappaS >= MaximumAllocation) {
       Rprintf("**************************************************************************************************************\n");
       Rprintf("** ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR \n");
       Rprintf("** BayesSpikeGibbs.cpp::: AddAllNewCoords()  Error on tt = %d\n", tt); 
       Rprintf("*** ERROR: Memory Allocation Impossible tt = %d, OnKappaS=%d, AllNewCoords=%d, MaximumAllocation=%d\n",
        tt, OnKappaS, AllNewCoords, MaximumAllocation);  R_FlushConsole();
       Rprintf("*** BayesSpikeGibbs(): We cannot succeed with this algorithm \n");
       TestMemAllocateFailure();
       Rprintf("*** BayesSpikeGibbs(): AddAllNewCoords, why did TestMemAllocateFailure Say?\n");
       R_FlushConsole();
       Rf_error("*** AddAllNewCoords() we fail. \n");
   }
   double MultipleFact = 1.0;
   if (OnKappaS + AllNewCoords >= OnKappaMem) {
     MultipleFact = ( ((double)OnKappaS + AllNewCoords)/((double)OnKappaMem) - 1.0) * 2 + 1.0;
     if (OnKappaMem <= 512 && OnKappaMem * 2.0 > OnKappaS+2*AllNewCoords) {
       MultipleFact = 2.0;
     }
     if (MultipleFact * OnKappaMem >= MaximumAllocation) {
       Rprintf("***********************************************************************************************\n");
       Rprintf("** BayesSpikeGibbs.cpp:: AddAllNewCoords(), Risky Resize of Memory\n");
       Rprintf("** OnKappaS = %d, OnKappaMem=%d, NumActive=%d, AllNewCoords=%d, NewFixedCoords=%d \n",
         OnKappaS, OnKappaMem, NumActive, AllNewCoords, NewFixedCoords);
       Rprintf("** We are running against a maximum memory allocation if MultipleFact=%f, MaximumAllocation=%d\n",
         MultipleFact, MaximumAllocation);
       Rprintf("** If we wind up not having enough memory to allocate all of the coordinates we need this will fail!\n");
       Rprintf("** Correct MultipleFact to MaximumAllocation / OnKappaMem = %f.\n",
         ((double) MaximumAllocation) / ((double) OnKappaMem));
       MultipleFact = ((double) MaximumAllocation) / ((double) OnKappaMem);
     }
     if (Verbose >= 3) {
       Rprintf("***********************************************************************************************\n");
       Rprintf("** BayesSpikeGibbs.cpp:: AddAllNewCoords(), We must resize memory - tt = %d\n", tt);
       Rprintf("** OnKappaS = %d\n"); R_FlushConsole();
       Rprintf("** We will Multiply by %f, OnKappaMem=%d \n", MultipleFact, OnKappaMem); R_FlushConsole();
     }
     Resize(MultipleFact);
     if (p >= 100000) {
       Rprintf("*** BayesSpikeGibbs.cpp:: Add AllNewCoords, just resized to OnKappaMem=%d, OnKappaS=%d, AllNewCoords=%d\n", 
         OnKappaMem, OnKappaS, AllNewCoords);
     }
     if (OnKappaMem < AllNewCoords + OnKappaS) {
       Rprintf("***********************************************************************************************\n");
       Rprintf("** ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR\n"); R_FlushConsole();
       Rprintf("** Error: BayesSpikeGibbs.cpp:: AddAllNewCoords(), tt = %d, failure!\n", tt);
       Rprintf("**   We had OldKappaMem =%d, needed to add %d AllNewCoords\n", 
         OldKappaMem, AllNewCoords); R_FlushConsole();
       Rprintf("**  MultipleFact was %f, by default it should be %f.\n", MultipleFact,
         ( ((double)OnKappaS + AllNewCoords)/((double)OnKappaMem) - 1.0) * 2 + 1.0); R_FlushConsole();
       Rprintf("** After Resize we have OnKappaS = %d, and OldKappaS = %d\n", OnKappaS, OldKappaS); R_FlushConsole();
       Rprintf("** Note that sOnPiA = %f, %f\n", REAL(sOnPiA)[0], Rf_length(sOnPiA) >= 2 ?
           REAL(sOnPiA)[1] : -1); R_FlushConsole();
       if (!Rf_isNull(PiAPrior) && Rf_length(PiAPrior) >= 2) {
           Rprintf("** Note that PiAPrior = %f,%f,%f,%f\n", REAL(PiAPrior)[0],
           REAL(PiAPrior)[1], Rf_length(PiAPrior) >= 4 ? REAL(PiAPrior)[2] : -1,
           Rf_length(PiAPrior) >= 4 ? REAL(PiAPrior)[3] : -1);
           R_FlushConsole();     
        }
        Rprintf("** The Number Of Active are %d/%d.  \n", NumActive, p);
        Rprintf("************************************************************************************************\n");
        Rprintf("** BayesSpikeGibbs.cpp:: AddAllNewCoords(): Let's count those active\n");
        int CActive = 0;  int CNan = 0;  
        for (int ii = 0; ii < p; ii ++) {
          if (R_isnancpp(REAL(sBeta)[ii])) {
            CNan++;
          } else if (REAL(sBeta)[ii] != 0.0) {
          CActive++;
          } 
        }
        Rprintf("** There are %d active coefficients and %d NAN coefficients out of %d\n",
          CActive, CNan, p); R_FlushConsole();
        int CsOnTau = 0;  int CsNanTau = 0;
        for (int ii = 0; ii < Rf_length(sOnTau); ii++) {
          if (R_isnancpp(REAL(sOnTau)[ii])) { CsNanTau++;
          } else if (REAL(sOnTau)[ii] > 0) { CsOnTau++; }
        }
        Rprintf("** BayesSpikeGibbs.cpp: AddAllNewCoords(): There are %d nonzero sOnTau and %d NAN sOnTau out of %d\n", 
          CsOnTau, CsNanTau, Rf_length(sOnTau));
        Rprintf("** BayesSpikeGibbs.cpp: AddAllNewCoords(): Note tauPriorMean (taupriornu) = %f  and df (taupriordf) = %f", 
          get_staubarnu(), get_staupriordf());
        R_FlushConsole();
        Rprintf("** Perhaps one needs a stronger GroupPrior Mean.\n"); R_FlushConsole();
        Rprintf("** You've got more than OnKappaMem = %d, Factor = %f, p = %d.\n", OnKappaMem, MultipleFact, p);
        Rprintf("** We have MaxSquaresAllocation = %d and MaximumAllocation=%d for safety.\n", 
          MaxSquaresAllocation, MaximumAllocation); R_FlushConsole();
        Rprintf("** Too many active factors, BayesSpike Does not have enough memory to achieve this!\n"); R_FlushConsole();
        Rprintf("** Last Check Again. \n");
        TestMemAllocateFailure();
        Rprintf("** Fatal Error to end Algorithm \n");
        Rprintf("****************************************************************************************************\n");
        Rf_error("** BayesSpikeGibbs.cpp():: AllAllNewCoords() We will Not be able to pass memory requirements\n");
      }
   }
   
   if (Verbose >= 4) {
     Rprintf("** BayesSpikeGibbs.cpp()[tt=%d/%d]:: AddAllNewCoords(): About to start adding Coordinates AllNewCoords = %d\n", 
       tt, MaxGibbsIters, AllNewCoords); R_FlushConsole();
   }
   
   int TotalAdded = 0;
   int TotalFixed = 0;
   if (NewFixedCoords > 0) {
     TotalFixed = p;
     if (tauEndList != NULL && !Rf_isNull(tauEndList) &&
       iFirstRandom > 0 && iFirstRandom  < p) { TotalFixed = iFirstRandom; }
     if (TotalFixed > 0) {
     if (BFixed == NULL) {
       Rprintf("**ERROR BayesSpikeGibbs.cpp():: AddAllNewCoords, no BFixed is NULL!\n"); R_FlushConsole();
       Rf_error("** BFixed is NULL!\n");
     }
     for (int iti = 0; iti < TotalFixed; iti++) {
       if ((REAL(sBeta)[iti] != 0.0 || BFixed[iti] > 0) && XLC[iti] < 0) {
         AddCoord(iti);  TotalAdded++; BFixed[iti] = 1;
       }
     }
     }
     if (TotalAdded != NewFixedCoords) {
       Rprintf("** BayesSpikeGibbs.cpp():: AddAllNewCoords: Weird, We just added TotalAdded=%d, NewFixedCoords = %d, TotalFixed was %d why the conflict?\n",
         TotalAdded, NewFixedCoords, TotalFixed); R_FlushConsole();
       Rprintf("**     We are going to count some things.  \n");
       int TotalXLCLessZero = 0;
       int TotalBFixedG0 = 0;
       int TotalsBetaNotZero = 0;
       int TotalBFixedG0OrBetaNotZero = 0;
       int TotalXLCLessZeroAndStuff = 0;
       for (int iti = 0; iti < TotalFixed; iti++) {
         if (BFixed[iti] > 0) { TotalBFixedG0++; }
         if (REAL(sBeta)[iti] != 0.0) {
           TotalsBetaNotZero++; 
         }
         if (XLC[iti] < 0) {
           TotalXLCLessZero++;
         }
         if (REAL(sBeta)[iti] || BFixed[iti] > 0) {
           TotalBFixedG0OrBetaNotZero++;
         }
         if ((REAL(sBeta)[iti] != 0.0 || BFixed[iti] > 0) && XLC[iti] < 0) {
           TotalXLCLessZeroAndStuff++;
         }
       }
       Rprintf("**  We Had TotalBFixedG0=%d, TotalsBetaNotZero=%d, TotalXLCLessZero = %d \n",
         TotalBFixedG0, TotalsBetaNotZero, TotalXLCLessZero);
       Rprintf("**  Furthermore TotalBFixedG0OrBetaNotZero=%d, TotalXLCLessZeroAndStuff = %d \n",
         TotalBFixedG0OrBetaNotZero, TotalXLCLessZeroAndStuff);
     }
     if (NewFixedCoords == AllNewCoords) {
        if (Verbose >= 4) {
          Rprintf("** BayesSpikeGibbs.cpp():: AddAllNewCoords, FixedCoords was all it needs to add\n"); R_FlushConsole();
        }
     }
   }
   if (tauEndList != NULL && !Rf_isNull(tauEndList) && Rf_length(tauEndList) >= 1 &&
     AllNewCoords > NewFixedCoords) {
     int St = iFirstRandom;
     if (Verbose >= 4) {
       Rprintf("** BayesSpikeGibbs.cpp():: AddAllNewCoords: Attempt to add %d coordinates, OnKappaS=%d\n",
         AllNewCoords-NewFixedCoords, OnKappaS); R_FlushConsole();
     }
     for (int iti=0; iti < Rf_length(tauEndList); iti++) {
       if (Verbose >= 6) {
         Rprintf("**              On iti=%d/%d, OnTau[%d]=%f, XLC[St=%d]=%d \n", iti, Rf_length(tauEndList),
           iti, REAL(sOnTau)[iti], St, XLC[St]); R_FlushConsole(); 
       }
       if (REAL(sOnTau)[iti] > 0.0) {
         if (iti > 0) {
           St = ANINT(tauEndList, iti-1)+1;
         } else {
           St = iFirstRandom;
         }
         if (XLC[St] < 0) {
           if (Verbose >= 4) {
             Rprintf("** BayesSpikeGibbs.cpp():: AddAllNewCoords: iti = %d, XLC[St=%d] = %d, we'll add coords, OnTau=%f.\n",
               iti, St, XLC[St], REAL(sOnTau)[iti]); R_FlushConsole();
           }
           for (int jj = St; jj <= ANINT(tauEndList, iti); jj++) {
             if (XLC[jj] < 0) {
               AddCoord(jj);  TotalAdded++;
             }
           }
         }
       }
       //St = ANINT(tauEndList,iti)+1;
     }
     if (TotalAdded != AllNewCoords) {
       Rprintf("** BayesSpikeGibbs.cpp():: AddAllNewCoords: Finishing, but weird, AllNewCoords=%d, and TotalAdded=%d.\n",
         AllNewCoords, TotalAdded); R_FlushConsole();
     }
   }
   //CountXtResid = TestXtResid(); 
   //if (CountXtResid > 0) {
   //    Rprintf("**  AddAllNewCoords error, at end, we gout CountXtResid = %d, bad count!\n", CountXtResid);
   //    Rprintf("**  tt = %d, AllNewCoords=%d, OnKappaS = %d, bad situation!\n", tt, AllNewCoords, OnKappaS);
   //    Rf_error("BayesSpikeGibbs.cpp:: AddAllNewCoords XtResid is bad at end. \n"); R_FlushConsole();
   //}
   //if (Verbose >= 2) {
   //  Rprintf("** AddAllNewCoords(tt=%d): At End OnKappaS=%d/%d add %d: Note that we pass XtResid path!\n", tt, OnKappaS, OnKappaMem, AllNewCoords);
   //}
   //if (tt <= 1) {
   //  Rprintf("** NOTE: BayesSpikeGibbs.cpp::AddAllNewCoords(tt=%d), we will be testing OrderedActive here at end!\n", tt); R_FlushConsole();
   //}
   //int AOCountTA = TestOrderedActive();
   //if (AOCountTA >= 1) {
   //  Rprintf("**  Uh Oh Error, BayesSpikeGibbs.cpp:AddAllNewCoords(tt=%d) at top, we have ACountTA=%d, fail for TestOrderedActive()", tt, AOCountTA);
   //  Rf_error("**  TestOrderedActive inside AddAllNewCoords start has ACountTA = %d!\n", AOCountTA);
   //}
   

   Rprintf("** BayesSpikeGibbs.cpp()[tt=%d/%d]:: AddAllNewCoords -- All Finished, TotalAdded was %d, for AllNewCoords = %d, NumActive=%d\n", tt,
     MaxGibbsIters, TotalAdded, AllNewCoords, NumActive);
   AllNewCoords = 0;    NewFixedCoords = 0;
   //CheckkXToXLC((char*) "AddAllNewCorods, after all Added");
   return(1);
}


int BayesSpikeCL::WriteYBuffer() {
  FILE *YFile = NULL;
  if (LengthWrittenYBuffer<= 0 ) {
    return(0);
  }
  if (Verbose >= 5) {
    Rprintf("Starting Writing YBuffer tt=%d, =LengthWrittenYBuffer=%d/%d, Total Written now = %d, LengthTotalWrittenYBuffer\n",
      tt, LengthWrittenYBuffer, LengthYBuffer); R_FlushConsole();
  }
  if (NewYBufferWrite == 1)  {
    YFile = fopen(CHAR(STRING_ELT(RsYFile->asSexp(), 0)), "wb");
    if (YFile != NULL) { TotalOpenedFiles++; }
    NewYBufferWrite = 0;
  } else {
    YFile = fopen(CHAR(STRING_ELT(RsYFile->asSexp(), 0)), "ab");
    if (YFile != NULL) { TotalOpenedFiles++; }
    // int fSeekFail = fseek (PostProbFile , ((long) sizeof(double) * 
    //  (long) LengthTotalWrittenPostProb * LengthProbCoda), SEEK_SET);
     if(Verbose >= 3) {
       Rprintf("WritePostProb, YFile opened append, fSeekFail=%d, opened to append.\n");
       R_FlushConsole();
     }
  }
  if (YFile == NULL) {
    Rprintf("*************************************************************\n");
    Rprintf("** ERROR WriteYBuffer: Error opening: %s",
      CHAR(STRING_ELT(RsYFile->asSexp(), 0))); R_FlushConsole();
    Rprintf("** Second chance at opening! \n");
    YFile = fopen(CHAR(STRING_ELT(RsYFile->asSexp(), 0)), "wb");
    if (YFile != NULL) { TotalOpenedFiles++; }
    if (YFile == NULL) {
      Rprintf("** No, total fail at write YBuffer.  Clear YBuffer! \n");
      DDelete(RsYFile, "RsYFile"); RsYFile = NULL;
      FFree(YBuffer, "YBuffer");  YBuffer = NULL;
      LengthWrittenYBuffer = 0; LengthTotalWrittenYBuffer = 0;
      return(-1);
    } 

  }
  int MyWrite = -1; 
  if (Verbose >= 3) {  
    Rprintf("WriteFileCoda, writing YBuffer, LWPP = %d, LPC = %d \n",
      LengthWrittenYBuffer, LengthYBuffer); R_FlushConsole();
  }
  MyWrite = fwrite( (void*) YBuffer, 
    sizeof(double), LengthWrittenYBuffer*n, YFile ); 
  if (MyWrite != LengthWrittenYBuffer * n) {
    Rprintf("WriteYBuffer: we got MyWrite = %d, but LengthWritten=%d, n = %d!\n",
      MyWrite, LengthWrittenYBuffer, n);
    R_FlushConsole();
  }
  if (Verbose >= 3) {  
    Rprintf("writeFileCoda, wrote YBuffer, LWWBB = %d, LYBuffer = %d, MyWrite = %d \n",
      LengthWrittenYBuffer, LengthYBuffer, MyWrite); R_FlushConsole();
  }
  LengthTotalWrittenYBuffer +=  LengthWrittenYBuffer;
  fclose(YFile); TotalClosedFiles++; 
  LengthWrittenYBuffer = 0;
  if (Verbose >= 5) {
    Rprintf("Finished Writing YBuffer!\n"); R_FlushConsole();
  }
  return(1);
}
int BayesSpikeCL::WriteWeightBuffer() {
  FILE *WeightFile = NULL;
  if (LengthWrittenWeightBuffer <= 0 ) {
    return(0);
  }
  if (Verbose >= 5) {
    Rprintf("Starting Writing WeightBuffer tt=%d, =LengthWrittenWeightBuffer=%d/%d, Total Written now = %d, LengthTotalWrittenWeightBuffer\n",
      tt, LengthWrittenWeightBuffer, LengthWeightBuffer); R_FlushConsole();
  }
  if (NewWeightBufferWrite == 1)  {
    WeightFile = fopen(CHAR(STRING_ELT(RsWeightFile->asSexp(), 0)), "wb");
    if (WeightFile != NULL) { TotalOpenedFiles++; }
    NewWeightBufferWrite = 0;
  } else {
    WeightFile = fopen(CHAR(STRING_ELT(RsWeightFile->asSexp(), 0)), "ab");
    if (WeightFile != NULL) { TotalOpenedFiles++; }
    // int fSeekFail = fseek (PostProbFile , ((long) sizeof(double) * 
    //  (long) LengthTotalWrittenPostProb * LengthProbCoda), SEEK_SET);
     if(Verbose >= 3) {
       Rprintf("writePostProb, opened append, fSeekFail=%d, opened to append.\n");
       R_FlushConsole();
     }
  }
  if (WeightFile == NULL) {
    Rprintf("***************************************************************\n");
    Rprintf("** WriteWeightBuffer: Error opening: %s",
      CHAR(STRING_ELT(RsWeightFile->asSexp(), 0))); R_FlushConsole();
    Rprintf("** Try one more time opening! \n"); R_FlushConsole();
    WeightFile = fopen(CHAR(STRING_ELT(RsWeightFile->asSexp(), 0)), "wb");
    if (WeightFile != NULL) { TotalOpenedFiles++; }
    if (WeightFile == NULL) {
      Rprintf("**  Nope WeightFile could not be opened.  Freeing Buffer! \n");
      DDelete(RsWeightFile, "RsWeightFile"); RsWeightFile = NULL;
      FFree(WeightBuffer, "WeightBuffer"); WeightBuffer = NULL;
      LengthWrittenWeightBuffer = 0;  
      return(-1);
    }
  }
  int MyWrite = -1; 
  if (Verbose >= 3) {  
    Rprintf("writeFileCoda, writing WeightBuffer, LWPP = %d, LPC = %d \n",
      LengthWrittenWeightBuffer, LengthWeightBuffer); R_FlushConsole();
  }
  MyWrite = fwrite( (void*) WeightBuffer, 
    sizeof(double), LengthWrittenWeightBuffer*n, WeightFile ); 
  if (MyWrite != LengthWrittenWeightBuffer * n) {
    Rprintf("WriteWeightBuffer: we got MyWrite = %d, but LengthWritten=%d, n = %d!\n",
      MyWrite, LengthWrittenWeightBuffer, n);
    R_FlushConsole();
  }
  if (Verbose >= 3) {  
    Rprintf("WriteFileCoda, wrote WeightBuffer, LWWBB = %d, LWeightBuffer = %d, MyWrite = %d \n",
      LengthWrittenWeightBuffer, LengthWeightBuffer, MyWrite); R_FlushConsole();
  }
  LengthTotalWrittenWeightBuffer +=  LengthWrittenWeightBuffer;
  fclose(WeightFile);    TotalClosedFiles++; WeightFile=NULL;
  LengthWrittenWeightBuffer = 0;
  if (Verbose >= 5) {
    Rprintf("Finished Writing WeightBuffer!\n"); R_FlushConsole();
  }
  return(1);
}

int BayesSpikeCL::WritePiACodaBuffer() {
  FILE *PiAFile = NULL;
  if (LengthWrittenPiACodaBuffer<= 0 ) {
    return(0);
  }
  //int Verbose = 5;
  if (Verbose >= 2) {
    Rprintf("Starting Writing PiACodaBuffer tt=%d, =LengthWrittenPiACodaBuffer=%d/%d, Total Written now = %d, LengthTotalWrittenPiACodaBuffer\n",
      tt, LengthWrittenPiACodaBuffer, LengthPiACodaBuffer); R_FlushConsole();
  }
  if (PiACodaBuffer == NULL) {
    Rf_error("WritePiACodaBuffer: but PiACodaBuffer is NULL!\n");
  }
  int MyWrite = -1;
  if (NewWritePiACodaBuffer == 1)  {
    if (Verbose >= 1) {
      Rprintf("*************************************************** \n");
      Rprintf("** Writing First time to PiAFile\n"); R_FlushConsole();
    }
    PiAFile = fopen(CHAR(STRING_ELT(RsPiACodaFile->asSexp(), 0)), "wb");
    if (PiAFile != NULL) { TotalOpenedFiles++; }
    if (PiAFile == NULL) {
      Rprintf("*********************************************************\n");
      Rprintf("** WritePiACodaBuffer: ERROR ERROR ERROR \n"); R_FlushConsole();
      Rprintf("** Well, rejected to writing to PiAFile, this isn't well \n");
      R_FlushConsole();
      Rprintf("** We are going to close the PiA Buffer because your harddrive does not support it!\n");
      R_FlushConsole();
      DDelete(RsPiACodaFile, "RsPiACodaFile");  RsPiACodaFile = NULL;
      FFree(PiACodaBuffer, "PiACodaBuffer");   PiACodaBuffer = NULL;
      LengthWrittenPiACodaBuffer = 0;
      LengthPiACodaBuffer = 0;
      Rprintf("** PiACodaBuffer Cleared! \n"); R_FlushConsole();
      return(-1);
    }
    double Writer[2];
    if (Rf_length(sOnPiA) == 1) {
      Writer[0] = 1.0;
      MyWrite = fwrite( (void*) Writer, 
        sizeof(double), 1, PiAFile );    
    } else if (Rf_length(sOnPiA) == 2) {
      Writer[0] = 2.0;
      MyWrite = fwrite( (void*) Writer, 
        sizeof(double), 1, PiAFile );    
    }
    NewWritePiACodaBuffer = 0;
  } else {
    if (Verbose >= 2) {
      Rprintf("** Writing second time to PiAFile! \n"); R_FlushConsole();
    }
    PiAFile = fopen(CHAR(STRING_ELT(RsPiACodaFile->asSexp(), 0)), "ab");
    if (PiAFile != NULL) { TotalOpenedFiles++; }
    // int fSeekFail = fseek (PostProbFile , ((long) sizeof(double) * 
    //  (long) LengthTotalWrittenPostProb * LengthProbCoda), SEEK_SET);
    if (PiAFile == NULL) {
      Rprintf("*********************************************************\n");
      Rprintf("** WritePiACodaBuffer: ERROR ERROR ERROR \n"); R_FlushConsole();
      Rprintf("** Well, rejected on ab to writing to PiAFile, this isn't well \n");
      Rprintf("** We are going to close the PiA Buffer because your harddrive does not support it!\n");
      R_FlushConsole();
      DDelete(RsPiACodaFile, "RsPiACodaFile");  RsPiACodaFile = NULL;
      FFree(PiACodaBuffer, "PiACodaBuffer");   PiACodaBuffer = NULL;
      LengthWrittenPiACodaBuffer = 0;
      LengthPiACodaBuffer = 0;
      Rprintf("** PiACodaBuffer Cleared! \n"); R_FlushConsole();
      return(-1);
    }
     if(Verbose >= 3) {
       Rprintf("writePostProb, YFile opened append, fSeekFail=%d, opened to append.\n");
       R_FlushConsole();
     }
  }
  if (PiAFile == NULL) {
    Rprintf("WritePiACodaBuffer: Error opening: ",
      CHAR(STRING_ELT(RsPiACodaFile->asSexp(), 0))); R_FlushConsole();
    return(-1);
  }
  MyWrite = -1;
  if (Verbose >= 3) {  
    Rprintf("writeFileCoda, writing PiACodaBuffer, LWPP = %d, LPC = %d \n",
      LengthWrittenPiACodaBuffer, LengthPiACodaBuffer); R_FlushConsole();
  }
  if (LengthWrittenPiACodaBuffer > LengthPiACodaBuffer) {
    Rprintf("Uh Oh, Written PiA Buffer has length %d.\n",
      LengthWrittenPiACodaBuffer); R_FlushConsole();
  }
  if (Rf_length(sOnPiA) == 1) {
    if (Verbose >= 2) {
      Rprintf("writeFileCoda: PiACodaBuffer = %d\n.", LengthWrittenPiACodaBuffer);
      R_FlushConsole();
    }
    MyWrite = fwrite( (void*) PiACodaBuffer, 
      sizeof(double), LengthWrittenPiACodaBuffer*1, PiAFile ); 
    if (MyWrite != LengthWrittenPiACodaBuffer * 1) {
      Rprintf("WritePiACodaBuffer: PiA length 1 we got MyWrite = %d, but LengthWritten=%d, n = %d!\n",
        MyWrite, LengthWrittenPiACodaBuffer, 1);
      R_FlushConsole();
    } 
  } else if (Rf_length(sOnPiA) == 2) {
    if (Verbose >= 2) {
      Rprintf("writeFileCoda: PiACodaBuffer = %d\n.", LengthWrittenPiACodaBuffer);
      R_FlushConsole();
    }
    MyWrite = fwrite( (void*) PiACodaBuffer, 
      sizeof(double), LengthWrittenPiACodaBuffer*2, PiAFile );  
    if (MyWrite != LengthWrittenPiACodaBuffer * 2) {
      Rprintf("WritePiACodaBuffer: PiA length 2, we got MyWrite = %d, but LengthWritten=%d, n = %d!\n",
        MyWrite, LengthWrittenPiACodaBuffer, 2);
      R_FlushConsole();
    } 
  }

  if (Verbose >= 3) {  
    Rprintf("WriteFileCoda, wrote PiACodaBuffer, LWWBB = %d, LPiACodaBuffer = %d, MyWrite = %d \n",
      LengthWrittenPiACodaBuffer, LengthPiACodaBuffer, MyWrite); R_FlushConsole();
  }
  LengthTotalWrittenPiACodaBuffer +=  LengthWrittenPiACodaBuffer;
  fclose(PiAFile);  TotalClosedFiles++;
  LengthWrittenPiACodaBuffer = 0;
  if (Verbose >= 5) {
    Rprintf("Finished Writing PiACodaBuffer!\n"); R_FlushConsole();
  }
  return(1);
}



int BayesSpikeCL::WriteSigCodaBuffer() {
  FILE *SigCodaFile = NULL;
  if (LengthWrittenSigCodaBuffer<= 0 ) {
    return(0);
  }
  if (Verbose >= 5) {
    Rprintf("Starting Writing SigCodaBuffer tt=%d, =LengthWrittenSigCodaBuffer=%d/%d, Total Written now = %d, LengthTotalWrittenSigCodaBuffer\n",
      tt, LengthWrittenSigCodaBuffer, LengthSigCodaBuffer); R_FlushConsole();
  }
  int OldWriteSigCodaBuffer = NewWriteSigCodaBuffer;
  if (NewWriteSigCodaBuffer == 1)  {
    SigCodaFile = fopen(CHAR(STRING_ELT(RsSigCodaFile->asSexp(), 0)), "wb");
    if (SigCodaFile != NULL) { TotalOpenedFiles++; }
    NewWriteSigCodaBuffer = 0;
  } else {
    SigCodaFile = fopen(CHAR(STRING_ELT(RsSigCodaFile->asSexp(), 0)), "ab");
    if (SigCodaFile != NULL) { TotalOpenedFiles++; }
    // int fSeekFail = fseek (PostProbFile , ((long) sizeof(double) * 
    //  (long) LengthTotalWrittenPostProb * LengthProbCoda), SEEK_SET);
     if(Verbose >= 3) {
       Rprintf("writePostProb, YFile opened append, fSeekFail=%d, opened to append.\n");
       R_FlushConsole();
     }
  }
  if (SigCodaFile == NULL) {
    Rprintf("*********************************************************\n"); R_FlushConsole();
    Rprintf("**WriteSigCodaBuffer: Error opening: ",
      CHAR(STRING_ELT(RsSigCodaFile->asSexp(), 0))); R_FlushConsole();
    Rprintf("** One more try! \n");
    if (OldWriteSigCodaBuffer == 1) {
      SigCodaFile = fopen(CHAR(STRING_ELT(RsSigCodaFile->asSexp(), 0)), "wb");
      if (SigCodaFile != NULL) { TotalOpenedFiles++; }
    } else {
      SigCodaFile = fopen(CHAR(STRING_ELT(RsSigCodaFile->asSexp(), 0)), "ab");
      if (SigCodaFile != NULL) { TotalOpenedFiles++; }
    }
    if (SigCodaFile == NULL) {
      Rprintf("** WriteSigCodaBuffer: No Error fail.  must quit! \n");  R_FlushConsole();
      DDelete(RsSigCodaFile, "RsSigCodaFile"); RsSigCodaFile = NULL;
      FFree(SigCodaBuffer, "SigCodaBuffer"); SigCodaBuffer = NULL;
      Rprintf("** Okay, cleared SigCodaBuffer!  Should not return! \n"); R_FlushConsole();
      LengthTotalWrittenSigCodaBuffer = 0;    LengthTotalWrittenSigCodaBuffer = 0;
      return(-1);
    }
  }
  int MyWrite = -1; 
  if (Verbose >= 3) {  
    Rprintf("writeFileCoda, writing SigCodaBuffer, LWPP = %d, LPC = %d \n",
      LengthWrittenSigCodaBuffer, LengthSigCodaBuffer); R_FlushConsole();
  }
  MyWrite = fwrite( (void*) SigCodaBuffer, 
    sizeof(double), LengthWrittenSigCodaBuffer, SigCodaFile ); 
  if (MyWrite != LengthWrittenSigCodaBuffer) {
    Rprintf("WriteSigCodaBuffer: we got MyWrite = %d, but LengthWritten=%d, n = %d!\n",
      MyWrite, LengthWrittenSigCodaBuffer, n);
    R_FlushConsole();
  }
  if (Verbose >= 3) {  
    Rprintf("writeFileCoda, wrote SigCodaBuffer, LWWBB = %d, LSigCodaBuffer = %d, MyWrite = %d \n",
      LengthWrittenSigCodaBuffer, LengthSigCodaBuffer, MyWrite); R_FlushConsole();
  }
  LengthTotalWrittenSigCodaBuffer +=  LengthTotalWrittenSigCodaBuffer;
  fclose(SigCodaFile);  TotalClosedFiles++; SigCodaFile = NULL;
  LengthWrittenSigCodaBuffer = 0;
  if (Verbose >= 5) {
    Rprintf("Finished Writing SigCodaBuffer!\n"); R_FlushConsole();
  }
  return(1);
}

int BayesSpikeCL::FillSigCodaBuffer() {
  if (RsSigCodaFile== NULL) {
    return(0);
  }
  if (Verbose >= 4) {
    Rprintf("FillSigCodaBuffer: Filling LWPP = %d/%d\n",
      LengthWrittenSigCodaBuffer,LengthSigCodaBuffer);
  }
  if (SigCodaBuffer == NULL) {
    Rf_error("FillSigCodaBuffer: Hey, RsSigFile is not null but unsatisfactory  SigCodaBuffer!\n");
  } else if (LengthSigCodaBuffer <= 0) {
    Rf_error("FillSigCodaBuffer: Hey, RsSigFile is not null but LengthSigCodaBuffer = %d!\n", LengthSigCodaBuffer);
  } else if  (LengthSigCodaBuffer <= 0) {
    Rf_error("FillSigCodaBuffer: Hey, RsSigFile is not null but LengthSigCodaBuffer = %d!\n", LengthSigCodaBuffer);
  }
  if (LengthWrittenSigCodaBuffer >= LengthSigCodaBuffer) {
    int Out = WriteSigCodaBuffer();
    if (Out < 0) {
      Rprintf("FillSigCodaBuffer: ERROR, WriteSigCodaBuffer returns %d \n", Out);
      return(Out);
    }
    LengthWrittenSigCodaBuffer=0;
  }
  
  SigCodaBuffer[LengthWrittenSigCodaBuffer] =  REAL(sOnSigma)[0];
  LengthWrittenSigCodaBuffer++;
  if (LengthWrittenSigCodaBuffer >= LengthSigCodaBuffer)  {
     int Out2 = WriteSigCodaBuffer();
     if (Out2 < 0) {
       Rprintf("FillSigCodaBuffer: ERROR, WriteSigCodaBuffer returns %d \n", Out2);
       return(Out2);
     }
     LengthWrittenSigCodaBuffer=0;
  }
  if (Verbose >= 5) {
    Rprintf("FillSigCodaBuffer: Finished with LengthWrittenSigCodaBuffer=%d/%d\n",
      LengthWrittenSigCodaBuffer,LengthSigCodaBuffer);
  }
  return(1);
}        

int BayesSpikeCL::FillPiACodaBuffer() {
  if (RsPiACodaFile== NULL) {
    return(0);
  }
  if (Verbose >= 4) {
    Rprintf("FillPiABuffer: Filling LWPP = %d/%d\n",
      LengthWrittenPiACodaBuffer,LengthPiACodaBuffer);
  }
  if (PiACodaBuffer == NULL) {
    Rf_error("FillPiABuffer: Hey, RsYFile is not null but unsatisfactory  PiABuffer!\n");
  } else if (LengthPiACodaBuffer <= 0) {
    Rf_error("FillPiACodaBuffer: Hey, RsYFile is not null but LengthPiABuffer = %d!\n", LengthPiACodaBuffer);
  } else if  (LengthPiACodaBuffer <= 0) {
    Rf_error("FillPiACodaBuffer: Hey, RsYFile is not null but LengthPiABuffer = %d!\n", LengthPiACodaBuffer);
  }
  if (LengthWrittenPiACodaBuffer >= LengthPiACodaBuffer) {
    int Out = WritePiACodaBuffer();
    if (Out < 0) {
      Rprintf("FillPiACodaBuffer: ERROR, WritePiABuffer returns %d \n", Out);
      return(Out);
    }
    LengthWrittenPiACodaBuffer=0;
  }
  int One = 1;
  if (Verbose >= 5) {
    Rprintf("FillPiACodaBuffer: One = %d.\n", One); R_FlushConsole();
  }
  if (Rf_length(sOnPiA) == 1) {
    PiACodaBuffer[LengthWrittenPiACodaBuffer] = REAL(sOnPiA)[0];
  } else if (Rf_length(sOnPiA) == 2) {
    PiACodaBuffer[2*LengthWrittenPiACodaBuffer] = REAL(sOnPiA)[0];
    PiACodaBuffer[2*LengthWrittenPiACodaBuffer+1] = REAL(sOnPiA)[1];
  }
  LengthWrittenPiACodaBuffer++;
  if (LengthWrittenPiACodaBuffer >= LengthPiACodaBuffer)  {
     int Out2 = WritePiACodaBuffer();
     if (Out2 < 0) {
       Rprintf("FillPiABuffer: ERROR, WritePiABuffer returns %d \n", Out2);
       return(Out2);
     }
     LengthWrittenPiACodaBuffer=0;
  }
  if (Verbose >= 5) {
    Rprintf("FillPiABuffer: Finished with LengthWrittenPiABuffer=%d/%d\n",
      LengthWrittenPiACodaBuffer,LengthPiACodaBuffer);
  }
  return(1);
}          

int BayesSpikeCL::FillYBuffer() {
  if (RsYFile== NULL) {
    return(0);
  }
  if (Verbose >= 4) {
    Rprintf("FillYBuffer: Filling LWPP = %d/%d\n",
      LengthWrittenYBuffer,LengthYBuffer);
  }
  if (YBuffer == NULL) {
    Rf_error("FillYBuffer: Hey, RsYFile is not null but unsatisfactory  YBuffer!\n");
  } else if (LengthYBuffer <= 0) {
    Rf_error("FillYBuffer: Hey, RsYFile is not null but LengthYBuffer = %d!\n", LengthYBuffer);
  } else if  (LengthYBuffer <= 0) {
    Rf_error("FillYBuffer: Hey, RsYFile is not null but LengthYBuffer = %d!\n", LengthYBuffer);
  }
  if (LengthWrittenYBuffer >= LengthYBuffer) {
    int Out = WriteYBuffer();
    if (Out < 0) {
      Rprintf("FillYBuffer: ERROR, WriteYBuffer returns %d \n", Out);
      return(Out);
    }
    LengthWrittenYBuffer=0;
  }
  int One = 1;
  F77_CALL(dcopy)(&n, REAL(sY), &One, YBuffer + LengthWrittenYBuffer*n, &One);
  LengthWrittenYBuffer++;
  if (LengthWrittenYBuffer >= LengthYBuffer)  {
     int Out2 = WriteYBuffer();
     if (Out2 < 0) {
       Rprintf("FillYBuffer: ERROR, WriteYBuffer returns %d \n", Out2);
       return(Out2);
     }
     LengthWrittenYBuffer=0;
  }
  if (Verbose >= 5) {
    Rprintf("FillYBuffer: Finished with LengthWrittenYBuffer=%d/%d\n",
      LengthWrittenYBuffer,LengthYBuffer);
  }
  return(1);
}          


int BayesSpikeCL::FillWeightBuffer() {
  if (RsWeightFile== NULL) {
    return(0);
  }
  if (Verbose >= 5) {
    Rprintf("FillWeightBuffer:  Filling LWPP = %d/%d\n",
      LengthWrittenWeightBuffer,LengthWeightBuffer);
  }
  if (WeightBuffer == NULL) {
    Rf_error("FillWeightBuffer: Hey, RsWeightFile is not null but unsatisfactory  WeightBuffer!\n");
  } else if (LengthWeightBuffer <= 0) {
    Rf_error("FillWeightBuffer: Hey, RsWeightFile is not null but  LengthWeightBuffer = %d!\n", LengthWeightBuffer);
  } else if  (LengthWeightBuffer <= 0) {
    Rf_error("FillWeightBuffer: Hey, RsWeightFile is not null but LengthWeightBuffer = %d!\n", LengthWeightBuffer);
  }
  if (iiWeight == NULL) {
    Rf_error("FillWeightBuffer: Can't do because iiWeight is NULL!\n");
  }
  if (LengthWrittenWeightBuffer >= LengthWeightBuffer) {
    int Out = WriteWeightBuffer();
    if (Out < 0) {
      Rprintf("FillWeightBuffer: ERROR, WriteWeightBuffer returns %d \n", Out);
      return(Out);
    }
    LengthWrittenWeightBuffer=0;
  }
  int One = 1;
  F77_CALL(dcopy)(&n, iiWeight, &One, WeightBuffer + LengthWrittenWeightBuffer*n, &One);
  LengthWrittenWeightBuffer++;
  if (LengthWrittenWeightBuffer >= LengthWeightBuffer)  {
     int Out2 = WriteWeightBuffer();
     if (Out2 < 0) {
       Rprintf("FillWeightBuffer: ERROR, WriteWeightBuffer returns %d \n", Out2);
       return(Out2);
     }
     LengthWrittenWeightBuffer=0;
  }
  if (Verbose >= 5) {
    Rprintf("FillWeightBuffer: Finished with LengthWrittenWeightBuffer=%d/%d\n",
      LengthWrittenWeightBuffer,LengthWeightBuffer);
  }
  return(1);
}     

////////////////////////////////////////////////////////////////////////////////
// ReDrawFunction
//
//   If we are asked to fill a buffer until it has filled to ReadLen elements
//    however, we are only done with OnLoc elements, we first copy the remaining
//    unused elements to the beginning and just read in ReadLen-OnLoc
//   If we finally read in less than the buffer length, then the buffer length
//    is shrunk to account for this shortage.
//

// Buff: a Buffer of 
// ASize: sizeof(Buffer)
//
// OnLoc: What point is Buff read to, draw this back.
// LenCurrent: is the rest that Buff is filled to.
// LenWantAdd
// BuffFILE: a file pointer
// LenFinal
//
//  We then add to the Buff
// ReDrawFunction(bufferILoc, sizeof(long int), bufferiLocLength, WantLocBuffer, GetLocBuffer, (fCodaILocFile), OnB, "bufferLoc from fCodaILocFile");
#define ReDrawFunction(Buff, ASize, LenCurrent, LenWantAdd, LenFinal, BuffFILE, OnLoc, CurName) \
  if (OnLoc > 0 && OnLoc < LenCurrent )  {                       \
   for (jt = 0; (int) jt < (int) (LenCurrent - OnLoc); jt++) {   \
     Buff[jt] = Buff[jt+OnLoc];             }                    \
  }                                                              \
  if (LenWantAdd <= 0)  {                                        \
    Rprintf("ReDrawFunction(%s), ISSUE LenWantAdd: LenTheCurrent=%d, LenTheFinal=%d, but LenTheWantAdd = %d?\n",\
    CurName, LenCurrent, LenFinal, LenWantAdd);                                                    \
  }                                                                                       \
  result = fread(Buff + (LenCurrent-OnLoc), ASize, LenWantAdd, BuffFILE);                 \
  if ((int)result != (int) LenWantAdd) {                                                  \
     if (result == 0)        {                                                           \
       Rprintf("ReDrawFunction(%s): Oh No: Read Issue rtheesult=0, LentTheWantAdd=%d, LenTheCurrent=%d, OnTheLoc = %d\n",\
         CurName, LenWantAdd, LenCurrent, OnLoc); R_FlushConsole();                            \
     }                                                                                   \
      LenWantAdd = result;}                                       \
   LenFinal = result+(LenCurrent-OnLoc);                          \
   OnLoc = 0

#define ReDrawTwoBuffFunction(Buff1, Buff2, BuffSize1, BuffSize2, LenCurrent, LenWantAdd, LenFinal, BuffFILE1, BuffFILE2, PrepString, OnLoc) \
  if (OnLoc > 0 && OnLoc < LenCurrent) {                         \
    if(Verbose >= 2) {                                           \
      Rprintf("ReDrawBuffer for %s, On %d \n", PrepString, OnLoc);             \
      R_FlushConsole();                                                        \
    }                                                                          \
    for (jt = 0; (int) jt < (int) (LenCurrent - OnLoc); jt++) {                   \
      Buff1[jt] = Buff1[jt+OnLoc]; \
      Buff2[jt] = Buff2[jt+OnLoc]; \
    }                              \
  }                                \
  result = fread(Buff1 + (LenCurrent-OnLoc), BuffSize1, LenWantAdd, BuffFILE1); \
  if ((int)  result != (int) LenWantAdd) {                         \
    Rprintf("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE\n");   \
    Rprintf("EEE ReDrawTwoBuff we wanted %d but result was only %d\n", LenWantAdd, result); \
    Rprintf("EEE ReDrawTwoBuff, we have OnTLoc = %d, LenTCurrent=%d, LenTWantAdd=%d, LenTFinal=%d\n", \
      OnLoc, LenCurrent, LenWantAdd, OnLoc); R_FlushConsole();    \
    Rprintf("EEE This is an error and bad thing!\n");             \
    Rprintf("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE\n"); \
    R_FlushConsole();                                              \
    LenWantAdd = result;                  }                        \
   result = fread(Buff2 + (LenCurrent-OnLoc), BuffSize2, LenWantAdd, BuffFILE2); \
   if ((int) result !=(int) LenWantAdd) {                       \
     Rprintf("ERROR ERROR ERROR: ReDrawBuffer2 for %s, on %d \n",  \
        PrepString, OnLoc);                                        \
     Rprintf("ERROR Sorry, from file 1 we got Read=%d, but from 2 we got %d \n",\
       LenWantAdd, result);                                                     \
     Rprintf("ERROR Bad Coordination. \n"); R_FlushConsole();                   \
     Rf_error("ERROR in ReDrawTwoBuffF in BayesSpikeGibbs.cpp\n");}             \
   LenFinal = result+(LenCurrent-OnLoc);                                        \
   OnLoc = 0

#define ffclose(AF, TXT)                              \
  if (Verbose >= 2) {                                 \
    Rprintf("Closing %s \n", TXT);  R_FlushConsole(); \
  }                                                   \
  if (AF != NULL) { fclose(AF); }                     \
  AF = NULL
   
#define LaunchEnd(Lev)                                      \
  if (Verbose >= Lev) {                                     \
    Rprintf("LaunchEnd: we quit after.\n");                 \
    R_FlushConsole();                                       \
  }                                                         \
  FFree(bufferI, "bufferI");                                \
  FFree(bufferD, "bufferD");                                \
                                                            \
  if (Verbose >= 2) {                                       \
    Rprintf("Closing fCodaIFile \n"); R_FlushConsole();   \
  }                                                         \
  if (fCodaIFile != NULL) {                                 \
    weClosed = fclose(fCodaIFile);}                         \
    TotalClosedFiles++; \
  fCodaIFile = NULL;                                        \
  if (weClosed != 0) {                                      \
    Rprintf("GiveCodaSubset: Well, we failed to close sCodaIFile !\n");    \
    R_FlushConsole();                                       \
  }                                                         \
  if (Verbose >= 2) {                                       \
    Rprintf("Closing fCodaDFile \n"); R_FlushConsole();   \
  }                                                         \
  if (fCodaDFile != NULL) {                                 \
    weClosed = fclose(fCodaDFile);}                         \
    TotalClosedFiles++; \
  fCodaDFile = NULL;                                        \
  if (weClosed != 0) {                                      \
    Rprintf("GiveCodaSubset: Well, we failed to close sCodaJFile !\n");   \
    R_FlushConsole();                                       \
  }                                                         \
  if (Verbose >= 2) {                                       \
    Rprintf("Closing fCodaiTFile \n"); R_FlushConsole();  \
  }                                                         \
  if (fCodaiTFile != NULL) {                                \
    weClosed = fclose(fCodaiTFile);}                        \
    TotalClosedFiles++; \
    fCodaiTFile = NULL;                                     \
  if (Verbose >= 2) {                                       \
    Rprintf("Closing fCodadTFile \n"); R_FlushConsole();    \
  }                                                         \
  if (fCodadTFile != NULL) {                                \
    weClosed = fclose(fCodadTFile);}                        \
    TotalClosedFiles++; \
  fCodadTFile = NULL;                                       \
  if (Verbose >= 2) {                                       \
    Rprintf("Closing fCodaiTLocFile \n"); R_FlushConsole(); \
  }                                                         \
  if (fCodaiTLocFile != NULL) {                             \
    weClosed = fclose(fCodaiTLocFile);}                     \
    TotalClosedFiles++; \
  fCodaiTLocFile = NULL;                                    \
  if (Verbose >= 2) {                                       \
    Rprintf("Closing fCodaSigFile \n"); R_FlushConsole();   \
  }                                                         \
  if (fCodaSigFile != NULL) {                               \
    weClosed = fclose(fCodaSigFile);}                       \
    TotalClosedFiles++; \
  fCodaSigFile = NULL;                                      \
  if (fCodaPiAFile != NULL) {                               \
    weClosed = fclose(fCodaPiAFile);}                       \
    TotalClosedFiles++; \
  fCodaPiAFile = NULL;                                      \
  if (fCodaILocFile != NULL) {                              \
    weClosed = fclose(fCodaILocFile);}                      \
    TotalClosedFiles++; \
  fCodaILocFile = NULL;                                     \
  if (Verbose >= 2) {                                       \
    Rprintf("Closing fCodaProbFile start \n");              \
    R_FlushConsole();                                       \
  }                                                         \
  if (fCodaProbFile != NULL) {                              \
    weClosed = fclose(fCodaProbFile);}                      \
    TotalClosedFiles++; \
  fCodaProbFile = NULL;                                     \
  if (Verbose >= 2) {                                       \
    Rprintf("LeaveEnd: Done Closing Files!\n");             \
    R_FlushConsole();                                       \
  }                                                         \
  if (OnOrderedActive != NULL) {                            \
    if (Verbose >= 3) {                                     \
      Rprintf("Freeing OnOrderedActive\n");                 \
      R_FlushConsole();}                                    \
    Free(OnOrderedActive); OnOrderedActive = NULL;          \
  }                                                         \
  if (Verbose >= 2) {                                       \
    Rprintf("To Delete OnProbBeta \n"); R_FlushConsole();   \
  }                                                         \
  if (OnProbBeta != NULL) {                                 \
    if (Verbose >= 3) {                                     \
      Rprintf("Freeing OnProbBeta\n");                      \
      R_FlushConsole();  }                                  \
    Free(OnProbBeta); OnProbBeta = NULL;                    \
  }                                                         \
  if (OnTauIndices != NULL) {                               \
    if (Verbose >= 3) {                                     \
      Rprintf("Freeing OnTauIndices\n"); R_FlushConsole();} \
    Free(OnTauIndices); OnTauIndices = NULL;                \
  }                                                         \
  if (Verbose >= 3) {                                       \
    Rprintf("Off to close OnTauVals?\n"); R_FlushConsole(); \
  }                                                         \
  if (OnTauVals != NULL) {                                  \
    if (Verbose >= 3) {                                     \
      Rprintf("Freeing OnTauVals\n");  R_FlushConsole();}   \
    Free(OnTauVals); OnTauVals = NULL;                      \
  }                                                         \
  if (bufferiT != NULL) {                                   \
    if (Verbose >= 3) {                                     \
      Rprintf("Freeing bufferiT\n"); R_FlushConsole(); }    \
    Free(bufferiT); bufferiT = NULL;                        \
  }                                                         \
  if (Verbose >= 3) {                                       \
    Rprintf("Freeing bufferdT?\n"); R_FlushConsole();       \
  }                                                         \
  if (bufferdT != NULL) {                                   \
    if (Verbose >= 3) {                                     \
      Rprintf("Freeing bufferdT\n"); }                      \
    Free(bufferdT); bufferdT = NULL;                        \
  }                                                         \
  if (bufferILoc != NULL) {                                  \
    if (Verbose >= 3) {                                     \
      Rprintf("Freeing bufferILoc\n"); }                     \
    Free(bufferILoc); bufferILoc = NULL;                      \
  }                                                         \
  if (Verbose >= 3) {                                       \
    Rprintf("Free bufferSig?\n"); R_FlushConsole();         \
  }                                                         \
  if (bufferSig != NULL) {                                  \
    if (Verbose >= 3) {                                     \
      Rprintf("Freeing bufferSig\n"); R_FlushConsole();  }  \
    Free(bufferSig); bufferSig = NULL;                      \
  }                                                         \
  if (Verbose >= 3) {                                       \
    Rprintf("About to FreePiA?\n"); R_FlushConsole();       \
  }                                                         \
  if (bufferPiA != NULL) {                                  \
    if (Verbose >= 3) {                                     \
      Rprintf("Freeing bufferPiA\n"); R_FlushConsole(); }   \
    Free(bufferPiA); bufferPiA = NULL;                      \
  }                                                         \
  if (bufferProb != NULL) {                                 \
    if (Verbose >= 3) {                                     \
      Rprintf("Freeing bufferProb\n"); R_FlushConsole();}   \
    Free(bufferProb); bufferProb = NULL;                    \
  }                                                         \
  if (Verbose >= 3) {                                       \
    Rprintf("Freeing bufferILoc?\n"); R_FlushConsole();      \
  }                                                         \
  if (bufferILoc != NULL) {                                  \
    if (Verbose >= 3) {                                     \
      Rprintf("Freeing bufferILoc\n"); R_FlushConsole();}    \
    Free(bufferILoc); bufferILoc = NULL;                      \
  }                                                         \
  if (Verbose >= 3) {                                       \
    Rprintf("Big issue Local is done. \n");                 \
    R_FlushConsole();                                       \
  }                                                         \
  bufferILoc = NULL                                     


double BayesSpikeCL::CalcNewProb(int LenBeta, double *Beta, 
  int *IDBeta, int LenTau,  double *Tau, int *IDTau, 
  double AlterWeightdfRobit, double AlterWeightdfTNoise,
  double PiA1, double PiA2, double *EY, double Sigma) {
  int One = 1;
  int ii; 
  double ZeroD =0.0;
  //int Verbose = 9;
  if (Verbose >= 4) {
    Rprintf("CalcNewProb: Starting, with LenTau = %d\n"); R_FlushConsole();
  }
  F77_CALL(dscal)(&n, &ZeroD, EY, &One);
  for (ii = 0; ii < LenBeta; ii++) {
    F77_CALL(daxpy)(&n, Beta+ii, 
      REAL(sX) + n * IDBeta[ii], &One, 
      EY, &One); 
  } 
  if (Verbose >= 4) {
    int MaxPP = 10;
    if (n < MaxPP) { MaxPP = n; }
    Rprintf("After Fill we have EY = :");
    PrintVector(EY, MaxPP);
    Rprintf("\n"); R_FlushConsole();
  }
  double Total = 0;
  double isqrtSig = 1.0 / sqrt(Sigma);
  double APT = 0.0;  
  if (AlterWeightdfRobit == 0.0) {
    for (ii = 0; ii < n; ii++) {
      if (intY[ii] == 1) {
        APT= Rf_pnorm5( EY[ii]*isqrtSig, 0.0,1.0,1,1);
      } else {
        APT = Rf_pnorm5( EY[ii]*isqrtSig, 0.0,1.0,0,1);
      }
      if (R_isnancpp(APT) || !R_finite(APT)) {
        Rprintf("CalcNewProb: for dfRobit=%f, ii=%d, EY[%d]=%f, intY[%d]=%d, but APT=%f.\n",
          dfRobit, ii, ii, EY[ii], ii, intY[ii], APT);
        R_FlushConsole();
        Rf_error("CalcNewProb: Error, ii=%d/%d.\n", ii, n); 
      }
      Total += APT;
    }
  } else if (AlterWeightdfRobit > 0.0) {
    for (ii = 0; ii < n; ii++) {
      if (intY[ii] == 1) {
        Total += Rf_pt(EY[ii]*isqrtSig, AlterWeightdfRobit, 1, 1);
      } else {
        Total += Rf_pt(EY[ii]*isqrtSig, AlterWeightdfRobit, 0, 1);
      }
    }
  } else {
    if (AlterWeightdfTNoise < 0) {
      Rprintf("CalcNewProbError: AlterWeightdfRobit and AlterWeightdfTNoise are both < 0!\n",
        AlterWeightdfRobit, AlterWeightdfTNoise); R_FlushConsole();
      Rf_error("No!.\n");
    }
    
    for (ii = 0; ii < n; ii++) {
        APT += Rf_dt(isqrtSig* (REAL(sY)[ii] - EY[ii]), AlterWeightdfTNoise, 1);
        if (R_isnancpp(APT) || !R_finite(APT)) {
          Rprintf("CalcNewProb: for dfT=%f, ii=%d, EY[%d]=%f, intY[%d]=%d, but APT=%f\n",
            dfRobit, ii, ii, EY[ii], ii, intY[ii], APT);
          R_FlushConsole();
          Rf_error("CalcNewProb: Error, ii=%d/%d\n", ii, n); 
        }        
        Total+=APT;
    }  
  }
  if (Verbose >= 4) {
    Rprintf("CalcProb: after intY, pT analysis we have Total = %f.\n", Total); 
    R_FlushConsole();
  }
  if (!Rf_isNull(sOnPiA)) {
    if (!Rf_isNull(PiAPrior)) {
      Total += REAL(PiAPrior)[0] * log( PiA1 ) +
        REAL(PiAPrior)[1] * log(1.0 - PiA1 );
    }
  }
  if (Verbose >= 4) {
    Rprintf("CalcProb: After ProbY+PiAProor calc, we have Total = %f\n", Total); R_FlushConsole();
  }  
  int EndList = iFirstRandom;
  if (sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) > 0 &&
    tauEndList != NULL && !Rf_isNull(tauEndList)) {
    EndList = p;    
  }
  ii = 0;
  double Adder = 0.0;
  while (ii < LenBeta && IDBeta[ii] < EndList) {
      if (rPriorProbFixed != NULL && !Rf_isNull(rPriorProbFixed->asSexp()) &&
        Rf_length(rPriorProbFixed->asSexp()) == p && 
        (iFirstRandom < 0 || iFirstRandom > IDBeta[ii])) {
        Adder = REAL(rPriorProbFixed->asSexp())[IDBeta[ii]];
        if (R_isnancpp(Adder) || !R_finite(Adder) || R_IsNA(Adder) || R_IsNaN(Adder) ||
          Adder < 0.0 || Adder > 1.0) {
          if (iFirstRandom < 0 || IDBeta[ii] < iFirstRandom || Rf_length(sOnPiA) <= 1) {
            //Total += log(REAL(sOnPiA)[0]) - log(1.0-REAL(sOnPiA)[0]);
            Total += log(PiA1) - log(1.0-PiA1);
          } else {
            //Total += log(REAL(sOnPiA)[1]) - log(1.0-REAL(sOnPiA)[1]);
            Total += log(PiA2) - log(1.0-PiA2);
          }
        } else if (Adder == 0.0 || Adder == 1.0) {
        
        } else {
          Total += log(Adder)-log(1.0-Adder);
        }
      } else if (rPriorProbFixed != NULL && !Rf_isNull(rPriorProbFixed->asSexp()) &&
        Rf_length(rPriorProbFixed->asSexp()) == 2*p && 
        (iFirstRandom < 0 || iFirstRandom > IDBeta[ii])) {
        Adder = REAL(rPriorProbFixed->asSexp())[2*IDBeta[ii]];
        if (R_isnancpp(Adder)|| !R_finite(Adder) || R_IsNA(Adder) || R_IsNaN(Adder) ||
          Adder < 0.0 || Adder > 1.0) {
          if (iFirstRandom < 0 || IDBeta[ii] < iFirstRandom || Rf_length(sOnPiA) <= 1) {
            Total += log(REAL(sOnPiA)[0]) - log(1.0-REAL(sOnPiA)[0]);
          } else {
            Total += log(REAL(sOnPiA)[1]) - log(1.0-REAL(sOnPiA)[1]);
          }
        } else if (Adder == 0.0 || Adder == 1.0) {
        
        } else {
          Total += log(Adder)-log(1.0-Adder);
        }
      }
      if (TypePrior <= 1) {
        if (tauFixed == NULL || Rf_isNull(tauFixed) || Rf_length(tauFixed) <= 0) {
          Rprintf("BayesSpikeGibbs.cpp:CalcNewProb: Why does tauFixed not exist well?\n"); R_FlushConsole();
        } if (Rf_length(tauFixed) > 3 && Rf_length(tauFixed) > IDBeta[ii]) {
          if (REAL(tauFixed)[IDBeta[ii]] > 0.0) {
          Adder = -.5 * Beta[ii] * Beta[ii] / REAL(tauFixed)[IDBeta[ii]]-
            .5 * log(REAL(tauFixed)[IDBeta[ii]])-logSqrt2PI;
          } else {
          Adder = -.5 * Beta[ii] * Beta[ii] / (REAL(sOnSigma)[0] * fabs(REAL(tauFixed)[IDBeta[ii]]))-
            .5 * log(REAL(sOnSigma)[0] * fabs(REAL(tauFixed)[IDBeta[ii]]))-logSqrt2PI;          
          }
        } else {
          if (tauFixed == NULL || Rf_isNull(tauFixed) || Rf_length(tauFixed) <= 0) {
            Rprintf("BayesSpikeGibbs.cpp::SampleBFixed: hey, tauFixed is NULL \n"); R_FlushConsole();
          } else if (REAL(tauFixed)[0] > 0.0) {
          Adder = -.5 * Beta[ii] * Beta[ii] / REAL(tauFixed)[0]-
            .5 * log(REAL(tauFixed)[0])-logSqrt2PI;
          } else {
          Adder = -.5 * Beta[ii] * Beta[ii] / fabs(REAL(sOnSigma)[0] * REAL(tauFixed)[0])-
            .5 * log(fabs(REAL(tauFixed)[0]) * REAL(sOnSigma)[0] )-logSqrt2PI;          
          }
        }
        if (R_isnancpp(Adder)|| !R_finite(Adder)) {
          Rprintf("CalcProb: UhOh, ii=%d, Adder=%f \n", ii, Adder); R_FlushConsole();
          Rprintf(" -- tauFixed Length is %d, but IDBeta[ii=%d]=%d is %f\n",
            Rf_length(tauFixed), ii, IDBeta[ii], REAL(tauFixed)[IDBeta[ii]]); R_FlushConsole();
          Rprintf(" -- Beta[ii=%d] = %f \n", ii, Beta[ii]);
          Rf_error(" -- CalcProb: ii=%d, and EndList=%d, TP1: Adder fail for Prior tauFixed[IDBeta[ii=%d]=%d]=%f, TypePrior=%d<1 move.\n",
            ii, EndList, ii, IDBeta[ii], REAL(tauFixed)[IDBeta[ii]], TypePrior);
        }
        Total+=Adder;
      } else if (TypePrior == 3) {
        if (Rf_length(tauFixed) > 3 && Rf_length(tauFixed) > IDBeta[ii]) {
          if (REAL(tauFixed)[IDBeta[ii]] > 0.0) {
          Adder = - fabs( Beta[ii] * REAL(tauFixed)[IDBeta[ii]]) +
            log(REAL(tauFixed)[IDBeta[ii]]);
          } else {
          Adder = - fabs( Beta[ii] * REAL(sOnSigma)[0] * fabs(REAL(tauFixed)[IDBeta[ii]])) +
            log(REAL(sOnSigma)[0] * fabs(REAL(tauFixed)[IDBeta[ii]]));          
          }
        } else {
          if (REAL(tauFixed)[0] > 0.0) {
          Adder = - fabs( Beta[ii] *  REAL(tauFixed)[0] ) +
            log(REAL(tauFixed)[0]);
          } else {
          Adder = - fabs( Beta[ii] *  REAL(sOnSigma)[0] * REAL(tauFixed)[0] ) +
            log(REAL(sOnSigma)[0] * REAL(tauFixed)[0]);
          }
        } 
        if (R_isnancpp(Adder)|| !R_finite(Adder)) {
          Rf_error("CalcProb: ii=%d, and EndList=%d, Adder fail for Prior tauFixed[IDBeta[ii=%d]=%d]=%f, TP3: Total=%f\n",
            ii, EndList, ii, IDBeta[ii], REAL(tauFixed)[IDBeta[ii]], Total);
        }  
        Total+=Adder;       
      } else if (TypePrior == 4) {
      
          Adder = - log( REAL(tauFixed)[1] - REAL(tauFixed)[0]);
          if (R_isnancpp(Adder) || !R_finite(Adder)) {
            Rf_error("CalcProb: ii=%d, EndList = %d, tauFixed = %f, %f, but Adder = %f\n",
              ii, EndList, REAL(tauFixed)[1], REAL(tauFixed)[0]); R_FlushConsole();
          }
          Total+=Adder;
      }
      ii++;
  }
  if (sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) > 0 &&
    tauEndList != NULL && !Rf_isNull(tauEndList) && Rf_length(tauEndList) > 0) {
    int ot = 0;
    double HinvTauSq = 0.0;
    if (Verbose >= 4) {
      Rprintf("BayesSpikeGibbs.cpp:::CalcProb Starting Tau Modification[%d/%d]\n", 
        LenTau, Rf_length(sOnTau)); R_FlushConsole();
    }
    if (rPriorProbFixed == NULL || Rf_isNull(rPriorProbFixed->asSexp()) ||
      Rf_length(rPriorProbFixed->asSexp()) <= 0) {
      //Total += ii* log(REAL(rPriorProbFixed->asSexp())[ii]) - 
      // ((double)iFirstRandom-ii)*log(1.0 - REAL(rPriorProbFixed->asSexp())[ii]);
    }
    if (Verbose >= 5) {
      Rprintf(" Check Tau -- "); R_FlushConsole();
      for (int it = 0; it < LenTau; it++) {
        Tau[it] = Tau[it] + 0.0;
      }
      Rprintf(" PASS \n"); R_FlushConsole();
      Rprintf(" Check Beta -- "); R_FlushConsole();
      for (int iii = 0; iii <LenBeta; iii++) {
        IDBeta[iii] = IDBeta[iii] + 0;
        Beta[iii] = Beta[iii] + 0.0;
      }
    }
    ii = 0;
    for (int it = 0; it < LenTau; it++) {
      if (Verbose >= 5) {
        if (it > 0 && it % 10 == 0) { Rprintf("\n"); }
        Rprintf("it=%d-", it); R_FlushConsole(); 
      }
      while(ot < Rf_length(sOnTau) && ot < IDTau[it]) { 
        if (rPriorProbTau != NULL && !Rf_isNull(rPriorProbTau->asSexp()) && Rf_length(rPriorProbTau->asSexp()) >= Rf_length(sOnTau) &&
          Rf_length(rPriorProbTau->asSexp()) > ot) {
          if (REAL(rPriorProbTau->asSexp())[ot] > 0.0 && REAL(rPriorProbTau->asSexp())[ot] < 1.0) {
            Total += log(1.0 - REAL(rPriorProbTau->asSexp())[ot]);
          } else if (REAL(rPriorProbTau->asSexp())[ot] < 0.0) {
             double FAC1 = fabs(REAL(rPriorProbTau->asSexp())[ot]);
             Total += log(1.0 - (PiA2 * FAC1)/(PiA2*FAC1 + 1.0-PiA2));
          }
        }
        ot++;
      } 
      if (Verbose >= 5) {
        //Rprintf("a"); R_FlushConsole();
      }
      if (ot >= Rf_length(sOnTau)) {
        Rf_error("BayesSpikeGibbs.cpp:::CalcProb: why is ot=%d/%d so bad?\n", ot, Rf_length(sOnTau));
        R_FlushConsole();
      }
      if (it == 0) {
        while (ii < LenBeta && IDBeta[ii] < iFirstRandom) { ii++; }
        //Total += (ii * log(PiA1) + (iFirstRandom -ii) * log(1.0-PiA1);
      }
      if (Verbose >= 5) {
        //Rprintf("b"); R_FlushConsole();
      }
      if (ii >= LenBeta) {
        Rprintf("BayesSpikeGibbs.cpp:::CalcNewProb: oops, ii=%d/%d, but iFirstRandom=%d, IDBeta[ii-1=%d]=%d\n",
          ii, LenBeta, iFirstRandom, ii-1, IDBeta[ii-1]); 
      }
      if (Verbose >= 5) {
        //Rprintf("c"); R_FlushConsole();
      }
      if (ot <= 0) {
        //Total += (ii) * log(PiA1) + (iFirstRandom - ii) * log( 1.0-PiA1); 
        //  // Add probability for on and off!
        Total += (ANINT(tauEndList, 0) - iFirstRandom+1) * (
          -.5 * log(Tau[it]) - logSqrt2PI);
      } else {
        while (ii < LenBeta && IDBeta[ii] <= ANINT(tauEndList, ot-1)) { ii++; }
        Total += (ANINT(tauEndList,ot) - ANINT(tauEndList,ot-1)) * (
          -.5 * log(Tau[it]) - logSqrt2PI);
      }
      HinvTauSq = .5 / (Tau[it] * Tau[it]);
      while(ii < LenBeta && IDBeta[ii] <= ANINT(tauEndList, ot)) {
        Total -= HinvTauSq * Beta[ii] * Beta[ii];
        ii++;
      }
      if (Verbose >= 5) {
        //Rprintf("d"); R_FlushConsole();
      }
      if (Tau[it] > 0.0 && rPriorProbTau != NULL 
        && !Rf_isNull(rPriorProbTau->asSexp()) && 
        Rf_length(rPriorProbTau->asSexp()) > ot) { 
        if (REAL(rPriorProbTau->asSexp())[ot] > 0.0 && REAL(rPriorProbTau->asSexp())[ot] < 1.0) {
            Total += log(REAL(rPriorProbTau->asSexp())[ot]);
        } else if (REAL(rPriorProbTau->asSexp())[ot] < 0.0) {
             double FAC1 = fabs(REAL(rPriorProbTau->asSexp())[ot]);
             Total += log((PiA2 * FAC1)/(PiA2*FAC1 + 1.0-PiA2));
        }
      }
    } 
    if (rPriorProbTau != NULL && !Rf_isNull(rPriorProbTau->asSexp()) && Rf_length(rPriorProbTau->asSexp()) >= Rf_length(sOnTau)) {
    } else {
      Total += log(PiA2) * LenTau + log(1.0-PiA2) * (Rf_length(sOnTau) - LenTau);
    }
    if (!Rf_isNull(sOnPiA)) {
      if (!Rf_isNull(PiAPrior)) {
        if (Rf_length(PiAPrior) == 2) {
          Total += REAL(PiAPrior)[0] * log( PiA1 ) +
            REAL(PiAPrior)[1] * log(1.0 - PiA1 );
        } else {
          Total += REAL(PiAPrior)[2] * log( PiA2 ) +
            REAL(PiAPrior)[3] * log(1.0 - PiA2 );        
        }
      }
    }
    if (R_isnancpp(Total)) {
      Rprintf("Using CalcAlterProperty: We hit a nan on ");
      Rprintf("Total, ot = %d,  Tau=%f\n",
        ot, Tau[ot]);
      Rf_error("Error, NaN Cpp on CalcAlterProperty!\n"); R_FlushConsole();
    }
    if (Verbose >= 3) {
      Rprintf("CalcNewProb: Finished All ProbTau. \n"); R_FlushConsole();
    }
  }
  return(Total);

}
int CalcLengthNeedBetas(int OnLocationBuffer, int LengthLocationBuffer, bufferILocType *LocationBuffer, long int LengthThisBuffer, int *pOnFinal) {
  if (OnLocationBuffer < 0 || OnLocationBuffer >= LengthLocationBuffer) {
    Rf_error("BayesSpikeGibbs.cpp:::Error, CalcLengthNeedBetas, OnLocationBuffer is bad = %d but length = %d\n",
      OnLocationBuffer, LengthLocationBuffer);
  }
  int TotLengthSoFar = 0;
  bufferILocType MyLastOn = (bufferILocType) LocationBuffer[OnLocationBuffer];
  bufferILocType MyStart = (bufferILocType) LocationBuffer[OnLocationBuffer];
  for (int ii = OnLocationBuffer; ii < LengthLocationBuffer; ii++) {
    if (LocationBuffer[ii] - MyStart < LengthThisBuffer) {
       MyLastOn = LocationBuffer[ii];  TotLengthSoFar =  LocationBuffer[ii] - MyStart;
    } else if (LocationBuffer[ii] - MyStart == LengthThisBuffer) {
       MyLastOn = LocationBuffer[ii];  pOnFinal[0] = ii; 
       TotLengthSoFar = LengthThisBuffer;
       return(LengthThisBuffer);
    } else {
       MyLastOn = LocationBuffer[ii-1]; pOnFinal[0] = ii-1;
       TotLengthSoFar = MyLastOn - MyStart;
       return(TotLengthSoFar); 
    }
  }
  pOnFinal[0] = LengthLocationBuffer-1;
  if (1==0) {
    Rprintf("TotLengthSoFar was %d, MyLastOn=%d\n", TotLengthSoFar, MyLastOn);
  }
  return(LocationBuffer[pOnFinal[0]] - MyStart);
}
int CalcLengthTauDraws(int LpOnFinal, int OnTLocationBuffer, 
 int LengthTLocationBuffer, bufferILocType *TLocationBuffer, long int LengthThisBuffer, int *pOnFinalT, int Verbose) {
   if (OnTLocationBuffer < 0 || OnTLocationBuffer >= LengthTLocationBuffer) {
    Rf_error("BayesSpikeGibbs.cpp:::Error, CalcLengthTauDraws, OnTLocationBuffer is bad = %d but length = %d\n",
      OnTLocationBuffer, LengthTLocationBuffer);
  }
  if (Verbose >= 2) {
    Rprintf("CalcLengthTauDraws ----  we start");
    Rprintf("-- LpOnFinal = %d, OnTLocationBuffer=%d, LengthTLocationBuffer=%d, LengthThisBuffer=%d\n",
      LpOnFinal, OnTLocationBuffer, LengthTLocationBuffer, LengthThisBuffer);
    R_FlushConsole();
  }
  int TotLengthSoFar = 0;
  bufferILocType MyLastOn = (bufferILocType) TLocationBuffer[OnTLocationBuffer];
  bufferILocType MyStart = (bufferILocType) TLocationBuffer[OnTLocationBuffer];
  if (LpOnFinal < LengthTLocationBuffer) {
    pOnFinalT[0] =  LpOnFinal;
    return(TLocationBuffer[pOnFinalT[0]] - MyStart);
  } else {
    Rf_error("CalcLengthTauDraws: Something is not right, LpOnFinal=%d, but LengthTLocationBuffer=%d\n", LpOnFinal,
      LengthTLocationBuffer);
  }
  if (LengthTLocationBuffer <= 1) {
    Rprintf("CalcLengthTauDraws: Issue: LengthTLocationBuffer = %d there is no length!\n",
      LengthTLocationBuffer); R_FlushConsole();
  }
  if (LengthThisBuffer <= 0) {
    Rprintf("CalcLengthTau: ERROR ERROR LengthThisBuber is %d, not good!\n",
      LengthThisBuffer);
  }
  // Walk ii from current location to final location in T buffer.
  // If TLocationBuffer wants us to go over length, identify MyLastOn as previous location.
  for (int ii = OnTLocationBuffer; ii < LengthTLocationBuffer; ii++) {
    if (TLocationBuffer[ii] - MyStart < LengthThisBuffer) {
       MyLastOn = TLocationBuffer[ii];  TotLengthSoFar =  TLocationBuffer[ii] - MyStart;
    } else if (TLocationBuffer[ii] - MyStart == LengthThisBuffer) {
       MyLastOn = TLocationBuffer[ii];  pOnFinalT[0] = ii; 
       TotLengthSoFar = LengthThisBuffer;
       if (Verbose >= 3) {
         Rprintf("CalcLength: Exit on perfection: ii=%d/%d from %d start, MyLastOn=%d, MyStart=%d, LengthThisBuffer was %d\n",
           ii, LengthTLocationBuffer, OnTLocationBuffer, MyLastOn, MyStart, LengthThisBuffer); R_FlushConsole();
       }
       return(LengthThisBuffer);
    } else {
       MyLastOn = TLocationBuffer[ii-1]; pOnFinalT[0] = ii-1;
       TotLengthSoFar = MyLastOn - MyStart;
       if (Verbose >= 3) {
         Rprintf("CalcLength: Exit on overlap: ii=%d/%d from %d start, MyLastOn=%d, MyStart=%d, LengthThisBuffer was %d\n",
           ii, LengthTLocationBuffer, OnTLocationBuffer, MyLastOn, MyStart, LengthThisBuffer); R_FlushConsole();
       }
       return(TotLengthSoFar); 
    }
  }
  pOnFinalT[0] = LengthTLocationBuffer-1;
  if (1==0) {
    Rprintf("TotLengthSoFar was %d, MyLastOn=%d\n", TotLengthSoFar, MyLastOn);
  }
  if (Verbose >= 3) {
    Rprintf("-- CalcTau: got to end: pOnFinalT[0] = %d, returning TLocation[%d]-%d=%d\n",
      pOnFinalT[0], pOnFinalT[0], TLocationBuffer[pOnFinalT[0]], MyStart,
      TLocationBuffer[pOnFinalT[0]] - MyStart); R_FlushConsole();
  }
  return(TLocationBuffer[pOnFinalT[0]] - MyStart);
}
int BayesSpikeCL::DeriveAlterProbabilityStartCheck(int *pLenFixed, int *pNumGroups) {
  if (RsAlterWeightFile == NULL || AlterWeightBuffer == NULL) {
    Rf_error("DeriveAlterProbability: NoAlterWeightFile!");
  }
  if (Verbose >= 1) {
    Rprintf("DeriveAlterProbability, Checking AlterWeightBuffer "); R_FlushConsole();
    for (int iti = 0; iti < LengthAlterWeightBuffer; iti++) {
      AlterWeightBuffer[iti]+=1.0;
    }
    Rprintf(" --- Pass \n"); R_FlushConsole();
  }
  if (RsCodaIFile == NULL) {
    Rf_error("DeriveAlterProbability: No CodaIFile!\n");
  }
  if (RsCodaJFile == NULL) {
    Rf_error("DeriveAlterProbability: No CodaDFile!\n");
  }
  pLenFixed[0] = p;  pNumGroups[0] = 0;
  if (sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) > 0 &&
    tauEndList != NULL && !Rf_isNull(tauEndList) && Rf_length(tauEndList) > 0) {
    pLenFixed[0]  = iFirstRandom;
    pNumGroups[0] = Rf_length(sOnTau);
    if (RsCodaiTFile == NULL) {
      Rf_error("DeriveAlterProbability: No CodaProbFile!\n");
    }
    if (RsCodadTFile == NULL) {
      Rf_error("DeriveAlterProbability: No CodadTFile!\n");
    }
  }
  if (RsPiACodaFile == NULL) {
    Rf_error("DeriveAlterProbability: No PiACodaFile!\n");
  }
  if (RsSigCodaFile == NULL) {
    Rf_error("DeriveAlterProbability: No SigCodaFile!\n");
  }
  if (RsCodaProbFile == NULL) {
    Rf_error("DeriveAlterProbability: No CodaProbFile!\n");
  }
  if (RsCodaILocFile == NULL) {
    Rf_error("DeriveAlterProbability: No CodaProbFile!\n");
  }
  if ((RsCodaiTLocFile == NULL  || Rf_isNull(RsCodaiTLocFile->asSexp()) || 
       !Rf_isString(RsCodaiTLocFile->asSexp())) && 
       (sOnTau != NULL && !Rf_isNull(sOnTau) 
         && Rf_length(sOnTau) > 0)) {
    Rf_error("DeriveAlterProbability: No RsCodaiTLocFile!\n");
  }
  if (!Rf_isString(RsAlterWeightFile->asSexp())) {
    Rf_error("DeriveAlterProbability: AlterWeightFile is not a String!\n");
  }
  if (!Rf_isString(RsCodaIFile->asSexp())) {
    Rprintf("ERROR DeriveAlterProbability: ");
    Rprintf("Unfortunately, sCodaIFile is not A String!\n"); R_FlushConsole();
    Rf_error("sCodaIFile Fail!\n");
  }
  if (!Rf_isString(RsCodaJFile->asSexp())) {
    Rf_error("ERROR: DeriveAlterProbability: Unfortunately, ");
    Rprintf(" sCodaDFile is not A String!\n"); R_FlushConsole();
    Rf_error("ERROR: sCodaDFile Fail.\n");
  }
  if (PiACodaBuffer == NULL) {
    Rf_error("DeriveAlterProbability: Setup PiACodaBuffer First!\n");
  }
  if (SigCodaBuffer == NULL) {
    Rf_error("DeriveAlterProbability: Setup SigCodaBuffer First!\n");
  }
  if (Verbose > 0) {
    Rprintf("DeriveAlterProbability: DeriveAlterProbabilityStartCheck() pass: About to read from IFile %s, DFile %s \n",
      CHAR(STRING_ELT(RsCodaIFile->asSexp(), 0)),
      CHAR(STRING_ELT(RsCodaJFile->asSexp(), 0)));
  }
  return(1);
}

void PrintLongVector(long int *VVector, int Length) {
  int AB = 10;
  if (Length < AB) { AB = Length; }
  if (AB <= 0) {
    Rprintf("()"); R_FlushConsole(); return;
  }
  Rprintf("(");
  for (int iti = 0; iti < AB; iti++) {
    if (iti == Length-1) {
      Rprintf("%d)", (int) VVector[iti]);
    } else if (iti == AB-1) {
      Rprintf("%d,..., %d)", (int) VVector[iti], (int) VVector[Length-1]);
    } else {
      Rprintf("%d, ", (int) VVector[iti]);
    }
    R_FlushConsole();
  }
  return;
}
/////////////////////////////////////////////////////////////////////////////
// DeriveAlterProbability(SEXP sType, SEXP sStartIter, SEXP sEndIter)
//
//  If the RsProbCoda has the Probability storing the likelihood based
//   upon fixed weights, this code attempts to re-read in the draws and
//   then calculate the 
//
int BayesSpikeCL::DeriveAlterProbability(int AInput) {
  int Verbose = this->Verbose;
  //Verbose = 4;
  if (Verbose >= 1) { Rprintf("We're Running DeriveAlterProbability!\n");  R_FlushConsole(); }
  int LenFixed = p; int NumGroups = 0;
  DeriveAlterProbabilityStartCheck(&LenFixed, &NumGroups);
  int weClosed = 0;

  if (Verbose >= 6) {  Rprintf("DeriveAlterProbability: LenFixed = %d\n", LenFixed); R_FlushConsole();
  }
 
  // IFile is integer Beta identities
  // DFile is double Beta values
  // iTFile is integer tau on identities
  // dTFile is double tau Values
  // ILoc and iTLoc suggest locations for IFile, iTFile
  // SigFile, PiAFile record PiA and ProbFile records jumps
  FILE *fCodaIFile=NULL, *fCodaDFile=NULL, *fCodaiTFile=NULL, 
    *fCodadTFile=NULL,
    *fCodaProbFile=NULL, *fCodaILocFile=NULL, *fCodaiTLocFile=NULL, 
    *fCodaSigFile=NULL, *fCodaPiAFile=NULL;
        
  int *OnOrderedActive = NULL; int aOnKappaMem = this->OnKappaMem;
  double PiA1=.5, PiA2 = .5;  double Sigma;
  int *OnTauIndices=NULL;  double *OnTauVals = NULL;
  double *OnProbBeta = NULL; double *EY = NULL; 
  
  int BuffSize = this->LengthAlterWeightBuffer; int MaxBuffSize = BuffSize;

  int bufferiLocLength = 0;
  int *bufferiT = NULL; double *bufferdT = NULL;
  bufferILocType *bufferiTLoc = NULL;
  int *bufferI = NULL;  double *bufferD = NULL;
  double *bufferProb = NULL;  long int *bufferILoc = NULL;
  double *bufferSig = NULL; double *bufferPiA = NULL;
  
  
  if (Verbose >= 2) { Rprintf("DeriveAlterProbability: Allocating several memory blocks!\n"); R_FlushConsole(); }
  //bufferILoc = (long int*) Calloc(BuffSize+1, bufferILocType);
  
  if (this->OnKappaMem <= 2) { aOnKappaMem = 10; }
  OnOrderedActive = (int*) Calloc(aOnKappaMem, int);
  if (OnOrderedActive == NULL) { Rprintf("OnOrderedActiveFail!\n"); LaunchEnd(-1);
    Rprintf("ERROR: DeriveAlterProbability: Issue, could not allocate ");
    Rprintf("in OnOrderedActive!\n");   R_FlushConsole(); Rf_error("ERROR: DeriveAlterProbability\n");
  }

  PropBeta =  (double*) Calloc(aOnKappaMem, double);
  if (PropBeta == NULL) {
    Rprintf("OnPropBeta Fail!\n"); LaunchEnd(-1);
    Rf_error("DeriveAlterProbability: Issue, could not allocate in OnPropBeta");
  }
  EY = (double*) Calloc(n+1, double);
  if (EY == NULL) {
    Rprintf("EY Alloc Fail!\n");
    LaunchEnd(-1);
    Rf_error("DeriveAlterProbability: Issue, could not allocate EY.\n");
  }
  
  int OnSigBuff=0, TOnSigBuff = 0, OnPiABuff = 0, TOnPiABuff = 0;
  int OnProbBuff=0; int TOnProbBuff = 0;
  int OnBetaBuff = 0; long TOnBetaBuff = 0;
  int OnTauBuff = 0; long int TOnTauBuff = 0;
  long int FirstOnILoc = 0;  //long int FirstOniTLoc = 0;
  int ReadLastLocBeta = 0; //int ReadLastOnB = 0;
  bufferSig = (double*) Calloc(MaxBuffSize, double);
  bufferPiA = (double*) Calloc(Rf_length(sOnPiA) * MaxBuffSize, double);

  
  int BLOCKSIZE = MaxBuffSize * aOnKappaMem;
  if (BLOCKSIZE > MaxCODABeta) {  BLOCKSIZE = MaxCODABeta; }
  int LenTBuffer = BuffSize *NumGroups;

  if (Verbose >= 2) {
    Rprintf("DeriveAlterProbability: Deriving bufferI at BLOCKSIZE %d\n",
      BLOCKSIZE); R_FlushConsole();
  }
  bufferI = (int*) Calloc(BLOCKSIZE, int);
  bufferD = (double*) Calloc(BLOCKSIZE, double);  
  bufferProb = (double*) Calloc(MaxBuffSize, double);
  bufferILoc = (bufferILocType*) Calloc(MaxBuffSize, bufferILocType);
  if (bufferProb == NULL || bufferILoc == NULL) {
    Rprintf("bufferProb, bufferILoc Fail!\n");
    LaunchEnd(-1);
    Rf_error("DeriveAlterProbability: Issue could not allocate bufferILoc.\n");
  }
  if (sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) > 0 &&
    tauEndList != NULL && !Rf_isNull(tauEndList) && Rf_length(tauEndList) > 0) {
    if (NumGroups < aOnKappaMem) {
      LenTBuffer = MaxBuffSize * NumGroups;
    }
    if (Verbose >= 2) {
      Rprintf("DeriveAlterProbability: Allocating TauIndices and Vals\n");
    }
    OnTauIndices = (int*) Calloc(NumGroups, int);
    OnTauVals = (double*) Calloc(NumGroups, double);
    
    bufferiT = (int*) Calloc( LenTBuffer, int);
    bufferdT = (double*) Calloc( LenTBuffer, double); 
    if (OnTauIndices == NULL || OnTauVals == NULL ||
      bufferiT == NULL || bufferdT == NULL) {
      Rprintf("DeriveAlterProbability: Tau Indices, bufferiT fail!\n");
      LaunchEnd(-1);
      Rf_error("DeriveAlterProbability: Could not allocate %d Groups of TauIndices\n", NumGroups);
    }  
    bufferiTLoc = (bufferILocType*) Calloc(MaxBuffSize, bufferILocType);
    if (Verbose >= 4) {
      Rprintf("DeriveAlterProbability: done deriving bufferiTLoc\n");
      R_FlushConsole();
    }
  }

  int fd1=-1, fd2=-1,fd3=-1,fd4=-1; //,fd5,fd6, fd7, fd8,fd9;
  int fdILoc=-1, fdProb=-1, fdiTLoc=-1;
  struct stat stbuf; 
  if (Verbose >= 4) {
    Rprintf("DerivealterProbability: About to open IJ files for Length Read!\n");
    R_FlushConsole();
  }
  fd1 = open( CHAR(STRING_ELT(sCodaIFile, 0)), O_RDONLY);
  if (fd1 == -1) {
    Rprintf("DeriveAlterProbability: no open sCodaIFile fail!\n");
    LaunchEnd(-1);
    Rf_error("Setup GiveCodaSubset: We can't open sCodaIFile"); 
  } 
  TotalOpenedFiles++;
  fd2 = open( CHAR(STRING_ELT(RsCodaJFile->asSexp(), 0)), O_RDONLY);
  if (fd2 == -1) {
    Rprintf("DeriveAlterProbability: no open sCodaDFile fail!\n");
    LaunchEnd(-1);
    AKILLDERIVEFILE(-1);
    Rf_error("Setup DeriveAlterProbability: We can't open sCodaJFile"); 
  } 
  TotalOpenedFiles++;
  if (NumGroups > 0 && sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) > 0 &&
    tauEndList != NULL && !Rf_isNull(tauEndList) && Rf_length(tauEndList) > 0) {
    if (RsCodaiTFile!=NULL) {
      fd3 = open( CHAR(STRING_ELT(RsCodaiTFile->asSexp(),0)), O_RDONLY);
    }
    if (fd3 == -1) {
      Rprintf("DeriveAlterProbability: Well, fd3 was not opened\n");
      AKILLDERIVEFILE(-1);
      Rf_error("DeriveAlterProbability: fd3 open error ! \n");
    }
    if (RsCodadTFile!=NULL) {
      fd4 = open( CHAR(STRING_ELT(RsCodaiTFile->asSexp(),0)), O_RDONLY);
    }
    if (fd4 == -1) {
      Rprintf("DeriveAlterProbability: Well, fd4 was not opened\n");
      AKILLDERIVEFILE(-1);
      Rf_error("DeriveAlterProbability: fd4 open error ! \n");
    }
  }


  
  fCodaIFile = fdopen( fd1, "rb");
  if (fCodaIFile != NULL) { TotalOpenedFiles++; }
  fCodaDFile = fdopen( fd2, "rb");
  if (fCodaDFile != NULL) { TotalOpenedFiles++; }
  fCodaiTFile = NULL;  fCodadTFile = NULL;
  if (sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) > 0 &&
    tauEndList != NULL && !Rf_isNull(tauEndList) && Rf_length(tauEndList) > 0)  {
    if (fd3 != -1) {
      fCodaiTFile = fdopen( fd3, "rb");
    }
    if (fd4 != -1) {
      fCodadTFile = fdopen( fd4, "rb");
    }
  }
  if (fCodaiTFile != NULL) { TotalOpenedFiles++; }
  if (fCodadTFile != NULL) { TotalOpenedFiles++; }
  
  if (fstat(fd1, &stbuf) == -1) {
    AKILLDERIVEFILE(-1);
    Rf_error("GiveCodaSubSet sCodaIFile: could not use fstat");
  }
  long lSizeI = stbuf.st_size / sizeof(int);
  if (fstat(fd2, &stbuf) == -1) {
    AKILLDERIVEFILE(-1);
    Rf_error("GiveCodaSubSet sCodaDFile: could not use fstat");
  }

  long lSizeD = stbuf.st_size / sizeof(double);
  if ((lSizeI  ) < 
    (lSizeD )) {
    Rprintf("DeriveAlterProbability:  Length Check, Error After checking length files %s, and %s\n",
      CHAR(STRING_ELT(RsCodaIFile->asSexp(), 0)), CHAR(STRING_ELT(RsCodaJFile->asSexp(), 0))  ); R_FlushConsole();
    Rprintf("lSizeI = %d, lSizeD = %d, sizeof(int) = %d, sizeof(char) = %d, sizeof(double) = %d\n",
      lSizeI, lSizeD, sizeof(int), sizeof(char), sizeof(double));
    R_FlushConsole();
    AKILLDERIVEFILE(-1);
    Rf_error("GiveCodaSubSet: Sorry about the Error");
  }  
  if (Verbose >= 2) {
    Rprintf("DeriveAlterProbability lSizeD = %d = lSizeI = %d!\n", lSizeD, lSizeI); R_FlushConsole();
  }
  size_t result = 0;


  if (Verbose >= 2) {
    Rprintf("Looking at Prob and Loc!\n"); R_FlushConsole();
    Rprintf("Note that fdILoc is %s. \n", CHAR(STRING_ELT(RsCodaILocFile->asSexp(), 0)));
    R_FlushConsole();
  }
  fdILoc = open( CHAR(STRING_ELT(RsCodaILocFile->asSexp(), 0)), O_RDONLY);
  if (fdILoc != -1) { TotalOpenedFiles++; }
  if (fdILoc == -1) {
    AKILLDERIVEFILE(-1);
    Rprintf("DeriveAlterProbability: no open sCodaLocFile fail!\n"); LaunchEnd(-1);
    Rf_error("Setup DeriveAlterProbability: We can't open sCodaLocFile"); 
  } 
  fdProb = open( CHAR(STRING_ELT(RsCodaProbFile->asSexp(), 0)), O_RDONLY);
  if (fdProb != -1) { TotalOpenedFiles++; }
  if (fdProb == -1) {
    AKILLDERIVEFILE(-1);
    Rprintf("DeriveAlterProbability: no open sCodaProbFile fail!\n"); 
    LaunchEnd(-1);
    Rf_error("Setup GiveCodaSubset: We can't open sCodaProbFile"); 
  } 

  if (Verbose >= 3) {
    Rprintf("DeriveAlterProbability, Open fdILoc, fdProb. \n"); R_FlushConsole();
  }
  fCodaILocFile = fdopen( fdILoc, "rb");
  fCodaProbFile = fdopen( fdProb, "rb");
  if (fCodaILocFile != NULL) { TotalOpenedFiles++; }
  if (fCodaProbFile != NULL) { TotalOpenedFiles++; }

  if (fstat(fdILoc, &stbuf) == -1) {
    AKILLDERIVEFILE(-1);
    Rf_error("GiveCodaSubSet sCodaLocFile: could not use fstat");
  }
  long lSizeLoc = stbuf.st_size / sizeof(bufferILocType);
  if (fstat(fdProb, &stbuf) == -1) {
    Rf_error("GiveCodaSubSet sCodaProbFile: could not use fstat");
  }
  long lSizeProb = stbuf.st_size / sizeof(double);
  if (lSizeLoc != lSizeProb) {
    Rprintf("DeriveAlterProbability: lSizeLoc=%d, lSizeProb = %d! Uh Oh NOT GOOD\n", 
      lSizeLoc, lSizeProb);
    R_FlushConsole();
  }
  if (Verbose >= 4) {
    Rprintf("DeriveAlterProbability: fdLoc = %d, fdProb = %d, lSizeLoc=%d, lSizeProb.\n", 
      fdILoc, fdProb, lSizeLoc, lSizeProb);
    R_FlushConsole();
  }
  
  // obtain file sizes:
  //  It is important at the start to know exactly the length of all of the buffer files
  //   and to know they've been recorded correctly according to how lsizeLoc has suggested.
  
  if ((lSizeLoc) <  (lSizeProb)) {
    Rprintf("DeriveAlterProbability:  Length Check, Error After checking length files %s, and %s.\n",
      CHAR(STRING_ELT(RsCodaILocFile->asSexp(), 0)), CHAR(STRING_ELT(RsCodaProbFile->asSexp(), 0))  ); R_FlushConsole();
    Rprintf("lSizeLoc = %d, lSizeProb = %d, sizeof(int) = %d, sizeof(char) = %d, sizeof(double) = %d.\n",
      lSizeLoc, lSizeProb, sizeof(long int), sizeof(char), sizeof(double));
    R_FlushConsole();
    AKILLDERIVEFILE(-1);
    Rf_error("GiveCodaSubSet: Sorry about the Error");
  }

  if (Verbose >= 2) {
    Rprintf("DeriveAlterProbability: we got lSizeI=%d, lSizeD=%d, lSizeLoc=%d, lSizeProb=%d.\n",
      lSizeI, lSizeD, lSizeLoc, lSizeProb); R_FlushConsole();
  }
  
  if (Verbose >= 3) { Rprintf("About to look at Tau Sizes. \n"); }
  long lSizeiT = 0, lSizedT = 0, lSizeTLoc = 0;  
  if (sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) > 0 &&
    tauEndList != NULL && !Rf_isNull(tauEndList) && Rf_length(tauEndList) > 0) {
    if (Verbose >= 3) {
      Rprintf("DeriveAlterProbability(): fstat: using to look at sCodaiTFile, sCodaiDFile. \n"); R_FlushConsole();
    }
    if (fstat(fd3, &stbuf) == -1) { Rf_error("DeriveAlterProbability(): GiveCodaSubSet sCodaiTFile: could not use fstat"); }
    lSizeiT = stbuf.st_size / sizeof(int);
    if (fstat(fd4, &stbuf) == -1) { Rf_error("DeriveAlterProbability(): GiveCodaSubSet sCodaiDFile: could not use fstat"); }
    lSizedT = stbuf.st_size / sizeof(double);
    if (Verbose >= 3) {
      printf("DeriveAlterProbability(): about to look nto fCodaiT, fCodadTFile. \n"); R_FlushConsole();
    }
    if (fCodaiTFile != NULL) { fclose(fCodaiTFile); fCodaiTFile= NULL; TotalClosedFiles++; }
    if (fCodadTFile != NULL) { fclose(fCodadTFile); fCodadTFile= NULL; TotalClosedFiles++; }
    close(fd3); fd3 = -1; close(fd4); fd4 = -1; TotalClosedFiles+=2;
    if (Verbose >= 2) { Rprintf("Looking at Prob and Loc!\n"); R_FlushConsole(); }
    fdiTLoc = open( CHAR(STRING_ELT(RsCodaiTLocFile->asSexp(), 0)), O_RDONLY);
    if (fdiTLoc != -1) { TotalOpenedFiles++; }
    fCodaiTLocFile = fdopen( fdiTLoc, "rb");
    if (fCodaiTLocFile != NULL) { TotalOpenedFiles++; }
    if (fstat(fdiTLoc, &stbuf) == -1) {
      Rf_error("GiveCodaSubSet sCodaiTLocFile: could not use fstat");
    }
    lSizeTLoc = stbuf.st_size / sizeof(bufferILocType);
    if (Verbose >= 3) {
      Rprintf("DeriveAlterProb: lSizeiT=%d, lSizedT=%d, lSizeTLoc=%d.\n",
        lSizeiT, lSizedT, lSizeTLoc); Rprintf("Good to go? \n"); R_FlushConsole();
    }
    if ((lSizeiT) < (lSizedT)) {
      Rprintf("DeriveAlterProb:  Error After checking length files %s, and %s\n",
        CHAR(STRING_ELT(sCodaiTFile, 0)), CHAR(STRING_ELT(sCodadTFile, 0))  ); R_FlushConsole();
      Rprintf("lSizeI = %d, lSizeD = %d, sizeof(int) = %d, sizeof(char) = %d, sizeof(double) = %d\n",
        lSizeiT, lSizedT, sizeof(int), sizeof(char), sizeof(double));
      if (fCodaiTLocFile != NULL) { fclose(fCodaiTLocFile); fCodaiTLocFile = NULL; TotalClosedFiles++; }
      if (fdiTLoc != -1) { close(fdiTLoc);  fdiTLoc = -1;  TotalClosedFiles++;  }
      Rf_error("DeriveAlterProb: Sorry about the Error");
    }
    if (Verbose >= 5) {
      Rprintf("DeriveAlterProb, read lSizeiT = %d, = lSizedT = %d!\n", lSizeiT, lSizedT); R_FlushConsole();
    }
    if (Verbose >= 5) {
      Rprintf("DeriveAlterProb, also lSizeTLoc = %d!\n", lSizeTLoc);
      R_FlushConsole();
    }
  }
  
  //size_t result2 = 0;
  // obtain file size:
  if (fd1 != -1 && fCodaIFile == NULL) { close(fd1); fd1=-1; TotalClosedFiles++;}
  if (fCodaIFile != NULL) { fclose(fCodaIFile); fCodaIFile = NULL; TotalClosedFiles++; }
  if (fd1 != -1) { fd1=-1; TotalClosedFiles++;}
  
  if (fd2 != -1 && fCodaDFile == NULL) { close(fd2); fd2=-1; TotalClosedFiles++;} 
  if (fCodaDFile != NULL) { fclose(fCodaDFile); fCodaDFile = NULL; TotalClosedFiles++; }
  if (fd2 != -1) { fd2=-1; TotalClosedFiles++;} 

  if (fd3 != -1 && fCodaiTFile == NULL) { close(fd3); fd3=-1; TotalClosedFiles++;}    
  if (fCodaiTFile != NULL) { fclose(fCodaiTFile); fCodaiTFile = NULL; TotalClosedFiles++; }
  if (fd3 != -1) { fd3=-1; TotalClosedFiles++;} 
  
  if (fd4 != -1 && fCodadTFile == NULL) { close(fd4); fd4=-1; TotalClosedFiles++;} 
  if (fCodadTFile != NULL) { fclose(fCodadTFile); fCodadTFile = NULL; TotalClosedFiles++; }
  if (fd4 != -1) { fd4=-1; TotalClosedFiles++;} 
  
  if (fdiTLoc != -1 && fCodaiTLocFile == NULL) { close(fdiTLoc);  fdiTLoc = -1;  TotalClosedFiles++;  }
  if (fCodaiTLocFile != NULL) { fclose(fCodaiTLocFile); fCodaiTLocFile = NULL; TotalClosedFiles++; }
  if (fdiTLoc != -1) {fdiTLoc = -1;  TotalClosedFiles++; }
   
  if (fdILoc != -1 && fCodaILocFile == NULL) { close(fdILoc); fdILoc=-1; TotalClosedFiles++;} 
  if (fCodaILocFile != NULL) { fclose(fCodaILocFile); fCodaILocFile = NULL; TotalClosedFiles++; } 
  if (fdILoc != -1) { fdILoc = -1;   TotalClosedFiles++; }
  
  if (fdProb != -1  && fCodaProbFile == NULL) { close(fdProb);  fdProb = -1;  TotalClosedFiles++;  }
  if (fCodaProbFile != NULL) { fclose(fCodaProbFile); fCodaProbFile = NULL; TotalClosedFiles++; }
  if (fdProb != -1) { fdProb = -1;  TotalClosedFiles++; }

  if (Verbose >= 4) {
    Rprintf("DeriveAlterProbability: AFter fCodaProbFile closed TotalClosedFiles=%d. \n",
      TotalClosedFiles); R_FlushConsole();
  }

  if (Verbose >= 3) {
    Rprintf("----------------------------------------------------------------------\n");
    Rprintf("--- DeriveAlterProbability(): we have gotten sizes. \n");
    Rprintf("--- lSizeLoc = %d, lSizeProb=%d, lSizeiT, lSizedT\n",
      lSizeLoc, lSizeProb, lSizeiT, lSizedT); R_FlushConsole();
    Rprintf("--- Closed files about to open them again. \n");
    Rprintf("----------------------------------------------------------------------\n");
    R_FlushConsole();
  }
            
  fCodaIFile = fopen( CHAR(STRING_ELT(RsCodaIFile->asSexp(), 0)), "rb");
  fCodaDFile = fopen( CHAR(STRING_ELT(RsCodaJFile->asSexp(), 0)), "rb");
  fCodaProbFile = fopen( CHAR(STRING_ELT(RsCodaProbFile->asSexp(), 0)), "rb");
  fCodaSigFile = fopen( CHAR(STRING_ELT(RsSigCodaFile->asSexp(), 0)), "rb");
  fCodaPiAFile = fopen( CHAR(STRING_ELT(RsPiACodaFile->asSexp(), 0)), "rb");   
  fCodaILocFile = fopen( CHAR(STRING_ELT(RsCodaILocFile->asSexp(), 0)), "rb");
  if (fCodaIFile != NULL) { TotalOpenedFiles++; }
  if (fCodaDFile != NULL) { TotalOpenedFiles++; }
  if (fCodaProbFile != NULL) { TotalOpenedFiles++; }
  if (fCodaSigFile != NULL) { TotalOpenedFiles++; }
  if (fCodaPiAFile != NULL) { TotalOpenedFiles++; }
  if (fCodaILocFile != NULL) { TotalOpenedFiles++; }
  if (sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) > 0) {    
    fCodaiTLocFile = fopen( CHAR(STRING_ELT(RsCodaiTLocFile->asSexp(), 0)), "rb");
    fCodaiTFile = fopen( CHAR(STRING_ELT(RsCodaiTFile->asSexp(), 0)), "rb");
    fCodadTFile = fopen( CHAR(STRING_ELT(RsCodadTFile->asSexp(), 0)), "rb");  
    if (fCodaiTLocFile != NULL) { TotalOpenedFiles++; }
    if (fCodaiTFile != NULL) { TotalOpenedFiles++; }
    if (fCodadTFile != NULL) { TotalOpenedFiles++; }
  }
  FILE *fAlterWeightFile = NULL; 
  
  if (Verbose >= 4) {
    Rprintf("fCodaIFile got open along with other files to TotalOpenedFiles=%d!\n", TotalOpenedFiles); R_FlushConsole();
  }

  if (fCodaIFile==NULL) {
    Rf_error("DeiveAlterWeight: sCodaIFile failed to open");
  }    
  if (fCodaDFile==NULL) {
    Rf_error("DeriveAlterWeight: sCodaDFile failed to open");
  } 
    

  if (sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) > 0 &&
    lSizeLoc != lSizeTLoc) {
    Rprintf("DeriveAlterWeight: Issue, lSizeLoc = %d, but lSizeTLoc = %d\n",
      lSizeLoc, lSizeTLoc); R_FlushConsole();
    Rprintf("-- Heavy chance this will lead to an issue. \n"); R_FlushConsole();
  }

  if (Verbose > 2) {
    Rprintf("DeriveAlterWeight: Goal is to Generate Alternative Weights for model draws.\n");
    R_FlushConsole();
  }
  /*     
    SEXP ReturnMat;
    Rf_protect(ReturnMat = Rf_allocMatrix(REALSXP, 
      EndIter - StartIter+1, Rf_length(SubSetCoords)));
    double ZeroD = 0.0;  int One = 1;
    int RLenR = Rf_length(ReturnMat);
    F77_CALL(dscal)(&RLenR, &ZeroD, REAL(ReturnMat), &One); //Zero Out matrix!
  */
  
  if (lSizeI < BLOCKSIZE) { BLOCKSIZE = lSizeI; }  
  
  if (Verbose >= 2) {
    Rprintf("DeriveAlterWeight: Allocating Buffers. \n"); R_FlushConsole();
  }
  // allocate memory to contain the whole file:

  if (bufferI == NULL) {
    Rf_error("GiveCodaSubSet: Cannot allocate bufferI");
  }  
  if (bufferD == NULL) {
    Rf_error("GiveCodaSubSet: Cannot allocate bufferD");
  } 
  //
  //  TOnB will be total Beta read and processed.  OnB will be location in 
  //    the Buffer where the Beta is located.  
  int ReadBlocks = 0;
  int OnB = 0;  int TOnB = 0;  int SOnB = 0;
  int OnT = 0;  int TOnT = 0;  int AllNeedB = 0;
  int LenBeta, LenTau = 0;  int WantLocBuffer = 0;  int GetLocBuffer = 0;
  int GetProbBuffer = 0;
  int ResidDiff = 0;  // long int LastLocationInBetaBuffer = 0;  
  //int BottomTOnT = 0;
  int jt = 0;
  double NewProb, OldProb;
  FirstOnILoc=0;  //FirstOniTLoc=0;
  int WantBetaBuffer = 0; int GetBetaBuffer = 0; int FinalLocationInLoc = 0; 
  int WantTauBuffer= 0;  int GetTauBuffer = 0; int FinalLocationInTauLoc = 0; 
  
  if (fCodaPiAFile != NULL) { 
    if (Verbose >= 4) {
      Rprintf("                   About to read from fCodaPiAFile!\n");
    }
    result = fread(bufferPiA, sizeof(double), 1, fCodaPiAFile);
    if (Verbose >= 4) {
      Rprintf("                   Read 1 from bufferPiA, got result = %d\n",
        result); R_FlushConsole();
    }
    if (fabs(bufferPiA[0] - 2.0) <= .001) {
      if (Rf_length(sOnPiA) != 2.0) {
        Rprintf("DeriveAlterWeight: First entry is %f, but length OnPiA = %d!\n",
          bufferPiA[0], Rf_length(sOnPiA)); R_FlushConsole();
      } else {
        if (Verbose >= 3) {
          Rprintf("DeriveAlterWeight: First entry in PiABuffer = %f, and length is 2!\n",
            bufferPiA[0]);
        }
      }
    } else if (fabs(bufferPiA[0] - 1.0) <= .001) {
      if (Rf_length(sOnPiA) != 1.0) {
        Rprintf("DeriveAlterWeight: First entry is %f, but length OnPiA = %d!\n",
          bufferPiA[0], Rf_length(sOnPiA)); R_FlushConsole();
      } else {
        if (Verbose >= 3) {
          Rprintf("DeriveAlterWeight: First entry in PiABuffer = %f, and length is 1!\n",
            bufferPiA[0]); R_FlushConsole();
        }
      }    
    } else {
      Rprintf("DeriveAlterWeight: First entry in PiABuffer is wrong is %f\n",
        bufferPiA[0]); R_FlushConsole(); 
    }
    OnPiABuff = 0;
  }

  if (Verbose > 1) {
    Rprintf("------------------------------------------------------------------\n");
    Rprintf("DeriveAlterWeight: About to read first element from ");
    Rprintf("bufferILoc, bufferiTLoc. \n"); R_FlushConsole();
  }
  //fread(bufferILoc, sizeof(bufferILocType), 1, fCodaILocFile);

  if (bufferILoc[0] != 0) {
    Rprintf("DeriveAlterWeight: Weird, first read bufferILoc[0] = %d!\n", bufferILoc[0]);
    R_FlushConsole();
  }
  if (bufferiTLoc != NULL && fCodaiTLocFile != NULL) {
    //fread(bufferiTLoc, sizeof(long int),1, fCodaiTLocFile); 
    if (bufferiTLoc[0] != 0) {
      Rprintf("DeriveAlterWeight: Weird, first read bufferiTLoc[0] = %d.\n", bufferiTLoc[0]);
      R_FlushConsole();
    }
  }  
  
  // Note ReDrawFunction, defined above will take a buffer read up to location
  //  OnLoc, copy info (OnLoc):(Length-1) to the front, and then read
  //  the aditional (Length-OnLoc)
  if (Verbose >= 6) {
    Rprintf("DeriveAlterWeight: Starting with lSizeLoc = %d, lSized = %d!\n",
      lSizeLoc, lSizeD); R_FlushConsole();
    Rprintf("*****************************************************************\n");
    Rprintf("  Now Testing bufferILoc  \n"); R_FlushConsole();
    for (int iti = 0; iti < MaxBuffSize; iti++) {
      bufferILoc[iti] += (bufferILocType) 1;
    }
    Rprintf("  BufferILoc Pass now testing fCoda");
    OnB = MaxBuffSize;
    if (OnB > 0 && OnB < MaxBuffSize )  {                       
      for (jt = 0; (int) jt < (int) (MaxBuffSize - OnB); jt++) {  
        bufferILoc[jt] = (bufferILocType) bufferILoc[jt+OnB];             }                    
    }                                                              
    int LengthTryAdd = 100; if (MaxBuffSize < LengthTryAdd) {
      LengthTryAdd = MaxBuffSize;
    }
    if (LengthTryAdd > lSizeLoc) { LengthTryAdd = lSizeLoc; }
    result = fread(bufferILoc + (MaxBuffSize-OnB), sizeof(long int), LengthTryAdd, fCodaILocFile); 
    Rprintf("  BufferLoc read result = %d for LengthTryAdd = %d\n"); R_FlushConsole();
    if (result >= 0) {
      Rprintf("  Here It Is: "); R_FlushConsole(); PrintLongVector(bufferILoc, result); R_FlushConsole();
      Rprintf("\n");
    }
    Rprintf("  Is that good? \n"); R_FlushConsole();
    fclose(fCodaILocFile);
    fCodaILocFile = fopen( CHAR(STRING_ELT(RsCodaILocFile->asSexp(), 0)), "rb");
    if (Verbose >= 4) {
      Rprintf("---- We have opened fCodaILocFile. \n"); R_FlushConsole();
    }
  }
  //fclose(fCodaILocFile);
  //fCodaILocFile = fopen( CHAR(STRING_ELT(RsCodaILocFile->asSexp(), 0)), "rb");
  //if (fCodaiTLocFile != NULL && RsCodaiTLocFile != NULL) {
  //   fclose(fCodaiTLocFile);
  //   fCodaiTLocFile = fopen( CHAR(STRING_ELT(RsCodaiTLocFile->asSexp(), 0)), "rb");
  //}
  bufferiLocLength = MaxBuffSize;  OnB = bufferiLocLength;
  while(ReadLastLocBeta == 0) {
    ReadBlocks++;
    if (Verbose >= 3) {
      Rprintf("\n\nBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB\n");
      Rprintf("BBBB(%3d)  ReadBlocks = %d. \n", ReadBlocks, ReadBlocks); R_FlushConsole();
      Rprintf("BBBB(%3d)  DeriveAlterProbability: We're reading Block ", ReadBlocks);
      Rprintf("%d, ReadLastLocBeta = %d.  TOnB=%d/%d so far\n", ReadBlocks, ReadLastLocBeta, TOnB, lSizeLoc);
      R_FlushConsole();
    }
    //if (ReadBlocks >= 3) {
    //  Rprintf("DeriveAlterProbability: Early Quit, ReadBlocks = %d.\n", ReadBlocks);
    //}
    if (TOnB >= lSizeLoc)  {
      Rprintf("BBBB(%3d)  ISSUE ISSUE ISSUE? \n", ReadBlocks); R_FlushConsole();
      Rprintf("BBBBB DeriveAlterProbability, ReadBlocks = %d, TOnB=%d=lSizeLoc, why are we here? \n",
        ReadBlocks, TOnB, lSizeLoc);
      Rprintf("      I'm worried we should not finish.  "); R_FlushConsole();
    } else if (OnB == bufferiLocLength) {
      if (Verbose >= 2) {
        Rprintf("BBBB(%3d) DeriveAlterProbability: Here we are at a maximum end, OnB=%d==bufferiLocLength, but TOnB=%d", ReadBlocks, OnB, TOnB);
        Rprintf(" = %d/%d, GetLocBuffer=%d, bufferiLocLength=%d\n",
          OnB, bufferiLocLength, TOnB, lSizeLoc, GetLocBuffer, bufferiLocLength);  R_FlushConsole();
      }
      ResidDiff = (int) ( ((long) lSizeLoc) - ((long)TOnB));
      if (ResidDiff <= 0) {
        Rprintf("BBBB(%3d) DeriveAlterProbability, TOnB=%d=BuffSize, but lSizeLoc=%d and ResidDiff=%d do we really kill here?\n",
          ReadBlocks, TOnB, lSizeLoc, ResidDiff);  R_FlushConsole();
      }
      WantLocBuffer = bufferiLocLength;
      if (ResidDiff < bufferiLocLength) { WantLocBuffer = ResidDiff; }
      ReadLastLocBeta = WantLocBuffer;   GetLocBuffer = 0;
      if (Verbose >= 2) {
        Rprintf("BBBB(%3d) DeriveAlterProbability: At BufferiLocLength: OnB=%d/%d, TOnB=%d/%d, lSizeLoc=%d, ",
          ReadBlocks, OnB, bufferiLocLength, TOnB, lSizeLoc, lSizeLoc); 
        Rprintf("BBBB(%3d) ResidDiff=%d will Draw WantLocBuffer=%d\n",
          ReadBlocks, ResidDiff, WantLocBuffer);   R_FlushConsole();
        Rprintf("BBBB(%3d): We set WantLocBuffer=%d=bufferiLocLength, and then ResidDiff=%d. \n",
          ReadBlocks, WantLocBuffer, bufferiLocLength, ResidDiff);   
        Rprintf("BBBB(%3d): And we will have GetLocBuffer to %d \n", ReadBlocks, GetLocBuffer); R_FlushConsole();
      }
      OnB = bufferiLocLength;
      ReDrawFunction(bufferILoc, sizeof(bufferILocType), bufferiLocLength, WantLocBuffer, GetLocBuffer, fCodaILocFile, OnB, "bufferILoc from fCodaILoc");
      if (GetLocBuffer <= 0) {
        Rprintf("BBBB(%3d): ISSUE GetLocBuffer=%d <= 0, DeriveAlterProbability, OnB at max, GetLocBuffer = %d after ReDraw, WantLocBuffer=%d, ResidDiff=%d\n",
          ReadBlocks, GetLocBuffer, GetLocBuffer, WantLocBuffer, ResidDiff); R_FlushConsole();
      }
      if (TOnB == 0 && OnB == 0 && Verbose >= 2)  {
        Rprintf("BBBB(%3d) DeriveAlterProbability: Here is bufferILoc after our load in. \n", ReadBlocks); R_FlushConsole();
        if (GetLocBuffer <= 0 || GetLocBuffer > bufferiLocLength) {
          Rprintf("BBBB(%3d) NOOOO GetLocBuffer == %d\n", ReadBlocks, GetLocBuffer, bufferiLocLength); R_FlushConsole();
        } else {
          Rprintf("BBBB(%3d) ** bufferILoc = ", ReadBlocks); R_FlushConsole(); 
          PrintLongVector(bufferILoc, GetLocBuffer); R_FlushConsole();
          Rprintf("\n");
        }
        Rprintf("BBBB(%3d) *** DAP Was that Good? \n", ReadBlocks);  R_FlushConsole();
      }
      if (WantLocBuffer + OnB > GetLocBuffer || GetLocBuffer == 0 || TOnB + WantLocBuffer >= lSizeLoc) {
        ReadLastLocBeta = 1;  AllNeedB = GetLocBuffer;
        if (Verbose >= 4) {
          Rprintf("BBBB(%3d) DeriveAlterProbability: Trigger last end because ", ReadBlocks);
          Rprintf("ReadLast=%d, GetLocBuffer=%d, TOnB=%d, lSizeLoc=%d\n",
            ReadLastLocBeta, GetLocBuffer, TOnB, lSizeLoc); R_FlushConsole();
        }
      } else { 
        ReadLastLocBeta = 0; AllNeedB = bufferiLocLength; 
      }
      if (NumGroups > 0 && sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) > 0) { 
         OnT = bufferiLocLength;
         if (Verbose >= 4) {
           Rprintf("BBBB(%3d): with OnT=%d set equal bufferiLocLength=%d do to sOnTau we ReDraw bufferiTLoc. \n",
             ReadBlocks, OnT, bufferiLocLength); R_FlushConsole();
         }
         ReDrawFunction(bufferiTLoc, sizeof(bufferILocType), bufferiLocLength, WantLocBuffer, GetLocBuffer, fCodaiTLocFile, OnT, "bufferiTLoc from iTLocFile");
      }
      OnProbBuff = bufferiLocLength;
      if (Verbose >= 4) {
        Rprintf("BBBB(%3d): we are on OnProbBuff=%d=bufferiLocLength, conduct ReDraw for bufferProb. \n",
          ReadBlocks, OnProbBuff); R_FlushConsole();
      }
      ReDrawFunction(bufferProb, sizeof(double), bufferiLocLength, WantLocBuffer, GetProbBuffer, fCodaProbFile, OnProbBuff, "bufferProb from fCodaProbFile");  
      if (Verbose >= 4) {
        Rprintf(" -- Read Beta: We read, TOnB = %d/%d, now bufferiLocLength = %d, GetLocBuffer=%d\n", 
          TOnB, lSizeLoc, bufferiLocLength, GetLocBuffer); R_FlushConsole();
      }
    } else {
      ResidDiff = (int) ( ((long) lSizeLoc) - ((long)TOnB));
      if (ResidDiff <= 0) {
        Rprintf("BBBB(%3d) DeriveAlterProbability(): -- ReadBeta: Okay, OnB=%d/%d, TOnB=%d/%d, ResidDiff is thus %d, TOnB != BuffSize, maybe good or no.\n",
          ReadBlocks, OnB, AllNeedB, TOnB, lSizeLoc, ResidDiff); R_FlushConsole();
      }
      WantLocBuffer = bufferiLocLength-OnB;
      if (ResidDiff < bufferiLocLength-OnB) { WantLocBuffer = ResidDiff; }
      if (Verbose >= 4) {
        Rprintf("BBBB(%3d) -- DeriveAlterProbability: OnB=%d/%d (biLL): we set ReadLastLocBeta=%d to WantLocBuffer=%d and GetLocBuffer=%d to 0.\n",
          ReadBlocks, OnB, bufferiLocLength, ReadLastLocBeta, WantLocBuffer, GetLocBuffer); R_FlushConsole();
      }
      ReadLastLocBeta = WantLocBuffer; GetLocBuffer=0;
      if (Verbose >= 2) {
        Rprintf("BBBB(%d) -- DeriveAlterProbability: In Top OnB=%d/%d, bufferiLocLength=%d, TOnB=%d/%d, ResidDiff=%d will Draw WantLocBuffer=%d, before, GetLocBuffer=%d,\n",
          ReadBlocks, OnB, bufferiLocLength, bufferiLocLength, TOnB, lSizeLoc, ResidDiff, WantLocBuffer, GetLocBuffer);   R_FlushConsole();
        Rprintf("BBBB(%d) ----- Note that GetLocBuffer is now. \n", ReadBlocks, GetLocBuffer); R_FlushConsole();
      }
      ReDrawFunction(bufferILoc, sizeof(bufferILocType), bufferiLocLength, WantLocBuffer, GetLocBuffer, (fCodaILocFile), OnB, "bufferILoc from fCodaILocFile");
      if (Verbose >= 2) {
        Rprintf("----- Note that, after Ran ReDrawFunction: GetLocBuffer is now %d from I File. \n", GetLocBuffer);
      }
      if (GetLocBuffer > bufferiLocLength) {
        Rprintf("Error: DeriveAlterProbability() after ReDrawFunction, bufferiLocLength=%d, < GetLocBuffer = %d \n",
          bufferiLocLength, GetLocBuffer); R_FlushConsole();
        Rprintf("DeriveAlterProbability: OnB=%d/%d, TOnB=%d/%d but WantLocBuffer=%d, GetLocBuffer=%d, ResidDiff=%d, bad!\n",
          OnB, GetLocBuffer, TOnB, lSizeLoc, WantLocBuffer, GetLocBuffer, ResidDiff);
        Rf_error("DeriveAlterProbability, this is bad. \n");
      } else if (GetLocBuffer <= 0) {
        Rprintf("ERROR (GetLocBuffer<=0): DeriveAlterProbability: Error, GetLocBuffer=%d, what is this? result=%d\n", GetLocBuffer, result); R_FlushConsole();
        Rf_error("DeriveAlterProbability: OnB=%d/%d, TOnB=%d, WantLocBuffer=%d, ResidDiff=%d\n",
          OnB, GetLocBuffer, TOnB, WantLocBuffer); 
      }
      if (Verbose >= 3 && OnB == 0) {
        Rprintf("BBBB(%d) DeriveAlterProbability: Success, OnB=%d/%d, TOnB=%d/%d, WantLocBuffer=%d, GetLocBuffer=%d\n",
          ReadBlocks, OnB, GetLocBuffer, TOnB, lSizeLoc, WantLocBuffer, GetLocBuffer); R_FlushConsole();
        Rprintf("BBBB(%d)   The bufferILoc is : ", ReadBlocks); R_FlushConsole(); PrintLongVector(bufferILoc, GetLocBuffer);
        Rprintf("\n"); R_FlushConsole();
      }
      if (GetLocBuffer < WantLocBuffer + OnB || GetLocBuffer == 0 || TOnB + WantLocBuffer >= lSizeLoc) {
        ReadLastLocBeta = 1;  AllNeedB = GetLocBuffer;
        if (Verbose >= 5) {
          Rprintf("BayesSpikeGibbs.cpp:::DeriveAlterProbability: AT01 Trigger last end because ReadLast=%d, Buff=%d,TOnB=%d, lSizeLoc=%d\n",
            ReadLastLocBeta, BuffSize, TOnB, lSizeLoc); R_FlushConsole();
        }
      } else { ReadLastLocBeta = 0; AllNeedB = GetLocBuffer;}
      if (NumGroups > 0 && sOnTau != NULL && 
          !Rf_isNull(sOnTau) && Rf_length(sOnTau) > 0) { 
          if (Verbose >= 5) {
            Rprintf("BayesSpikeGibbs.cpp(): DeriveAlterProbability: about to trigger bufferiTLoc=%d redraw. \n", bufferiTLoc);
            R_FlushConsole();
          }
          ReDrawFunction(bufferiTLoc, sizeof(bufferILocType), 
            bufferiLocLength, WantLocBuffer, GetLocBuffer, fCodaiTLocFile, OnT, "bufferiTLoc load in bufferiLength " );
      }
      OnProbBuff = bufferiLocLength;
      ReDrawFunction(bufferProb, sizeof(double), bufferiLocLength, WantLocBuffer, GetProbBuffer, fCodaProbFile, OnProbBuff, "bufferProb to OnProbBuff");
     
      if (Verbose >= 4) {
         Rprintf(" -- Read Beta: We read, OnB=%d/%d, TOnB = %d/%d, now BuffSize = %d\n", 
            OnB, GetLocBuffer, TOnB, lSizeLoc, GetLocBuffer); R_FlushConsole();
      }
    }
      OnSigBuff = bufferiLocLength;
      ReDrawFunction(bufferSig, sizeof(double), 
            bufferiLocLength, WantLocBuffer, GetLocBuffer, fCodaSigFile, OnSigBuff, "SigmaProb on Sigma");
      int WantPiABuffer = WantLocBuffer;  int GetPiABuffer = 0;
      if (RsOnPiA != NULL && Rf_length(RsOnPiA->asSexp()) == 2) {
        WantPiABuffer = WantLocBuffer * 2;
        OnPiABuff = bufferiLocLength*2;
        ReDrawFunction(bufferPiA, sizeof(double), 
            (bufferiLocLength*2), WantPiABuffer, GetPiABuffer, fCodaPiAFile, OnPiABuff, "bufferPiA, p1, OnPiABuff");    
      } else {
        WantPiABuffer = WantLocBuffer;
        OnPiABuff = bufferiLocLength;
        ReDrawFunction(bufferPiA, sizeof(double), 
            bufferiLocLength, WantPiABuffer, GetPiABuffer, fCodaPiAFile, OnPiABuff, "bufferPiA, p2, OnPiABuff");         
      }
      if (bufferPiA[0] <= 0.0 || (GetPiABuffer > 1 && bufferPiA[1] <= 0.0)) {
        Rprintf("DeriveAlterProb: Uh oh, on read in OnPiABuff = %d/%d, but bufferPiA = ", OnPiABuff, GetPiABuffer);
        R_FlushConsole(); PrintVector(bufferPiA, GetPiABuffer);
        Rprintf("\n"); R_FlushConsole();
      }
    if (Verbose >= 3) {
      Rprintf("BBBB(%3d) DeriveAlterProb: After bufferILoc Fill: TOnB = %d/%d, OnB = %d/%d, BuffSize=%d.\n", 
        ReadBlocks, TOnB, lSizeLoc, OnB, AllNeedB, BuffSize);
      R_FlushConsole();
    }
    //LastLocationInBetaBuffer = bufferILoc[0];

    SOnB = OnB;
    while (GetLocBuffer > 0 && OnB < GetLocBuffer) {
      if (Verbose >= 5) {
        Rprintf("\n\n##############################################################\n");
        Rprintf("## DeriveAlterProb, OnB=%d/%d: TOnB=%d/%d, GetLocBuffer=%d, Here we go.  \n", OnB, AllNeedB, TOnB, lSizeLoc, GetLocBuffer);
        R_FlushConsole();
      }
      
      int OldWantBetaBuffer;
      WantBetaBuffer = CalcLengthNeedBetas(OnB, GetLocBuffer, bufferILoc, BLOCKSIZE, &FinalLocationInLoc);
      if (WantBetaBuffer < 0 || WantBetaBuffer > BLOCKSIZE) {
        Rf_error("DeriveAlterProb: OnB=%d/%d, TOnB=%d/%d but WantBetaBuffer=%d, BLOCKSIZE=%d, GetLocBuffer=%d, FinalLocationInLoc=%d\n",
          OnB, GetLocBuffer, TOnB, lSizeLoc, WantBetaBuffer, BLOCKSIZE, GetLocBuffer, FinalLocationInLoc);
      } 
      if (FinalLocationInLoc > GetLocBuffer) {
        Rf_error("DeriveAlterProb: OnB=%d/%d, TOnB=%d/%d and WantBetaBuffer=%d, BLOCKSIZE=%d, FinalLocationInLoc=%d, GetLocBuffer=%d. lSizeLoc=%d\n",
          OnB, GetLocBuffer, TOnB, lSizeLoc, WantBetaBuffer, BLOCKSIZE, FinalLocationInLoc, GetLocBuffer, lSizeLoc); 
      }
      if (WantBetaBuffer > BLOCKSIZE) {
        Rprintf("DeriveAlterProb, Oh bad, OnB=%d/%d, TOnB=%d/%d, but WantBetaBuffer=%d, BLOCKSIZE=%d\n",
          OnB, GetLocBuffer, TOnB, lSizeLoc, WantBetaBuffer, BLOCKSIZE); R_FlushConsole();
      }
      if (Verbose >= 5) {
        Rprintf("DeriveAlterProb: In Middle, OnB=%d/%d, TOnB=%d/%d, GetLocBuffer = %d, solved WantBetaBuffer = %d, FinalLocationInLoc=%d\n",
          OnB, GetLocBuffer, TOnB, lSizeLoc, GetLocBuffer, WantBetaBuffer); R_FlushConsole();
      }
      OldWantBetaBuffer = WantBetaBuffer;
      OnBetaBuff = OldWantBetaBuffer;
      result = fread(bufferI, sizeof(int), WantBetaBuffer, fCodaIFile);
      GetBetaBuffer = result;
      if ((int) result != (int) WantBetaBuffer) {
        Rprintf("DeriveAlterProb: ERROR IN Middle, Reading I OnB=%d/%d, TOnB=%d/%d TOnBetaBuff=%d/%d, wanted WantBetaBuffer=%d loaded result=%d\n",
          OnB, GetLocBuffer, TOnB, lSizeLoc, TOnBetaBuff, lSizeI, WantBetaBuffer, result);
      }
      result = fread(bufferD, sizeof(double), WantBetaBuffer, fCodaDFile);
      if ((int) result != (int) WantBetaBuffer) {
        Rprintf("DeriveAlterProb: ERROR IN Middle, Reading D, DOnB=%d/%d, TOnB=%d/%d TOnBetaBuff=%d/%d, wanted WantBetaBuffer=%d loaded result=%d\n",
          OnB, GetLocBuffer, TOnB, lSizeLoc, TOnBetaBuff, lSizeD, WantBetaBuffer, result);
      }
      OnBetaBuff = 0;
      //ReDrawTwoBuffFunction(bufferI, bufferD, sizeof(int), 
      //  sizeof(double), OldWantBetaBuffer, WantBetaBuffer, GetBetaBuffer, fCodaIFile, fCodaDFile, 
      //  "  Beta Buffers ", OnBetaBuff);
      if (GetBetaBuffer <= 0  ||  GetBetaBuffer > BLOCKSIZE) {
        Rprintf("ERROR ReDrawTwoBuffFunction, WantBetaBuffer = %d, OnBetaBuff=%d/%d, TOnBetaBuff=%d/%d\n",
          WantBetaBuffer, OnBetaBuff, WantBetaBuffer, TOnBetaBuff, lSizeI);
        Rprintf("We Got GetBetaBuffer = %d for OnB=%d/%d, TOnB=%d/%d\n",
          GetBetaBuffer, OnB, GetLocBuffer, TOnB, lSizeLoc);
        Rprintf("Note OldWantBetaBuffer = %d\n", OldWantBetaBuffer);
        R_FlushConsole();
        Rf_error("Error ReDrawTwoBuffFunction I don't like OnBetaBuff=%d and GetBetaBuff=%d\n",
          OnBetaBuff, GetBetaBuffer);
      }
      if (OldWantBetaBuffer != GetBetaBuffer) {
        Rprintf("ReDraw Read ISSUE: Redraw Read ISSUE:  ReDraw Read ISSUE\n"); R_FlushConsole();
        Rprintf("OldWantBetaBuffer = %d, but GetBetaBuffer = %d\n", OldWantBetaBuffer,
         GetBetaBuffer); R_FlushConsole();
        Rprintf("We have that ReadBlocks = %d", ReadBlocks); R_FlushConsole();
        Rprintf("We had TOnBetaBuff = %d/%d and bufferILoc[OnB=%d]=%d I wonder. \n", TOnBetaBuff, lSizeI,
          OnB, bufferILoc[OnB]); R_FlushConsole();
        Rprintf("We would have bufferILoc[FinalLocationInLoc = %d] = %ld and we wanted %d.\n",
          FinalLocationInLoc, bufferILoc[FinalLocationInLoc], BLOCKSIZE);
        Rf_error("We are going to call this an error\n");
      }
      if (NumGroups > 0 && OnTauBuff >= 0) {
        int BFinalLocationInLoc = FinalLocationInLoc;
        int BOnT = OnT;  int BGetLocBuffer = GetLocBuffer;
        int BWantTauBuffer = WantTauBuffer;  int BFinalLocationInTauLoc = FinalLocationInTauLoc;
        int BBlockSize = BLOCKSIZE;
        WantTauBuffer = CalcLengthTauDraws(FinalLocationInLoc, OnT, GetLocBuffer, bufferiTLoc, BLOCKSIZE, &FinalLocationInTauLoc, Verbose);
        OnTauBuff = WantTauBuffer;  TOnTauBuff = bufferiTLoc[0];
        if (WantTauBuffer == 0) {
          Rprintf("BBBB(%3d): ERROR Discovered. \n", ReadBlocks); R_FlushConsole();
          Rprintf("BBBB(%3d) -----  Error after CalcLengthTauDraws. \n", ReadBlocks); R_FlushConsole();
          Rprintf("BBBB(%3d) After CalcLengthTauDraws: Is This for real?  WantTauBuffer = 0\n", ReadBlocks );
          Rprintf("BBBB(%3d) ERROR --  We get OnT = %d/%d, TOnT=%d/%d, FinalLocationInLoc=%d", ReadBlocks, OnT, GetTauBuffer, TOnT, lSizeTLoc, FinalLocationInLoc);
          Rprintf("BBBB(%3d) ERROR --  We got FinalLocationInTauLoc = %d \n", ReadBlocks, FinalLocationInTauLoc); R_FlushConsole();
          Rprintf("BBBB(%3d) ERROR -- Heres Old FinalLocationInLoc=%d, BOnT=%d, BWantTauBuffer = %d, BBlockSize=%d",
            ReadBlocks, BFinalLocationInLoc, BOnT, BWantTauBuffer, BBlockSize);
          Rprintf("BBBB(%3d) FinalLocationInTauLoc turned out to be %d from %d and WantTauBuffer=%d", 
            ReadBlocks, FinalLocationInTauLoc, BFinalLocationInTauLoc, WantTauBuffer); R_FlushConsole();
          Rprintf("BBBB(%3d) Error  bufferiTLoc[OnT=%d]=%d, bufferiTLoc[FinalLocationInTau=%d]=%d\n",
            ReadBlocks, OnT, bufferiTLoc[OnT], FinalLocationInTauLoc, bufferiTLoc[FinalLocationInTauLoc]); R_FlushConsole();
          Rf_error("--  I suspect this is an error.  End On WantTauBuffer=%d \n", WantTauBuffer);
        }
        if (WantTauBuffer < 0 || WantTauBuffer > BLOCKSIZE) {
          Rprintf("ERROR WantTauBuffer, DeriveAlterProb: OnB=%d/%d, TOnB=%d/%d, OnTauBuff=%d, but WantBetaBuffer=%d, BLOCKSIZE=%d, GetLocBuffer=%d, FinalLocationInLoc=%d\n",
            OnB, GetLocBuffer, TOnB, lSizeLoc, OnTauBuff, WantBetaBuffer, BLOCKSIZE, GetLocBuffer, FinalLocationInLoc);
          Rf_error("DeriveAlterProb: OnT=%d/%d, TOnT=%d/%d, WantTauBuffer=%d, BLOCKSIZE=%d, FinalLocationTauLoc=%d, GetLocBuffer=%d\n",
            OnT, GetLocBuffer, TOnT, lSizeLoc, WantTauBuffer, BLOCKSIZE, FinalLocationInTauLoc, GetLocBuffer);
        } 
        if (FinalLocationInTauLoc > GetLocBuffer) {
          Rprintf("ERROR FinalLocationTauLoc: DeriveAlterProb: OnB=%d/%d, TOnB=%d/%d and WantBetaBuffer=%d, BLOCKSIZE=%d, FinalLocationInLoc=%d, GetLocBuffer=%d. lSizeLoc=%d\n",
            OnB, GetLocBuffer, TOnB, lSizeLoc, WantBetaBuffer, BLOCKSIZE, FinalLocationInLoc, GetLocBuffer, lSizeLoc); 
          Rf_error("DeriveAlterProb: OnT=%d/%d, TOnT=%d/%d, WantTauBuffer=%d, BLOCKSIZE=%d, FinalLocationTauLoc=%d, GetLocBuffer=%d\n",
            OnT, GetLocBuffer, TOnT, lSizeLoc, WantTauBuffer, BLOCKSIZE, FinalLocationInTauLoc, GetLocBuffer);
        }
        result = fread(bufferiT, sizeof(int), WantTauBuffer, fCodaiTFile);  
        if (result <= 0) {
          Rprintf("DeriveAlterProb: IN MIDDLE ERROR, OnB=%d/%d, OnT=%d/%d, TOnB=%d/%D, TOnTauBuff=%d/%D, WantTauBuffer=%d, but result=%d\n",
            OnB, GetLocBuffer, OnT, GetTauBuffer, TOnB, lSizeLoc, TOnT, lSizeTLoc, WantTauBuffer, result ); R_FlushConsole();
          Rprintf("  ---   No way is this going to be working right");
        }
        GetTauBuffer = result;
        if ((int) result != (int) WantTauBuffer) {
         Rprintf("DeriveAlterProb: ERROR IN Middle, Reading iT OnB=%d/%d, OnT=%d/%d, TOnB=%d/%d TOnTauBuff=%d/%d, wanted WantTauaBuffer=%d loaded result=%d\n",
           OnB, GetLocBuffer, OnT, GetLocBuffer, TOnB, lSizeLoc, TOnTauBuff, lSizeiT, WantTauBuffer, result);
         OnTauBuff = 0;
       }
       result = fread(bufferdT, sizeof(double), WantTauBuffer, fCodadTFile);
       if ((int) result != (int) WantTauBuffer) {
         Rprintf("DeriveAlterProb: ERROR IN Middle, Reading dT OnB=%d/%d, OnT=%d/%d, TOnB=%d/%d TOnTauBuff=%d/%d, wanted WantTauaBuffer=%d loaded result=%d\n",
           OnB, GetLocBuffer, OnT, GetLocBuffer, TOnB, lSizeLoc, TOnTauBuff, lSizeiT, WantTauBuffer, result);
       }
       // ReDrawTwoBuffFunction(bufferiT, bufferdT, sizeof(int), 
       //   sizeof(double), WantTauBuffer, WantTauBuffer, GetTauBuffer, fCodaiTFile, fCodadTFile, 
       //   "  Tau Buffers ", OnTauBuff);
        OnTauBuff = 0;
        if (GetTauBuffer <= 0) {
          Rprintf("DeriveAlterProb: Why after all this is GetTauBuffer=%d\n?", GetTauBuffer); R_FlushConsole();
        }
     }
     if (Verbose >= 6) {
       Rprintf("New Read in Buffer, TOnB=%d, OnB=%d, TOnT = %d, OnT =%d",
         TOnB, OnB, TOnT, OnT);
       Rprintf(" = %d, OnBetaBuff = %d/%d, TOnBetaBuff = %d/%d.\n",
         OnBetaBuff, GetBetaBuffer, TOnBetaBuff, lSizeI); R_FlushConsole();
     }
     while(OnBetaBuff < GetBetaBuffer && OnB < GetLocBuffer-1) {    
      if (Verbose >= 6) {
        Rprintf("DeriveAlterProb: About to do While OnBetaBuff/ReadIn B loop.\n");
        R_FlushConsole();
      }
      if (OnB == 0 && OnBetaBuff == 0 && Verbose >= 4) {
        Rprintf("*************************************************************************\n");
        Rprintf("**** BayesSpikeGibbs.cpp:::DeriveAlterProb() here we are to read files in\n");
        Rprintf("**** OnB=%d/%d for TOnB=%d/%d and bufferLoc=", OnB, GetLocBuffer,
          TOnB, lSizeLoc); PrintLongVector(bufferILoc, GetLocBuffer);   R_FlushConsole();
        Rprintf("\n**** OnSigBuff=%d/%d for TOnSigBuff=%d/%d and bufferSig =", OnSigBuff, GetLocBuffer,
          TOnSigBuff, lSizeLoc); R_FlushConsole(); PrintVector(bufferSig, GetLocBuffer); R_FlushConsole();
        Rprintf("\n**** OnPiABuff=%d/%d for TOnPiABuff=%d/%d and bufferILoc=", OnPiABuff, GetPiABuffer,
          TOnPiABuff, lSizeLoc); R_FlushConsole(); PrintVector(bufferPiA, GetLocBuffer);   R_FlushConsole();
        Rprintf("\n**** OnProbBuff=%d/%d for TOnProbBuff=%d/%d and bufferProb=", OnProbBuff, GetProbBuffer,
          TOnProbBuff, lSizeLoc); PrintVector(bufferProb, GetLocBuffer); R_FlushConsole();
        Rprintf("\n**** OnBetaBuff=%d/%d for TOnBetaBuff=%d/%d and bufferI =", OnBetaBuff, GetBetaBuffer,
          TOnBetaBuff, lSizeI); PrintVector(bufferI, GetBetaBuffer); R_FlushConsole();
        Rprintf("\n**** OnBetaBuff=%d/%d for TOnBetaBuff=%d/%d and bufferD =", OnBetaBuff, GetBetaBuffer,
          TOnBetaBuff, lSizeD); PrintVector(bufferD, GetBetaBuffer);  R_FlushConsole();
        if (NumGroups > 0 && sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) >= 0) {
          Rprintf("\n**** OnT=%d/%d for TOnT=%d/%d and bufferiTLoc=", OnT, GetLocBuffer,
          TOnT, lSizeLoc); PrintLongVector(bufferiTLoc, GetLocBuffer); R_FlushConsole();
          Rprintf("\n**** OnTauBuff=%d/%d for TOnTauBuff=%d/%d and bufferI =", OnTauBuff, GetTauBuffer,
          TOnTauBuff, lSizeiT); PrintVector(bufferiT, GetTauBuffer); R_FlushConsole();
          Rprintf("\n**** OnBetaBuff=%d/%d for TOnBetaBuff=%d/%d and bufferD =", OnTauBuff, GetTauBuffer,
          TOnTauBuff, lSizedT); PrintVector(bufferdT, GetTauBuffer); R_FlushConsole();  
        }
        Rprintf("\n****    BayesSpikeGibbs.cpp:::DeriveAlterProb(), Will this work? \n"); R_FlushConsole();
        Rprintf("*************************************************************************\n");
        R_FlushConsole();                          
      }

      if (Verbose >= 6) {
        Rprintf("DeriveAlterProb: Reading in OnBetaBuff = %d/%d, for TOnB = %d \n", 
          OnBetaBuff, GetBetaBuffer, TOnB); R_FlushConsole();
      }

      if (OnBetaBuff >= GetBetaBuffer) {
        Rprintf("DeriveAlterProb: Possible Error issue, OnBetaBuff=%d/%d/%d, TOnBetaBuff=%d/%d, OnB=%d/%d, TOnB=%d/%d\n",
          OnBetaBuff, GetBetaBuffer, BLOCKSIZE, TOnBetaBuff, lSizeI, OnB, GetLocBuffer, TOnB, lSizeLoc); R_FlushConsole();
      }
      if (bufferI[OnBetaBuff] >= 0) {
        Rprintf("DeriveAlterProb: Error in Buffer reads, bufferI[OnBetaBuff = %d/%d] = %d, for TOnBetaBuff = %d, in file %s.\n",
          OnBetaBuff, GetBetaBuffer, bufferI[OnBetaBuff], (int) TOnBetaBuff,
          CHAR(STRING_ELT(RsCodaIFile->asSexp(), 0))); R_FlushConsole();
        int ATT1 = 10;
        if (GetBetaBuffer - OnBetaBuff < ATT1) {
          ATT1 = GetBetaBuffer - OnBetaBuff;
        }
        Rprintf("bufferI[OBB+0:%d] = ", ATT1-1);  R_FlushConsole();

        PrintVector(bufferI, ATT1);  Rprintf("\n"); R_FlushConsole();
        Rprintf(" And and bufferD[OBB+0:%d]: ", ATT1-1);
        PrintVector(bufferD, ATT1); Rprintf("\n"); R_FlushConsole();
        Rf_error("DerivAlterProb:  Let's Call This an Error\n");
      }
      if (fabs( (round(bufferD[OnBetaBuff])) - (double) bufferD[OnBetaBuff]  ) >= .1) {
        Rprintf("Error in Buffer reads, fluff float bufferD[OnBetaBuff = %d] = %f, for TOnBetaBuff = %d, in file %s.\n",
          OnBetaBuff, bufferD[OnBetaBuff], TOnBetaBuff,
          CHAR(STRING_ELT(RsCodaJFile->asSexp(), 0))); R_FlushConsole();  
        Rf_error("This Is probably terrible\n");           
      }
      LenBeta = (int) round(fabs(bufferD[OnBetaBuff]));
      if (LenBeta+1+OnBetaBuff > GetBetaBuffer) {
        Rprintf("UhOh, bufferD[OnBetaBuff=%d]=%f, but GetBetaBuffer=%d, LenBeta=%d, might go bad.\n",
          OnBetaBuff, bufferD[OnBetaBuff], GetBetaBuffer, LenBeta); R_FlushConsole();
        Rprintf(" Not that bufferILoc[OnB=%d/%d] = %ld, bufferILoc[OnB-1] = %ld, bufferILoc[OnB+1]=%d, bufferILoc[OnB+2] = %d\n",
          OnB, GetLocBuffer, bufferILoc[OnB], OnB >= 1 ? bufferILoc[OnB-1] : -1,
          OnB+1 < GetLocBuffer ? bufferILoc[OnB+1] : -1, OnB+2 < GetLocBuffer ? bufferILoc[OnB+2] : -1);
        Rprintf(" Meanwhile TOnBetaBuff = %d/%d \n", TOnBetaBuff, lSizeI); R_FlushConsole();
        R_FlushConsole();
        Rprintf(" Here is  bufferD[OnBetaBuff:GetBetaBuffer-1] ("); R_FlushConsole();
        int Ado = 0;
        for (int iti = OnBetaBuff; iti < GetBetaBuffer; iti++ ) {
          if (Ado >= 8) { Rprintf("\n"); R_FlushConsole(); Ado = 0; }
          if (iti == GetBetaBuffer-1) {
             Rprintf("%.3f)\n", bufferD[iti]);  R_FlushConsole();
          } else {
            Rprintf("%.3f, ", bufferD[iti]); R_FlushConsole();
          }
          Ado++;
        }
        result=fread(bufferD, sizeof(double), LenBeta, fCodaDFile);
        if ((int) result > 0) {
          Rprintf(" Here is the remainder result=%d, LenBeta=%d, of fCodaDFile : \n(", result, LenBeta); R_FlushConsole();
          Ado = 0;
          for (int iti = 0; (int) iti < (int) result; iti++) {
            if (Ado >= 8) { Rprintf("\n"); R_FlushConsole(); Ado = 0; }
            if (iti == (int) result-1) {
              Rprintf("%.3f)\n", bufferD[iti]); R_FlushConsole();
            } else {
              Rprintf("%.3f, ", bufferD[iti]); R_FlushConsole();
            }
            Ado++;
          }
        } else {
          Rprintf("  Nothing was read in from bufferD, it sucks!\n"); R_FlushConsole();
        }
        int OnLocationBuffer = SOnB;  int LengthLocationBuffer = GetLocBuffer;
        if (OnLocationBuffer < 0 || OnLocationBuffer >= LengthLocationBuffer) {
         Rf_error("BayesSpikeGibbs.cpp:::Error, CalcNewProb, OnLocationBuffer is bad = %d but length = %d, SOnB=%d\n",
         OnLocationBuffer, LengthLocationBuffer, SOnB);
        }
        int TotLengthSoFar = 0; bufferILocType * LocationBuffer = bufferILoc;
        long int MyLastOn = LocationBuffer[OnLocationBuffer];
        long int MyStart = LocationBuffer[OnLocationBuffer];
        int LengthThisBuffer = BLOCKSIZE;
        Rprintf("About to Rerun CalcuLengthNeedBetas again, BlockSize=%d but SOnB=%d \n",
          BLOCKSIZE, SOnB); R_FlushConsole();
        for (int ii = OnLocationBuffer; ii < LengthLocationBuffer; ii++) {
         if (LocationBuffer[ii] - MyStart < LengthThisBuffer) {
           MyLastOn = LocationBuffer[ii];  TotLengthSoFar =  LocationBuffer[ii] - MyStart;
         } else if (LocationBuffer[ii] - MyStart == LengthThisBuffer) {
           MyLastOn = LocationBuffer[ii]; 
           TotLengthSoFar = LengthThisBuffer;
           Rprintf("CalcuLengthNeedBetas Vic: The ii LocationBuffer ends equal ii = %d/%d MyLastOn=%d, MyStart=%d, TotLengthSoFar = %d",
             ii, LengthLocationBuffer, MyLastOn, MyStart, TotLengthSoFar); R_FlushConsole();
           break;
         } else {
           MyLastOn = LocationBuffer[ii-1];
           TotLengthSoFar = MyLastOn - MyStart;
           Rprintf(" CalcuLengthNeedBetas Vic: The ii LocationBuffer ends more ii = %d/%d MyLastOn=%d, MyStart=%d, TotLengthSoFar = %d",
             ii, LengthLocationBuffer, MyLastOn, MyStart, TotLengthSoFar); R_FlushConsole();
           break;
        }
       } 
       Rprintf("  When we let ii = %d/%d, MyLastOn=%d, MyStart=%d still, LocationBuffer[%d] = %d\n",
         LengthLocationBuffer, LengthLocationBuffer, MyLastOn, MyStart, LengthLocationBuffer-1,
         LocationBuffer[LengthLocationBuffer-1]); R_FlushConsole();
       Rprintf("  Is CalcLengthNeedBetasWorking?"); R_FlushConsole();   
       
        Rf_error("We will call this an error. \n"); R_FlushConsole();
      }

      if (NumGroups > 0 && sOnTau != NULL && 
        !Rf_isNull(sOnTau) && Rf_length(sOnTau) > 0) {
        if (OnTauBuff >= GetTauBuffer) {
          if (Verbose >= 7) {
            Rprintf("DeriveAlterProb: Considering Redrawing OnTauBuff, but won't!\n");
            R_FlushConsole();
          }
          Rf_error("BayesSpikeGibbs:cpp::DeriveAlterProb, OnTauBuff=%d, but GetTauBuffer=%d, can't do this. \n", OnTauBuff, GetTauBuffer); 
        }
        if (Verbose >= 7) {
          Rprintf("DeriveAlterProb: Reading OnTauBuff=%d/%d, for bufferiT\n",
            OnTauBuff, GetTauBuffer); R_FlushConsole();
        }
        if (bufferiT[OnTauBuff] >= 0) {
          Rprintf("Error in Buffer reads, bufferiT[OnTauBuff = %d] = %d, for TOnTauBuff = %d, in file %s.\n",
            OnTauBuff, bufferiT[OnTauBuff], TOnTauBuff,
           CHAR(STRING_ELT(RsCodaiTFile->asSexp(), 0))); R_FlushConsole();
        }
        if (fabs( (round(bufferdT[OnTauBuff])) - (double) bufferdT[OnTauBuff]  ) >= .1) {
          Rprintf("Error in Buffer reads, fluff float bufferdT[OnTauBuff = %d] = %f, for TOnTauBuff = %d, in file %s.\n",
            OnTauBuff, bufferdT[OnTauBuff], TOnTauBuff,
            CHAR(STRING_ELT(RsCodadTFile->asSexp(), 0))); R_FlushConsole();             
        }
        LenTau = (int) round(fabs(bufferdT[OnTauBuff]));
        if (LenTau + 1+ OnTauBuff > GetTauBuffer) {
          Rprintf("DeriveAlterProb: Uh Oh, LenTau=%d, bufferdT[OnTauBuff=%d]=%f but GetTauBuffer = %d\n",
            LenTau, OnTauBuff, bufferdT[OnTauBuff], GetTauBuffer); R_FlushConsole();
          Rprintf(" --- What is This, OnTauBuff=%d/%d, TOnTauBuff=%d/%d \n", OnTauBuff, GetTauBuffer, TOnTauBuff, lSizedT);
          R_FlushConsole();
          Rprintf(" --- I am not happy about where this is going. \n"); R_FlushConsole();
          if (GetTauBuffer <= 0) {
            Rprintf(" --- Why is GetTauBuffer = %d ? \n", GetTauBuffer); R_FlushConsole();
          }  else {
            Rprintf(" --- Hey, GetTauBuffer = %d \n", GetTauBuffer); R_FlushConsole();
          }
        }
      }
     

      if (OnProbBuff >= GetLocBuffer) {
        Rprintf("DeriveAlterProb: UhOh, OnProbBuff = %d, but GetLocBuffer=%d on OnB=%d/%d, OnBetaBuff=%d/%d\n",
          OnProbBuff, GetLocBuffer, OnB, GetLocBuffer, OnBetaBuff, GetBetaBuffer); R_FlushConsole();
      }
      if (OnSigBuff >= GetLocBuffer) {
        Rprintf("DeriveAlterProb: UhOh, OnSigBuff = %d, but GetLocBuffer=%d on OnB=%d/%d, OnBetaBuff=%d/%d\n",
          OnSigBuff, GetLocBuffer, OnB, GetLocBuffer, OnBetaBuff, GetBetaBuffer); R_FlushConsole();
      }
      if (OnPiABuff >= GetLocBuffer) {
        Rprintf("DeriveAlterProb: UhOh, OnPiABuff = %d, but GetLocBuffer=%d on OnB=%d/%d, OnBetaBuff=%d/%d\n",
          OnPiABuff, GetLocBuffer, OnB, GetLocBuffer, OnBetaBuff, GetBetaBuffer); R_FlushConsole();
      }
      OldProb = bufferProb[OnProbBuff];
      Sigma = bufferSig[OnSigBuff];
      if (Rf_length(sOnPiA) == 1) {
        PiA1 = bufferPiA[OnPiABuff];  PiA2 = PiA1;
      } else {
        PiA1 = bufferPiA[OnPiABuff * Rf_length(sOnPiA)];
        PiA2 = bufferPiA[OnPiABuff * Rf_length(sOnPiA)+1];
      }
     if (PiA1 <= 0.0 || ( Rf_length(sOnPiA) >= 2 && PiA2 <= 0.0)) {
       Rprintf("Error: Uh Oh, PiA1 Set set in middle %d/%d.  We have GetLocBuffer = %d, length sOnPiA=%d, PiA1=%f, PiA2 = %f\n",
         OnPiABuff, GetLocBuffer, GetLocBuffer, Rf_length(sOnPiA), PiA1, PiA2);
       Rprintf("Error: OnB=%d/%d, TOnB=%d/%d, OnbetaBuff =%d/%d \n",
         OnB, GetLocBuffer, TOnB, lSizeLoc, OnBetaBuff, GetBetaBuffer); R_FlushConsole();
       Rprintf("  Here is bufferPiA: "); PrintVector(bufferPiA, GetLocBuffer);
       Rprintf("\n  Was it helpful? \n"); R_FlushConsole();
       Rf_error(" This will generate a New Prob Error. \n");
     }
           
      if (Sigma <= 0.0 || PiA1 >= 1.0) {
        Rprintf("DeriveAlterProb: Bad Issue we have OnSigBuff=%d/%d=%f, OnPiABuff=%d/%d=%f,%f\n",
          OnSigBuff, GetLocBuffer, Sigma, GetPiABuffer, PiA1, PiA2);
        Rprintf("  Here's Sigma Buffer: "); PrintVector(bufferSig+OnSigBuff, GetLocBuffer-OnSigBuff);
        Rprintf("\n   And here's PiA Buffer: "); PrintVector(bufferPiA, GetPiABuffer-OnSigBuff);
        Rprintf("\n   And here's ProbBuffer: "); PrintVector(bufferProb, GetLocBuffer-OnProbBuff);
        Rprintf("\n"); R_FlushConsole();
      }
      if (Verbose >= 5) {
        Rprintf("DeriveAlterProb: Considering writing to AlterWeight Buffer:%d/%d.\n",
          LengthWrittenAlterWeightBuffer, LengthAlterWeightBuffer);
        R_FlushConsole();
      }
      if (LengthWrittenAlterWeightBuffer >= LengthAlterWeightBuffer -2) {
        if (Verbose >= 3) {
          Rprintf("*****************************************************************\n");
          Rprintf("** Save Moment Saving to Coda Weight File. \n"); R_FlushConsole();
        }
        if (NewAlterWeightBufferWrite == 1) {
          fAlterWeightFile =  
            fopen( CHAR(STRING_ELT(RsAlterWeightFile->asSexp(), 0)), "wb");
          NewAlterWeightBufferWrite = 0;
        } else {
          fAlterWeightFile =  
            fopen( CHAR(STRING_ELT(RsAlterWeightFile->asSexp(), 0)), "ab");
        }
        if (fAlterWeightFile != NULL) { TotalOpenedFiles++; }
        if (fAlterWeightFile == NULL) {
          Rprintf("***********************************************\n");
          Rprintf("** Derive Alter Probability fAlterWeightFile, bad problem, you opened to null! \n");
          R_FlushConsole();
        }
        fwrite( AlterWeightBuffer, sizeof(double), 
          LengthWrittenAlterWeightBuffer, fAlterWeightFile );
        ffclose(fAlterWeightFile, "fAlterWeightFile");
        TotalClosedFiles++;
        LengthWrittenAlterWeightBuffer = 0;
        if (Verbose >= 4) {
        Rprintf("*** Done Saving.   \n");
        Rprintf("*****************************************************************\n");
        R_FlushConsole();
        }
      }
      if (Verbose >= 5) {
        Rprintf("  --  About to Calc using buffer dT OnT=%d/%d, OnBetaBuff=%d/%d\n", 
          OnT, GetTauBuffer, OnBetaBuff, GetBetaBuffer); R_FlushConsole();
      }
      if (LenTau > 0 && LenTau + OnTauBuff+1 > GetTauBuffer) {
        Rprintf("CalcNewProb: This is going to fail, LenTau=%d, OnTauBuff=%d, GetTauBuffer=%d\n",
          LenTau, OnTauBuff, GetTauBuffer); R_FlushConsole();
      }

      if (bufferdT != NULL && NumGroups > 0 && sOnTau != NULL && !Rf_isNull(sOnTau)) {
        NewProb = CalcNewProb(LenBeta, bufferD+1+OnBetaBuff, bufferI+1+OnBetaBuff, 
          LenTau,  bufferdT+1+OnTauBuff, bufferiT+1+OnTauBuff, 
          AlterWeightdfRobit, AlterWeightdfTNoise, PiA1, PiA2, EY,Sigma);
      } else {
        NewProb = CalcNewProb(LenBeta, bufferD+1+OnBetaBuff, bufferI+1+OnBetaBuff, 
          0,  NULL, NULL, 
          AlterWeightdfRobit, AlterWeightdfTNoise, PiA1, PiA2, EY,Sigma);      
      }
      if (Verbose >= 6) {
        Rprintf("DeriveAlterProb: CalcNewProb finished, it is %f\n", NewProb);
        R_FlushConsole();
      }
      AlterWeightBuffer[LengthWrittenAlterWeightBuffer] = 
        NewProb - OldProb / AlterWeightTemperature;
      LengthWrittenAlterWeightBuffer++;  LengthTotalWrittenAlterWeightBuffer++;
      if (R_isnancpp(NewProb)  || !R_finite(NewProb)) {
        Rprintf("DeriveAlterProb: Error, NewProb = %f \n", NewProb);
        Rprintf("Beta = "); PrintVector(bufferD+1+OnBetaBuff, LenBeta);
        if (bufferdT != NULL) {
          Rprintf("\n Tau = "); PrintVector(bufferdT, LenTau);
        }
        Rprintf(" and PiAs = %f,%f, and Sigmas = %f \n", PiA1, PiA2, Sigma);
        Rprintf("We Have OnBetaBuff = %d/%d, TOnBetaBuff=%d/%d, OnTauBuff=%d/%d, OnB=%d/%d, TOnB=%d\n",
          OnBetaBuff, GetBetaBuffer, TOnBetaBuff, lSizeI, OnTauBuff, GetTauBuffer, OnB, GetLocBuffer, TOnB);
        Rprintf("  We'rd going to reject an error \n"); R_FlushConsole();
        LaunchEnd( 0);
        Rf_error("Its Time to show a reject!\n");  
      }
      if (Verbose >= 6) {
        Rprintf("DeriveAlterProb: Finished Reading OnSigBuff = %d/%d, to OnB=%d/%d, with NewProb = %f\n",
          OnSigBuff, GetLocBuffer, OnB, GetLocBuffer, NewProb-OldProb);
        R_FlushConsole();
      }
      if (NumGroups > 0) {
        OnTauBuff+= LenTau+1;  TOnTauBuff+=LenTau+1; 
      } 
      if (LenBeta <= 0) { LenBeta = 0; }
      OnBetaBuff+=LenBeta+1; TOnBetaBuff+=LenBeta+1;
      OnPiABuff++;  OnSigBuff++;  OnProbBuff++;   
      TOnPiABuff++;  TOnSigBuff++;  TOnProbBuff++;
      OnT++; TOnT++;  OnB++;  TOnT++;  TOnB++;
      if (GetLocBuffer == 0) {
        Rf_error(" -- DeriveAlterProb: GetLocBuffer = %d,  OnB=%d/%d, TOnB=%d/%d not good! \n", 
          GetLocBuffer, OnB, GetLocBuffer, TOnB, lSizeLoc);
      }
      if (Verbose >= 4) {
        Rprintf("DAP: OnB=%d/%d, TOnB=%d/%d, BetaBufferLength=%d,", OnB, GetLocBuffer, TOnB,
          lSizeLoc, GetBetaBuffer);
        Rprintf("OnBetaBuff=%d/%d, TOnBetaBuff = %d/%d\n", OnBetaBuff, GetBetaBuffer, TOnBetaBuff,
          lSizeD);
        R_FlushConsole();
      }
      if (OnB >= GetLocBuffer-1 && Verbose >= 3) {
        Rprintf("DAP: OnB=%d/%d in loop on end, with OnBetaBuff=%d/%d, I wonder what's next. \n", OnB, GetLocBuffer,
          OnBetaBuff, GetBetaBuffer); R_FlushConsole();
      }
    }
    if (Verbose >= 3) {
      Rprintf("DAP: OnB=%d/%d, TOnB=%d/%d, OnBetaBuff=%d/%d/%d we have moved out of a level \n",
        OnB, GetLocBuffer, TOnB, lSizeLoc, OnBetaBuff, GetBetaBuffer, BLOCKSIZE); R_FlushConsole();
    }
  if (GetLocBuffer > 0 && OnB >= GetLocBuffer -1) {
     if (Verbose >= 3) {
       Rprintf("DAP: we are running end of a buffer OnB=%d/%d for TOnB=%d/%d\n", OnB, GetLocBuffer,
         TOnB, lSizeLoc); R_FlushConsole();
     }
     if (GetLocBuffer > MaxBuffSize) {
       Rf_error("DAP: Error, GetLocBuffer = %d, but MaxxBuffSize=%d\n", GetLocBuffer, MaxBuffSize); 
     }
     int IdEndBeta = 0; int LengthEndBeta = 0;  double InLengthEndBeta = 0.0;
     int IdEndTau = 0; double InLengthEndTau; int LengthEndTau = 0;
     int result;
     OldProb = bufferProb[GetLocBuffer-1];
     Sigma = bufferSig[GetLocBuffer-1];
     if (Rf_length(sOnPiA) == 1) {
        PiA1 = bufferPiA[GetLocBuffer-1];  PiA2 = PiA1;
     } else {
        PiA1 = bufferPiA[(GetLocBuffer-1) * Rf_length(sOnPiA)];
        PiA2 = bufferPiA[(GetLocBuffer-1) * Rf_length(sOnPiA)+1];
     }
     if (PiA1 <= 0.0 || ( Rf_length(sOnPiA) >= 2 && PiA2 <= 0.0)) {
       Rprintf("Error: Uh Oh, PiA1 Set at End.  We have GetLocBuffer = %d, length sOnPiA=%d, PiA1=%f, PiA2 = %f\n",
         GetLocBuffer, Rf_length(sOnPiA), PiA1, PiA2);
       Rprintf("Error: OnB=%d/%d, TOnB=%d/%d, OnbetaBuff =%d/%d \n",
         OnB, GetLocBuffer, TOnB, lSizeLoc, OnBetaBuff, GetBetaBuffer); R_FlushConsole();
       Rf_error(" This will generate a New Prob Error. \n");
     }

     result = fread(&IdEndBeta, sizeof(int),  1, fCodaIFile);
     result = fread(&InLengthEndBeta, sizeof(double),  1, fCodaDFile);
     if (fabs(InLengthEndBeta - round(InLengthEndBeta)) >= .1) {
       Rprintf("DeriveAlterProb, At end OnB=%d/%d, TOnB=%d/%d we got LengthEndBeta = %d, not good should be count of betas. \n", OnB, GetLocBuffer,
         TOnB, lSizeLoc, InLengthEndBeta); R_FlushConsole();
      Rprintf("IdEndBeta is %d too. \n", IdEndBeta);
      Rf_error("DeriveAlterProb: Not reading Error correctly.\n");
     }
     if (IdEndBeta > 0) {
       Rprintf("DeriveAlterProb, At end OnB=%d/%d, TOnB=%d/%d we got LengthEndBeta = %d, not good should be count of betas. \n", OnB, GetLocBuffer,
         TOnB, lSizeLoc, InLengthEndBeta); R_FlushConsole();
      Rprintf("IdEndBeta is %d too. \n", IdEndBeta);
      Rf_error("DeriveAlterProb: Not reading Error correctly.\n");
     }
     LengthEndBeta = (int)(fabs(round(InLengthEndBeta)));
     if (Verbose >= 4) {
       Rprintf("DAP: at end OnB=%d/%d, we read in LenBeta=%d for position %d, TOnB=%d/%d\n", OnB, GetLocBuffer,
         LengthEndBeta, IdEndBeta, TOnB, lSizeLoc); R_FlushConsole();
     }
     OnBetaBuff = 0; 
     result = fread(bufferD, sizeof(double), LengthEndBeta, fCodaDFile);
     if (result < LengthEndBeta) {
       Rprintf("DeriveAlterProb, no, at end OnB=%d/%d, TOnB=%d/%d, LengthEndBeta = %d, but for double read result = %d\n",
         OnB, GetLocBuffer,TOnB, lSizeLoc, LengthEndBeta); R_FlushConsole();
       Rf_error("DeriveAlterProb: Not Reading Error correctly.\n");
     }
     result = fread(bufferI, sizeof(int), LengthEndBeta, fCodaIFile);
     if (result < LengthEndBeta) {
       Rprintf("DeriveAlterProb, no, at end OnB=%d/%d, TOnB=%d/%d, LengthEndBeta = %d, but for int read result = %d\n",
         OnB, GetLocBuffer,TOnB, lSizeLoc, LengthEndBeta); R_FlushConsole();
       Rf_error("DeriveAlterProb: Not Reading Error correctly.\n");
     }
     if (NumGroups > 0 && sOnTau != NULL && Rf_length(sOnTau) >= 1) {
       result = fread(&IdEndTau, sizeof(int),  1, fCodaiTFile);
       result = fread(&InLengthEndTau, sizeof(double),  1, fCodadTFile);
       if (fabs(InLengthEndTau - round(InLengthEndTau)) >= .1) {
         Rprintf("DeriveAlterProb, At end OnB=%d/%d, TOnB=%d/%d we got LengthEndTau = %d, not good should be count of betas. \n", OnB, GetLocBuffer,
           TOnB, lSizeLoc, InLengthEndTau); R_FlushConsole();
         Rprintf("IdEndBeta is %d too. \n", IdEndTau);
        Rf_error("DeriveAlterProb: Not reading Error correctly.\n");
       }
       if (IdEndTau > 0) {
         Rprintf("DeriveAlterProb, At end OnB=%d/%d, TOnB=%d/%d we got LengthEndTau = %d, not good should be count of betas. \n", OnB, GetLocBuffer,
           TOnB, lSizeLoc, InLengthEndTau); R_FlushConsole();
        Rprintf("IdEndBeta is %d too. \n", IdEndBeta);
        Rf_error("DeriveAlterProb: Not reading Error correctly.\n");
       }
       LengthEndTau = (int)(fabs(round(InLengthEndTau)));
       OnBetaBuff = 0; 
       result = fread(bufferdT, sizeof(double), LengthEndTau, fCodadTFile);
       if (result < LengthEndTau) {
         Rprintf("DeriveAlterProb, no, at end OnB=%d/%d, TOnB=%d/%d, LengthEndTau = %d, but for double read result = %d\n",
           OnB, GetLocBuffer,TOnB, lSizeLoc, LengthEndTau); R_FlushConsole();
         Rf_error("deriveAlterProb: Not Reading Error correctly.\n");
       }
       result = fread(bufferiT, sizeof(int), LengthEndTau, fCodaiTFile);
       if (result < LengthEndTau) {
         Rprintf("DeriveAlterProb, no, at end OnB=%d/%d, TOnB=%d/%d, LengthEndTau = %d, but for int read result = %d\n",
           OnB, GetLocBuffer,TOnB, lSizeLoc, LengthEndTau); R_FlushConsole();
         Rf_error("deriveAlterProb: Not Reading Error correctly.\n");
       }
     }
     if (bufferdT != NULL && NumGroups > 0 && sOnTau != NULL && !Rf_isNull(sOnTau)) {
        NewProb = CalcNewProb(LengthEndBeta, bufferD, bufferI, 
          LengthEndTau,  bufferdT, bufferiT, 
          AlterWeightdfRobit, AlterWeightdfTNoise, PiA1, PiA2, EY,Sigma);
     } else {
        NewProb = CalcNewProb(LengthEndBeta, bufferD, bufferI, 
          0,  NULL, NULL, 
          AlterWeightdfRobit, AlterWeightdfTNoise, PiA1, PiA2, EY,Sigma);      
     }
     AlterWeightBuffer[LengthWrittenAlterWeightBuffer] = 
        NewProb - OldProb / AlterWeightTemperature;
     LengthWrittenAlterWeightBuffer++;  LengthTotalWrittenAlterWeightBuffer++;
     OnB++;  TOnB++;  OnT++; TOnT++;  OnPiABuff++;  OnProbBuff++; OnSigBuff++;
     if (Verbose >= 3) {
       Rprintf("DeriveAlterProb: Finished Bonus Loop, OnB=%d/%d, OnBetaBuff=%d/%d \n", OnB, GetLocBuffer,
         OnBetaBuff, GetBetaBuffer);
     }
    } 
    if (Verbose >= 3) {
      Rprintf("DeriveAlterProb: Finished: a Loop of Buffer, OnB=%d/%d, TOnB=%d/%d, OnBetaBuff=%d/%d \n",
        OnB, GetLocBuffer, TOnB, lSizeLoc, OnBetaBuff, GetBetaBuffer); 
      R_FlushConsole();    
    } 
  }
  if (Verbose >= 3) {
    int ACount = 0;  ACount = lSizeLoc % bufferiLocLength;
    if (ACount > 0) {ACount = 1;}
    Rprintf("DeriveAlterProb: We just finished the Read Blocks = %d/%d\n",
      ReadBlocks,  lSizeLoc / bufferiLocLength + ACount);
    R_FlushConsole();
  }
  if (GetBetaBuffer == 0) {
    ReadLastLocBeta = 1;
  }
 }
  if (Verbose >= 3) {
    Rprintf("DAP: We're nearly at end, flush Weight Buffer %d/%d to file!\n",
      LengthWrittenAlterWeightBuffer, LengthAlterWeightBuffer);
    R_FlushConsole();
  }
  if (LengthWrittenAlterWeightBuffer >=  1) {
    if (NewAlterWeightBufferWrite == 1) {
      fAlterWeightFile =  
        fopen( CHAR(STRING_ELT(RsAlterWeightFile->asSexp(), 0)), "wb");
      NewAlterWeightBufferWrite = 0;
    } else {
      fAlterWeightFile =  
        fopen( CHAR(STRING_ELT(RsAlterWeightFile->asSexp(), 0)), "ab");
    }
    if (fAlterWeightFile == NULL) {
      Rprintf("DAP: bad error because fAlterWeightFile didn't open");
    }
    TotalOpenedFiles++;
    fwrite( AlterWeightBuffer, sizeof(double), 
      LengthWrittenAlterWeightBuffer, fAlterWeightFile );
    ffclose(fAlterWeightFile, "fAlterWeightFile");
    TotalClosedFiles++;
    LengthWrittenAlterWeightBuffer = 0;
  }
  if (OnBetaBuff < GetBetaBuffer) {
    if (Verbose >= 4) {
      Rprintf("DAP: At end, OnBetaBuff = %d/%d, TOnBetaBuff=%d/%d\n",
        OnBetaBuff, GetBetaBuffer, TOnBetaBuff, lSizeD);
    }
  }
  if (Verbose >= 2) {
    Rprintf("DAP: at End, OnB=%d/%d, TOnB=%d/%d, TOnBetaBuff = %d/%d\n",
      OnB, GetLocBuffer, TOnB, lSizeLoc, TOnBetaBuff, lSizeD);
  }
  if (Verbose >= 2) {
    Rprintf("DefineAlterProbability: Made it all the way to end successfuly!\n");
    R_FlushConsole();
  }  
  LaunchEnd( 2);

  return(1);
}
 //End Of Extern!                   

  #ifndef ReDeDouble
    #define ReDeDouble( aIn )  aIn > 10 ?  1.0 - exp(-aIn) : aIn < -10 ? exp(aIn) : exp(aIn) / (1.0 + exp(aIn))
  #endif
 SEXP BayesSpikeCL::BSDeDouble(SEXP saIn) {
   if (Rf_isNull(saIn) || Rf_length(saIn) <= 0 || !(Rf_isReal(saIn) || Rf_isInteger(saIn))) {
     Rf_error("BSDeDouble: saIn is bad!\n");
   } 
   double aIn = 0.0;
   if (Rf_isInteger(saIn)) { aIn = (double) INTEGER(saIn)[0]; }
   if (Rf_isReal(saIn))    { aIn = (double) REAL(saIn)[0];    }
   SEXP sOut = R_NilValue;
   double aAns =  0.0;
   aAns = ReDeDouble( aIn );
   Rf_protect(sOut = Rf_allocVector(REALSXP, 1));
   REAL(sOut)[0] = aAns;
   Rf_unprotect(1);
   return(sOut);
 }
 void GetRegionNeed(int aPoint, int AWidth, int MaxSpace, int *MyUp, int *MyDown) {
   int NUp = 0;  int NDown = 0;   //int NNeed;
   if (MaxSpace < AWidth) {
     MyDown[0] = 0;  MyUp[0] = MaxSpace -1;
     return;
   }
   if (AWidth <= 1) { return; }
   if (AWidth % 2 == 1) {
     NUp = (AWidth -1 ) /2; NDown = NUp;
   } else {
     NUp = (AWidth) / 2;  NDown = NUp -1;
   }
   if (aPoint + NUp >= MaxSpace) {
     NUp = MaxSpace-1 - aPoint;
     NDown = AWidth - NUp - 1;
   }
   if (aPoint - NDown < 0) {
     NDown = aPoint;
     NUp = AWidth - NDown - 1;
   }
   if (aPoint + NUp >= MaxSpace) {
     MyUp[0] = MaxSpace-1;
   } else { MyUp[0] = aPoint + NUp; }
   if (aPoint - NDown < 0) {
     MyDown[0] = 0;
   } else {  MyDown[0] = aPoint - NDown; }
   return;
 }
 SEXP BayesSpikeCL::ShowRegionNeed(SEXP aPoint, SEXP AWidth, SEXP MaxSpace) {
   SEXP sOn = R_NilValue;
   if (Rf_isNull(aPoint) || Rf_isNull(AWidth) || Rf_isNull(MaxSpace) ||
    !(Rf_isInteger(aPoint) || Rf_isReal(aPoint)) ||
    !(Rf_isInteger(aPoint) || Rf_isReal(aPoint)) ||
    !(Rf_isInteger(aPoint) || Rf_isReal(aPoint)) ) {
    Rf_error("ShowRegionNeed: No, aPoint = %d, AWidth = %d, MaxSpace=%d\n", aPoint, AWidth, MaxSpace);
   }
   int iaPoint = GetFirstInteger(aPoint);
   int iAWidth = GetFirstInteger(AWidth);
   int iMaxSpace = GetFirstInteger(MaxSpace);
   if (MaxSpace <= 0) {
     Rf_error("ShowRegionNeed: MaxSpace = %d, not enough to window!\n", MaxSpace);
   }
   if (iaPoint < 0 || iaPoint >= p) {
     Rf_error("ShowRegionNeed: No, iaPoint = %d, but p = %d!\n", iaPoint, p);
   }
   if (iAWidth <= 1) {
     Rf_error("AWidth = %d \n", iAWidth);
   }

   Rf_protect(sOn = Rf_allocVector(INTSXP, 2));                      
   GetRegionNeed(iaPoint, iAWidth, iMaxSpace, INTEGER(sOn),  INTEGER(sOn)+1);
   Rf_unprotect(1);
   return(sOn);
 }
 int BayesSpikeCL::AddToRunProbRegionVector() {
   if (RunProbRegionVector == NULL) {
      if (Verbose >= 2) {
        Rprintf("BayesSpikeGibbs.cpp()::AddToRunProbRegionVector, NULL setup, we shouldn't be doing this, though RegionWidth = %d\n", RegionWidth);
      }
      return(-1);
   }
   if (RegionWidth <= 1) {
      Rf_error("Error: AddToRunProbRegionVector, No, RegionWidth = %d, but RunProbRegionVector != NULL!\n");
   }
   //int CNTotal = 0; 
   int NUp = 0;  int NDown = 0; double MyScore;
   int TFixed = p;  double iAIn = 0.0;
   int TotalAll = 0;
   if (iFirstRandom >= 0 && iFirstRandom < p) {
     TFixed = iFirstRandom; TotalAll = iFirstRandom;
     if (tauEndList != NULL && !Rf_isNull(tauEndList) && Rf_length(tauEndList) >= 1) {
       TotalAll = iFirstRandom + Rf_length(tauEndList);
     } else {
       Rf_error("AddToRunProbRegionVector, Why is iFirstRandom=%d, p=%d, but tauEndList is bad?\n",
         iFirstRandom, p);
     }
   } else if ((iFirstRandom < 0 || iFirstRandom >= p+1) && 
     tauEndList != NULL && !Rf_isNull(tauEndList) &&
     Rf_length(tauEndList) >= 1) {
     TFixed = 0;  TotalAll = Rf_length(tauEndList);
   } else {
     if (tauEndList == NULL || Rf_isNull(tauEndList) || Rf_length(tauEndList) <= 0) {
       TFixed = p;  TotalAll = p;
     } else {
       Rf_error("AddToRunProbRegionVector, Why is iFirstRandom=%d, p=%d, but tauEndList is bad?\n",
         iFirstRandom, p);
     }
   }
   if (TFixed > 0) {
     for (int ii = 0; ii < TFixed; ii++) {
       GetRegionNeed(ii, RegionWidth, TFixed, &NUp, &NDown);
       MyScore = 0.0;
       if (NUp >= TFixed || NDown < 0) {
         Rf_error("Get Region Need, no way, ii=%d, RegionWidth=%d, TFixed=%d, NUp=%d, NDown=%d!\n",
           ii, RegionWidth, TFixed, NUp, NDown);
       } 
       for (int jj = NDown; jj <= NUp; jj++) {
         if (BFixed[jj] == 1.0) {
           MyScore = 1.0; break;
         }                                 
         iAIn =  ReDeDouble(  (ProbFixed[ii]) );
         if (iAIn > MyScore) {
           MyScore = iAIn;
         }
       }
       RunProbRegionVector[ii] += MyScore;
     }
   }
   int OnPos = TFixed;
   if (tauEndList != NULL && !Rf_isNull(tauEndList) && Rf_length(tauEndList) >= 1 &&
      sOnTau != NULL && !Rf_isNull(sOnTau) && Rf_length(sOnTau) >= 1) {
      for (int ii = 0; ii < Rf_length(sOnTau); ii++) {
        GetRegionNeed(ii, RegionWidth, Rf_length(sOnTau), &NUp, &NDown);
        MyScore = 0.0;
        if (NDown < 0 || NUp >= Rf_length(sOnTau)) {
          Rf_error("Error: AddRunProbRegion, if ii=%d, RegionWidth=%d, length Tau=%d, NUp = %d, NDown=%d!\n",
            ii, RegionWidth, Rf_length(sOnTau), NUp, NDown); 
        }
        for (int jj = NDown; jj <= NUp; jj++) {
          if (REAL(sOnTau)[jj]  > 0.0) {
            MyScore = 1.0; break;
          }                                 
          iAIn =  ReDeDouble(  (ProbTau[ii]) );
          if (iAIn > MyScore) {
            MyScore = iAIn;
          }
        }
        if (OnPos >= TotalAll) {
          Rf_error("Error: AddRunProbRegion, OnPos=%d, TotalAll=%d, TFixed=%d, length Tau=%d, iFirstRandom=%d!\n",
            OnPos, TotalAll, TFixed, Rf_length(sOnTau), iFirstRandom);
        }
        RunProbRegionVector[OnPos] += MyScore;    OnPos++;
      }                                                     
   }
   if (Verbose >= 4) {
     Rprintf("BayesSpikeGibbs.cpp::AddToRunProbRegionVector() Finish. \n");
     R_FlushConsole();
   }
   return(1);
 }