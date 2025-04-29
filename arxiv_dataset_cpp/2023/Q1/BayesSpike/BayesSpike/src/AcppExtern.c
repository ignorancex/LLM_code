/* ========================================================================== */
/*                                                                            */
/*   AcppExtern.c                                                             */
/*   (c) 2014 Alan Lenarcic                                                   */
/*                                                                            */
/*   Description:                                                             */
/*     Code attempting to duplicate old Rcpp functionality                    */
/*   When Package Rcpp deprecated the Object function "asSexp()", which the   */
/*   BayesSpike code relied upon to transfer SEXP vectors into LAPACK, a      */
/*   a replacement for the Rcpp class "RObject()" needed to be devised, that  */
/*   would replace the Rcpp old RObject wrapper for R's S-Expressions (which  */
/*   protected R objects from the Garbage Collector in an invisible list)     */
/*                                                                            */
/*   We still use Rcpp in BayesSpike, especially the very useful MODULES      */
/*   interface, however, the need to use R's builtin LAPACK as opposed to     */
/*   Armadillo() coding forced the creation of the Acpp.c code.               */ 
/*                                                                            */
/*   AcppExtern.c contains Extern pointing functions that can write and       */
/*    Access the C kernel                                                     */
/*                                                                            */
/* ========================================================================== */


#ifndef ACPPCONFIGUREME
  #include "AcppConfigureMe.h"
  #define ACPPCONFIGUREME 0
#endif


#define UpLo 'L'
#define TransWant 'N'
#ifndef RNeeds 
 #define RNeeds 0
 #include <R.h>
 #include <Rinternals.h>
 #include <Rdefines.h>
#endif


#ifndef RDYNLOADH
  #include <R_ext/Rdynload.h>
  #define RDYNLOADH 0
#endif


SEXP LockMeIn(SEXP aList, SEXP aOb) {
  SEXP gVI = R_NilValue;
  //Rprintf(" -- ACppExtern.c: Starting LockMeIn \n"); R_FlushConsole();
  //Rprintf(" -- aList is length %d\n", Rf_length(aList)); R_FlushConsole();
  if (Rf_isSymbol(aList)) {
    Rprintf("Acpp::AcppExtern.c:::LockMeIn:: Error aList is a symbol\n");
    if (Rf_isNull(aList)) {
       Rprintf("Acpp::AcppExtern.c:::LockMeIn:: Error aList is a NULL\n");
    }
    Rf_error("Acpp::AcppExtern.c:::LockMeIn:: This ain't good. \n");
  }
  if (Rf_length(aList) <= 0) {
    Rf_error("Acpp::AcppExtern:::LockMeIn:: not if a List is Length %d\n",
      Rf_length(aList));
  }
  if (Rf_isEnvironment(aList)) {
    Rf_error("Acpp::AcppExtern:::LockMeIn:: Not good because aList is an environment. \n");
  }
  for (int ii = 0; ii < Rf_length(aList); ii++) {
    gVI = VECTOR_ELT(aList, ii);
    if (Rf_isNull(gVI)) {
      //Rprintf(" -- ACppExtern.c::LockMeIn(): we now are setting into aList at postition %d \n", ii); R_FlushConsole();
      SET_VECTOR_ELT(aList, ii, aOb);
      SEXP iS = Rf_allocVector(INTSXP, 1);
      INTEGER(iS)[0] = ii;
      return(iS);
    }
  }
  SEXP iS = Rf_allocVector(INTSXP, 1);
  INTEGER(iS)[0] = -1;
  return(iS);
}
                  
SEXP GetMyPackageNamespace() {
  return(R_FindNamespace(Rf_mkString(PCKGNAME))); 
}
SEXP DoubleUpaList(SEXP aList) {
  SEXP newList = R_NilValue;
  Rf_protect(newList = Rf_allocVector(VECSXP, Rf_length(aList) *2));
  if (Rf_isNull(newList)) {
    Rf_error("DoubleUpaList: Error, we could not allocnewList");
  }
  int ii;
  for (ii = 0; ii < Rf_length(aList); ii++) {
    SET_VECTOR_ELT(newList, ii, VECTOR_ELT(aList, ii));
  }
  for (ii = Rf_length(aList); ii < Rf_length(newList); ii++) {
    SET_VECTOR_ELT(newList, ii, R_NilValue);
  }
  Rf_unprotect(1);
  return(newList);
}
SEXP KillFromaList(SEXP aList, SEXP skInt, SEXP sEnvObject) {
  int kInt = -1;
  if (Rf_isNull(skInt) || Rf_length(skInt) <= 0) {
    Rf_error("Kill From a List, skInt is NULL length. \n");
  }
  if (Rf_isReal(skInt)) { kInt = (int) REAL(skInt)[0];
  } else if (Rf_isInteger(skInt)) { kInt = INTEGER(skInt)[0]; }
  if (kInt < 0) {
    Rf_error("Error:KillFromaList, skInt not supplied well. \n");
  }
  if (Rf_isNull(aList) || Rf_length(aList) <= 0) {
    Rf_error("Error:KillFromaList, aList is NULL length!\n ");
  }
  if (Rf_length(aList) <= kInt) {
    Rf_error("Error: KillFromaList, kInt=%d but length aList = %d\n",
      kInt, Rf_length(aList));
  }
  if (Rf_isSymbol(aList)) {
    Rprintf(" -- KillFromaList, this will fail because aList is a symbol.\n");
    R_FlushConsole();
    return(R_NilValue);
  }
  //Rprintf("--KillFromaList: Setting Now the aList[%d] to R_NilValue \n", kInt);
  R_FlushConsole();
  SET_VECTOR_ELT(aList, kInt, R_NilValue);
  if (Rf_isNull(sEnvObject)) {
    Rprintf(" -- sEnvObject: Provided is not NULL!\n"); R_FlushConsole();
    return(R_NilValue);
  }
  SEXP aList2 =  Rf_findVarInFrame( sEnvObject, Rf_install(".AcppList"));
  if (!Rf_isNull(aList2) && Rf_length(aList2) > kInt) {
    SET_VECTOR_ELT(aList2, kInt, R_NilValue);
  }
  return(skInt);
}

R_CallMethodDef callMethods[] = {
  {"LockMeIn", (DL_FUNC) &LockMeIn, 2},
  {"DoubleUpaList", (DL_FUNC) &DoubleUpaList, 1},
  {"KillFromaList", (DL_FUNC) & KillFromaList, 3},
  {NULL, NULL, 0}
};

void
R_init_Acpp(DllInfo *info)
{
R_registerRoutines(info, NULL, callMethods, NULL, NULL);
}
