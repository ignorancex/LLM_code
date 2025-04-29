#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

SEXP Ccoxcount1(SEXP y2, SEXP strat2) {
  static SEXP(*fun)(SEXP, SEXP) = NULL;
  if (fun == NULL)
    fun = (SEXP(*)(SEXP, SEXP)) R_GetCCallable("survival", "Ccoxcount1");
  return fun(y2, strat2);
}

SEXP Ccoxcount2(SEXP y2, SEXP isort1, SEXP isort2, SEXP strat2) {
  static SEXP(*fun)(SEXP, SEXP, SEXP, SEXP) = NULL;
  if (fun == NULL)
    fun = (SEXP(*)(SEXP, SEXP, SEXP, SEXP)) R_GetCCallable("survival", "Ccoxcount2");
  return fun(y2, isort1, isort2, strat2);
}


