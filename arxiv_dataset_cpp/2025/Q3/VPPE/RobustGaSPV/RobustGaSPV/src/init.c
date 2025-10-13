#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME: 
   Check these declarations against the C/Fortran source code.
*/

/* .Call calls */
extern SEXP _RobustGaSPV_construct_ppgasp(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_construct_rgasp(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_euclidean_distance(SEXP, SEXP);
extern SEXP _RobustGaSPV_generate_predictive_mean_cov(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_log_approx_ref_prior(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_log_approx_ref_prior_deriv(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_log_marginal_lik(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_log_marginal_lik_deriv(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_log_marginal_lik_deriv_ppgasp(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_log_marginal_lik_ppgasp(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_log_profile_lik(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_log_profile_lik_deriv(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_log_profile_lik_deriv_ppgasp(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_log_profile_lik_ppgasp(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_log_ref_marginal_post(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_log_ref_marginal_post_ppgasp(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_matern_3_2_deriv(SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_matern_3_2_funct(SEXP, SEXP);
extern SEXP _RobustGaSPV_matern_5_2_deriv(SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_matern_5_2_funct(SEXP, SEXP);
extern SEXP _RobustGaSPV_periodic_exp_deriv(SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_periodic_exp_funct(SEXP, SEXP);
extern SEXP _RobustGaSPV_periodic_gauss_deriv(SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_periodic_gauss_funct(SEXP, SEXP);
extern SEXP _RobustGaSPV_pow_exp_deriv(SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_pow_exp_funct(SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_pred_ppgasp(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_pred_rgasp(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_separable_kernel(SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_separable_multi_kernel(SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_separable_multi_kernel_pred_periodic(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _RobustGaSPV_test_const_column(SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"_RobustGaSPV_construct_ppgasp",                     (DL_FUNC) &_RobustGaSPV_construct_ppgasp,                      8},
    {"_RobustGaSPV_construct_rgasp",                      (DL_FUNC) &_RobustGaSPV_construct_rgasp,                       8},
    {"_RobustGaSPV_euclidean_distance",                   (DL_FUNC) &_RobustGaSPV_euclidean_distance,                    2},
    {"_RobustGaSPV_generate_predictive_mean_cov",         (DL_FUNC) &_RobustGaSPV_generate_predictive_mean_cov,         18},
    {"_RobustGaSPV_log_approx_ref_prior",                 (DL_FUNC) &_RobustGaSPV_log_approx_ref_prior,                  6},
    {"_RobustGaSPV_log_approx_ref_prior_deriv",           (DL_FUNC) &_RobustGaSPV_log_approx_ref_prior_deriv,            6},
    {"_RobustGaSPV_log_marginal_lik",                     (DL_FUNC) &_RobustGaSPV_log_marginal_lik,                      9},
    {"_RobustGaSPV_log_marginal_lik_deriv",               (DL_FUNC) &_RobustGaSPV_log_marginal_lik_deriv,                9},
    {"_RobustGaSPV_log_marginal_lik_deriv_ppgasp",        (DL_FUNC) &_RobustGaSPV_log_marginal_lik_deriv_ppgasp,         9},
    {"_RobustGaSPV_log_marginal_lik_ppgasp",              (DL_FUNC) &_RobustGaSPV_log_marginal_lik_ppgasp,               9},
    {"_RobustGaSPV_log_profile_lik",                      (DL_FUNC) &_RobustGaSPV_log_profile_lik,                       9},
    {"_RobustGaSPV_log_profile_lik_deriv",                (DL_FUNC) &_RobustGaSPV_log_profile_lik_deriv,                 9},
    {"_RobustGaSPV_log_profile_lik_deriv_ppgasp",         (DL_FUNC) &_RobustGaSPV_log_profile_lik_deriv_ppgasp,          9},
    {"_RobustGaSPV_log_profile_lik_ppgasp",               (DL_FUNC) &_RobustGaSPV_log_profile_lik_ppgasp,                9},
    {"_RobustGaSPV_log_ref_marginal_post",                (DL_FUNC) &_RobustGaSPV_log_ref_marginal_post,                 9},
    {"_RobustGaSPV_log_ref_marginal_post_ppgasp",         (DL_FUNC) &_RobustGaSPV_log_ref_marginal_post_ppgasp,          9},
    {"_RobustGaSPV_matern_3_2_deriv",                     (DL_FUNC) &_RobustGaSPV_matern_3_2_deriv,                      3},
    {"_RobustGaSPV_matern_3_2_funct",                     (DL_FUNC) &_RobustGaSPV_matern_3_2_funct,                      2},
    {"_RobustGaSPV_matern_5_2_deriv",                     (DL_FUNC) &_RobustGaSPV_matern_5_2_deriv,                      3},
    {"_RobustGaSPV_matern_5_2_funct",                     (DL_FUNC) &_RobustGaSPV_matern_5_2_funct,                      2},
    {"_RobustGaSPV_periodic_exp_deriv",                   (DL_FUNC) &_RobustGaSPV_periodic_exp_deriv,                    3},
    {"_RobustGaSPV_periodic_exp_funct",                   (DL_FUNC) &_RobustGaSPV_periodic_exp_funct,                    2},
    {"_RobustGaSPV_periodic_gauss_deriv",                 (DL_FUNC) &_RobustGaSPV_periodic_gauss_deriv,                  3},
    {"_RobustGaSPV_periodic_gauss_funct",                 (DL_FUNC) &_RobustGaSPV_periodic_gauss_funct,                  2},
    {"_RobustGaSPV_pow_exp_deriv",                        (DL_FUNC) &_RobustGaSPV_pow_exp_deriv,                         4},
    {"_RobustGaSPV_pow_exp_funct",                        (DL_FUNC) &_RobustGaSPV_pow_exp_funct,                         3},
    {"_RobustGaSPV_pred_ppgasp",                          (DL_FUNC) &_RobustGaSPV_pred_ppgasp,                          19},
    {"_RobustGaSPV_pred_rgasp",                           (DL_FUNC) &_RobustGaSPV_pred_rgasp,                           19},
    {"_RobustGaSPV_separable_kernel",                     (DL_FUNC) &_RobustGaSPV_separable_kernel,                      4},
    {"_RobustGaSPV_separable_multi_kernel",               (DL_FUNC) &_RobustGaSPV_separable_multi_kernel,                4},
    {"_RobustGaSPV_separable_multi_kernel_pred_periodic", (DL_FUNC) &_RobustGaSPV_separable_multi_kernel_pred_periodic,  5},
    {"_RobustGaSPV_test_const_column",                    (DL_FUNC) &_RobustGaSPV_test_const_column,                     1},
    {NULL, NULL, 0}
};

void R_init_RobustGaSPV(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
