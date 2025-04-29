// #include "r_exports.h"
// #include <RInside.h>
// #include "OrderScoring.h"

// // [[Rcpp::plugins(cpp17)]]

// // [[Rcpp::export]]
// Rcpp::List r_opruner_right(Rcpp::List ret, Rcpp::NumericVector initial_right_orders)
// {
//     OrderScoring scoring = get_score(ret);

//     const auto &[order, log_score, max_n_particles, tot_n_particles] = opruner_right(scoring);

//     Rcpp::List L = Rcpp::List::create(Rcpp::Named("order") = order,
//                                       Rcpp::Named("log_score") = log_score,
//                                       Rcpp::Named("max_n_particles") = max_n_particles,
//                                       Rcpp::Named("tot_n_particles") = tot_n_particles);

//     return (L);
// }