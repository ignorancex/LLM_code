#include <Rcpp.h>

using namespace Rcpp;

const int score_llr = 1;
const int score_chisq = 2;
const int score_diff = 3;
const int score_logratio = 4;
const int score_ratio = 5;

const int return_scores = 1;
// const int return_summary = 2;
// currently not needed as there are only two possibilities

const double one_over_log2 = 1.442695040888963387;

struct ABCD{
  IntegerVector A;
  IntegerVector B;
  IntegerVector C;
  IntegerVector D;
  int NcorpA;
  int NcorpB;
};

ABCD getABCD(IntegerVector ind, 
             IntegerVector start_vek, 
             IntegerVector nterm, 
             IntegerVector freqs, 
             IntegerVector termlist, 
             IntegerVector rowsums, 
             IntegerVector colsums, 
             int ntotal,
             int scoretype) {
  
  int nt = rowsums.size();
  int nind = ind.size();

  ABCD out;
  
  out.A = Rcpp::IntegerVector(nt);
  out.B = Rcpp::IntegerVector(nt);
  out.C = Rcpp::IntegerVector(nt);
  out.D = Rcpp::IntegerVector(nt);
  out.NcorpA = 0;
  out.NcorpB = 0;
  
  for (int i = 0; i < nind; i ++) {
    int currind = ind[i];
    int currnterms = nterm[currind - 1];
    int currstart = start_vek[currind - 1] - 1;
    for (int j = 0; j < currnterms; j ++) {
      int currterm = termlist[currstart + j];
      out.A[currterm - 1] += freqs[currstart + j]; 
    }
    out.NcorpA += colsums[currind - 1];
  }
  
  out.NcorpB = ntotal - out.NcorpA;
  
  if (scoretype == score_llr || scoretype == score_chisq) {
    for (int i = 0; i < nt; i ++) {
      out.B[i] = out.NcorpA - out.A[i];
      out.C[i] = rowsums[i] - out.A[i];
      out.D[i] = ntotal - rowsums[i] - out.B[i];
    }
  } else {
    for (int i = 0; i < nt; i ++) {
      out.C[i] = rowsums[i] - out.A[i]; // don't need C or D for the other
    }
  }
  
  return out;
}

// this is only a wrapper that returns the contingency tables as a list
// not needed internally and currently not exported
// Rcpp::export
// Rcpp::List getABCD2(IntegerVector ind, 
//                     IntegerVector start_vek, 
//                     IntegerVector nterm, 
//                     IntegerVector freqs, 
//                     IntegerVector termlist, 
//                     IntegerVector rowsums, 
//                     IntegerVector colsums, 
//                     int ntotal) {
//   ABCD abcd = getABCD(ind, 
//                       start_vek, 
//                       nterm, 
//                       freqs, 
//                       termlist, 
//                       rowsums, 
//                       colsums, 
//                       ntotal);
//   return Rcpp::List::create(Rcpp::Named("A") = abcd.A,
//                             Rcpp::Named("B") = abcd.B,
//                             Rcpp::Named("C") = abcd.C,
//                             Rcpp::Named("D") = abcd.D,
//                             Rcpp::Named("NcorpA") = abcd.NcorpA,
//                             Rcpp::Named("NcorpB") = abcd.NcorpB);
//   
// }

// [[Rcpp::export]]
NumericVector getScores(IntegerVector ind, 
                        IntegerVector start_vek, 
                        IntegerVector nterm, 
                        IntegerVector freqs, 
                        IntegerVector termlist, 
                        IntegerVector rowsums, 
                        IntegerVector colsums, 
                        int ntotal,
                        int scoretype,
                        double laplace = 0.0) {
  
  NumericVector out;
  
  ABCD abcd = getABCD(ind, 
                      start_vek, 
                      nterm, 
                      freqs, 
                      termlist, 
                      rowsums, 
                      colsums, 
                      ntotal,
                      scoretype);
  
  NumericVector Anum = as<NumericVector>(abcd.A);
  NumericVector Bnum = as<NumericVector>(abcd.B);
  NumericVector Cnum = as<NumericVector>(abcd.C);
  NumericVector Dnum = as<NumericVector>(abcd.D);
  double NcorpA = static_cast<double>(abcd.NcorpA);
  double NcorpB = static_cast<double>(abcd.NcorpB);
  
  if (scoretype == score_llr) {
    out = (-2) * (ifelse(Anum == 0, 0, Anum*log((Anum + Bnum)*(Anum + Cnum)/Anum)) + 
      ifelse(Bnum == 0, 0, Bnum*log((Anum + Bnum)*(Bnum + Dnum)/Bnum)) +
      ifelse(Cnum == 0, 0, Cnum*log((Cnum + Dnum)*(Anum + Cnum)/Cnum)) +
      ifelse(Dnum == 0, 0, Dnum*log((Bnum + Dnum)*(Cnum + Dnum)/Dnum)) - 
      ntotal*log(ntotal)
    );
  } else if (scoretype == score_chisq) {  // this cannot cope with zero column sums, probably...
    out = ifelse(rowsums ==0, 
                 0, 
                 ntotal * (Anum * Dnum - Bnum * Cnum) *  (Anum * Dnum - Bnum * Cnum) / ((Anum + Bnum) * (Anum + Cnum) * (Cnum + Dnum) * (Bnum + Dnum)) ); 
  } else if (scoretype == score_diff) {  
    out = Anum / NcorpA  - Cnum / NcorpB;   
  } else if (scoretype == score_logratio) {  
    out = one_over_log2 * log((Anum + laplace) / (NcorpA  + laplace))  - one_over_log2 *log((Cnum  + laplace)/ (NcorpB  + laplace));   
  } else if (scoretype == score_ratio) {  
    out = ((Anum + laplace) / (NcorpA  + laplace)) / ((Cnum  + laplace)/ (NcorpB  + laplace));   
  }
  
  
  return(out);
}

// [[Rcpp::export]]

NumericMatrix genPerm(IntegerVector ind, 
                      IntegerVector start_vek, 
                      IntegerVector nterm, 
                      IntegerVector freqs, 
                      IntegerVector termlist, 
                      IntegerVector rowsums, 
                      IntegerVector colsums, 
                      int ntotal,
                      int nperm,
                      int output,
                      int scoretype,
                      NumericVector observed,
                      double laplace = 0.0) {
  
  int nt = rowsums.size();
  int ndoc = colsums.size();
  int ndocA = ind.size();
  
  // output sizes depends on what type of output we want
  
  int ncols_out;
  
  if (output == return_scores) 
    ncols_out = nperm;
  else
    ncols_out = 3;
  
  NumericMatrix out(nt, ncols_out);
  
  IntegerVector ind2;
  
  for (int j = 0; j < nperm; j++) {
    ind2 = Rcpp::sample(ndoc, ndocA, FALSE);
    NumericVector G = getScores(ind2, start_vek, nterm, freqs, termlist, rowsums, colsums, ntotal, scoretype, laplace);
    
    // do calculations conditional on whether we want the full results (1) or just a summary (# <, # =,  #>) (2)
    
    if (output == return_scores) {
      for (int i = 0; i < nt; i++) {
        out(i, j) = G[i];
      }
    }
    else{     
      for (int i = 0; i < nt; i++) {
        double scorediff = observed[i] - G[i];
        if (std::isinf(observed[i]) && std::isinf(G[i]) && (observed[i] * G[i] > 0 )) 
          out(i, 1) += 1; // case where both are +Inf or both are -Inf
        else if (std::fabs(scorediff) <= 2 * std::numeric_limits<double>::epsilon()) 
          out(i, 1) += 1;
        else if (scorediff > 0) 
          out(i, 0) += 1;
        else if (scorediff < 0)
          out(i, 2) += 1;
      }
      
    }
    
  }
  
  return out;
}
