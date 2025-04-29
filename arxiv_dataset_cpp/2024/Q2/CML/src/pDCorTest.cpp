#include "pDCorTest.h"

// [[Rcpp::export]]
arma::vec matrix_to_string (arma::mat sep_vectors)
{
  int row_n = sep_vectors.n_rows;
  int col_n = sep_vectors.n_cols;
  arma::vec con_string(row_n);
  
  for(int i = 0; i < row_n; i++)
  {
    double counter = 0;
    for(int j = 0; j < col_n; j++)
    {
      counter = counter + sep_vectors(i,j) * pow(10,col_n - j - 1);
      
    }
    con_string(i) = counter;
  }
  return con_string;
}

// [[Rcpp::export]]
double get_G2_one(arma::vec A, arma::vec B, int tot_Au_size, int tot_Bu_size)
{
  //Contingency Table
  arma::vec A_uniq = unique(A);
  arma::vec B_uniq = unique(B);
  
  int Au_size = A_uniq.size();
  int Bu_size = B_uniq.size();
  
  int A_size = A.size();
  
  //Counting each combination
  arma::mat O (tot_Au_size, tot_Bu_size);
  
  for (int m = 0; m < Au_size; m ++)
  {
    for (int n = 0; n < Bu_size; n ++)
    {
      for (int i = 0; i < A_size; i ++)
      {
        if((A[i] == A_uniq[m]) & (B[i] == B_uniq[n]))
        {
          O (m, n) ++;
        }
      }
    }
  }
  
  
  //Expected counts
  arma::rowvec rowz = arma::sum(O, 0);
  arma::colvec colz= arma::sum(O, 1);
  arma::mat E =  colz * rowz / arma::accu(O);
  
  
  //G2 Adding, accounting for 0 observed counts
  double G2 = 0;
  
  for(int m = 0; m < tot_Au_size; m ++)
  {
    for(int n = 0; n < tot_Bu_size; n ++)
    {
      if(O(m, n) != 0)
      {
        G2 = G2 + O(m, n)*(log(O(m, n)/E(m, n))); 
      }
    }
  }
  
  return 2*G2;
}


// [[Rcpp::export]]
double get_G2_all(arma::vec A, arma::vec B, arma::vec S)
{
  arma::vec A_uniq = unique(A);
  arma::vec B_uniq = unique(B);
  arma::vec S_uniq = unique(S);
  
  int tot_Au_size = A_uniq.size();
  int tot_Bu_size = B_uniq.size();
  int Su_size = S_uniq.size();
  int S_size = S.size();
  
  arma::vec G_squares (Su_size);
  
  //Calculate for each level of S
  for (int i = 0; i < Su_size; i++)
  {
    
    NumericVector A_sub;
    NumericVector B_sub;
    
    for (int j = 0; j < S_size; j++)
    {
      if (S[j] == S_uniq[i])
      {
        A_sub.push_back(A[j]);
        B_sub.push_back(B[j]);
      }
    }
    G_squares[i] = get_G2_one(A_sub, B_sub, tot_Au_size, tot_Bu_size);
  }
  
  double G_stat = arma::sum(G_squares);
  return G_stat;
}

// [[Rcpp::export]]
List condInttestdis(arma::mat df, const size_t &i,const size_t &j,
                    const arma::uvec &k, const double &signif_level)
{
  size_t k_size = k.size();
  
  //Setting up vectors and conditioning set
  arma::vec A = df.col(i);
  arma::vec B = df.col(j);
  arma::mat S_m(df.n_rows, k_size);
  
  arma::vec A_u = unique(A);
  arma::vec B_u = unique(A);
  
  int S_df = 1;
  
  //Looping through to get the elements and df of conditioning set
  for (size_t j = 0; j < k_size; j++)
  {
    arma::vec con_col = df.col(k(j));
    S_m.col(j) = con_col;
    arma::vec con_col_u = unique(con_col);
    S_df = S_df * con_col_u.size();
  }
  
  //Convert to one vector to examine the different vector levels
  arma::vec S = matrix_to_string(S_m);
  
  //Calculating the relevant info
  double statistic = get_G2_all(A, B, S);
  int dof = (A_u.size() - 1) * (B_u.size() - 1) * S_df;
  double cutoff = R::qchisq(1-signif_level,dof,true,false);
  bool accept_H0 = std::abs(statistic) <= cutoff;
  double pval = 1 - R::pchisq(statistic, dof, true,false);
  return List::create(
    _["result"]=accept_H0,
    _["statistic"]=statistic,
    _["pval"]=pval
  );
}
