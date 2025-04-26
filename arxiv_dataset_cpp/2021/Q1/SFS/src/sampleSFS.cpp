
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;
// Amat creates the transformation matrix given a lambda
// and two different signatures s and smix and the dimension of N.
mat Amat(double lambda, int s, int smix, int N){
  mat A(N,N,fill::eye);
  A(s,s) = 1 - lambda;
  A(smix,s) = lambda;
  return A;
}

// lambdaRange creates the feasible interval of lambda given a matrix 
// P with two columns and E with two rows
vec lambdaRange(mat P, mat E){
  vec p1 = P.col(0);
  vec p2 = P.col(1);
  vec pdiff = p1 - p2;
  uvec neg = find(pdiff < 0);
  
  vec frac1 = p1.elem(neg)/pdiff.elem(neg);
  double lmin = max(frac1);
  if(lmin > 0){
    Rcpp::Rcout << "positive minimum: \n p1 vector " << p1 << " \n p2 vector " << p2 << "\n";
  }
  
  vec e1 = trans(E.row(0));
  vec e2 = trans(E.row(1));
  uvec zero = find(e1 > 0 || e2 > 0);
  
  vec frac2 = e2.elem(zero)/(e1.elem(zero) + e2.elem(zero));
  double lmax = min(frac2);
  if(lmax < 0){
    Rcpp::Rcout << "negative maximum: \n e1 vector " << e1 << " \n e2 vector " << e2 << "\n";
  }
  
  vec res = {lmin, lmax};
  return res;
}

//' @title Finds the SFS from a given solution of P and E
//'
//' @param P One of the factorized two matrices, which will control the stopping criteria
//' @param E The other of the facotrized matrizes
//' @param maxIter The maximum number of iterations
//' @param check Number of iterations between checking to stop
//' @param beta The two shape parameters in the be3ta distribution to sample lambda
//' @param eps Epsilon for the stopping criteria
//'      
//' @return A list of outputs from the sampleSFS function
//' \itemize{
//'         \item avgChangeFinal     - final average change for each entrance in P
//'         \item avgChangeEachCheck - average change for each entrance in P at each check
//'         \item totalIter          - total number of iterations needed for convergence of the average change
//'         \item Pminimum           - minimum for each entry in P
//'         \item Pmaximum           - maximum for each entry in P
//'         \item Eminimum           - minimum for each entry in E
//'         \item Emaximum           - maximum for each entry in E
//'         \item P_lastCheckResults - results of P for the last 'check' iterations 
//'         \item E_lastCheckResults - results of E for the last 'check' iterations 
//'         }
//'
// [[Rcpp::export]]
Rcpp::List sampleSFS(arma::mat P, arma::mat E, int maxIter = 10^5, int check = 1000, double beta = 0.5 , double eps = 1e-10){
  int N = P.n_cols;
  int K = P.n_rows;
  int G = E.n_cols;
  
  mat probs ((check+2)*N, K);
  mat expos ((check+2)*N, G);
  probs( span(0,N-1), span::all) = trans(P);
  probs(span(N, 2*N -1), span::all) = trans(P);
  expos( span(0,N-1), span::all) = E;
  expos(span(N, 2*N -1), span::all) = E;
  
  vec diffres (N);
  
  P.clean(1e-10);
  E.clean(1e-10);
  
  
  vec sig =  vectorise(repmat(regspace(0, N-1), 1, check+2),0);
  vec avgDev(floor(maxIter/check));
  mat A(N,N);
  mat Ainv(N,N);
  vec bound(2);
  mat probdiff(N,K);
  mat probmin(N,K);
  mat probmax(N,K);
  mat exposmin(N,G);
  mat exposmax(N,G);
  double diffnew = 1;
  double diffold = 0;
  int i = 0;
  for(i = 0; i < maxIter; i++){
    for(int s = 0; s < N; s++){
      int smix = s;
      while(smix == s){
        smix = randi(distr_param(0,N-1));
      }
      
      uvec both(2);
      both(0) = s;
      both(1) = smix;
      mat Pmix = P.cols(both);
      mat Emix = E.rows(both);
      bound = lambdaRange(Pmix,Emix);
      vec gvar = randg<vec>(2, distr_param(beta,1.0));
      double x = gvar(0)/sum(gvar);
      double lambda = bound(0)*x + bound(1)*(1-x);
      if(abs(lambda) > 1e-10){
        A = Amat(lambda, s,smix,N);
        Ainv = Amat(-lambda/(1-lambda), s,smix,N);

        P = P*A;
        E = Ainv*E;
        
        P.clean(1e-10);
        E.clean(1e-10);
        
      }
    }

    int iter = i - floor(i/check)*check;
    probs(span(N*(iter + 2),N*(iter + 3)-1), span::all) = trans(P);
    expos(span(N*(iter + 2),N*(iter + 3)-1), span::all) = E;
    
    // Minimum and maximum of Exposures 
    if(i > 0 && iter == 0){
      for(int g=0; g < G; g++){
        vec evec = expos.col(g);
        for(int n=0; n < N; n++){
          vec echoice = evec.elem(find(sig == n));
          exposmin(n,g) = min(echoice);
          exposmax(n,g) = max(echoice);
          
        }
      }
      expos( span(0,N-1), span::all) = exposmin;
      expos(span(N, 2*N -1), span::all) = exposmax;
      
      // Minimum and maximum of Signature probabilities
      for(int k=0; k < K; k++){
        vec pvec = probs.col(k);
        for(int n=0; n < N; n++){
          vec pchoice = pvec.elem(find(sig == n));
          probdiff(n,k) = range(pchoice);
          probmin(n,k) = min(pchoice);
          probmax(n,k) = max(pchoice);
          
        }
      }
      probs( span(0,N-1), span::all) = probmin;
      probs(span(N, 2*N -1), span::all) = probmax;
      diffnew = mean(vectorise(probdiff));
      
      if(diffnew - diffold < eps){
        int idx = i/check - 1;
        avgDev.shed_rows(idx,floor(maxIter/check)-1);
        break;
      } else {
        diffold = diffnew;
        int idx = i/check - 1;
        avgDev(idx) = diffnew;
      }
    }
  }
  Rcpp::List Output = Rcpp::List::create(Rcpp::Named("avgChangeFinal") = diffnew, 
                                         Rcpp::Named("avgChangeEachCheck") = avgDev,
                                         Rcpp::Named("totalIter") = i, 
                                         Rcpp::Named("Pminimum") = probmin, Rcpp::Named("Pmaximum") = probmax, 
                                         Rcpp::Named("Eminimum") = exposmin, Rcpp::Named("Emaximum") = exposmax,
                                         Rcpp::Named("P_lastCheckResults") = probs(span(2*N, probs.n_rows-1), span::all),
                                         Rcpp::Named("E_lastCheckResults") = expos(span(2*N, expos.n_rows-1), span::all));
  return Output;
}