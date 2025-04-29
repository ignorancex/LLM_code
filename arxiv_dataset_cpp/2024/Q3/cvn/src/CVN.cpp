#include <Rcpp.h>
using namespace Rcpp;

//' Generalized LASSO
//' 
//' Solving Generalized LASSO with fixed \eqn{\lambda = 1}
//' Solves efficiently the generalized LASSO problem of the form
//' \deqn{
//'   \hat{\beta} = \text{argmin } \frac{1}{2} || y - \beta ||_2^2 + ||D\beta||_1
//' }
//' where \eqn{\beta} and \eqn{y} are \eqn{m}-dimensional vectors and
//' \eqn{D} is a \eqn{(c \times m)}-matrix where \eqn{c \geq m}.
//' We solve this optimization problem using an adaption of the ADMM
//' algorithm presented in Zhu (2017).
//'
//' @param Y The \eqn{y} vector of length \eqn{m}
//' @param W The weight matrix \eqn{W} of dimensions \eqn{m x m}
//' @param m The number of graphs
//' @param eta1 Equals \eqn{\lambda_1 / rho}
//' @param eta2 Equals \eqn{\lambda_2 / rho}
//' @param a Value added to the diagonal of \eqn{-D'D} so that
//'          the matrix is positive definite, see
//'          \code{\link{matrix_A_inner_ADMM}}
//' @param rho The ADMM's parameter
//' @param max_iter Maximum number of iterations
//' @param eps Stopping criterion. If differences
//'            are smaller than \eqn{\epsilon}, algorithm
//'            is halted
//' @param truncate Values below \code{truncate} are
//'                 set to \code{0}
//' @return The estimated vector \eqn{\hat{\beta}}
//' @author Louis Dijkstra
//' @references
//' Zhu, Y. (2017). An Augmented ADMM Algorithm With Application to the
//' Generalized Lasso Problem. Journal of Computational and Graphical Statistics,
//' 26(1), 195–204. https://doi.org/10.1080/10618600.2015.1114491
//'
//' @seealso \code{\link{genlasso_wrapper}}
// [[Rcpp::export]]
 NumericVector genlassoRcpp(const NumericVector Y,
                            const NumericMatrix W,
                            const int m,
                            const double eta1,
                            const double eta2,
                            double a,
                            const double rho,
                            const int max_iter,
                            const double eps,
                            const double truncate) {
   
   
   // some frequently used constants
   a = rho*a ;
   const double C = 1 / (1 + a) ;
   
   // number of rows in matrix D
   const int c = (m*m + m) / 2 ;
   
   /* Compute y to a double array. I did this to make sure that the C compilation
    * environment does not have an effect */
   double* y = new double[m + 1] ;
   for (int i = 0; i < m; i ++) {
     *(y + i) = (double)(Y[i]) ;
   }
   
   /* initialize vectors for beta-update step in the ADMM */
   double *beta_new = new double[m+1];
   double *beta_old = new double[m+1] ;
   double *delta    = new double[m+1] ;
   
   for (int i = 0; i < m; i ++) {
     *(beta_new + i) = 0;
     *(beta_old + i) = 0;
     *(delta + i)    = 0;
   }
   
   /* initialize vectors for alpha-update step in the ADMM */
   double *alpha_new  = new double[c+1] ;
   double *alpha_old1 = new double[c+1] ;
   double *alpha_old2 = new double[c+1] ;
   double *alpha      = new double[c+1];
   
   for (int i = 0; i < c; i ++) {
     *(alpha_new + i)  = 0 ;
     *(alpha_old1 + i) = 0 ;
     *(alpha_old2 + i) = 0 ;
     *(alpha + i)      = 0 ;
   }
   
   /* Indices used for the alpha-update step */
   int *steps = new int[m-1] ;
   
   *steps = 0 ;
   
   for (int i = 1; i < (m-1); i ++) {
     *(steps + i) = m - i + *(steps + i - 1) ;
   }
   
   int iter = 0 ;    // number of iterations
   double diff = 0 ; // absolute difference between beta^(k+1) and beta^k
   
   /* loop until either the max. no. of iterations are reached
    * or the difference (diff) is smaller then eps
    */
   while (iter < max_iter) {
     
     /* ------- beta-update step ---------*/
     // alpha = 2*alpha_old1 - alpha_old2 ;
     for (int i = 0; i < c; i++) {
       *(alpha + i) = 2*(*(alpha_old1 + i)) - *(alpha_old2 + i) ;
     }
     
     // go over all possible pairs (i,j), same as D^T %*% (2 alpha^(k) - alpha^(k-1))
     for (int i = 0; i < m; i++) {
       *(delta + i) = eta1 * (*(alpha + i)) ;
       
       for (int j = i+1; j < m; j++) {
         *(delta + i) += eta2* W(i,j)*(*(alpha + m + steps[i] + (j - i) - 1)) ;
       }
       
       for (int j = 0; j < i; j++) {
         *(delta + i) -= eta2*W(i,j)*(*(alpha + m + steps[j] - (j - i) - 1)) ;
       }
     }
     
     diff = 0 ;
     for (int i = 0; i < m; i ++) {
       *(beta_new + i) = C*(a*(*(beta_old + i)) + y[i] - *(delta + i)) ;
       diff += fabs(*(beta_new + i) - *(beta_old + i)) ; //abs(beta_new[i] - beta_old[i]) ;
     }
     
     // determine whether converged or not
     if (diff < eps) {
       /* Turn to zero when really close */
       for (int i = 0; i < m; i ++) {
         if (fabs(*(beta_new + i)) < truncate) {
           *(beta_new + i) = 0 ;
         }
       }
       
       // convert results to NumericVector for R
       NumericVector res = NumericVector(beta_new, beta_new + m) ;
       
       // Clean-up ------------
       delete[] beta_old;
       delete[] alpha_old1;
       delete[] alpha_old2;
       delete[] alpha_new;
       delete[] alpha;
       delete[] y;
       delete[] beta_new ;
       
       return(res);
     }
     
     /* --------- alpha update step ----------- */
     for (int i = 0; i < m; i++) {
       *(alpha_new + i) = *(alpha_old1 + i) + rho * eta1 * *(beta_new + i) ;
     }
     
     int k = m;
     // go over all unique pairs (i,j)
     for (int i = 0; i < m-1; i++) {
       for (int j = i+1; j < m; j++) {
         *(alpha_new + k) = *(alpha_old1 + k) + rho * eta2 * W(i,j) * (*(beta_new + i) - *(beta_new + j)) ;
         k ++;
       }
     }
     
     /* Threshold alpha. Must lie in [-1, 1] range */
     for (int i = 0; i < c; i ++) {
       if (alpha_new[i] > 1) {
         alpha_new[i] = 1 ;
       }
       if (alpha_new[i] < -1) {
         alpha_new[i] = -1 ;
       }
     }
     
     /* update beta and alpha for the next iteration step */
     for (int i = 0; i < m; i ++) {
       *(beta_old + i) = *(beta_new + i) ;
     }
     
     for (int i = 0; i < c; i ++) {
       *(alpha_old2 + i) = *(alpha_old1 + i) ;
       *(alpha_old1 + i) = *(alpha_new + i) ;
     }
     
     iter ++;
   }
   
   /* Turn to zero when really close */
   for (int i = 0; i < m; i ++) {
     if (fabs(*(beta_new + i)) < truncate) {
       *(beta_new + i) = 0 ;
     }
   }
   
   // convert results to NumericVector for R
   NumericVector res = NumericVector(beta_new, beta_new + m) ;
   
   // Clean-up ------------
   delete[] beta_old;
   delete[] alpha_old1;
   delete[] alpha_old2;
   delete[] alpha_new;
   delete[] alpha;
   delete[] y;
   delete[] beta_new ;
   delete[] steps;
   
   return(res);
 }
 

//' The \eqn{Z}-update Step
//' 
//' A \code{C} implementation of the \eqn{Z}-update step. We
//' solve a generalized LASSO problem repeatedly for each of the
//' individual edges
//'
//' @param m The number of graphs
//' @param p The number of variables
//' @param Theta A list of matrices with the \eqn{\Theta}-matrices
//' @param Y A list of matrices with the \eqn{Y}-matrices
//' @param W The weight matrix \eqn{W} of dimensions \eqn{m x m}
//' @param eta1 Equals \eqn{\lambda_1 / rho}
//' @param eta2 Equals \eqn{\lambda_2 / rho}
//' @param a Value added to the diagonal of \eqn{-D'D} so that
//'          the matrix is positive definite, see
//'          \code{\link{matrix_A_inner_ADMM}}
//' @param rho The ADMM's parameter
//' @param max_iter Maximum number of iterations
//' @param eps Stopping criterion. If differences
//'            are smaller than \eqn{\epsilon}, algorithm
//'            is halted
//' @param truncate Values below \code{truncate} are
//'                 set to \code{0}
//' @author Louis Dijkstra
//' @return The estimated vector \eqn{\hat{\beta}}
//'
// [[Rcpp::export]]
    Rcpp::ListMatrix updateZRcpp(const int m,
                                 const int p,
                                 Rcpp::ListMatrix Theta,
                                 Rcpp::ListMatrix Y,
                                 const Rcpp::NumericMatrix& W,
                                 const double eta1,
                                 const double eta2,
                                 const double a,
                                 const double rho,
                                 const int max_iter,
                                 const double eps,
                                 const double truncate) {
      
      // indices
      int i,j,k,l ;
      
      /* Initialize variables ----------- */
      
      /* The y-vector and the resulting beta values are
       stored in a vector and a matrix. The rows of beta
       represent the edges. */
      Rcpp::NumericVector y (m) ;
      Rcpp::NumericMatrix beta (p*(p-1)/2, m) ;
      
      // Final results will be stored here
      Rcpp::ListMatrix Z (m);
      
      // set the diagonal of Z
      for (k = 0; k < m; k ++) {
        Rcpp::NumericMatrix A = Theta(k,0) ;
        Rcpp::NumericMatrix B = Y(k,0) ;
        Rcpp::NumericMatrix C (p,p) ;
        
        for (i = 0; i < p; i ++) {
          for (j = 0; j < p; j ++) {
            C(i,j) = 0;
          }
        }
        
        for (i = 0; i < p; i ++) {
          C(i,i) = A(i,i) + B(i,i) ;
        }
        
        Z(k,0) = clone(C) ;
      }
      
      /* Compute betas -------------- */
      l = 0 ; // the index for the edge
      
      // go over all edges
      for (i = 0; i < (p-1); i ++) {
        for (j = i+1; j < p; j ++) {
          
          // go over the different graphs
          for (k = 0; k < m; k ++) {
            // get the matrix Theta and Y and store them in A and B
            Rcpp::NumericMatrix A = Theta(k,0) ;
            Rcpp::NumericMatrix B = Y(k,0) ;
            y[k] = A(i,j) + B(i,j) ;
          }
          
          // determine the beta-vector
          Rcpp::DoubleVector b = genlassoRcpp(y, W, m, eta1, eta2, a, rho, max_iter, eps, truncate) ;
          
          // store the results in the beta matrix
          for (k = 0; k < m; k ++) {
            beta(l, k) = b[k] ;
          }
          
          l ++; // update the edge index
        }
      }
      
      // set the entries of Z
      for (k = 0; k < m; k ++) {
        Rcpp::NumericMatrix A = Z(k,0) ;
        
        l = 0;
        for (i = 0; i < (p-1); i ++) {
          for (j = i+1; j < p; j ++) {
            A(i,j) = beta(l, k) ;
            A(j,i) = beta(l, k) ;
            
            l ++;
          }
        }
        
        Z(k,0) = clone(A) ;
      }
      
      return(Z) ;
    }