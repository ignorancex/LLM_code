#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

/*
// [[Rcpp::export]]
//double logsumexp(arma::vec x){
//	double xmax;
//	xmax = x.max
//	return(log(sum(exp(x - xmax))) + xmax);
//}
*/


// [[Rcpp::export]]
double log_mix_prior_derivative(double b, double t1, double t2, double p){

	double logv, v;
	
	logv = 0.5 * log(t1) - 0.5 * log(t2) + log(1.0 - p) - log(p) + 0.5*pow(b, 2.0)*(1.0/t1 - 1.0/t2);
	v = exp(logv);
	return( -b/(t1 * (1.0 + v)) - v * b/(t2 * (1.0 + v)) );

}


// [[Rcpp::export]]
double log_prior_mix(double b, double t1, double t2, double p){
	double v1, v2, vmax;
	v1 = log(p) + arma::log_normpdf( b, 0.0, sqrt(t1) );
	v2 = log(1.0 - p) + arma::log_normpdf( b, 0.0, sqrt(t2) );
	vmax = v1;
	if(v1 < v2){
		vmax = v2;
	}
	return( log(exp(v1 - vmax) + exp(v2 - vmax)) + vmax );

}


// [[Rcpp::export]]
List mala_proposal_cpp( arma::mat y, arma::mat X, arma::cube b, IntegerVector z, double tau, arma::vec pr, double nu2 ) {
	int k, i, r, d, npars, acceptance;
	int n = y.n_rows;
	int p = X.n_cols;
	int K = b.n_slices;
	int D_minus_1 = b.n_rows; 
	arma::cube  theta(n, D_minus_1 + 1, K), theta_prop(n,D_minus_1 + 1,K), log_posterior_gradient(D_minus_1, p, K), log_posterior_gradient_prop(D_minus_1, p, K), b_prop(D_minus_1, p, K), prop_mean(D_minus_1, p, K);
	arma::mat FirstDerivative(p, D_minus_1), w(n, K), G(D_minus_1, K);
	arma::vec sum1(D_minus_1 + 1), logpr(K), motorhead(D_minus_1), s(n); 
	double cll, cll_prop, lastcol, xmax, prop_var, prop_mean_old, log_proposalRatio, log_priorRatio, lar; 


	
	acceptance = 0;
	logpr = log(pr);
	npars = p * D_minus_1 * K;
	w.fill(0.0);
	cll = 0.0;
	cll_prop = 0.0;
	sum1(0) = 0.0;
	for(i = 0; i < n; i++){
		w(i, z(i)-1) = 1.0;
		s(i) = y(i, D_minus_1);
		for(d = 0; d < D_minus_1; d++){
			s(i) = s(i) + y(i, d);
		}
	}

//	for(r = 0; r < p; r++){
//		Rcout <<  X(0, r) <<  ", ";
//	}
//	Rcout << std::endl;

//	for(d = 0; d < D_minus_1; d++){
//		for(r = 0; r < p; r++){
//			Rcout << std::fixed;
//			Rcout << std::setprecision(4);
//			Rcout <<  b(d, r, 0) <<  " ";
//		}
//		Rcout << std::endl;
//	}
//	Rcout << std::endl;

	
	
	log_posterior_gradient.fill(0.0); // = - b/nu2;
	for(i = 0; i < n; i++){
		G.fill(0.0);
		for(k = 0; k < K; k++){
//			if(i == 0){Rcout << "G = " <<  std::endl;}	
			for(d = 0; d < D_minus_1; d++){		
				for(r = 0; r < p; r++){
					G(d, k) = G(d, k) + b(d, r, k) * X(i, r);	
				}
//				if(i == 0){
//					Rcout << std::fixed;
//					Rcout << std::setprecision(4);
//					Rcout <<  G(d, k) <<  ", ";
//				}				
				sum1(d + 1) = G(d, k);
			}
			//			if(i == 0){Rcout <<  std::endl;}
			xmax = sum1.max();
			lastcol = log(sum(exp(sum1 - xmax))) + xmax;
//			if(i == 0){Rcout << "LASTCOL = " << lastcol <<  std::endl;}
			for(d = 0; d < D_minus_1; d++){					
//				if(i == 0){Rcout <<  sum1(d) <<  ", ";}
				G(d, k) = G(d, k) - lastcol;
				theta( i, d, k ) = G(d, k);
				cll = cll + w(i,k) * theta(i, d, k) * y(i,d);
				G(d, k) = exp(G(d, k));
//				if(i == 0){Rcout <<  G(d, k) <<  ", ";}
				for(r = 0; r < p; r++){
					log_posterior_gradient(d, r, k) = log_posterior_gradient(d, r, k) + w(i, k) * (y(i, d) - s(i) * G(d, k)) * X(i, r);
				}
			}
//			if(i == 0){Rcout << sum1(D_minus_1) << std::endl;}
//			if(i == 0){Rcout <<  std::endl;}
			theta( i, D_minus_1, k ) = - lastcol;	
//			if(i == 0){Rcout << "theta = " << std::endl;}
//			if(i == 0){
//				for(d = 0; d < D_minus_1 + 1; d++){	
		//			Rcout << theta( i, d, k ) << ", ";
//				}
//			}
			cll = cll + w(i,k) * logpr(k) + w(i,k) * theta(i, D_minus_1, k) * y(i,D_minus_1);
		}
		
//		if(i == 0){
//			Rcout << std::endl;
//		}
	}
//	Rcout << "cll = " << cll <<  std::endl;
	
	prop_var = 2.0 * tau; 
	for(k = 0; k < K; k++){
		for(d = 0; d < D_minus_1; d++){
			for(r = 0; r < p; r++){
				log_posterior_gradient(d, r, k) = log_posterior_gradient(d, r, k) - b(d, r, k)/nu2;
				prop_mean(d, r, k) = b(d, r, k) + tau*log_posterior_gradient(d, r, k); 
				b_prop(d, r, k) = prop_mean(d, r, k) + sqrt(prop_var) * arma::randn(); 
			}
		}
	}
	
	
	
	
	log_posterior_gradient_prop.fill(0.0); //- b_prop/nu2;
	sum1(0) = 0.0;
	for(i = 0; i < n; i++){
		G.fill(0.0);
		for(k = 0; k < K; k++){	
			for(d = 0; d < D_minus_1; d++){		
				for(r = 0; r < p; r++){
					G(d, k) = G(d, k) + b_prop(d, r, k) * X(i, r);	
				}
				sum1(d + 1) = G(d, k);
			}
			xmax = sum1.max();
			lastcol = log(sum(exp(sum1 - xmax))) + xmax;
			for(d = 0; d < D_minus_1; d++){					
				G(d, k) = G(d, k) - lastcol;
				theta_prop( i, d, k ) = G(d, k);
				cll_prop = cll_prop + w(i,k) * theta_prop(i, d, k) * y(i,d);
				G(d, k) = exp(G(d, k));
				for(r = 0; r < p; r++){
					log_posterior_gradient_prop(d, r, k) = log_posterior_gradient_prop(d, r, k) + w(i, k) * (y(i, d) - s(i) * G(d, k)) * X(i, r);
				}
			}
			theta_prop( i, D_minus_1, k ) = - lastcol;	
			cll_prop = cll_prop + w(i,k) * logpr(k) + w(i,k) * theta_prop(i, D_minus_1, k) * y(i,D_minus_1);			
		}
	}

//	Rcout << "cll_prop = " << cll_prop <<  std::endl;
	log_proposalRatio = 0.0;
	log_priorRatio = 0.0;
	for(k = 0; k < K; k++){
		for(d = 0; d < D_minus_1; d++){
			for(r = 0; r < p; r++){
				log_posterior_gradient_prop(d, r, k) = log_posterior_gradient_prop(d, r, k) - b_prop(d, r, k)/nu2;
				prop_mean_old = b_prop(d, r, k) + tau*log_posterior_gradient_prop(d, r, k);				
				log_proposalRatio = log_proposalRatio + pow(b_prop(d, r, k) - prop_mean(d, r, k), 2.0)/(2.0*prop_var) - pow(b(d, r, k) - prop_mean_old, 2.0)/(2.0*prop_var);
				log_priorRatio = log_priorRatio + pow(b(d, r, k), 2.0)/(2.0*nu2) - pow(b_prop(d, r, k), 2.0)/(2.0*nu2);

			}
		}
	}

	List result;
	result["b"] = b;
	result["theta"] = theta;
	result["acceptance"] = acceptance;
	result["gradient"] = log_posterior_gradient;


	lar = log_priorRatio + log_proposalRatio + cll_prop - cll;
//	Rcout << "log_priorRatio = " << log_priorRatio << std::endl;
//	Rcout << "log_proposalRatio = " << log_proposalRatio << std::endl;
//	Rcout << "lar = " << lar << std::endl;
	if( log(arma::randu()) <  lar ){
		result["b"] = b_prop;
		result["theta"] = theta_prop;
		result["acceptance"] = 1;
	}

	return(result);

}


