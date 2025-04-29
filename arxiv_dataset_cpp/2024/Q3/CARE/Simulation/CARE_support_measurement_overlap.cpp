#include <armadillo>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <Rcpp.h>
#include <vector>

using namespace Rcpp;
using namespace arma;


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double CD_measureOverlapC(arma::vec b_exp, arma::vec b_out, arma::vec se2_exp,  arma::vec se2_out, arma::vec w, arma::vec se_exp,  arma::vec se_out, double rho) {
    
    // CHECK this
    double a = arma::accu(w % ((arma::pow(b_exp,2) - se2_exp) / se2_out) );
    double b = arma::accu(w % ( (b_exp  % b_out - rho * se_exp % se_out) / se2_out ) );
    
    double out = b / a;
    return(out);
}



// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List  CD_estimateOverlapC2(arma::vec &b_exp, arma::vec &b_out, arma::vec & se2_exp,  arma::vec & se2_out, arma::vec w, const int K,  double initial_theta, const int maxit, arma::vec se_exp,  arma::vec se_out, double rho) {
    
    const int p = b_exp.size();
    
    double theta = initial_theta;
    double thetaold = theta - 1;
    
    
    arma::vec vimportance;
    vimportance.zeros(p);
    
    arma::vec vbg;
    vbg.zeros(p);
    
    int iteindx = 0;
    
    double dlx = 1.0;
    if(K > 0) {
        while ( (dlx > 1e-5)  & (iteindx < maxit) ) {
            thetaold = theta;
            iteindx = iteindx + 1;
            
            vimportance = (arma::pow(b_out - theta * b_exp,2) ) / (se2_out) - theta * theta * se2_exp / se2_out + 2 * theta * rho * se_exp % se_out;
            
            vimportance = w % vimportance;
            
            arma::uvec tmpindx = arma::sort_index(vimportance,"descend");
            arma::uvec tmpindx2 = tmpindx.tail(p-K);
            vbg = b_out - theta * b_exp;
            
            vbg(tmpindx2).zeros();
            
            
            arma::vec b_exp2 = b_exp.elem(tmpindx2);
            arma::vec se2_out2 = se2_out.elem(tmpindx2);
            arma::vec b_out2 = b_out.elem(tmpindx2);
            arma::vec se2_exp2 = se2_exp.elem(tmpindx2);
            arma::vec w2 = w.elem(tmpindx2);
            
            arma::vec se_exp2 = se_exp.elem(tmpindx2);
            arma::vec se_out2 = se_out.elem(tmpindx2);

            //update theta
            theta = CD_measureOverlapC(b_exp2, b_out2, se2_exp2, se2_out2, w2, se_exp2, se_out2, rho);
            
            dlx = thetaold - theta;
            if(dlx <0) {
                dlx = -1 * dlx;
            }
        }
        
        // update vbg and muvec once we find optimal theta
        arma::uvec nonzeroind = arma::find(vbg != 0);
        //muvec.elem(nonzeroind) = b_exp.elem(nonzeroind);
        
        arma::vec tmpvbg = b_out - theta * b_exp;
        vbg.elem(nonzeroind) = tmpvbg.elem(nonzeroind);
        
    } else {
        theta = CD_measureOverlapC(b_exp, b_out, se2_exp, se2_out, w, se_exp,se_out, rho);
    }
    
    Rcpp::List out;
    out["theta"] = theta;
    out["rvec"] = vbg;
    out["iteindx"] = iteindx;
    
    return (out);
    
}



// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List  CD_randomOverlapC(arma::vec &b_exp, arma::vec &b_out, arma::vec &se2_exp,  arma::vec & se2_out, arma::vec & w, const int K,  const int random_start, const int maxit, double initial_theta, arma::vec se_exp,  arma::vec se_out, double rho) {
    
    const int p = b_exp.size();
    
    arma:: vec tmp = b_out / b_exp;
    double minTheta = tmp.min();
    double maxTheta = tmp.max();
    
    double rangeTheta = maxTheta - minTheta;
    //double initial_theta = 0.0;
    
    arma::vec initial_mu;
    initial_mu.zeros(p);
    
    arma::vec invalidIV;
    invalidIV.zeros(p);
    
    double NegL = -999;
    double theta = -999;
    //double sdTheta;
    
    int outj = 0;
    for (int j = 0; j <= random_start; j++) {
        
        if (j > 0) {
            arma::vec tmpInitial_theta = rangeTheta * arma::randu(1) + minTheta;
            initial_theta = tmpInitial_theta(0);
        }
        
        Rcpp::List MLEresult = CD_estimateOverlapC2(b_exp, b_out, se2_exp, se2_out, w, K, initial_theta, maxit, se_exp, se_out, rho);
        
        double thetatmp = MLEresult[0];
        //double sdThetaTmp = MLEresult[1];
        arma::vec r_vec = MLEresult[1];
        
        
        // this needs to be revised
        arma::uvec zeroind = arma::find(r_vec == 0.0);
        
        arma::vec b_exp2 = b_exp.elem(zeroind);
        arma::vec b_out2 = b_out.elem(zeroind);
        arma::vec se2_exp2 = se2_exp.elem(zeroind);
        arma::vec se2_out2 = se2_out.elem(zeroind);
        arma::vec w2 = w.elem(zeroind);
        
        arma::vec se_exp2 = se_exp.elem(zeroind);
        arma::vec se_out2 = se_out.elem(zeroind);
        
        // CHECK this
        //double NegLtmp = 0.5 * arma::accu( w2 % (arma::pow(b_out2 - thetatmp * b_exp2 ,2) / se2_out2)) - 0.5 * arma::accu( thetatmp * thetatmp * (w2 % se2_exp2 / se2_out2));
        
        // Need to use the proflie likelihood for overlapping sample scenario
        double NegLtmp = 0.5 * arma::accu( w2 % arma::pow(b_out2 - thetatmp * b_exp2,2) / (thetatmp * thetatmp * se2_exp2 + se2_out2 + 2 * thetatmp * rho * se_exp2 % se_out2) );

        
        if(outj == 0) {
            NegL = NegLtmp;
            theta = thetatmp;
            invalidIV = r_vec;
            //sdTheta = sdThetaTmp;
        }
        
        if(NegLtmp < NegL) {
            NegL = NegLtmp;
            theta = thetatmp;
            invalidIV = r_vec;
            //sdTheta = sdThetaTmp;
        }
        
        outj = outj + 1;
        
    }
    
    Rcpp::List out;
    out["theta"] = theta;
    //out["se"] = sdTheta;
    out["NegL"] = NegL;
    out["r_est"] = invalidIV;
    
    return (out);
}




// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec  cMLCOverlap_CD(arma::vec &b_exp, arma::vec &b_out, arma::vec &se2_exp,  arma::vec & se2_out, arma::vec K_vec, arma::vec w,  const int random_start, const int maxit, const int n, arma::vec se_exp,  arma::vec se_out, double rho) {
    
    const int M = K_vec.size();
    //const int p = b_exp.size();
    
    arma::vec theta_v;
    arma::vec sd_v;
    arma::vec l_v;
    // arma::mat invalid_mat(p,M);
    
    double initial_theta = 0.0;
    
    
    theta_v.zeros(M);
    sd_v.zeros(M);
    l_v.zeros(M);
    // invalid_mat.fill(0);
    
    for (int j = 0; j < M; j++) {
        int K = K_vec(j);
        Rcpp::List randRes = CD_randomOverlapC(b_exp, b_out, se2_exp, se2_out, w, K, random_start, maxit, initial_theta, se_exp, se_out, rho);
        
        theta_v(j) = randRes[0];
        //initial_theta = theta_v(j);
        //sd_v(j) = randRes[1];
        l_v(j) = randRes[1];
        
        //arma::vec tmpInvalid = randRes[3];
        //invalid_mat.col(j) = tmpInvalid;
    }
    
    
    arma::vec BIC_v = log(n) * K_vec + 2 * l_v;
    BIC_v = BIC_v - BIC_v.min();
    arma::vec weight_v = arma::exp(-0.5 * BIC_v);
    weight_v = weight_v / arma::accu(weight_v);
    
    double MA_BIC_Theta = arma::accu(theta_v % weight_v);
    //double MA_BIC_se = arma::accu(weight_v % sqrt(arma::pow(sd_v2,2) + arma::pow(theta_v2 - MA_BIC_Theta,2) ) );
    
    arma::uword i = BIC_v.index_min();
    
    double BIC_Theta = theta_v(i);
    
    
    arma::vec AIC_v = 2 * K_vec + 2 * l_v;
    AIC_v = AIC_v - AIC_v.min();
    arma::vec weight_v2 = arma::exp(-0.5 * AIC_v);
    weight_v2 = weight_v2 / arma::accu(weight_v2);
    
    double MA_AIC_Theta = arma::accu(theta_v % weight_v2);
    //double MA_BIC_se = arma::accu(weight_v % sqrt(arma::pow(sd_v2,2) + arma::pow(theta_v2 - MA_BIC_Theta,2) ) );
    
    arma::uword i2 = AIC_v.index_min();
    
    double AIC_Theta = theta_v(i2);
    
    arma::vec out;
    out.zeros(4);
    out(0) = MA_BIC_Theta;
    out(1) = BIC_Theta;
    out(2) = MA_AIC_Theta;
    out(3) = AIC_Theta;
    
    return (out);
}



// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat  cMLCOverlap_CD_boot_efron(arma::vec &b_exp, arma::vec &b_out, arma::vec &se2_exp,  arma::vec & se2_out, Rcpp::List K_vec, arma::mat wAll,  const int random_start, const int maxit, const int n, const int nrep, arma::vec se_exp,  arma::vec se_out, double rho) {
        
    arma::mat out(nrep,4);
    
    for (int j = 0; j < nrep; j++) {
        arma::vec w = wAll.col(j);
        
        arma::uvec zeroind = arma::find(w != 0.0);
        
        arma::vec b_exp2 = b_exp.elem(zeroind);
        arma::vec b_out2 = b_out.elem(zeroind);
        arma::vec se2_exp2 = se2_exp.elem(zeroind);
        arma::vec se2_out2 = se2_out.elem(zeroind);
        arma::vec w2 = w.elem(zeroind);
        
        arma::vec se_exp2 = se_exp.elem(zeroind);
        arma::vec se_out2 = se_out.elem(zeroind);
        
        //int p = b_exp2.size();
        
        arma::vec K_vec2 = K_vec[j];
        
        arma::vec tmp = cMLCOverlap_CD(b_exp2, b_out2, se2_exp2, se2_out2,  K_vec2, w2, random_start, maxit, n, se_exp2, se_out2, rho);
        //Rcout << "Finish test" << j << std::endl;
        
        out.row(j) = tmp.t();
    }
    
    return(out);
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List cMLCOverlap_CD3(arma::vec &b_exp, arma::vec &b_out, arma::vec &se2_exp,  arma::vec & se2_out, arma::vec K_vec, arma::vec w,  const int random_start, const int maxit, const int n, arma::vec se_exp,  arma::vec se_out, double rho) {// output the invalid IVs
    
    const int M = K_vec.size();
    const int p = b_exp.size();
    
    arma::vec theta_v;
    arma::vec sd_v;
    arma::vec l_v;
    arma::mat invalid_mat(p,M);
    
    theta_v.zeros(M);
    sd_v.zeros(M);
    l_v.zeros(M);
    invalid_mat.fill(0);
    
    double initial_theta = 0.0;

    //Rcout << "Finish test" << std::endl;
    
    for (int j = 0; j < M; j++) {
        int K = K_vec(j);
        Rcpp::List randRes = CD_randomOverlapC(b_exp, b_out, se2_exp, se2_out, w, K, random_start, maxit,initial_theta, se_exp, se_out, rho);
        
        theta_v(j) = randRes[0];
        //initial_theta = theta_v(j);
        //sd_v(j) = randRes[1];
        l_v(j) = randRes[1];
        
        arma::vec tmpInvalid = randRes[2];
        invalid_mat.col(j) = tmpInvalid;
    }
    
    //Rcout << "Finish test" << std::endl;    
    
    arma::vec BIC_v = log(n) * K_vec + 2 * l_v;
    BIC_v = BIC_v - BIC_v.min();
    arma::vec weight_v = arma::exp(-0.5 * BIC_v);
    weight_v = weight_v / arma::accu(weight_v);
    
    double MA_BIC_Theta = arma::accu(theta_v % weight_v);
    //double MA_BIC_se = arma::accu(weight_v % sqrt(arma::pow(sd_v2,2) + arma::pow(theta_v2 - MA_BIC_Theta,2) ) );
    
    arma::uword i = BIC_v.index_min();
    
    double BIC_Theta = theta_v(i);
    
    
    arma::vec AIC_v = 2 * K_vec + 2 * l_v;
    AIC_v = AIC_v - AIC_v.min();
    arma::vec weight_v2 = arma::exp(-0.5 * AIC_v);
    weight_v2 = weight_v2 / arma::accu(weight_v2);
    
    double MA_AIC_Theta = arma::accu(theta_v % weight_v2);
    //double MA_BIC_se = arma::accu(weight_v % sqrt(arma::pow(sd_v2,2) + arma::pow(theta_v2 - MA_BIC_Theta,2) ) );
    
    arma::uword i2 = AIC_v.index_min();
    
    double AIC_Theta = theta_v(i2);
    
    arma::vec out;
    out.zeros(4);
    out(0) = MA_BIC_Theta;
    out(1) = BIC_Theta;
    out(2) = MA_AIC_Theta;
    out(3) = AIC_Theta;
    
    Rcpp::List out2;
    
    out2["theta"] = out;
    out2["Invalid_BIC"] = invalid_mat.col(i);
    out2["Invalid_AIC"] = invalid_mat.col(i2);
    
    return (out2);
}


