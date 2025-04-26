#include "operators.hpp"

/**
 *  get time stepping operator (real version)
 */
arma::sp_mat get_Al(double lambda, int Nl){
    // set size
    arma::sp_mat Al(Nl, Nl);
    // set ones on diagonal
    for(int i = 0; i < Nl; i++){
        Al(i,i) = 1.0;
    }
    // set -lambda on off-diagonal (one below)
    for(int i = 0; i < Nl-1; i++){
        Al(i+1,i) = -lambda;
    }
    return Al;
}

/**
 *  get time stepping operator (complex version)
 */
arma::sp_cx_mat get_Al(arma::cx_double lambda, int Nl){
    // set size
    arma::sp_cx_mat Al(Nl, Nl);
    // set ones on diagonal
    for(int i = 0; i < Nl; i++){
        Al(i,i) = {1.0,0.0};
    }
    // set -lambda on off-diagonal (one below)
    for(int i = 0; i < Nl-1; i++){
        Al(i+1,i) = -lambda;
    }
    return Al;
}

/**
 *  get restriction operator (real version)
 */
arma::sp_mat get_Rl(double lambda, int Nl, int ml){
    // compute coarse-grid size
    int Nl1 = (Nl - 1) / ml + 1;
    // set size
    arma::sp_mat Rl(Nl1, Nl);
    // get non-zero values
    arma::vec vals = arma::ones<arma::vec>(ml);
    for(int i = 0; i < ml; i++){
        vals(i) = std::pow(lambda, ml-1-i);
    }
    // set Rl
    int j = 0;
    Rl(j,j) = 1.0;
    for(int i = 1; i < Nl-1; i += ml){
        j++;
        Rl(j,arma::span(i,i+ml-1)) = vals.st();
    }
    return Rl;
}

/**
 *  get restriction operator (complex version)
 */
arma::sp_cx_mat get_Rl(arma::cx_double lambda, int Nl, int ml){
    // compute coarse-grid size
    int Nl1 = (Nl - 1) / ml + 1;
    // set size
    arma::sp_cx_mat Rl(Nl1, Nl);
    // get non-zero values
    arma::cx_vec vals = arma::zeros<arma::cx_vec>(ml);
    for(int i = 0; i < ml; i++){
        vals(i) = std::pow(lambda, ml-1-i);
    }
    // set Rl
    int j = 0;
    Rl(j,j) = {1.0,0.0};
    for(int i = 1; i < Nl-1; i += ml){
        j++;
        Rl(j,arma::span(i,i+ml-1)) = vals.st();
    }
    return Rl;
}

/**
 *  get injection operator (real version)
 */
arma::sp_mat get_RIl(double lambda, int Nl, int ml){
    // compute coarse-grid size
    int Nl1 = (Nl - 1) / ml + 1;
    // set size
    arma::sp_mat RIl(Nl1, Nl);
    // set Pl
    int j = 0;
    for(int i = 0; i < Nl1; i++){
        RIl(i,j) = 1.0;
        j += ml;
    }
    return RIl;
}

/**
 *  get injection operator (complex version)
 */
arma::sp_cx_mat get_RIl(arma::cx_double lambda, int Nl, int ml){
    // compute coarse-grid size
    int Nl1 = (Nl - 1) / ml + 1;
    // set size
    arma::sp_cx_mat RIl(Nl1, Nl);
    // set Pl
    int j = 0;
    for(int i = 0; i < Nl1; i++){
        RIl(i,j) = {1.0,0.0};
        j += ml;
    }
    return RIl;
}

arma::sp_mat get_Sl(double lambda, int Nl, int ml){
    // compute coarse-grid size
    int Nl1 = (Nl - 1) / ml + 1;
    // set size
    arma::sp_mat Sl(Nl, Nl-Nl1);
    // set Sl
    int idx = 0;
    for(int i = 0; i < Nl; i++){
        if((i % ml) > 0){
            Sl(i,idx) = 1.0;
            idx++;
        }
    }
    return Sl;
}

arma::sp_cx_mat get_Sl(arma::cx_double lambda, int Nl, int ml){
    // compute coarse-grid size
    int Nl1 = (Nl - 1) / ml + 1;
    // set size
    arma::sp_cx_mat Sl(Nl, Nl-Nl1);
    // set Sl
    int idx = 0;
    for(int i = 0; i < Nl; i++){
        if((i % ml) > 0){
            Sl(i,idx) = {1.0,0.0};
            idx++;
        }
    }
    return Sl;
}

/**
 *  get interpolation operator (real version)
 */
arma::sp_mat get_Pl(double lambda, int Nl, int ml){
    // compute coarse-grid size
    int Nl1 = (Nl - 1) / ml + 1;
    // set size
    arma::sp_mat Pl(Nl, Nl1);
    // get non-zero values
    arma::vec vals = arma::ones<arma::vec>(ml);
    for(int i = 0; i < ml; i++){
        vals(i) = std::pow(lambda, i);
    }
    // set Pl
    int j = 0;
    for(int i = 0; i < Nl-1; i += ml){
        Pl(arma::span(i,i+ml-1),j) = vals;
        j++;
    }
    Pl(Nl-1,Nl1-1) = 1.0;
    return Pl;
}

/**
 *  get interpolation operator (complex version)
 */
arma::sp_cx_mat get_Pl(arma::cx_double lambda, int Nl, int ml){
    // compute coarse-grid size
    int Nl1 = (Nl - 1) / ml + 1;
    // set size
    arma::sp_cx_mat Pl(Nl, Nl1);
    // get non-zero values
    arma::cx_vec vals = arma::ones<arma::cx_vec>(ml);
    for(int i = 0; i < ml; i++){
        vals(i) = std::pow(lambda, i);
    }
    // set Pl
    int j = 0;
    for(int i = 0; i < Nl-1; i += ml){
        Pl(arma::span(i,i+ml-1),j) = vals;
        j++;
    }
    Pl(Nl-1,Nl1-1) = {1.0,0.0};
    return Pl;
}

/**
 *  get all operators needed for constructing error and residual propagator (real version)
 */
void get_operators(arma::sp_mat **ptrA, arma::sp_mat **ptrR, arma::sp_mat **ptrRI, arma::sp_mat **ptrS, arma::sp_mat **ptrP,
                           arma::Col<double> lambda, arma::Col<int> Nl, arma::Col<int> ml){
    int numLevels   = lambda.n_elem;
    for(int i = 0; i < numLevels-1; i++){
        ptrA[i]     = new arma::sp_mat();
        ptrP[i]     = new arma::sp_mat();
        ptrR[i]     = new arma::sp_mat();
        ptrRI[i]    = new arma::sp_mat();
        ptrS[i]     = new arma::sp_mat();
        (*ptrA[i])  = get_Al(lambda(i), Nl(i));
        (*ptrP[i])  = get_Pl(lambda(i), Nl(i), ml(i));
        (*ptrR[i])  = get_Rl(lambda(i), Nl(i), ml(i));
        (*ptrRI[i]) = get_RIl(lambda(i), Nl(i), ml(i));
        (*ptrS[i])  = get_Sl(lambda(i), Nl(i), ml(i));
    }
    ptrA[numLevels-1]       = new arma::sp_mat();
    (*ptrA[numLevels-1])    = get_Al(lambda(numLevels-1), Nl(numLevels-1));
}

/**
 *  get all operators needed for constructing error and residual propagator (complex version)
 */
void get_operators(arma::sp_cx_mat **ptrA, arma::sp_cx_mat **ptrR, arma::sp_cx_mat **ptrRI, arma::sp_cx_mat **ptrS, arma::sp_cx_mat **ptrP,
                   arma::Col<arma::cx_double> lambda, arma::Col<int> Nl, arma::Col<int> ml){
    int numLevels   = lambda.n_elem;
    for(int i = 0; i < numLevels-1; i++){
        ptrA[i]     = new arma::sp_cx_mat();
        ptrP[i]     = new arma::sp_cx_mat();
        ptrR[i]     = new arma::sp_cx_mat();
        ptrRI[i]    = new arma::sp_cx_mat();
        ptrS[i]     = new arma::sp_cx_mat();
        (*ptrA[i])  = get_Al(lambda(i), Nl(i));
        (*ptrP[i])  = get_Pl(lambda(i), Nl(i), ml(i));
        (*ptrR[i])  = get_Rl(lambda(i), Nl(i), ml(i));
        (*ptrRI[i]) = get_RIl(lambda(i), Nl(i), ml(i));
        (*ptrS[i])  = get_Sl(lambda(i), Nl(i), ml(i));
    }
    ptrA[numLevels-1]       = new arma::sp_cx_mat();
    (*ptrA[numLevels-1])    = get_Al(lambda(numLevels-1), Nl(numLevels-1));
}
