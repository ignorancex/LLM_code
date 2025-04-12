#include "propagators.hpp"

/**
 *  get contributions to error propagator for V-cycle with F-relaxation and > 2 grid levels from intermediate levels (real eigenvalues)
 */
int get_E_F_intermediate(arma::sp_mat *E_F, arma::Col<double> lambda, arma::Col<int> Nl, arma::Col<int> ml, int theoryLevel){
    // check for valid arguments
    if(theoryLevel != 1){
        std::cout << ">>>ERROR: Intermediate level contributions of error propagator only implemented for level 1." << std::endl;
        throw;
    }
    if((lambda.n_elem != Nl.n_elem) || (lambda.n_elem != ml.n_elem+1)){
        std::cout << ">>>ERROR: Error propagator encountered invalid definition of number of levels." << std::endl;
    }
    // check if time stepper is stable
    if(arma::any(arma::abs(lambda) > constants::time_stepper_stability_limit)){
        return -1;
    }
    // get number of levels
    int                 numLevels   = lambda.n_elem;
    // pointers to all operators
    arma::sp_mat *ptrA[numLevels];
    arma::sp_mat *ptrR[numLevels-1];
    arma::sp_mat *ptrRI[numLevels-1];
    arma::sp_mat *ptrS[numLevels-1];
    arma::sp_mat *ptrP[numLevels-1];
    // compute operators
    get_operators(ptrA, ptrR, ptrRI, ptrS, ptrP, lambda, Nl, ml);
    // compute error propagator contributions for F-relaxation, using form: I - term * R0A0P0 - summ * R0A0P0
    // pre-compute R0*A0*P0 = RI0*A0*P0
    arma::sp_mat R0A0P0 = (*ptrRI[0]) * (*ptrA[0]) * (*ptrP[0]);
    // pre-compute contributions from intermediate levels
    arma::sp_mat  summ(Nl(1),Nl(1));
    arma::sp_mat term2 = arma::sp_mat();
    for(int level = 0; level < numLevels-2; level++){
        term2 = (*ptrS[level+1]) * arma::sp_mat(arma::mat((*ptrS[level+1]).st() * (*ptrA[level+1]) * (*ptrS[level+1])).i()) * (*ptrS[level+1]).st();
        for(int prepostIdx = level; prepostIdx > 0; prepostIdx--){
            term2 = (*ptrP[prepostIdx]) * term2 * (*ptrR[prepostIdx]);
        }
        summ += term2;
    }
    // compute intermediate level contributions to error propagator for F-relaxation on level 1
    (*E_F) = summ * R0A0P0;
    return 0;
}

/**
 *  get contributions to error propagator for V-cycle with F-relaxation and > 2 grid levels from intermediate levels (complex eigenvalues)
 */
int get_E_F_intermediate(arma::sp_cx_mat *E_F, arma::Col<arma::cx_double> lambda, arma::Col<int> Nl, arma::Col<int> ml, int theoryLevel){
    // check for valid arguments
    if(theoryLevel != 1){
        std::cout << ">>>ERROR: Intermediate level contributions of error propagator only implemented for level 1." << std::endl;
        throw;
    }
    if((lambda.n_elem != Nl.n_elem) || (lambda.n_elem != ml.n_elem+1)){
        std::cout << ">>>ERROR: Error propagator encountered invalid definition of number of levels." << std::endl;
    }
    // check if time stepper is stable
    if(arma::any(arma::abs(lambda) > constants::time_stepper_stability_limit)){
        return -1;
    }
    // get number of levels
    int                 numLevels   = lambda.n_elem;
    // pointers to all operators
    arma::sp_cx_mat *ptrA[numLevels];
    arma::sp_cx_mat *ptrR[numLevels-1];
    arma::sp_cx_mat *ptrRI[numLevels-1];
    arma::sp_cx_mat *ptrS[numLevels-1];
    arma::sp_cx_mat *ptrP[numLevels-1];
    // compute operators
    get_operators(ptrA, ptrR, ptrRI, ptrS, ptrP, lambda, Nl, ml);
    // compute error propagator contributions for F-relaxation, using form: I - term * R0A0P0 - summ * R0A0P0
    // pre-compute R0*A0*P0 = RI0*A0*P0
    arma::sp_cx_mat R0A0P0 = (*ptrRI[0]) * (*ptrA[0]) * (*ptrP[0]);
    // pre-compute contributions from intermediate levels
    arma::sp_cx_mat  summ(Nl(1),Nl(1));
    arma::sp_cx_mat term2 = arma::sp_cx_mat();
    for(int level = 0; level < numLevels-2; level++){
        term2 = (*ptrS[level+1]) * arma::sp_cx_mat(arma::cx_mat((*ptrS[level+1]).st() * (*ptrA[level+1]) * (*ptrS[level+1])).i()) * (*ptrS[level+1]).st();
        for(int prepostIdx = level; prepostIdx > 0; prepostIdx--){
            term2 = (*ptrP[prepostIdx]) * term2 * (*ptrR[prepostIdx]);
        }
        summ += term2;
    }
    // compute intermediate level contributions to error propagator for F-relaxation on level 1
    (*E_F) = summ * R0A0P0;
    return 0;
}

/**
 *  get error propagator for V-cycle with F-relaxation and >= 2 grid levels (real eigenvalues)
 */
int get_E_F(arma::sp_mat *E_F, arma::Col<double> lambda, arma::Col<int> Nl, arma::Col<int> ml, int theoryLevel){
    // check for valid arguments
    if((theoryLevel != 0) && (theoryLevel != 1)){
        std::cout << ">>>ERROR: Error propagator only implemented for level 0 and level 1." << std::endl;
        throw;
    }
    if((lambda.n_elem != Nl.n_elem) || (lambda.n_elem != ml.n_elem+1)){
        std::cout << ">>>ERROR: Error propagator encountered invalid definition of number of levels." << std::endl;
    }
    // check if time stepper is stable
    if(arma::any(arma::abs(lambda) > constants::time_stepper_stability_limit)){
        return -1;
    }
    // get number of levels
    int                 numLevels   = lambda.n_elem;
    // pointers to all operators
    arma::sp_mat *ptrA[numLevels];
    arma::sp_mat *ptrR[numLevels-1];
    arma::sp_mat *ptrRI[numLevels-1];
    arma::sp_mat *ptrS[numLevels-1];
    arma::sp_mat *ptrP[numLevels-1];
    // compute operators
    get_operators(ptrA, ptrR, ptrRI, ptrS, ptrP, lambda, Nl, ml);
    // compute error propagator for F-relaxation, using form: I - term * R0A0P0 - summ * R0A0P0
    // pre-compute R0*A0*P0 = RI0*A0*P0
    arma::sp_mat R0A0P0 = (*ptrRI[0]) * (*ptrA[0]) * (*ptrP[0]);
    // pre-compute coarse-grid contribution
    arma::sp_mat term = arma::sp_mat(arma::mat((*ptrA[numLevels-1])).i());
    for(int prepostIdx = numLevels-2; prepostIdx > 0; prepostIdx--){
        term = (*ptrP[prepostIdx]) * term * (*ptrR[prepostIdx]);
    }
    // pre-compute contributions from intermediate levels
    arma::sp_mat  summ(Nl(1),Nl(1));
    arma::sp_mat term2 = arma::sp_mat();
    arma::sp_mat identity = arma::speye(Nl(1),Nl(1));
    for(int level = 0; level < numLevels-2; level++){
        term2 = (*ptrS[level+1]) * arma::sp_mat(arma::mat((*ptrS[level+1]).st() * (*ptrA[level+1]) * (*ptrS[level+1])).i()) * (*ptrS[level+1]).st();
        for(int prepostIdx = level; prepostIdx > 0; prepostIdx--){
            term2 = (*ptrP[prepostIdx]) * term2 * (*ptrR[prepostIdx]);
        }
        summ += term2;
    }
    // compute error propagator for F-relaxation on level 0
    if(theoryLevel == 1){
        (*E_F) = identity - term * R0A0P0 - summ * R0A0P0;
    // compute error propagator for F-relaxation on level 1
    } else if(theoryLevel == 0){
        (*E_F) = (*ptrP[0]) * (identity - term * R0A0P0 - summ * R0A0P0) * (*ptrRI[0]);
    }
    return 0;
}

/**
 *  get error propagator for V-cycle with F-relaxation and >= 2 grid levels (complex eigenvalues)
 */
int get_E_F(arma::sp_cx_mat *E_F, arma::Col<arma::cx_double> lambda, arma::Col<int> Nl, arma::Col<int> ml, int theoryLevel){
    // check for valid arguments
    if((theoryLevel != 0) && (theoryLevel != 1)){
        std::cout << ">>>ERROR: Error propagator only implemented for level 0 and level 1." << std::endl;
        throw;
    }
    if((lambda.n_elem != Nl.n_elem) || (lambda.n_elem != ml.n_elem+1)){
        std::cout << ">>>ERROR: Error propagator encountered invalid definition of number of levels." << std::endl;
    }
    // check if time stepper is stable
    if(arma::any(arma::abs(lambda) > constants::time_stepper_stability_limit)){
        return -1;
    }
    // get number of levels
    int                 numLevels   = lambda.n_elem;
    // pointers to all operators
    arma::sp_cx_mat *ptrA[numLevels];
    arma::sp_cx_mat *ptrR[numLevels-1];
    arma::sp_cx_mat *ptrRI[numLevels-1];
    arma::sp_cx_mat *ptrS[numLevels-1];
    arma::sp_cx_mat *ptrP[numLevels-1];
    // compute operators
    get_operators(ptrA, ptrR, ptrRI, ptrS, ptrP, lambda, Nl, ml);
    // compute error propagator for F-relaxation, using form: I - term * R0A0P0 - summ * R0A0P0
    // pre-compute R0*A0*P0 = RI0*A0*P0
    arma::sp_cx_mat R0A0P0 = (*ptrRI[0]) * (*ptrA[0]) * (*ptrP[0]);
    // pre-compute coarse-grid contribution
    arma::sp_cx_mat term = arma::sp_cx_mat(arma::cx_mat((*ptrA[numLevels-1])).i());
    for(int prepostIdx = numLevels-2; prepostIdx > 0; prepostIdx--){
        term = (*ptrP[prepostIdx]) * term * (*ptrR[prepostIdx]);
    }
    // pre-compute contributions from intermediate levels
    arma::sp_cx_mat  summ(Nl(1),Nl(1));
    arma::sp_cx_mat term2 = arma::sp_cx_mat();
    arma::sp_cx_mat identity = arma::sp_cx_mat(arma::speye(Nl(1),Nl(1)),arma::speye(Nl(1),Nl(1)).zeros());
    for(int level = 0; level < numLevels-2; level++){
        term2 = (*ptrS[level+1]) * arma::sp_cx_mat(arma::cx_mat((*ptrS[level+1]).st() * (*ptrA[level+1]) * (*ptrS[level+1])).i()) * (*ptrS[level+1]).st();
        for(int prepostIdx = level; prepostIdx > 0; prepostIdx--){
            term2 = (*ptrP[prepostIdx]) * term2 * (*ptrR[prepostIdx]);
        }
        summ += term2;
    }
    // compute error propagator for F-relaxation on level 0
    if(theoryLevel == 1){
        (*E_F) = identity - term * R0A0P0 - summ * R0A0P0;
    // compute error propagator for F-relaxation on level 1
    } else if(theoryLevel == 0){
        (*E_F) = (*ptrP[0]) * (identity - term * R0A0P0 - summ * R0A0P0) * (*ptrRI[0]);
    }
    return 0;
}

/**
 *  get error propagator for V-cycle with FCF-relaxation and >= 2 grid levels (real eigenvalues)
 */
int get_E_FCF(arma::sp_mat *E_FCF, arma::Col<double> lambda, arma::Col<int> Nl, arma::Col<int> ml, int theoryLevel){
    // check for valid arguments
    if((theoryLevel != 0) && (theoryLevel != 1)){
        std::cout << ">>>ERROR: Error propagator only implemented for level 0 and level 1." << std::endl;
        throw;
    }
    if((lambda.n_elem != Nl.n_elem) || (lambda.n_elem != ml.n_elem+1)){
        std::cout << ">>>ERROR: Error propagator encountered invalid definition of number of levels." << std::endl;
    }
    // check if time stepper is stable
    if(arma::any(arma::abs(lambda) > constants::time_stepper_stability_limit)){
        return -1;
    }
    // get number of levels
    int numLevels = lambda.n_elem;
    // pointers to all operators
    arma::sp_mat *ptrA[numLevels];
    arma::sp_mat *ptrR[numLevels-1];
    arma::sp_mat *ptrRI[numLevels-1];
    arma::sp_mat *ptrS[numLevels-1];
    arma::sp_mat *ptrP[numLevels-1];
    // compute operators
    get_operators(ptrA, ptrR, ptrRI, ptrS, ptrP, lambda, Nl, ml);
    // compute error propagator for FCF-relaxation
    // pre-compute A0*P0 = A0*P0
    arma::sp_mat A0P0 = (*ptrA[0]) * (*ptrP[0]);
    // I - (T_0^T A_0 T_0)^{-1} R_{I_0} A_0 P_0
    // with T_l = R_l^T
    arma::sp_mat identity = arma::speye(Nl(1),Nl(1));
    arma::sp_mat term;
    arma::sp_mat term1;
    arma::sp_mat term2;
    arma::sp_mat result = identity - arma::sp_mat(arma::mat(((*ptrRI[0]) * (*ptrA[0]) * (*ptrRI[0]).st())).i()) * (*ptrRI[0]) * A0P0;
    // coarsest grid
    // - (prod_k P_k)
    // * A_{n_l - 1}^{-1}
    // * (prod_k [R_k [I - A_k T_k inv(T_k^T A_k T_k) T_k^T] ] )
    // * A_0 P_0
    term = arma::sp_mat(arma::mat(*ptrA[numLevels-1]).i());
    for(int preIdx = numLevels-2; preIdx > 0; preIdx--){
        term = (*ptrP[preIdx]) * term;
    }
    for(int postIdx = numLevels-2; postIdx >= 0; postIdx--){
        term = term * (*ptrR[postIdx]) * (
            arma::speye(Nl(postIdx),Nl(postIdx))
            - (*ptrA[postIdx]) * (*ptrRI[postIdx]).st() * arma::sp_mat(arma::mat((*ptrRI[postIdx]) * (*ptrA[postIdx]) * (*ptrRI[postIdx]).st()).i()) * (*ptrRI[postIdx])
        );
    }
    result = result - term * A0P0;
    // and contribution of intermediate levels
    for(int level = 1; level < numLevels-1; level++){
        term1 = (*ptrRI[level]).st() * arma::sp_mat(arma::mat((*ptrRI[level])     * (*ptrA[level]) * (*ptrRI[level]).st()).i()) * (*ptrRI[level]);
        term2 = (*ptrS[level])       * arma::sp_mat(arma::mat((*ptrS[level]).st() * (*ptrA[level]) * (*ptrS[level])      ).i()) * (*ptrS[level]).st() * (
            arma::speye(Nl(level),Nl(level))
            - (*ptrA[level]) * (*ptrRI[level]).st() * arma::sp_mat(arma::mat((*ptrRI[level]) * (*ptrA[level]) * (*ptrRI[level]).st()).i()) * (*ptrRI[level])
        );
        for(int preIdx = level - 1; preIdx > 0; preIdx--){
            term1 = (*ptrP[preIdx]) * term1;
            term2 = (*ptrP[preIdx]) * term2;
        }
        for(int postIdx = level - 1; postIdx >= 0; postIdx--){
            term1 = term1 * (*ptrR[postIdx]) * (
                arma::speye(Nl(postIdx),Nl(postIdx))
                - (*ptrA[postIdx]) * (*ptrRI[postIdx]).st() * arma::sp_mat(arma::mat((*ptrRI[postIdx]) * (*ptrA[postIdx]) * (*ptrRI[postIdx]).st()).i()) * (*ptrRI[postIdx])
            );
            term2 = term2 * (*ptrR[postIdx]) * (
                arma::speye(Nl(postIdx),Nl(postIdx))
                - (*ptrA[postIdx]) * (*ptrRI[postIdx]).st() * arma::sp_mat(arma::mat((*ptrRI[postIdx]) * (*ptrA[postIdx]) * (*ptrRI[postIdx]).st()).i()) * (*ptrRI[postIdx])
            );
        }
        result = result - (term1 + term2) * A0P0;
    }
    // compute error propagator for FCF-relaxation on level 0
    if(theoryLevel == 1){
        (*E_FCF) = result;
    // compute error propagator for FCF-relaxation on level 1
    } else if(theoryLevel == 0){
        (*E_FCF) = (*ptrP[0]) * result * (*ptrRI[0]);
    }
    return 0;
}

/**
 *  get error propagator for V-cycle with FCF-relaxation and >= 2 grid levels (complex eigenvalues)
 */
int get_E_FCF(arma::sp_cx_mat *E_FCF, arma::Col<arma::cx_double> lambda, arma::Col<int> Nl, arma::Col<int> ml, int theoryLevel){
    // check for valid arguments
    if((theoryLevel != 0) && (theoryLevel != 1)){
        std::cout << ">>>ERROR: Error propagator only implemented for level 0 and level 1." << std::endl;
        throw;
    }
    if((lambda.n_elem != Nl.n_elem) || (lambda.n_elem != ml.n_elem+1)){
        std::cout << ">>>ERROR: Error propagator encountered invalid definition of number of levels." << std::endl;
    }
    // check if time stepper is stable
    if(arma::any(arma::abs(lambda) > constants::time_stepper_stability_limit)){
        return -1;
    }
    // get number of levels
    int numLevels = lambda.n_elem;
    // pointers to all operators
    arma::sp_cx_mat *ptrA[numLevels];
    arma::sp_cx_mat *ptrR[numLevels-1];
    arma::sp_cx_mat *ptrRI[numLevels-1];
    arma::sp_cx_mat *ptrS[numLevels-1];
    arma::sp_cx_mat *ptrP[numLevels-1];
    // compute operators
    get_operators(ptrA, ptrR, ptrRI, ptrS, ptrP, lambda, Nl, ml);
    // compute error propagator for FCF-relaxation
    // pre-compute A0*P0 = A0*P0
    arma::sp_cx_mat A0P0 = (*ptrA[0]) * (*ptrP[0]);
    // I - (T_0^T A_0 T_0)^{-1} R_{I_0} A_0 P_0
    // with T_l = R_l^T
    arma::sp_cx_mat identity = arma::sp_cx_mat(arma::speye(Nl(1),Nl(1)),arma::speye(Nl(1),Nl(1)).zeros());
    arma::sp_cx_mat term;
    arma::sp_cx_mat term1;
    arma::sp_cx_mat term2;
    arma::sp_cx_mat result = identity - arma::sp_cx_mat(arma::cx_mat(((*ptrRI[0]) * (*ptrA[0]) * (*ptrRI[0]).st())).i()) * (*ptrRI[0]) * A0P0;
    // coarsest grid
    // - (prod_k P_k)
    // * A_{n_l - 1}^{-1}
    // * (prod_k [R_k [I - A_k T_k inv(T_k^T A_k T_k) T_k^T] ] )
    // * A_0 P_0
    term = arma::sp_cx_mat(arma::cx_mat(*ptrA[numLevels-1]).i());
    for(int preIdx = numLevels-2; preIdx > 0; preIdx--){
        term = (*ptrP[preIdx]) * term;
    }
    for(int postIdx = numLevels-2; postIdx >= 0; postIdx--){
        term = term * (*ptrR[postIdx]) * (
            arma::sp_cx_mat(arma::speye(Nl(postIdx),Nl(postIdx)),arma::speye(Nl(postIdx),Nl(postIdx)).zeros())
            - (*ptrA[postIdx]) * (*ptrRI[postIdx]).st() * arma::sp_cx_mat(arma::cx_mat((*ptrRI[postIdx]) * (*ptrA[postIdx]) * (*ptrRI[postIdx]).st()).i()) * (*ptrRI[postIdx])
        );
    }
    result = result - term * A0P0;
    // and contribution of intermediate levels
    for(int level = 1; level < numLevels-1; level++){
        term1 = (*ptrRI[level]).st() * arma::sp_cx_mat(arma::cx_mat((*ptrRI[level])     * (*ptrA[level]) * (*ptrRI[level]).st()).i()) * (*ptrRI[level]);
        term2 = (*ptrS[level])       * arma::sp_cx_mat(arma::cx_mat((*ptrS[level]).st() * (*ptrA[level]) * (*ptrS[level])      ).i()) * (*ptrS[level]).st() * (
            arma::sp_cx_mat(arma::speye(Nl(level),Nl(level)),arma::speye(Nl(level),Nl(level)).zeros())
            - (*ptrA[level]) * (*ptrRI[level]).st() * arma::sp_cx_mat(arma::cx_mat((*ptrRI[level]) * (*ptrA[level]) * (*ptrRI[level]).st()).i()) * (*ptrRI[level])
        );
        for(int preIdx = level - 1; preIdx > 0; preIdx--){
            term1 = (*ptrP[preIdx]) * term1;
            term2 = (*ptrP[preIdx]) * term2;
        }
        for(int postIdx = level - 1; postIdx >= 0; postIdx--){
            term1 = term1 * (*ptrR[postIdx]) * (
                arma::sp_cx_mat(arma::speye(Nl(postIdx),Nl(postIdx)),arma::speye(Nl(postIdx),Nl(postIdx)).zeros())
                - (*ptrA[postIdx]) * (*ptrRI[postIdx]).st() * arma::sp_cx_mat(arma::cx_mat((*ptrRI[postIdx]) * (*ptrA[postIdx]) * (*ptrRI[postIdx]).st()).i()) * (*ptrRI[postIdx])
            );
            term2 = term2 * (*ptrR[postIdx]) * (
                arma::sp_cx_mat(arma::speye(Nl(postIdx),Nl(postIdx)),arma::speye(Nl(postIdx),Nl(postIdx)).zeros())
                - (*ptrA[postIdx]) * (*ptrRI[postIdx]).st() * arma::sp_cx_mat(arma::cx_mat((*ptrRI[postIdx]) * (*ptrA[postIdx]) * (*ptrRI[postIdx]).st()).i()) * (*ptrRI[postIdx])
            );
        }
        result = result - (term1 + term2) * A0P0;
    }
    // compute error propagator for FCF-relaxation on level 0
    if(theoryLevel == 1){
        (*E_FCF) = result;
    // compute error propagator for FCF-relaxation on level 1
    } else if(theoryLevel == 0){
        (*E_FCF) = (*ptrP[0]) * result * (*ptrRI[0]);
    }
    return 0;
}

/**
 *  get error propagator for F-cycle with F-relaxation and >= 2 grid levels (real eigenvalues)
 */
int get_F_F(arma::sp_mat *F_F, arma::Col<double> lambda, arma::Col<int> Nl, arma::Col<int> ml, int theoryLevel){
    // check for valid arguments
    if((theoryLevel != 0) && (theoryLevel != 1)){
        std::cout << ">>>ERROR: Error propagator only implemented for level 0 and level 1." << std::endl;
        throw;
    }
    if((lambda.n_elem != Nl.n_elem) || (lambda.n_elem != ml.n_elem+1)){
        std::cout << ">>>ERROR: Error propagator encountered invalid definition of number of levels." << std::endl;
    }
    // check if time stepper is stable
    if(arma::any(arma::abs(lambda) > constants::time_stepper_stability_limit)){
        return -1;
    }
    // get number of levels
    int numLevels = lambda.n_elem;
    // if two-level algorithm, all cycling strategies coincide
    arma::sp_mat *E_F = new arma::sp_mat();
    get_E_F(E_F, lambda, Nl, ml, theoryLevel);
    // get parameters for coarsest two grids
    arma::Col<double>   rec_lambda  = arma::Col<double>(lambda.rows(numLevels-2,numLevels-1));
    arma::Col<int>      rec_Nl      = arma::Col<int>(Nl.rows(numLevels-2,numLevels-1));
    arma::Col<int>      rec_ml      = arma::Col<int>(ml.row(numLevels-2));
    // pointers to all operators
    arma::sp_mat *ptrA[2];
    arma::sp_mat *ptrR[1];
    arma::sp_mat *ptrRI[1];
    arma::sp_mat *ptrS[1];
    arma::sp_mat *ptrP[1];
    // compute operators
    get_operators(ptrA, ptrR, ptrRI, ptrS, ptrP, rec_lambda, rec_Nl, rec_ml);
    // get coarsest-grid correction on level n_l-2
    arma::sp_mat identity = arma::speye(rec_Nl(1),rec_Nl(1));
    arma::sp_mat R0A0P0 = (*ptrRI[0]) * (*ptrA[0]) * (*ptrP[0]);
    arma::sp_mat MF1 = (*ptrP[0]) * (identity - arma::sp_mat(arma::mat(*ptrA[1]).i()) * R0A0P0) * (*ptrRI[0]);
    arma::sp_mat MV1 = MF1;
    arma::sp_mat MF0 = MF1;
    arma::sp_mat MV0 = MV1;
    for(int level = numLevels-3; level >= 0; level--){
        MV1         = MV0;
        MF1         = MF0;
        rec_lambda  = arma::Col<double>(lambda.rows(level,level+1));
        rec_Nl      = arma::Col<int>(Nl.rows(level,level+1));
        rec_ml      = arma::Col<int>(ml.row(level));
        get_operators(ptrA, ptrR, ptrRI, ptrS, ptrP, rec_lambda, rec_Nl, rec_ml);
        identity = arma::speye(rec_Nl(1),rec_Nl(1));
        R0A0P0 = (*ptrRI[0]) * (*ptrA[0]) * (*ptrP[0]);
        MF0 = (*ptrP[0]) * (identity - (identity - MV1 * MF1) * arma::sp_mat(arma::mat(*ptrA[1]).i()) * R0A0P0) * (*ptrRI[0]);
        MV0 = (*ptrP[0]) * (identity - (identity - MV1) * arma::sp_mat(arma::mat(*ptrA[1]).i()) * R0A0P0) * (*ptrRI[0]);
    }
    if(theoryLevel == 1){
        // on level 1, remove the pre-/post-multiplied P_0 and R_{I_0}
        (*F_F) = (*ptrRI[0]) * MF0 * (*ptrP[0]);
    }else if(theoryLevel == 0){
        (*F_F) = MF0;
    }
    return 0;
}

/**
 *  get error propagator for F-cycle with F-relaxation and >= 2 grid levels (complex eigenvalues)
 */
int get_F_F(arma::sp_cx_mat *F_F, arma::Col<arma::cx_double> lambda, arma::Col<int> Nl, arma::Col<int> ml, int theoryLevel){
    // check for valid arguments
    if((theoryLevel != 0) && (theoryLevel != 1)){
        std::cout << ">>>ERROR: Error propagator only implemented for level 0 and level 1." << std::endl;
        throw;
    }
    if((lambda.n_elem != Nl.n_elem) || (lambda.n_elem != ml.n_elem+1)){
        std::cout << ">>>ERROR: Error propagator encountered invalid definition of number of levels." << std::endl;
    }
    // check if time stepper is stable
    if(arma::any(arma::abs(lambda) > constants::time_stepper_stability_limit)){
        return -1;
    }
    // get number of levels
    int numLevels = lambda.n_elem;
    // if two-level algorithm, all cycling strategies coincide
    arma::sp_cx_mat *E_F = new arma::sp_cx_mat();
    get_E_F(E_F, lambda, Nl, ml, theoryLevel);
    // get parameters for coarsest two grids
    arma::Col<arma::cx_double>  rec_lambda  = arma::Col<arma::cx_double>(lambda.rows(numLevels-2,numLevels-1));
    arma::Col<int>              rec_Nl      = arma::Col<int>(Nl.rows(numLevels-2,numLevels-1));
    arma::Col<int>              rec_ml      = arma::Col<int>(ml.row(numLevels-2));
    // pointers to all operators
    arma::sp_cx_mat *ptrA[2];
    arma::sp_cx_mat *ptrR[1];
    arma::sp_cx_mat *ptrRI[1];
    arma::sp_cx_mat *ptrS[1];
    arma::sp_cx_mat *ptrP[1];
    // compute operators
    get_operators(ptrA, ptrR, ptrRI, ptrS, ptrP, rec_lambda, rec_Nl, rec_ml);
    // get coarsest-grid correction on level n_l-2
    arma::sp_cx_mat identity = arma::sp_cx_mat(arma::speye(rec_Nl(1),rec_Nl(1)),arma::speye(rec_Nl(1),rec_Nl(1)).zeros());
    arma::sp_cx_mat R0A0P0 = (*ptrRI[0]) * (*ptrA[0]) * (*ptrP[0]);
    arma::sp_cx_mat MF1 = (*ptrP[0]) * (identity - arma::sp_cx_mat(arma::cx_mat(*ptrA[1]).i()) * R0A0P0) * (*ptrRI[0]);
    arma::sp_cx_mat MV1 = MF1;
    arma::sp_cx_mat MF0 = MF1;
    arma::sp_cx_mat MV0 = MV1;
    for(int level = numLevels-3; level >= 0; level--){
        MV1         = MV0;
        MF1         = MF0;
        rec_lambda  = arma::Col<arma::cx_double>(lambda.rows(level,level+1));
        rec_Nl      = arma::Col<int>(Nl.rows(level,level+1));
        rec_ml      = arma::Col<int>(ml.row(level));
        get_operators(ptrA, ptrR, ptrRI, ptrS, ptrP, rec_lambda, rec_Nl, rec_ml);
        identity = arma::sp_cx_mat(arma::speye(rec_Nl(1),rec_Nl(1)),arma::speye(rec_Nl(1),rec_Nl(1)).zeros());
        R0A0P0 = (*ptrRI[0]) * (*ptrA[0]) * (*ptrP[0]);
        MF0 = (*ptrP[0]) * (identity - (identity - MV1 * MF1) * arma::sp_cx_mat(arma::cx_mat(*ptrA[1]).i()) * R0A0P0) * (*ptrRI[0]);
        MV0 = (*ptrP[0]) * (identity - (identity - MV1) * arma::sp_cx_mat(arma::cx_mat(*ptrA[1]).i()) * R0A0P0) * (*ptrRI[0]);
    }
    if(theoryLevel == 1){
        // on level 1, remove the pre-/post-multiplied P_0 and R_{I_0}
        (*F_F) = (*ptrRI[0]) * MF0 * (*ptrP[0]);
    }else if(theoryLevel == 0){
        (*F_F) = MF0;
    }
    return 0;
}

/**
 *  get error propagator for F-cycle with FCF-relaxation and >= 2 grid levels (real eigenvalues)
 */
int get_F_FCF(arma::sp_mat *F_FCF, arma::Col<double> lambda, arma::Col<int> Nl, arma::Col<int> ml, int theoryLevel){
    // check for valid arguments
    if((theoryLevel != 0) && (theoryLevel != 1)){
        std::cout << ">>>ERROR: Error propagator only implemented for level 0 and level 1." << std::endl;
        throw;
    }
    if((lambda.n_elem != Nl.n_elem) || (lambda.n_elem != ml.n_elem+1)){
        std::cout << ">>>ERROR: Error propagator encountered invalid definition of number of levels." << std::endl;
    }
    // check if time stepper is stable
    if(arma::any(arma::abs(lambda) > constants::time_stepper_stability_limit)){
        return -1;
    }
    // get number of levels
    int numLevels = lambda.n_elem;
    // get parameters for coarsest two grids
    arma::Col<double>   rec_lambda  = arma::Col<double>(lambda.rows(numLevels-2,numLevels-1));
    arma::Col<int>      rec_Nl      = arma::Col<int>(Nl.rows(numLevels-2,numLevels-1));
    arma::Col<int>      rec_ml      = arma::Col<int>(ml.row(numLevels-2));
    // pointers to all operators
    arma::sp_mat *ptrA[2];
    arma::sp_mat *ptrR[1];
    arma::sp_mat *ptrRI[1];
    arma::sp_mat *ptrS[1];
    arma::sp_mat *ptrP[1];
    // compute operators
    get_operators(ptrA, ptrR, ptrRI, ptrS, ptrP, rec_lambda, rec_Nl, rec_ml);
    // get coarsest-grid correction on level n_l-2
    arma::sp_mat identity = arma::speye(rec_Nl(1),rec_Nl(1));
    arma::sp_mat R0A0P0 = (*ptrRI[0]) * (*ptrA[0]) * (*ptrP[0]);
    arma::sp_mat MF1 = (*ptrP[0]) * (identity - arma::sp_mat(arma::mat(*ptrA[1]).i()) * R0A0P0) * (identity - R0A0P0) * (*ptrRI[0]);
    arma::sp_mat MV1 = MF1;
    arma::sp_mat MF0 = MF1;
    arma::sp_mat MV0 = MV1;
    for(int level = numLevels-3; level >= 0; level--){
        MV1         = MV0;
        MF1         = MF0;
        rec_lambda  = arma::Col<double>(lambda.rows(level,level+1));
        rec_Nl      = arma::Col<int>(Nl.rows(level,level+1));
        rec_ml      = arma::Col<int>(ml.row(level));
        get_operators(ptrA, ptrR, ptrRI, ptrS, ptrP, rec_lambda, rec_Nl, rec_ml);
        identity = arma::speye(rec_Nl(1),rec_Nl(1));
        R0A0P0 = (*ptrRI[0]) * (*ptrA[0]) * (*ptrP[0]);
        MF0 = (*ptrP[0]) * (identity - (identity - MV1 * MF1) * arma::sp_mat(arma::mat(*ptrA[1]).i()) * R0A0P0) * (identity - R0A0P0) * (*ptrRI[0]);
        MV0 = (*ptrP[0]) * (identity - (identity - MV1) * arma::sp_mat(arma::mat(*ptrA[1]).i()) * R0A0P0) * (identity - R0A0P0) * (*ptrRI[0]);
    }
    if(theoryLevel == 1){
        // on level 1, remove the pre-/post-multiplied P_0 and R_{I_0}
        (*F_FCF) = (*ptrRI[0]) * MF0 * (*ptrP[0]);
    }else if(theoryLevel == 0){
        (*F_FCF) = MF0;
    }
    return 0;
}

/**
 *  get error propagator for F-cycle with FCF-relaxation and >= 2 grid levels (complex eigenvalues)
 */
int get_F_FCF(arma::sp_cx_mat *F_FCF, arma::Col<arma::cx_double> lambda, arma::Col<int> Nl, arma::Col<int> ml, int theoryLevel){
    // check for valid arguments
    if((theoryLevel != 0) && (theoryLevel != 1)){
        std::cout << ">>>ERROR: Error propagator only implemented for level 0 and level 1." << std::endl;
        throw;
    }
    if((lambda.n_elem != Nl.n_elem) || (lambda.n_elem != ml.n_elem+1)){
        std::cout << ">>>ERROR: Error propagator encountered invalid definition of number of levels." << std::endl;
    }
    // check if time stepper is stable
    if(arma::any(arma::abs(lambda) > constants::time_stepper_stability_limit)){
        return -1;
    }
    // get number of levels
    int numLevels = lambda.n_elem;
    // if two-level algorithm, all cycling strategies coincide
    arma::sp_cx_mat *E_F = new arma::sp_cx_mat();
    get_E_F(E_F, lambda, Nl, ml, theoryLevel);
    // get parameters for coarsest two grids
    arma::Col<arma::cx_double>  rec_lambda  = arma::Col<arma::cx_double>(lambda.rows(numLevels-2,numLevels-1));
    arma::Col<int>              rec_Nl      = arma::Col<int>(Nl.rows(numLevels-2,numLevels-1));
    arma::Col<int>              rec_ml      = arma::Col<int>(ml.row(numLevels-2));
    // pointers to all operators
    arma::sp_cx_mat *ptrA[2];
    arma::sp_cx_mat *ptrR[1];
    arma::sp_cx_mat *ptrRI[1];
    arma::sp_cx_mat *ptrS[1];
    arma::sp_cx_mat *ptrP[1];
    // compute operators
    get_operators(ptrA, ptrR, ptrRI, ptrS, ptrP, rec_lambda, rec_Nl, rec_ml);
    // get coarsest-grid correction on level n_l-2
    arma::sp_cx_mat identity = arma::sp_cx_mat(arma::speye(rec_Nl(1),rec_Nl(1)),arma::speye(rec_Nl(1),rec_Nl(1)).zeros());
    arma::sp_cx_mat R0A0P0 = (*ptrRI[0]) * (*ptrA[0]) * (*ptrP[0]);
    arma::sp_cx_mat MF1 = (*ptrP[0]) * (identity - arma::sp_cx_mat(arma::cx_mat(*ptrA[1]).i()) * R0A0P0) * (identity - R0A0P0) * (*ptrRI[0]);
    arma::sp_cx_mat MV1 = MF1;
    arma::sp_cx_mat MF0 = MF1;
    arma::sp_cx_mat MV0 = MV1;
    for(int level = numLevels-3; level >= 0; level--){
        MV1         = MV0;
        MF1         = MF0;
        rec_lambda  = arma::Col<arma::cx_double>(lambda.rows(level,level+1));
        rec_Nl      = arma::Col<int>(Nl.rows(level,level+1));
        rec_ml      = arma::Col<int>(ml.row(level));
        get_operators(ptrA, ptrR, ptrRI, ptrS, ptrP, rec_lambda, rec_Nl, rec_ml);
        identity = arma::sp_cx_mat(arma::speye(rec_Nl(1),rec_Nl(1)),arma::speye(rec_Nl(1),rec_Nl(1)).zeros());
        R0A0P0 = (*ptrRI[0]) * (*ptrA[0]) * (*ptrP[0]);
        MF0 = (*ptrP[0]) * (identity - (identity - MV1 * MF1) * arma::sp_cx_mat(arma::cx_mat(*ptrA[1]).i()) * R0A0P0) * (identity - R0A0P0) * (*ptrRI[0]);
        MV0 = (*ptrP[0]) * (identity - (identity - MV1) * arma::sp_cx_mat(arma::cx_mat(*ptrA[1]).i()) * R0A0P0) * (identity - R0A0P0) * (*ptrRI[0]);
    }
    if(theoryLevel == 1){
        // on level 1, remove the pre-/post-multiplied P_0 and R_{I_0}
        (*F_FCF) = (*ptrRI[0]) * MF0 * (*ptrP[0]);
    }else if(theoryLevel == 0){
        (*F_FCF) = MF0;
    }
    return 0;
}

/**
 *  get residual propagator for V-cycle with F-relaxation and >= 2 grid levels (real eigenvalues)
 */
int get_R_F(arma::sp_mat *R_F, arma::Col<double> lambda, arma::Col<int> Nl, arma::Col<int> ml, int theoryLevel, int cycle){
    if((theoryLevel != 0) && (theoryLevel != 1)){
        std::cout << ">>>ERROR: Error propagator only implemented for level 0 and level 1." << std::endl;
        throw;
    }
    arma::sp_mat A0 = get_Al(lambda(0), Nl(0));
    int errCode = 0;
    arma::sp_mat *E_F0 = new arma::sp_mat();
    if(cycle == mgritestimate::V_cycle){
        errCode = get_E_F(E_F0, lambda, Nl, ml, 0);
    }else if(cycle == mgritestimate::F_cycle){
        errCode = get_F_F(E_F0, lambda, Nl, ml, 0);
    }
    if(errCode == -1){
        return errCode;
    }
    // residual propagation on level 0 is computed using A*E*inv(A) because error and residual propagation are formally similar
    if(theoryLevel == 0){
        (*R_F) = A0 * (*E_F0) * arma::sp_mat(arma::mat(A0).i());
    }else{
        arma::sp_mat RI0 = get_RIl(lambda(0), Nl(0), ml(0));
        arma::sp_mat P0 = get_Pl(lambda(0), Nl(0), ml(0));
        (*R_F) = RI0 * A0 * (*E_F0) * arma::sp_mat(arma::mat(A0).i()) * RI0.st();
    }
    return errCode;
}

/**
 *  get residual propagator for V-cycle with F-relaxation and >= 2 grid levels (complex eigenvalues)
 */
int get_R_F(arma::sp_cx_mat *R_F, arma::Col<arma::cx_double> lambda, arma::Col<int> Nl, arma::Col<int> ml, int theoryLevel, int cycle){
    if((theoryLevel != 0) && (theoryLevel != 1)){
        std::cout << ">>>ERROR: Error propagator only implemented for level 0 and level 1." << std::endl;
        throw;
    }
    arma::sp_cx_mat A0 = get_Al(lambda(0), Nl(0));
    int errCode = 0;
    arma::sp_cx_mat *E_F0 = new arma::sp_cx_mat();
    if(cycle == mgritestimate::V_cycle){
        errCode = get_E_F(E_F0, lambda, Nl, ml, 0);
    }else if(cycle == mgritestimate::F_cycle){
        errCode = get_F_F(E_F0, lambda, Nl, ml, 0);
    }
    if(errCode == -1){
        return errCode;
    }
    // residual propagation on level 0 is computed using A*E*inv(A) because error and residual propagation are formally similar
    if(theoryLevel == 0){
        (*R_F) = A0 * (*E_F0) * arma::sp_cx_mat(arma::cx_mat(A0).i());
    }else{
        arma::sp_cx_mat RI0 = get_RIl(lambda(0), Nl(0), ml(0));
        arma::sp_cx_mat P0 = get_Pl(lambda(0), Nl(0), ml(0));
        (*R_F) = RI0 * A0 * (*E_F0) * arma::sp_cx_mat(arma::cx_mat(A0).i()) * RI0.st();
    }
    return errCode;
}

/**
 *  get residual propagator for V-cycle with FCF-relaxation and >= 2 grid levels (real eigenvalues)
 */
int get_R_FCF(arma::sp_mat *R_FCF, arma::Col<double> lambda, arma::Col<int> Nl, arma::Col<int> ml, int theoryLevel, int cycle){
    if((theoryLevel != 0) && (theoryLevel != 1)){
        std::cout << ">>>ERROR: Error propagator only implemented for level 0 and level 1." << std::endl;
        throw;
    }
    arma::sp_mat A0 = get_Al(lambda(0), Nl(0));
    int errCode = 0;
    arma::sp_mat *E_FCF0 = new arma::sp_mat();
    if(cycle == mgritestimate::V_cycle){
        errCode = get_E_FCF(E_FCF0, lambda, Nl, ml, 0);
    }else if(cycle == mgritestimate::F_cycle){
        errCode = get_F_FCF(E_FCF0, lambda, Nl, ml, 0);
    }
    if(errCode == -1){
        return errCode;
    }
    // residual propagation on level 0 is computed using A*E*inv(A) because error and residual propagation are formally similar
    if(theoryLevel == 0){
        (*R_FCF) = A0 * (*E_FCF0) * arma::sp_mat(arma::mat(A0).i());
    }else{
        arma::sp_mat RI0 = get_RIl(lambda(0), Nl(0), ml(0));
        arma::sp_mat P0 = get_Pl(lambda(0), Nl(0), ml(0));
        (*R_FCF) = RI0 * A0 * (*E_FCF0) * arma::sp_mat(arma::mat(A0).i()) * RI0.st();
    }
    return errCode;
}

/**
 *  get residual propagator for V-cycle with FCF-relaxation and >= 2 grid levels (complex eigenvalues)
 */
int get_R_FCF(arma::sp_cx_mat *R_FCF, arma::Col<arma::cx_double> lambda, arma::Col<int> Nl, arma::Col<int> ml, int theoryLevel, int cycle){
    if((theoryLevel != 0) && (theoryLevel != 1)){
        std::cout << ">>>ERROR: Error propagator only implemented for level 0 and level 1." << std::endl;
        throw;
    }
    arma::sp_cx_mat A0 = get_Al(lambda(0), Nl(0));
    int errCode = 0;
    arma::sp_cx_mat *E_FCF0 = new arma::sp_cx_mat();
    if(cycle == mgritestimate::V_cycle){
        errCode = get_E_FCF(E_FCF0, lambda, Nl, ml, 0);
    }else if(cycle == mgritestimate::F_cycle){
        errCode = get_F_FCF(E_FCF0, lambda, Nl, ml, 0);
    }
    if(errCode == -1){
        return errCode;
    }
    // residual propagation on level 0 is computed using A*E*inv(A) because error and residual propagation are formally similar
    if(theoryLevel == 0){
        (*R_FCF) = A0 * (*E_FCF0) * arma::sp_cx_mat(arma::cx_mat(A0).i());
    }else{
        arma::sp_cx_mat RI0 = get_RIl(lambda(0), Nl(0), ml(0));
        arma::sp_cx_mat P0 = get_Pl(lambda(0), Nl(0), ml(0));
        (*R_FCF) = RI0 * A0 * (*E_FCF0) * arma::sp_cx_mat(arma::cx_mat(A0).i()) * RI0.st();
    }
    return errCode;
}
