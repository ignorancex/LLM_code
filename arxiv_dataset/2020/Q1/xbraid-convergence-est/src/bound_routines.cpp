#include "bound_routines.hpp"

/**
 *  wrapper routine, that calls bound-dependent routine to compute bound for residual or error propagator for multilevel MGRIT algorithm (real version)
 */
void get_propagator_bound(const int bound,                  ///< requested bound, see constants.hpp
                          int theoryLevel,                  ///< time grid level, where bound is computed
                          const int cycle,                  ///< cycling strategy, see constants.hpp
                          const int relax,                  ///< relaxation scheme, see constants.hpp
                          Col<int> numberOfTimeSteps,       ///< number of time steps on all time grids
                          Col<int> coarseningFactors,       ///< temporal coarsening factors for levels 0-->1, 1-->2, etc.
                          Col<double> **lambda,             ///< eigenvalues on each time grid
                          Col<double> *&estimate            ///< on return, the estimate for all eigenvalues
                          ){
    if(cycle == mgritestimate::V_cycle){
        if((bound == mgritestimate::error_l2_upper_bound)
            || (bound == mgritestimate::error_l2_sqrt_upper_bound)
            || (bound == mgritestimate::error_l2_sqrt_expression_upper_bound)
            || (bound == mgritestimate::error_l2_tight_twogrid_upper_bound)
            || (bound == mgritestimate::error_l2_approximate_lower_bound)
            || (bound == mgritestimate::error_l2_sqrt_approximate_lower_bound)
            || (bound == mgritestimate::error_l2_tight_twogrid_lower_bound)
            || (bound == mgritestimate::error_l2_tight_twogrid_bound)
            || (bound == mgritestimate::error_l2_tight_twogrid_bound_single_iter)
            || (bound == mgritestimate::lower_bound_southworth2019_equation63)
            || (bound == mgritestimate::upper_bound_southworth2019_equation63)
            || (bound == mgritestimate::error_l2_sqrt_expression_approximate_rate)){
            get_error_l2_propagator_bound(bound, theoryLevel, cycle, relax, numberOfTimeSteps, coarseningFactors, lambda, estimate);
        }else if((bound == mgritestimate::residual_l2_upper_bound)
            || (bound == mgritestimate::residual_l2_sqrt_upper_bound)
            || (bound == mgritestimate::residual_l2_lower_bound)){
            get_residual_l2_propagator_bound(bound, theoryLevel, cycle, relax, numberOfTimeSteps, coarseningFactors, lambda, estimate);
        }else{
            cout << ">>>ERROR: Unknown bound type " << bound << " for V-cycle." << endl << endl;
            throw;
        }
    }else if(cycle == mgritestimate::F_cycle){
        if((bound == mgritestimate::error_l2_upper_bound)
            || (bound == mgritestimate::error_l2_sqrt_expression_upper_bound)
            || (bound == mgritestimate::lower_bound_southworth2019_equation63)
            || (bound == mgritestimate::upper_bound_southworth2019_equation63)
            || (bound == mgritestimate::error_l2_sqrt_upper_bound)){
                get_error_l2_propagator_bound(bound, theoryLevel, cycle, relax, numberOfTimeSteps, coarseningFactors, lambda, estimate);
        }else if((bound == mgritestimate::residual_l2_upper_bound)
            || (bound == mgritestimate::residual_l2_sqrt_upper_bound)){
                get_residual_l2_propagator_bound(bound, theoryLevel, cycle, relax, numberOfTimeSteps, coarseningFactors, lambda, estimate);
        }else{
            cout << ">>>ERROR Unknown bound type " << bound << " for F-cycle." << endl << endl;
            throw;
        }
    }else{
        cout << ">>>ERROR: Unknown cycling strategy " << cycle << " for bound type " << bound << endl << endl;
        throw;
    }
}

/**
 *  wrapper routine, that calls bound-dependent routine to compute bound for residual or error propagator for multilevel MGRIT algorithm (complex version)
 */
void get_propagator_bound(const int bound,                  ///< requested bound, see constants.hpp
                          int theoryLevel,                  ///< time grid level, where bound is computed
                          const int cycle,                  ///< cycling strategy, see constants.hpp
                          const int relax,                  ///< relaxation scheme, see constants.hpp
                          Col<int> numberOfTimeSteps,       ///< number of time steps on all time grids
                          Col<int> coarseningFactors,       ///< temporal coarsening factors for levels 0-->1, 1-->2, etc.
                          Col<cx_double> **lambda,             ///< eigenvalues on each time grid
                          Col<double> *&estimate            ///< on return, the estimate for all eigenvalues
                          ){
    if(cycle == mgritestimate::V_cycle){
        if((bound == mgritestimate::error_l2_upper_bound)
            || (bound == mgritestimate::error_l2_sqrt_upper_bound)
            || (bound == mgritestimate::error_l2_sqrt_expression_upper_bound)
            || (bound == mgritestimate::error_l2_tight_twogrid_upper_bound)
            || (bound == mgritestimate::error_l2_approximate_lower_bound)
            || (bound == mgritestimate::error_l2_sqrt_approximate_lower_bound)
            || (bound == mgritestimate::error_l2_tight_twogrid_lower_bound)
            || (bound == mgritestimate::error_l2_tight_twogrid_bound)
            || (bound == mgritestimate::error_l2_tight_twogrid_bound_single_iter)
            || (bound == mgritestimate::lower_bound_southworth2019_equation63)
            || (bound == mgritestimate::upper_bound_southworth2019_equation63)
            || (bound == mgritestimate::error_l2_sqrt_expression_approximate_rate)){
            get_error_l2_propagator_bound(bound, theoryLevel, cycle, relax, numberOfTimeSteps, coarseningFactors, lambda, estimate);
        }else if((bound == mgritestimate::residual_l2_upper_bound)
            || (bound == mgritestimate::residual_l2_sqrt_upper_bound)
            || (bound == mgritestimate::residual_l2_lower_bound)){
            get_residual_l2_propagator_bound(bound, theoryLevel, cycle, relax, numberOfTimeSteps, coarseningFactors, lambda, estimate);
        }else{
            cout << ">>>ERROR: Unknown bound type " << bound << " for V-cycle." << endl << endl;
            throw;
        }
    }else if(cycle == mgritestimate::F_cycle){
        if((bound == mgritestimate::error_l2_upper_bound)
            || (bound == mgritestimate::error_l2_sqrt_expression_upper_bound)
            || (bound == mgritestimate::lower_bound_southworth2019_equation63)
            || (bound == mgritestimate::upper_bound_southworth2019_equation63)
            || (bound == mgritestimate::error_l2_sqrt_upper_bound)){
                get_error_l2_propagator_bound(bound, theoryLevel, cycle, relax, numberOfTimeSteps, coarseningFactors, lambda, estimate);
        }else if((bound == mgritestimate::residual_l2_upper_bound)
            || (bound == mgritestimate::residual_l2_sqrt_upper_bound)){
                get_residual_l2_propagator_bound(bound, theoryLevel, cycle, relax, numberOfTimeSteps, coarseningFactors, lambda, estimate);
        }else{
            cout << ">>>ERROR Unknown bound type " << bound << " for F-cycle." << endl << endl;
            throw;
        }
    }else{
        cout << ">>>ERROR: Unknown cycling strategy " << cycle << " for bound type " << bound << endl << endl;
        throw;
    }
}

/**
 *  compute bound for error propagator for multilevel MGRIT algorithm (real version)
 */
void get_error_l2_propagator_bound(const int bound,                 ///< requested bound, see constants.hpp
                                   int theoryLevel,                 ///< time grid level, where bound is computed
                                   const int cycle,                 ///< cycling strategy, see constants.hpp
                                   const int relax,                 ///< relaxation scheme, see constants.hpp
                                   Col<int> numberOfTimeSteps,      ///< number of time steps on all time grids
                                   Col<int> coarseningFactors,      ///< temporal coarsening factors for levels 0-->1, 1-->2, etc.
                                   Col<double> **lambda,            ///< eigenvalues on each time grid
                                   Col<double> *&estimate           ///< on return, the estimate for all eigenvalues
                                   ){
    int errCode = 0;
    bool pinvSuccess = true;
    // check MPI environment
    int world_rank;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // get number of eigenvalues and levels
    int numberOfSamples = (*lambda[0]).n_elem;
    int numberOfLevels  = numberOfTimeSteps.n_elem;
    estimate = new Col<double>;
    (*estimate).set_size(numberOfSamples);
    (*estimate).fill(-2.0);
    // setup function pointer for error propagator
    int (*get_E)(arma::sp_mat *, arma::Col<double>, arma::Col<int>, arma::Col<int>, int);
    switch(relax){
        case mgritestimate::F_relaxation:{
            if(cycle == mgritestimate::V_cycle){
                get_E = get_E_F;
            }else if(cycle == mgritestimate::F_cycle){
                get_E = get_F_F;
            }
            break;
        }
        case mgritestimate::FCF_relaxation:{
            if(cycle == mgritestimate::V_cycle){
                get_E = get_E_FCF;
            }else if(cycle == mgritestimate::F_cycle){
                get_E = get_F_FCF;
            }
            break;
        }
        default:{
            if(bound == mgritestimate::error_l2_sqrt_expression_upper_bound){
                // fine
            }
            else{
                cout << ">>>ERROR: Only F- and FCF-relaxation are implemented for this bound." << endl;
                throw;
            }
        }
    }
    // get local eigenvalue index range on rank
    if(numberOfSamples < world_size){
        cout << ">>>ERROR: Sample size is too small for number of MPI processes" << endl;
        throw;
    }
    // get index range for local MPI process
    int samplesRankStartIdx;
    int samplesRankStopIdx;
    get_samples_index_range(numberOfSamples, samplesRankStartIdx, samplesRankStopIdx);
    // get estimate
    switch(bound){
        case mgritestimate::error_l2_upper_bound:{
            errCode = 0;
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                // for a given spatial mode, get eigenvalues for all levels
                Col<double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                sp_mat *E = new sp_mat();
                // get error propagator
                errCode = get_E(E, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel);
                // compute bound only if time stepper is stable, i.e., \f$\lambda_k < 1\f$
                if(errCode == -1){
                    (*estimate)(evalIdx) = -1.0;
                    continue;
                }
                (*estimate)(evalIdx) = norm(mat(*E), 2);
            }
            break;
        }
        case mgritestimate::error_l2_approximate_lower_bound:{
            errCode = 0;
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                // for a given spatial mode, get eigenvalues for all levels
                Col<double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                sp_mat *E = new sp_mat();
                mat pinvE;
                pinvSuccess = true;
                // get error propagator
                errCode = get_E(E, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel);
                // compute bound only if time stepper is stable, i.e., \f$\lambda_k < 1\f$
                if(errCode == -1){
                    (*estimate)(evalIdx) = -1.0;
                    continue;
                }
                // compute pseudo-inverse of error propagator
                pinvSuccess = pinv(pinvE, mat(*E));
                if(!pinvSuccess){
                    cout << ">>>ERROR: Computing pseudo-inverse of error propagator failed." << endl;
                    throw;
                }
                // l2-norm of pseudo-inverse bounds error propagator from below
                /// \todo should we check for division by zero?
                (*estimate)(evalIdx) = 1.0 / norm(pinvE, 2);
            }
            break;
        }
        case mgritestimate::error_l2_sqrt_upper_bound:{
            errCode = 0;
            double norm1 = 0.0;
            double normInf = 0.0;
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                // for a given spatial mode, get eigenvalues for all levels
                Col<double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                sp_mat *E = new sp_mat();
                // get error propagator
                errCode = get_E(E, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel);
                // compute bound only if time stepper is stable, i.e., \f$\lambda_k < 1\f$
                if(errCode == -1){
                    (*estimate)(evalIdx) = -1.0;
                    continue;
                }
                /// \todo Seems to be required for Armadillo 6.500.5. Can skip mat() conversion for later versions?
                norm1   = norm(mat(*E), 1);
                normInf = norm(mat(*E), "inf");
                (*estimate)(evalIdx) = sqrt(norm1 * normInf);
            }
            break;
        }
        case mgritestimate::error_l2_sqrt_expression_upper_bound:{
            if((cycle == mgritestimate::F_cycle) && (numberOfLevels == 3)){
                if(world_rank == 0){
                    cout << ">>>INFO-RANK-0: Running experimental implementation of analytic 3-grid F-cycle bound" << endl << endl;
                }
            }
            errCode = 0;
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                // for a given spatial mode, get eigenvalues for all levels
                Col<double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                // evaluate expression
                (*estimate)(evalIdx) = get_error_l2_sqrt_expression_upper_bound(cycle, relax, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel);
            }
            break;
        }
        case mgritestimate::error_l2_sqrt_approximate_lower_bound:{
            if(cycle == mgritestimate::F_cycle){
                cout << endl << ">>>ERROR: error_l2_sqrt_approximate_lower_bound not implemented for F-cycle." << endl << endl;
                throw;
            }
            if(numberOfLevels < 3){
                cout << endl << ">>>ERROR: error_l2_sqrt_approximate_lower_bound only implemenented for three or more levels." << endl << endl;
                throw;
            }
            errCode             = 0;
            double normEB1      = 0.0;
            double normEBInf    = 0.0;
            double normB1       = 0.0;
            double normBInf     = 0.0;
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                // for a given spatial mode, get eigenvalues for all levels
                Col<double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                sp_mat *E   = new sp_mat();
                sp_mat *B   = new sp_mat();
                sp_mat *EB  = new sp_mat();
                // get error propagator
                errCode = get_E(E, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel);
                // compute bound only if time stepper is stable, i.e., \f$\lambda_k < 1\f$
                if(errCode == -1){
                    (*estimate)(evalIdx) = -1.0;
                    continue;
                }
                // get error propagator, intermediate level contributions
                errCode = get_E_F_intermediate(B, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel);
                // compute coarse-grid correction part
                (*EB) = (*E) + (*B);
                // compute norms for approximate lower bound
                normEB1     = norm((*EB), 1);
                normEBInf   = norm((*EB), "inf");
                normB1      = norm((*B), 1);
                normBInf    = norm((*B), "inf");
                // compute approximate lower bound
                (*estimate)(evalIdx) = abs(sqrt(normEB1 * normEBInf) - sqrt(normB1 * normBInf));
            }
            break;
        }
        case mgritestimate::error_l2_tight_twogrid_upper_bound:{
            errCode = 0;
            Col<double> estimateLevels(numberOfLevels-1);
            Col<int> cf(1);
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                estimateLevels.fill(-1.0);
                // for a given spatial mode, get eigenvalues for all levels
                Col<double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                // evaluate expression for all subsequent levels, then get max
                for(int level = 0; level < numberOfLevels-1; level++){
                    cf.fill(coarseningFactors(level));
                    estimateLevels(level) = get_error_l2_tight_twogrid_upper_bound(relax, Col<double>(lambda_k.subvec(level,level+1)),
                         Col<int>(numberOfTimeSteps.subvec(level,level+1)), cf, theoryLevel);
                }
                (*estimate)(evalIdx) = arma::max(estimateLevels);
            }
            break;
        }
        case mgritestimate::error_l2_tight_twogrid_lower_bound:{
            errCode = 0;
            Col<double> estimateLevels(numberOfLevels-1);
            Col<int> cf(1);
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                estimateLevels.fill(-1.0);
                // for a given spatial mode, get eigenvalues for all levels
                Col<double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                // evaluate expression for all subsequent levels, then get max
                for(int level = 0; level < numberOfLevels-1; level++){
                    cf.fill(coarseningFactors(level));
                    estimateLevels(level) = get_error_l2_tight_twogrid_lower_bound(relax, Col<double>(lambda_k.subvec(level,level+1)),
                         Col<int>(numberOfTimeSteps.subvec(level,level+1)), cf, theoryLevel);
                }
                (*estimate)(evalIdx) = arma::max(estimateLevels);
            }
            break;
        }
        case mgritestimate::error_l2_tight_twogrid_bound:{
            errCode = 0;
            Col<double> estimateLevels(numberOfLevels-1);
            Col<int> cf(1);
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                estimateLevels.fill(-1.0);
                // for a given spatial mode, get eigenvalues for all levels
                Col<double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                // evaluate expression for all subsequent levels, then get max
                for(int level = 0; level < numberOfLevels-1; level++){
                    cf.fill(coarseningFactors(level));
                    estimateLevels(level) = get_error_l2_tight_twogrid_bound(relax, Col<double>(lambda_k.subvec(level,level+1)),
                         Col<int>(numberOfTimeSteps.subvec(level,level+1)), cf, theoryLevel);
                }
                (*estimate)(evalIdx) = arma::max(estimateLevels);
            }
            break;
        }
        case mgritestimate::error_l2_tight_twogrid_bound_single_iter:{
            errCode = 0;
            Col<double> estimateLevels(numberOfLevels-1);
            Col<int> cf(1);
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                estimateLevels.fill(-1.0);
                // for a given spatial mode, get eigenvalues for all levels
                Col<double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                // evaluate expression for all subsequent levels, then get max
                for(int level = 0; level < numberOfLevels-1; level++){
                    cf.fill(coarseningFactors(level));
                    estimateLevels(level) = get_error_l2_tight_twogrid_bound_single_iter(relax, Col<double>(lambda_k.subvec(level,level+1)),
                         Col<int>(numberOfTimeSteps.subvec(level,level+1)), cf, theoryLevel);
                }
                (*estimate)(evalIdx) = arma::max(estimateLevels);
            }
            break;
        }
        case mgritestimate::lower_bound_southworth2019_equation63:{
            errCode = 0;
            Col<double> estimateLevels(numberOfLevels-1);
            Col<int> cf(1);
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                estimateLevels.fill(-1.0);
                // for a given spatial mode, get eigenvalues for all levels
                Col<double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                // evaluate expression for all subsequent levels, then get max
                for(int level = 0; level < numberOfLevels-1; level++){
                    cf.fill(coarseningFactors(level));
                    estimateLevels(level) = get_lower_bound_southworth2019_equation63(relax, Col<double>(lambda_k.subvec(level,level+1)),
                         Col<int>(numberOfTimeSteps.subvec(level,level+1)), cf, theoryLevel);
                }
                (*estimate)(evalIdx) = arma::max(estimateLevels);
            }
            break;
        }
        case mgritestimate::upper_bound_southworth2019_equation63:{
            errCode = 0;
            Col<double> estimateLevels(numberOfLevels-1);
            Col<int> cf(1);
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                estimateLevels.fill(-1.0);
                // for a given spatial mode, get eigenvalues for all levels
                Col<double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                // evaluate expression for all subsequent levels, then get max
                for(int level = 0; level < numberOfLevels-1; level++){
                    cf.fill(coarseningFactors(level));
                    estimateLevels(level) = get_upper_bound_southworth2019_equation63(relax, Col<double>(lambda_k.subvec(level,level+1)),
                         Col<int>(numberOfTimeSteps.subvec(level,level+1)), cf, theoryLevel);
                }
                (*estimate)(evalIdx) = arma::max(estimateLevels);
            }
            break;
        }
        case mgritestimate::error_l2_sqrt_expression_approximate_rate:{
            errCode = 0;
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                // for a given spatial mode, get eigenvalues for all levels
                Col<double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                // evaluate expression
                (*estimate)(evalIdx) = get_error_l2_sqrt_expression_approximate_rate(relax, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel);
            }
            break;
        }
        default:{
            cout << ">>>ERROR: Bound " << bound << " not implemented" << endl;
            throw;
        }
    }
    // communicate data
    communicateBounds(estimate, numberOfSamples, samplesRankStartIdx, samplesRankStopIdx);
}

/**
 *  compute bound for error propagator for multilevel MGRIT algorithm (complex version)
 */
void get_error_l2_propagator_bound(const int bound,                 ///< requested bound, see constants.hpp
                                   int theoryLevel,                 ///< time grid level, where bound is computed
                                   const int cycle,                 ///< cycling strategy, see constants.hpp
                                   const int relax,                 ///< relaxation scheme, see constants.hpp
                                   Col<int> numberOfTimeSteps,      ///< number of time steps on all time grids
                                   Col<int> coarseningFactors,      ///< temporal coarsening factors for levels 0-->1, 1-->2, etc.
                                   Col<cx_double> **lambda,         ///< eigenvalues on each time grid
                                   Col<double> *&estimate           ///< on return, the estimate for all eigenvalues
                                   ){
    int errCode = 0;
    bool pinvSuccess = true;
    // check MPI environment
    int world_rank;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // get number of eigenvalues and levels
    int numberOfSamples = (*lambda[0]).n_elem;
    int numberOfLevels  = numberOfTimeSteps.n_elem;
    estimate = new Col<double>;
    (*estimate).set_size(numberOfSamples);
    (*estimate).fill(-2.0);
    // setup function pointer for error propagator
    int (*get_E)(arma::sp_cx_mat *, arma::Col<arma::cx_double>, arma::Col<int>, arma::Col<int>, int);
    switch(relax){
        case mgritestimate::F_relaxation:{
            if(cycle == mgritestimate::V_cycle){
                get_E = get_E_F;
            }else if(cycle == mgritestimate::F_cycle){
                get_E = get_F_F;
            }
            break;
        }
        case mgritestimate::FCF_relaxation:{
            if(cycle == mgritestimate::V_cycle){
                get_E = get_E_FCF;
            }else if(cycle == mgritestimate::F_cycle){
                get_E = get_F_FCF;
            }
            break;
        }
        default:{
            if(bound == mgritestimate::error_l2_sqrt_expression_upper_bound){
                // fine
            }
            else{
                cout << ">>>ERROR: Only F- and FCF-relaxation are implemented for this bound." << endl;
                throw;
            }
        }
    }
    // get local eigenvalue index range on rank
    if(numberOfSamples < world_size){
        cout << ">>>ERROR: Sample size is too small for number of MPI processes" << endl;
        throw;
    }
    // get index range for local MPI process
    int samplesRankStartIdx;
    int samplesRankStopIdx;
    get_samples_index_range(numberOfSamples, samplesRankStartIdx, samplesRankStopIdx);
    // get estimate
    switch(bound){
        case mgritestimate::error_l2_upper_bound:{
            errCode = 0;
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                // for a given spatial mode, get eigenvalues for all levels
                Col<cx_double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                sp_cx_mat *E = new sp_cx_mat();
                // get error propagator
                errCode = get_E(E, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel);
                // compute bound only if time stepper is stable, i.e., \f$\lambda_k < 1\f$
                if(errCode == -1){
                    (*estimate)(evalIdx) = -1.0;
                    continue;
                }
                (*estimate)(evalIdx) = norm(cx_mat(*E), 2);
            }
            break;
        }
        case mgritestimate::error_l2_approximate_lower_bound:{
            errCode = 0;
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                // for a given spatial mode, get eigenvalues for all levels
                Col<cx_double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                sp_cx_mat *E = new sp_cx_mat();
                cx_mat pinvE;
                pinvSuccess = true;
                // get error propagator
                errCode = get_E(E, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel);
                // compute bound only if time stepper is stable, i.e., \f$\lambda_k < 1\f$
                if(errCode == -1){
                    (*estimate)(evalIdx) = -1.0;
                    continue;
                }
                // compute pseudo-inverse of error propagator
                pinvSuccess = pinv(pinvE, cx_mat(*E));
                if(!pinvSuccess){
                    cout << ">>>ERROR: Computing pseudo-inverse of error propagator failed." << endl;
                    throw;
                }
                // l2-norm of pseudo-inverse bounds error propagator from below
                /// \todo should we check for division by zero?
                (*estimate)(evalIdx) = 1.0 / norm(pinvE, 2);
            }
            break;
        }
        case mgritestimate::error_l2_sqrt_upper_bound:{
            errCode = 0;
            double norm1 = 0.0;
            double normInf = 0.0;
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                // for a given spatial mode, get eigenvalues for all levels
                Col<cx_double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                sp_cx_mat *E = new sp_cx_mat();
                // get error propagator
                errCode = get_E(E, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel);
                // compute bound only if time stepper is stable, i.e., \f$\lambda_k < 1\f$
                if(errCode == -1){
                    (*estimate)(evalIdx) = -1.0;
                    continue;
                }
                /// \todo Seems to be required for Armadillo 6.500.5. Can skip cx_mat() conversion for later versions?
                norm1   = norm(cx_mat(*E), 1);
                normInf = norm(cx_mat(*E), "inf");
                (*estimate)(evalIdx) = sqrt(norm1 * normInf);
            }
            break;
        }
        case mgritestimate::error_l2_sqrt_expression_upper_bound:{
            errCode = 0;
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                // for a given spatial mode, get eigenvalues for all levels
                Col<cx_double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                // evaluate expression
                (*estimate)(evalIdx) = get_error_l2_sqrt_expression_upper_bound(cycle, relax, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel);
            }
            break;
        }
        case mgritestimate::error_l2_sqrt_approximate_lower_bound:{
            if(cycle == mgritestimate::F_cycle){
                cout << endl << ">>>ERROR: error_l2_sqrt_approximate_lower_bound not implemented for F-cycle." << endl << endl;
                throw;
            }
            if(numberOfLevels < 3){
                cout << endl << ">>>ERROR: error_l2_sqrt_approximate_lower_bound only implemenented for three or more levels." << endl << endl;
                throw;
            }
            errCode             = 0;
            double normEB1      = 0.0;
            double normEBInf    = 0.0;
            double normB1       = 0.0;
            double normBInf     = 0.0;
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                // for a given spatial mode, get eigenvalues for all levels
                Col<cx_double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                sp_cx_mat *E    = new sp_cx_mat();
                sp_cx_mat *B    = new sp_cx_mat();
                sp_cx_mat *EB   = new sp_cx_mat();
                // get error propagator
                errCode = get_E(E, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel);
                // compute bound only if time stepper is stable, i.e., \f$\lambda_k < 1\f$
                if(errCode == -1){
                    (*estimate)(evalIdx) = -1.0;
                    continue;
                }
                // get error propagator, intermediate level contributions
                errCode = get_E_F_intermediate(B, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel);
                // compute coarse-grid correction part
                (*EB) = (*E) + (*B);
                // compute norms for approximate lower bound
                normEB1     = arma::norm(arma::abs(*EB), 1);
                normEBInf   = arma::norm(arma::abs(*EB), "inf");
                normB1      = arma::norm(arma::abs(*B), 1);
                normBInf    = arma::norm(arma::abs(*B), "inf");
                // compute approximate lower bound
                (*estimate)(evalIdx) = abs(sqrt(normEB1 * normEBInf) - sqrt(normB1 * normBInf));
            }
            break;
        }
        case mgritestimate::error_l2_tight_twogrid_upper_bound:{
            errCode = 0;
            Col<double> estimateLevels(numberOfLevels-1);
            Col<int> cf(1);
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                estimateLevels.fill(-1.0);
                // for a given spatial mode, get eigenvalues for all levels
                Col<cx_double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                // evaluate expression for all subsequent levels, then get max
                for(int level = 0; level < numberOfLevels-1; level++){
                    cf.fill(coarseningFactors(level));
                    estimateLevels(level) = get_error_l2_tight_twogrid_upper_bound(relax, Col<cx_double>(lambda_k.subvec(level,level+1)),
                         Col<int>(numberOfTimeSteps.subvec(level,level+1)), cf, theoryLevel);
                }
                (*estimate)(evalIdx) = arma::max(estimateLevels);
            }
            break;
        }
        case mgritestimate::error_l2_tight_twogrid_lower_bound:{
            errCode = 0;
            Col<double> estimateLevels(numberOfLevels-1);
            Col<int> cf(1);
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                estimateLevels.fill(-1.0);
                // for a given spatial mode, get eigenvalues for all levels
                Col<cx_double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                // evaluate expression for all subsequent levels, then get max
                for(int level = 0; level < numberOfLevels-1; level++){
                    cf.fill(coarseningFactors(level));
                    estimateLevels(level) = get_error_l2_tight_twogrid_lower_bound(relax, Col<cx_double>(lambda_k.subvec(level,level+1)),
                         Col<int>(numberOfTimeSteps.subvec(level,level+1)), cf, theoryLevel);
                }
                (*estimate)(evalIdx) = arma::max(estimateLevels);
            }
            break;
        }
        case mgritestimate::error_l2_tight_twogrid_bound:{
            errCode = 0;
            Col<double> estimateLevels(numberOfLevels-1);
            Col<int> cf(1);
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                estimateLevels.fill(-1.0);
                // for a given spatial mode, get eigenvalues for all levels
                Col<cx_double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                // evaluate expression for all subsequent levels, then get max
                for(int level = 0; level < numberOfLevels-1; level++){
                    cf.fill(coarseningFactors(level));
                    estimateLevels(level) = get_error_l2_tight_twogrid_bound(relax, Col<cx_double>(lambda_k.subvec(level,level+1)),
                         Col<int>(numberOfTimeSteps.subvec(level,level+1)), cf, theoryLevel);
                }
                (*estimate)(evalIdx) = arma::max(estimateLevels);
            }
            break;
        }
        case mgritestimate::error_l2_tight_twogrid_bound_single_iter:{
            errCode = 0;
            Col<double> estimateLevels(numberOfLevels-1);
            Col<int> cf(1);
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                estimateLevels.fill(-1.0);
                // for a given spatial mode, get eigenvalues for all levels
                Col<cx_double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                // evaluate expression for all subsequent levels, then get max
                for(int level = 0; level < numberOfLevels-1; level++){
                    cf.fill(coarseningFactors(level));
                    estimateLevels(level) = get_error_l2_tight_twogrid_bound_single_iter(relax, Col<cx_double>(lambda_k.subvec(level,level+1)),
                         Col<int>(numberOfTimeSteps.subvec(level,level+1)), cf, theoryLevel);
                }
                (*estimate)(evalIdx) = arma::max(estimateLevels);
            }
            break;
        }
        case mgritestimate::lower_bound_southworth2019_equation63:{
            errCode = 0;
            Col<double> estimateLevels(numberOfLevels-1);
            Col<int> cf(1);
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                estimateLevels.fill(-1.0);
                // for a given spatial mode, get eigenvalues for all levels
                Col<cx_double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                // evaluate expression for all subsequent levels, then get max
                for(int level = 0; level < numberOfLevels-1; level++){
                    cf.fill(coarseningFactors(level));
                    estimateLevels(level) = get_lower_bound_southworth2019_equation63(relax, Col<cx_double>(lambda_k.subvec(level,level+1)),
                         Col<int>(numberOfTimeSteps.subvec(level,level+1)), cf, theoryLevel);
                }
                (*estimate)(evalIdx) = arma::max(estimateLevels);
            }
            break;
        }
        case mgritestimate::upper_bound_southworth2019_equation63:{
            errCode = 0;
            Col<double> estimateLevels(numberOfLevels-1);
            Col<int> cf(1);
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                estimateLevels.fill(-1.0);
                // for a given spatial mode, get eigenvalues for all levels
                Col<cx_double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                // evaluate expression for all subsequent levels, then get max
                for(int level = 0; level < numberOfLevels-1; level++){
                    cf.fill(coarseningFactors(level));
                    estimateLevels(level) = get_upper_bound_southworth2019_equation63(relax, Col<cx_double>(lambda_k.subvec(level,level+1)),
                         Col<int>(numberOfTimeSteps.subvec(level,level+1)), cf, theoryLevel);
                }
                (*estimate)(evalIdx) = arma::max(estimateLevels);
            }
            break;
        }
        case mgritestimate::error_l2_sqrt_expression_approximate_rate:{
            errCode = 0;
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                // for a given spatial mode, get eigenvalues for all levels
                Col<cx_double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                // evaluate expression
                (*estimate)(evalIdx) = get_error_l2_sqrt_expression_approximate_rate(relax, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel);
            }
            break;
        }
        default:{
            cout << ">>>ERROR: Bound " << bound << " not implemented" << endl;
            throw;
        }
    }
    // communicate data
    communicateBounds(estimate, numberOfSamples, samplesRankStartIdx, samplesRankStopIdx);
}

/**
 *  compute bound for residual propagator for multilevel MGRIT algorithm (real version)
 */
void get_residual_l2_propagator_bound(const int bound,                  ///< requested bound, see constants.hpp
                                      int theoryLevel,                  ///< time grid level, where bound is computed
                                      const int cycle,                  ///< cycling strategy, see constants.hpp
                                      const int relax,                  ///< relaxation scheme, see constants.hpp
                                      Col<int> numberOfTimeSteps,       ///< number of time steps on all time grids
                                      Col<int> coarseningFactors,       ///< temporal coarsening factors for levels 0-->1, 1-->2, etc.
                                      Col<double> **lambda,             ///< eigenvalues on each time grid
                                      Col<double> *&estimate            ///< on return, the estimate for all eigenvalues
                                      ){
    int errCode = 0;
    bool pinvSuccess = true;
    // check MPI environment
    int world_rank;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // get number of eigenvalues and levels
    int numberOfSamples = (*lambda[0]).n_elem;
    int numberOfLevels  = numberOfTimeSteps.n_elem;
    estimate = new Col<double>;
    (*estimate).set_size(numberOfSamples);
    (*estimate).fill(-2.0);
    // setup function pointer for error propagator
    int (*get_R)(arma::sp_mat *, arma::Col<double>, arma::Col<int>, arma::Col<int>, int, int);
    switch(relax){
        case mgritestimate::F_relaxation:{
            get_R = get_R_F;
            break;
        }
        case mgritestimate::FCF_relaxation:{
            get_R = get_R_FCF;
            break;
        }
        default:{
            if(bound == mgritestimate::error_l2_sqrt_expression_upper_bound){
                // fine
            }
            else{
                cout << ">>>ERROR: Only F- and FCF-relaxation are implemented for this bound." << endl;
                throw;
            }
        }
    }
    // get local eigenvalue index range on rank
    if(numberOfSamples < world_size){
        cout << ">>>ERROR: Sample size is too small for number of MPI processes." << endl;
        throw;
    }
    // get index range for local MPI process
    int samplesRankStartIdx;
    int samplesRankStopIdx;
    get_samples_index_range(numberOfSamples, samplesRankStartIdx, samplesRankStopIdx);
    // get estimate
    switch(bound){
        case mgritestimate::residual_l2_upper_bound:{
            errCode = 0;
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                // for a given spatial mode, get eigenvalues for all levels
                Col<double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                sp_mat *R = new sp_mat();
                // get residual propagator
                errCode = get_R(R, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel, cycle);
                // compute bound only if time stepper is stable, i.e., \f$\lambda_k < 1\f$
                if(errCode == -1){
                    (*estimate)(evalIdx) = -1.0;
                    continue;
                }
                (*estimate)(evalIdx) = norm(mat(*R), 2);
            }
            break;
        }
        case mgritestimate::residual_l2_lower_bound:{
            errCode = 0;
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                // for a given spatial mode, get eigenvalues for all levels
                Col<double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                sp_mat *R = new sp_mat();
                mat pinvR;
                pinvSuccess = true;
                // get residual propagator
                errCode = get_R(R, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel, cycle);
                // compute bound only if time stepper is stable, i.e., \f$\lambda_k < 1\f$
                if(errCode == -1){
                    (*estimate)(evalIdx) = -1.0;
                    continue;
                }
                // compute pseudo-inverse of residual propagator
                pinvSuccess = pinv(pinvR, mat(*R));
                if(!pinvSuccess){
                    cout << ">>>ERROR: Computing pseudo-inverse of residual propagator failed." << endl;
                    throw;
                }
                // l2-norm of pseudo-inverse bounds residual propagator from below
                /// \todo should we check for division by zero?
                (*estimate)(evalIdx) = 1.0 / norm(pinvR, 2);
            }
            break;
        }
        case mgritestimate::residual_l2_sqrt_upper_bound:{
            errCode = 0;
            double norm1 = 0.0;
            double normInf = 0.0;
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                // for a given spatial mode, get eigenvalues for all levels
                Col<double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                sp_mat *R = new sp_mat();
                // get residual propagator
                errCode = get_R(R, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel, cycle);
                // compute bound only if time stepper is stable, i.e., \f$\lambda_k < 1\f$
                if(errCode == -1){
                    (*estimate)(evalIdx) = -1.0;
                    continue;
                }
                /// \todo Seems to be required for Armadillo 6.500.5. Can skip mat() conversion for later versions?
                norm1   = norm(mat(*R), 1);
                normInf = norm(mat(*R), "inf");
                (*estimate)(evalIdx) = sqrt(norm1 * normInf);
            }
            break;
        }
        default:{
            cout << ">>>ERROR: Bound " << bound << " not implemented" << endl;
            throw;
        }
    }
    // communicate data
    communicateBounds(estimate, numberOfSamples, samplesRankStartIdx, samplesRankStopIdx);
}

/**
 *  compute bound for residual propagator for multilevel MGRIT algorithm (complex version)
 */
void get_residual_l2_propagator_bound(const int bound,                  ///< requested bound, see constants.hpp
                                      int theoryLevel,                  ///< time grid level, where bound is computed
                                      const int cycle,                  ///< cycling strategy, see constants.hpp
                                      const int relax,                  ///< relaxation scheme, see constants.hpp
                                      Col<int> numberOfTimeSteps,       ///< number of time steps on all time grids
                                      Col<int> coarseningFactors,       ///< temporal coarsening factors for levels 0-->1, 1-->2, etc.
                                      Col<cx_double> **lambda,          ///< eigenvalues on each time grid
                                      Col<double> *&estimate            ///< on return, the estimate for all eigenvalues
                                      ){
    int errCode = 0;
    bool pinvSuccess = true;
    // check MPI environment
    int world_rank;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // get number of eigenvalues and levels
    int numberOfSamples = (*lambda[0]).n_elem;
    int numberOfLevels  = numberOfTimeSteps.n_elem;
    estimate = new Col<double>;
    (*estimate).set_size(numberOfSamples);
    (*estimate).fill(-2.0);
    // setup function pointer for residual propagator
    int (*get_R)(arma::sp_cx_mat *, arma::Col<arma::cx_double>, arma::Col<int>, arma::Col<int>, int, int);
    switch(relax){
        case mgritestimate::F_relaxation:{
            get_R = get_R_F;
            break;
        }
        case mgritestimate::FCF_relaxation:{
            get_R = get_R_FCF;
            break;
        }
        default:{
            if(bound == mgritestimate::error_l2_sqrt_expression_upper_bound){
                // fine
            }
            else{
                cout << ">>>ERROR: Only F- and FCF-relaxation are implemented for this bound." << endl;
                throw;
            }
        }
    }
    // get local eigenvalue index range on rank
    if(numberOfSamples < world_size){
        cout << ">>>ERROR: Sample size is too small for number of MPI processes" << endl;
        throw;
    }
    // get index range for local MPI process
    int samplesRankStartIdx;
    int samplesRankStopIdx;
    get_samples_index_range(numberOfSamples, samplesRankStartIdx, samplesRankStopIdx);
    // get estimate
    switch(bound){
        case mgritestimate::residual_l2_upper_bound:{
            errCode = 0;
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                // for a given spatial mode, get eigenvalues for all levels
                Col<cx_double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                sp_cx_mat *R = new sp_cx_mat();
                // get residual propagator
                errCode = get_R(R, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel, cycle);
                // compute bound only if time stepper is stable, i.e., \f$\lambda_k < 1\f$
                if(errCode == -1){
                    (*estimate)(evalIdx) = -1.0;
                    continue;
                }
                (*estimate)(evalIdx) = norm(cx_mat(*R), 2);
            }
            break;
        }
        case mgritestimate::residual_l2_lower_bound:{
            errCode = 0;
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                // for a given spatial mode, get eigenvalues for all levels
                Col<cx_double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                sp_cx_mat *R = new sp_cx_mat();
                cx_mat pinvR;
                pinvSuccess = true;
                // get residual propagator
                errCode = get_R(R, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel, cycle);
                // compute bound only if time stepper is stable, i.e., \f$\lambda_k < 1\f$
                if(errCode == -1){
                    (*estimate)(evalIdx) = -1.0;
                    continue;
                }
                // compute pseudo-inverse of residual propagator
                pinvSuccess = pinv(pinvR, cx_mat(*R));
                if(!pinvSuccess){
                    cout << ">>>ERROR: Computing pseudo-inverse of residual propagator failed." << endl;
                    throw;
                }
                // l2-norm of pseudo-inverse bounds residual propagator from below
                /// \todo should we check for division by zero?
                (*estimate)(evalIdx) = 1.0 / norm(pinvR, 2);
            }
            break;
        }
        case mgritestimate::residual_l2_sqrt_upper_bound:{
            errCode = 0;
            double norm1 = 0.0;
            double normInf = 0.0;
            for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
                // for a given spatial mode, get eigenvalues for all levels
                Col<cx_double> lambda_k(numberOfLevels);
                for(int level = 0; level < numberOfLevels; level++){
                    lambda_k(level) = (*lambda[level])(evalIdx);
                }
                sp_cx_mat *R = new sp_cx_mat();
                // get residual propagator
                errCode = get_R(R, lambda_k, numberOfTimeSteps, coarseningFactors, theoryLevel, cycle);
                // compute bound only if time stepper is stable, i.e., \f$\lambda_k < 1\f$
                if(errCode == -1){
                    (*estimate)(evalIdx) = -1.0;
                    continue;
                }
                /// \todo Seems to be required for Armadillo 6.500.5. Can skip cx_mat() conversion for later versions?
                norm1   = norm(cx_mat(*R), 1);
                normInf = norm(cx_mat(*R), "inf");
                (*estimate)(evalIdx) = sqrt(norm1 * normInf);
            }
            break;
        }
        default:{
            cout << ">>>ERROR: Bound " << bound << " not implemented" << endl;
            throw;
        }
    }
    // communicate data
    communicateBounds(estimate, numberOfSamples, samplesRankStartIdx, samplesRankStopIdx);
}

/**
 *  Compute tight two-grid upper bound using expression.
 *
 *  Note: Evaluation is reasonably cheap, so let's just wrap the complex equivalent.
 */
double get_error_l2_tight_twogrid_upper_bound(int r,                   ///< number of FC relaxation steps
                                     Col<double> lambda,      ///< eigenvalues of \f$\Phi_l\f$
                                     Col<int> N,              ///< number of time steps on each grid level
                                     Col<int> m,              ///< coarsening factors between all grid levels
                                     int theoryLevel          ///< expression for error propagator on grid level
                                     ){
    Col<cx_double> lambdac(lambda, 0.0*lambda);
    double val = get_error_l2_tight_twogrid_upper_bound(r, lambdac, N, m, theoryLevel);
    return val;
}

/**
 *  Compute tight two-grid upper bound using expression (complex version)
 */
// note: we use std::abs/std::pow here instead of arma::abs/arma::pow because it allows mixed complex/real operands
double get_error_l2_tight_twogrid_upper_bound(int r,                   ///< number of FC relaxation steps
                                     Col<cx_double> lambda,   ///< eigenvalues of \f$\Phi_l\f$
                                     Col<int> N,              ///< number of time steps on each grid level
                                     Col<int> m,              ///< coarsening factors between all grid levels
                                     int theoryLevel          ///< expression for error propagator on grid level
                                     ){
    // check if time stepper is stable
    if(arma::any(arma::abs(lambda) > constants::time_stepper_stability_limit)){
        return -1.0;
    }
    // sanity check
    int numberOfLevels  = N.n_elem;
    if(numberOfLevels != 2){
        cout << ">>>ERROR: tight_twogrid_upper_bound only works for two time grids" << endl;
    }
    // evaluate expression depending on relaxation scheme
    double val = -2.0;
    switch(r){
        case mgritestimate::F_relaxation:{
            val = std::abs(std::pow(lambda(0), m(0)) - lambda(1))
                    / std::sqrt(1.0 + std::pow(std::abs(lambda(1)), 2.0) + 2.0 * std::abs(lambda(1)) * std::cos(N(1) * constants::pi / (N(1) + 0.5)));
            break;
        }
        case mgritestimate::FCF_relaxation:{
            val = std::abs(std::pow(lambda(0), m(0)))
                    * std::abs(std::pow(lambda(0), m(0)) - lambda(1))
                    / std::sqrt(1.0 + std::pow(std::abs(lambda(1)), 2.0) + 2.0 * std::abs(lambda(1)) * std::cos(N(1) * constants::pi / (N(1) + 0.5)));
            break;
        }
        default:{
            cout << ">>>ERROR: tight_twogrid_upper_bound only implemented for F- and FCF-relaxation" << endl;
            throw;
        }
    }
    val *= std::sqrt((1.0 - std::pow(std::abs(lambda(0)), 2.0*m(0))) / (1.0 - std::pow(std::abs(lambda(0)), 2.0)));
    return val;
}

/**
 *  Compute tight two-grid lower bound using expression.
 *
 *  Note: Evaluation is reasonably cheap, so let's just wrap the complex equivalent.
 */
double get_error_l2_tight_twogrid_lower_bound(int r,                   ///< number of FC relaxation steps
                                     Col<double> lambda,      ///< eigenvalues of \f$\Phi_l\f$
                                     Col<int> N,              ///< number of time steps on each grid level
                                     Col<int> m,              ///< coarsening factors between all grid levels
                                     int theoryLevel          ///< expression for error propagator on grid level
                                     ){
    Col<cx_double> lambdac(lambda, 0.0*lambda);
    double val = get_error_l2_tight_twogrid_lower_bound(r, lambdac, N, m, theoryLevel);
    return val;
}

/**
 *  Compute tight two-grid lower bound using expression (complex version).
 */
// note: we use std::abs/std::pow here instead of arma::abs/arma::pow because it allows mixed complex/real operands
double get_error_l2_tight_twogrid_lower_bound(int r,                   ///< number of FC relaxation steps
                                     Col<cx_double> lambda,   ///< eigenvalues of \f$\Phi_l\f$
                                     Col<int> N,              ///< number of time steps on each grid level
                                     Col<int> m,              ///< coarsening factors between all grid levels
                                     int theoryLevel          ///< expression for error propagator on grid level
                                     ){
    // check if time stepper is stable
    if(arma::any(arma::abs(lambda) > constants::time_stepper_stability_limit)){
        return -1.0;
    }
    // sanity check
    int numberOfLevels  = N.n_elem;
    if(numberOfLevels != 2){
        cout << ">>>ERROR: get_error_l2_tight_twogrid_lower_bound only works for two time grids" << endl;
    }
    // evaluate expression depending on relaxation scheme
    double val = -2.0;
    switch(r){
        case mgritestimate::F_relaxation:{
            val = std::abs(std::pow(lambda(0), m(0)) - lambda(1))
                    / std::sqrt(1.0 + std::pow(std::abs(lambda(1)), 2.0) + 2.0 * std::abs(lambda(1)) * std::cos(N(1) * constants::pi / (N(1) + 1.0)));
            break;
        }
        case mgritestimate::FCF_relaxation:{
            val = std::abs(std::pow(lambda(0), m(0)))
                    * std::abs(std::pow(lambda(0), m(0)) - lambda(1))
                    / std::sqrt(1.0 + std::pow(std::abs(lambda(1)), 2.0) + 2.0 * std::abs(lambda(1)) * std::cos(N(1) * constants::pi / (N(1) + 1.0)));
            break;
        }
        default:{
            cout << ">>>ERROR: get_error_l2_tight_twogrid_lower_bound only implemented for F- and FCF-relaxation" << endl;
            throw;
        }
    }
    val *= std::sqrt((1.0 - std::pow(std::abs(lambda(0)), 2.0*m(0))) / (1.0 - std::pow(std::abs(lambda(0)), 2.0)));
    return val;
}

/**
 *  Compute tight two-grid bound using expression. See Theorem 13 in [Southworth (2019)].
 *
 *  Note: Evaluation is reasonably cheap, so let's just wrap the complex equivalent.
 */
double get_error_l2_tight_twogrid_bound(int r,                      ///< number of FC relaxation steps
                                     Col<double> lambda,      ///< eigenvalues of \f$\Phi_l\f$
                                     Col<int> N,              ///< number of time steps on each grid level
                                     Col<int> m,              ///< coarsening factors between all grid levels
                                     int theoryLevel          ///< expression for error propagator on grid level
                                     ){
    Col<cx_double> lambdac(lambda, 0.0*lambda);
    double val = get_error_l2_tight_twogrid_bound(r, lambdac, N, m, theoryLevel);
    return val;
}

/**
 *  Compute tight two-grid bound using expression (complex version). See Theorem 13 in [Southworth (2019)].
 */
// note: we use std::abs/std::pow here instead of arma::abs/arma::pow because it allows mixed complex/real operands
double get_error_l2_tight_twogrid_bound(int r,                   ///< number of FC relaxation steps
                                     Col<cx_double> lambda,   ///< eigenvalues of \f$\Phi_l\f$
                                     Col<int> N,              ///< number of time steps on each grid level
                                     Col<int> m,              ///< coarsening factors between all grid levels
                                     int theoryLevel          ///< expression for error propagator on grid level
                                     ){
    // check if time stepper is stable
    if(arma::any(arma::abs(lambda) > constants::time_stepper_stability_limit)){
        return -1.0;
    }
    // sanity check
    int numberOfLevels  = N.n_elem;
    if(numberOfLevels != 2){
        cout << ">>>ERROR: get_error_l2_tight_twogrid_bound only works for two time grids" << endl;
    }
    // evaluate expression depending on relaxation scheme
    double val = -2.0;
    switch(r){
        case mgritestimate::F_relaxation:{
            val = std::abs(std::pow(lambda(0), m(0)) - lambda(1))
                    / (1.0 - std::abs(lambda(1)));
            break;
        }
        case mgritestimate::FCF_relaxation:{
            val = std::abs(std::pow(lambda(0), m(0)))
                    * std::abs(std::pow(lambda(0), m(0)) - lambda(1))
                    / (1.0 - std::abs(lambda(1)));
            break;
        }
        default:{
            cout << ">>>ERROR: get_error_l2_tight_twogrid_bound only implemented for F- and FCF-relaxation" << endl;
            throw;
        }
    }
    return val;
}

/**
 *  Compute tight two-grid bound using expression. See Corollary 2.4 in [Southworth, Mitchell, Hessenthaler (in preparation)].
 *
 *  Note: Evaluation is reasonably cheap, so let's just wrap the complex equivalent.
 */
double get_error_l2_tight_twogrid_bound_single_iter(int r,                      ///< number of FC relaxation steps
                                                    Col<double> lambda,      ///< eigenvalues of \f$\Phi_l\f$
                                                    Col<int> N,              ///< number of time steps on each grid level
                                                    Col<int> m,              ///< coarsening factors between all grid levels
                                                    int theoryLevel          ///< expression for error propagator on grid level
                                                    ){
    Col<cx_double> lambdac(lambda, 0.0*lambda);
    double val = get_error_l2_tight_twogrid_bound_single_iter(r, lambdac, N, m, theoryLevel);
    return val;
}

/**
 *  Compute tight two-grid bound using expression (complex version). See Corollary 2.4 in [Southworth, Mitchell, Hessenthaler (in preparation)].
 */
// note: we use std::abs/std::pow here instead of arma::abs/arma::pow because it allows mixed complex/real operands
double get_error_l2_tight_twogrid_bound_single_iter(int r,                   ///< number of FC relaxation steps
                                                    Col<cx_double> lambda,   ///< eigenvalues of \f$\Phi_l\f$
                                                    Col<int> N,              ///< number of time steps on each grid level
                                                    Col<int> m,              ///< coarsening factors between all grid levels
                                                    int theoryLevel          ///< expression for error propagator on grid level
                                                    ){
    // check if time stepper is stable
    if(arma::any(arma::abs(lambda) > constants::time_stepper_stability_limit)){
        return -1.0;
    }
    // sanity check
    int numberOfLevels  = N.n_elem;
    if(numberOfLevels != 2){
        cout << ">>>ERROR: get_error_l2_tight_twogrid_bound_single_iter only works for two time grids" << endl;
    }
    // evaluate expression depending on relaxation scheme
    double val = -2.0;
    switch(r){
        case mgritestimate::F_relaxation:{
            val = std::abs(std::pow(lambda(0), m(0)) - lambda(1))
                    / (1.0 - std::abs(lambda(1)));
            break;
        }
        case mgritestimate::FCF_relaxation:{
            val = std::abs(std::pow(lambda(0), m(0)))
                    * std::abs(std::pow(lambda(0), m(0)) - lambda(1))
                    / (1.0 - std::abs(lambda(1)));
            break;
        }
        default:{
            cout << ">>>ERROR: get_error_l2_tight_twogrid_bound_single_iter only implemented for F- and FCF-relaxation" << endl;
            throw;
        }
    }
    val *= std::sqrt((1.0 - std::pow(std::abs(lambda(0)), 2.0*m(0))) / (1.0 - std::pow(std::abs(lambda(0)), 2.0)));
    return val;
}

/**
 *  Compute lower bound from Equation (63) in [Southworth (2019): Necessary conditions and tight two-level convergence bounds for Parareal and multigrid reduction in time].
 *
 *  Note: Evaluation is reasonably cheap, so let's just wrap the complex equivalent.
 */
double get_lower_bound_southworth2019_equation63(int r,                   ///< number of FC relaxation steps
                                                 Col<double> lambda,      ///< eigenvalues of \f$\Phi_l\f$
                                                 Col<int> N,              ///< number of time steps on each grid level
                                                 Col<int> m,              ///< coarsening factors between all grid levels
                                                 int theoryLevel          ///< expression for error propagator on grid level
                                                 ){
    Col<cx_double> lambdac(lambda, 0.0*lambda);
    double val = get_lower_bound_southworth2019_equation63(r, lambdac, N, m, theoryLevel);
    return val;
}

/**
 *  Compute lower bound from Equation (63) in [Southworth (2019): Necessary conditions and tight two-level convergence bounds for Parareal and multigrid reduction in time] (complex version).
 */
// note: we use std::abs/std::pow here instead of arma::abs/arma::pow because it allows mixed complex/real operands
double get_lower_bound_southworth2019_equation63(int r,                   ///< number of FC relaxation steps
                                                 Col<cx_double> lambda,   ///< eigenvalues of \f$\Phi_l\f$
                                                 Col<int> N,              ///< number of time steps on each grid level
                                                 Col<int> m,              ///< coarsening factors between all grid levels
                                                 int theoryLevel          ///< expression for error propagator on grid level
                                                 ){
    // check if time stepper is stable
    if(arma::any(arma::abs(lambda) > constants::time_stepper_stability_limit)){
        return -1.0;
    }
    // sanity check
    int numberOfLevels  = N.n_elem;
    if(numberOfLevels != 2){
        cout << ">>>ERROR: get_lower_bound_southworth2019_equation63 only works for two time grids" << endl;
    }
    // evaluate expression depending on relaxation scheme
    double val = -2.0;
    switch(r){
        case mgritestimate::F_relaxation:{
            val = std::abs(std::pow(lambda(0), m(0)) - lambda(1))
                    / std::sqrt(std::pow(1.0 - std::abs(lambda(1)), 2.0) + std::abs(lambda(1)) * constants::pi * constants::pi / (N(1) * N(1)));
            break;
        }
        case mgritestimate::FCF_relaxation:{
            val = std::abs(std::pow(lambda(0), m(0)))
                    * std::abs(std::pow(lambda(0), m(0)) - lambda(1))
                    / std::sqrt(std::pow(1.0 - std::abs(lambda(1)), 2.0) + std::abs(lambda(1)) * constants::pi * constants::pi / (N(1) * N(1)));
            break;
        }
        default:{
            cout << ">>>ERROR: get_lower_bound_southworth2019_equation63 only implemented for F- and FCF-relaxation" << endl;
            throw;
        }
    }
    return val;
}

/**
 *  Compute upper bound from Equation (63) in [Southworth (2019): Necessary conditions and tight two-level convergence bounds for Parareal and multigrid reduction in time].
 *
 *  Note: Evaluation is reasonably cheap, so let's just wrap the complex equivalent.
 */
double get_upper_bound_southworth2019_equation63(int r,                      ///< number of FC relaxation steps
                                                 Col<double> lambda,      ///< eigenvalues of \f$\Phi_l\f$
                                                 Col<int> N,              ///< number of time steps on each grid level
                                                 Col<int> m,              ///< coarsening factors between all grid levels
                                                 int theoryLevel          ///< expression for error propagator on grid level
                                                 ){
    Col<cx_double> lambdac(lambda, 0.0*lambda);
    double val = get_upper_bound_southworth2019_equation63(r, lambdac, N, m, theoryLevel);
    return val;
}

/**
 *  Compute upper bound from Equation (63) in [Southworth (2019): Necessary conditions and tight two-level convergence bounds for Parareal and multigrid reduction in time] (complex version).
 */
// note: we use std::abs/std::pow here instead of arma::abs/arma::pow because it allows mixed complex/real operands
double get_upper_bound_southworth2019_equation63(int r,                   ///< number of FC relaxation steps
                                                 Col<cx_double> lambda,   ///< eigenvalues of \f$\Phi_l\f$
                                                 Col<int> N,              ///< number of time steps on each grid level
                                                 Col<int> m,              ///< coarsening factors between all grid levels
                                                 int theoryLevel          ///< expression for error propagator on grid level
                                                 ){
    // check if time stepper is stable
    if(arma::any(arma::abs(lambda) > constants::time_stepper_stability_limit)){
        return -1.0;
    }
    // sanity check
    int numberOfLevels  = N.n_elem;
    if(numberOfLevels != 2){
        cout << ">>>ERROR: get_upper_bound_southworth2019_equation63 only works for two time grids" << endl;
    }
    // evaluate expression depending on relaxation scheme
    double val = -2.0;
    switch(r){
        case mgritestimate::F_relaxation:{
            val = std::abs(std::pow(lambda(0), m(0)) - lambda(1))
                    / std::sqrt(std::pow(1.0 - std::abs(lambda(1)), 2.0) + std::abs(lambda(1)) * constants::pi * constants::pi / (6.0 * N(1) * N(1)));
            break;
        }
        case mgritestimate::FCF_relaxation:{
            val = std::abs(std::pow(lambda(0), m(0)))
                    * std::abs(std::pow(lambda(0), m(0)) - lambda(1))
                    / std::sqrt(std::pow(1.0 - std::abs(lambda(1)), 2.0) + std::abs(lambda(1)) * constants::pi * constants::pi / (6.0 * N(1) * N(1)));
            break;
        }
        default:{
            cout << ">>>ERROR: get_upper_bound_southworth2019_equation63 only implemented for F- and FCF-relaxation" << endl;
            throw;
        }
    }
    return val;
}

/**
 *  Compute \f$ \sqrt{\| E \|_1 \| E \|_\infty} \f$ using expression.
 *
 *  Note: Evaluation is reasonably cheap, so let's just wrap the complex equivalent.
 */
// note: we use std::abs/std::pow here instead of arma::abs/arma::pow because it allows mixed complex/real operands
double get_error_l2_sqrt_expression_upper_bound(int cycle,    ///< cycling strategy
                                     int r,                   ///< number of FC relaxation steps
                                     Col<double> lambda,      ///< eigenvalues of \f$\Phi_l\f$
                                     Col<int> N,              ///< number of time steps on each grid level
                                     Col<int> m,              ///< coarsening factors between all grid levels
                                     int theoryLevel          ///< expression for error propagator on grid level
                                     ){
    Col<cx_double> lambdac(lambda, 0.0*lambda);
    double val = get_error_l2_sqrt_expression_upper_bound(cycle, r, lambdac, N, m, theoryLevel);
    return val;
}

/**
 *  Compute \f$ \sqrt{\| E \|_1 \| E \|_\infty} \f$ using expression (complex version).
 */
// note: we use std::abs/std::pow here instead of arma::abs/arma::pow because it allows mixed complex/real operands
double get_error_l2_sqrt_expression_upper_bound(int cycle,      ///< cycling strategy
                                       int r,                   ///< number of FC relaxation steps
                                       Col<cx_double> lambda,   ///< eigenvalues of \f$\Phi_l\f$
                                       Col<int> N,              ///< number of time steps on each grid level
                                       Col<int> m,              ///< coarsening factors between all grid levels
                                       int theoryLevel          ///< expression for error propagator on grid level
                                       ){
    // check if time stepper is stable
    if(arma::any(arma::abs(lambda) > constants::time_stepper_stability_limit)){
        return -1.0;
    }
    // evaluate expression depending on number of time grids and relaxation scheme
    int numberOfLevels  = N.n_elem;
    double val = -2.0;
    if(numberOfLevels == 2){
        if(theoryLevel == 1){
            val = std::abs(std::pow(lambda(0), m(0)) - lambda(1))
                    * std::pow(std::abs(lambda(0)), r*m(0))
                    * (1.0 - std::pow(std::abs(lambda(1)), N(1)-1-r)) / (1.0 - std::abs(lambda(1)));
        }else{
            cout << ">>>ERROR: error_l2_sqrt_expression_upper_bound not implemented for two levels on level " << theoryLevel << endl;
            throw;
        }
    }else if(numberOfLevels == 3){
        if(cycle == mgritestimate::F_cycle){
            switch(r){
                case mgritestimate::F_relaxation:{
                    if(theoryLevel == 0){
                        cout << ">>>ERROR: error_l2_sqrt_expression_upper_bound not implemented for three level F-cycles with F-relaxation on level " << theoryLevel << endl;
                        throw;
                    }
                    double norm1;
                    double normInf;
                    double sumC;
                    double sumF;
                    double sum1;
                    double sum2;
                    double sum3;
                    double sum4;
                    double b;
                    b               = N(1) - m(1) * N(2) + 3 * m(1) - 3;
                    // C-point column sum
                    sumC            = 0.0;
                    for(int r = 0; r <= b; r++){
                        sumC       += std::abs(lambda(1) - std::pow(lambda(0), m(0))) * std::pow(std::abs(lambda(1)), r);
                    }
                    sum1            = 0.0;
                    for(int r = 0; r <= m(1)-1; r++){
                        sum1       += std::pow(std::abs(lambda(1)), r);
                    }
                    sum2            = 0.0;
                    for(int q = 1; q <= N(2)-3; q++){
                        sum2       += std::pow(std::abs(lambda(2)), q-1)
                                        * std::abs(q * 1.0 * std::pow(lambda(0), m(0)) * std::pow(lambda(1), b+1)
                                                   - (q + 1.0) * std::pow(lambda(1), m(1)) * lambda(2)
                                                   - (q - 1.0) * std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1) * lambda(2)
                                                   + q * 1.0 * std::pow(lambda(2), 2.0)
                                            );
                    }
                    sumC           += sum1 * sum2
                                        + std::pow(std::abs(lambda(2)), N(2)-3)
                                        * std::abs((N(2) - 2.0) * std::pow(lambda(0), m(0)) * std::pow(lambda(1), b+1)
                                                   - (N(2) - 1.0) * std::pow(lambda(1), m(1)) * lambda(2)
                                                   - (N(2) - 3.0) * std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1) * lambda(2)
                                                   + (N(2) - 2.0) * std::pow(lambda(2), 2.0)
                                        );
                    norm1           = sumC;
                    // F-point column sum
                    for(int j = 1; j <= m(1)-1; j++){
                        sum1        = 0.0;
                        for(int r = 0; r <= b+m(1)-j; r++){
                            sum1   += std::pow(std::abs(lambda(1)), r);
                        }
                        sum2        = 0.0;
                        for(int r = 0; r <= m(1)-1; r++){
                            sum2   += std::pow(std::abs(lambda(1)), r+m(1)-1-j);
                        }
                        sum3        = 0.0;
                        for(int q = 1; q <= N(2)-4; q++){
                            sum3   += std::pow(std::abs(lambda(2)), q)
                                        * std::abs(q * 1.0 * lambda(2)
                                                   - (q + 1.0) * std::pow(lambda(1), m(1))
                                                  );
                        }
                        sumF        = sum1
                                        + sum2 * sum3
                                        + std::pow(std::abs(lambda(1)), m(1)-1-j)
                                        * std::pow(std::abs(lambda(2)), N(2)-3)
                                        * std::abs((N(2) - 3.0) * lambda(2)
                                                   - (N(2) - 2.0) * std::pow(lambda(1), m(1))
                                                  );
                        sumF       *= std::abs(lambda(1) - std::pow(lambda(0), m(0)));
                        norm1       = std::max(sumF, norm1);
                    }
                    // C-point row sum
                    sumC            = 0.0;
                    for(int r = 0; r <= b; r++){
                        sumC       += std::abs(lambda(1) - std::pow(lambda(0), m(0))) * std::pow(std::abs(lambda(1)), r);
                    }
                    sum1            = 0.0;
                    for(int q = 1; q <= N(2)-2; q++){
                        sum1       += std::pow(std::abs(lambda(2)), q-1)
                                        * std::abs(q * 1.0 * std::pow(lambda(0), m(0)) * std::pow(lambda(1), b+1)
                                                   - (q + 1.0) * std::pow(lambda(1), m(1)) * lambda(2)
                                                   - (q - 1.0) * std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1) * lambda(2)
                                                   + q * 1.0 * std::pow(lambda(2), 2.0)
                                                  );
                    }
                    sum2            = 0.0;
                    for(int r = 0; r <= m(1)-2; r++){
                        sum2       += std::pow(std::abs(lambda(1)), r);
                    }
                    sum3            = 0.0;
                    for(int q = 1; q <= N(2)-3; q++){
                        sum3       += std::pow(std::abs(lambda(2)), q)
                                        * std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                        * std::abs(q * 1.0 * lambda(2)
                                                   - (q + 1.0) * std::pow(lambda(1), m(1))
                                                  );
                    }
                    sumC           += sum1 + sum2 * sum3;
                    normInf         = sumC;
                    // F-point row sum
                    for(int j = N(1)-m(1)+1; j <= N(1)-1; j++){
                        sum1        = 0.0;
                        for(int r = 0; r <= b+m(1)+j-N(1); r++){
                            sum1   += std::pow(std::abs(lambda(1)), r)
                                        * std::abs(lambda(1) - std::pow(lambda(0), m(0)));
                        }
                        sum2        = 0.0;
                        for(int q = 1; q <= N(2)-3; q++){
                            sum2   += std::pow(std::abs(lambda(1)), m(1)-1+j-(N(1)-1))
                                        * std::pow(std::abs(lambda(2)), q-1)
                                        * std::abs(q * 1.0 * std::pow(lambda(0), m(0)) * std::pow(lambda(1), b+1)
                                                   - (q + 1.0) * std::pow(lambda(1), m(1)) * lambda(2)
                                                   - (q - 1.0) * std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1) * lambda(2)
                                                   + q * 1.0 * std::pow(lambda(2), 2.0)
                                                  );
                        }
                        sum3        = 0.0;
                        for(int r = b-1-m(1)+2; r <= b-1; r++){
                            sum3   += std::pow(std::abs(lambda(1)), r-(N(1)-1)+j);
                        }
                        sum4        = 0.0;
                        for(int q = 1; q <= N(2)-4; q++){
                            sum4   += std::pow(std::abs(lambda(2)), q)
                                        * std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                        * std::abs(q * 1.0 * lambda(2)
                                                   - (q + 1.0) * std::pow(lambda(1), m(1))
                                                  );
                        }
                        sumF        = sum1 + sum2 + sum3 * sum4;
                        normInf     = std::max(sumF, normInf);
                    }
                    val             = sqrt(norm1 * normInf);
                    break;
                }
                case mgritestimate::FCF_relaxation:{
                    cout << ">>>ERROR: error_l2_sqrt_expression_upper_bound for F-cycles only implemented for three-grid with F-relaxation" << endl;
                    throw;
                    break;
                }
                default:{
                    cout << ">>>ERROR: error_l2_sqrt_expression_upper_bound for F-cycles only implemented for three-grid with F-relaxation" << endl;
                    throw;
                }
            }
        }else{
            switch(r){
                /*  On level 1, we have the following points:
                *
                *  C0      F1      F2      F3     C1
                *  |-------|-------|-------|-------|
                *  |-------------------------------|
                *
                */
                case mgritestimate::F_relaxation:{
                    double norm1;
                    double normInf;
                    double colSumF;
                    double rowSumF;
                    if(theoryLevel == 0){
                        // C-point column sum
                        norm1   = std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1))
                                    * (1.0 - std::pow(std::abs(lambda(2)), N(2)-2)) / (1.0 - std::abs(lambda(2)))
                                    * (1.0 - std::pow(std::abs(lambda(1)), m(1)))   / (1.0 - std::abs(lambda(1)))
                                    * (1.0 - std::pow(std::abs(lambda(0)), m(0)))   / (1.0 - std::abs(lambda(0)))
                                    + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                    * (1.0 - std::pow(std::abs(lambda(1)), m(1)-1)) / (1.0 - std::abs(lambda(1)))
                                    * (1.0 - std::pow(std::abs(lambda(0)), m(0)))   / (1.0 - std::abs(lambda(0)))
                                    + std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1))
                                    * std::pow(std::abs(lambda(2)), N(2)-2);
                        // F-point column sum
                        for(int j = 1; j <= m(1)-1; j++){
                            colSumF = (std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                        * (1.0 - std::pow(std::abs(lambda(2)), N(2)-2)) / (1.0 - std::abs(lambda(2)))
                                        * (1.0 - std::pow(std::abs(lambda(1)), m(1)))   / (1.0 - std::abs(lambda(1)))
                                        * (1.0 - std::pow(std::abs(lambda(0)), m(0)))   / (1.0 - std::abs(lambda(0)))
                                        + std::pow(std::abs(lambda(2)), N(2)-2)
                                        * std::abs(lambda(1) - std::pow(lambda(0), m(0)))) * std::pow(std::abs(lambda(1)), j-1)
                                        + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                        * (1.0 - std::pow(std::abs(lambda(1)), m(1)-2)) / (1.0 - std::abs(lambda(1)))
                                        * (1.0 - std::pow(std::abs(lambda(0)), m(0)))   / (1.0 - std::abs(lambda(0)));
                            norm1   = std::max(colSumF, norm1);
                        }
                    }else if(theoryLevel == 1){
                        // C-point column sum
                        norm1   = std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1))
                                    * (1.0 - std::pow(std::abs(lambda(2)), N(2)-2)) / (1.0 - std::abs(lambda(2)))
                                    * (1.0 - std::pow(std::abs(lambda(1)), m(1)))   / (1.0 - std::abs(lambda(1)))
                                    + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                    * (1.0 - std::pow(std::abs(lambda(1)), m(1)-1)) / (1.0 - std::abs(lambda(1)))
                                    + std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1))
                                    * std::pow(std::abs(lambda(2)), N(2)-2);
                        // F-point column sum
                        for(int j = 1; j <= m(1)-1; j++){
                            colSumF = (std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                        * (1.0 - std::pow(std::abs(lambda(2)), N(2)-2)) / (1.0 - std::abs(lambda(2)))
                                        * (1.0 - std::pow(std::abs(lambda(1)), m(1)))   / (1.0 - std::abs(lambda(1)))
                                        + std::pow(std::abs(lambda(2)), N(2)-2)
                                        * std::abs(lambda(1) - std::pow(lambda(0), m(0)))) * std::pow(std::abs(lambda(1)), j-1)
                                        + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                        * (1.0 - std::pow(std::abs(lambda(1)), m(1)-2)) / (1.0 - std::abs(lambda(1)));
                            norm1   = std::max(colSumF, norm1);
                        }
                    }else{
                        cout << ">>>ERROR: error_l2_sqrt_expression_upper_bound not implemented for three levels (F-relaxation) on level " << theoryLevel << endl;
                        throw;
                    }
                    // C-point row sum
                    normInf  = std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1))
                                * (1.0 - std::pow(std::abs(lambda(2)), N(2)-1)) / (1.0 - std::abs(lambda(2)))
                                + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                * (1.0 - std::pow(std::abs(lambda(2)), N(2)-1)) / (1.0 - std::abs(lambda(2)))
                                * (1.0 - std::pow(std::abs(lambda(1)), m(1)-1))   / (1.0 - std::abs(lambda(1)));
                    for(int j = 1; j <= m(1)-1; j++){
                        rowSumF = std::pow(std::abs(lambda(1)), j)
                                    * (1.0 - std::pow(std::abs(lambda(2)), N(2)-2)) / (1.0 - std::abs(lambda(2)))
                                    * (
                                    std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1))
                                    + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                    * (1.0 - std::pow(std::abs(lambda(1)), m(1)-1)) / (1.0 - std::abs(lambda(1)))
                                    )
                                    + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                    * (1.0 - std::pow(std::abs(lambda(1)), j)) / (1.0 - std::abs(lambda(1)));
                        normInf = std::max(rowSumF, normInf);
                    }
                    val = sqrt(norm1 * normInf);
                    break;
                }
                case mgritestimate::FCF_relaxation:{
                    if(cycle == mgritestimate::F_cycle){
                        cout << ">>>ERROR: error_l2_sqrt_expression_upper_bound and F-cycles only implemented for three-grid with F-relaxation" << endl;
                        throw;
                    }
                    double norm1;
                    double normInf;
                    double colSumF;
                    double rowSumF;
                    if(theoryLevel == 0){
                        // C-point column sum
                        norm1   = std::pow(std::abs(lambda(0)), m(0))
                                    * std::pow(std::abs(lambda(1)), m(1)-1)
                                    * std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                    * (1.0 - std::pow(std::abs(lambda(2)), N(2)-3)) / (1.0 - std::abs(lambda(2)))
                                    * (1.0 - std::pow(std::abs(lambda(1)), m(1)))   / (1.0 - std::abs(lambda(1)))
                                    * (1.0 - std::pow(std::abs(lambda(0)), m(0)))   / (1.0 - std::abs(lambda(0)))
                                    + std::pow(std::abs(lambda(0)), m(0))
                                    * std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                    * (1.0 - std::pow(std::abs(lambda(1)), m(1)-1)) / (1.0 - std::abs(lambda(1)))
                                    * (1.0 - std::pow(std::abs(lambda(0)), m(0)))   / (1.0 - std::abs(lambda(0)))
                                    + std::pow(std::abs(lambda(0)), m(0))
                                    * std::pow(std::abs(lambda(1)), m(1)-1)
                                    * std::pow(std::abs(lambda(2)), N(2)-3)
                                    * std::abs(lambda(1) - std::pow(lambda(0), m(0)));
                        // F-point column sum
                        colSumF = std::pow(std::abs(lambda(0)), m(0))
                                    * std::abs(lambda(1))
                                    * std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1))
                                    * (1.0 - std::pow(std::abs(lambda(1)), m(1)))   / (1.0 - std::abs(lambda(1)))
                                    * (1.0 - std::pow(std::abs(lambda(2)), N(2)-3)) / (1.0 - std::abs(lambda(2)))
                                    * (1.0 - std::pow(std::abs(lambda(0)), m(0)))   / (1.0 - std::abs(lambda(0)))
                                    + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                    * std::pow(std::abs(lambda(0)), m(0))
                                    * (1.0 - std::pow(std::abs(lambda(1)), m(1)))   / (1.0 - std::abs(lambda(1)))
                                    * (1.0 - std::pow(std::abs(lambda(0)), m(0)))   / (1.0 - std::abs(lambda(0)))
                                    + std::pow(std::abs(lambda(0)), m(0))
                                    * std::abs(lambda(1))
                                    * std::pow(std::abs(lambda(2)), N(2)-3)
                                    * std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1));
                        norm1   = std::max(colSumF, norm1);
                        for(int j = 1; j <= m(1)-2; j++){
                            colSumF = std::pow(std::abs(lambda(0)), m(0))
                                        * std::pow(std::abs(lambda(1)), j)
                                        * std::abs(lambda(1) - std::pow(lambda(0), m(0))) * (
                                        (1.0 - std::pow(std::abs(lambda(2)), N(2)-2)) / (1.0 - std::abs(lambda(2)))
                                        * (1.0 - std::pow(std::abs(lambda(1)), m(1)))   / (1.0 - std::abs(lambda(1)))
                                        * (1.0 - std::pow(std::abs(lambda(0)), m(0)))   / (1.0 - std::abs(lambda(0)))
                                        + std::pow(std::abs(lambda(2)), N(2)-2)
                                        + (1.0 - std::pow(std::abs(lambda(1)), j))      / (1.0 - std::abs(lambda(1)))
                                        * (1.0 - std::pow(std::abs(lambda(0)), m(0)))   / (1.0 - std::abs(lambda(0)))
                                        );
                            norm1   = std::max(colSumF, norm1);
                        }
                    }else if(theoryLevel == 1){
                        // C-point column sum
                        norm1   = std::pow(std::abs(lambda(0)), m(0))
                                    * std::pow(std::abs(lambda(1)), m(1)-1)
                                    * std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                    * (1.0 - std::pow(std::abs(lambda(2)), N(2)-3)) / (1.0 - std::abs(lambda(2)))
                                    * (1.0 - std::pow(std::abs(lambda(1)), m(1)))   / (1.0 - std::abs(lambda(1)))
                                    + std::pow(std::abs(lambda(0)), m(0))
                                    * std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                    * (1.0 - std::pow(std::abs(lambda(1)), m(1)-1)) / (1.0 - std::abs(lambda(1)))
                                    + std::pow(std::abs(lambda(0)), m(0))
                                    * std::pow(std::abs(lambda(1)), m(1)-1)
                                    * std::pow(std::abs(lambda(2)), N(2)-3)
                                    * std::abs(lambda(1) - std::pow(lambda(0), m(0)));
                        // F-point column sum
                        colSumF = std::pow(std::abs(lambda(0)), m(0))
                                    * std::abs(lambda(1))
                                    * std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1))
                                    * (1.0 - std::pow(std::abs(lambda(1)), m(1)))   / (1.0 - std::abs(lambda(1)))
                                    * (1.0 - std::pow(std::abs(lambda(2)), N(2)-3)) / (1.0 - std::abs(lambda(2)))
                                    + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                    * std::pow(std::abs(lambda(0)), m(0))
                                    * (1.0 - std::pow(std::abs(lambda(1)), m(1)))   / (1.0 - std::abs(lambda(1)))
                                    + std::pow(std::abs(lambda(0)), m(0))
                                    * std::abs(lambda(1))
                                    * std::pow(std::abs(lambda(2)), N(2)-3)
                                    * std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1));
                        norm1   = std::max(colSumF, norm1);
                        for(int j = 1; j <= m(1)-2; j++){
                            colSumF = std::pow(std::abs(lambda(0)), m(0))
                                        * std::pow(std::abs(lambda(1)), j)
                                        * std::abs(lambda(1) - std::pow(lambda(0), m(0))) * (
                                        (1.0 - std::pow(std::abs(lambda(2)), N(2)-2)) / (1.0 - std::abs(lambda(2)))
                                        * (1.0 - std::pow(std::abs(lambda(1)), m(1)))   / (1.0 - std::abs(lambda(1)))
                                        + std::pow(std::abs(lambda(2)), N(2)-2)
                                        + (1.0 - std::pow(std::abs(lambda(1)), j))   / (1.0 - std::abs(lambda(1)))
                                        );
                            norm1   = std::max(colSumF, norm1);
                        }
                    }else{
                        cout << ">>>ERROR: error_l2_sqrt_expression_upper_bound not implemented for three levels (FCF-relaxation) on level " << theoryLevel << endl;
                        throw;
                    }
                    // C-point row sum
                    double summ = 0.0;
                    for(int k = 0; k <= m(1)-3; k++){
                        summ += std::pow(std::abs(lambda(1)), k);
                    }
                    normInf = std::pow(std::abs(lambda(0)), m(0))
                                * std::abs(lambda(1))
                                * std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1))
                                * (1.0 - std::pow(std::abs(lambda(2)), N(2)-2)) / (1.0 - std::abs(lambda(2)))
                                + std::pow(std::abs(lambda(0)), m(0))
                                * std::abs(lambda(1))
                                * std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                * (1.0 - std::pow(std::abs(lambda(2)), N(2)-2)) / (1.0 - std::abs(lambda(2)))
                                * (1.0 - std::pow(std::abs(lambda(1)), m(1)-1)) / (1.0 - std::abs(lambda(1)))
                                + std::pow(std::abs(lambda(0)), m(0))
                                * std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                + std::pow(std::abs(lambda(0)), m(0))
                                * std::abs(lambda(1))
                                * std::pow(std::abs(lambda(2)), N(2)-2)
                                * std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                * summ;
                    // F-point row sum
                    for(int j = 1; j <= m(1)-1; j++){
                        rowSumF = std::pow(std::abs(lambda(0)), m(0))
                                * std::pow(std::abs(lambda(1)), j+1)
                                * std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1))
                                * (1.0 - std::pow(std::abs(lambda(2)), N(2)-3)) / (1.0 - std::abs(lambda(2)))
                                + std::pow(std::abs(lambda(0)), m(0))
                                * std::pow(std::abs(lambda(1)), j+1)
                                * std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                * (1.0 - std::pow(std::abs(lambda(2)), N(2)-3)) / (1.0 - std::abs(lambda(2)))
                                * (1.0 - std::pow(std::abs(lambda(1)), m(1)-1)) / (1.0 - std::abs(lambda(1)))
                                + std::pow(std::abs(lambda(0)), m(0))
                                * std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                * (1.0 - std::pow(std::abs(lambda(1)), j+1)) / (1.0 - std::abs(lambda(1)));
                        normInf = std::max(rowSumF, normInf);
                    }
                    // final result
                    val = sqrt(norm1 * normInf);
                    break;
                }
                default:{
                    cout << ">>>ERROR: error_l2_sqrt_expression_upper_bound only implemented for F- and FCF-relaxation" << endl;
                    throw;
                }
            }
        }
    }else if(numberOfLevels == 4){
        if(cycle == mgritestimate::F_cycle){
            cout << ">>>ERROR: error_l2_sqrt_expression_upper_bound and F-cycles only implemented for three-grid with F-relaxation" << endl;
            throw;
        }
        switch(r){
            case mgritestimate::F_relaxation:{
                /*  On level 1, we have the following points:
                 *
                 *  C0  F1 CF1 F2 CF2  F3 CF3  F4  C1
                 *  |---|---|---|---|---|---|---|---|
                 *  |-------|-------|-------|-------|
                 *  |---------------|---------------|
                 *
                 */
                double colSumC0 = 0.0;  // abs sum of C0-column
                double colSumCF = 0.0;  // abs sum of CF-columns
                double colSumF  = 0.0;  // abs sum of F-columns
                double rowSumCN = 0.0;  // abs sum of CN-rows
                double rowSumCF = 0.0;  // abs sum of CF-rows
                double rowSumF  = 0.0;  // abs sum of F-rows
                double norm1    = 0.0;
                double normInf  = 0.0;
                if(theoryLevel == 0){
                    // abs column sums
                    colSumC0    = std::pow(std::abs(lambda(3)), N(3)-2)
                                    * std::abs(lambda(3) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1) * std::pow(lambda(2), m(2)-1))
                                    + std::abs(lambda(3) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1) * std::pow(lambda(2), m(2)-1))
                                        * (1.0 - std::pow(std::abs(lambda(3)), N(3)-2)) / (1.0 - std::abs(lambda(3)))
                                        * (1.0 - std::pow(std::abs(lambda(2)), m(2)))   / (1.0 - std::abs(lambda(2)))
                                        * (1.0 - std::pow(std::abs(lambda(1)), m(1)))   / (1.0 - std::abs(lambda(1)))
                                        * (1.0 - std::pow(std::abs(lambda(0)), m(0)))   / (1.0 - std::abs(lambda(0)))
                                    + std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1))
                                        * (1.0 - std::pow(std::abs(lambda(2)), m(2)-1)) / (1.0 - std::abs(lambda(2)))
                                        * (1.0 - std::pow(std::abs(lambda(1)), m(1)))   / (1.0 - std::abs(lambda(1)))
                                        * (1.0 - std::pow(std::abs(lambda(0)), m(0)))   / (1.0 - std::abs(lambda(0)))
                                    + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                        * (1.0 - std::pow(std::abs(lambda(1)), m(1)-1)) / (1.0 - std::abs(lambda(1)))
                                        * (1.0 - std::pow(std::abs(lambda(0)), m(0)))   / (1.0 - std::abs(lambda(0)));
                    norm1       = std::max(colSumC0, norm1);
                    for(int j = 1; j <= m(2)-1; j++){
                        double summ = 0.0;
                        for(int k = 0; k <= j-2; k++){
                            summ    = summ + std::pow(std::abs(lambda(2)), k);
                        }
                        colSumCF    = std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1)) * (
                                        std::pow(std::abs(lambda(3)), N(3)-2)
                                            + (1.0 - std::pow(std::abs(lambda(3)), N(3)-2)) / (1.0 - std::abs(lambda(3)))
                                                * (1.0 - std::pow(std::abs(lambda(2)), m(2))) / (1.0 - std::abs(lambda(2)))
                                                * (1.0 - std::pow(std::abs(lambda(1)), m(1))) / (1.0 - std::abs(lambda(1)))
                                                * (1.0 - std::pow(std::abs(lambda(0)), m(0))) / (1.0 - std::abs(lambda(0)))
                                        ) * std::pow(std::abs(lambda(2)), j-1)
                                        + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                            * (1.0 - std::pow(std::abs(lambda(1)), m(1)-1)) / (1.0 - std::abs(lambda(1)))
                                            * (1.0 - std::pow(std::abs(lambda(0)), m(0)))   / (1.0 - std::abs(lambda(0)))
                                        + std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1))
                                            * (1.0 - std::pow(std::abs(lambda(1)), m(1))) / (1.0 - std::abs(lambda(1)))
                                            * (1.0 - std::pow(std::abs(lambda(0)), m(0))) / (1.0 - std::abs(lambda(0)))
                                            * summ;
                        norm1       = std::max(colSumCF, norm1);
                    }
                    for(int r = 0; r <= m(1)-2; r++){
                        for(int j = 0; j <= m(2)-1; j++){
                            double summ1 = 0.0;
                            double summ2 = 0.0;
                            for(int k = 0; k <= j-1; k++){
                                summ1   = summ1 + std::pow(std::abs(lambda(2)), k);
                            }
                            for(int k = 0; k <= r-1; k++){
                                summ2   = summ2 + std::pow(std::abs(lambda(1)), k);
                            }
                            colSumF = std::abs(lambda(1) - std::pow(lambda(0), m(0))) * (
                                        std::pow(std::abs(lambda(3)), N(3)-2)
                                        + (1.0 - std::pow(std::abs(lambda(3)), N(3)-2)) / (1.0 - std::abs(lambda(3)))
                                            * (1.0 - std::pow(std::abs(lambda(2)), m(2))) / (1.0 - std::abs(lambda(2)))
                                            * (1.0 - std::pow(std::abs(lambda(1)), m(1))) / (1.0 - std::abs(lambda(1)))
                                            * (1.0 - std::pow(std::abs(lambda(0)), m(0))) / (1.0 - std::abs(lambda(0)))
                                        ) * std::pow(std::abs(lambda(1)), r) * std::pow(std::abs(lambda(2)), j)
                                        + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                            * (1.0 - std::pow(std::abs(lambda(1)), m(1))) / (1.0 - std::abs(lambda(1)))
                                            * (1.0 - std::pow(std::abs(lambda(0)), m(0))) / (1.0 - std::abs(lambda(0)))
                                            * summ1;
                                        + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                            * (1.0 - std::pow(std::abs(lambda(0)), m(0))) / (1.0 - std::abs(lambda(0)))
                                            * summ2;
                            norm1   = std::max(colSumF, norm1);
                        }
                    }
                }else if(theoryLevel == 1){
                    // abs column sums
                    colSumC0    = std::pow(std::abs(lambda(3)), N(3)-2)
                                    * std::abs(lambda(3) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1) * std::pow(lambda(2), m(2)-1))
                                    + std::abs(lambda(3) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1) * std::pow(lambda(2), m(2)-1))
                                        * (1.0 - std::pow(std::abs(lambda(3)), N(3)-2)) / (1.0 - std::abs(lambda(3)))
                                        * (1.0 - std::pow(std::abs(lambda(2)), m(2)))   / (1.0 - std::abs(lambda(2)))
                                        * (1.0 - std::pow(std::abs(lambda(1)), m(1)))   / (1.0 - std::abs(lambda(1)))
                                    + std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1))
                                        * (1.0 - std::pow(std::abs(lambda(2)), m(2)-1)) / (1.0 - std::abs(lambda(2)))
                                        * (1.0 - std::pow(std::abs(lambda(1)), m(1)))   / (1.0 - std::abs(lambda(1)))
                                    + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                        * (1.0 - std::pow(std::abs(lambda(1)), m(1)-1)) / (1.0 - std::abs(lambda(1)));
                    norm1       = std::max(colSumC0, norm1);
                    for(int j = 1; j <= m(2)-1; j++){
                        double summ = 0.0;
                        for(int k = 0; k <= j-2; k++){
                            summ    = summ + std::pow(std::abs(lambda(2)), k);
                        }
                        colSumCF    = std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1)) * (
                                        std::pow(std::abs(lambda(3)), N(3)-2)
                                            + (1.0 - std::pow(std::abs(lambda(3)), N(3)-2)) / (1.0 - std::abs(lambda(3)))
                                                * (1.0 - std::pow(std::abs(lambda(2)), m(2))) / (1.0 - std::abs(lambda(2)))
                                                * (1.0 - std::pow(std::abs(lambda(1)), m(1))) / (1.0 - std::abs(lambda(1)))
                                        ) * std::pow(std::abs(lambda(2)), j-1)
                                        + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                            * (1.0 - std::pow(std::abs(lambda(1)), m(1)-1)) / (1.0 - std::abs(lambda(1)))
                                        + std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1))
                                            * (1.0 - std::pow(std::abs(lambda(1)), m(1))) / (1.0 - std::abs(lambda(1)))
                                            * summ;
                        norm1       = std::max(colSumCF, norm1);
                    }
                    for(int r = 0; r <= m(1)-2; r++){
                        for(int j = 0; j <= m(2)-1; j++){
                            double summ1 = 0.0;
                            double summ2 = 0.0;
                            for(int k = 0; k <= j-1; k++){
                                summ1   = summ1 + std::pow(std::abs(lambda(2)), k);
                            }
                            for(int k = 0; k <= r-1; k++){
                                summ2   = summ2 + std::pow(std::abs(lambda(1)), k);
                            }
                            colSumF = std::abs(lambda(1) - std::pow(lambda(0), m(0))) * (
                                        std::pow(std::abs(lambda(3)), N(3)-2)
                                        + (1.0 - std::pow(std::abs(lambda(3)), N(3)-2)) / (1.0 - std::abs(lambda(3)))
                                            * (1.0 - std::pow(std::abs(lambda(2)), m(2))) / (1.0 - std::abs(lambda(2)))
                                            * (1.0 - std::pow(std::abs(lambda(1)), m(1))) / (1.0 - std::abs(lambda(1)))
                                        ) * std::pow(std::abs(lambda(1)), r) * std::pow(std::abs(lambda(2)), j)
                                        + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                            * (1.0 - std::pow(std::abs(lambda(1)), m(1))) / (1.0 - std::abs(lambda(1)))
                                            * summ1
                                        + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                            * summ2;
                            norm1   = std::max(colSumF, norm1);
                        }
                    }
                }else{
                    cout << ">>>ERROR: error_l2_sqrt_expression_upper_bound not implemented for four levels (FCF-relaxation) on level " << theoryLevel << endl;
                    throw;
                }
                // abs row sums
                rowSumCN    = std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                * (1.0 - std::pow(std::abs(lambda(3)), N(3)-1)) / (1.0 - std::abs(lambda(3)))
                                * (1.0 - std::pow(std::abs(lambda(2)), m(2)))   / (1.0 - std::abs(lambda(2)))
                                * (1.0 - std::pow(std::abs(lambda(1)), m(1)-1)) / (1.0 - std::abs(lambda(1)))
                                + std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1))
                                    * (1.0 - std::pow(std::abs(lambda(3)), N(3)-1)) / (1.0 - std::abs(lambda(3)))
                                    * (1.0 - std::pow(std::abs(lambda(2)), m(2)-1))   / (1.0 - std::abs(lambda(2)))
                                + std::abs(lambda(3) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1) * std::pow(lambda(2), m(2)-1))
                                    * (1.0 - std::pow(std::abs(lambda(3)), N(3)-1)) / (1.0 - std::abs(lambda(3)));
                normInf     = std::max(rowSumCN, normInf);
                for(int j = 1; j <= m(2)-1; j++){
                    rowSumCF    = std::pow(std::abs(lambda(2)), j) * (
                                        std::abs(lambda(3) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1) * std::pow(lambda(2), m(2)-1))
                                        * (1.0 - std::pow(std::abs(lambda(3)), N(3)-2)) / (1.0 - std::abs(lambda(3)))
                                        + std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1))
                                        * (1.0 - std::pow(std::abs(lambda(3)), N(3)-2)) / (1.0 - std::abs(lambda(3)))
                                        * (1.0 - std::pow(std::abs(lambda(2)), m(2)-1))   / (1.0 - std::abs(lambda(2)))
                                        + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                        * (1.0 - std::pow(std::abs(lambda(3)), N(3)-2)) / (1.0 - std::abs(lambda(3)))
                                        * (1.0 - std::pow(std::abs(lambda(2)), m(2)))   / (1.0 - std::abs(lambda(2)))
                                        * (1.0 - std::pow(std::abs(lambda(1)), m(1)-1)) / (1.0 - std::abs(lambda(1)))
                                    ) + (1.0 - std::pow(std::abs(lambda(2)), j)) / (1.0 - std::abs(lambda(2))) * (
                                        std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1))
                                        + (1.0 - std::pow(std::abs(lambda(1)), m(1)-1))   / (1.0 - std::abs(lambda(1)))
                                        * std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                    );
                    normInf     = std::max(rowSumCF, normInf);
                }
                for(int j = 0; j <= m(2)-1; j++){
                    for(int r = 1; r <= m(1)-1; r++){
                        double summ = 0.0;
                        for(int k = 0; k <= j-1; k++){
                            summ    = summ + std::pow(std::abs(lambda(2)), k);
                        }
                        rowSumF = std::pow(std::abs(lambda(1)), r) * (
                                        std::pow(std::abs(lambda(2)), j)
                                            * std::abs(lambda(3) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1) * std::pow(lambda(2), m(2)-1))
                                            * (1.0 - std::pow(std::abs(lambda(3)), N(3)-2)) / (1.0 - std::abs(lambda(3)))
                                        + std::pow(std::abs(lambda(2)), j)
                                            * std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1))
                                            * (1.0 - std::pow(std::abs(lambda(3)), N(3)-2)) / (1.0 - std::abs(lambda(3)))
                                            * (1.0 - std::pow(std::abs(lambda(2)), m(2)-1))   / (1.0 - std::abs(lambda(2)))
                                        + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                            * (1.0 - std::pow(std::abs(lambda(3)), N(3)-2)) / (1.0 - std::abs(lambda(3)))
                                            * (1.0 - std::pow(std::abs(lambda(2)), m(2)))   / (1.0 - std::abs(lambda(2)))
                                            * (1.0 - std::pow(std::abs(lambda(1)), m(1)-1)) / (1.0 - std::abs(lambda(1)))
                                        + std::abs(lambda(2) - std::pow(lambda(0), m(0)) * std::pow(lambda(1), m(1)-1))
                                            * summ
                                        + std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                                            * summ
                                            * (1.0 - std::pow(std::abs(lambda(1)), m(1)-1)) / (1.0 - std::abs(lambda(1)))
                                    ) + std::abs(lambda(1) - std::pow(lambda(0), m(0))) * (1.0 - std::pow(std::abs(lambda(1)), r)) / (1.0 - std::abs(lambda(1)));
                        normInf = std::max(rowSumF, normInf);
                    }
                }
                val = sqrt(norm1 * normInf);
                break;
            }
            case mgritestimate::FCF_relaxation:{
                cout << ">>>ERROR: error_l2_sqrt_expression_upper_bound with FCF-relaxation for four grids not implemented." << endl;
                break;
            }
            default:{
                cout << ">>>ERROR: error_l2_sqrt_expression_upper_bound only implemented for F- and FCF-relaxation." << endl;
                throw;
            }
        }
    }else{
            cout << ">>>ERROR: error_l2_sqrt_expression_upper_bound not implemented for " << numberOfLevels << " levels." << endl;
            throw;
    }
    return val;
}

/**
 *  Computes an approximation of \f$ \sqrt{\| E \|_1 \| E \|_\infty} \f$ using expression.
 *
 *  Note: Evaluation is reasonably cheap, so let's just wrap the complex equivalent.
 */
// note: we use std::abs/std::pow here instead of arma::abs/arma::pow because it allows mixed complex/real operands
double get_error_l2_sqrt_expression_approximate_rate(int r,                     ///< number of FC relaxation steps
                                                     Col<double> lambda,        ///< eigenvalues of \f$\Phi_l\f$
                                                     Col<int> N,                ///< number of time steps on each grid level
                                                     Col<int> m,                ///< coarsening factors between all grid levels
                                                     int theoryLevel            ///< expression for error propagator on grid level
                                                     ){
    Col<cx_double> lambdac(lambda, 0.0*lambda);
    double val = get_error_l2_sqrt_expression_approximate_rate(r, lambdac, N, m, theoryLevel);
    return val;
}

/**
 *  Computes an approximation of \f$ \sqrt{\| E \|_1 \| E \|_\infty} \f$ using expression (complex version).
 */
// note: we use std::abs/std::pow here instead of arma::abs/arma::pow because it allows mixed complex/real operands
double get_error_l2_sqrt_expression_approximate_rate(int r,                     ///< number of FC relaxation steps
                                                     Col<cx_double> lambda,     ///< eigenvalues of \f$\Phi_l\f$
                                                     Col<int> N,                ///< number of time steps on each grid level
                                                     Col<int> m,                ///< coarsening factors between all grid levels
                                                     int theoryLevel            ///< expression for error propagator on grid level
                                                     ){
    // check if time stepper is stable
    if(arma::any(arma::abs(lambda) > constants::time_stepper_stability_limit)){
        return -1.0;
    }
    // evaluate expression depending on number of time grids and relaxation scheme
    int numberOfLevels  = N.n_elem;
    double val      = 0.0;
    double colSum   = 0.0;
    double rowSum   = 0.0;
    switch(r){
        case mgritestimate::F_relaxation:{
            // set up \f$ \tilde{m} \f$
            Col<int> mt;
            mt.set_size(numberOfLevels);
            for(int l = 0; l < numberOfLevels-1; l++){
                mt(l) = m(l);
            }
            mt(numberOfLevels-1) = N(numberOfLevels-1);
            if(theoryLevel == 0){
                // compute approximation for column sum
                for(int l = 1; l <= numberOfLevels-1; l++){
                    cx_double prod1 = std::pow(lambda(0), mt(0));
                    double prod2 = 1.0;
                    for(int k = 1; k <= l-1; k++){
                        prod1 *= std::pow(lambda(k), mt(k)-1);
                    }
                    for(int k = 1; k <= l; k++){
                        prod2 *= (1.0 - std::pow(std::abs(lambda(k)), mt(k)-1)) / (1.0 - std::abs(lambda(k)));
                    }
                    colSum += std::abs(lambda(l) - prod1) * prod2;
                }
                colSum *= (1.0 - std::pow(std::abs(lambda(0)), mt(0))) / (1.0 - std::abs(lambda(0)));
                if(numberOfLevels > 2){
                    cx_double prod1 = std::pow(lambda(0), mt(0));
                    for(int k = 1; k <= numberOfLevels-2; k++){
                        prod1 *= std::pow(lambda(k), mt(k)-1);
                    }
                    colSum += std::pow(std::abs(lambda(numberOfLevels-1)), mt(numberOfLevels-1)-1) * std::abs(lambda(numberOfLevels-1) - prod1);
                }
                // compute approximation of row sum
                for(int l = 1; l <= numberOfLevels-1; l++){
                    cx_double prod1 = std::pow(lambda(0), mt(0));
                    double prod2 = 1.0;
                    for(int k = 1; k <= l-1; k++){
                        prod1 *= std::pow(lambda(k), mt(k)-1);
                    }
                    for(int k = l; k <= numberOfLevels-1; k++){
                        prod2 *= (1.0 - std::pow(std::abs(lambda(k)), mt(k))) / (1.0 - std::abs(lambda(k)));
                    }
                    rowSum += std::abs(lambda(l) - prod1) * prod2;
                }
            }else if(theoryLevel == 1){
                // compute approximation for column sum
                for(int l = 1; l <= numberOfLevels-1; l++){
                    cx_double prod1 = std::pow(lambda(0), mt(0));
                    double prod2 = 1.0;
                    for(int k = 1; k <= l-1; k++){
                        prod1 *= std::pow(lambda(k), mt(k)-1);
                    }
                    for(int k = 1; k <= l; k++){
                        prod2 *= (1.0 - std::pow(std::abs(lambda(k)), mt(k)-1)) / (1.0 - std::abs(lambda(k)));
                    }
                    colSum += std::abs(lambda(l) - prod1) * prod2;
                }
                if(numberOfLevels > 2){
                    cx_double prod1 = std::pow(lambda(0), mt(0));
                    for(int k = 1; k <= numberOfLevels-2; k++){
                        prod1 *= std::pow(lambda(k), mt(k)-1);
                    }
                    colSum += std::pow(std::abs(lambda(numberOfLevels-1)), mt(numberOfLevels-1)-1) * std::abs(lambda(numberOfLevels-1) - prod1);
                }
                // compute approximation of row sum
                for(int l = 1; l <= numberOfLevels-1; l++){
                    cx_double prod1 = std::pow(lambda(0), mt(0));
                    double prod2 = 1.0;
                    for(int k = 1; k <= l-1; k++){
                        prod1 *= std::pow(lambda(k), mt(k)-1);
                    }
                    for(int k = l; k <= numberOfLevels-1; k++){
                        prod2 *= (1.0 - std::pow(std::abs(lambda(k)), mt(k))) / (1.0 - std::abs(lambda(k)));
                    }
                    rowSum += std::abs(lambda(l) - prod1) * prod2;
                }
            }else{
                cout << ">>>ERROR: error_l2_sqrt_expression_approximate_rate not implemented on level " << theoryLevel << "." << endl << endl;
                throw;
            }
            break;
        }
        case mgritestimate::FCF_relaxation:{
            double summ     = 0.0;
            double prod1    = 0.0;
            cx_double prod2 = 0.0;
            double prod3    = 0.0;
            // approximate column sum
            if(numberOfLevels > 2){
                colSum = std::pow(std::abs(lambda(0)), m(0))
                            * std::abs(lambda(1) - std::pow(lambda(0), m(0)))
                            * (1.0 - std::pow(std::abs(lambda(1)), m(1))) / (1.0 - std::abs(lambda(1)));
            }
            summ    = 0.0;
            for(int k = 2; k <= numberOfLevels-2; k++){
                prod1   = 1.0;
                prod2   = std::pow(lambda(0), m(0));
                prod3   = 1.0;
                for(int j = 1; j <= k-1; j++){
                    prod1 *= std::abs(lambda(j));
                    prod2 *= std::pow(lambda(j), m(j)-1);
                }
                for(int j = 1; j <= k; j++){
                    prod3 *= (1.0 - std::pow(std::abs(lambda(j)), m(j))) / (1.0 - std::abs(lambda(j)));
                }
                summ += prod1 * std::abs(lambda(k) - prod2) * prod3;
            }
            colSum += 1.0 / (numberOfLevels - 1.0) * std::pow(std::abs(lambda(0)), m(0)) * summ;
            summ    = 0.0;
            prod1   = std::pow(std::abs(lambda(0)), m(0));
            prod3   = 1.0;
            for(int k = 1; k <= numberOfLevels-2; k++){
                prod1 *= std::abs(lambda(k));
                prod3 *= (1.0 - std::pow(std::abs(lambda(k)), m(k))) / (1.0 - std::abs(lambda(k)));
            }
            for(int k = 2; k <= numberOfLevels-1; k++){
                prod2   = std::pow(lambda(0), m(0));
                for(int j = 1; j <= k-1; j++){
                    prod2 *= std::pow(lambda(j), m(j)-1);
                }
                summ += std::abs(lambda(k) - prod2);
            }
            colSum += 1.0 / (numberOfLevels - 1.0)
                        * (1.0 - std::pow(std::abs(lambda(numberOfLevels-1)), N(numberOfLevels-1)-1)) / (1.0 - std::abs(lambda(numberOfLevels-1)))
                        * prod1 * summ * prod3;
            prod1   = std::abs(lambda(0));
            prod3   = 1.0;
            for(int k = 0; k <= numberOfLevels-2; k++){
                prod1 *= std::pow(std::abs(lambda(k)), m(k)-1);
            }
            for(int k = 1; k <= numberOfLevels-2; k++){
                prod3 *= (1.0 - std::pow(std::abs(lambda(k)), m(k))) / (1.0 - std::abs(lambda(k)));
            }
            colSum += 1.0 / (numberOfLevels - 1.0)
                        * (1.0 - std::pow(std::abs(lambda(numberOfLevels-1)), N(numberOfLevels-1)-1)) / (1.0 - std::abs(lambda(numberOfLevels-1)))
                        * prod1 * prod3 * std::abs(lambda(1) - std::pow(lambda(0), m(0)));
            // approximate row sum
            rowSum = 0.0;
            for(int k = 1; k <= numberOfLevels-1; k++){
                prod1   = 1.0;
                prod2   = std::pow(lambda(0), m(0));
                prod3   = 1.0;
                for(int j = 1; j <= k-1; j++){
                    prod1 *= std::abs(lambda(j));
                    prod2 *= std::pow(lambda(j), m(j)-1);

                }
                for(int j = k; j <= numberOfLevels-2; j++){
                    prod3 *= (1.0 - std::pow(std::abs(lambda(j)), m(j))) / (1.0 - std::abs(lambda(j)));
                }
                rowSum += prod1 * std::abs(lambda(k) - prod2) * prod3;
            }
            rowSum *= std::pow(std::abs(lambda(0)), m(0))
                        * (1.0 - std::pow(std::abs(lambda(numberOfLevels-1)), N(numberOfLevels-1))) / (1.0 - std::abs(lambda(numberOfLevels-1)));
            if(theoryLevel == 0){
                colSum *= (1.0 - std::pow(std::abs(lambda(0)), m(0))) / (1.0 - std::abs(lambda(0)));
            }else if(theoryLevel == 1){
                // do nothing
            }else{
                cout << ">>>ERROR: error_l2_sqrt_expression_approximate_rate not implemented on level " << theoryLevel << "." << endl << endl;
                throw;
            }
            break;
        }
        default:{
            cout << ">>>ERROR: error_l2_sqrt_expression_approximate_rate only implemented for F- and FCF-relaxation." << endl;
            throw;
        }
    }
    // compute approximation of convergence rate
    val = std::sqrt(colSum * rowSum);
    if(std::isnan(val)){
        val = 0.0;
    }
    return val;
}

/**
 *  compute global index range for local MPI rank
 */
void get_samples_index_range(int numberOfSamples,       ///< global number of samples
                             int &samplesRankStartIdx,  ///< start index of samples on MPI rank
                             int &samplesRankStopIdx    ///< stop index of samples on MPI rank
                             ){
    // check MPI environment
    int world_rank;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // compute sample sizes on MPI rank
    int samplesPerRank      = numberOfSamples / world_size;
    int leftover            = numberOfSamples - samplesPerRank * world_size;
    samplesRankStartIdx     = samplesPerRank * world_rank           + leftover * (world_rank > 0);
    samplesRankStopIdx      = samplesPerRank * (world_rank + 1) - 1 + leftover;
    // cout << "Rank " << world_rank << " / " << world_size << ": eigenvalue index range is "
    //      << samplesRankStartIdx << " - " << samplesRankStopIdx << " (" << numberOfSamples << ")"
    //      << endl;
}

/**
 *  collect computed bounds on rank 0
 */
void communicateBounds(Col<double> *&estimate,          ///< on return, the estimate for all eigenvalues
                       int numberOfSamples,             ///< global number of samples
                       int samplesRankStartIdx,         ///< start index of samples on MPI rank
                       int samplesRankStopIdx           ///< stop index of samples on MPI rank
                       ){
    // check MPI environment
    int world_rank;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // send data to rank 0
    int idx = 0;
    if(world_rank == 0){
        int samplesPerRank  = numberOfSamples / world_size; // minimum number of samples on each MPI rank
        int receiveSize     = samplesPerRank;               // size of MPI receive
        int receiveStartIdx = samplesRankStartIdx;          // start index of received samples
        int receiveStopIdx  = samplesRankStopIdx;           // stop index of received samples
        for(int rank = 1; rank < world_size; rank++){
            receiveStartIdx = receiveStopIdx + 1;               // increment start index for MPI rank
            receiveStopIdx  = receiveStopIdx + samplesPerRank;  // increment stop index for MPI rank
            double data[samplesPerRank];
            // post blocking receive
            MPI_Recv(data, receiveSize, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
            idx = 0;
            // unpack data
            for(int evalIdx = receiveStartIdx; evalIdx <= receiveStopIdx; evalIdx++){
                (*estimate)(evalIdx) = data[idx++];
            }
        }
    }else{
        int samplesOnRank = samplesRankStopIdx - samplesRankStartIdx + 1;   // size of MPI send
        double data[samplesOnRank];
        idx = 0;
        // pack data
        for(int evalIdx = samplesRankStartIdx; evalIdx <= samplesRankStopIdx; evalIdx++){
            data[idx++] = (*estimate)(evalIdx);
        }
        // post blocking send
        MPI_Send(data, samplesOnRank, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}
