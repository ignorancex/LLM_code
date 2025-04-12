#include "io_routines.hpp"

void export_estimates(appStruct &app, arma::file_type type){
    if(app.world_rank == 0){
        if(!app.userOutputFile){
            get_default_filename(app.cycle, app.bound, app.relax, &app.userOutputFileName);
        }
        export_vector(app.estimate, app.userOutputFileName, type);
        export_vector_minmax(app.bound, app.estimate, app.userOutputFileName, type);
    }
}

void export_matrix(arma::sp_mat *m, const string filename, arma::file_type type){
    string suffix;
    get_filename_suffix(type, suffix);
    arma::mat(*m).save(filename+suffix, type);
}

/// \todo check newer Armadillo versions and arma::arma_ascii format
void export_matrix(arma::sp_cx_mat *m, const string filename, arma::file_type type){
    string suffix;
    get_filename_suffix(type, suffix);
    arma::mat(arma::real(*m)).save(filename+"_real"+suffix, type);
    arma::mat(arma::imag(*m)).save(filename+"_imag"+suffix, type);
}

void export_vector(arma::Col<double> *v, const string filename, arma::file_type type){
    string suffix;
    get_filename_suffix(type, suffix);
    arma::mat(*v).save(filename+suffix, type);
}

void export_vector(arma::Col<arma::cx_double> *v, const string filename, arma::file_type type){
    string suffix;
    get_filename_suffix(type, suffix);
    arma::mat(arma::real(*v)).save(filename+"_real"+suffix, type);
    arma::mat(arma::imag(*v)).save(filename+"_imag"+suffix, type);
}

void export_vector_minmax(int bound, arma::Col<double> *v, const string filename, arma::file_type type){
    string prefix;
    string suffix;
    arma::mat tmp(1, 1);
    get_filename_suffix(type, suffix);
    if((bound == mgritestimate::error_l2_upper_bound)
        || (bound == mgritestimate::error_l2_sqrt_upper_bound)
        || (bound == mgritestimate::error_l2_sqrt_expression_upper_bound)
        || (bound == mgritestimate::error_l2_tight_twogrid_upper_bound)
        || (bound == mgritestimate::error_l2_tight_twogrid_bound)
        || (bound == mgritestimate::error_l2_tight_twogrid_bound_single_iter)
        || (bound == mgritestimate::error_l2_sqrt_expression_approximate_rate)
        || (bound == mgritestimate::upper_bound_southworth2019_equation63)
        || (bound == mgritestimate::lower_bound_southworth2019_equation63)
        || (bound == mgritestimate::residual_l2_upper_bound)
        || (bound == mgritestimate::residual_l2_sqrt_upper_bound)){
        prefix = "max_";
        tmp.fill(arma::max(*v));
        cout << "Convergence <= " << std::setprecision(17) << tmp[0, 0] << endl << endl;
        tmp.save(prefix+filename+suffix, type);
    }else if((bound == mgritestimate::error_l2_approximate_lower_bound)
        || (bound == mgritestimate::error_l2_sqrt_approximate_lower_bound)
        || (bound == mgritestimate::error_l2_tight_twogrid_lower_bound)
        || (bound == mgritestimate::residual_l2_lower_bound)){
        arma::uvec tmpIdx = find((*v) > 0.0);
        tmp.fill(arma::min((*v).elem(tmpIdx)));
        prefix = "min_";
        tmp.save(prefix+filename+suffix, type);
        cout << "Convergence >= " << std::setprecision(17) << tmp[0, 0] << endl << endl;
    }else{
        cout << ">>>WARNING: Unknown bound type " << bound << " in export routine. Exporting min and max." << endl << endl;
        tmp.fill(arma::max(*v));
        tmp.save("max_"+filename+suffix, type);
        cout << "Convergence <= " << std::setprecision(17) << tmp[0, 0] << endl << endl;
        arma::uvec tmpIdx = find((*v) > 0.0);
        tmp.fill(arma::min((*v).elem(tmpIdx)));
        tmp.save("min_"+filename+suffix, type);
        cout << "Convergence >= " << std::setprecision(17) << tmp[0, 0] << endl << endl;
    }
}

void get_filename_suffix(arma::file_type type, string &suffix){
    if(type == arma::raw_ascii){
        suffix = ".txt";
    }else{
        cout << ">>>WARNING: Defaulting to raw_ascii format." << endl << endl;
        suffix = ".txt";
    }
}

/**
 *  set default filename depending on bound/relaxation type
 */
void get_default_filename(const int cycle, const int bound, const int relax, string *filename){
    string cyc;
    if(cycle == mgritestimate::V_cycle){
        cyc = "Vcycle_";
    }else if(cycle == mgritestimate::F_cycle){
        cyc = "Fcycle_";
    }else{
        cout << ">>>ERROR: Unknown cycling strategy." << endl << endl;
        throw;
    }
    switch(relax){
        case mgritestimate::F_relaxation:{
            if(bound == mgritestimate::error_l2_upper_bound){
                *filename = cyc + "error_l2_upper_bound_E_F";
            }else if(bound == mgritestimate::error_l2_sqrt_upper_bound){
                *filename = cyc + "error_l2_sqrt_upper_bound_E_F";
            }else if(bound == mgritestimate::error_l2_sqrt_expression_upper_bound){
                *filename = cyc + "error_l2_sqrt_expression_upper_bound_E_F";
            }else if(bound == mgritestimate::error_l2_tight_twogrid_upper_bound){
                *filename = cyc + "error_l2_tight_twogrid_upper_bound_E_F";
            }else if(bound == mgritestimate::error_l2_tight_twogrid_bound){
                *filename = cyc + "error_l2_tight_twogrid_bound_E_F";
            }else if(bound == mgritestimate::error_l2_tight_twogrid_bound_single_iter){
                *filename = cyc + "error_l2_tight_twogrid_bound_single_iter_E_F";
            }else if(bound == mgritestimate::error_l2_approximate_lower_bound){
                *filename = cyc + "error_l2_approximate_lower_bound_E_F";
            }else if(bound == mgritestimate::error_l2_sqrt_approximate_lower_bound){
                *filename = cyc + "error_l2_sqrt_approximate_lower_bound_E_F";
            }else if(bound == mgritestimate::error_l2_sqrt_expression_approximate_rate){
                *filename = cyc + "error_l2_sqrt_expression_approximate_rate_E_F";
            }else if(bound == mgritestimate::lower_bound_southworth2019_equation63){
                *filename = cyc + "lower_bound_southworth2019_equation63_F";
            }else if(bound == mgritestimate::upper_bound_southworth2019_equation63){
                *filename = cyc + "upper_bound_southworth2019_equation63_F";
            }else if(bound == mgritestimate::residual_l2_upper_bound){
                *filename = cyc + "residual_l2_upper_bound_R_F";
            }else if(bound == mgritestimate::residual_l2_sqrt_upper_bound){
                *filename = cyc + "residual_l2_sqrt_upper_bound_R_F";
            }else if(bound == mgritestimate::residual_l2_lower_bound){
                *filename = cyc + "residual_l2_lower_bound_R_F";
            }else{
                *filename = cyc + "bound_E_F";
            }
            break;
        }
        case mgritestimate::FCF_relaxation:{
            if(bound == mgritestimate::error_l2_upper_bound){
                *filename = cyc + "error_l2_upper_bound_E_FCF";
            }else if(bound == mgritestimate::error_l2_sqrt_upper_bound){
                *filename = cyc + "error_l2_sqrt_upper_bound_E_FCF";
            }else if(bound == mgritestimate::error_l2_sqrt_expression_upper_bound){
                *filename = cyc + "error_l2_sqrt_expression_upper_bound_E_FCF";
            }else if(bound == mgritestimate::error_l2_tight_twogrid_upper_bound){
                *filename = cyc + "error_l2_tight_twogrid_upper_bound_E_FCF";
            }else if(bound == mgritestimate::error_l2_tight_twogrid_bound){
                *filename = cyc + "error_l2_tight_twogrid_bound_E_FCF";
            }else if(bound == mgritestimate::error_l2_tight_twogrid_bound_single_iter){
                *filename = cyc + "error_l2_tight_twogrid_bound_single_iter_E_FCF";
            }else if(bound == mgritestimate::error_l2_approximate_lower_bound){
                *filename = cyc + "error_l2_approximate_lower_bound_E_FCF";
            }else if(bound == mgritestimate::error_l2_sqrt_approximate_lower_bound){
                *filename = cyc + "error_l2_sqrt_approximate_lower_bound_E_FCF";
            }else if(bound == mgritestimate::error_l2_sqrt_expression_approximate_rate){
                *filename = cyc + "error_l2_sqrt_expression_approximate_rate_E_FCF";
            }else if(bound == mgritestimate::lower_bound_southworth2019_equation63){
                *filename = cyc + "lower_bound_southworth2019_equation63_FCF";
            }else if(bound == mgritestimate::upper_bound_southworth2019_equation63){
                *filename = cyc + "upper_bound_southworth2019_equation63_F";
            }else if(bound == mgritestimate::residual_l2_upper_bound){
                *filename = cyc + "residual_l2_upper_bound_R_FCF";
            }else if(bound == mgritestimate::residual_l2_sqrt_upper_bound){
                *filename = cyc + "residual_l2_sqrt_upper_bound_R_FCF";
            }else if(bound == mgritestimate::residual_l2_lower_bound){
                *filename = cyc + "residual_l2_lower_bound_R_FCF";
            }else{
                *filename = cyc + "bound_E_FCF";
            }
            break;
        }
        case mgritestimate::FCFCF_relaxation:{
            if(bound == mgritestimate::error_l2_upper_bound){
                *filename = cyc + "error_l2_upper_bound_E_FCFCF";
            }else if(bound == mgritestimate::error_l2_sqrt_upper_bound){
                *filename = cyc + "error_l2_sqrt_upper_bound_E_FCFCF";
            }else if(bound == mgritestimate::error_l2_sqrt_expression_upper_bound){
                *filename = cyc + "error_l2_sqrt_expression_upper_bound_E_FCFCF";
            }else if(bound == mgritestimate::error_l2_tight_twogrid_upper_bound){
                *filename = cyc + "error_l2_tight_twogrid_upper_bound_E_FCFCF";
            }else if(bound == mgritestimate::error_l2_approximate_lower_bound){
                *filename = cyc + "error_l2_approximate_lower_bound_E_FCFCF";
            }else if(bound == mgritestimate::error_l2_sqrt_approximate_lower_bound){
                *filename = cyc + "error_l2_sqrt_approximate_lower_bound_E_FCFCF";
            }else if(bound == mgritestimate::error_l2_sqrt_expression_approximate_rate){
                *filename = cyc + "error_l2_sqrt_expression_approximate_rate_E_FCFCF";
            }else if(bound == mgritestimate::residual_l2_upper_bound){
                *filename = cyc + "residual_l2_upper_bound_R_FCFCF";
            }else if(bound == mgritestimate::residual_l2_sqrt_upper_bound){
                *filename = cyc + "residual_l2_sqrt_upper_bound_R_FCFCF";
            }else if(bound == mgritestimate::residual_l2_lower_bound){
                *filename = cyc + "residual_l2_lower_bound_R_FCFCF";
            }else{
                *filename = cyc + "bound_E_FCFCF";
            }
            break;
        }
        case mgritestimate::FCFCFCF_relaxation:{
            if(bound == mgritestimate::error_l2_upper_bound){
                *filename = cyc + "error_l2_upper_bound_E_FCFCFCF";
            }else if(bound == mgritestimate::error_l2_sqrt_upper_bound){
                *filename = cyc + "error_l2_sqrt_upper_bound_E_FCFCFCF";
            }else if(bound == mgritestimate::error_l2_sqrt_expression_upper_bound){
                *filename = cyc + "error_l2_sqrt_expression_upper_bound_E_FCFCFCF";
            }else if(bound == mgritestimate::error_l2_tight_twogrid_upper_bound){
                *filename = cyc + "error_l2_tight_twogrid_upper_bound_E_FCFCFCF";
            }else if(bound == mgritestimate::error_l2_approximate_lower_bound){
                *filename = cyc + "error_l2_approximate_lower_bound_E_FCFCFCF";
            }else if(bound == mgritestimate::error_l2_sqrt_approximate_lower_bound){
                *filename = cyc + "error_l2_sqrt_approximate_lower_bound_E_FCFCFCF";
            }else if(bound == mgritestimate::error_l2_sqrt_expression_approximate_rate){
                *filename = cyc + "error_l2_sqrt_expression_approximate_rate_E_FCFCFCF";
            }else if(bound == mgritestimate::residual_l2_upper_bound){
                *filename = cyc + "residual_l2_upper_bound_R_FCFCFCF";
            }else if(bound == mgritestimate::residual_l2_sqrt_upper_bound){
                *filename = cyc + "residual_l2_sqrt_upper_bound_R_FCFCFCF";
            }else if(bound == mgritestimate::residual_l2_lower_bound){
                *filename = cyc + "residual_l2_lower_bound_R_FCFCFCF";
            }else{
                *filename = cyc + "bound_E_FCFCFCF";
            }
            break;
        }
        case mgritestimate::FCFCFCFCF_relaxation:{
            if(bound == mgritestimate::error_l2_upper_bound){
                *filename = cyc + "error_l2_upper_bound_E_FCFCFCFCF";
            }else if(bound == mgritestimate::error_l2_sqrt_upper_bound){
                *filename = cyc + "error_l2_sqrt_upper_bound_E_FCFCFCFCF";
            }else if(bound == mgritestimate::error_l2_sqrt_expression_upper_bound){
                *filename = cyc + "error_l2_sqrt_expression_upper_bound_E_FCFCFCFCF";
            }else if(bound == mgritestimate::error_l2_tight_twogrid_upper_bound){
                *filename = cyc + "error_l2_tight_twogrid_upper_bound_E_FCFCFCFCF";
            }else if(bound == mgritestimate::error_l2_approximate_lower_bound){
                *filename = cyc + "error_l2_approximate_lower_bound_E_FCFCFCFCF";
            }else if(bound == mgritestimate::error_l2_sqrt_approximate_lower_bound){
                *filename = cyc + "error_l2_sqrt_approximate_lower_bound_E_FCFCFCFCF";
            }else if(bound == mgritestimate::error_l2_sqrt_expression_approximate_rate){
                *filename = cyc + "error_l2_sqrt_expression_approximate_rate_E_FCFCFCFCF";
            }else if(bound == mgritestimate::residual_l2_upper_bound){
                *filename = cyc + "residual_l2_upper_bound_R_FCFCFCFCF";
            }else if(bound == mgritestimate::residual_l2_sqrt_upper_bound){
                *filename = cyc + "residual_l2_sqrt_upper_bound_R_FCFCFCFCF";
            }else if(bound == mgritestimate::residual_l2_lower_bound){
                *filename = cyc + "residual_l2_lower_bound_R_FCFCFCFCF";
            }else{
                *filename = cyc + "bound_E_FCFCFCFCF";
            }
            break;
        }
        default:{
            cout << ">>>ERROR: Unknown relaxation type when setting default filename." << endl;
            throw;
        }
    }
}

/**
 *  Read commandline options (default arguments):
 *
 *      --help                                                               prints all commandline options
 *      --number-of-timesteps 1025                                           number of timesteps on fine grid (must contain time point corresponding to initial condition)
 *      --number-of-time-grids 2                                             number of levels in time grid hierarchy
 *      --coarsening-factors 2                                               temporal coarsening factors for level 0-->1, 1-->2, ...
 *                                                                           note: must not be used before option --number-of-time-grids
 *      --runge-kutta-method L_stable_SDIRK1                                 Runge-Kutta method, if spatial eigenvalues are supplied, see constants.hpp
 *      --sample-complex-plane -10.0 1.0 -4.0 4.0                            sampling of complex plane with ranges [-10.0, 1.0] x [-4.0, 4.0]
 *      --complex-plane-sample-size 12 9                                     sampling of real and complex axis using 12 and 9 points
 *      --file-spatial-real-eigenvalues dteta_real.txt                       name of file that contains real part of spatial eigenvalues
 *      --file-spatial-complex-eigenvalues dteta_real.txt dteta_imag.txt     name of files that contain real and complex part of spatial eigenvalues, respectively
 *      --file-phi-real-eigenvalues lambda_real.txt                          name of file that contains real part of eigenvalues of Phi
 *      --file-phi-complex-eigenvalues lambda_real.txt lambda_imag.txt       name of file that contains complex part of eigenvalues of Phi
 *      --bound error_l2_tight_twogrid_upper_bound                           bound to evaluate, see constants.hpp
 *      --bound-on-level 1                                                   bound is evaluated on this level (options: 0, 1)
 *      --V-cycle                                                            cycling strategy, see constants.hpp
 *      --F-cycle                                                            cycling strategy, see constants.hpp
 *      --relaxation-scheme F_relaxation                                     relaxation scheme, see constants.hpp
 *      --output-file Vcycle_error_l2                                        user-defined output file name (without suffix)
 */
int parse_commandline_options(appStruct &app, int argc, char** argv){
    // need to set number of levels before reading temporal coarsening factors
    bool numberOfLevelsSet = false;
    // parse commandline arguments
    for(int argIdx = 1; argIdx < argc; argIdx++){
        if(string(argv[argIdx]) == "--help"){
            if(app.world_rank == 0){
                cout << "Commandline options:" << endl << endl;
                cout << "    --help                                                              prints all commandline options" << endl;
                cout << "    --number-of-timesteps 1025                                          number of timesteps on fine grid (must contain time point corresponding to initial condition)" << endl;
                cout << "    --number-of-time-grids 2                                            number of levels in time grid hierarchy" << endl;
                cout << "    --coarsening-factors 2                                              temporal coarsening factors for level 0-->1, 1-->2, ..." << endl;
                cout << "                                                                        note: must not be used before option --number-of-time-grids" << endl;
                cout << "    --runge-kutta-method L_stable_SDIRK1                                Runge-Kutta method, if spatial eigenvalues are supplied, see constants.hpp" << endl;
                cout << "    --sample-complex-plane -10.0 1.0 -4.0 4.0                           sampling of complex plane with ranges [-10.0, 1.0] x [-4.0, 4.0]" << endl;
                cout << "    --complex-plane-sample-size 12 9                                    sampling of real and complex axis using 12 and 9 points" << endl;
                cout << "    --file-spatial-real-eigenvalues dteta_real.txt                      name of file that contains real part of spatial eigenvalues" << endl;
                cout << "    --file-spatial-complex-eigenvalues dteta_real.txt dteta_imag.txt    name of files that contain real and complex part of spatial eigenvalues" << endl;
                cout << "    --file-phi-real-eigenvalues lambda_real.txt                         name of file that contains real part of eigenvalues of Phi" << endl;
                cout << "    --file-phi-complex-eigenvalues lambda_real.txt lambda_imag.txt      name of file that contains complex part of eigenvalues of Phi" << endl;
                cout << "    --bound error_l2_tight_twogrid_upper_bound                          bound to evaluate, see constants.hpp" << endl;
                cout << "    --bound-on-level 1                                                  bound is evaluated on this level (options: 0, 1)" << endl;
                cout << "    --V-cycle                                                           cycling strategy, see constants.hpp" << endl;
                cout << "    --F-cycle                                                           cycling strategy, see constants.hpp" << endl;
                cout << "    --relaxation-scheme F_relaxation                                    relaxation scheme, see constants.hpp" << endl;
                cout << "    --output-file Vcycle_error_l2                                       user-defined output file name (without suffix)" << endl;
                cout << endl;
            }
            return 1;
        }else if(string(argv[argIdx]) == "--number-of-timesteps"){
            app.numberOfTimeSteps_l0 = stoi(argv[++argIdx]);
        }else if(string(argv[argIdx]) == "--number-of-time-grids"){
            app.numberOfLevels = stoi(argv[++argIdx]);
            app.coarseningFactors.set_size(app.numberOfLevels-1);
            app.coarseningFactors.fill(2);
            app.numberOfTimeSteps.set_size(app.numberOfLevels);
            numberOfLevelsSet = true;
        }else if(string(argv[argIdx]) == "--coarsening-factors"){
            if(!numberOfLevelsSet){
                cout << ">>>ERROR: Please set number of time grid levels before specifying coarsening factors." << endl;
                throw;
            }
            for(int level = 0; level < app.numberOfLevels-1; level++){
                app.coarseningFactors(level) = stoi(argv[++argIdx]);
            }
        }else if(string(argv[argIdx]) == "--runge-kutta-method"){
            argIdx++;
            if(string(argv[argIdx]) == "A_stable_LobattoIIIA_order2"){
                app.method = rkconst::A_stable_LobattoIIIA_order2;
            }else if(string(argv[argIdx]) == "A_stable_LobattoIIIA_order4"){
                app.method = rkconst::A_stable_LobattoIIIA_order4;
            }else if(string(argv[argIdx]) == "A_stable_LobattoIIIA_order6"){
                app.method = rkconst::A_stable_LobattoIIIA_order6;
            }else if(string(argv[argIdx]) == "A_stable_LobattoIIIB_order2"){
                app.method = rkconst::A_stable_LobattoIIIB_order2;
            }else if(string(argv[argIdx]) == "A_stable_LobattoIIIB_order4"){
                app.method = rkconst::A_stable_LobattoIIIB_order4;
            }else if(string(argv[argIdx]) == "A_stable_LobattoIIIB_order6"){
                app.method = rkconst::A_stable_LobattoIIIB_order6;
            }else if(string(argv[argIdx]) == "A_stable_LobattoIIIB_order8"){
                app.method = rkconst::A_stable_LobattoIIIB_order8;
            }else if(string(argv[argIdx]) == "A_stable_Gauss_order2"){
                app.method = rkconst::A_stable_Gauss_order2;
            }else if(string(argv[argIdx]) == "A_stable_Gauss_order4"){
                app.method = rkconst::A_stable_Gauss_order4;
            }else if(string(argv[argIdx]) == "A_stable_Gauss_order6"){
                app.method = rkconst::A_stable_Gauss_order6;
            }else if(string(argv[argIdx]) == "A_stable_SDIRK2"){
                app.method = rkconst::A_stable_SDIRK2;
            }else if(string(argv[argIdx]) == "A_stable_SDIRK3"){
                app.method = rkconst::A_stable_SDIRK3;
            }else if(string(argv[argIdx]) == "A_stable_SDIRK4"){
                app.method = rkconst::A_stable_SDIRK4;
            }else if(string(argv[argIdx]) == "L_stable_SDIRK1"){
                app.method = rkconst::L_stable_SDIRK1;
            }else if(string(argv[argIdx]) == "L_stable_SDIRK2"){
                app.method = rkconst::L_stable_SDIRK2;
            }else if(string(argv[argIdx]) == "L_stable_SDIRK3"){
                app.method = rkconst::L_stable_SDIRK3;
            }else if(string(argv[argIdx]) == "L_stable_SDIRK4"){
                app.method = rkconst::L_stable_SDIRK4;
            }else if(string(argv[argIdx]) == "L_stable_SDIRK5"){
                app.method = rkconst::L_stable_SDIRK5;
            }else if(string(argv[argIdx]) == "L_stable_RadauIIA_order1"){
                app.method = rkconst::L_stable_RadauIIA_order1;
            }else if(string(argv[argIdx]) == "L_stable_RadauIIA_order3"){
                app.method = rkconst::L_stable_RadauIIA_order3;
            }else if(string(argv[argIdx]) == "L_stable_RadauIIA_order5"){
                app.method = rkconst::L_stable_RadauIIA_order5;
            }else if(string(argv[argIdx]) == "L_stable_LobattoIIIC_order2"){
                app.method = rkconst::L_stable_LobattoIIIC_order2;
            }else if(string(argv[argIdx]) == "L_stable_LobattoIIIC_order4"){
                app.method = rkconst::L_stable_LobattoIIIC_order4;
            }else if(string(argv[argIdx]) == "L_stable_LobattoIIIC_order6"){
                app.method = rkconst::L_stable_LobattoIIIC_order6;
            }else if(string(argv[argIdx]) == "L_stable_LobattoIIIC_order8"){
                app.method = rkconst::L_stable_LobattoIIIC_order8;
            }else if(string(argv[argIdx]) == "LobattoIIICast_order2"){
                app.method = rkconst::LobattoIIICast_order2;
            }else if(string(argv[argIdx]) == "LobattoIIICast_order4"){
                app.method = rkconst::LobattoIIICast_order4;
            }else if(string(argv[argIdx]) == "LobattoIIICast_order6"){
                app.method = rkconst::LobattoIIICast_order6;
            }else if(string(argv[argIdx]) == "LobattoIIICast_order8"){
                app.method = rkconst::LobattoIIICast_order8;
            }else{
                cout << ">>>ERROR: Unknown Runge-Kutta method " << string(argv[argIdx]) << "." << endl;
                throw;
            }
        }else if(string(argv[argIdx]) == "--sample-complex-plane"){
            app.sampleComplexPlane  = true;
            app.min_dteta_real_l0   = stod(argv[++argIdx]);
            app.max_dteta_real_l0   = stod(argv[++argIdx]);
            app.min_dteta_imag_l0   = stod(argv[++argIdx]);
            app.max_dteta_imag_l0   = stod(argv[++argIdx]);
        }else if(string(argv[argIdx]) == "--complex-plane-sample-size"){
            app.sampleComplexPlane  = true;
            app.numberOfRealSamples = stoi(argv[++argIdx]);
            app.numberOfImagSamples = stoi(argv[++argIdx]);
        }else if(string(argv[argIdx]) == "--file-spatial-real-eigenvalues"){
            app.sampleComplexPlane              = false;
            app.fileSpatialEigenvalues          = true;
            app.fileNameSpatialEigenvaluesReal  = string(argv[++argIdx]);
        }else if(string(argv[argIdx]) == "--file-spatial-complex-eigenvalues"){
            app.sampleComplexPlane              = false;
            app.fileSpatialEigenvalues          = true;
            app.fileComplexEigenvalues          = true;
            app.fileNameSpatialEigenvaluesReal  = string(argv[++argIdx]);
            app.fileNameSpatialEigenvaluesImag  = string(argv[++argIdx]);
        }else if(string(argv[argIdx]) == "--file-phi-real-eigenvalues"){
            app.sampleComplexPlane          = false;
            app.filePhiEigenvalues          = true;
            app.fileNamePhiEigenvaluesReal  = string(argv[++argIdx]);
        }else if(string(argv[argIdx]) == "--file-phi-complex-eigenvalues"){
            app.sampleComplexPlane          = false;
            app.filePhiEigenvalues          = true;
            app.fileComplexEigenvalues      = true;
            app.fileNamePhiEigenvaluesReal  = string(argv[++argIdx]);
            app.fileNamePhiEigenvaluesImag  = string(argv[++argIdx]);
        }else if(string(argv[argIdx]) == "--bound"){
            argIdx++;
            if(string(argv[argIdx]) == "error_l2_upper_bound"){
                app.bound = mgritestimate::error_l2_upper_bound;
            }else if(string(argv[argIdx]) == "error_l2_sqrt_upper_bound"){
                app.bound = mgritestimate::error_l2_sqrt_upper_bound;
            }else if(string(argv[argIdx]) == "error_l2_sqrt_expression_upper_bound"){
                app.bound = mgritestimate::error_l2_sqrt_expression_upper_bound;
            }else if(string(argv[argIdx]) == "error_l2_tight_twogrid_upper_bound"){
                app.bound = mgritestimate::error_l2_tight_twogrid_upper_bound;
            }else if(string(argv[argIdx]) == "error_l2_tight_twogrid_bound"){
                app.bound = mgritestimate::error_l2_tight_twogrid_bound;
            }else if(string(argv[argIdx]) == "error_l2_tight_twogrid_bound_single_iter"){
                app.bound = mgritestimate::error_l2_tight_twogrid_bound_single_iter;
            }else if(string(argv[argIdx]) == "error_l2_approximate_lower_bound"){
                app.bound = mgritestimate::error_l2_approximate_lower_bound;
            }else if(string(argv[argIdx]) == "error_l2_sqrt_approximate_lower_bound"){
                app.bound = mgritestimate::error_l2_sqrt_approximate_lower_bound;
            }else if(string(argv[argIdx]) == "error_l2_tight_twogrid_lower_bound"){
                app.bound = mgritestimate::error_l2_tight_twogrid_lower_bound;
            }else if(string(argv[argIdx]) == "error_l2_sqrt_expression_approximate_rate"){
                app.bound = mgritestimate::error_l2_sqrt_expression_approximate_rate;
            }else if(string(argv[argIdx]) == "lower_bound_southworth2019_equation63"){
                app.bound = mgritestimate::lower_bound_southworth2019_equation63;
            }else if(string(argv[argIdx]) == "upper_bound_southworth2019_equation63"){
                app.bound = mgritestimate::upper_bound_southworth2019_equation63;
            }else if(string(argv[argIdx]) == "residual_l2_upper_bound"){
                app.bound = mgritestimate::residual_l2_upper_bound;
            }else if(string(argv[argIdx]) == "residual_l2_sqrt_upper_bound"){
                app.bound = mgritestimate::residual_l2_sqrt_upper_bound;
            }else if(string(argv[argIdx]) == "residual_l2_lower_bound"){
                app.bound = mgritestimate::residual_l2_lower_bound;
            }else{
                cout << "ERROR: Unknown bound " << string(argv[argIdx]) << "." << endl;
                throw;
            }
        }else if(string(argv[argIdx]) == "--bound-on-level"){
            app.theoryLevel = stoi(argv[++argIdx]);
        }else if(string(argv[argIdx]) == "--V-cycle"){
            app.cycle = mgritestimate::V_cycle;
        }else if(string(argv[argIdx]) == "--F-cycle"){
            app.cycle = mgritestimate::F_cycle;
        }else if(string(argv[argIdx]) == "--relaxation-scheme"){
            argIdx++;
            if((string(argv[argIdx]) == "F_relaxation") || (string(argv[argIdx]) == "0")){
                app.relax = mgritestimate::F_relaxation;
            }else if((string(argv[argIdx]) == "FCF_relaxation") || (string(argv[argIdx]) == "1")){
                app.relax = mgritestimate::FCF_relaxation;
            }else if((string(argv[argIdx]) == "FCFCF_relaxation") || (string(argv[argIdx]) == "2")){
                app.relax = mgritestimate::FCFCF_relaxation;
            }else if((string(argv[argIdx]) == "FCFCFCF_relaxation") || (string(argv[argIdx]) == "3")){
                app.relax = mgritestimate::FCFCFCF_relaxation;
            }else if((string(argv[argIdx]) == "FCFCFCFCF_relaxation") || (string(argv[argIdx]) == "4")){
                app.relax = mgritestimate::FCFCFCFCF_relaxation;
            }
        }else if(string(argv[argIdx]) == "--output-file"){
            app.userOutputFile      = true;
            app.userOutputFileName  = string(argv[++argIdx]);
        }else{
            cout << ">>>ERROR: Unknown argument " << string(argv[argIdx]) << endl;
            throw;
        }
    }
    // sanity checks for user-defined parameters
    if((app.sampleComplexPlane && app.fileSpatialEigenvalues)
        || (app.sampleComplexPlane && app.filePhiEigenvalues)
        || (app.fileSpatialEigenvalues && app.filePhiEigenvalues)){
        cout << ">>>ERROR: Defined multiple input sources." << endl;
        throw;
    }
    // set number of time steps based on fine grid and coarsening factors
    app.numberOfTimeSteps(0) = app.numberOfTimeSteps_l0;
    for(int level = 1; level < app.numberOfLevels; level++){ app.numberOfTimeSteps(level) = (app.numberOfTimeSteps(level-1) - 1) / app.coarseningFactors(level-1) + 1; }
    return 0;
}

/**
 *  computes or imports real/complex eigenvalues
 */
void setget_eigenvalues(appStruct &app){
    // spatial and Phi eigenvalues
    if(app.sampleComplexPlane || app.fileComplexEigenvalues){
        app.dtetac  = new arma::Col<arma::cx_double>*[app.numberOfLevels];
        app.lambdac = new arma::Col<arma::cx_double>*[app.numberOfLevels];
    }else{
        app.dtetar  = new arma::Col<double>*[app.numberOfLevels];
        app.lambdar = new arma::Col<double>*[app.numberOfLevels];
    }
    // sample complex plane for dt*eta
    if(app.sampleComplexPlane){
        int multiplyBy = 1;
        for(int level = 0; level < app.numberOfLevels; level++){
            if(level > 0){
                multiplyBy *= app.coarseningFactors(level-1);
            }
            sample_complex_plane(app.method,
                                 app.min_dteta_real_l0*multiplyBy, app.max_dteta_real_l0*multiplyBy, app.min_dteta_imag_l0*multiplyBy, app.max_dteta_imag_l0*multiplyBy,
                                 app.numberOfRealSamples, app.numberOfImagSamples,
                                 app.dtetac[level], app.lambdac[level]);
            // z = new arma::Col<arma::cx_double>(numberOfRealSamples*numberOfImagSamples);
        }
        if(app.world_rank == 0){
            arma::mat(join_rows(real(*app.dtetac[0]), imag(*app.dtetac[0]))).save("dteta_l0.txt", arma::raw_ascii);
        }
    // import spatial eigenvalues
    }else if(app.fileSpatialEigenvalues){
        if(app.fileComplexEigenvalues){
            // read real and complex part of spatial eigenvalues
            arma::mat realEigs;
            arma::mat imagEigs;
            realEigs.load(app.fileNameSpatialEigenvaluesReal, arma::raw_ascii);
            imagEigs.load(app.fileNameSpatialEigenvaluesImag, arma::raw_ascii);
            int numberOfSamples = realEigs.n_rows;
            // check whether we have at least as many columns as number of levels
            if(realEigs.n_cols < app.numberOfLevels){
                cout << ">>>ERROR: Supplied eigenvalues are invalid for " << app.numberOfLevels << " levels." << endl;
                throw;
            }
            if(imagEigs.n_cols < app.numberOfLevels){
                cout << ">>>ERROR: Supplied eigenvalues are invalid for " << app.numberOfLevels << " levels." << endl;
                throw;
            }
            // store spatial eigenvalues and compute eigenvalues of Phi for given method
            for(int level = 0; level < app.numberOfLevels; level++){
                arma::Col<arma::cx_double> dteta(realEigs.col(level), imagEigs.col(level));
                app.dtetac[level]       = new arma::Col<arma::cx_double>(numberOfSamples);
                app.lambdac[level]      = new arma::Col<arma::cx_double>(numberOfSamples);
                *(app.dtetac[level])    = dteta;
                stability_function(app.method, app.dtetac[level], app.lambdac[level]);
            }
        }else{
            // read real part of spatial eigenvalues
            arma::mat realEigs;
            realEigs.load(app.fileNameSpatialEigenvaluesReal, arma::raw_ascii);
            int numberOfSamples = realEigs.n_rows;
            // check whether we have at least as many columns as number of levels
            if(realEigs.n_cols < app.numberOfLevels){
                cout << ">>>ERROR: Supplied eigenvalues are invalid for " << app.numberOfLevels << " levels." << endl;
                throw;
            }
            // read/store spatial eigenvalues and compute eigenvalues of Phi for given method
            for(int level = 0; level < app.numberOfLevels; level++){
                arma::Col<double> dteta(realEigs.col(level));
                app.dtetar[level]       = new arma::Col<double>(numberOfSamples);
                app.lambdar[level]      = new arma::Col<double>(numberOfSamples);
                *(app.dtetar[level])    = dteta;
                stability_function(app.method, app.dtetar[level], app.lambdar[level]);
            }
        }
    // import Phi eigenvalues
    }else if(app.filePhiEigenvalues){
        if(app.fileComplexEigenvalues){
            // read real and complex part of spatial eigenvalues
            arma::mat realEigs;
            arma::mat imagEigs;
            realEigs.load(app.fileNamePhiEigenvaluesReal, arma::raw_ascii);
            imagEigs.load(app.fileNamePhiEigenvaluesImag, arma::raw_ascii);
            int numberOfSamples = realEigs.n_rows;
            // check whether we have at least as many columns as number of levels
            if(realEigs.n_cols < app.numberOfLevels){
                cout << ">>>ERROR: Supplied eigenvalues are invalid for " << app.numberOfLevels << " levels." << endl;
                throw;
            }
            if(imagEigs.n_cols < app.numberOfLevels){
                cout << ">>>ERROR: Supplied eigenvalues are invalid for " << app.numberOfLevels << " levels." << endl;
                throw;
            }
            // store spatial eigenvalues and compute eigenvalues of Phi for given method
            for(int level = 0; level < app.numberOfLevels; level++){
                arma::Col<arma::cx_double> lambda(realEigs.col(level), imagEigs.col(level));
                app.lambdac[level]      = new arma::Col<arma::cx_double>(numberOfSamples);
                *(app.lambdac[level])   = lambda;
            }
        }else{
            // read real part of spatial eigenvalues
            arma::mat realEigs;
            realEigs.load(app.fileNamePhiEigenvaluesReal, arma::raw_ascii);
            int numberOfSamples = realEigs.n_rows;
            // check whether we have at least as many columns as number of levels
            if(realEigs.n_cols < app.numberOfLevels){
                cout << ">>>ERROR: Supplied eigenvalues are invalid for " << app.numberOfLevels << " levels." << endl;
                throw;
            }
            // store spatial eigenvalues and compute eigenvalues of Phi for given method
            for(int level = 0; level < app.numberOfLevels; level++){
                arma::Col<double> lambda(realEigs.col(level));
                app.lambdar[level]      = new arma::Col<double>(numberOfSamples);
                *(app.lambdar[level])   = lambda;
            }
        }
    }
}
