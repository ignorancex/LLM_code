# include "types.hpp"

/**
 *  set MGRIT parameters and parameters for convergence analysis to default values
 */
void initalize_app_struct(appStruct &app){
    app.numberOfTimeSteps_l0 = 1025;
    app.numberOfLevels = 2;
    app.sampleComplexPlane = true;
    app.coarseningFactors;
    app.numberOfTimeSteps;
    app.coarseningFactors.set_size(app.numberOfLevels-1);
    app.coarseningFactors.fill(2);
    app.numberOfTimeSteps.set_size(app.numberOfLevels);
    app.method = rkconst::L_stable_SDIRK1;
    app.min_dteta_real_l0 = -10.0;
    app.max_dteta_real_l0 =   1.0;
    app.min_dteta_imag_l0 =  -4.0;
    app.max_dteta_imag_l0 =   4.0;
    app.numberOfRealSamples = 12;
    app.numberOfImagSamples = 9;
    app.bound = mgritestimate::residual_l2_upper_bound;
    app.theoryLevel = 1;
    app.cycle = mgritestimate::V_cycle;
    app.relax = mgritestimate::F_relaxation;
    app.fileSpatialEigenvalues = false;
    app.filePhiEigenvalues = false;
    app.fileComplexEigenvalues = false;
    app.userOutputFile = false;
}
