#include <mpi.h>
#include <iostream>
#include <ctime>
#include "armadillo"
#include "io_routines.hpp"
#include "bound_routines.hpp"
#include "types.hpp"

using namespace arma;
using namespace std;


int main(int argc, char** argv){
    MPI_Init(NULL, NULL);
    appStruct app;
    MPI_Comm_size(MPI_COMM_WORLD, &app.world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &app.world_rank);
    clock_t begin,end;
    int errCode = 0;
    if(app.world_rank == 0){
        cout << endl << "Armadillo version: " << arma_version::as_string() << endl << endl;
    }

    // MGRIT settings - set defaults and parse commandline options
    initalize_app_struct(app);
    errCode = parse_commandline_options(app, argc, argv);
    if(errCode != 0){ MPI_Finalize(); return 0; }
    // read eigenvalues based on commandline options
    setget_eigenvalues(app);

    begin = clock();
    // evaluate bound for residual or error propagator for complex eigenvalue case
    if(app.sampleComplexPlane || app.fileComplexEigenvalues){
        get_propagator_bound(app.bound, app.theoryLevel, app.cycle, app.relax, app.numberOfTimeSteps, app.coarseningFactors, app.lambdac, app.estimate);
    // evaluate bound for residual or error propagator for real eigenvalue case
    }else{
        get_propagator_bound(app.bound, app.theoryLevel, app.cycle, app.relax, app.numberOfTimeSteps, app.coarseningFactors, app.lambdar, app.estimate);
    }
    end = clock();
    cout << "Rank " << app.world_rank << " / " << app.world_size << " - Elapsed time: " << double(end-begin)/CLOCKS_PER_SEC << " seconds" << endl << endl;

    // export results
    export_estimates(app);

    MPI_Finalize();
    return 0;
}
