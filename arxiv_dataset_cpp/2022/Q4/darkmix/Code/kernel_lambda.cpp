/*********************************************
 Program: kernel_lambda
 Author: Ll. Hurtado-Gil (eDreams ODIGEO)
 Creation date: 01/06/2019
 Last modified: 19/08/2019
 Version: 0.1

 Program to compute in the simplest way the kernel smoothed distribution
 of an estimated density model of a fitted three dimensional point process.

 We will need as input the data and dummy points together with the residuals,
 the voronoi weights and the Sigma distribution value at each location. The
 parameters defining the calculation (definition of window geometry) are also required.

 Compilation:
 g++ -o kernel_lambda kernel_lambda.cpp

 Version history:

 V. 0.1 (16/11/2019): Initial version.
 **********************************************/

/* INCLUDE LIBRARIES */

#include <assert.h>
#include <string.h>
#include <time.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <locale>
#include <sstream>
#define PI 3.14159265358979323846
using namespace std;

//Global parameters (these are for input arguments)
char in_data_file[80];              //Name of input file containing the data
char model_dens_file[80];        //Name of output file containing the residuals density field
int Verbose ;                       //Fixes the verbosity level
//Point process characteristc
int N;             //Number of points in the data set
int M;             //Number of points in the used quadrature
double bandwidth;  //Gaussian kernel bandwidth
int *grid;      //Number of steps per grid dimension
double *window;    //Minimum and maximum values of the window

/* FUNCTION DECLARATIONS */
static void usage(char **);
void get_args(int , char **);
int readcat(double **, char *, int , int );
int density(double **, double **, int, int , double *, double *, double, double, int , int);
double gaussk(double *, int , double , double );
int write_fields(double **, double *, int , int , char *);
int jitter(double **, double **, int, int, double);

/* MAIN FUNCTION */
int main(int argc, char ** argv)
{

    //Set the output buffers to avoid buffering
    //(so we can follow easier the development of the program)
    setvbuf(stdout, NULL, _IOLBF, 1024);
    setvbuf(stderr, NULL, _IOLBF, 1024);

    std::cout << std::fixed;
    std::cout << std::setprecision(15);

    // Define defaults
    Verbose = 1;
    strcpy(in_data_file, "residuals_smoothed.txt");
    N = 1000;
    M = 10000;
    bandwidth = 1.0;
    grid = (int *) malloc(4*sizeof(int)); assert(grid);
    grid[1] = 10; grid[2] = 10; grid[3] = 10;
    window = (double *) malloc(7*sizeof(double)); assert(window);
    window[1] = 0; window[2] = 1;
    window[3] = 0; window[4] = 1;
    window[5] = 0; window[6] = 1;

    get_args(argc, argv);

    //Read arguments:

    /* Variable declaration */
    int i,j,ndata,D;
    double diagonal,stepd;

    // Values where the gaussian kernel is calculated
    diagonal = ceil(sqrt(pow(window[6]-window[5],2) + pow(window[4]-window[3],2) + pow(window[2]-window[1],2)));
    stepd = 0.001;
    D = diagonal/stepd;

    //Data and Dummy points plus residuals, voronoi weights and Sigma density function
    double **process, **Ujitter;

    fprintf(stdout, "Vales %d %d \n", N, M);

    process  = (double **) malloc((N+M+1)*sizeof(double)); assert(process); //Read all
    Ujitter  = (double **) malloc((N+M+1)*sizeof(double)); assert(Ujitter); //Read all
    // input:   x    y   z   r   voro   sigma
    for(i=1;i<=N;i++)
    {
        process[i]  = (double *) malloc((7)*sizeof(double)); assert(process[i]);
        Ujitter[i]  = (double *) malloc((4)*sizeof(double)); assert(Ujitter[i]);
        for(j=1;j<=6;j++) {
            process[i][j] = 0.0;
        }
        for(j=1;j<=3;j++) {
            Ujitter[i][j] = 0.0;
        }
    }
    for(i=1;i<=M;i++)
    {
        process[N+i] = (double *) malloc((7)*sizeof(double)); assert(process[N+i]);
        Ujitter[N+i] = (double *) malloc((4)*sizeof(double)); assert(Ujitter[N+i]);
        for(j=1;j<=6;j++) {
            process[N+i][j] = 0.0;
        }
        for(j=1;j<=3;j++) {
            Ujitter[N+i][j] = 0.0;
        }
    }

    // Arrays for kernel density and residuals density field
    double *kernel, *model_dens;

    model_dens = (double *) malloc((N+M+1)*sizeof(double)); assert(model_dens);
    kernel = (double *) malloc((D+1)*sizeof(double)); assert(kernel);

    // Kernel array
    ndata = gaussk(kernel, D, stepd, bandwidth);

    // Read all input data
    fprintf(stdout, "Read");
    ndata = readcat(process, in_data_file, N, M);

    // Jitter
    fprintf(stdout, "Jitter");
    ndata = jitter(process, Ujitter, N, M, bandwidth);

    // Compute density field for residuals
    fprintf(stdout, "Density");
    ndata = density(process, Ujitter, 5, 6, model_dens, kernel, stepd, bandwidth, N, M);

    // Write residuals density field
    ndata = write_fields(process, model_dens, N, M, model_dens_file);
}

/* DEFINITION OF FUNCTIONS */

//FUNCTION: 'usage'
static void usage(char **argv)
{
    /* This functions explains the usage of the program
     (basically, how to pass the different arguments)
     */

    fprintf(stderr, "Usage: %s options  outprefix\n\n", argv[0]);
    fprintf(stderr, "   where options = \n\n");

    fprintf(stderr, "   -i Residuals file name\n");
    fprintf(stderr, "    Name of the file containing the residuals.\n");
    fprintf(stderr, "    The file should contain 6 columns: X, Y, Z (arbitrary units, as long as consistent), Residuals, Voronoi weights and Sigma values.\n");
    fprintf(stderr, "    Default is %s.\n\n", in_data_file);

    fprintf(stderr, "   -n\n");
    fprintf(stderr, "    Number of data points.\n");

    fprintf(stderr, "   -m\n");
    fprintf(stderr, "    Number of dummy points.\n");

    fprintf(stderr, "   -g\n");
    fprintf(stderr, "    Number of tiles in the dummy grid. Three values expected.\n");

    fprintf(stderr, "   -w\n");
    fprintf(stderr, "    Minimum and maximum values of the window. Six values expected.\n");

    fprintf(stderr, "   -b\n");
    fprintf(stderr, "    Used bandwidth.\n");

    fprintf(stderr, "   -r\n");
    fprintf(stderr, "    Name of the output file.\n");

    exit(-1);
}

//FUNCTION: 'get_args'
void get_args(int argc, char **argv)
{
    /* This functions reads the needed CL arguments into global variables.
     In case of problems, calls 'usage()'.
     */

    int i,k,start;

    //Only do something if there are arguments
    if(argc == 1){
        usage(argv);
    }

    //Check for a switch
    //Start at i=1 to skip the command name
    for(i=1;i<argc;i++) {
        if(argv[i][0] == '-') {
            //Use next character to decide what to do
            switch(argv[i][1]){

                case 'i': strcpy(in_data_file, argv[++i]);
                    break;

                case 'n': N = atoi(argv[++i]);
                    break;

                case 'm': M = atoi(argv[++i]);
                    break;

                case 'g':
                    k = 1; start=++i;
                    for (i=start; i < (start+3); ++i) {
                        grid[k]=atoi(argv[i]);
                        k++;
                    }
                    i = i - 1;
                    break;

                case 'w':
                    k=1; start = ++i;
                    for (i=start; i < (start+6); ++i) {
                        window[k]=atof(argv[i]);
                        k++;
                    }
                    i = i - 1;
                    break;

                case 'b': bandwidth = atof(argv[++i]);
                    break;

                case 'r': strcpy(model_dens_file, argv[++i]);
                    break;

                case '?': usage(argv);
                    break;

                default: usage(argv);
                    break;
            }
        }
    }
}

//FUNCTION: 'readcat'
int readcat(double **all, char *filename, int N, int M)
{
    int i=0;
    FILE *fin;
    char linestr[100];
    double x, y, z, r, w, s;
    fin = fopen(filename, "r"); assert(fin);

    //Read lines in file one by one
    for(i=1;i<=N;i++) {
        //while(fgets(linestr, 100, fin)!=NULL){
        fgets(linestr, 100, fin);
        //Verbose control
        if(Verbose && (i%10000==0)){
            fprintf(stdout, "  Reading catalogue, line %d\n", i);
        }
        //Ignore lines starting with comment character
        if(linestr[0]!='#'){
            //If everything's OK, read data to appropriate arrays
            sscanf(linestr, "%lf  %lf  %lf %lf %lf %lf", &x, &y, &z, &r, &w, &s);
            all[i][1] = x;
            all[i][2] = y;
            all[i][3] = z;
            all[i][4] = r;
            all[i][5] = w;
            all[i][6] = s;
        }
    }
    for(i=1;i<=M;i++) {
        //while(fgets(linestr, 100, fin)!=NULL){
        fgets(linestr, 100, fin);
        //Verbose control
        if(Verbose && (i%10000==0)){
            fprintf(stdout, "  Reading catalogue, line %d\n", i);
        }
        //Ignore lines starting with comment character
        if(linestr[0]!='#'){
            //If everything's OK, read data to appropriate arrays
            sscanf(linestr, "%lf  %lf  %lf %lf %lf %lf", &x, &y, &z, &r, &w, &s);
            all[N+i][1] = x;
            all[N+i][2] = y;
            all[N+i][3] = z;
            all[N+i][4] = r;
            all[N+i][5] = w;
            all[N+i][6] = s;
        }
    }

    if(Verbose){
        fprintf(stdout, "  Read %d points from catalogue %s\n", i-1, filename);
    }
    //Finally, return number of points read
    return i-1;
}

//FUNCTION: 'jitter'
int jitter(double **process, double **Ujitter, int N, int M, double w)
{
    int i;
    // Data for function density
    for(i=1;i<=N;i++)
    {
        Ujitter[i][1] = process[i][1] + ((double) rand() / (RAND_MAX))*2*w - w;
        Ujitter[i][2] = process[i][2] + ((double) rand() / (RAND_MAX))*2*w - w;
        Ujitter[i][3] = process[i][3] + ((double) rand() / (RAND_MAX))*2*w - w;
    }
    for(i=1;i<=M;i++)
    {
        Ujitter[N+i][1] = process[N+i][1] + ((double) rand() / (RAND_MAX))*2*w - w;
        Ujitter[N+i][2] = process[N+i][2] + ((double) rand() / (RAND_MAX))*2*w - w;
        Ujitter[N+i][3] = process[N+i][3] + ((double) rand() / (RAND_MAX))*2*w - w;
    }
    return M;
}

//FUNCTION: 'density'
int density(double **all, double **Ujitter, int col1, int col2, double *dens, double *kernel, double stepd, double r, int N, int M)
{
    fprintf(stdout, "Density");
    int i,j,factor, index;
    double stepx, stepy, stepz, a,b,c, d, d1,d2,d3, g,sum1,sum2, volumeunit, term;
    stepx = (window[2]-window[1])/grid[1];
    stepy = (window[4]-window[3])/grid[2];
    stepz = (window[6]-window[5])/grid[3];
    volumeunit = stepx*stepy*stepz;

    if( M > 20 ){
        factor = M/10;
    }else{
        factor = 1;
    }

    //First of all, set counts to zero
    for(i=1; i<=M+N; i++){
        dens[i] = 0;
    }
    fprintf(stdout, "%d %d \n", N, M);

    //Start loop over grid locations
#pragma omp parallel for private(i, a, b, c, sum1, sum2, j, d1, d2, d3, d, index, term) \
      shared(M, dens, Verbose, factor, all, N, Ujitter, stepd, kernel, col1, col2, volumeunit)
    for(i=1;i<=M;i++)  // Mdfy: i=M;i<=M+N;i++; i=i-N
    {

        //Tell people what we are doing
        if(Verbose && (i%factor==0)){
            fprintf(stdout, "  Doing density field calculations. Now working in location %d of %d...\n", i, M);
        }

        a = all[i+N][1];
        b = all[i+N][2];
        c = all[i+N][3];
        sum1=0.0;
        sum2=0.0;
        for(j=1;j<=N;j++)
        {
            d1 = a-Ujitter[j][1];
            d2 = b-Ujitter[j][2];
            d3 = c-Ujitter[j][3];
            d = sqrt(pow(d1,2) + pow(d2,2) + pow(d3,2));
            index = (int)round(d/stepd) + 1;
            term = kernel[index]*all[j][col1]/volumeunit;
            sum1 = sum1 + term;
            sum2 = sum2 + term*all[j][col2];
        }
        for(j=(N+1);j<=(M+N);j++)
        {
            d1 = a-Ujitter[j][1];
            d2 = b-Ujitter[j][2];
            d3 = c-Ujitter[j][3];
            d = sqrt(pow(d1,2) + pow(d2,2) + pow(d3,2));
            index = (int)round(d/stepd) + 1;
            term = kernel[index]*all[j][col1]/volumeunit;
            sum1 = sum1 + term;
            sum2 = sum2 + term*all[j][col2];
        }
        dens[N+i] = sum2/sum1;
//        fprintf(stdout, "%g  %g  %g  %g\n", a, b, c, dens[N+i]);
    }
    return i;
}

//FUNCTION: 'gaussk'
double gaussk(double *kernel, int D, double stepd, double r)
{
    int i;
    double x;

    for(i=1;i<=D;i++)
    {
        x = i*stepd;
        kernel[i] = exp(-x*x/(2*r*r))/(pow(2*PI,3/2)*r*r*r);
    }
    return D;
}

//FUNCTION: 'write_fields'
int write_fields(double **all, double *dens, int N, int M, char *filename)
{
    int i;
    FILE *fout;
    char outfile[95];

    strcpy(outfile, filename);
    fout = fopen(outfile, "w"); assert(fout);

    for(i=1;i<=M;i++)
    {
        fprintf(fout, "%g  %g  %g  %g\n", all[N+i][1], all[N+i][2], all[N+i][3], dens[N+i]);
    }
    fclose(fout);
    if(Verbose){
        fprintf(stdout, "Results written to file %s\n", outfile);
    }
    return i;
}
