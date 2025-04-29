/*********************************************
 Program: kernel_absolute_errors
 Author: Ll. Hurtado-Gil (CEU San Pablo)
 Creation date: 01/06/2019
 Last modified: 19/08/2019
 Version: 0.1
 
 Program to compute in the simplest way the kernel smoothed distribution
 of the absolute error distribution of a fitted three dimensional point process.
 
 We will need as input the data and dummy points together with the residuals,
 the voronoi weights and the Sigma distribution value at each location. The
 parameters defining the calculation (definition of window geometry) are also required.

 Compilation:
 g++ -o kernel_absolute_residuals kernel_absolute_residuals.cpp
 
 Version history:
 
 V. 0.1 (01/06/2019): Initial version.
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
char residual_dens_file[80];        //Name of output file containing the residuals density field
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
int density(double **, int , double *, double *, double, double, int , int);
double gaussk(double *, int , double , double );
int write_fields(double **, double *, int , int , char *);

/* MAIN FUNCTION */
int main(int argc, char ** argv)
{
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
    double **process;
    
    process  = (double **) malloc((N+M+1)*sizeof(double)); assert(process); //Read all
    // input:   x    y   z   r    lambda  voro
    for(i=1;i<=N;i++)
    {
        process[i]  = (double *) malloc((7)*sizeof(double)); assert(process[i]);
        for(j=1;j<=6;j++) {
            process[i][j] = 0.0;
        }
    }
    for(i=1;i<=M;i++)
    {
        process[N+i] = (double *) malloc((7)*sizeof(double)); assert(process[N+i]);
        for(j=1;j<=6;j++) {
            process[N+i][j] = 0.0;
        }
    }

    // Arrays for kernel density and residuals density field
    double *kernel, *residual_dens;
    
    residual_dens = (double *) malloc((M+1)*sizeof(double)); assert(residual_dens);
    kernel = (double *) malloc((D+1)*sizeof(double)); assert(kernel);
    
    // Kernel array
    ndata = gaussk(kernel, D, stepd, bandwidth);
    
    // Read all input data
    ndata = readcat(process, in_data_file, N, M);
    
    // Compute density field for residuals
    ndata = density(process, 6, residual_dens, kernel, stepd, bandwidth, N, M);

    // Write residuals density field
    ndata = write_fields(process, residual_dens, N, M, residual_dens_file);
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
                    
                case 'r': strcpy(residual_dens_file, argv[++i]);
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
            all[i][6] = 1.0;
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


//FUNCTION: 'density'
int density(double **all, int col, double *dens, double *kernel, double stepd, double r, int N, int M)
{
    int i,j,factor, index;
    double stepx, stepy, stepz, a,b,c, x,y,z, d, d1,d2,d3, g,sum1,sum2, volumeunit;
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
    for(i=1; i<=M; i++){
        dens[i] = 0;
    }
    
    //Start loop over grid locations
    for(i=1;i<=M;i++)
    {
        //Tell people what we are doing
        if(Verbose && (i%factor==0)){
            fprintf(stdout, "  Doing density field calculations. Now working in location %d of %d...\n", i, M);
        }
        
        a = all[i+N][1];
        b = all[i+N][2];
        c = all[i+N][3];
        sum1=0.0;
        for(j=1;j<=N;j++)
        {
            x = all[j][1];
            y = all[j][2];
            z = all[j][3];
            
            d1 = a-x;
            d2 = b-y;
            d3 = c-z;
            d = sqrt(pow(d1,2) + pow(d2,2) + pow(d3,2));
            index = (int)round(d/stepd) + 1;
            g = kernel[index]; //gaussk(d1,d2,d3,r);
            sum1 = sum1 + g*all[j][col];
        }
        sum2=0.0;
        for(j=(N+1);j<=(M+N);j++)
        {
            x = all[j][1];
            y = all[j][2];
            z = all[j][3];
            
            d1 = a-x;
            d2 = b-y;
            d3 = c-z;
            d = sqrt(pow(d1,2) + pow(d2,2) + pow(d3,2));
            index = (int)round(d/stepd) + 1;
            g = kernel[index]; //gaussk(d1,d2,d3,r);
            sum2 = sum2 + g*all[j][col]*volumeunit;
        }
        dens[i] = sum1-sum2;
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

//FUNCTION: 'gaussk'
/*double gaussk(double a, double b, double c, double r)
{
    double p;
    
    p = exp(-(pow(a,2)+pow(b,2)+pow(c,2))/(2*r*r))/(pow(2*PI,3/2)*r*r*r);
    
    return p;
}*/

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
        fprintf(fout, "%g  %g  %g  %g\n", all[N+i][1], all[N+i][2], all[N+i][3], dens[i]);
    }
    fclose(fout);
    if(Verbose){
        fprintf(stdout, "Results written to file %s\n", outfile);
    }
    return i;
}
