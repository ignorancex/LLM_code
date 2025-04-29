/*********************************************
 Program: kernel_fourier_residuals
 Author: Ll. Hurtado-Gil (CEU San Pablo)
 Creation date: 01/06/2019
 Last modified: 19/08/2019
 Version: 0.1
 
 Program to compute the kernel smoothed distribution of the absolute error 
 distribution of a fitted three dimensional point process. This program used Fourier
 transformatios and is faster than kernel_absolute_residuals.cpp. However, due to a binning
 simplification, catastrophic events in the calculations may occur.
 
 We will need as input the data and dummy points together with the residuals,
 the voronoi weights and the Sigma distribution value at each location. The
 parameters defining the calculation (definition of window geometry) are also required.

 Compilation:
 g++ -o kernel_fourier_residuals kernel_fourier_residuals.cpp -lfftw3 -lm

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
#include <fftw3.h>
#include <complex.h>
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
int readcat(double **, double **, double *, double *, char *, int , int );
double gauss(double , double );
void write(fftw_complex *, int, int, int, char *);
//int write_fields(double **, double *, int , char *);
int fdensity(double **, double **, double *, int , int , int *, double , double *, char *);
void findinterval(double **, double **, double *, fftw_complex *, int, int, int, int, int, double, double, double, double *);
bool inside(double, double, double, double *);

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
    double **data, **dummy, *residuals;
    
    data  = (double **) malloc((N+1)*sizeof(double)); assert(data); //Read all
    // input:   x    y   z   r    lambda  voro
    for(i=1;i<=N;i++)
    {
        data[i]  = (double *) malloc(4*sizeof(double)); assert(data[i]);
        for(j=1;j<=3;j++) {
            data[i][j] = 0.0;
        }
    }
    dummy  = (double **) malloc((M+1)*sizeof(double)); assert(dummy); //Read all
    for(i=1;i<=M;i++)
    {
        dummy[i] = (double *) malloc(4*sizeof(double)); assert(dummy[i]);
        for(j=1;j<=3;j++) {
            dummy[i][j] = 0.0;
        }
    }
    residuals = (double *) malloc((N+M+1)*sizeof(double)); assert(residuals);
    
    // Arrays for residuals density field
    double *residual_dens;
    
    residual_dens = (double *) malloc((M+1)*sizeof(double)); assert(residual_dens);

    // Read all input data
    ndata = readcat(data, dummy, residuals, window, in_data_file, N, M);
    
    // Relocate window in corner (0,0,0)
    window[2] = window[2] - window[1]; window[1] = 0.0;
    window[4] = window[4] - window[3]; window[3] = 0.0;
    window[6] = window[6] - window[5]; window[5] = 0.0;

    // Compute density field for residuals
    cout << "Fast Residuals Calculation (" << N+M << "): " << endl;
    ndata = fdensity(dummy, data, residuals, M, N, grid, bandwidth, window, residual_dens_file);
    
    // Write residuals density field
    //ndata = write_fields(dummy, residual_dens, M, residual_dens_file);
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
int readcat(double **data, double **dummy, double *residuals, double *window, char *filename, int N, int M)
{
    int i=0;
    FILE *fin;
    char linestr[100];
    double x, y, z, r, w, s;
    fin = fopen(filename, "r"); assert(fin);
    
    //Read lines in file one by one
    for(i=1;i<=N;i++) {
        fgets(linestr, 100, fin);
        //Verbose control
        if(Verbose && (i%10000==0)){
            fprintf(stdout, "  Reading catalogue, line %d\n", i);
        }
        //Ignore lines starting with comment character
        if(linestr[0]!='#'){
            //If everything's OK, read data to appropriate arrays
            sscanf(linestr, "%lf  %lf  %lf %lf %lf %lf", &x, &y, &z, &r, &w, &s);
            data[i][1] = x - window[1];
            data[i][2] = y - window[3];
            data[i][3] = z - window[5];
            residuals[i] = r;
        }
    }
    for(i=1;i<=M;i++) {
        fgets(linestr, 100, fin);
        //Verbose control
        if(Verbose && (i%10000==0)){
            fprintf(stdout, "  Reading catalogue, line %d\n", i);
        }
        //Ignore lines starting with comment character
        if(linestr[0]!='#'){
            //If everything's OK, read data to appropriate arrays
            sscanf(linestr, "%lf  %lf  %lf %lf %lf %lf", &x, &y, &z, &r, &w, &s);
            dummy[i][1] = x - window[1];
            dummy[i][2] = y - window[3];
            dummy[i][3] = z - window[5];
            residuals[N+i] = r;
        }
    }
    
    if(Verbose){
        fprintf(stdout, "  Read %d points from catalogue %s\n", i-1, filename);
    }
    //Finally, return number of points read
    return i-1;
}

//FUNCTION: 'write_fields'
/*int write_fields(double **dummy, double *dens, int M, char *filename)
{
    int i;
    FILE *fout;
    char outfile[95];
    
    strcpy(outfile, filename);
    fout = fopen(outfile, "w"); assert(fout);
    
    for(i=1;i<=M;i++)
    {
        fprintf(fout, "%g  %g  %g  %g\n", dummy[i][1], dummy[i][2], dummy[i][3], dens[i]);
    }
    fclose(fout);
    if(Verbose){
        fprintf(stdout, "Results written to file %s\n", outfile);
    }
    return i;
}*/

//FUNCTION: 'gauss'
double gauss(double a, double r)
{
    double p;
    
    p = exp(-(pow(a,2))/(2*r*r))/(sqrt(2*PI)*r);
    
    return p;
}

//FUNCTION: 'fdensity'
int fdensity(double **dummy, double **data, double *weight, int M, int N, int *nodes, double w, double *window, char *name_file)
{
    bool verbose=false;
    int i,j,k,T,nr,nc,nd,n0,n1,n2,index,sign,posi,posj,posk;
    double kerpixarea,minx,maxx,miny,maxy,minz,maxz,stepx,stepy,stepz; //h,
    unsigned flags = FFTW_ESTIMATE;
    nr = nodes[1];    nc = nodes[2];    nd = nodes[3];
    n0 = nr*2;    n1 = nc*2;    n2 = nd*2;
    minx = window[1];    miny = window[3];    minz = window[5];
    maxx = window[2];    maxy = window[4];    maxz = window[6];
    stepx = (maxx-minx)/nr;
    stepy = (maxy-miny)/nc;
    stepz = (maxz-minz)/nd;
    kerpixarea = stepx*stepy*stepz;
    T = (n2-1) + n2*((n1-1) + n1*(n0-1)) + 1;
    
    // Kernel -> fK
    cout << "Kernel -> fK" << endl;
    // Generate Kernel
    cout << "Generate, ";
    double *grid;

    /*ofstream out;
    string name("grid.dat");
    out.open(name.c_str());*/

    grid = (double *) malloc((n0+1)*sizeof(double)); assert(grid);
    for(i=0;i<nr;i++)
    {
        grid[i] = gauss(stepx*i,w); //pad: h*(i-1) + (h/2);
        //out << i << " " << stepx*i << " " << grid[i] << endl;
    }
    for(i=-nr;i<0;i++)
    {
        grid[n0+i] = gauss(stepx*i,w);
        //out << n0+i << " " << stepx*i << " " << grid[n0+i] << endl;
    }
    //out.close();
    
    fftw_complex *kernel_in,*kernel_out;
    kernel_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * T);
    for(i=0;i<n0;i++)
    {
        for(j=0;j<n1;j++)
        {
            for(k=0;k<n2;k++)
            {
                index = k + n2*(j + n1*i);
                kernel_in[index][0] = grid[i]*grid[j]*grid[k]*kerpixarea;
                kernel_in[index][1] = 0.0;
            }
        }
    }

    // Transform
    cout << "Transform, ";
    sign = +1;
    fftw_plan fftw_plan1;
    kernel_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * T);
    fftw_plan1 = fftw_plan_dft_3d(n0,n1,n2,kernel_in,kernel_out,sign,flags);
    fftw_execute(fftw_plan1);
    fftw_destroy_plan(fftw_plan1);
    // Write
    if(verbose == true)
    {
        cout << "Write" << endl;
        char name_file1[200] = "Kernel_in.dat"; // No funciona
        write(kernel_in,n0,n1,n2,name_file1);
        char name_file2[200] = "Kernel_out.dat";
        write(kernel_out,n0,n1,n2,name_file2);
    }
    fftw_free(kernel_in);
        
    // Ypad -> fY
    cout << "Ypad -> fY" << endl;
    // Generate Ypad
    cout << "Generate, ";
    fftw_complex *ypad_in,*ypad_out;
    ypad_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * T);
    for(i=0;i<n0;i++)
    {
        for(j=0;j<n1;j++)
        {
            for(k=0;k<n2;k++)
            {
                index = k + n2*(j + n1*i);
                ypad_in[index][0] = 0.0;
                ypad_in[index][1] = 0.0;
            }
        }
    }
    // Find interval
    findinterval(data,dummy,weight,ypad_in,N,M,nr,nc,nd,stepx,stepy,stepz,window); //r,step,side);

    // Transform
    cout << "Transform, ";
    sign = -1;
    fftw_plan fftw_plan2;
    ypad_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * T);
    fftw_plan2 = fftw_plan_dft_3d(n0,n1,n2,ypad_in,ypad_out,sign,flags);
    fftw_execute(fftw_plan2);
    fftw_destroy_plan(fftw_plan2);
    // Write
    if(verbose == true)
    {
        cout << "Write" << endl;
        char name_file3[200] = "Ypad_in.dat";
        write(ypad_in,n0,n1,n2,name_file3);
        char name_file4[200] = "Ypad_out.dat";
        write(ypad_out,n0,n1,n2,name_file4);
    }
    fftw_free(ypad_in);

    // fY * fK -> sm // edge == FALSE & diggle == FALSE
    cout << "fY * fK -> sm (edge == FALSE & diggle == FALSE)" << endl;
    // Generate fY * fK
    cout << "Generate, ";
    fftw_complex *sm_in,*sm_out;
    fftw_plan fftw_plan3;
    
    sm_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * T);
    for(i=0;i<n0;i++)
    {
        for(j=0;j<n1;j++)
        {
            for(k=0;k<n2;k++)
            {
                index = k + n2*(j + n1*i);
                sm_in[index][0] = kernel_out[index][0]*ypad_out[index][0] - kernel_out[index][1]*ypad_out[index][1];
                sm_in[index][1] = kernel_out[index][1]*ypad_out[index][0] + kernel_out[index][0]*ypad_out[index][1];
            }
        }
    }
    fftw_free(ypad_out);
    // Transform
    cout << "Transform, ";
    sm_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * T);//(round(N/2.0)+1)
    sign = +1; // FFTW_BACKWARD
    fftw_plan3 = fftw_plan_dft_3d(n0,n1,n2,sm_in,sm_out,sign,flags);
    fftw_execute(fftw_plan3); // repeat as needed
    for(i=0;i<n0;i++){for(j=0;j<n1;j++){for(k=0;k<n2;k++){index=k+n2*(j+n1*i); sm_out[index][0]=sm_out[index][0]/T; sm_out[index][1]=sm_out[index][1]/T;}}}
    fftw_destroy_plan(fftw_plan3);
    // Write
    if(verbose == true)
    {
        cout << "Write" << endl;
        char name_file5[200] = "smooth_in.dat";
        write(sm_in,n0,n1,n2,name_file5);
        char name_file6[200] = "smooth_out.dat";
        write(sm_out,n0,n1,n2,name_file6);
    }
    fftw_free(sm_in);

    // EDGE: Mpad -> fM
    cout << "EDGE: Mpad -> fM" << endl;
    // Generate Mpad
    cout << "Generate, ";
    fftw_complex *Mpad_in,*Mpad_out;
    fftw_plan fftw_plan4;
    
    Mpad_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * T);
    for(i=0;i<n0;i++)
    {
        for(j=0;j<n1;j++)
        {
            for(k=0;k<n2;k++)
            {
                index = k + n2*(j + n1*i);
                Mpad_in[index][0] = 0.0;
                Mpad_in[index][1] = 0.0;
            }
        }
    }
    for(i=0;i<nr;i++)
    {
        for(j=0;j<nc;j++)
        {
            for(k=0;k<nd;k++)
            {
                index = k + n2*(j + n1*i);
                Mpad_in[index][0] = 1.0;
                Mpad_in[index][1] = 0.0;
            }
        }
    }
    // Transform
    cout << "Transform, ";
    Mpad_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * T);
    sign = +1; // FFTW_FORWARD
    fftw_plan4 = fftw_plan_dft_3d(n0,n1,n2,Mpad_in,Mpad_out,sign,flags);
    fftw_execute(fftw_plan4);
    fftw_destroy_plan(fftw_plan4);
    // Write
    if(verbose == true)
    {
        cout << "Write" << endl;
        char name_file7[200] = "Mpad_in.dat";
        write(Mpad_in,n0,n1,n2,name_file7);
        char name_file8[200] = "Mpad_out.dat";
        write(Mpad_out,n0,n1,n2,name_file8);
    }
    fftw_free(Mpad_in);

    // fM*fK -> edg
    cout << "fM*fK -> edg" << endl;
    // Generate
    cout << "Generate, ";
    fftw_complex *edg_in,*edg_out;
    fftw_plan fftw_plan5;
    
    edg_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * T);
    for(i=0;i<n0;i++)
    {
        for(j=0;j<n1;j++)
        {
            for(k=0;k<n2;k++)
            {
                index = k + n2*(j + n1*i);
                edg_in[index][0] = Mpad_out[index][0]*kernel_out[index][0] - Mpad_out[index][1]*kernel_out[index][1];
                edg_in[index][1] = Mpad_out[index][1]*kernel_out[index][0] + Mpad_out[index][0]*kernel_out[index][1];
            }
        }
    }
    fftw_free(kernel_out);
    fftw_free(Mpad_out);
    // Transform
    cout << "Transform, ";
    edg_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * T);
    sign = -1; // FFTW_BACKWARD
    fftw_plan5 = fftw_plan_dft_3d(n0,n1,n2,edg_in,edg_out,sign,flags);
    fftw_execute(fftw_plan5);
    for(i=0;i<n0;i++){for(j=0;j<n1;j++){for(k=0;k<n2;k++){index=k+n2*(j+n1*i); edg_out[index][0]=edg_out[index][0]/T; edg_out[index][1]=edg_out[index][1]/T;}}}
    fftw_destroy_plan(fftw_plan5);
    // Write
    if(verbose == true)
    {
        cout << "Write" << endl;
        char name_file9[200] = "Edge_in.dat";
        write(edg_in,n0,n1,n2,name_file9);
        char name_file10[200] = "Edge_out.dat";
        write(edg_out,n0,n1,n2,name_file10);
    }
    fftw_free(edg_in);
    
    // Last calculations / edge == TRUE & diggle == FALSE
    cout << "Last calculations (edge == TRUE & diggle == FALSE)" << endl;
    ofstream file;
    string nom(name_file);
    file.open(nom.c_str());
    for(i=0;i<nr;i++)
    {
        for(j=0;j<nc;j++)
        {
            for(k=0;k<nd;k++)
            {
                index = k + n2*(j + n1*i);
                file << stepx*i + (stepx/2) + minx << " "; //+ (stepx/2)
                file << stepy*j + (stepy/2) + miny << " "; //+ (stepy/2)
                file << stepz*k + (stepz/2) + minz << " "; //+ (stepz/2)
                file << sm_out[index][0]/(edg_out[index][0]*kerpixarea) << endl; //" " << sm_out[index][0]/kerpixarea << endl;
            }
        }
    }
    file.close();
    fftw_free(sm_out);
    fftw_free(edg_out);
    
    return N;
    
    // Very last calculations / edge == TRUE & diggle == TRUE
    /*cout << " IGNORE # IGNORE # IGNORE # IGNORE " << endl;
    cout << "Very last calculations / (edge == FALSE & diggle == TRUE)" << endl;
    // Generate
    cout << "Generate, ";
    fftw_complex *diggle_in,*diggle_out;
    diggle_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * T); // No està bé
    for(i=0;i<n0;i++)
    {
        for(j=0;j<n1;j++)
        {
            for(k=0;k<n2;k++)
            {
                index = k + n2*(j + n1*i);
                diggle_in[index][0] = 0.0;
                diggle_in[index][1] = 0.0;
            }
        }
    }
    // Find interval
    findinterval(data,dummy,weight,diggle_in,N,M,nr,r,h);

    // Transform
    cout << "Transform, ";
    sign = -1;
    fftw_plan fftw_plan6;
    diggle_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * T);
    fftw_plan6 = fftw_plan_dft_3d(n0,n1,n2,diggle_in,diggle_out,sign,flags);
    fftw_execute(fftw_plan6);
    fftw_destroy_plan(fftw_plan6);
    // Write
    cout << "Write" << endl;
    char name_file11[200] = "Diggle_in.dat";
    write(diggle_in,n0,n1,n2,name_file11);
    char name_file12[200] = "Diggle_out.dat";
    write(diggle_out,n0,n1,n2,name_file12);
    fftw_free(diggle_in);
    fftw_free(edg_out);
    
    // fY * fK -> sm // edge == FALSE & diggle == TRUE
    cout << "fY * fK -> sm (edge == FALSE & diggle == TRUE)" << endl;
    // Generate fY * fK
    cout << "Generate, ";
    fftw_complex *sm2_in,*sm2_out;
    fftw_plan fftw_plan7;
    
    sm2_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * T);
    for(i=0;i<n0;i++)
    {
        for(j=0;j<n1;j++)
        {
            for(k=0;k<n2;k++)
            {
                index = k + n2*(j + n1*i);
                sm2_in[index][0] = 0.0;
                sm2_in[index][1] = kernel_out[index][1]*diggle_out[index][1];
            }
        }
    }
    fftw_free(kernel_out);
    fftw_free(diggle_out);
    // Transform
    cout << "Transform, ";
    sm2_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * T);
    sign = -1; // FFTW_BACKWARD
    fftw_plan7 = fftw_plan_dft_3d(n0,n1,n2,sm2_in,sm2_out,sign,flags);
    fftw_execute(fftw_plan7); // repeat as needed
    for(i=0;i<n0;i++){for(j=0;j<n1;j++){for(k=0;k<n2;k++){index=k+n2*(j+n1*i); sm2_out[index][1]=sm2_out[index][1]/T;}}}
    fftw_destroy_plan(fftw_plan7);
    // Write
    cout << "Write" << endl;
    char name_file13[200] = "smooth_diggle_in.dat";
    write(sm2_in,n0,n1,n2,name_file13);
    char name_file14[200] = "smooth_diggle_out.dat";
    write(sm2_out,n0,n1,n2,name_file14);
    fftw_free(sm2_in);
    fftw_free(sm2_out);*/
}

//FUNCTION: 'write'
void write(fftw_complex *fftw_data, int n0, int n1, int n2, char *name_file)
{
    int i,j,k,index;
    ofstream file;
    string nom(name_file);
    file.open(nom.c_str());
    
    for(i=0;i<n0;i++)
    {
        for(j=0;j<n1;j++)
        {
            for(k=0;k<n2;k++)
            {
                index = k+n2*(j+n1*i);
                file << i+1 << " " << j+1 << " " << k+1 << " " << fftw_data[index][0] << " " << fftw_data[index][1] << endl;
            }
        }
    }
    file.close();
}

//FUNCTION: 'findinterval'
void findinterval(double **data, double **dummy, double *weight, fftw_complex *ypad_in, int N, int M, int nr, int nc, int nd, double stepx, double stepy, double stepz, double *window)
{
    int l,posi,posj,posk,index,n0=2*nr,n1=2*nc,n2=2*nd;
    double x,y,z;
    
    for(l=1;l<=N;l++)
    {
        x = data[l][1];
        y = data[l][2];
        z = data[l][3];
        if(inside(x,y,z,window)==true)
        {
            posi=round((x-(stepx/2))/stepx); //position(x,h,nr,r); //ample de l'interval = h
            posj=round((y-(stepy/2))/stepy); //trunc(y/h); //position(y,h,nr,r);
            posk=round((z-(stepz/2))/stepz); //trunc(z/h); //position(z,h,nr,r);
            index = posk + n2*(posj + n1*(posi));
            //cout << x << " " << y << " " << z << " " << posi << " " << posj << " " << posk << " " << index << " " << weight[l] << endl;
            //index = posk + n2*(posj + n1*(n1-posi-1)); // i dimension is inverted
            ypad_in[index][0] = ypad_in[index][0] + weight[l]; // A[posj,posi] <- weights[i] // Image in R
        }
    }
    for(l=1;l<=M;l++)
    {
        x = dummy[l][1];
        y = dummy[l][2];
        z = dummy[l][3];
        if(inside(x,y,z,window)==true)
        {
            posi=round((x-(stepx/2))/stepx); //position(x,h,nr,r); //ample de l'interval = h
            posj=round((y-(stepy/2))/stepy); //trunc(y/h); //position(y,h,nr,r);
            posk=round((z-(stepz/2))/stepz); //trunc(z/h); //position(z,h,nr,r);
            index = posk + n2*(posj + n1*(posi));
            //cout << x << " " << y << " " << z << " " << posi << " " << posj << " " << posk << " " << index << endl;
            //index = posk + n2*(posj + n1*(n1-posi-1)); // i dimension is inverted
            ypad_in[index][0] = ypad_in[index][0] + weight[N+l]; // A[posj,posi] <- weights[i] // Image in R
        }
    }
    
    /*ofstream out;
    string name("interval.dat");
    out.open(name.c_str());
    for(posi=1;posi<=n0;posi++)
    {
        for(posj=1;posj<=n1;posj++)
        {
            for(posk=1;posk<=n2;posk++)
            {
                index = posk + n2*(posj + n1*(posi));
                out << posi << " " << posj << " " << posk << " " << index << " " << ypad_in[index][0] << endl;
            }
        }
    }
    out.close();*/
}

//FUNCTION: 'inside'
bool inside(double x, double y, double z, double *window)
{
    bool in=false;
    if(x > window[1] && x < window[2])
    {
        if(y > window[3] && y < window[4])
        {
            if(z > window[5] && z < window[6])
            {
                in=true;
            }
        }
    }
    return in;
}

