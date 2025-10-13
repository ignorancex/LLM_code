#include "k3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpfr.h> 

static double centerx = CENTERX, centery = CENTERY, radius = RADIUS;
static double length=LENGTH, sep = SEP;
static int drawtop = 1, drawbot = 1, verbose=VERBOSE, spin=0, unravel = 0;


double px(double s, double t);//partial_x p
double py(double s, double t);//partal_y p
//double pxx(double s, double t);//partial_xx p
//double pxy(double s, double t);//partial_xy p
//double pyy(double s, double t);//partial_yy p
int chart(point p); //outputs a chart, labelled 1-6, which p belongs to (p might belong to two charts)
double D(double x, double y);
double Dx(double x, double y);//partial_x D
double Dy(double x, double y);//partal_y D
double Dxx(double x, double y);//partial_xx D
double Dxy(double x, double y);//partial_xy D
double Dyy(double x, double y);//partial_xx D

double D1(point p);// DIx
double D2(point p);// DIy
double D3(point p);// DIz

int main(int argc, char *argv[]) {
	
    //starting point
	point p;
    p.x = -1.726895448754;
    p.y = 1.0416430939367222401;
    p.z = 1.726895448754;
    
    //iter = number of iterations of f
    int iter = 10; 


    //initialize matrices
    double** matrix = (double**)malloc(3 * sizeof(double*));
    double** coordmatrix = (double**)malloc(3 * sizeof(double*));
    double** matrix_temp = (double**)malloc(3 * sizeof(double*));
    double** temp1 = (double**)malloc(3 * sizeof(double*));
    double** temp2 = (double**)malloc(3 * sizeof(double*));
    double** temp3 = (double**)malloc(3 * sizeof(double*));
    double** temp4 = (double**)malloc(3 * sizeof(double*));
    double** temp5 = (double**)malloc(3 * sizeof(double*));
    double** x_mat = (double**)malloc(3 * sizeof(double*));
    double** y_mat = (double**)malloc(3 * sizeof(double*));
    double** z_mat = (double**)malloc(3 * sizeof(double*));

    for (int i = 0; i < 3; ++i) {
        matrix[i] = (double*)malloc(3 * sizeof(double));
        coordmatrix[i] = (double*)malloc(3 * sizeof(double));
        matrix_temp[i] = (double*)malloc(3 * sizeof(double));
        temp1[i] = (double*)malloc(3 * sizeof(double));
        temp2[i] = (double*)malloc(3 * sizeof(double));
        temp3[i] = (double*)malloc(3 * sizeof(double));
        temp4[i] = (double*)malloc(3 * sizeof(double));
        temp5[i] = (double*)malloc(3 * sizeof(double));
        x_mat[i] = (double*)malloc(3 * sizeof(double));
        y_mat[i] = (double*)malloc(3 * sizeof(double));
        z_mat[i] = (double*)malloc(3 * sizeof(double));
    }
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            if(i==j){
                matrix_temp[i][j] = 1;
                matrix[i][j] = 1;
                coordmatrix[i][j] = 1;
                temp1[i][j] = 1;
                temp2[i][j] = 1;
                temp3[i][j] = 1;
                temp4[i][j] = 1;
                temp5[i][j] = 1;
                x_mat[i][j] = 1;
                y_mat[i][j] = 1;
                z_mat[i][j] = 1;

            } else{
                matrix_temp[i][j] = 0;
                coordmatrix[i][j] = 0;
                matrix[i][j] = 0;
                temp1[i][j] = 0;
                temp2[i][j] = 0;
                temp3[i][j] = 0;
                temp4[i][j] = 0;
                temp5[i][j] = 0;
                x_mat[i][j] = 0;
                y_mat[i][j] = 0;
                z_mat[i][j] = 0;        
            }
        }
    }




    for(int i = 0; i <iter; i++){

        double a, b;
        if(chart(p) == 1 || chart(p) == 4){
           a = p.x;
           b = p.y;
        }
        if(chart(p) == 2 || chart(p) == 5){
            a = p.x;
            b = p.z;
         }
         if(chart(p) == 3 || chart(p) == 6){
            a = p.y;
            b = p.z;
         }

        fprintf(stderr, "(x,y) is (%f, %f) \n", a,b);
        fprintf(stderr, "D(x,y) is %f \n", D(a,b));
        fprintf(stderr, "Dx(x,y) is %f \n", Dx(a,b));
        fprintf(stderr, "Dy(x,y) is %f \n", Dy(a,b));
        fprintf(stderr, "Dxx(x,y) is %f \n", Dxx(a,b));
        fprintf(stderr, "Dyy(x,y) is %f \n", Dyy(a,b));
        fprintf(stderr, "Dxy(x,y) is %f \n", Dxy(a,b));


        int k = 0;
        int start = chart(p);
        int end_temp;
        if(i == 0){
            end_temp = start;
        }
        for(int i = 0; i < 3; i++){
            if(i == 3-start){
                temp4[i][0] = px(a, b);
                temp4[i][1] = py(a, b);
            } else if(i == 6-start){
                temp4[i][0] = px(-a,b); //because p_+(-x,y) = -p_-(x,y)
                temp4[i][1] = -py(-a,b); //because p_+(-x,y) = -p_-(x,y)
            } else{
                if(k == 0){
                    temp4[i][0] = 1;
                    temp4[i][1] = 0;
                }
                if(k == 1){
                    temp4[i][0] = 0;
                    temp4[i][1] = 1;
                }
                if(k == 2){
                    fprintf(stderr, "error");
                    abort();
                }
                k++;
            }
            
        }
        k = 0;

        int end = chart(f(p));
        if(i==iter - 1){
            end = end_temp; //forces the last chart to equal the first chart (comment out if p not an iter-pseudo-orbit)
        }
        for(int j = 0; j < 3; j++){
            if(j == 3-end){
                temp5[0][j] = 0;
                temp5[1][j] = 0;
            } else if(j == 6-end){
                temp5[0][j] = 0;
                temp5[1][j] = 0;
            } else{
                if(k == 0){
                    temp5[0][j] = 1;
                    temp5[1][j] = 0;
                }
                if(k == 1){
                    temp5[0][j] = 0;
                    temp5[1][j] = 1;
                }
                if(k == 2){
                    fprintf(stderr, "error");
                    abort();
                }
                k++;
            }
        }
        
        fprintf(stderr, "chart is %d \n", chart(p));
        fprintf(stderr, "nect chart is %d \n", chart(f(p)));


        fprintf(stderr, "(D psi_%d)_{x_%d^c} =  \n", i,i);
        printMatrix(temp4, 3, 2);
        fprintf(stderr, "(D phi_{k_%d})_{x_{%d}} =  \n", i+1, i+1);
        printMatrix(temp5, 2, 3);

        Dix(p, x_mat); 
        p = ix(p);
        Diy(p, y_mat); 
        p = iy(p);
        Diz(p, z_mat);
        p = iz(p); 

        multiplyMatrices(y_mat, x_mat, temp2, 3, 3, 3, 3);
        multiplyMatrices(z_mat, temp2, temp3, 3, 3, 3, 3);
        fprintf(stderr, "(Df}_{x_%d} =  \n", i);
        printMatrix(temp3, 3, 3);

        multiplyMatricesAux(temp3, matrix, temp1, 3, 3, 3, 3);

        multiplyMatricesAux(temp5, temp3, temp1, 2, 3, 3, 3);
        multiplyMatricesAux(temp3, temp4, temp1, 2, 3, 3, 2);
        fprintf(stderr, "(Df^c}_{x_%d^c} =  \n", i);
        printMatrix(temp4, 2, 2);


        //multiplyMatricesAux(temp4, coordmatrix, temp1, 2, 2, 2, 2); //coordmatrix is a copy of temp4

    }

    //fprintf(stderr, "Df^{%d}_{x_0} =  \n", i+1);
    //printMatrix(matrix, 3, 3);
    //fprintf(stderr, "(Df^c)^{%d}_{x_0} =  \n", i+1);
    //printMatrix(coordmatrix, 2, 2);
    //pprint(f(p));
	exit(0);
}



double D(double x, double y){
    double new = 100*x*x*y*y + 8*(1+x*x)*(1+y*y) - 4*(1+x*x)*(1+x*x)*(1+y*y)*(1+y*y);
	return new;
}

double Dx(double x, double y){
    double new = 200*x*y*y + 16*x*(1 + y*y) - 16*x*(1 + x*x)*(1 + y*y)*(1+y*y);
    return new;
}

double Dy(double x, double y){
    double new = Dx(y,x);
    return new;
}
double Dxx(double x, double y){
    double new = 200*y*y + 16*(1 + y*y) - 32*x*x* (1 + y*y)*(1 + y*y) - 16*(1 + x*x)*(1 + y*y)*(1 + y*y);
    return new;
}
double Dyy(double x, double y){
    double new = Dxx(y,x);
    return new;
}
double Dxy(double x, double y){
    double new = 432*x*y - 64*x*(1 + x*x)*y*(1 + y*y);
    return new;
}



double D1(point p){
    double vx = 2*p.x*(1+p.y*p.y)*(1+p.z*p.z)+A*p.y*p.z;
	return vx*vx;
}
double D2(point p){
    double vy = 2*p.y*(1+p.x*p.x)*(1+p.z*p.z)+A*p.x*p.z;
	return vy*vy;
}
double D3(point p){
    double vz = 2*p.z*(1+p.y*p.y)*(1+p.x*p.x)+A*p.x*p.y;
	return vz*vz;
}

double px(double s, double t) { //specific for A=10, B = 2. 
    double numerator = s * (-2 + 23 * t * t) - s * s * s * (2 + 27 * t * t) - 
                       5 * t * sqrt(1 - t * t * t * t - s * s * s * s * (1 + t * t) * (1 + t * t) + s * s * (23 * t * t - 2 * t * t * t * t)) + 
                       5 * s * s * t * sqrt(1 - t * t * t * t - s * s * s * s * (1 + t * t) * (1 + t * t) + s * s * (23 * t * t - 2 * t * t * t * t));
    
    double denominator = (1 + s * s) * (1 + s * s) * (1 + t * t) * 
                         sqrt(1 - t * t * t * t - s * s * s * s * (1 + t * t) * (1 + t * t) + s * s * (23 * t * t - 2 * t * t * t * t));
    
    return numerator / denominator;
}

double py(double s, double t) { 
    return px(t,s);
}

int chart(point p){
    double d1 =  D1(p);
    double d2 = D2(p);
    double d3 = D3(p);

    if (d1 >= d2 && d1 >= d3) {
        if(topx(p)){
            return 3;
        } else{
            return 6;
        }
    } else if (d2 >= d1 && d2 >= d3) {
        if(topy(p)){
            return 2;
        } else{
            return 5;
        }
    } else {
        if(top(p)){
            return 1;
        } else{
            return 4;
        }
    }
    return 0;
}

/* TO KEEP:
double pxx(double s, double t) {
    double term1 = pow((200 * s * t * t + 16 * s * (1 + t * t) - 16 * s * (1 + s * s) * pow(1 + t * t, 2)), 2) / 
                   (4 * pow((100 * s * s * t * t + 8 * (1 + s * s) * (1 + t * t) - 4 * pow((1 + s * s), 2) * pow((1 + t * t), 2)), 1.5));
    
    double term2 = (200 * t * t + 16 * (1 + t * t) - 32 * s * s * pow((1 + t * t), 2) - 16 * (1 + s * s) * pow((1 + t * t), 2)) / 
                   (2 * sqrt(100 * s * s * t * t + 8 * (1 + s * s) * (1 + t * t) - 4 * pow((1 + s * s), 2) * pow((1 + t * t), 2)));
    
    double term3 = 2 * s * (-10 * t + (200 * s * t * t + 16 * s * (1 + t * t) - 16 * s * (1 + s * s) * pow((1 + t * t), 2)) / 
                            (2 * sqrt(100 * s * s * t * t + 8 * (1 + s * s) * (1 + t * t) - 4 * pow((1 + s * s), 2) * pow((1 + t * t), 2))));
    
    double term4 = 4 * s * s * (-10 * s * t + sqrt(100 * s * s * t * t + 8 * (1 + s * s) * (1 + t * t) - 4 * pow((1 + s * s), 2) * pow((1 + t * t), 2)));
    
    double term5 = -(-10 * s * t + sqrt(100 * s * s * t * t + 8 * (1 + s * s) * (1 + t * t) - 4 * pow((1 + s * s), 2) * pow((1 + t * t), 2)));
    
    double denominator = 2 * (1 + s * s) * (1 + t * t);
    
    return (-(term1) + term2) / denominator - term3 / pow((1 + s * s), 2) / (1 + t * t) + term4 / pow((1 + s * s), 3) / (1 + t * t) - term5 / pow((1 + s * s), 2) / (1 + t * t);
}

double pyy(double s, double t) {
    return pxx(t,s);
}

double pxy(double s, double t) {
    double term1 = (200 * s * s * t + 16 * (1 + s * s) * t - 16 * pow((1 + s * s), 2) * t * (1 + t * t)) * 
                   (200 * s * t * t + 16 * s * (1 + t * t) - 16 * s * pow((1 + s * s), 2) * pow((1 + t * t), 2));
    

    double term2 = 4 * pow((100 * s * s * t * t + 8 * (1 + s * s) * (1 + t * t) - 4 * pow((1 + s * s), 2) * pow((1 + t * t), 2)), 1.5);
    double term3 = 432 * s * t - 64 * s * (1 + s * s) * t * (1 + t * t);
    double term4 = 2 * sqrt(100 * s * s * t * t + 8 * (1 + s * s) * (1 + t * t) - 4 * pow((1 + s * s), 2) * pow((1 + t * t), 2));
    double term5 = s * (-10 * s + (200 * s * s * t + 16 * (1 + s * s) * t - 16 * pow((1 + s * s), 2) * t * (1 + t * t)) / term4);
    double term6 = t * (-10 * t + (200 * s * t * t + 16 * s * (1 + t * t) - 16 * s * pow((1 + s * s), 2) * pow((1 + t * t), 2)) / term4);
    double term7 = 2 * s * t * (-10 * s * t + sqrt(100 * s * s * t * t + 8 * (1 + s * s) * (1 + t * t) - 4 * pow((1 + s * s), 2) * pow((1 + t * t), 2)));
    
    double denominator = 2 * (1 + s * s) * (1 + t * t);
    
    return (-10 - term1 / term2 + term3 / term4) / denominator - term5 / pow((1 + s * s), 2) / (1 + t * t) - term6 / (1 + s * s) / pow((1 + t * t), 2) + term7 / pow((1 + s * s), 2) / pow((1 + t * t), 2);
}

int chart(point p){

    if(top(p)){
        if(fabs(px(p.x, p.y)) < alpha && fabs(py(p.x, p.y)) < alpha && fabs(pxx(p.x, p.y)) < beta && fabs(pxy(p.x, p.y)) < beta && fabs(pyy(p.x, p.y)) < beta) {
            return 1;
        }
    } else{
        if(fabs(px(-p.x, p.y)) < alpha && fabs(py(-p.x, p.y)) < alpha && fabs(pxx(-p.x, p.y)) < beta && fabs(-pxy(p.x, p.y)) < beta && fabs(-pyy(p.x, p.y)) < beta) {
            return 4;
        }
    }
    if(topx(p)){
        if(fabs(px(p.y, p.z)) < alpha && fabs(py(p.y, p.z)) < alpha && fabs(pxx(p.y, p.z)) < beta && fabs(pxy(p.y, p.z)) < beta && fabs(pyy(p.y, p.z)) < beta) {
            return 3;
        }
    } else{
        if(fabs(px(-p.y, p.z)) < alpha && fabs(py(-p.y, p.z)) < alpha && fabs(pxx(-p.y, p.z)) < beta && fabs(-pxy(p.y, p.z)) < beta && fabs(-pyy(p.y, p.z)) < beta) {
            return 6;
        }
    }

    if(topy(p)){
        if(fabs(px(p.x, p.z)) < alpha && fabs(py(p.x, p.z)) < alpha && fabs(pxx(p.x, p.z)) < beta && fabs(pxy(p.x, p.z)) < beta && fabs(pyy(p.x, p.z)) < beta) {
            return 2;
        }
    } else{
        if(fabs(px(-p.x, p.z)) < alpha && fabs(py(-p.x, p.z)) < alpha && fabs(pxx(-p.x, p.z)) < beta && fabs(-pxy(p.x, p.z)) < beta && fabs(-pyy(p.x, p.z)) < beta) {
            return 5;
        }
    }
    return 0;
}
*/