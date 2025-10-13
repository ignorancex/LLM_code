
/* A particular pseudo-orbit */
#include <math.h>
#include "k3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdarg.h>
#include <mpfr.h>  //for precision calculations

//new point structure
typedef struct {
    mpfr_t x, y, z;
} mpfr_point;

void init_point(mpfr_point *p) {
    mpfr_init(p->x);
    mpfr_init(p->y);
    mpfr_init(p->z);
}

//Charts \tilde \phi_i 
//3 = \Phi_1^+
//2 = \Phi_2^+
//1 = \Phi_3^+
//4 = \Phi_1^-
//5 = \Phi_2^-
//6 = \Phi_3^-

const int charts[10] = {6, 4, 5, 6, 5, 5, 5, 1, 5, 3};


//function declarations

/*reused from periodicExact*/
void alphaExact(mpfr_t result, mpfr_t x, mpfr_t y); // [alpha]
void betaExact(mpfr_t result, mpfr_t x, mpfr_t y); // [beta]
void DExact(mpfr_t result, mpfr_t x, mpfr_t y); // [D]
void ix_exact(mpfr_point *p); // [I_x]
void iy_exact(mpfr_point *p); // [I_y]
void iz_exact(mpfr_point *p); // [I_z]
void f_exact(mpfr_point *p); // [f]
void pPlusExact(mpfr_t result,  mpfr_t x,  mpfr_t y); // [p_+]
void PMinusExact(mpfr_t result,  mpfr_t x,  mpfr_t y); // [p_-]
void lift_exact_charts(mpfr_point *p,  mpfr_t x,  mpfr_t y, int i); // [Phi_k^{pm}] 
void f_c(mpfr_t a,  mpfr_t b,  int i, int j); // [f_c] where i = incoming chart, j = outgoing chart
void dist_2plane(mpfr_t result, mpfr_t a1, mpfr_t b1, mpfr_t a2, mpfr_t b2); // distance between (a1,b1) and (a2,b2)

/*new functions*/
void DixExact(mpfr_point p, mpfr_t **matrix);// [DIx]
void DiyExact(mpfr_point p, mpfr_t **matrix);// [DIy]
void DizExact(mpfr_point p, mpfr_t **matrix);// [DIz]
void DxExact(mpfr_t result, mpfr_t x, mpfr_t y);// [partial_x D]
void DyExact(mpfr_t result, mpfr_t x, mpfr_t y);// [partial_y D]
void DxxExact(mpfr_t result, mpfr_t x, mpfr_t y);// [partial_xx D]
void DyyExact(mpfr_t result, mpfr_t x, mpfr_t y);// [partial_yy D]
void DxyExact(mpfr_t result, mpfr_t x, mpfr_t y);// [partial_xy D]
void pxExact(mpfr_t result, mpfr_t s, mpfr_t t);// [partial_x p]
void pyExact(mpfr_t result, mpfr_t s, mpfr_t t);// [partial_y p]
void alpha_xExact(mpfr_t result, const mpfr_t x, const mpfr_t y);// [partial_x alpha]
void alpha_yExact(mpfr_t result, const mpfr_t x, const mpfr_t y);// [partial_y alpha]
void multiplyMatricesExact(mpfr_t **firstMatrix, mpfr_t **secondMatrix, mpfr_t **result, int ROWS1, int COLS1, int ROWS2, int COLS2); //result = (firstMatrix)(secondMatrix)
void multiplyMatricesAuxExact(mpfr_t **firstMatrix, mpfr_t **secondMatrix, mpfr_t **result, int ROWS1, int COLS1, int ROWS2, int COLS2); //secondMatrix = (firstMatrix)(secondMatrix)
void printMatrixExact(mpfr_t **matrix, int ROWS1, int COLS1); //prints matrices


int main(int argc, char *argv[]) {
    //Setting default rounding mode
    mpfr_set_default_rounding_mode(MPFR_RNDZ);

    // Set precision to 5000 bits and exponent range to [-1073,1024]
    mpfr_set_default_prec(5000); 
    mpfr_set_emin (-1073); mpfr_set_emax (1024); 

    //Initializing Matrices
    mpfr_t** temp1 = (mpfr_t**)malloc(3 * sizeof(mpfr_t*));
    mpfr_t** temp2 = (mpfr_t**)malloc(3 * sizeof(mpfr_t*));
    mpfr_t** temp3 = (mpfr_t**)malloc(3 * sizeof(mpfr_t*));
    mpfr_t** temp4 = (mpfr_t**)malloc(3 * sizeof(mpfr_t*));
    mpfr_t** temp5 = (mpfr_t**)malloc(3 * sizeof(mpfr_t*));
    mpfr_t** x_mat = (mpfr_t**)malloc(3 * sizeof(mpfr_t*));
    mpfr_t** y_mat = (mpfr_t**)malloc(3 * sizeof(mpfr_t*));
    mpfr_t** z_mat = (mpfr_t**)malloc(3 * sizeof(mpfr_t*));

    for (int i = 0; i < 3; ++i) {
        temp1[i] = (mpfr_t*)malloc(3 * sizeof(mpfr_t));
        temp2[i] = (mpfr_t*)malloc(3 * sizeof(mpfr_t));
        temp3[i] = (mpfr_t*)malloc(3 * sizeof(mpfr_t));
        temp4[i] = (mpfr_t*)malloc(3 * sizeof(mpfr_t));
        temp5[i] = (mpfr_t*)malloc(3 * sizeof(mpfr_t));
        x_mat[i] = (mpfr_t*)malloc(3 * sizeof(mpfr_t));
        y_mat[i] = (mpfr_t*)malloc(3 * sizeof(mpfr_t));
        z_mat[i] = (mpfr_t*)malloc(3 * sizeof(mpfr_t));

        for (int j = 0; j < 3; ++j) {
            mpfr_init(temp1[i][j]);
            mpfr_init(temp2[i][j]);
            mpfr_init(temp3[i][j]);
            mpfr_init(temp4[i][j]);
            mpfr_init(temp5[i][j]);
            mpfr_init(x_mat[i][j]);
            mpfr_init(y_mat[i][j]);
            mpfr_init(z_mat[i][j]);

            if (i == j) {
                mpfr_set_ui(temp1[i][j], 1, MPFR_RNDZ);
                mpfr_set_ui(temp2[i][j], 1, MPFR_RNDZ);
                mpfr_set_ui(temp3[i][j], 1, MPFR_RNDZ);
                mpfr_set_ui(temp4[i][j], 1, MPFR_RNDZ);
                mpfr_set_ui(temp5[i][j], 1, MPFR_RNDZ);
                mpfr_set_ui(x_mat[i][j], 1, MPFR_RNDZ);
                mpfr_set_ui(y_mat[i][j], 1, MPFR_RNDZ);
                mpfr_set_ui(z_mat[i][j], 1, MPFR_RNDZ);
            } else {
                mpfr_set_ui(temp1[i][j], 0, MPFR_RNDZ);
                mpfr_set_ui(temp2[i][j], 0, MPFR_RNDZ);
                mpfr_set_ui(temp3[i][j], 0, MPFR_RNDZ);
                mpfr_set_ui(temp4[i][j], 0, MPFR_RNDZ);
                mpfr_set_ui(temp5[i][j], 0, MPFR_RNDZ);
                mpfr_set_ui(x_mat[i][j], 0, MPFR_RNDZ);
                mpfr_set_ui(y_mat[i][j], 0, MPFR_RNDZ);
                mpfr_set_ui(z_mat[i][j], 0, MPFR_RNDZ);
            }
        }
    }


    //Initialize pseudo-orbit
    mpfr_t a[10];
    mpfr_t b[10];

    for (int i = 0; i < 10; ++i) {
        mpfr_init(a[i]);
        mpfr_init(b[i]);
    }

    mpfr_set_str(a[0], "1.041643093944314148360673792017", 10, MPFR_RNDZ);
    mpfr_set_str(a[1], "-0.439586738044637984442175311821", 10, MPFR_RNDZ);
    mpfr_set_str(a[2], "1.111402054756051352317454938205", 10, MPFR_RNDZ);
    mpfr_set_str(a[3], "-0.328869789067645570794396144391", 10, MPFR_RNDZ);
    mpfr_set_str(a[4], "1.488818954806569814700993326668", 10, MPFR_RNDZ);
    mpfr_set_str(a[5], "0.533829900932504729554816817729", 10, MPFR_RNDZ);
    mpfr_set_str(a[6], "-1.004464450964796276608907444033", 10, MPFR_RNDZ);
    mpfr_set_str(a[7], "-0.867950394543647373540816310310", 10, MPFR_RNDZ);
    mpfr_set_str(a[8], "0.926435350008842162121508383319", 10, MPFR_RNDZ);
    mpfr_set_str(a[9], "0.555943953085459715621476373770", 10, MPFR_RNDZ);

    

    mpfr_set_str(b[0], "1.726895448754858426328854724474", 10, MPFR_RNDZ);
    mpfr_set_str(b[1], "0.555943953085459715621476373770", 10, MPFR_RNDZ);
    mpfr_set_str(b[2], "-0.926435350008842162121508383319", 10, MPFR_RNDZ);
    mpfr_set_str(b[3], "0.867950394543647373540816310310", 10, MPFR_RNDZ);
    mpfr_set_str(b[4], "1.004464450964796276608907444033", 10, MPFR_RNDZ);
    mpfr_set_str(b[5], "-0.533829900932504729554816817729", 10, MPFR_RNDZ);
    mpfr_set_str(b[6], "-1.488818954806569814700993326668", 10, MPFR_RNDZ);
    mpfr_set_str(b[7], "-0.328869789067645570794396144391", 10, MPFR_RNDZ);
    mpfr_set_str(b[8], "-1.111402054756051352317454938205", 10, MPFR_RNDZ);
    mpfr_set_str(b[9], "0.439586738044637984442175311821", 10, MPFR_RNDZ);



    //Initialize variables
    mpfr_t out;
    mpfr_init(out);
    char a_str[64], b_str[64];

    for (int i = 0; i < 10; i++) {
        mpfr_sprintf(a_str, "%.15Rf", a[i]);
        mpfr_sprintf(b_str, "%.15Rf", b[i]);
        fprintf(stderr, "(x,y) is (%s, %s) \n", a_str, b_str);

        //Print current chart
        fprintf(stderr, "chart is %d \n", charts[i]);
    
        //Print values for D and its derivatives
        DExact(out, a[i], b[i]);
        mpfr_sprintf(a_str, "%.15Rf", out);
        fprintf(stderr, "DExact(x,y) is %s \n", a_str);
    
        DxExact(out, a[i], b[i]);
        mpfr_sprintf(a_str, "%.15Rf", out);
        fprintf(stderr, "DxExact(x,y) is %s \n", a_str);
    
        DyExact(out, a[i], b[i]);
        mpfr_sprintf(a_str, "%.15Rf", out);
        fprintf(stderr, "DyExact(x,y) is %s \n", a_str);
    
        DxxExact(out, a[i], b[i]);
        mpfr_sprintf(a_str, "%.15Rf", out);
        fprintf(stderr, "DxxExact(x,y) is %s \n", a_str);
    
        DyyExact(out, a[i], b[i]);
        mpfr_sprintf(a_str, "%.15Rf", out);
        fprintf(stderr, "DyyExact(x,y) is %s \n", a_str);
    
        DxyExact(out, a[i], b[i]);
        mpfr_sprintf(a_str, "%.15Rf", out);
        fprintf(stderr, "DxyExact(x,y) is %s \n", a_str);
        
        //Calculating D(phi_i)
        int k = 0;
        for (int j = 0; j < 3; j++) {
            if (j == 3 - charts[i]) {
                pxExact(temp4[j][0], a[i], b[i]);
                pyExact(temp4[j][1], a[i], b[i]);
            } else if (j == 6 - charts[i]) {
                //because p_+(-x,y) = -p_-(x,y):
                mpfr_neg(temp4[j][0], a[i], MPFR_RNDZ);
                pxExact(temp4[j][0], temp4[j][0], b[i]);

                //because p_+(-x,y) = -p_-(x,y):
                mpfr_neg(temp4[j][1], a[i], MPFR_RNDZ);
                pyExact(temp4[j][1], temp4[j][1], b[i]);
                mpfr_neg(temp4[j][1], temp4[j][1], MPFR_RNDZ);

            } else {
                if (k == 0) {
                    mpfr_set_ui(temp4[j][0], 1, MPFR_RNDZ);
                    mpfr_set_ui(temp4[j][1], 0, MPFR_RNDZ);
                }
                if (k == 1) {
                    mpfr_set_ui(temp4[j][0], 0, MPFR_RNDZ);
                    mpfr_set_ui(temp4[j][1], 1, MPFR_RNDZ);
                }
                if (k == 2) {
                    fprintf(stderr, "error");
                    abort();
                }
                k++;
            }
        }

        //Calculating D(phi_{i+1}^{-1})
        k = 0;
        for (int j = 0; j < 3; j++) {
            if (j == 3 - charts[(i+1)%10]) {
                mpfr_set_ui(temp5[0][j], 0, MPFR_RNDZ);
                mpfr_set_ui(temp5[1][j], 0, MPFR_RNDZ);
            } else if (j == 6 - charts[(i+1)%10]) {
                mpfr_set_ui(temp5[0][j], 0, MPFR_RNDZ);
                mpfr_set_ui(temp5[1][j], 0, MPFR_RNDZ);
            } else {
                if (k == 0) {
                    mpfr_set_ui(temp5[0][j], 1, MPFR_RNDZ);
                    mpfr_set_ui(temp5[1][j], 0, MPFR_RNDZ);
                }
                if (k == 1) {
                    mpfr_set_ui(temp5[0][j], 0, MPFR_RNDZ);
                    mpfr_set_ui(temp5[1][j], 1, MPFR_RNDZ);
                }
                if (k == 2) {
                    fprintf(stderr, "error");
                    abort();
                }
                k++;
            }
        }
        
        
        //Lift to X(\R)
        mpfr_point p;
	    init_point(&p);
	    lift_exact_charts(&p, a[i], b[i], charts[i]);

        //Apply Ix, Iy, Iz and calculate  D(Ix), D(Iy), D(Iz)
        DixExact(p, x_mat);
        ix_exact(&p); 
        DiyExact(p, y_mat);
        iy_exact(&p);
        DizExact(p, z_mat);
        iz_exact(&p);
    
        //Compute temp3 = Df
        multiplyMatricesExact(y_mat, x_mat, temp2, 3, 3, 3, 3);
        multiplyMatricesExact(z_mat, temp2, temp3, 3, 3, 3, 3);

        //Compute and print Df^c
        multiplyMatricesAuxExact(temp5, temp3, temp1, 2, 3, 3, 3);
        multiplyMatricesAuxExact(temp3, temp4, temp1, 2, 3, 3, 2);
        fprintf(stderr, "(Df^c}_{x_%d^c} =  \n", i);
        printMatrixExact(temp4, 2, 2);
    }
	exit(0);

}

// alpha = (xy)/((1+x^2)(1+y^2))
void alphaExact(mpfr_t result, mpfr_t x, mpfr_t y) {
    mpfr_t temp1, temp2, temp3;

    mpfr_inits(temp1, temp2, temp3, NULL);

    // temp1 = x * y
    mpfr_mul(temp1, x, y, MPFR_RNDZ);

    // temp2 = 1 + x * x
    mpfr_mul(temp2, x, x, MPFR_RNDZ);
    mpfr_add_ui(temp2, temp2, 1, MPFR_RNDZ);

    // temp3 = 1 + y * y
    mpfr_mul(temp3, y, y, MPFR_RNDZ);
    mpfr_add_ui(temp3, temp3, 1, MPFR_RNDZ);

    // temp2 = temp2 * temp3
    mpfr_mul(temp2, temp2, temp3, MPFR_RNDZ);

    // result = temp1 / temp2
    mpfr_div(result, temp1, temp2, MPFR_RNDZ);

    mpfr_clears(temp1, temp2, temp3, NULL);
}

// beta = 1/((1+x^2)(1+y^2))

void betaExact(mpfr_t result, mpfr_t x, mpfr_t y) {
    mpfr_t temp2, temp3;

    mpfr_inits(temp2, temp3, NULL);

    // temp2 = 1 + x * x
    mpfr_mul(temp2, x, x, MPFR_RNDZ);
    mpfr_add_ui(temp2, temp2, 1, MPFR_RNDZ);

    // temp3 = 1 + y * y
    mpfr_mul(temp3, y, y, MPFR_RNDZ);
    mpfr_add_ui(temp3, temp3, 1, MPFR_RNDZ);

    // temp2 = temp2 * temp3
    mpfr_mul(temp2, temp2, temp3, MPFR_RNDZ);

    // result = 1 / temp2
    mpfr_ui_div(result, 1, temp2, MPFR_RNDZ);
}


void ix_exact(mpfr_point *p) {
    mpfr_point q;
    init_point(&q);

    mpfr_set(q.x, p->y, MPFR_RNDZ);
    mpfr_set(q.y, p->z, MPFR_RNDZ);
    mpfr_set(q.z, p->x, MPFR_RNDZ);
    
    iz_exact(&q);
    mpfr_set(p->x, q.z, MPFR_RNDZ);
}

void iy_exact(mpfr_point *p) {
    mpfr_point q;
    init_point(&q);

    mpfr_set(q.x, p->x, MPFR_RNDZ);
    mpfr_set(q.y, p->z, MPFR_RNDZ);
    mpfr_set(q.z, p->y, MPFR_RNDZ);

	iz_exact(&q);
    mpfr_set(p->y, q.z, MPFR_RNDZ);
}

//iz (x,y,z) = (x,y, -z - (10 alpha (x,y)))
void iz_exact(mpfr_point *p) {
	mpfr_t temp1;
    mpfr_inits(temp1, NULL);

    // temp1 = -10 * alpha(x,y)
    alphaExact(temp1, p->x, p->y);
    mpfr_mul_ui(temp1, temp1, 10, MPFR_RNDZ);
    mpfr_neg(temp1, temp1, MPFR_RNDZ);

    // result = temp1 - z
    mpfr_sub(p->z, temp1, p->z, MPFR_RNDZ);
    mpfr_clears(temp1, NULL);
}

// D(x,y) = 100((x^2)(y^2))) + 8((1+x^2)(1+y^2)) + 4((1+x^2)^2 (1+y^2)^2)
void DExact(mpfr_t result, mpfr_t x, mpfr_t y) {
    mpfr_t temp1, temp2, temp3, temp4;
    mpfr_inits(temp1, temp2, temp3, temp4, NULL);

    // temp1 = 100 * x * x * y * y
    mpfr_mul(temp1, x, x, MPFR_RNDZ);
    mpfr_mul(temp2, y, y, MPFR_RNDZ);
    mpfr_mul(temp1, temp1, temp2, MPFR_RNDZ);
    mpfr_mul_ui(temp1, temp1, 100, MPFR_RNDZ);

    // temp2 = 8 * (1 + x * x) * (1 + y * y)
    mpfr_mul(temp2, x, x, MPFR_RNDZ);
    mpfr_add_ui(temp2, temp2, 1, MPFR_RNDZ);
    mpfr_mul(temp3, y, y, MPFR_RNDZ);
    mpfr_add_ui(temp3, temp3, 1, MPFR_RNDZ);
    mpfr_mul(temp2, temp2, temp3, MPFR_RNDZ);
    mpfr_mul_ui(temp2, temp2, 8, MPFR_RNDZ);

    // temp3 = 4 * ((1 + x * x) * (1 + x * x)) * ((1 + y * y) * (1 + y * y))
    mpfr_mul(temp3, x, x, MPFR_RNDZ);
    mpfr_add_ui(temp3, temp3, 1, MPFR_RNDZ);
    mpfr_mul(temp3, temp3, temp3, MPFR_RNDZ);

    mpfr_mul(temp4, y, y, MPFR_RNDZ);
    mpfr_add_ui(temp4, temp4, 1, MPFR_RNDZ);
    mpfr_mul(temp4, temp4, temp4, MPFR_RNDZ);

    mpfr_mul(temp3, temp3, temp4, MPFR_RNDZ);
    mpfr_mul_ui(temp3, temp3, 4, MPFR_RNDZ);

    // result = temp1 + temp2 - temp3
    mpfr_add(result, temp1, temp2, MPFR_RNDZ);
    mpfr_sub(result, result, temp3, MPFR_RNDZ);

    mpfr_clears(temp1, temp2, temp3, temp4, NULL);
}

//p(x,y) = -5 alpha(x,y) + .5 (\beta(x,y) \sqrt(D(x,y)))
void pPlusExact(mpfr_t result,  mpfr_t x,  mpfr_t y) {
    mpfr_t temp1, temp2, temp3;

    mpfr_inits(temp1, temp2, temp3, NULL);

    //temp1 = -5 alpha
    alphaExact(temp1, x, y);
    mpfr_mul_si(temp1, temp1, -5, MPFR_RNDZ);

    //temp2 = (1/2)beta \sqrt{D}
    betaExact(temp2, x, y);
    DExact(temp3, x, y);
    mpfr_sqrt(temp3, temp3, MPFR_RNDZ);
    mpfr_mul(temp2, temp2, temp3, MPFR_RNDZ);
    mpfr_div_ui(temp2, temp2, 2, MPFR_RNDZ);

    //p_+ = -5 alpha + (1/2)beta \sqrt{D}
    mpfr_add(result, temp1, temp2, MPFR_RNDZ);

    mpfr_clears(temp1, temp2, temp3, NULL);
}
void pMinusExact(mpfr_t result,  mpfr_t x,  mpfr_t y) {
    mpfr_t temp1, temp2, temp3;

    mpfr_inits(temp1, temp2, temp3, NULL);

    //temp1 = -5 alpha
    alphaExact(temp1, x, y);
    mpfr_mul_si(temp1, temp1,-5, MPFR_RNDZ);

    //temp2 = (1/2)beta \sqrt{D}
    betaExact(temp2, x, y);
    DExact(temp3, x, y);
    mpfr_sqrt(temp3, temp3, MPFR_RNDZ);
    mpfr_mul(temp2, temp2, temp3, MPFR_RNDZ);
    mpfr_div_ui(temp2, temp2, 2, MPFR_RNDZ);

    //p_- = -5 alpha - (1/2)beta \sqrt{D}
    mpfr_sub(result, temp1, temp2, MPFR_RNDZ);

    mpfr_clears(temp1, temp2, temp3, NULL);
}


void lift_exact_charts(mpfr_point *p,  mpfr_t a,  mpfr_t b, int i) {
	mpfr_t temp0;
	mpfr_init(temp0);
	if(i == 1){
        pPlusExact(temp0,a,b);
		mpfr_set(p->x, a, MPFR_RNDZ);
		mpfr_set(p->y, b, MPFR_RNDZ);
        mpfr_set(p->z, temp0, MPFR_RNDZ);
	}
	if(i == 2){
		pPlusExact(temp0,a,b);
		mpfr_set(p->x, a, MPFR_RNDZ);
		mpfr_set(p->y, temp0, MPFR_RNDZ);
        mpfr_set(p->z, b, MPFR_RNDZ);
	}
	if(i == 3){
		pPlusExact(temp0,a,b);
		mpfr_set(p->x, temp0, MPFR_RNDZ);
		mpfr_set(p->y, a, MPFR_RNDZ);
        mpfr_set(p->z, b, MPFR_RNDZ);
	}
	if(i == 4){
        pMinusExact(temp0,a,b);
		mpfr_set(p->x, a, MPFR_RNDZ);
		mpfr_set(p->y, b, MPFR_RNDZ);
        mpfr_set(p->z, temp0, MPFR_RNDZ);
	}
	if(i == 5){
		pMinusExact(temp0,a,b);
		mpfr_set(p->x, a, MPFR_RNDZ);
		mpfr_set(p->y, temp0, MPFR_RNDZ);
        mpfr_set(p->z, b, MPFR_RNDZ);
	}
	if(i == 6){
		pMinusExact(temp0,a,b);
		mpfr_set(p->x, temp0, MPFR_RNDZ);
		mpfr_set(p->y, a, MPFR_RNDZ);
        mpfr_set(p->z, b, MPFR_RNDZ);
	}
	mpfr_clears(temp0, NULL);
}



void multiplyMatricesExact(mpfr_t **firstMatrix, mpfr_t **secondMatrix, mpfr_t **result, int ROWS1, int COLS1, int ROWS2, int COLS2) {
    for (int i = 0; i < ROWS1; ++i) {
        for (int j = 0; j < COLS2; ++j) {
            mpfr_set_ui(result[i][j], 0, MPFR_RNDZ);
            for (int k = 0; k < COLS1; ++k) {
                mpfr_t temp;
                mpfr_init(temp);
                mpfr_mul(temp, firstMatrix[i][k], secondMatrix[k][j], MPFR_RNDZ);
                mpfr_add(result[i][j], result[i][j], temp, MPFR_RNDZ);
                mpfr_clear(temp);
            }
        }
    }
}

void multiplyMatricesAuxExact(mpfr_t **firstMatrix, mpfr_t **secondMatrix, mpfr_t **result, int ROWS1, int COLS1, int ROWS2, int COLS2) {
    for (int i = 0; i < ROWS1; ++i) {
        for (int j = 0; j < COLS2; ++j) {
            mpfr_set_ui(result[i][j], 0, MPFR_RNDZ);
            for (int k = 0; k < COLS1; ++k) {
                mpfr_t temp;
                mpfr_init(temp);
                mpfr_mul(temp, firstMatrix[i][k], secondMatrix[k][j], MPFR_RNDZ);
                mpfr_add(result[i][j], result[i][j], temp, MPFR_RNDZ);
                mpfr_clear(temp);
            }
        }
    }
    for (int i = 0; i < ROWS1; ++i) {
        for (int j = 0; j < COLS2; ++j) {
            mpfr_set(secondMatrix[i][j], result[i][j], MPFR_RNDZ);
        }
    }
}

void printMatrixExact(mpfr_t **matrix, int ROWS1, int COLS1) {
    fprintf(stderr, "{");
    for (int i = 0; i < ROWS1; ++i) {
        fprintf(stderr, "{");
        for (int j = 0; j < COLS1; ++j) {
            char buffer[64];
            mpfr_sprintf(buffer, "%.15Rf", matrix[i][j]);
            fprintf(stderr, "%s", buffer);
            if (j < COLS1 - 1) {
                fprintf(stderr, ",");
            }
        }
        if (i == ROWS1 - 1) {
            fprintf(stderr, "}}\n");
        } else {
            fprintf(stderr, "},\n");
        }
    }
}
void alpha_xExact(mpfr_t result, const mpfr_t x, const mpfr_t y) {
    mpfr_t temp1, temp2, temp3;
    mpfr_inits(temp1, temp2, temp3, NULL);

    // y * (1 - x^2) / ((1 + x^2)^2 * (1+y^2))

    //mpfr_printf("x,y is %.10Rf, %.10Rf\n", x,y);

    // Calculate temp1 = y^2 + 1
    mpfr_mul(temp1, y, y, MPFR_RNDZ);
    mpfr_add_ui(temp1, temp1, 1, MPFR_RNDZ);
   // mpfr_printf("temp1 (y^2 + 1): %.10Rf\n", temp1);

    // Calculate temp2 = (1 + x^2)^2
    mpfr_mul(temp2, x, x, MPFR_RNDZ);
    mpfr_add_ui(temp2, temp2, 1, MPFR_RNDZ);
    mpfr_mul(temp2, temp2, temp2, MPFR_RNDZ);
   // mpfr_printf("temp2 ((1 + x^2)^2): %.10Rf\n", temp2);

    // Calculate temp1 = (y^2 + 1) * (1 + x^2)^2
    mpfr_mul(temp1, temp1, temp2, MPFR_RNDZ);
   // mpfr_printf("temp1 ((y^2 + 1) * (1 + x^2)^2): %.10Rf\n", temp1);

    // Calculate temp2 = 1 - x^2
    mpfr_mul(temp2, x, x, MPFR_RNDZ);
    mpfr_ui_sub(temp2, 1, temp2, MPFR_RNDZ);
   // mpfr_printf("temp2 (1 - x^2): %.10Rf\n", temp2);

    // Calculate temp2 = (1 - x^2) * y
    mpfr_mul(temp2, temp2, y, MPFR_RNDZ);
   // mpfr_printf("temp2 ((1 - x^2) * y): %.10Rf\n", temp2);

    // Calculate result = temp2 / temp1
    mpfr_div(result, temp2, temp1, MPFR_RNDZ);
   // mpfr_printf("result: %.10Rf\n", result);

    // Clear MPFR variables
    mpfr_clears(temp1, temp2, temp3, NULL);
}
void alpha_yExact(mpfr_t result, const mpfr_t x, const mpfr_t y) {
    alpha_xExact(result, y, x);
}

void DixExact(mpfr_point p, mpfr_t **matrix) {
    mpfr_set_si(matrix[0][0], -1, MPFR_RNDZ);
    mpfr_t temp1, temp2, temp3;
    mpfr_inits(temp1, temp2, temp3, NULL);

    // matrix[0][1] = -a  * p.z * (1 - p.y * p.y) / ((1 + p.z * p.z) * (1 + p.y * p.y) * (1 + p.y * p.y))
    
    alpha_xExact(temp1, p.y, p.z);
    mpfr_mul_ui(matrix[0][1], temp1, 10, MPFR_RNDZ);
    mpfr_neg(matrix[0][1], matrix[0][1], MPFR_RNDZ);



    // matrix[0][2] = -a * p.y * (1 - p.z * p.z) / ((1 + p.y * p.y) * (1 + p.z * p.z) * (1 + p.z * p.z))
    alpha_yExact(temp2, p.y, p.z);
    mpfr_mul_ui(matrix[0][2], temp2, 10, MPFR_RNDZ);
    mpfr_neg(matrix[0][2], matrix[0][2], MPFR_RNDZ);


    mpfr_set_ui(matrix[1][0], 0, MPFR_RNDZ);
    mpfr_set_ui(matrix[1][1], 1, MPFR_RNDZ);
    mpfr_set_ui(matrix[1][2], 0, MPFR_RNDZ);
    mpfr_set_ui(matrix[2][0], 0, MPFR_RNDZ);
    mpfr_set_ui(matrix[2][1], 0, MPFR_RNDZ);
    mpfr_set_ui(matrix[2][2], 1, MPFR_RNDZ);

    mpfr_clears(temp1, temp2, temp3, NULL);
}

void DiyExact(mpfr_point p, mpfr_t **matrix) {
    mpfr_set_ui(matrix[0][0], 1, MPFR_RNDZ);
    mpfr_set_ui(matrix[0][1], 0, MPFR_RNDZ);
    mpfr_set_ui(matrix[0][2], 0, MPFR_RNDZ);

    mpfr_t temp1, temp2, temp3;
    mpfr_inits(temp1, temp2, temp3, NULL);

    // matrix[1][0] = -a * p.z * (1 - p.x * p.x) / ((1 + p.z * p.z) * (1 + p.x * p.x) * (1 + p.x * p.x))
    alpha_xExact(temp1, p.x, p.z);
    mpfr_mul_ui(matrix[1][0], temp1, 10, MPFR_RNDZ);
    mpfr_neg(matrix[1][0], matrix[1][0], MPFR_RNDZ);

    mpfr_set_si(matrix[1][1], -1, MPFR_RNDZ);

    // matrix[1][2] = -a * p.x * (1 - p.z * p.z) / ((1 + p.x * p.x) * (1 + p.z * p.z) * (1 + p.z * p.z))
    alpha_yExact(temp2, p.x, p.z);
    mpfr_mul_ui(matrix[1][2], temp2, 10, MPFR_RNDZ);
    mpfr_neg(matrix[1][2], matrix[1][2], MPFR_RNDZ);


    mpfr_set_ui(matrix[2][0], 0, MPFR_RNDZ);
    mpfr_set_ui(matrix[2][1], 0, MPFR_RNDZ);
    mpfr_set_ui(matrix[2][2], 1, MPFR_RNDZ);

    mpfr_clears(temp1, temp2, temp3, NULL);
}

void DizExact(mpfr_point p, mpfr_t **matrix) {
    mpfr_set_ui(matrix[0][0], 1, MPFR_RNDZ);
    mpfr_set_ui(matrix[0][1], 0, MPFR_RNDZ);
    mpfr_set_ui(matrix[0][2], 0, MPFR_RNDZ);
    mpfr_set_ui(matrix[1][0], 0, MPFR_RNDZ);
    mpfr_set_ui(matrix[1][1], 1, MPFR_RNDZ);
    mpfr_set_ui(matrix[1][2], 0, MPFR_RNDZ);
    mpfr_t temp1, temp2, temp3;
    mpfr_inits(temp1, temp2, temp3, NULL);

    // matrix[2][0] = -a * p.y * (1 - p.x * p.x) / ((1 + p.y * p.y) * (1 + p.x * p.x) * (1 + p.x * p.x))
    alpha_xExact(temp1, p.x, p.y);
    mpfr_mul_ui(matrix[2][0], temp1, 10, MPFR_RNDZ);
    mpfr_neg(matrix[2][0], matrix[2][0], MPFR_RNDZ);

    // matrix[2][1] = -a * p.x * (1 - p.y * p.y) / ((1 + p.x * p.x) * (1 + p.y * p.y) * (1 + p.y * p.y))
    alpha_yExact(temp1, p.x, p.y);
    mpfr_mul_ui(matrix[2][1], temp1, 10, MPFR_RNDZ);
    mpfr_neg(matrix[2][1], matrix[2][1], MPFR_RNDZ);

    mpfr_set_si(matrix[2][2], -1, MPFR_RNDZ);

    mpfr_clears(temp1, temp2, temp3, NULL);
}

//Dx(x,y) = 200 * x * y * y + 16 * x * (1 + y * y) - 16 * x * (1 + x * x) * (1 + y * y) * (1 + y * y)
void DxExact(mpfr_t result, mpfr_t x, mpfr_t y) {
    mpfr_t temp1, temp2, temp3, temp4;
    mpfr_inits(temp1, temp2, temp3, temp4, NULL);

    // temp1 = 200 * x * y * y
    mpfr_mul(temp1, x, y, MPFR_RNDZ);
    mpfr_mul(temp1, temp1, y, MPFR_RNDZ);
    mpfr_mul_ui(temp1, temp1, 200, MPFR_RNDZ);

    // temp2 = 16 * x * (1 + y * y)
    mpfr_mul(temp2, y, y, MPFR_RNDZ);
    mpfr_add_ui(temp2, temp2, 1, MPFR_RNDZ);
    mpfr_mul(temp2, temp2, x, MPFR_RNDZ);
    mpfr_mul_ui(temp2, temp2, 16, MPFR_RNDZ);

    // temp3 = 16 * x * (1 + x * x) * (1 + y * y) * (1 + y * y)
    mpfr_mul(temp3, x, x, MPFR_RNDZ);
    mpfr_add_ui(temp3, temp3, 1, MPFR_RNDZ);
    mpfr_mul(temp4, y, y, MPFR_RNDZ);
    mpfr_add_ui(temp4, temp4, 1, MPFR_RNDZ);
    mpfr_mul(temp4, temp4, temp4, MPFR_RNDZ);
    mpfr_mul(temp3, temp3, temp4, MPFR_RNDZ);
    mpfr_mul(temp3, temp3, x, MPFR_RNDZ);
    mpfr_mul_ui(temp3, temp3, 16, MPFR_RNDZ);

    // result = temp1 + temp2 - temp3
    mpfr_add(result, temp1, temp2, MPFR_RNDZ);
    mpfr_sub(result, result, temp3, MPFR_RNDZ);

    mpfr_clears(temp1, temp2, temp3, temp4, NULL);
}

void DyExact(mpfr_t result, mpfr_t x, mpfr_t y) {
    DxExact(result, y, x);
}

//Dxx(x,y) = 200 * y * y + 16 * (1 + y * y) - 32 * x * x * (1 + y * y) * (1 + y * y) - 16 * (1 + x * x) * (1 + y * y) * (1 + y * y)
void DxxExact(mpfr_t result, mpfr_t x, mpfr_t y) {
    mpfr_t temp1, temp2, temp3, temp4, temp5;
    mpfr_inits(temp1, temp2, temp3, temp4, temp5, NULL);

    // temp1 = 200 * y * y
    mpfr_mul(temp1, y, y, MPFR_RNDZ);
    mpfr_mul_ui(temp1, temp1, 200, MPFR_RNDZ);

    // temp2 = 16 * (1 + y * y)
    mpfr_mul(temp2, y, y, MPFR_RNDZ);
    mpfr_add_ui(temp2, temp2, 1, MPFR_RNDZ);
    mpfr_mul_ui(temp2, temp2, 16, MPFR_RNDZ);

    // temp3 = 32 * x * x * (1 + y * y) * (1 + y * y)
    mpfr_mul(temp3, x, x, MPFR_RNDZ);
    mpfr_mul(temp4, y, y, MPFR_RNDZ);
    mpfr_add_ui(temp4, temp4, 1, MPFR_RNDZ);
    mpfr_mul(temp4, temp4, temp4, MPFR_RNDZ);
    mpfr_mul(temp3, temp3, temp4, MPFR_RNDZ);
    mpfr_mul_ui(temp3, temp3, 32, MPFR_RNDZ);

    // temp4 = 16 * (1 + x * x) * (1 + y * y) * (1 + y * y)
    mpfr_mul(temp4, x, x, MPFR_RNDZ);
    mpfr_add_ui(temp4, temp4, 1, MPFR_RNDZ);
    mpfr_mul(temp5, y, y, MPFR_RNDZ);
    mpfr_add_ui(temp5, temp5, 1, MPFR_RNDZ);
    mpfr_mul(temp5, temp5, temp5, MPFR_RNDZ);
    mpfr_mul(temp4, temp4, temp5, MPFR_RNDZ);
    mpfr_mul_ui(temp4, temp4, 16, MPFR_RNDZ);

    // result = temp1 + temp2 - temp3 - temp4
    mpfr_add(result, temp1, temp2, MPFR_RNDZ);
    mpfr_sub(result, result, temp3, MPFR_RNDZ);
    mpfr_sub(result, result, temp4, MPFR_RNDZ);

    mpfr_clears(temp1, temp2, temp3, temp4, temp5, NULL);
}
void DyyExact(mpfr_t result, mpfr_t x, mpfr_t y) {
    DxxExact(result, y, x);
}

//Dxx(x,y) = 432 * x * y - 64 * x * (1 + x * x) * y * (1 + y * y)
void DxyExact(mpfr_t result, mpfr_t x, mpfr_t y) {
    mpfr_t temp1, temp2, temp3;
    mpfr_inits(temp1, temp2, temp3, NULL);

    // temp1 = 432 * x * y
    mpfr_mul(temp1, x, y, MPFR_RNDZ);
    mpfr_mul_ui(temp1, temp1, 432, MPFR_RNDZ);

    // temp2 = 64 * x * (1 + x * x) * y * (1 + y * y)
    mpfr_mul(temp2, x, x, MPFR_RNDZ);
    mpfr_add_ui(temp2, temp2, 1, MPFR_RNDZ);
    mpfr_mul(temp2, temp2, x, MPFR_RNDZ);
    mpfr_mul(temp2, temp2, y, MPFR_RNDZ);
    mpfr_mul(temp3, y, y, MPFR_RNDZ);
    mpfr_add_ui(temp3, temp3, 1, MPFR_RNDZ);
    mpfr_mul(temp2, temp2, temp3, MPFR_RNDZ);
    mpfr_mul_ui(temp2, temp2, 64, MPFR_RNDZ);

    // result = temp1 - temp2
    mpfr_sub(result, temp1, temp2, MPFR_RNDZ);

    mpfr_clears(temp1, temp2, temp3, NULL);
}


//px(s,t)= num/denom where 
/* num = (
    s(-2+23t^2) - s^3(2+27t^2)
    - 5t(sqrt(1 - t^4 - s^4(1 + t^2)^2 + s^2(23t^2 - 2t^4)))
    + 5s^2t(sqrt(1 - t^4 - s^4(1 + t^2)^2 + s^2(23t^2 - 2t^4)))
)*/

// denom = (1+s^2)^2*(1+t^2)*sqrt(1-t^4-s^4*(1+t^2)^2+s^2*(23*t^2-2*t^4))

void pxExact(mpfr_t result, mpfr_t s, mpfr_t t) {
    mpfr_t numerator, denominator, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
    mpfr_inits(numerator, denominator, temp1, temp2, temp3, temp4, temp5, temp6, temp7, NULL);

    // numerator = s * (-2 + 23 * t * t) - s * s * s * (2 + 27 * t * t)
    mpfr_mul(temp1, t, t, MPFR_RNDZ);
    mpfr_mul_ui(temp1, temp1, 23, MPFR_RNDZ);
    mpfr_add_si(temp1, temp1, -2, MPFR_RNDZ);
    mpfr_mul(temp1, temp1, s, MPFR_RNDZ);

    mpfr_mul(temp2, s, s, MPFR_RNDZ);
    mpfr_mul(temp2, temp2, s, MPFR_RNDZ);

    mpfr_mul_ui(temp3, t, 27, MPFR_RNDZ);
    mpfr_mul(temp3, t, temp3, MPFR_RNDZ);
    mpfr_add_ui(temp3, temp3, 2, MPFR_RNDZ);

    mpfr_mul(temp2, temp2, temp3, MPFR_RNDZ);
    mpfr_sub(temp1, temp1, temp2, MPFR_RNDZ);

    //temp4 = 1 - t^4 - s^4 * (1 + t^2)^2
    mpfr_pow_ui(temp4, t, 4, MPFR_RNDZ);
    mpfr_pow_ui(temp5, s, 4, MPFR_RNDZ);
    mpfr_mul(temp6, t, t, MPFR_RNDZ);
    mpfr_add_ui(temp6, temp6, 1, MPFR_RNDZ);
    mpfr_mul(temp6, temp6, temp6, MPFR_RNDZ);
    mpfr_mul(temp5, temp5, temp6, MPFR_RNDZ);
    mpfr_ui_sub(temp4, 1, temp4, MPFR_RNDZ); //ethan changed
    mpfr_sub(temp4, temp4, temp5, MPFR_RNDZ);

    //temp4 = 1 - t^4 - s^4 * (1 + t^2)^2 + s^2 * (23 * t^2 - 2 * t^4)
    mpfr_mul(temp5, s, s, MPFR_RNDZ);
    mpfr_mul_ui(temp6, t, 23, MPFR_RNDZ);
    mpfr_mul(temp6, temp6, t, MPFR_RNDZ);
    mpfr_pow_ui(temp7, t, 4, MPFR_RNDZ);
    mpfr_mul_ui(temp7, temp7, 2, MPFR_RNDZ);
    mpfr_sub(temp6, temp6, temp7, MPFR_RNDZ);
    mpfr_mul(temp6, temp6, s, MPFR_RNDZ);
    mpfr_mul(temp6, temp6, s, MPFR_RNDZ);
    mpfr_add(temp4, temp4, temp6, MPFR_RNDZ);

    //temp4 = sqrt(1 - t^4 - s^4 * (1 + t^2)^2 + s^2 * (23 * t^2 - 2 * t^4))
    mpfr_sqrt(temp4, temp4, MPFR_RNDZ);


    // numerator += -5 * t * temp4 + 5 * s^2 * t * temp4
    mpfr_mul(temp5, t, temp4, MPFR_RNDZ);
    mpfr_mul_si(temp5, temp5, -5, MPFR_RNDZ);
    mpfr_add(numerator, temp1, temp5, MPFR_RNDZ);

    mpfr_mul(temp5, s, s, MPFR_RNDZ);
    mpfr_mul(temp5, temp5, t, MPFR_RNDZ);
    mpfr_mul(temp5, temp5, temp4, MPFR_RNDZ);
    mpfr_mul_ui(temp5, temp5, 5, MPFR_RNDZ);
    mpfr_add(numerator, numerator, temp5, MPFR_RNDZ);

    // denominator = (1 + s^2)^2 * (1 + t^2) * sqrt(1 - t^4 - s^4 * (1 + t^2)^2 + s^2 * (23 * t^2 - 2 * t^4))
    mpfr_mul(temp1, s, s, MPFR_RNDZ);
    mpfr_add_ui(temp1, temp1, 1, MPFR_RNDZ);
    mpfr_mul(temp1, temp1, temp1, MPFR_RNDZ);

    mpfr_mul(temp2, t, t, MPFR_RNDZ);
    mpfr_add_ui(temp2, temp2, 1, MPFR_RNDZ);

    mpfr_mul(denominator, temp1, temp2, MPFR_RNDZ);
    mpfr_mul(denominator, denominator, temp4, MPFR_RNDZ);

    // result = numerator / denominator
    mpfr_div(result, numerator, denominator, MPFR_RNDZ);

    mpfr_clears(numerator, denominator, temp1, temp2, temp3, temp4, temp5, temp6, temp7, NULL);
}

void pyExact(mpfr_t result, mpfr_t s, mpfr_t t) {
    pxExact(result, t, s);
}