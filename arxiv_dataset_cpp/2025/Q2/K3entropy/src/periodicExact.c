#include <math.h>
#include "k3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdarg.h>
#include <mpfr.h> 


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

int main(int argc, char *argv[]) {
    //Setting default rounding mode
    mpfr_set_default_rounding_mode(MPFR_RNDZ);

    // Set precision to 5000 bits and exponent range to [-1073,1024]
    mpfr_set_default_prec(500); 
    mpfr_set_emin (-1073); mpfr_set_emax (1024); 

    //Looking for delta pseudo-orbit
    mpfr_t delta;
    mpfr_inits(delta, NULL);
    mpfr_set_str(delta, "1e-29", 10, MPFR_RNDZ); //set delta here

    //Define pseudo-orbit
    //specify length of pseudo_orbit
    int porbit_length = 10;
    mpfr_t a[porbit_length];
    mpfr_t b[porbit_length];
    for (int i = 0; i < porbit_length; ++i) {
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

    

    //Check for delta pseudo-orbit
    mpfr_t a_temp, b_temp, temp;
    mpfr_inits(a_temp, b_temp, temp, NULL);
    for (int i = 0; i < porbit_length; i++) {
        mpfr_set (a_temp, a[i], MPFR_RNDZ);
        mpfr_set (b_temp, b[i], MPFR_RNDZ);
        f_c(a_temp, b_temp, charts[i], charts[(i + 1) % porbit_length]);
        dist_2plane(temp, a_temp, b_temp, a[(i + 1) % porbit_length], b[(i + 1) % porbit_length]);
        if (mpfr_cmp(temp, delta) > 0) {
            fprintf(stderr, "failure \n");
            break;
        } 
        if(i == porbit_length-1){
            fprintf(stderr, "successful delta pseudo-orbit \n");
        }
    }
    mpfr_clears(a_temp,b_temp, temp, NULL);
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



void f_exact(mpfr_point *p) {
	ix_exact(p);
    iy_exact(p);
    iz_exact(p);
}


void f_c(mpfr_t s,  mpfr_t t, int i, int j) {
	mpfr_point temp;
	init_point(&temp);
	lift_exact_charts(&temp, s, t, i);
	f_exact(&temp);
	if(j == 1 || j== 4){
		mpfr_set(s, temp.x,MPFR_RNDZ);
		mpfr_set(t, temp.y,MPFR_RNDZ);
	}
	if(j == 2 || j== 5){
		mpfr_set(s, temp.x,MPFR_RNDZ);
		mpfr_set(t, temp.z,MPFR_RNDZ);
	}
	if(j == 3 || j== 6){
		mpfr_set(s, temp.y,MPFR_RNDZ);
		mpfr_set(t, temp.z,MPFR_RNDZ);
	}
	mpfr_clears(temp.x, temp.y, temp.z, NULL); 
}


void dist_2plane(mpfr_t result, mpfr_t a1, mpfr_t b1, mpfr_t a2, mpfr_t b2) {
    mpfr_t dx, dy, sum;

    mpfr_inits(dx, dy, sum, NULL);

    mpfr_sub(dx, a1, a2, MPFR_RNDZ);
    mpfr_sub(dy, b1, b2, MPFR_RNDZ);

    mpfr_mul(dx, dx, dx, MPFR_RNDZ);
    mpfr_mul(dy, dy, dy, MPFR_RNDZ);

    mpfr_add(sum, dx, dy, MPFR_RNDZ);

    mpfr_sqrt(result, sum, MPFR_RNDZ);

    mpfr_clears(dx, dy, sum, NULL);
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