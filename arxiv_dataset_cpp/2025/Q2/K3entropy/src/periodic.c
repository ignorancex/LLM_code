
/* For finding recurrent points with controllable precision */

#include <math.h>
#include "k3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <gmp.h> //for precision orbits


static double centerx = CENTERX, centery = CENTERY, radius = RADIUS;
static double length=LENGTH, sep = SEP;
static int drawtop = 1, drawbot = 1, verbose=VERBOSE, spin=0, unravel = 0;

/* Very large array to dynamically store paths*/
point ps[PSMAX];

//very important for determine_arrays
int pos_array[PSMAX]; //HOW BIG SHOULD I MAKE THESE??
int above_array[PSMAX];

int orbit_length = 23;
const char *word[PSMAX];
int above = -100000;
int current_pos = -10000;

//new point structure
typedef struct {
    mpf_t x, y, z;
} mpf_point;

void init_point(mpf_point *p) {
    mpf_init(p->x);
    mpf_init(p->y);
    mpf_init(p->z);
}

//function declarations
void involute_exact(mpf_t x, mpf_t y, mpf_t z);
void ix_exact(mpf_point *p);
void iy_exact(mpf_point *p);
void iz_exact(mpf_point *p);
void f_exact(mpf_point *p);
void dist_exact(mpf_t result, mpf_point p1, mpf_point p2);
void copy_point(mpf_point *dest, const mpf_point *src);
bool is_recurrent_exact(mpf_point p, mpf_t delta);
void search_near_exact(mpf_point *q, mpf_t epsilon, mpf_t delta, int N);
void search_near_exact_line(mpf_point *q, mpf_t epsilon, mpf_t delta, int N);
void lift_exact(mpf_point *p,  mpf_t x,  mpf_t y);
void lift_exact_under(mpf_point *p,  mpf_t x,  mpf_t y);
void lift_exact_charts(mpf_point *p,  mpf_t x,  mpf_t y, int i); //lifts with the ith chart
bool unreal_exact( mpf_t x,  mpf_t y);
void lift_exact_y(mpf_point *p,  mpf_t x,  mpf_t z);
void f_c(mpf_t a,  mpf_t b,  int i, int j); //i = incoming chart, j = outgoing chart
void projection_chart(mpf_point p, mpf_t a, mpf_t b, int i);
void dist_2plane(mpf_t result, mpf_t a1, mpf_t b1, mpf_t a2, mpf_t b2);

mpf_t ae, be;

int main(int argc, char *argv[]) {
	int i,n;
	double a,b;

	//read in values from stable.run
	for(i=1; i<argc; i++)
	{	if(argv[i][0] != '-') usage();
		switch(argv[i][1])
	{
		case 'b':
		drawtop = 0;
		break;

		case 'c':
		if(argc <= (i+2)) usage();
		sscanf(argv[++i],"%lf",&centerx);
		sscanf(argv[++i],"%lf",&centery);
		break;

		case 'e':
		if(argc <= (i+1)) usage();
		sscanf(argv[++i],"%lf",&sep);
		setsep(sep);
		break;

		case 'l':
		if(argc <= (i+1)) usage();
		sscanf(argv[++i],"%lf",&length);
		break;

		case 'p':
		if(argc <= (i+2)) usage();
		sscanf(argv[++i],"%lf",&a);
		sscanf(argv[++i],"%lf",&b);
		setk3(a,b);
		break;

		case 'q':
		verbose = 0;
		break;

		case 'r':
		if(argc <= (i+1)) usage();
		sscanf(argv[++i],"%lf",&radius);
		break;

		case 's':
		spin = 1;
		break;

		case 't':
		drawbot = 0;
		break;

		case 'u': /*ethan added unravel*/
		unravel = 1;
		break;

		default:
		usage();
	}
	}

	//recurrent search corner 
	
	/*
	

	for(int i = 0; i <= 1000; i++){
		if(i % 23 == 0){
			pprint(p);
		}
		p = f(p);
	}*/
    //open postscript file, declare window
	point q = {0.3249175414, 0.0042844224, 0.89316291607056474966};

	//ps_open(argc,argv);
	//ps_window(centerx,centery,radius);

    //point starter = lift(.1,.1);
	//draw_background(starter,200000);

	//search_near(q, 1e-11, 100);

	//close postscript file
	//ps_close();
	/*
	point p;
	p.x = -1.72122700000000000000;
	p.y = 1.07494096536647620365;
	p.z =  1.72122700000000000000;
	for(int i = 0; i < 7; i++){
		pprint(p);
		p = f(p);
	}
	//abort(); */
	

	mpf_set_default_prec(500); //change to 167 later
    mpf_t x, z, y, s, result, epsilon, delta;
    mpf_inits(x, z, s,y, ae, be, result, epsilon, delta, NULL);
	

	//mpf_set_str(x, "1.52907383338931296461", 10);
	//mpf_set_str(z, "1.52907383338931296461", 10);

	mpf_set_str(x, "-1.7268954487548584263288547244743691602526437", 10); //-1.72689538537
	mpf_set_str(z, "-1.7268954487548584263288547244743691602526437", 10); //-1.72689538537
	mpf_set_str(y, "0", 10);

	mpf_set_str(s, "4.5", 10);
	mpf_set_str(ae, "10", 10);
	mpf_set_str(be, "2", 10);


	mpf_neg(z,z);  

    

	mpf_point point;
	mpf_point point2;
    init_point(&point);
	init_point(&point2);
	mpf_set_str(epsilon, "1e-43", 10);//10 is the base //1e-12
	mpf_set_str(delta, "1e-38", 10); //recurrence requirement //1e-7
	//mpf_set_str(epsilon, "1e-9", 10);
	//mpf_set_str(delta, "1e-11", 10); //recurrence requirement

	gmp_fprintf(stderr, "%.50Ff",x) ;
	
	lift_exact_y(&point, x, z);

	gmp_fprintf(stderr, "starting point: (%.50Ff, %.50Ff, %.50Ff)\n", point.x, point.y, point.z);



	copy_point(&point2, &point);

	search_near_exact_line(&point, epsilon, delta, 1);
	//search_near_exact_line(&point, epsilon, delta, 1e5);

	abort();


  	//gmp_fprintf(stderr, "Result: (%.50Ff, %.50Ff, %.50Ff)\n", point.x, point.y, point.z);
	f_exact(&point);
	f_exact(&point);
	//pprint(f(q));
	//gmp_fprintf(stderr, "Result: (%.50Ff, %.50Ff, %.50Ff)\n", point.x, point.y, point.z);
	dist_exact(result, point, point2);
	//gmp_fprintf(stderr, "Distance is: %.260Ff\n", result);

    // Clear variables
    mpf_clears(x, z, s,y, ae, be, result, NULL);


	exit(0);

	//
}


void usage()
{
	fprintf(stderr,"Usage:  stable [options]\n");
	fprintf(stderr,"  -b          draw only bottom\n");
	fprintf(stderr,"  -c [x y]    window center\n");
	fprintf(stderr,"  -e [eps]    accuracy\n");
	fprintf(stderr,"  -l [length] length of stable manifold drawn\n");
	fprintf(stderr,"  -p [a b]    K3 parameters\n");
	fprintf(stderr,"  -q          quiet\n");
	fprintf(stderr,"  -r [r]      window radius\n");
	fprintf(stderr,"  -s          spin picture\n");
	fprintf(stderr,"  -t          draw only top\n");
	fprintf(stderr,"Postscript file written to stdout\n");
	exit(1);
}

point initial_point()
{
	point p;
	p = lift(sep/10,-sep/10);
	return(p);
}

void draw_background(point z, int n)/*Ethan added this*/
{
	for (int i = 0; i<n; i++){
		ps_dot_transparent(z);
		z = f(z);
	}
}

bool is_recurrent(point p) /*returns true of p is delta, k-recurrent for some k<=22*/
{
    double delta =  1e-5; // 1e-12;
	point q = f(p);
    for (int k = 1; k <= 22; k++) {
		//pprint(q);
		if(dist(p, q) < delta) {
			fprintf(stderr, "(%d)\n", k);
			return true;
    	}
		q = f(q);
    };
    return false;
}
void search_near(point q, double epsilon, int N){
	double x = q.x - N*epsilon;
	double y = q.y - N*epsilon;    
    int length = 2*N;
    point p = lift(x,y);
    for (int j = 1; j <= length; j++) {
		for(int i = 1; i <= length; i++){
		    if(!unreal(x,y)){
                p = lift(x,y);
                if(is_recurrent(p)) {
                    fprintf(stderr, "Success at (%.20lf, %.20lf, %.20lf)\n", p.x, p.y, p.z);
                    ps_dot(p,unravel);
                }
                ;//draws dot
            };
        	x = x + epsilon;
		};
		y = y + epsilon;
		x = q.x - N*epsilon; //ethan changed
    };
}
void recurrent_search() /*searches in epsilon-steps for points that are delta, k-recurrent for some k<=22*/
{
	double epsilon = 1e-2;
    double x = -.1;
	double y = -.1;    
    int length = radius/epsilon;
    point p = lift(x,y);
    for (int j = 1; j <= length; j++) {
		for(int i = 1; i <= length; i++){
		    if(!unreal(x,y)){
                p = lift(x,y);
                if(is_recurrent(p)) {
                    fprintf(stderr, "Success at (%.10lf, %.10lf, %.10lf)\n", p.x, p.y, p.z);
                    ps_dot(p,unravel);
                }
                ;//draws dot
            };
        	x = x + epsilon;
		};
		y = y + epsilon;
		x = -.1; //ethan changed
    };
}






void involute_exact(mpf_t x, mpf_t y, mpf_t z) {
    mpf_t temp1, temp2, temp3;

    mpf_inits(temp1, temp2, temp3, NULL);

    // temp1 = x * y
    mpf_mul(temp1, x, y);

    // temp2 = 1 + x * x
    mpf_mul(temp2, x, x);
    mpf_add_ui(temp2, temp2, 1);

    // temp3 = 1 + y * y
    mpf_mul(temp3, y, y);
    mpf_add_ui(temp3, temp3, 1);

    // temp2 = temp2 * temp3
    mpf_mul(temp2, temp2, temp3);

    // temp1 = -a * temp1 / temp2
    mpf_mul(temp1, temp1, ae);
    mpf_neg(temp1, temp1);
    mpf_div(temp1, temp1, temp2);

    // result = temp1 - z
    mpf_sub(z, temp1, z);

    mpf_clears(temp1, temp2, temp3, NULL);
}

void ix_exact(mpf_point *p)
{
	involute_exact(p->z,p->y,p->x);
}
void iy_exact(mpf_point *p)
{
	involute_exact(p->z,p->x,p->y);
}
void iz_exact(mpf_point *p)
{
	involute_exact(p->x,p->y,p->z);
}
void f_exact(mpf_point *p)
{
	ix_exact(p);
	iy_exact(p);
	iz_exact(p);
}
void f_c(mpf_t s,  mpf_t t, int i, int j) {
	mpf_point temp;
	init_point(&temp);
	lift_exact_charts(&temp, s, t, i);
	//gmp_fprintf(stderr, "(%.5Ff, %.5Ff, %.5Ff)\n", temp.x, temp.y, temp.z);
	f_exact(&temp);
	//gmp_fprintf(stderr, "after (%.5Ff, %.5Ff, %.5Ff)\n", temp.x, temp.y, temp.z);

	if(j == 1 || j== 4){
		mpf_set(s, temp.x);
		mpf_set(t, temp.y);
	}
	if(j == 2 || j== 5){
		mpf_set(s, temp.x);
		mpf_set(t, temp.z);
	}
	if(j == 3 || j== 6){
		mpf_set(s, temp.y);
		mpf_set(t, temp.z);
	}
	//gmp_fprintf(stderr, "inside (%.5Ff, %.5Ff)\n", s,t);
	mpf_clears(temp.x, temp.y, temp.z, NULL); //this right?
}


void copy_point(mpf_point *dest, const mpf_point *src) {
    mpf_set(dest->x, src->x);
    mpf_set(dest->y, src->y);
    mpf_set(dest->z, src->z);
}

void dist_exact(mpf_t result, mpf_point p1, mpf_point p2) {
    mpf_t dx, dy, dz, sum;

    mpf_inits(dx, dy, dz, sum, NULL);

    mpf_sub(dx, p1.x, p2.x);
    mpf_sub(dy, p1.y, p2.y);
    mpf_sub(dz, p1.z, p2.z);

    mpf_mul(dx, dx, dx);
    mpf_mul(dy, dy, dy);
    mpf_mul(dz, dz, dz);

    mpf_add(sum, dx, dy);
    mpf_add(sum, sum, dz);

    mpf_sqrt(result, sum);

    mpf_clears(dx, dy, dz, sum, NULL);
}

void dist_2plane(mpf_t result, mpf_t a1, mpf_t b1, mpf_t a2, mpf_t b2) {
    mpf_t dx, dy, sum;

    mpf_inits(dx, dy, sum, NULL);

    mpf_sub(dx, a1, a2);
    mpf_sub(dy, b1, b2);

    mpf_mul(dx, dx, dx);
    mpf_mul(dy, dy, dy);

    mpf_add(sum, dx, dy);

    mpf_sqrt(result, sum);

    mpf_clears(dx, dy, sum, NULL);
}

void projection_chart(mpf_point p, mpf_t a, mpf_t b, int i) {
   if (i == 1 || i == 4) {
		mpf_set(a, p.x);
		mpf_set(b, p.y);
   }
   if (i == 2 || i == 5) {
		mpf_set(a, p.x);
		mpf_set(b, p.z);
   }
   if (i == 3 || i == 6) {
		mpf_set(a, p.y);
		mpf_set(b, p.z);
   }
}

bool is_recurrent_exact(mpf_point p, mpf_t delta) /*returns true of p is delta, k-recurrent for k <= 22*/
{

	mpf_t temp2;
	mpf_init(temp2);
	mpf_point temp;
	init_point(&temp);
	copy_point(&temp, &p);
	//gmp_fprintf(stderr, "Result: (%.5Ff, %.5Ff, %.5Ff)\n", temp.x, temp.y, temp.z);
	//f_exact(&temp); just commented this out! mar 1
    for (int k = 1; k <= 10; k++) {
		//gmp_fprintf(stderr, "Result: (%.5Ff, %.5Ff, %.5Ff)\n", p.x, p.y, p.z);
		f_exact(&temp);	
		//		fprintf(stderr, "(%d)\n", k)
    };
	dist_exact(temp2, p, temp);
	//gmp_fprintf(stderr, "Result: (%.5Ff, %.5Ff, %.5Ff)\n", temp.x, temp.y, temp.z);
	//gmp_fprintf(stderr, "distance is: (%.5Ff)\n",temp2);
	if(mpf_cmp(temp2, delta) < 0) {
		//gmp_fprintf(stderr, "Result: (%.5Ff, %.5Ff, %.5Ff)\n", temp.x, temp.y, temp.z);
		mpf_clears(temp2, NULL);

		return true;
	}
	mpf_clears(temp2, NULL);
	
    return false;
}


bool is_recurrent_exact_specific(mpf_t a, mpf_t b, mpf_t delta) /*returns true of p is delta, k-recurrent for k <= 22*/
{
	//fprintf(stderr, "ummm");
	mpf_t temp2;
	mpf_t a_start;
	mpf_t b_start;
	mpf_init(a_start);
	mpf_init(b_start);
	mpf_init(temp2);

	mpf_set(a_start, a);
	mpf_set(b_start, b);
	
	gmp_fprintf(stderr, "(%.30Ff, %.30Ff)\n", a,b);
	f_c(a, b, 6, 4);	
	gmp_fprintf(stderr, "(%.30Ff, %.30Ff)\n", a,b);
	f_c(a, b, 4, 5);	
	gmp_fprintf(stderr, "(%.30Ff, %.30Ff)\n", a,b);

	f_c(a, b, 5, 6);
	gmp_fprintf(stderr, "(%.30Ff, %.30Ff)\n", a,b);
	
	f_c(a, b, 6, 5);
	gmp_fprintf(stderr, "(%.30Ff, %.30Ff)\n", a,b);
	
	f_c(a, b, 5, 5);
	gmp_fprintf(stderr, "(%.30Ff, %.30Ff)\n", a,b);
	
	f_c(a, b, 5, 5);
	gmp_fprintf(stderr, "(%.30Ff, %.30Ff)\n", a,b);
	
	f_c(a, b, 5, 1);	
	gmp_fprintf(stderr, "(%.30Ff, %.30Ff)\n", a,b);

	f_c(a, b, 1, 5);
	gmp_fprintf(stderr, "(%.30Ff, %.30Ff)\n", a,b);
	
	f_c(a, b, 5, 3);
	gmp_fprintf(stderr, "(%.30Ff, %.30Ff)\n", a,b);
	
	f_c(a, b, 3, 6);
	gmp_fprintf(stderr, "(%.30Ff, %.30Ff)\n", a,b);
	



	
	dist_2plane(temp2, a_start, b_start, a,b);

	if(mpf_cmp(temp2, delta) < 0) {
		mpf_clears(temp2, NULL);
		return true;
	}
	mpf_clears(temp2, NULL);
    return false;
}


bool unreal_exact( mpf_t x,  mpf_t y) {
	mpf_t qa, qb, qc, temp1, temp2, sqrt_term;

    mpf_inits(qa, qb, qc, temp1, temp2, sqrt_term, NULL);

    // qa = (1 + x*x) * (1 + y*y)
    mpf_mul(temp1, x, x);
    mpf_add_ui(temp1, temp1, 1);
    mpf_mul(temp2, y, y);
    mpf_add_ui(temp2, temp2, 1);
    mpf_mul(qa, temp1, temp2);

    // qb = a * x * y
    mpf_mul(qb, ae, x);
    mpf_mul(qb, qb, y);

    // qc = qa - b
    mpf_sub(qc, qa, be);


    // p.z = (-qb + sqrt(qb*qb - 4*qa*qc)) / (2*qa)
    mpf_mul(temp1, qb, qb); // temp1 = qb * qb
    mpf_mul_ui(temp2, qa, 4); // temp2 = 4 * qa
    mpf_mul(temp2, temp2, qc); // temp2 = 4 * qa * qc
    mpf_sub(sqrt_term, temp1, temp2); // sqrt_term = qb*qb - 4*qa*qc
	if(mpf_cmp_ui(sqrt_term, 0) < 0){
		return true;
	} else {
		return false;
	}
}
void lift_exact(mpf_point *p,  mpf_t x,  mpf_t y) { //MAKE SURE TO THROW ERROR IF NOT REAL!

    mpf_t qa, qb, qc, temp1, temp2, sqrt_term;

    mpf_inits(qa, qb, qc, temp1, temp2, sqrt_term, NULL);

    // qa = (1 + x*x) * (1 + y*y)
    mpf_mul(temp1, x, x);
    mpf_add_ui(temp1, temp1, 1);
    mpf_mul(temp2, y, y);
    mpf_add_ui(temp2, temp2, 1);
    mpf_mul(qa, temp1, temp2);

    // qb = a * x * y
    mpf_mul(qb, ae, x);
    mpf_mul(qb, qb, y);

    // qc = qa - b
    mpf_sub(qc, qa, be);


    // p.x = x
    mpf_set(p->x, x);

    // p.y = y
    mpf_set(p->y, y);

    // p.z = (-qb + sqrt(qb*qb - 4*qa*qc)) / (2*qa)
    mpf_mul(temp1, qb, qb); // temp1 = qb * qb
    mpf_mul_ui(temp2, qa, 4); // temp2 = 4 * qa

    mpf_mul(temp2, temp2, qc); // temp2 = 4 * qa * qc
    mpf_sub(sqrt_term, temp1, temp2); // sqrt_term = qb*qb - 4*qa*qc

    mpf_sqrt(sqrt_term, sqrt_term); // sqrt_term = sqrt(qb*qb - 4*qa*qc)

    mpf_neg(qb, qb); // qb = -qb
    mpf_add(temp1, qb, sqrt_term); // temp1 = -qb + sqrt(qb*qb - 4*qa*qc)
    mpf_mul_ui(temp2, qa, 2); // temp2 = 2 * qa
    mpf_div(p->z, temp1, temp2); // p.z = temp1 / temp2


    mpf_clears(qa, qb, qc, temp1, temp2, sqrt_term, NULL);
}

void lift_exact_under(mpf_point *p,  mpf_t x,  mpf_t y) { //MAKE SURE TO THROW ERROR IF NOT REAL!

    mpf_t qa, qb, qc, temp1, temp2, sqrt_term;

    mpf_inits(qa, qb, qc, temp1, temp2, sqrt_term, NULL);

    // qa = (1 + x*x) * (1 + y*y)
    mpf_mul(temp1, x, x);
    mpf_add_ui(temp1, temp1, 1);
    mpf_mul(temp2, y, y);
    mpf_add_ui(temp2, temp2, 1);
    mpf_mul(qa, temp1, temp2);

    // qb = a * x * y
    mpf_mul(qb, ae, x);
    mpf_mul(qb, qb, y);

    // qc = qa - b
    mpf_sub(qc, qa, be);


    // p.x = x
    mpf_set(p->x, x);

    // p.y = y
    mpf_set(p->y, y);

    // p.z = (-qb + sqrt(qb*qb - 4*qa*qc)) / (2*qa)
    mpf_mul(temp1, qb, qb); // temp1 = qb * qb
    mpf_mul_ui(temp2, qa, 4); // temp2 = 4 * qa

    mpf_mul(temp2, temp2, qc); // temp2 = 4 * qa * qc
    mpf_sub(sqrt_term, temp1, temp2); // sqrt_term = qb*qb - 4*qa*qc

    mpf_sqrt(sqrt_term, sqrt_term); // sqrt_term = sqrt(qb*qb - 4*qa*qc)

    mpf_neg(qb, qb); // qb = -qb
    mpf_sub(temp1, qb, sqrt_term); // temp1 = -qb - sqrt(qb*qb - 4*qa*qc)
    mpf_mul_ui(temp2, qa, 2); // temp2 = 2 * qa
    mpf_div(p->z, temp1, temp2); // p.z = temp1 / temp2
    mpf_clears(qa, qb, qc, temp1, temp2, sqrt_term, NULL);
}
void lift_exact_y(mpf_point *p,  mpf_t x,  mpf_t z) {
	mpf_t temp0;
	mpf_init(temp0);

	lift_exact(p, x, z);
	mpf_set(temp0, p->y); //swap y and z !!!
	mpf_set(p->y, p->z);
	mpf_set(p->z, temp0);

	mpf_clears(temp0, NULL);
}

void lift_exact_charts(mpf_point *p,  mpf_t a,  mpf_t b, int i) {
	mpf_t temp0;
	mpf_init(temp0);

	if(i == 1){
		lift_exact(p, a, b);
	}
	if(i == 2){
		lift_exact(p, a, b);
		mpf_set(temp0, p->y);
		mpf_set(p->y, p->z);
		mpf_set(p->z, temp0);
	}
	if(i == 3){
		lift_exact(p, a, b);
		mpf_set(temp0, p->z);
		mpf_set(p->z, p->y);
		mpf_set(p->y, p->x);
		mpf_set(p->x, temp0);
	}
	if(i == 4){
		lift_exact_under(p, a, b);
	}
	if(i == 5){
		lift_exact_under(p, a, b);
		mpf_set(temp0, p->y);
		mpf_set(p->y, p->z);
		mpf_set(p->z, temp0);
	}
	if(i == 6){
		lift_exact_under(p, a, b);
		mpf_set(temp0, p->z);
		mpf_set(p->z, p->y);
		mpf_set(p->y, p->x);
		mpf_set(p->x, temp0);
	}
	mpf_clears(temp0, NULL);
}


void search_near_exact(mpf_point *q, mpf_t epsilon, mpf_t delta, int N) {
    mpf_t x, y, temp;
    mpf_point p;
    int length = 2 * N;

    mpf_inits(x, y, temp, NULL);
    init_point(&p);

    // x = q.x - N * epsilon
    mpf_mul_ui(temp, epsilon, N);
    mpf_sub(x, q->x, temp);

    // y = q.y - N * epsilon
    mpf_sub(y, q->y, temp);
	//gmp_fprintf(stderr, " at (%.20Ff, %.20Ff)\n", x,y);
    for (int j = 0; j <= length; j++) {
        for (int i = 0; i <= length; i++) {
			//gmp_fprintf(stderr, " at (%.20Ff, %.20Ff)\n", x,y);
            if (!unreal_exact(x, y)) { //check unreal
                lift_exact(&p, x, y); //add lift
				//gmp_fprintf(stderr, " at (%.20Ff, %.20Ff, %.20Ff)\n", p.x, p.y, p.z);
				//gmp_fprintf(stderr, "tester at (%.20Ff, %.20Ff, %.20Ff)\n", p.x, p.y, p.z);
                if (is_recurrent_exact(p, delta)) {
					//fprintf(stderr, "made it!\n");
                    gmp_fprintf(stderr, "Success at (%.20Ff, %.20Ff, %.20Ff)\n", p.x, p.y, p.z);
                }
            }
            // x = x + epsilon
            mpf_add(x, x, epsilon);
        }
        // y = y + epsilon
        mpf_add(y, y, epsilon);

        // x = q.x - N * epsilon
        mpf_sub(x, q->x, temp);
    }

    mpf_clears(x, y, temp, NULL);
    //clear_point(&p);
}


//searches along line x = -z
void search_near_exact_line(mpf_point *q, mpf_t epsilon, mpf_t delta, int N) {
    mpf_t x, z, temp; //x_t & y_t used for substitution
	mpf_t a,b;
	mpf_inits(a,b, NULL);

    mpf_point p;
    int length = 2 * N;

    mpf_inits(x, z, temp, NULL);
    init_point(&p);

    // x = q.x - N * epsilon
    mpf_mul_ui(temp, epsilon, N);
    mpf_sub(x, q->x, temp);
    // y = q.y - N * epsilon
    mpf_add(z, q->z, temp);
    for (int j = 0; j <= length; j++) {
        if (!unreal_exact(x, z)) { //check unreal
            
			lift_exact_y(&p, x, z); //add lift
			projection_chart(p, a, b, 6);
			/*using f, not f_c
			if (is_recurrent_exact(p, delta)) {
                gmp_fprintf(stderr, "Success at (%.20Ff, %.20Ff, %.20Ff)\n", p.x, p.y, p.z);
            } */

			if (is_recurrent_exact_specific(a,b, delta)) {
                gmp_fprintf(stderr, "Success at (%.50Ff, %.50Ff, %.50Ff)\n", p.x, p.y, p.z);
            }
			projection_chart(p, a, b, 6);

			/*
			if (is_recurrent_exact(p, delta)) {			//this bit checks bottoms as well:
                gmp_fprintf(stderr, "Success at (%.20Ff, %.20Ff, %.20Ff)\n", p.x, p.y, p.z);
            } */

        }
            // x = x + epsilon
        mpf_add(x, x, epsilon);
		mpf_sub(z, z, epsilon);
	}
        // x = q.x - N * epsilon
    mpf_clears(x, z, temp, NULL);
    //clear_point(&p);
}

