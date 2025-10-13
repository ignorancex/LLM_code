
/* Automorphisms of a K3 surface in P^1 x P^1 x P^1	*/
/* (1+x^2)(1+y^2)(1+z^2) + Axyz = B */

#include "k3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


static double a=A, b=B, sep=SEP;                

// Very large array
point qs[PSMAX];


// Prints a point
void pprint(point p){
	fprintf(stderr, "(%.10lf, %.10lf, %.10lf)\n", p.x, p.y, p.z);
}
    

// Set p.z from {x,y} so p lies on K3 
point lift(x,y)
	double x, y;
{
	double qa, qb, qc;
	point p;
	qa = (1+x*x)*(1+y*y);
	qb = a*x*y;
	qc = qa - b;
	p.x = x;
	p.y = y;
	p.z = (-qb + sqrt(qb*qb - 4*qa*qc))/(2*qa);
	return(p);
}

// check if (x,y) lifts to a real point 
int unreal(x,y)
	double x,y;
{
	double qa, qb, qc;

	qa = (1+x*x)*(1+y*y);
	qb = a*x*y;
	qc = qa - b;
	return(qb*qb - 4*qa*qc < 0);
}


// Involution of z, fixing x and y 
// z -> (-axy)/((1+x^2)(1+y^2)) - z 
double involute(x,y,z)
	double x,y,z;
{
	z = (-a*x*y)/((1+x*x)*(1+y*y)) - z;
	return(z);
}

// involutions 
point ix(p)
	point p;
{
	p.x = involute(p.z,p.y,p.x);
	return(p);
}

point iy(p)
	point p;
{
	p.y = involute(p.x,p.z,p.y);
	return(p);
}

point iz(p)
	point p;
{
	p.z = involute(p.x,p.y,p.z);
	return(p);
}



// Symmetry (x,y,z) -> (-z,y,-x) 
point sym(p)
	point p;
{
	double temp = p.z;
	p.z = -p.x;
	p.x = -temp;
	return(p);
}

// f = iz iy ix
point f(p)
	point p;
{
	return iz(iy(ix(p))); 
}
	

// Change parameters for K3 surface 
void setk3(newa,newb)
	double newa, newb;
{
	a = newa;
	b = newb;
}

void setsep(double s)
{
	sep = s;
}

// Test if real point is on upper sheet (max z) 
// Just check if invz makes the z coordinate smaller 
int top(p)
	point p;
{
	point q;

	q = iz(p);
	return(q.z <= p.z);
}
int topx(p)
	point p;
{
	point q;

	q = ix(p);
	return(q.x <= p.x);
}
int topy(p)
	point p;
{
	point q;

	q = iy(p);
	return(q.y <= p.y);
}

// Defining equation of surface 
double surf(p)
	point p;
{
	double s, x, y, z;

	x=p.x; y=p.y; z=p.z;
	s = (1+x*x)*(1+y*y)*(1+z*z)+a*x*y*z-b;
	return(s);
}


// Gradient/|Grad|^2 of defining equation of surface
point grad(p)
	point p;
{
	point v;
	double g,x,y,z;

	x=p.x; y=p.y; z=p.z;

	v.x = 2*x*(1+y*y)*(1+z*z)+a*y*z;
	v.y = 2*y*(1+z*z)*(1+x*x)+a*z*x;
	v.z = 2*z*(1+x*x)*(1+y*y)+a*x*y;
	g=norm(v);
	v.x = v.x/g;
	v.y = v.y/g;
	v.z = v.z/g;
	return(v);
}

// Follow gradient from p to point on surface 
point newton(p)
	point p;
{
	point v;
	double s;
	int i;

	for(i=0; i<NEWTONMAX; i++)
	{	s = surf(p);
		if(s < NEWTONEPS && (-s) < NEWTONEPS) return(p);
		v = grad(p);
		p.x = p.x - s*v.x*NEWTONCAUTION;
		p.y = p.y - s*v.y*NEWTONCAUTION;
		p.z = p.z - s*v.z*NEWTONCAUTION;
	}
	fprintf(stderr,"Error:  Newton iteration failed after 100 steps\n");
	exit(1);
}

// Follow radial segments from unit sphere to point on surface 
// WARNING: this has caused jagged edges in the past
point newton_plane(p)
	point p;
{
	point v;
	double s;
	int i;

	for(i=0; i<100000; i++) 
	{	s = surf(p);
		if(s < .00001 && (-s) < .00001) return(p); 
		v = rescale(p);
		p.x = p.x - s*v.x*.01; 
		p.y = p.y - s*v.y*.01;
		p.z = p.z - s*v.z*.01;
	}
	fprintf(stderr,"Error:  Newton iteration failed after 100000 steps\n");
	exit(1);
}

// Subdivide a segment into n segments WITH NEWTON ITERATION
void subdivide(point p,point q,point ps[],int n)
{
	int i;

	for(i=1; i<n; i++) 
	{	ps[i].x = (i*q.x + (n-i)*p.x)/n;
	 	ps[i].y = (i*q.y + (n-i)*p.y)/n;
	 	ps[i].z = (i*q.z + (n-i)*p.z)/n;
		ps[i] = newton(ps[i]);
	}
}

// Subdivide a segment into n segments USING TRANSFORMATION TO PLANE
void subdivide_plane(point p,point q,point ps[],int n)
{
	int i;
	point p_plane = transform(p);
	point q_plane = transform(q);
	point temp;
	for(i=1; i<n; i++) 
	{	temp.x = (i*q_plane.x + (n-i)*p_plane.x)/n;
	 	temp.y = (i*q_plane.y + (n-i)*p_plane.y)/n;
	 	temp.z = (i*q_plane.z + (n-i)*p_plane.z)/n;
		ps[i] = inv_transform(temp);
	}
}

// Norm of a point 
double norm(p)
	point p;
{
	return(p.x*p.x + p.y*p.y + p.z*p.z);
}

// Distance between points 
double dist(p,q)
	point p, q;
{
	p.x=p.x-q.x; 
	p.y=p.y-q.y; 
	p.z=p.z-q.z;
	return(sqrt(norm(p)));
}

// Rotate point 
point rotate(p)
	point p;
{
	point q;
	double t;
	q.x = p.y;
	q.y = p.z;
	q.z = p.x;
	return(q);
}

// Map segment
// Subdivides and returns new length 
int fseg(p,q,ps)
	point p, q, ps[];
{
	int i, n;

	ps[0]=f(p);
	ps[1]=f(q);
	n=dist(ps[0],ps[1])/sep;
	
	if(n <= 1) 
		return(1);
	if(n >= SUBMAX) 
		suberr(n);
	ps[n]=ps[1];
	subdivide(p,q,ps,n);
	for(i=1; i<n; i++) ps[i] = f(ps[i]);
	return(n);
}

// Map path 
// n = number of segments; returns new n 
int fpath(point ps[],int np)
{
	point rs[SUBMAX];
	int i, j, nq, nr;

	qs[0]=f(ps[0]);
	nq=0;
	for(i=0; i<np; i++)
	{	nr = fseg(ps[i], ps[i+1], rs); /*output fseg as rs array, return lenth of array*/
		if(nq+nr >= PSMAX) 
		{	patherr();
			break;
		}
		for(j=1; j<=nr; j++) /*append nr-array onto end of qs*/
		{	nq++;
			qs[nq]=rs[j];
		}
	}
	ps[0] = qs[0];
	np = 0;
	for(i=1; i<=nq; i++)
		if(dist(ps[np],qs[i]) >= sep/2.0)
		{	np++;
			ps[np]=qs[i];
		}
	if(np == 0)
	{	ps[1]=qs[nq]; 
		np=1;
	}
	return(np);
}

// ERROR:  Too many subdivisions 
void suberr(n)
	int n;
{
	fprintf(stderr,
		"Error: %d subdivs requested, %d avail\n",
		n,SUBMAX);
	exit(1);
}

// ERROR:  Path too long 
void patherr()
{
	fprintf(stderr,
		"Error: max path length of %d exceeded\n",
		PSMAX);
}

// rescales to unit sphere
point rescale(point p){
	double n = sqrt(norm(p));
	p.x = p.x/n;
	p.y = p.y/n;
	p.z = p.z/n;
	return p;
}

//stereogrphic projection from (0,0,1)
point stereo_proj(point p){ 
	point q;
	q.x = p.x/(1-p.z);
	q.y = p.y/(1-p.z);
	q.z = 0;
	return q;
}


//inverse stereographic projection
point inv_proj(point p){
	point q;
	q.x = 2*p.x/(1+(p.x*p.x) + (p.y*p.y));
	q.y = 2*p.y/(1+(p.x*p.x) + (p.y*p.y));
	q.z = (-1 + (p.x*p.x) + (p.y*p.y))/(1+(p.x*p.x) + (p.y*p.y));
	return q;
}


//from surface to x-y plane
point transform(point p){
	return stereo_proj(rescale(p));
}

//from x-y plane to surface
point inv_transform(point p){
	return newton_plane(inv_proj(p));
}


//derivatives of the involutions
void Dix(point p, double **matrix) {
	matrix[0][0] = -1;
    matrix[0][1] = -a*p.z*(1-p.y*p.y)/((1 + p.z*p.z)*(1 + p.y*p.y)*(1 + p.y*p.y));
	matrix[0][2] = -a*p.y*(1-p.z*p.z)/((1 + p.y*p.y)*(1 + p.z*p.z)*(1 + p.z*p.z));
    matrix[1][0] = 0;
    matrix[1][1] = 1;
	matrix[1][2] = 0;
	matrix[2][0] = 0;
    matrix[2][1] = 0;
	matrix[2][2] = 1;
}

void Diy(point p, double **matrix) {
	matrix[0][0] = 1;
    matrix[0][1] = 0;
	matrix[0][2] = 0;
    matrix[1][0] = -a*p.z*(1-p.x*p.x)/((1 + p.z*p.z)*(1 + p.x*p.x)*(1 + p.x*p.x));
    matrix[1][1] = -1;
	matrix[1][2] = -a*p.x*(1-p.z*p.z)/((1 + p.x*p.x)*(1 + p.z*p.z)*(1 + p.z*p.z));
	matrix[2][0] = 0;
    matrix[2][1] = 0;
	matrix[2][2] = 1;
}

void Diz(point p, double **matrix) {
	matrix[0][0] = 1;
    matrix[0][1] = 0;
	matrix[0][2] = 0;
    matrix[1][0] = 0;
    matrix[1][1] = 1;
	matrix[1][2] = 0;
	matrix[2][0] = -a*p.y*(1-p.x*p.x)/((1 + p.y*p.y)*(1 + p.x*p.x)*(1 + p.x*p.x)); //this has an arror you CUNT
    matrix[2][1] = -a*p.x*(1-p.y*p.y)/((1 + p.x*p.x)*(1 + p.y*p.y)*(1 + p.y*p.y));
	matrix[2][2] = -1;
}

point vtan(point p) {
	point q;
	q.x = 2*p.z*(1+p.x*p.x)*(1+p.y*p.y) + a*p.x*p.y;
    q.y = 0;
	q.z = -2*p.x*(1 + p.y*p.y)*(1+p.z*p.z) - a*p.y*p.z;
	return q;
}


//matrix multiplication
void multiplyMatrices(double **firstMatrix, double **secondMatrix, double **result, int ROWS1, int COLS1, int ROWS2, int COLS2) {
    for (int i = 0; i < ROWS1; ++i) {
        for (int j = 0; j < COLS2; ++j) {
            result[i][j] = 0;
            for (int k = 0; k < COLS1; ++k) {
                result[i][j] += firstMatrix[i][k] * secondMatrix[k][j];
            }
        }
    }
}

void multiplyMatricesAux(double **firstMatrix, double **secondMatrix, double **result, int ROWS1, int COLS1, int ROWS2, int COLS2) {
	for (int i = 0; i < ROWS1; ++i) {
        for (int j = 0; j < COLS2; ++j) {
            result[i][j] = 0;
            for (int k = 0; k < COLS1; ++k) {
                result[i][j] += firstMatrix[i][k] * secondMatrix[k][j];
            }
        }
    }
	for (int i = 0; i < ROWS1; ++i) {
        for (int j = 0; j < COLS2; ++j) {
            secondMatrix[i][j] = result[i][j];
        }
    }
}

void printMatrix(double **matrix, int ROWS1, int COLS1) {
    fprintf(stderr, "{");
	for (int i = 0; i < ROWS1; ++i) {
        fprintf(stderr, "{");
		for (int j = 0; j < COLS1; ++j) {
            fprintf(stderr, "%.15f", matrix[i][j]);
			if(j < COLS1-1){
				fprintf(stderr, ",");
			} 
        }
		
		if(i==ROWS1-1){
			fprintf(stderr, "}}\n");
		} else{
			fprintf(stderr, "},\n");
		}
    }
}

//square of L2-norm of a matrix 
double mat_norm(double **matrix, int ROWS1, int COLS1)
{
	double x = 0;
	for (int i = 0; i < ROWS1; ++i) {
        for (int j = 0; j < COLS1; ++j) {
            x += (matrix[i][j]) * (matrix[i][j]);
        }
    }
	return x;
}