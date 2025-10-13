//BEFORE BEGINNING: make sure to read the 'readme' arcs.c section

//Functions used from k3.c [aside from read-ins]:
	// subdivide_plane(), dist(), transform(), stereo_proj(), rescale(), fpath(), unreal()

//Functions used from ps.c:
	// ps_open(), ps_window(), ps_close(), ps_dot(), ps_dot_transparent(), ps_line()

//NOTES:
	//line width and dot side can be changed in ps.c 
	//throughout: 'path' = 'arc'
	//must have ps (postscript) to pdf installed, 'ps2pdf'




#include "k3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

//Plotting data
static double centerx = CENTERX, centery = CENTERY, radius = RADIUS;
static double length=LENGTH, sep = SEP;
static int drawtop = 1, drawbot = 1, verbose=VERBOSE, spin=0, unravel = 0, sphere = 0, example = -1, iterations = -1;

//Very large array to dynamically store paths
point ps[PSMAX];

//Initialize arrays to path position data; used in determine_arrays
int pos_array[PSMAX]; 
int above_array[PSMAX];

//Orbit_length is the order of the periodic point we are using
int orbit_length = 10; 

//Constants
int above = -100000;
int current_pos = -10000;

/*Function headers & descriptions*/

int connect_path(point z, point w, int k);
// connects z and w by a straightline and gradient-flows to surface. 
// Stores path f^k(p) in ps[]. Returns array length

int diagonal_path(double delta);
//draws {x = -z} \cap X(\R)

int diagonal_path_near(point q, double delta);
//draws x = -z near p


void fully_simplify_arrays(int *arr1, int *arr2, int *size);
//fully simplifies array_pos and array_height

void plot_orbit(point marked_orbit[], point z, int n);
//sets marked_orbit = {z, f z, f^2 z, ... , f^n z} and prints orbit points

void draw_background(point z, int n);
//sketches X(\R) by plotting many f-iterates of z

void draw_path(int n);
//plots ps[0] to ps[n] on X(\R) as a path. path is orange if on bottom of projection.

void determine_position(double *x_vals, double *y_vals, point p);
// If ps[t] is between ith and i+1st punctures, determine_position sets the universally declared variable current_pos equal to i+1

void determine_arrays(double *x_vals, double *y_vals, int *pos, int *height, int *m, point *marked_orbit, int *b);
//Encodes data of path stored in ps[] of length m
		// punctures in the plane are indexed from left to right. 
		// if the path begins at the ith puncture, then pos[0] = i and height[0] = -1000
		// when the path passes above/below a puncture, the index of the puncture is appended to pos[]
		// when the path passes above a puncture, 1 is appended to height[]
		// when the path passes below a puncture, 0 is appended to height[]
		// if the path ends at the jth puncture, then pos[b-1] = j and height[b-1] = -1000
// x_vals, y_vals are coordinates for the periodic orbit
// pos and height are the output arrays. 
// b is the length of pos and height -- it must start at 1, and will be increased throughout the program

int compare(const void *a, const void *b);
// p1 < p2 if p1.x < p2.x after (1) rescaling to unit sphere and (2) stereo projecting through 0,0,1 

bool simplify_arrays(int *arr1, int *arr2, int *size);
// if a simplification is possible, performs it and returns true. otherwise, returns false

void fully_simplify_arrays(int *arr1, int *arr2, int *size) ;
// fully simplifies array_pos and array_height

int main(int argc, char *argv[]) {
	int i,n;
	double a,b;

	//read-in values from paths.run
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

		case 't':
		drawbot = 0;
		break;

		case 'u': 
		unravel = 1;
		break;

		case 's': 
		sphere = 1;
		break;

		case 'v':
		if(argc <= (i+1)) usage();
		sscanf(argv[++i],"%d",&example);
		break;

		case 'w':
		if(argc <= (i+1)) usage();
		sscanf(argv[++i],"%d",&iterations);
		break;

		default:
		usage();
	}
	}


	//Enter periodic point here
point marked_point;
marked_point.x = -1.726895448754858426328854724474;  
marked_point.y = 1.041643093944314148360673792017;
marked_point.z = 1.726895448754858426328854724474;

	//Create (orbit_length-1) pos/height arrays to record paths 
	int **array_pos = malloc((orbit_length-1) * sizeof(int*));
	int **array_height = malloc((orbit_length -1) * sizeof(int*));
	int path_lengths[orbit_length-1];

	
	//Allocate memory for each array
	for (int i = 0; i < orbit_length-1; i++) {
        array_pos[i] = malloc(100000 * sizeof(int)); //100,000 might not be enough - make larger if causing problems!!!!
		array_height[i] = malloc(100000 * sizeof(int));
    }

	//Initialize path_lengths = 1
	for (int i = 0; i < orbit_length-1; i++) { 
        path_lengths[i] = 1;
    }

	//open postscript file, declare window
	ps_open(argc,argv);
	ps_window(centerx,centery,radius);
	
	//draw background. COMMENT out if unravel = 1
	if(unravel != 1){
		point starter = lift(.1,.1);
		draw_background(starter, 20000);
	}
	
	//store finite orbit in marked_orbit, and plot orbit
	point marked_orbit[orbit_length];
	plot_orbit(marked_orbit, marked_point, orbit_length);


	// Uncomment the following block to produce figure showing intersection of {z = -w} and f^5({z = -w})
			/*
			int well = diagonal_path_near(marked_orbit[0], .003);
			n = well;
			draw_path(n-2);

			well = diagonal_path_near(marked_orbit[5], 1e-5);
			n = well - 2;

			for (int i = 0; i <5; ++i){
				n=fpath(ps, n);
			}
			fprintf(stderr, "well is");
			draw_path(n-2);
			plot_orbit(marked_orbit, marked_point, orbit_length);

			ps_close(); 
			abort(); */

	//sort finite_orbit from left to right  after (1) rescale to unit sphere (2) stereo projection through 0,0,1 
	qsort(marked_orbit, orbit_length, sizeof(point), compare);
	
	//store x,y coordinates of finite orbit after (1) rescaling to unit sphere and (2) stereo projecting through 0,0,1 
	double x_vals[orbit_length];
	double y_vals[orbit_length];

	for(int i = 0; i < orbit_length; i++){
		x_vals[i] = transform(marked_orbit[i]).x;
		y_vals[i] = transform(marked_orbit[i]).y;
	}
	
	fprintf(stderr, "Finite orbit stored\n");
	
	int m = 1;
	for(int i = 0; i < orbit_length-1; i++){ 
		// connect_path(x,y,k) connects x and y via a path p and stores f^k(p) in ps[].
		// m is the length of the output
		
		m = connect_path(marked_orbit[i], marked_orbit[i+1], iterations);

		//plots the points of ps[] from 0 to m-1 in postscript
		if(example == -1){
			draw_path(m-1);
		}
		if(example == i){
			draw_path(m-1);
		}
		
				
		//determines over/under arrays from ps[] data
		determine_arrays(x_vals, y_vals, array_pos[i], array_height[i], &m, marked_orbit, &path_lengths[i]);		
	}
	
	fprintf(stderr, "Done determining arrays\n");
	

	//simplify arrays
	for(int i = 0; i<orbit_length-1; i++){ 
		fully_simplify_arrays(array_pos[i], array_height[i], &path_lengths[i]);
	}
	fprintf(stderr, "Done simplifying arrays\n");


	//print contents of arrays

	fprintf(stderr, "int path_lengths[orbit_length-1] = {");
	for (int j = 0; j < orbit_length-1; j++) {
		fprintf(stderr,"%d", path_lengths[j]);
		if (j < orbit_length - 2) {
			fprintf(stderr, ", ");
		}
	}
	fprintf(stderr,"};\n");

	fprintf(stderr, "int array_pos[orbit_length-1][10000] = {\n");
	for (int i = 0; i < orbit_length -1 ; i++) {
		fprintf(stderr,"    {");
		for (int j = 0; j < path_lengths[i]; j++) {
			fprintf(stderr,"%d", array_pos[i][j]);
			if (j < path_lengths[i] - 1) {
				fprintf(stderr,", ");
			}
		}
		fprintf(stderr,"}");
		if (i < orbit_length - 2) {
			fprintf(stderr,",");
		}
		fprintf(stderr,"\n");
	}
	fprintf(stderr,"};\n");

	fprintf(stderr,"int array_height[orbit_length-1][10000] = {\n");
	for (int i = 0; i < orbit_length -1 ; i++) {
		fprintf(stderr, "    {");
		for (int j = 0; j < path_lengths[i]; j++) {
			fprintf(stderr, "%d", array_height[i][j]);
			if (j < path_lengths[i] - 1) {
				fprintf(stderr, ", ");
			}
		}
		fprintf(stderr, "}");
		if (i < orbit_length - 2) {
			fprintf(stderr, ",");
		}
		fprintf(stderr, "\n");
	}
	fprintf(stderr, "};\n");

	//close postscript file
	ps_close();

	exit(0);
}


void usage() {
	fprintf(stderr,"Usage:  see ReadMe\n");
	fprintf(stderr,"Postscript file written to stdout\n");
	exit(1);
}

int connect_path(point z, point w, int k) {
	//subdivision happens in the plane
	int n;
	point z_plane = transform(z);
	point w_plane = transform(w);
	ps[0] = z;
	ps[1] = w;
	n = dist(ps[0], ps[1])/sep;
	ps[n] = ps[1]; 

	subdivide_plane(ps[0], ps[1], ps, n);

	//iterate each sub-path by f
	for (int i = 0; i < k; ++i){
		n=fpath(ps,n);
	}

	return n+1; //stuff Curt coded runs on size - 1. this returns size.
}


int diagonal_path(double delta) {
	int n = 0;
	
	double z = 0;
	double x = -3.0;
	double y = 3.0;
	point p;
	while(unreal(x,y)){
		x = x + delta;
		y = y - delta;
	}
	
	double swap;

	while(!unreal(x,y)){
		p = lift(x,y);
		swap = p.z;
		p.z = p.y;
		p.y = swap;
		ps[n] = p;
		x = x + delta;
		y = y - delta;
		n++;
		//lenth of ps ++
	}

	while(unreal(x,y)){
		x = x - delta;
		y = y + delta;
	}

	while(!unreal(x,y)){
		p = lift(x,y);
		p = iz(p);
		swap = p.z;
		p.z = p.y;
		p.y = swap;
		ps[n] = p;
		x = x - delta;
		y = y + delta;
		n++;
		//lenth of ps ++
	}
	ps[n] = ps[0];
	ps[n+1] = ps[1];
	return n+1; 
}

int diagonal_path_near(point q, double delta) {
	int n = 0;
	int l = 0;

	pprint(q);
	double z = 0;
	double x = q.x - 100*delta;
	double y = q.z + 100*delta;
	point p;
	while(unreal(x,y)){
		x = x + delta;
		y = y - delta;
	}
	
	double swap;
	point temp = iy(q);
	while((!unreal(x,y)) && l < 201){
		p = lift(x,y);
		if(q.y <  temp.y){
			p = iz(p);
		}
		swap = p.z;
		p.z = p.y;
		p.y = swap;
		
		ps[n] = p;
		x = x + delta;
		y = y - delta;
		n++;
		l++;
		//lenth of ps ++
	}
	//draw_path(n); 
	return n+1; //differs from Curt's code
}

void plot_orbit(point marked_orbit[], point z, int n){
	marked_orbit[0] = z;
	pprint(z);
	for (int i = 0; i<n; i++){
		marked_orbit[i] = z;
		ps_dot(z, unravel, sphere);
		pprint(z);
		z = f(z);
	}
}

void draw_background(point z, int n){
	for (int i = 0; i<n; i++){
		ps_dot_transparent(z, sphere);
		z = f(z);
	}
}

void draw_path(int n){
	int i;
	point p, q;

	for(i=0; i<n; i++)
	{	p = ps[i]; q = ps[i+1];
		//if(rotate) {p=rotate(p); q=rotate(q);} 
		if(drawtop) ps_line(p,q, unravel, sphere);
		//if(drawbot && !top(p)) ps_line_color(p,q, unravel); 
	}
}


void determine_arrays(double *x_vals, double *y_vals, int *pos, int *height, int *m, point *marked_orbit, int *b) {	

	pos_array[0] = -1000; //will be changed to starting position shortly
	above_array[0] = -1000; //path_origin //above_array is used as an intermediate.


	for(int t = 1; t < *m; t++){ 
		determine_position(x_vals, y_vals, ps[t]); 
		
		//runs on first iteration:
		if(*b==1){
			pos_array[*b] = current_pos; //sets 
			above_array[*b] = 0; //this does not matter

			//increase b, skip the rest of the for loop
			(*b)++;
			continue; 		
		}

		//records previous array position
		int last = pos_array[(*b)-1];

		//checks if path is moving left, i.e. current_pos < pos_array[(*b)-1]
		if(current_pos - last == -1){
			//update pos_array
			pos_array[*b] = current_pos;

			//determing if path is above or below puncture
			if(transform(ps[t]).y > y_vals[current_pos]){ // change to: && transform(ps[t-1]).y > y_vals[current_pos]
				above_array[*b] = 1;
			} else {
				above_array[*b] = 0;
			}

			//increase b, skip the rest of the for loop
			(*b)++;
			continue; 
		}

		//checks if path is moving right, i.e. current_pos > pos_array[(*b)-1]
		if(current_pos - pos_array[(*b)-1] == 1){ 
			//update pos_array
			pos_array[*b] = current_pos;

			//determing if path is above or below puncture
			if(transform(ps[t]).y > y_vals[pos_array[(*b)-1]]){
				above_array[*b] = 1;
			} else {
				above_array[*b] = 0;
			}
			
			//increase b, skip the rest of the for loop
			(*b)++;
			continue; 
		}

		// path is moving right,  but more than one puncture lies between ps[t-1] and ps[t] 
		// in other words, current_pos - last > 1
		if (last < current_pos) {
			fprintf(stderr, "accounting for skipped punctures\n");

			//iterate through skipped punctures, and make sure that ps[t-1] and ps[t] both above or both below skipped puncture
			//throws error if ps[t-1] and ps[t] on different sides of a skipped puncture

			for (int i = last; i <= current_pos-1; i++) { 
				//update pos_array
				pos_array[*b] = i+1; 
				
				if((transform(ps[t-1]).y > y_vals[i]) && (transform(ps[t]).y > y_vals[i])){
					above_array[*b] = 1;
				} else if((transform(ps[t-1]).y < y_vals[i]) && (transform(ps[t]).y < y_vals[i])){
					above_array[*b] = 0;
				} else{
					fprintf(stderr, "here issue with over/under \n");
					fprintf(stderr, "current_pos is (%d,%d, %d) \n", pos_array[*b], above_array[*b], *b);
					abort();
				}
				//increase b
				(*b)++;
			}
			//skip rest of for loop
			continue; 
    	}

		//doing the same in the other direction:
		if (pos_array[(*b)-1] > current_pos) { 
			for (int i = last - 1; i >= current_pos; i--) {
				pos_array[*b] = i;
				if((transform(ps[t-1]).y > y_vals[i]) && (transform(ps[t]).y > y_vals[i])){
					above_array[*b] = 1;
				} else if((transform(ps[t-1]).y < y_vals[i]) && (transform(ps[t]).y < y_vals[i])){
					above_array[*b] = 0;
				} else{
					fprintf(stderr, "issue with over/under");
					fprintf(stderr, "current_pos is (%d,%d, %d) \n", pos_array[*b], above_array[*b], *b);
					abort();
				}
				(*b)++;
			}
			continue; 
		}
	} //for loop ended!

	//determine position of starting point. print error if fails
	int test = 0;
	for(int i  = 0; i < orbit_length; i++){
		if(dist(transform(ps[0]), transform(marked_orbit[i]))<1e-2){
			pos[0] = i;
			test = 1;
			break;
		}
	}
	if(!test){
		fprintf(stderr, "error in starting point assignment\n");
		pprint(ps[0]);
		abort();
	}

	// determine position of ending point. print error if fails
	test = 0;
	height[(*b)-1] = -1000;
	for(int i  = 0; i < orbit_length; i++){
		if(dist(transform(ps[(*m)-1]), transform(marked_orbit[i]))<1e-2){
			pos[(*b)-1] = i;
			test = 1;
			break;
		}
	}
	if(!test){
		fprintf(stderr, "error in endpoint assignment\n");
		pprint(ps[*m-1]);
		abort();
	}
	//determing position inbetween

	//translate 'pos_array' to 'pos'
	//translate 'above_array' to 'height'
	height[0] = -1000;
	for(int i = 1; i<(*b)-1; i++){
		if((pos_array[i] < pos_array[i+1])){
			pos[i] = pos_array[i];
			if(above_array[i+1]){
				height[i]=1;
			} else {
				height[i]=0;
			}
		}
		if((pos_array[i] > pos_array[i+1])){
			pos[i] = pos_array[i+1];
			if(above_array[i+1]){
				height[i]=1;
			} else {
				height[i]=0;
			}
		}
	}
}


void determine_position(double *x_vals, double *y_vals, point p){
	above = 0;
	double x = transform(p).x;
	double y = transform(p).y;
	for (int l = 0; l < orbit_length-1; l++){
		if((x_vals[l] < x) && (x <= x_vals[l+1])) {
			current_pos = l+1;
			if(y > y_vals[l]){
				above = 1;
			}
		}
	}
	if(x > x_vals[orbit_length - 1]){
		current_pos = orbit_length;
		if(y > y_vals[orbit_length - 1]){
			above = 1;
		}
	}
	if(x <= x_vals[0]) {
		current_pos = 0;
		if(y > y_vals[0]){
			above = 1;
		}
	}
}


int compare(const void *a, const void *b) {
    double diff = transform(*(point*)a).x - transform(*(point*)b).x;
    if (diff < 0) return -1;
    else if (diff > 0) return 1;
    else return 0;
}


bool simplify_arrays(int *arr1, int *arr2, int *size) {
    bool simplified = false;
	if ((arr1)[0] == (arr1)[1]){
			for (int j = 1; j < *size - 1; j++) {
                (arr1)[j] = (arr1)[j + 1];
                (arr2)[j] = (arr2)[j + 1];
            }
            (*size) -= 1;
            simplified = true;
			return simplified;
	}

	if ((arr1)[*size - 1] == (arr1)[*size - 2]){
			arr2[*size - 2] = -1000;
            (*size) -= 1;
            simplified = true;
			return simplified;
	}

    for (int i = 0; i < *size - 2; i++) {
        if ((arr1)[i] == (arr1)[i+1] && (arr2)[i] == (arr2)[i+1]) {
            //fprintf(stderr, "made it");
			// Remove the ith and (i+1)th elements
            for (int j = i; j < *size - 2; j++) {
                (arr1)[j] = (arr1)[j + 2];
                (arr2)[j] = (arr2)[j + 2];
            }
            *size -= 2;
            simplified = true;
            return simplified;;
        }
    }
    return simplified;
}

void fully_simplify_arrays(int *arr1, int *arr2, int *size) {
    bool simplified;
    do {
        simplified = simplify_arrays(arr1, arr2, size);
    } while (simplified && *size > 2);
}
