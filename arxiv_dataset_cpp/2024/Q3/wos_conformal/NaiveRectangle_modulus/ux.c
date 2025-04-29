#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <string.h>

/*
typedef struct {
	double x;
	double y;
	double result;
	unsigned int num_shots;
} point_t, *Point;
*/

#define HEIGHT 1.0
#define PI 3.14159265358979323846
#define x_center 0.5 
#define y_center (HEIGHT / 2) 
#define STEP 0.1  //Step size of the five-point formula. To be passed as in function dFivePoint as h
#define ANGLE 0.0 
#define NUM_POINTS 1000 //Number of node in each list of fm2h, fmh, fph, fp2h
#define NUM_SHOTS 1.0E6 /* Total number of shots for each of fm2h, fmh, fph, fp2h. Must be a float number */
#define EPSILON 1e-6
//#define TEST
#define length(x, y) sqrt((x)*(x)+(y)*(y))
//inline Point create_point(double, double, double);
//inline Point copy_point(Point);
//inline Point* get_points(int*);
//inline void destroy_points(Point*, int);
//inline void bounce_inside(Point);
inline unsigned int walk_on_spheres(double x, double y, unsigned int num_shots, double epsilon); 
inline double dFivePoint(double fm2h, double fmh, double fph, double fp2h, double h);
double calculateAverage(unsigned int array[], int size);
inline void print_fm2h_results(unsigned int* list, int num_of_pts, char* str);
inline void print_fmh_results(unsigned int* list, int num_of_pts, char* str);
inline void print_fph_results(unsigned int* list, int num_of_pts, char* str);
inline void print_fp2h_results(unsigned int* list, int num_of_pts, char* str);

int main(int argc, char** argv) {
	if (argc == 1) {
		fprintf(stderr, "Usage: ./prog num_procs proc\n");
		return -1;
	}
	time_t rawtime;
  	struct tm * timeinfo;
  	time( &rawtime );
  	timeinfo = localtime( &rawtime );
  	printf( "Current local time and date: %s", asctime (timeinfo) );
	printf( "Every point has %lf shots\n", NUM_SHOTS);
	//Get 4 lists of points for fm2h, fmh, fph, fp2h
	unsigned int fm2h_list[NUM_POINTS] = {0};
	unsigned int fmh_list[NUM_POINTS] = {0};
	unsigned int fph_list[NUM_POINTS] = {0};
	unsigned int fp2h_list[NUM_POINTS] = {0};
	srand(time(NULL));
//	omp_set_num_threads(100);
	int total_section = atoi(argv[1]);
	int section = atoi(argv[2]);
	int part = NUM_POINTS/(total_section);
	int start = section * part;
	int end = start + part;
	end = end > NUM_POINTS ? NUM_POINTS : end;

	double x1 = x_center + (-2) * STEP * cos(ANGLE) ;
	double y1 = y_center + (-2) * STEP * sin(ANGLE) ;
	double x2 = x_center + (-1) * STEP * cos(ANGLE) ;
	double y2 = y_center + (-1) * STEP * sin(ANGLE) ;
	double x3 = x_center + 1 * STEP * cos(ANGLE) ;
	double y3 = y_center + 1 * STEP * sin(ANGLE) ;
	double x4 = x_center + 2 * STEP * cos(ANGLE) ;
	double y4 = y_center + 2 * STEP * sin(ANGLE) ;
	#pragma omp parallel for schedule(dynamic)
	for (int i = start; i < end; i++) {
		//DO hard work
		fm2h_list[i] = walk_on_spheres(x1, y1, NUM_SHOTS, EPSILON);
		fmh_list[i] = walk_on_spheres(x2, y2, NUM_SHOTS, EPSILON);
		fph_list[i] = walk_on_spheres(x3, y3, NUM_SHOTS, EPSILON);
		fp2h_list[i] = walk_on_spheres(x4, y4, NUM_SHOTS, EPSILON);
	}

	print_fm2h_results(fm2h_list+start, end-start, argv[2]);
	print_fmh_results(fmh_list+start, end-start, argv[2]);
	print_fph_results(fph_list+start, end-start, argv[2]);
	print_fp2h_results(fp2h_list+start, end-start, argv[2]);
	time( &rawtime );
        timeinfo = localtime( &rawtime );
	//printf("Totally %d points, fails %d points.\n", end - start, fail_count);
	return 0;
}



inline unsigned int walk_on_spheres(double x1, double y1, unsigned int num_shots, double epsilon){
        double wx = x1;
        double wy = y1;
        unsigned int mark = 0;
        unsigned int j = 0;
        while (j < num_shots) {
                if (wy < 0) {
                        wy = -wy;
                }else if (wy > 2 * HEIGHT){
                        wy = 2 * HEIGHT - wy;
                }
                double r = (1-wx) > wx ? wx : (1-wx);
                if(r<EPSILON){
                        if(wx > 0.5) mark++;
                        wx = x1;                                                                                                                           wy = y1;
                        j++;
                }
                double u1 = ((double)(rand()))*2*PI/(RAND_MAX);
                double z1 = sin(u1);
                double z2 = cos(u1);
                wx += r * z1;
                wy += r * z2;
        }
        return mark;
}


inline double dFivePoint(double fm2h, double fmh, double fph, double fp2h, double h){
	return (fm2h - 8*fmh + 8*fph - fp2h)/(12 * h);	
}

// Function to calculate the average of an array
double calculateAverage(unsigned int array[], int size) {
    int sum = 0;

    for (int i = 0; i < size; i++) {
        sum += array[i];
    }

    return (double)sum / size;
}

inline void print_fm2h_results(unsigned int* list, int num_of_pts, char* str) {
	char name[15] = "fm2h_";
	strcat(name, str);
	strcat(name, ".txt");
	FILE* fd = fopen(name, "w");
	for (int i = 0; i < num_of_pts; i++) {	
		fprintf(fd, "%u\n", *(list+i) );
	}
	fclose(fd);
}
inline void print_fmh_results(unsigned int* list, int num_of_pts, char* str) {
	char name[15] = "fmh_";
	strcat(name, str);
	strcat(name, ".txt");
	FILE* fd = fopen(name, "w");
	for (int i = 0; i < num_of_pts; i++) {	
		fprintf(fd, "%u\n", *(list+i) );
	}
	fclose(fd);
}
inline void print_fph_results(unsigned int* list, int num_of_pts, char* str) {
	char name[15] = "fph_";
	strcat(name, str);
	strcat(name, ".txt");
	FILE* fd = fopen(name, "w");
	for (int i = 0; i < num_of_pts; i++) {	
		fprintf(fd, "%u\n", *(list+i) );
	}
	fclose(fd);
}
inline void print_fp2h_results(unsigned int* list, int num_of_pts, char* str) {
	char name[15] = "fp2h_";
	strcat(name, str);
	strcat(name, ".txt");
	FILE* fd = fopen(name, "w");
	for (int i = 0; i < num_of_pts; i++) {	
		fprintf(fd, "%u\n", *(list+i) );
	}
	fclose(fd);
}
/*
inline Point copy_point(Point pt) {
	Point new = create_point(pt->x, pt->y);
	new->result = pt->result;
	return new;
}

inline Point create_point(double x, double y, double num_shots) {
	Point new = malloc(sizeof(*new));
	new->x = x;
	new->y = y;
	new->result = 0;
	new->num_shots = num_shots; 
	return new;
}
	
inline Point* get_points(unsigned int num_of_pts) {
	Point* list = malloc(sizeof(*list) * num_of_pts);
	if (!list) {
        	perror("Failed to allocate memory");
        	exit(EXIT_FAILURE);
    	}
	for (int i = 0; i < num_of_pts; i++){
		list[i] = create_point(x_center, y_center, TOTAL_NUM_SHOTS/NUM_POINTS); 
	}
	return list;
}

inline void destroy_points(Point* list, int num_of_pts) {
	for (int i = 0; i < num_of_pts; i++)
		free(list[i]);
	free(list);
}


inline double find_min_radius(double x, double y) {
    if (y <= 1) {
        double dist1 = y;
        double dist2 = length(x - 2, y - 1);
        double dist3 = length(x - 4, y - 1);
        return (dist1 < dist2) ? ((dist1 < dist3) ? dist1 : dist3) : ((dist2 < dist3) ? dist2 : dist3);
    } else {
        if (x < 2) {
            return (2 - y < 2 - x) ? (2 - y) : (2 - x);
        } else if (x > 4) {
            return (2 - y < x - 4) ? (2 - y) : (x - 4);
        } else {
            return (x - 2 < 4 - x) ? (x - 2) : (4 - x);
        }
    }
}


inline void move_to_next(Point pt, double r) {
	double u1 = ((double)(rand()))*2*PI/(RAND_MAX);
	double z1 = sin(u1);
	double z2 = cos(u1);
	pt->x += r * z1;
	pt->y += r * z2;
}


inline void bounce_inside(Point pt) {
        if (pt->x < 0) pt->x = -(pt->x);  // Out of Neumann boundary x=0, 0<=y<=2, Reflection
	else if (pt->x > 6) pt->x = pt->x - 6;  // Out of Neumann boundary x=6, 0<=y<=2, Reflection and translation
        else if (2 < pt->x && pt->x < 4 && pt->y > 1) pt->y = 2 - pt->y;  // Out of Neumann boundary y=1, 2<=x<=4, Reflection
        // printf("Move to %.6lf, %.6lf, %.6lf\n", pt->x, pt->y, pt->z);
}
*/

