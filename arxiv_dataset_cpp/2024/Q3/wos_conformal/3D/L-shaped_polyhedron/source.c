#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <string.h>

typedef enum {
	one, two, three
} domainLabel;

typedef struct {
	double x;
	double y;
	double z;
	double result;
	domainLabel label;
} point_t, *Point;

/* slide the volumn to select points  */
#define X_SLIDES 12
#define Y_SLIDES 10
#define Z_SLIDES 20
#define NUM_POINTS X_SLIDES * Y_SLIDES * Z_SLIDES
#define NUM_SHOTS 100000.0 /* Must be a float number */
#define FAILURE_RADIUS 1e-5 /* When radius less than this, it fails, we try again */
#define FAILURE_BOUND 500 /* Used for kill point that failure on this number */
#define PI 3.14159
#define PRECISE 1e-6

#define write_result(point, mark) \
	do { \
		if (mark == -1) point->result = -1; \
		else point->result = mark/NUM_SHOTS; \
	} while(0)

#define length(x, y) sqrt((x)*(x)+(y)*(y))
inline Point create_point(double, double, double);
inline Point copy_point(Point);
inline Point* get_points(int*);
inline void destroy_points(Point*, int);
inline void print_points(Point*, int, char*);
inline double find_min_radius(Point pt);
inline void move_to_next(Point, double);
inline void bounce_inside(Point);
inline bool isOnPlanes(Point, int*);

#ifdef TEST
#include "test.c"
#else
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
	//Get a list of points
	int num_of_pts;
	Point* list = get_points(&num_of_pts);
	srand(time(NULL));
//	omp_set_num_threads(100);
	int total_section = atoi(argv[1]);
	int section = atoi(argv[2]);
	int part = num_of_pts/(total_section-1);
	int start = section * part;
	int end = start + part;
	end = end>num_of_pts?num_of_pts:end;
	printf("This job is %d to %d\n", start+1, end);
//	for (int i = start; i < end; i++) {
//		printf("%lf, %lf, %lf\n", list[i]->x, list[i]->y, list[i]->z);
//	}
	#pragma omp parallel for schedule(dynamic)
	for (int i = start; i < end; i++) {
		//DO hard work
		int mark = 0;
		for (int j = 0; j < NUM_SHOTS; j++) {
			Point copy = copy_point(list[i]);
			//fprintf(stderr, "Start %d:%d shots... \n", i, j);
			while(1) {
				if (length(1 - copy->x, 1 - copy->z) < FAILURE_RADIUS) {
			//		fprintf(stderr, "Fail %lf, %lf, %lf : %lf, %lf, %lf\n", 
			//			list[i]->x, list[i]->y, list[i]->z,
			//			copy->x, copy->y, copy->z);
					copy->x = list[i]->x;
					copy->y = list[i]->y;
					copy->z = list[i]->z;
					copy->label = list[i]->label;
				}
				double r = find_min_radius(copy);
				move_to_next(copy, r);
				bounce_inside(copy);
				if (isOnPlanes(copy, &mark)) break;
			}
			free(copy);
		}
		write_result(list[i], mark);
	}
	print_points(list+start, end-start, argv[2]);
	destroy_points(list, num_of_pts);
	time( &rawtime );
        timeinfo = localtime( &rawtime );
	//printf("Totally %d points, fails %d points.\n", end - start, fail_count);
        printf( "Finish time and date: %s\n", asctime (timeinfo) );
	return 0;
}
#endif /* Switch between test and release mode */

inline Point copy_point(Point pt) {
	Point new = create_point(pt->x, pt->y, pt->z);
	new->result = pt->result;
	return new;
}

inline Point create_point(double x, double y, double z) {
	Point new = malloc(sizeof(*new));
	new->x = x;
	new->y = y;
	new->z = z;
	new->result = 0;
	if (new->x >= 1) new->label = three;
	else if (new->z >= 1) new->label = one;
	else new->label = two;
	return new;
}
	
inline Point* get_points(int* num_of_pts) {
	int counter = 0;
	double x_d = 3.0/X_SLIDES;
	double y_d = 1.0/Y_SLIDES;
	double z_d = 2.0/Z_SLIDES;
	Point* list = malloc(sizeof(*list) * NUM_POINTS);
	for (int i = 0; i < NUM_POINTS; i++) list[i] = NULL;
	for (int i = 0; i <= X_SLIDES; i++) {
		double x = i * x_d;
		for (int j = 0; j <= Y_SLIDES; j++) {
			double y = j * y_d;
			for (int k = 0; k <= Z_SLIDES; k++) {
				double z = k*z_d;
				//check valid
				if (x > 1 && z > 1) continue; //Not in box
				else if (length(1-x, 1-z)<1e-3) continue;
				list[counter++] = create_point(x, y, z);
			}
		}
	}
#ifndef TEST
	printf("Generate %d points.\n", counter);
#endif
	*num_of_pts = counter;
	return list;
}

inline void destroy_points(Point* list, int num_of_pts) {
	for (int i = 0; i < num_of_pts; i++)
		free(list[i]);
	free(list);
}

inline void print_points(Point* list, int num_of_pts, char* str) {
	char name[15] = "points";
	strcat(name, str);
	strcat(name, ".txt");
	FILE* fd = fopen(name, "w");
	for (int i = 0; i < num_of_pts; i++) {
		Point pt = list[i];
		fprintf(fd, "%1.6lf, %1.6lf, %1.6lf, %1.6lf\n", pt->x, pt->y, pt->z, pt->result);
	}
	fclose(fd);
}


inline double find_min_radius(Point pt) {
	double l1, l2;
	double z = pt->z;
	double x = pt->x;
	if (pt->z >= 1 && pt->x <= 1) {
		l1 = 2-z;
		l2 = length(1-x, 1-z);
		pt->label = one;
	}
	else if (pt->x > 1) {
		l1 = 3-x;
		l2 = length(1-x, 1-z);
		pt->label = three;
	}
	else {
		l1 = length(1-x, 1-z);
		l2 = l1;
		pt->label = two;
	}
//	printf("Find radius: %lf\n", l1>=l2?l2:l1);
	return l1>=l2?l2:l1;
}

inline void move_to_next(Point pt, double r) {
	double u1 = (double)(rand()+1)/(RAND_MAX);
	double u2 = (double)(rand()+1)/(RAND_MAX);
	double u3 = (double)(rand()+1)/(RAND_MAX);
	double u4 = (double)(rand()+1)/(RAND_MAX);
	double z1 = sqrt(-2*log(u1))*cos(2*PI*u2);
	double z2 = sqrt(-2*log(u1))*sin(2*PI*u2);
	double z3 = sqrt(-2*log(u3))*cos(2*PI*u4);
	double norm = sqrt(z1*z1 + z2*z2 + z3*z3);
	if (z1/norm<-1 || z1/norm>1 || z2/norm<-1 || z2/norm>1 || z3/norm<-1 || z3/norm>1) {
		FILE* file = fopen("catcherror.txt", "w");
		fprintf(file, "%1.6lf, %1.6lf, %1.6lf\n", pt->x, pt->y, pt->z);
		fclose(file);
	}
	pt->x += r * (z1/norm);
	pt->y += r * (z2/norm);
	pt->z += r * (z3/norm);
}

inline void bounce_inside(Point pt) {
	while (! (0 <= pt->x && pt->x <= 3 && 0 <= pt->y && pt->y <= 1 && 
   0 <= pt->z && pt->z <= 2 && (! (1 < pt->x && pt->x <= 3 && 0 <= pt->y && pt->y <= 1 && 1 < pt->z && pt->z <= 2)))) {
		//Check x
		if (pt->x < 0) pt->x = -(pt->x);
		if (pt->y < 0) pt->y = -(pt->y);
		if (pt->y > 1) pt->y = 2 - (pt->y);
		if (pt->z < 0) pt->z = -(pt->z);
		if (pt->x >= 1 && pt->x <= 3 && pt->z >=1 && pt->z <= 2){ 
			if(pt->label == one)
				pt->x = 2 - pt->x;
			else
				pt->z = 2 - (pt->z);
		}
}
		
	if (pt->x >= 1) pt->label = three;
	else if (pt->z >= 1) pt->label = one;
	else pt->label = two;

	
//	printf("Move to %.6lf, %.6lf, %.6lf\n", pt->x, pt->y, pt->z);
}

inline bool isOnPlanes(Point pt, int* mark) {
	if ( pt->x >= 3 - PRECISE) {
		(*mark)++;
		#ifndef TEST
	//	fprintf(stderr, "Right plane\n");
		#endif
		return true;
	}
	else if (pt->z >= 2 - PRECISE) {
		#ifndef TEST
	//	fprintf(stderr, "Up plane\n");
		#endif
		return true;
	}
	return false;
}

