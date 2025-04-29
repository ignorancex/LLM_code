#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <string.h>

typedef struct {
	double x;
	double y;
	double result;
	unsigned int num_shots;
} point_t, *Point;

/* slide the volumn to select points  */
#define X_SLIDES 120 
#define Y_SLIDES 100
//#define Z_SLIDES 20
#define NUM_POINTS X_SLIDES * Y_SLIDES
#define NUM_SHOTS 5000000.0 /* Must be a float number */
#define PI 3.14159265358979323846
#define EPSILON 1e-6
//#define TEST
#define length(x, y) sqrt((x)*(x)+(y)*(y))
inline Point create_point(double, double);
inline Point copy_point(Point);
inline Point* get_points(int*);
inline void destroy_points(Point*, int);
inline void print_points(Point*, int, char*);
inline double find_min_radius(double, double);
inline void move_to_next(Point, double);
inline void bounce_inside(Point);
inline bool isOnDirichlet(Point, int*);
inline double walk_on_spheres(double x, double y, unsigned int num_shots, double epsilon); 

void random_permute(Point *array, int length) {                                                                                        srand(time(NULL));

    for (int i = length - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        Point temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

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
	random_permute(list, num_of_pts);
	srand(time(NULL));
	int total_section = atoi(argv[1]);
	int section = atoi(argv[2]);
	int part = num_of_pts/(total_section);
	int remainder = num_of_pts%(total_section);
	int start;
	int end;
	if (section < remainder){
		start = section * (part +1);
		end = start + part +1;
	}else{
		start = section * part + remainder;
		end = start + part;
	}
#ifdef TEST
	printf("This job is from %d to %d\n", start+1, end);
#endif
//	for (int i = start; i < end; i++) {
//		printf("%lf, %lf, %lf\n", list[i]->x, list[i]->y, list[i]->z);
//	}
	#pragma omp parallel for schedule(dynamic)
	for (int i = start; i < end; i++) {
		//DO hard work
		double x = list[i]->x;
		double y = list[i]->y;
		unsigned int num_shots = list[i]->num_shots;
#ifdef TEST
		printf("list[%d]: x = %lf, y = %lf\n",i, x, y);
		printf("list[%d]: walk_on_spheres(%lf, %lf, %u, %lf) = %lf\n",i , x, y, num_shots, EPSILON, walk_on_spheres(x, y, num_shots, EPSILON));
#endif
		list[i]->result = walk_on_spheres(x, y, num_shots, EPSILON);
#ifdef TEST
		printf("list[%d]->result = %lf\n",i, list[i]->result);
#endif
	}
	print_points(list+start, end-start, argv[2]);
	destroy_points(list, num_of_pts);
	time( &rawtime );
        timeinfo = localtime( &rawtime );
	//printf("Totally %d points, fails %d points.\n", end - start, fail_count);
#ifdef TEST
        printf( "Finish time and date: %s\n", asctime (timeinfo) );
#endif
	return 0;
}



inline double walk_on_spheres(double x1, double y1, unsigned int num_shots, double epsilon){
	double wx = x1;
	double wy = y1;
	unsigned int mark = 0;
	unsigned int j = 0;
#ifdef TEST
	printf("Inside walk_on_spheres, wx = %lf, wy = %lf, j = %u, num_shots = %u\n", wx, wy, j, num_shots);	
#endif
	while (j < num_shots) {
#ifdef TEST
		printf("Inside while loop\n");
#endif
		//fprintf(stderr, "Start %d:%d shots... \n", i, j);
		if (wy < 0) {
			wy = -wy;
        	} else if (wy > 4) {
            		wy = wy - 4;
        	} else if (wy > 1 && wy < 3 && wx > 2) {
        		    wx = 4 - wx;
        	}
		double dist1 = wx;
		double dist2 = length(wx - 2, wy - 1);
		double dist3 = length(wx - 2, wy - 3);
		double r = (wx <= 2) ? ((dist1 < dist2) ? ((dist1 < dist3) ? dist1 : dist3) : ((dist2 < dist3) ? dist2 : dist3)) :
			   ((wy < 1) ? ((1 - wy < 3 - wx) ? 1 - wy : 3 - wx) :
			   ((wy > 3) ? ((3 - wx < wy - 3) ? 3 - wx : wy - 3) :
			   ((wy - 1 < 3 - wy) ? wy - 1 : 3 - wy)));

		if(r<EPSILON){
			if(wx > 0.1) mark++;
			wx = x1;
			wy = y1;
			j++;
		}
		double u1 = ((double)(rand()))*2*PI/(RAND_MAX);
		double z1 = sin(u1);
		double z2 = cos(u1);
		wx += r * z1;
		wy += r * z2;
	}
#ifdef TEST
	printf("mark = %d, num_shots = %u\n", mark, num_shots);
#endif
	return ((double)mark)/num_shots;
}


inline Point copy_point(Point pt) {
	Point new = create_point(pt->x, pt->y);
	new->result = pt->result;
	return new;
}

inline Point create_point(double x, double y) {
	Point new = malloc(sizeof(*new));
	new->x = x;
	new->y = y;
	new->result = 0;
	return new;
}
	
inline Point* get_points(int* num_of_pts) {
	int counter = 0;
	double x_d = 3.0/X_SLIDES;
	double y_d = 2.0/Y_SLIDES;
	int list_Size = NUM_POINTS;
	Point* list = malloc(sizeof(*list) * NUM_POINTS);
	if (!list) {
        	perror("Failed to allocate memory");
        	exit(EXIT_FAILURE);
    	}
	for (int i = 0; i < list_Size; i++) list[i] = NULL;
	for (int i = 0; i <= X_SLIDES; i++) {
		double x = i * x_d;
		for (int j = 0; j <= Y_SLIDES; j++) {
			double y = j * y_d;
			//check validity
			if (x > 2 && y > 1) continue; //The point is not in the domain
			list[counter] = create_point(x, y);
			if (y > 1.5 || y < 0.5) list[counter]->num_shots = NUM_SHOTS *3;
			else list[counter]->num_shots = NUM_SHOTS;

			counter++;
		}
	}

	//Densify the numeric solution points at certain subregions
	for (int i = 0; i <= 20; i++){
		double x = 1.9 + (i+1) * 0.01;
		for(int j = 0; j <= 20; j++){
			double y = 0.95 + (j+1) * 0.0101;
			//check validity
			if (x > 2 && y > 1) continue; //The point is not in the domain
			if (counter >= list_Size){
				list_Size *= 2; // Double the size of the list
				list = (Point*)realloc(list, sizeof(*list) * list_Size);
				if (!list) {
				    perror("Failed to reallocate memory");
				    exit(EXIT_FAILURE);
				}
			}
			list[counter] = create_point(x, y);
			list[counter]->num_shots = NUM_SHOTS;
			counter++;
		}
	}

#ifdef TEST
	printf("Generated %d points.\n", counter);
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
		fprintf(fd, "%lf, %lf, %lf\n", pt->x, pt->y, pt->result);
	}
	fclose(fd);
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
//	printf("Find radius: %lf\n", l1>=l2?l2:l1);
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


inline bool isOnDirichlet(Point pt, int* mark) {
	if ( 3-pt->y < EPSILON) {                    //(*If y < 0, it means the Dirichlet boundary is the left one*);
		(*mark)++;
		#ifndef TEST
	//	fprintf(stderr, "Right plane\n");
		#endif
		return true;
	}
	else if(pt->y < EPSILON){
		#ifndef TEST
	//	fprintf(stderr, "Up plane\n");
		#endif
		return true;
	}
	return false;
}

