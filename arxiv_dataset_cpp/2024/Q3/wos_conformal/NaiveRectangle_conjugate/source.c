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


//Define the height of the rectangle
#define HEIGHT 1.0
/* slide the volumn to select points  */
#define X_SLIDES 10 
#define Y_SLIDES 10
//#define Z_SLIDES 20
#define NUM_POINTS (X_SLIDES + 1) * (Y_SLIDES + 1)
#define NUM_SHOTS 1.0E7 /* Must be a float number */
#define PI 3.14159265358979323846
#define EPSILON 1e-6
//#define TEST
#define length(x, y) sqrt((x)*(x)+(y)*(y))
inline Point create_point(double, double);
inline Point copy_point(Point);
inline Point* get_points(int*);
inline void destroy_points(Point*, int);
inline void print_points(Point*, int, char*);
inline double walk_on_spheres(double x, double y, unsigned int num_shots, double epsilon); 


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
	printf("This job is from %d to %d\n", start+1, end);
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
	while (j < num_shots) {
		//fprintf(stderr, "Start %d:%d shots... \n", i, j);
		if (wx < 0) {
			wx = -wx;
        	}else if (wx > 1){
			wx = 2 - wx;
		}
		double r = (HEIGHT-wy) > wy ? wy : (HEIGHT-wy); 
		if(r<EPSILON){
			if(wy > HEIGHT/2) mark++;
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
	double x_d = 1.0/X_SLIDES;
	double y_d = 1.0/Y_SLIDES;
	int list_Size = NUM_POINTS;
	Point* list = malloc(sizeof(*list) * NUM_POINTS);
	if (!list) {
        	perror("Failed to allocate memory");
        	exit(EXIT_FAILURE);
    	}
	for (int i = 0; i < list_Size; i++) list[i] = NULL;
	for (int i = 0; i <= X_SLIDES; i++) {
		double x = i * x_d;
		printf("i = %d\n", i);
		for (int j = 0; j <= Y_SLIDES; j++) {
			double y = j * y_d;
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
/*
	int k = 0;
	while(k != NUM)
		printf("k = %d, list[%d]->x = %lf, list[%d]->y = %lf\n", k, k, list[k]->x, k, list[k]->y);
*/
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

