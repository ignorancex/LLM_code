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
#define Y_SLIDES 100
#define X_SLIDES 100
#define NUM_POINTS ((X_SLIDES + 1) * (Y_SLIDES + 1))

#define NUM_SHOTS 100000.0 /* Must be a float number */
#define PI 3.14159265358979323846
#define EPSILON 1e-6
#define length(x, y) sqrt((x)*(x)+(y)*(y))

#define center2x -1.164524664599176
#define center2y -0.4823619097949585
#define r2 0.7673269879789604
#define center4x 1.0
#define center4y 0.5773502691896258
#define r4 0.5773502691896258
#define Tarc2Centerx 0.8617632503267921
#define Tarc2Centery 0.509672053455795
#define Tarc2Radius 0.04900511899428376
#define Tarc4Centerx -0.9317405419866858
#define Tarc4Centery -0.3683950080353163
#define Tarc4Radius 0.06209121940325892


inline Point create_point(double, double);
inline Point copy_point(Point);
inline Point* get_points(int*);
inline void destroy_points(Point*, int);
inline void print_points(Point*, int, char*);
inline double find_min_radius(double, double);
inline void move_to_next(Point, double);


inline bool inside_domain(double x, double y) {
    if (x*x + y*y > 1.0) return false;
    if (sqrt((x - center2x)*(x - center2x) + (y - center2y)*(y - center2y)) < r2) return false;
    if (sqrt((x - center4x)*(x - center4x) + (y - center4y)*(y - center4y)) < r4) return false;
    return true;
}


inline double walk_on_spheres(double x1, double y1, unsigned int NN, double eps) {
	double wx = x1, wy = y1;
	unsigned int cnt = 0;
	int j = 0;
	double distance, theta, denom;

	while (j < NN) {
		denom = (wx*wx + wy*wy); 
		if (denom > 1) {
			wx = wx/denom;
			wy = wy/denom;
		}
		double dist1 = length( wx-center2x, wy-center2y ) - r2;
		double dist2 = length( wx-center4x, wy-center4y ) - r4;  
		distance = dist1 > dist2 ? dist2 : dist1;
		if (distance < eps) {
			cnt += (wx < 0) ? 0 : 1;
			wx = x1;
			wy = y1;
			j++;
		} else {
			theta = ((double)(rand()))*2*PI/(RAND_MAX);
			wx += distance * cos(theta);
			wy += distance * sin(theta);
		}
	}

	return ((double)cnt) / NN;
}

int main(int argc, char** argv) {

    time_t rawtime;
    struct tm * timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    printf("Current local time and date: %s", asctime(timeinfo));
    printf("Every point has %lf shots\n", NUM_SHOTS);
    // Get a list of points
    int num_of_pts;
    Point* list = get_points(&num_of_pts);
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
    int i;
    #pragma omp parallel for schedule(dynamic)
    for (i = start; i < end; i++) {
        double x = list[i]->x;
        double y = list[i]->y;
        unsigned int num_shots = list[i]->num_shots;
        list[i]->result = walk_on_spheres(x, y, num_shots, EPSILON);
    }
    print_points(list + start, end - start, argv[2]);
    destroy_points(list, num_of_pts);
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    return 0;
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
    double x_d = 2.0 / X_SLIDES;
    double y_d = 2.0 / Y_SLIDES;
    double x;
    double y;
    int list_Size = NUM_POINTS;
    Point* list = malloc(sizeof(*list) * NUM_POINTS);
    if (!list) {
        perror("Failed to allocate memory");
        exit(EXIT_FAILURE);
    }
    int i;
    int j;
	//Smoothen the boundary
    for (i = 0; i < list_Size; i++) list[i] = NULL;
    for (i = 0; i <= 500; i++){
	if (counter >= list_Size){
		list_Size *= 2; // Double the size of the list
		list = (Point*)realloc(list, sizeof(*list) * list_Size);
		if (!list) {
		    perror("Failed to reallocate memory");
		    exit(EXIT_FAILURE);
		}
	}
    	double dtheta = ((11.0/12.0)-(1.0/3.0))*PI/500.0;
	double t = PI/3 + i * dtheta; 
	x = cos(t);
	y = sin(t);
	list[counter] = create_point(x, y); 
	list[counter]->num_shots = NUM_SHOTS;
	counter++;
    } 
    for (i = 0; i <= 500; i++){
	if (counter >= list_Size){
		list_Size *= 2; // Double the size of the list
		list = (Point*)realloc(list, sizeof(*list) * list_Size);
		if (!list) {
		    perror("Failed to reallocate memory");
		    exit(EXIT_FAILURE);
		}
	}
    	double dtheta = (2.0/3.0)*PI/500.0;
	double t = (-2.0/3.0)*PI + i * dtheta; 
	x = cos(t);
	y = sin(t);
	list[counter] = create_point(x, y); 
	list[counter]->num_shots = NUM_SHOTS;
	counter++;
    } 
    for (i = 0; i <= X_SLIDES; i++) {
        x = -1 + i * x_d;
        for (j = 0; j <= Y_SLIDES; j++) {
	    if (counter >= list_Size){
		list_Size *= 2; // Double the size of the list
		list = (Point*)realloc(list, sizeof(*list) * list_Size);
		if (!list) {
		    	perror("Failed to reallocate memory");
		    	exit(EXIT_FAILURE);
		}
	    }
            y = -1 + j * y_d;
            // check validity
            if (!inside_domain(x, y)) continue; // The point is not in the domain
            list[counter] = create_point(x, y);
            list[counter]->num_shots = NUM_SHOTS;
            counter++;
        }
    }

    *num_of_pts = counter;
    return list;
}

inline void destroy_points(Point* list, int num_of_pts) {
	int i;
    for (i = 0; i < num_of_pts; i++)
        free(list[i]);
    free(list);
}

inline void print_points(Point* list, int num_of_pts, char* str) {
    char name[15] = "points";
    strcat(name, str);
    strcat(name, ".txt");
    FILE* fd = fopen(name, "w");
    int i;
    for (i = 0; i < num_of_pts; i++) {
        Point pt = list[i];
        fprintf(fd, "%lf, %lf, %lf\n", pt->x, pt->y, pt->result);
    }
    fclose(fd);
}

inline double find_min_radius(double x, double y) {
    //double center2x = -1.16452, center2y = -0.482362, r2 = 0.767327;
    //double center4x = 1.0, center4y = 0.57735, r4 = 0.57735;
    double dist1 = sqrt(x * x + y * y) - 1;
    double dist2 = length(x-center2x, y-center2y) - r2;
    double dist3 = length(x-center4x, y-center4y) - r4;
    return fmin(dist1, fmin(dist2, dist3));
}

inline void move_to_next(Point pt, double r) {
    double theta = ((double)(rand())) * 2 * PI / (RAND_MAX);
    pt->x += r * cos(theta);
    pt->y += r * sin(theta);
}

