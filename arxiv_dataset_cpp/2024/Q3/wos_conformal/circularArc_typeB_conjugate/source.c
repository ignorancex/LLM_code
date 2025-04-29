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

#define NUM_SHOTS 1.0E7 /* Must be a float number */
#define PI 3.14159265358979323846
#define EPSILON 1e-6
#define length(x, y) sqrt((x)*(x)+(y)*(y))

#define v1_x 0.96592582628
#define v1_y 0.25881904510
#define v2_x 0.25881904510
#define v2_y 0.96592582628
#define v3_x 0.0
#define v3_y 1.0 
#define v4_x 1.0 
#define v4_y 0.0

inline Point create_point(double, double);
inline Point copy_point(Point);
inline Point* get_points(int*);
inline void destroy_points(Point*, int);
inline void print_points(Point*, int, char*);


inline bool inside_domain(double x, double y) {
    if (x*x + y*y > 1.0) return false;
    return true;
}


inline double walk_on_spheres(double x1, double y1, unsigned int NN, double eps) {
	double wx = x1, wy = y1;
	unsigned int ux = 0;
	int j = 0;
	double distance, theta, denom;
	while (j < NN) {
		denom = wx*wx + wy*wy;
		if (denom > 1) {
			wx = wx/denom;
			wy = wy/denom;
		}

		if (wx >= 0 && wy >= 0 && !(tan(2.0 / 24.0 * PI)*wx < wy  && wy < wx*tan(10.0 / 24.0 * PI))) {
		    distance = sqrt(fmin(
			(wx - 1) * (wx - 1) + wy * wy,
			fmin(
			    (wx - v1_x) * (wx - v1_x) + (wy - v1_y) * (wy - v1_y),
			    fmin(
				(wx - v2_x) * (wx - v2_x) + (wy - v2_y) * (wy - v2_y),
				(wx - v3_x) * (wx - v3_x) + (wy - v3_y) * (wy - v3_y)
			    )
			)
		    ));
		} else {
		    distance = 1 - sqrt(denom);
		} 			
		if (distance < eps) {
			ux += (wx > 0.1 && wy > 0.1) ? 1 : 0;
			wx = x1;
			wy = y1;
			j++;
		} else {
			theta = ((double)(rand()))*2*PI/(RAND_MAX);
			wx += distance * cos(theta);
			wy += distance * sin(theta);
		}
	}

	return ((double)ux) / NN;
}

void random_permute(Point *array, int length) {
    srand(time(NULL));
    
    for (int i = length - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        Point temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
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
	
    random_permute(list, num_of_pts);
    //for(int i=0; i<num_of_pts; i++) printf("counter = %d, list[%d]->x = %lf, list[%d]->y = %lf, length = %lf\n",num_of_pts,i,list[i]->x,i,list[i]->y, length(list[i]->x,list[i]->y));
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
    printf("This job is from %d to %d\n", start+1, end);
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
    for (i = 0; i < list_Size; i++) list[i] = NULL;
    for (i = 0; i <= X_SLIDES; i++) {
        x = -1 + i * x_d;
        for (j = 0; j <= Y_SLIDES; j++) {
            y = -1 + j * y_d;
            // check validity
            //printf("x = %lf, y = %lf\n", x, y);
            if (!inside_domain(x, y)) continue; // The point is not in the domain
            list[counter] = create_point(x, y);
            list[counter]->num_shots = NUM_SHOTS;
            counter++;
        }
    }

    //Smoothen the boundary
    for (i = 0; i <= 500; i++){
    	for (j = 0; j < 6; j++){
		if (counter >= list_Size){
			list_Size *= 2; // Double the size of the list
			list = (Point*)realloc(list, sizeof(*list) * list_Size);
			if (!list) {
			    perror("Failed to reallocate memory");
			    exit(EXIT_FAILURE);
			}
		}
		double r = 0.90 + 0.02 * j;
		double dtheta = 2*PI/500.0;
		double t = i * dtheta; 
		x = r * cos(t);
		y = r * sin(t);
		list[counter] = create_point(x, y); 
		list[counter]->num_shots = NUM_SHOTS;
		counter++;
	}
    } 
    *num_of_pts = counter;
    //printf("Reached partition\n");
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


