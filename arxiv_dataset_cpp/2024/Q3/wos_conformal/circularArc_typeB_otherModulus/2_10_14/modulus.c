#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>


#define NUM_POINTS 1000 //Number of node in each list of fm2h, fmh, fph, fp2h
#define NUM_SHOTS 1.0E5 /* Total number of shots for each of fm2h, fmh, fph, fp2h. Must be a float number */
#define STEP 0.1
inline double dFivePoint(double fm2h, double fmh, double fph, double fp2h, double h);
double calculateAverage(unsigned int array[], int size);
void readData(const char* filename, unsigned int array[], int size); 

int main(int argc, char** argv) {
	unsigned int fm2h_list[NUM_POINTS] = {0};
	unsigned int fmh_list[NUM_POINTS] = {0};
	unsigned int fph_list[NUM_POINTS] = {0};
	unsigned int fp2h_list[NUM_POINTS] = {0};
	unsigned int cfm2h_list[NUM_POINTS] = {0};
	unsigned int cfmh_list[NUM_POINTS] = {0};
	unsigned int cfph_list[NUM_POINTS] = {0};
	unsigned int cfp2h_list[NUM_POINTS] = {0};

	readData("fm2h.txt", fm2h_list, NUM_POINTS);
	readData("fmh.txt", fmh_list, NUM_POINTS);
	readData("fph.txt", fph_list, NUM_POINTS);
	readData("fp2h.txt", fp2h_list, NUM_POINTS);
	readData("cfm2h.txt", cfm2h_list, NUM_POINTS);
	readData("cfmh.txt", cfmh_list, NUM_POINTS);
	readData("cfph.txt", cfph_list, NUM_POINTS);
	readData("cfp2h.txt", cfp2h_list, NUM_POINTS);

	double fm2h = calculateAverage(fm2h_list, NUM_POINTS)/NUM_SHOTS;
	double fmh = calculateAverage(fmh_list, NUM_POINTS)/NUM_SHOTS;
	double fph = calculateAverage(fph_list, NUM_POINTS)/NUM_SHOTS;
	double fp2h = calculateAverage(fp2h_list, NUM_POINTS)/NUM_SHOTS;
	double cfm2h = calculateAverage(cfm2h_list, NUM_POINTS)/NUM_SHOTS;
	double cfmh = calculateAverage(cfmh_list, NUM_POINTS)/NUM_SHOTS;
	double cfph = calculateAverage(cfph_list, NUM_POINTS)/NUM_SHOTS;
	double cfp2h = calculateAverage(cfp2h_list, NUM_POINTS)/NUM_SHOTS;
	double ux = dFivePoint(fm2h, fmh, fph, fp2h, STEP);	
	double vy = dFivePoint(cfm2h, cfmh, cfph, cfp2h, STEP);	
	const char* name = "result.txt";
	FILE* fd = fopen(name, "w");
	fprintf(fd, "fm2h = %lf, fmh = %lf, fph = %lf, fp2h = %lf, ux = %lf\n", fm2h, fmh, fph, fp2h, ux);
	fprintf(fd, "cfm2h = %lf, cfmh = %lf, cfph = %lf, cfp2h = %lf, vy = %lf\n", cfm2h, cfmh, cfph, cfp2h, vy);
	fprintf(fd, "modulus h = ux/vy = %lf\n", ux/vy);
	fprintf(fd, "modulus of the conjugate quadrilateral h = vy/ux = %lf\n", vy/ux);
	fclose(fd);
	return 0;
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

void readData(const char* filename, unsigned int array[], int size) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file for reading");
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%u", &array[i]) != 1) {
            perror("Error reading data from file");
            fclose(file);
            exit(1);
        }
    }

    fclose(file);
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

inline void print_results(Point* list, int num_of_pts, char* str) {
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

