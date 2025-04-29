#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#define NUM_POINTS 1000
#define NUM_SHOTS 1.0E7
#define STEP 0.1
inline double dFivePoint(double fm2h, double fmh, double fph, double fp2h, double h);
double calculateAverage(unsigned long int array[], int size);
void readData(const char* filename, unsigned long int array[], int size);

int main(int argc, char** argv) {
    unsigned long int fm2h_list[NUM_POINTS] = {0};
    unsigned long int fmh_list[NUM_POINTS] = {0};
    unsigned long int fph_list[NUM_POINTS] = {0};
    unsigned long int fp2h_list[NUM_POINTS] = {0};
    unsigned long int cfm2h_list[NUM_POINTS] = {0};
    unsigned long int cfmh_list[NUM_POINTS] = {0};
    unsigned long int cfph_list[NUM_POINTS] = {0};
    unsigned long int cfp2h_list[NUM_POINTS] = {0};

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
    if (fd == NULL) {
        perror("Error opening result file for writing");
        exit(1);
    }
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

double calculateAverage(unsigned long int array[], int size) {
    unsigned long int sum = 0;

    for (int i = 0; i < size; i++) {
        sum += array[i];
    }

    return (double)sum / size;
}

void readData(const char* filename, unsigned long int array[], int size) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file for reading");
        exit(1);
    }
    printf("Reading data from %s\n", filename);

    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%lu", &array[i]) != 1) {
            printf("Error reading data at index %d\n", i);
            perror("Error reading data from file");
            fclose(file);
            exit(1);
        }
        //printf("Read value %u at index %d\n", array[i], i);
    }

    fclose(file);
    printf("Finished reading data from %s\n", filename);
}

