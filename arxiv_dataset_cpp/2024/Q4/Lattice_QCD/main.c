/**
  * This file is part of Lattice_QCD.
  *
  * Lattice_QCD is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
  *
  * Lattice_QCD is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  * GNU General Public License for more details.
  *
  * You should have received a copy of the GNU General Public License
  * along with Lattice_QCD.  If not, see <https://www.gnu.org/licenses/>.
  */
/**
 *  @file main.c
 *  @brief the main function
 *  This program takes in the variables and covariance for nominal and higher order
 *  form factors as well as several constants and prints out the data points for the
 *  decay of the given system.
 *  @author Bradley V Smith
 *  @version 2
 *  @param results_ptr1: the file containing the nominal form factor variables
 *  @param covariance_ptr1: the file containing the nominal from factor covariances
 *  @param results_ptr2: the file containing the higher order form factor variables
 *  @param covariance_ptr2: the file containing the higher order form factor covariances
 *  @param gs:
 *  @param gp:
 *  @param gL:
 *  @param gR:
 *  @param gT:
 *  @param ml:
 *  @return data.txt the file containing the points for the decay rate
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "File_Manipulate.h"
#include "Calculate.h"

int main()
{
    /// finishes the declaration of some important variables
    tminus = pow(mlambdab - mlambdac, 2);
    t0 = tminus;
    qsqrmax = tminus;
    end = qsqrmax;
    start = qsqrmin(ml);
    /**
     *  Sets up the pointers to take in the files as well as the string to take in the file name
     *  This must be commented out when using preset file name
     */
    /// For the nominal form factor parameters
    FILE *results_ptr1;
    char results_name1[file_name_length];
    /// For the nominal form factor covariances
    FILE *covariance_ptr1;
    char covariance_name1[file_name_length];
    /// For the higher order form factor parameters
    FILE *results_ptr2;
    char results_name2[file_name_length];
    /// For the higher order form factor covariances
    FILE *covariance_ptr2;
    char covariance_name2[file_name_length];
    ///for testing, has a preset name for the files, must comment when using previous section
    /**
     *  Sets up the pointers to take in the files as well as a string containing a preset file name
     *  This section must be commented out when using the normal declaration section
     *  For testing purposes
     *
    /// For the  nominal form factor parameters
    FILE *results_ptr1;
    char results_name1[] = "results.dat";
    /// For the nominal form factor covariances
    FILE *covariance_ptr1;
    char covariance_name1[] = "covariance.dat";
    /// For the higher order form factor parameters
    FILE *results_ptr2;
    char results_name2[] = "HO_results.dat";
    /// For the higher order form factor covariances
    FILE *covariance_ptr2;
    char covariance_name2[] = "HO_covariance.dat";
    /**
     *  Acquires the filename of the file with the nominal form factor parameters from the user
     *  This section must be commented out when using preset file name
     */
    printf("Enter in the filename of the file with the form factor parameters for the first set: ");
    scanf("%s", results_name1);
    /// Prints out file name, comment out
    //printf("%s\n", results_name1);
    /**
     *  Opens the file with the nominal form factor parameters, ends program if file isn't readable
     */
    results_ptr1 = fopen(results_name1, "r");
    if(results_ptr1 == NULL){
        printf("file cannot be opened, please check if it is in correct location");
        exit(0);
    }
    /**
     *  Acquires the filename of the file with the nominal form factor covariances from the user
     *  This section must be commented out when using preset file name
     */
    printf("Enter in the file name of the file with the covariances for the first set: "); //asks for file name
    scanf("%s", covariance_name1); //reads in file name
    /// Prints out file name, comment out
    //printf("%s\n", covariance_name); //tests if file name is read in correctly
    /**
     *  Opens the file with the nominal form factor covariances, ends program if file isn't readable
     */
    covariance_ptr1 = fopen(covariance_name1, "r");
    if(covariance_ptr1 == NULL){
        printf("file cannot be opened, please check if it is in correct location");
        exit(0);
    }
    /**
     *  Acquires the filename of the file with the higher order form factor parameters from the user
     *  This section must be commented out when using preset file name
     */
    printf("Enter in the filename of the file with the form factor parameters for the second set: "); //asks for file name
    scanf("%s", results_name2); //reads in file name
    /// Prints out file name, comment out
    //printf("%s\n", results_name);
    /**
     *  Opens the file with the higher order form factor parameters, ends program if file isn't readable
     */
    results_ptr2 = fopen(results_name2, "r");
    if(results_ptr2 == NULL){
        printf("file cannot be opened, please check if it is in correct location");
        exit(0);
    }
    /**
     *  Acquires the filename of the file with the higher order form factor covariances from the user
     *  This section must be commented out when using preset file name
     */
    printf("Enter in the file name of the file with the covariances for the second set: "); //asks for file name
    scanf("%s", covariance_name2); //reads in file name
    /// Prints out file name, comment out
    //printf("%s\n", covariance_name);
    /**
     *  Opens the file with the higher order form factor covariances, ends program if file isn't readable
     */
    covariance_ptr2 = fopen(covariance_name2, "r");
    if(covariance_ptr2 == NULL){
        printf("file cannot be opened, please check if it is in correct location");
        exit(0);
    }
    // this takes in the additional variables
    /**
     *  Takes in the additional variables
     *  Comment out if you want to use default variables
     */
    printf("please enter in the gs: ");
    scanf("%lf", &gs);
    printf("please enter in the gp: ");
    scanf("%lf", &gp);
    printf("please enter in the gL: ");
    scanf("%lf", &gL);
    printf("please enter in the gR: ");
    scanf("%lf", &gR);
    printf ("please enter in the gT: ");
    scanf("%lf", &gT);
    /**
     *  Organize the parameters and covariances for the nominal form factors into a usable order
     *  and place into their respective arrays
     *  Order for the nominal form factors
     *   a0fplus - 0, a1fplus - 1, a0fperp - 2, a1fperp - 3, a0f0 - 4, a1f0 - 5, a0gpp - 6, a1gplus - 7,
     *   a1gperp - 8, a0g0 - 9, a1g0 - 10, a0hplus - 11, a1hplus - 12, a0hperp - 13, a1hperp - 14,
     *   a0htildepp - 15, a1htildeplus - 16, a1htildeperp - 17
    */
    param_num = set1;
    double parameters1[param_num];
    double covariances1[param_num][param_num];
    param_org(results_ptr1, parameters1, nominalparam);
    covar_org(covariance_ptr1, (double *)covariances1, nominalparam);
    /// Checks that the parameters where organized correctly, comment out
    /*for(int i = 0; i < param_num; i++){
        printf("%.20f\n", parameters1[i]);
    }*/
    /**
     *  Organizes the parameters and covariances for the nominal form factors into a usable order
     *  and place into their respective arrays
     *  Order for the higher order form factors
     *   a0fplus - 0, a1fplus - 1, a2fplus - 2, a0fperp - 3, a1fperp - 4, a2fperp -5, a0f0 - 6, a1f0 - 7,
     *   a2f0 - 8, a0gpp - 9, a1gplus - 10, a2gplus - 11, a1gperp - 12, a2gperp - 13, a0g0 - 14, a1g0 - 15,
     *   a2g0 - 16, a0hplus - 17, a1hplus - 18, a2hplus - 19, a0hperp - 20, a1hperp -21, a2hperp - 22,
     *   a0htildepp - 23, a1htildeplus - 24, ahtildeplus - 25, a1htildeperp - 26, a2htildeperp - 27
    */
    param_num = set2;
    double parameters2[param_num];
    double covariances2[param_num][param_num];
    param_org(results_ptr2, parameters2, hoparam);
    covar_org(covariance_ptr2, (double *)covariances2, hoparam);
    /// Prints parameters in order stored, comment out
    /*for(int i = 0; i < param_num; i++){
        printf("%.20f\n", parameters2[i]);
    }*/
    /**
     *  Closes the files, they are no longer needed
     */
    fclose(results_ptr1);
    fclose(results_ptr2);
    fclose(covariance_ptr1);
    fclose(covariance_ptr2);
    /**
     *  Sets up the q^2 points, which acts as the independent variable
     */
    double qsquared[points];
    qsquared[0] = start;
    for(int i = 1; i < points; i++){
        qsquared[i] = qsquared[i - 1] + ((end - start)/points);
    }
    /// A test value for a 1 point test, comment out
    //qsquared[0] = 10; //a test value
    /**
     *  Stores the decay rate, the dependent variable
     */
    double decay[points];
    /**
     *  Stores the error value for the decay rate
     */
    double error[points];
    /**
     *  Stores the values for the lower, "negative", error band
     */
    double negerror[points];
    /**
     *  Stores the values for the upper, "positive", error band
     */
    double poserror[points];
    /**
     *  This loop finds the decay and error values for each q^2 value
     */
    for(int i = 0; i <points; i++){
        /// Stores the nominal decay rate for the given q^2
        double resultnom = nominalcalc(qsquared[i], parameters1);
        /// Stores the higher order decay rate for the given q^2
        double resultHO = HOcalc(qsquared[i], parameters2);
        param_num = set1;
        /// Stores the nominal error for the given q^2
        double resulterrnom = resulterrorcalc(qsquared[i], resultnom, parameters1, (double *)covariances1);
        param_num = set2;
        /// Stores the higher order error for the given q^2
        double resulterrHO = resulterrorcalc(qsquared[i], resultHO, parameters2, (double *)covariances2);
        decay[i] = resultnom;
        error[i] = totalerrorcalc(resultnom, resultHO, resulterrnom, resulterrHO);
        negerror[i] = decay[i] - error[i];
        poserror[i] = decay[i] + error[i];
        /**
         * Some tests to check if the decay calculations are going well
         */
        /// Prints out the decay rates, comment out
        //printf ("nominal = %lf,   HO = %lf\n", resultnom, resultHO);
        /// Prints out the errors, comment out
        //printf("errornom = %lf,  errorHO = %lf\n", resulterrnom, resulterrHO);
        /// Pints out the decay rate and the final error, comment out
        //printf("decay = %lf,  error = %lf\n", decay[i], error[i]);
    }
    /**
     *  Organizes the data into a file for a graphing program to take in and create the charts
     */
    /// The filename of the file used to store the data
    char datafilename[] = "data.txt";
    /// Sets up the file for reading in the data, if file exists it clears it out
    FILE *datafile = fopen(datafilename, "w");
    fprintf(datafile, "qsqr decay negerror poserror\n");
    /// Places the data into the file
    for(int i = 0; i < points; i++){
        fprintf(datafile, "%lf %lf %lf %lf\n", qsquared[i], decay[i], negerror[i], poserror[i]);
        /// Prints out the data in the same format as placed in the file, comment out
        //printf("%lf %lf %lf %lf\n", qsquared[i], decay[i], negerror[i], poserror[i]);
    }
    /// closes the file, it is finished
    fclose(datafile);

    return 0;
}

