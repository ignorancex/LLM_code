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
 *  @file File_Manipulate.c
 *  @brief The functions used to take in and organize the parameters and covariances
 *  @author Bradley V Smith
 *  @version 1
 */
 #include <stdio.h>
 #include <string.h>
 #include <stdlib.h>
 #include "constant.h"
/**
 *  Prints out the inputed file to make sure that it is read properly
 *  @param the file to be printed out
 */
void print_file(FILE *file_ptr){
    /// File is not readable
    if(file_ptr == NULL){
        printf("file is empty\n");
    }
    /// Stores char to be printed
    char ch;
    ch = fgetc(file_ptr);
    /// Print out all chars in file
    while(ch != EOF){
        printf("%c",ch);
        ch = fgetc(file_ptr);
    }
    ///close file
    fclose(file_ptr);
    return;
}
/**
 *  Extracts the parameters from their file and organizes them into an array in an order that is usable
 *  @param file_ptr: the file to be read in
 *  @param param: the array that will store the parameters
 *  @param factorlist: the reference for the order to organize the parameters
 */
void param_org(FILE *file_ptr, double *param, const char *factorlist[]){
    /// Used to check which array to place data; 0 read in parameter name, 1 read in parameter
    int check = 0;
    /// Used to read in the chars
    char ch;
    /**
     *  Takes in the data and places it in the correct part of the array
     */
    for(int i = 0; i < param_num; i++){
        /// Starts extraction of next letter
        ch = fgetc(file_ptr);
        /// the string version of the parameter
        char num[pressision];
        /// Stores the name of the variable being extracted
        char factor[file_name_length];
        /// Stores the position of the array that the respective char will be stored on
        int k = 0;
        /// Extracts the variable and variable name
        while (ch != '\n'){
            /// Reads in the parameter name
            if (check == 0){
                /// Time to read in the parameter
                if(ch == ' '){
                    check = 1;
                    factor[k] = '\0';
                    k = 0;
                /// Read in next char of parameter name
                } else {
                    factor[k] = ch;
                    k++;

                }
            /// Reads in the parameter
            } else {
                num[k] = ch;
                k++;
            }
            /// Gets next char
            ch = fgetc(file_ptr);
        }
        /**
         *  Places the parameters into their correct positions in the array
         */
        for(int j = 0; j < param_num; j++){
            /// Finds the array position using the parameter name
            if(strcmp(factor, factorlist[j]) == 0){
                /// Places parameter in found position as a double
                *(param + j) = atof(num);
                /// Lets you know value is stored in said position, comment out
                //printf("%d  %s =  %s = %.20f\n", j,  factor, factorlist[j], param[j]);
            /// Check next position
            }else{
                /// Lets you know value is not in said position, comment out
                //printf("%s != %s\n", factor, factorlist[j]);
            }
        }
        /// get next parameter
        check = 0;
    }
}
/**
 *  Extracts the covariances from their file and organizes them into an array in an order that is usable
 *  @param file_ptr: the file to be read in
 *  @param covar: the array that will store the covariances
 *  @param factorlist: the reference for the order to organize the covariances
 */
void covar_org(FILE *file_ptr, double *covar, const char *factorlist[]){
    /// Used to check which array to place data; 0 read in parameter 1 name, 1 read in parameter 2 name, 3 read in covariance value
    int check = 0;
    /// Used to read in chars
    char ch;
    /**
     *  Takes in data and places it into the correct part of the array
     */
    for(int i = 0; i < param_num; i++){
        for(int j = 0; j < param_num; j++){
            /// Start the extraction process
            ch = fgetc(file_ptr);
            /// Stores the string version of the covariance
            char num[pressision];
            /// Stores parameter 1 name
            char factor1[file_name_length];
            /// Stores parameter 2 name
            char factor2[file_name_length];
            ///Stores position of array char will be stored in
            int k = 0;
            /// Extracts covariance and parameter names
            while(ch != '\n'){
                /// Extract parameter 1 name
                if(check == 0){
                    /// Time to read in parameter 2 name
                    if(ch == ' '){
                        check = 1;
                        factor1[k] = '\0';
                        k = 0;
                        fgetc(file_ptr);
                        /// Prints out parameter 1 name, comment out
                        //printf("factor 1 is %s\n", factor1);
                    /// Read in next char of parameter 1 name
                    }else{
                        factor1[k] = ch;
                        k++;
                    }
                /// Extract parameter 2 name
                }else if(check == 1){
                    /// Time to read in covariance
                    if(ch == ' '){
                        check = 2;
                        factor2[k] = '\0';
                        k = 0;
                        /// Prints out parameter 2 name, comment out
                        //printf("factor 2 is %s\n", factor2);
                    /// Read in next char of parameter 2 name
                    }else{
                        factor2[k] = ch;
                        k++;
                    }
                /// Extract covariance
                }else {
                    num[k] = ch;//grabs number
                    k++;
                }
                /// Extract next char
                ch = fgetc(file_ptr);
            }
            /// Get next covariance
            check = 0;
            /**
             *  Places covariances into the correct part of the array
             */
            for(int x = 0; x < param_num; x++){
                /// Uses parameter 1 name to find position 1
                if(strcmp(factor1, factorlist[x]) == 0){
                    /// Uses parameter 2 name to find position 2
                    for(int y = 0; y < param_num; y++){
                        if(strcmp(factor2,factorlist[y]) == 0){
                            /// Places covariance in found position as double
                            *(covar + (x* param_num+y)) = atof(num);
                            /// Lets you know value is stored in said position, comment out
                            //printf("%d   %s - %s = %.20f\n", y, factorlist[i], factorlist[j], covar[x][y]);
                        }
                    }
                }
            }
        }
    }
}
