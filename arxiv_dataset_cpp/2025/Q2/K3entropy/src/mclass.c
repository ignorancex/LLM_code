#include "k3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

//Initialize arrays to path position data; used in determine_arrays
int pos_array[PSMAX]; 
int above_array[PSMAX];

//Orbit_length is the order of the periodic point we are using
#define orbit_length 10

//Matrix for storing twist data
const char *word[PSMAX];

/*Function headers and descriptions*/


/*repeats from paths.c*/
bool simplify_path(int *arr1, int *arr2, int *size);
// eliminates a redundancy in the path if one exists. if none exist, returns false

void fully_simplify_path(int *arr1, int *arr2, int *size) ;
// fully simplifies array_pos and array_height

/*new functions*/
void s(int *arr1, int *arr2, int *size, int i);
//apply s_i dehn half-twist to paths
	//must apply s to SIMPLIFIED sequences
	//s_i makes COUNTERCLOCKWISE rotation

void S(int *arr1, int *arr2, int *size, int i);
//apply S_i = s_i^{-1} to paths
	
bool reducing_move(int *arr1, int *arr2, int *size, int *twistsy, int *len);
//applies reducing move to array, if possible
	//returns true if reduction is possible, false otherwise

void star_algorithm(int *arr1, int *arr2, int *size, int *twistsy, int *len, int k);
//applies twists to path until path starts at k and moves one step in postive direction.
	//stores twists in twistsy

void contracted_system(int *arr1, int *arr2, int *size, int *arr1_temp, int *arr2_temp, int k, int *len);
//returns new data after contracting p_0.p_1...p_k-1.

void expand_twist_data(int *arr, int *size, int k);
//translates twist data on contracted system to twist data on full system

int main(int argc, char *argv[]) {

	//copied output from paths.c
	int path_lengths[orbit_length-1] = {13, 3, 8, 12, 11, 11, 2, 7, 7};
	int array_pos[orbit_length-1][10000] = {
		{2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5},
		{5, 6, 7},
		{7, 6, 5, 5, 6, 7, 8, 9},
		{9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8},
		{8, 9, 9, 8, 7, 6, 5, 4, 3, 3, 4},
		{4, 3},
		{3, 4, 4, 3, 2, 1, 0},
		{0, 1, 2, 3, 4, 5, 6}
	};
	int array_height[orbit_length-1][10000] = {
		{-1000, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, -1000},
		{-1000, 1, -1000},
		{-1000, 1, 1, 0, 1, 0, 0, -1000},
		{-1000, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, -1000},
		{-1000, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1000},
		{-1000, 1, 0, 0, 0, 0, 0, 0, 0, 1, -1000},
		{-1000, -1000},
		{-1000, 0, 1, 1, 0, 1, -1000},
		{-1000, 1, 1, 1, 1, 0, -1000}
	};


	//copy arrays for testing
	int **array_pos_testing = malloc((orbit_length-1) * sizeof(int*));
	int **array_height_testing = malloc((orbit_length -1) * sizeof(int*));
	int path_lengths_testing[orbit_length-1];
	
	for (int i = 0; i < orbit_length-1; i++) {
		array_pos_testing[i] = malloc(100000 * sizeof(int)); 
		array_height_testing[i] = malloc(100000 * sizeof(int));
    }

	for (int i = 0; i < orbit_length-1; i++) {
		path_lengths_testing[i] =  path_lengths[i];
		for (int j = 0; j < path_lengths[i]; j++) {
			array_pos_testing[i][j] = array_pos[i][j];
			array_height_testing[i][j] = array_height[i][j];
		}
	}

	//Creates arrays to record twists and keep track of its length.
	char *twists_word1[10000]; //twists as characters
	char *twists_word2[10000]; //twists as characters
	int twists[10000]; //twists with integer coding, meaning i = s_i; -i = S_{i-1}
	int twists_length = 0; //length of twist array


	//Creates temporary arrays to store path after contracting first through kth vertices
	int arr1_temp[100000]; //size 1000 is safe
	int arr2_temp[100000];
	int size_pos_temp = 0;

	//Creates temporary array to store new twists
	int new_twists[100000];
	int size_temp = 0;

	//MAIN MACHINERY: iterates through paths, applies twists, stores result in twist array
	fprintf(stderr, "to main machinery:\n");

	int twists_length_old = 0;

	for(int i = 0; i < orbit_length-1; i++){
		
		size_temp = 0; //size_temp will be used to keep track of the number of twists

		fprintf(stderr, "\n ITERATION NUMBER %d\n", i);

		//applies all previous twists to ith path
		for(int o = 0; o < orbit_length - 1; o++){
			if(o >= i){
				for(int j = twists_length_old; j < twists_length; j++){
					if(twists[j] >= 0){
						s(array_pos[o], array_height[o], &path_lengths[o], twists[j]);
					}
					if(twists[j] < 0){
						S(array_pos[o], array_height[o], &path_lengths[o], -twists[j]-1); //S not yet defined
					}
				}
			}
		}
		twists_length_old = twists_length;

		// print input data:
		for(int o = 0; o < orbit_length - 1; o++){
			for(int j = 0; j < path_lengths[o]; j++){
				fprintf(stderr, "(%d, %d)\n", array_pos[o][j], array_height[o][j]);
			}
		}

		//contracts vertex 0, 1, .., i, stores new data in temporary position/height arrays
		delete_indices(array_pos[i], array_height[i], &path_lengths[i], arr1_temp, arr2_temp, i, &size_pos_temp);

		//simplifies contracted path data
		fully_simplify_path(arr1_temp, arr2_temp, &size_pos_temp);

		//stores twist data of contracted path in temporary twist array
		star_algorithm(arr1_temp, arr2_temp, &size_pos_temp, new_twists, &size_temp, i); //orbit_length - 2 > number of contracted strands
		
		//print new twists in contracted system
		fprintf(stderr, " Apply \\tilde g_%d: ", i);
		for(int j = size_temp-1; j >= 0; j--){
			if(new_twists[j] >= 0){
				if(new_twists[j] == i && i > 0){
					fprintf(stderr, "\\tilde s_%d.", new_twists[j]);
				} else{
					fprintf(stderr, "s_%d.", new_twists[j]);
				}
			}
			if(new_twists[j] < 0){
				if( -new_twists[j]-1 == i && i > 0){
					fprintf(stderr, "\\tilde s_%d^{-1}.", -new_twists[j]-1);
				} else{
					fprintf(stderr, "s_%d^{-1}.", -new_twists[j]-1);
				}
			}
		}

		//translates twist data on contracted path to twist data on full path
		process_array(new_twists, &size_temp, i);

		//print new twists in full system
		fprintf(stderr, "\n \n Apply g_%d': ", i);
		for(int j = size_temp - 1; j >= 0; j--){
			if(new_twists[j] >= 0){
				fprintf(stderr, "s_%d.", new_twists[j]);				
			}
			if(new_twists[j] < 0){
				fprintf(stderr, "s_%d^{-1}.", -new_twists[j]-1);
			}
		}

		fprintf(stderr, "\n \n (g_%d')^{-1}: ", i);
		for(int j = 0; j < size_temp; j++){
			if(new_twists[j] >= 0){
				fprintf(stderr, "s_%d^{-1}.", new_twists[j]);				
			}
			if(new_twists[j] < 0){
				fprintf(stderr, "s_%d.", -new_twists[j]-1);
			}
		}

		//stores new translated twist data in final twist array
		for(int j = 0; j < size_temp; j++){
			twists[twists_length + j] = new_twists[j];
		}
		twists_length += size_temp;


		//applies new twists to all paths <= i
		for(int k = 0; k<=i; k++){
			for(int j = 0; j < size_temp; j++){
				if(new_twists[j] >= 0){
					s(array_pos[k], array_height[k], &path_lengths[k],new_twists[j]);
				}
				if(new_twists[j] < 0){
					S(array_pos[k], array_height[k], &path_lengths[k], -new_twists[j]-1); //S not yet defined}
				}
			}
		}

		
		

		/*DEHN TWISTS*/
			//path_lengths[i] > 2 means a Dehn twist is needed.
			//want to use star algorithm to move ending point of ith path

		if(path_lengths[i] > 2){ //
			fprintf(stderr, "twisting!\n");
			size_temp = 0;
		
			star_algorithm(array_pos[i], array_height[i], &path_lengths[i], new_twists, &size_temp, i); //this will apply dehn twists
			
			//Stores new translated twist data in final twist array
			for(int j = 0; j < size_temp; j++){
				twists[twists_length + j] = new_twists[j];
			}
			twists_length += size_temp;

			fprintf(stderr, "Apply twist: ");
			for(int j = size_temp-1; j >= 0; j--){
				if(new_twists[j] >= 0){
					if(new_twists[j] == i && i > 0){
						fprintf(stderr, "\\tilde s_%d.", new_twists[j]);
					} else{
						fprintf(stderr, "s_%d.", new_twists[j]);
					}
					
				}
				if(new_twists[j] < 0 && i > 0){
					if( -new_twists[j]-1 == i){
						fprintf(stderr, "\\tilde s_%d^{-1}.", -new_twists[j]-1);
					} else{
						fprintf(stderr, "s_%d^{-1}.", -new_twists[j]-1);
					}
				}
			}

		}
		

		//Manually adding a twist to make up for the reversal!
		//if(size_temp > 0){
		//	if(twists[twists_length-1] < 0){
		//		twists[twists_length] = -i-1;
		//	} else{
		//		twists[twists_length] = i;
		//	} 
		//	twists_length++;
		//	fprintf(stderr, "s_%d\n", twists[twists_length-1]);
		//}
	}

	fprintf(stderr, "main machinery complete\n");
	
	//simplify_final_array(twists, &twists_length);
	//fprintf(stderr, "array is simplified!\n");

	//check machinery
	fprintf(stderr, "checking machinery\n");
	for(int k = 0; k < orbit_length-1; k++){
		for(int j = 0; j < path_lengths_testing[k]; j++){
			fprintf(stderr, "(%d, %d)\n", array_pos_testing[k][j], array_height_testing[k][j]);
		}	
		for(int j = 0; j < twists_length; j++){
			if(twists[j] >= 0){
				s(array_pos_testing[k], array_height_testing[k], &path_lengths_testing[k], twists[j]);
				//fprintf(stderr, "s_%d \n", twists[j]);
			}
			if(twists[j] < 0){
				S(array_pos_testing[k], array_height_testing[k], &path_lengths_testing[k], -twists[j]-1); 
				//fprintf(stderr, "S_%d \n", -twists[j]-1);
			}
		}
		for(int n = 0; n < path_lengths_testing[k]; n++){
			fprintf(stderr, "(%d, %d)\n", array_pos_testing[k][n], array_height_testing[k][n]); 
		}
		fprintf(stderr, "end\n");
	}

	int k = 0;

	//y_final_array(twists, &twists_length);
	//fprintf(stderr, "array is simplified!\n");


	//translate final twist array into s_i's. 
	for(int i = 0; i < twists_length; i++){
		twists_word1[i] = malloc(5); // Allocate memory for "S_i" 
		twists_word2[i] = malloc(5);
		if(twists[i]>=0){
			sprintf(twists_word1[i], "s_%d", twists[i]);
			sprintf(twists_word2[i], "S_%d", twists[i]);
		}
		if(twists[i]<0){
			sprintf(twists_word1[i], "S_%d", -twists[i]-1);
			sprintf(twists_word2[i], "s_%d", -twists[i]-1);
		}
	}

	//fprintf(stderr, "length of g is %d\n", twists_length);

	//print g, g^{-1}, \hat g, and f^2

	fprintf(stderr, "\n g = ");
	for (int j = twists_length-1; j >= 0  ; j--) {
        fprintf(stderr, "%s.", twists_word1[j]);
    }

	fprintf(stderr, "\n \n g^{-1} = ");
	for (int j = 0; j <  twists_length; j++) {
        fprintf(stderr, "%s.", twists_word2[j]);
    }

	fprintf(stderr, "\n \n\\overline g = ");
	for (int j = 0; j <  twists_length; j++) {
        fprintf(stderr, "%s.", twists_word1[j]);
    }

	fprintf(stderr, "\n \n f^2 = g^{-1}\\overline g = ");
	for (int j = 0; j <  twists_length; j++) {
        fprintf(stderr, "%s.", twists_word2[j]);
    }
	for (int j = 0; j <  twists_length; j++) {
        fprintf(stderr, "%s.", twists_word1[j]);
    }

	exit(0);
}



void s(int *arr1, int *arr2, int *size, int i) { 
	 //j=1 since skipping start and <*size-2 since skipping end
	 //moving right:
	for (int j = 1; j < *size - 2; j++) { //*size - 1 increments after pointing, *size-- incremenents before pointing
		//fprintf(stderr, "made it");
        if (arr1[j] == i && arr2[j] == 1 && arr1[j + 1] == i + 1 && arr2[j + 1] == 0) {
            arr2[j] = 0;
            arr2[j + 1] = 1;
			continue;
        }
		if (arr1[j] == i && arr2[j] == 0 && arr1[j + 1] == i + 1 && arr2[j + 1] == 1) {
            // Make room for four new entries
            for (int k = *size + 3; k >= j + 5; k--) {
                arr1[k] = arr1[k - 4];
                arr2[k] = arr2[k - 4];
            }
            // Insert the new sequence
            arr1[j + 1] = i + 1; arr2[j + 1] = 0;
            arr1[j + 2] = i + 1; arr2[j + 2] = 1;
            arr1[j + 3] = i; arr2[j + 3] = 0;
            arr1[j + 4] = i; arr2[j + 4] = 1;
			j+=4;
			(*size) +=4;
			continue;
		}

		//moving left:
		if (arr1[j+1] == i && arr2[j+1] == 1 && arr1[j] == i + 1 && arr2[j] == 0) {
            arr2[j+1] = 0;
            arr2[j] = 1;
			continue;
        }
		if (arr1[j+1] == i && arr2[j+1] == 0 && arr1[j] == i + 1 && arr2[j] == 1) {
            // Make room for four new entries
            for (int k = *size + 3; k >= j + 5; k--) {
                arr1[k] = arr1[k - 4];
                arr2[k] = arr2[k - 4];
            }
            // Insert the new sequence
            arr1[j + 4] = i + 1; arr2[j + 4] = 0;
            arr1[j + 3] = i + 1; arr2[j + 3] = 1;
            arr1[j + 2] = i; arr2[j + 2] = 0;
            arr1[j + 1] = i; arr2[j + 1] = 1;
			j+=4;
			(*size) +=4;
			continue;
		}
		//turning around
		if(arr1[j] == i && arr1[j+1] == i && arr1[j+2] == i - 1){
			int first = arr2[j];		
			int second = arr2[j+1];
			for (int k = *size + 1; k >= j + 4; k--) {
                arr1[k] = arr1[k - 2];
                arr2[k] = arr2[k - 2];
            }
            // Insert the new sequence
            arr1[j] = i ; arr2[j] = 0;
			arr1[j+1] = i+1 ; arr2[j+1] = first;
			arr1[j+2] = i+1 ; arr2[j+2] = second;
            arr1[j+3] = i ; arr2[j + 3] = 0;
			j+=4;
			(*size) +=2;
			continue;
		}
		//turning around
		if(arr1[j] == i+1 && arr1[j+1] == i+1  && arr1[j+2] == i + 2){
			int first = arr2[j];		
			int second = arr2[j+1];
			for (int k = *size + 1; k >= j + 4; k--) {
                arr1[k] = arr1[k - 2];
                arr2[k] = arr2[k - 2];
            }
            // Insert the new sequence
            arr1[j] = i+1 ; arr2[j] = 1;
			arr1[j+1] = i ; arr2[j+1] = first;
			arr1[j+2] = i ; arr2[j+2] = second;
            arr1[j+3] = i+1 ; arr2[j + 3] = 1;
			j+=4;
			(*size) +=2;
			continue;
		}
    } 

	//specific case for size = 2:
	if(arr1[0]==i && arr1[1]==i+1 && *size == 2){
		arr1[0] = i+1;
		arr1[1] = i;
		//fprintf(stderr, "welp");
		return; //ethan added mar 31
	} else if(arr1[0]==i+1 && arr1[1]==i && *size == 2){
		arr1[0] = i;
		arr1[1] = i+1;
		//fprintf(stderr, "welp");
		return; //ethan added mar 31
	}

	//change end of path before first move becuase changing size doesn't affect first move.
	//works the same as changing start of path
	if(arr1[*size-1]==i && arr1[*size-2]==i+1 && arr2[*size-2]== 0 ){
		arr2[*size-2] = -1000;
    	(*size)--;
	 } else if(arr1[*size-1]==i && arr1[*size-2]==i+1 && arr2[*size-2]== 1){//ethan changed this
    	arr1[*size+1] = i+1; arr2[*size+1] = -1000;
		arr1[*size] = i; arr2[*size] = 0;
		arr1[*size-1] = i; arr2[*size-1] = 1;
    	(*size) += 2;
	 }else if(arr1[*size-1]==i && arr1[*size-2]==i-1){ //SURE THIS WORKS???
		/*for (int i = *size; i > 0; i--) {
			arr1[i] = arr1[i - 1];
			arr2[i] = arr2[i - 1];
		}*/
    	arr1[*size] = i+1; arr2[*size] = -1000;
		arr1[*size-1] = i; arr2[*size-1] = 0;
    	(*size) += 1;
	 } 
	//accounting for ith spot = i+1
	 else if (arr1[*size-2]==i && arr1[*size-1]==i+1 && arr2[*size-2]== 1 ){ 	
		arr2[*size-2] = -1000;
    	(*size)--;
	 } else if(arr1[*size-2]==i && arr1[*size-1]==i+1 && arr2[1]== 0 ){
    	arr1[*size+1] = i; arr2[*size+1] = -1000;
		arr1[*size] = i+1; arr2[*size] = 1;
		arr1[*size-1] = i+1; arr2[*size-1] = 0;
    	(*size) += 2;
	 } 
	 else if(arr1[*size-2]==i+2 && arr1[*size-1]==i+1){
    	arr1[*size] = i; arr2[*size] = -1000;
		arr1[*size-1] = i+1; arr2[*size-1] = 1;
    	(*size) += 1;
	 }

	//first moves are different: need to test these more!
	//accounting for ith spot = start
	 if(arr1[0]==i && arr1[1]==i+1 && arr2[1]== 0 ){
		for (int i = 0; i < *size - 1; i++) {
        arr1[i] = arr1[i + 1];
		arr2[i] = arr2[i + 1];
    	}
		arr2[0] = -1000;
    	(*size)--;
	 } else if(arr1[0]==i && arr1[1]==i+1 && arr2[1]== 1 ){
		for (int i = *size + 1; i > 0; i--) {
			arr1[i] = arr1[i - 2];
			arr2[i] = arr2[i - 2];
		}
    	// Add the new entry to the beginning
    	arr1[0] = i+1; arr2[0] = -1000;
		arr1[1] = i; arr2[1] = 0;
		arr1[2] = i; arr2[2] = 1;
    	(*size) += 2;
	 }else if(arr1[0]==i && arr1[1]==i-1){
		for (int i = *size; i > 0; i--) {
			arr1[i] = arr1[i - 1];
			arr2[i] = arr2[i - 1];
		}
    	// Add the new entry to the beginning
    	arr1[0] = i+1; arr2[0] = -1000;
		arr1[1] = i; arr2[1] = 0;
    	(*size) += 1;
	 }

	 //accounting for ith spot = i+1
	 else if(arr1[1]==i && arr1[0]==i+1 && arr2[1]== 1 ){
		for (int i = 0; i < *size - 1; i++) {
        	arr1[i] = arr1[i + 1];
			arr2[i] = arr2[i + 1];
    	}
		arr2[0] = -1000;
    	(*size)--;
	 } else if(arr1[1]==i && arr1[0]==i+1 && arr2[1]== 0 ){
		for (int i = *size + 1; i > 0; i--) {
			arr1[i] = arr1[i - 2];
			arr2[i] = arr2[i - 2];
		}
    	// Add the new entry to the beginning
    	arr1[0] = i; arr2[0] = -1000;
		arr1[1] = i+1; arr2[1] = 1;
		arr1[2] = i+1; arr2[2] = 0;
    	(*size) += 2;
	 } 
	 else if(arr1[1]==i+2 && arr1[0]==i+1){
		for (int i = *size + 1; i > 0; i--) {
			arr1[i] = arr1[i - 1];
			arr2[i] = arr2[i - 1];
		}
    	// Add the new entry to the beginning
    	arr1[0] = i; arr2[0] = -1000;
		arr1[1] = i+1; arr2[1] = 1;
    	(*size) += 1;
	 }

	 //fully simplify result

	fully_simplify_path(arr1, arr2, size);
}

void S(int *arr1, int *arr2, int *size, int i){
	//flip
	for(int i = 1; i < *size-1; i++){
		if(arr2[i]==0){
			arr2[i]=1;

		} else if(arr2[i]==1){
			arr2[i]=0;
		}
	}
	
	//apply s
	s(arr1, arr2, size, i);

	//flip
	for(int i = 1; i < *size-1; i++){
		if(arr2[i]==0){
			arr2[i]=1;
		} else if(arr2[i]==1){
			//fprintf(stderr, "flipping");
			arr2[i]=0;
		}
	}
}


bool reducing_move(int *arr1, int *arr2, int *size, int *twistsy, int *len) {
    bool simplified = false;

	//fprintf(stderr, "size is %d\n", *size);

	//array should never have size 0 or 1
	if(*size < 2){
		fprintf(stderr, "ERROR: ARRAY TOO SMALL");
	}

	//return false if array is fully simplified
	if(*size == 2){
		return simplified;
	}

	//simplify otherwise
	if(arr1[0] < arr1[1] && arr2[1] == 0){
		//apply s_{arr1[0]}, append s_{arr1[0]} to twist array
		twistsy[*len] = arr1[0];
		s(arr1, arr2, size, arr1[0]);
		(*len)++;
		simplified = true;
		return simplified;

	}
	if(arr1[0] < arr1[1] && arr2[1] == 1){
		//apply + append S_{arr1[0]}
		twistsy[*len] = -arr1[0]-1;
		S(arr1, arr2, size, arr1[0]); //STILL MUST DEFINE S
		(*len)++;
		simplified = true;
		return simplified;
	}
	if(arr1[0] > arr1[1] && arr2[1] == 1){
		//apply + append s_{arr1[1]}
		twistsy[*len] = arr1[1];
		s(arr1, arr2, size, arr1[1]);
		(*len)++;
		simplified = true;
		return simplified;
	}
	if(arr1[0] > arr1[1] && arr2[1] == 0){
		//apply + append S_{arr1[1]}
		twistsy[*len] = -arr1[1]-1;
		S(arr1, arr2, size, arr1[1]);
		(*len)++;
		simplified = true;
		return simplified;
	}
    fprintf(stderr, "ERROR: simplification failed");
	return simplified;
}


void star_algorithm(int *arr1, int *arr2, int *size, int *twistsy, int *len, int k){
	//reverse array
	// do this because the rest of the program is coded to move the starting point of the path, but it's simpler to move the ending points.
	reverse_array(arr1, *size);
	reverse_array(arr2, *size);

	bool simplified;
    do {
		//fprintf(stderr, "size is %d\n", *size);
        simplified = reducing_move(arr1, arr2, size, twistsy, len);
    } while (simplified && *size > 2);

	while(arr1[*size-1] != k || arr1[0] != k+1){
		//Moving starting point to the left of k or at k
		if(arr1[*size-1]>k){
			twistsy[*len] = arr1[*size-1]-1;
			s(arr1, arr2, size, arr1[*size-1]-1);
			(*len)++;
			continue;
		}

		//Put ending point at k+1. Starting point might move, but only left.
		if(arr1[0]< k+1){
			twistsy[*len] = arr1[0];
			s(arr1, arr2, size, arr1[0]);
			(*len)++;
			continue;
		} 

		if(arr1[0]> k+1){
			twistsy[*len] = arr1[0]-1;
			s(arr1, arr2, size, arr1[0]-1);
			(*len)++;
			continue;
		}

		//move starting point to k if it's not already there
		if(arr1[*size-1]< k){
			twistsy[*len] = arr1[*size-1];
			s(arr1, arr2, size, arr1[arr1[*size-1]]);
			(*len)++;
			continue;
		}
	}
	reverse_array(arr1, *size);
	reverse_array(arr2, *size);

}

void delete_indices(int *arr1, int *arr2, int *size, int *arr1_temp, int *arr2_temp, int k, int *len) {
	int write_index = 0;
	//important that path starts from right side of contracted segment
    for (int i = 0; i < *size; i++) {
        if (arr1[i] >= k) { 
            arr1_temp[write_index] = arr1[i];
            arr2_temp[write_index] = arr2[i];
            write_index++;
        }
    }
    *len = write_index;
}



void process_array(int *arr, int *size, int k) {
	int location = k; //location of mega-strand
	for(int i = 0; i < *size; i++) {
		if(arr[i]>=0){
			if(arr[i] == location){
				for(int j = *size - 1 + k; j > i + k ; j--){
					arr[j] = arr[j - k];
				}
				for(int j = i+1; j <= i + k; j++){
					arr[j] = location-(j-i);
				}
				(*size) += k;
				i+=k;
				location++;
				continue;
			} else if(arr[i] == location-1){	
				for(int j = *size - 1 + k; j >= i + k ; j--){
					arr[j] = arr[j - k];
				}
				for(int j = 0; j < k; j++){
					arr[i + j] = location-1-k + j;
				}
				(*size) += k;
				i+=k;
				location--;
				continue;
			}
		} else if(arr[i]<0){
			if(-arr[i]-1 == location){
				for(int j = *size - 1 + k; j > i + k ; j--){
					arr[j] = arr[j - k];
				}
				for(int j = i+1; j <= i + k; j++){
					arr[j] = -location-1+(j-i); 
				}
				i+=k;
				(*size) += k;
				location++;
				continue;
			} else if(-arr[i]-1 == location-1){	
				for(int j = *size - 1 + k; j >= i + k ; j--){
					arr[j] = arr[j - k];
				}
				for(int j = i; j < i + k; j++){
					arr[j] = -location +k - (j-i); 
				}
				(*size) += k;
				i+=k;
				location--;
				continue;
			}
		}
	}
}

void simplify_final_array(int *arr, int *size){
    bool simplified;
    do {
        simplified = bool_simplify_final_array(arr, size);
    } while (simplified);
}

bool bool_simplify_final_array(int *arr, int *size){
	bool simplified = false;
	for (int i = 0; i < *size-1 ; i++) {
		// If consecutive terms add up to -1, skip them
		if (arr[i] + arr[i + 1] == -1) {
			//delete array elements
			for(int j = i; j < *size - 2; j++){
				arr[j] = arr[j + 2];
			}
			*size = (*size) -2;
			simplified = true;
			return simplified;
		}
	}
	return simplified;
}


void reverse_array(int *arr, int size){
	for (int i = 0; i < size / 2; i++) {
        int temp = arr[i];
        arr[i] = arr[size - 1 - i];
        arr[size - 1 - i] = temp;
	}
}



bool simplify_path(int *arr1, int *arr2, int *size) {
    bool simplified = false;
	if ((arr1)[0] == (arr1)[1]){
			for (int j = 1; j < *size - 1; j++) {
                (arr1)[j] = (arr1)[j + 1];
                (arr2)[j] = (arr2)[j + 1];
            }
            (*size) -= 1;
            simplified = true;
			return simplified;
	}

	if ((arr1)[*size - 1] == (arr1)[*size - 2]){
			arr2[*size - 2] = -1000;
            (*size) -= 1;
            simplified = true;
			return simplified;
	}

    for (int i = 0; i < *size - 2; i++) {
        if ((arr1)[i] == (arr1)[i+1] && (arr2)[i] == (arr2)[i+1]) {
            for (int j = i; j < *size - 2; j++) {
                (arr1)[j] = (arr1)[j + 2];
                (arr2)[j] = (arr2)[j + 2];
            }
            *size -= 2;
            simplified = true;
            return simplified;;
        }
    }
    return simplified;
}

void fully_simplify_path(int *arr1, int *arr2, int *size) {
    bool simplified;
    do {
        simplified = simplify_path(arr1, arr2, size);
    } while (simplified && *size > 2);
}


//TESTING CORNER

	//test 1 passes. Answer: s0.s1.s2.s2.S1.
	/*
	int a_array_pos[60][60] = {
    {1, 2, 3, 3, 2, 1, 0},
    {0, 1, 2},
    {2, 1, 1, 2, 3}
    // The rest will be initialized to 0
	};
	int a_array_height[60][60] = {
		{-1000, 1, 0, 1, 1, 1, -1000},
		{-1000, 0, -1000},
		{-1000, 0, 1, 1, -1000}	
		// The rest will be initialized to 0
	};
	path_lengths[0] = 7;
	path_lengths[1] = 3;
	path_lengths[2] = 5;
	path_lengths_testing[0] = 7;
	path_lengths_testing[1] = 3;
	path_lengths_testing[2] = 5;
	

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 60; j++) {
			array_pos[i][j] = a_array_pos[i][j];
			array_pos_testing[i][j] = a_array_pos[i][j];
			array_height[i][j] = a_array_height[i][j];
			array_height_testing[i][j] = a_array_height[i][j];
		}
	}*/
	//test 2 passes!
	//outputes s_2.s_1.s_0.s_0.s_1.s_2.s_1.s_0.s_0.s_1.s_2.s_1.s_0.s_1.S_2.S_2.s_1
	//solution is S_0s_2s_1S_1s_0S_2s_1
	// check they are the same in SB4 (!!) using:
	/*w = 'S_0s_2s_1S_2s_0S_2s_1'
	q = 's_2.s_1.s_0.s_0.s_1.s_2.s_1.s_0.s_0.s_1.s_2.s_1.s_0.s_1.S_2.S_2.s_1'
	W = S.mapping_class(w)
	Q = S.mapping_class(q)
	if(W == Q): print("yay")*/
	/*
	int a_array_pos[60][60] = {
		{1, 2, 3, 3, 2, 1, 1, 2},
		{2, 1, 1, 2, 3},
		{3, 2, 1, 1, 2, 3, 3, 2, 1, 0}
		// The rest will be initialized to 0
	};
	int a_array_height[60][60] = {
		{-1000, 0, 1, 0, 0, 0, 1, -1000},
		{-1000, 1, 0, 0, -1000},
		{-1000, 0, 0, 1, 0, 1, 0, 0, 0, -1000}	
		// The rest will be initialized to 0
	};
	path_lengths[0] = 8;
	path_lengths[1] = 5;
	path_lengths[2] = 10;
	path_lengths_testing[0] = 8;
	path_lengths_testing[1] = 5;
	path_lengths_testing[2] = 10;
	

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 60; j++) {
			array_pos[i][j] = a_array_pos[i][j];
			array_pos_testing[i][j] = a_array_pos[i][j];
			array_height[i][j] = a_array_height[i][j];
			array_height_testing[i][j] = a_array_height[i][j];
		}
	}*/

	//Test 3 tbd: 
	/*
	int a_array_pos[60][60] = {
		{0, 1},
		{1,2,3,4,4,3,2,2,3},
		{3,2},
		{2,3,3,2,2,3,4}
		// The rest will be initialized to 0
	};
	int a_array_height[60][60] = {
		{-1000, -1000},
		{-1000,1,1,1,2,1,1,0, -1000},
		{-1000, -1000},
		{-1000,1,0,0,1,1, -1000}
		// The rest will be initialized to 0
	};
	path_lengths[0] = 2;
	path_lengths[1] = 9;
	path_lengths[2] = 2;
	path_lengths[3] = 7;
	path_lengths_testing[0] = 2;
	path_lengths_testing[1] = 9;
	path_lengths_testing[2] = 2;
	path_lengths_testing[3] = 7;
	

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 60; j++) {
			array_pos[i][j] = a_array_pos[i][j];
			array_pos_testing[i][j] = a_array_pos[i][j];
			array_height[i][j] = a_array_height[i][j];
			array_height_testing[i][j] = a_array_height[i][j];
		}
	}
	*/