#include <stdio.h>
#include <stdlib.h>

// Structure to represent pairs of indices
typedef struct {
    int i;
    int j;
} IndexPair;

// Function to find and return pairs of indices (i, j) that satisfy the condition
IndexPair* findPairs(float X[], int X_length, float Y[], int Y_length, int* result_length) {
    int i = 0, j = 0;
    int pairCount = 0;
    
    // Allocate memory for the result array
    IndexPair* result = (IndexPair*)malloc(X_length * Y_length * sizeof(IndexPair));
    
    while (i < X_length && j < Y_length) {
        if (X[i] < Y[j]) {
            if (i > 0 && Y[j - 1] < X[i]) {
                // Store the pair of indices
                result[pairCount].i = i;
                result[pairCount].j = j;
                pairCount++;
            }
            i++;
        } else {
            j++;
        }
    }
    
    // Update the result length
    *result_length = pairCount;
    
    return result;
}

int main() {
    float X[] = {0.1, 0.3, 0.5, 0.7, 0.71, 0.9};
    float Y[] = {0.2, 0.4, 0.6, 0.8, 0.8, 0.81, 0.82, 1.0};
    
    int X_length = sizeof(X) / sizeof(X[0]);
    int Y_length = sizeof(Y) / sizeof(Y[0]);
    
    int result_length;
    IndexPair* pairs = findPairs(X, X_length, Y, Y_length, &result_length);
    
    // Print the pairs of indices
    for (int k = 0; k < result_length; k++) {
        printf("(%d, %d)\n", pairs[k].i, pairs[k].j);
    }
    
    // Free the dynamically allocated memory
    free(pairs);
    
    return 0;
}
