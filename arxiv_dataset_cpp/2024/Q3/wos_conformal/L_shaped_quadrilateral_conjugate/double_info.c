#include <stdio.h>
#include <float.h>

int main() {
    printf("Size of double: %zu bytes\n", sizeof(double));
    printf("Precision of double: %d decimal digits\n", DBL_DIG);
    printf("Minimum positive value of double: %e\n", DBL_MIN);
    printf("Maximum value of double: %e\n", DBL_MAX);
    printf("Epsilon value of double: %e\n", DBL_EPSILON);
    return 0;
}

