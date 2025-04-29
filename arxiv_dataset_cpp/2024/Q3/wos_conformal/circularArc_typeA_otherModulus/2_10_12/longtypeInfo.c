#include <stdio.h>
#include <limits.h>

int main() {
    long int signed_long = 1234567890L;
    unsigned long int unsigned_long = 12345678901234567890UL;

    printf("Range of signed int:\n");
    printf("Minimum value: %d\n", INT_MIN);
    printf("Maximum value: %d\n", INT_MAX);

    printf("\nRange of unsigned int:\n");
    printf("Minimum value: %u\n", 0);  // Unsigned int minimum is always 0
    printf("Maximum value: %u\n", UINT_MAX);

    printf("Range of signed long int: %ld to %ld\n", LONG_MIN, LONG_MAX);
    printf("Range of unsigned long int: 0 to %lu\n", ULONG_MAX);

    return 0;
}

