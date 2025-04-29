#include "dcl.h"

// Function to print the binary representation of a float
void printFloatBits(float f) {
    // Create an integer pointer to reinterpret the float as an integer
    uint32_t* ptr = (uint32_t*)&f;

    // Extract the integer value representing the float
    uint32_t floatAsInt = *ptr;

    // Print the binary representation of the float
    printf("Binary representation of %.10g (float):\n", f);

    // Print each bit (from MSB to LSB) of the integer value
    for (int i = 31; i >= 0; --i) {
        // Use bitwise AND to extract the i-th bit
        uint32_t bit = (floatAsInt >> i) & 1;
        printf("%u", bit);

        // Insert space for better readability
        if (i == 31 | i == 23) {
            printf(" ");  // Space after sign bit and exponent bits
        }

        // if (i % 8 == 0) {
        //     printf(" ");
        // }
    }

    printf("\n");
}



// Function to print the binary representation of a half
void printHalfBits(half f) {
    
    uint16_t* ptr = (uint16_t*)&f;
    printf("Binary representation of %.10g (half):\n", static_cast<float>(f));

    // Print each bit (from MSB to LSB) of the integer value
    for (int i = 15; i >= 0; --i) {
        
        uint16_t bit = ((*ptr) >> i) & 1;
        printf("%u", bit);

        // Insert space for better readability
        if (i == 15 | i == 10) {
            printf(" ");  // Space after sign bit and exponent bits
        }
    }

    printf("\n");
}


float randomFloat32() {
    // Generate a random 32-bit integer
    // must be a valid 32-bit floating point value

    while(true) {
        uint32_t randomInt = rand();
        float* ptr = (float*)&randomInt;
        ap_uint<32>* ptr_i = (ap_uint<32>*)&randomInt;
        
        if( ! ((*ptr_i).range(30, 23) == 0 | (*ptr_i).range(30, 23) == 255) ) {
            float randomFloat = *ptr;
            return randomFloat;
        }
    }
}



float randomFloatRange(float min, float max) {

    float min_value = min;
    float max_value = max;

    // Use random_device to obtain a seed for the random number engine
    std::random_device rd;
    
    // Use the Mersenne Twister 19937 engine
    std::mt19937 gen(rd());
    
    // Define a uniform distribution for floating-point numbers
    std::uniform_real_distribution<float> dis(min_value, max_value);

    // Generate a random floating-point number within the specified range
    float random_value = dis(gen);

    return random_value;

}

// Function to convert a 32-bit floating-point number to its binary representation
std::string floatToBinary(float num) {
    uint32_t numInt;
    std::memcpy(&numInt, &num, sizeof(uint32_t)); // Copy the float's bits to an integer
    std::string binaryStr;

    // Extract each bit from the integer representation
    for (int i = 31; i >= 0; --i) {
        binaryStr += ((numInt >> i) & 1) ? '1' : '0';
    }

    return binaryStr;
}



// Function to convert a 16-bit floating-point number to its binary representation
std::string floatToBinary(half num) {
    uint16_t numInt;
    std::memcpy(&numInt, &num, sizeof(uint16_t)); // Copy the float's bits to an integer
    std::string binaryStr;

    // Extract each bit from the integer representation
    for (int i = 15; i >= 0; --i) {
        binaryStr += ((numInt >> i) & 1) ? '1' : '0';
    }

    return binaryStr;
}


// Function to compare two 32-bit floating-point numbers bit by bit
bool compareFloatsBitwise(float a, float b) {
    // Convert floats to their binary representations
    std::string binaryA = floatToBinary(a);
    std::string binaryB = floatToBinary(b);

    // Compare the binary representations bit by bit
    if (binaryA == binaryB) {
        return true;
    } else {
        return false;
    }
}