#include <stdio.h>

#include "arithmetic.h"
#include "audio_input_14287_posit32.h"
#include "fft.h"

int main() {
    struct complex x[SIGNAL_LENGTH];
    struct complex DFT[SIGNAL_LENGTH];

    // Convert input signal to complex numbers
    for (int i = 0; i < SIGNAL_LENGTH; ++i) {
        x[i].Re = audio_input_14287_posit32[i];
        x[i].Im = 0;
    }

    // Compute FFT
    rad2FFT(SIGNAL_LENGTH, x, DFT);

    // Print the real and imaginary parts of the result
    for (int i = 0; i < SIGNAL_LENGTH; ++i) {
        printf("%x %x\n", DFT[i].Re, DFT[i].Im);
    }

    return 0;
}