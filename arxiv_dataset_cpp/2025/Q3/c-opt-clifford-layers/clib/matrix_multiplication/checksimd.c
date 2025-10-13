#include <arm_neon.h>
#include <stdio.h>

int main(void) {
    float32x4_t a;
    ((float*)(&a))[0] = 0;
    ((float*)(&a))[1] = 1;
    ((float*)(&a))[2] = 2;
    ((float*)(&a))[3] = 3;
    float32x4_t b;
    ((float*)(&b))[0] = 4;
    ((float*)(&b))[1] = 5;
    ((float*)(&b))[2] = 6;
    ((float*)(&b))[3] = 7;
    float32x4_t c = vfmaq_f32(a, b, b);
    printf("c[0] = %f\n", ((float*)(&c))[0]);
    printf("c[1] = %f\n", ((float*)(&c))[1]);
    printf("c[2] = %f\n", ((float*)(&c))[2]);
    printf("c[3] = %f\n", ((float*)(&c))[3]);
    return 0;
}