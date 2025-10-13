#include "affine_forward_opt3.h"
#include<string.h>
#include<stdlib.h>

void affine_forward(float* g, int dim, int n_blades, int in_channels, int out_channels, float* weight, float* bias, float* x, int batches, float* output) {
    int mem = n_blades * in_channels;
    if (n_blades * out_channels > mem) {
        mem = n_blades * out_channels;
    }

    float* buffer = (float*)malloc(mem * sizeof(float));
    for (int i=0;i<batches;++i) {
        for (int j=0;j<in_channels;++j) {
            for(int k=0;k<n_blades;++k) {
                buffer[k * in_channels + j] = x[i * in_channels * n_blades + j * n_blades + k];
            }
        }
        for (int j=0;j<n_blades;++j) {
            for (int k=0;k<in_channels;++k) {
                x[i * in_channels * n_blades + j * in_channels + k] = buffer[j * in_channels + k];
            }
        }
    }

    if(dim == 1) {
        if(g[0] == -1.) {
            affine_forward_opt3_1d_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else { // g[0] == 1.
            affine_forward_opt3_1d_01(in_channels, out_channels, weight, bias, x, batches, output);
        }
    } else if(dim == 2) {
        if(g[0] == -1. && g[1] == -1.) {
            affine_forward_opt3_2d_11_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == 0.) {
            affine_forward_opt3_2d_11_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == 1.) {
            affine_forward_opt3_2d_11_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == -1.) {
            affine_forward_opt3_2d_00_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == 1.) {
            affine_forward_opt3_2d_00_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == -1.) {
            affine_forward_opt3_2d_01_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == 0.) {
            affine_forward_opt3_2d_01_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == 1.) {
            affine_forward_opt3_2d_01_01(in_channels, out_channels, weight, bias, x, batches, output);
        }
    } else if(dim == 3) {
        if(g[0] == -1. && g[1] == -1. && g[2] == -1.) {
            affine_forward_opt3_3d_11_11_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == -1. && g[2] == 0.) {
            affine_forward_opt3_3d_11_11_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == -1. && g[2] == 1.) {
            affine_forward_opt3_3d_11_11_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == 0. && g[2] == -1.) {
            affine_forward_opt3_3d_11_00_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == 0. && g[2] == 0.) {
            affine_forward_opt3_3d_11_00_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == 0. && g[2] == 1.) {
            affine_forward_opt3_3d_11_00_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == 1. && g[2] == -1.) {
            affine_forward_opt3_3d_11_01_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == 1. && g[2] == 0.) {
            affine_forward_opt3_3d_11_01_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == 1. && g[2] == 1.) {
            affine_forward_opt3_3d_11_01_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == -1. && g[2] == -1.) {
            affine_forward_opt3_3d_00_11_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == -1. && g[2] == 0.) {
            affine_forward_opt3_3d_00_11_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == -1. && g[2] == 1.) {
            affine_forward_opt3_3d_00_11_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == 0. && g[2] == -1.) {
            affine_forward_opt3_3d_00_00_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == 0. && g[2] == 1.) {
            affine_forward_opt3_3d_00_00_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == 1. && g[2] == -1.) {
            affine_forward_opt3_3d_00_01_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == 1. && g[2] == 0.) {
            affine_forward_opt3_3d_00_01_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == 1. && g[2] == 1.) {
            affine_forward_opt3_3d_00_01_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == -1. && g[2] == -1.) {
            affine_forward_opt3_3d_01_11_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == -1. && g[2] == 0.) {
            affine_forward_opt3_3d_01_11_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == -1. && g[2] == 1.) {
            affine_forward_opt3_3d_01_11_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == 0. && g[2] == -1.) {
            affine_forward_opt3_3d_01_00_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == 0. && g[2] == 0.) {
            affine_forward_opt3_3d_01_00_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == 0. && g[2] == 1.) {
            affine_forward_opt3_3d_01_00_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == 1. && g[2] == -1.) {
            affine_forward_opt3_3d_01_01_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == 1. && g[2] == 0.) {
            affine_forward_opt3_3d_01_01_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == 1. && g[2] == 1.) {
            affine_forward_opt3_3d_01_01_01(in_channels, out_channels, weight, bias, x, batches, output);
        }
    }

    for (int i=0;i<batches;++i) {
        for (int j=0;j<out_channels;++j) {
            for(int k=0;k<n_blades;++k) {
                buffer[j * n_blades + k] = output[i * out_channels * n_blades + k * out_channels + j];
            }
        }
        for(int j=0;j<out_channels;++j) {
            for (int k=0;k<n_blades;++k) {
                output[i * out_channels * n_blades + j * n_blades + k] = buffer[j * n_blades + k];
            }
        }
    }

    for (int i=0;i<batches;++i) {
        for (int j=0;j<in_channels;++j) {
            for(int k=0;k<n_blades;++k) {
                buffer[j*n_blades + k] = x[i * in_channels * n_blades + k*in_channels + j];
            }
        }
        for (int j=0;j<in_channels;++j) {
            for (int k=0;k<n_blades;++k) {
                x[i * in_channels * n_blades + j * n_blades + k] = buffer[j * n_blades + k];
            }
        }
    }

    free(buffer);
}