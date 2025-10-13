#include "conv_opt3.h"

void conv1d(float* g, int n_batches, int in_channels, int d1, int out_channels, int filter_size, float* weight, float* bias, float* input, float* output) {
    if(g[0] == -1.) {
        conv_opt2_1d_11(n_batches, in_channels, d1, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 1.) {
        conv_opt2_1d_01(n_batches, in_channels, d1, out_channels, filter_size, weight, bias, input, output);
    }
}

void conv2d(float* g, int n_batches, int in_channels, int d1, int d2, int out_channels, int filter_size, float* weight, float* bias, float* input, float* output) {
    if(g[0] == -1. && g[1] == -1.) {
        conv_opt2_2d_11_11(n_batches, in_channels, d1, d2, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == -1. && g[1] == 0.) {
        conv_opt2_2d_11_00(n_batches, in_channels, d1, d2, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == -1. && g[1] == 1.) {
        conv_opt2_2d_11_01(n_batches, in_channels, d1, d2, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 0. && g[1] == -1.) {
        conv_opt2_2d_00_11(n_batches, in_channels, d1, d2, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 0. && g[1] == 1.) {
        conv_opt2_2d_00_01(n_batches, in_channels, d1, d2, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 1. && g[1] == -1.) {
        conv_opt2_2d_01_11(n_batches, in_channels, d1, d2, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 1. && g[1] == 0.) {
        conv_opt2_2d_01_00(n_batches, in_channels, d1, d2, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 1. && g[1] == 1.) {
        conv_opt2_2d_01_01(n_batches, in_channels, d1, d2, out_channels, filter_size, weight, bias, input, output);
    }
}

void conv3d(float* g, int n_batches, int in_channels, int d1, int d2, int d3, int out_channels, int filter_size, float* weight, float* bias, float* input, float* output) {
    if(g[0] == -1. && g[1] == -1. && g[2] == -1.) {
        conv_opt2_3d_11_11_11(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == -1. && g[1] == -1. && g[2] == 0.) {
        conv_opt2_3d_11_11_00(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == -1. && g[1] == -1. && g[2] == 1.) {
        conv_opt2_3d_11_11_01(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == -1. && g[1] == 0. && g[2] == -1.) {
        conv_opt2_3d_11_00_11(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == -1. && g[1] == 0. && g[2] == 0.) {
        conv_opt2_3d_11_00_00(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == -1. && g[1] == 0. && g[2] == 1.) {
        conv_opt2_3d_11_00_01(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == -1. && g[1] == 1. && g[2] == -1.) {
        conv_opt2_3d_11_01_11(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == -1. && g[1] == 1. && g[2] == 0.) {
        conv_opt2_3d_11_01_00(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == -1. && g[1] == 1. && g[2] == 1.) {
        conv_opt2_3d_11_01_01(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 0. && g[1] == -1. && g[2] == -1.) {
        conv_opt2_3d_00_11_11(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 0. && g[1] == -1. && g[2] == 0.) {
        conv_opt2_3d_00_11_00(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 0. && g[1] == -1. && g[2] == 1.) {
        conv_opt2_3d_00_11_01(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 0. && g[1] == 0. && g[2] == -1.) {
        conv_opt2_3d_00_00_11(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 0. && g[1] == 0. && g[2] == 1.) {
        conv_opt2_3d_00_00_01(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 0. && g[1] == 1. && g[2] == -1.) {
        conv_opt2_3d_00_01_11(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 0. && g[1] == 1. && g[2] == 0.) {
        conv_opt2_3d_00_01_00(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 0. && g[1] == 1. && g[2] == 1.) {
        conv_opt2_3d_00_01_01(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 1. && g[1] == -1. && g[2] == -1.) {
        conv_opt2_3d_01_11_11(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 1. && g[1] == -1. && g[2] == 0.) {
        conv_opt2_3d_01_11_00(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 1. && g[1] == -1. && g[2] == 1.) {
        conv_opt2_3d_01_11_01(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 1. && g[1] == 0. && g[2] == -1.) {
        conv_opt2_3d_01_00_11(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 1. && g[1] == 0. && g[2] == 0.) {
        conv_opt2_3d_01_00_00(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 1. && g[1] == 0. && g[2] == 1.) {
        conv_opt2_3d_01_00_01(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 1. && g[1] == 1. && g[2] == -1.) {
        conv_opt2_3d_01_01_11(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 1. && g[1] == 1. && g[2] == 0.) {
        conv_opt2_3d_01_01_00(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    } else if(g[0] == 1. && g[1] == 1. && g[2] == 1.) {
        conv_opt2_3d_01_01_01(n_batches, in_channels, d1, d2, d3, out_channels, filter_size, weight, bias, input, output);
    }
}