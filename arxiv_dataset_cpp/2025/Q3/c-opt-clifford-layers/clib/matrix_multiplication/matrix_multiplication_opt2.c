#include "affine_forward_opt2.h"

void affine_forward(float* g, int dim, int n_blades, int in_channels, int out_channels, float* weight, float* bias, float* x, int batches, float* output) {
    if(dim == 1) {
        // affine_forward_opt1(g, dim, n_blades, in_channels, out_channels, weight, bias, x, batches, output);
        if(g[0] == -1.) {
            affine_forward_opt2_1d_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else { // g[0] == 1.
            affine_forward_opt2_1d_01(in_channels, out_channels, weight, bias, x, batches, output);
        }
    } else if(dim == 2) {
        // affine_forward_opt1(g, dim, n_blades, in_channels, out_channels, weight, bias, x, batches, output);
        if(g[0] == -1. && g[1] == -1.) {
            affine_forward_opt2_2d_11_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == 0.) {
            affine_forward_opt2_2d_11_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == 1.) {
            affine_forward_opt2_2d_11_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == -1.) {
            affine_forward_opt2_2d_00_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == 1.) {
            affine_forward_opt2_2d_00_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == -1.) {
            affine_forward_opt2_2d_01_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == 0.) {
            affine_forward_opt2_2d_01_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == 1.) {
            affine_forward_opt2_2d_01_01(in_channels, out_channels, weight, bias, x, batches, output);
        }
    } else if(dim == 3) {
        // affine_forward_opt1(g, dim, n_blades, in_channels, out_channels, weight, bias, x, batches, output);
        if(g[0] == -1. && g[1] == -1. && g[2] == -1.) {
            affine_forward_opt2_3d_11_11_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == -1. && g[2] == 0.) {
            affine_forward_opt2_3d_11_11_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == -1. && g[2] == 1.) {
            affine_forward_opt2_3d_11_11_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == 0. && g[2] == -1.) {
            affine_forward_opt2_3d_11_00_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == 0. && g[2] == 0.) {
            affine_forward_opt2_3d_11_00_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == 0. && g[2] == 1.) {
            affine_forward_opt2_3d_11_00_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == 1. && g[2] == -1.) {
            affine_forward_opt2_3d_11_01_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == 1. && g[2] == 0.) {
            affine_forward_opt2_3d_11_01_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == -1. && g[1] == 1. && g[2] == 1.) {
            affine_forward_opt2_3d_11_01_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == -1. && g[2] == -1.) {
            affine_forward_opt2_3d_00_11_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == -1. && g[2] == 0.) {
            affine_forward_opt2_3d_00_11_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == -1. && g[2] == 1.) {
            affine_forward_opt2_3d_00_11_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == 0. && g[2] == -1.) {
            affine_forward_opt2_3d_00_00_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == 0. && g[2] == 1.) {
            affine_forward_opt2_3d_00_00_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == 1. && g[2] == -1.) {
            affine_forward_opt2_3d_00_01_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == 1. && g[2] == 0.) {
            affine_forward_opt2_3d_00_01_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 0. && g[1] == 1. && g[2] == 1.) {
            affine_forward_opt2_3d_00_01_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == -1. && g[2] == -1.) {
            affine_forward_opt2_3d_01_11_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == -1. && g[2] == 0.) {
            affine_forward_opt2_3d_01_11_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == -1. && g[2] == 1.) {
            affine_forward_opt2_3d_01_11_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == 0. && g[2] == -1.) {
            affine_forward_opt2_3d_01_00_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == 0. && g[2] == 0.) {
            affine_forward_opt2_3d_01_00_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == 0. && g[2] == 1.) {
            affine_forward_opt2_3d_01_00_01(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == 1. && g[2] == -1.) {
            affine_forward_opt2_3d_01_01_11(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == 1. && g[2] == 0.) {
            affine_forward_opt2_3d_01_01_00(in_channels, out_channels, weight, bias, x, batches, output);
        } else if(g[0] == 1. && g[1] == 1. && g[2] == 1.) {
            affine_forward_opt2_3d_01_01_01(in_channels, out_channels, weight, bias, x, batches, output);
        }
    }
}