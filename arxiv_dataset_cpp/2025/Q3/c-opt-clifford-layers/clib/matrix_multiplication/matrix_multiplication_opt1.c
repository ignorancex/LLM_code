#include "affine_forward_opt1.h"

void affine_forward(float* g, int dim, int n_blades, int in_channels, int out_channels, float* weight, float* bias, float* x, int batches, float* output) {
    affine_forward_opt1(g, dim, n_blades, in_channels, out_channels, weight, bias, x, batches, output);
}