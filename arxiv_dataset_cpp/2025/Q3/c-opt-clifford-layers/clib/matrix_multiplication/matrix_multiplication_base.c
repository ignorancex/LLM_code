// Since randn gives by default float32 which is also
// the default everywhere in the codebase, we use float32
#include "affine_forward_base.h"

void affine_forward(float* g, int dim, int n_blades, int in_channels, int out_channels, float* weight, float* bias, float* x, int batches, float* output) {
    affine_forward_base(g, dim, n_blades, in_channels, out_channels, weight, bias, x, batches, output);
}