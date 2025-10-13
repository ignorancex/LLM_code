#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

// Transpose batches of 2D matrices
void transpose_batches(float* x, int batches, int n, int m) {
    float* newx = (float*) malloc(batches * n * m * sizeof(float));
    assert(newx);
    for(int i=0;i<batches;++i) {
        for(int j=0;j<n;++j) {
            for(int k=0;k<m;++k) {
                newx[i*n*m + k*n + j] = x[i*n*m + j*m + k];
            }
        }
    }
    for(int i=0;i<batches;++i) {
        for(int j=0;j<m;++j) {
            for(int k=0;k<n;++k) {
                x[i*n*m + j*n + k] = newx[i*n*m + j*n + k];
            }
        }
    }
    free(newx);
}

void conv1d(int out_channels, int in_channels, int filter_size, int d1, float* kernel, float* x, float* output, float* bias) {
    for (int i = 0; i < out_channels; ++i) {
        for (int j = 0; j < d1 - filter_size + 1; ++j) {
            output[i * (d1 - filter_size + 1) + j] = bias[i];
            for (int k = 0; k < in_channels; ++k) {
                for (int l = 0; l < filter_size; ++l) {
                    output[i * (d1 - filter_size + 1) + j] += kernel[i * in_channels * filter_size + k * filter_size + l] * x[k * d1 + j + l];
                }
            }
        }
    }
}

void conv2d(int out_channels, int in_channels, int filter_size, int d1, int d2, float* kernel, float* x, float* output, float* bias) {
    for (int i = 0; i < out_channels; ++i) {
        for (int j = 0; j < d1 - filter_size + 1; ++j) {
            for (int k = 0; k < d2 - filter_size + 1; ++k) {
                output[i * (d1 - filter_size + 1) * (d2 - filter_size + 1) + j * (d2 - filter_size + 1) + k] = bias[i];
                for (int l = 0; l < in_channels; ++l) {
                    for (int m = 0; m < filter_size; ++m) {
                        for (int n = 0; n < filter_size; ++n) {
                            output[i * (d1 - filter_size + 1) * (d2 - filter_size + 1) + j * (d2 - filter_size + 1) + k] += kernel[i * in_channels * filter_size * filter_size + l * filter_size * filter_size + m * filter_size + n] * x[l * d1 * d2 + (j + m) * d2 + (k + n)];
                        }
                    }
                }
            }
        }
    }
}

void conv3d(int out_channels, int in_channels, int filter_size, int d1, int d2, int d3, float* kernel, float* x, float* output, float* bias) {
    for (int i = 0; i < out_channels; ++i) {
        for (int j = 0; j < d1 - filter_size + 1; ++j) {
            for (int k = 0; k < d2 - filter_size + 1; ++k) {
                for (int l = 0; l < d3 - filter_size + 1; ++l) {
                    output[i * (d1 - filter_size + 1) * (d2 - filter_size + 1) * (d3 - filter_size + 1) + j * (d2 - filter_size + 1) * (d3 - filter_size + 1) + k * (d3 - filter_size + 1) + l] = bias[i];
                    for (int m = 0; m < in_channels; ++m) {
                        for (int n = 0; n < filter_size; ++n) {
                            for (int o = 0; o < filter_size; ++o) {
                                for (int p = 0; p < filter_size; ++p) {
                                    output[i * (d1 - filter_size + 1) * (d2 - filter_size + 1) * (d3 - filter_size + 1) + j * (d2 - filter_size + 1) * (d3 - filter_size + 1) + k * (d3 - filter_size + 1) + l] += kernel[i * in_channels * filter_size * filter_size * filter_size + m * filter_size * filter_size * filter_size + n * filter_size * filter_size + o * filter_size + p] * x[m * d1 * d2 * d3 + (j + n) * d2 * d3 + (k + o) * d3 + (l + p)];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void conv_base(float* g, int n_batches, int dim, int n_blades, int in_channels, int d1, int d2, int d3, int out_channels, int filter_size, float* weight, float* bias, float* x, float* output) {
    int x_mem_per_batch_blade = in_channels*d1;
    int output_mem_per_batch_blade = out_channels*(d1-filter_size+1);
    int filter_mem = filter_size;
    if(d2 != -1) {
        x_mem_per_batch_blade *= d2;
        output_mem_per_batch_blade *= (d2-filter_size+1);
        filter_mem *= filter_size;
    }
    if(d3 != -1) {
        x_mem_per_batch_blade *= d3;
        output_mem_per_batch_blade *= (d3-filter_size+1);
        filter_mem *= filter_size;
    }
    transpose_batches(x, n_batches, x_mem_per_batch_blade, n_blades);

    float* kernel=NULL;
    if(dim == 1) {
        kernel = (float*) malloc(2 * out_channels * 2 * in_channels * filter_mem * sizeof(float));
        assert(kernel); // Handle memory allocation failure
        for(int i=0;i<out_channels;++i) {
            for(int j=0;j<in_channels*filter_mem;++j) {
                kernel[i*2*in_channels*filter_mem + j] = weight[i*in_channels*filter_mem + j];
                kernel[i*2*in_channels*filter_mem + in_channels*filter_mem + j] = g[0] * weight[out_channels*in_channels*filter_mem + i*in_channels*filter_mem + j];
                kernel[out_channels*2*in_channels*filter_mem + i*2*in_channels*filter_mem + j] = weight[out_channels*in_channels*filter_mem + i*in_channels*filter_mem + j];
                kernel[out_channels*2*in_channels*filter_mem + i*2*in_channels*filter_mem + in_channels*filter_mem + j] = weight[i*in_channels*filter_mem + j];
            }
        }
    } else if(dim == 2) {
        kernel = (float*) malloc(4 * out_channels * 4 * in_channels * filter_mem * sizeof(float));
        assert(kernel); // Handle memory allocation failure
        for(int i = 0; i < out_channels; ++i) {
            for(int j = 0; j < in_channels*filter_mem; ++j) {
                // k0 row
                kernel[i * 4 * in_channels*filter_mem + j] = weight[i * in_channels*filter_mem + j]; // w[0]
                kernel[i * 4 * in_channels*filter_mem + in_channels*filter_mem + j] = g[0] * weight[out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // g[0] * w[1]
                kernel[i * 4 * in_channels*filter_mem + 2 * in_channels*filter_mem + j] = g[1] * weight[2 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // g[1] * w[2]
                kernel[i * 4 * in_channels*filter_mem + 3 * in_channels*filter_mem + j] = -g[0] * g[1] * weight[3 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // -g[0] * g[1] * w[3]
                
                // k1 row
                kernel[out_channels * 4 * in_channels*filter_mem + i * 4 * in_channels*filter_mem + j] = weight[out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[1]
                kernel[out_channels * 4 * in_channels*filter_mem + i * 4 * in_channels*filter_mem + in_channels*filter_mem + j] = weight[i * in_channels*filter_mem + j]; // w[0]
                kernel[out_channels * 4 * in_channels*filter_mem + i * 4 * in_channels*filter_mem + 2 * in_channels*filter_mem + j] = -g[1] * weight[3 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // -g[1] * w[3]
                kernel[out_channels * 4 * in_channels*filter_mem + i * 4 * in_channels*filter_mem + 3 * in_channels*filter_mem + j] = g[1] * weight[2 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // g[1] * w[2]
                
                // k2 row
                kernel[2 * out_channels * 4 * in_channels*filter_mem + i * 4 * in_channels*filter_mem + j] = weight[2 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[2]
                kernel[2 * out_channels * 4 * in_channels*filter_mem + i * 4 * in_channels*filter_mem + in_channels*filter_mem + j] = g[0] * weight[3 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // g[0] * w[3]
                kernel[2 * out_channels * 4 * in_channels*filter_mem + i * 4 * in_channels*filter_mem + 2 * in_channels*filter_mem + j] = weight[i * in_channels*filter_mem + j]; // w[0]
                kernel[2 * out_channels * 4 * in_channels*filter_mem + i * 4 * in_channels*filter_mem + 3 * in_channels*filter_mem + j] = -g[0] * weight[out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // -g[0] * w[1]
                
                // k3 row
                kernel[3 * out_channels * 4 * in_channels*filter_mem + i * 4 * in_channels*filter_mem + j] = weight[3 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[3]
                kernel[3 * out_channels * 4 * in_channels*filter_mem + i * 4 * in_channels*filter_mem + in_channels*filter_mem + j] = weight[2 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[2]
                kernel[3 * out_channels * 4 * in_channels*filter_mem + i * 4 * in_channels*filter_mem + 2 * in_channels*filter_mem + j] = -weight[out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // -w[1]
                kernel[3 * out_channels * 4 * in_channels*filter_mem + i * 4 * in_channels*filter_mem + 3 * in_channels*filter_mem + j] = weight[i * in_channels*filter_mem + j]; // w[0]
            }
        }
    } else if(dim == 3) {
        kernel = (float*) malloc(8 * out_channels * 8 * in_channels * filter_mem * sizeof(float));
        assert(kernel); // Handle memory allocation failure

        // Build 3D kernel based on the provided PyTorch expressions
        for(int i = 0; i < out_channels; ++i) {
            for(int j = 0; j < in_channels*filter_mem; ++j) {
                // k0 row
                kernel[i * 8 * in_channels*filter_mem + 0 * in_channels*filter_mem + j] = weight[0 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[0]
                kernel[i * 8 * in_channels*filter_mem + 1 * in_channels*filter_mem + j] = weight[1 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[0]; // w[1] * g[0]
                kernel[i * 8 * in_channels*filter_mem + 2 * in_channels*filter_mem + j] = weight[2 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[1]; // w[2] * g[1]
                kernel[i * 8 * in_channels*filter_mem + 3 * in_channels*filter_mem + j] = weight[3 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[2]; // w[3] * g[2]
                kernel[i * 8 * in_channels*filter_mem + 4 * in_channels*filter_mem + j] = -weight[4 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[0] * g[1]; // -w[4] * g[0] * g[1]
                kernel[i * 8 * in_channels*filter_mem + 5 * in_channels*filter_mem + j] = -weight[5 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[0] * g[2]; // -w[5] * g[0] * g[2]
                kernel[i * 8 * in_channels*filter_mem + 6 * in_channels*filter_mem + j] = -weight[6 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[1] * g[2]; // -w[6] * g[1] * g[2]
                kernel[i * 8 * in_channels*filter_mem + 7 * in_channels*filter_mem + j] = -weight[7 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[0] * g[1] * g[2]; // -w[7] * g[0] * g[1] * g[2]

                // k1 row
                kernel[(1 * out_channels + i) * 8 * in_channels*filter_mem + 0 * in_channels*filter_mem + j] = weight[1 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[1]
                kernel[(1 * out_channels + i) * 8 * in_channels*filter_mem + 1 * in_channels*filter_mem + j] = weight[0 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[0]
                kernel[(1 * out_channels + i) * 8 * in_channels*filter_mem + 2 * in_channels*filter_mem + j] = -weight[4 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[1]; // -w[4] * g[1]
                kernel[(1 * out_channels + i) * 8 * in_channels*filter_mem + 3 * in_channels*filter_mem + j] = -weight[5 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[2]; // -w[5] * g[2]
                kernel[(1 * out_channels + i) * 8 * in_channels*filter_mem + 4 * in_channels*filter_mem + j] = weight[2 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[1]; // w[2] * g[1]
                kernel[(1 * out_channels + i) * 8 * in_channels*filter_mem + 5 * in_channels*filter_mem + j] = weight[3 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[2]; // w[3] * g[2]
                kernel[(1 * out_channels + i) * 8 * in_channels*filter_mem + 6 * in_channels*filter_mem + j] = -weight[7 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[1] * g[2]; // -w[7] * g[1] * g[2]
                kernel[(1 * out_channels + i) * 8 * in_channels*filter_mem + 7 * in_channels*filter_mem + j] = -weight[6 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[2] * g[1]; // -w[6] * g[2] * g[1]

                // k2 row
                kernel[(2 * out_channels + i) * 8 * in_channels*filter_mem + 0 * in_channels*filter_mem + j] = weight[2 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[2]
                kernel[(2 * out_channels + i) * 8 * in_channels*filter_mem + 1 * in_channels*filter_mem + j] = weight[4 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[0]; // w[4] * g[0]
                kernel[(2 * out_channels + i) * 8 * in_channels*filter_mem + 2 * in_channels*filter_mem + j] = weight[0 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[0]
                kernel[(2 * out_channels + i) * 8 * in_channels*filter_mem + 3 * in_channels*filter_mem + j] = -weight[6 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[2]; // -w[6] * g[2]
                kernel[(2 * out_channels + i) * 8 * in_channels*filter_mem + 4 * in_channels*filter_mem + j] = -weight[1 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[0]; // -w[1] * g[0]
                kernel[(2 * out_channels + i) * 8 * in_channels*filter_mem + 5 * in_channels*filter_mem + j] = weight[7 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[0] * g[2]; // w[7] * g[0] * g[2]
                kernel[(2 * out_channels + i) * 8 * in_channels*filter_mem + 6 * in_channels*filter_mem + j] = weight[3 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[2]; // w[3] * g[2]
                kernel[(2 * out_channels + i) * 8 * in_channels*filter_mem + 7 * in_channels*filter_mem + j] = weight[5 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[2] * g[0]; // w[5] * g[2] * g[0]

                // k3 row
                kernel[(3 * out_channels + i) * 8 * in_channels*filter_mem + 0 * in_channels*filter_mem + j] = weight[3 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[3]
                kernel[(3 * out_channels + i) * 8 * in_channels*filter_mem + 1 * in_channels*filter_mem + j] = weight[5 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[0]; // w[5] * g[0]
                kernel[(3 * out_channels + i) * 8 * in_channels*filter_mem + 2 * in_channels*filter_mem + j] = weight[6 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[1]; // w[6] * g[1]
                kernel[(3 * out_channels + i) * 8 * in_channels*filter_mem + 3 * in_channels*filter_mem + j] = weight[0 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[0]
                kernel[(3 * out_channels + i) * 8 * in_channels*filter_mem + 4 * in_channels*filter_mem + j] = -weight[7 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[0] * g[1]; // -w[7] * g[0] * g[1]
                kernel[(3 * out_channels + i) * 8 * in_channels*filter_mem + 5 * in_channels*filter_mem + j] = -weight[1 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[0]; // -w[1] * g[0]
                kernel[(3 * out_channels + i) * 8 * in_channels*filter_mem + 6 * in_channels*filter_mem + j] = -weight[2 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[1]; // -w[2] * g[1]
                kernel[(3 * out_channels + i) * 8 * in_channels*filter_mem + 7 * in_channels*filter_mem + j] = -weight[4 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[0] * g[1]; // -w[4] * g[0] * g[1]

                // k4 row
                kernel[(4 * out_channels + i) * 8 * in_channels*filter_mem + 0 * in_channels*filter_mem + j] = weight[4 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[4]
                kernel[(4 * out_channels + i) * 8 * in_channels*filter_mem + 1 * in_channels*filter_mem + j] = weight[2 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[2]
                kernel[(4 * out_channels + i) * 8 * in_channels*filter_mem + 2 * in_channels*filter_mem + j] = -weight[1 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // -w[1]
                kernel[(4 * out_channels + i) * 8 * in_channels*filter_mem + 3 * in_channels*filter_mem + j] = g[2] * weight[7 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // g[2] * w[7]
                kernel[(4 * out_channels + i) * 8 * in_channels*filter_mem + 4 * in_channels*filter_mem + j] = weight[0 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[0]
                kernel[(4 * out_channels + i) * 8 * in_channels*filter_mem + 5 * in_channels*filter_mem + j] = -weight[6 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[2]; // -w[6] * g[2]
                kernel[(4 * out_channels + i) * 8 * in_channels*filter_mem + 6 * in_channels*filter_mem + j] = weight[5 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[2]; // w[5] * g[2]
                kernel[(4 * out_channels + i) * 8 * in_channels*filter_mem + 7 * in_channels*filter_mem + j] = weight[3 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[2]; // w[3] * g[2]

                // k5 row
                kernel[(5 * out_channels + i) * 8 * in_channels*filter_mem + 0 * in_channels*filter_mem + j] = weight[5 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[5]
                kernel[(5 * out_channels + i) * 8 * in_channels*filter_mem + 1 * in_channels*filter_mem + j] = weight[3 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[3]
                kernel[(5 * out_channels + i) * 8 * in_channels*filter_mem + 2 * in_channels*filter_mem + j] = -weight[7 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[1]; // -w[7] * g[1]
                kernel[(5 * out_channels + i) * 8 * in_channels*filter_mem + 3 * in_channels*filter_mem + j] = -weight[1 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // -w[1]
                kernel[(5 * out_channels + i) * 8 * in_channels*filter_mem + 4 * in_channels*filter_mem + j] = weight[6 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[1]; // w[6] * g[1]
                kernel[(5 * out_channels + i) * 8 * in_channels*filter_mem + 5 * in_channels*filter_mem + j] = weight[0 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[0]
                kernel[(5 * out_channels + i) * 8 * in_channels*filter_mem + 6 * in_channels*filter_mem + j] = -weight[4 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[1]; // -w[4] * g[1]
                kernel[(5 * out_channels + i) * 8 * in_channels*filter_mem + 7 * in_channels*filter_mem + j] = -weight[2 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[1]; // -w[2] * g[1]

                // k6 row
                kernel[(6 * out_channels + i) * 8 * in_channels*filter_mem + 0 * in_channels*filter_mem + j] = weight[6 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[6]
                kernel[(6 * out_channels + i) * 8 * in_channels*filter_mem + 1 * in_channels*filter_mem + j] = weight[7 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[0]; // w[7] * g[0]
                kernel[(6 * out_channels + i) * 8 * in_channels*filter_mem + 2 * in_channels*filter_mem + j] = weight[3 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[3]
                kernel[(6 * out_channels + i) * 8 * in_channels*filter_mem + 3 * in_channels*filter_mem + j] = -weight[2 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // -w[2]
                kernel[(6 * out_channels + i) * 8 * in_channels*filter_mem + 4 * in_channels*filter_mem + j] = -weight[5 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[0]; // -w[5] * g[0]
                kernel[(6 * out_channels + i) * 8 * in_channels*filter_mem + 5 * in_channels*filter_mem + j] = weight[4 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[0]; // w[4] * g[0]
                kernel[(6 * out_channels + i) * 8 * in_channels*filter_mem + 6 * in_channels*filter_mem + j] = weight[0 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[0]
                kernel[(6 * out_channels + i) * 8 * in_channels*filter_mem + 7 * in_channels*filter_mem + j] = weight[1 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j] * g[0]; // w[1] * g[0]

                // k7 row
                kernel[(7 * out_channels + i) * 8 * in_channels*filter_mem + 0 * in_channels*filter_mem + j] = weight[7 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[7]
                kernel[(7 * out_channels + i) * 8 * in_channels*filter_mem + 1 * in_channels*filter_mem + j] = weight[6 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[6]
                kernel[(7 * out_channels + i) * 8 * in_channels*filter_mem + 2 * in_channels*filter_mem + j] = -weight[5 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // -w[5]
                kernel[(7 * out_channels + i) * 8 * in_channels*filter_mem + 3 * in_channels*filter_mem + j] = weight[4 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[4]
                kernel[(7 * out_channels + i) * 8 * in_channels*filter_mem + 4 * in_channels*filter_mem + j] = weight[3 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[3]
                kernel[(7 * out_channels + i) * 8 * in_channels*filter_mem + 5 * in_channels*filter_mem + j] = -weight[2 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // -w[2]
                kernel[(7 * out_channels + i) * 8 * in_channels*filter_mem + 6 * in_channels*filter_mem + j] = weight[1 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[1]
                kernel[(7 * out_channels + i) * 8 * in_channels*filter_mem + 7 * in_channels*filter_mem + j] = weight[0 * out_channels * in_channels*filter_mem + i * in_channels*filter_mem + j]; // w[0]
            }
        }
    }

    for(int batch=0;batch<n_batches;++batch) {
        if (d3 != -1) {
            conv3d(out_channels*n_blades, in_channels*n_blades, filter_size, d1, d2, d3, kernel, x+batch*x_mem_per_batch_blade*n_blades, output+batch*output_mem_per_batch_blade*n_blades, bias);
        } else if (d2 != -1) {
            conv2d(out_channels*n_blades, in_channels*n_blades, filter_size, d1, d2, kernel, x+batch*x_mem_per_batch_blade*n_blades, output+batch*output_mem_per_batch_blade*n_blades, bias);
        } else {
            conv1d(out_channels*n_blades, in_channels*n_blades, filter_size, d1, kernel, x+batch*x_mem_per_batch_blade*n_blades, output+batch*output_mem_per_batch_blade*n_blades, bias);
        }
    }
    
    transpose_batches(output, n_batches, n_blades, output_mem_per_batch_blade);
    transpose_batches(x, n_batches, n_blades, x_mem_per_batch_blade);
    free(kernel);
}