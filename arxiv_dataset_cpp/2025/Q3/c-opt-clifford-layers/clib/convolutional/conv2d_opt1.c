#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#define B 8

float* vtranspose2(float* x, int n_batches, int in_channels, int d1, int d2) {
    float* newx = (float*) malloc(n_batches * in_channels * d1 * d2 * 4 * sizeof(float));
    assert(newx);
    // From (n_batches/B) * B * in_channels * d1 * d2 * 4 to in_channels * d1 * d2 * (n_batches/B) * 4 * B
    for(int batch_b=0;batch_b<n_batches/B;++batch_b) {
        for(int b=0;b<B;++b) {
            for(int in_channel=0;in_channel<in_channels;++in_channel) {
                for(int id1=0;id1<d1;++id1) {
                    for(int id2=0;id2<d2;++id2) {
                        newx[in_channel*d1*d2*n_batches*4 + id1*d2*n_batches*4 + id2*n_batches*4 + batch_b*B*4 + b] = 
                            x[batch_b*B*in_channels*d1*d2*4 + b*in_channels*d1*d2*4 + in_channel*d1*d2*4 + id1*d2*4 + id2*4];
                        newx[in_channel*d1*d2*n_batches*4 + id1*d2*n_batches*4 + id2*n_batches*4 + batch_b*B*4 + b + B] =
                            x[batch_b*B*in_channels*d1*d2*4 + b*in_channels*d1*d2*4 + in_channel*d1*d2*4 + id1*d2*4 + id2*4 + 1];
                        newx[in_channel*d1*d2*n_batches*4 + id1*d2*n_batches*4 + id2*n_batches*4 + batch_b*B*4 + b + 2*B] =
                            x[batch_b*B*in_channels*d1*d2*4 + b*in_channels*d1*d2*4 + in_channel*d1*d2*4 + id1*d2*4 + id2*4 + 2];
                        newx[in_channel*d1*d2*n_batches*4 + id1*d2*n_batches*4 + id2*n_batches*4 + batch_b*B*4 + b + 3*B] =
                            x[batch_b*B*in_channels*d1*d2*4 + b*in_channels*d1*d2*4 + in_channel*d1*d2*4 + id1*d2*4 + id2*4 + 3];
                    }
                }
            }
        }
    }
    return newx;
}

void conv2d(float* g, int n_batches, int in_channels, int d1, int d2, int out_channels, int filter_size, float* weight, float* bias, float* input, float* output) {
    assert(n_batches % B == 0);

    int filter_mem = filter_size*filter_size;

    float* x = vtranspose2(input, n_batches, in_channels, d1, d2);

    float* kernel=(float*) malloc(4 * out_channels * in_channels * filter_mem * sizeof(float));

    // From 4 * out_channels * in_channels * filter_size * filter_size to out_channels * in_channels * filter_size * filter_size * 4
    for(int out_channel=0;out_channel<out_channels;++out_channel) {
        for(int in_channel=0;in_channel<in_channels;++in_channel) {
            for(int u=0;u<filter_size;++u) {
                for(int v=0;v<filter_size;++v) {
                    kernel[out_channel*in_channels*filter_mem*4 + in_channel*filter_mem*4 + u*filter_size*4 + v*4] =
                        weight[out_channel*in_channels*filter_mem + in_channel*filter_mem + u*filter_size + v];
                    kernel[out_channel*in_channels*filter_mem*4 + in_channel*filter_mem*4 + u*filter_size*4 + v*4 + 1] =
                        weight[out_channels * in_channels * filter_mem + out_channel*in_channels*filter_mem + in_channel*filter_mem + u*filter_size + v];
                    kernel[out_channel*in_channels*filter_mem*4 + in_channel*filter_mem*4 + u*filter_size*4 + v*4 + 2] =
                        weight[2 * out_channels * in_channels * filter_mem + out_channel*in_channels*filter_mem + in_channel*filter_mem + u*filter_size + v];
                    kernel[out_channel*in_channels*filter_mem*4 + in_channel*filter_mem*4 + u*filter_size*4 + v*4 + 3] =
                        weight[3 * out_channels * in_channels * filter_mem + out_channel*in_channels*filter_mem + in_channel*filter_mem + u*filter_size + v];
                }
            }
        }
    }

    // x shape: in_channels * d1 * d2 * (n_batches/B) * 4 * B
    // kernel shape: out_channels * in_channels * filter_size * filter_size * 4
    int out_d1 = d1 - filter_size + 1;
    int out_d2 = d2 - filter_size + 1;
    // y shape: out_channels * out_d1 * out_d2 * (n_batches/B) * 4 * B
    // bias shape: 4*out_channels
    float* y = (float*) malloc(out_channels * out_d1 * n_batches * out_d2 * 4 * sizeof(float));
    for(int out_channel=0;out_channel<out_channels;++out_channel) {
        for(int id1=0;id1<out_d1;++id1) {
            for(int id2=0;id2<out_d2;++id2) {
                for(int batch_b=0;batch_b<n_batches/B;++batch_b) {
                    for(int b=0;b<B;++b) {
                        y[out_channel*out_d1*out_d2*n_batches*4 + id1*out_d2*n_batches*4 + id2*n_batches*4 + batch_b*B*4 + b] =
                            bias[out_channel];
                        y[out_channel*out_d1*out_d2*n_batches*4 + id1*out_d2*n_batches*4 + id2*n_batches*4 + batch_b*B*4 + b + B] =
                            bias[out_channels + out_channel];
                        y[out_channel*out_d1*out_d2*n_batches*4 + id1*out_d2*n_batches*4 + id2*n_batches*4 + batch_b*B*4 + b + 2*B] =
                            bias[2*out_channels + out_channel];
                        y[out_channel*out_d1*out_d2*n_batches*4 + id1*out_d2*n_batches*4 + id2*n_batches*4 + batch_b*B*4 + b + 3*B] =
                            bias[3*out_channels + out_channel];
                    }
                }
            }
        }
    }

    // printf("Output y:\n");
    // for(int i=0;i<out_channels * out_d1 * n_batches * out_d2 * 4;++i) {
    //     printf("%f ", y[i]);
    // }
    // printf("End\n");

    for(int out_channel=0;out_channel<out_channels;++out_channel) {
        for(int in_channel=0;in_channel<in_channels;++in_channel) {
            for(int id1=0;id1<d1;++id1) {
                for(int id2=0;id2<d2;++id2) {
                    for(int batch_b=0;batch_b<n_batches/B;++batch_b) {
                        float vec0[B], vec1[B], vec2[B], vec3[B];
                        for(int m=0;m<B;++m) {
                            vec0[m] = x[in_channel*d1*d2*n_batches*4 + id1*d2*n_batches*4 + id2*n_batches*4 + batch_b*B*4 + m];
                            vec1[m] = x[in_channel*d1*d2*n_batches*4 + id1*d2*n_batches*4 + id2*n_batches*4 + batch_b*B*4 + m + B];
                            vec2[m] = x[in_channel*d1*d2*n_batches*4 + id1*d2*n_batches*4 + id2*n_batches*4 + batch_b*B*4 + m + 2*B];
                            vec3[m] = x[in_channel*d1*d2*n_batches*4 + id1*d2*n_batches*4 + id2*n_batches*4 + batch_b*B*4 + m + 3*B];
                        }
                        for(int u=id1+filter_size<=d1?0:id1-d1+filter_size;u<filter_size && u<=id1;++u) {
                            for(int v=id2+filter_size<=d2?0:id2-d2+filter_size;v<filter_size && v<=id2;++v) {
                                int od1 = id1 - u;
                                int od2 = id2 - v;
                                float k0 = kernel[out_channel*in_channels*filter_mem*4 + in_channel*filter_mem*4 + u*filter_size*4 + v*4];
                                float k1 = kernel[out_channel*in_channels*filter_mem*4 + in_channel*filter_mem*4 + u*filter_size*4 + v*4 + 1];
                                float k2 = kernel[out_channel*in_channels*filter_mem*4 + in_channel*filter_mem*4 + u*filter_size*4 + v*4 + 2];
                                float k3 = kernel[out_channel*in_channels*filter_mem*4 + in_channel*filter_mem*4 + u*filter_size*4 + v*4 + 3];
                                float out0[B], out1[B], out2[B], out3[B];
                                for(int m=0;m<B;++m) {
                                    out0[m] = y[out_channel*out_d1*out_d2*n_batches*4 + od1*out_d2*n_batches*4 + od2*n_batches*4 + batch_b*B*4 + m];
                                    out1[m] = y[out_channel*out_d1*out_d2*n_batches*4 + od1*out_d2*n_batches*4 + od2*n_batches*4 + batch_b*B*4 + m + B];
                                    out2[m] = y[out_channel*out_d1*out_d2*n_batches*4 + od1*out_d2*n_batches*4 + od2*n_batches*4 + batch_b*B*4 + m + 2*B];
                                    out3[m] = y[out_channel*out_d1*out_d2*n_batches*4 + od1*out_d2*n_batches*4 + od2*n_batches*4 + batch_b*B*4 + m + 3*B];
                                }
                                for(int m=0;m<B;++m) {
                                    out0[m] += vec0[m] * k0;
                                    out0[m] += g[0] * vec1[m] * k1;
                                    out0[m] += g[1] * vec2[m] * k2;
                                    out0[m] -= g[0] * g[1] * vec3[m] * k3;

                                    out1[m] += vec0[m] * k1;
                                    out1[m] += vec1[m] * k0;
                                    out1[m] -= g[1] * vec2[m] * k3;
                                    out1[m] += g[1] * vec3[m] * k2;

                                    out2[m] += vec0[m] * k2;
                                    out2[m] += g[0] * vec1[m] * k3;
                                    out2[m] += vec2[m] * k0;
                                    out2[m] -= g[0] * vec3[m] * k1;

                                    out3[m] += vec0[m] * k3;
                                    out3[m] += vec1[m] * k2;
                                    out3[m] -= vec2[m] * k1;
                                    out3[m] += vec3[m] * k0;
                                }
                                for(int m=0;m<B;++m) {
                                    y[out_channel*out_d1*out_d2*n_batches*4 + od1*out_d2*n_batches*4 + od2*n_batches*4 + batch_b*B*4 + m] = out0[m];
                                    y[out_channel*out_d1*out_d2*n_batches*4 + od1*out_d2*n_batches*4 + od2*n_batches*4 + batch_b*B*4 + m + B] = out1[m];
                                    y[out_channel*out_d1*out_d2*n_batches*4 + od1*out_d2*n_batches*4 + od2*n_batches*4 + batch_b*B*4 + m + 2*B] = out2[m];
                                    y[out_channel*out_d1*out_d2*n_batches*4 + od1*out_d2*n_batches*4 + od2*n_batches*4 + batch_b*B*4 + m + 3*B] = out3[m];
                                    // printf("(%d, %d)%f\n", out_channel*out_d1*out_d2*n_batches*4 + od1*out_d2*n_batches*4 + od2*n_batches*4 + batch_b*B*4 + m + 3*B, out_channels * out_d1 * n_batches * out_d2 * 4, out3[m]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // printf("Output y:\n");
    // for(int i=0;i<out_channels * out_d1 * n_batches * out_d2 * 4;++i) {
    //     printf("%f ", y[i]);
    // }
    // printf("End\n");

    // Reshape y out_channels * out_d1 * out_d2 * (n_batches/B) * 4 * B to output (n_batches/B) * B * out_channels * out_d1 * out_d2 * 4
    for(int batch_b=0;batch_b<n_batches/B;++batch_b) {
        for(int b=0;b<B;++b) {
            for(int out_channel=0;out_channel<out_channels;++out_channel) {
                for(int id1=0;id1<out_d1;++id1) {
                    for(int id2=0;id2<out_d2;++id2) {
                        output[batch_b*B*out_channels*out_d1*out_d2*4 + b*out_channels*out_d1*out_d2*4 + out_channel*out_d1*out_d2*4 + id1*out_d2*4 + id2*4] = 
                            y[out_channel*out_d1*out_d2*n_batches*4 + id1*out_d2*n_batches*4 + id2*n_batches*4 + batch_b*B*4 + b];
                        output[batch_b*B*out_channels*out_d1*out_d2*4 + b*out_channels*out_d1*out_d2*4 + out_channel*out_d1*out_d2*4 + id1*out_d2*4 + id2*4 + 1] = 
                            y[out_channel*out_d1*out_d2*n_batches*4 + id1*out_d2*n_batches*4 + id2*n_batches*4 + batch_b*B*4 + b + B];
                        output[batch_b*B*out_channels*out_d1*out_d2*4 + b*out_channels*out_d1*out_d2*4 + out_channel*out_d1*out_d2*4 + id1*out_d2*4 + id2*4 + 2] = 
                            y[out_channel*out_d1*out_d2*n_batches*4 + id1*out_d2*n_batches*4 + id2*n_batches*4 + batch_b*B*4 + b + 2*B];
                        output[batch_b*B*out_channels*out_d1*out_d2*4 + b*out_channels*out_d1*out_d2*4 + out_channel*out_d1*out_d2*4 + id1*out_d2*4 + id2*4 + 3] = 
                            y[out_channel*out_d1*out_d2*n_batches*4 + id1*out_d2*n_batches*4 + id2*n_batches*4 + batch_b*B*4 + b + 3*B];
                    }
                }
            }
        }
    }

    free(x);
    free(y);
    free(kernel);
}