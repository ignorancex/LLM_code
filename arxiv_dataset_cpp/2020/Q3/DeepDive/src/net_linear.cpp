#include <hw/net_linear.hpp>

#define __SIMD_CORE_SIZE__ 160

#pragma SDS data zero_copy(                                                                                        \
    input_feature [0:__AVG_POOLING_INPUT_CHAN_SIZE__ * __AVG_POOLING_KERNEL_SIZE__ * __AVG_POOLING_KERNEL_SIZE__], \
    weights [0:__CLASSIFIER__WEIGHT_LENGTH__],                                                                     \
    biases [0:__LINEAR_ROW_SIZE__],                                                                                \
    out_0 [0:__LINEAR_ROW_SIZE__],                                                                                 \
    weight_zp [0:__CLASSIFIER_W_ZERO_POINT_LENGTH__],                                                              \
    iMult_bias_acc [0:__CLASSIFIER_B_SCALE_LENGTH__],                                                              \
    nShift_bias_acc [0:__CLASSIFIER_B_SCALE_LENGTH__],                                                             \
    iMult_output [0:__CLASSIFIER_B_SCALE_LENGTH__],                                                                \
    nShift_output [0:__CLASSIFIER_B_SCALE_LENGTH__])

#pragma SDS data access_pattern( \
    input_feature                \
    : SEQUENTIAL,                \
      weights                    \
    : SEQUENTIAL,                \
      biases                     \
    : SEQUENTIAL,                \
      out_0                      \
    : SEQUENTIAL,                \
      weight_zp                  \
    : SEQUENTIAL,                \
      iMult_bias_acc             \
    : SEQUENTIAL,                \
      nShift_bias_acc            \
    : SEQUENTIAL,                \
      iMult_output               \
    : SEQUENTIAL,                \
      nShift_output              \
    : SEQUENTIAL)

void compute_linear(dType_8u *input_feature,
                    dType_8u *weights,
                    dType_8u *biases,
                    dType_8u *out_0,

                    dType_8u *weight_zp,
                    dType_8u bias_zp,

                    dType_8u *iMult_bias_acc,
                    dType_8t *nShift_bias_acc,
                    dType_8u *iMult_output,
                    dType_8u *nShift_output,

                    //opzp
                    dType_8u input_zp_linear,
                    dType_8u output_zp_linear)
{
  dType_4u weights_linear[__LINEAR_ROW_SIZE__][__LINEAR_DEPTH_SIZE__];
#pragma HLS array_partition variable = weights_linear cyclic factor = 160 dim = 2

  // Layer 1
  dType_4u biases_local[__CLASSIFIER_BASE_B_Q_LENGTH__];
  dType_4u weights_zp[__CLASSIFIER_BASE_B_Q_LENGTH__];
  dType_8u downscaling_iMult[__CLASSIFIER_BASE_B_Q_LENGTH__];
  dType_8u downscaling_nShift[__CLASSIFIER_BASE_B_Q_LENGTH__];
  dType_8t nShif_biases_acc[__CLASSIFIER_BASE_B_Q_LENGTH__];
  dType_8u iMult_biases_acc[__CLASSIFIER_BASE_B_Q_LENGTH__];

  fused_pipes Pipe_0("TDDR_LINEAR");
  fused_pipes Pipe_1("TLINEAR");
  weight_pipes wPipe("TWEIGHT");

#pragma HLS DATAFLOW

  burst_read_weights_linear<__LINEAR_ROW_SIZE__, __LINEAR_DEPTH_SIZE__>(weights, weights_linear, __CLASSIFIER__WEIGHT_LENGTH__);

  copy_biases_linear<__CLASSIFIER_BASE_B_Q_LENGTH__, __CLASSIFIER_BASE_B_Q_OFFSET__>(biases, biases_local);

  copy_weight_zp_linear<__CLASSIFIER_W_ZERO_POINT_LENGTH__, __CLASSIFIER_W_ZERO_POINT_OFFSET__>(weight_zp, weights_zp);

  copy_iMult_downscaling_linear<__CLASSIFIER_ACCUM_SCALE_LENGTH__, __CLASSIFIER_ACCUM_SCALE_OFFSET__>(iMult_output, downscaling_iMult);

  copy_nShift_downscaling_linear<__CLASSIFIER_ACCUM_SCALE_LENGTH__, __CLASSIFIER_ACCUM_SCALE_OFFSET__>(nShift_output, downscaling_nShift);

  copy_iMult_upscaling_linear<__CLASSIFIER_B_SCALE_LENGTH__, __CLASSIFIER_B_SCALE_OFFSET__>(iMult_bias_acc, iMult_biases_acc);

  copy_nShift_upscaling_linear<__CLASSIFIER_B_SCALE_LENGTH__, __CLASSIFIER_B_SCALE_OFFSET__>(nShift_bias_acc, nShif_biases_acc);

  readImage<__LINEAR_DEPTH_SIZE__, 1>(input_feature, Pipe_0);
  // readWeight<__LINEAR_DEPTH_SIZE__, __LINEAR_ROW_SIZE__>(weights, wPipe);

  Linear<__LINEAR_ROW_SIZE__, __LINEAR_DEPTH_SIZE__, __SIMD_CORE_SIZE__>(Pipe_0, weights_linear, Pipe_1, weights_zp,
                                                          biases_local, bias_zp, iMult_biases_acc, nShif_biases_acc,
                                                          downscaling_iMult, downscaling_nShift, input_zp_linear, output_zp_linear);

  writeData<fused_pipes, dType_8u, __LINEAR_ROW_SIZE__, 1>(out_0, Pipe_1);
}