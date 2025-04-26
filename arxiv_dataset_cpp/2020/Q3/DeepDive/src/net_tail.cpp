#include <hw/net_tail.hpp>

#define __CORE_SIZE_PW_TAIL__ 80
//---------------------------- TOP LEVEL FUNCTION -------------------------------------------//
void compute_tail(dType_8u *input_feature,
                  dType_8u *weights,
                  dType_8u *biases,
                  dType_8u *out_0,

                  dType_8u *weight_zp,
                  dType_8u *iMult_bias_acc,
                  dType_8t *nShift_bias_acc,
                  dType_8u *iMult_output,
                  dType_8u *nShift_output,

                  //opzp
                  dType_8u opzp1,
                  dType_8u opzp2,
                  dType_4u bzp_pw)
{

  dType_4u weights_pw[__CONV_0__OUTPUT_CHAN__][__CONV_0__INPUT_CHAN__];
  CORE_SIZE(weights_pw, __CORE_SIZE_PW_TAIL__, 2)

  dType_8u reshape[__AVG_POOLING_INPUT_CHAN_SIZE__ * __AVG_POOLING_KERNEL_SIZE__ * __AVG_POOLING_KERNEL_SIZE__];

  //Layer 1
  dType_4u biases_local_1[__CONV_0_BASE_B_Q_LENGTH__];
  dType_4u weights_zp_1[__CONV_0_W_ZERO_POINT_LENGTH__];
  dType_8u downscaling_iMult_1[__CONV_0_W_ZERO_POINT_LENGTH__];
  dType_8u downscaling_nShift_1[__CONV_0_W_ZERO_POINT_LENGTH__];
  dType_8t nShif_biases_acc_1[__CONV_0_W_ZERO_POINT_LENGTH__];
  dType_8u iMult_biases_acc_1[__CONV_0_W_ZERO_POINT_LENGTH__];

  fused_pipes Pipe_0("TDDR");
  fused_pipes Pipe_1("TPW");
  fused_pipes Pipe_2("AVG");

#pragma HLS DATAFLOW

  burst_read_weights_linear<__CONV_0__OUTPUT_CHAN__, __CONV_0__INPUT_CHAN__>(weights, weights_pw, __CONV_0__WEIGHT_LENGTH__);

  copy_biases_linear<__CONV_0_B_SCALE_LENGTH__, __CONV_0_B_SCALE_OFFSET__>(biases, biases_local_1);

  copy_weight_zp_linear<__CONV_0_W_ZERO_POINT_LENGTH__, __CONV_0_W_ZERO_POINT_OFFSET__>(weight_zp, weights_zp_1);

  copy_iMult_downscaling_linear<__CONV_0_ACCUM_SCALE_LENGTH__, __CONV_0_ACCUM_SCALE_OFFSET__>(iMult_output, downscaling_iMult_1);

  copy_nShift_downscaling_linear<__CONV_0_BASE_B_Q_LENGTH__, __CONV_0_BASE_B_Q_OFFSET__>(nShift_output, downscaling_nShift_1);

  copy_iMult_upscaling_linear<__CONV_0_B_SCALE_LENGTH__, __CONV_0_B_SCALE_OFFSET__>(iMult_bias_acc, iMult_biases_acc_1);

  copy_nShift_upscaling_linear<__CONV_0_BASE_B_Q_LENGTH__, __CONV_0_BASE_B_Q_OFFSET__>(nShift_bias_acc, nShif_biases_acc_1);

  readImage<__CONV_0__INPUT_CHAN__, __CONV_0__INPUT_SIZE__>(input_feature, Pipe_0);

  //Layer 1 - Pointwise Convolution - Channelwise Output
  PointWiseConvolution<ap_accuracy_truc_clip, dType_Reg,
                       __CONV_0__OUTPUT_CHAN__,
                       __CONV_0__INPUT_CHAN__,
                       __CORE_SIZE_PW_TAIL__,
                       fused_pipes, dType_8u,
                       ap_resource_dflt>(Pipe_0, Pipe_1, weights_pw, iMult_biases_acc_1,
                                         nShif_biases_acc_1, downscaling_iMult_1, downscaling_nShift_1,
                                         weights_zp_1, bzp_pw, opzp1, opzp2, biases_local_1,
                                         __CONV_0__OUTPUT_SIZE__, __CONV_0__INPUT_SIZE__,
                                         __CONV_0__OUTPUT_CHAN__, __CONV_0__INPUT_CHAN__, ap_accuracy_truc_clip(), ap_resource_dflt());

  //writeData<fused_pipes, dType_8u, __CONV_0__OUTPUT_CHAN__, __CONV_0__OUTPUT_SIZE__>(out_0, Pipe_1);

  channelWiseToColumnWise<__AVG_POOLING_INPUT_CHAN_SIZE__, __AVG_POOLING_KERNEL_SIZE__>(Pipe_1, reshape);

  Avg_Pooling<ap_accuracy_none, dType_16t, __AVG_POOLING_INPUT_CHAN_SIZE__,
              __AVG_POOLING_KERNEL_SIZE__>(reshape, Pipe_2, __AVG_POOLING_iMult_KERNEL__,
                                           __AVG_POOLING__nShift_KERNEL__, ap_accuracy_none());

  writeData<fused_pipes, dType_8u, __AVG_POOLING_INPUT_CHAN_SIZE__, 1>(out_0, Pipe_2);
}